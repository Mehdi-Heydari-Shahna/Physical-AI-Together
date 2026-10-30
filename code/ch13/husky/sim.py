"""
The Husky in MuJoCo, stepped with the A300's drive controller and the speed
loops of its motor drivers.

Physical AI, Together, Chapter 13 (Section 13.1).

Husky holds the compiled model, its data and a drive controller. Physics
steps every 0.5 ms. The controller updates at its rate, 50 Hz, every 40
steps, and sets the target speeds of the wheels, which the drivers hold
between updates. Each driver runs a speed loop at every step: proportional,
through MuJoCo's velocity actuator (torque = KV (ctrl - speed), integrated
implicitly by implicitfast), and integral, through an offset of ctrl that
integrates the speed error with the time constant TI. The offset is kept
where the loop's torque, KV times the speed error plus the offset, stays
within the torque limit (anti-windup): it stops growing at the limit and,
while the speed error alone asks for more than the limit, it unwinds. A
diagnostic may set the target speeds itself (wheel_targets), bypassing the
controller.

In the yard, the friction of each wheel's contact with the ground is set
before every step from the ground under the wheel's centre
(world.ground_at_xy), so that the grounds meet without seams. A weather,
if one is set (Section 13.2.5), gives that friction instead, from the
ground and the time since the robot was placed.

The robot is placed PLACE_DEPTH below touching. Placed so that its wheels
exactly touch the ground, MuJoCo 3.16.0 gives each wheel one contact whose
normal points from the wheel upwards, away from the ground: it carries no
load, and the robot falls freely for a step (Section 13.1.3).
"""
import math

import mujoco
import numpy as np

from . import drive, model, world


def yaw_of(quat):
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class Husky:
    def __init__(self, spec=None, ti=model.TI, integral=True, **build):
        self.spec = spec if spec is not None else model.build(**build)
        self.yard = self.spec.geom("ground").size[0] < 39.0      # the pad is 40 m
        self.m = self.spec.compile()
        self.d = mujoco.MjData(self.m)
        self.drive = drive.A300Drive()
        self.steps_per_update = int(round(self.drive.period / self.m.opt.timestep))
        self.base = self.m.body("base_link").id
        self.qadr = {w: self.m.jnt_qposadr[self.m.joint(f"{w}_wheel_joint").id]
                     for w in model.WHEELS}
        self.vadr = {w: self.m.jnt_dofadr[self.m.joint(f"{w}_wheel_joint").id]
                     for w in model.WHEELS}
        self.act = {w: self.m.actuator(f"{w}_servo").id for w in model.WHEELS}
        self.wheel_geom = {w: self.m.geom(model.wheel_geom_name(w)).id for w in model.WHEELS}
        self.wheel_body = {w: self.m.body(f"{w}_wheel_link").id for w in model.WHEELS}
        self.ground = self.m.geom("ground").id
        self.pair = {w: self.m.pair(f"ground | {model.wheel_geom_name(w)}").id
                     for w in model.WHEELS}
        self.kv = {w: float(self.m.actuator_gainprm[self.act[w]][0]) for w in model.WHEELS}
        self.limit = {w: float(self.m.actuator_forcerange[self.act[w]][1]) for w in model.WHEELS}
        self.ti, self.integral = ti, integral
        self.command = (0.0, 0.0)
        self.direct = None
        self.target = {w: 0.0 for w in model.WHEELS}
        self.z = {w: 0.0 for w in model.WHEELS}
        self.step_count = 0
        self.weather = None          # (ground, t) -> friction, or None: the grounds' own

    # --- placing and stepping -------------------------------------------------------
    def place(self, x=0.0, y=0.0, heading=0.0, speed=0.0, depth=model.PLACE_DEPTH):
        """Puts the robot on the ground at (x, y) with a heading, depth below
        touching, its wheels turning at the speed that rolls it forward at
        speed (no slip), and resets the drive controller and the drivers."""
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[0:3] = [x, y, model.rest_height(self.m) - depth]
        self.d.qpos[3:7] = [math.cos(heading / 2), 0.0, 0.0, math.sin(heading / 2)]
        self.d.qvel[0:3] = [speed * math.cos(heading), speed * math.sin(heading), 0.0]
        r = self.wheel_radius()
        for w in model.WHEELS:
            self.d.qvel[self.vadr[w]] = speed / r
            self.target[w] = speed / r
            self.d.ctrl[self.act[w]] = speed / r
            self.z[w] = 0.0
        self.set_friction()
        mujoco.mj_forward(self.m, self.d)
        self.drive.reset(*self.side_positions())
        if speed:
            self.drive.previous.clear()
            self.drive.previous.extend([(speed, 0.0), (speed, 0.0)])
        self.command = (speed, 0.0)
        self.direct = None
        self.step_count = 0

    def wheel_radius(self):
        return float(self.m.geom_size[self.wheel_geom["front_left"]][0])

    def side_positions(self):
        q = self.d.qpos
        left = 0.5 * sum(q[self.qadr[w]] for w in model.LEFT)
        right = 0.5 * sum(q[self.qadr[w]] for w in model.RIGHT)
        return left, right

    def wheel_targets(self, left, right):
        """Bypass the controller: hold these wheel speeds [rad/s]."""
        self.direct = (left, right)

    def set_friction(self, mu=None):
        """The friction of each wheel's pair with the ground: mu, or, in the
        yard, the ground under the wheel's centre."""
        for w in model.WHEELS:
            if mu is None:
                if not self.yard:
                    continue
                x, y = self.d.xpos[self.wheel_body[w]][:2]
                ground = world.ground_at_xy(x, y)
                f = ground.friction if self.weather is None else \
                    self.weather(ground, self.step_count * self.m.opt.timestep)
            else:
                f = mu
            self.m.pair_friction[self.pair[w]][0:2] = f

    def step(self, n=1):
        dt = self.m.opt.timestep
        for _ in range(n):
            if self.step_count % self.steps_per_update == 0:
                if self.direct is None:
                    left, right = self.drive.update(*self.command, *self.side_positions())
                else:
                    left, right = self.direct
                for w in model.LEFT:
                    self.target[w] = left
                for w in model.RIGHT:
                    self.target[w] = right
            for w in model.WHEELS:
                e = self.target[w] - self.d.qvel[self.vadr[w]]
                if self.integral:
                    z = self.z[w] + dt / self.ti * e
                    room = self.limit[w] / self.kv[w]
                    self.z[w] = min(max(z, -room - e), room - e)
                self.d.ctrl[self.act[w]] = self.target[w] + self.z[w]
            if self.yard:
                self.set_friction()
            mujoco.mj_step(self.m, self.d)
            self.step_count += 1

    def run(self, seconds):
        self.step(int(round(seconds / self.m.opt.timestep)))

    # --- what the simulator knows ---------------------------------------------------
    def pose(self):
        q = self.d.qpos
        return float(q[0]), float(q[1]), yaw_of(q[3:7])

    def body_velocity(self):
        """(forward, leftward, yaw rate) of base_link in its own frame."""
        heading = self.pose()[2]
        vx, vy = self.d.qvel[0], self.d.qvel[1]
        c, s = math.cos(heading), math.sin(heading)
        return float(c * vx + s * vy), float(-s * vx + c * vy), float(self.d.qvel[5])

    def wheel_speeds(self):
        return {w: float(self.d.qvel[self.vadr[w]]) for w in model.WHEELS}

    def rim_speeds(self):
        r = self.wheel_radius()
        return {w: r * v for w, v in self.wheel_speeds().items()}

    def torques(self):
        return {w: float(self.d.actuator_force[self.act[w]]) for w in model.WHEELS}

    def wheel_contacts(self):
        """Per wheel, the contacts with the ground: (position, force on the
        wheel, velocity of the wheel's material point there), in the world
        frame."""
        m, d = self.m, self.d
        out = {w: [] for w in model.WHEELS}
        geoms = {g: w for w, g in self.wheel_geom.items()}
        force = np.zeros(6)
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if self.ground not in (g1, g2):
                continue
            w = geoms.get(g1 if g2 == self.ground else g2)
            if w is None:
                continue
            mujoco.mj_contactForce(m, d, i, force)
            frame = np.array(c.frame).reshape(3, 3)
            f = frame.T @ force[:3]                 # force on geom2
            if g1 == self.wheel_geom[w]:
                f = -f
            mujoco.mj_jac(m, d, jacp, jacr, c.pos, self.wheel_body[w])
            out[w].append((np.array(c.pos), f, jacp @ d.qvel))
        return out

    def loads(self):
        """Normal force of the ground on each wheel [N]."""
        return {w: float(sum(f[2] for _, f, _ in cs)) for w, cs in self.wheel_contacts().items()}

    def total_mass(self):
        return float(self.m.body_mass.sum())
