"""
The simulated machine of Section 14.3, one period at a time.

Physical AI, Together, Chapter 14 (Section 14.3).

The jointed machine of model.build stands on its four wheel contacts in
MuJoCo and holds or moves its arm on position servos, as in
model.simulated_lift, and the base's motion is that of Section 14.2: the
brakes, the drive and the steering realize the commanded motion of the
base's middle along the route of route.py, and the simulation holds the
machine in the base's own frame, which turns with the route, every body
feeling that frame's forces and torques at its centre of mass
(model.simulated_lift's frame_forces) and gravity tilted by the acceleration
of the base's middle. What the simulation adds is the arm's motion on its
servos and the contacts' compliance: the machine creeps on its contacts, or
slides if the ground cannot hold it, and its wheels can leave the ground,
none of which the commanded motion knows. Each wheel stands on the ground
under it (route.wheel_frictions): the ground plane's own friction is set to
0, and MuJoCo gives a contact the larger of its two geoms' frictions, so the
wheel's friction is the ground's.

The servos' targets are offset by the torque that holds the planned pose
under gravity alone for the payload that the controller believes the arm
holds (believed), which need not be the payload that it does hold.
"""
import math

import mujoco
import numpy as np

from . import labels as hl
from . import model, route

FIELDS = ("t", "s", "v", "a", "ay", "w", "alpha", "cop_x", "cop_y", "base_x", "base_y",
          "yaw", "pitch", "wheels", "q0", "q1", "q2", "p0", "p1", "p2", "tau0", "tau1", "tau2",
          "mu_f", "mu_r")


def _angles(quat):
    qw, qx, qy, qz = quat
    yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx))))
    return yaw, pitch


class Machine:
    """The simulated machine with a payload [kg] in the pose start, its
    middle at the arc length s0 of the route; the controller's servos offset
    for the payload believed (by default the payload held). It settles for
    settle seconds at rest or, with motion0 = (v, a), in the commanded state
    of moving at v and accelerating at a at s0, which the run then starts
    from. friction, if given, replaces the route's grounds."""

    def __init__(self, payload, start, s0=0.0, believed=None, kp=2e6, impratio=model.IMPRATIO,
                 friction=None, settle=2.0, motion0=None):
        self.payload = float(payload)
        self.believed = self.payload if believed is None else float(believed)
        self.kp, self.friction = kp, friction
        self.m = model.build(payload=self.payload, pose=None, friction=route.FIRM, kp=kp,
                             impratio=impratio).compile()
        self.m.geom_friction[self.m.geom("ground").id][0] = 0.0
        self.front_wheels = [self.m.geom(f"wheel_f{x}").id for x in "lr"]
        self.rear_wheels = [self.m.geom(f"wheel_r{x}").id for x in "lr"]
        self.wheel_ids = set(self.front_wheels + self.rear_wheels)
        self.d = mujoco.MjData(self.m)
        self.mf = model.build(payload=self.believed, pose=None, fixed_base=True).compile()
        self.mf.opt.gravity[:] = [0.0, 0.0, -hl.G]
        self.df = mujoco.MjData(self.mf)
        self.base = self.m.body("base").id
        self.jacp = np.zeros((3, self.m.nv))
        self.dt = self.m.opt.timestep
        self.t, self.s = 0.0, float(s0)
        self._cache, self._turning = {}, True
        self.log = {f: [] for f in FIELDS}
        q = np.array(start, dtype=float)
        self.d.qpos[7:10] = q
        self.d.ctrl[:] = self.target(q)
        self._frictions(self.s)
        v0, a0 = (0.0, 0.0) if motion0 is None else (float(motion0[0]), float(motion0[1]))
        mujoco.mj_forward(self.m, self.d)
        for _ in range(int(round(settle / self.dt))):
            self._motion(self.s, v0, a0)
            mujoco.mj_step(self.m, self.d)
        self.origin = self.d.qpos[:2].copy()
        self.yaw0 = _angles(self.d.qpos[3:7])[0]

    # --- the servos, the grounds and the frame ---------------------------------
    def target(self, q):
        """The servos' target for the planned pose q: q offset by the torque
        that holds it under gravity alone for the believed payload, divided
        by the servos' stiffness."""
        key = tuple(np.round(q, 12))
        if key not in self._cache:
            self.df.qpos[:3] = q
            mujoco.mj_forward(self.mf, self.df)
            if len(self._cache) > 20000:
                self._cache.clear()
            self._cache[key] = np.array(q) + self.df.qfrc_bias[:3] / self.kp
        return self._cache[key]

    def _frictions(self, s):
        if self.friction is not None:
            mu_f = mu_r = self.friction
        else:
            mu_f, mu_r = route.wheel_frictions(s)
        for g in self.front_wheels:
            self.m.geom_friction[g][0] = mu_f
        for g in self.rear_wheels:
            self.m.geom_friction[g][0] = mu_r
        return mu_f, mu_r

    def _motion(self, s, v, a):
        """Gravity tilted by the acceleration of the base's middle, and the
        turning frame's forces; returns (ay, w, alpha)."""
        k, dk = route.curvature_at(s)
        ay, w, alpha = v * v * k, v * k, a * k + v * v * dk
        self.m.opt.gravity[:] = [-a, -ay, -hl.G]
        if w != 0.0 or alpha != 0.0:
            self._frame_forces(w, alpha)
            self._turning = True
        elif self._turning:
            self.d.xfrc_applied[:] = 0.0
            self._turning = False
        return ay, w, alpha

    def _frame_forces(self, w, alpha):
        m, d = self.m, self.d
        o = d.xpos[self.base]
        for b in range(1, m.nbody):
            r = d.xipos[b] - o
            mujoco.mj_jacBodyCom(m, d, self.jacp, None, b)
            u = self.jacp @ d.qvel
            mass = m.body_mass[b]
            d.xfrc_applied[b, 0] = mass * (alpha * r[1] + w * w * r[0] + 2.0 * w * u[1])
            d.xfrc_applied[b, 1] = mass * (-alpha * r[0] + w * w * r[1] - 2.0 * w * u[0])
            d.xfrc_applied[b, 2] = 0.0
            rot = d.ximat[b].reshape(3, 3)
            iz = rot @ (m.body_inertia[b] * rot[2, :])
            d.xfrc_applied[b, 3] = -(alpha * iz[0] - w * w * iz[1])
            d.xfrc_applied[b, 4] = -(alpha * iz[1] + w * w * iz[0])
            d.xfrc_applied[b, 5] = -alpha * iz[2]

    def wheels_touching(self):
        d = self.d
        return len({g for c in d.contact[:d.ncon] for g in (int(c.geom1), int(c.geom2))
                    if g in self.wheel_ids})

    # --- one period -----------------------------------------------------------
    def advance(self, duration, base, arm):
        """Runs the machine for duration seconds: base(tau) gives the
        commanded motion of the base's middle (s, v, a) at the time tau
        within the run, which the brakes, the drive and the steering realize;
        arm(tau) the planned pose, which the servos follow. Logs every step
        (FIELDS) and returns the number of steps."""
        m, d, log = self.m, self.d, self.log
        n = int(round(duration / self.dt))
        for i in range(n):
            tau = (i + 1) * self.dt
            s, v, a = base(tau)
            ay, w, alpha = self._motion(s, v, a)
            mu_f, mu_r = self._frictions(s)
            q = np.asarray(arm(tau), dtype=float)
            d.ctrl[:] = self.target(q)
            mujoco.mj_step(m, d)
            p, _ = model.centre_of_pressure(m, d)
            yaw, pitch = _angles(d.qpos[3:7])
            row = (self.t + tau, s, v, a, ay, w, alpha, p[0], p[1],
                   d.qpos[0] - self.origin[0], d.qpos[1] - self.origin[1], yaw - self.yaw0,
                   pitch, self.wheels_touching(), d.qpos[7], d.qpos[8], d.qpos[9],
                   q[0], q[1], q[2], d.actuator_force[0], d.actuator_force[1],
                   d.actuator_force[2], mu_f, mu_r)
            for f, x in zip(FIELDS, row):
                log[f].append(x)
        self.t += n * self.dt
        if n:
            self.s = log["s"][-1]
        return n

    def joints(self):
        return self.d.qpos[7:10].copy()

    def creep(self):
        """The base's travel on its contacts since it settled, along and
        across its heading [m], and its turn on them [rad]."""
        return (float(self.d.qpos[0] - self.origin[0]), float(self.d.qpos[1] - self.origin[1]),
                _angles(self.d.qpos[3:7])[0] - self.yaw0)

    def arrays(self, since=0):
        """The log from step 'since' on, as arrays."""
        return {f: np.asarray(v[since:], dtype=float) for f, v in self.log.items()}

    def steps(self):
        return len(self.log["t"])


def margins(log, payload):
    """For a log of Machine.arrays: the simulated centre of pressure's
    distance inside the rectangle of the wheels' contacts and the rigid
    model's for the planned pose under the same motion (labels.motion_margin)
    [m], step by step; the simulated one is 0 when the centre of pressure
    lies on an edge, the wheels away from that edge off the ground."""
    sim = np.minimum(model.AXLE - np.abs(log["cop_x"]), model.HALF_TRACK - np.abs(log["cop_y"]))
    rigid = np.empty(len(sim))
    planned = np.stack([log["p0"], log["p1"], log["p2"]], axis=1)
    keys, inverse = np.unique(np.round(planned, 9), axis=0, return_inverse=True)
    inverse = np.ravel(inverse)
    for j, q in enumerate(keys):
        idx = np.nonzero(inverse == j)[0]
        rigid[idx] = hl.motion_margin(tuple(q), payload, log["a"][idx], log["ay"][idx],
                                      log["w"][idx], log["alpha"][idx])
    return sim, rigid
