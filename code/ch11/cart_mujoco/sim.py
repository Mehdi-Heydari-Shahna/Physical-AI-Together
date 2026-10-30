"""The simulated cart (Section 11.1.2): the MuJoCo model, the track
coordinate, placement at a state, and the drive loop that realizes a
commanded acceleration over one period of the governor.

MuJoCo computes the quantities that it derives from the state (positions of
bodies, velocities of frames, contacts, and the sensor values that do not
depend on accelerations) at the start of a step, before it advances the
state; between two calls of mj_step they are one time step old. The drive
loop therefore splits each step: mj_step1 computes them for the current
state, the loop reads the state and sets the controls, and mj_step2 finishes
the step. After the last step of a period, mj_forward brings them up to date
before the state is read.
"""
import math

import mujoco
import numpy as np

from .scene import Design, mjcf

G = 9.81               # gravity of MuJoCo's default option [m/s^2]
V_REST = 5e-3          # below 5 mm/s the drive loop counts the cart as at rest [m/s]


class Sim:
    """The cart on its track in MuJoCo. variant is one of the variants of
    scene.mjcf; read="after step" and coordinate="ground" are mistakes of
    Section 11.1.3: reading the derived quantities after mj_step, and
    following the ground coordinate."""

    def __init__(self, design=Design(), variant=None, read="current", coordinate="axle"):
        self.design, self.variant = design, variant
        self.pr = design.profile()
        self.xml = mjcf(design, variant)
        self.m = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.m)
        self.read, self.coordinate = read, coordinate
        m = self.m
        names = ("x", "z", "pitch", "front", "rear", "reach")
        self.body = {n: m.body(n).id for n in ("chassis", "front", "rear", "tool")}
        self.jq = {n: int(m.jnt_qposadr[m.joint(n).id]) for n in names}
        self.jv = {n: int(m.jnt_dofadr[m.joint(n).id]) for n in names}
        self.servos = variant == "speed servos"
        acts = ("drive front", "drive rear", "tool") + (
            ("speed front", "speed rear") if self.servos else ())
        self.act = {n: m.actuator(n).id for n in acts}
        self.mass = float(sum(m.body_mass))
        self.wheel_inertia = float(m.body_inertia[self.body["front"]][1])
        self.lead = design.half_length - design.wheel_offset    # bumper ahead of the axle
        self.steps = int(round(0.1 / m.opt.timestep))           # steps per period
        self.sink = 0.0
        self.sink = self.static_sink()

    # --- the state of the cart ------------------------------------------------------
    def foot(self, wheel="front"):
        x, _, z = self.data.xpos[self.body[wheel]]
        return self.pr.foot(x, z)

    def _front(self):
        """The foot of the front axle and the axle's velocity (vx, vz)."""
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                 self.body["front"], vel, 0)
        return self.foot(), vel[3], vel[5]

    def state(self, coordinate=None):
        """(p, v, l): the track coordinate of the bumper, its rate, and the
        reach of the tool, from quantities that must be current."""
        f, vx, vz = self._front()
        if (coordinate or self.coordinate) == "ground":
            p, v = f.s + self.lead, self.pr.ground_rate(f, vx, vz)
        else:
            p, v = self.pr.axle(f.s) + self.lead, self.pr.axle_rate(f, vx, vz)
        return p, v, float(self.data.qpos[self.jq["reach"]])

    def axle_speed(self):
        """The speed of the front axle along the ground's tangent at its
        foot. On the crest, the rate of p exceeds it by the factor
        (R + r) / (R + d), where d, the axle's height above the ground, is
        r less the penetration."""
        f, vx, vz = self._front()
        return vx * math.cos(f.angle) - vz * math.sin(f.angle)

    # --- placement -----------------------------------------------------------------------
    def _axle(self, s):
        x, z, _ = self.pr.point(s)
        nx, nz = self.pr.normal(s)
        h = self.design.wheel_radius - self.sink
        return x + h * nx, z + h * nz

    def _pose(self, p):
        """Chassis position (x, z) and pitch with the bumper at track
        coordinate p and both wheels on the ground."""
        sf = self.pr.ground(p - self.lead)
        F = self._axle(sf)
        base = 2 * self.design.wheel_offset
        lo, hi = sf - base - 0.2, sf
        for _ in range(80):                        # the rear axle, at distance base
            mid = (lo + hi) / 2
            R = self._axle(mid)
            if math.hypot(F[0] - R[0], F[1] - R[1]) > base:
                lo = mid
            else:
                hi = mid
        R = self._axle((lo + hi) / 2)
        pitch = math.atan2(-(F[1] - R[1]), F[0] - R[0])
        wd = self.design.wheel_drop
        return ((F[0] + R[0]) / 2 + wd * math.sin(pitch),
                (F[1] + R[1]) / 2 + wd * math.cos(pitch), pitch)

    def place(self, p, v, l):
        """Put the cart at the state (p, v, l), rolling without slip at
        speed v, its wheels sunk by the static penetration, the tool still."""
        d, q, w = self.data, self.jq, self.jv
        mujoco.mj_resetData(self.m, d)
        cx, cz, pitch = self._pose(p)
        eps = 1e-6
        a, b = self._pose(p + eps), self._pose(p - eps)
        hl, zc = self.design.half_length, self.design.wheel_radius + self.design.wheel_drop
        d.qpos[q["x"]], d.qpos[q["z"]], d.qpos[q["pitch"]] = cx + hl, cz - zc, pitch
        d.qpos[q["reach"]] = l
        for name, i in (("x", 0), ("z", 1), ("pitch", 2)):
            d.qvel[w[name]] = v * (a[i] - b[i]) / (2 * eps)
        rate = v / self.design.wheel_radius - d.qvel[w["pitch"]]   # relative to the chassis
        d.qvel[w["front"]] = d.qvel[w["rear"]] = rate
        d.ctrl[self.act["tool"]] = l
        mujoco.mj_forward(self.m, d)

    def static_sink(self):
        """Penetration of the wheels at rest on the firm ground: MuJoCo's
        contacts are soft."""
        self.place(5.0, 0.0, 0.0)
        for _ in range(30):
            self.period(0.0, 0.0)
        return -float(min(c.dist for c in self.data.contact[:self.data.ncon]))

    # --- the drive loop ------------------------------------------------------------------
    def controls(self, a_ref, v_ref, ell):
        """The drive loop for the current state. The wheels' motors get a
        force M (a_ref - g sin(pitch)) for the acceleration a_ref and for
        gravity along the chassis, and a feedback force M K (v_ref - v) on
        the cart's speed v, with the speed gain K, both shared between the
        wheels in proportion to their measured loads, and each wheel what its
        own inertia needs. The wheel loads come from touch sensors, which
        MuJoCo computes from the contact forces later in the step: they are
        those of the previous step. The tool servo is set to the reach ell.
        In the variant "speed servos", MuJoCo's velocity actuators provide
        the feedback instead: each is set to the wheel rate that moves the
        cart's speed to v_ref."""
        d, r = self.data, self.design.wheel_radius
        pitch = d.qpos[self.jq["pitch"]]
        v = self.state()[1]
        gain = 0.0 if self.servos else self.design.speed_gain
        force = self.mass * (a_ref - G * math.sin(pitch) + gain * (v_ref - v))
        loads = [float(d.sensor(w + " load").data[0]) for w in ("front", "rear")]
        lim = self.design.drive_limit
        for wheel, load in zip(("front", "rear"), loads):
            share = load / sum(loads) if sum(loads) > 0 else 0.5
            tau = force * r * share + self.wheel_inertia * a_ref / r
            d.ctrl[self.act["drive " + wheel]] = max(-lim, min(lim, tau))
            if self.servos:
                d.ctrl[self.act["speed " + wheel]] = (d.qvel[self.jv[wheel]] + a_ref *
                                                      self.m.opt.timestep / r + (v_ref - v) / r)
        d.ctrl[self.act["tool"]] = ell

    def period(self, u, ell, kind="segment", record=None,
               steps=None):
        """One period of 0.1 s under the commanded acceleration u
        and the commanded reach ell, held over the period; or the
        first 'steps' steps of one."""
        m, d = self.m, self.data
        v0 = self.state()[1]
        if abs(v0) < V_REST:
            v0 = 0.0                     # at rest: start from zero
        if kind == "rest":               # brake, then hold
            brake = 1 if v0 > 0 else 0
        else:
            brake = 1 if v0 > 0 > u else -1 if v0 < 0 < u else 0
        for i in range(steps or self.steps):
            tau = i * m.opt.timestep
            v_ref, a_ref = v0 + u * tau, u
            if (kind == "rest" and v0 <= 0) or (
                    brake and v_ref * v0 <= 0):
                v_ref, a_ref = 0.0, 0.0  # at rest: hold
            if self.read == "after step":
                self.controls(a_ref, v_ref, ell)
                mujoco.mj_step(m, d)     # the mistake
                continue
            mujoco.mj_step1(m, d)        # derived quantities now
            if record is not None:
                record.append(self.sample(a_ref, v_ref))
            self.controls(a_ref, v_ref, ell)
            mujoco.mj_step2(m, d)        # controls, then advance
        if self.read != "after step":
            mujoco.mj_forward(m, d)
        return self.state()

    def sample(self, a_ref, v_ref):
        """The signals at the current step, for the records."""
        d = self.data
        p, v, l = self.state()
        return dict(t=d.time, p=p, v=v, l=l, a_ref=a_ref, v_ref=v_ref, axle=self.axle_speed(),
                    front_rate=float(d.qvel[self.jv["front"]]),
                    rear_rate=float(d.qvel[self.jv["rear"]]),
                    pitch=float(d.qpos[self.jq["pitch"]]), contacts=int(d.ncon),
                    range=float(d.sensor("range").data[0]),
                    front_load=float(d.sensor("front load").data[0]),
                    rear_load=float(d.sensor("rear load").data[0]))
