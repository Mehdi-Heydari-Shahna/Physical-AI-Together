"""The Go2 of model.py on its stance controller (stance.py): the simulated
machine that the governed runs execute.

Machine.reset settles the robot at rest on its four feet: one second on
joint servos at the home angles of Menagerie's keyframe, then two seconds
on the stance controller, the base raised to BASE_HEIGHT and the centre of mass
held, with the integral correction, at 'home' (by default where the first
second left it). That
state is home: the feet's places, the centre of mass, its height. The run
continues the correction and the servos' targets from there.

Machine.advance runs one period of a law: at each step it asks the law for
the planned centre of mass (horizontal), its planned acceleration, the
share of its load that the front-right foot (FR) should carry, and FR's
height above its stance place; the controller holds the base level at
BASE_HEIGHT, its height above the floor, and moves the centre of mass slowly
back onto the plan (an integral correction, KI). Pushes on the trunk, the
world's disturbances, act at the base's centre of mass, horizontal.

The log, one row per step: the time, the measured and planned centre of
mass, the planned zero-moment point, the centre of pressure of the full
ground wrench (the contacts' torques included) and of the forces alone, the
cart-table point of the simulated centre of mass with the acceleration that
the ground's force gives it, each
foot's contact force, the ground force on other parts of the robot, the
joint torques that the controller asked for (before the motors' limits) and
those applied, FR's height and its target, and the base's roll, pitch and
height.
"""
import math

import mujoco
import numpy as np

from . import model
from .model import G, LEGS, Ids, ground_wrench
from .stance import Stance

TIMESTEP = 0.001         # [s] (Menagerie's go2.xml leaves MuJoCo's default, 0.002 s)
BASE_HEIGHT = 0.27       # [m] the base's height above the floor in stance (Menagerie's keyframe)
KI = 2.0                 # [1/s] the integral correction of the centre of mass
SETTLE = (1.0, 2.0)      # [s] the two settling phases
HOME_Q = (0.0, 0.9, -1.8)

KEYS = ("t", "cx", "cy", "cz", "px", "py", "zx", "zy", "copx", "copy", "fcopx", "fcopy", "ctx", "cty",
        "f0", "f1", "f2", "f3", "fx1", "fy1", "other", "frz", "frt", "roll", "pitch", "bz",
        "push") + tuple(f"tau{i}" for i in range(12)) + tuple(f"u{i}" for i in range(12))


def roll_pitch(q):
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    return roll, pitch


class Push:
    """A horizontal push on the trunk: magnitude 'force' [N] toward 'angle'
    [rad, from +x], rising and falling along half-cosines over 'rise' [s],
    at full force from t0 to t1 (the times of the run, from its start)."""

    def __init__(self, t0, t1, force, angle, rise=0.1):
        self.t0, self.t1, self.force, self.angle, self.rise = t0, t1, force, angle, rise

    def at(self, t):
        r = self.rise
        if t < self.t0 - r or t > self.t1 + r:
            return 0.0
        if t < self.t0:
            w = 0.5 * (1 - math.cos(math.pi * (t - self.t0 + r) / r))
        elif t > self.t1:
            w = 0.5 * (1 + math.cos(math.pi * (t - self.t1) / r))
        else:
            w = 1.0
        return self.force * w

    def vector(self, t):
        f = self.at(t)
        return np.array([f * math.cos(self.angle), f * math.sin(self.angle), 0.0])


class Machine:
    def __init__(self, timestep=TIMESTEP, pushes=(), kp=None, kd=None, home=None,
                 foot_solimp=None):
        self.timestep, self.pushes, self.home = timestep, list(pushes), home
        self.m = model.build(timestep, foot_solimp)
        self.d = mujoco.MjData(self.m)
        self.ids = Ids(self.m)
        kw = {}
        if kp is not None:
            kw["kp"] = kp
        if kd is not None:
            kw["kd"] = kd
        self.st = Stance(self.m, self.ids, **kw)
        self.reset()

    # --- settling ------------------------------------------------------------
    def reset(self):
        m, d, st = self.m, self.d, self.st
        mujoco.mj_resetData(m, d)
        d.qpos[2] = BASE_HEIGHT + 0.03
        d.qpos[3] = 1.0
        d.qpos[7:] = HOME_Q * 4
        mujoco.mj_forward(m, d)
        q0 = np.array(HOME_Q * 4)
        lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
        for _ in range(int(round(SETTLE[0] / self.timestep))):
            d.ctrl[:] = np.clip(60.0 * (q0 - d.qpos[7:]) - 2.0 * d.qvel[6:] + d.qfrc_bias[6:],
                                lo, hi)
            mujoco.mj_step(m, d)
        self.feet0 = np.array([d.geom_xpos[g].copy() for g in self.ids.feet])
        self.com_first = d.subtree_com[1].copy()
        home = self.com_first[:2].copy() if self.home is None else np.asarray(self.home, float)
        st.prev_qs = None
        self.corr = np.zeros(2)
        for _ in range(int(round(SETTLE[1] / self.timestep))):
            b = st.base_for_com(d, home + self.corr, BASE_HEIGHT, self.feet0)
            f = st.forces(d, d.subtree_com[1].copy(), (0.0, 0.0), (1, 1, 1, 1))
            d.ctrl[:], _, _ = st.torques(d, b, self.feet0, f, self.timestep)
            mujoco.mj_step(m, d)
            self.corr += KI * self.timestep * (home - d.subtree_com[1][:2])
        self.com0 = d.subtree_com[1].copy()
        self.t0 = d.time
        self.log = {k: [] for k in KEYS}
        self.pending = None
        return self

    # --- one period ----------------------------------------------------------
    def advance(self, duration, plan, power=None):
        """Run 'duration' seconds; plan(tau) -> (com_xy, acc_xy, fr_load, fr_height)
        or (com_xy, acc_xy, fr_load, fr_height, base_height) for tau in
        (0, duration]; power(tau), if given, scales the motors' torques (1: on,
        0: off)."""
        m, d, st, ids = self.m, self.d, self.st, self.ids
        n = int(round(duration / self.timestep))
        for i in range(n):
            tau = (i + 1) * self.timestep
            out = plan(tau)
            c_d, a_d, load, h = out[:4]
            zb = out[4] if len(out) > 4 else BASE_HEIGHT
            t = self.time() + self.timestep
            feet = self.feet0.copy()
            feet[1, 2] += h
            b = st.base_for_com(d, np.asarray(c_d) + self.corr, zb, feet)
            z_com = self.com0[2] + (zb - BASE_HEIGHT)
            f = st.forces(d, np.array([c_d[0], c_d[1], z_com]), a_d, (1.0, load, 1.0, 1.0))
            tau_c, raw, _ = st.torques(d, b, feet, f, self.timestep)
            if power is not None:
                w = power(tau)
                tau_c, raw = w * tau_c, w * raw
            d.ctrl[:] = tau_c
            push = np.zeros(3)
            for p in self.pushes:
                push += p.vector(t)
            d.xfrc_applied[ids.base, :3] = push
            mujoco.mj_step(m, d)
            com = d.subtree_com[1].copy()
            self.corr += KI * self.timestep * (np.asarray(c_d) - com[:2])
            F, other, cop, fcop = ground_wrench(m, d, ids)
            zmp = np.asarray(c_d) - (self.com0[2] / G) * np.asarray(a_d)
            ground = F.sum(axis=0) + other
            acc = ground / self.ids.mass - np.array([0.0, 0.0, G])      # the ground's share
            ct = com[:2] - com[2] / (G + acc[2]) * acc[:2]
            roll, pitch = roll_pitch(d.qpos[3:7])
            row = dict(t=t, cx=com[0], cy=com[1], cz=com[2], px=c_d[0], py=c_d[1], zx=zmp[0],
                       zy=zmp[1], copx=cop[0], copy=cop[1], fcopx=fcop[0], fcopy=fcop[1],
                       ctx=ct[0], cty=ct[1],
                       f0=F[0, 2], f1=F[1, 2], f2=F[2, 2], f3=F[3, 2], fx1=F[1, 0], fy1=F[1, 1],
                       other=float(np.linalg.norm(other)),
                       frz=d.geom_xpos[ids.feet[1]][2] - self.feet0[1, 2], frt=h,
                       roll=roll, pitch=pitch, bz=d.qpos[2], push=float(np.linalg.norm(push)))
            for j in range(12):
                row[f"tau{j}"] = raw[j]
                row[f"u{j}"] = d.ctrl[j]
            for k_, v_ in row.items():
                self.log[k_].append(float(v_))

    def time(self):
        return self.d.time - self.t0

    def steps(self):
        return len(self.log["t"])

    def com(self):
        return self.d.subtree_com[1].copy()

    def fr_height(self):
        return float(self.d.geom_xpos[self.ids.feet[1]][2] - self.feet0[1, 2])

    def arrays(self, since=0):
        return {k: np.array(v[since:]) for k, v in self.log.items()}


_STATIC = {}


def static_shares(s, three):
    """The joints' static torques with the centre of mass at rest at s on the
    shift line (reduced.py), on four feet or on three with FR held at the
    operation's height: the force distribution of stance.py with no
    acceleration, the legs' Jacobians and gravity, as the largest share of a
    motor's limit."""
    from . import reduced as rm
    if "machine" not in _STATIC:
        _STATIC["machine"] = Machine()
    mc = _STATIC["machine"]
    m, st = mc.m, mc.st
    d = mujoco.MjData(m)
    com = rm.COM0 + s * rm.U
    feet = mc.feet0.copy()
    if three:
        feet[1, 2] += rm.SINK + rm.LIFT
    b = st.base_for_com(d, com, BASE_HEIGHT, feet)
    d.qpos[:3] = b
    d.qpos[3:7] = [1, 0, 0, 0]
    from . import kinematics as kin
    for i, leg in enumerate(LEGS):
        d.qpos[7 + 3 * i:10 + 3 * i] = kin.ik(leg, feet[i] - b)
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    load = (1.0, 0.0 if three else 1.0, 1.0, 1.0)
    f = st.forces(d, d.subtree_com[1].copy(), (0.0, 0.0), load)
    tau = d.qfrc_bias[6:].copy()
    jac = np.zeros((3, m.nv))
    for i in range(4):
        if not np.any(f[i]):
            continue
        p = d.geom_xpos[mc.ids.feet[i]] - np.array([0, 0, mc.ids.radius])
        mujoco.mj_jac(m, d, jac, None, p, mc.ids.calf[i])
        tau -= jac[:, 6:].T @ f[i]
    limits = m.actuator_ctrlrange[:, 1]
    return float(np.max(np.abs(tau) / limits))
