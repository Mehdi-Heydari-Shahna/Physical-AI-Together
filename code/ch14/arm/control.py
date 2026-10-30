"""
The UR5e's controller and its stop (ours).

Physical AI, Together, Chapter 14 (Section 14.1).

UR's own controller is not public; the controller here is ours. It runs at
RATE Hz in joint space: computed torque, tau = M(q) (qdd_c + KP e + KD de) +
bias(q, qd), with e and de the errors of position and speed against the
commanded motion, each joint's torque then clipped to its limit; between two
updates the torques are held. The gains (KP 900 1/s^2, KD 60 1/s, a double
pole at 30 rad/s) and the rate are ours.

The stop of the swing: from the base joint's speed w0 (negative, towards the
post), the commanded base angle slows at the constant deceleration b to rest,
while the other joints hold the carry pose. A label of the base joint is
such a b; it is valid for a payload when the commanded stop needs no torque
beyond any joint's limit, so that the controller can execute it.

Because the base's axis is vertical, the mass matrix M and gravity g do not
depend on the base angle, and while only the base joint moves the bias of
joint j is g_j + C_j w^2 (centrifugal). The torque that the commanded stop
needs is then tau_j = M_j0 b + C_j w^2 + g_j, with w from w0 to 0, so its
extremes lie at the start (w = w0) and at the end (w = 0) of the stop, and
the largest valid b follows from twelve linear bounds (capability()).
"""
import math

import mujoco
import numpy as np

from . import model, world

RATE = 1000.0      # [Hz] the controller's updates (ours)
KP = 900.0         # [1/s^2]
KD = 60.0          # [1/s]


class Arm:
    """The UR5e in the swing's world, with a payload of the given mass, and
    the controller above. step() advances one update of the controller."""

    def __init__(self, payload=0.0, timestep=model.TIMESTEP, armature=0.0, post=True,
                 payload_offset=model.PAYLOAD_OFFSET, box_offset=None):
        spec = model.build(payload=payload, payload_offset=payload_offset, timestep=timestep,
                           armature=armature, world=world.add_world if post else None,
                           box_offset=box_offset)
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.has_post = post
        self.substeps = max(1, int(round(1.0 / (RATE * timestep))))
        self.limit = np.array([model.limits()[j]["torque"] for j in model.JOINTS])
        self.mass = np.zeros((self.m.nv, self.m.nv))
        self.pose, self.pose_err = world.carry_pose(self.m, mujoco.MjData(self.m))
        self.tool = self.m.site("tool0").id
        self.saturated = np.zeros(self.m.nu, dtype=int)

    def place(self, q, qd):
        self.d.qpos[:] = q
        self.d.qvel[:] = qd
        self.d.time = 0.0
        mujoco.mj_forward(self.m, self.d)

    def torque(self, qc, qdc, qddc):
        """The controller's torque for the commanded motion, before and after
        the clip, from the present state."""
        m, d = self.m, self.d
        mujoco.mj_forward(m, d)
        mujoco.mj_fullM(m, d, self.mass)
        acc = qddc + KP * (qc - d.qpos) + KD * (qdc - d.qvel)
        raw = self.mass @ acc + d.qfrc_bias
        return raw, np.clip(raw, -self.limit, self.limit)

    def step(self, qc, qdc, qddc):
        raw, tau = self.torque(qc, qdc, qddc)
        self.saturated += (np.abs(raw) > self.limit).astype(int)
        self.d.ctrl[:] = tau
        for _ in range(self.substeps):
            mujoco.mj_step(self.m, self.d)
        return raw, tau


# ----------------------------------------------------------------------------
# The capability: the largest valid label of the base joint
# ----------------------------------------------------------------------------
def coefficients(arm):
    """M_j0, C_j (the centrifugal bias per w^2) and g_j at the carry pose."""
    m, d = arm.m, mujoco.MjData(arm.m)
    d.qpos[:] = arm.pose
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    mass = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, mass)
    g = d.qfrc_bias.copy()
    d.qvel[0] = 1.0
    mujoco.mj_forward(m, d)
    c = d.qfrc_bias.copy() - g
    return mass[:, 0].copy(), c, g


def bounds(arm, w0, share=1.0):
    """Each joint's own bound on the deceleration b of the base joint's
    commanded stop from w0: the largest b for which that joint's torque stays
    within share of its limit at the start and at the end of the stop, and
    which of the two binds ((inf, None) for a joint that b does not load;
    (0, end) for one that is over its limit without any b)."""
    mj0, c, g = coefficients(arm)
    out = []
    for j in range(len(mj0)):
        best, which = math.inf, None
        limit = share * arm.limit[j]
        for end, w2 in (("start", w0 * w0), ("end", 0.0)):
            rest = c[j] * w2 + g[j]
            if abs(rest) > limit:
                best, which = 0.0, end
                break
            if mj0[j] > 1e-12:
                bound = (limit - rest) / mj0[j]
            elif mj0[j] < -1e-12:
                bound = (-limit - rest) / mj0[j]
            else:
                continue
            if bound < best:
                best, which = bound, end
        out.append((best, which))
    return out


def capability(arm, w0, share=1.0):
    """The largest deceleration b of the base joint whose commanded stop
    from w0 needs no torque beyond share of a joint's limit, with the joint
    that binds and whether the bound comes from the start or the end of the
    stop. Also the torques that cruising at w0 needs."""
    mj0, c, g = coefficients(arm)
    per_joint = bounds(arm, w0, share)
    j = min(range(len(per_joint)), key=lambda k: per_joint[k][0])
    best, end = per_joint[j]
    return best, (model.JOINTS[j], end), c * w0 * w0 + g


# ----------------------------------------------------------------------------
# The stop, simulated
# ----------------------------------------------------------------------------
def stop(arm, b, w0, start=None, settle=0.5, record=None):
    """The base joint swinging at w0 < 0 from the task's angle start slows at
    the commanded b to rest; the run continues settle seconds after the
    commanded rest. Returns the facts of the stop, the angles those of the
    task."""
    a0 = world.PLACE + 0.5 if start is None else start       # an angle of the task
    q0 = world.turned(arm.pose, a0)
    offset = arm.pose[0]          # the base joint's angle less the task's
    qd0 = np.zeros(arm.m.nv)
    qd0[0] = w0
    arm.place(q0, qd0)
    arm.saturated[:] = 0
    t_stop = -w0 / b
    rest = a0 - w0 * w0 / (2 * b)
    dt = 1.0 / RATE
    steps = int(math.ceil((t_stop + settle) * RATE))
    worst_err = np.zeros(arm.m.nv)
    peak = np.zeros(arm.m.nu)
    path, last = 0.0, arm.d.site_xpos[arm.tool].copy()
    farthest = a0
    least_clearance = world.clearance(arm.m, arm.d) if arm.has_post else math.inf

    def commanded(t):
        qc, qdc, qddc = q0.copy(), np.zeros(arm.m.nv), np.zeros(arm.m.nv)
        if t < t_stop:
            qc[0] = offset + a0 + w0 * t + 0.5 * b * t * t
            qdc[0] = w0 + b * t
            qddc[0] = b
        else:
            qc[0] = offset + rest
        return qc, qdc, qddc

    departure = np.zeros(arm.m.nv)        # each joint's largest distance from q0 (base: 0)
    top_speed = np.zeros(arm.m.nv)        # each joint's largest speed after the start
    t_farthest, t_clipped = 0.0, None     # when the farthest angle is reached; the last
    for k in range(steps):                # update whose base torque was clipped
        t = k * dt
        raw, tau = arm.step(*commanded(t))
        after = commanded(t + dt)[0]          # the commanded motion when the update ends
        peak = np.maximum(peak, np.abs(raw) / arm.limit)
        if abs(raw[0]) > arm.limit[0]:
            t_clipped = t + dt
        worst_err = np.maximum(worst_err, np.abs(arm.d.qpos - after))
        moved = np.abs(arm.d.qpos - q0)
        moved[0] = 0.0
        departure = np.maximum(departure, moved)
        top_speed = np.maximum(top_speed, np.abs(arm.d.qvel))
        if float(arm.d.qpos[0]) - offset < farthest:
            farthest, t_farthest = float(arm.d.qpos[0]) - offset, t + dt
        if arm.has_post:
            least_clearance = min(least_clearance, world.clearance(arm.m, arm.d))
        p = arm.d.site_xpos[arm.tool].copy()
        path += float(np.linalg.norm(p - last))
        last = p
        if record is not None:
            record.append((t + dt, after[0] - offset, arm.d.qpos[0] - offset, arm.d.qvel[0], *tau))
    angle = float(arm.d.qpos[0]) - offset
    return dict(start=a0, commanded_rest=rest, rest=angle, farthest=farthest,
                beyond=rest - farthest, commanded_travel=a0 - rest, travel=a0 - farthest,
                time=t_stop, peak=peak, saturated=arm.saturated.copy(), err=worst_err,
                path=path, speed=float(arm.d.qvel[0]), clearance=least_clearance,
                departure=departure, top_speed=top_speed, t_farthest=t_farthest,
                t_clipped=t_clipped)


def held_stop(arm, b, w0, start=None, settle=0.5, at=None):
    """The same commanded stop with the other joints held exactly in the
    carry pose: the base alone, of the constant inertia M00, under the same
    controller (its computed torque clipped at the limit, held between
    updates), integrated exactly between the updates. Returns the farthest
    angle, the overrun of the commanded rest, the least clearance of the
    held pose along the stop, and the base's speed at the time at [s]."""
    a0 = world.PLACE + 0.5 if start is None else start
    mj0, c, g = coefficients(arm)
    inertia, limit = mj0[0], arm.limit[0]
    t_stop = -w0 / b
    rest = a0 - w0 * w0 / (2 * b)
    dt = 1.0 / RATE
    steps = int(math.ceil((t_stop + settle) * RATE))
    d = mujoco.MjData(arm.m)
    angle, speed = a0, w0
    farthest, speed_at = a0, None
    least = math.inf
    for k in range(steps):
        t = k * dt
        if t < t_stop:
            qc, qdc, qddc = a0 + w0 * t + 0.5 * b * t * t, w0 + b * t, b
        else:
            qc, qdc, qddc = rest, 0.0, 0.0
        bias = c[0] * speed * speed + g[0]
        raw = inertia * (qddc + KP * (qc - angle) + KD * (qdc - speed)) + bias
        tau = min(limit, max(-limit, raw))
        acc = (tau - bias) / inertia            # c[0] = g[0] = 0 for the vertical axis
        angle += speed * dt + 0.5 * acc * dt * dt
        speed += acc * dt
        farthest = min(farthest, angle)
        if at is not None and speed_at is None and t + dt >= at - 1e-12:
            speed_at = speed
        if arm.has_post:
            d.qpos[:] = world.turned(arm.pose, angle)
            mujoco.mj_kinematics(arm.m, d)
            least = min(least, world.clearance(arm.m, d))
    return dict(farthest=farthest, beyond=rest - farthest, clearance=least, speed_at=speed_at)
