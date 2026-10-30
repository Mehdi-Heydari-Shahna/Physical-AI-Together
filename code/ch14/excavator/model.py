"""
The excavator in MuJoCo: the documented model of front.py and mass.py.

Physical AI, Together, Chapter 14 (Section 14.1).

Bodies: the undercarriage on a free joint; the upper structure on the swing
joint, with the counterweight fixed to it; the boom, the arm and the bucket
on their pins, each turning about the axis (0, -1, 0) so that a positive
angle lifts it (theta1, theta2 and theta3 of front.py); the load, a point
mass at the bucket pin, fixed to the arm. Each part's mass and centre of mass
are mass.py's, its inertia that of the box mass.py gives.

Contacts: the undercarriage touches the ground through four small spheres at
the ends of the two tipping lines of ISO 10567 as capacity.py reads them,
(+-1.44 m, +-0.995 m), so that the simulated machine tips about the same
lines as the static model. The tracks' boxes are drawn but do not touch.

With pose=(theta1, theta2, theta3) and slew, the front is built fixed in that
pose and the upper structure turned (no joints but the free one), for the
static checks; without, the four joints have torque actuators, unlimited,
which Section 14.1 does not use.
"""
import math

import mujoco
import numpy as np

from . import capacity, front, mass
from . import published as pub

TIMESTEP = 0.001
SPHERE = 0.02                 # [m] the contact spheres' radius (ours)


def _quat_y(angle):
    """A rotation by angle about (0, -1, 0)."""
    return [math.cos(angle / 2), 0.0, -math.sin(angle / 2), 0.0]


def _inertial(body, m, com, box):
    body.explicitinertial = True
    body.mass = m
    body.ipos = list(com)
    body.iquat = [1, 0, 0, 0]
    body.inertia = list(mass.box_inertia(m, box))


def build(load=0.0, pose=None, slew=0.0, timestep=TIMESTEP):
    """The MjSpec of the PC138US-8 with a load [kg] at the bucket pin."""
    spec = mujoco.MjSpec()
    spec.modelname = "pc138us8"
    spec.compiler.degree = False
    spec.option.timestep = timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.worldbody.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[20, 20, 0.1],
                            friction=[0.8, 0.005, 0.0001], rgba=[0.55, 0.5, 0.42, 1])
    lower = spec.worldbody.add_body(name="lower", pos=[0, 0, SPHERE])
    lower.add_freejoint(name="ground_joint")
    com = list(mass.LOWER["com"])
    com[2] -= SPHERE
    _inertial(lower, mass.LOWER["mass"], com, mass.LOWER["box"])
    for sx in (-1, 1):
        for sy in (-1, 1):
            lower.add_geom(name=f"support_{'f' if sx > 0 else 'r'}{'l' if sy > 0 else 'r'}",
                           type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[SPHERE, 0, 0],
                           pos=[sx * capacity.FRONT_LINE, sy * capacity.SIDE_LINE, 0.0],
                           friction=[0.8, 0.005, 0.0001], density=0)
    for sy in (-1, 1):
        lower.add_geom(name=f"track_{'l' if sy > 0 else 'r'}", type=mujoco.mjtGeom.mjGEOM_BOX,
                       size=[pub.TRACK_LENGTH / 2, pub.SHOE_WIDTH / 2, 0.35],
                       pos=[0, sy * capacity.SIDE_LINE, 0.35], contype=0, conaffinity=0,
                       density=0, rgba=[0.15, 0.15, 0.15, 1])
    upper = lower.add_body(name="upper", pos=[0, 0, mass.SWING_HEIGHT - SPHERE],
                           quat=[math.cos(slew / 2), 0, 0, math.sin(slew / 2)])
    if pose is None:
        upper.add_joint(name="swing", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1])
    uc = list(mass.UPPER["com"])
    uc[2] -= mass.SWING_HEIGHT
    _inertial(upper, mass.UPPER["mass"], uc, mass.UPPER["box"])
    upper.add_geom(name="upper_box", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[s / 2 for s in mass.UPPER["box"]], pos=[-0.3, 0, 0.6], contype=0,
                   conaffinity=0, density=0, rgba=[0.9, 0.75, 0.1, 1])
    cwc = list(mass.COUNTERWEIGHT["com"])
    cwc[2] -= mass.SWING_HEIGHT
    cw = upper.add_body(name="counterweight", pos=cwc)
    _inertial(cw, mass.COUNTERWEIGHT["mass"], [0, 0, 0], mass.COUNTERWEIGHT["box"])
    cw.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[s / 2 for s in mass.COUNTERWEIGHT["box"]],
                contype=0, conaffinity=0, density=0, rgba=[0.3, 0.3, 0.3, 1])
    t1, t2, t3 = (0.0, 0.0, 0.0) if pose is None else pose
    links = []
    parent, pos = upper, [front.X0, 0.0, front.Z0 - mass.SWING_HEIGHT]
    for name, part, length, angle in (("boom", mass.BOOM, pub.BOOM_LENGTH, t1),
                                      ("arm", mass.ARM, pub.ARM_LENGTH, t2),
                                      ("bucket", mass.BUCKET, front.R_BUCKET, t3)):
        body = parent.add_body(name=name, pos=pos, quat=_quat_y(angle))
        if pose is None:
            body.add_joint(name=name, type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, -1, 0])
        _inertial(body, part["mass"], [part["along"], 0.0, part["across"]], part["box"])
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, 0, length, 0, 0],
                      size=[0.12, 0, 0], contype=0, conaffinity=0, density=0,
                      rgba=[0.9, 0.75, 0.1, 1])
        links.append(body)
        parent, pos = body, [length, 0.0, 0.0]
    arm = links[1]
    arm.add_site(name="lift_point", pos=[pub.ARM_LENGTH, 0, 0], size=[0.05, 0, 0])
    if load > 0:
        hook = arm.add_body(name="load", pos=[pub.ARM_LENGTH, 0, 0])
        _inertial(hook, load, [0.0, 0.0, 0.0], (0.01, 0.01, 0.01))
        hook.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.15, 0, 0], contype=0,
                      conaffinity=0, density=0, rgba=[0.7, 0.2, 0.2, 1])
    if pose is None:
        for joint in ("swing", "boom", "arm", "bucket"):
            act = spec.add_actuator(name=joint, target=joint)
            act.trntype = mujoco.mjtTrn.mjTRN_JOINT
            act.set_to_motor()
    return spec


def support_forces(m, d):
    """The normal force on each of the four contact spheres [N]."""
    out = {}
    f = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        name = None
        for g in (c.geom1, c.geom2):
            n = m.geom(g).name
            if n.startswith("support_"):
                name = n
        if name is None:
            continue
        mujoco.mj_contactForce(m, d, i, f)
        out[name] = out.get(name, 0.0) + f[0]
    return out


def settle(load, pose, slew, seconds=3.0):
    """The machine, its front fixed in pose and its upper structure turned by
    slew, with the load at the bucket pin, left to settle on its supports;
    returns the supports' normal forces and the undercarriage's tilt [rad]."""
    spec = build(load=load, pose=pose, slew=slew)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
    z = d.xmat[m.body("lower").id].reshape(3, 3)[:, 2]
    return support_forces(m, d), math.acos(max(-1.0, min(1.0, z[2]))), m


def onset(reach, height, side=False, settle=2.0, ramp=5.0, hold=24.0,
          tipped=math.radians(0.1), bracket=(0.98, 1.02), iterations=10, g=9.81):
    """The load from which the simulated machine tips over, as a fraction of
    the static tipping load of capacity.py, and that load [kg]. The machine,
    its front fixed in the lifting pose and its upper structure turned for
    the side, settles on its four supports without a load for settle
    seconds; a downward force at the lift point, the load's weight, then
    rises linearly over ramp seconds to f times the static tipping load's
    and is held for hold seconds (the load's own inertia, which a slow lift
    hardly excites, is left out); the machine has tipped if the
    undercarriage then leans more than tipped from upright (0.1 deg; the
    hold is long, since a machine that starts to tip near its tipping load
    leans slowly). Bisection on f within bracket. The contacts have MuJoCo's
    default softness (solref 0.02 s, 1)."""
    analytic = capacity.tipping(reach, height, side)
    pose = capacity.pose(reach, height)
    m = build(load=0.0, pose=pose, slew=math.pi / 2 if side else 0.0).compile()
    lower, arm, point = m.body("lower").id, m.body("arm").id, m.site("lift_point").id
    dt = m.opt.timestep
    n_settle, n_ramp = int(round(settle / dt)), int(round(ramp / dt))
    n_total = n_settle + n_ramp + int(round(hold / dt))
    torque = np.zeros(3)

    def tips(f):
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        for i in range(n_total):
            d.qfrc_applied[:] = 0.0
            share = min(1.0, max(0.0, (i + 1 - n_settle) / n_ramp))
            if share > 0.0:
                force = np.array([0.0, 0.0, -share * f * analytic * g])
                mujoco.mj_applyFT(m, d, force, torque, d.site_xpos[point].copy(), arm,
                                  d.qfrc_applied)
            mujoco.mj_step(m, d)
        z = d.xmat[lower].reshape(3, 3)[:, 2]
        return math.acos(max(-1.0, min(1.0, z[2]))) > tipped

    lo, hi = bracket
    if tips(lo) or not tips(hi):
        raise RuntimeError("the bracket does not hold the onset")
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if tips(mid) else (mid, hi)
    return 0.5 * (lo + hi), analytic


def tipping_load(reach, height, side=False):
    """The load at which the supports away from the tipping line would carry
    nothing, extrapolated from the settled forces at two loads well below it,
    and the analytic tipping load of capacity.py. While all four supports
    touch, a rigid machine's forces are linear in the load, so this checks
    the simulator's load balance against the static model's, not the onset
    of tipping (onset() simulates that)."""
    analytic = capacity.tipping(reach, height, side)
    pose = capacity.pose(reach, height)
    slew = math.pi / 2 if side else 0.0
    far = ("support_rl", "support_rr") if not side else ("support_fr", "support_rr")
    loads = (0.25 * analytic, 0.5 * analytic)
    forces = []
    for w in loads:
        f, tilt, _ = settle(w, pose, slew)
        forces.append(sum(f.get(n, 0.0) for n in far))
    slope = (forces[1] - forces[0]) / (loads[1] - loads[0])
    return loads[0] - forces[0] / slope, analytic
