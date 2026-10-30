"""
A heavy-duty mobile manipulator for Section 14.1: synthetic, every number ours.

Physical AI, Together, Chapter 14 (Section 14.1).

The machine that Chapters 1 to 9 describe in words, a heavy mobile
manipulator that drives a payload through turns, has no published model;
the all-electric heavy-duty manipulators of the literature that we read
(Bahari et al. 2025, Barjini et al. 2025, Bahari et al. 2024) publish no
masses, link lengths or joint limits. This model is therefore a stand-in
whose every number is ours, chosen round and of the size of such a machine,
to show the coupling of a base and an arm; it is not a model of a machine.

  * the base: a rigid chassis of 3,500 kg, its centre of mass on its centre
    line, 0.75 m above the ground; it stands on four wheels whose contacts
    with the ground are at x = +-1.10 m (wheelbase 2.2 m) and y = +-0.85 m
    (track 1.7 m), modelled as four small spheres there;
  * the arm on the base's top, its slew axis at x = 0.60 m, its shoulder (the
    lift joint) 1.30 m above the ground: a boom of 2.4 m and 350 kg (centre
    of mass 1.1 m from the shoulder), a stick of 2.0 m and 200 kg (0.9 m from
    the elbow, the tilt joint), a tool of 100 kg at the stick's end, and a
    payload of up to 300 kg there (Bahari et al. 2025 emulated the loads on
    their all-electric heavy-duty manipulator's lift joint on an actuator
    testbed of one degree of freedom, Section V, with payloads from 0 to
    300 kg, Section VII);
  * the arm's joints: slew about z, lift and tilt about (0, -1, 0); with
    pose given, the arm is built rigid in that pose; their ranges (ours):
    lift 0 to 60 deg, tilt -150 to 0 deg, the tool's centre at least 0.2 m
    above the ground, the slew free.
"""
import math

import mujoco
import numpy as np

BASE = dict(mass=3500.0, com=(0.0, 0.0, 0.75), box=(3.0, 1.8, 0.8))
AXLE = 1.10          # [m] the wheels' contacts at x = +-AXLE
HALF_TRACK = 0.85    # [m] and at y = +-HALF_TRACK
SLEW = (0.60, 1.30)  # [m] the slew axis's x and the shoulder's height
BOOM = dict(mass=350.0, length=2.4, com=1.1, box=(2.4, 0.3, 0.35))
STICK = dict(mass=200.0, length=2.0, com=0.9, box=(2.0, 0.25, 0.3))
TOOL = 100.0         # [kg] at the stick's end
PAYLOAD_MAX = 300.0  # [kg]
SPHERE = 0.02
TIMESTEP = 0.001
IMPRATIO = 20.0      # the ratio of frictional to normal impedance of the simulated contacts:
                     # at MuJoCo's default, 1, the machine creeps on its contacts under a
                     # steady horizontal load; at 20, far less (ours; Section 14.2)
POSES = {            # (slew, lift, tilt) [rad] (ours)
    "stowed": (0.0, math.radians(60.0), math.radians(-150.0)),
    "carry": (0.0, math.radians(20.0), math.radians(-70.0)),
    "reach": (0.0, 0.0, 0.0),
}
LIFT_RANGE = (math.radians(0.0), math.radians(60.0))      # [rad] the arm's ranges (ours)
TILT_RANGE = (math.radians(-150.0), math.radians(0.0))
TOOL_ABOVE = 0.2     # [m] the tool's centre at least this high: its box's half-height


def _quat_y(angle):
    return [math.cos(angle / 2), 0.0, -math.sin(angle / 2), 0.0]


def _inertial(body, m, com, box):
    a, b, c = box
    body.explicitinertial = True
    body.mass = m
    body.ipos = list(com)
    body.iquat = [1, 0, 0, 0]
    body.inertia = [m * (b * b + c * c) / 12, m * (a * a + c * c) / 12, m * (a * a + b * b) / 12]


def tip_point(slew, lift, tilt):
    """The stick's end (tool and payload) in the base's frame."""
    r = BOOM["length"] * math.cos(lift) + STICK["length"] * math.cos(lift + tilt)
    z = SLEW[1] + BOOM["length"] * math.sin(lift) + STICK["length"] * math.sin(lift + tilt)
    return np.array([SLEW[0] + r * math.cos(slew), r * math.sin(slew), z])


def parts(slew, lift, tilt, payload):
    """Each part's mass and centre of mass in the base's frame."""
    c, s = math.cos(slew), math.sin(slew)

    def at(r, z):
        return np.array([SLEW[0] + r * c, r * s, z])

    boom = at(BOOM["com"] * math.cos(lift), SLEW[1] + BOOM["com"] * math.sin(lift))
    elbow_r = BOOM["length"] * math.cos(lift)
    elbow_z = SLEW[1] + BOOM["length"] * math.sin(lift)
    stick = at(elbow_r + STICK["com"] * math.cos(lift + tilt),
               elbow_z + STICK["com"] * math.sin(lift + tilt))
    tip = tip_point(slew, lift, tilt)
    return [(BASE["mass"], np.array(BASE["com"])), (BOOM["mass"], boom), (STICK["mass"], stick),
            (TOOL, tip), (payload, tip)]


def centre_of_mass(slew, lift, tilt, payload):
    p = parts(slew, lift, tilt, payload)
    m = sum(x[0] for x in p)
    return m, sum(x[0] * x[1] for x in p) / m


def quasi_static(slew, lift, tilt, payload, g=9.81):
    """The decelerations [m/s^2] at which the rear wheels (braking forwards),
    the front wheels (accelerating), the inner or the outer wheels (a turn's
    centripetal acceleration to either side) start to lift, for the rigid
    machine on flat ground: a = g * (distance from the centre of mass to the
    tipping line) / (its height); negative when the machine already tips at
    rest."""
    _, c = centre_of_mass(slew, lift, tilt, payload)
    h = c[2]
    return {"brake": g * (AXLE - c[0]) / h, "accelerate": g * (AXLE + c[0]) / h,
            "left": g * (HALF_TRACK - c[1]) / h, "right": g * (HALF_TRACK + c[1]) / h,
            "com": c}


def turn_threshold(slew, lift, tilt, payload, radius, g=9.81):
    """The lateral acceleration at the centre of mass [m/s^2] at which the
    inner wheels lift in a steady turn of the given radius [m] about a
    vertical axis level with the centre of mass, the machine tipping
    outwards to the left: each part feels w^2 times its own distance from the
    axis, so that parts farther out feel more; for a large radius the result
    tends to quasi_static()["left"], which takes the acceleration uniform."""
    p = parts(slew, lift, tilt, payload)
    total, c = centre_of_mass(slew, lift, tilt, payload)
    spread = sum(m * (q[1] - c[1]) * q[2] for m, q in p)   # sum of m (y - y_c) z
    return g * (HALF_TRACK - c[1]) / (c[2] + spread / (total * radius))


def worst_braking(payload, coarse=math.radians(0.5), fine=math.radians(0.01), g=9.81):
    """The smallest braking at which the rear wheels lift over the arm's
    ranges, LIFT_RANGE and TILT_RANGE, with the tool's centre at least
    TOOL_ABOVE above the ground: (deceleration [m/s^2], lift, tilt [rad]),
    from a grid of 0.5 deg refined by one of 0.01 deg around its smallest.
    The arm points forward: slewing it moves the centre of mass back and
    leaves its height alone, which can only raise the braking limit."""
    def search(lifts, tilts):
        lift, tilt = np.meshgrid(lifts, tilts, indexing="ij")
        r_tip = BOOM["length"] * np.cos(lift) + STICK["length"] * np.cos(lift + tilt)
        z_tip = SLEW[1] + BOOM["length"] * np.sin(lift) + STICK["length"] * np.sin(lift + tilt)
        x_boom = SLEW[0] + BOOM["com"] * np.cos(lift)
        z_boom = SLEW[1] + BOOM["com"] * np.sin(lift)
        x_stick = (SLEW[0] + BOOM["length"] * np.cos(lift)
                   + STICK["com"] * np.cos(lift + tilt))
        z_stick = (SLEW[1] + BOOM["length"] * np.sin(lift)
                   + STICK["com"] * np.sin(lift + tilt))
        tip_mass = TOOL + payload
        total = BASE["mass"] + BOOM["mass"] + STICK["mass"] + tip_mass
        x = (BASE["mass"] * BASE["com"][0] + BOOM["mass"] * x_boom + STICK["mass"] * x_stick
             + tip_mass * (SLEW[0] + r_tip)) / total
        z = (BASE["mass"] * BASE["com"][2] + BOOM["mass"] * z_boom + STICK["mass"] * z_stick
             + tip_mass * z_tip) / total
        a = np.where(z_tip >= TOOL_ABOVE, g * (AXLE - x) / z, np.inf)
        k = np.unravel_index(np.argmin(a), a.shape)
        return float(a[k]), float(lift[k]), float(tilt[k])

    def grid(lo, hi, step):           # from lo to hi, both included, at about step
        return np.linspace(lo, hi, int(round((hi - lo) / step)) + 1)

    a, lift, tilt = search(grid(*LIFT_RANGE, coarse), grid(*TILT_RANGE, coarse))
    return search(grid(max(LIFT_RANGE[0], lift - coarse), min(LIFT_RANGE[1], lift + coarse), fine),
                  grid(max(TILT_RANGE[0], tilt - coarse), min(TILT_RANGE[1], tilt + coarse), fine))


def build(payload=0.0, pose=None, friction=0.6, timestep=TIMESTEP, kp=None, fixed_base=False,
          solref=None, impratio=IMPRATIO):
    """The MjSpec of the machine with a payload [kg]: the base on a free joint
    on four contact spheres of the given friction (or fixed to the world),
    the arm rigid in pose (a triple slew, lift, tilt) or jointed with
    position servos of stiffness kp [N m/rad] (ours, to hold the arm while
    the base moves). The contacts have MuJoCo's default softness (solref
    0.02 s, 1; solimp 0.9, 0.95, 0.001) unless solref is given, and the ratio
    of frictional to normal impedance IMPRATIO, or impratio; None leaves
    MuJoCo's default, 1."""
    spec = mujoco.MjSpec()
    spec.modelname = "hdmm"
    spec.compiler.degree = False
    spec.option.timestep = timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    if impratio is not None:
        spec.option.impratio = impratio
    ground = spec.worldbody.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE,
                                     size=[200, 50, 0.1], friction=[friction, 0.005, 0.0001],
                                     rgba=[0.5, 0.5, 0.45, 1])
    if solref is not None:
        ground.solref = list(solref)
    base = spec.worldbody.add_body(name="base", pos=[0, 0, SPHERE])
    if not fixed_base:
        base.add_freejoint(name="ground_joint")
    com = list(BASE["com"])
    com[2] -= SPHERE
    _inertial(base, BASE["mass"], com, BASE["box"])
    for sx in (-1, 1):
        for sy in (-1, 1):
            wheel = base.add_geom(name=f"wheel_{'f' if sx > 0 else 'r'}{'l' if sy > 0 else 'r'}",
                                  type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[SPHERE, 0, 0],
                                  pos=[sx * AXLE, sy * HALF_TRACK, 0.0],
                                  friction=[friction, 0.005, 0.0001], density=0)
            if solref is not None:
                wheel.solref = list(solref)
    base.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[s / 2 for s in BASE["box"]],
                  pos=[0, 0, 0.75 - SPHERE], contype=0, conaffinity=0, density=0,
                  rgba=[0.2, 0.4, 0.6, 1])
    sl, lf, tl = (0.0, 0.0, 0.0) if pose is None else pose
    turret = base.add_body(name="turret", pos=[SLEW[0], 0, SLEW[1] - SPHERE],
                           quat=[math.cos(sl / 2), 0, 0, math.sin(sl / 2)])
    _inertial(turret, 1e-3, [0, 0, 0], (0.1, 0.1, 0.1))
    boom = turret.add_body(name="boom", quat=_quat_y(lf))
    _inertial(boom, BOOM["mass"], [BOOM["com"], 0, 0], BOOM["box"])
    boom.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, 0, BOOM["length"], 0, 0],
                  size=[0.1, 0, 0], contype=0, conaffinity=0, density=0, rgba=[0.9, 0.6, 0.1, 1])
    stick = boom.add_body(name="stick", pos=[BOOM["length"], 0, 0], quat=_quat_y(tl))
    _inertial(stick, STICK["mass"], [STICK["com"], 0, 0], STICK["box"])
    stick.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, 0, STICK["length"], 0, 0],
                   size=[0.08, 0, 0], contype=0, conaffinity=0, density=0, rgba=[0.9, 0.6, 0.1, 1])
    tool = stick.add_body(name="tool", pos=[STICK["length"], 0, 0])
    _inertial(tool, TOOL + payload, [0, 0, 0], (0.4, 0.4, 0.4))
    tool.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.2, 0.2, 0.2], contype=0,
                  conaffinity=0, density=0, rgba=[0.7, 0.2, 0.2, 1])
    if pose is None:
        for body, name, axis in ((turret, "slew", [0, 0, 1]), (boom, "lift", [0, -1, 0]),
                                 (stick, "tilt", [0, -1, 0])):
            body.add_joint(name=name, type=mujoco.mjtJoint.mjJNT_HINGE, axis=axis)
            act = spec.add_actuator(name=name, target=name)
            act.trntype = mujoco.mjtTrn.mjTRN_JOINT
            act.set_to_position(kp=kp or 2e6, kv=(kp or 2e6) * 0.05)
    return spec


def wheel_loads(m, d):
    out = {}
    f = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        for g in (c.geom1, c.geom2):
            n = m.geom(g).name
            if n.startswith("wheel_"):
                mujoco.mj_contactForce(m, d, i, f)
                out[n] = out.get(n, 0.0) + f[0]
    return out


# ----------------------------------------------------------------------------
# The simulated machine under a steady acceleration
# ----------------------------------------------------------------------------
def held(payload, pose, accel=(0.0, 0.0), seconds=2.0, jointed=False, g=9.81):
    """The machine at rest under gravity tilted by a horizontal acceleration
    accel = (ax, ay) of its base [m/s^2]: in the base's frame a steady
    acceleration a acts on every part as the force -m a (d'Alembert), so the
    contact forces under gravity (-ax, -ay, -g) are those of the machine
    accelerating at a on flat ground, the ground's friction (1.5 here, so that
    nothing slides, with the frictional impedance IMPRATIO, which keeps the
    base's creep small, labels_demand.py) giving the horizontal forces, as
    braking wheels would.
    Returns the wheels' normal loads [N], and the data and model."""
    spec = build(payload=payload, pose=None if jointed else pose, friction=1.5)
    m = spec.compile()
    d = mujoco.MjData(m)
    m.opt.gravity[:] = [-accel[0], -accel[1], -g]
    if jointed:
        d.qpos[7:10] = pose
        d.ctrl[:] = pose
    mujoco.mj_forward(m, d)
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
    return wheel_loads(m, d), d, m


TIP_WHEELS = {"brake": ("wheel_rl", "wheel_rr"), "accelerate": ("wheel_fl", "wheel_fr"),
              "left": ("wheel_fr", "wheel_rr"), "right": ("wheel_fl", "wheel_rl")}
DIRECTIONS = {"brake": (-1.0, 0.0), "accelerate": (1.0, 0.0), "left": (0.0, -1.0),
              "right": (0.0, 1.0)}


def simulated_threshold(payload, pose, which, a=(0.5, 1.0)):
    """The acceleration at which the wheels away from the tipping line would
    carry nothing, extrapolated from the simulator's settled loads at two
    accelerations well below it. While all four wheels touch, a rigid
    machine's loads are linear in the acceleration, so this checks the
    simulator's load balance against the rigid machine's, not the onset of
    tipping (onset() simulates that). which: brake (decelerating forwards),
    accelerate, left or right (the net horizontal acceleration towards that
    side, so that the machine would tip towards it)."""
    ux, uy = DIRECTIONS[which]
    loads = []
    for k in a:
        w, _, _ = held(payload, pose, accel=(ux * k, uy * k))
        loads.append(sum(w.get(n, 0.0) for n in TIP_WHEELS[which]))
    slope = (loads[1] - loads[0]) / (a[1] - a[0])
    return a[0] - loads[0] / slope


def onset(payload, pose, which, at_once=False, settle=2.0, ramp=5.0, hold=24.0,
          tipped=math.radians(0.1), bracket=(0.98, 1.02), iterations=10, g=9.81,
          solref=None):
    """The steady acceleration from which the simulated machine tips over, as
    a fraction of the rigid machine's threshold quasi_static(...)[which], and
    that threshold [m/s^2]. The machine, rigid in pose on its four wheel
    contacts (friction 1.5, as in held()), settles at rest for settle
    seconds; the acceleration then rises to f times the threshold, linearly
    over ramp seconds or at once, and is held for hold seconds; the machine
    has tipped if its base then leans more than tipped from upright (0.1 deg,
    against the contacts' lean of under 0.01 deg at rest; a machine that
    starts to tip near the threshold leans slowly, so the hold is long).
    Bisection on f within bracket. solref: the contacts' softness, as in
    build()."""
    threshold = quasi_static(*pose, payload, g=g)[which]
    ux, uy = DIRECTIONS[which]
    m = build(payload=payload, pose=pose, friction=1.5, solref=solref).compile()
    base = m.body("base").id
    dt = m.opt.timestep

    def tips(f):
        d = mujoco.MjData(m)
        m.opt.gravity[:] = [0.0, 0.0, -g]
        mujoco.mj_forward(m, d)
        for _ in range(int(round(settle / dt))):
            mujoco.mj_step(m, d)
        n = 1 if at_once else int(round(ramp / dt))
        for i in range(n):
            k = f * threshold * (i + 1) / n
            m.opt.gravity[:] = [-ux * k, -uy * k, -g]
            mujoco.mj_step(m, d)
        for _ in range(int(round(hold / dt))):
            mujoco.mj_step(m, d)
        z = d.xmat[base].reshape(3, 3)[:, 2]
        return math.acos(max(-1.0, min(1.0, z[2]))) > tipped

    lo, hi = bracket
    if tips(lo) or not tips(hi):
        raise RuntimeError("the bracket does not hold the onset")
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if tips(mid) else (mid, hi)
    return 0.5 * (lo + hi), threshold


def centre_of_pressure(m, d):
    """The centre of pressure (x, y) [m] of the normal forces of the
    simulated machine's contacts with the ground, relative to the base's
    position (which a slow creep on the contacts moves; the base does not
    turn), and their sum [N]."""
    f = np.zeros(6)
    load, moment = 0.0, np.zeros(2)
    for i in range(d.ncon):
        mujoco.mj_contactForce(m, d, i, f)
        load += f[0]
        moment += f[0] * d.contact[i].pos[:2]
    if load <= 0:
        return np.full(2, np.nan), load
    return moment / load - d.qpos[:2], load


_RIGID = {}


def motion_basis(payload, pose, g=9.81):
    """The rigid machine in pose under the base's motion, from MuJoCo's
    inverse dynamics: the jointed machine (build() without a pose) with its
    contacts disabled, the base upright, its origin at its middle, SPHERE
    above the ground, and the arm at rest in pose; the base's middle accelerating at (ax, ay),
    along x and y, and the base turning about the vertical at the yaw rate w
    with the yaw acceleration alpha. The generalised forces that this motion
    needs, the first six the wrench that the ground must supply (the force,
    and the moment about the base's middle, its origin, SPHERE above the
    ground) and the last three the arm's holding torques, slew, lift and
    tilt, are affine in (ax, ay, alpha, w^2), and do not depend on the
    base's speed: returns the forces at rest under gravity and their changes
    per unit of each, a dict of arrays of nine, q0, ax, ay, alpha and w2."""
    key = (float(payload), tuple(float(x) for x in pose), g)
    if key in _RIGID:
        return _RIGID[key]
    if float(payload) not in _RIGID:
        m = build(payload=payload, pose=None, friction=1.5).compile()
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        _RIGID[float(payload)] = (m, mujoco.MjData(m))
    m, d = _RIGID[float(payload)]
    m.opt.gravity[:] = [0.0, 0.0, -g]

    def forces(ax=0.0, ay=0.0, alpha=0.0, w=0.0):
        d.qpos[:] = 0.0
        d.qpos[2] = SPHERE
        d.qpos[3] = 1.0
        d.qpos[7:10] = pose
        d.qvel[:] = 0.0
        d.qvel[5] = w
        d.qacc[:] = 0.0
        d.qacc[0], d.qacc[1], d.qacc[5] = ax, ay, alpha
        mujoco.mj_inverse(m, d)
        return d.qfrc_inverse.copy()

    q0 = forces()
    out = dict(q0=q0, ax=forces(ax=1.0) - q0, ay=forces(ay=1.0) - q0,
               alpha=forces(alpha=1.0) - q0, w2=forces(w=1.0) - q0)
    _RIGID[key] = out
    return out


def wrench_zmp(q):
    """The zero-moment point (x, y) [m] relative to the base's middle of the
    ground's wrench q[..., :6] (motion_basis), on the ground, SPHERE below
    the base's origin, and the normal force [N]."""
    q = np.asarray(q)
    f, mo = q[..., 0:3], q[..., 3:6]
    px = -(mo[..., 1] + SPHERE * f[..., 0]) / f[..., 2]
    py = (mo[..., 0] - SPHERE * f[..., 1]) / f[..., 2]
    return px, py, f[..., 2]


def simulated_lift(payload, start, end, duration, accel=(0.0, 0.0), onset=None, settle=2.0,
                   hold=2.0, kp=None, impratio=IMPRATIO, offset=True, rise=0.0,
                   before=(0.0, 0.0), changes=None, friction=1.5, contacts=False, motion=None,
                   g=9.81):
    """The jointed machine (build() without a pose) on its four wheel
    contacts (friction 1.5, as in held(), unless given) moves its arm from
    the pose start to the pose end, both triples (slew, lift, tilt), its
    servos' targets following q(s) = start + (end - start) s with s = (1 -
    cos(pi t / duration)) / 2; with start equal to end, it holds the arm in
    that pose. Gravity is tilted by the base's acceleration: accel
    throughout, the machine settling under it, or, with onset given, before
    until onset seconds after the lift's start and from then on accel,
    reached at once or, with rise, along a half-cosine over rise seconds, as
    the base's actuators would change it; or, with changes, a list of
    (time, acceleration) in order, before until the first and each reached
    from the one in force along a half-cosine over rise seconds (each at
    least rise seconds after the last). Or, with motion, a function of the
    time from the lift's start that gives the base's motion, (ax, ay, w,
    alpha): the acceleration of the base's middle, which tilts gravity, and
    the yaw rate and acceleration of a base that turns; the base does not
    turn in the simulation, which holds the machine in the base's frame, and
    every body feels that frame's forces instead, applied at its centre of
    mass: -m alpha z x r, m w^2 r and -2 m w z x u, r its distance from the
    base's middle across the vertical and u its velocity, and the torque
    -(I alpha z + w^2 z x I z); the run settles under motion(0). With
    offset, each servo's target
    is offset by the torque that holds the arm exactly in the target pose
    under gravity alone (holding_torques without an acceleration), divided
    by its stiffness, so that the servos hold the planned pose at rest
    instead of sagging under the load; the base's acceleration is not fed
    forward, and the servos meet it only through their stiffness. It settles
    in start for settle seconds and holds end for hold seconds. Returns,
    every step from the lift's start to the end of the hold, the time from
    the lift's start [s], the centre of pressure (x, y) relative to the base
    [m], the arm's joints' angles [rad], the servos' torques [N m], the
    acceleration acting (ax, ay) [m/s^2] and the base's travel on its
    contacts since the lift's start (x, y) [m]; with motion, also the yaw
    rate and acceleration acting (w, alpha) and the base's turn on its
    contacts [rad]; with contacts, also the
    number of contacts and the wheels touching the ground. kp as in build()
    (the servos' stiffness, by default 2e6 N m/rad; their damping is 0.05 kp
    s), impratio as in build()."""
    stiffness = kp or 2e6
    m = build(payload=payload, pose=None, friction=friction, kp=stiffness,
              impratio=impratio).compile()
    d = mujoco.MjData(m)
    mf = build(payload=payload, pose=None, fixed_base=True).compile()
    df = mujoco.MjData(mf)
    mf.opt.gravity[:] = [0.0, 0.0, -g]
    q0, q1 = np.array(start, dtype=float), np.array(end, dtype=float)
    if changes is None:
        first = np.array(accel if onset is None else before, dtype=float)
        steps = [] if onset is None else [(float(onset), np.array(accel, dtype=float))]
    else:
        first = np.array(before, dtype=float)
        steps = [(float(t_), np.array(a_, dtype=float)) for t_, a_ in changes]
    for (ta, _), (tb, _) in zip(steps, steps[1:]):
        if tb < ta + rise - 1e-9:
            raise ValueError("a change begins before the last has been reached")
    dt = m.opt.timestep
    wheels = {m.geom(f"wheel_{w}").id: w for w in ("fl", "fr", "rl", "rr")}

    def acting(t):
        a = first
        for t_, a_ in steps:
            if t < t_ - 1e-12:
                break
            if rise <= 0.0 or t >= t_ + rise:
                a = a_
                continue
            u = 0.5 * (1.0 - math.cos(math.pi * (t - t_) / rise))
            return a + (a_ - a) * u
        return a

    def target(q):
        if not offset:
            return q
        df.qpos[:3] = q
        mujoco.mj_forward(mf, df)
        return q + df.qfrc_bias[:3] / stiffness

    held_target = target(q0) if np.array_equal(q0, q1) else None
    base_id = m.body("base").id
    jacp = np.zeros((3, m.nv))

    def frame_forces(w, alpha):
        o = d.xpos[base_id]
        for b in range(1, m.nbody):
            r = d.xipos[b] - o
            mujoco.mj_jacBodyCom(m, d, jacp, None, b)
            u = jacp @ d.qvel
            mass = m.body_mass[b]
            d.xfrc_applied[b, 0] = mass * (alpha * r[1] + w * w * r[0] + 2.0 * w * u[1])
            d.xfrc_applied[b, 1] = mass * (-alpha * r[0] + w * w * r[1] - 2.0 * w * u[0])
            d.xfrc_applied[b, 2] = 0.0
            rot = d.ximat[b].reshape(3, 3)
            iz = rot @ (m.body_inertia[b] * rot[2, :])     # the inertia tensor's column z
            d.xfrc_applied[b, 3] = -(alpha * iz[0] - w * w * iz[1])
            d.xfrc_applied[b, 4] = -(alpha * iz[1] + w * w * iz[0])
            d.xfrc_applied[b, 5] = -alpha * iz[2]

    def state(t):
        if motion is None:
            a = acting(t)
            return a[0], a[1], 0.0, 0.0
        return motion(max(t, 0.0))

    ax, ay, w, alpha = state(-settle)
    m.opt.gravity[:] = [-ax, -ay, -g]
    d.qpos[7:10] = q0
    d.ctrl[:] = target(q0)
    mujoco.mj_forward(m, d)
    out = {"t": [], "cop": [], "q": [], "torque": [], "accel": [], "base": []}
    if motion is not None:
        out.update(turn=[], yaw=[])
    if contacts:
        out.update(ncon=[], touching=[])
    base0 = None
    for i in range(int(round((settle + duration + hold) / dt))):
        t = (i + 1) * dt - settle
        if t <= 0.0:
            s = 0.0
        elif t >= duration:
            s = 1.0
        else:
            s = 0.5 * (1.0 - math.cos(math.pi * t / duration))
        ax, ay, w, alpha = state(t)
        m.opt.gravity[:] = [-ax, -ay, -g]
        if motion is not None:
            frame_forces(w, alpha)
        d.ctrl[:] = held_target if held_target is not None else target(q0 + (q1 - q0) * s)
        mujoco.mj_step(m, d)
        if t < 0.0:
            continue
        if base0 is None:
            base0 = d.qpos[:2].copy()
        p, _ = centre_of_pressure(m, d)
        out["t"].append(t)
        out["cop"].append(p)
        out["q"].append(d.qpos[7:10].copy())
        out["torque"].append(d.actuator_force.copy())
        out["accel"].append(np.array((ax, ay), dtype=float))
        out["base"].append(d.qpos[:2] - base0)
        if motion is not None:
            qw, qx, qy, qz = d.qpos[3:7]
            out["turn"].append(np.array((w, alpha), dtype=float))
            out["yaw"].append(math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
        if contacts:
            out["ncon"].append(d.ncon)
            out["touching"].append(frozenset(wheels[g_] for c in range(d.ncon)
                                             for g_ in (d.contact[c].geom1, d.contact[c].geom2)
                                             if g_ in wheels))
    return {k: (v if k == "touching" else np.array(v)) for k, v in out.items()}


def holding_torques(payload, pose, accel=(0.0, 0.0), g=9.81):
    """The torques [N m] of the slew, the lift and the tilt that hold the arm
    exactly in pose while the base moves at the steady acceleration accel =
    (ax, ay): in the base's frame, which does not turn, the acceleration acts
    on every part as gravity tilted to (-ax, -ay, -g), so the torques are the
    arm's generalised gravity forces under that gravity, with the base held
    fixed and the arm at rest."""
    m = build(payload=payload, pose=None, fixed_base=True).compile()
    d = mujoco.MjData(m)
    m.opt.gravity[:] = [-accel[0], -accel[1], -g]
    d.qpos[:3] = pose
    mujoco.mj_forward(m, d)
    return d.qfrc_bias[:3].copy()
