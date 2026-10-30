"""
The synthetic heavy-duty mobile manipulator's labels, constraints, recovery
and demands: Section 14.2.

Physical AI, Together, Chapter 14 (Section 14.2).

Every number here is ours, like the machine's (model.py).

A label of the machine is a capability of the kind that Section 4.1 names
for a mobile robot: an interval for the payload and a lower bound on the
ground's friction, Label(payloads, friction). What a plan may command under a
label, and the reserves that it keeps, follow from it for each pose of the
arm:

  * the base's actuators: brakes that decelerate at up to BRAKES, a drive
    that accelerates at up to DRIVE, a top speed SPEED, the brakes and the
    drive following every change of their command along a half-cosine over
    RISE, a stop fading its deceleration to zero at standstill; the base
    steers along its route, which enters and leaves each turn along a
    transition of TRANSITION metres, its curvature changing along a
    half-cosine of the distance (route_curvature);
  * the ground: each axle's horizontal force within friction times the
    axle's load, the loads shared between the axles as the zero-moment
    point sets them, the force along the route, braking or driving, shared
    between the axles and between each axle's wheels in proportion to their
    loads (ours, an assumption about the synthetic machine), and the axles'
    lateral forces both holding the machine in a turn and turning it
    (axle_grip);
  * the arm's actuators: the torques that hold the arm in its pose under the
    base's motion, the part that the motion adds counted KAPPA times for the
    servos' overshoot, within ARM_LIMITS (slew, lift, tilt), for every
    payload of the label;
  * tipping over, a constraint with a reserve: the zero-moment point, where
    the resultant of gravity and of the parts' inertial forces meets the
    ground, inside the rectangle of the wheels' contacts; the reserve is its
    least distance inside, over every payload of the label, less ALLOWANCE,
    and its floor FLOOR;
  * the clearance, a constraint with a reserve: the machine's front-most
    point, the tool's box or the chassis, behind a stop line; the reserve is
    its distance from the line less ALLOWANCE, and its floor CLEAR_FLOOR.

The plans hold the arm in one of the poses of labels_demand.py (stowed,
carry, reach and reach left) or lift it from carrying to reaching on a
straight; no other arm motion is commanded. They keep their speed from a
turn's entry to its exit, and their recovery brakes there at TURN_BRAKING,
a recovery that would reach a turn braking harder easing its deceleration
to that over RISE before the turn's entry; from rest in a turn, a plan
drives off at TURN_DRIVE to a speed that it then keeps. The machine is taken rigid in its pose, every
part sharing the base's motion; the contacts' compliance and the arm's
motion on its servos are left out. The test takes ALLOWANCE off both
reserves for them: labels_demand.py finds the simulated machine, its arm on
its servos, within it at every instant with the arm held, at every braking
limit of these poses and through the turns at their speeds, and within it
at the least of every simulated lift; during a lift it falls further short
at an instant, where the rigid machine's distance is far above the lift's
least, so a segment of a lift is certified at no more than the level at the
lift's end. The extremes over a payload interval lie at its ends, because
the zero-moment point and the torques are monotone in the payload's mass;
the functions here also take its middle.

margin() gives the rigid machine's zero-moment point's distance inside the
rectangle on a straight, without the allowance, against which the simulator
is compared; tipping_reserve() gives the reserve. braking_limit() gives the
largest deceleration that a plan may command on a straight in a pose under
a label while it keeps a given tipping reserve, and which of the four
limits binds. The recovery brakes within the braking limit, its
deceleration rising over RISE and fading over RISE at standstill, the arm
held in its pose, to rest (stop_distance), and the rest segment holds the
machine on its brakes.

In a turn the base also turns, and every part feels the turn's
acceleration at its own place: turn_states() gives the states of the base's
middle through a turn of a radius and its transitions, turn_slack() the
rigid machine's slack against each constraint there, from MuJoCo's inverse
dynamics (model.motion_basis), and turn_speed() the largest speed at which a
plan may take the turn, keeping it through the turn and braking in it at
TURN_BRAKING from that speed or any lower one.

A lift with the arm extended consumes margin against tipping over and, with
a stop line ahead, clearance: lift_demand() gives both, and the
requirement, the floor plus what is consumed, from which a plan that begins
the lift can be certified at the floor (the book's assumption of reserve
consumption, Section 4.3). A turn on the route needs no demand to be taken
at the floor: every plan that takes it keeps its speed within turn_speed().
"""
import math
from collections import namedtuple

import numpy as np

from . import model

G = 9.81
BRAKES = 4.0                 # [m/s^2] the brakes' largest deceleration (ours)
DRIVE = 1.5                  # [m/s^2] the drive's largest acceleration (ours)
SPEED = 3.0                  # [m/s] the base's top speed (ours)
RISE = 0.6                   # [s] the base's acceleration along its route, braking or driving,
                             # follows every change of its command along a half-cosine over RISE:
                             # the brakes and the drive cannot change it at once, and a stop fades
                             # its deceleration to zero at standstill (ours)
TRANSITION = 3.0             # [m] the route enters and leaves each turn along a transition over
                             # which its curvature changes along a half-cosine of the distance
                             # (ours); at the top speed it takes 1 s, more than RISE
TURN_ANGLE = math.pi / 2     # [rad] the route's turns are corners (ours)
TURN_BRAKING = 0.5           # [m/s^2] the deceleration at which a plan's recovery brakes in a turn
                             # or its transitions (ours)
TURN_DRIVE = 0.5             # [m/s^2] the acceleration at which a plan drives off from rest in a
                             # turn or its transitions, where a recovery may have stopped the
                             # machine (ours)
TURN_STEP = 0.1              # [m] the starts of a braking in a turn that the test evaluates
TURN_DT = 0.01               # [s] the time step of a braking that the test evaluates
TURN_POINTS = 121            # points along a transition that the test evaluates
KAPPA = 1.2                  # the torque that the base's motion adds to the arm's holding torque
                             # is counted KAPPA times, for the servos' overshoot (ours)
ARM_LIMITS = (8.0e3, 30.0e3, 12.0e3)   # [N m] slew, lift, tilt (ours)
JOINTS = ("slew", "lift", "tilt")
FLOOR = 0.10                 # [m] the tipping reserve's floor (ours)
CLEAR_FLOOR = 0.10           # [m] the clearance's floor (ours)
ALLOWANCE = 0.005            # [m] taken off both reserves for what the rigid model leaves out:
                             # the contacts' compliance and the arm's motion on its servos (ours,
                             # above the largest shortfall of a plan's least distance in the runs
                             # of labels_demand.py)
GRID = 0.01                  # [m] the levels of both reserves: whole centimetres
TOOL_HALF = 0.2              # [m] half the tool's box
CHASSIS_FRONT = 1.5          # [m] the chassis's front, half its 3.0 m box
STEPS = 41                   # points along a lift

Label = namedtuple("Label", "payloads friction")
DECLARED = Label((243.0, 297.0), 0.5)  # 270 kg +- 10 %, within the machine's 300 kg, and
                                       # friction at least 0.5 (ours; the ground of Section
                                       # 14.1 has 0.6, the simulator's contacts 1.5)
EMPTY = Label((0.0, 0.0), 0.5)
GLOBAL = Label((0.0, model.PAYLOAD_MAX), 0.3)   # the global capability: any payload of the
                                                # machine, friction at least 0.3 (ours)


def payloads_of(label):
    lo, hi = label.payloads
    return (lo, 0.5 * (lo + hi), hi)


def front_point(pose):
    """The machine's front-most point along x [m], relative to the base's
    middle: the tool's box or the chassis."""
    return max(CHASSIS_FRONT, model.tip_point(*pose)[0] + TOOL_HALF)


def zmp(pose, payload, ax=0.0, ay=0.0):
    """The zero-moment point (x, y) [m] of the rigid machine whose base moves
    on a straight at the acceleration (ax, ay), uniform over the machine: ax
    along x, negative when braking forwards; ay across it."""
    _, c = model.centre_of_mass(*pose, payload)
    return c[0] - c[2] * ax / G, c[1] - c[2] * ay / G


def edge_margin(px, py):
    """The distance [m] of the point (px, py) inside the rectangle of the
    wheels' contacts, negative outside, and the edge that is nearest."""
    edges = {"front": model.AXLE - px, "rear": model.AXLE + px,
             "left": model.HALF_TRACK - py, "right": model.HALF_TRACK + py}
    edge = min(edges, key=edges.get)
    return edges[edge], edge


def margin(pose, payload, ax=0.0, ay=0.0):
    """The rigid machine's zero-moment point's distance inside the rectangle
    of the wheels' contacts [m], negative outside, and the edge that is
    nearest, on a straight; without the allowance."""
    return edge_margin(*zmp(pose, payload, ax, ay))


def tipping_reserve(pose, label, ax=0.0, ay=0.0):
    """The tipping reserve [m]: the least distance of the zero-moment point
    inside the rectangle over the payloads of the label, less ALLOWANCE, with
    the edge and the payload."""
    r, edge, p = min((margin(pose, p, ax, ay) + (p,) for p in payloads_of(label)),
                     key=lambda t: t[0])
    return r - ALLOWANCE, edge, p


def torques(pose, payload, ax=0.0, ay=0.0):
    """The torques [N m] of the slew, the lift and the tilt that hold the arm
    exactly in pose under the base's acceleration (model.holding_torques)."""
    return model.holding_torques(payload, pose, accel=(ax, ay))


def _largest(feasible, hi, tol=1e-6):
    """The largest a in [0, hi] for which feasible(a) holds, feasible being
    monotone (true up to some a); 0 if feasible(0) fails."""
    if not feasible(0.0):
        return 0.0
    if feasible(hi):
        return hi
    lo = 0.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if feasible(mid) else (lo, mid)
    return lo


def arm_bound(pose, label, direction, offset=(0.0, 0.0), kappa=KAPPA):
    """The largest acceleration a >= 0 of the base along the unit vector
    direction, added to the acceleration offset, for which the arm's holding
    torques, with the part that the acceleration adds to gravity's counted
    kappa times, stay within ARM_LIMITS, for every payload of the label: the
    torques are affine in the acceleration (gravity tilted by it), so each
    joint gives one bound; 0 if a torque exceeds its limit at the offset, inf
    if no joint is loaded."""
    best = math.inf
    for p in payloads_of(label):
        tg = torques(pose, p)
        t0 = torques(pose, p, ax=offset[0], ay=offset[1])
        t1 = torques(pose, p, ax=offset[0] + direction[0], ay=offset[1] + direction[1])
        for j, limit in enumerate(ARM_LIMITS):
            base = tg[j] + kappa * (t0[j] - tg[j])
            slope = kappa * (t1[j] - t0[j])
            if abs(base) > limit * (1.0 + 1e-9):     # at the offset already beyond the limit
                return 0.0
            if slope > 1e-9:
                best = min(best, (limit - base) / slope)
            elif slope < -1e-9:
                best = min(best, (-limit - base) / slope)
    return best


def braking_limits(pose, label=DECLARED, level=FLOOR):
    """Each of the four limits on the deceleration that a plan may command
    on a straight in pose under the label while the zero-moment point stays
    at least level inside the rectangle: the brakes, the ground, tipping and
    the arm (None for tipping if the machine lacks the level at rest)."""
    if tipping_reserve(pose, label)[0] < level:
        tip = None
    else:
        tip = _largest(lambda b: tipping_reserve(pose, label, ax=-b)[0] >= level, 20.0)
    return {"brakes": BRAKES, "friction": label.friction * G, "tipping": tip,
            "arm": arm_bound(pose, label, (-1.0, 0.0))}


def braking_limit(pose, label=DECLARED, level=FLOOR):
    """The largest deceleration that a plan may command on a straight in pose
    under the label while the zero-moment point stays at least level inside
    the rectangle, and which limit binds; (0, 'tipping without braking') if
    the machine lacks the level before it brakes."""
    limits = braking_limits(pose, label, level)
    if limits["tipping"] is None:
        return 0.0, "tipping without braking"
    which = min(limits, key=limits.get)
    return limits[which], which


# ----------------------------------------------------------------------------
# Turns
# ----------------------------------------------------------------------------
def turn_length(radius, transition=TRANSITION, angle=TURN_ANGLE):
    """The lengths [m] of a turn's entry, arc and exit."""
    arc = radius * angle - transition
    if arc < 0.0:
        raise ValueError("the transitions alone turn the route by more than the angle")
    return transition, arc, transition


def route_curvature(s, radius, side=1.0, transition=TRANSITION, angle=TURN_ANGLE):
    """The curvature [1/m] of the route at the distances s [m] from the
    start of a turn's entry, and its rate along the route [1/m^2]: straight
    before the entry; over the entry the curvature rises along a half-cosine
    of the distance to 1/radius, the arc keeps it, the exit lowers it along a
    half-cosine, and the route runs straight after; side 1 for a turn to the
    left, -1 to the right."""
    s = np.asarray(s, dtype=float)
    entry, arc, _ = turn_length(radius, transition, angle)
    k, dk = np.zeros_like(s), np.zeros_like(s)
    rise = (s >= 0.0) & (s < entry)
    k[rise] = (1.0 - np.cos(math.pi * s[rise] / entry)) / (2.0 * radius)
    dk[rise] = math.pi / (2.0 * entry * radius) * np.sin(math.pi * s[rise] / entry)
    on = (s >= entry) & (s < entry + arc)
    k[on] = 1.0 / radius
    fall = (s >= entry + arc) & (s < 2.0 * entry + arc)
    u = s[fall] - entry - arc
    k[fall] = (1.0 + np.cos(math.pi * u / entry)) / (2.0 * radius)
    dk[fall] = -math.pi / (2.0 * entry * radius) * np.sin(math.pi * u / entry)
    return side * k, side * dk


def braking_run(speed, b, rise=RISE, dt=TURN_DT):
    """A braking from speed to rest at b, its deceleration rising along a
    half-cosine over rise and fading over rise at standstill (slowing): the
    times [s], speeds [m/s], distances [m] and decelerations [m/s^2]."""
    peak = min(b, speed / rise)
    t1 = speed / peak
    t = np.arange(0.0, t1 + rise + 0.5 * dt, dt)
    dec = np.where(t < rise, 0.5 * peak * (1.0 - np.cos(math.pi * t / rise)),
                   np.where(t < t1, peak,
                            0.5 * peak * (1.0 + np.cos(math.pi * np.clip(t - t1, 0.0, rise) / rise))))
    lost = np.concatenate([[0.0], np.cumsum(0.5 * (dec[1:] + dec[:-1]) * dt)])
    v = np.maximum(speed - lost, 0.0)
    dist = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * dt)])
    return t, v, dist, dec


def drive_run(speed, a, rise=RISE, dt=TURN_DT):
    """A drive off from rest to speed at a, the mirror of braking_run: the
    times [s], speeds [m/s], distances [m] and accelerations [m/s^2]."""
    t, vb, sb, db = braking_run(speed, a, rise, dt)
    return t, speed - vb, speed * t - sb, db


def turn_starts(radius, speed, braking=None, drive=None):
    """The starts [m] from the entry's start of the brakings or drives off
    that the test evaluates in a turn of the radius, TURN_STEP apart: a
    braking from a stop's distance before the entry to the exit's end, a
    drive off from rest anywhere from the entry's start to the exit's end;
    and the run itself (braking_run or drive_run)."""
    entry, arc, _ = turn_length(radius)
    run = braking_run(speed, braking) if braking is not None else drive_run(speed, drive)
    first = -run[2][-1] if braking is not None else 0.0
    return np.arange(first, 2.0 * entry + arc + 0.5 * TURN_STEP, TURN_STEP), run


def turn_states(radius, speed, braking=None, side=1.0, drive=None):
    """States of the base's middle on a turn of the radius: at the constant
    speed through the entry, the arc and the exit, TURN_POINTS points along
    each transition; or, with braking, braking from speed to rest at that
    deceleration (braking_run) from every start of turn_starts; or, with
    drive, driving off from rest to speed at that acceleration (drive_run)
    from every start of turn_starts. Returns arrays of the speed, the
    acceleration along the route, the curvature and its rate."""
    entry, arc, _ = turn_length(radius)
    if braking is None and drive is None:
        s = np.concatenate([np.linspace(0.0, entry, TURN_POINTS), [entry + 0.5 * arc],
                            entry + arc + np.linspace(0.0, entry, TURN_POINTS)])
        v, ax = np.full(len(s), float(speed)), np.zeros(len(s))
    else:
        starts, (_, vr, sr, ar) = turn_starts(radius, speed, braking, drive)
        s = (starts[:, None] + sr[None, :]).ravel()
        v = np.tile(vr, len(starts))
        ax = np.tile(-ar if braking is not None else ar, len(starts))
    k, dk = route_curvature(s, radius, side)
    return v, ax, k, dk


def axle_grip(q, px, py):
    """The largest share of an axle's load that the ground's friction must
    supply, for the ground's wrench q[..., :6] (model.motion_basis) and its
    zero-moment point: the axles' loads as on a beam between them, the
    braking or driving force shared in proportion to them, at the lateral
    place of the zero-moment point, and the lateral forces of the two axles
    giving the lateral force and the moment that turns the machine."""
    fx, fy, fz, mz = q[..., 0], q[..., 1], q[..., 2], q[..., 5]
    front = 0.5 * fz * (1.0 + px / model.AXLE)
    rear = 0.5 * fz * (1.0 - px / model.AXLE)
    split = (mz + py * fx) / model.AXLE
    need_f = np.hypot(fx * front / fz, 0.5 * (fy + split)) / np.maximum(front, 1e-9)
    need_r = np.hypot(fx * rear / fz, 0.5 * (fy - split)) / np.maximum(rear, 1e-9)
    return np.maximum(need_f, need_r)


def turn_slacks(pose, label, speed, ax, k, dk, level=FLOOR, kappa=KAPPA):
    """The rigid machine's least slack over the payloads of the label at
    each of the given states of the base's middle (turn_states): the speed,
    the acceleration along the route, the route's curvature and its rate;
    the base's yaw rate is speed * k and its yaw acceleration ax * k +
    speed^2 * dk. Returns a dict of arrays, state by state, of the slack of
    tipping (the zero-moment point's distance inside less the allowance,
    less level [m]), the slew, the lift and the tilt (1 less the torque's
    share of its limit, the part that the motion adds counted kappa times)
    and the ground (the label's friction less the share that an axle
    needs)."""
    speed, ax, k, dk = (np.asarray(x, dtype=float) for x in (speed, ax, k, dk))
    ay, w2, alpha = speed * speed * k, (speed * k) ** 2, ax * k + speed * speed * dk
    out = {}
    for p in payloads_of(label):
        b = model.motion_basis(p, pose)
        q = (b["q0"] + np.multiply.outer(ax, b["ax"]) + np.multiply.outer(ay, b["ay"])
             + np.multiply.outer(alpha, b["alpha"]) + np.multiply.outer(w2, b["w2"]))
        px, py, _ = model.wrench_zmp(q)
        inside = np.minimum(np.minimum(model.AXLE - px, model.AXLE + px),
                            np.minimum(model.HALF_TRACK - py, model.HALF_TRACK + py))
        held = b["q0"][6:9] + kappa * (q[..., 6:9] - b["q0"][6:9])
        slack = {"tipping": inside - ALLOWANCE - level, "ground": label.friction - axle_grip(q, px, py)}
        for j, name in enumerate(JOINTS):
            slack[name] = 1.0 - np.abs(held[..., j]) / ARM_LIMITS[j]
        for name, x in slack.items():
            out[name] = x if name not in out else np.minimum(out[name], x)
    return out


def turn_slack(pose, label, speed, ax, k, dk, level=FLOOR, kappa=KAPPA):
    """The least of each slack of turn_slacks over the states."""
    return {name: float(np.min(x))
            for name, x in turn_slacks(pose, label, speed, ax, k, dk, level, kappa).items()}


def motion_margin(pose, payload, ax, ay, w, alpha):
    """The rigid machine's zero-moment point's distance inside the rectangle
    [m], without the allowance, its base's middle accelerating at (ax, ay)
    and the base turning at the yaw rate w with the yaw acceleration alpha
    (arrays), from model.motion_basis."""
    b = model.motion_basis(payload, pose)
    ax, ay, w, alpha = (np.asarray(x, dtype=float) for x in (ax, ay, w, alpha))
    q = (b["q0"] + np.multiply.outer(ax, b["ax"]) + np.multiply.outer(ay, b["ay"])
         + np.multiply.outer(alpha, b["alpha"]) + np.multiply.outer(w * w, b["w2"]))
    px, py, _ = model.wrench_zmp(q)
    return np.minimum(model.AXLE - np.abs(px), model.HALF_TRACK - np.abs(py))


def sides_of(pose):
    """The sides to which a turn must be checked: one, by symmetry, with the
    arm pointing forward."""
    return (1.0,) if pose[0] == 0.0 else (1.0, -1.0)


def turn_holds(pose, radius, speed, label=DECLARED, level=FLOOR, braking=TURN_BRAKING,
               sides=None, drive=TURN_DRIVE):
    """Whether a plan may take the turn of the radius at the speed in pose
    under the label at the level: through the turn at that speed, braking in
    it at braking from that speed and from three quarters, a half and a
    quarter of it, and driving off in it from rest at drive to the same
    speeds, to either side (or to the sides given, 1 left and -1 right); and
    the constraint that binds most."""
    worst = (math.inf, None)
    for side in sides or sides_of(pose):
        runs = [turn_states(radius, speed, side=side)]
        runs += [turn_states(radius, f * speed, braking, side) for f in (1.0, 0.75, 0.5, 0.25)]
        runs += [turn_states(radius, f * speed, side=side, drive=drive)
                 for f in (1.0, 0.75, 0.5, 0.25)]
        for states in runs:
            for name, x in turn_slack(pose, label, *states, level=level).items():
                if x < worst[0]:
                    worst = (x, name)
    return worst[0] >= 0.0, worst[1]


def turn_worst_start(pose, radius, speed, label=DECLARED, side=1.0, braking=None, drive=None,
                     constraint="tipping", level=FLOOR):
    """The start [m] from the entry's start of the braking at braking, or
    the drive off at drive, from the speed in the turn of the radius after
    which the rigid machine's slack against the constraint (a name of
    turn_slacks) is least (turn_starts), and that slack."""
    starts, run = turn_starts(radius, speed, braking, drive)
    states = turn_states(radius, speed, braking, side, drive)
    slack = turn_slacks(pose, label, *states, level=level)[constraint]
    per_start = slack.reshape(len(starts), len(run[0])).min(axis=1)
    i = int(np.argmin(per_start))
    return float(starts[i]), float(per_start[i])


def turn_speed(pose, radius, label=DECLARED, level=FLOOR, braking=TURN_BRAKING, grid=0.01,
               sides=None):
    """The largest speed [m/s] on a grid of 0.01 m/s at which a plan may take
    the turn of the radius (turn_holds), within the top speed, and what binds
    at the next speed of the grid ('top speed' if the top speed holds; 0 and
    the constraint if no speed does). The search assumes that a speed holds
    if a higher one does; labels_demand.py checks the lower speeds."""
    ok, _ = turn_holds(pose, radius, SPEED, label, level, braking, sides)
    if ok:
        return SPEED, "top speed"
    lo, hi = 0, int(round(SPEED / grid))
    ok, why = turn_holds(pose, radius, grid, label, level, braking, sides)
    if not ok:
        return 0.0, why
    lo = 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        ok, w = turn_holds(pose, radius, mid * grid, label, level, braking, sides)
        if ok:
            lo = mid
        else:
            hi, why = mid, w
    _, why = turn_holds(pose, radius, hi * grid, label, level, braking, sides)
    return round(lo * grid, 2), why


def lift_path(steps=STEPS, start="carry", end="reach"):
    """The poses of a lift from one pose to another, linear in the joints'
    angles, with the share of the lift done."""
    q0, q1 = np.array(model.POSES[start]), np.array(model.POSES[end])
    return [(k / (steps - 1), tuple(q0 + (q1 - q0) * k / (steps - 1))) for k in range(steps)]


def slowing(speed, b, until=0.0, rise=RISE):
    """The distance [m] and the time [s] in which the base, cruising at
    speed, slows to the speed until: its deceleration rises from zero along a
    half-cosine over rise seconds, stays at b for as long as the slowing
    needs, and falls back to zero along a half-cosine over rise seconds as
    the speed reaches until; a slowing too small for that holds a lower peak
    between the two. The profile is symmetric, so the base moves at the mean
    of the two speeds on average, for dv / peak + rise seconds."""
    dv = speed - until
    if dv <= 0.0:
        return 0.0, 0.0
    if rise <= 0.0:
        t = dv / b
    else:
        t = dv / min(b, dv / rise) + rise
    return 0.5 * (speed + until) * t, t


def stop_distance(speed, b, rise=RISE):
    """The distance [m] in which the recovery stops the base from speed,
    braking at up to b with its deceleration rising and fading over rise
    seconds (slowing)."""
    return slowing(speed, b, 0.0, rise)[0]


def level_up(x, grid=GRID):
    """x rounded up to the levels' grid."""
    return math.ceil(round(x / grid, 9)) * grid


def level_down(x, grid=GRID):
    """x rounded down to the levels' grid."""
    return math.floor(round(x / grid, 9)) * grid


def lift_level(b, label=DECLARED, start="carry", end="reach", steps=STEPS):
    """The least tipping reserve [m] along a lift from start to end under the
    label, the recovery braking at b with the arm held wherever the lift has
    brought it."""
    return min(tipping_reserve(q, label, ax=-b)[0] for _, q in lift_path(steps, start, end))


def lift_demand(start="carry", end="reach", label=DECLARED, floor=FLOOR, clear_floor=CLEAR_FLOOR,
                steps=STEPS):
    """The demand of a lift from start to end, with the machine at rest or
    moving, its recovery braking with the arm held wherever the lift has
    brought it. The lift consumes tipping reserve: at a deceleration b of the
    recovery, the reserve in the start pose less its least value along the
    lift (a difference, which the allowance does not change). After the
    lift, the recovery may brake at no more than b_after, the end pose's
    braking limit at the floor; the consumption is taken as its largest over
    0 <= b <= b_after, and the requirement is the floor plus it, rounded up to
    the levels' grid. The check of the assumption of reserve consumption: a
    plan in force that holds the requirement in the start pose brakes at no
    more than b_req, the start pose's braking limit at the requirement, and a
    plan that begins the lift and brakes at b_req keeps, along the lift, the
    reserve lift_level; beyond the forecast horizon, where every pair
    carries the global capability, at no more than b_req_global, keeping
    lift_level_global. With a stop line ahead, the lift also consumes
    clearance, the forward travel of the machine's front-most point; the
    clearance's requirement is the floor plus that travel, on the levels'
    grid, and clear_after what the lift leaves of it, both reserves less the
    allowance."""
    path = [q for _, q in lift_path(steps, start, end)]
    q0 = path[0]
    b_after, which_after = braking_limit(path[-1], label, floor)

    def consumption(b):
        before = tipping_reserve(q0, label, ax=-b)[0]
        return before - min(tipping_reserve(q, label, ax=-b)[0] for q in path)

    bs = np.linspace(0.0, b_after, 21)
    consumed = [consumption(float(b)) for b in bs]
    k = int(np.argmax(consumed))
    requirement = level_up(floor + consumed[k])
    b_req, which_req = braking_limit(q0, label, requirement)
    b_req_global, which_global = braking_limit(q0, GLOBAL, requirement)
    forward = max(front_point(q) for q in path) - front_point(q0)
    return dict(rest_before=tipping_reserve(q0, label)[0],
                rest_after=min(tipping_reserve(q, label)[0] for q in path),
                consumed_rest=consumed[0], consumed=consumed[k], at=float(bs[k]),
                b_before=braking_limit(q0, label, floor)[0], b_after=b_after,
                which_after=which_after, requirement=requirement, b_req=b_req,
                which_req=which_req, lift_level=lift_level(b_req, label, start, end, steps),
                b_req_global=b_req_global, which_global=which_global,
                lift_level_global=lift_level(b_req_global, GLOBAL, start, end, steps),
                forward=forward, clear_requirement=level_up(clear_floor + forward),
                clear_after=level_up(clear_floor + forward) - forward)
