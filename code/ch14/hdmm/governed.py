"""
The heavy-duty mobile manipulator as modules of the allovi core of Chapter
10 (Section 14.3).

Physical AI, Together, Chapter 14 (Section 14.3).

The state is x = (s, v, a, slew, lift, tilt): the arc length of the base's
middle along the route of route.py, its speed and its acceleration along
the route, and the arm's joints' angles. Each of the two families of
constraints has its own norm, a weighted sum of the components' sizes, in
metres:

    ||x||_t = sum_j NORM_TIP[j] |x_j|      (the margin against tipping over)
    ||x||_c = sum_j NORM_CLEAR[j] |x_j|    (the clearance at the stop line),

the weights per metre of arc length, per m/s, per m/s^2 and per radian of
each joint. The state's norm, in which the measurement and the alarm check
are measured, is the larger of the two. Each weight is at least the
constant of Lemma 14.1 (bounds.py; hdmm_governed.py checks it): no change of
the state moves the rigid machine's zero-moment point, or its front-most
point, by more than the change's size in the family's norm, within 1 m of a
state whose motion the package allows, so that a level that the test
certifies is a reserve of the tipping family in the sense of Definition 4.4,
and the clearance's floor one of the clearance family (Proposition 14.2).
The tipping norm measures every component: the acceleration, and in a turn
the speed and the arc length, move the point through the accelerations that
the parts feel. The clearance norm measures the arc length and the joints;
the speed and the acceleration enter it with the tipping norm's weights,
only so that it is a norm. In operation only the alarm check uses the norms,
and with the measurement exact (measurement_bound is 0) it raises an alarm
at any excess over the funnel, whatever the weights: no decision depends on
them.

The control package hdmm.route-arm plans the base's motion along the route
and the arm's pose; the base steers along the route, as in Section 14.2:

  * a segment is the commanded motion of one period of T = 0.6 s, the base's
    actuators' RISE: the base's acceleration along the route goes from the
    one in force, a0, to the commanded a1 along a half-cosine over the
    period, as the brakes and the drive change it; the arm is held in a
    pose, carrying or reaching, or lifts from carrying to reaching along the
    cosine profile of LIFT_TIME = 4 s of Section 14.2, which a segment
    continues from the time u0 since the lift began;
  * a plan is a list of (segment, label) pairs that start where the head of
    the plan in force ends, and a rest: the arc length and the pose in which
    its last pair comes to rest;
  * the funnel of a segment is its commanded motion, which the brakes, the
    drive and the steering realize, with the base within DS of its commanded
    arc length (its creep on its contacts) and the arm, on its servos, near
    its planned pose: held, each joint within DQ of it; during a lift, each
    joint within DQ of a pose on the lift's path (a straight line in the
    joints' angles) at most LAG_W of the path behind the planned pose and
    AHEAD_W ahead of it, so that the funnel of a segment that begins the
    lift contains the held one from which it begins; the segment in which
    the lift ends ends with the arm held, reaching (Segment.end), so the
    funnels chain as conditions (P1) and (P3) of Definition 4.7 ask. These
    bounds are ours; hdmm_governed.py measures them in its runs and in every
    plan executed unchanged, as Section 13.2 measured the Husky's;
  * the constraints are the margin against tipping over and the clearance at
    the stop line, the two families of Section 14.2. The levels carry the
    tipping reserve. A pair's is the least, over the payloads of its label,
    of the zero-moment point's distance inside the rectangle of the wheels'
    contacts less an allowance for what the rigid machine at the planned
    pose leaves out (the arm's lag and swing on its servos, the contacts'
    yield): labels.ALLOWANCE with the arm held, and LIFT_ALLOWANCE at the
    poses before a lift's end. On a straight the test takes the distance at
    the segment's accelerations at its start and end, between which the
    point moves monotonically, at the planned pose or, during a lift, at the
    lift's poses every LIFT_GRID seconds and at the ends of the funnel's
    window along the path; in a turn or its transitions, over the segment's own states
    every labels.TURN_DT, from the rigid machine's inverse dynamics, as
    Section 14.2's test takes them (turn_slack_of). The rest's level is
    the tipping reserve at rest under the global capability (Definition 4.6).
    A profile's entry n is the least of the levels of the pairs from n on
    and of the rest, rounded down to whole centimetres, or NEG_INF if a pair
    from n on fails a limit or lies below the floor. The allowances are
    taken from what the simulated machine falls short of the rigid one at
    the planned pose in the runs and in every plan executed unchanged, with
    room for the worst corner of the funnel's bounds on the arm
    (hdmm_governed.py measures both). The clearance is checked at its floor:
    a plan that rests on the last straight keeps its front-most point at the
    rest, moved by the funnel's creep and the worst corner of its bounds on
    the arm (FRONT_BOX), behind CLEAR_LIMIT, the clearance's floor and the
    allowance before the stop line. The core's levels are scalar, and they
    carry the tipping reserve, on which the lift's demand falls;
  * the limits: the speed within [0, SPEED], the acceleration within the
    drive's DRIVE and the deceleration within the brakes' BRAKES; on a
    straight, the deceleration within the ground's (the label's friction
    times g) and the arm's (labels.arm_bound), and the acceleration within
    the arm's, for each pose of the segment; in a turn or its transitions,
    the arm held, the acceleration within +-TURN_ACCEL, the speed within the
    turn's speed for the pose under the label (labels.turn_speed), and at
    each of the segment's states the arm's torques, the motion's part
    counted labels.KAPPA times, within their limits and the share of each
    axle's load that the ground must supply within the label's friction.
    Section 14.2 found the turn's speeds for plans that keep their speed
    through a turn, brake in it at TURN_BRAKING and drive off in it at
    TURN_DRIVE, both TURN_ACCEL; a plan here may also change its speed in a
    turn, within +-TURN_ACCEL and below the turn's speed, so the test also
    evaluates each segment's own states;
  * the recovery continuation brakes to rest. On a straight it brakes at the
    braking limit of the pose (during a lift, which the recovery finishes,
    of the lift's end pose with the lift's allowance) under each period's
    label at the tail's level: the floor, or a pending requirement's level.
    Before a turn that it
    cannot stop short of, it eases its deceleration to TURN_ACCEL in the
    period before it reaches the turn, and in or after a turn it brakes at
    TURN_ACCEL; its last period lands it at rest, the speed and the
    acceleration reaching zero together. A package may brake more gently
    (GENTLE). The rest segment holds the machine on its brakes;
  * the generator: a proposal is a speed [m/s], which means the commanded
    acceleration that brings the speed nearest to it after the head and one
    period more, within the head's limits, at most SPEED and, where the head
    or the next period may reach a turn, at most the turn's speed (and the
    command within +-TURN_ACCEL there). The candidates mix it with the
    fallback's command (ALPHAS), each with the recovery at the floor and, if
    a requirement is pending, at its level, and the recovery alone; when
    the task asks for the lift, every candidate begins it. The generator
    plans no approach to a turn or to the stop line: a proposal that asks
    for more than the recovery can stop from is refused there, and the
    governor keeps a plan that can;
  * after an alarm, the rest feedback brakes to rest at the global
    capability's limits (TURN_ACCEL in a turn), finishing a lift. A restart
    is offered at rest, from the measured arc length.

A label is a capability of labels.py, (payloads, friction): the payload
interval of the declaration and the least friction of the grounds under the
segment's region, the arc lengths of its wheels from its start's less
model.AXLE to its end's plus model.AXLE, widened by DS, as the predictor
knows the grounds; beyond the horizon, the global capability.

The robot adapter measures the world's state exactly: the commanded base
state with the creep, and the joints' angles, with the arm's commanded
state, which the controller knows. The world adapter hdmm.yard runs the
machine of sim.py along the route under each period's law.
"""
import math
import os
import sys
from collections import namedtuple

import mujoco
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "ch10"))
from allovi import (NEG_INF, Actuation, Bound, Candidate, Observation,         # noqa: E402
                    Profile, Quantity, Requirement, Schema, ScalarLevels, Why, register)

from . import labels as hl                                                    # noqa: E402
from . import model, route, sim                                               # noqa: E402

G = hl.G
T = hl.RISE              # [s] the governor's period: the base's actuators' rise
LIFT_TIME = 4.0          # [s] the lift from carrying to reaching (Section 14.2)
LIFT_GRID = 0.1          # [s] the test takes the lift's poses this often
SPEED, DRIVE, BRAKES = hl.SPEED, hl.DRIVE, hl.BRAKES
TURN_ACCEL = hl.TURN_BRAKING
assert hl.TURN_DRIVE == hl.TURN_BRAKING
FLOOR, GRID = hl.FLOOR, hl.GRID
LEVELS = ScalarLevels(floor=FLOOR, step=GRID, top=1.0)
ALPHAS = (1.0, 0.75, 0.5, 0.25)
HORIZON = 50             # periods
DS = 0.001               # [m] the funnel's bound on the creep (ours, measured in Section 14.3)
DQ = (0.005, 0.002, 0.002)   # [rad] the funnel's bounds on the arm held, slew, lift, tilt (ours)
LAG_W = 0.04             # the funnel during a lift: each joint within DQ of a pose of the lift's
AHEAD_W = 0.001          # path at most this much of it behind the planned pose and this much ahead
LIFT_ALLOWANCE = 0.010   # [m] the allowance at the poses before a lift's end (ours, measured)
LIFT_EXTRA = LIFT_ALLOWANCE - hl.ALLOWANCE
# The norms' weights, for (s [m], v [m/s], a [m/s^2], slew, lift, tilt [rad]), in metres per
# unit: at least the constants of Lemma 14.1 (bounds.py), which hdmm_governed.py checks
NORM_TIP = (1.5, 1.0, 0.25, 0.85, 1.0, 0.4)     # the margin against tipping over
NORM_CLEAR = (1.0, 1.0, 0.25, 4.4, 4.4, 2.0)    # the clearance at the stop line
CLEAR_LIMIT = route.S_LINE - hl.CLEAR_FLOOR - hl.ALLOWANCE
AT_REST = 1e-9
POSES = model.POSES
DECLARED = hl.DECLARED.payloads
Q_CARRY, Q_REACH = np.array(POSES["carry"]), np.array(POSES["reach"])
PATH = Q_REACH - Q_CARRY                       # the lift's path, a straight line in the angles


def norm_of(x):
    """The state's norm of x (s, v, a, slew, lift, tilt) [m]: the larger of
    the tipping norm and the clearance norm."""
    x = np.abs(np.asarray(x, dtype=float))
    return max(float(np.dot(NORM_TIP, x)), float(np.dot(NORM_CLEAR, x)))


STATE = Schema(
    "mobile manipulator on the route",
    (Quantity("s", "m", "route"),          # arc length of the base's middle
     Quantity("v", "m/s", "base x"),       # speed along the route
     Quantity("a", "m/s^2", "base x"),     # acceleration along the route
     Quantity("slew", "rad", "z"),
     Quantity("lift", "rad", "-y"),
     Quantity("tilt", "rad", "-y")),
    norm="the larger of the tipping norm {} and the clearance norm {}".format(*(
        " + ".join(f"{w:g} {u}|{q}|" for w, u, q in zip(
            weights, ("", "s ", "s^2 ", "m/rad ", "m/rad ", "m/rad "),
            ("s", "v", "a", "slew", "lift", "tilt"))) for weights in (NORM_TIP, NORM_CLEAR))))

ACTUATION = Actuation(
    inputs=(Quantity("a_c", "m/s^2", "base x"), Quantity("slew_c", "rad", "z"),
            Quantity("lift_c", "rad", "-y"), Quantity("tilt_c", "rad", "-y")),
    bounds=(Bound("a_c", low=-BRAKES, high=DRIVE),),
    hold="the acceleration along the route follows each period's command along a "
         "half-cosine over the period; the servos follow their targets",
    meaning="the base's acceleration along the route, realized by its brakes and drive, "
            "the steering following the route; the arm's servos' targets, offset against "
            "gravity for the believed payload")


# ----------------------------------------------------------------------------
# Segments, plans and states
# ----------------------------------------------------------------------------
State = namedtuple("State", "s v a arm u")
"""A commanded state at a period boundary: the arc length, the speed and the
acceleration in force, and the arm's state: 'carry', 'reach', or 'lift' with
u the time since the lift began. It is also the initial set, the funnel's
centre."""

Rest = namedtuple("Rest", "s arm")


def lift_weight(u):
    """How much of the lift's path the planned pose has covered u seconds
    after the lift began, 0 carrying to 1 reaching."""
    return 0.5 * (1.0 - math.cos(math.pi * min(max(u, 0.0), LIFT_TIME) / LIFT_TIME))


def path_pose(w):
    """The pose that has covered w of the lift's path."""
    return tuple(round(float(x), 12) for x in Q_CARRY + PATH * w)


def lift_pose(u):
    """The planned pose u seconds after the lift began."""
    return path_pose(lift_weight(u))


def front_box(name):
    """How far forward the worst corner of the funnel's bounds on the arm
    held (DQ) moves the front-most point of the pose [m]."""
    q, f0, worst = np.array(POSES[name]), hl.front_point(POSES[name]), 0.0
    for signs in ((a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)):
        worst = max(worst, hl.front_point(tuple(q + np.array(DQ) * np.array(signs))) - f0)
    return worst


FRONT_BOX = {name: front_box(name) for name in ("carry", "reach")}


class Segment(namedtuple("Segment", "s0 v0 a0 a1 arm u0")):
    """The commanded motion of one period (module docstring)."""
    __slots__ = ()

    def accel(self, tau):
        return self.a0 + (self.a1 - self.a0) * 0.5 * (1.0 - math.cos(math.pi * tau / T))

    def speed(self, tau):
        return (self.v0 + self.a0 * tau
                + (self.a1 - self.a0) * (0.5 * tau - T / (2.0 * math.pi)
                                         * math.sin(math.pi * tau / T)))

    def arc(self, tau):
        return (self.s0 + self.v0 * tau + 0.5 * self.a0 * tau * tau
                + (self.a1 - self.a0) * (0.25 * tau * tau - T * T / (2.0 * math.pi ** 2)
                                         * (1.0 - math.cos(math.pi * tau / T))))

    def end(self):
        v = self.speed(T)
        if abs(v) < 1e-9:
            v = 0.0
        arm, u = self.arm, self.u0
        if arm == "lift":
            u = u + T
            if u >= LIFT_TIME - 1e-9:
                arm, u = "reach", None
        return State(self.arc(T), v, self.a1, arm, u)

    def speeds(self):
        """The least and the largest speed over the segment."""
        vs = [self.v0, self.speed(T)]
        if self.a0 * self.a1 < 0.0:          # the acceleration passes through zero
            w = -self.a0 / (self.a1 - self.a0)
            vs.append(self.speed(T / math.pi * math.acos(1.0 - 2.0 * w)))
        return min(vs), max(vs)

    def pose(self, tau):
        if self.arm != "lift":
            return POSES[self.arm]
        return lift_pose(self.u0 + tau)

    def test_poses(self):
        """The poses that the test takes, each with what it takes off beyond
        labels.ALLOWANCE: the held pose, 0; during a lift, the lift's poses
        every LIFT_GRID seconds over the segment and the ends of the funnel's
        window along the lift's path, from LAG_W behind the segment's start to
        AHEAD_W beyond its end, LIFT_EXTRA at the poses before the lift's
        end."""
        if self.arm != "lift":
            return [(POSES[self.arm], 0.0)]
        u1 = min(self.u0 + T, LIFT_TIME)
        n = int(round((u1 - self.u0) / LIFT_GRID))
        ws = {lift_weight(self.u0 + i * LIFT_GRID) for i in range(n + 1)}
        ws.add(max(0.0, lift_weight(self.u0) - LAG_W))
        ws.add(min(1.0, lift_weight(u1) + AHEAD_W))
        return [(path_pose(w), LIFT_EXTRA if w < 1.0 - 1e-12 else 0.0) for w in sorted(ws)]

    def held(self):
        """The pose name whose limits bind the recovery: the lift's end pose
        during a lift."""
        return "reach" if self.arm == "lift" else self.arm


class Plan:
    """(segment, label) pairs from period j0 on, and the rest."""

    def __init__(self, j0, pairs, rest):
        self.j0, self.pairs, self.rest = j0, list(pairs), rest

    def shift(self):
        if not self.pairs:
            return self
        return Plan(self.j0 + 1, self.pairs[1:], self.rest)

    def head(self):
        if self.pairs:
            return self.pairs[0][0]
        return Segment(self.rest.s, 0.0, 0.0, 0.0, self.rest.arm, None)


def region(seg):
    """The arc lengths that the wheels of a machine in the segment's funnel
    can occupy (module docstring)."""
    s1 = seg.arc(T)
    return min(seg.s0, s1) - model.AXLE - DS, max(seg.s0, s1) + model.AXLE + DS


def level_of(reserve):
    """The largest level of the grid not above the reserve, or NEG_INF below
    the floor, the reserve rounded to 1e-9 m first."""
    x = round(reserve, 9)
    if x < FLOOR:
        return NEG_INF
    return min(LEVELS.top, round(hl.level_down(x), 9))


# ----------------------------------------------------------------------------
# What the test needs of the labels, computed once and kept
# ----------------------------------------------------------------------------
class Limits:
    """The rigid machine's quantities that the test and the recovery use, for
    each pose, payload and label, computed on first use and kept: the
    centre of mass, the arm's bounds on the base's acceleration, the braking
    limits at a level, the turns' speeds and their tipping reserves, and the
    rest's level."""

    def __init__(self, radius=route.RADIUS):
        self.radius = radius
        self._com, self._arm, self._braking, self._turn, self._turn_res = {}, {}, {}, {}, {}
        self._fixed = {}

    def com(self, pose, payload):
        key = (pose, payload)
        if key not in self._com:
            self._com[key] = model.centre_of_mass(*pose, payload)[1]
        return self._com[key]

    def reserve(self, pose, label, a):
        """The tipping reserve on a straight at the acceleration a along the
        route: the least over the label's payloads (labels.tipping_reserve)."""
        best = math.inf
        for p in hl.payloads_of(label):
            c = self.com(pose, p)
            px, py = c[0] - c[2] * a / G, c[1]
            best = min(best, model.AXLE - abs(px), model.HALF_TRACK - abs(py))
        return best - hl.ALLOWANCE

    def _torques(self, payload, pose, ax):
        """labels.torques (model.holding_torques) without rebuilding the
        model each time."""
        if payload not in self._fixed:
            m = model.build(payload=payload, pose=None, fixed_base=True).compile()
            self._fixed[payload] = (m, mujoco.MjData(m))
        m, d = self._fixed[payload]
        m.opt.gravity[:] = [-ax, 0.0, -G]
        d.qpos[:3] = pose
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        return d.qfrc_bias[:3].copy()

    def arm(self, pose, label):
        """The arm's bounds on the base's deceleration and acceleration along
        the route in the pose under the label (labels.arm_bound)."""
        key = (pose, label)
        if key not in self._arm:
            bounds = []
            for sign in (-1.0, 1.0):
                best = math.inf
                for p in hl.payloads_of(label):
                    tg = self._torques(p, pose, 0.0)
                    t1 = self._torques(p, pose, sign)
                    for j, limit in enumerate(hl.ARM_LIMITS):
                        slope = hl.KAPPA * (t1[j] - tg[j])
                        if abs(tg[j]) > limit * (1.0 + 1e-9):
                            best = 0.0
                        elif slope > 1e-9:
                            best = min(best, (limit - tg[j]) / slope)
                        elif slope < -1e-9:
                            best = min(best, (-limit - tg[j]) / slope)
                bounds.append(best)
            self._arm[key] = tuple(bounds)
        return self._arm[key]

    def braking(self, name, label, level):
        """The braking limit of the pose under the label at the level
        (labels.braking_limit); 0 if the pose lacks the level at rest."""
        key = (name, label, round(level, 9))
        if key not in self._braking:
            self._braking[key] = hl.braking_limit(POSES[name], label, level)[0]
        return self._braking[key]

    def turn_speed(self, name, label):
        key = (name, label)
        if key not in self._turn:
            self._turn[key] = hl.turn_speed(POSES[name], self.radius, label)
        return self._turn[key][0]

    def turn_binds(self, name, label):
        self.turn_speed(name, label)
        return self._turn[(name, label)][1]

    def turn_reserve(self, name, label):
        key = (name, label)
        if key not in self._turn_res:
            self._turn_res[key] = turn_reserve(POSES[name], label,
                                               self.turn_speed(name, label), self.radius)
        return self._turn_res[key]

    @staticmethod
    def rest(name):
        return level_of(hl.tipping_reserve(POSES[name], hl.GLOBAL)[0])


def turn_reserve(pose, label, speed, radius=route.RADIUS):
    """The least tipping reserve [m] over the states of a turn of the radius
    that Section 14.2's test evaluates at the speed (labels.turn_holds): at
    that speed through the turn, braking in it at TURN_BRAKING from that
    speed and from three quarters, a half and a quarter of it, and driving
    off in it at TURN_DRIVE to the same speeds, to either side."""
    worst = math.inf
    for side in hl.sides_of(pose):
        runs = [hl.turn_states(radius, speed, side=side)]
        runs += [hl.turn_states(radius, f * speed, hl.TURN_BRAKING, side)
                 for f in (1.0, 0.75, 0.5, 0.25)]
        runs += [hl.turn_states(radius, f * speed, side=side, drive=hl.TURN_DRIVE)
                 for f in (1.0, 0.75, 0.5, 0.25)]
        for states in runs:
            worst = min(worst, hl.turn_slack(pose, label, *states, level=0.0)["tipping"])
    return worst


LIMITS = Limits()


def pair_check(seg, label, lim=LIMITS):
    """(reserve, failure): the pair's tipping reserve [m], before rounding,
    and None; or (None, what keeps the test from certifying the pair)."""
    lo_v, hi_v = seg.speeds()
    if lo_v < -1e-9:
        return None, "the speed below zero"
    if hi_v > SPEED + 1e-9:
        return None, "beyond the top speed"
    for a in (seg.a0, seg.a1):
        if a > DRIVE + 1e-9:
            return None, "beyond the drive"
        if -a > BRAKES + 1e-9:
            return None, "beyond the brakes"
    turn = route.turn_on(seg.s0, seg.arc(T))
    if turn is not None:
        if seg.arm == "lift":
            return None, f"a lift in the {turn.name} turn"
        if max(abs(seg.a0), abs(seg.a1)) > TURN_ACCEL + 1e-9:
            return None, f"an acceleration beyond {TURN_ACCEL:g} m/s^2 in the {turn.name} turn"
        limit = lim.turn_speed(seg.arm, label)
        if hi_v > limit + 1e-9:
            return None, f"beyond the {turn.name} turn's speed, {limit:.2f} m/s"
        slack = turn_slack_of(seg, label)
        for name, what in (("slew", "the slew's torque"), ("lift", "the lift's torque"),
                           ("tilt", "the tilt's torque"), ("ground", "the ground's friction")):
            if slack[name] < 0.0:
                return None, f"beyond {what} in the {turn.name} turn"
        return slack["tipping"], None
    worst = math.inf
    for q, extra in seg.test_poses():
        dec, acc = lim.arm(q, label)
        for a in (seg.a0, seg.a1):
            if a < 0.0:
                if -a > label.friction * G + 1e-9:
                    return None, "beyond the ground's friction"
                if -a > dec + 1e-9:
                    return None, "beyond the arm's torques"
            elif a > acc + 1e-9:
                return None, "beyond the arm's torques"
            worst = min(worst, lim.reserve(q, label, a) - extra)
    return worst, None


STATE_STEPS = int(round(T / hl.TURN_DT))     # the states of a segment in a turn, every TURN_DT


def turn_slack_of(seg, label):
    """The rigid machine's least slacks (labels.turn_slacks) over the
    segment's own states, every labels.TURN_DT, in its held pose under the
    label: the tipping reserve [m] (the allowance taken off, no level), and
    the slacks of the slew, the lift, the tilt and the ground."""
    tau = np.linspace(0.0, T, STATE_STEPS + 1)
    w = 0.5 * (1.0 - np.cos(np.pi * tau / T))
    a = seg.a0 + (seg.a1 - seg.a0) * w
    v = (seg.v0 + seg.a0 * tau
         + (seg.a1 - seg.a0) * (0.5 * tau - T / (2.0 * np.pi) * np.sin(np.pi * tau / T)))
    s = (seg.s0 + seg.v0 * tau + 0.5 * seg.a0 * tau ** 2
         + (seg.a1 - seg.a0) * (0.25 * tau ** 2 - T * T / (2.0 * np.pi ** 2)
                                * (1.0 - np.cos(np.pi * tau / T))))
    k, dk = route.curvature(s)
    return hl.turn_slack(POSES[seg.arm], label, v, a, k, dk, level=0.0)


def rest_problem(rest):
    """What keeps the rest from being certified, or None: the clearance at
    the stop line, on the last straight, of the front-most point at rest
    moved by the funnel's creep and the worst corner of its bounds on the
    arm."""
    front = rest.s + DS + hl.front_point(POSES[rest.arm]) + FRONT_BOX[rest.arm]
    if rest.s > route.TURNS[-1].end and front > CLEAR_LIMIT:
        return "the front-most point at rest beyond the clearance's floor"
    return None


def stop_point(s, v, a, b):
    """The arc length at which a stop from (s, v, a) braking at b with the
    landing of the recovery comes to rest (no turns, no labels)."""
    for _ in range(1000):
        if v <= AT_REST and abs(a) <= 1e-12:
            return s
        seg = Segment(s, v, a, -min(b, max(0.0, (v + a * T / 2.0) / T)), "carry", None)
        s, v, a, _, _ = seg.end()
    return math.inf


# ----------------------------------------------------------------------------
# The capability space
# ----------------------------------------------------------------------------
class Capability:
    """Labels of labels.py, (payloads, friction). A label c is at least as
    informative as d if its payload interval lies within d's and its
    friction is at least d's."""
    NAME = "hdmm.payload-friction"
    global_capability = hl.GLOBAL

    def informative(self, c, d):
        return (d.payloads[0] <= c.payloads[0] and c.payloads[1] <= d.payloads[1]
                and c.friction >= d.friction)

    def labelling(self, capability):
        return lambda j, segment: capability


class Law:
    """The feedback law of one period: the commanded motion of a segment.
    Called with the time within the period and the state, it returns the
    commanded acceleration and the servos' targets (ACTUATION)."""

    def __init__(self, segment, kind="segment"):
        self.segment, self.kind = segment, kind

    def __call__(self, tau, state):
        return (self.segment.accel(tau),) + tuple(self.segment.pose(tau))


# ----------------------------------------------------------------------------
# The control package
# ----------------------------------------------------------------------------
@register
class RouteArm:
    """Segments of the base's acceleration along the route and of the arm's
    pose, the generator, the test, the recovery continuation, the rest
    segment and the rest feedback (module docstring)."""
    ROLE, NAME, CONTRACT = "control", "hdmm.route-arm", "1.0"
    STATE, ACTUATION = STATE, ACTUATION
    GENTLE = math.inf     # [m/s^2] the most at which the recovery brakes on a straight

    def __init__(self, limits=LIMITS):
        self.capabilities, self.levels, self.lim = Capability(), LEVELS, limits

    # --- plans and the test ------------------------------------------------
    def initial_plan(self, measured, bound):
        if bound > 1e-12:
            raise ValueError("this package's funnels assume an exact measurement")
        if abs(measured.v) > AT_REST or abs(measured.a) > 1e-12 or measured.arm == "lift":
            raise ValueError("this package certifies an initial plan only at rest, the arm held")
        rest = Rest(measured.s, measured.arm)
        return Plan(0, [], rest), State(measured.s, 0.0, 0.0, measured.arm, None)

    def initial_set(self, plan):
        return plan.head().end()

    def head(self, plan):
        """The pair that the plan executes in its first period (for the
        checks of Section 10.3)."""
        if plan.pairs:
            return plan.pairs[0]
        return plan.head(), hl.GLOBAL

    def profile(self, plan, initial):
        """Condition (P1): the plan starts at the initial set's commanded
        state. Then, for each n, the least level of the pairs from n on and
        of the rest, or NEG_INF if a pair from n on, or the rest, is not
        certified at the floor."""
        if not _starts_at(plan, initial):
            return Profile.uncertified()
        rest = NEG_INF if rest_problem(plan.rest) else self.lim.rest(plan.rest.arm)
        entries, running = [], rest
        for seg, label in reversed(plan.pairs):
            r, failure = pair_check(seg, label, self.lim)
            level = NEG_INF if failure else level_of(r)
            running = NEG_INF if (running == NEG_INF or level == NEG_INF) else min(running, level)
            entries.append(running)
        entries.reverse()
        return Profile(entries + [rest])

    def explain(self, plan, initial):
        if not _starts_at(plan, initial):
            return [Why("the plan does not start at the initial set", "pair")]
        problem = rest_problem(plan.rest)
        rest = self.lim.rest(plan.rest.arm)
        last = (Why(problem, "pair") if problem else
                Why(f"the rest at {plan.rest.s:.3f} m {plan.rest.arm}, level {rest:.2f} m"))
        out, first = [], (None if not problem else (len(plan.pairs), problem))
        for i in range(len(plan.pairs) - 1, -1, -1):
            r, failure = pair_check(*plan.pairs[i], self.lim)
            if failure is None and level_of(r) == NEG_INF:
                failure = f"a tipping reserve of {r:.4f} m, below the floor"
            if failure is not None:
                first = (i, failure)
            out.append(Why(f"pair {first[0]}: {first[1]}", "pair") if first else
                       Why(f"pair {i}: tipping reserve {r:.4f} m"))
        out.reverse()
        return out + [last]

    # --- the generator -------------------------------------------------------
    def generate(self, step, plan, initial):
        st, labelling, k = initial, step.labelling, step.k
        levels = [FLOOR]
        if step.requirement.level > FLOOR + 1e-9:
            levels.append(step.requirement.level)
        begin = step.operation == "lift" and st.arm == "carry"
        if step.proposal is None:
            return self._recovery_only(labelling, k, st, levels, begin)
        fallback = plan.shift()
        a_fb = fallback.head().a1 if fallback.pairs else 0.0
        heads = {}
        for alpha in ALPHAS:
            a1 = self.command(step.proposal, alpha, a_fb, labelling, k + 1, st, begin)
            key = round(a1, 12)
            if key not in heads:
                heads[key] = (a1, alpha)
        arm, u = ("lift", 0.0) if begin else (st.arm, st.u)
        out = []
        for a1, alpha in heads.values():
            head = Segment(st.s, st.v, st.a, a1, arm, u)
            for level in levels:
                tail, rest = self.recovery(labelling, k + 2, head.end(), level)
                out.append(Candidate(Plan(k + 1, [(head, labelling(k + 1, head))] + tail, rest),
                                     alpha))
        return out + self._recovery_only(labelling, k, st, levels, begin)

    def command(self, proposal, alpha, a_fb, labelling, j, st, begin=False):
        """The head's commanded acceleration: the proposal's meaning mixed
        with the fallback's command by alpha, within the head's limits."""
        arm, u = ("lift", 0.0) if begin else (st.arm, st.u)
        name = "reach" if arm == "lift" else arm
        target = min(SPEED, max(0.0, float(proposal)))
        coast = Segment(st.s, st.v, st.a, 0.0, arm, u)
        b = self.braking(arm, labelling(j, coast), FLOOR)
        far = Segment(st.s, st.v, st.a, DRIVE, arm, u)
        if arm != "lift" and route.turn_on(st.s, _next_reach(far)) is not None:
            target = min(target, self.lim.turn_speed(name, labelling(j, far)))
        a_q = (target - st.v - st.a * T / 2.0) / T
        a1 = min(DRIVE, max(-b, alpha * a_q + (1.0 - alpha) * a_fb))
        if route.turn_on(st.s, _next_reach(Segment(st.s, st.v, st.a, a1, arm, u))) is not None:
            a1 = min(TURN_ACCEL, max(-TURN_ACCEL, a1))   # this period or the next reaches a turn
        return 0.0 if abs(a1) < 1e-9 else a1

    def _recovery_only(self, labelling, k, st, levels, begin=False):
        out = []
        for level in levels:
            s0 = st._replace(arm="lift", u=0.0) if begin else st
            tail, rest = self.recovery(labelling, k + 1, s0, level)
            if not tail:
                hold = Segment(st.s, 0.0, 0.0, 0.0, st.arm, None)
                tail = [(hold, labelling(k + 1, hold))]
            out.append(Candidate(Plan(k + 1, tail, rest), 0.0))
        return out

    def braking(self, arm, label, level):
        """The braking limit on a straight of a pair in arm under the label
        at the level: during a lift, that of the lift's end pose, the least
        along the lift's path, with the lift's allowance."""
        if arm == "lift":
            return self.lim.braking("reach", label, level + LIFT_EXTRA)
        return self.lim.braking(arm, label, level)

    def recovery(self, labelling, j, st, level):
        """The recovery continuation from the commanded state st from period
        j on (module docstring): its pairs and its rest."""
        s, v, a, arm, u = st
        pairs, turned = [], False
        for _ in range(500):
            if v <= AT_REST and abs(a) <= 1e-12 and arm != "lift":
                return pairs, Rest(s, arm)
            coast = Segment(s, v, a, 0.0, arm, u)
            if route.turn_on(s, coast.arc(T)) is not None:
                turned = True
            if turned:
                b = TURN_ACCEL
            else:
                b = min(self.braking(arm, labelling(j, coast), level), self.GENTLE)
                turn = route.next_turn(s)
                if turn is not None and b > TURN_ACCEL and stop_point(s, v, a, b) > turn.start:
                    first = Segment(s, v, a, -min(b, max(0.0, (v + a * T / 2.0) / T)), arm, u)
                    e = first.end()
                    second = Segment(e.s, e.v, e.a,
                                     -min(b, max(0.0, (e.v + e.a * T / 2.0) / T)), arm, u)
                    if second.arc(T) > turn.start:
                        b = TURN_ACCEL           # ease now: the next period reaches the turn
            a1 = 0.0 if v <= AT_REST else -min(b, max(0.0, (v + a * T / 2.0) / T))
            seg = Segment(s, v, a, a1, arm, u)
            pairs.append((seg, labelling(j, seg)))
            s, v, a, arm, u = seg.end()
            j += 1
        raise RuntimeError("the recovery does not come to rest")

    # --- execution and the alarm check ---------------------------------------
    def law(self, plan):
        return Law(plan.head())

    def distance(self, measured, initial):
        """In the state's norm (norm_of): how far the measurement lies outside
        the funnel at the initial set (module docstring), or, during a lift,
        a bound on that from above which is zero exactly on the funnel. Held,
        the funnel's set is the box of DS, the commanded speed and
        acceleration and DQ about the commanded state and the planned pose,
        and the distance is the norm of each component's excess over its
        bound. During a lift, it is that box about a pose of the window along
        the lift's path, from LAG_W behind the planned pose to AHEAD_W ahead
        of it, within the path. The poses of the window about which the lift
        and the tilt lie within DQ form an interval; if it is not empty, its
        middle is taken, and otherwise the best of the window's ends and the
        poses at which a joint's excess begins or ends."""
        q = np.array((measured.slew, measured.lift, measured.tilt))
        base = np.array((max(0.0, abs(measured.s - initial.s) - DS),
                         abs(measured.v - initial.v), abs(measured.a - initial.a)))

        def size(centre):
            excess = np.maximum(0.0, np.abs(q - centre) - np.array(DQ))
            return norm_of(np.concatenate((base, excess)))
        if initial.arm != "lift":
            return size(np.array(POSES[initial.arm]))
        planned = lift_weight(initial.u)
        lo, hi = max(0.0, planned - LAG_W), min(1.0, planned + AHEAD_W)
        left, right, ends = lo, hi, [lo, hi]
        for j in range(3):
            if PATH[j] != 0.0:
                a, b = sorted(((q[j] - Q_CARRY[j] - DQ[j]) / PATH[j],
                               (q[j] - Q_CARRY[j] + DQ[j]) / PATH[j]))
                left, right = max(left, a), min(right, b)
                ends += [min(hi, max(lo, a)), min(hi, max(lo, b))]
        poses = ([0.5 * (left + right)] if left <= right else []) + ends
        return min(size(Q_CARRY + PATH * w) for w in poses)

    # --- after an alarm ------------------------------------------------------
    def rest_law(self, measured):
        st = State(measured.s, measured.v, measured.a, measured.arm, measured.u)
        tail, _ = self.recovery(lambda j, seg: hl.GLOBAL, 0, st, FLOOR)
        if tail:
            return Law(tail[0][0], "rest")
        return Law(Segment(measured.s, 0.0, 0.0, 0.0, measured.arm, None), "rest")

    def reach(self, measured, bound):
        if bound > 1e-12:
            return None, "the funnels assume an exact measurement"
        if abs(measured.v) > AT_REST or abs(measured.a) > 1e-12 or measured.arm == "lift":
            return None, "no restart candidate while the machine moves or lifts"
        return State(measured.s, 0.0, 0.0, measured.arm, None), ""

    def restart(self, step, reach):
        return self.generate(step, Plan(step.k + 1, [], Rest(reach.s, reach.arm)), reach)

    # --- diagnostic descriptions ---------------------------------------------
    def describe_plan(self, plan):
        h = plan.head()
        label = plan.pairs[0][1] if plan.pairs else hl.GLOBAL
        return dict(a1=h.a1, arm=h.arm, friction=label.friction, pairs=len(plan.pairs),
                    rest=plan.rest.s, rest_arm=plan.rest.arm)

    def describe_set(self, initial):
        return dict(s_c=initial.s, v_c=initial.v, a_c=initial.a, arm_c=initial.arm)


@register
class RouteArmGentle(RouteArm):
    """hdmm.route-arm with a recovery that brakes at no more than
    GENTLE = 1.5 m/s^2 on a straight: a different control package."""
    NAME = "hdmm.route-arm-gentle"
    GENTLE = 1.5


def _next_reach(seg):
    """An arc length beyond which neither the segment nor the next one can
    reach: its end plus a period at its end's speed and the drive's
    acceleration."""
    e = seg.end()
    return e.s + (e.v + DRIVE * T) * T


def _starts_at(plan, initial):
    """The plan's head starts at the initial set's commanded state, the arm
    as it is there or beginning the lift from carrying."""
    h = plan.head()
    same = (abs(h.s0 - initial.s) <= 1e-9 and abs(h.v0 - initial.v) <= 1e-9
            and abs(h.a0 - initial.a) <= 1e-9)
    arm = ((h.arm, h.u0) == (initial.arm, initial.u)
           or (h.arm == "lift" and h.u0 == 0.0 and initial.arm == "carry"))
    return same and arm


# ----------------------------------------------------------------------------
# Proposal methods
# ----------------------------------------------------------------------------
def front_of(measured):
    return measured.s + hl.front_point((measured.slew, measured.lift, measured.tilt))


@register
class Cruise:
    """Asks for the top speed until the measured front-most point first
    reaches the drop zone, within route.DROP_ZONE of the stop line, and for
    rest from then on. It knows nothing of the turns, the grounds or the
    arm."""
    ROLE, NAME, CONTRACT = "proposal", "hdmm.cruise", "1.0"

    def __init__(self, speed=SPEED):
        self.speed, self.arrived = speed, False

    def propose(self, observation):
        if front_of(observation.measured) >= route.S_LINE - route.DROP_ZONE:
            self.arrived = True
        return 0.0 if self.arrived else self.speed


@register
class RouteAware:
    """A stand-in for a learner trained on earlier governed runs in the yard,
    which has learned from their executed motion the speed at which the
    machine reaching took each turn (turn_speeds). It asks for the top
    speed; before each turn, for the speed from which a slowing at decel,
    its deceleration rising and fading over the actuators' rise, reaches the
    turn's speed two periods before its entry, the room that the machine
    needs to have eased its deceleration there; and before the stop line,
    for the speed from which such a slowing reaches creep_speed creep metres
    before the front-most point lies inside the drop zone by inside, then
    for creep_speed until it will, and then for rest. Each
    distance is taken from where the machine will be lead periods on, when
    the command that it asks for has taken effect, and a request slows down
    by no more than rate periods of decel. It knows neither the grounds nor
    the labels."""
    ROLE, NAME, CONTRACT = "proposal", "hdmm.route-aware", "1.0"

    def __init__(self, turn_speeds=(1.96, 1.46), decel=1.0, lead=2.0, creep=0.5,
                 creep_speed=0.25, rate=2.0, inside=0.15):
        self.turn_speeds, self.decel, self.lead, self.rate = tuple(turn_speeds), decel, lead, rate
        self.creep, self.creep_speed, self.inside = creep, creep_speed, inside
        self.arrived = False

    def curve(self, until, d):
        """The largest speed from which a slowing at decel reaches 'until'
        within the distance d."""
        if d <= 0.0:
            return until
        lo, hi = until, SPEED
        if hl.slowing(hi, self.decel, until)[0] <= d:
            return hi
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if hl.slowing(mid, self.decel, until)[0] <= d else (lo, mid)
        return lo

    def propose(self, observation):
        m = observation.measured
        ahead = m.s + self.lead * m.v * T
        zone = route.S_LINE - route.DROP_ZONE
        if front_of(m) >= zone + self.inside:      # inside the zone: rest
            self.arrived = True
        if self.arrived or front_of(m) + ahead - m.s >= zone + self.inside:
            return 0.0                             # inside by the time the command acts
        ask = SPEED
        for t, speed in zip(route.TURNS, self.turn_speeds):
            if m.s <= t.end:
                ask = min(ask, self.curve(speed, t.start - 2.0 * speed * T - ahead))
        stop = zone + self.inside - hl.front_point(POSES["reach"]) - self.creep
        ask = min(ask, max(self.creep_speed, self.curve(self.creep_speed, stop - ahead)))
        if ask < m.v and self.rate is not None:  # slow down at decel, not at once
            ask = max(ask, m.v - self.rate * self.decel * T)
        return ask


# ----------------------------------------------------------------------------
# Capability predictors
# ----------------------------------------------------------------------------
class _Labels:
    ROLE, CONTRACT, SPACE = "capability", "1.0", Capability.NAME
    horizon = HORIZON

    def __init__(self, payloads=DECLARED):
        self.payloads = tuple(float(p) for p in payloads)

    def observe(self, k, before, after):
        pass

    def friction(self, lo, hi):
        raise NotImplementedError

    def labels(self, observation):
        k = observation.k

        def labelling(j, segment):
            if j > k + 1 + self.horizon:
                return hl.GLOBAL
            return hl.Label(self.payloads, self.friction(*region(segment)))
        return labelling


@register
class Fixed(_Labels):
    """The global capability everywhere."""
    NAME = "hdmm.fixed"

    def labels(self, observation):
        return lambda j, segment: hl.GLOBAL


@register
class OldMap(_Labels):
    """A map of the yard drawn before the wet patch was reported: the
    survey's friction of the firm ground everywhere."""
    NAME = "hdmm.old-map"

    def friction(self, lo, hi):
        return route.SURVEY_FIRM


@register
class Survey(_Labels):
    """The site's survey: the least of the surveyed frictions of the grounds
    under the region, the wet patch included."""
    NAME = "hdmm.survey"

    def friction(self, lo, hi):
        return min(g.survey for g in route.grounds_on(lo, hi))


@register
class Oracle(_Labels):
    """The grounds' true frictions: a diagnostic, which no machine has
    (Chapter 16)."""
    NAME = "hdmm.oracle"

    def friction(self, lo, hi):
        return min(g.friction for g in route.grounds_on(lo, hi))


# ----------------------------------------------------------------------------
# The lift and the demand predictor that reads its schedule
# ----------------------------------------------------------------------------
@register
class Lift:
    """The lift from carrying to reaching, due in period 'onset' and needing
    'level' of tipping reserve (labels.lift_demand). It begins if the plan
    in force during its onset period holds that level (Assumption 4.13):
    from then on, the generator is asked to begin it, and every candidate
    does until the lift has begun. Otherwise an overload occurs, and the
    operation is postponed: to the forecast onset if that is still ahead,
    else by 'postpone' periods, announced (Section 8.2.3). announce=None:
    no announcement before the onset; always: the requirement from the first
    period on (fixed conservatism, Section 8.2.1)."""
    ROLE, NAME, CONTRACT = "task", "hdmm.lift", "1.0"

    def __init__(self, level, announce=2, onset=12, postpone=5, always=False):
        self.level, self.announce, self.onset = round(level, 9), announce, onset
        self.predicted, self.postpone, self.always = onset, postpone, always
        self.begun, self.overloads = None, []

    def pending(self):
        return self.begun is None

    def begin_period(self, k, level_in_force):
        if not self.pending() or k != self.onset:
            return
        if level_in_force != NEG_INF and level_in_force >= self.level - 1e-9:
            self.begun = k
            return
        self.overloads.append(k)
        if self.announce is not None and self.predicted > k:
            self.onset = self.predicted
        else:
            self.onset = self.predicted = k + self.postpone
            self.announce = k

    def operation(self, k):
        return "lift" if self.begun is not None else None


@register
class LiftPlan:
    """The demand predictor: from the announcement on, the lift's level for
    the onset that the task plan forecasts, as the cart's task plan
    (Section 8.2); 'always' requires it from the first pair on in every
    period while the lift is pending."""
    ROLE, NAME, CONTRACT = "demand", "hdmm.lift-plan", "1.0"

    def __init__(self, task):
        self.task = task

    def requirement(self, observation):
        task, k = self.task, observation.k
        if task.pending():
            if task.always:
                return Requirement(task.level, 0)
            if task.announce is not None and k >= task.announce:
                return Requirement(task.level, max(task.predicted - k - 1, 0))
        return Requirement(FLOOR, 0)


# ----------------------------------------------------------------------------
# Robot and world adapters
# ----------------------------------------------------------------------------
Measured = namedtuple("Measured", "s v a slew lift tilt arm u")
"""The measurement at a period boundary: the arc length (the commanded one
with the creep), the speed and the acceleration, the joints' angles, and the
arm's commanded state, which the controller knows."""


class YardState(tuple):
    """The world's state at a period boundary: a tuple of the declared
    components (s, v, a, slew, lift, tilt), s the true arc length, with the
    commanded state (commanded), the creep (creep: along, across, turned)
    and the simulated centre of pressure's distance inside the contacts
    (margin) as attributes."""

    def __new__(cls, values, commanded, creep, margin, t):
        obj = super().__new__(cls, values)
        obj.commanded, obj.creep, obj.margin, obj.t = commanded, creep, margin, t
        return obj


@register
class Robot:
    """The state and the actuation of the machine; the measurement is the
    simulated state, exact."""
    ROLE, NAME, CONTRACT = "robot", "hdmm.robot", "1.0"
    STATE, ACTUATION = STATE, ACTUATION
    measurement_bound = 0.0

    def measure(self, state):
        c = state.commanded
        return Measured(*state, c.arm, c.u)

    def describe_state(self, measured):
        return dict(s_hat=measured.s, v_hat=measured.v)


@register
class Yard:
    """The machine of sim.py on the route of route.py, the payload held and
    the payload believed for the servos' offsets. It realizes the commanded
    motion of each period's law exactly, the base creeping or sliding on its
    contacts as the simulation lets it; a law that starts from another arc
    length than the last one commanded, as the rest feedback's and a
    restart's from a measurement, re-anchors the creep there."""
    ROLE, NAME, CONTRACT = "world", "hdmm.yard", "1.0"
    STATE, ACTUATION = STATE, ACTUATION

    def __init__(self, payload=270.0, believed=None):
        self.payload, self.believed = float(payload), believed

    def reset(self):
        self.machine = sim.Machine(self.payload, POSES["carry"], s0=0.0, believed=self.believed)
        self.commanded, self.ref = State(0.0, 0.0, 0.0, "carry", None), (0.0, 0.0, 0.0)
        self.bounds = [0]          # the log's index at each period boundary
        self.laws = []
        return self._state()

    def _state(self):
        c = self.commanded
        cx, cy, yaw = (x - r for x, r in zip(self.machine.creep(), self.ref))
        q = self.machine.joints()
        log = self.machine.log
        if log["t"]:
            margin = min(model.AXLE - abs(log["cop_x"][-1]),
                         model.HALF_TRACK - abs(log["cop_y"][-1]))
        else:
            p, _ = model.centre_of_pressure(self.machine.m, self.machine.d)
            margin = min(model.AXLE - abs(p[0]), model.HALF_TRACK - abs(p[1]))
        return YardState((c.s + cx, c.v, c.a, float(q[0]), float(q[1]), float(q[2])),
                         c, (cx, cy, yaw), margin, self.machine.t)

    def step(self, state, law):
        seg = law.segment
        if abs(seg.s0 - self.commanded.s) > 1e-9:
            self.ref = self.machine.creep()
        self.machine.advance(T, lambda tau: (seg.arc(tau), seg.speed(tau), seg.accel(tau)),
                             seg.pose)
        self.commanded = seg.end()
        self.bounds.append(self.machine.steps())
        self.laws.append(law)
        return self._state()

    # --- the oracle, for the checks of Section 10.3 (simulation only) ---------
    def admits(self, label, segment, k):
        """True if the world admits the label on the segment's region: the
        payload within the label's interval and the friction under every
        wheel at least the label's."""
        lo, hi = region(segment)
        return (label.payloads[0] <= self.payload <= label.payloads[1]
                and label.friction <= min(g.friction for g in route.grounds_on(lo, hi)) + 1e-12)

    def clearance(self, state):
        """The true tipping reserve of a state: the simulated centre of
        pressure's distance inside the rectangle of the contacts at its
        instant."""
        return state.margin


# ----------------------------------------------------------------------------
# Without the governor
# ----------------------------------------------------------------------------
def ungoverned(world, proposal, task, periods):
    """The proposal executed at the actuators' limits, without the governor
    or the test: each period, the commanded acceleration that brings the
    speed nearest to the proposal after the period and one more (as the
    package's meaning of a proposal), within the brakes' and the drive's
    limits, the steering following the route at whatever speed; the lift
    begun in the period after its onset whatever the reserve. Returns the
    states at the period boundaries and the segments."""
    robot = Robot()
    x = world.reset()
    states, segs = [x], []
    for k in range(periods):
        m = robot.measure(x)
        q = proposal.propose(Observation(k, k * T, m, NEG_INF, 0, False))
        task.begin_period(k, math.inf)
        c = x.commanded
        arm, u = c.arm, c.u
        if task.begun is not None and k > task.begun and arm == "carry":
            arm, u = "lift", 0.0
        target = min(SPEED, max(0.0, q))
        a1 = min(DRIVE, max(-BRAKES, (target - c.v - c.a * T / 2.0) / T))
        if c.v <= AT_REST and target <= 0.0:
            a1 = 0.0
        seg = Segment(c.s, c.v, c.a, a1, arm, u)
        x = world.step(x, Law(seg))
        states.append(x)
        segs.append(seg)
    return states, segs
