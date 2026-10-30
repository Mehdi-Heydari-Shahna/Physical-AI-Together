"""
The Husky as modules of the allovi core of Chapter 10 (Section 13.2).

Physical AI, Together, Chapter 13 (Section 13.2).

The state is x = (s, u, d, e): the arc length of the robot's centre along
the route of world.py, its forward speed, its offset from the centre line,
positive to the left, and its heading error, the heading less the route's
direction at s. Its norm is the length

    ||x|| = max(|s|, |u| TAU_U, |d| / 2, |e| RHO_E),  TAU_U = 0.5 s, RHO_E = 2 m:

the scales turn the speed into the distance covered in half a second, and
the heading error into the distance that it opens sideways over 2 m.

The control package husky.route-speed plans the robot's motion along the
route, as the cart's package plans the cart's along its track, and leaves
the route itself to the feedback of follow.py:

  * a segment is the commanded motion of one period of T = 0.2 s, ten
    updates of the drive controller: a commanded speed that changes at a
    constant rate from c0 to c1, at most V_MAX = 1 m/s, accelerating at most
    A_MAX = 1 m/s^2 and decelerating at most its label, and at most
    B_MAX = 4 m/s^2, the drive controller's limit;
  * a plan is a list of (segment, label) pairs that start where the head of
    the plan in force ends, and a rest point, the commanded arc length at
    which the last pair comes to rest;
  * the funnel of a segment is a box around the commanded motion: at each
    instant, the states within DS = E_S + F_S of the commanded arc length,
    DU = E_U + F_U of the commanded speed, DD = E_D + F_D of the centre line
    and DE = E_E + F_E of the route's direction. The estimate lies within
    E_S, E_U, E_D and E_E of the state, which is the measurement-error bound
    VARPI in the norm, and the feedback holds the estimate within F_S, F_U,
    F_D and F_E of the commanded motion, at every update of the controller.
    The box is weaker than a funnel of Definition 4.4, whose condition (i)
    asks that every motion that starts anywhere in it stay in it: what the
    package assumes is that the executed motion keeps the box's bounds.
    These bounds are assumptions, ours: Section 13.2 measures how well the
    simulated robot keeps them, and no proof establishes them;
  * the constraints are the dock and the lane, two families with a reserve
    each (Section 4.2.7). The commanded arc length of a plan is largest at
    its rest point. At a heading error e, the footprint reaches
    reach(e) = l/2 |cos e| + w/2 |sin e| ahead of the centre, which grows
    with |e| up to PSI_STAR = atan(w / l), where half the footprint's
    diagonal points along the route, and never exceeds that half diagonal;
    front(psi), the largest reach over the heading errors up to psi, is
    therefore reach(min(psi, PSI_STAR)). The dock's reserve of a plan is the
    largest eps with eps + front(DE + eps / RHO_E) <= DOCK - s_rest - DS; for
    reserves up to RHO_E (PSI_STAR - DE), 1.098 m, front is the reach at
    DE + eps / RHO_E itself. The test rounds the reserve down to whole
    centimetres from the floor of 0.1 m. The lane's reserve is
    the same for every plan, since every funnel has the same bounds across
    the route, and the narrowest allowance of the route gives it:
    LANE_RESERVE, the largest eps for which an offset of DD + 2 eps lies
    within the lane's narrowest allowance at the heading error
    DE + eps / RHO_E (lane_allowance, lane_reserve), a lower bound on the
    lane's reserve of each plan. The lane's levels are that single level,
    which is also their floor; the core, which has no vector levels, carries
    the dock's reserve in its scalar levels. The lane keeps the robot 0.1 m
    from the pallets, which need no family of their own. The task's speed
    limit is a limit on the commanded speed (Section 13.1.1): the plans'
    limit, which the test checks and at which the feedback's bounds were
    measured; it is no constraint on the state, and the robot's own speed is
    bounded only by the funnel, at V_MAX + DU;
  * the recovery continuation of every candidate is the stop: the
    commanded speed brought to zero at the braking of each period's label;
    the rest segment then commands rest at the plan's point of rest, its
    drivers holding the wheels, which on the level yard they can under the
    global capability; its funnel is the box at rest;
  * after an alarm, the rest feedback brings the commanded speed to zero at
    the global capability, from the estimated speed. A restart is offered
    once the estimated speed is below AT_REST and the estimate lies within
    the feedback's bounds across the route; the package then assumes, as
    its bound on what the rest feedback can reach (Section 8.3), that the
    rest feedback holds every state within VARPI of the measurement in the
    funnel of a rest at the estimated arc length, from which the restart
    candidates start.

A label is a braking b [m/s^2] that the plans may command in a segment,
with this meaning: the robot, commanded to slow at up to b there, keeps in
its executed motion the four bounds of the funnel. It is read from the
grounds under the wheels of every state of the segment's region, the union
of its funnel, the ground of a wheel being the ground under its centre
(sim.py): the grounds change only on the straights and where the turn
begins. On a straight, a wheel's arc length lies within
HALF_WHEELBASE cos e + HALF_TRACK |sin e| of the centre's, so the
predictors read the grounds on the commanded arc lengths of the segment
widened by REACH = DS + HALF_WHEELBASE + HALF_TRACK sin DE; where the turn
begins, an inner rear wheel can lie further behind, on the asphalt, whose
label exceeds the gravel's, so the least label is the same. The
labels of the map grant ETA = 0.6 of mu g, at most B_MAX; the global
capability is B_GLOBAL = 0.8 m/s^2, our statement of what every ground of
the yard allows in every weather of Section 13.2.5. Neither ETA nor B_GLOBAL
is measured from a real ground.

The robot adapter's measurement is the estimate of estimate.py, which the
world adapter runs at 50 Hz with the drive controller, with the error bound
VARPI. The world adapter husky.yard runs the robot of sim.py in the yard,
with the drive controller, the drivers, the estimator and the feedback of
the executed law.
"""
import math
import os
import sys
from collections import namedtuple

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "ch10"))
from allovi import (NEG_INF, Actuation, Bound, Candidate, Profile, Quantity,   # noqa: E402
                    Schema, ScalarLevels, Why, register)

from . import estimate, follow, sim, world                                    # noqa: E402

G = 9.81
T = 0.2                  # s, the governor's period
V_MAX = 1.0              # m/s, the route's limit
A_MAX = 1.0              # m/s^2, the plans' acceleration (ours; the controller allows 2)
B_MAX = 4.0              # m/s^2, the plans' deceleration at most (the controller's limit)
ETA = 0.6                # the share of mu g that a label of the map grants (ours)
B_GLOBAL = 0.8           # m/s^2, the global capability (ours)

TAU_U = 0.5              # s, the norm's scale of the speed
D_SCALE = 2.0            # the norm takes half the offset
RHO_E = 2.0              # m, the norm's scale of the heading error
VARPI = 0.05             # m, the measurement-error bound (S5), in the norm
E_S, E_U, E_D, E_E = VARPI, VARPI / TAU_U, D_SCALE * VARPI, VARPI / RHO_E
F_S = 0.02               # m, the feedback's bound on |s_hat - s_c| (ours, measured)
F_U = 0.15               # m/s, on |u_hat - u_c| (ours, measured)
F_D = 0.06               # m, on |d_hat| (ours, measured)
F_E = 0.04               # rad, on |e_hat| (ours, measured)
DS, DU, DD, DE = E_S + F_S, E_U + F_U, E_D + F_D, E_E + F_E   # the funnel

FOOT_L, FOOT_W = world.FOOTPRINT
HALF_WHEELBASE = 0.256   # m, the wheels' centres ahead of and behind the robot's
HALF_TRACK = 0.2829      # m, half the model's track of 0.5658 m (asset.py)
REACH = DS + HALF_WHEELBASE + HALF_TRACK * math.sin(DE)
DOCK = world.DOCK_S
DOCK_ZONE = 0.40         # m: the task asks for rest with the front within this of the dock
FLOOR, GRID = 0.10, 0.01
LEVELS = ScalarLevels(floor=FLOOR, step=GRID, top=DOCK)
ALPHAS = (1.0, 0.75, 0.5, 0.25)
# the lane (lane_allowance, lane_reserve; husky_route.py computes these and checks them)
LANE_ALLOWANCE = 0.289   # m, the narrowest allowance at the heading error DE
LANE_RESERVE = 0.0586    # m, the lane's reserve in the norm, the same for every plan

STATE = Schema(
    "Husky on the route",
    (Quantity("s", "m", "route"),          # arc length of the robot's centre
     Quantity("u", "m/s", "base_link x"),  # forward speed
     Quantity("d", "m", "route normal"),   # offset from the centre line, to the left
     Quantity("e", "rad", "z")),           # heading less the route's direction
    norm="max(|s|, |u| * 0.5 s, |d| / 2, |e| * 2 m)")

ACTUATION = Actuation(
    inputs=(Quantity("v", "m/s", "base_link x"),
            Quantity("omega", "rad/s", "z")),
    bounds=(Bound("v", low=-2.0, high=2.0), Bound("omega", low=-2.0, high=2.0)),
    hold="the drive controller's: 50 Hz, its limits of speed, acceleration and "
         "deceleration",
    meaning="the velocity command of the A300's drive controller "
            "(diff_drive_controller, diff_4wd.yaml), computed by the feedback of "
            "the executed law at each update of the controller")


# ----------------------------------------------------------------------------
# Plans, funnels and the capability space
# ----------------------------------------------------------------------------
Motion = follow.Motion

Funnel = namedtuple("Funnel", "s c width")
"""An initial set: the states whose arc length lies within 'width' of s,
whose speed lies within DU of the commanded speed c, and whose offset and
heading error lie within DD and DE; c is the commanded speed to continue
from."""

Measured = namedtuple("Measured", "s u d e")
"""The estimate at a period boundary: arc length, speed, offset, heading error."""


PSI_STAR = math.atan2(FOOT_W, FOOT_L)        # rad: half the diagonal points along the route
HALF_DIAGONAL = 0.5 * math.hypot(FOOT_L, FOOT_W)


def reach(e):
    """How far the footprint reaches ahead of the robot's centre, along the
    route, at the heading error e."""
    return 0.5 * FOOT_L * abs(math.cos(e)) + 0.5 * FOOT_W * abs(math.sin(e))


def front(psi):
    """The largest reach of the footprint over the heading errors of
    magnitude up to psi: reach grows with |e| up to PSI_STAR and never
    exceeds HALF_DIAGONAL, its value there."""
    return reach(min(abs(psi), PSI_STAR))


def dock_reserve(s_rest):
    """The largest eps with eps + front(DE + eps / RHO_E) <= DOCK - s_rest - DS:
    the dock's reserve, in the norm, of a plan that comes to rest at s_rest,
    to 1e-12 m by bisection; negative if the funnel itself reaches the dock."""
    room = DOCK - s_rest - DS
    if room - front(DE) < 0.0:
        return room - front(DE)
    lo, hi = 0.0, room
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if mid + front(DE + mid / RHO_E) <= room:
            lo = mid
        else:
            hi = mid
    return lo


class Plan:
    """(segment, label) pairs from period j0 on, and the rest point."""

    def __init__(self, j0, pairs, s_rest):
        self.j0, self.pairs, self.s_rest = j0, list(pairs), s_rest

    def shift(self):
        if not self.pairs:
            return self
        return Plan(self.j0 + 1, self.pairs[1:], self.s_rest)

    def head(self):
        if self.pairs:
            return self.pairs[0][0]
        return Motion(self.s_rest, 0.0, 0.0, T)

    def reserve(self):
        return dock_reserve(self.s_rest)


def level_of(reserve):
    """The largest level of the grid not above the reserve, or NEG_INF
    below the floor: whole centimetres from the floor, with the reserve
    rounded to 1e-9 m first, so that a reserve of exactly 0.11 m is not lost
    to the floating-point division."""
    x = round(reserve, 9)
    if x < FLOOR:
        return NEG_INF
    n = math.floor((x - FLOOR) / GRID + 1e-9)
    return min(LEVELS.top, round(FLOOR + n * GRID, 9))


def pair_failure(m, label):
    """What keeps the test from certifying a pair, or None."""
    top = V_MAX + 1e-12
    if not (0.0 <= m.c0 <= top and 0.0 <= m.c1 <= top):
        return "speed beyond the route's limit"
    rate = (m.c1 - m.c0) / m.T
    if rate > A_MAX + 1e-12:
        return f"acceleration {rate:.3g} m/s^2 beyond {A_MAX:g}"
    if -rate > min(label, B_MAX) + 1e-12:
        return (f"deceleration {-rate:.3g} m/s^2 beyond the "
                f"label {label:.3g}")
    return None


class Braking:
    """The capability space: a label is the braking [m/s^2] that the plans
    may command in a segment. More braking admits fewer worlds: c is at
    least as informative as d if c >= d."""
    NAME = "husky.braking"
    global_capability = B_GLOBAL

    def informative(self, c, d):
        return c >= d

    def labelling(self, capability):
        return lambda j, segment: capability


def region(m):
    """The arc lengths that the wheels of a robot in the segment's funnel can
    occupy (module docstring)."""
    return m.s0 - REACH, m.arc(m.T) + REACH


def label_of(friction):
    """The braking that a ground of friction mu grants: ETA mu g, at most B_MAX."""
    return min(B_MAX, ETA * friction * G)


def recovery(labelling, j, s, c):
    """The stop from (s, c) at the braking of each period's label, from
    period j on: the pairs and the rest point. Each period brakes at the
    label of the region that a coasting segment would cover, which contains
    the region of the braking segment."""
    pairs = []
    while c > 0.0:
        coast = Motion(s, c, c, T)
        b = min(B_MAX, labelling(j, coast))
        if b <= 0.0:
            raise ValueError("a label without braking")
        m = Motion(s, c, max(0.0, c - b * T), T)
        pairs.append((m, labelling(j, m)))
        s, c = m.end()
        j += 1
    return pairs, s


# ----------------------------------------------------------------------------
# The lane
# ----------------------------------------------------------------------------
LANE_POSITIONS = (5.0, 20.0) + tuple(round(9.4 + 0.1 * i, 1) for i in range(61))


def lane_allowance(psi, positions=LANE_POSITIONS):
    """The narrowest room of the lane for the robot's centre: the largest
    offset, outwards or inwards, at which every point of the footprint lies
    in the lane (world.footprint_offset) with the heading error psi to
    either side, by bisection to 1e-9 m, at arc lengths on the straights and
    every 0.1 m from 9.4 to 15.4 m, through the turn and half a length and
    more beyond its ends."""
    best = math.inf
    for s in positions:
        x, y, th = world.point(s)
        nx, ny = -math.sin(th), math.cos(th)
        for side in (1.0, -1.0):
            for sign in (1.0, -1.0):
                lo, hi = 0.0, world.LANE_HALF_WIDTH
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    if world.footprint_offset(x + side * mid * nx, y + side * mid * ny,
                                              th + sign * psi) <= world.LANE_HALF_WIDTH:
                        lo = mid
                    else:
                        hi = mid
                best = min(best, lo)
    return best


def lane_reserve(iterations=4):
    """The lane's reserve in the norm: the largest eps with
    DD + D_SCALE eps <= lane_allowance(DE + eps / RHO_E), by iterating the
    map eps -> (lane_allowance(DE + eps / RHO_E) - DD) / D_SCALE, which
    decreases, from eps = 0; the smaller of its last two values, which
    bracket the solution."""
    values = [0.0]
    for _ in range(iterations):
        values.append((lane_allowance(DE + values[-1] / RHO_E) - DD) / D_SCALE)
    return min(values[-2:])


def follower_state():
    return follow.Follower()


class Law:
    """The feedback law of one period: the commanded motion of the head, and
    the feedback that follows it. Called with the time within the period and
    the estimate, it returns the velocity command of the drive controller."""

    def __init__(self, motion, kind, feedback):
        self.motion, self.kind, self.feedback = motion, kind, feedback
        self._last = None

    def __call__(self, tau, est):
        dt = 0.0 if self._last is None else tau - self._last
        if dt <= 0.0:
            dt = 0.02
        self._last = tau
        return self.feedback.command(self.motion, tau, est, dt)


AT_REST = 1e-3           # m/s: the estimated speed below which the robot is taken to be at rest


# ----------------------------------------------------------------------------
# The control package
# ----------------------------------------------------------------------------
@register
class RouteSpeed:
    """Segments of a commanded speed along the route, the generator, the test,
    the recovery continuation, the rest segment and the rest feedback (module
    docstring)."""
    ROLE, NAME, CONTRACT = "control", "husky.route-speed", "1.0"
    STATE, ACTUATION = STATE, ACTUATION

    def __init__(self):
        self.capabilities, self.levels = Braking(), LEVELS
        self.feedback = follower_state()
        if LANE_RESERVE < 0.0:
            raise ValueError("the funnel's offsets and headings do not fit the lane")

    # --- plans and the test ------------------------------------------------
    def initial_plan(self, measured, bound):
        if bound > VARPI + 1e-12:
            raise ValueError(f"the funnels hold the states within {VARPI} of a measurement")
        if abs(measured.u) > AT_REST:
            raise ValueError("this package certifies an initial plan only at rest")
        if abs(measured.d) > F_D or abs(measured.e) > F_E:
            raise ValueError("the estimate lies beyond the feedback's bounds across the route")
        return Plan(0, [], measured.s), Funnel(measured.s, 0.0, DS)

    def initial_set(self, plan):
        s, c = plan.head().end()
        return Funnel(s, c, DS)

    def generate(self, step, plan, initial):
        s0, c0 = initial.s, initial.c
        labelling = step.labelling
        k = step.k
        if step.proposal is None:
            return [self._recovery_only(labelling, k, s0, c0)]
        fallback = plan.shift()
        fb_c1 = fallback.head().c1 if fallback.pairs else 0.0
        q_c1 = self.proposed_speed(step.proposal, labelling, k + 1, s0, c0)
        out = []
        for a in ALPHAS:
            c1 = a * q_c1 + (1.0 - a) * fb_c1
            head = Motion(s0, c0, c1, T)
            tail, s_rest = recovery(labelling, k + 2, *head.end())
            out.append(Candidate(Plan(k + 1, [(head, labelling(k + 1, head))] + tail, s_rest), a))
        out.append(self._recovery_only(labelling, k, s0, c0))
        return out

    def proposed_speed(self, proposal, labelling, j, s0, c0):
        """The physical meaning of a proposal, a speed asked for [m/s]: the
        commanded speed at the end of the head that comes nearest to it within
        the limits of a segment, braking at most at the label of a coasting
        segment's region."""
        b = min(B_MAX, labelling(j, Motion(s0, c0, c0, T)))
        target = min(V_MAX, max(0.0, proposal))
        return min(c0 + A_MAX * T, max(c0 - b * T, target, 0.0))

    def _recovery_only(self, labelling, k, s0, c0):
        tail, s_rest = recovery(labelling, k + 1, s0, c0)
        if not tail:
            hold = Motion(s0, 0.0, 0.0, T)
            tail = [(hold, labelling(k + 1, hold))]
        return Candidate(Plan(k + 1, tail, s_rest), 0.0)

    def profile(self, plan, initial):
        """Condition (P1): the plan starts at the commanded point of the
        initial set, a set no wider than its first funnel. Then, for each n,
        the dock's reserve level if every pair from n on passes, else
        NEG_INF."""
        if not _starts_at(plan, initial):
            return Profile.uncertified()
        level = level_of(plan.reserve())
        entries, ok = [], True
        for m, label in reversed(plan.pairs):
            ok = ok and pair_failure(m, label) is None
            entries.append(level if ok else NEG_INF)
        entries.reverse()
        return Profile(entries + [level])

    def explain(self, plan, initial):
        if not _starts_at(plan, initial):
            return [Why("the plan does not start at the initial set", "pair")]
        r = plan.reserve()
        text = f"the dock's reserve {r:.4f} m at the rest point {plan.s_rest:.4f} m"
        kind = "certified" if r >= FLOOR else "floor"
        out, first = [], None
        for i in range(len(plan.pairs) - 1, -1, -1):
            f = pair_failure(*plan.pairs[i])
            if f is not None:
                first = (i, f)
            out.append(Why(f"pair {first[0]} not certified: {first[1]}", "pair")
                       if first else Why(text, kind))
        out.reverse()
        return out + [Why(text + (", below the floor" if r < FLOOR else ""), kind)]

    # --- execution and the alarm check -------------------------------------
    def law(self, plan):
        return Law(plan.head(), "segment", self.feedback)

    def distance(self, measured, initial):
        """In the norm of the state: how far the estimate lies outside the
        initial set."""
        return max(0.0, abs(measured.s - initial.s) - initial.width,
                   (abs(measured.u - initial.c) - DU) * TAU_U,
                   (abs(measured.d) - DD) / D_SCALE,
                   (abs(measured.e) - DE) * RHO_E)

    # --- after an alarm ------------------------------------------------------
    def rest_law(self, measured):
        c = measured.u if measured.u > AT_REST else 0.0
        return Law(Motion(measured.s, c, max(0.0, c - B_GLOBAL * T), T), "rest",
                   self.feedback)

    def reach(self, measured, bound):
        if bound > VARPI + 1e-12:
            return None, f"the funnels hold the states within {VARPI} of a measurement, no more"
        if abs(measured.u) > AT_REST:
            return None, ("no restart candidate while the robot moves: the stop of the "
                          "rest feedback depends on the ground")
        if abs(measured.d) > F_D or abs(measured.e) > F_E:
            return None, ("the estimate lies beyond the feedback's bounds across the route: "
                          "no funnel of the package holds the states that it allows")
        return Funnel(measured.s, 0.0, DS), ""

    def restart(self, step, reach):
        return self.generate(step, Plan(step.k + 1, [], reach.s), reach)

    # --- diagnostic descriptions ---------------------------------------------
    def describe_plan(self, plan):
        h = plan.head()
        return dict(c0=h.c0, c1=h.c1, label=plan.pairs[0][1] if plan.pairs else B_GLOBAL,
                    pairs=len(plan.pairs), rest=plan.s_rest)

    def describe_set(self, initial):
        return dict(s_c=initial.s, c=initial.c)


def _starts_at(plan, initial):
    h = plan.head()
    return (abs(h.s0 - initial.s) <= 1e-9 and abs(h.c0 - initial.c) <= 1e-9
            and initial.width <= DS + 1e-12)


# ----------------------------------------------------------------------------
# Proposal method
# ----------------------------------------------------------------------------
@register
class Cruise:
    """Asks for the route's limit, 1 m/s, until the measured front first
    reaches the dock zone, within DOCK_ZONE of the dock's face, and for rest
    from then on. It knows nothing of the grounds or the weather."""
    ROLE, NAME, CONTRACT = "proposal", "husky.cruise", "1.0"

    def __init__(self, speed=V_MAX):
        self.speed, self.arrived = speed, False

    def propose(self, observation):
        if observation.measured.s + 0.5 * FOOT_L >= DOCK - DOCK_ZONE:
            self.arrived = True
        return 0.0 if self.arrived else self.speed


# ----------------------------------------------------------------------------
# Capability predictors
# ----------------------------------------------------------------------------
class Shower:
    """A shower: from t_start on, the frictions of the grounds named in
    'frictions' (ours, Section 13.2.5); the time is the crossing's, from
    the first period."""

    def __init__(self, t_start, frictions, announced=None):
        self.t_start, self.frictions, self.announced = t_start, dict(frictions), announced

    def friction(self, ground, t):
        if t >= self.t_start and ground.name in self.frictions:
            return self.frictions[ground.name]
        return ground.friction


def grounds_on(lo, hi):
    """The grounds of the yard under the arc lengths [lo, hi] of the route."""
    return [g for g in world.GROUNDS if g.s_from < hi and g.s_to > lo] or [world.ground_at(lo)]


class _Labels:
    ROLE, CONTRACT, SPACE = "capability", "1.0", "husky.braking"
    horizon = 200

    def observe(self, k, before, after):
        pass


@register
class Fixed(_Labels):
    """The global capability everywhere."""
    NAME = "husky.fixed"

    def labels(self, observation):
        return lambda j, segment: B_GLOBAL


@register
class YardMap(_Labels):
    """The yard's map: the least label of the grounds on the region of the
    segment, from their frictions in world.py; beyond the horizon, the
    global capability."""
    NAME = "husky.map"

    def labels(self, observation):
        k = observation.k

        def labelling(j, m):
            if j > k + 1 + self.horizon:
                return B_GLOBAL
            return min(label_of(g.friction) for g in grounds_on(*region(m)))
        return labelling


@register
class Forecast(_Labels):
    """The map with a weather forecast: once the shower is announced, the
    labels of the periods from its start on are those of its frictions."""
    NAME = "husky.forecast"

    def __init__(self, shower):
        self.shower = shower

    def labels(self, observation):
        k, t = observation.k, observation.t
        sh = self.shower
        known = sh.announced is not None and t >= sh.announced

        def labelling(j, m):
            if j > k + 1 + self.horizon:
                return B_GLOBAL
            # period j runs from j T to (j + 1) T: the shower counts if it has begun by its end
            wet = known and (j + 1) * T > sh.t_start
            out = []
            for g in grounds_on(*region(m)):
                mu = sh.frictions.get(g.name, g.friction) if wet else g.friction
                out.append(label_of(mu))
            return min(out)
        return labelling


@register
class Reactive(_Labels):
    """A stand-in for a reactive predictor, which issues the braking measured
    on the positions that it has traversed (Section 3.1): on a segment whose
    region lies within the stretch that the wheels have covered, by the
    estimate, less E_S at its front, the least label of the grounds there,
    taken from the map where a reactive predictor would measure it; the
    global capability on any other segment. Reading the map, it is a
    forecasting predictor by the kinds of Section 4.3; and since a region
    reaches REACH beyond its commanded arc lengths, more than the covered
    stretch's HALF_WHEELBASE - E_S beyond the estimate, it issues no label of
    the map in the task of Section 13.2."""
    NAME = "husky.reactive"

    def __init__(self):
        self.start, self.reached = None, -math.inf

    def observe(self, k, before, after):
        self.reached = max(self.reached, before.s, after.s)

    def labels(self, observation):
        s_hat = observation.measured.s
        if self.start is None:
            self.start = s_hat - HALF_WHEELBASE      # the start is known exactly
        lo_ok = self.start
        hi_ok = max(self.reached, s_hat) + HALF_WHEELBASE - E_S

        def labelling(j, m):
            lo, hi = region(m)
            if lo >= lo_ok and hi <= hi_ok:
                return min(label_of(g.friction) for g in grounds_on(lo, hi))
            return B_GLOBAL
        return labelling


# ----------------------------------------------------------------------------
# Robot and world adapters
# ----------------------------------------------------------------------------
YardState = namedtuple("YardState", "s u d e x y heading t estimate")
"""The world's state at a period boundary: the true arc length, forward
speed, offset and heading error, the pose in the world frame, the time,
and the estimate (a Measured) that the robot holds then."""


@register
class Robot:
    """The state (s, u, d, e) and the velocity command of the drive
    controller; the measurement is the estimate, with the error bound VARPI."""
    ROLE, NAME, CONTRACT = "robot", "husky.robot", "1.0"
    STATE, ACTUATION = STATE, ACTUATION
    measurement_bound = VARPI

    def measure(self, state):
        return state.estimate

    def describe_state(self, measured):
        return dict(s_hat=measured.s, u_hat=measured.u, d_hat=measured.d, e_hat=measured.e)


def heading_error(s, heading):
    e = heading - world.point(s)[2]
    return (e + math.pi) % (2.0 * math.pi) - math.pi


@register
class Yard:
    """The Husky of sim.py in the yard: the drive controller, the drivers, the
    estimator of estimate.py and the feedback of the executed law. The
    shower's clock is the crossing's: it starts with the first period, after
    the estimator's calibration at rest. It logs the motion at every update
    of the controller (trace) and every contact with an obstacle (touches)."""
    ROLE, NAME, CONTRACT = "world", "husky.yard", "1.0"
    STATE, ACTUATION = STATE, ACTUATION

    TRACE = ("t", "s", "d", "u", "s_hat", "d_hat", "u_hat", "s_c", "u_c", "footprint",
             "e", "e_hat", "clearance", "v")
    ROBOT = sim.Husky        # the robot's class (Section 13.3 records its drives' torques)

    def __init__(self, estimator="sideslip", shower=None, dt=None, build=None, spec=None):
        self.kind, self.shower = estimator, shower
        self.dt = dt
        self.build, self.spec = dict(build or {}), spec

    def reset(self):
        """A new robot at the start, calibrated. The estimator is a kind of
        estimate.estimator, or a function of the robot; build holds arguments
        of model.build, and spec, if given, a function of them that returns
        the model's MjSpec (Section 13.3)."""
        build = dict(self.build)
        if self.dt is not None:
            build["timestep"] = self.dt
        h = self.ROBOT(**build) if self.spec is None else self.ROBOT(spec=self.spec(**build))
        h.place()
        h.command = (0.0, 0.0)
        est = self.kind(h) if callable(self.kind) else estimate.estimator(self.kind, h)
        if hasattr(est, "calibrate") and not est.skip_calibration:
            est.calibrate()
        else:
            h.run(estimate.CALIBRATION)
        est.start()
        self.h, self.est = h, est
        self.t0 = h.step_count * h.m.opt.timestep
        if self.shower is not None:
            t0, friction = self.t0, self.shower.friction
            h.weather = lambda ground, t: friction(ground, t - t0)
        self.trace, self.touches = [], []
        self.obstacles = {h.m.geom(o.name).id: o.name for o in world.OBSTACLES}
        return self._state()

    def time(self):
        return self.h.step_count * self.h.m.opt.timestep - self.t0

    def _state(self):
        h, est = self.h, self.est
        x, y, heading = h.pose()
        s, d = world.project(x, y)
        se, de, ee = est.route()
        return YardState(s, h.body_velocity()[0], d, heading_error(s, heading), x, y, heading,
                         self.time(), Measured(se, est.speed, de, ee))

    def step(self, state, law):
        h, est = self.h, self.est
        per = h.steps_per_update
        dt = per * h.m.opt.timestep
        n = int(round(T / dt))
        for i in range(n):
            tau = i * dt
            h.command = law(tau, est)
            v = h.command[0] / law.feedback.scale      # the feedback's speed command
            h.step(per)
            est.update(dt)
            self._log(law, tau + dt, v)
        return self._state()

    def _log(self, law, tau, v):
        h, est = self.h, self.est
        x, y, heading = h.pose()
        s, d = world.project(x, y)
        e = heading_error(s, heading)
        se, de, ee = est.route()
        self.trace.append((self.time(), s, d, h.body_velocity()[0], se, de, est.speed,
                           law.motion.arc(tau), law.motion.speed(tau),
                           world.footprint_offset(x, y, heading), e, ee,
                           DOCK - (s + reach(e)), v))
        for c in h.d.contact[:h.d.ncon]:
            for g in (int(c.geom1), int(c.geom2)):
                if g in self.obstacles:
                    self.touches.append((self.time(), self.obstacles[g], s))


@register
class Metrics:
    """From the record and the world's log: when the front first lies in the
    dock zone, the true clearance of the footprint from the dock at the end
    and its least value, the largest offset of the footprint from the
    centre line, the largest errors of the estimate, the largest distances
    of the estimate and of the state from the commanded motion, the largest
    distance of the state from its funnel in the norm (outside), the
    robot's largest speed, the largest speed that the plans command and the
    largest speed command of the feedback, the alarms and the touches."""
    ROLE, NAME, CONTRACT = "evaluator", "husky.metrics", "1.0"

    def __init__(self, world_adapter):
        self.world = world_adapter

    def evaluate(self, record):
        tr = self.world.trace
        i = {k: n for n, k in enumerate(Yard.TRACE)}

        def most(f):
            return max(f(r) for r in tr)
        zone = [r[0] for r in tr if r[i["clearance"]] <= DOCK_ZONE]
        rest = [r[0] for r in tr if r[i["clearance"]] <= DOCK_ZONE and abs(r[i["u"]]) < 1e-3]
        fin = record.final_state
        return dict(
            in_zone=zone[0] if zone else None,
            at_rest_in_zone=rest[0] if rest else None,
            final_clearance=DOCK - (fin.s + reach(fin.e)),
            least_clearance=min(r[i["clearance"]] for r in tr),
            final_s=fin.s, final_s_hat=fin.estimate.s,
            footprint=most(lambda r: r[i["footprint"]]),
            error_s=most(lambda r: abs(r[i["s"]] - r[i["s_hat"]])),
            error_u=most(lambda r: abs(r[i["u"]] - r[i["u_hat"]])),
            error_d=most(lambda r: abs(r[i["d"]] - r[i["d_hat"]])),
            error_e=most(lambda r: abs(r[i["e"]] - r[i["e_hat"]])),
            track_s=most(lambda r: abs(r[i["s_hat"]] - r[i["s_c"]])),
            track_u=most(lambda r: abs(r[i["u_hat"]] - r[i["u_c"]])),
            track_d=most(lambda r: abs(r[i["d_hat"]])),
            track_e=most(lambda r: abs(r[i["e_hat"]])),
            true_s=most(lambda r: abs(r[i["s"]] - r[i["s_c"]])),
            true_u=most(lambda r: abs(r[i["u"]] - r[i["u_c"]])),
            true_d=most(lambda r: abs(r[i["d"]])),
            true_e=most(lambda r: abs(r[i["e"]])),
            outside=most(lambda r: max(0.0, abs(r[i["s"]] - r[i["s_c"]]) - DS,
                                       (abs(r[i["u"]] - r[i["u_c"]]) - DU) * TAU_U,
                                       (abs(r[i["d"]]) - DD) / D_SCALE,
                                       (abs(r[i["e"]]) - DE) * RHO_E)),
            speed=most(lambda r: r[i["u"]]), commanded=most(lambda r: r[i["u_c"]]),
            command=most(lambda r: r[i["v"]]),
            alarms=list(record.alarms), restarts=list(record.restarts),
            touches=self.world.touches[:1])
