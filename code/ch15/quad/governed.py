"""
The modules of the governed Go2 (Section 15.2): the robot stands, shifts
its centre of mass, and lifts its front-right foot (FR) for an operation.

The control package go2.stance plans the centre of mass's motion along the
shift line of reduced.py, from its home place towards the incentre of the
triangle of the other three feet (FL, RL, RR), and the operation that lifts
FR: unload it, lift it, hold it up, lower it, touch down and load it again
(reduced.PHASES, three seconds). Its parts:
  * segments: one period T of 0.2 s, in which the centre of mass's
    acceleration along the line moves from the one in force to the commanded
    one along a half-cosine, as stance.py tracks it; during the operation, a
    segment also continues the operation's profile for FR from where the
    last one left it, and the centre of mass, brought to rest, is held;
  * funnels: the commanded motion, with the measured centre of mass within
    DC of its commanded place, a disc in the horizontal plane, and FR
    within DH of its target height (ours, measured in Section 15.2);
  * two families of constraints, as in Section 4.2.7. The four feet (f4):
    the zero-moment point inside the rectangle of the four feet, checked at
    its floor FLOOR4. The lift (f3): the zero-moment point inside the
    triangle of FL, RL and RR, enlarged by C3 while FR carries its share of
    the load, and the triangle itself from the period in which the
    operation begins until FR carries its share again. The levels carry
    f3: a lift consumes exactly C3 of it (Assumption 4.13). Each reserve
    is the least distance over a segment of the zero-moment point of the
    cart-table model (reduced.py) from its polygon's edges, less ALLOWANCE
    for what the model leaves out and K_PUSH per newton of the label's push
    on the trunk (ours, measured);
  * the test: those reserves, the acceleration within A_MAX, the ground's
    friction (at least MU) for the motion and the push, and the joints'
    static torques with TORQUE_FACTOR for the motion within the motors'
    limits; a profile's entry n is the least level of the pairs from n on
    and of the rest;
  * the rest: the four feet at rest, powered, FR loaded, its level that of
    f3 under the global capability; that it holds the robot in its funnel
    for as long as it is repeated is an assumption of this package, which
    needs power and control to last (Section 15.2);
  * the recovery continuation: the fastest stop of the centre of mass within
    A_MAX, or, at a requirement's level, the fastest motion to rest at a
    place where the rest holds that level; during the operation, the rest
    of the operation, the centre of mass held;
  * the generator: the proposal's command mixed with the fallback's by the
    weights ALPHAS, each with the recovery at the floor and at a pending
    requirement's level, and those recoveries alone; when the task asks for
    the operation, its one candidate begins it with the fallback's motion of
    the centre of mass, so that its zero-moment point is the fallback's and
    the operation consumes exactly C3 (Assumption 4.13); during the
    operation, its one candidate continues it, the centre of mass brought to
    rest as fast as it can be and held;
  * after an alarm, the rest feedback stops the centre of mass and holds
    it; if the operation has begun, it continues it, and if FR is held up,
    it lowers FR at once. A restart is offered at rest outside the
    operation.

The touchdown is a contact that the reduced model does not describe: the
test checks the segments in which FR may meet the floor under the
assumption, stated in Section 15.2, that the floor under FR lies within the
band of where the operation's profile expects it, and that from FR's first
contact until FR carries its share again the state stays in the funnel.
The assumption bounds neither when the contact comes nor how fast: a push
within the label can bring it into the lowering and make it faster than
the slow descent's commanded speed. That assumption is not a result of
this book.

A label is a bound on the horizontal push that the world may exert on the
trunk during the period (reduced.py); the global capability is GLOBAL.

The robot adapter measures the simulated state exactly: the centre of
mass's place along the line and across it, FR's height, and the commanded
speed, acceleration and operation, which the controller knows. The world
adapter go2.floor runs sim.Machine under each period's law.
"""
import math
import os
import sys
from collections import namedtuple

import numpy as np
from scipy.optimize import linprog

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "ch10"))
from allovi import (NEG_INF, Actuation, Bound, Candidate, Profile, Quantity,        # noqa: E402
                    Requirement, Schema, ScalarLevels, Why, register)

from . import reduced as rm                                                       # noqa: E402
from . import sim, support                                                        # noqa: E402

T = rm.T
A_MAX = 0.25             # [m/s^2] the largest commanded acceleration of the centre of mass (ours)
FLOOR = 0.020            # [m] the floor of f3, the lift's family (ours)
FLOOR4 = 0.020           # [m] the floor, and single level, of f4, the four feet's family (ours)
C3 = 0.060               # [m] the fourth foot's credit to f3 (ours): what a lift consumes
GRID = 0.005             # [m] the levels' grid
LEVELS = ScalarLevels(floor=FLOOR, step=GRID, top=0.300)
ALLOWANCE = 0.005        # [m] what the cart-table model leaves out (ours, measured)
K_PUSH = 0.0030          # [m/N] the centre of pressure's move per newton of push (ours, measured)
MU = 0.5                 # the ground's friction, at least (declared)
TORQUE_FACTOR = 1.5      # the motion's part of the torques, counted this many times (ours)
DC = 0.008               # [m] the funnel's bound on the centre of mass (ours, measured)
DH = 0.020               # [m] the funnel's bound on FR's height (ours, measured)
ALPHAS = (1.0, 0.75, 0.5, 0.25)
S_MIN, S_MAX = -0.030, rm.S_INCENTRE     # [m] the line's declared extent
AT_REST = 1e-9
N_MAX = 40               # periods: the longest motion that the recovery plans
HOLD_START = sum(n for _, n in rm.PHASES[:2]) * T     # [s] of the operation: FR held up from
LOWER_START = sum(n for _, n in rm.PHASES[:3]) * T    # and lowered from

Label = namedtuple("Label", "push")
"""A capability: the largest horizontal push on the trunk [N] during the period."""
GLOBAL = Label(10.0)     # the global capability (ours)

STATE = Schema(
    "Go2 standing, its centre of mass on the shift line",
    (Quantity("s", "m", "shift line"),        # the centre of mass along the line (measured)
     Quantity("e", "m", "across the line"),   # and across it (measured)
     Quantity("v", "m/s", "shift line"),      # the commanded speed
     Quantity("a", "m/s^2", "shift line"),    # the commanded acceleration
     Quantity("h", "m", "z")),                # FR's height above its stance place (measured)
    norm="max(|s| + |e| + K |a|, |h|), K = z_c/g")

ACTUATION = Actuation(
    inputs=(Quantity("a_c", "m/s^2", "shift line"), Quantity("load_fr", "1", ""),
            Quantity("h_fr", "m", "z")),
    bounds=(Bound("a_c", low=-A_MAX, high=A_MAX), Bound("load_fr", low=0.0, high=1.0)),
    hold="the centre of mass's acceleration follows each period's command along a "
         "half-cosine over the period; FR's load share and height follow the operation's "
         "profile",
    meaning="the stance controller's planned acceleration of the centre of mass, FR's share "
            "of the load and FR's target height")


# ----------------------------------------------------------------------------
# States, segments and plans
# ----------------------------------------------------------------------------
State = namedtuple("State", "s v a op")
"""A commanded state at a period boundary: the centre of mass's place on the
line, its speed and acceleration, and the operation: 'before', 'after', or
the time since it began [s]. It is also the initial set, the funnel's
centre."""

Rest = namedtuple("Rest", "s op")
"""The rest: the four feet at rest, powered, the centre of mass at s; op is
'before' or 'after' the operation."""


def _next_op(op, dt):
    if not isinstance(op, float):
        return op
    u = op + dt
    return "after" if u >= rm.OP_TIME - 1e-9 else u


class Segment(namedtuple("Segment", "s0 v0 a0 a1 op0")):
    """One period: the acceleration from a0 to a1 along a half-cosine, from
    (s0, v0); op0 the operation's state at the start ('before', 'after', or
    the time since it began)."""
    __slots__ = ()

    def accel(self, tau):
        return rm.accel(self.a0, self.a1, tau)

    def speed(self, tau):
        return rm.speed(self.v0, self.a0, self.a1, tau)

    def position(self, tau):
        return rm.position(self.s0, self.v0, self.a0, self.a1, tau)

    def end(self):
        return State(self.position(T), self.speed(T), self.a1, _next_op(self.op0, T))

    def op_time(self, tau):
        return self.op0 + tau if isinstance(self.op0, float) else None

    def fr(self, tau, ground=0.0):
        """FR's load share and target height at tau."""
        u = self.op_time(tau)
        if u is None or u >= rm.OP_TIME - 1e-12:
            return 1.0, 0.0
        return rm.fr_command(u, ground)

    def three(self):
        return isinstance(self.op0, float)

    def resting(self):
        return abs(self.v0) <= AT_REST and abs(self.a0) <= 1e-12 and abs(self.a1) <= 1e-12


class Plan:
    """A labelled plan: its first pair executes in period j0."""

    def __init__(self, j0, pairs, rest):
        self.j0, self.pairs, self.rest = j0, list(pairs), rest

    def shift(self):
        if not self.pairs:
            return self
        return Plan(self.j0 + 1, self.pairs[1:], self.rest)

    def head(self):
        if self.pairs:
            return self.pairs[0][0]
        return Segment(self.rest.s, 0.0, 0.0, 0.0, self.rest.op)


def region(seg):
    """The positions of the centre of mass that the segment covers [m]."""
    q0, q1 = seg.s0, seg.position(T)
    return min(q0, q1) - DC, max(q0, q1) + DC


def level_of(reserve):
    """The largest level of the grid that the reserve reaches, or NEG_INF."""
    if reserve < FLOOR - 1e-12:
        return NEG_INF
    n = math.floor((reserve - FLOOR) / GRID + 1e-9)
    return round(FLOOR + n * GRID, 9)


# ----------------------------------------------------------------------------
# The test
# ----------------------------------------------------------------------------
_TORQUES = {}


def static_shares(s, three):
    """The joints' static torques at rest with the centre of mass at s on the
    line, on four feet or on three with FR held up, as shares of the motors'
    limits, the largest of each motor's (stance.py's force distribution and
    the legs' Jacobians; cached every millimetre)."""
    key = (round(s, 3), bool(three))
    if key not in _TORQUES:
        _TORQUES[key] = sim.static_shares(s, three)
    return _TORQUES[key]


def pair_check(seg, label):
    """The pair's reserve of f3 [m] and the first limit that fails, or None."""
    if max(abs(seg.a0), abs(seg.a1)) > A_MAX + 1e-12:
        return NEG_INF, "the acceleration exceeds A_MAX"
    lo, hi = region(seg)
    if lo + DC < S_MIN - 1e-12 or hi - DC > S_MAX + 1e-12:
        return NEG_INF, "the centre of mass leaves the line's declared extent"
    q_lo, q_hi = rm.zmp_range(seg.s0, seg.v0, seg.a0, seg.a1)
    allowance = ALLOWANCE + K_PUSH * label.push
    d_tri, d_rect = rm.distances(q_lo, q_hi)      # the funnel adds a disc of radius DC
    if seg.three():
        reserve = d_tri - DC - allowance
    else:
        reserve = C3 + d_tri - DC - allowance
        if d_rect - DC - allowance < FLOOR4 - 1e-12:
            return NEG_INF, "the four feet's reserve lies below its floor"
    a = max(abs(seg.a0), abs(seg.a1))
    if (rm.MASS * a + label.push) / (rm.MASS * rm.G) > MU:
        return NEG_INF, "the ground's friction does not suffice"
    for s in (seg.s0, seg.position(T)):
        if TORQUE_FACTOR * static_shares(s, seg.three()) > 1.0:
            return NEG_INF, "a motor's torque limit"
    return reserve, None


def rest_reserve(rest, label=GLOBAL):
    """The rest's reserve of f3 under the label, or NEG_INF if f4 fails."""
    allowance = ALLOWANCE + K_PUSH * label.push
    d_tri, d_rect = rm.distances(rest.s, rest.s)
    if d_rect - DC - allowance < FLOOR4 - 1e-12 or not (S_MIN <= rest.s <= S_MAX):
        return NEG_INF
    if TORQUE_FACTOR * static_shares(rest.s, False) > 1.0:
        return NEG_INF
    return C3 + d_tri - DC - allowance


def rest_place(level, label=GLOBAL):
    """The nearest place to home on the line, towards the incentre, at which
    the rest holds the level, or None."""
    lo, hi = 0.0, S_MAX
    if rest_reserve(Rest(hi, "before"), label) < level - 1e-12:
        return None
    if rest_reserve(Rest(lo, "before"), label) >= level - 1e-12:
        return lo
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if rest_reserve(Rest(mid, "before"), label) >= level - 1e-12:
            hi = mid
        else:
            lo = mid
    return math.ceil(hi * 1e4) / 1e4


# ----------------------------------------------------------------------------
# Motions of the centre of mass: the fastest within A_MAX
# ----------------------------------------------------------------------------
def _affine(s0, v0, a0, n):
    """s_n and v_n after n periods as affine functions of the end
    accelerations x_1..x_n: (constant, coefficients)."""
    def run(x):
        s, v, a = s0, v0, a0
        for a1 in x:
            s, v, a = rm.position(s, v, a, a1, T), rm.speed(v, a, a1, T), a1
        return s, v
    c_s, c_v = run(np.zeros(n))
    A_s, A_v = np.zeros(n), np.zeros(n)
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        s, v = run(e)
        A_s[j], A_v[j] = s - c_s, v - c_v
    return c_s, A_s, c_v, A_v


def motion(s0, v0, a0, lo=None, hi=None):
    """The end accelerations of the fewest periods that bring the centre of
    mass from (s0, v0, a0) to rest, within A_MAX, at a place in [lo, hi]
    (None: anywhere), with the least sum of their sizes; [] if it is at rest
    there."""
    if abs(v0) <= AT_REST and abs(a0) <= 1e-12 and (lo is None or s0 >= lo - 1e-9) \
            and (hi is None or s0 <= hi + 1e-9):
        return []
    for n in range(1, N_MAX + 1):
        c_s, A_s, c_v, A_v = _affine(s0, v0, a0, n)
        # variables p, m >= 0 with x = p - m; x_n = 0
        cost = np.ones(2 * n)
        A_eq = [np.concatenate([A_v, -A_v])]
        b_eq = [-c_v]
        last = np.zeros(2 * n)
        last[n - 1], last[2 * n - 1] = 1.0, -1.0
        A_eq.append(last)
        b_eq.append(0.0)
        A_ub, b_ub = [], []
        if lo is not None:
            A_ub.append(-np.concatenate([A_s, -A_s]))
            b_ub.append(c_s - lo)
        if hi is not None:
            A_ub.append(np.concatenate([A_s, -A_s]))
            b_ub.append(hi - c_s)
        res = linprog(cost, A_ub=np.array(A_ub) if A_ub else None,
                      b_ub=np.array(b_ub) if b_ub else None, A_eq=np.array(A_eq),
                      b_eq=np.array(b_eq), bounds=[(0.0, A_MAX)] * (2 * n), method="highs")
        if res.status == 0:
            x = res.x[:n] - res.x[n:]
            x[np.abs(x) < 1e-12] = 0.0
            x[-1] = 0.0
            return [float(min(A_MAX, max(-A_MAX, a))) for a in x]
    raise RuntimeError("no motion to rest within N_MAX periods")


# ----------------------------------------------------------------------------
# The capability space and the feedback law
# ----------------------------------------------------------------------------
class Capability:
    """Labels are pushes' bounds; c is at least as informative as d if its
    bound is at most d's."""
    NAME = "go2.push"
    global_capability = GLOBAL

    def informative(self, c, d):
        return c.push <= d.push + 1e-12

    def labelling(self, capability):
        return lambda j, segment: capability


class Law:
    """The feedback law of one period: the segment that the stance controller
    tracks. Called with the time within the period, it returns the planned
    acceleration of the centre of mass along the line, FR's load share and
    FR's target height (ACTUATION)."""

    def __init__(self, segment, kind="segment", ground=0.0):
        self.segment, self.kind, self.ground = segment, kind, ground

    def __call__(self, tau, state=None):
        load, h = self.segment.fr(tau, self.ground)
        return self.segment.accel(tau), load, h


# ----------------------------------------------------------------------------
# The control package
# ----------------------------------------------------------------------------
@register
class StancePackage:
    """Segments of the centre of mass's motion and of the operation, the
    generator, the test, the recovery continuation, the rest and the rest
    feedback (module docstring)."""
    ROLE, NAME, CONTRACT = "control", "go2.stance", "1.0"
    STATE, ACTUATION = STATE, ACTUATION

    def __init__(self, ground=0.0):
        self.capabilities, self.levels = Capability(), LEVELS
        self.ground = ground      # where the package believes the floor under FR, above it [m]

    # --- plans and the test ------------------------------------------------
    def initial_plan(self, measured, bound):
        if bound > 1e-12:
            raise ValueError("this package's funnels assume an exact measurement")
        if abs(measured.v) > AT_REST or abs(measured.a) > 1e-12 or isinstance(measured.op, float):
            raise ValueError("this package certifies an initial plan only at rest, on four feet")
        return Plan(0, [], Rest(measured.sc, measured.op)), State(measured.sc, 0.0, 0.0, measured.op)

    def initial_set(self, plan):
        return plan.head().end()

    def head(self, plan):
        if plan.pairs:
            return plan.pairs[0]
        return plan.head(), GLOBAL

    def profile(self, plan, initial):
        if not _starts_at(plan, initial):
            return Profile.uncertified()
        rr = rest_reserve(plan.rest)
        rest = level_of(rr) if rr != NEG_INF else NEG_INF
        entries, running = [], rest
        for seg, label in reversed(plan.pairs):
            r, failure = pair_check(seg, label)
            level = NEG_INF if failure else level_of(r)
            running = NEG_INF if (running == NEG_INF or level == NEG_INF) else min(running, level)
            entries.append(running)
        entries.reverse()
        return Profile(entries + [rest])

    def explain(self, plan, initial):
        if not _starts_at(plan, initial):
            return [Why("the plan does not start at the initial set", "pair")]
        out = []
        for i, (seg, label) in enumerate(plan.pairs):
            r, failure = pair_check(seg, label)
            out.append(Why(f"pair {i}: {failure}", "pair") if failure else
                       Why(f"pair {i}: reserve {r:.4f} m"))
        out.append(Why(f"the rest at {plan.rest.s:.4f} m, reserve "
                       f"{rest_reserve(plan.rest):.4f} m"))
        return out

    # --- the generator -------------------------------------------------------
    def generate(self, step, plan, initial):
        st, labelling, k = initial, step.labelling, step.k
        levels = [FLOOR]
        if step.requirement.level > FLOOR + 1e-9:
            levels.append(step.requirement.level)
        begin = step.operation == "lift-fr" and st.op == "before"
        if begin:
            tail, rest = self.begin_operation(labelling, k + 1, plan.shift(), st)
            return [Candidate(Plan(k + 1, tail, rest), 1.0)]
        if isinstance(st.op, float):
            tail, rest = self.recovery(labelling, k + 1, st, FLOOR)
            return [Candidate(Plan(k + 1, tail, rest), 1.0)]
        if step.proposal is None:
            return self._recovery_only(labelling, k, st, levels)
        fallback = plan.shift()
        a_fb = fallback.head().a1 if fallback.pairs else 0.0
        heads = {}
        for alpha in ALPHAS:
            a1 = self.command(step.proposal, alpha, a_fb, st)
            key = round(a1, 12)
            if key not in heads:
                heads[key] = (a1, alpha)
        out = []
        for a1, alpha in heads.values():
            head = Segment(st.s, st.v, st.a, a1, st.op)
            for level in levels:
                tail, rest = self.recovery(labelling, k + 2, head.end(), level)
                out.append(Candidate(Plan(k + 1, [(head, labelling(k + 1, head))] + tail, rest),
                                     alpha))
        return out + self._recovery_only(labelling, k, st, levels)

    def begin_operation(self, labelling, j, fallback, st):
        """The candidate that begins the operation from the commanded state
        st, from period j on: the fallback's commanded accelerations of the
        centre of mass, unchanged, with the operation's profile for FR, and
        then the centre of mass held where the fallback comes to rest until
        the operation ends. Its zero-moment point is the fallback's at every
        instant, so its reserve of f3 is the fallback's less C3, exactly, and
        its acceleration, friction and rest are the fallback's."""
        s, v, a, op = st.s, st.v, st.a, 0.0
        pairs = []
        for seg_fb, _ in fallback.pairs:
            seg = Segment(s, v, a, seg_fb.a1, op)
            pairs.append((seg, labelling(j, seg)))
            s, v, a, op = seg.end()
            j += 1
        while isinstance(op, float):
            seg = Segment(s, v, a, 0.0, op)
            pairs.append((seg, labelling(j, seg)))
            s, v, a, op = seg.end()
            j += 1
        return pairs, Rest(s, op)

    def command(self, proposal, alpha, a_fb, st):
        """The head's commanded acceleration: the first of the fastest motion
        to rest at the proposal's place, mixed with the fallback's command by
        alpha, within A_MAX."""
        target = min(S_MAX, max(S_MIN, float(proposal)))
        x = motion(st.s, st.v, st.a, target, target)
        a_q = x[0] if x else 0.0
        a1 = min(A_MAX, max(-A_MAX, alpha * a_q + (1.0 - alpha) * a_fb))
        return 0.0 if abs(a1) < 1e-12 else a1

    def _recovery_only(self, labelling, k, st, levels):
        out = []
        for level in levels:
            tail, rest = self.recovery(labelling, k + 1, st, level)
            if not tail:
                hold = Segment(st.s, 0.0, 0.0, 0.0, st.op)
                tail = [(hold, labelling(k + 1, hold))]
            out.append(Candidate(Plan(k + 1, tail, rest), 0.0))
        return out

    def recovery(self, labelling, j, st, level):
        """The recovery continuation from the commanded state st from period j
        on (module docstring): its pairs and its rest."""
        s, v, a, op = st
        if isinstance(op, float):
            accs = motion(s, v, a)                 # brought to rest, then held
        elif level > FLOOR + 1e-9:
            place = rest_place(level)
            if place is None or s >= place - 1e-9:
                accs = motion(s, v, a)
            else:
                accs = motion(s, v, a, place, place + 0.002)
        else:
            accs = motion(s, v, a)
        pairs = []
        for a1 in accs:
            seg = Segment(s, v, a, a1, op)
            pairs.append((seg, labelling(j, seg)))
            s, v, a, op = seg.end()
            j += 1
        while isinstance(op, float):
            seg = Segment(s, 0.0, 0.0, 0.0, op)
            pairs.append((seg, labelling(j, seg)))
            s, v, a, op = seg.end()
            j += 1
        return pairs, Rest(s, op)

    # --- execution and the alarm check ---------------------------------------
    def law(self, plan):
        return Law(plan.head(), ground=self.ground)

    def distance(self, measured, initial):
        """How far the measurement lies outside the funnel at the initial set
        [m]: the centre of mass beyond DC of its commanded place, in the
        horizontal plane (the disc that the test takes off), or FR beyond DH of
        its target height."""
        h = Segment(initial.s, initial.v, initial.a, initial.a, initial.op).fr(0.0, self.ground)[1]
        return max(0.0, math.hypot(measured.s - initial.s, measured.e) - DC,
                   abs(measured.h - h) - DH)

    # --- after an alarm ------------------------------------------------------
    def rest_law(self, measured):
        op = measured.op
        if isinstance(op, float) and HOLD_START - 1e-9 <= op < LOWER_START - 1e-9:
            op = LOWER_START          # FR is held up: lower it now
        st = State(measured.sc, measured.v, measured.a, op)
        tail, _ = self.recovery(lambda j, seg: GLOBAL, 0, st, FLOOR)
        if tail:
            return Law(tail[0][0], "rest", ground=self.ground)
        return Law(Segment(measured.sc, 0.0, 0.0, 0.0, measured.op), "rest", ground=self.ground)

    def reach(self, measured, bound):
        if bound > 1e-12:
            return None, "the funnels assume an exact measurement"
        if abs(measured.v) > AT_REST or abs(measured.a) > 1e-12 or isinstance(measured.op, float):
            return None, "no restart candidate while the centre of mass moves or FR is lifted"
        return State(measured.s, 0.0, 0.0, measured.op), ""

    def restart(self, step, reach):
        return self.generate(step, Plan(step.k + 1, [], Rest(reach.s, reach.op)), reach)

    # --- diagnostic descriptions ---------------------------------------------
    def describe_plan(self, plan):
        h = plan.head()
        label = plan.pairs[0][1] if plan.pairs else GLOBAL
        return dict(a1=h.a1, op=h.op0, push=label.push, pairs=len(plan.pairs), rest=plan.rest.s)

    def describe_set(self, initial):
        return dict(s_c=initial.s, v_c=initial.v, a_c=initial.a, op=str(initial.op))


def _starts_at(plan, initial):
    """The plan's head starts at the initial set's commanded state, the
    operation as it is there or beginning it."""
    h = plan.head()
    same = (abs(h.s0 - initial.s) <= 1e-9 and abs(h.v0 - initial.v) <= 1e-9
            and abs(h.a0 - initial.a) <= 1e-9)
    op = (h.op0 == initial.op) or (h.op0 == 0.0 and initial.op == "before")
    return same and op


# ----------------------------------------------------------------------------
# The proposal method
# ----------------------------------------------------------------------------
@register
class Operator:
    """Asks for the centre of mass at home, and, while the operation goes on,
    where it is: it knows nothing of the triangle that the lift needs."""
    ROLE, NAME, CONTRACT = "proposal", "go2.operator", "1.0"

    def __init__(self, home=0.0):
        self.home = home

    def propose(self, observation):
        m = observation.measured
        return m.sc if isinstance(m.op, float) else self.home


# ----------------------------------------------------------------------------
# Capability predictors
# ----------------------------------------------------------------------------
@register
class Fixed:
    """The global capability everywhere."""
    ROLE, NAME, CONTRACT, SPACE = "capability", "go2.fixed", "1.0", Capability.NAME
    horizon = 0

    def observe(self, k, before, after):
        pass

    def labels(self, observation):
        return lambda j, segment: GLOBAL


# ----------------------------------------------------------------------------
# The operation and the demand predictor that reads its schedule
# ----------------------------------------------------------------------------
@register
class LiftFR:
    """The operation that lifts FR, due in period 'onset' and needing 'level'
    of f3 (FLOOR + C3). It begins if the plan in force during its onset
    period holds that level (Assumption 4.13): from then on the generator is
    asked to begin it, and every candidate does. Otherwise an overload
    occurs, and the operation is postponed: to the forecast onset if that is
    still ahead, else by 'postpone' periods, announced (Section 8.2.3).
    announce=None: no announcement before the onset."""
    ROLE, NAME, CONTRACT = "task", "go2.lift-fr", "1.0"

    def __init__(self, level=FLOOR + C3, announce=2, onset=15, postpone=5):
        self.level, self.announce, self.onset = round(level, 9), announce, onset
        self.predicted, self.postpone = onset, postpone
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
        return "lift-fr" if self.begun is not None else None


@register
class LiftSchedule:
    """The demand predictor: from the announcement on, the operation's level
    for the onset that the task forecasts."""
    ROLE, NAME, CONTRACT = "demand", "go2.lift-schedule", "1.0"

    def __init__(self, task):
        self.task = task

    def requirement(self, observation):
        task, k = self.task, observation.k
        if task.pending() and task.announce is not None and k >= task.announce:
            return Requirement(task.level, max(task.predicted - k - 1, 0))
        return Requirement(FLOOR, 0)


# ----------------------------------------------------------------------------
# Robot and world adapters
# ----------------------------------------------------------------------------
Measured = namedtuple("Measured", "s e v a h sc op")
"""The measurement at a period boundary: the centre of mass along the line
and across it, the commanded speed and acceleration, FR's height, and the
commanded place and operation, which the controller knows."""


class StandState(tuple):
    """The world's state at a period boundary: (s, e, v, a, h) as declared,
    with the commanded state (commanded), the simulated centre of pressure's
    distance inside the current support polygon (margin) and the time."""

    def __new__(cls, values, commanded, margin, t):
        obj = super().__new__(cls, values)
        obj.commanded, obj.margin, obj.t = commanded, margin, t
        return obj


@register
class Robot:
    """The state and the actuation of the robot; the measurement is the
    simulated state, exact."""
    ROLE, NAME, CONTRACT = "robot", "go2.robot", "1.0"
    STATE, ACTUATION = STATE, ACTUATION
    measurement_bound = 0.0

    def measure(self, state):
        c = state.commanded
        return Measured(state[0], state[1], c.v, c.a, state[4], c.s, c.op)

    def describe_state(self, measured):
        return dict(s=measured.s, e=measured.e, h=measured.h, op=str(measured.op))


@register
class Floor:
    """The simulated Go2 (sim.Machine) on its floor, with the world's pushes
    on the trunk. Each period it runs the stance controller on the law's
    segment; FR's profile is the one that the control package believes."""
    ROLE, NAME, CONTRACT = "world", "go2.floor", "1.0"
    STATE, ACTUATION = STATE, ACTUATION

    def __init__(self, pushes=(), timestep=sim.TIMESTEP):
        self.pushes, self.timestep = list(pushes), timestep

    def reset(self):
        self.machine = sim.Machine(self.timestep, pushes=self.pushes, home=rm.COM0)
        self.commanded = State(0.0, 0.0, 0.0, "before")
        self.bounds = [0]
        self.laws = []
        return self._state()

    def _state(self):
        c = self.commanded
        com = self.machine.com()[:2] - rm.COM0
        s = float(com @ rm.U)
        e = float(com @ np.array([-rm.U[1], rm.U[0]]))
        h = self.machine.fr_height()
        log = self.machine.log
        if log["t"]:
            cop = np.array([log["copx"][-1], log["copy"][-1]])
            poly = rm.TRI if log["f1"][-1] <= 1e-6 else rm.RECT
            margin = support.signed_distance(poly, cop)
        else:
            margin = support.signed_distance(rm.RECT, self.machine.com()[:2])
        return StandState((s, e, c.v, c.a, h), c, margin, self.machine.time())

    def step(self, state, law):
        seg = law.segment
        u = rm.U
        jump = seg.s0 - self.commanded.s
        if abs(jump) > 1e-9:
            # a law that starts elsewhere than the last command, as a restart from a
            # measurement does: re-anchor the controller's correction, so that the
            # base's target does not jump
            self.machine.corr -= jump * u

        def plan(tau):
            load, h = law(tau)[1:]
            return (rm.COM0 + seg.position(tau) * u, seg.accel(tau) * u, load, h)
        self.machine.advance(T, plan)
        self.commanded = seg.end()
        self.bounds.append(self.machine.steps())
        self.laws.append(law)
        return self._state()

    # --- the oracle, for the checks of Section 10.3 (simulation only) ---------
    def admits(self, label, segment, k):
        t0, t1 = k * T, (k + 1) * T
        ts = np.linspace(t0, t1, 41)
        worst = max(sum(p.at(t) for p in self.pushes) for t in ts) if self.pushes else 0.0
        return worst <= label.push + 1e-9

    def clearance(self, state):
        return state.margin


# ----------------------------------------------------------------------------
# Without the governor
# ----------------------------------------------------------------------------
def ungoverned(world, proposal, task, periods, ground=0.0):
    """The proposal executed directly: each period, the first command of the
    fastest motion to the proposal's place; the operation begun in its onset
    period, whatever the reserve, the centre of mass held during it. Returns
    the states at the period boundaries and the segments."""
    from allovi import Observation
    robot = Robot()
    x = world.reset()
    states, segs = [x], []
    for k in range(periods):
        m = robot.measure(x)
        q = proposal.propose(Observation(k, k * T, m, NEG_INF, 0, False))
        task.begin_period(k, math.inf)
        c = x.commanded
        op = c.op
        if task.begun is not None and k > task.begun and op == "before":
            op = 0.0
        if isinstance(op, float):
            x_ = motion(c.s, c.v, c.a)
            a1 = x_[0] if x_ else 0.0
        else:
            x_ = motion(c.s, c.v, c.a, float(q), float(q))
            a1 = x_[0] if x_ else 0.0
        seg = Segment(c.s, c.v, c.a, a1, op)
        x = world.step(x, Law(seg, ground=ground))
        states.append(x)
        segs.append(seg)
    return states, segs
