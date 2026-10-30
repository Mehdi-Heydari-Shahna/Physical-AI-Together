"""The control package of the cart: segments of one constant command, the
generator and the sound test of Chapter 3, the rest segment and the rest
feedback (Sections 3.1, 4.2 and 8.3)."""
from collections import namedtuple
from fractions import Fraction as Fr

from allovi import NEG_INF, Candidate, Profile, Why, register
from allovi.checks import FAILED, PASSED, Result

from .model import LEVELS, MODEL, ce

class Law(namedtuple("Law", "u ell kind")):
    """A feedback law for one period: a constant commanded acceleration u, the
    commanded reach ell (None: keep the current reach), and its kind:
    'segment' for the head of a plan, 'rest' for the rest feedback after an
    alarm, which commands u until the cart is at rest and 0 from then on.
    Like every law of the contract, it can be called with the time within the
    period and the state, and it returns the input (u_a, u_l) in the declared
    actuation; the track of the cart uses its fields instead, to compute the
    motion exactly."""
    __slots__ = ()

    def __call__(self, tau, state):
        ell = state[2] if self.ell is None else self.ell
        if self.kind == "rest" and state[1] <= 0:
            return (Fr(0), ell)
        return (self.u, ell)

NO_REACH = ("no restart candidate: the cart moves towards the wall under maximal "
            "braking, so the states that the rest feedback can reach differ between "
            "worlds")


class Braking:
    """The capability space of the cart: a label is the braking, in m/s^2,
    that the ground provides at least on the region of a segment. More
    braking admits fewer worlds, so c is at least as informative as d if
    c >= d. The global capability is 0.5 m/s^2 everywhere."""
    NAME = "cart.braking"
    global_capability = ce.B_GLOBAL

    def informative(self, c, d):
        return c >= d

    def labelling(self, capability):
        return lambda j, segment: capability


class _Labels:
    """Presents a labelling map lambda_k(j, segment) to the generator of
    Chapter 3, which asks predictor.label(k, j, segment).

    Argument: labelling, the map step.labelling of the current period, a
    function of a period j and a segment (a ce.Segment) that returns the
    label: the braking [m/s^2] assumed on the segment's region in period j.
    label(k, j, segment) returns labelling(j, segment) and ignores k. The
    map is valid only during the period in which the core issued it, so an
    instance must not be kept for a later period. The class is private to
    the cart's package: the contract does not cover it, and a later release
    may rename it. ce.braking_step, ce.backward_step, ce.brake_to_rest,
    ce.recovery and ce.generate accept any object with this method, so a
    module can define the same three lines instead
    (../templates/HELPERS.md).

    >>> labels = _Labels(lambda j, segment: Fr(1, 2))
    >>> labels.label(7, 8, None)
    Fraction(1, 2)
    """

    def __init__(self, labelling):
        self.labelling = labelling

    def label(self, k, j, segment):
        return self.labelling(j, segment)


def _starts_at(plan, initial):
    """Condition (P1) for a single-state initial set: the head starts there."""
    p, v, _ = initial                 # the reach is set by each segment for its period
    if plan.pairs:
        seg = plan.pairs[0][0]
        return (seg.p0, seg.v0) == (p, v)
    return (plan.p_rest, Fr(0)) == (p, v)


@register
class ConstantCommand:
    """Generator, test, rest segment and rest feedback of the cart.

    To extend it, as bench.constant-command of Chapter 16 and
    reader.full-reverse of Chapter 17 do, subclass it, give the subclass a
    NAME of its own (ROLE and CONTRACT are inherited) and register it. What
    its methods exchange: a plan is a ce.Plan; the initial set is a single
    state (p, v, l), exact, in m, m/s and m, the end of the funnel of the
    head of the plan in force; a candidate is an allovi.Candidate whose plan
    starts at the initial set in period step.k + 1. The test, profile(),
    refuses a plan whose head does not start at the initial set (condition
    (P1)); it relies on the chaining that ce.Plan states, which every added
    candidate must meet. ../templates/HELPERS.md documents these helpers,
    with examples."""
    ROLE, NAME, CONTRACT = "control", "cart.constant-command", "1.0"
    STATE, ACTUATION = MODEL.state, MODEL.actuation

    def __init__(self, rest="maximal"):
        if rest not in ("maximal", "global"):
            raise ValueError("rest feedback: 'maximal' braking or braking at the "
                             "'global' capability")
        self.capabilities, self.levels, self.rest = Braking(), LEVELS, rest

    # --- plans and the test ------------------------------------------------
    def initial_plan(self, measured, bound):
        p, v, l = measured
        if v != 0 or bound != 0:
            raise ValueError("this package certifies an initial plan only at rest "
                             "and with an exact measurement")
        return ce.Plan(0, [], p, l), (p, Fr(0), l)

    def initial_set(self, plan):
        """The end of the funnel of the head: a single state."""
        if plan.pairs:
            seg = plan.pairs[0][0]
            return (seg.p1, seg.v1, plan.ell)
        return (plan.p_rest, Fr(0), plan.ell)

    def generate(self, step, plan, initial):
        """The candidates of period step.k (Definition 4.14), each an
        allovi.Candidate that starts at the initial set (p, v, l): the blends
        of the proposed command (ce.proposed_command) with the head of the
        fallback plan.shift(), for the ambitions 1, 3/4, 1/2, 1/4 and 0, each
        followed by the recovery continuation, then the recovery
        continuation alone with ambition 0 (ce.generate); the continuation
        alone if step.proposal is None. A requirement above the floor sets
        the place, 25 m less its level, at or before which every
        continuation brings the cart to rest; every segment has the reach
        step.operation, or 0 if it is None."""
        p, v, _ = initial
        req = step.requirement.level
        target = ce.WALL - req if req > ce.EPS_MIN else None
        ell = Fr(0) if step.operation is None else step.operation
        labels = _Labels(step.labelling)
        if step.proposal is None:                      # recovery candidates only
            return [_recovery_only(labels, step.k, p, v, target, ell)]
        u_q = ce.proposed_command(step.proposal, (p, v))
        return [Candidate(c, a) for c, a in
                ce.generate(labels, step.k, (p, v), u_q, plan.shift(), target, ell)]

    def profile(self, plan, initial):
        """The verdicts of the test of Chapter 3 as a profile; a plan that does
        not start at the initial set, condition (P1), is not certified."""
        if not _starts_at(plan, initial):
            return Profile.uncertified()
        return Profile(NEG_INF if x is ce.NEG_INF else x for x in plan.profile())

    def explain(self, plan, initial):
        """For every n: the first pair from n on that the test refuses, or the
        pair or rest point with the least clearance (Section 8.1)."""
        if not _starts_at(plan, initial):
            return [Why("the plan does not start at the initial set", "pair")]
        n_pairs = len(plan.pairs)
        why = [None] * (n_pairs + 1)
        running = ce.margin(plan.p_rest, plan.ell)
        where = f"the rest point {ce.fmt(plan.p_rest)} m"
        why[n_pairs] = _clearance(running, where)
        first_fail = None
        for i in range(n_pairs - 1, -1, -1):
            seg, beta = plan.pairs[i]
            failure = _pair_failure(seg, beta)
            if failure is not None:
                first_fail = (i, failure)
            c = ce.margin(seg.hi, plan.ell)
            if c < running:
                running, where = c, f"pair {i}, up to {ce.fmt(seg.hi)} m"
            why[i] = (Why(f"pair {first_fail[0]} not certified: {first_fail[1]}", "pair")
                      if first_fail is not None else _clearance(running, where))
        return why

    # --- execution and the alarm check -------------------------------------
    def law(self, plan):
        return Law(plan.head_command(), plan.ell, "segment")

    def distance(self, measured, initial):
        """In the norm max(|p| + |l|, |v| * 1 s) of Section 4.1.2."""
        dp, dv, dl = (measured[i] - initial[i] for i in range(3))
        return max(abs(dp) + abs(dl), abs(dv))

    # --- after an alarm ------------------------------------------------------
    def rest_law(self, measured):
        if measured[1] == 0:
            return Law(Fr(0), None, "rest")            # the brakes hold the cart
        return Law(-ce.U_MAX if self.rest == "maximal" else -ce.B_GLOBAL, None, "rest")

    def reach(self, measured, bound):
        if bound != 0:
            return None, "this package represents reach sets only for an exact measurement"
        p, v, l = measured
        if v == 0:                                     # the brakes hold it in every world
            return (p, Fr(0), l), ""
        if self.rest == "global":                      # 0.5 m/s^2 is realized in every world
            p1, v1 = ramp_braking(p, v)                # consistent with the global capability
            return (p1, v1, l), ""
        return None, NO_REACH

    def restart(self, step, reach):
        p0, v0, _ = reach
        if v0 == 0:
            rf = ce.Plan(step.k + 1, [], p0)
        else:                                          # the command of the rest feedback
            rf = ce.Plan(step.k + 1, [(ce.Segment(p0, v0, -min(ce.B_GLOBAL, v0 / ce.T)),
                                       ce.B_GLOBAL)], p0)
        req = step.requirement.level
        target = ce.WALL - req if req > ce.EPS_MIN else None
        ell = Fr(0) if step.operation is None else step.operation
        labels = _Labels(step.labelling)
        if step.proposal is None:                      # recovery candidates only
            return [_recovery_only(labels, step.k, p0, v0, target, ell)]
        u_q = ce.proposed_command(step.proposal, step.measured[:2])
        return [Candidate(c, a) for c, a in
                ce.generate(labels, step.k, (p0, v0), u_q, rf, target, ell)]

    # --- for the checks of Section 10.3 ----------------------------------------
    def head(self, plan):
        """The pair that the plan executes in its first period: (segment,
        label); a plan without pairs executes its rest segment, which holds
        the cart under the global capability."""
        if plan.pairs:
            return plan.pairs[0]
        return ce.Segment(plan.p_rest, Fr(0), Fr(0)), ce.B_GLOBAL

    def checks(self):
        """A check of this package's own test, beyond what the generic suite
        can do, which sees only the plans that the generator offers. A plan
        whose pairs brake harder than their labels must not be certified: a
        world that brakes exactly as labelled is admitted by the labels, and
        in it the cart would not follow the plan. The plans brake a forward
        motion on the ramp at 0.5 + d m/s^2, under labels of 0.5 m/s^2, until
        a last pair stops the cart within its period."""
        failures, cases = [], 0
        for p, v in ((Fr(20), Fr(2)), (Fr(21), Fr(3, 2)), (Fr(22), Fr(1)),
                     (Fr(23), Fr(1, 2))):
            for d in (Fr(1, 1000), Fr(1, 200), Fr(1, 100), Fr(1, 10)):
                plan, p_rest = _braking_beyond(p, v, ce.B_RAMP + d, ce.B_RAMP)
                b = ce.B_RAMP + d
                cases += 1
                level = self.profile(plan, (p, v, Fr(0))).level
                if level != NEG_INF:
                    stop = p + v * v / (2 * ce.B_RAMP)
                    failures.append(
                        f"from {ce.fmt(p)} m at {ce.fmt(v)} m/s, braking at "
                        f"{ce.fmt(b)} m/s^2 under labels of 0.5 m/s^2 is certified at "
                        f"{ce.fmt(level)} m; it plans rest at {ce.fmt(p_rest)} m, and in a "
                        f"world braking at 0.5 m/s^2 the cart cannot stop before "
                        f"{ce.fmt(stop)} m")
        detail = f"{len(failures)} failed; first: {failures[0]}" if failures else ""
        return [Result("own test refuses plans that brake beyond their labels",
                       "module", FAILED if failures else PASSED, detail, cases)]

    # --- diagnostic descriptions ---------------------------------------------
    def describe_plan(self, plan):
        return dict(head=plan.head_command(),
                    label=plan.pairs[0][1] if plan.pairs else ce.B_GLOBAL,
                    pairs=len(plan.pairs), rest=plan.p_rest, ell=plan.ell)

    def describe_set(self, initial):
        return dict(p_init=initial[0], v_init=initial[1])

    def describe_proposal(self, proposal, at):
        return dict(mode=proposal,
                    proposed=None if proposal is None else
                    ce.proposed_command(proposal, (at[0], at[1])))


def _braking_beyond(p, v, b, label):
    """A plan from (p, v), v > 0, that brakes at b in every period, labelled
    'label', until a last pair stops the cart within its period."""
    pairs = []
    while v > b * ce.T:
        seg = ce.Segment(p, v, -b)
        pairs.append((seg, label))
        p, v = seg.p1, seg.v1
    seg = ce.Segment(p, v, -v / ce.T)
    pairs.append((seg, label))
    return ce.Plan(1, pairs, seg.p1), seg.p1


def _recovery_only(labels, k, p, v, target, ell):
    """The recovery continuation alone from (p, v), with ambition 0: the only
    candidate without a proposal (Section 4.3.4)."""
    tail, p_rest = ce.recovery(labels, k, k + 1, p, v, target)
    if not tail:                                       # at rest: hold for one period
        hold = ce.Segment(p, v, Fr(0))
        tail = [(hold, labels.label(k, k + 1, hold))]
    return Candidate(ce.Plan(k + 1, tail, p_rest, ell), Fr(0))


def ramp_braking(p, v, b=ce.B_RAMP):
    """One period of a constant deceleration b from a forward motion: braked to
    standstill within the period if v <= b T, which it then keeps."""
    if v <= b * ce.T:
        return p + v * v / (2 * b), Fr(0)
    return p + v * ce.T - b * ce.T * ce.T / 2, v - b * ce.T


def _pair_failure(seg, beta):
    if abs(seg.u) > ce.U_MAX:
        return "command beyond the bound"
    if seg.u < 0 and -seg.u > beta:
        return (f"acceleration towards the start {ce.fmt(-seg.u)} beyond the label "
                f"{ce.fmt(beta)}")
    if seg.v0 * seg.v1 < 0:
        return "motion reverses within the period"
    return None


def _clearance(running, where):
    text = f"clearance {ce.fmt(running)} m at {where}"
    if running >= ce.EPS_MIN:
        return Why(text, "certified")
    return Why(text + ", below the floor", "floor")
