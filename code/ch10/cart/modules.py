"""The other modules of the cart: proposal method, capability predictors,
the tool operation with its demand predictor, the robot and simulator
adapters, and the evaluator (Chapters 3, 7 and 8)."""
from fractions import Fraction as Fr

from allovi import NEG_INF, Requirement, register

from .control import ramp_braking
from .model import MODEL, OPERATION_LEVEL, ce


# ----------------------------------------------------------------------------
# Proposal method
# ----------------------------------------------------------------------------
@register
class Naive:
    """The proposal method of Chapter 3: 'accelerate' until the measured
    position first reaches the work zone, 'hold' from then on."""
    ROLE, NAME, CONTRACT = "proposal", "cart.naive", "1.0"

    def __init__(self):
        self.method = ce.ProposalMethod()

    def propose(self, observation):
        return self.method(observation.measured[0])


# ----------------------------------------------------------------------------
# Capability predictors (Section 3.1; the survey predictor of Section 7.2.3)
# ----------------------------------------------------------------------------
class _Chapter3:
    ROLE, CONTRACT, SPACE = "capability", "1.0", "cart.braking"
    horizon = ce.HORIZON
    KIND = None

    def __init__(self):
        self.predictor = ce.Predictor(self.KIND)

    def labels(self, observation):
        k, predictor = observation.k, self.predictor
        return lambda j, segment: predictor.label(k, j, segment)

    def observe(self, k, before, after):
        p, p1 = before[0], after[0]
        self.predictor.observe(min(p, p1), max(p, p1), p1)


@register
class Fixed(_Chapter3):
    """The global capability everywhere."""
    NAME, KIND = "cart.fixed", "fixed"


@register
class Reactive(_Chapter3):
    """The braking measured on positions already traversed, else the global capability."""
    NAME, KIND = "cart.reactive", "reactive"


@register
class Forecasting(_Chapter3):
    """The minimum of a correct map of the braking over the region of the segment."""
    NAME, KIND = "cart.forecasting", "forecasting"


@register
class Persisting(_Chapter3):
    """The braking measured at the position of the cart, for every segment."""
    NAME, KIND = "cart.persisting", "persisting"


@register
class Survey:
    """The survey predictor of Section 7.2.3 without its offset:
    2 m/s^2 for a segment whose region lies before the surveyed start
    of the ramp, 0.5 m/s^2 for the others."""
    ROLE, NAME, CONTRACT = "capability", "cart.survey", "1.0"
    SPACE, horizon = "cart.braking", ce.HORIZON

    def __init__(self, ramp_start):
        self.ramp_start = ramp_start

    def labels(self, observation):
        k = observation.k

        def labelling(j, segment):
            if j > k + 1 + self.horizon:
                return ce.B_GLOBAL
            if segment.hi >= self.ramp_start:
                return ce.B_RAMP
            return ce.B_FIRM
        return labelling

    def observe(self, k, before, after):
        pass


# ----------------------------------------------------------------------------
# The tool operation and the demand predictor that reads its schedule
# ----------------------------------------------------------------------------
@register
class Operation:
    """The tool operation of Chapter 3, due in period 'onset' and needing
    1.3 m of reserve. It begins if the plan in force during its onset period
    holds that level (Assumption 4.13): from then on, the generator is asked
    for candidates with the tool unfolded. Otherwise an overload occurs, and
    the operation is dropped (postpone=None, as in Chapter 3) or postponed:
    to the forecast onset if that is still ahead, else by 'postpone' periods,
    announced (Section 8.2.3).

    Arguments and attributes, in periods of 0.1 s: announce, the period from
    which the task plan announces the operation, or None if it is never
    announced; onset, the period in which it is due; predicted, the onset
    that the task plan forecasts (onset unless given); always, True if the
    demand predictor is to require the reserve in every period; postpone,
    None or the periods by which an overload postpones it. A postponement
    changes onset and, without a forecast onset still ahead, predicted and
    announce too. A run sets begun, the period in which the operation began,
    or None; dropped; overloads, the periods in which it was due and the
    plan in force did not hold 1.3 m; and reach, which operation(k) returns:
    0 until the operation begins, ce.ELL_UNFOLDED (0.8 m) from then on.
    pending() is True until it has begun or been dropped. The demand
    predictor and the evaluator of a configuration read the same instance,
    which they receive as the argument 'task' (../templates/HELPERS.md).

    >>> task = Operation()                  # announced in period 140, due in 200
    >>> task.pending(), task.predicted, task.operation(150)
    (True, 200, Fraction(0, 1))
    >>> task.begin_period(200, Fr(13, 10))  # the plan in force holds 1.3 m
    >>> task.begun, task.pending(), task.operation(200)
    (200, False, Fraction(4, 5))
    """
    ROLE, NAME, CONTRACT = "task", "cart.operation", "1.0"

    def __init__(self, announce=140, onset=200, predicted=None, always=False,
                 postpone=None):
        self.announce, self.onset = announce, onset
        self.predicted = onset if predicted is None else predicted
        self.always, self.postpone = always, postpone
        self.begun, self.dropped, self.overloads = None, False, []
        self.reach = Fr(0)

    def pending(self):
        return self.begun is None and not self.dropped

    def begin_period(self, k, level_in_force):
        if not self.pending() or k != self.onset:
            return
        if level_in_force != NEG_INF and level_in_force >= OPERATION_LEVEL:
            self.begun, self.reach = k, ce.ELL_UNFOLDED
            return
        self.overloads.append(k)
        if self.postpone is None:
            self.dropped = True
        elif self.announce is not None and self.predicted > k:
            self.onset = self.predicted
        else:
            self.onset = self.predicted = k + self.postpone
            self.announce = k

    def operation(self, k):
        return self.reach


@register
class TaskPlan:
    """The demand predictor of Chapter 8: from the announcement on, the
    reserve of the operation for the onset that the task plan forecasts;
    once that onset has passed, from the first pair on; 'always' requires it
    from the first pair on in every period."""
    ROLE, NAME, CONTRACT = "demand", "cart.task-plan", "1.0"

    def __init__(self, task):
        self.task = task

    def requirement(self, observation):
        task, k = self.task, observation.k
        if task.pending():
            if task.always:
                return Requirement(OPERATION_LEVEL, 0)
            if task.announce is not None and k >= task.announce:
                return Requirement(OPERATION_LEVEL, max(task.predicted - k - 1, 0))
        return Requirement(ce.EPS_MIN, 0)


# ----------------------------------------------------------------------------
# Robot and simulator adapters
# ----------------------------------------------------------------------------
@register
class Cart:
    """The state (p, v, l) and the actuation of the cart; the measurement is
    exact (Chapter 3), so its error bound is zero."""
    ROLE, NAME, CONTRACT = "robot", "cart.robot", "1.0"
    STATE, ACTUATION = MODEL.state, MODEL.actuation
    measurement_bound = Fr(0)

    def measure(self, state):
        return state

    def describe_state(self, measured):
        return dict(p=measured[0], v=measured[1])


@register
class Track:
    """The world of Chapter 3: firm ground before 20 m, the ramp beyond. It
    realizes the laws of the control package cart.constant-command from
    their fields, exactly where it can. A head realized as commanded moves
    exactly; a head that brakes harder than the ground allows moves as
    computed in floating point by the function of Chapter 3, unless it
    touches the ramp only at the end of the period (Section 7.3). The rest
    feedback brakes a forward motion on the ramp at the smaller of its
    command and the 0.5 m/s^2 that the ramp allows, until the cart is at
    rest, and the brakes then hold it (Section 8.3); elsewhere, which no run
    of the book needs, the adapter raises NotImplementedError. The attribute
    'inexact' counts the periods since the last reset in which a head braked
    harder than the ground allows (for the records)."""
    ROLE, NAME, CONTRACT = "world", "cart.track", "1.0"
    STATE, ACTUATION = MODEL.state, MODEL.actuation

    def reset(self):
        self.inexact = 0
        return (Fr(0), Fr(0), Fr(0))

    def step(self, state, law):
        p, v, l = state
        ell = l if law.ell is None else law.ell
        if law.kind == "segment":
            p1, v1, exact = ce.realize(p, v, law.u)
            if not exact:
                self.inexact += 1
                seg = ce.Segment(p, v, law.u)
                if seg.hi == ce.RAMP:            # the overstated braking never acts
                    p1, v1 = seg.p1, seg.v1
                else:
                    p1, v1 = self.on_the_ramp(seg, p1, v1)
            return (p1, v1, ell)
        if v == 0:                               # rest feedback at rest: the brakes hold
            return (p, v, ell)
        if law.u >= 0 or v < 0 or p < ce.RAMP:
            raise NotImplementedError(
                "this simulator adapter realizes the rest feedback only while it "
                "brakes a forward motion on the ramp, as after every alarm of the book")
        p1, v1 = ramp_braking(p, v, min(-law.u, ce.B_RAMP))
        return (p1, v1, ell)

    # --- the oracle, for the checks of Section 10.3 (simulation only) ---------
    def admits(self, label, segment, k):
        """True if the world of period k is admitted by the label on the
        region of the segment: the ground brakes at least 'label' everywhere
        on it (the world of this adapter does not change over time)."""
        return ce.b_min(segment.lo, segment.hi) >= label

    def clearance(self, state):
        """The true reserve of a state: its distance, in the norm of the
        state, to the states that violate p + l <= 25 m."""
        return ce.WALL - (state[0] + state[2])

    def on_the_ramp(self, seg, p1f, v1f):
        """The state at the end of a period in which the head braked harder
        than the ground allows and the cart reached the ramp, from the
        floating-point motion (p1f, v1f) of Chapter 3. On the ramp, the
        deceleration is a constant 0.5 m/s^2, so p + v^2 / (2 * 0.5) is the
        point at which the cart comes to rest, which follows exactly from the
        start of the period, as in Section 8.3. The velocity is kept as
        computed, and the position is corrected by its rounding error, which
        is checked to be below 1e-9 m, so that the stopping point is exact
        and every later period on the ramp is computed exactly. The state
        then implies the exact stopping point, so that the set that the rest
        feedback can reach, which the control package computes from the
        measurement alone, is the one that Chapter 8 computes from the exact
        stopping point (Section 10.1)."""
        return _on_the_ramp(seg, p1f, v1f)


def _on_the_ramp(seg, p1f, v1f):
    if not (seg.u < 0 and -seg.u > ce.B_RAMP and seg.v0 > 0):
        raise NotImplementedError("a head that brakes harder than the ground allows "
                                  "is realized only for a forward motion")
    if seg.p0 < ce.RAMP:
        v2 = seg.v0 * seg.v0 - 2 * min(-seg.u, ce.B_FIRM) * (ce.RAMP - seg.p0)
        stop = ce.RAMP + v2 / (2 * ce.B_RAMP)
    else:
        stop = seg.p0 + seg.v0 * seg.v0 / (2 * ce.B_RAMP)
    v1 = Fr(v1f)
    p1 = stop - v1 * v1 / (2 * ce.B_RAMP)
    if p1 < ce.RAMP or abs(float(p1) - p1f) >= 1e-9:
        raise ArithmeticError(f"the correction of the position, {float(p1) - p1f:.3g} m, "
                              "is larger than a rounding error")
    return p1, v1


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
@register
class Metrics:
    """Task performance alongside the guarantees, from the states at the
    period boundaries: arrival in the work zone and the time spent there,
    when the tool unfolds, the least clearance, overloads, alarms and
    restarts.

    evaluate(record), for the record of allovi.run, returns a dictionary:
    arrival_in_work_zone [s], the first period boundary at which p >= 24 m,
    or None; time_in_work_zone [s], 0.1 s for each period boundary, the end
    of the run included, at which p >= 24 m; tool_unfolds [s], the start of
    the period in which the tool unfolds, or None; least_clearance [m], the
    least of 25 m - (p + l) at the period boundaries; overloads, the number
    of the task's overloads (None without a task); alarms and restarts,
    their numbers. Times and the clearance are exact.

    >>> import allovi, cart
    >>> config = cart.configuration("forecasting")   # the run of Chapter 3
    >>> m = config.evaluator.evaluate(allovi.run(config, 220))
    >>> cart.fmt(m["arrival_in_work_zone"]), cart.fmt(m["tool_unfolds"]), m["alarms"]
    ('8.6', '20.1', 0)
    """
    ROLE, NAME, CONTRACT = "evaluator", "cart.metrics", "1.0"

    def __init__(self, task=None):
        self.task = task

    def evaluate(self, record):
        states = [p.state for p in record.periods] + [record.final_state]
        inside = [i for i, s in enumerate(states) if s[0] >= ce.WORK_ZONE]
        unfolded = [i for i, s in enumerate(states) if s[2] > 0]
        return dict(
            arrival_in_work_zone=None if not inside else inside[0] * ce.T,
            time_in_work_zone=len(inside) * ce.T,
            tool_unfolds=None if not unfolded else (unfolded[0] - 1) * ce.T,
            least_clearance=min(ce.WALL - (s[0] + s[2]) for s in states),
            overloads=None if self.task is None else len(self.task.overloads),
            alarms=len(record.alarms), restarts=len(record.restarts))
