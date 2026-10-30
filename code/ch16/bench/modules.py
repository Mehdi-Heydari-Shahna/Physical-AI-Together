"""The modules of the anticipation benchmark (Sections 16.1 and 16.2).

Every configuration of the benchmark uses the control package
bench.constant-command: the control package cart.constant-command of
Chapter 10, with its generator, test, rest segment and rest feedback
unchanged, and one more proposal that it can turn into candidates,
'retreat', a command of -0.5 m/s^2, which only the preview planner uses.
For the proposals of Chapter 3, 'accelerate' and 'hold', the package is
cart.constant-command, as the benchmark checks against Chapter 8.

The preview planner is a model-based proposal method that reads the task
plan, as the demand predictor does, and plans its own back-away; the
evaluator bench.metrics measures, from the records and the world, what
Section 16.3 reports.
"""
import time
from fractions import Fraction as Fr

import allovi
from allovi import Candidate, register
from cart import modules as cm
from cart.control import ConstantCommand, _Labels, _recovery_only
from cart.model import OPERATION_LEVEL, T, ce

RETREAT = "retreat"
TARGET = ce.WALL - OPERATION_LEVEL        # 23.7 m: at rest there, the operation can begin
POSTPONE = 21                             # periods, as in Section 8.2.3


def command_of(q, xhat):
    """The physical meaning of a proposal at the initial set xhat: that of
    Chapter 3 for 'accelerate' and 'hold'; 'retreat' is a command of
    -0.5 m/s^2, towards the start, which every label admits."""
    if q == RETREAT:
        return -ce.B_GLOBAL
    return ce.proposed_command(q, xhat)


@register
class Retreating(ConstantCommand):
    """cart.constant-command with the proposal 'retreat'.

    generate(step, plan, initial) and restart(step, reach) are those of
    cart.constant-command, with command_of() in place of
    ce.proposed_command, so that 'retreat' is a command of -0.5 m/s^2 and
    'accelerate' and 'hold' mean what they mean in Chapter 3. A subclass
    that adds candidates, as reader.full-reverse of Chapter 17 does, calls
    super().generate() and appends to the list it returns
    (../../ch10/templates/HELPERS.md, section 9)."""
    NAME = "bench.constant-command"

    def generate(self, step, plan, initial):
        p, v, _ = initial
        req = step.requirement.level
        target = ce.WALL - req if req > ce.EPS_MIN else None
        ell = Fr(0) if step.operation is None else step.operation
        labels = _Labels(step.labelling)
        if step.proposal is None:                      # recovery candidates only
            return [_recovery_only(labels, step.k, p, v, target, ell)]
        u_q = command_of(step.proposal, (p, v))
        return [Candidate(c, a) for c, a in
                ce.generate(labels, step.k, (p, v), u_q, plan.shift(), target, ell)]

    def restart(self, step, reach):
        p0, v0, _ = reach
        if v0 == 0:
            rf = ce.Plan(step.k + 1, [], p0)
        else:
            rf = ce.Plan(step.k + 1, [(ce.Segment(p0, v0, -min(ce.B_GLOBAL, v0 / ce.T)),
                                       ce.B_GLOBAL)], p0)
        req = step.requirement.level
        target = ce.WALL - req if req > ce.EPS_MIN else None
        ell = Fr(0) if step.operation is None else step.operation
        labels = _Labels(step.labelling)
        if step.proposal is None:
            return [_recovery_only(labels, step.k, p0, v0, target, ell)]
        u_q = command_of(step.proposal, step.measured[:2])
        return [Candidate(c, a) for c, a in
                ce.generate(labels, step.k, (p0, v0), u_q, rf, target, ell)]

    def describe_proposal(self, proposal, at):
        return dict(mode=proposal,
                    proposed=None if proposal is None else command_of(proposal, (at[0], at[1])))


# ----------------------------------------------------------------------------
# The preview planner
# ----------------------------------------------------------------------------
def braked(p, v):
    """From a backward motion (v < 0), the place and the number of periods
    at which braking, as the generator brakes a backward motion (a command
    of -v/T, at most 2 m/s^2, in each period), brings the cart to rest."""
    n = 0
    while v < 0:
        seg = ce.Segment(p, v, min(ce.U_MAX, -v / ce.T))
        p, v = seg.p1, seg.v1
        n += 1
    return p, n


def periods_to_target(p):
    """From rest at p on the ramp, the periods that the back-away of the
    generator takes to bring the cart to rest at or before TARGET: m periods
    at -0.5 m/s^2, the fewest that suffice, then braking at 2 m/s^2. Every
    world of the family has the ramp before 21 m, so the back-away always
    lies on the ramp, where every label is 0.5 m/s^2."""
    if p <= TARGET:
        return 0
    m = 0
    while True:
        m += 1
        q, w = p, Fr(0)
        for _ in range(m):
            seg = ce.Segment(q, w, -ce.B_GLOBAL)
            q, w = seg.p1, seg.v1
        q, n = braked(q, w)
        if q <= TARGET:
            return m + n


def predict(p, v, u):
    """The state after one period under the command u, as the planner
    predicts it (the motion of the cart's model)."""
    seg = ce.Segment(p, v, u)
    if v * seg.v1 < 0:                        # braked to rest within the period
        return p - v * v / (2 * u), Fr(0)
    return seg.p1, seg.v1


@register
class Preview:
    """A model-based proposal method that reads the task plan. Until the
    measured position first reaches the work zone it proposes 'accelerate',
    then 'hold'. Once the operation is announced, it predicts the state at
    the next period boundary from its own last proposal (if the governor
    committed it in full; otherwise as if the cart kept its speed), and
    proposes 'retreat' from the latest period from which the back-away of
    the generator still brings the cart to rest at 23.7 m by the forecast
    onset, 'late' periods later if late > 0; it retreats until braking as
    the generator brakes brings the cart to rest at or before 23.7 m, and
    then proposes 'hold'. Once the operation has begun, it proposes
    'hold'."""
    ROLE, NAME, CONTRACT = "proposal", "bench.preview", "1.0"

    def __init__(self, task, late=0):
        self.task, self.late = task, late
        self.arrived, self.retreating, self.last = False, False, None

    def propose(self, observation):
        k = observation.k
        p, v, ell = observation.measured
        if self.last is not None and observation.ambition == 1:
            u = command_of(self.last, (p, v))
        else:
            u = Fr(0)
        p1, v1 = predict(p, v, u)
        q = self.decide(k, p, ell, p1, v1)
        self.last = q
        return q

    def decide(self, k, p, ell, p1, v1):
        task = self.task
        if ell > 0 or not task.pending():
            return "hold"
        if not self.arrived:
            if p >= ce.WORK_ZONE:
                self.arrived = True
            else:
                return "accelerate"
        if task.announce is None or k < task.announce:
            return "hold"
        if self.retreating:
            if v1 < 0 and braked(p1, v1)[0] > TARGET:
                return RETREAT
            if v1 == 0 and p1 > TARGET:
                return RETREAT
            return "hold"
        if v1 == 0 and p1 > TARGET:
            if k + 1 + periods_to_target(p1) - self.late >= task.predicted:
                self.retreating = True
                return RETREAT
        return "hold"


# ----------------------------------------------------------------------------
# The evaluator
# ----------------------------------------------------------------------------
@register
class Metrics:
    """What a run did, from the record of the core and the world, which the
    simulation knows: the task's outcomes, the interventions, the labels'
    validity, alarms and restarts, the true clearance, and the timing of the
    decisions."""
    ROLE, NAME, CONTRACT = "evaluator", "bench.metrics", "1.0"

    def __init__(self, world, task):
        self.world, self.task = world, task

    def evaluate(self, record):
        w, task = self.world, self.task
        states = [x.state for x in record.periods] + [record.final_state]
        arrival = next((i for i, s in enumerate(states) if s[0] >= ce.WORK_ZONE), None)
        begun = task.begun
        until = len(record.periods) if begun is None else begun
        normal = [x for x in record.periods if x.governor == "normal"]
        invalid, exercised, first_invalid = 0, 0, None
        for x in normal:
            if not x.plan.pairs:
                continue
            seg, beta = x.plan.pairs[0]
            b = ce.b_min(seg.lo, seg.hi)
            if beta > b:
                invalid += 1
                first_invalid = x.k if first_invalid is None else first_invalid
                exercised += int(seg.u < 0 and -seg.u > b)
        reach = []
        for i in range(len(states) - 1):
            reach.append(max(states[i][0], states[i + 1][0]) + states[i + 1][2])
        least = min(ce.WALL - r for r in reach)
        before = [x for x in normal if x.k < (arrival if arrival is not None else until)]
        after = [x for x in normal if (arrival is not None and arrival <= x.k < until)]
        overridden = 0
        for x in after:
            if x.proposal is None:
                overridden += 1
                continue
            head = x.decision.plan.pairs[0][0].u if x.decision.plan.pairs else Fr(0)
            overridden += head != command_of(x.proposal, x.initial_set[:2])
        n_fresh = [len(x.decision.entries) - 1 for x in normal]
        evaluated = [x.decision.evaluated for x in normal]
        return dict(
            arrival=arrival, begun=begun,
            p_onset=states[w.onset][0] if w.onset < len(states) else None,
            delay=None if begun is None else begun - w.onset,
            overloads=list(task.overloads),
            in_work_zone=sum(1 for s in states[:-1] if s[0] >= ce.WORK_ZONE),
            modified_before_arrival=sum(1 for x in before if x.decision.ambition < 1),
            modified_after_arrival=sum(1 for x in after if x.decision.ambition < 1),
            overridden_after_arrival=overridden,
            periods_after_arrival=len(after),
            fallbacks=sum(1 for x in normal if x.decision.selected == 0),
            evaluated=sum(evaluated), offered=sum(n_fresh),
            cut=sum(1 for e, n in zip(evaluated, n_fresh) if e < n),
            least_clearance=least, violation=least < 0,
            invalid=invalid, exercised=exercised, first_invalid=first_invalid,
            alarms=list(record.alarms), restarts=list(record.restarts),
            rest_feedback=sum(1 for x in record.periods if x.governor != "normal"),
            final=states[-1])


# ----------------------------------------------------------------------------
# The timing of the decisions
# ----------------------------------------------------------------------------
class Clock:
    """Wraps the decision of the core to measure its duration with
    time.perf_counter (wall-clock time of the machine; for the report only,
    never for a decision)."""

    def __init__(self):
        self.durations = []
        self._decide = allovi.loop.decide

    def __enter__(self):
        def timed(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return self._decide(*args, **kwargs)
            finally:
                self.durations.append(time.perf_counter() - t0)
        allovi.loop.decide = timed
        return self

    def __exit__(self, *exc):
        allovi.loop.decide = self._decide
        return False


def operation(world, demand):
    """The tool operation of the world, with the schedule that the
    configuration's demand predictor reads: 'forecast', 'exact', 'always',
    'due' (nothing announced until the operation is due) or 'none' (the
    task plan is announced, and no demand predictor reads it)."""
    if demand == "due":
        return cm.Operation(announce=None, onset=world.onset, predicted=world.onset,
                            postpone=POSTPONE)
    if demand == "exact":
        return cm.Operation(announce=world.announce, onset=world.onset,
                            predicted=world.onset, postpone=POSTPONE)
    return cm.Operation(announce=world.announce, onset=world.onset,
                        predicted=world.onset_forecast, always=(demand == "always"),
                        postpone=POSTPONE)
