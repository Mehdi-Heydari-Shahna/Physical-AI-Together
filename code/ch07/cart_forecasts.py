#!/usr/bin/env python3
"""
Capability and demand predictors for the cart.

Physical AI, Together, Chapter 7, Sections 7.1 to 7.3.

The cart, its control method, the governor and the proposal method are those
of Chapter 3, imported from ../ch03/cart_example.py and used unchanged. This
script adds what a predictor researcher supplies.

Capability predictors (Section 7.1)
    Each issues labels 'braking of at least beta on the positions that a
    segment covers, in the period in which it would be executed', and
    states for every label its provenance and its coverage claim:
    fixed      the global capability, 0.5 m/s^2, everywhere (declared)
    reactive   the braking measured on positions already traversed, and the
               global capability elsewhere (measured)
    survey     a survey of where the ramp starts, with a stated accuracy of
               0.2 m, moved 0.2 m towards the start (bounded error)
    map        the true map, as in Chapters 3 to 6 (known only in simulation)
    persisting the braking measured at the cart's position, applied to every
               segment ahead (extrapolated; no coverage claim)
    They run in a family of 50 worlds that differ in where the ramp starts.
    A validation record compares each executed label with the realized
    world.

Demand predictor (Section 7.2)
    The operation of Chapter 3, the tool unfolded by 0.8 m, becomes a
    requirement: the reserve floor plus the reserve the operation consumes,
    from the onset period on. Errors are injected into the announcement, the
    level and the onset of the requirement and, for comparison, into the
    survey that the capability predictor uses, with biases of both signs.

Calibration and validation (Section 7.3)
    Surveys are now precise (error within 0.2 m) or, unknown to anyone,
    unreliable (error within 0.6 m). Four rules turn a survey into labels:
    a point estimate, the stated accuracy as a bound, a bound calibrated on
    100 worlds by split conformal prediction, and the worst-case bound.
    Their miscoverage is counted on 2000 held-out worlds, with exact
    one-sided upper confidence bounds, and their consequences are simulated
    in 50 worlds. The point estimate also runs with a cautious proposal
    method, to show what the closed loop selects, and the calibrated bound
    runs after a shift in the share of unreliable surveys and after the
    ground has changed since the survey.

Reproducibility
    Every planned and certified quantity is computed with exact fractions,
    as in Chapter 3, and so is the realized motion whenever the executed
    label is valid. Floating point is used only for motion under an invalid
    label and after an alarm, through the functions of Chapter 3, and for
    the bound 1 - 0.05^(1/50) on clean worlds, printed to one decimal. The
    probabilities of Section 7.3 are computed exactly, and its confidence
    bounds by bisection in integer arithmetic, never below the exact value.
    Random numbers come only from random.Random(seed).random() with integer
    seeds. Means are computed exactly and rounded half up. The results were
    identical, bit for bit, in runs with CPython 3.8, 3.13 and 3.14 on one
    x86-64 Linux machine; other processors, operating systems and versions
    have not been tested.

Usage
    python3 cart_forecasts.py            # all runs; writes results/
    python3 cart_forecasts.py --check    # also checks every quoted number
    python3 cart_forecasts.py --jobs 1   # one process (default: up to 2)
    python3 cart_forecasts.py --bias 0.02 --onset-error 3
                                         # Exercise 7.1: errors of one's own
                                         # choice; prints, writes no file

Outputs, in results/ next to this file
    summary.txt       the numbers quoted in Chapter 7
    runs.csv          one row per run in the family of worlds
    validation.csv    one validation record per executed label that claims
                      more braking than the global capability
    demand.csv        one row per run with an injected demand error
    survey-bias.csv   one row per run with an injected survey error
    calibration.csv   the 100 calibration worlds of Section 7.3
    held-out.csv      the 2000 held-out worlds, under the three conditions
    rules.csv         one row per run of Section 7.3 with the operation
    loop.csv          one row per run of the two proposal methods

Python 3.8 or later; no other packages.
"""
import argparse
import csv
import math
import os
import random
import sys
from fractions import Fraction as Fr
from multiprocessing import Pool

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ch03"))
import cart_example as ce    # noqa: E402

T = ce.T
NEG_INF = ce.NEG_INF
SEEDS = range(1, 51)              # the family: worlds with seeds 1 to 50
SURVEY_BOUND = Fr(1, 5)           # stated accuracy of the survey [m]
SURVEY_OFFSET = Fr(1, 5)          # shift of the surveyed start towards the start [m]
RAMP_CH3 = ce.RAMP                # the world of Chapter 3: the ramp at 20 m
PREDICTORS = ("fixed", "reactive", "survey", "map", "persisting")


# ----------------------------------------------------------------------------
# The family of worlds
# ----------------------------------------------------------------------------
class World:
    """One world: where the ramp starts, and the error of the survey of that
    position. The ramp start is drawn as in the world family of Chapter 6,
    from [17, 21] m on a 1 cm grid, and the survey error from
    [-0.20, 0.20] m on a 1 cm grid, within the survey's stated accuracy."""

    def __init__(self, seed, ramp=None, survey_error=None):
        rng = random.Random(seed)
        self.seed = seed
        self.ramp = Fr(round((17 + 4 * rng.random()) * 100), 100)
        self.survey_error = Fr(round((2 * rng.random() - 1) * 20), 100)
        if ramp is not None:
            self.ramp = ramp
        if survey_error is not None:
            self.survey_error = survey_error

    def install(self):
        ce.RAMP = self.ramp       # read by every map and motion function of Chapter 3


def restore_world():
    ce.RAMP = RAMP_CH3


def chapter3_world():
    return World(0, ramp=RAMP_CH3, survey_error=Fr(0))


# ----------------------------------------------------------------------------
# Capability predictors with provenance and coverage claims
# ----------------------------------------------------------------------------
class CapabilityPredictor:
    """Issues the label of a segment for a period: braking of at least beta
    on the positions the segment covers. Beyond the forecast horizon, the
    label is the global capability. Every label attached to a pair of a
    candidate is also checked against the realized world, which the
    simulation knows; a real predictor cannot do this, and the counts serve
    only to evaluate it. The generator also asks for the labels of trial
    segments that it then discards; those are not counted."""

    name = "base"
    provenance = ""
    claim = ""

    def __init__(self):
        self.issued = 0           # labels attached to candidate pairs
        self.raised = 0           # ... that claim more than the global capability
        self.invalid = 0          # ... that the realized world makes invalid

    def value(self, k, j, seg):
        raise NotImplementedError

    def label(self, k, j, seg):
        return ce.B_GLOBAL if j > k + 1 + ce.HORIZON else self.value(k, j, seg)

    def count(self, cands):
        """Counts the labels attached to the pairs of the candidates of one
        period, and those that the realized world makes invalid."""
        for plan, _ in cands:
            for seg, beta in plan.pairs:
                self.issued += 1
                self.raised += beta > ce.B_GLOBAL
                self.invalid += beta > ce.b_min(seg.lo, seg.hi)

    def observe(self, p_lo, p_hi, p_now):
        pass


class Fixed(CapabilityPredictor):
    name = "fixed"
    provenance = "declared global capability"
    claim = "valid in every period, by the assumption of a global capability"

    def value(self, k, j, seg):
        return ce.B_GLOBAL


class Reactive(CapabilityPredictor):
    name = "reactive"
    provenance = "braking measured on positions already traversed"
    claim = "valid if the measurement is exact and the ground does not change"

    def __init__(self):
        super().__init__()
        self.seen_lo, self.seen_hi = Fr(0), Fr(0)

    def observe(self, p_lo, p_hi, p_now):
        self.seen_lo = min(self.seen_lo, p_lo)
        self.seen_hi = max(self.seen_hi, p_hi)

    def value(self, k, j, seg):
        if self.seen_lo <= seg.lo and seg.hi <= self.seen_hi:
            beta = ce.b_min(seg.lo, seg.hi)
            if beta > ce.B_GLOBAL:          # where it claims more than c-bar
                self.standing_at_start = (getattr(self, "standing_at_start", True)
                                          and seg.lo == seg.hi == 0 and seg.u == 0)
            return beta
        return ce.B_GLOBAL


class Survey(CapabilityPredictor):
    """Labels from a surveyed ramp start: 2 m/s^2 on positions that end
    before s_hat = surveyed start - offset, 0.5 m/s^2 elsewhere. With an
    offset at least the survey error, every label is valid."""

    name = "survey"
    provenance = "survey of the ramp start, stated accuracy 0.2 m; offset 0.2 m"
    claim = "valid if the survey error is at most the offset and the ground is unchanged"

    def __init__(self, surveyed, offset=SURVEY_OFFSET):
        super().__init__()
        self.surveyed, self.offset = surveyed, offset
        self.s_hat = surveyed - offset

    def value(self, k, j, seg):
        return ce.B_RAMP if seg.hi >= self.s_hat else ce.B_FIRM


class Map(CapabilityPredictor):
    name = "map"
    provenance = "the true map, available only in simulation"
    claim = "valid"

    def value(self, k, j, seg):
        return ce.b_min(seg.lo, seg.hi)


class Persisting(CapabilityPredictor):
    name = "persisting"
    provenance = "braking measured at the cart's position, applied ahead"
    claim = "none"

    def __init__(self):
        super().__init__()
        self.b_here = ce.B_FIRM

    def observe(self, p_lo, p_hi, p_now):
        self.b_here = ce.b_true(p_now)

    def value(self, k, j, seg):
        return self.b_here


def make_predictor(name, world, offset=SURVEY_OFFSET):
    if name == "survey":
        return Survey(world.ramp + world.survey_error, offset)
    return {"fixed": Fixed, "reactive": Reactive, "map": Map,
            "persisting": Persisting}[name]()


# ----------------------------------------------------------------------------
# The operation and the demand predictor
# ----------------------------------------------------------------------------
class Operation:
    """What the task plan states about an operation, and what follows from
    it for the reserve. The reserve is the clearance to the wall, in metres.
    Unfolding the tool by ell consumes ell of it, because the constraint is
    p + ell <= 25 m, so the operation needs the floor plus ell from its onset
    period on: then, and only then, a candidate that unfolds the tool can be
    certified at the floor, as the assumption of reserve consumption
    requires (it holds with equality for this cart)."""

    def __init__(self, announce=ce.T_ANNOUNCE, onset=ce.T_ONSET, ell=ce.ELL_UNFOLDED):
        self.announce, self.onset, self.ell = announce, onset, ell

    def consumption(self):
        return self.ell

    def level(self):
        return ce.EPS_MIN + self.consumption()


class DemandPredictor:
    """Issues during period k the requirement (level, index of the onset
    pair) for the plan committed for period k + 1, from its announcement
    until the operation begins or an overload is declared. Its predicted
    level and onset may differ from those of the operation; once its
    predicted onset has passed, it requires its level from pair 0 on."""

    def __init__(self, k0, kd_pred, level):
        self.k0, self.kd_pred, self.level = k0, kd_pred, level

    def requirement(self, k):
        if k < self.k0:
            return ce.EPS_MIN, 0
        if k < self.kd_pred:
            return self.level, self.kd_pred - k - 1
        return self.level, 0


def correct_demand(op):
    return DemandPredictor(ce.period(op.announce), ce.period(op.onset), op.level())


# ----------------------------------------------------------------------------
# One governed run: the loop of Chapter 3, with the predictors as objects
# ----------------------------------------------------------------------------
class Chapter3Proposals:
    """The proposal method of Chapter 3: 'accelerate' (+2 m/s^2) until the
    measured position first reaches the work zone, 'hold' from then on."""

    name = "chapter 3"

    def __init__(self):
        self.method = ce.ProposalMethod()

    def __call__(self, p, xhat):
        q = self.method(p)
        return q, ce.proposed_command(q, xhat)


def run(pred, world, demand, op, t_end=ce.T_END, log=False, proposals=None):
    """The run of cart_example.run with the capability predictor 'pred', the
    demand predictor 'demand', the operation 'op' and the proposal method
    'proposals' (that of Chapter 3 if None). The operation begins
    in its onset period if the plan in force then holds the level it needs;
    otherwise an overload is recorded and the operation does not take place,
    as in Chapter 3. The run stops at the first label alarm. Returns the
    rows of Chapter 3, a summary and, if log is set, the validation records
    of the executed labels that claim more than the global capability."""
    world.install()
    try:
        return _run(pred, world, demand, op, t_end, log,
                    proposals if proposals is not None else Chapter3Proposals())
    finally:
        restore_world()


def _run(pred, world, demand, op, t_end, log, proposal):
    kd, k_end, need = ce.period(op.onset), ce.period(t_end), op.level()
    p, v = Fr(0), Fr(0)
    plan = ce.Plan(0, [], Fr(0))
    prof = plan.profile()
    issued = [0]                  # period in which each pair of the plan was labelled
    ell_op, k_met, overload = Fr(0), None, False
    req_prev, onset_prev = ce.EPS_MIN, 0
    rows, records = [], []
    for k in range(k_end):
        u_exec = plan.head_command()
        beta_exec = plan.pairs[0][1] if plan.pairs else ce.B_GLOBAL
        if plan.pairs:
            seg = plan.pairs[0][0]
            xhat = (seg.p1, seg.v1)
            b_real = ce.b_min(seg.lo, seg.hi)
            valid, exercised = beta_exec <= b_real, u_exec < 0 and -u_exec > b_real
        else:
            seg, xhat = None, (plan.p_rest, Fr(0))
            valid, exercised = True, False          # the rest segment: global capability
        q, u_q = proposal(p, xhat)
        active = k_met is None and not overload
        if active and k == kd:                      # the onset: can the operation begin?
            if prof[0] is not NEG_INF and prof[0] >= need:
                k_met, ell_op = k, op.ell
            else:
                overload = True
            req, onset = ce.EPS_MIN, 0
        elif active and k < kd:
            req, onset = demand.requirement(k)
        else:
            req, onset = ce.EPS_MIN, 0
        target = ce.WALL - req if req > ce.EPS_MIN else None
        fallback = plan.shift()
        fb_prof = prof[1:] if len(prof) > 1 else prof
        cands = ce.generate(pred, k, xhat, u_q, fallback, target, ell_op)
        pred.count(cands)
        profiles = [c[0].profile() for c in cands]
        sel = ce.select(cands, profiles, fb_prof, req, onset)
        if sel == 0:
            new_plan, new_prof, alpha = fallback, fb_prof, Fr(0)
            new_issued = issued[1:] if len(issued) > 1 else issued
            assert k_met != k, "the operation must begin with a fresh plan"
        else:
            new_plan, new_prof = cands[sel - 1][0], profiles[sel - 1]
            alpha = cands[sel - 1][1]
            new_issued = [k] * max(1, len(new_plan.pairs))
        p1, v1, ex = ce.realize(p, v, u_exec)
        if not ex and seg.hi == ce.RAMP:
            # The segment touches the ramp only at one end, at one instant, so
            # the braking that its invalid label overstates never acts: the
            # realized motion is the planned one, exactly, and no alarm is
            # raised. (This happens in two runs of Section 7.3, never in the
            # runs of Chapter 3.)
            p1, v1, ex, exercised = seg.p1, seg.v1, True, False
        alarm = not (ex and (p1, v1) == xhat)
        if log and seg is not None and beta_exec > ce.B_GLOBAL:
            records.append(dict(
                seed=world.seed, predictor=pred.name, issued=issued[0], period=k,
                lo=seg.lo, hi=seg.hi, beta=beta_exec, command=u_exec,
                used=int(u_exec < 0 and -u_exec > ce.B_GLOBAL), realized=b_real,
                valid=int(valid), exercised=int(exercised), alarm=int(alarm)))
        rows.append(dict(
            k=k, t=k * T, p=p, v=v, u=u_exec, beta=beta_exec,
            level=prof[0], rest=plan.p_rest,
            req_exec=req_prev, onset_level_exec=ce.level_at(prof, onset_prev),
            mode=q, q=u_q, alpha=alpha, sel=sel, eps=new_prof[0],
            reserve=new_plan.reserve(), plan=new_plan,
            eps_on=ce.level_at(new_prof, onset), req=req, onset=onset,
            p_rest=new_plan.p_rest, ell=plan.ell, alarm=int(alarm),
            valid=valid, exercised=exercised))
        if alarm:
            rows.append(dict(k=k + 1, t=(k + 1) * T, p=p1, v=v1, final=True))
            return rows, summarize(rows, pred, world, kd, k_met, overload,
                                   alarm_state=(p1, v1), ell=plan.ell), records
        pred.observe(min(p, p1), max(p, p1), p1)
        p, v = p1, v1
        plan, prof, issued = new_plan, new_prof, new_issued
        req_prev, onset_prev = req, onset
    rows.append(dict(k=k_end, t=k_end * T, p=p, v=v, final=True))
    return rows, summarize(rows, pred, world, kd, k_met, overload), records


def summarize(rows, pred, world, kd, k_met, overload, alarm_state=None, ell=Fr(0)):
    """What a run shows: arrival, the operation, the labels, alarms and
    violations. The governed run ends at the first alarm; what follows is
    calculated from the measured state at the alarm, not simulated. Every
    alarm occurs on the ramp (checked below), where maximal braking, the rest
    feedback, decelerates the cart at the global capability, 0.5 m/s^2; the
    function of Chapter 3 integrates that motion to rest in the true world.
    While the cart moves, no single-trajectory funnel contains every state
    that maximal braking can reach in the worlds consistent with the global
    capability, so a restart can be certified only at rest, by a plan that
    holds the cart where it stopped: exactly when it stops at or before the
    floor, which is the test p + v^2/(2 * 0.5) <= 24.5 m of Chapter 3 at the
    alarm state. A violation is motion beyond p + l = 25 m."""
    body = [x for x in rows if not x.get("final")]
    ends = [x["p"] for x in rows[1:]]
    k0 = ce.period(ce.T_ANNOUNCE)
    arr = next((x["k"] for x in body if x["p"] >= ce.WORK_ZONE), None)
    back = next((x["k"] for x in body if x["k"] >= k0 and x["u"] < 0 and x["v"] <= 0), None)
    s = dict(seed=world.seed, predictor=pred.name, ramp=world.ramp,
             survey_error=world.survey_error, s_hat=getattr(pred, "s_hat", None),
             standing_at_start=getattr(pred, "standing_at_start", True),
             arrival=None if arr is None else arr * T,
             met=k_met is not None and k_met == kd, overload=overload,
             issued=pred.issued, raised=pred.raised, invalid=pred.invalid,
             periods=len(body),
             executed_raised=sum(1 for x in body if x["beta"] > ce.B_GLOBAL),
             executed_invalid=sum(1 for x in body if not x["valid"]),
             exercised=sum(1 for x in body if x["exercised"]),
             first_invalid=next((x["k"] for x in body if not x["valid"]), None),
             backaway=None if back is None else back * T,
             onset_position=body[kd]["p"] if kd < len(body) else None,
             onset_level=body[kd]["level"] if kd < len(body) else None,
             tool_level=body[kd]["eps"] if k_met is not None else None,
             min_level=min(x["eps"] for x in body),
             max_p_plus_ell=max(max(x["p"], e) + x["ell"] for x, e in zip(body, ends)),
             alarm=None, restart=None, stop=None, violation=False)
    s["violation"] = s["max_p_plus_ell"] > ce.WALL
    if alarm_state is not None:
        assert alarm_state[0] >= ce.RAMP, "every alarm of these runs occurs on the ramp"
        certified, track = ce.continue_after_alarm(alarm_state)
        stop = track[-1][1] if track else float(alarm_state[0])
        s.update(alarm=body[-1]["k"] + 1, restart=certified, stop=stop,
                 violation=s["violation"] or stop + float(ell) > float(ce.WALL))
    return s


# ----------------------------------------------------------------------------
# The experiments
# ----------------------------------------------------------------------------
def family_run(spec):
    """One run in the family of worlds, with the operation of Chapter 3."""
    name, seed = spec
    world, op = World(seed), Operation()
    pred = make_predictor(name, world)
    rows, s, records = run(pred, world, correct_demand(op), op, log=True)
    s["trace"] = [(x["k"], x["p"], x["v"], x["u"], x["beta"], x["alpha"], x["sel"])
                  for x in rows if not x.get("final")]
    return s, records


def chapter3_consistency():
    """The loop of this script reproduces the runs of Chapter 3: with the
    world, the operation and the predictors of Chapter 3, every row that
    cart_example.run writes is the same."""
    keys = ["k", "t", "p", "v", "u", "beta", "mode", "q", "alpha", "sel", "eps",
            "eps_on", "req", "onset", "p_rest", "ell", "alarm", "level", "reserve"]
    same = True
    for name, kind in (("fixed", "fixed"), ("reactive", "reactive"),
                       ("map", "forecasting"), ("persisting", "persisting")):
        rows_ce, _, _ = ce.run(kind)
        op = Operation()
        rows, _, _ = run(make_predictor(name, chapter3_world()), chapter3_world(),
                         correct_demand(op), op)
        a = [{c: x.get(c) for c in keys} for x in rows_ce]
        b = [{c: x.get(c) for c in keys} for x in rows]
        same = same and a == b
    return same


def demand_cases():
    """Errors injected into the requirement: the world, the operation and
    the capability predictor (the map) are those of Chapter 3."""
    op = Operation()
    k0, kd, lvl = ce.period(op.announce), ce.period(op.onset), op.level()
    cases = [("as stated", k0, kd, lvl)]
    for na in (30, 21, 20, 10, 0):
        cases.append((f"lead time {ce.fmt(na * T)} s", kd - na, kd, lvl))
    for level in (Fr(6, 5), Fr(5, 4), Fr(7, 5), Fr(8, 5)):
        cases.append((f"level {ce.fmt(level)} m", k0, kd, level))
    for d in (-10, -5, 1, 5):
        cases.append((f"onset {'+' if d > 0 else ''}{ce.fmt(d * T)} s", k0, kd + d, lvl))
    return cases


def demand_run(case):
    label, k0, kd_pred, level = case
    op = Operation()
    rows, s, _ = run(Map(), chapter3_world(), DemandPredictor(k0, kd_pred, level), op)
    s["case"] = label
    s["final_position"] = rows[-1]["p"]
    s["tool_tip"] = rows[-1]["p"] + op.ell if s["met"] else None
    s["holding_at_work_zone"] = sum(
        1 for x in rows if not x.get("final") and x["k"] >= ce.period(op.announce)
        and x["k"] < ce.period(op.onset) and x["p"] >= ce.WORK_ZONE) * T
    return s


BIASES = (Fr(-1), Fr(-1, 2), Fr(-1, 5), Fr(0), Fr(1, 20), Fr(1, 10), Fr(1, 5),
          Fr(3, 10), Fr(7, 20), Fr(1, 2), Fr(1))


def bias_run(bias):
    """An error injected into the survey: in the world of Chapter 3, the
    surveyed ramp start is 20 m + bias, and the predictor uses it without
    an offset, as a point estimate. A negative bias understates the braking
    available and leaves every label valid; a positive one overstates it."""
    world, op = chapter3_world(), Operation()
    pred = Survey(RAMP_CH3 + bias, offset=Fr(0))
    rows, s, _ = run(pred, world, correct_demand(op), op)
    s["bias"] = bias
    return s


# ----------------------------------------------------------------------------
# Section 7.3: calibration and validation of the survey's coverage
# ----------------------------------------------------------------------------
UNRELIABLE_BOUND = Fr(3, 5)       # error bound of an unreliable survey [m]
UNRELIABLE = Fr(1, 5)             # share of unreliable surveys, as calibrated
UNRELIABLE_SHIFTED = Fr(1, 2)     # ... after the shift
CHANGE = Fr(1, 2)                 # ground softened before the ramp since the survey [m]
CAL_SEEDS = range(1001, 1101)     # the 100 calibration worlds
EVAL_SEEDS = range(2001, 4001)    # the 2000 held-out worlds of the evaluation
DELTA = Fr(1, 20)                 # miscoverage that the calibration targets
GAMMA = Fr(1, 20)                 # the evaluation bounds hold with confidence 1 - GAMMA
V_CAUTIOUS = Fr(3, 2)             # speed requested by the cautious proposal method [m/s]
T_END_LOOP = Fr(30)               # end of the runs without the operation [s]
CONDITIONS = {                    # (share of unreliable surveys, change since the survey)
    "as calibrated": (UNRELIABLE, Fr(0)),
    "shift": (UNRELIABLE_SHIFTED, Fr(0)),
    "stale": (UNRELIABLE, CHANGE),
}
RULES = ("point", "stated", "calibrated", "worst case")


class SurveyWorld(World):
    """A world of Section 7.3. The ramp start is drawn as in Section 7.1.
    The survey is unreliable with probability 'unreliable', its error then
    drawn from [-0.60, 0.60] m, and precise otherwise, its error drawn from
    [-0.20, 0.20] m, both on a 1 cm grid; no record says which. If 'change'
    is positive, the ground has softened over that length before the ramp
    since the survey, so the ramp now starts that much earlier than when
    the survey was made. The score is the surveyed start minus the true
    start: the labelling map of a survey rule with offset q is correct for
    every segment if and only if the score is at most q."""

    def __init__(self, seed, unreliable=UNRELIABLE, change=Fr(0)):
        rng = random.Random(seed)
        self.seed = seed
        ramp = Fr(round((17 + 4 * rng.random()) * 100), 100)
        u, w = rng.random(), rng.random()
        self.unreliable_survey = u < unreliable
        cm = int((UNRELIABLE_BOUND if self.unreliable_survey else SURVEY_BOUND) * 100)
        self.survey_error = Fr(round((2 * w - 1) * cm), 100)
        self.surveyed = ramp + self.survey_error
        self.ramp = ramp - change
        self.score = self.surveyed - self.ramp


def condition_world(seed, condition):
    unreliable, change = CONDITIONS[condition]
    return SurveyWorld(seed, unreliable, change)


def calibrate(scores, delta=DELTA):
    """Split conformal calibration of the offset: the m-th smallest of the
    n scores, m = ceil((n + 1)(1 - delta)). If m > n, no offset is supported
    and every label must be the global capability (returned as None)."""
    n = len(scores)
    m = math.ceil((n + 1) * (1 - delta))
    return m, (sorted(scores)[m - 1] if m <= n else None)


def calibration():
    worlds = [SurveyWorld(seed) for seed in CAL_SEEDS]
    m, q_hat = calibrate([w.score for w in worlds])
    return worlds, m, q_hat


def offsets(q_hat):
    return {"point": Fr(0), "stated": SURVEY_BOUND, "calibrated": q_hat,
            "worst case": UNRELIABLE_BOUND}


def binomial_cdf(n, x, p):
    """P(Bin(n, p) <= x), exactly, for a fraction p."""
    return sum((math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(x + 1)),
               Fr(0))


def upper_bound(x, n, gamma=GAMMA, bits=24):
    """One-sided upper confidence bound of level 1 - gamma on a binomial
    proportion, from x failures in n independent trials, in the one-sided
    form of the exact bound of Clopper and Pearson: the p at which
    P(Bin(n, p) <= x) = gamma. Found by bisection
    in exact integer arithmetic on the multiples of 2^-bits and returned as
    the upper end of the final interval, so never below the exact value."""
    if x >= n:
        return Fr(1)
    b = 1 << bits
    lo, hi = 0, b                 # P(Bin(n, lo/b) <= x) >= gamma > P(Bin(n, hi/b) <= x)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        a, c = mid, b - mid
        term = c ** n             # b^n P(Bin(n, a/b) = 0)
        total = term
        for i in range(x):        # b^n P(Bin(n, a/b) = i + 1), exactly
            term = term * (n - i) * a // ((i + 1) * c)
            total += term
        if total * gamma.denominator >= gamma.numerator * b ** n:
            lo = mid
        else:
            hi = mid
    return Fr(hi, b)


class CautiousProposals:
    """A cautious proposal method: it requests a speed of 1.5 m/s until the
    measured position first reaches the work zone, and 'hold' from then on.
    At that speed, braking at the global capability stops the cart within
    1.5^2 / (2 * 0.5) = 2.25 m, and no survey places the ramp start beyond
    21 m + 0.6 m, so the governor can always bring the cart to rest before
    the floor by braking where every label is the global capability."""

    name = "cautious"

    def __init__(self):
        self.arrived = False

    def __call__(self, p, xhat):
        if p >= ce.WORK_ZONE:
            self.arrived = True
        if self.arrived:
            return "hold", ce.proposed_command("hold", xhat)
        return "cruise", max(-ce.U_MAX, min(ce.U_MAX, (V_CAUTIOUS - xhat[1]) / T))


def rule_predictor(rule, offset, world):
    pred = Survey(world.surveyed, offset)
    pred.name = rule
    return pred


def rule_run(spec):
    """One run of Section 7.3 with the task and the operation of Chapter 3:
    the survey rule 'rule' with its offset, in world 'seed' under
    'condition'."""
    rule, offset, condition, seed = spec
    world, op = condition_world(seed, condition), Operation()
    pred = rule_predictor(rule, offset, world)
    rows, s, records = run(pred, world, correct_demand(op), op, log=True)
    s.update(rule=rule, offset=offset, condition=condition, score=world.score,
             surveyed=world.surveyed, unreliable_survey=world.unreliable_survey,
             executed_used=sum(x["used"] for x in records))
    return s


def loop_run(spec):
    """One run of the closed-loop comparison: the point estimate, in world
    'seed' as calibrated, with the proposal method 'method' and the task of
    Chapter 3 without the operation, until 30 s."""
    method, seed = spec
    world = SurveyWorld(seed)
    op = Operation(announce=T_END_LOOP, onset=T_END_LOOP)    # never reached
    demand = DemandPredictor(ce.period(T_END_LOOP), ce.period(T_END_LOOP), ce.EPS_MIN)
    pred = rule_predictor("point", Fr(0), world)
    proposals = CautiousProposals() if method == "cautious" else Chapter3Proposals()
    rows, s, records = run(pred, world, demand, op, t_end=T_END_LOOP, log=True,
                           proposals=proposals)
    s.update(method=method, score=world.score, surveyed=world.surveyed,
             executed_used=sum(x["used"] for x in records),
             executed_raised_invalid=sum(1 for x in records if not x["valid"]),
             used_invalid=sum(1 for x in records if x["used"] and not x["valid"]))
    return s


def dispatch(job):
    kind, arg = job
    return kind, {"family": family_run, "demand": demand_run, "bias": bias_run,
                  "rule": rule_run, "loop": loop_run}[kind](arg)


# ----------------------------------------------------------------------------
# Rounding, formatting and reporting
# ----------------------------------------------------------------------------
def r(x, nd):
    """x rounded half up to nd decimals, as a string (exact for fractions)."""
    if isinstance(x, float):
        x = Fr(x).limit_denominator(10 ** 12)
    q = Fr(1, 10 ** nd)
    n = (Fr(x) / q + Fr(1, 2)) // 1
    return f"{float(n * q):.{nd}f}"


def mean(xs):
    xs = list(xs)
    return sum((Fr(x) for x in xs), Fr(0)) / len(xs)


def yes(b):
    return "yes" if b else "no"


def family_facts(results):
    """The numbers that Section 7.1 quotes about the family of worlds."""
    facts = {}
    ramps = [World(seed).ramp for seed in SEEDS]
    errs = [World(seed).survey_error for seed in SEEDS]
    facts["family"] = dict(
        worlds=str(len(SEEDS)),
        ramp_range=f"{r(min(ramps), 2)} to {r(max(ramps), 2)} m",
        survey_error_range=f"{r(min(errs), 2)} to {r(max(errs), 2)} m",
        # a miscoverage delta gives len(SEEDS) clean worlds with probability
        # (1 - delta)^n; it exceeds 5 % for every delta below this value
        clean_worlds_consistent_with=f"{100 * (1 - 0.05 ** (1 / len(SEEDS))):.1f} %")
    traces, by = {}, {}
    for name in PREDICTORS:
        rs = [s for s, _ in results if s["predictor"] == name]
        recs = [x for s, rec in results if s["predictor"] == name for x in rec]
        traces[name] = {s["seed"]: s["trace"] for s in rs}
        by[name] = {s["seed"]: s for s in rs}
        arr = [s["arrival"] for s in rs if s["arrival"] is not None]
        f = dict(
            arrived=f"{len(arr)} of {len(rs)}",
            arrival=(r(mean(arr), 2) + " s") if arr else "-",
            arrival_range=(f"{r(min(arr), 1)} to {r(max(arr), 1)} s") if arr else "-",
            operation_performed=f"{sum(s['met'] for s in rs)} of {len(rs)}",
            candidate_labels=str(sum(s["issued"] for s in rs)),
            candidate_labels_above_global=str(sum(s["raised"] for s in rs)),
            invalid_candidate_labels=str(sum(s["invalid"] for s in rs)),
            runs_with_invalid_candidate_label=str(sum(s["invalid"] > 0 for s in rs)),
            executed_labels_above_global=str(len(recs)),
            executed_labels_used=str(sum(x["used"] for x in recs)),
            used_by_command=", ".join(
                f"{sum(1 for x in recs if x['used'] and x['command'] == u)} at "
                f"{ce.fmt(u)} m/s^2"
                for u in sorted({x["command"] for x in recs if x["used"]})) or "-",
            executed_invalid_labels=str(sum(s["executed_invalid"] for s in rs)),
            exercised_invalid_labels=str(sum(s["exercised"] for s in rs)),
            alarms=str(sum(s["alarm"] is not None for s in rs)),
            violations=str(sum(s["violation"] for s in rs)),
            min_level=r(min(s["min_level"] for s in rs), 2) + " m")
        if name == "persisting":
            al = [s["alarm"] for s in rs if s["alarm"] is not None]
            f["alarm_range"] = f"{r(min(al) * T, 1)} to {r(max(al) * T, 1)} s"
            f["alarm_after_first_executed_invalid_label"] = ",".join(sorted(
                {str(s["alarm"] - s["first_invalid"]) for s in rs}))
            f["certified_restarts"] = str(sum(bool(s["restart"]) for s in rs))
            stops = [s["stop"] for s in rs if s["alarm"] is not None]
            f["stop_range_after_alarm"] = f"{min(stops):.1f} to {max(stops):.1f} m"
        else:
            f["largest_p_plus_l"] = r(max(s["max_p_plus_ell"] for s in rs), 3) + " m"
        if name == "survey":
            gaps = [s["ramp"] - s["s_hat"] for s in rs]
            f["conservatism"] = (f"{r(min(gaps), 2)} to {r(max(gaps), 2)} m, "
                                 f"mean {r(mean(gaps), 3)} m")
        facts[name] = f
    later = [by["survey"][sd]["arrival"] - by["map"][sd]["arrival"] for sd in SEEDS]
    four = [s for name in ("fixed", "reactive", "survey", "map") for s in by[name].values()]
    facts["comparison of the family"] = dict(
        candidate_labels_of_the_four_with_claims=str(sum(s["issued"] for s in four)),
        invalid_among_them=str(sum(s["invalid"] for s in four)),
        reactive_labels_above_global=(
            f"{sum(s['raised'] for s in by['reactive'].values())}, "
            + ("all for standing still at the start"
               if all(s["standing_at_start"] for s in by["reactive"].values())
               else "not all for standing still at the start")),
        reactive_equals_fixed=yes(traces["reactive"] == traces["fixed"]),
        fixed_later_than_map=r(mean(by["fixed"][sd]["arrival"] - by["map"][sd]["arrival"]
                                    for sd in SEEDS), 2) + " s",
        survey_later_than_map=r(mean(later), 3) + " s",
        survey_later_than_map_worlds=f"{sum(1 for d in later if d > 0)} of {len(later)}",
        survey_later_than_map_max=r(max(later), 1) + " s")
    return facts


def label_record_example(results, seed=1):
    """The validation record of one executed label of the survey predictor:
    in world 1, the label of the last braking segment that the label of
    2 m/s^2 allowed, before the surveyed start minus the offset."""
    s, recs = next((s, rec) for s, rec in results
                   if s["predictor"] == "survey" and s["seed"] == seed)
    x = max((x for x in recs if x["used"]), key=lambda x: x["period"])
    w = World(seed)
    return dict(world=str(seed), ramp=f"{ce.fmt(w.ramp)} m",
                surveyed=f"{ce.fmt(w.ramp + w.survey_error)} m",
                s_hat=f"{ce.fmt(s['s_hat'])} m",
                issued_in_period=str(x["issued"]), period=str(x["period"]),
                region=f"{r(x['lo'], 3)} to {r(x['hi'], 3)} m",
                label=f"{ce.fmt(x['beta'])} m/s^2", command=f"{ce.fmt(x['command'])} m/s^2",
                realized=f"{ce.fmt(x['realized'])} m/s^2", valid=yes(x["valid"]),
                used=yes(x["used"]), alarm=yes(x["alarm"]),
                executed_labels_in_run=str(len(recs)),
                used_in_run=str(sum(y["used"] for y in recs)))


def operation_facts(demand_results):
    """The record of Section 7.2 that turns the operation into a demand, and
    the check of the assumption of reserve consumption in the run as
    stated."""
    op = Operation()
    k0, kd = ce.period(op.announce), ce.period(op.onset)
    s = next(x for x in demand_results if x["case"] == "as stated")
    return dict(
        consumption=f"{ce.fmt(op.consumption())} m",
        floor=f"{ce.fmt(ce.EPS_MIN)} m", level=f"{ce.fmt(op.level())} m",
        onset_period=str(kd), announced_in_period=str(k0),
        lead_time=f"{kd - k0} periods",
        level_in_force_at_onset=r(s["onset_level"], 2) + " m",
        level_of_first_plan_with_tool=r(s["tool_level"], 2) + " m",
        position_at_onset=r(s["onset_position"], 3) + " m",
        met=yes(s["met"]))


def demand_facts(demand_results):
    out = {}
    for s in demand_results:
        out[s["case"]] = (
            f"{'met' if s['met'] else 'overload'}; "
            f"back-away from {r(s['backaway'], 1) + ' s' if s['backaway'] is not None else '-'}; "
            f"at onset {r(s['onset_position'], 3)} m; "
            f"{'tool tip ' + r(s['tool_tip'], 2) + ' m; ' if s['tool_tip'] is not None else ''}"
            f"at the work zone {r(s['holding_at_work_zone'], 1)} s after the announcement; "
            f"alarms {0 if s['alarm'] is None else 1}; lowest level {r(s['min_level'], 2)} m")
    met = [s for s in demand_results if s["case"].startswith("lead time") and s["met"]]
    out["shortest_lead_time_met"] = ce.fmt(min(Fr(s["case"].split()[2]) for s in met)) + " s"
    out["all_without_alarm_or_violation"] = yes(all(
        s["alarm"] is None and not s["violation"] for s in demand_results))
    return out


def bias_facts(biased):
    out = {}
    for s in biased:
        if s["alarm"] is None:
            out[f"bias {ce.fmt(s['bias'])} m"] = (
                f"labels valid; arrival {r(s['arrival'], 1)} s; operation "
                f"{'performed' if s['met'] else 'not performed'}")
        else:
            out[f"bias {ce.fmt(s['bias'])} m"] = (
                f"{s['invalid']} invalid candidate labels; "
                f"{s['executed_invalid']} executed, {s['exercised']} used; "
                f"alarm at {r(s['alarm'] * T, 1)} s; "
                f"certified restart {yes(s['restart'])}; stops at {s['stop']:.2f} m; "
                f"violation {yes(s['violation'])}")
    return out


def pct(x, nd=1):
    """A proportion in per cent, rounded half up."""
    return r(100 * Fr(x), nd) + " %"


def pct_up(x, nd=1):
    """A proportion in per cent, rounded up: for upper bounds."""
    q = Fr(1, 10 ** nd)
    return f"{float(math.ceil(100 * Fr(x) / q) * q):.{nd}f} %"


def calibration_facts(cal_worlds, m, q_hat):
    """Section 7.3: the calibration, and what it promises given the
    calibration set. If the n + 1 scores are independent and identically
    distributed, the miscoverage given the calibration set exceeds a level a
    with probability at most P(Bin(n, a) <= n - m), with equality for
    continuous scores. Returns the offset of the m that keeps that
    miscoverage at most DELTA with probability at least 1 - GAMMA, and the
    facts."""
    n = len(cal_worlds)
    scores = sorted(w.score for w in cal_worlds)
    m_conf = next((mm for mm in range(m, n + 1)
                   if binomial_cdf(n, n - mm, DELTA) <= GAMMA), None)
    return scores[m_conf - 1], dict(
        calibration_worlds=str(n),
        unreliable_surveys=f"{sum(w.unreliable_survey for w in cal_worlds)} of {n}",
        score_range=f"{r(scores[0], 2)} to {r(scores[-1], 2)} m",
        m=str(m),
        q_hat=f"{r(q_hat, 2)} m",
        given_calibration_above_5_percent=pct_up(binomial_cdf(n, n - m, DELTA)),
        given_calibration_above_10_percent=pct_up(binomial_cdf(n, n - m, 2 * DELTA)),
        m_for_95_percent_confidence=str(m_conf),
        q_for_95_percent_confidence=f"{r(scores[m_conf - 1], 2)} m")


def heldout(q_hat, q_conf):
    """Section 7.3: the miscoverage of each rule on the held-out worlds,
    under each condition, with its upper confidence bound (two decimals
    below 1 %). A rule miscovers a world when the score exceeds its offset,
    that is, when its labelling map is incorrect for some segment, whatever
    the proposal method and the generator, so no run is needed."""
    rows, facts = [], {}
    offs = offsets(q_hat)
    for cond in CONDITIONS:
        worlds = [condition_world(seed, cond) for seed in EVAL_SEEDS]
        rows += [dict(condition=cond, seed=w.seed, ramp=w.ramp, surveyed=w.surveyed,
                      unreliable=w.unreliable_survey, score=w.score) for w in worlds]
        n = len(worlds)
        named = [(rule, offs[rule]) for rule in RULES]
        if cond == "as calibrated":
            named.append(("calibrated with 95 % confidence", q_conf))
        for rule, q in named:
            x = sum(1 for w in worlds if w.score > q)
            u = upper_bound(x, n)
            facts[f"{cond}: {rule}"] = (f"{x} of {n}, {pct(Fr(x, n))}; "
                                        f"at most {pct_up(u, 2 if u < Fr(1, 100) else 1)}")
    facts["worlds"] = str(len(EVAL_SEEDS))
    facts["bound_with_no_failure"] = pct_up(upper_bound(0, len(EVAL_SEEDS)), 2)
    for target in (Fr(1, 100), Fr(1, 1000)):
        # worlds needed for the bound with no failure to reach the target:
        # the smallest N with (1 - target)^N <= gamma, found exactly
        n_needed, clean = 1, 1 - target
        while clean > GAMMA:
            n_needed, clean = n_needed + 1, clean * (1 - target)
        facts[f"worlds_for_{ce.fmt(100 * target)}_percent"] = str(n_needed)
    return rows, facts


def rule_facts(results, family):
    """Section 7.3: the four rules in the 50 worlds, with the task and the
    operation of Chapter 3; arrivals compared with the map predictor in the
    same worlds (Section 7.1), whose ramp starts they share."""
    by_map = {s["seed"]: s["arrival"] for s, _ in family if s["predictor"] == "map"}
    out = {}
    for cond in CONDITIONS:
        for rule in RULES:
            rs = [s for s in results if s["condition"] == cond and s["rule"] == rule]
            if not rs:
                continue
            arr = [s for s in rs if s["arrival"] is not None]
            f = (f"offset {r(rs[0]['offset'], 2)} m; "
                 f"miscovered worlds {sum(s['score'] > s['offset'] for s in rs)}; "
                 f"worlds with an invalid candidate label {sum(s['invalid'] > 0 for s in rs)}; "
                 f"executed invalid labels {sum(s['executed_invalid'] for s in rs)}, "
                 f"exercised {sum(s['exercised'] for s in rs)}; "
                 f"runs with an executed invalid label and no alarm "
                 f"{sum(s['executed_invalid'] > 0 and s['alarm'] is None for s in rs)}; "
                 f"alarms {sum(s['alarm'] is not None for s in rs)}; "
                 f"certified restarts {sum(bool(s['restart']) for s in rs)}; "
                 f"violations {sum(s['violation'] for s in rs)}; "
                 f"arrived {len(arr)}; operation performed {sum(s['met'] for s in rs)}")
            if arr:
                f += f"; mean arrival {r(mean(s['arrival'] for s in arr), 2)} s"
                if cond != "stale":
                    f += (f"; later than with the map by "
                          f"{r(mean(s['arrival'] - by_map[s['seed']] for s in arr), 3)} s")
            out[f"{cond}: {rule}"] = f
            if cond == "as calibrated" and rule == "point":
                out["as calibrated: largest score"] = r(max(s["score"] for s in rs), 2) + " m"
            stops = [s["stop"] for s in rs if s["alarm"] is not None]
            if stops:
                out[f"{cond}: {rule}: stops after an alarm"] = (
                    f"{min(stops):.2f} to {max(stops):.2f} m")
    return out


def loop_facts(results, method):
    """Section 7.3: what the closed loop selects. The point estimate in the
    50 worlds as calibrated, without the operation, with one proposal
    method. The simulation checks every label; a robot can detect an
    invalid label only where it used it."""
    rs = [s for s in results if s["method"] == method]
    issued, invalid = sum(s["issued"] for s in rs), sum(s["invalid"] for s in rs)
    above = sum(s["raised"] for s in rs)
    raised = sum(s["executed_raised"] for s in rs)
    used = sum(s["executed_used"] for s in rs)
    used_invalid = sum(s["used_invalid"] for s in rs)
    arr = [s["arrival"] for s in rs if s["arrival"] is not None]
    return dict(
        miscovered_worlds=str(sum(s["score"] > 0 for s in rs)),
        worlds_with_invalid_candidate_label=str(sum(s["invalid"] > 0 for s in rs)),
        invalid_candidate_labels=f"{invalid} of {issued}, {pct(Fr(invalid, issued), 2)}",
        invalid_among_candidate_labels_above_global=(
            f"{invalid} of {above}, {pct(Fr(invalid, above), 2)}"),
        executed_labels_above_global=str(raised),
        executed_invalid_labels=str(sum(s["executed_raised_invalid"] for s in rs)),
        worlds_with_executed_invalid_label=str(sum(s["executed_raised_invalid"] > 0 for s in rs)),
        executed_labels_used=str(used),
        used_invalid_labels=str(used_invalid),
        invalid_among_used=pct(Fr(used_invalid, used), 2) if used else "-",
        alarms=str(sum(s["alarm"] is not None for s in rs)),
        certified_restarts=str(sum(bool(s["restart"]) for s in rs)),
        violations=str(sum(s["violation"] for s in rs)),
        arrived=f"{len(arr)} of {len(rs)}",
        mean_arrival=(r(mean(arr), 2) + " s") if arr else "-")


def write_csv(path, cols, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for row in rows:
            out = []
            for c in cols:
                x = row.get(c)
                if x is None:
                    out.append("")
                elif isinstance(x, bool):
                    out.append(yes(x))
                elif isinstance(x, float):
                    out.append(f"{x:.4f}")
                else:
                    out.append(ce.fmt(x))
            w.writerow(out)


RUN_COLS = ["predictor", "seed", "ramp", "survey_error", "s_hat", "arrival", "met",
            "issued", "raised", "invalid", "executed_raised", "executed_invalid",
            "exercised", "alarm", "restart", "stop", "violation", "min_level"]
REC_COLS = ["predictor", "seed", "issued", "period", "lo", "hi", "beta", "command",
            "used", "realized", "valid", "exercised", "alarm"]
DEMAND_COLS = ["case", "met", "overload", "backaway", "onset_position",
               "holding_at_work_zone", "alarm", "violation", "min_level"]
BIAS_COLS = ["bias", "s_hat", "issued", "invalid", "executed_invalid", "exercised",
             "first_invalid", "alarm", "restart", "stop", "violation", "arrival", "met"]
CAL_COLS = ["seed", "ramp", "surveyed", "survey_error", "unreliable_survey", "score"]
HELD_COLS = ["condition", "seed", "ramp", "surveyed", "unreliable", "score"]
RULE_COLS = ["condition", "rule", "offset", "seed", "ramp", "surveyed", "score", "s_hat",
             "arrival", "met", "issued", "invalid", "executed_raised", "executed_invalid",
             "exercised", "alarm", "restart", "stop", "violation"]
LOOP_COLS = ["method", "seed", "ramp", "surveyed", "score", "arrival", "issued", "raised",
             "invalid", "executed_raised", "executed_raised_invalid", "executed_used",
             "used_invalid", "alarm", "restart", "stop", "violation"]


def exercise(args):
    """Exercise 7.1: errors of the reader's choice injected into one
    forecast at a time, in the world and with the operation of Chapter 3, as
    in Section 7.2.3. Prints one line per error and writes no file."""
    op = Operation()
    k0, kd, lvl = ce.period(op.announce), ce.period(op.onset), op.level()
    for b in args.bias:
        x = bias_run(b)
        print(f"survey bias {ce.fmt(b)} m: {x['invalid']} invalid candidate labels, "
              f"{x['executed_invalid']} executed, {x['exercised']} of them used; "
              + (f"arrival {r(x['arrival'], 1)} s; operation "
                 f"{'performed' if x['met'] else 'not performed'}; no alarm"
                 if x["alarm"] is None else
                 f"alarm at {r(x['alarm'] * T, 1)} s; certified restart {yes(x['restart'])}; "
                 f"stops at {x['stop']:.2f} m; violation {yes(x['violation'])}"))
    cases = [(f"onset {'+' if d > 0 else ''}{ce.fmt(d * T)} s", k0, kd + d, lvl)
             for d in args.onset_error]
    cases += [(f"level {ce.fmt(v)} m", k0, kd, v) for v in args.level]
    cases += [(f"lead time {ce.fmt(n * T)} s", kd - n, kd, lvl) for n in args.lead]
    for case in cases:
        x = demand_run(case)
        print(f"requirement with {case[0]}: "
              f"{'met' if x['met'] else 'overload'}; back-away from "
              f"{r(x['backaway'], 1) + ' s' if x['backaway'] is not None else '-'}; "
              f"at onset {r(x['onset_position'], 3)} m; "
              f"{'tool tip ' + r(x['tool_tip'], 2) + ' m; ' if x['tool_tip'] is not None else ''}"
              f"at the work zone {r(x['holding_at_work_zone'], 1)} s after the announcement; "
              f"alarms {0 if x['alarm'] is None else 1}; lowest level {r(x['min_level'], 2)} m; "
              f"violation {yes(x['violation'])}")


def main():
    ap = argparse.ArgumentParser(description="Chapter 7: capability and demand predictors")
    ap.add_argument("--check", action="store_true", help="check every number quoted in Chapter 7")
    ap.add_argument("--jobs", type=int, default=2, help="processes (default 2)")
    ap.add_argument("--bias", type=Fr, action="append", default=[], metavar="B",
                    help="Exercise 7.1: a survey biased by B m, used without offset "
                         "(repeatable); prints, writes no file")
    ap.add_argument("--onset-error", type=int, action="append", default=[], metavar="N",
                    help="Exercise 7.1: the onset predicted N periods late (N < 0: early)")
    ap.add_argument("--level", type=Fr, action="append", default=[], metavar="L",
                    help="Exercise 7.1: the requirement's level L m")
    ap.add_argument("--lead", type=int, action="append", default=[], metavar="N",
                    help="Exercise 7.1: the requirement issued N periods before the onset")
    args = ap.parse_args()
    if args.bias or args.onset_error or args.level or args.lead:
        if args.check:
            ap.error("the options of Exercise 7.1 are used without --check")
        exercise(args)
        return
    out = os.path.join(HERE, "results")
    os.makedirs(out, exist_ok=True)

    cal_worlds, m, q_hat = calibration()
    offs = offsets(q_hat)
    jobs = [("family", (name, seed)) for name in PREDICTORS for seed in SEEDS]
    jobs += [("demand", case) for case in demand_cases()]
    jobs += [("bias", b) for b in BIASES]
    jobs += [("rule", (rule, offs[rule], "as calibrated", seed))
             for rule in RULES for seed in SEEDS]
    jobs += [("rule", ("calibrated", q_hat, cond, seed))
             for cond in ("shift", "stale") for seed in SEEDS]
    jobs += [("rule", ("worst case", offs["worst case"], "stale", seed)) for seed in SEEDS]
    jobs += [("loop", (method, seed)) for method in ("chapter 3", "cautious")
             for seed in SEEDS]
    if args.jobs > 1:
        with Pool(min(args.jobs, os.cpu_count() or 1)) as pool:
            done = pool.map(dispatch, jobs, chunksize=1)
    else:
        done = [dispatch(j) for j in jobs]
    family = [x for kind, x in done if kind == "family"]
    demand = [x for kind, x in done if kind == "demand"]
    biased = [x for kind, x in done if kind == "bias"]
    rules = [x for kind, x in done if kind == "rule"]
    loops = [x for kind, x in done if kind == "loop"]

    facts = {"consistency": {"same_as_chapter_3": yes(chapter3_consistency())}}
    facts.update(family_facts(family))
    facts["label record"] = label_record_example(family)
    facts["operation"] = operation_facts(demand)
    facts["demand errors"] = demand_facts(demand)
    facts["survey errors"] = bias_facts(biased)
    q_conf, facts["calibration"] = calibration_facts(cal_worlds, m, q_hat)
    held_rows, facts["held-out evaluation"] = heldout(q_hat, q_conf)
    facts["rules in 50 worlds"] = rule_facts(rules, family)
    facts["closed loop: chapter 3"] = loop_facts(loops, "chapter 3")
    facts["closed loop: cautious"] = loop_facts(loops, "cautious")

    write_csv(os.path.join(out, "runs.csv"), RUN_COLS, [s for s, _ in family])
    write_csv(os.path.join(out, "validation.csv"), REC_COLS,
              [x for s, rec in family if s["predictor"] in ("survey", "persisting")
               for x in rec])
    write_csv(os.path.join(out, "demand.csv"), DEMAND_COLS, demand)
    write_csv(os.path.join(out, "survey-bias.csv"), BIAS_COLS, biased)
    write_csv(os.path.join(out, "calibration.csv"), CAL_COLS,
              [dict(seed=w.seed, ramp=w.ramp, surveyed=w.surveyed,
                    survey_error=w.survey_error, unreliable_survey=w.unreliable_survey,
                    score=w.score) for w in cal_worlds])
    write_csv(os.path.join(out, "held-out.csv"), HELD_COLS, held_rows)
    write_csv(os.path.join(out, "rules.csv"), RULE_COLS, rules)
    write_csv(os.path.join(out, "loop.csv"), LOOP_COLS, loops)

    lines, failures = [], []
    for group, items in facts.items():
        lines.append(f"[{group}]")
        for key, val in items.items():
            lines.append(f"  {key} = {val}")
            want = EXPECTED.get(group, {}).get(key)
            if args.check and want is not None and want != val:
                failures.append(f"{group}.{key}: expected {want!r}, got {val!r}")
    if args.check:
        failures += [f"{group}.{key}: not computed" for group, items in EXPECTED.items()
                     for key in items if key not in facts.get(group, {})]
        n = sum(len(v) for v in EXPECTED.values())
        lines.append(f"[check] {n} values quoted in Chapter 7: " +
                     ("all as quoted" if not failures else "FAILED: " + "; ".join(failures)))
    with open(os.path.join(out, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if failures:
        raise SystemExit(1)


EXPECTED = {   # every number quoted in Chapter 7, as in the text
    'consistency': {
        'same_as_chapter_3': 'yes',
    },
    'family': {
        'worlds': '50',
        'ramp_range': '17.05 to 20.86 m',
        'survey_error_range': '-0.20 to 0.19 m',
        'clean_worlds_consistent_with': '5.8 %',
    },
    'fixed': {
        'arrival': '9.90 s',
        'arrival_range': '9.9 to 9.9 s',
        'operation_performed': '50 of 50',
        'invalid_candidate_labels': '0',
        'executed_invalid_labels': '0',
        'alarms': '0',
        'violations': '0',
    },
    'reactive': {
        'arrival': '9.90 s',
        'arrival_range': '9.9 to 9.9 s',
        'operation_performed': '50 of 50',
        'invalid_candidate_labels': '0',
        'executed_invalid_labels': '0',
        'alarms': '0',
        'violations': '0',
    },
    'survey': {
        'arrival': '8.80 s',
        'arrival_range': '8.4 to 9.1 s',
        'operation_performed': '50 of 50',
        'invalid_candidate_labels': '0',
        'executed_invalid_labels': '0',
        'alarms': '0',
        'violations': '0',
        'executed_labels_above_global': '2579',
        'executed_labels_used': '993',
        'used_by_command': '959 at -2 m/s^2, 34 at -1 m/s^2',
        'conservatism': '0.01 to 0.40 m, mean 0.194 m',
    },
    'map': {
        'arrival': '8.76 s',
        'arrival_range': '8.3 to 9.1 s',
        'operation_performed': '50 of 50',
        'invalid_candidate_labels': '0',
        'executed_invalid_labels': '0',
        'alarms': '0',
        'violations': '0',
    },
    'persisting': {
        'arrived': '0 of 50',
        'operation_performed': '0 of 50',
        'candidate_labels': '304454',
        'invalid_candidate_labels': '116473',
        'runs_with_invalid_candidate_label': '50',
        'executed_invalid_labels': '50',
        'exercised_invalid_labels': '50',
        'alarms': '50',
        'violations': '50',
        'alarm_range': '4.4 to 5.2 s',
        'alarm_after_first_executed_invalid_label': '1',
        'certified_restarts': '0',
        'stop_range_after_alarm': '35.4 to 46.8 m',
    },
    'comparison of the family': {
        'candidate_labels_of_the_four_with_claims': '6365330',
        'invalid_among_them': '0',
        'reactive_labels_above_global': '100, all for standing still at the start',
        'reactive_equals_fixed': 'yes',
        'fixed_later_than_map': '1.14 s',
        'survey_later_than_map': '0.042 s',
        'survey_later_than_map_max': '0.1 s',
    },
    'label record': {
        'world': '1',
        'ramp': '17.54 m',
        'surveyed': '17.68 m',
        's_hat': '17.48 m',
        'issued_in_period': '48',
        'period': '49',
        'region': '17.200 to 17.470 m',
        'label': '2 m/s^2',
        'command': '-2 m/s^2',
        'realized': '2 m/s^2',
        'valid': 'yes',
        'used': 'yes',
        'alarm': 'no',
    },
    'operation': {
        'consumption': '0.8 m',
        'floor': '0.5 m',
        'level': '1.3 m',
        'onset_period': '200',
        'announced_in_period': '140',
        'lead_time': '60 periods',
        'level_in_force_at_onset': '1.30 m',
        'level_of_first_plan_with_tool': '0.50 m',
        'position_at_onset': '23.700 m',
        'met': 'yes',
    },
    'demand errors': {
        'as stated': 'met; back-away from 18.0 s; at onset 23.700 m; tool tip 24.50 m; at the work zone 5.5 s after the announcement; alarms 0; lowest level 0.50 m',
        'lead time 3 s': 'met; back-away from 18.0 s; at onset 23.700 m; tool tip 24.50 m; at the work zone 5.5 s after the announcement; alarms 0; lowest level 0.50 m',
        'lead time 2.1 s': 'met; back-away from 18.0 s; at onset 23.700 m; tool tip 24.50 m; at the work zone 5.5 s after the announcement; alarms 0; lowest level 0.50 m',
        'lead time 2 s': 'overload; back-away from 18.1 s; at onset 23.710 m; at the work zone 5.6 s after the announcement; alarms 0; lowest level 0.50 m',
        'lead time 1 s': 'overload; back-away from 19.1 s; at onset 24.298 m; at the work zone 6.0 s after the announcement; alarms 0; lowest level 0.50 m',
        'lead time 0 s': 'overload; back-away from -; at onset 24.500 m; at the work zone 6.0 s after the announcement; alarms 0; lowest level 0.50 m',
        'level 1.2 m': 'overload; back-away from 18.1 s; at onset 23.795 m; at the work zone 5.6 s after the announcement; alarms 0; lowest level 0.50 m',
        'level 1.25 m': 'overload; back-away from 18.2 s; at onset 23.740 m; at the work zone 5.7 s after the announcement; alarms 0; lowest level 0.50 m',
        'level 1.4 m': 'met; back-away from 17.9 s; at onset 23.598 m; tool tip 24.40 m; at the work zone 5.4 s after the announcement; alarms 0; lowest level 0.50 m',
        'level 1.6 m': 'met; back-away from 17.7 s; at onset 23.378 m; tool tip 24.17 m; at the work zone 5.2 s after the announcement; alarms 0; lowest level 0.50 m',
        'onset -1 s': 'met; back-away from 17.0 s; at onset 23.700 m; tool tip 24.50 m; at the work zone 4.5 s after the announcement; alarms 0; lowest level 0.50 m',
        'onset -0.5 s': 'met; back-away from 17.5 s; at onset 23.700 m; tool tip 24.50 m; at the work zone 5.0 s after the announcement; alarms 0; lowest level 0.50 m',
        'onset +0.1 s': 'overload; back-away from 18.1 s; at onset 23.710 m; at the work zone 5.6 s after the announcement; alarms 0; lowest level 0.50 m',
        'onset +0.5 s': 'overload; back-away from 18.5 s; at onset 23.938 m; at the work zone 6.0 s after the announcement; alarms 0; lowest level 0.50 m',
        'shortest_lead_time_met': '2.1 s',
        'all_without_alarm_or_violation': 'yes',
    },
    'survey errors': {
        'bias -1 m': 'labels valid; arrival 8.8 s; operation performed',
        'bias -0.5 m': 'labels valid; arrival 8.7 s; operation performed',
        'bias -0.2 m': 'labels valid; arrival 8.6 s; operation performed',
        'bias 0 m': 'labels valid; arrival 8.6 s; operation performed',
        'bias 0.05 m': '43 invalid candidate labels; 1 executed, 1 used; alarm at 5.6 s; certified restart yes; stops at 24.50 m; violation no',
        'bias 0.1 m': '77 invalid candidate labels; 1 executed, 1 used; alarm at 5.6 s; certified restart yes; stops at 24.36 m; violation no',
        'bias 0.2 m': '127 invalid candidate labels; 1 executed, 1 used; alarm at 5.6 s; certified restart yes; stops at 24.48 m; violation no',
        'bias 0.3 m': '167 invalid candidate labels; 1 executed, 1 used; alarm at 5.6 s; certified restart yes; stops at 24.48 m; violation no',
        'bias 0.35 m': '237 invalid candidate labels; 1 executed, 1 used; alarm at 5.6 s; certified restart no; stops at 24.99 m; violation no',
        'bias 0.5 m': '335 invalid candidate labels; 1 executed, 1 used; alarm at 5.5 s; certified restart no; stops at 25.59 m; violation yes',
        'bias 1 m': '568 invalid candidate labels; 1 executed, 1 used; alarm at 5.4 s; certified restart no; stops at 27.12 m; violation yes',
    },
    'calibration': {
        'calibration_worlds': '100',
        'unreliable_surveys': '16 of 100',
        'm': '96',
        'q_hat': '0.40 m',
        'given_calibration_above_5_percent': '43.6 %',
        'given_calibration_above_10_percent': '2.4 %',
        'm_for_95_percent_confidence': '99',
        'q_for_95_percent_confidence': '0.45 m',
    },
    'held-out evaluation': {
        'as calibrated: point': '973 of 2000, 48.7 %; at most 50.6 %',
        'as calibrated: stated': '144 of 2000, 7.2 %; at most 8.3 %',
        'as calibrated: calibrated': '68 of 2000, 3.4 %; at most 4.2 %',
        'as calibrated: worst case': '0 of 2000, 0.0 %; at most 0.15 %',
        'as calibrated: calibrated with 95 % confidence': '48 of 2000, 2.4 %; at most 3.1 %',
        'shift: calibrated': '166 of 2000, 8.3 %; at most 9.4 %',
        'stale: calibrated': '1408 of 2000, 70.4 %; at most 72.1 %',
        'stale: worst case': '528 of 2000, 26.4 %; at most 28.1 %',
        'worlds': '2000',
        'bound_with_no_failure': '0.15 %',
        'worlds_for_1_percent': '299',
        'worlds_for_0.1_percent': '2995',
    },
    'rules in 50 worlds': {
        'as calibrated: point': 'offset 0.00 m; miscovered worlds 28; worlds with an invalid candidate label 28; executed invalid labels 17, exercised 17; runs with an executed invalid label and no alarm 0; alarms 17; certified restarts 13; violations 0; arrived 33; operation performed 33; mean arrival 8.80 s; later than with the map by 0.024 s',
        'as calibrated: point: stops after an alarm': '24.20 to 24.62 m',
        'as calibrated: largest score': '0.28 m',
        'as calibrated: stated': 'offset 0.20 m; miscovered worlds 2; worlds with an invalid candidate label 2; executed invalid labels 2, exercised 2; runs with an executed invalid label and no alarm 0; alarms 2; certified restarts 2; violations 0; arrived 48; operation performed 48; mean arrival 8.81 s; later than with the map by 0.038 s',
        'as calibrated: calibrated': 'offset 0.40 m; miscovered worlds 0; worlds with an invalid candidate label 0; executed invalid labels 0, exercised 0; runs with an executed invalid label and no alarm 0; alarms 0; certified restarts 0; violations 0; arrived 50; operation performed 50; mean arrival 8.84 s; later than with the map by 0.084 s',
        'as calibrated: worst case': 'offset 0.60 m; miscovered worlds 0; worlds with an invalid candidate label 0; executed invalid labels 0, exercised 0; runs with an executed invalid label and no alarm 0; alarms 0; certified restarts 0; violations 0; arrived 50; operation performed 50; mean arrival 8.88 s; later than with the map by 0.120 s',
        'shift: calibrated': 'offset 0.40 m; miscovered worlds 1; worlds with an invalid candidate label 1; executed invalid labels 1, exercised 1; runs with an executed invalid label and no alarm 0; alarms 1; certified restarts 1; violations 0; arrived 49; operation performed 49; mean arrival 8.85 s; later than with the map by 0.090 s',
        'stale: calibrated': 'offset 0.40 m; miscovered worlds 34; worlds with an invalid candidate label 34; executed invalid labels 29, exercised 28; runs with an executed invalid label and no alarm 1; alarms 28; certified restarts 10; violations 0; arrived 22; operation performed 22; mean arrival 8.93 s',
        'stale: worst case': 'offset 0.60 m; miscovered worlds 18; worlds with an invalid candidate label 18; executed invalid labels 14, exercised 13; runs with an executed invalid label and no alarm 1; alarms 13; certified restarts 10; violations 0; arrived 37; operation performed 37; mean arrival 8.90 s',
    },
    'closed loop: chapter 3': {
        'miscovered_worlds': '28',
        'worlds_with_invalid_candidate_label': '28',
        'invalid_candidate_labels': '1957 of 834627, 0.23 %',
        'invalid_among_candidate_labels_above_global': '1957 of 214285, 0.91 %',
        'executed_labels_above_global': '2599',
        'executed_invalid_labels': '17',
        'worlds_with_executed_invalid_label': '17',
        'executed_labels_used': '1009',
        'used_invalid_labels': '17',
        'invalid_among_used': '1.68 %',
        'alarms': '17',
        'arrived': '33 of 50',
        'mean_arrival': '8.80 s',
    },
    'closed loop: cautious': {
        'miscovered_worlds': '28',
        'worlds_with_invalid_candidate_label': '28',
        'invalid_candidate_labels': '1150 of 721818, 0.16 %',
        'invalid_among_candidate_labels_above_global': '1150 of 308483, 0.37 %',
        'executed_labels_above_global': '6457',
        'executed_invalid_labels': '23',
        'worlds_with_executed_invalid_label': '20',
        'executed_labels_used': '0',
        'used_invalid_labels': '0',
        'alarms': '0',
        'arrived': '50 of 50',
        'mean_arrival': '17.00 s',
    },
}


if __name__ == "__main__":
    main()
