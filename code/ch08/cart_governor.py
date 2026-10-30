#!/usr/bin/env python3
"""
The viability governor of ALLOVI, step by step, for the cart.

Physical AI, Together, Chapter 8, Sections 8.1 to 8.3.

The cart, its control method (segments, generator and test), the capability
predictors and the proposal method are those of Chapter 3, imported from
../ch03/cart_example.py and used unchanged. This script implements the common
part of the governor explicitly. For every candidate of every period it
records the profile, the pair or the condition that fixes the level and the
onset level, whether the candidate is admitted, its two keys in the selection
rule and why it is or is not committed: the decision trace of Section 8.1.
With the configuration of Chapter 3, it reproduces every row of the runs of
Chapter 3 with the fixed, reactive and forecasting capability predictors and
selects the same candidate as the selection function of Chapter 3 in every
period.

Section 8.1
    The decision trace of the forecasting run of Chapter 3, and the reasons
    for every admission and every selection.

Section 8.2
    Five ways of meeting the tool operation of Chapter 3, due in period 200,
    with the forecasting capability predictor, each run until 25 s. The runs
    differ only in what the demand predictor issues:
    forecast-informed     announced in period 140 for its onset, as in
                          Chapter 3
    reactive-preparation  not announced before it is due
    fixed-conservatism    the reserve of the operation, 1.3 m, required from
                          the first pair on in every period until the
                          operation begins
    forecast-early        announced in period 140 for an onset forecast 1 s
                          early
    forecast-late         announced in period 140 for an onset forecast 1 s
                          late
    "Reactive" and "fixed" name ways of meeting the operation here, not the
    capability predictors of Chapter 3. In all five, if the operation is due
    and the plan in force does not hold its reserve, an overload occurs and
    the operation is postponed: to the onset that was forecast, if that is
    still ahead, and otherwise by 21 periods, the shortest postponement that
    this generator can meet from rest (Section 3.2.3; checked below), with
    the new onset announced. In Chapter 3, such an operation is dropped.

Section 8.3
    The governor at run time (Algorithm 8.1), with the configuration of
    Chapter 3: the forecasting predictor and the operation announced in
    period 140 for period 200, dropped if its reserve is not held when it is
    due; each run until 22 s.
    Time budgets: the governor evaluates the fresh candidates in the order in
    which the generator lists them until the deadline, and a candidate not
    evaluated counts as not certified:
    full            every candidate (the run of Chapter 3; checked)
    one-candidate   only the first, the candidate of ambition 1
    stall           none in periods 10 to 29, every one otherwise
    no-evaluation   none
    Alarms: the survey predictor of Section 7.2.3, used without an offset,
    with surveys biased by 0.2 m and 0.35 m. After the alarm, the rest
    feedback brakes maximally, as in Chapter 4, or, for comparison, at the
    global capability, and the governor searches for a restart candidate in
    every period; once one is admitted, the loop restarts with the fixed
    predictor. The alarms and the stops are those of Section 7.2.3 (checked
    against the script of Chapter 7).

Reproducibility
    Every planned and certified quantity is computed with exact fractions,
    as in Chapter 3. In the runs of Sections 8.1 and 8.2 and the time-budget
    runs, every label is valid, so the realized motion is exact as well; the
    script checks that no alarm is raised in them. In the two alarm runs,
    the motion of the period of the alarm, under an invalid label, is
    computed in floating point by the function of Chapter 3; the rest
    feedback then continues from the exact value of that floating-point
    state, and the point of rest is computed exactly (stop_after_alarm).
    Floating point is otherwise used only to print fractions, through the
    formatting function of Chapter 3, which prints six decimals; times are
    rounded half up in exact arithmetic. No random numbers are used. The results were identical
    in runs with CPython 3.8, 3.13 and 3.14 on one x86-64 Linux machine;
    other processors, operating systems and versions have not been tested.

Usage
    python3 cart_governor.py            # all runs; writes results/
    python3 cart_governor.py --check    # also checks every quoted number
    python3 cart_governor.py --period 179
                                        # prints the decision of one period
                                        # of the forecasting run

Outputs, in results/ next to this file
    trace-<configuration>.csv      one row per candidate and period: profile,
                                   reasons, admission, keys, outcome
    decisions-<configuration>.csv  one row per period: the committed
                                   candidate and the reason for the choice
    anticipation.csv               position and clearance of the five runs,
                                   per period (Figures 8.3 and 8.4)
    trace-budget-<run>.csv, decisions-budget-<run>.csv,
    trace-alarm-bias-<b>[-global].csv, decisions-alarm-bias-<b>[-global].csv
                                   the same for the runs of Section 8.3; the
                                   decisions also record the mode of the
                                   governor, the candidates evaluated before
                                   the deadline and the restart search
                                   ('-global': the rest feedback brakes at
                                   the global capability)
    summary.txt                    the numbers quoted in Chapter 8

Python 3.8 or later; no other packages.
"""
import argparse
import csv
import os
import sys
from fractions import Fraction as Fr

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ch03"))
import cart_example as ce    # noqa: E402

T = ce.T
NEG_INF = ce.NEG_INF
LEVEL = ce.DEMAND_LEVEL          # reserve that the operation requires: 1.3 m
ONSET = ce.period(ce.T_ONSET)    # period in which the operation is due: 200
ANNOUNCE = ce.period(ce.T_ANNOUNCE)   # period of the announcement: 140
POSTPONE = 21                    # preparation time of this generator [periods]
T_END = Fr(25)                   # end of the runs of Section 8.2 [s]


# ----------------------------------------------------------------------------
# Certification and profile, with the reason for every level
# ----------------------------------------------------------------------------
def pair_failure(seg, beta):
    """Why the test does not certify a segment under its label (condition (i)
    of Chapter 3), or None if it does. The checks are those of
    Segment.certified, in the same order."""
    if abs(seg.u) > ce.U_MAX:
        return "command beyond the bound"
    if seg.u < 0 and -seg.u > beta:
        return f"acceleration towards the start {ce.fmt(-seg.u)} beyond the label {ce.fmt(beta)}"
    if seg.v0 * seg.v1 < 0:
        return "motion reverses within the period"
    return None


def clearance_reason(running, where):
    text = f"clearance {ce.fmt(running)} m at {where}"
    return text if running >= ce.EPS_MIN else text + ", below the floor"


def profile_with_reasons(plan):
    """The profile of Plan.profile, and for every n the reason for the level
    certified from pair n on: the first pair from n on that the test does not
    certify, or the pair or rest point with the least clearance, which may be
    below the floor (no level of the grid is then certified)."""
    n_pairs = len(plan.pairs)
    prof = [NEG_INF] * (n_pairs + 1)
    why = [""] * (n_pairs + 1)
    running = ce.margin(plan.p_rest, plan.ell)
    where = f"the rest point {ce.fmt(plan.p_rest)} m"
    prof[n_pairs] = ce.floor_level(running)
    why[n_pairs] = clearance_reason(running, where)
    first_fail = None                  # smallest index from n on that fails
    for i in range(n_pairs - 1, -1, -1):
        seg, beta = plan.pairs[i]
        failure = pair_failure(seg, beta)
        if failure is not None:
            first_fail = (i, failure)
        c = ce.margin(seg.hi, plan.ell)
        if c < running:                # ties go to the rest point or a later pair
            running, where = c, f"pair {i}, up to {ce.fmt(seg.hi)} m"
        if first_fail is not None:
            prof[i] = NEG_INF
            why[i] = f"pair {first_fail[0]} not certified: {first_fail[1]}"
        else:
            prof[i] = ce.floor_level(running)
            why[i] = clearance_reason(running, where)
    assert prof == plan.profile(), "the profile must be that of Chapter 3"
    return prof, why


# ----------------------------------------------------------------------------
# Admission and selection, with the reason for every outcome
# ----------------------------------------------------------------------------
def select_with_reasons(entries, req, onset):
    """The selection rule of Chapter 4 (eq. 4.2), with the outcome of every
    entry. entries: (index, profile, ambition, reason for the level); index 0
    is the fallback. Returns the selected index, the outcome of each entry,
    the reason for the choice and the keys of the admitted entries."""
    admitted, keys, outcome = [], {}, {}
    for idx, prof, a, why0 in entries:
        lvl = prof[0]
        if lvl is NEG_INF or lvl < ce.EPS_MIN:
            if why0 == NOT_EVALUATED:
                outcome[idx] = "not admitted: " + NOT_EVALUATED
            elif "below the floor" in why0:
                outcome[idx] = "not admitted: below the floor"
            else:
                outcome[idx] = "not admitted: a pair not certified"
            continue
        key1 = min(ce.level_at(prof, onset), req)
        keys[idx] = (key1, a, 1 if idx >= 1 else 0, -idx)
        admitted.append(idx)
    best = max(admitted, key=lambda i: keys[i])
    k1, a1 = keys[best][0], keys[best][1]
    same_first = [i for i in admitted if keys[i][0] == k1]
    same_both = [i for i in same_first if keys[i][1] == a1]
    for idx in admitted:
        if idx == best:
            outcome[idx] = "committed"
        elif keys[idx][0] < k1:
            outcome[idx] = "admitted: lower first key"
        elif keys[idx][1] < a1:
            outcome[idx] = "admitted: same first key, lower ambition"
        elif idx == 0:
            outcome[idx] = "admitted: tie, a fresh candidate is preferred"
        else:
            outcome[idx] = "admitted: tie, a smaller index is preferred"
    if best == 0:
        if len(admitted) == 1:
            why = "fallback: the only admitted candidate"
        elif len(same_first) == 1:
            why = "fallback: the only candidate with the highest first key"
        else:
            why = "fallback: highest first key, and no fresh candidate as ambitious"
    elif req == ce.EPS_MIN:
        why = "no demand: the most ambitious admitted candidate"
    elif len(same_first) == 1:
        why = "demand: the only candidate with the highest first key"
    elif len(same_both) > 1:
        why = ("demand: highest first key and ambition, tie broken for a fresh"
               " candidate" if 0 in same_both else
               "demand: highest first key and ambition, tie broken for the smallest index")
    else:
        why = "demand: the most ambitious of the candidates with the highest first key"
    return best, outcome, why, keys


# ----------------------------------------------------------------------------
# How the operation is announced
# ----------------------------------------------------------------------------
class Schedule:
    """What the demand predictor issues about the operation, which is due in
    period 'onset'. From period 'announce' on it requires the reserve of the
    operation for the onset that it forecasts, 'predicted'; once that onset
    has passed, it requires the reserve from the first pair on. 'always'
    requires the reserve from the first pair on in every period. If the
    operation is due and the plan in force does not hold its reserve, an
    overload occurs, and the operation is postponed to the forecast onset if
    that is still ahead, and otherwise by 'postpone' periods, announced for
    the new onset; with postpone None it does not take place, as in
    Chapter 3."""

    def __init__(self, name, announce=ANNOUNCE, onset=ONSET, predicted=None,
                 always=False, postpone=POSTPONE):
        self.name, self.announce, self.onset = name, announce, onset
        self.predicted = onset if predicted is None else predicted
        self.always, self.postpone = always, postpone


SCHEDULES = {
    "forecast-informed": Schedule("forecast-informed"),
    "reactive-preparation": Schedule("reactive-preparation", announce=None),
    "fixed-conservatism": Schedule("fixed-conservatism", announce=None, always=True),
    "forecast-early": Schedule("forecast-early", predicted=ONSET - 10),
    "forecast-late": Schedule("forecast-late", predicted=ONSET + 10),
}
CHAPTER3 = dict(postpone=None)   # the operation of Chapter 3 is not postponed
CONFIGS = ("forecast-informed", "reactive-preparation", "fixed-conservatism",
           "forecast-early", "forecast-late")


def column(name):
    """The name of a configuration as used in the columns of anticipation.csv."""
    return name.replace("-", "_")


# ----------------------------------------------------------------------------
# One governed run
# ----------------------------------------------------------------------------
def run(kind, sched, t_end=T_END, trace=True):
    """The run of cart_example.run with capability predictor 'kind' and the
    operation announced as 'sched' states, with the common part of the
    governor written out and every decision recorded."""
    pred = ce.Predictor(kind)
    proposal = ce.ProposalMethod()
    k_end = ce.period(t_end)
    p, v = Fr(0), Fr(0)
    plan = ce.Plan(0, [], Fr(0))
    prof = plan.profile()
    ell_op, k_met, cancelled = Fr(0), None, False
    onset, announce, predicted = sched.onset, sched.announce, sched.predicted
    overloads = []
    announced = [] if announce is None else [(announce, predicted)]
    req_prev, onset_prev = ce.EPS_MIN, 0
    rows, entries_all, decisions = [], [], []
    for k in range(k_end):
        u_exec = plan.head_command()
        beta_exec = plan.pairs[0][1] if plan.pairs else ce.B_GLOBAL
        if plan.pairs:
            xhat = (plan.pairs[0][0].p1, plan.pairs[0][0].v1)
        else:
            xhat = (plan.p_rest, Fr(0))
        q = proposal(p)
        u_q = ce.proposed_command(q, xhat)
        # --- the demand predictor -------------------------------------------
        req, onset_idx = ce.EPS_MIN, 0
        if k_met is None and not cancelled:
            if k == onset:                       # the operation is due
                if prof[0] is not NEG_INF and prof[0] >= LEVEL:
                    k_met, ell_op = k, ce.ELL_UNFOLDED
                else:
                    overloads.append(k)
                    if sched.postpone is None:
                        cancelled = True
                    elif announce is not None and predicted > k:
                        onset = predicted        # postponed to the forecast onset
                    else:                        # postponed and announced
                        onset = predicted = k + sched.postpone
                        announce = k
                        announced.append((announce, predicted))
            if k_met is None and not cancelled:
                if sched.always:
                    req, onset_idx = LEVEL, 0
                elif announce is not None and k >= announce:
                    req, onset_idx = LEVEL, max(predicted - k - 1, 0)
        target = ce.WALL - req if req > ce.EPS_MIN else None
        # --- the governor ---------------------------------------------------
        fallback = plan.shift()
        fb_prof = prof[1:] if len(prof) > 1 else prof
        cands = ce.generate(pred, k, xhat, u_q, fallback, target, ell_op)
        evals = [profile_with_reasons(c[0]) for c in cands]
        profiles = [e[0] for e in evals]
        fb_eval = profile_with_reasons(fallback)
        assert fb_eval[0] == fb_prof, "the fallback needs no evaluation"
        sel, outcome, why, keys = select_with_reasons(
            [(0, fb_prof, Fr(0), fb_eval[1][0])]
            + [(i + 1, profiles[i], cands[i][1], evals[i][1][0])
               for i in range(len(cands))], req, onset_idx)
        assert sel == ce.select(cands, profiles, fb_prof, req, onset_idx), \
            "the selection must be that of Chapter 3"
        if sel == 0:
            new_plan, new_prof, alpha = fallback, fb_prof, Fr(0)
            assert k_met != k, "the operation must begin with a fresh plan"
        else:
            new_plan, new_prof = cands[sel - 1][0], profiles[sel - 1]
            alpha = cands[sel - 1][1]
        if trace:
            listed = [(0, fallback, Fr(0), fb_prof, fb_eval[1], "fallback")]
            listed += [(i + 1, cands[i][0], cands[i][1], profiles[i], evals[i][1], "fresh")
                       for i in range(len(cands))]
            for idx, pl, a, pr, wh, kind_c in listed:
                key = keys.get(idx)
                entries_all.append(dict(
                    k=k, t=k * T, index=idx, candidate=kind_c, ambition=a,
                    head=pl.head_command(), label=pl.pairs[0][1] if pl.pairs else ce.B_GLOBAL,
                    pairs=len(pl.pairs), rest=pl.p_rest, ell=pl.ell,
                    level=pr[0], onset_level=ce.level_at(pr, onset_idx),
                    level_reason=wh[0],
                    onset_reason=wh[min(onset_idx, len(wh) - 1)],
                    admitted=key is not None,
                    key1=None if key is None else key[0],
                    key2=None if key is None else key[1],
                    outcome=outcome[idx]))
            decisions.append(dict(
                k=k, t=k * T, p=p, v=v, p_init=xhat[0], v_init=xhat[1],
                mode=q, proposed=u_q, req=req, onset=onset_idx,
                candidates=len(cands) + 1, admitted=len(keys), selected=sel,
                ambition=alpha, level=new_prof[0],
                onset_level=ce.level_at(new_prof, onset_idx), reason=why))
        # --- execution: every label in these runs is valid -------------------
        p1, v1, ex = ce.realize(p, v, u_exec)
        assert ex and (p1, v1) == xhat, "no alarm: every executed label is valid"
        rows.append(dict(
            k=k, t=k * T, p=p, v=v, u=u_exec, beta=beta_exec,
            level=prof[0], rest=plan.p_rest,
            req_exec=req_prev, onset_level_exec=ce.level_at(prof, onset_prev),
            mode=q, q=u_q, alpha=alpha, sel=sel, eps=new_prof[0],
            reserve=new_plan.reserve(), plan=new_plan,
            eps_on=ce.level_at(new_prof, onset_idx), req=req, onset=onset_idx,
            p_rest=new_plan.p_rest, ell=plan.ell, alarm=0))
        pred.observe(min(p, p1), max(p, p1), p1)
        p, v = p1, v1
        plan, prof = new_plan, new_prof
        req_prev, onset_prev = req, onset_idx
    rows.append(dict(k=k_end, t=k_end * T, p=p, v=v, final=True))
    return dict(rows=rows, entries=entries_all, decisions=decisions,
                k_met=k_met, overloads=overloads, cancelled=cancelled,
                announced=announced)


def chapter3_consistency():
    """With the world, the operation and the predictors of Chapter 3, the
    loop of this script writes the same rows as cart_example.run."""
    keys = ["k", "t", "p", "v", "u", "beta", "mode", "q", "alpha", "sel", "eps",
            "eps_on", "req", "onset", "p_rest", "ell", "alarm", "level", "reserve"]
    same = True
    for kind in ("fixed", "reactive", "forecasting"):
        rows_ce, _, _ = ce.run(kind)
        out = run(kind, Schedule("chapter 3", **CHAPTER3), t_end=ce.T_END, trace=False)
        a = [{c: x.get(c) for c in keys} for x in rows_ce]
        b = [{c: x.get(c) for c in keys} for x in out["rows"]]
        same = same and a == b
    return same


# ----------------------------------------------------------------------------
# Section 8.3: deadlines, alarms and restart
# ----------------------------------------------------------------------------
NOT_EVALUATED = "not evaluated before the deadline"
T_END_83 = ce.T_END              # the runs of Section 8.3 end at 22 s, as in Chapter 3


class SurveyPredictor:
    """The survey predictor of Section 7.2.3, used without an offset, as a
    point estimate: braking of 2 m/s^2 on positions that end before the
    surveyed ramp start, 0.5 m/s^2 elsewhere, and the global capability
    beyond the horizon. A survey that places the ramp start too far issues
    invalid labels of 2 m/s^2 on the first metres of the ramp."""

    def __init__(self, surveyed):
        self.s_hat = surveyed

    def observe(self, p_lo, p_hi, p_now):
        pass

    def label(self, k, j, seg):
        if j > k + 1 + ce.HORIZON:
            return ce.B_GLOBAL
        return ce.B_RAMP if seg.hi >= self.s_hat else ce.B_FIRM


def stop_after_alarm(seg):
    """Where maximal braking brings the cart to rest after an alarm, exactly.
    The executed segment braked harder than the ground allows: on firm ground
    at its command, up to 2 m/s^2, and on the ramp at 0.5 m/s^2, the rate at
    which the rest feedback then continues. At a constant deceleration b,
    p + v^2/(2b) does not change, so the stop follows from the speed at
    which the cart reaches the ramp, without the instant of the crossing,
    which the floating-point motion of Chapter 3 needs."""
    assert seg.u < 0 and -seg.u > ce.B_RAMP and seg.v0 > 0
    if seg.p0 < ce.RAMP:
        v2 = seg.v0 * seg.v0 - 2 * min(-seg.u, ce.B_FIRM) * (ce.RAMP - seg.p0)
        return ce.RAMP + v2 / (2 * ce.B_RAMP)
    return seg.p0 + seg.v0 * seg.v0 / (2 * ce.B_RAMP)


def rest_feedback(p, v, stop):
    """One period of a rest feedback in the true world, from a state on the
    ramp, where the ground allows 0.5 m/s^2: maximal braking and braking at
    the global capability then both decelerate the cart at that rate until
    it is at rest at 'stop', and the brakes then hold it (Section 3.1). The
    state at the alarm is the floating-point state of Chapter 3; the point of
    rest is exact."""
    assert p >= ce.RAMP and v >= 0, "every alarm of these runs occurs on the ramp"
    if v == 0:
        return p, v
    b = ce.B_RAMP
    if v <= b * T:
        assert abs(float(p + v * v / (2 * b) - stop)) < 1e-9
        return stop, Fr(0)
    return p + v * T - b * T * T / 2, v - b * T


REST_FEEDBACKS = {
    "maximal": "maximal braking, the rest feedback of the cart (Section 4.2.3)",
    "global": "braking at the global capability, 0.5 m/s^2",
}


def governed_run(pred, sched, t_end=T_END_83, budget=None, rest="maximal"):
    """Algorithm 8.1 for the cart: the loop of run(), with a time budget and
    the response to a label alarm.

    budget(k) is the number of fresh candidates that the governor can evaluate
    before the deadline of period k, in the order in which the generator lists
    them (None: all); the others count as not certified. The fallback needs no
    evaluation.

    After an alarm, the control method applies its rest feedback ('rest',
    see REST_FEEDBACKS), and the governor searches for a restart candidate in
    every period, from the latest measurement. A restart candidate is
    labelled by the global capability and must be certified on a set that
    contains every state that the rest feedback can reach at the next period
    boundary in the worlds consistent with the global capability. Under
    maximal braking, while the cart moves towards the wall, those states
    differ from world to world (braking of 0.5 m/s^2 and of 2 m/s^2 are both
    consistent with it), and no single-trajectory funnel of this control
    method contains two states; at rest, the brakes hold the cart, and the set
    is the measured state. Braking at exactly 0.5 m/s^2 is realized in every
    world consistent with the global capability, so its set is a single state
    while the cart moves as well. The restart candidates are those of the
    generator with the fixed predictor, from that state; their blends use the
    command of the rest feedback in place of the head of a fallback. An
    admitted restart candidate is committed at the next period boundary, and
    the loop restarts with the fixed predictor. Without a plan in force, an
    operation that falls due is an overload."""
    assert rest in REST_FEEDBACKS
    proposal = ce.ProposalMethod()
    fixed = ce.Predictor("fixed")
    labels = pred
    k_end = ce.period(t_end)
    p, v = Fr(0), Fr(0)
    plan = ce.Plan(0, [], Fr(0))
    prof = plan.profile()
    mode = "normal"
    ell_op, k_met, cancelled = Fr(0), None, False
    onset, announce, predicted = sched.onset, sched.announce, sched.predicted
    overloads, alarms, restarts = [], [], []
    req_prev, onset_prev = ce.EPS_MIN, 0
    rows, entries_all, decisions = [], [], []
    for k in range(k_end):
        q = proposal(p)
        # --- the demand predictor (as in run) --------------------------------
        req, onset_idx = ce.EPS_MIN, 0
        if k_met is None and not cancelled:
            if k == onset:                       # the operation is due
                in_force = prof[0] if mode == "normal" else NEG_INF
                if in_force is not NEG_INF and in_force >= LEVEL:
                    k_met, ell_op = k, ce.ELL_UNFOLDED
                else:
                    overloads.append(k)
                    if sched.postpone is None:
                        cancelled = True
                    elif announce is not None and predicted > k:
                        onset = predicted
                    else:
                        onset = predicted = k + sched.postpone
                        announce = k
            if k_met is None and not cancelled:
                if sched.always:
                    req, onset_idx = LEVEL, 0
                elif announce is not None and k >= announce:
                    req, onset_idx = LEVEL, max(predicted - k - 1, 0)
        target = ce.WALL - req if req > ce.EPS_MIN else None
        if mode == "normal":
            # --- the governor, with a deadline --------------------------------
            u_exec = plan.head_command()
            beta_exec = plan.pairs[0][1] if plan.pairs else ce.B_GLOBAL
            if plan.pairs:
                seg_exec = plan.pairs[0][0]
                xhat = (seg_exec.p1, seg_exec.v1)
            else:
                seg_exec, xhat = None, (plan.p_rest, Fr(0))
            u_q = ce.proposed_command(q, xhat)
            fallback = plan.shift()
            fb_prof = prof[1:] if len(prof) > 1 else prof
            cands = ce.generate(labels, k, xhat, u_q, fallback, target, ell_op)
            n_eval = len(cands) if budget is None or budget(k) is None \
                else min(budget(k), len(cands))
            evals = []
            for i, c in enumerate(cands):
                if i < n_eval:
                    evals.append(profile_with_reasons(c[0]))
                else:
                    n = len(c[0].pairs) + 1
                    evals.append(([NEG_INF] * n, [NOT_EVALUATED] * n))
            profiles = [e[0] for e in evals]
            fb_eval = profile_with_reasons(fallback)
            assert fb_eval[0] == fb_prof, "the fallback needs no evaluation"
            sel, outcome, why, keys = select_with_reasons(
                [(0, fb_prof, Fr(0), fb_eval[1][0])]
                + [(i + 1, profiles[i], cands[i][1], evals[i][1][0])
                   for i in range(len(cands))], req, onset_idx)
            assert sel == ce.select(cands, profiles, fb_prof, req, onset_idx)
            if sel == 0:
                # without a fresh candidate, the tool stays folded: the
                # operation begins with the first committed plan that unfolds it
                new_plan, new_prof, alpha = fallback, fb_prof, Fr(0)
            else:
                new_plan, new_prof = cands[sel - 1][0], profiles[sel - 1]
                alpha = cands[sel - 1][1]
            listed = [(0, fallback, Fr(0), fb_prof, fb_eval[1], "fallback")]
            listed += [(i + 1, cands[i][0], cands[i][1], profiles[i], evals[i][1], "fresh")
                       for i in range(len(cands))]
            record_entries(entries_all, k, listed, keys, outcome, onset_idx)
            decisions.append(dict(
                k=k, t=k * T, p=p, v=v, governor="normal", mode=q, proposed=u_q,
                req=req, onset=onset_idx, candidates=len(cands) + 1, evaluated=n_eval,
                admitted=len(keys), selected=sel, ambition=alpha, level=new_prof[0],
                onset_level=ce.level_at(new_prof, onset_idx), reason=why))
            # --- execution, measurement and alarm check -----------------------
            p1, v1, ex = ce.realize(p, v, u_exec)
            if not ex and seg_exec is not None and seg_exec.hi == ce.RAMP:
                # touching the ramp only at its end, the overstated braking
                # never acts (Section 7.3): the motion is the planned one
                p1, v1, ex = seg_exec.p1, seg_exec.v1, True
            alarm = not (ex and (p1, v1) == xhat)
            rows.append(dict(
                k=k, t=k * T, p=p, v=v, u=u_exec, beta=beta_exec,
                level=prof[0], rest=plan.p_rest,
                req_exec=req_prev, onset_level_exec=ce.level_at(prof, onset_prev),
                mode=q, q=u_q, alpha=alpha, sel=sel, eps=new_prof[0],
                reserve=new_plan.reserve(), plan=new_plan,
                eps_on=ce.level_at(new_prof, onset_idx), req=req, onset=onset_idx,
                p_rest=new_plan.p_rest, ell=plan.ell, alarm=int(alarm),
                governor="normal"))
            if alarm:
                stop = stop_after_alarm(seg_exec)
                alarms.append(dict(k=k + 1, measured=(p1, v1), planned=xhat, stop=stop))
                p1, v1 = Fr(p1), Fr(v1)          # the exact values of the floats
                mode = "rest feedback"
                ell_alarm = plan.ell             # the rest feedback keeps the mode
                plan, prof = None, None
            else:
                plan, prof = new_plan, new_prof
        else:
            # --- after an alarm: rest feedback and the search for a restart ---
            u_q = ce.proposed_command(q, (p, v))
            committed = None
            if v == 0:                           # the brakes hold the cart in every world
                reach = (p, Fr(0))
            elif rest == "global":               # 0.5 m/s^2 is realized in every such world
                reach = rest_feedback(p, v, alarms[-1]["stop"])
            else:
                reach = None
            if reach is None:
                why = ("no restart candidate: the cart moves towards the wall under "
                       "maximal braking, so the states that the rest feedback can reach "
                       "differ between worlds")
                n_c = n_adm = 0
            else:
                xhat = reach
                if xhat[1] == 0:
                    rf = ce.Plan(k + 1, [], xhat[0])
                else:                            # supplies the command of the rest feedback
                    rf = ce.Plan(k + 1, [(ce.Segment(xhat[0], xhat[1],
                                                     -min(ce.B_GLOBAL, xhat[1] / T)),
                                          ce.B_GLOBAL)], xhat[0])
                cands = ce.generate(fixed, k, xhat, u_q, rf, target, ell_op)
                evals = [profile_with_reasons(c[0]) for c in cands]
                listed, keys, outcome = [], {}, {}
                for i, (c, e) in enumerate(zip(cands, evals)):
                    lvl = e[0][0]
                    if lvl is not NEG_INF and lvl >= ce.EPS_MIN:
                        keys[i + 1] = (min(ce.level_at(e[0], onset_idx), req), c[1], -(i + 1))
                    listed.append((i + 1, c[0], c[1], e[0], e[1], "restart"))
                n_c, n_adm = len(cands), len(keys)
                if keys:
                    best = max(keys, key=lambda i: keys[i])
                    for i in keys:
                        outcome[i] = "committed" if i == best else "admitted, not selected"
                    committed = (cands[best - 1][0], evals[best - 1][0], cands[best - 1][1])
                    why = "restart: an admitted restart candidate, labelled by the global capability"
                else:
                    why = ("no restart candidate admitted: " + evals[-1][1][0])
                for i, *_ in listed:
                    outcome.setdefault(i, "not admitted")
                record_entries(entries_all, k, listed, {i: (keys[i][0], keys[i][1])
                                                        for i in keys}, outcome, onset_idx)
            decisions.append(dict(
                k=k, t=k * T, p=p, v=v, governor="rest feedback", mode=q, proposed=u_q,
                req=req, onset=onset_idx, candidates=n_c, evaluated=n_c, admitted=n_adm,
                selected=None if committed is None else "restart",
                ambition=Fr(0) if committed is None else committed[2],
                level=None if committed is None else committed[1][0],
                onset_level=None if committed is None
                else ce.level_at(committed[1], onset_idx), reason=why))
            p1, v1 = rest_feedback(p, v, alarms[-1]["stop"])
            rows.append(dict(k=k, t=k * T, p=p, v=v, u=None, beta=None, level=None,
                             rest=None, mode=q, q=u_q,
                             alpha=Fr(0) if committed is None else committed[2],
                             sel=None, eps=None,
                             req=req, onset=onset_idx, ell=ell_alarm, alarm=0,
                             governor="rest feedback"))
            if committed is not None:
                plan, prof = committed[0], committed[1]
                labels, mode = fixed, "normal"
                restarts.append(dict(k=k + 1, plan=plan, profile=prof))
        labels.observe(min(p, p1), max(p, p1), p1)
        p, v = p1, v1
        req_prev, onset_prev = req, onset_idx
    rows.append(dict(k=k_end, t=k_end * T, p=p, v=v, final=True))
    return dict(rows=rows, entries=entries_all, decisions=decisions, k_met=k_met,
                overloads=overloads, cancelled=cancelled, alarms=alarms,
                restarts=restarts)


def record_entries(entries_all, k, listed, keys, outcome, onset_idx):
    """Adds the trace rows of the candidates of one period."""
    for idx, pl, a, pr, wh, kind_c in listed:
        key = keys.get(idx)
        entries_all.append(dict(
            k=k, t=k * T, index=idx, candidate=kind_c, ambition=a,
            head=pl.head_command(), label=pl.pairs[0][1] if pl.pairs else ce.B_GLOBAL,
            pairs=len(pl.pairs), rest=pl.p_rest, ell=pl.ell,
            level=pr[0], onset_level=ce.level_at(pr, onset_idx),
            level_reason=wh[0], onset_reason=wh[min(onset_idx, len(wh) - 1)],
            admitted=key is not None,
            key1=None if key is None else key[0],
            key2=None if key is None else key[1],
            outcome=outcome[idx]))


STALL = (10, 30)                  # periods without any evaluation in the stall run
RUNTIME = {                       # time budgets: fresh candidates evaluated per period
    "full": None,
    "one-candidate": lambda k: 1,
    "stall": lambda k: 0 if STALL[0] <= k < STALL[1] else None,
    "no-evaluation": lambda k: 0,
}
BIASES_83 = (Fr(1, 5), Fr(7, 20))   # survey biases of Section 7.2.3 [m]


def runtime_consistency():
    """With every candidate evaluated and valid labels, governed_run writes
    the rows of run(), which are those of Chapter 3."""
    keys = ["k", "t", "p", "v", "u", "beta", "mode", "q", "alpha", "sel", "eps",
            "eps_on", "req", "onset", "p_rest", "ell", "alarm", "level", "reserve"]
    a = run("forecasting", Schedule("chapter 3", **CHAPTER3), t_end=ce.T_END,
            trace=False)["rows"]
    b = governed_run(ce.Predictor("forecasting"), Schedule("chapter 3", **CHAPTER3))["rows"]
    return [{c: x.get(c) for c in keys} for x in a] == [{c: x.get(c) for c in keys}
                                                         for x in b]


def chapter7_consistency(outs):
    """The alarms and the stops of the two biased surveys are those of
    Section 7.2.3, computed by the script of Chapter 7 (its run ends at the
    alarm, and it integrates the rest feedback numerically)."""
    sys.path.insert(0, os.path.join(HERE, "..", "ch07"))
    import cart_forecasts as cf    # noqa: E402
    same = True
    for bias in BIASES_83:
        s7 = cf.bias_run(bias)
        out = outs[bias]
        stop = next(x for x in out["rows"]
                    if x.get("governor") == "rest feedback" and x["v"] == 0)["p"]
        same = (same and s7["alarm"] == out["alarms"][0]["k"]
                and f"{s7['stop']:.2f}" == r(stop, 2))
    return same


def rest_after_alarm(out):
    """Where the cart first comes to rest after the alarm, whatever governs it."""
    k_a = out["alarms"][0]["k"]
    return next(x["p"] for x in out["rows"] if x["k"] >= k_a and x["v"] == 0)


def runtime_facts(out):
    """What a run of Section 8.3 shows. Times are period boundaries."""
    body = [x for x in out["rows"] if not x.get("final")]
    final = out["rows"][-1]
    normal = [x for x in body if x.get("governor") == "normal"]
    arrival = next((x["k"] for x in body if x["p"] >= ce.WORK_ZONE), None)
    k_met = out["k_met"]
    ends = [x["p"] for x in out["rows"][1:]]
    clear = [ce.WALL - (max(x["p"], e) + x["ell"]) for x, e in zip(body, ends)]
    dec = out["decisions"]
    n_fresh = sum(x["candidates"] - 1 for x in dec if x["governor"] == "normal")
    n_eval = sum(x["evaluated"] for x in dec if x["governor"] == "normal")
    moved = next((x["k"] for x in body if x["v"] > 0), None)
    rest = None if moved is None else next(
        (x for x in body if x["k"] > moved and x["v"] == 0), None)
    back = next((x["k"] for x in body if x["k"] >= ANNOUNCE and x.get("u") is not None
                 and x["u"] < 0 and x["v"] <= 0), None)
    f = dict(
        arrival=(r(arrival * T, 1) + " s") if arrival is not None else "never",
        first_rest=("never moves" if moved is None else "never" if rest is None else
                    f"{r(rest['p'], 2)} m from {r(rest['k'] * T, 1)} s"),
        backaway_from=(r(back * T, 1) + " s") if back is not None else "-",
        position_at_onset=f"{r(body[ONSET]['p'], 2)} m",
        operation=operation_begins(out, normal),
        fresh_candidates_evaluated=f"{n_eval} of {n_fresh}",
        fallback_periods=str(sum(1 for x in normal if x["sel"] == 0)),
        lowest_level=lv(min(x["eps"] for x in normal
                            if not out["alarms"] or x["k"] < out["alarms"][0]["k"])),
        least_clearance=r(min(clear), 2) + " m",
        violations=str(sum(1 for c in clear if c < 0)),
        final_state=f"{r(final['p'], 2)} m, {r(final['v'], 2)} m/s")
    if out["alarms"]:
        a = out["alarms"][0]
        pa, va = a["measured"]
        f["alarm"] = (f"period {a['k']}, {r(a['k'] * T, 1)} s; measured "
                      f"{r(Fr(pa), 3)} m, {r(Fr(va), 3)} m/s; initial set "
                      f"{r(a['planned'][0], 3)} m, {r(a['planned'][1], 3)} m/s")
        rest = next((x for x in body if x.get("governor") == "rest feedback"
                     and x["v"] == 0), None)
        f["unchanged_proposals_before_alarm"] = str(sum(
            1 for x in normal if x["k"] < a["k"] and x["alpha"] == 1))
        f["at_rest"] = ("never" if rest is None else
                        f"{r(rest['p'], 2)} m from {r(rest['k'] * T, 1)} s")
        f["stop_exact"] = ce.fmt(a["stop"])
        moving = sum(1 for x in dec if x["governor"] == "rest feedback" and x["v"] != 0
                     and x["selected"] is None)
        f["periods_without_restart_while_moving"] = str(moving)
        if out["restarts"]:
            rs = out["restarts"][0]
            f["restart"] = (f"committed at {r(rs['k'] * T, 1)} s (period {rs['k']}), "
                            f"level {lv(rs['profile'][0])}")
            after = [x for x in normal if x["k"] >= rs["k"]]
            f["lowest_level_after_restart"] = lv(min(x["eps"] for x in after))
            f["labels_after_restart"] = ", ".join(sorted({ce.fmt(x["beta"]) for x in after}))
        else:
            reasons = sorted({x["reason"] for x in dec if x["governor"] == "rest feedback"
                              and x["v"] == 0})
            f["restart"] = "never; " + "; ".join(reasons)
    return f


def budget_facts(out):
    """How often the proposal was committed unchanged, and in which periods
    of the approach, from the first modified proposal to the first rest."""
    body = [x for x in out["rows"] if not x.get("final")]
    moved = next(x["k"] for x in body if x["v"] > 0)
    rest = next(x for x in body if x["k"] > moved and x["v"] == 0)
    first_mod = next(x["k"] for x in body if x["alpha"] < 1)
    return dict(
        proposals_committed_unchanged=str(sum(1 for x in body if x["alpha"] == 1)),
        approach_fresh_commitments="periods " + ", ".join(
            str(x["k"]) for x in body if first_mod <= x["k"] < rest["k"] and x["sel"] != 0))


def operation_begins(out, normal):
    """When the tool unfolds: with the first committed plan in operation mode."""
    tool = next((x["k"] for x in normal if x["plan"].ell == ce.ELL_UNFOLDED), None)
    if tool is not None:
        return r((tool + 1) * T, 1) + " s"
    if out["overloads"]:
        return "overload in period " + ", ".join(str(k) for k in out["overloads"])
    if out["k_met"] is not None:
        return "never: the reserve is held when due, but no candidate that unfolds the tool is evaluated"
    return "never"


def stall_facts(out):
    """What the cart does while no candidate can be evaluated."""
    body = [x for x in out["rows"] if not x.get("final")]
    window = [x for x in body if STALL[0] <= x["k"] < STALL[1]]
    rest = next((x for x in window if x["v"] == 0 and x["k"] > STALL[0]), None)
    resumed = next(x for x in body if x["k"] == STALL[1])
    return dict(
        stall=f"periods {STALL[0]} to {STALL[1] - 1}, {r(STALL[0] * T, 1)} to "
              f"{r(STALL[1] * T, 1)} s",
        state_when_stall_begins=f"{r(window[0]['p'], 3)} m, {r(window[0]['v'], 2)} m/s",
        plan_followed=f"committed in period {STALL[0] - 1}, {len(body[STALL[0] - 1]['plan'].pairs)}"
                      f" pairs, rest point {ce.fmt(body[STALL[0] - 1]['p_rest'])} m",
        at_rest=("never" if rest is None else
                 f"{r(rest['p'], 3)} m from {r(rest['k'] * T, 1)} s"),
        fallbacks_in_stall=str(sum(1 for x in window if x["sel"] == 0)),
        state_when_evaluation_resumes=f"{r(resumed['p'], 3)} m, {r(resumed['v'], 2)} m/s")


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def r(x, nd):
    """x rounded half up to nd >= 1 decimals, as a string, in exact arithmetic."""
    n = int((Fr(x) * 10 ** nd + Fr(1, 2)) // 1)
    digits = str(abs(n)).rjust(nd + 1, "0")
    return ("-" if n < 0 else "") + digits[:-nd] + "." + digits[-nd:]


def yes(b):
    return "yes" if b else "no"


def lv(x):
    return "-inf" if x is NEG_INF else ce.fmt(x)


def run_facts(out):
    """What a run of Section 8.2 shows. Times are period boundaries."""
    body = [x for x in out["rows"] if not x.get("final")]
    in_zone = [x for x in body if x["p"] >= ce.WORK_ZONE]
    arrival = next((x["k"] for x in body if x["p"] >= ce.WORK_ZONE), None)
    k_met = out["k_met"]
    back = next((x["k"] for x in body if x["k"] >= 100 and x["u"] < 0 and x["v"] <= 0), None)
    clear = [ce.WALL - (x["p"] + x["ell"]) for x in body]
    moved = next(x["k"] for x in body if x["v"] > 0)
    ready = next((x["k"] for x in body if x["k"] > moved and x["v"] == 0
                  and x["p"] == ce.WALL - LEVEL), None)
    before = [x["eps"] for x in body if k_met is None or x["k"] < k_met]
    left = next((x["k"] for x in body if arrival is not None and x["k"] > arrival
                 and x["p"] < ce.WORK_ZONE), None)
    early_rows = [x for x in body if x["k"] < ONSET]
    return dict(
        announced=("; ".join(f"in period {a} for period {o}" for a, o in out["announced"])
                   or "never"),
        arrival=(r(arrival * T, 1) + " s") if arrival is not None else "never",
        time_in_work_zone=r(len(in_zone) * T, 1) + " s",
        leaves_work_zone=(r(left * T, 1) + " s") if left is not None else "-",
        backaway_from=(r(back * T, 1) + " s") if back is not None else "-",
        at_rest_at_23_7_from=(r(ready * T, 1) + " s") if ready is not None else "never",
        level_in_force_when_due=lv(body[ONSET]["level"]),
        reserve_in_force_from="period " + str(next(
            x["k"] for x in body if x["k"] >= ONSET and x["level"] is not NEG_INF
            and x["level"] >= LEVEL)),
        overloads=(", ".join(f"period {k}" for k in out["overloads"]) or "none"),
        operation_begins=(r((k_met + 1) * T, 1) + " s") if k_met is not None else "never",
        operation_delay=(r((k_met - ONSET) * T, 1) + " s") if k_met is not None else "-",
        lowest_level_before_operation=lv(min(before)),
        lowest_level=lv(min(x["eps"] for x in body)),
        least_clearance=ce.fmt(min(clear)) + " m",
        violations=str(sum(1 for c in clear if c < 0)),
        fallback_periods=str(sum(1 for x in body if x["sel"] == 0)),
        proposals_modified_before_onset=str(sum(1 for x in early_rows if x["alpha"] < 1)),
        last_unchanged_before_onset="period " + str(max(
            x["k"] for x in early_rows if x["alpha"] == 1)))


def backaway_window(out):
    """The periods from the start of the back-away to the first rest at
    23.7 m, as (start, end), or None."""
    body = [x for x in out["rows"] if not x.get("final")]
    back = next((x["k"] for x in body if x["k"] >= 100 and x["u"] < 0 and x["v"] <= 0), None)
    if back is None:
        return None
    ready = next(x["k"] for x in body if x["k"] > back and x["v"] == 0
                 and x["p"] == ce.WALL - LEVEL)
    return back, ready


def same_backaway(ref, out):
    """Whether the back-away of 'out' is that of 'ref', shifted in time."""
    w_ref, w = backaway_window(ref), backaway_window(out)
    if w_ref is None or w is None:
        return "-"
    shift = w[0] - w_ref[0]
    same = all(out["rows"][k + shift]["p"] == ref["rows"][k]["p"]
               and out["rows"][k + shift]["v"] == ref["rows"][k]["v"]
               for k in range(w_ref[0], w_ref[1] + 1))
    return f"{yes(same)}, shifted by {shift} periods"


def postponement_facts():
    """Why the postponement is 21 periods: with 20, the cart at rest at
    24.5 m cannot meet the postponed onset either."""
    facts = {}
    for n in (POSTPONE - 1, POSTPONE):
        out = run("forecasting", Schedule(f"reactive-preparation, {n}", announce=None,
                                          postpone=n), trace=False)
        facts[f"reactive_preparation_postponed_by_{n}"] = (
            "overloads in " + ", ".join(f"period {k}" for k in out["overloads"])
            + "; operation begins " + (r((out["k_met"] + 1) * T, 1) + " s"
                                      if out["k_met"] is not None else "never"))
    return facts


def decision_facts(out, periods):
    """The decisions of selected periods of the forecasting run, and the
    reasons over the whole run."""
    facts = {}
    for k in periods:
        d = next(x for x in out["decisions"] if x["k"] == k)
        es = [e for e in out["entries"] if e["k"] == k]
        facts[f"period {k}"] = (
            f"p {ce.fmt(d['p'])} m, v {ce.fmt(d['v'])} m/s; initial set "
            f"({ce.fmt(d['p_init'])} m, {ce.fmt(d['v_init'])} m/s); {d['mode']}, proposed "
            f"{ce.fmt(d['proposed'])}; requirement {ce.fmt(d['req'])} m from pair "
            f"{d['onset']}; {d['candidates']} candidates, {d['admitted']} admitted; "
            f"committed {d['selected']} (ambition {ce.fmt(d['ambition'])}): {d['reason']}")
        for e in es:
            facts[f"period {k} candidate {e['index']}"] = (
                f"ambition {ce.fmt(e['ambition'])}; head {ce.fmt(e['head'])} under "
                f"{ce.fmt(e['label'])}; {e['pairs']} pairs; rest {ce.fmt(e['rest'])} m; "
                f"level {lv(e['level'])} ({e['level_reason']}); onset level "
                f"{lv(e['onset_level'])} ({e['onset_reason']}); {e['outcome']}")
    count = {}
    for e in out["entries"]:
        count[e["outcome"]] = count.get(e["outcome"], 0) + 1
    for key in sorted(count):
        facts[f"outcomes: {key}"] = str(count[key])
    unc = [e for e in out["entries"] if e["outcome"] == "not admitted: a pair not certified"]
    ks = sorted({e["k"] for e in unc})
    pairs = sorted({e["level_reason"].split(" ")[1] for e in unc})
    facts["not_certified: periods"] = (
        f"{ks[0]}-{ks[-1]}" + (", every period" if ks == list(range(ks[0], ks[-1] + 1))
                               else ", with gaps"))
    facts["not_certified: failing pairs"] = ", ".join("pair " + x for x in pairs)
    fail = {}
    for e in out["entries"]:
        if e["level"] is NEG_INF:
            if "below the floor" in e["level_reason"]:
                kind_f = "clearance below the floor"
            else:
                kind_f = "pair not certified: " + e["level_reason"].split(": ", 1)[1].split(" ")[0]
            fail[kind_f] = fail.get(kind_f, 0) + 1
    for key in sorted(fail):
        facts[f"level -inf: {key}"] = str(fail[key])
    why = {}
    for d in out["decisions"]:
        why[d["reason"]] = why.get(d["reason"], 0) + 1
    for key in sorted(why):
        facts[f"decisions: {key}"] = str(why[key])
    spans, start = [], None
    window = [d for d in out["decisions"] if ANNOUNCE <= d["k"] < ONSET]
    for i, d in enumerate(window):
        if start is None:
            start = d
        if i + 1 == len(window) or window[i + 1]["selected"] != d["selected"]:
            span = (f"{start['k']}" if start["k"] == d["k"] else f"{start['k']}-{d['k']}")
            spans.append(f"{span}: {d['selected']}")
            start = None
    facts["committed_from_announcement_to_onset"] = "; ".join(spans)
    held = [d["onset_level"] for d in out["decisions"] if ANNOUNCE <= d["k"] < ONSET]
    facts["committed_onset_level_from_announcement_to_onset"] = (
        ", ".join(sorted({lv(x) for x in held})) + f" m in all {len(held)} periods")
    facts["periods"] = str(len(out["decisions"]))
    facts["candidates"] = str(len(out["entries"]))
    facts["fallback_profile_equals_its_evaluation"] = (
        f"in all {len(out['decisions'])} periods")
    return facts


def print_period(out, k):
    d = next(x for x in out["decisions"] if x["k"] == k)
    print(f"period {k}: p = {ce.fmt(d['p'])} m, v = {ce.fmt(d['v'])} m/s, "
          f"proposal {d['mode']} ({ce.fmt(d['proposed'])} m/s^2), requirement "
          f"{ce.fmt(d['req'])} m from pair {d['onset']}")
    for e in out["entries"]:
        if e["k"] == k:
            print(f"  {e['index']} {e['candidate']:8s} a={ce.fmt(e['ambition']):5s} "
                  f"head={ce.fmt(e['head']):7s} label={ce.fmt(e['label']):4s} "
                  f"rest={ce.fmt(e['rest']):10s} level={lv(e['level']):6s} "
                  f"onset={lv(e['onset_level']):6s} {e['outcome']}")
            print(f"      level: {e['level_reason']}; onset: {e['onset_reason']}")
    print(f"  -> {d['selected']}: {d['reason']}")


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
                elif x is NEG_INF:
                    out.append("-inf")
                else:
                    out.append(ce.fmt(x))
            w.writerow(out)


TRACE_COLS = ["k", "t", "index", "candidate", "ambition", "head", "label", "pairs",
              "rest", "ell", "level", "onset_level", "level_reason", "onset_reason",
              "admitted", "key1", "key2", "outcome"]
DECISION_COLS = ["k", "t", "p", "v", "p_init", "v_init", "mode", "proposed", "req",
                 "onset", "candidates", "admitted", "selected", "ambition", "level",
                 "onset_level", "reason"]
RUNTIME_COLS = ["k", "t", "p", "v", "governor", "mode", "proposed", "req", "onset",
                "candidates", "evaluated", "admitted", "selected", "ambition", "level",
                "onset_level", "reason"]
WORKED = (86, 179, 185)          # the periods of Section 8.1


def main():
    ap = argparse.ArgumentParser(description="Chapter 8: the governor, step by step")
    ap.add_argument("--check", action="store_true", help="check every number quoted in Chapter 8")
    ap.add_argument("--period", type=int, help="print the decision of one period")
    args = ap.parse_args()
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)

    chapter3 = run("forecasting", Schedule("chapter 3", **CHAPTER3), t_end=ce.T_END)
    if args.period is not None:
        print_period(chapter3, args.period)
        return
    runs = {name: run("forecasting", SCHEDULES[name]) for name in CONFIGS}

    facts = {"consistency": {"same_as_chapter_3": yes(chapter3_consistency())}}
    facts["decisions of the forecasting run"] = decision_facts(chapter3, WORKED)
    write_csv(os.path.join(out_dir, "trace-chapter3.csv"), TRACE_COLS, chapter3["entries"])
    write_csv(os.path.join(out_dir, "decisions-chapter3.csv"), DECISION_COLS,
              chapter3["decisions"])
    facts["postponement"] = postponement_facts()
    for name in CONFIGS:
        facts[f"configuration: {name}"] = run_facts(runs[name])
        facts[f"configuration: {name}"]["backaway_as_in_forecast_run"] = same_backaway(
            runs["forecast-informed"], runs[name])
        write_csv(os.path.join(out_dir, f"trace-{name}.csv"), TRACE_COLS, runs[name]["entries"])
        write_csv(os.path.join(out_dir, f"decisions-{name}.csv"), DECISION_COLS,
                  runs[name]["decisions"])
    series = []
    for i, x in enumerate(r_ for r_ in runs["forecast-informed"]["rows"]):
        row = {"t": x["t"]}
        for name in CONFIGS:
            y = runs[name]["rows"][i]
            row[f"p_{column(name)}"] = y["p"]
            ell = y.get("ell", runs[name]["rows"][i - 1]["ell"] if i else Fr(0))
            row[f"clearance_{column(name)}"] = ce.WALL - (y["p"] + ell)
        series.append(row)
    write_csv(os.path.join(out_dir, "anticipation.csv"),
              ["t"] + [f"{c}_{column(n)}" for n in CONFIGS for c in ("p", "clearance")],
              series)

    # Section 8.3: time budgets, alarms and restart
    facts["runtime consistency"] = {"same_as_chapter_3_with_full_evaluation":
                                    yes(runtime_consistency())}
    for name, budget in RUNTIME.items():
        out = governed_run(ce.Predictor("forecasting"), Schedule("chapter 3", **CHAPTER3),
                           budget=budget)
        assert not out["alarms"], "no alarm in the time-budget runs: every label is valid"
        f = runtime_facts(out)
        if name == "one-candidate":
            f.update(budget_facts(out))
        if name == "stall":
            f.update(stall_facts(out))
        facts[f"time budget: {name}"] = f
        if name != "full":            # the full run is the run of Chapter 3
            write_csv(os.path.join(out_dir, f"trace-budget-{name}.csv"), TRACE_COLS,
                      out["entries"])
            write_csv(os.path.join(out_dir, f"decisions-budget-{name}.csv"), RUNTIME_COLS,
                      out["decisions"])
    alarm_runs = {}
    for rest in REST_FEEDBACKS:
        for bias in BIASES_83:
            out = governed_run(SurveyPredictor(ce.RAMP + bias),
                               Schedule("chapter 3", **CHAPTER3), rest=rest)
            name = f"survey bias {ce.fmt(bias)} m" + (
                "" if rest == "maximal" else ", rest feedback at 0.5 m/s^2")
            suffix = "" if rest == "maximal" else "-global"
            if rest == "maximal":
                alarm_runs[bias] = out
            facts[f"alarm: {name}"] = runtime_facts(out)
            write_csv(os.path.join(out_dir, f"trace-alarm-bias-{ce.fmt(bias)}{suffix}.csv"),
                      TRACE_COLS, out["entries"])
            write_csv(os.path.join(out_dir,
                                   f"decisions-alarm-bias-{ce.fmt(bias)}{suffix}.csv"),
                      RUNTIME_COLS, out["decisions"])
    facts["runtime consistency"]["alarms_and_stops_as_in_section_7_2_3"] = yes(
        chapter7_consistency(alarm_runs))
    global_02 = governed_run(SurveyPredictor(ce.RAMP + BIASES_83[0]),
                             Schedule("chapter 3", **CHAPTER3), rest="global")
    p_max, p_glob = rest_after_alarm(alarm_runs[BIASES_83[0]]), rest_after_alarm(global_02)
    facts["alarm: survey bias 0.2 m, rest feedback at 0.5 m/s^2"]["at_rest_after_alarm"] = (
        f"{ce.fmt(p_glob)} m, {r((p_glob - p_max) * 1000, 2)} mm beyond the rest under "
        f"maximal braking, {ce.fmt(p_max)} m")

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
        lines.append(f"[check] {n} values quoted in Chapter 8: " +
                     ("all as quoted" if not failures else "FAILED: " + "; ".join(failures)))
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if failures:
        raise SystemExit(1)


EXPECTED = {   # every number quoted in Chapter 8, as in the text
    'consistency': {
        'same_as_chapter_3': 'yes',
    },
    'decisions of the forecasting run': {
        'period 86': 'p 24.060625 m, v 0.6625 m/s; initial set (24.124375 m, 0.6125 m/s); hold, proposed -2; requirement 0.5 m from pair 0; 7 candidates, 3 admitted; committed 5 (ambition 0): no demand: the most ambitious admitted candidate',
        'period 86 candidate 0': 'ambition 0; head -0.5 under 0.5; 13 pairs; rest 24.5 m; level 0.5 (clearance 0.5 m at the rest point 24.5 m); onset level 0.5 (clearance 0.5 m at the rest point 24.5 m); admitted: tie, a fresh candidate is preferred',
        'period 86 candidate 1': 'ambition 1; head -2 under 0.5; 10 pairs; rest 24.34625 m; level -inf (pair 0 not certified: acceleration towards the start 2 beyond the label 0.5); onset level -inf (pair 0 not certified: acceleration towards the start 2 beyond the label 0.5); not admitted: a pair not certified',
        'period 86 candidate 2': 'ambition 0.75; head -1.625 under 0.5; 10 pairs; rest 24.38 m; level -inf (pair 0 not certified: acceleration towards the start 1.625 beyond the label 0.5); onset level -inf (pair 0 not certified: acceleration towards the start 1.625 beyond the label 0.5); not admitted: a pair not certified',
        'period 86 candidate 3': 'ambition 0.5; head -1.25 under 0.5; 11 pairs; rest 24.4175 m; level -inf (pair 0 not certified: acceleration towards the start 1.25 beyond the label 0.5); onset level -inf (pair 0 not certified: acceleration towards the start 1.25 beyond the label 0.5); not admitted: a pair not certified',
        'period 86 candidate 4': 'ambition 0.25; head -0.875 under 0.5; 12 pairs; rest 24.4575 m; level -inf (pair 0 not certified: acceleration towards the start 0.875 beyond the label 0.5); onset level -inf (pair 0 not certified: acceleration towards the start 0.875 beyond the label 0.5); not admitted: a pair not certified',
        'period 86 candidate 5': 'ambition 0; head -0.5 under 0.5; 13 pairs; rest 24.5 m; level 0.5 (clearance 0.5 m at the rest point 24.5 m); onset level 0.5 (clearance 0.5 m at the rest point 24.5 m); committed',
        'period 86 candidate 6': 'ambition 0; head -0.5 under 0.5; 13 pairs; rest 24.5 m; level 0.5 (clearance 0.5 m at the rest point 24.5 m); onset level 0.5 (clearance 0.5 m at the rest point 24.5 m); admitted: tie, a smaller index is preferred',
        'period 179': 'p 24.5 m, v 0 m/s; initial set (24.5 m, 0 m/s); hold, proposed 0; requirement 1.3 m from pair 20; 7 candidates, 7 admitted; committed 6 (ambition 0): demand: highest first key and ambition, tie broken for a fresh candidate',
        'period 179 candidate 0': 'ambition 0; head -0.5 under 0.5; 20 pairs; rest 23.7 m; level 0.5 (clearance 0.5 m at pair 0, up to 24.5 m); onset level 1.3 (clearance 1.3 m at the rest point 23.7 m); admitted: tie, a fresh candidate is preferred',
        'period 179 candidate 1': 'ambition 1; head 0 under 0.5; 21 pairs; rest 23.7 m; level 0.5 (clearance 0.5 m at pair 1, up to 24.5 m); onset level 1.29 (clearance 1.29 m at pair 20, up to 23.71 m); admitted: lower first key',
        'period 179 candidate 2': 'ambition 0.75; head -0.125 under 0.5; 22 pairs; rest 23.69875 m; level 0.5 (clearance 0.5 m at pair 0, up to 24.5 m); onset level 1.26 (clearance 1.26125 m at pair 20, up to 23.73875 m); admitted: lower first key',
        'period 179 candidate 3': 'ambition 0.5; head -0.25 under 0.5; 22 pairs; rest 23.6975 m; level 0.5 (clearance 0.5 m at pair 0, up to 24.5 m); onset level 1.26 (clearance 1.2625 m at pair 20, up to 23.7375 m); admitted: lower first key',
        'period 179 candidate 4': 'ambition 0.25; head -0.375 under 0.5; 22 pairs; rest 23.69625 m; level 0.5 (clearance 0.5 m at pair 0, up to 24.5 m); onset level 1.26 (clearance 1.26375 m at pair 20, up to 23.73625 m); admitted: lower first key',
        'period 179 candidate 5': 'ambition 0; head -0.5 under 0.5; 22 pairs; rest 23.695 m; level 0.5 (clearance 0.5 m at pair 0, up to 24.5 m); onset level 1.26 (clearance 1.265 m at pair 20, up to 23.735 m); admitted: lower first key',
        'period 179 candidate 6': 'ambition 0; head -0.5 under 0.5; 20 pairs; rest 23.7 m; level 0.5 (clearance 0.5 m at pair 0, up to 24.5 m); onset level 1.3 (clearance 1.3 m at the rest point 23.7 m); committed',
        'period 185': 'p 24.4375 m, v -0.25 m/s; initial set (24.41 m, -0.3 m/s); hold, proposed 2; requirement 1.3 m from pair 14; 7 candidates, 7 admitted; committed 0 (ambition 0): fallback: the only candidate with the highest first key',
        'period 185 candidate 0': 'ambition 0; head -0.5 under 0.5; 14 pairs; rest 23.7 m; level 0.59 (clearance 0.59 m at pair 0, up to 24.41 m); onset level 1.3 (clearance 1.3 m at the rest point 23.7 m); committed',
        'period 185 candidate 1': 'ambition 1; head 2 under 0.5; 21 pairs; rest 23.68 m; level 0.59 (clearance 0.59 m at pair 0, up to 24.41 m); onset level 0.97 (clearance 0.975 m at pair 14, up to 24.025 m); admitted: lower first key',
        'period 185 candidate 2': 'ambition 0.75; head 1.375 under 0.5; 21 pairs; rest 23.67375 m; level 0.59 (clearance 0.59 m at pair 0, up to 24.41 m); onset level 0.98 (clearance 0.98125 m at pair 14, up to 24.01875 m); admitted: lower first key',
        'period 185 candidate 3': 'ambition 0.5; head 0.75 under 0.5; 22 pairs; rest 23.665 m; level 0.59 (clearance 0.59 m at pair 0, up to 24.41 m); onset level 0.93 (clearance 0.9325 m at pair 14, up to 24.0675 m); admitted: lower first key',
        'period 185 candidate 4': 'ambition 0.25; head 0.125 under 0.5; 22 pairs; rest 23.6525 m; level 0.59 (clearance 0.59 m at pair 0, up to 24.41 m); onset level 0.94 (clearance 0.945 m at pair 14, up to 24.055 m); admitted: lower first key',
        'period 185 candidate 5': 'ambition 0; head -0.5 under 0.5; 22 pairs; rest 23.64 m; level 0.59 (clearance 0.59 m at pair 0, up to 24.41 m); onset level 0.95 (clearance 0.9575 m at pair 14, up to 24.0425 m); admitted: lower first key',
        'period 185 candidate 6': 'ambition 0; head 2 under 0.5; 21 pairs; rest 23.68 m; level 0.59 (clearance 0.59 m at pair 0, up to 24.41 m); onset level 0.97 (clearance 0.975 m at pair 14, up to 24.025 m); admitted: lower first key',
        'outcomes: admitted: lower first key': '99',
        'outcomes: admitted: same first key, lower ambition': '835',
        'outcomes: admitted: tie, a fresh candidate is preferred': '64',
        'outcomes: admitted: tie, a smaller index is preferred': '62',
        'outcomes: committed': '220',
        'outcomes: not admitted: a pair not certified': '48',
        'outcomes: not admitted: below the floor': '212',
        'not_certified: periods': '86-97, every period',
        'not_certified: failing pairs': 'pair 0',
        'level -inf: clearance below the floor': '212',
        'level -inf: pair not certified: acceleration': '48',
        'decisions: demand: highest first key and ambition, tie broken for a fresh candidate': '2',
        'decisions: demand: the most ambitious of the candidates with the highest first key': '44',
        'decisions: fallback: the only candidate with the highest first key': '14',
        'decisions: no demand: the most ambitious admitted candidate': '160',
        'committed_from_announcement_to_onset': '140-178: 1; 179: 6; 180-193: 0; 194: 5; 195-199: 1',
        'committed_onset_level_from_announcement_to_onset': '1.3 m in all 60 periods',
        'periods': '220',
        'candidates': '1540',
        'fallback_profile_equals_its_evaluation': 'in all 220 periods',
    },
    'postponement': {
        'reactive_preparation_postponed_by_20': 'overloads in period 200, period 220; operation begins 24.1 s',
        'reactive_preparation_postponed_by_21': 'overloads in period 200; operation begins 22.2 s',
    },
    'configuration: forecast-informed': {
        'announced': 'in period 140 for period 200',
        'arrival': '8.6 s',
        'time_in_work_zone': '10.9 s',
        'backaway_from': '18.0 s',
        'at_rest_at_23_7_from': '20.0 s',
        'level_in_force_when_due': '1.3',
        'overloads': 'none',
        'operation_begins': '20.1 s',
        'operation_delay': '0.0 s',
        'lowest_level_before_operation': '0.5',
        'lowest_level': '0.5',
        'least_clearance': '0.5 m',
        'violations': '0',
        'leaves_work_zone': '19.5 s',
    },
    'configuration: reactive-preparation': {
        'announced': 'in period 200 for period 221',
        'arrival': '8.6 s',
        'time_in_work_zone': '13.0 s',
        'backaway_from': '20.1 s',
        'at_rest_at_23_7_from': '22.1 s',
        'level_in_force_when_due': '0.5',
        'overloads': 'period 200',
        'operation_begins': '22.2 s',
        'operation_delay': '2.1 s',
        'lowest_level_before_operation': '0.5',
        'lowest_level': '0.5',
        'least_clearance': '0.5 m',
        'violations': '0',
        'leaves_work_zone': '21.6 s',
        'reserve_in_force_from': 'period 221',
        'backaway_as_in_forecast_run': 'yes, shifted by 21 periods',
    },
    'configuration: fixed-conservatism': {
        'arrival': 'never',
        'time_in_work_zone': '0.0 s',
        'backaway_from': '-',
        'at_rest_at_23_7_from': '9.6 s',
        'level_in_force_when_due': '1.3',
        'overloads': 'none',
        'operation_begins': '20.1 s',
        'operation_delay': '0.0 s',
        'lowest_level_before_operation': '1.3',
        'lowest_level': '0.5',
        'least_clearance': '0.5 m',
        'violations': '0',
        'proposals_modified_before_onset': '168',
        'last_unchanged_before_onset': 'period 31',
    },
    'configuration: forecast-early': {
        'announced': 'in period 140 for period 190',
        'arrival': '8.6 s',
        'time_in_work_zone': '9.9 s',
        'backaway_from': '17.0 s',
        'at_rest_at_23_7_from': '19.0 s',
        'level_in_force_when_due': '1.3',
        'overloads': 'none',
        'operation_begins': '20.1 s',
        'operation_delay': '0.0 s',
        'lowest_level_before_operation': '0.5',
        'lowest_level': '0.5',
        'least_clearance': '0.5 m',
        'violations': '0',
        'leaves_work_zone': '18.5 s',
        'backaway_as_in_forecast_run': 'yes, shifted by -10 periods',
    },
    'configuration: forecast-late': {
        'announced': 'in period 140 for period 210',
        'arrival': '8.6 s',
        'time_in_work_zone': '11.9 s',
        'backaway_from': '19.0 s',
        'at_rest_at_23_7_from': '21.0 s',
        'level_in_force_when_due': '0.75',
        'overloads': 'period 200',
        'operation_begins': '21.1 s',
        'operation_delay': '1.0 s',
        'lowest_level_before_operation': '0.5',
        'lowest_level': '0.5',
        'least_clearance': '0.5 m',
        'violations': '0',
        'reserve_in_force_from': 'period 210',
        'backaway_as_in_forecast_run': 'yes, shifted by 10 periods',
    },
    'runtime consistency': {
        'same_as_chapter_3_with_full_evaluation': 'yes',
        'alarms_and_stops_as_in_section_7_2_3': 'yes',
    },
    'time budget: full': {
        'arrival': '8.6 s',
        'first_rest': '24.50 m from 10.0 s',
        'backaway_from': '18.0 s',
        'operation': '20.1 s',
        'fresh_candidates_evaluated': '1320 of 1320',
        'fallback_periods': '14',
        'lowest_level': '0.5',
        'least_clearance': '0.50 m',
        'violations': '0',
    },
    'time budget: one-candidate': {
        'arrival': '8.9 s',
        'first_rest': '24.48 m from 10.2 s',
        'backaway_from': '18.1 s',
        'operation': '20.1 s',
        'fresh_candidates_evaluated': '220 of 1320',
        'fallback_periods': '82',
        'lowest_level': '0.52',
        'least_clearance': '0.52 m',
        'violations': '0',
        'proposals_committed_unchanged': '138',
        'approach_fresh_commitments': 'periods 52, 78, 100, 101',
    },
    'time budget: stall': {
        'arrival': '11.2 s',
        'first_rest': '2.00 m from 2.1 s',
        'backaway_from': '18.1 s',
        'operation': '20.1 s',
        'fresh_candidates_evaluated': '1200 of 1320',
        'fallback_periods': '34',
        'lowest_level': '0.52',
        'least_clearance': '0.52 m',
        'violations': '0',
        'stall': 'periods 10 to 29, 1.0 to 3.0 s',
        'state_when_stall_begins': '0.810 m, 1.80 m/s',
        'plan_followed': 'committed in period 9, 11 pairs, rest point 2 m',
        'at_rest': '2.000 m from 2.1 s',
        'state_when_evaluation_resumes': '2.000 m, 0.00 m/s',
    },
    'time budget: no-evaluation': {
        'arrival': 'never',
        'first_rest': 'never moves',
        'backaway_from': '-',
        'operation': 'never: the reserve is held when due, but no candidate that unfolds the tool is evaluated',
        'fresh_candidates_evaluated': '0 of 1320',
        'fallback_periods': '220',
        'lowest_level': '25',
        'least_clearance': '25.00 m',
        'violations': '0',
    },
    'alarm: survey bias 0.2 m': {
        'arrival': '8.4 s',
        'alarm': 'period 56, 5.6 s; measured 20.123 m, 2.087 m/s; initial set 20.120 m, 2.000 m/s',
        'periods_without_restart_while_moving': '42',
        'at_rest': '24.48 m from 9.8 s',
        'restart': 'committed at 9.9 s (period 99), level 0.52',
        'labels_after_restart': '0.5',
        'backaway_from': '18.1 s',
        'operation': '20.1 s',
        'least_clearance': '0.52 m',
        'violations': '0',
    },
    'alarm: survey bias 0.35 m': {
        'alarm': 'period 56, 5.6 s; measured 20.148 m, 2.200 m/s; initial set 20.145 m, 2.100 m/s',
        'periods_without_restart_while_moving': '45',
        'at_rest': '24.99 m from 10.1 s',
        'restart': 'never; no restart candidate admitted: clearance 0.01 m at pair 0, up to 24.99 m, below the floor; no restart candidate admitted: clearance 0.01 m at the rest point 24.99 m, below the floor',
        'operation': 'overload in period 200',
        'least_clearance': '0.01 m',
        'violations': '0',
        'final_state': '24.99 m, 0.00 m/s',
    },
    'alarm: survey bias 0.2 m, rest feedback at 0.5 m/s^2': {
        'alarm': 'period 56, 5.6 s; measured 20.123 m, 2.087 m/s; initial set 20.120 m, 2.000 m/s',
        'periods_without_restart_while_moving': '0',
        'restart': 'committed at 5.7 s (period 57), level 0.51',
        'first_rest': '24.48 m from 9.8 s',
        'backaway_from': '18.1 s',
        'operation': '20.1 s',
        'least_clearance': '0.52 m',
        'violations': '0',
        'at_rest_after_alarm': '24.48047 m, 0.47 mm beyond the rest under maximal braking, 24.48 m',
    },
    'alarm: survey bias 0.35 m, rest feedback at 0.5 m/s^2': {
        'restart': 'never; no restart candidate admitted: clearance 0.01 m at pair 0, up to 24.99 m, below the floor; no restart candidate admitted: clearance 0.01 m at the rest point 24.99 m, below the floor',
        'operation': 'overload in period 200',
        'final_state': '24.99 m, 0.00 m/s',
    },
}


if __name__ == "__main__":
    main()
