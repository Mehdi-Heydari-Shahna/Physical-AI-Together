#!/usr/bin/env python3
"""
The lead time of the cart's operation, and the arithmetic of Section 9.3.

Physical AI, Together, Chapter 9 (Section 9.3).

The cart of Chapter 3 holds at rest at 24.5 m, with 0.5 m of reserve, when
the operation is announced; the operation needs 1.3 m from its onset on.
This script computes, in exact rational arithmetic:

  * the profile of the back-away that the generator of Chapter 3 offers
    from that state, and the lead time that Theorem 9.12(b) gives for it
    (21 periods), with labels of the forecasting and of the fixed predictor;
  * the rate form of Theorem 9.12(b) for the same candidate, which gives no
    positive rate on the grid of levels, and 321 periods without it;
  * the lower bound of Theorem 9.12(c) for every governor whose initial
    sets contain the state, from the largest acceleration towards the start
    on the ramp (19 periods);
  * a continuation that reverses at that acceleration until the cart is at
    or before 23.7 m and then brakes (the full reverse), which holds 1.3 m
    at the onset with 19 periods of lead time;
  * runs of the governor of Chapter 3, with its own generator and with a
    generator that also offers the full reverse, at every lead time from 1
    to 24 periods (Figure 9.4);
  * runs in which the requirement does not reach the governor in one period
    between the announcement and the onset (Section 9.3.3);
  * the retention property of Theorem 9.12(a) on the recorded decisions of
    the runs of Chapter 8 that carry a requirement;
  * the arithmetic of Proposition 9.16 (the persisting predictor) and of
    Example 9.17 (a label kept beyond its horizon).

It imports the cart, its segments, its test, its generator, its predictors
and its governor from ../ch03/cart_example.py and uses them unchanged; the
only additions are the full reverse and, for the runs of Section 9.3.3, a
requirement replaced by its default in one period. The retention check reads
the decision files in ../ch08/results/, so run ../ch08/cart_governor.py
first if they are missing.

Usage
    python3 cart_lead_time.py            # writes results/ (about 3 minutes)
    python3 cart_lead_time.py --check    # also checks every number quoted
                                         # in Chapter 9 and the references
                                         # to it in Chapters 3, 7 and 8

Outputs, in results/ next to this file
    lead-time.csv            level in force at the onset against the lead
                             time, in the runs with each generator, and the
                             bound of Theorem 9.12(c) (Figure 9.4)
    dropped-requirement.csv  outcome of the run with the requirement
                             missing in each period from 140 to 199
    retention.csv            every pair of consecutive periods checked for
                             retention, with the levels compared
    summary.txt              the numbers quoted in the text

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
LEVEL = ce.DEMAND_LEVEL                  # reserve the operation needs: 1.3 m
P_HOLD = ce.WALL - ce.EPS_MIN            # the cart holds at 24.5 m
TARGET = ce.WALL - LEVEL                 # a reserve of 1.3 m: front at 23.7 m
ONSET = ce.period(ce.T_ONSET)            # period 200
D = LEVEL - ce.EPS_MIN                   # reserve to be restored: 0.8 m
CH08 = os.path.join(HERE, "..", "ch08", "results")


def yes(b):
    return "yes" if b else "no"


# ----------------------------------------------------------------------------
# Two continuations from rest at 24.5 m
# ----------------------------------------------------------------------------
def full_reverse(pred, k, j, p, v, target):
    """Continuation from (p, v) at the start of period j: maximal certified
    braking to rest if the cart moves towards the wall, then the strongest
    certified backward command until the front is at or before the target,
    then maximal braking to rest. Unlike the back-away of Chapter 3, it does
    not stop at the target: it passes it at full authority and stops
    beyond."""
    pairs = []
    if v > 0:
        tail, p = ce.brake_to_rest(pred, k, j, p, v)
        pairs += tail
        j += len(tail)
        v = Fr(0)
    while p > target:
        seg, beta = ce.backward_step(pred, k, j, p, v)
        pairs.append((seg, beta))
        p, v, j = seg.p1, seg.v1, j + 1
    tail, p_stop = ce.brake_to_rest(pred, k, j, p, v)
    return pairs + tail, p_stop


def continuations(k0):
    """The back-away of the generator of Chapter 3 and the full reverse,
    both from rest at 24.5 m, as candidates of period k0 (labels of the
    forecasting predictor, which does not depend on the period within the
    horizon)."""
    pred = ce.Predictor("forecasting")
    pairs, p_rest = ce.recovery(pred, k0, k0 + 1, P_HOLD, Fr(0), TARGET)
    backaway = ce.Plan(k0 + 1, pairs, p_rest)
    pairs, p_rest = full_reverse(pred, k0, k0 + 1, P_HOLD, Fr(0), TARGET)
    reverse = ce.Plan(k0 + 1, pairs, p_rest)
    return backaway, reverse


def exact_suffix_reserve(plan):
    """Exact smallest clearance from pair n on, n = 0..N (entry N: the rest
    segment alone), without the grid of levels."""
    out = [None] * (len(plan.pairs) + 1)
    running = ce.margin(plan.p_rest, plan.ell)
    out[len(plan.pairs)] = running
    for i in range(len(plan.pairs) - 1, -1, -1):
        running = min(running, ce.margin(plan.pairs[i][0].hi, plan.ell))
        out[i] = running
    return out


def restoration_pairs(profile, level):
    """Smallest n from which the profile holds the level."""
    for n, e in enumerate(profile):
        if e is not ce.NEG_INF and e >= level:
            return n
    return None


def commands(plan):
    """The commands of a plan, run-length encoded: '16 at -0.5, 4 at 2'."""
    runs = []
    for seg, _ in plan.pairs:
        if runs and runs[-1][1] == seg.u:
            runs[-1][0] += 1
        else:
            runs.append([1, seg.u])
    return ", ".join(f"{n} at {ce.fmt(u)}" for n, u in runs)


def bound(j):
    """Largest gain of reserve in j periods of motion from rest on the ramp:
    the ground realizes at most 0.5 m/s^2 towards the start there, so the
    front moves back by at most 0.5 * 0.5 * (j T)^2."""
    return ce.B_RAMP * (j * T) ** 2 / 2


def level_from(profile, n):
    """Level of a plan from pair n on (the rest segment alone beyond N)."""
    return profile[min(n, len(profile) - 1)]


def intervals(ks):
    """Consecutive runs of a sorted list of integers, as (first, last)."""
    out = []
    for k in ks:
        if out and k == out[-1][1] + 1:
            out[-1][1] = k
        else:
            out.append([k, k])
    return [tuple(x) for x in out]


def smallest_lead(held_at):
    """Smallest lead time N_a >= 1 with held_at(N_a) >= 1.3 m."""
    for na in range(1, 200):
        if held_at(na) >= LEVEL:
            return na
    return None


# ----------------------------------------------------------------------------
# Runs of the governor of Chapter 3 at a given lead time
# ----------------------------------------------------------------------------
def lead_time_run(lead, extended):
    """The forecasting run of Chapter 3 with the operation announced `lead`
    periods before its onset; with extended=True, the generator also offers
    the full reverse, with ambition 0, whenever a demand is pending."""
    original = ce.generate

    def generate(pred, k, xhat, u_q, fallback, target, ell):
        cands = original(pred, k, xhat, u_q, fallback, target, ell)
        if target is not None:
            pairs, p_rest = full_reverse(pred, k, k + 1, xhat[0], xhat[1], target)
            if pairs:
                cands.append((ce.Plan(k + 1, pairs, p_rest, ell), Fr(0)))
        return cands

    if extended:
        ce.generate = generate
    try:
        rows, _, info = ce.run("forecasting", t_announce=ce.T_ONSET - lead * T,
                               t_end=ce.T_ONSET + T)
    finally:
        ce.generate = original
    assert info["alarm_k"] is None, "every label of these runs is valid"
    k0 = ONSET - lead
    r0 = next(r for r in rows if r.get("k") == k0)
    assert (r0["p"], r0["v"], r0["level"]) == (P_HOLD, 0, ce.EPS_MIN), \
        "the cart holds at 24.5 m when the operation is announced"
    met = info["k_met"] == ONSET
    assert met != info["overload"]
    level = next(r["level"] for r in rows if r.get("k") == ONSET)
    return met, level


def dropped_requirement_run(k_drop):
    """The forecasting run of Chapter 3 (announcement in period 140, onset
    in period 200), except that in period k_drop the requirement does not
    reach the governor, which then uses its default, the floor from pair 0
    on (no demand pending), for generation and selection alike."""
    orig_gen, orig_sel = ce.generate, ce.select
    now = {}

    def generate(pred, k, xhat, u_q, fallback, target, ell):
        now["k"] = k
        if k == k_drop:
            target = None
        return orig_gen(pred, k, xhat, u_q, fallback, target, ell)

    def select(cands, profiles, fb_profile, req, onset):
        if now.get("k") == k_drop:
            req, onset = ce.EPS_MIN, 0
        return orig_sel(cands, profiles, fb_profile, req, onset)

    ce.generate, ce.select = generate, select
    try:
        rows, _, info = ce.run("forecasting", t_end=ce.T_ONSET + T)
    finally:
        ce.generate, ce.select = orig_gen, orig_sel
    assert info["alarm_k"] is None
    met = info["k_met"] == ONSET
    assert met != info["overload"]
    level = next(r["level"] for r in rows if r.get("k") == ONSET)
    return met, level


# ----------------------------------------------------------------------------
# Retention on the recorded decisions of Chapter 8
# ----------------------------------------------------------------------------
RETENTION_RUNS = ("chapter3", "forecast-informed", "reactive-preparation",
                  "fixed-conservatism", "forecast-early", "forecast-late",
                  "budget-one-candidate", "budget-stall", "budget-no-evaluation")


def read_decisions(name):
    with open(os.path.join(CH08, f"decisions-{name}.csv"), newline="",
              encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        out.append(dict(k=int(r["k"]), req=Fr(r["req"]), onset=int(r["onset"]),
                        level=Fr(r["level"]), onset_level=Fr(r["onset_level"])))
    return out


def retention_check(name):
    """Theorem 9.12(a) on one run: for consecutive periods with the same
    requirement for the same onset period, the onset level of the committed
    plan, capped at the requirement, never decreases; once the onset pair is
    pair 0 in both periods, the same holds for the level; and in the period
    of an announcement, the capped onset level is at least the capped level
    of the plan in force."""
    rows = read_decisions(name)
    checked, records = 0, []
    for a, b in zip(rows, rows[1:]):
        assert b["k"] == a["k"] + 1
        if b["req"] <= ce.EPS_MIN:
            continue
        if a["req"] <= ce.EPS_MIN:                     # announcement in period b
            kind, before, after = "announcement", a["level"], b["onset_level"]
        elif a["req"] != b["req"]:
            continue
        elif b["onset"] == a["onset"] - 1:             # same onset period
            kind, before, after = "same onset", a["onset_level"], b["onset_level"]
        elif a["onset"] == 0 and b["onset"] == 0:      # from the first pair on
            kind, before, after = "first pair", a["level"], b["level"]
        else:
            continue
        ok = min(after, b["req"]) >= min(before, b["req"])
        assert ok, f"retention fails in run {name}, period {b['k']}"
        checked += 1
        records.append(dict(run=name, k=b["k"], kind=kind, req=b["req"],
                            before=before, after=after))
    return checked, records


# ----------------------------------------------------------------------------
# Proposition 9.16 and Example 9.17: arithmetic
# ----------------------------------------------------------------------------
def persisting_arithmetic():
    """The proof of Proposition 9.16. Step 1: a head ends at most at 7.2 m/s,
    and its continuation, braking at min(2, v/T) per period, takes at most
    36 periods; only the last of them brakes less, which adds at most
    max over v_r in (0, 2T] of v_r (T/2 - v_r/4) = T^2/4 = 0.0025 m.
    Step 2: if the initial run of full acceleration ended at or before
    11.53 m, the next full-acceleration head would end at or before
    12.21914 m and be certified, because sqrt(11.53) <= 3.3957. Steps 3 and
    4: v^2 at 20 m and the distance needed on the ramp."""
    periods = -(-Fr(72, 10) // (2 * T))
    extra = max(vr * (T / 2 - vr / 4) for vr in (Fr(i, 1000) for i in range(1, 201)))
    assert extra == T * T / 4                      # the maximum, at v_r = T
    p_hat, s = Fr(1153, 100), Fr(33957, 10000)
    assert s * s >= p_hat
    p1_max = p_hat + 2 * s * T + T * T             # one more period at 2 m/s^2
    p1_reject = (ce.WALL - ce.EPS_MIN - extra) / 2  # 2 p1 + 0.0025 > 24.5
    v2_ramp = 8 * p_hat - 4 * ce.RAMP              # v^2 at 20 m, braking <= 2
    stop_min = ce.RAMP + v2_ramp / (2 * ce.B_RAMP)
    return dict(
        continuation_periods_at_most=periods,
        last_braking_period_adds_at_most=ce.fmt(extra),
        rejection_threshold=ce.fmt(p1_reject),
        next_head_if_initial_run_ended_at_11_53=ce.fmt(p1_max),
        contradiction=yes(p1_max < p1_reject),
        v_squared_at_end_of_initial_run_more_than=ce.fmt(4 * p_hat),
        v_squared_at_ramp_more_than=ce.fmt(v2_ramp),
        stops_beyond=ce.fmt(stop_min),
    )


def stale_label_arithmetic():
    """Example 9.17: 35 periods at 2 m/s^2 from rest, then 35 periods of
    braking at 2 m/s^2, planned under labels of 2 m/s^2 issued in period 0;
    pair n is executed in period 1 + n. In the world, braking is 2 m/s^2 on
    firm ground and 0.5 m/s^2 on the ramp."""
    pairs, p, v = [], Fr(0), Fr(0)
    for u in [Fr(2)] * 35 + [Fr(-2)] * 35:
        seg = ce.Segment(p, v, u)
        pairs.append(seg)
        p, v = seg.p1, seg.v1
    assert v == 0 and all(seg.certified(ce.B_FIRM) for seg in pairs)
    top = pairs[34]
    first_on_ramp = next(n for n, seg in enumerate(pairs) if seg.hi >= ce.RAMP)
    v2_ramp = top.v1 ** 2 - 2 * ce.B_FIRM * (ce.RAMP - top.p1)
    stop = ce.RAMP + v2_ramp / (2 * ce.B_RAMP)
    return dict(after_acceleration=f"{ce.fmt(top.p1)} m, {ce.fmt(top.v1)} m/s "
                f"after 35 periods",
                pairs=len(pairs),
                planned_rest=f"{ce.fmt(p)} m",
                first_period_reaching_the_ramp=1 + first_on_ramp,
                true_horizon_at_least=len(pairs) - 1,
                v_squared_at_ramp=ce.fmt(v2_ramp),
                speed_at_ramp=f"{float(v2_ramp) ** 0.5:.2f} m/s",
                stop=f"{ce.fmt(stop)} m")


def fixed_backaway_is_the_same(k0):
    """Section 9.3.3: with the fixed predictor, the back-away from rest at
    24.5 m has the same commands, labels aside, and the same profile."""
    fixed, forecast = ce.Predictor("fixed"), ce.Predictor("forecasting")
    a, pa = ce.recovery(fixed, k0, k0 + 1, P_HOLD, Fr(0), TARGET)
    b, pb = ce.recovery(forecast, k0, k0 + 1, P_HOLD, Fr(0), TARGET)
    same = ([s.u for s, _ in a] == [s.u for s, _ in b] and pa == pb
            and ce.Plan(k0 + 1, a, pa).profile() == ce.Plan(k0 + 1, b, pb).profile())
    return same


def exercise_sensor_bias(bias):
    """Exercise 9.1: the cart starts at rest at p = bias, and the position
    sensor reads bias metres short, so the governor receives exactly the
    measurements of the fixed run of Chapter 3. Checks that the ground
    realizes every executed command exactly from the true, shifted state, so
    that each measurement equals the end of the executed funnel and no alarm
    is raised, and returns the true positions that the answer quotes."""
    rows, _, info = ce.run("fixed")
    assert info["alarm_k"] is None
    steps = [r for r in rows if not r.get("final")]
    for r, nxt in zip(steps, rows[1:]):
        p1, v1, exact = ce.realize(r["p"] + bias, r["v"], r["u"])
        assert exact and (p1 - bias, v1) == (nxt["p"], nxt["v"])
    rest = next(r["p"] for r in steps if r["p"] > 0 and r["v"] == 0)
    unfold = next(r["p"] for r in steps if r["ell"] > 0)
    # largest p + l in the measured motion, as in Chapter 3: positions are
    # monotone within a period and the mode is constant
    ends = [r["p"] for r in rows[1:]]
    reach = max(max(r["p"], e) + r["ell"] for r, e in zip(steps, ends))
    return dict(measured_rest=rest, true_rest=rest + bias,
                true_front_at_unfolding=unfold + bias,
                true_largest_reach=reach + bias)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Chapter 9: the lead time of the cart")
    ap.add_argument("--check", action="store_true",
                    help="check every number quoted in Chapter 9")
    args = ap.parse_args()
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    facts = {}

    # --- the two continuations from rest at 24.5 m ---------------------------
    backaway, reverse = continuations(ONSET - 19)
    prof_b, prof_r = backaway.profile(), reverse.profile()
    ex_b, ex_r = exact_suffix_reserve(backaway), exact_suffix_reserve(reverse)
    nr_b, nr_r = restoration_pairs(prof_b, LEVEL), restoration_pairs(prof_r, LEVEL)
    facts["back-away of Chapter 3"] = dict(
        commands=commands(backaway), pairs=len(backaway.pairs),
        rest=f"{ce.fmt(backaway.p_rest)} m",
        levels_from_pairs_0_1_16_19_20=", ".join(
            ce.fmt(prof_b[n]) for n in (0, 1, 16, 19, 20)),
        exact_reserve_from_pairs_1_2_16=", ".join(ce.fmt(ex_b[n]) for n in (1, 2, 16)),
        restoration_pairs=nr_b,
        sufficient_lead_time=f"{nr_b + 1} periods (N0 = 0)",
    )
    # rate form: the largest rate with level(n) >= min(0.5 + n rate, 1.3)
    def best_rate(prof):
        rates = [(prof[n] - ce.EPS_MIN) / n for n in range(1, len(prof))
                 if prof[n] < LEVEL]
        return min(rates)
    rate_grid, rate_exact = best_rate(prof_b), best_rate(ex_b)
    facts["rate form for the back-away"] = dict(
        largest_rate_on_the_grid=ce.fmt(rate_grid),
        largest_rate_without_the_grid=f"{ce.fmt(rate_exact)} m per period",
        restoration_pairs_from_that_rate=-(-D // rate_exact),
        lead_time_from_that_rate=f"{-(-D // rate_exact) + 1} periods",
    )
    facts["back-away of Chapter 3"]["same_with_the_fixed_predictor"] = yes(
        fixed_backaway_is_the_same(ONSET - 19))
    facts["full reverse"] = dict(
        commands=commands(reverse), pairs=len(reverse.pairs),
        rest=f"{ce.fmt(reverse.p_rest)} m",
        level=ce.fmt(prof_r[0]),
        position_at_start_of_pair_18=f"{ce.fmt(reverse.pairs[18][0].p0)} m",
        speed_at_start_of_pair_18=f"{ce.fmt(reverse.pairs[18][0].v0)} m/s",
        level_from_pair_17=ce.fmt(prof_r[17]),
        level_from_pair_18=ce.fmt(prof_r[18]),
        restoration_pairs=nr_r,
        lead_time=f"{nr_r + 1} periods",
        rest_farther_back_than_back_away=f"{ce.fmt(backaway.p_rest - reverse.p_rest)} m",
        periods_fewer_than_back_away=nr_b - nr_r,
    )
    # every pair of both continuations is certified under its label
    assert all(seg.certified(b) for seg, b in backaway.pairs + reverse.pairs)

    # --- the lower bound for every governor ------------------------------------
    na_low = smallest_lead(lambda na: ce.EPS_MIN + bound(na - 1))
    assert P_HOLD - bound(na_low + 2) >= ce.RAMP, "the cart stays on the ramp"
    facts["lower bound for every governor"] = dict(
        reserve_at_announcement=f"{ce.fmt(ce.EPS_MIN)} m",
        reserve_to_restore=f"{ce.fmt(D)} m",
        reserve_gain_in_j_periods=f"{ce.fmt(bound(1))} j^2 m",
        commitment_delay_periods=1,
        smallest_lead_time=f"{na_low} periods",
        smallest_lead_time_seconds=f"{ce.fmt(na_low * T)} s",
        gain_with_one_period_less=ce.fmt(bound(na_low - 2)),
        bound_at_one_period_less=ce.fmt(ce.EPS_MIN + bound(na_low - 2)),
        bound_at_smallest=ce.fmt(ce.EPS_MIN + bound(na_low - 1)),
    )
    # the full reverse attains the bound until it brakes
    assert all(ex_r[n] == ce.EPS_MIN + bound(n) for n in range(0, nr_r + 1))

    # --- runs of the governor at every lead time from 1 to 24 periods ---------
    # (Figure 9.4). Below the smallest lead time that meets the demand, the
    # level in force at the onset is the level of the continuation committed
    # at the announcement from its pair N_a - 1 on; from it on, 1.3 m.
    leads = range(1, 25)
    runs = {(ext, na): lead_time_run(na, ext) for ext in (False, True) for na in leads}
    for ext, prof in ((False, prof_b), (True, prof_r)):
        for na in leads:
            met, level = runs[(ext, na)]
            planned = level_from(prof, na - 1)
            assert met == (planned >= LEVEL)
            assert level == (LEVEL if met else planned), (ext, na)
    smallest_b = min(na for na in leads if runs[(False, na)][0])
    smallest_r = min(na for na in leads if runs[(True, na)][0])

    def outcome(ext, na):
        met, level = runs[(ext, na)]
        return f"{'met' if met else 'overload'}, level {ce.fmt(level)} m at the onset"
    facts["governor runs"] = dict(
        lead_times_run="1 to 24 periods, with each generator",
        chapter3_generator_lead_21=outcome(False, 21),
        chapter3_generator_lead_20=outcome(False, 20),
        with_full_reverse_lead_19=outcome(True, 19),
        with_full_reverse_lead_18=outcome(True, 18),
        smallest_lead_time_chapter3_generator=f"{smallest_b} periods",
        smallest_lead_time_with_full_reverse=f"{smallest_r} periods",
        announcement_for_lead_21="17.9 s",
        announcement_for_lead_19="18.1 s",
    )
    assert (smallest_b, smallest_r) == (nr_b + 1, nr_r + 1)
    assert ce.T_ONSET - 21 * T == Fr(179, 10) and ce.T_ONSET - 19 * T == Fr(181, 10)
    with open(os.path.join(out_dir, "lead-time.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["lead", "run_backaway", "run_reverse", "bound",
                    "met_backaway", "met_reverse"])
        for na in leads:
            w.writerow([na, ce.fmt(runs[(False, na)][1]), ce.fmt(runs[(True, na)][1]),
                        ce.fmt(ce.EPS_MIN + bound(na - 1)),
                        int(runs[(False, na)][0]), int(runs[(True, na)][0])])

    # --- the requirement missing in one period (Section 9.3.3) ----------------
    k0, k_last = ONSET - 60, ONSET - 1
    drops = {k: dropped_requirement_run(k) for k in range(k0, k_last + 1)}
    harmful = [k for k, (met, _) in drops.items() if not met]
    assert harmful == list(range(harmful[0], harmful[-1] + 1)), "one interval"
    with open(os.path.join(out_dir, "dropped-requirement.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["period_missing", "met", "level_at_onset"])
        for k in sorted(drops):
            w.writerow([k, int(drops[k][0]), ce.fmt(drops[k][1])])
    facts["requirement missing in one period"] = dict(
        demand="announced in period 140 for period 200 (60 periods ahead)",
        periods_tried=f"{k0} to {k_last}",
        overload_if_missing_in=f"periods {harmful[0]} to {harmful[-1]}",
        met_if_missing_in=", ".join(
            f"{a} to {b}" if a != b else f"{a}"
            for a, b in intervals(sorted(set(drops) - set(harmful)))),
        level_at_onset_when_overloaded=(
            f"{ce.fmt(min(drops[k][1] for k in harmful))} to "
            f"{ce.fmt(max(drops[k][1] for k in harmful))} m"),
    )

    # --- retention on the decisions of Chapter 8 ---------------------------------
    total, all_records = 0, []
    per_run = {}
    for name in RETENTION_RUNS:
        n, recs = retention_check(name)
        per_run[name] = n
        total += n
        all_records += recs
    with open(os.path.join(out_dir, "retention.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "k", "kind", "req", "before", "after"])
        for r in all_records:
            w.writerow([r["run"], r["k"], r["kind"], ce.fmt(r["req"]),
                        ce.fmt(r["before"]), ce.fmt(r["after"])])
    facts["retention on the runs of Chapter 8"] = dict(
        runs=len(RETENTION_RUNS), periods_checked=total, violations=0,
        **{f"periods_{name}": per_run[name] for name in RETENTION_RUNS})

    facts["proposition 9.16"] = persisting_arithmetic()
    facts["example 9.17"] = stale_label_arithmetic()

    # --- the exercise: a position sensor biased by 0.6 m, and by 0.4 m --------
    ex6, ex4 = exercise_sensor_bias(Fr(3, 5)), exercise_sensor_bias(Fr(2, 5))
    facts["exercise 9.1"] = {
        "measured_rest_0.6": ce.fmt(ex6["measured_rest"]),
        "true_rest_0.6": ce.fmt(ex6["true_rest"]),
        "beyond_wall_0.6": ce.fmt(ex6["true_rest"] - ce.WALL),
        "true_front_at_unfolding_0.6": ce.fmt(ex6["true_front_at_unfolding"]),
        "tool_tip_beyond_wall_0.6": yes(ex6["true_largest_reach"] > ce.WALL),
        "true_rest_0.4": ce.fmt(ex4["true_rest"]),
        "past_the_floor_0.4": ce.fmt(ex4["true_rest"] - P_HOLD),
        "inside_the_constraint_0.4": yes(ex4["true_largest_reach"] <= ce.WALL),
    }

    lines, failures = [], []
    for group, items in facts.items():
        lines.append(f"[{group}]")
        for key, val in items.items():
            lines.append(f"  {key} = {val}")
            want = EXPECTED.get(group, {}).get(key)
            if args.check and want is not None and want != str(val):
                failures.append(f"{group}.{key}: expected {want!r}, got {val!r}")
    if args.check:
        failures += [f"{group}.{key}: not computed" for group, items in EXPECTED.items()
                     for key in items if key not in facts.get(group, {})]
        n = sum(len(v) for v in EXPECTED.values())
        lines.append(f"[check] {n} values quoted in Chapter 9: " +
                     ("all as quoted" if not failures else "FAILED: " + "; ".join(failures)))
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if failures:
        raise SystemExit(1)


EXPECTED = {   # every number quoted in Chapter 9, and in its references in
               # Chapters 3, 7 and 8, as in the text
    "back-away of Chapter 3": {
        "commands": "16 at -0.5, 4 at 2",
        "pairs": "20",
        "rest": "23.7 m",
        "levels_from_pairs_0_1_16_19_20": "0.5, 0.5, 1.14, 1.29, 1.3",
        "exact_reserve_from_pairs_1_2_16": "0.5025, 0.51, 1.14",
        "restoration_pairs": "20",
        "sufficient_lead_time": "21 periods (N0 = 0)",
        "same_with_the_fixed_predictor": "yes",
    },
    "rate form for the back-away": {
        "largest_rate_on_the_grid": "0",
        "largest_rate_without_the_grid": "0.0025 m per period",
        "restoration_pairs_from_that_rate": "320",
        "lead_time_from_that_rate": "321 periods",
    },
    "full reverse": {
        "commands": "18 at -0.5, 4 at 2, 1 at 1",
        "pairs": "23",
        "rest": "23.485 m",
        "position_at_start_of_pair_18": "23.69 m",
        "speed_at_start_of_pair_18": "-0.9 m/s",
        "level_from_pair_17": "1.22",
        "level_from_pair_18": "1.31",
        "restoration_pairs": "18",
        "lead_time": "19 periods",
        "rest_farther_back_than_back_away": "0.215 m",
        "periods_fewer_than_back_away": "2",
    },
    "lower bound for every governor": {
        "reserve_at_announcement": "0.5 m",
        "reserve_to_restore": "0.8 m",
        "reserve_gain_in_j_periods": "0.0025 j^2 m",
        "commitment_delay_periods": "1",
        "smallest_lead_time": "19 periods",
        "smallest_lead_time_seconds": "1.9 s",
        "gain_with_one_period_less": "0.7225",
        "bound_at_one_period_less": "1.2225",
        "bound_at_smallest": "1.31",
    },
    "governor runs": {
        "lead_times_run": "1 to 24 periods, with each generator",
        "chapter3_generator_lead_21": "met, level 1.3 m at the onset",
        "chapter3_generator_lead_20": "overload, level 1.29 m at the onset",
        "with_full_reverse_lead_19": "met, level 1.3 m at the onset",
        "with_full_reverse_lead_18": "overload, level 1.22 m at the onset",
        "smallest_lead_time_chapter3_generator": "21 periods",
        "smallest_lead_time_with_full_reverse": "19 periods",
        "announcement_for_lead_21": "17.9 s",
        "announcement_for_lead_19": "18.1 s",
    },
    "requirement missing in one period": {
        "demand": "announced in period 140 for period 200 (60 periods ahead)",
        "periods_tried": "140 to 199",
        "overload_if_missing_in": "periods 179 to 194",
        "met_if_missing_in": "140 to 178, 195 to 199",
        "level_at_onset_when_overloaded": "0.91 to 1.29 m",
    },
    "retention on the runs of Chapter 8": {
        "runs": "9",
        "periods_checked": "650",
        "violations": "0",
    },
    "proposition 9.16": {
        "continuation_periods_at_most": "36",
        "last_braking_period_adds_at_most": "0.0025",
        "rejection_threshold": "12.24875",
        "next_head_if_initial_run_ended_at_11_53": "12.21914",
        "contradiction": "yes",
        "v_squared_at_end_of_initial_run_more_than": "46.12",
        "v_squared_at_ramp_more_than": "12.24",
        "stops_beyond": "32.24",
    },
    "example 9.17": {
        "after_acceleration": "12.25 m, 7 m/s after 35 periods",
        "pairs": "70",
        "planned_rest": "24.5 m",
        "first_period_reaching_the_ramp": "49",
        "true_horizon_at_least": "69",
        "v_squared_at_ramp": "18",
        "speed_at_ramp": "4.24 m/s",
        "stop": "38 m",
    },
    "exercise 9.1": {
        "measured_rest_0.6": "24.495",
        "true_rest_0.6": "25.095",
        "beyond_wall_0.6": "0.095",
        "true_front_at_unfolding_0.6": "24.295",
        "tool_tip_beyond_wall_0.6": "yes",
        "true_rest_0.4": "24.895",
        "past_the_floor_0.4": "0.395",
        "inside_the_constraint_0.4": "yes",
    },
}


if __name__ == "__main__":
    main()
