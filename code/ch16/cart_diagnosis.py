#!/usr/bin/env python3
"""
The diagnosis of the anticipation benchmark's exceptions (Section 16.3).

Physical AI, Together, Chapter 16.

Written after the first full run of cart_benchmark.py, which found
hypotheses A5 and A6 not supported (HYPOTHESES.md, Deviations). This script
is not part of the registered benchmark: it decides nothing, and it changes
no module. It reruns the runs that the exceptions concern, in the same
configurations and worlds, reads their records, and reports what the
governor and the proposal methods did there:

  * A5, world 16063: where the cart came to rest with and without the
    capability forecast, and the level of the plan in force at the
    postponed onset;
  * A6, the planner's timing: in the valid worlds in which the filter with
    the preview planner (C11) spends one or two periods less in the work
    zone than ALLOVI (C6), when each begins to back away, and whether its
    cart is at rest when the operation begins;
  * A6, short lead times: in the valid worlds with a lead time of 18 periods
    or fewer in which C6 spends three or more periods longer in the work
    zone, where each cart waits for the postponed onset;
  * A6, world 16070: the requirement once the forecast onset has passed,
    where C6's cart stops, and when each configuration begins the operation;
  * A6, the planner under ALLOVI (C14): how often the governor modifies its
    'retreat', and when its cart begins to back away;
  * the worlds that are not valid: every run of a survey configuration with
    an invalid executed label, and the one such run without an alarm;
  * the substitution grid (cart_substitution.py), whose B2 to B4 were not
    supported either: for each package, the braking it declares on the
    ramp and that braking on the grid of 1/1000 m/s^2 on which the braking
    curve and the planners round their requests; in the 20 evaluation
    worlds, how often the braking curve and the expert planner ask for more
    braking than declared on the ramp, and how often the governor commits
    such a request in full; and, from results/substitution-evaluation.csv,
    the quantities of B2 to B4 without the contraction-metric package, an
    exclusion made after the run, which tests nothing.

Usage
    python3 cart_diagnosis.py            # writes results/diagnosis-summary.txt
    python3 cart_diagnosis.py --check    # also checks every value
    python3 cart_diagnosis.py --jobs 1   # one process (default: up to two)

Python 3.8 or later; no other packages. About five minutes with two processes,
ten with one.
"""
import argparse
import csv
import os
import sys
from fractions import Fraction as Fr
from multiprocessing import Pool

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cart_benchmark as cb                                       # noqa: E402
import allovi                                                     # noqa: E402
from bench import modules as bm                                   # noqa: E402
from bench import worlds as bw                                    # noqa: E402
from cart.model import ce                                         # noqa: E402

SURVEY = [c for c, v in cb.CONFIGS.items() if v[1] == "survey"]
PACKAGES = ("tube", "qp", "ccm", "mpc")
ROUNDING = ("braking curve", "expert planner")
EVAL_WORLDS = range(1000, 1020)


def record(spec):
    """One configuration in one world, as cart_benchmark.py runs it: the
    evaluator's dictionary and, per period, what the diagnosis reads."""
    cid, seed = spec
    world = bw.World(seed)
    cfg = cb.configuration(cid, world)
    ce.RAMP = world.ramp
    try:
        rec = allovi.run(cfg, cb.PERIODS, strict=True)
        m = cfg.evaluator.evaluate(rec)
        per = []
        for x in rec.periods:
            d = x.decision
            head = None
            if x.governor == "normal" and x.plan is not None and x.plan.pairs:
                seg, beta = x.plan.pairs[0]       # the label's validity in this world
                head = (seg.lo, seg.hi, seg.u, beta, seg.p1, seg.v1,
                        beta > ce.b_min(seg.lo, seg.hi))
            per.append(dict(k=x.k, p=x.state[0], v=x.state[1], ell=x.state[2],
                            proposal=x.proposal, onset=x.requirement.onset,
                            req=x.requirement.level, level=x.level,
                            ambition=None if d is None else d.ambition, head=head,
                            governor=x.governor))
        per.append(dict(k=len(rec.periods), p=rec.final_state[0], v=rec.final_state[1],
                        ell=rec.final_state[2]))
    finally:
        ce.RAMP = cb.RAMP_CH3
    return cid, seed, m, per


def grid_record(spec):
    """One evaluation run of the substitution grid, as cart_substitution.py
    runs it, for a proposal method that rounds its requests: the periods
    before arrival, those whose request asks for more braking than the
    package declares on the ramp while the cart is on it, those of them
    committed in full, and the run's mean intervention."""
    import cart_substitution as cs
    package, method, seed = spec
    cl = cs.cl
    cs.use_package(package)
    try:
        pm = cl.BrakingCurve() if method == "braking curve" else cl.expert()
        rows, arrived = cl.episode(pm, cl.World(seed), seed=seed)
        b_low = cl.DECLARED["b_low"]
        pre = rows if arrived is None else rows[:arrived]

        def beyond(q):
            if q is None:
                return False
            qs = q if isinstance(q, (list, tuple)) else [q]
            return any(Fr(a) < -b_low for a in qs)

        over = [r for r in pre if r["d_low"] == 0 and beyond(r["q"])]
        load = cl.run_metrics(rows, arrived)["load_exact"]
        return dict(package=package, method=method, seed=seed, b_low=b_low, n=len(pre),
                    beyond=len(over), full=sum(1 for r in over if r["alpha"] == 1),
                    load=load)
    finally:
        cs.use_package("tube")


def without(cells, drop):
    """B2 to B4's quantities over the packages other than 'drop', from the
    evaluation runs: arrivals exact, interventions as the CSV rounds them."""
    import cart_substitution as cs
    pk = [p for p in PACKAGES if p != drop]
    mean = lambda xs: sum(xs, Fr(0)) / len(xs)
    arr = {k: mean([a for a, _ in v]) for k, v in cells.items()}
    load = {k: mean([b for _, b in v]) for k, v in cells.items()}
    pm = {p: mean([arr[(p, m)] for m in cs.METHODS]) for p in pk}
    mm = {m: mean([arr[(p, m)] for p in pk]) for m in cs.METHODS}
    out = {"B2: the ranges of the packages' and of the methods' mean arrival [s]":
           f"{float(max(pm.values()) - min(pm.values())):.3f}, "
           f"{float(max(mm.values()) - min(mm.values())):.3f}"}
    orders = {m: " < ".join(sorted(pk, key=lambda p: arr[(p, m)])) for m in cs.METHODS}
    out["B3: the packages from earliest to latest, by method"] = "; ".join(
        f"{m}: {o}" for m, o in orders.items())
    for group, methods in (("reference", cs.REFERENCE), ("sequence", cs.SEQUENCE)):
        me = {m: mean([load[(p, m)] for p in pk]) for m in methods}
        pp = {p: mean([load[(p, m)] for m in methods]) for p in pk}
        out[f"B4, {group}: the ranges of the methods' and of the packages' mean "
            f"intervention"] = (f"{float(max(me.values()) - min(me.values())):.3f}, "
                                f"{float(max(pp.values()) - min(pp.values())):.3f}")
    return out


def m4(x):
    return f"{float(x):.4f}"


def first_back(per, k0):
    """The first period from k0 on whose state at its start moves backwards:
    the cart moved backwards in the period before it."""
    return next(r["k"] for r in per if r["k"] >= k0 and r["v"] < 0) - 1


def counts(values):
    d = {}
    for x in values:
        d[x] = d.get(x, 0) + 1
    return ", ".join(f"{x} ({n})" for x, n in sorted(d.items()))


def rest_spells(per, k0, k1):
    """Spells of rest in the work zone between periods k0 and k1: (first
    period, last period, position)."""
    out, cur = [], None
    for r in per:
        if not k0 <= r["k"] < k1:
            continue
        here = r["v"] == 0 and r["p"] >= ce.WORK_ZONE and r["ell"] == 0
        if here and cur is None:
            cur = [r["k"], r["k"], r["p"]]
        elif here and r["p"] == cur[2]:
            cur[1] = r["k"]
        else:
            if cur is not None:
                out.append(tuple(cur))
            cur = [r["k"], r["k"], r["p"]] if here else None
    if cur is not None:
        out.append(tuple(cur))
    return out


def main():
    ap = argparse.ArgumentParser(description="Chapter 16: the diagnosis of the exceptions")
    ap.add_argument("--check", action="store_true", help="check every value")
    ap.add_argument("--jobs", type=int, default=2, help="processes (default 2)")
    args = ap.parse_args()
    worlds = {w.seed: w for w in bw.family()}
    valid = sorted(s for s, w in worlds.items() if cb.valid(w))
    invalid = sorted(s for s in worlds if s not in valid)
    specs = sorted({(c, s) for c in ("C6", "C11", "C14") for s in valid}
                   | {(c, 16063) for c in ("C6", "C7", "C8")}
                   | {(c, 16070) for c in ("C6", "C11", "C13", "C14")}
                   | {(c, s) for c in SURVEY for s in invalid})
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            out = pool.map(record, specs, chunksize=4)
    else:
        out = [record(s) for s in specs]
    R = {(c, s): (m, per) for c, s, m, per in out}
    facts = {}

    # A5: world 16063 ---------------------------------------------------------
    w = worlds[16063]
    g = {}
    g["the announcement; the forecast onset; the true onset; the onset after the first "
      "postponement"] = f"{w.announce}; {w.onset_forecast}; {w.onset}; {w.onset_forecast}"
    g["the cart at rest at the announcement, C6; C7; C8 [m]"] = "; ".join(
        m4(R[(c, 16063)][1][w.announce]["p"]) for c in ("C6", "C7", "C8"))
    g["the level of the plan in force in the postponed onset's period, C6; C7; C8 [m]"] = "; ".join(
        f"{float(R[(c, 16063)][1][w.onset_forecast]['level']):.2f}" for c in ("C6", "C7", "C8"))
    g["the operation begun, C6; C7; C8"] = "; ".join(
        str(R[(c, 16063)][0]["begun"]) for c in ("C6", "C7", "C8"))
    facts["A5, world 16063"] = g

    # A6: the planner's timing ------------------------------------------------
    wz = {s: R[("C6", s)][0]["in_work_zone"] - R[("C11", s)][0]["in_work_zone"] for s in valid}
    small = [s for s in valid if wz[s] in (1, 2)]
    g = {}
    g["valid worlds in which C11 spends one or two periods less in the work zone than C6"] = (
        f"{len(small)}: " + ", ".join(map(str, small)))
    g["the period in which the cart first moves backwards after the announcement, C6 less "
      "C11: value (worlds)"] = counts(
        first_back(R[("C6", s)][1], worlds[s].announce)
        - first_back(R[("C11", s)][1], worlds[s].announce) for s in small)
    at_rest = {c: sum(1 for s in small
                      if R[(c, s)][1][R[(c, s)][0]["begun"]]["v"] == 0) for c in ("C11", "C6")}
    g["the cart at rest at the start of the period in which the operation begins, C11; C6 "
      "(worlds)"] = f"{at_rest['C11']}; {at_rest['C6']}"
    behind = [s for s in small
              if R[("C6", s)][1][R[("C6", s)][0]["begun"]]["p"] <= bm.TARGET]
    g["C6's front at or before 23.7 m at the start of that period (worlds)"] = str(len(behind))
    facts["A6, the planner's timing"] = g

    # A6: short lead times ------------------------------------------------------
    big = [s for s in valid if wz[s] >= 3 and worlds[s].lead <= 18
           and R[("C6", s)][0]["begun"] == R[("C11", s)][0]["begun"]]
    g = {"valid worlds with a lead time of 18 periods or fewer in which C6 spends three or more "
         "periods longer in the work zone and begins the operation with C11":
         ", ".join(map(str, big))}
    for s in big:
        w = worlds[s]
        m6, per6 = R[("C6", s)]
        m11, per11 = R[("C11", s)]
        first = m6["overloads"][0]
        spells = rest_spells(per6, first, m6["begun"])
        a, b, p = max(spells, key=lambda x: x[1] - x[0])
        r11 = next(r for r in per11
                   if r["k"] > w.announce and r["v"] == 0 and r["p"] <= bm.TARGET)
        g[f"world {s}: lead; slip; overloads; begun, C6 and C11"] = (
            f"{w.lead}; {w.slip}; {' '.join(map(str, m6['overloads']))}; {m6['begun']}, "
            f"{m11['begun']}")
        g[f"world {s}: C6 at rest in the work zone at [m], from period, to period; C11 at rest "
          f"at [m] from period"] = f"{m4(p)}, {a}, {b}; {m4(r11['p'])}, {r11['k']}"
    facts["A6, short lead times"] = g

    # A6: world 16070 -------------------------------------------------------------
    w = worlds[16070]
    m6, per6 = R[("C6", 16070)]
    g = {}
    g["the announcement; the forecast onset; the true onset; the lead time"] = (
        f"{w.announce}; {w.onset_forecast}; {w.onset}; {w.lead}")
    after = [r["onset"] for r in per6 if w.onset_forecast - 1 <= r["k"] < w.onset]
    g["C6: the requirement's onset pair from the period before the forecast onset to the "
      "period before the true onset"] = ",".join(sorted({str(x) for x in after}))
    spells = rest_spells(per6, w.onset_forecast, m6["begun"])
    a, b, p = spells[0]
    g["C6: at rest in the work zone at [m], from period, to period"] = f"{m4(p)}, {a}, {b}"
    g["C6: the level of the plan in force in the true onset's period [m]; overloads"] = (
        f"{float(per6[w.onset]['level']):.2f}; {' '.join(map(str, m6['overloads']))}")
    g["the operation begun, C6; C11; C13; C14"] = "; ".join(
        str(R[(c, 16070)][0]["begun"]) for c in ("C6", "C11", "C13", "C14"))
    g["periods in the work zone, C6; C11"] = (f"{m6['in_work_zone']}; "
                                              f"{R[('C11', 16070)][0]['in_work_zone']}")
    facts["A6, world 16070"] = g

    # A6: the planner under ALLOVI ------------------------------------------------
    two = [s for s in valid
           if R[("C14", s)][0]["in_work_zone"] - R[("C11", s)][0]["in_work_zone"] == 2]
    g = {"valid worlds in which C14 spends two periods longer in the work zone than C11":
         f"{len(two)}: " + ", ".join(map(str, two))}
    mod = []
    for s in valid:
        per = R[("C14", s)][1]
        mod.append(sum(1 for r in per if r.get("proposal") == bm.RETREAT
                       and r.get("ambition") is not None and r["ambition"] < 1))
    g["valid worlds: periods in which the planner proposes 'retreat' and the governor "
      "commits an ambition below 1 (C14): value (worlds)"] = counts(mod)
    g["in those worlds, the period in which the cart first moves backwards after the "
      "announcement, C14 less C11: value (worlds)"] = counts(
        first_back(R[("C14", s)][1], worlds[s].announce)
        - first_back(R[("C11", s)][1], worlds[s].announce) for s in two)

    def stops(c, s):
        """Periods, after the first backward motion and before the operation,
        at whose start the cart is at rest beyond 23.7 m."""
        per, k0 = R[(c, s)][1], first_back(R[(c, s)][1], worlds[s].announce)
        end = R[(c, s)][0]["begun"]
        return sum(1 for r in per if k0 < r["k"] < end and r["v"] == 0 and r["p"] > bm.TARGET)
    g["in those worlds, periods of rest beyond 23.7 m after the first backward motion, C14; "
      "C11: value (worlds)"] = (f"{counts(stops('C14', s) for s in two)}; "
                                f"{counts(stops('C11', s) for s in two)}")
    facts["A6, the planner under ALLOVI"] = g

    # the worlds that are not valid ---------------------------------------------
    g = {}
    for s in invalid:
        w = worlds[s]
        rows = [(c, R[(c, s)][0]) for c in SURVEY]
        bad = [(c, m) for c, m in rows if m["invalid"]]
        g[f"world {s} (survey error {float(w.survey_error):+.2f} m): survey configurations "
          f"with an invalid executed label; with an alarm; restarted; the least true "
          f"clearance among them [m]"] = (
            f"{len(bad)} of {len(rows)}; {sum(1 for _, m in bad if m['alarms'])}; "
            f"{sum(1 for _, m in bad if m['restarts'])}; "
            + (f"{float(min(m['least_clearance'] for _, m in bad)):.4f}" if bad else "-"))
    silent = [(c, s) for c in SURVEY for s in invalid
              if R[(c, s)][0]["invalid"] and not R[(c, s)][0]["alarms"]]
    g["runs with an invalid executed label and no alarm"] = ", ".join(
        f"{c} world {s}" for c, s in silent) or "none"
    for c, s in silent:
        m, per = R[(c, s)]
        r = next(r for r in per if r.get("head") is not None and r["head"][6])
        lo, hi, u, beta, p1, v1, _ = r["head"]
        nxt = per[r["k"] + 1]
        g[f"{c} world {s}: the period; the segment's positions [m]; the true ramp's start [m]; "
          f"the command and the label [m/s^2]; the end of its funnel and the measured state"] = (
            f"{r['k']}; {m4(lo)} to {m4(hi)}; {float(worlds[s].ramp):.2f}; {float(u):g}, "
            f"{float(beta):g}; ({m4(p1)}, {m4(v1)}) and ({m4(nxt['p'])}, {m4(nxt['v'])})")
    facts["the worlds that are not valid"] = g

    # the substitution grid ------------------------------------------------------
    gspecs = [(p, m, s) for p in PACKAGES for m in ROUNDING for s in EVAL_WORLDS]
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            gout = pool.map(grid_record, gspecs, chunksize=4)
    else:
        gout = [grid_record(x) for x in gspecs]
    g = {}
    for p in PACKAGES:
        b = next(x["b_low"] for x in gout if x["package"] == p)
        on_grid = Fr(round(b * 1000), 1000)
        g[f"{p}: the declared braking on the ramp; rounded to 1/1000 [m/s^2]"] = (
            f"{float(b):g}; {float(on_grid):g}")
        for m in ROUNDING:
            xs = [x for x in gout if x["package"] == p and x["method"] == m]
            mean_load = sum((x["load"] for x in xs), Fr(0)) / len(xs)
            g[f"{p}, {m}: periods before arrival; asking for more braking than declared on "
              f"the ramp; of these, committed in full; mean intervention"] = (
                f"{sum(x['n'] for x in xs)}; {sum(x['beyond'] for x in xs)}; "
                f"{sum(x['full'] for x in xs)}; {float(mean_load):.3f}")
    cells = {}
    with open(os.path.join(HERE, "results", "substitution-evaluation.csv"),
              encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            cells.setdefault((row["package"], row["policy"]), []).append(
                (Fr(row["arrival_s"]), Fr(row["mean_intervention"])))
    for key, val in without(cells, "ccm").items():
        g[f"without ccm (an exclusion after the run), {key}"] = val
    facts["the substitution grid"] = g

    lines, failed = [], []
    for group, items in facts.items():
        lines.append(f"[{group}]")
        for key, val in items.items():
            lines.append(f"  {key} = {val}")
            want = EXPECTED.get(group, {}).get(key)
            if args.check and want is not None and want != str(val):
                failed.append(f"{group}.{key}: expected {want!r}, got {val!r}")
    if args.check:
        failed += [f"{group}.{key}: not computed" for group, items in EXPECTED.items()
                   for key in items if key not in facts.get(group, {})]
        n = sum(len(v) for v in EXPECTED.values())
        lines.append(f"[check] {n} values: " +
                     ("all as expected" if not failed and n else "FAILED: " + "; ".join(failed)))
    out_dir = os.path.join(HERE, "results")
    with open(os.path.join(out_dir, "diagnosis-summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if failed:
        raise SystemExit(1)


EXPECTED = {   # every value of the summary but the versions of the libraries
 'A5, world 16063': {'the announcement; the forecast onset; the true onset; the onset after the first postponement': '198; '
                                                                                                                     '218; '
                                                                                                                     '213; '
                                                                                                                     '218',
                     'the cart at rest at the announcement, C6; C7; C8 [m]': '24.4800; 24.4950; '
                                                                             '24.4950',
                     "the level of the plan in force in the postponed onset's period, C6; C7; C8 [m]": '1.31; '
                                                                                                       '1.29; '
                                                                                                       '1.29',
                     'the operation begun, C6; C7; C8': '218; 239; 239'},
 "A6, the planner's timing": {'valid worlds in which C11 spends one or two periods less in the work zone than C6': '16: '
                                                                                                                   '16006, '
                                                                                                                   '16009, '
                                                                                                                   '16010, '
                                                                                                                   '16011, '
                                                                                                                   '16012, '
                                                                                                                   '16017, '
                                                                                                                   '16022, '
                                                                                                                   '16023, '
                                                                                                                   '16026, '
                                                                                                                   '16031, '
                                                                                                                   '16047, '
                                                                                                                   '16066, '
                                                                                                                   '16078, '
                                                                                                                   '16079, '
                                                                                                                   '16092, '
                                                                                                                   '16095',
                              'the period in which the cart first moves backwards after the announcement, C6 less C11: value (worlds)': '1 '
                                                                                                                                        '(13), '
                                                                                                                                        '2 '
                                                                                                                                        '(3)',
                              'the cart at rest at the start of the period in which the operation begins, C11; C6 (worlds)': '16; '
                                                                                                                             '1',
                              "C6's front at or before 23.7 m at the start of that period (worlds)": '16'},
 'A6, short lead times': {'valid worlds with a lead time of 18 periods or fewer in which C6 spends three or more periods longer in the work zone and begins the operation with C11': '16002, '
                                                                                                                                                                                     '16020, '
                                                                                                                                                                                     '16083',
                          'world 16002: lead; slip; overloads; begun, C6 and C11': '14; 0; 204; '
                                                                                   '225, 225',
                          'world 16002: C6 at rest in the work zone at [m], from period, to period; C11 at rest at [m] from period': '24.0075, '
                                                                                                                                     '207, '
                                                                                                                                     '213; '
                                                                                                                                     '23.6950, '
                                                                                                                                     '211',
                          'world 16020: lead; slip; overloads; begun, C6 and C11': '14; -7; 195 '
                                                                                   '202; 223, 223',
                          'world 16020: C6 at rest in the work zone at [m], from period, to period; C11 at rest at [m] from period': '24.0087, '
                                                                                                                                     '205, '
                                                                                                                                     '211; '
                                                                                                                                     '23.6962, '
                                                                                                                                     '209',
                          'world 16083: lead; slip; overloads; begun, C6 and C11': '13; 0; 205; '
                                                                                   '226, 226',
                          'world 16083: C6 at rest in the work zone at [m], from period, to period; C11 at rest at [m] from period': '24.0000, '
                                                                                                                                     '208, '
                                                                                                                                     '214; '
                                                                                                                                     '23.6500, '
                                                                                                                                     '213'},
 'A6, world 16070': {'the announcement; the forecast onset; the true onset; the lead time': '191; '
                                                                                            '202; '
                                                                                            '212; '
                                                                                            '11',
                     "C6: the requirement's onset pair from the period before the forecast onset to the period before the true onset": '0',
                     'C6: at rest in the work zone at [m], from period, to period': '24.2000, 205, '
                                                                                    '218',
                     "C6: the level of the plan in force in the true onset's period [m]; overloads": '0.80; '
                                                                                                     '212',
                     'the operation begun, C6; C11; C13; C14': '233; 212; 212; 212',
                     'periods in the work zone, C6; C11': '141; 121'},
 'A6, the planner under ALLOVI': {'valid worlds in which C14 spends two periods longer in the work zone than C11': '8: '
                                                                                                                   '16009, '
                                                                                                                   '16010, '
                                                                                                                   '16023, '
                                                                                                                   '16026, '
                                                                                                                   '16031, '
                                                                                                                   '16079, '
                                                                                                                   '16092, '
                                                                                                                   '16095',
                                  "valid worlds: periods in which the planner proposes 'retreat' and the governor commits an ambition below 1 (C14): value (worlds)": '9 '
                                                                                                                                                                      '(1), '
                                                                                                                                                                      '11 '
                                                                                                                                                                      '(1), '
                                                                                                                                                                      '12 '
                                                                                                                                                                      '(2), '
                                                                                                                                                                      '13 '
                                                                                                                                                                      '(2), '
                                                                                                                                                                      '14 '
                                                                                                                                                                      '(3), '
                                                                                                                                                                      '15 '
                                                                                                                                                                      '(77), '
                                                                                                                                                                      '16 '
                                                                                                                                                                      '(8)',
                                  'in those worlds, the period in which the cart first moves backwards after the announcement, C14 less C11: value (worlds)': '0 '
                                                                                                                                                              '(8)',
                                  'in those worlds, periods of rest beyond 23.7 m after the first backward motion, C14; C11: value (worlds)': '1 '
                                                                                                                                              '(8); '
                                                                                                                                              '0 '
                                                                                                                                              '(8)'},
 'the worlds that are not valid': {'world 16042 (survey error +0.42 m): survey configurations with an invalid executed label; with an alarm; restarted; the least true clearance among them [m]': '8 '
                                                                                                                                                                                                  'of '
                                                                                                                                                                                                  '10; '
                                                                                                                                                                                                  '8; '
                                                                                                                                                                                                  '8; '
                                                                                                                                                                                                  '0.5000',
                                   'world 16043 (survey error +0.50 m): survey configurations with an invalid executed label; with an alarm; restarted; the least true clearance among them [m]': '1 '
                                                                                                                                                                                                  'of '
                                                                                                                                                                                                  '10; '
                                                                                                                                                                                                  '1; '
                                                                                                                                                                                                  '1; '
                                                                                                                                                                                                  '1.2500',
                                   'world 16059 (survey error +0.50 m): survey configurations with an invalid executed label; with an alarm; restarted; the least true clearance among them [m]': '8 '
                                                                                                                                                                                                  'of '
                                                                                                                                                                                                  '10; '
                                                                                                                                                                                                  '8; '
                                                                                                                                                                                                  '8; '
                                                                                                                                                                                                  '0.5281',
                                   'world 16064 (survey error +0.44 m): survey configurations with an invalid executed label; with an alarm; restarted; the least true clearance among them [m]': '9 '
                                                                                                                                                                                                  'of '
                                                                                                                                                                                                  '10; '
                                                                                                                                                                                                  '8; '
                                                                                                                                                                                                  '8; '
                                                                                                                                                                                                  '0.5000',
                                   'world 16077 (survey error +0.54 m): survey configurations with an invalid executed label; with an alarm; restarted; the least true clearance among them [m]': '10 '
                                                                                                                                                                                                  'of '
                                                                                                                                                                                                  '10; '
                                                                                                                                                                                                  '10; '
                                                                                                                                                                                                  '2; '
                                                                                                                                                                                                  '0.4500',
                                   'world 16100 (survey error +0.49 m): survey configurations with an invalid executed label; with an alarm; restarted; the least true clearance among them [m]': '1 '
                                                                                                                                                                                                  'of '
                                                                                                                                                                                                  '10; '
                                                                                                                                                                                                  '1; '
                                                                                                                                                                                                  '1; '
                                                                                                                                                                                                  '1.2000',
                                   'runs with an invalid executed label and no alarm': 'C15 world '
                                                                                       '16064',
                                   "C15 world 16064: the period; the segment's positions [m]; the true ramp's start [m]; the command and the label [m/s^2]; the end of its funnel and the measured state": '56; '
                                                                                                                                                                                                           '19.6900 '
                                                                                                                                                                                                           'to '
                                                                                                                                                                                                           '19.9000; '
                                                                                                                                                                                                           '19.90; '
                                                                                                                                                                                                           '-2, '
                                                                                                                                                                                                           '2; '
                                                                                                                                                                                                           '(19.9000, '
                                                                                                                                                                                                           '2.0000) '
                                                                                                                                                                                                           'and '
                                                                                                                                                                                                           '(19.9000, '
                                                                                                                                                                                                           '2.0000)'},
 'the substitution grid': {'tube: the declared braking on the ramp; rounded to 1/1000 [m/s^2]': '0.375; '
                                                                                                '0.375',
                           'tube, braking curve: periods before arrival; asking for more braking than declared on the ramp; of these, committed in full; mean intervention': '1911; '
                                                                                                                                                                             '0; '
                                                                                                                                                                             '0; '
                                                                                                                                                                             '0.246',
                           'tube, expert planner: periods before arrival; asking for more braking than declared on the ramp; of these, committed in full; mean intervention': '1932; '
                                                                                                                                                                              '0; '
                                                                                                                                                                              '0; '
                                                                                                                                                                              '0.203',
                           'qp: the declared braking on the ramp; rounded to 1/1000 [m/s^2]': '0.385; '
                                                                                              '0.385',
                           'qp, braking curve: periods before arrival; asking for more braking than declared on the ramp; of these, committed in full; mean intervention': '1898; '
                                                                                                                                                                           '0; '
                                                                                                                                                                           '0; '
                                                                                                                                                                           '0.246',
                           'qp, expert planner: periods before arrival; asking for more braking than declared on the ramp; of these, committed in full; mean intervention': '1917; '
                                                                                                                                                                            '0; '
                                                                                                                                                                            '0; '
                                                                                                                                                                            '0.199',
                           'ccm: the declared braking on the ramp; rounded to 1/1000 [m/s^2]': '0.39875; '
                                                                                               '0.399',
                           'ccm, braking curve: periods before arrival; asking for more braking than declared on the ramp; of these, committed in full; mean intervention': '1876; '
                                                                                                                                                                            '755; '
                                                                                                                                                                            '0; '
                                                                                                                                                                            '0.648',
                           'ccm, expert planner: periods before arrival; asking for more braking than declared on the ramp; of these, committed in full; mean intervention': '1921; '
                                                                                                                                                                             '792; '
                                                                                                                                                                             '0; '
                                                                                                                                                                             '0.588',
                           'mpc: the declared braking on the ramp; rounded to 1/1000 [m/s^2]': '0.39875; '
                                                                                               '0.399',
                           'mpc, braking curve: periods before arrival; asking for more braking than declared on the ramp; of these, committed in full; mean intervention': '1848; '
                                                                                                                                                                            '729; '
                                                                                                                                                                            '586; '
                                                                                                                                                                            '0.265',
                           'mpc, expert planner: periods before arrival; asking for more braking than declared on the ramp; of these, committed in full; mean intervention': '1891; '
                                                                                                                                                                             '764; '
                                                                                                                                                                             '689; '
                                                                                                                                                                             '0.209',
                           "without ccm (an exclusion after the run), B2: the ranges of the packages' and of the methods' mean arrival [s]": '0.251, '
                                                                                                                                             '0.247',
                           'without ccm (an exclusion after the run), B3: the packages from earliest to latest, by method': 'rule: '
                                                                                                                            'mpc '
                                                                                                                            '< '
                                                                                                                            'qp '
                                                                                                                            '< '
                                                                                                                            'tube; '
                                                                                                                            'braking '
                                                                                                                            'curve: '
                                                                                                                            'mpc '
                                                                                                                            '< '
                                                                                                                            'qp '
                                                                                                                            '< '
                                                                                                                            'tube; '
                                                                                                                            'expert '
                                                                                                                            'planner: '
                                                                                                                            'mpc '
                                                                                                                            '< '
                                                                                                                            'qp '
                                                                                                                            '< '
                                                                                                                            'tube; '
                                                                                                                            'off-policy '
                                                                                                                            'learner: '
                                                                                                                            'mpc '
                                                                                                                            '< '
                                                                                                                            'qp '
                                                                                                                            '< '
                                                                                                                            'tube; '
                                                                                                                            'planner: '
                                                                                                                            'mpc '
                                                                                                                            '< '
                                                                                                                            'qp '
                                                                                                                            '< '
                                                                                                                            'tube; '
                                                                                                                            'chunk '
                                                                                                                            'imitator: '
                                                                                                                            'mpc '
                                                                                                                            '< '
                                                                                                                            'qp '
                                                                                                                            '< '
                                                                                                                            'tube',
                           "without ccm (an exclusion after the run), B4, reference: the ranges of the methods' and of the packages' mean intervention": '0.609, '
                                                                                                                                                         '0.004',
                           "without ccm (an exclusion after the run), B4, sequence: the ranges of the methods' and of the packages' mean intervention": '0.066, '
                                                                                                                                                        '0.005'}}


if __name__ == "__main__":
    main()
