#!/usr/bin/env python3
"""
The substitution grid: four control packages crossed with six proposal
methods on the task of Chapter 6.

Physical AI, Together, Chapter 16 (Section 16.2).

The control packages of Section 5.3 (tube, qp, ccm, mpc) are crossed with
the proposal methods of Chapter 6: the rule of Chapters 3 and 5, the
braking curve, the expert planner, the off-policy learner (the stand-in for
SAC), the planner with a learned model (for TD-MPC2) and the chunk
imitator (for a diffusion policy). The governor, the forecasting predictor,
the world family, the training protocol and the evaluation worlds are those
of Chapter 6, imported from ../ch06/cart_learning.py and used unchanged.
Each learner is trained with the package it is evaluated with.

What the grid adds is the mpc package's adapters. Chapter 6 gave the tube
package, and with it qp and ccm, which share its segments, generators for
a reference proposal and for a sequence; the mpc package of Section 5.3
offered only the proposals of Chapter 5, 'accelerate' and 'hold'. Its
adapters below are those of Chapter 6 with the mpc package's segments: the
head starts at the centre predicted from the measured state, its tube
grows by the recursion of Section 5.3, and the reference's bound is the
package's, 2 m/s^2 less four times the tube's next scale. Nothing else
changes.

This script
  * checks that the grid's runs with the tube package, and the off-policy
    learner's with ccm, are those of Chapter 6, run for run
    (../ch06/results/evaluation.csv);
  * runs every pairing, training each learner with its package, and
    applies the decision rules of hypotheses B1 to B4 of HYPOTHESES.md;
  * reports every cell: arrivals, the mean intervention, fallbacks, alarms,
    steps outside a funnel, the largest position, and its training runs.

Usage
    python3 cart_substitution.py            # writes results/substitution-*
    python3 cart_substitution.py --check    # also checks every value of the
                                            # summary
    python3 cart_substitution.py --jobs 1   # one process (default: up to two)

Outputs, in results/ next to this file
    substitution-summary.txt      every value, by group; with --check, the result
    substitution-evaluation.csv   one row per evaluation run, as Chapter 6's
    substitution-cells.csv        one row per pairing: the mean arrival, its
                                  range and the mean intervention (Figure 16.1)

Python 3.8 or later; no other packages. About forty minutes with one process;
less with two. The results do not depend on the number of processes.
"""
import argparse
import csv
import os
import sys
import time
from fractions import Fraction as Fr
from multiprocessing import Pool

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "ch06"))
import cart_learning as cl                                       # noqa: E402

cf, cm = cl.cf, cl.cm
T, U_MAX = cf.T, cf.U_MAX
PACKAGES = ("tube", "qp", "ccm", "mpc")
NON_LEARNERS = ("rule", "braking curve", "expert planner")
LEARNERS = ("off-policy learner", "planner", "chunk imitator")
METHODS = NON_LEARNERS + LEARNERS
REFERENCE = ("rule", "braking curve", "off-policy learner")      # one reference per period
SEQUENCE = ("expert planner", "planner", "chunk imitator")         # eight references
CH06 = os.path.join(HERE, "..", "ch06", "results", "evaluation.csv")
ORIGINAL = {name: getattr(cl, name) for name in
            ("generate_reference", "generate_prefixes", "initial_set", "realizable")}


# ----------------------------------------------------------------------------
# The mpc package's adapters for reference and sequence proposals
# ----------------------------------------------------------------------------
def _scale(init):
    """The predicted centre and the tube's scale of an initial set of the mpc
    package (Section 5.3)."""
    base = init[1] if init[0] == "cap" else init
    if base[0] == "ell4":
        return base[1], base[2], base[3]
    return cm.box_scale(base)


def _budget(s):
    """The head's input budget for a tube of scale s: 4 s' 81/80, s' the
    next scale."""
    return 4 * cm.scale_next(s) * cm.MARGIN


def _clip(a, s):
    lim = U_MAX - _budget(s)
    return max(-lim, min(lim, a))


def generate_reference_mpc(pred, k, init, q, fallback):
    """As cart_learning.generate_reference, with the mpc package's segments."""
    q = None if q is None else cl.exact(q)
    Plan = cf.Plan
    cands = []
    if init[0] == "rest":
        stay = Plan(k + 1, [], cm.rest_containing_m(init))
        if q is not None and q > 0 and init[1] == init[2]:
            p0, a_l = init[1], min(q, U_MAX)
            for a in cf.ALPHAS[:-1]:
                head = cf.Launch(p0, a * a_l)
                pairs = [(head, pred.label(k, k + 1, head))]
                _, _, s0 = cm.box_scale(head.end())
                tail, rest = cm.recovery_m(pred, k, k + 2, head.p1, head.v1, s0)
                if not tail:
                    rest = cm.rest_containing_m(head.end())
                cands.append((Plan(k + 1, pairs + tail, rest), a))
            cands.append((stay, Fr(0)))
        else:
            cands.append((stay, Fr(1) if (q is not None and q <= 0) else Fr(0)))
        return cands
    pc, vc, s0 = _scale(init)
    if q is not None:
        a_q = _clip(q, s0)
        if a_q < 0:
            a_q = -min(-a_q, (vc - cm.CCM_V_H) / T) if vc > cm.CCM_V_H else None
        fb_head = fallback.head()
        a_fb = fb_head.a if fb_head.kind == "track" else Fr(0)
        if a_q is None:
            cands.append((Plan(k + 1, [], cm.rest_containing_m(init)), Fr(1)))
        else:
            for alpha in cf.ALPHAS:
                head = cm.TrackS(pc, vc, alpha * a_q + (1 - alpha) * a_fb, s0)
                pairs = [(head, pred.label(k, k + 1, head))]
                tail, rest = cm.recovery_m(pred, k, k + 2, head.p1, head.v1, head.s1)
                if rest is None:
                    rest = cm.rest_containing_m(head.end())
                cands.append((Plan(k + 1, pairs + tail, rest), alpha))
    tail, rest = cm.recovery_m(pred, k, k + 1, pc, vc, s0)
    if rest is None:
        rest = cm.rest_containing_m(init)
    cands.append((Plan(k + 1, tail, rest), Fr(0)))
    return cands


def follow_mpc(pred, k, j, p, v, s, seq):
    """Tracking pairs of the mpc package that follow the references seq from
    (p, v) with tube scale s, the first in period j."""
    pairs = []
    for a in seq:
        a = _clip(a, s)
        if a < 0:
            if v <= cm.CCM_V_H:
                break
            a = -min(-a, (v - cm.CCM_V_H) / T)
        seg = cm.TrackS(p, v, a, s)
        pairs.append((seg, pred.label(k, j, seg)))
        p, v, s, j = seg.p1, seg.v1, seg.s1, j + 1
    return pairs, p, v, s, j


def generate_prefixes_mpc(pred, k, init, seq, fallback):
    """As cart_learning.generate_prefixes, with the mpc package's segments."""
    if seq is not None:
        seq = [cl.exact(a) for a in seq]
        if not seq or any(a is None for a in seq):
            seq = None
    Plan = cf.Plan
    cands = []
    if init[0] == "rest":
        stay = Plan(k + 1, [], cm.rest_containing_m(init))
        if seq is not None and seq[0] > 0 and init[1] == init[2]:
            p0 = init[1]
            for m, amb in cl.PREFIX:
                head = cf.Launch(p0, min(seq[0], U_MAX))
                pairs = [(head, pred.label(k, k + 1, head))]
                _, _, s0 = cm.box_scale(head.end())
                more, p, v, s, j = follow_mpc(pred, k, k + 2, head.p1, head.v1, s0, seq[1:m])
                pairs += more
                tail, rest = cm.recovery_m(pred, k, j, p, v, s)
                if rest is None:
                    rest = cm.rest_containing_m(pairs[-1][0].end())
                cands.append((Plan(k + 1, pairs + tail, rest), amb))
            cands.append((stay, Fr(0)))
        else:
            cands.append((stay, Fr(1) if (seq is not None and seq[0] <= 0) else Fr(0)))
        return cands
    pc, vc, s0 = _scale(init)
    if seq is not None:
        if seq[0] <= 0 and vc <= cm.CCM_V_H:
            cands.append((Plan(k + 1, [], cm.rest_containing_m(init)), Fr(1)))
        else:
            for m, amb in cl.PREFIX:
                pairs, p, v, s, j = follow_mpc(pred, k, k + 1, pc, vc, s0, seq[:m])
                tail, rest = cm.recovery_m(pred, k, j, p, v, s)
                if rest is None:
                    rest = cm.rest_containing_m(pairs[-1][0].end() if pairs else init)
                cands.append((Plan(k + 1, pairs + tail, rest), amb))
    tail, rest = cm.recovery_m(pred, k, k + 1, pc, vc, s0)
    if rest is None:
        rest = cm.rest_containing_m(init)
    cands.append((Plan(k + 1, tail, rest), Fr(0)))
    return cands


def initial_set_mpc(seg_ex, p, v):
    """The initial set of the mpc package (Section 5.3), with the standstill
    restart of Chapter 6."""
    if seg_ex.kind == "rest" and v == 0.0 and seg_ex.p_lo <= Fr(p) <= seg_ex.p_hi:
        x = Fr(p)
        return ("rest", x, x)
    return cm.initial_set_m(seg_ex, p, v)


def realizable_mpc(q, init):
    """As cart_learning.realizable, with the mpc package's bound."""
    if q is None:
        return False
    q = cl.exact(q)
    if q is None:
        return False
    if init[0] == "rest":
        return q <= 0 or (init[1] == init[2] and q <= U_MAX)
    _, vc, s0 = _scale(init)
    if _clip(q, s0) != q:
        return False
    return q >= 0 or (vc > cm.CCM_V_H and -q <= (vc - cm.CCM_V_H) / T)


MPC_ADAPTERS = dict(generate_reference=generate_reference_mpc,
                    generate_prefixes=generate_prefixes_mpc,
                    initial_set=initial_set_mpc, realizable=realizable_mpc)


def use_package(name):
    """Installs a control package of Section 5.3, and for mpc its adapters,
    and reads its declared constants (as cart_learning.use_package)."""
    cm.install({"tube": {}, "qp": cm.QP, "ccm": cm.CCM, "mpc": cm.MPC}[name])
    for key, fn in (MPC_ADAPTERS if name == "mpc" else ORIGINAL).items():
        setattr(cl, key, fn)
    cl.DECLARED = cl.declared()


cl.use_package = use_package          # job() calls it before and after each unit


def job(spec):
    return cl.job(spec)


# ----------------------------------------------------------------------------
# Reporting and the hypotheses
# ----------------------------------------------------------------------------
def mean(xs):
    return sum(xs, Fr(0)) / len(xs)


def r(x, nd):
    return cf.r(x, nd)


def cells(results):
    """Per pairing: the evaluation runs, pooled over the seeds."""
    out = {}
    for res in results:
        out.setdefault((res["package"], res["name"]), []).extend(res["eval"])
    return out


def cell_stats(evals):
    arr = [Fr(e["periods"], 10) for e in evals if e["periods"] is not None]
    return dict(n=len(evals), arrived=len(arr), arrival=mean(arr) if arr else None,
                earliest=min(arr) if arr else None, latest=max(arr) if arr else None,
                load=mean([e["load_exact"] for e in evals]),
                fallbacks=mean([Fr(e["fallbacks"]) for e in evals]),
                alarms=sum(e["alarms"] for e in evals), out=sum(e["out"] for e in evals),
                pmax=max(e["pmax"] for e in evals))


def b1(results):
    runs = exc = 0
    bad = []
    for res in results:
        logs = [x for x in res["log"] if "alarms" in x]
        for e in res["eval"] + logs:
            runs += 1
            fault = e["alarms"] or e["out"] or e["pmax"] >= float(cf.WALL - cf.EPS_MIN)
            arrived_not_rest = (e in res["eval"] and e["periods"] is not None
                                and e["rest"] is None)
            if fault or arrived_not_rest:
                exc += 1
                bad.append(f"{res['package']} {res['name']} seed {res['seed']}")
    return dict(supported=exc == 0, cases=runs, exceptions=bad)


def b2(st):
    pk = {p: mean([st[(p, m)]["arrival"] for m in METHODS]) for p in PACKAGES}
    me = {m: mean([st[(p, m)]["arrival"] for p in PACKAGES]) for m in METHODS}
    rp, rm_ = max(pk.values()) - min(pk.values()), max(me.values()) - min(me.values())
    return dict(supported=rp > rm_, packages=pk, methods=me, range_packages=rp,
                range_methods=rm_)


def b3(st):
    orders = {}
    for m in METHODS:
        orders[m] = sorted(PACKAGES, key=lambda p: st[(p, m)]["arrival"])
    agree = True
    for i, p in enumerate(PACKAGES):
        for q in PACKAGES[i + 1:]:
            signs = set()
            for m in METHODS:
                d = st[(p, m)]["arrival"] - st[(q, m)]["arrival"]
                if d != 0:
                    signs.add(d > 0)
            if len(signs) > 1:
                agree = False
    return dict(supported=agree, orders=orders)


def b4(st):
    out, ok = {}, True
    for group, methods in (("reference", REFERENCE), ("sequence", SEQUENCE)):
        me = {m: mean([st[(p, m)]["load"] for p in PACKAGES]) for m in methods}
        pk = {p: mean([st[(p, m)]["load"] for m in methods]) for p in PACKAGES}
        rme, rpk = max(me.values()) - min(me.values()), max(pk.values()) - min(pk.values())
        ok = ok and rme > rpk
        out[group] = (me, pk, rme, rpk)
    return dict(supported=ok, groups=out)


def consistency(results):
    """The grid's runs that Chapter 6 also made, compared run for run."""
    ours = {}
    for res in results:
        for e in res["eval"]:
            ours[(res["name"], res["package"], str(res["seed"]), str(e["world"]))] = (
                str(e["arrival"]), f"{e['load']:.4f}", str(e["fallbacks"]), str(e["alarms"]),
                str(e["out"]), f"{e['pmax']:.4f}",
                "none" if e["rest"] is None else f"{e['rest']:.4f}")
    same = compared = 0
    with open(CH06, newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row["policy"], row["package"], row["seed"], row["world"])
            if key not in ours:
                continue
            compared += 1
            theirs = (row["arrival_s"], row["mean_intervention"], row["fallbacks"],
                      row["alarms"], row["steps_outside"], row["max_position"],
                      row["rest_position"])
            same += ours[key] == theirs
    return compared, same


def main():
    ap = argparse.ArgumentParser(description="Chapter 16: the substitution grid")
    ap.add_argument("--check", action="store_true", help="check every value of the summary")
    ap.add_argument("--jobs", type=int, default=2, help="processes (default 2)")
    args = ap.parse_args()
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    cfg = dict(eval=cl.EVAL_SEEDS, train=cl.TRAIN_EPISODES, demos=cl.DEMOS)
    specs = [(m, 0, p, cfg) for p in PACKAGES for m in NON_LEARNERS]
    specs += [(m, s, p, cfg) for p in PACKAGES for m in LEARNERS for s in cl.SEEDS]
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            results = pool.map(job, specs, chunksize=1)
    else:
        results = [job(s) for s in specs]
    st = {key: cell_stats(ev) for key, ev in cells(results).items()}
    facts = {}
    compared, same = consistency(results)
    facts["consistency with Chapter 6"] = {
        "evaluation runs that Chapter 6 also made; the same, run for run": f"{compared}; {same}"}
    facts["the pairings"] = {}
    for p in PACKAGES:
        for m in METHODS:
            c = st[(p, m)]
            facts["the pairings"][f"{p}, {m}: runs arrived; mean arrival [s]; earliest, latest; "
                                  f"mean intervention; fallbacks per run; alarms; steps outside "
                                  f"a funnel; largest position [m]"] = (
                f"{c['arrived']} of {c['n']}; {r(c['arrival'], 2)}; {r(c['earliest'], 1)}, "
                f"{r(c['latest'], 1)}; {r(c['load'], 3)}; {r(c['fallbacks'], 2)}; {c['alarms']}; "
                f"{c['out']}; {c['pmax']:.3f}")
    train = {}
    for res in results:
        for x in res["log"]:
            if "alarms" in x:
                t = train.setdefault((res["package"], res["name"]), [0, 0, 0, 0])
                t[0] += 1
                t[1] += x["alarms"]
                t[2] += x["out"]
                t[3] += x["periods"] is not None
    facts["training"] = {f"{p}, {m}: episodes; alarms; steps outside a funnel; arrived": (
        f"{v[0]}; {v[1]}; {v[2]}; {v[3]}") for (p, m), v in sorted(train.items())}
    h1, h2, h3, h4 = b1(results), b2(st), b3(st), b4(st)
    facts["the hypotheses"] = {
        "B1 (the certificate in every pairing): supported; runs; exceptions":
            f"{'yes' if h1['supported'] else 'no'}; {h1['cases']}; "
            f"{', '.join(h1['exceptions'][:6]) or 'none'}",
        "B2 (the package sets the arrival): supported; the packages' means [s]; the methods' "
        "means [s]; their ranges [s]":
            (f"{'yes' if h2['supported'] else 'no'}; "
             + ", ".join(f"{p} {r(x, 3)}" for p, x in h2["packages"].items()) + "; "
             + ", ".join(f"{m} {r(x, 3)}" for m, x in h2["methods"].items()) + "; "
             f"{r(h2['range_packages'], 3)}, {r(h2['range_methods'], 3)}"),
        "B3 (no partnership): supported; the packages from earliest to latest, by method":
            f"{'yes' if h3['supported'] else 'no'}; " + "; ".join(
                f"{m}: {' < '.join(o)}" for m, o in h3["orders"].items()),
        "B4 (interventions belong to the proposal method): supported; by group, the methods' "
        "and the packages' ranges of the mean intervention":
            f"{'yes' if h4['supported'] else 'no'}; " + "; ".join(
                f"{g}: methods {r(v[2], 3)}, packages {r(v[3], 3)}"
                for g, v in h4["groups"].items()),
    }
    with open(os.path.join(out_dir, "substitution-evaluation.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["policy", "package", "seed", "world", "arrival_s", "mean_intervention",
                    "fallbacks", "alarms", "steps_outside", "max_position", "rest_position"])
        for res in results:
            for e in res["eval"]:
                w.writerow([res["name"], res["package"], res["seed"], e["world"], e["arrival"],
                            f"{e['load']:.4f}", e["fallbacks"], e["alarms"], e["out"],
                            f"{e['pmax']:.4f}",
                            "none" if e["rest"] is None else f"{e['rest']:.4f}"])
    with open(os.path.join(out_dir, "substitution-cells.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["package", "method", "index_package", "index_method", "arrival",
                    "earliest", "latest", "intervention"])
        for i, p in enumerate(PACKAGES):
            for j, m in enumerate(METHODS):
                c = st[(p, m)]
                w.writerow([p, m, i, j, f"{float(c['arrival']):.3f}",
                            f"{float(c['earliest']):.1f}", f"{float(c['latest']):.1f}",
                            f"{float(c['load']):.3f}"])
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
        lines.append(f"[check] {n} values, every value of the summary: " +
                     ("all as expected" if not failed and n else "FAILED: " + "; ".join(failed)))
    lines.append(f"[time] {time.time() - t0:.0f} s")
    with open(os.path.join(out_dir, "substitution-summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines[:-1]) + "\n")
    print("\n".join(lines))
    if failed:
        raise SystemExit(1)


EXPECTED = {   # every value of the summary but the versions of the libraries
 'consistency with Chapter 6': {'evaluation runs that Chapter 6 also made; the same, run for run': '300; '
                                                                                                   '300'},
 'the pairings': {'tube, rule: runs arrived; mean arrival [s]; earliest, latest; mean intervention; fallbacks per run; alarms; steps outside a funnel; largest position [m]': '20 '
                                                                                                                                                                              'of '
                                                                                                                                                                              '20; '
                                                                                                                                                                              '9.55; '
                                                                                                                                                                              '9.1, '
                                                                                                                                                                              '10.0; '
                                                                                                                                                                              '0.656; '
                                                                                                                                                                              '0.00; '
                                                                                                                                                                              '0; '
                                                                                                                                                                              '0; '
                                                                                                                                                                              '24.474',
                  'tube, braking curve: runs arrived; mean arrival [s]; earliest, latest; mean intervention; fallbacks per run; alarms; steps outside a funnel; largest position [m]': '20 '
                                                                                                                                                                                       'of '
                                                                                                                                                                                       '20; '
                                                                                                                                                                                       '9.56; '
                                                                                                                                                                                       '9.0, '
                                                                                                                                                                                       '10.0; '
                                                                                                                                                                                       '0.246; '
                                                                                                                                                                                       '0.00; '
                                                                                                                                                                                       '0; '
                                                                                                                                                                                       '0; '
                                                                                                                                                                                       '24.473',
                  'tube, expert planner: runs arrived; mean arrival [s]; earliest, latest; mean intervention; fallbacks per run; alarms; steps outside a funnel; largest position [m]': '20 '
                                                                                                                                                                                        'of '
                                                                                                                                                                                        '20; '
                                                                                                                                                                                        '9.66; '
                                                                                                                                                                                        '9.2, '
                                                                                                                                                                                        '10.1; '
                                                                                                                                                                                        '0.203; '
                                                                                                                                                                                        '0.00; '
                                                                                                                                                                                        '0; '
                                                                                                                                                                                        '0; '
                                                                                                                                                                                        '24.471',
                  'tube, off-policy learner: runs arrived; mean arrival [s]; earliest, latest; mean intervention; fallbacks per run; alarms; steps outside a funnel; largest position [m]': '60 '
                                                                                                                                                                                            'of '
                                                                                                                                                                                            '60; '
                                                                                                                                                                                            '9.65; '
                                                                                                                                                                                            '9.1, '
                                                                                                                                                                                            '10.2; '
                                                                                                                                                                                            '0.041; '
                                                                                                                                                                                            '0.00; '
                                                                                                                                                                                            '0; '
                                                                                                                                                                                            '0; '
                                                                                                                                                                                            '24.474',
                  'tube, planner: runs arrived; mean arrival [s]; earliest, latest; mean intervention; fallbacks per run; alarms; steps outside a funnel; largest position [m]': '60 '
                                                                                                                                                                                 'of '
                                                                                                                                                                                 '60; '
                                                                                                                                                                                 '9.72; '
                                                                                                                                                                                 '9.1, '
                                                                                                                                                                                 '10.3; '
                                                                                                                                                                                 '0.138; '
                                                                                                                                                                                 '0.00; '
                                                                                                                                                                                 '0; '
                                                                                                                                                                                 '0; '
                                                                                                                                                                                 '24.472',
                  'tube, chunk imitator: runs arrived; mean arrival [s]; earliest, latest; mean intervention; fallbacks per run; alarms; steps outside a funnel; largest position [m]': '60 '
                                                                                                                                                                                        'of '
                                                                                                                                                                                        '60; '
                                                                                                                                                                                        '9.75; '
                                                                                                                                                                                        '9.1, '
                                                                                                                                                                                        '10.3; '
                                                                                                                                                                                        '0.139; '
                                                                                                                                                                                        '0.00; '
                                                                                                                                                                                        '0; '
                                                                                                                                                                                        '0; '
                                                                                                                                                                                        '24.475',
                  'qp, rule: runs arrived; mean arrival [s]; earliest, latest; mean intervention; fallbacks per run; alarms; steps outside a funnel; largest position [m]': '20 '
                                                                                                                                                                            'of '
                                                                                                                                                                            '20; '
                                                                                                                                                                            '9.45; '
                                                                                                                                                                            '9.0, '
                                                                                                                                                                            '9.9; '
                                                                                                                                                                            '0.653; '
                                                                                                                                                                            '0.00; '
                                                                                                                                                                            '0; '
                                                                                                                                                                            '0; '
                                                                                                                                                                            '24.474',
                  'qp, braking curve: runs arrived; mean arrival [s]; earliest, latest; mean intervention; fallbacks per run; alarms; steps outside a funnel; largest position [m]': '20 '
                                                                                                                                                                                     'of '
                                                                                                                                                                                     '20; '
                                                                                                                                                                                     '9.49; '
                                                                                                                                                                                     '9.0, '
                                                                                                                                                                                     '9.9; '
                                                                                                                                                                                     '0.246; '
                                                                                                                                                                                     '0.00; '
                                                                                                                                                                                     '0; '
                                                                                                                                                                                     '0; '
                                                                                                                                                                                     '24.451',
                  'qp, expert planner: runs arrived; mean arrival [s]; earliest, latest; mean intervention; fallbacks per run; alarms; steps outside a funnel; largest position [m]': '20 '
                                                                                                                                                                                      'of '
                                                                                                                                                                                      '20; '
                                                                                                                                                                                      '9.59; '
                                                                                                                                                                                      '9.1, '
                                                                                                                                                                                      '10.0; '
                                                                                                                                                                                      '0.199; '
                                                                                                                                                                                      '0.00; '
                                                                                                                                                                                      '0; '
                                                                                                                                                                                      '0; '
                                                                                                                                                                                      '24.474',
                  'qp, off-policy learner: runs arrived; mean arrival [s]; earliest, latest; mean intervention; fallbacks per run; alarms; steps outside a funnel; largest position [m]': '60 '
                                                                                                                                                                                          'of '
                                                                                                                                                                                          '60; '
                                                                                                                                                                                          '9.54; '
                                                                                                                                                                                          '9.0, '
                                                                                                                                                                                          '10.1; '
                                                                                                                                                                                          '0.041; '
                                                                                                                                                                                          '0.00; '
                                                                                                                                                                                          '0; '
                                                                                                                                                                                          '0; '
                                                                                                                                                                                          '24.475',
                  'qp, planner: runs arrived; mean arrival [s]; earliest, latest; mean intervention; fallbacks per run; alarms; steps outside a funnel; largest position [m]': '60 '
                                                                                                                                                                               'of '
                                                                                                                                                                               '60; '
                                                                                                                                                                               '9.64; '
                                                                                                                                                                               '9.1, '
                                                                                                                                                                               '10.2; '
                                                                                                                                                                               '0.130; '
                                                                                                                                                                               '0.00; '
                                                                                                                                                                               '0; '
                                                                                                                                                                               '0; '
                                                                                                                                                                               '24.474',
                  'qp, chunk imitator: runs arrived; mean arrival [s]; earliest, latest; mean intervention; fallbacks per run; alarms; steps outside a funnel; largest position [m]': '60 '
                                                                                                                                                                                      'of '
                                                                                                                                                                                      '60; '
                                                                                                                                                                                      '9.70; '
                                                                                                                                                                                      '9.2, '
                                                                                                                                                                                      '10.1; '
                                                                                                                                                                                      '0.143; '
                                                                                                                                                                                      '0.00; '
                                                                                                                                                                                      '0; '
                                                                                                                                                                                      '0; '
                                                                                                                                                                                      '24.476',
                  'ccm, rule: runs arrived; mean arrival [s]; earliest, latest; mean intervention; fallbacks per run; alarms; steps outside a funnel; largest position [m]': '20 '
                                                                                                                                                                             'of '
                                                                                                                                                                             '20; '
                                                                                                                                                                             '9.33; '
                                                                                                                                                                             '8.9, '
                                                                                                                                                                             '9.7; '
                                                                                                                                                                             '0.650; '
                                                                                                                                                                             '0.00; '
                                                                                                                                                                             '0; '
                                                                                                                                                                             '0; '
                                                                                                                                                                             '24.482',
                  'ccm, braking curve: runs arrived; mean arrival [s]; earliest, latest; mean intervention; fallbacks per run; alarms; steps outside a funnel; largest position [m]': '20 '
                                                                                                                                                                                      'of '
                                                                                                                                                                                      '20; '
                                                                                                                                                                                      '9.38; '
                                                                                                                                                                                      '8.8, '
                                                                                                                                                                                      '9.7; '
                                                                                                                                                                                      '0.648; '
                                                                                                                                                                                      '0.00; '
                                                                                                                                                                                      '0; '
                                                                                                                                                                                      '0; '
                                                                                                                                                                                      '24.480',
                  'ccm, expert planner: runs arrived; mean arrival [s]; earliest, latest; mean intervention; fallbacks per run; alarms; steps outside a funnel; largest position [m]': '20 '
                                                                                                                                                                                       'of '
                                                                                                                                                                                       '20; '
                                                                                                                                                                                       '9.61; '
                                                                                                                                                                                       '9.1, '
                                                                                                                                                                                       '10.0; '
                                                                                                                                                                                       '0.588; '
                                                                                                                                                                                       '0.00; '
                                                                                                                                                                                       '0; '
                                                                                                                                                                                       '0; '
                                                                                                                                                                                       '24.441',
                  'ccm, off-policy learner: runs arrived; mean arrival [s]; earliest, latest; mean intervention; fallbacks per run; alarms; steps outside a funnel; largest position [m]': '60 '
                                                                                                                                                                                           'of '
                                                                                                                                                                                           '60; '
                                                                                                                                                                                           '9.40; '
                                                                                                                                                                                           '8.9, '
                                                                                                                                                                                           '9.9; '
                                                                                                                                                                                           '0.076; '
                                                                                                                                                                                           '0.00; '
                                                                                                                                                                                           '0; '
                                                                                                                                                                                           '0; '
                                                                                                                                                                                           '24.483',
                  'ccm, planner: runs arrived; mean arrival [s]; earliest, latest; mean intervention; fallbacks per run; alarms; steps outside a funnel; largest position [m]': '60 '
                                                                                                                                                                                'of '
                                                                                                                                                                                '60; '
                                                                                                                                                                                '9.85; '
                                                                                                                                                                                '9.1, '
                                                                                                                                                                                '10.5; '
                                                                                                                                                                                '0.492; '
                                                                                                                                                                                '0.00; '
                                                                                                                                                                                '0; '
                                                                                                                                                                                '0; '
                                                                                                                                                                                '24.443',
                  'ccm, chunk imitator: runs arrived; mean arrival [s]; earliest, latest; mean intervention; fallbacks per run; alarms; steps outside a funnel; largest position [m]': '60 '
                                                                                                                                                                                       'of '
                                                                                                                                                                                       '60; '
                                                                                                                                                                                       '9.64; '
                                                                                                                                                                                       '8.9, '
                                                                                                                                                                                       '10.1; '
                                                                                                                                                                                       '0.544; '
                                                                                                                                                                                       '0.00; '
                                                                                                                                                                                       '0; '
                                                                                                                                                                                       '0; '
                                                                                                                                                                                       '24.483',
                  'mpc, rule: runs arrived; mean arrival [s]; earliest, latest; mean intervention; fallbacks per run; alarms; steps outside a funnel; largest position [m]': '20 '
                                                                                                                                                                             'of '
                                                                                                                                                                             '20; '
                                                                                                                                                                             '9.25; '
                                                                                                                                                                             '8.8, '
                                                                                                                                                                             '9.7; '
                                                                                                                                                                             '0.644; '
                                                                                                                                                                             '0.00; '
                                                                                                                                                                             '0; '
                                                                                                                                                                             '0; '
                                                                                                                                                                             '24.473',
                  'mpc, braking curve: runs arrived; mean arrival [s]; earliest, latest; mean intervention; fallbacks per run; alarms; steps outside a funnel; largest position [m]': '20 '
                                                                                                                                                                                      'of '
                                                                                                                                                                                      '20; '
                                                                                                                                                                                      '9.24; '
                                                                                                                                                                                      '8.8, '
                                                                                                                                                                                      '9.6; '
                                                                                                                                                                                      '0.265; '
                                                                                                                                                                                      '0.00; '
                                                                                                                                                                                      '0; '
                                                                                                                                                                                      '0; '
                                                                                                                                                                                      '24.494',
                  'mpc, expert planner: runs arrived; mean arrival [s]; earliest, latest; mean intervention; fallbacks per run; alarms; steps outside a funnel; largest position [m]': '20 '
                                                                                                                                                                                       'of '
                                                                                                                                                                                       '20; '
                                                                                                                                                                                       '9.46; '
                                                                                                                                                                                       '8.9, '
                                                                                                                                                                                       '9.9; '
                                                                                                                                                                                       '0.209; '
                                                                                                                                                                                       '0.00; '
                                                                                                                                                                                       '0; '
                                                                                                                                                                                       '0; '
                                                                                                                                                                                       '24.494',
                  'mpc, off-policy learner: runs arrived; mean arrival [s]; earliest, latest; mean intervention; fallbacks per run; alarms; steps outside a funnel; largest position [m]': '60 '
                                                                                                                                                                                           'of '
                                                                                                                                                                                           '60; '
                                                                                                                                                                                           '9.34; '
                                                                                                                                                                                           '8.8, '
                                                                                                                                                                                           '9.9; '
                                                                                                                                                                                           '0.042; '
                                                                                                                                                                                           '0.00; '
                                                                                                                                                                                           '0; '
                                                                                                                                                                                           '0; '
                                                                                                                                                                                           '24.494',
                  'mpc, planner: runs arrived; mean arrival [s]; earliest, latest; mean intervention; fallbacks per run; alarms; steps outside a funnel; largest position [m]': '60 '
                                                                                                                                                                                'of '
                                                                                                                                                                                '60; '
                                                                                                                                                                                '9.55; '
                                                                                                                                                                                '9.1, '
                                                                                                                                                                                '10.1; '
                                                                                                                                                                                '0.145; '
                                                                                                                                                                                '0.00; '
                                                                                                                                                                                '0; '
                                                                                                                                                                                '0; '
                                                                                                                                                                                '24.494',
                  'mpc, chunk imitator: runs arrived; mean arrival [s]; earliest, latest; mean intervention; fallbacks per run; alarms; steps outside a funnel; largest position [m]': '60 '
                                                                                                                                                                                       'of '
                                                                                                                                                                                       '60; '
                                                                                                                                                                                       '9.54; '
                                                                                                                                                                                       '9.0, '
                                                                                                                                                                                       '10.0; '
                                                                                                                                                                                       '0.134; '
                                                                                                                                                                                       '0.00; '
                                                                                                                                                                                       '0; '
                                                                                                                                                                                       '0; '
                                                                                                                                                                                       '24.494'},
 'training': {'ccm, chunk imitator: episodes; alarms; steps outside a funnel; arrived': '30; 0; 0; '
                                                                                        '30',
              'ccm, off-policy learner: episodes; alarms; steps outside a funnel; arrived': '120; '
                                                                                            '0; 0; '
                                                                                            '120',
              'ccm, planner: episodes; alarms; steps outside a funnel; arrived': '120; 0; 0; 120',
              'mpc, chunk imitator: episodes; alarms; steps outside a funnel; arrived': '30; 0; 0; '
                                                                                        '30',
              'mpc, off-policy learner: episodes; alarms; steps outside a funnel; arrived': '120; '
                                                                                            '0; 0; '
                                                                                            '120',
              'mpc, planner: episodes; alarms; steps outside a funnel; arrived': '120; 0; 0; 120',
              'qp, chunk imitator: episodes; alarms; steps outside a funnel; arrived': '30; 0; 0; '
                                                                                       '30',
              'qp, off-policy learner: episodes; alarms; steps outside a funnel; arrived': '120; '
                                                                                           '0; 0; '
                                                                                           '120',
              'qp, planner: episodes; alarms; steps outside a funnel; arrived': '120; 0; 0; 120',
              'tube, chunk imitator: episodes; alarms; steps outside a funnel; arrived': '30; 0; '
                                                                                         '0; 29',
              'tube, off-policy learner: episodes; alarms; steps outside a funnel; arrived': '120; '
                                                                                             '0; '
                                                                                             '0; '
                                                                                             '120',
              'tube, planner: episodes; alarms; steps outside a funnel; arrived': '120; 0; 0; 120'},
 'the hypotheses': {'B1 (the certificate in every pairing): supported; runs; exceptions': 'yes; '
                                                                                          '2040; '
                                                                                          'none',
                    "B2 (the package sets the arrival): supported; the packages' means [s]; the methods' means [s]; their ranges [s]": 'no; '
                                                                                                                                       'tube '
                                                                                                                                       '9.646, '
                                                                                                                                       'qp '
                                                                                                                                       '9.565, '
                                                                                                                                       'ccm '
                                                                                                                                       '9.534, '
                                                                                                                                       'mpc '
                                                                                                                                       '9.395; '
                                                                                                                                       'rule '
                                                                                                                                       '9.393, '
                                                                                                                                       'braking '
                                                                                                                                       'curve '
                                                                                                                                       '9.416, '
                                                                                                                                       'expert '
                                                                                                                                       'planner '
                                                                                                                                       '9.576, '
                                                                                                                                       'off-policy '
                                                                                                                                       'learner '
                                                                                                                                       '9.482, '
                                                                                                                                       'planner '
                                                                                                                                       '9.688, '
                                                                                                                                       'chunk '
                                                                                                                                       'imitator '
                                                                                                                                       '9.654; '
                                                                                                                                       '0.251, '
                                                                                                                                       '0.295',
                    'B3 (no partnership): supported; the packages from earliest to latest, by method': 'no; '
                                                                                                       'rule: '
                                                                                                       'mpc '
                                                                                                       '< '
                                                                                                       'ccm '
                                                                                                       '< '
                                                                                                       'qp '
                                                                                                       '< '
                                                                                                       'tube; '
                                                                                                       'braking '
                                                                                                       'curve: '
                                                                                                       'mpc '
                                                                                                       '< '
                                                                                                       'ccm '
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
                                                                                                       'ccm '
                                                                                                       '< '
                                                                                                       'tube; '
                                                                                                       'off-policy '
                                                                                                       'learner: '
                                                                                                       'mpc '
                                                                                                       '< '
                                                                                                       'ccm '
                                                                                                       '< '
                                                                                                       'qp '
                                                                                                       '< '
                                                                                                       'tube; '
                                                                                                       'planner: '
                                                                                                       'mpc '
                                                                                                       '< '
                                                                                                       'qp '
                                                                                                       '< '
                                                                                                       'tube '
                                                                                                       '< '
                                                                                                       'ccm; '
                                                                                                       'chunk '
                                                                                                       'imitator: '
                                                                                                       'mpc '
                                                                                                       '< '
                                                                                                       'ccm '
                                                                                                       '< '
                                                                                                       'qp '
                                                                                                       '< '
                                                                                                       'tube',
                    "B4 (interventions belong to the proposal method): supported; by group, the methods' and the packages' ranges of the mean intervention": 'no; '
                                                                                                                                                             'reference: '
                                                                                                                                                             'methods '
                                                                                                                                                             '0.601, '
                                                                                                                                                             'packages '
                                                                                                                                                             '0.144; '
                                                                                                                                                             'sequence: '
                                                                                                                                                             'methods '
                                                                                                                                                             '0.074, '
                                                                                                                                                             'packages '
                                                                                                                                                             '0.384'}}


if __name__ == "__main__":
    main()
