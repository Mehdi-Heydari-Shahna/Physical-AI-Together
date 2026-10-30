#!/usr/bin/env python3
"""
The results of the second experiment, PROTOCOL-R2.md: SAC and TD-MPC2
trained with the time reward.

Physical AI, Together, Chapter 6, Section 6.2.4.

Reads results/sac-time/ and results/tdmpc2-time/ (written by train_sac.py
and train_tdmpc2.py with --reward time), the runs of PROTOCOL.md in
results/sac/ and results/tdmpc2/, and the stand-ins' runs in
../results/evaluation.csv, and writes results/summary-r2.txt:
    [table]        per method and budget, the columns of Table 6.3, for the
                   runs of PROTOCOL-R2.md and, beside them, of PROTOCOL.md
    [<method>]     the group of each method and budget of PROTOCOL-R2.md
    [H1]           the guarantee during training and evaluation
    [primary]      Q1 to Q4: each method and budget with the time reward
                   against the same method and budget under PROTOCOL.md, in
                   arrival; sign test over the 20 evaluation worlds, Holm's
                   correction over the family of four
    [secondary]    the other comparisons of PROTOCOL-R2.md, uncorrected
    [record]       versions, commits, the reward and the discount, hardware
                   and wall-clock times
It uses the functions of report.py (version 2), so the unit, the sign test,
Holm's correction with bounds while tests are missing, and the rule that a
method is reported only when all three seeds are in, are those of
PROTOCOL.md. It never changes results/summary.txt.

Usage
    python report_r2.py            # results/summary-r2.txt
    python report_r2.py --smoke    # results/smoke/summary-r2.txt, from the --smoke runs
"""
import argparse
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import protocol    # noqa: E402
import report      # noqa: E402

R2 = [("sac-time", "SAC, R2", ("M", "L")), ("tdmpc2-time", "TD-MPC2, R2", ("M", "L"))]
R1 = [("sac", "SAC", ("M", "L")), ("tdmpc2", "TD-MPC2", ("M", "L"))]
PRIMARY = [("SAC, R2, M", "SAC, M", "arrival"), ("SAC, R2, L", "SAC, L", "arrival"),
           ("TD-MPC2, R2, M", "TD-MPC2, M", "arrival"),
           ("TD-MPC2, R2, L", "TD-MPC2, L", "arrival")]
FAMILY = len(PRIMARY)


def secondary_specs():
    specs = []
    for m in ("SAC, R2", "TD-MPC2, R2"):
        r1 = m.split(",")[0]
        for b in ("M", "L"):
            key = f"{m}, {b}"
            stand = "off-policy learner" if m.startswith("SAC") else "planner"
            specs.append((key, stand, "arrival"))
            specs.append((key, "off-policy learner", "intervention"))
            for base in ("rule", "braking curve", "expert planner"):
                specs.append((key, base, "arrival"))
            for base in ("rule", "braking curve"):
                specs.append((key, base, "intervention"))
            specs.append((key, f"{r1}, {b}", "intervention"))
    for b in ("M", "L"):
        specs += [(f"TD-MPC2, R2, {b}", f"SAC, R2, {b}", "arrival"),
                  (f"TD-MPC2, R2, {b}", f"SAC, R2, {b}", "intervention")]
    for m in ("SAC, R2", "TD-MPC2, R2"):
        specs += [(f"{m}, L", f"{m}, M", "arrival"), (f"{m}, L", f"{m}, M", "intervention")]
    return specs


def main():
    ap = argparse.ArgumentParser(description="Results of PROTOCOL-R2.md")
    ap.add_argument("--smoke", action="store_true", help="the --smoke runs, results/smoke/")
    args = ap.parse_args()
    root = os.path.join(protocol.RESULTS, "smoke") if args.smoke else protocol.RESULTS
    required = (1,) if args.smoke else tuple(protocol.SEEDS)
    header = ["Results of the second experiment, PROTOCOL-R2.md (report_r2.py): SAC and TD-MPC2 "
              "with the time reward; the results of PROTOCOL.md stand unchanged"
              + ("; SMOKE RUNS, not results" if args.smoke else "")]

    data, records, status = {}, {}, {}
    for folder, label, budgets in R2 + R1:
        d, recs, seeds_in = report.load_method(root, folder, budgets)
        if not seeds_in:
            status[label] = "no results"
        elif set(required) <= set(seeds_in):
            status[label] = "complete, seeds " + ",".join(map(str, seeds_in))
        else:
            missing = [s for s in required if s not in seeds_in]
            status[label] = (f"incomplete: seeds {','.join(map(str, seeds_in))} in, "
                             f"{','.join(map(str, missing))} not; not reported, and its "
                             f"comparisons withheld")
            d = {b: ([], []) for b in budgets}
        data[label], records[label] = d, recs

    table, groups, pub = {}, {}, {}
    for folder, label, budgets in R2 + R1:
        for b in budgets:
            evals, trains = data[label][b]
            key = f"{label}, {b}"
            if not evals:
                table[key] = {"status": status[label]}
                continue
            g = protocol.summarize(evals, trains)
            g["seeds"] = ",".join(str(ev[0]["seed"]) for ev in evals)
            if (folder, label, budgets) in R2:
                groups[key] = g
            table[key] = {c: g[c] for c in ("arrived", "arrival", "arrival_range",
                                            "mean_intervention", "largest_position_all")}
            table[key]["status"] = status[label]
            rows = report.method_rows(evals)
            pub[key] = (report.per_world(rows, "arrival"), report.per_world(rows, "intervention"),
                        "reference")

    h1 = {}
    all_runs, all_eps = [], []
    for folder, label, budgets in R2:
        runs = [m for b in budgets for ev in data[label][b][0] for m in ev]
        eps = [m for t in data[label][budgets[-1]][1] for m in t]
        if not runs:
            h1[label] = status[label]
            continue
        g = protocol.guarantee_check(runs, eps)
        h1[label] = "; ".join(f"{k} {v}" for k, v in g.items())
        all_runs += runs
        all_eps += eps
    if all_runs:
        g = protocol.guarantee_check(all_runs, all_eps)
        h1["both methods of PROTOCOL-R2.md"] = "; ".join(f"{k} {v}" for k, v in g.items())

    sw = {n: (report.per_world(report.standin_runs(n), "arrival"),
              report.per_world(report.standin_runs(n), "intervention"), kind)
          for n, kind in report.STANDINS.items()}

    def get(key):
        return pub.get(key) or sw.get(key)

    def available(spec):
        return get(spec[0]) is not None and get(spec[1]) is not None

    def compare(a_key, b_key, what):
        a, b = get(a_key), get(b_key)
        i = 0 if what == "arrival" else 1
        if what == "intervention" and a[2] != b[2]:
            raise AssertionError(f"interventions of {a_key} and {b_key} do not compare")
        wins, losses, ties, p = report.sign_test(a[i], b[i])
        mean = report.mean_of if what == "arrival" else report.mean_of_load
        return dict(a=a_key, b=b_key, what=what, wins=wins, losses=losses, ties=ties, p=p,
                    mean_a=mean(a[i]), mean_b=mean(b[i]))

    def words(c):
        return ("earlier", "later") if c["what"] == "arrival" else ("lower", "higher")

    def head(c):
        w = words(c)
        return (f"{c['a']} against {c['b']}, {c['what']}: {w[0]} in {c['wins']} of the "
                f"worlds, {w[1]} in {c['losses']}, the same in {c['ties']}; mean over the "
                f"worlds {c['mean_a']} against {c['mean_b']}")

    def not_available(spec):
        missing = [k for k in spec[:2] if get(k) is None]
        why = "; ".join(f"{k}: {status.get(k.rsplit(', ', 1)[0], 'no results')}" for k in missing)
        return f"{spec[0]} against {spec[1]}, {spec[2]}: not available ({why})"

    primary = [compare(*s) if available(s) else None for s in PRIMARY]
    run = [c for c in primary if c is not None]
    upper, lower = report.holm_bounds([c["p"] for c in run], FAMILY) if run else ([], [])
    complete = len(run) == FAMILY
    lines_p, j = [], 0
    for s, c in zip(PRIMARY, primary):
        if c is None:
            lines_p.append(not_available(s))
            continue
        u, lo = upper[j], lower[j]
        j += 1
        if complete:
            holm_txt = f"Holm-adjusted over the family of {FAMILY}: {report.fmt_p(u)}"
        else:
            holm_txt = (f"Holm-adjusted over the family of {FAMILY}, with {FAMILY - len(run)} "
                        f"not run: between {report.fmt_p(lo)} and {report.fmt_p(u)}")
        lines_p.append(f"{head(c)}; sign test p = {report.fmt_p(c['p'])}; {holm_txt}; decision: "
                       f"{report.primary_decision(c['wins'], c['losses'], u, lo, words(c), complete)}")
    lines_s = []
    for s in secondary_specs():
        if not available(s):
            lines_s.append(not_available(s))
            continue
        c = compare(*s)
        lines_s.append(f"{head(c)}; sign test p = {report.fmt_p(c['p'])} (uncorrected); decision: "
                       f"{report.verdict(c['wins'], c['losses'], c['p'], words(c))}")

    rec = {}
    for folder, label, budgets in R2:
        rs = records[label]
        if not rs:
            rec[label] = status[label]
            continue
        env = rs[0].get("environment", {})
        gpu = env.get("gpu", {})
        rec[label] = "; ".join([
            f"status {status[label]}",
            f"seeds {','.join(str(r['seed']) for r in rs)}",
            "reward " + ", ".join(str(r.get("reward")) for r in rs),
            "discount " + ", ".join(str(r.get("discount", r.get("hyperparameters", {})
                                               .get("gamma"))) for r in rs),
            f"python {env.get('python', '?').split()[0]}",
            "packages " + ", ".join(f"{k} {v}" for k, v in env.get("packages", {}).items()),
            f"commit {rs[0].get('checkout', {}).get('commit', 'not applicable')}",
            f"gpu {gpu.get('gpus', gpu.get('nvidia_smi', 'none'))}",
            f"cpu {env.get('cpu', '?')}",
            "environment steps " + ", ".join(f"seed {r['seed']}: {r.get('environment_steps')}"
                                             for r in rs),
            "wall clock " + ", ".join(f"seed {r['seed']}: {r.get('wall_clock_s')}" for r in rs),
            "inference " + ", ".join(f"seed {r['seed']}: {r.get('inference_latency')}" for r in rs)])
        bad = [r["seed"] for r in rs if r.get("reward") != "time"]
        if bad:
            rec[label] += f"; WARNING: seeds {bad} were not trained with the time reward"

    out = {"table": {k: "; ".join(f"{c} {v}" for c, v in d.items()) for k, d in table.items()}}
    out.update(groups)
    out["H1"] = h1
    out["primary"] = {f"Q{i + 1}": s for i, s in enumerate(lines_p)}
    out["secondary"] = {f"T{i + 1}": s for i, s in enumerate(lines_s)}
    out["record"] = rec
    os.makedirs(root, exist_ok=True)
    lines = protocol.write_summary(os.path.join(root, "summary-r2.txt"), out, header)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
