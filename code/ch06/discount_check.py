#!/usr/bin/env python3
"""
The off-policy learner of Chapter 6 with SAC's discount factor: a check of
the comparison of Section 6.2.4 that the protocol did not register.

Physical AI, Together, Chapter 6 (Section 6.2.4).

SAC, as Stable-Baselines3 sets it by default, discounts with 0.99; the
tabular stand-in of Section 6.2 with 0.5. The comparison of the two in
Section 6.2.4 therefore changes two things at once: the algorithm and the
discount. This script trains and evaluates the stand-in exactly as
cart_learning.py does (tube package, seeds 1, 2 and 3, 40 training
episodes in the worlds 100 s + i, evaluation in the worlds 1000 to 1019),
once with the discount of the book, 0.5, which must reproduce the stand-in's
rows of results/evaluation.csv run for run, and once with 0.99, every other
setting unchanged. It then compares SAC (published/results/sac/, budgets M
and L) with the stand-in at both discounts, world by world, with the sign
test of published/report.py; and, once SAC has been run with the stand-in's
discount (published/train_sac.py --gamma 0.5, which writes
published/results/sac-gamma0.5/), that SAC with the stand-in at 0.5 and with
the rule, the braking curve and the expert planner. These
comparisons were chosen after SAC's results were known: they are
exploratory and uncorrected, and they do not replace the registered
comparisons of published/PROTOCOL.md.

Writes results/discount-check.txt and results/discount-check.csv (one row
per evaluation run of the stand-in at each discount).

Usage
    python3 discount_check.py

Python 3.8 or later and NumPy (for the import of published/report.py); no
other package. About three minutes with two processes.
"""
import csv
import os
import sys
from multiprocessing import Pool

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "published"))
import cart_learning as cl    # noqa: E402
import report as pr           # noqa: E402  (published/report.py: per_world, sign_test)
import protocol               # noqa: E402

GAMMAS = (0.5, 0.99)
RESULTS = os.path.join(HERE, "results")


def job(spec):
    gamma, seed = spec
    cl.use_package("tube")
    log = []
    pm = cl.train(cl.OffPolicyLearner(seed, gamma=gamma),
                  cl.training_worlds(seed, cl.TRAIN_EPISODES), log)
    ev = cl.evaluate(pm, cl.EVAL_SEEDS)
    return gamma, seed, ev, log


def rows_of(runs):
    """Evaluation runs as rows of results/evaluation.csv."""
    return [dict(world=str(e["world"]), arrival_s="" if e["arrival"] is None else repr(e["arrival"]),
                 mean_intervention=f"{e['load']:.4f}", seed=str(e["seed"])) for e in runs]


def main():
    with Pool(2) as pool:
        out = pool.map(job, [(g, s) for g in GAMMAS for s in cl.SEEDS])
    runs = {g: [] for g in GAMMAS}
    for gamma, seed, ev, log in out:
        for e in ev:
            e["seed"] = seed
        runs[gamma].extend(ev)

    # 1. the discount of the book reproduces results/evaluation.csv, run for run
    book = {}
    with open(os.path.join(RESULTS, "evaluation.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            if r["policy"] == "off-policy learner" and r["package"] == "tube":
                book[(r["seed"], r["world"])] = (r["arrival_s"] or "None", r["mean_intervention"])
    mine = {(str(e["seed"]), str(e["world"])): (str(e["arrival"]), f"{e['load']:.4f}")
            for e in runs[0.5]}
    reproduced = mine == book

    lines = ["The off-policy learner of Chapter 6 with SAC's discount factor (discount_check.py)",
             f"  discount 0.5 reproduces the 60 evaluation runs of results/evaluation.csv: "
             f"{'yes' if reproduced else 'NO'}"]
    for g in GAMMAS:
        d = cl.describe(runs[g])
        lines.append(f"  discount {g}: " + "; ".join(
            f"{k} {d[k]}" for k in ("arrived", "arrival", "arrival_range", "mean_intervention",
                                    "alarms", "steps_outside_funnel", "passed_unchanged",
                                    "proposals_at_largest_acceleration")))

    # 2. SAC against the stand-in, world by world (exploratory): SAC's registered
    #    runs (discount 0.99) against the stand-in at both discounts, and SAC run
    #    with the stand-in's discount (train_sac.py --gamma 0.5), if its three
    #    seeds are in, against the stand-in at 0.5
    for folder, label, gammas in (("sac", "SAC (discount 0.99)", GAMMAS),
                                  ("sac-gamma0.5", "SAC (discount 0.5)", (0.5,))):
        data, _, seeds_in = pr.load_method(protocol.RESULTS, folder, ("M", "L"))
        if sorted(seeds_in) != sorted(protocol.SEEDS):
            lines.append(f"  {label}: seeds {seeds_in or 'none'} found in published/results/"
                         f"{folder}; its comparisons need seeds 1, 2 and 3")
            continue
        for g in gammas:
            rows = rows_of(runs[g])
            sw = (pr.per_world(rows, "arrival"), pr.per_world(rows, "intervention"))
            for b in ("M", "L"):
                srows = pr.method_rows(data[b][0])
                sac = (pr.per_world(srows, "arrival"), pr.per_world(srows, "intervention"))
                for i, what, words in ((0, "arrival", ("earlier", "later")),
                                       (1, "intervention", ("lower", "higher"))):
                    wins, losses, ties, p = pr.sign_test(sac[i], sw[i])
                    mean = pr.mean_of if i == 0 else pr.mean_of_load
                    lines.append(
                        f"  {label}, {b} against the stand-in at discount {g}, {what}: "
                        f"{words[0]} in {wins} of the worlds, {words[1]} in {losses}, the same "
                        f"in {ties}; mean over the worlds {mean(sac[i])} against {mean(sw[i])}; "
                        f"sign test p = {pr.fmt_p(p)} (exploratory, uncorrected)")

    # 3. SAC with the stand-in's discount against the baselines (exploratory), with the
    #    secondary comparisons that PROTOCOL.md registers for SAC: arrival against the
    #    rule, the braking curve and the expert planner, intervention against the rule
    #    and the braking curve
    data, _, seeds_in = pr.load_method(protocol.RESULTS, "sac-gamma0.5", ("M", "L"))
    if sorted(seeds_in) == sorted(protocol.SEEDS):
        for b in ("M", "L"):
            srows = pr.method_rows(data[b][0])
            sac = (pr.per_world(srows, "arrival"), pr.per_world(srows, "intervention"))
            for base, quantities in (("rule", (0, 1)), ("braking curve", (0, 1)),
                                     ("expert planner", (0,))):
                brows = pr.standin_runs(base)
                bw = (pr.per_world(brows, "arrival"), pr.per_world(brows, "intervention"))
                for i in quantities:
                    what, words = (("arrival", ("earlier", "later")) if i == 0
                                   else ("intervention", ("lower", "higher")))
                    wins, losses, ties, p = pr.sign_test(sac[i], bw[i])
                    mean = pr.mean_of if i == 0 else pr.mean_of_load
                    lines.append(
                        f"  SAC (discount 0.5), {b} against the {base}, {what}: {words[0]} in "
                        f"{wins} of the worlds, {words[1]} in {losses}, the same in {ties}; mean "
                        f"over the worlds {mean(sac[i])} against {mean(bw[i])}; sign test p = "
                        f"{pr.fmt_p(p)}; decision {pr.verdict(wins, losses, p, words)} "
                        f"(exploratory, uncorrected)")
    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(RESULTS, "discount-check.txt"), "w") as fh:
        fh.write(text)
    with open(os.path.join(RESULTS, "discount-check.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["discount", "seed", "world", "arrival_s", "mean_intervention", "fallbacks",
                    "alarms", "steps_outside", "max_position", "rest_position"])
        for g in GAMMAS:
            for e in runs[g]:
                w.writerow([g, e["seed"], e["world"], e["arrival"], f"{e['load']:.4f}",
                            e["fallbacks"], e["alarms"], e["out"], f"{e['pmax']:.4f}",
                            "none" if e["rest"] is None else f"{e['rest']:.4f}"])


if __name__ == "__main__":
    main()
