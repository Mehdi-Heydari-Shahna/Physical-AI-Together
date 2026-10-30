#!/usr/bin/env python3
"""
The results of the published algorithms under the protocol of PROTOCOL.md.

Physical AI, Together, Chapter 6, Section 6.2.4.

Reads results/sac/, results/tdmpc2/ and results/diffusion/ (one folder per
seed, written by train_sac.py, train_tdmpc2.py and train_diffusion.py) and
the stand-ins' runs in ../results/evaluation.csv, and writes
results/summary.txt:
    [table]        per method and budget, the columns of Table 6.4, and the
                   status of each method: complete, incomplete or no results
    [<method>]     the group of ../results/summary.txt for each method and
                   budget, with the training episodes of that budget
    [H1]           the guarantee during training and evaluation
    [primary]      the registered comparisons with the stand-ins, decided by
                   the sign test over the 20 evaluation worlds with Holm's
                   correction over the registered family of eight
    [secondary]    the other registered comparisons, uncorrected
    [interim ...]  for a method whose seeds are not all in: what its finished
                   seeds show so far, not a result under the protocol
    [record]       versions, commits, hardware and wall-clock times
The decision rules are those of PROTOCOL.md; this script applies them as
written there.

Version 2 (29 October 2026, after the SAC runs; PROTOCOL.md, Deviations):
  * Holm's correction is taken over the registered family of eight primary
    tests, as PROTOCOL.md states, also while some of them have not been
    run. Version 1 took it over the tests that had been run, which agrees
    with the protocol only once all eight are in. With some missing, each
    adjusted p-value lies between two bounds: the upper bound, Holm over
    the eight with every missing p-value set to 1, and the lower bound,
    Holm over the tests run alone. A decision is stated when the upper
    bound is below 0.05 (it then holds whatever the missing tests give),
    "no difference shown" when the lower bound is not below 0.05, and the
    decision is "pending" otherwise. With all eight in, the two bounds are
    Holm's adjusted p-value, and versions 1 and 2 decide alike.
  * A method is reported, compared and checked under H1 only when the
    folders of all the protocol's seeds (1, 2 and 3) hold their files,
    since the protocol's unit is the mean over the three trained policies
    in each world. A method with some seeds finished is reported as
    incomplete, with an [interim ...] group for the seeds that are in, and
    its comparisons are withheld. Version 1 used whatever seeds it found.
    With --smoke, the seeds that the smoke runs use (seed 1) suffice.
  * p-values are printed with three significant digits, not four decimals
    (which printed 0.0000 for p = 1.9e-06), and a mean arrival over the
    worlds that is undefined because some run did not arrive is printed
    without a unit, with the number of worlds in which every run arrived.

Usage
    python report.py            # results/summary.txt
    python report.py --pack     # and results/send-back.zip, without the policies
    python report.py --smoke    # results/smoke/summary.txt, from the --smoke runs
"""
import argparse
import csv
import json
import math
import os
import sys
import zipfile
from fractions import Fraction as Fr
from math import comb

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import protocol    # noqa: E402

cf = protocol.cf
BOOK = os.path.join(HERE, "..", "results")
METHODS = [  # (folder, label, budgets, proposal)
    ("sac", "SAC", ("M", "L"), "reference"),
    ("tdmpc2", "TD-MPC2", ("M", "L"), "reference"),
    ("diffusion", "diffusion policy", ("demonstrations",), "sequence"),
]
STANDINS = {"off-policy learner": "reference", "planner": "sequence", "chunk imitator": "sequence",
            "rule": "reference", "braking curve": "reference", "expert planner": "sequence"}
ALPHA = Fr(1, 20)            # level of each test, and of the family after Holm's correction
PRIMARY_SPECS = [("SAC, M", "off-policy learner", "arrival"),
                 ("SAC, M", "off-policy learner", "intervention"),
                 ("SAC, L", "off-policy learner", "arrival"),
                 ("SAC, L", "off-policy learner", "intervention"),
                 ("TD-MPC2, M", "planner", "arrival"),
                 ("TD-MPC2, L", "planner", "arrival"),
                 ("diffusion policy, demonstrations", "chunk imitator", "arrival"),
                 ("diffusion policy, demonstrations", "chunk imitator", "intervention")]
FAMILY = len(PRIMARY_SPECS)  # the registered family of eight


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_method(root, folder, budgets):
    """{budget: (evaluation runs per seed, training episodes per seed)}, the
    records and the seeds, for the seeds whose three files exist."""
    base = os.path.join(root, folder)
    out, records, seeds_in = {b: ([], []) for b in budgets}, [], []
    if not os.path.isdir(base):
        return out, records, seeds_in
    seeds = sorted(int(d[4:]) for d in os.listdir(base) if d.startswith("seed") and d[4:].isdigit())
    for s in seeds:
        files = [os.path.join(base, f"seed{s}", f)
                 for f in ("evaluation.csv", "training.csv", "record.json")]
        if not all(os.path.exists(f) for f in files):
            continue
        with open(files[2]) as fh:
            rec = json.load(fh)
        records.append(rec)
        seeds_in.append(s)
        ev, tr = protocol.read_runs(files[0]), protocol.read_runs(files[1])
        n = rec.get("budgets") or {}           # SAC, TD-MPC2: M is the first n["M"] episodes
        for b in budgets:
            runs = [m for m in ev if m["budget"] == b]
            if runs:
                out[b][0].append(runs)
                out[b][1].append(tr[:n[b]] if b in n else tr)
    return out, records, seeds_in


def standin_runs(name):
    """The stand-ins' evaluation runs (tube package), per world and seed."""
    rows = []
    with open(os.path.join(BOOK, "evaluation.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            if r["policy"] == name and r["package"] == "tube":
                rows.append(r)
    return rows


def per_world(rows, key):
    """The mean over the trained policies of a quantity in each world:
    arrival (infinite if a run in that world does not arrive) or the mean
    intervention, as printed with four decimals in both files."""
    by = {}
    for r in rows:
        w = int(r["world"])
        if key == "arrival":
            v = Fr(r["arrival_s"]) if r["arrival_s"] else None
        else:
            v = Fr(r["mean_intervention"])
        by.setdefault(w, []).append(v)
    return {w: (math.inf if any(v is None for v in vs) else sum(vs) / len(vs))
            for w, vs in by.items()}


def method_rows(runs_by_seed):
    rows = []
    for runs in runs_by_seed:
        for m in runs:
            rows.append(dict(world=m["world"],
                             arrival_s="" if m["arrival"] is None else repr(m["arrival"]),
                             mean_intervention=f"{m['load']:.4f}"))
    return rows


# ----------------------------------------------------------------------------
# The decision rule
# ----------------------------------------------------------------------------
def sign_test(a, b):
    """Two-sided exact sign test of 'lower in a than in b' over the worlds
    that both hold; ties are dropped. Returns wins, losses, ties and p."""
    worlds = sorted(set(a) & set(b))
    wins = sum(1 for w in worlds if a[w] < b[w])
    losses = sum(1 for w in worlds if a[w] > b[w])
    ties = len(worlds) - wins - losses
    n = wins + losses
    if n == 0:
        return wins, losses, ties, Fr(1)
    k = min(wins, losses)
    p = min(Fr(1), 2 * Fr(sum(comb(n, i) for i in range(k + 1)), 2 ** n))
    return wins, losses, ties, p


def holm(ps):
    """Holm's step-down adjustment of a list of p-values."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [None] * len(ps), Fr(0)
    for rank, i in enumerate(order):
        running = max(running, min(Fr(1), (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def holm_bounds(ps, family=FAMILY):
    """Holm's adjusted p-values, over a registered family of `family` tests,
    for the len(ps) tests that have been run: (upper, lower) bounds over every
    value that the missing tests could take. The upper bound sets every
    missing p-value to 1, which leaves the tests run at the first ranks with
    the largest multipliers; the lower bound lets every missing p-value be
    smaller than all of these, which leaves Holm over the tests run alone.
    With the family complete, both are Holm's adjusted p-values."""
    if len(ps) > family:
        raise ValueError("more tests than the registered family")
    upper = holm(list(ps) + [Fr(1)] * (family - len(ps)))[:len(ps)]
    lower = holm(list(ps))
    return upper, lower


def verdict(wins, losses, p, word=("earlier", "later")):
    if p < ALPHA and wins > losses:
        return word[0]
    if p < ALPHA and losses > wins:
        return word[1]
    return "no difference shown"


def primary_decision(wins, losses, upper, lower, word, complete):
    """The decision of a primary test, final or pending (see the docstring)."""
    if complete:
        return verdict(wins, losses, upper, word)
    if upper < ALPHA and wins != losses:
        return verdict(wins, losses, upper, word) + " (final whatever the tests not run give)"
    if lower >= ALPHA or wins == losses:
        return "no difference shown (final whatever the tests not run give)"
    return "pending: depends on the tests not run"


def fmt_p(p):
    """A p-value with three significant digits."""
    return f"{float(p):.3g}"


def mean_of(world_values):
    vals = list(world_values.values())
    if any(v == math.inf for v in vals):
        n = sum(1 for v in vals if v != math.inf)
        return f"not every run arrived (every run arrived in {n} of the {len(vals)} worlds)"
    return f"{float(sum(vals) / len(vals)):.3f} s"


def mean_of_load(world_values):
    vals = list(world_values.values())
    return f"{float(sum(vals) / len(vals)):.3f}"


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Results of the published algorithms")
    ap.add_argument("--smoke", action="store_true", help="the --smoke runs, results/smoke/")
    ap.add_argument("--pack", action="store_true",
                    help="also write send-back.zip: the text and CSV files, not the policies")
    args = ap.parse_args()
    root = os.path.join(protocol.RESULTS, "smoke") if args.smoke else protocol.RESULTS
    required = (1,) if args.smoke else tuple(protocol.SEEDS)
    groups, table, interim = {}, {}, {}
    data, records, status = {}, {}, {}
    for folder, label, budgets, kind in METHODS:
        d, recs, seeds_in = load_method(root, folder, budgets)
        if not seeds_in:
            status[label] = "no results"
        elif set(required) <= set(seeds_in):
            status[label] = "complete, seeds " + ",".join(map(str, seeds_in))
        else:
            missing = [s for s in required if s not in seeds_in]
            status[label] = (f"incomplete: seeds {','.join(map(str, seeds_in))} in, "
                             f"{','.join(map(str, missing))} not; not reported, and its "
                             f"comparisons withheld")
        data[label], records[label] = d, recs
        if status[label].startswith("incomplete"):
            for b in budgets:
                evals, trains = d[b]
                if evals:
                    g = protocol.summarize(evals, trains)
                    g["seeds"] = ",".join(str(ev[0]["seed"]) for ev in evals)
                    g["status"] = "INTERIM: not a result under PROTOCOL.md"
                    interim[f"interim {label}, {b}"] = g
            data[label] = {b: ([], []) for b in budgets}     # withheld from what follows
    header = ["Results of the published algorithms on the governed cart (report.py, version 2); "
              "protocol: PROTOCOL.md" + ("; SMOKE RUNS, not results" if args.smoke else "")]

    # summaries
    for folder, label, budgets, kind in METHODS:
        for b in budgets:
            evals, trains = data[label][b]
            key = f"{label}, {b}"
            if not evals:
                table[key] = {"status": status[label]}
                continue
            d = protocol.summarize(evals, trains)
            d["seeds"] = ",".join(str(ev[0]["seed"]) for ev in evals)
            groups[key] = d
            table[key] = {c: d[c] for c in ("arrived", "arrival", "arrival_range",
                                            "mean_intervention", "largest_position_all")}
            table[key]["status"] = status[label]

    # H1: the guarantee in every training episode and evaluation run
    h1 = {}
    all_runs, all_eps = [], []
    for folder, label, budgets, kind in METHODS:
        runs = [m for b in budgets for ev in data[label][b][0] for m in ev]
        last = budgets[-1]
        eps = [m for t in data[label][last][1] for m in t]       # the longest budget's episodes
        if not runs:
            h1[label] = status[label]
            continue
        g = protocol.guarantee_check(runs, eps)
        h1[label] = "; ".join(f"{k} {v}" for k, v in g.items())
        all_runs += runs
        all_eps += eps
    if all_runs:
        g = protocol.guarantee_check(all_runs, all_eps)
        h1["all methods reported"] = "; ".join(f"{k} {v}" for k, v in g.items())
    for key, g in interim.items():           # H1 for the seeds in, kept apart
        label = key[len("interim "):].rsplit(", ", 1)[0]
        folder, _, budgets, _ = next(m for m in METHODS if m[1] == label)
        if key.endswith(budgets[-1]):
            d, _, _ = load_method(root, folder, budgets)
            runs = [m for b in budgets for ev in d[b][0] for m in ev]
            eps = [m for t in d[budgets[-1]][1] for m in t]
            gc = protocol.guarantee_check(runs, eps)
            h1[f"{label} (INTERIM, seeds in only)"] = "; ".join(f"{k} {v}" for k, v in gc.items())

    # comparisons
    stand = {n: standin_runs(n) for n in STANDINS}
    sw = {n: (per_world(r, "arrival"), per_world(r, "intervention")) for n, r in stand.items()}
    pub = {}
    for folder, label, budgets, kind in METHODS:
        for b in budgets:
            evals = data[label][b][0]
            if evals:
                rows = method_rows(evals)
                pub[f"{label}, {b}"] = (per_world(rows, "arrival"), per_world(rows, "intervention"),
                                        kind)

    def compare(a_key, b_key, what):
        a = pub[a_key] if a_key in pub else (*sw[a_key], STANDINS[a_key])
        b = pub[b_key] if b_key in pub else (*sw[b_key], STANDINS[b_key])
        i = 0 if what == "arrival" else 1
        if what == "intervention" and a[2] != b[2]:
            raise AssertionError(f"interventions of {a_key} and {b_key} do not compare")
        wins, losses, ties, p = sign_test(a[i], b[i])
        mean = mean_of if what == "arrival" else mean_of_load
        return dict(a=a_key, b=b_key, what=what, wins=wins, losses=losses, ties=ties, p=p,
                    mean_a=mean(a[i]), mean_b=mean(b[i]))

    secondary_specs = []
    for key in ("SAC, M", "SAC, L", "TD-MPC2, M", "TD-MPC2, L",
                "diffusion policy, demonstrations"):
        for base in ("rule", "braking curve", "expert planner"):
            secondary_specs.append((key, base, "arrival"))
        if key.startswith(("SAC", "TD-MPC2")):
            for base in ("rule", "braking curve"):
                secondary_specs.append((key, base, "intervention"))
        else:
            secondary_specs.append((key, "expert planner", "intervention"))
    for b in ("M", "L"):
        secondary_specs += [(f"TD-MPC2, {b}", "off-policy learner", "intervention"),
                            (f"TD-MPC2, {b}", f"SAC, {b}", "arrival"),
                            (f"TD-MPC2, {b}", f"SAC, {b}", "intervention")]
    for m in ("SAC", "TD-MPC2"):
        secondary_specs += [(f"{m}, L", f"{m}, M", "arrival"), (f"{m}, L", f"{m}, M", "intervention")]

    def available(spec):
        return all(k in pub or k in STANDINS for k in spec[:2])

    def words(c):
        return ("earlier", "later") if c["what"] == "arrival" else ("lower", "higher")

    def head(c):
        w = words(c)
        return (f"{c['a']} against {c['b']}, {c['what']}: {w[0]} in {c['wins']} of the "
                f"worlds, {w[1]} in {c['losses']}, the same in {c['ties']}; mean over the "
                f"worlds {c['mean_a']} against {c['mean_b']}")

    def not_available(spec):
        missing = [k for k in spec[:2] if not (k in pub or k in STANDINS)]
        why = "; ".join(f"{k}: {status[k.rsplit(', ', 1)[0]]}" for k in missing)
        return f"{spec[0]} against {spec[1]}, {spec[2]}: not available ({why})"

    primary = [compare(*s) if available(s) else None for s in PRIMARY_SPECS]
    run = [c for c in primary if c is not None]
    upper, lower = holm_bounds([c["p"] for c in run]) if run else ([], [])
    complete = len(run) == FAMILY
    lines_p, j = [], 0
    for s, c in zip(PRIMARY_SPECS, primary):
        if c is None:
            lines_p.append(not_available(s))
            continue
        u, lo = upper[j], lower[j]
        j += 1
        if complete:
            holm_txt = f"Holm-adjusted over the family of {FAMILY}: {fmt_p(u)}"
        else:
            holm_txt = (f"Holm-adjusted over the family of {FAMILY}, with {FAMILY - len(run)} "
                        f"not run: between {fmt_p(lo)} and {fmt_p(u)}")
        lines_p.append(f"{head(c)}; sign test p = {fmt_p(c['p'])}; {holm_txt}; decision: "
                       f"{primary_decision(c['wins'], c['losses'], u, lo, words(c), complete)}")
    lines_s = [(f"{head(cc)}; sign test p = {fmt_p(cc['p'])} (uncorrected); decision: "
                f"{verdict(cc['wins'], cc['losses'], cc['p'], words(cc))}"
                if available(s) else not_available(s))
               for s in secondary_specs for cc in [compare(*s) if available(s) else None]]

    # the record
    rec = {}
    for label, rs in records.items():
        if not rs:
            rec[label] = status[label]
            continue
        r0 = rs[0]
        env = r0.get("environment", {})
        gpu = env.get("gpu", {})
        rec[label] = "; ".join([
            f"status {status[label]}",
            f"seeds {','.join(str(r['seed']) for r in rs)}",
            f"python {env.get('python', '?').split()[0]}",
            "packages " + ", ".join(f"{k} {v}" for k, v in env.get("packages", {}).items()),
            f"commit {r0.get('checkout', {}).get('commit', 'not applicable')}",
            f"gpu {gpu.get('gpus', gpu.get('nvidia_smi', 'none'))}",
            f"cpu {env.get('cpu', '?')}",
            "environment steps " + ", ".join(f"seed {r['seed']}: {r.get('environment_steps')}"
                                             for r in rs),
            "wall clock " + ", ".join(f"seed {r['seed']}: {r.get('wall_clock_s')}" for r in rs),
            "inference " + ", ".join(f"seed {r['seed']}: {r.get('inference_latency')}" for r in rs)])

    out_groups = {"table": {k: "; ".join(f"{c} {v}" for c, v in d.items()) for k, d in table.items()}}
    out_groups.update(groups)
    out_groups["H1"] = h1
    out_groups["primary"] = {f"P{i + 1}": s for i, s in enumerate(lines_p)}
    out_groups["secondary"] = {f"S{i + 1}": s for i, s in enumerate(lines_s)}
    out_groups.update(interim)
    out_groups["record"] = rec
    os.makedirs(root, exist_ok=True)
    lines = protocol.write_summary(os.path.join(root, "summary.txt"), out_groups, header)
    print("\n".join(lines))
    if args.pack:
        pack(root, skip_smoke=not args.smoke)


def pack(root, skip_smoke=True):
    """send-back.zip: every .txt, .csv, .json and .md file under root, not
    the trained policies (.zip, .pt), which are large."""
    path = os.path.join(root, "send-back.zip")
    n = 0
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for d, dirs, files in os.walk(root):
            if skip_smoke and os.path.relpath(d, root).split(os.sep)[0] == "smoke":
                continue
            for f in sorted(files):
                if f.endswith((".txt", ".csv", ".json", ".md")):
                    full = os.path.join(d, f)
                    z.write(full, os.path.relpath(full, root))
                    n += 1
    print(f"{path}: {n} files")


if __name__ == "__main__":
    main()
