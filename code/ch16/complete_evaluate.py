#!/usr/bin/env python3
"""
The paired evaluation of the complete case.

Physical AI, Together, Chapter 16 (Section 16.4). Protocol: complete/PROTOCOL.md.

Every configuration of complete/config.py (A-rule, A-preview, A-learner,
F-preview and F-rule, each with complete.tube and complete.mpc) runs for 320
periods in each of the 100 evaluation worlds (seeds 45001 to 45100), through
the allovi core (allovi.run), with the frozen bounds (complete/FROZEN.json),
the calibrated offset (results/complete-calibration.txt) and the trained
learners (results/complete-learner-*.json). The script refuses to run if any
file of complete/ differs from the hashes recorded when the bounds were
frozen. It applies the decision rules of the hypotheses of PROTOCOL.md and
writes
    results/complete-runs.csv       one row per configuration and world
    results/complete-failures.csv   every run with a violation, an alarm, a
                                    departure from the model's bounds, an
                                    invalid executed label, an operation not
                                    begun on time, or no arrival
    results/complete-summary.txt    the counts by configuration, the paired
                                    comparisons, and the verdicts
    results/complete-record.txt     the record of the run

Usage
    python3 complete_evaluate.py            # writes the files above
    python3 complete_evaluate.py --worlds 3 # a smoke test on development
                                            # worlds with provisional values;
                                            # writes nothing

Requires MuJoCo 3.16.0 and NumPy. About forty minutes with two processes.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from multiprocessing import Pool

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import complete                                       # noqa: E402,F401
import allovi                                         # noqa: E402
import mujoco                                         # noqa: E402
from complete import config, model as md, worlds      # noqa: E402
from complete.modules import Preview                  # noqa: E402
from complete_identify import hashes, script_hashes   # noqa: E402
from bench import stats                               # noqa: E402

RESULTS = os.path.join(HERE, "results")


def frozen_offset():
    with open(os.path.join(RESULTS, "complete-calibration.txt")) as fh:
        for line in fh:
            if line.strip().startswith("offset ="):
                return float(line.split("=")[1].split()[0])
    raise RuntimeError("no calibrated offset")


def check_frozen():
    with open(md.FROZEN_FILE) as fh:
        frozen = json.load(fh)
    now = hashes()
    changed = [k for k in frozen["hashes"] if now.get(k) != frozen["hashes"][k]]
    if changed:
        raise SystemExit(f"complete/ changed since the bounds were frozen: {changed}")
    return frozen


class Clock:
    """Times every decision of the core with time.perf_counter, for the
    report only (the machine's wall clock; offline runs do not depend on it)."""

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


def back_periods(package, bounds):
    """N_back of PROTOCOL.md: the periods that the package's declared back-away
    takes from rest at the floor's rest point to the operation's target, as the
    preview planner computes them."""
    control = config.PACKAGES[package](bounds=bounds)
    d = control.declared()
    pv = Preview(None, d)
    x_h = md.WALL - 0.5 - d["e_p"] - d["e_l"]
    return pv.periods_to_target(x_h)


def job(spec):
    seed, name, pkg, bounds, offset, tables = spec
    config.LEARNERS.update(tables)
    world = worlds.World(seed)
    with Clock() as clk:
        rec, m, sim = config.run(world, name, pkg, bounds, offset)
    dur = clk.durations
    row = dict(config=name, package=pkg, world=seed, ramp=float(world.ramp),
               score=float(world.surveyed - world.ramp), lead=world.lead, slip=world.slip,
               onset=world.onset, onset_forecast=world.onset_forecast,
               arrival=m["arrival"], begun=m["begun"], delay=m["delay"],
               on_time=m["on_time"], overloads=len(m["overloads"]), dropped=m["dropped"],
               in_work_zone=m["in_work_zone"],
               modified_before=m["modified_before_arrival"],
               modified_after=m["modified_after_arrival"],
               periods_after=m["periods_after_arrival"], fallbacks=m["fallbacks"],
               evaluated=m["evaluated"], alarms=len(m["alarms"]),
               first_alarm=m["alarms"][0] if m["alarms"] else None,
               restarts=len(m["restarts"]), rest_feedback=m["rest_feedback"],
               least_clearance=round(m["least_clearance"], 6), violation=m["violation"],
               labels_issued=m["labels_issued"], invalid_issued=m["invalid_issued"],
               invalid_committed=m["invalid_committed"],
               invalid_executed=m["invalid_executed"],
               max_ratio=None if m["max_ratio"] is None else round(m["max_ratio"], 6),
               max_w=round(m["max_w"], 6), departure_periods=m["departure_periods"],
               decide_median_ms=round(1000 * statistics.median(dur), 3),
               decide_max_ms=round(1000 * max(dur), 3),
               final_p=round(float(m["final"][0]), 6), final_l=round(float(m["final"][2]), 6))
    for k, v in m["departures"].items():
        row["dep_" + k] = v
    return row


def failed(r):
    return (r["violation"] or r["alarms"] or r["departure_periods"] or r["invalid_executed"]
            or not r["on_time"] or r["arrival"] is None)


def summarize(rows, q_hat, n_back, frozen_bounds):
    out = []
    cfgs = [(c, p) for p in config.PACKAGES for c in config.CONFIGS]
    by = {(r["config"], r["package"], r["world"]): r for r in rows}
    seeds = sorted({r["world"] for r in rows})
    valid = [s for s in seeds if by[(cfgs[0][0], cfgs[0][1], s)]["score"] <= q_hat + 1e-12]
    out.append("[evaluation]")
    out.append(f"  worlds = {len(seeds)}; valid (score <= offset {q_hat}) = {len(valid)}; "
               f"miscovered = {len(seeds) - len(valid)}")
    out.append(f"  configurations = {len(cfgs)}; runs = {len(rows)}")
    out.append("  bounds = " + ", ".join(f"{k} {v}" for k, v in frozen_bounds.items()))
    out.append("  N_back [periods] = " + ", ".join(f"{p} {n}" for p, n in n_back.items()))
    for c, p in cfgs:
        rs = [by[(c, p, s)] for s in seeds]
        vs = [by[(c, p, s)] for s in valid]
        arr = [r["arrival"] for r in vs if r["arrival"] is not None]
        on = sum(1 for r in vs if r["on_time"])
        lo, hi = stats.clopper_pearson(on, len(vs)) if vs else (0, 0)
        out.append(f"[{c}-{p}]")
        out.append(f"  valid worlds: operation on time {on} of {len(vs)} "
                   f"(95% exact interval {float(lo):.3f} to {float(hi):.3f}); late "
                   f"{sum(1 for r in vs if r['begun'] is not None and not r['on_time'])}; "
                   f"never begun {sum(1 for r in vs if r['begun'] is None)}")
        out.append(f"  valid worlds: arrived {len(arr)}; mean arrival "
                   f"{(sum(arr) / len(arr) / 10 if arr else float('nan')):.2f} s; mean time in the "
                   f"work zone {sum(r['in_work_zone'] for r in vs) / max(1, len(vs)) / 10:.2f} s")
        out.append(f"  all worlds: violations {sum(1 for r in rs if r['violation'])}; alarms "
                   f"{sum(r['alarms'] for r in rs)} in {sum(1 for r in rs if r['alarms'])} runs; "
                   f"restarts {sum(r['restarts'] for r in rs)}; least true clearance "
                   f"{min(r['least_clearance'] for r in rs):.4f} m")
        out.append(f"  all worlds: labels issued {sum(r['labels_issued'] for r in rs)}, invalid "
                   f"{sum(r['invalid_issued'] for r in rs)}, in committed plans "
                   f"{sum(r['invalid_committed'] for r in rs)}, executed "
                   f"{sum(r['invalid_executed'] for r in rs)} in "
                   f"{sum(1 for r in rs if r['invalid_executed'])} runs")
        out.append(f"  all worlds: periods with a departure from the bounds "
                   f"{sum(r['departure_periods'] for r in rs)}; largest V/level "
                   f"{max(r['max_ratio'] or 0 for r in rs):.4f}; largest |w| "
                   f"{max(r['max_w'] for r in rs):.4f} m/s^2")
        out.append(f"  decisions (this machine, offline): median of medians "
                   f"{statistics.median(r['decide_median_ms'] for r in rs):.2f} ms; largest "
                   f"{max(r['decide_max_ms'] for r in rs):.2f} ms")
    return out, valid, by, seeds


def mcnemar(by, a, b, worlds_):
    """Exact two-sided McNemar test of 'on time' between configurations a and b."""
    only_a = sum(1 for s in worlds_ if by[a + (s,)]["on_time"] and not by[b + (s,)]["on_time"])
    only_b = sum(1 for s in worlds_ if by[b + (s,)]["on_time"] and not by[a + (s,)]["on_time"])
    both = sum(1 for s in worlds_ if by[a + (s,)]["on_time"] and by[b + (s,)]["on_time"])
    n = only_a + only_b
    if n == 0:
        return both, only_a, only_b, 1.0
    k = min(only_a, only_b)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return both, only_a, only_b, p


def verdicts(rows, valid, by, seeds, n_back):
    out = ["[hypotheses, as PROTOCOL.md states them]"]
    cfgs = [(c, p) for p in config.PACKAGES for c in config.CONFIGS]
    # H1: safety in valid worlds
    bad = [r for r in rows if r["world"] in valid and (r["violation"] or r["alarms"] or
                                                        r["departure_periods"] or
                                                        r["invalid_executed"])]
    out.append(f"  H1 (valid worlds: no violation, alarm, departure or invalid executed label "
               f"in any configuration): {'supported' if not bad else 'FAILED'} "
               f"({len(bad)} runs)")
    viol = [r for r in rows if r["violation"]]
    out.append(f"  H2 (no violation in any run, miscovered worlds included): "
               f"{'supported' if not viol else 'FAILED'} ({len(viol)} runs)")
    # H3: completion with enough lead
    h3 = []
    for c, p in cfgs:
        if not c.startswith("A") and c != "F-preview":
            continue
        for s in valid:
            r = by[(c, p, s)]
            if r["slip"] >= 0 and r["lead"] >= n_back[p] + 2 and not r["on_time"]:
                h3.append((c, p, s))
    out.append(f"  H3 (A-configurations and F-preview begin the operation on time in every "
               f"valid world with no early slip and lead >= N_back + 2): "
               f"{'supported' if not h3 else 'FAILED'} ({len(h3)} runs: "
               f"{', '.join(f'{c}-{p} {s}' for c, p, s in h3[:12])}"
               f"{' ...' if len(h3) > 12 else ''})")
    # H4: the learner prepares as the rule does
    h4 = []
    for p in config.PACKAGES:
        for s in valid:
            if by[("A-rule", p, s)]["on_time"] and not by[("A-learner", p, s)]["on_time"]:
                h4.append((p, s))
    out.append(f"  H4 (A-learner on time in every valid world in which A-rule is): "
               f"{'supported' if not h4 else 'FAILED'} ({len(h4)} worlds: {h4[:12]})")
    # H5: F-rule never prepares
    h5 = [(p, s) for p in config.PACKAGES for s in valid
          if by[("F-rule", p, s)]["on_time"] and by[("F-rule", p, s)]["arrival"] is not None
          and by[("F-rule", p, s)]["arrival"] < by[("F-rule", p, s)]["onset"]]
    out.append(f"  H5 (F-rule begins the operation on time in no valid world in which it "
               f"arrived before the onset): {'supported' if not h5 else 'FAILED'} "
               f"({len(h5)} worlds)")
    out.append("[paired comparisons, exploratory: exact McNemar test of 'on time', valid worlds]")
    for p in config.PACKAGES:
        for a, b in (("A-rule", "F-preview"), ("A-preview", "F-preview"),
                     ("A-learner", "A-rule"), ("A-rule", "F-rule")):
            both, oa, ob, pv = mcnemar(by, (a, p), (b, p), valid)
            out.append(f"  {a} vs {b}, {p}: both {both}, only the first {oa}, only the "
                       f"second {ob}, p = {pv:.3g}")
    for c in config.CONFIGS:
        both, oa, ob, pv = mcnemar(by, (c, "tube"), (c, "mpc"), valid)
        out.append(f"  {c}: tube vs mpc: both {both}, only tube {oa}, only mpc {ob}, "
                   f"p = {pv:.3g}")
    return out


def record_lines(frozen, q_hat, seconds):
    return [
        "[record]",
        f"  time (UTC) = {datetime.datetime.now(datetime.timezone.utc).isoformat()}",
        f"  Python = {platform.python_version()}; MuJoCo = {mujoco.__version__}",
        f"  machine = {platform.machine()}, {os.cpu_count()} processors, {platform.system()}",
        f"  bounds frozen at = {frozen['time']}",
        f"  calibrated offset = {q_hat}",
        f"  wall time [s] = {seconds:.0f}",
        f"  frozen files, unchanged since the bounds were frozen = {len(frozen['hashes'])}",
        "  scripts = " + ", ".join(f"{k} {v[:12]}" for k, v in script_hashes().items()),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--worlds", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    t0 = time.time()
    if args.worlds:
        seeds = list(worlds.PHASES["development"])[:args.worlds]
        bounds, q_hat, frozen = md.bounds(False), 0.40, None
        tables = {}
        configs = [c for c in config.CONFIGS if c != "A-learner"]
    else:
        frozen = check_frozen()
        seeds = list(worlds.PHASES["evaluation"])
        bounds, q_hat = md.bounds(True), frozen_offset()
        config.load_learners(RESULTS)
        tables = dict(config.LEARNERS)
        configs = list(config.CONFIGS)
    specs = [(s, c, p, bounds, q_hat, tables) for s in seeds for p in config.PACKAGES
             for c in configs]
    with Pool(args.jobs) as pool:
        rows = pool.map(job, specs, chunksize=1)
    n_back = {p: back_periods(p, bounds) for p in config.PACKAGES}
    if args.worlds:
        for r in rows:
            print({k: r[k] for k in ("config", "package", "world", "arrival", "begun", "on_time",
                                     "alarms", "violation", "departure_periods", "max_ratio",
                                     "decide_max_ms")})
        print("N_back", n_back)
        return
    lines, valid, by, s_all = summarize(rows, q_hat, n_back, bounds)
    lines += verdicts(rows, valid, by, s_all, n_back)
    lines += record_lines(frozen, q_hat, time.time() - t0)
    text = "\n".join(lines) + "\n"
    print(text)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "complete-summary.txt"), "w") as fh:
        fh.write(text)
    cols = list(rows[0])
    with open(os.path.join(RESULTS, "complete-runs.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(RESULTS, "complete-failures.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerows([r for r in rows if failed(r)])


if __name__ == "__main__":
    main()
