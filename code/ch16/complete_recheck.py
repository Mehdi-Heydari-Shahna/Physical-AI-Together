#!/usr/bin/env python3
"""
Reruns the offline phases of the complete case with the core of this release.

Physical AI, Together, Chapter 16 (Section 16.4); added in release v1.1.

Exploratory, outside the protocol (complete/PROTOCOL.md). Release v1.1 adds
two checks of Algorithm 8.1 to ../ch10/allovi/loop.py and
complete/realtime.py, files that complete/FROZEN.json names
(complete/DEVIATIONS.md, deviation 9), so the scripts of the complete case
refuse to run, and the case runs from tag v1.0. This script reruns the
offline phases with the files of this release, the frozen bounds, and the
calibrated offset and trained learners of results/, and compares every run
with the records of release v1.0:

    python3 complete_recheck.py --stress        # S1 to S3, 180 runs
    python3 complete_recheck.py --evaluation    # the 1000 runs of the evaluation

It compares every column of results/complete-stress.csv or
results/complete-runs.csv but the wall-clock times (names ending in _ms) and
writes the comparison to results/complete-recheck-stress.txt or
results/complete-recheck-evaluation.txt; it changes no other file. The timed
stress tests S4 and S5 and the real-time phase run on the wall clock and are
not rerun.

Requires MuJoCo 3.16.0 and NumPy, as the complete case. With two processes,
about ten minutes for --stress and forty for --evaluation.
"""
import argparse
import datetime
import csv
import json
import os
import platform
import sys
import time
from multiprocessing import Pool

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mujoco                                          # noqa: E402
import numpy                                           # noqa: E402
import complete_evaluate as ce                         # noqa: E402
import complete_stress as cs                           # noqa: E402
from complete import config, model as md, worlds       # noqa: E402

RESULTS = ce.RESULTS


def cell(value):
    """A value as the csv module writes it."""
    return "" if value is None else str(value)


def changed_frozen():
    with open(md.FROZEN_FILE) as fh:
        frozen = json.load(fh)
    now = ce.hashes()
    return [k for k in frozen["hashes"] if now.get(k) != frozen["hashes"][k]]


def shipped(name, key):
    with open(os.path.join(RESULTS, name), newline="", encoding="utf-8") as fh:
        return {key(r): r for r in csv.DictReader(fh)}


def differences(rows, ship, key):
    """Every run whose columns, the wall-clock times aside, differ from the
    shipped row: (run, shipped row, differing columns)."""
    out = []
    for r in rows:
        s = ship[key(r)]
        cols = [c for c in s if not c.endswith("_ms") and cell(r.get(c)) != s[c]]
        if cols:
            out.append((r, s, cols))
    return out


def head(part, seconds):
    return [f"Rerun of the {part} of the complete case with the files of this release, "
            "outside the protocol, against the records of release v1.0",
            "",
            "[record]",
            f"  date (UTC) = {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M}",
            f"  Python {platform.python_version()}, MuJoCo {mujoco.__version__}, "
            f"NumPy {numpy.__version__}; {platform.system()} {platform.machine()}",
            "  frozen files that differ from complete/FROZEN.json = "
            + (", ".join(changed_frozen()) or "none"),
            f"  wall time [s] = {seconds:.0f}",
            ""]


def listing(diffs, key_cols, cols_shown):
    lines = []
    for r, s, cols in diffs:
        shown = [c for c in cols_shown if c in cols] + [c for c in cols if c not in cols_shown]
        lines.append("  " + ", ".join(cell(r[c]) for c in key_cols) + ": " + "; ".join(
            f"{c} {s[c]} -> {cell(r.get(c))}" for c in shown))
    return lines


def stress(jobs):
    seeds = list(worlds.PHASES["stress"])
    bounds, q_hat = md.bounds(True), ce.frozen_offset()
    specs = [(t, s, p, bounds, q_hat) for t in cs.TESTS for s in seeds for p in config.PACKAGES]
    t0 = time.time()
    with Pool(jobs) as pool:
        rows = pool.map(cs.offline, specs, chunksize=1)
    key = lambda r: (cell(r["test"]), cell(r["package"]), cell(r["world"]))   # noqa: E731
    ship = shipped("complete-stress.csv", key)
    diffs = differences(rows, ship, key)
    lines = head("stress tests S1 to S3", time.time() - t0)
    lines.append("[by test and package: release v1.0 -> this release]")
    for t in cs.TESTS:
        for p in config.PACKAGES:
            mine = [r for r in rows if r["test"] == t and r["package"] == p]
            old = [ship[key(r)] for r in mine]
            pairs = [
                ("runs with an alarm", sum(1 for s in old if int(s["alarms"])),
                 sum(1 for r in mine if r["alarms"])),
                ("restarts", sum(int(s["restarts"]) for s in old),
                 sum(r["restarts"] for r in mine)),
                ("violations", sum(1 for s in old if s["violation"] == "True"),
                 sum(1 for r in mine if r["violation"])),
                ("least clearance [m]", f"{min(float(s['least_clearance']) for s in old):.3f}",
                 f"{min(float(r['least_clearance']) for r in mine):.3f}"),
                ("arrived", sum(1 for s in old if s["arrival"] != ""),
                 sum(1 for r in mine if r["arrival"] is not None)),
            ]
            lines.append(f"  {t} {p}: runs {len(mine)}; " + "; ".join(
                f"{name} {a} -> {b}" for name, a, b in pairs))
    lines += ["", f"[runs that differ: {len(diffs)} of {len(rows)}]"]
    lines += listing(diffs, ("test", "package", "world"),
                     ("alarms", "restarts", "violation", "least_clearance"))
    return "complete-recheck-stress.txt", lines


def evaluation(jobs):
    seeds = list(worlds.PHASES["evaluation"])
    bounds, q_hat = md.bounds(True), ce.frozen_offset()
    config.load_learners(RESULTS)
    tables = dict(config.LEARNERS)
    specs = [(s, c, p, bounds, q_hat, tables) for s in seeds for p in config.PACKAGES
             for c in config.CONFIGS]
    t0 = time.time()
    with Pool(jobs) as pool:
        rows = pool.map(ce.job, specs, chunksize=1)
    key = lambda r: (cell(r["config"]), cell(r["package"]), cell(r["world"]))  # noqa: E731
    ship = shipped("complete-runs.csv", key)
    diffs = differences(rows, ship, key)
    lines = head("evaluation", time.time() - t0)
    lines.append(f"[runs that differ: {len(diffs)} of {len(rows)}]")
    lines += listing(diffs, ("config", "package", "world"), ("alarms", "violation"))
    return "complete-recheck-evaluation.txt", lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    part = ap.add_mutually_exclusive_group(required=True)
    part.add_argument("--stress", action="store_true", help="the stress tests S1 to S3")
    part.add_argument("--evaluation", action="store_true", help="the 1000 runs of the evaluation")
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    name, lines = stress(args.jobs) if args.stress else evaluation(args.jobs)
    text = "\n".join(lines) + "\n"
    with open(os.path.join(RESULTS, name), "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
