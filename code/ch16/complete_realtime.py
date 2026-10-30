#!/usr/bin/env python3
"""
The complete case on the wall clock: the timing evidence.

Physical AI, Together, Chapter 16 (Section 16.4). Protocol: complete/PROTOCOL.md.

Every configuration of the evaluation runs again in the first 20 evaluation
worlds (seeds 45001 to 45020) with the plant and the governor in separate
processes (complete/realtime.py): the plant advances on time.monotonic and
never waits; the governor decides with an elapsed-time deadline 6 ms before
each boundary and sends the plan, which the plant executes only if it
arrived before the boundary. One run uses the machine at a time. For every
run the script compares the record with that of the offline run
(allovi.run) of the same configuration and world, period by period, and
records the timing of every period. It writes
    results/complete-realtime-runs.csv      one row per run
    results/complete-realtime-periods.csv   one row per period of every run
    results/complete-realtime-summary.txt   the counts, the distributions and
                                            the verdict of H6

Usage
    python3 complete_realtime.py               # writes the files above
    python3 complete_realtime.py --part 1      # the first 10 worlds only, then
    python3 complete_realtime.py --part 2      # the last 10, each to its own files,
    python3 complete_realtime.py --merge       # then the files above from both
    python3 complete_realtime.py --worlds 1    # a smoke test on a development
                                               # world; writes nothing

Requires MuJoCo 3.16.0 and NumPy; two processors. About two hours.
"""
import argparse
import csv
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import complete                                       # noqa: E402,F401
from complete import config, model as md, realtime, worlds   # noqa: E402
from complete_evaluate import RESULTS, check_frozen, frozen_offset   # noqa: E402

N_WORLDS = 20
CONCURRENT = 1


def compare(rec_a, rec_b):
    """The first period at which two records differ in the state or in the
    decision (the committed entry and its ambition), or None."""
    for a, b in zip(rec_a.periods, rec_b.periods):
        if a.state != b.state or a.measured != b.measured:
            return a.k
        da, db = a.decision, b.decision
        sa = None if da is None else (da.selected, da.ambition)
        sb = None if db is None else (db.selected, db.ambition)
        if sa != sb:
            return a.k
    if len(rec_a.periods) != len(rec_b.periods) or rec_a.final_state != rec_b.final_state:
        return len(rec_a.periods)
    return None


def one(spec):
    seed, name, pkg, bounds, offset = spec
    world = worlds.World(seed)
    t0 = time.time()
    rec_rt, m_rt, timing = realtime.run(world, name, pkg, bounds, offset)
    wall = time.time() - t0
    rec_off, m_off, _ = config.run(world, name, pkg, bounds, offset)
    diff = compare(rec_rt, rec_off)
    g, p = timing["governor"], timing["plant"]
    row = dict(config=name, package=pkg, world=seed, wall_s=round(wall, 2),
               identical=diff is None, first_difference=diff,
               missed=sum(1 for x in g if x["source"] == "fallback"),
               late_proposals=sum(1 for x in g if x["late"]),
               cut=sum(1 for x in g if x["timed_out"]),
               recv_max_ms=round(1000 * max(x["recv"] for x in g), 3),
               decide_median_ms=round(1000 * statistics.median(x["decide"] for x in g), 3),
               decide_max_ms=round(1000 * max(x["decide"] for x in g), 3),
               slack_min_ms=round(1000 * min(x["slack"] for x in g), 3),
               physics_max_ms=round(1000 * max(x[3] for x in p), 3),
               publish_late_max_ms=round(1000 * max(x[2] for x in p), 3),
               on_time=m_rt["on_time"], alarms=len(m_rt["alarms"]),
               violation=m_rt["violation"], same_metrics=(
                   m_rt["on_time"] == m_off["on_time"] and m_rt["arrival"] == m_off["arrival"]))
    periods = [dict(config=name, package=pkg, world=seed, k=x["k"],
                    recv_ms=round(1000 * x["recv"], 4),
                    proposal_ms=round(1000 * x["proposal"], 4),
                    decide_ms=round(1000 * x["decide"], 4), slack_ms=round(1000 * x["slack"], 4),
                    evaluated=x["evaluated"], cut=x["timed_out"], source=x["source"],
                    physics_ms=round(1000 * pp[3], 4), publish_late_ms=round(1000 * pp[2], 4))
               for x, pp in zip(g, p)]
    return row, periods


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--worlds", type=int, default=None)
    ap.add_argument("--part", type=int, choices=(1, 2), default=None)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        rows, periods = [], []
        for part in (1, 2):
            for name, data in (("runs", rows), ("periods", periods)):
                with open(os.path.join(RESULTS, f"complete-realtime-{name}-part{part}.csv")) as fh:
                    data.extend(csv.DictReader(fh))
        conv = dict(identical=lambda x: x == "True", missed=int, cut=int, late_proposals=int)
        for r in rows:
            for k, f in conv.items():
                r[k] = f(r[k])
        for x in periods:
            for k in ("decide_ms", "slack_ms", "recv_ms", "physics_ms", "publish_late_ms"):
                x[k] = float(x[k])
        report(rows, periods, len({r["world"] for r in rows}),
               len({(r["config"], r["package"]) for r in rows}), write=True)
        return
    if args.worlds:
        seeds = list(worlds.PHASES["development"])[:args.worlds]
        bounds, q_hat = md.bounds(False), 0.40
        configs = [c for c in config.CONFIGS if c != "A-learner"]
    else:
        check_frozen()
        seeds = list(worlds.PHASES["evaluation"])[:N_WORLDS]
        if args.part:
            seeds = seeds[:N_WORLDS // 2] if args.part == 1 else seeds[N_WORLDS // 2:]
        bounds, q_hat = md.bounds(True), frozen_offset()
        config.load_learners(RESULTS)
        configs = list(config.CONFIGS)
    specs = [(s, c, p, bounds, q_hat) for s in seeds for p in config.PACKAGES for c in configs]
    with ThreadPoolExecutor(CONCURRENT) as ex:
        out = list(ex.map(one, specs))
    rows = [r for r, _ in out]
    periods = [x for _, ps in out for x in ps]
    if args.part:
        for name, data in (("runs", rows), ("periods", periods)):
            with open(os.path.join(RESULTS, f"complete-realtime-{name}-part{args.part}.csv"),
                      "w", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=list(data[0]))
                wr.writeheader()
                wr.writerows(data)
        return
    report(rows, periods, len(seeds), len(configs) * len(config.PACKAGES),
           write=not args.worlds)
    if args.worlds:
        for r in rows:
            print(r)


def report(rows, periods, n_worlds, n_configs, write):
    allp = periods
    lines = ["[real time]",
             f"  runs = {len(rows)} ({n_worlds} worlds, {n_configs} "
             f"configurations), {CONCURRENT} at a time; periods = {len(allp)}",
             f"  periods whose plan arrived after the boundary (fallback executed) = "
             f"{sum(r['missed'] for r in rows)}",
             f"  periods whose evaluation the deadline cut = {sum(r['cut'] for r in rows)}",
             f"  proposals later than the cutoff = {sum(r['late_proposals'] for r in rows)}",
             f"  runs identical to the offline run, period by period = "
             f"{sum(1 for r in rows if r['identical'])} of {len(rows)}",
             f"  decision, median and largest [ms] = "
             f"{statistics.median(x['decide_ms'] for x in allp):.3f}, "
             f"{max(x['decide_ms'] for x in allp):.3f}",
             f"  slack before the boundary, least [ms] = {min(x['slack_ms'] for x in allp):.3f}",
             f"  measurement received after the boundary, median and largest [ms] = "
             f"{statistics.median(x['recv_ms'] for x in allp):.3f}, "
             f"{max(x['recv_ms'] for x in allp):.3f}",
             f"  plant: computation of a period, median and largest [ms] = "
             f"{statistics.median(x['physics_ms'] for x in allp):.3f}, "
             f"{max(x['physics_ms'] for x in allp):.3f}",
             f"  plant: publication after the boundary, largest [ms] = "
             f"{max(x['publish_late_ms'] for x in allp):.3f}"]
    q = sorted(x["decide_ms"] for x in allp)
    lines.append(f"  decision, 99th and 99.9th percentile [ms] = {q[int(0.99 * len(q))]:.3f}, "
                 f"{q[int(0.999 * len(q))]:.3f}")
    ok = (sum(r["missed"] for r in rows) == 0 and sum(r["cut"] for r in rows) == 0
          and all(r["identical"] for r in rows))
    lines.append(f"  H6 (no plan late, no evaluation cut, every run identical to its offline "
                 f"run): {'supported' if ok else 'FAILED'}")
    text = "\n".join(lines) + "\n"
    print(text)
    if not write:
        return
    with open(os.path.join(RESULTS, "complete-realtime-summary.txt"), "w") as fh:
        fh.write(text)
    for name, data in (("complete-realtime-runs.csv", rows),
                       ("complete-realtime-periods.csv", periods)):
        with open(os.path.join(RESULTS, name), "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(data[0]))
            wr.writeheader()
            wr.writerows(data)


if __name__ == "__main__":
    main()
