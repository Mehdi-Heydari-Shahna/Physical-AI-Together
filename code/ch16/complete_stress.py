#!/usr/bin/env python3
"""
The shifted stress tests of the complete case.

Physical AI, Together, Chapter 16 (Section 16.4). Protocol: complete/PROTOCOL.md.

Each test changes one thing that the evaluation held within the declared
model or its calibration, in the 30 stress worlds (seeds 46001 to 46030),
with A-rule and both packages, the frozen bounds, offset and software:
    S1  icy ground: sliding friction 0.20 instead of 0.25 everywhere, so that
        the ground brakes at most 1.96 m/s^2 on level ground and 0.31 m/s^2
        on the slope: both labels overstate the braking, outside the declared
        model (offline)
    S2  an unmodelled payload of 30 kg on the chassis, which the drive does
        not know (offline)
    S3  a surveyed start 0.6 m beyond the true one in every world, a score
        above the calibrated offset: labels of level ground invalid on the
        first part of the crest (offline)
    S4  timing: on the wall clock (complete/realtime.py), each proposal delayed
        by a time uniform on [0, 0.15] s with probability 0.3, and each
        evaluation of the test delayed by 15 ms, in the first 10 stress worlds,
        with the allowance for one evaluation raised to 25 ms, which bounds the
        slowed evaluation
    S5  as S4, with the nominal allowance of 10 ms, which the slowed evaluation
        exceeds: plans can arrive late, and the plant then executes the fallback
It writes results/complete-stress.csv (one row per run) and
results/complete-stress-summary.txt.

Usage
    python3 complete_stress.py              # writes the files above
    python3 complete_stress.py --worlds 1   # a smoke test; writes nothing

Requires MuJoCo 3.16.0 and NumPy. About fifteen minutes.
"""
import argparse
import csv
import os
import random
import sys
import time
from multiprocessing import Pool

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import complete                                       # noqa: E402,F401
from complete import config, model as md, realtime, worlds   # noqa: E402
from complete_evaluate import RESULTS, check_frozen, frozen_offset   # noqa: E402

TESTS = {"S1": dict(design_kw=dict(friction=0.20)),
         "S2": dict(design_kw=dict(chassis_mass=110.0), drive_mass=90.1),
         "S3": dict(survey_bias=0.6)}


class SlowProposal:
    """Delays the wrapped proposal method by U[0, 0.15] s with probability 0.3."""

    def __init__(self, inner, seed):
        self.inner, self.rng = inner, random.Random(seed)
        for name in ("ROLE", "NAME", "CONTRACT"):
            setattr(self, name, getattr(inner, name))

    def propose(self, observation):
        if self.rng.random() < 0.3:
            time.sleep(0.15 * self.rng.random())
        return self.inner.propose(observation)


class SlowTest:
    """The wrapped control package, with every evaluation of the test
    delayed by 15 ms."""

    def __init__(self, inner):
        self.__dict__["inner"] = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def profile(self, plan, initial):
        time.sleep(0.015)
        return self.inner.profile(plan, initial)


def offline(spec):
    test, seed, pkg, bounds, offset = spec
    world = worlds.World(seed)
    kw = dict(TESTS[test])
    bias = kw.pop("survey_bias", None)
    if bias is not None:
        kw["surveyed"] = world.ramp + type(world.ramp)(bias).limit_denominator(100)
    rec, m, sim = config.run(world, "A-rule", pkg, bounds, offset, **kw)
    return row(test, seed, pkg, m, sim, world)


def row(test, seed, pkg, m, sim, world, timing=None):
    r = dict(test=test, package=pkg, world=seed, arrival=m["arrival"], begun=m["begun"],
             on_time=m["on_time"], alarms=len(m["alarms"]), restarts=len(m["restarts"]),
             rest_feedback=m["rest_feedback"], violation=bool(m["violation"]),
             least_clearance=round(float(m["least_clearance"]), 6),
             invalid_executed=m["invalid_executed"], departure_periods=m["departure_periods"],
             dep_w=m["departures"]["w"], dep_tube=m["departures"]["tube"],
             max_w=round(m["max_w"], 4),
             max_ratio=None if m["max_ratio"] is None else round(m["max_ratio"], 4))
    if timing is not None:
        g = timing["governor"]
        r.update(missed=sum(1 for x in g if x["source"] == "fallback"),
                 late_proposals=sum(1 for x in g if x["late"]),
                 cut=sum(1 for x in g if x["timed_out"]),
                 slack_min_ms=round(1000 * min(x["slack"] for x in g), 3))
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--worlds", type=int, default=None)
    args = ap.parse_args()
    if args.worlds:
        seeds = list(worlds.PHASES["development"])[:args.worlds]
        bounds, q_hat = md.bounds(False), 0.40
    else:
        check_frozen()
        seeds = list(worlds.PHASES["stress"])
        bounds, q_hat = md.bounds(True), frozen_offset()
    specs = [(t, s, p, bounds, q_hat) for t in TESTS for s in seeds for p in config.PACKAGES]
    with Pool(2) as pool:
        rows = pool.map(offline, specs, chunksize=1)
    for test, allowance in (("S4", 0.025), ("S5", realtime.ALLOWANCE_S)):
        for s in seeds[:10]:
            for p in config.PACKAGES:
                world = worlds.World(s)
                rec, m, timing = realtime.run(
                    world, "A-rule", p, bounds, q_hat, allowance=allowance,
                    wrap_proposal=lambda pm, s=s: SlowProposal(pm, s), wrap_control=SlowTest)
                rows.append(row(test, s, p, m, None, world, timing))
    lines = ["[stress tests]"]
    for t in ("S1", "S2", "S3", "S4", "S5"):
        for p in config.PACKAGES:
            rs = [r for r in rows if r["test"] == t and r["package"] == p]
            if not rs:
                continue
            line = (f"  {t} {p}: runs {len(rs)}; violations {sum(r['violation'] for r in rs)}; "
                    f"runs with an alarm {sum(1 for r in rs if r['alarms'])}; restarts "
                    f"{sum(r['restarts'] for r in rs)}; invalid executed labels in "
                    f"{sum(1 for r in rs if r['invalid_executed'])} runs; periods with a "
                    f"departure {sum(r['departure_periods'] for r in rs)} (w "
                    f"{sum(r['dep_w'] for r in rs)}, tube {sum(r['dep_tube'] for r in rs)}); "
                    f"largest |w| {max(r['max_w'] for r in rs):.3f}; least clearance "
                    f"{min(r['least_clearance'] for r in rs):.3f} m; on time "
                    f"{sum(1 for r in rs if r['on_time'])}; arrived "
                    f"{sum(1 for r in rs if r['arrival'] is not None)}")
            if t in ("S4", "S5"):
                line += (f"; plans late {sum(r['missed'] for r in rs)}; evaluations cut in "
                         f"{sum(r['cut'] for r in rs)} periods; late proposals "
                         f"{sum(r['late_proposals'] for r in rs)}; least slack "
                         f"{min(r['slack_min_ms'] for r in rs):.1f} ms")
            lines.append(line)
    s4 = [r for r in rows if r["test"] == "S4"]
    ok = s4 and all(r["missed"] == 0 for r in s4) and not any(r["violation"] for r in s4)
    lines.append(f"  H7 (S4: no plan late and no violation): {'supported' if ok else 'FAILED'}")
    s5 = [r for r in rows if r["test"] == "S5"]
    ok5 = s5 and not any(r["violation"] or r["alarms"] for r in s5)
    lines.append(f"  H8 (S5: no violation and no alarm, plans late or not): "
                 f"{'supported' if ok5 else 'FAILED'}")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.worlds:
        return
    with open(os.path.join(RESULTS, "complete-stress-summary.txt"), "w") as fh:
        fh.write(text)
    cols = sorted({k for r in rows for k in r}, key=lambda k: list(rows[-1]).index(k)
                  if k in rows[-1] else 99)
    with open(os.path.join(RESULTS, "complete-stress.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)


if __name__ == "__main__":
    main()
