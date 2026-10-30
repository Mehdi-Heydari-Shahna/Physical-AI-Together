#!/usr/bin/env python3
"""
The identification of the complete case: the bounds of the declared model.

Physical AI, Together, Chapter 16 (Section 16.4). Protocol: complete/PROTOCOL.md.

In each of the 40 identification worlds (seeds 43001 to 43040, used for
nothing else), the MuJoCo cart runs
  * one scripted excitation run, which drives the tracking law to the edges
    of the declared envelope: references accelerating and braking at up to
    1.98 m/s^2 on level ground at up to 6.5 m/s, crossing the crest at up to
    2.1 m/s with references between -0.48 and 0.2 m/s^2, braking on the
    slope at 0.48 m/s^2, holding, backing away at 0.45 m/s^2, braking a
    backward motion at 1.9 m/s^2, accelerating towards the wall on the
    slope, and unfolding the tool four times: at rest, while backing away,
    while stopping a backward motion and while moving towards the wall
    (folding it in between, which the task never does); its numbers are
    drawn from the world's seed;
  * the configurations A-rule, A-preview and F-preview with both packages,
    under the provisional bounds of complete/model.py, for 320 periods each,
    so that the identification also contains the maneuvers that the
    governed runs make.
From every period of every run it takes the plant's summary (plant.summarize)
and the boundary measurements, and sets each bound by the rule that
PROTOCOL.md states before the run: 1.25 times the largest value observed,
rounded up to its grid:
  w_bar    the realized acceleration less the command, over the time steps
           of every period outside the window of the tool's impulse, the
           period in which the tool unfolds (or folds) and the next
           [grid 0.005 m/s^2]
  d_bar    the position residual p[i+1] - p[i] - h v[i+1] [grid 1e-7 m]
  J        the impulse beyond w_bar of each unfolding, the sum of
           h max(0, |w[i]| - w_bar) over the window [grid 0.0005 m/s]
  varpi_p, varpi_v, varpi_l   the sensor's errors, within the periods and at
           their boundaries [grids 0.0001 m, 0.0001 m/s, 0.0001 m]
  e_l      the reach's error |l - u_l| in the periods in which the tool does
           not unfold or fold [grid 0.0001 m]
It writes complete/FROZEN.json with the bounds, the observed maxima, the
counts, the hash of every file of complete/ and the time; and
results/complete-identification.txt and .csv.

Usage
    python3 complete_identify.py            # writes the files above
    python3 complete_identify.py --worlds 4 # a smoke test on development
                                            # worlds; writes nothing

Requires MuJoCo 3.16.0 and NumPy. About fifteen minutes with two processes.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import random
import sys
from multiprocessing import Pool

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import complete                                       # noqa: E402
from complete import config, model as md, worlds      # noqa: E402
from complete.packages import Seg, TrackLaw           # noqa: E402
from complete.plant import Plant, Sensor              # noqa: E402
from complete.modules import windows                  # noqa: E402
from cart_mujoco.scene import Design                  # noqa: E402

RESULTS = os.path.join(HERE, "results")
FACTOR = 1.25
GRIDS = dict(w_bar=0.005, d_bar=1e-7, J=0.0005, varpi_p=1e-4, varpi_v=1e-4,
             varpi_l=1e-4, e_l=1e-4)


def ceil_to(x, grid):
    """x rounded up to the grid, as a float with the grid's digits."""
    digits = max(0, -int(math.floor(math.log10(grid))))
    return round(math.ceil(x / grid - 1e-9) * grid, digits)


# ----------------------------------------------------------------------------
# The scripted excitation run
# ----------------------------------------------------------------------------
class Script:
    """Chains references period by period from the cart's start and runs
    them in the plant; records every period's summary and its kind:
    'unfold' (the tool unfolds), 'fold' (it folds again, which the task never
    does), or None. The references never take the bumper and the reach
    beyond 24.6 m."""

    def __init__(self, plant, rng):
        self.plant, self.rng = plant, rng
        x = plant.start(0.0)
        self.p, self.v, self.ell = x[0], 0.0, 0.0
        self.count, self.kind = 0, None          # the tool's countdown while it moves
        self.rows = []

    def period(self, a, ell=None):
        ell = self.ell if ell is None else ell
        if ell != self.ell:
            self.count, self.kind = md.TOOL_PERIODS, "unfold" if ell > 0 else "fold"
        moving = self.count >= 1
        seg = Seg(0, self.p, self.v, a, ell, self.count, 1.0, 1.0, 0.0, 0.0)
        if max(seg.phi, seg.p1) + max(ell, self.ell, md.ELL if moving else 0.0) > 24.6:
            raise AssertionError(f"the script plans {seg.phi:.3f} m with the reach {ell}")
        self.plant.run_period(TrackLaw(seg))
        s = self.plant.summary
        self.rows.append((s, self.kind if moving else None))
        self.p, self.v, self.ell = seg.p1, seg.v1, ell
        self.count = self.count - 1 if self.count >= 1 else 0

    def accelerate_to(self, a, v_top, p_limit):
        while self.v < v_top and self.p + self.v * self.v / 3.0 < p_limit:
            self.period(min(a, (v_top - self.v) / md.T))

    def brake_to_rest(self, b):
        while self.v > 0:
            self.period(-min(b, self.v / md.T))

    def stop_backward(self, a):
        while self.v < 0:
            self.period(min(a, -self.v / md.T))

    def hold(self, n, ell=None):
        for _ in range(n):
            self.period(0.0, ell)

    def cruise_until(self, p):
        while self.p < p:
            self.period(0.0)

    def back_to(self, x):
        """From rest, back away at 0.3 m/s^2 until braking the backward motion
        at 1 m/s^2 stops the cart at or before x, then stop it."""
        while (self.p - self.v * self.v / 2.0 if self.v < 0 else self.p) > x:
            self.period(-0.3)
        self.stop_backward(1.0)


def excitation(world):
    """The scripted run of one identification world; returns the summaries
    with the kind of each period."""
    rng = random.Random(world.seed * 7 + 3)
    plant = Plant(Design(ramp_start=world.ramp_start), Sensor(world.sensor_seed))
    s = Script(plant, rng)
    ramp = world.ramp_start
    # level ground: hard acceleration and braking at high speed
    a1, b1 = 1.6 + 0.38 * rng.random(), 1.6 + 0.38 * rng.random()
    v_top = 4.0 + 2.5 * rng.random()
    s.accelerate_to(a1, v_top, ramp - 3.0)
    s.brake_to_rest(b1)
    s.hold(5)
    # the crest: approach at up to 2.1 m/s, but not faster than braking at
    # 0.45 m/s^2 from the crest's end can stop by 24.2 m; cross with a
    # reference in [-0.48, 0.2] m/s^2, then brake at 0.48 m/s^2 to rest
    v_max = min(2.1, math.sqrt(2 * 0.45 * max(0.0, 24.2 - (ramp + 1.7))))
    v_c = min(v_max, 1.5 + 0.6 * rng.random())
    while s.p + 0.5 * v_c * v_c < ramp - 0.8 and s.v < v_c:
        s.period(min(1.0, (v_c - s.v) / md.T))
    s.cruise_until(ramp - 0.3)
    a_c = -0.48 + 0.68 * rng.random()
    while s.p < ramp + 1.7 and s.v > 0.3:
        a = a_c if s.v + a_c * md.T <= v_c else 0.0
        if s.p + s.v * md.T + (s.v + a * md.T) ** 2 / (2 * 0.48) > 24.2:
            a = -0.48
        s.period(a)
    s.brake_to_rest(0.48)
    s.hold(10 + int(10 * rng.random()))
    # back away and stop the backward motion hard
    for _ in range(5 + int(10 * rng.random())):
        s.period(-0.45)
    s.stop_backward(1.9)
    s.hold(5)
    # towards the wall on the slope, then brake, if there is room
    if s.p < 23.0:
        for _ in range(3):
            s.period(1.0)
        s.brake_to_rest(0.48)
    s.hold(10)
    # the tool: unfold at rest; then, folded again in between, unfold while
    # backing away, while stopping a backward motion, and while moving
    # towards the wall; first, room for the tool
    s.back_to(23.2)
    s.hold(5)
    s.hold(1, md.ELL)
    s.hold(20)
    s.hold(1, 0.0)
    s.hold(10)
    m = 3 + int(5 * rng.random())
    for i in range(m + 6):
        s.period(-0.25, md.ELL if i >= m else 0.0)
    s.stop_backward(1.0)
    s.hold(10)
    s.hold(1, 0.0)
    s.hold(10)
    for _ in range(4 + int(4 * rng.random())):
        s.period(-0.25)
    s.period(min(1.0, -s.v / md.T), md.ELL)
    s.stop_backward(1.0)
    s.hold(10)
    s.hold(1, 0.0)
    s.hold(10)
    if s.p < 23.4:
        s.period(0.3)
        s.period(0.3, md.ELL)
        s.brake_to_rest(0.45)
    s.hold(20)
    return s.rows


# ----------------------------------------------------------------------------
# The governed runs
# ----------------------------------------------------------------------------
def governed(world):
    b = md.bounds(frozen=False)
    out = []
    for pkg in ("tube", "mpc"):
        for name in ("A-rule", "A-preview", "F-preview"):
            rec, m, sim = config.run(world, name, pkg, b, 0.40)
            unfold = {x.k for x in rec.periods if x.governor == "normal" and x.plan.pairs
                      and x.plan.pairs[0][0].unfold}
            rows = [(s, "unfold" if k in unfold else None) for k, s in enumerate(sim.periods)]
            out.append((f"{name}-{pkg}", rows, list(cfg_errors(rec, sim)),
                        m["alarms"], m["violation"]))
    return out


def cfg_errors(rec, sim):
    """The boundary measurements' errors of a run."""
    for x in rec.periods:
        yield tuple(a - b for a, b in zip(x.measured, x.state))


def job(seed):
    world = worlds.World(seed)
    runs = [("excitation", excitation(world), [], [], False)] + governed(world)
    stats = []
    for name, rows, boundary, alarms, violation in runs:
        st = dict(world=seed, run=name, periods=len(rows), alarms=len(alarms),
                  violation=violation)
        kinds = [kind for _, kind in rows]
        # the windows of the tool's impulse: the periods in which it moves
        # (unfolds or folds) and the next
        events = windows({k for k, kind in enumerate(kinds) if kind}, len(rows))
        window = set().union(*events) if events else set()
        plain = [s for k, (s, _) in enumerate(rows) if k not in window]
        st["w"] = max(s["w"] for s in plain)
        st["d"] = max(s["d"] for s, _ in rows)
        st["n_p"] = max([s["n_p"] for s, _ in rows] + [abs(e[0]) for e in boundary])
        st["n_v"] = max([s["n_v"] for s, _ in rows] + [abs(e[1]) for e in boundary])
        st["n_l"] = max([s["n_l"] for s, _ in rows] + [abs(e[2]) for e in boundary])
        folding = {k for k, kind in enumerate(kinds) if kind}
        st["l_err"] = max(s["l_err"] for k, (s, _) in enumerate(rows) if k not in folding)
        st["unfold_w_all"] = [[x for k in sorted(ev) for x in rows[k][0]["w_all"]]
                              for ev in events if kinds[min(ev)] == "unfold"]
        st["unfolds"] = sum(1 for ev in events if kinds[min(ev)] == "unfold")
        st["contacts_min"] = min(s["contacts"] for s, _ in rows)
        stats.append(st)
    return stats


FROZEN_SOURCES = ("complete", "../ch10/allovi", "../ch10/cart", "../ch11/cart_mujoco", "bench")
"""The folders whose Python files, and the protocol, the frozen record hashes:
the complete case, the core, the cart's task and demand predictor, the
simulated cart, and the benchmark's world family."""


def hashes():
    out = {}
    for folder in FROZEN_SOURCES:
        full = os.path.normpath(os.path.join(HERE, folder))
        for name in sorted(os.listdir(full)):
            if name.endswith(".py") or name == "PROTOCOL.md":
                with open(os.path.join(full, name), "rb") as fh:
                    out[f"{folder}/{name}"] = hashlib.sha256(fh.read()).hexdigest()
    return out


SCRIPTS = ("complete_identify.py", "complete_calibrate.py", "complete_train.py",
           "complete_evaluate.py", "complete_realtime.py", "complete_stress.py")


def script_hashes():
    """The scripts' hashes, recorded with every output; the scripts call the
    frozen modules and report, so they are not part of the frozen check."""
    out = {}
    for name in SCRIPTS:
        with open(os.path.join(HERE, name), "rb") as fh:
            out[name] = hashlib.sha256(fh.read()).hexdigest()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--worlds", type=int, default=None,
                    help="a smoke test on that many development worlds; writes nothing")
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    if args.worlds:
        seeds = list(worlds.PHASES["development"])[:args.worlds]
    else:
        seeds = list(worlds.PHASES["identification"])
    with Pool(args.jobs) as pool:
        stats = [st for out in pool.map(job, seeds) for st in out]
    obs = {k: max(st[k] for st in stats) for k in ("w", "d", "n_p", "n_v", "n_l", "l_err")}
    w_bar = ceil_to(FACTOR * obs["w"], GRIDS["w_bar"])
    impulse = 0.0
    for st in stats:
        for ws in st["unfold_w_all"]:
            impulse = max(impulse, sum(md.H * max(0.0, abs(x) - w_bar) for x in ws))
    obs["J"] = impulse
    bounds = dict(w_bar=w_bar, d_bar=ceil_to(FACTOR * obs["d"], GRIDS["d_bar"]),
                  J=ceil_to(FACTOR * impulse, GRIDS["J"]),
                  varpi_p=ceil_to(FACTOR * obs["n_p"], GRIDS["varpi_p"]),
                  varpi_v=ceil_to(FACTOR * obs["n_v"], GRIDS["varpi_v"]),
                  varpi_l=ceil_to(FACTOR * obs["n_l"], GRIDS["varpi_l"]),
                  e_l=ceil_to(FACTOR * obs["l_err"], GRIDS["e_l"]))
    periods = sum(st["periods"] for st in stats)
    lines = ["[identification]",
             f"  worlds = {len(seeds)} ({seeds[0]} to {seeds[-1]})",
             f"  runs = {len(stats)}; periods = {periods}; time steps = {periods * md.N_STEPS}",
             f"  periods in which the tool unfolds = {sum(st['unfolds'] for st in stats)}",
             f"  governed runs with an alarm, a violation = "
             f"{sum(1 for st in stats if st['alarms'])}, "
             f"{sum(1 for st in stats if st['violation'])}",
             f"  least number of contacts at a period's end = "
             f"{min(st['contacts_min'] for st in stats)}",
             "[largest observed]"]
    lines += [f"  {k} = {v:.6g}" for k, v in obs.items()]
    lines += ["[bounds: 1.25 times the largest, rounded up]"]
    lines += [f"  {k} = {v}" for k, v in bounds.items()]
    for pkg in ("tube", "mpc"):
        c = md.certificate(pkg, bounds)
        lines.append(f"[certificate {pkg}] phi = {c['phi']:.6g}, iota = {c['iota']:.6g}, "
                     f"w_tot = {c['w_tot']:.6g}, rho^100 = {c['rho_n']:.6g}")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.worlds:
        return
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "complete-identification.txt"), "w") as fh:
        fh.write(text)
    with open(os.path.join(RESULTS, "complete-identification.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        cols = ["world", "run", "periods", "alarms", "violation", "w", "d", "n_p", "n_v",
                "n_l", "l_err", "unfolds", "contacts_min"]
        wr.writerow(cols)
        for st in stats:
            wr.writerow([st[c] if not isinstance(st[c], float) else f"{st[c]:.6g}"
                         for c in cols])
    frozen = dict(bounds=bounds, observed=obs, factor=FACTOR, grids=GRIDS,
                  worlds=[seeds[0], seeds[-1]], runs=len(stats), periods=periods,
                  time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  hashes=hashes(), scripts=script_hashes())
    with open(md.FROZEN_FILE, "w") as fh:
        json.dump(frozen, fh, indent=2)
        fh.write("\n")


if __name__ == "__main__":
    main()
