#!/usr/bin/env python3
"""
The anticipation benchmark: the cart's tool operation in a family of worlds.

Physical AI, Together, Chapter 16 (Sections 16.1 to 16.3).

The cart of Chapter 3 runs through the allovi core of Chapter 10, with the
control package bench.constant-command (cart.constant-command with one more
proposal, 'retreat', for the preview planner), in the 100 worlds of
bench/worlds.py, under the fifteen configurations of HYPOTHESES.md: the
structural reading of a predictive safety filter with three capability
models, fixed conservatism, reactive preparation, ALLOVI with the survey's
labels and the task plan's forecast, its two ablations, three oracle
diagnostics, a model-based preview planner under the filter and under
ALLOVI, and ALLOVI with one fresh candidate per period. Information,
authority, task and budget are those of HYPOTHESES.md in every
configuration; only the modules named there change.

This script
  * checks that bench.constant-command reproduces, with the proposals of
    Chapter 3, the forecast-informed run of Section 8.2 as the core of
    Chapter 10 wrote it (../ch10/results/decisions-forecast-informed.csv);
  * runs every configuration in every world for 320 periods (32 s) and
    measures, from the core's records and the world's oracle, the task's
    outcomes (arrival in the work zone, time there, when the operation
    begins, overloads), the interventions, the labels' validity, the
    alarms, restarts and true clearance, the evaluations of the test, and
    the decisions' wall-clock time;
  * applies the decision rule of every hypothesis of HYPOTHESES.md;
  * reports every configuration with exact intervals (bench/stats.py), and
    lists every failed run: every run with an invalid executed label, an
    alarm, a violation or an operation that did not begin on time;
  * writes the data of Figure 16.2.

Usage
    python3 cart_benchmark.py            # writes results/benchmark-*
    python3 cart_benchmark.py --check    # also checks every value of the
                                         # summary but the timing
    python3 cart_benchmark.py --jobs 1   # one process (default: up to two)

Outputs, in results/ next to this file
    benchmark-summary.txt   every value, by group; with --check, the result
    benchmark-runs.csv      one row per configuration and world
    benchmark-failures.csv  every failed run
    benchmark-traces.csv    the cart's position about the true onset, by
                            configuration: median, least and largest over the
                            worlds of Figure 16.2
    benchmark-record.txt    the record of the experiment

Python 3.8 or later; no other packages. About fifteen minutes with two
processes. The results do not depend on the number of processes.
"""
import argparse
import csv
import filecmp
import os
import platform
import sys
import tempfile
import time
from fractions import Fraction as Fr
from multiprocessing import Pool

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "ch10"))
sys.path.insert(0, os.path.join(HERE, "..", "ch07"))
import allovi                                                    # noqa: E402
import cart                                                      # noqa: E402
from allovi.records import decision_rows, write_csv as core_csv  # noqa: E402
from cart.model import ce                                        # noqa: E402
import cart_forecasts as ch7                                     # noqa: E402

from bench import hypotheses as bh                               # noqa: E402
from bench import modules as bm                                  # noqa: E402
from bench import stats                                          # noqa: E402
from bench import worlds as bw                                   # noqa: E402

PERIODS = 320
RAMP_CH3 = ce.RAMP
LATE = 6                                   # periods by which the late planner is late
CH10 = os.path.join(HERE, "..", "ch10", "results")

# id: (name, capability, demand, proposal, budget)
CONFIGS = {
    "C1": ("filter, fixed model", "fixed", "none", "naive", None),
    "C2": ("filter, measured model", "reactive", "none", "naive", None),
    "C3": ("filter, survey", "survey", "none", "naive", None),
    "C4": ("fixed conservatism", "survey", "always", "naive", None),
    "C5": ("reactive preparation", "survey", "due", "naive", None),
    "C6": ("ALLOVI", "survey", "forecast", "naive", None),
    "C7": ("ALLOVI without the capability forecast", "fixed", "forecast", "naive", None),
    "C8": ("oracle capability (diagnostic)", "map", "forecast", "naive", None),
    "C9": ("oracle demand (diagnostic)", "survey", "exact", "naive", None),
    "C10": ("both oracles (diagnostic)", "map", "exact", "naive", None),
    "C11": ("filter, survey, preview planner", "survey", "none", "preview", None),
    "C12": ("filter, survey, late preview planner", "survey", "none", "late", None),
    "C13": ("ALLOVI, late preview planner", "survey", "forecast", "late", None),
    "C14": ("ALLOVI, preview planner", "survey", "forecast", "preview", None),
    "C15": ("ALLOVI, one candidate per period", "survey", "forecast", "naive", 1),
}
TRACED = ("C3", "C4", "C5", "C6", "C12")   # Figure 16.2
TAU = (-80, 40)                            # periods about the true onset

_Q_HAT = None


def q_hat():
    """The calibrated offset of Section 7.3, recomputed from its 100
    calibration worlds."""
    global _Q_HAT
    if _Q_HAT is None:
        _, _, _Q_HAT = ch7.calibration()
    return _Q_HAT


def valid(world):
    """Every label of the survey predictor is valid in the world: the survey
    places the start at most the offset after the true start."""
    return world.surveyed - world.ramp <= q_hat()


def capability(kind, world):
    if kind == "fixed":
        return ("cart.fixed",)
    if kind == "reactive":
        return ("cart.reactive",)
    if kind == "map":
        return ("cart.forecasting",)
    return ("cart.survey", {"ramp_start": world.surveyed - q_hat()})


def configuration(cid, world):
    name, cap, demand, proposal, budget = CONFIGS[cid]
    task = bm.operation(world, demand)
    modules = {"control": ("bench.constant-command",), "robot": ("cart.robot",),
               "world": ("cart.track",), "capability": capability(cap, world), "task": task,
               "evaluator": bm.Metrics(world, task)}
    if demand != "none":
        modules["demand"] = ("cart.task-plan", {"task": task})
    if proposal == "naive":
        modules["proposal"] = ("cart.naive",)
    else:
        modules["proposal"] = bm.Preview(task, late=LATE if proposal == "late" else 0)
    return allovi.build(f"{cid}-{world.seed}", ce.T, modules,
                        budget=(lambda k: budget) if budget is not None else None)


def run(spec):
    """One configuration in one world."""
    cid, seed = spec
    world = bw.World(seed)
    cfg = configuration(cid, world)
    ce.RAMP = world.ramp
    try:
        with bm.Clock() as clock:
            rec = allovi.run(cfg, PERIODS, strict=True)
        m = cfg.evaluator.evaluate(rec)
        trace = None
        if cid in TRACED:
            trace = [x.state[0] for x in rec.periods] + [rec.final_state[0]]
    finally:
        ce.RAMP = RAMP_CH3
    m.update(config=cid, world=seed, durations=clock.durations)
    return m, trace


def chapter8_consistency():
    """bench.constant-command with the proposals of Chapter 3: the
    forecast-informed run of Section 8.2 (the world of Chapter 3, announced in
    period 140 for period 200, postponed by 21), whose decision file the core
    wrote in ../ch10/results/."""
    task = cart.modules.Operation(announce=140, onset=200, postpone=21)
    cfg = allovi.build("forecast-informed", ce.T, {
        "control": ("bench.constant-command",), "proposal": ("cart.naive",),
        "capability": ("cart.forecasting",), "demand": ("cart.task-plan", {"task": task}),
        "task": task, "robot": ("cart.robot",), "world": ("cart.track",)})
    rec = allovi.run(cfg, 250, strict=True, diagnostics=True)
    cols = ["k", "t", "p", "v", "p_init", "v_init", "mode", "proposed", "req", "onset",
            "candidates", "admitted", "selected", "ambition", "level", "onset_level", "reason"]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "decisions.csv")
        core_csv(path, cols, decision_rows(rec, cfg.control, cfg.robot), cart.fmt)
        same = filecmp.cmp(path, os.path.join(CH10, "decisions-forecast-informed.csv"),
                           shallow=False)
    return same, len(rec.periods)


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def secs(periods):
    """A number of periods as seconds, exactly."""
    return None if periods is None else Fr(periods) * ce.T


def fmt_s(x, nd=1):
    return "never" if x is None else f"{float(x):.{nd}f}"


def proportion(x, n):
    lo, hi = stats.clopper_pearson(x, n)
    return f"{x} of {n} ({stats.percent(x, n)} %; {stats.interval_percent(lo, hi)})"


def median_text(values, nd=1, unit=" s"):
    if not values:
        return "none"
    med, lo, hi, cover = stats.median_interval(values)
    span = (f"{float(lo):.{nd}f} to {float(hi):.{nd}f}" if cover is not None
            else "no interval")
    return (f"{float(med):.{nd}f}{unit} (95 % interval {span}; range "
            f"{float(min(values)):.{nd}f} to {float(max(values)):.{nd}f}; n = {len(values)})")


def config_facts(cid, runs, worlds, valid_seeds):
    ms = [runs[(cid, s)] for s in sorted(worlds)]
    vs = [runs[(cid, s)] for s in sorted(valid_seeds)]
    n, nv = len(ms), len(vs)
    on_time = sum(1 for m in vs if m["delay"] == 0)
    never = sum(1 for m in vs if m["begun"] is None)
    late = [secs(m["delay"]) for m in vs if m["delay"] not in (None, 0)]
    arrived = [secs(m["arrival"]) for m in vs if m["arrival"] is not None]
    wz = [secs(m["in_work_zone"]) for m in vs]
    after = [m["modified_after_arrival"] for m in vs if m["arrival"] is not None]
    over = [m["overridden_after_arrival"] for m in vs if m["arrival"] is not None]
    invalid = [m for m in ms if m["invalid"]]
    alarms = [m for m in ms if m["alarms"]]
    viol = [m for m in ms if m["violation"]]
    evals = sum(m["evaluated"] for m in ms)
    offered = sum(m["offered"] for m in ms)
    cut = sum(m["cut"] for m in ms)
    least = min(m["least_clearance"] for m in ms if not m["invalid"])
    return {
        "valid worlds: the operation on time": proportion(on_time, nv),
        "valid worlds: the operation late, by [s]": median_text(late),
        "valid worlds: the operation never begun": f"{never} of {nv}",
        "valid worlds: arrival in the work zone [s]": median_text(arrived) + (
            f"; never in {nv - len(arrived)}" if len(arrived) < nv else ""),
        "valid worlds: time in the work zone [s]": median_text(wz),
        "valid worlds: periods after arrival, before the operation, committed with an "
        "ambition below 1": median_text(after, 0, ""),
        "valid worlds: periods after arrival, before the operation, in which the executed "
        "command is not the proposed one": median_text(over, 0, ""),
        "all worlds: runs with an invalid executed label; with an alarm; restarts; violations":
            (f"{len(invalid)}; {len(alarms)}; {sum(len(m['restarts']) for m in ms)}; "
             f"{len(viol)} of {n}"),
        "all worlds: the least true clearance in runs whose executed labels are valid [m]":
            f"{float(least):.2f}",
        "all worlds: fresh candidates evaluated, of those offered; periods cut by the budget":
            f"{evals} of {offered}; {cut}",
    }


def failures(runs, worlds):
    """Every run that failed in some respect, one row each."""
    out = []
    for (cid, seed), m in sorted(runs.items(), key=lambda x: (int(x[0][0][1:]), x[0][1])):
        what = []
        if m["invalid"]:
            what.append(f"invalid executed labels {m['invalid']} (exercised {m['exercised']}), "
                        f"the first in period {m['first_invalid']}")
        if m["alarms"]:
            what.append(f"alarms in periods {m['alarms']}, restarts {m['restarts']}")
        if m["violation"]:
            what.append("violation")
        if m["delay"] is None:
            what.append("the operation never begun")
        elif m["delay"] > 0:
            what.append(f"the operation {m['delay']} periods late")
        if m["arrival"] is None:
            what.append("never in the work zone")
        if what:
            w = worlds[seed]
            out.append(dict(config=cid, world=seed, valid="yes" if valid(w) else "no",
                            lead=w.lead, slip=w.slip, what="; ".join(what)))
    return out


def failure_facts(rows, worlds):
    """The failed runs, by kind and by configuration."""
    by = {}
    for r in rows:
        by.setdefault(r["config"], []).append(r)
    out = {}
    for cid in CONFIGS:
        rs = by.get(cid, [])
        alarm = sorted(r["world"] for r in rs if "alarms" in r["what"])
        invalid = sorted(r["world"] for r in rs if "invalid" in r["what"])
        never = sum(1 for r in rs if "never begun" in r["what"])
        late = sum(1 for r in rs if "periods late" in r["what"])
        out[f"{cid}: failed runs; never begun, late; with invalid executed labels (worlds); "
            f"with alarms (worlds)"] = (
            f"{len(rs)}; {never}, {late}; {', '.join(map(str, invalid)) or 'none'}; "
            f"{', '.join(map(str, alarm)) or 'none'}")
    return out


def statement_facts(verdicts, runs, valid_seeds):
    """Each statement of each hypothesis: its cases and exceptions, with the
    worlds of the exceptions; for A6, the quantities in which the runs
    differ, and by how much the time in the work zone differs."""
    out = {}
    for hid, _, v in verdicts:
        for tag, text, cases, ws in v["statements"]:
            uniq = sorted(set(ws))
            out[f"{hid}.{tag} {text}: cases; exceptions"] = (
                f"{cases}; {len(ws)}" + (f"; worlds {', '.join(map(str, uniq))}" if uniq else ""))
    diff = bh.a6_keys(runs, valid_seeds)
    for a, b in (("C11", "C6"), ("C14", "C11")):
        out[f"A6: valid worlds in which {a} differs from {b} in the operation's period; the "
            f"arrival; the time in the work zone"] = "; ".join(
                str(len(diff[(a, b, k)])) for k in bh.A6_KEYS)
    for a, b in (("C6", "C11"), ("C14", "C11"), ("C14", "C6")):
        d = {}
        for s in sorted(valid_seeds):
            x = runs[(a, s)]["in_work_zone"] - runs[(b, s)]["in_work_zone"]
            d[x] = d.get(x, 0) + 1
        out[f"A6: periods in the work zone, {a} less {b}: value (valid worlds)"] = ", ".join(
            f"{x} ({n})" for x, n in sorted(d.items()))
    return out


def world_facts(worlds, valid_seeds):
    ws = [worlds[s] for s in sorted(worlds)]
    inval = [w for w in ws if w.seed not in valid_seeds]
    return {
        "worlds; valid; unreliable surveys": (f"{len(ws)}; {len(valid_seeds)}; "
                                              f"{sum(1 for w in ws if w.unreliable)}"),
        "the calibrated offset of Section 7.3 [m]": f"{float(q_hat()):.2f}",
        "the worlds that are not valid, and their surveys' errors [m]": ", ".join(
            f"{w.seed} ({float(w.survey_error):+.2f})" for w in inval) or "none",
        "the survey's labels valid (the coverage of the calibrated offset, which claims a "
        "miscoverage of at most 5 %)": proportion(len(valid_seeds), len(ws)),
        "lead times of 21 periods or more; of 18 or fewer": (
            f"{sum(1 for w in ws if w.lead >= 21)}; {sum(1 for w in ws if w.lead <= 18)}"),
        "onsets that slip later; earlier": (f"{sum(1 for w in ws if w.slip > 0)}; "
                                            f"{sum(1 for w in ws if w.slip < 0)}"),
    }


def timing_facts(runs):
    """Wall-clock time of the decisions, by configuration (this machine)."""
    out = {}
    for cid in CONFIGS:
        ds = sorted(d for (c, _), m in runs.items() if c == cid for d in m["durations"])
        n = len(ds)
        out[cid] = (f"median {1000 * ds[n // 2]:.2f} ms, 99th percentile "
                    f"{1000 * ds[min(n - 1, (99 * n) // 100)]:.2f} ms, largest "
                    f"{1000 * ds[-1]:.2f} ms over {n} decisions")
    return out


def trace_rows(traces, worlds, valid_seeds):
    """Figure 16.2: in the valid worlds with no slip and a lead time of at
    least 27 periods, the position at each period boundary about the true
    onset: median, least and largest."""
    keep = [s for s in sorted(valid_seeds) if worlds[s].slip == 0 and worlds[s].lead >= 27]
    rows = []
    for tau in range(TAU[0], TAU[1] + 1):
        row = dict(t=f"{float(Fr(tau) * ce.T):.1f}")
        for cid in TRACED:
            xs = sorted(traces[(cid, s)][worlds[s].onset + tau] for s in keep)
            n = len(xs)
            med = xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2
            row[f"{cid}_med"] = f"{float(med):.4f}"
            row[f"{cid}_min"] = f"{float(xs[0]):.4f}"
            row[f"{cid}_max"] = f"{float(xs[-1]):.4f}"
        rows.append(row)
    return rows, len(keep)


def write_rows(path, rows, cols=None):
    cols = cols or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def run_rows(runs, worlds):
    rows = []
    for (cid, seed), m in sorted(runs.items(), key=lambda x: (int(x[0][0][1:]), x[0][1])):
        w = worlds[seed]
        rows.append(dict(
            config=cid, world=seed, valid="yes" if valid(w) else "no", ramp=f"{float(w.ramp):.2f}",
            survey_error=f"{float(w.survey_error):.2f}", announce=w.announce,
            onset_forecast=w.onset_forecast, onset=w.onset, lead=w.lead, slip=w.slip,
            arrival=m["arrival"] if m["arrival"] is not None else "",
            in_work_zone=m["in_work_zone"], begun=m["begun"] if m["begun"] is not None else "",
            delay=m["delay"] if m["delay"] is not None else "",
            overloads=" ".join(map(str, m["overloads"])),
            modified_before_arrival=m["modified_before_arrival"],
            modified_after_arrival=m["modified_after_arrival"],
            overridden_after_arrival=m["overridden_after_arrival"], fallbacks=m["fallbacks"],
            evaluated=m["evaluated"], offered=m["offered"], cut=m["cut"],
            least_clearance=f"{float(m['least_clearance']):.4f}",
            violation="yes" if m["violation"] else "no", invalid=m["invalid"],
            exercised=m["exercised"], alarms=" ".join(map(str, m["alarms"])),
            restarts=" ".join(map(str, m["restarts"]))))
    return rows


def record_lines():
    return [
        "Record of the anticipation benchmark of Chapter 16",
        "",
        "model and asset provenance",
        "  the cart of Chapter 3 (../ch03/cart_example.py), its modules of Chapter 10 "
        "(../ch10/cart), the survey's calibration of Section 7.3 (../ch07/cart_forecasts.py); "
        "the benchmark's modules ours (bench/)",
        "simulator and solver",
        f"  none: the motion is computed in exact rational arithmetic; Python "
        f"{platform.python_version()}",
        "time steps",
        f"  the governor's period 0.1 s; {PERIODS} periods",
        "actuators and sensors",
        "  the command bound 2 m/s^2; the measurement exact",
        "frames and units",
        "  SI units; the position along the track from the start",
        "uncertainty bounds",
        "  the global capability, braking of 0.5 m/s^2; the survey's offset, calibrated in "
        "Section 7.3",
        "constraints",
        "  p + l <= 25 m, the reserve floor 0.5 m; the operation's reserve 1.3 m",
        "terminal behaviour",
        "  the rest at standstill, the brakes engaged; after an alarm, maximal braking and a "
        "restart at rest",
        "configurations",
    ] + [f"  {cid}: {v[0]}" for cid, v in CONFIGS.items()] + [
        "training and evaluation split",
        "  none: no learned module; the calibration worlds of Section 7.3 (seeds 1001-1100) "
        "set the survey's offset, and the benchmark's worlds are used for nothing else",
        "seeds",
        f"  the worlds {bw.SEEDS.start} to {bw.SEEDS.stop - 1}, through random.Random(seed)",
        "hardware",
        f"  {platform.machine()}, {platform.system()}",
        "guarantee status",
        "  hypothesis A1 tests the safety theorem of Chapter 9 under its assumption, valid "
        "executed labels, assessed by the world's oracle; the other hypotheses concern useful "
        "performance and are measured, not guaranteed",
    ]


def main():
    ap = argparse.ArgumentParser(description="Chapter 16: the anticipation benchmark")
    ap.add_argument("--check", action="store_true", help="check every value of the summary")
    ap.add_argument("--jobs", type=int, default=2, help="processes (default 2)")
    args = ap.parse_args()
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    facts = {}
    same, n_periods = chapter8_consistency()
    facts["consistency with Chapter 8"] = {
        "bench.constant-command with the proposals of Chapter 3 repeats the forecast-informed "
        "run's decision file of ../ch10/results, byte for byte; periods":
            f"{'yes' if same else 'no'}; {n_periods}"}
    worlds = {w.seed: w for w in bw.family()}
    valid_seeds = {s for s, w in worlds.items() if valid(w)}
    specs = [(cid, s) for cid in CONFIGS for s in sorted(worlds)]
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            out = pool.map(run, specs, chunksize=4)
    else:
        out = [run(s) for s in specs]
    runs = {(m["config"], m["world"]): m for m, _ in out}
    traces = {(m["config"], m["world"]): tr for m, tr in out if tr is not None}
    facts["the worlds"] = world_facts(worlds, valid_seeds)
    verdicts = bh.decide(runs, worlds, valid_seeds)
    facts["the hypotheses"] = {
        f"{hid} ({name}): supported; cases; exceptions": (
            f"{'yes' if v['supported'] else 'no'}; {v['cases']}; "
            f"{'; '.join(v['exceptions'][:6]) or 'none'}"
            + (f"; and {len(v['exceptions']) - 6} more" if len(v['exceptions']) > 6 else ""))
        for hid, name, v in verdicts}
    facts["the hypotheses, by statement (reported since the first full run)"] = \
        statement_facts(verdicts, runs, valid_seeds)
    for cid, (name, *_rest) in CONFIGS.items():
        facts[f"{cid}, {name}"] = config_facts(cid, runs, worlds, valid_seeds)
    fails = failures(runs, worlds)
    facts["the failed runs"] = failure_facts(fails, worlds)
    rows, n_traced = trace_rows(traces, worlds, valid_seeds)
    facts["Figure 16.2"] = {"worlds traced": str(n_traced)}
    write_rows(os.path.join(out_dir, "benchmark-runs.csv"), run_rows(runs, worlds))
    write_rows(os.path.join(out_dir, "benchmark-failures.csv"), fails,
               ["config", "world", "valid", "lead", "slip", "what"])
    write_rows(os.path.join(out_dir, "benchmark-traces.csv"), rows)
    with open(os.path.join(out_dir, "benchmark-record.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(record_lines()) + "\n")
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
        lines.append(f"[check] {n} values, every value of the summary but the timing: " +
                     ("all as expected" if not failed and n else "FAILED: " + "; ".join(failed)))
    lines.append("[timing, this machine; not checked]")
    lines += [f"  {cid} = {v}" for cid, v in timing_facts(runs).items()]
    lines.append(f"[time] {time.time() - t0:.0f} s")
    with open(os.path.join(out_dir, "benchmark-summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines[:-1]) + "\n")
    print("\n".join(lines))
    if failed:
        raise SystemExit(1)


EXPECTED = {   # every value of the summary but the versions of the libraries
 'consistency with Chapter 8': {"bench.constant-command with the proposals of Chapter 3 repeats the forecast-informed run's decision file of ../ch10/results, byte for byte; periods": 'yes; '
                                                                                                                                                                                       '250'},
 'the worlds': {'worlds; valid; unreliable surveys': '100; 94; 20',
                'the calibrated offset of Section 7.3 [m]': '0.40',
                "the worlds that are not valid, and their surveys' errors [m]": '16042 (+0.42), '
                                                                                '16043 (+0.50), '
                                                                                '16059 (+0.50), '
                                                                                '16064 (+0.44), '
                                                                                '16077 (+0.54), '
                                                                                '16100 (+0.49)',
                "the survey's labels valid (the coverage of the calibrated offset, which claims a miscoverage of at most 5 %)": '94 '
                                                                                                                                'of '
                                                                                                                                '100 '
                                                                                                                                '(94 '
                                                                                                                                '%; '
                                                                                                                                '87.3 '
                                                                                                                                'to '
                                                                                                                                '97.8 '
                                                                                                                                '%)',
                'lead times of 21 periods or more; of 18 or fewer': '79; 18',
                'onsets that slip later; earlier': '12; 21'},
 'the hypotheses': {'A1 (the certificate): supported; cases; exceptions': 'yes; 1463; none',
                    'A2 (no demand channel): supported; cases; exceptions': 'yes; 282; none',
                    'A3 (preparation from the forecast): supported; cases; exceptions': 'yes; 256; '
                                                                                        'none',
                    'A4 (fixed conservatism): supported; cases; exceptions': 'yes; 94; none',
                    'A5 (the capability forecast): supported; cases; exceptions': 'no; 382; world '
                                                                                  '16063: begun C6 '
                                                                                  '218, C7 239',
                    'A6 (equal information): supported; cases; exceptions': 'no; 328; world 16002: '
                                                                            'C11 begun 225, '
                                                                            'arrival 90, '
                                                                            'in_work_zone 116; C6 '
                                                                            'begun 225, arrival '
                                                                            '90, in_work_zone 125; '
                                                                            'world 16006: C11 '
                                                                            'begun 229, arrival '
                                                                            '87, in_work_zone 136; '
                                                                            'C6 begun 229, arrival '
                                                                            '87, in_work_zone 137; '
                                                                            'world 16009: C11 '
                                                                            'begun 204, arrival '
                                                                            '85, in_work_zone 113; '
                                                                            'C6 begun 204, arrival '
                                                                            '85, in_work_zone 115; '
                                                                            'world 16009: C14 '
                                                                            'begun 204, arrival '
                                                                            '85, in_work_zone 115; '
                                                                            'C11 begun 204, '
                                                                            'arrival 85, '
                                                                            'in_work_zone 113; '
                                                                            'world 16010: C11 '
                                                                            'begun 229, arrival '
                                                                            '88, in_work_zone 135; '
                                                                            'C6 begun 229, arrival '
                                                                            '88, in_work_zone 136; '
                                                                            'world 16010: C14 '
                                                                            'begun 229, arrival '
                                                                            '88, in_work_zone 137; '
                                                                            'C11 begun 229, '
                                                                            'arrival 88, '
                                                                            'in_work_zone 135; and '
                                                                            '23 more',
                    'A7 (errors of the forecast): supported; cases; exceptions': 'yes; 98; none',
                    'A8 (the budget): supported; cases; exceptions': 'yes; 94; none'},
 'the hypotheses, by statement (reported since the first full run)': {'A1.1 no alarm and a true clearance of at least the floor in every run whose executed labels are valid: cases; exceptions': '1463; '
                                                                                                                                                                                                  '0',
                                                                      'A2.1 C1 to C3 never begin the operation when the cart is in the work zone at the onset: cases; exceptions': '282; '
                                                                                                                                                                                   '0',
                                                                      'A3.1 no slip, a lead time of 21 periods or more: C6, C7 and C8 on time: cases; exceptions': '156; '
                                                                                                                                                                   '0',
                                                                      'A3.2 no slip, a lead time of 18 periods or fewer: C6, C7 and C8 not on time: cases; exceptions': '36; '
                                                                                                                                                                        '0',
                                                                      'A3.3 no slip: C5 21 periods late when the cart is in the work zone at the onset: cases; exceptions': '64; '
                                                                                                                                                                            '0',
                                                                      'A4.1 C4 on time and never in the work zone: cases; exceptions': '94; '
                                                                                                                                       '0',
                                                                      'A5.1 C6 arrives before C7: cases; exceptions': '94; '
                                                                                                                      '0',
                                                                      'A5.2 C8 arrives no later than C6: cases; exceptions': '94; '
                                                                                                                             '0',
                                                                      'A5.3 C6 and C7 begin the operation in the same period: cases; exceptions': '94; '
                                                                                                                                                  '1; '
                                                                                                                                                  'worlds '
                                                                                                                                                  '16063',
                                                                      'A5.4 every world: C2 arrives in the same period as C1: cases; exceptions': '100; '
                                                                                                                                                  '0',
                                                                      "A6.1 C11 as C6: the operation's period, the arrival, the time in the work zone: cases; exceptions": '94; '
                                                                                                                                                                           '20; '
                                                                                                                                                                           'worlds '
                                                                                                                                                                           '16002, '
                                                                                                                                                                           '16006, '
                                                                                                                                                                           '16009, '
                                                                                                                                                                           '16010, '
                                                                                                                                                                           '16011, '
                                                                                                                                                                           '16012, '
                                                                                                                                                                           '16017, '
                                                                                                                                                                           '16020, '
                                                                                                                                                                           '16022, '
                                                                                                                                                                           '16023, '
                                                                                                                                                                           '16026, '
                                                                                                                                                                           '16031, '
                                                                                                                                                                           '16047, '
                                                                                                                                                                           '16066, '
                                                                                                                                                                           '16070, '
                                                                                                                                                                           '16078, '
                                                                                                                                                                           '16079, '
                                                                                                                                                                           '16083, '
                                                                                                                                                                           '16092, '
                                                                                                                                                                           '16095',
                                                                      "A6.2 C14 as C11: the operation's period, the arrival, the time in the work zone: cases; exceptions": '94; '
                                                                                                                                                                            '8; '
                                                                                                                                                                            'worlds '
                                                                                                                                                                            '16009, '
                                                                                                                                                                            '16010, '
                                                                                                                                                                            '16023, '
                                                                                                                                                                            '16026, '
                                                                                                                                                                            '16031, '
                                                                                                                                                                            '16079, '
                                                                                                                                                                            '16092, '
                                                                                                                                                                            '16095',
                                                                      'A6.3 C13 begins the operation in the same period as C6: cases; exceptions': '94; '
                                                                                                                                                   '1; '
                                                                                                                                                   'worlds '
                                                                                                                                                   '16070',
                                                                      'A6.4 no slip and a lead time of 27 periods or more: C12 begins later than C6: cases; exceptions': '46; '
                                                                                                                                                                         '0',
                                                                      'A7.1 the true onset 21 periods or more after the announcement: C9 on time: cases; exceptions': '76; '
                                                                                                                                                                      '0',
                                                                      'A7.2 a lead time of 21 periods or more and a later onset: C6 on time, 0.1 s less in the work zone than C9 per period of slip: cases; exceptions': '9; '
                                                                                                                                                                                                                         '0',
                                                                      'A7.3 a lead time of 21 periods or more and an earlier onset: C6 begins at the forecast onset: cases; exceptions': '13; '
                                                                                                                                                                                         '0',
                                                                      'A8.1 C15 begins the operation in the same period as C6: cases; exceptions': '94; '
                                                                                                                                                   '0',
                                                                      "A6: valid worlds in which C11 differs from C6 in the operation's period; the arrival; the time in the work zone": '1; '
                                                                                                                                                                                         '0; '
                                                                                                                                                                                         '20',
                                                                      "A6: valid worlds in which C14 differs from C11 in the operation's period; the arrival; the time in the work zone": '0; '
                                                                                                                                                                                          '0; '
                                                                                                                                                                                          '8',
                                                                      'A6: periods in the work zone, C6 less C11: value (valid worlds)': '0 '
                                                                                                                                         '(74), '
                                                                                                                                         '1 '
                                                                                                                                         '(13), '
                                                                                                                                         '2 '
                                                                                                                                         '(3), '
                                                                                                                                         '8 '
                                                                                                                                         '(1), '
                                                                                                                                         '9 '
                                                                                                                                         '(2), '
                                                                                                                                         '20 '
                                                                                                                                         '(1)',
                                                                      'A6: periods in the work zone, C14 less C11: value (valid worlds)': '0 '
                                                                                                                                          '(86), '
                                                                                                                                          '2 '
                                                                                                                                          '(8)',
                                                                      'A6: periods in the work zone, C14 less C6: value (valid worlds)': '-20 '
                                                                                                                                         '(1), '
                                                                                                                                         '-9 '
                                                                                                                                         '(2), '
                                                                                                                                         '-8 '
                                                                                                                                         '(1), '
                                                                                                                                         '-1 '
                                                                                                                                         '(8), '
                                                                                                                                         '0 '
                                                                                                                                         '(77), '
                                                                                                                                         '1 '
                                                                                                                                         '(5)'},
 'C1, filter, fixed model': {'valid worlds: the operation on time': '0 of 94 (0 %; 0.0 to 3.9 %)',
                             'valid worlds: the operation late, by [s]': 'none',
                             'valid worlds: the operation never begun': '94 of 94',
                             'valid worlds: arrival in the work zone [s]': '9.9 s (95 % interval '
                                                                           '9.9 to 9.9; range 9.9 '
                                                                           'to 9.9; n = 94)',
                             'valid worlds: time in the work zone [s]': '22.1 s (95 % interval '
                                                                        '22.1 to 22.1; range 22.1 '
                                                                        'to 22.1; n = 94)',
                             'valid worlds: periods after arrival, before the operation, committed with an ambition below 1': '12 '
                                                                                                                              '(95 '
                                                                                                                              '% '
                                                                                                                              'interval '
                                                                                                                              '12 '
                                                                                                                              'to '
                                                                                                                              '12; '
                                                                                                                              'range '
                                                                                                                              '12 '
                                                                                                                              'to '
                                                                                                                              '12; '
                                                                                                                              'n '
                                                                                                                              '= '
                                                                                                                              '94)',
                             'valid worlds: periods after arrival, before the operation, in which the executed command is not the proposed one': '12 '
                                                                                                                                                 '(95 '
                                                                                                                                                 '% '
                                                                                                                                                 'interval '
                                                                                                                                                 '12 '
                                                                                                                                                 'to '
                                                                                                                                                 '12; '
                                                                                                                                                 'range '
                                                                                                                                                 '12 '
                                                                                                                                                 'to '
                                                                                                                                                 '12; '
                                                                                                                                                 'n '
                                                                                                                                                 '= '
                                                                                                                                                 '94)',
                             'all worlds: runs with an invalid executed label; with an alarm; restarts; violations': '0; '
                                                                                                                     '0; '
                                                                                                                     '0; '
                                                                                                                     '0 '
                                                                                                                     'of '
                                                                                                                     '100',
                             'all worlds: the least true clearance in runs whose executed labels are valid [m]': '0.51',
                             'all worlds: fresh candidates evaluated, of those offered; periods cut by the budget': '192000 '
                                                                                                                    'of '
                                                                                                                    '192000; '
                                                                                                                    '0'},
 'C2, filter, measured model': {'valid worlds: the operation on time': '0 of 94 (0 %; 0.0 to 3.9 '
                                                                       '%)',
                                'valid worlds: the operation late, by [s]': 'none',
                                'valid worlds: the operation never begun': '94 of 94',
                                'valid worlds: arrival in the work zone [s]': '9.9 s (95 % '
                                                                              'interval 9.9 to '
                                                                              '9.9; range 9.9 to '
                                                                              '9.9; n = 94)',
                                'valid worlds: time in the work zone [s]': '22.1 s (95 % interval '
                                                                           '22.1 to 22.1; range '
                                                                           '22.1 to 22.1; n = 94)',
                                'valid worlds: periods after arrival, before the operation, committed with an ambition below 1': '12 '
                                                                                                                                 '(95 '
                                                                                                                                 '% '
                                                                                                                                 'interval '
                                                                                                                                 '12 '
                                                                                                                                 'to '
                                                                                                                                 '12; '
                                                                                                                                 'range '
                                                                                                                                 '12 '
                                                                                                                                 'to '
                                                                                                                                 '12; '
                                                                                                                                 'n '
                                                                                                                                 '= '
                                                                                                                                 '94)',
                                'valid worlds: periods after arrival, before the operation, in which the executed command is not the proposed one': '12 '
                                                                                                                                                    '(95 '
                                                                                                                                                    '% '
                                                                                                                                                    'interval '
                                                                                                                                                    '12 '
                                                                                                                                                    'to '
                                                                                                                                                    '12; '
                                                                                                                                                    'range '
                                                                                                                                                    '12 '
                                                                                                                                                    'to '
                                                                                                                                                    '12; '
                                                                                                                                                    'n '
                                                                                                                                                    '= '
                                                                                                                                                    '94)',
                                'all worlds: runs with an invalid executed label; with an alarm; restarts; violations': '0; '
                                                                                                                        '0; '
                                                                                                                        '0; '
                                                                                                                        '0 '
                                                                                                                        'of '
                                                                                                                        '100',
                                'all worlds: the least true clearance in runs whose executed labels are valid [m]': '0.51',
                                'all worlds: fresh candidates evaluated, of those offered; periods cut by the budget': '192000 '
                                                                                                                       'of '
                                                                                                                       '192000; '
                                                                                                                       '0'},
 'C3, filter, survey': {'valid worlds: the operation on time': '0 of 94 (0 %; 0.0 to 3.9 %)',
                        'valid worlds: the operation late, by [s]': 'none',
                        'valid worlds: the operation never begun': '94 of 94',
                        'valid worlds: arrival in the work zone [s]': '8.9 s (95 % interval 8.8 to '
                                                                      '9.0; range 8.4 to 9.2; n = '
                                                                      '94)',
                        'valid worlds: time in the work zone [s]': '23.1 s (95 % interval 23.0 to '
                                                                   '23.2; range 22.8 to 23.6; n = '
                                                                   '94)',
                        'valid worlds: periods after arrival, before the operation, committed with an ambition below 1': '12 '
                                                                                                                         '(95 '
                                                                                                                         '% '
                                                                                                                         'interval '
                                                                                                                         '12 '
                                                                                                                         'to '
                                                                                                                         '12; '
                                                                                                                         'range '
                                                                                                                         '11 '
                                                                                                                         'to '
                                                                                                                         '12; '
                                                                                                                         'n '
                                                                                                                         '= '
                                                                                                                         '94)',
                        'valid worlds: periods after arrival, before the operation, in which the executed command is not the proposed one': '12 '
                                                                                                                                            '(95 '
                                                                                                                                            '% '
                                                                                                                                            'interval '
                                                                                                                                            '12 '
                                                                                                                                            'to '
                                                                                                                                            '12; '
                                                                                                                                            'range '
                                                                                                                                            '11 '
                                                                                                                                            'to '
                                                                                                                                            '12; '
                                                                                                                                            'n '
                                                                                                                                            '= '
                                                                                                                                            '94)',
                        'all worlds: runs with an invalid executed label; with an alarm; restarts; violations': '4; '
                                                                                                                '4; '
                                                                                                                '3; '
                                                                                                                '0 '
                                                                                                                'of '
                                                                                                                '100',
                        'all worlds: the least true clearance in runs whose executed labels are valid [m]': '0.50',
                        'all worlds: fresh candidates evaluated, of those offered; periods cut by the budget': '189540 '
                                                                                                               'of '
                                                                                                               '189540; '
                                                                                                               '0'},
 'C4, fixed conservatism': {'valid worlds: the operation on time': '94 of 94 (100 %; 96.1 to 100.0 '
                                                                   '%)',
                            'valid worlds: the operation late, by [s]': 'none',
                            'valid worlds: the operation never begun': '0 of 94',
                            'valid worlds: arrival in the work zone [s]': 'none; never in 94',
                            'valid worlds: time in the work zone [s]': '0.0 s (95 % interval 0.0 '
                                                                       'to 0.0; range 0.0 to 0.0; '
                                                                       'n = 94)',
                            'valid worlds: periods after arrival, before the operation, committed with an ambition below 1': 'none',
                            'valid worlds: periods after arrival, before the operation, in which the executed command is not the proposed one': 'none',
                            'all worlds: runs with an invalid executed label; with an alarm; restarts; violations': '3; '
                                                                                                                    '3; '
                                                                                                                    '3; '
                                                                                                                    '0 '
                                                                                                                    'of '
                                                                                                                    '100',
                            'all worlds: the least true clearance in runs whose executed labels are valid [m]': '0.50',
                            'all worlds: fresh candidates evaluated, of those offered; periods cut by the budget': '191220 '
                                                                                                                   'of '
                                                                                                                   '191220; '
                                                                                                                   '0'},
 'C5, reactive preparation': {'valid worlds: the operation on time': '0 of 94 (0 %; 0.0 to 3.9 %)',
                              'valid worlds: the operation late, by [s]': '2.1 s (95 % interval '
                                                                          '2.1 to 2.1; range 2.1 '
                                                                          'to 2.1; n = 94)',
                              'valid worlds: the operation never begun': '0 of 94',
                              'valid worlds: arrival in the work zone [s]': '8.9 s (95 % interval '
                                                                            '8.8 to 9.0; range 8.4 '
                                                                            'to 9.2; n = 94)',
                              'valid worlds: time in the work zone [s]': '14.8 s (95 % interval '
                                                                         '14.4 to 15.0; range 12.5 '
                                                                         'to 17.3; n = 94)',
                              'valid worlds: periods after arrival, before the operation, committed with an ambition below 1': '28 '
                                                                                                                               '(95 '
                                                                                                                               '% '
                                                                                                                               'interval '
                                                                                                                               '28 '
                                                                                                                               'to '
                                                                                                                               '28; '
                                                                                                                               'range '
                                                                                                                               '27 '
                                                                                                                               'to '
                                                                                                                               '28; '
                                                                                                                               'n '
                                                                                                                               '= '
                                                                                                                               '94)',
                              'valid worlds: periods after arrival, before the operation, in which the executed command is not the proposed one': '28 '
                                                                                                                                                  '(95 '
                                                                                                                                                  '% '
                                                                                                                                                  'interval '
                                                                                                                                                  '28 '
                                                                                                                                                  'to '
                                                                                                                                                  '28; '
                                                                                                                                                  'range '
                                                                                                                                                  '27 '
                                                                                                                                                  'to '
                                                                                                                                                  '28; '
                                                                                                                                                  'n '
                                                                                                                                                  '= '
                                                                                                                                                  '94)',
                              'all worlds: runs with an invalid executed label; with an alarm; restarts; violations': '4; '
                                                                                                                      '4; '
                                                                                                                      '3; '
                                                                                                                      '0 '
                                                                                                                      'of '
                                                                                                                      '100',
                              'all worlds: the least true clearance in runs whose executed labels are valid [m]': '0.50',
                              'all worlds: fresh candidates evaluated, of those offered; periods cut by the budget': '189540 '
                                                                                                                     'of '
                                                                                                                     '189540; '
                                                                                                                     '0'},
 'C6, ALLOVI': {'valid worlds: the operation on time': '62 of 94 (66 %; 55.4 to 75.5 %)',
                'valid worlds: the operation late, by [s]': '2.1 s (95 % interval 0.6 to 2.1; '
                                                            'range 0.1 to 2.8; n = 32)',
                'valid worlds: the operation never begun': '0 of 94',
                'valid worlds: arrival in the work zone [s]': '8.9 s (95 % interval 8.8 to 9.0; '
                                                              'range 8.4 to 9.2; n = 94)',
                'valid worlds: time in the work zone [s]': '12.8 s (95 % interval 12.4 to 13.3; '
                                                           'range 10.4 to 15.0; n = 94)',
                'valid worlds: periods after arrival, before the operation, committed with an ambition below 1': '28 '
                                                                                                                 '(95 '
                                                                                                                 '% '
                                                                                                                 'interval '
                                                                                                                 '28 '
                                                                                                                 'to '
                                                                                                                 '28; '
                                                                                                                 'range '
                                                                                                                 '27 '
                                                                                                                 'to '
                                                                                                                 '35; '
                                                                                                                 'n '
                                                                                                                 '= '
                                                                                                                 '94)',
                'valid worlds: periods after arrival, before the operation, in which the executed command is not the proposed one': '28 '
                                                                                                                                    '(95 '
                                                                                                                                    '% '
                                                                                                                                    'interval '
                                                                                                                                    '28 '
                                                                                                                                    'to '
                                                                                                                                    '28; '
                                                                                                                                    'range '
                                                                                                                                    '27 '
                                                                                                                                    'to '
                                                                                                                                    '35; '
                                                                                                                                    'n '
                                                                                                                                    '= '
                                                                                                                                    '94)',
                'all worlds: runs with an invalid executed label; with an alarm; restarts; violations': '4; '
                                                                                                        '4; '
                                                                                                        '3; '
                                                                                                        '0 '
                                                                                                        'of '
                                                                                                        '100',
                'all worlds: the least true clearance in runs whose executed labels are valid [m]': '0.50',
                'all worlds: fresh candidates evaluated, of those offered; periods cut by the budget': '189540 '
                                                                                                       'of '
                                                                                                       '189540; '
                                                                                                       '0'},
 'C7, ALLOVI without the capability forecast': {'valid worlds: the operation on time': '62 of 94 '
                                                                                       '(66 %; '
                                                                                       '55.4 to '
                                                                                       '75.5 %)',
                                                'valid worlds: the operation late, by [s]': '2.1 s '
                                                                                            '(95 % '
                                                                                            'interval '
                                                                                            '0.7 '
                                                                                            'to '
                                                                                            '2.1; '
                                                                                            'range '
                                                                                            '0.1 '
                                                                                            'to '
                                                                                            '2.8; '
                                                                                            'n = '
                                                                                            '32)',
                                                'valid worlds: the operation never begun': '0 of '
                                                                                           '94',
                                                'valid worlds: arrival in the work zone [s]': '9.9 '
                                                                                              's '
                                                                                              '(95 '
                                                                                              '% '
                                                                                              'interval '
                                                                                              '9.9 '
                                                                                              'to '
                                                                                              '9.9; '
                                                                                              'range '
                                                                                              '9.9 '
                                                                                              'to '
                                                                                              '9.9; '
                                                                                              'n = '
                                                                                              '94)',
                                                'valid worlds: time in the work zone [s]': '11.8 s '
                                                                                           '(95 % '
                                                                                           'interval '
                                                                                           '11.4 '
                                                                                           'to '
                                                                                           '12.0; '
                                                                                           'range '
                                                                                           '9.6 to '
                                                                                           '13.6; '
                                                                                           'n = '
                                                                                           '94)',
                                                'valid worlds: periods after arrival, before the operation, committed with an ambition below 1': '28 '
                                                                                                                                                 '(95 '
                                                                                                                                                 '% '
                                                                                                                                                 'interval '
                                                                                                                                                 '28 '
                                                                                                                                                 'to '
                                                                                                                                                 '28; '
                                                                                                                                                 'range '
                                                                                                                                                 '28 '
                                                                                                                                                 'to '
                                                                                                                                                 '35; '
                                                                                                                                                 'n '
                                                                                                                                                 '= '
                                                                                                                                                 '94)',
                                                'valid worlds: periods after arrival, before the operation, in which the executed command is not the proposed one': '28 '
                                                                                                                                                                    '(95 '
                                                                                                                                                                    '% '
                                                                                                                                                                    'interval '
                                                                                                                                                                    '28 '
                                                                                                                                                                    'to '
                                                                                                                                                                    '28; '
                                                                                                                                                                    'range '
                                                                                                                                                                    '28 '
                                                                                                                                                                    'to '
                                                                                                                                                                    '35; '
                                                                                                                                                                    'n '
                                                                                                                                                                    '= '
                                                                                                                                                                    '94)',
                                                'all worlds: runs with an invalid executed label; with an alarm; restarts; violations': '0; '
                                                                                                                                        '0; '
                                                                                                                                        '0; '
                                                                                                                                        '0 '
                                                                                                                                        'of '
                                                                                                                                        '100',
                                                'all worlds: the least true clearance in runs whose executed labels are valid [m]': '0.51',
                                                'all worlds: fresh candidates evaluated, of those offered; periods cut by the budget': '192000 '
                                                                                                                                       'of '
                                                                                                                                       '192000; '
                                                                                                                                       '0'},
 'C8, oracle capability (diagnostic)': {'valid worlds: the operation on time': '62 of 94 (66 %; '
                                                                               '55.4 to 75.5 %)',
                                        'valid worlds: the operation late, by [s]': '2.1 s (95 % '
                                                                                    'interval 0.7 '
                                                                                    'to 2.1; range '
                                                                                    '0.1 to 2.8; n '
                                                                                    '= 32)',
                                        'valid worlds: the operation never begun': '0 of 94',
                                        'valid worlds: arrival in the work zone [s]': '8.8 s (95 % '
                                                                                      'interval '
                                                                                      '8.7 to 8.9; '
                                                                                      'range 8.3 '
                                                                                      'to 9.1; n = '
                                                                                      '94)',
                                        'valid worlds: time in the work zone [s]': '12.8 s (95 % '
                                                                                   'interval 12.4 '
                                                                                   'to 13.3; range '
                                                                                   '10.5 to 15.1; '
                                                                                   'n = 94)',
                                        'valid worlds: periods after arrival, before the operation, committed with an ambition below 1': '28 '
                                                                                                                                         '(95 '
                                                                                                                                         '% '
                                                                                                                                         'interval '
                                                                                                                                         '28 '
                                                                                                                                         'to '
                                                                                                                                         '28; '
                                                                                                                                         'range '
                                                                                                                                         '27 '
                                                                                                                                         'to '
                                                                                                                                         '35; '
                                                                                                                                         'n '
                                                                                                                                         '= '
                                                                                                                                         '94)',
                                        'valid worlds: periods after arrival, before the operation, in which the executed command is not the proposed one': '28 '
                                                                                                                                                            '(95 '
                                                                                                                                                            '% '
                                                                                                                                                            'interval '
                                                                                                                                                            '28 '
                                                                                                                                                            'to '
                                                                                                                                                            '28; '
                                                                                                                                                            'range '
                                                                                                                                                            '27 '
                                                                                                                                                            'to '
                                                                                                                                                            '35; '
                                                                                                                                                            'n '
                                                                                                                                                            '= '
                                                                                                                                                            '94)',
                                        'all worlds: runs with an invalid executed label; with an alarm; restarts; violations': '0; '
                                                                                                                                '0; '
                                                                                                                                '0; '
                                                                                                                                '0 '
                                                                                                                                'of '
                                                                                                                                '100',
                                        'all worlds: the least true clearance in runs whose executed labels are valid [m]': '0.50',
                                        'all worlds: fresh candidates evaluated, of those offered; periods cut by the budget': '192000 '
                                                                                                                               'of '
                                                                                                                               '192000; '
                                                                                                                               '0'},
 'C9, oracle demand (diagnostic)': {'valid worlds: the operation on time': '76 of 94 (81 %; 71.4 '
                                                                           'to 88.3 %)',
                                    'valid worlds: the operation late, by [s]': '2.1 s (95 % '
                                                                                'interval 2.1 to '
                                                                                '2.1; range 2.1 to '
                                                                                '2.1; n = 18)',
                                    'valid worlds: the operation never begun': '0 of 94',
                                    'valid worlds: arrival in the work zone [s]': '8.9 s (95 % '
                                                                                  'interval 8.8 to '
                                                                                  '9.0; range 8.4 '
                                                                                  'to 9.2; n = 94)',
                                    'valid worlds: time in the work zone [s]': '12.7 s (95 % '
                                                                               'interval 12.5 to '
                                                                               '13.0; range 10.4 '
                                                                               'to 15.2; n = 94)',
                                    'valid worlds: periods after arrival, before the operation, committed with an ambition below 1': '28 '
                                                                                                                                     '(95 '
                                                                                                                                     '% '
                                                                                                                                     'interval '
                                                                                                                                     '28 '
                                                                                                                                     'to '
                                                                                                                                     '28; '
                                                                                                                                     'range '
                                                                                                                                     '27 '
                                                                                                                                     'to '
                                                                                                                                     '36; '
                                                                                                                                     'n '
                                                                                                                                     '= '
                                                                                                                                     '94)',
                                    'valid worlds: periods after arrival, before the operation, in which the executed command is not the proposed one': '28 '
                                                                                                                                                        '(95 '
                                                                                                                                                        '% '
                                                                                                                                                        'interval '
                                                                                                                                                        '28 '
                                                                                                                                                        'to '
                                                                                                                                                        '28; '
                                                                                                                                                        'range '
                                                                                                                                                        '27 '
                                                                                                                                                        'to '
                                                                                                                                                        '36; '
                                                                                                                                                        'n '
                                                                                                                                                        '= '
                                                                                                                                                        '94)',
                                    'all worlds: runs with an invalid executed label; with an alarm; restarts; violations': '4; '
                                                                                                                            '4; '
                                                                                                                            '3; '
                                                                                                                            '0 '
                                                                                                                            'of '
                                                                                                                            '100',
                                    'all worlds: the least true clearance in runs whose executed labels are valid [m]': '0.50',
                                    'all worlds: fresh candidates evaluated, of those offered; periods cut by the budget': '189540 '
                                                                                                                           'of '
                                                                                                                           '189540; '
                                                                                                                           '0'},
 'C10, both oracles (diagnostic)': {'valid worlds: the operation on time': '76 of 94 (81 %; 71.4 '
                                                                           'to 88.3 %)',
                                    'valid worlds: the operation late, by [s]': '2.1 s (95 % '
                                                                                'interval 2.1 to '
                                                                                '2.1; range 2.1 to '
                                                                                '2.1; n = 18)',
                                    'valid worlds: the operation never begun': '0 of 94',
                                    'valid worlds: arrival in the work zone [s]': '8.8 s (95 % '
                                                                                  'interval 8.7 to '
                                                                                  '8.9; range 8.3 '
                                                                                  'to 9.1; n = 94)',
                                    'valid worlds: time in the work zone [s]': '12.8 s (95 % '
                                                                               'interval 12.5 to '
                                                                               '13.1; range 10.4 '
                                                                               'to 15.3; n = 94)',
                                    'valid worlds: periods after arrival, before the operation, committed with an ambition below 1': '28 '
                                                                                                                                     '(95 '
                                                                                                                                     '% '
                                                                                                                                     'interval '
                                                                                                                                     '28 '
                                                                                                                                     'to '
                                                                                                                                     '28; '
                                                                                                                                     'range '
                                                                                                                                     '27 '
                                                                                                                                     'to '
                                                                                                                                     '35; '
                                                                                                                                     'n '
                                                                                                                                     '= '
                                                                                                                                     '94)',
                                    'valid worlds: periods after arrival, before the operation, in which the executed command is not the proposed one': '28 '
                                                                                                                                                        '(95 '
                                                                                                                                                        '% '
                                                                                                                                                        'interval '
                                                                                                                                                        '28 '
                                                                                                                                                        'to '
                                                                                                                                                        '28; '
                                                                                                                                                        'range '
                                                                                                                                                        '27 '
                                                                                                                                                        'to '
                                                                                                                                                        '35; '
                                                                                                                                                        'n '
                                                                                                                                                        '= '
                                                                                                                                                        '94)',
                                    'all worlds: runs with an invalid executed label; with an alarm; restarts; violations': '0; '
                                                                                                                            '0; '
                                                                                                                            '0; '
                                                                                                                            '0 '
                                                                                                                            'of '
                                                                                                                            '100',
                                    'all worlds: the least true clearance in runs whose executed labels are valid [m]': '0.50',
                                    'all worlds: fresh candidates evaluated, of those offered; periods cut by the budget': '192000 '
                                                                                                                           'of '
                                                                                                                           '192000; '
                                                                                                                           '0'},
 'C11, filter, survey, preview planner': {'valid worlds: the operation on time': '63 of 94 (67 %; '
                                                                                 '56.5 to 76.4 %)',
                                          'valid worlds: the operation late, by [s]': '2.1 s (95 % '
                                                                                      'interval '
                                                                                      '0.6 to 2.1; '
                                                                                      'range 0.1 '
                                                                                      'to 2.8; n = '
                                                                                      '31)',
                                          'valid worlds: the operation never begun': '0 of 94',
                                          'valid worlds: arrival in the work zone [s]': '8.9 s (95 '
                                                                                        '% '
                                                                                        'interval '
                                                                                        '8.8 to '
                                                                                        '9.0; '
                                                                                        'range 8.4 '
                                                                                        'to 9.2; n '
                                                                                        '= 94)',
                                          'valid worlds: time in the work zone [s]': '12.7 s (95 % '
                                                                                     'interval '
                                                                                     '12.3 to '
                                                                                     '13.1; range '
                                                                                     '10.4 to '
                                                                                     '15.0; n = '
                                                                                     '94)',
                                          'valid worlds: periods after arrival, before the operation, committed with an ambition below 1': '12 '
                                                                                                                                           '(95 '
                                                                                                                                           '% '
                                                                                                                                           'interval '
                                                                                                                                           '12 '
                                                                                                                                           'to '
                                                                                                                                           '12; '
                                                                                                                                           'range '
                                                                                                                                           '11 '
                                                                                                                                           'to '
                                                                                                                                           '12; '
                                                                                                                                           'n '
                                                                                                                                           '= '
                                                                                                                                           '94)',
                                          'valid worlds: periods after arrival, before the operation, in which the executed command is not the proposed one': '12 '
                                                                                                                                                              '(95 '
                                                                                                                                                              '% '
                                                                                                                                                              'interval '
                                                                                                                                                              '12 '
                                                                                                                                                              'to '
                                                                                                                                                              '12; '
                                                                                                                                                              'range '
                                                                                                                                                              '11 '
                                                                                                                                                              'to '
                                                                                                                                                              '12; '
                                                                                                                                                              'n '
                                                                                                                                                              '= '
                                                                                                                                                              '94)',
                                          'all worlds: runs with an invalid executed label; with an alarm; restarts; violations': '4; '
                                                                                                                                  '4; '
                                                                                                                                  '3; '
                                                                                                                                  '0 '
                                                                                                                                  'of '
                                                                                                                                  '100',
                                          'all worlds: the least true clearance in runs whose executed labels are valid [m]': '0.50',
                                          'all worlds: fresh candidates evaluated, of those offered; periods cut by the budget': '189540 '
                                                                                                                                 'of '
                                                                                                                                 '189540; '
                                                                                                                                 '0'},
 'C12, filter, survey, late preview planner': {'valid worlds: the operation on time': '4 of 94 (4 '
                                                                                      '%; 1.1 to '
                                                                                      '10.6 %)',
                                               'valid worlds: the operation late, by [s]': '2.1 s '
                                                                                           '(95 % '
                                                                                           'interval '
                                                                                           '2.1 to '
                                                                                           '2.1; '
                                                                                           'range '
                                                                                           '2.1 to '
                                                                                           '3.0; n '
                                                                                           '= 90)',
                                               'valid worlds: the operation never begun': '0 of 94',
                                               'valid worlds: arrival in the work zone [s]': '8.9 '
                                                                                             's '
                                                                                             '(95 '
                                                                                             '% '
                                                                                             'interval '
                                                                                             '8.8 '
                                                                                             'to '
                                                                                             '9.0; '
                                                                                             'range '
                                                                                             '8.4 '
                                                                                             'to '
                                                                                             '9.2; '
                                                                                             'n = '
                                                                                             '94)',
                                               'valid worlds: time in the work zone [s]': '13.3 s '
                                                                                          '(95 % '
                                                                                          'interval '
                                                                                          '12.9 to '
                                                                                          '13.6; '
                                                                                          'range '
                                                                                          '11.0 to '
                                                                                          '15.5; n '
                                                                                          '= 94)',
                                               'valid worlds: periods after arrival, before the operation, committed with an ambition below 1': '12 '
                                                                                                                                                '(95 '
                                                                                                                                                '% '
                                                                                                                                                'interval '
                                                                                                                                                '12 '
                                                                                                                                                'to '
                                                                                                                                                '12; '
                                                                                                                                                'range '
                                                                                                                                                '11 '
                                                                                                                                                'to '
                                                                                                                                                '12; '
                                                                                                                                                'n '
                                                                                                                                                '= '
                                                                                                                                                '94)',
                                               'valid worlds: periods after arrival, before the operation, in which the executed command is not the proposed one': '12 '
                                                                                                                                                                   '(95 '
                                                                                                                                                                   '% '
                                                                                                                                                                   'interval '
                                                                                                                                                                   '12 '
                                                                                                                                                                   'to '
                                                                                                                                                                   '12; '
                                                                                                                                                                   'range '
                                                                                                                                                                   '11 '
                                                                                                                                                                   'to '
                                                                                                                                                                   '12; '
                                                                                                                                                                   'n '
                                                                                                                                                                   '= '
                                                                                                                                                                   '94)',
                                               'all worlds: runs with an invalid executed label; with an alarm; restarts; violations': '4; '
                                                                                                                                       '4; '
                                                                                                                                       '3; '
                                                                                                                                       '0 '
                                                                                                                                       'of '
                                                                                                                                       '100',
                                               'all worlds: the least true clearance in runs whose executed labels are valid [m]': '0.50',
                                               'all worlds: fresh candidates evaluated, of those offered; periods cut by the budget': '189540 '
                                                                                                                                      'of '
                                                                                                                                      '189540; '
                                                                                                                                      '0'},
 'C13, ALLOVI, late preview planner': {'valid worlds: the operation on time': '63 of 94 (67 %; '
                                                                              '56.5 to 76.4 %)',
                                       'valid worlds: the operation late, by [s]': '2.1 s (95 % '
                                                                                   'interval 0.6 '
                                                                                   'to 2.1; range '
                                                                                   '0.1 to 2.8; n '
                                                                                   '= 31)',
                                       'valid worlds: the operation never begun': '0 of 94',
                                       'valid worlds: arrival in the work zone [s]': '8.9 s (95 % '
                                                                                     'interval 8.8 '
                                                                                     'to 9.0; '
                                                                                     'range 8.4 to '
                                                                                     '9.2; n = 94)',
                                       'valid worlds: time in the work zone [s]': '12.7 s (95 % '
                                                                                  'interval 12.3 '
                                                                                  'to 13.1; range '
                                                                                  '10.4 to 15.0; n '
                                                                                  '= 94)',
                                       'valid worlds: periods after arrival, before the operation, committed with an ambition below 1': '28 '
                                                                                                                                        '(95 '
                                                                                                                                        '% '
                                                                                                                                        'interval '
                                                                                                                                        '28 '
                                                                                                                                        'to '
                                                                                                                                        '28; '
                                                                                                                                        'range '
                                                                                                                                        '21 '
                                                                                                                                        'to '
                                                                                                                                        '31; '
                                                                                                                                        'n '
                                                                                                                                        '= '
                                                                                                                                        '94)',
                                       'valid worlds: periods after arrival, before the operation, in which the executed command is not the proposed one': '28 '
                                                                                                                                                           '(95 '
                                                                                                                                                           '% '
                                                                                                                                                           'interval '
                                                                                                                                                           '28 '
                                                                                                                                                           'to '
                                                                                                                                                           '28; '
                                                                                                                                                           'range '
                                                                                                                                                           '11 '
                                                                                                                                                           'to '
                                                                                                                                                           '31; '
                                                                                                                                                           'n '
                                                                                                                                                           '= '
                                                                                                                                                           '94)',
                                       'all worlds: runs with an invalid executed label; with an alarm; restarts; violations': '4; '
                                                                                                                               '4; '
                                                                                                                               '3; '
                                                                                                                               '0 '
                                                                                                                               'of '
                                                                                                                               '100',
                                       'all worlds: the least true clearance in runs whose executed labels are valid [m]': '0.50',
                                       'all worlds: fresh candidates evaluated, of those offered; periods cut by the budget': '189540 '
                                                                                                                              'of '
                                                                                                                              '189540; '
                                                                                                                              '0'},
 'C14, ALLOVI, preview planner': {'valid worlds: the operation on time': '63 of 94 (67 %; 56.5 to '
                                                                         '76.4 %)',
                                  'valid worlds: the operation late, by [s]': '2.1 s (95 % '
                                                                              'interval 0.6 to '
                                                                              '2.1; range 0.1 to '
                                                                              '2.8; n = 31)',
                                  'valid worlds: the operation never begun': '0 of 94',
                                  'valid worlds: arrival in the work zone [s]': '8.9 s (95 % '
                                                                                'interval 8.8 to '
                                                                                '9.0; range 8.4 to '
                                                                                '9.2; n = 94)',
                                  'valid worlds: time in the work zone [s]': '12.8 s (95 % '
                                                                             'interval 12.3 to '
                                                                             '13.1; range 10.4 to '
                                                                             '15.0; n = 94)',
                                  'valid worlds: periods after arrival, before the operation, committed with an ambition below 1': '27 '
                                                                                                                                   '(95 '
                                                                                                                                   '% '
                                                                                                                                   'interval '
                                                                                                                                   '27 '
                                                                                                                                   'to '
                                                                                                                                   '27; '
                                                                                                                                   'range '
                                                                                                                                   '21 '
                                                                                                                                   'to '
                                                                                                                                   '28; '
                                                                                                                                   'n '
                                                                                                                                   '= '
                                                                                                                                   '94)',
                                  'valid worlds: periods after arrival, before the operation, in which the executed command is not the proposed one': '12 '
                                                                                                                                                      '(95 '
                                                                                                                                                      '% '
                                                                                                                                                      'interval '
                                                                                                                                                      '12 '
                                                                                                                                                      'to '
                                                                                                                                                      '12; '
                                                                                                                                                      'range '
                                                                                                                                                      '11 '
                                                                                                                                                      'to '
                                                                                                                                                      '13; '
                                                                                                                                                      'n '
                                                                                                                                                      '= '
                                                                                                                                                      '94)',
                                  'all worlds: runs with an invalid executed label; with an alarm; restarts; violations': '4; '
                                                                                                                          '4; '
                                                                                                                          '3; '
                                                                                                                          '0 '
                                                                                                                          'of '
                                                                                                                          '100',
                                  'all worlds: the least true clearance in runs whose executed labels are valid [m]': '0.50',
                                  'all worlds: fresh candidates evaluated, of those offered; periods cut by the budget': '189540 '
                                                                                                                         'of '
                                                                                                                         '189540; '
                                                                                                                         '0'},
 'C15, ALLOVI, one candidate per period': {'valid worlds: the operation on time': '62 of 94 (66 %; '
                                                                                  '55.4 to 75.5 %)',
                                           'valid worlds: the operation late, by [s]': '2.1 s (95 '
                                                                                       '% interval '
                                                                                       '0.6 to '
                                                                                       '2.1; range '
                                                                                       '0.1 to '
                                                                                       '2.8; n = '
                                                                                       '32)',
                                           'valid worlds: the operation never begun': '0 of 94',
                                           'valid worlds: arrival in the work zone [s]': '9.0 s '
                                                                                         '(95 % '
                                                                                         'interval '
                                                                                         '9.0 to '
                                                                                         '9.1; '
                                                                                         'range '
                                                                                         '8.3 to '
                                                                                         '9.4; n = '
                                                                                         '94)',
                                           'valid worlds: time in the work zone [s]': '12.6 s (95 '
                                                                                      '% interval '
                                                                                      '12.2 to '
                                                                                      '13.0; range '
                                                                                      '10.2 to '
                                                                                      '14.8; n = '
                                                                                      '94)',
                                           'valid worlds: periods after arrival, before the operation, committed with an ambition below 1': '28 '
                                                                                                                                            '(95 '
                                                                                                                                            '% '
                                                                                                                                            'interval '
                                                                                                                                            '27 '
                                                                                                                                            'to '
                                                                                                                                            '28; '
                                                                                                                                            'range '
                                                                                                                                            '22 '
                                                                                                                                            'to '
                                                                                                                                            '35; '
                                                                                                                                            'n '
                                                                                                                                            '= '
                                                                                                                                            '94)',
                                           'valid worlds: periods after arrival, before the operation, in which the executed command is not the proposed one': '28 '
                                                                                                                                                               '(95 '
                                                                                                                                                               '% '
                                                                                                                                                               'interval '
                                                                                                                                                               '27 '
                                                                                                                                                               'to '
                                                                                                                                                               '28; '
                                                                                                                                                               'range '
                                                                                                                                                               '22 '
                                                                                                                                                               'to '
                                                                                                                                                               '35; '
                                                                                                                                                               'n '
                                                                                                                                                               '= '
                                                                                                                                                               '94)',
                                           'all worlds: runs with an invalid executed label; with an alarm; restarts; violations': '2; '
                                                                                                                                   '1; '
                                                                                                                                   '1; '
                                                                                                                                   '0 '
                                                                                                                                   'of '
                                                                                                                                   '100',
                                           'all worlds: the least true clearance in runs whose executed labels are valid [m]': '0.50',
                                           'all worlds: fresh candidates evaluated, of those offered; periods cut by the budget': '31953 '
                                                                                                                                  'of '
                                                                                                                                  '191718; '
                                                                                                                                  '31953'},
 'the failed runs': {'C1: failed runs; never begun, late; with invalid executed labels (worlds); with alarms (worlds)': '100; '
                                                                                                                        '100, '
                                                                                                                        '0; '
                                                                                                                        'none; '
                                                                                                                        'none',
                     'C2: failed runs; never begun, late; with invalid executed labels (worlds); with alarms (worlds)': '100; '
                                                                                                                        '100, '
                                                                                                                        '0; '
                                                                                                                        'none; '
                                                                                                                        'none',
                     'C3: failed runs; never begun, late; with invalid executed labels (worlds); with alarms (worlds)': '100; '
                                                                                                                        '100, '
                                                                                                                        '0; '
                                                                                                                        '16042, '
                                                                                                                        '16059, '
                                                                                                                        '16064, '
                                                                                                                        '16077; '
                                                                                                                        '16042, '
                                                                                                                        '16059, '
                                                                                                                        '16064, '
                                                                                                                        '16077',
                     'C4: failed runs; never begun, late; with invalid executed labels (worlds); with alarms (worlds)': '100; '
                                                                                                                        '2, '
                                                                                                                        '0; '
                                                                                                                        '16043, '
                                                                                                                        '16077, '
                                                                                                                        '16100; '
                                                                                                                        '16043, '
                                                                                                                        '16077, '
                                                                                                                        '16100',
                     'C5: failed runs; never begun, late; with invalid executed labels (worlds); with alarms (worlds)': '100; '
                                                                                                                        '1, '
                                                                                                                        '99; '
                                                                                                                        '16042, '
                                                                                                                        '16059, '
                                                                                                                        '16064, '
                                                                                                                        '16077; '
                                                                                                                        '16042, '
                                                                                                                        '16059, '
                                                                                                                        '16064, '
                                                                                                                        '16077',
                     'C6: failed runs; never begun, late; with invalid executed labels (worlds); with alarms (worlds)': '37; '
                                                                                                                        '1, '
                                                                                                                        '34; '
                                                                                                                        '16042, '
                                                                                                                        '16059, '
                                                                                                                        '16064, '
                                                                                                                        '16077; '
                                                                                                                        '16042, '
                                                                                                                        '16059, '
                                                                                                                        '16064, '
                                                                                                                        '16077',
                     'C7: failed runs; never begun, late; with invalid executed labels (worlds); with alarms (worlds)': '35; '
                                                                                                                        '0, '
                                                                                                                        '35; '
                                                                                                                        'none; '
                                                                                                                        'none',
                     'C8: failed runs; never begun, late; with invalid executed labels (worlds); with alarms (worlds)': '35; '
                                                                                                                        '0, '
                                                                                                                        '35; '
                                                                                                                        'none; '
                                                                                                                        'none',
                     'C9: failed runs; never begun, late; with invalid executed labels (worlds); with alarms (worlds)': '23; '
                                                                                                                        '1, '
                                                                                                                        '20; '
                                                                                                                        '16042, '
                                                                                                                        '16059, '
                                                                                                                        '16064, '
                                                                                                                        '16077; '
                                                                                                                        '16042, '
                                                                                                                        '16059, '
                                                                                                                        '16064, '
                                                                                                                        '16077',
                     'C10: failed runs; never begun, late; with invalid executed labels (worlds); with alarms (worlds)': '20; '
                                                                                                                         '0, '
                                                                                                                         '20; '
                                                                                                                         'none; '
                                                                                                                         'none',
                     'C11: failed runs; never begun, late; with invalid executed labels (worlds); with alarms (worlds)': '36; '
                                                                                                                         '1, '
                                                                                                                         '33; '
                                                                                                                         '16042, '
                                                                                                                         '16059, '
                                                                                                                         '16064, '
                                                                                                                         '16077; '
                                                                                                                         '16042, '
                                                                                                                         '16059, '
                                                                                                                         '16064, '
                                                                                                                         '16077',
                     'C12: failed runs; never begun, late; with invalid executed labels (worlds); with alarms (worlds)': '96; '
                                                                                                                         '1, '
                                                                                                                         '94; '
                                                                                                                         '16042, '
                                                                                                                         '16059, '
                                                                                                                         '16064, '
                                                                                                                         '16077; '
                                                                                                                         '16042, '
                                                                                                                         '16059, '
                                                                                                                         '16064, '
                                                                                                                         '16077',
                     'C13: failed runs; never begun, late; with invalid executed labels (worlds); with alarms (worlds)': '36; '
                                                                                                                         '1, '
                                                                                                                         '33; '
                                                                                                                         '16042, '
                                                                                                                         '16059, '
                                                                                                                         '16064, '
                                                                                                                         '16077; '
                                                                                                                         '16042, '
                                                                                                                         '16059, '
                                                                                                                         '16064, '
                                                                                                                         '16077',
                     'C14: failed runs; never begun, late; with invalid executed labels (worlds); with alarms (worlds)': '36; '
                                                                                                                         '1, '
                                                                                                                         '33; '
                                                                                                                         '16042, '
                                                                                                                         '16059, '
                                                                                                                         '16064, '
                                                                                                                         '16077; '
                                                                                                                         '16042, '
                                                                                                                         '16059, '
                                                                                                                         '16064, '
                                                                                                                         '16077',
                     'C15: failed runs; never begun, late; with invalid executed labels (worlds); with alarms (worlds)': '35; '
                                                                                                                         '0, '
                                                                                                                         '35; '
                                                                                                                         '16064, '
                                                                                                                         '16077; '
                                                                                                                         '16077'},
 'Figure 16.2': {'worlds traced': '46'}}


if __name__ == "__main__":
    main()
