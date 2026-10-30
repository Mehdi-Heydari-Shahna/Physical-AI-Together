#!/usr/bin/env python3
"""
The results of Section 11.2, read and checked without Isaac Sim.

Physical AI, Together, Chapter 11 (Section 11.2).

isaac_cart.py runs only inside Isaac Sim, on a computer with an NVIDIA RTX
GPU. Its results for the book were produced once, on the machine that
results/isaac/record.txt describes, and are shipped in results/isaac/ (the
full run) and results/isaac-quick/ (an earlier run with --quick), with the
follow-up of isaac_trace.py, on the same machine, in results/isaac-trace/. This
script needs only Python 3.10 or later and the standard library. It

  * compares the two runs value by value, and their per-period tables
    byte by byte;
  * hashes the stage text without the render settings that the application
    writes into it, which differ between runs;
  * reproduces the steps per period of the loop on the simulation clock
    from the clock's definition in Isaac Sim 6.2.0 (step count divided by
    steps per second) and the steps that the diagnostics took before it;
  * summarizes the per-period table of the run of Chapter 3 by phase, and
    against the configuration's own criteria;
  * reads the follow-up traces of isaac_trace.py (results/isaac-trace/):
    the positions against the reported speed and against the rounding of
    TGS's substeps to single precision, on the firm ground and, in period
    83, on the slope, the encoders, the seven variants of
    the comparison, the loads and the jumps of the reported speed, and
    writes the data of Figure S11.2;
  * with --check, confirms every value that Section 11.2 quotes, from the
    shipped summaries and from the values derived here.

Usage:
    python3 isaac_check.py            # writes results/isaac-derived.txt
    python3 isaac_check.py --check    # also checks the values quoted
"""
import argparse
import csv
import hashlib
import math
import os
import re
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FULL = os.path.join(RES, "isaac")
QUICK = os.path.join(RES, "isaac-quick")
MUJOCO_CSV = os.path.join(RES, "agreement-chapter3.csv")
BOUND = 0.05               # the disturbance bound of Chapter 5 [m/s^2]
DT = 0.001                 # the time step of the adapter's runs [s]
T_PERIOD = 0.1             # the period of the run of Chapter 3 [s]
SLOPE = math.radians(9.5)  # the angle of the slope (cart_mujoco/scene.py) [rad]


def g3(x):
    return f"{x:.3g}"


# ----------------------------------------------------------------------------
# The shipped results
# ----------------------------------------------------------------------------
def summary(folder):
    """{group: {key: value}} from isaac-summary.txt (any line endings)."""
    return summary_file(os.path.join(folder, "isaac-summary.txt"))


def summary_file(path):
    """{group: {key: value}} from a summary written by the Isaac scripts."""
    out, group = {}, None
    with open(path, encoding="utf-8", newline="") as fh:
        for line in fh.read().splitlines():
            if line.startswith("[") and line.endswith("]"):
                group = line[1:-1]
                out[group] = {}
            elif line.startswith("  ") and group is not None and " = " in line:
                key, val = line.strip().split(" = ", 1)
                out[group][key] = val
    return out


def rows(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return [{k.strip(): (int(v) if k.strip() == "k" else float(v)) for k, v in r.items()}
                for r in csv.DictReader(fh)]


def _part_of(q, f):
    """True if the quick run's value q is the full run's value f, or the
    part of it that the quick run computed (it skips the smaller time
    steps): 'skipped', or the leading items of each '; '-separated part."""
    if q == "skipped":
        return True
    qp, fp = q.split("; "), f.split("; ")
    if len(qp) > len(fp):
        return False
    for a, b in zip(qp, fp):
        qi, fi = a.split(", "), b.split(", ")
        if a != b and fi[:len(qi)] != qi:
            return False
    return True


def compare_runs(full, quick):
    """The values that both runs report: identical, computed only in part
    by the quick run and consistent with the full run, and different."""
    same, partial, differ = 0, 0, []
    for group, items in quick.items():
        for key, val in items.items():
            other = full.get(group, {}).get(key)
            if other is None:
                continue
            if other == val:
                same += 1
            elif _part_of(val, other):
                partial += 1
            else:
                differ.append(f"{group}.{key}")
    return same, partial, differ


# ----------------------------------------------------------------------------
# The stage
# ----------------------------------------------------------------------------
def strip_render(text):
    """The stage text without the prims that the application writes for its
    renderer: the scope /Render and the override of the viewport camera
    /OmniverseKit_Persp. Their settings differ between runs; the rest is
    the stage that the script wrote and PhysX simulated."""
    lines = text.replace("\r\n", "\n").split("\n")
    out, i = [], 0
    while i < len(lines):
        if re.match(r'^(def "Render"|over "OmniverseKit_Persp")', lines[i]):
            depth, opened = 0, False
            while i < len(lines):
                depth += lines[i].count("{") - lines[i].count("}")
                opened = opened or "{" in lines[i]
                i += 1
                if opened and depth == 0:
                    break
            if i < len(lines) and lines[i].strip() == "":
                i += 1
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def stage_facts():
    texts = {}
    for name, folder in (("full", FULL), ("quick", QUICK)):
        with open(os.path.join(folder, "cart.usda"), encoding="utf-8", newline="") as fh:
            texts[name] = fh.read()
    stripped = {k: strip_render(v) for k, v in texts.items()}
    return {
        "stage texts of the two runs identical": str(texts["full"] == texts["quick"]),
        "the same without the render prims": str(stripped["full"] == stripped["quick"]),
        "stage text without the render prims, SHA-256":
            hashlib.sha256(stripped["full"].encode("utf-8")).hexdigest()[:16],
        "physics scene APIs on the stage": ", ".join(re.findall(
            r'def PhysicsScene "PhysicsScene" \(\s*prepend apiSchemas = \[([^\]]*)\]',
            stripped["full"])[0].replace('"', "").split(", ")),
    }


# ----------------------------------------------------------------------------
# The loop on the simulation clock
# ----------------------------------------------------------------------------
def steps_before_clock_loop(model_count):
    """The physics steps that the simulation manager had counted when the
    diagnostic of the clock began: the count reported with the model, then
    every step of the diagnostics that isaac_cart.py runs before it, in
    order (their durations are fixed by the script, not by the results)."""
    n = model_count                                     # after the static sink
    n += 10 * 100                                       # wheel loads at rest
    n += 100                                            # one period at 1 m/s
    n += 2 * (round(1.2 / DT) + 1)                      # crest, both coordinates
    lo, hi, iters = 0.0, 5.0, 0                         # each bisection: 10 halvings
    while hi - lo > 0.005:
        hi, iters = (lo + hi) / 2, iters + 1
    n += 4 * iters * (round(0.5 / DT) + 1)              # four capacities
    n += 3 * (round(0.5 / DT) + 1)                      # three label trackings
    n += round(1.4 / DT)                                # crest, braking at the label
    n += sum(round((22.6 - 19.9) / s / DT) for s in (2.3, 2.4, 2.5))
    n += round(1.0 / DT)                                # rolling freely
    n += 5 * 100                                        # encoders
    n += round(1.0 / DT)                                # coasting
    return n


def clock_pattern(n0, steps_per_second=1000, periods=220, steps=100, T=0.1):
    """The steps of each period of a loop that steps while the clock,
    measured from the start of the loop, is below (k + 1) T, with the
    clock of Isaac Sim 6.2.0's simulation manager: the step count divided
    by the steps per second, in double precision."""
    t0, n, counts = n0 / steps_per_second, n0, []
    for k in range(periods):
        c = 0
        while n / steps_per_second - t0 < (k + 1) * T and c < 2 * steps:
            n, c = n + 1, c + 1
        counts.append(c)
    exceptions = "; ".join(f"{c} in period {k}" for k, c in enumerate(counts) if c != steps)
    after = (n0 + 22000) / steps_per_second - t0
    return (exceptions or f"{steps} in each") + f"; clock after 22,000 steps {after!r}"


# ----------------------------------------------------------------------------
# The run of Chapter 3, period by period
# ----------------------------------------------------------------------------
CREST = (20.0, 21.66)      # the crest, in the ground coordinate [m] (Section 11.1)


def phase(r, T=0.1):
    """Where a period of the run of Chapter 3 takes the cart. A moving
    period is 'with a wheel on the crest' if the ground coordinate of
    either axle lies on the crest at some time in it, taking the front axle
    0.1 m and the rear axle 0.6 m behind the bumper (beyond the crest, the
    track coordinate exceeds the ground coordinate by at most 16.6 mm)."""
    if r["l"] > 0 or r["ell"] > 0:
        return "tool unfolding or unfolded"
    if r["v"] == 0 and r["u"] == 0:
        return "at rest, firm ground" if r["p"] < 20 else "at rest on the slope"
    p1 = r["p"] + r["v"] * T + r["u"] * T * T / 2
    lo, hi = min(r["p"], p1), max(r["p"], p1)
    if hi - 0.1 < CREST[0]:
        return "moving on the firm ground"
    if lo - 0.6 <= CREST[1]:
        return "moving with a wheel on the crest"
    return "moving on the slope"


def beyond_reported_speed(r, T=0.1, h=DT):
    """How far the position at the end of a period lies outside what the
    speed that the simulator reports allows [m]. Positions advance with the
    velocity at the end of each step, so if the reported speed v were the
    rate of the position, p(T) - p0 = h (v_1 + ... + v_N), and with every
    v_i within [low, high] of the nominal speed, the end difference dp
    would lie within [low T, high T] + u h T / 2."""
    lo = r["low"] * T + r["u"] * h * T / 2
    hi = r["high"] * T + r["u"] * h * T / 2
    return max(0.0, r["dp"] - hi, lo - r["dp"])


def per_period(isaac, mujoco):
    out = {}
    for name, table in (("Isaac Sim", isaac), ("MuJoCo", mujoco)):
        ex = [(beyond_reported_speed(r), r) for r in table]
        worst = max(ex, key=lambda e: e[0])
        out[f"{name}: periods whose end position lies beyond the reported speed by "
            f"more than 0.01, 0.1, 1 mm"] = ", ".join(
            str(sum(e > t for e, _ in ex)) for t in (1e-5, 1e-4, 1e-3))
        out[f"{name}: largest such excess [m], period"] = f"{g3(worst[0])}, {worst[1]['k']}"
    flagged = [r for r in isaac if beyond_reported_speed(r) > 1e-5]
    out["Isaac Sim: those beyond by more than 0.01 mm, by phase"] = "; ".join(
        f"{name} {sum(phase(r) == name for r in flagged)}"
        for name in ("moving on the firm ground", "moving with a wheel on the crest",
                     "moving on the slope", "at rest on the slope", "tool unfolding or unfolded"))
    folded = [r for r in isaac if r["l"] == 0 and r["ell"] == 0]
    out["periods with the tool folded"] = str(len(folded))
    out["axle: periods above the bound, over 1 ms or one step, and the largest [m/s^2]"] = (
        f"{sum(max(r['w_axle'], r['w_axle_step']) > BOUND for r in folded)}, "
        f"{g3(max(max(r['w_axle'], r['w_axle_step']) for r in folded))}")
    out["axle over 1 ms, tool folded: median over the periods [m/s^2]"] = \
        g3(statistics.median(r["w_axle"] for r in folded))
    for name in ("at rest, firm ground", "moving on the firm ground", "moving with a wheel on the crest",
                 "moving on the slope", "at rest on the slope", "tool unfolding or unfolded"):
        sel = [r for r in isaac if phase(r) == name]
        if not sel:
            continue
        worst = max(sel, key=lambda r: r["w_axle"])
        out[f"{name}: periods; axle over 1 ms, largest [m/s^2], period"] = (
            f"{len(sel)}; {g3(worst['w_axle'])}, {worst['k']}")
    by_k = {r["k"]: r for r in isaac}
    out["axle over 1 ms, accelerating on the firm ground: period 1 (from rest), "
        "period 29 (5.6 m/s) [m/s^2]"] = f"{g3(by_k[1]['w_axle'])}, {g3(by_k[29]['w_axle'])}"
    rest = [r for r in isaac if r["u"] == 0 and r["v"] == 0 and r["ell"] == r["l"]]
    out["at rest, the tool still: periods; largest |dp|, |dv| at the end [m], [m/s]"] = (
        f"{len(rest)}; {g3(max(abs(r['dp']) for r in rest))}, "
        f"{g3(max(abs(r['dv']) for r in rest))}")
    small = [r for r in isaac if 0 < abs(r["u"]) < 0.5 and r["l"] == 0]
    out["periods commanding less than 0.5 m/s^2 but not zero: period, u, dp [m]"] = "; ".join(
        f"{r['k']}, {r['u']:g}, {g3(r['dp'])}" for r in small)
    acc = [r for r in isaac if r["u"] == 2.0 and r["v"] >= 0 and r["l"] == 0]
    out["accelerating towards the wall at 2 m/s^2: dp from, to [m]"] = (
        f"{g3(min(r['dp'] for r in acc))}, {g3(max(r['dp'] for r in acc))}")
    out["accelerating towards the wall: start position and reach from, to [m] (a h t/2 over 0.05 s: 5e-05)"] = (
        f"{g3(min(r['start_pl'] for r in acc if r['k'] > 1))}, "
        f"{g3(max(r['start_pl'] for r in acc if r['k'] > 1))}")
    speed_norm = [r for r in isaac if abs(r["dv"]) > abs(r["dp"]) + abs(r["dl"])]
    out["periods in which the speed sets the norm at the end, of 220"] = str(len(speed_norm))
    ends = [r["end"] for r in isaac]
    out["end of period above 1 mm: periods"] = str(sum(e > 1e-3 for e in ends))
    # against MuJoCo, as Section 12.1 compares (the norm of the difference of the error vectors)
    ref = {r["k"]: r for r in mujoco}
    diff = {r["k"]: max(abs(r["dp"] - ref[r["k"]]["dp"]) + abs(r["dl"] - ref[r["k"]]["dl"]),
                        abs(r["dv"] - ref[r["k"]]["dv"])) for r in isaac}
    k_max = max(diff, key=diff.get)
    out["against MuJoCo, per period: median, largest [m], period"] = (
        f"{g3(statistics.median(diff.values()))}, {g3(diff[k_max])}, {k_max}")
    return out


def mujoco_side(mujoco):
    folded = [r for r in mujoco if r["l"] == 0 and r["ell"] == 0]
    rest = [r for r in mujoco if phase(r) == "at rest on the slope"]
    return {
        "MuJoCo: axle over 1 ms, tool folded: median over the periods [m/s^2]":
            g3(statistics.median(r["w_axle"] for r in folded)),
        "MuJoCo: at rest on the slope: axle over 1 ms, largest [m/s^2]":
            g3(max(r["w_axle"] for r in rest)),
        "MuJoCo: periods in which the speed sets the norm at the end, of 220":
            str(sum(abs(r["dv"]) > abs(r["dp"]) + abs(r["dl"]) for r in mujoco)),
    }


# ----------------------------------------------------------------------------
# The follow-up traces (isaac_trace.py, results/isaac-trace/)
# ----------------------------------------------------------------------------
TRACE = os.path.join(RES, "isaac-trace")
SUBSTEPS = 32              # TGS: one substep per position iteration (Section S11.10.1)
RADIUS = 0.1               # the wheels' radius [m]
VARIANTS = (("settings of Section 11.2", "settings-of-Section-112", SUBSTEPS),
            ("no velocity iteration", "no-velocity-iteration", SUBSTEPS),
            ("four velocity iterations", "four-velocity-iterations", SUBSTEPS),
            ("64 position iterations", "64-position-iterations", 64),
            ("PGS solver", "PGS-solver", 1),
            ("contact offset 2 mm", "contact-offset-2-mm", SUBSTEPS),
            ("filtered loads", "filtered-loads", SUBSTEPS))


def f32_spacing(x):
    """The spacing of single-precision numbers at x (x not zero)."""
    return 2.0 ** (math.floor(math.log2(abs(x))) - 23)


def trace(name):
    """The rows of a file of the follow-up (any line endings)."""
    with open(os.path.join(TRACE, name), encoding="utf-8", newline="") as fh:
        return [{k: (v if k == "phase" else float(v)) for k, v in r.items()}
                for r in csv.DictReader(fh)]


def rounded_rate(p, v, h=DT, n=SUBSTEPS, pitch=0.0):
    """The rate of the track coordinate when each of n substeps of a step h
    advances the slider that carries the cart horizontally by its velocity,
    v cos(pitch), times h / n, the sum rounded to the nearest number that
    single precision represents, and the cart moves along the ground by the
    slider's advance divided by cos(pitch). The slider's coordinate lies near
    p, in the same interval between powers of two, except within
    millimetres of those powers: on the firm ground it is the track
    coordinate p itself (cart_isaac/sim.py, IsaacCart.place: the slider at
    the chassis' centre plus half its length)."""
    q = f32_spacing(p)
    c = math.cos(pitch)
    return n * round(v * c * h / n / q) * q / h / c


def rounding_bound(p, h=DT, n=SUBSTEPS):
    """How far the rate of such a coordinate can depart from v: half a
    spacing per substep [m/s]."""
    return f32_spacing(p) / 2 / (h / n)


def rounding_bound_track(p, h=DT, n=SUBSTEPS, pitch=SLOPE):
    """The same bound for the rate of the track coordinate, which advances by
    the slider's advance divided by cos(pitch): on the slope, the slider's
    bound divided by cos(SLOPE) [m/s]; on the firm ground (pitch 0), the
    slider's bound itself."""
    return rounding_bound(p, h, n) / math.cos(pitch)


def steps_of(rows):
    """For each step of a trace: the row before it, the rate of the position
    over it, and the reported speed and the encoder speed (the front
    wheel's rate times its radius) after it. Positions advance with the
    velocity at the end of each step (beyond_reported_speed)."""
    return [(a, (b["p"] - a["p"]) / DT, b["v"], RADIUS * b["front_rate"])
            for a, b in zip(rows, rows[1:])]


def rms(xs):
    xs = list(xs)
    return math.sqrt(sum(x * x for x in xs) / len(xs))


def windows_of(rows, n=10):
    """Over windows of n steps: (encoder less reported speed, rate of the
    position less reported speed, encoder less rate of the position), the
    speeds averaged over the steps of the window."""
    out = []
    for i in range(len(rows) - n):
        rate = (rows[i + n]["p"] - rows[i]["p"]) / (n * DT)
        enc = statistics.mean(RADIUS * rows[j]["front_rate"] for j in range(i + 1, i + n + 1))
        v = statistics.mean(rows[j]["v"] for j in range(i + 1, i + n + 1))
        out.append((enc - v, rate - v, enc - rate))
    return out


def trace_rounding(name, rows):
    """The position's rate against the reported speed and against the
    rounding of the substeps, over a trace [mm/s]; the pitch after each step
    gives the slope."""
    st = steps_of(rows)
    obs_rep = [rate - v for _, rate, v, _ in st]
    obs_pred = [rate - rounded_rate(a["p"], v, pitch=b["pitch"])
                for (a, rate, v, _), b in zip(st, rows[1:])]
    return obs_rep, obs_pred


def trace_facts():
    out = {}
    key = "traces: "
    # the run: the release, and the packages against those of the first run
    tsum = summary_file(os.path.join(TRACE, "summary.txt"))
    pins = [open(os.path.join(d, "pip-freeze.txt"), encoding="utf-8", newline="").read().split()
            for d in (FULL, TRACE)]
    out[key + "record: Isaac Sim; omni.physx; the packages installed as in the first run"] = (
        f"{tsum['record']['Isaac Sim']}; {tsum['record']['extension omni.physx']}; "
        f"{pins[0] == pins[1]}")
    out[key + "steps that failed"] = open(os.path.join(TRACE, "errors.txt"), encoding="utf-8",
                                          newline="").read().strip()
    # the comparison with the settings of Section 11.2 against the shipped one
    shipped = {r["k"]: r for r in rows(os.path.join(FULL, "agreement-chapter3-isaac.csv"))}
    again = {int(r["k"]): r for r in trace("periods-settings-of-Section-112.csv")}
    cols = [c for c in next(iter(shipped.values())) if c in next(iter(again.values()))]
    worst = max(abs(shipped[k][c] - again[k][c]) for k in shipped for c in cols)
    out[key + "comparison with the settings of Section 11.2: periods, columns in common with the "
              "shipped table, largest difference"] = f"{len(again)}, {len(cols)}, {worst:g}"

    # one period at 1 m/s from 5 m: whole spacings, 66 per substep
    one = trace("trace-one-period-1ms-from-5m.csv")
    st = steps_of(one)
    q5 = f32_spacing(5.0)
    counts = [round(rate * DT / q5) for _, rate, _, _ in st]
    off = max(abs(rate * DT / q5 - n) for (_, rate, _, _), n in zip(st, counts))
    out[key + "from 5 m: steps; displacement per step in single-precision spacings, from, to; "
              "steps with 2112, 32 times 66; largest departure from whole spacings (the "
              "nine digits printed)"] = (
        f"{len(st)}; {min(counts)}, {max(counts)}; {counts.count(2112)}; {off:.2g}")
    once = [round(v * DT / q5) for _, _, v, _ in st]
    out[key + "from 5 m: spacings per step if the advance were rounded once per step, at the "
              "reported speed, from, to; steps with 2097"] = (
        f"{min(once)}, {max(once)}; {once.count(2097)}")
    per_sub = [v * DT / SUBSTEPS / q5 for _, _, v, _ in st]
    out[key + "from 5 m: spacings per substep that the reported speed implies, mean; "
              "rounded"] = f"{statistics.mean(per_sub):.3g}; {round(statistics.mean(per_sub))}"
    obs_rep, obs_pred = trace_rounding("one", one)
    pred_rep = [rounded_rate(a["p"], v) - v for a, _, v, _ in st]
    # at exactly 1 m/s, each substep's advance of 65.536 spacings rounds to 66
    n_sub = 1.0 * DT / SUBSTEPS / q5
    extra = round(n_sub) - n_sub
    out[key + "from 5 m at exactly 1 m/s: spacings per substep; the rounding's excess per "
              "substep [spacings], as a speed [mm/s], over a period of 3200 substeps [mm]"] = (
        f"{n_sub:.5g}; {extra:.3g}, {1e3 * extra * q5 / (DT / SUBSTEPS):.3g}, "
        f"{1e3 * extra * q5 * SUBSTEPS * round(T_PERIOD / DT):.3g}")
    predicted = sum(rounded_rate(a["p"], v) for a, _, v, _ in st[:100]) * DT
    observed = one[100]["p"] - one[0]["p"]
    out[key + "from 5 m, the first period: end beyond 5.1 m [mm]; displacement less 0.1 m, "
              "predicted by rounding each substep's advance at the reported speed [mm]; "
              "observed less predicted [spacings]"] = (
        f"{1e3 * (one[100]['p'] - 5.1):.3g}; {1e3 * (predicted - 0.1):.3g}; "
        f"{(observed - predicted) / q5:.2g}")
    out[key + "from 5 m: rate of the position less reported speed, mean; rounding less "
              "reported speed, mean [mm/s]"] = \
        f"{1e3 * statistics.mean(obs_rep):.3g}; {1e3 * statistics.mean(pred_rep):.3g}"
    out[key + "from 5 m: rate of the position less reported speed, less the rounding: "
              "RMS [mm/s]"] = f"{1e3 * rms(obs_rep):.3g}, {1e3 * rms(obs_pred):.3g}"
    out[key + "from 5 m: steps in which the rounding predicts the displacement to within one, "
              "two spacings"] = ", ".join(
        f"{sum(abs(x) <= n * q5 / DT + 1e-9 for x in obs_pred)} of {len(st)}" for n in (1, 2))
    # the encoders follow the motion
    enc_rep = [enc - v for _, _, v, enc in st]
    w = windows_of(one)
    out[key + "from 5 m: encoder less reported speed, mean [mm/s]; over 10 steps, encoder less "
              "rate of the position, RMS [mm/s]; per step, RMS [mm/s]"] = (
        f"{1e3 * statistics.mean(enc_rep):.3g}; {1e3 * rms(x[2] for x in w):.3g}; "
        f"{1e3 * rms(enc - rate for _, rate, _, enc in st):.3g}")

    # period 49, braking near 18 m: the sawtooth
    p49 = trace("trace-period-49.csv")
    st49 = steps_of(p49)
    obs_rep, obs_pred = trace_rounding("49", p49)
    q18 = f32_spacing(18.0)
    half = [abs((v * DT / SUBSTEPS / q18) % 1 - 0.5) for _, _, v, _ in st49]
    one = q18 / DT + 1e-9
    good = sum(abs(x) <= one for x in obs_pred)
    miss = [h for x, h in zip(obs_pred, half) if abs(x) > one]
    out[key + "period 49: steps; rate of the position less reported speed, from, to [mm/s]; "
              "bound of the rounding [mm/s]"] = (
        f"{len(st49)}; {1e3 * min(obs_rep):.3g}, {1e3 * max(obs_rep):.3g}; "
        f"{1e3 * rounding_bound(18.0):.3g}")
    out[key + "period 49: steps predicted by the rounding to within one spacing per step; the "
              "others: the increment per substep at most this far from half a spacing"] = (
        f"{good} of {len(st49)}; {max(miss):.2g}")
    out[key + "period 49: rate of the position less reported speed, less the rounding: RMS "
              "[mm/s]"] = f"{1e3 * rms(obs_rep):.3g}, {1e3 * rms(obs_pred):.3g}"
    w = windows_of(p49)
    out[key + "period 49: over 10 steps, encoder less reported speed, encoder less rate of the "
              "position: RMS [mm/s]"] = f"{1e3 * rms(x[0] for x in w):.3g}, {1e3 * rms(x[2] for x in w):.3g}"
    # how fast the sawtooth repeats: one spacing per substep of speed, at 2 m/s^2
    out[key + "period 49: change of speed by one spacing per substep, and its time at 2 m/s^2 "
              "[mm/s], [ms]"] = (f"{1e3 * q18 / (DT / SUBSTEPS):.3g}, "
                                 f"{1e3 * q18 / (DT / SUBSTEPS) / 2.0:.3g}")

    # the bounds of the rounding by distance from the origin and by update
    out[key + "bound of the rounding with 32 substeps of 1 ms between 4 and 8, 8 and 16, 16 and "
              "32 m [mm/s]"] = ", ".join(f"{1e3 * rounding_bound(x):.3g}" for x in (5.0, 10.0, 20.0))
    out[key + "the same beyond 16 m with 64 substeps, and with one update per step (PGS) "
              "[mm/s]"] = (f"{1e3 * rounding_bound(20.0, n=64):.3g}, "
                           f"{1e3 * rounding_bound(20.0, n=1):.3g}")
    out[key + "the same beyond 16 m for the track coordinate along the slope, the slider's "
              "bound divided by cos(9.5 deg), with 32 and 64 substeps and with one update per "
              "step (PGS) [mm/s]; the factor less 1 [%]"] = (
        ", ".join(f"{1e3 * rounding_bound_track(20.0, n=n):.3g}" for n in (SUBSTEPS, 64, 1))
        + f"; {100 * (1 / math.cos(SLOPE) - 1):.2g}")
    out[key + "the same at 2.25 m, the encoders' diagnostic, at 1, 0.5 and 0.25 ms [mm/s]"] = \
        ", ".join(f"{1e3 * rounding_bound(2.25, h=h):.3g}" for h in (0.001, 0.0005, 0.00025))

    # the variants of the comparison
    for label, stem, n in VARIANTS:
        per = trace(f"periods-{stem}.csv")
        folded = [r for r in per if r["l"] == 0 and r["ell"] == 0]
        moving = [r for r in folded if not (r["u"] == 0 and r["v"] == 0)]
        rest = [r for r in folded if r["u"] == 0 and r["v"] == 0]
        dist = lambda r: max(r["w_axle"], r["w_axle_step"])    # noqa: E731
        ends = sorted(r["end"] for r in per)
        out[key + f"{label}: position updates per step; periods beyond the reported speed by more "
                  f"than 0.01 mm; largest [mm]"] = (
            f"{n}; {sum(r['beyond'] > 1e-5 for r in per)}; "
            f"{1e3 * max(r['beyond'] for r in per):.3g}")
        out[key + f"{label}: reported speed less rate of the position over 10 steps, largest; "
                  f"bound of the rounding beyond 16 m [mm/s]"] = (
            f"{1e3 * max(r['mismatch_10_steps'] for r in per):.3g}; "
            f"{1e3 * rounding_bound(20.0, n=n):.3g}")
        out[key + f"{label}: bound of the rounding beyond 16 m for the track coordinate along "
                  f"the slope [mm/s]"] = f"{1e3 * rounding_bound_track(20.0, n=n):.3g}"
        out[key + f"{label}: disturbance read from the reported speed, tool folded: periods above "
                  f"the bound, moving, at rest; median over the moving periods, largest "
                  f"[m/s^2]"] = (
            f"{sum(dist(r) > BOUND for r in moving)} of {len(moving)}, "
            f"{sum(dist(r) > BOUND for r in rest)} of {len(rest)}; "
            f"{g3(statistics.median(dist(r) for r in moving))}, "
            f"{g3(max(dist(r) for r in folded))}")
        out[key + f"{label}: end of period, median, 90th percentile (nearest rank), largest [mm], "
                  f"period"] = (
            f"{1e3 * statistics.median(ends):.3g}, {1e3 * ends[math.ceil(0.9 * len(ends)) - 1]:.3g}, "
            f"{1e3 * ends[-1]:.3g}, {int(max(per, key=lambda r: r['end'])['k'])}")
        out[key + f"{label}: start, largest |dp| + |dl| [mm], |dv| [mm/s]; speed within a period "
                  f"outside period 201, largest [mm/s]"] = (
            f"{1e3 * max(r['start_pl'] for r in per):.3g}, {1e3 * max(r['start_v'] for r in per):.3g}; "
            f"{1e3 * max(max(abs(r['low']), abs(r['high'])) for r in per if int(r['k']) != 201):.3g}")
        if stem == "contact-offset-2-mm":
            base = trace("periods-settings-of-Section-112.csv")
            out[key + "contact offset 2 mm: every column of every period as with the settings of "
                      "Section 11.2"] = str(all(a == b for a, b in zip(per, base)))
        if stem == "PGS-solver":
            worst = max(folded, key=dist)
            out[key + "PGS solver: the largest disturbance, tool folded: period; where"] = (
                f"{int(worst['k'])}; {phase(worst)}")
            out[key + "PGS solver: periods beyond the reported speed by more than 1 um, the "
                      "tolerance of the check of estimation (Section 12.3)"] = (
                str(sum(r["beyond"] > 1e-6 for r in per)))
        if stem in ("settings-of-Section-112",):
            out[key + f"{label}: periods beyond by more than 0.01 mm that begin beyond 16 m"] = (
                f"{sum(r['beyond'] > 1e-5 and r['p'] >= 16.0 for r in per)} of "
                f"{sum(r['beyond'] > 1e-5 for r in per)}; periods that begin there "
                f"{sum(r['p'] >= 16.0 for r in per)} of {len(per)}")

    # the loads
    weight = 90.101 * 9.81
    rest = trace("trace-rest-firm-ground.csv")
    tot = [r["front_load"] + r["rear_load"] for r in rest]
    out[key + "at rest on the firm ground: steps; first step's loads [N]; total, from, to, mean "
              "[N]; mean less the weight [%]"] = (
        f"{len(rest)}; {rest[0]['front_load']:.2f}, {rest[0]['rear_load']:.2f}; "
        f"{min(tot):.1f}, {max(tot):.1f}, {statistics.mean(tot):.3f}; "
        f"{100 * (statistics.mean(tot) / weight - 1):.2g}")
    m = statistics.mean(tot)
    lag = sum((tot[i] - m) * (tot[i + 1] - m) for i in range(len(tot) - 1)) / \
        sum((x - m) ** 2 for x in tot)
    out[key + "at rest on the firm ground: standard deviation of the total [N]; correlation of "
              "successive steps; means over 10 and 100 steps, largest departure from the weight "
              "[%]"] = (
        f"{statistics.pstdev(tot):.3g}; {lag:.2f}; " + ", ".join(
            f"{100 * max(abs(statistics.mean(tot[i:i + n]) - weight) for i in range(len(tot) - n + 1)) / weight:.2g}"
            for n in (10, 100)))
    slope = trace("trace-rest-slope.csv")
    tot_s = [r["front_load"] + r["rear_load"] for r in slope]
    rigid = weight * math.cos(statistics.mean(r["pitch"] for r in slope))
    out[key + "at rest on the slope: steps; total, mean [N]; the rigid cart's [N]; difference "
              "[%]"] = (f"{len(slope)}; {statistics.mean(tot_s):.3f}; {rigid:.3f}; "
                        f"{100 * (statistics.mean(tot_s) / rigid - 1):.1g}")
    # a row reports the contact forces of the step before it; the first row of
    # a trace that starts with a placement holds those of the step before the
    # placement, and is left out
    for label, name in (("from 5 m, the three periods", "trace-one-period-1ms-from-5m.csv"),
                        ("braking from 4 m/s at 8 m", "trace-braking-2-from-4.csv")):
        rr = trace(name)[1:]
        t = [r["front_load"] + r["rear_load"] for r in rr]
        out[key + f"{label}: total load from, to, mean [N]; mean less the weight [%]"] = (
            f"{min(t):.0f}, {max(t):.0f}, {statistics.mean(t):.2f}; "
            f"{100 * (statistics.mean(t) / weight - 1):.2g}")
    # the jumps of the reported speed at rest, and the settling after a placement
    for label, rr in (("at rest on the firm ground", rest), ("at rest on the slope", slope)):
        out[key + f"{label}: largest jump of the reported speed in one step [mm/s]"] = \
            g3(1e3 * max(abs(b["v"] - a["v"]) for a, b in zip(rr, rr[1:])))
    p83 = trace("trace-period-83.csv")
    tot83 = [r["front_load"] + r["rear_load"] for r in p83]
    lead = sum(r["phase"] == "lead-in" for r in p83)
    out[key + "period 83: total load in the first ten steps after the placement, from, to "
              "[N]; in the rest of the lead-in, from, to [N]"] = (
        f"{min(tot83[1:11]):.0f}, {max(tot83[1:11]):.0f}; "
        f"{min(tot83[11:lead + 1]):.0f}, {max(tot83[11:lead + 1]):.0f}")
    # period 83, on the slope, the largest excess over the reported speed: the
    # rounding of the horizontal slider's advance at each substep
    obs_rep, obs_pred = trace_rounding("83", p83)
    body = slice(lead, None)
    out[key + "period 83, on the slope: pitch [rad]; over the period, rate of the position less "
              "reported speed, mean, RMS [mm/s]; less the rounding of the horizontal slider's "
              "advance, mean, RMS [mm/s]"] = (
        f"{statistics.mean(r['pitch'] for r in p83[lead:]):.4g}; "
        f"{1e3 * statistics.mean(obs_rep[body]):.3g}, {1e3 * rms(obs_rep[body]):.3g}; "
        f"{1e3 * statistics.mean(obs_pred[body]):.2g}, {1e3 * rms(obs_pred[body]):.3g}")
    return out


# ----------------------------------------------------------------------------
def derived():
    full, quick = summary(FULL), summary(QUICK)
    out = {}
    same, partial, differ = compare_runs(full, quick)
    out["values reported by both runs: identical, in part by the quick run and consistent, "
        "different"] = f"{same}, {partial}, {len(differ)}"
    out["values that differ"] = ", ".join(differ) or "none"
    out["per-period tables of the two runs identical"] = str(
        open(os.path.join(FULL, "agreement-chapter3-isaac.csv"), "rb").read() ==
        open(os.path.join(QUICK, "agreement-chapter3-isaac.csv"), "rb").read())
    out.update(stage_facts())
    model = full["the model as PhysX reports it"]
    counted = int(model["physics steps counted by the simulation manager, and by the adapter"]
                  .split(",")[0])
    n0 = steps_before_clock_loop(counted)
    out["steps counted by the simulation manager before the loop on the clock"] = str(n0)
    observed = full["diagnostics"]["steps per period, loop on the simulation clock"]
    predicted = clock_pattern(n0)
    out["loop on the clock, from the clock's definition: as observed"] = str(predicted == observed)
    out["loop on the clock: periods with 101 or 99 steps"] = str(
        len(re.findall(r"(?:101|99) in period", observed)))
    out["loop on the clock: the clock's difference after the 400 steps of periods 0 to 3 [s]"] = \
        repr((n0 + 400) / 1000 - n0 / 1000)
    out["start counts from 35,000 to 35,999 that give the same pattern"] = \
        f"{sum(clock_pattern(n) == observed for n in range(35000, 36000))} of 1000"
    d = full["diagnostics"]
    loads = d["wheel loads at rest [N]"]
    together = float(re.search(r"= ([\d.]+);", loads).group(1))
    weight = float(d["weight of the parts the wheels carry [N]"])
    out["wheel loads at rest less the weight [N], [%]"] = (
        f"{together - weight:.4g}, {100 * (together - weight) / weight:.3g}")
    v = full["variants"]
    cm = float(v["stage in centimetres: wheel loads at rest, together [stage force units]"])
    out["stage in centimetres: loads / 100 less the weight, relative [%]"] = \
        f"{100 * (cm / 100 - weight) / weight:.2g}"
    gains = re.findall(r"damping as PhysX reports it ([\d.]+)",
                       v["servo damping per radian: encoder less cart speed at 2 m/s^2 [mm/s], by time step"])
    out["servo damping per radian: reported gain over the intended one"] = \
        g3(float(gains[0]) / 45.5)
    mj = mujoco_summary()
    mj_loads = [float(x) for x in re.findall(r"[\d.]+", mj["diagnostics"]["wheel loads at rest [N]"])[:2]]
    is_loads = [float(x) for x in re.findall(r"[\d.]+", loads)[:2]]
    out["wheel loads at rest, front and rear, less MuJoCo's [N]"] = ", ".join(
        f"{a - b:.2f}" for a, b in zip(is_loads, mj_loads))
    isaac = rows(os.path.join(FULL, "agreement-chapter3-isaac.csv"))
    mujoco = rows(MUJOCO_CSV)
    out.update(per_period(isaac, mujoco))
    out.update(mujoco_side(mujoco))
    out["float32 spacing at 25 m [m]"] = g3(2.0 ** (math.floor(math.log2(25.0)) - 23))
    out["float32 spacing at 0.1 m [m]"] = g3(2.0 ** (math.floor(math.log2(0.1)) - 23))
    out["displacement in one step of 1 ms by 0.05 m/s^2 [m]"] = g3(BOUND * DT * DT / 2)
    out.update(environment_facts(quick))
    out.update(tool_facts())
    out["crest: v^2 / (R + r) at 2.5 m/s [m/s^2]; speed at which it reaches g cos(slope) [m/s]"] = (
        f"{2.5 ** 2 / 10.1:.2g}; {math.sqrt(9.81 * 10.1 * math.cos(math.radians(9.5))):.2g}")
    out.update(trace_facts())
    return full, out


def write_rounding(path_5m, path_49):
    """For Figure S11.2: per step of the traces from 5 m and of period 49,
    the rate of the position, the rounding of the substeps and the encoder
    speed, each less the reported speed [mm/s]."""
    for path, name in ((path_5m, "trace-one-period-1ms-from-5m.csv"),
                       (path_49, "trace-period-49.csv")):
        rr = trace(name)
        with open(path, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["step", "position", "rounding", "encoder"])
            for i, (a, rate, v, enc) in enumerate(steps_of(rr)):
                w.writerow([i, f"{1e3 * (rate - v):.6g}", f"{1e3 * (rounded_rate(a['p'], v) - v):.6g}",
                            f"{1e3 * (enc - v):.6g}"])


def mujoco_summary():
    out, group = {}, None
    with open(os.path.join(RES, "mujoco-summary.txt"), encoding="utf-8") as fh:
        for line in fh.read().splitlines():
            if line.startswith("[") and line.endswith("]"):
                group = line[1:-1]
                out[group] = {}
            elif line.startswith("  ") and group is not None and " = " in line:
                key, val = line.strip().split(" = ", 1)
                out[group][key] = val
    return out


def tool_shortfall(h, method="rk4", recoil=False, m=0.1, M=90.0, kp=1000.0, kv=20.0,
                   force=100.0, g=9.81, slope=math.radians(9.5), ell=0.8, T=0.1):
    """How far short of its stop the tool ends period 201, from its equation
    of motion: the servo's force kp (ell - l) - kv l', saturated at +-force,
    and the tool's weight along the slide, which on the slope pulls it
    outwards; from rest at its sag on the slope, m g sin(slope) / kp. With
    recoil, the cart (M) is free to move back, and the tool's motion
    relative to it has the reduced mass m M / (m + M)."""
    mass = m * M / (m + M) if recoil else m

    def acc(l, v):
        f = max(-force, min(force, kp * (ell - l) - kv * v))
        return f / mass + g * math.sin(slope)
    l, v = m * g * math.sin(slope) / kp, 0.0
    for _ in range(int(round(T / h))):
        if method == "rk4":
            k1l, k1v = v, acc(l, v)
            k2l, k2v = v + h / 2 * k1v, acc(l + h / 2 * k1l, v + h / 2 * k1v)
            k3l, k3v = v + h / 2 * k2v, acc(l + h / 2 * k2l, v + h / 2 * k2v)
            k4l, k4v = v + h * k3v, acc(l + h * k3l, v + h * k3v)
            l += h / 6 * (k1l + 2 * k2l + 2 * k3l + k4l)
            v += h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        else:                                   # semi-implicit Euler
            v += h * acc(l, v)
            l += h * v
    return ell - l


def tool_facts():
    held, free = tool_shortfall(1e-6), tool_shortfall(1e-6, recoil=True)
    return {
        "tool's last approach, integrated directly (RK4, 1 us): shortfall with the cart held, "
        "free to recoil [m]": f"{held:.3g}, {free:.3g}",
        "the same with RK4 at 10 us, cart held [m] (convergence)": f"{tool_shortfall(1e-5):.4g}",
        "the same by the semi-implicit Euler method at 1, 0.5, 0.25, 0.125 ms, cart held [m]":
            ", ".join(f"{tool_shortfall(h, 'euler'):.3g}" for h in (1e-3, 5e-4, 2.5e-4, 1.25e-4)),
    }


def environment_facts(quick):
    """The environment of the run, from the files kept with the results:
    the installed packages, the record, the console log and the stage."""
    out = {}
    with open(os.path.join(FULL, "pip-freeze.txt"), encoding="utf-8") as fh:
        pins = dict(line.strip().split("==", 1) for line in fh if "==" in line)
    out["installed: isaacsim, mujoco-usd-converter, newton, mujoco"] = ", ".join(
        pins.get(k, "-") for k in ("isaacsim", "mujoco-usd-converter", "newton", "mujoco"))
    out["installed: a learning tool (isaaclab)"] = pins.get("isaaclab", "none")
    out["installed: isaacsim-rl, torch"] = f"{pins.get('isaacsim-rl', '-')}, {pins.get('torch', '-')}"
    with open(os.path.join(FULL, "record.txt"), encoding="utf-8") as fh:
        rec = fh.read().splitlines()
    out["record: solver"] = next(l.strip() for l in rec if l.strip().startswith("PhysX "))
    text = "\n".join(rec)
    out["record: drive limit of the wheels [N m]"] = re.search(r"drive_limit = ([\d.]+)", text).group(1)
    m = {k: float(re.search(rf"{k} = ([\d.e-]+)", text).group(1))
         for k in ("chassis_mass", "wheel_mass", "tool_mass", "aux_mass")}
    out["mass that the drive loop moves, and that the wheels carry [kg]"] = (
        f"{m['chassis_mass'] + 2 * m['wheel_mass'] + m['tool_mass'] + 2 * m['aux_mass']:.6g}, "
        f"{m['chassis_mass'] + 2 * m['wheel_mass'] + m['tool_mass'] + m['aux_mass']:.6g}")
    with open(os.path.join(RES, "isaac-logs", "console-full.txt"), encoding="utf-8") as fh:
        log = fh.read()
    out["console: operating system"] = re.search(r"\| OS: ([^,]+), Version: [\d.]+ \((\w+)\)", log).group(0)[6:]
    out["console: processor"] = re.search(r"\| Processor: (.+)", log).group(1).strip()
    out["console: memory [MB]"] = re.search(r"Total Memory \(MB\): (\d+)", log).group(1)
    with open(os.path.join(FULL, "cart.usda"), encoding="utf-8") as fh:
        usda = fh.read()
    base = re.search(r'def Xform "base".*?double3 xformOp:translate = \(([^,]+),', usda, re.S)
    out["stage: x of the articulation's base [m]"] = base.group(1)
    out["MJCF importer, quick run: error"] = re.search(
        r"'Python runtime error: ([^']+)'", quick["the MJCF importer"]["FAILED"]).group(1)
    out["MJCF importer, full run"] = summary(FULL)["the MJCF importer"].get("skipped", "-")
    return out


def write_periods(path):
    """One row per period, for Figure S11.6: the end differences from the
    nominal motion in both simulators (norm of the state), how far Isaac
    Sim's end position lies beyond its reported speed, and the realized
    acceleration of the front axle less the command, from the reported
    speed over 1 ms, in both simulators."""
    isaac = rows(os.path.join(FULL, "agreement-chapter3-isaac.csv"))
    ref = {r["k"]: r for r in rows(MUJOCO_CSV)}
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["k", "t", "end_isaac", "end_mujoco", "beyond_isaac", "axle_isaac",
                    "axle_mujoco"])
        for r in isaac:
            m = ref[r["k"]]
            w.writerow([r["k"], f"{r['t']:.9g}", f"{r['end']:.9g}", f"{m['end']:.9g}",
                        f"{beyond_reported_speed(r):.9g}", f"{r['w_axle']:.9g}",
                        f"{m['w_axle']:.9g}"])


def main():
    ap = argparse.ArgumentParser(description="Section 11.2: the results, checked without Isaac Sim")
    ap.add_argument("--check", action="store_true", help="check every value quoted in Section 11.2")
    args = ap.parse_args()
    full, out = derived()
    write_periods(os.path.join(RES, "isaac-periods.csv"))
    write_rounding(os.path.join(RES, "isaac-rounding-5m.csv"),
                   os.path.join(RES, "isaac-rounding-period49.csv"))
    lines = ["[derived from the shipped results]"] + [f"  {k} = {v}" for k, v in out.items()]
    failures = []
    if args.check:
        for group, items in EXPECTED.items():
            have = out if group == "derived" else full.get(group, {})
            for key, want in items.items():
                got = have.get(key)
                if got is None:
                    failures.append(f"{group}.{key}: not found")
                elif got != want:
                    failures.append(f"{group}.{key}: expected {want!r}, got {got!r}")
        n = sum(len(v) for v in EXPECTED.values())
        lines.append(f"[check] {n} values quoted in Section 11.2: " +
                     ("all as quoted" if not failures else "FAILED: " + "; ".join(failures)))
    with open(os.path.join(RES, "isaac-derived.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if failures:
        raise SystemExit(1)


EXPECTED = {   # every value quoted in Section 11.2, from the full run's summary or derived here
    'record': {
        'Isaac Sim': '6.2.0-rc.14',
        'Kit': '110.4.0+feature.386251.187cf21b.gl',
        'extension omni.physx': 'omni.physx-110.4.4',
        'extension isaacsim.asset.importer.mjcf': 'isaacsim.asset.importer.mjcf-3.13.0',
        'Python, NumPy': '3.12.15, 2.3.1',
        'OS, machine': 'Windows 11, AMD64',
        'GPU (name, driver, memory)': 'NVIDIA GeForce RTX 4060 Laptop GPU, 596.08, 8188 MiB',
        'default stage: metersPerUnit, kilogramsPerUnit, upAxis': '1, 1, Z',
        'default scene: gravity magnitude': '-inf',
        'default scene: TimeStepsPerSecond': '60',
        'default scene: SolverType': 'TGS',
        'default scene: EnableGPUDynamics': 'True',
        'default scene: BroadphaseType': 'GPU',
        'default scene: FrictionType': 'patch',
        'default rigid body: AngularDamping': '0.05000000074505806',
        'default collider: ContactOffset': '-inf',
        'default collider: RestOffset': '-inf',
        'default articulation: SolverPositionIterationCount': '32',
        'default articulation: SolverVelocityIterationCount': '1',
        'default articulation: SleepThreshold': '4.999999873689376e-05',
        'default articulation: StabilizationThreshold': '9.999999747378752e-06',
        'default articulation: EnabledSelfCollisions': 'True',
    },
    'diagnostics': {
        'wheel loads at rest [N]': '442.54 + 443.96 = 886.504; contact forces point up: True',
        'weight of the parts the wheels carry [N]': '883.891',
        'geometry: largest gap between profile and collision surface [m]':
            '14 positions; off the crest 8.94e-08, on it 1.31e-06',
        'rangefinder less clearance at p 23.7 m, tool folded and unfolded [m]':
            '-1.17e-06, -2.78e-06 (hit /World/wall)',
        'one period at 1 m/s from 5 m: dp [m]': '0.000707',
        'steps per period, adapter': '100 in each, counted',
        'crest, axle coordinate: largest |v - v_ref| [m/s]': '0.00283',
        'crest, ground coordinate [m/s]': 'jump -0.0201 at p 20.1; true speed above v_ref by up '
                                          'to 0.022; more than 15 mm/s above for p in 20.13 to 21.78',
        'capacity: braking on firm ground from 4 m/s [m/s^2]': '0.142',
        'capacity: braking on the slope from 2 m/s [m/s^2]': '0',
        'capacity: from rest towards the start on the slope [m/s^2]': '0.00488',
        'capacity: from rest towards the wall on firm ground [m/s^2]': '2.45',
        'label 2 on firm ground: speed error, largest and change after 0.1 s [mm/s]': '1.35, 1.64',
        'label 0.5 over the end of the crest: speed error, largest and change after 0.1 s [mm/s]':
            '2.78, 4.03',
        'command bound towards the wall: speed error, largest and change after 0.1 s [mm/s]':
            '0.742, 0.819',
        'crest, braking at 0.5 from 2.12 m/s at 20 m':
            'least penetration -0.00379 mm; steps with a wheel unloaded 7; load within 86.8 % of '
            'M(g cos - v^2/(R+r))',
        'crest at a constant 2.3 m/s': 'least penetration -0.00248 mm; steps with a wheel unloaded 4',
        'crest at a constant 2.4 m/s': 'least penetration -0.00248 mm; steps with a wheel unloaded 4',
        'crest at a constant 2.5 m/s': 'least penetration -0.0028 mm; steps with a wheel unloaded 10',
        'crest, rolling freely from 3 m/s': 'least penetration -0.00349 mm; steps with a wheel unloaded 4',
        'encoder less cart speed at 2 m/s^2, drive loop, at 1 ms [mm/s]': '2.57',
        'coasting 1 s from 1 m/s, no torque: speed lost [mm/s]': '0.0985',
        'physics steps per update of the application': '16 (20 times), 17 (40 times); first six '
                                                       'updates 100 steps',
    },
    'the model as PhysX reports it': {
        'physics stepped through': 'simulation manager',
        'physics time step [s]': '0.001',
        'steps per period': '100',
        'link masses [kg]': 'base 0.001, slider_x 0.001, slider_z 0.001, chassis 80, front 5, '
                            'rear 5, tool 0.1',
        'DOF stiffnesses': 'x 0, z 0, pitch 0, front 0, rear 0, reach 1000',
        'DOF dampings': 'x 0, z 0, pitch 0, front 0, rear 0, reach 20',
        'DOF max_forces': 'x 0, z 0, pitch 0, front 0, rear 0, reach 100',
        'penetration at rest [mm]': '-8.94e-06',
        'physics steps counted by the simulation manager, and by the adapter': '3018, 3000',
    },
    'variants': {
        'wall in radians: rangefinder less clearance, folded and unfolded [m]': '0.0569, 0.0569',
        'stage in centimetres: wheel loads at rest, together [stage force units]': '88390.3',
        'physx defaults: coasting 1 s from 1 m/s, speed lost [mm/s]': '2.28',
        'speed servos: encoder less cart speed at 2 m/s^2 [mm/s], by time step':
            '1.74, 18.6, 6.74; damping as PhysX reports it 45.5',
        'servo damping per radian: encoder less cart speed at 2 m/s^2 [mm/s], by time step':
            '-8.96; damping as PhysX reports it 2606.96',
        'drive loop: encoder less cart speed at 0.5 and 0.25 ms [mm/s]': '4.42, 17.2',
        'sled (welded wheels) from 4 m/s':
            '1 ms: front wheel loaded in 500 of 500 steps, largest load 530 N, mean deceleration '
            '2.45 m/s^2; 0.5 ms: front wheel loaded in 1000 of 1000 steps, largest load 609 N, mean '
            'deceleration 2.45 m/s^2; 0.25 ms: front wheel loaded in 2000 of 2000 steps, largest '
            'load 937 N, mean deceleration 2.45 m/s^2',
    },
    'the run of Chapter 3, period by period': {
        'periods': '220',
        'end of period: median, 90th percentile (nearest rank) [m]': '0.00017, 0.000758',
        'end of period: largest [m], period, t [s]': '0.00189, 49, 4.9',
        'end of period, largest: dp, dv, dl': '2.73e-05, 0.00189, 0.000198',
        'periods at rest at 24.5 m: count, median dl [m]': '80, 0.000162',
        'periods at rest at 24.5 m: end differences from, to [m]': '0.000162, 0.000197',
        'accelerating at 2 m/s^2 from rest (period 1): dp, dl [m]': '0.0001, 1.85e-08',
        'tool folded: largest speed difference within a period [m/s], period': '0.00364, 186',
        'within period: largest speed difference elsewhere [m/s], period': '0.00364, 186',
        'within period: largest position difference [m], period': '0.00134, 83',
        'tool unfolds: period, speed difference within it, lowest and highest [m/s]':
            '201, -0.01, 0.00667',
        'tool unfolds: largest reach within the period [m], difference at its end [m]':
            '0.79888, -0.00112',
        'largest reach in any period [m], period': '0.80000, 213',
        'start of a period: largest |dp| + |dl| [m], period; largest |dv| [m/s], period':
            '0.00107, 193; 0.0018, 97',
        'axle: acceleration less command, tool folded: periods above 0.05 m/s^2': '195 of 201',
        'axle: acceleration less command, tool folded, over 1 ms: largest, period, at p':
            '3.02, 67, 22.12',
        'axle: acceleration less command, tool folded, over one step: largest, period': '3.02, 67',
        'jump of the speed signal against the axle in one step: largest [m/s], period, at p':
            '7.28e-07, 60, 20.87',
        'braking claims of the exact run\'s labels: available on the simulated ground': '220 of 220',
        'smallest margin of a label [m/s^2], period, label': '0.232, 65, 0.5',
    },
    'the same at smaller time steps': {
        '0.5 ms: end of period, median, 90th percentile (nearest rank) [m]': '0.000168, 0.00242',
        '0.5 ms: end of period, largest [m], period': '0.00381, 71',
        '0.5 ms: tool unfolds, difference of the reach at the end [m]': '-0.00113',
        '0.5 ms: within period, largest speed difference outside it [m/s], period': '0.00525, 79',
        '0.5 ms: acceleration less command over 1 ms, tool folded, above 0.05 m/s^2':
            '160; largest 4.81 in period 64',
        '0.5 ms: accelerating at 2 m/s^2 from rest (period 1): dp [m]': '4.98e-05',
        '0.5 ms: axle: acceleration less command, tool folded, over 1 ms and one step': '4.81, 9.59',
        '0.5 ms: jump of the speed signal against the axle in one step [m/s], period': '5.95e-07, 58',
        '0.25 ms: end of period, median, 90th percentile (nearest rank) [m]': '0.000309, 0.00636',
        '0.25 ms: end of period, largest [m], period': '0.00969, 87',
        '0.25 ms: tool unfolds, difference of the reach at the end [m]': '-0.00113',
        '0.25 ms: within period, largest speed difference outside it [m/s], period': '0.0101, 201',
        '0.25 ms: acceleration less command over 1 ms, tool folded, above 0.05 m/s^2':
            '154; largest 9.77 in period 187',
        '0.25 ms: accelerating at 2 m/s^2 from rest (period 1): dp [m]': '2.61e-05',
        '0.25 ms: axle: acceleration less command, tool folded, over 1 ms and one step': '9.77, 36.2',
        '0.25 ms: jump of the speed signal against the axle in one step [m/s], period': '7.07e-07, 61',
        'speed gain 100 1/s: largest speed difference within periods 56 to 67 [m/s]': '0.00256',
        'speed gain 200 1/s: largest speed difference within periods 56 to 67 [m/s]': '0.00248',
        'tool unfolds, reach short of the stop, at 0.125, 0.0625 ms [m]': '0.00113, 0.00114',
    },
    'the cases of Section 10.4': {
        'differences with tolerance 0': '26 of 26',
        'differences with tolerance 0.001': '15 of 26',
        'largest difference of a component': '0.00228',
        'largest difference, component': 'p',
    },
    'the configuration of Chapter 3': {
        'declarations of robot, world and control vs model: problems': '0',
        'alarms': '1',
        'first alarm [s]': '0.1',
        'distance that raised it, in the norm of the state': '1.72e-09',
        'measured state at the alarm': '5.96e-09, 1.72e-09, 1.87e-10',
        'set that the rest feedback can reach, after the alarm': 'none',
        'restarts': '0',
        'least clearance [m]': '25',
        'largest |p| over the run [m]': '5.42e-07',
        'cart.metrics: tool unfolds at [s]': '0',
        'largest reach over the run [m]': '2.47e-07',
        'tool operation': 'dropped after an overload at 20 s',
    },
    'derived': {
        'values reported by both runs: identical, in part by the quick run and consistent, '
        'different': '125, 3, 1',
        'values that differ': 'the model as PhysX reports it.stage text, SHA-256',
        'per-period tables of the two runs identical': 'True',
        'the same without the render prims': 'True',
        'stage text without the render prims, SHA-256': 'b8df766ee0f7cf7b',
        'steps counted by the simulation manager before the loop on the clock': '35342',
        'loop on the clock, from the clock\'s definition: as observed': 'True',
        'loop on the clock: periods with 101 or 99 steps': '88',
        'wheel loads at rest less the weight [N], [%]': '2.613, 0.296',
        'wheel loads at rest, front and rear, less MuJoCo\'s [N]': '0.01, 2.61',
        'stage in centimetres: loads / 100 less the weight, relative [%]': '0.0014',
        'servo damping per radian: reported gain over the intended one': '57.3',
        'Isaac Sim: periods whose end position lies beyond the reported speed by more than '
        '0.01, 0.1, 1 mm': '41, 31, 1',
        'Isaac Sim: largest such excess [m], period': '0.00128, 83',
        'MuJoCo: periods whose end position lies beyond the reported speed by more than '
        '0.01, 0.1, 1 mm': '0, 0, 0',
        'Isaac Sim: those beyond by more than 0.01 mm, by phase':
            'moving on the firm ground 3; moving with a wheel on the crest 6; moving on the slope '
            '32; at rest on the slope 0; tool unfolding or unfolded 0',
        'moving with a wheel on the crest: periods; axle over 1 ms, largest [m/s^2], period':
            '13; 3.02, 67',
        'periods with the tool folded': '201',
        'axle over 1 ms, tool folded: median over the periods [m/s^2]': '0.794',
        'MuJoCo: axle over 1 ms, tool folded: median over the periods [m/s^2]': '0.00136',
        'at rest, firm ground: periods; axle over 1 ms, largest [m/s^2], period': '1; 0.0417, 0',
        'at rest on the slope: periods; axle over 1 ms, largest [m/s^2], period': '81; 1.64, 137',
        'axle over 1 ms, accelerating on the firm ground: period 1 (from rest), period 29 '
        '(5.6 m/s) [m/s^2]': '0.133, 1.49',
        'at rest, the tool still: periods; largest |dp|, |dv| at the end [m], [m/s]':
            '100; 3.6e-05, 0.000304',
        'periods in which the speed sets the norm at the end, of 220': '75',
        'end of period above 1 mm: periods': '10',
        'float32 spacing at 25 m [m]': '1.91e-06',
        'float32 spacing at 0.1 m [m]': '7.45e-09',
        'displacement in one step of 1 ms by 0.05 m/s^2 [m]': '2.5e-08',
        'installed: isaacsim, mujoco-usd-converter, newton, mujoco': '6.2.0.0, 0.6.1, 1.6.1, 3.12.0',
        'installed: a learning tool (isaaclab)': 'none',
        'installed: isaacsim-rl, torch': '6.2.0.0, 2.11.0+cu128',
        'loop on the clock: the clock\'s difference after the 400 steps of periods 0 to 3 [s]':
            '0.3999999999999986',
        'start counts from 35,000 to 35,999 that give the same pattern': '200 of 1000',
        'tool\'s last approach, integrated directly (RK4, 1 us): shortfall with the cart held, '
        'free to recoil [m]': '0.00115, 0.00115',
        'the same by the semi-implicit Euler method at 1, 0.5, 0.25, 0.125 ms, cart held [m]':
            '0.00203, 0.00159, 0.00137, 0.00126',
        'crest: v^2 / (R + r) at 2.5 m/s [m/s^2]; speed at which it reaches g cos(slope) [m/s]':
            '0.62; 9.9',
        'record: solver': 'PhysX TGS, 32 position and 1 velocity iterations, GPU dynamics False, '
                          'broadphase MBP, friction patch; contact offset 0.02 m, rest offset 0.0 m',
        'record: drive limit of the wheels [N m]': '40.0',
        'mass that the drive loop moves, and that the wheels carry [kg]': '90.102, 90.101',
        'MuJoCo: periods in which the speed sets the norm at the end, of 220': '12',
        'console: operating system': 'Windows 11 Enterprise, Version: 10.0 (24H2)',
        'console: processor': '13th Gen Intel(R) Core(TM) i7-13700H',
        'console: memory [MB]': '32439',
        'stage: x of the articulation\'s base [m]': '-0.3499999940395355',
        'MJCF importer, quick run: error': 'Mesh ground has no file',
        # the follow-up traces (Section S11.6)
        'traces: record: Isaac Sim; omni.physx; the packages installed as in the first run':
            '6.2.0-rc.14; omni.physx-110.4.4; True',
        'traces: steps that failed':
            'no step failed',
        'traces: comparison with the settings of Section 11.2: periods, columns in common with the shipped table, largest difference':
            '220, 19, 0',
        'traces: from 5 m: steps; displacement per step in single-precision spacings, from, to; steps with 2112, 32 times 66; largest departure from whole spacings (the nine digits printed)':
            '300; 2110, 2114; 254; 0.021',
        'traces: from 5 m: spacings per step if the advance were rounded once per step, at the reported speed, from, to; steps with 2097':
            '2096, 2098; 256',
        'traces: from 5 m: spacings per substep that the reported speed implies, mean; rounded':
            '65.5; 66',
        'traces: from 5 m at exactly 1 m/s: spacings per substep; the rounding\'s excess per substep [spacings], as a speed [mm/s], over a period of 3200 substeps [mm]':
            '65.536; 0.464, 7.08, 0.708',
        'traces: from 5 m, the first period: end beyond 5.1 m [mm]; displacement less 0.1 m, predicted by rounding each substep\'s advance at the reported speed [mm]; observed less predicted [spacings]':
            '0.707; 0.708; -2',
        'traces: from 5 m: rate of the position less reported speed, mean; rounding less reported speed, mean [mm/s]':
            '7.01; 7.01',
        'traces: from 5 m: rate of the position less reported speed, less the rounding: RMS [mm/s]':
            '7.01, 0.198',
        'traces: from 5 m: steps in which the rounding predicts the displacement to within one, two spacings':
            '268 of 300, 300 of 300',
        'traces: from 5 m: encoder less reported speed, mean [mm/s]; over 10 steps, encoder less rate of the position, RMS [mm/s]; per step, RMS [mm/s]':
            '7.11; 0.662; 1.97',
        'traces: period 49: steps; rate of the position less reported speed, from, to [mm/s]; bound of the rounding [mm/s]':
            '150; -29.9, 29.8; 30.5',
        'traces: period 49: steps predicted by the rounding to within one spacing per step; the others: the increment per substep at most this far from half a spacing':
            '142 of 150; 0.013',
        'traces: period 49: rate of the position less reported speed, less the rounding: RMS [mm/s]':
            '17.1, 5.09',
        'traces: period 49: over 10 steps, encoder less reported speed, encoder less rate of the position: RMS [mm/s]':
            '12, 0.901',
        'traces: period 49: change of speed by one spacing per substep, and its time at 2 m/s^2 [mm/s], [ms]':
            '61, 30.5',
        'traces: bound of the rounding with 32 substeps of 1 ms between 4 and 8, 8 and 16, 16 and 32 m [mm/s]':
            '7.63, 15.3, 30.5',
        'traces: the same beyond 16 m with 64 substeps, and with one update per step (PGS) [mm/s]':
            '61, 0.954',
        "traces: the same beyond 16 m for the track coordinate along the slope, the slider's bound divided by cos(9.5 deg), with 32 and 64 substeps and with one update per step (PGS) [mm/s]; the factor less 1 [%]":
            '30.9, 61.9, 0.967; 1.4',
        "traces: the same at 2.25 m, the encoders' diagnostic, at 1, 0.5 and 0.25 ms [mm/s]":
            '3.81, 7.63, 15.3',
        'traces: settings of Section 11.2: position updates per step; periods beyond the reported speed by more than 0.01 mm; largest [mm]':
            '32; 41; 1.28',
        'traces: settings of Section 11.2: reported speed less rate of the position over 10 steps, largest; bound of the rounding beyond 16 m [mm/s]':
            '28; 30.5',
        'traces: settings of Section 11.2: bound of the rounding beyond 16 m for the track coordinate along the slope [mm/s]':
            '30.9',
        'traces: settings of Section 11.2: disturbance read from the reported speed, tool folded: periods above the bound, moving, at rest; median over the moving periods, largest [m/s^2]':
            '119 of 119, 76 of 82; 1.63, 3.02',
        'traces: settings of Section 11.2: end of period, median, 90th percentile (nearest rank), largest [mm], period':
            '0.17, 0.758, 1.89, 49',
        'traces: settings of Section 11.2: start, largest |dp| + |dl| [mm], |dv| [mm/s]; speed within a period outside period 201, largest [mm/s]':
            '1.07, 1.8; 3.64',
        'traces: settings of Section 11.2: periods beyond by more than 0.01 mm that begin beyond 16 m':
            '41 of 41; periods that begin there 176 of 220',
        'traces: no velocity iteration: position updates per step; periods beyond the reported speed by more than 0.01 mm; largest [mm]':
            '32; 37; 1.27',
        'traces: no velocity iteration: reported speed less rate of the position over 10 steps, largest; bound of the rounding beyond 16 m [mm/s]':
            '28.3; 30.5',
        'traces: no velocity iteration: bound of the rounding beyond 16 m for the track coordinate along the slope [mm/s]':
            '30.9',
        'traces: no velocity iteration: disturbance read from the reported speed, tool folded: periods above the bound, moving, at rest; median over the moving periods, largest [m/s^2]':
            '119 of 119, 82 of 82; 1.95, 4.41',
        'traces: no velocity iteration: end of period, median, 90th percentile (nearest rank), largest [mm], period':
            '0.174, 0.958, 2.53, 91',
        'traces: no velocity iteration: start, largest |dp| + |dl| [mm], |dv| [mm/s]; speed within a period outside period 201, largest [mm/s]':
            '1.1, 2.1; 6',
        'traces: four velocity iterations: position updates per step; periods beyond the reported speed by more than 0.01 mm; largest [mm]':
            '32; 36; 1.25',
        'traces: four velocity iterations: reported speed less rate of the position over 10 steps, largest; bound of the rounding beyond 16 m [mm/s]':
            '28.2; 30.5',
        'traces: four velocity iterations: bound of the rounding beyond 16 m for the track coordinate along the slope [mm/s]':
            '30.9',
        'traces: four velocity iterations: disturbance read from the reported speed, tool folded: periods above the bound, moving, at rest; median over the moving periods, largest [m/s^2]':
            '119 of 119, 64 of 82; 2.02, 3.9',
        'traces: four velocity iterations: end of period, median, 90th percentile (nearest rank), largest [mm], period':
            '0.166, 0.833, 2.72, 84',
        'traces: four velocity iterations: start, largest |dp| + |dl| [mm], |dv| [mm/s]; speed within a period outside period 201, largest [mm/s]':
            '1.06, 2.27; 4.77',
        'traces: 64 position iterations: position updates per step; periods beyond the reported speed by more than 0.01 mm; largest [mm]':
            '64; 70; 3.42',
        'traces: 64 position iterations: reported speed less rate of the position over 10 steps, largest; bound of the rounding beyond 16 m [mm/s]':
            '58.2; 61',
        'traces: 64 position iterations: bound of the rounding beyond 16 m for the track coordinate along the slope [mm/s]':
            '61.9',
        'traces: 64 position iterations: disturbance read from the reported speed, tool folded: periods above the bound, moving, at rest; median over the moving periods, largest [m/s^2]':
            '119 of 119, 72 of 82; 2.05, 5.4',
        'traces: 64 position iterations: end of period, median, 90th percentile (nearest rank), largest [mm], period':
            '0.232, 2.42, 3.78, 82',
        'traces: 64 position iterations: start, largest |dp| + |dl| [mm], |dv| [mm/s]; speed within a period outside period 201, largest [mm/s]':
            '2.54, 3.56; 6.06',
        'traces: PGS solver: position updates per step; periods beyond the reported speed by more than 0.01 mm; largest [mm]':
            '1; 0; 0.00227',
        'traces: PGS solver: reported speed less rate of the position over 10 steps, largest; bound of the rounding beyond 16 m [mm/s]':
            '0.844; 0.954',
        'traces: PGS solver: bound of the rounding beyond 16 m for the track coordinate along the slope [mm/s]':
            '0.967',
        'traces: PGS solver: disturbance read from the reported speed, tool folded: periods above the bound, moving, at rest; median over the moving periods, largest [m/s^2]':
            '72 of 119, 0 of 82; 0.0588, 0.745',
        'traces: PGS solver: end of period, median, 90th percentile (nearest rank), largest [mm], period':
            '0.164, 0.276, 1.53, 201',
        'traces: PGS solver: start, largest |dp| + |dl| [mm], |dv| [mm/s]; speed within a period outside period 201, largest [mm/s]':
            '0.245, 0.4; 0.958',
        'traces: PGS solver: the largest disturbance, tool folded: period; where':
            '66; moving with a wheel on the crest',
        'traces: PGS solver: periods beyond the reported speed by more than 1 um, the tolerance of the check of estimation (Section 12.3)':
            '11',
        'traces: contact offset 2 mm: position updates per step; periods beyond the reported speed by more than 0.01 mm; largest [mm]':
            '32; 41; 1.28',
        'traces: contact offset 2 mm: reported speed less rate of the position over 10 steps, largest; bound of the rounding beyond 16 m [mm/s]':
            '28; 30.5',
        'traces: contact offset 2 mm: bound of the rounding beyond 16 m for the track coordinate along the slope [mm/s]':
            '30.9',
        'traces: contact offset 2 mm: disturbance read from the reported speed, tool folded: periods above the bound, moving, at rest; median over the moving periods, largest [m/s^2]':
            '119 of 119, 76 of 82; 1.63, 3.02',
        'traces: contact offset 2 mm: end of period, median, 90th percentile (nearest rank), largest [mm], period':
            '0.17, 0.758, 1.89, 49',
        'traces: contact offset 2 mm: start, largest |dp| + |dl| [mm], |dv| [mm/s]; speed within a period outside period 201, largest [mm/s]':
            '1.07, 1.8; 3.64',
        'traces: contact offset 2 mm: every column of every period as with the settings of Section 11.2':
            'True',
        'traces: filtered loads: position updates per step; periods beyond the reported speed by more than 0.01 mm; largest [mm]':
            '32; 46; 1.32',
        'traces: filtered loads: reported speed less rate of the position over 10 steps, largest; bound of the rounding beyond 16 m [mm/s]':
            '27.8; 30.5',
        'traces: filtered loads: bound of the rounding beyond 16 m for the track coordinate along the slope [mm/s]':
            '30.9',
        'traces: filtered loads: disturbance read from the reported speed, tool folded: periods above the bound, moving, at rest; median over the moving periods, largest [m/s^2]':
            '119 of 119, 76 of 82; 1.66, 3.04',
        'traces: filtered loads: end of period, median, 90th percentile (nearest rank), largest [mm], period':
            '0.194, 0.758, 1.62, 68',
        'traces: filtered loads: start, largest |dp| + |dl| [mm], |dv| [mm/s]; speed within a period outside period 201, largest [mm/s]':
            '1.08, 2.32; 3.2',
        "traces: at rest on the firm ground: steps; first step's loads [N]; total, from, to, mean [N]; mean less the weight [%]":
            '200; 442.54, 443.96; 867.4, 910.2, 883.921; 0.0034',
        'traces: at rest on the firm ground: standard deviation of the total [N]; correlation of successive steps; means over 10 and 100 steps, largest departure from the weight [%]':
            '3.13; -0.42; 0.29, 0.029',
        "traces: at rest on the slope: steps; total, mean [N]; the rigid cart's [N]; difference [%]":
            '500; 871.767; 871.769; -0.0002',
        'traces: from 5 m, the three periods: total load from, to, mean [N]; mean less the weight [%]':
            '774, 1016, 883.89; -9e-05',
        'traces: braking from 4 m/s at 8 m: total load from, to, mean [N]; mean less the weight [%]':
            '742, 1008, 883.37; -0.059',
        'traces: at rest on the firm ground: largest jump of the reported speed in one step [mm/s]':
            '0.0428',
        'traces: at rest on the slope: largest jump of the reported speed in one step [mm/s]':
            '0.0342',
        'traces: period 83: total load in the first ten steps after the placement, from, to [N]; in the rest of the lead-in, from, to [N]':
            '596, 1743; 833, 903',
        "traces: period 83, on the slope: pitch [rad]; over the period, rate of the position less reported speed, mean, RMS [mm/s]; less the rounding of the horizontal slider's advance, mean, RMS [mm/s]":
            '0.1658; -13.3, 13.8; 0.00073, 1.13',
    },
}


if __name__ == "__main__":
    main()
