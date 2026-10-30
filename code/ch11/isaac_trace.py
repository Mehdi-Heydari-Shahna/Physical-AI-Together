#!/usr/bin/env python3
"""
Traces of the cart in Isaac Sim: a follow-up of Section 11.2.

Physical AI, Together, Chapter 11 (Section 11.2).

isaac_cart.py found that the speed that PhysX reports for the front axle is
not always the rate of its position: in 41 of the 220 periods of the run of
Chapter 3, the position at the end of the period lies outside what the
reported speed allows. It also found that the reported speed jumps from one
step to the next by up to 3 mm/s, and that the contact forces of single steps
are not the loads of a rolling cart. Its summary could only bound these
effects. This script records them step by step and repeats the comparison
with settings and a drive loop that could change them:

  * traces, one row per step: the position, the reported speed and the rate
    of the position, the wheels' loads and rates, the pitch, the contacts'
    separations, in six situations: one period at 1 m/s from 5 m without a
    lead-in; the loads at rest on the firm ground; the cart at rest on the
    slope; braking at 2 m/s^2 from 4 m/s on the firm ground; and periods 83
    and 49 of the run of Chapter 3, with their lead-in;
  * the comparison of Section S11.9.2, all 220 periods at 1 ms, with the
    settings of Section 11.2 (which must reproduce the shipped results) and
    with variants: no velocity iteration, four velocity iterations, 64
    position iterations, the PGS solver, a contact offset of 2 mm, and a
    drive loop that shares its force by the wheels' loads filtered over
    10 ms instead of by their loads of the previous step.

Every step runs on its own: a step that fails is reported in errors.txt,
and the others go on.

Usage, with the Python of Isaac Sim 6.2.0, from this folder (code/ch11):
    python isaac_trace.py 2>&1 | tee isaac-trace-console.txt
Options:
    --traces-only        only the traces (a few minutes)
    --variants a,b,...   only these variants of the comparison (names below)

Outputs, in results/isaac-trace/ next to this file:
    summary.txt          every value, by group
    trace-*.csv          one row per step of each trace
    periods-*.csv        one row per period of each variant of the comparison
    errors.txt           the steps that failed, with tracebacks
"""
import argparse
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "isaac-trace")
SHIPPED = os.path.join(HERE, "results", "isaac", "agreement-chapter3-isaac.csv")

VARIANTS = ("settings of Section 11.2", "no velocity iteration", "four velocity iterations",
            "64 position iterations", "PGS solver", "contact offset 2 mm", "filtered loads")

parser = argparse.ArgumentParser(description="Section 11.2: traces in Isaac Sim")
parser.add_argument("--traces-only", action="store_true", help="only the traces")
parser.add_argument("--variants", default="", help="comma-separated names among: " + "; ".join(VARIANTS))
args, _unknown = parser.parse_known_args()

from isaacsim import SimulationApp                                    # noqa: E402

APP = SimulationApp({"headless": True})

import csv                                                            # noqa: E402
import dataclasses                                                    # noqa: E402
import math                                                           # noqa: E402
import statistics                                                     # noqa: E402
import types                                                          # noqa: E402

import numpy as np                                                    # noqa: E402

sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
import cart_isaac as ci                                               # noqa: E402
from cart_isaac import sim as cs                                      # noqa: E402

cs.APP = APP
import allovi                                                         # noqa: E402
import cart                                                           # noqa: E402

T = 0.1
DESIGN = ci.Design()
SETTINGS = ci.IsaacSettings()
FACTS, ERRORS = {}, []
STARTED = time.time()


def g3(x):
    return f"{x:.3g}"


def say(text):
    print(f"[{time.time() - STARTED:7.1f} s] {text}", flush=True)


def norm(dp, dv, dl):
    return max(abs(dp) + abs(dl), abs(dv))


def write_outputs():
    os.makedirs(OUT, exist_ok=True)
    lines = [f"# Section 11.2, follow-up: traces in Isaac Sim; {time.time() - STARTED:.0f} s so far"]
    for name, items in FACTS.items():
        lines.append(f"[{name}]")
        for key, val in items.items():
            lines.append(f"  {key} = {val}")
    with open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT, "errors.txt"), "w", encoding="utf-8") as fh:
        if not ERRORS:
            fh.write("no step failed\n")
        for key, tb in ERRORS:
            fh.write(f"=== {key}\n{tb}\n")


def attempt(out, key, fn):
    try:
        out[key] = fn()
    except Exception as exc:                       # noqa: BLE001 (report and go on)
        out[key] = f"FAILED: {exc!r}"
        ERRORS.append((key, traceback.format_exc()))
        say(f"  failed: {key}: {exc!r}")
    write_outputs()


def write_csv(name, rows):
    if not rows:
        return
    cols = list(rows[0])
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] if isinstance(r[c], (int, str)) else f"{r[c]:.9g}" for c in cols])


# ----------------------------------------------------------------------------
# The cart, with a drive loop that can share its force equally
# ----------------------------------------------------------------------------
class TraceCart(cs.IsaacCart):
    """IsaacCart of Section 11.2, unchanged with sharing="measured"; with
    sharing="filtered", the drive loop shares its force in proportion to
    the wheels' loads filtered over 10 ms (a first-order filter, restarted
    at every placement) instead of their loads of the last step."""
    TAU = 0.01                                     # time constant of the filter [s]

    def __init__(self, design=DESIGN, settings=SETTINGS, sharing="measured"):
        self.sharing, self._lf = sharing, None     # before IsaacCart runs the static sink
        super().__init__(design, settings)

    def place(self, p, v, l):
        self._lf = None
        super().place(p, v, l)

    def controls(self, a_ref, v_ref, ell):
        if self.sharing == "measured":
            return super().controls(a_ref, v_ref, ell)
        d, r = self.design, self.design.wheel_radius
        pitch = self.pitch()
        v = self.state()[1]
        force = self.mass * (a_ref - cs.G * math.sin(pitch) + d.speed_gain * (v_ref - v))
        loads = self.loads()
        if self._lf is None:
            self._lf = list(loads)
        else:
            a = min(1.0, self.dt / self.TAU)
            self._lf = [f + a * (x - f) for f, x in zip(self._lf, loads)]
        total, lim = sum(self._lf), d.drive_limit
        effort = np.zeros((1, self.ndof), dtype=np.float32)
        for wheel, load in zip(("front", "rear"), self._lf):
            share = load / total if total > 0 else 0.5
            tau = force * r * share + self.wheel_inertia * a_ref / r
            effort[0, self.dof[wheel]] = max(-lim, min(lim, tau))
        self.art.set_dof_actuation_forces(effort, self.idx)
        target = np.zeros((1, self.ndof), dtype=np.float32)
        target[0, self.dof["reach"]] = ell
        self.art.set_dof_position_targets(target, self.idx)


def variant_cart(name):
    s, sharing = SETTINGS, "measured"
    if name == "no velocity iteration":
        s = dataclasses.replace(s, velocity_iterations=0)
    elif name == "four velocity iterations":
        s = dataclasses.replace(s, velocity_iterations=4)
    elif name == "64 position iterations":
        s = dataclasses.replace(s, position_iterations=64)
    elif name == "PGS solver":
        s = dataclasses.replace(s, solver="PGS")
    elif name == "contact offset 2 mm":
        s = dataclasses.replace(s, contact_offset=0.002)
    elif name == "filtered loads":
        sharing = "filtered"
    elif name != "settings of Section 11.2":
        raise ValueError(name)
    say(f"  stage: {name}")
    return TraceCart(DESIGN, s, sharing)


def capacity_of(sim):
    """The braking bound of cart.isaac (cart_isaac/world.py) for this cart."""
    holder = types.SimpleNamespace(sim=sim)
    return lambda lo, hi, speed: ci.CartIsaac.capacity(holder, lo, hi, speed)


# ----------------------------------------------------------------------------
# Traces
# ----------------------------------------------------------------------------
def sample(sim, a_ref, v_ref, k, phase):
    row = dict(step=k, phase=phase)
    row.update(sim.sample(a_ref, v_ref))
    sep = sim.separations()
    for w in ("front", "rear"):
        row[f"sep_{w}"] = sep[w] if sep[w] is not None else float("nan")
        row[f"z_{w}"] = sim.position(w)[2]
    return row


def with_rates(rows, dt):
    """Adds the rate of the position, (p_next - p) / dt, and the reported
    speed after the step, to each row but the last."""
    for a, b in zip(rows, rows[1:]):
        a["rate_of_p_over_step"] = (b["p"] - a["p"]) / dt
        a["v_after_step"] = b["v"]
    rows[-1]["rate_of_p_over_step"] = float("nan")
    rows[-1]["v_after_step"] = float("nan")
    return rows


def period_traced(sim, u, ell, kind, rows, phase, steps=None):
    """sim.period, with a row before every step (and none skipped)."""
    v0 = sim.state()[1]
    if abs(v0) < cs.V_REST:
        v0 = 0.0
    if kind == "rest":
        brake = 1 if v0 > 0 else 0
    else:
        brake = 1 if v0 > 0 > u else -1 if v0 < 0 < u else 0
    for i in range(steps or sim.steps):
        tau = i * sim.dt
        v_ref, a_ref = v0 + u * tau, u
        if (kind == "rest" and v0 <= 0) or (brake and v_ref * v0 <= 0):
            v_ref, a_ref = 0.0, 0.0
        rows.append(sample(sim, a_ref, v_ref, len(rows), phase))
        sim.controls(a_ref, v_ref, ell)
        sim.step()


def trace_summary(rows, dt):
    """The largest jump of the reported speed in one step, and the largest
    difference between the reported speed after a step and the rate of
    the position over it, and over windows of 10 steps."""
    jumps = [abs(b["v"] - a["v"]) for a, b in zip(rows, rows[1:])]
    over_step = [abs(r["v_after_step"] - r["rate_of_p_over_step"]) for r in rows[:-1]]
    win = []
    for i in range(len(rows) - 10):
        rate = (rows[i + 10]["p"] - rows[i]["p"]) / (10 * dt)
        mean_v = sum(rows[j]["v"] for j in range(i + 1, i + 11)) / 10
        win.append(abs(rate - mean_v))
    loads = [r["front_load"] + r["rear_load"] for r in rows]
    zero = sum(1 for r in rows if r["front_load"] <= 0 or r["rear_load"] <= 0)
    return (f"largest jump of the reported speed in one step {g3(max(jumps))} m/s; reported speed "
            f"less rate of the position, largest over one step {g3(max(over_step))}, over 10 steps "
            f"{g3(max(win) if win else float('nan'))} m/s; total load from {g3(min(loads))} to "
            f"{g3(max(loads))} N, mean {g3(statistics.mean(loads))} N; steps with a wheel unloaded "
            f"{zero} of {len(rows)}")


def traces(out):
    sim = TraceCart()
    dt = sim.dt
    try:
        def one_period():
            rows = []
            sim.place(5.0, 1.0, 0.0)
            for k in range(3):
                period_traced(sim, 0.0, 0.0, "segment", rows, f"period {k}")
            rows.append(sample(sim, 0.0, rows[-1]["v_ref"], len(rows), "end"))
            write_csv("trace-one-period-1ms-from-5m.csv", with_rates(rows, dt))
            first = [r for r in rows if r["phase"] == "period 0"] + [rows[100]]
            return (f"p after the first period less 5.1 m {g3(rows[100]['p'] - 5.1)} m; mean reported "
                    f"speed over it {g3(statistics.mean(r['v_after_step'] for r in first[:-1]))} m/s; "
                    + trace_summary(rows, dt))
        attempt(out, "one period at 1 m/s from 5 m, no lead-in, three periods", one_period)

        def rest_firm():
            rows = []
            sim.place(5.0, 0.0, 0.0)
            for _ in range(10):
                sim.period(0.0, 0.0)
            for k in range(2):
                period_traced(sim, 0.0, 0.0, "segment", rows, f"period {k}")
            write_csv("trace-rest-firm-ground.csv", with_rates(rows, dt))
            return trace_summary(rows, dt)
        attempt(out, "at rest on the firm ground at 5 m, after 1 s", rest_firm)

        def rest_slope():
            rows = []
            sim.place(24.5, 0.0, 0.0)
            for _ in range(10):
                sim.period(0.0, 0.0)
            for k in range(5):
                period_traced(sim, 0.0, 0.0, "segment", rows, f"period {k}")
            write_csv("trace-rest-slope.csv", with_rates(rows, dt))
            return trace_summary(rows, dt)
        attempt(out, "at rest on the slope at 24.5 m, after 1 s", rest_slope)

        def braking():
            rows = []
            sim.place(8.0, 4.0, 0.0)
            for i in range(int(round(0.5 / dt)) + 1):
                v_ref = 4.0 - 2.0 * i * dt
                rows.append(sample(sim, -2.0, v_ref, i, "braking"))
                sim.controls(-2.0, v_ref, 0.0)
                sim.step()
            write_csv("trace-braking-2-from-4.csv", with_rates(rows, dt))
            return trace_summary(rows, dt)
        attempt(out, "braking at 2 m/s^2 from 4 m/s at 8 m", braking)

        for k in (83, 49):
            def period_k(k=k):
                rows = run_periods(sim, only={k}, trace_rows=[])[1]
                write_csv(f"trace-period-{k}.csv", with_rates(rows, dt))
                body = [r for r in rows if r["phase"] == f"period {k}"]
                return trace_summary(body + [rows[-1]], dt)
            attempt(out, f"period {k} of the run of Chapter 3, with its lead-in", period_k)
    finally:
        sim.close()


# ----------------------------------------------------------------------------
# The comparison of Section S11.9.2 (as isaac_cart.agreement, with two more
# columns computed from the same samples)
# ----------------------------------------------------------------------------
PERIODS = None


def chapter3_periods():
    cfg = cart.configuration("chapter3", postpone=None)
    rec = allovi.run(cfg, 220, strict=True)
    out = []
    for per in rec.periods:
        segment, label = cfg.control.head(per.plan)
        out.append((per.k, per.state, cfg.control.law(per.plan), label, segment))
    return out


def disturbance(rec, key, win):
    worst = 0.0
    for i in range(len(rec) - win):
        a, b = rec[i], rec[i + win]
        commanded = sum(r["a_ref"] for r in rec[i:i + win]) / win
        worst = max(worst, abs((b[key] - a[key]) / (b["t"] - a["t"]) - commanded))
    return worst


def run_periods(sim, only=None, trace_rows=None):
    """Each period (or those in 'only') placed 0.05 s early on the motion of
    the previous period, driven under the previous law until it starts,
    then executed and compared with the nominal motion of the model."""
    global PERIODS
    if PERIODS is None:
        PERIODS = chapter3_periods()
    rows = []
    pre = int(round(0.05 / sim.dt))
    win = max(1, int(round(0.001 / sim.dt)))
    previous = (0.0, 0.0, "segment")
    capacity = capacity_of(sim)
    for k, x, law, label, segment in PERIODS:
        p, v, l = (float(a) for a in x)
        u_prev, ell_prev, kind_prev = previous
        u = float(law.u)
        ell = l if law.ell is None else float(law.ell)
        previous = (u, ell, law.kind)
        if only is not None and k not in only:
            continue
        dt = pre * sim.dt
        sim.place(p - v * dt + u_prev * dt * dt / 2, v - u_prev * dt, l)
        if trace_rows is not None:
            period_traced(sim, u_prev, ell_prev, kind_prev, trace_rows, "lead-in", steps=pre)
        else:
            sim.period(u_prev, ell_prev, kind_prev, steps=pre)
        p0, v0, l0 = sim.state()
        rec = []
        if trace_rows is not None:
            period_traced(sim, u, ell, law.kind, trace_rows, f"period {k}")
            p1, v1, l1 = sim.state()
            trace_rows.append(sample(sim, u, trace_rows[-1]["v_ref"] + u * sim.dt, len(trace_rows),
                                     "end"))
            rec = [dict(t=r["t"], p=r["p"], v=r["v"], l=r["l"], a_ref=r["a_ref"], axle=r["axle"])
                   for r in trace_rows if r["phase"] in (f"period {k}", "end")]
        else:
            p1, v1, l1 = sim.period(u, ell, law.kind, record=rec)
            last = rec[-1]
            rec.append(sim.sample(last["a_ref"], last["v_ref"] + last["a_ref"] * sim.dt))
        np_, nv, nl = cart.model.MODEL.dynamics((p0, v0, l0), (u, ell), T)
        dp, dv, dl = p1 - np_, v1 - nv, l1 - nl
        low = high = 0.0
        t0 = rec[0]["t"]
        for r in rec:
            tau = r["t"] - t0
            vn = v0 + u * tau
            low, high = min(low, r["v"] - vn), max(high, r["v"] - vn)
        band_lo = low * T + u * sim.dt * T / 2
        band_hi = high * T + u * sim.dt * T / 2
        beyond = max(0.0, dp - band_hi, band_lo - dp)
        mism = 0.0
        for i in range(len(rec) - 10):
            rate = (rec[i + 10]["p"] - rec[i]["p"]) / (10 * sim.dt)
            mean_v = sum(rec[j]["v"] for j in range(i + 1, i + 11)) / 10
            mism = max(mism, abs(rate - mean_v))
        rows.append(dict(k=k, t=k * T, p=p, v=v, l=l, u=u, ell=ell, label=float(label),
                         capacity=capacity(float(segment.lo), float(segment.hi),
                                           float(max(abs(segment.v0), abs(segment.v1)))),
                         start_pl=abs(p0 - p) + abs(l0 - l), start_v=abs(v0 - v),
                         dp=dp, dv=dv, dl=dl, end=norm(dp, dv, dl), low=low, high=high,
                         w_axle=disturbance(rec, "axle", win),
                         w_axle_step=disturbance(rec, "axle", 1),
                         beyond=beyond, mismatch_10_steps=mism))
        if only is None and k % 40 == 0:
            say(f"    period {k}")
    return rows, (trace_rows if trace_rows is not None else [])


def comparison_facts(rows, out):
    ends = sorted(r["end"] for r in rows)
    folded = [r for r in rows if r["l"] == 0 and r["ell"] == 0]
    out["end of period: median, 90th percentile (nearest rank), largest [m], period"] = (
        f"{g3(statistics.median(ends))}, {g3(ends[math.ceil(0.9 * len(ends)) - 1])}, "
        f"{g3(ends[-1])}, {max(rows, key=lambda r: r['end'])['k']}")
    out["end position beyond the reported speed by more than 0.01, 0.1, 1 mm: periods; largest [m], "
        "period"] = (", ".join(str(sum(r["beyond"] > t for r in rows)) for t in (1e-5, 1e-4, 1e-3))
                     + f"; {g3(max(r['beyond'] for r in rows))}, "
                       f"{max(rows, key=lambda r: r['beyond'])['k']}")
    out["reported speed less rate of the position over 10 steps: median, largest [m/s], period"] = (
        f"{g3(statistics.median(r['mismatch_10_steps'] for r in rows))}, "
        f"{g3(max(r['mismatch_10_steps'] for r in rows))}, "
        f"{max(rows, key=lambda r: r['mismatch_10_steps'])['k']}")
    out["reported speed of the axle less command, tool folded, over 1 ms or a step: periods above "
        "0.05 m/s^2; median; largest [m/s^2]"] = (
        f"{sum(max(r['w_axle'], r['w_axle_step']) > 0.05 for r in folded)} of {len(folded)}; "
        f"{g3(statistics.median(r['w_axle'] for r in folded))}; "
        f"{g3(max(max(r['w_axle'], r['w_axle_step']) for r in folded))}")
    out["start of a period: largest |dp| + |dl| [m], largest |dv| [m/s]"] = (
        f"{g3(max(r['start_pl'] for r in rows))}, {g3(max(r['start_v'] for r in rows))}")


def compare_with_shipped(rows):
    if not os.path.exists(SHIPPED):
        return "no shipped results found"
    with open(SHIPPED, newline="", encoding="utf-8") as fh:
        ref = {int(r["k"]): r for r in csv.DictReader(fh)}
    worst = 0.0
    for r in rows:
        q = ref[r["k"]]
        worst = max(worst, max(abs(r[c] - float(q[c])) for c in ("dp", "dv", "dl")))
    return f"largest difference of dp, dv or dl from the shipped run [m]: {worst:.3g}"


def variants(out):
    chosen = [v.strip() for v in args.variants.split(",") if v.strip()] or list(VARIANTS)
    for name in chosen:
        group = FACTS.setdefault(f"comparison: {name}", {})

        def run(name=name, group=group):
            sim = variant_cart(name)
            try:
                rows = run_periods(sim)[0]
            finally:
                sim.close()
            write_csv("periods-" + name.replace(" ", "-").replace(".", "") + ".csv", rows)
            comparison_facts(rows, group)
            if name == "settings of Section 11.2":
                group["against the shipped run"] = compare_with_shipped(rows)
            return "done"
        say(f"comparison: {name}")
        attempt(group, "status", run)


def record(out):
    try:
        from isaacsim.core.version import get_version
        v = get_version()
        out["Isaac Sim"] = v[0] + (f"-{v[1]}" if v[1] else "")
    except Exception as exc:                       # noqa: BLE001
        out["Isaac Sim"] = f"unknown ({exc!r})"
    try:
        import omni.kit.app
        mgr = omni.kit.app.get_app().get_extension_manager()
        out["extension omni.physx"] = mgr.get_enabled_extension_id("omni.physx") or "-"
    except Exception:                              # noqa: BLE001
        out["extension omni.physx"] = "-"
    out["settings of Section 11.2"] = ", ".join(f"{k} = {v}" for k, v in SETTINGS.as_dict().items())
    out["single-precision spacing of positions at 5 m and at 25 m [m]"] = "4.77e-07, 1.91e-06"
    out["so the rate of the position resolves, over one step and over 10 steps, at 25 m [m/s]"] = (
        "about 1.9e-03 and 3.8e-04")


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, fn in (("record", record), ("traces", traces)) + (
            () if args.traces_only else (("variants", variants),)):
        say(name)
        out = FACTS.setdefault(name, {})
        try:
            fn(out)
        except Exception as exc:                   # noqa: BLE001
            out["FAILED"] = repr(exc)
            ERRORS.append((name, traceback.format_exc()))
            say(f"  failed: {name}: {exc!r}")
        write_outputs()
    say(f"done: {len(ERRORS)} steps failed; results in {OUT}")


if __name__ == "__main__":
    try:
        main()
    finally:
        write_outputs()
        APP.close()
