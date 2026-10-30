"""The declared experiment of Chapter 11, for the ports of the cart
(Section 12.1): the port checks, the run of Chapter 3 period by period,
the cases of Section 10.4 and the configuration of Chapter 3. The functions
take a port of the cart (cart_ports.base.PortedCart) or its simulator
adapter (cart.newton or cart.genesis) and use only what every port offers,
so that the same code runs in both simulators. MuJoCo's own diagnostics and
runs are those of Section 11.1 (../ch11/mujoco_cart.py).
"""
import math
import statistics
from fractions import Fraction as Fr

import allovi
import cart
from allovi import model as am

T = 0.1


def g3(x):
    return f"{x:.3g}"


def norm(dp, dv, dl):
    """The norm of the state, max(|p| + |l|, |v| * 1 s)."""
    return max(abs(dp) + abs(dl), abs(dv))


def ranges(ks):
    """'57 to 66' for a run of consecutive numbers, joined by commas."""
    out, ks = [], sorted(ks)
    i = 0
    while i < len(ks):
        j = i
        while j + 1 < len(ks) and ks[j + 1] == ks[j] + 1:
            j += 1
        out.append(f"{ks[i]}" if i == j else f"{ks[i]} to {ks[j]}")
        i = j + 1
    return ", ".join(out)


# ----------------------------------------------------------------------------
# Port checks: the diagnostics of Section S11.9.1 that catch a mistake of
# conversion, with criteria declared before the runs
# ----------------------------------------------------------------------------
CRITERIA = {
    "loads": "the wheel loads at rest add up to the weight within 0.1 %",
    "timing": "one period at 1 m/s on the firm ground ends within 0.1 mm of 0.1 m further",
    "friction": "the braking that the drive loop sustains on the firm ground is within 1 % "
                "of mu g = 2.4525 m/s^2",
    "crest": "braking at the label 0.5 m/s^2 from 2.12 m/s at 20 m, both wheels carry load "
             "at every step over the crest",
}


def drive(sim, p, v, a, duration, sample=None):
    """Places the cart at (p, v) and holds the constant acceleration a with
    the drive loop for 'duration' seconds; sample(sim, i, v_ref) is called
    before every step."""
    sim.place(p, v, 0.0)
    for i in range(int(round(duration / sim.dt)) + 1):
        v_ref = v + a * i * sim.dt
        if sample is not None:
            sample(sim, i, v_ref)
        sim.controls(a, v_ref, 0.0)
        sim.step()


def tracking(sim, p, v, a, duration=0.5):
    """The errors of the cart's speed against v + a t: the largest, and its
    change after a transient of 0.1 s."""
    errs = []
    drive(sim, p, v, a, duration, lambda s, i, v_ref: errs.append(s.state()[1] - v_ref))
    k = int(round(0.1 / sim.dt))
    return max(abs(e) for e in errs), max(abs(e - errs[k]) for e in errs[k:])


def capacity(sim, p, v, sign):
    """The largest acceleration of the given sign that the drive loop
    sustains from (p, v), by bisection to 0.005 m/s^2, as in Section 11.1."""
    lo, hi = 0.0, 5.0
    while hi - lo > 0.005:
        mid = (lo + hi) / 2
        if tracking(sim, p, v, sign * mid)[1] <= 1e-3:
            lo = mid
        else:
            hi = mid
    return lo


def crest_loads(sim, p, v, a, duration):
    """While the cart crosses the crest from (p, v) under the drive loop at
    the acceleration a: the steps in which a wheel carries no load, while a
    wheel is on the crest, the largest relative difference of the total
    load from M (g cos(pitch) - v^2 / (R + r)) while both are on it, and the
    deepest that an axle on the crest comes below the wheel's radius
    above the ground."""
    pr, r, G = sim.pr, sim.design.wheel_radius, 9.81
    unloaded, worst, n, deepest = 0, 0.0, 0, 0.0
    sim.place(p, v, 0.0)
    for i in range(int(round(duration / sim.dt))):
        sim.controls(a, v + a * i * sim.dt, 0.0)
        sim.step()
        parts = {w: sim.foot(w).part for w in ("front", "rear")}
        if "crest" not in parts.values():
            continue
        loads = sim.loads()
        n += 1
        for w in ("front", "rear"):
            if parts[w] == "crest":
                deepest = max(deepest, r - sim.foot(w).distance)
        if loads is None:
            return None
        unloaded += min(loads) <= 0.0
        if all(x == "crest" for x in parts.values()) and min(loads) > 0.0:
            speed = sim.state()[1]
            expected = sim.mass * (G * math.cos(sim.kin().pitch) - speed ** 2 / (pr.R + r))
            worst = max(worst, abs(sum(loads) - expected) / expected)
    return unloaded, worst, n, deepest


def port_checks(sim):
    """The port checks, each with its value and verdict. A check whose run
    fails, for instance because the simulated state stops being a number,
    is recorded with the failure and fails."""
    out, verdict = {}, {}
    weight = sim.mass * 9.81

    def loads():
        sim.place(5.0, 0.0, 0.0)
        for _ in range(10):
            sim.period(0.0, 0.0)
        ld = sim.loads()
        if ld is None:
            out["wheel loads at rest [N]"] = "not available"
            return "not available"
        out["wheel loads at rest [N]"] = (f"{ld[0]:.5g} + {ld[1]:.5g} = {sum(ld):.6g}; "
                                          f"weight {weight:.6g}")
        return "pass" if abs(sum(ld) - weight) <= 1e-3 * weight else "fail"

    def timing():
        out["penetration at rest [mm]"] = g3(1000 * sim.sink)
        sim.place(5.0, 1.0, 0.0)
        p, v, l = sim.period(0.0, 0.0)
        out["one period at 1 m/s from 5 m: dp [m]"] = g3(p - 5.1)
        out["steps per period, adapter"] = f"{sim.steps}, counted"
        others = []
        for start in (0.5, 15.0):        # where the spacing of the numbers differs
            sim.place(start, 1.0, 0.0)
            others.append(g3(sim.period(0.0, 0.0)[0] - start - 0.1))
        out["one period at 1 m/s from 0.5 m and from 15 m: dp [m]"] = ", ".join(others)
        return "pass" if abs(p - 5.1) <= 1e-4 else "fail"

    def friction():
        cap = capacity(sim, 8.0, 4.0, -1)
        out["capacity: braking on firm ground from 4 m/s [m/s^2]"] = g3(cap)
        sim.place(2.0, 0.0, 0.0)
        for _ in range(5):
            sim.period(2.0, 0.0)
        k = sim.kin()
        creep = k.front_rate * sim.design.wheel_radius - sim.state()[1]
        out["encoder less cart speed at 2 m/s^2 [mm/s]"] = g3(1000 * creep)
        return "pass" if abs(cap - 2.4525) <= 0.01 * 2.4525 else "fail"

    def crest():
        res = crest_loads(sim, 20.0, 2.12, -0.5, 1.4)
        if res is None:
            out["crest, braking at 0.5 from 2.12 m/s at 20 m"] = "loads not available"
            return "not available"
        unloaded, worst, n, deepest = res
        out["crest, braking at 0.5 from 2.12 m/s at 20 m: steps with a wheel unloaded, of steps "
            "on the crest; load difference from the rigid cart's"] = (
            f"{unloaded} of {n}; {g3(100 * worst)} %")
        out["crest, the same: deepest an axle on the crest comes below the wheel's radius "
            "[mm]"] = g3(1000 * deepest)
        return "pass" if unloaded == 0 else "fail"

    for name, fn in (("loads", loads), ("timing", timing), ("friction", friction),
                     ("crest", crest)):
        try:
            verdict[name] = fn()
        except Exception as exc:                   # noqa: BLE001 (record and go on)
            out[f"{name} check failed"] = f"{type(exc).__name__}: {exc}"
            verdict[name] = "fail"
    out["verdicts: loads, timing, friction, crest"] = ", ".join(
        verdict[c] for c in ("loads", "timing", "friction", "crest"))
    return out, verdict


# ----------------------------------------------------------------------------
# The run of Chapter 3, period by period (Section S11.9.2)
# ----------------------------------------------------------------------------
PERIODS = None


def chapter3_periods():
    """(k, state, law, label, segment) of the run of Chapter 3 with the
    exact track (the configuration 'chapter3' of Chapter 10)."""
    cfg = cart.configuration("chapter3", postpone=None)
    rec = allovi.run(cfg, 220, strict=True)
    out = []
    for per in rec.periods:
        segment, label = cfg.control.head(per.plan)
        out.append((per.k, per.state, cfg.control.law(per.plan), label, segment))
    return out


def disturbance(rec, key, win):
    """The largest |realized acceleration - commanded acceleration| over
    windows of 'win' steps, the speed read from rec[i][key], and the
    position at which it occurs."""
    worst, at = 0.0, rec[0]["p"]
    for i in range(len(rec) - win):
        a, b = rec[i], rec[i + win]
        commanded = sum(r["a_ref"] for r in rec[i:i + win]) / win
        dev = abs((b[key] - a[key]) / (b["t"] - a["t"]) - commanded)
        if dev > worst:
            worst, at = dev, a["p"]
    return worst, at


def agreement(world, only=None, progress=None):
    """Each period of the run of Chapter 3 in the simulator of 'world' (or
    those in 'only'), as in Section S11.9.2: placed 0.05 s early on the motion
    of the previous period, driven under the previous law until it starts,
    then executed and compared with the nominal motion of the model. The
    records cover every step of the period, its end included."""
    global PERIODS
    if PERIODS is None:
        PERIODS = chapter3_periods()
    sim = world.sim
    rows = []
    pre = int(round(0.05 / sim.dt))
    win = max(1, int(round(0.001 / sim.dt)))
    previous = (0.0, 0.0, "segment")
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
        sim.period(u_prev, ell_prev, kind_prev, steps=pre)
        p0, v0, l0 = sim.state()
        rec = []
        p1, v1, l1 = sim.period(u, ell, law.kind, record=rec)
        last = rec[-1]
        rec.append(sim.sample(last["a_ref"], last["v_ref"] + last["a_ref"] * sim.dt))
        np_, nv, nl = cart.model.MODEL.dynamics((p0, v0, l0), (u, ell), T)
        dp, dv, dl = p1 - np_, v1 - nv, l1 - nl
        within_p = within_v = low = high = 0.0
        t0 = rec[0]["t"]
        for r in rec:
            tau = r["t"] - t0
            pn, vn = p0 + v0 * tau + u * tau * tau / 2, v0 + u * tau
            within_p = max(within_p, abs(r["p"] - pn))
            within_v = max(within_v, abs(r["v"] - vn))
            low, high = min(low, r["v"] - vn), max(high, r["v"] - vn)
        w, w_at = disturbance(rec, "v", win)
        w_axle, w_axle_at = disturbance(rec, "axle", win)
        w_axle_step = disturbance(rec, "axle", 1)[0]
        load_jump = max((abs(b[key] - a[key]) for a, b in zip(rec, rec[1:])
                         for key in ("front_load", "rear_load")), default=float("nan"))
        speed = float(max(abs(segment.v0), abs(segment.v1)))
        cap = world.capacity(float(segment.lo), float(segment.hi), speed)
        rows.append(dict(k=k, t=k * T, p=p, v=v, l=l, u=u, ell=ell, label=float(label),
                         capacity=cap, start=norm(p0 - p, v0 - v, l0 - l),
                         p1=p1, v1=v1, l1=l1, dp=dp, dv=dv, dl=dl, end=norm(dp, dv, dl),
                         within_p=within_p, within_v=within_v, low=low, high=high,
                         lmax=max(max(r["l"] for r in rec), l1), w=w, w_at=w_at,
                         w_axle=w_axle, w_axle_at=w_axle_at, w_axle_step=w_axle_step,
                         load_jump=load_jump))
        if progress is not None and k % 20 == 0:
            progress(k)
    return rows


def summary(rows):
    ends = sorted(r["end"] for r in rows)
    folded = [r for r in rows if r["l"] == 0 and r["ell"] == 0]
    worst_v = max(rows, key=lambda r: r["within_v"])
    return dict(median=statistics.median(ends), p90=ends[math.ceil(0.9 * len(ends)) - 1],
                worst=max(rows, key=lambda r: r["end"]), worst_v=worst_v,
                worst_v2=max((r for r in rows if r["k"] != worst_v["k"]),
                             key=lambda r: r["within_v"]),
                folded=folded,
                folded_p=max(folded, key=lambda r: r["within_p"]),
                folded_v=max(folded, key=lambda r: r["within_v"]),
                waxle=max(folded, key=lambda r: r["w_axle"]),
                waxle_step=max(folded, key=lambda r: r["w_axle_step"]),
                above_axle=[r["k"] for r in folded
                            if max(r["w_axle"], r["w_axle_step"]) > 0.05],
                unfold=next(r for r in rows if r["l"] == 0 and r["ell"] > 0),
                brake=next(r for r in rows if r["k"] == 40),
                accel=next(r for r in rows if r["k"] == 1),
                rest=[r for r in rows if r["l"] == 0 and r["p"] == 24.5 and r["u"] == 0])


def agreement_facts(rows, reference=None):
    """The values of a comparison; with 'reference', the rows of MuJoCo's own
    run of Section 11.1 (results/agreement-chapter3.csv of ../ch11), also
    the differences of the end states between the two simulators."""
    s = summary(rows)
    worst, unfold = s["worst"], s["unfold"]
    margin = min(rows, key=lambda r: r["capacity"] - r["label"])
    out = {
        "periods": len(rows),
        "end of period: median, 90th percentile (nearest rank) [m]":
            f"{g3(s['median'])}, {g3(s['p90'])}",
        "end of period: largest [m], period": f"{g3(worst['end'])}, {worst['k']}",
        "periods at rest at 24.5 m: end differences from, to [m]":
            (f"{g3(min(r['end'] for r in s['rest']))}, {g3(max(r['end'] for r in s['rest']))}"
             if s["rest"] else "none"),
        "braking at 2 m/s^2 (period 40): dp, dl [m]; end difference [m]":
            f"{g3(s['brake']['dp'])}, {g3(s['brake']['dl'])}; {g3(s['brake']['end'])}",
        "accelerating at 2 m/s^2 from rest (period 1): dp, dl [m]":
            f"{g3(s['accel']['dp'])}, {g3(s['accel']['dl'])}",
        "tool folded: largest position difference within a period [m], period":
            f"{g3(s['folded_p']['within_p'])}, {s['folded_p']['k']}",
        "tool folded: largest speed difference within a period [m/s], period":
            f"{g3(s['folded_v']['within_v'])}, {s['folded_v']['k']}",
        "within period: largest speed difference [m/s], period":
            f"{g3(s['worst_v']['within_v'])}, {s['worst_v']['k']}",
        "tool unfolds: period, reach difference at the end [m], largest reach [m]":
            f"{unfold['k']}, {g3(unfold['dl'])}, {unfold['lmax']:.5f}",
        "axle: acceleration less command, tool folded, over 1 ms: largest, period, at p":
            f"{g3(s['waxle']['w_axle'])}, {s['waxle']['k']}, {s['waxle']['w_axle_at']:.4g}",
        "axle: acceleration less command, tool folded, over one step: largest, period":
            f"{g3(s['waxle_step']['w_axle_step'])}, {s['waxle_step']['k']}",
        "axle: periods with the tool folded above 0.05 m/s^2":
            f"{len(s['above_axle'])} of {len(s['folded'])}",
        "largest change of a wheel's load in one step, tool folded [N], period":
            f"{g3(max(r['load_jump'] for r in s['folded']))}, "
            f"{max(s['folded'], key=lambda r: r['load_jump'])['k']}",
        "speed signal: acceleration less command over 1 ms, tool folded: largest, period":
            f"{g3(max(r['w'] for r in s['folded']))}, "
            f"{max(s['folded'], key=lambda r: r['w'])['k']}",
        "braking claims of the exact run's labels: available on the simulated ground":
            f"{sum(r['capacity'] >= r['label'] for r in rows)} of {len(rows)}",
        "smallest margin of a label [m/s^2], period, label":
            f"{g3(margin['capacity'] - margin['label'])}, {margin['k']}, {margin['label']:g}",
    }
    if reference:
        ref = {int(r["k"]): r for r in reference}
        diffs = []
        for r in rows:
            q = ref.get(r["k"])
            if q is None:
                continue
            # the ends of a period in the two simulators, each from its own start
            dp = r["dp"] - float(q["dp"])
            dv = r["dv"] - float(q["dv"])
            dl = r["dl"] - float(q["dl"])
            diffs.append((norm(dp, dv, dl), r["k"], dp, dv, dl))
        diffs.sort()
        top = diffs[-1]
        out["against MuJoCo 3.16.0 (Section 11.1): end differences less MuJoCo's, median, "
            "largest [m], period"] = (
            f"{g3(statistics.median(d[0] for d in diffs))}, {g3(top[0])}, {top[1]}")
        out["against MuJoCo 3.16.0: at that period, dp, dv, dl less MuJoCo's"] = \
            f"{g3(top[2])}, {g3(top[3])}, {g3(top[4])}"
    return out


# ----------------------------------------------------------------------------
# The cases of Section 10.4 and the configuration of Chapter 3
# ----------------------------------------------------------------------------
def motion_cases_facts(world, cases):
    out = {}
    worst, which, big = 0.0, None, []
    for state, inputs, law in cases:
        theirs = world.step(state, law)
        mine = cart.model.MODEL.dynamics(state, inputs, cart.T)
        diffs = [abs(float(a - b)) for a, b in zip(mine, theirs)]
        for name, dd in zip(("p", "v", "l"), diffs):
            if dd > worst:
                worst, which = dd, name
        if max(diffs) > 1e-3:
            big.append(f"from ({', '.join(f'{float(a):g}' for a in state)}) under "
                       f"({', '.join(f'{float(a):g}' for a in inputs)})")
    for tol in (0, Fr(1, 1000)):
        diffs, n = am.motion_agreement(cart.model.MODEL, world, cases, cart.T, tolerance=tol)
        out[f"differences with tolerance {float(tol):g}"] = f"{len(diffs)} of {n}"
    out["largest difference of a component, component"] = f"{g3(worst)}, {which}"
    out["cases with a difference above 1 mm"] = "; ".join(big) if big else "none"
    return out


def configuration(name, world):
    """The configuration of Chapter 3 (cart.configuration of ../ch10, with
    its defaults) with the given world instance in place of the exact track."""
    task = cart.modules.Operation()
    return allovi.build(name, cart.T, {
        "control": ("cart.constant-command", {"rest": "maximal"}),
        "proposal": ("cart.naive",),
        "capability": ("cart.forecasting", {}),
        "demand": ("cart.task-plan", {"task": task}),
        "task": task,
        "robot": ("cart.robot",),
        "world": world,
        "evaluator": ("cart.metrics", {"task": task}),
    })


def closed_loop_facts(name, world):
    cfg = configuration(name, world)
    out = {"declarations of robot, world and control vs model: problems":
           len(am.agreement(cart.model.MODEL, cfg.robot, cfg.world, cfg.control))}
    rec = allovi.run(cfg, 220, strict=True)
    first = rec.alarms[0] if rec.alarms else None
    out["alarms"] = len(rec.alarms)
    out["first alarm [s]"] = "none" if first is None else g3(first * T)
    if first is not None:
        before, after = rec.periods[first - 1], rec.periods[first]
        dist = cfg.control.distance(after.measured, before.initial_set)
        out["distance that raised it, in the norm of the state"] = g3(float(dist))
    out["restarts"] = len(rec.restarts)
    states = [p.state for p in rec.periods] + [rec.final_state]
    out["least clearance [m]"] = g3(min(float(cfg.world.clearance(s)) for s in states))
    out["largest |p| over the run [m]"] = g3(max(abs(float(s[0])) for s in states))
    out["largest reach over the run [m]"] = g3(max(float(s[2]) for s in states))
    out["tool operation"] = ("dropped after an overload at "
                             f"{g3(cfg.task.overloads[0] * T)} s") if cfg.task.dropped else "done"
    return out
