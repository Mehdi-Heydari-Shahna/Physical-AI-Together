#!/usr/bin/env python3
"""
The Husky under the governor of Chapter 10: what it knows of where it is, how
it follows the route, where it stops, and what it does when the grass turns
slippery.

Physical AI, Together, Chapter 13 (Section 13.2).

This script runs the Husky of the package husky in the yard, through the
allovi core of Chapter 10 (../ch10), with the modules of husky/governed.py,
the estimators of husky/estimate.py and the feedback of husky/follow.py, and
  * writes the record of the experiment, with the lane's allowance and
    reserve, which it computes and checks against the constants of
    governed.py;
  * calibrates what the estimators need: the rolling radius, against the
    value of Section 13.1, the sideslip of the turn on each ground, checked
    in a turn to the right, and the gyro's bias at rest;
  * crosses the yard under the governor with each of the five estimators,
    the map's labels and the proposal husky.cruise, and measures the
    estimate's errors, the feedback's bounds, the state's distance from the
    commanded motion, the footprint against the lane and the clearance from
    the dock;
  * crosses it with the fixed, reactive and map predictors, and, in a shower
    that turns the wet grass slippery from 20 s on, with the map and with a
    forecast that announces the shower at 10 s;
  * stops the robot at its labels from 0.25 to 1 m/s on each ground, in dry
    weather and in the shower, and at the global capability in the shower,
    and checks the bounds that a label promises;
  * repeats the crossing with the map at time steps of 1 and 0.25 ms, and
    with twenty seeds of the gyro's noise.

Usage
    python3 husky_route.py            # writes results/
    python3 husky_route.py --check    # also checks every value quoted in
                                      # Section 13.2

Outputs, in results/ next to this file
    route-record.txt        the record of the experiment
    route-summary.txt       every value, by group; with --check, the
                            comparison with the values quoted in Section 13.2
    route-errors.csv        the crossing with the map: the estimators' errors
                            along the route, every 0.1 s (Figure 13.5)
    route-approach.csv      the last 4 s before rest at the dock, dry and in
                            the shower, every 20 ms (Figure 13.8)
    route-decisions.csv     the decisions of the crossing with the map and the
                            estimator with sideslip, in the format of
                            Chapter 8, written by the core

Requires Python 3.10 or later, MuJoCo 3.16.0 and NumPy (requirements.txt),
and the allovi core of ../ch10, which needs nothing else.
"""
import argparse
import csv
import math
import os
import platform
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mujoco                                     # noqa: E402
import numpy as np                                # noqa: E402

import husky_mujoco as h131                       # noqa: E402  (Section 13.1's checks)
from husky import drive, estimate, follow, governed as hg, model, sim, world   # noqa: E402
import allovi                                     # noqa: E402  (from ../ch10, via governed)
from allovi import records                        # noqa: E402

G = 9.81
SHOWER = dict(t_start=20.0, frictions={"wet grass": 0.15}, announced=10.0)
PERIODS = 170                                     # 34 s
STOP_SPEEDS = (0.25, 0.5, 0.75, 1.0)
SEEDS = tuple(range(1, 21))
COMPONENTS = "along [m], in speed [m/s], across [m], in heading [rad]"


def shower():
    return hg.Shower(SHOWER["t_start"], SHOWER["frictions"], SHOWER["announced"])


def four(values, digits=(4, 3, 4, 4)):
    return ", ".join(f"{v:.{n}f}" for v, n in zip(values, digits))


def kept(m):
    """Whether a run kept the estimate's bounds and the feedback's, every
    component (the values of husky.metrics, or of stop_run)."""
    est = (m["error_s"] <= hg.E_S and m["error_u"] <= hg.E_U and m["error_d"] <= hg.E_D and
           m["error_e"] <= hg.E_E)
    fb = (m["track_s"] <= hg.F_S and m["track_u"] <= hg.F_U and m["track_d"] <= hg.F_D and
          m["track_e"] <= hg.F_E)
    return est, fb


def in_funnel(m):
    return (m["true_s"] <= hg.DS and m["true_u"] <= hg.DU and m["true_d"] <= hg.DD and
            m["true_e"] <= hg.DE)


def yes(x):
    return "yes" if x else "no"


# ----------------------------------------------------------------------------
# One crossing of the yard under the governor
# ----------------------------------------------------------------------------
def crossing(name, capability="husky.map", estimator="sideslip", weather=None, dt=None,
             periods=PERIODS):
    """Runs the configuration; returns the record, the world adapter, the
    configuration and the metrics of husky.metrics."""
    yard = hg.Yard(estimator=estimator, shower=weather, dt=dt)
    cap = (capability,) if capability != "husky.forecast" else \
        ("husky.forecast", {"shower": weather or shower()})
    cfg = allovi.build(name, hg.T, {
        "control": ("husky.route-speed",), "proposal": ("husky.cruise",),
        "capability": cap, "robot": ("husky.robot",), "world": yard,
        "evaluator": hg.Metrics(yard)})
    rec = allovi.run(cfg, periods=periods, strict=True, diagnostics=True)
    return rec, yard, cfg, cfg.evaluator.evaluate(rec)


def committed(rec):
    """Periods in which the committed candidate had ambition 1, less than 1
    but more than 0, and 0 (recovery or fallback)."""
    amb = [p.decision.ambition for p in rec.periods if p.decision is not None]
    return (sum(a == 1.0 for a in amb), sum(0.0 < a < 1.0 for a in amb),
            sum(a == 0.0 for a in amb))


def decisions(rec):
    """Per period: the committed ambition, level and point of rest."""
    return [(p.k, p.decision.ambition, p.decision.profile.level, p.decision.plan.s_rest)
            if p.decision is not None and p.decision.plan is not None else (p.k, None)
            for p in rec.periods]


def final_level(rec):
    return rec.periods[-1].decision.profile.level


def fmt_touch(m):
    if not m["touches"]:
        return "nothing"
    t, what, s = m["touches"][0]
    return f"{what} at {t:.2f} s, s = {s:.2f} m"


# ----------------------------------------------------------------------------
# The record
# ----------------------------------------------------------------------------
def record_facts():
    out = {}
    out["MuJoCo"] = mujoco.__version__
    out["NumPy, Python"] = f"{np.__version__}, {platform.python_version()}"
    out["allovi core, contract"] = f"{allovi.__version__}, {allovi.CONTRACT}"
    out["governor's period [s]; updates of the drive controller per period"] = (
        f"{hg.T:g}; {int(round(hg.T * 50))}")
    dr = drive.A300Drive()
    out["the drive controller: wheel radius, separation [m]; limits of acceleration, "
        "deceleration [m/s^2]"] = (f"{dr.radius_left:g}, {dr.separation:g}; "
                                   f"{dr.linear.max_pos:g}, {-dr.linear.min_pos:g}")
    out["plans: speed limit [m/s], acceleration [m/s^2], deceleration at most [m/s^2]"] = (
        f"{hg.V_MAX:g}, {hg.A_MAX:g}, {hg.B_MAX:g}")
    out["the state (s, u, d, e): the norm's scales of the speed [s] and the heading error [m]; "
        "the measurement-error bound VARPI [m]"] = (
        f"{hg.TAU_U:g}, {hg.RHO_E:g}; {hg.VARPI:g}")
    out[f"bounds of the estimate E_S, E_U, E_D, E_E: {COMPONENTS}"] = (
        f"{hg.E_S:g}, {hg.E_U:g}, {hg.E_D:g}, {hg.E_E:g}")
    out[f"bounds of the feedback F_S, F_U, F_D, F_E: {COMPONENTS}"] = (
        f"{hg.F_S:g}, {hg.F_U:g}, {hg.F_D:g}, {hg.F_E:g}")
    out[f"the funnel DS, DU, DD, DE: {COMPONENTS}"] = (
        f"{hg.DS:g}, {hg.DU:g}, {hg.DD:g}, {hg.DE:g}")
    out["the wheels: half the wheelbase, half the track [m]; a label's reach beyond the "
        "commanded arc lengths REACH [m]"] = (
        f"{hg.HALF_WHEELBASE:g}, {hg.HALF_TRACK:g}; {hg.REACH:.3f}")
    out["levels: floor, step [m]; the dock zone [m]; ambitions"] = (
        f"{hg.FLOOR:g}, {hg.GRID:g}; {hg.DOCK_ZONE:g}; " + ", ".join(f"{a:g}" for a in hg.ALPHAS))
    out["the dock: the footprint's reach ahead of the centre at the heading error DE [m]; "
        "a plan's room beyond its point of rest for the floor's reserve [m]"] = (
        f"{hg.front(hg.DE):.4f}; {hg.DS + hg.FLOOR + hg.front(hg.DE + hg.FLOOR / hg.RHO_E):.4f}")
    out["the dock: the heading error up to which the footprint's reach grows, atan(w / l) "
        "[rad]; the reach there, half the footprint's diagonal [m]; the largest reserve whose "
        "heading error DE + eps / RHO_E stays within it [m]"] = (
        f"{hg.PSI_STAR:.4f}; {hg.HALF_DIAGONAL:.4f}; {hg.RHO_E * (hg.PSI_STAR - hg.DE):.4f}")
    out["the speed: the plans' commanded speed at most [m/s]; the robot's speed by the "
        "funnel's bound at most [m/s]; the commanded speed that would keep the robot's speed "
        "within the limit by that bound [m/s]"] = (
        f"{hg.V_MAX:g}; {hg.V_MAX + hg.DU:g}; {hg.V_MAX - hg.DU:g}")
    a0, a1 = hg.lane_allowance(0.0), hg.lane_allowance(hg.DE)
    r = hg.lane_reserve()
    out["lane: the narrowest allowance of the centre at no heading error and at DE [m]; the "
        "constant [m]"] = f"{a0:.3f}, {a1:.3f}; {hg.LANE_ALLOWANCE:.3f}"
    out["lane: its reserve in the norm [m], the constant; the offset and heading error it "
        "covers [m], [rad]; the allowance there [m]"] = (
        f"{r:.3g}, {hg.LANE_RESERVE:.3g}; {hg.DD + hg.D_SCALE * r:.4f}, "
        f"{hg.DE + r / hg.RHO_E:.4f}; {hg.lane_allowance(hg.DE + r / hg.RHO_E):.4f}")
    out["lane: the constants agree with the computation, to 3 digits"] = str(
        f"{a1:.3f}" == f"{hg.LANE_ALLOWANCE:.3f}" and f"{r:.3g}" == f"{hg.LANE_RESERVE:.3g}")
    out["grounds: friction of asphalt, gravel, wet grass; their labels [m/s^2]; the share of "
        "mu g; the global capability [m/s^2]"] = (
        ", ".join(f"{g.friction:g}" for g in world.GROUNDS) + "; " +
        ", ".join(f"{hg.label_of(g.friction):.3g}" for g in world.GROUNDS) +
        f"; {hg.ETA:g}; {hg.B_GLOBAL:g}")
    sh = shower()
    out["the shower: from [s], announced at [s]; friction of the wet grass; its label "
        "[m/s^2]"] = (f"{sh.t_start:g}, {sh.announced:g}; {sh.frictions['wet grass']:g}; "
                      f"{hg.label_of(sh.frictions['wet grass']):.3g}")
    out["feedback gains: K_S [1/s], K_D [1/m^2], K_E [1/m], W_MIN [m/s], K_I [1/s]"] = (
        f"{follow.K_S:g}, {follow.K_D:g}, {follow.K_E:g}, {follow.W_MIN:g}, {follow.K_I:g}")
    out["gyro: bias [rad/s], noise per reading at 50 Hz [rad/s], seed; calibration at rest "
        "[s]"] = (f"{estimate.GYRO_BIAS:g}, {estimate.GYRO_NOISE:g}, {estimate.SEED}; "
                  f"{estimate.CALIBRATION:g}")
    out["stops: from rest at [m] on asphalt, gravel, wet grass"] = ", ".join(
        f"{s:g}" for s in (1.0, 15.0, 19.0))
    return out


# ----------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------
def calibration_facts():
    out = {}
    # the rolling radius, as Section 13.1.4 measured it
    h = sim.Husky(ground=0.8, obstacles=False)
    h.place()
    h.command = (1.0, 0.0)
    h.run(2.0)
    x0 = h.pose()[0]
    q0 = np.mean([h.d.qpos[h.qadr[w]] for w in model.WHEELS])
    h.run(10.0)
    rho = (h.pose()[0] - x0) / (np.mean([h.d.qpos[h.qadr[w]] for w in model.WHEELS]) - q0)
    out["rolling radius on asphalt over 10 s at the command 1 m/s [m]; the estimator's"] = (
        f"{rho:.5f}; {estimate.RHO:.5f}")
    # the sideslip of the turn of Section 13.1.3
    ks = []
    r_model = h.wheel_radius()                          # the rim speeds' radius
    for g in world.GROUNDS:
        h131.turn_check(g.friction, inertia=True)
        u, v, w = h131.turn_check.motion
        rim = sum(h131.turn_check.rims) / 2.0
        counted = rim * estimate.RHO / r_model          # what the estimators count
        out[f"{g.name}: turn of radius 3 m at 1 m/s: speed [m/s]; the wheels' mean rim speed "
            f"[m/s], short of it by [%]; counted at the rolling radius [m/s], short by [%]"] = (
            f"{u:.4f}; {rim:.4f}, {100 * (1 - rim / u):.2f}; {counted:.4f}, "
            f"{100 * (1 - counted / u):.2f}")
        x = -v / w
        k = x / (u * abs(w))
        ks.append((g.name, u, v, w, x, k))
        out[f"{g.name}: turn of radius 3 m at 1 m/s: sideways speed [m/s], yaw rate [rad/s], "
            f"short of the command 1/3 rad/s by [%]; centre of rotation ahead of the centre "
            f"[m]; K [s^2], the estimator's"] = (
            f"{v:+.4f}, {w:.4f}, {100 * (1 - 3 * w):.1f}; {x:.4f}; {k:.3f}, "
            f"{estimate.SIDESLIP[g.name]:.3f}")
        h131.turn_check(g.friction, w=-1.0 / 3.0)
        u_r, v_r, w_r = h131.turn_check.motion
        out[f"{g.name}: the same turn to the right: sideways speed [m/s], yaw rate [rad/s]; "
            f"centre of rotation ahead of the centre [m]"] = (
            f"{v_r:+.4f}, {w_r:+.4f}; {-v_r / w_r:.4f}")
    out["sideslip constants of the estimator agree with the turns, to 3 digits"] = str(
        all(f"{k:.3f}" == f"{estimate.SIDESLIP[n]:.3f}" for n, *_, k in ks))
    # the gyro's bias at rest
    h = sim.Husky()
    h.place()
    est = estimate.DeadReckoning(h)
    bias = est.calibrate()
    out["gyro: bias estimated over 2 s at rest [rad/s]; its error [rad/s]"] = (
        f"{bias:.5f}; {bias - estimate.GYRO_BIAS:+.1e}")
    return out


# ----------------------------------------------------------------------------
# The estimators
# ----------------------------------------------------------------------------
ESTIMATORS = (("a300", "the A300's odometry"),
              ("controller radius", "dead reckoning with sideslip at the controller's radius"),
              ("uncalibrated", "dead reckoning with sideslip, the gyro not calibrated"),
              ("dead reckoning", "dead reckoning"),
              ("sideslip", "dead reckoning with sideslip"))


def estimator_facts(out_dir):
    out, traces = {}, {}
    i = {k: n for n, k in enumerate(hg.Yard.TRACE)}
    for kind, label in ESTIMATORS:
        rec, yard, cfg, m = crossing(kind, estimator=kind)
        traces[kind] = yard.trace
        out[f"{label}: the estimate's largest error {COMPONENTS}"] = four(
            (m["error_s"], m["error_u"], m["error_d"], m["error_e"]))
        out[f"{label}: the estimate's largest distance from the commanded motion {COMPONENTS}"] = \
            four((m["track_s"], m["track_u"], m["track_d"], m["track_e"]))
        out[f"{label}: the state's largest distance from the commanded motion {COMPONENTS}; "
            f"within the funnel"] = four(
            (m["true_s"], m["true_u"], m["true_d"], m["true_e"])) + f"; {yes(in_funnel(m))}"
        first = f", the first at {rec.alarms[0] * hg.T:.1f} s" if rec.alarms else ""
        out[f"{label}: the footprint's largest offset from the centre line [m]; touches; "
            f"alarms, restarts"] = (
            f"{m['footprint']:.3f}; {fmt_touch(m)}; {len(m['alarms'])}{first}, "
            f"{len(m['restarts'])}")
        if rec.alarms:
            k = rec.alarms[0] - 1                   # the period whose check raised it
            ini, mk = rec.periods[k].initial_set, rec.periods[k + 1].measured
            out[f"{label}: the first alarm: the estimate beyond the initial set along [m], in "
                f"speed [m/s], across [m], in heading [rad]; in the norm [m]"] = (
                four((abs(mk.s - ini.s) - ini.width, abs(mk.u - ini.c) - hg.DU,
                      abs(mk.d) - hg.DD, abs(mk.e) - hg.DE)) +
                f"; {cfg.control.distance(mk, ini):.4f}")
        out[f"{label}: at the end: arc length, its estimate [m]; clearance of the footprint "
            f"from the dock [m]"] = (
            f"{m['final_s']:.4f}, {m['final_s_hat']:.4f}; {m['final_clearance']:.4f}")
        if kind == "sideslip":
            write_decisions(os.path.join(out_dir, "route-decisions.csv"), rec, cfg)
        if kind in ("dead reckoning", "sideslip"):
            a = next(r for r in yard.trace if r[i["s"]] >= world.STRAIGHT_1)
            b = next(r for r in yard.trace if r[i["s"]] >= world.STRAIGHT_1 + world.TURN)
            out[f"{label}: over the turn, the estimate advances beyond the true arc length by "
                f"[mm]"] = (f"{1e3 * ((b[i['s_hat']] - a[i['s_hat']]) - (b[i['s']] - a[i['s']])):+.1f}")
            # where the errors arise
            for where, s_mark in (("after the acceleration", 5.0), ("at the start of the turn", 10.0),
                                  ("at its end", world.STRAIGHT_1 + world.TURN), ("at 23 m", 23.0)):
                row = next(r for r in yard.trace if r[i["s"]] >= s_mark)
                out[f"{label}: {where}, at {s_mark:.2f} m: error along, across [m]"] = (
                    f"{row[i['s']] - row[i['s_hat']]:+.4f}, {row[i['d']] - row[i['d_hat']]:+.4f}")
    rows = []
    n = min(len(traces["dead reckoning"]), len(traces["sideslip"]))
    for k in range(0, n, 5):
        a, b = traces["dead reckoning"][k], traces["sideslip"][k]
        rows.append(dict(t=a[0], s_dr=a[i["s"]], es_dr=a[i["s"]] - a[i["s_hat"]],
                         ed_dr=a[i["d"]] - a[i["d_hat"]], s_ss=b[i["s"]],
                         es_ss=b[i["s"]] - b[i["s_hat"]], ed_ss=b[i["d"]] - b[i["d_hat"]]))
    write_csv(os.path.join(out_dir, "route-errors.csv"), rows)
    return out


def write_decisions(path, rec, cfg):
    cols = ("k", "t", "governor", "s_hat", "u_hat", "d_hat", "e_hat", "s_c", "c", "candidates",
            "evaluated", "admitted", "selected", "ambition", "level", "reason")
    rows = records.decision_rows(rec, cfg.control, cfg.robot)
    records.write_csv(path, cols, rows, fmt=lambda x: f"{x:.6g}" if isinstance(x, float) else str(x))


# ----------------------------------------------------------------------------
# The predictors and the shower
# ----------------------------------------------------------------------------
def governed_facts(out_dir):
    out, approach, ends, plans = {}, {}, {}, {}
    i = {k: n for n, k in enumerate(hg.Yard.TRACE)}
    cases = (("fixed", "husky.fixed", None), ("reactive", "husky.reactive", None),
             ("map", "husky.map", None), ("shower, map", "husky.map", shower()),
             ("shower, forecast", "husky.forecast", shower()))
    for name, cap, weather in cases:
        rec, yard, cfg, m = crossing(name, capability=cap, weather=weather)
        plans[name] = decisions(rec)
        a1, ap, a0 = committed(rec)
        out[f"{name}: front in the dock zone at [s]; at rest there from [s]"] = (
            f"{m['in_zone']:.2f}; {m['at_rest_in_zone']:.2f}")
        out[f"{name}: level of the plan in force at the end [m]; clearance of the footprint "
            f"from the dock, least and at the end [m]"] = (
            f"{final_level(rec):.2f}; {m['least_clearance']:.4f}, {m['final_clearance']:.4f}")
        out[f"{name}: rest point commanded, estimated arc length, true arc length at the end "
            f"[m]"] = (f"{rec.periods[-1].decision.plan.s_rest:.4f}, {m['final_s_hat']:.4f}, "
                       f"{m['final_s']:.4f}")
        est_ok, fb_ok = kept(m)
        out[f"{name}: the estimate's largest error along [m], in speed [m/s]; its bounds kept, "
            f"the feedback's kept; alarms; touches"] = (
            f"{m['error_s']:.4f}, {m['error_u']:.3f}; {yes(est_ok)}, {yes(fb_ok)}; "
            f"{len(m['alarms'])}; {fmt_touch(m)}")
        out[f"{name}: the state's largest distance from the commanded motion {COMPONENTS}; "
            f"within the funnel; its largest distance from the funnel in the norm [m]"] = four(
            (m["true_s"], m["true_u"], m["true_d"], m["true_e"])) + \
            f"; {yes(in_funnel(m))}; {m['outside']:.4f}"
        out[f"{name}: the robot's largest speed [m/s]; the largest speed that the plans "
            f"command, and that the feedback commands [m/s]"] = (
            f"{m['speed']:.4f}; {m['commanded']:.4f}, {m['command']:.4f}")
        plan = rec.periods[-1].decision.plan
        out[f"{name}: the plan in force at the end: the dock's reserve [m]; the least clearance "
            f"of the footprint in its funnel [m]"] = (
            f"{plan.reserve():.4f}; {hg.DOCK - plan.s_rest - hg.DS - hg.front(hg.DE):.4f}")
        out[f"{name}: periods committing ambition 1, between 0 and 1, 0"] = f"{a1}, {ap}, {a0}"
        below = [(p.k, p.decision.ambition, p.decision.profile.level) for p in rec.periods
                 if p.decision is not None and p.decision.ambition < 1.0]
        out[f"{name}: periods committing an ambition below 1: period, ambition, level [m]"] = \
            "; ".join(f"{k}, {a:g}, {lev:.2f}" for k, a, lev in below)
        ends[name] = (m["final_s"], m["at_rest_in_zone"])
        if name == "map":
            grass = next(r[0] for r in yard.trace if r[i["s"]] >= world.GRASS_S)
            moving = [p for p in rec.periods if p.initial_set is not None and p.initial_set.c > 0]
            stop_k = moving[-1].k + 1
            s_rest = rec.periods[-1].decision.plan.s_rest
            out["map: the robot's centre reaches the wet grass at [s]; the commanded motion comes "
                "to rest at the end of period [k], at [s]"] = (
                f"{grass:.2f}; {stop_k}, {(stop_k + 1) * hg.T:.1f}")
            out["map: at the end, the true arc length beyond the point of rest and beyond its "
                "estimate [mm]; the robot's largest overshoot of its final arc length [mm]"] = (
                f"{1e3 * (m['final_s'] - s_rest):+.1f}, "
                f"{1e3 * (m['final_s'] - m['final_s_hat']):+.1f}; "
                f"{1e3 * (max(r[i['s']] for r in yard.trace) - m['final_s']):.1f}")
        if name in ("map", "shower, map", "shower, forecast"):
            rest = m["at_rest_in_zone"]
            approach[name] = [(r[0] - rest,) + tuple(r[1:]) for r in yard.trace
                              if rest - 4.0 <= r[0] <= rest + 0.2]
            # the largest deceleration commanded, and the true one, in the last stop
            stop = [r for r in yard.trace if r[0] >= rest - 1.5]
            dec_c = max((a[i["u_c"]] - b[i["u_c"]]) / (b[0] - a[0]) for a, b in zip(stop, stop[1:]))
            dec_u = max((a[i["u"]] - b[i["u"]]) / (b[0] - a[0]) for a, b in zip(stop, stop[1:]))
            out[f"{name}: last stop: largest commanded and true deceleration over 20 ms "
                f"[m/s^2]"] = f"{dec_c:.2f}, {dec_u:.2f}"
            # the error of the estimate as the last stop begins
            last = max(k for k, r in enumerate(yard.trace) if r[i["u_c"]] >= hg.V_MAX - 1e-9)
            start = yard.trace[last]
            out[f"{name}: last stop: the estimate's error along when the commanded speed last "
                f"holds the limit [mm]"] = f"{1e3 * (start[i['s']] - start[i['s_hat']]):+.1f}"
    out["reactive: the decisions of every period are those with husky.fixed"] = str(
        plans["reactive"] == plans["fixed"])
    for name in ("shower, map", "shower, forecast"):
        out[f"{name}: at the end, beyond the dry crossing with the map [mm]; at rest later "
            f"than it by [s]"] = (f"{1e3 * (ends[name][0] - ends['map'][0]):+.1f}; "
                                  f"{ends[name][1] - ends['map'][1]:+.2f}")
    rows = []
    names = list(approach)
    n = min(len(v) for v in approach.values())
    for k in range(n):
        row = {}
        for nm in names:
            r = approach[nm][k]
            key = nm.replace("shower, ", "sh_").replace(" ", "_")
            row.update({f"t_{key}": r[0], f"sc_{key}": r[i["s_c"]], f"s_{key}": r[i["s"]],
                        f"shat_{key}": r[i["s_hat"]], f"c_{key}": r[i["u_c"]],
                        f"u_{key}": r[i["u"]]})
        rows.append(row)
    write_csv(os.path.join(out_dir, "route-approach.csv"), rows)
    return out


# ----------------------------------------------------------------------------
# Stops at the labels
# ----------------------------------------------------------------------------
STOP_PLACES = (("asphalt", 1.0, None, None), ("gravel", 15.0, None, None),
               ("wet grass", 19.0, None, None),
               ("wet grass, braking at 3 m/s^2, 0.87 mu g", 19.0, None, 3.0),
               ("wet grass in the shower, its label", 19.0, "shower", "shower"),
               ("wet grass in the shower, the global capability", 19.0, "shower", hg.B_GLOBAL),
               ("wet grass in the shower, the map's label", 19.0, "shower", None))


def place_on_route(h, s):
    x, y, heading = world.point(s)
    h.place(x, y, heading)


def stop_run(start, speed, weather, label_from, make=None, follower=None):
    """From rest at arc length start: accelerate at A_MAX to 'speed', hold it
    for 1 s, then stop as the recovery of governed.py does, at the label of
    the map (label_from None) or of the shower; the feedback and the
    estimator with sideslip throughout, the shower on from the start.
    label_from may also be a braking [m/s^2], which every period then uses;
    make and follower, functions that return the robot and the feedback
    (sim.Husky and follow.Follower by default; Section 13.3 gives them
    others). Returns the label, the commanded rest point, the true and
    estimated arc length at rest, and the largest errors of the estimate,
    distances of the estimate from the commanded motion and of the state from
    it, as husky.metrics names them, with the largest error of the estimate
    in the norm at the ends of the periods (s5, the quantity of (S5))."""
    h = sim.Husky() if make is None else make()
    place_on_route(h, start)
    h.command = (0.0, 0.0)
    est = estimate.DeadReckoning(h, sideslip=True)
    est.calibrate()
    est.start()
    t_weather = h.step_count * h.m.opt.timestep           # the shower is on from the start
    if weather is not None:
        h.weather = lambda g, t: weather.friction(g, t - t_weather + weather.t_start)
    fb = follow.Follower() if follower is None else follower()
    s_hat0 = est.route()[0]
    motions = []
    s, c = s_hat0, 0.0
    while c < speed - 1e-12:
        m = follow.Motion(s, c, min(speed, c + hg.A_MAX * hg.T), hg.T)
        motions.append(m)
        s, c = m.end()
    for _ in range(int(round(1.0 / hg.T))):
        m = follow.Motion(s, c, c, hg.T)
        motions.append(m)
        s, c = m.end()
    wet = weather is not None and label_from == "shower"
    if isinstance(label_from, float):
        labelling = lambda j, m_: label_from                  # noqa: E731
    else:
        labelling = (lambda j, m_: min(hg.label_of(weather.frictions.get(g.name, g.friction)
                                                   if wet else g.friction)
                                       for g in hg.grounds_on(*hg.region(m_))))
    tail, s_rest = hg.recovery(labelling, 0, s, c)
    label = min(lab for _, lab in tail)
    motions += [m for m, _ in tail] + [follow.Motion(s_rest, 0.0, 0.0, hg.T)] * 5
    per = h.steps_per_update
    dt = per * h.m.opt.timestep
    keys = ("error_s", "error_u", "error_d", "error_e", "track_s", "track_u", "track_d",
            "track_e", "true_s", "true_u", "true_d", "true_e")
    worst = dict.fromkeys(keys + ("speed", "s5"), 0.0)
    for m in motions:
        law = hg.Law(m, "segment", fb)
        for k in range(int(round(hg.T / dt))):
            h.command = law(k * dt, est)
            h.step(per)
            est.update(dt)
            x, y, heading = h.pose()
            st, dd = world.project(x, y)
            ee = hg.heading_error(st, heading)
            u = h.body_velocity()[0]
            se, de, ee_hat = est.route()
            sc, uc = m.arc((k + 1) * dt), m.speed((k + 1) * dt)
            for key, val in zip(keys, (st - se, u - est.speed, dd - de, ee - ee_hat,
                                       se - sc, est.speed - uc, de, ee_hat,
                                       st - sc, u - uc, dd, ee)):
                worst[key] = max(worst[key], abs(val))
            worst["speed"] = max(worst["speed"], u)
            if k == int(round(hg.T / dt)) - 1:              # the end of the period
                worst["s5"] = max(worst["s5"], abs(st - se), abs(u - est.speed) * hg.TAU_U,
                                  abs(dd - de) / hg.D_SCALE, abs(ee - ee_hat) * hg.RHO_E)
    x, y, _ = h.pose()
    st, _ = world.project(x, y)
    return label, s_rest, st, est.route()[0], worst


def stop_facts():
    out = {}
    for name, start, weather, label_from in STOP_PLACES:
        wx = shower() if weather else None
        rows = []
        for v in STOP_SPEEDS:
            label, s_rest, st, se, w = stop_run(start, v, wx, label_from)
            rows.append((v, label, st - s_rest, st - se, w))
        out[f"{name}: label [m/s^2]; beyond the commanded rest point at 0.25, 0.5, 0.75, "
            f"1 m/s [mm]"] = (f"{rows[0][1]:.3g}; " + ", ".join(f"{1e3 * r[2]:+.1f}" for r in rows))
        out[f"{name}: estimate's error at rest at 0.25, 0.5, 0.75, 1 m/s [mm]"] = \
            ", ".join(f"{1e3 * r[3]:+.1f}" for r in rows)
        worst = {k: max(r[4][k] for r in rows) for k in rows[0][4]}
        est_ok, fb_ok = kept(worst)
        out[f"{name}: over the four stops, the estimate's largest error along [mm], in speed "
            f"[mm/s], across [mm], in heading [mrad]; its largest distance from the commanded "
            f"motion, the same; the estimate's bounds kept, the feedback's kept; within the "
            f"funnel"] = (
            ", ".join(f"{1e3 * worst[k]:.1f}" for k in ("error_s", "error_u", "error_d",
                                                         "error_e")) + "; " +
            ", ".join(f"{1e3 * worst[k]:.1f}" for k in ("track_s", "track_u", "track_d",
                                                         "track_e")) +
            f"; {yes(est_ok)}, {yes(fb_ok)}; {yes(in_funnel(worst))}")
        outside = max(0.0, worst["true_s"] - hg.DS, (worst["true_u"] - hg.DU) * hg.TAU_U,
                      (worst["true_d"] - hg.DD) / hg.D_SCALE, (worst["true_e"] - hg.DE) * hg.RHO_E)
        out[f"{name}: over the four stops, the state's largest distance from the commanded "
            f"motion along [mm], in speed [mm/s]; from the funnel in the norm [mm]; the robot's "
            f"largest speed [m/s]"] = (
            f"{1e3 * worst['true_s']:.1f}, {1e3 * worst['true_u']:.1f}; {1e3 * outside:.1f}; "
            f"{worst['speed']:.4f}")
    return out


# ----------------------------------------------------------------------------
# Time steps and seeds
# ----------------------------------------------------------------------------
def numerics_facts():
    out = {}
    for dt in (0.001, 0.0005, 0.00025):
        rec, yard, cfg, m = crossing(f"map at {dt}", dt=dt)
        est_ok, fb_ok = kept(m)
        out[f"map, time step {1000 * dt:g} ms: at rest in the dock zone from [s]; level at the "
            f"end [m]; clearance at the end [m]; estimate's largest error along, across [m]; "
            f"the estimate's bounds kept, the feedback's kept; within the funnel"] = (
            f"{m['at_rest_in_zone']:.2f}; {final_level(rec):.2f}; {m['final_clearance']:.4f}; "
            f"{m['error_s']:.4f}, {m['error_d']:.4f}; {yes(est_ok)}, {yes(fb_ok)}; "
            f"{yes(in_funnel(m))}")
    for kind, label in (("sideslip", "dead reckoning with sideslip"),
                        ("dead reckoning", "dead reckoning")):
        runs = []
        for seed in SEEDS:
            old = estimate.SEED
            estimate.SEED = seed
            try:
                rec, yard, cfg, m = crossing(f"map, seed {seed}", estimator=kind)
            finally:
                estimate.SEED = old
            runs.append((seed, m, yard.est.bias - estimate.GYRO_BIAS))
        ms = [m for _, m, _ in runs]
        out[f"{label}, map, seeds {SEEDS[0]} to {SEEDS[-1]} of the gyro's noise: the estimate's "
            f"largest error {COMPONENTS}; seeds beyond E_S along, E_D across, E_E in heading"] = (
            four(tuple(max(m[k] for m in ms) for k in ("error_s", "error_u", "error_d",
                                                         "error_e"))) + "; " +
            f"{sum(m['error_s'] > hg.E_S for m in ms)}, {sum(m['error_d'] > hg.E_D for m in ms)}, "
            f"{sum(m['error_e'] > hg.E_E for m in ms)}")
        out[f"{label}, map, seeds {SEEDS[0]} to {SEEDS[-1]}: the estimate's largest distance "
            f"from the commanded motion {COMPONENTS}; seeds that keep the feedback's bounds; "
            f"seeds within the funnel"] = (
            four(tuple(max(m[k] for m in ms) for k in ("track_s", "track_u", "track_d",
                                                         "track_e"))) + "; " +
            f"{sum(kept(m)[1] for m in ms)}; {sum(in_funnel(m) for m in ms)}")
        worst = max(runs, key=lambda r: r[1]["error_d"])
        biased = max(runs, key=lambda r: abs(r[2]))
        out[f"{label}, map, seeds {SEEDS[0]} to {SEEDS[-1]}: clearance at the end, least and "
            f"largest [m]; the seed whose calibration left the largest error of the bias, that "
            f"error [rad/s]; the seed with the largest error across, its error [m] and its "
            f"bias's error [rad/s]"] = (
            f"{min(m['final_clearance'] for m in ms):.4f}, "
            f"{max(m['final_clearance'] for m in ms):.4f}; "
            f"{biased[0]}, {biased[2]:+.2e}; "
            f"{worst[0]}, {worst[1]['error_d']:.4f}, {worst[2]:+.2e}")
    return out


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def write_record(path, facts):
    rec = facts["record"]
    period = rec["governor's period [s]; updates of the drive controller per period"].split(";")[0] + " s"
    lines = [
        "Record of the experiment of Section 13.2: the Husky A300 under the governor of "
        "Chapter 10, in the yard of Section 13.1",
        "",
        "model and asset provenance",
        "  the model, the drive controller, the drivers and the declared contacts of "
        "Section 13.1 (record.txt), unchanged",
        "simulator and solver",
        f"  MuJoCo {rec['MuJoCo']}, NumPy and Python {rec['NumPy, Python']}; the settings of "
        "Section 13.1",
        "time steps",
        "  simulator 0.0005 s; drive controller, estimator and feedback at 50 Hz; governor "
        f"every {period}; the crossing with the map repeated at 1 and 0.25 ms",
        "modules",
        f"  allovi core and contract {rec['allovi core, contract']} (../ch10); control "
        "package husky.route-speed, proposal husky.cruise, predictors husky.fixed, "
        "husky.reactive, husky.map and husky.forecast, robot husky.robot, world husky.yard, "
        "evaluator husky.metrics (husky/governed.py)",
        "actuators and sensors",
        "  the drive controller's velocity command, from the feedback of husky/follow.py, gains " +
        rec["feedback gains: K_S [1/s], K_D [1/m^2], K_E [1/m], W_MIN [m/s], K_I [1/s]"],
        "  gyro (ours): bias, noise, seed; calibration " +
        rec["gyro: bias [rad/s], noise per reading at 50 Hz [rad/s], seed; calibration at rest [s]"],
        "  encoders exact; the estimator with sideslip of husky/estimate.py",
        "frames and units",
        "  SI units; route coordinates s and d, positive to the left; the state (s, u, d, e), "
        "norm max(|s|, |u| * 0.5 s, |d| / 2, |e| * 2 m)",
        "uncertainty bounds",
        "  grounds and labels: " + rec["grounds: friction of asphalt, gravel, wet grass; their labels [m/s^2]; the share of mu g; the global capability [m/s^2]"],
        "  the shower: " + rec["the shower: from [s], announced at [s]; friction of the wet grass; its label [m/s^2]"],
        "  the measurement-error bound VARPI (ours): " +
        rec["the state (s, u, d, e): the norm's scales of the speed [s] and the heading error [m]; the measurement-error bound VARPI [m]"].split("; ")[1] + " m",
        "  the estimate's bounds (ours): " + rec[f"bounds of the estimate E_S, E_U, E_D, E_E: {COMPONENTS}"],
        "  the feedback's bounds (ours, measured): " + rec[f"bounds of the feedback F_S, F_U, F_D, F_E: {COMPONENTS}"],
        "  the funnel: " + rec[f"the funnel DS, DU, DD, DE: {COMPONENTS}"],
        "constraints",
        "  the dock and the lane of Section 13.1, two families with a reserve each; the lane "
        "keeps the robot 0.1 m from the pallets; reserve floor of the dock 0.1 m; the lane's "
        "reserve, the same for every plan, " +
        rec["lane: its reserve in the norm [m], the constant; the offset and heading error it covers [m], [rad]; the allowance there [m]"].split(",")[0] + " m; the commanded speed at most 1 m/s, the task's limit, which no reserve keeps; the robot's speed, by the funnel's bound, at most 1.25 m/s",
        "terminal behaviour",
        "  the stop at the label of each period's region, then rest; after an alarm, the stop "
        "at the global capability, and a restart from rest",
        "training and evaluation split",
        "  none: nothing learns in Section 13.2",
        "seeds",
        f"  the gyro's noise, PCG64, seed {estimate.SEED}; seeds {SEEDS[0]} to {SEEDS[-1]} in a "
        "repetition",
        "hardware",
        f"  {platform.system()} {platform.machine()}, {h131.hardware()}",
        "guarantee status",
        "  conditional: the test is sound for the declared bounds, the labels and VARPI; the "
        "bounds are measured in simulation, not proved",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Section 13.2: the Husky under the governor")
    ap.add_argument("--check", action="store_true", help="check every value quoted in Section 13.2")
    args = ap.parse_args()
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    facts = {"record": record_facts(),
             "calibration": calibration_facts(),
             "the estimators": estimator_facts(out_dir),
             "the predictors and the shower": governed_facts(out_dir),
             "stops at the labels": stop_facts(),
             "time steps and seeds": numerics_facts()}
    write_record(os.path.join(out_dir, "route-record.txt"), facts)
    lines, failures = [], []
    for group, items in facts.items():
        lines.append(f"[{group}]")
        for key, val in items.items():
            lines.append(f"  {key} = {val}")
            want = EXPECTED.get(group, {}).get(key)
            if args.check and want is not None and want != str(val):
                failures.append(f"{group}.{key}: expected {want!r}, got {val!r}")
    if args.check:
        failures += [f"{group}.{key}: not computed" for group, items in EXPECTED.items()
                     for key in items if key not in facts.get(group, {})]
        n = sum(len(v) for v in EXPECTED.values())
        lines.append(f"[check] {n} values, among them every value that Section 13.2 quotes: " +
                     ("all as expected" if not failures else "FAILED: " + "; ".join(failures)))
    with open(os.path.join(out_dir, "route-summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if failures:
        raise SystemExit(1)


EXPECTED = {   # every value of the summary but the versions of NumPy and Python
    'record': {
        'MuJoCo':
            '3.16.0',
        # the shipped results were written with contract 1.0; the core is now
        # at 1.1, whose two optional additions this script does not use
        'allovi core, contract':
            '0.1.0, 1.1',
        "governor's period [s]; updates of the drive controller per period":
            '0.2; 10',
        'the drive controller: wheel radius, separation [m]; limits of acceleration, deceleration [m/s^2]':
            '0.1625, 0.9835; 2, 4',
        'plans: speed limit [m/s], acceleration [m/s^2], deceleration at most [m/s^2]':
            '1, 1, 4',
        "the state (s, u, d, e): the norm's scales of the speed [s] and the heading error [m]; the measurement-error bound VARPI [m]":
            '0.5, 2; 0.05',
        'bounds of the estimate E_S, E_U, E_D, E_E: along [m], in speed [m/s], across [m], in heading [rad]':
            '0.05, 0.1, 0.1, 0.025',
        'bounds of the feedback F_S, F_U, F_D, F_E: along [m], in speed [m/s], across [m], in heading [rad]':
            '0.02, 0.15, 0.06, 0.04',
        'the funnel DS, DU, DD, DE: along [m], in speed [m/s], across [m], in heading [rad]':
            '0.07, 0.25, 0.16, 0.065',
        "the wheels: half the wheelbase, half the track [m]; a label's reach beyond the commanded arc lengths REACH [m]":
            '0.256, 0.2829; 0.344',
        'levels: floor, step [m]; the dock zone [m]; ambitions':
            '0.1, 0.01; 0.4; 1, 0.75, 0.5, 0.25',
        "the dock: the footprint's reach ahead of the centre at the heading error DE [m]; a plan's room beyond its point of rest for the floor's reserve [m]":
            '0.5166; 0.7018',
        "the dock: the heading error up to which the footprint's reach grows, atan(w / l) [rad]; the reach there, half the footprint's diagonal [m]; the largest reserve whose heading error DE + eps / RHO_E stays within it [m]":
            '0.6141; 0.6057; 1.0982',
        "the speed: the plans' commanded speed at most [m/s]; the robot's speed by the funnel's bound at most [m/s]; the commanded speed that would keep the robot's speed within the limit by that bound [m/s]":
            '1; 1.25; 0.75',
        'lane: the narrowest allowance of the centre at no heading error and at DE [m]; the constant [m]':
            '0.318, 0.289; 0.289',
        'lane: its reserve in the norm [m], the constant; the offset and heading error it covers [m], [rad]; the allowance there [m]':
            '0.0586, 0.0586; 0.2772, 0.0943; 0.2772',
        'lane: the constants agree with the computation, to 3 digits':
            'True',
        'grounds: friction of asphalt, gravel, wet grass; their labels [m/s^2]; the share of mu g; the global capability [m/s^2]':
            '0.8, 0.6, 0.35; 4, 3.53, 2.06; 0.6; 0.8',
        'the shower: from [s], announced at [s]; friction of the wet grass; its label [m/s^2]':
            '20, 10; 0.15; 0.883',
        'feedback gains: K_S [1/s], K_D [1/m^2], K_E [1/m], W_MIN [m/s], K_I [1/s]':
            '2, 1.5, 3, 0.2, 4',
        'gyro: bias [rad/s], noise per reading at 50 Hz [rad/s], seed; calibration at rest [s]':
            '0.004, 0.001, 13; 2',
        'stops: from rest at [m] on asphalt, gravel, wet grass':
            '1, 15, 19',
    },
    'calibration': {
        "rolling radius on asphalt over 10 s at the command 1 m/s [m]; the estimator's":
            '0.16499; 0.16499',
        "asphalt: turn of radius 3 m at 1 m/s: speed [m/s]; the wheels' mean rim speed [m/s], short of it by [%]; counted at the rolling radius [m/s], short by [%]":
            '1.0189; 1.0160, 0.28; 1.0153, 0.35',
        "asphalt: turn of radius 3 m at 1 m/s: sideways speed [m/s], yaw rate [rad/s], short of the command 1/3 rad/s by [%]; centre of rotation ahead of the centre [m]; K [s^2], the estimator's":
            '-0.0154, 0.3166, 5.0; 0.0487; 0.151, 0.151',
        'asphalt: the same turn to the right: sideways speed [m/s], yaw rate [rad/s]; centre of rotation ahead of the centre [m]':
            '+0.0154, -0.3166; 0.0487',
        "gravel: turn of radius 3 m at 1 m/s: speed [m/s]; the wheels' mean rim speed [m/s], short of it by [%]; counted at the rolling radius [m/s], short by [%]":
            '1.0188; 1.0160, 0.27; 1.0153, 0.34',
        "gravel: turn of radius 3 m at 1 m/s: sideways speed [m/s], yaw rate [rad/s], short of the command 1/3 rad/s by [%]; centre of rotation ahead of the centre [m]; K [s^2], the estimator's":
            '-0.0188, 0.3175, 4.7; 0.0593; 0.183, 0.183',
        'gravel: the same turn to the right: sideways speed [m/s], yaw rate [rad/s]; centre of rotation ahead of the centre [m]':
            '+0.0188, -0.3175; 0.0593',
        "wet grass: turn of radius 3 m at 1 m/s: speed [m/s]; the wheels' mean rim speed [m/s], short of it by [%]; counted at the rolling radius [m/s], short by [%]":
            '1.0184; 1.0160, 0.24; 1.0153, 0.31',
        "wet grass: turn of radius 3 m at 1 m/s: sideways speed [m/s], yaw rate [rad/s], short of the command 1/3 rad/s by [%]; centre of rotation ahead of the centre [m]; K [s^2], the estimator's":
            '-0.0290, 0.3217, 3.5; 0.0901; 0.275, 0.275',
        'wet grass: the same turn to the right: sideways speed [m/s], yaw rate [rad/s]; centre of rotation ahead of the centre [m]':
            '+0.0290, -0.3217; 0.0901',
        'sideslip constants of the estimator agree with the turns, to 3 digits':
            'True',
        'gyro: bias estimated over 2 s at rest [rad/s]; its error [rad/s]':
            '0.00410; +9.5e-05',
    },
    'the estimators': {
        "the A300's odometry: the estimate's largest error along [m], in speed [m/s], across [m], in heading [rad]":
            '5.6989, 1.124, 0.7309, 0.6035',
        "the A300's odometry: the estimate's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0176, 0.161, 0.0074, 0.0285',
        "the A300's odometry: the state's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]; within the funnel":
            '5.6839, 1.126, 0.7305, 0.6034; no',
        "the A300's odometry: the footprint's largest offset from the centre line [m]; touches; alarms, restarts":
            '1.119; pallets on the second straight at 19.26 s, s = 18.51 m; 0, 0',
        "the A300's odometry: at the end: arc length, its estimate [m]; clearance of the footprint from the dock [m]":
            '18.6071, 24.3061; 5.7873',
        "dead reckoning with sideslip at the controller's radius: the estimate's largest error along [m], in speed [m/s], across [m], in heading [rad]":
            '0.2061, 0.031, 0.2221, 0.0578',
        "dead reckoning with sideslip at the controller's radius: the estimate's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0135, 0.057, 0.0361, 0.0342',
        "dead reckoning with sideslip at the controller's radius: the state's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]; within the funnel":
            '0.2064, 0.049, 0.2576, 0.0764; no',
        "dead reckoning with sideslip at the controller's radius: the footprint's largest offset from the centre line [m]; touches; alarms, restarts":
            '0.641; nothing; 0, 0',
        "dead reckoning with sideslip at the controller's radius: at the end: arc length, its estimate [m]; clearance of the footprint from the dock [m]":
            '24.4944, 24.2910; 0.0092',
        "dead reckoning with sideslip, the gyro not calibrated: the estimate's largest error along [m], in speed [m/s], across [m], in heading [rad]":
            '0.9424, 0.707, 0.7284, 0.1396',
        "dead reckoning with sideslip, the gyro not calibrated: the estimate's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0340, 0.252, 0.0393, 0.2609',
        "dead reckoning with sideslip, the gyro not calibrated: the state's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]; within the funnel":
            '0.9399, 0.899, 0.7337, 0.3422; no',
        "dead reckoning with sideslip, the gyro not calibrated: the footprint's largest offset from the centre line [m]; touches; alarms, restarts":
            '1.148; pallets on the second straight at 19.70 s, s = 18.61 m; 1, the first at 19.8 s, 1',
        'dead reckoning with sideslip, the gyro not calibrated: the first alarm: the estimate beyond the initial set along [m], in speed [m/s], across [m], in heading [rad]; in the norm [m]':
            '-0.0543, -0.170, -0.1328, 0.0699; 0.1399',
        'dead reckoning with sideslip, the gyro not calibrated: at the end: arc length, its estimate [m]; clearance of the footprint from the dock [m]':
            '23.3273, 24.2697; 1.1340',
        "dead reckoning: the estimate's largest error along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0470, 0.029, 0.0833, 0.0122',
        "dead reckoning: the estimate's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0074, 0.056, 0.0041, 0.0167',
        "dead reckoning: the state's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]; within the funnel":
            '0.0440, 0.042, 0.0828, 0.0208; yes',
        "dead reckoning: the footprint's largest offset from the centre line [m]; touches; alarms, restarts":
            '0.472; nothing; 0, 0',
        'dead reckoning: at the end: arc length, its estimate [m]; clearance of the footprint from the dock [m]':
            '24.2470, 24.2939; 0.2567',
        'dead reckoning: over the turn, the estimate advances beyond the true arc length by [mm]':
            '+49.0',
        'dead reckoning: after the acceleration, at 5.00 m: error along, across [m]':
            '+0.0084, +0.0014',
        'dead reckoning: at the start of the turn, at 10.00 m: error along, across [m]':
            '+0.0083, +0.0049',
        'dead reckoning: at its end, at 14.71 m: error along, across [m]':
            '-0.0407, -0.0831',
        'dead reckoning: at 23 m, at 23.00 m: error along, across [m]':
            '-0.0409, -0.0650',
        "dead reckoning with sideslip: the estimate's largest error along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0233, 0.029, 0.0216, 0.0090',
        "dead reckoning with sideslip: the estimate's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0074, 0.056, 0.0391, 0.0298',
        "dead reckoning with sideslip: the state's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]; within the funnel":
            '0.0251, 0.042, 0.0600, 0.0310; yes',
        "dead reckoning with sideslip: the footprint's largest offset from the centre line [m]; touches; alarms, restarts":
            '0.450; nothing; 0, 0',
        'dead reckoning with sideslip: at the end: arc length, its estimate [m]; clearance of the footprint from the dock [m]':
            '24.3095, 24.2939; 0.1941',
        'dead reckoning with sideslip: over the turn, the estimate advances beyond the true arc length by [mm]':
            '-13.3',
        'dead reckoning with sideslip: after the acceleration, at 5.00 m: error along, across [m]':
            '+0.0084, +0.0014',
        'dead reckoning with sideslip: at the start of the turn, at 10.00 m: error along, across [m]':
            '+0.0083, +0.0049',
        'dead reckoning with sideslip: at its end, at 14.71 m: error along, across [m]':
            '+0.0216, -0.0208',
        'dead reckoning with sideslip: at 23 m, at 23.00 m: error along, across [m]':
            '+0.0217, -0.0031',
    },
    'the predictors and the shower': {
        'fixed: front in the dock zone at [s]; at rest there from [s]':
            '24.92; 25.82',
        'fixed: level of the plan in force at the end [m]; clearance of the footprint from the dock, least and at the end [m]':
            '0.12; 0.2149, 0.2160',
        'fixed: rest point commanded, estimated arc length, true arc length at the end [m]':
            '24.2720, 24.2744, 24.2877',
        "fixed: the estimate's largest error along [m], in speed [m/s]; its bounds kept, the feedback's kept; alarms; touches":
            '0.0233, 0.019; yes, yes; 0; nothing',
        "fixed: the state's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]; within the funnel; its largest distance from the funnel in the norm [m]":
            '0.0226, 0.027, 0.0600, 0.0310; yes; 0.0000',
        "fixed: the robot's largest speed [m/s]; the largest speed that the plans command, and that the feedback commands [m/s]":
            '1.0265; 1.0000, 1.0125',
        "fixed: the plan in force at the end: the dock's reserve [m]; the least clearance of the footprint in its funnel [m]":
            '0.1229; 0.1414',
        'fixed: periods committing ambition 1, between 0 and 1, 0':
            '165, 1, 4',
        'fixed: periods committing an ambition below 1: period, ambition, level [m]':
            '120, 0.75, 0.12; 121, 0, 0.12; 122, 0, 0.12; 123, 0, 0.12; 124, 0, 0.12',
        'reactive: front in the dock zone at [s]; at rest there from [s]':
            '24.92; 25.82',
        'reactive: level of the plan in force at the end [m]; clearance of the footprint from the dock, least and at the end [m]':
            '0.12; 0.2149, 0.2160',
        'reactive: rest point commanded, estimated arc length, true arc length at the end [m]':
            '24.2720, 24.2744, 24.2877',
        "reactive: the estimate's largest error along [m], in speed [m/s]; its bounds kept, the feedback's kept; alarms; touches":
            '0.0233, 0.019; yes, yes; 0; nothing',
        "reactive: the state's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]; within the funnel; its largest distance from the funnel in the norm [m]":
            '0.0226, 0.027, 0.0600, 0.0310; yes; 0.0000',
        "reactive: the robot's largest speed [m/s]; the largest speed that the plans command, and that the feedback commands [m/s]":
            '1.0265; 1.0000, 1.0125',
        "reactive: the plan in force at the end: the dock's reserve [m]; the least clearance of the footprint in its funnel [m]":
            '0.1229; 0.1414',
        'reactive: periods committing ambition 1, between 0 and 1, 0':
            '165, 1, 4',
        'reactive: periods committing an ambition below 1: period, ambition, level [m]':
            '120, 0.75, 0.12; 121, 0, 0.12; 122, 0, 0.12; 123, 0, 0.12; 124, 0, 0.12',
        'map: front in the dock zone at [s]; at rest there from [s]':
            '24.80; 25.58',
        'map: level of the plan in force at the end [m]; clearance of the footprint from the dock, least and at the end [m]':
            '0.10; 0.1935, 0.1941',
        'map: rest point commanded, estimated arc length, true arc length at the end [m]':
            '24.2910, 24.2939, 24.3095',
        "map: the estimate's largest error along [m], in speed [m/s]; its bounds kept, the feedback's kept; alarms; touches":
            '0.0233, 0.029; yes, yes; 0; nothing',
        "map: the state's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]; within the funnel; its largest distance from the funnel in the norm [m]":
            '0.0251, 0.042, 0.0600, 0.0310; yes; 0.0000',
        "map: the robot's largest speed [m/s]; the largest speed that the plans command, and that the feedback commands [m/s]":
            '1.0265; 1.0000, 1.0125',
        "map: the plan in force at the end: the dock's reserve [m]; the least clearance of the footprint in its funnel [m]":
            '0.1063; 0.1224',
        'map: periods committing ambition 1, between 0 and 1, 0':
            '167, 1, 2',
        'map: periods committing an ambition below 1: period, ambition, level [m]':
            '122, 0.75, 0.10; 123, 0, 0.10; 124, 0, 0.10',
        "map: the robot's centre reaches the wet grass at [s]; the commanded motion comes to rest at the end of period [k], at [s]":
            '18.68; 126, 25.4',
        "map: at the end, the true arc length beyond the point of rest and beyond its estimate [mm]; the robot's largest overshoot of its final arc length [mm]":
            '+18.6, +15.7; 0.9',
        'map: last stop: largest commanded and true deceleration over 20 ms [m/s^2]':
            '2.06, 2.31',
        "map: last stop: the estimate's error along when the commanded speed last holds the limit [mm]":
            '+21.7',
        'shower, map: front in the dock zone at [s]; at rest there from [s]':
            '24.80; 25.70',
        'shower, map: level of the plan in force at the end [m]; clearance of the footprint from the dock, least and at the end [m]':
            '0.10; 0.1051, 0.1067',
        'shower, map: rest point commanded, estimated arc length, true arc length at the end [m]':
            '24.2910, 24.2941, 24.3970',
        "shower, map: the estimate's largest error along [m], in speed [m/s]; its bounds kept, the feedback's kept; alarms; touches":
            '0.1029, 0.243; no, yes; 0; nothing',
        "shower, map: the state's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]; within the funnel; its largest distance from the funnel in the norm [m]":
            '0.1078, 0.258, 0.0600, 0.0310; no; 0.0378',
        "shower, map: the robot's largest speed [m/s]; the largest speed that the plans command, and that the feedback commands [m/s]":
            '1.0265; 1.0000, 1.0125',
        "shower, map: the plan in force at the end: the dock's reserve [m]; the least clearance of the footprint in its funnel [m]":
            '0.1063; 0.1224',
        'shower, map: periods committing ambition 1, between 0 and 1, 0':
            '167, 1, 2',
        'shower, map: periods committing an ambition below 1: period, ambition, level [m]':
            '122, 0.75, 0.10; 123, 0, 0.10; 124, 0, 0.10',
        'shower, map: last stop: largest commanded and true deceleration over 20 ms [m/s^2]':
            '2.06, 1.47',
        "shower, map: last stop: the estimate's error along when the commanded speed last holds the limit [mm]":
            '+21.7',
        'shower, forecast: front in the dock zone at [s]; at rest there from [s]':
            '24.90; 25.80',
        'shower, forecast: level of the plan in force at the end [m]; clearance of the footprint from the dock, least and at the end [m]':
            '0.12; 0.2151, 0.2158',
        'shower, forecast: rest point commanded, estimated arc length, true arc length at the end [m]':
            '24.2703, 24.2725, 24.2879',
        "shower, forecast: the estimate's largest error along [m], in speed [m/s]; its bounds kept, the feedback's kept; alarms; touches":
            '0.0233, 0.019; yes, yes; 0; nothing',
        "shower, forecast: the state's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]; within the funnel; its largest distance from the funnel in the norm [m]":
            '0.0234, 0.027, 0.0600, 0.0310; yes; 0.0000',
        "shower, forecast: the robot's largest speed [m/s]; the largest speed that the plans command, and that the feedback commands [m/s]":
            '1.0265; 1.0000, 1.0125',
        "shower, forecast: the plan in force at the end: the dock's reserve [m]; the least clearance of the footprint in its funnel [m]":
            '0.1245; 0.1431',
        'shower, forecast: periods committing ambition 1, between 0 and 1, 0':
            '166, 0, 4',
        'shower, forecast: periods committing an ambition below 1: period, ambition, level [m]':
            '121, 0, 0.12; 122, 0, 0.12; 123, 0, 0.12; 124, 0, 0.12',
        'shower, forecast: last stop: largest commanded and true deceleration over 20 ms [m/s^2]':
            '0.88, 1.03',
        "shower, forecast: last stop: the estimate's error along when the commanded speed last holds the limit [mm]":
            '+21.7',
        'reactive: the decisions of every period are those with husky.fixed':
            'True',
        'shower, map: at the end, beyond the dry crossing with the map [mm]; at rest later than it by [s]':
            '+87.5; +0.12',
        'shower, forecast: at the end, beyond the dry crossing with the map [mm]; at rest later than it by [s]':
            '-21.7; +0.22',
    },
    'stops at the labels': {
        'asphalt: label [m/s^2]; beyond the commanded rest point at 0.25, 0.5, 0.75, 1 m/s [mm]':
            '4; +1.6, +3.2, +10.5, +13.5',
        "asphalt: estimate's error at rest at 0.25, 0.5, 0.75, 1 m/s [mm]":
            '-0.0, +0.0, +3.8, +5.5',
        "asphalt: over the four stops, the estimate's largest error along [mm], in speed [mm/s], across [mm], in heading [mrad]; its largest distance from the commanded motion, the same; the estimate's bounds kept, the feedback's kept; within the funnel":
            '8.6, 49.0, 0.0, 0.3; 14.0, 125.1, 0.0, 0.2; yes, yes; yes',
        "asphalt: over the four stops, the state's largest distance from the commanded motion along [mm], in speed [mm/s]; from the funnel in the norm [mm]; the robot's largest speed [m/s]":
            '21.3, 88.7; 0.0; 1.0265',
        'gravel: label [m/s^2]; beyond the commanded rest point at 0.25, 0.5, 0.75, 1 m/s [mm]':
            '3.53; +1.6, +3.7, +8.6, +10.0',
        "gravel: estimate's error at rest at 0.25, 0.5, 0.75, 1 m/s [mm]":
            '-0.0, +0.5, +4.9, +4.9',
        "gravel: over the four stops, the estimate's largest error along [mm], in speed [mm/s], across [mm], in heading [mrad]; its largest distance from the commanded motion, the same; the estimate's bounds kept, the feedback's kept; within the funnel":
            '8.6, 42.5, 0.0, 0.3; 10.0, 109.6, 0.0, 0.2; yes, yes; yes',
        "gravel: over the four stops, the state's largest distance from the commanded motion along [mm], in speed [mm/s]; from the funnel in the norm [mm]; the robot's largest speed [m/s]":
            '17.4, 79.4; 0.0; 1.0265',
        'wet grass: label [m/s^2]; beyond the commanded rest point at 0.25, 0.5, 0.75, 1 m/s [mm]':
            '2.06; +1.6, +3.7, +5.3, +5.9',
        "wet grass: estimate's error at rest at 0.25, 0.5, 0.75, 1 m/s [mm]":
            '-0.0, +1.8, +1.8, +2.6',
        "wet grass: over the four stops, the estimate's largest error along [mm], in speed [mm/s], across [mm], in heading [mrad]; its largest distance from the commanded motion, the same; the estimate's bounds kept, the feedback's kept; within the funnel":
            '8.6, 29.5, 0.0, 0.4; 7.0, 65.2, 0.0, 0.2; yes, yes; yes',
        "wet grass: over the four stops, the state's largest distance from the commanded motion along [mm], in speed [mm/s]; from the funnel in the norm [mm]; the robot's largest speed [m/s]":
            '12.3, 44.8; 0.0; 1.0265',
        'wet grass, braking at 3 m/s^2, 0.87 mu g: label [m/s^2]; beyond the commanded rest point at 0.25, 0.5, 0.75, 1 m/s [mm]':
            '3; +1.6, +8.3, +13.9, +17.3',
        "wet grass, braking at 3 m/s^2, 0.87 mu g: estimate's error at rest at 0.25, 0.5, 0.75, 1 m/s [mm]":
            '-0.0, +5.1, +11.0, +12.8',
        "wet grass, braking at 3 m/s^2, 0.87 mu g: over the four stops, the estimate's largest error along [mm], in speed [mm/s], across [mm], in heading [mrad]; its largest distance from the commanded motion, the same; the estimate's bounds kept, the feedback's kept; within the funnel":
            '14.6, 29.7, 0.0, 0.3; 8.0, 90.6, 0.0, 0.2; yes, yes; yes',
        "wet grass, braking at 3 m/s^2, 0.87 mu g: over the four stops, the state's largest distance from the commanded motion along [mm], in speed [mm/s]; from the funnel in the norm [mm]; the robot's largest speed [m/s]":
            '22.6, 75.5; 0.0; 1.0265',
        'wet grass in the shower, its label: label [m/s^2]; beyond the commanded rest point at 0.25, 0.5, 0.75, 1 m/s [mm]':
            '0.883; +0.1, +0.0, -0.9, -2.2',
        "wet grass in the shower, its label: estimate's error at rest at 0.25, 0.5, 0.75, 1 m/s [mm]":
            '-0.9, -1.9, -2.5, -4.5',
        "wet grass in the shower, its label: over the four stops, the estimate's largest error along [mm], in speed [mm/s], across [mm], in heading [mrad]; its largest distance from the commanded motion, the same; the estimate's bounds kept, the feedback's kept; within the funnel":
            '4.5, 13.0, 0.1, 0.5; 4.4, 31.4, 0.1, 0.2; yes, yes; yes',
        "wet grass in the shower, its label: over the four stops, the state's largest distance from the commanded motion along [mm], in speed [mm/s]; from the funnel in the norm [mm]; the robot's largest speed [m/s]":
            '3.9, 24.1; 0.0; 1.0241',
        'wet grass in the shower, the global capability: label [m/s^2]; beyond the commanded rest point at 0.25, 0.5, 0.75, 1 m/s [mm]':
            '0.8; -0.3, -1.4, -1.7, -4.1',
        "wet grass in the shower, the global capability: estimate's error at rest at 0.25, 0.5, 0.75, 1 m/s [mm]":
            '-1.4, -2.6, -3.7, -5.8',
        "wet grass in the shower, the global capability: over the four stops, the estimate's largest error along [mm], in speed [mm/s], across [mm], in heading [mrad]; its largest distance from the commanded motion, the same; the estimate's bounds kept, the feedback's kept; within the funnel":
            '5.8, 13.4, 0.1, 0.5; 4.4, 31.4, 0.0, 0.2; yes, yes; yes',
        "wet grass in the shower, the global capability: over the four stops, the state's largest distance from the commanded motion along [mm], in speed [mm/s]; from the funnel in the norm [mm]; the robot's largest speed [m/s]":
            '4.1, 24.1; 0.0; 1.0241',
        "wet grass in the shower, the map's label: label [m/s^2]; beyond the commanded rest point at 0.25, 0.5, 0.75, 1 m/s [mm]":
            '2.06; +4.0, +27.9, +65.5, +109.5',
        "wet grass in the shower, the map's label: estimate's error at rest at 0.25, 0.5, 0.75, 1 m/s [mm]":
            '+2.4, +26.0, +61.6, +105.9',
        "wet grass in the shower, the map's label: over the four stops, the estimate's largest error along [mm], in speed [mm/s], across [mm], in heading [mrad]; its largest distance from the commanded motion, the same; the estimate's bounds kept, the feedback's kept; within the funnel":
            '105.9, 252.8, 0.0, 0.4; 6.6, 58.3, 0.0, 0.2; no, yes; no',
        "wet grass in the shower, the map's label: over the four stops, the state's largest distance from the commanded motion along [mm], in speed [mm/s]; from the funnel in the norm [mm]; the robot's largest speed [m/s]":
            '111.3, 268.5; 41.3; 1.0241',
    },
    'time steps and seeds': {
        "map, time step 1 ms: at rest in the dock zone from [s]; level at the end [m]; clearance at the end [m]; estimate's largest error along, across [m]; the estimate's bounds kept, the feedback's kept; within the funnel":
            '25.52; 0.10; 0.2020; 0.0286, 0.0299; yes, yes; yes',
        "map, time step 0.5 ms: at rest in the dock zone from [s]; level at the end [m]; clearance at the end [m]; estimate's largest error along, across [m]; the estimate's bounds kept, the feedback's kept; within the funnel":
            '25.58; 0.10; 0.1941; 0.0233, 0.0216; yes, yes; yes',
        "map, time step 0.25 ms: at rest in the dock zone from [s]; level at the end [m]; clearance at the end [m]; estimate's largest error along, across [m]; the estimate's bounds kept, the feedback's kept; within the funnel":
            '25.60; 0.10; 0.1900; 0.0216, 0.0174; yes, yes; yes',
        "dead reckoning with sideslip, map, seeds 1 to 20 of the gyro's noise: the estimate's largest error along [m], in speed [m/s], across [m], in heading [rad]; seeds beyond E_S along, E_D across, E_E in heading":
            '0.0309, 0.029, 0.0823, 0.0102; 0, 0, 0',
        "dead reckoning with sideslip, map, seeds 1 to 20: the estimate's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]; seeds that keep the feedback's bounds; seeds within the funnel":
            '0.0074, 0.056, 0.0393, 0.0301; 20; 20',
        "dead reckoning with sideslip, map, seeds 1 to 20: clearance at the end, least and largest [m]; the seed whose calibration left the largest error of the bias, that error [rad/s]; the seed with the largest error across, its error [m] and its bias's error [rad/s]":
            '0.1850, 0.2238; 5, -2.24e-04; 5, 0.0823, -2.24e-04',
        "dead reckoning, map, seeds 1 to 20 of the gyro's noise: the estimate's largest error along [m], in speed [m/s], across [m], in heading [rad]; seeds beyond E_S along, E_D across, E_E in heading":
            '0.0787, 0.029, 0.1441, 0.0175; 18, 6, 0',
        "dead reckoning, map, seeds 1 to 20: the estimate's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]; seeds that keep the feedback's bounds; seeds within the funnel":
            '0.0074, 0.056, 0.0050, 0.0236; 20; 19',
        "dead reckoning, map, seeds 1 to 20: clearance at the end, least and largest [m]; the seed whose calibration left the largest error of the bias, that error [rad/s]; the seed with the largest error across, its error [m] and its bias's error [rad/s]":
            '0.2482, 0.2867; 5, -2.24e-04; 5, 0.1441, -2.24e-04',
    },
}


if __name__ == "__main__":
    main()
