#!/usr/bin/env python3
"""
The Go2 under the governor: a lift of its front-right foot (FR) on three feet.

Physical AI, Together, Chapter 15 (Section 15.2).

This script
  * writes the record of the experiment;
  * states what the test of quad/governed.py derives from the reduced model
    of quad/reduced.py: the home stance's distances inside the triangle of
    FL, RL and RR and inside the rectangle of the four feet, the levels of
    the rest at home and at the place that the lift's requirement needs, the
    lift's consumption, and the joints' static torques;
  * runs the simulated Go2 (quad/sim.py) under the allovi core with the
    modules of quad/governed.py: a reference configuration (the global
    capability, a push of at most 10 N, everywhere; the lift announced in
    period 2 for period 15; the operator's proposal: home), and
    configurations that change one thing each: pushes of 10 N on the trunk,
    the label's bound, towards FR, in the shift, the hold and the
    touchdown; the lift not announced; a shove of 20 N in the hold, beyond
    the label; the floor under FR 30 mm higher than the package declares
    (an obstacle) and 30 mm lower (a hole), and 10 mm higher and lower, the
    edges of the touchdown's assumed band; and the operator's proposal
    executed without the governor, with the pushes of the second
    configuration;
  * for each run: when the operation begins and whether it is overloaded,
    the levels, the simulated centre of pressure's least distance inside the
    triangle while FR may carry less than its share and inside the rectangle
    otherwise, and above the level in force, the centre of mass's and FR's
    deviations from the funnel's centre, the joints' torques, FR's
    touchdown, the governor's interventions, the alarms and the restarts,
    and whether any part other than the feet touches the floor;
  * runs the checks of Section 10.3 (allovi.checks) on the reference and on
    the shove;
  * sweeps pushes of 10 N in eight directions over nine phases of the
    reference run, and compares the funnel's bounds and the allowances with
    what they measure;
  * executes every plan in force of the reference and of the pushed run
    unchanged, from the simulated state at its period's start;
  * touches FR down from the hold at speeds from 0.05 to 0.4 m/s;
  * shuts the robot down: from home it crouches, holds, cuts the motors'
    torques, and lies on the floor.

Every distance, level and margin is rounded down, and every deviation,
shortfall, share and speed rounded up, at the digits printed.

Usage
    python3 quad_governed.py            # writes results/governed-*
    python3 quad_governed.py --check    # also checks every value of its
                                        # summary

Outputs, in results/ next to this file
    governed-record.txt     the record of the experiment
    governed-summary.txt    every value, by group; with --check, the
                            comparison with the values expected
    governed-trace-*.csv    for the reference, the pushed run, the shove and
                            the ungoverned run, every 10 ms: the time (t,
                            s), the centre of mass's place on the line,
                            commanded and measured (s_c, s_m, m), FR's
                            height and its target (h, h_t, m), the simulated
                            centre of pressure's distance inside the
                            triangle and inside the rectangle (d_tri,
                            d_rect, m), FR's contact force (f_fr, N), the
                            push (push, N) and the level in force (level, m)
                            (Figure 15.4)

Requires MuJoCo 3.16.0, NumPy and SciPy (requirements.txt); run with CPython
3.13.17, other versions of Python not tried. About twelve minutes on two
cores of the machine of the record.
"""
import argparse
import copy
import csv
import math
import os
import platform
import sys
import time
from multiprocessing import Pool

import mujoco
import numpy as np
import scipy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "ch10"))
import allovi                                                    # noqa: E402
from allovi import checks                                        # noqa: E402

from quad import governed as gv                                  # noqa: E402
from quad import reduced as rm                                   # noqa: E402
from quad import sim, support                                    # noqa: E402

PERIODS = 50
ONSET, ANNOUNCE, POSTPONE = 15, 2, 5
PUSH = gv.GLOBAL.push
SHOVE = 20.0                     # [N] beyond the label
GROUND = 0.030                   # [m] the floor's error under FR in two runs
BAND = 0.010                     # [m] the touchdown assumption's band (the slow descent's)
TRACE_DT = 0.01
MENAGERIE = "f3b6013629c7b94f8ee93fd977b161fa8350f9bc"


def down(x, digits=2):
    q = 10 ** digits
    return f"{math.floor(round(float(x) * q, 6)) / q + 0.0:.{digits}f}"


def up(x, digits=2):
    q = 10 ** digits
    return f"{math.ceil(round(float(x) * q, 6)) / q + 0.0:.{digits}f}"


def mm_down(x, digits=1):
    return down(1000.0 * x, digits)


def mm_up(x, digits=1):
    return up(1000.0 * x, digits)


def toward_fr():
    """The direction [rad] of the outward normal of the triangle's edge from FL
    to RR, towards FR: the direction of a push that brings the centre of
    pressure nearest to that edge."""
    fl, rr = rm.FEET[0], rm.FEET[3]
    e = rr - fl
    n = np.array([e[1], -e[0]]) / np.linalg.norm(e)
    if np.dot(rm.FEET[1] - fl, n) < 0:
        n = -n
    return math.atan2(n[1], n[0])


ANGLE = toward_fr()
PUSHES = (sim.Push(2.0, 2.6, PUSH, ANGLE),       # in the shift
          sim.Push(4.4, 4.9, PUSH, ANGLE),       # in the hold
          sim.Push(5.4, 5.8, PUSH, ANGLE))       # in the touchdown
SHOVES = (sim.Push(4.4, 4.9, SHOVE, ANGLE),)

CONFIGS = (
    ("reference", "the global capability, the lift announced in period 2", {}),
    ("pushed", "pushes of 10 N across the triangle's long edge, on FR's side, in the shift, "
     "the hold and the touchdown",
     dict(pushes=PUSHES)),
    ("unannounced", "the lift not announced", dict(announce=None)),
    ("shove", "a shove of 20 N in the pushes' direction in the hold, beyond the global "
     "capability", dict(pushes=SHOVES)),
    ("obstacle", "the floor under FR 30 mm higher than declared", dict(ground=-GROUND)),
    ("hole", "the floor under FR 30 mm lower than declared", dict(ground=GROUND)),
    ("band-higher", "the floor under FR 10 mm higher, the band's edge", dict(ground=-BAND)),
    ("band-lower", "the floor under FR 10 mm lower, the band's edge", dict(ground=BAND)),
)
TRACE_RUNS = ("reference", "pushed", "shove", "ungoverned")


# ----------------------------------------------------------------------------
# Configurations and runs
# ----------------------------------------------------------------------------
def configuration(name, pushes=(), announce=ANNOUNCE, ground=0.0):
    task = gv.LiftFR(announce=announce, onset=ONSET, postpone=POSTPONE)
    world = gv.Floor(pushes=pushes)
    cfg = allovi.build(name, gv.T, dict(control=gv.StancePackage(ground=ground), robot=gv.Robot(),
                                        world=world, proposal=gv.Operator(),
                                        capability=gv.Fixed(), demand=gv.LiftSchedule(task),
                                        task=task))
    return cfg, task, world


def commanded(world, rec=None, ground=0.0):
    """Per step of the log: the commanded place on the line and acceleration,
    whether the step's segment lies where FR may carry less than its share,
    FR's target height, the level in force (NaN under the rest feedback)."""
    n = world.machine.steps()
    cs, ca, ht = np.zeros(n), np.zeros(n), np.zeros(n)
    three, level = np.zeros(n, bool), np.full(n, np.nan)
    for k, law in enumerate(world.laws):
        b0, b1 = world.bounds[k], world.bounds[k + 1]
        seg = law.segment
        taus = (np.arange(b0, b1) - b0 + 1) * world.machine.timestep
        cs[b0:b1] = [seg.position(t) for t in taus]
        ca[b0:b1] = [seg.accel(t) for t in taus]
        ht[b0:b1] = [seg.fr(t, law.ground)[1] for t in taus]
        three[b0:b1] = seg.three()
        if rec is not None and rec.periods[k].governor == "normal":
            level[b0:b1] = rec.periods[k].level
    return cs, ca, ht, three, level


def distances(log):
    cop = np.stack([log["copx"], log["copy"]], axis=1)
    d_tri = np.array([support.signed_distance(rm.TRI, p) for p in cop])
    d_rect = np.array([support.signed_distance(rm.RECT, p) for p in cop])
    return d_tri, d_rect


CONTACT = 0.05            # [N] FR's contact force from which a touchdown counts


def touchdown(log, after):
    """FR's first contact after time 'after' [s], when its contact force
    first exceeds CONTACT: its time, FR's mean downward speed over the five
    steps of 1 ms before it [m/s], and FR's largest force within 50 ms of it
    [N]."""
    idx = np.nonzero((log["t"] > after) & (log["f1"] > CONTACT))[0]
    if len(idx) == 0:
        return None
    i = int(idx[0])
    j = max(0, i - 5)
    v = (log["frz"][i] - log["frz"][j]) / (log["t"][i] - log["t"][j])
    w = (log["t"] >= log["t"][i]) & (log["t"] <= log["t"][i] + 0.05)
    return float(log["t"][i]), float(-v), float(log["f1"][w].max())


def analyse(name, world, rec=None, task=None, ground=0.0):
    log = world.machine.arrays()
    cs, ca, ht, three, level = commanded(world, rec, ground)
    d_tri, d_rect = distances(log)
    com = np.stack([log["cx"], log["cy"]], axis=1) - rm.COM0
    s_m = com @ rm.U
    e_m = com @ np.array([-rm.U[1], rm.U[0]])
    normal = ~np.isnan(level) if rec is not None else np.ones(len(cs), bool)
    out = dict(name=name)
    out["dc"] = float(np.max(np.hypot(s_m - cs, e_m)[normal]))
    out["dh"] = float(np.max(np.abs(log["frz"] - ht)[normal]))
    lift = three & (log["f1"] <= 1e-6)
    out["tri"] = float(np.min(d_tri[three])) if three.any() else None
    out["rect"] = float(np.min(d_rect[~three]))
    out["lifted_tri"] = float(np.min(d_tri[lift])) if lift.any() else None
    out["lifted_tri_t"] = float(log["t"][lift][np.argmin(d_tri[lift])]) if lift.any() else None
    if rec is not None:
        m3 = three & normal
        out["above3"] = float(np.min((d_tri - level)[m3])) if m3.any() else None
        m4 = (~three) & normal
        out["above4"] = float(np.min((gv.C3 + d_tri - level)[m4]))
    lim = np.array([23.7, 23.7, 45.43] * 4)
    out["share"] = float(np.max(np.abs(np.stack([log[f"tau{j}"] for j in range(12)], 1)) / lim))
    out["clipped"] = int(np.sum(np.abs(np.stack([log[f"tau{j}"] for j in range(12)], 1)) > lim))
    stance = np.where(three[:, None], np.array([[1, 0, 1, 1]]), np.array([[1, 1, 1, 1]])).astype(bool)
    f = np.stack([log["f0"], log["f1"], log["f2"], log["f3"]], axis=1)
    loads = np.min(np.where(stance & ~((np.arange(4) == 1)[None, :] & three[:, None]), f, np.inf),
                   axis=1)
    out["least_load"] = float(np.min(loads))
    out["least_load_t"] = float(log["t"][int(np.argmin(loads))])
    out["other"] = float(np.max(log["other"]))
    out["roll"] = float(np.max(np.abs(log["roll"])))
    out["bz"] = float(np.min(log["bz"]))
    t_op = None
    for k, law in enumerate(world.laws):
        if law.segment.op0 == 0.0:
            t_op = k * gv.T
            break
    out["op"] = t_op
    if t_op is not None:
        i_end = int(np.nonzero(log["t"] >= t_op + 0.4 - 1e-9)[0][0])
        out["unloaded"] = float(log["f1"][i_end])
        off = np.nonzero((log["t"] > t_op) & (log["f1"] <= CONTACT))[0]
        out["liftoff"] = float(log["t"][off[0]]) if len(off) else None
        k_op = int(round(t_op / gv.T))
        first = world.laws[k_op].segment
        out["op_start"] = (first.v0, first.a0)
    else:
        out["unloaded"], out["liftoff"], out["op_start"] = None, None, None
    out["touch"] = touchdown(log, (t_op or 0.0) + gv.HOLD_START) if t_op is not None else None
    hold = (log["t"] > (t_op or 0) + gv.HOLD_START) & (log["t"] <= (t_op or 0) + gv.LOWER_START)
    out["fr_hold_min"] = float(np.min(log["frz"][hold])) if t_op is not None and hold.any() else None
    out["f1_max"] = float(np.max(log["f1"]))
    out["f1_max_t"] = float(log["t"][int(np.argmax(log["f1"]))])
    out["refloat"] = None
    if out["touch"] is not None:
        again = np.nonzero((log["t"] > out["touch"][0]) & (log["f1"] <= CONTACT))[0]
        out["refloat"] = float(log["t"][again[0]]) if len(again) else None
    if rec is not None:
        out["alarms"] = list(rec.alarms)
        out["alarm_tri"] = None
        if rec.alarms:
            j = int(np.argmin(np.abs(log["t"] - rec.alarms[0] * gv.T)))
            out["alarm_tri"] = (rec.alarms[0] * gv.T, float(d_tri[j]))
        out["overload_levels"] = [next(p.level for p in rec.periods if p.k == k)
                                  for k in task.overloads]
        out["restarts"] = list(rec.restarts)
        normalp = [p for p in rec.periods if p.governor == "normal"]
        out["interventions"] = [p.k for p in normalp if p.decision.ambition < 1.0]
        out["onset_level"] = next((p.level for p in rec.periods if p.k == task.begun), None) \
            if task.begun is not None else None
        out["overloads"] = list(task.overloads)
        out["begun"] = task.begun
        out["due"] = ONSET
        out["invalid"] = [p.k for p in normalp
                          if not world.admits(*_head_label(p.plan), p.k)]
        ops = [p.level for p in normalp if isinstance(p.state.commanded.op, float)]
        out["op_levels"] = sorted(set(round(x, 3) for x in ops))
        rest_at = [p.k for p in rec.periods
                   if p.state.commanded.op == "after" and abs(p.state.commanded.v) <= 1e-9
                   and abs(p.state.commanded.s) <= 1e-6 and abs(p.state.commanded.a) <= 1e-12]
        out["home"] = rest_at[0] * gv.T if rest_at else None
        out["op_place"] = next((law.segment.s0 for law in world.laws
                                if law.segment.op0 == 0.0), None)
    return out, log, (cs, ca, ht, three, level, d_tri, d_rect, s_m)


def _head_label(plan):
    if plan.pairs:
        return plan.pairs[0][1], plan.pairs[0][0]
    return gv.GLOBAL, plan.head()


def run_governed(name, kw):
    cfg, task, world = configuration(name, **kw)
    rec = allovi.run(cfg, PERIODS)
    out, log, extra = analyse(name, world, rec, task, kw.get("ground", 0.0))
    return out, trace_rows(log, extra), rec, world, task


def run_ungoverned(name="ungoverned", pushes=PUSHES):
    task = gv.LiftFR(announce=ANNOUNCE, onset=ONSET, postpone=POSTPONE)
    world = gv.Floor(pushes=pushes)
    gv.ungoverned(world, gv.Operator(), task, PERIODS)
    out, log, extra = analyse(name, world)
    return out, trace_rows(log, extra)


def trace_rows(log, extra):
    cs, ca, ht, three, level, d_tri, d_rect, s_m = extra
    step = int(round(TRACE_DT / sim.TIMESTEP))
    rows = []
    for j in range(step - 1, len(log["t"]), step):
        rows.append(dict(t=f"{log['t'][j]:.3f}", s_c=f"{cs[j]:.5f}", s_m=f"{s_m[j]:.5f}",
                         h=f"{log['frz'][j]:.5f}", h_t=f"{ht[j]:.5f}", d_tri=f"{d_tri[j]:.5f}",
                         d_rect=f"{d_rect[j]:.5f}", f_fr=f"{log['f1'][j]:.2f}",
                         push=f"{log['push'][j]:.2f}",
                         level="" if np.isnan(level[j]) else f"{level[j]:.3f}"))
    return rows


# ----------------------------------------------------------------------------
# Facts
# ----------------------------------------------------------------------------
def record_facts():
    m = sim.Machine().m
    return {
        "MuJoCo": mujoco.__version__,
        "NumPy, SciPy, Python": f"{np.__version__}, {scipy.__version__}, "
                                f"{platform.python_version()}",
        "the model: MuJoCo Menagerie's unitree_go2/go2.xml, commit; license":
            f"{MENAGERIE[:7]}; BSD-3-Clause",
        "the model's mass [kg]; its motors' torque limits, hip, thigh, knee [N m]; the time step "
        "[s]; MuJoCo's default [s]":
            (f"{np.sum(m.body_mass):.3f}; " + ", ".join(f"{x:g}" for x in m.actuator_ctrlrange[:3, 1])
             + f"; {m.opt.timestep:g}; 0.002"),
        "the governor's period [s]; the operation's phases [periods]; the lift [m]; the "
        "touchdown's descent [m/s] from [m] above the floor":
            (f"{gv.T:g}; " + ", ".join(f"{n} {k}" for n, k in rm.PHASES) +
             f"; {rm.LIFT:g}; {rm.TOUCH_SPEED:g}, {rm.APPROACH:g}"),
        "the lift's onset, announcement and postponement [periods]": f"{ONSET}, {ANNOUNCE}, {POSTPONE}",
        "the levels: f3's floor, the grid, f4's floor, the fourth foot's credit C3, the "
        "requirement [m]":
            f"{gv.FLOOR:g}, {gv.GRID:g}, {gv.FLOOR4:g}, {gv.C3:g}, {gv.FLOOR + gv.C3:g}",
        "the funnel's bounds: the centre of mass, FR's height [m]; the allowance [m]; per newton "
        "of push [m/N]; the global capability [N]":
            f"{gv.DC:g}, {gv.DH:g}; {gv.ALLOWANCE:g}; {gv.K_PUSH:g}; {gv.GLOBAL.push:g}",
        "the largest acceleration [m/s^2]; the friction declared; the torques' factor; the "
        "generator's weights":
            (f"{gv.A_MAX:g}; {gv.MU:g}; {gv.TORQUE_FACTOR:g}; "
             + ", ".join(f"{x:g}" for x in gv.ALPHAS)),
        "the pushes: towards [deg]; the pushed run's [s]; the shove [N, s]; the floor's error "
        "under FR [m]":
            (f"{math.degrees(ANGLE):.1f}; " + ", ".join(f"{p.t0:g}-{p.t1:g}" for p in PUSHES)
             + f"; {SHOVE:g}, {SHOVES[0].t0:g}-{SHOVES[0].t1:g}; {GROUND:g}"),
    }


def model_facts():
    d_tri0, d_rect0 = rm.distances(0.0, 0.0)
    place = gv.rest_place(gv.FLOOR + gv.C3)
    rest0 = gv.rest_reserve(gv.Rest(0.0, "before"))
    restp = gv.rest_reserve(gv.Rest(place, "before"))
    d_tri_p, d_rect_p = rm.distances(place, place)
    three = d_tri_p - gv.DC - gv.ALLOWANCE - gv.K_PUSH * gv.GLOBAL.push
    mc = sim.Machine(home=rm.COM0)
    z_push = float(mc.d.xipos[mc.ids.base][2])
    need = rm.MASS * gv.A_MAX + gv.GLOBAL.push
    n_lo, n_hi = math.ceil(gv.S_MIN * 1000), math.floor(gv.S_MAX * 1000)
    worst3 = max(gv.static_shares(i / 1000, True) for i in range(n_lo, n_hi + 1))
    return {
        "a push's point, the trunk's centre of mass, above the floor [m]; the zero-moment "
        "point's move per newton of push, its height over the weight [mm/N]":
            f"{z_push:.3f}; {1000 * z_push / (rm.MASS * rm.G):.2f}",
        "the horizontal force that the largest acceleration and the global capability's push "
        "need [N]; as a share of the weight":
            f"{up(need, 1)}; {up(need / (rm.MASS * rm.G), 3)}",
        "the triangle FL, RL, RR: its incentre [m]; its inradius [m]; the shift line's direction "
        "[deg]; its length to the incentre [m]":
            (f"{rm.INCENTRE[0]:.4f}, {rm.INCENTRE[1]:.4f}; {down(rm.INRADIUS, 4)}; "
             f"{math.degrees(math.atan2(rm.U[1], rm.U[0])):.1f}; {down(rm.S_INCENTRE, 4)}"),
        "home: the centre of mass inside the triangle and inside the rectangle [m]":
            f"{down(d_tri0, 4)}, {down(d_rect0, 4)}",
        "the allowance in all, at the global capability [m]":
            f"{gv.DC + gv.ALLOWANCE + gv.K_PUSH * gv.GLOBAL.push:.3f}",
        "the rest at home: its reserve [m], its level [m]":
            f"{down(rest0, 4)}, {gv.level_of(rest0):.3f}",
        "the requirement's place on the line [m]; there, the triangle's and the rectangle's "
        "distances [m]; the rest's reserve [m]; the reserve on three feet [m]":
            (f"{place:.4f}; {down(d_tri_p, 4)}, {down(d_rect_p, 4)}; {down(restp, 4)}; "
             f"{down(three, 4)}"),
        "the joints' static torques as shares of the limits, at home and at that place on four "
        "feet; at that place on three":
            (f"{up(gv.static_shares(0.0, False), 3)}, {up(gv.static_shares(place, False), 3)}; "
             f"{up(gv.static_shares(place, True), 3)}"),
        "on three feet, the largest static torque share anywhere on the line's extent, every "
        "millimetre; counted 1.5 times":
            f"{up(worst3, 3)}; {up(gv.TORQUE_FACTOR * worst3, 3)}",
    }


def _opt(x, fmt):
    return "none" if x is None else fmt(x)


def run_facts(results):
    out = {}
    for r in results:
        touch = r["touch"]
        tt = _opt(touch, lambda x: f"{x[0]:.3f} s, {up(x[1], 3)} m/s, {up(x[2], 1)} N")
        op = _opt(r["op"], lambda x: f"{x:.1f} s")
        lo = _opt(r["liftoff"], lambda x: f"{x:.3f} s")
        unl = _opt(r["unloaded"], lambda x: f"{x:.2f} N")
        text = (f"operation from {op}; FR's force at the unloading's end {unl}, off the floor "
                f"from {lo}; inside the triangle while FR may carry less "
                f"{_opt(r['tri'], mm_down)} mm, while FR is off the floor "
                f"{_opt(r['lifted_tri'], mm_down)} mm; inside the rectangle otherwise "
                f"{mm_down(r['rect'])} mm; deviations, centre of mass {mm_up(r['dc'])} mm, FR "
                f"{mm_up(r['dh'])} mm; torque share {up(r['share'], 3)}, clipped steps "
                f"{r['clipped']}; least stance load {down(r['least_load'], 1)} N; touchdown {tt}; "
                f"FR's largest force {up(r['f1_max'], 1)} N; FR's lowest in the hold "
                f"{_opt(r['fr_hold_min'], mm_down)} mm; roll {up(math.degrees(r['roll']), 1)} deg; "
                f"other parts on the floor {up(r['other'], 1)} N; base's lowest {down(r['bz'], 3)} m")
        if "alarms" in r:
            v0, a0 = r["op_start"] if r["op_start"] is not None else (float("nan"), float("nan"))
            text += f"; the operation's first segment from speed {v0:.4f} m/s, acceleration {a0:.4f} m/s^2"
            levels = ", ".join(f"{x:.3f}" for x in r["op_levels"]) or "none"
            joined = lambda xs: ", ".join(str(k) for k in xs) or "none"
            begins = "never" if r["begun"] is None else str(r["begun"] + 1)
            text += (f"; due in period {r['due']}, begins in period {begins}, overloads "
                     f"{joined(r['overloads'])}; level at "
                     f"the onset {_opt(r['onset_level'], lambda x: f'{x:.3f}')} m; levels during "
                     f"the operation {levels} m; the operation's place "
                     f"{_opt(r['op_place'], lambda x: f'{1000 * x:.1f}')} mm; above the level, "
                     f"three feet {_opt(r['above3'], mm_down)} mm, four feet "
                     f"{mm_down(r['above4'])} mm; interventions in periods "
                     f"{joined(r['interventions'])}; invalid labels {joined(r['invalid'])}; alarms "
                     f"{joined(r['alarms'])}; restarts {joined(r['restarts'])}; at rest at home "
                     f"from {_opt(r['home'], lambda x: f'{x:.1f} s')}")
        out[r["name"]] = text
    return out


def failure_facts(results):
    """When things happen in the runs of the unsuccessful conditions, as
    Section 15.2 tells them."""
    r = {x["name"]: x for x in results}
    sh, un, ug = r["shove"], r["unannounced"], r["ungoverned"]
    at, d_at = sh["alarm_tri"]
    return {
        "the shove: the alarm [s] and the distance inside the triangle then [mm]; the least "
        "distance while FR is off the floor [mm] and when [s]; a stance foot's least load [N] "
        "and when [s]; FR's touchdown [s]":
            (f"{at:.1f}, {mm_down(d_at)}; {mm_down(sh['lifted_tri'])}, "
             f"{sh['lifted_tri_t']:.3f}; {down(sh['least_load'], 1)}, {sh['least_load_t']:.3f}; "
             f"{sh['touch'][0]:.3f}"),
        "the unannounced run: the level in force during the periods of its overloads [m]":
            ", ".join(f"{x:.3f}" for x in un["overload_levels"]),
        "the operator without the governor, pushed: FR's touchdown [s]; off the floor again "
        "from [s]; FR's largest force [N] and when [s]":
            (f"{ug['touch'][0]:.3f}; {_opt(ug['refloat'], lambda x: f'{x:.3f}')}; "
             f"{up(ug['f1_max'], 1)}, {ug['f1_max_t']:.3f}"),
    }


LEADS = (7, 8, 9)               # the announcement's periods of the lead-time runs


def lead_time_facts():
    """The lift announced later than in the reference: the overloads and the
    period in which the operation begins (its first period of execution)."""
    out = {}
    for announce in LEADS:
        cfg, task, world = configuration("lead", announce=announce)
        allovi.run(cfg, ONSET + 2 * POSTPONE + 3)
        begins = "never" if task.begun is None else str(task.begun + 1)
        out[f"announced in period {announce}, the onset in {ONSET}: overloads; the operation "
            "begins in period"] = f"{', '.join(map(str, task.overloads)) or 'none'}; {begins}"
    return out


def detail_facts(rec, world, task):
    """The reference run's preparation, as Section 15.2 quotes it."""
    out = {}
    first = next(p for p in rec.periods if p.requirement.level > gv.FLOOR + 1e-9)
    out["the requirement first issued: period; level [m]; onset [pairs]"] = \
        f"{first.k}; {first.requirement.level:.3f}; {first.requirement.onset}"
    out["the level in force in periods 0 to 34 [m]"] = ", ".join(
        f"{p.level:.3f}" for p in rec.periods[:35])
    committed, offered = [], []
    for p in rec.periods[first.k:task.begun + 1]:
        d, n = p.decision, p.requirement.onset
        e = d.entries[d.selected]
        committed.append(f"{p.k}: {e.ambition:g}, {e.profile.level:.3f}, {e.profile.at(n):.3f}")
        best = max((x for x in d.entries if x.kind == "fresh" and x.ambition == 1.0),
                   key=lambda x: x.profile.at(n), default=None)
        if best is not None:
            offered.append(f"{p.k}: {best.profile.level:.3f}, {best.profile.at(n):.3f}")
    out[f"periods {first.k} to {task.begun}: the committed plan's ambition, level and entry at "
        "the onset pair [m]"] = "; ".join(committed)
    out[f"periods {first.k} to {task.begun}: the best candidate of ambition 1, its level and "
        "entry at the onset pair [m]"] = "; ".join(offered)
    moving = next(k for k, law in enumerate(world.laws)
                  if abs(law.segment.a1) > 1e-12 or abs(law.segment.v0) > 1e-9)
    out["the first period in which the centre of mass moves; its place at the onset's start "
        "and where the operation begins [m]; the requirement's place [m]"] = (
        f"{moving}; {next(p.state.commanded.s for p in rec.periods if p.k == task.begun):.4f}, "
        f"{next(law.segment.s0 for law in world.laws if law.segment.op0 == 0.0):.4f}; "
        f"{gv.rest_place(gv.FLOOR + gv.C3):.4f}")
    return out


def shortfalls(world, rec):
    """The simulated centre of pressure's shortfall below the commanded
    zero-moment point's distance from the triangle (f3) and from the
    rectangle (f4, four feet only), the largest over the steps of the
    periods that the governor governs."""
    log = world.machine.arrays()
    cs, ca, ht, three, level = commanded(world, rec)
    normal = ~np.isnan(level)
    zmp = rm.COM0[None, :] + (cs - rm.K * ca)[:, None] * rm.U[None, :]
    d_tri, d_rect = distances(log)
    p_tri = np.array([support.signed_distance(rm.TRI, z) for z in zmp])
    p_rect = np.array([support.signed_distance(rm.RECT, z) for z in zmp])
    s3 = (p_tri - d_tri)[normal]
    s4 = (p_rect - d_rect)[normal & ~three]
    return float(s3.max()), float(s4.max()) if len(s4) else 0.0


PHASES_SWEPT = (("shift's start", 1.6, 2.0), ("shift's end", 2.6, 3.0), ("unload", 3.2, 3.6),
                ("lift", 3.6, 4.0), ("hold", 4.2, 4.8), ("lower", 5.0, 5.4),
                ("touchdown", 5.4, 5.8), ("reload", 5.8, 6.2), ("return", 6.4, 7.0))
DIRECTIONS = tuple(range(0, 360, 45))      # [deg] from the direction towards FR


def _sweep_one(job):
    phase, t0, t1, deg = job
    push = sim.Push(t0, t1, PUSH, ANGLE + math.radians(deg))
    cfg, task, world = configuration("sweep", pushes=(push,))
    rec = allovi.run(cfg, PERIODS)
    out, log, extra = analyse("sweep", world, rec, task)
    s3, s4 = shortfalls(world, rec)
    touch = out["touch"]
    speed = touch[1] if touch is not None else None
    # whether FR leaves the floor again, as in the touchdowns from the hold; and
    # whether it meets the floor outside the slow descent: before its start or
    # after its end, by how long [s], and FR's height then against its target [m]
    again, outside = False, None
    if touch is not None:
        later = (log["t"] > touch[0]) & (log["t"] <= touch[0] + 0.1)
        again = bool(np.any(log["f1"][later] <= CONTACT * 0.5))
        start = out["op"] + sum(n for _, n in rm.PHASES[:4]) * gv.T
        end = start + rm.PHASES[4][1] * gv.T
        i = int(np.nonzero(log["t"] >= touch[0] - 1e-12)[0][0])
        if touch[0] < start - 1e-9:
            outside = ("before", start - touch[0], float(log["frz"][i] - extra[2][i]))
        elif touch[0] > end + 1e-9:
            outside = ("after", touch[0] - end, float(log["frz"][i] - extra[2][i]))
    return (phase, deg, out["dc"], out["dh"], s3, s4, list(rec.alarms), out["above3"],
            out["above4"], speed, again, outside)


def sweep_facts(reference_world, reference_rec, workers):
    jobs = [(ph, t0, t1, deg) for ph, t0, t1 in PHASES_SWEPT for deg in DIRECTIONS]
    with Pool(workers) as pool:
        res = pool.map(_sweep_one, jobs)
    s3_0, s4_0 = shortfalls(reference_world, reference_rec)
    out = {"without a push: the shortfall, f3 and f4 [mm]": f"{mm_up(s3_0)}, {mm_up(s4_0)}"}
    for ph, _, _ in PHASES_SWEPT:
        rows = [r for r in res if r[0] == ph]
        worst = max(rows, key=lambda r: r[4])
        out[f"{ph}: the centre of mass's and FR's largest deviations [mm]; the largest shortfall, "
            "f3 [mm], its direction [deg]; f4 [mm]; alarms"] = (
            f"{mm_up(max(r[2] for r in rows))}, {mm_up(max(r[3] for r in rows))}; "
            f"{mm_up(worst[4])}, {worst[1]}; {mm_up(max(r[5] for r in rows))}; "
            f"{sum(len(r[6]) for r in rows)}")
    total = gv.DC + gv.ALLOWANCE + gv.K_PUSH * PUSH
    out["over the sweep: the centre of mass's and FR's largest deviations [mm]; the bounds [mm]"] = (
        f"{mm_up(max(r[2] for r in res))}, {mm_up(max(r[3] for r in res))}; "
        f"{1000 * gv.DC:g}, {1000 * gv.DH:g}")
    out["over the sweep: the largest shortfall, f3 and f4 [mm]; the allowance in all at the "
        "label's push [mm]; without a push [mm]"] = (
        f"{mm_up(max(r[4] for r in res))}, {mm_up(max(r[5] for r in res))}; {1000 * total:g}; "
        f"{1000 * (gv.DC + gv.ALLOWANCE):g}")
    out["over the sweep: the least distance above the level, three feet and four feet [mm]; "
        "runs; alarms"] = (
        f"{mm_down(min(r[7] for r in res))}, {mm_down(min(r[8] for r in res))}; {len(res)}; "
        f"{sum(len(r[6]) for r in res)}")
    # FR's actual speed at its first contact, against the profile's commanded descent
    touched = [r for r in res if r[9] is not None]
    fastest = max(touched, key=lambda r: r[9])
    during = [r[9] for r in touched if r[0] == "touchdown"]
    bounced = [r for r in res if r[10]]
    out["over the sweep: FR's largest downward speed at its first contact [m/s], the phase and "
        "the direction [deg] of the push; with the push during the touchdown, the least and the "
        "largest [m/s]; runs without a first contact; runs in which FR leaves the floor again "
        "within 0.1 s of it, as in the touchdowns from the hold, with the phase and the "
        "direction [deg] of the push and FR's speed at its first contact [m/s]; the profile's "
        "commanded descent [m/s]"] = (
        f"{up(fastest[9], 3)}, {fastest[0]}, {fastest[1]}; {down(min(during), 3)}, "
        f"{up(max(during), 3)}; {len(res) - len(touched)}; {len(bounced)}"
        + "".join(f", {r[0]} {r[1]} at {up(r[9], 3)}" for r in bounced)
        + f"; {rm.TOUCH_SPEED:g}")
    before = [r for r in res if r[11] is not None and r[11][0] == "before"]
    after = [r for r in res if r[11] is not None and r[11][0] == "after"]
    out["over the sweep: runs in which FR meets the floor before the slow descent begins, with "
        "the phase and the direction [deg] of the push, how long before [s] and FR's height then "
        "against its target [mm]; runs in which it meets the floor after the slow descent ends"] = (
        f"{len(before)}"
        + "".join(f", {r[0]} {r[1]}, {r[11][1]:.3f}, {1000 * r[11][2]:+.1f}" for r in before)
        + f"; {len(after)}")
    return out


def branch_facts(name, kw):
    """Every plan in force of a governed run executed unchanged from its
    period's start, its head, its recovery and three periods of its rest,
    from the simulated state at that period's start (a copy of the world).
    The least simulated distance above each pair's profile entry: inside the
    triangle where FR may carry less than its share; on four feet, the
    triangle's distance with the credit C3 above the entry, and the
    rectangle's above f4's floor."""
    cfg, task, world = configuration(name, **kw)
    rec = allovi.run(cfg, PERIODS)
    ctl = cfg.control
    w2 = gv.Floor(pushes=kw.get("pushes", ()))
    w2.reset()
    gap3, gap4, gap_rect, plans = math.inf, math.inf, math.inf, 0
    dc, dh = 0.0, 0.0
    for k, law in enumerate(world.laws):
        p = rec.periods[k]
        plan = p.plan
        if p.governor == "normal" and plan is not None and plan.pairs:
            plans += 1
            log_keep = w2.machine.log
            w2.machine.log = {key: [] for key in sim.KEYS}
            br = copy.deepcopy(w2)
            w2.machine.log = log_keep
            br.bounds, br.laws = [0], []
            prof = ctl.profile(plan, _initial_of(plan))
            for seg, _ in plan.pairs:
                br.step(None, gv.Law(seg, ground=ctl.ground))
            rest = gv.Segment(plan.rest.s, 0.0, 0.0, 0.0, plan.rest.op)
            for _ in range(3):
                br.step(None, gv.Law(rest, ground=ctl.ground))
            log = br.machine.arrays()
            d_tri, d_rect = distances(log)
            cs, ca, ht, three, _ = commanded(br)
            com = np.stack([log["cx"], log["cy"]], axis=1) - rm.COM0
            s_m, e_m = com @ rm.U, com @ np.array([-rm.U[1], rm.U[0]])
            dc = max(dc, float(np.max(np.hypot(s_m - cs, e_m))))
            dh = max(dh, float(np.max(np.abs(log["frz"] - ht))))
            for n in range(len(plan.pairs) + 3):
                a, b = br.bounds[n], br.bounds[n + 1]
                entry = prof.at(min(n, len(plan.pairs)))
                if three[a]:
                    gap3 = min(gap3, float(np.min(d_tri[a:b])) - entry)
                else:
                    gap4 = min(gap4, float(np.min(gv.C3 + d_tri[a:b])) - entry)
                    gap_rect = min(gap_rect, float(np.min(d_rect[a:b])) - gv.FLOOR4)
        w2.step(None, law)
    return dict(plans=plans, gap3=gap3, gap4=gap4, gap_rect=gap_rect, dc=dc, dh=dh)


def _initial_of(plan):
    h = plan.head()
    op = "before" if h.op0 == 0.0 else h.op0
    return gv.State(h.s0, h.v0, h.a0, op)


TOUCH_SPEEDS = (0.05, 0.1, 0.2, 0.4)
CONTACTS = (("Menagerie's soft feet", None), ("MuJoCo's default impedance", "stiff"))


def _touch_once(v, solimp):
    """From the hold at the requirement's place, FR is lowered at a constant
    speed v from 30 mm above the floor's contact to 5 mm below it, held
    there, and reloaded 0.3 s after its first contact."""
    from quad import model
    mc = sim.Machine(home=rm.COM0, foot_solimp=None if solimp is None else model.STIFF_SOLIMP)
    onset = mc.ids.radius - float(mc.feet0[1, 2])       # the contact's onset above the stance place
    place = gv.rest_place(gv.FLOOR + gv.C3)
    c1 = rm.COM0 + place * rm.U
    hc = rm.hc
    zero = np.zeros(2)
    mc.advance(1.6, lambda t: (rm.COM0 + (c1 - rm.COM0) * hc(t, 1.6), zero, 1.0, 0.0))
    mc.advance(0.4, lambda t: (c1, zero, 1.0, 0.0))
    share = float(mc.log["f1"][-1])
    top = onset + rm.LIFT
    start, end = onset + 0.030, onset - 0.005
    mc.advance(0.4, lambda t: (c1, zero, 1.0 - hc(t, 0.4), onset * hc(t, 0.4)))
    mc.advance(0.4, lambda t: (c1, zero, 0.0, onset + rm.LIFT * hc(t, 0.4)))
    mc.advance(0.4, lambda t: (c1, zero, 0.0, top))
    mc.advance(0.4, lambda t: (c1, zero, 0.0, top - (top - start) * hc(t, 0.4)))
    n0 = mc.steps()
    mc.advance((start - end) / v + 0.3, lambda t: (c1, zero, 0.0, max(end, start - v * t)))
    mc.advance(0.4, lambda t: (c1, zero, hc(t, 0.4), end * (1.0 - hc(t, 0.4))))
    mc.advance(0.5, lambda t: (c1, zero, 1.0, 0.0))
    log = mc.arrays()
    td = touchdown(log, log["t"][n0] - 1e-9)
    i = int(np.nonzero(log["t"] >= td[0])[0][0])
    w = (log["t"] >= td[0]) & (log["t"] <= td[0] + 0.05)
    cop = np.stack([log["copx"], log["copy"]], axis=1)
    moves = cop[w] - cop[i - 1]
    j = int(np.argmax(np.linalg.norm(moves, axis=1)))
    jump = float(np.linalg.norm(moves[j]))
    to_fr = mc.feet0[1, :2] - cop[i - 1]
    towards = float(np.dot(moves[j], to_fr) / (np.linalg.norm(moves[j]) * np.linalg.norm(to_fr)
                                               + 1e-12))
    w2 = (log["t"] >= td[0]) & (log["t"] <= td[0] + 0.2)
    com = np.stack([log["cx"], log["cy"]], axis=1)
    dev = float(np.max(np.linalg.norm(com[w2] - c1, axis=1)))
    others = float(np.min(np.stack([log["f0"], log["f2"], log["f3"]], axis=1)[w]))
    later = (log["t"] > td[0]) & (log["t"] <= td[0] + 0.1)
    bounce = bool(np.any(log["f1"][later] <= CONTACT * 0.5))
    return onset, share, td, jump, dev, others, bounce, towards


def touchdown_facts():
    """Touchdowns from the hold at four speeds on two contact models: the
    contact's speed, FR's largest force within 50 ms of it against its share
    on four feet, the largest jump of the centre of pressure in that time,
    the centre of mass's largest deviation from its place in 0.2 s, the
    other feet's least load in 50 ms, and whether FR leaves the floor again
    within 0.1 s."""
    out = {}
    for name, solimp in CONTACTS:
        onset = None
        for v in TOUCH_SPEEDS:
            onset, share, td, jump, dev, others, bounce, towards = _touch_once(v, solimp)
            out[f"{name}, at {v:g} m/s: the contact's speed [m/s]; FR's largest force in 50 ms "
                "[N], against its share [N]; the centre of pressure's largest jump in 50 ms [mm], "
                "the cosine of its angle to FR's foot; the centre of mass's largest deviation in "
                "0.2 s [mm]; the other feet's least load in 50 ms [N]; FR leaves the floor again "
                "in 0.1 s"] = (
                f"{up(td[1], 3)}; {up(td[2], 1)}, {down(share, 1)}; {mm_up(jump)}, "
                f"{towards:.2f}; {mm_up(dev)}; {down(others, 1)}; {'yes' if bounce else 'no'}")
        out[f"{name}: a loaded foot's sink [mm]"] = f"{1000 * onset:.1f}"
    return out


def shutdown_facts():
    """From home the robot crouches, its base from 0.27 to 0.11 m over 2 s,
    holds 0.5 s, cuts its motors' torques over 0.1 s and lies 12 s."""
    mc = sim.Machine(home=rm.COM0)
    c = rm.COM0.copy()
    z0, z1 = sim.BASE_HEIGHT, 0.11
    hc = rm.hc
    mc.advance(2.0, lambda t: (c, np.zeros(2), 1.0, 0.0, z0 + (z1 - z0) * hc(t, 2.0)))
    mc.advance(0.5, lambda t: (c, np.zeros(2), 1.0, 0.0, z1))
    n_off = mc.steps()
    knee_crouch = min(float(mc.d.qpos[7 + 3 * leg + 2]) for leg in range(4))
    knee_limit = float(mc.m.jnt_range[3][0])
    first_part = None

    def watch(m, d):
        for ci in range(d.ncon):
            con = d.contact[ci]
            g = con.geom2 if con.geom1 == mc.ids.floor else con.geom1
            if g not in mc.ids.feet:
                return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
        return None
    mc.advance(0.1, lambda t: (c, np.zeros(2), 1.0, 0.0, z1), power=lambda t: 1.0 - hc(t, 0.1))
    for _ in range(int(round(12.0 / 0.001))):
        mc.advance(0.001, lambda t: (c, np.zeros(2), 1.0, 0.0, z1), power=lambda t: 0.0)
        if first_part is None:
            first_part = watch(mc.m, mc.d)
    log = mc.arrays()
    t = log["t"]
    d_tri, d_rect = distances(log)
    lim = np.array([23.7, 23.7, 45.43] * 4)
    shares = np.max(np.abs(np.stack([log[f"tau{j}"] for j in range(12)], 1)) / lim, axis=1)
    crouch = np.arange(len(t)) < n_off
    first = np.nonzero(log["other"] > 0.5)[0]
    vz = np.gradient(log["bz"], t)
    i = int(first[0])
    m, d = mc.m, mc.d
    parts = set()
    for ci in range(d.ncon):
        con = d.contact[ci]
        g = con.geom2 if con.geom1 == mc.ids.floor else con.geom1
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
        parts.add("foot" if g in mc.ids.feet else name.split("_")[-1])
    last = t > t[-1] - 1.0
    settled = None
    for T0 in np.arange(t[n_off], t[-1] - 1.0, 0.5):
        w = (t > T0) & (t <= T0 + 1.0)
        if 1000 * np.hypot(np.ptp(log["cx"][w]), np.ptp(log["cy"][w])) < 0.01:
            settled = T0 - t[n_off]
            break
    return {
        "the crouch: the centre of pressure's least distance inside the rectangle [mm]; the "
        "largest torque share; the knees' angle [rad], against the end of their range [rad]":
            (f"{mm_down(np.min(d_rect[crouch]))}; {up(np.max(shares[crouch]), 3)}; "
             f"{knee_crouch:.3f}, {knee_limit:.4f}"),
        "after the torques are cut: the first part other than a foot on the floor":
            f"{first_part}",
        "after the torques are cut: the first contact of another part [s after]; the base's "
        "height then [m] and downward speed [m/s]; the largest downward speed [m/s]":
            (f"{t[i] - t[n_off]:.3f}; {down(log['bz'][i], 4)}, {up(-vz[i], 3)}; "
             f"{up(-np.min(vz[n_off:]), 3)}"),
        "after the torques are cut: when the base falls fastest [s after]":
            f"{t[n_off + int(np.argmin(vz[n_off:]))] - t[n_off]:.3f}",
        "at the end: the base's height [m]; its pitch and roll [deg]; the parts on the floor; the "
        "load on parts other than the feet [N] of the weight [N]":
            (f"{down(log['bz'][-1], 4)}; {up(math.degrees(abs(log['pitch'][-1])), 2)}, "
             f"{up(math.degrees(abs(log['roll'][-1])), 2)}; {', '.join(sorted(parts))}; "
             f"{down(log['other'][-1], 1)}, {rm.MASS * rm.G:.1f}"),
        "the centre of mass's motion in the last second [mm]; when it first moves less than "
        "0.01 mm in a second [s after the torques are cut]":
            f"{up(1000 * np.hypot(np.ptp(log['cx'][last]), np.ptp(log['cy'][last])), 4)}; "
            f"{_opt(settled, lambda x: f'{x:.1f}')}",
    }


def check_facts(make):
    results = checks.check(make, PERIODS, fmt=lambda x: f"{x}")
    out = {"summary": checks.summary(results)}
    for res in results:
        out[res.name] = f"{res.status}" + (f" ({res.cases} cases)" if res.cases else "") + \
            (f": {res.detail}" if res.detail and res.status != checks.PASSED else "")
    return out


def write_record(path, rec):
    bounds = rec["the funnel's bounds: the centre of mass, FR's height [m]; the allowance [m]; "
                 "per newton of push [m/N]; the global capability [N]"]
    lines = [
        "Record of the experiment of Section 15.2: the Go2 governed through a lift of its "
        "front-right foot",
        "",
        "model and asset provenance",
        f"  MuJoCo Menagerie's unitree_go2/go2.xml, commit {MENAGERIE}, BSD-3-Clause, derived by "
        "Menagerie from Unitree's URDF; its visual meshes, which carry no mass and no collision, "
        "deleted; a floor plane added (quad/model.py); the stance controller, the reduced model, "
        "the task, the modules and the runs ours (quad/)",
        "simulator and solver",
        f"  MuJoCo {rec['MuJoCo']}; NumPy, SciPy and Python {rec['NumPy, SciPy, Python']}; "
        "go2.xml's options: Euler, elliptic cones, impratio 100, the feet's soft contacts of "
        "Menagerie",
        "time steps",
        f"  the simulation 1 ms (go2.xml leaves MuJoCo's default of 2 ms); the governor's period "
        f"{gv.T:g} s; {PERIODS} periods, the robot settled 3 s before the first",
        "actuators and sensors",
        "  the twelve motors of go2.xml, torques within 23.7, 23.7 and 45.43 N m; the stance "
        "controller of quad/stance.py at every step; the measurement the simulated state, exact",
        "frames and units",
        "  SI units; world x forward, y left, z up; the state's place along the shift line from "
        "home towards the incentre of the triangle FL, RL, RR",
        "uncertainty bounds",
        "  the labels' bound on a push on the trunk (10 N, the global capability); the funnel's "
        f"bounds, the allowance, its part per newton of push and the global capability: {bounds}",
        "constraints",
        "  f4, the zero-moment point inside the rectangle of the four feet, at its floor; f3, "
        "inside the triangle of FL, RL, RR, with the fourth foot's credit C3 while FR carries its "
        "share; the levels: " + rec["the levels: f3's floor, the grid, f4's floor, the fourth "
                                    "foot's credit C3, the requirement [m]"] + " m",
        "terminal behaviour",
        "  the rest on four feet, powered, a rest under the assumption that power and control "
        "last; after an alarm the rest feedback, which lowers FR at once; the shutdown "
        "construction run apart, not certified",
        "configurations",
    ]
    lines += [f"  {name}: {what}" for name, what, _ in CONFIGS]
    lines += [
        "  ungoverned: the operator's proposal executed directly, with the pushed run's pushes",
        "  ungoverned-calm: the same, without pushes",
        "training and evaluation split",
        "  none: no learned module",
        "seeds",
        "  none: no random numbers",
        "hardware",
        f"  {platform.system()} {platform.machine()}, {_hardware()}",
        "guarantee status",
        "  tested in simulation: the certificate rests on the reduced model, the allowances and "
        "the funnel's bounds, which the runs, the sweep and every plan executed unchanged check in "
        "MuJoCo, in sample, and on the assumptions on the touchdown and on power; no physical "
        "robot",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _hardware():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "not reported"


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description="Section 15.2: the Go2 governed")
    ap.add_argument("--check", action="store_true", help="check every value of the summary")
    ap.add_argument("--workers", type=int, default=2, help="processes for the push sweep")
    args = ap.parse_args()
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    facts = {"record": record_facts(), "the reduced model": model_facts()}
    results, traces, kept = [], {}, {}
    for name, _, kw in CONFIGS:
        r, rows, rec, world, task = run_governed(name, kw)
        results.append(r)
        traces[name] = rows
        kept[name] = (rec, world, task)
    r, rows = run_ungoverned()
    results.append(r)
    traces["ungoverned"] = rows
    r, rows = run_ungoverned("ungoverned-calm", ())
    results.append(r)
    facts["the runs"] = run_facts(results)
    facts["the reference run"] = detail_facts(*kept["reference"])
    facts["the unsuccessful conditions"] = failure_facts(results)
    facts["the lead time"] = lead_time_facts()
    kw = {c[0]: c[2] for c in CONFIGS}
    facts["the checks of Section 10.3, on the reference"] = check_facts(
        lambda: configuration("reference")[0])
    facts["the checks of Section 10.3, on the shove"] = check_facts(
        lambda: configuration("shove", **kw["shove"])[0])
    facts["pushes of 10 N swept over the reference run"] = sweep_facts(
        kept["reference"][1], kept["reference"][0], args.workers)
    branches = {}
    for name in ("reference", "pushed"):
        b = branch_facts(name, kw[name])
        branches[name] = b
    facts["every plan in force executed unchanged"] = {
        f"{name}: plans; the least distance above the entries, three feet, four feet with C3 "
        "[mm]; above f4's floor [mm]; the deviations, centre of mass and FR [mm]":
            (f"{b['plans']}; {mm_down(b['gap3'])}, {mm_down(b['gap4'])}; "
             f"{mm_down(b['gap_rect'])}; {mm_up(b['dc'])}, {mm_up(b['dh'])}")
        for name, b in branches.items()}
    facts["touchdowns from the hold"] = touchdown_facts()
    facts["the shutdown"] = shutdown_facts()
    for name in TRACE_RUNS:
        write_csv(os.path.join(out_dir, f"governed-trace-{name}.csv"), traces[name])
    write_record(os.path.join(out_dir, "governed-record.txt"), facts["record"])
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
        lines.append(f"[check] {n} values, among them every value that Section 15.2 quotes: " +
                     ("all as expected" if not failures and n else "FAILED: " + "; ".join(failures)))
    lines.append(f"[time] {time.time() - t0:.0f} s")
    with open(os.path.join(out_dir, "governed-summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines[:-1]) + "\n")
    print("\n".join(lines))
    if failures:
        raise SystemExit(1)


EXPECTED = {   # every value of the summary but the versions of the libraries
 'record': {"the model: MuJoCo Menagerie's unitree_go2/go2.xml, commit; license": 'f3b6013; '
                                                                                  'BSD-3-Clause',
            "the model's mass [kg]; its motors' torque limits, hip, thigh, knee [N m]; the time step [s]; MuJoCo's default [s]": '15.206; '
                                                                                                                                 '23.7, '
                                                                                                                                 '23.7, '
                                                                                                                                 '45.43; '
                                                                                                                                 '0.001; '
                                                                                                                                 '0.002',
            "the governor's period [s]; the operation's phases [periods]; the lift [m]; the touchdown's descent [m/s] from [m] above the floor": '0.2; '
                                                                                                                                                 'unload '
                                                                                                                                                 '2, '
                                                                                                                                                 'lift '
                                                                                                                                                 '2, '
                                                                                                                                                 'hold '
                                                                                                                                                 '5, '
                                                                                                                                                 'lower '
                                                                                                                                                 '2, '
                                                                                                                                                 'touchdown '
                                                                                                                                                 '2, '
                                                                                                                                                 'reload '
                                                                                                                                                 '2; '
                                                                                                                                                 '0.06; '
                                                                                                                                                 '0.05, '
                                                                                                                                                 '0.01',
            "the lift's onset, announcement and postponement [periods]": '15, 2, 5',
            "the levels: f3's floor, the grid, f4's floor, the fourth foot's credit C3, the requirement [m]": '0.02, '
                                                                                                              '0.005, '
                                                                                                              '0.02, '
                                                                                                              '0.06, '
                                                                                                              '0.08',
            "the funnel's bounds: the centre of mass, FR's height [m]; the allowance [m]; per newton of push [m/N]; the global capability [N]": '0.008, '
                                                                                                                                                '0.02; '
                                                                                                                                                '0.005; '
                                                                                                                                                '0.003; '
                                                                                                                                                '10',
            "the largest acceleration [m/s^2]; the friction declared; the torques' factor; the generator's weights": '0.25; '
                                                                                                                     '0.5; '
                                                                                                                     '1.5; '
                                                                                                                     '1, '
                                                                                                                     '0.75, '
                                                                                                                     '0.5, '
                                                                                                                     '0.25',
            "the pushes: towards [deg]; the pushed run's [s]; the shove [N, s]; the floor's error under FR [m]": '-53.1; '
                                                                                                                 '2-2.6, '
                                                                                                                 '4.4-4.9, '
                                                                                                                 '5.4-5.8; '
                                                                                                                 '20, '
                                                                                                                 '4.4-4.9; '
                                                                                                                 '0.03'},
 'the reduced model': {"a push's point, the trunk's centre of mass, above the floor [m]; the zero-moment point's move per newton of push, its height over the weight [mm/N]": '0.265; '
                                                                                                                                                                              '1.77',
                       "the horizontal force that the largest acceleration and the global capability's push need [N]; as a share of the weight": '13.9; '
                                                                                                                                                 '0.093',
                       "the triangle FL, RL, RR: its incentre [m]; its inradius [m]; the shift line's direction [deg]; its length to the incentre [m]": '-0.1002, '
                                                                                                                                                        '0.0485; '
                                                                                                                                                        '0.0968; '
                                                                                                                                                        '148.1; '
                                                                                                                                                        '0.0917',
                       'home: the centre of mass inside the triangle and inside the rectangle [m]': '0.0112, '
                                                                                                    '0.1453',
                       'the allowance in all, at the global capability [m]': '0.043',
                       'the rest at home: its reserve [m], its level [m]': '0.0282, 0.025',
                       "the requirement's place on the line [m]; there, the triangle's and the rectangle's distances [m]; the rest's reserve [m]; the reserve on three feet [m]": '0.0555; '
                                                                                                                                                                                  '0.0630, '
                                                                                                                                                                                  '0.1159; '
                                                                                                                                                                                  '0.0800; '
                                                                                                                                                                                  '0.0200',
                       "the joints' static torques as shares of the limits, at home and at that place on four feet; at that place on three": '0.153, '
                                                                                                                                             '0.242; '
                                                                                                                                             '0.280',
                       "on three feet, the largest static torque share anywhere on the line's extent, every millimetre; counted 1.5 times": '0.331; '
                                                                                                                                            '0.496'},
 'the runs': {'reference': "operation from 3.2 s; FR's force at the unloading's end 0.43 N, off "
                           'the floor from 3.611 s; inside the triangle while FR may carry less '
                           '55.4 mm, while FR is off the floor 57.3 mm; inside the rectangle '
                           'otherwise 112.2 mm; deviations, centre of mass 3.4 mm, FR 7.8 mm; '
                           'torque share 0.286, clipped steps 0; least stance load 14.0 N; '
                           "touchdown 5.592 s, 0.050 m/s, 0.2 N; FR's largest force 37.7 N; FR's "
                           'lowest in the hold 69.4 mm; roll 0.8 deg; other parts on the floor 0.0 '
                           "N; base's lowest 0.268 m; the operation's first segment from speed "
                           '0.0030 m/s, acceleration -0.0297 m/s^2; due in period 15, begins in '
                           'period 16, overloads none; level at the onset 0.080 m; levels during '
                           "the operation 0.020 m; the operation's place 56.0 mm; above the level, "
                           'three feet 35.4 mm, four feet 41.3 mm; interventions in periods 8, 9, '
                           '10, 11, 12, 13, 14; invalid labels none; alarms none; restarts none; '
                           'at rest at home from 7.6 s',
              'pushed': "operation from 3.2 s; FR's force at the unloading's end 0.39 N, off the "
                        'floor from 3.610 s; inside the triangle while FR may carry less 30.5 mm, '
                        'while FR is off the floor 35.6 mm; inside the rectangle otherwise 110.8 '
                        'mm; deviations, centre of mass 6.9 mm, FR 13.0 mm; torque share 0.301, '
                        'clipped steps 0; least stance load 13.5 N; touchdown 5.544 s, 0.085 m/s, '
                        "0.5 N; FR's largest force 45.8 N; FR's lowest in the hold 60.0 mm; roll "
                        "2.2 deg; other parts on the floor 0.0 N; base's lowest 0.268 m; the "
                        "operation's first segment from speed 0.0030 m/s, acceleration -0.0297 "
                        'm/s^2; due in period 15, begins in period 16, overloads none; level at '
                        "the onset 0.080 m; levels during the operation 0.020 m; the operation's "
                        'place 56.0 mm; above the level, three feet 10.5 mm, four feet 17.8 mm; '
                        'interventions in periods 8, 9, 10, 11, 12, 13, 14; invalid labels none; '
                        'alarms none; restarts none; at rest at home from 7.6 s',
              'unannounced': "operation from 5.2 s; FR's force at the unloading's end 0.65 N, off "
                             'the floor from 5.618 s; inside the triangle while FR may carry less '
                             '52.6 mm, while FR is off the floor 55.1 mm; inside the rectangle '
                             'otherwise 112.5 mm; deviations, centre of mass 3.6 mm, FR 8.5 mm; '
                             'torque share 0.286, clipped steps 0; least stance load 14.2 N; '
                             "touchdown 7.591 s, 0.050 m/s, 0.2 N; FR's largest force 37.7 N; FR's "
                             'lowest in the hold 69.0 mm; roll 0.9 deg; other parts on the floor '
                             "0.0 N; base's lowest 0.268 m; the operation's first segment from "
                             'speed 0.0000 m/s, acceleration 0.0000 m/s^2; due in period 15, '
                             'begins in period 26, overloads 15, 20; level at the onset 0.080 m; '
                             "levels during the operation 0.020 m; the operation's place 55.5 mm; "
                             'above the level, three feet 32.6 mm, four feet 41.3 mm; '
                             'interventions in periods 15, 16, 17, 19, 20, 21, 22, 23, 24; invalid '
                             'labels none; alarms none; restarts none; at rest at home from 9.6 s',
              'shove': "operation from 3.2 s; FR's force at the unloading's end 0.43 N, off the "
                       'floor from 3.611 s; inside the triangle while FR may carry less -30.1 mm, '
                       'while FR is off the floor 0.4 mm; inside the rectangle otherwise 109.6 mm; '
                       'deviations, centre of mass 13.5 mm, FR 17.9 mm; torque share 0.354, '
                       'clipped steps 0; least stance load 0.2 N; touchdown 4.817 s, 0.249 m/s, '
                       "10.5 N; FR's largest force 37.7 N; FR's lowest in the hold 0.5 mm; roll "
                       "6.3 deg; other parts on the floor 0.0 N; base's lowest 0.268 m; the "
                       "operation's first segment from speed 0.0030 m/s, acceleration -0.0297 "
                       'm/s^2; due in period 15, begins in period 16, overloads none; level at the '
                       "onset 0.080 m; levels during the operation 0.020 m; the operation's place "
                       '56.0 mm; above the level, three feet -11.2 mm, four feet 41.3 mm; '
                       'interventions in periods 8, 9, 10, 11, 12, 13, 14; invalid labels 21, 22; '
                       'alarms 23; restarts 30; at rest at home from 7.2 s',
              'obstacle': "operation from 3.2 s; FR's force at the unloading's end 0.43 N, off the "
                          'floor from 3.620 s; inside the triangle while FR may carry less 29.9 '
                          'mm, while FR is off the floor 58.0 mm; inside the rectangle otherwise '
                          '112.6 mm; deviations, centre of mass 10.0 mm, FR 27.9 mm; torque share '
                          '0.310, clipped steps 0; least stance load 14.8 N; touchdown 5.234 s, '
                          "0.192 m/s, 4.9 N; FR's largest force 37.7 N; FR's lowest in the hold "
                          "39.9 mm; roll 3.5 deg; other parts on the floor 0.0 N; base's lowest "
                          "0.268 m; the operation's first segment from speed 0.0030 m/s, "
                          'acceleration -0.0297 m/s^2; due in period 15, begins in period 16, '
                          'overloads none; level at the onset 0.080 m; levels during the operation '
                          "0.020 m; the operation's place 56.0 mm; above the level, three feet "
                          '22.8 mm, four feet 41.3 mm; interventions in periods 8, 9, 10, 11, 12, '
                          '13, 14; invalid labels none; alarms 29; restarts 32; at rest at home '
                          'from 7.6 s',
              'hole': "operation from 3.2 s; FR's force at the unloading's end 0.43 N, off the "
                      'floor from 3.608 s; inside the triangle while FR may carry less 31.9 mm, '
                      'while FR is off the floor 57.1 mm; inside the rectangle otherwise 100.3 mm; '
                      'deviations, centre of mass 4.7 mm, FR 10.2 mm; torque share 0.286, clipped '
                      'steps 0; least stance load 15.8 N; touchdown 5.968 s, 0.208 m/s, 7.5 N; '
                      "FR's largest force 37.7 N; FR's lowest in the hold 99.0 mm; roll 1.1 deg; "
                      "other parts on the floor 0.0 N; base's lowest 0.267 m; the operation's "
                      'first segment from speed 0.0030 m/s, acceleration -0.0297 m/s^2; due in '
                      'period 15, begins in period 16, overloads none; level at the onset 0.080 m; '
                      "levels during the operation 0.020 m; the operation's place 56.0 mm; above "
                      'the level, three feet 11.9 mm, four feet 41.3 mm; interventions in periods '
                      '8, 9, 10, 11, 12, 13, 14; invalid labels none; alarms none; restarts none; '
                      'at rest at home from 7.6 s',
              'band-higher': "operation from 3.2 s; FR's force at the unloading's end 0.43 N, off "
                             'the floor from 3.613 s; inside the triangle while FR may carry less '
                             '55.4 mm, while FR is off the floor 57.5 mm; inside the rectangle '
                             'otherwise 112.6 mm; deviations, centre of mass 4.9 mm, FR 9.5 mm; '
                             'torque share 0.286, clipped steps 0; least stance load 13.9 N; '
                             "touchdown 5.401 s, 0.028 m/s, 0.3 N; FR's largest force 37.7 N; FR's "
                             'lowest in the hold 59.6 mm; roll 1.1 deg; other parts on the floor '
                             "0.0 N; base's lowest 0.268 m; the operation's first segment from "
                             'speed 0.0030 m/s, acceleration -0.0297 m/s^2; due in period 15, '
                             'begins in period 16, overloads none; level at the onset 0.080 m; '
                             "levels during the operation 0.020 m; the operation's place 56.0 mm; "
                             'above the level, three feet 35.4 mm, four feet 38.1 mm; '
                             'interventions in periods 8, 9, 10, 11, 12, 13, 14; invalid labels '
                             'none; alarms none; restarts none; at rest at home from 7.6 s',
              'band-lower': "operation from 3.2 s; FR's force at the unloading's end 0.43 N, off "
                            'the floor from 3.610 s; inside the triangle while FR may carry less '
                            '51.7 mm, while FR is off the floor 57.2 mm; inside the rectangle '
                            'otherwise 109.4 mm; deviations, centre of mass 3.4 mm, FR 8.5 mm; '
                            'torque share 0.286, clipped steps 0; least stance load 14.7 N; '
                            "touchdown 5.792 s, 0.050 m/s, 0.2 N; FR's largest force 37.7 N; FR's "
                            'lowest in the hold 79.3 mm; roll 0.8 deg; other parts on the floor '
                            "0.0 N; base's lowest 0.268 m; the operation's first segment from "
                            'speed 0.0030 m/s, acceleration -0.0297 m/s^2; due in period 15, '
                            'begins in period 16, overloads none; level at the onset 0.080 m; '
                            "levels during the operation 0.020 m; the operation's place 56.0 mm; "
                            'above the level, three feet 31.7 mm, four feet 41.3 mm; interventions '
                            'in periods 8, 9, 10, 11, 12, 13, 14; invalid labels none; alarms '
                            'none; restarts none; at rest at home from 7.6 s',
              'ungoverned': "operation from 3.2 s; FR's force at the unloading's end 3.72 N, off "
                            'the floor from 3.653 s; inside the triangle while FR may carry less '
                            '-139.8 mm, while FR is off the floor -13.1 mm; inside the rectangle '
                            'otherwise 85.1 mm; deviations, centre of mass 83.8 mm, FR 72.5 mm; '
                            'torque share 0.416, clipped steps 0; least stance load 0.0 N; '
                            "touchdown 4.977 s, 0.250 m/s, 28.8 N; FR's largest force 113.6 N; "
                            "FR's lowest in the hold 7.9 mm; roll 17.1 deg; other parts on the "
                            "floor 0.0 N; base's lowest 0.266 m",
              'ungoverned-calm': "operation from 3.2 s; FR's force at the unloading's end 3.92 N, "
                                 'off the floor from 3.653 s; inside the triangle while FR may '
                                 'carry less -1.2 mm, while FR is off the floor 3.9 mm; inside the '
                                 'rectangle otherwise 143.1 mm; deviations, centre of mass 5.8 mm, '
                                 'FR 14.7 mm; torque share 0.299, clipped steps 0; least stance '
                                 "load 2.5 N; touchdown 5.512 s, 0.052 m/s, 0.2 N; FR's largest "
                                 "force 33.6 N; FR's lowest in the hold 61.6 mm; roll 1.8 deg; "
                                 "other parts on the floor 0.0 N; base's lowest 0.267 m"},
 'the reference run': {'the requirement first issued: period; level [m]; onset [pairs]': '2; '
                                                                                         '0.080; '
                                                                                         '12',
                       'the level in force in periods 0 to 34 [m]': '0.025, 0.025, 0.025, 0.020, '
                                                                    '0.020, 0.020, 0.020, 0.020, '
                                                                    '0.020, 0.020, 0.020, 0.030, '
                                                                    '0.055, 0.060, 0.070, 0.080, '
                                                                    '0.020, 0.020, 0.020, 0.020, '
                                                                    '0.020, 0.020, 0.020, 0.020, '
                                                                    '0.020, 0.020, 0.020, 0.020, '
                                                                    '0.020, 0.020, 0.020, 0.080, '
                                                                    '0.065, 0.040, 0.020',
                       "periods 2 to 15: the committed plan's ambition, level and entry at the onset pair [m]": '2: '
                                                                                                                '1, '
                                                                                                                '0.020, '
                                                                                                                '0.080; '
                                                                                                                '3: '
                                                                                                                '1, '
                                                                                                                '0.020, '
                                                                                                                '0.080; '
                                                                                                                '4: '
                                                                                                                '1, '
                                                                                                                '0.020, '
                                                                                                                '0.080; '
                                                                                                                '5: '
                                                                                                                '1, '
                                                                                                                '0.020, '
                                                                                                                '0.080; '
                                                                                                                '6: '
                                                                                                                '1, '
                                                                                                                '0.020, '
                                                                                                                '0.080; '
                                                                                                                '7: '
                                                                                                                '1, '
                                                                                                                '0.020, '
                                                                                                                '0.080; '
                                                                                                                '8: '
                                                                                                                '0, '
                                                                                                                '0.020, '
                                                                                                                '0.080; '
                                                                                                                '9: '
                                                                                                                '0, '
                                                                                                                '0.020, '
                                                                                                                '0.080; '
                                                                                                                '10: '
                                                                                                                '0.75, '
                                                                                                                '0.030, '
                                                                                                                '0.080; '
                                                                                                                '11: '
                                                                                                                '0, '
                                                                                                                '0.055, '
                                                                                                                '0.080; '
                                                                                                                '12: '
                                                                                                                '0.25, '
                                                                                                                '0.060, '
                                                                                                                '0.080; '
                                                                                                                '13: '
                                                                                                                '0.25, '
                                                                                                                '0.070, '
                                                                                                                '0.080; '
                                                                                                                '14: '
                                                                                                                '0, '
                                                                                                                '0.080, '
                                                                                                                '0.080; '
                                                                                                                '15: '
                                                                                                                '1, '
                                                                                                                '0.020, '
                                                                                                                '0.020',
                       'periods 2 to 15: the best candidate of ambition 1, its level and entry at the onset pair [m]': '2: '
                                                                                                                       '0.020, '
                                                                                                                       '0.080; '
                                                                                                                       '3: '
                                                                                                                       '0.020, '
                                                                                                                       '0.080; '
                                                                                                                       '4: '
                                                                                                                       '0.020, '
                                                                                                                       '0.080; '
                                                                                                                       '5: '
                                                                                                                       '0.020, '
                                                                                                                       '0.080; '
                                                                                                                       '6: '
                                                                                                                       '0.020, '
                                                                                                                       '0.080; '
                                                                                                                       '7: '
                                                                                                                       '0.020, '
                                                                                                                       '0.080; '
                                                                                                                       '8: '
                                                                                                                       '0.020, '
                                                                                                                       '0.075; '
                                                                                                                       '9: '
                                                                                                                       '0.020, '
                                                                                                                       '0.070; '
                                                                                                                       '10: '
                                                                                                                       '0.030, '
                                                                                                                       '0.075; '
                                                                                                                       '11: '
                                                                                                                       '0.055, '
                                                                                                                       '0.070; '
                                                                                                                       '12: '
                                                                                                                       '0.060, '
                                                                                                                       '0.070; '
                                                                                                                       '13: '
                                                                                                                       '0.070, '
                                                                                                                       '0.075; '
                                                                                                                       '14: '
                                                                                                                       '0.065, '
                                                                                                                       '0.065; '
                                                                                                                       '15: '
                                                                                                                       '0.020, '
                                                                                                                       '0.020',
                       "the first period in which the centre of mass moves; its place at the onset's start and where the operation begins [m]; the requirement's place [m]": '9; '
                                                                                                                                                                             '0.0541, '
                                                                                                                                                                             '0.0560; '
                                                                                                                                                                             '0.0555'},
 'the unsuccessful conditions': {"the shove: the alarm [s] and the distance inside the triangle then [mm]; the least distance while FR is off the floor [mm] and when [s]; a stance foot's least load [N] and when [s]; FR's touchdown [s]": '4.6, '
                                                                                                                                                                                                                                             '8.8; '
                                                                                                                                                                                                                                             '0.4, '
                                                                                                                                                                                                                                             '4.761; '
                                                                                                                                                                                                                                             '0.2, '
                                                                                                                                                                                                                                             '4.863; '
                                                                                                                                                                                                                                             '4.817',
                                 'the unannounced run: the level in force during the periods of its overloads [m]': '0.025, '
                                                                                                                    '0.070',
                                 "the operator without the governor, pushed: FR's touchdown [s]; off the floor again from [s]; FR's largest force [N] and when [s]": '4.977; '
                                                                                                                                                                     '5.372; '
                                                                                                                                                                     '113.6, '
                                                                                                                                                                     '5.960'},
 'the lead time': {'announced in period 7, the onset in 15: overloads; the operation begins in period': 'none; '
                                                                                                        '16',
                   'announced in period 8, the onset in 15: overloads; the operation begins in period': 'none; '
                                                                                                        '16',
                   'announced in period 9, the onset in 15: overloads; the operation begins in period': '15; '
                                                                                                        '21'},
 'the checks of Section 10.3, on the reference': {'summary': '12 passed, 0 failed, 0 not '
                                                             'applicable',
                                                  'registration and declarations': 'passed (7 '
                                                                                   'cases)',
                                                  'outputs of the contracted kinds': 'passed (50 '
                                                                                     'cases)',
                                                  'fallback keeps its shifted profile': 'passed '
                                                                                        '(50 '
                                                                                        'cases)',
                                                  'labels at least as informative as the global capability': 'passed '
                                                                                                             '(720 '
                                                                                                             'cases)',
                                                  'the global capability beyond the horizon': 'passed '
                                                                                              '(720 '
                                                                                              'cases)',
                                                  'labelling maps stable within their period': 'passed '
                                                                                               '(720 '
                                                                                               'cases)',
                                                  'inputs within the declared bounds': 'passed (50 '
                                                                                       'cases)',
                                                  'states with the declared components': 'passed '
                                                                                         '(51 '
                                                                                         'cases)',
                                                  'a second run repeats every record': 'passed '
                                                                                       '(292 '
                                                                                       'cases)',
                                                  'executed labels valid': 'passed (50 cases)',
                                                  'true reserve at least the level in force': 'passed '
                                                                                              '(50 '
                                                                                              'cases)',
                                                  'no alarm while every executed label was valid': 'passed '
                                                                                                   '(50 '
                                                                                                   'cases)'},
 'the checks of Section 10.3, on the shove': {'summary': '11 passed, 1 failed, 0 not applicable',
                                              'registration and declarations': 'passed (7 cases)',
                                              'outputs of the contracted kinds': 'passed (50 '
                                                                                 'cases)',
                                              'fallback keeps its shifted profile': 'passed (50 '
                                                                                    'cases)',
                                              'labels at least as informative as the global capability': 'passed '
                                                                                                         '(604 '
                                                                                                         'cases)',
                                              'the global capability beyond the horizon': 'passed '
                                                                                          '(604 '
                                                                                          'cases)',
                                              'labelling maps stable within their period': 'passed '
                                                                                           '(604 '
                                                                                           'cases)',
                                              'inputs within the declared bounds': 'passed (50 '
                                                                                   'cases)',
                                              'states with the declared components': 'passed (51 '
                                                                                     'cases)',
                                              'a second run repeats every record': 'passed (280 '
                                                                                   'cases)',
                                              'executed labels valid': 'failed (43 cases): 2 '
                                                                       'failed; first: period 21: '
                                                                       'label Label(push=10.0)',
                                              'true reserve at least the level in force': 'passed '
                                                                                          '(41 '
                                                                                          'cases)',
                                              'no alarm while every executed label was valid': 'passed '
                                                                                               '(41 '
                                                                                               'cases)'},
 'pushes of 10 N swept over the reference run': {'without a push: the shortfall, f3 and f4 [mm]': '8.3, '
                                                                                                  '4.1',
                                                 "shift's start: the centre of mass's and FR's largest deviations [mm]; the largest shortfall, f3 [mm], its direction [deg]; f4 [mm]; alarms": '4.2, '
                                                                                                                                                                                               '7.8; '
                                                                                                                                                                                               '23.0, '
                                                                                                                                                                                               '0; '
                                                                                                                                                                                               '22.3; '
                                                                                                                                                                                               '0',
                                                 "shift's end: the centre of mass's and FR's largest deviations [mm]; the largest shortfall, f3 [mm], its direction [deg]; f4 [mm]; alarms": '4.4, '
                                                                                                                                                                                             '8.3; '
                                                                                                                                                                                             '24.2, '
                                                                                                                                                                                             '0; '
                                                                                                                                                                                             '26.3; '
                                                                                                                                                                                             '0',
                                                 "unload: the centre of mass's and FR's largest deviations [mm]; the largest shortfall, f3 [mm], its direction [deg]; f4 [mm]; alarms": '5.1, '
                                                                                                                                                                                        '12.1; '
                                                                                                                                                                                        '28.7, '
                                                                                                                                                                                        '0; '
                                                                                                                                                                                        '15.8; '
                                                                                                                                                                                        '0',
                                                 "lift: the centre of mass's and FR's largest deviations [mm]; the largest shortfall, f3 [mm], its direction [deg]; f4 [mm]; alarms": '7.6, '
                                                                                                                                                                                      '15.2; '
                                                                                                                                                                                      '27.1, '
                                                                                                                                                                                      '0; '
                                                                                                                                                                                      '4.2; '
                                                                                                                                                                                      '0',
                                                 "hold: the centre of mass's and FR's largest deviations [mm]; the largest shortfall, f3 [mm], its direction [deg]; f4 [mm]; alarms": '7.2, '
                                                                                                                                                                                      '12.6; '
                                                                                                                                                                                      '27.1, '
                                                                                                                                                                                      '0; '
                                                                                                                                                                                      '4.1; '
                                                                                                                                                                                      '0',
                                                 "lower: the centre of mass's and FR's largest deviations [mm]; the largest shortfall, f3 [mm], its direction [deg]; f4 [mm]; alarms": '7.4, '
                                                                                                                                                                                       '11.9; '
                                                                                                                                                                                       '29.1, '
                                                                                                                                                                                       '0; '
                                                                                                                                                                                       '4.7; '
                                                                                                                                                                                       '0',
                                                 "touchdown: the centre of mass's and FR's largest deviations [mm]; the largest shortfall, f3 [mm], its direction [deg]; f4 [mm]; alarms": '6.3, '
                                                                                                                                                                                           '9.3; '
                                                                                                                                                                                           '37.3, '
                                                                                                                                                                                           '0; '
                                                                                                                                                                                           '4.8; '
                                                                                                                                                                                           '0',
                                                 "reload: the centre of mass's and FR's largest deviations [mm]; the largest shortfall, f3 [mm], its direction [deg]; f4 [mm]; alarms": '4.7, '
                                                                                                                                                                                        '7.8; '
                                                                                                                                                                                        '22.8, '
                                                                                                                                                                                        '0; '
                                                                                                                                                                                        '20.2; '
                                                                                                                                                                                        '0',
                                                 "return: the centre of mass's and FR's largest deviations [mm]; the largest shortfall, f3 [mm], its direction [deg]; f4 [mm]; alarms": '5.2, '
                                                                                                                                                                                        '7.8; '
                                                                                                                                                                                        '24.0, '
                                                                                                                                                                                        '0; '
                                                                                                                                                                                        '23.0; '
                                                                                                                                                                                        '0',
                                                 "over the sweep: the centre of mass's and FR's largest deviations [mm]; the bounds [mm]": '7.6, '
                                                                                                                                           '15.2; '
                                                                                                                                           '8, '
                                                                                                                                           '20',
                                                 "over the sweep: the largest shortfall, f3 and f4 [mm]; the allowance in all at the label's push [mm]; without a push [mm]": '37.3, '
                                                                                                                                                                              '26.3; '
                                                                                                                                                                              '43; '
                                                                                                                                                                              '13',
                                                 'over the sweep: the least distance above the level, three feet and four feet [mm]; runs; alarms': '6.4, '
                                                                                                                                                    '25.1; '
                                                                                                                                                    '72; '
                                                                                                                                                    '0',
                                                 "over the sweep: FR's largest downward speed at its first contact [m/s], the phase and the direction [deg] of the push; with the push during the touchdown, the least and the largest [m/s]; runs without a first contact; runs in which FR leaves the floor again within 0.1 s of it, as in the touchdowns from the hold, with the phase and the direction [deg] of the push and FR's speed at its first contact [m/s]; the profile's commanded descent [m/s]": '0.093, lower, 0; 0.036, 0.083; 0; 1, touchdown 180 at 0.037; 0.05',
                                                 "over the sweep: runs in which FR meets the floor before the slow descent begins, with the phase and the direction [deg] of the push, how long before [s] and FR's height then against its target [mm]; runs in which it meets the floor after the slow descent ends": '1, lower 0, 0.031, -10.2; 0'},
 'every plan in force executed unchanged': {"reference: plans; the least distance above the entries, three feet, four feet with C3 [mm]; above f4's floor [mm]; the deviations, centre of mass and FR [mm]": '49; '
                                                                                                                                                                                                             '35.4, '
                                                                                                                                                                                                             '39.3; '
                                                                                                                                                                                                             '90.4; '
                                                                                                                                                                                                             '3.4, '
                                                                                                                                                                                                             '7.8',
                                            "pushed: plans; the least distance above the entries, three feet, four feet with C3 [mm]; above f4's floor [mm]; the deviations, centre of mass and FR [mm]": '49; '
                                                                                                                                                                                                          '10.5, '
                                                                                                                                                                                                          '17.8; '
                                                                                                                                                                                                          '84.7; '
                                                                                                                                                                                                          '6.9, '
                                                                                                                                                                                                          '13.0'},
 'touchdowns from the hold': {"Menagerie's soft feet, at 0.05 m/s: the contact's speed [m/s]; FR's largest force in 50 ms [N], against its share [N]; the centre of pressure's largest jump in 50 ms [mm], the cosine of its angle to FR's foot; the centre of mass's largest deviation in 0.2 s [mm]; the other feet's least load in 50 ms [N]; FR leaves the floor again in 0.1 s": '0.050; '
                                                                                                                                                                                                                                                                                                                                                                                      '0.2, '
                                                                                                                                                                                                                                                                                                                                                                                      '17.2; '
                                                                                                                                                                                                                                                                                                                                                                                      '0.2, '
                                                                                                                                                                                                                                                                                                                                                                                      '0.92; '
                                                                                                                                                                                                                                                                                                                                                                                      '0.2; '
                                                                                                                                                                                                                                                                                                                                                                                      '38.7; '
                                                                                                                                                                                                                                                                                                                                                                                      'no',
                              "Menagerie's soft feet, at 0.1 m/s: the contact's speed [m/s]; FR's largest force in 50 ms [N], against its share [N]; the centre of pressure's largest jump in 50 ms [mm], the cosine of its angle to FR's foot; the centre of mass's largest deviation in 0.2 s [mm]; the other feet's least load in 50 ms [N]; FR leaves the floor again in 0.1 s": '0.101; '
                                                                                                                                                                                                                                                                                                                                                                                     '0.8, '
                                                                                                                                                                                                                                                                                                                                                                                     '17.2; '
                                                                                                                                                                                                                                                                                                                                                                                     '1.2, '
                                                                                                                                                                                                                                                                                                                                                                                     '0.98; '
                                                                                                                                                                                                                                                                                                                                                                                     '0.2; '
                                                                                                                                                                                                                                                                                                                                                                                     '39.0; '
                                                                                                                                                                                                                                                                                                                                                                                     'no',
                              "Menagerie's soft feet, at 0.2 m/s: the contact's speed [m/s]; FR's largest force in 50 ms [N], against its share [N]; the centre of pressure's largest jump in 50 ms [mm], the cosine of its angle to FR's foot; the centre of mass's largest deviation in 0.2 s [mm]; the other feet's least load in 50 ms [N]; FR leaves the floor again in 0.1 s": '0.201; '
                                                                                                                                                                                                                                                                                                                                                                                     '0.9, '
                                                                                                                                                                                                                                                                                                                                                                                     '17.2; '
                                                                                                                                                                                                                                                                                                                                                                                     '1.8, '
                                                                                                                                                                                                                                                                                                                                                                                     '1.00; '
                                                                                                                                                                                                                                                                                                                                                                                     '0.4; '
                                                                                                                                                                                                                                                                                                                                                                                     '39.1; '
                                                                                                                                                                                                                                                                                                                                                                                     'yes',
                              "Menagerie's soft feet, at 0.4 m/s: the contact's speed [m/s]; FR's largest force in 50 ms [N], against its share [N]; the centre of pressure's largest jump in 50 ms [mm], the cosine of its angle to FR's foot; the centre of mass's largest deviation in 0.2 s [mm]; the other feet's least load in 50 ms [N]; FR leaves the floor again in 0.1 s": '0.389; '
                                                                                                                                                                                                                                                                                                                                                                                     '0.5, '
                                                                                                                                                                                                                                                                                                                                                                                     '17.2; '
                                                                                                                                                                                                                                                                                                                                                                                     '3.9, '
                                                                                                                                                                                                                                                                                                                                                                                     '1.00; '
                                                                                                                                                                                                                                                                                                                                                                                     '0.6; '
                                                                                                                                                                                                                                                                                                                                                                                     '38.8; '
                                                                                                                                                                                                                                                                                                                                                                                     'yes',
                              "Menagerie's soft feet: a loaded foot's sink [mm]": '12.6',
                              "MuJoCo's default impedance, at 0.05 m/s: the contact's speed [m/s]; FR's largest force in 50 ms [N], against its share [N]; the centre of pressure's largest jump in 50 ms [mm], the cosine of its angle to FR's foot; the centre of mass's largest deviation in 0.2 s [mm]; the other feet's least load in 50 ms [N]; FR leaves the floor again in 0.1 s": '0.051; '
                                                                                                                                                                                                                                                                                                                                                                                           '9.5, '
                                                                                                                                                                                                                                                                                                                                                                                           '17.1; '
                                                                                                                                                                                                                                                                                                                                                                                           '9.6, '
                                                                                                                                                                                                                                                                                                                                                                                           '1.00; '
                                                                                                                                                                                                                                                                                                                                                                                           '2.4; '
                                                                                                                                                                                                                                                                                                                                                                                           '37.0; '
                                                                                                                                                                                                                                                                                                                                                                                           'no',
                              "MuJoCo's default impedance, at 0.1 m/s: the contact's speed [m/s]; FR's largest force in 50 ms [N], against its share [N]; the centre of pressure's largest jump in 50 ms [mm], the cosine of its angle to FR's foot; the centre of mass's largest deviation in 0.2 s [mm]; the other feet's least load in 50 ms [N]; FR leaves the floor again in 0.1 s": '0.101; '
                                                                                                                                                                                                                                                                                                                                                                                          '18.0, '
                                                                                                                                                                                                                                                                                                                                                                                          '17.1; '
                                                                                                                                                                                                                                                                                                                                                                                          '23.0, '
                                                                                                                                                                                                                                                                                                                                                                                          '1.00; '
                                                                                                                                                                                                                                                                                                                                                                                          '2.4; '
                                                                                                                                                                                                                                                                                                                                                                                          '37.2; '
                                                                                                                                                                                                                                                                                                                                                                                          'no',
                              "MuJoCo's default impedance, at 0.2 m/s: the contact's speed [m/s]; FR's largest force in 50 ms [N], against its share [N]; the centre of pressure's largest jump in 50 ms [mm], the cosine of its angle to FR's foot; the centre of mass's largest deviation in 0.2 s [mm]; the other feet's least load in 50 ms [N]; FR leaves the floor again in 0.1 s": '0.201; '
                                                                                                                                                                                                                                                                                                                                                                                          '31.7, '
                                                                                                                                                                                                                                                                                                                                                                                          '17.1; '
                                                                                                                                                                                                                                                                                                                                                                                          '45.8, '
                                                                                                                                                                                                                                                                                                                                                                                          '1.00; '
                                                                                                                                                                                                                                                                                                                                                                                          '2.3; '
                                                                                                                                                                                                                                                                                                                                                                                          '37.5; '
                                                                                                                                                                                                                                                                                                                                                                                          'no',
                              "MuJoCo's default impedance, at 0.4 m/s: the contact's speed [m/s]; FR's largest force in 50 ms [N], against its share [N]; the centre of pressure's largest jump in 50 ms [mm], the cosine of its angle to FR's foot; the centre of mass's largest deviation in 0.2 s [mm]; the other feet's least load in 50 ms [N]; FR leaves the floor again in 0.1 s": '0.388; '
                                                                                                                                                                                                                                                                                                                                                                                          '25.1, '
                                                                                                                                                                                                                                                                                                                                                                                          '17.1; '
                                                                                                                                                                                                                                                                                                                                                                                          '40.3, '
                                                                                                                                                                                                                                                                                                                                                                                          '1.00; '
                                                                                                                                                                                                                                                                                                                                                                                          '2.2; '
                                                                                                                                                                                                                                                                                                                                                                                          '36.1; '
                                                                                                                                                                                                                                                                                                                                                                                          'yes',
                              "MuJoCo's default impedance: a loaded foot's sink [mm]": '0.2'},
 'the shutdown': {"the crouch: the centre of pressure's least distance inside the rectangle [mm]; the largest torque share; the knees' angle [rad], against the end of their range [rad]": '145.2; '
                                                                                                                                                                                           '0.180; '
                                                                                                                                                                                           '-2.657, '
                                                                                                                                                                                           '-2.7227',
                  'after the torques are cut: the first part other than a foot on the floor': 'base',
                  "after the torques are cut: the first contact of another part [s after]; the base's height then [m] and downward speed [m/s]; the largest downward speed [m/s]": '0.066; '
                                                                                                                                                                                   '0.1083, '
                                                                                                                                                                                   '0.095; '
                                                                                                                                                                                   '0.238',
                  'after the torques are cut: when the base falls fastest [s after]': '0.114',
                  "at the end: the base's height [m]; its pitch and roll [deg]; the parts on the floor; the load on parts other than the feet [N] of the weight [N]": '0.0771; '
                                                                                                                                                                      '5.97, '
                                                                                                                                                                      '0.01; '
                                                                                                                                                                      'base, '
                                                                                                                                                                      'calf, '
                                                                                                                                                                      'foot, '
                                                                                                                                                                      'thigh; '
                                                                                                                                                                      '102.9, '
                                                                                                                                                                      '149.2',
                  "the centre of mass's motion in the last second [mm]; when it first moves less than 0.01 mm in a second [s after the torques are cut]": '0.0025; '
                                                                                                                                                          '10.0'}}


if __name__ == "__main__":
    main()
