#!/usr/bin/env python3
"""
The heavy-duty mobile manipulator under the governor: a lift and two turns.

Physical AI, Together, Chapter 14 (Section 14.3).

This script
  * writes the record of the experiment;
  * states the limits that the test of hdmm/governed.py uses for the labels
    of the runs: by pose and label, the braking limit at the floor and at the
    lift's requirement, and the speed at which a plan may take a turn of the
    yard (route.py), with what binds; the lift's requirement for the declared
    payload and for the payload that one run misdeclares; and the terms of
    the lead time of Theorem 9.12(b) for the lift;
  * runs the machine of sim.py along the yard under the allovi core with the
    modules of hdmm/governed.py: a reference configuration (the site's
    survey, the lift announced ten periods ahead, the cruise proposal, the
    control package hdmm.route-arm), and configurations that change one
    module each: the fixed predictor, a map without the wet patch, the true
    frictions (an oracle, a diagnostic), the lift not announced, its
    requirement held from the start, a route-aware proposal (a stand-in for
    a learner), a control package whose recovery brakes gently, with the lift
    announced and not, and a payload declared lighter than the one held;
    and the cruise proposal without the governor, at the actuators' limits;
  * for each run: when the machine rests in the drop zone, when the lift
    begins and whether it is overloaded, the speeds in the turns, the
    simulated centre of pressure's least distance inside the contacts against
    the rigid machine's for the planned poses and the true payload, and its
    least distance above the level in force, the shortfalls against the
    allowance, the arm's torques, the base's creep and turn on its contacts,
    the wheels in contact, the arm's lag behind its planned pose, the
    governor's interventions, the labels that the world does not admit, and
    the alarms;
  * runs the checks of Section 10.3 (allovi.checks) on the reference
    configuration;
  * probes the lead time: the lift announced one period before its onset,
    and at its onset;
  * for the map without the wet patch: the share of each axle's load that
    the rigid machine's ground must supply, against the patch's true
    friction, along the executed motion in the second turn and along the
    recoveries that the plans in force there kept; and the worst of those
    recoveries simulated from its start, on the wet patch and on the
    friction that the map claimed;
  * replays the second turn with the reference's and the old map's motions
    and reports the friction that each wheel uses, and whether its contact
    slides;
  * states the funnel's bounds on the arm: how far a change of each joint
    moves each family's distance from its constraint and the two points
    themselves at the poses evaluated, and what the bounds cost a turn
    pair's level;
  * states the norms of the two families and checks them against Lemma 14.1
    and Proposition 14.2 (hdmm/bounds.py): the constants that bound how far
    a change of the state moves each family's point, that every weight is at
    least its constant and every rate measured at the states evaluated below
    it, and what the funnel's spread and the test's sampling can cost a
    pair's least distance, for the labels of the runs, with the room that
    this leaves beyond the level; and compares that room with the shortfalls
    of the simulated machine that Assumption 14.3 bounds;
  * executes every plan in force of five runs unchanged, its head, its
    recovery and its rest, from the simulated state at its period's start:
    the least simulated distance above each pair's and the rest's level,
    the shortfalls, and the arm's deviations from its planned pose;
  * compares the funnel's bounds and the allowances with the largest values
    that the runs with valid labels and those plans measure;
  * reports further details that Section 14.3 quotes: when the reference
    drives off and reaches the top speed, the fixed predictor's braking
    under the plan of period 28, where the route-aware proposal first
    rests, and the misdeclared run's largest deviation of the lift joint
    and its arm at the period boundaries.

Every limit, reserve, distance and level is rounded down, and every share,
shortfall, creep, lag and torque rounded up, at the digits printed.

Usage
    python3 hdmm_governed.py            # writes results/governed-*
    python3 hdmm_governed.py --check    # also checks every value of its
                                        # summary

Outputs, in results/ next to this file
    governed-record.txt     the record of the experiment
    governed-summary.txt    every value, by group; with --check, the
                            comparison with the values expected
    governed-route.csv      the route's points every 0.25 m (s, x, y, m) and
                            whether the ground there is wet (wet) (Figure 14.8)
    governed-trace-*.csv    for each run of the figures, every 20 ms: the
                            time (t, s), the arc length (s_m, m), the speed
                            (v, m/s), the simulated centre of pressure's
                            distance inside the contacts and the rigid
                            machine's (sim, rigid, m), and the level in force
                            (level, m) (Figures 14.11 and 14.12)

Requires MuJoCo 3.16.0 and NumPy (requirements.txt); run with CPython
3.13.17, other versions of Python not tried. About ten minutes on the machine
of the record.
"""
import argparse
import copy
import csv
import math
import os
import platform
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mujoco                                     # noqa: E402
import numpy as np                                # noqa: E402

from hdmm import governed as gv                   # noqa: E402  (puts ../ch10 on the path)
from hdmm import bounds as bd                     # noqa: E402
from hdmm import labels as hl                     # noqa: E402
from hdmm import model, route, sim                # noqa: E402
import allovi                                     # noqa: E402
from allovi import checks as achecks              # noqa: E402

PERIODS = 100                  # 60 s
HELD = 270.0                   # [kg] the payload held, within the declared 243 to 297 kg
MIS = (180.0, 220.0)           # [kg] a declaration of 200 kg +- 10 %, for a payload of
MIS_HELD = 290.0               # 290 kg (ours)
ONSET, ANNOUNCE, POSTPONE = 12, 2, 5      # periods: the lift's onset and announcement, and the
                                          # task plan's postponement after an overload (ours)
TRACE_DT = 0.02                # [s] the figures' traces
BRANCHES = (("reference", 243.0), ("reference", 297.0), ("gentle", 297.0),
            ("route-aware", 297.0), ("fixed", 300.0))   # the payloads at the labels' ends
FIGURE_RUNS = ("reference", "fixed", "old-map", "route-aware", "misdeclared", "ungoverned")
FIRM = hl.Label(hl.DECLARED.payloads, route.SURVEY_FIRM)
WET = hl.Label(hl.DECLARED.payloads, route.SURVEY_WET)
TRUE_WET = hl.Label(hl.DECLARED.payloads, route.WET)

CONFIGS = (   # name: what differs from the reference
    ("reference", "the site's survey, the lift announced, cruise, hdmm.route-arm", {}),
    ("fixed", "the fixed predictor: the global capability", dict(capability="fixed")),
    ("old-map", "a map without the wet patch", dict(capability="old-map")),
    ("oracle", "the true frictions (a diagnostic)", dict(capability="oracle")),
    ("reactive", "the lift not announced", dict(announce=None)),
    ("always", "the lift's requirement from the start", dict(always=True)),
    ("route-aware", "the route-aware proposal", dict(proposal="route-aware")),
    ("gentle", "a recovery braking at 1.5 m/s^2 at most", dict(control="gentle")),
    ("gentle-reactive", "that recovery, the lift not announced",
     dict(control="gentle", announce=None)),
    ("misdeclared", "200 kg +- 10 % declared, 290 kg held",
     dict(payloads=MIS, held=MIS_HELD, believed=0.5 * (MIS[0] + MIS[1]))),
)


def down(x, digits=2):
    q = 10 ** digits
    return f"{math.floor(round(float(x) * q, 6)) / q + 0.0:.{digits}f}"


def up(x, digits=2):
    q = 10 ** digits
    return f"{math.ceil(round(float(x) * q, 6)) / q + 0.0:.{digits}f}"


def label_text(label):
    return f"{label.payloads[0]:g} to {label.payloads[1]:g} kg on {label.friction:g}"


def requirement_of(payloads):
    """The lift's demand for the payloads declared (labels.lift_demand), with
    the package's allowance before a lift's end: the consumption grows by
    LIFT_EXTRA, the requirement is the floor plus it on the levels' grid,
    and a plan that holds the requirement carrying brakes at no more than
    b_req; lift_level is what such a plan keeps along the lift if it begins
    the lift and brakes at b_req, with the package's allowances."""
    label = hl.Label(tuple(payloads), hl.DECLARED.friction)
    d = dict(hl.lift_demand(label=label))
    d["requirement"] = hl.level_up(hl.FLOOR + d["consumed"] + gv.LIFT_EXTRA)
    d["b_req"] = hl.braking_limit(model.POSES["carry"], label, d["requirement"])[0]
    d["lift_level"] = hl.lift_level(d["b_req"], label) - gv.LIFT_EXTRA
    return d


# ----------------------------------------------------------------------------
# Configurations and runs
# ----------------------------------------------------------------------------
def configuration(name, capability="survey", proposal="cruise", control="route-arm",
                  announce=ANNOUNCE, always=False, payloads=hl.DECLARED.payloads, held=HELD,
                  believed=None):
    req = requirement_of(payloads)["requirement"]
    task = gv.Lift(req, announce=announce, onset=ONSET, postpone=POSTPONE, always=always)
    world = gv.Yard(held, believed=believed)
    cap = {"survey": lambda: gv.Survey(payloads), "fixed": gv.Fixed,
           "old-map": lambda: gv.OldMap(payloads), "oracle": lambda: gv.Oracle(payloads)}
    prop = {"cruise": gv.Cruise, "route-aware": gv.RouteAware}
    ctl = {"route-arm": gv.RouteArm, "gentle": gv.RouteArmGentle}
    cfg = allovi.build(name, gv.T, dict(control=ctl[control](), robot=gv.Robot(), world=world,
                                        proposal=prop[proposal](), capability=cap[capability](),
                                        demand=gv.LiftPlan(task), task=task))
    return cfg, task, world


def lift_period(world):
    """The period whose law begins the lift, or None."""
    for k, law in enumerate(world.laws):
        if law.segment.arm == "lift" and law.segment.u0 == 0.0:
            return k
    return None


def arrival(states):
    """When the machine rests in the drop zone with the arm reaching, from
    the states at the period boundaries: the first time, and the time from
    which it rests there until the end of the run; None for one that does
    not occur."""
    at = []
    for x in states:
        c = x.commanded
        ok = False
        if c.v <= 1e-9 and abs(c.a) <= 1e-9 and c.arm == "reach":
            clear = route.S_LINE - (x[0] + hl.front_point((x[3], x[4], x[5])))
            ok = 0.0 <= clear <= route.DROP_ZONE
        at.append(ok)
    first = next((i * gv.T for i, ok in enumerate(at) if ok), None)
    final = None
    if at and at[-1]:
        i = len(at) - 1
        while i > 0 and at[i - 1]:
            i -= 1
        final = i * gv.T
    return first, final


def lifting_mask(log):
    """The steps at which the planned pose is neither carrying nor reaching."""
    out = np.zeros(len(log["t"]), dtype=bool)
    for j, (c, r) in enumerate(zip(model.POSES["carry"], model.POSES["reach"])):
        p = log[f"p{j}"]
        out |= (np.abs(p - c) > 1e-9) & (np.abs(p - r) > 1e-9)
    return out


def arm_deviation(log):
    """The arm's deviation from its planned pose over a log, as the funnel
    bounds it (hdmm/governed.py): with the arm held, the largest of each
    joint [rad]; during a lift, the largest share of the lift's path by
    which the arm lies behind the planned pose and ahead of it, and the
    largest distance of a joint from the path [rad]."""
    lifting = lifting_mask(log)
    q = np.stack([log["q0"], log["q1"], log["q2"]], axis=1)
    p = np.stack([log["p0"], log["p1"], log["p2"]], axis=1)
    held = np.abs(q - p)[~lifting].max(axis=0) if (~lifting).any() else np.zeros(3)
    lift = (0.0, 0.0, 0.0)
    if lifting.any():
        d = gv.PATH
        wq = (q[lifting] - gv.Q_CARRY) @ d / (d @ d)
        wp = (p[lifting] - gv.Q_CARRY) @ d / (d @ d)
        cross = np.max(np.abs(q[lifting] - (gv.Q_CARRY + np.outer(wq, d))), axis=1)
        lift = (max(0.0, float(np.max(wp - wq))), max(0.0, float(np.max(wq - wp))),
                float(np.max(cross)))
    return tuple(float(x) for x in held), lift


def shortfalls(log, simm, rigid, payload, every=5):
    """The simulated machine's shortfall against the rigid one at the
    planned pose, rigid less simulated, with the arm held and during a lift
    [m], and the part of each that remains at the arm's actual pose (the
    rigid machine at the actual pose less the simulated), every 'every'
    steps."""
    lifting = lifting_mask(log)
    short = rigid - simm
    out = {}
    idx = np.arange(0, len(log["t"]), every)
    actual = np.array([hl.motion_margin((float(log["q0"][i]), float(log["q1"][i]),
                                         float(log["q2"][i])), payload, [log["a"][i]],
                                        [log["ay"][i]], [log["w"][i]], [log["alpha"][i]])[0]
                       for i in idx])
    rest = actual - simm[idx]
    for name, mask in (("held", ~lifting), ("lift", lifting)):
        sub = mask[idx]
        out[name] = (float(np.nanmax(short[mask])) if mask.any() else 0.0,
                     float(np.nanmax(rest[sub])) if sub.any() else 0.0)
    return out


def analyse(name, states, world, payload, rec=None, task=None):
    """The metrics of a run (module docstring)."""
    log = world.machine.arrays()
    simm, rigid = sim.margins(log, payload)
    short = rigid - simm
    lifting = lifting_mask(log)
    i = int(np.nanargmin(simm))
    out = dict(name=name)
    out["arrival"] = arrival(states)
    k_lift = lift_period(world)
    out["lift"] = None if k_lift is None else k_lift * gv.T
    lifting_laws = [law.segment for law in world.laws if law.segment.arm == "lift"]
    out["lift_span"] = None if not lifting_laws else (lifting_laws[0].s0,
                                                      lifting_laws[-1].arc(gv.T))
    out["overloads"] = [] if task is None else list(task.overloads)
    out["turns"] = []
    for t in route.TURNS:
        m = (log["s"] >= t.start) & (log["s"] <= t.end)
        out["turns"].append(float(log["v"][m].max()))
    out["least"], out["least_t"], out["least_s"] = float(simm[i]), float(log["t"][i]), \
        float(log["s"][i])
    out["least_rigid"] = float(np.nanmin(rigid))
    out["short_held"] = float(np.nanmax(short[~lifting]))
    out["short_lift"] = float(np.nanmax(short[lifting])) if lifting.any() else 0.0
    over = (lifting & (short > gv.LIFT_ALLOWANCE)) | (~lifting & (short > hl.ALLOWANCE))
    out["over_rigid"] = float(np.nanmin(rigid[over])) if over.any() else None
    out["shares"] = [float(np.max(np.abs(log[f"tau{j}"]))) / hl.ARM_LIMITS[j] for j in range(3)]
    j = int(np.argmax(np.abs(log["tau0"])))
    turn = route.turn_on(float(log["s"][j]), float(log["s"][j]) + 1e-9)
    out["slew_at"] = (float(log["t"][j]), float(log["s"][j]),
                      "the " + turn.name + " turn" if turn is not None else "a straight")
    out["creep"] = (float(np.max(np.abs(log["base_x"]))), float(np.max(np.abs(log["base_y"]))),
                    float(np.max(np.abs(log["yaw"]))))
    out["wheels"] = int(log["wheels"].min())
    out["pitch"] = float(np.max(np.abs(log["pitch"])))
    out["held_dev"], out["lift_dev"] = arm_deviation(log)
    fin = states[-1]
    out["clearance"] = route.S_LINE - (fin[0] + hl.front_point((fin[3], fin[4], fin[5])))
    if rec is not None:
        normal = [p for p in rec.periods if p.governor == "normal"]
        out["interventions"] = [(p.k, p.state.commanded.s) for p in normal
                                if p.decision.ambition < 1.0]
        invalid = [(p.k, p.state.commanded.s) for p in normal
                   if not world.admits(*_head_label(p.plan), p.k)]
        out["invalid"] = invalid
        out["alarms"] = list(rec.alarms)
        gaps = [p.state.margin - p.level for p in normal if p.level != allovi.NEG_INF]
        out["above_level"] = float(min(gaps))
        out["onset_level"] = next((p.level for p in rec.periods if p.k == ONSET), None)
        moving = [p.level for p in normal if p.state.commanded.v > 1e-9]
        out["least_level"] = float(min(moving)) if moving else None
    return out, log, simm, rigid


def _head_label(plan):
    if plan.pairs:
        return plan.pairs[0][1], plan.pairs[0][0]
    return hl.GLOBAL, plan.head()


def trace_rows(log, simm, rigid, world, rec=None):
    """The figures' trace, every TRACE_DT."""
    step = int(round(TRACE_DT / model.TIMESTEP))
    levels = np.full(len(log["t"]), np.nan)
    if rec is not None:
        for p in rec.periods:
            if p.governor == "normal" and p.level != allovi.NEG_INF:
                levels[world.bounds[p.k]:world.bounds[p.k + 1]] = p.level
    rows = []
    for j in range(0, len(log["t"]), step):
        rows.append(dict(t=log["t"][j], s=log["s"][j], v=log["v"][j], sim=simm[j],
                         rigid=rigid[j], level=levels[j]))
    return rows


def run_governed(name, kw):
    cfg, task, world = configuration(name, **kw)
    rec = allovi.run(cfg, PERIODS)
    states = [p.state for p in rec.periods] + [rec.final_state]
    out, log, simm, rigid = analyse(name, states, world, kw.get("held", HELD), rec, task)
    return out, trace_rows(log, simm, rigid, world, rec), rec, world


def run_ungoverned():
    task = gv.Lift(requirement_of(hl.DECLARED.payloads)["requirement"], announce=ANNOUNCE,
                   onset=ONSET, postpone=POSTPONE)
    world = gv.Yard(HELD)
    states, _ = gv.ungoverned(world, gv.Cruise(), task, PERIODS)
    out, log, simm, rigid = analyse("ungoverned", states, world, HELD, task=task)
    # the turns' slew torque and the stop's lift-off
    out["turn_shares"] = []
    for t in route.TURNS:
        m = (log["s"] >= t.start) & (log["s"] <= t.end)
        out["turn_shares"].append(float(np.max(np.abs(log["tau0"][m]))) / hl.ARM_LIMITS[0])
    off = np.nonzero(log["wheels"] < 4)[0]
    out["off"] = (float(log["t"][off[0]]), float(log["t"][off[-1]])) if len(off) else None
    out["off_s"] = float(log["s"][off[0]]) if len(off) else None
    rigid_min_stop = float(np.nanmin(rigid[log["s"] > route.TURNS[-1].end]))
    out["stop_rigid"] = rigid_min_stop
    return out, trace_rows(log, simm, rigid, world)


# ----------------------------------------------------------------------------
# Facts
# ----------------------------------------------------------------------------
def record_facts():
    m = model.build(payload=HELD, pose=None, friction=route.FIRM).compile()
    return {
        "MuJoCo": mujoco.__version__,
        "NumPy, Python": f"{np.__version__}, {platform.python_version()}",
        "the route: straights [m]; the turns' radius [m], sides; the stop line [m]":
            (f"{route.S_TURN1:g}, {route.STRAIGHT_2:g}, {route.STRAIGHT_3:g}; "
             f"{route.RADIUS:g}, left then right; {route.S_LINE:.3f}"),
        "the grounds: firm and wet frictions; the survey's; the wet patch [m]":
            (f"{route.FIRM:g}, {route.WET:g}; {route.SURVEY_FIRM:g}, {route.SURVEY_WET:g}; "
             f"{route.WET_FROM:.3f} to {route.WET_TO:.3f}"),
        "the governor's period [s]; the lift's onset, announcement, postponement [periods]":
            f"{gv.T:g}; {ONSET}, {ANNOUNCE}, {POSTPONE}",
        "the funnel's bounds: the creep [m]; the arm held, slew, lift, tilt [rad]; lifting, "
        "within those of a pose on the path, behind and ahead of the planned one along it "
        "[share of the path]":
            f"{gv.DS:g}; {', '.join(f'{x:g}' for x in gv.DQ)}; {gv.LAG_W:g}, {gv.AHEAD_W:g}",
        "the allowances, the arm held and before a lift's end [m]; the norms' weights for the "
        "arc length [m/m], the speed [s], the acceleration [s^2], the slew, the lift and the "
        "tilt [m/rad], tipping; clearance (the state's norm the larger); the drop zone [m]":
            (f"{hl.ALLOWANCE:g}, {gv.LIFT_ALLOWANCE:g}; {_weights(gv.NORM_TIP)}; "
             f"{_weights(gv.NORM_CLEAR)}; {route.DROP_ZONE:g}"),
        "the generator's weights; the labels' horizon [periods]; the test's lift grid and turn "
        "step [s]":
            (f"{', '.join(f'{x:g}' for x in gv.ALPHAS)}; {gv.HORIZON}; {gv.LIFT_GRID:g}, "
             f"{hl.TURN_DT:g}"),
        "the route-aware proposal: the turns' speeds given [m/s]; its deceleration [m/s^2]; its "
        "lead and rate [periods]; the creep's speed [m/s] and length, and how far inside the "
        "zone [m]":
            _route_aware_text(),
        "the payload held [kg]; the misdeclared run's declaration and payload [kg]":
            f"{HELD:g}; {MIS[0]:g} to {MIS[1]:g}, {MIS_HELD:g}",
        "the simulated contacts: ratio of frictional to normal impedance; time step [s]":
            f"{m.opt.impratio:g}; {m.opt.timestep:g}",
    }


def _weights(w):
    return ", ".join(f"{x:g}" for x in w)


def _route_aware_text():
    p = gv.RouteAware()
    return (f"{', '.join(f'{x:g}' for x in p.turn_speeds)}; {p.decel:g}; {p.lead:g}, {p.rate:g}; "
            f"{p.creep_speed:g}, {p.creep:g}, {p.inside:g}")


def limit_facts():
    lim = gv.LIMITS
    out = {}
    req = requirement_of(hl.DECLARED.payloads)
    mis = requirement_of(MIS)
    ctl = gv.RouteArm()
    for name in ("carry", "reach", "lift"):
        parts = []
        for label in (FIRM, WET, hl.GLOBAL, hl.Label(MIS, route.SURVEY_FIRM)):
            r = mis if label.payloads == MIS else req
            parts.append(f"{label_text(label)}: {down(ctl.braking(name, label, hl.FLOOR))}, "
                         f"{down(ctl.braking(name, label, r['requirement']))}")
        out[f"braking limit, {name}, at the floor and at the lift's requirement for the "
            "label's payloads [m/s^2]"] = "; ".join(parts)
    for name in ("carry", "reach"):
        parts = []
        for label in (FIRM, WET, hl.GLOBAL, hl.Label(FIRM.payloads, route.FIRM),
                      hl.Label(FIRM.payloads, route.WET), hl.Label(MIS, route.SURVEY_FIRM),
                      hl.Label(MIS, route.SURVEY_WET)):
            parts.append(f"{label_text(label)}: {lim.turn_speed(name, label):.2f}, "
                         f"{lim.turn_binds(name, label)}")
        out[f"turn speed, {name} [m/s], what binds"] = "; ".join(parts)
    out["the lift's requirement [m]: declared; misdeclared"] = \
        f"{req['requirement']:.2f}; {mis['requirement']:.2f}"
    # Assumption 4.13 with the package's allowances: a plan that holds the requirement
    # carrying brakes at no more than b_req, and a plan that begins the lift and brakes at
    # b_req keeps, along the lift, the lift's level less what the lift's allowance adds
    for what, r in (("declared", req), ("misdeclared", mis)):
        out[f"the lift, {what}: a plan holding the requirement brakes at most [m/s^2]; "
            "beginning the lift and braking so, it keeps along the lift [m]"] = \
            f"{down(r['b_req'])}; {down(r['lift_level'], 4)}"
    return out


def run_facts(results):
    out = {}
    for r in results:
        n = r["name"]
        first, final = r["arrival"]
        arr = "never" if final is None else f"{final:.1f} s"
        if first is not None and first != final:
            arr += f", first at rest there {first:.1f} s"
        lift = "never" if r["lift"] is None else f"{r['lift']:.1f}"
        over = ", ".join(str(k) for k in r["overloads"]) or "none"
        text = (f"at rest in the drop zone from {arr}; lift from {lift} s, overloads {over}; "
                f"turns {down(r['turns'][0])}, {down(r['turns'][1])} m/s; least distance "
                f"{down(r['least'], 4)} m at {r['least_t']:.2f} s, rigid {down(r['least_rigid'], 4)}"
                f" m; shortfall held {up(1000 * r['short_held'])} mm, lifting "
                f"{up(1000 * r['short_lift'])} mm")
        if r["over_rigid"] is not None:
            text += f", beyond its allowance where the rigid distance is at least " \
                    f"{down(r['over_rigid'], 4)} m"
        hd, ld = r["held_dev"], r["lift_dev"]
        text += (f"; torque shares {up(r['shares'][0], 3)}, {up(r['shares'][1], 3)}, "
                 f"{up(r['shares'][2], 3)}, the slew's at {r['slew_at'][0]:.2f} s, "
                 f"{r['slew_at'][1]:.2f} m, in {r['slew_at'][2]}; creep "
                 f"{up(1000 * r['creep'][0])}, {up(1000 * r['creep'][1])} mm, turn "
                 f"{up(1000 * r['creep'][2], 3)} mrad; wheels {r['wheels']}; the arm held within "
                 f"{up(hd[0], 4)}, {up(hd[1], 4)}, {up(hd[2], 4)} rad, lifting behind "
                 f"{up(ld[0], 4)} and ahead {up(ld[1], 4)} of the path, across {up(ld[2], 4)} rad;"
                 f" clearance at the end {down(r['clearance'], 3)} m")
        if r["lift_span"] is not None:
            text += f"; the lift from {down(r['lift_span'][0])} to {down(r['lift_span'][1])} m"
        if "interventions" in r:
            inv = r["invalid"]
            where = ", ".join(f"{k} ({x:.1f} m)" for k, x in r["interventions"]) or "none"
            text += (f"; interventions in periods {where}; invalid labels "
                     f"{len(inv)}" + (f" from period {inv[0][0]} ({inv[0][1]:.1f} m) to period "
                                       f"{inv[-1][0]} ({inv[-1][1]:.1f} m)" if inv else "")
                     + f"; alarms {len(r['alarms'])}; least distance above the level in force "
                     f"{down(r['above_level'], 4)} m; level at the onset "
                     f"{r['onset_level']:.2f} m; least level while moving "
                     f"{r['least_level']:.2f} m")
        else:
            text += (f"; slew in the turns {up(r['turn_shares'][0], 3)}, "
                     f"{up(r['turn_shares'][1], 3)}; wheels off the ground from "
                     f"{r['off'][0]:.3f} to {r['off'][1]:.3f} s at {r['off_s']:.2f} m; pitch "
                     f"{up(math.degrees(r['pitch']), 3)} deg; the rigid distance at the stop "
                     f"{down(r['stop_rigid'], 4)} m" if r["off"] else "; no wheel off the ground")
        out[n] = text
    return out


def detail_facts(runs):
    """The details of the reference run, and the gentle package's levels,
    that Section 14.3 quotes."""
    rec, world = runs["reference"]
    out = {}
    first = next(p for p in rec.periods if p.requirement.level > hl.FLOOR + 1e-9)
    out["the requirement first issued: period; level [m]; onset [pairs]"] = \
        f"{first.k}; {first.requirement.level:.2f}; {first.requirement.onset}"
    out["the level in force in periods 0 to 20 [m]"] = ", ".join(
        f"{p.level:.2f}" for p in rec.periods[:21])
    committed, offered = [], []
    for p in rec.periods[first.k:first.k + 8]:
        d, n = p.decision, p.requirement.onset
        e = d.entries[d.selected]
        committed.append(f"{p.k}: {e.profile.level:.2f}, {e.profile.at(n):.2f}")
        ones = [x for x in d.entries if x.kind == "fresh" and x.ambition == 1.0]
        if len(ones) > 1:
            offered.append(f"{p.k}: {ones[1].profile.level:.2f}, {ones[1].profile.at(n):.2f}")
    out[f"periods {first.k} to {first.k + 7}: the committed plan's level and its entry at the "
        "onset pair [m]"] = "; ".join(committed)
    out[f"periods {first.k} to {first.k + 7}: the candidate that follows the proposal with its "
        "recovery at the requirement, its level and its entry at the onset pair [m]"] = \
        "; ".join(offered)
    log = world.machine.arrays()
    before = (log["s"] >= route.TURNS[0].start - 10.0) & (log["s"] <= route.TURNS[0].start)
    i = np.nonzero(before)[0][int(np.argmin(log["v"][before]))]
    entries = []
    for t in route.TURNS:
        j = int(np.nonzero(log["s"] >= t.start)[0][0])
        entries.append(log["v"][j])
    out["the least speed before the first turn [m/s] and where [m]; the speeds at the turns' "
        "entries [m/s]"] = (f"{down(log['v'][i])}, {down(log['s'][i])}; "
                            f"{down(entries[0])}, {down(entries[1])}")
    for name in ("reference", "gentle"):
        rec, _ = runs[name]
        cruising, turning = {"carry": set(), "lift": set(), "reach": set()}, set()
        for p in rec.periods:
            c = p.state.commanded
            if p.governor != "normal" or c.v <= 1e-9:
                continue
            if route.turn_on(c.s, c.s + 1e-9) is not None:
                turning.add(round(p.level, 2))
            elif abs(c.v - hl.SPEED) < 1e-9 and abs(c.a) < 1e-9:
                cruising[c.arm].add(round(p.level, 2))
        text = "; ".join(", ".join(f"{x:.2f}" for x in sorted(cruising[a]))
                         for a in ("carry", "lift", "reach"))
        out[f"{name}: the levels in force at period boundaries at the top speed on the "
            "straights, carrying; lifting; reaching [m]"] = text
        out[f"{name}: the levels in force at period boundaries within the turns [m]"] = \
            ", ".join(f"{x:.2f}" for x in sorted(turning))
        low = [(p.k, p.state.commanded.s, p.level) for p in rec.periods
               if p.governor == "normal" and p.state.commanded.v > 1e-9 and p.level < 0.2 - 1e-9]
        if name == "gentle":
            out[f"{name}: the periods in which the level in force while moving is below 0.20 m: "
                "period (arc length [m]), level [m]"] = \
                ", ".join(f"{k} ({x:.1f}) {lv:.2f}" for k, x, lv in low) or "none"
    out["the rest's level carrying and reaching [m]; how far behind the stop line the "
        "front-most point must rest, carrying and reaching, with the funnel's creep and the "
        "worst corner of its bounds on the arm [m]"] = (
        f"{gv.Limits.rest('carry'):.2f}, {gv.Limits.rest('reach'):.2f}; "
        + ", ".join(f"{route.S_LINE - gv.CLEAR_LIMIT + gv.DS + gv.FRONT_BOX[a]:.4f}"
                    for a in ("carry", "reach")))
    return out


def quoted_facts(runs):
    """Further details of the runs that Section 14.3 quotes."""
    out = {}
    _, world = runs["reference"]
    log = world.machine.arrays()
    first = int(np.nonzero(log["v"] > 1e-6)[0][0])
    top = int(np.nonzero(log["v"] >= hl.SPEED - 1e-6)[0][0])
    out["reference: the period in which the machine first moves; when it first reaches the "
        "top speed [s]"] = f"{int(log['t'][first] // gv.T)}; {log['t'][top]:.2f}"
    rec, world = runs["fixed"]
    log = world.machine.arrays()
    lo, hi = world.bounds[29], world.bounds[30]
    picked = next(p for p in rec.periods if p.k == 28).decision.ambition
    out["fixed: the largest deceleration in period 29, under the plan committed in period 28 "
        "[m/s^2]; that plan's ambition"] = f"{down(-float(np.min(log['a'][lo:hi])))}; {picked:g}"
    rec, _ = runs["route-aware"]
    clear = None
    for x in [p.state for p in rec.periods] + [rec.final_state]:
        c = x.commanded
        if c.v <= 1e-9 and abs(c.a) <= 1e-9 and c.arm == "reach":
            gap = route.S_LINE - (x[0] + hl.front_point((x[3], x[4], x[5])))
            if 0.0 <= gap <= route.DROP_ZONE:
                clear = gap
                break
    out["route-aware: at its first rest in the drop zone, the front-most point's distance "
        "from the stop line [m]; how far inside the zone [mm]"] = \
        f"{down(clear, 3)}; {round(1000 * (route.DROP_ZONE - clear)):d}"
    _, world = runs["misdeclared"]
    log = world.machine.arrays()
    held = ~lifting_mask(log)
    dev = np.where(held, np.abs(log["q1"] - log["p1"]), -1.0)
    j = int(np.argmax(dev))
    k_lift = lift_period(world)
    excess = -np.inf
    for b in world.bounds[1:]:
        i = b - 1
        if held[i]:
            excess = max(excess, max(abs(float(log[f"q{n}"][i] - log[f"p{n}"][i])) - gv.DQ[n]
                                     for n in range(3)))
    out["misdeclared: the lift joint's largest deviation with the arm held [rad], when [s];"
        " the lift's end [s]; the largest excess of a joint over its bound at a period "
        "boundary with the arm held [rad]"] = (
        f"{up(float(dev[j]), 4)}, {log['t'][j]:.3f}; {k_lift * gv.T + gv.LIFT_TIME:.1f}; "
        f"{excess:.4f}")
    return out


def replay_wheels(world, payload, believed=None):
    """Replays the executed laws of a governed run on a fresh machine, as the
    world adapter executed them, and reads each wheel's contact force at
    every step in the turns: for each turn, each wheel's largest ratio of
    the tangential to the normal force, the rear axle's as a whole, and the
    steps at which a wheel uses at least 0.995 of the ground's friction
    under it (its contact on the friction cone, sliding), with where."""
    m = sim.Machine(payload, model.POSES["carry"], s0=0.0, believed=believed)
    names = {m.m.geom(f"wheel_{w}").id: w for w in ("fl", "fr", "rl", "rr")}
    f = np.zeros(6)
    out = {t.name: dict(use={w: 0.0 for w in names.values()}, axle=0.0, cone=[])
           for t in route.TURNS}
    for law in world.laws:
        seg = law.segment
        for i in range(int(round(gv.T / m.dt))):
            tau = (i + 1) * m.dt
            s, v, a = seg.arc(tau), seg.speed(tau), seg.accel(tau)
            m._motion(s, v, a)
            mu_f, mu_r = m._frictions(s)
            m.d.ctrl[:] = m.target(np.asarray(seg.pose(tau), dtype=float))
            mujoco.mj_step(m.m, m.d)
            turn = route.turn_on(s, s + 1e-9)
            if turn is None:
                continue
            rec = out[turn.name]
            rear_t, rear_n = np.zeros(2), 0.0
            for c in range(m.d.ncon):
                con = m.d.contact[c]
                g = con.geom1 if con.geom1 in names else con.geom2
                if g not in names:
                    continue
                mujoco.mj_contactForce(m.m, m.d, c, f)
                w = names[g]
                frame = con.frame.reshape(3, 3)
                ft = f[1] * frame[1] + f[2] * frame[2]
                if w[0] == "r":
                    rear_t += ft[:2]
                    rear_n += f[0]
                if f[0] <= 1.0:
                    continue
                ratio = math.hypot(f[1], f[2]) / f[0]
                rec["use"][w] = max(rec["use"][w], ratio)
                if ratio >= 0.995 * (mu_f if w[0] == "f" else mu_r):
                    rec["cone"].append((w, s))
            if rear_n > 1.0:
                rec["axle"] = max(rec["axle"], float(np.hypot(*rear_t)) / rear_n)
    same = np.allclose(m.d.qpos, world.machine.d.qpos, atol=0.0, rtol=0.0)
    return out, same


def wheel_facts(runs):
    """Each wheel's use of the ground's friction in the second turn, along
    the executed motion of the reference run and of the run with the map
    without the wet patch."""
    out = {}
    for name in ("reference", "old-map"):
        rec, world = runs[name]
        res, same = replay_wheels(world, HELD)
        r = res[route.TURNS[1].name]
        use = ", ".join(f"{w} {up(r['use'][w], 3)}" for w in ("fl", "fr", "rl", "rr"))
        if r["cone"]:
            wheels = sorted({w for w, _ in r["cone"]})
            cone = (f"{', '.join(wheels)} for {len(r['cone']) * 0.001:.3f} s from "
                    f"{down(min(x for _, x in r['cone']))} to {up(max(x for _, x in r['cone']))} m")
        else:
            cone = "none"
        out[f"{name}: in the second turn, the wheels' largest ratio of tangential to normal "
            "force; the rear axle's as a whole; on the friction cone (0.995 of the ground's)"] = \
            f"{use}; {up(r['axle'], 3)}; {cone}" + ("" if same else "; THE REPLAY DIFFERS")
    return out


def branch_facts(name, held, kw):
    """Every plan in force of a governed run executed unchanged in MuJoCo from
    its period's start: its head, its whole recovery and three periods of
    its rest, from the simulated state at that period's start (a copy of the
    world). The least simulated distance above each pair's own profile
    entry and above the rest's level; over all the plans, the largest
    shortfall against the rigid machine at the planned pose with the arm
    held and before a lift's end, and the part of it that remains at the
    arm's actual pose; and the arm's deviation from its planned pose."""
    kw = dict(kw, held=held)
    cfg, task, world = configuration(name, **kw)
    rec = allovi.run(cfg, PERIODS)
    ctl = cfg.control
    w2 = gv.Yard(held, believed=kw.get("believed"))
    w2.reset()
    gap, plans = math.inf, 0
    short = {"held": (0.0, 0.0), "lift": (0.0, 0.0)}
    hd, ld = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    for k, law in enumerate(world.laws):
        p = rec.periods[k]
        plan = p.plan
        if p.governor == "normal" and plan is not None and plan.pairs:
            plans += 1
            h = plan.head()
            begin = h.arm == "lift" and h.u0 == 0.0
            initial = gv.State(h.s0, h.v0, h.a0, "carry" if begin else h.arm,
                               None if (begin or h.arm != "lift") else h.u0)
            prof = ctl.profile(plan, initial)
            assert abs(prof.level - p.level) < 1e-9, (k, prof.level, p.level)
            br = copy.deepcopy(w2)
            start = br.machine.steps()
            bounds = [start]
            for seg, _ in plan.pairs:
                br.step(None, gv.Law(seg))
                bounds.append(br.machine.steps())
            rest = gv.Segment(plan.rest.s, 0.0, 0.0, 0.0, plan.rest.arm, None)
            for _ in range(3):
                br.step(None, gv.Law(rest))
            log = br.machine.arrays(start)
            simm, rigid = sim.margins(log, held)
            for n in range(len(plan.pairs)):
                a, b = bounds[n] - start, bounds[n + 1] - start
                gap = min(gap, float(np.min(simm[a:b])) - prof.at(n))
            gap = min(gap, float(np.min(simm[bounds[-1] - start:])) - prof.at(len(plan.pairs)))
            sf = shortfalls(log, simm, rigid, held)
            short = {x: tuple(max(u, v) for u, v in zip(short[x], sf[x])) for x in short}
            h1, l1 = arm_deviation(log)
            hd = tuple(max(u, v) for u, v in zip(hd, h1))
            ld = tuple(max(u, v) for u, v in zip(ld, l1))
        w2.step(None, law)
    return dict(plans=plans, gap=gap, short=short, held_dev=hd, lift_dev=ld)


def box_facts():
    """What the funnel's bounds on the arm can cost, measured at the states
    evaluated: the worst corner of the bounds (DQ), held and about a pose of
    a lift's path, moves the zero-moment point's distance by at most this
    much, on a straight at the accelerations that the plans take and over
    the turns' states, and the front-most point; the largest change, per
    radian of each joint, of the front-most point's distance from the stop
    line and of the zero-moment point's from the rectangle's edge (on a
    straight, and its least over each turn's states); the largest move of
    the zero-moment point itself per radian of each joint, on a straight and
    at the turns' states, the largest change of its distance at a single
    state of a turn, and the slew's sideways move of the point at full reach
    with the heaviest declared payload, against the slew's term in the
    tipping norm; which edge lies nearest the point, and how much farther
    the side edges lie; the front-most point's own move per radian of each
    joint and its part along the route; the largest change per radian of a
    turn pair's level, the least over one period's states, at the turns'
    speeds kept; and the worst corner's cost at a single state of the turns,
    which bounds its cost to the least over any of a turn's states. Returns
    these facts and the rates, which norm_facts checks against Lemma
    14.1's constants."""
    lab = hl.Label(hl.DECLARED.payloads, route.SURVEY_FIRM)
    corners = [np.array((a, b, c)) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]
    straight, turning, across = 0.0, 0.0, 0.0
    held = [model.POSES["carry"], model.POSES["reach"]]
    lifting = [gv.lift_pose(u) for u in np.linspace(0.0, gv.LIFT_TIME, 21)]
    poses = held + lifting
    for q in poses:
        bound = np.array(gv.DQ)
        for p in hl.payloads_of(lab):
            for ax in (-hl.BRAKES, -2.7, -1.0, 0.0, hl.DRIVE):
                base = hl.margin(q, p, ax)[0]
                for c in corners:
                    loss = base - hl.margin(tuple(np.array(q) + bound * c), p, ax)[0]
                    if q in held:
                        straight = max(straight, loss)
                    else:
                        across = max(across, loss)
    for name, speed in (("carry", 2.39), ("reach", 1.96), ("reach", 1.46)):
        q = model.POSES[name]
        for states in (hl.turn_states(route.RADIUS, speed, side=1.0),
                       hl.turn_states(route.RADIUS, speed, hl.TURN_BRAKING, 1.0),
                       hl.turn_states(route.RADIUS, speed, side=1.0, drive=hl.TURN_DRIVE)):
            base = hl.turn_slack(q, lab, *states, level=0.0)["tipping"]
            for c in corners:
                t = hl.turn_slack(tuple(np.array(q) + np.array(gv.DQ) * c), lab, *states,
                                  level=0.0)["tipping"]
                turning = max(turning, base - t)
    h = 1e-3
    per_front, per_zmp = np.zeros(3), np.zeros(3)
    for q in poses:
        for j in range(3):
            for sgn in (-1.0, 1.0):
                dq = np.zeros(3)
                dq[j] = sgn * h
                qq = tuple(np.array(q) + dq)
                per_front[j] = max(per_front[j], abs(hl.front_point(q) - hl.front_point(qq)) / h)
                for p in hl.payloads_of(lab):
                    for ax in (-hl.BRAKES, 0.0, hl.DRIVE):
                        per_zmp[j] = max(per_zmp[j], abs(hl.margin(q, p, ax)[0]
                                                         - hl.margin(qq, p, ax)[0]) / h)
    for name, speed in (("carry", 2.39), ("reach", 1.96)):
        q = model.POSES[name]
        for states in (hl.turn_states(route.RADIUS, speed, side=1.0),
                       hl.turn_states(route.RADIUS, speed, hl.TURN_BRAKING, 1.0)):
            base = hl.turn_slack(q, lab, *states, level=0.0)["tipping"]
            for j in range(3):
                for sgn in (-1.0, 1.0):
                    dq = np.zeros(3)
                    dq[j] = sgn * h
                    t = hl.turn_slack(tuple(np.array(q) + dq), lab, *states, level=0.0)["tipping"]
                    per_zmp[j] = max(per_zmp[j], abs(base - t) / h)
    # the zero-moment point itself, and its distance at a single state of a turn
    def turn_zmps(q, p, states):
        v, ax, k, dk = (np.asarray(x, dtype=float) for x in states)
        ay, w2, alpha = v * v * k, (v * k) ** 2, ax * k + v * v * dk
        b = model.motion_basis(p, q)
        w = (b["q0"] + np.multiply.outer(ax, b["ax"]) + np.multiply.outer(ay, b["ay"])
             + np.multiply.outer(alpha, b["alpha"]) + np.multiply.outer(w2, b["w2"]))
        px, py, _ = model.wrench_zmp(w)
        return np.asarray(px), np.asarray(py)

    def inside(px, py):
        return np.minimum(np.minimum(model.AXLE - px, model.AXLE + px),
                          np.minimum(model.HALF_TRACK - py, model.HALF_TRACK + py))

    point_s, point_t, single_t = np.zeros(3), np.zeros(3), np.zeros(3)
    for q in poses:
        for j in range(3):
            for sgn in (-1.0, 1.0):
                dq = np.zeros(3)
                dq[j] = sgn * h
                qq = tuple(np.array(q) + dq)
                for p in hl.payloads_of(lab):
                    for ax in (-hl.BRAKES, 0.0, hl.DRIVE):
                        a, b = np.array(hl.zmp(q, p, ax)), np.array(hl.zmp(qq, p, ax))
                        point_s[j] = max(point_s[j], float(np.hypot(*(a - b))) / h)
    for name, speed in (("carry", 2.39), ("reach", 1.96)):
        q = model.POSES[name]
        for states in (hl.turn_states(route.RADIUS, speed, side=1.0),
                       hl.turn_states(route.RADIUS, speed, hl.TURN_BRAKING, 1.0)):
            for p in hl.payloads_of(lab):
                x0, y0 = turn_zmps(q, p, states)
                for j in range(3):
                    for sgn in (-1.0, 1.0):
                        dq = np.zeros(3)
                        dq[j] = sgn * h
                        x1, y1 = turn_zmps(tuple(np.array(q) + dq), p, states)
                        point_t[j] = max(point_t[j], float(np.max(np.hypot(x1 - x0, y1 - y0))) / h)
                        single_t[j] = max(single_t[j],
                                          float(np.max(np.abs(inside(x1, y1) - inside(x0, y0)))) / h)
    # which edge lies nearest the zero-moment point at the states that the
    # worst corner evaluates, and how much farther the nearer side edge lies
    nearest, gap_s, gap_t = set(), math.inf, math.inf

    def edges(px, py):
        return {"front": model.AXLE - px, "rear": model.AXLE + px,
                "left": model.HALF_TRACK - py, "right": model.HALF_TRACK + py}

    for q in poses:
        for p in hl.payloads_of(lab):
            for ax in (-hl.BRAKES, -2.7, -1.0, 0.0, hl.DRIVE):
                e = edges(*hl.zmp(q, p, ax))
                nearest.add(min(e, key=e.get))
                gap_s = min(gap_s, min(e["left"], e["right"]) - min(e["front"], e["rear"]))
    for name, speed in (("carry", 2.39), ("reach", 1.96), ("reach", 1.46)):
        q = model.POSES[name]
        for states in (hl.turn_states(route.RADIUS, speed, side=1.0),
                       hl.turn_states(route.RADIUS, speed, hl.TURN_BRAKING, 1.0),
                       hl.turn_states(route.RADIUS, speed, side=1.0, drive=hl.TURN_DRIVE)):
            for p in hl.payloads_of(lab):
                e = edges(*turn_zmps(q, p, states))
                for i in range(len(e["front"])):
                    nearest.add(min(e, key=lambda k: e[k][i]))
                gap_t = min(gap_t, float(np.min(np.minimum(e["left"], e["right"])
                                                - np.minimum(e["front"], e["rear"]))))
    # the front-most point's own move, where the tool's box is the front-most point
    tip_full, tip_x = np.zeros(3), np.zeros(3)
    for q in poses:
        if model.tip_point(*q)[0] + hl.TOOL_HALF <= hl.CHASSIS_FRONT:
            continue
        a = np.asarray(model.tip_point(*q), dtype=float)
        for j in range(3):
            for sgn in (-1.0, 1.0):
                dq = np.zeros(3)
                dq[j] = sgn * h
                b = np.asarray(model.tip_point(*(np.array(q) + dq)), dtype=float)
                tip_full[j] = max(tip_full[j], float(np.linalg.norm(a - b)) / h)
                tip_x[j] = max(tip_x[j], abs(float(a[0] - b[0])) / h)
    # a turn pair's level is the least over its own segment's states: one
    # period's states along a turn, at the turns' speeds held constant
    entry, arc, _ = hl.turn_length(route.RADIUS)
    s_turn = np.arange(-1.5, 2.0 * entry + arc + 1.5, 0.01)
    k_turn, dk_turn = hl.route_curvature(s_turn, route.RADIUS, 1.0)
    window = np.lib.stride_tricks.sliding_window_view
    seg_rate = np.zeros(3)
    for name, speed in (("carry", 2.39), ("reach", 1.96), ("reach", 1.46)):
        q = model.POSES[name]
        v = np.full(len(s_turn), speed)
        a0 = np.zeros(len(s_turn))
        n = int(round(speed * gv.T / 0.01)) + 1
        lows = window(hl.turn_slacks(q, lab, v, a0, k_turn, dk_turn, level=0.0)["tipping"],
                      n).min(axis=1)
        for j in range(3):
            for sgn in (-1.0, 1.0):
                dq = np.zeros(3)
                dq[j] = sgn * h
                t = hl.turn_slacks(tuple(np.array(q) + dq), lab, v, a0, k_turn, dk_turn,
                                   level=0.0)["tipping"]
                seg_rate[j] = max(seg_rate[j],
                                  float(np.max(np.abs(lows - window(t, n).min(axis=1)))) / h)
    # the worst corner at a single state of the turns, which bounds its cost to
    # the least over any of a turn's states, a segment's among them
    single_corner = 0.0
    for name, speed in (("carry", 2.39), ("reach", 1.96), ("reach", 1.46)):
        q = model.POSES[name]
        for states in (hl.turn_states(route.RADIUS, speed, side=1.0),
                       hl.turn_states(route.RADIUS, speed, hl.TURN_BRAKING, 1.0),
                       hl.turn_states(route.RADIUS, speed, side=1.0, drive=hl.TURN_DRIVE)):
            base = hl.turn_slacks(q, lab, *states, level=0.0)["tipping"]
            for c in corners:
                t = hl.turn_slacks(tuple(np.array(q) + np.array(gv.DQ) * c), lab, *states,
                                   level=0.0)["tipping"]
                single_corner = max(single_corner, float(np.max(base - t)))
    heaviest = hl.DECLARED.payloads[1]
    reach = model.POSES["reach"]
    sideways = abs(hl.zmp((reach[0] + gv.DQ[0],) + tuple(reach[1:]), heaviest)[1]
                   - hl.zmp(reach, heaviest)[1])
    facts = {
        "the worst corner of the funnel's bounds on the arm: held, the zero-moment point's "
        "distance on a straight and over the turns' states; about a pose of a lift's path, on a "
        "straight; held, the front-most point carrying and reaching [mm]":
            (f"{up(1000 * straight)}, {up(1000 * turning)}; {up(1000 * across)}; "
             f"{up(1000 * gv.FRONT_BOX['carry'])}, {up(1000 * gv.FRONT_BOX['reach'])}"),
        "the largest change per radian of the slew, the lift and the tilt, at the poses "
        "evaluated, of the front-most point's distance from the stop line; of the zero-moment "
        "point's from the rectangle's edge, on a straight and least over each turn's states "
        "[m/rad]":
            (f"{up(per_front[0], 3)}, {up(per_front[1], 3)}, {up(per_front[2], 3)}; "
             f"{up(per_zmp[0], 3)}, {up(per_zmp[1], 3)}, {up(per_zmp[2], 3)}"),
        "the largest move of the zero-moment point itself per radian of the slew, the lift and "
        "the tilt, on a straight; at the turns' states; the largest change of its distance at a "
        "single state of a turn [m/rad]":
            (f"{up(point_s[0], 3)}, {up(point_s[1], 3)}, {up(point_s[2], 3)}; "
             f"{up(point_t[0], 3)}, {up(point_t[1], 3)}, {up(point_t[2], 3)}; "
             f"{up(single_t[0], 3)}, {up(single_t[1], 3)}, {up(single_t[2], 3)}"),
        "the edge nearest the zero-moment point at every state that the worst corner "
        "evaluates; how much farther the nearer side edge lies, at least, on a straight and "
        "over the turns' states [mm]":
            f"{', '.join(sorted(nearest))}; {down(1000 * gap_s, 1)}, {down(1000 * gap_t, 1)}",
        "the front-most point's own move per radian of the slew, the lift and the tilt, where "
        "the tool's box is the front-most point; its part along the route [m/rad]":
            (f"{up(tip_full[0], 3)}, {up(tip_full[1], 3)}, {up(tip_full[2], 3)}; "
             f"{up(tip_x[0], 3)}, {up(tip_x[1], 3)}, {up(tip_x[2], 3)}"),
        "a turn pair's level, the least over its own segment's states: over one period's "
        "states of a turn at the turns' speeds held constant, its largest change per radian of "
        "the slew, the lift and the tilt [m/rad]":
            f"{up(seg_rate[0], 3)}, {up(seg_rate[1], 3)}, {up(seg_rate[2], 3)}",
        "the worst corner of the funnel's bounds on the arm held at a single state of the turns, "
        "which bounds its cost to the least over any of a turn's states [mm]":
            f"{up(1000 * single_corner)}",
        f"at full reach with {heaviest:g} kg, at rest: the zero-moment point's sideways move "
        "for the funnel's bound on the slew; the slew's term in the tipping norm for that "
        "bound [mm]":
            f"{up(1000 * sideways, 2)}; {up(1000 * gv.NORM_TIP[3] * gv.DQ[0], 2)}",
    }
    rates = dict(clearance=[per_front, tip_full, tip_x],
                 tipping=[per_zmp, point_s, point_t, single_t, seg_rate])
    return facts, rates


RUN_LABELS = (FIRM, WET, hl.GLOBAL, hl.Label(FIRM.payloads, route.FIRM),
              hl.Label(FIRM.payloads, route.WET), hl.Label(MIS, route.SURVEY_FIRM),
              hl.Label(MIS, route.SURVEY_WET))   # every label that the runs issue (limit_facts)


def norm_facts(rates):
    """The norms of the two families against Lemma 14.1 and Proposition 14.2
    (hdmm/bounds.py). Lemma 14.1: the constants that bound how far a change
    of the state moves each family's point, for payloads up to the machine's
    and every state within the levels' top in the tipping norm of one whose
    motion the package allows; that every weight is at least its constant;
    and that every rate measured at the states evaluated, here and in
    box_facts (rates), lies below its constant. Proposition 14.2, for the
    payload intervals of the runs' labels, each with the largest speed at
    which the test lets a plan take the yard's turns under them: what the
    funnel's spread about the states that the test evaluates can cost the
    front edge's distance, held on a straight and in a turn and about a
    lift's path, and what the test's sampling of those states can cost it,
    in a turn, along a lift's path and near its end; that the side and the
    rear edges, less their spread, lie farther than the front edge does at
    the interval's heaviest payload, and so farther than any level; and the
    room beyond the level that this leaves, the arm held and before a lift's
    end. The clearance: how much farther forward than the point that the
    test takes the front-most point can lie within the bounds on the arm,
    and the room beyond the floor. Returns the facts and the two rooms [m]."""
    lim = gv.LIMITS
    out = {}
    out["the zero-moment point of hdmm/bounds.py against MuJoCo's inverse dynamics, over a "
        "grid of poses, payloads and motions: within 1e-9 m"] = \
        "yes" if bd.check_zmp() < 1e-9 else "no"
    out["the levels' grid: the floor, the step and the largest level [m]"] = \
        f"{gv.LEVELS.floor:g}, {gv.LEVELS.step:g}, {gv.LEVELS.top:g}"
    out["the route's turns: the largest curvature [1/m], its largest rate along the route "
        "[1/m^2] and that rate's [1/m^3]"] = f"{up(bd.K1, 3)}, {up(bd.K2, 3)}, {up(bd.K3, 3)}"
    lc = bd.clearance_constants()
    lt = bd.tipping_constants(gv.NORM_TIP, gv.LEVELS.top, gv.DS)
    out["Lemma 14.1(a), the clearance's constants: the arc length [m/m]; the slew, the lift "
        "and the tilt [m/rad]"] = f"{up(lc[0])}; {up(lc[3])}, {up(lc[4])}, {up(lc[5])}"
    out[f"Lemma 14.1(b), the tipping constants, for payloads up to {model.PAYLOAD_MAX:g} kg, "
        f"at a state whose motion the package allows and within {gv.LEVELS.top:g} m of it in "
        "the tipping norm: the arc length [m/m], the speed [s], the acceleration [s^2], the "
        "slew, the lift and the tilt [m/rad]"] = ", ".join(up(x) for x in lt)
    ok = (all(w >= c for w, c in zip(gv.NORM_TIP, lt))
          and all(w >= c for w, c in zip(gv.NORM_CLEAR, lc)))
    out["every weight of the two norms at least its constant"] = "yes" if ok else "no"
    carry, reach = model.POSES["carry"], model.POSES["reach"]
    poses = [("carry", carry), ("reach", reach)] + [
        ("lift", gv.lift_pose(u)) for u in np.linspace(0.0, gv.LIFT_TIME, 21)]
    moves, measured = bd.measured_rates(poses, hl.DECLARED.payloads,
                                        {"carry": (2.39,), "reach": (1.96, 1.46)})
    out["at the states evaluated, per metre of arc length, m/s of speed, m/s^2 of acceleration "
        "and radian of the slew, the lift and the tilt: the largest move of the zero-moment point "
        "along either axis of the base; the largest change of its distance from the rectangle's "
        "edge [m per unit]"] = (", ".join(up(x, 3) for x in moves) + "; "
                                + ", ".join(up(x, 3) for x in measured))
    below = (all(float(np.max(r[j])) <= lt[3 + j] for r in rates["tipping"] for j in range(3))
             and all(x <= c for x, c in zip(moves, lt)) and all(x <= c for x, c in zip(measured, lt))
             and all(float(np.max(r[j])) <= lc[3 + j] for r in rates["clearance"] for j in range(3)))
    out["every rate measured at the states evaluated, here and in the funnel's bounds on the "
        "arm, below its constant"] = "yes" if below else "no"
    groups = {}
    for label in RUN_LABELS:
        g = groups.setdefault(tuple(label.payloads), {"carry": 0.0, "reach": 0.0})
        for name in ("carry", "reach"):
            g[name] = max(g[name], lim.turn_speed(name, label))
    held_cost, lift_cost, far = 0.0, 0.0, True
    for payloads, speeds in groups.items():
        lo, hi = payloads
        on_straight = [bd.spread(model.POSES[n], payloads, gv.DQ, -hl.BRAKES, hl.DRIVE, hl.SPEED,
                                 False, gv.DS) for n in ("carry", "reach")]
        in_turn = [bd.spread(model.POSES[n], payloads, gv.DQ, -gv.TURN_ACCEL, gv.TURN_ACCEL,
                             speeds[n], True, gv.DS) for n in ("carry", "reach")]
        samples = [bd.turn_sampling(model.POSES[n], payloads, speeds[n], gv.T, hl.TURN_DT)
                   for n in ("carry", "reach")]
        lifting = bd.lift_spread(payloads, gv.DQ, gv.DS)
        path = bd.lift_sampling(payloads, gv.LAG_W, gv.LIFT_GRID, gv.LIFT_TIME)
        straight = max(s["front"] for s in on_straight)
        turn = max(s["front"] + t for s, t in zip(in_turn, samples))
        held_cost = max(held_cost, straight, turn)
        lift_cost = max(lift_cost, lifting["front"] + path["between"] + path["near_end"])
        every = on_straight + in_turn + [lifting]
        side = min(s["side_least"] for s in every)
        rear = min(s["rear_least"] for s in every)
        front = max(s["front_most"] for s in every)
        far = far and side >= front and rear >= front
        what = (f"Proposition 14.2, the labels of {lo:g} to {hi:g} kg, turns at up to "
                f"{speeds['carry']:.2f} m/s carrying and {speeds['reach']:.2f} m/s reaching")
        out[f"{what}: the most that the funnel's spread costs the front edge's distance, the arm "
            "held on a straight, in a turn, about a lift's path [mm]; the most that the test's "
            "sampling costs it in a turn, along a lift's path, near its end [mm]"] = (
            f"{up(1000 * straight)}, {up(1000 * max(s['front'] for s in in_turn))}, "
            f"{up(1000 * lifting['front'])}; {up(1000 * max(samples))}, "
            f"{up(1000 * path['between'])}, {up(1000 * path['near_end'])}")
        out[f"{what}: the side and the rear edges' least distance less their spread; the front "
            "edge's largest distance at the heaviest payload [m]"] = \
            f"{down(side, 4)}, {down(rear, 4)}; {up(front, 4)}"
    out["the side and the rear edges farther than the front edge, so than any level, for every "
        "label of the runs"] = "yes" if far else "no"
    end_gap = bd.lift_sampling(hl.DECLARED.payloads, gv.LAG_W, gv.LIFT_GRID, gv.LIFT_TIME)["end_gap"]
    out["along a lift's path: the largest gap between the poses that the test takes; the last "
        "pose before the end, from it [share of the path]"] = \
        f"{up(max(gv.LAG_W, math.pi / (2.0 * gv.LIFT_TIME) * gv.LIFT_GRID), 4)}; {up(end_gap, 4)}"
    room_held, room_lift = hl.ALLOWANCE - held_cost, gv.LIFT_ALLOWANCE - lift_cost
    out["Proposition 14.2: the most that the funnel's spread and the test's sampling cost a "
        "state of a pair's funnel, the arm held and before a lift's end [mm]; the room that this "
        "leaves beyond the level [mm]"] = \
        (f"{up(1000 * held_cost)}, {up(1000 * lift_cost)}; "
         f"{down(1000 * room_held)}, {down(1000 * room_lift)}")
    excess = [bd.clearance_excess(model.POSES[n], gv.DQ) for n in ("carry", "reach")]
    out["the clearance: the most that the front-most point lies beyond the point that the test "
        "takes at the rest, carrying and reaching [mm]; the room beyond the floor [mm]"] = \
        (f"{up(1000 * excess[0])}, {up(1000 * excess[1])}; "
         f"{down(1000 * (hl.ALLOWANCE - max(excess)))}")
    return out, room_held, room_lift


def check_facts(make):
    results = achecks.check(make, PERIODS, fmt=lambda x: f"{x}")
    out = {"summary": achecks.summary(results)}
    for res in results:
        out[res.name] = f"{res.status}" + (f" ({res.cases} cases)" if res.cases else "") + \
            (f": {res.detail}" if res.detail and res.status != achecks.PASSED else "")
    return out


def lead_facts():
    out = {}
    for announce in (ONSET - 1, ONSET):
        cfg, task, world = configuration("lead", announce=announce)
        allovi.run(cfg, ONSET + POSTPONE + 3)
        k = lift_period(world)
        out[f"announced in period {announce}, the onset in {ONSET}: overloads; the lift begins "
            "in period"] = (f"{', '.join(map(str, task.overloads)) or 'none'}; "
                            f"{'never' if k is None else k}")
    return out


def old_map_facts(rec, world):
    """The rigid model's ground in the second turn under the true friction,
    along the executed motion and the recoveries kept, and the worst
    recovery simulated."""
    turn = route.TURNS[1]
    executed, kept, worst = [], [], None
    for p in rec.periods:
        if p.governor != "normal" or not p.plan.pairs:
            continue
        for n, (seg, label) in enumerate(p.plan.pairs):
            if route.turn_on(seg.s0, seg.arc(gv.T)) is not turn:
                continue
            need = TRUE_WET.friction - gv.turn_slack_of(seg, TRUE_WET)["ground"]
            (executed if n == 0 else kept).append(need)
            if n > 0 and (worst is None or need > worst[0]):
                worst = (need, p.k, n)
    out = {"the old map's label in the second turn [friction]; the patch's true friction":
           f"{route.SURVEY_FIRM:g}; {route.WET:g}"}
    out["the largest share of an axle's load that the rigid machine's ground must supply in "
        "the second turn: along the executed motion; along the recoveries kept"] = \
        f"{up(max(executed), 3)}; {up(max(kept), 3)}"
    need, k, n = worst
    plan = next(p.plan for p in rec.periods if p.k == k)
    first = plan.pairs[n][0]
    pre = plan.pairs[n - 1][0] if n > 0 else None
    start = pre if pre is not None else first
    segs = [s for s, _ in plan.pairs[n - 1 if pre is not None else n:]]
    texts = []
    for friction in (None, route.SURVEY_FIRM):
        m = sim.Machine(HELD, model.POSES[start.arm], s0=start.s0, motion0=(start.v0, start.a0),
                        friction=friction)
        for seg in segs:
            m.advance(gv.T, lambda tau, sg=seg: (sg.arc(tau), sg.speed(tau), sg.accel(tau)),
                      seg.pose)
        cx, cy, yaw = m.creep()
        texts.append(f"{'the patch' if friction is None else f'friction {friction:g}'}: "
                     f"{up(1000 * abs(cx))}, {up(1000 * abs(cy))} mm, "
                     f"{up(1000 * abs(yaw), 3)} mrad")
    out[f"the worst recovery kept (period {k}, from pair {n}), simulated from its start: the "
        "base's travel along and across and its turn on its contacts"] = "; ".join(texts)
    return out


def write_record(path, rec, configs):
    ratio, step = (x.strip() for x in rec["the simulated contacts: ratio of frictional to "
                                         "normal impedance; time step [s]"].split(";"))
    funnel = rec["the funnel's bounds: the creep [m]; the arm held, slew, lift, tilt [rad]; "
                 "lifting, within those of a pose on the path, behind and ahead of the planned "
                 "one along it [share of the path]"]
    creep, held, lifting = (x.strip() for x in funnel.split(";"))
    allow = rec["the allowances, the arm held and before a lift's end [m]; the norms' weights "
                "for the arc length [m/m], the speed [s], the acceleration [s^2], the slew, the "
                "lift and the tilt [m/rad], tipping; clearance (the state's norm the larger); the "
                "drop zone [m]"]
    allowances, tip, clear, zone = (x.strip() for x in allow.split(";"))
    lines = [
        "Record of the experiment of Section 14.3: the heavy-duty mobile manipulator governed "
        "through a lift and two turns",
        "",
        "model and asset provenance",
        "  the synthetic mobile manipulator of Section 14.1 (hdmm/model.py), every number ours; "
        "its labels, limits, floors, lift and allowance of 0.005 m with the arm held those of "
        "Section 14.2 (hdmm/labels.py); the yard, its grounds and survey, the task, the modules, "
        "the allowance of 0.01 m before a lift's end and the runs ours (hdmm/route.py, "
        "hdmm/governed.py, hdmm_governed.py); no parameter identified from a machine",
        "simulator and solver",
        f"  MuJoCo {rec['MuJoCo']}; NumPy and Python {rec['NumPy, Python']}; implicitfast, "
        "elliptic cones, MuJoCo's default contact softness, the ratio of frictional to normal "
        "impedance " + ratio + "; each wheel's contact on the friction of the ground under it; "
        "the machine held in the base's frame, which turns with the route, every body feeling "
        "that frame's forces (hdmm/sim.py)",
        "time steps",
        f"  the simulation {1000 * float(step):g} ms; the governor's period "
        + rec["the governor's period [s]; the lift's onset, announcement, postponement [periods]"]
        .split(";")[0] + f" s; {PERIODS} periods, the machine settled 2 s at rest before the "
        "first",
        "actuators and sensors",
        "  those of Section 14.2: brakes 4 and drive 1.5 m/s^2, top speed 3 m/s, each change of "
        "the acceleration along a half-cosine over a period, within +-0.5 m/s^2 in a turn; slew, "
        "lift, tilt 8, 30, 12 kN m, the motion's share counted 1.2 times; position servos of "
        "2e6 N m/rad offset against gravity for the payload believed; the measurement the "
        "simulated state, exact",
        "frames and units",
        "  SI units; the state's arc length along the route of the base's middle; the base's x "
        "forward, y left, z up",
        "uncertainty bounds",
        "  the labels' payload intervals and frictions (hdmm/governed.py's predictors); the "
        f"funnel's bounds: the creep {creep} m; the arm held, slew, lift and tilt within {held} "
        "rad of its planned pose; lifting, within those of a pose on the lift's path at most "
        f"{lifting.split(',')[0].strip()} of the path behind the planned pose and "
        f"{lifting.split(',')[1].strip()} ahead of it; the norms, weighting the arc length "
        "[m/m], the speed [s], the acceleration [s^2], the slew, the lift and the tilt [m/rad] "
        f"by {tip} for the margin against tipping over and {clear} for the clearance, each "
        "weight at least the constant of Lemma 14.1 that bounds how far a change of that "
        "component moves the family's point; the state's norm, of the measurement and the "
        "alarm check, the larger of the two (Section 14.3)",
        "constraints",
        "  the tipping reserve and the clearance at the stop line, floors 0.10 m, the "
        f"allowances {allowances} m with the arm held and before a lift's end, levels on whole "
        f"centimetres; the drop zone {zone} m",
        "terminal behaviour",
        "  the recovery of hdmm/governed.py, then rest on the brakes; after an alarm the rest "
        "feedback at the global capability's limits and a restart at rest",
        "configurations",
    ]
    lines += [f"  {name}: {what}" for name, what, _ in configs]
    lines += [
        "  ungoverned: the cruise proposal at the actuators' limits, without the governor",
        "training and evaluation split",
        "  none: the route-aware proposal is a fixed stand-in for a learner, nothing learns",
        "seeds",
        "  none: no random numbers",
        "hardware",
        f"  {platform.system()} {platform.machine()}, {_hardware()}",
        "guarantee status",
        "  the levels are reserves of the margin against tipping over, and the floor one of the "
        "clearance, in the sense of Definition 4.4, for the rigid machine in its norms "
        "(Proposition 14.2); the funnel's bounds and the allowances' cover of what the rigid "
        "machine leaves out (Assumption 14.3) are assumptions of the package, tested in MuJoCo "
        "in the runs and in every plan executed unchanged, not proved; no physical machine",
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
            w.writerow({k: ("nan" if isinstance(v, float) and math.isnan(v) else f"{v:.6g}")
                        for k, v in row.items()})


def route_rows():
    rows = []
    for s in np.arange(-5.0, route.S_LINE + 5.0001, 0.25):
        x, y, _ = route.pose_at(float(s))
        wet = 1.0 if route.WET_FROM <= s < route.WET_TO else 0.0
        rows.append(dict(s=float(s), x=x, y=y, wet=wet))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Section 14.3: the mobile manipulator governed")
    ap.add_argument("--check", action="store_true", help="check every value of the summary")
    args = ap.parse_args()
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    facts = {"record": record_facts(), "the limits": limit_facts()}
    results, traces, kept = [], {}, {}
    for name, _, kw in CONFIGS:
        r, rows, rec, world = run_governed(name, kw)
        results.append(r)
        traces[name] = rows
        if name in ("reference", "gentle", "old-map", "misdeclared", "fixed", "route-aware"):
            kept[name] = (rec, world)
    r, rows = run_ungoverned()
    results.append(r)
    traces["ungoverned"] = rows
    facts["the runs"] = run_facts(results)
    facts["the lead time"] = lead_facts()
    facts["the reference run"] = detail_facts(kept)
    facts["further details that Section 14.3 quotes"] = quoted_facts(kept)
    facts["the map without the wet patch"] = old_map_facts(*kept["old-map"])
    facts["the checks of Section 10.3, on the reference"] = check_facts(
        lambda: configuration("reference")[0])
    kw = {c[0]: c[2] for c in CONFIGS}
    facts["the checks of Section 10.3, on the map without the wet patch"] = check_facts(
        lambda: configuration("old-map", **kw["old-map"])[0])
    facts["the checks of Section 10.3, on the misdeclared payload"] = check_facts(
        lambda: configuration("misdeclared", **kw["misdeclared"])[0])
    facts["the wheels in the second turn"] = wheel_facts(kept)
    facts["the funnel's bounds on the arm"], rates = box_facts()
    facts["the norms and the reserves"], room_held, room_lift = norm_facts(rates)
    branches = {}
    for name, held_kg in BRANCHES:
        b = branch_facts(name, held_kg, kw[name])
        branches[(name, held_kg)] = b
    facts["every plan in force executed unchanged"] = {
        f"{name}, {held_kg:g} kg: plans; least simulated distance above a pair's or the rest's "
        "level [m]; shortfall held, before a lift's end [mm], of which at the actual pose [mm];"
        " the arm held within [rad]; lifting behind, ahead [share of the path], across [rad]":
            (f"{b['plans']}; {down(b['gap'], 4)}; {up(1000 * b['short']['held'][0])}, "
             f"{up(1000 * b['short']['lift'][0])}; {up(1000 * b['short']['held'][1])}, "
             f"{up(1000 * b['short']['lift'][1])}; "
             + ", ".join(up(x, 4) for x in b["held_dev"]) + "; "
             + ", ".join(up(x, 4) for x in b["lift_dev"]))
        for (name, held_kg), b in branches.items()}
    valid = [r for r in results if r["name"] not in ("misdeclared", "old-map", "ungoverned")]
    allb = list(branches.values())
    worst_held = max([r["short_held"] for r in valid] + [b["short"]["held"][0] for b in allb])
    worst_lift = max([r["short_lift"] for r in valid] + [b["short"]["lift"][0] for b in allb])
    rest_held = max(b["short"]["held"][1] for b in allb)
    rest_lift = max(b["short"]["lift"][1] for b in allb)
    hd = [max(x) for x in zip(*([r["held_dev"] for r in valid] + [b["held_dev"] for b in allb]))]
    ld = [max(x) for x in zip(*([r["lift_dev"] for r in valid] + [b["lift_dev"] for b in allb]))]
    facts["the funnel and the allowances, over the runs with valid labels and every plan "
          "executed unchanged"] = {
        "the base's largest creep [mm]; the bound":
            f"{up(1000 * max(max(r['creep'][:2]) for r in valid))}; {1000 * gv.DS:g}",
        "the arm held, the largest deviation of the slew, the lift and the tilt [rad]; the "
        "bounds": ", ".join(up(x, 4) for x in hd) + "; " + ", ".join(f"{x:g}" for x in gv.DQ),
        "lifting, the largest share of the path behind and ahead, and across it [rad]; the "
        "bounds behind and ahead [share of the path], and those across, the arm's held ones "
        "[rad]": ", ".join(up(x, 4) for x in ld)
                 + f"; {gv.LAG_W:g}, {gv.AHEAD_W:g}; {', '.join(f'{x:g}' for x in gv.DQ)}",
        "the largest shortfall, the arm held and before a lift's end [mm]; the part of it at "
        "the arm's actual pose [mm]; the allowances [mm]":
            (f"{up(1000 * worst_held)}, {up(1000 * worst_lift)}; {up(1000 * rest_held)}, "
             f"{up(1000 * rest_lift)}; {1000 * hl.ALLOWANCE:g}, {1000 * gv.LIFT_ALLOWANCE:g}"),
        "the least distance above the level in force at a period boundary, over the runs [m]":
            down(min(r["above_level"] for r in valid), 4),
        "Assumption 14.3 in the plans executed unchanged: the largest shortfall at the arm's "
        "actual pose, the arm held and before a lift's end [mm]; the room that Proposition 14.2 "
        "leaves [mm]; the shortfalls within it":
            (f"{up(1000 * rest_held)}, {up(1000 * rest_lift)}; {down(1000 * room_held)}, "
             f"{down(1000 * room_lift)}; "
             + ("yes" if rest_held <= room_held and rest_lift <= room_lift else "no")),
    }
    for name in FIGURE_RUNS:
        write_csv(os.path.join(out_dir, f"governed-trace-{name}.csv"), traces[name])
    write_csv(os.path.join(out_dir, "governed-route.csv"), route_rows())
    write_record(os.path.join(out_dir, "governed-record.txt"), facts["record"], CONFIGS)
    lines, failures = [], []
    for group, items in facts.items():
        lines.append(f"[{group}]")
        for key, val in items.items():
            lines.append(f"  {key} = {val}")
            want = EXPECTED.get(group, {}).get(key)
            if args.check and group != "record" and want != str(val):
                failures.append(f"{group}.{key}: expected {want!r}, got {val!r}")
            elif args.check and group == "record" and want is not None and want != str(val):
                failures.append(f"{group}.{key}: expected {want!r}, got {val!r}")
    if args.check:
        failures += [f"{group}.{key}: not computed" for group, items in EXPECTED.items()
                     for key in items if key not in facts.get(group, {})]
        n = sum(len(v) for v in EXPECTED.values())
        lines.append(f"[check] {n} values, among them every value that Section 14.3 quotes: " +
                     ("all as expected" if not failures else "FAILED: " + "; ".join(failures)))
    lines.append(f"[time] {time.time() - t0:.0f} s")
    with open(os.path.join(out_dir, "governed-summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines[:-1]) + "\n")
    print("\n".join(lines))
    if failures:
        raise SystemExit(1)


EXPECTED = {   # every value of the summary but the versions of the libraries
    'record': {
        "the route: straights [m]; the turns' radius [m], sides; the stop line [m]":
            '50, 20, 20; 5, left then right; 111.708',
        "the grounds: firm and wet frictions; the survey's; the wet patch [m]":
            '0.6, 0.35; 0.5, 0.3; 77.854 to 94.708',
        "the governor's period [s]; the lift's onset, announcement, postponement [periods]":
            '0.6; 12, 2, 5',
        "the funnel's bounds: the creep [m]; the arm held, slew, lift, tilt [rad]; lifting, within those of a pose on the path, behind and ahead of the planned one along it [share of the path]":
            '0.001; 0.005, 0.002, 0.002; 0.04, 0.001',
        "the allowances, the arm held and before a lift's end [m]; the norms' weights for the arc length [m/m], the speed [s], the acceleration [s^2], the slew, the lift and the tilt [m/rad], tipping; clearance (the state's norm the larger); the drop zone [m]":
            '0.005, 0.01; 1.5, 1, 0.25, 0.85, 1, 0.4; 1, 1, 0.25, 4.4, 4.4, 2; 0.5',
        "the generator's weights; the labels' horizon [periods]; the test's lift grid and turn step [s]":
            '1, 0.75, 0.5, 0.25; 50; 0.1, 0.01',
        "the route-aware proposal: the turns' speeds given [m/s]; its deceleration [m/s^2]; its lead and rate [periods]; the creep's speed [m/s] and length, and how far inside the zone [m]":
            '1.96, 1.46; 1; 2, 2; 0.25, 0.5, 0.15',
        "the payload held [kg]; the misdeclared run's declaration and payload [kg]":
            '270; 180 to 220, 290',
        'the simulated contacts: ratio of frictional to normal impedance; time step [s]':
            '20; 0.001',
    },
    'the limits': {
        "braking limit, carry, at the floor and at the lift's requirement for the label's payloads [m/s^2]":
            '243 to 297 kg on 0.5: 4.00, 2.59; 243 to 297 kg on 0.3: 2.94, 2.59; 0 to 300 kg on 0.3: 2.94, 2.57; 180 to 220 kg on 0.5: 4.00, 3.41',
        "braking limit, reach, at the floor and at the lift's requirement for the label's payloads [m/s^2]":
            '243 to 297 kg on 0.5: 2.70, 1.35; 243 to 297 kg on 0.3: 2.70, 1.35; 0 to 300 kg on 0.3: 2.67, 1.31; 180 to 220 kg on 0.5: 3.58, 2.33',
        "braking limit, lift, at the floor and at the lift's requirement for the label's payloads [m/s^2]":
            '243 to 297 kg on 0.5: 2.65, 1.29; 243 to 297 kg on 0.3: 2.65, 1.29; 0 to 300 kg on 0.3: 2.61, 1.26; 180 to 220 kg on 0.5: 3.52, 2.27',
        'turn speed, carry [m/s], what binds':
            '243 to 297 kg on 0.5: 2.39, slew; 243 to 297 kg on 0.3: 1.95, ground; 0 to 300 kg on 0.3: 1.95, ground; 243 to 297 kg on 0.6: 2.39, slew; 243 to 297 kg on 0.35: 2.13, ground; 180 to 220 kg on 0.5: 2.56, slew; 180 to 220 kg on 0.3: 2.15, ground',
        'turn speed, reach [m/s], what binds':
            '243 to 297 kg on 0.5: 1.96, ground; 243 to 297 kg on 0.3: 1.46, ground; 0 to 300 kg on 0.3: 1.45, ground; 243 to 297 kg on 0.6: 2.04, slew; 243 to 297 kg on 0.35: 1.61, ground; 180 to 220 kg on 0.5: 2.20, slew; 180 to 220 kg on 0.3: 1.73, ground',
        "the lift's requirement [m]: declared; misdeclared":
            '0.22; 0.21',
        'the lift, declared: a plan holding the requirement brakes at most [m/s^2]; beginning the lift and braking so, it keeps along the lift [m]':
            '2.59; 0.1046',
        'the lift, misdeclared: a plan holding the requirement brakes at most [m/s^2]; beginning the lift and braking so, it keeps along the lift [m]':
            '3.41; 0.1098',
    },
    'the runs': {
        'reference':
            "at rest in the drop zone from 46.2 s; lift from 7.8 s, overloads none; turns 1.96, 1.46 m/s; least distance 0.1314 m at 45.00 s, rigid 0.1316 m; shortfall held 3.34 mm, lifting 5.07 mm; torque shares 0.740, 0.880, 0.757, the slew's at 20.04 s, 51.71 m, in the first turn; creep 0.13, 0.05 mm, turn 0.026 mrad; wheels 4; the arm held within 0.0030, 0.0005, 0.0009 rad, lifting behind 0.0192 and ahead 0.0000 of the path, across 0.0003 rad; clearance at the end 0.354 m; the lift from 17.64 to 30.24 m; interventions in periods 28 (44.6 m), 50 (75.5 m), 73 (101.7 m), 74 (103.5 m), 75 (105.2 m); invalid labels 0; alarms 0; least distance above the level in force 0.0314 m; level at the onset 0.22 m; least level while moving 0.10 m",
        'fixed':
            "at rest in the drop zone from 48.6 s; lift from 7.8 s, overloads none; turns 1.45, 1.45 m/s; least distance 0.1343 m at 33.60 s, rigid 0.1345 m; shortfall held 3.34 mm, lifting 5.07 mm; torque shares 0.402, 0.880, 0.757, the slew's at 20.39 s, 51.71 m, in the first turn; creep 0.12, 0.04 mm, turn 0.019 mrad; wheels 4; the arm held within 0.0017, 0.0005, 0.0009 rad, lifting behind 0.0192 and ahead 0.0000 of the path, across 0.0003 rad; clearance at the end 0.318 m; the lift from 17.64 to 30.24 m; interventions in periods 54 (76.2 m), 76 (101.4 m), 77 (103.2 m), 78 (104.9 m), 79 (105.9 m); invalid labels 0; alarms 0; least distance above the level in force 0.0343 m; level at the onset 0.22 m; least level while moving 0.10 m",
        'old-map':
            "at rest in the drop zone from 44.4 s; lift from 7.8 s, overloads none; turns 1.96, 1.96 m/s; least distance 0.1314 m at 31.20 s, rigid 0.1316 m; shortfall held 3.34 mm, lifting 5.07 mm; torque shares 0.740, 0.880, 0.757, the slew's at 20.04 s, 51.71 m, in the first turn; creep 0.12, 0.05 mm, turn 0.026 mrad; wheels 4; the arm held within 0.0030, 0.0005, 0.0009 rad, lifting behind 0.0192 and ahead 0.0000 of the path, across 0.0003 rad; clearance at the end 0.254 m; the lift from 17.64 to 30.24 m; interventions in periods 28 (44.6 m), 50 (75.5 m), 69 (101.0 m), 70 (102.8 m), 71 (104.5 m), 72 (105.8 m); invalid labels 17 from period 50 (75.5 m) to period 66 (95.6 m); alarms 0; least distance above the level in force 0.0314 m; level at the onset 0.22 m; least level while moving 0.10 m",
        'oracle':
            "at rest in the drop zone from 45.6 s; lift from 7.8 s, overloads none; turns 2.04, 1.61 m/s; least distance 0.1314 m at 18.00 s, rigid 0.1316 m; shortfall held 3.34 mm, lifting 5.07 mm; torque shares 0.810, 0.880, 0.757, the slew's at 20.04 s, 51.74 m, in the first turn; creep 0.12, 0.05 mm, turn 0.027 mrad; wheels 4; the arm held within 0.0033, 0.0005, 0.0009 rad, lifting behind 0.0192 and ahead 0.0000 of the path, across 0.0003 rad; clearance at the end 0.115 m; the lift from 17.64 to 30.24 m; interventions in periods 28 (44.6 m), 71 (101.6 m), 72 (103.4 m), 73 (105.1 m); invalid labels 0; alarms 0; least distance above the level in force 0.0314 m; level at the onset 0.22 m; least level while moving 0.10 m",
        'reactive':
            "at rest in the drop zone from 46.2 s; lift from 10.8 s, overloads 12; turns 1.96, 1.46 m/s; least distance 0.1314 m at 45.00 s, rigid 0.1316 m; shortfall held 3.34 mm, lifting 5.07 mm; torque shares 0.740, 0.880, 0.757, the slew's at 20.04 s, 51.71 m, in the first turn; creep 0.13, 0.05 mm, turn 0.026 mrad; wheels 4; the arm held within 0.0030, 0.0005, 0.0009 rad, lifting behind 0.0192 and ahead 0.0000 of the path, across 0.0003 rad; clearance at the end 0.354 m; the lift from 26.64 to 39.24 m; interventions in periods 28 (44.6 m), 50 (75.5 m), 73 (101.7 m), 74 (103.5 m), 75 (105.2 m); invalid labels 0; alarms 0; least distance above the level in force 0.0314 m; level at the onset 0.10 m; least level while moving 0.10 m",
        'always':
            "at rest in the drop zone from 46.2 s; lift from 7.8 s, overloads none; turns 1.96, 1.46 m/s; least distance 0.1314 m at 45.00 s, rigid 0.1316 m; shortfall held 3.34 mm, lifting 5.07 mm; torque shares 0.740, 0.880, 0.757, the slew's at 20.04 s, 51.71 m, in the first turn; creep 0.13, 0.05 mm, turn 0.026 mrad; wheels 4; the arm held within 0.0030, 0.0005, 0.0009 rad, lifting behind 0.0192 and ahead 0.0000 of the path, across 0.0003 rad; clearance at the end 0.354 m; the lift from 17.64 to 30.24 m; interventions in periods 28 (44.6 m), 50 (75.5 m), 73 (101.7 m), 74 (103.5 m), 75 (105.2 m); invalid labels 0; alarms 0; least distance above the level in force 0.0314 m; level at the onset 0.22 m; least level while moving 0.10 m",
        'route-aware':
            "at rest in the drop zone from 51.0 s, first at rest there 48.0 s; lift from 7.8 s, overloads none; turns 1.96, 1.46 m/s; least distance 0.2488 m at 47.40 s, rigid 0.2488 m; shortfall held 3.34 mm, lifting 5.07 mm; torque shares 0.741, 0.880, 0.757, the slew's at 20.19 s, 51.70 m, in the first turn; creep 0.08, 0.05 mm, turn 0.026 mrad; wheels 4; the arm held within 0.0030, 0.0005, 0.0009 rad, lifting behind 0.0192 and ahead 0.0000 of the path, across 0.0003 rad; clearance at the end 0.197 m; the lift from 17.64 to 30.24 m; interventions in periods none; invalid labels 0; alarms 0; least distance above the level in force 0.0335 m; level at the onset 0.22 m; least level while moving 0.10 m",
        'gentle':
            "at rest in the drop zone from 46.8 s; lift from 7.8 s, overloads none; turns 1.96, 1.46 m/s; least distance 0.1439 m at 30.60 s, rigid 0.1441 m; shortfall held 3.34 mm, lifting 5.07 mm; torque shares 0.741, 0.880, 0.757, the slew's at 19.75 s, 51.65 m, in the first turn; creep 0.09, 0.05 mm, turn 0.026 mrad; wheels 4; the arm held within 0.0030, 0.0005, 0.0009 rad, lifting behind 0.0192 and ahead 0.0000 of the path, across 0.0003 rad; clearance at the end 0.178 m; the lift from 17.64 to 30.24 m; interventions in periods 28 (44.6 m), 29 (46.4 m), 72 (100.2 m), 73 (102.0 m), 74 (103.7 m), 75 (105.1 m), 76 (105.9 m); invalid labels 0; alarms 0; least distance above the level in force 0.0332 m; level at the onset 0.31 m; least level while moving 0.11 m",
        'gentle-reactive':
            "at rest in the drop zone from 46.8 s; lift from 7.8 s, overloads none; turns 1.96, 1.46 m/s; least distance 0.1439 m at 30.60 s, rigid 0.1441 m; shortfall held 3.34 mm, lifting 5.07 mm; torque shares 0.741, 0.880, 0.757, the slew's at 19.75 s, 51.65 m, in the first turn; creep 0.09, 0.05 mm, turn 0.026 mrad; wheels 4; the arm held within 0.0030, 0.0005, 0.0009 rad, lifting behind 0.0192 and ahead 0.0000 of the path, across 0.0003 rad; clearance at the end 0.178 m; the lift from 17.64 to 30.24 m; interventions in periods 28 (44.6 m), 29 (46.4 m), 72 (100.2 m), 73 (102.0 m), 74 (103.7 m), 75 (105.1 m), 76 (105.9 m); invalid labels 0; alarms 0; least distance above the level in force 0.0332 m; level at the onset 0.31 m; least level while moving 0.11 m",
        'misdeclared':
            "at rest in the drop zone from 44.4 s; lift from 7.8 s, overloads none; turns 2.20, 1.73 m/s; least distance 0.0346 m at 43.20 s, rigid 0.0343 m; shortfall held 3.58 mm, lifting 5.15 mm; torque shares 0.941, 0.909, 0.791, the slew's at 20.22 s, 51.79 m, in the first turn; creep 0.13, 0.06 mm, turn 0.029 mrad; wheels 4; the arm held within 0.0038, 0.0021, 0.0017 rad, lifting behind 0.0194 and ahead 0.0000 of the path, across 0.0020 rad; clearance at the end 0.162 m; the lift from 17.64 to 30.24 m; interventions in periods 28 (44.6 m), 70 (102.2 m), 71 (104.0 m), 72 (105.6 m); invalid labels 99 from period 1 (0.0 m) to period 99 (106.3 m); alarms 0; least distance above the level in force -0.0654 m; level at the onset 0.21 m; least level while moving 0.10 m",
        'ungoverned':
            "at rest in the drop zone from never; lift from 7.8 s, overloads none; turns 3.00, 3.00 m/s; least distance 0.0000 m at 37.80 s, rigid 0.0178 m; shortfall held 17.88 mm, lifting 5.07 mm, beyond its allowance where the rigid distance is at least 0.0178 m; torque shares 1.756, 0.880, 0.757, the slew's at 18.56 s, 51.71 m, in the first turn; creep 0.14, 0.09 mm, turn 0.041 mrad; wheels 2; the arm held within 0.0070, 0.0005, 0.0009 rad, lifting behind 0.0192 and ahead 0.0000 of the path, across 0.0003 rad; clearance at the end -3.293 m; the lift from 19.44 to 32.04 m; slew in the turns 1.756, 1.756; wheels off the ground from 37.802 to 37.802 s at 109.23 m; pitch 0.007 deg; the rigid distance at the stop 0.0178 m",
    },
    'the lead time': {
        'announced in period 11, the onset in 12: overloads; the lift begins in period':
            'none; 13',
        'announced in period 12, the onset in 12: overloads; the lift begins in period':
            '12; 18',
    },
    'the reference run': {
        'the requirement first issued: period; level [m]; onset [pairs]':
            '2; 0.22; 9',
        'the level in force in periods 0 to 20 [m]':
            '0.43, 0.31, 0.18, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.22, 0.22, 0.22, 0.14, 0.11, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10',
        "periods 2 to 9: the committed plan's level and its entry at the onset pair [m]":
            '2: 0.10, 0.43; 3: 0.10, 0.43; 4: 0.10, 0.43; 5: 0.10, 0.43; 6: 0.10, 0.43; 7: 0.10, 0.43; 8: 0.10, 0.35; 9: 0.22, 0.22',
        'periods 2 to 9: the candidate that follows the proposal with its recovery at the requirement, its level and its entry at the onset pair [m]':
            '2: 0.22, 0.43; 3: 0.22, 0.43; 4: 0.22, 0.43; 5: 0.22, 0.43; 6: 0.22, 0.43; 7: 0.22, 0.43; 8: 0.22, 0.23; 9: 0.22, 0.22',
        "the least speed before the first turn [m/s] and where [m]; the speeds at the turns' entries [m/s]":
            '1.47, 48.86; 1.79, 1.46',
        'reference: the levels in force at period boundaries at the top speed on the straights, carrying; lifting; reaching [m]':
            '0.10, 0.14, 0.22; 0.10, 0.11; 0.10',
        'reference: the levels in force at period boundaries within the turns [m]':
            '0.10, 0.12, 0.28, 0.29',
        'gentle: the levels in force at period boundaries at the top speed on the straights, carrying; lifting; reaching [m]':
            '0.20, 0.31; 0.20; 0.11, 0.20',
        'gentle: the levels in force at period boundaries within the turns [m]':
            '0.20, 0.28, 0.29',
        'gentle: the periods in which the level in force while moving is below 0.20 m: period (arc length [m]), level [m]':
            '50 (76.7) 0.11, 51 (78.3) 0.11',
        "the rest's level carrying and reaching [m]; how far behind the stop line the front-most point must rest, carrying and reaching, with the funnel's creep and the worst corner of its bounds on the arm [m]":
            '0.43, 0.33; 0.1104, 0.1060',
    },
    'further details that Section 14.3 quotes': {
        'reference: the period in which the machine first moves; when it first reaches the top speed [s]':
            '1; 3.59',
        "fixed: the largest deceleration in period 29, under the plan committed in period 28 [m/s^2]; that plan's ambition":
            '2.58; 1',
        "route-aware: at its first rest in the drop zone, the front-most point's distance from the stop line [m]; how far inside the zone [mm]":
            '0.497; 3',
        "misdeclared: the lift joint's largest deviation with the arm held [rad], when [s]; the lift's end [s]; the largest excess of a joint over its bound at a period boundary with the arm held [rad]":
            '0.0021, 12.112; 11.8; -0.0001',
    },
    'the map without the wet patch': {
        "the old map's label in the second turn [friction]; the patch's true friction":
            '0.5; 0.35',
        "the largest share of an axle's load that the rigid machine's ground must supply in the second turn: along the executed motion; along the recoveries kept":
            '0.425; 0.496',
        "the worst recovery kept (period 60, from pair 1), simulated from its start: the base's travel along and across and its turn on its contacts":
            'the patch: 0.04, 0.01 mm, 0.008 mrad; friction 0.5: 0.04, 0.01 mm, 0.007 mrad',
    },
    'the checks of Section 10.3, on the reference': {
        'summary':
            '12 passed, 0 failed, 0 not applicable',
        'registration and declarations':
            'passed (7 cases)',
        'outputs of the contracted kinds':
            'passed (100 cases)',
        'fallback keeps its shifted profile':
            'passed (100 cases)',
        'labels at least as informative as the global capability':
            'passed (3415 cases)',
        'the global capability beyond the horizon':
            'passed (3415 cases)',
        'labelling maps stable within their period':
            'passed (3415 cases)',
        'inputs within the declared bounds':
            'passed (100 cases)',
        'states with the declared components':
            'passed (101 cases)',
        'a second run repeats every record':
            'passed (664 cases)',
        'executed labels valid':
            'passed (100 cases)',
        'true reserve at least the level in force':
            'passed (100 cases)',
        'no alarm while every executed label was valid':
            'passed (100 cases)',
    },
    'the checks of Section 10.3, on the map without the wet patch': {
        'summary':
            '11 passed, 1 failed, 0 not applicable',
        'registration and declarations':
            'passed (7 cases)',
        'outputs of the contracted kinds':
            'passed (100 cases)',
        'fallback keeps its shifted profile':
            'passed (100 cases)',
        'labels at least as informative as the global capability':
            'passed (3410 cases)',
        'the global capability beyond the horizon':
            'passed (3410 cases)',
        'labelling maps stable within their period':
            'passed (3410 cases)',
        'inputs within the declared bounds':
            'passed (100 cases)',
        'states with the declared components':
            'passed (101 cases)',
        'a second run repeats every record':
            'passed (654 cases)',
        'executed labels valid':
            'failed (100 cases): 17 failed; first: period 50: label Label(payloads=(243.0, 297.0), friction=0.5)',
        'true reserve at least the level in force':
            'passed (50 cases)',
        'no alarm while every executed label was valid':
            'passed (50 cases)',
    },
    'the checks of Section 10.3, on the misdeclared payload': {
        'summary':
            '11 passed, 1 failed, 0 not applicable',
        'registration and declarations':
            'passed (7 cases)',
        'outputs of the contracted kinds':
            'passed (100 cases)',
        'fallback keeps its shifted profile':
            'passed (100 cases)',
        'labels at least as informative as the global capability':
            'passed (3242 cases)',
        'the global capability beyond the horizon':
            'passed (3242 cases)',
        'labelling maps stable within their period':
            'passed (3242 cases)',
        'inputs within the declared bounds':
            'passed (100 cases)',
        'states with the declared components':
            'passed (101 cases)',
        'a second run repeats every record':
            'passed (657 cases)',
        'executed labels valid':
            'failed (100 cases): 99 failed; first: period 1: label Label(payloads=(180.0, 220.0), friction=0.5)',
        'true reserve at least the level in force':
            'passed (1 cases)',
        'no alarm while every executed label was valid':
            'passed (1 cases)',
    },
    'the wheels in the second turn': {
        "reference: in the second turn, the wheels' largest ratio of tangential to normal force; the rear axle's as a whole; on the friction cone (0.995 of the ground's)":
            'fl 0.067, fr 0.073, rl 0.174, rr 0.179; 0.147; none',
        "old-map: in the second turn, the wheels' largest ratio of tangential to normal force; the rear axle's as a whole; on the friction cone (0.995 of the ground's)":
            'fl 0.118, fr 0.137, rl 0.319, rr 0.350; 0.267; rr for 0.432 s from 82.19 to 83.04 m',
    },
    "the funnel's bounds on the arm": {
        "the worst corner of the funnel's bounds on the arm: held, the zero-moment point's distance on a straight and over the turns' states; about a pose of a lift's path, on a straight; held, the front-most point carrying and reaching [mm]":
            '0.93, 0.59; 0.93; 4.43, 0.00',
        "the largest change per radian of the slew, the lift and the tilt, at the poses evaluated, of the front-most point's distance from the stop line; of the zero-moment point's from the rectangle's edge, on a straight and least over each turn's states [m/rad]":
            '0.003, 0.714, 1.533; 0.009, 0.268, 0.226',
        "the largest move of the zero-moment point itself per radian of the slew, the lift and the tilt, on a straight; at the turns' states; the largest change of its distance at a single state of a turn [m/rad]":
            '0.628, 0.268, 0.226; 0.641, 0.157, 0.190; 0.036, 0.094, 0.190',
        "the edge nearest the zero-moment point at every state that the worst corner evaluates; how much farther the nearer side edge lies, at least, on a straight and over the turns' states [mm]":
            'front; 231.0, 236.4',
        "the front-most point's own move per radian of the slew, the lift and the tilt, where the tool's box is the front-most point; its part along the route [m/rad]":
            '4.400, 4.400, 2.000; 0.003, 0.714, 1.533',
        "a turn pair's level, the least over its own segment's states: over one period's states of a turn at the turns' speeds held constant, its largest change per radian of the slew, the lift and the tilt [m/rad]":
            '0.034, 0.073, 0.184',
        "the worst corner of the funnel's bounds on the arm held at a single state of the turns, which bounds its cost to the least over any of a turn's states [mm]":
            '0.66',
        "at full reach with 297 kg, at rest: the zero-moment point's sideways move for the funnel's bound on the slew; the slew's term in the tipping norm for that bound [mm]":
            '3.14; 4.25',
    },
    'the norms and the reserves': {
        "the zero-moment point of hdmm/bounds.py against MuJoCo's inverse dynamics, over a grid of poses, payloads and motions: within 1e-9 m":
            'yes',
        "the levels' grid: the floor, the step and the largest level [m]":
            '0.1, 0.01, 1',
        "the route's turns: the largest curvature [1/m], its largest rate along the route [1/m^2] and that rate's [1/m^3]":
            '0.200, 0.105, 0.110',
        "Lemma 14.1(a), the clearance's constants: the arc length [m/m]; the slew, the lift and the tilt [m/rad]":
            '1.00; 4.40, 4.40, 2.00',
        'Lemma 14.1(b), the tipping constants, for payloads up to 300 kg, at a state whose motion the package allows and within 1 m of it in the tipping norm: the arc length [m/m], the speed [s], the acceleration [s^2], the slew, the lift and the tilt [m/rad]':
            '1.46, 0.67, 0.22, 0.83, 0.97, 0.36',
        'every weight of the two norms at least its constant':
            'yes',
        "at the states evaluated, per metre of arc length, m/s of speed, m/s^2 of acceleration and radian of the slew, the lift and the tilt: the largest move of the zero-moment point along either axis of the base; the largest change of its distance from the rectangle's edge [m per unit]":
            '0.069, 0.094, 0.089, 0.641, 0.268, 0.226; 0.011, 0.016, 0.089, 0.038, 0.268, 0.226',
        "every rate measured at the states evaluated, here and in the funnel's bounds on the arm, below its constant":
            'yes',
        "Proposition 14.2, the labels of 243 to 297 kg, turns at up to 2.39 m/s carrying and 2.04 m/s reaching: the most that the funnel's spread costs the front edge's distance, the arm held on a straight, in a turn, about a lift's path [mm]; the most that the test's sampling costs it in a turn, along a lift's path, near its end [mm]":
            '0.97, 0.83, 0.97; 0.02, 0.07, 0.04',
        "Proposition 14.2, the labels of 243 to 297 kg, turns at up to 2.39 m/s carrying and 2.04 m/s reaching: the side and the rear edges' least distance less their spread; the front edge's largest distance at the heaviest payload [m]":
            '0.7020, 1.5800; 0.5757',
        "Proposition 14.2, the labels of 0 to 300 kg, turns at up to 1.95 m/s carrying and 1.45 m/s reaching: the most that the funnel's spread costs the front edge's distance, the arm held on a straight, in a turn, about a lift's path [mm]; the most that the test's sampling costs it in a turn, along a lift's path, near its end [mm]":
            '0.97, 0.73, 0.98; 0.02, 0.07, 0.04',
        "Proposition 14.2, the labels of 0 to 300 kg, turns at up to 1.95 m/s carrying and 1.45 m/s reaching: the side and the rear edges' least distance less their spread; the front edge's largest distance at the heaviest payload [m]":
            '0.7482, 1.3710; 0.5733',
        "Proposition 14.2, the labels of 180 to 220 kg, turns at up to 2.56 m/s carrying and 2.20 m/s reaching: the most that the funnel's spread costs the front edge's distance, the arm held on a straight, in a turn, about a lift's path [mm]; the most that the test's sampling costs it in a turn, along a lift's path, near its end [mm]":
            '0.84, 0.77, 0.84; 0.02, 0.07, 0.03',
        "Proposition 14.2, the labels of 180 to 220 kg, turns at up to 2.56 m/s carrying and 2.20 m/s reaching: the side and the rear edges' least distance less their spread; the front edge's largest distance at the heaviest payload [m]":
            '0.6838, 1.5282; 0.6378',
        'the side and the rear edges farther than the front edge, so than any level, for every label of the runs':
            'yes',
        "along a lift's path: the largest gap between the poses that the test takes; the last pose before the end, from it [share of the path]":
            '0.0400; 0.0016',
        "Proposition 14.2: the most that the funnel's spread and the test's sampling cost a state of a pair's funnel, the arm held and before a lift's end [mm]; the room that this leaves beyond the level [mm]":
            '0.97, 1.07; 4.03, 8.93',
        'the clearance: the most that the front-most point lies beyond the point that the test takes at the rest, carrying and reaching [mm]; the room beyond the floor [mm]':
            '0.05, 0.00; 4.95',
    },
    'every plan in force executed unchanged': {
        "reference, 243 kg: plans; least simulated distance above a pair's or the rest's level [m]; shortfall held, before a lift's end [mm], of which at the actual pose [mm]; the arm held within [rad]; lifting behind, ahead [share of the path], across [rad]":
            '99; 0.0471; 3.02, 5.32; 3.02, 7.26; 0.0028, 0.0005, 0.0014; 0.0192, 0.0000, 0.0003',
        "reference, 297 kg: plans; least simulated distance above a pair's or the rest's level [m]; shortfall held, before a lift's end [mm], of which at the actual pose [mm]; the arm held within [rad]; lifting behind, ahead [share of the path], across [rad]":
            '99; 0.0043; 3.67, 5.99; 3.67, 8.13; 0.0031, 0.0005, 0.0015; 0.0192, 0.0001, 0.0003',
        "gentle, 297 kg: plans; least simulated distance above a pair's or the rest's level [m]; shortfall held, before a lift's end [mm], of which at the actual pose [mm]; the arm held within [rad]; lifting behind, ahead [share of the path], across [rad]":
            '99; 0.0059; 3.67, 5.67; 3.67, 7.81; 0.0032, 0.0005, 0.0008; 0.0192, 0.0001, 0.0003',
        "route-aware, 297 kg: plans; least simulated distance above a pair's or the rest's level [m]; shortfall held, before a lift's end [mm], of which at the actual pose [mm]; the arm held within [rad]; lifting behind, ahead [share of the path], across [rad]":
            '99; 0.0043; 3.67, 5.99; 3.67, 8.13; 0.0031, 0.0005, 0.0015; 0.0192, 0.0001, 0.0003',
        "fixed, 300 kg: plans; least simulated distance above a pair's or the rest's level [m]; shortfall held, before a lift's end [mm], of which at the actual pose [mm]; the arm held within [rad]; lifting behind, ahead [share of the path], across [rad]":
            '99; 0.0043; 3.70, 6.02; 3.70, 8.16; 0.0017, 0.0005, 0.0012; 0.0192, 0.0001, 0.0003',
    },
    'the funnel and the allowances, over the runs with valid labels and every plan executed unchanged': {
        "the base's largest creep [mm]; the bound":
            '0.13; 1',
        'the arm held, the largest deviation of the slew, the lift and the tilt [rad]; the bounds':
            '0.0033, 0.0005, 0.0015; 0.005, 0.002, 0.002',
        "lifting, the largest share of the path behind and ahead, and across it [rad]; the bounds behind and ahead [share of the path], and those across, the arm's held ones [rad]":
            '0.0192, 0.0001, 0.0003; 0.04, 0.001; 0.005, 0.002, 0.002',
        "the largest shortfall, the arm held and before a lift's end [mm]; the part of it at the arm's actual pose [mm]; the allowances [mm]":
            '3.70, 6.02; 3.70, 8.16; 5, 10',
        'the least distance above the level in force at a period boundary, over the runs [m]':
            '0.0314',
        "Assumption 14.3 in the plans executed unchanged: the largest shortfall at the arm's actual pose, the arm held and before a lift's end [mm]; the room that Proposition 14.2 leaves [mm]; the shortfalls within it":
            '3.70, 8.16; 4.03, 8.93; yes',
    },
}


if __name__ == "__main__":
    main()
