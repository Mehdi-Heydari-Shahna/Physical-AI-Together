#!/usr/bin/env python3
"""
The Husky's modules substituted, and the modules' assumptions changed, on
the common task of Section 13.2.

Physical AI, Together, Chapter 13 (Section 13.3).

This script keeps the task, the yard, the observations and the record of
Section 13.2 (husky_route.py) and changes one thing at a time:
  * substitutions: the feedback of the control package replaced by pure
    pursuit, at four look-aheads (husky.route-speed-pursuit); at 0.5 m, with
    every test of Section 13.2 repeated (the stops of its Table S13.13, the
    crossings with its four predictors, two more time steps and the gyro's
    seeds 1 to 20), and at 1.5 m with the stops on the dry grounds;
    the proposal replaced by one that asks for 1.5 m/s and never for rest
    (husky.hurry), and by one that asks for 0.5 m/s (husky.careful);
  * a model mismatch: the manual's all-terrain payload of 51.5 kg on the
    chassis, which the estimator, the feedback's bounds and the labels do
    not know; and a ground with less friction than the map says, the wet
    grass's friction swept from the map's 0.35 down to 0.15 under the
    map's label;
  * actuator saturation: the drivers' torque limit swept from 30 N m down,
    in stops on asphalt at its label and in the crossing;
  * a delayed sensor: the encoders and the gyro read one to five controller
    updates late, without and with a compensation of the delay.
For every run it writes the observations of husky.metrics, and for every
change which of the stated assumptions failed: the estimate's bounds (the
measurement-error bound varpi), the feedback's bounds, the funnel, and
whether the alarm check saw it, with the alarm check's largest distance
over the periods of normal operation and the measurement's largest error at
the period boundaries, the quantity of (S5). The runs take about four
minutes on one core.

Usage
    python3 husky_variants.py            # writes results/
    python3 husky_variants.py --check    # also checks every value of the
                                         # summary against EXPECTED

Outputs, in results/ next to this file
    variants-record.txt      the record of the experiment
    variants-summary.txt     every value, by group; with --check, the
                             comparison with EXPECTED, which holds every value
                             of the summary but the versions of NumPy and
                             Python, among them every value that Section 13.3
                             quotes
    variants-friction.csv    stops on the wet grass at the map's label, by the
                             grass's true friction: the largest errors and
                             distances over the four stops [mm, mm/s]
                             (Figure 13.9a)
    variants-torque.csv      stops on asphalt at its label, by the torque
                             limit, the same (Figure 13.9b)
    variants-delay.csv       the crossing with the map, every 0.1 s, by the
                             delay: the true arc length less its estimate
                             (es_*) and less the commanded arc length (ds_*)
                             [mm] (Figure 13.10)

Requires what husky_route.py requires.
"""
import argparse
import os
import platform
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mujoco                                     # noqa: E402
import numpy as np                                # noqa: E402

import husky_route as hr                          # noqa: E402  (Section 13.2's runs)
import husky_mujoco as h131                       # noqa: E402
from husky import estimate, governed as hg, model, sim, variants as hv, world   # noqa: E402
import allovi                                     # noqa: E402

LOOKAHEADS = (0.5, 0.75, 1.0, 1.5)
FRICTIONS = (0.35, 0.30, 0.25, 0.22, 0.20, 0.19, 0.18, 0.17, 0.16, 0.15)
STOP_TORQUES = (30.0, 25.0, 20.0, 15.0, 13.0, 10.0)
CROSSING_TORQUES = (20.0, 15.0, 12.0, 10.0)
DELAYS = (0, 1, 2, 3, 5)
COMPENSATED = (2, 5)
CAREFUL_PERIODS = 300
PURSUIT_STOPS = (1.5,)          # look-ahead whose stops on the dry grounds are run
PURSUIT_REPEATED = 0.5          # the look-ahead repeated at two time steps and twenty seeds
STOP_GROUNDS = (("asphalt", 1.0), ("gravel", 15.0), ("wet grass", 19.0))
STOP_SPEEDS = hr.STOP_SPEEDS
COMPONENTS = hr.COMPONENTS


# ----------------------------------------------------------------------------
# A run on the common task, and its observations
# ----------------------------------------------------------------------------
TR = {k: n for n, k in enumerate(hg.Yard.TRACE)}


class TorqueYard(hg.Yard):
    """husky.yard with the robot of husky.variants.PeakHusky: it also logs,
    at every update of the controller, the largest torque of a wheel's drive
    over the simulator's steps since the last update, the number of those
    steps with a drive at its limit, the feedback's yaw-rate command, the
    mean torques of the left and the right pair at the update, the yaw rate
    that the drive controller passed, and the yaw rate that the feedback
    asked without its integral."""
    ROBOT = hv.PeakHusky

    def reset(self):
        self.drive = []
        self.limit = self.build.get("torque", model.TORQUE)
        state = super().reset()
        self.h.take()                          # the calibration at rest is not logged
        return state

    def _log(self, law, tau, v):
        super()._log(law, tau, v)
        peak, steps = self.h.take()
        tq = self.h.torques()
        self.drive.append((self.time(), peak, steps, self.h.command[1],
                           sum(tq[w] for w in model.LEFT) / len(model.LEFT),
                           sum(tq[w] for w in model.RIGHT) / len(model.RIGHT),
                           self.h.drive.command[1], law.feedback.asked))

    @property
    def torque(self):
        return max(r[1] for r in self.drive)


def run(name, control=("husky.route-speed",), proposal=("husky.cruise",), estimator="sideslip",
        weather=None, build=None, spec=None, periods=hr.PERIODS, dt=None,
        capability=("husky.map",)):
    """The configuration of Section 13.2 with the given substitutions."""
    yard = TorqueYard(estimator=estimator, shower=weather, build=build, spec=spec, dt=dt)
    cfg = allovi.build(name, hg.T, {
        "control": control, "proposal": proposal, "capability": capability,
        "robot": ("husky.robot",), "world": yard, "evaluator": hg.Metrics(yard)})
    rec = allovi.run(cfg, periods=periods, strict=True, diagnostics=True)
    return rec, yard, cfg, cfg.evaluator.evaluate(rec)


def level_at_end(rec):
    d = rec.periods[-1].decision
    if d is None or d.profile is None:
        return "none: the rest feedback"
    return f"{d.profile.level:.2f}"


def end(m):
    zone = f"{m['in_zone']:.2f}" if m["in_zone"] is not None else "never"
    rest = f"{m['at_rest_in_zone']:.2f}" if m["at_rest_in_zone"] is not None else "never"
    return zone, rest


def worst_of(rows):
    """The largest errors of the estimate, distances of the estimate and of
    the state from the commanded motion, and distance of the state from its
    funnel in the norm, as husky.metrics computes them, over rows of the
    world's log."""
    out = {}
    for r in rows:
        g = lambda k: r[TR[k]]                                     # noqa: E731
        vals = dict(error_s=abs(g("s") - g("s_hat")), error_u=abs(g("u") - g("u_hat")),
                    error_d=abs(g("d") - g("d_hat")), error_e=abs(g("e") - g("e_hat")),
                    track_s=abs(g("s_hat") - g("s_c")), track_u=abs(g("u_hat") - g("u_c")),
                    track_d=abs(g("d_hat")), track_e=abs(g("e_hat")),
                    true_s=abs(g("s") - g("s_c")), true_u=abs(g("u") - g("u_c")),
                    true_d=abs(g("d")), true_e=abs(g("e")))
        vals["outside"] = outside_of(vals)
        for k, v in vals.items():
            out[k] = max(out.get(k, 0.0), v)
    return out


def before_alarm(yard, rec):
    """The rows of the world's log up to the first alarm, and its instant
    [s]; the whole log, and None, without an alarm."""
    if not rec.alarms:
        return yard.trace, None
    ta = rec.alarms[0] * hg.T
    return [r for r in yard.trace if r[0] <= ta + 1e-9], ta


def boundaries(rec):
    """Per period of normal operation: the instant t_{k+1} [s], the initial
    set I_{k+1}, and the estimate and the state at t_{k+1}."""
    out = []
    for k, p in enumerate(rec.periods):
        if p.governor != "normal" or p.initial_set is None:
            continue
        nxt = rec.periods[k + 1] if k + 1 < len(rec.periods) else None
        xm = nxt.measured if nxt is not None else rec.final_measured
        x = nxt.state if nxt is not None else rec.final_state
        out.append(((k + 1) * hg.T, p.initial_set, xm, x))
    return out


def alarm_check(rec, cfg):
    """The alarm check's largest distance, in the norm, of the estimate at
    t_{k+1} from the initial set I_{k+1}, over the periods of normal
    operation, and its instant t_{k+1} [s]."""
    best, when = 0.0, 0.0
    for t, ini, xm, _ in boundaries(rec):
        d = cfg.control.distance(xm, ini)
        if d > best:
            best, when = d, t
    return best, when


def s5(rec, ta):
    """(S5): the measurement's largest error in the norm at the period
    boundaries t_k, up to the first alarm, and its instant [s]."""
    pairs = [(p.t, p.state, p.measured) for p in rec.periods]
    pairs.append((len(rec.periods) * hg.T, rec.final_state, rec.final_measured))
    best, when = 0.0, 0.0
    for t, x, xm in pairs:
        if ta is not None and t > ta + 1e-9:
            break
        e = max(abs(x.s - xm.s), abs(x.u - xm.u) * hg.TAU_U, abs(x.d - xm.d) / hg.D_SCALE,
                abs(x.e - xm.e) * hg.RHO_E)
        if e > best:
            best, when = e, t
    return best, when


def episodes(rows):
    """The runs of consecutive updates with the state outside its funnel:
    (first and last instant [s], first and last arc length [m], the largest
    distance in the norm [m])."""
    out, cur, last = [], None, None
    for i, r in enumerate(rows):
        d = outside_of(worst_of([r]))
        if d <= 0.0:
            continue
        if cur is not None and i == last + 1:
            cur[1], cur[3], cur[4] = r[0], r[TR["s"]], max(cur[4], d)
        else:
            if cur is not None:
                out.append(tuple(cur))
            cur = [r[0], r[0], r[TR["s"]], r[TR["s"]], d]
        last = i
    if cur is not None:
        out.append(tuple(cur))
    return out


def failing(m):
    """The components of the estimate's and the feedback's bounds that a run
    did not keep."""
    names = (("error_s", hg.E_S, "E_S"), ("error_u", hg.E_U, "E_U"), ("error_d", hg.E_D, "E_D"),
             ("error_e", hg.E_E, "E_E"), ("track_s", hg.F_S, "F_S"), ("track_u", hg.F_U, "F_U"),
             ("track_d", hg.F_D, "F_D"), ("track_e", hg.F_E, "F_E"))
    out = [label for key, bound, label in names if m[key] > bound]
    return ", ".join(out) if out else "none"


def observe(out, name, rec, yard, cfg, m, first_alarm=True):
    """The common observations of a crossing."""
    zone, rest = end(m)
    out[f"{name}: front in the dock zone at [s]; at rest there from [s]; level of the plan in "
        f"force at the end [m]; clearance of the footprint from the dock, least and at the "
        f"end [m]"] = (f"{zone}; {rest}; {level_at_end(rec)}; {m['least_clearance']:.4f}, "
                       f"{m['final_clearance']:.4f}")
    out[f"{name}: the estimate's largest error {COMPONENTS}"] = hr.four(
        (m["error_s"], m["error_u"], m["error_d"], m["error_e"]))
    out[f"{name}: the estimate's largest distance from the commanded motion {COMPONENTS}"] = \
        hr.four((m["track_s"], m["track_u"], m["track_d"], m["track_e"]))
    out[f"{name}: the bounds of the estimate and of the feedback not kept"] = failing(m)
    out[f"{name}: the state's largest distance from the commanded motion {COMPONENTS}; within "
        f"the funnel; its largest distance from the funnel in the norm [m]"] = (
        hr.four((m["true_s"], m["true_u"], m["true_d"], m["true_e"])) +
        f"; {hr.yes(hr.in_funnel(m))}; {m['outside']:.4f}")
    first = f", the first at {rec.alarms[0] * hg.T:.1f} s" if rec.alarms else ""
    out[f"{name}: the footprint's largest offset from the centre line [m]; touches; alarms, "
        f"restarts; the largest torque of a wheel's drive [N m]"] = (
        f"{m['footprint']:.3f}; {hr.fmt_touch(m)}; {len(m['alarms'])}{first}, "
        f"{len(m['restarts'])}; {yard.torque:.1f}")
    dist, when = alarm_check(rec, cfg)
    out[f"{name}: the alarm check's largest distance of the estimate from the initial set, in "
        f"the norm [m]; at [s]"] = f"{dist:.4f}; {when:.1f}"
    rows, ta = before_alarm(yard, rec)
    worst5, when5 = s5(rec, ta)
    out[f"{name}: (S5), before any alarm: the measurement's largest error at the period "
        f"boundaries, in the norm [m], at [s]; within varpi"] = (
        f"{worst5:.4f}, at {when5:.1f}; {hr.yes(worst5 <= hg.VARPI)}")
    side = max(rows, key=lambda r: abs(r[TR["d"]]))
    turn = max(rows, key=lambda r: abs(r[TR["e"]]))
    out[f"{name}: before any alarm, the state's largest offset [m] and heading error [rad], "
        f"signed, each at [s], s [m]"] = (
        f"{side[TR['d']]:+.4f} at {side[0]:.2f}, {side[TR['s']]:.2f}; {turn[TR['e']]:+.4f} at "
        f"{turn[0]:.2f}, {turn[TR['s']]:.2f}")
    steer = max(rows, key=lambda r: abs(r[TR["e_hat"]]))
    out[f"{name}: before any alarm, the estimate's largest distance from the route's direction "
        f"[rad], at [s], s [m]"] = (f"{abs(steer[TR['e_hat']]):.4f}, at {steer[0]:.2f}, "
                                    f"{steer[TR['s']]:.2f}")
    eps = episodes(rows)
    if eps:
        inside = [r for r in rows if eps[0][0] - 1e-9 <= r[0] <= eps[-1][1] + 1e-9]
        n_out = sum(outside_of(worst_of([r])) > 0.0 for r in rows)
        out[f"{name}: before any alarm, the state outside its funnel: episodes of consecutive "
            f"updates; the first four, from [s] to [s], s from [m] to [m], the largest distance "
            f"in the norm [mm]; the share of the updates with the state outside, over the run "
            f"and from the first exit to the last"] = (
            f"{len(eps)}; " + "; ".join(
                f"{a:.2f} to {b:.2f}, {sa:.2f} to {sb:.2f}, {1e3 * d:.1f}"
                for a, b, sa, sb, d in eps[:4]) +
            f"; {n_out / len(rows):.2f}, {n_out / len(inside):.2f}")
    bad = [(t, cfg.control.distance(xm, ini), cfg.control.distance(x, ini))
           for t, ini, xm, x in boundaries(rec)]
    bad = [b for b in bad if b[1] > 0.0 or b[2] > 0.0]
    if bad:
        out[f"{name}: the period boundaries at which the estimate or the state lay outside the "
            f"initial set: number; the largest distance of the estimate and of the state [mm]; "
            f"the first four, instant [s], the estimate's and the state's distance [mm]"] = (
            f"{len(bad)}; {1e3 * max(b[1] for b in bad):.1f}, {1e3 * max(b[2] for b in bad):.1f}; "
            + "; ".join(f"{t:.1f}, {1e3 * de:.1f}, {1e3 * dx:.1f}" for t, de, dx in bad[:4]))
    lane = [r for r in yard.trace if r[TR["footprint"]] > world.LANE_HALF_WIDTH]
    if lane:
        tt = m["touches"][0][0] if m["touches"] else None
        at_touch = (min(yard.trace, key=lambda r: abs(r[0] - tt))[TR["footprint"]]
                    if tt is not None else None)
        out[f"{name}: the footprint first outside the lane at [s], s [m]; after the first alarm "
            f"by [s]; its offset from the centre line at the update of the first touch [m]"] = (
            f"{lane[0][0]:.2f}, {lane[0][TR['s']]:.2f}; " +
            (f"{lane[0][0] - ta:.2f}" if ta is not None else "no alarm") + "; " +
            (f"{at_touch:.3f}" if at_touch is not None else "no touch"))
    head = max(rows, key=lambda r: abs(r[TR["e"]] - r[TR["e_hat"]]))
    out[f"{name}: before any alarm, the estimate's largest error in heading [rad], at [s], "
        f"s [m]"] = (f"{abs(head[TR['e']] - head[TR['e_hat']]):.4f}, at {head[0]:.2f}, "
                     f"{head[TR['s']]:.2f}")
    plans = [p.decision.plan for p in rec.periods
             if p.decision is not None and p.decision.plan is not None]
    if plans and not rec.alarms:
        plan = plans[-1]
        out[f"{name}: the plan in force at the end: the dock's reserve [m]; the least clearance "
            f"of the footprint in its funnel [m]"] = (
            f"{plan.reserve():.4f}; {hg.DOCK - plan.s_rest - hg.DS - hg.front(hg.DE):.4f}")
    drive = [r for r in yard.drive if ta is None or r[0] <= ta + 1e-9]
    top = max(drive, key=lambda r: r[1])
    at_limit = [r[0] for r in drive if r[2] > 0]
    s_top = min(yard.trace, key=lambda r: abs(r[0] - top[0]))[TR["s"]]
    out[f"{name}: before any alarm, over the simulator's steps, the largest torque of a wheel's "
        f"drive [N m], in the update ending at [s], s [m]; the steps with a drive at its limit; "
        f"the updates with such steps, first and last ending at [s]"] = (
        f"{top[1]:.3f}, at {top[0]:.2f}, {s_top:.2f}; {sum(r[2] for r in drive)}" +
        (f"; {len(at_limit)}, {at_limit[0]:.2f}, {at_limit[-1]:.2f}" if at_limit else ""))
    if rec.alarms and first_alarm:
        k = rec.alarms[0] - 1
        ini, mk = rec.periods[k].initial_set, rec.periods[k + 1].measured
        out[f"{name}: the first alarm: the estimate beyond the initial set {COMPONENTS}; in "
            f"the norm [m]"] = (
            hr.four((abs(mk.s - ini.s) - ini.width, abs(mk.u - ini.c) - hg.DU,
                     abs(mk.d) - hg.DD, abs(mk.e) - hg.DE)) +
            f"; {cfg.control.distance(mk, ini):.4f}")
        w = worst_of(rows)
        out[f"{name}: before the first alarm: the bounds of the estimate and of the feedback not "
            f"kept; the state's largest distance from the commanded motion {COMPONENTS}; within "
            f"the funnel; its largest distance from it in the norm [m]"] = (
            failing(w) + "; " + hr.four((w["true_s"], w["true_u"], w["true_d"], w["true_e"])) +
            f"; {hr.yes(w['outside'] == 0.0)}; {w['outside']:.4f}")
        after = [p for p in rec.periods if p.governor == "rest feedback"]
        reasons = {}
        for p in after:
            if p.reach is None:
                reasons[p.reach_reason] = reasons.get(p.reach_reason, 0) + 1
        out[f"{name}: after the first alarm: the periods of the rest feedback; those in which "
            f"the package offered a set to restart from; the reasons given in the others, with "
            f"their number"] = (
            f"{len(after)}; {sum(p.reach is not None for p in after)}; " +
            "; ".join(f"{why} ({n})" for why, n in sorted(reasons.items())))
        if m["touches"] and m["touches"][0][0] > ta:
            tt = m["touches"][0][0]
            mid = [r for r in yard.trace if ta < r[0] < tt - 1e-9]
            mid_drive = [r for r in yard.drive if ta < r[0] < tt - 1e-9]
            per = yard.h.steps_per_update
            out[f"{name}: between the first alarm and the update of the first touch: the "
                f"robot's least and largest speed [m/s]; the share of the simulator's steps with "
                f"a drive at its limit; at the controller's updates, the mean torque of the left "
                f"and of the right pair [N m]; the feedback's yaw-rate command without its "
                f"integral, least and largest, and with it, largest [rad/s]; the largest yaw "
                f"rate that the drive controller passed [rad/s]"] = (
                f"{min(r[TR['u']] for r in mid):.3f}, {max(r[TR['u']] for r in mid):.3f}; "
                f"{sum(r[2] for r in mid_drive) / (per * len(mid_drive)):.2f}; "
                f"{sum(r[4] for r in mid_drive) / len(mid_drive):+.2f}, "
                f"{sum(r[5] for r in mid_drive) / len(mid_drive):+.2f}; "
                f"{min(r[7] for r in mid_drive):.2f}, {max(r[7] for r in mid_drive):.2f}, "
                f"{max(abs(r[3]) for r in mid_drive):.2f}; "
                f"{max(abs(r[6]) for r in mid_drive):.2f}")
    if m["touches"]:
        tt = m["touches"][0][0]
        later = [r for r in yard.trace if r[0] >= tt - 1e-9]
        out[f"{name}: from the update of the first touch to the end: the robot's least and "
            f"largest forward speed [m/s]; the footprint's largest offset from the centre line "
            f"[m]; the true arc length at the end [m]"] = (
            f"{min(r[TR['u']] for r in later):+.3f}, {max(r[TR['u']] for r in later):+.3f}; "
            f"{max(r[TR['footprint']] for r in later):.3f}; {rec.final_state.s:.2f}")


def outside_of(worst):
    return max(0.0, worst["true_s"] - hg.DS, (worst["true_u"] - hg.DU) * hg.TAU_U,
               (worst["true_d"] - hg.DD) / hg.D_SCALE, (worst["true_e"] - hg.DE) * hg.RHO_E)


def stops(start, weather=None, spec=None, follower=None, label_from=None, **build):
    """Stops from rest at the four speeds of Section 13.2 (husky_route.stop_run),
    at the map's label, by the robot of husky.variants.PeakHusky built with
    'build', or from the MjSpec that spec(**build) returns: the label, beyond
    the commanded rest point and the estimate's error at rest at each speed,
    and the worst values, with the largest torque of a wheel's drive over
    the simulator's steps and the number of steps with a drive at its limit,
    over the four stops, after the calibration at rest that precedes each;
    follower, a function that returns the feedback (follow.Follower by
    default); label_from as in husky_route.stop_run."""
    robots = []
    after = estimate.CALIBRATION

    def make():
        robots.append(hv.PeakHusky(after=after, **build) if spec is None else
                      hv.PeakHusky(spec=spec(**build), after=after))
        return robots[-1]
    rows = [hr.stop_run(start, v, weather, label_from, make=make, follower=follower)
            for v in STOP_SPEEDS]
    label = rows[0][0]
    beyond = [st - s_rest for _, s_rest, st, _, _ in rows]
    error = [st - se for _, _, st, se, _ in rows]
    worst = {k: max(r[4][k] for r in rows) for k in rows[0][4]}
    worst["torque"] = max(h.peak_all for h in robots)
    worst["limit_steps"] = sum(h.steps_all for h in robots)
    return label, beyond, error, worst


def stop_facts(out, name, label, beyond, error, worst):
    out[f"{name}: label [m/s^2]; beyond the commanded rest point at 0.25, 0.5, 0.75, 1 m/s "
        f"[mm]"] = f"{label:.3g}; " + ", ".join(f"{1e3 * b:+.1f}" for b in beyond)
    out[f"{name}: the estimate's error at rest at 0.25, 0.5, 0.75, 1 m/s [mm]"] = \
        ", ".join(f"{1e3 * e:+.1f}" for e in error)
    out[f"{name}: over the four stops, the estimate's largest error along [mm]; its largest "
        f"distance from the commanded arc length [mm] and speed [mm/s]; the bounds not kept"] = (
        f"{1e3 * worst['error_s']:.1f}; {1e3 * worst['track_s']:.1f}, "
        f"{1e3 * worst['track_u']:.1f}; {failing(worst)}")
    out[f"{name}: over the four stops, the state's largest distance from the commanded arc "
        f"length [mm] and speed [mm/s]; within the funnel; its largest distance from it in the "
        f"norm [mm]"] = (
        f"{1e3 * worst['true_s']:.1f}, {1e3 * worst['true_u']:.1f}; "
        f"{hr.yes(hr.in_funnel(worst))}; {1e3 * outside_of(worst):.1f}")
    out[f"{name}: over the four stops, the estimate's largest error in speed [mm/s]; over the "
        f"simulator's steps, the largest torque of a wheel's drive [N m] and the steps with a "
        f"drive at its limit"] = (f"{1e3 * worst['error_u']:.1f}; {worst['torque']:.1f}, "
                                  f"{worst['limit_steps']}")
    out[f"{name}: over the four stops, (S5): the measurement's largest error at the ends of "
        f"the periods, in the norm [mm]; within varpi"] = (
        f"{1e3 * worst['s5']:.1f}; {hr.yes(worst['s5'] <= hg.VARPI)}")


def last_kept(rows, flag, key):
    """Down a sweep, the last value of 'key' before the first row whose
    'flag' is 0; 'none' if the first row has it 0."""
    out = "none"
    for r in rows:
        if not r[flag]:
            break
        out = f"{r[key]:g}"
    return out


def monotonic(rows):
    """Whether, down a sweep, no row keeps the bounds or the funnel after a
    row that did not."""
    return all(not (rows[j][f] and not rows[i][f]) for f in ("kept", "funnel")
               for i in range(len(rows)) for j in range(i + 1, len(rows)))


def stop_row(**head):
    """A row of the CSV files of the stops (Figure 13.9)."""
    def row(label, beyond, error, worst):
        est_ok, fb_ok = hr.kept(worst)
        return dict(head, label=label,
                    **{f"beyond_{v:g}": 1e3 * b for v, b in zip(STOP_SPEEDS, beyond)},
                    **{f"error_{v:g}": 1e3 * e for v, e in zip(STOP_SPEEDS, error)},
                    **{k: 1e3 * worst[k] for k in ("error_s", "error_u", "track_s", "track_u",
                                                   "true_s", "true_u")},
                    torque_max=worst["torque"], limit_steps=worst["limit_steps"],
                    kept=int(est_ok and fb_ok),
                    funnel=int(hr.in_funnel(worst)), outside=1e3 * outside_of(worst))
    return row


# ----------------------------------------------------------------------------
# The groups
# ----------------------------------------------------------------------------
def record_facts():
    out = {}
    out["MuJoCo"] = mujoco.__version__
    out["NumPy, Python"] = f"{np.__version__}, {platform.python_version()}"
    out["allovi core, contract"] = f"{allovi.__version__}, {allovi.CONTRACT}"
    out["the reference: control package, proposal, predictor, estimator; gyro seed; time step "
        "[ms]; periods"] = (f"husky.route-speed, husky.cruise, husky.map, dead reckoning with "
                            f"sideslip; {estimate.SEED}; {1000 * model.TIMESTEP:g}; {hr.PERIODS}")
    out["pure pursuit: look-aheads [m]"] = ", ".join(f"{x:g}" for x in LOOKAHEADS)
    out["proposals: husky.hurry asks for [m/s]; husky.careful asks for [m/s], periods"] = (
        f"1.5; 0.5, {CAREFUL_PERIODS}")
    h = sim.Husky(spec=hv.payload_spec())
    out["payload: mass [kg]; box half sizes [m]; centre above the ground [m]; the robot's mass "
        "with it [kg]"] = (f"{hv.PAYLOAD:g}; " + ", ".join(f"{x:g}" for x in hv.PAYLOAD_HALF) +
                           f"; {hv.PAYLOAD_HEIGHT:g}; {h.total_mass():g}")
    out["less friction: the wet grass's true friction under the map's label"] = ", ".join(
        f"{x:g}" for x in FRICTIONS)
    out["weaker actuator: torque limits in the stops on asphalt and in the crossing [N m]"] = (
        ", ".join(f"{x:g}" for x in STOP_TORQUES) + "; " +
        ", ".join(f"{x:g}" for x in CROSSING_TORQUES))
    out["late sensor: delays [controller updates of 20 ms]; compensated"] = (
        ", ".join(str(n) for n in DELAYS) + "; " + ", ".join(str(n) for n in COMPENSATED))
    return out


def reference_facts():
    out = {}
    rec, yard, cfg, m = run("reference")
    observe(out, "reference", rec, yard, cfg, m)
    # the same values as the crossing with the map of Section 13.2
    want = hr.EXPECTED["the predictors and the shower"]
    zone, rest = end(m)
    same = (want["map: front in the dock zone at [s]; at rest there from [s]"] == f"{zone}; {rest}"
            and want["map: level of the plan in force at the end [m]; clearance of the footprint "
                     "from the dock, least and at the end [m]"] ==
            f"{hr.final_level(rec):.2f}; {m['least_clearance']:.4f}, {m['final_clearance']:.4f}")
    out["reference: the crossing with the map of Section 13.2, in zone, rest, level and "
        "clearance"] = str(same)
    rest = [p.k for p in rec.periods if p.proposal == 0.0]
    out["reference: the first period in which husky.cruise asks for rest; it asks for rest in "
        "every later period"] = f"{rest[0]}; {rest == list(range(rest[0], len(rec.periods)))}"
    return out, (hr.decisions(rec), hr.committed(rec), heads(rec))


def heads(rec):
    """Per period, the head of the committed plan: (s0, c0, c1), or None."""
    return [(p.decision.plan.head().s0, p.decision.plan.head().c0, p.decision.plan.head().c1)
            if p.decision is not None and p.decision.plan is not None else None
            for p in rec.periods]


def spans(ks):
    """Period numbers as ranges: '125 to 169 (45 periods)'."""
    if not ks:
        return "none"
    out, start, last = [], ks[0], ks[0]
    for k in ks[1:] + [None]:
        if k is not None and k == last + 1:
            last = k
            continue
        out.append(f"{start}" if start == last else f"{start} to {last}")
        if k is not None:
            start = last = k
    return ", ".join(out) + f" ({len(ks)} periods)"


def substitution_facts(reference):
    out = {}
    for la in LOOKAHEADS:
        name = f"pure pursuit, look-ahead {la:g} m"
        rec, yard, cfg, m = run(name, control=("husky.route-speed-pursuit", {"lookahead": la}))
        observe(out, name, rec, yard, cfg, m)
    for la in PURSUIT_STOPS:
        for ground, start in STOP_GROUNDS:
            label, beyond, error, worst = stops(
                start, follower=lambda la=la: hv.PurePursuit(lookahead=la))
            stop_facts(out, f"pure pursuit, look-ahead {la:g} m, stops on {ground}", label,
                       beyond, error, worst)
    la = PURSUIT_REPEATED
    # every stop of Table S13.13, and every crossing with a predictor of Section 13.2, with
    # pure pursuit, against the verdicts of Section 13.2 (husky_route.EXPECTED)
    same, n = True, 0
    want = hr.EXPECTED["stops at the labels"]
    for place, start, wx, label_from in hr.STOP_PLACES:
        label, beyond, error, worst = stops(
            start, weather=hr.shower() if wx else None, label_from=label_from,
            follower=lambda: hv.PurePursuit(lookahead=la))
        stop_facts(out, f"pure pursuit, look-ahead {la:g} m, stops: {place}", label, beyond,
                   error, worst)
        est_ok, fb_ok = hr.kept(worst)
        mine = f"{hr.yes(est_ok)}, {hr.yes(fb_ok)}; {hr.yes(hr.in_funnel(worst))}"
        ref = next(v for k, v in want.items() if k.startswith(f"{place}: over the four stops, "
                                                              f"the estimate's largest error"))
        same = same and ref.endswith(mine)
        n += 1
    out[f"pure pursuit, look-ahead {la:g} m: the stops of Table 13.6; the estimate's bounds "
        f"kept, the feedback's and the funnel in each as with the feedback of Section 13.2"] = (
        f"{n}; {same}")
    same = True
    want = hr.EXPECTED["the predictors and the shower"]
    for pre, cap, wx in (("fixed", ("husky.fixed",), None),
                         ("reactive", ("husky.reactive",), None),
                         ("shower, map", ("husky.map",), hr.shower()),
                         ("shower, forecast", ("husky.forecast", {"shower": hr.shower()}),
                          hr.shower())):
        name = f"pure pursuit, look-ahead {la:g} m, {pre}"
        rec, yard, cfg, m = run(name, control=("husky.route-speed-pursuit", {"lookahead": la}),
                                weather=wx, capability=cap)
        zone, rest = end(m)
        est_ok, fb_ok = hr.kept(m)
        out[f"{name}: at rest in the dock zone from [s]; level at the end [m]; clearance at the "
            f"end [m]; the bounds of the estimate and of the feedback not kept; within the "
            f"funnel, its largest distance from it in the norm [m]; alarms; touches"] = (
            f"{rest}; {level_at_end(rec)}; {m['final_clearance']:.4f}; {failing(m)}; "
            f"{hr.yes(hr.in_funnel(m))}, {m['outside']:.4f}; {len(m['alarms'])}; "
            f"{hr.fmt_touch(m)}")
        ref_e = want[f"{pre}: the estimate's largest error along [m], in speed [m/s]; its bounds "
                     f"kept, the feedback's kept; alarms; touches"]
        ref_x = want[f"{pre}: the state's largest distance from the commanded motion {COMPONENTS}"
                     f"; within the funnel; its largest distance from the funnel in the norm [m]"]
        same = (same and ref_e.split("; ", 1)[1] ==
                f"{hr.yes(est_ok)}, {hr.yes(fb_ok)}; {len(m['alarms'])}; {hr.fmt_touch(m)}"
                and ref_x.split("; ")[1] == hr.yes(hr.in_funnel(m)))
    out[f"pure pursuit, look-ahead {la:g} m: the crossings with the predictors of Section 13.2, "
        f"dry and in the shower; the bounds kept, the funnel, the alarms and the touches in each "
        f"as with the feedback of Section 13.2"] = f"4; {same}"
    for dt in (0.001, 0.00025):
        name = f"pure pursuit, look-ahead {la:g} m, time step {1000 * dt:g} ms"
        rec, yard, cfg, m = run(name, control=("husky.route-speed-pursuit", {"lookahead": la}),
                                dt=dt)
        zone, rest = end(m)
        out[f"{name}: at rest in the dock zone from [s]; level at the end [m]; clearance at the "
            f"end [m]; the bounds of the estimate and of the feedback not kept; within the "
            f"funnel"] = (f"{rest}; {level_at_end(rec)}; {m['final_clearance']:.4f}; "
                          f"{failing(m)}; {hr.yes(hr.in_funnel(m))}")
    ms = []
    for seed in hr.SEEDS:
        old = estimate.SEED
        estimate.SEED = seed
        try:
            rec, yard, cfg, m = run(f"pure pursuit, seed {seed}",
                                    control=("husky.route-speed-pursuit", {"lookahead": la}))
        finally:
            estimate.SEED = old
        ms.append(m)
    name = f"pure pursuit, look-ahead {la:g} m, seeds {hr.SEEDS[0]} to {hr.SEEDS[-1]}"
    out[f"{name}: the estimate's largest error {COMPONENTS}; its largest distance from the "
        f"commanded motion, the same; seeds that keep the estimate's bounds, the feedback's; "
        f"seeds within the funnel; alarms; touches"] = (
        hr.four(tuple(max(m[k] for m in ms) for k in ("error_s", "error_u", "error_d",
                                                     "error_e"))) + "; " +
        hr.four(tuple(max(m[k] for m in ms) for k in ("track_s", "track_u", "track_d",
                                                     "track_e"))) + "; " +
        f"{sum(hr.kept(m)[0] for m in ms)}, {sum(hr.kept(m)[1] for m in ms)}; "
        f"{sum(hr.in_funnel(m) for m in ms)}; {sum(len(m['alarms']) for m in ms)}; "
        f"{sum(bool(m['touches']) for m in ms)}")
    out[f"{name}: clearance at the end, least and largest [m]; the state's largest distance "
        f"from the commanded motion {COMPONENTS}"] = (
        f"{min(m['final_clearance'] for m in ms):.4f}, {max(m['final_clearance'] for m in ms):.4f}"
        "; " + hr.four(tuple(max(m[k] for m in ms) for k in ("true_s", "true_u", "true_d",
                                                            "true_e"))))
    rec, yard, cfg, m = run("husky.hurry", proposal=("husky.hurry",))
    observe(out, "husky.hurry", rec, yard, cfg, m)
    mine = hr.decisions(rec)
    out["husky.hurry: periods committing ambition 1, between 0 and 1, 0; the reference's"] = (
        "{}, {}, {}".format(*hr.committed(rec)) + "; " + "{}, {}, {}".format(*reference[1]))
    same = all((a[0], a[2:], heads_a) == (b[0], b[2:], heads_b) for a, b, heads_a, heads_b in
               zip(mine, reference[0], heads(rec), reference[2]))
    differ = [a[0] for a, b in zip(mine, reference[0]) if a[1] != b[1]]
    out["husky.hurry: the plan committed in every period that of the reference (level, point of "
        "rest and head); the periods whose committed ambition differs, the reference's there "
        "and husky.hurry's"] = (
        f"{same}; {spans(differ)}; " +
        ", ".join(sorted({f'{b[1]:g}' for a, b in zip(mine, reference[0]) if a[1] != b[1]})) +
        "; " + ", ".join(sorted({f'{a[1]:g}' for a, b in zip(mine, reference[0])
                                 if a[1] != b[1]})))
    rec, yard, cfg, m = run("husky.careful", proposal=("husky.careful",), periods=CAREFUL_PERIODS)
    observe(out, "husky.careful", rec, yard, cfg, m)
    return out


def payload_facts():
    out = {}
    # the rolling radius, as Section 13.1.4 measured it, with the payload
    h = sim.Husky(spec=hv.payload_spec(ground=0.8, obstacles=False))
    h.place()
    h.command = (1.0, 0.0)
    h.run(2.0)
    x0 = h.pose()[0]
    q0 = np.mean([h.d.qpos[h.qadr[w]] for w in model.WHEELS])
    h.run(10.0)
    rho = (h.pose()[0] - x0) / (np.mean([h.d.qpos[h.qadr[w]] for w in model.WHEELS]) - q0)
    out["payload: rolling radius on asphalt over 10 s at 1 m/s [m]; the estimator's; the "
        "difference [%]"] = f"{rho:.5f}; {estimate.RHO:.5f}; {100 * (rho / estimate.RHO - 1):+.2f}"
    loaded = sim.Husky(spec=hv.payload_spec())
    torque = loaded.wheel_radius() * loaded.total_mass() * hg.label_of(0.8) / 4
    out["payload: the torque at each wheel with which four wheels give the asphalt's label, "
        "r m 4 / 4 [N m]"] = f"{torque:.1f}"
    rec, yard, cfg, m = run("payload", spec=hv.payload_spec)
    observe(out, "payload", rec, yard, cfg, m)
    for ground, start in STOP_GROUNDS:
        label, beyond, error, worst = stops(start, spec=hv.payload_spec)
        stop_facts(out, f"payload, stops on {ground}", label, beyond, error, worst)
    return out


def friction_facts(out_dir):
    out, rows = {}, []
    for mu in FRICTIONS:
        weather = hg.Shower(0.0, {"wet grass": mu})
        label, beyond, error, worst = stops(19.0, weather=weather)
        name = f"wet grass of friction {mu:g}"
        out[f"{name}: the map's label as a share of mu g"] = f"{label / (mu * hg.G):.2f}"
        stop_facts(out, name, label, beyond, error, worst)
        rows.append(stop_row(mu=mu, share=label / (mu * hg.G))(label, beyond, error, worst))
    hr.write_csv(os.path.join(out_dir, "variants-friction.csv"), rows)
    out["less friction: down the sweep, the last friction at which every bound was kept; at "
        "which the state stayed in its funnel; whether no bound or funnel kept further down; the "
        "friction at which the map's label is mu g"] = (
        f"{last_kept(rows, 'kept', 'mu')}; {last_kept(rows, 'funnel', 'mu')}; "
        f"{monotonic(rows)}; {hg.label_of(0.35) / hg.G:.3f}")
    for mu in (0.20, 0.15):
        name = f"crossing, wet grass of friction {mu:g}"
        rec, yard, cfg, m = run(name, weather=hg.Shower(0.0, {"wet grass": mu}))
        observe(out, name, rec, yard, cfg, m)
    # the last is the crossing with the map in the shower of Section 13.2, but for the
    # friction on the wet grass before 20 s
    want = hr.EXPECTED["the predictors and the shower"]
    zone, rest = end(m)
    pre = "shower, map: "
    same = (want[pre + "front in the dock zone at [s]; at rest there from [s]"] ==
            f"{zone}; {rest}" and
            want[pre + "level of the plan in force at the end [m]; clearance of the footprint "
                 "from the dock, least and at the end [m]"] ==
            f"{hr.final_level(rec):.2f}; {m['least_clearance']:.4f}, {m['final_clearance']:.4f}"
            and want[pre + "the estimate's largest error along [m], in speed [m/s]; its bounds "
                     "kept, the feedback's kept; alarms; touches"].startswith(
                f"{m['error_s']:.4f}, {m['error_u']:.3f};")
            and want[pre + "the state's largest distance from the commanded motion along [m], in "
                     "speed [m/s], across [m], in heading [rad]; within the funnel; its largest "
                     "distance from the funnel in the norm [m]"] ==
            hr.four((m["true_s"], m["true_u"], m["true_d"], m["true_e"])) +
            f"; {hr.yes(hr.in_funnel(m))}; {m['outside']:.4f}")
    out["crossing, wet grass of friction 0.15: the crossing with the map in the shower of "
        "Section 13.2, in zone, rest, level, clearance, the estimate's error and the state's "
        "distances"] = str(same)
    return out


def torque_facts(out_dir):
    out, rows = {}, []
    mass, radius = sim.Husky().total_mass(), sim.Husky().wheel_radius()
    for tq in STOP_TORQUES:
        label, beyond, error, worst = stops(1.0, torque=tq)
        name = f"asphalt, torque limit {tq:g} N m"
        out[f"{name}: the deceleration that four wheels at the limit give the robot, "
            f"4 T / (r m) [m/s^2]"] = f"{4 * tq / (radius * mass):.2f}"
        stop_facts(out, name, label, beyond, error, worst)
        rows.append(stop_row(torque=tq, decel=4 * tq / (radius * mass))(label, beyond, error,
                                                                         worst))
    hr.write_csv(os.path.join(out_dir, "variants-torque.csv"), rows)
    out["asphalt: down the sweep, the last torque limit at which every bound was kept; at which "
        "the state stayed in its funnel; whether no bound or funnel kept further down; the torque "
        "limit at which four wheels give the label, r m 4 / 4 [N m]"] = (
        f"{last_kept(rows, 'kept', 'torque')}; {last_kept(rows, 'funnel', 'torque')}; "
        f"{monotonic(rows)}; {radius * mass * hg.label_of(0.8) / 4:.1f}")
    for tq in CROSSING_TORQUES:
        name = f"crossing, torque limit {tq:g} N m"
        rec, yard, cfg, m = run(name, build={"torque": tq})
        observe(out, name, rec, yard, cfg, m)
    return out


def delay_facts(out_dir):
    out, traces = {}, {}
    i = {k: n for n, k in enumerate(hg.Yard.TRACE)}
    cases = [(n, False) for n in DELAYS] + [(n, True) for n in COMPENSATED]
    for n, comp in cases:
        name = f"readings {20 * n} ms late" + (", compensated" if comp else "")
        rec, yard, cfg, m = run(name, estimator=lambda h, n=n, c=comp:
                                hv.DelayedDeadReckoning(h, n, compensate=c))
        observe(out, name, rec, yard, cfg, m)
        plan = next(p.decision.plan for p in reversed(rec.periods)
                    if p.decision is not None and p.decision.plan is not None)
        out[f"{name}: at the end, the true arc length beyond the point of rest and beyond its "
            f"estimate [mm]"] = (
            f"{1e3 * (m['final_s'] - plan.s_rest):+.1f}, "
            f"{1e3 * (m['final_s'] - m['final_s_hat']):+.1f}")
        traces[(n, comp)] = yard.trace
        if n == 0:
            turn = [r[0] for r in yard.trace if world.STRAIGHT_1 <= r[TR["s"]] <=
                    world.STRAIGHT_1 + world.TURN]
            out[f"{name}: the robot's centre in the turn from [s] to [s]"] = (
                f"{turn[0]:.2f}; {turn[-1]:.2f}")
    rows = []
    keys = [(0, False), (2, False), (5, False), (5, True)]
    n_rows = min(len(traces[k]) for k in keys)
    for r in range(0, n_rows, 5):
        row = dict(t=traces[keys[0]][r][0])
        for n, comp in keys:
            tr = traces[(n, comp)][r]
            row[f"es_{n}{'c' if comp else ''}"] = 1e3 * (tr[i["s"]] - tr[i["s_hat"]])
            row[f"ds_{n}{'c' if comp else ''}"] = 1e3 * (tr[i["s"]] - tr[i["s_c"]])
        rows.append(row)
    hr.write_csv(os.path.join(out_dir, "variants-delay.csv"), rows)
    return out


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
def write_record(path, facts):
    rec = facts["record"]

    def listed(values, unit=""):
        values = [f"{v:g}" for v in values]
        return (", ".join(values[:-1]) + " and " + values[-1] if len(values) > 1 else
                values[0]) + unit

    mass = rec["payload: mass [kg]; box half sizes [m]; centre above the ground [m]; the robot's "
               "mass with it [kg]"].split("; ")[-1]
    lines = [
        "Record of the experiment of Section 13.3: the Husky A300 of Section 13.2 with one module "
        "substituted, or one assumption changed, at a time",
        "",
        "model and asset provenance",
        "  the model, the drive controller, the drivers and the declared contacts of Section 13.1 "
        f"(record.txt); with the payload, {hv.PAYLOAD:g} kg in a box of half sizes "
        f"{listed(hv.PAYLOAD_HALF, ' m')}, centred {hv.PAYLOAD_HEIGHT:g} m above the ground, "
        f"{mass} kg in all (ours, after the manual's payload)",
        "simulator and solver",
        f"  MuJoCo {rec['MuJoCo']}, NumPy and Python {rec['NumPy, Python']}; the settings of "
        "Section 13.1",
        "time steps",
        "  simulator 0.0005 s; drive controller, estimator and feedback at 50 Hz; governor every "
        f"0.2 s; the crossing with pure pursuit at {PURSUIT_REPEATED:g} m repeated at 1 and "
        "0.25 ms",
        "modules",
        "  the reference of Section 13.2 (route-record.txt): control package husky.route-speed, "
        "proposal husky.cruise, predictor husky.map, the estimator with sideslip; substitutes "
        f"(husky/variants.py): husky.route-speed-pursuit at look-aheads of "
        f"{listed(LOOKAHEADS, ' m')}, at {PURSUIT_REPEATED:g} m with the stops and the "
        "crossings with the predictors of Section 13.2, at "
        f"{listed(PURSUIT_STOPS, ' m')} with the stops on the dry grounds; husky.hurry; and "
        f"husky.careful over {CAREFUL_PERIODS} periods",
        "actuators and sensors",
        "  as in Section 13.2; the drivers' torque limit at "
        f"{listed(STOP_TORQUES, ' N m')} in the stops on asphalt and at "
        f"{listed(CROSSING_TORQUES, ' N m')} in the crossing; the estimator's readings late by "
        f"{listed(DELAYS)} updates of 20 ms, and by {listed(COMPENSATED)} with the "
        "compensation; the torques recorded at every step of the simulator",
        "frames and units",
        "  as in Section 13.2",
        "uncertainty bounds",
        "  the bounds and labels of Section 13.2, unchanged in every run; the wet grass's true "
        f"friction {listed(FRICTIONS)}, in force from the start, under the map's label",
        "constraints",
        "  as in Section 13.2",
        "terminal behaviour",
        "  as in Section 13.2",
        "training and evaluation split",
        "  none: nothing learns in Section 13.3",
        "seeds",
        f"  the gyro's noise, PCG64, seed {estimate.SEED}; seeds {hr.SEEDS[0]} to {hr.SEEDS[-1]} "
        f"with pure pursuit at {PURSUIT_REPEATED:g} m",
        "hardware",
        f"  {platform.system()} {platform.machine()}, {h131.hardware()}",
        "guarantee status",
        "  that of Section 13.2 for the reference, for husky.hurry, whose plans are the "
        f"reference's, and for pure pursuit at {PURSUIT_REPEATED:g} m, which passed the same tests; "
        "for the other runs that kept every bound, the evidence of one crossing; for the others, "
        "the summary names the stated assumptions that the runs found broken, and there is no "
        "guarantee",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Section 13.3: substitutions and changed assumptions")
    ap.add_argument("--check", action="store_true", help="check every value quoted in Section 13.3")
    args = ap.parse_args()
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    reference, ref_decisions = reference_facts()
    facts = {"record": record_facts(),
             "the reference": reference,
             "substitutions": substitution_facts(ref_decisions),
             "a payload": payload_facts(),
             "less friction": friction_facts(out_dir),
             "a weaker actuator": torque_facts(out_dir),
             "a late sensor": delay_facts(out_dir)}
    write_record(os.path.join(out_dir, "variants-record.txt"), facts)
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
        lines.append(f"[check] {n} values, among them every value that Section 13.3 quotes: " +
                     ("all as expected" if not failures else "FAILED: " + "; ".join(failures)))
    with open(os.path.join(out_dir, "variants-summary.txt"), "w", encoding="utf-8") as fh:
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
        'the reference: control package, proposal, predictor, estimator; gyro seed; time step [ms]; periods':
            'husky.route-speed, husky.cruise, husky.map, dead reckoning with sideslip; 13; 0.5; 170',
        'pure pursuit: look-aheads [m]':
            '0.5, 0.75, 1, 1.5',
        'proposals: husky.hurry asks for [m/s]; husky.careful asks for [m/s], periods':
            '1.5; 0.5, 300',
        "payload: mass [kg]; box half sizes [m]; centre above the ground [m]; the robot's mass with it [kg]":
            '51.5; 0.2, 0.15, 0.05; 0.3; 131.5',
        "less friction: the wet grass's true friction under the map's label":
            '0.35, 0.3, 0.25, 0.22, 0.2, 0.19, 0.18, 0.17, 0.16, 0.15',
        'weaker actuator: torque limits in the stops on asphalt and in the crossing [N m]':
            '30, 25, 20, 15, 13, 10; 20, 15, 12, 10',
        'late sensor: delays [controller updates of 20 ms]; compensated':
            '0, 1, 2, 3, 5; 2, 5',
    },
    'the reference': {
        'reference: front in the dock zone at [s]; at rest there from [s]; level of the plan in force at the end [m]; clearance of the footprint from the dock, least and at the end [m]':
            '24.80; 25.58; 0.10; 0.1935, 0.1941',
        "reference: the estimate's largest error along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0233, 0.029, 0.0216, 0.0090',
        "reference: the estimate's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0074, 0.056, 0.0391, 0.0298',
        'reference: the bounds of the estimate and of the feedback not kept':
            'none',
        "reference: the state's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]; within the funnel; its largest distance from the funnel in the norm [m]":
            '0.0251, 0.042, 0.0600, 0.0310; yes; 0.0000',
        "reference: the footprint's largest offset from the centre line [m]; touches; alarms, restarts; the largest torque of a wheel's drive [N m]":
            '0.450; nothing; 0, 0; 29.5',
        "reference: the alarm check's largest distance of the estimate from the initial set, in the norm [m]; at [s]":
            '0.0000; 0.0',
        "reference: (S5), before any alarm: the measurement's largest error at the period boundaries, in the norm [m], at [s]; within varpi":
            '0.0233, at 14.2; yes',
        "reference: before any alarm, the state's largest offset [m] and heading error [rad], signed, each at [s], s [m]":
            '-0.0600 at 15.42, 14.74; +0.0310 at 15.54, 14.86',
        "reference: before any alarm, the estimate's largest distance from the route's direction [rad], at [s], s [m]":
            '0.0298, at 15.52, 14.84',
        "reference: before any alarm, the estimate's largest error in heading [rad], at [s], s [m]":
            '0.0090, at 14.22, 13.54',
        "reference: the plan in force at the end: the dock's reserve [m]; the least clearance of the footprint in its funnel [m]":
            '0.1063; 0.1224',
        "reference: before any alarm, over the simulator's steps, the largest torque of a wheel's drive [N m], in the update ending at [s], s [m]; the steps with a drive at its limit; the updates with such steps, first and last ending at [s]":
            '29.486, at 10.78, 10.09; 0',
        'reference: the crossing with the map of Section 13.2, in zone, rest, level and clearance':
            'True',
        'reference: the first period in which husky.cruise asks for rest; it asks for rest in every later period':
            '125; True',
    },
    'substitutions': {
        'pure pursuit, look-ahead 0.5 m: front in the dock zone at [s]; at rest there from [s]; level of the plan in force at the end [m]; clearance of the footprint from the dock, least and at the end [m]':
            '24.80; 25.58; 0.10; 0.1944, 0.1952',
        "pure pursuit, look-ahead 0.5 m: the estimate's largest error along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0232, 0.029, 0.0208, 0.0086',
        "pure pursuit, look-ahead 0.5 m: the estimate's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0074, 0.055, 0.0145, 0.0295',
        'pure pursuit, look-ahead 0.5 m: the bounds of the estimate and of the feedback not kept':
            'none',
        "pure pursuit, look-ahead 0.5 m: the state's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]; within the funnel; its largest distance from the funnel in the norm [m]":
            '0.0249, 0.042, 0.0353, 0.0262; yes; 0.0000',
        "pure pursuit, look-ahead 0.5 m: the footprint's largest offset from the centre line [m]; touches; alarms, restarts; the largest torque of a wheel's drive [N m]":
            '0.419; nothing; 0, 0; 20.6',
        "pure pursuit, look-ahead 0.5 m: the alarm check's largest distance of the estimate from the initial set, in the norm [m]; at [s]":
            '0.0000; 0.0',
        "pure pursuit, look-ahead 0.5 m: (S5), before any alarm: the measurement's largest error at the period boundaries, in the norm [m], at [s]; within varpi":
            '0.0232, at 14.4; yes',
        "pure pursuit, look-ahead 0.5 m: before any alarm, the state's largest offset [m] and heading error [rad], signed, each at [s], s [m]":
            '-0.0353 at 15.54, 14.86; +0.0262 at 10.70, 10.01',
        "pure pursuit, look-ahead 0.5 m: before any alarm, the estimate's largest distance from the route's direction [rad], at [s], s [m]":
            '0.0295, at 10.70, 10.01',
        "pure pursuit, look-ahead 0.5 m: before any alarm, the estimate's largest error in heading [rad], at [s], s [m]":
            '0.0086, at 14.22, 13.54',
        "pure pursuit, look-ahead 0.5 m: the plan in force at the end: the dock's reserve [m]; the least clearance of the footprint in its funnel [m]":
            '0.1063; 0.1224',
        "pure pursuit, look-ahead 0.5 m: before any alarm, over the simulator's steps, the largest torque of a wheel's drive [N m], in the update ending at [s], s [m]; the steps with a drive at its limit; the updates with such steps, first and last ending at [s]":
            '20.635, at 10.62, 9.93; 0',
        'pure pursuit, look-ahead 0.75 m: front in the dock zone at [s]; at rest there from [s]; level of the plan in force at the end [m]; clearance of the footprint from the dock, least and at the end [m]':
            '24.80; 25.58; 0.10; 0.1948, 0.1956',
        "pure pursuit, look-ahead 0.75 m: the estimate's largest error along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0230, 0.029, 0.0202, 0.0085',
        "pure pursuit, look-ahead 0.75 m: the estimate's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0074, 0.056, 0.0289, 0.0471',
        'pure pursuit, look-ahead 0.75 m: the bounds of the estimate and of the feedback not kept':
            'F_E',
        "pure pursuit, look-ahead 0.75 m: the state's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]; within the funnel; its largest distance from the funnel in the norm [m]":
            '0.0246, 0.042, 0.0491, 0.0438; yes; 0.0000',
        "pure pursuit, look-ahead 0.75 m: the footprint's largest offset from the centre line [m]; touches; alarms, restarts; the largest torque of a wheel's drive [N m]":
            '0.424; nothing; 0, 0; 20.1',
        "pure pursuit, look-ahead 0.75 m: the alarm check's largest distance of the estimate from the initial set, in the norm [m]; at [s]":
            '0.0000; 0.0',
        "pure pursuit, look-ahead 0.75 m: (S5), before any alarm: the measurement's largest error at the period boundaries, in the norm [m], at [s]; within varpi":
            '0.0230, at 14.4; yes',
        "pure pursuit, look-ahead 0.75 m: before any alarm, the state's largest offset [m] and heading error [rad], signed, each at [s], s [m]":
            '-0.0491 at 15.68, 15.00; +0.0438 at 10.70, 10.01',
        "pure pursuit, look-ahead 0.75 m: before any alarm, the estimate's largest distance from the route's direction [rad], at [s], s [m]":
            '0.0471, at 10.70, 10.01',
        "pure pursuit, look-ahead 0.75 m: before any alarm, the estimate's largest error in heading [rad], at [s], s [m]":
            '0.0085, at 14.22, 13.54',
        "pure pursuit, look-ahead 0.75 m: the plan in force at the end: the dock's reserve [m]; the least clearance of the footprint in its funnel [m]":
            '0.1063; 0.1224',
        "pure pursuit, look-ahead 0.75 m: before any alarm, over the simulator's steps, the largest torque of a wheel's drive [N m], in the update ending at [s], s [m]; the steps with a drive at its limit; the updates with such steps, first and last ending at [s]":
            '20.114, at 10.58, 9.89; 0',
        'pure pursuit, look-ahead 1 m: front in the dock zone at [s]; at rest there from [s]; level of the plan in force at the end [m]; clearance of the footprint from the dock, least and at the end [m]':
            '24.80; 25.58; 0.10; 0.1949, 0.1958',
        "pure pursuit, look-ahead 1 m: the estimate's largest error along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0222, 0.029, 0.0196, 0.0082',
        "pure pursuit, look-ahead 1 m: the estimate's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0074, 0.056, 0.0492, 0.0641',
        'pure pursuit, look-ahead 1 m: the bounds of the estimate and of the feedback not kept':
            'F_E',
        "pure pursuit, look-ahead 1 m: the state's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]; within the funnel; its largest distance from the funnel in the norm [m]":
            '0.0238, 0.042, 0.0688, 0.0610; yes; 0.0000',
        "pure pursuit, look-ahead 1 m: the footprint's largest offset from the centre line [m]; touches; alarms, restarts; the largest torque of a wheel's drive [N m]":
            '0.435; nothing; 0, 0; 19.9',
        "pure pursuit, look-ahead 1 m: the alarm check's largest distance of the estimate from the initial set, in the norm [m]; at [s]":
            '0.0000; 0.0',
        "pure pursuit, look-ahead 1 m: (S5), before any alarm: the measurement's largest error at the period boundaries, in the norm [m], at [s]; within varpi":
            '0.0222, at 14.6; yes',
        "pure pursuit, look-ahead 1 m: before any alarm, the state's largest offset [m] and heading error [rad], signed, each at [s], s [m]":
            '-0.0688 at 15.78, 15.10; +0.0610 at 10.70, 10.01',
        "pure pursuit, look-ahead 1 m: before any alarm, the estimate's largest distance from the route's direction [rad], at [s], s [m]":
            '0.0641, at 10.70, 10.01',
        "pure pursuit, look-ahead 1 m: before any alarm, the estimate's largest error in heading [rad], at [s], s [m]":
            '0.0082, at 14.22, 13.54',
        "pure pursuit, look-ahead 1 m: the plan in force at the end: the dock's reserve [m]; the least clearance of the footprint in its funnel [m]":
            '0.1063; 0.1224',
        "pure pursuit, look-ahead 1 m: before any alarm, over the simulator's steps, the largest torque of a wheel's drive [N m], in the update ending at [s], s [m]; the steps with a drive at its limit; the updates with such steps, first and last ending at [s]":
            '19.906, at 10.68, 9.99; 0',
        'pure pursuit, look-ahead 1.5 m: front in the dock zone at [s]; at rest there from [s]; level of the plan in force at the end [m]; clearance of the footprint from the dock, least and at the end [m]':
            '24.80; 25.58; 0.10; 0.1971, 0.1980',
        "pure pursuit, look-ahead 1.5 m: the estimate's largest error along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0197, 0.029, 0.0188, 0.0074',
        "pure pursuit, look-ahead 1.5 m: the estimate's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0112, 0.056, 0.1117, 0.0976',
        'pure pursuit, look-ahead 1.5 m: the bounds of the estimate and of the feedback not kept':
            'F_D, F_E',
        "pure pursuit, look-ahead 1.5 m: the state's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]; within the funnel; its largest distance from the funnel in the norm [m]":
            '0.0226, 0.042, 0.1305, 0.0947; no; 0.0594',
        "pure pursuit, look-ahead 1.5 m: the footprint's largest offset from the centre line [m]; touches; alarms, restarts; the largest torque of a wheel's drive [N m]":
            '0.493; nothing; 0, 0; 19.8',
        "pure pursuit, look-ahead 1.5 m: the alarm check's largest distance of the estimate from the initial set, in the norm [m]; at [s]":
            '0.0345; 10.8',
        "pure pursuit, look-ahead 1.5 m: (S5), before any alarm: the measurement's largest error at the period boundaries, in the norm [m], at [s]; within varpi":
            '0.0197, at 14.4; yes',
        "pure pursuit, look-ahead 1.5 m: before any alarm, the state's largest offset [m] and heading error [rad], signed, each at [s], s [m]":
            '-0.1305 at 16.04, 15.35; +0.0947 at 10.70, 10.01',
        "pure pursuit, look-ahead 1.5 m: before any alarm, the estimate's largest distance from the route's direction [rad], at [s], s [m]":
            '0.0976, at 10.70, 10.01',
        'pure pursuit, look-ahead 1.5 m: before any alarm, the state outside its funnel: episodes of consecutive updates; the first four, from [s] to [s], s from [m] to [m], the largest distance in the norm [mm]; the share of the updates with the state outside, over the run and from the first exit to the last':
            '2; 10.52 to 10.88, 9.83 to 10.19, 59.4; 15.28 to 15.50, 14.59 to 14.81, 33.7; 0.02, 0.12',
        "pure pursuit, look-ahead 1.5 m: the period boundaries at which the estimate or the state lay outside the initial set: number; the largest distance of the estimate and of the state [mm]; the first four, instant [s], the estimate's and the state's distance [mm]":
            '3; 34.5, 33.7; 10.6, 31.4, 30.8; 10.8, 34.5, 27.8; 15.4, 22.4, 33.7',
        "pure pursuit, look-ahead 1.5 m: before any alarm, the estimate's largest error in heading [rad], at [s], s [m]":
            '0.0074, at 14.22, 13.54',
        "pure pursuit, look-ahead 1.5 m: the plan in force at the end: the dock's reserve [m]; the least clearance of the footprint in its funnel [m]":
            '0.1063; 0.1224',
        "pure pursuit, look-ahead 1.5 m: before any alarm, over the simulator's steps, the largest torque of a wheel's drive [N m], in the update ending at [s], s [m]; the steps with a drive at its limit; the updates with such steps, first and last ending at [s]":
            '19.804, at 10.68, 9.99; 0',
        'pure pursuit, look-ahead 1.5 m, stops on asphalt: label [m/s^2]; beyond the commanded rest point at 0.25, 0.5, 0.75, 1 m/s [mm]':
            '4; +1.6, +3.2, +10.5, +13.5',
        "pure pursuit, look-ahead 1.5 m, stops on asphalt: the estimate's error at rest at 0.25, 0.5, 0.75, 1 m/s [mm]":
            '-0.0, +0.0, +3.8, +5.5',
        "pure pursuit, look-ahead 1.5 m, stops on asphalt: over the four stops, the estimate's largest error along [mm]; its largest distance from the commanded arc length [mm] and speed [mm/s]; the bounds not kept":
            '8.6; 14.0, 125.1; none',
        "pure pursuit, look-ahead 1.5 m, stops on asphalt: over the four stops, the state's largest distance from the commanded arc length [mm] and speed [mm/s]; within the funnel; its largest distance from it in the norm [mm]":
            '21.4, 88.7; yes; 0.0',
        "pure pursuit, look-ahead 1.5 m, stops on asphalt: over the four stops, the estimate's largest error in speed [mm/s]; over the simulator's steps, the largest torque of a wheel's drive [N m] and the steps with a drive at its limit":
            '49.0; 30.0, 15',
        "pure pursuit, look-ahead 1.5 m, stops on asphalt: over the four stops, (S5): the measurement's largest error at the ends of the periods, in the norm [mm]; within varpi":
            '22.9; yes',
        'pure pursuit, look-ahead 1.5 m, stops on gravel: label [m/s^2]; beyond the commanded rest point at 0.25, 0.5, 0.75, 1 m/s [mm]':
            '3.53; +1.6, +3.7, +8.5, +10.1',
        "pure pursuit, look-ahead 1.5 m, stops on gravel: the estimate's error at rest at 0.25, 0.5, 0.75, 1 m/s [mm]":
            '-0.0, +0.5, +4.9, +4.9',
        "pure pursuit, look-ahead 1.5 m, stops on gravel: over the four stops, the estimate's largest error along [mm]; its largest distance from the commanded arc length [mm] and speed [mm/s]; the bounds not kept":
            '8.6; 10.0, 109.6; none',
        "pure pursuit, look-ahead 1.5 m, stops on gravel: over the four stops, the state's largest distance from the commanded arc length [mm] and speed [mm/s]; within the funnel; its largest distance from it in the norm [mm]":
            '17.4, 79.4; yes; 0.0',
        "pure pursuit, look-ahead 1.5 m, stops on gravel: over the four stops, the estimate's largest error in speed [mm/s]; over the simulator's steps, the largest torque of a wheel's drive [N m] and the steps with a drive at its limit":
            '42.5; 30.0, 16',
        "pure pursuit, look-ahead 1.5 m, stops on gravel: over the four stops, (S5): the measurement's largest error at the ends of the periods, in the norm [mm]; within varpi":
            '20.7; yes',
        'pure pursuit, look-ahead 1.5 m, stops on wet grass: label [m/s^2]; beyond the commanded rest point at 0.25, 0.5, 0.75, 1 m/s [mm]':
            '2.06; +1.6, +3.7, +5.3, +5.9',
        "pure pursuit, look-ahead 1.5 m, stops on wet grass: the estimate's error at rest at 0.25, 0.5, 0.75, 1 m/s [mm]":
            '-0.0, +1.8, +1.8, +2.6',
        "pure pursuit, look-ahead 1.5 m, stops on wet grass: over the four stops, the estimate's largest error along [mm]; its largest distance from the commanded arc length [mm] and speed [mm/s]; the bounds not kept":
            '8.6; 7.0, 65.2; none',
        "pure pursuit, look-ahead 1.5 m, stops on wet grass: over the four stops, the state's largest distance from the commanded arc length [mm] and speed [mm/s]; within the funnel; its largest distance from it in the norm [mm]":
            '12.3, 44.8; yes; 0.0',
        "pure pursuit, look-ahead 1.5 m, stops on wet grass: over the four stops, the estimate's largest error in speed [mm/s]; over the simulator's steps, the largest torque of a wheel's drive [N m] and the steps with a drive at its limit":
            '29.5; 17.2, 0',
        "pure pursuit, look-ahead 1.5 m, stops on wet grass: over the four stops, (S5): the measurement's largest error at the ends of the periods, in the norm [mm]; within varpi":
            '14.7; yes',
        'pure pursuit, look-ahead 0.5 m, stops: asphalt: label [m/s^2]; beyond the commanded rest point at 0.25, 0.5, 0.75, 1 m/s [mm]':
            '4; +1.6, +3.2, +10.5, +13.5',
        "pure pursuit, look-ahead 0.5 m, stops: asphalt: the estimate's error at rest at 0.25, 0.5, 0.75, 1 m/s [mm]":
            '-0.0, +0.0, +3.8, +5.5',
        "pure pursuit, look-ahead 0.5 m, stops: asphalt: over the four stops, the estimate's largest error along [mm]; its largest distance from the commanded arc length [mm] and speed [mm/s]; the bounds not kept":
            '8.6; 14.0, 125.1; none',
        "pure pursuit, look-ahead 0.5 m, stops: asphalt: over the four stops, the state's largest distance from the commanded arc length [mm] and speed [mm/s]; within the funnel; its largest distance from it in the norm [mm]":
            '21.3, 88.7; yes; 0.0',
        "pure pursuit, look-ahead 0.5 m, stops: asphalt: over the four stops, the estimate's largest error in speed [mm/s]; over the simulator's steps, the largest torque of a wheel's drive [N m] and the steps with a drive at its limit":
            '49.1; 30.0, 15',
        "pure pursuit, look-ahead 0.5 m, stops: asphalt: over the four stops, (S5): the measurement's largest error at the ends of the periods, in the norm [mm]; within varpi":
            '23.2; yes',
        'pure pursuit, look-ahead 0.5 m, stops: gravel: label [m/s^2]; beyond the commanded rest point at 0.25, 0.5, 0.75, 1 m/s [mm]':
            '3.53; +1.6, +3.7, +8.6, +10.1',
        "pure pursuit, look-ahead 0.5 m, stops: gravel: the estimate's error at rest at 0.25, 0.5, 0.75, 1 m/s [mm]":
            '-0.0, +0.5, +4.9, +4.9',
        "pure pursuit, look-ahead 0.5 m, stops: gravel: over the four stops, the estimate's largest error along [mm]; its largest distance from the commanded arc length [mm] and speed [mm/s]; the bounds not kept":
            '8.6; 10.0, 109.6; none',
        "pure pursuit, look-ahead 0.5 m, stops: gravel: over the four stops, the state's largest distance from the commanded arc length [mm] and speed [mm/s]; within the funnel; its largest distance from it in the norm [mm]":
            '17.5, 79.4; yes; 0.0',
        "pure pursuit, look-ahead 0.5 m, stops: gravel: over the four stops, the estimate's largest error in speed [mm/s]; over the simulator's steps, the largest torque of a wheel's drive [N m] and the steps with a drive at its limit":
            '42.5; 30.0, 16',
        "pure pursuit, look-ahead 0.5 m, stops: gravel: over the four stops, (S5): the measurement's largest error at the ends of the periods, in the norm [mm]; within varpi":
            '20.7; yes',
        'pure pursuit, look-ahead 0.5 m, stops: wet grass: label [m/s^2]; beyond the commanded rest point at 0.25, 0.5, 0.75, 1 m/s [mm]':
            '2.06; +1.6, +3.7, +5.3, +5.9',
        "pure pursuit, look-ahead 0.5 m, stops: wet grass: the estimate's error at rest at 0.25, 0.5, 0.75, 1 m/s [mm]":
            '-0.0, +1.8, +1.8, +2.6',
        "pure pursuit, look-ahead 0.5 m, stops: wet grass: over the four stops, the estimate's largest error along [mm]; its largest distance from the commanded arc length [mm] and speed [mm/s]; the bounds not kept":
            '8.6; 7.0, 65.2; none',
        "pure pursuit, look-ahead 0.5 m, stops: wet grass: over the four stops, the state's largest distance from the commanded arc length [mm] and speed [mm/s]; within the funnel; its largest distance from it in the norm [mm]":
            '12.3, 44.8; yes; 0.0',
        "pure pursuit, look-ahead 0.5 m, stops: wet grass: over the four stops, the estimate's largest error in speed [mm/s]; over the simulator's steps, the largest torque of a wheel's drive [N m] and the steps with a drive at its limit":
            '29.5; 17.2, 0',
        "pure pursuit, look-ahead 0.5 m, stops: wet grass: over the four stops, (S5): the measurement's largest error at the ends of the periods, in the norm [mm]; within varpi":
            '14.7; yes',
        'pure pursuit, look-ahead 0.5 m, stops: wet grass, braking at 3 m/s^2, 0.87 mu g: label [m/s^2]; beyond the commanded rest point at 0.25, 0.5, 0.75, 1 m/s [mm]':
            '3; +1.6, +8.3, +13.9, +17.3',
        "pure pursuit, look-ahead 0.5 m, stops: wet grass, braking at 3 m/s^2, 0.87 mu g: the estimate's error at rest at 0.25, 0.5, 0.75, 1 m/s [mm]":
            '-0.0, +5.1, +11.0, +12.8',
        "pure pursuit, look-ahead 0.5 m, stops: wet grass, braking at 3 m/s^2, 0.87 mu g: over the four stops, the estimate's largest error along [mm]; its largest distance from the commanded arc length [mm] and speed [mm/s]; the bounds not kept":
            '14.6; 8.0, 90.6; none',
        "pure pursuit, look-ahead 0.5 m, stops: wet grass, braking at 3 m/s^2, 0.87 mu g: over the four stops, the state's largest distance from the commanded arc length [mm] and speed [mm/s]; within the funnel; its largest distance from it in the norm [mm]":
            '22.6, 75.5; yes; 0.0',
        "pure pursuit, look-ahead 0.5 m, stops: wet grass, braking at 3 m/s^2, 0.87 mu g: over the four stops, the estimate's largest error in speed [mm/s]; over the simulator's steps, the largest torque of a wheel's drive [N m] and the steps with a drive at its limit":
            '29.7; 27.0, 0',
        "pure pursuit, look-ahead 0.5 m, stops: wet grass, braking at 3 m/s^2, 0.87 mu g: over the four stops, (S5): the measurement's largest error at the ends of the periods, in the norm [mm]; within varpi":
            '14.9; yes',
        'pure pursuit, look-ahead 0.5 m, stops: wet grass in the shower, its label: label [m/s^2]; beyond the commanded rest point at 0.25, 0.5, 0.75, 1 m/s [mm]':
            '0.883; +0.1, +0.0, -0.9, -2.2',
        "pure pursuit, look-ahead 0.5 m, stops: wet grass in the shower, its label: the estimate's error at rest at 0.25, 0.5, 0.75, 1 m/s [mm]":
            '-0.9, -1.9, -2.6, -4.5',
        "pure pursuit, look-ahead 0.5 m, stops: wet grass in the shower, its label: over the four stops, the estimate's largest error along [mm]; its largest distance from the commanded arc length [mm] and speed [mm/s]; the bounds not kept":
            '4.6; 4.4, 31.4; none',
        "pure pursuit, look-ahead 0.5 m, stops: wet grass in the shower, its label: over the four stops, the state's largest distance from the commanded arc length [mm] and speed [mm/s]; within the funnel; its largest distance from it in the norm [mm]":
            '3.9, 24.1; yes; 0.0',
        "pure pursuit, look-ahead 0.5 m, stops: wet grass in the shower, its label: over the four stops, the estimate's largest error in speed [mm/s]; over the simulator's steps, the largest torque of a wheel's drive [N m] and the steps with a drive at its limit":
            '13.0; 7.9, 0',
        "pure pursuit, look-ahead 0.5 m, stops: wet grass in the shower, its label: over the four stops, (S5): the measurement's largest error at the ends of the periods, in the norm [mm]; within varpi":
            '6.5; yes',
        'pure pursuit, look-ahead 0.5 m, stops: wet grass in the shower, the global capability: label [m/s^2]; beyond the commanded rest point at 0.25, 0.5, 0.75, 1 m/s [mm]':
            '0.8; -0.3, -1.4, -1.7, -4.2',
        "pure pursuit, look-ahead 0.5 m, stops: wet grass in the shower, the global capability: the estimate's error at rest at 0.25, 0.5, 0.75, 1 m/s [mm]":
            '-1.4, -2.6, -3.7, -5.8',
        "pure pursuit, look-ahead 0.5 m, stops: wet grass in the shower, the global capability: over the four stops, the estimate's largest error along [mm]; its largest distance from the commanded arc length [mm] and speed [mm/s]; the bounds not kept":
            '5.9; 4.4, 31.4; none',
        "pure pursuit, look-ahead 0.5 m, stops: wet grass in the shower, the global capability: over the four stops, the state's largest distance from the commanded arc length [mm] and speed [mm/s]; within the funnel; its largest distance from it in the norm [mm]":
            '4.2, 24.1; yes; 0.0',
        "pure pursuit, look-ahead 0.5 m, stops: wet grass in the shower, the global capability: over the four stops, the estimate's largest error in speed [mm/s]; over the simulator's steps, the largest torque of a wheel's drive [N m] and the steps with a drive at its limit":
            '13.4; 7.9, 0',
        "pure pursuit, look-ahead 0.5 m, stops: wet grass in the shower, the global capability: over the four stops, (S5): the measurement's largest error at the ends of the periods, in the norm [mm]; within varpi":
            '6.6; yes',
        "pure pursuit, look-ahead 0.5 m, stops: wet grass in the shower, the map's label: label [m/s^2]; beyond the commanded rest point at 0.25, 0.5, 0.75, 1 m/s [mm]":
            '2.06; +4.0, +27.9, +65.5, +109.5',
        "pure pursuit, look-ahead 0.5 m, stops: wet grass in the shower, the map's label: the estimate's error at rest at 0.25, 0.5, 0.75, 1 m/s [mm]":
            '+2.4, +26.0, +61.6, +105.9',
        "pure pursuit, look-ahead 0.5 m, stops: wet grass in the shower, the map's label: over the four stops, the estimate's largest error along [mm]; its largest distance from the commanded arc length [mm] and speed [mm/s]; the bounds not kept":
            '105.9; 6.6, 58.3; E_S, E_U',
        "pure pursuit, look-ahead 0.5 m, stops: wet grass in the shower, the map's label: over the four stops, the state's largest distance from the commanded arc length [mm] and speed [mm/s]; within the funnel; its largest distance from it in the norm [mm]":
            '111.3, 268.5; no; 41.3',
        "pure pursuit, look-ahead 0.5 m, stops: wet grass in the shower, the map's label: over the four stops, the estimate's largest error in speed [mm/s]; over the simulator's steps, the largest torque of a wheel's drive [N m] and the steps with a drive at its limit":
            '252.8; 15.9, 0',
        "pure pursuit, look-ahead 0.5 m, stops: wet grass in the shower, the map's label: over the four stops, (S5): the measurement's largest error at the ends of the periods, in the norm [mm]; within varpi":
            '120.3; no',
        "pure pursuit, look-ahead 0.5 m: the stops of Table 13.6; the estimate's bounds kept, the feedback's and the funnel in each as with the feedback of Section 13.2":
            '7; True',
        'pure pursuit, look-ahead 0.5 m, fixed: at rest in the dock zone from [s]; level at the end [m]; clearance at the end [m]; the bounds of the estimate and of the feedback not kept; within the funnel, its largest distance from it in the norm [m]; alarms; touches':
            '25.82; 0.12; 0.2167; none; yes, 0.0000; 0; nothing',
        'pure pursuit, look-ahead 0.5 m, reactive: at rest in the dock zone from [s]; level at the end [m]; clearance at the end [m]; the bounds of the estimate and of the feedback not kept; within the funnel, its largest distance from it in the norm [m]; alarms; touches':
            '25.82; 0.12; 0.2167; none; yes, 0.0000; 0; nothing',
        'pure pursuit, look-ahead 0.5 m, shower, map: at rest in the dock zone from [s]; level at the end [m]; clearance at the end [m]; the bounds of the estimate and of the feedback not kept; within the funnel, its largest distance from it in the norm [m]; alarms; touches':
            '25.70; 0.10; 0.1078; E_S, E_U; no, 0.0375; 0; nothing',
        'pure pursuit, look-ahead 0.5 m, shower, forecast: at rest in the dock zone from [s]; level at the end [m]; clearance at the end [m]; the bounds of the estimate and of the feedback not kept; within the funnel, its largest distance from it in the norm [m]; alarms; touches':
            '25.80; 0.12; 0.2174; none; yes, 0.0000; 0; nothing',
        'pure pursuit, look-ahead 0.5 m: the crossings with the predictors of Section 13.2, dry and in the shower; the bounds kept, the funnel, the alarms and the touches in each as with the feedback of Section 13.2':
            '4; True',
        'pure pursuit, look-ahead 0.5 m, time step 1 ms: at rest in the dock zone from [s]; level at the end [m]; clearance at the end [m]; the bounds of the estimate and of the feedback not kept; within the funnel':
            '25.24; 0.10; 0.2035; none; yes',
        'pure pursuit, look-ahead 0.5 m, time step 0.25 ms: at rest in the dock zone from [s]; level at the end [m]; clearance at the end [m]; the bounds of the estimate and of the feedback not kept; within the funnel':
            '25.60; 0.10; 0.1910; none; yes',
        "pure pursuit, look-ahead 0.5 m, seeds 1 to 20: the estimate's largest error along [m], in speed [m/s], across [m], in heading [rad]; its largest distance from the commanded motion, the same; seeds that keep the estimate's bounds, the feedback's; seeds within the funnel; alarms; touches":
            '0.0311, 0.029, 0.0851, 0.0100; 0.0074, 0.056, 0.0146, 0.0300; 20, 20; 20; 0; 0',
        "pure pursuit, look-ahead 0.5 m, seeds 1 to 20: clearance at the end, least and largest [m]; the state's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]":
            '0.1858, 0.2245; 0.0338, 0.042, 0.0856, 0.0272',
        'husky.hurry: front in the dock zone at [s]; at rest there from [s]; level of the plan in force at the end [m]; clearance of the footprint from the dock, least and at the end [m]':
            '24.80; 25.58; 0.10; 0.1935, 0.1941',
        "husky.hurry: the estimate's largest error along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0233, 0.029, 0.0216, 0.0090',
        "husky.hurry: the estimate's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0074, 0.056, 0.0391, 0.0298',
        'husky.hurry: the bounds of the estimate and of the feedback not kept':
            'none',
        "husky.hurry: the state's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]; within the funnel; its largest distance from the funnel in the norm [m]":
            '0.0251, 0.042, 0.0600, 0.0310; yes; 0.0000',
        "husky.hurry: the footprint's largest offset from the centre line [m]; touches; alarms, restarts; the largest torque of a wheel's drive [N m]":
            '0.450; nothing; 0, 0; 29.5',
        "husky.hurry: the alarm check's largest distance of the estimate from the initial set, in the norm [m]; at [s]":
            '0.0000; 0.0',
        "husky.hurry: (S5), before any alarm: the measurement's largest error at the period boundaries, in the norm [m], at [s]; within varpi":
            '0.0233, at 14.2; yes',
        "husky.hurry: before any alarm, the state's largest offset [m] and heading error [rad], signed, each at [s], s [m]":
            '-0.0600 at 15.42, 14.74; +0.0310 at 15.54, 14.86',
        "husky.hurry: before any alarm, the estimate's largest distance from the route's direction [rad], at [s], s [m]":
            '0.0298, at 15.52, 14.84',
        "husky.hurry: before any alarm, the estimate's largest error in heading [rad], at [s], s [m]":
            '0.0090, at 14.22, 13.54',
        "husky.hurry: the plan in force at the end: the dock's reserve [m]; the least clearance of the footprint in its funnel [m]":
            '0.1063; 0.1224',
        "husky.hurry: before any alarm, over the simulator's steps, the largest torque of a wheel's drive [N m], in the update ending at [s], s [m]; the steps with a drive at its limit; the updates with such steps, first and last ending at [s]":
            '29.486, at 10.78, 10.09; 0',
        "husky.hurry: periods committing ambition 1, between 0 and 1, 0; the reference's":
            '122, 1, 47; 167, 1, 2',
        "husky.hurry: the plan committed in every period that of the reference (level, point of rest and head); the periods whose committed ambition differs, the reference's there and husky.hurry's":
            'True; 125 to 169 (45 periods); 1; 0',
        'husky.careful: front in the dock zone at [s]; at rest there from [s]; level of the plan in force at the end [m]; clearance of the footprint from the dock, least and at the end [m]':
            '48.64; 49.48; 0.11; 0.2058, 0.2078',
        "husky.careful: the estimate's largest error along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0187, 0.025, 0.0300, 0.0051',
        "husky.careful: the estimate's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0047, 0.034, 0.0097, 0.0106',
        'husky.careful: the bounds of the estimate and of the feedback not kept':
            'none',
        "husky.careful: the state's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]; within the funnel; its largest distance from the funnel in the norm [m]":
            '0.0196, 0.024, 0.0300, 0.0136; yes; 0.0000',
        "husky.careful: the footprint's largest offset from the centre line [m]; touches; alarms, restarts; the largest torque of a wheel's drive [N m]":
            '0.401; nothing; 0, 0; 23.2',
        "husky.careful: the alarm check's largest distance of the estimate from the initial set, in the norm [m]; at [s]":
            '0.0000; 0.0',
        "husky.careful: (S5), before any alarm: the measurement's largest error at the period boundaries, in the norm [m], at [s]; within varpi":
            '0.0187, at 28.8; yes',
        "husky.careful: before any alarm, the state's largest offset [m] and heading error [rad], signed, each at [s], s [m]":
            '+0.0300 at 49.20, 24.30; +0.0136 at 30.22, 14.90',
        "husky.careful: before any alarm, the estimate's largest distance from the route's direction [rad], at [s], s [m]":
            '0.0106, at 30.18, 14.88',
        "husky.careful: before any alarm, the estimate's largest error in heading [rad], at [s], s [m]":
            '0.0051, at 58.36, 24.30',
        "husky.careful: the plan in force at the end: the dock's reserve [m]; the least clearance of the footprint in its funnel [m]":
            '0.1170; 0.1346',
        "husky.careful: before any alarm, over the simulator's steps, the largest torque of a wheel's drive [N m], in the update ending at [s], s [m]; the steps with a drive at its limit; the updates with such steps, first and last ending at [s]":
            '23.159, at 20.52, 10.03; 0',
    },
    'a payload': {
        "payload: rolling radius on asphalt over 10 s at 1 m/s [m]; the estimator's; the difference [%]":
            '0.16496; 0.16499; -0.02',
        "payload: the torque at each wheel with which four wheels give the asphalt's label, r m 4 / 4 [N m]":
            '21.7',
        'payload: front in the dock zone at [s]; at rest there from [s]; level of the plan in force at the end [m]; clearance of the footprint from the dock, least and at the end [m]':
            '24.80; 25.26; 0.10; 0.1899, 0.1902',
        "payload: the estimate's largest error along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0303, 0.033, 0.0232, 0.0125',
        "payload: the estimate's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0083, 0.065, 0.0391, 0.0314',
        'payload: the bounds of the estimate and of the feedback not kept':
            'none',
        "payload: the state's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]; within the funnel; its largest distance from the funnel in the norm [m]":
            '0.0320, 0.052, 0.0615, 0.0355; yes; 0.0000',
        "payload: the footprint's largest offset from the centre line [m]; touches; alarms, restarts; the largest torque of a wheel's drive [N m]":
            '0.450; nothing; 0, 0; 30.0',
        "payload: the alarm check's largest distance of the estimate from the initial set, in the norm [m]; at [s]":
            '0.0000; 0.0',
        "payload: (S5), before any alarm: the measurement's largest error at the period boundaries, in the norm [m], at [s]; within varpi":
            '0.0303, at 14.6; yes',
        "payload: before any alarm, the state's largest offset [m] and heading error [rad], signed, each at [s], s [m]":
            '-0.0615 at 15.42, 14.74; -0.0355 at 10.86, 10.17',
        "payload: before any alarm, the estimate's largest distance from the route's direction [rad], at [s], s [m]":
            '0.0314, at 10.84, 10.15',
        "payload: before any alarm, the estimate's largest error in heading [rad], at [s], s [m]":
            '0.0125, at 14.22, 13.54',
        "payload: the plan in force at the end: the dock's reserve [m]; the least clearance of the footprint in its funnel [m]":
            '0.1063; 0.1224',
        "payload: before any alarm, over the simulator's steps, the largest torque of a wheel's drive [N m], in the update ending at [s], s [m]; the steps with a drive at its limit; the updates with such steps, first and last ending at [s]":
            '30.000, at 10.78, 10.09; 254; 9, 10.78, 10.94',
        'payload, stops on asphalt: label [m/s^2]; beyond the commanded rest point at 0.25, 0.5, 0.75, 1 m/s [mm]':
            '4; +1.4, +2.7, +21.8, +24.2',
        "payload, stops on asphalt: the estimate's error at rest at 0.25, 0.5, 0.75, 1 m/s [mm]":
            '-0.1, -0.1, +2.1, +2.6',
        "payload, stops on asphalt: over the four stops, the estimate's largest error along [mm]; its largest distance from the commanded arc length [mm] and speed [mm/s]; the bounds not kept":
            '8.2; 27.5, 185.3; F_S, F_U',
        "payload, stops on asphalt: over the four stops, the state's largest distance from the commanded arc length [mm] and speed [mm/s]; within the funnel; its largest distance from it in the norm [mm]":
            '31.8, 135.6; yes; 0.0',
        "payload, stops on asphalt: over the four stops, the estimate's largest error in speed [mm/s]; over the simulator's steps, the largest torque of a wheel's drive [N m] and the steps with a drive at its limit":
            '52.8; 30.0, 18',
        "payload, stops on asphalt: over the four stops, (S5): the measurement's largest error at the ends of the periods, in the norm [mm]; within varpi":
            '24.9; yes',
        'payload, stops on gravel: label [m/s^2]; beyond the commanded rest point at 0.25, 0.5, 0.75, 1 m/s [mm]':
            '3.53; +1.4, +3.1, +18.1, +18.0',
        "payload, stops on gravel: the estimate's error at rest at 0.25, 0.5, 0.75, 1 m/s [mm]":
            '-0.1, +0.2, +5.4, +5.4',
        "payload, stops on gravel: over the four stops, the estimate's largest error along [mm]; its largest distance from the commanded arc length [mm] and speed [mm/s]; the bounds not kept":
            '8.2; 18.5, 128.8; none',
        "payload, stops on gravel: over the four stops, the state's largest distance from the commanded arc length [mm] and speed [mm/s]; within the funnel; its largest distance from it in the norm [mm]":
            '26.0, 95.1; yes; 0.0',
        "payload, stops on gravel: over the four stops, the estimate's largest error in speed [mm/s]; over the simulator's steps, the largest torque of a wheel's drive [N m] and the steps with a drive at its limit":
            '46.8; 30.0, 16',
        "payload, stops on gravel: over the four stops, (S5): the measurement's largest error at the ends of the periods, in the norm [mm]; within varpi":
            '22.4; yes',
        'payload, stops on wet grass: label [m/s^2]; beyond the commanded rest point at 0.25, 0.5, 0.75, 1 m/s [mm]':
            '2.06; +1.4, +2.3, +3.7, +3.2',
        "payload, stops on wet grass: the estimate's error at rest at 0.25, 0.5, 0.75, 1 m/s [mm]":
            '-0.1, +1.0, +0.9, +0.8',
        "payload, stops on wet grass: over the four stops, the estimate's largest error along [mm]; its largest distance from the commanded arc length [mm] and speed [mm/s]; the bounds not kept":
            '8.2; 8.0, 78.7; none',
        "payload, stops on wet grass: over the four stops, the state's largest distance from the commanded arc length [mm] and speed [mm/s]; within the funnel; its largest distance from it in the norm [mm]":
            '13.0, 54.3; yes; 0.0',
        "payload, stops on wet grass: over the four stops, the estimate's largest error in speed [mm/s]; over the simulator's steps, the largest torque of a wheel's drive [N m] and the steps with a drive at its limit":
            '35.3; 23.5, 0',
        "payload, stops on wet grass: over the four stops, (S5): the measurement's largest error at the ends of the periods, in the norm [mm]; within varpi":
            '17.4; yes',
    },
    'less friction': {
        "wet grass of friction 0.35: the map's label as a share of mu g":
            '0.60',
        'wet grass of friction 0.35: label [m/s^2]; beyond the commanded rest point at 0.25, 0.5, 0.75, 1 m/s [mm]':
            '2.06; +1.6, +3.7, +5.3, +5.9',
        "wet grass of friction 0.35: the estimate's error at rest at 0.25, 0.5, 0.75, 1 m/s [mm]":
            '-0.0, +1.8, +1.8, +2.6',
        "wet grass of friction 0.35: over the four stops, the estimate's largest error along [mm]; its largest distance from the commanded arc length [mm] and speed [mm/s]; the bounds not kept":
            '8.6; 7.0, 65.2; none',
        "wet grass of friction 0.35: over the four stops, the state's largest distance from the commanded arc length [mm] and speed [mm/s]; within the funnel; its largest distance from it in the norm [mm]":
            '12.3, 44.8; yes; 0.0',
        "wet grass of friction 0.35: over the four stops, the estimate's largest error in speed [mm/s]; over the simulator's steps, the largest torque of a wheel's drive [N m] and the steps with a drive at its limit":
            '29.5; 17.3, 0',
        "wet grass of friction 0.35: over the four stops, (S5): the measurement's largest error at the ends of the periods, in the norm [mm]; within varpi":
            '14.7; yes',
        "wet grass of friction 0.3: the map's label as a share of mu g":
            '0.70',
        'wet grass of friction 0.3: label [m/s^2]; beyond the commanded rest point at 0.25, 0.5, 0.75, 1 m/s [mm]':
            '2.06; +1.6, +5.6, +7.7, +10.2',
        "wet grass of friction 0.3: the estimate's error at rest at 0.25, 0.5, 0.75, 1 m/s [mm]":
            '+0.0, +3.8, +4.2, +6.9',
        "wet grass of friction 0.3: over the four stops, the estimate's largest error along [mm]; its largest distance from the commanded arc length [mm] and speed [mm/s]; the bounds not kept":
            '8.7; 7.0, 64.6; none',
        "wet grass of friction 0.3: over the four stops, the state's largest distance from the commanded arc length [mm] and speed [mm/s]; within the funnel; its largest distance from it in the norm [mm]":
            '15.4, 45.9; yes; 0.0',
        "wet grass of friction 0.3: over the four stops, the estimate's largest error in speed [mm/s]; over the simulator's steps, the largest torque of a wheel's drive [N m] and the steps with a drive at its limit":
            '25.9; 17.4, 0',
        "wet grass of friction 0.3: over the four stops, (S5): the measurement's largest error at the ends of the periods, in the norm [mm]; within varpi":
            '13.0; yes',
        "wet grass of friction 0.25: the map's label as a share of mu g":
            '0.84',
        'wet grass of friction 0.25: label [m/s^2]; beyond the commanded rest point at 0.25, 0.5, 0.75, 1 m/s [mm]':
            '2.06; +1.8, +8.5, +13.1, +17.2',
        "wet grass of friction 0.25: the estimate's error at rest at 0.25, 0.5, 0.75, 1 m/s [mm]":
            '+0.2, +6.7, +9.6, +13.9',
        "wet grass of friction 0.25: over the four stops, the estimate's largest error along [mm]; its largest distance from the commanded arc length [mm] and speed [mm/s]; the bounds not kept":
            '15.0; 7.1, 63.3; none',
        "wet grass of friction 0.25: over the four stops, the state's largest distance from the commanded arc length [mm] and speed [mm/s]; within the funnel; its largest distance from it in the norm [mm]":
            '22.0, 49.4; yes; 0.0',
        "wet grass of friction 0.25: over the four stops, the estimate's largest error in speed [mm/s]; over the simulator's steps, the largest torque of a wheel's drive [N m] and the steps with a drive at its limit":
            '21.6; 17.8, 0',
        "wet grass of friction 0.25: over the four stops, (S5): the measurement's largest error at the ends of the periods, in the norm [mm]; within varpi":
            '14.7; yes',
        "wet grass of friction 0.22: the map's label as a share of mu g":
            '0.95',
        'wet grass of friction 0.22: label [m/s^2]; beyond the commanded rest point at 0.25, 0.5, 0.75, 1 m/s [mm]':
            '2.06; +2.2, +11.0, +18.0, +23.6',
        "wet grass of friction 0.22: the estimate's error at rest at 0.25, 0.5, 0.75, 1 m/s [mm]":
            '+0.6, +9.1, +14.5, +20.3',
        "wet grass of friction 0.22: over the four stops, the estimate's largest error along [mm]; its largest distance from the commanded arc length [mm] and speed [mm/s]; the bounds not kept":
            '21.3; 7.1, 62.1; none',
        "wet grass of friction 0.22: over the four stops, the state's largest distance from the commanded arc length [mm] and speed [mm/s]; within the funnel; its largest distance from it in the norm [mm]":
            '28.1, 56.6; yes; 0.0',
        "wet grass of friction 0.22: over the four stops, the estimate's largest error in speed [mm/s]; over the simulator's steps, the largest torque of a wheel's drive [N m] and the steps with a drive at its limit":
            '18.9; 18.1, 0',
        "wet grass of friction 0.22: over the four stops, (S5): the measurement's largest error at the ends of the periods, in the norm [mm]; within varpi":
            '20.4; yes',
        "wet grass of friction 0.2: the map's label as a share of mu g":
            '1.05',
        'wet grass of friction 0.2: label [m/s^2]; beyond the commanded rest point at 0.25, 0.5, 0.75, 1 m/s [mm]':
            '2.06; +2.7, +13.8, +24.6, +36.6',
        "wet grass of friction 0.2: the estimate's error at rest at 0.25, 0.5, 0.75, 1 m/s [mm]":
            '+1.2, +12.0, +21.1, +33.3',
        "wet grass of friction 0.2: over the four stops, the estimate's largest error along [mm]; its largest distance from the commanded arc length [mm] and speed [mm/s]; the bounds not kept":
            '33.9; 7.0, 61.1; none',
        "wet grass of friction 0.2: over the four stops, the state's largest distance from the commanded arc length [mm] and speed [mm/s]; within the funnel; its largest distance from it in the norm [mm]":
            '40.2, 93.4; yes; 0.0',
        "wet grass of friction 0.2: over the four stops, the estimate's largest error in speed [mm/s]; over the simulator's steps, the largest torque of a wheel's drive [N m] and the steps with a drive at its limit":
            '68.5; 18.0, 0',
        "wet grass of friction 0.2: over the four stops, (S5): the measurement's largest error at the ends of the periods, in the norm [mm]; within varpi":
            '33.4; yes',
        "wet grass of friction 0.19: the map's label as a share of mu g":
            '1.11',
        'wet grass of friction 0.19: label [m/s^2]; beyond the commanded rest point at 0.25, 0.5, 0.75, 1 m/s [mm]':
            '2.06; +3.0, +15.9, +30.8, +47.1',
        "wet grass of friction 0.19: the estimate's error at rest at 0.25, 0.5, 0.75, 1 m/s [mm]":
            '+1.4, +14.0, +27.2, +43.8',
        "wet grass of friction 0.19: over the four stops, the estimate's largest error along [mm]; its largest distance from the commanded arc length [mm] and speed [mm/s]; the bounds not kept":
            '44.1; 6.9, 60.6; E_U',
        "wet grass of friction 0.19: over the four stops, the state's largest distance from the commanded arc length [mm] and speed [mm/s]; within the funnel; its largest distance from it in the norm [mm]":
            '50.1, 127.0; yes; 0.0',
        "wet grass of friction 0.19: over the four stops, the estimate's largest error in speed [mm/s]; over the simulator's steps, the largest torque of a wheel's drive [N m] and the steps with a drive at its limit":
            '104.0; 17.6, 0',
        "wet grass of friction 0.19: over the four stops, (S5): the measurement's largest error at the ends of the periods, in the norm [mm]; within varpi":
            '49.8; yes',
        "wet grass of friction 0.18: the map's label as a share of mu g":
            '1.17',
        'wet grass of friction 0.18: label [m/s^2]; beyond the commanded rest point at 0.25, 0.5, 0.75, 1 m/s [mm]':
            '2.06; +3.3, +18.3, +38.2, +60.1',
        "wet grass of friction 0.18: the estimate's error at rest at 0.25, 0.5, 0.75, 1 m/s [mm]":
            '+1.7, +16.4, +34.6, +56.7',
        "wet grass of friction 0.18: over the four stops, the estimate's largest error along [mm]; its largest distance from the commanded arc length [mm] and speed [mm/s]; the bounds not kept":
            '56.7; 6.8, 60.1; E_S, E_U',
        "wet grass of friction 0.18: over the four stops, the state's largest distance from the commanded arc length [mm] and speed [mm/s]; within the funnel; its largest distance from it in the norm [mm]":
            '62.4, 161.5; yes; 0.0',
        "wet grass of friction 0.18: over the four stops, the estimate's largest error in speed [mm/s]; over the simulator's steps, the largest torque of a wheel's drive [N m] and the steps with a drive at its limit":
            '140.3; 17.2, 0',
        "wet grass of friction 0.18: over the four stops, (S5): the measurement's largest error at the ends of the periods, in the norm [mm]; within varpi":
            '67.0; no',
        "wet grass of friction 0.17: the map's label as a share of mu g":
            '1.24',
        'wet grass of friction 0.17: label [m/s^2]; beyond the commanded rest point at 0.25, 0.5, 0.75, 1 m/s [mm]':
            '2.06; +3.5, +21.0, +46.3, +74.8',
        "wet grass of friction 0.17: the estimate's error at rest at 0.25, 0.5, 0.75, 1 m/s [mm]":
            '+1.9, +19.1, +42.6, +71.3',
        "wet grass of friction 0.17: over the four stops, the estimate's largest error along [mm]; its largest distance from the commanded arc length [mm] and speed [mm/s]; the bounds not kept":
            '71.3; 6.7, 59.5; E_S, E_U',
        "wet grass of friction 0.17: over the four stops, the state's largest distance from the commanded arc length [mm] and speed [mm/s]; within the funnel; its largest distance from it in the norm [mm]":
            '76.9, 196.7; no; 6.9',
        "wet grass of friction 0.17: over the four stops, the estimate's largest error in speed [mm/s]; over the simulator's steps, the largest torque of a wheel's drive [N m] and the steps with a drive at its limit":
            '177.4; 16.8, 0',
        "wet grass of friction 0.17: over the four stops, (S5): the measurement's largest error at the ends of the periods, in the norm [mm]; within varpi":
            '84.6; no',
        "wet grass of friction 0.16: the map's label as a share of mu g":
            '1.31',
        'wet grass of friction 0.16: label [m/s^2]; beyond the commanded rest point at 0.25, 0.5, 0.75, 1 m/s [mm]':
            '2.06; +3.8, +24.1, +55.3, +91.1',
        "wet grass of friction 0.16: the estimate's error at rest at 0.25, 0.5, 0.75, 1 m/s [mm]":
            '+2.1, +22.2, +51.5, +87.6',
        "wet grass of friction 0.16: over the four stops, the estimate's largest error along [mm]; its largest distance from the commanded arc length [mm] and speed [mm/s]; the bounds not kept":
            '87.6; 6.7, 58.9; E_S, E_U',
        "wet grass of friction 0.16: over the four stops, the state's largest distance from the commanded arc length [mm] and speed [mm/s]; within the funnel; its largest distance from it in the norm [mm]":
            '93.1, 232.4; no; 23.1',
        "wet grass of friction 0.16: over the four stops, the estimate's largest error in speed [mm/s]; over the simulator's steps, the largest torque of a wheel's drive [N m] and the steps with a drive at its limit":
            '214.9; 16.4, 0',
        "wet grass of friction 0.16: over the four stops, (S5): the measurement's largest error at the ends of the periods, in the norm [mm]; within varpi":
            '102.4; no',
        "wet grass of friction 0.15: the map's label as a share of mu g":
            '1.40',
        'wet grass of friction 0.15: label [m/s^2]; beyond the commanded rest point at 0.25, 0.5, 0.75, 1 m/s [mm]':
            '2.06; +4.0, +27.9, +65.5, +109.5',
        "wet grass of friction 0.15: the estimate's error at rest at 0.25, 0.5, 0.75, 1 m/s [mm]":
            '+2.4, +26.0, +61.6, +105.9',
        "wet grass of friction 0.15: over the four stops, the estimate's largest error along [mm]; its largest distance from the commanded arc length [mm] and speed [mm/s]; the bounds not kept":
            '105.9; 6.6, 58.3; E_S, E_U',
        "wet grass of friction 0.15: over the four stops, the state's largest distance from the commanded arc length [mm] and speed [mm/s]; within the funnel; its largest distance from it in the norm [mm]":
            '111.3, 268.5; no; 41.3',
        "wet grass of friction 0.15: over the four stops, the estimate's largest error in speed [mm/s]; over the simulator's steps, the largest torque of a wheel's drive [N m] and the steps with a drive at its limit":
            '252.8; 15.9, 0',
        "wet grass of friction 0.15: over the four stops, (S5): the measurement's largest error at the ends of the periods, in the norm [mm]; within varpi":
            '120.3; no',
        "less friction: down the sweep, the last friction at which every bound was kept; at which the state stayed in its funnel; whether no bound or funnel kept further down; the friction at which the map's label is mu g":
            '0.2; 0.18; True; 0.210',
        'crossing, wet grass of friction 0.2: front in the dock zone at [s]; at rest there from [s]; level of the plan in force at the end [m]; clearance of the footprint from the dock, least and at the end [m]':
            '24.80; 25.34; 0.10; 0.1661, 0.1671',
        "crossing, wet grass of friction 0.2: the estimate's largest error along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0429, 0.062, 0.0216, 0.0090',
        "crossing, wet grass of friction 0.2: the estimate's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0074, 0.051, 0.0391, 0.0298',
        'crossing, wet grass of friction 0.2: the bounds of the estimate and of the feedback not kept':
            'none',
        "crossing, wet grass of friction 0.2: the state's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]; within the funnel; its largest distance from the funnel in the norm [m]":
            '0.0495, 0.086, 0.0600, 0.0310; yes; 0.0000',
        "crossing, wet grass of friction 0.2: the footprint's largest offset from the centre line [m]; touches; alarms, restarts; the largest torque of a wheel's drive [N m]":
            '0.450; nothing; 0, 0; 29.5',
        "crossing, wet grass of friction 0.2: the alarm check's largest distance of the estimate from the initial set, in the norm [m]; at [s]":
            '0.0000; 0.0',
        "crossing, wet grass of friction 0.2: (S5), before any alarm: the measurement's largest error at the period boundaries, in the norm [m], at [s]; within varpi":
            '0.0427, at 33.6; yes',
        "crossing, wet grass of friction 0.2: before any alarm, the state's largest offset [m] and heading error [rad], signed, each at [s], s [m]":
            '-0.0600 at 15.42, 14.74; +0.0310 at 15.54, 14.86',
        "crossing, wet grass of friction 0.2: before any alarm, the estimate's largest distance from the route's direction [rad], at [s], s [m]":
            '0.0298, at 15.52, 14.84',
        "crossing, wet grass of friction 0.2: before any alarm, the estimate's largest error in heading [rad], at [s], s [m]":
            '0.0090, at 14.22, 13.54',
        "crossing, wet grass of friction 0.2: the plan in force at the end: the dock's reserve [m]; the least clearance of the footprint in its funnel [m]":
            '0.1063; 0.1224',
        "crossing, wet grass of friction 0.2: before any alarm, over the simulator's steps, the largest torque of a wheel's drive [N m], in the update ending at [s], s [m]; the steps with a drive at its limit; the updates with such steps, first and last ending at [s]":
            '29.486, at 10.78, 10.09; 0',
        'crossing, wet grass of friction 0.15: front in the dock zone at [s]; at rest there from [s]; level of the plan in force at the end [m]; clearance of the footprint from the dock, least and at the end [m]':
            '24.80; 25.70; 0.10; 0.1051, 0.1067',
        "crossing, wet grass of friction 0.15: the estimate's largest error along [m], in speed [m/s], across [m], in heading [rad]":
            '0.1029, 0.243, 0.0216, 0.0090',
        "crossing, wet grass of friction 0.15: the estimate's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0070, 0.048, 0.0391, 0.0298',
        'crossing, wet grass of friction 0.15: the bounds of the estimate and of the feedback not kept':
            'E_S, E_U',
        "crossing, wet grass of friction 0.15: the state's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]; within the funnel; its largest distance from the funnel in the norm [m]":
            '0.1078, 0.258, 0.0600, 0.0310; no; 0.0378',
        "crossing, wet grass of friction 0.15: the footprint's largest offset from the centre line [m]; touches; alarms, restarts; the largest torque of a wheel's drive [N m]":
            '0.450; nothing; 0, 0; 29.5',
        "crossing, wet grass of friction 0.15: the alarm check's largest distance of the estimate from the initial set, in the norm [m]; at [s]":
            '0.0000; 0.0',
        "crossing, wet grass of friction 0.15: (S5), before any alarm: the measurement's largest error at the period boundaries, in the norm [m], at [s]; within varpi":
            '0.1156, at 25.2; no',
        "crossing, wet grass of friction 0.15: before any alarm, the state's largest offset [m] and heading error [rad], signed, each at [s], s [m]":
            '-0.0600 at 15.42, 14.74; +0.0310 at 15.54, 14.86',
        "crossing, wet grass of friction 0.15: before any alarm, the estimate's largest distance from the route's direction [rad], at [s], s [m]":
            '0.0298, at 15.52, 14.84',
        'crossing, wet grass of friction 0.15: before any alarm, the state outside its funnel: episodes of consecutive updates; the first four, from [s] to [s], s from [m] to [m], the largest distance in the norm [mm]; the share of the updates with the state outside, over the run and from the first exit to the last':
            '1; 25.18 to 34.00, 24.35 to 24.40, 37.8; 0.26, 1.00',
        "crossing, wet grass of friction 0.15: the period boundaries at which the estimate or the state lay outside the initial set: number; the largest distance of the estimate and of the state [mm]; the first four, instant [s], the estimate's and the state's distance [mm]":
            '45; 0.0, 37.4; 25.2, 0.0, 8.0; 25.4, 0.0, 37.4; 25.6, 0.0, 36.3; 25.8, 0.0, 36.0',
        "crossing, wet grass of friction 0.15: before any alarm, the estimate's largest error in heading [rad], at [s], s [m]":
            '0.0090, at 14.22, 13.54',
        "crossing, wet grass of friction 0.15: the plan in force at the end: the dock's reserve [m]; the least clearance of the footprint in its funnel [m]":
            '0.1063; 0.1224',
        "crossing, wet grass of friction 0.15: before any alarm, over the simulator's steps, the largest torque of a wheel's drive [N m], in the update ending at [s], s [m]; the steps with a drive at its limit; the updates with such steps, first and last ending at [s]":
            '29.486, at 10.78, 10.09; 0',
        "crossing, wet grass of friction 0.15: the crossing with the map in the shower of Section 13.2, in zone, rest, level, clearance, the estimate's error and the state's distances":
            'True',
    },
    'a weaker actuator': {
        'asphalt, torque limit 30 N m: the deceleration that four wheels at the limit give the robot, 4 T / (r m) [m/s^2]':
            '9.09',
        'asphalt, torque limit 30 N m: label [m/s^2]; beyond the commanded rest point at 0.25, 0.5, 0.75, 1 m/s [mm]':
            '4; +1.6, +3.2, +10.5, +13.5',
        "asphalt, torque limit 30 N m: the estimate's error at rest at 0.25, 0.5, 0.75, 1 m/s [mm]":
            '-0.0, +0.0, +3.8, +5.5',
        "asphalt, torque limit 30 N m: over the four stops, the estimate's largest error along [mm]; its largest distance from the commanded arc length [mm] and speed [mm/s]; the bounds not kept":
            '8.6; 14.0, 125.1; none',
        "asphalt, torque limit 30 N m: over the four stops, the state's largest distance from the commanded arc length [mm] and speed [mm/s]; within the funnel; its largest distance from it in the norm [mm]":
            '21.3, 88.7; yes; 0.0',
        "asphalt, torque limit 30 N m: over the four stops, the estimate's largest error in speed [mm/s]; over the simulator's steps, the largest torque of a wheel's drive [N m] and the steps with a drive at its limit":
            '49.0; 30.0, 15',
        "asphalt, torque limit 30 N m: over the four stops, (S5): the measurement's largest error at the ends of the periods, in the norm [mm]; within varpi":
            '22.9; yes',
        'asphalt, torque limit 25 N m: the deceleration that four wheels at the limit give the robot, 4 T / (r m) [m/s^2]':
            '7.57',
        'asphalt, torque limit 25 N m: label [m/s^2]; beyond the commanded rest point at 0.25, 0.5, 0.75, 1 m/s [mm]':
            '4; +1.6, +3.2, +16.6, +20.1',
        "asphalt, torque limit 25 N m: the estimate's error at rest at 0.25, 0.5, 0.75, 1 m/s [mm]":
            '-0.0, +0.0, +4.6, +5.3',
        "asphalt, torque limit 25 N m: over the four stops, the estimate's largest error along [mm]; its largest distance from the commanded arc length [mm] and speed [mm/s]; the bounds not kept":
            '8.6; 19.1, 137.6; none',
        "asphalt, torque limit 25 N m: over the four stops, the state's largest distance from the commanded arc length [mm] and speed [mm/s]; within the funnel; its largest distance from it in the norm [mm]":
            '26.1, 100.5; yes; 0.0',
        "asphalt, torque limit 25 N m: over the four stops, the estimate's largest error in speed [mm/s]; over the simulator's steps, the largest torque of a wheel's drive [N m] and the steps with a drive at its limit":
            '46.7; 25.0, 17',
        "asphalt, torque limit 25 N m: over the four stops, (S5): the measurement's largest error at the ends of the periods, in the norm [mm]; within varpi":
            '20.6; yes',
        'asphalt, torque limit 20 N m: the deceleration that four wheels at the limit give the robot, 4 T / (r m) [m/s^2]':
            '6.06',
        'asphalt, torque limit 20 N m: label [m/s^2]; beyond the commanded rest point at 0.25, 0.5, 0.75, 1 m/s [mm]':
            '4; +1.6, +3.2, +28.3, +31.3',
        "asphalt, torque limit 20 N m: the estimate's error at rest at 0.25, 0.5, 0.75, 1 m/s [mm]":
            '-0.0, +0.0, +1.2, +1.6',
        "asphalt, torque limit 20 N m: over the four stops, the estimate's largest error along [mm]; its largest distance from the commanded arc length [mm] and speed [mm/s]; the bounds not kept":
            '8.6; 31.0, 210.2; F_S, F_U',
        "asphalt, torque limit 20 N m: over the four stops, the state's largest distance from the commanded arc length [mm] and speed [mm/s]; within the funnel; its largest distance from it in the norm [mm]":
            '34.0, 156.6; yes; 0.0',
        "asphalt, torque limit 20 N m: over the four stops, the estimate's largest error in speed [mm/s]; over the simulator's steps, the largest torque of a wheel's drive [N m] and the steps with a drive at its limit":
            '54.0; 20.0, 18',
        "asphalt, torque limit 20 N m: over the four stops, (S5): the measurement's largest error at the ends of the periods, in the norm [mm]; within varpi":
            '26.8; yes',
        'asphalt, torque limit 15 N m: the deceleration that four wheels at the limit give the robot, 4 T / (r m) [m/s^2]':
            '4.54',
        'asphalt, torque limit 15 N m: label [m/s^2]; beyond the commanded rest point at 0.25, 0.5, 0.75, 1 m/s [mm]':
            '4; +1.6, +12.3, +46.5, +53.8',
        "asphalt, torque limit 15 N m: the estimate's error at rest at 0.25, 0.5, 0.75, 1 m/s [mm]":
            '-0.0, +0.2, +0.3, +0.4',
        "asphalt, torque limit 15 N m: over the four stops, the estimate's largest error along [mm]; its largest distance from the commanded arc length [mm] and speed [mm/s]; the bounds not kept":
            '8.6; 53.3, 300.9; F_S, F_U',
        "asphalt, torque limit 15 N m: over the four stops, the state's largest distance from the commanded arc length [mm] and speed [mm/s]; within the funnel; its largest distance from it in the norm [mm]":
            '53.8, 244.1; yes; 0.0',
        "asphalt, torque limit 15 N m: over the four stops, the estimate's largest error in speed [mm/s]; over the simulator's steps, the largest torque of a wheel's drive [N m] and the steps with a drive at its limit":
            '59.1; 15.0, 31',
        "asphalt, torque limit 15 N m: over the four stops, (S5): the measurement's largest error at the ends of the periods, in the norm [mm]; within varpi":
            '28.4; yes',
        'asphalt, torque limit 13 N m: the deceleration that four wheels at the limit give the robot, 4 T / (r m) [m/s^2]':
            '3.94',
        'asphalt, torque limit 13 N m: label [m/s^2]; beyond the commanded rest point at 0.25, 0.5, 0.75, 1 m/s [mm]':
            '4; +1.6, +19.1, +54.9, +67.4',
        "asphalt, torque limit 13 N m: the estimate's error at rest at 0.25, 0.5, 0.75, 1 m/s [mm]":
            '-0.0, +0.2, +0.2, +0.4',
        "asphalt, torque limit 13 N m: over the four stops, the estimate's largest error along [mm]; its largest distance from the commanded arc length [mm] and speed [mm/s]; the bounds not kept":
            '8.6; 67.1, 338.3; F_S, F_U',
        "asphalt, torque limit 13 N m: over the four stops, the state's largest distance from the commanded arc length [mm] and speed [mm/s]; within the funnel; its largest distance from it in the norm [mm]":
            '67.4, 283.0; no; 16.5',
        "asphalt, torque limit 13 N m: over the four stops, the estimate's largest error in speed [mm/s]; over the simulator's steps, the largest torque of a wheel's drive [N m] and the steps with a drive at its limit":
            '58.7; 13.0, 32',
        "asphalt, torque limit 13 N m: over the four stops, (S5): the measurement's largest error at the ends of the periods, in the norm [mm]; within varpi":
            '27.7; yes',
        'asphalt, torque limit 10 N m: the deceleration that four wheels at the limit give the robot, 4 T / (r m) [m/s^2]':
            '3.03',
        'asphalt, torque limit 10 N m: label [m/s^2]; beyond the commanded rest point at 0.25, 0.5, 0.75, 1 m/s [mm]':
            '4; +1.6, +30.9, +68.2, +91.5',
        "asphalt, torque limit 10 N m: the estimate's error at rest at 0.25, 0.5, 0.75, 1 m/s [mm]":
            '-0.0, +0.2, +1.0, +3.5',
        "asphalt, torque limit 10 N m: over the four stops, the estimate's largest error along [mm]; its largest distance from the commanded arc length [mm] and speed [mm/s]; the bounds not kept":
            '8.6; 88.0, 390.9; F_S, F_U',
        "asphalt, torque limit 10 N m: over the four stops, the state's largest distance from the commanded arc length [mm] and speed [mm/s]; within the funnel; its largest distance from it in the norm [mm]":
            '91.5, 351.0; no; 50.5',
        "asphalt, torque limit 10 N m: over the four stops, the estimate's largest error in speed [mm/s]; over the simulator's steps, the largest torque of a wheel's drive [N m] and the steps with a drive at its limit":
            '46.8; 10.0, 352',
        "asphalt, torque limit 10 N m: over the four stops, (S5): the measurement's largest error at the ends of the periods, in the norm [mm]; within varpi":
            '20.3; yes',
        'asphalt: down the sweep, the last torque limit at which every bound was kept; at which the state stayed in its funnel; whether no bound or funnel kept further down; the torque limit at which four wheels give the label, r m 4 / 4 [N m]':
            '25; 15; True; 13.2',
        'crossing, torque limit 20 N m: front in the dock zone at [s]; at rest there from [s]; level of the plan in force at the end [m]; clearance of the footprint from the dock, least and at the end [m]':
            '24.80; 25.58; 0.10; 0.1908, 0.1915',
        "crossing, torque limit 20 N m: the estimate's largest error along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0267, 0.029, 0.0242, 0.0111',
        "crossing, torque limit 20 N m: the estimate's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0074, 0.056, 0.0390, 0.0298',
        'crossing, torque limit 20 N m: the bounds of the estimate and of the feedback not kept':
            'none',
        "crossing, torque limit 20 N m: the state's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]; within the funnel; its largest distance from the funnel in the norm [m]":
            '0.0281, 0.042, 0.0624, 0.0300; yes; 0.0000',
        "crossing, torque limit 20 N m: the footprint's largest offset from the centre line [m]; touches; alarms, restarts; the largest torque of a wheel's drive [N m]":
            '0.451; nothing; 0, 0; 20.0',
        "crossing, torque limit 20 N m: the alarm check's largest distance of the estimate from the initial set, in the norm [m]; at [s]":
            '0.0000; 0.0',
        "crossing, torque limit 20 N m: (S5), before any alarm: the measurement's largest error at the period boundaries, in the norm [m], at [s]; within varpi":
            '0.0267, at 14.2; yes',
        "crossing, torque limit 20 N m: before any alarm, the state's largest offset [m] and heading error [rad], signed, each at [s], s [m]":
            '-0.0624 at 15.42, 14.74; +0.0300 at 15.54, 14.86',
        "crossing, torque limit 20 N m: before any alarm, the estimate's largest distance from the route's direction [rad], at [s], s [m]":
            '0.0298, at 15.52, 14.84',
        "crossing, torque limit 20 N m: before any alarm, the estimate's largest error in heading [rad], at [s], s [m]":
            '0.0111, at 14.22, 13.54',
        "crossing, torque limit 20 N m: the plan in force at the end: the dock's reserve [m]; the least clearance of the footprint in its funnel [m]":
            '0.1063; 0.1224',
        "crossing, torque limit 20 N m: before any alarm, over the simulator's steps, the largest torque of a wheel's drive [N m], in the update ending at [s], s [m]; the steps with a drive at its limit; the updates with such steps, first and last ending at [s]":
            '20.000, at 10.74, 10.05; 8; 7, 10.74, 10.86',
        'crossing, torque limit 15 N m: front in the dock zone at [s]; at rest there from [s]; level of the plan in force at the end [m]; clearance of the footprint from the dock, least and at the end [m]':
            '24.80; 25.58; 0.10; 0.1852, 0.1858',
        "crossing, torque limit 15 N m: the estimate's largest error along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0318, 0.026, 0.0315, 0.0144',
        "crossing, torque limit 15 N m: the estimate's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0084, 0.056, 0.0386, 0.0450',
        'crossing, torque limit 15 N m: the bounds of the estimate and of the feedback not kept':
            'F_E',
        "crossing, torque limit 15 N m: the state's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]; within the funnel; its largest distance from the funnel in the norm [m]":
            '0.0334, 0.041, 0.0667, 0.0505; yes; 0.0000',
        "crossing, torque limit 15 N m: the footprint's largest offset from the centre line [m]; touches; alarms, restarts; the largest torque of a wheel's drive [N m]":
            '0.454; nothing; 0, 0; 15.0',
        "crossing, torque limit 15 N m: the alarm check's largest distance of the estimate from the initial set, in the norm [m]; at [s]":
            '0.0000; 0.0',
        "crossing, torque limit 15 N m: (S5), before any alarm: the measurement's largest error at the period boundaries, in the norm [m], at [s]; within varpi":
            '0.0318, at 14.2; yes',
        "crossing, torque limit 15 N m: before any alarm, the state's largest offset [m] and heading error [rad], signed, each at [s], s [m]":
            '-0.0667 at 15.44, 14.76; -0.0505 at 10.96, 10.27',
        "crossing, torque limit 15 N m: before any alarm, the estimate's largest distance from the route's direction [rad], at [s], s [m]":
            '0.0450, at 10.96, 10.27',
        "crossing, torque limit 15 N m: before any alarm, the estimate's largest error in heading [rad], at [s], s [m]":
            '0.0144, at 13.92, 13.25',
        "crossing, torque limit 15 N m: the plan in force at the end: the dock's reserve [m]; the least clearance of the footprint in its funnel [m]":
            '0.1063; 0.1224',
        "crossing, torque limit 15 N m: before any alarm, over the simulator's steps, the largest torque of a wheel's drive [N m], in the update ending at [s], s [m]; the steps with a drive at its limit; the updates with such steps, first and last ending at [s]":
            '15.000, at 10.74, 10.05; 587; 34, 10.74, 25.22',
        'crossing, torque limit 12 N m: front in the dock zone at [s]; at rest there from [s]; level of the plan in force at the end [m]; clearance of the footprint from the dock, least and at the end [m]':
            'never; never; none: the rest feedback; 12.6758, 15.9372',
        "crossing, torque limit 12 N m: the estimate's largest error along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0263, 0.865, 0.0188, 0.0101',
        "crossing, torque limit 12 N m: the estimate's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]":
            '0.1621, 0.905, 0.9314, 0.5464',
        'crossing, torque limit 12 N m: the bounds of the estimate and of the feedback not kept':
            'E_U, F_S, F_U, F_D, F_E',
        "crossing, torque limit 12 N m: the state's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]; within the funnel; its largest distance from the funnel in the norm [m]":
            '0.1775, 1.077, 0.9316, 0.5409; no; 0.9519',
        "crossing, torque limit 12 N m: the footprint's largest offset from the centre line [m]; touches; alarms, restarts; the largest torque of a wheel's drive [N m]":
            '1.449; pallets outside the turn at 12.60 s, s = 11.72 m; 1, the first at 11.0 s, 0; 12.0',
        "crossing, torque limit 12 N m: the alarm check's largest distance of the estimate from the initial set, in the norm [m]; at [s]":
            '0.0591; 11.0',
        "crossing, torque limit 12 N m: (S5), before any alarm: the measurement's largest error at the period boundaries, in the norm [m], at [s]; within varpi":
            '0.0087, at 11.0; yes',
        "crossing, torque limit 12 N m: before any alarm, the state's largest offset [m] and heading error [rad], signed, each at [s], s [m]":
            '-0.0101 at 11.00, 10.31; -0.0970 at 11.00, 10.31',
        "crossing, torque limit 12 N m: before any alarm, the estimate's largest distance from the route's direction [rad], at [s], s [m]":
            '0.0945, at 11.00, 10.31',
        'crossing, torque limit 12 N m: before any alarm, the state outside its funnel: episodes of consecutive updates; the first four, from [s] to [s], s from [m] to [m], the largest distance in the norm [mm]; the share of the updates with the state outside, over the run and from the first exit to the last':
            '1; 10.90 to 11.00, 10.21 to 10.31, 63.9; 0.01, 1.00',
        "crossing, torque limit 12 N m: the period boundaries at which the estimate or the state lay outside the initial set: number; the largest distance of the estimate and of the state [mm]; the first four, instant [s], the estimate's and the state's distance [mm]":
            '1; 59.1, 63.9; 11.0, 59.1, 63.9',
        'crossing, torque limit 12 N m: the footprint first outside the lane at [s], s [m]; after the first alarm by [s]; its offset from the centre line at the update of the first touch [m]':
            '11.82, 11.09; 0.82; 1.088',
        "crossing, torque limit 12 N m: before any alarm, the estimate's largest error in heading [rad], at [s], s [m]":
            '0.0024, at 11.00, 10.31',
        "crossing, torque limit 12 N m: before any alarm, over the simulator's steps, the largest torque of a wheel's drive [N m], in the update ending at [s], s [m]; the steps with a drive at its limit; the updates with such steps, first and last ending at [s]":
            '12.000, at 10.74, 10.05; 560; 14, 10.74, 11.00',
        'crossing, torque limit 12 N m: the first alarm: the estimate beyond the initial set along [m], in speed [m/s], across [m], in heading [rad]; in the norm [m]':
            '-0.0691, -0.250, -0.1457, 0.0295; 0.0591',
        "crossing, torque limit 12 N m: before the first alarm: the bounds of the estimate and of the feedback not kept; the state's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]; within the funnel; its largest distance from it in the norm [m]":
            'F_E; 0.0084, 0.027, 0.0101, 0.0970; no; 0.0639',
        'crossing, torque limit 12 N m: after the first alarm: the periods of the rest feedback; those in which the package offered a set to restart from; the reasons given in the others, with their number':
            '115; 0; no restart candidate while the robot moves: the stop of the rest feedback depends on the ground (115)',
        "crossing, torque limit 12 N m: between the first alarm and the update of the first touch: the robot's least and largest speed [m/s]; the share of the simulator's steps with a drive at its limit; at the controller's updates, the mean torque of the left and of the right pair [N m]; the feedback's yaw-rate command without its integral, least and largest, and with it, largest [rad/s]; the largest yaw rate that the drive controller passed [rad/s]":
            '0.997, 1.021; 1.00; -12.00, +12.00; 0.64, 2.46, 12.15; 2.00',
        "crossing, torque limit 12 N m: from the update of the first touch to the end: the robot's least and largest forward speed [m/s]; the footprint's largest offset from the centre line [m]; the true arc length at the end [m]":
            '-0.363, +0.126; 1.449; 8.47',
        'crossing, torque limit 10 N m: front in the dock zone at [s]; at rest there from [s]; level of the plan in force at the end [m]; clearance of the footprint from the dock, least and at the end [m]':
            'never; never; none: the rest feedback; 12.6779, 16.0590',
        "crossing, torque limit 10 N m: the estimate's largest error along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0220, 0.816, 0.0165, 0.0089',
        "crossing, torque limit 10 N m: the estimate's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]":
            '0.1609, 0.894, 0.7952, 0.5494',
        'crossing, torque limit 10 N m: the bounds of the estimate and of the feedback not kept':
            'E_U, F_S, F_U, F_D, F_E',
        "crossing, torque limit 10 N m: the state's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]; within the funnel; its largest distance from the funnel in the norm [m]":
            '0.1735, 1.057, 0.7956, 0.5452; no; 0.9603',
        "crossing, torque limit 10 N m: the footprint's largest offset from the centre line [m]; touches; alarms, restarts; the largest torque of a wheel's drive [N m]":
            '1.290; pallets outside the turn at 12.60 s, s = 11.72 m; 1, the first at 11.0 s, 0; 10.0',
        "crossing, torque limit 10 N m: the alarm check's largest distance of the estimate from the initial set, in the norm [m]; at [s]":
            '0.0607; 11.0',
        "crossing, torque limit 10 N m: (S5), before any alarm: the measurement's largest error at the period boundaries, in the norm [m], at [s]; within varpi":
            '0.0087, at 11.0; yes',
        "crossing, torque limit 10 N m: before any alarm, the state's largest offset [m] and heading error [rad], signed, each at [s], s [m]":
            '-0.0102 at 11.00, 10.31; -0.0978 at 11.00, 10.31',
        "crossing, torque limit 10 N m: before any alarm, the estimate's largest distance from the route's direction [rad], at [s], s [m]":
            '0.0954, at 11.00, 10.31',
        'crossing, torque limit 10 N m: before any alarm, the state outside its funnel: episodes of consecutive updates; the first four, from [s] to [s], s from [m] to [m], the largest distance in the norm [mm]; the share of the updates with the state outside, over the run and from the first exit to the last':
            '1; 10.90 to 11.00, 10.21 to 10.31, 65.5; 0.01, 1.00',
        "crossing, torque limit 10 N m: the period boundaries at which the estimate or the state lay outside the initial set: number; the largest distance of the estimate and of the state [mm]; the first four, instant [s], the estimate's and the state's distance [mm]":
            '1; 60.7, 65.5; 11.0, 60.7, 65.5',
        'crossing, torque limit 10 N m: the footprint first outside the lane at [s], s [m]; after the first alarm by [s]; its offset from the centre line at the update of the first touch [m]':
            '11.82, 11.09; 0.82; 1.093',
        "crossing, torque limit 10 N m: before any alarm, the estimate's largest error in heading [rad], at [s], s [m]":
            '0.0024, at 11.00, 10.31',
        "crossing, torque limit 10 N m: before any alarm, over the simulator's steps, the largest torque of a wheel's drive [N m], in the update ending at [s], s [m]; the steps with a drive at its limit; the updates with such steps, first and last ending at [s]":
            '10.000, at 10.72, 10.03; 588; 15, 10.72, 11.00',
        'crossing, torque limit 10 N m: the first alarm: the estimate beyond the initial set along [m], in speed [m/s], across [m], in heading [rad]; in the norm [m]':
            '-0.0691, -0.250, -0.1456, 0.0304; 0.0607',
        "crossing, torque limit 10 N m: before the first alarm: the bounds of the estimate and of the feedback not kept; the state's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]; within the funnel; its largest distance from it in the norm [m]":
            'F_E; 0.0084, 0.027, 0.0102, 0.0978; no; 0.0655',
        'crossing, torque limit 10 N m: after the first alarm: the periods of the rest feedback; those in which the package offered a set to restart from; the reasons given in the others, with their number':
            '115; 0; no restart candidate while the robot moves: the stop of the rest feedback depends on the ground (115)',
        "crossing, torque limit 10 N m: between the first alarm and the update of the first touch: the robot's least and largest speed [m/s]; the share of the simulator's steps with a drive at its limit; at the controller's updates, the mean torque of the left and of the right pair [N m]; the feedback's yaw-rate command without its integral, least and largest, and with it, largest [rad/s]; the largest yaw rate that the drive controller passed [rad/s]":
            '0.997, 1.021; 1.00; -10.00, +10.00; 0.64, 2.48, 12.27; 2.00',
        "crossing, torque limit 10 N m: from the update of the first touch to the end: the robot's least and largest forward speed [m/s]; the footprint's largest offset from the centre line [m]; the true arc length at the end [m]":
            '-0.368, +0.127; 1.290; 8.36',
    },
    'a late sensor': {
        'readings 0 ms late: front in the dock zone at [s]; at rest there from [s]; level of the plan in force at the end [m]; clearance of the footprint from the dock, least and at the end [m]':
            '24.80; 25.58; 0.10; 0.1935, 0.1941',
        "readings 0 ms late: the estimate's largest error along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0233, 0.029, 0.0216, 0.0090',
        "readings 0 ms late: the estimate's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0074, 0.056, 0.0391, 0.0298',
        'readings 0 ms late: the bounds of the estimate and of the feedback not kept':
            'none',
        "readings 0 ms late: the state's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]; within the funnel; its largest distance from the funnel in the norm [m]":
            '0.0251, 0.042, 0.0600, 0.0310; yes; 0.0000',
        "readings 0 ms late: the footprint's largest offset from the centre line [m]; touches; alarms, restarts; the largest torque of a wheel's drive [N m]":
            '0.450; nothing; 0, 0; 29.5',
        "readings 0 ms late: the alarm check's largest distance of the estimate from the initial set, in the norm [m]; at [s]":
            '0.0000; 0.0',
        "readings 0 ms late: (S5), before any alarm: the measurement's largest error at the period boundaries, in the norm [m], at [s]; within varpi":
            '0.0233, at 14.2; yes',
        "readings 0 ms late: before any alarm, the state's largest offset [m] and heading error [rad], signed, each at [s], s [m]":
            '-0.0600 at 15.42, 14.74; +0.0310 at 15.54, 14.86',
        "readings 0 ms late: before any alarm, the estimate's largest distance from the route's direction [rad], at [s], s [m]":
            '0.0298, at 15.52, 14.84',
        "readings 0 ms late: before any alarm, the estimate's largest error in heading [rad], at [s], s [m]":
            '0.0090, at 14.22, 13.54',
        "readings 0 ms late: the plan in force at the end: the dock's reserve [m]; the least clearance of the footprint in its funnel [m]":
            '0.1063; 0.1224',
        "readings 0 ms late: before any alarm, over the simulator's steps, the largest torque of a wheel's drive [N m], in the update ending at [s], s [m]; the steps with a drive at its limit; the updates with such steps, first and last ending at [s]":
            '29.486, at 10.78, 10.09; 0',
        'readings 0 ms late: at the end, the true arc length beyond the point of rest and beyond its estimate [mm]':
            '+18.6, +15.7',
        "readings 0 ms late: the robot's centre in the turn from [s] to [s]":
            '10.70; 15.38',
        'readings 20 ms late: front in the dock zone at [s]; at rest there from [s]; level of the plan in force at the end [m]; clearance of the footprint from the dock, least and at the end [m]':
            '24.78; 25.54; 0.10; 0.1814, 0.1834',
        "readings 20 ms late: the estimate's largest error along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0431, 0.072, 0.0212, 0.0089',
        "readings 20 ms late: the estimate's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0200, 0.093, 0.0391, 0.0365',
        'readings 20 ms late: the bounds of the estimate and of the feedback not kept':
            'none',
        "readings 20 ms late: the state's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]; within the funnel; its largest distance from the funnel in the norm [m]":
            '0.0437, 0.068, 0.0603, 0.0375; yes; 0.0000',
        "readings 20 ms late: the footprint's largest offset from the centre line [m]; touches; alarms, restarts; the largest torque of a wheel's drive [N m]":
            '0.450; nothing; 0, 0; 29.5',
        "readings 20 ms late: the alarm check's largest distance of the estimate from the initial set, in the norm [m]; at [s]":
            '0.0000; 0.0',
        "readings 20 ms late: (S5), before any alarm: the measurement's largest error at the period boundaries, in the norm [m], at [s]; within varpi":
            '0.0431, at 14.2; yes',
        "readings 20 ms late: before any alarm, the state's largest offset [m] and heading error [rad], signed, each at [s], s [m]":
            '-0.0603 at 15.40, 14.74; +0.0375 at 15.52, 14.86',
        "readings 20 ms late: before any alarm, the estimate's largest distance from the route's direction [rad], at [s], s [m]":
            '0.0365, at 15.54, 14.88',
        "readings 20 ms late: before any alarm, the estimate's largest error in heading [rad], at [s], s [m]":
            '0.0089, at 13.94, 13.28',
        "readings 20 ms late: the plan in force at the end: the dock's reserve [m]; the least clearance of the footprint in its funnel [m]":
            '0.1063; 0.1224',
        "readings 20 ms late: before any alarm, over the simulator's steps, the largest torque of a wheel's drive [N m], in the update ending at [s], s [m]; the steps with a drive at its limit; the updates with such steps, first and last ending at [s]":
            '29.486, at 10.78, 10.11; 0',
        'readings 20 ms late: at the end, the true arc length beyond the point of rest and beyond its estimate [mm]':
            '+29.2, +16.1',
        'readings 40 ms late: front in the dock zone at [s]; at rest there from [s]; level of the plan in force at the end [m]; clearance of the footprint from the dock, least and at the end [m]':
            '24.76; 25.52; 0.10; 0.1669, 0.1705',
        "readings 40 ms late: the estimate's largest error along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0629, 0.121, 0.0215, 0.0155',
        "readings 40 ms late: the estimate's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0331, 0.131, 0.0391, 0.0431',
        'readings 40 ms late: the bounds of the estimate and of the feedback not kept':
            'E_S, E_U, F_S, F_E',
        "readings 40 ms late: the state's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]; within the funnel; its largest distance from the funnel in the norm [m]":
            '0.0628, 0.094, 0.0606, 0.0441; yes; 0.0000',
        "readings 40 ms late: the footprint's largest offset from the centre line [m]; touches; alarms, restarts; the largest torque of a wheel's drive [N m]":
            '0.451; nothing; 0, 0; 29.5',
        "readings 40 ms late: the alarm check's largest distance of the estimate from the initial set, in the norm [m]; at [s]":
            '0.0000; 0.0',
        "readings 40 ms late: (S5), before any alarm: the measurement's largest error at the period boundaries, in the norm [m], at [s]; within varpi":
            '0.0629, at 14.2; no',
        "readings 40 ms late: before any alarm, the state's largest offset [m] and heading error [rad], signed, each at [s], s [m]":
            '-0.0606 at 15.38, 14.73; +0.0441 at 15.52, 14.88',
        "readings 40 ms late: before any alarm, the estimate's largest distance from the route's direction [rad], at [s], s [m]":
            '0.0431, at 15.56, 14.92',
        "readings 40 ms late: before any alarm, the estimate's largest error in heading [rad], at [s], s [m]":
            '0.0155, at 10.70, 10.05',
        "readings 40 ms late: the plan in force at the end: the dock's reserve [m]; the least clearance of the footprint in its funnel [m]":
            '0.1063; 0.1224',
        "readings 40 ms late: before any alarm, over the simulator's steps, the largest torque of a wheel's drive [N m], in the update ending at [s], s [m]; the steps with a drive at its limit; the updates with such steps, first and last ending at [s]":
            '29.484, at 10.78, 10.13; 0',
        'readings 40 ms late: at the end, the true arc length beyond the point of rest and beyond its estimate [mm]':
            '+42.1, +17.0',
        'readings 60 ms late: front in the dock zone at [s]; at rest there from [s]; level of the plan in force at the end [m]; clearance of the footprint from the dock, least and at the end [m]':
            '24.74; 25.52; 0.10; 0.1521, 0.1549',
        "readings 60 ms late: the estimate's largest error along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0824, 0.168, 0.0216, 0.0222',
        "readings 60 ms late: the estimate's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0468, 0.166, 0.0394, 0.0500',
        'readings 60 ms late: the bounds of the estimate and of the feedback not kept':
            'E_S, E_U, F_S, F_U, F_E',
        "readings 60 ms late: the state's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]; within the funnel; its largest distance from the funnel in the norm [m]":
            '0.0826, 0.100, 0.0610, 0.0509; no; 0.0126',
        "readings 60 ms late: the footprint's largest offset from the centre line [m]; touches; alarms, restarts; the largest torque of a wheel's drive [N m]":
            '0.451; nothing; 0, 0; 29.5',
        "readings 60 ms late: the alarm check's largest distance of the estimate from the initial set, in the norm [m]; at [s]":
            '0.0000; 0.0',
        "readings 60 ms late: (S5), before any alarm: the measurement's largest error at the period boundaries, in the norm [m], at [s]; within varpi":
            '0.0835, at 25.0; no',
        "readings 60 ms late: before any alarm, the state's largest offset [m] and heading error [rad], signed, each at [s], s [m]":
            '-0.0610 at 15.36, 14.73; +0.0509 at 15.52, 14.90',
        "readings 60 ms late: before any alarm, the estimate's largest distance from the route's direction [rad], at [s], s [m]":
            '0.0500, at 15.58, 14.96',
        'readings 60 ms late: before any alarm, the state outside its funnel: episodes of consecutive updates; the first four, from [s] to [s], s from [m] to [m], the largest distance in the norm [mm]; the share of the updates with the state outside, over the run and from the first exit to the last':
            '1; 11.36 to 25.18, 10.73 to 24.35, 12.6; 0.41, 1.00',
        "readings 60 ms late: the period boundaries at which the estimate or the state lay outside the initial set: number; the largest distance of the estimate and of the state [mm]; the first four, instant [s], the estimate's and the state's distance [mm]":
            '69; 0.0, 11.1; 11.4, 0.0, 0.2; 11.6, 0.0, 0.9; 11.8, 0.0, 1.7; 12.0, 0.0, 2.6',
        "readings 60 ms late: before any alarm, the estimate's largest error in heading [rad], at [s], s [m]":
            '0.0222, at 10.72, 10.09',
        "readings 60 ms late: the plan in force at the end: the dock's reserve [m]; the least clearance of the footprint in its funnel [m]":
            '0.1063; 0.1224',
        "readings 60 ms late: before any alarm, over the simulator's steps, the largest torque of a wheel's drive [N m], in the update ending at [s], s [m]; the steps with a drive at its limit; the updates with such steps, first and last ending at [s]":
            '29.500, at 10.80, 10.17; 0',
        'readings 60 ms late: at the end, the true arc length beyond the point of rest and beyond its estimate [mm]':
            '+57.8, +18.5',
        'readings 100 ms late: front in the dock zone at [s]; at rest there from [s]; level of the plan in force at the end [m]; clearance of the footprint from the dock, least and at the end [m]':
            '24.70; 25.50; 0.10; 0.1190, 0.1235',
        "readings 100 ms late: the estimate's largest error along [m], in speed [m/s], across [m], in heading [rad]":
            '0.1224, 0.275, 0.0221, 0.0357',
        "readings 100 ms late: the estimate's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0760, 0.245, 0.0397, 0.0635',
        'readings 100 ms late: the bounds of the estimate and of the feedback not kept':
            'E_S, E_U, E_E, F_S, F_U, F_E',
        "readings 100 ms late: the state's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]; within the funnel; its largest distance from the funnel in the norm [m]":
            '0.1221, 0.120, 0.0615, 0.0642; no; 0.0521',
        "readings 100 ms late: the footprint's largest offset from the centre line [m]; touches; alarms, restarts; the largest torque of a wheel's drive [N m]":
            '0.452; nothing; 0, 0; 29.5',
        "readings 100 ms late: the alarm check's largest distance of the estimate from the initial set, in the norm [m]; at [s]":
            '0.0030; 25.2',
        "readings 100 ms late: (S5), before any alarm: the measurement's largest error at the period boundaries, in the norm [m], at [s]; within varpi":
            '0.1377, at 25.0; no',
        "readings 100 ms late: before any alarm, the state's largest offset [m] and heading error [rad], signed, each at [s], s [m]":
            '-0.0615 at 15.32, 14.73; +0.0642 at 15.52, 14.94',
        "readings 100 ms late: before any alarm, the estimate's largest distance from the route's direction [rad], at [s], s [m]":
            '0.0635, at 15.62, 15.04',
        'readings 100 ms late: before any alarm, the state outside its funnel: episodes of consecutive updates; the first four, from [s] to [s], s from [m] to [m], the largest distance in the norm [mm]; the share of the updates with the state outside, over the run and from the first exit to the last':
            '1; 1.28 to 34.00, 0.65 to 24.38, 52.1; 0.96, 1.00',
        "readings 100 ms late: the period boundaries at which the estimate or the state lay outside the initial set: number; the largest distance of the estimate and of the state [mm]; the first four, instant [s], the estimate's and the state's distance [mm]":
            '164; 3.0, 51.6; 1.4, 0.0, 12.7; 1.6, 0.0, 23.5; 1.8, 0.0, 29.6; 2.0, 0.0, 33.2',
        "readings 100 ms late: before any alarm, the estimate's largest error in heading [rad], at [s], s [m]":
            '0.0357, at 10.72, 10.13',
        "readings 100 ms late: the plan in force at the end: the dock's reserve [m]; the least clearance of the footprint in its funnel [m]":
            '0.1063; 0.1224',
        "readings 100 ms late: before any alarm, over the simulator's steps, the largest torque of a wheel's drive [N m], in the update ending at [s], s [m]; the steps with a drive at its limit; the updates with such steps, first and last ending at [s]":
            '29.525, at 10.80, 10.21; 0',
        'readings 100 ms late: at the end, the true arc length beyond the point of rest and beyond its estimate [mm]':
            '+89.3, +20.0',
        'readings 40 ms late, compensated: front in the dock zone at [s]; at rest there from [s]; level of the plan in force at the end [m]; clearance of the footprint from the dock, least and at the end [m]':
            '24.80; 25.58; 0.10; 0.1939, 0.1945',
        "readings 40 ms late, compensated: the estimate's largest error along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0230, 0.035, 0.0219, 0.0149',
        "readings 40 ms late, compensated: the estimate's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0075, 0.038, 0.0400, 0.0256',
        'readings 40 ms late, compensated: the bounds of the estimate and of the feedback not kept':
            'none',
        "readings 40 ms late, compensated: the state's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]; within the funnel; its largest distance from the funnel in the norm [m]":
            '0.0249, 0.042, 0.0619, 0.0305; yes; 0.0000',
        "readings 40 ms late, compensated: the footprint's largest offset from the centre line [m]; touches; alarms, restarts; the largest torque of a wheel's drive [N m]":
            '0.452; nothing; 0, 0; 29.5',
        "readings 40 ms late, compensated: the alarm check's largest distance of the estimate from the initial set, in the norm [m]; at [s]":
            '0.0000; 0.0',
        "readings 40 ms late, compensated: (S5), before any alarm: the measurement's largest error at the period boundaries, in the norm [m], at [s]; within varpi":
            '0.0230, at 14.2; yes',
        "readings 40 ms late, compensated: before any alarm, the state's largest offset [m] and heading error [rad], signed, each at [s], s [m]":
            '-0.0619 at 15.42, 14.73; +0.0305 at 15.54, 14.86',
        "readings 40 ms late, compensated: before any alarm, the estimate's largest distance from the route's direction [rad], at [s], s [m]":
            '0.0256, at 15.54, 14.86',
        "readings 40 ms late, compensated: before any alarm, the estimate's largest error in heading [rad], at [s], s [m]":
            '0.0149, at 10.74, 10.05',
        "readings 40 ms late, compensated: the plan in force at the end: the dock's reserve [m]; the least clearance of the footprint in its funnel [m]":
            '0.1063; 0.1224',
        "readings 40 ms late, compensated: before any alarm, over the simulator's steps, the largest torque of a wheel's drive [N m], in the update ending at [s], s [m]; the steps with a drive at its limit; the updates with such steps, first and last ending at [s]":
            '29.476, at 10.78, 10.09; 0',
        'readings 40 ms late, compensated: at the end, the true arc length beyond the point of rest and beyond its estimate [mm]':
            '+18.2, +15.3',
        'readings 100 ms late, compensated: front in the dock zone at [s]; at rest there from [s]; level of the plan in force at the end [m]; clearance of the footprint from the dock, least and at the end [m]':
            '24.80; 25.36; 0.10; 0.1945, 0.1949',
        "readings 100 ms late, compensated: the estimate's largest error along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0227, 0.035, 0.0233, 0.0260',
        "readings 100 ms late, compensated: the estimate's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]":
            '0.0078, 0.039, 0.0415, 0.0205',
        'readings 100 ms late, compensated: the bounds of the estimate and of the feedback not kept':
            'E_E',
        "readings 100 ms late, compensated: the state's largest distance from the commanded motion along [m], in speed [m/s], across [m], in heading [rad]; within the funnel; its largest distance from the funnel in the norm [m]":
            '0.0247, 0.043, 0.0645, 0.0299; yes; 0.0000',
        "readings 100 ms late, compensated: the footprint's largest offset from the centre line [m]; touches; alarms, restarts; the largest torque of a wheel's drive [N m]":
            '0.454; nothing; 0, 0; 29.5',
        "readings 100 ms late, compensated: the alarm check's largest distance of the estimate from the initial set, in the norm [m]; at [s]":
            '0.0000; 0.0',
        "readings 100 ms late, compensated: (S5), before any alarm: the measurement's largest error at the period boundaries, in the norm [m], at [s]; within varpi":
            '0.0520, at 10.8; no',
        "readings 100 ms late, compensated: before any alarm, the state's largest offset [m] and heading error [rad], signed, each at [s], s [m]":
            '-0.0645 at 15.42, 14.73; +0.0299 at 15.54, 14.86',
        "readings 100 ms late, compensated: before any alarm, the estimate's largest distance from the route's direction [rad], at [s], s [m]":
            '0.0205, at 15.42, 14.73',
        "readings 100 ms late, compensated: before any alarm, the estimate's largest error in heading [rad], at [s], s [m]":
            '0.0260, at 10.80, 10.11',
        "readings 100 ms late, compensated: the plan in force at the end: the dock's reserve [m]; the least clearance of the footprint in its funnel [m]":
            '0.1063; 0.1224',
        "readings 100 ms late, compensated: before any alarm, over the simulator's steps, the largest torque of a wheel's drive [N m], in the update ending at [s], s [m]; the steps with a drive at its limit; the updates with such steps, first and last ending at [s]":
            '29.479, at 10.78, 10.09; 0',
        'readings 100 ms late, compensated: at the end, the true arc length beyond the point of rest and beyond its estimate [mm]':
            '+17.7, +15.0',
    },
}


if __name__ == "__main__":
    main()
