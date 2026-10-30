#!/usr/bin/env python3
"""
Labels, reserves and demands for the UR5e and the heavy-duty mobile manipulator.

Physical AI, Together, Chapter 14 (Section 14.2).

This script
  * writes the record of the experiment;
  * for the UR5e (arm/labels.py): finds the label that a declared payload
    allows, an interval (4.5 kg +- 10 %, its centre of mass 0.05 +- 0.025 m
    from the flange), keeping 10 % of every joint's torque for the feedback,
    against the nominal payload's, the empty arm's and the global
    capability's; the reserve, the payload's clearance from the post where a
    stop at the label from full speed at the place comes to rest, and the
    latest angle at which such a stop may begin; the torque that the arm
    needs at rest, for the rest segment, for every payload that the global
    capability's label admits; simulates the stop at the label for every
    payload of the grid over the interval, for payloads outside it, and for
    the loaded arm under the empty arm's label; writes the bound against the
    payload's mass for Figure 14.4;
  * for the mobile manipulator (hdmm/labels.py): finds, by pose, the
    tipping reserve at rest, the braking limit and the speeds that turns of
    3, 5 and 10 m allow, for the declared payload (270 kg +- 10 %) and empty,
    with what binds, and under the global capability, both reserves less the
    allowance (hdmm.labels.ALLOWANCE) that the test takes off for what the
    rigid model leaves out; the tipping reserve at rest under the global
    capability, which the rest segment carries, and its least value over the
    arm's ranges, and the arm's torques at rest at full reach; the centre of
    pressure of the simulated machine's wheel loads at the braking limits,
    its arm held on its servos, the deceleration raised over 5 s, with the
    actuators' rise (hdmm.labels.RISE) from cruising and from the drive's
    largest acceleration, and at once, against the rigid machine's
    zero-moment point, at several ratios of MuJoCo's frictional to normal
    impedance; through the turns at their speeds, the base held in its own
    frame and every body feeling that frame's forces, through the turn and
    braking in it; the base's creep and the front-most point's excess on the
    labels' least friction; the arm's torques at the limits;
  * maps a lift and a turn to what the contract asks of them: what a lift
    from carrying to reaching consumes of the tipping reserve in the static
    model; the requirement, and the check that a plan in force that holds it
    leaves a lift that keeps the floor, in the static model, within the
    forecast horizon and beyond it, and in the simulator: the plan in force
    braking with the arm held on its servos, and the arm lifting over 4 s on
    those servos, offset against gravity alone (and, for comparison, not
    offset), the base braking with the actuators' rise, from the drive (and,
    once, from cruising), from an instant of the lift or after it, every 0.1
    s and every 0.02 s from 3.2 to 4.4 s, where the least lies; what the lift
    consumes of the clearance, and how far the simulated front-most point
    and base move beyond the static model; the stops at the top speed; the
    tipping reserve against the recovery's deceleration for Figure 14.7; the
    speed that a turn allows, by pose and radius, with the distance and time
    in which to slow to it, and the largest level that a demand for a turn
    can ask;
  * finds the largest shortfall of a simulated plan's least distance below
    the rigid model's, against the allowance, and the largest at a single
    instant.

Every limit and reserve is rounded down, and every stop, consumption,
shortfall, travel and torque rounded up, at the digits printed, so that no
printed number is less safe than the computed one (the latest angles of the
UR5e's stops are rounded towards the place).

Usage
    python3 labels_demand.py            # writes results/labels-*
    python3 labels_demand.py --check    # also checks every value of its
                                        # summary

Outputs, in results/ next to this file
    labels-record.txt       the record of the experiment
    labels-summary.txt      every value, by group; with --check, the
                            comparison with the values expected
    labels-arm.csv          the UR5e's bound against the payload's mass
                            (mass, kg) for four centres of mass, 0.025,
                            0.05, 0.075 and 0.10 m from the flange (b025,
                            b050, b075, b100, rad/s^2) (Figure 14.4)
    labels-reserve.csv      the mobile manipulator's tipping reserve [m]
                            against the recovery's deceleration (b, m/s^2),
                            carrying and reaching with 243 to 297 kg
                            (carry, reach) (Figure 14.7)

Requires MuJoCo 3.16.0, NumPy and PyYAML (requirements.txt); run with
CPython 3.13.17, other versions of Python not tried. About eighteen minutes.
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
import yaml                                       # noqa: E402

from arm import control, world                    # noqa: E402
from arm import labels as al                      # noqa: E402
from arm import model as arm_model                # noqa: E402
from hdmm import labels as hl                     # noqa: E402
from hdmm import model as hdmm                    # noqa: E402

FULL = al.FULL
OUTSIDE = ((6.0, 0.05), (4.5, 0.10), (5.0, 0.10), (6.0, 0.10))  # payloads outside the declaration
POSES = {"stowed": hdmm.POSES["stowed"], "carry": hdmm.POSES["carry"],
         "reach": hdmm.POSES["reach"], "reach left": (math.pi / 2, 0.0, 0.0)}
LABELS = {"empty": hl.EMPTY, "declared": hl.DECLARED}
LIFT_TIME = 4.0                # [s] the lift from carrying to reaching (ours)
LIFT_KP = 2.0e6                # [N m/rad] the servos' stiffness in the simulated lift (ours)
ONSETS = sorted({round(0.1 * k, 2) for k in range(61)}          # [s] the recovery's onsets from
                 | {round(3.2 + 0.02 * k, 2) for k in range(61)})  # the lift's start, finer where
                                                                   # the least lies
RATIOS = (5.0, 10.0, 20.0, 50.0)        # MuJoCo's ratio of frictional to normal impedance
HEAVY = 7.0                    # [kg] a mass that the UR5e's global capability's label admits at
                               # no centre of mass in the box (checked), for the rest's torque
RADII = (3.0, 5.0, 10.0)       # [m] turns
TURN_LEAD = 0.5                # [s] a simulated turn starts this long before its entry
BRAKE_STARTS = ("entry", "arc", "exit")   # a simulated braking in a turn begins in the middle of
                                          # its entry, its arc or its exit
JOINT_NAMES = {"shoulder_pan_joint": "base", "shoulder_lift_joint": "shoulder",
               "elbow_joint": "elbow", "wrist_1_joint": "wrist 1", "wrist_2_joint": "wrist 2",
               "wrist_3_joint": "wrist 3"}


def down(x, digits=2):
    """x rounded towards minus infinity at the digits printed."""
    q = 10 ** digits
    return f"{math.floor(round(float(x) * q, 6)) / q + 0.0:.{digits}f}"


def up(x, digits=2):
    """x rounded towards plus infinity at the digits printed."""
    q = 10 ** digits
    return f"{math.ceil(round(float(x) * q, 6)) / q + 0.0:.{digits}f}"


def ddown(x, digits=3):
    return down(math.degrees(x), digits)


def dup(x, digits=3):
    return up(math.degrees(x), digits)


def kn_up(x, digits=2):
    """A torque [N m] in kN m, its magnitude rounded up, its sign kept
    unless the magnitude prints as zero."""
    v = float(x) / 1000.0
    text = up(abs(v), digits)
    return ("-" if v < 0 and float(text) > 0 else "") + text


# ----------------------------------------------------------------------------
# The record
# ----------------------------------------------------------------------------
def record_facts():
    arm = control.Arm(payload=0.0)
    post = arm.m.geom("post").id
    hm = hdmm.build(payload=hl.DECLARED.payloads[1], pose=hdmm.POSES["carry"],
                    friction=1.5).compile()
    return {
        "MuJoCo": mujoco.__version__,
        "NumPy, PyYAML, Python": f"{np.__version__}, {yaml.__version__}, {platform.python_version()}",
        "UR5e: declared mass [kg]; centre of mass from the flange [m]; the box's centre [m]":
            (f"{al.MASS[0]:g} to {al.MASS[1]:g}; {al.OFFSET[0]:g} to {al.OFFSET[1]:g}; "
             f"{al.BOX:g}"),
        "UR5e: global capability, mass [kg]; centre of mass [m]":
            (f"{al.GLOBAL_MASS[0]:g} to {al.GLOBAL_MASS[1]:g}; {al.GLOBAL_OFFSET[0]:g} to "
             f"{al.GLOBAL_OFFSET[1]:g}"),
        "UR5e: share of each joint's torque for the label; reserve floor [m]; levels' grid [m]":
            f"{al.SHARE:g}; {al.FLOOR:g}; {al.GRID:g}",
        "UR5e: the post's contact, solref; solimp (MuJoCo's defaults)": (
            f"{', '.join(f'{x:g}' for x in arm.m.geom_solref[post])}; "
            f"{', '.join(f'{x:g}' for x in arm.m.geom_solimp[post])}"),
        "mobile manipulator: declared payload [kg]; friction; global capability's payload "
        "[kg] and friction":
            (f"{hl.DECLARED.payloads[0]:g} to {hl.DECLARED.payloads[1]:g}; "
             f"{hl.DECLARED.friction:g}; {hl.GLOBAL.payloads[0]:g} to "
             f"{hl.GLOBAL.payloads[1]:g}, {hl.GLOBAL.friction:g}"),
        "mobile manipulator: brakes, drive [m/s^2]; top speed [m/s]; the rise of the base's "
        "acceleration along a half-cosine [s]":
            f"{hl.BRAKES:g}, {hl.DRIVE:g}; {hl.SPEED:g}; {hl.RISE:g}",
        "mobile manipulator: the turns' transitions [m]; the turns' angle [deg]; the braking "
        "and the driving off in a turn [m/s^2]":
            f"{hl.TRANSITION:g}; {math.degrees(hl.TURN_ANGLE):g}; {hl.TURN_BRAKING:g}, "
            f"{hl.TURN_DRIVE:g}",
        "mobile manipulator: the arm's torque limits, slew, lift, tilt [kN m]; the factor on the "
        "torque that the base's acceleration adds":
            ", ".join(f"{x / 1000:g}" for x in hl.ARM_LIMITS) + f"; {hl.KAPPA:g}",
        "mobile manipulator: floors of the tipping reserve and the clearance [m]; the allowance "
        "that both keep for what the rigid model leaves out [m]; levels' grid [m]":
            f"{hl.FLOOR:g}, {hl.CLEAR_FLOOR:g}; {hl.ALLOWANCE:g}; {hl.GRID:g}",
        "mobile manipulator: the simulated contacts' solref; solimp; ratio of frictional to "
        "normal impedance": (
            f"{', '.join(f'{x:g}' for x in hm.geom_solref[hm.geom('ground').id])}; "
            f"{', '.join(f'{x:g}' for x in hm.geom_solimp[hm.geom('ground').id])}; "
            f"{hm.opt.impratio:g}"),
        "mobile manipulator: the lift's duration [s]; the servos' stiffness [N m/rad], damping "
        "[N m s/rad]; the recovery's onsets [s]":
            (f"{LIFT_TIME:g}; {LIFT_KP:g}, {0.05 * LIFT_KP:g}; 0 to 6 every 0.1 and 3.2 to 4.4 "
             "every 0.02"),
    }


# ----------------------------------------------------------------------------
# The UR5e
# ----------------------------------------------------------------------------
def arm_label_facts(rows):
    out = {}
    b, (m, r, joint, end), table = al.interval_label()
    out["the declared interval: the label [rad/s^2]; the payload that binds, mass [kg], centre "
        "of mass [m]; the joint"] = f"{down(b)}; {m:g}, {r:g}; {JOINT_NAMES[joint]} at the {end}"
    out["the grid over the interval: each payload's own bound [rad/s^2], mass and centre of "
        "mass"] = "; ".join(f"{mm:g} kg {rr:g} m {down(bb)}" for mm, rr, bb, _, _ in table)
    b1, (_, _, j1, e1), _ = al.interval_label(share=1.0)
    out["the same without the feedback's share [rad/s^2]; the joint"] = \
        f"{down(b1)}; {JOINT_NAMES[j1]} at the {e1}"
    bn, (jn, en), _ = control.capability(al.arm_for(4.5, 0.05), FULL, al.SHARE)
    be, (je, ee), _ = control.capability(control.Arm(payload=0.0), FULL, al.SHARE)
    bg, (_, _, jg, eg), _ = al.interval_label(al.GLOBAL_MASS, al.GLOBAL_OFFSET)
    out["with the feedback's share: the nominal payload's (4.5 kg, 0.05 m), the empty arm's and "
        "the global capability's label [rad/s^2]; the joints"] = (
        f"{down(bn)}, {down(be)}, {down(bg)}; {JOINT_NAMES[jn]} at the {en}, {JOINT_NAMES[je]} at "
        f"the {ee}, {JOINT_NAMES[jg]} at the {eg}")
    out["the stop from full speed at the label [deg]: the declared interval's, the nominal's, "
        "the empty arm's, the global capability's"] = \
        ", ".join(dup(FULL * FULL / (2 * x)) for x in (b, bn, be, bg))
    out["the nominal payload's label above the declared interval's [%]"] = \
        down(100 * (bn / b - 1), 1)
    for mass in np.arange(3.0, 7.0001, 0.25):
        row = {"mass": float(mass)}
        for offset, col in ((0.025, "b025"), (0.05, "b050"), (0.075, "b075"), (0.10, "b100")):
            row[col] = control.capability(al.arm_for(float(mass), offset), FULL, al.SHARE)[0]
        rows.append(row)
    return out, b, be, bg


def arm_reserve_facts(b, b_empty, b_global):
    out = {}
    arm = al.arm_for(al.MASS[1], al.OFFSET[1])
    out["the payload's clearance from the post at the place [m]"] = \
        down(al.clearance_at(arm, world.PLACE), 4)
    parts = []
    for x in (b, b_global):
        rest = al.stop_angle(x)
        parts.append(f"{ddown(rest)}, {down(al.clearance_at(arm, rest), 4)}")
    out["a stop from full speed at the place: its point of rest [deg] and the clearance there, "
        "the plan's reserve [m], under the declared label; under the global capability's"] = \
        "; ".join(parts)
    latest = {name: al.latest_brake(arm, x)
              for name, x in (("declared", b), ("global", b_global), ("empty", b_empty))}
    out["the angle at which the clearance falls to the floor [deg]"] = \
        dup(al.stop_angle(b, latest["declared"]))
    out["the latest angle from which a stop from full speed keeps the floor [deg], beyond the "
        "place [deg]: under the declared label; the global capability's; the empty arm's, the "
        "box in place"] = "; ".join(f"{dup(x)}, {ddown(world.PLACE - x)}" for x in latest.values())
    def hold_share(m, r):
        a = al.arm_for(m, r)
        _, _, g = control.coefficients(a)
        return float(np.max(np.abs(g) / a.limit))

    rated = max(hold_share(m, r) for m in al.grid(al.GLOBAL_MASS) for r in al.grid(al.GLOBAL_OFFSET))
    offsets = al.grid(al.GLOBAL_OFFSET, 5)
    masses = [0.25 * k for k in range(int(round(HEAVY / 0.25)) + 1)]
    heaviest, heavy_bounds = [], []
    for r in offsets:
        last = al.GLOBAL_MASS[1]
        for m in masses:
            if m <= al.GLOBAL_MASS[1]:
                continue
            if control.capability(al.arm_for(m, r), FULL, al.SHARE)[0] < b_global:
                break
            last = m
        heaviest.append(last)
        heavy_bounds.append(control.capability(al.arm_for(HEAVY, r), FULL, al.SHARE)[0])
    if max(heavy_bounds) >= b_global:
        raise RuntimeError("the global capability's label admits a payload of HEAVY kg")
    heavy = max(hold_share(m, r) for m in masses for r in offsets)
    out["the global capability's label: the heaviest payload that it admits, on a grid of 0.25 kg, "
        "with its centre of mass at " + ", ".join(f"{r:g}" for r in offsets) + " m [kg]; the "
        f"bound of {HEAVY:g} kg there, every one below the label [rad/s^2]"] = (
        ", ".join(f"{m:g}" for m in heaviest) + "; " + ", ".join(down(x) for x in heavy_bounds))
    out["at rest in the carry pose: the largest share of a joint's torque that holds the arm, "
        "every payload of the global capability's grid up to the rated 5 kg; every payload up to "
        f"{HEAVY:g} kg with its centre of mass in the box, on the grid of 0.25 kg and 0.025 m"] = \
        f"{up(rated, 3)}; {up(heavy, 3)}"
    return out, latest


def arm_stop_line(arm, b, start=world.PLACE):
    r = control.stop(arm, b, FULL, start=start)
    order = [int(j) for j in np.argsort(-r["peak"])]
    asked = ", ".join(f"{JOINT_NAMES[arm_model.JOINTS[j]]} {up(r['peak'][j], 3)}"
                      for k, j in enumerate(order) if k == 0 or r["peak"][j] > 1.0)
    clipped = ", ".join(f"{JOINT_NAMES[arm_model.JOINTS[j]]} {int(n)}"
                        for j, n in enumerate(r["saturated"]) if n > 0) or "none"
    return (f"{asked}; {clipped}; {dup(r['err'][0])}; {dup(r['beyond'])}; "
            f"{dup(r['departure'][4])}; {down(r['clearance'], 4)}")


STOP_HEAD = ("the joint asked the largest share of its torque, and any other asked more than its "
             "limit, with the shares; the updates in which each joint's torque is clipped; largest "
             "error of the base [deg]; beyond the commanded rest [deg]; wrist 2's largest "
             "departure from the pose [deg]; least clearance [m]")


def arm_stop_facts(b, b_empty, latest):
    out = {}
    peaks, clipped, errs, beyond, clear = [], 0, [], [], []
    for m in al.grid(al.MASS):
        for r in al.grid(al.OFFSET):
            res = control.stop(al.arm_for(m, r), b, FULL, start=world.PLACE)
            peaks.append(max(res["peak"]))
            clipped += int(res["saturated"].sum())
            errs.append(res["err"][0])
            beyond.append(res["beyond"])
            clear.append(res["clearance"])
    out["the 9 payloads of the grid, stopped at the label from full speed at the place: "
        "largest torque asked over its limit; updates clipped, all joints; largest error of the "
        "base [deg]; largest overrun of the commanded rest [deg]; least clearance [m]"] = (
        f"{up(max(peaks), 3)}; {clipped}; {dup(max(errs))}; {dup(max(beyond))}; "
        f"{down(min(clear), 4)}")
    corner = al.arm_for(al.MASS[1], al.OFFSET[1])
    out["the interval's heaviest corner (4.95 kg, 0.075 m), stopped at the label from full speed "
        "at the latest angle: " + STOP_HEAD] = arm_stop_line(corner, b, latest["declared"])
    for m, r in OUTSIDE:
        bb, (jj, ee), _ = control.capability(al.arm_for(m, r), FULL, 1.0)
        bs, _, _ = control.capability(al.arm_for(m, r), FULL, al.SHARE)
        out[f"outside the declaration, {m:g} kg at {r:g} m: its own bound with and without the "
            "share [rad/s^2], the joint; stopped at the label from full speed at the place: "
            + STOP_HEAD] = (f"{down(bs)}, {down(bb)}, {JOINT_NAMES[jj]}; "
                            + arm_stop_line(al.arm_for(m, r), b))
    m, r = OUTSIDE[-1]
    out[f"outside the declaration, {m:g} kg at {r:g} m, stopped at the label from full speed at "
        "the latest angle: " + STOP_HEAD] = arm_stop_line(al.arm_for(m, r), b, latest["declared"])
    out["the interval's heaviest corner under the empty arm's label "
        f"({down(b_empty)} rad/s^2), from full speed at the place: " + STOP_HEAD] = \
        arm_stop_line(corner, b_empty)
    out["the same from the latest angle that the empty arm's label allows with the box in place: "
        + STOP_HEAD] = arm_stop_line(corner, b_empty, latest["empty"])
    r = control.stop(corner, b_empty, FULL, start=world.PLACE)
    out["the heaviest corner under the empty arm's label from the place: its farthest angle "
        "[deg]; the stop's angle in all [deg]"] = f"{ddown(r['farthest'])}; {dup(r['travel'])}"
    r = control.stop(corner, b_empty, FULL, start=latest["empty"])
    out["the same from the latest angle that the empty arm's label allows: how far the box enters "
        "the post [mm]"] = up(-1000 * r["clearance"], 1)
    m6, _ = OUTSIDE[0]
    out[f"{m6:g} kg against the declaration's heaviest payload: heavier by [%]"] = \
        down(100 * (m6 / al.MASS[1] - 1), 0)
    bb, _, _ = control.capability(corner, FULL, 1.0)
    out["the heaviest corner's own stop without the share: its angle from full speed [deg]"] = \
        dup(FULL * FULL / (2 * bb))
    return out


# ----------------------------------------------------------------------------
# The mobile manipulator
# ----------------------------------------------------------------------------
def limit_text(v):
    return "none" if v is None else ("inf" if math.isinf(v) else down(v))


def speeds_text(speeds):
    return ", ".join(f"{down(v)} {w}" for v, w in speeds)


def lower_speeds_hold(q, radius, label, v, step=0.5):
    """Whether every speed below v on a grid of step m/s holds in the turn
    (hl.turn_holds), as the search for the turn's speed assumes."""
    return all(hl.turn_holds(q, radius, step * k, label)[0] for k in range(1, int(v / step) + 1)
               if step * k < v - 1e-9)


def hdmm_label_facts():
    out = {}
    lower = True
    for name, q in POSES.items():
        for tag, label in LABELS.items():
            r0, e0, _ = hl.tipping_reserve(q, label)
            b, wb = hl.braking_limit(q, label)
            speeds = [hl.turn_speed(q, radius, label) for radius in RADII]
            lower &= all(lower_speeds_hold(q, radius, label, v)
                         for radius, (v, _) in zip(RADII, speeds))
            out[f"{name}, {tag}: tipping reserve at rest [m], its edge; braking limit [m/s^2], "
                "what binds; the speed in a turn of 3, 5 and 10 m [m/s], what binds"] = (
                f"{down(r0, 4)}, {e0}; {down(b)}, {wb}; {speeds_text(speeds)}")
        lim = hl.braking_limits(q, hl.DECLARED)
        out[f"{name}, declared: the four limits on the braking at the floor, the brakes, the "
            "ground, tipping, the arm [m/s^2]"] = ", ".join(limit_text(v) for v in lim.values())
        r, e, _ = hl.tipping_reserve(q, hl.GLOBAL)
        out[f"{name}: tipping reserve at rest under the global capability [m], its edge"] = \
            f"{down(r, 4)}, {e}"
        out[f"{name}: the front-most point [m]"] = up(hl.front_point(q), 3)
    b_reach = hl.braking_limit(POSES["reach"], hl.DECLARED)[0]
    out["reach, declared: the braking limit below the brakes' [%]"] = \
        down(100 * (1 - b_reach / hl.BRAKES), 0)
    out["under the global capability: braking limit [m/s^2], what binds, stowed, carrying, "
        "reaching, reaching left"] = "; ".join(
        f"{down(b)}, {w}" for b, w in (hl.braking_limit(q, hl.GLOBAL) for q in POSES.values()))
    parts = []
    for q in POSES.values():
        speeds = [hl.turn_speed(q, radius, hl.GLOBAL) for radius in RADII]
        lower &= all(lower_speeds_hold(q, radius, hl.GLOBAL, v)
                     for radius, (v, _) in zip(RADII, speeds))
        parts.append(speeds_text(speeds))
    out["under the global capability: the speed in a turn of 3, 5 and 10 m [m/s], what binds, "
        "stowed, carrying, reaching, reaching left"] = "; ".join(parts)
    out["every turn's speed: every lower speed on a grid of 0.5 m/s holds too"] = \
        "yes" if lower else "no"
    return out


def rest_facts():
    """The rest segment under the global capability: the least tipping
    reserve at rest over the arm's ranges, which lies at full reach (where
    every part's horizontal distance from the slew axis is largest), over
    the slew in steps of 0.5 deg; and the arm's torques at rest at full
    reach, the largest that its ranges allow."""
    out = {}
    g0, g1 = hl.GLOBAL.payloads
    best = (math.inf, None, None)
    for k in range(720):
        slew = math.radians(0.5 * k)
        r, edge, _ = hl.tipping_reserve((slew, 0.0, 0.0), hl.GLOBAL)
        if r < best[0]:
            best = (r, slew, edge)
    out[f"the least tipping reserve at rest over the arm's ranges with {g0:g} to {g1:g} kg [m]; "
        "the slew [deg], at full reach; the edge"] = \
        f"{down(best[0], 4)}; {math.degrees(best[1]):.1f}; {best[2]}"
    t = hl.torques((0.0, 0.0, 0.0), g1)
    out[f"the arm's torques at rest at full reach with {g1:g} kg, lift, tilt [kN m]; their shares "
        "of the limits"] = (f"{kn_up(t[1])}, {kn_up(t[2])}; "
                            f"{up(abs(t[1]) / hl.ARM_LIMITS[1], 3)}, "
                            f"{up(abs(t[2]) / hl.ARM_LIMITS[2], 3)}")
    return out


HOLD_ONSET = 0.5               # [s] the acceleration's onset in a held run, after the arm settles
SLOW = 5.0                     # [s] the rise of the slow check, against the actuators' RISE
LIMIT_LABELS = (("empty", hl.EMPTY), ("declared", hl.DECLARED), ("global", hl.GLOBAL))
SHORT = {"least": [], "instant": [], "held": [], "torque": [], "creep": [], "front": [],
         "over": []}


def held(payload, pose, accel, rise, before=(0.0, 0.0), changes=None, impratio=hdmm.IMPRATIO,
         friction=1.5, contacts=False):
    """The jointed machine holding its arm in pose on the servos of the
    simulated lift (hdmm.model.simulated_lift with the start and the end
    pose the same, the servos offset against gravity), the base's
    acceleration going from before to accel from HOLD_ONSET on, at once or
    along a half-cosine over rise seconds, then held; or following changes,
    a list of (time, acceleration): the least distance of the simulated
    centre of pressure inside the contacts over the run [m], the largest
    shortfall of it below the rigid model's distance for the planned pose
    under the acceleration acting at the same instant [m], the base's travel
    on its contacts [m], the front-most point's largest excess over the rigid
    machine's, the arm's swing and the base's travel together [m], the
    servos' largest torques [N m] and, with contacts, the fewest contacts at
    a step and the wheels that lose their contact at some step."""
    if changes is None:
        changes = [(HOLD_ONSET, accel)]
    end = changes[-1][0]
    r = hdmm.simulated_lift(payload, pose, pose, 0.5, changes=changes, before=before,
                            kp=LIFT_KP, offset=True, rise=rise, impratio=impratio,
                            friction=friction, contacts=contacts,
                            hold=max(6.0, end + rise + 4.0))
    sim = np.minimum(hdmm.AXLE - np.abs(r["cop"][:, 0]), hdmm.HALF_TRACK - np.abs(r["cop"][:, 1]))
    static = np.array([hl.margin(tuple(pose), payload, *a)[0] for a in r["accel"]])
    finite = bool(np.all(np.isfinite(sim)))
    front = r["base"][:, 0] + np.array([hl.front_point(tuple(q)) for q in r["q"]])
    out = dict(least=float(np.min(sim)) if finite else -math.inf,
               instant=float(np.max(static - sim)) if finite else math.inf,
               travel=float(np.linalg.norm(r["base"], axis=1).max()),
               front=float(front.max() - hl.front_point(tuple(pose))),
               torque=np.abs(r["torque"]).max(axis=0))
    if contacts:
        out["fewest"] = int(r["ncon"].min())
        out["dropped"] = sorted({w for s in r["touching"] for w in ("fl", "fr", "rl", "rr")
                                 if w not in s})
    return out


def limit_case(name, label):
    """A braking limit of tab:hdmm-limits: the pose, the deceleration at the
    limit under the label, and the payload that binds, the label's
    heaviest."""
    q = POSES[name]
    a, _ = hl.braking_limit(q, label)
    return q, a, (-a, 0.0), label.payloads[1]


def stop_changes(a):
    """A stop at the deceleration a from the top speed: its rise from
    HOLD_ONSET on, its plateau, and its fading at standstill."""
    plateau = max(0.0, hl.SPEED / a - hl.RISE)
    return [(HOLD_ONSET, (-a, 0.0)), (HOLD_ONSET + hl.RISE + plateau, (0.0, 0.0))]


def limit_runs(top, q, a, accel, friction=1.5):
    """The held runs at a braking limit, with the actuators' rise: braking
    from cruising and from the drive's largest acceleration, held; and the
    stop's end, the deceleration fading at standstill, from the top speed
    and from as low a speed as leaves no plateau."""
    return [held(top, q, accel, hl.RISE, friction=friction),
            held(top, q, accel, hl.RISE, before=(hl.DRIVE, 0.0), friction=friction),
            held(top, q, accel, hl.RISE, changes=stop_changes(a), friction=friction),
            held(top, q, accel, hl.RISE, changes=[(HOLD_ONSET, accel),
                                                   (HOLD_ONSET + hl.RISE, (0.0, 0.0))],
                 friction=friction)]


def turn_motion(radius, speed, side, brake=None, drive_from=None, dt=hdmm.TIMESTEP):
    """The base's motion through a turn of the radius to the side, for
    hdmm.model.simulated_lift: its middle along the route at the speed from
    TURN_LEAD seconds before the entry, through the turn and on; or, with
    brake, a place in the turn ("entry", "arc" or "exit", its middle) or a
    distance from the entry's start [m], braking from there at
    hl.TURN_BRAKING to rest, the deceleration rising and fading over the
    actuators' rise; or, with drive_from, a distance from the entry's start
    [m], at rest there for TURN_LEAD seconds and then driving off at
    hl.TURN_DRIVE to the speed, which it keeps through the turn and on.
    Returns the motion, a function of time giving (ax, ay, w, alpha), and
    its duration [s]."""
    entry, arc, exit_ = hl.turn_length(radius)
    end = entry + arc + exit_
    if drive_from is not None:
        n0 = int(round(TURN_LEAD / dt))
        _, vd, sd, ad = hl.drive_run(speed, hl.TURN_DRIVE, dt=dt)
        s1 = drive_from + sd[-1]
        n1 = max(0, int(math.ceil((end - s1) / speed / dt)))
        s = np.concatenate([np.full(n0, float(drive_from)), drive_from + sd,
                            s1 + speed * dt * np.arange(1, n1 + 1)])
        v = np.concatenate([np.zeros(n0), vd, np.full(n1, float(speed))])
        ax = np.concatenate([np.zeros(n0), ad, np.zeros(n1)])
    elif brake is None:
        s0 = -speed * TURN_LEAD
        t = np.arange(0.0, (end - s0) / speed + 0.5 * dt, dt)
        s, v, ax = s0 + speed * t, np.full(len(t), float(speed)), np.zeros(len(t))
    else:
        at = brake if not isinstance(brake, str) else \
            {"entry": 0.5 * entry, "arc": entry + 0.5 * arc, "exit": entry + arc + 0.5 * exit_}[brake]
        s0 = min(0.0, at) - speed * TURN_LEAD
        n0 = max(1, int(round((at - s0) / speed / dt)))
        _, vb, sb, db = hl.braking_run(speed, hl.TURN_BRAKING, dt=dt)
        before = s0 + speed * dt * np.arange(n0)
        s = np.concatenate([before, before[-1] + speed * dt + sb])
        v = np.concatenate([np.full(n0, float(speed)), vb])
        ax = np.concatenate([np.zeros(n0), -db])
    t = dt * np.arange(len(s))
    k, dk = hl.route_curvature(s, radius, side)
    ay, w, alpha = v * v * k, v * k, ax * k + v * v * dk

    def motion(time):
        i = int(round(time / dt))
        if i >= len(t):
            return 0.0, 0.0, 0.0, 0.0
        return ax[i], ay[i], w[i], alpha[i]

    return motion, float(t[-1])


def turn_run(payload, pose, radius, speed, side, brake=None, drive_from=None, friction=1.5):
    """The jointed machine holding its arm in pose on the servos of the
    simulated lift through a turn (turn_motion): the least distance of the
    simulated centre of pressure inside the contacts [m], the rigid model's
    least for the planned pose (hl.motion_margin), the largest shortfall of
    the first below the second at the same instant [m], the servos' largest
    torques [N m], and the base's largest travel [m] and turn [rad] on its
    contacts."""
    motion, duration = turn_motion(radius, speed, side, brake, drive_from)
    r = hdmm.simulated_lift(payload, pose, pose, 0.5, motion=motion, kp=LIFT_KP, offset=True,
                            friction=friction, hold=duration + 1.5)
    sim = np.minimum(hdmm.AXLE - np.abs(r["cop"][:, 0]), hdmm.HALF_TRACK - np.abs(r["cop"][:, 1]))
    rigid = hl.motion_margin(tuple(pose), payload, r["accel"][:, 0], r["accel"][:, 1],
                             r["turn"][:, 0], r["turn"][:, 1])
    return dict(least=float(sim.min()), rigid=float(rigid.min()),
                instant=float(np.max(rigid - sim)), torque=np.abs(r["torque"]).max(axis=0),
                travel=float(np.linalg.norm(r["base"], axis=1).max()),
                turn=float(np.abs(r["yaw"]).max()))


def turn_cases(label):
    """The simulated turns under a label: in each pose, each radius of RADII
    at the turn's speed, to each side that hl.sides_of needs, with the
    label's heaviest payload; with the starts at which the test binds, of a
    braking and of a drive off from rest in the turn, for the constraint
    that limits the turn's speed (tipping where the top speed does)."""
    for name, q in POSES.items():
        for radius in RADII:
            v, why = hl.turn_speed(q, radius, label)
            constraint = "tipping" if why == "top speed" else why
            for side in hl.sides_of(q):
                brake, _ = hl.turn_worst_start(q, radius, v, label, side,
                                               braking=hl.TURN_BRAKING, constraint=constraint)
                drive, _ = hl.turn_worst_start(q, radius, v, label, side, drive=hl.TURN_DRIVE,
                                               constraint=constraint)
                yield name, q, radius, v, side, brake, drive


def shares(torque):
    return [float(x) / L for x, L in zip(torque, hl.ARM_LIMITS)]


def hdmm_simulated_facts():
    out = {}
    worst_slow = 0.0
    for name in ("stowed", "carry", "reach", "reach left"):
        q, a, accel, top = limit_case(name, hl.DECLARED)
        rigid = hl.margin(q, top, *accel)[0]
        worst_slow = max(worst_slow, rigid - held(top, q, accel, SLOW)["least"])
    out["the declared label's braking limits in the four poses, the arm held on its servos, the "
        "deceleration raised over 5 s: the largest shortfall of the simulated centre of "
        "pressure's least below the rigid model's [mm]"] = up(1000 * worst_slow)
    for tag, label in LIMIT_LABELS:
        parts = []
        for name in ("stowed", "carry", "reach", "reach left"):
            q, a, accel, top = limit_case(name, label)
            rigid = hl.margin(q, top, *accel)[0]
            runs = limit_runs(top, q, a, accel)
            least = min(r["least"] for r in runs)
            instant = max(r["instant"] for r in runs)
            torque = np.max([r["torque"] for r in runs], axis=0)
            SHORT["least"].append(rigid - least)
            SHORT["held"].append(instant)
            SHORT["torque"].append(shares(torque))
            parts.append(f"{name} {down(a)}: {down(rigid, 4)}, {down(least, 4)}, "
                         f"{up(1000 * instant)}, " + "/".join(up(s, 3) for s in shares(torque)))
        out[f"{tag} ({label.payloads[1]:g} kg), at each braking limit, the arm held on its servos, "
            "with the actuators' rise, braking from cruising and from the drive's largest "
            "acceleration and stopping: the limit [m/s^2]; the rigid model's zero-moment point "
            "inside the contacts and the simulated centre of pressure's least [m]; the largest "
            "shortfall at an instant [mm]; the servos' largest torques, slew/lift/tilt, as shares "
            "of their limits"] = "; ".join(parts)
    for tag, label in LIMIT_LABELS:
        runs = []
        for name, q, radius, v, side, bind, drive in turn_cases(label):
            for brake in (None,) + BRAKE_STARTS + (bind,):
                runs.append(turn_run(label.payloads[1], q, radius, v, side, brake))
            runs.append(turn_run(label.payloads[1], q, radius, v, side, drive_from=drive))
        least = min(runs, key=lambda r: r["least"])
        instant = max(r["instant"] for r in runs)
        torque = np.max([r["torque"] for r in runs], axis=0)
        SHORT["least"] += [r["rigid"] - r["least"] for r in runs]
        SHORT["held"].append(instant)
        SHORT["torque"].append(shares(torque))
        out[f"{tag} ({label.payloads[1]:g} kg), the turns of 3, 5 and 10 m at their speeds in the "
            "four poses, the arm held on its servos, through the turn, braking in it from the "
            "middle of its entry, its arc and its exit and from where the test binds, and driving "
            "off in it from rest where the test binds: the runs; the least of the simulated "
            "centre of pressure inside the contacts [m], the rigid model's least in that run [m]; "
            "the largest shortfall at an instant [mm]; the servos' largest torques, "
            "slew/lift/tilt, as shares of their limits"] = (
            f"{len(runs)}; {down(least['least'], 4)}, {down(least['rigid'], 4)}; "
            f"{up(1000 * instant)}; " + "/".join(up(x, 3) for x in shares(torque)))
    q, a, accel, top = limit_case("carry", hl.DECLARED)
    once = [held(top, q, accel, 0.0)["least"],
            held(top, q, accel, 0.0, before=(hl.DRIVE, 0.0))["least"]]
    out["without the actuators' rise, at once: carry, 297 kg, braking at its limit, the simulated "
        "centre of pressure's least from cruising and from the drive's largest acceleration "
        "[m]"] = f"{down(once[0], 4)}, {down(once[1], 4)}"
    by_ratio = {ratio: [] for ratio in RATIOS}
    for name in ("carry", "reach", "reach left"):
        q, a, accel, top = limit_case(name, hl.DECLARED)
        rigid = hl.margin(q, top, *accel)[0]
        for ratio in RATIOS:
            slow = held(top, q, accel, SLOW, impratio=ratio)["least"]
            fast = held(top, q, accel, hl.RISE, impratio=ratio)["least"]
            by_ratio[ratio].append((rigid - slow, rigid - fast))
    for ratio, rows in by_ratio.items():
        out[f"MuJoCo's ratio of frictional to normal impedance {ratio:g}: the shortfall at the "
            "braking limit [mm], raised over 5 s and with the rise, carrying, reaching, reaching "
            "left"] = "; ".join(f"{up(1000 * s)}, {up(1000 * o)}" for s, o in rows)
    change = max(abs(x - y) for r20, r50 in zip(by_ratio[20.0], by_ratio[50.0])
                 for x, y in zip(r20, r50))
    out["the largest change of a shortfall between the ratios 20 and 50 [mm]"] = up(1000 * change)
    q, a, accel, top = limit_case("reach", hl.DECLARED)
    for ratio in (None, hdmm.IMPRATIO):
        r = held(top, q, accel, SLOW, impratio=ratio, contacts=True)
        out[f"reaching, braking at its limit raised over 5 s, MuJoCo's ratio of frictional to normal "
            f"impedance {1 if ratio is None else ratio:g}, friction 1.5: the base's creep over the "
            "run [mm]; the fewest wheel contacts at a step; the wheels that lose their contact"] = (
            f"{up(1000 * r['travel'])}; {r['fewest']}; "
            + (", ".join(r["dropped"]) if r["dropped"] else "none"))
    parts = []
    for tag, label in LIMIT_LABELS:
        creep, front = 0.0, 0.0
        for name in ("stowed", "carry", "reach", "reach left"):
            q, a, accel, top = limit_case(name, label)
            for before in ((0.0, 0.0), (hl.DRIVE, 0.0)):
                r = held(top, q, accel, hl.RISE, before=before, changes=stop_changes(a),
                         friction=label.friction)
                creep, front = max(creep, r["travel"]), max(front, r["front"])
        SHORT["creep"].append(creep)
        SHORT["front"].append(front)
        parts.append(f"{tag}, on friction {label.friction:g}: {up(1000 * creep)}, {up(1000 * front)}")
    out["stopping from the top speed at each braking limit of the four poses, the deceleration "
        "rising from cruising and from the drive's largest acceleration, on the label's least "
        "friction: the base's largest creep on its contacts [mm]; the front-most point's largest "
        "excess over the rigid machine's, the arm's swing and the creep together [mm]"] = \
        "; ".join(parts)
    parts = []
    for tag, label in LIMIT_LABELS:
        travel, turn = 0.0, 0.0
        for name, q, radius, v, side, bind, drive in turn_cases(label):
            runs = [turn_run(label.payloads[1], q, radius, v, side, brake, friction=label.friction)
                    for brake in (None, "exit", bind)]
            runs.append(turn_run(label.payloads[1], q, radius, v, side, drive_from=drive,
                                 friction=label.friction))
            for r in runs:
                travel, turn = max(travel, r["travel"]), max(turn, r["turn"])
        SHORT["creep"].append(travel)
        parts.append(f"{tag}, on friction {label.friction:g}: {up(1000 * travel)}, "
                     f"{up(1000 * turn, 3)}")
    out["the turns at their speeds on the label's least friction, through the turn, braking in it "
        "from the middle of its exit and from where the test binds, and driving off in it from "
        "rest where the test binds: the base's largest creep on its contacts [mm], and its "
        "largest turn on them [mrad]"] = "; ".join(parts)
    return out


def hdmm_torque_facts():
    out = {}
    top = hl.DECLARED.payloads[1]
    for name in ("carry", "reach", "reach left"):
        q = POSES[name]
        b, _ = hl.braking_limit(q)
        tb = hl.torques(q, top, ax=-b)
        share = max(abs(x) / L for x, L in zip(tb, hl.ARM_LIMITS))
        out[f"{name}, {top:g} kg, at its braking limit: slew, lift, tilt torque [kN m]; the "
            "largest share of a limit"] = ", ".join(kn_up(x) for x in tb) + f"; {up(share, 3)}"
    out[f"the slew's torque for every rad/s^2 of the base's yaw acceleration, the arm held with "
        f"{top:g} kg, carrying, reaching, reaching left [kN m]"] = ", ".join(
        kn_up(abs(hdmm.motion_basis(top, POSES[name])["alpha"][6]))
        for name in ("carry", "reach", "reach left"))
    return out


# ----------------------------------------------------------------------------
# The demands
# ----------------------------------------------------------------------------
def lift_margins(payload, accel=(0.0, 0.0), onset=None, offset=True, rise=0.0,
                 before=(0.0, 0.0)):
    """A simulated lift from carrying to reaching (hdmm.model.simulated_lift)
    on servos of stiffness LIFT_KP, their targets offset against gravity or
    not, the base's acceleration going from before to accel at the onset,
    along a half-cosine over rise seconds: the least distance of the
    simulated centre of pressure inside the contacts and when; the rigid
    model's distance (hdmm.labels.margin) for the planned pose under the
    acceleration acting, its least, and the largest shortfall of the first
    below it at the same instant, with its time and the rigid model's
    distance then; the rigid model's least distance at an instant at which
    the shortfall exceeds the allowance (inf if at none); the largest excess;
    the servos' largest torques; the front-most point's farthest forward of
    the base's middle at the simulated angles; the base's largest travel on
    its contacts; and the arm's largest departure from the planned pose at
    the end of the run."""
    r = hdmm.simulated_lift(payload, hdmm.POSES["carry"], hdmm.POSES["reach"], LIFT_TIME,
                            accel=accel, onset=onset, kp=LIFT_KP, offset=offset, rise=rise,
                            before=before,
                            hold=2.0 if onset is None else
                            max(2.0, ONSETS[-1] - LIFT_TIME + hl.RISE + 2.0))
    sim = np.minimum(hdmm.AXLE - np.abs(r["cop"][:, 0]), hdmm.HALF_TRACK - np.abs(r["cop"][:, 1]))
    q0, q1 = np.array(hdmm.POSES["carry"]), np.array(hdmm.POSES["reach"])

    def planned(t):
        s = 0.0 if t <= 0.0 else 1.0 if t >= LIFT_TIME else \
            0.5 * (1.0 - math.cos(math.pi * t / LIFT_TIME))
        return tuple(q0 + (q1 - q0) * s)

    static = np.array([hl.margin(planned(t), payload, *a)[0] for t, a in zip(r["t"], r["accel"])])
    diff = sim - static
    k, j = int(np.argmin(diff)), int(np.argmin(sim))
    over = -diff > hl.ALLOWANCE
    return dict(sim=float(sim[j]), at=float(r["t"][j]), static=float(static.min()),
                short=float(-diff[k]), when=float(r["t"][k]), static_then=float(static[k]),
                over=float(static[over].min()) if np.any(over) else math.inf,
                excess=float(diff.max()), torque=np.abs(r["torque"]).max(axis=0),
                front=max(hl.front_point(tuple(q)) for q in r["q"]),
                travel=float(np.linalg.norm(r["base"], axis=1).max()),
                droop=float(np.abs(r["q"][-1] - np.array(hdmm.POSES["reach"])).max()))


def rigid_least(payload, b):
    """The rigid model's least distance of the zero-moment point inside the
    contacts along the planned lift, the recovery braking at b, for one
    payload, without the allowance [m]."""
    return min(hl.margin(q, payload, ax=-b)[0] for _, q in hl.lift_path())


def torque_text(torque):
    return (", ".join(kn_up(x) for x in torque) + "; "
            f"{up(torque[1] / hl.ARM_LIMITS[1], 3)}, {up(torque[2] / hl.ARM_LIMITS[2], 3)}")


def lift_facts(rows):
    out = {}
    lo, top = hl.DECLARED.payloads
    dm = hl.lift_demand()
    out[f"a lift from carrying to reaching, {lo:g} to {top:g} kg: tipping reserve at rest before and "
        "after [m]; braking limit at the floor before and after [m/s^2]; what binds after"] = (
        f"{down(dm['rest_before'], 4)}, {down(dm['rest_after'], 4)}; {down(dm['b_before'])}, "
        f"{down(dm['b_after'])}; {dm['which_after']}")
    out["the tipping reserve that the lift consumes, statically: at rest; at the recovery's "
        "largest deceleration after the lift [m]"] = \
        f"{up(dm['consumed_rest'], 4)}; {up(dm['consumed'], 4)} at {down(dm['at'])} m/s^2"
    out["the demand: the floor plus the static consumption [m], and the requirement, on the "
        "levels' grid [m]; a plan in force that holds it, carrying: its recovery's largest "
        "deceleration within the forecast horizon [m/s^2], what binds, and beyond it, under the "
        "global capability; a plan that begins the lift and brakes at that deceleration: its "
        "tipping reserve along the lift, within the horizon and beyond it [m]"] = (
        f"{up(hl.FLOOR + dm['consumed'], 4)}, {dm['requirement']:.2f}; {down(dm['b_req'], 3)}, "
        f"{dm['which_req']}; {down(dm['b_req_global'], 3)}, {dm['which_global']}; "
        f"{down(dm['lift_level'], 4)}, {down(dm['lift_level_global'], 4)}")
    carry = hdmm.POSES["carry"]
    parts = []
    for payload, b in ((top, dm["b_req"]), (hl.GLOBAL.payloads[1], dm["b_req_global"])):
        rigid = hl.margin(carry, payload, ax=-b)[0]
        held_runs = [held(payload, carry, (-b, 0.0), hl.RISE, before=before)
                     for before in ((0.0, 0.0), (hl.DRIVE, 0.0))]
        runs = [r["least"] for r in held_runs]
        SHORT["least"] += [rigid - x for x in runs]
        SHORT["held"] += [r["instant"] for r in held_runs]
        SHORT["torque"] += [shares(r["torque"]) for r in held_runs]
        parts.append(f"{payload:g} kg at {down(b, 3)} m/s^2: {down(rigid, 4)}; "
                     f"{down(runs[0], 4)}, {down(runs[1], 4)}")
    out["the plan in force at the requirement, carrying, the arm held on its servos, the base braking "
        "with the actuators' rise, within the forecast horizon and beyond it: the rigid model's "
        "zero-moment point inside the contacts [m]; the simulated centre of pressure's least from "
        "cruising and from the drive's largest acceleration [m]"] = "; ".join(parts)
    rest = lift_margins(top)
    rigid = rigid_least(top, 0.0)
    SHORT["least"].append(rigid - rest["sim"])
    SHORT["instant"].append(rest["short"])
    SHORT["over"].append(rest["over"])
    SHORT["torque"].append(shares(rest["torque"]))
    out[f"the simulated lift, {LIFT_TIME:g} s, {top:g} kg, at rest, the servos offset against "
        "gravity: least distance of the centre of pressure inside the contacts [m]; the rigid "
        "model's least along the planned lift [m]; the shortfall of the first below the second "
        "[mm]; the largest shortfall below the rigid model for the planned pose at the same "
        "instant [mm], when [s]; the largest excess [mm]"] = (
        f"{down(rest['sim'], 4)}; {down(rigid, 4)}; {up(1000 * (rigid - rest['sim']))}; "
        f"{up(1000 * rest['short'])}, {rest['when']:.2f}; {down(1000 * rest['excess'])}")
    out["the same: the servos' largest torques, slew, lift, tilt [kN m]; the shares of the lift's "
        "and the tilt's limits"] = torque_text(rest["torque"])
    gtop = hl.GLOBAL.payloads[1]
    b_end_global = hl.braking_limit(hdmm.POSES["reach"], hl.GLOBAL)[0]
    drive = (hl.DRIVE, 0.0)
    sweeps = (("within the forecast horizon, at the requirement's braking limit, the servos offset "
               "against gravity", top, dm["b_req"], True, drive),
              ("within the forecast horizon, at the end pose's braking limit, the servos offset "
               "against gravity", top, dm["b_after"], True, drive),
              ("beyond the forecast horizon, at the requirement's braking limit, the servos offset "
               "against gravity", gtop, dm["b_req_global"], True, drive),
              ("beyond the forecast horizon, at the end pose's braking limit, the servos offset "
               "against gravity", gtop, b_end_global, True, drive),
              ("within the forecast horizon, at the requirement's braking limit, the servos offset "
               "against gravity", top, dm["b_req"], True, (0.0, 0.0)),
              ("within the forecast horizon, at the requirement's braking limit, the servos not "
               "offset", top, dm["b_req"], False, drive))
    torques = []
    for name, payload, b, offset, before in sweeps:
        runs = [(onset, lift_margins(payload, accel=(-b, 0.0), onset=onset, offset=offset,
                                     rise=hl.RISE, before=before))
                for onset in ONSETS]
        o_least, least = min(runs, key=lambda t: t[1]["sim"])
        o_short, short = max(runs, key=lambda t: t[1]["short"])
        rigid = rigid_least(payload, b)
        SHORT["least"].append(rigid - least["sim"])
        SHORT["instant"].append(short["short"])
        SHORT["over"].append(min(r["over"] for _, r in runs))
        torques.append(np.max([r["torque"] for _, r in runs], axis=0))
        SHORT["torque"].append(shares(torques[-1]))
        start = "the drive's largest acceleration" if before[0] else "cruising"
        head = (f"the simulated lift, {payload:g} kg, {name}, the base braking at {down(b, 3)} "
                f"m/s^2 with the actuators' rise from {start}, from an onset of the sweep on")
        out[head + ": the least distance of the centre of pressure inside the contacts [m], its "
            "onset [s], how long after it [s]; the rigid model's least along the planned lift [m]; "
            "the shortfall of the first below the second [mm]; the largest shortfall below the "
            "rigid model for the planned pose at the same instant [mm], its onset [s], when [s], "
            "the rigid model's distance then [m]; the rigid model's least distance at an instant "
            "at which the shortfall exceeds the allowance [m]"] = (
            f"{down(least['sim'], 4)}, {o_least:.2f}, {up(least['at'] - o_least)}; "
            f"{down(rigid, 4)}; {up(1000 * (rigid - least['sim']))}; "
            f"{up(1000 * short['short'])}, {o_short:.2f}, {short['when']:.2f}, "
            f"{down(short['static_then'], 4)}; "
            + ("none" if math.isinf(min(r["over"] for _, r in runs))
               else down(min(r["over"] for _, r in runs), 4)))
        excess = max(r["front"] for _, r in runs) - max(hl.front_point(q) for _, q in hl.lift_path())
        out[head + ": the servos' largest torques, slew, lift, tilt [kN m]; the shares of the "
            "lift's and the tilt's limits; the front-most point's largest excess over the rigid "
            "model's farthest [mm]; the base's largest travel on its contacts [mm]; the arm's "
            "largest departure from the planned pose at the end of the run [deg]"] = (
            torque_text(torques[-1]) + "; "
            + (up(1000 * excess) if excess > 0 else "none") + "; "
            + f"{up(1000 * max(r['travel'] for _, r in runs))}; "
            + f"{dup(max(r['droop'] for _, r in runs))}")
    most = np.max(torques + [rest["torque"]], axis=0)
    out["the servos' largest torques over every simulated lift, lift, tilt [kN m]; the shares of "
        "their limits"] = (f"{kn_up(most[1])}, {kn_up(most[2])}; "
                           f"{up(most[1] / hl.ARM_LIMITS[1], 3)}, {up(most[2] / hl.ARM_LIMITS[2], 3)}")
    out["with a stop line ahead: the front-most point's travel, the clearance that the lift "
        "consumes [m]; the requirement [m]; the clearance after the lift [m]"] = \
        (f"{up(dm['forward'], 4)}; {dm['clear_requirement']:.2f}; "
         f"{down(dm['clear_after'], 4)}")
    s0 = hl.stop_distance(hl.SPEED, dm["b_before"])
    s1 = hl.stop_distance(hl.SPEED, dm["b_req"])
    out["at the top speed, carrying, the deceleration rising and fading over the actuators' rise: the "
        "recovery's stop at the floor's braking limit and at the requirement's [m]; the difference "
        "[m]; the same stops were the braking to begin at once [m]"] = (
        f"{up(s0, 3)}, {up(s1, 3)}; {up(s1 - s0, 3)}; "
        f"{up(hl.stop_distance(hl.SPEED, dm['b_before'], rise=0.0), 3)}, "
        f"{up(hl.stop_distance(hl.SPEED, dm['b_req'], rise=0.0), 3)}")
    for k in range(0, 41):
        b = 0.1 * k
        rows.append({"b": b,
                     "carry": hl.tipping_reserve(hdmm.POSES["carry"], hl.DECLARED, ax=-b)[0],
                     "reach": hl.tipping_reserve(hdmm.POSES["reach"], hl.DECLARED, ax=-b)[0]})
    return out


def turn_facts():
    out = {}
    for name in ("carry", "reach", "reach left"):
        q = POSES[name]
        b, _ = hl.braking_limit(q)
        parts = []
        for radius in RADII:
            v, which = hl.turn_speed(q, radius)
            if which != "top speed":
                dist, time_ = hl.slowing(hl.SPEED, b, until=v)
                parts.append(f"R {radius:g} m: {down(v)} m/s, {which}, {down(v * v / radius)} "
                             f"m/s^2; slowing {up(dist)} m, {up(time_)} s")
            else:
                parts.append(f"R {radius:g} m: {hl.SPEED:g} m/s, the top speed")
        out[f"{name}, declared: by radius, the speed that a turn allows, what binds, and its lateral "
            "acceleration; the distance and time to slow to it from the top speed at the braking "
            f"limit ({down(b)} m/s^2), the deceleration rising and fading over the actuators' "
            "rise"] = "; ".join(parts)
        v5, _ = hl.turn_speed(q, RADII[1])
        dist, time_ = hl.slowing(v5, hl.TURN_BRAKING)
        out[f"{name}, declared: a stop at the braking in a turn from the speed of the turn of "
            f"{RADII[1]:g} m [m], [s]"] = f"{up(dist)}, {up(time_)}"
    q = POSES["reach left"]
    sides = [[down(hl.turn_speed(q, radius, sides=(side,))[0]) for radius in RADII[:2]]
             for side in (1.0, -1.0)]
    out[f"reach left, declared: the speed that a turn of {RADII[0]:g} and of {RADII[1]:g} m allows "
        "to the left; to the right [m/s]"] = "; ".join(", ".join(x) for x in sides)
    cap = hl.level_down(hl.tipping_reserve(q, hl.GLOBAL)[0])
    speeds = [hl.turn_speed(q, radius, level=cap) for radius in RADII[:2]]
    out["reach left, declared: the largest level that a demand for a turn can ask, within the "
        "rest's tipping reserve under the global capability [m]; the speed that a turn of "
        f"{RADII[0]:g} and of {RADII[1]:g} m allows at that level [m/s], what binds"] = (
        f"{cap:.2f}; {speeds_text(speeds)}")
    return out


def allowance_facts():
    most = np.max(SHORT["torque"], axis=0)
    return {"the largest shortfall of a simulated least distance below the rigid model's least, over "
            "every run with the actuators' rise: the braking limits of the four poses under the "
            "three labels, the turns, the plans in force and the lifts [mm]; the allowance [mm]":
            f"{up(1000 * max(SHORT['least']))}; {1000 * hl.ALLOWANCE:g}",
            "the largest shortfall below the rigid model for the planned pose at a single instant: "
            "with the arm held, over the same runs; during the lifts [mm]; where a lift's exceeds "
            "the allowance, the rigid model's least distance [m]":
            f"{up(1000 * max(SHORT['held']))}; {up(1000 * max(SHORT['instant']))}; "
            + ("none" if math.isinf(min(SHORT["over"])) else down(min(SHORT["over"]), 4)),
            "the servos' largest torques over the same runs, as shares of the slew's, the lift's and "
            "the tilt's limits; the base's largest creep, on the labels' least friction [mm]; the "
            "front-most point's largest excess in a stop there [mm]":
            ", ".join(up(x, 3) for x in most) + f"; {up(1000 * max(SHORT['creep']))}; "
            f"{up(1000 * max(SHORT['front']))}"}


# ----------------------------------------------------------------------------
# The record file and main
# ----------------------------------------------------------------------------
def hardware():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "not reported"


def write_record(path, rec):
    share, floor, grid = (s.strip() for s in rec["UR5e: share of each joint's torque for the "
                                                 "label; reserve floor [m]; levels' grid [m]"]
                          .split(";"))
    lift = rec["mobile manipulator: the lift's duration [s]; the servos' stiffness [N m/rad], "
               "damping [N m s/rad]; the recovery's onsets [s]"].split("; ")
    lines = [
        "Record of the experiment of Section 14.2: labels, reserves and demands for the UR5e and "
        "the heavy-duty mobile manipulator",
        "",
        "model and asset provenance",
        "  the UR5e of Section 14.1 (UR's description, release 3.6.0, mapped by arm/model.py), its "
        "payload a box of 0.1 m hanging from the flange, its mass and centre of mass declared",
        "  the synthetic mobile manipulator of Section 14.1 (hdmm/model.py), every number ours; "
        "its actuators' limits, labels, floors and lift ours (hdmm/labels.py, labels_demand.py)",
        "simulator and solver",
        f"  MuJoCo {rec['MuJoCo']}; NumPy, PyYAML and Python {rec['NumPy, PyYAML, Python']}; the "
        "UR5e at 0.5 ms with implicitfast, the post's contact with MuJoCo's defaults, solref; "
        "solimp " + rec["UR5e: the post's contact, solref; solimp (MuJoCo's defaults)"]
        + "; the mobile manipulator at 1 ms with implicitfast and elliptic cones, the ground's "
        "friction 1.5, its contacts' solref; solimp; ratio of frictional to normal impedance "
        + rec["mobile manipulator: the simulated contacts' solref; solimp; ratio of frictional "
              "to normal impedance"] + ", so that nothing slides; that ratio also at "
        + ", ".join(f"{r:g}" for r in RATIOS if r != hdmm.IMPRATIO) + " and 1 for comparison, "
        "and for the base's creep the labels' least friction, 0.5 and 0.3",
        "time steps",
        "  the UR5e's stops: 0.5 ms, the controller at 1 kHz; the mobile manipulator with its "
        "arm held: 1 ms, settled 2.5 s, the deceleration raised over 5 s, with the actuators' rise "
        "(braking from cruising and from the drive, stopping from the top speed and with no "
        "plateau) or at once, held at least 4 s after; through a turn: 1 ms, settled 2 s, at the "
        "turn's speed from 0.5 s before its entry, through it or braking in it from the middle of "
        "its entry, arc or exit or from where the test binds, or at rest in it for 0.5 s and "
        "driving off from where the test binds, held 2 s after; its lift: 1 "
        "ms, settled 2 s, the lift over "
        f"{lift[0]} s along a cosine profile, held 2 s, or, with the base braking from an onset "
        f"with the actuators' rise, onsets {lift[2]} s from the lift's start, held until 2.6 s "
        "after the last",
        "actuators and sensors",
        f"  UR5e: torque at the joints, limited to 150 and 28 N m; the labels use {share} of each "
        "limit; the controller of Section 14.1",
        "  mobile manipulator: brakes and drive [m/s^2], top speed [m/s], the rise of the "
        "base's acceleration along a half-cosine [s] "
        + rec["mobile manipulator: brakes, drive [m/s^2]; top speed [m/s]; the rise of the base's "
              "acceleration along a half-cosine [s]"]
        + "; the turns' transitions [m], their angle [deg], and the braking and the driving off "
        "in a turn [m/s^2] "
        + rec["mobile manipulator: the turns' transitions [m]; the turns' angle [deg]; the braking "
              "and the driving off in a turn [m/s^2]"]
        + "; the arm's torque limits, slew, lift, tilt [kN m], and the factor on the torque that "
        "the base's motion adds "
        + rec["mobile manipulator: the arm's torque limits, slew, lift, tilt [kN m]; the factor on "
              "the torque that the base's acceleration adds"]
        + "; in every simulated run the arm held or moved by position servos without limits, "
        f"stiffness and damping {lift[1]}, their targets offset by the torque that holds the "
        "planned pose under gravity alone, the base's motion not fed forward (and, for "
        "comparison in a lift, not offset); in a turn the base held in its own frame, every body "
        "feeling that frame's forces and torques; no sensors: the state from the simulator",
        "frames and units",
        "  SI units; the UR5e's angles those of tool0 about the base's axis; the mobile "
        "manipulator's x forward, y left, z up, the ground at z = 0",
        "uncertainty bounds",
        "  UR5e: the declared payload, mass [kg]; centre of mass [m]; the box's centre [m] "
        + rec["UR5e: declared mass [kg]; centre of mass from the flange [m]; the box's centre [m]"]
        + "; the global capability, mass [kg]; centre of mass [m] "
        + rec["UR5e: global capability, mass [kg]; centre of mass [m]"],
        "  mobile manipulator: the declared payload [kg]; friction; the global capability's "
        "payload [kg] and friction "
        + rec["mobile manipulator: declared payload [kg]; friction; global capability's payload "
              "[kg] and friction"],
        "constraints",
        f"  UR5e: no contact with the post, the clearance's floor {floor} m, its levels' grid "
        f"{grid} m; mobile manipulator: the zero-moment point inside the rectangle of the wheels' "
        "contacts and the front-most point behind a stop line, their floors [m], the allowance "
        "that both reserves keep for what the rigid model leaves out [m] and the levels' grid [m] "
        + rec["mobile manipulator: floors of the tipping reserve and the clearance [m]; the "
              "allowance that both keep for what the rigid model leaves out [m]; levels' grid "
              "[m]"],
        "terminal behaviour",
        "  UR5e: the base joint's stop at the label, the other joints holding the carry pose, then "
        "rest there; mobile manipulator: braking within the braking limit to rest with the arm "
        "held, or, during a lift, while the lift finishes, in a turn or its transitions at the "
        "braking in a turn, then rest on the brakes; from rest in a turn, driving off at the "
        "driving in a turn",
        "training and evaluation split",
        "  none: nothing learns in Section 14.2",
        "seeds",
        "  none: no random numbers",
        "hardware",
        f"  {platform.system()} {platform.machine()}, {hardware()}",
        "guarantee status",
        "  none: Section 14.2 states the labels, the reserves and the demand and checks them in "
        "the simulator; no "
        "plan is certified or governed",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow({k: f"{v:.6g}" for k, v in row.items()})


def main():
    ap = argparse.ArgumentParser(description="Section 14.2: labels, reserves and demands")
    ap.add_argument("--check", action="store_true", help="check every value of the summary")
    args = ap.parse_args()
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    arm_rows, reserve_rows = [], []
    labels, b, b_empty, b_global = arm_label_facts(arm_rows)
    reserve, latest = arm_reserve_facts(b, b_empty, b_global)
    facts = {"record": record_facts(),
             "the UR5e's labels": labels,
             "the UR5e's reserve": reserve,
             "the UR5e's stops": arm_stop_facts(b, b_empty, latest),
             "the mobile manipulator's limits": hdmm_label_facts(),
             "the rest": rest_facts(),
             "the simulated mobile manipulator": hdmm_simulated_facts(),
             "the arm's torques at the limits": hdmm_torque_facts(),
             "a lift": lift_facts(reserve_rows),
             "a turn": turn_facts(),
             "the allowance": allowance_facts()}
    write_csv(os.path.join(out_dir, "labels-arm.csv"), arm_rows)
    write_csv(os.path.join(out_dir, "labels-reserve.csv"), reserve_rows)
    write_record(os.path.join(out_dir, "labels-record.txt"), facts["record"])
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
        lines.append(f"[check] {n} values, among them every value that Section 14.2 quotes: " +
                     ("all as expected" if not failures else "FAILED: " + "; ".join(failures)))
    with open(os.path.join(out_dir, "labels-summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if failures:
        raise SystemExit(1)


EXPECTED = {   # every value of the summary but the versions of the libraries
    'record': {
        "UR5e: declared mass [kg]; centre of mass from the flange [m]; the box's centre [m]":
            '4.05 to 4.95; 0.025 to 0.075; 0.05',
        'UR5e: global capability, mass [kg]; centre of mass [m]':
            '0 to 5; 0 to 0.1',
        "UR5e: share of each joint's torque for the label; reserve floor [m]; levels' grid [m]":
            '0.9; 0.01; 0.001',
        "UR5e: the post's contact, solref; solimp (MuJoCo's defaults)":
            '0.02, 1; 0.9, 0.95, 0.001, 0.5, 2',
        "mobile manipulator: declared payload [kg]; friction; global capability's payload [kg] and friction":
            '243 to 297; 0.5; 0 to 300, 0.3',
        "mobile manipulator: brakes, drive [m/s^2]; top speed [m/s]; the rise of the base's acceleration along a half-cosine [s]":
            '4, 1.5; 3; 0.6',
        "mobile manipulator: the turns' transitions [m]; the turns' angle [deg]; the braking and the driving off in a turn [m/s^2]":
            '3; 90; 0.5, 0.5',
        "mobile manipulator: the arm's torque limits, slew, lift, tilt [kN m]; the factor on the torque that the base's acceleration adds":
            '8, 30, 12; 1.2',
        "mobile manipulator: floors of the tipping reserve and the clearance [m]; the allowance that both keep for what the rigid model leaves out [m]; levels' grid [m]":
            '0.1, 0.1; 0.005; 0.01',
        "mobile manipulator: the simulated contacts' solref; solimp; ratio of frictional to normal impedance":
            '0.02, 1; 0.9, 0.95, 0.001, 0.5, 2; 20',
        "mobile manipulator: the lift's duration [s]; the servos' stiffness [N m/rad], damping [N m s/rad]; the recovery's onsets [s]":
            '4; 2e+06, 100000; 0 to 6 every 0.1 and 3.2 to 4.4 every 0.02',
    },
    "the UR5e's labels": {
        'the declared interval: the label [rad/s^2]; the payload that binds, mass [kg], centre of mass [m]; the joint':
            '56.76; 4.95, 0.075; wrist 2 at the end',
        "the grid over the interval: each payload's own bound [rad/s^2], mass and centre of mass":
            '4.05 kg 0.025 m 65.27; 4.05 kg 0.05 m 65.27; 4.05 kg 0.075 m 65.27; 4.5 kg 0.025 m 61.88; 4.5 kg 0.05 m 61.88; 4.5 kg 0.075 m 61.88; 4.95 kg 0.025 m 58.83; 4.95 kg 0.05 m 58.83; 4.95 kg 0.075 m 56.76',
        "the same without the feedback's share [rad/s^2]; the joint":
            '63.06; wrist 2 at the end',
        "with the feedback's share: the nominal payload's (4.5 kg, 0.05 m), the empty arm's and the global capability's label [rad/s^2]; the joints":
            '61.88, 128.69, 49.56; base at the start, base at the start, wrist 2 at the end',
        "the stop from full speed at the label [deg]: the declared interval's, the nominal's, the empty arm's, the global capability's":
            '4.982, 4.569, 2.198, 5.705',
        "the nominal payload's label above the declared interval's [%]":
            '9.0',
    },
    "the UR5e's reserve": {
        "the payload's clearance from the post at the place [m]":
            '0.0578',
        "a stop from full speed at the place: its point of rest [deg] and the clearance there, the plan's reserve [m], under the declared label; under the global capability's":
            '-64.982, 0.0184; -65.705, 0.0127',
        'the angle at which the clearance falls to the floor [deg]':
            '-66.054',
        "the latest angle from which a stop from full speed keeps the floor [deg], beyond the place [deg]: under the declared label; the global capability's; the empty arm's, the box in place":
            '-61.072, 1.072; -60.349, 0.349; -63.857, 3.857',
        "the global capability's label: the heaviest payload that it admits, on a grid of 0.25 kg, with its centre of mass at 0, 0.025, 0.05, 0.075, 0.1 m [kg]; the bound of 7 kg there, every one below the label [rad/s^2]":
            '6.5, 6.5, 6.5, 5.5, 5; 48.03, 48.03, 47.35, 40.87, 35.96',
        "at rest in the carry pose: the largest share of a joint's torque that holds the arm, every payload of the global capability's grid up to the rated 5 kg; every payload up to 7 kg with its centre of mass in the box, on the grid of 0.25 kg and 0.025 m":
            '0.312; 0.375',
    },
    "the UR5e's stops": {
        'the 9 payloads of the grid, stopped at the label from full speed at the place: largest torque asked over its limit; updates clipped, all joints; largest error of the base [deg]; largest overrun of the commanded rest [deg]; least clearance [m]':
            '0.898; 0; 0.050; 0.000; 0.0184',
        "the interval's heaviest corner (4.95 kg, 0.075 m), stopped at the label from full speed at the latest angle: the joint asked the largest share of its torque, and any other asked more than its limit, with the shares; the updates in which each joint's torque is clipped; largest error of the base [deg]; beyond the commanded rest [deg]; wrist 2's largest departure from the pose [deg]; least clearance [m]":
            'wrist 2 0.898; none; 0.050; 0.000; 0.001; 0.0100',
        "outside the declaration, 6 kg at 0.05 m: its own bound with and without the share [rad/s^2], the joint; stopped at the label from full speed at the place: the joint asked the largest share of its torque, and any other asked more than its limit, with the shares; the updates in which each joint's torque is clipped; largest error of the base [deg]; beyond the commanded rest [deg]; wrist 2's largest departure from the pose [deg]; least clearance [m]":
            '52.75, 58.61, base; base 0.969; none; 0.050; 0.000; 0.001; 0.0184',
        "outside the declaration, 4.5 kg at 0.1 m: its own bound with and without the share [rad/s^2], the joint; stopped at the label from full speed at the place: the joint asked the largest share of its torque, and any other asked more than its limit, with the shares; the updates in which each joint's torque is clipped; largest error of the base [deg]; beyond the commanded rest [deg]; wrist 2's largest departure from the pose [deg]; least clearance [m]":
            '54.74, 60.82, wrist 2; wrist 2 0.932; none; 0.050; 0.000; 0.001; 0.0184',
        "outside the declaration, 5 kg at 0.1 m: its own bound with and without the share [rad/s^2], the joint; stopped at the label from full speed at the place: the joint asked the largest share of its torque, and any other asked more than its limit, with the shares; the updates in which each joint's torque is clipped; largest error of the base [deg]; beyond the commanded rest [deg]; wrist 2's largest departure from the pose [deg]; least clearance [m]":
            '49.56, 55.07, wrist 2; wrist 2 1.083; wrist 2 42; 0.131; 0.000; 0.339; 0.0183',
        "outside the declaration, 6 kg at 0.1 m: its own bound with and without the share [rad/s^2], the joint; stopped at the label from full speed at the place: the joint asked the largest share of its torque, and any other asked more than its limit, with the shares; the updates in which each joint's torque is clipped; largest error of the base [deg]; beyond the commanded rest [deg]; wrist 2's largest departure from the pose [deg]; least clearance [m]":
            '41.68, 46.31, wrist 2; wrist 2 2.090; wrist 2 56; 1.613; 0.000; 6.108; 0.0112',
        "outside the declaration, 6 kg at 0.1 m, stopped at the label from full speed at the latest angle: the joint asked the largest share of its torque, and any other asked more than its limit, with the shares; the updates in which each joint's torque is clipped; largest error of the base [deg]; beyond the commanded rest [deg]; wrist 2's largest departure from the pose [deg]; least clearance [m]":
            'wrist 2 2.090; wrist 2 56; 1.613; 0.000; 6.108; 0.0028',
        "the interval's heaviest corner under the empty arm's label (128.69 rad/s^2), from full speed at the place: the joint asked the largest share of its torque, and any other asked more than its limit, with the shares; the updates in which each joint's torque is clipped; largest error of the base [deg]; beyond the commanded rest [deg]; wrist 2's largest departure from the pose [deg]; least clearance [m]":
            'wrist 2 3.773, base 3.613; base 40, wrist 2 42; 2.496; 2.496; 0.863; 0.0217',
        "the same from the latest angle that the empty arm's label allows with the box in place: the joint asked the largest share of its torque, and any other asked more than its limit, with the shares; the updates in which each joint's torque is clipped; largest error of the base [deg]; beyond the commanded rest [deg]; wrist 2's largest departure from the pose [deg]; least clearance [m]":
            'wrist 2 3.773, base 3.613; base 35, wrist 2 34; 2.364; 2.364; 1.494; -0.0049',
        "the heaviest corner under the empty arm's label from the place: its farthest angle [deg]; the stop's angle in all [deg]":
            '-64.693; 4.693',
        "the same from the latest angle that the empty arm's label allows: how far the box enters the post [mm]":
            '4.9',
        "6 kg against the declaration's heaviest payload: heavier by [%]":
            '21',
        "the heaviest corner's own stop without the share: its angle from full speed [deg]":
            '4.484',
    },
    "the mobile manipulator's limits": {
        'stowed, empty: tipping reserve at rest [m], its edge; braking limit [m/s^2], what binds; the speed in a turn of 3, 5 and 10 m [m/s], what binds':
            '0.8450, left; 4.00, brakes; 3.00 top speed, 3.00 top speed, 3.00 top speed',
        'stowed, declared: tipping reserve at rest [m], its edge; braking limit [m/s^2], what binds; the speed in a turn of 3, 5 and 10 m [m/s], what binds':
            '0.7628, front; 4.00, brakes; 3.00 top speed, 3.00 top speed, 3.00 top speed',
        'stowed, declared: the four limits on the braking at the floor, the brakes, the ground, tipping, the arm [m/s^2]':
            '4.00, 4.90, 6.48, 10.26',
        'stowed: tipping reserve at rest under the global capability [m], its edge':
            '0.7618, front',
        'stowed: the front-most point [m]':
            '2.000',
        'carry, empty: tipping reserve at rest [m], its edge; braking limit [m/s^2], what binds; the speed in a turn of 3, 5 and 10 m [m/s], what binds':
            '0.6919, front; 4.00, brakes; 2.61 slew, 3.00 top speed, 3.00 top speed',
        'carry, declared: tipping reserve at rest [m], its edge; braking limit [m/s^2], what binds; the speed in a turn of 3, 5 and 10 m [m/s], what binds':
            '0.4423, front; 4.00, brakes; 1.85 slew, 2.39 slew, 3.00 top speed',
        'carry, declared: the four limits on the braking at the floor, the brakes, the ground, tipping, the arm [m/s^2]':
            '4.00, 4.90, 4.00, 20.26',
        'carry: tipping reserve at rest under the global capability [m], its edge':
            '0.4399, front',
        'carry: the front-most point [m]':
            '4.341',
        'reach, empty: tipping reserve at rest [m], its edge; braking limit [m/s^2], what binds; the speed in a turn of 3, 5 and 10 m [m/s], what binds':
            '0.6431, front; 4.00, brakes; 2.29 slew, 2.96 slew, 3.00 top speed',
        'reach, declared: tipping reserve at rest [m], its edge; braking limit [m/s^2], what binds; the speed in a turn of 3, 5 and 10 m [m/s], what binds':
            '0.3394, front; 2.70, tipping; 1.45 ground, 1.96 ground, 2.84 ground',
        'reach, declared: the four limits on the braking at the floor, the brakes, the ground, tipping, the arm [m/s^2]':
            '4.00, 4.90, 2.70, inf',
        'reach: tipping reserve at rest under the global capability [m], its edge':
            '0.3365, front',
        'reach: the front-most point [m]':
            '5.200',
        'reach left, empty: tipping reserve at rest [m], its edge; braking limit [m/s^2], what binds; the speed in a turn of 3, 5 and 10 m [m/s], what binds':
            '0.4871, left; 4.00, brakes; 2.64 slew, 3.00 top speed, 3.00 top speed',
        'reach left, declared: tipping reserve at rest [m], its edge; braking limit [m/s^2], what binds; the speed in a turn of 3, 5 and 10 m [m/s], what binds':
            '0.2172, left; 2.38, arm; 1.63 slew, 2.17 slew, 3.00 top speed',
        'reach left, declared: the four limits on the braking at the floor, the brakes, the ground, tipping, the arm [m/s^2]':
            '4.00, 4.90, 9.81, 2.38',
        'reach left: tipping reserve at rest under the global capability [m], its edge':
            '0.2146, left',
        'reach left: the front-most point [m]':
            '1.500',
        "reach, declared: the braking limit below the brakes' [%]":
            '32',
        'under the global capability: braking limit [m/s^2], what binds, stowed, carrying, reaching, reaching left':
            '2.94, friction; 2.94, friction; 2.67, tipping; 2.37, arm',
        'under the global capability: the speed in a turn of 3, 5 and 10 m [m/s], what binds, stowed, carrying, reaching, reaching left':
            '2.46 ground, 3.00 top speed, 3.00 top speed; 1.45 ground, 1.95 ground, 2.81 ground; 1.01 ground, 1.45 ground, 2.15 ground; 1.62 slew, 2.16 slew, 3.00 top speed',
        "every turn's speed: every lower speed on a grid of 0.5 m/s holds too":
            'yes',
    },
    'the rest': {
        "the least tipping reserve at rest over the arm's ranges with 0 to 300 kg [m]; the slew [deg], at full reach; the edge":
            '0.2146; 90.0; left',
        "the arm's torques at rest at full reach with 300 kg, lift, tilt [kN m]; their shares of the limits":
            '27.52, 9.62; 0.918, 0.802',
    },
    'the simulated mobile manipulator': {
        "the declared label's braking limits in the four poses, the arm held on its servos, the deceleration raised over 5 s: the largest shortfall of the simulated centre of pressure's least below the rigid model's [mm]":
            '0.64',
        "empty (0 kg), at each braking limit, the arm held on its servos, with the actuators' rise, braking from cruising and from the drive's largest acceleration and stopping: the limit [m/s^2]; the rigid model's zero-moment point inside the contacts and the simulated centre of pressure's least [m]; the largest shortfall at an instant [mm]; the servos' largest torques, slew/lift/tilt, as shares of their limits":
            'stowed 4.00: 0.4752, 0.4736, 1.64, 0.001/0.259/0.128; carry 4.00: 0.3474, 0.3471, 0.34, 0.000/0.432/0.237; reach 4.00: 0.3072, 0.3070, 0.19, 0.000/0.486/0.311; reach left 4.00: 0.4921, 0.4920, 0.10, 0.777/0.486/0.311',
        "declared (297 kg), at each braking limit, the arm held on its servos, with the actuators' rise, braking from cruising and from the drive's largest acceleration and stopping: the limit [m/s^2]; the rigid model's zero-moment point inside the contacts and the simulated centre of pressure's least [m]; the largest shortfall at an instant [mm]; the servos' largest torques, slew/lift/tilt, as shares of their limits":
            'stowed 4.00: 0.3592, 0.3561, 3.10, 0.001/0.380/0.331; carry 4.00: 0.1051, 0.1041, 1.94, 0.001/0.771/0.605; reach 2.70: 0.1050, 0.1047, 0.29, 0.001/0.914/0.797; reach left 2.38: 0.2222, 0.2219, 0.22, 0.944/0.914/0.797',
        "global (300 kg), at each braking limit, the arm held on its servos, with the actuators' rise, braking from cruising and from the drive's largest acceleration and stopping: the limit [m/s^2]; the rigid model's zero-moment point inside the contacts and the simulated centre of pressure's least [m]; the largest shortfall at an instant [mm]; the servos' largest torques, slew/lift/tilt, as shares of their limits":
            'stowed 2.94: 0.4661, 0.4635, 2.55, 0.001/0.360/0.245; carry 2.94: 0.1932, 0.1924, 1.60, 0.001/0.774/0.609; reach 2.67: 0.1050, 0.1047, 0.29, 0.001/0.918/0.802; reach left 2.37: 0.2196, 0.2194, 0.22, 0.945/0.918/0.802',
        "empty (0 kg), the turns of 3, 5 and 10 m at their speeds in the four poses, the arm held on its servos, through the turn, braking in it from the middle of its entry, its arc and its exit and from where the test binds, and driving off in it from rest where the test binds: the runs; the least of the simulated centre of pressure inside the contacts [m], the rigid model's least in that run [m]; the largest shortfall at an instant [mm]; the servos' largest torques, slew/lift/tilt, as shares of their limits":
            '90; 0.2558, 0.2565; 3.19; 0.850/0.486/0.311',
        "declared (297 kg), the turns of 3, 5 and 10 m at their speeds in the four poses, the arm held on its servos, through the turn, braking in it from the middle of its entry, its arc and its exit and from where the test binds, and driving off in it from rest where the test binds: the runs; the least of the simulated centre of pressure inside the contacts [m], the rigid model's least in that run [m]; the largest shortfall at an instant [mm]; the servos' largest torques, slew/lift/tilt, as shares of their limits":
            '90; 0.1180, 0.1189; 3.04; 0.875/0.913/0.797',
        "global (300 kg), the turns of 3, 5 and 10 m at their speeds in the four poses, the arm held on its servos, through the turn, braking in it from the middle of its entry, its arc and its exit and from where the test binds, and driving off in it from rest where the test binds: the runs; the least of the simulated centre of pressure inside the contacts [m], the rigid model's least in that run [m]; the largest shortfall at an instant [mm]; the servos' largest torques, slew/lift/tilt, as shares of their limits":
            '90; 0.1166, 0.1174; 1.81; 0.872/0.918/0.802',
        "without the actuators' rise, at once: carry, 297 kg, braking at its limit, the simulated centre of pressure's least from cruising and from the drive's largest acceleration [m]":
            '0.0803, 0.0713',
        "MuJoCo's ratio of frictional to normal impedance 5: the shortfall at the braking limit [mm], raised over 5 s and with the rise, carrying, reaching, reaching left":
            '105.17, 105.17; 0.06, 0.27; 1.63, 1.01',
        "MuJoCo's ratio of frictional to normal impedance 10: the shortfall at the braking limit [mm], raised over 5 s and with the rise, carrying, reaching, reaching left":
            '0.45, 0.86; 0.06, 0.23; 1.08, 0.43',
        "MuJoCo's ratio of frictional to normal impedance 20: the shortfall at the braking limit [mm], raised over 5 s and with the rise, carrying, reaching, reaching left":
            '0.45, 0.86; 0.06, 0.21; 0.64, 0.14',
        "MuJoCo's ratio of frictional to normal impedance 50: the shortfall at the braking limit [mm], raised over 5 s and with the rise, carrying, reaching, reaching left":
            '0.45, 0.87; 0.06, 0.20; 0.30, 0.16',
        'the largest change of a shortfall between the ratios 20 and 50 [mm]':
            '0.35',
        "reaching, braking at its limit raised over 5 s, MuJoCo's ratio of frictional to normal impedance 1, friction 1.5: the base's creep over the run [mm]; the fewest wheel contacts at a step; the wheels that lose their contact":
            '7.53; 2; rl, rr',
        "reaching, braking at its limit raised over 5 s, MuJoCo's ratio of frictional to normal impedance 20, friction 1.5: the base's creep over the run [mm]; the fewest wheel contacts at a step; the wheels that lose their contact":
            '0.38; 4; none',
        "stopping from the top speed at each braking limit of the four poses, the deceleration rising from cruising and from the drive's largest acceleration, on the label's least friction: the base's largest creep on its contacts [mm]; the front-most point's largest excess over the rigid machine's, the arm's swing and the creep together [mm]":
            'empty, on friction 0.5: 0.07, 1.65; declared, on friction 0.5: 0.08, 4.03; global, on friction 0.3: 0.26, 3.20',
        "the turns at their speeds on the label's least friction, through the turn, braking in it from the middle of its exit and from where the test binds, and driving off in it from rest where the test binds: the base's largest creep on its contacts [mm], and its largest turn on them [mrad]":
            'empty, on friction 0.5: 0.10, 0.023; declared, on friction 0.5: 0.11, 0.034; global, on friction 0.3: 0.31, 0.029',
    },
    "the arm's torques at the limits": {
        'carry, 297 kg, at its braking limit: slew, lift, tilt torque [kN m]; the largest share of a limit':
            '0.00, 22.41, 3.16; 0.747',
        'reach, 297 kg, at its braking limit: slew, lift, tilt torque [kN m]; the largest share of a limit':
            '0.00, 27.39, 9.56; 0.913',
        'reach left, 297 kg, at its braking limit: slew, lift, tilt torque [kN m]; the largest share of a limit':
            '6.67, 27.39, 9.56; 0.913',
        "the slew's torque for every rad/s^2 of the base's yaw acceleration, the arm held with 297 kg, carrying, reaching, reaching left [kN m]":
            '8.55, 12.22, 10.54',
    },
    'a lift': {
        'a lift from carrying to reaching, 243 to 297 kg: tipping reserve at rest before and after [m]; braking limit at the floor before and after [m/s^2]; what binds after':
            '0.4423, 0.3394; 4.00, 2.70; tipping',
        "the tipping reserve that the lift consumes, statically: at rest; at the recovery's largest deceleration after the lift [m]":
            '0.1029; 0.1107 at 2.70 m/s^2',
        "the demand: the floor plus the static consumption [m], and the requirement, on the levels' grid [m]; a plan in force that holds it, carrying: its recovery's largest deceleration within the forecast horizon [m/s^2], what binds, and beyond it, under the global capability; a plan that begins the lift and brakes at that deceleration: its tipping reserve along the lift, within the horizon and beyond it [m]":
            '0.2107, 0.22; 2.599, tipping; 2.572, tipping; 0.1096, 0.1091',
        "the plan in force at the requirement, carrying, the arm held on its servos, the base braking with the actuators' rise, within the forecast horizon and beyond it: the rigid model's zero-moment point inside the contacts [m]; the simulated centre of pressure's least from cruising and from the drive's largest acceleration [m]":
            '297 kg at 2.599 m/s^2: 0.2250; 0.2244, 0.2242; 300 kg at 2.572 m/s^2: 0.2250; 0.2244, 0.2242',
        "the simulated lift, 4 s, 297 kg, at rest, the servos offset against gravity: least distance of the centre of pressure inside the contacts [m]; the rigid model's least along the planned lift [m]; the shortfall of the first below the second [mm]; the largest shortfall below the rigid model for the planned pose at the same instant [mm], when [s]; the largest excess [mm]":
            '0.3407; 0.3444; 3.67; 5.33, 1.87; 9.73',
        "the same: the servos' largest torques, slew, lift, tilt [kN m]; the shares of the lift's and the tilt's limits":
            '0.00, 27.56, 9.63; 0.919, 0.802',
        "the simulated lift, 297 kg, within the forecast horizon, at the requirement's braking limit, the servos offset against gravity, the base braking at 2.599 m/s^2 with the actuators' rise from the drive's largest acceleration, from an onset of the sweep on: the least distance of the centre of pressure inside the contacts [m], its onset [s], how long after it [s]; the rigid model's least along the planned lift [m]; the shortfall of the first below the second [mm]; the largest shortfall below the rigid model for the planned pose at the same instant [mm], its onset [s], when [s], the rigid model's distance then [m]; the rigid model's least distance at an instant at which the shortfall exceeds the allowance [m]":
            '0.1110, 3.66, 0.60; 0.1146; 3.68; 6.86, 1.60, 1.79, 0.4250; 0.1428',
        "the simulated lift, 297 kg, within the forecast horizon, at the requirement's braking limit, the servos offset against gravity, the base braking at 2.599 m/s^2 with the actuators' rise from the drive's largest acceleration, from an onset of the sweep on: the servos' largest torques, slew, lift, tilt [kN m]; the shares of the lift's and the tilt's limits; the front-most point's largest excess over the rigid model's farthest [mm]; the base's largest travel on its contacts [mm]; the arm's largest departure from the planned pose at the end of the run [deg]":
            '0.01, 27.56, 9.63; 0.919, 0.802; none; 0.48; 0.001',
        "the simulated lift, 297 kg, within the forecast horizon, at the end pose's braking limit, the servos offset against gravity, the base braking at 2.708 m/s^2 with the actuators' rise from the drive's largest acceleration, from an onset of the sweep on: the least distance of the centre of pressure inside the contacts [m], its onset [s], how long after it [s]; the rigid model's least along the planned lift [m]; the shortfall of the first below the second [mm]; the largest shortfall below the rigid model for the planned pose at the same instant [mm], its onset [s], when [s], the rigid model's distance then [m]; the rigid model's least distance at an instant at which the shortfall exceeds the allowance [m]":
            '0.1013, 3.66, 0.60; 0.1050; 3.68; 6.90, 1.60, 1.79, 0.4228; 0.1333',
        "the simulated lift, 297 kg, within the forecast horizon, at the end pose's braking limit, the servos offset against gravity, the base braking at 2.708 m/s^2 with the actuators' rise from the drive's largest acceleration, from an onset of the sweep on: the servos' largest torques, slew, lift, tilt [kN m]; the shares of the lift's and the tilt's limits; the front-most point's largest excess over the rigid model's farthest [mm]; the base's largest travel on its contacts [mm]; the arm's largest departure from the planned pose at the end of the run [deg]":
            '0.01, 27.57, 9.63; 0.919, 0.802; none; 0.52; 0.001',
        "the simulated lift, 300 kg, beyond the forecast horizon, at the requirement's braking limit, the servos offset against gravity, the base braking at 2.572 m/s^2 with the actuators' rise from the drive's largest acceleration, from an onset of the sweep on: the least distance of the centre of pressure inside the contacts [m], its onset [s], how long after it [s]; the rigid model's least along the planned lift [m]; the shortfall of the first below the second [mm]; the largest shortfall below the rigid model for the planned pose at the same instant [mm], its onset [s], when [s], the rigid model's distance then [m]; the rigid model's least distance at an instant at which the shortfall exceeds the allowance [m]":
            '0.1104, 3.66, 0.61; 0.1141; 3.72; 6.91, 1.60, 1.79, 0.4229; 0.1410',
        "the simulated lift, 300 kg, beyond the forecast horizon, at the requirement's braking limit, the servos offset against gravity, the base braking at 2.572 m/s^2 with the actuators' rise from the drive's largest acceleration, from an onset of the sweep on: the servos' largest torques, slew, lift, tilt [kN m]; the shares of the lift's and the tilt's limits; the front-most point's largest excess over the rigid model's farthest [mm]; the base's largest travel on its contacts [mm]; the arm's largest departure from the planned pose at the end of the run [deg]":
            '0.01, 27.70, 9.69; 0.924, 0.807; none; 0.47; 0.001',
        "the simulated lift, 300 kg, beyond the forecast horizon, at the end pose's braking limit, the servos offset against gravity, the base braking at 2.675 m/s^2 with the actuators' rise from the drive's largest acceleration, from an onset of the sweep on: the least distance of the centre of pressure inside the contacts [m], its onset [s], how long after it [s]; the rigid model's least along the planned lift [m]; the shortfall of the first below the second [mm]; the largest shortfall below the rigid model for the planned pose at the same instant [mm], its onset [s], when [s], the rigid model's distance then [m]; the rigid model's least distance at an instant at which the shortfall exceeds the allowance [m]":
            '0.1012, 3.66, 0.61; 0.1050; 3.72; 6.94, 1.60, 1.79, 0.4208; 0.1321',
        "the simulated lift, 300 kg, beyond the forecast horizon, at the end pose's braking limit, the servos offset against gravity, the base braking at 2.675 m/s^2 with the actuators' rise from the drive's largest acceleration, from an onset of the sweep on: the servos' largest torques, slew, lift, tilt [kN m]; the shares of the lift's and the tilt's limits; the front-most point's largest excess over the rigid model's farthest [mm]; the base's largest travel on its contacts [mm]; the arm's largest departure from the planned pose at the end of the run [deg]":
            '0.01, 27.70, 9.69; 0.924, 0.807; none; 0.51; 0.001',
        "the simulated lift, 297 kg, within the forecast horizon, at the requirement's braking limit, the servos offset against gravity, the base braking at 2.599 m/s^2 with the actuators' rise from cruising, from an onset of the sweep on: the least distance of the centre of pressure inside the contacts [m], its onset [s], how long after it [s]; the rigid model's least along the planned lift [m]; the shortfall of the first below the second [mm]; the largest shortfall below the rigid model for the planned pose at the same instant [mm], its onset [s], when [s], the rigid model's distance then [m]; the rigid model's least distance at an instant at which the shortfall exceeds the allowance [m]":
            '0.1111, 3.66, 0.60; 0.1146; 3.59; 6.12, 1.70, 1.89, 0.3276; 0.1441',
        "the simulated lift, 297 kg, within the forecast horizon, at the requirement's braking limit, the servos offset against gravity, the base braking at 2.599 m/s^2 with the actuators' rise from cruising, from an onset of the sweep on: the servos' largest torques, slew, lift, tilt [kN m]; the shares of the lift's and the tilt's limits; the front-most point's largest excess over the rigid model's farthest [mm]; the base's largest travel on its contacts [mm]; the arm's largest departure from the planned pose at the end of the run [deg]":
            '0.01, 27.56, 9.63; 0.919, 0.802; none; 0.48; 0.001',
        "the simulated lift, 297 kg, within the forecast horizon, at the requirement's braking limit, the servos not offset, the base braking at 2.599 m/s^2 with the actuators' rise from the drive's largest acceleration, from an onset of the sweep on: the least distance of the centre of pressure inside the contacts [m], its onset [s], how long after it [s]; the rigid model's least along the planned lift [m]; the shortfall of the first below the second [mm]; the largest shortfall below the rigid model for the planned pose at the same instant [mm], its onset [s], when [s], the rigid model's distance then [m]; the rigid model's least distance at an instant at which the shortfall exceeds the allowance [m]":
            '0.1138, 3.42, 0.84; 0.1146; 0.85; 6.97, 1.70, 1.86, 0.4486; 0.3325',
        "the simulated lift, 297 kg, within the forecast horizon, at the requirement's braking limit, the servos not offset, the base braking at 2.599 m/s^2 with the actuators' rise from the drive's largest acceleration, from an onset of the sweep on: the servos' largest torques, slew, lift, tilt [kN m]; the shares of the lift's and the tilt's limits; the front-most point's largest excess over the rigid model's farthest [mm]; the base's largest travel on its contacts [mm]; the arm's largest departure from the planned pose at the end of the run [deg]":
            '0.01, 27.62, 9.65; 0.921, 0.804; none; 0.47; 0.782',
        "the servos' largest torques over every simulated lift, lift, tilt [kN m]; the shares of their limits":
            '27.70, 9.69; 0.924, 0.807',
        "with a stop line ahead: the front-most point's travel, the clearance that the lift consumes [m]; the requirement [m]; the clearance after the lift [m]":
            '0.8592; 0.96; 0.1008',
        "at the top speed, carrying, the deceleration rising and fading over the actuators' rise: the recovery's stop at the floor's braking limit and at the requirement's [m]; the difference [m]; the same stops were the braking to begin at once [m]":
            '2.025, 2.632; 0.607; 1.125, 1.732',
    },
    'a turn': {
        "carry, declared: by radius, the speed that a turn allows, what binds, and its lateral acceleration; the distance and time to slow to it from the top speed at the braking limit (4.00 m/s^2), the deceleration rising and fading over the actuators' rise":
            'R 3 m: 1.85 m/s, slew, 1.14 m/s^2; slowing 2.91 m, 1.20 s; R 5 m: 2.39 m/s, slew, 1.14 m/s^2; slowing 3.24 m, 1.20 s; R 10 m: 3 m/s, the top speed',
        'carry, declared: a stop at the braking in a turn from the speed of the turn of 5 m [m], [s]':
            '6.43, 5.38',
        "reach, declared: by radius, the speed that a turn allows, what binds, and its lateral acceleration; the distance and time to slow to it from the top speed at the braking limit (2.70 m/s^2), the deceleration rising and fading over the actuators' rise":
            'R 3 m: 1.45 m/s, ground, 0.70 m/s^2; slowing 2.67 m, 1.20 s; R 5 m: 1.96 m/s, ground, 0.76 m/s^2; slowing 2.98 m, 1.20 s; R 10 m: 2.84 m/s, ground, 0.80 m/s^2; slowing 3.51 m, 1.20 s',
        'reach, declared: a stop at the braking in a turn from the speed of the turn of 5 m [m], [s]':
            '4.43, 4.52',
        "reach left, declared: by radius, the speed that a turn allows, what binds, and its lateral acceleration; the distance and time to slow to it from the top speed at the braking limit (2.38 m/s^2), the deceleration rising and fading over the actuators' rise":
            'R 3 m: 1.63 m/s, slew, 0.88 m/s^2; slowing 2.78 m, 1.20 s; R 5 m: 2.17 m/s, slew, 0.94 m/s^2; slowing 3.11 m, 1.20 s; R 10 m: 3 m/s, the top speed',
        'reach left, declared: a stop at the braking in a turn from the speed of the turn of 5 m [m], [s]':
            '5.36, 4.94',
        'reach left, declared: the speed that a turn of 3 and of 5 m allows to the left; to the right [m/s]':
            '1.86, 2.37; 1.63, 2.17',
        "reach left, declared: the largest level that a demand for a turn can ask, within the rest's tipping reserve under the global capability [m]; the speed that a turn of 3 and of 5 m allows at that level [m/s], what binds":
            '0.21; 0.41 tipping, 0.57 tipping',
    },
    'the allowance': {
        "the largest shortfall of a simulated least distance below the rigid model's least, over every run with the actuators' rise: the braking limits of the four poses under the three labels, the turns, the plans in force and the lifts [mm]; the allowance [mm]":
            '3.72; 5',
        "the largest shortfall below the rigid model for the planned pose at a single instant: with the arm held, over the same runs; during the lifts [mm]; where a lift's exceeds the allowance, the rigid model's least distance [m]":
            '3.19; 6.97; 0.1321',
        "the servos' largest torques over the same runs, as shares of the slew's, the lift's and the tilt's limits; the base's largest creep, on the labels' least friction [mm]; the front-most point's largest excess in a stop there [mm]":
            '0.945, 0.924, 0.807; 0.31; 4.03',
    },
}


if __name__ == "__main__":
    main()
