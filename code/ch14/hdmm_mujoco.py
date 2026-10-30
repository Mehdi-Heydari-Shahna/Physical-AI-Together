#!/usr/bin/env python3
"""
A heavy-duty mobile manipulator in MuJoCo: how its arm sets what its base can do.

Physical AI, Together, Chapter 14 (Section 14.1).

The machine of the package hdmm is synthetic, every number ours (see its
docstring). This script
  * writes the record of the experiment;
  * computes, for three poses of the arm and payloads from 0 to 300 kg, the
    machine's centre of mass and the steady accelerations of the base at
    which a pair of wheels starts to lift, for the rigid machine (braking,
    accelerating, and the lateral acceleration of a turn to either side);
  * checks ten of these against the simulated machine, which stands on its
    four wheel contacts in MuJoCo under gravity tilted by the acceleration:
    its settled wheel loads, extrapolated to zero, against the rigid
    machine's load balance, and the acceleration, raised slowly or applied
    at once, from which it tips over;
  * finds which limits the base's braking on a ground of friction 0.6, the
    friction or the tipping, and the worst pose within the arm's ranges;
  * computes the torques that hold the arm exactly in the carrying pose at
    rest, while the base brakes at 3 m/s^2 and while it turns at 2 m/s^2;
  * writes the braking limit against the payload for Figure 14.3.

Usage
    python3 hdmm_mujoco.py            # writes results/hdmm-*
    python3 hdmm_mujoco.py --check    # also checks every value quoted in
                                      # Section 14.1

Outputs, in results/ next to this file
    hdmm-record.txt         the record of the experiment
    hdmm-summary.txt        every value, by group
    hdmm-braking.csv        the braking limit against the payload, by pose

Requires MuJoCo 3.16.0 and NumPy (requirements.txt); run with CPython
3.13.17, other versions of Python not tried.
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

from hdmm import model                            # noqa: E402

G = 9.81
MU = 0.6                      # the ground's friction (ours)
PAYLOADS = (0.0, 150.0, 300.0)
SIDE = (math.pi / 2, 0.0, 0.0)   # the arm at full reach, slewed to the left


def record_facts():
    m = model.build(payload=model.PAYLOAD_MAX).compile()
    return {
        "MuJoCo": mujoco.__version__,
        "NumPy, Python": f"{np.__version__}, {platform.python_version()}",
        "time step [s]; integrator; friction cone": (
            f"{m.opt.timestep:g}; {mujoco.mjtIntegrator(m.opt.integrator).name[6:].lower()}; "
            f"{mujoco.mjtCone(m.opt.cone).name[7:].lower()}"),
        "base: mass [kg]; centre of mass [m]; wheel contacts at x, y [m]": (
            f"{model.BASE['mass']:g}; {', '.join(f'{x:g}' for x in model.BASE['com'])}; "
            f"+-{model.AXLE:g}, +-{model.HALF_TRACK:g}"),
        "arm: slew axis x, shoulder height [m]; boom, stick length [m]; mass [kg]": (
            f"{model.SLEW[0]:g}, {model.SLEW[1]:g}; {model.BOOM['length']:g}, "
            f"{model.STICK['length']:g}; {model.BOOM['mass']:g}, {model.STICK['mass']:g}"),
        "tool [kg]; largest payload [kg]": f"{model.TOOL:g}; {model.PAYLOAD_MAX:g}",
        "poses (slew, lift, tilt) [deg]": "; ".join(
            f"{k} {', '.join(f'{math.degrees(x):g}' for x in v)}" for k, v in model.POSES.items()),
        "machine's mass with the largest payload [kg]": f"{float(m.body_mass.sum()):g}",
        "contacts' softness, MuJoCo's defaults: solref; solimp": (
            f"{', '.join(f'{x:g}' for x in m.geom_solref[m.geom('ground').id])}; "
            f"{', '.join(f'{x:g}' for x in m.geom_solimp[m.geom('ground').id])}"),
        "arm's ranges (ours): lift, tilt [deg]; the tool's centre above the ground [m]": (
            f"{math.degrees(model.LIFT_RANGE[0]):g} to {math.degrees(model.LIFT_RANGE[1]):g}, "
            f"{math.degrees(model.TILT_RANGE[0]):g} to {math.degrees(model.TILT_RANGE[1]):g}; "
            f"at least {model.TOOL_ABOVE:g}"),
        "ground's friction (ours); in the simulated checks": f"{MU:g}; 1.5",
        "contacts' ratio of frictional to normal impedance (ours; MuJoCo's default is 1)":
            f"{m.opt.impratio:g}",
    }


def machine_facts(rows):
    out = {}
    for name, pose in model.POSES.items():
        tip = model.tip_point(*pose)
        out[f"{name}: the tool at x, z [m]"] = f"{tip[0]:.3f}, {tip[2]:.3f}"
        for payload in PAYLOADS:
            q = model.quasi_static(*pose, payload)
            c = q["com"]
            out[f"{name}, {payload:g} kg: centre of mass x, z [m]; braking, accelerating, "
                f"turning [m/s^2] at which wheels lift"] = (
                f"{c[0]:.3f}, {c[2]:.3f}; {q['brake']:.2f}, {q['accelerate']:.2f}, "
                f"{q['left']:.2f}")
    q = model.quasi_static(*SIDE, model.PAYLOAD_MAX)
    q0 = model.quasi_static(*SIDE, 0.0)
    out["reach slewed to the left: centre of mass y [m], empty and with 300 kg"] = \
        f"{q0['com'][1]:.3f}, {q['com'][1]:.3f}"
    out["reach slewed to the left: the lateral acceleration [m/s^2] that tips it to the left, "
        "empty and with 300 kg; to the right with 300 kg"] = \
        f"{q0['left']:.2f}, {q['left']:.2f}; {q['right']:.2f}"
    for payload in np.arange(0.0, model.PAYLOAD_MAX + 1e-9, 25.0):
        row = {"payload": float(payload)}
        for name, pose in model.POSES.items():
            row[name] = model.quasi_static(*pose, float(payload))["brake"]
        row["worst"] = model.worst_braking(float(payload))[0]
        row["friction"] = MU * G
        rows.append(row)
    return out


STIFF = (0.005, 1.0)             # contacts four times stiffer than MuJoCo's default (0.02, 1)


def simulated_facts():
    out = {}
    worst = 0.0
    slow, sudden, stiff = [], [], []
    cases = [(name, pose, payload, "brake") for name, pose in model.POSES.items()
             for payload in PAYLOADS] + [("reach slewed left", SIDE, model.PAYLOAD_MAX, "left")]
    for name, pose, payload, which in cases:
        sim = model.simulated_threshold(payload, pose, which)
        qs = model.quasi_static(*pose, payload)[which]
        worst = max(worst, abs(sim / qs - 1.0))
        f_slow, _ = model.onset(payload, pose, which)
        f_sudden, _ = model.onset(payload, pose, which, at_once=True)
        f_stiff, _ = model.onset(payload, pose, which, at_once=True, solref=STIFF)
        slow.append(f_slow)
        sudden.append(f_sudden)
        stiff.append(f_stiff)
        if (name, payload) in (("reach", 300.0), ("reach slewed left", 300.0)):
            out[f"{name}, {payload:g} kg, {which}: the settled loads extrapolated to zero, the "
                "rigid machine [m/s^2]"] = f"{sim:.4f}, {qs:.4f}"
            out[f"{name}, {payload:g} kg, {which}: tips over, the acceleration raised slowly "
                "and at once [m/s^2]"] = f"{f_slow * qs:.4f}, {f_sudden * qs:.4f}"
    n = len(cases)
    out[f"the {n} cases (the nine braking points and the loaded, slewed turn): largest "
        "difference of the extrapolated loads from the rigid machine's"] = f"{100 * worst:.3f} %"
    out[f"the {n} cases: tips over from, as a fraction of the rigid threshold, the acceleration "
        "raised slowly: least, largest"] = f"{min(slow):.4f}, {max(slow):.4f}"
    out[f"the {n} cases: the same, the acceleration applied at once to the machine at rest: "
        "least, largest"] = f"{min(sudden):.4f}, {max(sudden):.4f}"
    out[f"the {n} cases: the same, applied at once, contacts four times stiffer (solref "
        f"{STIFF[0]:g} s): least, largest"] = f"{min(stiff):.4f}, {max(stiff):.4f}"
    out["reach slewed left, 300 kg: the lateral acceleration at the centre of mass that tips "
        "it in a steady turn of radius 5 m and 10 m, and taken uniform [m/s^2]"] = ", ".join(
            f"{x:.2f}" for x in (model.turn_threshold(*SIDE, model.PAYLOAD_MAX, 5.0),
                                 model.turn_threshold(*SIDE, model.PAYLOAD_MAX, 10.0),
                                 model.quasi_static(*SIDE, model.PAYLOAD_MAX)["left"]))
    return out


def label_facts():
    out = {}
    for name, pose in model.POSES.items():
        parts = []
        for payload in PAYLOADS:
            a = model.quasi_static(*pose, payload)["brake"]
            parts.append(f"{min(a, MU * G):.2f} ({'friction' if MU * G <= a else 'tipping'})")
        out[f"{name}: braking limit on friction {MU:g} [m/s^2], 0, 150, 300 kg"] = ", ".join(parts)
    for name, pose in model.POSES.items():
        f = lambda p: model.quasi_static(*pose, p)["brake"] - MU * G      # noqa: E731
        if f(model.PAYLOAD_MAX) >= 0:
            out[f"{name}: the payload above which tipping limits the braking [kg]"] = "none up to 300"
            continue
        lo_p, hi_p = 0.0, model.PAYLOAD_MAX
        if f(lo_p) < 0:
            out[f"{name}: the payload above which tipping limits the braking [kg]"] = "0"
            continue
        for _ in range(60):
            mid = 0.5 * (lo_p + hi_p)
            lo_p, hi_p = (mid, hi_p) if f(mid) >= 0 else (lo_p, mid)
        out[f"{name}: the payload above which tipping limits the braking [kg]"] = f"{lo_p:.1f}"
    lo = min(model.quasi_static(*pose, p)["brake"] for pose in model.POSES.values()
             for p in PAYLOADS)
    out["the smallest of these, over the three poses and the three payloads [m/s^2]; its share "
        "of mu g"] = f"{min(lo, MU * G):.2f}; {100 * min(lo, MU * G) / (MU * G):.1f} %"
    for payload in PAYLOADS:
        a, lift, tilt = model.worst_braking(payload)
        tip = model.tip_point(0.0, lift, tilt)
        out[f"{payload:g} kg, the worst pose within the arm's ranges: braking at which the rear "
            "wheels lift [m/s^2]; lift, tilt [deg]; the tool at x, z [m]"] = (
            f"{a:.2f}; {round(math.degrees(lift), 1) + 0.0:.1f}, "
            f"{round(math.degrees(tilt), 1) + 0.0:.1f}; "
            f"{tip[0]:.2f}, {tip[2]:.2f}")
    a, _, _ = model.worst_braking(model.PAYLOAD_MAX)
    out["the braking that holds for every pose within the ranges and every payload [m/s^2]; its "
        "share of mu g"] = f"{min(a, MU * G):.2f}; {100 * min(a, MU * G) / (MU * G):.1f} %"
    return out


def loads_facts():
    out = {}
    pose = model.POSES["carry"]
    torques = {}
    for label, accel in (("at rest", (0.0, 0.0)), ("braking at 3 m/s^2", (-3.0, 0.0)),
                         ("turning at 2 m/s^2, tipping it to the left", (0.0, -2.0))):
        f = model.holding_torques(model.PAYLOAD_MAX, pose, accel=accel)
        torques[label] = f
        # round first, then add 0.0, so that a torque of -0.001 N m prints 0.00, not -0.00
        out[f"carry, 300 kg, {label}: slew, lift, tilt torque [kN m]"] = \
            ", ".join(f"{round(x / 1000, 2) + 0.0:.2f}" for x in f)
    rest, brake = torques["at rest"], torques["braking at 3 m/s^2"]
    out["carry, 300 kg, braking at 3 m/s^2: the tilt torque less than at rest"] = \
        f"{100 * (1 - brake[2] / rest[2]):.1f} %"
    return out


def hardware():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "not reported"


def write_record(path, facts):
    rec = facts["record"]
    lines = [
        "Record of the experiment of Section 14.1: a synthetic heavy-duty mobile manipulator, how "
        "its arm sets what its base can do",
        "",
        "model and asset provenance",
        "  synthetic, every number ours (hdmm/model.py); no published model of such a machine was "
        "found; the payload range after Bahari et al. (2025); the arm's ranges ours: "
        + rec["arm's ranges (ours): lift, tilt [deg]; the tool's centre above the ground [m]"],
        "  base " + rec["base: mass [kg]; centre of mass [m]; wheel contacts at x, y [m]"] +
        "; arm " + rec["arm: slew axis x, shoulder height [m]; boom, stick length [m]; mass [kg]"] +
        "; tool, payload " + rec["tool [kg]; largest payload [kg]"],
        "simulator and solver",
        f"  MuJoCo {rec['MuJoCo']}; NumPy and Python {rec['NumPy, Python']}; "
        + rec["time step [s]; integrator; friction cone"] + "; contacts' softness (MuJoCo's "
        "defaults) solref, solimp " + rec["contacts' softness, MuJoCo's defaults: solref; solimp"]
        + "; also solref 0.005, 1 for one comparison; the contacts' ratio of frictional to normal "
        "impedance " + rec["contacts' ratio of frictional to normal impedance (ours; MuJoCo's "
                           "default is 1)"] + ", which keeps the machine's creep on them small "
        "(Section 14.2, labels_demand.py)",
        "time steps",
        "  1 ms; the extrapolated loads after 2 s at rest; the onset runs settle 2 s at rest, then "
        "raise the acceleration over 5 s (or at once) and hold it 24 s; tipped if the base then "
        "leans more than 0.1 deg",
        "actuators and sensors",
        "  the arm rigid in its pose; the torques that hold it exactly in its pose computed on a "
        "fixed base under the tilted gravity; no sensors: the state from the simulator",
        "frames and units",
        "  SI units; the base's x forward, y left, z up, the ground at z = 0",
        "uncertainty bounds",
        "  none: the machine's numbers are exact by construction; the ground's friction 0.6 "
        "(ours); 1.5 in the simulated checks, so that nothing slides",
        "constraints",
        "  all four wheels on the ground (the margin against tipping)",
        "terminal behaviour",
        "  none: Section 14.1 measures limits",
        "training and evaluation split",
        "  none: nothing learns in Section 14.1",
        "seeds",
        "  none: no random numbers",
        "hardware",
        f"  {platform.system()} {platform.machine()}, {hardware()}",
        "guarantee status",
        "  none: a synthetic machine; the limits are those of its model, not of any machine",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Section 14.1: a heavy-duty mobile manipulator")
    ap.add_argument("--check", action="store_true", help="check every value quoted in Section 14.1")
    args = ap.parse_args()
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    facts = {"record": record_facts(),
             "the machine": machine_facts(rows),
             "the simulated machine": simulated_facts(),
             "the base's braking": label_facts(),
             "the arm's loads": loads_facts()}
    with open(os.path.join(out_dir, "hdmm-braking.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow({k: f"{v:.6g}" for k, v in row.items()})
    write_record(os.path.join(out_dir, "hdmm-record.txt"), facts)
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
        lines.append(f"[check] {n} values, among them every value that Section 14.1 quotes: " +
                     ("all as expected" if not failures else "FAILED: " + "; ".join(failures)))
    with open(os.path.join(out_dir, "hdmm-summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if failures:
        raise SystemExit(1)


EXPECTED = {   # every value of the summary but the versions of the libraries
    'record': {
        'time step [s]; integrator; friction cone':
            '0.001; implicitfast; elliptic',
        'base: mass [kg]; centre of mass [m]; wheel contacts at x, y [m]':
            '3500; 0, 0, 0.75; +-1.1, +-0.85',
        'arm: slew axis x, shoulder height [m]; boom, stick length [m]; mass [kg]':
            '0.6, 1.3; 2.4, 2; 350, 200',
        'tool [kg]; largest payload [kg]':
            '100; 300',
        'poses (slew, lift, tilt) [deg]':
            'stowed 0, 60, -150; carry 0, 20, -70; reach 0, 0, 0',
        "machine's mass with the largest payload [kg]":
            '4450',
        "contacts' softness, MuJoCo's defaults: solref; solimp":
            '0.02, 1; 0.9, 0.95, 0.001, 0.5, 2',
        "arm's ranges (ours): lift, tilt [deg]; the tool's centre above the ground [m]":
            '0 to 60, -150 to 0; at least 0.2',
        "ground's friction (ours); in the simulated checks":
            '0.6; 1.5',
        "contacts' ratio of frictional to normal impedance (ours; MuJoCo's default is 1)":
            '20',
    },
    'the machine': {
        'stowed: the tool at x, z [m]':
            '1.800, 1.378',
        'stowed, 0 kg: centre of mass x, z [m]; braking, accelerating, turning [m/s^2] at which wheels lift':
            '0.227, 0.975; 8.78, 13.35, 8.55',
        'stowed, 150 kg: centre of mass x, z [m]; braking, accelerating, turning [m/s^2] at which wheels lift':
            '0.282, 0.989; 8.11, 13.70, 8.43',
        'stowed, 300 kg: centre of mass x, z [m]; braking, accelerating, turning [m/s^2] at which wheels lift':
            '0.333, 1.002; 7.51, 14.03, 8.32',
        'carry: the tool at x, z [m]':
            '4.141, 0.589',
        'carry, 0 kg: centre of mass x, z [m]; braking, accelerating, turning [m/s^2] at which wheels lift':
            '0.403, 0.857; 7.98, 17.20, 9.73',
        'carry, 150 kg: centre of mass x, z [m]; braking, accelerating, turning [m/s^2] at which wheels lift':
            '0.533, 0.848; 6.56, 18.90, 9.84',
        'carry, 300 kg: centre of mass x, z [m]; braking, accelerating, turning [m/s^2] at which wheels lift':
            '0.655, 0.839; 5.20, 20.52, 9.94',
        'reach: the tool at x, z [m]':
            '5.000, 1.300',
        'reach, 0 kg: centre of mass x, z [m]; braking, accelerating, turning [m/s^2] at which wheels lift':
            '0.452, 0.836; 7.60, 18.21, 9.97',
        'reach, 150 kg: centre of mass x, z [m]; braking, accelerating, turning [m/s^2] at which wheels lift':
            '0.610, 0.852; 5.63, 19.69, 9.78',
        'reach, 300 kg: centre of mass x, z [m]; braking, accelerating, turning [m/s^2] at which wheels lift':
            '0.758, 0.867; 3.86, 21.02, 9.61',
        'reach slewed to the left: centre of mass y [m], empty and with 300 kg':
            '0.358, 0.630',
        'reach slewed to the left: the lateral acceleration [m/s^2] that tips it to the left, empty and with 300 kg; to the right with 300 kg':
            '5.77, 2.48; 16.74',
    },
    'the simulated machine': {
        'reach, 300 kg, brake: the settled loads extrapolated to zero, the rigid machine [m/s^2]':
            '3.8625, 3.8630',
        'reach, 300 kg, brake: tips over, the acceleration raised slowly and at once [m/s^2]':
            '3.8622, 3.8583',
        'reach slewed left, 300 kg, left: the settled loads extrapolated to zero, the rigid machine [m/s^2]':
            '2.4833, 2.4843',
        'reach slewed left, 300 kg, left: tips over, the acceleration raised slowly and at once [m/s^2]':
            '2.4830, 2.4781',
        "the 10 cases (the nine braking points and the loaded, slewed turn): largest difference of the extrapolated loads from the rigid machine's":
            '0.037 %',
        'the 10 cases: tips over from, as a fraction of the rigid threshold, the acceleration raised slowly: least, largest':
            '0.9995, 0.9999',
        'the 10 cases: the same, the acceleration applied at once to the machine at rest: least, largest':
            '0.9975, 0.9991',
        'the 10 cases: the same, applied at once, contacts four times stiffer (solref 0.005 s): least, largest':
            '0.9994, 0.9997',
        'reach slewed left, 300 kg: the lateral acceleration at the centre of mass that tips it in a steady turn of radius 5 m and 10 m, and taken uniform [m/s^2]':
            '2.34, 2.41, 2.48',
    },
    "the base's braking": {
        'stowed: braking limit on friction 0.6 [m/s^2], 0, 150, 300 kg':
            '5.89 (friction), 5.89 (friction), 5.89 (friction)',
        'carry: braking limit on friction 0.6 [m/s^2], 0, 150, 300 kg':
            '5.89 (friction), 5.89 (friction), 5.20 (tipping)',
        'reach: braking limit on friction 0.6 [m/s^2], 0, 150, 300 kg':
            '5.89 (friction), 5.63 (tipping), 3.86 (tipping)',
        'stowed: the payload above which tipping limits the braking [kg]':
            'none up to 300',
        'carry: the payload above which tipping limits the braking [kg]':
            '223.4',
        'reach: the payload above which tipping limits the braking [kg]':
            '129.9',
        'the smallest of these, over the three poses and the three payloads [m/s^2]; its share of mu g':
            '3.86; 65.6 %',
        "0 kg, the worst pose within the arm's ranges: braking at which the rear wheels lift [m/s^2]; lift, tilt [deg]; the tool at x, z [m]":
            '6.72; 34.4, 0.0; 4.23, 3.79',
        "150 kg, the worst pose within the arm's ranges: braking at which the rear wheels lift [m/s^2]; lift, tilt [deg]; the tool at x, z [m]":
            '4.95; 26.8, 0.0; 4.53, 3.28',
        "300 kg, the worst pose within the arm's ranges: braking at which the rear wheels lift [m/s^2]; lift, tilt [deg]; the tool at x, z [m]":
            '3.44; 19.3, 0.0; 4.75, 2.75',
        'the braking that holds for every pose within the ranges and every payload [m/s^2]; its share of mu g':
            '3.44; 58.4 %',
    },
    "the arm's loads": {
        'carry, 300 kg, at rest: slew, lift, tilt torque [kN m]':
            '0.00, 23.00, 6.18',
        'carry, 300 kg, braking at 3 m/s^2: slew, lift, tilt torque [kN m]':
            '0.00, 22.62, 3.93',
        'carry, 300 kg, turning at 2 m/s^2, tipping it to the left: slew, lift, tilt torque [kN m]':
            '-4.69, 23.00, 6.18',
        'carry, 300 kg, braking at 3 m/s^2: the tilt torque less than at rest':
            '36.4 %',
    },
}


if __name__ == "__main__":
    main()
