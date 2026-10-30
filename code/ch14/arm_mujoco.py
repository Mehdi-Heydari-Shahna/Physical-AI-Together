#!/usr/bin/env python3
"""
The UR5e in MuJoCo: its description, its swing towards a post, and its stop.

Physical AI, Together, Chapter 14 (Section 14.1).

This script builds the UR5e of the package arm from Universal Robots'
description and
  * writes the record of the experiment;
  * checks the model against what UR publishes: the joints' frames against
    UR's table of Denavit-Hartenberg parameters, the links' masses and
    centres of mass against UR's table, the total against the datasheet; and
    records what MuJoCo Menagerie's model of the arm holds instead;
  * states the task's geometry: the carry pose, where the loaded arm first
    touches the post, and how close the empty arm, which passes over it,
    comes;
  * finds the largest deceleration of the base joint whose commanded stop
    the joints' torque limits allow, by payload and by where the payload's
    mass sits, and which joint binds;
  * simulates the stop from the place at full speed at that deceleration and
    beyond it, and with the deceleration of the empty arm and a payload of
    5 kg, with what the base and the wrists do in each;
  * repeats one stop at time steps of 1 and 0.25 ms, and the capability with
    the rotor inertia that MuJoCo Menagerie's model adds.

Usage
    python3 arm_mujoco.py            # writes results/arm-*
    python3 arm_mujoco.py --check    # also checks every value quoted in
                                     # Section 14.1

Outputs, in results/ next to this file
    arm-record.txt          the record of the experiment
    arm-summary.txt         every value, by group; with --check, the
                            comparison with the values that the book quotes
    arm-stops.csv           the stops of Figure 14.1, every 2 ms: commanded and
                            actual angle and the base's torque over its limit

Requires MuJoCo 3.16.0, NumPy and PyYAML (requirements.txt); run with
CPython 3.13.17, other versions of Python not tried.
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

from arm import control, model, world             # noqa: E402

FULL = -math.pi                 # [rad/s] the base joint at its speed limit, towards the post
PAYLOADS = (0.0, 1.0, 2.5, 5.0)
OFFSETS = (0.05, 0.10, 0.15)    # [m] the payload's centre of mass from the flange
STEPS = (0.001, 0.0005, 0.00025)
# UR's table (universal-robots.com, "DH Parameters for calculations of
# kinematics and dynamics", last modified 4 October 2018), UR5e/UR7e
DH = dict(d=(0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996), a=(0.0, -0.425, -0.3922, 0.0, 0.0, 0.0),
          alpha=(math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0))
UR_MASSES = (3.761, 8.058, 2.846, 1.37, 1.3, 0.365)
UR_COMS = ((0.0, -0.02561, 0.00193), (0.2125, 0.0, 0.11336), (0.15, 0.0, 0.0265),
           (0.0, -0.0018, 0.01634), (0.0, 0.0018, 0.01634), (0.0, 0.0, -0.001159))
DATASHEET_MASS = 20.6           # [kg] UR5e datasheet (08/2023, updated December 2024), with cable
MENAGERIE = dict(commit="f3b6013629c7b94f8ee93fd977b161fa8350f9bc",
                 masses=(4.0, 3.7, 8.393, 2.275, 1.219, 1.219, 0.1889), armature=0.1)


def deg(x, digits=2):
    # round first, then add 0.0, so that an angle that rounds to zero prints 0.000, not -0.000
    return f"{round(math.degrees(x), digits) + 0.0:.{digits}f}"


def dh_transform(q):
    t = np.eye(4)
    out = []
    for i in range(6):
        ct, st = math.cos(q[i]), math.sin(q[i])
        ca, sa = math.cos(DH["alpha"][i]), math.sin(DH["alpha"][i])
        a = np.array([[ct, -st * ca, st * sa, DH["a"][i] * ct],
                      [st, ct * ca, -ct * sa, DH["a"][i] * st],
                      [0.0, sa, ca, DH["d"][i]], [0.0, 0.0, 0.0, 1.0]])
        t = t @ a
        out.append(t.copy())
    return out


# ----------------------------------------------------------------------------
# The record
# ----------------------------------------------------------------------------
def record_facts():
    spec = model.build(world=world.add_world)
    m = spec.compile()
    lim = model.limits()
    package = open(os.path.join(model.UR, "package.xml"), encoding="utf-8").read()
    version = package.split("<version>")[1].split("</version>")[0]
    return {
        "MuJoCo": mujoco.__version__,
        "NumPy, PyYAML, Python": f"{np.__version__}, {yaml.__version__}, {platform.python_version()}",
        "description": model.SOURCE,
        "ur_description's version in package.xml": version,
        "time step [s]; integrator": f"{m.opt.timestep:g}; "
                                     f"{mujoco.mjtIntegrator(m.opt.integrator).name[6:].lower()}",
        "bodies, joints, actuators, geoms": f"{m.nbody - 1}, {m.njnt}, {m.nu}, {m.ngeom}",
        "torque limits [N m]": ", ".join(f"{lim[j]['torque']:g}" for j in model.JOINTS),
        "speed limit [deg/s]": ", ".join(f"{math.degrees(lim[j]['speed']):g}" for j in model.JOINTS),
        "joint ranges [deg]": ", ".join(
            f"+-{math.degrees(lim[j]['upper']):g}" for j in model.JOINTS),
        "controller: rate [Hz], KP [1/s^2], KD [1/s]": f"{control.RATE:g}, {control.KP:g}, {control.KD:g}",
        "payload: box side, centre of mass from the flange [m]":
            f"{model.PAYLOAD_SIZE:g}, {model.PAYLOAD_OFFSET:g}",
    }


# ----------------------------------------------------------------------------
# The description against UR's own numbers
# ----------------------------------------------------------------------------
def description_facts():
    spec = model.build()
    m = spec.compile()
    d = mujoco.MjData(m)
    rng = np.random.default_rng(14)
    base, tool = m.site("base").id, m.site("tool0").id
    worst_p = worst_r = worst_c = 0.0
    for _ in range(200):
        q = rng.uniform(-math.pi, math.pi, 6)
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        rb, pb = d.site_xmat[base].reshape(3, 3), d.site_xpos[base]
        frames = dh_transform(q)
        p = rb.T @ (d.site_xpos[tool] - pb)
        r = rb.T @ d.site_xmat[tool].reshape(3, 3)
        worst_p = max(worst_p, float(np.max(np.abs(p - frames[-1][:3, 3]))))
        worst_r = max(worst_r, float(np.max(np.abs(r - frames[-1][:3, :3]))))
        for i, link in enumerate(model.LINKS):
            c_ur = frames[i][:3, :3] @ np.array(UR_COMS[i]) + frames[i][:3, 3]
            c_m = rb.T @ (d.xipos[m.body(link + "_link").id] - pb)
            worst_c = max(worst_c, float(np.linalg.norm(c_m - c_ur)))
    masses = [float(m.body_mass[m.body(link + "_link").id]) for link in model.LINKS]
    base_mass = float(m.body_mass[m.body("base_link_inertia").id])
    total = base_mass + sum(masses)
    desc = model.description()
    lim = desc["limits"]
    return {
        "tool0 against UR's DH table, 200 random poses: largest error of position [m], of rotation":
            f"{worst_p:.1e}, {worst_r:.1e}",
        "links' centres of mass against UR's table: largest distance [m]": f"{worst_c:.1e}",
        "links' masses [kg], shoulder to wrist 3": ", ".join(f"{x:g}" for x in masses),
        "links' masses equal to UR's table": str(all(abs(a - b) < 1e-12 for a, b in zip(masses, UR_MASSES))),
        "base's mass [kg] (the description: 'This mass might be incorrect')": f"{base_mass:g}",
        "total [kg]; the datasheet, with cable [kg]": f"{total:.2f}; {DATASHEET_MASS:g}",
        "MuJoCo Menagerie's UR5e (commit f3b6013): masses [kg], base to wrist 3; total":
            ", ".join(f"{x:g}" for x in MENAGERIE["masses"]) + f"; {sum(MENAGERIE['masses']):.2f}",
        "the description's elbow range [deg]; the datasheet's":
            f"+-{math.degrees(lim['elbow_joint']['max_position']):g}; +-360",
        "inertia tensors: in the description; in UR's table":
            "measured, nonzero (pull request 256, April 2025); 'negligible', zero for links 1 to 5",
    }


# ----------------------------------------------------------------------------
# The task's geometry
# ----------------------------------------------------------------------------
def task_facts():
    out = {}
    for payload in (0.0, 5.0):
        arm = control.Arm(payload=payload)
        d = mujoco.MjData(arm.m)
        contact = world.contact_angle(arm.m, d, arm.pose)
        d.qpos[:] = world.turned(arm.pose, world.PLACE)
        mujoco.mj_kinematics(arm.m, d)
        gap = world.clearance(arm.m, d)
        if payload == 0.0:
            out["carry pose [deg], base to wrist 3"] = ", ".join(deg(x) for x in arm.pose)
            out["carry pose: residual of the inverse kinematics"] = f"{arm.pose_err:.0e}"
            out["the base joint's angle less the task's angle in the carry pose [deg]"] = deg(arm.pose[0])
        tag = "empty" if payload == 0.0 else "5 kg payload"
        out[f"{tag}: first contact with the post at the angle [deg]; clearance at the place [m]"] = \
            f"{'none' if contact is None else deg(contact)}; {gap:.4f}"
        if contact is not None:
            out[f"{tag}: angle from the place to the contact [deg]"] = deg(world.PLACE - contact)
        else:
            least, _ = world.least_clearance(arm.m, d, arm.pose)
            out[f"{tag}: least clearance from the place to 10 deg past the post's centre "
                "[m]"] = f"{least:.4f}"
    return out


# ----------------------------------------------------------------------------
# The capability: the largest valid label of the base joint
# ----------------------------------------------------------------------------
def capability_facts():
    out = {}
    for payload in PAYLOADS:
        arm = control.Arm(payload=payload)
        b, which, cruise = control.capability(arm, FULL)
        mj0, _, _ = control.coefficients(arm)
        reach = FULL * FULL / (2 * b)
        out[f"{payload:g} kg: base's inertia M00 [kg m^2]; largest deceleration from 180 deg/s "
            f"[rad/s^2]; the bound"] = f"{mj0[0]:.4f}; {b:.2f}; {which[0]} at the {which[1]}"
        out[f"{payload:g} kg: the stop's angle [deg]; tool0's arc [m]"] = \
            f"{math.degrees(reach):.3f}; {world.CARRY_RADIUS * reach:.4f}"
        if payload == 5.0:
            per_joint = control.bounds(arm, FULL)
            out["5 kg: each joint's own bound [rad/s^2], base to wrist 3"] = ", ".join(
                f"{b_j:.2f} ({end})" for b_j, end in per_joint)
    for offset in OFFSETS[1:]:
        arm = control.Arm(payload=5.0, payload_offset=offset)
        b, which, _ = control.capability(arm, FULL)
        out[f"5 kg, centre of mass {offset:g} m from the flange: largest deceleration [rad/s^2]; "
            f"the bound"] = f"{b:.2f}; {which[0]} at the {which[1]}"
    for payload in (0.0, 5.0):
        arm = control.Arm(payload=payload, armature=MENAGERIE["armature"])
        b, which, _ = control.capability(arm, FULL)
        out[f"{payload:g} kg with Menagerie's armature 0.1 kg m^2: largest deceleration "
            f"[rad/s^2]; the bound"] = f"{b:.2f}; {which[0]} at the {which[1]}"
    return out


# ----------------------------------------------------------------------------
# The stops, simulated
# ----------------------------------------------------------------------------
def stop_lines(label, r, arm, b, trace=None):
    """The facts of one simulated stop at the deceleration b."""
    out = {}
    out[f"{label}: farthest angle [deg]; beyond the commanded rest [deg]; least clearance "
        "[m]"] = f"{deg(r['farthest'], 3)}; {deg(r['beyond'], 3)}; {r['clearance']:.4f}"
    out[f"{label}: largest torque asked over its limit, base and second wrist; updates "
        "clipped, base and second wrist"] = (f"{r['peak'][0]:.3f}, {r['peak'][4]:.3f}; "
                                            f"{r['saturated'][0]}, {r['saturated'][4]}")
    out[f"{label}: largest error of the base [deg]; largest departure of the second and "
        "third wrists from the carry pose [deg]; their largest speeds [deg/s]"] = (
        f"{deg(r['err'][0], 3)}; {deg(r['departure'][4])}, {deg(r['departure'][5])}; "
        f"{deg(r['top_speed'][4], 1)}, {deg(r['top_speed'][5], 1)}")
    base_only = control.bounds(arm, FULL)[0][0]     # the deceleration of the base's limit
    if b > base_only * (1 + 1e-9):
        shortest = FULL * FULL / (2 * base_only) - FULL * FULL / (2 * b)
        out[f"{label}: the base's torque last clipped [s]; the farthest angle reached [s]; "
            "the overrun of the commanded rest if the base braked at its limit to rest with the "
            "pose held [deg]"] = f"{r['t_clipped']:.3f}; {r['t_farthest']:.3f}; {deg(shortest, 3)}"
        held = control.held_stop(arm, b, FULL, start=world.PLACE, at=r["t_clipped"])
        simulated = next(w for t, _, _, w, *_ in trace if abs(t - r["t_clipped"]) < 1e-9)
        out[f"{label}, the other joints held in the pose (the base alone, our controller, "
            "integrated exactly): overrun of the commanded rest [deg]; farthest angle [deg]; "
            "least clearance [m]"] = (f"{deg(held['beyond'], 3)}; {deg(held['farthest'], 3)}; "
                                     f"{held['clearance']:.4f}")
        out[f"{label}: the base's speed when its torque was last clipped, simulated and with "
            "the pose held [rad/s]"] = f"{abs(simulated):.3f}, {abs(held['speed_at']):.3f}"
    return out


def stop_facts(out_dir):
    out = {}
    traces = {}
    arm5 = control.Arm(payload=5.0)
    b5, _, _ = control.capability(arm5, FULL)
    arm0 = control.Arm(payload=0.0)
    b0, _, _ = control.capability(arm0, FULL)
    fastest = []
    for factor, tag in ((1.0, "1"), (1.2, "12"), (2.0, "2")):
        trace = []
        r = control.stop(arm5, factor * b5, FULL, start=world.PLACE, record=trace)
        out.update(stop_lines(f"5 kg from the place at 180 deg/s, at {factor:g} x its largest "
                              "deceleration", r, arm5, factor * b5, trace))
        traces[tag] = trace
        fastest.append(r["top_speed"])
    trace = []
    r = control.stop(arm5, b0, FULL, start=world.PLACE, record=trace)
    out.update(stop_lines(f"5 kg at the empty arm's largest deceleration ({b0:.2f} rad/s^2)",
                          r, arm5, b0, trace))
    traces["e"] = trace
    fastest.append(r["top_speed"])
    r = control.stop(arm0, b0, FULL, start=world.PLACE)
    out.update(stop_lines("empty at its largest deceleration", r, arm0, b0))
    fastest.append(r["top_speed"])
    top = np.max(np.array(fastest), axis=0)
    out["these five stops: each joint's largest speed after the start [deg/s], base to wrist 3"] = \
        ", ".join(deg(x, 1) for x in top)
    # one row every 2 ms; the columns of each stop: commanded and actual
    # angle [deg] and the base's torque over its limit, empty after its end
    n = max(len(t) for t in traces.values())
    with open(os.path.join(out_dir, "arm-stops.csv"), "w", newline="", encoding="utf-8") as fh:
        cols = ["t"] + [f"{c}{tag}" for tag in traces for c in ("c", "a", "u")]
        w = csv.writer(fh)
        w.writerow(cols)
        for k in range(0, n, 2):
            row = [f"{(k + 1) / control.RATE:.6g}"]
            for tag, tr in traces.items():
                if k < len(tr):
                    t, qc, q, _, *tau = tr[k]
                    row += [f"{math.degrees(qc):.6g}", f"{math.degrees(q):.6g}",
                            f"{tau[0] / arm5.limit[0]:.6g}"]
                else:
                    row += ["nan", "nan", "nan"]
            w.writerow(row)
    return out


def timestep_facts():
    vals = []
    for dt in STEPS:
        arm = control.Arm(payload=5.0, timestep=dt)
        b, _, _ = control.capability(arm, FULL)
        r = control.stop(arm, 1.2 * b, FULL, start=world.PLACE)
        vals.append(r)
    v = [math.degrees(r["beyond"]) for r in vals]
    order = math.log2((v[0] - v[1]) / (v[1] - v[2]))
    limit = v[2] + (v[2] - v[1]) / (2.0 ** order - 1.0)
    return {"5 kg at 1.2 x from the place, at 1, 0.5, 0.25 ms: beyond the commanded rest [deg]":
                ", ".join(deg(r["beyond"], 4) for r in vals),
            "the same: observed order of convergence; extrapolated to a zero time step [deg]":
                f"{order:.2f}; {limit:.3f}"}


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
        "Record of the experiment of Section 14.1: the UR5e in MuJoCo, its swing towards a post "
        "and its stop",
        "",
        "model and asset provenance",
        "  " + rec["description"] + "; the configuration files and collision meshes copied "
        "(description/NOTICE.md); the mapping of UR's macro done by arm/model.py, without xacro",
        "  added by arm/model.py (ours): torque actuators at the joints' limits, the payload "
        "(box of 0.1 m, centre of mass 0.05 m from the flange), the post and the table",
        "simulator and solver",
        f"  MuJoCo {rec['MuJoCo']}; NumPy, PyYAML and Python {rec['NumPy, PyYAML, Python']}; "
        f"time step and integrator {rec['time step [s]; integrator']}",
        "time steps",
        f"  simulator 0.5 ms; controller at {control.RATE:g} Hz; one stop repeated at 1 and 0.25 ms",
        "actuators and sensors",
        f"  torque at each joint, limits {rec['torque limits [N m]']} N m (joint_limits.yaml); "
        f"no friction, damping or rotor inertia (the description's); the state from the simulator",
        f"  controller (ours): computed torque, KP {control.KP:g} 1/s^2, KD {control.KD:g} 1/s, "
        "torques clipped at the limits",
        "frames and units",
        "  SI units; base_link x forward, z up, the table at z = 0; the task's angles those of "
        "tool0 about the base's axis",
        "uncertainty bounds",
        "  none: the model is nominal, and the capabilities found carry no margin for its errors",
        "constraints",
        "  no contact with the post; the joints within the ranges of joint_limits.yaml (MuJoCo's "
        "joint limits), their speeds within 180 deg/s (not enforced; the summary gives the "
        "largest) and their torques within its limits (clipped by the controller)",
        "terminal behaviour",
        "  the commanded stop of the base joint at a constant deceleration",
        "training and evaluation split",
        "  none: nothing learns in Section 14.1",
        "seeds",
        "  14, for the random poses of the kinematic check",
        "hardware",
        f"  {platform.system()} {platform.machine()}, {hardware()}",
        "guarantee status",
        "  none: Section 14.1 states the task and measures the simulated arm; no plan is certified",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Section 14.1: the UR5e in MuJoCo")
    ap.add_argument("--check", action="store_true", help="check every value quoted in Section 14.1")
    args = ap.parse_args()
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    facts = {"record": record_facts(),
             "the description": description_facts(),
             "the task": task_facts(),
             "the capability": capability_facts(),
             "the stops": stop_facts(out_dir),
             "time steps": timestep_facts()}
    write_record(os.path.join(out_dir, "arm-record.txt"), facts)
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
    with open(os.path.join(out_dir, "arm-summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if failures:
        raise SystemExit(1)


EXPECTED = {   # every value of the summary but the versions of the libraries
    'record': {
        'description':
            'ur_description 3.6.0, https://github.com/UniversalRobots/Universal_Robots_ROS2_Description, commit 7ce98d8c3512b9d0239da93b75de68ea64f61bd6 (22 October 2026), BSD 3-Clause',
        "ur_description's version in package.xml":
            '3.6.0',
        'time step [s]; integrator':
            '0.0005; implicitfast',
        'bodies, joints, actuators, geoms':
            '8, 6, 6, 9',
        'torque limits [N m]':
            '150, 150, 150, 28, 28, 28',
        'speed limit [deg/s]':
            '180, 180, 180, 180, 180, 180',
        'joint ranges [deg]':
            '+-360, +-360, +-180, +-360, +-360, +-360',
        'controller: rate [Hz], KP [1/s^2], KD [1/s]':
            '1000, 900, 60',
        'payload: box side, centre of mass from the flange [m]':
            '0.1, 0.05',
    },
    'the description': {
        "tool0 against UR's DH table, 200 random poses: largest error of position [m], of rotation":
            '2.0e-10, 6.0e-10',
        "links' centres of mass against UR's table: largest distance [m]":
            '2.1e-10',
        "links' masses [kg], shoulder to wrist 3":
            '3.761, 8.058, 2.846, 1.37, 1.3, 0.365',
        "links' masses equal to UR's table":
            'True',
        "base's mass [kg] (the description: 'This mass might be incorrect')":
            '4',
        'total [kg]; the datasheet, with cable [kg]':
            '21.70; 20.6',
        "MuJoCo Menagerie's UR5e (commit f3b6013): masses [kg], base to wrist 3; total":
            '4, 3.7, 8.393, 2.275, 1.219, 1.219, 0.1889; 20.99',
        "the description's elbow range [deg]; the datasheet's":
            '+-180; +-360',
        "inertia tensors: in the description; in UR's table":
            "measured, nonzero (pull request 256, April 2025); 'negligible', zero for links 1 to 5",
    },
    'the task': {
        'carry pose [deg], base to wrist 3':
            '-15.46, -81.00, 117.39, -126.39, -90.00, -15.46',
        'carry pose: residual of the inverse kinematics':
            '5e-14',
        "the base joint's angle less the task's angle in the carry pose [deg]":
            '-15.46',
        'empty: first contact with the post at the angle [deg]; clearance at the place [m]':
            'none; 0.1022',
        "empty: least clearance from the place to 10 deg past the post's centre [m]":
            '0.0499',
        '5 kg payload: first contact with the post at the angle [deg]; clearance at the place [m]':
            '-67.32; 0.0578',
        '5 kg payload: angle from the place to the contact [deg]':
            '7.32',
    },
    'the capability': {
        "0 kg: base's inertia M00 [kg m^2]; largest deceleration from 180 deg/s [rad/s^2]; the bound":
            '1.0490; 142.99; shoulder_pan_joint at the start',
        "0 kg: the stop's angle [deg]; tool0's arc [m]":
            '1.977; 0.0173',
        "1 kg: base's inertia M00 [kg m^2]; largest deceleration from 180 deg/s [rad/s^2]; the bound":
            '1.3007; 115.33; shoulder_pan_joint at the start',
        "1 kg: the stop's angle [deg]; tool0's arc [m]":
            '2.452; 0.0214',
        "2.5 kg: base's inertia M00 [kg m^2]; largest deceleration from 180 deg/s [rad/s^2]; the bound":
            '1.6782; 89.38; shoulder_pan_joint at the start',
        "2.5 kg: the stop's angle [deg]; tool0's arc [m]":
            '3.163; 0.0276',
        "5 kg: base's inertia M00 [kg m^2]; largest deceleration from 180 deg/s [rad/s^2]; the bound":
            '2.3073; 65.01; shoulder_pan_joint at the start',
        "5 kg: the stop's angle [deg]; tool0's arc [m]":
            '4.349; 0.0380',
        "5 kg: each joint's own bound [rad/s^2], base to wrist 3":
            '65.01 (start), 320.07 (start), 471.11 (start), 287.76 (start), 72.18 (end), 3259.27 (start)',
        '5 kg, centre of mass 0.1 m from the flange: largest deceleration [rad/s^2]; the bound':
            '55.07; wrist_2_joint at the end',
        '5 kg, centre of mass 0.15 m from the flange: largest deceleration [rad/s^2]; the bound':
            '44.52; wrist_2_joint at the end',
        "0 kg with Menagerie's armature 0.1 kg m^2: largest deceleration [rad/s^2]; the bound":
            '130.55; shoulder_pan_joint at the start',
        "5 kg with Menagerie's armature 0.1 kg m^2: largest deceleration [rad/s^2]; the bound":
            '62.31; shoulder_pan_joint at the start',
    },
    'the stops': {
        '5 kg from the place at 180 deg/s, at 1 x its largest deceleration: farthest angle [deg]; beyond the commanded rest [deg]; least clearance [m]':
            '-64.349; 0.000; 0.0235',
        '5 kg from the place at 180 deg/s, at 1 x its largest deceleration: largest torque asked over its limit, base and second wrist; updates clipped, base and second wrist':
            '1.000, 0.898; 0, 0',
        '5 kg from the place at 180 deg/s, at 1 x its largest deceleration: largest error of the base [deg]; largest departure of the second and third wrists from the carry pose [deg]; their largest speeds [deg/s]':
            '0.056; 0.00, 0.00; 0.0, 0.0',
        '5 kg from the place at 180 deg/s, at 1.2 x its largest deceleration: farthest angle [deg]; beyond the commanded rest [deg]; least clearance [m]':
            '-65.341; 1.717; 0.0263',
        '5 kg from the place at 180 deg/s, at 1.2 x its largest deceleration: largest torque asked over its limit, base and second wrist; updates clipped, base and second wrist':
            '1.812, 1.296; 41, 41',
        '5 kg from the place at 180 deg/s, at 1.2 x its largest deceleration: largest error of the base [deg]; largest departure of the second and third wrists from the carry pose [deg]; their largest speeds [deg/s]':
            '1.717; 4.52, 1.79; 150.8, 61.4',
        "5 kg from the place at 180 deg/s, at 1.2 x its largest deceleration: the base's torque last clipped [s]; the farthest angle reached [s]; the overrun of the commanded rest if the base braked at its limit to rest with the pose held [deg]":
            '0.041; 0.060; 0.725',
        '5 kg from the place at 180 deg/s, at 1.2 x its largest deceleration, the other joints held in the pose (the base alone, our controller, integrated exactly): overrun of the commanded rest [deg]; farthest angle [deg]; least clearance [m]':
            '0.843; -64.467; 0.0225',
        "5 kg from the place at 180 deg/s, at 1.2 x its largest deceleration: the base's speed when its torque was last clipped, simulated and with the pose held [rad/s]":
            '1.011, 0.476',
        '5 kg from the place at 180 deg/s, at 2 x its largest deceleration: farthest angle [deg]; beyond the commanded rest [deg]; least clearance [m]':
            '-65.411; 3.236; 0.0255',
        '5 kg from the place at 180 deg/s, at 2 x its largest deceleration: largest torque asked over its limit, base and second wrist; updates clipped, base and second wrist':
            '3.699, 3.140; 40, 32',
        '5 kg from the place at 180 deg/s, at 2 x its largest deceleration: largest error of the base [deg]; largest departure of the second and third wrists from the carry pose [deg]; their largest speeds [deg/s]':
            '3.236; 4.64, 3.36; 145.4, 129.4',
        "5 kg from the place at 180 deg/s, at 2 x its largest deceleration: the base's torque last clipped [s]; the farthest angle reached [s]; the overrun of the commanded rest if the base braked at its limit to rest with the pose held [deg]":
            '0.040; 0.052; 2.175',
        '5 kg from the place at 180 deg/s, at 2 x its largest deceleration, the other joints held in the pose (the base alone, our controller, integrated exactly): overrun of the commanded rest [deg]; farthest angle [deg]; least clearance [m]':
            '2.197; -64.372; 0.0233',
        "5 kg from the place at 180 deg/s, at 2 x its largest deceleration: the base's speed when its torque was last clipped, simulated and with the pose held [rad/s]":
            '0.947, 0.541',
        "5 kg at the empty arm's largest deceleration (142.99 rad/s^2): farthest angle [deg]; beyond the commanded rest [deg]; least clearance [m]":
            '-65.443; 3.466; 0.0255',
        "5 kg at the empty arm's largest deceleration (142.99 rad/s^2): largest torque asked over its limit, base and second wrist; updates clipped, base and second wrist":
            '3.954, 3.392; 41, 33',
        "5 kg at the empty arm's largest deceleration (142.99 rad/s^2): largest error of the base [deg]; largest departure of the second and third wrists from the carry pose [deg]; their largest speeds [deg/s]":
            '3.466; 4.80, 3.51; 142.2, 133.9',
        "5 kg at the empty arm's largest deceleration (142.99 rad/s^2): the base's torque last clipped [s]; the farthest angle reached [s]; the overrun of the commanded rest if the base braked at its limit to rest with the pose held [deg]":
            '0.041; 0.052; 2.372',
        "5 kg at the empty arm's largest deceleration (142.99 rad/s^2), the other joints held in the pose (the base alone, our controller, integrated exactly): overrun of the commanded rest [deg]; farthest angle [deg]; least clearance [m]":
            '2.387; -64.365; 0.0233',
        "5 kg at the empty arm's largest deceleration (142.99 rad/s^2): the base's speed when its torque was last clipped, simulated and with the pose held [rad/s]":
            '0.859, 0.476',
        'empty at its largest deceleration: farthest angle [deg]; beyond the commanded rest [deg]; least clearance [m]':
            '-61.977; 0.000; 0.0889',
        'empty at its largest deceleration: largest torque asked over its limit, base and second wrist; updates clipped, base and second wrist':
            '1.000, 0.140; 1, 0',
        'empty at its largest deceleration: largest error of the base [deg]; largest departure of the second and third wrists from the carry pose [deg]; their largest speeds [deg/s]':
            '0.042; 0.00, 0.00; 0.0, 0.0',
        "these five stops: each joint's largest speed after the start [deg/s], base to wrist 3":
            '177.9, 30.6, 31.4, 27.5, 150.8, 133.9',
    },
    'time steps': {
        '5 kg at 1.2 x from the place, at 1, 0.5, 0.25 ms: beyond the commanded rest [deg]':
            '1.6753, 1.7168, 1.7377',
        'the same: observed order of convergence; extrapolated to a zero time step [deg]':
            '0.99; 1.759',
    },
}


if __name__ == "__main__":
    main()
