#!/usr/bin/env python3
"""
One Apptronik Apollo carries a beam: the mission of the Team-Impedance-
Control repository with one robot, in MuJoCo.

Physical AI, Together, Chapter 15 (Section 15.3; Figure S15.1).

The repository (Heydari Shahna, 2026; a checkout given with
--team-impedance) simulates humanoids that carry a girder together on a
construction site with its two-loop controller. This script runs its
mission with one robot (humanoid/one_robot.py says what one robot adds):
the robot walks two gait cycles to an aluminium beam of the HEA 200
section, 1.2 m long, lying on two trestles, crouches, puts its palms under
the beam, takes its weight from the trestles, lifts it, carries it four
cycles (1 m) to two other trestles, sets it down, lets go, stands up and
steps back one cycle. The gait is the repository's quasi-static one: the
weight shifts onto one foot, the other is unloaded, lifted, moved and set
down, and the weight shifts back, so that every step lifts a foot off and
makes a contact.

The script records the plant's state at every step of 1 ms
(render.shapes.Recorder, which only copies it after MuJoCo's step), with the
soles' and the palms' contact forces, and reports from it and from the
mission's own log:
  * the mission's phases, its contact transitions and whether it completed;
  * in single support, where the zero-moment point is the centre of
    pressure of the stance sole, how far that centre stayed inside the
    sole's edges, unloaded and carrying, and, carrying, when and at which
    edge it was at one;
  * each touchdown: the sole's speed when it touched, and the largest force
    on it in the 50 ms after;
  * the actuators' largest use of their torque limits;
  * the beam: how closely it followed its reference while carried, how much
    of its weight the palms carried, and where it was set down.

Distances and margins are rounded down; deviations, forces, speeds and the
largest distance inside the edges of the samples between those at an edge,
up; at the digits printed.

Usage
    python3 humanoid_carry.py --team-impedance DIR [--qp nnls|daqp] [--check]
                              [--out DIR2] [--without-follow]

--qp daqp solves the load-sharing QP with daqp, as the repository does;
nnls (the default) with humanoid/qp.py, the same QP, where daqp is not
installed. The record states which. --out DIR2 writes the outputs into DIR2
instead of results/, so that a run to compare with the book's, such as one
with --qp daqp, leaves the book's results as they are. --without-follow
runs the mission without the third adaptation of humanoid/one_robot.py, the
beam's reference following the pelvis sideways, and reports only whether
and where it fails and which actuators reached their torque limits before
it (humanoid-without-follow.txt).

Outputs, in results/ next to this file
    humanoid-record.txt     the record of the experiment
    humanoid-summary.txt    every value, by group
    humanoid-trace.csv      every 10 ms: the time, the phase, the feet and
                            palms in contact, the soles' normal forces, the
                            stance sole's centre of pressure's distance
                            inside its edges in single support, the
                            actuators' largest share of their limits, the
                            beam's height and its distance from its
                            reference
    humanoid-frames.npz     the state at 30 frames per second, with the
                            contact forces, for humanoid_video.py
    humanoid-without-follow.txt   with --without-follow, instead of the above

Requires MuJoCo 3.16.0, NumPy and SciPy, and a checkout of
Team-Impedance-Control (commit eb530ea for the book's run); with --qp daqp,
daqp. About four minutes on one core.
"""
import argparse
import csv
import math
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import mujoco                                                     # noqa: E402
import numpy as np                                                # noqa: E402
import scipy                                                      # noqa: E402

FPS = 30
TOUCH_WINDOW = 0.05               # [s] after a touchdown's first contact
SOLE = (0.100, 0.0425)            # [m] the sole's half length and half width (the repository's)
EDGE = 0.0001                     # [m] a centre of pressure this close to an edge is at it
COMMIT = "eb530ea40ee1a8dbec96a7c80df36d976403606d"


def down(x, digits):
    f = 10 ** digits
    return f"{math.floor(x * f + 1e-9) / f:.{digits}f}"


def up(x, digits):
    f = 10 ** digits
    return f"{math.ceil(x * f - 1e-9) / f:.{digits}f}"


def contacts(R):
    """For the recorder: each sole's normal force [N] and centre of pressure
    in world x, y [m], each palm's normal force [N]."""
    m, d = R.m, R.d
    sole_ids, palm_ids = R.sole_geom, R.palm_geom
    c6 = np.zeros(6)

    def extra(m_, d_):
        out = np.zeros(8)
        num = {"l": np.zeros(2), "r": np.zeros(2)}
        for k in range(d.ncon):
            con = d.contact[k]
            for g in (con.geom1, con.geom2):
                if g in sole_ids:
                    mujoco.mj_contactForce(m, d, k, c6)
                    s = sole_ids[g][1]
                    j = 0 if s == "l" else 1
                    out[j] += c6[0]
                    num[s] += c6[0] * con.pos[:2]
                elif g in palm_ids:
                    mujoco.mj_contactForce(m, d, k, c6)
                    out[6 + (0 if palm_ids[g][1] == "l" else 1)] += c6[0]
        for j, s in enumerate("lr"):
            if out[j] > 1e-9:
                out[2 + 2 * j: 4 + 2 * j] = num[s] / out[j]
        return {"c": out}

    return extra


def run(checkout, qp, follow=True):
    from humanoid import one_robot
    mods = one_robot.load(checkout, qp)
    cfg, Mission = one_robot.adapt(mods, follow=follow)
    from render import shapes
    t0 = time.time()
    M = Mission(cfg)
    ph = one_robot.phases(mods, M)
    with shapes.Recorder(1000.0, select=lambda m, d: d is M.d, extra=contacts(M.R)) as rec:
        M.run(ph)
    wall = time.time() - t0
    t, qpos, qvel = rec.of(M.d)
    c = np.array([e["c"] for e in rec.runs[id(M.d)]["extra"]])
    return dict(M=M, mods=mods, cfg=cfg, t=t, qpos=qpos, qvel=qvel, c=c, wall=wall)


def phase_table(M):
    out = []
    for L in M.logs:
        if not out or out[-1][0] != L["phase"]:
            out.append([L["phase"], L["t"], L["t"]])
        out[-1][2] = L["t"]
    return out


def touchdowns(r):
    """Each touchdown: the foot, the time of its first contact and of the
    mode's switch [s], the sole's speed at the first contact [m/s] and the
    largest normal force in the 50 ms after it [N]."""
    M = r["M"]
    m = M.m
    d = mujoco.MjData(m)
    t, qpos, c = r["t"], r["qpos"], r["c"]
    out = []
    for (te, why, nfeet, ngr) in M.events:
        if not why.startswith("touch-down"):
            continue
        side = "l" if "'l'" in why else "r"
        j = 0 if side == "l" else 1
        k = int(np.searchsorted(t, te - 1e-9))
        k = min(k, len(t) - 1)
        first = k
        while first > 0 and c[first - 1, j] > 0.0:
            first -= 1
        site = m.site(f"r0_{side}_sole").id
        z = []
        for i in (max(first - 2, 0), first):
            d.qpos[:] = qpos[i]
            mujoco.mj_kinematics(m, d)
            z.append(d.site_xpos[site][2])
        speed = (z[0] - z[1]) / (t[first] - t[max(first - 2, 0)]) if first >= 2 else float("nan")
        win = (t >= t[first] - 1e-9) & (t <= t[first] + TOUCH_WINDOW + 1e-9)
        out.append(dict(side=side, first=float(t[first]), switch=float(te), speed=float(speed),
                        peak=float(c[win, j].max())))
    return out


def facts(r):
    M, cfg = r["M"], r["cfg"]
    L = M.logs
    W = M.mass * 9.81
    table = phase_table(M)
    lift = [e for e in M.events if e[1].startswith("lift-off")]
    touch = [e for e in M.events if e[1].startswith("touch-down")]
    # single support: the stance sole's centre of pressure inside its edges
    single = {"unloaded": [], "carrying": []}
    for x in L:
        if x["mode"][0] != 1:
            continue
        cm = [cc[6][1] for cc in x["conn"] if cc[0] == "foot"][0]
        single["carrying" if x["mode"][1] else "unloaded"].append((cm, x["t"], x["phase"]))
    least = {k: min(v) for k, v in single.items() if v}
    steps = {}
    for cm, t, ph in single["carrying"]:
        steps.setdefault(ph, []).append((cm, t))
    edge = [t for cm, t, ph in single["carrying"] if cm < EDGE]
    # the edge that those samples were at, in the sole's frame (x forward, y to the left,
    # z up), and the other samples between the first and the last of them
    at, others = set(), []
    if edge:
        X, Y = r["mods"]["share"].SOLE["X"], r["mods"]["share"].SOLE["Y"]
        for x in L:
            if x["mode"][0] != 1 or not x["mode"][1] or not min(edge) <= x["t"] <= max(edge):
                continue
            cc = [cc for cc in x["conn"] if cc[0] == "foot"][0]
            if cc[6][1] < EDGE:
                w = cc[4]                       # the ground's wrench in the sole's frame
                px, py = -w[4] / w[2], w[3] / w[2]
                gaps = {"back": px - X[0], "front": X[1] - px, "right": py - Y[0],
                        "left": Y[1] - py}
                at.add(min(gaps, key=gaps.get))
            else:
                others.append(cc[6][1])
    # actuators
    J = r["mods"]["scene"].JOINTS
    sat = np.array([x["sat"] for x in L])
    k = int(np.argmax(sat.max(axis=0)))
    i = int(np.argmax(sat[:, k]))
    carry = np.array([x["phase"].startswith("carry") for x in L])
    sat_carry = sat[carry].max() if carry.any() else float("nan")
    # the beam while carried, and where it was set down
    err = [np.linalg.norm(x["beam"][:3] - x["beam_ref"]) for x in L
           if x["phase"].startswith("carry") and "beam_ref" in x]
    share = [sum(cc[4][2] for cc in x["conn"] if cc[0] == "grasp") / W for x in L
             if x["phase"].startswith("carry")]
    h = M.h
    target = np.array([0.0, cfg.YB, cfg.zB + h / 2 + 0.0005])
    final = L[-1]["beam"][:3]
    eI = np.array([np.linalg.norm(x["eI"]) for x in L if "eI" in x and x["mode"][1]])
    td = touchdowns(r)
    out = {
        "the run": {
            "the robot; the beam, its length [m] and mass [kg]":
                f"one Apptronik Apollo (MuJoCo Menagerie's model, the repository's copy); "
                f"aluminium, the HEA 200 section, {cfg.beam_len:g}, {M.mass:.2f}",
            # 25 kg: "capable of lifting 55 lbs (25kg)", The Robot Report, 23 August 2023
            # (oitzman2023apollo in references.bib)
            "the beam's weight [N]; the payload reported for the Apollo [kg]": f"{up(W, 1)}; 25",
            "completed; simulated time [s]; failure":
                f"{M.failed is None}; {M.R.t:.2f}; {M.failed[1] if M.failed else 'none'}",
            "gait cycles: to the beam, carrying, back; their length [m]":
                f"2, 4, 1; {cfg.step:g}",
            "lift-offs; touchdowns; grasps closed and opened":
                f"{len(lift)}; {len(touch)}; "
                f"{sum(1 for e in M.events if e[1] == 'grasps closed')}, "
                f"{sum(1 for e in M.events if e[1] == 'grasps opened')}",
            "the phases' starts [s]: walking, reaching, taking the weight, lifting, carrying, "
            "setting down, letting go, stepping back":
                ", ".join(f"{next(p[1] for p in table if p[0].startswith(n)):.2f}"
                          for n in ("go1", "pre-reach", "load-transfer", "lift", "carry1", "place",
                                    "release", "back1")),
        },
        "single support": {
            "samples every 10 ms, unloaded; carrying": f"{len(single['unloaded'])}; "
                                                       f"{len(single['carrying'])}",
            "the stance sole's centre of pressure, least distance inside its edges, unloaded [mm]; "
            "when [s]; phase": (f"{down(1000 * least['unloaded'][0], 1)}; {least['unloaded'][1]:.2f}; "
                                f"{least['unloaded'][2]}"),
            "the same, carrying [mm]; when [s]; phase":
                (f"{down(1000 * least['carrying'][0], 1)}; {least['carrying'][1]:.2f}; "
                 f"{least['carrying'][2]}"),
            "the sole's half length and half width [mm]": f"{1000 * SOLE[0]:g}, {1000 * SOLE[1]:g}",
            "carrying: each step's least distance [mm], the stance foot in brackets": "; ".join(
                f"{ph} ({'r' if ph.endswith('-l') else 'l'}) {down(1000 * min(v)[0], 1)}"
                for ph, v in steps.items()),
            "carrying: samples within 0.1 mm of an edge; from, to [s]":
                f"{len(edge)}; " + (f"{min(edge):.2f}, {max(edge):.2f}" if edge else "-"),
            "carrying: the edge that they were at (the sole's x axis forward); the other samples "
            "from the first to the last of them, and their largest distance inside the edges [mm]":
                (f"{', '.join(sorted(at))}; {len(others)}, "
                 + (up(1000 * max(others), 1) if others else "-")) if edge else "-",
        },
        "touchdowns": {
            "count; the sole's speed at the first contact, largest [m/s]; the largest force in "
            "the 50 ms after it [N]":
                f"{len(td)}; {up(max(x['speed'] for x in td), 3)}; {up(max(x['peak'] for x in td), 0)}",
            "the first contact before the mode's switch, longest [ms]":
                f"{up(1000 * max(x['switch'] - x['first'] for x in td), 0)}",
            "each: foot, first contact [s], speed [m/s], largest force in 50 ms [N]": "; ".join(
                f"{x['side']} {x['first']:.3f} {up(x['speed'], 3)} {up(x['peak'], 0)}" for x in td),
        },
        "actuators": {
            "largest share of a torque limit; joint; when [s]; phase":
                f"{up(sat[i, k], 3)}; {J[k]}; {L[i]['t']:.2f}; {L[i]['phase']}",
            "largest share while carrying": f"{up(sat_carry, 3)}",
        },
        "the beam": {
            "carried: largest distance from its reference [mm]": f"{up(1000 * max(err), 1)}",
            "carried: the palms' normal forces as a share of its weight, least, largest":
                f"{down(min(share), 3)}, {up(max(share), 3)}",
            "set down: its centre from the trestles' middle, along the beam, across, height [mm]":
                ", ".join(f"{1000 * v:+.1f}" for v in (final - target)),
            "internal-wrench error while grasped (the repository's metric), RMS and median [N]":
                f"{up(float(np.sqrt(np.mean(eI ** 2))), 2)}, {up(float(np.median(eI)), 2)}",
        },
    }
    return out, td


def hardware():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "not reported"


def write_record(path, r, checkout, qp):
    M, cfg = r["M"], r["cfg"]
    sha = one_commit(checkout)
    import importlib.metadata as md
    try:
        daqp_v = md.version("daqp") if qp == "daqp" else None
    except md.PackageNotFoundError:
        daqp_v = "not reported"
    lines = [
        "Record of the experiment of Section 15.3: one Apollo carries a beam",
        "",
        "model and asset provenance",
        "  the Team-Impedance-Control repository (Heydari Shahna, 2026), commit " + sha +
        (" (the book's run: " + COMMIT[:7] + ")" if sha != COMMIT else "") +
        ", no project-wide licence (LICENSE.md): its package teamimp unchanged, with its copy of MuJoCo Menagerie's "
        "Apptronik Apollo (Apache License 2.0, visual meshes decimated by the repository); one "
        "robot's station, the beam of the HEA 200 section in aluminium, the loaded pelvis's "
        "shift and the beam's reference following it ours (humanoid/one_robot.py)",
        "simulator and solver",
        f"  MuJoCo {mujoco.__version__}; NumPy, SciPy and Python {np.__version__}, {scipy.__version__}, "
        f"{platform.python_version()}; the repository's options: "
        f"{mujoco.mjtIntegrator(M.m.opt.integrator).name[6:].lower()}, "
        f"{mujoco.mjtCone(M.m.opt.cone).name[7:].lower()} cones, impratio {M.m.opt.impratio:g}, "
        f"{M.m.opt.iterations} solver and {M.m.opt.noslip_iterations} noslip iterations; the "
        "load-sharing QP solved by " +
        ("daqp " + str(daqp_v) if qp == "daqp" else
         "humanoid/qp.py (SciPy's NNLS, the same QP as daqp's)"),
        "time steps",
        f"  the simulation 1 ms; the controller every {cfg.dt_ctrl * 1000:g} ms; the load sharing "
        "every 10 control steps",
        "actuators and sensors",
        "  the 32 joints' torque actuators of the repository's model, within their limits; the "
        "palms' and soles' contact wrenches from the simulator, without noise",
        "frames and units",
        "  SI units; world x along the beam, y the direction of walking, z up",
        "uncertainty bounds",
        "  none: the controller's model is the plant's, without contacts",
        "constraints",
        "  the actuators' torque limits; the palms' and soles' contact-wrench cones as the "
        "repository's load sharing weighs them",
        "terminal behaviour",
        "  standing on both feet after the step back; no rest is certified",
        "training and evaluation split",
        "  none: no learned module",
        "seeds",
        "  none: no random numbers",
        "hardware",
        f"  {platform.system()} {platform.machine()}, {hardware()}",
        "guarantee status",
        "  none: an empirical demonstration in simulation; the repository's decoupling and decay "
        "certificates concern a fixed mode of its model, and every touchdown is a contact made, "
        "which no certificate of this book covers; no physical robot",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Section 15.3: one Apollo carries a beam")
    ap.add_argument("--team-impedance", required=True, help="a checkout of Team-Impedance-Control")
    ap.add_argument("--qp", choices=("nnls", "daqp"), default="nnls")
    ap.add_argument("--check", action="store_true", help="compare the summary with EXPECTED")
    ap.add_argument("--out", metavar="DIR", help="write the outputs into DIR instead of results/ "
                    "(for a run to compare, such as one with --qp daqp)")
    ap.add_argument("--without-follow", action="store_true",
                    help="run the mission without the beam's reference following the pelvis "
                         "sideways and report how it fails (humanoid-without-follow.txt)")
    args = ap.parse_args()
    out_dir = os.path.abspath(args.out) if args.out else os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    if args.without_follow:
        values = without_follow(run(args.team_impedance, args.qp, follow=False))
        write_summary(os.path.join(out_dir, "humanoid-without-follow.txt"), values,
                      EXPECTED_WITHOUT_FOLLOW, args.check)
        return
    r = run(args.team_impedance, args.qp)
    M = r["M"]
    values, td = facts(r)
    values["the run"]["wall time [s]; the QP's solver"] = (
        f"{r['wall']:.0f}; {'daqp' if args.qp == 'daqp' else 'humanoid/qp.py (NNLS)'}")
    write_record(os.path.join(out_dir, "humanoid-record.txt"), r, args.team_impedance, args.qp)
    # the trace, every 10 ms
    with open(os.path.join(out_dir, "humanoid-trace.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(("t", "phase", "feet", "palms", "f_left", "f_right", "cop_margin_mm",
                     "max_torque_share", "beam_z", "beam_error_mm"))
        for x in M.logs:
            foot = [cc for cc in x["conn"] if cc[0] == "foot"]
            margin = foot[0][6][1] if x["mode"][0] == 1 else float("nan")
            f = {cc[2]: cc[4][2] for cc in foot}
            err = (1000 * np.linalg.norm(x["beam"][:3] - x["beam_ref"]) if "beam_ref" in x
                   else float("nan"))
            wr.writerow((f"{x['t']:.2f}", x["phase"], x["mode"][0], x["mode"][1],
                         f"{f.get('l', 0.0):.1f}", f"{f.get('r', 0.0):.1f}",
                         "" if np.isnan(margin) else down(1000 * margin, 1),
                         up(float(np.max(x["sat"])), 3), f"{x['beam'][2]:.4f}",
                         "" if np.isnan(err) else up(err, 1)))
    # the frames for the video: the recorded states nearest 30 per second
    t = r["t"]
    times = np.arange(0.0, t[-1] + 1e-9, 1.0 / FPS)
    idx = np.clip(np.searchsorted(t, times - 1e-9), 0, len(t) - 1)
    starts = phase_table(M)
    phase = [next(p[0] for p in reversed(starts) if p[1] <= t[i] + 1e-9) if t[i] >= starts[0][1]
             else starts[0][0] for i in idx]
    log_t = np.array([x["t"] for x in M.logs])
    mode = np.array([M.logs[min(int(np.searchsorted(log_t, t[i])), len(M.logs) - 1)]["mode"]
                     for i in idx])
    np.savez_compressed(os.path.join(out_dir, "humanoid-frames.npz"), t=t[idx], qpos=r["qpos"][idx],
                        contacts=r["c"][idx], phase=np.array(phase), mode=mode,
                        mass=np.array([M.mass]), weight=np.array([M.mass * 9.81]),
                        profile=np.array([r["cfg"].profile]), length=np.array([r["cfg"].beam_len]),
                        x_sup=np.array([r["cfg"].x_sup]), yb=np.array([r["cfg"].YB]),
                        touch=np.array([[x["first"], 0 if x["side"] == "l" else 1] for x in td]),
                        commit=np.array([one_commit(args.team_impedance)]),
                        qp=np.array([args.qp]))
    write_summary(os.path.join(out_dir, "humanoid-summary.txt"), values, EXPECTED, args.check)


def write_summary(path, values, expected, check):
    """Writes every value by group; with check, compares them with expected
    and stops with an error if one differs."""
    lines, failures = [], []
    for group, items in values.items():
        lines.append(f"[{group}]")
        for key, val in items.items():
            lines.append(f"  {key} = {val}")
            want = expected.get(group, {}).get(key)
            if check and want is not None and want != str(val):
                failures.append(f"{group}.{key}: expected {want!r}, got {val!r}")
    if check:
        n = sum(len(v) for v in expected.values())
        lines.append(f"[check] {n} values: " + ("all as expected" if not failures
                                                 else "FAILED: " + "; ".join(failures)))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if failures:
        raise SystemExit(1)


def without_follow(r):
    """The run without the beam's reference following the pelvis: whether
    and where it failed, and the actuators that reached their torque limits
    before it, each from when (the mission's log, every 10 ms)."""
    M = r["M"]
    J = list(r["mods"]["scene"].JOINTS)
    t_fail, why = M.failed if M.failed else (None, "none")
    phase = (next((x["phase"] for x in M.logs if x["t"] >= t_fail - 1e-9), M.logs[-1]["phase"])
             if t_fail is not None else "-")
    first = {}
    for x in M.logs:
        if t_fail is not None and x["t"] >= t_fail - 1e-9:
            break
        for k in np.where(np.asarray(x["sat"]) >= 0.999)[0]:
            first.setdefault(J[int(k)], (x["t"], x["phase"]))
    order = sorted(first.items(), key=lambda kv: (kv[1][0], kv[0]))
    legs = [j for j in J if any(p in j for p in ("hip", "knee", "ankle"))]
    wrists = [j for j in J if "wrist" in j]
    return {"the run without the beam's reference following the pelvis": {
        "completed; simulated time [s]; failure; when [s]; its phase":
            f"{M.failed is None}; {M.R.t:.2f}; {why}; "
            + (f"{t_fail:.2f}; {phase}" if t_fail is not None else "-; -"),
        "the actuators at their torque limits before it, each from [s] (phase)":
            "; ".join(f"{j} {t:.2f} ({ph})" for j, (t, ph) in order) or "none",
        "of them: the wrists' (of all wrists' actuators); the legs' (of all legs')":
            f"{sum(1 for j in first if j in wrists)} of {len(wrists)}; "
            f"{sum(1 for j in first if j in legs)} of {len(legs)}",
    }}


def one_commit(checkout):
    from humanoid import one_robot
    return one_robot.commit(checkout) or "not read"


EXPECTED = {   # every value of the summary but the wall time
    'the run': {
        'the robot; the beam, its length [m] and mass [kg]':
            "one Apptronik Apollo (MuJoCo Menagerie's model, the repository's copy); aluminium, the HEA 200 section, 1.2, 17.46",
        "the beam's weight [N]; the payload reported for the Apollo [kg]":
            '171.3; 25',
        'completed; simulated time [s]; failure':
            'True; 91.06; none',
        'gait cycles: to the beam, carrying, back; their length [m]':
            '2, 4, 1; 0.25',
        'lift-offs; touchdowns; grasps closed and opened':
            '14; 14; 1, 1',
        "the phases' starts [s]: walking, reaching, taking the weight, lifting, carrying, setting down, letting go, stepping back":
            '0.81, 20.86, 25.72, 29.22, 31.62, 71.70, 76.90, 80.10',
    },
    'single support': {
        'samples every 10 ms, unloaded; carrying':
            '809; 1087',
        "the stance sole's centre of pressure, least distance inside its edges, unloaded [mm]; when [s]; phase":
            '11.0; 81.72; back1-step-l',
        'the same, carrying [mm]; when [s]; phase':
            '-0.1; 33.89; carry1-step-l',
        "the sole's half length and half width [mm]":
            '100, 42.5',
        "carrying: each step's least distance [mm], the stance foot in brackets":
            'carry1-step-l (r) -0.1; carry1-step-r (l) 20.0; carry2-step-l (r) 4.9; carry2-step-r (l) 19.2; carry3-step-l (r) 4.9; carry3-step-r (l) 19.4; carry4-step-l (r) 3.9; carry4-step-r (l) 19.5',
        'carrying: samples within 0.1 mm of an edge; from, to [s]':
            '34; 33.67, 34.05',
        "carrying: the edge that they were at (the sole's x axis forward); the other samples from the first to the last of them, and their largest distance inside the edges [mm]":
            'front; 5, 2.3',
    },
    'touchdowns': {
        "count; the sole's speed at the first contact, largest [m/s]; the largest force in the 50 ms after it [N]":
            '14; 0.060; 305',
        "the first contact before the mode's switch, longest [ms]":
            '4',
        'each: foot, first contact [s], speed [m/s], largest force in 50 ms [N]':
            'l 3.672 0.052 196; r 9.027 0.058 209; l 13.698 0.053 198; r 19.051 0.058 287; l 34.471 0.056 238; r 39.824 0.058 215; l 44.491 0.054 237; r 49.843 0.058 228; l 54.513 0.054 235; r 59.861 0.059 222; l 64.527 0.054 242; r 69.886 0.058 217; l 82.914 0.060 297; r 88.257 0.058 305',
    },
    'actuators': {
        'largest share of a torque limit; joint; when [s]; phase':
            '0.962; r_ankle_pd; 33.93; carry1-step-l',
        'largest share while carrying':
            '0.962',
    },
    'the beam': {
        'carried: largest distance from its reference [mm]':
            '11.4',
        "carried: the palms' normal forces as a share of its weight, least, largest":
            '0.919, 1.035',
        "set down: its centre from the trestles' middle, along the beam, across, height [mm]":
            '-2.7, -3.3, -0.5',
        "internal-wrench error while grasped (the repository's metric), RMS and median [N]":
            '2.52, 0.41',
    },
}

EXPECTED_WITHOUT_FOLLOW = {   # --without-follow: every value of its report
    "the run without the beam's reference following the pelvis": {
        'completed; simulated time [s]; failure; when [s]; its phase':
            'False; 42.61; girder lost (pose error > 0.15 m); 40.11; carry1-step-r',
        'the actuators at their torque limits before it, each from [s] (phase)':
            'l_wrist_roll 33.51 (carry1-step-l); r_hip_ie 34.37 (carry1-step-l); r_wrist_roll 36.54 (carry1-shift); l_ankle_ie 39.15 (carry1-step-r); l_hip_ie 39.40 (carry1-step-r); l_ankle_pd 39.99 (carry1-step-r); l_knee_fe 39.99 (carry1-step-r); l_hip_fe 40.04 (carry1-step-r); l_hip_aa 40.09 (carry1-step-r); r_ankle_pd 40.09 (carry1-step-r); r_hip_fe 40.09 (carry1-step-r); r_knee_fe 40.09 (carry1-step-r); r_ankle_ie 40.10 (carry1-step-r)',
        "of them: the wrists' (of all wrists' actuators); the legs' (of all legs')":
            '2 of 6; 11 of 12',
    },
}


if __name__ == "__main__":
    main()
