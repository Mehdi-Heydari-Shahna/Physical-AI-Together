#!/usr/bin/env python3
"""
The Komatsu PC138US-8 as a documented model: what is published, what is ours,
what is fitted, and what the model then reproduces.

Physical AI, Together, Chapter 14 (Section 14.1).

This script builds the excavator of the package excavator and
  * writes the record of the experiment, with the sources of every
    published number (excavator/published.py);
  * fits the front's geometry to four numbers of the brochure's working
    range, and the arm cylinder's geometry to its stroke and the ISO arm
    force, and checks that the stored values still come out;
  * fits the one free parameter of the masses, the upper structure's centre
    of mass, to the brochure's ratings that tipping limits, and compares the
    model's front with the gravitational parameters identified on a
    PC138US-8 at Tampere University;
  * compares the model's lifting capacity with every entry of the
    brochure's lifting table, and says which limit binds;
  * checks the simulated machine, standing on the tipping lines in MuJoCo,
    against the static model: its settled support forces, extrapolated to
    zero, and the load, raised slowly at the lift point, from which it tips
    over;
  * writes the ratings against reach.

Usage
    python3 excavator_mujoco.py            # writes results/excavator-*
    python3 excavator_mujoco.py --check    # also checks every value quoted
                                           # in Section 14.1

Outputs, in results/ next to this file
    excavator-record.txt    the record of the experiment
    excavator-summary.txt   every value, by group
    excavator-lift-0.csv, excavator-lift-3.csv
                            the model's ratings against reach at the hook
                            heights 0 and 3 m: the rating, 75 % of the
                            tipping load over the front and over the side,
                            87 % of the hydraulic capacity

Requires MuJoCo 3.16.0, NumPy and SciPy (requirements.txt); run with
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
import scipy                                      # noqa: E402

from excavator import capacity, front, mass, model                 # noqa: E402
from excavator import published as pub                             # noqa: E402


def deg(x, digits=2):
    return f"{math.degrees(x):.{digits}f}"


def record_facts():
    m = model.build().compile()
    return {
        "MuJoCo": mujoco.__version__,
        "NumPy, SciPy, Python": f"{np.__version__}, {scipy.__version__}, {platform.python_version()}",
        "time step [s]; integrator; friction cone": (
            f"{m.opt.timestep:g}; {mujoco.mjtIntegrator(m.opt.integrator).name[6:].lower()}; "
            f"{mujoco.mjtCone(m.opt.cone).name[7:].lower()}"),
        "bodies, joints, actuators": f"{m.nbody - 1}, {m.njnt}, {m.nu}",
        "mass [kg]": f"{float(m.body_mass.sum()):g}",
        "contacts with the ground: spheres at x, y [m]":
            f"+-{capacity.FRONT_LINE:g}, +-{capacity.SIDE_LINE:g}",
        "contacts' softness, MuJoCo's defaults: solref; solimp": (
            f"{', '.join(f'{x:g}' for x in m.geom_solref[m.geom('ground').id])}; "
            f"{', '.join(f'{x:g}' for x in m.geom_solimp[m.geom('ground').id])}"),
    }


def published_facts():
    return {
        "operating weight [kg]; counterweight [kg]; bucket [kg] (brochure CEN00183-06)":
            f"{pub.OPERATING_WEIGHT:g}; {pub.COUNTERWEIGHT:g}; {pub.BUCKET_MASS:g}",
        "the web page's operating weight, without its configuration [kg]": "14700",
        "boom, arm [m]; track gauge, length on ground [m]":
            f"{pub.BOOM_LENGTH:g}, {pub.ARM_LENGTH:g}; {pub.TRACK_GAUGE:g}, {pub.TRACK_ON_GROUND:g}",
        "cylinders: boom, arm, bucket: number x bore x stroke [mm]": ", ".join(
            f"{n} x {1000 * b:g} x {1000 * s:g}" for n, b, s, _ in pub.CYLINDERS.values()),
        "relief pressure, implement [MPa]; pump's largest flow [L/min]":
            f"{pub.RELIEF_IMPLEMENT / 1e6:g}; {pub.PUMP_MAX_FLOW * 60000:g}",
        "ISO bucket and arm forces [kN]": f"{pub.BUCKET_FORCE_ISO / 1e3:g}, {pub.ARM_FORCE_ISO / 1e3:g}",
        "working range: reach, at ground, depth, height [m]": ", ".join(
            f"{front.TARGETS[k]:g}" for k in ("reach", "reach_ground", "depth", "height")),
        "lifting table: entries at a fixed reach; in the rows of the fit (3 and 0 m); limited by "
        "tipping there; at the maximum reach, not compared":
            f"{2 * len(pub.LIFT_TABLE)}; "
            f"{2 * sum(1 for v in pub.LIFT_TABLE.values() if v[4] == 'fit')}; "
            f"{len(capacity.tipping_entries())}; {2 * len(pub.LIFT_TABLE_MAX)}",
    }


def front_facts():
    p, res = front.fit_front()
    stored = (front.Z0, *front.THETA1, front.THETA2_OUT)
    arm, ares = front.fit_arm_cylinder()
    a_b, c_b, beta_b = front.boom_cylinder()
    rng = front.working_range()
    return {
        "fitted: boom foot height [m]; boom range [deg]; arm's outer limit [deg]":
            f"{p[0]:.5f}; {deg(p[1], 4)}, {deg(p[2], 4)}; {deg(p[3], 4)}",
        "the fit's residuals, reach, at ground, depth, height [mm]":
            ", ".join(f"{1000 * abs(r):.0e}" for r in res),
        "stored values reproduced (5 digits)":
            str(all(abs(a - b) < 5e-5 for a, b in zip(p, stored))),
        "the model's working range: reach, at ground, depth, height [m]":
            ", ".join(f"{rng[k]:.4f}" for k in ("reach", "reach_ground", "depth", "height")),
        "arm cylinder: arms a, c [m]; offset [deg]; largest lever [m]":
            f"{arm[0]:.4f}, {arm[1]:.4f}; {deg(arm[2], 3)}; {front.arm_lever_from_force():.4f}",
        "arm cylinder: stored values reproduced (4 digits)":
            str(all(abs(a - b) < 5e-5 for a, b in zip(arm, front.ARM_CYLINDER))),
        "boom cylinders: arms a, c [m]; offset [deg]; levers at the boom's lowest, level, highest [m]":
            (f"{a_b:.4f}, {c_b:.4f}; {deg(beta_b, 3)}; "
             + ", ".join(f"{front.derivative(front.boom_length, t):.3f}"
                         for t in (front.THETA1[0], 0.0, front.THETA1[1]))),
        "bucket cylinder's constant lever [m]": f"{front.bucket_lever():.4f}",
    }


def mass_facts():
    xu, rms = capacity.fit_upper()
    entries = capacity.tipping_entries()
    errs = [pub.TIPPING_SHARE * capacity.tipping(a, b, side) / r - 1 for b, a, side, r in entries]
    stick, boom = mass.tampere_moments()
    s_id, b_id = mass.tampere_identified()
    return {
        "parts [kg]: undercarriage, upper structure, counterweight, boom, arm, bucket": ", ".join(
            f"{x:g}" for x in (mass.LOWER["mass"], mass.UPPER["mass"], mass.COUNTERWEIGHT["mass"],
                               mass.BOOM["mass"], mass.ARM["mass"], mass.BUCKET["mass"])),
        "total [kg]": f"{mass.total():g}",
        "fitted: the upper structure's centre of mass behind the swing axis [m]; rms of the "
        "logarithms": f"{-xu:.5f}; {rms:.4f}",
        "stored value reproduced (5 digits)": str(abs(xu - mass.X_UPPER) < 5e-6),
        "tipping-limited entries: the model's rating less the brochure's, smallest, largest":
            f"{100 * min(errs):+.1f} %, {100 * max(errs):+.1f} %",
        "the same, entry by entry (hook, reach, over)": "; ".join(
            f"{b:g} m {a:g} m {'side' if side else 'front'} {100 * e:+.1f} %"
            for (b, a, side, _), e in zip(entries, errs)),
        "with a 750 kg attachment: |(pi_s3, pi_s4)|, |(pi_s5, pi_s6)| [kg m], model":
            f"{stick:.0f}, {boom:.0f}",
        "the same identified (Table 1) [kg m]; the model less them":
            f"{s_id:.0f}, {b_id:.0f}; {100 * (stick / s_id - 1):+.1f} %, {100 * (boom / b_id - 1):+.1f} %",
    }


def table_facts():
    out = {}
    below = []
    tested = []
    for (b, a), (cf, hf, cs, hs, role) in sorted(pub.LIFT_TABLE.items(), reverse=True):
        for side, given, starred in ((False, cf, hf), (True, cs, hs)):
            r = capacity.rating(a, b, side)
            key = (f"hook {b:g} m, reach {a:g} m, over the {'side' if side else 'front'}"
                   + (" (a row of the test)" if role == "test" else "")
                   + ": model [kg], limited by; brochure [kg]; difference")
            if r is None:
                out[key] = f"out of the front's reach; {given}{'*' if starred else ''}; -"
                continue
            out[key] = (f"{r[0]:.0f}, {r[1]}; {given}{'*' if starred else ''}; "
                        f"{100 * (r[0] / given - 1):+.1f} %")
            if role == "fit" and r[1].startswith("hydraulic") and r[0] < 0.95 * given:
                below.append(f"hook {b:g} m reach {a:g} m {'side' if side else 'front'}")
            if role == "test" and not starred:
                tested.append(r[0] / given - 1)
    out["rows of the fit: entries where the model's hydraulic limit lies more than 5 % below the "
        "brochure's rating"] = "; ".join(below)
    out["rows of the test: the entries that the brochure's tipping limits, the model less the "
        "brochure: smallest, largest"] = f"{100 * min(tested):+.1f} %, {100 * max(tested):+.1f} %"
    out["the maximum-reach column, whose reach the brochure does not give: hook [m]: over the "
        "front, over the side [kg]"] = "; ".join(
            f"{b:g}: {cf}{'*' if hf else ''}, {cs}{'*' if hs else ''}"
            for b, (cf, hf, cs, hs) in sorted(pub.LIFT_TABLE_MAX.items(), reverse=True))
    # why the entries out of the front's reach are: too far, or outside the joints' ranges
    links = pub.BOOM_LENGTH + pub.ARM_LENGTH
    for (b, a) in sorted(pub.LIFT_TABLE, reverse=True):
        if capacity.pose(a, b) is not None:
            continue
        key = f"hook {b:g} m, reach {a:g} m, out of the front's reach: "
        sol = front.inverse(a, b)
        if sol is None:
            out[key + "the bucket pin's distance from the boom foot; the boom and the arm "
                "together [m]"] = f"{math.hypot(a - front.X0, b - front.Z0):.3f}; {links:.3f}"
            continue
        t1, t2 = (math.degrees(v) for v in sol)
        lo, hi = (math.degrees(v) for v in front.THETA1)
        inner, outer = math.degrees(front.THETA2_IN), math.degrees(front.THETA2_OUT)
        out[key + "the boom's angle that the bucket pin needs, beyond the boom's fitted range; "
            "the arm's, beyond the arm's [deg]"] = (
            f"{t1:.2f}, {max(t1 - hi, lo - t1, 0.0):.2f}; "
            f"{t2:.2f}, {max(t2 - outer, inner - t2, 0.0):.2f}")
    out["hook 0 m: 87 % of the model's hydraulic capacity at reaches of 3.0 and 3.7 m [kg]"] = \
        ", ".join(f"{pub.HYDRAULIC_SHARE * capacity.hydraulic(a, 0.0)[0]:.0f}" for a in (3.0, 3.7))
    return out


def simulated_facts():
    out = {}
    worst = 0.0
    fractions = []
    for b, a, side, _ in capacity.tipping_entries():
        w, analytic = model.tipping_load(a, b, side)
        worst = max(worst, abs(w / analytic - 1))
        f, _ = model.onset(a, b, side)
        fractions.append(f)
        if (b, a) == (0.0, 4.6):
            out[f"hook 0 m, reach 4.6 m, over the {'side' if side else 'front'}: the settled "
                "forces extrapolated to zero; tips over, the load raised slowly; the static "
                "tipping load [kg]"] = f"{w:.1f}; {f * analytic:.1f}; {analytic:.1f}"
    n = len(capacity.tipping_entries())
    out[f"the {n} tipping-limited entries: largest difference of the extrapolated forces from "
        "the static model's"] = f"{100 * worst:.3f} %"
    out[f"the {n} tipping-limited entries: tips over from, as a fraction of the static tipping "
        "load, the load raised slowly: least, largest"] = f"{min(fractions):.4f}, {max(fractions):.4f}"
    return out


def lift_rows():
    rows = []
    for b in (0.0, 3.0):
        for a in np.arange(2.5, 7.3, 0.1):
            row = {"height": b, "reach": float(a)}
            for side, tag in ((False, "front"), (True, "side")):
                r = capacity.rating(float(a), b, side)
                t = capacity.tipping(float(a), b, side)
                h = capacity.hydraulic(float(a), b)
                row[tag] = r[0] if r else float("nan")
                row[tag + "_tipping"] = pub.TIPPING_SHARE * t if t else float("nan")
            row["hydraulic"] = pub.HYDRAULIC_SHARE * h[0] if h else float("nan")
            rows.append(row)
    return rows


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
        "Record of the experiment of Section 14.1: the Komatsu PC138US-8 as a documented model",
        "",
        "model and asset provenance",
        "  published (excavator/published.py): Komatsu's brochure CEN00183-06 (2014), the Australian "
        "web page (the arm cylinder's rod), the gravitational parameters identified by Ferlibas and "
        "Ghabcheloo (SICFP'21), the tipping lines and tool forces of ISO 10567:2007 and ISO "
        "6015:2006 (sample pages)",
        "  ours: the boom foot's x, the bucket's radius, the arm's inner and the bucket's limits, "
        "the cylinders' dead lengths and two rods, the boom cylinders' angle, the bucket's "
        "constant lever, the masses' split, the parts' centres of mass and boxes",
        "  fitted: the boom foot's height, the boom's range and the arm's outer limit (working range); "
        "the arm cylinder's triangle (stroke, ISO arm force); the upper structure's centre of mass "
        "(the tipping-limited ratings of the rows at 3 and 0 m; the rows at 6.1 and -3 m test "
        "it, their columns as the book's author read the printed table)",
        "simulator and solver",
        f"  MuJoCo {rec['MuJoCo']}; NumPy, SciPy and Python {rec['NumPy, SciPy, Python']}; "
        + rec["time step [s]; integrator; friction cone"] + "; contacts' softness (MuJoCo's "
        "defaults) solref, solimp " + rec["contacts' softness, MuJoCo's defaults: solref; solimp"],
        "time steps",
        "  1 ms; the extrapolated forces after 3 s; the onset runs settle 2 s without a load, "
        "raise the load's weight at the lift point over 5 s and hold it 24 s; tipped if the "
        "undercarriage then leans more than 0.1 deg",
        "actuators and sensors",
        "  the front rigid in its pose for the static checks; otherwise torque actuators at the "
        "swing, boom, arm and bucket joints, unlimited and not used in Section 14.1; no sensors",
        "frames and units",
        "  SI units; the upper structure's x along the boom, z up, the origin on the ground on the "
        "swing axis",
        "uncertainty bounds",
        "  none: the model is nominal; Section 14.1 compares it with the brochure",
        "constraints",
        "  the brochure's ratings, at most 75 % of the tipping load and 87 % of the hydraulic "
        "capacity (SAE J/ISO 10567)",
        "terminal behaviour",
        "  none: Section 14.1 measures limits",
        "training and evaluation split",
        "  none: nothing learns in Section 14.1",
        "seeds",
        "  0, for the starts of the working range's optimisation",
        "hardware",
        f"  {platform.system()} {platform.machine()}, {hardware()}",
        "guarantee status",
        "  none: a documented model, compared with published ratings; not a validation against "
        "the machine",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Section 14.1: the Komatsu PC138US-8")
    ap.add_argument("--check", action="store_true", help="check every value quoted in Section 14.1")
    args = ap.parse_args()
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    facts = {"record": record_facts(),
             "what is published": published_facts(),
             "the front": front_facts(),
             "the masses": mass_facts(),
             "the lifting table": table_facts(),
             "the simulated machine": simulated_facts()}
    rows = lift_rows()
    for height in (0.0, 3.0):
        name = f"excavator-lift-{height:g}.csv"
        with open(os.path.join(out_dir, name), "w", newline="", encoding="utf-8") as fh:
            part = [r for r in rows if r["height"] == height]
            w = csv.DictWriter(fh, fieldnames=list(part[0].keys()))
            w.writeheader()
            for row in part:
                w.writerow({k: f"{v:.6g}" for k, v in row.items()})
    write_record(os.path.join(out_dir, "excavator-record.txt"), facts)
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
    with open(os.path.join(out_dir, "excavator-summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if failures:
        raise SystemExit(1)


EXPECTED = {   # every value of the summary but the versions of the libraries
    'record': {
        'time step [s]; integrator; friction cone':
            '0.001; implicitfast; elliptic',
        'bodies, joints, actuators':
            '6, 5, 4',
        'mass [kg]':
            '13480',
        'contacts with the ground: spheres at x, y [m]':
            '+-1.44, +-0.995',
        "contacts' softness, MuJoCo's defaults: solref; solimp":
            '0.02, 1; 0.9, 0.95, 0.001, 0.5, 2',
    },
    'what is published': {
        'operating weight [kg]; counterweight [kg]; bucket [kg] (brochure CEN00183-06)':
            '13480; 3250; 399',
        "the web page's operating weight, without its configuration [kg]":
            '14700',
        'boom, arm [m]; track gauge, length on ground [m]':
            '4.6, 2.5; 1.99, 2.88',
        'cylinders: boom, arm, bucket: number x bore x stroke [mm]':
            '2 x 105 x 1055, 1 x 110 x 1175, 1 x 95 x 885',
        "relief pressure, implement [MPa]; pump's largest flow [L/min]":
            '34.8; 241.5',
        'ISO bucket and arm forces [kN]':
            '93.2, 61.8',
        'working range: reach, at ground, depth, height [m]':
            '8.3, 8.18, 5.48, 9.34',
        'lifting table: entries at a fixed reach; in the rows of the fit (3 and 0 m); limited by tipping there; at the maximum reach, not compared':
            '18; 12; 8; 8',
    },
    'the front': {
        "fitted: boom foot height [m]; boom range [deg]; arm's outer limit [deg]":
            '1.38910; -46.1821, 83.2024; -14.2481',
        "the fit's residuals, reach, at ground, depth, height [mm]":
            '2e-07, 2e-07, 7e-08, 2e-07',
        'stored values reproduced (5 digits)':
            'True',
        "the model's working range: reach, at ground, depth, height [m]":
            '8.3000, 8.1800, 5.4800, 9.3400',
        'arm cylinder: arms a, c [m]; offset [deg]; largest lever [m]':
            '2.2474, 0.6634; 18.235; 0.6634',
        'arm cylinder: stored values reproduced (4 digits)':
            'True',
        "boom cylinders: arms a, c [m]; offset [deg]; levers at the boom's lowest, level, highest [m]":
            '2.0907, 0.5837; -66.182; 0.268, 0.578, 0.238',
        "bucket cylinder's constant lever [m]":
            '0.3967',
    },
    'the masses': {
        'parts [kg]: undercarriage, upper structure, counterweight, boom, arm, bucket':
            '4950, 3041, 3250, 1180, 660, 399',
        'total [kg]':
            '13480',
        "fitted: the upper structure's centre of mass behind the swing axis [m]; rms of the logarithms":
            '0.76134; 0.0203',
        'stored value reproduced (5 digits)':
            'True',
        "tipping-limited entries: the model's rating less the brochure's, smallest, largest":
            '-2.7 %, +3.4 %',
        'the same, entry by entry (hook, reach, over)':
            '0 m 3 m side +1.2 %; 0 m 4.6 m front +1.9 %; 0 m 4.6 m side -1.6 %; 0 m 6.1 m front +3.4 %; 0 m 6.1 m side -0.2 %; 3 m 4.6 m side -2.7 %; 3 m 6.1 m front +1.9 %; 3 m 6.1 m side -1.8 %',
        'with a 750 kg attachment: |(pi_s3, pi_s4)|, |(pi_s5, pi_s6)| [kg m], model':
            '2537, 8851',
        'the same identified (Table 1) [kg m]; the model less them':
            '2756, 8378; -8.0 %, +5.6 %',
    },
    'the lifting table': {
        'hook 6.1 m, reach 4.6 m, over the front (a row of the test): model [kg], limited by; brochure [kg]; difference':
            '3606, hydraulic boom; 3060*; +17.8 %',
        'hook 6.1 m, reach 4.6 m, over the side (a row of the test): model [kg], limited by; brochure [kg]; difference':
            '3094, tipping; 3060*; +1.1 %',
        'hook 3 m, reach 6.1 m, over the front: model [kg], limited by; brochure [kg]; difference':
            '2934, tipping; 2880; +1.9 %',
        'hook 3 m, reach 6.1 m, over the side: model [kg], limited by; brochure [kg]; difference':
            '1797, tipping; 1830; -1.8 %',
        'hook 3 m, reach 4.6 m, over the front: model [kg], limited by; brochure [kg]; difference':
            '4376, hydraulic arm; 4320*; +1.3 %',
        'hook 3 m, reach 4.6 m, over the side: model [kg], limited by; brochure [kg]; difference':
            '2909, tipping; 2990; -2.7 %',
        'hook 3 m, reach 3 m, over the front: model [kg], limited by; brochure [kg]; difference':
            '5109, hydraulic arm; 5770*; -11.5 %',
        'hook 3 m, reach 3 m, over the side: model [kg], limited by; brochure [kg]; difference':
            '5109, hydraulic arm; 5770*; -11.5 %',
        'hook 0 m, reach 6.1 m, over the front: model [kg], limited by; brochure [kg]; difference':
            '2780, tipping; 2690; +3.4 %',
        'hook 0 m, reach 6.1 m, over the side: model [kg], limited by; brochure [kg]; difference':
            '1657, tipping; 1660; -0.2 %',
        'hook 0 m, reach 4.6 m, over the front: model [kg], limited by; brochure [kg]; difference':
            '4343, tipping; 4260; +1.9 %',
        'hook 0 m, reach 4.6 m, over the side: model [kg], limited by; brochure [kg]; difference':
            '2559, tipping; 2600; -1.6 %',
        'hook 0 m, reach 3 m, over the front: model [kg], limited by; brochure [kg]; difference':
            '2609, hydraulic arm; 5630*; -53.7 %',
        'hook 0 m, reach 3 m, over the side: model [kg], limited by; brochure [kg]; difference':
            '2609, hydraulic arm; 4840; -46.1 %',
        'hook -3 m, reach 4.6 m, over the front (a row of the test): model [kg], limited by; brochure [kg]; difference':
            '4359, tipping; 4180; +4.3 %',
        'hook -3 m, reach 4.6 m, over the side (a row of the test): model [kg], limited by; brochure [kg]; difference':
            '2573, tipping; 2540; +1.3 %',
        'hook -3 m, reach 3 m, over the front (a row of the test): model [kg], limited by; brochure [kg]; difference':
            '6726, hydraulic boom; 6040*; +11.3 %',
        'hook -3 m, reach 3 m, over the side (a row of the test): model [kg], limited by; brochure [kg]; difference':
            '5039, tipping; 4820; +4.5 %',
        "rows of the fit: entries where the model's hydraulic limit lies more than 5 % below the brochure's rating":
            'hook 3 m reach 3 m front; hook 3 m reach 3 m side; hook 0 m reach 3 m front; hook 0 m reach 3 m side',
        "rows of the test: the entries that the brochure's tipping limits, the model less the brochure: smallest, largest":
            '+1.3 %, +4.5 %',
        'the maximum-reach column, whose reach the brochure does not give: hook [m]: over the front, over the side [kg]':
            '6.1: 1690*, 1690*; 3: 1580*, 1370; 0: 1940*, 1290; -3: 3000, 1850',
        "hook 0 m: 87 % of the model's hydraulic capacity at reaches of 3.0 and 3.7 m [kg]":
            '2609, 7306',
    },
    'the simulated machine': {
        'hook 0 m, reach 4.6 m, over the front: the settled forces extrapolated to zero; tips over, the load raised slowly; the static tipping load [kg]':
            '5789.6; 5789.5; 5790.0',
        'hook 0 m, reach 4.6 m, over the side: the settled forces extrapolated to zero; tips over, the load raised slowly; the static tipping load [kg]':
            '3410.8; 3410.7; 3411.3',
        "the 8 tipping-limited entries: largest difference of the extrapolated forces from the static model's":
            '0.027 %',
        'the 8 tipping-limited entries: tips over from, as a fraction of the static tipping load, the load raised slowly: least, largest':
            '0.9996, 0.9999',
    },
}


if __name__ == "__main__":
    main()
