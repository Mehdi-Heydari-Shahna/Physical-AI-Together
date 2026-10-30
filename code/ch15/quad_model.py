#!/usr/bin/env python3
"""
The Go2 in MuJoCo and the reduced model that its governor uses.

Physical AI, Together, Chapter 15 (Section 15.1).

This script
  * writes the record of the experiment;
  * states what go2.xml of MuJoCo Menagerie (quad/description) declares:
    bodies, masses, joints, motors and their limits, the feet's contacts and
    the solver's options, and what quad/model.py changes;
  * settles the robot on its stance controller (quad/sim.py) and reports the
    stance: the feet's places, sink and loads, and the centre of mass, against
    the constants that quad/reduced.py declares;
  * reports how far a free foot sinks into MuJoCo's soft floor under loads
    from 1 to 80 N;
  * shifts the centre of mass along the shift line by 55.5 mm, the
    requirement's place of Section 15.2, at the largest accelerations 0.1,
    0.25 and 0.5 m/s^2, and compares the simulated centre of pressure with the
    cart-table model's zero-moment point, planned and for the simulated
    centre of mass and its acceleration, and with the centre of pressure of
    the contact forces alone;
  * repeats the shift at 0.25 m/s^2 with a time step of 0.5 ms.
With --menagerie DIR (a checkout of MuJoCo Menagerie at the commit of the
record), it also compiles go2.xml with its meshes and compares the masses,
inertias, centres of mass and collision geometries with the model of
quad/model.py; the shipped summary was written with it. Without it, --check
skips that one value.

Usage
    python3 quad_model.py            # writes results/model-*
    python3 quad_model.py --check    # also checks every value of its summary

Outputs, in results/ next to this file
    model-summary.txt    every value, by group; with --check, the comparison
                         with the values expected
    model-shift.csv      the shift at 0.25 m/s^2, every 5 ms: the time (t, s),
                         along the shift line from home (m): the planned
                         zero-moment point (zmp), the centre of pressure
                         (cop), that of the forces alone (fcop), the centre of
                         mass, planned and simulated (c_p, c) (Figure 15.2)

Requires MuJoCo 3.16.0, NumPy and SciPy (requirements.txt); run with CPython
3.13.17, other versions of Python not tried. About ten seconds.
"""
import argparse
import csv
import math
import os
import platform
import sys
import time

import mujoco
import numpy as np
import scipy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "ch10"))

from quad import governed as gv                                  # noqa: E402
from quad import model, sim, support                              # noqa: E402
from quad import reduced as rm                                   # noqa: E402

ACCELS = (0.1, 0.25, 0.5)
LOADS = (1.0, 5.0, 20.0, 40.0, 80.0)


def down(x, digits=2):
    q = 10 ** digits
    return f"{math.floor(round(float(x) * q, 6)) / q + 0.0:.{digits}f}"


def up(x, digits=2):
    q = 10 ** digits
    return f"{math.ceil(round(float(x) * q, 6)) / q + 0.0:.{digits}f}"


def mm(x, digits=1):
    return f"{1000.0 * x:.{digits}f}"


# ----------------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------------
def model_facts():
    m = model.build(0.001)
    spec = mujoco.MjSpec.from_file(model.XML)
    n_mesh = sum(1 for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
    mass = dict(zip(names, m.body_mass))
    leg = sum(mass[f"FR_{p}"] for p in ("hip", "thigh", "calf"))
    foot = m.geom(model.LEGS[0])
    others = [g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g])
              and g not in [m.geom(l).id for l in model.LEGS] and g != m.geom("floor").id]
    jr = m.jnt_range[1:4]
    return {
        "bodies besides the world; joints besides the free joint; motors; nq, nv":
            f"{m.nbody - 1}; {m.njnt - 1}; {m.nu}; {m.nq}, {m.nv}",
        "the mass [kg]; the trunk's [kg]; one leg's, hip, thigh and calf [kg]":
            f"{np.sum(m.body_mass):.3f}; {mass['base']:.3f}; {leg:.3f}",
        "the motors' torque limits, hip, thigh, knee [N m]; the joints' ranges of a front leg [rad]":
            (", ".join(f"{x:g}" for x in m.actuator_ctrlrange[:3, 1]) + "; "
             + ", ".join(f"{a:g} to {b:g}" for a, b in jr)),
        "the joints' damping [N m s/rad], armature [kg m^2], friction loss [N m]":
            f"{m.dof_damping[6]:g}, {m.dof_armature[6]:g}, {m.dof_frictionloss[6]:g}",
        "a foot: sphere radius [m]; friction, sliding, torsional, rolling; contact dimension; "
        "impedance (solimp); priority":
            (f"{m.geom_size[foot.id][0]:g}; " + ", ".join(f"{x:g}" for x in m.geom_friction[foot.id])
             + f"; {m.geom_condim[foot.id]}; " + ", ".join(f"{x:g}" for x in m.geom_solimp[foot.id])
             + f"; {m.geom_priority[foot.id]}"),
        "the other collision geometries: count; their contact dimension; friction":
            f"{len(others)}; {m.geom_condim[others[0]]}; {m.geom_friction[others[0]][0]:g}",
        "the solver: friction cone; impratio; integrator; the time step of go2.xml [s]; ours [s]":
            (f"{'elliptic' if m.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC else 'pyramidal'}; "
             f"{m.opt.impratio:g}; "
             f"{mujoco.mjtIntegrator(m.opt.integrator).name.replace('mjINT_', '').lower()}; "
             f"{mujoco.MjModel.from_xml_string('<mujoco/>').opt.timestep:g}; {m.opt.timestep:g}"),
        "quad/model.py: visual meshes deleted; a floor plane added":
            f"{n_mesh}; yes",
    }


def menagerie_facts(path):
    """go2.xml compiled with its meshes from a Menagerie checkout, against the
    model of quad/model.py."""
    full = mujoco.MjModel.from_xml_path(os.path.join(path, "unitree_go2", "go2.xml"))
    ours = model.build(full.opt.timestep)

    def collision(m):
        out = []
        for g in range(m.ngeom):
            if (m.geom_contype[g] or m.geom_conaffinity[g]) and m.geom_bodyid[g] != 0:
                out.append((int(m.geom_bodyid[g]), int(m.geom_type[g]),
                            tuple(np.round(m.geom_size[g], 9)), tuple(np.round(m.geom_pos[g], 9)),
                            int(m.geom_condim[g]), tuple(np.round(m.geom_friction[g], 9))))
        return sorted(out)
    same = collision(full) == collision(ours)
    return {
        "with its meshes: the largest difference of a body's mass [kg], inertia [kg m^2], centre "
        "of mass [m]; the collision geometries the same":
            (f"{np.max(np.abs(full.body_mass - ours.body_mass)):g}, "
             f"{np.max(np.abs(full.body_inertia - ours.body_inertia)):g}, "
             f"{np.max(np.abs(full.body_ipos - ours.body_ipos)):g}; {'yes' if same else 'no'}"),
    }


# ----------------------------------------------------------------------------
# The stance
# ----------------------------------------------------------------------------
def stance_facts():
    mc = sim.Machine(home=rm.COM0)
    F, other, cop, fcop = model.ground_wrench(mc.m, mc.d, mc.ids)
    sink = mc.ids.radius - mc.feet0[:, 2]
    weight = mc.ids.mass * model.G
    feet = mc.feet0[:, :2]
    return {
        "the feet's places, front and rear [m]":
            (f"x {feet[0, 0]:.4f}, y +-{feet[0, 1]:.4f}; x {feet[2, 0]:.4f}, y +-{feet[2, 1]:.4f}"),
        "the declared places in quad/reduced.py [m]; the largest difference [mm]":
            (f"x {rm.FEET[0, 0]:.4f}, y +-{rm.FEET[0, 1]:.4f}; x {rm.FEET[2, 0]:.4f}, "
             f"y +-{rm.FEET[2, 1]:.4f}; {up(1000 * np.max(np.abs(feet - rm.FEET)), 2)}"),
        "the feet's sink, front and rear [mm]; their loads, front and rear [N]; the loads' sum "
        "and the weight [N]":
            (f"{mm(sink[0])}, {mm(sink[2])}; {F[0, 2]:.1f}, {F[2, 2]:.1f}; "
             f"{np.sum(F[:, 2]):.1f}, {weight:.1f}"),
        "the centre of mass after the first second, on joint servos [m]; at home, the base at "
        "0.27 m [m]":
            (f"{mc.com_first[0]:.4f}, {mc.com_first[1]:.4f}, {mc.com_first[2]:.4f}; "
             f"{mc.com0[0]:.4f}, {mc.com0[1]:.4f}, {mc.com0[2]:.4f}"),
        "the declared home and height in quad/reduced.py [m]; the settled centre of mass's "
        "distance from the declared home [mm]":
            (f"{rm.COM0[0]:.4f}, {rm.COM0[1]:.4f}; {rm.Z_C:.4f}; "
             f"{up(1000 * math.hypot(mc.com0[0] - rm.COM0[0], mc.com0[1] - rm.COM0[1]), 2)}"),
        "the rectangle's sides [m]; the triangle FL, RL, RR's incentre [m] and inradius [m]; "
        "home inside the triangle and the rectangle [mm]":
            (f"{rm.FEET[0, 0] - rm.FEET[2, 0]:.4f}, {2 * rm.FEET[0, 1]:.4f}; "
             f"{rm.INCENTRE[0]:.4f}, {rm.INCENTRE[1]:.4f}; {down(rm.INRADIUS, 4)}; "
             f"{down(1000 * support.signed_distance(rm.TRI, rm.COM0), 1)}, "
             f"{down(1000 * support.signed_distance(rm.RECT, rm.COM0), 1)}"),
        "home behind the middle of the feet [mm]; the incentre's distance from home [mm]":
            (f"{down(1000 * (0.5 * (rm.FEET[0, 0] + rm.FEET[2, 0]) - rm.COM0[0]), 1)}; "
             f"{down(1000 * rm.S_INCENTRE, 1)}"),
    }


def sink_facts():
    """A free sphere with a foot's contact on the floor: its equilibrium sink
    under each load."""
    m0 = model.build(0.001)
    fid = m0.geom("FL").id
    imp = " ".join(f"{x:g}" for x in m0.geom_solimp[fid])
    fr = " ".join(f"{x:g}" for x in m0.geom_friction[fid])
    out = []
    for load in LOADS:
        xml = (f"<mujoco><option timestep='0.001' cone='elliptic' impratio='100'/><worldbody>"
               f"<geom type='plane' size='0 0 0.05'/><body pos='0 0 0.05'><joint type='slide' "
               f"axis='0 0 1'/><geom type='sphere' size='{m0.geom_size[fid][0]:g}' margin='0.001' "
               f"priority='1' solimp='{imp}' condim='6' friction='{fr}' "
               f"mass='{load / model.G:.9g}'/></body></worldbody></mujoco>")
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        for _ in range(4000):
            mujoco.mj_step(m, d)
        out.append(m0.geom_size[fid][0] - (0.05 + d.qpos[0]))
    return {"a free foot's sink under " + ", ".join(f"{x:g}" for x in LOADS) + " N [mm]":
            ", ".join(mm(x, 2) for x in out)}


# ----------------------------------------------------------------------------
# The cart-table model against MuJoCo
# ----------------------------------------------------------------------------
def shift(amax, timestep=sim.TIMESTEP, distance=None, keep=False):
    """Shift the centre of mass from home along the line by 'distance', with
    the planner of quad/governed.py at the largest acceleration amax, then
    hold 0.6 s. Returns the largest distances of the centre of pressure from
    the planned zero-moment point, from the cart-table point of the
    simulated centre of mass and its acceleration, and of the forces' centre
    of pressure from the full one [m]; the centre of mass's largest
    deviation from its plan [m]; and, if keep, rows for the figure."""
    distance = gv.rest_place(gv.FLOOR + gv.C3) if distance is None else distance
    saved = gv.A_MAX
    gv.A_MAX = amax
    try:
        accs = gv.motion(0.0, 0.0, 0.0, distance, distance) + [0.0, 0.0, 0.0]
    finally:
        gv.A_MAX = saved
    mc = sim.Machine(timestep=timestep, home=rm.COM0)
    s, v, a = 0.0, 0.0, 0.0
    rows, worst = [], np.zeros(4)
    zero = np.zeros(2)
    for a1 in accs:
        seg = gv.Segment(s, v, a, a1, "before")
        n0 = mc.steps()
        mc.advance(gv.T, lambda t, seg=seg: (rm.COM0 + seg.position(t) * rm.U,
                                              seg.accel(t) * rm.U, 1.0, 0.0))
        log = mc.arrays(n0)
        taus = (np.arange(len(log["t"])) + 1) * timestep
        planned = np.array([seg.position(t) for t in taus])
        acc = np.array([seg.accel(t) for t in taus])
        zmp = rm.COM0[None, :] + (planned - rm.K * acc)[:, None] * rm.U[None, :]
        cop = np.stack([log["copx"], log["copy"]], 1)
        fcop = np.stack([log["fcopx"], log["fcopy"]], 1)
        com = np.stack([log["cx"], log["cy"]], 1)
        worst[0] = max(worst[0], np.max(np.linalg.norm(cop - zmp, axis=1)))
        worst[2] = max(worst[2], np.max(np.linalg.norm(fcop - cop, axis=1)))
        worst[3] = max(worst[3], np.max(np.linalg.norm(com - (rm.COM0[None, :] + planned[:, None]
                                                             * rm.U[None, :]), axis=1)))
        ct = np.stack([log["ctx"], log["cty"]], 1)
        worst[1] = max(worst[1], np.max(np.linalg.norm(cop - ct, axis=1)))
        if keep:
            step = max(1, int(round(0.005 / timestep)))
            for j in range(step - 1, len(taus), step):
                rows.append(dict(t=f"{log['t'][j]:.3f}",
                                 zmp=f"{(zmp[j] - rm.COM0) @ rm.U:.5f}",
                                 cop=f"{(cop[j] - rm.COM0) @ rm.U:.5f}",
                                 fcop=f"{(fcop[j] - rm.COM0) @ rm.U:.5f}",
                                 c_p=f"{planned[j]:.5f}", c=f"{(com[j] - rm.COM0) @ rm.U:.5f}"))
        s, v, a = seg.end()[:3]
    return worst, rows, len(accs)


def shift_facts():
    out, rows = {}, None
    for amax in ACCELS:
        w, r, n = shift(amax, keep=(amax == gv.A_MAX))
        if amax == gv.A_MAX:
            rows, w_ref = r, w
        out[f"a shift of {1000 * gv.rest_place(gv.FLOOR + gv.C3):.1f} mm at {amax:g} m/s^2 "
            "and 0.6 s of rest: the centre of pressure's largest distance from the planned "
            "zero-moment point, and from the cart-table point of the simulated centre of mass "
            "[mm]; the forces' centre of pressure from it [mm]; the centre of mass from its "
            "plan [mm]; periods"] = (f"{up(1000 * w[0], 1)}, {up(1000 * w[1], 1)}; "
                                     f"{up(1000 * w[2], 1)}; {up(1000 * w[3], 1)}; {n}")
    w2, _, n = shift(gv.A_MAX, timestep=0.0005)
    out["the shift at 0.25 m/s^2 with a time step of 0.5 ms: the same distances [mm]"] = (
        f"{up(1000 * w2[0], 1)}, {up(1000 * w2[1], 1)}; {up(1000 * w2[2], 1)}; "
        f"{up(1000 * w2[3], 1)}")
    return out, rows


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description="Section 15.1: the Go2 and its reduced model")
    ap.add_argument("--check", action="store_true", help="check every value of the summary")
    ap.add_argument("--menagerie", default=None,
                    help="a checkout of MuJoCo Menagerie, to compare go2.xml with its meshes")
    args = ap.parse_args()
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    facts = {"record": {
        "MuJoCo": mujoco.__version__,
        "NumPy, SciPy, Python": f"{np.__version__}, {scipy.__version__}, "
                                f"{platform.python_version()}",
        "the model": "MuJoCo Menagerie's unitree_go2/go2.xml, commit f3b6013, BSD-3-Clause"}}
    facts["the model"] = model_facts()
    if args.menagerie:
        facts["the model"].update(menagerie_facts(args.menagerie))
    facts["the stance"] = stance_facts()
    facts["the floor's softness"] = sink_facts()
    shifts, rows = shift_facts()
    facts["the cart-table model against MuJoCo"] = shifts
    write_csv(os.path.join(out_dir, "model-shift.csv"), rows)
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
                     for key in items if key not in facts.get(group, {})
                     and not key.startswith("with its meshes")]
        n = sum(len(v) for v in EXPECTED.values())
        lines.append(f"[check] {n} values, among them every value that Section 15.1 quotes: " +
                     ("all as expected" if not failures and n else "FAILED: " + "; ".join(failures)))
    lines.append(f"[time] {time.time() - t0:.0f} s")
    with open(os.path.join(out_dir, "model-summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines[:-1]) + "\n")
    print("\n".join(lines))
    if failures:
        raise SystemExit(1)


EXPECTED = {   # every value of the summary but the versions of the libraries
    'record': {
        'the model':
            "MuJoCo Menagerie's unitree_go2/go2.xml, commit f3b6013, BSD-3-Clause",
    },
    'the model': {
        'bodies besides the world; joints besides the free joint; motors; nq, nv':
            '13; 12; 12; 19, 18',
        "the mass [kg]; the trunk's [kg]; one leg's, hip, thigh and calf [kg]":
            '15.206; 6.921; 2.071',
        "the motors' torque limits, hip, thigh, knee [N m]; the joints' ranges of a front leg [rad]":
            '23.7, 23.7, 45.43; -1.0472 to 1.0472, -1.5708 to 3.4907, -2.7227 to -0.83776',
        "the joints' damping [N m s/rad], armature [kg m^2], friction loss [N m]":
            '2, 0.01, 0.2',
        'a foot: sphere radius [m]; friction, sliding, torsional, rolling; contact dimension; impedance (solimp); priority':
            '0.022; 0.8, 0.02, 0.01; 6; 0.015, 1, 0.022, 0.5, 2; 1',
        'the other collision geometries: count; their contact dimension; friction':
            '19; 1; 0.6',
        'the solver: friction cone; impratio; integrator; the time step of go2.xml [s]; ours [s]':
            'elliptic; 100; euler; 0.002; 0.001',
        'quad/model.py: visual meshes deleted; a floor plane added':
            '33; yes',
        "with its meshes: the largest difference of a body's mass [kg], inertia [kg m^2], centre of mass [m]; the collision geometries the same":
            '0, 0, 0; yes',
    },
    'the stance': {
        "the feet's places, front and rear [m]":
            'x 0.1899, y +-0.1453; x -0.1970, y +-0.1453',
        'the declared places in quad/reduced.py [m]; the largest difference [mm]':
            'x 0.1899, y +-0.1453; x -0.1970, y +-0.1453; 0.05',
        "the feet's sink, front and rear [mm]; their loads, front and rear [N]; the loads' sum and the weight [N]":
            '12.6, 13.0; 33.5, 41.1; 149.2, 149.2',
        'the centre of mass after the first second, on joint servos [m]; at home, the base at 0.27 m [m]':
            '-0.0223, 0.0000, 0.2341; -0.0224, 0.0000, 0.2484',
        "the declared home and height in quad/reduced.py [m]; the settled centre of mass's distance from the declared home [mm]":
            '-0.0223, 0.0000; 0.2484; 0.11',
        "the rectangle's sides [m]; the triangle FL, RL, RR's incentre [m] and inradius [m]; home inside the triangle and the rectangle [mm]":
            '0.3869, 0.2906; -0.1002, 0.0485; 0.0968; 11.2, 145.3',
        "home behind the middle of the feet [mm]; the incentre's distance from home [mm]":
            '18.7; 91.7',
    },
    "the floor's softness": {
        "a free foot's sink under 1, 5, 20, 40, 80 N [mm]":
            '9.39, 9.39, 9.39, 9.39, 9.39',
    },
    'the cart-table model against MuJoCo': {
        "a shift of 55.5 mm at 0.1 m/s^2 and 0.6 s of rest: the centre of pressure's largest distance from the planned zero-moment point, and from the cart-table point of the simulated centre of mass [mm]; the forces' centre of pressure from it [mm]; the centre of mass from its plan [mm]; periods":
            '3.1, 0.5; 10.0; 2.5; 12',
        "a shift of 55.5 mm at 0.25 m/s^2 and 0.6 s of rest: the centre of pressure's largest distance from the planned zero-moment point, and from the cart-table point of the simulated centre of mass [mm]; the forces' centre of pressure from it [mm]; the centre of mass from its plan [mm]; periods":
            '6.3, 1.1; 10.0; 3.2; 9',
        "a shift of 55.5 mm at 0.5 m/s^2 and 0.6 s of rest: the centre of pressure's largest distance from the planned zero-moment point, and from the cart-table point of the simulated centre of mass [mm]; the forces' centre of pressure from it [mm]; the centre of mass from its plan [mm]; periods":
            '10.4, 1.4; 10.0; 3.5; 8',
        'the shift at 0.25 m/s^2 with a time step of 0.5 ms: the same distances [mm]':
            '6.3, 1.1; 10.0; 3.2',
    },
}


if __name__ == "__main__":
    main()
