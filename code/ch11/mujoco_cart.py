#!/usr/bin/env python3
"""
The cart of Chapter 3 in MuJoCo: the model, its record and its diagnostics.

Physical AI, Together, Chapter 11 (Section 11.1).

This script builds the simulated cart of cart_mujoco from its design and
  * writes the record of the experiment: the versions, the settings of the
    simulator and those that differ from MuJoCo's defaults, the size of the
    model and the hash of its MJCF text;
  * runs the diagnostics of Section S11.9.1, most with a variant that has the
    mistake they reveal: the wheel loads against the weight; the distance of
    the wheels from the ground along the track; the rangefinder against the
    clearance, also with the wall's rotation written in radians where the
    compiler reads degrees; one period at constant speed, also with the
    state read after mj_step; the steps of each period, also with a loop on
    MuJoCo's clock; the speed over the crest, also with the ground
    coordinate; the accelerations that the drive loop sustains on the
    simulated ground, against the labels; the contacts over the crest; a
    cart with welded wheels that slides; and the encoders at three time
    steps, also with stiff speed servos;
  * runs each of the 220 periods of the run of Chapter 3 (the forecasting
    run, through the allovi core) in MuJoCo from near the state of the exact
    run, and compares the result with the model's nominal motion, at the
    end of the period and within it, at time steps of 1, 0.5 and 0.25 ms,
    and the labels of the exact run with the braking of the simulated
    ground;
  * runs the 26 cases of Section 10.4 through the check motion_agreement;
  * runs the configuration of Chapter 3 with the simulator adapter
    cart.mujoco, and reports the braking claims of the labels, alarms and
    physical violations separately.

Usage
    python3 mujoco_cart.py            # writes results/
    python3 mujoco_cart.py --check    # also checks every value quoted in
                                      # Section 11.1

Outputs, in results/ next to this file
    record.txt                 the record of the experiment
    cart.xml                   the MJCF text that MuJoCo compiled
    agreement-chapter3.csv     one row per period of the run of Chapter 3
    mujoco-summary.txt         the values quoted in Section 11.1

Requires Python 3.10 or later, MuJoCo 3.16.0 and NumPy (requirements.txt).
Values are quoted to three significant digits unless a key says otherwise.
On another machine, MuJoCo's floating-point results may differ in the last
digits; the check reports any value that differs at the quoted precision.
"""
import argparse
import csv
import dataclasses
import hashlib
import math
import os
import platform
import statistics
import sys
from fractions import Fraction as Fr

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mujoco                                     # noqa: E402
import numpy                                      # noqa: E402

import cart_mujoco as cm                          # noqa: E402  (puts ../ch10 on the path)
import allovi                                     # noqa: E402
import cart                                       # noqa: E402
from allovi import model as am                    # noqa: E402
import model_providers                            # noqa: E402  (../ch10, Section 10.4)

T = 0.1
DESIGN = cm.Design()
G = cm.sim.G
STEPS = (0.001, 0.0005, 0.00025)                  # time steps of the convergence check [s]


def g3(x):
    """Three significant digits."""
    return f"{x:.3g}"


def ms(dt):
    return f"{1000 * dt:g} ms"


def norm(dp, dv, dl):
    """The norm of the state, max(|p| + |l|, |v| * 1 s)."""
    return max(abs(dp) + abs(dl), abs(dv))


def at_step(design, dt):
    return dataclasses.replace(design, timestep=dt)


# ----------------------------------------------------------------------------
# The record
# ----------------------------------------------------------------------------
def option_names(o):
    return {
        "time step": f"{o.timestep:g} s",
        "integrator": mujoco.mjtIntegrator(o.integrator).name[len("mjINT_"):].lower(),
        "solver": mujoco.mjtSolver(o.solver).name[len("mjSOL_"):].capitalize(),
        "friction cone": mujoco.mjtCone(o.cone).name[len("mjCONE_"):].lower(),
        "iterations": str(o.iterations),
        "tolerance": f"{o.tolerance:g}",
        "impratio": f"{o.impratio:g}",
        "noslip iterations": str(o.noslip_iterations),
        "gravity": " ".join(f"{x:g}" for x in o.gravity),
    }


def record_facts(sim):
    m, o = sim.m, sim.m.opt
    ours = option_names(o)
    default = mujoco.MjModel.from_xml_string("<mujoco/>")        # MuJoCo's defaults
    theirs = option_names(default.opt)
    differ = [k for k in ours if ours[k] != theirs[k]]
    mesh = m.mesh("ground").id
    return {
        "MuJoCo": mujoco.__version__,
        "time step [s]": g3(o.timestep),
        "integrator": ours["integrator"],
        "solver": ours["solver"],
        "friction cone": ours["friction cone"],
        "iterations, tolerance": f"{o.iterations}, {o.tolerance:g}",
        "impratio, noslip iterations": f"{o.impratio:g}, {o.noslip_iterations}",
        "gravity [m/s^2]": ours["gravity"],
        "contact solref, solimp": (" ".join(f"{x:g}" for x in m.geom_solref[0]) + "; " +
                                   " ".join(f"{x:g}" for x in m.geom_solimp[0])),
        "settings that differ from MuJoCo's defaults": ", ".join(differ),
        "MuJoCo's defaults for them": ", ".join(theirs[k] for k in differ),
        "sliding friction": g3(m.geom_friction[m.geom("front").id][0]),
        "bodies, joints, degrees of freedom": f"{m.nbody - 1}, {m.njnt}, {m.nv}",
        "geoms, actuators, sensors": f"{m.ngeom}, {m.nu}, {m.nsensor}",
        "ground mesh: vertices, facets of the crest":
            f"{m.mesh_vertnum[mesh]}, {DESIGN.crest_facets}",
        "mass [kg]": g3(sim.mass),
        "steps per period": sim.steps,
        "MJCF text, SHA-256": hashlib.sha256(sim.xml.encode()).hexdigest()[:16],
    }


def track_facts(sim):
    pr, d = sim.pr, DESIGN
    mu = d.friction
    slope = G * (mu * math.cos(pr.theta) - math.sin(pr.theta))
    # the fastest speed at 20 m from which braking at 0.5 m/s^2 stops the cart
    # by 24.5 m, as every plan certified under the labels of Chapter 3 must
    v_ramp = math.sqrt(2 * 0.5 * (24.5 - 20))
    chord = 2 * pr.R * math.sin(pr.theta / d.crest_facets / 2)
    sagitta = pr.R * (1 - math.cos(pr.theta / d.crest_facets / 2))
    edge = v_ramp ** 2 / d.wheel_radius
    return {
        "fastest speed at 20 m of a plan certified under the labels [m/s]": g3(v_ramp),
        "centripetal acceleration over a sharp edge at 2.12 m/s [m/s^2], in g":
            f"{g3(edge)}, {g3(edge / G)}",
        "centripetal acceleration of an axle over the crest at 2.12 m/s [m/s^2]":
            g3(v_ramp ** 2 / (pr.R + d.wheel_radius)),
        "centripetal acceleration of an axle over the crest at 3 m/s [m/s^2]":
            g3(3.0 ** 2 / (pr.R + d.wheel_radius)),
        "crest, ground coordinate [m]": f"{g3(pr.s0)} to {pr.s1:.4g}",
        "crest in the mesh: chords [mm], inside the circle by up to [um]":
            f"{g3(1000 * chord)}, {g3(1e6 * sagitta)}",
        "axle path beyond the crest, longer by [mm]": g3(1000 * d.wheel_radius * pr.theta),
        "wall face, ground coordinate [m]": f"{d.wall - d.wheel_radius * pr.theta:.6g}",
        "capacity, firm ground [m/s^2]": f"{G * mu:.5g}",
        "capacity, slope [m/s^2]": f"{slope:.4g}",
        "capacity, end of the crest at 2.12 m/s [m/s^2]": g3(slope - mu * v_ramp ** 2 / pr.R),
        "label 0.5 valid on the crest by the bound up to [m/s]":
            g3(math.sqrt((slope - 0.5) * pr.R / mu)),
        "penetration at rest [mm]": g3(1000 * sim.sink),
    }


# ----------------------------------------------------------------------------
# Diagnostics (Section S11.9.1)
# ----------------------------------------------------------------------------
def wheel_ground_distance(sim, p):
    """(distance, part) of each wheel from the ground's mesh, with the cart
    placed at p: the distance is negative when they overlap."""
    sim.place(p, 0.0, 0.0)
    m, d = sim.m, sim.data
    out = []
    for wheel in ("front", "rear"):
        fromto = numpy.zeros(6)
        dist = mujoco.mj_geomDistance(m, d, m.geom(wheel).id, m.geom("ground").id, 5.0, fromto)
        out.append((dist, sim.foot(wheel).part))
    return out


def drive(sim, p, v, a, duration, sample, u_tool=0.0):
    """Places the cart at (p, v) and holds the constant acceleration a with
    the drive loop for 'duration' seconds, stepping as Sim.period does;
    sample(sim, i, v_ref) is called after mj_step1 at every step."""
    sim.place(p, v, 0.0)
    m, d = sim.m, sim.data
    for i in range(int(round(duration / m.opt.timestep)) + 1):
        v_ref = v + a * i * m.opt.timestep
        mujoco.mj_step1(m, d)
        sample(sim, i, v_ref)
        sim.controls(a, v_ref, u_tool)
        mujoco.mj_step2(m, d)


def tracking(sim, p, v, a, duration=0.5):
    """The errors of the cart's speed against v + a t while the drive loop
    holds the constant acceleration a from (p, v) for 'duration' seconds:
    the largest, and its change after a transient of 0.1 s."""
    errs = []
    drive(sim, p, v, a, duration, lambda s, i, v_ref: errs.append(s.state()[1] - v_ref))
    k = int(round(0.1 / sim.m.opt.timestep))
    return max(abs(e) for e in errs), max(abs(e - errs[k]) for e in errs[k:])


def capacity(sim, p, v, sign):
    """The largest acceleration of the given sign (-1: towards the start,
    +1: towards the wall) that the drive loop sustains from (p, v) on
    uniform ground, found by bisection to 0.005 m/s^2: after 0.1 s, the
    error of the speed changes by no more than 1 mm/s in 0.4 s, which a
    deficit of 0.0025 m/s^2 would exceed."""
    lo, hi = 0.0, 5.0
    while hi - lo > 0.005:
        mid = (lo + hi) / 2
        if tracking(sim, p, v, sign * mid)[1] <= 1e-3:
            lo = mid
        else:
            hi = mid
    return lo


def wheel_contacts(sim):
    """{wheel: (penetration, normal force)} of the wheels' contacts now."""
    m, d = sim.m, sim.data
    f = numpy.zeros(6)
    out = {}
    for c in range(d.ncon):
        con = d.contact[c]
        names = {m.geom(con.geom1).name, m.geom(con.geom2).name}
        for wheel in ("front", "rear"):
            if wheel in names:
                mujoco.mj_contactForce(m, d, c, f)
                pen, load = out.get(wheel, (0.0, 0.0))
                out[wheel] = (max(pen, -con.dist), load + f[0])
    return out


def crest_contacts(sim, p, v, a, duration):
    """The contacts of the wheels while the cart crosses the crest under the
    drive loop from (p, v) at the acceleration a (a = None: no torques):
    the least penetration of a wheel on the crest, the steps in which a
    wheel has no contact, and the largest relative difference of the total
    load from M (g cos(pitch) - v^2 / (R + r)) while both wheels are on it."""
    pr, r = sim.pr, DESIGN.wheel_radius
    least, lost, worst = math.inf, 0, 0.0
    m, d = sim.m, sim.data
    sim.place(p, v, 0.0)
    for i in range(int(round(duration / m.opt.timestep))):
        v_ref = v + (a or 0.0) * i * m.opt.timestep
        mujoco.mj_step1(m, d)
        if a is None:
            d.ctrl[:] = 0.0
        else:
            sim.controls(a, v_ref, 0.0)
        mujoco.mj_step2(m, d)
        parts = {w: sim.foot(w).part for w in ("front", "rear")}
        if "crest" not in parts.values():
            continue
        con = wheel_contacts(sim)
        lost += any(w not in con for w in ("front", "rear"))
        for w in ("front", "rear"):
            if parts[w] == "crest" and w in con:
                least = min(least, con[w][0])
        if all(x == "crest" for x in parts.values()) and len(con) == 2:
            speed = sim.state()[1]
            pitch = float(d.qpos[sim.jq["pitch"]])
            expected = sim.mass * (G * math.cos(pitch) - speed ** 2 / (pr.R + r))
            total = con["front"][1] + con["rear"][1]
            worst = max(worst, abs(total - expected) / expected)
    return least, lost, worst


def sled(design, p=8.0, v=4.0, duration=0.5):
    """The cart with its wheels welded to the chassis, sliding from (p, v)
    on the firm ground: the steps in which the front wheel touches the
    ground, its largest load, and the mean deceleration over the last 0.3 s."""
    sim = cm.Sim(design, variant="welded wheels")
    sim.place(p, v, 0.0)
    m, d = sim.m, sim.data
    d.qvel[sim.jv["front"]] = d.qvel[sim.jv["rear"]] = 0.0
    mujoco.mj_forward(m, d)
    n = int(round(duration / m.opt.timestep))
    touching, loads, speeds = 0, [], []
    for _ in range(n):
        mujoco.mj_step1(m, d)
        speeds.append(sim.state()[1])
        touching += "front" in wheel_contacts(sim)
        d.ctrl[:] = 0.0
        mujoco.mj_step2(m, d)
        loads.append(float(d.sensor("front load").data[0]))
    k = int(round(0.3 / m.opt.timestep))
    return touching, n, max(loads), (speeds[n - k - 1] - speeds[n - 1]) / 0.3


def lone_box(design, v=4.0, duration=0.5):
    """A box of the cart's mass and the chassis's size sliding from v on a
    plane, with the friction and the options of the design: the steps in
    which it touches the plane, and its largest contact force against its
    weight."""
    mass = design.chassis_mass + 2 * design.wheel_mass + design.tool_mass
    xml = f"""<mujoco>
  <option timestep="{design.timestep}" integrator="{design.integrator}" solver="{design.solver}"
          cone="{design.cone}" iterations="{design.iterations}" tolerance="{design.tolerance}"/>
  <worldbody>
    <geom type="plane" size="50 5 0.1" friction="{design.friction} 0.005 0.0001"/>
    <body pos="0 0 0.08">
      <freejoint/>
      <geom type="box" size="{design.half_length} 0.25 0.08" mass="{mass}"
            friction="{design.friction} 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    for _ in range(500):                     # settle at rest
        mujoco.mj_step(m, d)
    d.qvel[0] = v
    f = numpy.zeros(6)
    n = int(round(duration / m.opt.timestep))
    touching, peak = 0, 0.0
    for _ in range(n):
        mujoco.mj_step(m, d)
        total = 0.0
        for c in range(d.ncon):
            mujoco.mj_contactForce(m, d, c, f)
            total += f[0]
        touching += d.ncon > 0
        peak = max(peak, total)
    return touching, n, peak / (mass * G)


def encoders(design, variant=None):
    """Encoder speed (the front wheel's rate times r) less the cart's speed
    after 0.5 s of accelerating at 2 m/s^2 from rest at p = 2 m [m/s]."""
    sim = cm.Sim(design, variant)
    sim.place(2.0, 0.0, 0.0)
    for _ in range(5):
        sim.period(2.0, 0.0)
    return float(sim.data.qvel[sim.jv["front"]]) * design.wheel_radius - sim.state()[1]


def diagnostics_facts(sim):
    out, d, pr = {}, sim.data, sim.pr
    # wheel loads against the weight
    sim.place(5.0, 0.0, 0.0)
    for _ in range(10):
        sim.period(0.0, 0.0)
    loads = [float(d.sensor(w + " load").data[0]) for w in ("front", "rear")]
    out["wheel loads at rest [N]"] = f"{loads[0]:.5g} + {loads[1]:.5g} = {sum(loads):.6g}"
    out["weight [N]"] = f"{sim.mass * G:.6g}"
    # geometry: the wheels on the ground along the track, also where each
    # axle reaches and leaves the crest
    crest_end = pr.s0 + (pr.R + pr.r) * pr.theta + sim.lead      # front axle leaves it
    base = 2 * DESIGN.wheel_offset
    positions = sorted({2.0, 10.0, 19.9, pr.s0 + sim.lead, 20.35, pr.s0 + sim.lead + base,
                        21.0, 21.4, crest_end, 22.0, crest_end + base, 23.0, 24.0, 24.9})
    off, on = 0.0, 0.0
    for p in positions:
        for dist, part in wheel_ground_distance(sim, p):
            if part == "crest":
                on = max(on, abs(dist + sim.sink))
            else:
                off = max(off, abs(dist + sim.sink))
    out["geometry: positions"] = str(len(positions))
    out["geometry: largest |distance + penetration| off the crest, on it [m]"] = \
        f"{g3(off)}, {g3(on)}"
    # the rangefinder against the clearance, and the wall rotated in radians
    out["wall in radians: the Euler angle written [rad]"] = f"{pr.theta:.4g}"
    for label, s in (("", sim), (", wall in radians", cm.Sim(variant="wall in radians"))):
        diffs = []
        for reach in (0.0, 0.8):
            s.place(23.7, 0.0, reach)
            p, v, l = s.state()
            diffs.append(float(s.data.sensor("range").data[0]) - (DESIGN.wall - p - l))
        out[f"rangefinder less clearance at p 23.7 m, tool folded and unfolded{label} [m]"] = \
            ", ".join(g3(x) for x in diffs)
    # timing: one period at constant speed
    for label, s in (("", sim), (", read after mj_step", cm.Sim(read="after step"))):
        s.place(5.0, 1.0, 0.0)
        p, v, l = s.period(0.0, 0.0)
        out[f"one period at 1 m/s from 5 m{label}: dp [m]"] = g3(p - 5.1)
    # timing: steps per period with a loop on MuJoCo's clock; period k
    # covers [k T, (k + 1) T]
    s = cm.Sim()
    s.place(0.0, 0.0, 0.0)
    counts = []
    for k in range(220):
        n = 0
        while s.data.time < (k + 1) * T:
            mujoco.mj_step(s.m, s.data)
            n += 1
        counts.append(n)
    out["steps per period, adapter"] = f"{sim.steps} in each of 220"
    out["steps per period, loop on the clock"] = "; ".join(
        f"{c} in period {k}" for k, c in enumerate(counts) if c != sim.steps)
    out["MuJoCo's clock after 22,000 steps [s]"] = repr(float(s.data.time))
    # frames: braking at 0.5 m/s^2 from 2.1 m/s over the crest
    errs = []
    drive(sim, 19.8, 2.1, -0.5, 1.2, lambda s, i, v_ref: errs.append(abs(s.state()[1] - v_ref)))
    out["crest, axle coordinate: largest |v - v_ref| [m/s]"] = g3(max(errs))
    # rows: the ground coordinate, the errors of the measured and the true
    # speed, and the track coordinate p
    wrong, rows = cm.Sim(coordinate="ground"), []
    drive(wrong, 19.8, 2.1, -0.5, 1.2, lambda s, i, v_ref: rows.append(
        (s.state()[0], s.state()[1] - v_ref, s.state(coordinate="axle")[1] - v_ref,
         s.state(coordinate="axle")[0])))
    first = next(i for i, r in enumerate(rows) if r[0] >= pr.s0 + sim.lead)
    jump = rows[first][1] - rows[first - 1][1]
    over = [r for r in rows if r[2] > 0.015]
    back = next(i for i in range(first, len(rows)) if abs(rows[i][1]) < 1e-3) - first
    out["crest, ground coordinate: jump of the measured rate on the crest [m/s], at p [m]"] = \
        f"{g3(jump)}, {rows[first][3]:.3g}"
    out["crest, ground coordinate: steps until the measured rate is within 1 mm/s of v_ref"] = \
        str(back)
    out["crest, ground coordinate: true speed above v_ref, largest [m/s]"] = \
        g3(max(r[2] for r in rows))
    out["crest, ground coordinate: true speed more than 15 mm/s above v_ref, for p in [m]"] = \
        f"{over[0][3]:.4g} to {over[-1][3]:.4g}"
    # the accelerations that the drive loop sustains on the simulated ground
    out["capacity: braking on firm ground from 4 m/s [m/s^2]"] = g3(capacity(sim, 8.0, 4.0, -1))
    out["capacity: braking on the slope from 2 m/s [m/s^2]"] = g3(capacity(sim, 22.5, 2.0, -1))
    out["capacity: from rest towards the start on the slope [m/s^2]"] = \
        g3(capacity(sim, 24.0, 0.0, -1))
    out["capacity: from rest towards the wall on firm ground [m/s^2]"] = \
        g3(capacity(sim, 2.0, 0.0, 1))
    # braking at 2 m/s^2 on firm ground from 4 m/s; at 0.5 m/s^2 over the end
    # of the crest from 2.12 m/s; 2 m/s^2 towards the wall from rest
    for key, case in (("label 2 on firm ground", (8.0, 4.0, -2.0)),
                      ("label 0.5 over the end of the crest", (21.6, 2.12, -0.5)),
                      ("command bound towards the wall", (2.0, 0.0, 2.0))):
        worst, drift = tracking(sim, *case)
        out[f"{key}: speed error, largest and change after 0.1 s [mm/s]"] = \
            f"{g3(1000 * worst)}, {g3(1000 * drift)}"
    # the contacts over the crest: braking at the label from 2.12 m/s, and
    # rolling freely from 3 m/s
    least, lost, worst = crest_contacts(sim, 20.0, 2.12, -0.5, 1.4)
    out["crest, braking at 0.5 from 2.12 m/s at 20 m: least penetration [mm], steps without "
        "contact"] = f"{g3(1000 * least)}, {lost}"
    out["crest, braking at 0.5 from 2.12 m/s at 20 m: largest load difference from the rigid "
        "cart's"] = f"{g3(100 * worst)} %"
    for v in (2.3, 2.4, 2.5):
        least, lost, worst = crest_contacts(sim, 19.9, v, 0.0, (22.6 - 19.9) / v)
        out[f"crest at a constant {v:g} m/s: least penetration [mm], steps without contact"] = \
            f"{g3(1000 * least)}, {lost}"
    least, lost, worst = crest_contacts(sim, 19.5, 3.0, None, 1.0)
    out["crest, rolling freely from 3 m/s: least penetration [mm], steps without contact"] = \
        f"{g3(1000 * least)}, {lost}"
    # the reference acceleration of the wheels' contacts at rest, and the speed
    # at which the crest's bias v^2 / (R + r) equals it
    sim.place(5.0, 0.0, 0.0)
    for _ in range(10):
        sim.period(0.0, 0.0)
    mujoco.mj_forward(sim.m, d)
    aref = [float(d.efc_aref[d.contact[c].efc_address]) for c in range(d.ncon)]
    out["reference acceleration of each wheel's contact at rest [m/s^2]"] = \
        ", ".join(g3(a) for a in aref)
    out["speed at which v^2 / (R + r) equals it [m/s]"] = \
        g3(math.sqrt(min(aref) * (pr.R + DESIGN.wheel_radius)))
    touching, n, peak = lone_box(DESIGN)
    out["a box of the cart's mass sliding from 4 m/s on a plane: steps touching it, largest "
        "load / weight"] = f"{touching} of {n}, {peak:.3g}"
    # a cart that slides: the wheels welded to the chassis
    for dt in STEPS:
        touching, n, load, decel = sled(at_step(DESIGN, dt))
        if dt == STEPS[0]:
            out["sled from 4 m/s: steps with the front wheel on the ground"] = f"{touching} of {n}"
            out["sled: largest front load [N], mean deceleration [m/s^2]"] = \
                f"{load:.3g}, {decel:.3g}"
        else:
            out[f"sled at {ms(dt)}: share of steps with the front wheel on the ground"] = \
                f"{100 * touching / n:.0f} %"
    # the encoders at three time steps, and with stiff speed servos
    for variant, label in ((None, "drive loop"), ("speed servos", "speed servos")):
        out[f"encoder less cart speed at 2 m/s^2, {label}, at 1, 0.5, 0.25 ms [mm/s]"] = ", ".join(
                g3(1000 * encoders(at_step(DESIGN, dt), variant)) for dt in STEPS)
    out["encoder less cart speed at 2 m/s^2, speed servos, Euler at 1 ms [mm/s]"] = \
        g3(1000 * encoders(dataclasses.replace(DESIGN, integrator="Euler"), "speed servos"))
    return out


# ----------------------------------------------------------------------------
# The run of Chapter 3, period by period, in MuJoCo
# ----------------------------------------------------------------------------
def chapter3_periods():
    """(k, state, law, label, segment) of the run of Chapter 3 with the
    exact track (the configuration 'chapter3' of Chapter 10)."""
    cfg = cart.configuration("chapter3", postpone=None)
    rec = allovi.run(cfg, 220, strict=True)
    out = []
    for per in rec.periods:
        segment, label = cfg.control.head(per.plan)
        out.append((per.k, per.state, cfg.control.law(per.plan), label, segment))
    return out


PERIODS = None


def agreement(design, only=None):
    """Each period of the run of Chapter 3 in MuJoCo (or those in 'only').
    The cart is placed 0.05 s before the start of the period, on the motion
    of the previous period, and driven under the previous law until the
    period starts, so that its contacts carry the load of that law, as they
    would in a run; the period's law is then executed, and the result
    compared with the nominal motion of the model from the state at which
    the period began. The realized acceleration less the commanded one is
    the disturbance of Section 5.1 that would explain the difference; it is
    read from the adapter's speed signal (w) and from the speed of the
    front axle along the ground (w_axle), over windows of 1 ms and over
    single steps. Where the curvature of the ground jumps, the two signals
    differ by a jump of the coordinate (jump)."""
    global PERIODS
    if PERIODS is None:
        PERIODS = chapter3_periods()
    world = cm.CartMuJoCo(design)
    sim = world.sim
    rows = []
    pre = int(round(0.05 / design.timestep))
    win = int(round(0.001 / design.timestep))
    previous = (0.0, 0.0, "segment")
    for k, x, law, label, segment in PERIODS:
        p, v, l = (float(a) for a in x)
        u_prev, ell_prev, kind_prev = previous
        u = float(law.u)
        ell = l if law.ell is None else float(law.ell)
        previous = (u, ell, law.kind)
        if only is not None and k not in only:
            continue
        dt = pre * design.timestep
        sim.place(p - v * dt + u_prev * dt * dt / 2, v - u_prev * dt, l)
        sim.period(u_prev, ell_prev, kind_prev, steps=pre)
        p0, v0, l0 = sim.state()
        rec = []
        p1, v1, l1 = sim.period(u, ell, law.kind, record=rec)
        last = rec[-1]                # the end of the period, so that the
        rec.append(sim.sample(last["a_ref"],      # records cover its last step
                              last["v_ref"] + last["a_ref"] * design.timestep))
        np_, nv, nl = cart.model.MODEL.dynamics((p0, v0, l0), (u, ell), T)
        dp, dv, dl = p1 - np_, v1 - nv, l1 - nl
        within_p = within_v = 0.0
        low = high = 0.0
        t0 = rec[0]["t"]
        for r in rec:                 # the nominal motion of the segment
            tau = r["t"] - t0
            pn, vn = p0 + v0 * tau + u * tau * tau / 2, v0 + u * tau
            within_p = max(within_p, abs(r["p"] - pn))
            within_v = max(within_v, abs(r["v"] - vn))
            low, high = min(low, r["v"] - vn), max(high, r["v"] - vn)
        w, w_at = disturbance(rec, "v", win)
        w_step = disturbance(rec, "v", 1)[0]
        w_axle, w_axle_at = disturbance(rec, "axle", win)
        w_axle_step = disturbance(rec, "axle", 1)[0]
        jump, jump_at = 0.0, p0
        for a, b in zip(rec, rec[1:]):
            dev = abs((b["v"] - a["v"]) - (b["axle"] - a["axle"]))
            if dev > jump:
                jump, jump_at = dev, b["p"]
        speed = float(max(abs(segment.v0), abs(segment.v1)))
        cap = world.capacity(float(segment.lo), float(segment.hi), speed)
        rows.append(dict(k=k, t=k * T, p=p, v=v, l=l, u=u, ell=ell, label=float(label),
                         capacity=cap, start=norm(p0 - p, v0 - v, l0 - l),
                         start_pl=abs(p0 - p) + abs(l0 - l), start_v=abs(v0 - v), dp=dp, dv=dv,
                         dl=dl, end=norm(dp, dv, dl), within_p=within_p, within_v=within_v,
                         low=low, high=high, lmax=max(max(r["l"] for r in rec), l1),
                         w=w, w_at=w_at, w_step=w_step, w_axle=w_axle, w_axle_at=w_axle_at,
                         w_axle_step=w_axle_step, jump=jump, jump_at=jump_at))
    return rows


def ranges(ks):
    """'57 to 66' for a run of consecutive numbers, joined by commas."""
    out, ks = [], sorted(ks)
    i = 0
    while i < len(ks):
        j = i
        while j + 1 < len(ks) and ks[j + 1] == ks[j] + 1:
            j += 1
        out.append(f"{ks[i]}" if i == j else f"{ks[i]} to {ks[j]}")
        i = j + 1
    return ", ".join(out)


def disturbance(rec, key, win):
    """The largest |realized acceleration - commanded acceleration| over
    windows of 'win' steps, the speed read from rec[i][key], and the
    position at which it occurs."""
    worst, at = 0.0, rec[0]["p"]
    for i in range(len(rec) - win):
        a, b = rec[i], rec[i + win]
        commanded = sum(r["a_ref"] for r in rec[i:i + win]) / win
        dev = abs((b[key] - a[key]) / (b["t"] - a["t"]) - commanded)
        if dev > worst:
            worst, at = dev, a["p"]
    return worst, at


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        cols = list(rows[0])
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] if c == "k" else f"{r[c]:.9g}" for c in cols])


def summary(rows):
    """The values of a comparison that the section quotes."""
    ends = sorted(r["end"] for r in rows)
    worst = max(rows, key=lambda r: r["end"])
    worst_v = max(rows, key=lambda r: r["within_v"])
    worst_v2 = max((r for r in rows if r["k"] != worst_v["k"]), key=lambda r: r["within_v"])
    folded = [r for r in rows if r["l"] == 0 and r["ell"] == 0]
    p90 = ends[math.ceil(0.9 * len(ends)) - 1]   # nearest rank
    above = [r for r in rows if r["end"] > p90]
    return dict(median=statistics.median(ends), p90=p90, worst=worst,
                above_p90=above, braking_above=[r for r in above if r["u"] == -2.0],
                worst_v=worst_v, worst_v2=worst_v2, folded=folded,
                folded_end=max(folded, key=lambda r: r["end"]),
                folded_v=max(folded, key=lambda r: r["within_v"]),
                folded_dp=max(folded, key=lambda r: abs(r["dp"])),
                folded_p=max(folded, key=lambda r: r["within_p"]),
                accelerating=[r for r in folded if r["u"] == 2.0 and r["v"] >= 0],
                above=[r["k"] for r in folded if r["w"] > 0.05],
                wmax=max(folded, key=lambda r: r["w"]),
                above_step=[r["k"] for r in folded if r["w_step"] > 0.05],
                wmax_step=max(folded, key=lambda r: r["w_step"]),
                above_axle=[r["k"] for r in folded
                            if max(r["w_axle"], r["w_axle_step"]) > 0.05],
                waxle=max(folded, key=lambda r: r["w_axle"]),
                waxle_step=max(folded, key=lambda r: r["w_axle_step"]),
                jump=max(rows, key=lambda r: r["jump"]),
                accel=next(r for r in rows if r["k"] == 1),
                brake=next(r for r in rows if r["k"] == 40))


def agreement_facts(out_dir):
    rows = agreement(DESIGN)
    write_csv(os.path.join(out_dir, "agreement-chapter3.csv"), rows)
    s = summary(rows)
    worst, unfold = s["worst"], next(r for r in rows if r["l"] == 0 and r["ell"] > 0)
    rest = [r for r in rows if r["l"] == 0 and r["p"] == 24.5 and r["u"] == 0]
    margin = min(rows, key=lambda r: r["capacity"] - r["label"])
    lmax = max(rows, key=lambda r: r["lmax"])
    sag = DESIGN.tool_mass * G * math.sin(math.radians(DESIGN.slope_deg)) / DESIGN.tool_kp
    return {
        "periods": len(rows),
        "end of period: median, 90th percentile (nearest rank) [m]":
            f"{g3(s['median'])}, {g3(s['p90'])}",
        "end of period, above the 90th percentile: braking at 2 m/s^2 (count, from, to [m]); "
        "the others (period: difference [m])":
            f"{len(s['braking_above'])}, {g3(min(r['end'] for r in s['braking_above']))}, "
            f"{g3(max(r['end'] for r in s['braking_above']))}; " + ", ".join(
                f"{r['k']}: {g3(r['end'])}" for r in s["above_p90"] if r["u"] != -2.0),
        "end of period: largest [m], period, t [s]": f"{g3(worst['end'])}, {worst['k']}, "
                                                     f"{worst['t']:.3g}",
        "end of period: periods in which the speed sets the norm": ranges(
            r["k"] for r in rows if abs(r["dv"]) > abs(r["dp"]) + abs(r["dl"])),
        "end of period, largest: dp, dv, dl": ", ".join(g3(worst[c]) for c in ("dp", "dv", "dl")),
        "periods at rest at 24.5 m: count, median dl [m]":
            f"{len(rest)}, {g3(statistics.median(r['dl'] for r in rest))}",
        "periods at rest at 24.5 m: end differences from, to [m]":
            f"{g3(min(r['end'] for r in rest))}, {g3(max(r['end'] for r in rest))}",
        "folded tool on the slope: m g sin(theta) / kp [m]": g3(sag),
        "braking at 2 m/s^2 (period 40): dp, dl [m]":
            f"{g3(s['brake']['dp'])}, {g3(s['brake']['dl'])}",
        "braking at 2 m/s^2 (period 40): end difference [m]": g3(s["brake"]["end"]),
        "folded tool, braking at 2 m/s^2: m a / kp [m]":
            g3(DESIGN.tool_mass * 2.0 / DESIGN.tool_kp),
        "accelerating at 2 m/s^2 from rest (period 1): dp, dl [m]; a h T / 2 [m]":
            f"{g3(s['accel']['dp'])}, {g3(s['accel']['dl'])}; {g3(2.0 * DESIGN.timestep * T / 2)}",
        "periods accelerating towards the wall at 2 m/s^2: count, end differences from, to [m]":
            f"{len(s['accelerating'])}, {g3(min(r['end'] for r in s['accelerating']))}, "
            f"{g3(max(r['end'] for r in s['accelerating']))}",
        "tool folded: largest end difference [m], period":
            f"{g3(s['folded_end']['end'])}, {s['folded_end']['k']}",
        "tool folded: largest |dp| at the end of a period [m], period":
            f"{g3(abs(s['folded_dp']['dp']))}, {s['folded_dp']['k']}",
        "tool folded: largest position difference within a period [m], period":
            f"{g3(s['folded_p']['within_p'])}, {s['folded_p']['k']}",
        "tool folded: largest speed difference within a period [m/s], period":
            f"{g3(s['folded_v']['within_v'])}, {s['folded_v']['k']}",
        "within period: largest speed difference [m/s], period":
            f"{g3(s['worst_v']['within_v'])}, {s['worst_v']['k']}",
        "within period: largest speed difference elsewhere [m/s], period":
            f"{g3(s['worst_v2']['within_v'])}, {s['worst_v2']['k']}",
        "within period: largest position difference [m], period":
            f"{g3(max(r['within_p'] for r in rows))}, "
            f"{max(rows, key=lambda r: r['within_p'])['k']}",
        "tool unfolds: period, speed difference within it, lowest and highest [m/s]":
            f"{unfold['k']}, {g3(unfold['low'])}, {g3(unfold['high'])}",
        "tool unfolds: largest reach within the period [m], difference at its end [m]":
            f"{unfold['lmax']:.5f}, {g3(unfold['dl'])}",
        "largest reach in any period [m], period": f"{lmax['lmax']:.5f}, {lmax['k']}",
        "start of a period: largest distance from the exact state [m]":
            g3(max(r["start"] for r in rows)),
        "start of a period: largest |dp| + |dl| [m], period; largest |dv| [m/s], period": (
            f"{g3(max(r['start_pl'] for r in rows))}, "
            f"{max(rows, key=lambda r: r['start_pl'])['k']}; "
            f"{g3(max(r['start_v'] for r in rows))}, "
            f"{max(rows, key=lambda r: r['start_v'])['k']}"),
        "speed signal: acceleration less command over 1 ms, tool folded: periods above 0.05":
            f"{len(s['above'])} of {len(s['folded'])}",
        "speed signal: acceleration less command over 1 ms, tool folded: largest, period, at p":
            f"{g3(s['wmax']['w'])}, {s['wmax']['k']}, {s['wmax']['w_at']:.4g}",
        "axle: acceleration less command, tool folded: periods above 0.05 m/s^2":
            f"{len(s['above_axle'])} of {len(s['folded'])}",
        "axle: acceleration less command, tool folded, over 1 ms: largest, period, at p":
            f"{g3(s['waxle']['w_axle'])}, {s['waxle']['k']}, {s['waxle']['w_axle_at']:.4g}",
        "axle: acceleration less command, tool folded, over one step: largest, period":
            f"{g3(s['waxle_step']['w_axle_step'])}, {s['waxle_step']['k']}",
        "jump of the speed signal against the axle in one step: largest [m/s], period, at p":
            f"{g3(s['jump']['jump'])}, {s['jump']['k']}, {s['jump']['jump_at']:.4g}",
        "acceleration less command over 1 ms, tool unfolding [m/s^2]": g3(unfold["w"]),
        "braking claims of the exact run's labels: available on the simulated ground":
            f"{sum(r['capacity'] >= r['label'] for r in rows)} of {len(rows)}",
        "smallest margin of a label [m/s^2], period, label": (
            f"{g3(margin['capacity'] - margin['label'])}, {margin['k']}, {margin['label']:g}"),
    }


def timestep_facts():
    out = {}
    for dt in STEPS[1:]:
        rows = agreement(at_step(DESIGN, dt))
        s = summary(rows)
        unfold = next(r for r in rows if r["l"] == 0 and r["ell"] > 0)
        out[f"{ms(dt)}: end of period, median, 90th percentile (nearest rank) [m]"] = \
            f"{g3(s['median'])}, {g3(s['p90'])}"
        out[f"{ms(dt)}: braking at 2 m/s^2 (period 40): end difference [m]"] = \
            g3(s["brake"]["end"])
        out[f"{ms(dt)}: end of period, largest [m], period"] = \
            f"{g3(s['worst']['end'])}, {s['worst']['k']}"
        out[f"{ms(dt)}: tool unfolds, difference of the reach at the end [m]"] = g3(unfold["dl"])
        out[f"{ms(dt)}: within period, largest speed difference outside it [m/s], period"] = \
            f"{g3(s['worst_v2']['within_v'])}, {s['worst_v2']['k']}"
        out[f"{ms(dt)}: acceleration less command over 1 ms, tool folded, above 0.05 m/s^2"] = \
            f"{len(s['above'])}; largest {g3(s['wmax']['w'])} in period {s['wmax']['k']}"
        out[f"{ms(dt)}: acceleration less command over one step, tool folded, above 0.05 m/s^2"] \
            = (f"{len(s['above_step'])}; largest {g3(s['wmax_step']['w_step'])} in period "
               f"{s['wmax_step']['k']}")
        out[f"{ms(dt)}: accelerating at 2 m/s^2 from rest (period 1): dp [m]"] = \
            g3(s["accel"]["dp"])
        out[f"{ms(dt)}: axle: acceleration less command, tool folded, over 1 ms and one step"] = \
            f"{g3(s['waxle']['w_axle'])}, {g3(s['waxle_step']['w_axle_step'])}"
        out[f"{ms(dt)}: jump of the speed signal against the axle in one step [m/s], period"] = \
            f"{g3(s['jump']['jump'])}, {s['jump']['k']}"
    # the drive loop at the crest: its speed differences against its gain
    crest = range(56, 68)
    for gain in (100.0, 200.0):
        rows = agreement(dataclasses.replace(DESIGN, speed_gain=gain), only=crest)
        out[f"speed gain {gain:g} 1/s: largest speed difference within periods 56 to 67 [m/s]"] = \
            g3(max(r["within_v"] for r in rows))
    # the tool's last approach (period 201): smaller steps, and the Euler integrator
    vals = []
    for dt in (0.000125, 0.0000625):
        vals.append(g3(-agreement(at_step(DESIGN, dt), only={201})[0]["dl"]))
    out["tool unfolds, reach short of the stop, implicitfast at 0.125, 0.0625 ms [m]"] = \
        ", ".join(vals)
    vals = []
    for dt in (0.001, 0.0005, 0.00025, 0.000125):
        euler = dataclasses.replace(DESIGN, timestep=dt, integrator="Euler")
        vals.append(g3(-agreement(euler, only={201})[0]["dl"]))
    out["tool unfolds, reach short of the stop, Euler at 1, 0.5, 0.25, 0.125 ms [m]"] = \
        ", ".join(vals)
    return out


def motion_cases_facts():
    world = cm.CartMuJoCo()
    cases = model_providers.motion_cases()
    out = {}
    worst, which = 0.0, None
    big = []
    for state, inputs, law in cases:
        theirs = world.step(state, law)
        mine = cart.model.MODEL.dynamics(state, inputs, cart.T)
        diffs = [abs(float(a - b)) for a, b in zip(mine, theirs)]
        for name, dd in zip(("p", "v", "l"), diffs):
            if dd > worst:
                worst, which = dd, name
        if max(diffs) > 1e-3:
            big.append(f"from ({', '.join(f'{float(a):g}' for a in state)}) under "
                       f"({', '.join(f'{float(a):g}' for a in inputs)})")
    for tol in (0, Fr(1, 1000)):
        diffs, n = am.motion_agreement(cart.model.MODEL, world, cases, cart.T, tolerance=tol)
        out[f"differences with tolerance {float(tol):g}"] = f"{len(diffs)} of {n}"
    out["largest difference of a component"] = g3(worst)
    out["largest difference, component"] = which
    out["cases with a difference above 1 mm"] = "; ".join(big) if big else "none"
    return out


# ----------------------------------------------------------------------------
# The configuration of Chapter 3 in MuJoCo
# ----------------------------------------------------------------------------
def closed_loop_facts():
    cfg = cart.configuration("chapter3, MuJoCo", postpone=None, world=cm.CartMuJoCo.NAME)
    out = {"declarations of robot, world and control vs model: problems":
           len(am.agreement(cart.model.MODEL, cfg.robot, cfg.world, cfg.control))}
    rec = allovi.run(cfg, 220, strict=True)
    first = rec.alarms[0] if rec.alarms else None
    out["alarms"] = len(rec.alarms)
    out["first alarm [s]"] = "none" if first is None else g3(first * T)
    if first is not None:
        before, after = rec.periods[first - 1], rec.periods[first]
        dist = cfg.control.distance(after.measured, before.initial_set)
        out["distance that raised it, in the norm of the state"] = g3(float(dist))
        out["measured state at the alarm"] = ", ".join(g3(float(a)) for a in after.measured)
        out["set that the rest feedback can reach, after the alarm"] = (
            "none" if after.reach is None else "computed")
    out["restarts"] = len(rec.restarts)
    states = [p.state for p in rec.periods] + [rec.final_state]
    out["least clearance [m]"] = g3(min(float(cfg.world.clearance(s)) for s in states))
    out["largest |p| over the run [m]"] = g3(max(abs(float(s[0])) for s in states))
    out["first period in the work zone"] = next(
        (str(i) for i, s in enumerate(states) if s[0] >= 24), "none")
    m = cfg.evaluator.evaluate(rec)
    out["cart.metrics: tool unfolds at [s]"] = "none" if m["tool_unfolds"] is None \
        else g3(float(m["tool_unfolds"]))
    out["largest reach over the run [m]"] = g3(max(float(s[2]) for s in states))
    out["tool operation"] = ("dropped after an overload at "
                             f"{g3(cfg.task.overloads[0] * T)} s") if cfg.task.dropped else "done"
    return out


# ----------------------------------------------------------------------------
def hardware():
    """The processor, where the platform says it."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "not reported"


def write_record(path, facts):
    """The record of the experiment, with the items that the book asks of
    every supported experiment (Section S11.9.2)."""
    rec, loop = facts["record"], facts["the configuration of Chapter 3"]
    d = DESIGN
    differ = rec["settings that differ from MuJoCo's defaults"]
    defaults = rec["MuJoCo's defaults for them"]
    lines = [
        "Record of the experiment of Section 11.1: the configuration of Chapter 3 "
        "with the simulator adapter cart.mujoco",
        "",
        "model and asset provenance",
        "  generated by cart_mujoco/scene.py from the design below (cart.xml); the "
        "ground is a convex mesh generated in the same text; no external assets; MJCF "
        "text SHA-256 " + rec["MJCF text, SHA-256"] + "...",
        "  design (SI units): " + ", ".join(f"{k} = {v}" for k, v in d.as_dict().items()),
        "simulator and solver",
        f"  MuJoCo {mujoco.__version__}, NumPy {numpy.__version__}, Python "
        f"{platform.python_version()}",
        "  " + "; ".join(f"{k} = {rec[k]}" for k in (
            "integrator", "solver", "friction cone", "iterations, tolerance",
            "impratio, noslip iterations", "contact solref, solimp", "sliding friction",
            "gravity [m/s^2]")),
        f"  settings that differ from MuJoCo's defaults: {differ} (defaults: {defaults})",
        "time steps",
        f"  simulator {rec['time step [s]']} s; governor period 0.1 s = "
        f"{rec['steps per period']} steps, counted; drive loop at every step; the "
        "comparison of Section 11.1.4 repeated at 0.5 and 0.25 ms",
        "actuators and sensors",
        f"  each wheel: torque motor, limit {d.drive_limit:g} N m; drive loop: "
        f"feedforward and speed feedback (gain {d.speed_gain:g} 1/s), shared by measured "
        f"load; tool: position servo kp {d.tool_kp:g} N/m, kv {d.tool_kv:g} N s/m, limit "
        f"{d.tool_force:g} N; standstill below 5 mm/s",
        "  sensors: wheel angles and rates, tool reach, bumper position and velocity, "
        "accelerometer, rangefinder at the tool tip, wheel loads, actuator efforts",
        "frames and units",
        "  SI units; simulator frame x towards the wall, z up; state (p, v, l): p the "
        "arc length of the curve at height r above the ground, to below the front axle, "
        "plus 0.1 m; v its rate; l the tool slide",
        "uncertainty bounds",
        "  labels of cart.forecasting: braking of at least 2 m/s^2 on regions before "
        "20 m, 0.5 m/s^2 elsewhere; measurement bound 0 (the simulator's state)",
        "constraints",
        "  p + l <= 25 m (the tool tip touches the wall at p + l = 25 m); |u_a| <= 2 "
        "m/s^2; u_l in {0, 0.8} m; in the simulator: contacts, the tool's stroke 0 to 0.8 m",
        "terminal behaviour",
        "  rest feedback after an alarm: braking at its command until at rest, then "
        "held; a cart at rest or moving back is held at once; restart if certified",
        "training and evaluation split",
        "  none: no module of this configuration learns",
        "seeds",
        "  none: the configuration and the simulator use no random numbers",
        "hardware",
        f"  {platform.system()} {platform.machine()}, {hardware()}",
        "guarantee status",
        "  none in this simulator: the funnels of cart.constant-command are single "
        "motions and the measurement bound is 0, while the simulated cart "
        "deviates from them; first alarm at "
        f"{loop.get('first alarm [s]')} s, raised by a distance of "
        f"{loop.get('distance that raised it, in the norm of the state')}; restarts: "
        f"{loop.get('restarts')}; least clearance {loop.get('least clearance [m]')} m",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Section 11.1: the cart in MuJoCo")
    ap.add_argument("--check", action="store_true",
                    help="check every value quoted in Section 11.1")
    args = ap.parse_args()
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    sim = cm.Sim()
    with open(os.path.join(out_dir, "cart.xml"), "w", encoding="utf-8") as fh:
        fh.write(sim.xml)
    facts = {"record": record_facts(sim),
             "the track": track_facts(sim),
             "diagnostics": diagnostics_facts(cm.Sim()),
             "the run of Chapter 3, period by period": agreement_facts(out_dir),
             "the same at smaller time steps": timestep_facts(),
             "the cases of Section 10.4": motion_cases_facts(),
             "the configuration of Chapter 3": closed_loop_facts()}
    write_record(os.path.join(out_dir, "record.txt"), facts)
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
        lines.append(f"[check] {n} values quoted in Section 11.1: " +
                     ("all as quoted" if not failures else "FAILED: " + "; ".join(failures)))
    with open(os.path.join(out_dir, "mujoco-summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if failures:
        raise SystemExit(1)


EXPECTED = {   # every value quoted in Section 11.1
    'record': {
        'MuJoCo': '3.16.0',
        'time step [s]': '0.001',
        'integrator': 'implicitfast',
        'solver': 'Newton',
        'friction cone': 'elliptic',
        'iterations, tolerance': '100, 1e-08',
        'impratio, noslip iterations': '1, 0',
        'gravity [m/s^2]': '0 0 -9.81',
        'contact solref, solimp': '0.02 1; 0.9 0.95 0.001 0.5 2',
        "settings that differ from MuJoCo's defaults": 'time step, integrator, friction cone',
        "MuJoCo's defaults for them": '0.002 s, euler, pyramidal',
        'sliding friction': '0.25',
        'bodies, joints, degrees of freedom': '4, 6, 6',
        'geoms, actuators, sensors': '6, 3, 14',
        'ground mesh: vertices, facets of the crest': '342, 166',
        'mass [kg]': '90.1',
        'steps per period': '100',
        'MJCF text, SHA-256': 'ed63fb3f13056da6',
    },
    'the track': {
        'fastest speed at 20 m of a plan certified under the labels [m/s]': '2.12',
        'centripetal acceleration over a sharp edge at 2.12 m/s [m/s^2], in g': '45, 4.59',
        'centripetal acceleration of an axle over the crest at 2.12 m/s [m/s^2]': '0.446',
        'centripetal acceleration of an axle over the crest at 3 m/s [m/s^2]': '0.891',
        'crest, ground coordinate [m]': '20 to 21.66',
        'crest in the mesh: chords [mm], inside the circle by up to [um]': '9.99, 1.25',
        'axle path beyond the crest, longer by [mm]': '16.6',
        'wall face, ground coordinate [m]': '24.9834',
        'capacity, firm ground [m/s^2]': '2.4525',
        'capacity, slope [m/s^2]': '0.7997',
        'capacity, end of the crest at 2.12 m/s [m/s^2]': '0.687',
        'label 0.5 valid on the crest by the bound up to [m/s]': '3.46',
        'penetration at rest [mm]': '0.233',
    },
    'diagnostics': {
        'wheel loads at rest [N]': '442.53 + 441.35 = 883.881',
        'weight [N]': '883.881',
        'geometry: positions': '14',
        'geometry: largest |distance + penetration| off the crest, on it [m]':
            '5.34e-08, 1.29e-06',
        'wall in radians: the Euler angle written [rad]': '0.1658',
        'rangefinder less clearance at p 23.7 m, tool folded and unfolded [m]':
            '-1.9e-11, -1.9e-11',
        'rangefinder less clearance at p 23.7 m, tool folded and unfolded, wall in radians [m]':
            '0.0569, 0.0569',
        'one period at 1 m/s from 5 m: dp [m]': '-5.19e-07',
        'one period at 1 m/s from 5 m, read after mj_step: dp [m]': '-0.001',
        'steps per period, adapter': '100 in each of 220',
        'steps per period, loop on the clock':
            '101 in period 10; 99 in period 49; 101 in period 98; 99 in period 188',
        "MuJoCo's clock after 22,000 steps [s]": '22.000000000003904',
        'crest, axle coordinate: largest |v - v_ref| [m/s]': '0.000364',
        'crest, ground coordinate: jump of the measured rate on the crest [m/s], at p [m]':
            '-0.02, 20.1',
        'crest, ground coordinate: steps until the measured rate is within 1 mm/s of v_ref':
            '32',
        'crest, ground coordinate: true speed above v_ref, largest [m/s]': '0.0201',
        'crest, ground coordinate: true speed more than 15 mm/s above v_ref, for p in [m]':
            '20.13 to 21.78',
        'capacity: braking on firm ground from 4 m/s [m/s^2]': '2.45',
        'capacity: braking on the slope from 2 m/s [m/s^2]': '0.796',
        'capacity: from rest towards the start on the slope [m/s^2]': '0.796',
        'capacity: from rest towards the wall on firm ground [m/s^2]': '2.45',
        'label 2 on firm ground: speed error, largest and change after 0.1 s [mm/s]':
            '0.122, 0.00466',
        'label 0.5 over the end of the crest: speed error, largest and change after 0.1 s [mm/s]':
            '0.381, 0.512',
        'command bound towards the wall: speed error, largest and change after 0.1 s [mm/s]':
            '0.0381, 0.00463',
        'crest, braking at 0.5 from 2.12 m/s at 20 m: least penetration [mm], steps without contact':
            '0.0845, 0',
        "crest, braking at 0.5 from 2.12 m/s at 20 m: largest load difference from the rigid cart's":
            '1.96 %',
        'crest at a constant 2.3 m/s: least penetration [mm], steps without contact':
            '0.0111, 0',
        'crest at a constant 2.4 m/s: least penetration [mm], steps without contact':
            '1.87e-06, 3',
        'crest at a constant 2.5 m/s: least penetration [mm], steps without contact':
            '0.000115, 9',
        'crest, rolling freely from 3 m/s: least penetration [mm], steps without contact':
            '3.53e-05, 39',
        "reference acceleration of each wheel's contact at rest [m/s^2]": '0.584, 0.584',
        'speed at which v^2 / (R + r) equals it [m/s]': '2.43',
        "a box of the cart's mass sliding from 4 m/s on a plane: steps touching it, largest load / weight":
            '150 of 500, 12.2',
        'sled from 4 m/s: steps with the front wheel on the ground': '63 of 500',
        'sled: largest front load [N], mean deceleration [m/s^2]': '5.02e+03, 2.46',
        'sled at 0.5 ms: share of steps with the front wheel on the ground': '13 %',
        'sled at 0.25 ms: share of steps with the front wheel on the ground': '13 %',
        'encoder less cart speed at 2 m/s^2, drive loop, at 1, 0.5, 0.25 ms [mm/s]':
            '2.05, 2.05, 2.05',
        'encoder less cart speed at 2 m/s^2, speed servos, at 1, 0.5, 0.25 ms [mm/s]':
            '-40.9, -19.4, -8.68',
        'encoder less cart speed at 2 m/s^2, speed servos, Euler at 1 ms [mm/s]': '2.06',
    },
    'the run of Chapter 3, period by period': {
        'periods': '220',
        'end of period: median, 90th percentile (nearest rank) [m]': '0.000163, 0.00026',
        'end of period, above the 90th percentile: braking at 2 m/s^2 (count, from, to [m]); the others (period: difference [m])':
            '20, 0.000302, 0.000303; 57: 0.000338, 201: 0.000388',
        'end of period: largest [m], period, t [s]': '0.000388, 201, 20.1',
        'end of period: periods in which the speed sets the norm': '0, 57 to 66, 201',
        'end of period, largest: dp, dv, dl': '-5.51e-06, 0.000388, -0.000205',
        'periods at rest at 24.5 m: count, median dl [m]': '80, 0.000162',
        'periods at rest at 24.5 m: end differences from, to [m]': '0.000163, 0.000163',
        'folded tool on the slope: m g sin(theta) / kp [m]': '0.000162',
        'braking at 2 m/s^2 (period 40): dp, dl [m]': '-0.000102, 0.0002',
        'braking at 2 m/s^2 (period 40): end difference [m]': '0.000303',
        'folded tool, braking at 2 m/s^2: m a / kp [m]': '0.0002',
        'accelerating at 2 m/s^2 from rest (period 1): dp, dl [m]; a h T / 2 [m]':
            '0.000102, -6.12e-05; 0.0001',
        'periods accelerating towards the wall at 2 m/s^2: count, end differences from, to [m]':
            '32, 0.000163, 0.000164',
        'tool folded: largest end difference [m], period': '0.000338, 57',
        'tool folded: largest |dp| at the end of a period [m], period': '0.000102, 35',
        'tool folded: largest position difference within a period [m], period': '0.000102, 35',
        'tool folded: largest speed difference within a period [m/s], period': '0.000347, 57',
        'within period: largest speed difference [m/s], period': '0.01, 201',
        'within period: largest speed difference elsewhere [m/s], period': '0.000347, 57',
        'within period: largest position difference [m], period': '0.000215, 201',
        'tool unfolds: period, speed difference within it, lowest and highest [m/s]':
            '201, -0.01, 0.00611',
        'tool unfolds: largest reach within the period [m], difference at its end [m]':
            '0.79979, -0.000205',
        'largest reach in any period [m], period': '0.80005, 202',
        'start of a period: largest distance from the exact state [m]': '0.000294',
        'start of a period: largest |dp| + |dl| [m], period; largest |dv| [m/s], period':
            '0.000248, 35; 0.000294, 58',
        'speed signal: acceleration less command over 1 ms, tool folded: periods above 0.05':
            '0 of 201',
        'speed signal: acceleration less command over 1 ms, tool folded: largest, period, at p':
            '0.0493, 56, 20.1',
        'axle: acceleration less command, tool folded: periods above 0.05 m/s^2': '0 of 201',
        'axle: acceleration less command, tool folded, over 1 ms: largest, period, at p':
            '0.0339, 65, 21.68',
        'axle: acceleration less command, tool folded, over one step: largest, period':
            '0.0339, 65',
        'jump of the speed signal against the axle in one step: largest [m/s], period, at p':
            '4.93e-05, 56, 20.1',
        'acceleration less command over 1 ms, tool unfolding [m/s^2]': '1.11',
        "braking claims of the exact run's labels: available on the simulated ground":
            '220 of 220',
        'smallest margin of a label [m/s^2], period, label': '0.232, 65, 0.5',
    },
    'the same at smaller time steps': {
        '0.5 ms: end of period, median, 90th percentile (nearest rank) [m]':
            '0.000163, 0.000252',
        '0.5 ms: braking at 2 m/s^2 (period 40): end difference [m]': '0.000253',
        '0.5 ms: end of period, largest [m], period': '0.000651, 201',
        '0.5 ms: tool unfolds, difference of the reach at the end [m]': '-0.000645',
        '0.5 ms: within period, largest speed difference outside it [m/s], period':
            '0.000345, 57',
        '0.5 ms: acceleration less command over 1 ms, tool folded, above 0.05 m/s^2':
            '0; largest 0.0493 in period 56',
        '0.5 ms: acceleration less command over one step, tool folded, above 0.05 m/s^2':
            '1; largest 0.0987 in period 56',
        '0.5 ms: accelerating at 2 m/s^2 from rest (period 1): dp [m]': '5.21e-05',
        '0.5 ms: axle: acceleration less command, tool folded, over 1 ms and one step':
            '0.0315, 0.0329',
        '0.5 ms: jump of the speed signal against the axle in one step [m/s], period':
            '4.94e-05, 56',
        '0.25 ms: end of period, median, 90th percentile (nearest rank) [m]':
            '0.000163, 0.000228',
        '0.25 ms: braking at 2 m/s^2 (period 40): end difference [m]': '0.000228',
        '0.25 ms: end of period, largest [m], period': '0.000894, 201',
        '0.25 ms: tool unfolds, difference of the reach at the end [m]': '-0.000888',
        '0.25 ms: within period, largest speed difference outside it [m/s], period':
            '0.000347, 57',
        '0.25 ms: acceleration less command over 1 ms, tool folded, above 0.05 m/s^2':
            '0; largest 0.0493 in period 56',
        '0.25 ms: acceleration less command over one step, tool folded, above 0.05 m/s^2':
            '2; largest 0.197 in period 56',
        '0.25 ms: accelerating at 2 m/s^2 from rest (period 1): dp [m]': '2.71e-05',
        '0.25 ms: axle: acceleration less command, tool folded, over 1 ms and one step':
            '0.0307, 0.0328',
        '0.25 ms: jump of the speed signal against the axle in one step [m/s], period':
            '4.94e-05, 56',
        'speed gain 100 1/s: largest speed difference within periods 56 to 67 [m/s]':
            '0.000347',
        'speed gain 200 1/s: largest speed difference within periods 56 to 67 [m/s]':
            '0.000178',
        'tool unfolds, reach short of the stop, implicitfast at 0.125, 0.0625 ms [m]':
            '0.00101, 0.00107',
        'tool unfolds, reach short of the stop, Euler at 1, 0.5, 0.25, 0.125 ms [m]':
            '0.00202, 0.00158, 0.00136, 0.00125',
    },
    'the cases of Section 10.4': {
        'differences with tolerance 0': '26 of 26',
        'differences with tolerance 0.001': '0 of 26',
        'largest difference of a component': '0.000666',
        'largest difference, component': 'v',
        'cases with a difference above 1 mm': 'none',
    },
    'the configuration of Chapter 3': {
        'declarations of robot, world and control vs model: problems': '0',
        'alarms': '1',
        'first alarm [s]': '0.1',
        'distance that raised it, in the norm of the state': '1.09e-08',
        'measured state at the alarm': '-1.34e-09, -1.09e-08, 3.98e-10',
        'set that the rest feedback can reach, after the alarm': 'none',
        'restarts': '0',
        'least clearance [m]': '25',
        'largest |p| over the run [m]': '4.59e-07',
        'first period in the work zone': 'none',
        'cart.metrics: tool unfolds at [s]': '0',
        'largest reach over the run [m]': '4.69e-08',
        'tool operation': 'dropped after an overload at 20 s',
    },
}


if __name__ == "__main__":
    main()
