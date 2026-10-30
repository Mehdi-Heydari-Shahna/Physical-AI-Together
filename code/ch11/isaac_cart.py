#!/usr/bin/env python3
"""
The cart of Chapter 3 in Isaac Sim: the stage, its record, its diagnostics
and the comparisons of Section 11.1, run in a second simulator.

Physical AI, Together, Chapter 11 (Section 11.2).

This script builds the cart and its track as a USD stage from the design of
Section 11.1 (cart_isaac/stage.py), simulates it with PhysX through Isaac
Sim, and
  * writes the record of the experiment: the versions, the settings of the
    simulator and those that differ from the defaults, the stage and its
    hash, and what PhysX reports of the model (links, joints, masses, drive
    gains);
  * runs the diagnostics of Section S11.9.1, each in its Isaac Sim form, with
    variants that have the mistakes they reveal: the wheel loads against the
    weight, also with a stage in centimetres; the ground under the wheels
    along the track; the rangefinder against the clearance, also with the
    wall rotated in radians where USD reads degrees; one period at constant
    speed; the steps of each period, also with a loop on the simulation
    clock and with updates of the application; the speed over the crest,
    also with the ground coordinate; the accelerations that the drive loop
    sustains; the contacts over the crest; a cart with welded wheels that
    slides; the encoders at three time steps, also with speed servos and
    with their damping written per radian where USD reads it per degree;
    and coasting with the default damping of PhysX;
  * runs each of the 220 periods of the run of Chapter 3 in Isaac Sim, as
    Section S11.9.2 does in MuJoCo, at time steps of 1, 0.5 and 0.25 ms,
    reading the disturbance from the speed signal and from the speed of the
    front axle; and, as there, the crest with twice the speed gain, and the
    tool's last approach to its stop at 0.125 and 0.0625 ms;
  * runs the 26 cases of Section 10.4 through the check motion_agreement;
  * runs the configuration of Chapter 3 with the simulator adapter
    cart.isaac, and reports the braking claims of the labels, alarms and
    physical violations separately;
  * imports results/cart.xml, the MJCF text of Section 11.1, with Isaac Sim's
    MJCF importer, and reports what the importer made of it.

Every step runs on its own: a step that fails is reported in
results/isaac/errors.txt with its traceback, and the others go on.

Usage, with the Python of Isaac Sim (6.x; 5.1 is attempted too):
    <isaac-sim>/python.sh isaac_cart.py 2>&1 | tee isaac-console.txt
    <isaac-sim>\\python.bat isaac_cart.py           (Windows)
    python isaac_cart.py                           (pip install of isaacsim)
Options:
    --quick         skip the runs at 0.5 and 0.25 ms
    --no-import     skip the import with the MJCF importer

Outputs, in results/isaac/ next to this file:
    isaac-summary.txt              every value, by group
    record.txt                     the record of the experiment
    cart.usda                      the stage that PhysX simulated
    agreement-chapter3-isaac.csv   one row per period of the run of Chapter 3
    import-mjcf.txt                what the MJCF importer made of cart.xml
    errors.txt                     the steps that failed, with tracebacks
"""
import argparse
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "isaac")

parser = argparse.ArgumentParser(description="Section 11.2: the cart in Isaac Sim")
parser.add_argument("--quick", action="store_true", help="skip the runs at 0.5 and 0.25 ms")
parser.add_argument("--no-import", action="store_true", help="skip the MJCF importer")
args, _unknown = parser.parse_known_args()

from isaacsim import SimulationApp                                    # noqa: E402

APP = SimulationApp({"headless": True})

import csv                                                            # noqa: E402
import dataclasses                                                    # noqa: E402
import hashlib                                                        # noqa: E402
import math                                                           # noqa: E402
import platform                                                       # noqa: E402
import statistics                                                     # noqa: E402
import subprocess                                                     # noqa: E402
from fractions import Fraction as Fr                                  # noqa: E402

import numpy as np                                                    # noqa: E402
import omni.kit.app                                                   # noqa: E402
import omni.timeline                                                  # noqa: E402
import omni.usd                                                       # noqa: E402

sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
import cart_isaac as ci                                               # noqa: E402
from cart_isaac import sim as cs                                      # noqa: E402
from cart_isaac import stage as cst                                   # noqa: E402

cs.APP = APP
import allovi                                                         # noqa: E402
import cart                                                           # noqa: E402
from allovi import model as am                                        # noqa: E402
import model_providers                                                # noqa: E402  (../ch10)

T = 0.1
DESIGN = ci.Design()
SETTINGS = ci.IsaacSettings()
G = cs.G
STEPS = (0.001, 0.0005, 0.00025)
FACTS, ERRORS = {}, []
STARTED = time.time()


def g3(x):
    return f"{x:.3g}"


def ms(dt):
    return f"{1000 * dt:g} ms"


def norm(dp, dv, dl):
    """The norm of the state, max(|p| + |l|, |v| * 1 s)."""
    return max(abs(dp) + abs(dl), abs(dv))


def at_step(dt):
    return dataclasses.replace(SETTINGS, timestep=dt)


def say(text):
    print(f"[{time.time() - STARTED:7.1f} s] {text}", flush=True)


def attempt(out, key, fn):
    """out[key] = fn(), or the failure, recorded with its traceback."""
    try:
        out[key] = fn()
    except Exception as exc:                       # noqa: BLE001 (report and go on)
        out[key] = f"FAILED: {exc!r}"
        ERRORS.append((key, traceback.format_exc()))
        say(f"  failed: {key}: {exc!r}")


def group(name, fn):
    say(name)
    out = FACTS.setdefault(name, {})
    try:
        fn(out)
    except Exception as exc:                       # noqa: BLE001
        out["FAILED"] = repr(exc)
        ERRORS.append((name, traceback.format_exc()))
        say(f"  failed: {name}: {exc!r}")
    write_outputs()


def new_cart(settings=SETTINGS, variant=None):
    say(f"  stage: {variant or 'adapter'}, {ms(settings.timestep)}")
    return cs.IsaacCart(DESIGN, settings, variant)


# ----------------------------------------------------------------------------
# The record
# ----------------------------------------------------------------------------
def ext_id(name):
    try:
        return omni.kit.app.get_app().get_extension_manager().get_enabled_extension_id(name) or "-"
    except Exception:                              # noqa: BLE001
        return "-"


def isaac_version():
    try:
        from isaacsim.core.version import get_version
        v = get_version()
        return v[0] + (f"-{v[1]}" if v[1] else "")
    except Exception as exc:                       # noqa: BLE001
        return f"unknown ({exc!r})"


def gpu():
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        return q.stdout.strip().replace("\n", "; ") or "not reported"
    except Exception:                              # noqa: BLE001
        return "not reported"


def cpu():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "not reported"


def defaults():
    """The fallback values of the attributes that the stage writes, read
    from schemas applied to fresh prims, and the units of a new stage."""
    from pxr import PhysxSchema, UsdGeom, UsdPhysics
    ctx = omni.usd.get_context()
    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_stopped():
        timeline.stop()
        APP.update()
    ctx.new_stage()
    APP.update()
    stage = ctx.get_stage()
    out = {"stage: metersPerUnit, kilogramsPerUnit, upAxis":
           f"{UsdGeom.GetStageMetersPerUnit(stage):g}, {UsdPhysics.GetStageKilogramsPerUnit(stage):g}, "
           f"{UsdGeom.GetStageUpAxis(stage)}"}
    scene = UsdPhysics.Scene.Define(stage, "/defaults/scene")
    px = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    body = UsdGeom.Xform.Define(stage, "/defaults/body").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(body)
    col = UsdGeom.Cube.Define(stage, "/defaults/body/cube").GetPrim()
    UsdPhysics.CollisionAPI.Apply(col)
    pc = PhysxSchema.PhysxCollisionAPI.Apply(col)
    root = UsdGeom.Xform.Define(stage, "/defaults/art").GetPrim()
    pa = PhysxSchema.PhysxArticulationAPI.Apply(root)

    def val(api, name):
        try:
            return str(getattr(api, f"Get{name}Attr")().Get())
        except Exception as exc:                   # noqa: BLE001
            return f"n/a ({type(exc).__name__})"
    out["scene: gravity magnitude"] = val(scene, "GravityMagnitude")
    for name in ("TimeStepsPerSecond", "SolverType", "EnableGPUDynamics", "BroadphaseType",
                 "FrictionType", "EnableCCD", "EnableStabilization"):
        out[f"scene: {name}"] = val(px, name)
    for name in ("LinearDamping", "AngularDamping", "MaxDepenetrationVelocity"):
        out[f"rigid body: {name}"] = val(rb, name)
    for name in ("ContactOffset", "RestOffset"):
        out[f"collider: {name}"] = val(pc, name)
    for name in ("SolverPositionIterationCount", "SolverVelocityIterationCount",
                 "SleepThreshold", "StabilizationThreshold", "EnabledSelfCollisions"):
        out[f"articulation: {name}"] = val(pa, name)
    return out


def record_facts(out):
    out["Isaac Sim"] = isaac_version()
    try:
        out["Kit"] = omni.kit.app.get_app().get_build_version()
    except Exception:                              # noqa: BLE001
        out["Kit"] = "-"
    for name in ("omni.physx", "omni.physics.tensors", "omni.physx.tensors",
                 "isaacsim.core.simulation_manager", "isaacsim.asset.importer.mjcf"):
        out[f"extension {name}"] = ext_id(name)
    out["Python, NumPy"] = f"{platform.python_version()}, {np.__version__}"
    out["OS, machine"] = f"{platform.system()} {platform.release()}, {platform.machine()}"
    out["CPU"] = cpu()
    out["GPU (name, driver, memory)"] = gpu()
    found = {}
    attempt(found, "defaults", defaults)
    if isinstance(found["defaults"], dict):
        for k, v in found["defaults"].items():
            out[f"default {k}"] = v
    else:
        out["defaults"] = found["defaults"]


def model_facts(out, sim):
    """What PhysX reports of the stage that the script built."""
    path = os.path.join(OUT, "cart.usda")
    sim.stage.GetRootLayer().Export(path)
    with open(path, "rb") as fh:
        text = fh.read()
    out["stage text, SHA-256"] = hashlib.sha256(text).hexdigest()[:16]
    out["physics time step: the scene's, as the simulation manager reports it [s]"] = \
        g3(sim.physics.scene_dt) if sim.physics.scene_dt is not None else "not reported"
    out["physics stepped through"] = sim.physics.mode
    out["link poses read from"] = sim.links_from
    out["physics time step [s]"] = g3(sim.dt)
    out["steps per period"] = sim.steps
    out["DOF names"] = ", ".join(sim.dof_names)
    out["link names"] = ", ".join(sim.link_names)
    attempt(out, "link masses [kg]", lambda: ", ".join(
        f"{n} {m:.6g}" for n, m in zip(sim.link_names, np.asarray(sim.art.get_masses())[0])))
    attempt(out, "DOF types", lambda: ", ".join(str(t) for t in sim.art.shared_metatype.dof_types))
    for what in ("stiffnesses", "dampings", "max_forces", "max_velocities"):
        attempt(out, f"DOF {what}", lambda what=what: ", ".join(
            f"{n} {x:.6g}" for n, x in zip(sim.dof_names,
                                           np.asarray(getattr(sim.art, f"get_dof_{what}")())[0])))
    attempt(out, "DOF limits", lambda: ", ".join(
        f"{n} [{lo:.6g}, {hi:.6g}]" for n, (lo, hi) in zip(sim.dof_names,
                                                          np.asarray(sim.art.get_dof_limits())[0])))
    out["penetration at rest [mm]"] = g3(1000 * sim.sink)
    out["physics steps counted by the simulation manager, and by the adapter"] = attempt_value(
        lambda: f"{sim.physics.count()}, {sim.clock}")


def attempt_value(fn):
    try:
        return fn()
    except Exception as exc:                       # noqa: BLE001
        ERRORS.append(("value", traceback.format_exc()))
        return f"FAILED: {exc!r}"


# ----------------------------------------------------------------------------
# Diagnostics (Section S11.9.1, in Isaac Sim)
# ----------------------------------------------------------------------------
def drive(sim, p, v, a, duration, sample, u_tool=0.0):
    """Places the cart at (p, v) and holds the constant acceleration a with
    the drive loop for 'duration' seconds; sample(sim, i, v_ref) is called
    before every step."""
    sim.place(p, v, 0.0)
    for i in range(int(round(duration / sim.dt)) + 1):
        v_ref = v + a * i * sim.dt
        sample(sim, i, v_ref)
        sim.controls(a, v_ref, u_tool)
        sim.step()


def tracking(sim, p, v, a, duration=0.5):
    errs = []
    drive(sim, p, v, a, duration, lambda s, i, v_ref: errs.append(s.state()[1] - v_ref))
    k = int(round(0.1 / sim.dt))
    return max(abs(e) for e in errs), max(abs(e - errs[k]) for e in errs[k:])


def capacity(sim, p, v, sign):
    """As in Section 11.1: the largest acceleration of the given sign that
    the drive loop sustains from (p, v), by bisection to 0.005 m/s^2."""
    lo, hi = 0.0, 5.0
    while hi - lo > 0.005:
        mid = (lo + hi) / 2
        if tracking(sim, p, v, sign * mid)[1] <= 1e-3:
            lo = mid
        else:
            hi = mid
    return lo


def rest_loads(sim, p=5.0):
    sim.place(p, 0.0, 0.0)
    for _ in range(10):
        sim.period(0.0, 0.0)
    return sim.loads(), sim.contact_forces()


def ground_gap(sim, s):
    """The distance along the ground's normal from the profile at ground
    coordinate s to the collision surface, by a ray cast 0.5 m beside the
    cart (y = 0.5) from 0.5 m above the profile; zero if they agree. A ray
    through an edge of the mesh may miss it; it is then cast 1 um aside."""
    for ds in (0.0, 1e-6, -1e-6):
        x, z, _ = sim.pr.point(s + ds)
        nx, nz = sim.pr.normal(s + ds)
        dist, path = sim.raycast((x + 0.5 * nx, 0.5, z + 0.5 * nz), (-nx, 0.0, -nz), 1.0)
        if dist is not None and path.endswith("/World/ground"):
            return dist - 0.5
    raise RuntimeError(f"no ray near s = {s} hit the ground (last hit {path!r})")


def rangefinder(sim, p, l):
    """The rangefinder at the tool tip, 1 mm ahead of it to miss the tool
    itself, pointing along the chassis, less the clearance 25 - (p + l)."""
    sim.place(p, 0.0, l)
    tip = sim.position("tool")
    ax = sim.axis_x("chassis")
    origin = tuple(tip[i] + 1e-3 * ax[i] for i in range(3))
    dist, path = sim.raycast(origin, ax, 5.0)
    if dist is None:
        raise RuntimeError("the rangefinder ray hit nothing")
    p1, _, l1 = sim.state()
    return dist + 1e-3 - (DESIGN.wall - p1 - l1), path


def crest_contacts(sim, p, v, a, duration):
    """While the cart crosses the crest from (p, v), under the drive loop at
    the acceleration a, or without torques (a = None): the least penetration
    of a wheel on the crest (the negative of its contacts' least separation;
    a negative value is a gap), the steps in which a wheel carries no load,
    and the largest relative difference of the total load from
    M (g cos(pitch) - v^2 / (R + r)) while both wheels are on the crest.
    PhysX reports contact points within the contact offset, before the
    surfaces touch, so a wheel counts as in contact while it carries load."""
    pr, r = sim.pr, DESIGN.wheel_radius
    least, lost, worst = math.inf, 0, 0.0
    sim.place(p, v, 0.0)
    zero = np.zeros((1, sim.ndof), dtype=np.float32)
    for i in range(int(round(duration / sim.dt))):
        if a is None:
            sim.art.set_dof_actuation_forces(zero, sim.idx)
        else:
            sim.controls(a, v + a * i * sim.dt, 0.0)
        sim.step()
        parts = {w: sim.foot(w).part for w in ("front", "rear")}
        if "crest" not in parts.values():
            continue
        sep = sim.separations()
        loads = dict(zip(("front", "rear"), sim.loads()))
        lost += any(loads[w] <= 0.0 for w in ("front", "rear"))
        for w in ("front", "rear"):
            if parts[w] == "crest" and sep[w] is not None:
                least = min(least, -sep[w])
        if all(x == "crest" for x in parts.values()) and all(loads[w] > 0.0 for w in loads):
            speed = sim.state()[1]
            expected = sim.mass * (G * math.cos(sim.pitch()) - speed ** 2 / (pr.R + r))
            worst = max(worst, abs(sum(loads.values()) - expected) / expected)
    return least, lost, worst


def sled(sim, p=8.0, v=4.0, duration=0.5):
    """The cart with welded wheels sliding from (p, v): steps in which the
    front wheel carries load, its largest load, and the mean deceleration
    over the last 0.3 s."""
    sim.place(p, v, 0.0)
    zero = np.zeros((1, sim.ndof), dtype=np.float32)
    n = int(round(duration / sim.dt))
    touching, loads, speeds = 0, [], []
    for _ in range(n):
        speeds.append(sim.state()[1])
        sim.art.set_dof_actuation_forces(zero, sim.idx)
        sim.step()
        load = sim.loads()[0]
        touching += load > 0.0
        loads.append(load)
    k = int(round(0.3 / sim.dt))
    return touching, n, max(loads), (speeds[n - k - 1] - speeds[n - 1]) / 0.3


def encoders(sim):
    """Encoder speed (the front wheel's rate times r) less the cart's speed
    after 0.5 s of accelerating at 2 m/s^2 from rest at p = 2 m [m/s]."""
    sim.place(2.0, 0.0, 0.0)
    for _ in range(5):
        sim.period(2.0, 0.0)
    return sim.wheel_rates()[0] * DESIGN.wheel_radius - sim.state()[1]


def coast(sim, duration=1.0):
    """Speed lost in 'duration' seconds by a cart coasting from 1 m/s on
    the firm ground with no torques [m/s]."""
    sim.place(3.0, 1.0, 0.0)
    v0 = sim.state()[1]
    zero = np.zeros((1, sim.ndof), dtype=np.float32)
    for _ in range(int(round(duration / sim.dt))):
        sim.art.set_dof_actuation_forces(zero, sim.idx)
        sim.step()
    return v0 - sim.state()[1]


def clock_loop(sim):
    """Steps of each of 220 periods when the loop steps while the
    simulation clock, from the start of the loop, is below the end of the
    period, (k + 1) T for period k; and the clock after 22,000 steps."""
    sim.place(0.0, 0.0, 0.0)
    t0 = sim.physics.time()
    counts, n = [], 0
    for k in range(220):
        c = 0
        while sim.physics.time() - t0 < (k + 1) * T and c < 2 * sim.steps:
            sim.step()
            c += 1
            n += 1
        counts.append(c)
    exceptions = "; ".join(f"{c} in period {k}" for k, c in enumerate(counts) if c != sim.steps)
    for _ in range(max(0, 22000 - n)):
        sim.step()
    return exceptions or f"{sim.steps} in each", repr(sim.physics.time() - t0)


def app_updates(sim, updates=60):
    """Physics steps per update of the application while the timeline
    plays: what a loop that stepped by updates would get."""
    sim.place(0.0, 0.0, 0.0)
    per = []
    for _ in range(updates):
        n0 = sim.physics.count()
        APP.update()
        per.append(sim.physics.count() - n0)
    sim._cache = None
    values = sorted(set(per))
    return ", ".join(f"{v} ({per.count(v)} times)" for v in values), sum(per[:6])


def diagnostics(out):
    sim = new_cart()
    model = {}
    model_facts(model, sim)
    FACTS["the model as PhysX reports it"] = model
    pr = sim.pr

    def loads():
        ld, forces = rest_loads(sim)
        up = all(forces[w][2] > 0 for w in forces)
        return (f"{ld[0]:.5g} + {ld[1]:.5g} = {sum(ld):.6g}; contact forces point up: {up}")
    attempt(out, "wheel loads at rest [N]", loads)
    out["weight of the parts the wheels carry [N]"] = (
        f"{(DESIGN.chassis_mass + 2 * DESIGN.wheel_mass + DESIGN.tool_mass + SETTINGS.aux_mass) * G:.6g}")

    def geometry():
        crest_end = pr.s0 + (pr.R + pr.r) * pr.theta + sim.lead
        base = 2 * DESIGN.wheel_offset
        positions = sorted({2.0, 10.0, 19.9, pr.s0 + sim.lead, 20.35, pr.s0 + sim.lead + base,
                            21.0, 21.4, crest_end, 22.0, crest_end + base, 23.0, 24.0, 24.9})
        off = on = 0.0
        for p in positions:
            sim.place(p, 0.0, 0.0)
            for wheel in ("front", "rear"):
                f = sim.foot(wheel)
                gap = abs(ground_gap(sim, f.s))
                if f.part == "crest":
                    on = max(on, gap)
                else:
                    off = max(off, gap)
        return f"{len(positions)} positions; off the crest {g3(off)}, on it {g3(on)}"
    attempt(out, "geometry: largest gap between profile and collision surface [m]", geometry)

    def range_adapter():
        a = rangefinder(sim, 23.7, 0.0)
        b = rangefinder(sim, 23.7, 0.8)
        return f"{g3(a[0])}, {g3(b[0])} (hit {a[1]})"
    attempt(out, "rangefinder less clearance at p 23.7 m, tool folded and unfolded [m]",
            range_adapter)

    def one_period():
        sim.place(5.0, 1.0, 0.0)
        p, v, l = sim.period(0.0, 0.0)
        return g3(p - 5.1)
    attempt(out, "one period at 1 m/s from 5 m: dp [m]", one_period)
    out["steps per period, adapter"] = f"{sim.steps} in each, counted"

    def crest_speed():
        errs = []
        drive(sim, 19.8, 2.1, -0.5, 1.2, lambda s, i, v_ref: errs.append(abs(s.state()[1] - v_ref)))
        return g3(max(errs))
    attempt(out, "crest, axle coordinate: largest |v - v_ref| [m/s]", crest_speed)

    def crest_ground():
        rows = []
        sim.coordinate = "ground"
        try:
            drive(sim, 19.8, 2.1, -0.5, 1.2, lambda s, i, v_ref: rows.append(
                (s.state()[0], s.state()[1] - v_ref, s.state(coordinate="axle")[1] - v_ref,
                 s.state(coordinate="axle")[0])))
        finally:
            sim.coordinate = "axle"
        first = next(i for i, r in enumerate(rows) if r[0] >= pr.s0 + sim.lead)
        over = [r for r in rows if r[2] > 0.015]
        span = f"{over[0][3]:.4g} to {over[-1][3]:.4g}" if over else "none"
        return (f"jump {g3(rows[first][1] - rows[first - 1][1])} at p {rows[first][3]:.3g}; "
                f"true speed above v_ref by up to {g3(max(r[2] for r in rows))}; "
                f"more than 15 mm/s above for p in {span}")
    attempt(out, "crest, ground coordinate [m/s]", crest_ground)

    for key, case in (("braking on firm ground from 4 m/s", (8.0, 4.0, -1)),
                      ("braking on the slope from 2 m/s", (22.5, 2.0, -1)),
                      ("from rest towards the start on the slope", (24.0, 0.0, -1)),
                      ("from rest towards the wall on firm ground", (2.0, 0.0, 1))):
        attempt(out, f"capacity: {key} [m/s^2]", lambda case=case: g3(capacity(sim, *case)))
    for key, case in (("label 2 on firm ground", (8.0, 4.0, -2.0)),
                      ("label 0.5 over the end of the crest", (21.6, 2.12, -0.5)),
                      ("command bound towards the wall", (2.0, 0.0, 2.0))):
        attempt(out, f"{key}: speed error, largest and change after 0.1 s [mm/s]",
                lambda case=case: ", ".join(g3(1000 * x) for x in tracking(sim, *case)))

    def crest_label():
        least, lost, worst = crest_contacts(sim, 20.0, 2.12, -0.5, 1.4)
        return (f"least penetration {g3(1000 * least)} mm; steps with a wheel unloaded {lost}; "
                f"load within {g3(100 * worst)} % of M(g cos - v^2/(R+r))")
    attempt(out, "crest, braking at 0.5 from 2.12 m/s at 20 m", crest_label)

    for speed in (2.3, 2.4, 2.5):
        def crest_constant(speed=speed):
            least, lost, worst = crest_contacts(sim, 19.9, speed, 0.0, (22.6 - 19.9) / speed)
            return f"least penetration {g3(1000 * least)} mm; steps with a wheel unloaded {lost}"
        attempt(out, f"crest at a constant {speed:g} m/s", crest_constant)

    def crest_free():
        least, lost, worst = crest_contacts(sim, 19.5, 3.0, None, 1.0)
        return f"least penetration {g3(1000 * least)} mm; steps with a wheel unloaded {lost}"
    attempt(out, "crest, rolling freely from 3 m/s", crest_free)
    attempt(out, "encoder less cart speed at 2 m/s^2, drive loop, at 1 ms [mm/s]",
            lambda: g3(1000 * encoders(sim)))
    attempt(out, "coasting 1 s from 1 m/s, no torque: speed lost [mm/s]",
            lambda: g3(1000 * coast(sim)))

    def clock():
        exceptions, after = clock_loop(sim)
        return f"{exceptions}; clock after 22,000 steps {after}"
    attempt(out, "steps per period, loop on the simulation clock", clock)

    def updates():
        dist, six = app_updates(sim)
        return f"{dist}; first six updates {six} steps"
    attempt(out, "physics steps per update of the application", updates)
    sim.close()


def variants(out):
    def wall():
        sim = new_cart(variant="wall in radians")
        try:
            a = rangefinder(sim, 23.7, 0.0)
            b = rangefinder(sim, 23.7, 0.8)
            return f"{g3(a[0])}, {g3(b[0])}"
        finally:
            sim.close()
    attempt(out, "wall in radians: rangefinder less clearance, folded and unfolded [m]", wall)

    def centimetres():
        sim = new_cart(variant="stage in centimetres")
        try:
            ld, _ = rest_loads(sim)
            return f"{sum(ld):.6g}"
        finally:
            sim.close()
    attempt(out, "stage in centimetres: wheel loads at rest, together [stage force units]",
            centimetres)

    def defaults_coast():
        sim = new_cart(variant="physx defaults")
        try:
            return g3(1000 * coast(sim))
        finally:
            sim.close()
    attempt(out, "physx defaults: coasting 1 s from 1 m/s, speed lost [mm/s]", defaults_coast)

    def servo(variant, dt):
        sim = new_cart(at_step(dt), variant)
        try:
            gains = np.asarray(sim.art.get_dof_dampings())[0]
            return 1000 * encoders(sim), float(gains[sim.dof["front"]])
        finally:
            sim.close()
    steps = STEPS[:1] if args.quick else STEPS
    for variant in ("speed servos", "servo damping per radian"):
        def run(variant=variant):
            vals = [servo(variant, dt) for dt in (steps if variant == "speed servos" else STEPS[:1])]
            return (", ".join(g3(x[0]) for x in vals) +
                    f"; damping as PhysX reports it {vals[0][1]:.6g}")
        attempt(out, f"{variant}: encoder less cart speed at 2 m/s^2 [mm/s], by time step", run)

    def drive_steps():
        vals = []
        for dt in steps[1:]:
            sim = new_cart(at_step(dt))
            try:
                vals.append(g3(1000 * encoders(sim)))
            finally:
                sim.close()
        return ", ".join(vals) or "skipped"
    attempt(out, "drive loop: encoder less cart speed at 0.5 and 0.25 ms [mm/s]", drive_steps)

    def welded():
        vals = []
        for dt in steps:
            sim = new_cart(at_step(dt), "welded wheels")
            try:
                touching, n, load, decel = sled(sim)
                vals.append(f"{ms(dt)}: front wheel loaded in {touching} of {n} steps, "
                            f"largest load {load:.3g} N, mean deceleration {decel:.3g} m/s^2")
            finally:
                sim.close()
        return "; ".join(vals)
    attempt(out, "sled (welded wheels) from 4 m/s", welded)


# ----------------------------------------------------------------------------
# The run of Chapter 3, period by period
# ----------------------------------------------------------------------------
PERIODS = None


def chapter3_periods():
    cfg = cart.configuration("chapter3", postpone=None)
    rec = allovi.run(cfg, 220, strict=True)
    out = []
    for per in rec.periods:
        segment, label = cfg.control.head(per.plan)
        out.append((per.k, per.state, cfg.control.law(per.plan), label, segment))
    return out


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


def agreement(sim, capacity_of, only=None):
    """As in Section S11.9.2: each period (or those in 'only') placed 0.05 s
    early on the motion of the previous period, driven under the previous
    law until it starts, then executed and compared with the model's nominal
    motion. The realized acceleration less the commanded one is read from
    the adapter's speed signal (w) and from the speed of the front axle
    along the ground (w_axle), over windows of 1 ms and over single steps;
    where the curvature of the ground jumps, the two signals differ by a
    jump of the coordinate (jump)."""
    global PERIODS
    if PERIODS is None:
        PERIODS = chapter3_periods()
    rows = []
    pre = int(round(0.05 / sim.dt))
    win = max(1, int(round(0.001 / sim.dt)))
    previous = (0.0, 0.0, "segment")
    for k, x, law, label, segment in PERIODS:
        p, v, l = (float(a) for a in x)
        u_prev, ell_prev, kind_prev = previous
        u = float(law.u)
        ell = l if law.ell is None else float(law.ell)
        previous = (u, ell, law.kind)
        if only is not None and k not in only:
            continue
        dt = pre * sim.dt
        sim.place(p - v * dt + u_prev * dt * dt / 2, v - u_prev * dt, l)
        sim.period(u_prev, ell_prev, kind_prev, steps=pre)
        p0, v0, l0 = sim.state()
        rec = []
        p1, v1, l1 = sim.period(u, ell, law.kind, record=rec)
        last = rec[-1]                # the end of the period, so that the
        rec.append(sim.sample(last["a_ref"],      # records cover its last step
                              last["v_ref"] + last["a_ref"] * sim.dt))
        np_, nv, nl = cart.model.MODEL.dynamics((p0, v0, l0), (u, ell), T)
        dp, dv, dl = p1 - np_, v1 - nv, l1 - nl
        within_p = within_v = low = high = 0.0
        t0 = rec[0]["t"]
        for r in rec:
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
        cap = capacity_of(float(segment.lo), float(segment.hi), speed)
        rows.append(dict(k=k, t=k * T, p=p, v=v, l=l, u=u, ell=ell, label=float(label),
                         capacity=cap, start=norm(p0 - p, v0 - v, l0 - l),
                         start_pl=abs(p0 - p) + abs(l0 - l), start_v=abs(v0 - v), dp=dp, dv=dv,
                         dl=dl, end=norm(dp, dv, dl), within_p=within_p, within_v=within_v,
                         low=low, high=high, lmax=max(max(r["l"] for r in rec), l1),
                         w=w, w_at=w_at, w_step=w_step, w_axle=w_axle, w_axle_at=w_axle_at,
                         w_axle_step=w_axle_step, jump=jump, jump_at=jump_at))
        if k % 20 == 0:
            say(f"    period {k}")
    return rows


def summary(rows):
    """The values of a comparison that Section 11.1 quotes for MuJoCo."""
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
                brake=next(r for r in rows if r["k"] == 40),
                unfold=next(r for r in rows if r["l"] == 0 and r["ell"] > 0),
                rest=[r for r in rows if r["l"] == 0 and r["p"] == 24.5 and r["u"] == 0])


def agreement_facts(out):
    world = ci.CartIsaac()                         # capacity oracle and a fresh stage
    try:
        rows = agreement(world.sim, world.capacity)
    finally:
        world.sim.close()
    with open(os.path.join(OUT, "agreement-chapter3-isaac.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        cols = list(rows[0])
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] if c == "k" else f"{r[c]:.9g}" for c in cols])
    s = summary(rows)
    worst, unfold = s["worst"], s["unfold"]
    margin = min(rows, key=lambda r: r["capacity"] - r["label"])
    lmax = max(rows, key=lambda r: r["lmax"])
    acc = s["accelerating"]
    out["periods"] = len(rows)
    out["end of period: median, 90th percentile (nearest rank) [m]"] = \
        f"{g3(s['median'])}, {g3(s['p90'])}"
    braking = s["braking_above"]
    out["end of period, above the 90th percentile: braking at 2 m/s^2 (count, from, to [m]); "
        "the others (period: difference [m])"] = (
        (f"{len(braking)}, {g3(min(r['end'] for r in braking))}, "
         f"{g3(max(r['end'] for r in braking))}" if braking else "0") + "; " + ", ".join(
            f"{r['k']}: {g3(r['end'])}" for r in s["above_p90"] if r["u"] != -2.0))
    out["end of period: periods in which the speed sets the norm"] = ranges(
        r["k"] for r in rows if abs(r["dv"]) > abs(r["dp"]) + abs(r["dl"]))
    out["end of period: largest [m], period, t [s]"] = \
        f"{g3(worst['end'])}, {worst['k']}, {worst['t']:.3g}"
    out["end of period, largest: dp, dv, dl"] = ", ".join(g3(worst[c]) for c in ("dp", "dv", "dl"))
    if s["rest"]:
        out["periods at rest at 24.5 m: count, median dl [m]"] = \
            f"{len(s['rest'])}, {g3(statistics.median(r['dl'] for r in s['rest']))}"
        out["periods at rest at 24.5 m: end differences from, to [m]"] = \
            f"{g3(min(r['end'] for r in s['rest']))}, {g3(max(r['end'] for r in s['rest']))}"
    out["braking at 2 m/s^2 (period 40): dp, dl [m]"] = \
        f"{g3(s['brake']['dp'])}, {g3(s['brake']['dl'])}"
    out["braking at 2 m/s^2 (period 40): end difference [m]"] = g3(s["brake"]["end"])
    out["accelerating at 2 m/s^2 from rest (period 1): dp, dl [m]"] = \
        f"{g3(s['accel']['dp'])}, {g3(s['accel']['dl'])}"
    out["periods accelerating towards the wall at 2 m/s^2: count, end differences from, to [m]"] = (
        f"{len(acc)}, {g3(min(r['end'] for r in acc))}, {g3(max(r['end'] for r in acc))}"
        if acc else "0")
    out["tool folded: largest end difference [m], period"] = \
        f"{g3(s['folded_end']['end'])}, {s['folded_end']['k']}"
    out["tool folded: largest |dp| at the end of a period [m], period"] = \
        f"{g3(abs(s['folded_dp']['dp']))}, {s['folded_dp']['k']}"
    out["tool folded: largest position difference within a period [m], period"] = \
        f"{g3(s['folded_p']['within_p'])}, {s['folded_p']['k']}"
    out["tool folded: largest speed difference within a period [m/s], period"] = \
        f"{g3(s['folded_v']['within_v'])}, {s['folded_v']['k']}"
    out["within period: largest speed difference [m/s], period"] = \
        f"{g3(s['worst_v']['within_v'])}, {s['worst_v']['k']}"
    out["within period: largest speed difference elsewhere [m/s], period"] = \
        f"{g3(s['worst_v2']['within_v'])}, {s['worst_v2']['k']}"
    out["within period: largest position difference [m], period"] = (
        f"{g3(max(r['within_p'] for r in rows))}, "
        f"{max(rows, key=lambda r: r['within_p'])['k']}")
    out["tool unfolds: period, speed difference within it, lowest and highest [m/s]"] = \
        f"{unfold['k']}, {g3(unfold['low'])}, {g3(unfold['high'])}"
    out["tool unfolds: largest reach within the period [m], difference at its end [m]"] = \
        f"{unfold['lmax']:.5f}, {g3(unfold['dl'])}"
    out["largest reach in any period [m], period"] = f"{lmax['lmax']:.5f}, {lmax['k']}"
    out["start of a period: largest distance from the exact state [m]"] = \
        g3(max(r["start"] for r in rows))
    out["start of a period: largest |dp| + |dl| [m], period; largest |dv| [m/s], period"] = (
        f"{g3(max(r['start_pl'] for r in rows))}, "
        f"{max(rows, key=lambda r: r['start_pl'])['k']}; "
        f"{g3(max(r['start_v'] for r in rows))}, "
        f"{max(rows, key=lambda r: r['start_v'])['k']}")
    out["speed signal: acceleration less command over 1 ms, tool folded: periods above 0.05"] = \
        f"{len(s['above'])} of {len(s['folded'])}"
    out["speed signal: acceleration less command over 1 ms, tool folded: largest, period, at p"] = \
        f"{g3(s['wmax']['w'])}, {s['wmax']['k']}, {s['wmax']['w_at']:.4g}"
    out["axle: acceleration less command, tool folded: periods above 0.05 m/s^2"] = \
        f"{len(s['above_axle'])} of {len(s['folded'])}"
    out["axle: acceleration less command, tool folded, over 1 ms: largest, period, at p"] = \
        f"{g3(s['waxle']['w_axle'])}, {s['waxle']['k']}, {s['waxle']['w_axle_at']:.4g}"
    out["axle: acceleration less command, tool folded, over one step: largest, period"] = \
        f"{g3(s['waxle_step']['w_axle_step'])}, {s['waxle_step']['k']}"
    out["jump of the speed signal against the axle in one step: largest [m/s], period, at p"] = \
        f"{g3(s['jump']['jump'])}, {s['jump']['k']}, {s['jump']['jump_at']:.4g}"
    out["acceleration less command over 1 ms, tool unfolding [m/s^2]"] = g3(unfold["w"])
    out["braking claims of the exact run's labels: available on the simulated ground"] = \
        f"{sum(r['capacity'] >= r['label'] for r in rows)} of {len(rows)}"
    out["smallest margin of a label [m/s^2], period, label"] = \
        f"{g3(margin['capacity'] - margin['label'])}, {margin['k']}, {margin['label']:g}"


def compare(settings=SETTINGS, design=DESIGN, only=None):
    """The comparison period by period on a fresh stage."""
    world = ci.CartIsaac(design=design, settings=settings)
    try:
        return agreement(world.sim, world.capacity, only)
    finally:
        world.sim.close()


def timestep_facts(out):
    if args.quick:
        out["skipped"] = "--quick"
        return
    for dt in STEPS[1:]:
        found = {}
        attempt(found, "rows", lambda dt=dt: compare(at_step(dt)))
        rows = found["rows"]
        if isinstance(rows, str):
            out[f"{ms(dt)}"] = rows
            continue
        s = summary(rows)
        out[f"{ms(dt)}: end of period, median, 90th percentile (nearest rank) [m]"] = \
            f"{g3(s['median'])}, {g3(s['p90'])}"
        out[f"{ms(dt)}: braking at 2 m/s^2 (period 40): end difference [m]"] = \
            g3(s["brake"]["end"])
        out[f"{ms(dt)}: end of period, largest [m], period"] = \
            f"{g3(s['worst']['end'])}, {s['worst']['k']}"
        out[f"{ms(dt)}: tool unfolds, difference of the reach at the end [m]"] = \
            g3(s["unfold"]["dl"])
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
        write_outputs()
    # the drive loop at the crest: its speed differences against its gain
    for gain in (100.0, 200.0):
        attempt(out, f"speed gain {gain:g} 1/s: largest speed difference within periods 56 to "
                     f"67 [m/s]",
                lambda gain=gain: g3(max(r["within_v"] for r in compare(
                    design=dataclasses.replace(DESIGN, speed_gain=gain), only=range(56, 68)))))
    # the tool's last approach (period 201) at smaller steps
    attempt(out, "tool unfolds, reach short of the stop, at 0.125, 0.0625 ms [m]",
            lambda: ", ".join(g3(-compare(at_step(dt), only={201})[0]["dl"])
                              for dt in (0.000125, 0.0000625)))


def motion_cases_facts(out):
    world = ci.CartIsaac()
    try:
        cases = model_providers.motion_cases()
        worst, which, big = 0.0, None, []
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
    finally:
        world.sim.close()


def closed_loop_facts(out):
    cfg = cart.configuration("chapter3, Isaac Sim", postpone=None, world=ci.CartIsaac.NAME)
    out["declarations of robot, world and control vs model: problems"] = \
        len(am.agreement(cart.model.MODEL, cfg.robot, cfg.world, cfg.control))
    try:
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
                                 f"{g3(cfg.task.overloads[0] * T)} s") if cfg.task.dropped \
            else "done"
    finally:
        try:
            cfg.world.sim.close()
        except Exception:                          # noqa: BLE001
            pass


# ----------------------------------------------------------------------------
# The MJCF text of Section 11.1 through Isaac Sim's importer
# ----------------------------------------------------------------------------
def import_facts(out):
    if args.no_import:
        out["skipped"] = "--no-import"
        return
    from pxr import PhysxSchema, Usd, UsdPhysics
    src = os.path.join(HERE, "results", "cart.xml")
    dst = os.path.join(OUT, "imported")
    os.makedirs(dst, exist_ok=True)
    mgr = omni.kit.app.get_app().get_extension_manager()
    for ext in ("omni.scene.optimizer.core", "isaacsim.robot.schema", "isaacsim.asset.importer.mjcf"):
        try:
            mgr.set_extension_enabled_immediate(ext, True)
        except Exception:                          # noqa: BLE001
            pass
    from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig
    config = MJCFImporterConfig()
    config.mjcf_path = src
    config.usd_path = dst
    usd = MJCFImporter(config).import_mjcf()
    out["importer"] = ext_id("isaacsim.asset.importer.mjcf")
    out["output"] = str(usd)
    APP.update()
    stage = Usd.Stage.Open(str(usd))
    lines = []
    for prim in stage.Traverse():
        kind, notes = prim.GetTypeName(), []
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            m = UsdPhysics.MassAPI(prim)
            notes.append(f"rigid body, mass {m.GetMassAttr().Get() if m else '-'}")
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            notes.append("collider")
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                notes.append(f"approximation {UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()}")
        if prim.IsA(UsdPhysics.Joint):
            j = UsdPhysics.Joint(prim)
            notes.append(f"joint {kind}, body0 {j.GetBody0Rel().GetTargets()}, "
                         f"body1 {j.GetBody1Rel().GetTargets()}")
            for attr in ("physics:axis", "physics:lowerLimit", "physics:upperLimit"):
                a = prim.GetAttribute(attr)
                if a and a.HasAuthoredValue():
                    notes.append(f"{attr.split(':')[1]} {a.Get()}")
            for name in ("linear", "angular"):
                if prim.HasAPI(UsdPhysics.DriveAPI, name):
                    dr = UsdPhysics.DriveAPI(prim, name)
                    notes.append(f"drive {name}: stiffness {dr.GetStiffnessAttr().Get()}, "
                                 f"damping {dr.GetDampingAttr().Get()}, max force "
                                 f"{dr.GetMaxForceAttr().Get()}")
        if prim.HasAPI(UsdPhysics.MaterialAPI):
            mat = UsdPhysics.MaterialAPI(prim)
            notes.append(f"material: static {mat.GetStaticFrictionAttr().Get()}, dynamic "
                         f"{mat.GetDynamicFrictionAttr().Get()}")
        if prim.HasAPI(PhysxSchema.PhysxSceneAPI) or kind == "PhysicsScene":
            notes.append("physics scene")
        if notes:
            lines.append(f"{prim.GetPath()}: " + "; ".join(notes))
    with open(os.path.join(OUT, "import-mjcf.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"MJCF importer on {src}\noutput {usd}\n\n" + "\n".join(lines) + "\n")
    out["prims with physics"] = str(len(lines))


# ----------------------------------------------------------------------------
def write_record():
    rec = FACTS.get("record", {})
    model = FACTS.get("the model as PhysX reports it", {})
    loop = FACTS.get("the configuration of Chapter 3", {})
    s, d = SETTINGS, DESIGN
    lines = [
        "Record of the experiment of Section 11.2: the configuration of Chapter 3 "
        "with the simulator adapter cart.isaac",
        "",
        "model and asset provenance",
        "  USD stage written by cart_isaac/stage.py from the design of Section 11.1 "
        f"(cart.usda); no external assets; stage text SHA-256 {model.get('stage text, SHA-256')}...",
        "  design (SI units): " + ", ".join(f"{k} = {v}" for k, v in d.as_dict().items()),
        "  Isaac Sim settings: " + ", ".join(f"{k} = {v}" for k, v in s.as_dict().items()),
        "simulator and solver",
        f"  Isaac Sim {rec.get('Isaac Sim')}, Kit {rec.get('Kit')}, "
        f"{rec.get('extension omni.physx')}, Python and NumPy {rec.get('Python, NumPy')}",
        f"  PhysX {s.solver}, {s.position_iterations} position and {s.velocity_iterations} velocity "
        f"iterations, GPU dynamics {s.gpu_dynamics}, broadphase {s.broadphase}, friction "
        f"{s.friction_type}; contact offset {s.contact_offset} m, rest offset {s.rest_offset} m",
        "time steps",
        f"  physics {model.get('physics time step [s]')} s; governor period 0.1 s = "
        f"{model.get('steps per period')} steps, counted; drive loop at every step; the "
        "comparison repeated at 0.5 and 0.25 ms",
        "actuators and sensors",
        f"  each wheel: joint effort, limit {d.drive_limit:g} N m; drive loop: feedforward and "
        f"speed feedback (gain {d.speed_gain:g} 1/s), shared by measured load; tool: linear drive "
        f"stiffness {d.tool_kp:g} N/m, damping {d.tool_kv:g} N s/m, limit {d.tool_force:g} N",
        "  sensors: link poses and velocities, joint positions and velocities, contact forces "
        "of the wheels with the ground, a ray cast from the tool tip as rangefinder",
        "frames and units",
        "  SI units, metersPerUnit 1, kilogramsPerUnit 1, z up; state (p, v, l) as in "
        "Section 11.1",
        "uncertainty bounds",
        "  labels of cart.forecasting: braking of at least 2 m/s^2 on regions before 20 m, "
        "0.5 m/s^2 elsewhere; measurement bound 0",
        "constraints",
        "  p + l <= 25 m; |u_a| <= 2 m/s^2; u_l in {0, 0.8} m; in the simulator: contacts, the "
        "tool's limits 0 to 0.8 m",
        "terminal behaviour",
        "  rest feedback after an alarm, as in Section 11.1",
        "training and evaluation split",
        "  none: no module of this configuration learns",
        "seeds",
        "  none: the configuration uses no random numbers",
        "hardware",
        f"  {rec.get('OS, machine')}, {rec.get('CPU')}; GPU {rec.get('GPU (name, driver, memory)')}"
        " (physics on the CPU)",
        "guarantee status",
        f"  first alarm at {loop.get('first alarm [s]')} s, raised by a distance of "
        f"{loop.get('distance that raised it, in the norm of the state')}; restarts: "
        f"{loop.get('restarts')}; least clearance {loop.get('least clearance [m]')} m",
    ]
    with open(os.path.join(OUT, "record.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def write_outputs():
    lines = [f"# Section 11.2: the cart in Isaac Sim; {time.time() - STARTED:.0f} s so far"]
    for name, items in FACTS.items():
        lines.append(f"[{name}]")
        for key, val in items.items():
            lines.append(f"  {key} = {val}")
    with open(os.path.join(OUT, "isaac-summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT, "errors.txt"), "w", encoding="utf-8") as fh:
        if not ERRORS:
            fh.write("no step failed\n")
        for key, tb in ERRORS:
            fh.write(f"=== {key}\n{tb}\n")
    write_record()


def main():
    os.makedirs(OUT, exist_ok=True)
    group("record", record_facts)
    group("diagnostics", diagnostics)
    group("variants", variants)
    group("the run of Chapter 3, period by period", agreement_facts)
    group("the same at smaller time steps", timestep_facts)
    group("the cases of Section 10.4", motion_cases_facts)
    group("the configuration of Chapter 3", closed_loop_facts)
    group("the MJCF importer", import_facts)
    say(f"done: {len(ERRORS)} steps failed; results in {OUT}")


if __name__ == "__main__":
    try:
        main()
    finally:
        write_outputs()
        APP.close()
