#!/usr/bin/env python3
"""
The Husky A300 in MuJoCo: the model, its record, the contact of its wheels
with the ground, and what the simulated robot can do on each ground.

Physical AI, Together, Chapter 13 (Section 13.1).

This script builds the Husky of the package husky from Clearpath's
description and
  * writes the record of the experiment: the versions, the settings of the
    simulator and those that differ from MuJoCo's defaults, the description
    and the hash of the URDF text that MuJoCo loads;
  * states the facts of the description that Section 13.1.2 quotes, with
    two departures that a simulator finds in it or makes of it: the track
    of the published description, and the mass that MuJoCo's defaults infer
    for the bumpers;
  * reproduces the contact that MuJoCo 3.16.0 gives a wheel placed so that
    it exactly touches the ground, and what it does in the first steps,
    with a cylinder on a box;
  * shows what MuJoCo's default contact parameters make of the skid-steered
    Husky, tests on a point mass the slip speed at which a sliding contact
    opens, and checks the declared parameters of the wheels' contacts
    against the steady motion that Coulomb friction gives for the model's
    own contacts and the creep that MuJoCo's soft friction predicts, with
    one parameter at a time changed;
  * measures what the simulated robot can do on each ground of the yard:
    braking through the A300's controller, turning, the route driven
    without feedback, the speed loops of the drivers and odometry;
  * repeats the stops, the spins and the turns at time steps of 1 and
    0.25 ms.

Usage
    python3 husky_mujoco.py            # writes results/
    python3 husky_mujoco.py --check    # also checks every value quoted in
                                       # Section 13.1

Outputs, in results/ next to this file
    record.txt              the record of the experiment
    husky_a300.urdf         the URDF text that MuJoCo loads (corrected track)
    mujoco-summary.txt      every value, by group; the values quoted in
                            Section 13.1 are checked with --check
    contact-onset.csv       the point mass pushed at Coulomb friction: least
                            and largest load against the slip speed
    spin-default.csv        the spin in place with MuJoCo's default contacts
                            and with the declared ones: yaw rate and load
    braking.csv             speed against time when braking on each ground

Requires Python 3.10 or later, MuJoCo 3.16.0 and NumPy (requirements.txt).
"""
import argparse
import csv
import hashlib
import math
import os
import platform
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mujoco                                     # noqa: E402
import numpy as np                                # noqa: E402

from husky import asset, coulomb, drive, model, sim, world    # noqa: E402

G = 9.81
GROUNDS = [(g.name, g.friction) for g in world.GROUNDS]
STEPS = (0.001, 0.0005, 0.00025)                  # time steps of the convergence check [s]


# ----------------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------------
def g3(x):
    return f"{x:.3g}"


def g4(x):
    return f"{x:.4g}"


def pct(x, digits=2):
    return f"{100.0 * x:+.{digits}f} %"


def ms(dt):
    return f"{1000 * dt:g} ms"


def husky(mu=None, contacts="declared", dt=model.TIMESTEP, **kw):
    """A Husky on a pad of friction mu (or in the yard for mu None)."""
    build = dict(ground=mu, obstacles=mu is None, contacts=contacts, timestep=dt)
    build.update({k: v for k, v in kw.items() if k in ("published", "bumper_mass", "torque", "kv")})
    spec = model.build(**build)
    tweak = kw.get("tweak")
    if tweak:
        tweak(spec)
    return sim.Husky(spec=spec, integral=kw.get("integral", True))


def all_wheels_down(h):
    return all(v > 0.0 for v in h.loads().values())


# ----------------------------------------------------------------------------
# The record
# ----------------------------------------------------------------------------
def record_facts():
    spec = model.build()
    m = spec.compile()
    urdf = asset.urdf_text()
    defaults = mujoco.MjModel.from_xml_string("<mujoco/>").opt
    differ, was = [], []
    if m.opt.timestep != defaults.timestep:
        differ.append("time step"); was.append(f"{defaults.timestep:g} s")
    if m.opt.integrator != defaults.integrator:
        differ.append("integrator"); was.append(mujoco.mjtIntegrator(defaults.integrator).name[6:].lower())
    if m.opt.cone != defaults.cone:
        differ.append("friction cone"); was.append(mujoco.mjtCone(defaults.cone).name[7:].lower())
    if m.opt.impratio != defaults.impratio:
        differ.append("impratio"); was.append(f"{defaults.impratio:g}")
    cfg = asset.controller()
    n_sensors = m.nsensor
    return {
        "MuJoCo": mujoco.__version__,
        "NumPy, Python": f"{np.__version__}, {platform.python_version()}",
        "time step [s]": f"{m.opt.timestep:g}",
        "integrator": mujoco.mjtIntegrator(m.opt.integrator).name[6:].lower(),
        "solver": mujoco.mjtSolver(m.opt.solver).name[6:].capitalize(),
        "friction cone": mujoco.mjtCone(m.opt.cone).name[7:].lower(),
        "iterations, tolerance": f"{m.opt.iterations}, {m.opt.tolerance:g}",
        "impratio, noslip iterations": f"{m.opt.impratio:g}, {m.opt.noslip_iterations}",
        "settings that differ from MuJoCo's defaults": ", ".join(differ),
        "MuJoCo's defaults for them": ", ".join(was),
        "wheel-ground pairs: solref, solimp, solreffriction":
            f"{model.TIMECONST:g} 1; {model.IMPEDANCE:g} {model.IMPEDANCE:g} 0.001 0.5 2; "
            f"-0 -{model.FRICTION_DAMPING * model.IMPEDANCE:g}",
        "friction damping b of the wheel-ground pairs [1/s]": f"{model.FRICTION_DAMPING:g}",
        "frictions of the grounds": ", ".join(f"{n} {mu:g}" for n, mu in GROUNDS),
        "bodies, joints, degrees of freedom": f"{m.nbody - 1}, {m.njnt}, {m.nv}",
        "geoms, contact pairs, actuators, sensors": f"{m.ngeom}, {m.npair}, {m.nu}, {n_sensors}",
        "mass [kg]": g4(float(m.body_mass.sum())),
        "description": asset.SOURCE,
        "URDF text loaded, SHA-256": hashlib.sha256(urdf.encode()).hexdigest()[:16],
        "controller: update rate [Hz], wheel separation x multiplier [m], wheel radius [m]":
            f"{asset.number(cfg['update_rate']):g}, "
            f"{asset.number(cfg['wheel_separation']):g} x {asset.number(cfg['wheel_separation_multiplier']):g}, "
            f"{asset.number(cfg['wheel_radius']):g}",
        "controller limits: speed, acceleration, deceleration [m/s, m/s^2]":
            f"{asset.number(cfg['linear.x.max_velocity']):g}, "
            f"{asset.number(cfg['linear.x.max_acceleration']):g}, "
            f"{-asset.number(cfg['linear.x.max_deceleration']):g}",
        "controller limits: yaw rate, yaw acceleration [rad/s, rad/s^2]":
            f"{asset.number(cfg['angular.z.max_velocity']):g}, "
            f"{asset.number(cfg['angular.z.max_acceleration']):g}",
        "drivers: proportional gain [N m s/rad], integral time [s], torque limit [N m]":
            f"{model.KV:g}, {model.TI:g}, {model.TORQUE:g}",
        "controller updates per step": f"one every {round(0.02 / m.opt.timestep)} steps",
    }


# ----------------------------------------------------------------------------
# The description
# ----------------------------------------------------------------------------
def settle(h, seconds=1.0):
    h.place()
    h.wheel_targets(0.0, 0.0)
    h.run(seconds)


def description_facts():
    v = asset.description_values()
    out = {}
    out["masses: chassis, each wheel [kg]"] = f"{v['masses']['chassis_link']:g}, {v['masses']['front_left_wheel_link']:g}"
    out["wheel cylinder: radius, width [m]"] = f"{v['wheel_radius']:g}, {v['wheel_width']:g}"
    out["properties of a300.urdf.xacro: wheel_base, wheel_track [m]"] = \
        f"{v['wheel_base_property']:g}, {v['wheel_track_property']:g}"
    out["for Gazebo, a wheel-slip system per wheel: wheels; slip compliance lateral, longitudinal; "
        "wheel normal force [N]"] = gazebo_wheel_slip()
    pub = husky(0.8, published=True)
    cor = husky(0.8)
    for name, h in (("published", pub), ("corrected", cor)):
        ys = {w: float(h.m.body_pos[h.wheel_body[w]][1]) for w in model.WHEELS}
        out[f"{name}: wheel centres, left and right of the centre line [m]"] = \
            f"{ys['front_left']:.4f}, {ys['front_right']:.4f}"
        out[f"{name}: track [m]"] = f"{ys['front_left'] - ys['front_right']:.4f}"
        settle(h)
        loads = h.loads()
        out[f"{name}: loads at rest, front and rear, left then right [N]"] = ", ".join(
            f"{loads[w]:.1f}" for w in ("front_left", "rear_left", "front_right", "rear_right"))
    out["corrected: total of the loads, weight [N]"] = \
        f"{sum(cor.loads().values()):.1f}, {cor.total_mass() * G:.1f}"
    xs = [float(cor.m.body_pos[cor.wheel_body[w]][0]) for w in ("front_left", "rear_left")]
    out["wheelbase [m]"] = f"{xs[0] - xs[1]:.3f}"
    bm = husky(0.8, bumper_mass=True)
    bumpers = [float(bm.m.body_mass[i]) for i in range(bm.m.nbody) if "bumper" in bm.m.body(i).name]
    out["mass that MuJoCo infers for each bumper [kg]; total mass [kg]"] = \
        f"{bumpers[0]:.3f}, {bumpers[1]:.3f}; {bm.total_mass():.3f}" if len(bumpers) == 2 else \
        f"bumpers as bodies: {len(bumpers)}; total {bm.total_mass():.3f}"
    if len(bumpers) != 2:
        # the bumpers are fused into base_link: the inferred mass is the difference
        out["mass that MuJoCo infers for each bumper [kg]; total mass [kg]"] = \
            f"{(bm.total_mass() - cor.total_mass()) / 2:.3f} each; {bm.total_mass():.3f}"
    out["model mass [kg]; manual, with the 40 Ah battery [kg]"] = f"{cor.total_mass():g}; 78.5"
    out["penetration at rest [mm]"] = f"{(model.rest_height(cor.m) - cor.d.qpos[2]) * 1e3:.3f}"
    out["clearance of the chassis at rest [mm]"] = clearance(cor)
    cs = coulomb.contacts_at_rest(cor)
    per = {w: [c for c in cs if c[0] == w] for w in model.WHEELS}
    out["contacts per wheel at rest; lateral offsets from the wheel centre [m]"] = \
        f"{len(per['front_left'])}; " + ", ".join(
            f"{c[2] - float(cor.m.body_pos[cor.wheel_body['front_left']][1]):+.4f}" for c in per["front_left"])
    return out


def lane_allowance(s, side, hi=0.5):
    """The largest offset of the robot's centre from the centre line at arc
    length s, to the left (side +1) or to the right (side -1), at which
    every point of its footprint, headed along the route, is in the lane
    (world.footprint_offset); by bisection to 1e-9 m."""
    x, y, heading = world.point(s)
    nx, ny = -math.sin(heading), math.cos(heading)
    lo = 0.0
    while hi - lo > 1e-9:
        d = 0.5 * (lo + hi)
        if world.footprint_offset(x + side * d * nx, y + side * d * ny,
                                  heading) <= world.LANE_HALF_WIDTH:
            lo = d
        else:
            hi = d
    return lo


def task_facts():
    """The lane's room for the robot's footprint (Section 13.1.1) and the
    torque that the manual's climbing ability asks (Section 13.1.3)."""
    out = {}
    length, width = world.FOOTPRINT
    R, h = world.RADIUS, world.LANE_HALF_WIDTH
    straight = min(lane_allowance(s, side) for s in (5.0, 20.0) for side in (1, -1))
    middle = world.STRAIGHT_1 + world.TURN / 2
    corner = math.sqrt((R + h) ** 2 - (length / 2) ** 2) - width / 2 - R
    out["lane: largest offset of the centre with the footprint headed along the route: on the "
        "straights; in the turn, outwards (to the right) and inwards [m]; outwards, from the "
        "outer corners, sqrt((R + h)^2 - (l/2)^2) - w/2 - R [m]"] = (
        f"{straight:.3f}; {lane_allowance(middle, -1):.3f}, {lane_allowance(middle, 1):.3f}; "
        f"{corner:.3f}")
    a = length / 2
    out["lane: outward offset allowed half a length before the turn, at its start, at its end, "
        "half a length after it [m]"] = ", ".join(
        f"{lane_allowance(s, -1):.3f}" for s in (world.STRAIGHT_1 - a, world.STRAIGHT_1,
                                                 world.STRAIGHT_1 + world.TURN,
                                                 world.STRAIGHT_1 + world.TURN + a))
    torque = lambda mass, r: (mass + 51.5) * G * math.sin(math.radians(30.0)) * r / 4   # noqa: E731
    hm = husky()
    m_model, r_model = hm.total_mass(), hm.wheel_radius()
    out["climbing the manual's grade of 30 deg with its all-terrain payload of 51.5 kg: torque "
        "at each wheel, with the manual's mass of 78.5 kg and effective tire radius of 0.1625 m, and with "
        "the model's mass and wheel radius [N m]; the model's mass [kg], wheel radius [m] and "
        "limit [N m]"] = (
        f"{torque(78.5, 0.1625):.1f}, {torque(m_model, r_model):.1f}; {m_model:g}, "
        f"{r_model:.4f}, {model.TORQUE:g}")
    return out


def gazebo_wheel_slip():
    """The parameters of Gazebo's WheelSlip system in the expanded URDF."""
    import xml.etree.ElementTree as ET
    robot = ET.parse(asset.URDF).getroot()
    wheels = robot.findall("gazebo/plugin/wheel")
    vals = {(w.findtext("slip_compliance_lateral"), w.findtext("slip_compliance_longitudinal"),
             w.findtext("wheel_normal_force")) for w in wheels}
    return f"{len(wheels)}; " + "; ".join(f"{a}, {b}; {c}" for a, b, c in sorted(vals))


def clearance(h):
    """The lowest point of the chassis' collision geometry above the ground."""
    lowest = math.inf
    for g in range(h.m.ngeom):
        if h.m.geom_bodyid[g] != h.base or h.m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = h.m.geom_dataid[g]
        a, n = h.m.mesh_vertadr[mesh], h.m.mesh_vertnum[mesh]
        verts = h.m.mesh_vert[a:a + n]
        world_v = (h.d.geom_xmat[g].reshape(3, 3) @ verts.T).T + h.d.geom_xpos[g]
        lowest = min(lowest, float(world_v[:, 2].min()))
    return f"{lowest * 1e3:.1f}"


# ----------------------------------------------------------------------------
# The touching start
# ----------------------------------------------------------------------------
def touching_facts():
    out = {}
    h = husky(0.8)
    h.place(depth=0.0)
    d = h.d
    ws = {g: w for w, g in h.wheel_geom.items()}
    normals = []
    for c in d.contact[:d.ncon]:
        w = ws.get(int(c.geom1), ws.get(int(c.geom2)))
        if w is None:
            continue
        sign = 1.0 if int(c.geom1) == h.wheel_geom[w] else -1.0    # normal from the wheel
        normals.append((w, sign * c.frame[2], c.dist))
    out["placed touching: contacts, normal from wheel to ground (z), distance [m]"] = \
        f"{len(normals)}, {g3(normals[0][1])}, {normals[0][2]:.1e}" if normals else "none"
    h.place(depth=1e-12)
    n2 = [c for c in d.contact[:d.ncon]]
    zs = sorted({round((1.0 if int(c.geom1) in ws else -1.0) * c.frame[2], 3) for c in n2})
    out["placed 1e-12 m lower: contacts, normal from wheel to ground (z)"] = \
        f"{len(n2)}, {', '.join(g3(z) for z in zs)}"
    cyl = cylinder_on_box()
    out["cylinder on a box, touching: contacts, normal from cylinder to box (z), distance [m]"] = cyl[""]
    out["the same, ccd_tolerance 1e-9"] = cyl["ccd_tolerance=\"1e-9\""]
    out["the same, nativeccd disabled"] = cyl["nativeccd"]
    out["the same, 1e-12 m lower"] = cyl["lower"]
    # what the reversed contacts do: the first step, and the settling
    first, settling = {}, {}
    for label, depth in (("touching", 0.0), ("0.1 mm lower", model.PLACE_DEPTH)):
        h = husky(0.8)
        h.place(depth=depth)
        h.wheel_targets(0.0, 0.0)
        z0 = float(h.d.qpos[2])
        h.step(1)                       # its contacts are those found at the placement
        first[label] = (sum(h.loads().values()), (z0 - float(h.d.qpos[2])) * 1e6)
        peak, lowest = 0.0, math.inf
        for _ in range(int(round(0.2 / h.m.opt.timestep))):
            h.step(1)
            peak = max(peak, sum(h.loads().values()))
            lowest = min(lowest, float(h.d.qpos[2]))
        settling[label] = (peak, (model.rest_height(h.m) - lowest) * 1e3)
    out["placed touching: load on the wheels in the first step [N]; fall of the robot in it [um]"] = \
        f"{first['touching'][0]:.1f}; {first['touching'][1]:.2f}"
    out["placed touching, and 0.1 mm lower: largest load on the wheels in the first 0.2 s [N]; "
        "deepest sink [mm]"] = \
        f"{settling['touching'][0]:.0f}, {settling['0.1 mm lower'][0]:.0f}; " \
        f"{settling['touching'][1]:.3f}, {settling['0.1 mm lower'][1]:.3f}"
    return out


def cylinder_on_box():
    """A cylinder of the wheels' size, its axis horizontal as the URDF
    rotates it, touching the top face of a box."""
    quat = "0.7071067811865476 0.7071067811865475 0 0"     # what MuJoCo makes of rpy = (pi/2, 0, 0)
    out = {}
    for key in ("", "ccd_tolerance=\"1e-9\"", "nativeccd", "lower"):
        opt = f"<option {key if key.startswith('ccd') else ''}>" + \
            ("<flag nativeccd=\"disable\"/>" if key == "nativeccd" else "") + "</option>"
        z = "0.1651" if key != "lower" else repr(0.1651 - 1e-12)
        xml = f"""<mujoco>{opt}<worldbody>
          <geom type="box" size="40 40 0.05" pos="0 0 -0.05"/>
          <body pos="0.256 0.2829 {z}" quat="{quat}"><freejoint/>
            <geom type="cylinder" size="0.1651 0.05715" mass="2.5"/></body>
        </worldbody></mujoco>"""
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        cs = [(c.frame[2] * (1.0 if int(c.geom1) == 1 else -1.0), c.dist) for c in d.contact[:d.ncon]]
        out[key] = (f"{len(cs)}, " + ", ".join(sorted({g3(z) for z, _ in cs})) +
                    f", {cs[0][1]:.1e}") if cs else "none"
    return out


# ----------------------------------------------------------------------------
# The contact of the wheels: MuJoCo's defaults, the condition, our parameters
# ----------------------------------------------------------------------------
def point_mass(mu, contacts, dt=model.TIMESTEP, damping=None):
    """A point mass of 20 kg with a sphere of radius 0.1 m on three slides,
    on a plane, through one explicit pair (declared or default parameters)."""
    if contacts == "declared":
        b = model.FRICTION_DAMPING if damping is None else damping
        params = (f'solref="{model.TIMECONST:g} 1" solimp="{model.IMPEDANCE:g} {model.IMPEDANCE:g} 0.001 0.5 2" '
                  f'solreffriction="-0 -{b * model.IMPEDANCE:g}"')
        imp = model.IMPRATIO
    else:
        params = ""
        imp = model.DEFAULT_IMPRATIO
    xml = f"""<mujoco><option timestep="{dt}" integrator="implicitfast" cone="elliptic"
        impratio="{imp:g}" solver="Newton" iterations="100" tolerance="1e-10"/>
      <worldbody><geom name="floor" type="plane" size="50 50 .1" contype="0" conaffinity="0"/>
        <body name="b" pos="0 0 0.1"><joint name="x" type="slide" axis="1 0 0"/>
          <joint name="y" type="slide" axis="0 1 0"/><joint name="z" type="slide" axis="0 0 1"/>
          <geom name="ball" type="sphere" size="0.1" mass="20" contype="0" conaffinity="0"/></body>
      </worldbody>
      <contact><pair geom1="floor" geom2="ball" condim="3" friction="{mu} {mu} 0.005 0.0001 0.0001" {params}/></contact>
    </mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    return m, mujoco.MjData(m)


def condition(m, d, mu):
    """The slip speed v* below which a sliding contact stays closed,
    (1 + mu^2 / impratio) R_n0 F / (mu b), for the first contact at rest:
    MuJoCo's own approximation of the inverse inertia along the normal
    (efc_diagA), normal force F and friction damping b, with the regularizer
    R_n0 = (1 - d0) / d0 * A_n of zero penetration, where the contact opens
    (d0 = solimp[0]). condition.at_rest holds the same v* with MuJoCo's
    regularizer at rest (efc_R) and condition.impedance the impedance at
    rest."""
    rows = [i for i in range(d.nefc) if d.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC]
    i0 = rows[0]
    con = d.contact[d.efc_id[i0]]
    d0 = con.solimp[0]
    r0 = (1.0 - d0) / d0 * d.efc_diagA[i0]
    force = d.efc_force[i0]
    bt = d.efc_KBIP[i0 + 1][1]
    factor = (1.0 + mu * mu / m.opt.impratio) / (mu * bt)
    condition.at_rest = factor * d.efc_R[i0] * force
    condition.impedance = float(d.efc_KBIP[i0][2])
    return factor * r0 * force, r0 * force, bt


def creep_prediction(m, d, push):
    """The creep of a robot at rest under a steady sideways push, if every
    contact sticks in MuJoCo's soft sense (the bottom zone of the elliptic
    cone): each friction force is the friction's residual b v over its
    regularizer R_n / impratio, so that v = push / (b impratio sum 1/R_n),
    from MuJoCo's own regularizers at rest."""
    ell = mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC
    normals = [i for i in range(d.nefc) if d.efc_type[i] == ell and
               (i == 0 or d.efc_type[i - 1] != ell or d.efc_id[i - 1] != d.efc_id[i])]
    bt = d.efc_KBIP[normals[0] + 1][1]
    return push / (bt * m.opt.impratio * sum(1.0 / d.efc_R[i] for i in normals))


def pushed(m, d, v, mu, window=0.25):
    """Push the settled point mass at mu m g from the slip speed v: the least
    and largest load over the window, after 0.05 s."""
    mujoco.mj_resetData(m, d)
    d.qpos[2] = -1e-4
    for _ in range(int(0.5 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    d.qvel[0] = v
    d.qfrc_applied[0] = mu * 20.0 * G
    lo, hi = math.inf, 0.0
    f = np.zeros(6)
    for k in range(int(window / m.opt.timestep)):
        mujoco.mj_step(m, d)
        if k * m.opt.timestep < 0.05:
            continue
        n = 0.0
        for i in range(d.ncon):
            mujoco.mj_contactForce(m, d, i, f)
            n += f[0]
        lo, hi = min(lo, n), max(hi, n)
    return lo, hi


def onset(mu, contacts, lo=1e-3, hi=8.0, damping=None):
    """The least slip speed at which the pushed point mass loses its contact
    at least once, by bisection to 0.1 %."""
    m, d = point_mass(mu, contacts, damping=damping)
    for _ in range(int(0.5 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    predicted, rf, bt = condition(m, d, mu)
    if pushed(m, d, hi, mu)[0] > 0.0:
        return predicted, None, rf, bt
    while hi / lo > 1.001:
        mid = math.sqrt(lo * hi)
        if pushed(m, d, mid, mu)[0] > 0.0:
            lo = mid
        else:
            hi = mid
    return predicted, math.sqrt(lo * hi), rf, bt


def spin_check(mu, rim, contacts="declared", tweak=None, dt=model.TIMESTEP, seconds=3.0, window=1.0,
               trace=None):
    """Spin in place with the rims at +-rim: the mean yaw rate over the last
    window against the Coulomb steady state for the model's own contacts."""
    h = husky(mu, contacts=contacts, tweak=tweak, dt=dt)
    cs = coulomb.contacts_at_rest(h)
    h.place()
    r = h.wheel_radius()
    h.wheel_targets(-rim / r, rim / r)
    n0 = int(round((seconds - window) / dt))
    for k in range(n0):
        h.step(1)
        if trace is not None and k % int(round(0.005 / dt)) == 0:
            trace.append((k * dt, h.body_velocity()[2], sum(h.loads().values())))
    n = int(round(window / dt))
    om, down = 0.0, 0
    rims = {w: 0.0 for w in model.WHEELS}
    angles = []
    for k in range(n):
        h.step(1)
        om += h.body_velocity()[2] / n
        down += all_wheels_down(h)
        for w, s in h.rim_speeds().items():
            rims[w] += s / n
        if trace is not None and k % int(round(0.005 / dt)) == 0:
            trace.append(((n0 + k) * dt, h.body_velocity()[2], sum(h.loads().values())))
        for cs_ in h.wheel_contacts().values():
            for _, f, v in cs_:
                ft, vt = f[:2], v[:2]
                if np.linalg.norm(vt) > 0.02 and np.linalg.norm(ft) > 1e-6:
                    c = float(np.dot(ft, -vt) / (np.linalg.norm(ft) * np.linalg.norm(vt)))
                    angles.append(math.degrees(math.acos(max(-1.0, min(1.0, c)))))
    q, _ = coulomb.steady(cs, rims, mu)
    spin_check.rims = sum(abs(v) for v in rims.values()) / 4.0
    spin_check.angle = float(np.median(angles)) if angles else float("nan")
    return om, q[2], down / n


def turn_check(mu, v=1.0, w=1.0 / 3.0, contacts="declared", tweak=None, dt=model.TIMESTEP,
               inertia=False):
    """The turn commanded at (v, w): its radius against the quasi-static
    Coulomb reference for the contacts at rest and the rims' measured speeds.
    With inertia=True, turn_check.dynamic also holds the radius of the
    reference with the centripetal force of the turn, for the loads at rest
    and for the loads measured in the turn (every 10 ms), and
    turn_check.loads the loads of the inner (left) and outer (right) wheels
    in the turn and at rest."""
    h = husky(mu, contacts=contacts, tweak=tweak, dt=dt)
    cs = coulomb.contacts_at_rest(h)
    com = coulomb.centre_of_mass(h)
    h.place()
    h.command = (v, w)
    h.run(6.0)
    n = int(round(2.0 / dt))
    every = int(round(0.01 / dt))
    acc = np.zeros(3)
    down = 0
    rims = {k: 0.0 for k in model.WHEELS}
    moving = coulomb.ContactsInMotion()
    for i in range(n):
        h.step(1)
        acc += np.array(h.body_velocity()) / n
        down += all_wheels_down(h)
        for k, s in h.rim_speeds().items():
            rims[k] += s / n
        if inertia and i % every == 0:
            moving.add(h)
    q, _ = coulomb.steady(cs, rims, mu, guess=(v, 0.0, w))
    turn_check.rims = (rims["front_left"] + rims["rear_left"]) / 2, (rims["front_right"] + rims["rear_right"]) / 2
    turn_check.motion = tuple(acc)
    if inertia:
        mass = h.total_mass()
        cs_turn = moving.mean()
        q_rest, _ = coulomb.steady(cs, rims, mu, guess=tuple(q), mass=mass, com=com)
        q_turn, _ = coulomb.steady(cs_turn, rims, mu, guess=tuple(q), mass=mass, com=com)
        turn_check.dynamic = (q_rest[0] / q_rest[2], q_turn[0] / q_turn[2])
        side = lambda c_, ws: sum(x[3] for x in c_ if x[0] in ws)     # noqa: E731
        turn_check.loads = (side(cs_turn, model.LEFT), side(cs_turn, model.RIGHT),
                            side(cs, model.LEFT), side(cs, model.RIGHT))
        turn_check.com = com
    return acc[0] / acc[2], q[0] / q[2], down / n


def slide_check(mu, contacts="declared", tweak=None, dt=model.TIMESTEP, v0=1.0, damping=None):
    """The robot slides sideways from v0 with its wheels held: the distance
    to rest against v0^2 / (2 mu g). Given the friction damping b of the
    contacts, slide_check.rate holds the rate at which the slip decays below
    mu g / b: the slope of the logarithm of the sideways speed, fitted by
    least squares to the steps in which the speed lies between 0.6 and 0.03
    times mu g / b."""
    h = husky(mu, contacts=contacts, tweak=tweak, dt=dt)
    h.place()
    h.d.qvel[1] = v0
    mujoco.mj_forward(h.m, h.d)
    h.wheel_targets(0.0, 0.0)
    x0, y0, _ = h.pose()
    t, down, n = 0.0, 0, 0
    tail = []
    lazy = None if damping is None else mu * G / damping
    while t < 5.0:
        h.step(1)
        t += dt
        n += 1
        down += all_wheels_down(h)
        sideways = h.body_velocity()[1]
        if lazy is not None and 0.03 * lazy < sideways < 0.6 * lazy:
            tail.append((t, math.log(sideways)))
        if math.hypot(*h.body_velocity()[:2]) < 1e-3:
            break
    x1, y1, _ = h.pose()
    slide_check.rate = float(-np.polyfit(*np.array(tail).T, 1)[0]) if len(tail) > 10 else None
    return math.hypot(x1 - x0, y1 - y0), v0 * v0 / (2.0 * mu * G), down / n


def lazy_excess(mu, rate, v0=1.0):
    """The excess over v0^2 / (2 mu g) of a slide that decelerates at mu g
    down to mu g / rate and at rate times the speed below it."""
    vc = mu * G / rate
    if vc >= v0:
        return (v0 / rate) / (v0 * v0 / (2 * mu * G)) - 1
    return ((v0 * v0 - vc * vc) / (2 * mu * G) + vc / rate) / (v0 * v0 / (2 * mu * G)) - 1


def creep_check(mu, contacts="declared", tweak=None, dt=model.TIMESTEP, share=0.3):
    """Push the robot at rest sideways, at its centre of mass, with a share
    of mu m g: its speed. creep_check.predicted holds the creep that the
    bottom zone of the elliptic cone predicts (creep_prediction)."""
    h = husky(mu, contacts=contacts, tweak=tweak, dt=dt)
    settle(h)
    push = share * mu * h.total_mass() * G
    creep_check.predicted = creep_prediction(h.m, h.d, push)
    y0 = h.pose()[1]
    h.d.xfrc_applied[h.base, 1] = push
    h.run(2.0)
    return abs(h.pose()[1] - y0) / 2.0


def with_impedance(d):
    def tweak(spec):
        for p in spec.pairs:
            if p.name.startswith("ground |"):
                p.solimp = [d, d, 0.001, 0.5, 2.0]
                p.solreffriction = [-0.0, -model.FRICTION_DAMPING * d]
    return tweak


def with_impratio(i):
    def tweak(spec):
        spec.option.impratio = i
    return tweak


def with_damping(b):
    def tweak(spec):
        for p in spec.pairs:
            if p.name.startswith("ground |"):
                p.solreffriction = [-0.0, -b * model.IMPEDANCE]
    return tweak


def contact_facts(out_dir):
    out = {}
    # the condition, on a point mass
    rows = []
    for label, mu, contacts, damping in (("defaults, mu 0.8", 0.8, "default", None),
                                          ("defaults, mu 0.35", 0.35, "default", None),
                                          ("declared, mu 0.8", 0.8, "declared", None),
                                          ("declared with b 600 1/s, mu 0.8", 0.8, "declared", 600.0)):
        pred, meas, rf, bt = onset(mu, contacts, damping=damping)
        out[f"point mass, {label}: R_n0 F [m/s^2], b [1/s]"] = f"{g3(rf)}, {g4(bt)}"
        out[f"point mass, {label}: v* predicted, contact first lost at [m/s]"] = \
            f"{g3(pred)}, " + (g3(meas) if meas else "not below 8")
        if label == "defaults, mu 0.8":
            out[f"point mass, {label}: impedance at rest; v* with the regularizer at rest [m/s]"] = \
                f"{condition.impedance:.3f}; {g3(condition.at_rest)}"
    m, d = point_mass(0.8, "default")
    md, dd = point_mass(0.8, "declared")
    for _ in range(1000):
        mujoco.mj_step(m, d)
        mujoco.mj_step(md, dd)
    for v in (0.003, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        a, b = pushed(m, d, v, 0.8)
        c, e = pushed(md, dd, v, 0.8)
        rows.append(dict(v=v, default_min=a, default_max=b, declared_min=c, declared_max=e))
    write_csv(os.path.join(out_dir, "contact-onset.csv"), rows)
    at1 = [r for r in rows if r["v"] == 1.0][0]
    out["point mass at 1 m/s, defaults: load from, to [N] (weight 196)"] = \
        f"{at1['default_min']:.0f}, {at1['default_max']:.0f}"
    out["point mass at 1 m/s, declared: load from, to [N]"] = \
        f"{at1['declared_min']:.1f}, {at1['declared_max']:.1f}"

    # the Husky's contacts: the condition with MuJoCo's own values
    for contacts in ("default", "declared"):
        for name, mu in GROUNDS:
            h = husky(mu, contacts=contacts)
            settle(h)
            pred, rf, bt = condition(h.m, h.d, mu)
            out[f"Husky, {contacts} contacts, {name}: R_n0 F [m/s^2], b [1/s], v* [m/s]"] = \
                f"{g3(rf)}, {g4(bt)}, {g3(pred)}"
            if contacts == "declared":
                # the slip speed at which a steadily loaded contact reaches
                # Coulomb's force: the edge of the bottom zone, mu R_n F / (I b)
                out[f"Husky, declared contacts, {name}: slip speed at which a steady load reaches "
                    f"mu F, mu R_n F / (impratio b) [mm/s]; mu g / b [m/s]"] = \
                    f"{mu * rf / (h.m.opt.impratio * bt) * 1e3:.1f}; {mu * G / bt:.3f}"
    # what the defaults make of the robot
    trace_d, trace_o = [], []
    for name, mu in (GROUNDS[0], GROUNDS[2]):
        om, ref, down = spin_check(mu, 0.49, "default", trace=trace_d if mu == 0.8 else None)
        out[f"defaults, {name}: spin, rims commanded at 0.49 m/s: mean rim speed [m/s], yaw rate, "
            f"Coulomb for those rims [rad/s], all wheels down"] = \
            f"{g3(spin_check.rims)}, {g3(om)}, {g3(ref)}, {down:.0%}"
        out[f"defaults, {name}: spin: median angle between a contact's friction and its slip [deg]"] = \
            f"{spin_check.angle:.1f}"
        R, Rc, down = turn_check(mu, contacts="default")
        out[f"defaults, {name}: turn at 1 m/s and 1/3 rad/s: rims left, right [m/s], radius, "
            f"Coulomb [m], all wheels down"] = \
            f"{g3(turn_check.rims[0])}, {g3(turn_check.rims[1])}, {g3(R)}, {g3(Rc)}, {down:.0%}"
        dist, ideal, down = slide_check(mu, "default")
        out[f"defaults, {name}: sideways slide from 1 m/s: distance, v^2/(2 mu g) [m], difference, all wheels down"] = \
            f"{dist:.4f}, {ideal:.4f}, {pct(dist / ideal - 1)}, {down:.0%}"
    spin_check(0.8, 0.49, "declared", trace=trace_o)
    rows = [dict(t=a[0], yaw_default=a[1], load_default=a[2], yaw_declared=b[1], load_declared=b[2])
            for a, b in zip(trace_d, trace_o)]
    write_csv(os.path.join(out_dir, "spin-default.csv"), rows)
    tail = [r for r in rows if r["t"] >= 1.0]
    out["defaults, asphalt: spin, load on the ground from, to [N] (weight 785)"] = \
        f"{min(r['load_default'] for r in tail):.0f}, {max(r['load_default'] for r in tail):.0f}"

    # the declared parameters, on each ground
    for name, mu in GROUNDS:
        h = husky(mu)
        settle(h)
        cr = creep_check(mu)
        out[f"declared, {name}: creep under a sideways push of 30 % of mu m g, measured, "
            f"predicted by the bottom zone [mm/s]"] = \
            f"{cr * 1e3:.2f}, {creep_check.predicted * 1e3:.2f}"
        for rim in (0.49, 0.98):
            om, ref, down = spin_check(mu, rim)
            out[f"declared, {name}: spin, rims at {rim} m/s: yaw rate, Coulomb [rad/s], difference, all wheels down"] = \
                f"{om:.4f}, {ref:.4f}, {pct(om / ref - 1)}, {down:.0%}"
            if rim == 0.49:
                out[f"declared, {name}: spin: median angle between a contact's friction and its slip [deg]"] = \
                    f"{spin_check.angle:.1f}"
        R, Rc, down = turn_check(mu, inertia=True)
        out[f"declared, {name}: turn at 1 m/s and 1/3 rad/s: radius, Coulomb [m], difference, all wheels down"] = \
            f"{R:.3f}, {Rc:.3f}, {pct(R / Rc - 1)}, {down:.0%}"
        u_, _, w_ = turn_check.motion
        out[f"declared, {name}: turn: centripetal acceleration u Omega [m/s^2], share of mu g; "
            f"loads of the inner and outer wheels in the turn, and at rest [N]"] = (
            f"{u_ * w_:.3f}, {u_ * w_ / (mu * G):.1%}; "
            + ", ".join(f"{x:.1f}" for x in turn_check.loads))
        Rr, Rt = turn_check.dynamic
        out[f"declared, {name}: turn: radius against the Coulomb reference with the centripetal "
            f"force, for the loads at rest and for the loads of the turn [m], differences"] = (
            f"{Rr:.3f}, {Rt:.3f}; {pct(R / Rr - 1)}, {pct(R / Rt - 1)}")
        dist, ideal, down = slide_check(mu, damping=model.FRICTION_DAMPING)
        out[f"declared, {name}: sideways slide from 1 m/s: distance, v^2/(2 mu g) [m], difference, all wheels down"] = \
            f"{dist:.4f}, {ideal:.4f}, {pct(dist / ideal - 1)}, {down:.0%}"
        rate = slide_check.rate
        out[f"declared, {name}: sideways slide: decay rate of the slip below mu g / b [1/s], share "
            f"of b; excess over v^2/(2 mu g) of a slide at mu g down to mu g / b and at b v below, "
            f"and with the decay rate for b"] = (
            f"{rate:.2f}, {rate / model.FRICTION_DAMPING:.3f}; "
            f"{pct(lazy_excess(mu, model.FRICTION_DAMPING), 1)}, {pct(lazy_excess(mu, rate), 1)}")
    # one parameter at a time changed
    for label, tweak in (("impedance 0.9 (with b 60 1/s)", with_impedance(0.9)),
                         ("impratio 1", with_impratio(1.0)),
                         ("friction damping 6 1/s", with_damping(6.0))):
        om, ref, down = spin_check(0.8, 0.98, tweak=tweak)
        b = 6.0 if label == "friction damping 6 1/s" else model.FRICTION_DAMPING
        dist, ideal, sdown = slide_check(0.8, tweak=tweak, damping=b)
        cr = creep_check(0.8, tweak=tweak)
        out[f"declared but {label}, asphalt: spin at 0.98 m/s against Coulomb, all wheels down; "
            f"slide against v^2/(2 mu g); creep [mm/s]"] = \
            f"{pct(om / ref - 1, 1)}, {down:.0%}; {pct(dist / ideal - 1, 1)}; {cr * 1e3:.2f}"
        if label == "friction damping 6 1/s":
            rate = slide_check.rate
            out["declared but friction damping 6 1/s, asphalt: slide from 1 m/s, entirely below "
                "mu g / b: decay rate of the slip [1/s], share of b; v0 / b, v0 / rate, the "
                "slide [m]; mu g / b [m/s]"] = (
                f"{rate:.3f}, {rate / b:.3f}; {1.0 / b:.4f}, {1.0 / rate:.4f}, {dist:.4f}; "
                f"{0.8 * G / b:.2f}")
    return out


# ----------------------------------------------------------------------------
# What the simulated robot can do on each ground
# ----------------------------------------------------------------------------
def ramp_distance(u0, v0=1.0):
    """The distance that a robot whose wheels followed the controller's
    commands exactly, without slip, would cover after the stop command: the
    A300's drive controller brings its command from v0 to rest, one update
    at a time, and the robot, which moves at u0 under the command v0, covers
    u0 / v0 times the distance commanded."""
    dr = drive.A300Drive()
    dr.previous.clear()
    dr.previous.extend([(v0, 0.0), (v0, 0.0)])
    commanded = 0.0
    while True:
        dr.update(0.0, 0.0, 0.0, 0.0)
        v = dr.command[0]
        commanded += v * dr.period
        if v <= 0.0:
            break
    return commanded * u0 / v0


def brake(mu, dt=model.TIMESTEP, v0=1.0, trace=None):
    """Drive at v0 through the controller, then command a stop: the
    distance and time from the command to rest (below 1 mm/s) and the
    largest slip of the robot over its rims. brake.rolled holds the
    distance that the wheels' turning accounts for, at the rolling radius
    of the steady drive before the stop (u0 over the wheels' mean speed);
    the rest of the distance is slip."""
    h = husky(mu, dt=dt)
    h.place()
    h.command = (v0, 0.0)
    h.run(3.0)
    x0 = h.pose()[0]
    u0 = h.body_velocity()[0]
    radius = u0 / (sum(h.wheel_speeds().values()) / 4.0)
    h.command = (0.0, 0.0)
    t, slip, rolled = 0.0, 0.0, 0.0
    while t < 5.0:
        h.step(1)
        t += dt
        u = h.body_velocity()[0]
        rim = sum(h.rim_speeds().values()) / 4.0
        slip = max(slip, u - rim)
        rolled += radius * sum(h.wheel_speeds().values()) / 4.0 * dt
        if trace is not None and round(t / dt) % round(0.005 / dt) == 0:
            trace.append((t, u))
        if t > 0.1 and abs(u) < 1e-3:
            break
    brake.rolled = rolled
    return u0, h.pose()[0] - x0, t, slip


def capability_facts(out_dir):
    out = {}
    traces = {}
    decel = -asset.number(asset.controller()["linear.x.max_deceleration"])
    for name, mu in GROUNDS:
        tr = []
        u0, dist, t, slip = brake(mu, trace=tr)
        traces[name] = tr
        out[f"{name}: deceleration of the controller's ramp at the ground, {decel:g} u0 / v0; mu g [m/s^2]"] = \
            f"{decel * u0:.3g}; {mu * G:.3g}"
        out[f"{name}: stop from the command at 1 m/s: speed before [m/s], distance [m], time to rest [s], largest slip [m/s]"] = \
            f"{u0:.4f}, {dist:.4f}, {t:.3f}, {slip:.3f}"
        ideal = ramp_distance(u0)
        out[f"{name}: wheels that followed the controller's commands exactly [m]; u0^2 / (2 mu g) [m]"] = \
            f"{ideal:.4f}; {u0 * u0 / (2 * mu * G):.4f}"
        out[f"{name}: stop beyond the wheels that followed the commands: by the wheels' turning, "
            f"by slip [mm]"] = f"{(brake.rolled - ideal) * 1e3:.1f}, {(dist - brake.rolled) * 1e3:.1f}"
    rows = []
    longest = max(traces.values(), key=len)     # its times cover every row
    for k in range(len(longest)):
        row = {"t": longest[k][0]}
        for name, _ in GROUNDS:
            tr = traces[name]
            row[name.replace(" ", "_")] = tr[k][1] if k < len(tr) else 0.0
        rows.append(row)
    write_csv(os.path.join(out_dir, "braking.csv"), rows)

    sep = drive.A300Drive().separation
    for name, mu in GROUNDS:
        for w in (1.0, 2.0):
            h = husky(mu)
            h.place()
            h.command = (0.0, w)
            h.run(3.0)
            n_ = int(round(1.0 / h.m.opt.timestep))
            om, tq = 0.0, 0.0
            for _ in range(n_):
                h.step(1)
                om += h.body_velocity()[2] / n_
                tq = max(tq, max(abs(x) for x in h.torques().values()))
            out[f"{name}: spin commanded at {w:g} rad/s: yaw rate [rad/s], ratio, W_eff [m], largest torque [N m]"] = \
                f"{om:.4f}, {om / w:.4f}, {sep * w / om:.4f}, {tq:.1f}"
        R, Rc, _ = turn_check(mu)
        out[f"{name}: turn commanded at 1 m/s and 1/3 rad/s (radius 3 m): radius [m]"] = f"{R:.3f}"
    # through the yard without feedback, switching on the odometry
    out.update(open_loop_route())
    # the speed of the robot and the odometry, straight, on asphalt
    h = husky(0.8)
    h.place()
    h.command = (1.0, 0.0)
    h.run(2.0)
    x0 = h.pose()[0]
    o0 = h.drive.x
    h.run(10.0)
    true = h.pose()[0] - x0
    odo = h.drive.x - o0
    out["asphalt: speed at the command 1 m/s [m/s]; rolling radius [m]"] = \
        f"{true / 10.0:.4f}; {true / 10.0 * h.drive.radius_left:.5f}"
    out["asphalt: 10 s at 1 m/s: distance, odometry [m], odometry short by"] = \
        f"{true:.4f}, {odo:.4f}, {pct(1 - odo / true)}"
    # the drivers' speed loops while the controller ramps to 1 m/s
    h = husky(0.8)
    h.place()
    h.command = (1.0, 0.0)
    err = 0.0
    for _ in range(int(round(1.0 / h.m.opt.timestep))):
        h.step(1)
        err = max(err, max(abs(h.target[w] - h.d.qvel[h.vadr[w]]) for w in model.WHEELS))
    out["asphalt: largest speed error of a wheel while ramping to 1 m/s at 2 m/s^2 [rad/s]"] = f"{err:.3f}"
    # heading odometry over a quarter turn
    h = husky(0.6)
    h.place()
    h.command = (1.0, 1.0 / 3.0)
    h.run(1.0)
    th0, od0 = h.pose()[2], h.drive.heading
    h.run(4.0)
    turned, counted = h.pose()[2] - th0, h.drive.heading - od0
    out["gravel: 4 s at 1 m/s and 1/3 rad/s: heading change, odometry [rad]; separation that "
        "would give the heading change [m]"] = \
        f"{turned:.4f}, {counted:.4f}; {sep * counted / turned:.4f}"
    return out


def open_loop_route(dt=model.TIMESTEP):
    """Through the yard without feedback: straight at 1 m/s until the
    odometry reads 10 m, the turn's command (1 m/s, 1/3 rad/s) until it reads
    a quarter turn, then straight on, until s = 23 m: the largest offset of
    the robot's centre from the centre line in the turn and after it, where
    its footprint first leaves the lane (world.footprint_offset, checked
    every millisecond once the centre is 0.09 m off the centre line, less
    than the lane's half width minus the footprint's half diagonal), and
    whether it touches a stack of pallets."""
    h = husky(None, dt=dt)
    h.place()
    h.command = (1.0, 0.0)
    phase, t = 0, 0.0
    worst = [0.0, 0.0, 0.0]
    left_lane, touched = None, None
    every = max(1, int(round(0.001 / dt)))
    obstacles = {h.m.geom(o.name).id: o.name for o in world.OBSTACLES}
    k = 0
    while t < 40.0:
        h.step(1)
        t += dt
        k += 1
        x, y, heading = h.pose()
        s_, d_ = world.project(x, y)
        if phase == 0 and h.drive.x >= world.STRAIGHT_1:
            h.command, phase = (1.0, 1.0 / 3.0), 1
        elif phase == 1 and h.drive.heading >= 0.5 * math.pi:
            h.command, phase = (1.0, 0.0), 2
        worst[phase] = max(worst[phase], abs(d_))
        if left_lane is None and abs(d_) > 0.09 and k % every == 0:
            if world.footprint_offset(x, y, heading) > world.LANE_HALF_WIDTH:
                left_lane = (s_, d_)
        if touched is None:
            for c in h.d.contact[:h.d.ncon]:
                for g in (int(c.geom1), int(c.geom2)):
                    if g in obstacles:
                        touched = (obstacles[g], s_)
        if s_ >= 23.0:
            break
    return {
        "open loop, switching on the odometry: largest offset in the turn, after it [m]":
            f"{worst[1]:.3f}, {worst[2]:.3f}",
        "open loop: the footprint first leaves the lane at s [m]; the centre's offset then [m]":
            f"{left_lane[0]:.2f}; {left_lane[1]:+.3f}" if left_lane else "never",
        "open loop: touches": f"{touched[0]} at s = {touched[1]:.2f} m" if touched else "nothing",
    }


# ----------------------------------------------------------------------------
# Time steps
# ----------------------------------------------------------------------------
def timestep_facts():
    out = {}
    for name, mu in GROUNDS:
        vals = [brake(mu, dt=dt)[1] for dt in STEPS]
        out[f"{name}: stop from 1 m/s, distance at 1, 0.5, 0.25 ms [m]"] = ", ".join(f"{v:.4f}" for v in vals)
    for name, mu in GROUNDS:
        vals = [spin_check(mu, 0.98, dt=dt) for dt in STEPS]
        out[f"{name}: spin at 0.98 m/s against Coulomb at 1, 0.5, 0.25 ms"] = \
            ", ".join(pct(om / ref - 1) for om, ref, _ in vals)
        vals = [turn_check(mu, dt=dt) for dt in STEPS]
        out[f"{name}: turn at 1, 0.5, 0.25 ms: radius [m]; against Coulomb"] = \
            ", ".join(f"{R:.3f}" for R, _, _ in vals) + "; " + \
            ", ".join(pct(R / Rc - 1) for R, Rc, _ in vals)
    return out


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


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
        "Record of the experiment of Section 13.1: the Husky A300 in MuJoCo, its contacts "
        "and its capability on the grounds of the yard",
        "",
        "model and asset provenance",
        "  " + rec["description"],
        "  expanded by xacro 2.1.1 into description/husky_a300.urdf (options: " + asset.OPTIONS + ")",
        "  loaded as asset.urdf_text gives it: visual elements and the collision meshes of "
        "massless links removed; the right beam's wheel mounts mirrored (track 0.5658 m, "
        "published 0.5468 m); SHA-256 of the text " + rec["URDF text loaded, SHA-256"] + "...",
        "  added by husky/model.py: free joint, no inferred bumper mass, wheel actuators, "
        "sensors, contact pairs, the yard",
        "simulator and solver",
        f"  MuJoCo {rec['MuJoCo']}, NumPy and Python {rec['NumPy, Python']}",
        "  " + "; ".join(f"{k} = {rec[k]}" for k in (
            "integrator", "solver", "friction cone", "iterations, tolerance",
            "impratio, noslip iterations")),
        "  wheel-ground pairs: solref, solimp, solreffriction = " +
        rec["wheel-ground pairs: solref, solimp, solreffriction"] +
        " (friction damping " + rec["friction damping b of the wheel-ground pairs [1/s]"] + " 1/s)",
        "  other contacts: MuJoCo's defaults (solref 0.02 1, solimp 0.9 0.95 0.001 0.5 2)",
        "  settings that differ from MuJoCo's defaults: " +
        rec["settings that differ from MuJoCo's defaults"] + " (defaults: " +
        rec["MuJoCo's defaults for them"] + ")",
        "time steps",
        f"  simulator {rec['time step [s]']} s; drive controller at 50 Hz, " +
        rec["controller updates per step"] + "; drivers' speed loops at every step; "
        "the stops, spins and turns repeated at 1 and 0.25 ms",
        "actuators and sensors",
        "  drive controller: diff_drive_controller (ros2_controllers, jazzy, d71703b) as "
        "ported in husky/drive.py, configuration diff_4wd.yaml: " +
        rec["controller: update rate [Hz], wheel separation x multiplier [m], wheel radius [m]"],
        "  drivers (ours): speed loop, proportional gain, integral time and torque limit " +
        rec["drivers: proportional gain [N m s/rad], integral time [s], torque limit [N m]"],
        "  sensors: four wheel encoders (positions, speeds), a 6-axis IMU at the chassis' "
        "centre (ours: the description places none); the chassis pose from the simulator",
        "frames and units",
        "  SI units; world x east, y north, z up, ground at z = 0; base_link x forward, "
        "y left; route arc length s, lateral offset d positive to the left",
        "uncertainty bounds",
        "  frictions of the grounds (the simulator's, not measured): " +
        rec["frictions of the grounds"] + "; no labels yet (Section 13.2)",
        "constraints",
        "  lane 1.4 m wide centred on the route, to hold the robot's footprint (0.990 m x 0.698 m); "
        "pallets 0.1 m outside its edges; the dock's "
        "face at s = 25 m; speed at most 1 m/s on the route; controller limits " +
        rec["controller limits: speed, acceleration, deceleration [m/s, m/s^2]"],
        "terminal behaviour",
        "  a stop commanded through the controller (deceleration limit 4 m/s^2)",
        "training and evaluation split",
        "  none: nothing learns in Section 13.1",
        "seeds",
        "  none: no random numbers",
        "hardware",
        f"  {platform.system()} {platform.machine()}, {hardware()}",
        "guarantee status",
        "  none: Section 13.1 states the task and measures the simulated robot; no plan is "
        "certified",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Section 13.1: the Husky A300 in MuJoCo")
    ap.add_argument("--check", action="store_true", help="check every value quoted in Section 13.1")
    args = ap.parse_args()
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    asset.write_urdf(os.path.join(out_dir, "husky_a300.urdf"), meshes="package")
    facts = {"record": record_facts(),
             "the task": task_facts(),
             "the description": description_facts(),
             "the touching start": touching_facts(),
             "the contact of the wheels": contact_facts(out_dir),
             "what the simulated robot can do": capability_facts(out_dir),
             "time steps": timestep_facts()}
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
        lines.append(f"[check] {n} values, among them every value that Section 13.1 quotes: " +
                     ("all as expected" if not failures else "FAILED: " + "; ".join(failures)))
    with open(os.path.join(out_dir, "mujoco-summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if failures:
        raise SystemExit(1)


EXPECTED = {   # every value of the summary but the versions of NumPy and Python
    'record': {
        'MuJoCo':
            '3.16.0',
        'time step [s]':
            '0.0005',
        'integrator':
            'implicitfast',
        'solver':
            'Newton',
        'friction cone':
            'elliptic',
        'iterations, tolerance':
            '100, 1e-08',
        'impratio, noslip iterations':
            '100, 0',
        "settings that differ from MuJoCo's defaults":
            'time step, integrator, friction cone, impratio',
        "MuJoCo's defaults for them":
            '0.002 s, euler, pyramidal, 1',
        'wheel-ground pairs: solref, solimp, solreffriction':
            '0.008 1; 0.05 0.05 0.001 0.5 2; -0 -3',
        'friction damping b of the wheel-ground pairs [1/s]':
            '60',
        'frictions of the grounds':
            'asphalt 0.8, gravel 0.6, wet grass 0.35',
        'bodies, joints, degrees of freedom':
            '5, 5, 10',
        'geoms, contact pairs, actuators, sensors':
            '15, 20, 4, 10',
        'mass [kg]':
            '80',
        'description':
            'clearpath_common, https://github.com/clearpathrobotics/clearpath_common, commit 78e3677412f1fed881f6436cdaf404f04f28ee4f (19 October 2026), clearpath_platform_description and clearpath_control 2.9.18, BSD 3-Clause',
        'URDF text loaded, SHA-256':
            'd7b2542c25c84fed',
        'controller: update rate [Hz], wheel separation x multiplier [m], wheel radius [m]':
            '50, 0.562 x 1.75, 0.1625',
        'controller limits: speed, acceleration, deceleration [m/s, m/s^2]':
            '2, 2, 4',
        'controller limits: yaw rate, yaw acceleration [rad/s, rad/s^2]':
            '2, 4',
        'drivers: proportional gain [N m s/rad], integral time [s], torque limit [N m]':
            '40, 0.1, 30',
        'controller updates per step':
            'one every 40 steps',
    },
    'the task': {
        'lane: largest offset of the centre with the footprint headed along the route: on the straights; in the turn, outwards (to the right) and inwards [m]; outwards, from the outer corners, sqrt((R + h)^2 - (l/2)^2) - w/2 - R [m]':
            '0.351; 0.318, 0.351; 0.318',
        'lane: outward offset allowed half a length before the turn, at its start, at its end, half a length after it [m]':
            '0.351, 0.318, 0.318, 0.351',
        "climbing the manual's grade of 30 deg with its all-terrain payload of 51.5 kg: torque at each wheel, with the manual's mass of 78.5 kg and effective tire radius of 0.1625 m, and with the model's mass and wheel radius [N m]; the model's mass [kg], wheel radius [m] and limit [N m]":
            '25.9, 26.6; 80, 0.1651, 30',
    },
    'the description': {
        'masses: chassis, each wheel [kg]':
            '70, 2.5',
        'wheel cylinder: radius, width [m]':
            '0.1651, 0.1143',
        'properties of a300.urdf.xacro: wheel_base, wheel_track [m]':
            '0.512, 0.562',
        'for Gazebo, a wheel-slip system per wheel: wheels; slip compliance lateral, longitudinal; wheel normal force [N]':
            '4; 0.75, 0.75; 113',
        'published: wheel centres, left and right of the centre line [m]':
            '0.2829, -0.2639',
        'published: track [m]':
            '0.5468',
        'published: loads at rest, front and rear, left then right [N]':
            '190.2, 190.3, 202.2, 202.0',
        'corrected: wheel centres, left and right of the centre line [m]':
            '0.2829, -0.2829',
        'corrected: track [m]':
            '0.5658',
        'corrected: loads at rest, front and rear, left then right [N]':
            '196.2, 196.2, 196.2, 196.2',
        'corrected: total of the loads, weight [N]':
            '784.8, 784.8',
        'wheelbase [m]':
            '0.512',
        'mass that MuJoCo infers for each bumper [kg]; total mass [kg]':
            '1.238 each; 82.476',
        'model mass [kg]; manual, with the 40 Ah battery [kg]':
            '80; 78.5',
        'penetration at rest [mm]':
            '0.226',
        'clearance of the chassis at rest [mm]':
            '135.7',
        'contacts per wheel at rest; lateral offsets from the wheel centre [m]':
            '2; -0.0572, +0.0571',
    },
    'the touching start': {
        'placed touching: contacts, normal from wheel to ground (z), distance [m]':
            '4, 1, -3.0e-10',
        'placed 1e-12 m lower: contacts, normal from wheel to ground (z)':
            '8, -1',
        'cylinder on a box, touching: contacts, normal from cylinder to box (z), distance [m]':
            '1, 1, -3.0e-10',
        'the same, ccd_tolerance 1e-9':
            'none',
        'the same, nativeccd disabled':
            'none',
        'the same, 1e-12 m lower':
            '2, -1, -1.0e-12',
        'placed touching: load on the wheels in the first step [N]; fall of the robot in it [um]':
            '0.0; 2.45',
        'placed touching, and 0.1 mm lower: largest load on the wheels in the first 0.2 s [N]; deepest sink [mm]':
            '836, 813; 0.226, 0.226',
    },
    'the contact of the wheels': {
        'declared, asphalt: turn: centripetal acceleration u Omega [m/s^2], share of mu g; loads of the inner and outer wheels in the turn, and at rest [N]':
            '0.323, 4.1%; 381.5, 403.3, 392.4, 392.4',
        'declared, asphalt: turn: radius against the Coulomb reference with the centripetal force, for the loads at rest and for the loads of the turn [m], differences':
            '3.202, 3.218; +0.50 %, -0.01 %',
        'declared, asphalt: sideways slide: decay rate of the slip below mu g / b [1/s], share of b; excess over v^2/(2 mu g) of a slide at mu g down to mu g / b and at b v below, and with the decay rate for b':
            '58.58, 0.976; +1.7 %, +1.8 %',
        'declared, gravel: turn: centripetal acceleration u Omega [m/s^2], share of mu g; loads of the inner and outer wheels in the turn, and at rest [N]':
            '0.323, 5.5%; 381.4, 403.4, 392.4, 392.4',
        'declared, gravel: turn: radius against the Coulomb reference with the centripetal force, for the loads at rest and for the loads of the turn [m], differences':
            '3.193, 3.209; +0.50 %, -0.01 %',
        'declared, gravel: sideways slide: decay rate of the slip below mu g / b [1/s], share of b; excess over v^2/(2 mu g) of a slide at mu g down to mu g / b and at b v below, and with the decay rate for b':
            '58.58, 0.976; +1.0 %, +1.0 %',
        'declared, wet grass: turn: centripetal acceleration u Omega [m/s^2], share of mu g; loads of the inner and outer wheels in the turn, and at rest [N]':
            '0.328, 9.5%; 381.3, 403.5, 392.4, 392.4',
        'declared, wet grass: turn: radius against the Coulomb reference with the centripetal force, for the loads at rest and for the loads of the turn [m], differences':
            '3.149, 3.166; +0.52 %, -0.02 %',
        'declared, wet grass: sideways slide: decay rate of the slip below mu g / b [1/s], share of b; excess over v^2/(2 mu g) of a slide at mu g down to mu g / b and at b v below, and with the decay rate for b':
            '58.58, 0.976; +0.3 %, +0.3 %',
        'declared but friction damping 6 1/s, asphalt: slide from 1 m/s, entirely below mu g / b: decay rate of the slip [1/s], share of b; v0 / b, v0 / rate, the slide [m]; mu g / b [m/s]':
            '5.605, 0.934; 0.1667, 0.1784, 0.1782; 1.31',
        'point mass, defaults, mu 0.8: R_n0 F [m/s^2], b [1/s]':
            '1.09, 105.3',
        'point mass, defaults, mu 0.8: v* predicted, contact first lost at [m/s]':
            '0.0212, 0.0214',
        'point mass, defaults, mu 0.8: impedance at rest; v* with the regularizer at rest [m/s]':
            '0.913; 0.0181',
        'point mass, defaults, mu 0.35: R_n0 F [m/s^2], b [1/s]':
            '1.09, 105.3',
        'point mass, defaults, mu 0.35: v* predicted, contact first lost at [m/s]':
            '0.0332, 0.0332',
        'point mass, declared, mu 0.8: R_n0 F [m/s^2], b [1/s]':
            '186, 60',
        'point mass, declared, mu 0.8: v* predicted, contact first lost at [m/s]':
            '3.91, 3.91',
        'point mass, declared with b 600 1/s, mu 0.8: R_n0 F [m/s^2], b [1/s]':
            '186, 600',
        'point mass, declared with b 600 1/s, mu 0.8: v* predicted, contact first lost at [m/s]':
            '0.391, 0.391',
        'point mass at 1 m/s, defaults: load from, to [N] (weight 196)':
            '0, 1345',
        'point mass at 1 m/s, declared: load from, to [N]':
            '195.7, 196.2',
        'Husky, default contacts, asphalt: R_n0 F [m/s^2], b [1/s], v* [m/s]':
            '0.413, 105.3, 0.00805',
        'Husky, default contacts, gravel: R_n0 F [m/s^2], b [1/s], v* [m/s]':
            '0.413, 105.3, 0.0089',
        'Husky, default contacts, wet grass: R_n0 F [m/s^2], b [1/s], v* [m/s]':
            '0.413, 105.3, 0.0126',
        'Husky, declared contacts, asphalt: R_n0 F [m/s^2], b [1/s], v* [m/s]':
            '70.6, 60, 1.48',
        'Husky, declared contacts, asphalt: slip speed at which a steady load reaches mu F, mu R_n F / (impratio b) [mm/s]; mu g / b [m/s]':
            '9.4; 0.131',
        'Husky, declared contacts, gravel: R_n0 F [m/s^2], b [1/s], v* [m/s]':
            '70.6, 60, 1.97',
        'Husky, declared contacts, gravel: slip speed at which a steady load reaches mu F, mu R_n F / (impratio b) [mm/s]; mu g / b [m/s]':
            '7.1; 0.098',
        'Husky, declared contacts, wet grass: R_n0 F [m/s^2], b [1/s], v* [m/s]':
            '70.6, 60, 3.37',
        'Husky, declared contacts, wet grass: slip speed at which a steady load reaches mu F, mu R_n F / (impratio b) [mm/s]; mu g / b [m/s]':
            '4.1; 0.057',
        'defaults, asphalt: spin, rims commanded at 0.49 m/s: mean rim speed [m/s], yaw rate, Coulomb for those rims [rad/s], all wheels down':
            '0.24, 0.17, 0.452, 33%',
        "defaults, asphalt: spin: median angle between a contact's friction and its slip [deg]":
            '28.2',
        'defaults, asphalt: turn at 1 m/s and 1/3 rad/s: rims left, right [m/s], radius, Coulomb [m], all wheels down':
            '0.869, 1.17, 10.7, 3.56, 47%',
        'defaults, asphalt: sideways slide from 1 m/s: distance, v^2/(2 mu g) [m], difference, all wheels down':
            '0.0644, 0.0637, +1.06 %, 23%',
        'defaults, wet grass: spin, rims commanded at 0.49 m/s: mean rim speed [m/s], yaw rate, Coulomb for those rims [rad/s], all wheels down':
            '0.49, 0.644, 0.93, 53%',
        "defaults, wet grass: spin: median angle between a contact's friction and its slip [deg]":
            '18.7',
        'defaults, wet grass: turn at 1 m/s and 1/3 rad/s: rims left, right [m/s], radius, Coulomb [m], all wheels down':
            '0.849, 1.18, 5.94, 3.21, 42%',
        'defaults, wet grass: sideways slide from 1 m/s: distance, v^2/(2 mu g) [m], difference, all wheels down':
            '0.1453, 0.1456, -0.21 %, 17%',
        'defaults, asphalt: spin, load on the ground from, to [N] (weight 785)':
            '0, 1224',
        'declared, asphalt: creep under a sideways push of 30 % of mu m g, measured, predicted by the bottom zone [mm/s]':
            '2.81, 2.83',
        'declared, asphalt: spin, rims at 0.49 m/s: yaw rate, Coulomb [rad/s], difference, all wheels down':
            '0.9296, 0.9300, -0.05 %, 100%',
        "declared, asphalt: spin: median angle between a contact's friction and its slip [deg]":
            '0.0',
        'declared, asphalt: spin, rims at 0.98 m/s: yaw rate, Coulomb [rad/s], difference, all wheels down':
            '1.8600, 1.8601, -0.01 %, 100%',
        'declared, asphalt: turn at 1 m/s and 1/3 rad/s: radius, Coulomb [m], difference, all wheels down':
            '3.218, 3.214, +0.12 %, 100%',
        'declared, asphalt: sideways slide from 1 m/s: distance, v^2/(2 mu g) [m], difference, all wheels down':
            '0.0647, 0.0637, +1.50 %, 100%',
        'declared, gravel: creep under a sideways push of 30 % of mu m g, measured, predicted by the bottom zone [mm/s]':
            '2.11, 2.12',
        'declared, gravel: spin, rims at 0.49 m/s: yaw rate, Coulomb [rad/s], difference, all wheels down':
            '0.9295, 0.9300, -0.06 %, 100%',
        "declared, gravel: spin: median angle between a contact's friction and its slip [deg]":
            '0.0',
        'declared, gravel: spin, rims at 0.98 m/s: yaw rate, Coulomb [rad/s], difference, all wheels down':
            '1.8595, 1.8601, -0.03 %, 100%',
        'declared, gravel: turn at 1 m/s and 1/3 rad/s: radius, Coulomb [m], difference, all wheels down':
            '3.209, 3.214, -0.17 %, 100%',
        'declared, gravel: sideways slide from 1 m/s: distance, v^2/(2 mu g) [m], difference, all wheels down':
            '0.0856, 0.0849, +0.76 %, 100%',
        'declared, wet grass: creep under a sideways push of 30 % of mu m g, measured, predicted by the bottom zone [mm/s]':
            '1.23, 1.24',
        'declared, wet grass: spin, rims at 0.49 m/s: yaw rate, Coulomb [rad/s], difference, all wheels down':
            '0.9293, 0.9300, -0.08 %, 100%',
        "declared, wet grass: spin: median angle between a contact's friction and its slip [deg]":
            '0.0',
        'declared, wet grass: spin, rims at 0.98 m/s: yaw rate, Coulomb [rad/s], difference, all wheels down':
            '1.8589, 1.8601, -0.06 %, 100%',
        'declared, wet grass: turn at 1 m/s and 1/3 rad/s: radius, Coulomb [m], difference, all wheels down':
            '3.166, 3.214, -1.51 %, 100%',
        'declared, wet grass: sideways slide from 1 m/s: distance, v^2/(2 mu g) [m], difference, all wheels down':
            '0.1459, 0.1456, +0.18 %, 100%',
        'declared but impedance 0.9 (with b 60 1/s), asphalt: spin at 0.98 m/s against Coulomb, all wheels down; slide against v^2/(2 mu g); creep [mm/s]':
            '-74.5 %, 29%; +1.2 %; 0.02',
        'declared but impratio 1, asphalt: spin at 0.98 m/s against Coulomb, all wheels down; slide against v^2/(2 mu g); creep [mm/s]':
            '+0.1 %, 100%; +124.4 %; 263.33',
        'declared but friction damping 6 1/s, asphalt: spin at 0.98 m/s against Coulomb, all wheels down; slide against v^2/(2 mu g); creep [mm/s]':
            '-0.1 %, 100%; +179.8 %; 25.75',
    },
    'what the simulated robot can do': {
        "asphalt: deceleration of the controller's ramp at the ground, 4 u0 / v0; mu g [m/s^2]":
            '4.06; 7.85',
        'asphalt: stop from the command at 1 m/s: speed before [m/s], distance [m], time to rest [s], largest slip [m/s]':
            '1.0153, 0.1240, 0.246, 0.005',
        "asphalt: wheels that followed the controller's commands exactly [m]; u0^2 / (2 mu g) [m]":
            '0.1170; 0.0657',
        "asphalt: stop beyond the wheels that followed the commands: by the wheels' turning, by slip [mm]":
            '8.5, -1.5',
        "gravel: deceleration of the controller's ramp at the ground, 4 u0 / v0; mu g [m/s^2]":
            '4.06; 5.89',
        'gravel: stop from the command at 1 m/s: speed before [m/s], distance [m], time to rest [s], largest slip [m/s]':
            '1.0153, 0.1282, 0.253, 0.020',
        "gravel: wheels that followed the controller's commands exactly [m]; u0^2 / (2 mu g) [m]":
            '0.1170; 0.0876',
        "gravel: stop beyond the wheels that followed the commands: by the wheels' turning, by slip [mm]":
            '10.0, 1.3',
        "wet grass: deceleration of the controller's ramp at the ground, 4 u0 / v0; mu g [m/s^2]":
            '4.06; 3.43',
        'wet grass: stop from the command at 1 m/s: speed before [m/s], distance [m], time to rest [s], largest slip [m/s]':
            '1.0153, 0.1552, 0.309, 0.178',
        "wet grass: wheels that followed the controller's commands exactly [m]; u0^2 / (2 mu g) [m]":
            '0.1170; 0.1501',
        "wet grass: stop beyond the wheels that followed the commands: by the wheels' turning, by slip [mm]":
            '11.1, 27.1',
        'asphalt: spin commanded at 1 rad/s: yaw rate [rad/s], ratio, W_eff [m], largest torque [N m]':
            '0.9478, 0.9478, 1.0377, 18.1',
        'asphalt: spin commanded at 2 rad/s: yaw rate [rad/s], ratio, W_eff [m], largest torque [N m]':
            '1.8965, 0.9482, 1.0372, 18.6',
        'asphalt: turn commanded at 1 m/s and 1/3 rad/s (radius 3 m): radius [m]':
            '3.218',
        'gravel: spin commanded at 1 rad/s: yaw rate [rad/s], ratio, W_eff [m], largest torque [N m]':
            '0.9476, 0.9476, 1.0378, 13.5',
        'gravel: spin commanded at 2 rad/s: yaw rate [rad/s], ratio, W_eff [m], largest torque [N m]':
            '1.8960, 0.9480, 1.0374, 14.0',
        'gravel: turn commanded at 1 m/s and 1/3 rad/s (radius 3 m): radius [m]':
            '3.209',
        'wet grass: spin commanded at 1 rad/s: yaw rate [rad/s], ratio, W_eff [m], largest torque [N m]':
            '0.9475, 0.9475, 1.0380, 7.9',
        'wet grass: spin commanded at 2 rad/s: yaw rate [rad/s], ratio, W_eff [m], largest torque [N m]':
            '1.8954, 0.9477, 1.0378, 8.1',
        'wet grass: turn commanded at 1 m/s and 1/3 rad/s (radius 3 m): radius [m]':
            '3.166',
        'open loop, switching on the odometry: largest offset in the turn, after it [m]':
            '0.503, 0.700',
        "open loop: the footprint first leaves the lane at s [m]; the centre's offset then [m]":
            '12.53; -0.277',
        'open loop: touches':
            'pallets on the second straight at s = 18.51 m',
        'asphalt: speed at the command 1 m/s [m/s]; rolling radius [m]':
            '1.0153; 0.16499',
        'asphalt: 10 s at 1 m/s: distance, odometry [m], odometry short by':
            '10.1531, 10.0000, +1.51 %',
        'asphalt: largest speed error of a wheel while ramping to 1 m/s at 2 m/s^2 [rad/s]':
            '0.303',
        'gravel: 4 s at 1 m/s and 1/3 rad/s: heading change, odometry [rad]; separation that would give the heading change [m]':
            '1.2700, 1.3333; 1.0325',
    },
    'time steps': {
        'asphalt: stop from 1 m/s, distance at 1, 0.5, 0.25 ms [m]':
            '0.1165, 0.1240, 0.1263',
        'gravel: stop from 1 m/s, distance at 1, 0.5, 0.25 ms [m]':
            '0.1232, 0.1282, 0.1299',
        'wet grass: stop from 1 m/s, distance at 1, 0.5, 0.25 ms [m]':
            '0.1540, 0.1552, 0.1560',
        'asphalt: spin at 0.98 m/s against Coulomb at 1, 0.5, 0.25 ms':
            '-0.01 %, -0.01 %, -0.01 %',
        'asphalt: turn at 1, 0.5, 0.25 ms: radius [m]; against Coulomb':
            '3.218, 3.218, 3.218; +0.12 %, +0.12 %, +0.12 %',
        'gravel: spin at 0.98 m/s against Coulomb at 1, 0.5, 0.25 ms':
            '-0.03 %, -0.03 %, -0.03 %',
        'gravel: turn at 1, 0.5, 0.25 ms: radius [m]; against Coulomb':
            '3.209, 3.209, 3.209; -0.17 %, -0.17 %, -0.17 %',
        'wet grass: spin at 0.98 m/s against Coulomb at 1, 0.5, 0.25 ms':
            '-0.06 %, -0.06 %, -0.06 %',
        'wet grass: turn at 1, 0.5, 0.25 ms: radius [m]; against Coulomb':
            '3.166, 3.166, 3.166; -1.51 %, -1.51 %, -1.51 %',
    },
}


if __name__ == "__main__":
    main()
