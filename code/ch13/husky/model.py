"""
The Husky A300 in MuJoCo: Clearpath's description, loaded by MuJoCo's own
URDF parser, with what the description does not hold.

Physical AI, Together, Chapter 13 (Section 13.1).

MuJoCo 3.16.0 loads the URDF of asset.urdf_text through MjSpec (its default
for URDF: static bodies fused into their parents, visual elements
discarded). The description holds the bodies, their masses and inertias,
the wheel joints and the collision geometry; it holds no free joint, no
actuator, no sensor, no contact parameters and no world. This module adds
them, each a declared choice of the model:

  * a free joint on base_link (a URDF robot is fixed to the world unless a
    joint says otherwise);
  * no mass for the links without an inertial element (the bumpers): URDF
    gives them none, but MuJoCo's defaults infer a mass from their collision
    meshes at 1000 kg/m^3 unless told otherwise (bumper_mass=True keeps
    that inferred mass);
  * a velocity actuator on each wheel, torque = KV (ctrl - speed), limited to
    TORQUE; the simulator adapter (sim.py) turns it into the drivers' speed
    loop, proportional and integral. Neither the drivers' gains nor the
    motors' torque appear in the description or the user manual: all three
    values are ours;
  * the options of the simulator: a time step of 0.5 ms, the implicitfast
    integrator, the Newton solver with the elliptic friction cone, as in
    Chapter 11, and impratio 100 (Section 13.1.3);
  * the contacts of the wheels with the ground through explicit pairs, whose
    parameters are ours (Section 13.1.3): impedance IMPEDANCE, normal time
    constant TIMECONST, friction damping FRICTION_DAMPING. With MuJoCo's
    defaults (contacts="default") a wheel that slides faster than 8 to
    13 mm/s, depending on the ground, lifts off it. The friction of each
    pair is set at every step by the adapter from the ground under the
    wheel, so that the yard is one flat surface without seams;
  * contacts with three dimensions (sliding friction, no torsional or rolling
    friction) everywhere; the wheels touch the obstacles through pairs too,
    with MuJoCo's default parameters and friction OBSTACLE_FRICTION; the
    chassis and the bumpers touch the ground and the obstacles through
    MuJoCo's dynamic contacts;
  * sensors that the A300 carries: the four wheel encoders (positions and
    speeds) and a 6-axis IMU at the chassis' centre (the description places
    none), and the pose of the chassis, which only the simulator knows;
  * the world: the ground and obstacles of world.py, or, for a diagnostic,
    one flat pad with a given friction.
"""
import math

import mujoco

from . import asset, world

TIMESTEP = 0.0005             # s
KV = 40.0                     # N m s/rad, proportional gain of the drivers' speed loop (ours)
TI = 0.1                      # s, integral time of the drivers' speed loop (ours)
TORQUE = 30.0                 # N m, the motors' torque limit at the wheel (ours; Section 13.1.3)
WHEELS = ("front_left", "rear_left", "front_right", "rear_right")
LEFT, RIGHT = ("front_left", "rear_left"), ("front_right", "rear_right")

# The contacts of the wheels with the ground (ours; Section 13.1.3)
IMPEDANCE = 0.05              # solimp d0 = d_width: constant impedance
TIMECONST = 0.008             # s, solref of the normal dimension, damping ratio 1
FRICTION_DAMPING = 60.0       # 1/s, b of the friction dimensions (direct solreffriction)
IMPRATIO = 100.0              # option impratio: friction impedance / normal impedance
OBSTACLE_FRICTION = 0.5
PLACE_DEPTH = 1e-4            # m, how far below touching the robot is placed (sim.place)

# MuJoCo's defaults, for the comparison of Section 13.1.3
DEFAULT_SOLREF = (0.02, 1.0)
DEFAULT_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)
DEFAULT_IMPRATIO = 1.0


def wheel_geom_name(w):
    return f"{w}_wheel"


def build(published=False, bumper_mass=False, kv=KV, torque=TORQUE, timestep=TIMESTEP,
          ground=None, obstacles=True, contacts="declared"):
    """The MjSpec of the Husky in its world. published: the description as
    published (asymmetric track); bumper_mass: keep the mass that MuJoCo
    infers for the bumpers; ground: None for the yard of world.py, or the
    friction coefficient of a single flat pad; contacts: "declared" (the
    parameters of this module) or "default" (MuJoCo's)."""
    if contacts not in ("declared", "default"):
        raise ValueError(contacts)
    spec = mujoco.MjSpec.from_string(asset.urdf_text(published=published))
    spec.meshdir = asset.PACKAGE
    spec.modelname = "husky_a300"
    spec.compiler.discardvisual = False      # keep the yard's patches (the URDF has no visuals left)

    if not bumper_mass:
        for body in spec.bodies:
            if not body.explicitinertial:
                for geom in body.geoms:
                    geom.density = 0.0

    base = spec.body("base_link")
    base.add_freejoint(name="root")
    base.pos = [0.0, 0.0, 0.0]

    opt = spec.option
    opt.timestep = timestep
    opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    opt.iterations = 100
    opt.tolerance = 1e-8
    opt.gravity = [0.0, 0.0, -9.81]
    opt.impratio = IMPRATIO if contacts == "declared" else DEFAULT_IMPRATIO

    for body in spec.bodies:
        for geom in body.geoms:
            geom.condim = 3
            geom.priority = 0
            geom.friction = [0.0, 0.0, 0.0]

    for w in WHEELS:
        g = spec.body(f"{w}_wheel_link").geoms[0]
        g.name = wheel_geom_name(w)
        g.contype, g.conaffinity = 0, 0          # the wheels touch through pairs only
        act = spec.add_actuator(name=f"{w}_servo", target=f"{w}_wheel_joint",
                                trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.set_to_velocity(kv=kv)
        act.forcelimited = True
        act.forcerange = [-torque, torque]
        spec.add_sensor(name=f"{w}_encoder", type=mujoco.mjtSensor.mjSENS_JOINTPOS,
                        objtype=mujoco.mjtObj.mjOBJ_JOINT, objname=f"{w}_wheel_joint")
        spec.add_sensor(name=f"{w}_speed", type=mujoco.mjtSensor.mjSENS_JOINTVEL,
                        objtype=mujoco.mjtObj.mjOBJ_JOINT, objname=f"{w}_wheel_joint")
    base.add_site(name="imu", pos=[0.0, 0.0, 0.1148])
    spec.add_sensor(name="gyro", type=mujoco.mjtSensor.mjSENS_GYRO,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
    spec.add_sensor(name="accelerometer", type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")

    wb = spec.worldbody
    if ground is None:
        x0, x1, y0, y1 = world.YARD
        slab(wb, "ground", (0.5 * (x0 + x1), 0.5 * (y0 + y1)), (0.5 * (x1 - x0), 0.5 * (y1 - y0)),
             world.GROUNDS[0].friction, (0.45, 0.45, 0.45, 1.0))
        for g in world.GROUNDS:               # what the eye sees: no collision
            gx0, gx1, gy0, gy1 = g.rect
            patch = wb.add_geom(name=f"{g.name} (patch)", type=mujoco.mjtGeom.mjGEOM_BOX,
                                pos=[0.5 * (gx0 + gx1), 0.5 * (gy0 + gy1), 0.0005],
                                size=[0.5 * (gx1 - gx0), 0.5 * (gy1 - gy0), 0.0005],
                                rgba=list(g.rgba))
            patch.contype, patch.conaffinity, patch.group = 0, 0, 1
        friction = world.GROUNDS[0].friction
    else:
        slab(wb, "ground", (0.0, 0.0), (40.0, 40.0), ground, (0.5, 0.5, 0.5, 1.0))
        friction = ground
    for w in WHEELS:
        pair(spec, "ground", wheel_geom_name(w), friction, contacts)
    if ground is None and obstacles:
        for o in world.OBSTACLES:
            box = wb.add_geom(name=o.name, type=mujoco.mjtGeom.mjGEOM_BOX,
                              pos=list(o.center), size=list(o.half),
                              quat=[math.cos(o.yaw / 2), 0.0, 0.0, math.sin(o.yaw / 2)],
                              rgba=[0.55, 0.40, 0.25, 1.0])
            box.priority, box.condim = 1, 3
            box.friction = [OBSTACLE_FRICTION, 0.005, 0.0001]
            for w in WHEELS:
                pair(spec, o.name, wheel_geom_name(w), OBSTACLE_FRICTION, "default")
    wb.add_light(pos=[0.0, 0.0, 20.0], dir=[0.0, 0.0, -1.0])
    return spec


def slab(wb, name, centre, half, friction, rgba):
    """A ground: a box 0.1 m thick whose top face is z = 0."""
    g = wb.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=[centre[0], centre[1], -0.05], size=[half[0], half[1], 0.05],
                    rgba=list(rgba))
    g.priority, g.condim, g.friction = 1, 3, [friction, 0.005, 0.0001]
    return g


def pair(spec, geom1, geom2, friction, contacts):
    """An explicit contact pair: the declared parameters of this module, or
    MuJoCo's defaults."""
    p = spec.add_pair(name=f"{geom1} | {geom2}", geomname1=geom1, geomname2=geom2, condim=3)
    p.friction = [friction, friction, 0.005, 0.0001, 0.0001]
    if contacts == "declared":
        p.solref = [TIMECONST, 1.0]
        p.solimp = [IMPEDANCE, IMPEDANCE, 0.001, 0.5, 2.0]
        p.solreffriction = [-0.0, -FRICTION_DAMPING * IMPEDANCE]   # b = damping / d_width
    else:
        p.solref = list(DEFAULT_SOLREF)
        p.solimp = list(DEFAULT_SOLIMP)
    return p


def rest_height(m):
    """The height of base_link at which the wheels touch z = 0, from the
    compiled wheel positions and radius."""
    b = m.body("front_left_wheel_link").id
    g = m.geom(wheel_geom_name("front_left")).id
    return float(m.geom_size[g][0] - m.body_pos[b][2])
