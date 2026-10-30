"""
The UR5e of Chapter 14 in MuJoCo, built from Universal Robots' description.

Physical AI, Together, Chapter 14 (Section 14.1).

Universal Robots' description of the UR5e (description/ur_description,
release 3.6.0; see description/NOTICE.md) is a set of configuration files
that UR's xacro macro turns into a URDF. This module reads the files and
does the macro's mapping itself, straight into MuJoCo's MjSpec, without
xacro or ROS. The mapping, as urdf/ur_macro.xacro and urdf/inc/ur_common.xacro
do it:

  * the chain world - base_link - base_link_inertia - shoulder_link -
    upper_arm_link - forearm_link - wrist_1_link - wrist_2_link -
    wrist_3_link; base_link_inertia turned by pi about z (base_link follows
    REP-103, x forward, while UR's own frames have x backwards); each of the
    six joints at the origin that default_kinematics.yaml gives (x, y, z, then
    roll, pitch and yaw about fixed axes), revolving about its link's z axis;
  * each link's mass, centre of mass, inertia tensor and the rotation of the
    tensor's frame, from physical_parameters.yaml; the base's tensor that of a
    solid cylinder of radius 0.06 m and length 0.05 m, with the macro's
    constant 0.0833333 for 1/12;
  * each link's collision mesh where the macro's legacy origins put it, with
    the offsets of physical_parameters.yaml (shoulder 0.138 m, elbow 0.007 m)
    and the visual offsets of visual_parameters.yaml (the file gives no
    mesh_offset for the UR5e, which would replace the legacy origins);
  * the frames ft_frame, flange and tool0 of wrist_3_link, and base of
    base_link, as sites;
  * the joint ranges of joint_limits.yaml (which limits the elbow to +-180
    deg; UR's datasheet gives +-360 deg for every joint).

What the description does not hold, and the model adds (each ours):

  * a torque actuator at each joint, limited to the joint's max_effort of
    joint_limits.yaml (150 N m at the base, shoulder and elbow, 28 N m at the
    wrists, after the UR5e's manual and UR's article on the joints' torques,
    which the file names as its sources);
  * no friction, damping or rotor inertia at the joints, as in the
    description, which sets damping and friction to 0 and gives no rotor
    inertia (armature= gives one; MuJoCo Menagerie's UR5e uses 0.1 kg m^2 at
    every joint, its own choice); the joints' speed limit of 180 deg/s is
    enforced neither here nor by the controller: the script reports the
    largest speeds of its runs;
  * a payload at tool0: a box of 0.1 m side whose centre of mass lies on
    tool0's z axis, 0.05 m from the flange (its mass, and the offset, are
    arguments);
  * the task's post and table (world.py);
  * the simulator's options: a time step of 0.5 ms, the implicitfast
    integrator, gravity 9.81 m/s^2 (MuJoCo's default);
  * contacts between the arm (with its payload) and the post; none between
    the arm's own links, none with the table.
"""
import math
import os

import mujoco
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
UR = os.path.join(HERE, "description", "ur_description")
CONFIG = os.path.join(UR, "config", "ur5e")
MESHES = os.path.join(UR, "meshes", "ur5e", "collision")
SOURCE = ("ur_description 3.6.0, https://github.com/UniversalRobots/Universal_Robots_ROS2_Description, "
          "commit 7ce98d8c3512b9d0239da93b75de68ea64f61bd6 (22 October 2026), BSD 3-Clause")

LINKS = ("shoulder", "upper_arm", "forearm", "wrist_1", "wrist_2", "wrist_3")
JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
TIMESTEP = 0.0005
PAYLOAD_SIZE = 0.1          # [m] the payload's box (ours)
PAYLOAD_OFFSET = 0.05       # [m] its centre of mass from the flange, along tool0's z (ours)
ARM_CONTYPE, ARM_CONAFFINITY = 2, 1      # the arm collides with the post only
POST_CONTYPE, POST_CONAFFINITY = 1, 2


# ----------------------------------------------------------------------------
# UR's configuration files
# ----------------------------------------------------------------------------
class _Loader(yaml.SafeLoader):
    """YAML with xacro's tag !degrees, which joint_limits.yaml uses."""


_Loader.add_constructor("!degrees", lambda loader, node: math.radians(
    float(loader.construct_scalar(node))))


def config(name):
    """One of UR's configuration files of the UR5e, as nested dictionaries."""
    with open(os.path.join(CONFIG, name), encoding="utf-8") as fh:
        return yaml.load(fh, Loader=_Loader)


def description():
    """The four configuration files, by the names that UR's macro gives them."""
    return dict(kinematics=config("default_kinematics.yaml")["kinematics"],
                physical=config("physical_parameters.yaml"),
                limits=config("joint_limits.yaml")["joint_limits"],
                visual=config("visual_parameters.yaml")["mesh_files"])


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------
def rpy_matrix(roll, pitch, yaw):
    """URDF's rotation: roll about x, then pitch about y, then yaw about z,
    all fixed axes, R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def quat(matrix):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(matrix, dtype=float).flatten())
    return q


def _origin(entry):
    return (np.array([entry["x"], entry["y"], entry["z"]], dtype=float),
            rpy_matrix(entry["roll"], entry["pitch"], entry["yaw"]))


def mesh_origins(desc):
    """The collision meshes' placement in their links: the macro's legacy
    origins, unless visual_parameters.yaml gives a mesh_offset (it does not
    for the UR5e)."""
    phys, vis = desc["physical"], desc["visual"]
    offsets = phys.get("offsets", {})
    half = math.pi / 2
    w3 = vis["wrist_3"].get("visual_offset_xyz")
    w3_xyz = ([float(v) for v in w3.split()] if w3 else [0.0, 0.0, vis["wrist_3"]["visual_offset"]])
    legacy = {
        "base": ((0, 0, 0), (0, 0, math.pi)),
        "shoulder": ((0, 0, 0), (0, 0, math.pi)),
        "upper_arm": ((0, 0, offsets.get("shoulder_offset", 0.0)), (half, 0, -half)),
        "forearm": ((0, 0, offsets.get("elbow_offset", 0.0)), (half, 0, -half)),
        "wrist_1": ((0, 0, vis["wrist_1"]["visual_offset"]), (half, 0, 0)),
        "wrist_2": ((0, 0, vis["wrist_2"]["visual_offset"]), (0, 0, 0)),
        "wrist_3": (tuple(w3_xyz), (half, 0, 0)),
    }
    out = {}
    for name, (xyz, rpy) in legacy.items():
        entry = vis[name].get("mesh_offset")
        if entry:
            out[name] = _origin(entry)
        else:
            out[name] = (np.array(xyz, dtype=float), rpy_matrix(*rpy))
    return out


def inertials(desc):
    """Each link's mass, centre of mass and inertia tensor in the link's
    frame (the tensor of physical_parameters.yaml turned by the rotation of
    its frame: I_link = R I R^T)."""
    ip = desc["physical"]["inertia_parameters"]
    out = {}
    m = float(ip["base_mass"])
    r, h = float(ip["links"]["base"]["radius"]), float(ip["links"]["base"]["length"])
    ixx = 0.0833333 * m * (3 * r * r + h * h)
    out["base"] = (m, np.zeros(3), np.diag([ixx, ixx, 0.5 * m * r * r]))
    for name in LINKS:
        cog = ip["center_of_mass"][name + "_cog"]
        rot = ip["rotation"][name]
        t = ip["tensor"][name]
        tensor = np.array([[t["ixx"], t["ixy"], t["ixz"]],
                           [t["ixy"], t["iyy"], t["iyz"]],
                           [t["ixz"], t["iyz"], t["izz"]]], dtype=float)
        rmat = rpy_matrix(rot["roll"], rot["pitch"], rot["yaw"])
        out[name] = (float(ip[name + "_mass"]), np.array([cog["x"], cog["y"], cog["z"]], float),
                     rmat @ tensor @ rmat.T)
    return out


def _set_inertial(body, mass, com, tensor):
    body.explicitinertial = True
    body.mass = mass
    body.ipos = com
    body.iquat = [1, 0, 0, 0]
    body.fullinertia = [tensor[0, 0], tensor[1, 1], tensor[2, 2],
                        tensor[0, 1], tensor[0, 2], tensor[1, 2]]


# ----------------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------------
def build(payload=0.0, payload_offset=PAYLOAD_OFFSET, armature=0.0, timestep=TIMESTEP,
          world=None, base_pos=(0.0, 0.0, 0.0), box_offset=None):
    """The MjSpec of the UR5e with a payload of the given mass [kg] at tool0.

    The payload's centre of mass lies payload_offset from the flange along
    tool0's z axis; its box is centred box_offset from the flange, by default
    at its centre of mass, or elsewhere for a box whose mass is not spread
    evenly. world, if given, is a function of the spec that adds the task's
    objects (world.py). base_pos places base_link."""
    desc = description()
    kin, lim = desc["kinematics"], desc["limits"]
    spec = mujoco.MjSpec()
    spec.modelname = "ur5e"
    spec.compiler.degree = False          # the joint ranges are in radians
    spec.option.timestep = timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    origins = mesh_origins(desc)
    inert = inertials(desc)
    names = {"base": "base.stl", "shoulder": "shoulder.stl", "upper_arm": "upperarm.stl",
             "forearm": "forearm.stl", "wrist_1": "wrist1.stl", "wrist_2": "wrist2.stl",
             "wrist_3": "wrist3.stl"}
    for link, fname in names.items():
        spec.add_mesh(name=link, file=os.path.join(MESHES, fname))

    def collision(body, link):
        xyz, rmat = origins[link]
        body.add_geom(name=link + "_collision", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=link,
                      pos=xyz, quat=quat(rmat), contype=ARM_CONTYPE,
                      conaffinity=ARM_CONAFFINITY, group=3, density=0, rgba=[0.6, 0.6, 0.65, 1])

    base_link = spec.worldbody.add_body(name="base_link", pos=list(base_pos))
    base_link.add_site(name="base", quat=quat(rpy_matrix(0, 0, math.pi)), size=[0.01, 0, 0])
    parent = base_link.add_body(name="base_link_inertia", quat=quat(rpy_matrix(0, 0, math.pi)))
    _set_inertial(parent, *inert["base"])
    collision(parent, "base")
    keys = {"shoulder": "shoulder", "upper_arm": "upper_arm", "forearm": "forearm",
            "wrist_1": "wrist_1", "wrist_2": "wrist_2", "wrist_3": "wrist_3"}
    for link, joint in zip(LINKS, JOINTS):
        xyz, rmat = _origin(kin[keys[link]])
        body = parent.add_body(name=link + "_link", pos=xyz, quat=quat(rmat))
        _set_inertial(body, *inert[link])
        jl = lim[joint]
        j = body.add_joint(name=joint, type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1],
                           armature=armature, damping=0.0, frictionloss=0.0)
        if jl.get("has_position_limits", True):
            j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
            j.range = [jl["min_position"], jl["max_position"]]
        collision(body, link)
        parent = body
    wrist3 = parent
    wrist3.add_site(name="ft_frame", quat=quat(rpy_matrix(math.pi, 0, 0)), size=[0.01, 0, 0])
    r_flange = rpy_matrix(0, -math.pi / 2, -math.pi / 2)
    r_tool0 = r_flange @ rpy_matrix(math.pi / 2, 0, math.pi / 2)
    wrist3.add_site(name="flange", quat=quat(r_flange), size=[0.01, 0, 0])
    wrist3.add_site(name="tool0", quat=quat(r_tool0), size=[0.01, 0, 0])
    if payload > 0:
        z_tool = r_tool0[:, 2]
        centre = payload_offset if box_offset is None else box_offset
        box = wrist3.add_body(name="payload", pos=z_tool * centre, quat=quat(r_tool0))
        box.explicitinertial = True
        box.mass = payload
        box.ipos = [0.0, 0.0, payload_offset - centre]   # MjSpec's default ipos is NaN
        box.iquat = [1.0, 0.0, 0.0, 0.0]
        side = PAYLOAD_SIZE
        box.inertia = [payload * side * side / 6.0] * 3
        box.add_geom(name="payload", type=mujoco.mjtGeom.mjGEOM_BOX, size=[side / 2] * 3,
                     contype=ARM_CONTYPE, conaffinity=ARM_CONAFFINITY, density=0,
                     rgba=[0.85, 0.55, 0.2, 1])
    for joint in JOINTS:
        effort = float(lim[joint]["max_effort"])
        act = spec.add_actuator(name=joint, target=joint)
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.set_to_motor()
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.ctrlrange = [-effort, effort]
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.forcerange = [-effort, effort]
    if world is not None:
        world(spec)
    return spec


def compile_model(**kw):
    spec = build(**kw)
    return spec, spec.compile()


def limits():
    """The joints' limits of joint_limits.yaml: range [rad], speed [rad/s],
    torque [N m]."""
    lim = description()["limits"]
    return {j: dict(lower=lim[j]["min_position"], upper=lim[j]["max_position"],
                    speed=lim[j]["max_velocity"], torque=float(lim[j]["max_effort"]))
            for j in JOINTS}
