"""The Go2 of MuJoCo Menagerie (go2.xml, commit f3b6013) on a floor plane."""
import os
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "description", "go2.xml")
LEGS = ("FL", "FR", "RL", "RR")
G = 9.81


FOOT_SOLIMP = (0.015, 1.0, 0.022, 0.5, 2.0)   # Menagerie's soft feet (go2.xml)
STIFF_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)   # MuJoCo's default impedance


def build(timestep=0.001, foot_solimp=None):
    """Compile go2.xml without its visual meshes (they carry no mass and no
    collision), with a floor plane and the given time step; foot_solimp, if
    given, replaces the feet's contact impedance (for a comparison)."""
    spec = mujoco.MjSpec.from_file(XML)
    for g in list(spec.geoms):
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
            spec.delete(g)
    for mesh in list(spec.meshes):
        spec.delete(mesh)
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0, 0, 0.05]
    spec.option.timestep = timestep
    if foot_solimp is not None:
        for g in spec.geoms:
            if g.name in LEGS:
                g.solimp = list(foot_solimp)
    return spec.compile()


class Ids:
    def __init__(self, m):
        self.feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, l) for l in LEGS]
        self.calf = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, l + "_calf") for l in LEGS]
        self.base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.radius = float(m.geom_size[self.feet[0]][0])
        self.mass = float(np.sum(m.body_mass))


def ground_wrench(m, d, ids):
    """Per-foot contact force (world, on the foot), the other contacts' total
    force, and the full ground wrench's centre of pressure (contact torques
    included) and the centre of pressure of the forces alone."""
    F = np.zeros((4, 3)); other = np.zeros(3)
    M = np.zeros(3); Mf = np.zeros(3); f6 = np.zeros(6)
    for c in range(d.ncon):
        con = d.contact[c]
        if con.geom1 != ids.floor and con.geom2 != ids.floor:
            continue
        mujoco.mj_contactForce(m, d, c, f6)
        fr = con.frame.reshape(3, 3)
        fw, tw = fr.T @ f6[:3], fr.T @ f6[3:]
        # mj_contactForce: the force that geom1 exerts on geom2... make it the force on the robot
        if con.geom2 == ids.floor:
            fw, tw = -fw, -tw
        g = con.geom2 if con.geom1 == ids.floor else con.geom1
        if g in ids.feet:
            F[ids.feet.index(g)] += fw
        else:
            other += fw
        M += np.cross(con.pos, fw) + tw
        Mf += np.cross(con.pos, fw)
    Fz = F[:, 2].sum() + other[2]
    if Fz <= 1e-9:
        return F, other, np.full(2, np.nan), np.full(2, np.nan)
    return F, other, np.array([-M[1] / Fz, M[0] / Fz]), np.array([-Mf[1] / Fz, Mf[0] / Fz])
