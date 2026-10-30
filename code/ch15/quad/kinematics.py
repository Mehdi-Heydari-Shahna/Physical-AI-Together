"""Prototype: Go2 leg kinematics, analytic IK checked against MuJoCo."""
import math
import numpy as np
import mujoco

L_HIP_Y = 0.0955      # thigh body offset along y from the hip body
L1 = 0.213            # thigh length (calf body at z = -0.213)
FOOT = np.array([-0.002, 0.0, -0.213])   # the foot sphere's centre in the calf frame
HIP = {"FL": (0.1934, 0.0465), "FR": (0.1934, -0.0465), "RL": (-0.1934, 0.0465), "RR": (-0.1934, -0.0465)}
SIDE = {"FL": 1.0, "FR": -1.0, "RL": 1.0, "RR": -1.0}
L2 = math.hypot(FOOT[0], FOOT[2])
DELTA = math.atan2(-FOOT[0], -FOOT[2])    # angle of the calf's foot vector from -z (positive for x<0?)

def fk(leg, q):
    """Foot sphere centre in the base frame for joint angles q = (abd, hip, knee)."""
    q1, q2, q3 = q
    s = SIDE[leg]
    c3, s3 = math.cos(q3), math.sin(q3)
    # calf foot vector rotated by the knee (about y): Ry(q3) @ FOOT
    v = np.array([FOOT[0] * c3 + FOOT[2] * s3, 0.0, -FOOT[0] * s3 + FOOT[2] * c3])
    v = v + np.array([0.0, 0.0, -L1])
    c2, s2 = math.cos(q2), math.sin(q2)
    w = np.array([v[0] * c2 + v[2] * s2, 0.0, -v[0] * s2 + v[2] * c2])
    w = w + np.array([0.0, s * L_HIP_Y, 0.0])
    c1, s1 = math.cos(q1), math.sin(q1)
    p = np.array([w[0], w[1] * c1 - w[2] * s1, w[1] * s1 + w[2] * c1])
    hx, hy = HIP[leg]
    return p + np.array([hx, hy, 0.0])

def ik(leg, p, q_guess=None):
    """Joint angles that put the foot sphere centre at p (base frame)."""
    s = SIDE[leg]
    hx, hy = HIP[leg]
    x, y, z = p[0] - hx, p[1] - hy, p[2]
    r2 = y * y + z * z
    zp = -math.sqrt(max(r2 - L_HIP_Y ** 2, 1e-12))
    q1 = math.atan2(z, y) - math.atan2(zp, s * L_HIP_Y)
    q1 = (q1 + math.pi) % (2 * math.pi) - math.pi
    # sagittal: (x, zp) from the thigh joint; links L1 (along -z) and the foot vector
    d2 = x * x + zp * zp
    # law of cosines between L1 and L2 (the foot vector), knee angle relative
    cos_k = (d2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_k = max(-1.0, min(1.0, cos_k))
    beta = math.acos(cos_k)          # angle between the thigh's -z and the foot vector, in the knee
    # the knee is bent backward: q3 negative (Go2's calf range is negative)
    q3 = -beta - DELTA                # account for the foot vector's offset from -z
    # solve q2 with q3 known: foot = Ry(q2) @ (u) where u = (0,0,-L1) + Ry(q3) @ FOOT
    c3, s3 = math.cos(q3), math.sin(q3)
    ux = FOOT[0] * c3 + FOOT[2] * s3
    uz = -FOOT[0] * s3 + FOOT[2] * c3 - L1
    # Ry(q2) (ux, uz) = (x, zp): x = ux c + uz s ; zp = -ux s + uz c
    q2 = math.atan2(ux * zp - uz * x, -(ux * x + uz * zp)) if False else math.atan2(x * uz - zp * ux, x * ux + zp * uz)
    return np.array([q1, q2, q3])

if __name__ == "__main__":
    spec = mujoco.MjSpec.from_file("go2.xml")
    for g in list(spec.geoms):
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
            spec.delete(g)
    for m_ in list(spec.meshes):
        spec.delete(m_)
    m = spec.compile(); d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    worst = 0.0; worst_ik = 0.0
    for trial in range(200):
        d.qpos[:] = 0; d.qpos[3] = 1.0
        q = np.array([rng.uniform(-0.5, 0.5), rng.uniform(0.2, 1.6), rng.uniform(-2.6, -0.9)] * 4)
        d.qpos[7:] = q
        mujoco.mj_kinematics(m, d)
        for i, leg in enumerate(("FL", "FR", "RL", "RR")):
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, leg)
            pw = d.geom_xpos[gid] - d.qpos[:3]
            pk = fk(leg, q[3*i:3*i+3])
            worst = max(worst, np.max(np.abs(pw - pk)))
            qi = ik(leg, pk)
            worst_ik = max(worst_ik, np.max(np.abs(fk(leg, qi) - pk)))
    print("FK vs MuJoCo largest difference [m]", worst, "IK round trip [m]", worst_ik)
