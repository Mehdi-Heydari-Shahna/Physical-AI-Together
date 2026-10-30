"""The stance controller: contact forces for the planned motion of the centre
of mass, mapped to joint torques, with joint servos on the legs' inverse
kinematics."""
import numpy as np
import mujoco
from .model import LEGS, G
from . import kinematics as kin

KP, KD = 120.0, 3.0          # joint servo gains [N m/rad], [N m s/rad]


def skew(r):
    return np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])


class Stance:
    def __init__(self, m, ids, kp=KP, kd=KD):
        self.m, self.ids, self.kp, self.kd = m, ids, kp, kd
        self.jac = np.zeros((3, m.nv))
        self.prev_qs = None
        self.scratch = mujoco.MjData(m)

    def forces(self, d, com, acc, load):
        """Ground forces on the feet with the least weighted norm that give the
        total force m (acc + g) and no moment about com; foot i's weight is
        1/load[i] (load 1: a stance foot; load 0: no force)."""
        idx = [i for i in range(4) if load[i] > 1e-9]
        A = np.zeros((6, 3 * len(idx))); b = np.zeros(6)
        b[:3] = self.ids.mass * np.array([acc[0], acc[1], G])
        w = np.zeros(3 * len(idx))
        for k, i in enumerate(idx):
            p = d.geom_xpos[self.ids.feet[i]] - np.array([0, 0, self.ids.radius])
            A[:3, 3 * k:3 * k + 3] = np.eye(3)
            A[3:, 3 * k:3 * k + 3] = skew(p - com)
            w[3 * k:3 * k + 3] = load[i]
        f = (w[:, None] * A.T) @ np.linalg.solve((A * w) @ A.T, b)
        out = np.zeros((4, 3))
        for k, i in enumerate(idx):
            out[i] = f[3 * k:3 * k + 3]
        return out

    def base_for_com(self, d, com_xy, z_base, feet):
        """The base position, level and facing +x, at which the model's centre of
        mass lies at com_xy with the feet at their targets."""
        s = self.scratch
        b = np.array([com_xy[0], com_xy[1], z_base])
        for _ in range(3):
            s.qpos[:3] = b; s.qpos[3:7] = [1, 0, 0, 0]
            for i, leg in enumerate(LEGS):
                s.qpos[7 + 3 * i:10 + 3 * i] = kin.ik(leg, feet[i] - b)
            mujoco.mj_kinematics(self.m, s); mujoco.mj_comPos(self.m, s)
            b[:2] += com_xy - s.subtree_com[1][:2]
        return b

    def torques(self, d, base, feet, f, dt):
        q, qd = d.qpos[7:], d.qvel[6:]
        qs = np.zeros(12)
        for i, leg in enumerate(LEGS):
            qs[3 * i:3 * i + 3] = kin.ik(leg, feet[i] - base)
        qsd = np.zeros(12) if self.prev_qs is None else (qs - self.prev_qs) / dt
        self.prev_qs = qs
        tau = self.kp * (qs - q) + self.kd * (qsd - qd) + d.qfrc_bias[6:]
        for i in range(4):
            if not np.any(f[i]):
                continue
            p = d.geom_xpos[self.ids.feet[i]] - np.array([0, 0, self.ids.radius])
            mujoco.mj_jac(self.m, d, self.jac, None, p, self.ids.calf[i])
            tau -= self.jac[:, 6:].T @ f[i]
        lo, hi = self.m.actuator_ctrlrange[:, 0], self.m.actuator_ctrlrange[:, 1]
        return np.clip(tau, lo, hi), tau, qs
