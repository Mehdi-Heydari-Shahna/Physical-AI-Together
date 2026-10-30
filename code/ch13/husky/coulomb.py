"""
The steady motion of a skid-steered robot under Coulomb friction, for the
contacts of a model: the reference against which Section 13.1.3 checks
MuJoCo's friction.

Physical AI, Together, Chapter 13 (Section 13.1).

A rigid robot moves in the plane with the twist (u, v, Omega) of base_link
in its own frame: forward speed, leftward speed and yaw rate. Each contact
k of a wheel with the ground lies at (x_k, y_k) in that frame and carries
the normal load N_k; the wheel's rim carries the robot forward at the rim
speed s_k (its spin times its radius). The contact slips at
    (u - Omega y_k - s_k,  v + Omega x_k),
and Coulomb friction acts against the slip with the magnitude mu N_k,
whatever the slip speed. In steady motion, without inertia and with the
robot's own forces only, the friction forces and their moment about the
origin sum to zero: three equations for (u, v, Omega). For the symmetric
spin in place, with point contacts at the wheel centres (+-L, +-B) whose
loads balance about the centre (the left wheels carry as much as the right,
the front wheels as much as the rear), the solution is
Omega = B s / (L^2 + B^2), whatever mu and however the loads are shared
otherwise; without that balance the robot also translates. With the contacts of a model, the motion
is solved by Newton's method.

This reference is quasi-static: it leaves out the robot's inertia, so that the
friction forces sum to zero. That is exact for a spin in place about the
centre of mass, which does not accelerate, but a robot that turns while it
translates needs a centripetal force: its centre of mass, at (c_x, c_y) in the
frame of base_link, moves at (u - Omega c_y, v + Omega c_x) and accelerates at
Omega times that velocity turned a quarter turn to the left. Given the mass
and the centre of mass, residual() and steady() also solve the steady motion
in which the friction forces supply that acceleration (mass times it) and
their moment about the origin is the moment of that force about the origin;
the loads then are those of the turn, which contacts_in_motion() measures.
"""
import math

import numpy as np


def residual(q, contacts, rims, mu, eps=1e-12, mass=None, com=(0.0, 0.0)):
    """The friction forces and their moment about the origin, less, given a
    mass, the force and moment that the steady motion's centripetal
    acceleration of the centre of mass com needs (zero: quasi-static)."""
    u, v, w = q
    out = np.zeros(3)
    for wheel, x, y, n in contacts:
        sx, sy = u - w * y - rims[wheel], v + w * x
        norm = math.hypot(sx, sy) + eps
        fx, fy = -mu * n * sx / norm, -mu * n * sy / norm
        out += (fx, fy, x * fy - y * fx)
    if mass is not None:
        vx, vy = u - w * com[1], v + w * com[0]
        ax, ay = -w * vy, w * vx
        out -= (mass * ax, mass * ay, mass * (com[0] * ay - com[1] * ax))
    return out


def steady(contacts, rims, mu, guess=(0.0, 0.0, 0.5), tol=1e-13, iterations=100,
           mass=None, com=(0.0, 0.0)):
    """(u, v, Omega) of the steady motion. contacts: (wheel, x, y, N) in the
    frame of base_link; rims: wheel -> rim speed [m/s]; mass and com: None
    for the quasi-static reference, or the robot's mass and the centre of
    mass in the frame of base_link."""
    f = lambda q_: residual(q_, contacts, rims, mu, mass=mass, com=com)   # noqa: E731
    q = np.array(guess, dtype=float)
    for _ in range(iterations):
        r = f(q)
        jac = np.zeros((3, 3))
        for j in range(3):
            dq = np.zeros(3)
            dq[j] = 1e-7
            jac[:, j] = (f(q + dq) - r) / 1e-7
        step = np.linalg.solve(jac, -r)
        q += step
        if np.linalg.norm(step) < tol:
            break
    return q, float(np.abs(f(q)).max())


def point_spin_rate(half_track, half_wheelbase, rim):
    """Omega of the spin in place with point contacts at the wheel centres."""
    return half_track * rim / (half_wheelbase ** 2 + half_track ** 2)


def contacts_at_rest(husky, seconds=1.0):
    """The contacts of a robot left to settle with its wheels held: (wheel,
    x, y, N) in the frame of base_link."""
    husky.place()
    husky.wheel_targets(0.0, 0.0)
    husky.run(seconds)
    x0, y0, yaw = husky.pose()
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for wheel, cs in husky.wheel_contacts().items():
        for pos, f, _ in cs:
            px, py = pos[0] - x0, pos[1] - y0
            out.append((wheel, c * px + s * py, -s * px + c * py, float(f[2])))
    return out


def body_frame(husky, contacts):
    """The contacts (wheel, position, force) of husky.wheel_contacts() as
    (wheel, x, y, N) in the frame of base_link."""
    x0, y0, yaw = husky.pose()
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for wheel, cs in contacts.items():
        for pos, f, _ in cs:
            px, py = pos[0] - x0, pos[1] - y0
            out.append((wheel, c * px + s * py, -s * px + c * py, float(f[2])))
    return out


def centre_of_mass(husky):
    """The robot's centre of mass in the frame of base_link: (x, y) [m]."""
    x0, y0, yaw = husky.pose()
    cx, cy = husky.d.subtree_com[husky.base][:2]
    c, s = math.cos(yaw), math.sin(yaw)
    return (c * (cx - x0) + s * (cy - y0), -s * (cx - x0) + c * (cy - y0))


class ContactsInMotion:
    """The contacts of a moving robot averaged over samples: each sample
    adds the contacts in the frame of base_link, which must come in the
    same order and number in every sample (the two contacts of each wheel's
    cylinder with the flat ground)."""

    def __init__(self):
        self.sum, self.n = None, 0

    def add(self, husky):
        cs = body_frame(husky, husky.wheel_contacts())
        if self.sum is None:
            self.sum = [[w, x, y, n] for w, x, y, n in cs]
        else:
            if [c[0] for c in cs] != [c[0] for c in self.sum]:
                raise ValueError("the contacts changed while they were averaged")
            for acc, c in zip(self.sum, cs):
                acc[1] += c[1]
                acc[2] += c[2]
                acc[3] += c[3]
        self.n += 1

    def mean(self):
        return [(w, x / self.n, y / self.n, n / self.n) for w, x, y, n in self.sum]

