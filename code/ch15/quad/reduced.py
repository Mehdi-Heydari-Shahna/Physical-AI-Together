"""The reduced model that the governor's test uses: the centre of mass on a
line through its home place, the zero-moment point of a point mass at a
constant height (the cart-table model), the support polygons of the four
feet and of the three that remain when the front-right foot (FR) is lifted,
and the profile of the operation that lifts FR.

Every number here is declared: the feet's places are those of the settled
stance of sim.py and the home centre of mass the place where its first
second leaves it, rounded to 0.1 mm, its height that of the stance; the
world holds the stance there, and quad_model.py compares them.
"""
import math

import numpy as np

from . import support

G = 9.81
MASS = 15.206408                 # [kg] the robot's mass in go2.xml
FEET = np.array([(0.1899, 0.1453), (0.1899, -0.1453), (-0.1970, 0.1453), (-0.1970, -0.1453)])
COM0 = np.array([-0.0223, 0.0])  # [m] the home centre of mass, horizontal
Z_C = 0.2484                     # [m] its height
RECT = support.ccw(FEET)
TRI = support.ccw(FEET[[0, 2, 3]])                 # FL, RL, RR
INCENTRE, INRADIUS = support.incentre(FEET[0], FEET[2], FEET[3])
U = (INCENTRE - COM0) / np.linalg.norm(INCENTRE - COM0)   # the shift line's direction
S_INCENTRE = float(np.linalg.norm(INCENTRE - COM0))
K = Z_C / G                      # [s^2] the zero-moment point's lead per unit acceleration

T = 0.2                          # [s] the governor's period
SINK = 0.013                     # [m] how far a loaded foot sinks into MuJoCo's soft floor
LIFT = 0.060                     # [m] how high the operation lifts FR above the floor's contact
APPROACH = 0.010                 # [m] the touchdown's slow descent starts this far above the floor
TOUCH_SPEED = 0.05               # [m/s] and descends at this speed
# The operation: (name, periods); FR's load share and target height over it
PHASES = (("unload", 2), ("lift", 2), ("hold", 5), ("lower", 2), ("touchdown", 2), ("reload", 2))
OP_PERIODS = sum(n for _, n in PHASES)
OP_TIME = OP_PERIODS * T


def hc(t, d):
    """A half-cosine blend from 0 to 1 over d."""
    t = min(max(t, 0.0), d)
    return 0.5 * (1.0 - math.cos(math.pi * t / d))


def phase_of(u):
    """The operation's phase at time u since it began, and the time within it."""
    t0 = 0.0
    for name, n in PHASES:
        if u < t0 + n * T - 1e-12:
            return name, u - t0
        t0 += n * T
    return "done", u - t0


def fr_command(u, ground=0.0):
    """FR's load share and target height above its stance place at time u of
    the operation; ground: where the plan believes the floor under FR lies,
    above where it does (0 for the declared floor)."""
    if u is None:
        return 1.0, 0.0
    name, t = phase_of(u)
    d = 2 * T
    top = SINK + LIFT + ground
    if name == "unload":
        return 1.0 - hc(t, d), SINK * hc(t, d)
    if name == "lift":
        return 0.0, SINK + (top - SINK) * hc(t, d)
    if name == "hold":
        return 0.0, top
    if name == "lower":
        return 0.0, top - (LIFT - APPROACH) * hc(t, d)
    if name == "touchdown":
        return 0.0, SINK + ground + APPROACH - TOUCH_SPEED * t
    if name == "reload":
        z0 = SINK + ground + APPROACH - TOUCH_SPEED * d
        return hc(t, d), z0 * (1.0 - hc(t, d))
    return 1.0, 0.0


def three_feet(u0):
    """True if a segment that begins at time u0 of the operation (None: no
    operation) lies, in part, where FR carries less than its share."""
    return u0 is not None and u0 < OP_TIME - 1e-12


# ----------------------------------------------------------------------------
# The centre of mass along the line, one period: a half-cosine acceleration
# ----------------------------------------------------------------------------
def accel(a0, a1, tau, d=T):
    return a0 + (a1 - a0) * 0.5 * (1.0 - math.cos(math.pi * tau / d))


def speed(v0, a0, a1, tau, d=T):
    return v0 + a0 * tau + (a1 - a0) * (0.5 * tau - d / (2 * math.pi) * math.sin(math.pi * tau / d))


def position(s0, v0, a0, a1, tau, d=T):
    return (s0 + v0 * tau + 0.5 * a0 * tau * tau
            + (a1 - a0) * (0.25 * tau * tau - d * d / (2 * math.pi ** 2)
                           * (1.0 - math.cos(math.pi * tau / d))))


def zmp_range(s0, v0, a0, a1, d=T, n=40):
    """The least and largest zero-moment point coordinate q = s - K a along
    the line over the period: sampled every d/n, widened by the most that q
    can move between samples."""
    qs = [position(s0, v0, a0, a1, i * d / n, d) - K * accel(a0, a1, i * d / n, d)
          for i in range(n + 1)]
    vmax = max(abs(v0), abs(speed(v0, a0, a1, d, d))) + max(abs(a0), abs(a1)) * d
    rate = vmax + K * abs(a1 - a0) * math.pi / (2 * d)
    pad = rate * d / n / 2.0
    return min(qs) - pad, max(qs) + pad


def point(q):
    return COM0 + q * U


def distances(q_lo, q_hi):
    """The least signed distance inside the triangle and inside the rectangle
    over the zero-moment points from q_lo to q_hi on the line (the signed
    distance is concave, so the ends bind)."""
    d_tri = min(support.signed_distance(TRI, point(q)) for q in (q_lo, q_hi))
    d_rect = min(support.signed_distance(RECT, point(q)) for q in (q_lo, q_hi))
    return d_tri, d_rect
