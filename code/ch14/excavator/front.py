"""
The excavator's front: its geometry, its cylinders, and what fixes them.

Physical AI, Together, Chapter 14 (Section 14.1).

The front of the PC138US-8 is modelled in the vertical plane of the upper
structure (x forward along the boom, z up, the origin on the ground on the
swing axis) as three straight links: the boom from its foot pin B0 to the arm
pin B1, the arm from B1 to the bucket pin B2, and the bucket from B2 to its
lip. The angles: theta1 of the line B0-B1 from the horizontal; theta2 of the
arm from the boom's line (negative: the arm folds down and in); theta3 of the
line B2-lip from the arm's line.

Published (published.py): the boom's 4.6 m and the arm's 2.5 m, taken as the
distances between their pins; the cylinders' bores and strokes; the relief
pressure; the working range and the tool forces.

Ours, chosen (each a round or typical value, not a measurement):

  * the boom foot X0 = 0.20 m ahead of the swing axis, on the machine's
    centre line (the real boom foot's place is not published);
  * the bucket's radius from its pin to its lip, R_BUCKET = 1.05 m;
  * the arm's inner limit, theta2 = -150 deg, and the bucket's range, theta3
    from -150 deg (curled) to +30 deg (open);
  * the cylinders' dead lengths (retracted length less stroke): boom 0.50 m,
    arm 0.55 m; the rods of the boom and bucket cylinders, 75 and 65 mm (the
    arm's 75 mm is published);
  * the boom cylinder's angle at the boom's lowest position, PSI_BOOM = 20
    deg between its two arms (see boom_cylinder);
  * the bucket's linkage, not modelled: the bucket cylinder acts on the
    bucket through a constant lever, the one that the published ISO bucket
    force implies at the relief pressure (bucket_lever).

Fitted to published numbers (fit_front, fit_arm_cylinder; the script checks
that the stored values still come out):

  * the boom foot's height Z0 and the boom's range THETA1, and the arm's
    outer limit THETA2_OUT, so that the model's maximum digging reach,
    reach at ground level, digging depth and digging height are the
    brochure's 8.30, 8.18, 5.48 and 9.34 m (four numbers, four unknowns);
  * the arm cylinder's two arms and their offset, so that its retracted and
    extended lengths fall at the arm's two limits and its largest force at
    the bucket lip, at the relief pressure, is the published ISO arm force
    of 61.8 kN, taken at the radius from the arm pin to the lip with the
    bucket in line with the arm (our reading; ISO 6015's sample pages give
    the radius, not the bucket's position).

Not modelled: the bucket's shape, and so the dumping height, the vertical
wall depth and the minimum swing radius, which depend on it; the boom's
bend; the cylinders' masses as separate bodies (mass.py lumps them).
"""
import math

import numpy as np

from . import published as pub

X0 = 0.20                     # [m] ours
R_BUCKET = 1.05               # [m] ours
THETA2_IN = math.radians(-150.0)
THETA3_CURL = math.radians(-150.0)
THETA3_OPEN = math.radians(30.0)
DEAD = {"boom": 0.50, "arm": 0.55}      # [m] ours
RODS = {"boom": 0.075, "arm": pub.CYLINDERS["arm"][3], "bucket": 0.065}   # [m]
PSI_BOOM = math.radians(20.0)  # ours

# fitted (fit_front), stored to the digits that the check compares
Z0 = 1.38910                  # [m]
THETA1 = (math.radians(-46.1821), math.radians(83.2024))
THETA2_OUT = math.radians(-14.2481)
ARM_CYLINDER = (2.24741, 0.66340, math.radians(18.2354))   # (a, c, offset), fit_arm_cylinder

TARGETS = {"reach": pub.MAX_DIGGING_REACH, "reach_ground": pub.MAX_REACH_AT_GROUND,
           "depth": pub.MAX_DIGGING_DEPTH, "height": pub.MAX_DIGGING_HEIGHT}


# ----------------------------------------------------------------------------
# Kinematics
# ----------------------------------------------------------------------------
def points(t1, t2, t3, z0=None):
    """B0, B1, B2 and the lip for the angles (theta1, theta2, theta3)."""
    z0 = Z0 if z0 is None else z0
    a1, a2 = t1, t1 + t2
    a3 = a2 + t3
    b0 = np.array([X0, z0])
    b1 = b0 + pub.BOOM_LENGTH * np.array([math.cos(a1), math.sin(a1)])
    b2 = b1 + pub.ARM_LENGTH * np.array([math.cos(a2), math.sin(a2)])
    lip = b2 + R_BUCKET * np.array([math.cos(a3), math.sin(a3)])
    return b0, b1, b2, lip


def inverse(reach, height, z0=None):
    """theta1 and theta2 that put the bucket pin B2 at (reach, height) from
    the swing axis and the ground, the arm folded down; None if out of the
    links' reach."""
    z0 = Z0 if z0 is None else z0
    dx, dz = reach - X0, height - z0
    d2 = dx * dx + dz * dz
    lb, la = pub.BOOM_LENGTH, pub.ARM_LENGTH
    c2 = (d2 - lb * lb - la * la) / (2 * lb * la)
    if abs(c2) > 1:
        return None
    t2 = -math.acos(c2)
    t1 = math.atan2(dz, dx) - math.atan2(la * math.sin(t2), lb + la * math.cos(t2))
    return t1, t2


def within(t1, t2, t3=None, theta1=None, theta2_out=None):
    lo, hi = THETA1 if theta1 is None else theta1
    out = THETA2_OUT if theta2_out is None else theta2_out
    ok = lo <= t1 <= hi and THETA2_IN <= t2 <= out
    if t3 is not None:
        ok = ok and THETA3_CURL <= t3 <= THETA3_OPEN
    return ok


# ----------------------------------------------------------------------------
# The working range, and the fit of the front to it
# ----------------------------------------------------------------------------
def working_range(z0=None, theta1=None, theta2_out=None, starts=24):
    """The lip's largest reach, reach at ground level, depth and height over
    the joints' ranges (bounded local optimisation from fixed starts)."""
    from scipy.optimize import minimize
    z0 = Z0 if z0 is None else z0
    lo, hi = THETA1 if theta1 is None else theta1
    out = THETA2_OUT if theta2_out is None else theta2_out
    bounds = [(lo, hi), (THETA2_IN, out), (THETA3_CURL, THETA3_OPEN)]
    starts = np.random.default_rng(0).uniform([b[0] for b in bounds], [b[1] for b in bounds],
                                              (starts, 3))

    def lip(a):
        return points(*a, z0=z0)[3]

    def best(f):
        return min(minimize(f, s, bounds=bounds, method="L-BFGS-B").fun for s in starts)

    result = {"reach": -best(lambda a: -lip(a)[0]), "height": -best(lambda a: -lip(a)[1]),
              "depth": -best(lambda a: lip(a)[1])}
    ground = {"type": "eq", "fun": lambda a: lip(a)[1]}
    vals = []
    for s in starts:
        r = minimize(lambda a: -lip(a)[0], s, bounds=bounds, constraints=[ground], method="SLSQP")
        if r.success and abs(lip(r.x)[1]) < 1e-7:
            vals.append(-r.fun)
    result["reach_ground"] = max(vals)
    return result


def fit_front():
    """Z0, THETA1 and THETA2_OUT from the brochure's four ranges."""
    from scipy.optimize import least_squares

    def residual(p):
        r = working_range(z0=p[0], theta1=(p[1], p[2]), theta2_out=p[3])
        return [r[k] - TARGETS[k] for k in ("reach", "reach_ground", "depth", "height")]

    d2r = math.radians
    sol = least_squares(residual, [1.4, d2r(-46), d2r(83), d2r(-14)],
                        bounds=([0.8, d2r(-80), d2r(40), d2r(-60)],
                                [2.5, d2r(-10), d2r(89), d2r(10)]),
                        diff_step=1e-5, xtol=1e-12, ftol=1e-12)
    return sol.x, residual(sol.x)


# ----------------------------------------------------------------------------
# The cylinders
# ----------------------------------------------------------------------------
def area(name, side="head"):
    n, bore, _, _ = pub.CYLINDERS[name]
    head = n * math.pi / 4 * bore * bore
    if side == "head":
        return head
    rod = RODS[name]
    return head - n * math.pi / 4 * rod * rod


def triangle_length(a, c, angle):
    return math.sqrt(a * a + c * c - 2 * a * c * math.cos(angle))


def boom_cylinder():
    """The boom cylinders' triangle: arms a (from B0 to the base pin on the
    upper structure) and c (from B0 to the rod pin on the boom) and the offset
    beta, with the angle between the arms psi = theta1 - beta. Retracted
    (dead length plus stroke) at the boom's lowest position, where
    psi = PSI_BOOM, and extended by the stroke at its highest."""
    stroke = pub.CYLINDERS["boom"][2]
    lmin, lmax = stroke + DEAD["boom"], 2 * stroke + DEAD["boom"]
    lo, hi = THETA1
    clo, chi = math.cos(PSI_BOOM), math.cos(PSI_BOOM + (hi - lo))
    ac = (lmax ** 2 - lmin ** 2) / (2 * (clo - chi))
    s = lmin ** 2 + 2 * ac * clo           # a^2 + c^2
    apc, amc = math.sqrt(s + 2 * ac), math.sqrt(max(s - 2 * ac, 0.0))
    return (apc + amc) / 2, (apc - amc) / 2, lo - PSI_BOOM


def boom_length(t1):
    a, c, beta = boom_cylinder()
    return triangle_length(a, c, t1 - beta)


def arm_length(t2, params=None):
    """The arm cylinder: arms a (from B1 back along the boom to its base pin)
    and c (from B1 to the rod pin on the arm's tail), the angle between them
    gamma = offset - theta2, so that the cylinder extends as the arm folds in
    (crowd)."""
    a, c, off = ARM_CYLINDER if params is None else params
    return triangle_length(a, c, off - t2)


def fit_arm_cylinder():
    from scipy.optimize import least_squares
    stroke = pub.CYLINDERS["arm"][2]
    lmin, lmax = stroke + DEAD["arm"], 2 * stroke + DEAD["arm"]
    lever_max = arm_lever_from_force()

    def residual(p):
        a, c, off = p
        angles = np.linspace(off - THETA2_OUT, off - THETA2_IN, 721)
        lever = max(a * c * math.sin(g) / triangle_length(a, c, g) for g in angles)
        return [triangle_length(a, c, off - THETA2_OUT) - lmin,
                triangle_length(a, c, off - THETA2_IN) - lmax, lever - lever_max]

    sol = least_squares(residual, [1.8, 0.6, math.radians(10)],
                        bounds=([0.3, 0.2, math.radians(-90)], [3.0, 1.5, math.radians(90)]),
                        xtol=1e-14, ftol=1e-14)
    return sol.x, residual(sol.x)


def arm_lever_from_force():
    """The arm cylinder's largest lever about B1 that the ISO arm force
    implies: F_ISO * radius / (relief pressure * head area), the radius from
    B1 to the lip with the bucket in line with the arm."""
    radius = pub.ARM_LENGTH + R_BUCKET
    return pub.ARM_FORCE_ISO * radius / (pub.RELIEF_IMPLEMENT * area("arm"))


def bucket_lever():
    """The bucket cylinder's constant lever about B2 that the ISO bucket
    force implies at the relief pressure, at the radius R_BUCKET."""
    return pub.BUCKET_FORCE_ISO * R_BUCKET / (pub.RELIEF_IMPLEMENT * area("bucket"))


def derivative(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)


def cylinder_forces(name):
    """The force that a cylinder can push (head side) and pull (rod side)
    with at the relief pressure [N]."""
    return pub.RELIEF_IMPLEMENT * area(name, "head"), -pub.RELIEF_IMPLEMENT * area(name, "rod")
