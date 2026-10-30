"""
The excavator's masses: what is published, what is ours, what is fitted.

Physical AI, Together, Chapter 14 (Section 14.1).

Published: the operating weight, 13,480 kg with 500 mm shoes and the front of
the lifting tables; the counterweight, 3,250 kg; the 0.50 m^3 bucket, 399 kg.

Ours: the split of the rest into the undercarriage, the upper structure, the
boom and the arm, and where each part's centre of mass lies, as below. The
cylinders are lumped: half of each boom cylinder with the upper structure,
half with the boom; the arm cylinder with the boom; the bucket cylinder and
the bucket's linkage with the arm. The inertias are those of solid boxes of
the sizes below (ours), about each part's centre of mass.

Fitted: the upper structure's centre of mass along the boom's direction,
X_UPPER, the one parameter that moves the machine's centre of mass without
changing its weight; fit_upper() chooses it so that the model's tipping
ratings (75 % of the tipping load) come closest, in the mean square of the
logarithms, to the brochure's ratings that are limited by tipping in the
rows of the lifting table at 3.0 and 0.0 m; the rows at 6.1 and -3.0 m are
kept to test it.
"""
import math

import numpy as np

from . import front
from . import published as pub

# name: mass [kg], centre of mass, box size [m] (ours unless said)
LOWER = dict(mass=4950.0, com=(0.0, 0.0, 0.45), box=(3.61, 2.49, 0.90))
COUNTERWEIGHT = dict(mass=pub.COUNTERWEIGHT, com=(-1.20, 0.0, 1.25), box=(0.55, 2.30, 0.70))
X_UPPER = -0.76134            # [m] fitted by fit_upper (see the docstring)
UPPER_MASS = (pub.OPERATING_WEIGHT - LOWER["mass"] - pub.COUNTERWEIGHT - 1180.0 - 660.0
              - pub.BUCKET_MASS)
UPPER = dict(mass=UPPER_MASS, com=(X_UPPER, 0.0, 1.40), box=(2.30, 2.40, 1.20))
# the front's parts: along their line from their pin, and across (up) from it
BOOM = dict(mass=1180.0, along=2.00, across=0.25, box=(4.60, 0.50, 0.60))
ARM = dict(mass=660.0, along=1.00, across=0.15, box=(2.50, 0.30, 0.40))
BUCKET = dict(mass=pub.BUCKET_MASS, along=0.45, across=0.10, box=(0.80, 0.86, 0.80))
SWING_HEIGHT = 0.95           # [m] the top of the swing bearing, the upper's frame (ours)


def total():
    return (LOWER["mass"] + COUNTERWEIGHT["mass"] + UPPER["mass"] + BOOM["mass"] + ARM["mass"]
            + BUCKET["mass"])


def box_inertia(mass, size):
    a, b, c = size
    return np.array([b * b + c * c, a * a + c * c, a * a + b * b]) * mass / 12.0


def front_parts(t1, t2, t3):
    """The boom's, the arm's and the bucket's mass and centre of mass in the
    front's plane (x along the boom's direction, z up)."""
    b0, b1, b2, _ = front.points(t1, t2, t3)
    a1, a2 = t1, t1 + t2
    a3 = a2 + t3

    def at(base, angle, part):
        u = np.array([math.cos(angle), math.sin(angle)])
        n = np.array([-math.sin(angle), math.cos(angle)])
        return base + part["along"] * u + part["across"] * n

    return [(BOOM["mass"], at(b0, a1, BOOM)), (ARM["mass"], at(b1, a2, ARM)),
            (BUCKET["mass"], at(b2, a3, BUCKET))]


def machine(t1, t2, t3, x_upper=None):
    """The machine's mass and its centre of mass along the boom's direction
    (the upper structure's x), without load."""
    xu = X_UPPER if x_upper is None else x_upper
    parts = front_parts(t1, t2, t3)
    m = total()
    mx = (UPPER["mass"] * xu + COUNTERWEIGHT["mass"] * COUNTERWEIGHT["com"][0]
          + LOWER["mass"] * LOWER["com"][0] + sum(p * c[0] for p, c in parts))
    return m, mx / m


def tampere_moments(attachment=pub.TAMPERE_ATTACHMENT):
    """The model's counterparts of the identified gravitational parameters,
    with an attachment of the given mass at the bucket pin in place of the
    bucket: the magnitudes |(pi_s3, pi_s4)| = |M_bu a3 + M_st r3| and
    |(pi_s5, pi_s6)| = |(M_bu + M_st) a2 + M_bo r2| [kg m], which do not
    depend on where each link's angle is measured from."""
    stick = np.array([attachment * pub.ARM_LENGTH + ARM["mass"] * ARM["along"],
                      ARM["mass"] * ARM["across"]])
    boom = np.array([(attachment + ARM["mass"]) * pub.BOOM_LENGTH + BOOM["mass"] * BOOM["along"],
                     BOOM["mass"] * BOOM["across"]])
    return float(np.linalg.norm(stick)), float(np.linalg.norm(boom))


def tampere_identified():
    s = pub.TAMPERE_STATIC
    return (math.hypot(s["pi_s3"], s["pi_s4"]), math.hypot(s["pi_s5"], s["pi_s6"]))
