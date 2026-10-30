"""
The excavator's lifting capacity: tipping and hydraulic, as ISO 10567 has it.

Physical AI, Together, Chapter 14 (Section 14.1).

The lift point is the bucket pin B2 (ISO 10567's second lift point, the
centreline of the bucket's pivot pin on the arm; the brochure says "bucket
hook", whose place on the bucket it does not give), with the bucket curled
(theta3 at its curled limit, ours). The tipping lines of a crawler machine
(ISO 10567, clause 4, sample pages): over the end, the line through the
centres of the idlers or sprockets, here at half the length of track on
ground, 1.44 m from the swing axis (our reading: the brochure gives the
length on ground, not the idlers' centres); over the side, the line of the
track rollers' contact with the track, here at half the track gauge,
0.995 m.

The tipping load at a lift point at reach A over a tipping line at distance
e from the swing axis is the load whose moment about the line equals the
machine's: W (A - e) = M (e - x_c), with M and x_c the machine's mass and
centre of mass along the boom's direction (mass.machine). Over the side the
upper structure is turned through 90 degrees; the undercarriage's centre of
mass is on the swing axis, so the same x_c serves. The hydraulic capacity is
the load at which a cylinder reaches its relief pressure, pushing or pulling,
while it holds the front still (hydraulic). The brochure's ratings are the
smaller of 75 % of the tipping load and 87 % of the hydraulic capacity.
"""
import math

from . import front, mass
from . import published as pub

G = 9.81
FRONT_LINE = pub.TRACK_ON_GROUND / 2.0      # [m] our reading of ISO 10567
SIDE_LINE = pub.TRACK_GAUGE / 2.0
LIFT_BUCKET = front.THETA3_CURL              # the bucket curled while lifting (ours)


def pose(reach, height):
    sol = front.inverse(reach, height)
    if sol is None or not front.within(*sol):
        return None
    return sol[0], sol[1], LIFT_BUCKET


def tipping(reach, height, side=False, x_upper=None):
    """The tipping load [kg] at the lift point, over the end or over the side."""
    p = pose(reach, height)
    if p is None:
        return None
    m, xc = mass.machine(*p, x_upper=x_upper)
    e = SIDE_LINE if side else FRONT_LINE
    return m * (e - xc) / (reach - e)


def potential(t1, t2, t3, load):
    """The front's and the load's potential energy [J] (the rest of the
    machine does not move with theta1 or theta2)."""
    parts = mass.front_parts(t1, t2, t3)
    b2 = front.points(t1, t2, t3)[2]
    return G * (sum(m * c[1] for m, c in parts) + load * b2[1])


def generalised(t1, t2, t3, load, h=1e-6):
    q1 = -(potential(t1 + h, t2, t3, load) - potential(t1 - h, t2, t3, load)) / (2 * h)
    q2 = -(potential(t1, t2 + h, t3, load) - potential(t1, t2 - h, t3, load)) / (2 * h)
    return q1, q2


def hydraulic(reach, height):
    """The load [kg] at which the boom's or the arm's cylinder reaches its
    relief pressure holding the front at the lift point, each cylinder's, and
    the cylinder that limits. A cylinder's force F holds its joint when
    F dL/dtheta + Q = 0, with Q the generalised gravity force on the joint;
    F lies between the pull of the rod side and the push of the head side."""
    p = pose(reach, height)
    if p is None:
        return None
    t1, t2, t3 = p
    dl_boom = front.derivative(front.boom_length, t1)
    dl_arm = front.derivative(front.arm_length, t2)
    q0 = generalised(t1, t2, t3, 0.0)
    q1 = generalised(t1, t2, t3, 1.0)
    caps = {}
    for name, dl, k in (("boom", dl_boom, 0), ("arm", dl_arm, 1)):
        push, pull = front.cylinder_forces(name)
        f0, f1 = -q0[k] / dl, -(q1[k] - q0[k]) / dl      # F = f0 + f1 W
        if f1 > 0:
            caps[name] = (push - f0) / f1
        elif f1 < 0:
            caps[name] = (pull - f0) / f1
        else:
            caps[name] = math.inf
    limit = min(caps, key=caps.get)
    return caps[limit], caps, limit


def rating(reach, height, side=False, x_upper=None):
    """The model's rating [kg]: the smaller of 75 % of the tipping load and
    87 % of the hydraulic capacity, and which of the two limits."""
    tip = tipping(reach, height, side, x_upper)
    hyd = hydraulic(reach, height)
    if tip is None or hyd is None:
        return None
    t, h = pub.TIPPING_SHARE * tip, pub.HYDRAULIC_SHARE * hyd[0]
    return (t, "tipping") if t <= h else (h, "hydraulic " + hyd[2])


def tipping_entries(role="fit"):
    """The brochure's ratings limited by tipping in the rows of the given
    role, "fit" (3.0 and 0.0 m, to which X_UPPER is fitted) or "test" (6.1
    and -3.0 m): (height, reach, side, rating)."""
    out = []
    for (b, a), (cf, hf, cs, hs, kind) in sorted(pub.LIFT_TABLE.items()):
        if kind != role:
            continue
        if not hf:
            out.append((b, a, False, cf))
        if not hs:
            out.append((b, a, True, cs))
    return out


def fit_upper():
    """X_UPPER that makes the model's tipping ratings closest to the
    brochure's tipping-limited ratings (least squares of the logarithms)."""
    from scipy.optimize import minimize_scalar
    entries = tipping_entries()

    def cost(xu):
        return sum(math.log(pub.TIPPING_SHARE * tipping(a, b, side, xu) / r) ** 2
                   for b, a, side, r in entries)

    res = minimize_scalar(cost, bounds=(-2.0, 1.0), method="bounded",
                          options={"xatol": 1e-9})
    return res.x, math.sqrt(cost(res.x) / len(entries))
