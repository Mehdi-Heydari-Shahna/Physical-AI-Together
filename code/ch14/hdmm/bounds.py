"""
The bounds behind the heavy machine's norms: Lemma 14.1 and Proposition 14.2.

Physical AI, Together, Chapter 14 (Section 14.3).

The package of governed.py declares its two families of constraints on the
rigid machine at the state x = (s, v, a, slew, lift, tilt):

  * the margin against tipping over: the distance of the rigid machine's
    zero-moment point inside the rectangle of the wheels' contacts,
    d_t(x; m) = min(AXLE - |p_x|, HALF_TRACK - |p_y|), for the payload m;
  * the clearance: the front-most point's distance behind the stop line,
    d_c(x) = S_LINE - s - labels.front_point(arm).

The rigid machine's zero-moment point is affine in the base's motion
u = (a, lam, w2, alpha): the acceleration of the base's middle along the
route and across it (lam = v^2 k), the squared yaw rate (w2 = (v k)^2) and
the yaw acceleration (alpha = a k + v^2 dk/ds), k the route's curvature:

  p_x = sum_i m_i [x_i - h_i (a - alpha y_i - w2 x_i) / g] / M
        - (alpha J_yz + w2 J_xz) / (M g),
  p_y = sum_i m_i [y_i - h_i (lam + alpha x_i - w2 y_i) / g] / M
        + (alpha J_xz - w2 J_yz) / (M g),

with (x_i, y_i) each part's centre of mass relative to the base's middle,
h_i its height above the ground, M the machine's mass and J the parts'
products of inertia in the base's frame (zmp_coefficients). It equals
model.motion_basis with model.wrench_zmp, MuJoCo's inverse dynamics, to
rounding (check_zmp, on a fixed grid of poses, payloads and motions).

Lemma 14.1 bounds how far a change of each component of the state moves the
two points: clearance_constants for every two states, and
tipping_constants for a state whose motion the package's segments allow and
every state within e_max of it in the tipping norm (governed.NORM_TIP), with
any payload up to model.PAYLOAD_MAX. A joint's angle moves a part at
distance rho from its axis by at most rho per radian; the parts' heights,
places and the motions allowed bound what that move does to the point; the
base's acceleration, speed and arc length enter only through the motion u.
measured_rates gives the largest rates at the states that the script
evaluates, which must lie below the constants.

Proposition 14.2 needs, besides, what the funnel's spread around the states
that the test evaluates can cost the front edge's distance, to the first
order at the planned pose with a bound on the second (spread), what the
test's sampling of those states can cost between them (turn_sampling,
lift_sampling), that the side and rear edges stay farther than any level
(both in spread), and the clearance's corner (clearance_excess).
"""
import itertools
import math

import numpy as np

from . import labels as hl
from . import model, route

G = hl.G
EZ = np.array([0.0, 0.0, 1.0])
SHOULDER = np.array([model.SLEW[0], 0.0, model.SLEW[1]])   # the slew's and the lift's axes
TRANSITION = hl.TRANSITION
RADIUS = route.RADIUS
K1 = 1.0 / RADIUS                                        # the route's largest curvature [1/m]
K2 = math.pi / (2.0 * TRANSITION * RADIUS)               # its largest rate along the route [1/m^2]
K3 = math.pi ** 2 / (2.0 * TRANSITION ** 2 * RADIUS)     # the rate's [1/m^3]
K4 = math.pi ** 3 / (2.0 * TRANSITION ** 3 * RADIUS)     # and the next [1/m^4]


def _rot_z(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_y(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _box(m, dims):
    a, b, c = dims
    return np.diag([m * (b * b + c * c), m * (a * a + c * c), m * (a * a + b * b)]) / 12.0


def _skew(u):
    return np.array([[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]])


# ----------------------------------------------------------------------------
# The rigid machine's zero-moment point
# ----------------------------------------------------------------------------
def parts(pose, payload):
    """The parts of the rigid machine of model.build in the pose (slew, lift,
    tilt) with the payload: each its mass, its centre of mass (x, y, h)
    relative to the base's middle on the ground, its inertia tensor in the
    base's frame, and for each joint the derivatives of both in the joint's
    angle (zero where the joint does not move the part). The base and the
    turret, which turns about its own axis, do not move with the arm."""
    slew, lift, tilt = pose
    rb = _rot_z(slew) @ _rot_y(-lift)
    rs = rb @ _rot_y(-tilt)
    elbow = SHOULDER + rb @ np.array([model.BOOM["length"], 0.0, 0.0])
    axis = _rot_z(slew) @ np.array([0.0, -1.0, 0.0])          # the lift's and the tilt's axis
    axes = ((EZ, SHOULDER), (axis, SHOULDER), (axis, elbow))
    tip_mass = model.TOOL + payload
    rows = [
        (model.BASE["mass"], np.array(model.BASE["com"], dtype=float),
         _box(model.BASE["mass"], model.BASE["box"]), ()),
        (1e-3, SHOULDER.copy(), _box(1e-3, (0.1, 0.1, 0.1)), ()),
        (model.BOOM["mass"], SHOULDER + rb @ np.array([model.BOOM["com"], 0.0, 0.0]),
         rb @ _box(model.BOOM["mass"], model.BOOM["box"]) @ rb.T, (0, 1)),
        (model.STICK["mass"], elbow + rs @ np.array([model.STICK["com"], 0.0, 0.0]),
         rs @ _box(model.STICK["mass"], model.STICK["box"]) @ rs.T, (0, 1, 2)),
        (tip_mass, elbow + rs @ np.array([model.STICK["length"], 0.0, 0.0]),
         rs @ _box(tip_mass, (0.4, 0.4, 0.4)) @ rs.T, (0, 1, 2)),
    ]
    out = []
    for m, r, inertia, moved in rows:
        dr, di = [], []
        for j, (u, o) in enumerate(axes):
            if j in moved:
                dr.append(np.cross(u, r - o))
                di.append(_skew(u) @ inertia - inertia @ _skew(u))
            else:
                dr.append(np.zeros(3))
                di.append(np.zeros((3, 3)))
        out.append((m, r, inertia, dr, di))
    return out


def zmp_coefficients(pose, payload):
    """The zero-moment point p = p0 + C u, u = (a, lam, w2, alpha), relative
    to the base's middle: p0 (2,) and C (2, 4), and their derivatives in the
    slew, the lift and the tilt, dp0 (3, 2) and dC (3, 2, 4)."""
    ps = parts(pose, payload)
    mass = sum(x[0] for x in ps)
    p0, c = np.zeros(2), np.zeros((2, 4))
    dp0, dc = np.zeros((3, 2)), np.zeros((3, 2, 4))

    def rows(x, y, h, inertia):
        cx = m * np.array([-h / G, 0.0, h * x / G, h * y / G]) / mass \
            + np.array([0.0, 0.0, -inertia[0, 2], -inertia[1, 2]]) / (mass * G)
        cy = m * np.array([0.0, -h / G, h * y / G, -h * x / G]) / mass \
            + np.array([0.0, 0.0, -inertia[1, 2], inertia[0, 2]]) / (mass * G)
        return cx, cy

    for m, r, inertia, dr, di in ps:
        x, y, h = r
        p0 += m * np.array([x, y]) / mass
        cx, cy = rows(x, y, h, inertia)
        c[0] += cx
        c[1] += cy
        for j in range(3):
            dx, dy, dh = dr[j]
            dp0[j] += m * np.array([dx, dy]) / mass
            # the derivative of each row: the product rule on h x, h y, and the inertia's change
            dc[j, 0] += m * np.array([-dh / G, 0.0, (dh * x + h * dx) / G, (dh * y + h * dy) / G]) / mass \
                + np.array([0.0, 0.0, -di[j][0, 2], -di[j][1, 2]]) / (mass * G)
            dc[j, 1] += m * np.array([0.0, -dh / G, (dh * y + h * dy) / G, -(dh * x + h * dx) / G]) / mass \
                + np.array([0.0, 0.0, -di[j][1, 2], di[j][0, 2]]) / (mass * G)
    return p0, c, dp0, dc


def motion(v, a, k, dk):
    """u = (a, lam, w2, alpha) for the speed v, the acceleration a along the
    route, the curvature k and its rate dk along the route."""
    return np.array([a, v * v * k, (v * k) ** 2, a * k + v * v * dk])


def distance(pose, payload, u):
    """d_t: the rigid machine's zero-moment point's distance inside the
    rectangle of the contacts [m]."""
    p0, c, _, _ = zmp_coefficients(pose, payload)
    p = p0 + c @ np.asarray(u, dtype=float)
    return min(model.AXLE - abs(p[0]), model.HALF_TRACK - abs(p[1]))


def check_zmp():
    """The largest difference [m] of zmp_coefficients' point from
    model.motion_basis with model.wrench_zmp, over a fixed grid of poses,
    payloads and motions."""
    worst = 0.0
    for slew, lift, tilt in itertools.product((0.0, 0.7, -2.1), (0.0, 0.35, 1.0), (-1.2, 0.0, -2.5)):
        pose = (slew, lift, tilt)
        for payload in (0.0, 270.0, model.PAYLOAD_MAX):
            p0, c, _, _ = zmp_coefficients(pose, payload)
            b = model.motion_basis(payload, pose)
            for u in itertools.product((-4.0, 1.5), (-1.2, 0.9), (0.0, 0.3), (-0.8, 1.1)):
                u = np.array(u)
                w = b["q0"] + u[0] * b["ax"] + u[1] * b["ay"] + u[2] * b["w2"] + u[3] * b["alpha"]
                px, py, _ = model.wrench_zmp(w)
                p = p0 + c @ u
                worst = max(worst, abs(float(px) - p[0]), abs(float(py) - p[1]))
    return worst


# ----------------------------------------------------------------------------
# Lemma 14.1: the constants
# ----------------------------------------------------------------------------
def _spread(m, dims):
    """The largest less the smallest principal moment of inertia of a box."""
    d = np.diag(_box(m, dims))
    return float(d.max() - d.min())


# Each part: its mass (the tool's without the payload), its largest distance
# from the slew's and the lift's axes (which pass through the shoulder) and
# from the tilt's (through the elbow), and the spread of its moments of
# inertia; the tool is a cube, whose inertia no rotation changes.
ARM = {
    "boom": (model.BOOM["mass"], model.BOOM["com"], 0.0,
             _spread(model.BOOM["mass"], model.BOOM["box"])),
    "stick": (model.STICK["mass"], model.BOOM["length"] + model.STICK["com"], model.STICK["com"],
              _spread(model.STICK["mass"], model.STICK["box"])),
    "tip": (model.TOOL, model.BOOM["length"] + model.STICK["length"], model.STICK["length"], 0.0),
}
FIXED_MASS = model.BASE["mass"] + 1e-3        # the base and the turret
MASS = FIXED_MASS + sum(x[0] for x in ARM.values())
J_BAR = sum(x[3] for x in ARM.values()) / 2.0  # the largest product of inertia of the arm's parts [kg m^2]


def _parts_extent():
    """Every part's mass (the tool's without the payload), and its largest
    height above the ground and largest |x| and |y| relative to the base's
    middle, over every angle of the joints."""
    out = [(model.BASE["mass"], model.BASE["com"][2], 0.0, 0.0), (1e-3, SHOULDER[2], SHOULDER[0], 0.0)]
    for m, d, _, _ in ARM.values():
        out.append((m, SHOULDER[2] + d, SHOULDER[0] + d, d))
    return out


def _over_payloads(f):
    """The largest of f(m) over 0 <= m <= PAYLOAD_MAX: every bound here is a
    ratio of two functions affine in m, the payload's, monotone in m, so an
    end binds."""
    return max(f(0.0), f(model.PAYLOAD_MAX))


def _motion_bounds(a_max, v_max, k_max, k2_max):
    """The largest |lam|, w2 and |alpha| for |a| <= a_max, |v| <= v_max,
    |k| <= k_max and |dk/ds| <= k2_max."""
    return v_max * v_max * k_max, (v_max * k_max) ** 2, a_max * k_max + v_max * v_max * k2_max


def joint_constants(a_max, v_max, k_max, k2_max):
    """For each joint, slew, lift and tilt, the most that a change of its
    angle by a radian moves the zero-moment point along x or along y, over
    every pose and payload, at the motions bounded so: each part that the
    joint moves moves by at most its distance rho from the joint's axis, and
    the point moves with it by at most rho times the size of the gradient of
    x - h A_x / g (or y - h A_y / g) in the part's place (Cauchy-Schwarz),
    h A / g the part's height times its acceleration over g; the slew moves
    no part up or down; and the parts' rotation changes their products of
    inertia by at most the spread of their moments per radian."""
    lam, w2, alpha = _motion_bounds(a_max, v_max, k_max, k2_max)
    out = []
    for j in range(3):
        def bound(m, axis):
            total = 0.0
            spread = 0.0
            for name, (mass, d_sh, d_el, ds) in ARM.items():
                rho = d_sh if j < 2 else d_el
                if rho == 0.0:
                    continue
                mk = mass + (m if name == "tip" else 0.0)
                h, x, y = SHOULDER[2] + d_sh, SHOULDER[0] + d_sh, d_sh
                c1, c2 = 1.0 + h * w2 / G, h * alpha / G
                acc = (a_max + alpha * y + w2 * x) if axis == "x" else (lam + alpha * x + w2 * y)
                n = math.sqrt(c1 * c1 + c2 * c2 + (0.0 if j == 0 else (acc / G) ** 2))
                total += mk * rho * n
                spread += ds
            return (total + (alpha + w2) * spread / G) / (MASS + m)
        out.append(max(_over_payloads(lambda m: bound(m, "x")), _over_payloads(lambda m: bound(m, "y"))))
    return out


def acceleration_constant(k_max):
    """The most that a change of the base's acceleration by 1 m/s^2 moves
    the point, at curvatures up to k_max: p is affine in a, through each
    part's h A / g and the yaw acceleration's term in the inertia."""
    ext = _parts_extent()

    def bx(m):
        s = sum((mk + (m if i == len(ext) - 1 else 0.0)) * h * (1.0 + k_max * y)
                for i, (mk, h, x, y) in enumerate(ext))
        return (s + k_max * J_BAR) / (G * (MASS + m))

    def by(m):
        s = sum((mk + (m if i == len(ext) - 1 else 0.0)) * h * k_max * x
                for i, (mk, h, x, y) in enumerate(ext))
        return (s + k_max * J_BAR) / (G * (MASS + m))
    return max(_over_payloads(bx), _over_payloads(by))


def speed_constant(v_max, k_max, k2_max):
    """The most that a change of the speed by 1 m/s moves the point, at
    speeds up to v_max: through lam, w2 and alpha, whose derivatives in v are
    2 v k, 2 v k^2 and 2 v dk/ds."""
    ext = _parts_extent()

    def b(m, axis):
        s = 0.0
        for i, (mk, h, x, y) in enumerate(ext):
            mk = mk + (m if i == len(ext) - 1 else 0.0)
            s += mk * h * ((k2_max * y + k_max ** 2 * x) if axis == "x"
                           else (k_max + k2_max * x + k_max ** 2 * y))
        return 2.0 * v_max * (s + (k2_max + k_max ** 2) * J_BAR) / (G * (MASS + m))
    return max(_over_payloads(lambda m: b(m, "x")), _over_payloads(lambda m: b(m, "y")))


def arc_constant(a_max, v_max, k_max=K1, k2_max=K2, k3_max=K3):
    """The most that a change of the arc length by 1 m moves the point, at
    accelerations up to a_max and speeds up to v_max: through the route's
    curvature k and its rate dk/ds, whose changes along the route are at most
    k2_max and k3_max per metre."""
    ext = _parts_extent()
    v2 = v_max * v_max

    def b(m):
        def S(f):
            return sum((mk + (m if i == len(ext) - 1 else 0.0)) * f(h, x, y)
                       for i, (mk, h, x, y) in enumerate(ext))
        # derivatives of p_x and p_y in k and in dk/ds
        dx_k = S(lambda h, x, y: h * (a_max * y + 2.0 * v2 * k_max * x)) + (a_max + 2.0 * v2 * k_max) * J_BAR
        dx_k2 = v2 * (S(lambda h, x, y: h * y) + J_BAR)
        dy_k = S(lambda h, x, y: h * (v2 + a_max * x + 2.0 * v2 * k_max * y)) + (a_max + 2.0 * v2 * k_max) * J_BAR
        dy_k2 = v2 * (S(lambda h, x, y: h * x) + J_BAR)
        return max(dx_k * k2_max + dx_k2 * k3_max, dy_k * k2_max + dy_k2 * k3_max) / (G * (MASS + m))
    return _over_payloads(b)


def creep_curvature(ds):
    """The largest curvature and rate of curvature [1/m, 1/m^2] within ds
    of a transition's ends, where a straight segment's funnel can reach."""
    return ((1.0 - math.cos(math.pi * ds / TRANSITION)) / (2.0 * RADIUS),
            math.pi * math.sin(math.pi * ds / TRANSITION) / (2.0 * TRANSITION * RADIUS))


def tipping_constants(weights, e_max, ds):
    """Lemma 14.1(b): the constants (arc length [m/m], speed [s],
    acceleration [s^2], slew, lift, tilt [m/rad]) for a state x whose motion
    the package's segments allow (speed within [0, SPEED], acceleration
    within [-BRAKES, DRIVE] on a straight, its funnel reaching at most ds into
    a transition, and within +-TURN_BRAKING where the route turns) and every
    state within e_max of x in the norm of the weights (s, v, a, slew, lift,
    tilt). Change the joints first, at x's motion; then the acceleration, the
    speed and the arc length, each within e_max over its weight of x's."""
    kds, k2ds = creep_curvature(ds)
    straight = joint_constants(hl.BRAKES, hl.SPEED, kds, k2ds)
    turning = joint_constants(hl.TURN_BRAKING, hl.SPEED, K1, K2)
    theta = [max(a, b) for a, b in zip(straight, turning)]
    v_max = hl.SPEED + e_max / weights[1]
    a_max = hl.BRAKES + e_max / weights[2]
    return [arc_constant(a_max, v_max), speed_constant(v_max, K1, K2),
            acceleration_constant(K1)] + theta


def clearance_constants():
    """Lemma 14.1(a): the most that a change of each component moves the
    front-most point along the route: the arc length one for one; a joint by
    the tool's largest distance from its axis per radian, the tool's box and
    the chassis's front being fixed to it or to the base; the speed and the
    acceleration not at all."""
    return [1.0, 0.0, 0.0, ARM["tip"][1], ARM["tip"][1], ARM["tip"][2]]


def _points(pose, payload, v, a, k, dk):
    """The zero-moment point (2, n) at many states (arrays of the speed, the
    acceleration, the curvature and its rate) in one pose."""
    p0, c, _, _ = zmp_coefficients(pose, payload)
    u = np.stack([a, v * v * k, (v * k) ** 2, a * k + v * v * dk])
    return p0[:, None] + c @ u


def _distances(p):
    return np.minimum(model.AXLE - np.abs(p[0]), model.HALF_TRACK - np.abs(p[1]))


def measured_rates(poses, payloads, turn_speeds, h=1e-4):
    """The largest move of the zero-moment point along either axis of the
    base, and the largest change of d_t, per unit of each component (s, v,
    a, slew, lift, tilt), by central differences of h, over the poses and
    payloads, on a straight at accelerations of -BRAKES, -1, 0 and DRIVE, and
    at the states of a turn of the yard's radius every 0.05 m at the speeds
    given for each pose name (keeping them, or changing them at
    +-TURN_BRAKING). Returns the two arrays of six."""
    moves, out = np.zeros(6), np.zeros(6)
    entry, arc, _ = hl.turn_length(RADIUS)
    ss = np.arange(-1.0, 2.0 * entry + arc + 1.0, 0.05)
    for name, pose in poses:
        rows = [(0.0, 0.5 * hl.SPEED, a) for a in (-hl.BRAKES, -1.0, 0.0, hl.DRIVE)]
        rows += [(s, v, a) for v in turn_speeds.get(name, ()) for s in ss
                 for a in (-hl.TURN_BRAKING, 0.0, hl.TURN_BRAKING)]
        s, v, a = (np.array(x, dtype=float) for x in zip(*rows))
        k0, dk0 = hl.route_curvature(s, RADIUS)
        kp, dkp = hl.route_curvature(s + h, RADIUS)
        km, dkm = hl.route_curvature(s - h, RADIUS)
        for m in payloads:
            pairs = [((pose, v, a, kp, dkp), (pose, v, a, km, dkm)),
                     ((pose, v + h, a, k0, dk0), (pose, v - h, a, k0, dk0)),
                     ((pose, v, a + h, k0, dk0), (pose, v, a - h, k0, dk0))]
            for j in range(3):
                e = np.zeros(3)
                e[j] = h
                pairs.append(((tuple(np.array(pose) + e), v, a, k0, dk0),
                              (tuple(np.array(pose) - e), v, a, k0, dk0)))
            for i, (up, dn) in enumerate(pairs):
                p1, p2 = _points(up[0], m, *up[1:]), _points(dn[0], m, *dn[1:])
                moves[i] = max(moves[i], float(np.max(np.abs(p1 - p2))) / (2 * h))
                out[i] = max(out[i], float(np.max(np.abs(_distances(p1) - _distances(p2)))) / (2 * h))
    return moves, out


# ----------------------------------------------------------------------------
# Proposition 14.2: the funnel's spread and the test's sampling
# ----------------------------------------------------------------------------
def second_bound(a_max, v_max, k_max, k2_max):
    """A bound on every second derivative of p_x and of p_y in the joints'
    angles, over every pose and payload: a part's place has second
    derivatives of at most its distance from the shoulder, and its products
    of inertia of at most three times the spread of its moments."""
    lam, w2, alpha = _motion_bounds(a_max, v_max, k_max, k2_max)

    def b(m):
        total, spread = 0.0, 0.0
        for name, (mass, d_sh, _, ds) in ARM.items():
            mk = mass + (m if name == "tip" else 0.0)
            h, x, y = SHOULDER[2] + d_sh, SHOULDER[0] + d_sh, d_sh
            c1, c2 = 1.0 + h * w2 / G, h * alpha / G
            n = max(math.sqrt(c1 * c1 + c2 * c2 + ((a_max + alpha * y + w2 * x) / G) ** 2),
                    math.sqrt(c1 * c1 + c2 * c2 + ((lam + alpha * x + w2 * y) / G) ** 2))
            total += mk * (n * d_sh + 2.0 * (w2 + alpha) * d_sh * d_sh / G)
            spread += ds
        return (total + 3.0 * (alpha + w2) * spread / G) / (MASS + m)
    return _over_payloads(b)


def _corners(a_lo, a_hi, v_max, k_max, k2_max):
    """The corners of a box of the motions u = (a, lam, w2, alpha) that holds
    every motion with a within [a_lo, a_hi], speeds up to v_max, |k| <= k_max
    and |dk/ds| <= k2_max."""
    lam, w2, alpha = _motion_bounds(max(abs(a_lo), abs(a_hi)), v_max, k_max, k2_max)
    return [np.array(u) for u in itertools.product((a_lo, a_hi), (-lam, lam), (0.0, w2), (-alpha, alpha))]


def spread(pose, payloads, bounds, a_lo, a_hi, v_max, turn, ds):
    """What the funnel's spread around a planned state, the arm within
    bounds (slew, lift, tilt) of the pose and the arc length within ds of the
    commanded one, can cost each edge's distance, for the payloads (an
    interval's ends) and every motion of the box: on a straight (turn False)
    the planned state's curvature is 0 and the creep reaches at most ds into a
    transition; in a turn, |k| <= K1 and |dk/ds| <= K2. Returns the most it
    costs the front edge and the side edges [m], the side and the rear edges'
    least distance at the planned states less their costs, and the front
    edge's largest distance there at the heaviest payload [m]."""
    kds, k2ds = creep_curvature(ds)
    k_max, k2_max = (K1, K2) if turn else (kds, k2ds)
    a_max = max(abs(a_lo), abs(a_hi))
    b = np.asarray(bounds, dtype=float)
    r2 = 0.5 * second_bound(a_max, v_max, k_max, k2_max) * float(b.sum()) ** 2
    front = side = 0.0
    centres = []
    for m in payloads:
        p0, c, dp0, dc = zmp_coefficients(pose, m)
        if turn:     # the creep changes k by K2 ds and dk/ds by K3 ds, at the pose
            creep = ((np.abs(c[:, 1]) * v_max ** 2 + 2.0 * np.abs(c[:, 2]) * v_max ** 2 * K1
                      + np.abs(c[:, 3]) * a_max) * K2 + np.abs(c[:, 3]) * v_max ** 2 * K3) * ds
        else:        # from k = 0 to at most kds, and dk/ds to at most k2ds
            creep = (np.abs(c[:, 1]) * v_max ** 2 * kds + np.abs(c[:, 2]) * (v_max * kds) ** 2
                     + np.abs(c[:, 3]) * (a_max * kds + v_max ** 2 * k2ds))
        for u in _corners(a_lo, a_hi, v_max, k_max, k2_max):
            grad = dp0 + np.einsum("jik,k->ji", dc, u)
            front = max(front, float(np.abs(grad[:, 0]) @ b) + r2 + float(creep[0]))
            side = max(side, float(np.abs(grad[:, 1]) @ b) + r2 + float(creep[1]))
            centres.append((m, p0 + c @ u))
    heaviest = max(payloads)
    side_least = min(model.HALF_TRACK - abs(p[1]) for _, p in centres) - side
    rear_least = min(model.AXLE + p[0] for _, p in centres) - front
    front_most = max(model.AXLE - p[0] for m, p in centres if m == heaviest)
    return dict(front=front, side=side, side_least=side_least, rear_least=rear_least,
                front_most=front_most, second=r2)


def turn_sampling(pose, payloads, v_max, period, dt):
    """What the test's sampling of a turn segment's own states every dt can
    cost the front edge's distance between them: dt^2 / 8 times a bound on
    its second derivative in time along the planned motion, the arm held
    unslewed, where p_x is affine in a and in w2 alone (its coefficients of
    lam and alpha vanish, so p_x has no kink where the route's transitions
    meet the straights and the arc). The acceleration follows a half-cosine
    over the period from one within +-TURN_BRAKING to another; the
    curvature's derivatives are at most K1, K2 and K3."""
    a_max = hl.TURN_BRAKING
    jump = 2.0 * a_max
    a1 = jump * math.pi / (2.0 * period)              # |da/dt|
    a2 = jump * math.pi ** 2 / (2.0 * period ** 2)    # |d2a/dt2|
    v = v_max
    w2_2 = (2.0 * (a_max ** 2 + v * a1) * K1 ** 2 + 10.0 * v * v * a_max * K1 * K2
            + 2.0 * v ** 4 * (K2 ** 2 + K1 * K3))
    worst = 0.0
    for m in payloads:
        _, c, _, _ = zmp_coefficients(pose, m)
        if abs(c[0, 1]) > 1e-12 or abs(c[0, 3]) > 1e-12:
            raise ValueError("p_x depends on lam or alpha: the pose is slewed")
        worst = max(worst, abs(c[0, 0]) * a2 + abs(c[0, 2]) * w2_2)
    return dt * dt / 8.0 * worst


def lift_sampling(payloads, lag, grid, lift_time):
    """What the test's sampling of a lift's path can cost the front edge's
    distance between the poses that it takes, on a straight, where p_x is
    affine in a, within [-BRAKES, DRIVE]: (gap^2 / 8) times a bound on the
    second derivative of p_x along the path, gap the largest gap between the
    poses that the test takes (the window behind, or a grid step); and near
    the lift's end, where the pose that the test takes at the end carries the
    smaller allowance, what the distance can change from the last pose before
    the end, at most 1 - w(lift_time - grid) of the path away."""
    carry, reach = np.array(model.POSES["carry"]), np.array(model.POSES["reach"])
    path = reach - carry
    th1, th2 = abs(path[1]), abs(path[1] + path[2])     # the boom's and the stick's angle per path
    b_boom = model.BOOM["com"] * th1 ** 2
    b_stick = model.BOOM["length"] * th1 ** 2 + model.STICK["com"] * th2 ** 2
    b_tip = model.BOOM["length"] * th1 ** 2 + model.STICK["length"] * th2 ** 2
    f = 1.0 + hl.BRAKES / G

    def second(m):
        return f * (model.BOOM["mass"] * b_boom + model.STICK["mass"] * b_stick
                    + (model.TOOL + m) * b_tip) / (MASS + m)
    d2 = _over_payloads(second)
    gap = max(lag, math.pi / (2.0 * lift_time) * grid)
    end_gap = 0.5 * (1.0 + math.cos(math.pi * (lift_time - grid) / lift_time))
    slope = 0.0
    for m in payloads:
        _, _, dp0, dc = zmp_coefficients(tuple(reach), m)
        for a in (-hl.BRAKES, hl.DRIVE):
            grad = dp0 + np.einsum("jik,k->ji", dc, np.array([a, 0.0, 0.0, 0.0]))
            slope = max(slope, abs(float(grad[:, 0] @ path)))
    return dict(between=gap * gap / 8.0 * d2, near_end=(slope + d2 * end_gap) * end_gap,
                second=d2, gap=gap, end_gap=end_gap)


def lift_spread(payloads, bounds, ds, n=2001):
    """spread over the poses of the lift's path every 1/(n - 1) of it, on a
    straight, with the remainder between those poses: the first-order cost
    changes along the path by at most the second-derivative bound times the
    path's length in the angles times the bounds' sum, per unit of the
    path."""
    carry, reach = np.array(model.POSES["carry"]), np.array(model.POSES["reach"])
    path = reach - carry
    out = None
    for w in np.linspace(0.0, 1.0, n):
        s = spread(tuple(carry + path * w), payloads, bounds, -hl.BRAKES, hl.DRIVE, hl.SPEED,
                   False, ds)
        if out is None:
            out = s
        else:
            out = dict(front=max(out["front"], s["front"]), side=max(out["side"], s["side"]),
                       side_least=min(out["side_least"], s["side_least"]),
                       rear_least=min(out["rear_least"], s["rear_least"]),
                       front_most=max(out["front_most"], s["front_most"]), second=s["second"])
    kds, k2ds = creep_curvature(ds)
    rem = (second_bound(hl.BRAKES, hl.SPEED, kds, k2ds) * float(np.abs(path).sum())
           * float(np.sum(bounds)) * 0.5 / (n - 1))
    out["front"] += rem
    out["side"] += rem
    out["grid"] = rem
    return out


def clearance_excess(pose, bounds):
    """How much farther forward than the test takes it, the pose's own
    front-most point or the worst corner of the bounds on the arm if that
    lies farther forward (governed.front_box), the front-most point can lie
    anywhere within the bounds [m]: the slew's cosine is largest at no slew,
    and the reach of the tool from the slew's axis is monotone in the lift
    and the tilt over the bounds where its gradient keeps its sign, so that
    a corner binds them; where the pose is the arm straight, no pose lies
    farther forward."""
    q = np.array(pose, dtype=float)
    b = np.asarray(bounds, dtype=float)

    def reach_of(lift, tilt):
        return model.BOOM["length"] * math.cos(lift) + model.STICK["length"] * math.cos(lift + tilt)
    g_lift = -(model.BOOM["length"] * math.sin(q[1]) + model.STICK["length"] * math.sin(q[1] + q[2]))
    g_tilt = -model.STICK["length"] * math.sin(q[1] + q[2])
    vary = ARM["tip"][1] * float(b[1] + b[2])          # the gradient's change over the bounds, at most
    if abs(g_lift) < 1e-12 and abs(g_tilt) < 1e-12:    # the arm straight: its reach is the largest
        best = reach_of(q[1], q[2])
    elif abs(g_lift) > vary and abs(g_tilt) > vary:
        best = reach_of(q[1] + math.copysign(b[1], g_lift), q[2] + math.copysign(b[2], g_tilt))
    else:
        raise ValueError("the reach's gradient may change its sign within the bounds")
    front_max = max(hl.CHASSIS_FRONT, model.SLEW[0] + best + hl.TOOL_HALF)
    taken = max([hl.front_point(tuple(q))] + [hl.front_point(tuple(q + b * np.array(sg)))
                                              for sg in itertools.product((-1, 1), repeat=3)])
    return max(0.0, front_max - taken)
