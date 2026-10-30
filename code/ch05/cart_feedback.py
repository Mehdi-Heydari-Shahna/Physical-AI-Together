#!/usr/bin/env python3
"""
A certified feedback control package for the cart of Chapter 3.

Physical AI, Together, Chapter 5, Sections 5.1 and 5.2.

The cart of Chapter 3 moves towards a wall over firm ground (braking of
2 m/s^2) and then a downhill ramp (braking of 0.5 m/s^2 from 20 m on). In this
chapter every world also contains a disturbance acceleration w(t), for
example from wind, rolling resistance and small slopes, bounded by
|w| <= 0.05 m/s^2. A segment that merely holds a command, as in Chapter 3, no
longer moves the cart along a known path, so the control method of Chapter 3
is unsound in this world. The control package implemented here consists of

    a tracking segment   feedback u = a - 8 (p - p_ref) - 4 (v - v_ref) around a
                         reference with constant acceleration a; its funnel is
                         the reference plus the tube {e : e'Pe <= phi^2} of the
                         Lyapunov certificate V = sqrt(e'Pe);
    a launch segment     an open-loop command from standstill, with a box of
                         reachable states as its funnel;
    a rest segment       brake and hold; its funnel is a stoppable set
                         {p >= p_lo, v >= 0, p + v^2/(2 * 0.45) <= p_hi};
    a generator          blends of the proposal with the head of the fallback,
                         each continued by braking references down to the
                         handover speed and then the rest segment;
    a test               exact rational checks of the certificate constants,
                         the input bound, the forward motion, the clearance
                         and the inclusions between funnels; the region of
                         each label holds by construction, because the
                         predictor issues it for the region of its segment.
    a rest feedback      the law of the rest segment, applied only after a
                         label alarm (no alarm occurs in these runs).

The governor, the capability predictors and the proposal method are those of
Chapter 3; there is no demand in this chapter. Every planned and certified
quantity is an exact rational number. The realized motion is integrated in
floating point (fourth-order Runge-Kutta, 200 steps per period) in a world
with one particular disturbance signal, and every executed segment is
checked against its funnel at every step (in floating point, with a
tolerance of 1e-9 for rounding).

Usage
    python3 cart_feedback.py            # all runs; writes results/
    python3 cart_feedback.py --check    # also checks every number quoted in
                                        # Sections 5.1 and 5.2

Outputs, in results/ next to this file
    trace-<configuration>.csv   one row per period: governor's view
    motion-<configuration>.csv  the realized motion every 0.01 s, with the
                                reference and the tube of the executed segment
    plan-k<period>.csv          the committed plan of a selected period: one row
                                per pair, then the rest segment
    tube.csv                    the boundaries of the tube and of the set
                                V <= 1/32 (Figure 5.1)
    errors-forecasting.csv      the tracking errors of the forecasting run
                                (Figure 5.1)
    endphase-*.csv              the end of the forecasting run in the phase
                                plane (Figure 5.3)
    summary.txt                 the numbers quoted in Sections 5.1 and 5.2

Requires Python 3.8 or later; no other packages.
"""

import argparse
import csv
import math
import os
from fractions import Fraction as Fr

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
T = Fr(1, 10)          # period of the governor [s]
U_MAX = Fr(2)          # command bound |u| <= 2 m/s^2
WALL = Fr(25)          # constraint p <= 25 m (the tool stays folded here)
RAMP = Fr(20)          # the downhill ramp starts here [m]
B_FIRM = Fr(2)         # braking available on firm ground [m/s^2]
B_RAMP = Fr(1, 2)      # braking available on the ramp [m/s^2]
B_GLOBAL = Fr(1, 2)    # global capability: braking of at least 0.5 everywhere
W_BAR = Fr(1, 20)      # bound on the disturbance, in every world [m/s^2]
EPS_MIN = Fr(1, 2)     # reserve floor [m]
GRID = Fr(1, 100)      # reserve levels: whole centimetres from 0.50 m
ALPHAS = (Fr(1), Fr(3, 4), Fr(1, 2), Fr(1, 4), Fr(0))   # ambition grid
HORIZON = 400          # forecast horizon [periods]
WORK_ZONE = Fr(24)     # the task: reach p >= 24 m and hold there
T_END = Fr(15)         # end of a run [s]

# ----------------------------------------------------------------------------
# The certificate of the tracking segments (Section 5.1)
# ----------------------------------------------------------------------------
KP, KV = Fr(8), Fr(4)                       # feedback gains [1/s^2], [1/s]
LAM = Fr(161, 100)                          # decay rate gamma of the certificate [1/s]
P11, P12, P22 = Fr(34, 5), Fr(2), Fr(1)     # Lyapunov matrix P
DELTA2 = P22 * W_BAR ** 2                   # delta^2 = G'PG w_bar^2
PHI2 = DELTA2 / LAM ** 2                    # funnel level: phi = delta/gamma
PHIBAR2 = Fr(1, 1024)                       # level on which the bounds hold: 1/32
E_P = Fr(19, 1000)     # bound on |p - p_ref| for V <= phibar [m]
E_V = Fr(49, 1000)     # bound on |v - v_ref| for V <= phibar [m/s]
E_U = Fr(1, 8)         # bound on the feedback part |u - a| for V <= phibar [m/s^2]
A_MAX = U_MAX - E_U    # largest reference acceleration: 15/8 m/s^2
V_H = Fr(1, 20)        # handover speed from tracking to the rest segment [m/s]
B_G = B_GLOBAL - W_BAR  # least net braking of the rest segment: 0.45 m/s^2

NEG_INF = None         # level of a plan that the test does not certify
N_SUB = 200            # integration steps per period
TOL = 1e-9             # tolerance of the floating-point funnel checks


def certificate_facts():
    """Exact checks of the certificate (Section 5.1). Returns a dict of
    facts; raises if a check fails."""
    # closed loop of the tracking error: e' = A_K e + G w,
    # A_K = [[0, 1], [-KP, -KV]], G = [0, 1]'. The certificate needs
    # M = A_K' P + P A_K + 2 LAM P to be negative semidefinite.
    m11 = -2 * KP * P12 + 2 * LAM * P11
    m12 = -KP * P22 + P11 - KV * P12 + 2 * LAM * P12
    m22 = 2 * (P12 - KV * P22) + 2 * LAM * P22
    det_m = m11 * m22 - m12 * m12
    assert m11 <= 0 and m22 <= 0 and det_m >= 0, "Lyapunov inequality fails"
    det = P11 * P22 - P12 * P12
    assert P11 > 0 and det > 0, "P is not positive definite"
    inv11, inv12, inv22 = P22 / det, -P12 / det, P11 / det
    kpk = KP * KP * inv11 + 2 * KP * KV * inv12 + KV * KV * inv22
    # the funnel lies strictly inside the set on which the bounds hold
    assert PHI2 < PHIBAR2, "funnel level must lie below the certified level"
    # largest |p - p_ref|, |v - v_ref| and |K e| on {e'Pe <= PHIBAR2}:
    # sqrt(PHIBAR2 inv11), sqrt(PHIBAR2 inv22), sqrt(PHIBAR2 K P^-1 K')
    assert E_P ** 2 >= PHIBAR2 * inv11, "position bound too small"
    assert E_V ** 2 >= PHIBAR2 * inv22, "velocity bound too small"
    assert E_U ** 2 >= PHIBAR2 * kpk, "input bound too small"
    assert V_H >= E_V and 0 < W_BAR < B_GLOBAL
    poles = (-KV / 2, math.sqrt(4 * KP - KV * KV) / 2)   # underdamped
    # largest decrease rate for this P: det(M(lam)) = 0, M(lam) = M0 + 2 lam P
    a0, a1, a2 = (-2 * KP * P12) * (2 * (P12 - KV * P22)) - (-KP * P22 + P11 - KV * P12) ** 2, \
        2 * P11 * (2 * (P12 - KV * P22)) + 2 * P22 * (-2 * KP * P12) \
        - 2 * (-KP * P22 + P11 - KV * P12) * 2 * P12, \
        4 * (P11 * P22 - P12 * P12)
    disc = float(a1 * a1 - 4 * a2 * a0)
    lam_max = (-float(a1) - math.sqrt(disc)) / (2 * float(a2))
    # the launch box at the end of a period, around its centre, in the tube
    box = [(sp * W_BAR * T * T / 2, sv * W_BAR * T) for sp in (-1, 1) for sv in (-1, 1)]
    box_v2 = max(P11 * dp * dp + 2 * P12 * dp * dv + P22 * dv * dv for dp, dv in box)
    assert box_v2 <= PHI2
    return {
        "lyapunov_matrix_M": f"{fmt(m11)},{fmt(m12)},{fmt(m22)}; det {fmt(det_m)}",
        "closed_loop_poles": f"{fmt(poles[0])} +- {poles[1]:.4f}i",
        "levels": (f"funnel {math.sqrt(PHI2):.6f} = 5/161, bounds 1/32; "
                   f"ratio {math.sqrt(PHI2 / PHIBAR2):.5f}"),
        "exact_extents": (f"{math.sqrt(PHIBAR2 * inv11):.5f} m, "
                          f"{math.sqrt(PHIBAR2 * inv22):.5f} m/s, "
                          f"{math.sqrt(PHIBAR2 * kpk):.5f} m/s^2"),
        "funnel_extents": (f"{math.sqrt(PHI2 * inv11):.5f} m, "
                           f"{math.sqrt(PHI2 * inv22):.5f} m/s, "
                           f"{math.sqrt(PHI2 * kpk):.5f} m/s^2"),
        "bounds_used": f"{fmt(E_P)} m, {fmt(E_V)} m/s, {fmt(E_U)} m/s^2",
        "reference_limits": (f"accelerate {fmt(A_MAX)}, brake {fmt(B_FIRM - E_U)} "
                             f"under label 2, {fmt(B_RAMP - E_U)} under label 0.5"),
        "rest_braking": fmt(B_G),
        "largest_decrease_rate_for_P": f"{lam_max:.4f}",
        "handover_offset": (f"{fmt(E_P + (V_H + E_V) ** 2 / (2 * B_G))} m; stop within "
                            f"{fmt((V_H + E_V) ** 2 / (2 * B_G))} m"),
        "launch_box_in_tube": (f"half-widths {fmt(W_BAR * T * T / 2)} m, "
                               f"{fmt(W_BAR * T)} m/s; V/phi {math.sqrt(box_v2 / PHI2):.3f}"),
        # time constant 1/|Re| = 2/KV of the poles -KV/2 +- i...; the damping
        # ratio squared KV^2/(4 KP) is exact
        "time_constant_and_damping": (f"{fmt(2 / KV)} s; damping ratio squared "
                                      f"{fmt(KV * KV / (4 * KP))}"),
        # the disturbance bound as the slope that gravity would turn into it
        "disturbance_as_slope": f"{100 * float(W_BAR) / 9.80665:.2f} %",
        # how much larger the set V <= 1/32 is than the tube, in linear size
        "bound_set_larger_by": f"{100 * (math.sqrt(PHIBAR2 / PHI2) - 1):.1f} %",
    }


def in_tube(dp, dv):
    """Whether the error (dp, dv) lies in the tube {e'Pe <= PHI2} (floats)."""
    return (float(P11) * dp * dp + 2 * float(P12) * dp * dv
            + float(P22) * dv * dv) <= float(PHI2) * (1 + TOL) + TOL


def tube_ratio(dp, dv):
    """sqrt(e'Pe) / phi: 1 on the boundary of the tube."""
    q = float(P11) * dp * dp + 2 * float(P12) * dp * dv + float(P22) * dv * dv
    return math.sqrt(max(q, 0.0) / float(PHI2))


# ----------------------------------------------------------------------------
# The world: braking and disturbance
# ----------------------------------------------------------------------------
def b_true(p):
    """Braking that the ground allows at position p [m/s^2]."""
    return B_FIRM if p < RAMP else B_RAMP


def b_true_f(p):
    """The same in floating point, for the realized motion."""
    return 2.0 if p < 20.0 else 0.5


def b_min(lo, hi):
    """Minimum of the true braking over the positions [lo, hi]."""
    return B_RAMP if hi >= RAMP else B_FIRM


W_PERIOD = 2.2         # period of the disturbance signal of the true world [s]


def disturbance(mode):
    """Disturbance signals w(t) of the true world, all within the bound.
    'sine' is the signal of the runs quoted in the text; 'zero' realizes no
    disturbance at all (the package is still certified for the bound);
    'plus' and 'minus' are constant; 'worst' chooses, at every instant, the
    sign that makes the certificate grow fastest in a tracking segment
    (w = w_bar sign(e'PB)) and the sign that opposes braking otherwise."""
    wb = float(W_BAR)
    if mode == "sine":
        return lambda t, seg, tau, p, v: wb * math.sin(2 * math.pi * t / W_PERIOD)
    if mode == "zero":
        return lambda t, seg, tau, p, v: 0.0
    if mode == "plus":
        return lambda t, seg, tau, p, v: wb
    if mode == "minus":
        return lambda t, seg, tau, p, v: -wb

    def worst(t, seg, tau, p, v):
        if seg.kind == "track":
            pr, vr = seg.ref(tau)
            return wb if float(P12) * (p - pr) + float(P22) * (v - vr) >= 0 else -wb
        return wb
    assert mode == "worst"
    return worst


# ----------------------------------------------------------------------------
# Sets: initial sets and the starts and ends of funnels
# ----------------------------------------------------------------------------
# ("ell", pc, vc)              the tube {(p, v) : e'Pe <= PHI2, e = (p-pc, v-vc)}
# ("box", plo, phi, vlo, vhi)  positions in [plo, phi], velocities in [vlo, vhi]
# ("rest", plo, phi)           the stoppable set {p >= plo, v >= 0,
#                              p + v^2/(2 B_G) <= phi}; at standstill if plo = phi


def contains(outer, inner):
    """Exact, sound test of inner <= outer for the sets that plans chain."""
    ko, ki = outer[0], inner[0]
    if ko == "ell":
        pc, vc = outer[1], outer[2]
        if ki == "ell":
            return (inner[1], inner[2]) == (pc, vc)
        if ki == "box":
            corners = [(p, v) for p in inner[1:3] for v in inner[3:5]]
        else:   # a standstill point
            if inner[1] != inner[2]:
                return False
            corners = [(inner[1], Fr(0))]
        return all(P11 * (p - pc) ** 2 + 2 * P12 * (p - pc) * (v - vc)
                   + P22 * (v - vc) ** 2 <= PHI2 for p, v in corners)
    if ko == "rest":
        plo, phi = outer[1], outer[2]
        if ki == "ell":    # via the bounding box of the tube
            pc, vc = inner[1], inner[2]
            return (vc - E_V >= 0 and pc - E_P >= plo
                    and pc + E_P + (vc + E_V) ** 2 / (2 * B_G) <= phi)
        if ki == "box":
            return (inner[3] >= 0 and inner[1] >= plo
                    and inner[2] + inner[4] ** 2 / (2 * B_G) <= phi)
        return inner[1] >= plo and inner[2] <= phi
    return False


def inside(s, p, v):
    """Whether the realized state (p, v) (floats) lies in the set s."""
    if s[0] == "ell":
        return in_tube(p - float(s[1]), v - float(s[2]))
    if s[0] == "box":
        return (float(s[1]) - TOL <= p <= float(s[2]) + TOL
                and float(s[3]) - TOL <= v <= float(s[4]) + TOL)
    plo, phi = float(s[1]), float(s[2])
    return (v >= -TOL and p >= plo - TOL
            and p + v * v / (2 * float(B_G)) <= phi + TOL)


# ----------------------------------------------------------------------------
# Segments
# ----------------------------------------------------------------------------
class Track:
    """Tracking segment: the reference starts at (p0, v0) and accelerates at a
    for one period; the feedback law is u = a - KP (p - p_ref) - KV (v - v_ref);
    the funnel is the reference plus the tube."""

    kind = "track"

    def __init__(self, p0, v0, a):
        self.p0, self.v0, self.a = p0, v0, a
        self.p1 = p0 + v0 * T + a * T * T / 2
        self.v1 = v0 + a * T
        # positions covered by the funnel; the reference moves forward when
        # the segment is certified, so its extremes are at the two ends
        self.lo = min(p0, self.p1) - E_P
        self.hi = max(p0, self.p1) + E_P

    def certified(self, beta):
        """Condition (i) under the label 'braking of at least beta on the
        region': the input u = a - K e stays in [-min(beta, 2), 2] on the
        tube, so it is admissible and every consistent world realizes it
        exactly, and the velocity stays positive on the tube, so the cart
        never comes to a standstill."""
        return (self.a - E_U >= -min(beta, U_MAX) and self.a + E_U <= U_MAX
                and self.v0 - E_V >= 0 and self.v1 - E_V >= 0)

    def clearance(self):
        return WALL - self.hi

    def start(self):
        return ("ell", self.p0, self.v0)

    def end(self):
        return ("ell", self.p1, self.v1)

    def ref(self, tau):
        """Reference at time tau into the period (floats)."""
        a, p0, v0 = float(self.a), float(self.p0), float(self.v0)
        return p0 + v0 * tau + 0.5 * a * tau * tau, v0 + a * tau

    def funnel_at(self, tau):
        pr, vr = self.ref(tau)
        return ("ell", pr, vr)

    def command(self, tau, p, v):
        pr, vr = self.ref(tau)
        return float(self.a) - float(KP) * (p - pr) - float(KV) * (v - vr)


class Launch:
    """Launch from standstill at p0: the open-loop command a, with
    W_BAR < a <= U_MAX, so the net acceleration a + w is positive and the cart
    moves forward. The funnel is the box of reachable states."""

    kind = "launch"

    def __init__(self, p0, a):
        self.p0, self.a = p0, a
        self.v0 = Fr(0)
        self.p1, self.v1 = p0 + a * T * T / 2, a * T      # centre of the end box
        self.box = ("box", p0 + (a - W_BAR) * T * T / 2, p0 + (a + W_BAR) * T * T / 2,
                    (a - W_BAR) * T, (a + W_BAR) * T)
        self.lo, self.hi = p0, self.box[2]

    def certified(self, beta):
        return W_BAR < self.a <= U_MAX

    def clearance(self):
        return WALL - self.hi

    def start(self):
        return ("rest", self.p0, self.p0)

    def end(self):
        return self.box

    def ref(self, tau):
        a, p0 = float(self.a), float(self.p0)
        return p0 + 0.5 * a * tau * tau, a * tau

    def funnel_at(self, tau):
        a, p0, wb = float(self.a), float(self.p0), float(W_BAR)
        return ("box", p0 + 0.5 * (a - wb) * tau * tau, p0 + 0.5 * (a + wb) * tau * tau,
                (a - wb) * tau, (a + wb) * tau)

    def command(self, tau, p, v):
        return float(self.a)


class Rest:
    """Rest segment: brake with the command -2 and hold at standstill. Its
    funnel is the stoppable set {p >= p_lo, v >= 0, p + v^2/(2 B_G) <= p_hi},
    which is invariant in every world of the global capability (Section 5.1);
    with p_lo = p_hi it is a single standstill state."""

    kind = "rest"

    def __init__(self, p_lo, p_hi):
        self.p_lo, self.p_hi = p_lo, p_hi

    def clearance(self):
        return WALL - self.p_hi

    def start(self):
        return ("rest", self.p_lo, self.p_hi)

    def end(self):
        return self.start()

    def funnel_at(self, tau):
        return self.start()

    def command(self, tau, p, v):
        return -float(U_MAX)


def rest_containing(s):
    """The smallest rest segment of this package whose funnel contains s."""
    if s[0] == "ell":
        pc, vc = s[1], s[2]
        return Rest(pc - E_P, pc + E_P + (vc + E_V) ** 2 / (2 * B_G))
    if s[0] == "box":
        return Rest(s[1], s[2] + s[4] ** 2 / (2 * B_G))
    return Rest(s[1], s[2])


# ----------------------------------------------------------------------------
# Labelled plans and the certification test
# ----------------------------------------------------------------------------
def floor_level(x):
    """Largest level of the grid L = {0.50, 0.51, ..., 25.00} not above x."""
    if x is NEG_INF or x < EPS_MIN:
        return NEG_INF
    return min(WALL, EPS_MIN + ((x - EPS_MIN) // GRID) * GRID)


class Plan:
    """Labelled plan: pairs (segment, label) executed in periods j0, j0+1, ...,
    followed by the rest segment."""

    def __init__(self, j0, pairs, rest):
        self.j0, self.pairs, self.rest = j0, pairs, rest

    def shift(self):
        return Plan(self.j0 + 1, self.pairs[1:], self.rest)

    def head(self):
        return self.pairs[0][0] if self.pairs else self.rest

    def inclusions(self, init):
        """(P1) on init, (P3) and (P4), decided exactly."""
        segs = [s for s, _ in self.pairs] + [self.rest]
        if not contains(segs[0].start(), init):
            return False
        return all(contains(b.start(), a.end()) for a, b in zip(segs, segs[1:]))

    def profile(self, init):
        """Levels certified from pair n on, n = 0..N (entry N: the rest
        segment alone); all -inf if an inclusion fails. The rest segments of
        this package are invariant and certified under the global capability
        at every level up to their clearance."""
        n = len(self.pairs)
        if not self.inclusions(init):
            return [NEG_INF] * (n + 1)
        out = [NEG_INF] * (n + 1)
        running = self.rest.clearance()
        out[n] = floor_level(running)
        ok = True
        for i in range(n - 1, -1, -1):
            seg, beta = self.pairs[i]
            ok = ok and seg.certified(beta)
            running = min(running, seg.clearance())
            out[i] = floor_level(running) if ok else NEG_INF
        return out

    def reserve(self):
        if not all(s.certified(b) for s, b in self.pairs):
            return None
        return min([s.clearance() for s, _ in self.pairs] + [self.rest.clearance()])


# ----------------------------------------------------------------------------
# Capability predictors (as in Chapter 3)
# ----------------------------------------------------------------------------
class Predictor:
    def __init__(self, kind):
        assert kind in ("fixed", "forecasting")
        self.kind = kind

    def label(self, k, j, seg):
        """Label issued during period k for segment seg in period j."""
        if j > k + 1 + HORIZON or self.kind == "fixed":
            return B_GLOBAL
        return b_min(seg.lo, seg.hi)          # minimum of a correct map


# ----------------------------------------------------------------------------
# The candidate generator
# ----------------------------------------------------------------------------
def recovery(pred, k, j, p, v):
    """Recovery continuation from the reference state (p, v) at the start of
    period j: the strongest certified braking references down to the
    handover speed, then the rest segment that contains the end of the tube."""
    pairs = []
    while v > V_H:
        for b_ref in (B_FIRM - E_U, B_RAMP - E_U):
            seg = Track(p, v, -min(b_ref, (v - V_H) / T))
            beta = pred.label(k, j, seg)
            if seg.certified(beta):
                break
        else:
            raise AssertionError("braking at the global capability is certified")
        pairs.append((seg, beta))
        p, v, j = seg.p1, seg.v1, j + 1
    if pairs:
        return pairs, rest_containing(pairs[-1][0].end())
    return pairs, None


def proposed_reference(q, vc):
    """Physical meaning of the proposal for a tracking head: 'accelerate' is
    the largest certified reference acceleration; 'hold' is the firm-ground
    braking reference down to the handover speed, and the rest segment once
    the reference is at the handover speed (None)."""
    if q == "accelerate":
        return A_MAX
    if vc > V_H:
        return -min(B_FIRM - E_U, (vc - V_H) / T)
    return None


def generate(pred, k, init, q, fallback):
    """Candidates of period k for the initial set init."""
    cands = []
    if init[0] == "rest":
        stay = Plan(k + 1, [], rest_containing(init))
        if q == "accelerate" and init[1] == init[2]:     # standstill point
            p0 = init[1]
            for a in ALPHAS[:-1]:
                head = Launch(p0, a * U_MAX)
                pairs = [(head, pred.label(k, k + 1, head))]
                tail, rest = recovery(pred, k, k + 2, head.p1, head.v1)
                if not tail:
                    rest = rest_containing(head.end())
                cands.append((Plan(k + 1, pairs + tail, rest), a))
            cands.append((stay, Fr(0)))
        else:
            cands.append((stay, Fr(1) if q == "hold" else Fr(0)))
        return cands
    pc, vc = (init[1], init[2]) if init[0] == "ell" else (
        (init[1] + init[2]) / 2, (init[3] + init[4]) / 2)
    a_q = proposed_reference(q, vc)
    fb_head = fallback.head()
    a_fb = fb_head.a if fb_head.kind == "track" else Fr(0)
    if a_q is None:                                     # hold at handover speed
        cands.append((Plan(k + 1, [], rest_containing(init)), Fr(1)))
    else:
        for alpha in ALPHAS:
            head = Track(pc, vc, alpha * a_q + (1 - alpha) * a_fb)
            pairs = [(head, pred.label(k, k + 1, head))]
            tail, rest = recovery(pred, k, k + 2, head.p1, head.v1)
            if rest is None:
                rest = rest_containing(head.end())
            cands.append((Plan(k + 1, pairs + tail, rest), alpha))
    tail, rest = recovery(pred, k, k + 1, pc, vc)
    if rest is None:
        rest = rest_containing(init)
    cands.append((Plan(k + 1, tail, rest), Fr(0)))
    return cands


# ----------------------------------------------------------------------------
# The proposal method and the governor (as in Chapter 3)
# ----------------------------------------------------------------------------
class ProposalMethod:
    """'accelerate' until the measured position first reaches the work zone,
    'hold' from then on."""

    def __init__(self):
        self.arrived = False

    def __call__(self, p):
        if p >= float(WORK_ZONE):
            self.arrived = True
        return "hold" if self.arrived else "accelerate"


def select(cands, profiles, fb_profile):
    """Admitted plans: level at least the floor. Without a demand, the most
    ambitious admitted plan; ties go to fresh candidates, then to the
    smallest index. Index 0 is the fallback."""
    best, best_key = None, None
    entries = [(0, fb_profile, Fr(0))]
    entries += [(i + 1, profiles[i], cands[i][1]) for i in range(len(cands))]
    for idx, prof, a in entries:
        if prof[0] is NEG_INF or prof[0] < EPS_MIN:
            continue
        key = (a, 1 if idx >= 1 else 0, -idx)
        if best_key is None or key > best_key:
            best, best_key = idx, key
    return best


# ----------------------------------------------------------------------------
# Realized motion in the true world
# ----------------------------------------------------------------------------
def accel(u, p, w):
    """Realized acceleration in the true world while the cart moves forward:
    accelerations towards the wall as commanded, towards the start limited
    by the braking of the ground; plus the disturbance w."""
    drive = u if u >= 0 else -min(-u, b_true_f(p))
    return drive + w


def simulate(seg, p, v, t0, wfun, record=None):
    """Integrates one period under the executed segment from the realized
    state (p, v) at time t0. Returns the end state and the largest values of
    the checked quantities. A forward motion braked to standstill stays at
    standstill until the period ends; the brakes of the rest segment hold the
    cart at standstill."""
    dt = float(T) / N_SUB
    stats = dict(tube=0.0, ufb=0.0, brake_margin=math.inf, out=0, pmax=p)
    held = v <= 0.0 and seg.kind == "rest"
    for i in range(N_SUB + 1):
        tau = i * dt
        u = seg.command(tau, p, v)
        if seg.kind == "track":
            pr, vr = seg.ref(tau)
            stats["tube"] = max(stats["tube"], tube_ratio(p - pr, v - vr))
            stats["ufb"] = max(stats["ufb"], abs(u - float(seg.a)))
        if not inside(seg.funnel_at(tau), p, v):
            stats["out"] += 1
        if seg.kind != "rest":
            # drive realized exactly: u >= -b(p) on the funnel
            stats["brake_margin"] = min(stats["brake_margin"], u + b_true_f(p))
        stats["pmax"] = max(stats["pmax"], p)
        if record is not None and i % 20 == 0 and i < N_SUB:
            record.append((t0 + tau, tau, p, v, u, seg))
        if i == N_SUB:
            break
        if held:
            continue

        def f(s, x, y):
            return y, accel(seg.command(s, x, y), x, wfun(t0 + s, seg, s, x, y))
        k1 = f(tau, p, v)
        k2 = f(tau + dt / 2, p + dt / 2 * k1[0], v + dt / 2 * k1[1])
        k3 = f(tau + dt / 2, p + dt / 2 * k2[0], v + dt / 2 * k2[1])
        k4 = f(tau + dt, p + dt * k3[0], v + dt * k3[1])
        p_new = p + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        v_new = v + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        if v > 0 and v_new <= 0:           # braked to standstill: held
            a_now = accel(seg.command(tau, p, v), p, wfun(t0 + tau, seg, tau, p, v))
            t_stop = min(dt, v / -a_now) if a_now < 0 else dt
            p, v, held = p + v * t_stop / 2, 0.0, True
        else:
            p, v = p_new, v_new
    return p, v, stats


# ----------------------------------------------------------------------------
# One run of the closed loop
# ----------------------------------------------------------------------------
def initial_set(seg_ex, p, v):
    """I_{k+1}, chosen during period k: the end of the executed funnel (the
    default of Chapter 4). A control method that propagates the measured
    state (p, v) at t_k replaces this function (Section 5.3)."""
    return seg_ex.end()


def run(kind, mode="sine", t_end=T_END, keep=()):
    pred = Predictor(kind)
    wfun = disturbance(mode)
    proposal = ProposalMethod()
    k_end = int(t_end / T)
    p, v = 0.0, 0.0                              # realized state (floats)
    plan = Plan(0, [], Rest(Fr(0), Fr(0)))       # initial plan: rest at p = 0
    init0 = ("rest", Fr(0), Fr(0))
    prof = plan.profile(init0)
    rows, kept, motion = [], {}, []
    for k in range(k_end):
        seg_ex = plan.head()
        beta_ex = plan.pairs[0][1] if plan.pairs else B_GLOBAL
        init = initial_set(seg_ex, p, v)         # I_{k+1}: by default the end of
                                                 # the executed funnel
        q = proposal(p)                          # measured state at t_k (exact)
        fallback = plan.shift()
        fb_prof = prof[1:] if len(prof) > 1 else prof
        cands = generate(pred, k, init, q, fallback)
        profiles = [c.profile(init) for c, _ in cands]
        sel = select(cands, profiles, fb_prof)
        if sel == 0:
            new_plan, new_prof, alpha = fallback, fb_prof, Fr(0)
        else:
            new_plan, new_prof = cands[sel - 1][0], profiles[sel - 1]
            alpha = cands[sel - 1][1]
        if k in keep:
            kept[k] = dict(init=init, q=q, cands=cands, profiles=profiles,
                           fb_prof=fb_prof, sel=sel, plan=new_plan, prof=new_prof)
        # realized motion during period k, measurement and label alarm
        p1, v1, st = simulate(seg_ex, p, v, float(k * T), wfun, motion)
        alarm = not inside(init, p1, v1)
        rows.append(dict(k=k, t=k * T, p=p, v=v, kind=seg_ex.kind,
                         a=getattr(seg_ex, "a", None), beta=beta_ex,
                         level=prof[0], q=q, alpha=alpha, sel=sel,
                         eps=new_prof[0], reserve=new_plan.reserve(),
                         plan=new_plan, alarm=int(alarm), **st))
        if alarm:
            break
        p, v = p1, v1
        plan, prof = new_plan, new_prof
    rows.append(dict(k=k_end, t=k_end * T, p=p, v=v, final=True))
    return rows, kept, motion


def ch3_package_first_period():
    """The control method of Chapter 3 in this world: its first moving
    segment holds the command 2 m/s^2 in period 1 from standstill, and its
    funnel is the single state (0.01 m, 0.2 m/s) at 0.2 s. Returns the
    realized state at 0.2 s."""
    seg = Launch(Fr(0), U_MAX)
    p1, v1, _ = simulate(seg, 0.0, 0.0, float(T), disturbance("sine"))
    return p1, v1


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def fmt(x, nd=6):
    if x is None:
        return "-inf"
    if isinstance(x, Fr):
        if x.denominator == 1:
            return str(x.numerator)
        s = f"{float(x):.{nd}f}".rstrip("0").rstrip(".")
        return s
    if isinstance(x, float):
        return f"{x:.{nd}f}".rstrip("0").rstrip(".")
    return str(x)


def r(x, nd):
    """Rounded decimal string with nd decimals (half up on exact values)."""
    x = Fr(x) if not isinstance(x, float) else Fr(x).limit_denominator(10**12)
    q = Fr(1, 10 ** nd)
    n = (x / q + Fr(1, 2)) // 1
    s = f"{float(n * q):.{nd}f}"
    return s


def facts_of_run(name, rows, motion=None):
    """The numbers that Section 5.2 quotes about one run."""
    body = [x for x in rows if not x.get("final")]
    f = {}
    arr = next((x for x in body if x["p"] >= float(WORK_ZONE)), None)
    f["arrival_s"] = r(arr["t"], 1) if arr else "none"
    k_rest = None                                  # at rest for good from here
    for x in reversed(body):
        if x["v"] == 0.0 and x["p"] > 0:
            k_rest = x["k"]
        else:
            break
    f["rest"] = (f"{body[-1]['p']:.3f} m, at rest from {r(k_rest * T, 1)} s"
                 if k_rest is not None else "not at rest")
    f["alarms"] = str(sum(x["alarm"] for x in body))
    f["max_position"] = f"{max(x['pmax'] for x in body):.3f} m"
    f["largest_tube_ratio"] = f"{max(x['tube'] for x in body):.3f}"
    f["largest_feedback_input"] = f"{max(x['ufb'] for x in body):.4f} m/s^2"
    f["steps_outside_funnel"] = str(sum(x["out"] for x in body))
    f["least_braking_margin"] = (
        f"{min(x['brake_margin'] for x in body):.3f} m/s^2")
    upto = [x for x in body if arr is None or x["k"] < arr["k"]]
    n1 = sum(1 for x in upto if x["alpha"] == 1)
    n0 = sum(1 for x in upto if x["alpha"] == 0)
    f["ambition_1_and_0_before_arrival"] = f"{n1},{n0},{len(upto)}"
    load = sum(1 - x["alpha"] for x in upto) / len(upto)
    f["load"] = f"{load} = {r(load, 2)}"
    hold = next((x for x in body if x["q"] == "hold" and x["alpha"] == 1), None)
    f["hold_accepted_from_s"] = r(hold["t"] + T, 1) if hold else "never"
    kinds = {}
    for x in body:
        kinds[x["kind"]] = kinds.get(x["kind"], 0) + 1
    f["executed_segments"] = ",".join(f"{kk} {kinds[kk]}" for kk in sorted(kinds))
    f["vmax"] = f"{max(x['v'] for x in body):.2f} m/s"
    first_brake = next((x for x in body if x["kind"] == "track" and x["a"] < 0), None)
    f["first_braking_reference"] = (
        f"{r(first_brake['t'], 1)} s at {first_brake['p']:.2f} m" if first_brake else "none")
    if motion:
        i_r = next(i for i, m in enumerate(motion) if m[2] >= float(RAMP))
        a, b = motion[i_r - 1], motion[i_r]
        lam = (float(RAMP) - a[2]) / (b[2] - a[2])
        f["ramp_entry"] = (f"{a[3] + lam * (b[3] - a[3]):.2f} m/s at "
                           f"{a[0] + lam * (b[0] - a[0]):.2f} s")
    f["final_plan_rest_set"] = (f"{fmt(body[-1]['plan'].rest.p_lo)} to "
                                f"{fmt(body[-1]['plan'].rest.p_hi)} m, level "
                                f"{fmt(body[-1]['eps'])}")
    return f


def robustness(kinds=("fixed", "forecasting"), modes=("plus", "minus", "worst")):
    """Runs under other disturbance signals: alarms, steps outside the tube
    and the largest tube ratio over all of them."""
    alarms, out, ratio, n = 0, 0, 0.0, 0
    worst_ratio = 0.0
    for kind in kinds:
        for mode in modes:
            rows, _, _ = run(kind, mode=mode)
            body = [x for x in rows if not x.get("final")]
            alarms += sum(x["alarm"] for x in body)
            out += sum(x["out"] for x in body)
            rr = max(x["tube"] for x in body)
            ratio = max(ratio, rr)
            if mode == "worst":
                worst_ratio = max(worst_ratio, rr)
            n += 1
    return {"runs": str(n), "alarms": str(alarms), "steps_outside_funnel": str(out),
            "largest_tube_ratio": f"{ratio:.3f}",
            "largest_tube_ratio_worst_signal": f"{worst_ratio:.3f}"}


def comparison_with_chapter3(here, facts):
    """Differences from the runs of Chapter 3, read from its summary."""
    path = os.path.join(here, "..", "ch03", "results", "summary.txt")
    ch3, group = {}, None
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith("["):
                group = line.strip("[]")
            elif " = " in line:
                k, v = line.strip().split(" = ", 1)
                ch3[(group, k)] = v
    out = {}
    for kind in ("forecasting", "fixed"):
        a3 = Fr(ch3[(kind, "arrival_s")])
        a5 = Fr(facts[kind]["arrival_s"])
        r3 = Fr(ch3[(kind, "rest")].split(" m")[0])
        r5 = Fr(facts[kind]["rest"].split(" m")[0])
        out[f"{kind}_later_and_short"] = (f"{fmt(a5 - a3)} s later, "
                                          f"{fmt((r3 - r5) * 1000)} mm further "
                                          f"from the wall")
    out["forecast_gain"] = (
        f"{fmt(Fr(facts['fixed']['arrival_s']) - Fr(facts['forecasting']['arrival_s']))} s, "
        f"against {fmt(Fr(ch3[('fixed', 'arrival_s')]) - Fr(ch3[('forecasting', 'arrival_s')]))} s")
    return out


def write_trace(path, rows):
    cols = ["k", "t", "p", "v", "kind", "a", "beta", "level", "q", "alpha", "sel",
            "eps", "reserve", "rest_p_hi", "alarm", "tube", "ufb"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for x in rows:
            if x.get("final"):
                continue
            w.writerow([x["k"], fmt(x["t"]), f"{x['p']:.6f}", f"{x['v']:.6f}",
                        x["kind"], fmt(x["a"]) if x["a"] is not None else "",
                        fmt(x["beta"]), fmt(x["level"]), x["q"], fmt(x["alpha"]),
                        x["sel"], fmt(x["eps"]), fmt(x["reserve"]),
                        fmt(x["plan"].rest.p_hi), x["alarm"],
                        f"{x['tube']:.4f}", f"{x['ufb']:.5f}"])


def write_motion(path, motion):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "p", "v", "u", "segment", "p_ref", "v_ref"])
        for t, tau, p, v, u, seg in motion:
            if seg.kind == "rest":
                pr, vr = "", ""
            else:
                pr, vr = (f"{c:.6f}" for c in seg.ref(tau))
            w.writerow([f"{t:.4f}", f"{p:.6f}", f"{v:.6f}", f"{u:.5f}", seg.kind, pr, vr])


def write_plan(path, plan, prof):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair", "period", "kind", "p0", "v0", "a", "label", "p1", "v1",
                    "clearance", "level_from_pair"])
        for n, (seg, beta) in enumerate(plan.pairs):
            w.writerow([n, plan.j0 + n, seg.kind, fmt(seg.p0), fmt(seg.v0),
                        fmt(seg.a), fmt(beta), fmt(seg.p1), fmt(seg.v1),
                        fmt(seg.clearance()), fmt(prof[n])])
        w.writerow(["rest", plan.j0 + len(plan.pairs), "rest", fmt(plan.rest.p_lo),
                    "", "", "0.5", fmt(plan.rest.p_hi), "", fmt(plan.rest.clearance()),
                    fmt(prof[-1])])


def write_tube(path):
    """Boundaries of the funnel cross-section {e'Pe <= PHI2} and of the larger
    set {e'Pe <= PHIBAR2} on which the bounds hold (Figure 5.1), in mm and
    mm/s."""
    l11 = math.sqrt(float(P11))
    l21 = float(P12) / l11
    l22 = math.sqrt(float(P22) - l21 * l21)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dp_funnel", "dv_funnel", "dp_bound", "dv_bound"])
        for i in range(181):
            th = 2 * math.pi * i / 180
            c, s_ = math.cos(th), math.sin(th)
            row = []
            for lev in (PHI2, PHIBAR2):
                phi = math.sqrt(float(lev))
                ev = phi * s_ / l22                 # solve L' e = phi (c, s)
                ep = (phi * c - l21 * ev) / l11
                row += [f"{1000 * ep:.4f}", f"{1000 * ev:.4f}"]
            w.writerow(row)


def plan_facts(plan, prof, init):
    """Composition of a committed plan, as quoted in Section 5.2."""
    f = {}
    segs = [s for s, _ in plan.pairs]
    labs = [b for _, b in plan.pairs]
    f["pairs"] = str(len(segs))
    f["head"] = (f"{segs[0].kind} {fmt(segs[0].a)} from {fmt(segs[0].p0)} m, "
                 f"{fmt(segs[0].v0)} m/s, label {fmt(labs[0])}") if segs else "rest"
    groups = []
    for s, b in zip(segs[1:], labs[1:]):
        key = (fmt(s.a), fmt(b))
        if groups and groups[-1][0] == key:
            groups[-1][1] += 1
        else:
            groups.append([key, 1])
    f["continuation"] = "; ".join(f"{n} at {a} label {b}" for (a, b), n in groups)
    last = segs[-1] if segs else None
    if last is not None:
        f["handover"] = f"{fmt(last.p1)} m, {fmt(last.v1)} m/s, period {plan.j0 + len(segs) - 1}"
    f["rest_set"] = f"{fmt(plan.rest.p_lo)} to {fmt(plan.rest.p_hi)} m"
    f["reserve"] = fmt(plan.reserve())
    f["level_above_floor"] = fmt(prof[0] - EPS_MIN)
    f["profile"] = ",".join(sorted({fmt(e) for e in prof}))
    f["initial_set"] = ",".join(fmt(c) for c in init[1:])
    sw = next((n for n, (_, b) in enumerate(plan.pairs) if b != labs[0]), None)
    if sw is not None:
        s0 = segs[sw]
        f["label_switch"] = (f"pair {sw}, period {plan.j0 + sw}, from {fmt(s0.p0)} m "
                             f"at {fmt(s0.v0)} m/s")
    return f


def candidate_facts(rec):
    """Levels and ambitions of the candidates of a kept period."""
    lv = ",".join(fmt(p[0]) for p in rec["profiles"])
    am = ",".join(fmt(a) for _, a in rec["cands"])
    return {"candidate_ambitions": am, "candidate_levels": lv,
            "fallback_level": fmt(rec["fb_prof"][0]), "selected": str(rec["sel"])}


EXPECTED = {   # every number quoted in Sections 5.1 and 5.2, as in the text
    "certificate": {
        "lyapunov_matrix_M": "-10.104,-2.76,-0.78; det 0.26352",
        "closed_loop_poles": "-2 +- 2.0000i",
        "levels": "funnel 0.031056 = 5/161, bounds 1/32; ratio 0.99379",
        "exact_extents": "0.01868 m, 0.04870 m/s, 0.12500 m/s^2",
        "bounds_used": "0.019 m, 0.049 m/s, 0.125 m/s^2",
        "reference_limits": "accelerate 1.875, brake 1.875 under label 2, "
                            "0.375 under label 0.5",
        "rest_braking": "0.45",
        "largest_decrease_rate_for_P": "1.6414",
        "handover_offset": "0.02989 m; stop within 0.01089 m",
        "launch_box_in_tube": "half-widths 0.00025 m, 0.005 m/s; V/phi 0.178",
        "time_constant_and_damping": "0.5 s; damping ratio squared 0.5",
        "disturbance_as_slope": "0.51 %",
        "bound_set_larger_by": "0.6 %",
    },
    "chapter3_package": {
        "realized_at_0.2s": "0.010093 m, 0.202070 m/s",
        "difference": "0.093 mm, 2.07 mm/s",
    },
    "forecasting": {
        "arrival_s": "9.2", "rest": "24.468 m, at rest from 10.7 s",
        "alarms": "0", "largest_tube_ratio": "0.516",
        "largest_feedback_input": "0.0613 m/s^2", "steps_outside_funnel": "0",
        "least_braking_margin": "0.064 m/s^2", "load": "117/184 = 0.64",
        "vmax": "6.20 m/s", "ramp_entry": "1.77 m/s at 5.83 s",
        "final_plan_rest_set": "24.449125 to 24.498015 m, level 0.5",
    },
    "fixed": {
        "arrival_s": "11.2", "rest": "24.467 m, at rest from 12.7 s",
        "alarms": "0", "largest_tube_ratio": "0.516",
        "largest_feedback_input": "0.0613 m/s^2", "steps_outside_funnel": "0",
        "least_braking_margin": "0.064 m/s^2", "load": "13/16 = 0.81",
        "vmax": "3.89 m/s", "ramp_entry": "1.81 m/s at 7.75 s",
    },
    "plan_k33": {
        "pairs": "71",
        "head": "track -0.9375 from 10.25 m, 6.2 m/s, label 2",
        "continuation": "23 at -1.875 label 2; 46 at -0.375 label 0.5; "
                        "1 at -0.1875 label 0.5",
        "handover": "24.24 m, 0.05 m/s, period 104",
        "rest_set": "24.221 to 24.26989 m", "reserve": "0.73011",
        "level_above_floor": "0.23",
        "profile": "0.73", "initial_set": "10.25,6.2",
        "label_switch": "pair 24, period 58, from 19.950312 m at 1.79375 m/s",
        "candidate_ambitions": "1,0.75,0.5,0.25,0,0",
        "candidate_levels": "-inf,-inf,-inf,0.73,2.03,2.03",
        "selected": "4",
    },
    "robustness": {
        "runs": "6", "alarms": "0", "steps_outside_funnel": "0",
        "largest_tube_ratio": "0.570",
    },
    "undisturbed": {
        "forecasting": "arrival 9.2 s; at rest at 24.471 m",
        "fixed": "arrival 11.2 s; at rest at 24.467 m",
    },
    "comparison": {
        "forecasting_later_and_short": "0.6 s later, 32 mm further from the wall",
        "fixed_later_and_short": "1.3 s later, 28 mm further from the wall",
        "forecast_gain": "2 s, against 1.3 s",
    },
}


def write_errors(path, motion):
    """Tracking errors of the executed tracking segments, every 0.01 s
    (Figure 5.1)."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "dp_mm", "dv_mm_s"])
        for t, tau, p, v, u, seg in motion:
            if seg.kind == "track":
                pr, vr = seg.ref(tau)
                w.writerow([f"{t:.2f}", f"{1000 * (p - pr):.4f}", f"{1000 * (v - vr):.4f}"])


def write_endphase(out, motion, plan, t_from):
    """The end of the forecasting run in the phase plane (Figure 5.3), in
    three files: the realized motion and the reference from t_from on; the
    tube around the reference at the start of every fifth period and at the
    end of the last tracking segment; and the boundary of the stoppable set
    of the final rest segment."""
    l11 = math.sqrt(float(P11))
    l21 = float(P12) / l11
    l22 = math.sqrt(float(P22) - l21 * l21)
    phi = math.sqrt(float(PHI2))

    def ellipse(pc, vc):
        pts = []
        for i in range(73):
            th = 2 * math.pi * i / 72
            ev = phi * math.sin(th) / l22
            ep = (phi * math.cos(th) - l21 * ev) / l11
            pts.append((pc + ep, vc + ev))
        return pts

    tubes = {}
    with open(os.path.join(out, "endphase-motion.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "p", "v", "p_ref", "v_ref"])
        for t, tau, p, v, u, seg in motion:
            if t < t_from - 1e-9:
                continue
            if seg.kind == "rest":
                w.writerow([f"{t:.2f}", f"{p:.5f}", f"{v:.5f}", "nan", "nan"])
                continue
            pr, vr = seg.ref(tau)
            w.writerow([f"{t:.2f}", f"{p:.5f}", f"{v:.5f}", f"{pr:.5f}", f"{vr:.5f}"])
            k = round(t / float(T))
            if seg.kind == "track" and abs(tau) < 1e-12 and k % 5 == 0:
                tubes[f"k{k}"] = ellipse(pr, vr)
    tubes["end"] = ellipse(float(plan.rest.p_lo + E_P), float(V_H))
    names = list(tubes)
    with open(os.path.join(out, "endphase-tubes.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([c for n in names for c in (f"p_{n}", f"v_{n}")])
        for i in range(73):
            w.writerow([f"{x:.5f}" for n in names for x in tubes[n][i]])
    plo, phi_s = float(plan.rest.p_lo), float(plan.rest.p_hi)
    vtop = math.sqrt(2 * float(B_G) * (phi_s - plo))
    with open(os.path.join(out, "endphase-stoppable.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["p", "v"])
        for i in range(61):
            vv = vtop * i / 60
            w.writerow([f"{phi_s - vv * vv / (2 * float(B_G)):.5f}", f"{vv:.5f}"])
    return names


def main():
    ap = argparse.ArgumentParser(description="Chapter 5: a certified feedback package")
    ap.add_argument("--check", action="store_true",
                    help="check every number quoted in Sections 5.1 and 5.2")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "results")
    os.makedirs(out, exist_ok=True)
    facts = {"certificate": certificate_facts()}
    write_tube(os.path.join(out, "tube.csv"))
    for kind in ("fixed", "forecasting"):
        rows, kept, motion = run(kind, keep=(1, 33))
        write_trace(os.path.join(out, f"trace-{kind}.csv"), rows)
        write_motion(os.path.join(out, f"motion-{kind}.csv"), motion)
        facts[kind] = facts_of_run(kind, rows, motion)
        if kind == "forecasting":
            write_errors(os.path.join(out, "errors-forecasting.csv"), motion)
            body = [x for x in rows if not x.get("final")]
            write_endphase(out, motion, body[-1]["plan"], 8.5)
            for kk, rec in sorted(kept.items()):
                write_plan(os.path.join(out, f"plan-k{kk}.csv"), rec["plan"], rec["prof"])
                facts[f"plan_k{kk}"] = plan_facts(rec["plan"], rec["prof"], rec["init"])
                facts[f"plan_k{kk}"].update(candidate_facts(rec))
    facts["robustness"] = robustness()
    # the same package in a world whose disturbance happens to be zero: the
    # plans are certified for the bound, so they do not change
    facts["undisturbed"] = {}
    for kind in ("forecasting", "fixed"):
        rows, _, motion = run(kind, mode="zero")
        f = facts_of_run(kind, rows, motion)
        facts["undisturbed"][kind] = (f"arrival {f['arrival_s']} s; at rest at "
                                      f"{f['rest'].split(' m')[0]} m")
    facts["comparison"] = comparison_with_chapter3(here, facts)
    p1, v1 = ch3_package_first_period()
    facts["chapter3_package"] = {
        "realized_at_0.2s": f"{p1:.6f} m, {v1:.6f} m/s",
        "difference": f"{(p1 - 0.01) * 1000:.3f} mm, {(v1 - 0.2) * 1000:.2f} mm/s",
    }
    lines, failures = [], []
    for group, items in facts.items():
        lines.append(f"[{group}]")
        for key, val in items.items():
            lines.append(f"  {key} = {val}")
            want = EXPECTED.get(group, {}).get(key)
            if args.check and want is not None and want != val:
                failures.append(f"{group}.{key}: expected {want!r}, got {val!r}")
    if args.check:
        failures += [f"{g}.{k}: not computed" for g, items in EXPECTED.items()
                     for k in items if k not in facts.get(g, {})]
        n = sum(len(v) for v in EXPECTED.values())
        lines.append(f"[check] {n} quoted values: " +
                     ("all as quoted" if not failures else "FAILED: " + "; ".join(failures)))
    with open(os.path.join(out, "summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
