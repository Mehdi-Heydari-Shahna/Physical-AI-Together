#!/usr/bin/env python3
"""
Three established control methods as control packages for the cart.

Physical AI, Together, Chapter 5, Section 5.3.

The governor, the capability predictors, the proposal method, the plans,
the selection rule and the simulation of the true world are those of
cart_feedback.py, imported and used unchanged. Each package below replaces
only the parts that a control method supplies: its certificate constants,
its segments and their funnels, the set inclusions, the recovery
continuation, the generator and, for the robust MPC package, the
prediction from which the initial set of each plan is chosen. The conditions
are matched: the same world, disturbance signals, labels, predictors,
proposal method, governor, reserve floor and grid of ambitions in every
run.

    tube   the package of Sections 5.1 and 5.2, the baseline
    qp     a control Lyapunov function quadratic program: the Lyapunov
           function and decrease rate of the tube package; the feedback is
           the smallest input that the hard Lyapunov constraint allows
    ccm    a contraction metric: the feedback gains of the tube package, with
           the constant metric M = [[8, 2], [2, 1]], which contracts at the
           rate 2 /s, the real part of the closed-loop eigenvalues
    mpc    a robust MPC tube in the metric M: the tube scale starts from the
           measured state, propagated over the period of computation, and
           grows by s+ = theta s + (1 - theta) phi per period, with
           theta = exp(-2 T); the handover speed is that of ccm

Every certified quantity is an exact rational number. The bound on the
feedback of the qp package is proved by Sturm's theorem on a polynomial with
rational coefficients; exp(-2 T) and the transition matrix of the closed loop
enter only through rational bounds.

Usage
    python3 cart_methods.py            # all packages and runs
    python3 cart_methods.py --check    # also checks every number quoted in
                                       # Section 5.3

Outputs, in results/ next to this file
    methods-summary.txt         the numbers quoted in Section 5.3
    methods-tubes.csv           cross-sections of the tubes (Figure 5.5)
    methods-endphase.csv        the realized motion of each package from 5 s
                                on in the forecasting run

Run cart_feedback.py first only if you want its own outputs; this script
imports it and does not need them. Python 3.8 or later; no other packages.
"""
import argparse
import csv
import math
import os
import sys
from fractions import Fraction as Fr

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cart_feedback as cf   # noqa: E402

T, U_MAX, W_BAR, WALL = cf.T, cf.U_MAX, cf.W_BAR, cf.WALL
B_FIRM, B_RAMP, B_G = cf.B_FIRM, cf.B_RAMP, cf.B_G
KP, KV = cf.KP, cf.KV                      # the gains of tube, ccm and mpc
fmt, r = cf.fmt, cf.r

# The names of cart_feedback.py that a control method supplies.
SUPPLIED = ["LAM", "P11", "P12", "P22", "DELTA2", "PHI2", "PHIBAR2", "E_P", "E_V",
            "E_U", "A_MAX", "V_H", "Track", "contains", "inside", "rest_containing",
            "recovery", "proposed_reference", "generate", "initial_set"]
ORIGINAL = {name: getattr(cf, name) for name in SUPPLIED}


def install(changes):
    """Install a control package: restore the tube package, then apply the
    changes. The governor and everything else stay as they are."""
    for name in SUPPLIED:
        setattr(cf, name, ORIGINAL[name])
    for name, value in changes.items():
        assert name in SUPPLIED, name
        setattr(cf, name, value)


# ----------------------------------------------------------------------------
# Exact helpers
# ----------------------------------------------------------------------------
def sqrt_up(x, prec=10 ** 12):
    """A rational upper bound of sqrt(x) on the grid 1/prec."""
    n = math.isqrt(math.ceil(x * prec * prec))
    while Fr(n * n, prec * prec) < x:
        n += 1
    return Fr(n, prec)


def round_up(x, prec=10 ** 12):
    """x rounded up to the grid 1/prec."""
    return Fr(math.ceil(x * prec), prec)


def percent(x):
    """The ratio x as the text quotes it: in whole percent, or to one decimal
    from 99 % on. Computed from the unrounded value, never from a rounded
    one."""
    p = 100 * float(x)
    return f"{p:.1f}" if p >= 99 else f"{p:.0f}"


def poly_eval(c, x):
    """Polynomial with coefficients c[0] + c[1] x + ... at x."""
    out = Fr(0)
    for a in reversed(c):
        out = out * x + a
    return out


def poly_trim(c):
    c = list(c)
    while len(c) > 1 and c[-1] == 0:
        c.pop()
    return c


def poly_mul(a, b):
    out = [Fr(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return poly_trim(out)


def poly_sub(a, b):
    n = max(len(a), len(b))
    return poly_trim([(a[i] if i < len(a) else 0) - (b[i] if i < len(b) else 0)
                      for i in range(n)])


def poly_rem(a, b):
    a = poly_trim(a)
    while len(a) >= len(b) and any(a):
        f = a[-1] / b[-1]
        shift = len(a) - len(b)
        a = poly_trim([a[i] - (f * b[i - shift] if i >= shift else 0)
                       for i in range(len(a))][:-1] or [Fr(0)])
    return a


def sturm_roots(c, lo, hi):
    """Number of distinct real roots of the polynomial c in (lo, hi], for lo
    and hi that are not roots of c; lo and hi may be -inf or +inf (None)."""
    assert len(poly_trim(c)) > 1
    assert all(x is None or poly_eval(c, x) != 0 for x in (lo, hi))
    seq = [poly_trim(c), poly_trim([i * c[i] for i in range(1, len(c))])]
    while len(seq[-1]) > 1 or seq[-1][0] != 0:
        rem = poly_rem(seq[-2], seq[-1])
        if len(rem) == 1 and rem[0] == 0:
            break
        seq.append([-x for x in rem])

    def changes(x, sign_at_inf):
        vals = []
        for p in seq:
            if x is None:
                v = p[-1] * (1 if sign_at_inf > 0 or (len(p) - 1) % 2 == 0 else -1)
            else:
                v = poly_eval(p, x)
            if v != 0:
                vals.append(v > 0)
        return sum(1 for a, b in zip(vals, vals[1:]) if a != b)
    return changes(lo, -1) - changes(hi, +1)


# ----------------------------------------------------------------------------
# Package qp: the least input that the hard Lyapunov constraint allows
# ----------------------------------------------------------------------------
# With e = (p - p_ref, v - v_ref) and V^2 = e'Pe, the constraint
# d/dt V^2 <= -2 LAM V^2 + 2 V sqrt(G'PG) |w| holds for every disturbance if
# e'PB u_fb <= -N(e), N(e) = e'P A0 e + LAM e'Pe, A0 = [[0, 1], [0, 0]].
# The smallest |u_fb| that satisfies it is u_fb = -N/(e'PB) where N > 0 and
# 0 elsewhere. With P of the tube package, N = 10.948 dp^2 + 13.24 dp dv
# + 3.61 dv^2 and e'PB = 2 dp + dv.
QP_E_U = Fr(23, 200)            # the certified bound on |u_fb| for V <= 1/32


def qp_coefficients():
    P11, P12, P22, LAM = ORIGINAL["P11"], ORIGINAL["P12"], ORIGINAL["P22"], ORIGINAL["LAM"]
    # N(dp, dv) = n_pp dp^2 + n_pv dp dv + n_vv dv^2; e'PB = P12 dp + P22 dv
    return (LAM * P11, P11 + 2 * LAM * P12, P12 + LAM * P22, P12, P22)


def qp_budget_proof():
    """Proves |u_fb| <= QP_E_U on {V <= phibar} exactly. Along the direction
    e = (1, m), |u_fb| / V = N / (|e'PB| sqrt(e'Pe)); the bound holds iff
    g(m) = C (e'PB)^2 (e'Pe) - N^2 >= 0 wherever N > 0, with
    C = (QP_E_U / phibar)^2. N <= 0 exactly on [m1, m2], so it suffices that g
    is positive at rationals r1 and r2 with m1 <= r1 <= r2 <= m2 and has no
    root on (-inf, r1] and none on [r2, inf); the direction (0, 1) is checked
    separately. Returns the facts quoted in Section 5.3."""
    npp, npv, nvv, d1, d2 = qp_coefficients()
    P11, P12, P22 = ORIGINAL["P11"], ORIGINAL["P12"], ORIGINAL["P22"]
    phibar2 = ORIGINAL["PHIBAR2"]
    C = QP_E_U ** 2 / phibar2
    N = [npp, npv, nvv]                     # in m, for e = (1, m)
    D = [d1, d2]
    Q = [P11, 2 * P12, P22]
    g = poly_sub(poly_mul([C], poly_mul(poly_mul(D, D), Q)), poly_mul(N, N))
    r1, r2 = Fr(-12, 5), Fr(-63, 50)        # -2.4 and -1.26, inside [m1, m2]
    assert poly_eval(N, r1) <= 0 and poly_eval(N, r2) <= 0
    assert poly_eval(g, r1) > 0 and poly_eval(g, r2) > 0
    assert sturm_roots(g, None, r1) == 0 and sturm_roots(g, r2, None) == 0
    # direction (0, 1): N = nvv, e'PB = d2, e'Pe = P22
    assert nvv ** 2 <= C * d2 ** 2 * P22
    # the largest ratio, found on a fine grid of directions, for the record
    best = 0.0
    for i in range(1, 200000):
        th = math.pi * i / 200000
        dp, dv = math.cos(th), math.sin(th)
        n = float(npp) * dp * dp + float(npv) * dp * dv + float(nvv) * dv * dv
        d = float(d1) * dp + float(d2) * dv
        q = float(P11) * dp * dp + 2 * float(P12) * dp * dv + float(P22) * dv * dv
        if n > 0:
            best = max(best, n / (abs(d) * math.sqrt(q)))
    return {"budget": f"{fmt(QP_E_U)} m/s^2, proved; largest value "
                      f"{best * math.sqrt(float(phibar2)):.4f} m/s^2",
            "ratio_to_linear": f"{best / 4:.3f} = {percent(best / 4)} %",
            "polynomial_degree": str(len(g) - 1)}


class TrackQP(ORIGINAL["Track"]):
    """Tracking segment of the qp package: the same reference, tube and test
    as the tube package; the feedback is the least input that the Lyapunov
    constraint requires."""

    def command(self, tau, p, v):
        pr, vr = self.ref(tau)
        npp, npv, nvv, d1, d2 = QP_FLOATS
        dp, dv = p - pr, v - vr
        n = npp * dp * dp + npv * dp * dv + nvv * dv * dv
        u_fb = -n / (d1 * dp + d2 * dv) if n > 0 else 0.0
        return float(self.a) + u_fb


QP_FLOATS = tuple(float(x) for x in qp_coefficients())

QP = dict(E_U=QP_E_U, A_MAX=U_MAX - QP_E_U, Track=TrackQP)


# ----------------------------------------------------------------------------
# Package ccm: the same controller, the metric M
# ----------------------------------------------------------------------------
M11, M12, M22 = Fr(8), Fr(2), Fr(1)
GAMMA_M = Fr(2)
PHI_M = W_BAR * 1 / GAMMA_M               # delta / gamma; sqrt(G'MG) = 1
MARGIN = Fr(81, 80)
PHIBAR_M = PHI_M * MARGIN                 # 81/3200
CCM_E_P, CCM_E_V, CCM_E_U = Fr(13, 1000), Fr(36, 1000), 4 * PHIBAR_M
CCM_V_H = Fr(1, 25)
SQ2_UP = Fr(14143, 10000)                 # >= sqrt(2)


def metric_facts():
    """Exact checks of the metric M for the gains of Section 5.1."""
    a = [[Fr(0), Fr(1)], [-KP, -KV]]
    m = [[M11, M12], [M12, M22]]
    ma = [[sum(m[i][k] * a[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
    lyap = [[ma[i][j] + ma[j][i] + 2 * GAMMA_M * m[i][j] for j in range(2)]
            for i in range(2)]
    assert lyap == [[0, 0], [0, 0]], "A_K'M + MA_K + 2 gamma M = 0"
    det = M11 * M22 - M12 * M12
    assert M11 > 0 and det > 0
    inv11, inv12, inv22 = M22 / det, -M12 / det, M11 / det
    kmk = KP * KP * inv11 + 2 * KP * KV * inv12 + KV * KV * inv22
    assert kmk == 16
    # the gains have the form K = (rho/2) B'M of a contraction metric controller
    rho = 2 * KV / M22
    assert (KP, KV) == (rho / 2 * M12, rho / 2 * M22)
    assert PHI_M < PHIBAR_M
    assert CCM_E_P ** 2 >= PHIBAR_M ** 2 * inv11
    assert CCM_E_V ** 2 >= PHIBAR_M ** 2 * inv22
    assert CCM_E_U ** 2 >= PHIBAR_M ** 2 * kmk
    assert SQ2_UP ** 2 >= 2 and CCM_V_H >= CCM_E_V
    # the launch box at the end of the period lies in the tube
    box = [(sp * W_BAR * T * T / 2, sv * W_BAR * T) for sp in (-1, 1) for sv in (-1, 1)]
    box_v2 = max(M11 * dp * dp + 2 * M12 * dp * dv + M22 * dv * dv for dp, dv in box)
    assert box_v2 <= PHI_M ** 2
    # the tube of M lies in the tube of the package of Section 5.1: M - P is
    # positive semidefinite and singular, so the largest V_P on {V_M <= phi_M}
    # is phi_M itself (lambda_max(M^-1 P) = 1), and the factor is exact
    P11, P12, P22 = ORIGINAL["P11"], ORIGINAL["P12"], ORIGINAL["P22"]
    d11, d12, d22 = M11 - P11, M12 - P12, M22 - P22
    assert d11 >= 0 and d22 >= 0 and d11 * d22 - d12 * d12 == 0
    ratio_sq = PHI_M ** 2 / ORIGINAL["PHI2"]
    assert ratio_sq < 1 and ratio_sq == Fr(161, 200) ** 2
    ratio = Fr(161, 200)
    return {
        "lyapunov_equation": "A_K'M + MA_K + 4M = 0",
        "rho": fmt(rho),
        "levels": f"tube {fmt(PHI_M)}, bounds {fmt(PHIBAR_M)}",
        "exact_extents": (f"{math.sqrt(PHIBAR_M ** 2 * inv11):.5f} m, "
                          f"{math.sqrt(PHIBAR_M ** 2 * inv22):.5f} m/s, "
                          f"{fmt(4 * PHIBAR_M)} m/s^2"),
        "bounds_used": f"{fmt(CCM_E_P)} m, {fmt(CCM_E_V)} m/s, {fmt(CCM_E_U)} m/s^2",
        "launch_box_V_over_phi": f"{math.sqrt(box_v2 / PHI_M ** 2):.3f}",
        "inside_tube_of_5_1": fmt(ratio),
        "handover_offset": fmt(CCM_E_P + (CCM_V_H + CCM_E_V) ** 2 / (2 * B_G)),
    }


CCM = dict(LAM=GAMMA_M, P11=M11, P12=M12, P22=M22, DELTA2=W_BAR ** 2 * M22,
           PHI2=PHI_M ** 2, PHIBAR2=PHIBAR_M ** 2, E_P=CCM_E_P, E_V=CCM_E_V,
           E_U=CCM_E_U, A_MAX=U_MAX - CCM_E_U, V_H=CCM_V_H)


# ----------------------------------------------------------------------------
# Package mpc: a tube that starts from the measured state
# ----------------------------------------------------------------------------
X_DECAY = GAMMA_M * T                       # gamma T = 1/5


def exp_neg_lower(x):
    """Lower bound of exp(-x) for 0 < x < 1: alternating series to x^5."""
    return sum((-x) ** n / math.factorial(n) for n in range(6))


A_LO = exp_neg_lower(X_DECAY)               # <= exp(-1/5)


def scale_next(s):
    """Upper bound of the tube scale one period after the scale s: the bound
    phi(tau) = exp(-gamma tau) s + phi_M (1 - exp(-gamma tau)) of Lemma 5.1
    grows to phi_M when s < phi_M and never exceeds s otherwise."""
    if s >= PHI_M:
        return s
    return round_up(PHI_M - A_LO * (PHI_M - s))


def transition_bound(n_terms=25):
    """Rational approximation E of exp(A_K T) and a bound eps on the entries
    of the error, from the Taylor series and the remainder
    ||A_K T||^n / n! e^{||A_K T||} in the maximum row-sum norm (1.2)."""
    a = [[Fr(0), T], [-KP * T, -KV * T]]
    e = [[Fr(1), Fr(0)], [Fr(0), Fr(1)]]
    term = [[Fr(1), Fr(0)], [Fr(0), Fr(1)]]
    for n in range(1, n_terms):
        term = [[sum(term[i][k] * a[k][j] for k in range(2)) / n for j in range(2)]
                for i in range(2)]
        e = [[e[i][j] + term[i][j] for j in range(2)] for i in range(2)]
    norm = max(abs(a[0][0]) + abs(a[0][1]), abs(a[1][0]) + abs(a[1][1]))
    eps = norm ** n_terms / math.factorial(n_terms) * Fr(34, 10)    # e^1.2 < 3.4
    return e, eps


E_T, EPS_T = transition_bound()
C_M = Fr(361, 100)          # ||x||_M <= 3.61 max|x_i|: x'Mx <= 8 + 4 + 1 = 13
GRID_Z = 10 ** 9            # the predicted centre is rounded to this grid


class TrackS:
    """Tracking segment of the mpc package: the reference starts at (p0, v0)
    and accelerates at a; the feedback is that of Section 5.1. The funnel is
    the reference surrounded by {V_M <= phi(tau)} of Lemma 5.1 with phi_0 = s0
    for tau < T, and by {V_M <= s1} at tau = T, which contains the tube of
    the lemma there. Its bounds hold on {V_M <= phibar}, phibar = s1 81/80;
    Lemma 5.1 needs phibar > max(s0, phi(T)) = phi(T), and phi(T) <= s1.
    The bounds on the error are rounded up to millimetres, as in the ccm
    package. funnel_at() returns the tube of the lemma, also at tau = T, so
    the realized motion is checked against the smaller set."""

    kind = "track"

    def __init__(self, p0, v0, a, s0):
        self.p0, self.v0, self.a, self.s0 = p0, v0, a, s0
        self.p1 = p0 + v0 * T + a * T * T / 2
        self.v1 = v0 + a * T
        self.s1 = scale_next(s0)
        phibar = self.s1 * MARGIN
        self.e_p, self.e_v, self.e_u = mm_up(phibar / 2), mm_up(phibar * SQ2_UP), 4 * phibar
        self.lo = min(p0, self.p1) - self.e_p
        self.hi = max(p0, self.p1) + self.e_p

    def certified(self, beta):
        return (self.a - self.e_u >= -min(beta, U_MAX) and self.a + self.e_u <= U_MAX
                and self.v0 - self.e_v >= 0 and self.v1 - self.e_v >= 0)

    def clearance(self):
        return WALL - self.hi

    def start(self):
        return ("ell4", self.p0, self.v0, self.s0)

    def end(self):
        return ("ell4", self.p1, self.v1, self.s1)

    def ref(self, tau):
        a, p0, v0 = float(self.a), float(self.p0), float(self.v0)
        return p0 + v0 * tau + 0.5 * a * tau * tau, v0 + a * tau

    def funnel_at(self, tau):
        pr, vr = self.ref(tau)
        d = math.exp(-float(GAMMA_M) * tau)
        return ("ell4", pr, vr, d * float(self.s0) + float(PHI_M) * (1 - d))

    def command(self, tau, p, v):
        pr, vr = self.ref(tau)
        return float(self.a) - float(KP) * (p - pr) - float(KV) * (v - vr)


def m_norm2(dp, dv):
    return M11 * dp * dp + 2 * M12 * dp * dv + M22 * dv * dv


def mm_up(x):
    """x rounded up to millimetres (or millimetres per second)."""
    return Fr(math.ceil(x * 1000), 1000)


def extents(s):
    """Bounds on |p - p_ref| and |v - v_ref| on {V_M <= s}, rounded up to
    millimetres: sqrt of the diagonal of M^-1 = [[1/4, -1/2], [-1/2, 2]]."""
    return mm_up(s / 2), mm_up(s * SQ2_UP)


def box_scale(box):
    """Smallest grid scale s with the box inside {V_M <= s} around its centre."""
    pc, vc = (box[1] + box[2]) / 2, (box[3] + box[4]) / 2
    v2 = max(m_norm2(p - pc, v - vc) for p in box[1:3] for v in box[3:5])
    return pc, vc, sqrt_up(v2)


def contains_m(outer, inner):
    """Exact, sound inclusion test for the sets of the mpc package."""
    if inner[0] == "cap":                 # the initial set lies in its first part
        inner = inner[1]
    ko, ki = outer[0], inner[0]
    if ko == "ell4":
        pc, vc, s = outer[1], outer[2], outer[3]
        if ki == "ell4":
            si = inner[3]
            return si <= s and m_norm2(inner[1] - pc, inner[2] - vc) <= (s - si) ** 2
        if ki == "box":
            return all(m_norm2(p - pc, v - vc) <= s * s
                       for p in inner[1:3] for v in inner[3:5])
        return inner[1] == inner[2] and m_norm2(inner[1] - pc, -vc) <= s * s
    if ko == "rest":
        plo, phi = outer[1], outer[2]
        if ki == "ell4":                  # via the bounding box of the tube
            pc, vc, s = inner[1], inner[2], inner[3]
            ep, ev = extents(s)
            return (vc - ev >= 0 and pc - ep >= plo
                    and pc + ep + (vc + ev) ** 2 / (2 * B_G) <= phi)
        return ORIGINAL["contains"](outer, inner)
    return False


def inside_m(s, p, v):
    """Whether the realized state (p, v) lies in the set s (floats)."""
    if s[0] == "cap":
        return inside_m(s[1], p, v) and inside_m(s[2], p, v)
    if s[0] == "ell4":
        dp, dv = p - float(s[1]), v - float(s[2])
        q = float(M11) * dp * dp + 2 * float(M12) * dp * dv + float(M22) * dv * dv
        sc = float(s[3])
        return q <= (sc * (1 + cf.TOL) + cf.TOL) ** 2
    return ORIGINAL["inside"](s, p, v)


def rest_containing_m(s):
    if s[0] == "cap":
        s = s[1]
    if s[0] == "ell4":
        pc, vc, sc = s[1], s[2], s[3]
        ep, ev = extents(sc)
        return cf.Rest(pc - ep, pc + ep + (vc + ev) ** 2 / (2 * B_G))
    return ORIGINAL["rest_containing"](s)


def initial_set_m(seg_ex, p, v):
    """I_{k+1}: the states that the executed tracking segment can reach at
    t_{k+1} from the measured state at t_k. The undisturbed motion from the
    measured state is predicted with the rational transition bound E_T, and
    the disturbance adds at most phi_M (1 - exp(-2 T)) in the metric M. The
    set is intersected with the end of the executed funnel, so it is
    admissible, and the fallback keeps its profile."""
    if seg_ex.kind != "track":
        return seg_ex.end()
    e0 = (Fr(p) - seg_ex.p0, Fr(v) - seg_ex.v0)
    z = [E_T[i][0] * e0[0] + E_T[i][1] * e0[1] for i in range(2)]
    zr = [Fr(round(x * GRID_Z), GRID_Z) for x in z]
    err = C_M * (EPS_T * max(abs(e0[0]), abs(e0[1])) + Fr(1, 2 * GRID_Z))
    s0 = round_up(PHI_M * (1 - A_LO) + err)
    return ("cap", ("ell4", seg_ex.p1 + zr[0], seg_ex.v1 + zr[1], s0), seg_ex.end())


def recovery_m(pred, k, j, p, v, s):
    """Recovery continuation of the mpc package from the reference state
    (p, v) with tube scale s at the start of period j: the strongest braking
    references that the labels and the bound of each pair allow, down to the
    handover speed, then the rest segment that contains the last tube."""
    pairs = []
    while v > CCM_V_H:
        for b_lab in (B_FIRM, B_RAMP):
            e_u = 4 * scale_next(s) * MARGIN
            seg = TrackS(p, v, -min(b_lab - e_u, (v - CCM_V_H) / T), s)
            beta = pred.label(k, j, seg)
            if seg.certified(beta):
                break
        else:
            raise AssertionError("braking at the global capability is certified")
        pairs.append((seg, beta))
        p, v, s, j = seg.p1, seg.v1, seg.s1, j + 1
    if pairs:
        return pairs, rest_containing_m(pairs[-1][0].end())
    return pairs, None


def generate_m(pred, k, init, q, fallback):
    """Candidates of period k: as in Section 5.1, with the tube scales of the
    mpc package and the head starting at the predicted centre."""
    Plan, ALPHAS = cf.Plan, cf.ALPHAS
    cands = []
    if init[0] == "rest":
        stay = Plan(k + 1, [], rest_containing_m(init))
        if q == "accelerate" and init[1] == init[2]:
            p0 = init[1]
            for a in ALPHAS[:-1]:
                head = cf.Launch(p0, a * U_MAX)
                pairs = [(head, pred.label(k, k + 1, head))]
                _, _, s0 = box_scale(head.end())
                tail, rest = recovery_m(pred, k, k + 2, head.p1, head.v1, s0)
                if not tail:
                    rest = rest_containing_m(head.end())
                cands.append((Plan(k + 1, pairs + tail, rest), a))
            cands.append((stay, Fr(0)))
        else:
            cands.append((stay, Fr(1) if q == "hold" else Fr(0)))
        return cands
    base = init[1] if init[0] == "cap" else init
    if base[0] == "ell4":
        pc, vc, s0 = base[1], base[2], base[3]
    else:
        pc, vc, s0 = box_scale(base)
    e_u = 4 * scale_next(s0) * MARGIN
    if q == "accelerate":
        a_q = U_MAX - e_u
    elif vc > CCM_V_H:
        a_q = -min(B_FIRM - e_u, (vc - CCM_V_H) / T)
    else:
        a_q = None
    fb_head = fallback.head()
    a_fb = fb_head.a if fb_head.kind == "track" else Fr(0)
    if a_q is None:
        cands.append((Plan(k + 1, [], rest_containing_m(init)), Fr(1)))
    else:
        for alpha in ALPHAS:
            head = TrackS(pc, vc, alpha * a_q + (1 - alpha) * a_fb, s0)
            pairs = [(head, pred.label(k, k + 1, head))]
            tail, rest = recovery_m(pred, k, k + 2, head.p1, head.v1, head.s1)
            if rest is None:
                rest = rest_containing_m(head.end())
            cands.append((Plan(k + 1, pairs + tail, rest), alpha))
    tail, rest = recovery_m(pred, k, k + 1, pc, vc, s0)
    if rest is None:
        rest = rest_containing_m(init)
    cands.append((Plan(k + 1, tail, rest), Fr(0)))
    return cands


def mpc_facts():
    """The tube scales of the mpc package."""
    assert A_LO < Fr(81873075, 10 ** 8) < 1     # exp(-1/5) = 0.8187307...
    s_init = round_up(PHI_M * (1 - A_LO))
    scales = [s_init]
    while len(scales) < 40:
        scales.append(scale_next(scales[-1]))
    head_budget = 4 * scales[1] * MARGIN
    n_half = next(n for n, s in enumerate(scales) if s >= PHI_M / 2)
    return {
        "decay_per_period": (f"{A_LO} = {float(A_LO):.8f} "
                             f"(lower bound of exp(-0.2))"),
        "initial_scale": f"{float(s_init):.5f} = {float(s_init / PHI_M):.3f} phi",
        "head_budget": f"{float(head_budget):.4f} m/s^2; accelerate {float(U_MAX - head_budget):.3f}",
        "periods_to_half_tube": str(n_half),
        "scale_after_20_periods": f"{float(scales[20] / PHI_M):.3f} phi",
        "transition_error_bound": f"{float(EPS_T):.1e}",
    }


MPC = dict(CCM, Track=TrackS, contains=contains_m, inside=inside_m,
           rest_containing=rest_containing_m, recovery=None,
           proposed_reference=None, generate=generate_m, initial_set=initial_set_m)
MPC = {k: v for k, v in MPC.items() if v is not None}

PACKAGES = [("tube", {}), ("qp", QP), ("ccm", CCM), ("mpc", MPC)]


# ----------------------------------------------------------------------------
# Runs and facts
# ----------------------------------------------------------------------------
def package_facts(name):
    """Certified quantities of the installed package."""
    if name == "mpc":
        e_u = 4 * PHIBAR_M
        e_p, e_v = extents(PHIBAR_M)
        offset = e_p + (CCM_V_H + e_v) ** 2 / (2 * B_G)
        return {"bounds_steady": f"{fmt(e_p * 1000)} mm, {fmt(e_v * 1000)} mm/s",
                "budget_steady": f"{fmt(e_u)} m/s^2",
                "ramp_braking_steady": fmt(B_RAMP - e_u),
                "handover_speed": fmt(CCM_V_H),
                "handover_offset_steady": f"{float(offset):.6f} m"}
    e_u, e_p, e_v, v_h = cf.E_U, cf.E_P, cf.E_V, cf.V_H
    return {"tube_extents": f"{fmt(e_p * 1000)} mm, {fmt(e_v * 1000)} mm/s",
            "budget": f"{fmt(e_u)} m/s^2",
            "ramp_braking": fmt(B_RAMP - e_u),
            "firm_braking_and_acceleration": fmt(U_MAX - e_u),
            "handover_speed": fmt(v_h),
            "handover_offset": fmt(e_p + (v_h + e_v) ** 2 / (2 * B_G))}


def run_facts(rows, motion):
    f = cf.facts_of_run("", rows, motion)
    keep = ("arrival_s", "rest", "alarms", "steps_outside_funnel",
            "largest_tube_ratio", "largest_feedback_input", "least_braking_margin",
            "vmax", "ramp_entry")
    out = {k: f[k] for k in keep}
    body = [x for x in rows if not x.get("final")]
    ramp = [x["a"] for x in body if x["kind"] == "track" and x["p"] >= float(cf.RAMP)]
    out["strongest_reference_braking_on_ramp"] = fmt(-min(ramp))
    out["final_level"] = fmt(body[-1]["eps"])
    # the realized error relative to the funnel at each instant, V(e)/phi(tau)
    worst = 0.0
    for t, tau, p, v, u, seg in motion:
        if seg.kind == "track":
            s = seg.funnel_at(tau)
            pr, vr = s[1], s[2]
            lev = s[3] if s[0] == "ell4" else math.sqrt(float(cf.PHI2))
            dp, dv = p - pr, v - vr
            q = (float(cf.P11) * dp * dp + 2 * float(cf.P12) * dp * dv
                 + float(cf.P22) * dv * dv)
            worst = max(worst, math.sqrt(q) / lev)
    out["largest_ratio_to_funnel"] = f"{worst:.3f}"
    return out, worst


def robustness_facts():
    alarms, out = 0, 0
    for kind in ("fixed", "forecasting"):
        for mode in ("plus", "minus", "worst"):
            rows, _, _ = cf.run(kind, mode=mode)
            body = [x for x in rows if not x.get("final")]
            alarms += sum(x["alarm"] for x in body)
            out += sum(x["out"] for x in body)
    return f"6 runs, {alarms} alarms, {out} steps outside a funnel"


def endphase(motion, t_from):
    return [(m[2], m[3]) for m in motion if m[0] >= t_from - 1e-9]


def chapter3_arrivals(here):
    """Arrival times of the runs of Chapter 3, read from its summary."""
    path = os.path.join(here, "..", "ch03", "results", "summary.txt")
    out, group = {}, None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("["):
                group = line.strip("[]")
            elif line.startswith("arrival_s = ") and group in ("fixed", "forecasting"):
                out[group] = Fr(line.split(" = ", 1)[1])
    return out


def comparison_facts(facts, worst, here):
    """The differences between the packages quoted in Section 5.3.4, computed
    from the values in Table 5.5 as printed, and the shares of the funnel from
    the unrounded ratios."""
    names = [n for n, _ in PACKAGES]
    kinds = ("fixed", "forecasting")

    def arrival(n, k):
        return Fr(facts[f"{n}_{k}"]["arrival_s"])

    def rest(n, k):
        return Fr(facts[f"{n}_{k}"]["rest"].split(" m")[0])

    def level(n, k):
        return Fr(facts[f"{n}_{k}"]["final_level"])

    ch3 = chapter3_arrivals(here)
    best = {k: min(arrival(n, k) for n in names) for k in kinds}
    return {
        "share_of_funnel": ", ".join(
            f"{n} {percent(max(worst[n, k] for k in kinds))} %" for n in names),
        "earlier_than_tube_s": "; ".join(
            f"{n} " + ", ".join(fmt(arrival("tube", k) - arrival(n, k)) for k in kinds)
            for n in names[1:]),
        "rest_closer_than_tube_mm": "; ".join(
            f"{n} " + ", ".join(fmt((rest(n, k) - rest("tube", k)) * 1000) for k in kinds)
            for n in names[1:]),
        "reserve_above_floor_m": "; ".join(
            f"{n} " + ", ".join(fmt(level(n, k) - cf.EPS_MIN) for k in kinds)
            for n in names),
        "recovered_of_chapter3_delay_s": "; ".join(
            f"{k} {fmt(arrival('tube', k) - best[k])} of "
            f"{fmt(arrival('tube', k) - ch3[k])}" for k in kinds),
    }


def write_tubes(path):
    """Cross-sections of the tubes in the plane of the tracking error, in mm
    and mm/s (Figure 5.5): the tube of Section 5.1, which the qp package
    shares; the tube of the metric M (ccm, and mpc at its steady scale); and
    the mpc tube at the predicted initial set and 1, 2, 4 and 9 periods
    later (scales s0, s1, s2, s4, s9)."""
    s = [round_up(PHI_M * (1 - A_LO))]
    while len(s) < 11:
        s.append(scale_next(s[-1]))
    sets = [("p", (ORIGINAL["P11"], ORIGINAL["P12"], ORIGINAL["P22"]),
             math.sqrt(float(ORIGINAL["PHI2"]))),
            ("m", (M11, M12, M22), float(PHI_M))]
    sets += [(f"s{n}", (M11, M12, M22), float(s[n])) for n in (0, 1, 2, 4, 9)]
    cols = []
    for name, (a, b, c), lev in sets:
        l11 = math.sqrt(float(a))
        l21 = float(b) / l11
        l22 = math.sqrt(float(c) - l21 * l21)
        pts = []
        for i in range(181):
            th = 2 * math.pi * i / 180
            ev = lev * math.sin(th) / l22
            ep = (lev * math.cos(th) - l21 * ev) / l11
            pts.append((1000 * ep, 1000 * ev))
        cols.append((name, pts))
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([c for name, _ in cols for c in (f"dp_{name}", f"dv_{name}")])
        for i in range(181):
            w.writerow([f"{x:.4f}" for _, pts in cols for x in pts[i]])
    return [float(x / PHI_M) for x in s]


EXPECTED = {   # every number quoted in Section 5.3, as in the text
    'qp_certificate': {
        'budget': '0.115 m/s^2, proved; largest value 0.1146 m/s^2',
        'ratio_to_linear': '0.917 = 92 %',
        'polynomial_degree': '4',
    },
    'ccm_certificate': {
        'lyapunov_equation': "A_K'M + MA_K + 4M = 0",
        'rho': '8',
        'levels': 'tube 0.025, bounds 0.025313',
        'exact_extents': '0.01266 m, 0.03580 m/s, 0.10125 m/s^2',
        'bounds_used': '0.013 m, 0.036 m/s, 0.10125 m/s^2',
        'launch_box_V_over_phi': '0.221',
        'inside_tube_of_5_1': '0.805',
        'handover_offset': '0.019418',
    },
    'mpc_certificate': {
        'decay_per_period': '38378/46875 = 0.81873067 (lower bound of exp(-0.2))',
        'initial_scale': '0.00453 = 0.181 phi',
        'head_budget': '0.0334 m/s^2; accelerate 1.967',
        'periods_to_half_tube': '3',
        'scale_after_20_periods': '0.985 phi',
        'transition_error_bound': '2.1e-23',
        'scales_in_figure': '0.181,0.330,0.451,0.632,0.865',
        'scales_in_figure_percent': '18, 33, 45, 63, 86',
    },
    'tube_package': {
        'tube_extents': '19 mm, 49 mm/s',
        'budget': '0.125 m/s^2',
        'ramp_braking': '0.375',
        'handover_speed': '0.05',
        'handover_offset': '0.02989',
    },
    'qp_package': {
        'tube_extents': '19 mm, 49 mm/s',
        'budget': '0.115 m/s^2',
        'ramp_braking': '0.385',
        'handover_speed': '0.05',
        'handover_offset': '0.02989',
    },
    'ccm_package': {
        'tube_extents': '13 mm, 36 mm/s',
        'budget': '0.10125 m/s^2',
        'ramp_braking': '0.39875',
        'handover_speed': '0.04',
        'handover_offset': '0.019418',
    },
    'mpc_package': {
        'bounds_steady': '13 mm, 36 mm/s',
        'budget_steady': '0.10125 m/s^2',
        'ramp_braking_steady': '0.39875',
        'handover_speed': '0.04',
    },
    'tube_fixed': {
        'arrival_s': '11.2',
        'rest': '24.467 m, at rest from 12.7 s',
        'alarms': '0',
        'steps_outside_funnel': '0',
        'largest_feedback_input': '0.0613 m/s^2',
        'largest_ratio_to_funnel': '0.516',
        'strongest_reference_braking_on_ramp': '0.375',
        'final_level': '0.5',
    },
    'tube_forecasting': {
        'arrival_s': '9.2',
        'rest': '24.468 m, at rest from 10.7 s',
        'alarms': '0',
        'steps_outside_funnel': '0',
        'largest_feedback_input': '0.0613 m/s^2',
        'largest_ratio_to_funnel': '0.516',
        'strongest_reference_braking_on_ramp': '0.375',
        'final_level': '0.5',
    },
    'tube_robustness': {
        'runs': '6 runs, 0 alarms, 0 steps outside a funnel',
    },
    'qp_fixed': {
        'arrival_s': '11.0',
        'rest': '24.471 m, at rest from 12.5 s',
        'alarms': '0',
        'steps_outside_funnel': '0',
        'largest_feedback_input': '0.0580 m/s^2',
        'largest_ratio_to_funnel': '0.571',
        'strongest_reference_braking_on_ramp': '0.385',
        'final_level': '0.5',
    },
    'qp_forecasting': {
        'arrival_s': '9.1',
        'rest': '24.459 m, at rest from 10.6 s',
        'alarms': '0',
        'steps_outside_funnel': '0',
        'largest_feedback_input': '0.0580 m/s^2',
        'largest_ratio_to_funnel': '0.571',
        'strongest_reference_braking_on_ramp': '0.385',
        'final_level': '0.51',
    },
    'qp_robustness': {
        'runs': '6 runs, 0 alarms, 0 steps outside a funnel',
    },
    'ccm_fixed': {
        'arrival_s': '10.9',
        'rest': '24.482 m, at rest from 12.4 s',
        'alarms': '0',
        'steps_outside_funnel': '0',
        'largest_feedback_input': '0.0613 m/s^2',
        'largest_ratio_to_funnel': '0.655',
        'strongest_reference_braking_on_ramp': '0.39875',
        'final_level': '0.5',
    },
    'ccm_forecasting': {
        'arrival_s': '9.1',
        'rest': '24.478 m, at rest from 10.7 s',
        'alarms': '0',
        'steps_outside_funnel': '0',
        'largest_feedback_input': '0.0613 m/s^2',
        'largest_ratio_to_funnel': '0.655',
        'strongest_reference_braking_on_ramp': '0.39875',
        'final_level': '0.5',
    },
    'ccm_robustness': {
        'runs': '6 runs, 0 alarms, 0 steps outside a funnel',
    },
    'mpc_fixed': {
        'arrival_s': '10.8',
        'rest': '24.458 m, at rest from 12.4 s',
        'alarms': '0',
        'steps_outside_funnel': '0',
        'largest_feedback_input': '0.0315 m/s^2',
        'largest_ratio_to_funnel': '0.995',
        'strongest_reference_braking_on_ramp': '0.454317',
        'final_level': '0.53',
    },
    'mpc_forecasting': {
        'arrival_s': '9.0',
        'rest': '24.442 m, at rest from 10.4 s',
        'alarms': '0',
        'steps_outside_funnel': '0',
        'largest_feedback_input': '0.0315 m/s^2',
        'largest_ratio_to_funnel': '0.995',
        'strongest_reference_braking_on_ramp': '0.454317',
        'final_level': '0.55',
    },
    'mpc_robustness': {
        'runs': '6 runs, 0 alarms, 0 steps outside a funnel',
    },
    'comparison': {
        'share_of_funnel': 'tube 52 %, qp 57 %, ccm 65 %, mpc 99.5 %',
        'earlier_than_tube_s': 'qp 0.2, 0.1; ccm 0.3, 0.1; mpc 0.4, 0.2',
        'rest_closer_than_tube_mm': 'qp 4, -9; ccm 15, 10; mpc -9, -26',
        'reserve_above_floor_m': 'tube 0, 0; qp 0, 0.01; ccm 0, 0; mpc 0.03, 0.05',
        'recovered_of_chapter3_delay_s': 'fixed 0.4 of 1.3; forecasting 0.2 of 0.6',
    },
}


def main():
    ap = argparse.ArgumentParser(description="Chapter 5: three established methods")
    ap.add_argument("--check", action="store_true",
                    help="check every number quoted in Section 5.3")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "results")
    os.makedirs(out, exist_ok=True)
    facts = {"qp_certificate": qp_budget_proof(), "ccm_certificate": metric_facts(),
             "mpc_certificate": mpc_facts()}
    scales = write_tubes(os.path.join(out, "methods-tubes.csv"))
    facts["mpc_certificate"]["scales_in_figure"] = ",".join(
        f"{scales[n]:.3f}" for n in (0, 1, 2, 4, 9))
    facts["mpc_certificate"]["scales_in_figure_percent"] = ", ".join(
        percent(scales[n]) for n in (0, 1, 2, 4, 9))
    curves, worst = {}, {}
    for name, changes in PACKAGES:
        install(changes)
        facts[f"{name}_package"] = package_facts(name)
        for kind in ("fixed", "forecasting"):
            rows, _, motion = cf.run(kind)
            facts[f"{name}_{kind}"], worst[name, kind] = run_facts(rows, motion)
            if kind == "forecasting":
                curves[name] = endphase(motion, 5.0)
        facts[f"{name}_robustness"] = {"runs": robustness_facts()}
    install({})
    facts["comparison"] = comparison_facts(facts, worst, here)
    with open(os.path.join(out, "methods-endphase.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        names = [n for n, _ in PACKAGES]
        w.writerow([c for n in names for c in (f"p_{n}", f"v_{n}")])
        length = max(len(curves[n]) for n in names)
        for i in range(length):
            row = []
            for n in names:
                if i < len(curves[n]):
                    row += [f"{curves[n][i][0]:.5f}", f"{curves[n][i][1]:.5f}"]
                else:
                    row += ["nan", "nan"]
            w.writerow(row)
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
    with open(os.path.join(out, "methods-summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
