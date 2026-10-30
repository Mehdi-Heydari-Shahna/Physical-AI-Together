"""The declared model of the complete case and the certificates of its two
control packages (Section 16.4).

The model is that of the simulated cart at the simulator's time step
h = 1 ms, in the track coordinate of Section 11.1: with the commanded
acceleration u_i held over step i,

    v[i+1] = v[i] + h (u[i] + w[i]),       p[i+1] = p[i] + h v[i+1] + d[i],

|w[i]| <= w_bar and |d[i]| <= d_bar, while the command stays within the
command bound, 2 m/s^2, and, towards the start, within the braking that the
label of the period asserts. The tool's reach follows its command within
e_l, except while it moves to a new command: it arrives within
TOOL_PERIODS = 4 periods (the plant moves it along a smooth path in 0.35 s),
and in those periods it lies between -e_l and 0.8 m + e_l; the tool's
reaction may add to the w[i] of those periods and the next at most an
impulse J [m/s] beyond w_bar, the sum of h max(0, |w[i]| - w_bar) over them. The sensor's errors are
bounded by varpi_p, varpi_v and varpi_l. Every bound is established on the
identification runs (complete_identify.py) and frozen in FROZEN.json before
the evaluation worlds are opened; PROVISIONAL holds the values used while the
software was developed, on development worlds only.

A tracking segment follows a reference with a constant acceleration a over
one period, on the simulator's grid,

    v_ref[i] = v0 + i h a,       p_ref[i] = p0 + v0 i h + a h^2 i (i + 1) / 2,

with the law u = a - kp (p_meas - p_ref) - kv (v_meas - v_ref), kp = 8 /s^2
and kv = 4 /s, the gains of Chapter 5. The error e = x - x_ref then obeys

    e[i+1] = A e[i] + g (w[i] - K n[i]) + (d[i], 0),

A = [[1 - h^2 kp, h - h^2 kv], [-h kp, 1 - h kv]], g = (h^2, h), n the
sensor's error. For a quadratic form Q with A'QA <= rho^2 Q, the norm
V(e) = sqrt(e'Qe) satisfies V(e[i+1]) <= rho V(e[i]) + |g|_Q w_tot +
sqrt(Q11) d_bar, w_tot = w_bar + kp varpi_p + kv varpi_v: the level

    phi = (|g|_Q w_tot + sqrt(Q11) d_bar) / (1 - rho)

is invariant, and an error that starts below a level s ends a period below
rho^100 s + (1 - rho^100) phi. The tube package uses Q = P of Chapter 5,
the MPC package Q = M of Section 5.3. certificate() checks the inequality in
exact rational arithmetic and returns every constant as a float rounded
outward.
"""
import json
import math
import os
from fractions import Fraction as Fr

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# The task and the track (Chapters 3, 10 and 16, Section 11.1)
# ----------------------------------------------------------------------------
T = 0.1                    # period of the governor [s]
H = 0.001                  # time step of the simulator [s]
N_STEPS = 100              # time steps per period
U_MAX = 2.0                # command bound [m/s^2]
WALL = 25.0                # constraint p + l <= 25 m
WORK_ZONE = 24.0           # the task: p >= 24 m
ELL = 0.8                  # reach of the unfolded tool [m]
LEAD = 0.1                 # the bumper leads the front axle by 0.1 m (Section 11.1)
EPS_MIN = Fr(1, 2)         # reserve floor [m]
GRID = Fr(1, 100)          # levels: whole centimetres ...
EPS_TOP = Fr(25)           # ... from 0.50 m to 25 m
OPERATION_LEVEL = Fr(13, 10)   # the operation's reserve: floor + reach
HORIZON = 400              # forecast horizon [periods]
TOOL_PERIODS = 4           # the tool reaches a new commanded reach within 4 periods
TOL = 1e-9                 # outward rounding of floating-point quantities [m, m/s]

# ----------------------------------------------------------------------------
# Capabilities
# ----------------------------------------------------------------------------
B_FIRM = 2.0     # label 'level ground': the front wheel's contact stays on the level
                 # ground before the crest; braking of at least 2 m/s^2 at any speed
B_GLOBAL = 0.5   # the global capability: braking of at least 0.5 m/s^2 everywhere,
                 # at speeds of at most V_C
V_C = 2.0        # [m/s]; above about 2.4 m/s the wheels leave the crest (Section 11.1.3)

# ----------------------------------------------------------------------------
# The law
# ----------------------------------------------------------------------------
KP, KV = 8.0, 4.0
KP_FR, KV_FR = Fr(8), Fr(4)

# The two quadratic forms, exact, with the discrete rates that certificate()
# checks: rho = 1 - gamma h.
FORMS = {
    "tube": dict(Q=(Fr(34, 5), Fr(2), Fr(1)), gamma=Fr(8, 5)),     # P of Section 5.1
    "mpc": dict(Q=(Fr(8), Fr(2), Fr(1)), gamma=Fr(199, 100)),      # M of Section 5.3
}

MARGIN = Fr(81, 80)        # the bounds hold on a level 1.25 % above the funnel's

# ----------------------------------------------------------------------------
# The bounds of the model
# ----------------------------------------------------------------------------
PROVISIONAL = dict(w_bar=0.06, d_bar=1e-6, J=0.005, varpi_p=0.0015, varpi_v=0.003,
                   varpi_l=0.0006, e_l=0.001)
"""Development values; never used for an evaluation run."""

FROZEN_FILE = os.path.join(HERE, "FROZEN.json")


def bounds(frozen=True):
    """The bounds of the model: those of FROZEN.json if frozen is True (an
    error if the file does not exist yet), else PROVISIONAL."""
    if not frozen:
        return dict(PROVISIONAL)
    with open(FROZEN_FILE) as fh:
        data = json.load(fh)
    return {k: float(data["bounds"][k]) for k in PROVISIONAL}


def up(x, digits=12):
    """A Fraction rounded up to a float on the grid 10**-digits."""
    q = Fr(10) ** digits
    return float(Fr(math.ceil(Fr(x) * q), 1) / q)


def sqrt_up(x, digits=12):
    """An upper bound of sqrt(x), x a nonnegative Fraction, as a float."""
    q = 10 ** digits
    n = math.isqrt(math.ceil(Fr(x) * q * q))
    while Fr(n * n, q * q) < x:
        n += 1
    return float(Fr(n, q))


def _psd(a, b, c):
    """[[a, b], [b, c]] is positive semidefinite (exact)."""
    return a >= 0 and c >= 0 and a * c - b * b >= 0


def certificate(package, b):
    """The certificate of the tracking law in the form of 'package' ('tube'
    or 'mpc') for the bounds b (bounds()). Checks exactly that Q is positive
    definite and that A'QA <= rho^2 Q, and returns floats, every bound
    rounded up and rho rounded up:
      rho, rho_n (rho**100), g (|g|_Q), q1 (sqrt(Q11)),
      w_tot, phi (the invariant level), iota (the rise of the level that the
      tool's impulse J can cause), and the factors that turn a level s into
      bounds: fp (|e_p| <= fp s), fv (|e_v| <= fv s), fu (|K e| <= fu s),
      with noise = kp varpi_p + kv varpi_v added to the feedback's bound."""
    form = FORMS[package]
    q11, q12, q22 = form["Q"]
    gamma = form["gamma"]
    h = Fr(1, 1000)
    rho = 1 - gamma * h
    assert q11 > 0 and q11 * q22 - q12 * q12 > 0, "Q is not positive definite"
    a11, a12 = 1 - h * h * KP_FR, h - h * h * KV_FR
    a21, a22 = -h * KP_FR, 1 - h * KV_FR
    # A'QA
    qa11 = q11 * a11 + q12 * a21
    qa12 = q11 * a12 + q12 * a22
    qa21 = q12 * a11 + q22 * a21
    qa22 = q12 * a12 + q22 * a22
    m11 = a11 * qa11 + a21 * qa21
    m12 = a11 * qa12 + a21 * qa22
    m22 = a12 * qa12 + a22 * qa22
    r2 = rho * rho
    assert _psd(r2 * q11 - m11, r2 * q12 - m12, r2 * q22 - m22), \
        f"{package}: A'QA <= rho^2 Q fails"
    det = q11 * q22 - q12 * q12
    inv11, inv12, inv22 = q22 / det, -q12 / det, q11 / det
    g2 = q11 * h ** 4 + 2 * q12 * h ** 3 + q22 * h ** 2
    kqk = KP_FR ** 2 * inv11 + 2 * KP_FR * KV_FR * inv12 + KV_FR ** 2 * inv22
    w_tot = Fr(b["w_bar"]) + KP_FR * Fr(b["varpi_p"]) + KV_FR * Fr(b["varpi_v"])
    g_up = Fr(sqrt_up(g2, 15))
    q1 = Fr(sqrt_up(q11))
    phi = (g_up * w_tot + q1 * Fr(b["d_bar"])) / (1 - rho)
    rho_n = up(rho ** 100, 15)
    return dict(
        package=package, rho=up(rho, 15), rho_n=rho_n, g=float(g_up), q1=float(q1),
        w_tot=up(w_tot), phi=up(phi), iota=up(g_up / h * Fr(b["J"])),
        fp=sqrt_up(inv11), fv=sqrt_up(inv22), fu=sqrt_up(kqk),
        noise=up(KP_FR * Fr(b["varpi_p"]) + KV_FR * Fr(b["varpi_v"])),
        q=(float(q11), float(q12), float(q22)), gamma=float(gamma))


def norm_q(q, dp, dv):
    """sqrt((dp, dv) Q (dp, dv)')."""
    q11, q12, q22 = q
    return math.sqrt(max(0.0, q11 * dp * dp + 2 * q12 * dp * dv + q22 * dv * dv))


def box_level(q, ep, ev):
    """The largest V over the box |dp| <= ep, |dv| <= ev: at a corner."""
    return max(norm_q(q, sp * ep, sv * ev) for sp in (-1, 1) for sv in (-1, 1))
