#!/usr/bin/env python3
"""
Learned proposal methods for the cart.

Physical AI, Together, Chapter 6, Sections 6.1 to 6.3.

The control package of Sections 5.1 and 5.2, the forecasting predictor and
the governor are imported from ../ch05/cart_feedback.py and used unchanged;
the contraction-metric package of Section 5.3 comes from cart_methods.py.
This script adds what a learning researcher supplies: proposal methods with
a stated physical meaning and horizon, and a training and evaluation
protocol. The adapters that turn a proposal into candidate plans belong to
the control method's generator; they are here because Chapter 5 offered only
the proposals 'accelerate' and 'hold'.

Proposals
    reference   one reference acceleration for the next period [m/s^2]; the
                generator blends it with the head of the fallback on the
                ambition grid of Chapter 5 (Off-policy learner, braking
                curve, rule)
    sequence    eight reference accelerations for the next 0.8 s; the
                generator offers prefixes of 8, 6, 4 and 2 periods with
                ambitions 1, 3/4, 1/2 and 1/4 (planner with a learned model)
    chunk       the same format, served from a chunk that the policy keeps
                for up to four periods (chunk imitator)
    None        no proposal before the deadline: recovery candidates only

The world family
    Each world draws, from its seed, the start of the downhill ramp in
    [17, 21] m (on a 1 cm grid) and a triangle-wave disturbance with
    amplitude below 0.05 m/s^2, period in [1.5, 3] s and a random phase. The
    forecasting predictor labels each segment with the braking of the
    world's own map, so every label is valid.

The run
    The task is to reach the work zone (24 m) within 20 s. Every run then
    continues until the cart is at rest, at most until 25 s; the proposal
    method keeps proposing, without exploring, and nothing after arrival
    enters its records or the mean intervention. A run stops at an alarm.

Reproducibility
    The governor and the test compute exactly, with fractions. The
    simulation and the learners use only floating-point +, -, *, /, sqrt and
    rounding, in a fixed order: floats are summed from left to right
    (total()), because Python's sum() of floats is compensated from version
    3.12 on. Means are computed exactly and rounded half up, as in Chapter 5.
    Two exceptions call the platform's math library, whose last digits may
    differ between platforms: math.exp in the funnel check of the
    contraction-metric package (cart_methods.py), and the sine disturbance
    of the world of Chapter 5. Random numbers come only from
    random.Random(seed).random() with integer seeds. Python promises only
    that random() keeps producing the same sequence when the compatible
    seeder is given the same seed; the script relies on integer seeding
    being that seeder. The results were identical, bit for bit, in runs
    with CPython 3.8, 3.13 and 3.14 on one x86-64 Linux machine; other
    processors, operating systems and versions have not been tested.

Usage
    python3 cart_learning.py              # the whole protocol; writes results/
    python3 cart_learning.py --check      # also checks every quoted number
    python3 cart_learning.py --jobs 1     # one process (default: up to 2)
    python3 cart_learning.py --quick      # a fast smoke test; numbers differ

Outputs, in results/ next to this file
    summary.txt         the numbers quoted in Chapter 6
    trace-rule.csv      proposed and executed references in the world of
    trace-learner.csv   Chapter 5, for the rule and the off-policy learner
                        (Table 6.2 and Figure 6.1)
    training.csv        one row per training episode of every learner
    evaluation.csv      one row per evaluation run of every policy

Python 3.8 or later; no other packages. A full run takes about fifteen
minutes with two processes.
"""
import argparse
import csv
import math
import os
import random
import sys
import time
from fractions import Fraction as Fr
from multiprocessing import Pool

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ch05"))
import cart_feedback as cf    # noqa: E402
import cart_methods as cm     # noqa: E402

T = cf.T
LOOK = 30                         # reach of the label summary [m]
T_LIMIT = Fr(20)                  # an episode ends at arrival or after 20 s
H = 8                             # length of a sequence or chunk [periods]
PREFIX = [(8, Fr(1)), (6, Fr(3, 4)), (4, Fr(1, 2)), (2, Fr(1, 4))]
WORK = float(cf.WORK_ZONE)        # arrival: the measured position reaches 24 m


# ----------------------------------------------------------------------------
# The world family
# ----------------------------------------------------------------------------
class World:
    """One world: where the ramp starts and which disturbance acts."""

    def __init__(self, seed):
        rng = random.Random(seed)
        self.seed = seed
        self.ramp = Fr(round((17 + 4 * rng.random()) * 100), 100)
        self.amp = 0.05 * rng.random()
        self.per = 1.5 + 1.5 * rng.random()
        self.phase = rng.random()
        self.sine = False

    @classmethod
    def chapter5(cls):
        """The world of Chapter 5: ramp at 20 m, the sine signal of its runs."""
        w = cls(0)
        w.ramp, w.sine, w.seed = Fr(20), True, "ch5"
        return w

    def install(self):
        r = float(self.ramp)
        cf.RAMP = self.ramp
        cf.b_true_f = lambda p, r=r: 2.0 if p < r else 0.5

    def w(self, t, seg, tau, p, v):
        if self.sine:
            return float(cf.W_BAR) * math.sin(2 * math.pi * t / cf.W_PERIOD)
        x = t / self.per + self.phase
        s = x - math.floor(x)
        return self.amp * (4 * abs(s - 0.5) - 1)


WORLD_DEFAULTS = dict(RAMP=cf.RAMP, b_true_f=cf.b_true_f)


def restore_world():
    cf.RAMP, cf.b_true_f = WORLD_DEFAULTS["RAMP"], WORLD_DEFAULTS["b_true_f"]


# ----------------------------------------------------------------------------
# Observations
# ----------------------------------------------------------------------------
def grid(x, n=1000):
    """A float on the grid 1/n, as an exact fraction."""
    return Fr(round(float(x) * n), n)


class Probe:
    """A hypothetical region [lo, hi], for querying the predictor's labels."""

    kind = "probe"

    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi


def low_braking_distance(pred, k, p):
    """The label summary of the observation: distance in whole metres (1 to
    LOOK) to the first region that the labels mark with less braking than
    firm ground; 0 if the cart is on it; None if there is none in reach."""
    pf = grid(p)
    if pred.label(k, k + 1, Probe(pf, pf)) < cf.B_FIRM:
        return 0
    for d in range(1, LOOK + 1):
        if pred.label(k, k + 1, Probe(pf, pf + d)) < cf.B_FIRM:
            return d
    return None


def declared():
    """Constants that the installed control package declares (Table 5.1 for
    the tube package): the largest reference acceleration, the largest
    reference braking on firm ground and on the ramp, the handover speed,
    and the handover position from which the rest segment stops before the
    reserve floor. The braking on firm ground and on the ramp is the braking
    that the labels issue, the same in every world of the family."""
    offset = cf.E_P + (cf.V_H + cf.E_V) ** 2 / (2 * cf.B_G)
    return dict(a_max=cf.A_MAX, b_firm=min(cf.B_FIRM, cf.U_MAX) - cf.E_U,
                b_low=cf.B_RAMP - cf.E_U, v_h=cf.V_H,
                x_h=cf.WALL - cf.EPS_MIN - offset)


DECLARED = None          # the declared constants of the installed package


def use_package(name):
    """Installs a control package of Chapter 5 ('tube' or 'ccm') and reads
    its declared constants."""
    global DECLARED
    cm.install(cm.CCM if name == "ccm" else {})
    DECLARED = declared()


def total(xs):
    """The sum of floats, added from left to right. Python's sum() of floats
    is compensated from version 3.12 on, so its last bits depend on the
    version; this sum does not."""
    s = 0.0
    for x in xs:
        s += x
    return s


def v_env(p, d_low):
    """A feature, not a certificate: the largest reference speed at p from
    which braking at the declared reference braking, on firm ground and from
    the low-braking region on, reaches the handover speed by the handover
    position."""
    c = DECLARED
    b1, b2, x_h, vh = (float(c[k]) for k in ("b_firm", "b_low", "x_h", "v_h"))
    if d_low is None:
        return math.sqrt(max(0.0, vh * vh + 2 * b1 * (x_h - p)))
    r = p + d_low
    vr2 = vh * vh + 2 * b2 * max(0.0, x_h - max(r, p))
    return math.sqrt(vr2 + 2 * b1 * max(0.0, r - p))


def predict_next(obs):
    """The state at t_{k+1}, where a proposal formed at t_k takes effect,
    predicted from the reference that the cart executes in period k: the
    head of P_k, which the governor reported at t_k. Nominal constant
    acceleration over one period; a rest head brakes at 2 m/s^2 to
    standstill. Returns the predicted position and velocity and the label
    summary shifted to the predicted position."""
    p, v, dt = obs["p"], obs["v"], float(T)
    if obs["kind_now"] == "rest":
        p1 = p + (v * v / 4.0 if v <= 2.0 * dt else v * dt - dt * dt)
        v1 = max(0.0, v - 2.0 * dt)
    else:
        a = obs["a_ex_now"]
        p1, v1 = p + v * dt + a * dt * dt / 2.0, max(0.0, v + a * dt)
    d = obs["d_low"]
    return p1, v1, None if d is None else max(0.0, d - (p1 - p))


# ----------------------------------------------------------------------------
# Adapters: from proposals to candidate plans (part of the generator)
# ----------------------------------------------------------------------------
def centre(init):
    if init[0] == "ell":
        return init[1], init[2]
    return (init[1] + init[2]) / 2, (init[3] + init[4]) / 2


def clip_reference(a):
    return max(-cf.A_MAX, min(cf.A_MAX, a))


def exact(x):
    """A requested acceleration as an exact rational; None if not finite.
    Any proposal method may return floats; the segments stay exact."""
    try:
        return Fr(x)
    except (TypeError, ValueError, OverflowError):
        return None


def generate_reference(pred, k, init, q, fallback):
    """Candidates for a reference proposal q [m/s^2], or q = None. As in
    Section 5.1.6, with the requested acceleration in place of 'accelerate';
    a request to brake becomes the braking reference down to the handover
    speed, and at the handover speed the rest segment."""
    q = None if q is None else exact(q)
    cands = []
    if init[0] == "rest":
        stay = cf.Plan(k + 1, [], cf.rest_containing(init))
        if q is not None and q > 0 and init[1] == init[2]:
            p0, a_l = init[1], min(q, cf.U_MAX)
            for a in cf.ALPHAS[:-1]:
                head = cf.Launch(p0, a * a_l)
                pairs = [(head, pred.label(k, k + 1, head))]
                tail, rest = cf.recovery(pred, k, k + 2, head.p1, head.v1)
                if not tail:
                    rest = cf.rest_containing(head.end())
                cands.append((cf.Plan(k + 1, pairs + tail, rest), a))
            cands.append((stay, Fr(0)))
        else:
            cands.append((stay, Fr(1) if (q is not None and q <= 0) else Fr(0)))
        return cands
    pc, vc = centre(init)
    if q is not None:
        a_q = clip_reference(q)
        if a_q < 0:
            a_q = -min(-a_q, (vc - cf.V_H) / T) if vc > cf.V_H else None
        fb_head = fallback.head()
        a_fb = fb_head.a if fb_head.kind == "track" else Fr(0)
        if a_q is None:
            cands.append((cf.Plan(k + 1, [], cf.rest_containing(init)), Fr(1)))
        else:
            for alpha in cf.ALPHAS:
                head = cf.Track(pc, vc, alpha * a_q + (1 - alpha) * a_fb)
                pairs = [(head, pred.label(k, k + 1, head))]
                tail, rest = cf.recovery(pred, k, k + 2, head.p1, head.v1)
                if rest is None:
                    rest = cf.rest_containing(head.end())
                cands.append((cf.Plan(k + 1, pairs + tail, rest), alpha))
    tail, rest = cf.recovery(pred, k, k + 1, pc, vc)
    if rest is None:
        rest = cf.rest_containing(init)
    cands.append((cf.Plan(k + 1, tail, rest), Fr(0)))
    return cands


def realizable(q, init):
    """Whether the generator of a reference proposal can realize the request
    q exactly: from standstill, a launch command of at most 2 m/s^2 (a
    request to brake or hold is realized by holding); in motion, a reference
    within the limit of 1.875 m/s^2 that, if it brakes, does not have to be
    cut at the handover speed. The candidate of ambition 1 then realizes the
    proposal unchanged (the faithful generator of Section 6.3)."""
    if q is None:
        return False
    q = exact(q)
    if q is None:
        return False
    if init[0] == "rest":
        return q <= 0 or (init[1] == init[2] and q <= cf.U_MAX)
    if clip_reference(q) != q:
        return False
    _, vc = centre(init)
    return q >= 0 or (vc > cf.V_H and -q <= (vc - cf.V_H) / T)


def follow(pred, k, j, p, v, seq):
    """Tracking pairs that follow the references seq from (p, v), the first
    in period j; a request to brake below the handover speed ends them."""
    pairs = []
    for a in seq:
        a = clip_reference(a)
        if a < 0:
            if v <= cf.V_H:
                break
            a = -min(-a, (v - cf.V_H) / T)
        seg = cf.Track(p, v, a)
        pairs.append((seg, pred.label(k, j, seg)))
        p, v, j = seg.p1, seg.v1, j + 1
    return pairs, p, v, j


def generate_prefixes(pred, k, init, seq, fallback):
    """Candidates for a sequence of H references, or seq = None: prefixes of
    8, 6, 4 and 2 periods with ambitions 1, 3/4, 1/2 and 1/4, each continued
    by the recovery of Section 5.1.6, and the recovery alone."""
    if seq is not None:
        seq = [exact(a) for a in seq]
        if not seq or any(a is None for a in seq):
            seq = None
    cands = []
    if init[0] == "rest":
        stay = cf.Plan(k + 1, [], cf.rest_containing(init))
        if seq is not None and seq[0] > 0 and init[1] == init[2]:
            p0 = init[1]
            for m, amb in PREFIX:
                head = cf.Launch(p0, min(seq[0], cf.U_MAX))
                pairs = [(head, pred.label(k, k + 1, head))]
                more, p, v, j = follow(pred, k, k + 2, head.p1, head.v1, seq[1:m])
                pairs += more
                tail, rest = cf.recovery(pred, k, j, p, v)
                if rest is None:
                    rest = cf.rest_containing(pairs[-1][0].end())
                cands.append((cf.Plan(k + 1, pairs + tail, rest), amb))
            cands.append((stay, Fr(0)))
        else:
            cands.append((stay, Fr(1) if (seq is not None and seq[0] <= 0) else Fr(0)))
        return cands
    pc, vc = centre(init)
    if seq is not None:
        if seq[0] <= 0 and vc <= cf.V_H:
            cands.append((cf.Plan(k + 1, [], cf.rest_containing(init)), Fr(1)))
        else:
            for m, amb in PREFIX:
                pairs, p, v, j = follow(pred, k, k + 1, pc, vc, seq[:m])
                tail, rest = cf.recovery(pred, k, j, p, v)
                if rest is None:
                    rest = cf.rest_containing(pairs[-1][0].end() if pairs else init)
                cands.append((cf.Plan(k + 1, pairs + tail, rest), amb))
    tail, rest = cf.recovery(pred, k, k + 1, pc, vc)
    if rest is None:
        rest = cf.rest_containing(init)
    cands.append((cf.Plan(k + 1, tail, rest), Fr(0)))
    return cands


# ----------------------------------------------------------------------------
# One governed episode
# ----------------------------------------------------------------------------
def initial_set(seg_ex, p, v):
    """I_{k+1}: the default of Chapter 4, except after a rest segment that
    holds the cart at standstill. Then the cart stays at the measured
    position, which is exact, so the standstill state alone is an admissible
    initial set (Definition 4.15), and a launch may start from it. Without
    this restart, a proposal that stops the cart short of the work zone would
    end the task (Table 5.3). The standstill state must lie in the funnel of
    the rest segment, which the test checks exactly."""
    if seg_ex.kind == "rest" and v == 0.0 and seg_ex.p_lo <= Fr(p) <= seg_ex.p_hi:
        x = Fr(p)
        return ("rest", x, x)
    return seg_ex.end()


POST = 50                         # periods allowed after t_limit to come to rest


def episode(pm, world, train=False, miss=0.0, seed=0, t_limit=T_LIMIT):
    """Runs the governed loop of Chapter 5 with the proposal method pm. The
    proposal formed from the observation at t_k serves the candidates of
    period k; the plan committed for period k+1 answers it. miss is the
    probability that a proposal misses its deadline. The task is to reach the
    work zone by t_limit; the run then continues until the cart is at rest,
    at most until t_limit + POST periods. After arrival, the proposal method keeps
    proposing, without exploring, and nothing enters its records. The run
    stops at an alarm, as in Chapter 5."""
    world.install()
    pred = cf.Predictor("forecasting")
    miss_rng = random.Random(10 ** 6 + seed)
    pm.reset(train)
    gen = generate_reference if pm.kind == "reference" else generate_prefixes
    p, v = 0.0, 0.0
    plan = cf.Plan(0, [], cf.Rest(Fr(0), Fr(0)))
    prof = plan.profile(("rest", Fr(0), Fr(0)))
    alpha_prev, alarm = Fr(1), False
    rows, arrived = [], None
    n = int(t_limit / T)
    for k in range(n + POST):
        if arrived is None and k >= n:
            break                                   # not arrived in time
        seg_ex = plan.head()
        init = initial_set(seg_ex, p, v)
        obs = dict(k=k, t=float(k * T), p=p, v=v, level=prof[0], alpha_prev=alpha_prev,
                   alarm=alarm, d_low=low_braking_distance(pred, k, p),
                   # reported at t_k with alpha_{k-1}: the head of P_k
                   kind_now=seg_ex.kind,
                   a_ex_now=float(seg_ex.a) if seg_ex.kind in ("track", "launch") else None)
        if arrived is None:
            q = pm.propose(obs)
        else:                                       # after arrival: no exploring
            training = getattr(pm, "train", False)
            pm.train = False
            q = pm.propose(obs)
            pm.train = training
        if miss and miss_rng.random() < miss:
            q = None
        fallback = plan.shift()
        fb_prof = prof[1:] if len(prof) > 1 else prof
        cands = gen(pred, k, init, q, fallback)
        profiles = [c.profile(init) for c, _ in cands]
        sel = cf.select(cands, profiles, fb_prof)
        if sel == 0:
            new_plan, new_prof, alpha = fallback, fb_prof, Fr(0)
        else:
            new_plan, new_prof = cands[sel - 1][0], profiles[sel - 1]
            alpha = cands[sel - 1][1]
        head = new_plan.head()
        p1, v1, st = cf.simulate(seg_ex, p, v, float(k * T), world.w)
        alarm = not cf.inside(init, p1, v1)
        done = arrived is None and p1 >= WORK
        row = dict(k=k, t=float(k * T), p=p, v=v, d_low=obs["d_low"], q=q, alpha=alpha,
                   sel=sel, a_now=getattr(seg_ex, "a", None),
                   a_next=head.a if head.kind in ("track", "launch") else None,
                   kind_next=head.kind, alarm=alarm, out=st["out"], pmax=st["pmax"],
                   p1=p1, v1=v1, done=done, after=arrived is not None,
                   unchanged=(alpha == 1 and realizable(q, init)
                              if pm.kind == "reference" else None))
        rows.append(row)
        if arrived is None:
            pm.record(row)
        p, v = p1, v1
        plan, prof, alpha_prev = new_plan, new_prof, alpha
        if done:
            arrived = k + 1
        if alarm or (arrived is not None and v1 == 0.0):
            break                                   # an alarm, or at rest
    pm.end_episode()
    restore_world()
    return rows, arrived


def run_metrics(rows, arrived):
    """Arrival time; mean intervention, 1 - alpha_k averaged exactly over the
    periods up to arrival; fallback commits before arrival; alarms, steps
    outside a funnel and the largest position over the whole run; and the
    position at rest after arrival (None if the cart is not at rest)."""
    pre = rows if arrived is None else rows[:arrived]
    load = Fr(sum(1 - Fr(x["alpha"]) for x in pre), len(pre))
    rest = rows[-1]["p1"] if arrived is not None and rows[-1]["v1"] == 0.0 else None
    return dict(arrival=None if arrived is None else arrived / 10, periods=arrived,
                load=float(load), load_exact=load,
                fallbacks=sum(1 for x in pre if x["sel"] == 0),
                alarms=sum(1 for x in rows if x["alarm"]),
                out=sum(x["out"] for x in rows),
                pmax=max(x["pmax"] for x in rows), rest=rest, n_pre=len(pre),
                full=sum(1 for x in pre if x["alpha"] == 1),
                unchanged=(None if pre[0]["unchanged"] is None
                           else sum(1 for x in pre if x["unchanged"])),
                at_max=(None if pre[0]["unchanged"] is None
                        else sum(1 for x in pre if x["q"] == DECLARED["a_max"])))


# ----------------------------------------------------------------------------
# Proposal methods
# ----------------------------------------------------------------------------
class ProposalMethod:
    """The replaceable module: a stated meaning and horizon, a proposal for
    each observation, and whatever it learns from the records it keeps."""

    kind = "reference"
    name = "proposal method"
    meaning = "reference acceleration for the next period [m/s^2]"
    horizon = "one period"

    def reset(self, train):
        self.train = train

    def propose(self, obs):
        raise NotImplementedError

    def record(self, row):
        pass

    def end_episode(self):
        pass


class Rule(ProposalMethod):
    """The proposal method of Chapters 3 and 5, in numbers: the largest
    acceleration, 2 m/s^2, until arrival ('accelerate'), then -2 ('hold')."""

    name = "rule"

    def propose(self, obs):
        return Fr(2) if obs["p"] < WORK else Fr(-2)


class BrakingCurve(ProposalMethod):
    """A hand-designed proposal method: track a speed 0.3 m/s below the
    feature v_env with gain 3 /s, braking at most as the labels allow."""

    name = "braking curve"

    def __init__(self, offset=0.3, gain=3.0):
        self.offset, self.gain = offset, gain

    def propose(self, obs):
        target = v_env(obs["p"], obs["d_low"]) - self.offset
        c = DECLARED
        lo = -c["b_low"] if obs["d_low"] == 0 else -c["b_firm"]
        return grid(max(lo, min(c["a_max"], self.gain * (target - obs["v"]))))


def actions():
    """The eight requests of the off-policy learner: the declared largest
    braking on firm ground and on the ramp, the declared largest
    acceleration, and five accelerations in between."""
    c = DECLARED
    return [-c["b_firm"], Fr(-1), -c["b_low"], Fr(0), Fr(1, 2), Fr(1), Fr(3, 2),
            c["a_max"]]
EDGES = [-0.3, -0.1, 0.0, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5]


class OffPolicyLearner(ProposalMethod):
    """A tabular off-policy learner with replay, the CPU stand-in for SAC.

    Observation features: the margin v_env - v in ten bins, whether the cart
    is on the low-braking region, and whether the previous proposal was
    committed in full. Actions: eight reference accelerations. Reward of the
    proposal q_k: the executed reference acceleration of period k+1 divided
    by the declared largest acceleration, minus chi times the intervention
    1 - alpha_k. Records: the
    proposal is the action (log='proposal'), or the executed reference
    rounded to the nearest action (log='executed', Section 6.3). Section 6.3
    also uses two observation variants: blind=True ignores the label summary
    (the features assume firm ground everywhere), and delay_aware=True
    computes the features at the state predicted for t_{k+1}, where the
    proposal takes effect, from the executed reference of period k that the
    governor reported at t_k; the other learners use that reference only in
    their records. The pairs (features, proposal, ambition) of the
    training periods are kept in self.outcomes for the diagnostics of
    Section 6.3; they do not enter learning."""

    name = "off-policy learner"

    def __init__(self, seed, chi=0.5, gamma=0.5, lr=0.3, sweeps=10, eps=0.2,
                 log="proposal", blind=False, delay_aware=False):
        self.rng = random.Random(seed)
        self.chi, self.gamma, self.lr, self.sweeps, self.eps = chi, gamma, lr, sweeps, eps
        self.log, self.blind, self.delay_aware = log, blind, delay_aware
        self.Q, self.buffer, self.pending, self.outcomes = {}, [], None, []
        self.acts = actions()

    def features(self, obs):
        p, v, d = obs["p"], obs["v"], obs["d_low"]
        if self.delay_aware:
            p, v, d = predict_next(obs)
        if self.blind:
            d = None
        m = v_env(p, d) - v
        return (sum(1 for e in EDGES if m >= e), 1 if d == 0 else 0,
                1 if obs["alpha_prev"] == 1 else 0)

    def values(self, s):
        if s not in self.Q:
            self.Q[s] = [0.0] * len(self.acts)
        return self.Q[s]

    def greedy(self, s):
        qs = self.values(s)
        best = max(qs)
        return max(i for i, x in enumerate(qs) if x == best)   # ties: faster

    def reset(self, train):
        self.train, self.pending = train, None

    def propose(self, obs):
        s = self.features(obs)
        if self.train and self.pending is not None:
            self.pending[3] = s
            self.buffer.append(tuple(self.pending))
            self.pending = None
        if self.train and self.rng.random() < self.eps:
            i = int(self.rng.random() * len(self.acts))
        else:
            i = self.greedy(s)
        self.current = (s, i)
        return self.acts[i]

    def record(self, row):
        if not self.train:
            return
        s, i = self.current
        self.outcomes.append((s, i, row["alpha"]))
        a_next = float(row["a_next"]) if row["a_next"] is not None else 0.0
        r = a_next / float(DECLARED["a_max"]) - self.chi * float(1 - row["alpha"])
        if self.log == "executed" and row["a_next"] is not None:
            i = min(range(len(self.acts)), key=lambda j: abs(self.acts[j] - row["a_next"]))
        if row["done"]:
            self.buffer.append((s, i, r, None, True))
        else:
            self.pending = [s, i, r, None, False]

    def end_episode(self):
        if not self.train:
            return
        self.pending = None                     # truncated: no successor
        data = self.buffer
        for _ in range(self.sweeps):
            for _j in range(len(data)):
                s, i, r, s2, done = data[int(self.rng.random() * len(data))]
                target = r if done else r + self.gamma * max(self.values(s2))
                qs = self.values(s)
                qs[i] += self.lr * (target - qs[i])


class Planner(ProposalMethod):
    """A planner with a learned model, the CPU stand-in for TD-MPC2.

    Model: dv = th0 a and dp = th1 v + th2 a per period, fitted by least
    squares to the executed references and the measured states (plant
    data). Plan: among sequences that accelerate at a in {a_max, 1.5, 1,
    0.5, 0} for m periods and then brake at the declared largest braking for
    the labels, the one that
    the model takes furthest while it ends at most zeta v_env and before
    24.45 m. The factor zeta falls by 0.005 after each intervention, to at
    least 0.85, and rises by 0.001 otherwise, to at most 1."""

    kind = "sequence"
    name = "planner"
    meaning = "reference accelerations for the next eight periods [m/s^2]"
    horizon = "eight periods; prefixes offered"

    def __init__(self, theta=None, zeta=1.0, adapt=True):
        self.fixed = theta is not None
        self.theta = theta if theta else (0.05, 0.1, 0.0)
        self.zeta, self.adapt, self.data = zeta, adapt, []

    def fit(self):
        if self.fixed or len(self.data) < 5:
            return
        saa = total(a * a for v, a, dv, dp in self.data)
        sav = total(a * dv for v, a, dv, dp in self.data)
        svv = total(v * v for v, a, dv, dp in self.data)
        sva = total(v * a for v, a, dv, dp in self.data)
        spv = total(v * dp for v, a, dv, dp in self.data)
        spa = total(a * dp for v, a, dv, dp in self.data)
        det = svv * saa - sva * sva
        if saa > 0 and abs(det) > 1e-12:
            self.theta = (sav / saa, (spv * saa - spa * sva) / det,
                          (svv * spa - sva * spv) / det)

    def plan(self, p, v, d):
        th0, th1, th2 = self.theta
        ramp = None if d is None else p + d
        best, best_p = None, -math.inf
        c = DECLARED
        b1, b2 = float(c["b_firm"]), float(c["b_low"])
        for a_acc in (float(c["a_max"]), 1.5, 1.0, 0.5, 0.0):
            for m in range(H + 1):
                seq, pp, vv = [], p, v
                for i in range(H):
                    on_low = ramp is not None and pp >= ramp - 0.1
                    a = a_acc if i < m else (-b2 if on_low else -b1)
                    if a < 0 and vv <= 0.06:
                        a = 0.0
                    seq.append(a)
                    pp, vv = pp + th1 * vv + th2 * a, vv + th0 * a
                dl = None if ramp is None else max(0.0, ramp - pp)
                if vv <= self.zeta * v_env(pp, dl) and pp <= 24.45 and pp > best_p:
                    best, best_p = seq, pp
        if best is None:
            best = [-b2 if d == 0 else -b1] * H
        return [grid(a) for a in best]

    def propose(self, obs):
        return self.plan(obs["p"], obs["v"], obs["d_low"])

    def record(self, row):
        if not self.train:                      # frozen in evaluation
            return
        if row["a_now"] is not None and row["v"] > 0.2:
            self.data.append((row["v"], float(row["a_now"]),
                              row["v1"] - row["v"], row["p1"] - row["p"]))
        if self.adapt:
            if row["alpha"] < 1:
                self.zeta = max(0.85, self.zeta - 0.005)
            else:
                self.zeta = min(1.0, self.zeta + 0.001)

    def end_episode(self):
        if self.train:
            self.fit()


NOMINAL_THETA = (0.1, 0.1, 0.005)  # nominal constant-acceleration kinematics: T, T, T^2/2


def expert():
    """The demonstrator: the planner with the nominal model in place of the
    learned one and zeta fixed at 0.95. The measured motion departs from the
    nominal model through the feedback and the disturbance, and the planner
    searches only a small family of sequences, so the expert is a
    benchmark, not an optimal policy."""
    pl = Planner(theta=NOMINAL_THETA, zeta=0.95, adapt=False)
    pl.name = "expert planner"
    return pl


def chunk_features(obs):
    d = obs["d_low"]
    return ((v_env(obs["p"], d) - obs["v"]) / 2.0, obs["v"] / 6.0,
            1.0 if d == 0 else 0.0, max(0.0, WORK - obs["p"]) / 6.0)


class ChunkImitator(ProposalMethod):
    """A nearest-neighbour chunk policy, the CPU stand-in for Diffusion
    Policy. It infers a chunk of H references from the nearest demonstrated
    observation, proposes its remainder in the following periods, and infers
    again after four periods or after an intervention. An inference takes
    'latency' periods; meanwhile the old chunk is served, and when it is
    used up no proposal is made. The new chunk is then proposed from its
    first reference, without skipping the periods that have passed since its
    observation."""

    kind = "chunk"
    name = "chunk imitator"
    meaning = "reference accelerations for the next eight periods [m/s^2]"
    horizon = "a chunk of eight periods, kept for up to four"

    def __init__(self, n_act=4, latency=0):
        self.D, self.n_act, self.latency = [], n_act, latency

    def infer(self, obs):
        f = chunk_features(obs)
        best = min(self.D, key=lambda x: total((a - b) * (a - b) for a, b in zip(x[0], f)))
        return list(best[1])

    def reset(self, train):
        self.train = train
        self.chunk, self.age, self.busy, self.pending = None, 0, 0, None

    def propose(self, obs):
        need = self.chunk is None or self.age >= self.n_act or obs["alpha_prev"] < 1
        if self.busy > 0:
            self.busy -= 1
            if self.busy == 0:
                self.chunk, self.age = self.pending, 0
        elif need:
            self.pending = self.infer(obs)
            if self.latency == 0:
                self.chunk, self.age = self.pending, 0
            else:
                self.busy = self.latency
        if self.chunk is None or self.age >= H:
            return None
        rest = self.chunk[self.age:]
        self.age += 1
        return rest + [rest[-1]] * (H - len(rest))


class Demonstration(ProposalMethod):
    """Records demonstrations through the governor. With 'act' the expert,
    the expert acts (behaviour cloning); with 'act' the imitator, the
    imitator acts and the expert is asked for its proposal at the states the
    imitator visits (dataset aggregation). Each record keeps the observation
    features, the demonstrated chunk and the executed reference; in the
    expert's episodes, the committed ambition is kept as well."""

    def __init__(self, teacher, act):
        self.teacher, self.act, self.kind = teacher, act, act.kind
        self.data, self.executed, self.ambitions = [], [], []
        self.pending = None

    def reset(self, train):
        self.act.reset(False)
        if self.teacher is not self.act:
            self.teacher.reset(False)

    def propose(self, obs):
        label = self.teacher.propose(obs)
        self.pending = (chunk_features(obs), label)
        return label if self.act is self.teacher else self.act.propose(obs)

    def record(self, row):
        self.data.append(self.pending)
        a = row["a_next"]
        self.executed.append(None if a is None else float(a))
        if self.act is self.teacher:
            self.ambitions.append(row["alpha"])


# ----------------------------------------------------------------------------
# The protocol
# ----------------------------------------------------------------------------
EVAL_SEEDS = list(range(1000, 1020))
TRAIN_EPISODES = 40
DEMOS = 5                       # expert episodes, then as many aggregation episodes
SEEDS = (1, 2, 3)


def training_worlds(seed, n):
    return [World(100 * seed + i) for i in range(n)]


def evaluate(pm, seeds, miss=0.0):
    out = []
    for s in seeds:
        m = run_metrics(*episode(pm, World(s), miss=miss, seed=s))
        m["world"] = s
        out.append(m)
    return out


def train(pm, worlds, log):
    for i, w in enumerate(worlds):
        m = run_metrics(*episode(pm, w, train=True))
        m.update(episode=i, world=w.seed)
        log.append(m)
    return pm


def aliasing(outcomes):
    """Pairs (features, proposal) recorded at least twice in training, and
    those among them after which the governor committed more than one
    ambition. Given the full state, a period's outcome is determined, so a
    pair with several outcomes shows state that the features do not hold."""
    seen = {}
    for s, i, a in outcomes:
        seen.setdefault((s, i), set()).add(a)
    counts = {}
    for s, i, a in outcomes:
        counts[(s, i)] = counts.get((s, i), 0) + 1
    twice = [key for key, n in counts.items() if n >= 2]
    return dict(pairs=len(twice), mixed=sum(1 for key in twice if len(seen[key]) > 1),
                periods=len(outcomes),
                periods_in_mixed=sum(counts[key] for key in twice if len(seen[key]) > 1))


def value_diagnostics(pm):
    """The feature states that a trained off-policy learner recorded in
    training, and those among them whose greedy action never appears as the
    recorded action in that state, so that its value is still the initial
    value 0. States first met in evaluation are not counted."""
    recorded = {}
    for s, i, r, s2, done in pm.buffer:
        recorded.setdefault(s, set()).add(i)
    return dict(states=len(recorded),
                greedy_never_recorded=sum(1 for s in recorded
                                          if pm.greedy(s) not in recorded[s]))


VARIANTS = {"off-policy learner": {},
            "off-policy learner, missed deadlines": {},
            "off-policy learner, executed records": dict(log="executed"),
            "off-policy learner, label-blind": dict(blind=True),
            "off-policy learner, delay-aware": dict(delay_aware=True)}
SECTION_6_3 = ("off-policy learner, executed records", "off-policy learner, label-blind",
               "off-policy learner, delay-aware")


def job(spec):
    """One independent unit of the protocol; returns its records."""
    name, seed, package, cfg = spec
    use_package(package)
    t0 = time.time()
    log, ev, qtable, alias, values = [], [], None, None, None
    if name == "rule":
        ev = evaluate(Rule(), cfg["eval"])
    elif name == "braking curve":
        ev = evaluate(BrakingCurve(), cfg["eval"])
    elif name == "expert planner":
        ev = evaluate(expert(), cfg["eval"])
    elif name in VARIANTS:
        pm = train(OffPolicyLearner(seed, **VARIANTS[name]),
                   training_worlds(seed, cfg["train"]), log)
        ev = evaluate(pm, cfg["eval"], miss=0.3 if "missed" in name else 0.0)
        qtable, alias, values = pm.Q, aliasing(pm.outcomes), value_diagnostics(pm)
    elif name == "planner":
        pm = train(Planner(), training_worlds(seed, cfg["train"]), log)
        ev = evaluate(pm, cfg["eval"])
    elif name in ("chunk imitator", "chunk imitator, latency 2"):
        im = ChunkImitator(latency=2 if "latency" in name else 0)
        teacher = expert()
        worlds = [World(100 * seed + 50 + i) for i in range(2 * cfg["demos"])]
        rec = Demonstration(teacher, teacher)
        for w in worlds[:cfg["demos"]]:
            log.append(dict(run_metrics(*episode(rec, w)), episode=len(log), world=w.seed,
                            phase="demonstration"))
        im.D = list(rec.data)
        agg = Demonstration(teacher, ChunkImitator())
        agg.act.D = im.D
        for w in worlds[cfg["demos"]:]:
            log.append(dict(run_metrics(*episode(agg, w)), episode=len(log), world=w.seed,
                            phase="aggregation"))
        im.D = im.D + agg.data
        ev = evaluate(im, cfg["eval"])
        log.append(dict(demonstrated=len(rec.data), aggregated=len(agg.data),
                        partial=sum(1 for a in rec.ambitions if a < 1),
                        none=sum(1 for a in rec.ambitions if a == 0),
                        periods=len(rec.ambitions), phase="data"))
    use_package("tube")
    return dict(name=name, seed=seed, package=package, log=log, eval=ev, qtable=qtable,
                aliasing=alias, values=values, seconds=round(time.time() - t0, 1))


def trace_chapter5_world(learner):
    """The rule and the trained off-policy learner in the world of Chapter
    5. The rule reproduces the forecasting run of Section 5.2.5: the same
    measured states and ambitions in every period until the cart is at
    rest."""
    out = {}
    for pm in (Rule(), learner):
        rows, arrived = episode(pm, World.chapter5())
        out[pm.name] = (rows, arrived)
    ref, _, _ = cf.run("forecasting")
    rows = out["rule"][0]
    body = [x for x in ref if not x.get("final")]
    same = len(body) >= len(rows) and all(
        a["p"] == b["p"] and a["v"] == b["v"] and a["alpha"] == b["alpha"]
        for a, b in zip(rows, body))
    return out, same


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def describe(evals):
    """Summary of the evaluation runs of one policy (several seeds pooled)."""
    arr = [e["periods"] for e in evals if e["periods"] is not None]
    return dict(arrived=f"{len(arr)} of {len(evals)}",
                arrival=f"{cf.r(Fr(sum(arr), 10 * len(arr)), 2)} s" if arr else "none",
                arrival_range=(f"{min(arr) / 10:.1f} to {max(arr) / 10:.1f} s"
                               if arr else "none"),
                mean_intervention=cf.r(sum(e["load_exact"] for e in evals) / len(evals), 3),
                fallbacks=cf.r(Fr(sum(e["fallbacks"] for e in evals), len(evals)), 2),
                alarms=str(sum(e["alarms"] for e in evals)),
                steps_outside_funnel=str(sum(e["out"] for e in evals)),
                max_position=f"{max(e['pmax'] for e in evals):.3f} m",
                at_rest_after_arrival=(f"{sum(1 for e in evals if e['rest'] is not None)} of "
                                       f"{len(arr)}"),
                committed_in_full=percent(sum(e["full"] for e in evals),
                                          sum(e["n_pre"] for e in evals)),
                passed_unchanged=("not applicable" if evals[0]["unchanged"] is None
                                  else percent(sum(e["unchanged"] for e in evals),
                                               sum(e["n_pre"] for e in evals))),
                passed_unchanged_count=(
                    "not applicable" if evals[0]["unchanged"] is None
                    else f"{sum(e['unchanged'] for e in evals)} of "
                         f"{sum(e['n_pre'] for e in evals)}"),
                proposals_at_largest_acceleration=(
                    "not applicable" if evals[0]["at_max"] is None
                    else percent(sum(e["at_max"] for e in evals),
                                 sum(e["n_pre"] for e in evals))))


def percent(n, d):
    """n/d as a percentage, exact and rounded half up to one decimal."""
    return f"{cf.r(Fr(100 * n, d), 1)} %"


TUBE_ROWS = ("rule", "braking curve", "expert planner", "off-policy learner", "planner",
             "chunk imitator")

EXPECTED = {   # every number quoted or summarized in Chapter 6, as in the text
    'rule': {
        'arrived': '20 of 20',
        'arrival': '9.55 s',
        'arrival_range': '9.1 to 10.0 s',
        'mean_intervention': '0.656',
        'alarms': '0',
        'steps_outside_funnel': '0',
        'at_rest_after_arrival': '20 of 20',
        'largest_position_all': '24.474 m',
        'committed_in_full': '33.6 %',
        'passed_unchanged': '1.0 %',
        'passed_unchanged_count': '20 of 1909',
    },
    'braking curve': {
        'arrived': '20 of 20',
        'arrival': '9.56 s',
        'arrival_range': '9.0 to 10.0 s',
        'mean_intervention': '0.246',
        'alarms': '0',
        'steps_outside_funnel': '0',
        'at_rest_after_arrival': '20 of 20',
        'largest_position_all': '24.473 m',
        'passed_unchanged': '74.3 %',
    },
    'expert planner': {
        'arrived': '20 of 20',
        'arrival': '9.66 s',
        'arrival_range': '9.2 to 10.1 s',
        'mean_intervention': '0.203',
        'alarms': '0',
        'steps_outside_funnel': '0',
        'at_rest_after_arrival': '20 of 20',
        'largest_position_all': '24.471 m',
    },
    'off-policy learner': {
        'arrived': '60 of 60',
        'arrival': '9.65 s',
        'arrival_range': '9.1 to 10.2 s',
        'mean_intervention': '0.041',
        'alarms': '0',
        'steps_outside_funnel': '0',
        'at_rest_after_arrival': '60 of 60',
        'largest_position_all': '24.474 m',
        'training_episodes': '120',
        'training_arrived': '120 of 120',
        'training_alarms': '0',
        'training_steps_outside_funnel': '0',
        'training_at_rest_after_arrival': '120 of 120',
        'passed_unchanged': '94.8 %',
        'training_intervention_first_10': '0.352',
        'training_intervention_last_10': '0.127',
        'periods_in_such_pairs': '22.3 %',
        'proposals_at_largest_acceleration': '34.3 %',
        'pairs_recorded_twice': '319',
        'pairs_with_several_ambitions': '109',
        'mean_intervention_by_seed': '0.037, 0.043, 0.043',
        'feature_states_recorded': '54',
        'states_with_greedy_action_never_recorded': '5',
    },
    'planner': {
        'arrived': '60 of 60',
        'arrival': '9.72 s',
        'arrival_range': '9.1 to 10.3 s',
        'mean_intervention': '0.138',
        'alarms': '0',
        'steps_outside_funnel': '0',
        'at_rest_after_arrival': '60 of 60',
        'largest_position_all': '24.474 m',
        'training_episodes': '120',
        'training_arrived': '120 of 120',
        'training_alarms': '0',
        'training_steps_outside_funnel': '0',
        'training_at_rest_after_arrival': '120 of 120',
        'committed_in_full': '84.6 %',
    },
    'chunk imitator': {
        'arrived': '60 of 60',
        'arrival': '9.75 s',
        'arrival_range': '9.1 to 10.3 s',
        'mean_intervention': '0.139',
        'alarms': '0',
        'steps_outside_funnel': '0',
        'at_rest_after_arrival': '60 of 60',
        'largest_position_all': '24.475 m',
        'training_episodes': '30',
        'training_arrived': '29 of 30',
        'training_alarms': '0',
        'training_steps_outside_funnel': '0',
        'training_at_rest_after_arrival': '29 of 29',
        'committed_in_full': '82.8 %',
        'demonstrated_states': '1428',
        'demonstrated_chunks_not_committed_in_full': '342 of 1428',
        'demonstrated_chunks_not_committed_at_all': '219 of 1428',
    },
    'off-policy learner with ccm': {
        'arrived': '60 of 60',
        'arrival': '9.40 s',
        'arrival_range': '8.9 to 9.9 s',
        'mean_intervention': '0.076',
        'alarms': '0',
        'steps_outside_funnel': '0',
        'at_rest_after_arrival': '60 of 60',
        'largest_position_all': '24.484 m',
        'training_episodes': '120',
        'training_arrived': '120 of 120',
        'training_alarms': '0',
        'training_steps_outside_funnel': '0',
        'training_at_rest_after_arrival': '120 of 120',
    },
    'off-policy learner, missed deadlines': {
        'arrived': '20 of 20',
        'arrival': '11.62 s',
        'arrival_range': '10.2 to 17.4 s',
        'mean_intervention': '0.310',
        'alarms': '0',
        'steps_outside_funnel': '0',
        'at_rest_after_arrival': '20 of 20',
        'largest_position_all': '24.474 m',
    },
    'chunk imitator, latency 2': {
        'arrived': '20 of 20',
        'arrival': '10.01 s',
        'arrival_range': '9.5 to 10.4 s',
        'mean_intervention': '0.187',
        'alarms': '0',
        'steps_outside_funnel': '0',
        'at_rest_after_arrival': '20 of 20',
        'largest_position_all': '24.474 m',
    },
    'off-policy learner, executed records': {
        'arrived': '60 of 60',
        'arrival': '9.54 s',
        'arrival_range': '9.0 to 10.0 s',
        'mean_intervention': '0.651',
        'alarms': '0',
        'steps_outside_funnel': '0',
        'at_rest_after_arrival': '60 of 60',
        'largest_position_all': '24.476 m',
        'training_episodes': '120',
        'training_arrived': '120 of 120',
        'training_alarms': '0',
        'training_steps_outside_funnel': '0',
        'training_at_rest_after_arrival': '120 of 120',
        'passed_unchanged': '33.8 %',
        'training_intervention_first_10': '0.603',
        'training_intervention_last_10': '0.599',
        'periods_in_such_pairs': '61.1 %',
        'feature_states_recorded': '49',
        'states_with_greedy_action_never_recorded': '34',
        'proposals_at_largest_acceleration': '95.6 %',
    },
    'off-policy learner, label-blind': {
        'arrived': '60 of 60',
        'arrival': '9.69 s',
        'arrival_range': '9.1 to 10.3 s',
        'mean_intervention': '0.355',
        'alarms': '0',
        'steps_outside_funnel': '0',
        'at_rest_after_arrival': '60 of 60',
        'largest_position_all': '24.475 m',
        'training_episodes': '120',
        'training_arrived': '120 of 120',
        'training_alarms': '0',
        'training_steps_outside_funnel': '0',
        'training_at_rest_after_arrival': '120 of 120',
        'passed_unchanged': '63.7 %',
        'training_intervention_first_10': '0.456',
        'training_intervention_last_10': '0.375',
        'periods_in_such_pairs': '92.9 %',
    },
    'off-policy learner, delay-aware': {
        'arrived': '60 of 60',
        'arrival': '9.62 s',
        'arrival_range': '9.2 to 10.1 s',
        'mean_intervention': '0.033',
        'alarms': '0',
        'steps_outside_funnel': '0',
        'at_rest_after_arrival': '60 of 60',
        'largest_position_all': '24.476 m',
        'training_episodes': '120',
        'training_arrived': '120 of 120',
        'training_alarms': '0',
        'training_steps_outside_funnel': '0',
        'training_at_rest_after_arrival': '120 of 120',
        'passed_unchanged': '95.9 %',
        'training_intervention_first_10': '0.357',
        'training_intervention_last_10': '0.153',
        'periods_in_such_pairs': '24.2 %',
        'mean_intervention_by_seed': '0.045, 0.036, 0.018',
    },
    'all runs of Section 6.3': {
        'evaluation_runs': '180',
        'training_episodes': '360',
        'alarms': '0',
        'steps_outside_funnel': '0',
        'arrived': '540 of 540',
        'at_rest_after_arrival': '540 of 540',
        'largest_position': '24.476 m',
    },
    'all runs': {
        'evaluation_runs': '340',
        'training_episodes': '390',
        'arrived': '729 of 730',
        'alarms': '0',
        'steps_outside_funnel': '0',
        'at_rest_after_arrival': '729 of 729',
        'largest_position': '24.484 m',
        'handover_position': '24.47 m',
        'not_arrived': 'chunk imitator, seed 2, aggregation, stopped at 23.61 m',
    },
    'comparison': {
        'arrival_spread_tube': '9.55 to 9.75 s',
        'learner_later_than_rule': '0.10 s',
        'imitator_later_than_expert': '0.09 s',
        'ccm_earlier_than_tube': '0.25 s',
        'missed_deadlines_later_by': '1.92 s',
        'latency_later_by': '0.27 s',
        'executed_records_arrival_change': '-0.11 s',
        'delay_aware_arrival_change': '-0.03 s',
    },
    'trace rule': {
        'arrival': '9.2 s',
        'mean_intervention': '0.636',
        'periods_committed_in_full': '33',
        'periods': '92',
        'periods_not_committed_in_full': '59',
        'rest_position': '24.468 m',
        'k0': 'q 2.000, alpha 1, executed 2.0000',
        'k30': 'q 2.000, alpha 1, executed 1.8750',
        'k31': 'q 2.000, alpha 1, executed 1.8750',
        'k32': 'q 2.000, alpha 1, executed 1.8750',
        'k33': 'q 2.000, alpha 1/4, executed -0.9375',
        'k34': 'q 2.000, alpha 0, executed -1.8750',
        'k35': 'q 2.000, alpha 0, executed -1.8750',
        'k36': 'q 2.000, alpha 0, executed -1.8750',
        'partial_from_3.4_s': '3.4 s at 1/4, 6.5 s at 1/4',
        'same_as_chapter_5_forecasting_run': 'yes',
    },
    'trace off-policy learner': {
        'arrival': '9.4 s',
        'mean_intervention': '0.035',
        'periods_committed_in_full': '90',
        'periods': '94',
        'k30': 'q 1.875, alpha 1, executed 1.8750',
        'k31': 'q 1.875, alpha 1, executed 1.8750',
        'k32': 'q 1.500, alpha 1, executed 1.5000',
        'k33': 'q -1.875, alpha 1, executed -1.8750',
        'k34': 'q -1.875, alpha 1, executed -1.8750',
        'k35': 'q -1.875, alpha 1, executed -1.8750',
        'k36': 'q -1.875, alpha 1, executed -1.8750',
        'proposals_not_braking_from_3.4_s': '3',
        'later_than_rule': '0.2 s',
    },
}


def main():
    ap = argparse.ArgumentParser(description="Chapter 6: learned proposal methods")
    ap.add_argument("--check", action="store_true", help="check every quoted number")
    ap.add_argument("--jobs", type=int, default=min(2, os.cpu_count() or 1))
    ap.add_argument("--quick", action="store_true", help="smoke test with small budgets")
    args = ap.parse_args()
    out = os.path.join(HERE, "results")
    os.makedirs(out, exist_ok=True)
    cfg = dict(eval=EVAL_SEEDS, train=TRAIN_EPISODES, demos=DEMOS)
    seeds = SEEDS
    if args.quick:
        cfg, seeds = dict(eval=EVAL_SEEDS[:3], train=4, demos=1), (1,)
    specs = [("rule", 0, "tube", cfg), ("braking curve", 0, "tube", cfg),
             ("expert planner", 0, "tube", cfg)]
    specs += [(n, s, "tube", cfg) for n in ("off-policy learner", "planner", "chunk imitator")
              for s in seeds]
    specs += [("off-policy learner", s, "ccm", cfg) for s in seeds]
    specs += [("off-policy learner, missed deadlines", seeds[0], "tube", cfg),
              ("chunk imitator, latency 2", seeds[0], "tube", cfg)]
    specs += [(n, s, "tube", cfg) for n in SECTION_6_3 for s in seeds]
    t0 = time.time()
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            results = pool.map(job, specs)
    else:
        results = [job(s) for s in specs]
    print(f"[protocol] {len(specs)} jobs in {time.time() - t0:.0f} s")
    # the trained off-policy learner of the first seed, in the world of Chapter 5
    use_package("tube")
    learner = OffPolicyLearner(seeds[0])
    learner.Q = next(r["qtable"] for r in results if r["name"] == "off-policy learner"
                     and r["package"] == "tube" and r["seed"] == seeds[0])
    traces, same = trace_chapter5_world(learner)
    facts = report(results, traces, out)
    facts["trace rule"]["same_as_chapter_5_forecasting_run"] = "yes" if same else "no"
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
        lines.append(f"[check] {n} values quoted or summarized in Chapter 6: " +
                     ("all as quoted" if not failures else "FAILED: " + "; ".join(failures)))
    with open(os.path.join(out, "summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 1 if failures else 0


def report(results, traces, out):
    facts = {}
    by = {}
    for r in results:
        by.setdefault((r["name"], r["package"]), []).append(r)
    for (name, package), rs in by.items():
        evals = [e for r in rs for e in r["eval"]]
        key = name if package == "tube" else f"{name} with {package}"
        d = describe(evals)
        d["seeds"] = ",".join(str(r["seed"]) for r in rs)
        if len(rs) > 1:                          # each trained policy on its own
            d["mean_intervention_by_seed"] = ", ".join(
                cf.r(sum(e["load_exact"] for e in r["eval"]) / len(r["eval"]), 3) for r in rs)
        tr = [m for r in rs for m in r["log"] if "load" in m]
        if tr:
            d["training_episodes"] = str(len(tr))
            d["training_alarms"] = str(sum(m["alarms"] for m in tr))
            d["training_steps_outside_funnel"] = str(sum(m["out"] for m in tr))
            d["training_max_position"] = f"{max(m['pmax'] for m in tr):.3f} m"
            n_arr = sum(1 for m in tr if m["periods"] is not None)
            d["training_arrived"] = f"{n_arr} of {len(tr)}"
            d["training_at_rest_after_arrival"] = (
                f"{sum(1 for m in tr if m['rest'] is not None)} of {n_arr}")
        d["largest_position_all"] = (
            f"{max([e['pmax'] for e in evals] + [m['pmax'] for m in tr]):.3f} m")
        data = [m for r in rs for m in r["log"] if m.get("phase") == "data"]
        if data:
            d["demonstrated_states"] = str(sum(m["demonstrated"] for m in data))
            d["aggregated_states"] = str(sum(m["aggregated"] for m in data))
            n_dem = sum(m["periods"] for m in data)
            d["demonstrated_chunks_not_committed_in_full"] = (
                f"{sum(m['partial'] for m in data)} of {n_dem}")
            d["demonstrated_chunks_not_committed_at_all"] = (
                f"{sum(m['none'] for m in data)} of {n_dem}")
        facts[key] = d
    # learning progress: the mean intervention of the first and the last ten
    # training episodes of each seed, pooled over the seeds
    for (name, package), rs in by.items():
        tr = [m for r in rs for m in r["log"] if "load" in m and "episode" in m]
        first = [m["load_exact"] for m in tr if m["episode"] < 10]
        last = [m["load_exact"] for m in tr if m["episode"] >= TRAIN_EPISODES - 10]
        if first and last and "missed" not in name and "imitator" not in name:
            key = name if package == "tube" else f"{name} with {package}"
            facts[key]["training_intervention_first_10"] = cf.r(sum(first) / len(first), 3)
            facts[key]["training_intervention_last_10"] = cf.r(sum(last) / len(last), 3)
    # aliasing in the reduced observation of the off-policy learners
    for (name, package), rs in by.items():
        al = [r["aliasing"] for r in rs if r["aliasing"] is not None]
        if al and package == "tube" and "missed" not in name:
            key = name
            facts[key]["pairs_recorded_twice"] = str(sum(a["pairs"] for a in al))
            facts[key]["pairs_with_several_ambitions"] = str(sum(a["mixed"] for a in al))
            facts[key]["periods_in_such_pairs"] = percent(
                sum(a["periods_in_mixed"] for a in al), sum(a["periods"] for a in al))
            vd = [r["values"] for r in rs]
            facts[key]["feature_states_recorded"] = str(sum(v["states"] for v in vd))
            facts[key]["states_with_greedy_action_never_recorded"] = str(
                sum(v["greedy_never_recorded"] for v in vd))
    # the runs of Section 6.3 are counted separately
    sec63 = [r for r in results if r["name"] in SECTION_6_3]
    e63 = [e for r in sec63 for e in r["eval"]]
    t63 = [m for r in sec63 for m in r["log"] if "load" in m]
    facts["all runs of Section 6.3"] = dict(
        evaluation_runs=str(len(e63)), training_episodes=str(len(t63)),
        alarms=str(sum(e["alarms"] for e in e63 + t63)),
        steps_outside_funnel=str(sum(e["out"] for e in e63 + t63)),
        arrived=(f"{sum(1 for e in e63 + t63 if e['periods'] is not None)} of "
                 f"{len(e63 + t63)}"),
        at_rest_after_arrival=(
            f"{sum(1 for e in e63 + t63 if e['rest'] is not None)} of "
            f"{sum(1 for e in e63 + t63 if e['periods'] is not None)}"),
        largest_position=f"{max(e['pmax'] for e in e63 + t63):.3f} m")
    # all runs of Sections 6.1 and 6.2; the jobs with late proposals or a slow
    # inference repeat the training of seed 1, so their training episodes are
    # not new
    r62 = [r for r in results if r["name"] not in SECTION_6_3]
    evals = [e for r in r62 for e in r["eval"]]
    trains = [m for r in r62 for m in r["log"] if "load" in m]
    new = [m for r in r62 if "late" not in r["name"] and "latency" not in r["name"]
           and "missed" not in r["name"] for m in r["log"] if "load" in m]
    facts["all runs"] = dict(
        evaluation_runs=str(len(evals)), training_episodes=str(len(new)),
        alarms=str(sum(e["alarms"] for e in evals + trains)),
        steps_outside_funnel=str(sum(e["out"] for e in evals + trains)),
        arrived=(f"{sum(1 for e in evals + new if e['periods'] is not None)} of "
                 f"{len(evals + new)}"),
        at_rest_after_arrival=(
            f"{sum(1 for e in evals + new if e['rest'] is not None)} of "
            f"{sum(1 for e in evals + new if e['periods'] is not None)}"),
        largest_position=f"{max(e['pmax'] for e in evals + trains):.3f} m",
        handover_position=f"{float(DECLARED['x_h']):.2f} m",
        not_arrived="; ".join(
            f"{r['name']}, seed {r['seed']}, {phase}, stopped at {m['pmax']:.2f} m"
            for r in r62
            for phase, m in ([(m.get("phase", "training"), m) for m in r["log"]
                              if "missed" not in r["name"] and "latency" not in r["name"]]
                             + [("evaluation", m) for m in r["eval"]])
            if "load" in m and m["periods"] is None) or "none")

    def arrival_of(key):
        return Fr(facts[key]["arrival"].split(" ")[0])

    def seed_arrival(name, seed):
        """Mean arrival of one trained policy, rounded as quoted."""
        arr = [e["periods"] for r in results if r["name"] == name
               and r["package"] == "tube" and r["seed"] == seed
               for e in r["eval"] if e["periods"] is not None]
        return Fr(cf.r(Fr(sum(arr), 10 * len(arr)), 2))

    s1 = next(r["seed"] for r in results
              if r["name"] == "off-policy learner, missed deadlines")
    learner_1 = seed_arrival("off-policy learner", s1)
    missed_1 = seed_arrival("off-policy learner, missed deadlines", s1)
    imitator_1 = seed_arrival("chunk imitator", s1)
    latency_1 = seed_arrival("chunk imitator, latency 2", s1)

    facts["comparison"] = dict(
        arrival_spread_tube=(f"{float(min(arrival_of(k) for k in TUBE_ROWS)):.2f} to "
                             f"{float(max(arrival_of(k) for k in TUBE_ROWS)):.2f} s"),
        learner_later_than_rule=f"{float(arrival_of('off-policy learner') - arrival_of('rule')):.2f} s",
        imitator_later_than_expert=(
            f"{float(arrival_of('chunk imitator') - arrival_of('expert planner')):.2f} s"),
        ccm_earlier_than_tube=(
            f"{float(arrival_of('off-policy learner') - arrival_of('off-policy learner with ccm')):.2f} s"),
        learner_seed_1_without_missed_deadlines=f"{float(learner_1):.2f} s",
        missed_deadlines_later_by=f"{float(missed_1 - learner_1):.2f} s",
        imitator_seed_1_without_latency=f"{float(imitator_1):.2f} s",
        latency_later_by=f"{float(latency_1 - imitator_1):.2f} s")
    base = facts["off-policy learner"]
    for name in SECTION_6_3:
        tag = name.split(", ")[1].replace(" ", "_").replace("-", "_")
        f = facts[name]
        facts["comparison"][f"{tag}_arrival_change"] = (
            f"{float(Fr(f['arrival'].split()[0]) - Fr(base['arrival'].split()[0])):+.2f} s")
        facts["comparison"][f"{tag}_intervention_change"] = (
            f"{float(Fr(f['mean_intervention']) - Fr(base['mean_intervention'])):+.3f}")
    with open(os.path.join(out, "evaluation.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["policy", "package", "seed", "world", "arrival_s", "mean_intervention",
                    "fallbacks", "alarms", "steps_outside", "max_position",
                    "rest_position"])
        for r in results:
            for e in r["eval"]:
                w.writerow([r["name"], r["package"], r["seed"], e["world"],
                            e["arrival"], f"{e['load']:.4f}", e["fallbacks"], e["alarms"],
                            e["out"], f"{e['pmax']:.4f}",
                            "none" if e["rest"] is None else f"{e['rest']:.4f}"])
    with open(os.path.join(out, "training.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["learner", "package", "seed", "episode", "phase", "world",
                    "arrival_s", "mean_intervention", "alarms", "steps_outside",
                    "max_position", "rest_position"])
        for r in results:
            for m in r["log"]:
                if "load" in m:
                    w.writerow([r["name"], r["package"], r["seed"], m["episode"],
                                m.get("phase", "training"), m["world"], m["arrival"],
                                f"{m['load']:.4f}", m["alarms"], m["out"],
                                f"{m['pmax']:.4f}",
                                "none" if m["rest"] is None else f"{m['rest']:.4f}"])
    for name, (rows_all, arrived) in traces.items():
        rows = rows_all[:arrived]                    # the periods up to arrival
        tag = "rule" if name == "rule" else "learner"
        with open(os.path.join(out, f"trace-{tag}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            # the proposal q_k and its answer both concern period k+1, which
            # starts at t = (k+1) T
            w.writerow(["period", "t", "p", "v", "proposed", "alpha", "executed",
                        "intervention"])
            for x in rows:
                q = x["q"]
                w.writerow([x["k"] + 1, f"{(x['k'] + 1) / 10:.1f}", f"{x['p1']:.4f}",
                            f"{x['v1']:.4f}", "nan" if q is None else f"{float(q):.4f}",
                            f"{float(x['alpha']):.2f}",
                            "nan" if x["a_next"] is None else f"{float(x['a_next']):.5f}",
                            f"{float(1 - x['alpha']):.2f}"])
        m = run_metrics(rows_all, arrived)
        full = sum(1 for x in rows if x["alpha"] == 1)
        d = dict(arrival=f"{arrived / 10:.1f} s", mean_intervention=cf.r(m["load_exact"], 3),
                 periods_committed_in_full=str(full), periods=str(len(rows)),
                 periods_not_committed_in_full=str(len(rows) - full),
                 rest_position="none" if m["rest"] is None else f"{m['rest']:.3f} m")
        for x in rows:
            if x["k"] == 0 or 30 <= x["k"] <= 36:
                d[f"k{x['k']}"] = (f"q {float(x['q']):.3f}, alpha {x['alpha']}, "
                                   f"executed {float(x['a_next']):.4f}")
        late = [x for x in rows if x["k"] >= 33]       # periods from 3.4 s on
        d["partial_from_3.4_s"] = ", ".join(
            f"{(x['k'] + 1) / 10:.1f} s at {x['alpha']}" for x in late
            if 0 < x["alpha"] < 1) or "none"
        d["proposals_not_braking_from_3.4_s"] = str(
            sum(1 for x in late if x["q"] is not None and x["q"] >= 0))
        facts[f"trace {name}"] = d
    a_rule = Fr(facts["trace rule"]["arrival"].split(" ")[0])
    a_learner = Fr(facts["trace off-policy learner"]["arrival"].split(" ")[0])
    facts["trace off-policy learner"]["later_than_rule"] = f"{float(a_learner - a_rule):.1f} s"
    return facts


if __name__ == "__main__":
    sys.exit(main())
