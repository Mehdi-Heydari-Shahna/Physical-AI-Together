#!/usr/bin/env python3
"""
The smallest complete example of ALLOVI: a cart approaching a wall.

Physical AI, Together, Chapter 3.

A cart moves on a line towards a wall. The ground limits every acceleration
towards the start, which brakes a forward motion or drives the cart
backwards, to 2 m/s^2 on firm ground (the first 20 m) and to 0.5 m/s^2 on the
downhill ramp (the last 5 m); accelerations towards the wall are realized up
to the command bound of 2 m/s^2. Each segment also holds an operation mode
l: l = 0 with the tool folded, l = 0.8 m with the tool unfolded ahead of the
front bumper. The declared constraint is p + l <= 25 m. A naive proposal
method requests full acceleration until the cart first reaches the work zone
next to the wall, and asks it to hold from then on. Later, the task plan
schedules an operation in mode l = 0.8 m, which consumes 0.8 m of reserve.
The five modules of ALLOVI and the viability governor are implemented as
described in Chapter 3.

Every planned and certified quantity is computed with exact rational
arithmetic (fractions.Fraction), so each number quoted in the chapter can be
checked by hand. The realized motion is exact as well whenever the executed
label is valid. Floating point is used only for the instant and speed at
which the cart crosses a position, for the proposals executed without the
governor, for motion under an invalid label (the persisting predictor of
Section 3.3), and for the checks of physical recovery, including the point
of no return.

Usage
    python3 cart_example.py            # all runs; writes results/
    python3 cart_example.py --check    # also checks every simulation result
                                       # quoted in Chapter 3
    python3 cart_example.py --period 32
                                       # prints the candidates of one period
                                       # of the forecasting run

Outputs, in results/ next to this file
    trace-<configuration>.csv     one row per period: governor's view
    plot-<configuration>.csv      one row per period: signals of the plan
                                  executed in that period (Figures 3.2, 3.3)
    track-persisting.csv          the motion of the persisting run, sampled
                                  every 0.01 s, and the motion predicted by
                                  its last committed plan (Figure 3.4)
    candidates-k<period>.csv      the candidates of selected periods of the
                                  forecasting run
    summary.txt                   the numbers quoted in the chapter

Requires Python 3.8 or later; no other packages.
"""

import argparse
import csv
import math
import os
from fractions import Fraction as Fr

# ----------------------------------------------------------------------------
# Parameters (Table 3.1 of the chapter)
# ----------------------------------------------------------------------------
T = Fr(1, 10)          # period of the governor [s]
U_MAX = Fr(2)          # command bound |u| <= 2 m/s^2
WALL = Fr(25)          # position of the wall [m]; constraint p + l <= 25 m
RAMP = Fr(20)          # the downhill ramp starts here [m]
B_FIRM = Fr(2)         # braking available on firm ground [m/s^2]
B_RAMP = Fr(1, 2)      # braking available on the ramp [m/s^2]
B_GLOBAL = Fr(1, 2)    # global capability: braking of at least 0.5 everywhere
EPS_MIN = Fr(1, 2)     # reserve floor [m]
GRID = Fr(1, 100)      # reserve levels: whole centimetres ...
EPS_TOP = WALL         # ... from 0.50 m up to 25 m
ALPHAS = (Fr(1), Fr(3, 4), Fr(1, 2), Fr(1, 4), Fr(0))   # ambition grid
HORIZON = 400          # forecast horizon [periods]; covers the whole run
WORK_ZONE = Fr(24)     # the task: reach p >= 24 m and hold there
ELL_UNFOLDED = Fr(4, 5)   # operation mode l with the tool unfolded: the
                          # tool reaches 0.8 m ahead of the front bumper
DEMAND_LEVEL = EPS_MIN + ELL_UNFOLDED   # reserve the operation requires: 1.3 m
T_ANNOUNCE = Fr(14)    # the task plan announces the operation at 14 s ...
T_ONSET = Fr(20)       # ... for 20 s (lead time 6 s)
T_END = Fr(22)         # end of the run [s]

NEG_INF = None         # level of a plan that the test does not certify


def period(t):
    """Index of the period that starts at time t."""
    k = t / T
    assert k.denominator == 1
    return int(k)


# ----------------------------------------------------------------------------
# The world: braking available on the ground
# ----------------------------------------------------------------------------
def b_true(p):
    """Braking that the ground allows at position p [m/s^2]."""
    return B_FIRM if p < RAMP else B_RAMP


def b_min(lo, hi):
    """Minimum of the true braking profile over the positions [lo, hi]."""
    return B_RAMP if hi >= RAMP else B_FIRM


# ----------------------------------------------------------------------------
# Segments: a constant command for one period, with a singleton funnel
# ----------------------------------------------------------------------------
class Segment:
    """Constant commanded acceleration u for one period from (p0, v0).

    If every world consistent with the label realizes the command exactly,
    the funnel is the single trajectory computed here.
    """

    __slots__ = ("u", "p0", "v0", "p1", "v1", "lo", "hi")

    def __init__(self, p0, v0, u):
        self.u, self.p0, self.v0 = u, p0, v0
        self.p1 = p0 + v0 * T + u * T * T / 2
        self.v1 = v0 + u * T
        if v0 * self.v1 >= 0:            # no reversal within the period
            self.lo, self.hi = min(p0, self.p1), max(p0, self.p1)
        else:                            # reversal: extreme at the stop
            ts = -v0 / u
            ps = p0 + v0 * ts + u * ts * ts / 2
            self.lo = min(p0, self.p1, ps)
            self.hi = max(p0, self.p1, ps)

    def certified(self, beta):
        """Condition (i) under the label 'braking of at least beta on the
        region of the segment': the command is admissible, its acceleration
        towards the start (braking a forward motion or driving backwards)
        does not exceed beta, and the motion does not reverse within the
        period, so every consistent world realizes the command exactly."""
        if abs(self.u) > U_MAX:
            return False
        if self.u < 0 and -self.u > beta:
            return False
        return self.v0 * self.v1 >= 0


def margin(hi, ell):
    """Clearance of a segment: 25 m minus the largest value of p + l that it
    reaches. Positions within a segment are monotone and its mode l is
    constant, so the largest position hi gives the closest approach at any
    instant, not only at the sampling instants."""
    return WALL - (hi + ell)


def floor_level(x):
    """Largest level of the grid L = {0.50, 0.51, ..., 25.00} not above x."""
    if x is NEG_INF or x < EPS_MIN:
        return NEG_INF
    steps = (x - EPS_MIN) // GRID
    return min(EPS_TOP, EPS_MIN + steps * GRID)


# ----------------------------------------------------------------------------
# Capability predictors: labels for a segment executed in a given period
# ----------------------------------------------------------------------------
class Predictor:
    """Issues the label of a segment: the braking assumed on its region."""

    def __init__(self, kind):
        assert kind in ("fixed", "reactive", "persisting", "forecasting")
        self.kind = kind
        self.seen_lo = Fr(0)     # positions traversed before the current time
        self.seen_hi = Fr(0)
        self.b_here = B_FIRM     # braking measured at the current position

    def observe(self, p_lo, p_hi, p_now):
        self.seen_lo = min(self.seen_lo, p_lo)
        self.seen_hi = max(self.seen_hi, p_hi)
        self.b_here = b_true(p_now)

    def label(self, k, j, seg):
        """Label issued during period k for segment seg in period j."""
        if j > k + 1 + HORIZON or self.kind == "fixed":
            return B_GLOBAL
        if self.kind == "forecasting":       # minimum of a correct map
            return b_min(seg.lo, seg.hi)
        if self.kind == "reactive":          # measured on traversed positions
            if self.seen_lo <= seg.lo and seg.hi <= self.seen_hi:
                return b_min(seg.lo, seg.hi)
            return B_GLOBAL
        return self.b_here                   # persisting


# ----------------------------------------------------------------------------
# Labelled plans and the certification test
# ----------------------------------------------------------------------------
class Plan:
    """Labelled plan: pairs (segment, label), numbered from 0, executed in
    periods j0, j0+1, ..., followed by the rest segment at p_rest. Every
    segment of the plan, its rest segment included, has the operation mode
    ell: 0 with the tool folded, ELL_UNFOLDED with the tool unfolded."""

    def __init__(self, j0, pairs, p_rest, ell=Fr(0)):
        self.j0, self.pairs, self.p_rest = j0, pairs, p_rest
        self.ell = ell

    def reserve(self):
        """Smallest clearance of the plan, its rest segment included, if
        every segment satisfies condition (i) under its label; else None."""
        if not all(seg.certified(beta) for seg, beta in self.pairs):
            return None
        clear = [margin(seg.hi, self.ell) for seg, _ in self.pairs]
        clear.append(margin(self.p_rest, self.ell))
        return min(clear)

    def head_command(self):
        """Command of the head; the rest segment holds the cart (zero)."""
        return self.pairs[0][0].u if self.pairs else Fr(0)

    def shift(self):
        """The plan without its executed head: its pair n is pair n+1 of
        this plan."""
        return Plan(self.j0 + 1, self.pairs[1:], self.p_rest, self.ell)

    def profile(self):
        """Levels certified from pair n on, n = 0..N; entry N is the level of
        the rest segment alone. The test certifies a pair at level e if the
        segment satisfies condition (i) under its label and its clearance is
        at least e, and the rest segment if its clearance is at least e; the
        chaining conditions hold by construction, because every segment
        starts where the previous one ends and the rest segment starts at
        standstill."""
        n_pairs = len(self.pairs)
        out = [NEG_INF] * (n_pairs + 1)
        running = margin(self.p_rest, self.ell)
        out[n_pairs] = floor_level(running)
        ok = True
        for i in range(n_pairs - 1, -1, -1):
            seg, beta = self.pairs[i]
            ok = ok and seg.certified(beta)
            running = min(running, margin(seg.hi, self.ell))
            out[i] = floor_level(running) if ok else NEG_INF
        return out


def level_at(profile, n):
    return profile[min(n, len(profile) - 1)]


# ----------------------------------------------------------------------------
# The control method: candidate generator
# ----------------------------------------------------------------------------
def braking_step(pred, k, j, p, v):
    """Strongest braking certified under the label of its own segment.
    Braking a forward motion accelerates the cart towards the start and is
    limited by the label; braking a backward motion accelerates it towards
    the wall and is realized up to the command bound."""
    if v < 0:
        seg = Segment(p, v, min(U_MAX, -v / T))
        return seg, pred.label(k, j, seg)
    for b_try in (B_FIRM, B_RAMP):
        u = -min(b_try, v / T)
        seg = Segment(p, v, u)
        beta = pred.label(k, j, seg)
        if seg.certified(beta):
            return seg, beta
    raise AssertionError("braking at the global capability is always certified")


def backward_step(pred, k, j, p, v):
    """Strongest backward drive certified under the label of its segment."""
    for b_try in (B_FIRM, B_RAMP):
        seg = Segment(p, v, -b_try)
        beta = pred.label(k, j, seg)
        if seg.certified(beta):
            return seg, beta
    raise AssertionError("driving at the global capability is always certified")


def brake_to_rest(pred, k, j, p, v):
    """Maximal certified braking to standstill, starting in period j."""
    pairs = []
    while v != 0:
        seg, beta = braking_step(pred, k, j, p, v)
        pairs.append((seg, beta))
        p, v, j = seg.p1, seg.v1, j + 1
    return pairs, p


def recovery(pred, k, j, p, v, target):
    """Recovery continuation from state (p, v) at the start of period j.

    Maximal certified braking to standstill. If a target position is given
    (a demand is pending) and the cart comes to rest beyond it, the cart then
    backs away: the strongest certified backward command for m periods, then
    maximal braking, with the smallest m that comes to rest at or before the
    target.
    """
    pairs, p = brake_to_rest(pred, k, j, p, v)
    j += len(pairs)
    if target is None or p <= target:
        return pairs, p
    for m in range(1, 200):
        trial, pm, vm, jm = [], p, Fr(0), j
        for _ in range(m):
            seg, beta = backward_step(pred, k, jm, pm, vm)
            trial.append((seg, beta))
            pm, vm, jm = seg.p1, seg.v1, jm + 1
        tail, p_stop = brake_to_rest(pred, k, jm, pm, vm)
        if p_stop <= target:
            return pairs + trial + tail, p_stop
    raise AssertionError("no back-away found")


def generate(pred, k, xhat, u_q, fallback, target, ell):
    """Candidates of period k: blends of the proposed command u_q with the
    fallback head for each ambition, each followed by the recovery
    continuation, and the recovery continuation alone with ambition 0. Every
    segment of every candidate has the operation mode ell: 0 until the
    operation begins, ELL_UNFOLDED from the onset period on once the demand
    is met."""
    p, v = xhat
    u_fb = fallback.head_command()
    cands = []
    for a in ALPHAS:
        head = Segment(p, v, a * u_q + (1 - a) * u_fb)
        pairs = [(head, pred.label(k, k + 1, head))]
        tail, p_rest = recovery(pred, k, k + 2, head.p1, head.v1, target)
        cands.append((Plan(k + 1, pairs + tail, p_rest, ell), a))
    tail, p_rest = recovery(pred, k, k + 1, p, v, target)
    if not tail:                 # already at rest: hold for one period
        hold = Segment(p, v, Fr(0))
        tail = [(hold, pred.label(k, k + 1, hold))]
    cands.append((Plan(k + 1, tail, p_rest, ell), Fr(0)))
    return cands


# ----------------------------------------------------------------------------
# The proposal method
# ----------------------------------------------------------------------------
class ProposalMethod:
    """Naive proposal method. It requests 'accelerate' until the measured
    position first reaches the work zone, and 'hold' from then on. It knows
    nothing about the wall, the ramp or the tool, and it never sets the
    operation mode."""

    def __init__(self):
        self.arrived = False

    def __call__(self, p):
        if p >= WORK_ZONE:
            self.arrived = True
        return "hold" if self.arrived else "accelerate"


def proposed_command(q, xhat):
    """Physical meaning of the proposal, applied by the generator to the
    initial set: 'accelerate' is a command of +2 m/s^2; 'hold' is the command
    that brings the cart to rest within one period if braking of 2 m/s^2
    allows it (firm ground assumed), and zero at rest."""
    if q == "accelerate":
        return U_MAX
    v = xhat[1]
    return max(-U_MAX, min(U_MAX, -v / T))


# ----------------------------------------------------------------------------
# The viability governor: admission and selection
# ----------------------------------------------------------------------------
def select(cands, profiles, fb_profile, req, onset):
    """Among admitted plans (level at least the floor), the lexicographic
    maximum of (min(onset level, requirement), ambition); ties go to fresh
    candidates, then to the smallest index. Index 0 is the fallback."""
    best, best_key = None, None
    entries = [(0, fb_profile, Fr(0))]
    entries += [(i + 1, profiles[i], cands[i][1]) for i in range(len(cands))]
    for idx, prof, a in entries:
        lvl = prof[0]
        if lvl is NEG_INF or lvl < EPS_MIN:
            continue
        key = (min(level_at(prof, onset), req), a, 1 if idx >= 1 else 0, -idx)
        if best_key is None or key > best_key:
            best, best_key = idx, key
    return best


# ----------------------------------------------------------------------------
# Realized motion in the true world
# ----------------------------------------------------------------------------
def realize(p, v, u):
    """Motion over one period under command u in the true world. Returns
    (p1, v1, exact). The motion is exact (rational) when the ground realizes
    the command as commanded: always for accelerations towards the wall,
    and for accelerations towards the start within the braking that the
    ground allows on the whole segment. Otherwise the command brakes harder
    than the ground allows somewhere on the segment; this happens only for
    forward motion in this example (Section 3.3), and the motion is then
    computed in floating point, with the ramp crossing found exactly and the
    brakes holding the cart once it stops."""
    seg = Segment(p, v, u)
    if u >= 0 or -u <= b_min(seg.lo, seg.hi):
        return seg.p1, seg.v1, True
    pf, vf = true_motion(float(p), float(v), float(u), float(T))
    return pf, vf, False


def true_motion(pf, vf, uf, left):
    """State after 'left' seconds under the constant command uf from
    (pf, vf) in the true world, in floating point. Accelerations towards
    the wall are realized as commanded; braking a forward motion is realized
    at min(-uf, b(p)), with the ramp crossing found exactly and the brakes
    holding the cart once it stops. Used for motion under an invalid label
    and for sampling the motion of the persisting run (Figure 3.4)."""
    if uf >= 0:
        return pf + vf * left + 0.5 * uf * left * left, vf + uf * left
    assert vf > 0, "only forward braking occurs under an invalid label here"
    ramp = float(RAMP)
    while left > 0 and vf > 0:
        a = -min(-uf, float(B_FIRM) if pf < ramp else float(B_RAMP))
        t_stop = vf / -a
        t_ev, crossing = min(left, t_stop), False
        if pf < ramp:                            # may cross onto the ramp
            disc = vf * vf + 2 * a * (ramp - pf)
            if disc >= 0:
                t_c = (-vf + math.sqrt(disc)) / a
                if t_c < t_ev:
                    t_ev, crossing = t_c, True
        pf += vf * t_ev + 0.5 * a * t_ev * t_ev
        vf += a * t_ev
        left -= t_ev
        if crossing:
            pf = ramp
        elif t_ev == t_stop:
            vf = 0.0
    return pf, vf


def stop_position_true(pf, vf):
    """Position at which maximal braking brings the cart to rest from
    (pf, vf) in the true world: 2 m/s^2 before the ramp, 0.5 m/s^2 on it.
    A cart that is not moving towards the wall is already where it can stop.
    Physical recovery is available exactly when this position is at most
    25 m (the tool is folded throughout the persisting run)."""
    if vf <= 0:
        return pf
    ramp, b1, b2 = float(RAMP), float(B_FIRM), float(B_RAMP)
    if pf >= ramp:
        return pf + vf * vf / (2 * b2)
    if pf + vf * vf / (2 * b1) <= ramp:
        return pf + vf * vf / (2 * b1)
    return ramp + (vf * vf - 2 * b1 * (ramp - pf)) / (2 * b2)


# ----------------------------------------------------------------------------
# One run of the closed loop
# ----------------------------------------------------------------------------
def run(kind, t_announce=T_ANNOUNCE, t_onset=T_ONSET, t_end=T_END,
        keep_candidates=()):
    """Runs the governor with capability predictor 'kind'. The demand is
    announced at t_announce for t_onset. If it is not met at its onset, the
    overload is recorded and the operation is not performed in this run.
    The run stops at the first label alarm (see continue_after_alarm)."""
    pred = Predictor(kind)
    proposal = ProposalMethod()
    k0, kd, k_end = period(t_announce), period(t_onset), period(t_end)
    p, v = Fr(0), Fr(0)
    plan = Plan(0, [], Fr(0))                    # initial plan: rest at p = 0
    prof = plan.profile()
    ell_op, k_met, overload = Fr(0), None, False   # operation mode offered
    req_prev, onset_prev = EPS_MIN, 0
    rows, kept = [], {}
    for k in range(k_end):
        # --- period k: the control method executes the head of the plan ----
        u_exec = plan.head_command()
        beta_exec = plan.pairs[0][1] if plan.pairs else B_GLOBAL
        if plan.pairs:
            xhat = (plan.pairs[0][0].p1, plan.pairs[0][0].v1)   # end of funnel
        else:
            xhat = (plan.p_rest, Fr(0))                         # rest funnel
        # --- outputs of the proposal method and the demand predictor -------
        q = proposal(p)
        u_q = proposed_command(q, xhat)
        demand_now = k_met is None and not overload and k >= k0
        if demand_now and k < kd:
            req, onset = DEMAND_LEVEL, kd - k - 1
        else:
            if demand_now:                       # k == kd: is the demand met?
                if prof[0] is not NEG_INF and prof[0] >= DEMAND_LEVEL:
                    # the operation begins: from now on, every candidate is
                    # in mode l = 0.8 m, and the tool unfolds only if the
                    # governor commits such a plan
                    k_met, ell_op = k, ELL_UNFOLDED
                else:
                    overload = True
            req, onset = EPS_MIN, 0
        target = WALL - req if req > EPS_MIN else None
        # --- the governor: candidates, certification, selection ------------
        fallback = plan.shift()
        fb_prof = prof[1:] if len(prof) > 1 else prof
        cands = generate(pred, k, xhat, u_q, fallback, target, ell_op)
        profiles = [c[0].profile() for c in cands]
        sel = select(cands, profiles, fb_prof, req, onset)
        if sel == 0:
            new_plan, new_prof, alpha = fallback, fb_prof, Fr(0)
            assert k_met != k, "the operation must begin with a fresh plan"
        else:
            new_plan, new_prof = cands[sel - 1][0], profiles[sel - 1]
            alpha = cands[sel - 1][1]
        if k in keep_candidates:
            kept[k] = dict(x=(p, v), xhat=xhat, mode=q, q=u_q,
                           u_fb=fallback.head_command(), req=req, onset=onset,
                           cands=cands, profiles=profiles, fb_prof=fb_prof,
                           fb_rest=fallback.p_rest, sel=sel)
        # --- realized motion, measurement and label alarm -------------------
        p1, v1, ex = realize(p, v, u_exec)
        alarm = not (ex and (p1, v1) == xhat)   # exact measurement
        rows.append(dict(
            k=k, t=k * T, p=p, v=v, u=u_exec, beta=beta_exec,
            level=prof[0], rest=plan.p_rest,
            req_exec=req_prev, onset_level_exec=level_at(prof, onset_prev),
            mode=q, q=u_q, alpha=alpha, sel=sel, eps=new_prof[0],
            reserve=new_plan.reserve(), plan=new_plan,
            eps_on=level_at(new_prof, onset), req=req, onset=onset,
            p_rest=new_plan.p_rest,
            ell=plan.ell,                       # mode of the executed segment
            alarm=int(alarm)))
        if alarm:
            rows.append(dict(k=k + 1, t=(k + 1) * T, p=p1, v=v1, final=True))
            return rows, kept, dict(alarm_k=k + 1, k_met=k_met, kd=kd,
                                    overload=overload, state=(p1, v1))
        pred.observe(min(p, p1), max(p, p1), p1)
        p, v = p1, v1
        plan, prof = new_plan, new_prof
        req_prev, onset_prev = req, onset   # onset pair of the new plan
    rows.append(dict(k=k_end, t=k_end * T, p=p, v=v, final=True))
    return rows, kept, dict(alarm_k=None, k_met=k_met, kd=kd,
                            overload=overload)


def direct_execution(t_end=Fr(12)):
    """The proposals executed directly, without the governor: in each period
    the cart applies at once the command that the proposal method requests
    for the measured state. Returns the first instant and speed at which the
    cart reaches the wall (floating point; braking realized by the ground)."""
    proposal = ProposalMethod()
    p, v = 0.0, 0.0
    for k in range(period(t_end)):
        u = float(proposed_command(proposal(Fr(p)), (Fr(p), Fr(v))))
        p1, v1, _ = realize(Fr(p), Fr(v), Fr(u))
        p1, v1 = float(p1), float(v1)
        if p1 >= float(WALL):
            # reach the wall within the period: constant realized acceleration
            # on the ramp, b = 0.5, if braking; otherwise the command
            a = -float(B_RAMP) if u < 0 else u
            d = float(WALL) - p
            s = (-v + math.sqrt(v * v + 2 * a * d)) / a if a != 0 else d / v
            return float(k * T) + s, v + a * s
        p, v = p1, v1
    return None, None


def continue_after_alarm(state):
    """Response to a label alarm: restart with the fixed predictor and a plan
    labelled by the global capability and certified on the measured state,
    if one exists; otherwise maximal braking without a guarantee. Returns
    whether a certified restart exists and the motion until standstill,
    integrated with 2000 steps per second in the true world."""
    pf, vf = state
    certified = pf + vf * vf / (2 * float(B_GLOBAL)) <= float(WALL - EPS_MIN)
    track, t, dt = [], 0.0, 1.0 / 2000.0
    while vf > 0:
        a = -float(B_GLOBAL) if certified else -float(b_true(Fr(pf)))
        dt_i = min(dt, -vf / a)
        pf += vf * dt_i + 0.5 * a * dt_i * dt_i
        vf += a * dt_i
        if vf < 1e-12:
            vf = 0.0
        t += dt_i
        track.append((t, pf, vf))
    return certified, track


# ----------------------------------------------------------------------------
# The run with the persisting predictor (Section 3.3)
# ----------------------------------------------------------------------------
def ramp_crossing(pf, vf, uf):
    """Instant within one period at which the front reaches the ramp under
    the constant command uf in the true world (firm ground until then), or
    None if it does not."""
    ramp = float(RAMP)
    if pf >= ramp:
        return None
    a = uf if uf >= 0 else -min(-uf, float(B_FIRM))
    d = ramp - pf
    if a == 0:
        tc = d / vf if vf > 0 else None
    else:
        disc = vf * vf + 2 * a * d
        tc = (-vf + math.sqrt(disc)) / a if disc >= 0 else None
    return tc if tc is not None and 0 < tc < float(T) else None


def plan_motion(plan, t):
    """State that a plan predicts at time t: the single trajectory of its
    funnels from the start of period j0, then its point of rest."""
    t0 = float(plan.j0 * T)
    i = int((t - t0) / float(T) + 1e-9)
    if i >= len(plan.pairs):
        return float(plan.p_rest), 0.0
    seg = plan.pairs[i][0]
    tau = t - (t0 + i * float(T))
    p0, v0, u = float(seg.p0), float(seg.v0), float(seg.u)
    return p0 + v0 * tau + 0.5 * u * tau * tau, v0 + u * tau


def alarm_track(rows, info, after):
    """Motion of a run that ends in a label alarm, for Figure 3.4: the
    realized motion every 0.01 s of each period before the alarm and at the
    instant the front reaches the ramp, then the motion after the alarm
    every 0.01 s and at standstill. From the start of the period of the
    alarm on, each row also gives the state predicted by the plan executed
    in that period. Rows (t, p, v, p_plan, v_plan) in floating point, with
    None where the prediction does not apply."""
    body = [x for x in rows if not x.get("final")]
    plan = body[info["alarm_k"] - 2]["plan"]     # executed in the alarm period
    t_plan = float(plan.j0 * T)
    pts = []
    for x in body:
        t0, p0, v0, u = (float(x[c]) for c in ("t", "p", "v", "u"))
        taus = [i / 100 for i in range(10)]
        tc = ramp_crossing(p0, v0, u)
        if tc is not None:
            taus = sorted(taus + [tc])
        pts += [(t0 + tau,) + true_motion(p0, v0, u, tau) for tau in taus]
    t_a = float(info["alarm_k"] * T)
    pts.append((t_a,) + tuple(info["state"]))
    pts += [(t_a + t, p, v) for i, (t, p, v) in enumerate(after)
            if (i + 1) % 20 == 0 or v == 0]
    return [(t, p, v) + (plan_motion(plan, t) if t >= t_plan - 1e-9
                         else (None, None)) for t, p, v in pts]


def write_track(path, track):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "p", "v", "p_plan", "v_plan"])
        for row in track:
            w.writerow([f"{row[0]:.4f}"] +
                       ["nan" if x is None else f"{x:.5f}" for x in row[1:]])


def first_reach(track, level):
    """First instant at which a sampled motion (t, p, v), with constant
    acceleration between samples, reaches the position 'level', and the
    speed then."""
    for (t0, p0, v0), (t1, p1, v1) in zip(track, track[1:]):
        if p0 < level <= p1:
            a = (v1 - v0) / (t1 - t0)
            d = level - p0
            s = d / v0 if a == 0 else (-v0 + math.sqrt(v0 * v0 + 2 * a * d)) / a
            return t0 + s, v0 + a * s
    return None, None


def point_of_no_return(rows, limit=WALL):
    """First instant of a run at which physical recovery is lost: from then
    on, maximal braking in the true world no longer brings the cart to rest
    at or before 'limit'. Within a period, the position at which maximal
    braking would stop the cart never decreases, so the first period
    boundary beyond the limit is found and the instant is located within
    the preceding period by bisection. Returns (t, p, v), or None."""
    lim = float(limit)
    for x, nxt in zip(rows, rows[1:]):
        if stop_position_true(float(nxt["p"]), float(nxt["v"])) > lim:
            p0, v0, u = float(x["p"]), float(x["v"]), float(x["u"])
            lo, hi = 0.0, float(T)
            for _ in range(60):
                mid = (lo + hi) / 2
                if stop_position_true(*true_motion(p0, v0, u, mid)) > lim:
                    hi = mid
                else:
                    lo = mid
            return (float(x["t"]) + hi,) + true_motion(p0, v0, u, hi)
    return None


def facts_of_persisting(rows, info, certified, after):
    """The numbers that Section 3.3 quotes about the run with the persisting
    predictor, as strings rounded as in the text."""
    body = [x for x in rows if not x.get("final")]
    f = {}
    first_mod = next(x["k"] for x in body if x["alpha"] < 1)
    f["accepted_unchanged"] = str(first_mod)
    x_end = body[first_mod + 1]          # start of the first modified period
    f["acceleration_ends"] = (f"{r(x_end['p'], 2)} m, {r(x_end['v'], 1)} m/s "
                              f"at {r(x_end['t'], 1)} s")

    def invalid(seg, beta):
        return beta > b_min(seg.lo, seg.hi)

    # the plan committed in period k is executed from period k + 1
    k_inv = next(x["k"] for x in body
                 if any(invalid(s, b) for s, b in x["plan"].pairs))
    f["first_invalid_plan_committed_s"] = r((k_inv + 1) * T, 1)
    ka = info["alarm_k"] - 1                             # period of the alarm
    f["invalid_labels_in_every_later_plan"] = str(all(
        any(invalid(s, b) for s, b in x["plan"].pairs)
        for x in body if k_inv <= x["k"] < ka))
    f["min_level"] = r(min(x["eps"] for x in body), 2)
    t_n, p_n, v_n = point_of_no_return(rows)
    f["point_of_no_return"] = f"{r(t_n, 2)} s, {r(p_n, 1)} m, {r(v_n, 2)} m/s"
    # from point A on, maximal braking no longer stops the cart before the
    # wall, at any later period boundary (and at the alarm)
    f["unrecoverable_after_point_of_no_return"] = str(all(
        stop_position_true(float(x["p"]), float(x["v"])) > float(WALL)
        for x in rows if float(x["t"]) > t_n))
    # executed labels: the head of the plan committed one period earlier
    bad = next(x["k"] + 1 for x in body
               if x["plan"].pairs and invalid(*x["plan"].pairs[0]))
    f["first_executed_invalid_label"] = f"period {bad}"
    plan = body[ka - 1]["plan"]
    seg, beta = plan.pairs[0]
    f["last_plan_committed_s"] = r(ka * T, 1)
    f["last_plan_pairs"] = (
        f"{len(plan.pairs)}, commands "
        f"{','.join(sorted({fmt(s.u) for s, _ in plan.pairs}))}, labels "
        f"{','.join(sorted({fmt(b) for _, b in plan.pairs}))}")
    f["last_plan_rest_level"] = f"{fmt(plan.p_rest)},{fmt(body[ka - 1]['eps'])}"
    f["last_plan_invalid_labels"] = (
        f"{sum(invalid(s, b) for s, b in plan.pairs)} of {len(plan.pairs)}")
    f["last_plan_head"] = (f"{fmt(seg.u)}, label {fmt(beta)}, {fmt(seg.p0)} to "
                           f"{fmt(seg.p1)} m, period {plan.j0}")
    f["last_plan_head_start"] = f"{fmt(seg.p0)},{fmt(seg.v0)}"
    f["levels_while_braking"] = ",".join(sorted(
        {fmt(x["eps"]) for x in body if first_mod <= x["k"] < ka}))
    t_d, v_d = crossing(rows, RAMP)
    f["divergence"] = f"{r(t_d, 3)} s at {r(v_d, 3)} m/s"
    f["stop_distance_at_ramp"] = f"{v_d * v_d / (2 * float(B_RAMP)):.2f} m"
    pa, va = info["state"]
    f["alarm"] = (f"{r(info['alarm_k'] * T, 1)} s; measured {pa:.4f} m, "
                  f"{va:.4f} m/s; initial set {fmt(seg.p1)} m, {fmt(seg.v1)} m/s")
    f["alarm_difference"] = (f"{(pa - float(seg.p1)) * 1000:.2f} mm, "
                             f"{va - float(seg.v1):.3f} m/s")
    need = va * va / (2 * float(B_GLOBAL))
    f["certified_restart"] = (f"{certified}; needs {need:.2f} m, "
                              f"{float(WALL - EPS_MIN) - pa:.2f} m available")
    t_a = float(info["alarm_k"] * T)
    motion = [(t_a, pa, va)] + [(t_a + t, p, v) for t, p, v in after]
    t_w, v_w = first_reach(motion, float(WALL))
    f["wall_reached"] = f"{t_w:.2f} s at {v_w:.2f} m/s"
    t_f, v_f = first_reach(motion, float(WALL - EPS_MIN))
    f["floor_lost"] = f"{t_f:.2f} s at {v_f:.2f} m/s"
    f["stops_at"] = f"{motion[-1][1]:.2f} m at {motion[-1][0]:.2f} s"
    return f


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def fmt(x, nd=6):
    if x is NEG_INF:
        return "-inf"
    if isinstance(x, Fr):
        if x.denominator == 1:
            return str(x.numerator)
        return f"{float(x):.{nd}f}".rstrip("0").rstrip(".")
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


TRACE_COLS = ["k", "t", "p", "v", "u", "ell", "beta", "mode", "q", "alpha",
              "sel", "reserve", "eps", "eps_on", "req", "onset", "p_rest",
              "alarm"]


def write_trace(path, rows):
    """Governor's view: in period k, the plan for period k+1 is chosen."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(TRACE_COLS)
        for r in rows:
            if not r.get("final"):
                w.writerow([fmt(r[c]) for c in TRACE_COLS])


def write_plot(path, rows):
    """Signals of the plan executed in each period k: the state at t_k, the
    executed command, the proposed command and the ambition with which the
    plan was committed, its rest position, its reserve and its reserve from
    the onset of a pending demand on, and its operation mode."""
    cols = ["t", "p", "v", "u", "q", "alpha", "rest", "level", "onset_level",
            "req", "ell"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        prev = None
        for r in rows:
            if r.get("final"):
                w.writerow([fmt(r["t"]), fmt(r["p"]), fmt(r["v"])] +
                           ["nan"] * (len(cols) - 3))
                break
            q = prev["q"] if prev else "nan"
            a = prev["alpha"] if prev else "nan"
            w.writerow([fmt(r["t"]), fmt(r["p"]), fmt(r["v"]), fmt(r["u"]),
                        fmt(q), fmt(a), fmt(r["rest"]), fmt(r["level"]),
                        fmt(r["onset_level_exec"]), fmt(r["req_exec"]),
                        fmt(r["ell"])])
            prev = r


def crossing(rows, level):
    """First instant at which the position reaches 'level', and the speed
    then (exact kinematics within the period, evaluated in floating point)."""
    for r0, r1 in zip(rows, rows[1:]):
        if r0.get("final"):
            break
        if r0["p"] < level <= r1["p"]:
            p, v, u = (float(r0[c]) for c in ("p", "v", "u"))
            d = float(level) - p
            s = d / v if u == 0 else (-v + math.sqrt(v * v + 2 * u * d)) / u
            return float(r0["t"]) + s, v + u * s
    return None, None


def candidate_table(path, rec):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "ambition", "head_command", "head_label",
                    "head_end_p", "head_end_v", "pairs", "rest_position",
                    "level", "onset_level", "admitted", "selected"])
        fbp = rec["fb_prof"]
        w.writerow([0, 0, fmt(rec["u_fb"]), "earlier", "", "", "",
                    fmt(rec["fb_rest"]), fmt(fbp[0]),
                    fmt(level_at(fbp, rec["onset"])),
                    int(fbp[0] is not NEG_INF and fbp[0] >= EPS_MIN),
                    int(rec["sel"] == 0)])
        for i, ((plan, a), prof) in enumerate(zip(rec["cands"], rec["profiles"]), 1):
            seg, beta = plan.pairs[0]
            w.writerow([i, fmt(a), fmt(seg.u), fmt(beta), fmt(seg.p1),
                        fmt(seg.v1), len(plan.pairs), fmt(plan.p_rest),
                        fmt(prof[0]), fmt(level_at(prof, rec["onset"])),
                        int(prof[0] is not NEG_INF and prof[0] >= EPS_MIN),
                        int(rec["sel"] == i)])


def print_period(k, rec):
    print(f"period {k}: state ({fmt(rec['x'][0])}, {fmt(rec['x'][1])}); "
          f"initial set {{({fmt(rec['xhat'][0])}, {fmt(rec['xhat'][1])})}}; "
          f"proposal '{rec['mode']}' = {fmt(rec['q'])} m/s^2; fallback head "
          f"{fmt(rec['u_fb'])} m/s^2; requirement ({fmt(rec['req'])} m, "
          f"onset pair {rec['onset']})")
    fbp = rec["fb_prof"]
    print(f"   0  fallback                 rest {fmt(rec['fb_rest']):>9}"
          f"  level {fmt(fbp[0]):>5}  onset level {fmt(level_at(fbp, rec['onset'])):>5}"
          f"{'  <- selected' if rec['sel'] == 0 else ''}")
    for i, ((plan, a), prof) in enumerate(zip(rec["cands"], rec["profiles"]), 1):
        seg, beta = plan.pairs[0]
        print(f"   {i}  ambition {fmt(a):>4}  u {fmt(seg.u):>5}  label {fmt(beta):>3}"
              f"  rest {fmt(plan.p_rest):>9}  level {fmt(prof[0]):>5}"
              f"  onset level {fmt(level_at(prof, rec['onset'])):>5}"
              f"{'  <- selected' if rec['sel'] == i else ''}")


def r(x, nd):
    """x rounded to nd decimals, as a string (exact for fractions)."""
    if isinstance(x, Fr):
        q = Fr(round(x * 10 ** nd), 10 ** nd)
        return f"{float(q):.{nd}f}"
    return f"{x:.{nd}f}"


def facts_of_run(name, rows, kept, info):
    """The numbers that Chapter 3 quotes about one run, as strings rounded
    as in the text."""
    ends = [x["p"] for x in rows[1:]]     # position at the end of each period
    rows = [x for x in rows if not x.get("final")]
    f = {}
    k_arr = next(x["k"] for x in rows if x["p"] >= WORK_ZONE)
    f["arrival_s"] = r(k_arr * T, 1)
    upto = [x for x in rows if x["k"] < k_arr]
    n1 = sum(1 for x in upto if x["alpha"] == 1)
    nq = sum(1 for x in upto if x["alpha"] == Fr(1, 4))
    n0 = sum(1 for x in upto if x["alpha"] == 0)
    f["counts_1_quarter_0"] = f"{n1},{nq},{n0},{len(upto)}"
    load = sum(1 - x["alpha"] for x in upto) / len(upto)
    f["load"] = f"{load} = {r(load, 2)}"
    first_mod = next(x["k"] for x in rows if x["alpha"] < 1)
    f["accepted_first"] = str(first_mod)              # proposals 0..first_mod-1
    f["first_modified_period"] = str(first_mod + 1)
    f["state_when_acceptance_ends"] = (
        f"{r(rows[first_mod + 1]['p'], 2)},{r(rows[first_mod + 1]['v'], 1)}")
    f["first_plan_rest"] = r(rows[0]["p_rest"], 2)
    f["last_accepted_plan_rest"] = r(rows[first_mod - 1]["p_rest"], 2)
    f["partial_periods_before_arrival"] = ",".join(
        str(x["k"] + 1) for x in upto if 0 < x["alpha"] < 1)
    t_ramp, v_ramp = crossing(rows, RAMP)
    f["ramp_entry"] = f"{r(v_ramp, 3)} m/s at {r(t_ramp, 2)} s"
    f["vmax"] = r(max(x["v"] for x in rows), 1)
    k_stop = next(x["k"] for x in rows if x["k"] > k_arr and x["v"] == 0)
    f["rest"] = f"{r(rows[k_stop]['p'], 3)} m at {r(k_stop * T, 1)} s"
    # first executed period after arrival whose plan follows 'hold' unchanged
    k_hold = next(x["k"] for x in rows if x["k"] >= k_arr and x["alpha"] == 1)
    f["hold_accepted_from_s"] = r((k_hold + 1) * T, 1)
    f["min_level"] = r(min(x["eps"] for x in rows), 2)
    f["min_reserve"] = r(min(x["reserve"] for x in rows), 3)
    f["alarms"] = str(sum(x["alarm"] for x in rows))
    f["fallbacks_before_arrival"] = str(sum(1 for x in upto if x["sel"] == 0))
    # the operation
    k0, kd = period(T_ANNOUNCE), info["kd"]
    f["requirement_issued_in_period"] = str(next(
        x["k"] for x in rows if x["req"] > EPS_MIN))
    back = next(x["k"] for x in rows if x["k"] >= k0 and x["u"] != 0)
    f["backaway_start_s"] = r(back * T, 1)
    f["hold_after_announcement_s"] = r(back * T - T_ANNOUNCE, 1)
    f["onset_state"] = f"{r(rows[kd]['p'], 3)},{r(rows[kd]['v'], 2)}"
    f["met_at_onset"] = str(info["k_met"] == kd)
    # first period whose executed segment is in mode l = 0.8 m
    k_unf = next((x["k"] for x in rows if x["ell"] > 0), None)
    f["unfolded_from_s"] = r(k_unf * T, 1) if k_unf is not None else "-"
    f["level_after_unfolding"] = r(rows[kd + 2]["eps"], 2)
    f["alpha_after_unfolding"] = str(rows[kd + 2]["alpha"])
    # the constraint p + l <= 25 m in the realized motion: positions are
    # monotone within each period and the mode is constant, so the largest
    # value of p + l in period k is max(p_k, p_k+1) + l_k
    f["max_p_plus_ell"] = fmt(max(max(x["p"], e) + x["ell"]
                                  for x, e in zip(rows, ends)))
    f["alarm_checks"] = str(len(rows))
    f["recoverable_at_every_boundary"] = str(all(
        stop_position_true(float(x["p"]), float(x["v"])) + float(x["ell"])
        <= float(WALL) for x in rows))
    # plans executed in the back-away periods back..kd-1 were committed in
    # periods back-1..kd-2
    committed = [x for x in rows if back - 1 <= x["k"] <= kd - 2]
    f["backaway_fallbacks"] = f"{sum(1 for x in committed if x['sel'] == 0)} of {len(committed)}"
    f["backaway_braking_accepted"] = ",".join(
        str(x["k"] + 1) for x in committed
        if x["alpha"] == 1 and x["plan"].head_command() > 0)
    f["onset_level_held_from_announcement"] = ",".join(sorted(
        {fmt(x["eps_on"]) for x in rows if k0 <= x["k"] < kd}))
    f["level_held_while_holding"] = ",".join(sorted(
        {fmt(x["eps"]) for x in rows if k0 <= x["k"] < back - 1}))
    f["ambition_while_holding"] = ",".join(sorted(
        {fmt(x["alpha"]) for x in rows if k_hold <= x["k"] < back - 1}))
    f["commands_of_partial_blends"] = ",".join(
        fmt(rows[x["k"] + 1]["u"]) for x in upto if 0 < x["alpha"] < 1)
    f["full_acceleration_periods"] = (
        f"{min(x['k'] for x in rows if x['u'] == U_MAX)}-"
        f"{max(x['k'] for x in rows if x['u'] == U_MAX and x['k'] <= first_mod)}")
    return f


def facts_of_candidates(kept):
    """The numbers that Section 3.1 quotes about periods 31, 32 and 179."""
    f = {}
    rec31 = kept[31]
    f["k31_selected_rest"] = r(rec31["cands"][rec31["sel"] - 1][0].p_rest, 2)
    rec = kept[32]
    f["k32_state"] = f"{r(rec['x'][0], 2)},{r(rec['x'][1], 1)}"
    f["k32_initial_set"] = f"{r(rec['xhat'][0], 2)},{r(rec['xhat'][1], 1)}"
    f["k32_proposed"] = fmt(rec["q"])
    f["k32_fallback"] = f"{fmt(rec['u_fb'])},{r(rec['fb_rest'], 2)},{fmt(rec['fb_prof'][0])}"
    f["k32_heads"] = ",".join(fmt(c[0].pairs[0][0].u) for c in rec["cands"])
    f["k32_head_labels"] = ",".join(fmt(c[0].pairs[0][1]) for c in rec["cands"])
    f["k32_rests"] = ",".join(fmt(c[0].p_rest) for c in rec["cands"])
    f["k32_levels"] = ",".join(fmt(p[0]) for p in rec["profiles"])
    f["k32_selected"] = f"{rec['sel']} (ambition {fmt(rec['cands'][rec['sel'] - 1][1])})"
    sel = rec["cands"][rec["sel"] - 1][0]
    f["k32_selected_reserve"] = fmt(sel.reserve())
    h = sel.pairs[0][0]
    f["ex31_head_end"] = f"{fmt(h.p1)},{fmt(h.v1)}"
    n2 = sum(1 for seg, _ in sel.pairs[1:] if seg.u == -2)
    seg22 = sel.pairs[n2][0]
    f["ex31_after_firm_braking"] = f"{n2} periods,{fmt(seg22.p1)},{fmt(seg22.v1)}"
    trial = Segment(seg22.p1, seg22.v1, Fr(-2))
    f["ex31_trial_segment"] = (f"{fmt(trial.p1)},label {fmt(b_min(trial.lo, trial.hi))},"
                               f"period {sel.j0 + n2 + 1}")
    s23 = sel.pairs[n2 + 1][0]
    f["ex31_on_ramp"] = f"{fmt(s23.u)},{fmt(s23.p1)},{fmt(s23.v1)}"
    f["ex31_rest"] = f"{fmt(sel.p_rest)} = {fmt(s23.p1 + s23.v1 ** 2)}"
    half = rec["cands"][2][0]
    m2 = sum(1 for seg, _ in half.pairs[1:] if seg.u == -2)
    a, b = half.pairs[m2][0], half.pairs[m2 + 1][0]
    f["ex31_half"] = (f"{m2} periods,{fmt(a.p1)},{fmt(a.v1)};{fmt(b.p1)},{fmt(b.v1)};"
                      f"rest {fmt(half.p_rest)} = {fmt(b.p1 + b.v1 ** 2)}")
    rec140 = kept[140]
    plan140 = rec140["cands"][0][0]
    f["k140_ambition1_at_rest_s"] = r((plan140.j0 + len(plan140.pairs)) * T, 1)
    rec = kept[179]
    alone = rec["cands"][5][0]
    back = [seg.u for seg, _ in alone.pairs]
    f["backaway_maneuver"] = (f"{back.count(Fr(-1, 2))} at -0.5, {back.count(Fr(2))} at 2, "
                              f"{len(back)} periods, rest {fmt(alone.p_rest)}")
    seg16 = alone.pairs[15][0]
    f["backaway_after_drive"] = f"{fmt(alone.pairs[0][0].p0 - seg16.p1)} m,{fmt(-seg16.v1)} m/s"
    first = rec["cands"][0]
    prof1 = rec["profiles"][0]
    onset = rec["onset"]
    seg_on = first[0].pairs[onset][0]
    f["k179_ambition1"] = (f"onset level {fmt(level_at(prof1, onset))},"
                           f"at the onset {fmt(seg_on.p0)}")
    f["k179_other_blends"] = ",".join(fmt(level_at(p, onset)) for p in rec["profiles"][1:5])
    first_back = next(i for i, (seg, _) in enumerate(first[0].pairs) if seg.u < 0)
    f["k179_backaway_start_periods"] = (
        f"ambition 1: {first[0].j0 + first_back}, alone: {alone.j0}")
    f["k179_alone_and_fallback"] = (f"{fmt(level_at(rec['profiles'][5], onset))},"
                                    f"{fmt(level_at(rec['fb_prof'], onset))},selected {rec['sel']}")
    return f


EXPECTED = {   # every simulation result quoted in Chapter 3, as in the text
    "forecasting": {
        "arrival_s": "8.6", "counts_1_quarter_0": "32,4,50,86",
        "load": "53/86 = 0.62", "accepted_first": "32",
        "first_modified_period": "33", "state_when_acceptance_ends": "10.24,6.4",
        "first_plan_rest": "0.02", "last_accepted_plan_rest": "21.95",
        "partial_periods_before_arrival": "33,46,55,83",
        "ramp_entry": "2.099 m/s at 5.60 s", "vmax": "6.4",
        "rest": "24.500 m at 10.0 s", "hold_accepted_from_s": "9.9",
        "min_level": "0.50", "min_reserve": "0.500", "alarms": "0",
        "fallbacks_before_arrival": "0", "requirement_issued_in_period": "140",
        "backaway_start_s": "18.0", "hold_after_announcement_s": "4.0",
        "onset_state": "23.700,0.00", "met_at_onset": "True",
        "unfolded_from_s": "20.1", "level_after_unfolding": "0.50",
        "alpha_after_unfolding": "1", "backaway_fallbacks": "14 of 20",
        "backaway_braking_accepted": "196,197,198,199",
        "onset_level_held_from_announcement": "1.3",
        "level_held_while_holding": "0.5",
        "ambition_while_holding": "1",
        "commands_of_partial_blends": "-1,-1,-1,0.125",
        "full_acceleration_periods": "1-32",
        "max_p_plus_ell": "24.5", "alarm_checks": "220",
        "recoverable_at_every_boundary": "True",
    },
    "fixed": {
        "arrival_s": "9.9", "counts_1_quarter_0": "22,1,76,99",
        "load": "307/396 = 0.78", "accepted_first": "22",
        "first_modified_period": "23", "state_when_acceptance_ends": "4.84,4.4",
        "partial_periods_before_arrival": "64",
        "ramp_entry": "2.120 m/s at 6.98 s", "vmax": "4.4",
        "rest": "24.495 m at 11.3 s", "min_level": "0.50",
        "min_reserve": "0.505", "alarms": "0",
        "fallbacks_before_arrival": "0", "backaway_start_s": "18.0",
        "onset_state": "23.695,0.00", "met_at_onset": "True",
        "unfolded_from_s": "20.1", "backaway_fallbacks": "14 of 20",
        "commands_of_partial_blends": "0.125",
    },
    "persisting": {
        "accepted_unchanged": "35",
        "acceleration_ends": "12.25 m, 7.0 m/s at 3.6 s",
        "first_invalid_plan_committed_s": "3.2",
        "invalid_labels_in_every_later_plan": "True",
        "min_level": "0.50",
        "point_of_no_return": "3.36 s, 10.6 m, 6.52 m/s",
        "unrecoverable_after_point_of_no_return": "True",
        "first_executed_invalid_label": "period 49",
        "last_plan_committed_s": "4.9",
        "last_plan_pairs": "22, commands -2, labels 2",
        "last_plan_rest_level": "24.5,0.5",
        "last_plan_invalid_labels": "22 of 22",
        "last_plan_head": "-2, label 2, 19.66 to 20.09 m, period 49",
        "last_plan_head_start": "19.66,4.4",
        "levels_while_braking": "0.5",
        "divergence": "4.979 s at 4.243 m/s",
        "stop_distance_at_ramp": "18.00 m",
        "alarm": "5.0 s; measured 20.0903 m, 4.2320 m/s; "
                 "initial set 20.09 m, 4.2 m/s",
        "alarm_difference": "0.34 mm, 0.032 m/s",
        "certified_restart": "False; needs 17.91 m, 4.41 m available",
        "wall_reached": "6.25 s at 3.61 m/s",
        "floor_lost": "6.12 s at 3.67 m/s",
        "stops_at": "38.00 m at 13.46 s",
    },
    "candidates": {
        "k31_selected_rest": "21.95", "k32_state": "9.61,6.2",
        "k32_initial_set": "10.24,6.4", "k32_proposed": "2",
        "k32_fallback": "-2,21.95,3.05", "k32_heads": "2,1,0,-1,-2,-2",
        "k32_head_labels": "2,2,2,2,2,2",
        "k32_rests": "27.66,26.135,24.75,23.505,21.95,21.95",
        "k32_levels": "-inf,-inf,-inf,1.49,3.05,3.05",
        "k32_selected": "4 (ambition 0.25)", "k32_selected_reserve": "1.495",
        "ex31_head_end": "10.875,6.3",
        "ex31_after_firm_braking": "22 periods,19.895,1.9",
        "ex31_trial_segment": "20.075,label 0.5,period 56",
        "ex31_on_ramp": "-0.5,20.0825,1.85", "ex31_rest": "23.505 = 23.505",
        "ex31_half": "21 periods,19.91,2.2;20.1275,2.15;rest 24.75 = 24.75",
        "backaway_maneuver": "16 at -0.5, 4 at 2, 20 periods, rest 23.7",
        "backaway_after_drive": "0.64 m,0.8 m/s",
        "k179_ambition1": "onset level 1.29,at the onset 23.71",
        "k179_other_blends": "1.26,1.26,1.26,1.26",
        "k179_alone_and_fallback": "1.3,1.3,selected 6",
        "k179_backaway_start_periods": "ambition 1: 181, alone: 180",
        "k140_ambition1_at_rest_s": "16.2",
    },
    "comparison": {
        "reactive_equals_fixed": "True", "arrival_earlier": "1.3 s, 13 per cent",
        "ramp_speed_limit": "2.121", "ramp_speed_wall_limit": "2.236",
        "lead_time_fixed": "2.1 s met, 2 s not met",
        "lead_time_reactive": "2.1 s met, 2 s not met",
        "lead_time_forecasting": "2.1 s met, 2 s not met",
        "direct_execution": "9.75 m/s at 5.0 s",
    },
}


def main():
    ap = argparse.ArgumentParser(description="Chapter 3: the smallest complete example")
    ap.add_argument("--period", type=int, action="append", default=[],
                    help="print the candidates of this period (forecasting run)")
    ap.add_argument("--check", action="store_true",
                    help="check every simulation result quoted in Chapter 3")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "results")
    os.makedirs(out, exist_ok=True)

    facts, traces = {}, {}
    keep = tuple(args.period) + (31, 32, 140, 178, 179)
    for kind in ("fixed", "reactive", "forecasting"):
        rows, kept, info = run(kind, keep_candidates=keep if kind == "forecasting" else ())
        write_trace(os.path.join(out, f"trace-{kind}.csv"), rows)
        write_plot(os.path.join(out, f"plot-{kind}.csv"), rows)
        traces[kind] = [{c: x.get(c) for c in TRACE_COLS} for x in rows]
        facts[kind] = facts_of_run(kind, rows, kept, info)
        if kind == "forecasting":
            facts["candidates"] = facts_of_candidates(kept)
            for kk in sorted(kept):
                candidate_table(os.path.join(out, f"candidates-k{kk}.csv"), kept[kk])
                if kk in args.period:
                    print_period(kk, kept[kk])
    facts["comparison"] = {
        "reactive_equals_fixed": str(traces["reactive"] == traces["fixed"]),
        "arrival_earlier": (
            f"{r(Fr(facts['fixed']['arrival_s']) - Fr(facts['forecasting']['arrival_s']), 1)} s, "
            f"{r(100 * (1 - Fr(facts['forecasting']['arrival_s']) / Fr(facts['fixed']['arrival_s'])), 0)} per cent"),
        "ramp_speed_limit": r(math.sqrt(2 * 0.5 * 4.5), 3),
        "ramp_speed_wall_limit": r(math.sqrt(2 * 0.5 * 5), 3),
    }
    # shortest lead time at which the operation begins at its onset
    for kind in ("fixed", "reactive", "forecasting"):
        res = []
        for na in (21, 20):
            _, _, info = run(kind, t_announce=T_ONSET - na * T)
            res.append(f"{fmt(na * T)} s {'met' if info['k_met'] == info['kd'] else 'not met'}")
        facts["comparison"][f"lead_time_{kind}"] = ", ".join(res)
    t_hit, v_hit = direct_execution()
    facts["comparison"]["direct_execution"] = f"{r(v_hit, 2)} m/s at {r(t_hit, 1)} s"
    # the persisting predictor (Section 3.3): the run stops at the alarm
    rows, _, info = run("persisting")
    write_trace(os.path.join(out, "trace-persisting.csv"), rows)
    certified, after = continue_after_alarm(info["state"])
    write_track(os.path.join(out, "track-persisting.csv"),
                alarm_track(rows, info, after))
    facts["persisting"] = facts_of_persisting(rows, info, certified, after)

    lines, failures = [], []
    for group, items in facts.items():
        lines.append(f"[{group}]")
        for key, val in items.items():
            lines.append(f"  {key} = {val}")
            want = EXPECTED.get(group, {}).get(key)
            if args.check and want is not None and want != val:
                failures.append(f"{group}.{key}: expected {want!r}, got {val!r}")
    if args.check:                       # every quoted value must be computed
        failures += [f"{group}.{key}: not computed"
                     for group, items in EXPECTED.items() for key in items
                     if key not in facts.get(group, {})]
    if args.check:
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
