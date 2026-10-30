"""The two control packages of the complete case (Section 16.4): tracking
segments with Lyapunov tubes, for the declared model of model.py.

complete.tube   the tube of Section 5.1 in the quadratic form P: every
                segment carries the same invariant level, phi (phi + iota
                once the tool unfolds), and the initial set of the next plan
                is the end of the executed funnel (Definition 4.15).
complete.mpc    the robust MPC tube of Section 5.3 in the form M: the tube of
                each new plan starts from the measured state, predicted over
                the period of computation, with a scale that grows by
                s+ = rho^100 s + (1 - rho^100) phi from pair to pair; the
                initial set is the predicted set within the end of the
                executed funnel.

Both execute the same law, u = a - 8 (p_meas - p_ref) - 4 (v_meas - v_ref),
around references with a constant acceleration per period, and both share
one generator: the proposal blended with the head of the fallback for the
ambitions 1, 3/4, 1/2, 1/4 and 0, each continued by the recovery
continuation, and the recovery continuation alone. The recovery
continuation brakes the reference as hard as its labels allow until it is
at rest, backs away towards the start if a demand is pending and the rest
point leaves less than the required reserve, and ends with the rest
segment: the same tracking law around a reference at rest, whose tube is
invariant. The test checks, for every pair, that the command stays within
the bound and the label's braking on the whole funnel, that a label of the
global capability is used only at speeds of at most V_C, that the funnels
chain, and the clearance 25 m - max(p + l) over the funnel; it computes in
floating point and rounds every quantity outward by TOL.

Labels are the braking of the capability space complete.braking, 2 m/s^2
('level ground') or 0.5 m/s^2 (the global capability).
"""
import math
from fractions import Fraction as Fr

import allovi
from allovi import NEG_INF, Candidate, Profile, Why, register
from allovi.schema import Actuation, Bound, Quantity, Schema

from . import model as md

ALPHAS = (1.0, 0.75, 0.5, 0.25, 0.0)
LEVELS = allovi.ScalarLevels(floor=md.EPS_MIN, step=md.GRID, top=md.EPS_TOP)

STATE = Schema(
    "cart state",
    (Quantity("p", "m", "track"),
     Quantity("v", "m/s", "track"),
     Quantity("l", "m", "tool")),
    norm="max(|p| + |l|, |v| * 1 s)")

ACTUATION = Actuation(
    inputs=(Quantity("u_a", "m/s^2", "track"),
            Quantity("u_l", "m", "tool")),
    bounds=(Bound("u_a", low=-2.0, high=2.0),
            Bound("u_l", values=(0.0, md.ELL))),
    hold="zero-order hold over each time step of 1 ms of the simulator",
    meaning="u_a: commanded acceleration of the cart along the track, realized "
            "within the bounds of the declared model while it lies in [-beta, 2] "
            "m/s^2, beta the braking of the label in force; u_l: commanded reach "
            "of the tool")

_LEVEL_CACHE = {}


def level_below(c):
    """The largest level of the grid not above the clearance c [m], rounded
    down by TOL; NEG_INF below the floor."""
    x = (c - 0.5 - md.TOL) * 100.0
    if x < 0:
        return NEG_INF
    n = min(2450, int(math.floor(x)))
    lv = _LEVEL_CACHE.get(n)
    if lv is None:
        lv = _LEVEL_CACHE[n] = md.EPS_MIN + n * md.GRID
    return lv


class Braking:
    """The capability space: a label is the braking [m/s^2] that the ground
    provides at least on the region of a segment. 2: the region is on level
    ground; 0.5: the global capability, which also asserts the declared
    model only at speeds up to V_C."""
    NAME = "complete.braking"
    global_capability = md.B_GLOBAL

    def informative(self, c, d):
        return c >= d

    def labelling(self, capability):
        return lambda j, segment: capability


# ----------------------------------------------------------------------------
# Segments, rest segments and plans
# ----------------------------------------------------------------------------
class Seg:
    """A tracking segment of one period: the reference from (p0, v0) with the
    constant acceleration a on the simulator's grid, the commanded reach ell,
    the tool's countdown (tool: 0 if the tool rests at ell; n >= 1 if it is
    moving to ell and arrives within n periods, this one included), and the
    tube's levels at the start (s0) and the end (s1) of the period; sbar, the
    level on which its bounds are taken, with the factors fp and fv of the
    package's form. Derived: p1, v1, the reference at the end; plo, phi,
    vlo, vhi, its extremes over the period; f_plo, f_phi, f_vlo, f_vhi, those
    of the funnel, the segment's region, to which its label refers; unfold,
    whether the tool moves in the period. impulse: the period lies in the
    window of the tool's impulse, the periods in which the tool moves and the
    next one."""
    __slots__ = ("j", "p0", "v0", "a", "ell", "tool", "unfold", "s0", "s1", "sbar",
                 "p1", "v1", "plo", "phi", "vlo", "vhi", "rest",
                 "f_plo", "f_phi", "f_vlo", "f_vhi", "impulse")

    def __init__(self, j, p0, v0, a, ell, tool, s0, s1, fp, fv, rest=False, impulse=None):
        self.j, self.p0, self.v0, self.a = j, p0, v0, a
        self.ell, self.tool, self.s0, self.s1 = ell, int(tool), s0, s1
        self.unfold = self.tool >= 1
        self.impulse = self.unfold if impulse is None else impulse
        self.sbar = max(s0, s1) * float(md.MARGIN)
        self.rest = rest
        n, h = md.N_STEPS, md.H
        self.v1 = v0 + n * h * a
        if abs(self.v1) < 1e-12:
            self.v1 = 0.0
        self.p1 = p0 + v0 * n * h + a * h * h * n * (n + 1) / 2
        cands = [0, n]
        if a != 0:
            ic = -(v0 / (a * h)) - 0.5             # vertex of p_ref(i)
            for i in (math.floor(ic), math.ceil(ic)):
                if 0 < i < n:
                    cands.append(i)
        ps = [p0 + v0 * i * h + a * h * h * i * (i + 1) / 2 for i in cands]
        self.plo, self.phi = min(ps), max(ps)
        self.vlo, self.vhi = min(v0, self.v1), max(v0, self.v1)
        # the funnel's extents, the region to which a label refers
        ep, ev = fp * self.sbar, fv * self.sbar
        self.f_plo, self.f_phi = self.plo - ep - md.TOL, self.phi + ep + md.TOL
        self.f_vlo, self.f_vhi = self.vlo - ev - md.TOL, self.vhi + ev + md.TOL

    def ref(self, i):
        """The reference (p, v) at time step i of the period."""
        h = md.H
        return (self.p0 + self.v0 * i * h + self.a * h * h * i * (i + 1) / 2,
                self.v0 + i * h * self.a)

    def l_range(self, e_l):
        """The reach on the funnel: anywhere between folded and unfolded while
        the tool moves, else within e_l of its command."""
        if self.tool >= 1:
            return (-e_l, md.ELL + e_l)
        return (self.ell - e_l, self.ell + e_l)

    def l_end(self, e_l):
        """The reach at the end of the period: within e_l of the command once
        the tool has arrived, at the latest at the end of its last period."""
        if self.tool >= 2:
            return (-e_l, md.ELL + e_l)
        return (self.ell - e_l, self.ell + e_l)


def next_tool(tool):
    """The tool's countdown in the next period."""
    return tool - 1 if tool >= 1 else 0


class Plan:
    """A labelled plan: pairs (Seg, label), executed in periods j0, j0 + 1,
    ..., followed by the rest segment 'rest' (a Seg with a = 0 at rest,
    repeated for ever). The reach of every segment is 'ell'."""
    __slots__ = ("j0", "pairs", "rest")

    def __init__(self, j0, pairs, rest):
        self.j0, self.pairs, self.rest = j0, pairs, rest

    def shift(self):
        if not self.pairs:
            return self
        return Plan(self.j0 + 1, self.pairs[1:], self.rest)

    def head(self):
        return self.pairs[0][0] if self.pairs else self.rest

    def head_label(self):
        return self.pairs[0][1] if self.pairs else md.B_GLOBAL


class Initial:
    """An initial set: the states x_ref + e with V(e) <= s in the package's
    form, for the reference point (p, v), and reach in [l_lo, l_hi]; for the
    MPC package also 'within', the end of the executed funnel (another such
    set), of which this set is the part that the prediction allows; folded,
    whether the tool is folded and at rest; tool, its countdown for the next
    head; after, whether the executed head was the last period in which the
    tool moved, so that the next period still lies in the window of the
    tool's impulse."""
    __slots__ = ("p", "v", "s", "l_lo", "l_hi", "within", "folded", "after", "tool")

    def __init__(self, p, v, s, l_lo, l_hi, folded, within=None, after=False, tool=0):
        self.p, self.v, self.s = p, v, s
        self.l_lo, self.l_hi, self.folded = l_lo, l_hi, folded
        self.within = within
        self.after = after            # the executed head was the tool's last moving period
        self.tool = tool              # the tool's countdown for the next head


class TrackLaw:
    """The tracking law of a segment, for the simulator: called with the time
    tau within the period and the measured state, it returns (u_a, u_l). It
    carries its segment and the package's form q, for the records."""
    __slots__ = ("seg", "kind", "q")

    def __init__(self, seg, kind="segment", q=None):
        self.seg, self.kind, self.q = seg, kind, q

    def __call__(self, tau, meas):
        s = self.seg
        i = int(round(tau / md.H))
        pr, vr = s.ref(i)
        return (s.a - md.KP * (meas[0] - pr) - md.KV * (meas[1] - vr), s.ell)


class RestFeedback:
    """The rest feedback after an alarm, without a guarantee: at speeds above
    5 cm/s, a command of 0.4 m/s^2 against the motion; from then on, the
    tracking law around rest at the position where it slowed below that
    speed, or at the measured position if the period starts below it."""

    kind = "rest"

    def __init__(self, measured, ell):
        self.hold = measured[0] if abs(measured[1]) <= 0.05 else None
        self.ell = ell

    def __call__(self, tau, meas):
        if self.hold is None and abs(meas[1]) <= 0.05:
            self.hold = meas[0]
        if self.hold is None:
            return (-math.copysign(0.4, meas[1]), self.ell)
        return (-md.KP * (meas[0] - self.hold) - md.KV * meas[1], self.ell)


# ----------------------------------------------------------------------------
# The packages
# ----------------------------------------------------------------------------
class _Tracking:
    """What the two packages share; a subclass fixes the form and the
    bookkeeping of the tube's levels."""
    ROLE, CONTRACT = "control", "1.1"
    STATE, ACTUATION = STATE, ACTUATION
    FORM = None
    INITIAL_FROM_MEASUREMENT = False

    def __init__(self, frozen=True, bounds=None):
        self.b = dict(bounds) if bounds is not None else md.bounds(frozen)
        self.c = md.certificate(self.FORM, self.b)
        c, b = self.c, self.b
        self.q = c["q"]
        self.phi = c["phi"]
        self.iota = c["iota"]
        self.e_l = b["e_l"]
        self.capabilities, self.levels = Braking(), LEVELS
        self.varpi = max(b["varpi_p"] + b["varpi_l"], b["varpi_v"])

    # --- the bounds of a level ------------------------------------------------
    def ep(self, s):
        return self.c["fp"] * s

    def ev(self, s):
        return self.c["fv"] * s

    def eu(self, s):
        """The bound on |u - a|: the feedback on the error, and on the sensor's error."""
        return self.c["fu"] * s + self.c["noise"]

    def a_max(self, sbar):
        """The largest reference acceleration of a segment whose bounds are
        taken on the level sbar."""
        return md.U_MAX - self.eu(sbar)

    def brake_max(self, sbar, beta):
        """The largest reference braking under the label beta, likewise."""
        return min(beta, md.U_MAX) - self.eu(sbar)

    def declared(self):
        """Constants for proposal methods (what a learning researcher may read
        of the package): the largest reference acceleration and braking under
        each label, the speed of the global claim, the reference's retreat,
        and the bounds on the position error and on the speed error, all at
        the package's steady level."""
        s = self.steady_bound_level()
        return dict(a_max=self.a_max(s), b_firm=self.brake_max(s, md.B_FIRM),
                    b_global=self.brake_max(s, md.B_GLOBAL), v_c=md.V_C,
                    retreat=-self.brake_max(s, md.B_GLOBAL), e_p=self.ep(s),
                    e_v=self.ev(s), e_l=self.e_l)

    # --- certification of one segment ----------------------------------------
    def seg_ok(self, seg, beta):
        """Conditions (i) of a pair under the label beta (Definition 4.4):
        the command within the bound and the label's braking on the whole
        funnel, and, under the global capability, speeds of at most V_C."""
        eu = self.eu(seg.sbar)
        if seg.a + eu > md.U_MAX + 1e-12 or seg.a - eu < -min(beta, md.U_MAX) - 1e-12:
            return False
        if beta < md.B_FIRM and max(abs(seg.f_vlo), abs(seg.f_vhi)) > md.V_C:
            return False
        return True

    def clearance(self, seg):
        """25 m less the largest p + l on the funnel."""
        return md.WALL - (seg.f_phi + seg.l_range(self.e_l)[1]) - md.TOL

    def profile(self, plan, initial):
        """The verdicts of the test as a profile (Definition 4.9)."""
        if not self.starts_in(plan, initial):
            return Profile.uncertified()
        pairs, rest = plan.pairs, plan.rest
        n = len(pairs)
        out = [NEG_INF] * (n + 1)
        ok = self.rest_ok(rest) and (n == 0 or self.chains(pairs[-1][0], rest))
        running = self.clearance(rest)
        out[n] = level_below(running) if ok else NEG_INF
        for i in range(n - 1, -1, -1):
            seg, beta = pairs[i]
            ok = ok and self.seg_ok(seg, beta)
            if i + 1 < n:
                ok = ok and self.chains(seg, pairs[i + 1][0])
            running = min(running, self.clearance(seg))
            out[i] = level_below(running) if ok else NEG_INF
        return Profile(out)

    def explain(self, plan, initial):
        if not self.starts_in(plan, initial):
            return [Why("the plan does not start at the initial set", "pair")]
        prof = list(self.profile(plan, initial))
        return [Why("certified" if x != NEG_INF else "refused",
                    "certified" if x != NEG_INF else "pair") for x in prof]

    def rest_ok(self, rest):
        if rest.v0 != 0.0 or rest.a != 0.0 or rest.s0 + 1e-15 < self.phi_of(rest):
            return False
        return self.seg_ok(rest, md.B_GLOBAL)

    def chains(self, a, b):
        """(P3) and (P4): the reference continues and the level does not drop;
        the reach at the end lies within the next segment's reach at its start."""
        if a.p1 != b.p0 or a.v1 != b.v0 or a.s1 > b.s0 + 1e-15:
            return False
        lo, hi = a.l_end(self.e_l)
        blo, bhi = b.l_range(self.e_l)
        return blo <= lo and hi <= bhi

    def phi_of(self, seg):
        """The invariant level for a segment: phi, raised by iota once the
        tool unfolds."""
        return self.phi + (self.iota if seg.ell > 0 else 0.0)

    # --- execution and the alarm ------------------------------------------------
    def law(self, plan):
        return TrackLaw(plan.head(), "segment" if plan.pairs else "rest segment", self.q)

    def distance(self, measured, initial):
        """The distance, in the norm of the state, from the measurement to the
        initial set; for the MPC package, the larger of the distances to the
        prediction and to the end of the executed funnel, a lower bound of
        the distance to their intersection."""
        d = self._dist_one(measured, initial)
        if initial.within is not None:
            d = max(d, self._dist_one(measured, initial.within))
        return d

    def _dist_one(self, x, st):
        dl = max(0.0, st.l_lo - x[2], x[2] - st.l_hi)
        cp, cv = x[0] - st.p, x[1] - st.v
        if md.norm_q(self.q, cp, cv) <= st.s:
            return dl
        lo, hi = dl, dl + abs(cp) + abs(cv) + 1.0
        for _ in range(60):
            t = (lo + hi) / 2
            if _box_meets_ellipse(self.q, cp, cv, t - dl, t, st.s):
                hi = t
            else:
                lo = t
        return hi

    # --- after an alarm -----------------------------------------------------------
    def rest_law(self, measured):
        ell = md.ELL if measured[2] > md.ELL / 2 else 0.0
        return RestFeedback(measured, ell)

    def reach(self, measured, bound):
        """At (near) rest, the tube around rest at the measured position,
        which the rest feedback holds; while the cart moves, no set."""
        p, v, l = measured
        s = self.steady_level(l > md.ELL / 2)
        bp, bv = self.b["varpi_p"], abs(v) + self.b["varpi_v"]
        if abs(v) > 0.05 or md.box_level(self.q, bp, bv) > s:
            return None, "moving: the rest feedback brakes the cart first"
        ell = md.ELL if l > md.ELL / 2 else 0.0
        if abs(l - ell) > self.e_l + self.b["varpi_l"]:
            return None, "the tool moves"
        return Initial(p, 0.0, s, ell - self.e_l, ell + self.e_l, ell == 0.0), ""

    def restart(self, step, reach):
        return self._candidates(step, reach, None)

    # --- the initial plan ------------------------------------------------------------
    def initial_plan(self, measured, bound):
        p, v, l = measured
        s = self.steady_level(False)
        if md.box_level(self.q, bound, abs(v) + bound) > s or l > md.ELL / 2:
            raise ValueError("this package certifies an initial plan only for a cart "
                             "at rest, its tool folded, within its tube")
        if abs(l) > self.e_l + bound:
            raise ValueError("the measured reach is not that of a folded tool")
        rest = self.rest_seg(0, p, 0.0)
        init = Initial(p, 0.0, s, -self.e_l, self.e_l, True)
        return Plan(0, [], rest), init

    # --- generation -------------------------------------------------------------------
    def generate(self, step, plan, initial):
        return self._candidates(step, initial, plan.shift())

    def _candidates(self, step, initial, fallback):
        k = step.k
        req = step.requirement.level
        ell = 0.0 if not step.operation else md.ELL
        if ell > 0 and initial.folded:
            tool = md.TOOL_PERIODS             # the tool starts to unfold
        else:
            tool = initial.tool
        imp = tool >= 1 or initial.after       # the head lies in the impulse's window
        p, v, s0 = self.head_start(initial)
        target = None
        if req > md.EPS_MIN and ell == 0.0:
            # the rest point from which the plan holds the requirement with the
            # tool folded and, once it unfolds, the floor: the unfolded tube's
            # bound, which is the larger
            s_r = self.rest_bound_level(True)
            target = md.WALL - float(req) - self.ep(s_r) - self.e_l - 2 * md.TOL
        lab = step.labelling
        if step.proposal is None:
            return [self._recovery_only(lab, k, p, v, s0, ell, tool, target, imp)]
        a_fb = 0.0
        if fallback is not None and fallback.pairs:
            a_fb = fallback.pairs[0][0].a
        a_q = self.meaning(step.proposal, p, v, s0, lab, k, ell, tool, imp)
        cands = []
        for alpha in ALPHAS:
            a = alpha * a_q + (1 - alpha) * a_fb
            head = self.seg(k + 1, p, v, a, ell, tool, s0, imp)
            pairs = [(head, lab(k + 1, head))]
            tail, rest = self.recovery(lab, k, k + 2, head.p1, head.v1, head.s1, ell,
                                       target, next_tool(tool), tool == 1)
            cands.append(Candidate(Plan(k + 1, pairs + tail, rest), alpha))
        cands.append(self._recovery_only(lab, k, p, v, s0, ell, tool, target, imp))
        return cands

    def _recovery_only(self, lab, k, p, v, s0, ell, tool, target, imp=False):
        """The recovery continuation alone, with ambition 0; a plan without
        pairs if the reference is at rest and the tool too."""
        tail, rest = self.recovery(lab, k, k + 1, p, v, s0, ell, target, tool, imp)
        return Candidate(Plan(k + 1, tail, rest), 0.0)

    def meaning(self, q, p, v, s, lab, k, ell, tool, imp=False):
        """The reference acceleration that a proposal requests at the start
        of the head: 'accelerate', the largest; 'hold', a stop within the
        period, or as near to it as the labels allow; 'retreat', the largest
        acceleration towards the start that the global capability allows; a
        number, that acceleration within the bounds of 'level ground'."""
        sb = self.seg_sbar(s, imp, ell)
        if q == "accelerate":
            return self.a_max(sb)
        if q == "retreat":
            return -self.brake_max(sb, md.B_GLOBAL)
        if q == "hold" or not isinstance(q, (int, float, Fr)):
            return self._brake_command(lab, k, k + 1, p, v, s, ell, tool, imp)
        return max(-self.brake_max(sb, md.B_FIRM), min(self.a_max(sb), float(q)))

    def _brake_command(self, lab, k, j, p, v, s, ell, tool, imp=False):
        """The command that stops the reference within the period, limited by
        the braking that the segment's own label allows (strongest first)."""
        if v == 0.0:
            return 0.0
        sb = self.seg_sbar(s, imp, ell)
        if v < 0:
            return min(-v / md.T, self.a_max(sb))
        for beta in (md.B_FIRM, md.B_GLOBAL):
            a = -min(v / md.T, self.brake_max(sb, beta))
            seg = self.seg(j, p, v, a, ell, tool, s, imp)
            if lab(j, seg) >= beta:
                return a
        return a

    def recovery(self, lab, k, j, p, v, s, ell, target, tool=0, imp=False):
        """The recovery continuation from the reference (p, v) with level s at
        the start of period j: braking to rest, holding at rest until the
        tool has arrived (its countdown 'tool'), a back-away if 'target' is
        given and the rest point lies beyond it, and the rest segment; imp,
        whether its first period lies in the window of the tool's impulse.
        Returns (pairs, rest)."""
        pairs = []
        p, v, s, j, tool, imp = self._to_rest(lab, k, j, p, v, s, ell, pairs, tool, imp)
        while tool >= 1 or imp:              # hold until the tool's window has passed
            seg = self.seg(j, p, 0.0, 0.0, ell, tool, s, imp)
            pairs.append((seg, lab(j, seg)))
            s, j, imp, tool = seg.s1, j + 1, tool == 1, next_tool(tool)
        if target is not None and p > target:
            back = self._back_away(lab, k, j, p, s, ell, target)
            if back is not None:
                pairs += back
                last = back[-1][0]
                p, v, s, j = last.p1, last.v1, last.s1, j + len(back)
        return pairs, self.rest_seg(j, p, s, ell)

    def _to_rest(self, lab, k, j, p, v, s, ell, pairs, tool=0, imp=False):
        guard = 0
        while v != 0.0:
            a = self._brake_command(lab, k, j, p, v, s, ell, tool, imp)
            seg = self.seg(j, p, v, a, ell, tool, s, imp)
            pairs.append((seg, lab(j, seg)))
            p, v, s, j = seg.p1, seg.v1, seg.s1, j + 1
            imp, tool = tool == 1, next_tool(tool)
            guard += 1
            if guard > 2000:
                raise ArithmeticError("the reference does not come to rest")
        return p, v, s, j, tool, imp

    def _back_away(self, lab, k, j, p, s, ell, target):
        """m periods of the retreat from rest at p, then braking of the
        backward motion, with the smallest m that comes to rest at or before
        'target' (the rest point moves back as m grows, so m is found by
        doubling and bisection); None if 300 periods do not suffice."""
        def trial(m):
            pairs, q, w, ss, jj = [], p, 0.0, s, j
            for _ in range(m):
                sb = self.seg_sbar(ss, False, ell)
                seg = self.seg(jj, q, w, -self.brake_max(sb, md.B_GLOBAL), ell, 0, ss)
                pairs.append((seg, lab(jj, seg)))
                q, w, ss, jj = seg.p1, seg.v1, seg.s1, jj + 1
            q, w, ss, jj, _, _ = self._to_rest(lab, k, jj, q, w, ss, ell, pairs)
            return pairs, q

        lo, hi = 0, 1
        while True:
            pairs, q = trial(hi)
            if q <= target:
                break
            lo, hi = hi, 2 * hi
            if hi > 300:
                return None
        while hi - lo > 1:
            mid = (lo + hi) // 2
            mp, mq = trial(mid)
            if mq <= target:
                hi, pairs = mid, mp
            else:
                lo = mid
        return pairs


def _box_meets_ellipse(q, cp, cv, rp, rv, s):
    """True if the box [cp - rp, cp + rp] x [cv - rv, cv + rv] contains a
    point e with e'Qe <= s^2: the least of e'Qe on the box, on its edges or
    inside."""
    if rp < 0 or rv < 0:
        return False
    q11, q12, q22 = q
    lo_p, hi_p, lo_v, hi_v = cp - rp, cp + rp, cv - rv, cv + rv
    if lo_p <= 0 <= hi_p and lo_v <= 0 <= hi_v:
        return True
    best = math.inf
    for ep in (lo_p, hi_p):                       # edges at fixed p: minimize over v
        ev = min(hi_v, max(lo_v, -q12 * ep / q22))
        best = min(best, q11 * ep * ep + 2 * q12 * ep * ev + q22 * ev * ev)
    for ev in (lo_v, hi_v):                       # edges at fixed v: minimize over p
        ep = min(hi_p, max(lo_p, -q12 * ev / q11))
        best = min(best, q11 * ep * ep + 2 * q12 * ep * ev + q22 * ev * ev)
    return best <= s * s


@register
class Tube(_Tracking):
    """complete.tube: one invariant level for every segment (Section 5.1)."""
    NAME, FORM = "complete.tube", "tube"

    def steady_level(self, unfolded):
        return self.phi + (self.iota if unfolded else 0.0)

    def steady_bound_level(self):
        return (self.phi + self.iota) * float(md.MARGIN)

    def rest_bound_level(self, unfolded):
        return self.steady_level(unfolded) * float(md.MARGIN)

    def seg_sbar(self, s0, imp, ell):
        """The bound level of a segment that starts at the level s0."""
        return self.steady_level(ell > 0) * float(md.MARGIN)

    def seg(self, j, p, v, a, ell, tool, s0, imp=None):
        s = self.steady_level(ell > 0)
        return Seg(j, p, v, a, ell, tool, s, s, self.c["fp"], self.c["fv"], impulse=imp)

    def rest_seg(self, j, p, s, ell=0.0):
        lv = self.steady_level(ell > 0)
        return Seg(j, p, 0.0, 0.0, ell, 0, lv, lv, self.c["fp"], self.c["fv"], rest=True)

    def head_start(self, initial):
        return initial.p, initial.v, initial.s

    def initial_set(self, plan, measured=None, bound=None):
        """The end of the funnel of the head (Definition 4.15)."""
        seg = plan.head()
        lo, hi = seg.l_end(self.e_l)
        return Initial(seg.p1, seg.v1, seg.s1, lo, hi, seg.ell == 0.0, after=seg.tool == 1,
                       tool=next_tool(seg.tool))

    def starts_in(self, plan, initial):
        """(P1): the head starts at the initial set's reference, with at least
        its level and a reach that covers the initial set's."""
        seg = plan.head()
        lo, hi = seg.l_range(self.e_l)
        return (seg.p0 == initial.p and seg.v0 == initial.v and seg.s0 >= initial.s
                and lo <= initial.l_lo and initial.l_hi <= hi)


@register
class MPC(_Tracking):
    """complete.mpc: the tube of each new plan starts from the measured
    state, predicted over the period of computation (Section 5.3)."""
    NAME, FORM = "complete.mpc", "mpc"
    INITIAL_FROM_MEASUREMENT = True

    def __init__(self, frozen=True, bounds=None):
        super().__init__(frozen, bounds)
        self.rho_n = self.c["rho_n"]
        # the closed loop of the error over one period, A^100, as floats
        h, kp, kv = Fr(1, 1000), md.KP_FR, md.KV_FR
        a = ((1 - h * h * kp, h - h * h * kv), (-h * kp, 1 - h * kv))
        m = ((Fr(1), Fr(0)), (Fr(0), Fr(1)))
        for _ in range(md.N_STEPS):
            m = tuple(tuple(sum(m[r][t] * a[t][c] for t in range(2)) for c in range(2))
                      for r in range(2))
        self.a_n = tuple(tuple(float(x) for x in row) for row in m)
        self.s_meas = md.box_level(self.q, self.b["varpi_p"], self.b["varpi_v"])

    def grow(self, s):
        return self.rho_n * s + (1 - self.rho_n) * self.phi + 1e-12

    def steady_level(self, unfolded):
        return self.phi + (self.iota if unfolded else 0.0)

    def steady_bound_level(self):
        return (self.phi + self.iota) * float(md.MARGIN)

    def rest_bound_level(self, unfolded):
        return self.steady_level(unfolded) * float(md.MARGIN)

    def seg_sbar(self, s0, imp, ell):
        """The bound level of a segment that starts at the level s0; imp, if
        it lies in the window of the tool's impulse."""
        return max(s0, self.grow(s0) + (self.iota if imp else 0.0)) * float(md.MARGIN)

    def seg(self, j, p, v, a, ell, tool, s0, imp=None):
        imp = tool >= 1 if imp is None else imp
        s1 = self.grow(s0) + (self.iota if imp else 0.0)
        return Seg(j, p, v, a, ell, tool, s0, s1, self.c["fp"], self.c["fv"], impulse=imp)

    def rest_seg(self, j, p, s, ell=0.0):
        lv = max(self.steady_level(ell > 0), s)
        return Seg(j, p, 0.0, 0.0, ell, 0, lv, lv, self.c["fp"], self.c["fv"], rest=True)

    def head_start(self, initial):
        return initial.p, initial.v, initial.s

    def initial_set(self, plan, measured=None, bound=None):
        """The set predicted from the measured state at t_k: the head's
        closed loop over the period without disturbance moves the measured
        error to A^100 e; the sensor's error at t_k, the disturbance and the
        sensor's errors within the period, and the tool's impulse if it
        unfolds, add at most rho^100 s_meas + (1 - rho^100) phi (+ iota) in
        the form M. The set is the part of that prediction within the end of
        the executed funnel, which is kept in 'within'."""
        seg = plan.head()
        lo, hi = seg.l_end(self.e_l)
        end = Initial(seg.p1, seg.v1, seg.s1, lo, hi, seg.ell == 0.0, after=seg.tool == 1,
                      tool=next_tool(seg.tool))
        if measured is None or seg.rest:
            return end
        e0p, e0v = measured[0] - seg.p0, measured[1] - seg.v0
        a = self.a_n
        ep = a[0][0] * e0p + a[0][1] * e0v
        ev = a[1][0] * e0p + a[1][1] * e0v
        s = (self.rho_n * self.s_meas + (1 - self.rho_n) * self.phi
             + (self.iota if seg.impulse else 0.0) + 1e-9)
        if s >= seg.s1:
            return end
        return Initial(seg.p1 + ep, seg.v1 + ev, s, lo, hi, seg.ell == 0.0, within=end,
                       after=seg.tool == 1, tool=next_tool(seg.tool))

    def starts_in(self, plan, initial):
        """(P1): the head starts at the prediction with at least its scale, or
        at the end of the executed funnel with at least that funnel's level
        (the fallback); the reach covers the initial set's."""
        seg = plan.head()
        lo, hi = seg.l_range(self.e_l)
        if not (lo <= initial.l_lo and initial.l_hi <= hi):
            return False
        if seg.p0 == initial.p and seg.v0 == initial.v and seg.s0 >= initial.s:
            return True
        w = initial.within
        return (w is not None and seg.p0 == w.p and seg.v0 == w.v and seg.s0 >= w.s)
