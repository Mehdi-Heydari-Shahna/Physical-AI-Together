"""The common step of the viability governor (Algorithm 8.1, from the
generation of the candidates to the selection, and the restart search).

Nothing here depends on the robot. Plans, segments, labels and initial sets
are opaque: the core passes them between the modules and asks the control
package about them, and it never certifies anything itself. What it does is
common to every application: it forms the fallback with its shifted
profile, has the fresh candidates evaluated in the order listed until the
budget of the period is spent, admits the candidates whose level reaches the
floor, selects by the rule (4.2) with its tie-breaking, and records the
reason for every outcome. Before it uses an output of a module, it checks
that the output is of the kind that the contract states (Guard).
"""
from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .contract import Candidate, Requirement, Step
from .levels import NEG_INF, Profile

NOT_EVALUATED = "not evaluated before the deadline"


class ModuleError(RuntimeError):
    """A supplied module failed while strict checking was on."""


class ImplementationCheckFailed(RuntimeError):
    """A check found an inconsistency that points to a mistake in the
    implementation of a module, such as a fallback whose re-evaluated profile
    differs from the shifted one."""


@dataclass(frozen=True)
class Why:
    """Why a profile has its level from some pair on, as the control package
    explains it: a sentence, and whether the test refused a pair ('pair'),
    certified the plan only below the floor ('floor') or certified it
    ('certified'). Used for records only."""
    text: str
    kind: str = "certified"


@dataclass
class Entry:
    """One candidate of a period, the fallback included, as evaluated."""
    index: int                 # 0: the fallback; then the fresh candidates in order
    kind: str                  # "fallback", "fresh" or "restart"
    plan: Any
    ambition: Any
    profile: Profile
    reasons: Optional[Sequence[Why]] = None
    evaluated: bool = True


@dataclass
class Decision:
    """The outcome of the step: the committed plan with its profile and
    ambition, and the record of every candidate."""
    plan: Any
    profile: Optional[Profile]
    ambition: Any
    selected: Any              # index of the committed entry; "restart"; or None
    reason: str
    entries: List[Entry]
    keys: Dict[int, tuple]
    outcomes: Dict[int, str]
    evaluated: int             # fresh candidates evaluated before the deadline
    timed_out: bool = False    # the elapsed-time deadline stopped the evaluation


class Guard:
    """How the core treats a supplied module that fails: its call raises an
    exception, or returns an output that 'check' finds unusable, such as None
    where a list of candidates is expected or a profile that decreases. With
    strict=True, used while developing a module, the failure stops the run
    with a ModuleError that names the call and the period. With strict=False,
    as in operation, the output is replaced by its default (Section 4.3.4)
    and the event is recorded; the decision then proceeds as if the output
    were missing. The guarantees hold for the defaults without any
    assumption on the module that failed, provided that the common steps
    still finish before the deadline (Section 8.3); a failure costs progress,
    not certification. None from a proposal method or a predictor is not a
    failure: it says that the output is not ready, and the loop replaces it
    by its default in both modes."""

    def __init__(self, strict=True):
        self.strict = strict
        self.events: List[tuple] = []

    def call(self, k, what, fn, *args, default=None, check=None):
        """fn(*args), or 'default' if it fails in operation. 'check' returns
        None for a usable output, or what is wrong with it."""
        try:
            out = fn(*args)
        except Exception as exc:                       # noqa: BLE001 (by design)
            if self.strict:
                raise ModuleError(f"period {k}: {what} raised {exc!r}") from exc
            self.events.append((k, what, f"{type(exc).__name__}: {exc}"))
            return default
        if check is not None:
            try:
                problem = check(out)
            except Exception as exc:                   # noqa: BLE001 (by design)
                problem = f"an output that cannot be checked ({type(exc).__name__}: {exc})"
            if problem:
                if self.strict:
                    raise ModuleError(f"period {k}: {what} returned {problem}")
                self.events.append((k, what, f"returned {problem}"))
                return default
        return out

    def check(self, k, what, ok, detail):
        """An implementation check: stops the run if strict, else records."""
        if ok:
            return
        if self.strict:
            raise ImplementationCheckFailed(f"period {k}: {what}: {detail}")
        self.events.append((k, what, detail))


# ----------------------------------------------------------------------------
# What the core requires of the outputs that it uses
# ----------------------------------------------------------------------------
def _is_level(levels, x):
    try:
        return bool(levels.contains(x))
    except Exception:                                  # noqa: BLE001 (by design)
        return False


def profile_problem(levels):
    """A profile must be a levels.Profile whose entries are levels of the
    level space or NEG_INF, and nondecreasing in its order (Definition 4.9:
    a plan certified from pair n on is certified from pair n + 1 on)."""
    def check(out):
        if not isinstance(out, Profile):
            return f"a {type(out).__name__}, not a Profile"
        for x in out:
            if not (x == NEG_INF or _is_level(levels, x)):
                return f"a profile with the entry {x!r}, not a level"
        if not out.nondecreasing(levels.leq):
            return f"a profile that decreases, {out!r}"
        return None
    return check


def candidates_problem(out):
    """Candidates must be a list of Candidate, each with a plan that can be
    shifted and an ambition in [0, 1]."""
    if not isinstance(out, (list, tuple)):
        return f"a {type(out).__name__}, not a list of candidates"
    for i, c in enumerate(out, start=1):
        if not isinstance(c, Candidate):
            return f"a {type(c).__name__} as candidate {i}"
        if not callable(getattr(c.plan, "shift", None)):
            return f"candidate {i} with a plan that has no method shift()"
        if not (isinstance(c.ambition, numbers.Real) and 0 <= c.ambition <= 1):
            return f"candidate {i} with the ambition {c.ambition!r}, not in [0, 1]"
    return None


def requirement_problem(levels):
    """A requirement must be None (not ready) or a Requirement whose level is
    a level of the level space and whose onset is a whole number >= 0."""
    def check(out):
        if out is None:
            return None
        if not isinstance(out, Requirement):
            return f"a {type(out).__name__}, not a Requirement"
        if isinstance(out.onset, bool) or not isinstance(out.onset, int) or out.onset < 0:
            return f"a requirement with the onset {out.onset!r}"
        if not _is_level(levels, out.level):
            return f"a requirement with the level {out.level!r}, not a level"
        return None
    return check


def labelling_problem(out):
    """A labelling map must be None (not ready) or callable."""
    return None if out is None or callable(out) else \
        f"a {type(out).__name__}, not a labelling map"


def reach_problem(out):
    """The answer of reach() must be (set, reason): a set, or None when the
    package cannot represent one, with a reason as a string."""
    if not (isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], str)):
        return f"a {type(out).__name__}, not (set, reason)"
    return None


def _reasons_problem(out):
    return None if isinstance(out, (list, tuple)) else \
        f"a {type(out).__name__}, not a list of reasons"


def evaluate(control, levels, plan, initial_set, k, guard, diagnostics):
    """The profile of a plan, from the verdicts of the test; a test that
    fails counts as a refusal. With diagnostics, also the reasons."""
    prof = guard.call(k, "test", control.profile, plan, initial_set,
                      default=Profile.uncertified(), check=profile_problem(levels))
    reasons = None
    if diagnostics and hasattr(control, "explain"):
        reasons = guard.call(k, "explain", control.explain, plan, initial_set,
                             check=_reasons_problem)
    return prof, reasons


def _refusal(entry: Entry) -> str:
    if not entry.evaluated:
        return NOT_EVALUATED
    kind = entry.reasons[0].kind if entry.reasons else None
    if kind == "floor":
        return "below the floor"
    if kind == "pair":
        return "a pair not certified"
    return "not certified at the floor"


def select(entries, levels, requirement: Requirement):
    """The selection rule (4.2): among the admitted entries, the lexicographic
    maximum of (first key, ambition), ties going to fresh candidates and then
    to the smallest index. Returns the index, the outcome of every entry, the
    reason for the choice and the keys of the admitted entries."""
    admitted, keys, outcome = [], {}, {}
    for e in entries:
        if not levels.admits(e.profile.level):
            outcome[e.index] = "not admitted: " + _refusal(e)
            continue
        key1 = levels.key(e.profile.at(requirement.onset), requirement.level)
        keys[e.index] = (key1, e.ambition, 1 if e.index >= 1 else 0, -e.index)
        admitted.append(e.index)
    if not admitted:       # impossible with checked profiles: the fallback is admitted
        raise ImplementationCheckFailed("no candidate is admitted, not even the fallback")
    best = max(admitted, key=lambda i: keys[i])
    k1, a1 = keys[best][0], keys[best][1]
    same_first = [i for i in admitted if keys[i][0] == k1]
    same_both = [i for i in same_first if keys[i][1] == a1]
    for i in admitted:
        if i == best:
            outcome[i] = "committed"
        elif keys[i][0] < k1:
            outcome[i] = "admitted: lower first key"
        elif keys[i][1] < a1:
            outcome[i] = "admitted: same first key, lower ambition"
        elif i == 0:
            outcome[i] = "admitted: tie, a fresh candidate is preferred"
        else:
            outcome[i] = "admitted: tie, a smaller index is preferred"
    if best == 0:
        if len(admitted) == 1:
            why = "fallback: the only admitted candidate"
        elif len(same_first) == 1:
            why = "fallback: the only candidate with the highest first key"
        else:
            why = "fallback: highest first key, and no fresh candidate as ambitious"
    elif requirement.level == levels.floor:
        why = "no demand: the most ambitious admitted candidate"
    elif len(same_first) == 1:
        why = "demand: the only candidate with the highest first key"
    elif len(same_both) > 1:
        why = ("demand: highest first key and ambition, tie broken for a fresh"
               " candidate" if 0 in same_both else
               "demand: highest first key and ambition, tie broken for the smallest index")
    else:
        why = "demand: the most ambitious of the candidates with the highest first key"
    return best, outcome, why, keys


def _in_time(deadline, allowance, clock):
    """True if an evaluation that takes at most 'allowance' seconds, started
    now, ends by the deadline; always True without a deadline."""
    return deadline is None or clock() + allowance <= deadline


def decide(control, levels, step: Step, plan, profile: Profile, initial_set,
           budget=None, guard: Optional[Guard] = None, diagnostics=False,
           deadline=None, allowance=0.0, clock=None) -> Decision:
    """One decision during period step.k: the plan to commit at t_{k+1}.

    With a deadline, an instant of clock() (by default time.monotonic), the
    fresh candidates are evaluated in order only while an evaluation that
    takes at most 'allowance' seconds still ends by the deadline; the first
    that would not, and every later one, count as not evaluated, so that the
    fallback, which needs no evaluation, can always be committed (Section
    16.4). The count budget applies as well."""
    guard = guard or Guard()
    if deadline is not None and clock is None:
        import time
        clock = time.monotonic
    k = step.k
    fallback, fb_profile = plan.shift(), profile.shift()
    if not callable(getattr(fallback, "shift", None)):
        # the fallback has no default: an unusable one stops the run in both modes
        raise ModuleError(f"period {k}: the shift() of the plan in force returned a "
                          f"{type(fallback).__name__}, not a plan")
    fresh = guard.call(k, "generator", control.generate, step, plan, initial_set,
                       default=[], check=candidates_problem)
    n_eval = len(fresh) if budget is None else min(budget, len(fresh))
    entries = [Entry(0, "fallback", fallback, 0, fb_profile)]
    if diagnostics:
        # an implementation check, never a decision: the fallback needs no
        # evaluation, but a test that decides the inclusions exactly must give
        # it its shifted profile when it evaluates it again (Lemma 4.10(c))
        fb_check, entries[0].reasons = evaluate(control, levels, fallback, initial_set,
                                                k, guard, diagnostics)
        guard.check(k, "fallback", fb_check == fb_profile,
                    f"re-evaluated profile {fb_check} differs from the shifted "
                    f"profile {fb_profile}")
    timed_out, done = False, 0
    for i, cand in enumerate(fresh):
        if i < n_eval and not timed_out and not _in_time(deadline, allowance, clock):
            timed_out = True
        if i < n_eval and not timed_out:
            prof, why = evaluate(control, levels, cand.plan, initial_set, k, guard,
                                 diagnostics)
            entries.append(Entry(i + 1, "fresh", cand.plan, cand.ambition, prof, why))
            done += 1
        else:
            entries.append(Entry(i + 1, "fresh", cand.plan, cand.ambition,
                                 Profile.uncertified(), [Why(NOT_EVALUATED, "none")],
                                 evaluated=False))
    best, outcomes, reason, keys = select(entries, levels, step.requirement)
    chosen = entries[best]
    return Decision(chosen.plan, chosen.profile, chosen.ambition, best, reason,
                    entries, keys, outcomes, done, timed_out)


def choose_restart(control, levels, step: Step, reach, budget=None,
                   guard: Optional[Guard] = None, diagnostics=False,
                   deadline=None, allowance=0.0, clock=None) -> Decision:
    """After an alarm (Algorithm 8.1, response to an alarm): the restart
    candidates on the reach set, labelled by the global capability and
    evaluated in the order listed until the budget of the period is spent,
    as in normal operation; a candidate not evaluated counts as not
    certified. The admitted one with the highest (first key, ambition), ties
    to the smallest index, is committed at the next period boundary."""
    guard = guard or Guard()
    if deadline is not None and clock is None:
        import time
        clock = time.monotonic
    k = step.k
    cands = guard.call(k, "restart", control.restart, step, reach, default=[],
                       check=candidates_problem)
    n_eval = len(cands) if budget is None else min(budget, len(cands))
    entries, keys, outcome = [], {}, {}
    timed_out = False
    for i, cand in enumerate(cands):
        if i < n_eval and not timed_out and not _in_time(deadline, allowance, clock):
            timed_out = True
            n_eval = i
        if i >= n_eval:
            entries.append(Entry(i + 1, "restart", cand.plan, cand.ambition,
                                 Profile.uncertified(), [Why(NOT_EVALUATED, "none")],
                                 evaluated=False))
            continue
        prof, why = evaluate(control, levels, cand.plan, reach, k, guard, diagnostics)
        e = Entry(i + 1, "restart", cand.plan, cand.ambition, prof, why)
        entries.append(e)
        if levels.admits(prof.level):
            keys[e.index] = (levels.key(prof.at(step.requirement.onset),
                                        step.requirement.level), cand.ambition, -e.index)
    if keys:
        best = max(keys, key=lambda i: keys[i])
        for i in keys:
            outcome[i] = "committed" if i == best else "admitted, not selected"
        chosen = entries[best - 1]
        reason = "restart: an admitted restart candidate, labelled by the global capability"
        decision = Decision(chosen.plan, chosen.profile, chosen.ambition, "restart",
                            reason, entries, keys, outcome, n_eval, timed_out)
    else:
        last = entries[-1].reasons[0].text if entries and entries[-1].reasons else \
            "no candidate certified at the floor"
        decision = Decision(None, None, 0, None, "no restart candidate admitted: " + last,
                            entries, keys, outcome, n_eval, timed_out)
    for e in entries:
        decision.outcomes.setdefault(
            e.index, "not admitted: " + NOT_EVALUATED if not e.evaluated else "not admitted")
    return decision
