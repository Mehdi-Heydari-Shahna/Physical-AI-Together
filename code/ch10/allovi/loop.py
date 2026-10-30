"""The closed loop of Algorithm 8.1: clocks, the order of the calls within a
period, defaults, the alarm check, the rest feedback and the restart.

Within period k (Figure 4.1), the core:
  1. informs the task of the level of the plan in force;
  2. forms the observation o_k from the measurement at t_k;
  3. receives the proposal and the requirement, or their defaults, and asks
     the task what the generator is to begin;
  4. normal operation: receives the labelling map, or its default (always
     the global capability after a restart), asks the control package for
     the initial set of the next plan, decides (governor.decide), lets the
     world execute the head of the plan in force, measures at t_{k+1} and
     checks for an alarm; after an alarm: asks the control package for the
     set that the rest feedback can reach, searches for a restart plan on it
     (governor.choose_restart), lets the world execute the rest feedback and
     measures at t_{k+1};
  5. reports the measurements of the period to the capability predictor.
The clock is the period index; times are k * period. This loop does not
measure time: a time budget is expressed as the number of fresh candidates
that can be evaluated before the deadline of each period, so that a run does
not depend on the speed of the machine. A loop on the wall clock, with an
elapsed-time deadline (governor.decide), is that of
../../ch16/complete/realtime.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

import numbers

from .contract import Observation, Requirement, Step
from .governor import (Decision, Guard, ImplementationCheckFailed, ModuleError,
                       choose_restart, decide, labelling_problem, profile_problem,
                       reach_problem, requirement_problem)
from .levels import NEG_INF


class NoCertifiedInitialPlan(RuntimeError):
    """The control package offers no usable plan certified at the floor on
    the initial set: the robot is outside the domain of the guarantees (S4),
    or the package does not implement the contract."""


@dataclass
class Configuration:
    """One instantiation: the modules, the period and the time budget."""
    name: str
    period: Any
    control: Any
    robot: Any
    world: Any
    proposal: Any = None
    capability: Any = None
    demand: Any = None
    task: Any = None
    evaluator: Any = None
    budget: Optional[Callable[[int], Optional[int]]] = None


@dataclass
class PeriodRecord:
    """Everything the core decided and observed in one period."""
    k: int
    t: Any
    governor: str                  # "normal" or "rest feedback"
    state: Any                     # state at t_k, as the world reports it
    measured: Any                  # measurement at t_k
    proposal: Any
    requirement: Requirement
    operation: Any
    labels: str                    # "predictor", "default" or "global"
    level: Any = None              # level of the plan in force; NEG_INF after an alarm
    plan: Any = None               # the plan in force during the period; None after an alarm
    initial_set: Any = None        # normal operation: the initial set of the next plan
    decision: Optional[Decision] = None
    reach: Any = None              # after an alarm: the reach set, or None
    reach_reason: str = ""
    alarm: bool = False            # an alarm is raised at t_{k+1}
    defaults: tuple = ()           # outputs that were missing and replaced by defaults


@dataclass
class RunRecord:
    """What run() returns: one PeriodRecord per period, in order, so that
    periods[k].state is the state at t_k and [p.state for p in periods] +
    [final_state] are the states at every period boundary, which evaluators
    read (cart.metrics); the instants at which an alarm was raised and those
    at which a restart plan was committed, each as the index k of t_k; the
    events that the core recorded, module failures and checks; and the
    final state and its measurement."""
    name: str
    periods: List[PeriodRecord] = field(default_factory=list)
    alarms: List[int] = field(default_factory=list)       # instants t_k of alarms, as k
    restarts: List[int] = field(default_factory=list)     # instants of restart commitments
    events: List[tuple] = field(default_factory=list)     # module failures and checks
    final_state: Any = None
    final_measured: Any = None


def run(config: Configuration, periods: int, strict=True, diagnostics=False) -> RunRecord:
    """Runs the configuration for the given number of periods. A call that
    has no default and raises stops the run in both modes, with a
    ModuleError that names the period and the call (Section 10.2.4)."""
    where = ["before period 0", "world.reset"]
    try:
        return _run(config, periods, strict, diagnostics, where)
    except (ModuleError, ImplementationCheckFailed, NoCertifiedInitialPlan):
        raise
    except Exception as exc:                           # noqa: BLE001 (by design)
        raise ModuleError(f"{where[0]}: {where[1]} raised {exc!r}; it has no "
                          f"default") from exc


def _run(c, periods, strict, diagnostics, where):
    """The loop; 'where' records the period and the call without a default
    that is under way, for the message of run()."""
    guard = Guard(strict)
    space, levels = c.control.capabilities, c.control.levels
    global_labels = space.labelling(space.global_capability)
    no_demand = Requirement(levels.floor, 0)
    bound = c.robot.measurement_bound

    x = c.world.reset()
    where[1] = "robot.measure"
    measured = c.robot.measure(x)
    where[1] = "control.initial_plan"
    plan, initial = c.control.initial_plan(measured, bound)
    where[1] = "control.profile"
    profile = c.control.profile(plan, initial)
    problem = profile_problem(levels)(profile)
    if problem:
        raise NoCertifiedInitialPlan(f"{c.name}: for the initial plan, the test "
                                     f"returned {problem}")
    if not callable(getattr(plan, "shift", None)):
        raise NoCertifiedInitialPlan(f"{c.name}: the initial plan has no method shift()")
    if not levels.admits(profile.level):
        raise NoCertifiedInitialPlan(f"{c.name}: initial plan has level {profile.level}")
    req_check = requirement_problem(levels)

    rec = RunRecord(c.name)
    mode, restarted, ambition = "normal", False, 0
    for k in range(periods):
        where[0] = f"period {k}"
        t = k * c.period
        in_force = profile.level if mode == "normal" else NEG_INF
        if c.task is not None:
            where[1] = "task.begin_period"
            c.task.begin_period(k, in_force)
        obs = Observation(k, t, measured, in_force, ambition, mode != "normal")
        defaults = []
        q = None
        if c.proposal is not None:
            q = guard.call(k, "proposal", c.proposal.propose, obs)
        if q is None:
            defaults.append("proposal")
        req = None
        if c.demand is not None:
            req = guard.call(k, "demand", c.demand.requirement, obs, check=req_check)
        if req is None:
            req = no_demand
            defaults.append("requirement")
        where[1] = "task.operation"
        op = c.task.operation(k) if c.task is not None else None

        if mode == "normal":
            labels, source = None, "global"
            if c.capability is not None and not restarted:
                labels = guard.call(k, "labels", c.capability.labels, obs,
                                    check=labelling_problem)
                source = "predictor" if labels is not None else "default"
                if labels is None:
                    defaults.append("labels")
            labels = labels or global_labels
            step = Step(k, q, labels, req, op, measured)
            where[1] = "control.initial_set"
            if getattr(c.control, "INITIAL_FROM_MEASUREMENT", False):
                initial = c.control.initial_set(plan, measured, bound)
            else:
                initial = c.control.initial_set(plan)
            budget = c.budget(k) if c.budget is not None else None
            where[1] = "the governor step"
            dec = decide(c.control, levels, step, plan, profile, initial, budget,
                         guard, diagnostics)
            where[1] = "control.law"
            law = c.control.law(plan)
            where[1] = "world.step"
            x_next = c.world.step(x, law)
            where[1] = "robot.measure"
            m_next = c.robot.measure(x_next)
            where[1] = "control.distance"
            dist = c.control.distance(m_next, initial)
            if not isinstance(dist, numbers.Real):
                raise ModuleError(f"period {k}: distance returned a "
                                  f"{type(dist).__name__}, not a number; it has no default")
            alarm = dist > bound
            rec.periods.append(PeriodRecord(k, t, "normal", x, measured, q, req, op,
                                            source, in_force, plan, initial, dec,
                                            alarm=alarm, defaults=tuple(defaults)))
            ambition = dec.ambition        # committed at t_{k+1} (Algorithm 8.1)
            if alarm:
                mode, plan, profile = "rest feedback", None, None
                rec.alarms.append(k + 1)
            else:
                plan, profile = dec.plan, dec.profile
        else:
            # (None, reason) is a legitimate answer: no restart can be certified
            # in this period; a call that fails has the same default
            reach, why = guard.call(k, "reach", c.control.reach, measured, bound,
                                    default=(None, "no set: the call of reach() failed"),
                                    check=reach_problem)
            dec = None
            if reach is not None:
                step = Step(k, q, global_labels, req, op, measured)
                budget = c.budget(k) if c.budget is not None else None
                where[1] = "the restart search"
                dec = choose_restart(c.control, levels, step, reach, budget, guard,
                                     diagnostics)
            where[1] = "control.rest_law"
            law = c.control.rest_law(measured)
            where[1] = "world.step"
            x_next = c.world.step(x, law)
            where[1] = "robot.measure"
            m_next = c.robot.measure(x_next)
            rec.periods.append(PeriodRecord(k, t, "rest feedback", x, measured, q, req,
                                            op, "global", in_force, None, None, dec, reach,
                                            why, defaults=tuple(defaults)))
            if dec is not None and dec.plan is not None:
                plan, profile, ambition = dec.plan, dec.profile, dec.ambition
                mode, restarted = "normal", True
                rec.restarts.append(k + 1)
            else:
                ambition = 0
        if c.capability is not None:
            guard.call(k, "observe", c.capability.observe, k, measured, m_next)
        x, measured = x_next, m_next
    rec.events = guard.events
    rec.final_state, rec.final_measured = x, measured
    return rec
