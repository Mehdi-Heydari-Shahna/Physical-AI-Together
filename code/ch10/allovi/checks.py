"""Compatibility checks for modules and configurations (Section 10.3).

The checks are of four kinds, kept apart because they show different
things.
  interface      the modules fit the core: registration and declarations,
                 and every output that the core acts on, of the contracted
                 kind and with the properties that the core checks (a
                 profile that does not decrease, for example);
  consistency    properties that the specification of a module implies and
                 that its outputs can be tested for: a fallback that keeps
                 its shifted profile, labels at least as informative as the
                 global capability and equal to it beyond the horizon, a
                 labelling map that does not change within its period,
                 inputs within the declared bounds at the start of each
                 period, states with the declared number of components, and
                 runs that repeat exactly;
  falsification  in simulation, against an oracle that knows the world: an
                 executed label that is invalid, a state whose true reserve
                 is smaller than the level certified for it while every
                 executed label was valid, and an alarm raised while every
                 executed label was valid;
  module         the checks that a module supplies, through a method
                 checks() that returns Results; the suite runs them with the
                 others.

A failed check shows a mistake, or an obligation that does not hold in the
cases that were run. A passed check shows only that those cases revealed
none: no check discharges an obligation of Table 4.4.

The wrappers that watch the run pass every call and answer on unchanged; to
check labels and inputs, they also call labelling maps and laws once more,
so the checks assume that these calls have no side effects.

The checks use only the interfaces of contract.py and five optional
methods: head(plan) of the control package, the pair that the plan executes
in its first period, and admits(label, segment, k) and clearance(state) of a
simulator adapter, which knows the world that it simulates, without which
the falsification checks are not applicable; informative(c, d) of the
capability space, without which labels are not compared with the global
capability; and checks() of any module.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, List

from .governor import ImplementationCheckFailed, ModuleError
from .loop import run
from .records import candidate_rows, decision_rows
from .registry import ConfigurationError, conformance

PASSED, FAILED, NOT_APPLICABLE = "passed", "failed", "not applicable"
KINDS = "outputs of the contracted kinds"
FALLBACK = "fallback keeps its shifted profile"
ROLES_IN_CONFIGURATION = ("control", "proposal", "capability", "demand", "task",
                          "robot", "world", "evaluator")


@dataclass
class Result:
    """The outcome of one check."""
    name: str
    kind: str                  # "interface", "consistency", "falsification" or "module"
    status: str                # PASSED, FAILED or NOT_APPLICABLE
    detail: str = ""
    cases: int = 0             # the cases that the check examined

    def line(self):
        cases = f" ({self.cases} cases)" if self.cases else ""
        detail = f": {self.detail}" if self.detail else ""
        return f"{self.status:<14} {self.kind:<13} {self.name}{cases}{detail}"


def outcome(name, kind, failures, cases, why_not=None):
    """A Result from a list of failures among 'cases' cases."""
    if why_not is not None:
        return Result(name, kind, NOT_APPLICABLE, why_not)
    if failures:
        return Result(name, kind, FAILED,
                      f"{len(failures)} failed; first: {failures[0]}", cases)
    return Result(name, kind, PASSED, "", cases)


# ----------------------------------------------------------------------------
# Recording wrappers: they observe what the modules return, change nothing
# ----------------------------------------------------------------------------
class _Wrapper:
    """Forwards every attribute to the module that it wraps."""

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _RecordedLabelling:
    def __init__(self, labelling, k, calls):
        self.labelling, self.k, self.calls = labelling, k, calls

    def __call__(self, j, segment):
        label = self.labelling(j, segment)
        self.calls.append((self.k, j, segment, label))
        return label


class _Capability(_Wrapper):
    """Records every label that the generator asks for. Just before the
    predictor learns the measurements of a period, it asks the labelling map
    of the period again for each of those labels, to see whether it still
    gives the same, and for the label of each of those segments in the first
    period beyond the horizon, k + 2 + horizon."""

    def __init__(self, inner, top):
        super().__init__(inner)
        for name, value in (("top", top), ("calls", []), ("maps", {}), ("unstable", []),
                            ("beyond", []), ("probes", 0)):
            object.__setattr__(self, name, value)

    def labels(self, observation):
        labelling = self._inner.labels(observation)
        if labelling is None or not callable(labelling):
            return labelling
        self.maps[observation.k] = (labelling, len(self.calls))
        return _RecordedLabelling(labelling, observation.k, self.calls)

    def observe(self, k, before, after):
        if k in self.maps:
            labelling, start = self.maps.pop(k)
            far = k + 2 + self._inner.horizon
            for _, j, segment, label in self.calls[start:]:
                again = labelling(j, segment)
                if again != label:
                    self.unstable.append(f"period {k}: label {label!r} for period {j}, "
                                         f"then {again!r}")
                late = labelling(far, segment)
                object.__setattr__(self, "probes", self.probes + 1)
                if late != self.top:
                    self.beyond.append(f"period {k}: label {late!r} for period {far}")
        return self._inner.observe(k, before, after)


class _World(_Wrapper):
    """Checks every law that the world receives against the declared
    actuation, at the start of its period, and every state that it returns
    against the declared state."""

    def __init__(self, inner, actuation, n_components):
        super().__init__(inner)
        object.__setattr__(self, "actuation", actuation)
        object.__setattr__(self, "n", n_components)
        object.__setattr__(self, "laws", 0)
        object.__setattr__(self, "states", 0)
        object.__setattr__(self, "law_failures", [])
        object.__setattr__(self, "state_failures", [])

    def reset(self):
        state = self._inner.reset()
        self._state(state)
        return state

    def step(self, state, law):
        self._law(state, law)
        new = self._inner.step(state, law)
        self._state(new)
        return new

    def _state(self, state):
        object.__setattr__(self, "states", self.states + 1)
        try:
            n = len(state)
        except TypeError:
            return
        if n != self.n:
            self.state_failures.append(f"a state with {n} components, not {self.n}")

    def _law(self, state, law):
        object.__setattr__(self, "laws", self.laws + 1)
        if not callable(law):
            self.law_failures.append(f"a {type(law).__name__} that cannot be called")
            return
        inputs = law(0, state)
        names = [q.name for q in self.actuation.inputs]
        if len(inputs) != len(names):
            self.law_failures.append(f"{len(inputs)} inputs, not {len(names)}")
            return
        bounds = {b.name: b for b in self.actuation.bounds}
        for name, u in zip(names, inputs):
            b = bounds.get(name)
            if b is None:
                continue
            ok = (u in b.values) if b.values else \
                ((b.low is None or u >= b.low) and (b.high is None or u <= b.high))
            if not ok:
                self.law_failures.append(f"input {name} = {u!r} outside its bounds")


# ----------------------------------------------------------------------------
# The suite
# ----------------------------------------------------------------------------
def check(make: Callable[[], Any], periods: int, fmt: Callable = repr) -> List[Result]:
    """Runs the checks on the configuration that make() builds; make() is
    called once for each run, so that every run starts afresh. fmt formats
    the values that the messages quote."""
    results: List[Result] = []
    try:
        config = make()
    except ConfigurationError as exc:
        return [Result("registration and declarations", "interface", FAILED, str(exc))]
    modules = [getattr(config, r) for r in ROLES_IN_CONFIGURATION
               if getattr(config, r) is not None]
    problems = [p for m in modules for p in conformance(type(m))]
    results.append(outcome("registration and declarations", "interface", problems,
                           len(modules)))

    # one strict run with diagnostics, observed by the recording wrappers
    capability = None
    if config.capability is not None:
        capability = _Capability(config.capability,
                                 config.control.capabilities.global_capability)
    world = _World(config.world, config.robot.ACTUATION, len(config.robot.STATE.quantities))
    observed = dataclasses.replace(config, capability=capability or config.capability,
                                   world=world)
    record, stopped = None, None
    try:
        record = run(observed, periods, strict=True, diagnostics=True)
    except (ModuleError, ImplementationCheckFailed) as exc:
        stopped = exc
    except Exception as exc:                           # noqa: BLE001 (by design)
        stopped = ModuleError(f"the run stopped: {exc!r}")   # run() names the call
    if isinstance(stopped, ModuleError):
        results.append(Result(KINDS, "interface", FAILED, str(stopped)))
    elif stopped is None:
        results.append(Result(KINDS, "interface", PASSED, "", periods))
    else:
        completed = max(world.states - 1, 0)       # periods completed before the stop
        results.append(Result(KINDS, "interface", PASSED,
                              f"until the strict run stopped, in period {completed}",
                              completed))
    if isinstance(stopped, ImplementationCheckFailed):
        results.append(Result(FALLBACK, "consistency", FAILED, str(stopped)))
    elif stopped is None:
        results.append(Result(FALLBACK, "consistency", PASSED, "", periods))
    else:
        results.append(Result(FALLBACK, "consistency", NOT_APPLICABLE,
                              "the strict run stopped"))
    results += _label_checks(config, capability, fmt)
    results.append(outcome("inputs within the declared bounds", "consistency",
                           world.law_failures, world.laws))
    results.append(outcome("states with the declared components", "consistency",
                           world.state_failures, world.states))
    results.append(_repeat_check(make, periods, record, observed))
    results += _falsification(config, record, fmt)
    for m in modules:
        own = getattr(m, "checks", None)
        if callable(own):
            try:
                results += list(own())
            except Exception as exc:                   # noqa: BLE001 (by design)
                results.append(Result(f"checks of {getattr(m, 'NAME', type(m).__name__)}",
                                      "module", FAILED, f"raised {exc!r}"))
    return results


def _label_checks(config, capability, fmt):
    names = ("labels at least as informative as the global capability",
             "the global capability beyond the horizon",
             "labelling maps stable within their period")
    if capability is None:
        return [outcome(n, "consistency", [], 0, "no capability predictor") for n in names]
    space = config.control.capabilities
    calls = capability.calls
    informative = getattr(space, "informative", None)
    if callable(informative):
        failures = [f"period {k}: label {fmt(label)} for period {j}"
                    for k, j, _, label in calls
                    if not informative(label, space.global_capability)]
        first = outcome(names[0], "consistency", failures, len(calls))
    else:
        first = outcome(names[0], "consistency", [], 0,
                        "the capability space has no method informative()")
    return [first,
            outcome(names[1], "consistency", capability.beyond, capability.probes),
            outcome(names[2], "consistency", capability.unstable, len(calls))]


def _repeat_check(make, periods, record, observed):
    name = "a second run repeats every record"
    if record is None:
        return Result(name, "consistency", NOT_APPLICABLE, "the strict run stopped")
    config = make()
    try:
        again = run(config, periods, strict=True, diagnostics=True)
    except (ModuleError, ImplementationCheckFailed) as exc:
        return Result(name, "consistency", FAILED, f"the second run stopped: {exc}")
    except Exception as exc:                           # noqa: BLE001 (by design)
        return Result(name, "consistency", FAILED, f"the second run stopped: {exc!r}")
    first = (decision_rows(record, observed.control, observed.robot),
             candidate_rows(record, observed.control))
    second = (decision_rows(again, config.control, config.robot),
              candidate_rows(again, config.control))
    for what, a, b in (("period", first[0], second[0]), ("candidate", first[1], second[1])):
        if len(a) != len(b):
            return Result(name, "consistency", FAILED, f"{len(a)} {what} rows, then {len(b)}")
        for ra, rb in zip(a, b):
            if ra != rb:
                return Result(name, "consistency", FAILED,
                              f"the {what} rows of period {ra.get('k')} differ")
    return Result(name, "consistency", PASSED, "", len(first[0]) + len(first[1]))


def _falsification(config, record, fmt):
    names = ("executed labels valid",
             "true reserve at least the level in force",
             "no alarm while every executed label was valid")
    world, control = config.world, config.control
    oracle = all(callable(getattr(world, m, None)) for m in ("admits", "clearance")) \
        and callable(getattr(control, "head", None))
    if not oracle:
        return [outcome(n, "falsification", [], 0, "no oracle") for n in names]
    if record is None:
        return [outcome(n, "falsification", [], 0, "the strict run stopped") for n in names]
    try:
        return _falsify(world, control, record, fmt, names)
    except Exception as exc:                           # noqa: BLE001 (by design)
        return [Result(n, "falsification", FAILED, f"the oracle or head() raised {exc!r}")
                for n in names]


def _falsify(world, control, record, fmt, names):
    invalid, short, alarms = [], [], []
    labels = reserves = 0
    valid_so_far, resting = True, False
    for p in record.periods:
        if p.governor != "normal":
            valid_so_far, resting = False, True   # nothing is certified after an alarm
            continue
        if resting:                               # a restart plan is in force: the
            valid_so_far, resting = True, False   # guarantees hold again (Section 9.2.3)
        segment, label = control.head(p.plan)
        labels += 1
        if not world.admits(label, segment, p.k):
            invalid.append(f"period {p.k}: label {fmt(label)}")
            valid_so_far = False
        if valid_so_far:
            reserves += 1
            true = world.clearance(p.state)
            if true < p.level:
                short.append(f"period {p.k}: level {fmt(p.level)} in force, true reserve "
                             f"{fmt(true)}")
            if p.alarm:
                alarms.append(f"alarm at the end of period {p.k}")
    return [outcome(names[0], "falsification", invalid, labels),
            outcome(names[1], "falsification", short, reserves),
            outcome(names[2], "falsification", alarms, reserves)]


def summary(results: List[Result]) -> str:
    """'n passed, n failed, n not applicable'."""
    counts = {s: sum(1 for r in results if r.status == s)
              for s in (PASSED, FAILED, NOT_APPLICABLE)}
    return ", ".join(f"{n} {s}" for s, n in counts.items())
