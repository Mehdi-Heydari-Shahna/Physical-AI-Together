"""The robot-model adapter (Section 10.4).

A robot model gives the control package, the predictors and the simulator
adapter one description of the robot: its coordinates, its kinematics, its
nominal dynamics and its actuation conventions. Who provides the model does
not matter to the core. A model written for the application, as the cart's,
or one compiled from a mechanism graph by a tool such as RoboCompiler,
reaches the modules through the same record, RobotModel, and is checked in
the same way.

A model is nominal: it carries no uncertainty. What a world may do
differently enters through capability labels (Section 4.2), and the
obligations of Table 4.4 stay with the modules that use the model. A model
that is mechanically consistent makes no test sound, no label valid and no
simulator faithful.

Python 3.8 or later; no other packages.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .schema import Actuation, Quantity, Schema

# The boundary: what a model delivers, and who usually supplies it.
ITEMS = ("coordinates", "kinematics", "dynamics", "actuation", "norm",
         "hold and meaning")
WHAT = {
    "coordinates": "the components of the state, with their units and frames",
    "kinematics": "the points that constraints and labels refer to, as "
                  "functions of the state",
    "dynamics": "the nominal state at the end of an interval in which the "
                "inputs are held",
    "actuation": "the inputs, with their units, frames and bounds",
    "norm": "the norm of the state, which fixes the unit of the reserve",
    "hold and meaning": "how an input acts over a period, and what is "
                        "commanded and what is realized",
}
USUALLY = {"coordinates": "model provider", "kinematics": "model provider",
           "dynamics": "model provider", "actuation": "model provider",
           "norm": "application", "hold and meaning": "application"}

PROVIDED, ELSEWHERE, MISSING = "provided", "elsewhere", "missing"


@dataclass(frozen=True)
class RobotModel:
    """What a model provider delivers at the boundary. A field left empty is
    an item that the record does not carry: the provider may offer it outside
    the record (elsewhere, with where), or no one has supplied it yet."""
    name: str
    provider: str                              # who produced it, from what
    state: Optional[Schema] = None             # coordinates; its norm
    actuation: Optional[Actuation] = None      # inputs, bounds, hold, meaning
    points: Tuple[Quantity, ...] = ()          # what kinematics returns
    kinematics: Optional[Callable] = None      # state -> points, in order
    dynamics: Optional[Callable] = None        # (state, inputs, duration) -> state
    assumptions: Tuple[str, ...] = ()          # of the nominal dynamics
    sources: Dict[str, str] = field(default_factory=dict)    # item -> origin
    elsewhere: Dict[str, str] = field(default_factory=dict)  # item -> where


def _present(model: RobotModel) -> Dict[str, bool]:
    state, act = model.state, model.actuation
    bounded = set() if act is None else {b.name for b in act.bounds}
    return {
        "coordinates": state is not None and bool(state.quantities)
                       and all(q.unit for q in state.quantities),
        "kinematics": callable(model.kinematics) and bool(model.points),
        "dynamics": callable(model.dynamics),
        "actuation": act is not None and bool(act.inputs)
                     and all(q.unit and q.name in bounded for q in act.inputs),
        "norm": state is not None and bool(state.norm),
        "hold and meaning": act is not None and bool(act.hold) and bool(act.meaning),
    }


def status(model: RobotModel) -> List[Tuple[str, str, str]]:
    """(item, status, origin) for each item of the boundary: provided by the
    record, offered by the provider elsewhere, or missing."""
    present, out = _present(model), []
    for item in ITEMS:
        if present[item]:
            out.append((item, PROVIDED, model.sources.get(item, "")))
        elif item in model.elsewhere:
            out.append((item, ELSEWHERE, model.elsewhere[item]))
        else:
            out.append((item, MISSING, model.sources.get(item, "")))
    return out


def missing(model: RobotModel) -> List[str]:
    """The items that a configuration cannot use the model without."""
    return [item for item, st, _ in status(model) if st != PROVIDED]


def _admissible(actuation: Actuation, inputs: Sequence[Any]) -> List[str]:
    bounds = {b.name: b for b in actuation.bounds}
    out = []
    for q, u in zip(actuation.inputs, inputs):
        b = bounds[q.name]
        ok = (u in b.values) if b.values else \
            ((b.low is None or u >= b.low) and (b.high is None or u <= b.high))
        if not ok:
            out.append(f"input {q.name} = {u!r} outside its bounds")
    return out


def check_model(model: RobotModel, states, inputs, duration) -> List[str]:
    """Checks that a complete model returns what it declares for sample
    states and inputs: as many points as it declares, and states with the
    declared number of components; and that the sample inputs are within the
    declared bounds. An incomplete model fails with the items it lacks."""
    problems = [f"{item}: {st}" for item, st, _ in status(model) if st != PROVIDED]
    if problems:
        return problems
    n, m = len(model.state.quantities), len(model.points)
    for u in inputs:
        if len(u) != len(model.actuation.inputs):
            problems.append(f"{len(u)} inputs, not {len(model.actuation.inputs)}")
        else:
            problems += _admissible(model.actuation, u)
    for x in states:
        if len(x) != n:
            problems.append(f"a sample state with {len(x)} components, not {n}")
            continue
        points = model.kinematics(x)
        if len(points) != m:
            problems.append(f"kinematics returned {len(points)} points, not {m}")
        for u in inputs:
            y = model.dynamics(x, u, duration)
            if len(y) != n:
                problems.append(f"dynamics returned {len(y)} components, not {n}")
    return problems


def agreement(model: RobotModel, robot, world, control=None) -> List[str]:
    """The declarations of the robot adapter, the simulator adapter and, if
    it declares them, the control package, compared with those of the model:
    every module of a configuration describes the robot as the model does.
    An incomplete model fails with the items it lacks."""
    problems = [f"{item}: {st}" for item, st, _ in status(model) if st != PROVIDED]
    if problems:
        return problems
    for role, module in (("robot", robot), ("world", world), ("control", control)):
        if module is None:
            continue
        for attr, mine in (("STATE", model.state), ("ACTUATION", model.actuation)):
            theirs = getattr(module, attr, None)
            if theirs is None and role == "control":
                continue                  # a control package may declare neither
            problems += [f"{role} {attr.lower()} vs model: {p}"
                         for p in mine.mismatch(theirs)]
    return problems


def motion_agreement(model: RobotModel, world, cases, duration,
                     tolerance=0, fmt: Callable = repr) -> Tuple[List[str], int]:
    """Runs the simulator adapter for one period from each case (state,
    inputs, law) and compares its state with the nominal dynamics of the
    model. A realized command does not by itself give the nominal motion
    (the same force moves different masses differently): the cases must be
    ones in which the simulated world moves as the model's nominal world
    does, or the tolerance must cover the variation among the admitted
    worlds. Then a difference shows that the simulator adapter does not
    realize what the model declares, such as a unit that it declares and
    does not use. fmt formats the values that the messages
    quote. Returns the differences and the number of cases run; an
    incomplete model fails with the items it lacks, and no case is run."""
    def show(xs):
        return "(" + ", ".join(fmt(x) for x in xs) + ")"

    problems = [f"{item}: {st}" for item, st, _ in status(model) if st != PROVIDED]
    if problems:
        return problems, 0
    for state, inputs, law in cases:
        mine = model.dynamics(state, inputs, duration)
        where = f"from {show(state)} under {show(inputs)}"
        try:
            theirs = tuple(world.step(state, law))
        except Exception as exc:                       # noqa: BLE001 (a failure)
            problems.append(f"{where}: the simulator adapter raised {exc!r}")
            continue
        if len(theirs) != len(mine) or any(abs(a - b) > tolerance
                                           for a, b in zip(mine, theirs)):
            problems.append(f"{where}: the model gives {show(mine)}, the simulator "
                            f"adapter {show(theirs)}")
    return problems, len(cases)
