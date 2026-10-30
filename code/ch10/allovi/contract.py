"""The contract in software (Table 4.4): the data that modules exchange and
the interfaces through which they are supplied.

Each interface below is a typing.Protocol: a module implements it by having
the methods and attributes listed, with the meanings stated in the
docstrings; it need not inherit from it. The registry (registry.py) checks
the names and the numbers of parameters when a module is registered, and the
declarations of units, frames and actuation when a configuration is built;
the core checks the kind of every output that it uses (governor.py, loop.py).
It cannot check meanings: that a test is sound, or that a label is valid as
often as its predictor states, are obligations of the supplier (Table 4.4),
which no software check discharges.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Protocol, Tuple

CONTRACT = "1.1"
"""Version of the interfaces, major.minor. A change that every module must
follow (a new required method, a changed signature or meaning) increases the
major number; an addition that no module must follow (a new optional method)
increases the minor number. A module declares the version that it
implements, and the registry accepts it only if its major number is that of
the core. Version 1.1 adds two options that no module must use: a control
package may compute the initial set from the measurement
(INITIAL_FROM_MEASUREMENT), and governor.decide may stop evaluating at an
elapsed-time deadline (Section 16.4)."""


# ----------------------------------------------------------------------------
# Data exchanged during a period
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Observation:
    """The observation o_k formed at t_k (Section 4.1.5, Table 4.4): what the
    proposal method and the predictors receive. The ambition is the one
    committed at t_k, by the decision of period k - 1, even if an alarm is
    raised at t_k (Algorithm 8.1); it is 0 after a period of rest feedback
    without a restart. The observation leaves out the labels in force, which
    Section 4.1.5 allows it to include: labels are attached to plans, which
    are opaque to the core, and a labelling map is valid only during the
    period in which it is issued."""
    k: int
    t: Any
    measured: Any        # measured state, in the units of the state schema
    level: Any           # level of the plan in force; NEG_INF after an alarm
    ambition: Any        # ambition committed at t_k
    alarm: bool          # True while the rest feedback acts after an alarm


@dataclass(frozen=True)
class Requirement:
    """(level, onset): the plan committed for the next period should hold the
    level, a level of the level space, from its pair 'onset' on, a whole
    number of at least 0 (Definition 4.12). The default, with no demand
    pending, is (floor, 0)."""
    level: Any
    onset: int = 0


@dataclass(frozen=True)
class Step:
    """What the generator receives during period k, besides the plan in force
    and the initial set (Definition 4.14). The labelling map is used only
    during period k, and it must give the same label for the same arguments
    throughout that period."""
    k: int
    proposal: Any                 # None: no proposal; offer recovery candidates only
    labelling: Callable[[int, Any], Any]   # labelling(j, segment): the label for period j
    requirement: Requirement
    operation: Any                # what the task asks the generator to begin, or None
    measured: Any                 # measured state at t_k


@dataclass(frozen=True)
class Candidate:
    """A labelled plan proposed by the generator, with its ambition, a number
    in [0, 1]."""
    plan: Any
    ambition: Any


# ----------------------------------------------------------------------------
# Objects that the control package supplies
# ----------------------------------------------------------------------------
class Plan(Protocol):
    """A labelled plan (Definition 4.6). Plans, their segments and labels,
    initial sets, proposals, laws and states are opaque to the core: it
    passes them between the modules and asks the control package about them.
    Of a plan it needs one operation."""

    def shift(self) -> "Plan":
        """The plan without its head: the fallback of the next period
        (Section 8.3.2). A plan without pairs returns itself."""


class LevelSpace(Protocol):
    """The levels at which the test certifies plans (Section 4.2.5): what the
    core needs to check profiles and requirements, to admit and to select
    (rule (4.2)). levels.ScalarLevels is one. The core marks a plan that is
    not certified at any level with the scalar NEG_INF, so a level space
    must accept it in admits() and leq()."""
    floor: Any                    # the reserve floor, the default requirement

    def admits(self, level: Any) -> bool:
        """True if the level reaches the floor."""

    def key(self, onset_level: Any, required: Any) -> Any:
        """The first key of the selection rule (4.2): a number."""

    def leq(self, a: Any, b: Any) -> bool:
        """The order of levels, with which profiles must be nondecreasing."""

    def contains(self, level: Any) -> bool:
        """True if 'level' is a level of the space (not NEG_INF)."""


class CapabilitySpace(Protocol):
    """The capabilities that labels take (Definition 4.2), shared by the test
    and the capability predictor."""
    NAME: str                     # compared with the SPACE of the predictor
    global_capability: Any        # valid in every period (S2): the application's obligation

    def labelling(self, capability: Any) -> Callable[[int, Any], Any]:
        """The labelling map that issues 'capability' for every period and
        segment; with the global capability, that of the fixed predictor."""


SUPPLIED = {
    "level space": (("floor",), ("admits", "key", "leq", "contains")),
    "capability space": (("NAME", "global_capability"), ("labelling",)),
}
"""Attributes and methods of the objects that the control package supplies
as its attributes 'levels' and 'capabilities'; checked when a configuration
is built. Plans are checked when the core receives them."""


# ----------------------------------------------------------------------------
# Interfaces of the supplied modules
# ----------------------------------------------------------------------------
class ControlPackage(Protocol):
    """Segments, generator, sound test, rest segment and rest feedback: one
    compatible package (Section 4.2, Table 4.4). Its methods without a
    default (initial_set, law, distance, rest_law) must return before the
    deadline, as the common steps must (S6)."""
    ROLE = "control"
    capabilities: CapabilitySpace
    levels: LevelSpace

    def initial_plan(self, measured: Any, bound: Any) -> Tuple[Any, Any]:
        """A plan certified, with its profile reaching the floor, on an initial
        set that contains every state within 'bound' of the measurement:
        (plan, initial set), assumption (S4)."""

    def initial_set(self, plan: Any) -> Any:
        """The admissible initial set for the plan to be committed while the
        head of 'plan' is executed: by default, the end of the head's funnel
        (Definition 4.15). Its admissibility is the package's obligation. A
        package whose class attribute INITIAL_FROM_MEASUREMENT is true is
        called as initial_set(plan, measured, bound), with the measurement
        at the start of the period and the measurement bound, and may return
        a smaller set predicted from them (contract 1.1)."""

    def generate(self, step: Step, plan: Any, initial_set: Any) -> List[Candidate]:
        """The candidates of the period (Definition 4.14), each starting from
        'initial_set', with the labels of step.labelling attached to their
        pairs, listed in the order of evaluation."""

    def profile(self, plan: Any, initial_set: Any) -> Any:
        """The verdicts of the sound test turned into a profile, a
        levels.Profile on the level space (Definition 4.9): nondecreasing,
        with entries that are levels of the space or NEG_INF. Its soundness
        is the supplier's obligation."""

    def law(self, plan: Any) -> Any:
        """The feedback law of the head of 'plan', to be executed in the
        period: an object that can be called with the time within the period
        and the state and returns the input, in the units of the actuation
        (Definition 4.4). A simulator adapter may also rely on a structure of
        the laws that the package documents, as the track of the cart does."""

    def distance(self, measured: Any, initial_set: Any) -> Any:
        """Distance from a measured state to an initial set, in the norm of the
        state schema (the alarm check)."""

    def rest_law(self, measured: Any) -> Any:
        """The rest feedback, applied without a guarantee after an alarm; a
        law in the sense of law()."""

    def reach(self, measured: Any, bound: Any) -> Tuple[Any, str]:
        """(set, reason): a set containing every state that the rest feedback
        can reach in one period from the states within 'bound' of the
        measurement, in the worlds consistent with the global capability.
        (None, reason) is a legitimate answer: the package cannot represent
        such a set now, so no restart is attempted in the period. A call that
        fails is replaced by that answer in operation."""

    def restart(self, step: Step, reach: Any) -> List[Candidate]:
        """Restart candidates on the set 'reach', labelled by the global
        capability (step.labelling is the global labelling map)."""


class ProposalMethod(Protocol):
    """Any causal map from observations to proposals (Definition 4.11)."""
    ROLE = "proposal"

    def propose(self, observation: Observation) -> Any:
        """The proposal q_k, or None if none is ready."""


class CapabilityPredictor(Protocol):
    """Labels for future periods and segments: at least as informative as the
    global capability (Assumption 4.3), and the global capability beyond the
    horizon (Section 4.2.1)."""
    ROLE = "capability"
    SPACE: str                    # the NAME of the capability space of its labels
    horizon: int

    def labels(self, observation: Observation) -> Optional[Callable[[int, Any], Any]]:
        """The labelling map of the period, or None (default: the global
        capability everywhere)."""

    def observe(self, k: int, before: Any, after: Any) -> None:
        """The measured states at the start and at the end of period k."""


class DemandPredictor(Protocol):
    """Requirements for the plan committed for the next period (Definition 4.12)."""
    ROLE = "demand"

    def requirement(self, observation: Observation) -> Optional[Requirement]:
        """The requirement, or None (default: no demand pending)."""


class RobotAdapter(Protocol):
    """The robot as the contract sees it: its state and actuation, in declared
    units and frames, and its measurement with an error bound."""
    ROLE = "robot"
    STATE: Any                    # schema.Schema of the state
    ACTUATION: Any                # schema.Actuation of the inputs
    measurement_bound: Any        # the bound on the measurement error (S5)

    def measure(self, state: Any) -> Any:
        """The measurement of a state that the simulator adapter returned,
        within measurement_bound of it."""


class SimulatorAdapter(Protocol):
    """The world that realizes the executed feedback laws (a simulator, or the
    machine itself). Its states have the components of its STATE, which must
    agree with that of the robot adapter."""
    ROLE = "world"
    STATE: Any                    # schema.Schema of the states that it returns
    ACTUATION: Any                # the actuation semantics that it realizes

    def reset(self) -> Any:
        """The initial state."""

    def step(self, state: Any, law: Any) -> Any:
        """The state at the end of one period under 'law', a law of the
        control package, from 'state'."""


class Task(Protocol):
    """The goal and the operations of the task, read by the demand predictor
    and the generator."""
    ROLE = "task"

    def begin_period(self, k: int, level_in_force: Any) -> None:
        """Informs the task of the level of the plan in force at t_k, which
        decides whether an operation that is due can begin (Assumption 4.13)."""

    def operation(self, k: int) -> Any:
        """What the generator is asked to begin during period k, or None."""


class Evaluator(Protocol):
    """Measures task performance alongside the guarantees (Chapter 16), from
    the record of a run; the core never calls it."""
    ROLE = "evaluator"

    def evaluate(self, record: Any) -> dict:
        """Metrics of a run record."""


# Names and numbers of positional parameters (without self) of the methods
# that each role must provide; the registry checks them.
ROLES = {
    "control": {"initial_plan": 2, "initial_set": 1, "generate": 3, "profile": 2,
                "law": 1, "distance": 2, "rest_law": 1, "reach": 2, "restart": 2},
    "proposal": {"propose": 1},
    "capability": {"labels": 1, "observe": 3},
    "demand": {"requirement": 1},
    "robot": {"measure": 1},
    "world": {"reset": 0, "step": 2},
    "task": {"begin_period": 2, "operation": 1},
    "evaluator": {"evaluate": 1},
}

ATTRIBUTES = {
    "control": ("capabilities", "levels"),
    "capability": ("SPACE", "horizon"),
    "robot": ("STATE", "ACTUATION", "measurement_bound"),
    "world": ("STATE", "ACTUATION"),
}

RECORDS = (Observation, Requirement, Step, Candidate)
"""The records that cross the interfaces in every period."""

# Optional methods: used for diagnostic traces only, never for decisions.
DIAGNOSTIC = {
    "control": ("explain", "describe_plan", "describe_set", "describe_proposal"),
    "robot": ("describe_state",),
}

MANDATORY = ("control", "robot", "world")
"""Roles without which no configuration is built. Proposals, labels,
requirements and tasks are optional (Section 4.3.4)."""
