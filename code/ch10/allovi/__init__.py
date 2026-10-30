"""allovi: the reference implementation of the common core of ALLOVI.

Physical AI, Together, Chapter 10.

The package implements once what is common to every application: the
viability governor of Algorithm 8.1 (fallback, evaluation within a time
budget, admission, selection, commitment, the alarm check, the rest
feedback and the restart search), the clock of the closed loop, the defaults
for missing outputs, and the records of every decision. Everything that
depends on the robot is supplied by modules, through the interfaces of
contract.py, and registered by name (registry.py).

Python 3.8 or later; no other packages.
"""
from .contract import (CONTRACT, Candidate, CapabilitySpace, CapabilityPredictor,
                       ControlPackage, DemandPredictor, Evaluator, LevelSpace,
                       Observation, Plan, ProposalMethod, Requirement, RobotAdapter,
                       SimulatorAdapter, Step, Task)
from .governor import (NOT_EVALUATED, Decision, Entry, Guard, ImplementationCheckFailed,
                       ModuleError, Why, choose_restart, decide, select)
from .levels import NEG_INF, Profile, ScalarLevels
from .loop import (Configuration, NoCertifiedInitialPlan, PeriodRecord, RunRecord, run)
from .registry import (REGISTRY, ConfigurationError, Registry, RegistrationError, build,
                       compatibility, conformance, register)
from .schema import Actuation, Bound, Quantity, Schema
from . import model
from .model import RobotModel

__version__ = "0.1.0"
