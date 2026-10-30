"""Registration of modules and assembly of configurations (Section 10.2).

A module is a class with three class attributes, ROLE, NAME and CONTRACT,
and the methods of its role (contract.ROLES). Registering it checks that the
role exists, that the contract version is compatible, and that every
required method is present and accepts the stated number of positional
arguments. Building a configuration instantiates the modules of one run and
checks that each has the attributes of its role, that the level space and
the capability space of the control package have theirs, and that the
declarations agree: the state of the robot adapter, of the simulator adapter
and of the control package, their actuation, and the capability space of the
control package and of the capability predictor. A registered module can
then be named in a configuration without any change to the core.
"""
from __future__ import annotations

import inspect
from typing import Dict, List, Tuple

from .contract import ATTRIBUTES, CONTRACT, MANDATORY, ROLES, SUPPLIED
from .loop import Configuration
from .schema import Actuation, Schema


class RegistrationError(ValueError):
    """A module does not provide the interface of its role."""


class ConfigurationError(ValueError):
    """The modules of a configuration do not fit together."""


def _major(version: str) -> str:
    return str(version).split(".")[0]


def _arity(cls, name) -> Tuple[int, float, bool]:
    """For the method 'name' of a class: the number of positional parameters
    that a call must supply, the largest number that it may supply (inf with
    *args), and whether a keyword-only parameter lacks a default; self, or
    cls for a class method, is not counted."""
    raw = inspect.getattr_static(cls, name)
    params = list(inspect.signature(getattr(cls, name)).parameters.values())
    if inspect.isfunction(raw):              # an instance method: drop self
        params = params[1:]
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY,
                                                  p.POSITIONAL_OR_KEYWORD)]
    required = sum(1 for p in positional if p.default is p.empty)
    most = float("inf") if any(p.kind is p.VAR_POSITIONAL for p in params) \
        else len(positional)
    keyword = any(p.kind is p.KEYWORD_ONLY and p.default is p.empty for p in params)
    return required, most, keyword


def conformance(cls) -> List[str]:
    """What a module class lacks to implement its role; empty if nothing."""
    role, name = getattr(cls, "ROLE", None), getattr(cls, "NAME", None)
    if role not in ROLES:
        return [f"{cls.__name__}: ROLE {role!r} is not one of {sorted(ROLES)}"]
    problems = []
    if not isinstance(name, str) or not name:
        problems.append(f"{cls.__name__}: NAME must be a non-empty string")
    version = getattr(cls, "CONTRACT", None)
    if version is None or _major(version) != _major(CONTRACT):
        problems.append(f"{cls.__name__}: CONTRACT {version!r} is not compatible "
                        f"with the core's {CONTRACT}")
    for method, n in ROLES[role].items():
        fn = getattr(cls, method, None)
        if not callable(fn):
            problems.append(f"{cls.__name__}: missing method {method}()")
            continue
        try:
            required, most, keyword = _arity(cls, method)
        except (TypeError, ValueError):          # no signature to inspect
            continue
        if not required <= n <= most:
            got = f"{required}" if required == most else \
                f"{required} to {most}".replace("inf", "any number")
            problems.append(f"{cls.__name__}.{method}() takes {got} positional "
                            f"parameters, the interface {n}")
        if keyword:
            problems.append(f"{cls.__name__}.{method}() has a keyword-only parameter "
                            "without a default")
    return problems


class Registry:
    """The modules known by name, for each role."""

    def __init__(self):
        self._modules: Dict[Tuple[str, str], type] = {}

    def register(self, cls):
        problems = conformance(cls)
        if problems:
            raise RegistrationError("; ".join(problems))
        key = (cls.ROLE, cls.NAME)
        if key in self._modules and self._modules[key] is not cls:
            raise RegistrationError(f"a {cls.ROLE} module named {cls.NAME!r} "
                                    "is already registered")
        self._modules[key] = cls
        return cls

    def names(self, role=None):
        return sorted(n for (r, n) in self._modules if role is None or r == role)

    def create(self, role, name, **kwargs):
        try:
            cls = self._modules[(role, name)]
        except KeyError:
            raise ConfigurationError(f"no {role} module named {name!r}; registered: "
                                     f"{self.names(role)}") from None
        return cls(**kwargs)

    def build(self, name, period, modules, budget=None) -> Configuration:
        """A configuration from {role: module instance, or (name, kwargs)}."""
        parts = {}
        for role, spec in modules.items():
            if role not in ROLES:
                raise ConfigurationError(f"unknown role {role!r}")
            if isinstance(spec, tuple):
                spec = self.create(role, spec[0], **(spec[1] if len(spec) > 1 else {}))
            parts[role] = spec
        problems = compatibility(parts)
        if problems:
            raise ConfigurationError(f"{name}: " + "; ".join(problems))
        return Configuration(name, period, budget=budget, **parts)


def compatibility(parts) -> List[str]:
    """Checks that the declarations of the modules of a configuration agree."""
    problems = []
    for role in MANDATORY:
        if parts.get(role) is None:
            problems.append(f"no {role} module")
    if problems:
        return problems
    for role, module in parts.items():
        if module is None:
            continue
        problems += [f"{role}: {p}" for p in conformance(type(module))]
        for attr in ATTRIBUTES.get(role, ()):
            if not hasattr(module, attr):
                problems.append(f"{role}: missing attribute {attr}")
    if problems:
        return problems
    control, robot, world = parts["control"], parts["robot"], parts["world"]
    for what, obj in (("level space", control.levels),
                      ("capability space", control.capabilities)):
        attrs, methods = SUPPLIED[what]
        problems += [f"control: the {what} lacks the attribute {a}"
                     for a in attrs if not hasattr(obj, a)]
        problems += [f"control: the {what} lacks the method {m}()"
                     for m in methods if not callable(getattr(obj, m, None))]
    if problems:
        return problems
    declared = [("robot", robot, "STATE", Schema), ("robot", robot, "ACTUATION", Actuation),
                ("world", world, "STATE", Schema), ("world", world, "ACTUATION", Actuation)]
    declared += [("control", control, attr, kind) for attr, kind in
                 (("STATE", Schema), ("ACTUATION", Actuation))
                 if getattr(control, attr, None) is not None]
    problems += [f"{role}: {attr} is not a schema.{kind.__name__} but a "
                 f"{type(getattr(m, attr)).__name__}" for role, m, attr, kind in declared
                 if not isinstance(getattr(m, attr), kind)]
    if problems:
        return problems
    problems += [f"state, world vs robot: {m}"
                 for m in world.STATE.mismatch(robot.STATE)]
    if getattr(control, "STATE", None) is not None:
        problems += [f"state, control vs robot: {m}"
                     for m in control.STATE.mismatch(robot.STATE)]
    if getattr(control, "ACTUATION", None) is not None:
        problems += [f"actuation, control vs robot: {m}"
                     for m in control.ACTUATION.mismatch(robot.ACTUATION)]
    problems += [f"actuation, world vs robot: {m}"
                 for m in world.ACTUATION.mismatch(robot.ACTUATION)]
    capability = parts.get("capability")
    if capability is not None:
        want = getattr(control.capabilities, "NAME", None)
        got = getattr(capability, "SPACE", None)
        if want != got:
            problems.append(f"capability predictor issues labels of {got!r}, the "
                            f"test accepts {want!r}")
    return problems


REGISTRY = Registry()
"""The default registry, used by the decorator register."""


def register(cls):
    """Class decorator: registers a module in the default registry."""
    return REGISTRY.register(cls)


def build(name, period, modules, budget=None) -> Configuration:
    """Builds a configuration from the default registry."""
    return REGISTRY.build(name, period, modules, budget)
