"""Declarations of units, frames and actuator semantics (Section 10.2).

A robot adapter declares the state and the inputs as the contract sees
them; a control package declares what it assumes about them; a simulator
adapter declares what it realizes. When a configuration is built, the
declarations must agree: the same names, in the same order, with the same
units and frames, the same norm, and the same actuation. Agreement of
declarations is a check of interfaces, not of physics: it catches a package
written for metres meeting a simulator that declares millimetres, but not a
simulator that declares metres and reports millimetres, nor a certificate
that is wrong.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Tuple


@dataclass(frozen=True)
class Quantity:
    """One component of a state or an input."""
    name: str
    unit: str            # SI unit symbol, e.g. "m", "m/s", "m/s^2"
    frame: str = ""      # the frame or axis in which it is expressed

    def describe(self):
        where = f", {self.frame}" if self.frame else ""
        return f"{self.name} [{self.unit}{where}]"


@dataclass(frozen=True)
class Schema:
    """The components of a state, in order, and the norm in which reserves
    and distances are measured (Section 4.1.2)."""
    name: str
    quantities: Tuple[Quantity, ...]
    norm: str = ""

    def names(self):
        return tuple(q.name for q in self.quantities)

    def mismatch(self, other: "Schema") -> List[str]:
        """The differences from another declaration, as readable lines."""
        out = []
        if self.names() != other.names():
            out.append(f"components {self.names()} differ from {other.names()}")
            return out
        for a, b in zip(self.quantities, other.quantities):
            for attr in ("unit", "frame"):
                if getattr(a, attr) != getattr(b, attr):
                    out.append(f"{a.name}: {attr} {getattr(a, attr)!r} differs from "
                               f"{getattr(b, attr)!r}")
        if self.norm != other.norm:
            out.append(f"norm {self.norm!r} differs from {other.norm!r}")
        return out


@dataclass(frozen=True)
class Bound:
    """The admissible values of one input: an interval, or a finite set."""
    name: str
    low: Any = None
    high: Any = None
    values: Tuple[Any, ...] = ()


@dataclass(frozen=True)
class Actuation:
    """Actuator semantics: what an input means and how it is applied."""
    inputs: Tuple[Quantity, ...]
    bounds: Tuple[Bound, ...] = ()
    hold: str = "zero-order hold over the period"
    meaning: str = ""

    def mismatch(self, other: "Actuation") -> List[str]:
        out = Schema("inputs", self.inputs).mismatch(Schema("inputs", other.inputs))
        if self.bounds != other.bounds:
            out.append(f"input bounds {self.bounds} differ from {other.bounds}")
        if self.hold != other.hold:
            out.append(f"hold {self.hold!r} differs from {other.hold!r}")
        if self.meaning != other.meaning:
            out.append(f"meaning {self.meaning!r} differs from {other.meaning!r}")
        return out
