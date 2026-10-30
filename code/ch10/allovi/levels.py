"""Levels and profiles (Section 4.2.5).

The test certifies plans at the levels of a finite grid that contains the
reserve floor. A profile lists, for n = 0, 1, ..., N, the largest level at
which the test certifies a plan from pair n on; it is constant from N on,
where only the rest segment counts. The core compares levels only through a
level space (contract.LevelSpace); this package provides the scalar grid.
The vector reserves of Section 4.2.7 are not supported: they would need
another level space, a separable test and one norm for each family of
constraints (Section 10.2).
"""
from __future__ import annotations

import numbers
from fractions import Fraction

NEG_INF = float("-inf")
"""The level of a plan that the test does not certify at any level of the grid."""


class Profile:
    """The profile of a plan on an initial set, constant from its last entry on."""

    __slots__ = ("_levels",)

    def __init__(self, levels):
        levels = tuple(levels)
        if not levels:
            raise ValueError("a profile has at least one entry: the rest segment")
        self._levels = levels

    @classmethod
    def uncertified(cls):
        """The profile of a candidate that is not certified, or not evaluated."""
        return cls((NEG_INF,))

    @property
    def level(self):
        """The level of the plan: the profile at pair 0."""
        return self._levels[0]

    def at(self, n):
        """The level certified from pair n on, n >= 0."""
        if n < 0:
            raise ValueError(f"no pair {n}: pairs are numbered from 0")
        return self._levels[min(n, len(self._levels) - 1)]

    def shift(self):
        """The profile of the plan without its executed head: entry n is entry
        n + 1 of this profile (Lemma 4.10(c)). A plan without pairs keeps its
        profile, as it keeps its rest segment."""
        return Profile(self._levels[1:]) if len(self._levels) > 1 else self

    def nondecreasing(self, leq=None):
        """True if no entry is above the next, in the order leq of the level
        space (by default, <=)."""
        leq = leq or (lambda a, b: a <= b)
        return all(leq(a, b) for a, b in zip(self._levels, self._levels[1:]))

    def __len__(self):
        return len(self._levels)

    def __iter__(self):
        return iter(self._levels)

    def __eq__(self, other):
        return isinstance(other, Profile) and self._levels == other._levels

    def __hash__(self):
        return hash(self._levels)

    def __repr__(self):
        return f"Profile({list(self._levels)!r})"


class ScalarLevels:
    """A grid of scalar levels {floor, floor + step, ..., top}.

    The first key of the selection rule (4.2) for this space is
    min(onset level, required level); a level is admitted if it reaches the
    floor. Levels can be exact numbers (integers or fractions), as in the
    cart, or floats. A value is a level of an exact grid if it is one
    exactly; with floats, it is one if it lies within 1e-9 steps of a point
    of the grid.
    """
    TOLERANCE = 1e-9

    def __init__(self, floor, step, top):
        if not floor <= top or step <= 0:
            raise ValueError("a level grid needs floor <= top and a positive step")
        self.floor, self.step, self.top = floor, step, top
        self.exact = all(isinstance(x, numbers.Rational) for x in (floor, step, top))
        if self.exact:
            self._n_top = int((Fraction(top) - Fraction(floor)) // Fraction(step))
        else:
            n = (top - floor) / step
            self._n_top = int(round(n)) if abs(n - round(n)) <= self.TOLERANCE \
                else int(n // 1)

    def admits(self, level):
        return level != NEG_INF and level >= self.floor

    def key(self, onset_level, required):
        return min(onset_level, required)

    def leq(self, a, b):
        return a <= b

    def below(self, x):
        """The largest level of the grid not above x, or NEG_INF if x is below
        the floor: how a test with exact clearances turns them into levels."""
        if x == NEG_INF or x < self.floor:
            return NEG_INF
        return min(self.top, self.floor + ((x - self.floor) // self.step) * self.step)

    def contains(self, level):
        """True if 'level' is a level of the grid (not NEG_INF)."""
        if level == NEG_INF:
            return False
        if self.exact and isinstance(level, numbers.Rational):
            n = (Fraction(level) - Fraction(self.floor)) / Fraction(self.step)
            return n.denominator == 1 and 0 <= n <= self._n_top
        n = (level - self.floor) / self.step
        m = round(n)
        return abs(n - m) <= self.TOLERANCE * max(1, abs(m)) and 0 <= m <= self._n_top

    def describe(self):
        return f"scalar levels from {self.floor} to {self.top} in steps of {self.step}"
