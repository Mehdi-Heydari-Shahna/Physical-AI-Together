"""
The yard of Section 14.3: the route of the base's middle, its grounds and
the stop line. Every number is ours.

Physical AI, Together, Chapter 14 (Section 14.3).

The route runs straight along x for S_TURN1 metres, turns to the left
through a corner of radius RADIUS, runs straight for STRAIGHT_2 metres,
turns to the right through a second such corner and runs straight again
to a stop line STRAIGHT_3 metres beyond the second turn's exit: a wall
before the drop-off, which the machine's front-most point must keep at least
the clearance's floor away. Each turn is entered and left along transitions
of labels.TRANSITION metres (labels.route_curvature). The task asks for rest
with the front-most point within DROP_ZONE of the line, the arm reaching.

The ground is firm, of friction FIRM, except for a wet patch from WET_BEFORE
before the second turn's entry to WET_BEFORE after its exit, of friction
WET. The site's survey, which the capability predictors read, lists the firm
ground at SURVEY_FIRM and, once the patch is reported, the wet patch at
SURVEY_WET, less than the frictions themselves; a map drawn before the patch
was reported lists the firm ground's SURVEY_FIRM everywhere. Along the
route, a wheel's ground is the one under the route's arc length at the
wheel, its middle's arc length plus or minus model.AXLE.
"""
import math
from collections import namedtuple

import numpy as np

from . import labels as hl
from . import model

RADIUS = 5.0             # [m] both turns' arcs
ENTRY, ARC, EXIT = hl.turn_length(RADIUS)
TURN_LENGTH = ENTRY + ARC + EXIT
S_TURN1 = 50.0           # [m] the first turn's entry begins here; it turns to the left
STRAIGHT_2 = 20.0        # [m] from the first turn's exit to the second's entry
S_TURN2 = S_TURN1 + TURN_LENGTH + STRAIGHT_2   # the second turn, to the right
STRAIGHT_3 = 20.0        # [m] from the second turn's exit to the stop line
S_LINE = S_TURN2 + TURN_LENGTH + STRAIGHT_3
DROP_ZONE = 0.5          # [m] the task asks for rest with the front-most point within this of the line

Turn = namedtuple("Turn", "name start end side")
TURNS = (Turn("first", S_TURN1, S_TURN1 + TURN_LENGTH, 1.0),
         Turn("second", S_TURN2, S_TURN2 + TURN_LENGTH, -1.0))

FIRM, WET = 0.6, 0.35    # the grounds' frictions (ours)
SURVEY_FIRM, SURVEY_WET = 0.5, 0.3   # what the site's survey lists for them (ours)
WET_BEFORE = 3.0         # [m] the wet patch reaches this far beyond the second turn's ends
WET_FROM, WET_TO = TURNS[1].start - WET_BEFORE, TURNS[1].end + WET_BEFORE

Ground = namedtuple("Ground", "name s_from s_to friction survey")
GROUNDS = (Ground("firm", -math.inf, WET_FROM, FIRM, SURVEY_FIRM),
           Ground("wet", WET_FROM, WET_TO, WET, SURVEY_WET),
           Ground("firm", WET_TO, math.inf, FIRM, SURVEY_FIRM))


def ground_at(s):
    for g in GROUNDS:
        if g.s_from <= s < g.s_to:
            return g
    return GROUNDS[-1]


def grounds_on(lo, hi):
    """The grounds under the arc lengths [lo, hi] of the route."""
    return [g for g in GROUNDS if g.s_from < hi and g.s_to > lo] or [ground_at(lo)]


def curvature(s):
    """The route's curvature [1/m], positive to the left, and its rate along
    the route [1/m^2] at the arc lengths s (arrays)."""
    s = np.asarray(s, dtype=float)
    k, dk = np.zeros_like(s), np.zeros_like(s)
    for t in TURNS:
        a, b = hl.route_curvature(s - t.start, RADIUS, t.side)
        k, dk = k + a, dk + b
    return k, dk


def curvature_at(s):
    """curvature() at one arc length, as two floats (for the simulator's
    steps)."""
    for t in TURNS:
        u = s - t.start
        if 0.0 <= u < TURN_LENGTH:
            if u < ENTRY:
                c = t.side / (2.0 * RADIUS)
                return (c * (1.0 - math.cos(math.pi * u / ENTRY)),
                        c * math.pi / ENTRY * math.sin(math.pi * u / ENTRY))
            if u < ENTRY + ARC:
                return t.side / RADIUS, 0.0
            w = u - ENTRY - ARC
            c = t.side / (2.0 * RADIUS)
            return (c * (1.0 + math.cos(math.pi * w / EXIT)),
                    -c * math.pi / EXIT * math.sin(math.pi * w / EXIT))
    return 0.0, 0.0


def turn_on(lo, hi):
    """The turn whose entry, arc or exit the arc lengths [lo, hi] reach into,
    or None."""
    for t in TURNS:
        if hi > t.start and lo < t.end:
            return t
    return None


def next_turn(s):
    """The first turn whose entry lies at or beyond the arc length s, or
    None."""
    for t in TURNS:
        if t.start >= s:
            return t
    return None


def wheel_frictions(s):
    """The frictions under the front and the rear wheels of a machine whose
    middle lies at the arc length s."""
    return ground_at(s + model.AXLE).friction, ground_at(s - model.AXLE).friction


_TABLE = {}


def pose_at(s):
    """The route's point (x, y) [m] and direction [rad] at the arc length s,
    from the curvature integrated every 5 mm (for the figures)."""
    if not _TABLE:
        ss = np.arange(-5.0, S_LINE + 10.0, 0.005)
        k, _ = curvature(ss)
        heading = np.concatenate([[0.0], np.cumsum(0.5 * (k[1:] + k[:-1]) * 0.005)])
        x = np.concatenate([[ss[0]], ss[0] + np.cumsum(0.5 * (np.cos(heading[1:])
                                                             + np.cos(heading[:-1])) * 0.005)])
        y = np.concatenate([[0.0], np.cumsum(0.5 * (np.sin(heading[1:])
                                                    + np.sin(heading[:-1])) * 0.005)])
        _TABLE.update(s=ss, x=x, y=y, heading=heading)
    t = _TABLE
    return (float(np.interp(s, t["s"], t["x"])), float(np.interp(s, t["s"], t["y"])),
            float(np.interp(s, t["s"], t["heading"])))
