"""What every module of the cart shares: the model of Chapter 3, imported
unchanged from ../ch03/cart_example.py, the declarations of its state,
actuation and levels (Section 10.2), and the robot model through which the
modules receive them (MODEL, Section 10.4).

The names that the cart's modules, and modules written for the cart,
import from here: ce, the example of Chapter 3 as a module (its parameters,
segments, plans and the functions of its generator); T, the period, 1/10 s;
OPERATION_LEVEL, the reserve that the tool operation needs, 13/10 m; STATE,
ACTUATION and LEVELS; MODEL; and fmt, which prints an exact number as the
records do, with at most six decimals. Every quantity is exact
(fractions.Fraction). Importing this module puts ../../ch03 and .. on
sys.path. ../templates/HELPERS.md documents each name, with examples."""
import os
import sys
from fractions import Fraction as Fr

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "ch03"))
sys.path.insert(0, os.path.join(_HERE, ".."))
import cart_example as ce                                 # noqa: E402
from allovi import (Actuation, Bound, Quantity, RobotModel, Schema,   # noqa: E402
                    ScalarLevels)

T = ce.T                       # period of the governor: 0.1 s
OPERATION_LEVEL = ce.DEMAND_LEVEL   # reserve that the tool operation needs: 1.3 m

# Positions and velocities along the track, positive towards
# the wall; the reach of the tool beyond the front bumper.
STATE = Schema(
    "cart state",
    (Quantity("p", "m", "track"),    # bumper, from the start
     Quantity("v", "m/s", "track"),
     Quantity("l", "m", "tool")),    # reach beyond the bumper
    norm="max(|p| + |l|, |v| * 1 s)")

ACTUATION = Actuation(
    inputs=(Quantity("u_a", "m/s^2", "track"),
            Quantity("u_l", "m", "tool")),
    bounds=(Bound("u_a", low=-ce.U_MAX, high=ce.U_MAX),
            Bound("u_l", values=(Fr(0), ce.ELL_UNFOLDED))),
    hold="zero-order hold over the period",
    meaning="u_a: commanded acceleration, realized up to the "
            "command bound towards the wall and up to the braking "
            "that the ground allows towards the start; u_l: "
            "commanded reach, counted as fully unfolded throughout "
            "a period in which the tool unfolds")

LEVELS = ScalarLevels(floor=ce.EPS_MIN, step=ce.GRID, top=ce.EPS_TOP)


# Where each item of the robot model comes from (Section 10.4).
SOURCES = {
    "coordinates": "Section 4.1.2: p, v and l along the track",
    "kinematics": "the bumper at p and the tool tip at p + l, which the "
                  "constraint p + l <= 25 m refers to",
    "dynamics": "a constant acceleration over the interval, as the "
                "segments of Chapter 3",
    "actuation": "|u_a| <= 2 m/s^2; u_l of 0 or 0.8 m (Section 4.1.2)",
    "norm": "max(|p| + |l|, |v| * 1 s), a length (Section 4.1.2)",
    "hold and meaning": "Section 10.2.2"}


def _motion(x, u, h):
    """The nominal state after an interval h under inputs u held
    over it: a constant acceleration, and the commanded reach."""
    p, v, _ = x
    return (p + v * h + u[0] * h * h / 2, v + u[0] * h, u[1])


# The robot model, supplied with the application as in Chapter 3.
# The robot adapter, the simulator adapter and the control package
# declare its state and actuation.
MODEL = RobotModel(
    name="cart.chapter-3",
    provider="supplied with the application (Chapter 3)",
    state=STATE,
    actuation=ACTUATION,
    points=(Quantity("bumper", "m", "track"),
            Quantity("tool tip", "m", "track")),
    kinematics=lambda x: (x[0], x[0] + x[2]),
    dynamics=_motion,
    assumptions=(
        "every input is realized as commanded; where the ground "
        "brakes less than a command asks, a capability label says "
        "so (Section 4.2)",
        "the reach is set at the start of a period and held"),
    sources=SOURCES)

fmt = ce.fmt
