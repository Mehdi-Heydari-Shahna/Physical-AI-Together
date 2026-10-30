"""Template of a robot adapter (Section 10.3; Section S10.9.2).

Obligations, which no check discharges: a valid measurement-error bound
(S5); a state and inputs declared with their units, frames, norm and
actuator semantics.

Checks that allovi.checks runs: registration; declarations of the right type
that agree with those of the simulator adapter and of the control package.

For a module of the cart, HELPERS.md in this folder documents the cart's
state (p, v, l), its declarations in cart/model.py, and how to run and
check a module in the cart's configurations.
"""


class MyRobotAdapter:
    ROLE, NAME, CONTRACT = "robot", "template.robot-adapter", "1.0"
    STATE = None              # allovi.Schema: components, units, frames, norm
    ACTUATION = None          # allovi.Actuation: inputs, bounds, hold, meaning
    measurement_bound = None  # the bound on the measurement error, in the norm

    def measure(self, state):
        """The measurement of a state that the simulator adapter returned."""
        raise NotImplementedError
