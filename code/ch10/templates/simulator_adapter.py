"""Template of a simulator adapter (Section 10.3; Section 10.2).

Obligations, which no check discharges: it realizes the laws of the control
package and the actuation that it declares, and returns states as its STATE
declares; for a simulator, its numerics belong to the record of every
experiment (Section 10.1).

Checks that allovi.checks runs: registration; declarations that agree with
those of the robot adapter; laws that it receives within the declared
bounds; states with the declared components; runs that repeat exactly. With
admits() and clearance(), it serves as the oracle of the falsification
checks.

For a module of the cart, HELPERS.md in this folder documents the cart's
state, the parameter ce.RAMP through which the scripts of Chapters 7, 16
and 17 set the world of a run, and how to run and check an adapter in the
cart's configurations.
"""


class MySimulatorAdapter:
    ROLE, NAME, CONTRACT = "world", "template.simulator-adapter", "1.0"
    STATE = None          # allovi.Schema, as the robot adapter declares it
    ACTUATION = None      # allovi.Actuation, as the robot adapter declares it

    def reset(self):
        """The initial state."""
        raise NotImplementedError

    def step(self, state, law):
        """The state at the end of one period under the law, from 'state'."""
        raise NotImplementedError

    # Optional, the oracle of the checks of Section 10.3 (simulation only):
    # def admits(self, label, segment, k): is the world of period k admitted
    #     by the label on the region of the segment?
    # def clearance(self, state): the true reserve of the state: its
    #     distance, in the norm of the state, to the states that violate the
    #     constraints
