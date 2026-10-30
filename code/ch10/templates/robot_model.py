"""Template of a robot model (Section 10.4; allovi/model.py).

A robot model is not a role. It is the record through which a model
provider, an application's own model or a tool such as RoboCompiler,
delivers coordinates, kinematics, dynamics and actuation conventions; the
robot adapter, the simulator adapter and the control package declare its
state and actuation (see cart/model.py).

Obligations, which no check discharges: the model is nominal, and whatever a
world may do differently enters through capability labels; the obligations
of Table 4.1 stay with the modules that use the model. The application
states and justifies the norm, the hold and the meaning of the commands,
which a mechanical model does not contain, and any bounds of the inputs that
the provider does not give.

Checks that allovi.model runs, each only on a model in which every item of
the boundary is present (status): kinematics and dynamics that return what
the model declares for sample states and inputs (check_model); declarations
of the robot adapter, the simulator adapter and the control package equal to
the model's (agreement); and, in simulation, the motion that the simulator
adapter realizes, compared with the nominal dynamics in cases where the
simulated world matches the model's nominal world, or within a tolerance
that covers the variation among the admitted worlds (motion_agreement).

HELPERS.md in this folder documents the names that cart/model.py offers to
the cart's modules, MODEL among them.
"""
from allovi import Actuation, Bound, Quantity, RobotModel, Schema


def kinematics(state):
    """The points of POINTS, in order, at a state."""
    raise NotImplementedError


def dynamics(state, inputs, duration):
    """The nominal state after 'duration' under 'inputs' held over it."""
    raise NotImplementedError


# components of the state, with units and frames, and the norm
STATE = Schema("my robot state", (Quantity("x", "m", "my frame"),),
               norm="the norm of the state, which fixes the unit of the reserve")
# inputs with units and frames; a bound for every input; hold and meaning
ACTUATION = Actuation(inputs=(Quantity("u", "N", "my frame"),),
                      bounds=(Bound("u", low=None, high=None),),
                      hold="zero-order hold over the period",
                      meaning="what is commanded, and what is realized")
# the points that constraints and labels refer to
POINTS = (Quantity("tip", "m", "my frame"),)

MY_MODEL = RobotModel(
    name="template.robot-model",
    provider="who produced the model, and from what",
    state=STATE, actuation=ACTUATION, points=POINTS,
    kinematics=kinematics, dynamics=dynamics,
    assumptions=("what the nominal dynamics assumes",),
    sources={},       # item -> where it comes from
    elsewhere={})     # item -> where the provider offers it outside this record
