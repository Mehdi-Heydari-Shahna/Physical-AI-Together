"""The design of the simulated cart and track, shared with Section 11.1, and
the settings of Isaac Sim that the design does not fix (Section 11.2).

Design, ground_profile and Profile are loaded from ../cart_mujoco without
running that package's __init__, which imports MuJoCo: Isaac Sim's Python
need not have MuJoCo installed, and both simulators build the same cart and
track from the same numbers."""
import os
import sys
import types
from dataclasses import asdict, dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
if "cart_mujoco" not in sys.modules:
    _pkg = types.ModuleType("cart_mujoco")
    _pkg.__path__ = [os.path.join(_HERE, "..", "cart_mujoco")]
    sys.modules["cart_mujoco"] = _pkg

from cart_mujoco.scene import Design, ground_profile  # noqa: E402
from cart_mujoco.track import Profile  # noqa: E402

__all__ = ["Design", "IsaacSettings", "Profile", "ground_profile"]


@dataclass(frozen=True)
class IsaacSettings:
    """The settings of the simulator, in SI units. Design.timestep and the
    MuJoCo fields of Design (integrator, solver, cone, iterations,
    tolerance) are not used here."""
    timestep: float = 0.001          # physics step [s]
    solver: str = "TGS"              # PhysX solver
    position_iterations: int = 32    # articulation solver iterations
    velocity_iterations: int = 1
    gpu_dynamics: bool = False       # PhysX on the CPU, read through NumPy
    broadphase: str = "MBP"
    friction_type: str = "patch"
    contact_offset: float = 0.02     # PhysX contact offset of every collider [m]
    rest_offset: float = 0.0         # PhysX rest offset of every collider [m]
    damping: float = 0.0             # linear and angular damping of every link [1/s]
    aux_mass: float = 1e-3           # each of the two auxiliary links of the planar joint [kg]
    aux_inertia: float = 1e-6        # their moments of inertia [kg m^2]

    def as_dict(self):
        return asdict(self)
