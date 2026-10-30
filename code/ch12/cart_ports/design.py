"""The design of the simulated cart and track, shared with Section 11.1.

Design, mjcf, ground_profile and Profile are loaded from
../../ch11/cart_mujoco without running that package's __init__, which
imports MuJoCo 3.16.0 and registers cart.mujoco: the Python environments of
this chapter carry the MuJoCo release that their simulator requires, if any,
and every simulator builds the same cart and track from the same numbers
and the same MJCF text."""
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
if "cart_mujoco" not in sys.modules:
    _pkg = types.ModuleType("cart_mujoco")
    _pkg.__path__ = [os.path.join(_HERE, "..", "..", "ch11", "cart_mujoco")]
    sys.modules["cart_mujoco"] = _pkg

from cart_mujoco.scene import Design, ground_profile, mjcf  # noqa: E402
from cart_mujoco.track import Profile  # noqa: E402

__all__ = ["Design", "Profile", "ground_profile", "mjcf"]
