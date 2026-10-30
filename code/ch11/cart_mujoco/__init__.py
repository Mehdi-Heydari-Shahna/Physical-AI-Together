"""The cart of Chapter 3 in MuJoCo (Section 11.1).

  track.py   the profile of the track and the track coordinate
  scene.py   the design and the MJCF model generated from it
  sim.py     the simulated cart: state, placement and the drive loop
  world.py   the simulator adapter cart.mujoco for the allovi core

Importing the package registers cart.mujoco in the default registry of
allovi, beside the modules of the cart (../ch10).

Requires MuJoCo 3.16.0 (see ../requirements.txt) and Python 3.10 or later.
"""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ch10"))

import cart                                           # noqa: E402,F401 (registers the cart)

from .scene import Design, mjcf                       # noqa: E402
from .sim import Sim                                  # noqa: E402
from .world import CartMuJoCo, exact                  # noqa: E402

__all__ = ["Design", "mjcf", "Sim", "CartMuJoCo", "exact"]
