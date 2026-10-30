"""The cart of Chapter 3 in Isaac Sim (Section 11.2).

  design.py  the design shared with Section 11.1, and the settings of Isaac Sim
  stage.py   the USD stage of the cart and its track, written from the design
  sim.py     the simulated cart: state, placement and the drive loop
  world.py   the simulator adapter cart.isaac for the allovi core

Import this package only after the SimulationApp has started: it imports
the USD and Omniverse modules that the application provides. Importing it
registers cart.isaac in the default registry of allovi, beside the modules
of the cart (../ch10).

Tested with: see ../README-isaac.md.
"""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ch10"))

import cart                                           # noqa: E402,F401 (registers the cart)

from .design import Design, IsaacSettings             # noqa: E402
from .sim import IsaacCart                            # noqa: E402
from .world import CartIsaac, exact                   # noqa: E402

__all__ = ["Design", "IsaacSettings", "IsaacCart", "CartIsaac", "exact"]
