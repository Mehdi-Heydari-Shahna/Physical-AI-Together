"""The cart of Chapter 3 in Newton and Genesis (Section 12.1).

  design.py         the design and the MJCF text of Section 11.1, without MuJoCo
  base.py           the track coordinate, placement and the drive loop, for any simulator
  newton_cart.py    the cart in Newton, with a choice of Newton's solvers
  genesis_cart.py   the cart in Genesis
  world.py          the simulator adapters cart.newton and cart.genesis

Importing the package registers cart.newton and cart.genesis in the default
registry of allovi, beside the modules of the cart (../../ch10); each
imports its simulator only when it is created.
"""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ch10"))

import cart                                           # noqa: E402,F401 (registers the cart)

from .base import G, Kin, PortedCart                  # noqa: E402
from .design import Design, Profile, mjcf             # noqa: E402
from .world import CartGenesis, CartNewton, exact     # noqa: E402

__all__ = ["Design", "Profile", "mjcf", "G", "Kin", "PortedCart", "CartNewton",
           "CartGenesis", "exact"]
