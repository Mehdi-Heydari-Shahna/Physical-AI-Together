"""The complete case: the cart's task in MuJoCo, end to end (Section 16.4).

  model.py      the declared model, its bounds and the certificates
  plant.py      the MuJoCo cart of Chapter 11 executing feedback laws, with a sensor
  packages.py   the control packages complete.tube and complete.mpc
  modules.py    the simulator and robot adapters, the survey predictor, the
                proposal methods complete.rule and complete.preview, the evaluator
  worlds.py     the worlds and the seeds of each phase of the protocol
  config.py     the configurations of the protocol
  realtime.py   the governor and the plant in two processes, on the wall clock

Importing the package puts ../../ch10 (the allovi core and the cart) and
../../ch11 (cart_mujoco) on the path and registers the modules.
Requires MuJoCo 3.16.0 and NumPy (../../ch11/requirements.txt).
"""
import os
import sys

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "..", "..", "ch10"), os.path.join(_HERE, "..", "..", "ch11")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cart                                            # noqa: E402,F401 (task, demand)
import cart_mujoco                                     # noqa: E402,F401

from . import model, packages, modules, worlds        # noqa: E402,F401
