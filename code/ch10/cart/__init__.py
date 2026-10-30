"""The cart of Chapter 3 as a set of registered modules for the allovi core.

Importing the package registers its modules in the default registry of
allovi; configuration() assembles those of one run. The model, the
generator and the test are those of ../ch03/cart_example.py, unchanged.
"""
from . import control, modules            # noqa: F401  (registers the modules)
from .model import T, ce, fmt

import allovi


def configuration(name, capability="cart.forecasting",
                  capability_args=None, rest="maximal", budget=None,
                  world="cart.track", **operation):
    """The modules of one run of the cart: the configuration of Chapter 3,
    with the given capability predictor, rest feedback, time budget,
    simulator adapter and schedule of the tool operation (Section 8.2.3).

    The control package, the proposal method, the demand predictor, the
    robot adapter and the evaluator are always the cart's. To run another
    module in the place of one of them, build the
    dictionary below with its name, as ../templates/HELPERS.md shows; the
    task is created once, and the demand predictor and the evaluator receive
    that instance."""
    task = modules.Operation(**operation)
    return allovi.build(name, T, {
        "control": ("cart.constant-command", {"rest": rest}),
        "proposal": ("cart.naive",),
        "capability": (capability, capability_args or {}),
        "demand": ("cart.task-plan", {"task": task}),
        "task": task,
        "robot": ("cart.robot",),
        "world": (world,),
        "evaluator": ("cart.metrics", {"task": task}),
    }, budget=budget)
