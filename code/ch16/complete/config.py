"""The configurations of the complete case (Section 16.4, PROTOCOL.md).

A configuration is a control package, a proposal method, and whether the
governor receives the task plan through the demand predictor. Every
configuration has the same world, sensor, survey predictor with the
calibrated offset, task, operation, rule for postponing it, period, number
of periods, and computation (every fresh candidate in every period).

    A-rule-<pkg>      ALLOVI: demand predictor, naive proposal method
    A-preview-<pkg>   ALLOVI: demand predictor, preview planner
    A-learner-<pkg>   ALLOVI: demand predictor, the off-policy learner, trained
                      with the package on the training worlds (complete_train.py)
    F-preview-<pkg>   the filter: no demand predictor, preview planner
    F-rule-<pkg>      the filter: no demand predictor, naive proposal method

with <pkg> tube or mpc. The filter is the governor without a demand
predictor, as in Section 16.1: it certifies whatever the proposal method
proposes, and it receives nothing about the operation; the preview planner
reads the task plan itself, so that A-preview and F-preview receive the
same information through different modules, and A-rule and F-preview the
same information as well.
"""
import allovi
from cart import modules as cm

from cart_mujoco.scene import Design

from . import model as md
from .modules import Learner, Metrics, MuJoCoCart, Preview, Robot, Rule, Survey
from .packages import MPC, Tube

PERIODS = 320              # 32 s, as in the anticipation benchmark
POSTPONE = 21              # periods, as in Section 8.2.3
PACKAGES = {"tube": Tube, "mpc": MPC}
CONFIGS = ("A-rule", "A-preview", "A-learner", "F-preview", "F-rule")


def operation(world):
    """The tool operation of the world, with the task plan that the demand
    predictor (or the preview planner) reads: announced lead periods before
    the forecast onset; due at the true onset; postponed by 21 periods after
    an overload, as in Section 8.2.3."""
    return cm.Operation(announce=world.announce, onset=world.onset,
                        predicted=world.onset_forecast, postpone=POSTPONE)


LEARNERS = {}
"""The trained learners' tables, by package (load_learners)."""


def load_learners(folder):
    """Reads the tables that complete_train.py wrote, results/complete-learner-<pkg>.json."""
    import json
    import os
    for pkg in PACKAGES:
        path = os.path.join(folder, f"complete-learner-{pkg}.json")
        if os.path.exists(path):
            with open(path) as fh:
                LEARNERS[pkg] = json.load(fh)["table"]


def configure(world, name, package, bounds, offset, record=False, sigma=None,
              proposal=None, task_on=True, design_kw=None, drive_mass=None, surveyed=None):
    """The configuration 'name' (one of CONFIGS) with the package 'package'
    ('tube' or 'mpc') in 'world', for the model's bounds and the survey's
    offset. Returns the configuration and the world adapter. 'proposal'
    replaces the proposal method (training); task_on=False runs without the
    operation and the demand predictor (training). design_kw, drive_mass and
    surveyed change the world for the stress tests (complete_stress.py): the
    design of the simulated cart, the drive's mass, and the surveyed start."""
    kind, method = name.split("-")
    control = PACKAGES[package](bounds=bounds)
    design = Design(ramp_start=world.ramp_start, **(design_kw or {}))
    from .plant import Sensor
    sig = Sensor.SIGMA if sigma is None else sigma
    sim = MuJoCoCart(design, world.sensor_seed, sig, record, drive_mass=drive_mass)
    robot = Robot(world.boundary_seed, bounds, sig)
    task = operation(world) if task_on else None
    survey = Survey(world.surveyed if surveyed is None else surveyed, offset)
    if proposal is None:
        if method == "rule":
            proposal = Rule()
        elif method == "preview":
            proposal = Preview(task, control.declared())
        elif method == "learner":
            proposal = Learner.from_table(control.declared(), LEARNERS[package])
        else:
            raise ValueError(method)
    if isinstance(proposal, Learner):
        proposal.survey = survey
    demand = cm.TaskPlan(task) if (kind == "A" and task is not None) else None
    evaluator = Metrics(world, task, sim, robot, bounds) if task is not None else None
    cfg = allovi.build(f"{name}-{package}", md.T,
                       dict(control=control, robot=robot, world=sim, proposal=proposal,
                            capability=survey, demand=demand, task=task,
                            evaluator=evaluator))
    return cfg, sim


def run(world, name, package, bounds, offset, periods=PERIODS, record=False, **kw):
    """One offline run (the core's loop, allovi.run): the record and the metrics."""
    cfg, sim = configure(world, name, package, bounds, offset, record, **kw)
    rec = allovi.run(cfg, periods)
    return rec, cfg.evaluator.evaluate(rec), sim
