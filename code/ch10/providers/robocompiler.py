"""A robot model from an export of RoboCompiler (Section 10.4).

RoboCompiler compiles a canonical mechanism graph (bodies, joints,
attachment frames, inertias and actuator ports) into consistent interfaces
for assembly, motion, actuation and rigid-body dynamics, including robots
with closed kinematic chains. Its compile step for the six-UPS Stewart
platform writes three files: physical_graph.json (the input graph),
compiled_model.json (the compiled tree, its point closures and its
coordinates) and topology_plan.json (the plan of the compilation).

This adapter reads the three files with the standard library and fills the
items of the boundary that they carry. It takes the independent coordinates
of the compiled model, with their rates, as the state; another adapter
could take the pose of the platform, which RoboCompiler's assembly computes
from them. It does not call RoboCompiler's code. The kinematics and the
dynamics are functions of RoboCompiler, which need NumPy and SciPy, and the
force limits are in the mechanism graph of RoboCompiler's simulation studies;
the adapter records where they are. The norm and the hold of the commands
are in none of them, and the meaning of the commands only in part: whoever
adapts the model for a control package supplies and justifies them, and the
checks of allovi.model refuse the model until every item is in the record.

Python 3.8 or later; no other packages.
"""
import collections
import json
import os

from allovi import Actuation, Quantity, RobotModel, Schema

UNITS = {"prismatic": ("m", "m/s"), "revolute": ("rad", "rad/s")}
FILES = ("physical_graph.json", "compiled_model.json", "topology_plan.json")
KINEMATICS = ("RoboCompiler's code: assembly from the independent coordinates, on the "
              "branch that a seed selects, body poses and tangent maps (PointGraph and "
              "PACDM, vendor/pacdm_original.py); it needs NumPy and SciPy")
DYNAMICS = ("RoboCompiler's code: closure maps and the curvature term, with the tree "
            "inertia and bias of a rigid-body evaluator: for the compiled model, the "
            "NumPy tree evaluator of the compiler workflow, checked against Pinocchio "
            "when it is installed; in the simulation studies, Pinocchio "
            "(stewart/projected.py)")
LIMITS = ("a force limit of 900 N and a servo on the leg lengths, in the mechanism graph "
          "of the simulation studies (data/stewart.cmg.json)")


def load(folder):
    """The three files of an export, as dictionaries."""
    out = []
    for name in FILES:
        with open(os.path.join(folder, name), encoding="utf-8") as fh:
            out.append(json.load(fh))
    return out


def from_export(folder):
    """The robot model that an export of RoboCompiler provides, and what the
    export contains (a dictionary of counts)."""
    graph, compiled, plan = load(folder)
    if graph.get("units") != "SI" or compiled.get("units") != "SI":
        raise ValueError("the export does not declare SI units")
    joints = {j["id"]: j for j in compiled["joints"]}
    coordinates, rates = [], []
    for name in compiled["independent_ids"]:
        joint = joints[name]
        position, velocity = UNITS[joint["type"]]
        frame = f"axis of {name} in {joint['base_body']}"
        coordinates.append(Quantity(name, position, frame))
        rates.append(Quantity(name + "_rate", velocity, frame))
    state = Schema(compiled["id"], tuple(coordinates + rates), norm="")
    inputs = tuple(Quantity(f"{a['joint_id']}_{a['effort_type']}", a["unit"],
                            f"along {a['joint_id']}") for a in graph["actuators"])
    actuation = Actuation(inputs=inputs, bounds=(), hold="", meaning="")
    ids = compiled["independent_ids"]
    strokes = sorted({(joints[n]["limits"]["lower"], joints[n]["limits"]["upper"],
                       UNITS[joints[n]["type"]][0]) for n in ids})
    stroke = (f"{strokes[0][0]} to {strokes[0][1]} {strokes[0][2]}"
              if len(strokes) == 1 else "not the same for every coordinate")
    model = RobotModel(
        name=compiled["id"],
        provider="RoboCompiler, export of its compile step",
        state=state,
        actuation=actuation,
        sources={
            "coordinates": "compiled_model.json: the independent coordinates and the "
                           "types and frames of their joints; SI units",
            "norm": "not in the export",
            "hold and meaning": "not in the export, which names each command a force "
                                "in N on its joint but states neither its hold over a "
                                "period nor the effort that is realized; the graph of "
                                "the simulation studies declares ideal force "
                                "actuators, and no hold"},
        elsewhere={
            "kinematics": KINEMATICS,
            "dynamics": DYNAMICS,
            "actuation": "the export names one force in N on each actuated joint and "
                         f"gives the strokes of those joints, {stroke}, but no force "
                         f"limit; {LIMITS}"})
    kinds = collections.Counter(j["type"] for j in graph["joints"])
    contents = {
        "bodies": len(graph["bodies"]),
        "joints": len(graph["joints"]),
        "joints by type": ", ".join(f"{n} {t}" for t, n in sorted(kinds.items(),
                                                                key=lambda kv: (-kv[1], kv[0]))),
        "actuators": len(graph["actuators"]),
        "actuator units": ", ".join(sorted({a["unit"] for a in graph["actuators"]})),
        "physical cycles": plan["physical_cycles"],
        "point closures": len(compiled["closures"]),
        "tree coordinates": len(compiled["coordinate_ids"]),
        "independent coordinates": len(compiled["independent_ids"]),
        "independent coordinate joints": ", ".join(sorted({joints[n]["type"] for n in
                                                          compiled["independent_ids"]})),
        "strokes of the independent coordinates": stroke,
        "point residual rows": plan["point_residual_rows"],
        "augmented coordinates": plan["augmented_coordinates"],
        "augmented residual rows": plan["augmented_residual_rows"],
        "declared actuators": plan["declared_actuators"],
        "dimensional mobility": plan["dimensional_mobility"],
    }
    return model, contents
