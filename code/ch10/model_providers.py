#!/usr/bin/env python3
"""
The robot-model adapter: the model of the cart, and an export of RoboCompiler.

Physical AI, Together, Chapter 10 (Section 10.4).

A robot model reaches the control package, the predictors and the simulator
adapter through one record, allovi.RobotModel, whoever provides it. This
script
  * reports the items of the boundary that the cart's model, supplied with
    the application as in Chapter 3, provides, and runs the checks of
    allovi.model on it: the model returns what it declares; the robot
    adapter, the simulator adapter and the control package of the cart
    declare what the model declares; and the simulator adapter realizes the
    model's nominal motion for commands that every admitted world realizes;
  * runs the same checks with a simulator adapter, defined here, that
    declares metres and reports millimetres: its declarations agree, so the
    configuration is built, and the check of its motion against the model
    finds the mistake; the check suite of Section 10.3 is run on it too;
  * reads the export of RoboCompiler's compile step for its six-UPS Stewart
    platform (providers/robocompiler-stewart) with providers/robocompiler.py,
    and reports what the export contains and which items of the boundary it
    provides.

Usage
    python3 model_providers.py            # writes results/models-summary.txt
    python3 model_providers.py --check    # also checks every number quoted
                                          # in Section 10.4

Python 3.8 or later; no other packages.
"""
import argparse
import os
import sys
from fractions import Fraction as Fr

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import allovi                                    # noqa: E402
import cart                                      # noqa: E402
from allovi import checks                        # noqa: E402
from allovi import model as am                   # noqa: E402
from providers import robocompiler               # noqa: E402

ce, fmt, T = cart.ce, cart.fmt, cart.T
MODEL = cart.model.MODEL
EXPORT = os.path.join(HERE, "providers", "robocompiler-stewart")

# Sample states (p, v, l) and inputs (u_a, u_l) for the checks of the model.
STATES = [(Fr(0), Fr(0), Fr(0)), (Fr(5), Fr(1), Fr(0)), (Fr(39, 2), Fr(2), Fr(0)),
          (Fr(21), Fr(3, 2), Fr(4, 5)), (Fr(237, 10), Fr(0), Fr(4, 5))]
COMMANDS = [Fr(2), Fr(0), -ce.B_GLOBAL]          # towards the wall; none; braking
REACHES = [Fr(0), ce.ELL_UNFOLDED]


def motion_cases():
    """States and inputs in which every admitted world realizes the command
    as the model assumes: an acceleration towards the wall, or none, which
    the ground always realizes, and braking at the global capability, which
    the world of every period allows (S2), from a forward motion that it does
    not reverse within the period. The commands are accelerations, so a
    world that realizes them moves the cart as the model does. Each case
    carries the law of the control package that applies the inputs."""
    cases = []
    for x in STATES:
        for u in COMMANDS:
            if u < 0 and x[1] + u * T < 0:
                continue                         # it would reverse the motion
            for ell in REACHES:
                cases.append((x, (u, ell), cart.control.Law(u, ell, "segment")))
    return cases


@allovi.register
class TrackInMillimetres(cart.modules.Track):
    """The cart's simulator adapter with one mistake: it reports positions
    and reaches in millimetres and velocities in millimetres per second, and
    reads the states that it is given in the same units, while it declares
    metres, as the cart's model does."""
    NAME = "broken.track-in-millimetres"

    def reset(self):
        super().reset()
        return (Fr(0), Fr(0), Fr(0))

    def step(self, state, law):
        metres = tuple(x / 1000 for x in state)
        return tuple(x * 1000 for x in super().step(metres, law))


def configuration(name, world="cart.track"):
    """The configuration of the forecasting run of Chapter 3, with the given
    simulator adapter."""
    return cart.configuration(name, world=world)


def cart_facts():
    out = {}
    st = am.status(MODEL)
    out["items provided"] = sum(1 for _, s, _ in st if s == am.PROVIDED)
    out["items of the boundary"] = len(am.ITEMS)
    problems = am.check_model(MODEL, STATES, [(u, l) for u in COMMANDS for l in REACHES], T)
    out["model check: sample states, inputs"] = f"{len(STATES)}, {len(COMMANDS) * len(REACHES)}"
    out["model check: problems"] = len(problems)
    cfg = configuration("cart")
    out["declarations of robot, world and control vs model: problems"] = \
        len(am.agreement(MODEL, cfg.robot, cfg.world, cfg.control))
    diffs, n = am.motion_agreement(MODEL, cfg.world, motion_cases(), T, fmt=fmt)
    out["motion of the track vs model: cases"] = n
    out["motion of the track vs model: differences"] = len(diffs)
    return out


def millimetre_facts():
    out = {}
    try:
        cfg = configuration("millimetres", world="broken.track-in-millimetres")
        out["configuration built"] = "yes"
    except allovi.ConfigurationError as exc:          # not expected: declarations agree
        out["configuration built"] = f"no: {exc}"
        return out
    out["declarations vs model: problems"] = \
        len(am.agreement(MODEL, cfg.robot, cfg.world, cfg.control))
    cases = motion_cases()
    diffs, n = am.motion_agreement(MODEL, cfg.world, cases, T, fmt=fmt)
    out["motion vs model: differences"] = f"{len(diffs)} of {n}"
    out["motion vs model: first difference"] = diffs[0] if diffs else "none"
    out["motion vs model: differences in which it raised"] = \
        sum(1 for d in diffs if "raised" in d)
    agree = [c for c in cases if not am.motion_agreement(MODEL, cfg.world, [c], T)[0]]
    out["motion vs model: cases that agree"] = (
        f"{len(agree)}, " + ("all with u_a = 0 and u_l = 0"
                             if all(u == (0, 0) for _, u, _ in agree) else "not all at rest"))
    record = allovi.run(cfg, 2, strict=True)
    out["run: first alarm"] = ", ".join(f"{fmt(a * T)} s" for a in record.alarms) or "none"
    results = checks.check(lambda: configuration("millimetres",
                                                 world="broken.track-in-millimetres"),
                           220, fmt)
    out["check suite of Section 10.3"] = checks.summary(results)
    failed = [r for r in results if r.status == checks.FAILED]
    out["check suite: first failure"] = f"{failed[0].name}: {failed[0].detail}" if failed \
        else "none"
    return out


def export_facts():
    model, contents = robocompiler.from_export(EXPORT)
    out = dict(contents)
    out["state components"] = len(model.state.quantities)
    out["state units"] = ", ".join(sorted({q.unit for q in model.state.quantities}))
    for item, st, _ in am.status(model):
        out[f"item: {item}"] = st
    out["model check"] = "; ".join(am.check_model(model, [], [], T))
    return out


def main():
    ap = argparse.ArgumentParser(description="Section 10.4: the robot-model adapter")
    ap.add_argument("--check", action="store_true",
                    help="check every number quoted in Section 10.4")
    args = ap.parse_args()
    facts = {"the cart's model": cart_facts(),
             "a simulator adapter that reports millimetres": millimetre_facts(),
             "the export of RoboCompiler (six-UPS Stewart platform)": export_facts()}
    lines, failures = [], []
    for group, items in facts.items():
        lines.append(f"[{group}]")
        for key, val in items.items():
            lines.append(f"  {key} = {val}")
            want = EXPECTED.get(group, {}).get(key)
            if args.check and want is not None and want != str(val):
                failures.append(f"{group}.{key}: expected {want!r}, got {val!r}")
    if args.check:
        failures += [f"{group}.{key}: not computed" for group, items in EXPECTED.items()
                     for key in items if key not in facts.get(group, {})]
        n = sum(len(v) for v in EXPECTED.values())
        lines.append(f"[check] {n} values quoted in Section 10.4: " +
                     ("all as quoted" if not failures else "FAILED: " + "; ".join(failures)))
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "models-summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if failures:
        raise SystemExit(1)


EXPECTED = {   # every number and message quoted in Section 10.4
    "the cart's model": {
        'items provided': '6',
        'items of the boundary': '6',
        'model check: sample states, inputs': '5, 6',
        'model check: problems': '0',
        'declarations of robot, world and control vs model: problems': '0',
        'motion of the track vs model: cases': '26',
        'motion of the track vs model: differences': '0',
    },
    'a simulator adapter that reports millimetres': {
        'configuration built': 'yes',
        'declarations vs model: problems': '0',
        'motion vs model: differences': '21 of 26',
        'motion vs model: first difference':
            'from (0, 0, 0) under (2, 0): the model gives (0.01, 0.2, 0), '
            'the simulator adapter (10, 200, 0)',
        'motion vs model: differences in which it raised': '6',
        'motion vs model: cases that agree': '5, all with u_a = 0 and u_l = 0',
        'run: first alarm': '0.2 s',
        'check suite of Section 10.3': '7 passed, 1 failed, 5 not applicable',
        'check suite: first failure':
            'outputs of the contracted kinds: period 2: world.step raised '
            "NotImplementedError('this simulator adapter realizes the "
            'rest feedback only while it brakes a forward motion on the '
            "ramp, as after every alarm of the book'); it has no default",
    },
    'the export of RoboCompiler (six-UPS Stewart platform)': {
        'bodies': '21',
        'joints': '25',
        'joints by type': '12 revolute, 6 prismatic, 6 spherical, 1 fixed',
        'actuators': '6',
        'actuator units': 'N',
        'physical cycles': '5',
        'point closures': '5',
        'tree coordinates': '21',
        'independent coordinates': '6',
        'independent coordinate joints': 'prismatic',
        'strokes of the independent coordinates': '0.5 to 0.91 m',
        'point residual rows': '15',
        'augmented coordinates': '36',
        'augmented residual rows': '30',
        'declared actuators': '6',
        'dimensional mobility': '6',
        'state components': '12',
        'state units': 'm, m/s',
        'item: coordinates': 'provided',
        'item: kinematics': 'elsewhere',
        'item: dynamics': 'elsewhere',
        'item: actuation': 'elsewhere',
        'item: norm': 'missing',
        'item: hold and meaning': 'missing',
        'model check':
            'kinematics: elsewhere; dynamics: elsewhere; actuation: '
            'elsewhere; norm: missing; hold and meaning: missing',
    },
}


if __name__ == "__main__":
    main()
