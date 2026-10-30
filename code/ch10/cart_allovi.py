#!/usr/bin/env python3
"""
The cart through the allovi core: the architecture of Chapter 10 checked
against the governor of Chapter 8.

Physical AI, Together, Chapter 10 (Sections 10.1 and 10.2).

The package allovi (next to this file) implements the common core of the
governor once, for every application. The package cart supplies the modules
of the cart of Chapter 3 through the interfaces of allovi.contract: a
control package (generator, test, rest segment and rest feedback), a
proposal method, five capability predictors, the tool operation with its
demand predictor, a robot adapter, a simulator adapter and an evaluator. The
model, the generator and the test are those of ../ch03/cart_example.py,
unchanged.

This script
  * runs the thirteen configurations of Chapter 8 through the core and
    compares every decision file and every trace file, byte for byte, with
    those that ../ch08/cart_governor.py wrote in ../ch08/results/;
  * shows what the registry refuses: a module without a method of its role,
    a simulator adapter that declares millimetres, a predictor of another
    capability space;
  * shows the defaults: a capability predictor that fails in every period
    is replaced by the global capability, in operation (strict=False), and
    the run is then that of the fixed predictor; with strict=True, the
    failure stops the run with a message that names the call and the
    period;
  * shows what the numerics of a simulator adapter can change: the four
    alarm runs with the first version of the adapter, which kept the
    floating-point state after a head that braked harder than the ground
    allows, compared with Chapter 8; and the exact states of the alarm runs
    against those of ../ch08/cart_governor.py, which agree in every printed
    digit;
  * counts the methods and records of the interfaces (Section 10.2).

Usage
    python3 cart_allovi.py            # writes results/
    python3 cart_allovi.py --check    # also checks every number quoted in
                                      # Chapter 10

Outputs, in results/ next to this file
    decisions-<run>.csv, trace-<run>.csv
                     the records of the thirteen runs, written by the core
                     (the columns of Chapter 8)
    summary.txt      the comparison with Chapter 8 and the numbers quoted in
                     the text

Python 3.8 or later; no other packages. The comparison needs the results of
../ch08/cart_governor.py, which are included.
"""
import argparse
import csv
import filecmp
import os
import sys
import tempfile
from fractions import Fraction as Fr

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import allovi                                    # noqa: E402
import cart                                      # noqa: E402
from allovi.records import candidate_rows, decision_rows, write_csv   # noqa: E402

ce, fmt = cart.ce, cart.fmt
CH08 = os.path.join(HERE, "..", "ch08", "results")

TRACE_COLS = ["k", "t", "index", "candidate", "ambition", "head", "label", "pairs",
              "rest", "ell", "level", "onset_level", "level_reason", "onset_reason",
              "admitted", "key1", "key2", "outcome"]
DECISION_COLS = ["k", "t", "p", "v", "p_init", "v_init", "mode", "proposed", "req",
                 "onset", "candidates", "admitted", "selected", "ambition", "level",
                 "onset_level", "reason"]
RUNTIME_COLS = ["k", "t", "p", "v", "governor", "mode", "proposed", "req", "onset",
                "candidates", "evaluated", "admitted", "selected", "ambition", "level",
                "onset_level", "reason"]

ONSET = 200
STALL = (10, 30)
SURVEY = {"0.2": ce.RAMP + Fr(1, 5), "0.35": ce.RAMP + Fr(7, 20)}

# name: (configuration arguments, periods, columns of the decision file)
RUNS = {
    "chapter3": (dict(postpone=None), 220, DECISION_COLS),
    "forecast-informed": (dict(postpone=21), 250, DECISION_COLS),
    "reactive-preparation": (dict(announce=None, postpone=21), 250, DECISION_COLS),
    "fixed-conservatism": (dict(announce=None, always=True, postpone=21), 250,
                           DECISION_COLS),
    "forecast-early": (dict(predicted=ONSET - 10, postpone=21), 250, DECISION_COLS),
    "forecast-late": (dict(predicted=ONSET + 10, postpone=21), 250, DECISION_COLS),
    "budget-one-candidate": (dict(postpone=None, budget=lambda k: 1), 220, RUNTIME_COLS),
    "budget-stall": (dict(postpone=None, budget=lambda k: 0 if STALL[0] <= k < STALL[1]
                          else None), 220, RUNTIME_COLS),
    "budget-no-evaluation": (dict(postpone=None, budget=lambda k: 0), 220, RUNTIME_COLS),
}
for b, start in SURVEY.items():
    for rest, suffix in (("maximal", ""), ("global", "-global")):
        RUNS[f"alarm-bias-{b}{suffix}"] = (
            dict(capability="cart.survey", capability_args={"ramp_start": start},
                 rest=rest, postpone=None), 220, RUNTIME_COLS)


def yes(b):
    return "yes" if b else "no"


def run_and_compare(out_dir):
    """Runs every configuration, writes its records and compares them with
    those of Chapter 8."""
    facts, records, beyond = {}, {}, {}
    n_files = n_same = n_dec = n_cand = 0
    for name, (args, periods, cols) in RUNS.items():
        cfg = cart.configuration(name, **args)
        rec = allovi.run(cfg, periods, strict=True, diagnostics=True)
        records[name] = (cfg, rec)
        beyond[name] = cfg.world.inexact
        dec = decision_rows(rec, cfg.control, cfg.robot)
        cand = candidate_rows(rec, cfg.control)
        n_dec, n_cand = n_dec + len(dec), n_cand + len(cand)
        for stem, columns, rows in (("decisions", cols, dec), ("trace", TRACE_COLS, cand)):
            path = os.path.join(out_dir, f"{stem}-{name}.csv")
            write_csv(path, columns, rows, fmt)
            n_files += 1
            n_same += filecmp.cmp(path, os.path.join(CH08, f"{stem}-{name}.csv"),
                                  shallow=False)
    some = [f"{n}: {v}" for n, v in beyond.items() if v]
    rest = sum(1 for v in beyond.values() if not v)
    facts["comparison with chapter 8"] = dict(
        runs=len(RUNS), files_compared=n_files, files_identical=n_same,
        decision_rows=n_dec, candidate_rows=n_cand,
        events_recorded=sum(len(r.events) for _, r in records.values()),
        periods_braking_beyond_the_ground=", ".join(some) +
        f"; the other {rest} runs: 0")
    return facts, records


def run_facts(records):
    cfg, rec = records["chapter3"]
    m = cfg.evaluator.evaluate(rec)
    out = {"forecasting run of chapter 3": dict(
        arrival_in_work_zone=f"{fmt(m['arrival_in_work_zone'])} s",
        time_in_work_zone=f"{fmt(m['time_in_work_zone'])} s",
        tool_unfolds=f"{fmt(m['tool_unfolds'])} s",
        least_clearance=f"{fmt(m['least_clearance'])} m",
        alarms=m["alarms"])}
    cfg, rec = records["alarm-bias-0.2"]
    m = cfg.evaluator.evaluate(rec)
    out["survey biased by 0.2 m"] = dict(
        alarm_at=f"{fmt(rec.alarms[0] * ce.T)} s",
        restart_committed_at=f"{fmt(rec.restarts[0] * ce.T)} s",
        tool_unfolds=f"{fmt(m['tool_unfolds'])} s")
    return out


# ----------------------------------------------------------------------------
# What the registry refuses (Section 10.2)
# ----------------------------------------------------------------------------
def refusals():
    out = {}
    registry = allovi.Registry()

    class NoRestart(cart.control.ConstantCommand):
        NAME = "cart.no-restart"
        restart = None                      # the method is missing
    try:
        registry.register(NoRestart)
        out["missing_method"] = "accepted"
    except allovi.RegistrationError as exc:
        out["missing_method"] = str(exc)

    millimetres = allovi.Actuation(
        inputs=(allovi.Quantity("u_a", "mm/s^2", "track"),
                allovi.Quantity("u_l", "m", "tool")),
        bounds=cart.model.ACTUATION.bounds, hold=cart.model.ACTUATION.hold,
        meaning=cart.model.ACTUATION.meaning)

    class TrackInMillimetres(cart.modules.Track):
        NAME = "cart.track-mm"
        ACTUATION = millimetres
    try:
        build_with(world=TrackInMillimetres())
        out["millimetres"] = "accepted"
    except allovi.ConfigurationError as exc:
        out["millimetres"] = str(exc)

    class OtherSpace(cart.modules.Fixed):
        NAME = "cart.fixed-friction"
        SPACE = "friction coefficient"
    try:
        build_with(capability=OtherSpace())
        out["capability_space"] = "accepted"
    except allovi.ConfigurationError as exc:
        out["capability_space"] = str(exc)
    return {"what the registry refuses": out}


def build_with(**replace):
    task = cart.modules.Operation(postpone=None)
    parts = {"control": cart.control.ConstantCommand(), "proposal": cart.modules.Naive(),
             "capability": cart.modules.Forecasting(),
             "demand": cart.modules.TaskPlan(task), "task": task,
             "robot": cart.modules.Cart(), "world": cart.modules.Track()}
    parts.update(replace)
    return allovi.build("refusal", ce.T, parts)


# ----------------------------------------------------------------------------
# Defaults for a failing module (Section 10.2)
# ----------------------------------------------------------------------------
class Failing(cart.modules.Forecasting):
    """A capability predictor whose labels are never ready: it raises."""
    NAME = "cart.failing"

    def labels(self, observation):
        raise TimeoutError("labels not ready before the deadline")


def defaults():
    allovi.REGISTRY.register(Failing)
    tolerant_cfg = cart.configuration("failing, operation", capability="cart.failing")
    tolerant = allovi.run(tolerant_cfg, 220, strict=False)
    fixed_cfg = cart.configuration("fixed", capability="cart.fixed")
    fixed = allovi.run(fixed_cfg, 220, strict=False)
    keys = ("selected", "ambition", "level", "onset_level", "reason", "p", "v")
    a = [{c: r.get(c) for c in keys}
         for r in decision_rows(tolerant, tolerant_cfg.control, tolerant_cfg.robot)]
    b = [{c: r.get(c) for c in keys}
         for r in decision_rows(fixed, fixed_cfg.control, fixed_cfg.robot)]
    try:
        allovi.run(cart.configuration("failing, strict", capability="cart.failing"), 220,
                   strict=True)
        strict = "no error"
    except allovi.ModuleError as exc:
        strict = str(exc)
    m_t = tolerant_cfg.evaluator.evaluate(tolerant)
    m_f = fixed_cfg.evaluator.evaluate(fixed)
    return {"defaults for a failing capability predictor": dict(
        strict=strict,
        operation_events=len(tolerant.events),
        first_event=f"period {tolerant.events[0][0]}, {tolerant.events[0][1]}: "
                    f"{tolerant.events[0][2]}",
        decisions_as_with_fixed_predictor=yes(a == b),
        arrival_in_work_zone=f"{fmt(m_t['arrival_in_work_zone'])} s",
        arrival_with_fixed_predictor=f"{fmt(m_f['arrival_in_work_zone'])} s",
        least_clearance=f"{fmt(m_t['least_clearance'])} m")}


# ----------------------------------------------------------------------------
# The numerics of a simulator adapter (Section 10.1)
# ----------------------------------------------------------------------------
class TrackAsComputed(cart.modules.Track):
    """The first version of the simulator adapter: after a head that braked
    harder than the ground allows, the state is kept as computed in floating
    point, without the correction of Track.on_the_ramp."""
    NAME = "cart.track-as-computed"

    def on_the_ramp(self, seg, p1f, v1f):
        return Fr(p1f), Fr(v1f)


def first_difference(path, reference):
    """The first row of a CSV file that differs from the reference file, as
    'k=..., <column> <value> instead of <value>, ...'; None if identical."""
    with open(path, newline="") as fa, open(reference, newline="") as fb:
        rows_a, rows_b = list(csv.reader(fa)), list(csv.reader(fb))
    header = rows_a[0]
    for ra, rb in zip(rows_a[1:], rows_b[1:]):
        if ra != rb:
            cols = [f"{c} {x or 'empty'} instead of {y or 'empty'}"
                    for c, x, y in zip(header, ra, rb) if x != y]
            where = f"k={ra[0]}" + (f", candidate {ra[2]}" if header[2] == "index" else "")
            return f"{where}: " + ", ".join(cols)
    if len(rows_a) != len(rows_b):
        return f"{len(rows_a) - 1} rows instead of {len(rows_b) - 1}"
    return None


def first_rest(rec):
    """The period record of the first period after the alarm that starts at
    rest."""
    return next(p for p in rec.periods if p.k >= rec.alarms[0] and p.state[1] == 0)


def adapter_numerics(records):
    allovi.REGISTRY.register(TrackAsComputed)
    out, differing, beyond = {}, [], []
    with tempfile.TemporaryDirectory() as tmp:
        for name, (args, periods, cols) in RUNS.items():
            if not name.startswith("alarm"):
                continue
            cfg = cart.configuration(name, world=TrackAsComputed.NAME, **args)
            rec = allovi.run(cfg, periods, strict=True, diagnostics=True)
            exact = first_rest(records[name][1])
            here = rec.periods[exact.k].state
            if here[1] != 0:
                raise RuntimeError(f"{name}: not at rest in period {exact.k}")
            beyond.append(f"{name}: {float(here[0] - exact.state[0]):.1e} m")
            for stem, columns, rows in (
                    ("decisions", cols, decision_rows(rec, cfg.control, cfg.robot)),
                    ("trace", TRACE_COLS, candidate_rows(rec, cfg.control))):
                fname = f"{stem}-{name}.csv"
                path = os.path.join(tmp, fname)
                write_csv(path, columns, rows, fmt)
                diff = first_difference(path, os.path.join(CH08, fname))
                if diff is not None:
                    differing.append(fname)
                    out[f"first difference, {fname}"] = diff
    out = dict(files_compared=8, files_differing=len(differing),
               differing=", ".join(differing),
               rest_point_beyond_the_exact_one=", ".join(beyond), **out)
    return {"simulator adapter without the correction": out}


def states_against_chapter_8(records):
    """The exact states of the four alarm runs against those of the script of
    Chapter 8, which keeps the floating-point state after the alarm and sets
    the exact point of rest when the rest feedback brings the cart to a stop;
    the records agree in every printed digit."""
    sys.path.insert(0, os.path.join(HERE, "..", "ch08"))
    import cart_governor as cg                   # noqa: E402
    out = {}
    for name, (args, periods, cols) in RUNS.items():
        if not name.startswith("alarm"):
            continue
        sched = cg.Schedule("chapter3", **cg.CHAPTER3)
        ref = cg.governed_run(cg.SurveyPredictor(args["capability_args"]["ramp_start"]),
                              sched, rest=args["rest"])
        theirs = {r["k"]: (r["p"], r["v"]) for r in ref["rows"] if "p" in r}
        rec = records[name][1]
        ours = {p.k: p.state[:2] for p in rec.periods}
        ours[len(rec.periods)] = rec.final_state[:2]
        if set(theirs) != set(ours):
            raise RuntimeError(f"{name}: the instants differ from those of Chapter 8")
        diffs = [abs(ours[k][0] - theirs[k][0]) for k in ours if ours[k] != theirs[k]]
        velocities = sum(1 for k in ours if ours[k][1] != theirs[k][1])
        out[name] = (f"{len(diffs)} instants, by at most {float(max(diffs)):.1e} m"
                     if diffs else "none") + f"; velocities differing: {velocities}"
    return {"exact states of the alarm runs against chapter 8": out}


def interface_facts():
    """The counts of Section 10.2: roles, methods and records."""
    from allovi import contract
    supplied = 1 + sum(len(methods) for _, methods in contract.SUPPLIED.values())
    return {"the interfaces": dict(
        roles=len(contract.ROLES),
        methods_of_the_roles=sum(len(m) for m in contract.ROLES.values()),
        methods_of_the_control_package=len(contract.ROLES["control"]),
        methods_of_plans_level_and_capability_spaces=supplied,
        records=len(contract.RECORDS),
        contract_version=contract.CONTRACT)}


def registered():
    names = {role: allovi.REGISTRY.names(role) for role in allovi.contract.ROLES}
    return {"registered modules": {role: ", ".join(n) for role, n in names.items() if n}}


def main():
    ap = argparse.ArgumentParser(description="Chapter 10: the cart through the allovi core")
    ap.add_argument("--check", action="store_true",
                    help="check every number quoted in Sections 10.1 and 10.2")
    args = ap.parse_args()
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    facts, records = run_and_compare(out_dir)
    facts.update(run_facts(records))
    facts.update(registered())
    facts.update(refusals())
    facts.update(defaults())
    facts.update(adapter_numerics(records))
    facts.update(states_against_chapter_8(records))
    facts.update(interface_facts())

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
        lines.append(f"[check] {n} values quoted in Sections 10.1 and 10.2: " +
                     ("all as quoted" if not failures else "FAILED: " + "; ".join(failures)))
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if failures:
        raise SystemExit(1)


EXPECTED = {     # every number and message quoted in Sections 10.1 and 10.2
    "comparison with chapter 8": {
        "runs": "13", "files_compared": "26", "files_identical": "26",
        "decision_rows": "3010", "candidate_rows": "20176", "events_recorded": "0",
        "periods_braking_beyond_the_ground":
            "alarm-bias-0.2: 1, alarm-bias-0.2-global: 1, alarm-bias-0.35: 1, "
            "alarm-bias-0.35-global: 1; the other 9 runs: 0"},
    "forecasting run of chapter 3": {"arrival_in_work_zone": "8.6 s"},
    "survey biased by 0.2 m": {
        "alarm_at": "5.6 s", "restart_committed_at": "9.9 s", "tool_unfolds": "20.1 s"},
    "registered modules": {
        "control": "cart.constant-command", "proposal": "cart.naive",
        "capability": "cart.fixed, cart.forecasting, cart.persisting, cart.reactive, "
                      "cart.survey",
        "demand": "cart.task-plan", "robot": "cart.robot", "world": "cart.track",
        "task": "cart.operation", "evaluator": "cart.metrics"},
    "what the registry refuses": {
        "missing_method": "NoRestart: missing method restart()",
        "millimetres": "refusal: actuation, world vs robot: u_a: unit 'mm/s^2' "
                       "differs from 'm/s^2'",
        "capability_space": "refusal: capability predictor issues labels of "
                            "'friction coefficient', the test accepts 'cart.braking'"},
    "defaults for a failing capability predictor": {
        "strict": "period 0: labels raised TimeoutError('labels not ready before "
                  "the deadline')",
        "operation_events": "220", "decisions_as_with_fixed_predictor": "yes",
        "arrival_in_work_zone": "9.9 s", "arrival_with_fixed_predictor": "9.9 s",
        "least_clearance": "0.505 m"},
    "simulator adapter without the correction": {
        "files_compared": "8", "files_differing": "4",
        "differing": "decisions-alarm-bias-0.2.csv, trace-alarm-bias-0.2.csv, "
                     "trace-alarm-bias-0.35.csv, trace-alarm-bias-0.35-global.csv",
        "rest_point_beyond_the_exact_one":
            "alarm-bias-0.2: 1.7e-15 m, alarm-bias-0.2-global: 1.7e-15 m, "
            "alarm-bias-0.35: 1.8e-15 m, alarm-bias-0.35-global: 1.8e-15 m",
        "first difference, decisions-alarm-bias-0.2.csv":
            "k=98: level 0.51 instead of 0.52, onset_level 0.51 instead of 0.52",
        "first difference, trace-alarm-bias-0.35.csv":
            "k=140, candidate 1: onset_level 1.38 instead of 1.39",
        "first difference, trace-alarm-bias-0.35-global.csv":
            "k=140, candidate 1: onset_level 1.38 instead of 1.39"},
    "exact states of the alarm runs against chapter 8": {
        "alarm-bias-0.2": "42 instants, by at most 1.7e-15 m; velocities differing: 0",
        "alarm-bias-0.2-global":
            "165 instants, by at most 1.7e-15 m; velocities differing: 0",
        "alarm-bias-0.35": "45 instants, by at most 1.8e-15 m; velocities differing: 0",
        "alarm-bias-0.35-global":
            "45 instants, by at most 1.8e-15 m; velocities differing: 0"},
    "the interfaces": {
        "roles": "8", "methods_of_the_roles": "19", "methods_of_the_control_package": "9",
        "methods_of_plans_level_and_capability_spaces": "6", "records": "4",
        "contract_version": "1.1"},
}


if __name__ == "__main__":
    main()
