#!/usr/bin/env python3
"""
Compatibility checks and verification of modules of the cart.

Physical AI, Together, Chapter 10 (Section 10.3).

The checks of allovi.checks are of four kinds: interface checks, which find
modules that do not fit the core; consistency checks, which test the outputs
of the modules for properties that their specifications imply; in
simulation, falsification checks against an oracle that knows the world;
and the checks that a module supplies itself. A failed check shows a
mistake, or an obligation that does not hold in the cases run; a passed
check discharges no obligation.

This script runs the checks, on the configuration of the forecasting run of
Chapter 3 (220 periods), with:
  * the modules of the cart (the reference);
  * the survey predictor of Section 7.2.3, biased by 0.2 m;
  * two broken control packages, defined here: one whose test forgets the
    tool, and one whose test accepts braking up to 0.01 m/s^2 beyond the
    label; each with the naive and with the eager proposal method;
  * the two example contributions in examples/: the eager proposal method,
    and the survey predictor used with a margin, of 0.25 m and of 0.1 m.
It also checks that each template in templates/ has every method of its
role, and runs the path of a contribution (--module) on both examples.

Usage
    python3 cart_checks.py                  # writes results/checks-*.txt
    python3 cart_checks.py --check          # also checks every number quoted
                                            # in Section 10.3
    python3 cart_checks.py --module FILE    # the checks of every module that
                                            # FILE registers, each in its role
                                            # (a task excepted)

Outputs, in results/ next to this file
    checks-<case>.txt      the report of the checks of one case
    checks-summary.txt     the numbers quoted in Section 10.3

Python 3.8 or later; no other packages.
"""
import argparse
import importlib.util
import math
import os
import sys
from fractions import Fraction as Fr

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import allovi                                    # noqa: E402
import cart                                      # noqa: E402
from allovi import checks                        # noqa: E402
from allovi.records import candidate_rows, decision_rows   # noqa: E402

ce, fmt = cart.ce, cart.fmt
PERIODS = 220
_LOADED = {}


# ----------------------------------------------------------------------------
# Loading a contribution: the modules that a file registers
# ----------------------------------------------------------------------------
def load(path):
    """Imports a file once and returns the (role, name) of every module that
    it registered in the default registry."""
    path = os.path.abspath(path)
    if path not in _LOADED:
        before = {(r, n) for r in allovi.contract.ROLES for n in allovi.REGISTRY.names(r)}
        name = "contribution_" + str(len(_LOADED))
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        after = {(r, n) for r in allovi.contract.ROLES for n in allovi.REGISTRY.names(r)}
        _LOADED[path] = sorted(after - before)
    return _LOADED[path]


def reference(name, **replace):
    """The configuration of the forecasting run of Chapter 3, with modules
    replaced by role: role=(registered name, arguments)."""
    task = cart.modules.Operation()
    parts = {"control": ("cart.constant-command",), "proposal": ("cart.naive",),
             "capability": ("cart.forecasting",), "demand": ("cart.task-plan", {"task": task}),
             "task": task, "robot": ("cart.robot",), "world": ("cart.track",),
             "evaluator": ("cart.metrics", {"task": task})}
    parts.update(replace)
    return allovi.build(name, cart.T, parts)


TASK_NOT_RUN = ("not run: the cart's demand predictor and evaluator are built with the "
                "cart's task; check a task with allovi.checks.check and a configuration "
                "of your own")


def check_file(path):
    """The path of a contribution: the checks of every module that the file
    registers, each in its role in the reference configuration; None for a
    task, which this configuration cannot take (TASK_NOT_RUN)."""
    reports = {}
    for role, name in load(path):
        if role == "task":
            reports[(role, name)] = None
            continue
        reports[(role, name)] = checks.check(lambda: reference(name, **{role: (name,)}),
                                             PERIODS, fmt)
    return reports


# ----------------------------------------------------------------------------
# Two broken control packages: each differs from the cart's in its test only
# ----------------------------------------------------------------------------
@allovi.register
class ForgetsTheTool(cart.control.ConstantCommand):
    """Its test computes every clearance as if the tool were folded."""
    NAME = "broken.forgets-the-tool"

    def profile(self, plan, initial):
        folded = ce.Plan(plan.j0, plan.pairs, plan.p_rest, Fr(0))
        return super().profile(folded, initial)


@allovi.register
class Generous(cart.control.ConstantCommand):
    """Its test accepts braking up to 0.01 m/s^2 beyond the label."""
    NAME = "broken.generous"

    def profile(self, plan, initial):
        relabelled = ce.Plan(plan.j0, [(s, b + Fr(1, 100)) for s, b in plan.pairs],
                             plan.p_rest, plan.ell)
        return super().profile(relabelled, initial)


EAGER = load(os.path.join(HERE, "examples", "eager.py"))[0][1]
MARGIN = load(os.path.join(HERE, "examples", "survey_margin.py"))[0][1]
SURVEY = {"ramp_start": ce.RAMP + Fr(1, 5)}

CASES = {
    "reference": lambda: reference("reference"),
    "survey-0.2": lambda: reference("survey-0.2", capability=("cart.survey", SURVEY)),
    "eager": lambda: reference("eager", proposal=(EAGER,)),
    "forgets-the-tool": lambda: reference("forgets-the-tool",
                                          control=("broken.forgets-the-tool",)),
    "forgets-the-tool-eager": lambda: reference("forgets-the-tool-eager",
                                                control=("broken.forgets-the-tool",),
                                                proposal=(EAGER,)),
    "generous": lambda: reference("generous", control=("broken.generous",)),
    "generous-eager": lambda: reference("generous-eager", control=("broken.generous",),
                                        proposal=(EAGER,)),
    "margin-0.25": lambda: reference("margin-0.25", capability=(MARGIN,)),
    "margin-0.1": lambda: reference("margin-0.1", capability=(MARGIN, {"margin": Fr(1, 10)})),
}


# ----------------------------------------------------------------------------
# Facts quoted in Section 10.3
# ----------------------------------------------------------------------------
def status_of(results):
    """'n passed, n failed, n not applicable'; and the failed checks."""
    failed = "; ".join(f"{r.name}: {r.detail}" for r in results if r.status == checks.FAILED)
    return checks.summary(results), failed or "none"


def run_plain(make):
    cfg = make()
    return cfg, allovi.run(cfg, PERIODS, strict=True, diagnostics=True)


def first_violation(cfg, rec):
    """The first period during which the tool passes the wall, and how long
    after the start of the period: the first period whose final state
    violates p + l <= 25 m, with the instant of the crossing from its exact
    motion (constant deceleration, the tool held, a forward motion)."""
    states = [p.state for p in rec.periods] + [rec.final_state]
    k = next(i for i in range(len(states) - 1) if cfg.world.clearance(states[i + 1]) < 0)
    (p0, v0, l0), (p1, v1, l1) = states[k], states[k + 1]
    b = (v0 - v1) / ce.T
    assert l0 == l1 and v1 > 0 and p1 == p0 + v0 * ce.T - b * ce.T * ce.T / 2
    c = cfg.world.clearance(states[k])
    tau = (c / v0 if b == 0 else
           (float(v0) - math.sqrt(float(v0 * v0 - 2 * b * c))) / float(b))
    return k, tau


def beyond_labels(rec):
    """The fresh candidates with a pair that accelerates towards the start
    beyond its label, and the least of their largest excesses."""
    excess = []
    for p in rec.periods:
        for e in (p.decision.entries if p.decision is not None else ()):
            over = [-s.u - b for s, b in e.plan.pairs if s.u < 0 and -s.u > b]
            if e.kind != "fallback" and over:
                excess.append(max(over))
    return len(excess), (min(excess) if excess else None)


def trace_facts():
    """What the records show for the broken packages."""
    out = {}
    cfg, rec = run_plain(CASES["forgets-the-tool-eager"])
    k, tau = first_violation(cfg, rec)
    out["forgets the tool, eager: first period in which the tool passes the wall"] = \
        f"{k}, {1000 * tau:.1f} ms after its start"
    flagged = next(p.k for p in rec.periods
                   if p.governor == "normal" and cfg.world.clearance(p.state) < p.level)
    out["forgets the tool, eager: periods from the first failed check to the violation"] = \
        k - flagged
    cfg, rec = run_plain(CASES["forgets-the-tool"])
    row = next(r for r in candidate_rows(rec, cfg.control)
               if r["k"] == 202 and r["outcome"] == "committed")
    out["forgets the tool, naive: committed in period 202"] = \
        f"level {fmt(row['level'])} m; reason: {row['level_reason']}"
    ref_cfg, ref = run_plain(CASES["reference"])
    mine, theirs = (decision_rows(rec, cfg.control, cfg.robot),
                    decision_rows(ref, ref_cfg.control, ref_cfg.robot))
    columns = sorted({c for a, b in zip(mine, theirs) for c in a if a[c] != b[c]})
    first = next(a["k"] for a, b in zip(mine, theirs) if a != b)
    out["forgets the tool, naive: decision columns that differ from the reference"] = \
        f"{', '.join(columns)}; from period {first}"
    gen_cfg, gen = run_plain(CASES["generous"])
    same = (decision_rows(gen, gen_cfg.control, gen_cfg.robot) ==
            decision_rows(ref, ref_cfg.control, ref_cfg.robot) and
            candidate_rows(gen, gen_cfg.control) == candidate_rows(ref, ref_cfg.control))
    out["generous: records identical to those of the reference"] = "yes" if same else "no"
    n, least = beyond_labels(gen)
    out["generous, naive: candidates that brake beyond their labels; least excess"] = \
        f"{n}; {fmt(least)} m/s^2"
    out["generous, eager: candidates that brake beyond their labels"] = \
        beyond_labels(run_plain(CASES["generous-eager"])[1])[0]
    runs = {}
    for name in CASES:
        c, r = run_plain(CASES[name])
        ev = c.evaluator.evaluate(r)
        runs[name] = (f"{fmt(ev['arrival_in_work_zone'])} s; {fmt(ev['tool_unfolds'])} s; "
                      f"{fmt(ev['least_clearance'])} m; "
                      + (", ".join(f"{fmt(a * ce.T)} s" for a in r.alarms) or "none"))
    return {"what the records show": out,
            "arrival; tool unfolds; least clearance; alarms": runs}


def template_facts():
    """Each template has every method of its role."""
    registry = allovi.Registry()
    folder = os.path.join(HERE, "templates")
    names, problems = [], []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".py"):
            continue
        spec = importlib.util.spec_from_file_location("template_" + fname[:-3],
                                                      os.path.join(folder, fname))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        classes = [v for v in vars(module).values()
                   if isinstance(v, type) and hasattr(v, "ROLE")]
        for cls in classes:
            names.append(cls.ROLE)
            mine = allovi.conformance(cls)
            if not mine:
                try:
                    registry.register(cls)
                except allovi.RegistrationError as exc:
                    mine = [str(exc)]
            problems.append(mine)
    return {"templates": dict(templates=len(names), roles=len(set(names)),
                              conforming=sum(1 for p in problems if not p),
                              roles_covered="yes" if set(names) == set(allovi.contract.ROLES)
                              else "no")}


def contribution_facts(reports_by_case):
    """The path of a contribution, on both examples, gives the reports of the
    corresponding cases."""
    out = {}
    for path, case in (("eager.py", "eager"), ("survey_margin.py", "margin-0.25")):
        reports = check_file(os.path.join(HERE, "examples", path))
        (role, name), results = next(iter(reports.items()))
        same = [r.line() for r in results] == [r.line() for r in reports_by_case[case]]
        out[f"--module examples/{path}"] = (f"{role} {name}: {checks.summary(results)}; "
                                            f"as the case {case}: {'yes' if same else 'no'}")
    return {"the path of a contribution": out}


def main():
    ap = argparse.ArgumentParser(description="Section 10.3: the checks of modules of the cart")
    ap.add_argument("--check", action="store_true",
                    help="check every number quoted in Section 10.3")
    ap.add_argument("--module", metavar="FILE",
                    help="run the checks on every module that FILE registers")
    args = ap.parse_args()
    if args.module:
        for (role, name), results in check_file(args.module).items():
            if results is None:
                print(f"[{role} {name}] {TASK_NOT_RUN}")
                continue
            print(f"[{role} {name}] {checks.summary(results)}")
            for r in results:
                print("  " + r.line())
        return
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    facts, reports = {}, {}
    for name, make in CASES.items():
        results = checks.check(make, PERIODS, fmt)
        reports[name] = results
        with open(os.path.join(out_dir, f"checks-{name}.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(r.line() for r in results) + "\n")
        summary, failed = status_of(results)
        facts.setdefault("checks", {})[name] = summary
        facts.setdefault("failed checks", {})[name] = failed
    ref = {r.name: r for r in reports["reference"]}
    facts["reference"] = {
        "checks": len(reports["reference"]),
        "labels examined": ref["labels at least as informative as the global capability"].cases,
        "rows compared with a second run": ref["a second run repeats every record"].cases,
        "executed labels examined by the oracle": ref["executed labels valid"].cases,
        "own test: plans tried": ref["own test refuses plans that brake beyond their labels"].cases}
    facts.update(trace_facts())
    facts.update(template_facts())
    facts.update(contribution_facts(reports))

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
        lines.append(f"[check] {n} values quoted in Section 10.3: " +
                     ("all as quoted" if not failures else "FAILED: " + "; ".join(failures)))
    with open(os.path.join(out_dir, "checks-summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if failures:
        raise SystemExit(1)


EXPECTED = {   # every number quoted in Section 10.3, and the outcomes behind Table S10.3
    'checks': {
        'reference': '13 passed, 0 failed, 0 not applicable',
        'survey-0.2': '12 passed, 1 failed, 0 not applicable',
        'eager': '13 passed, 0 failed, 0 not applicable',
        'forgets-the-tool': '12 passed, 1 failed, 0 not applicable',
        'forgets-the-tool-eager': '12 passed, 1 failed, 0 not applicable',
        'generous': '12 passed, 1 failed, 0 not applicable',
        'generous-eager': '12 passed, 1 failed, 0 not applicable',
        'margin-0.25': '13 passed, 0 failed, 0 not applicable',
        'margin-0.1': '12 passed, 1 failed, 0 not applicable',
    },
    'failed checks': {
        'reference': 'none',
        'survey-0.2': 'executed labels valid: 1 failed; first: period 55: label 2',
        'eager': 'none',
        'forgets-the-tool':
            'true reserve at least the level in force: 18 failed; first: period'
            ' 202: level 1.3 in force, true reserve 0.5',
        'forgets-the-tool-eager':
            'true reserve at least the level in force: 18 failed; first: period'
            ' 202: level 1.1 in force, true reserve 0.49125',
        'generous':
            'own test refuses plans that brake beyond their labels: 12 failed; '
            'first: from 20 m at 2 m/s, braking at 0.501 m/s^2 under labels of '
            '0.5 m/s^2 is certified at 1 m; it plans rest at 23.9922 m, and in '
            'a world braking at 0.5 m/s^2 the cart cannot stop before 24 m',
        'generous-eager':
            'own test refuses plans that brake beyond their labels: 12 failed; '
            'first: from 20 m at 2 m/s, braking at 0.501 m/s^2 under labels of '
            '0.5 m/s^2 is certified at 1 m; it plans rest at 23.9922 m, and in '
            'a world braking at 0.5 m/s^2 the cart cannot stop before 24 m',
        'margin-0.25': 'none',
        'margin-0.1': 'executed labels valid: 1 failed; first: period 55: label 2',
    },
    'reference': {
        'checks': '13',
        'labels examined': '124276',
        'rows compared with a second run': '1760',
        'executed labels examined by the oracle': '220',
        'own test: plans tried': '16',
    },
    'what the records show': {
        'forgets the tool, eager: first period in which the tool passes the wall':
            '210, 6.8 ms after its start',
        'forgets the tool, eager: periods from the first failed check to the violation':
            '8',
        'forgets the tool, naive: committed in period 202':
            'level 1.3 m; reason: clearance 0.5 m at the rest point 23.7 m',
        'forgets the tool, naive: decision columns that differ from the reference':
            'level, onset_level; from period 200',
        'generous: records identical to those of the reference': 'yes',
        'generous, naive: candidates that brake beyond their labels; least excess':
            '48; 0.03125 m/s^2',
        'generous, eager: candidates that brake beyond their labels': '0',
    },
    'arrival; tool unfolds; least clearance; alarms': {
        'reference': '8.6 s; 20.1 s; 0.5 m; none',
        'survey-0.2': '8.4 s; 20.1 s; 0.52 m; 5.6 s',
        'eager': '8.6 s; 20.1 s; 0.5 m; none',
        'forgets-the-tool': '8.6 s; 20.1 s; 0.5 m; none',
        'forgets-the-tool-eager': '8.6 s; 20.1 s; -0.29625 m; none',
        'generous': '8.6 s; 20.1 s; 0.5 m; none',
        'generous-eager': '8.6 s; 20.1 s; 0.5 m; none',
        'margin-0.25': '8.6 s; 20.1 s; 0.5 m; none',
        'margin-0.1': '8.6 s; 20.1 s; 0.545 m; 5.6 s',
    },
    'templates': {
        'templates': '8',
        'roles': '8',
        'conforming': '8',
        'roles_covered': 'yes',
    },
    'the path of a contribution': {
        '--module examples/eager.py':
            'proposal example.eager: 13 passed, 0 failed, 0 not applicable; as '
            'the case eager: yes',
        '--module examples/survey_margin.py':
            'capability example.survey-margin: 13 passed, 0 failed, 0 not '
            'applicable; as the case margin-0.25: yes',
    },
}


if __name__ == "__main__":
    main()
