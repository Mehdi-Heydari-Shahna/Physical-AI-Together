#!/usr/bin/env python3
"""
The cart of Chapter 3 in Newton and Genesis: the declared experiment of
Chapter 11 through the simulator adapters cart.newton and cart.genesis, and
the support matrix of Section 12.1.

Physical AI, Together, Chapter 12 (Section 12.1).

Each simulator runs in its own Python environment (requirements-newton.txt,
requirements-genesis.txt), one backend per call:

    python portability.py newton-mujoco-warp     Newton, MuJoCo Warp
    python portability.py newton-mujoco-c        Newton, MuJoCo's C library
    python portability.py newton-mujoco-c-hull64 the same with the importer's hull of 64 vertices
    python portability.py newton-mujoco-c-gap    the same with Newton's gap passed to MuJoCo
    python portability.py newton-xpbd            Newton, XPBD
    python portability.py newton-featherstone    Newton, Featherstone
    python portability.py newton-semi-implicit   Newton, semi-implicit
    python portability.py genesis                Genesis, the options of the MJCF text
    python portability.py genesis-mpr            the same with Genesis's MPR collision detection
    python portability.py genesis-defaults       Genesis, its own defaults
    python portability.py --matrix               the support matrix, from the results present

For a backend, the script writes the record of the experiment and runs, in
the simulator:
  * the port checks: the diagnostics of Section S11.9.1 that catch a mistake
    of conversion, with criteria declared in experiment.CRITERIA;
  * if they pass, the run of Chapter 3 period by period at 1 ms, as Section S11.9.2 does in MuJoCo, compared with the model and with MuJoCo's own run
    (../ch11/results/agreement-chapter3.csv);
  * the 26 cases of Section 10.4 through the check motion_agreement;
  * the configuration of Chapter 3 with the backend's adapter.

Options:
    --quick     the record and the port checks only
    --check     compare every value with the shipped results/<backend>/summary.txt
                (times and versions are not compared)
    --device D  Warp's device for Newton (default cpu)

Outputs, in results/<backend>/: summary.txt (every value), record.txt (the
record of the experiment), agreement.csv (one row per period). --matrix
writes results/support-matrix.txt and results/against-mujoco.csv.
"""
import argparse
import csv
import hashlib
import importlib.metadata as md
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True

BACKENDS = {
    "newton-mujoco-warp": ("cart.newton", {"solver": "mujoco-warp"}),
    "newton-mujoco-c": ("cart.newton", {"solver": "mujoco-c"}),
    "newton-mujoco-c-hull64": ("cart.newton", {"solver": "mujoco-c", "hull_vertices": 64}),
    "newton-mujoco-c-gap": ("cart.newton", {"solver": "mujoco-c", "gap": None}),
    "newton-xpbd": ("cart.newton", {"solver": "xpbd"}),
    "newton-featherstone": ("cart.newton", {"solver": "featherstone"}),
    "newton-semi-implicit": ("cart.newton", {"solver": "semi-implicit"}),
    "genesis": ("cart.genesis", {"settings": "mujoco"}),
    "genesis-mpr": ("cart.genesis", {"settings": "mujoco-mpr"}),
    "genesis-defaults": ("cart.genesis", {"settings": "defaults"}),
}
PACKAGES = {
    "cart.newton": ("newton", "warp-lang", "mujoco", "mujoco-warp", "numpy", "scipy"),
    "cart.genesis": ("genesis-world", "quadrants", "torch", "mujoco", "numpy"),
}
VOLATILE = ("time per step", "wall time")


def version(pkg):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return "not installed"


def cpu():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "not reported"


def read_summary(path):
    facts, group = {}, None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("[") and line.endswith("]"):
                group = line[1:-1]
                facts[group] = {}
            elif line.startswith("  ") and " = " in line and group is not None:
                k, v = line[2:].split(" = ", 1)
                facts[group][k] = v
    return facts


def reference_rows():
    path = os.path.join(HERE, "..", "ch11", "results", "agreement-chapter3.csv")
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def run_backend(name, args):
    import cart_ports as cp
    import experiment as ex
    import allovi
    import model_providers                       # ../ch10, Section 10.4

    adapter, options = BACKENDS[name]
    if adapter == "cart.newton":
        options = dict(options, device=args.device)
    out_dir = os.path.join(HERE, "results", name)
    os.makedirs(out_dir, exist_ok=True)
    facts = {}
    started = time.time()

    def say(text):
        print(f"[{time.time() - started:7.1f} s] {name}: {text}", flush=True)

    say("building")
    try:
        world = allovi.REGISTRY.create("world", adapter, **options)
    except Exception as exc:                       # noqa: BLE001 (a port that fails to settle)
        facts["record"] = {"backend": name, "adapter": adapter,
                           "options": ", ".join(f"{k} = {v}" for k, v in options.items()),
                           "state precision": "float32" if adapter == "cart.newton" else "-"}
        for pkg in PACKAGES[adapter]:
            facts["record"][f"version of {pkg}"] = version(pkg)
        facts["port checks"] = {
            "building and settling at rest failed": f"{type(exc).__name__}: {exc}",
            "verdicts: loads, timing, friction, crest": "fail, fail, fail, fail"}
        facts["the run of Chapter 3, period by period"] = {
            "not run": "the cart fails the port checks of loads or timing"}
        write_outputs(name, out_dir, facts, args)
        say("done: the port failed to build or settle")
        return
    sim = world.sim
    xml = cp.mjcf(cp.Design())
    rec = {"backend": name, "adapter": adapter,
           "options": ", ".join(f"{k} = {v}" for k, v in options.items()) or "defaults"}
    for pkg in PACKAGES[adapter]:
        rec[f"version of {pkg}"] = version(pkg)
    rec["Python"] = platform.python_version()
    rec["MJCF text of Section 11.1, SHA-256"] = hashlib.sha256(xml.encode()).hexdigest()[:16]
    rec["time step [s]"] = f"{sim.dt:g}"
    rec["steps per period"] = sim.steps
    rec.update(sim.record())
    facts["record"] = rec
    say("port checks")
    checks, verdict = ex.port_checks(sim)
    facts["port checks"] = checks
    passed = verdict["loads"] != "fail" and verdict["timing"] != "fail"
    if args.quick or not passed:
        why = "--quick" if args.quick else "the cart fails the port checks of loads or timing"
        facts["the run of Chapter 3, period by period"] = {"not run": why}
    else:
        say("the run of Chapter 3, period by period")
        steps0, t0 = sim.clock, time.time()
        rows = ex.agreement(world, progress=lambda k: say(f"  period {k}"))
        rec["time per step, in the comparison [ms]"] = (
            f"{1000 * (time.time() - t0) / max(1, sim.clock - steps0):.3g}")
        with open(os.path.join(out_dir, "agreement.csv"), "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.writer(fh)
            cols = list(rows[0])
            w.writerow(cols)
            for r in rows:
                w.writerow([r[c] if c == "k" else f"{r[c]:.9g}" for c in cols])
        facts["the run of Chapter 3, period by period"] = ex.agreement_facts(
            rows, reference_rows())
        say("the cases of Section 10.4")
        facts["the cases of Section 10.4"] = ex.motion_cases_facts(
            world, model_providers.motion_cases())
        say("the configuration of Chapter 3")
        facts["the configuration of Chapter 3"] = ex.closed_loop_facts(
            f"chapter3, {name}", allovi.REGISTRY.create("world", adapter, **options))
    rec["wall time of the run [s]"] = f"{time.time() - started:.0f}"
    write_outputs(name, out_dir, facts, args)
    say("done")


def write_outputs(name, out_dir, facts, args):
    path = os.path.join(out_dir, "summary.txt")
    failures = []
    if args.check:
        if not os.path.exists(path):
            failures.append("no shipped summary to compare with")
        else:
            shipped = read_summary(path)
            for group, items in shipped.items():
                for key, want in items.items():
                    if any(v in key for v in VOLATILE) or key.startswith("version of"):
                        continue
                    got = facts.get(group, {}).get(key)
                    if got is None or str(got) != want:
                        failures.append(f"{group}.{key}: shipped {want!r}, now {got!r}")
    lines = []
    for group, items in facts.items():
        lines.append(f"[{group}]")
        for key, val in items.items():
            lines.append(f"  {key} = {val}")
    if args.check:
        lines.append(f"[check] against the shipped summary: " +
                     ("all as shipped" if not failures else "DIFFERENT: " + "; ".join(failures)))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    write_record(name, out_dir, facts)
    print("\n".join(lines))
    if failures:
        raise SystemExit(1)


def write_record(name, out_dir, facts):
    rec = facts["record"]
    loop = facts.get("the configuration of Chapter 3", {})
    versions = ", ".join(f"{k[len('version of '):]} {v}" for k, v in rec.items()
                         if k.startswith("version of"))
    lines = [
        f"Record of the experiment of Section 12.1: {name}",
        "",
        "model and asset provenance",
        "  the MJCF text of Section 11.1 (../ch11/results/cart.xml), SHA-256 "
        f"{rec.get('MJCF text of Section 11.1, SHA-256', '-')}..., read by the simulator's own "
        "importer",
        "simulator and solver",
        f"  {versions}; Python {rec.get('Python', platform.python_version())}; adapter "
        f"{rec['adapter']}, {rec['options']}",
        "time steps",
        f"  {rec.get('time step [s]', '0.001')} s; governor period 0.1 s = "
        f"{rec.get('steps per period', 100)} steps, counted; drive loop at every step",
        "actuators and sensors",
        "  the drive loop of Section 11.1; wheel torques and the tool's servo target; the "
        "wheels' loads from the simulator's contact forces",
        "frames and units",
        "  as in Section 11.1: SI units, x towards the wall, z up; state (p, v, l)",
        "uncertainty bounds, constraints, terminal behaviour, training split, seeds",
        "  as in Section 11.1",
        "hardware",
        f"  {platform.system()} {platform.machine()}, {cpu()}",
        "guarantee status",
        (f"  first alarm at {loop.get('first alarm [s]')} s; restarts {loop.get('restarts')}; "
         f"least clearance {loop.get('least clearance [m]')} m" if loop else
         "  the configuration was not run"),
    ]
    with open(os.path.join(out_dir, "record.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


ROLES = {   # what each configuration is, for the support matrix
    "newton-mujoco-warp": "Newton, SolverMuJoCo on MuJoCo Warp, as the port sets it",
    "newton-mujoco-c": "Newton, SolverMuJoCo on MuJoCo's C library, as the port sets it",
    "newton-mujoco-c-hull64": "the same with the importer's hull of 64 vertices (a mistake)",
    "newton-mujoco-c-gap": "the same with Newton's gap of 0.1 m passed to MuJoCo (a mistake)",
    "newton-xpbd": "Newton, SolverXPBD with Newton's contacts and defaults",
    "newton-featherstone": "Newton, SolverFeatherstone with Newton's contacts and defaults",
    "newton-semi-implicit": "Newton, SolverSemiImplicit with Newton's contacts and defaults",
    "genesis": "Genesis, the options of the MJCF text, as the port sets it",
    "genesis-mpr": "the same with Genesis's MPR collision detection (its default)",
    "genesis-defaults": "Genesis, its own defaults",
}


def reference_verdicts(ref):
    """The port checks for MuJoCo itself: the diagnostics of Section 11.1
    (../ch11/results/mujoco-summary.txt), judged by the criteria of
    experiment.CRITERIA. Over the crest, Section 11.1 counts the steps in
    which a wheel has no contact."""
    facts = {k: v for items in ref.values() for k, v in items.items()}
    total = float(facts["wheel loads at rest [N]"].split(" = ")[-1])
    weight = float(facts["weight [N]"])
    dp = float(facts["one period at 1 m/s from 5 m: dp [m]"])
    cap = float(facts["capacity: braking on firm ground from 4 m/s [m/s^2]"])
    lost = int(facts["crest, braking at 0.5 from 2.12 m/s at 20 m: least penetration [mm], "
                     "steps without contact"].split(", ")[1])
    checks = (abs(total - weight) <= 1e-3 * weight, abs(dp) <= 1e-4,
              abs(cap - 2.4525) <= 0.01 * 2.4525, lost == 0)
    return ", ".join("pass" if ok else "fail" for ok in checks)


def matrix(args):
    """The support matrix of Section 12.1, from the summaries present, and
    the end differences of each configuration's run of Chapter 3 against
    MuJoCo's own run (results/against-mujoco.csv)."""
    rows = []
    ref = read_summary(os.path.join(HERE, "..", "ch11", "results", "mujoco-summary.txt"))
    run = ref.get("the run of Chapter 3, period by period", {})
    rows.append({
        "configuration": "mujoco",
        "what": "MuJoCo itself, the reference of Section 11.1",
        "versions": f"mujoco {ref['record'].get('MuJoCo', '?')}",
        "precision": "float64",
        "port checks: loads, timing, friction, crest": reference_verdicts(ref),
        "end of period, median and largest [m]":
            run["end of period: median, 90th percentile (nearest rank) [m]"].split(", ")[0] +
            ", " + run["end of period: largest [m], period, t [s]"].split(", ")[0],
        "axle disturbance, largest [m/s^2]; periods above 0.05":
            run["axle: acceleration less command, tool folded, over 1 ms: largest, period, at p"]
            .split(", ")[0] + "; " +
            run["axle: acceleration less command, tool folded: periods above 0.05 m/s^2"],
        "against MuJoCo: per period, the error vector less MuJoCo's, norm, median and largest [m]":
            "-",
        "cases of Section 10.4 beyond 1 mm":
            ref["the cases of Section 10.4"]["differences with tolerance 0.001"],
        "configuration of Chapter 3: alarms, restarts, least clearance [m]": ", ".join(
            ref["the configuration of Chapter 3"][k]
            for k in ("alarms", "restarts", "least clearance [m]")),
        "time per step on this machine, indicative [ms]": "-",
    })
    series = {}
    for name in BACKENDS:
        path = os.path.join(HERE, "results", name, "summary.txt")
        if not os.path.exists(path):
            continue
        f = read_summary(path)
        r = f.get("record", {})
        run = f.get("the run of Chapter 3, period by period", {})
        versions = ", ".join(f"{k[len('version of '):]} {v}" for k, v in r.items()
                             if k.startswith("version of") and v != "not installed"
                             and k[len('version of '):] not in ("numpy", "scipy", "torch"))
        ran = "not run" not in run

        def get(key, i=0, run=run):
            return run.get(key, "?").split(", ")[i] if ran else "not run"
        loop = f.get("the configuration of Chapter 3")
        against = run.get("against MuJoCo 3.16.0 (Section 11.1): end differences less MuJoCo's, "
                          "median, largest [m], period", "?, ?").split(", ")
        rows.append({
            "configuration": name,
            "what": ROLES.get(name, ""),
            "versions": versions,
            "precision": r.get("state precision", r.get("precision", "?")),
            "port checks: loads, timing, friction, crest":
                f.get("port checks", {}).get("verdicts: loads, timing, friction, crest", "?"),
            "end of period, median and largest [m]":
                (get("end of period: median, 90th percentile (nearest rank) [m]") + ", " +
                 get("end of period: largest [m], period")) if ran else "not run",
            "axle disturbance, largest [m/s^2]; periods above 0.05":
                (get("axle: acceleration less command, tool folded, over one step: largest, "
                     "period") + "; " +
                 run.get("axle: periods with the tool folded above 0.05 m/s^2", "?"))
                if ran else "not run",
            "against MuJoCo: per period, the error vector less MuJoCo's, norm, median and "
            "largest [m]": f"{against[0]}, {against[1]}" if ran else "not run",
            "cases of Section 10.4 beyond 1 mm": f.get("the cases of Section 10.4", {}).get(
                "differences with tolerance 0.001", "not run"),
            "configuration of Chapter 3: alarms, restarts, least clearance [m]":
                ", ".join(loop.get(k, "?") for k in ("alarms", "restarts", "least clearance [m]"))
                if loop else "not run",
            "time per step on this machine, indicative [ms]":
                r.get("time per step, in the comparison [ms]", "-"),
        })
        if ran:
            series[name] = against_mujoco(name)
    lines = ["Support matrix of Section 12.1: the declared experiment of Chapter 11 in each "
             "configuration", ""]
    for r in rows:
        lines.append(r["configuration"])
        for c, v in r.items():
            if c != "configuration":
                lines.append(f"  {c}: {v}")
    with open(os.path.join(HERE, "results", "support-matrix.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    if series:
        names = list(series)
        ks = sorted(set.intersection(*(set(series[n]) for n in names)))
        with open(os.path.join(HERE, "results", "against-mujoco.csv"), "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["k", "t"] + [n.replace("-", "_") for n in names])
            for k in ks:
                w.writerow([k, f"{0.1 * k:.1f}"] + [f"{series[n][k]:.6g}" for n in names])
    print("\n".join(lines))


def against_mujoco(name):
    """{period: the norm of the state of the difference between two vectors,
    this configuration's error over the period (dp, dv, dl) and MuJoCo's}.
    Each error is the end state of the period less the nominal motion from the
    state at which the period began, and every period starts afresh, so this
    compares errors over one period, not whole runs."""
    ref = {int(r["k"]): r for r in reference_rows()}
    out = {}
    path = os.path.join(HERE, "results", name, "agreement.csv")
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            q = ref[int(r["k"])]
            dp, dv, dl = (float(r[c]) - float(q[c]) for c in ("dp", "dv", "dl"))
            out[int(r["k"])] = max(abs(dp) + abs(dl), abs(dv))
    return out


def main():
    ap = argparse.ArgumentParser(description="Section 12.1: the cart in Newton and Genesis")
    ap.add_argument("backend", nargs="?", choices=sorted(BACKENDS))
    ap.add_argument("--matrix", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.matrix:
        matrix(args)
    elif args.backend:
        run_backend(args.backend, args)
    else:
        ap.error("name a backend, or --matrix")


if __name__ == "__main__":
    main()
