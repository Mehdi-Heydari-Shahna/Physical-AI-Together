#!/usr/bin/env python3
"""
Engine, solver and renderer: what each comparison between the
configurations of Section 12.1 changes, and what it measures.

Physical AI, Together, Chapter 12 (Section 12.2).

The per-period comparison of Section S11.9.2 in controlled variants, each in
the environment of its simulator, and the comparisons between them:

    python layers.py mujoco            MuJoCo itself, the release installed, through the
                                       script of Section 11.1: 3.16.0 with
                                       ../ch11/requirements.txt, 3.15.0 in the Newton
                                       environment (requirements-newton.txt)
    python layers.py mujoco --render   the same, and again with an image rendered by
                                       MuJoCo's OpenGL renderer after every step
    python layers.py mujoco-gap        MuJoCo itself, the contacts with and without a gap
                                       of 0.1 m, Newton's default, at the same states
    python layers.py mujoco-newton-changes
                                       MuJoCo 3.15.0 itself (the Newton environment) with
                                       the changes that Newton makes around it, one at a
                                       time and together: multiple contacts off, and the
                                       parameters, the state before every step and its
                                       readout in single precision
    python layers.py genesis-f32       Genesis, the options of the text, single precision
    python layers.py genesis-render    Genesis, the options of the text, without and with
                                       an image rendered by its rasterizer after every
                                       step, in 13 periods that cover every phase of the
                                       run
    python layers.py --compare         the comparisons of Section 12.2, from the results
                                       present (any Python 3.10 or later)

Options: --check compares every value with the shipped summary (times and
the renderer's mean pixel value are not compared).

Each period of the comparison starts afresh: the cart is placed 0.05 s before
the period on the motion of the previous period and driven under the previous
law until the period starts, and the period's error is its end state less the
nominal motion of the model from the state at which the period began, the
vector (dp, dv, dl). A comparison between two configurations takes, period by
period, the difference of their two vectors and measures it in the norm of
the state, max(|dp| + |dl|, |dv| * 1 s). It compares errors over one period,
not whole runs.

The renderers draw off screen through OSMesa: MUJOCO_GL and PYOPENGL_PLATFORM
are set to osmesa unless given, and Mesa's libOSMesa must be installed (the
package libosmesa6 on Debian and Ubuntu).

Outputs, in results/layers/: <variant>/agreement.csv (one row per period, to
17 significant digits), <variant>/summary.txt, and comparisons.txt.
"""
import argparse
import csv
import importlib.metadata as md
import os
import platform
import statistics
import sys
import time

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

HERE = os.path.dirname(os.path.abspath(__file__))
CH11 = os.path.join(HERE, "..", "ch11")
OUT = os.path.join(HERE, "results", "layers")
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True

# 13 periods that cover every phase of the run of Chapter 3: accelerating,
# braking on the firm ground, over the crest, on the ramp, at rest, backing
# off, and the tool unfolding
RENDER_PERIODS = (1, 36, 40, 57, 59, 64, 65, 66, 68, 100, 150, 198, 201)
VOLATILE = ("time", "mean pixel value")
COLUMNS = ("k", "t", "p", "v", "l", "u", "ell", "start", "dp", "dv", "dl", "end")

# The comparisons of Section 12.2: (configuration, reference, what changes)
PAIRS = (
    ("mujoco-3.15.0", "mujoco-3.16.0", "the solver's release: MuJoCo 3.15.0 for 3.16.0"),
    ("newton-mujoco-c", "mujoco-3.15.0",
     "the engine: Newton around the same MuJoCo, with four changes (Table 12.4): multiple "
     "contact points off, and its parameters, its state before every step and the state "
     "that the port reads in single precision"),
    ("newton-mujoco-warp", "newton-mujoco-c",
     "the solver's implementation: MuJoCo Warp, in single precision, for MuJoCo's "
     "C library"),
    ("genesis", "mujoco-3.16.0",
     "engine and solver: Genesis's reimplementation of MuJoCo's, in double precision"),
    ("genesis-f32", "genesis", "the arithmetic: single for double precision"),
    ("newton-xpbd", "newton-mujoco-c", "the solver: XPBD, with Newton's contacts"),
    ("genesis-defaults", "genesis", "the solver's settings: Genesis's own defaults"),
    ("newton-mujoco-c-hull64", "newton-mujoco-c",
     "a mistake: the importer's hull, which changes the ground"),
    ("newton-mujoco-c-gap", "newton-mujoco-c",
     "a mistake: Newton's gap, which changes the computation of the contacts"),
    ("genesis-mpr", "genesis",
     "a mistake: Genesis's MPR, which changes the computation of the contacts and fails on "
     "the crest"),
)


def g3(x):
    return f"{x:.3g}"


def norm(dp, dv, dl):
    """The norm of the state, max(|p| + |l|, |v| * 1 s)."""
    return max(abs(dp) + abs(dl), abs(dv))


def version(pkg):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return "not installed"


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow([r["k"]] + [f"{float(r[c]):.17g}" for c in COLUMNS[1:]])


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return {int(r["k"]): r for r in csv.DictReader(fh)}


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


def write_summary(name, facts, check):
    """Writes results/layers/<name>/summary.txt; with check, compares every
    value with the shipped summary first and fails on any difference."""
    path = os.path.join(OUT, name, "summary.txt")
    failures = []
    if check:
        if not os.path.exists(path):
            failures.append("no shipped summary to compare with")
        else:
            for group, items in read_summary(path).items():
                for key, want in items.items():
                    if any(v in key for v in VOLATILE) or key.startswith("version of"):
                        continue
                    got = facts.get(group, {}).get(key)
                    if got is None or str(got) != want:
                        failures.append(f"{group}.{key}: shipped {want!r}, now {got!r}")
    lines = []
    for group, items in facts.items():
        lines.append(f"[{group}]")
        lines += [f"  {k} = {v}" for k, v in items.items()]
    if check:
        lines.append("[check] against the shipped summary: " +
                     ("all as shipped" if not failures else "DIFFERENT: " + "; ".join(failures)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if failures:
        raise SystemExit(1)


def compare_rows(a, b):
    """(identical in every value, largest difference of the error vectors in
    the norm of the state) of two lists of rows of the same periods."""
    same = len(a) == len(b) and all(x[c] == y[c] for x, y in zip(a, b) for c in x)
    largest = max(norm(x["dp"] - y["dp"], x["dv"] - y["dv"], x["dl"] - y["dl"])
                  for x, y in zip(a, b))
    return same, largest


def run_facts(rows):
    ends = [r["end"] for r in rows]
    worst = max(rows, key=lambda r: r["end"])
    return {"periods": len(rows),
            "end of period: median, largest [m], period":
                f"{g3(statistics.median(ends))}, {g3(worst['end'])}, {worst['k']}"}


# ----------------------------------------------------------------------------
# MuJoCo itself, through the script of Section 11.1
# ----------------------------------------------------------------------------
def mujoco_variant(args):
    import mujoco
    sys.path.insert(0, CH11)
    import mujoco_cart as mc

    t0 = time.time()
    rows = mc.agreement(mc.DESIGN)
    name = f"mujoco-{mujoco.__version__}"
    write_csv(os.path.join(OUT, name, "agreement.csv"), rows)
    facts = {"record": {
        "variant": name,
        "what": "MuJoCo itself through the per-period comparison of Section 11.1, at 1 ms",
        "version of mujoco": mujoco.__version__,
        "version of numpy": version("numpy"),
        "Python": platform.python_version(),
        "time of the comparison [s]": f"{time.time() - t0:.0f}",
    }, "the run of Chapter 3, period by period": run_facts(rows)}
    if args.render:
        state = {"renderer": None, "camera": None, "frames": 0, "sum": 0.0}
        step2 = mujoco.mj_step2

        def rendered_step2(m, d):
            step2(m, d)
            if state["renderer"] is None:
                state["renderer"] = mujoco.Renderer(m, 64, 96)
                cam = mujoco.MjvCamera()
                cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                cam.trackbodyid = m.body("chassis").id
                cam.distance, cam.azimuth, cam.elevation = 2.5, 90.0, -15.0
                state["camera"] = cam
            state["renderer"].update_scene(d, camera=state["camera"])
            state["sum"] += float(state["renderer"].render().mean())
            state["frames"] += 1

        mujoco.mj_step2 = rendered_step2
        try:
            t1 = time.time()
            rendered = mc.agreement(mc.DESIGN)
        finally:
            mujoco.mj_step2 = step2
        same, largest = compare_rows(rows, rendered)
        facts["the renderer"] = {
            "renderer": f"MuJoCo's OpenGL renderer through {os.environ['MUJOCO_GL']}, "
                        "96 x 64 pixels, a camera tracking the chassis",
            "frames, one after every step": state["frames"],
            "mean pixel value of the frames": f"{state['sum'] / max(1, state['frames']):.3g}",
            "every value of every period identical with and without rendering":
                "yes" if same else "no",
            "largest difference of the error vectors, norm of the state [m]": g3(largest),
            "time of the rendered comparison [s]": f"{time.time() - t1:.0f}",
        }
    write_summary(name, facts, args.check)


def mujoco_gap(args):
    """What a gap of 0.1 m, Newton's default, does in MuJoCo itself: at every
    step of two motions run without a gap, the contacts are computed again at
    the same state, with and without the gap, and the contacts that act are
    compared; and the motion on the firm ground is run with and without it."""
    import mujoco
    import numpy
    sys.path.insert(0, CH11)
    import mujoco_cart as mc

    sim = mc.cm.Sim(mc.DESIGN)
    m, d = sim.m, sim.data
    probe = mujoco.MjData(m)

    def contacts(gap):
        m.geom_gap[:] = gap
        mujoco.mj_forward(m, probe)
        act, idle = {}, 0
        for c in probe.contact[:probe.ncon]:
            if c.efc_address < 0:
                idle += 1
            else:
                act[(c.geom1, c.geom2)] = (c.pos.copy(), c.frame[:3].copy(), float(c.dist))
        return act, idle

    facts = {"record": {
        "variant": f"mujoco-{mujoco.__version__}-gap",
        "what": "MuJoCo itself, the contacts with and without a gap of 0.1 m on every geom",
        "version of mujoco": mujoco.__version__,
        "version of numpy": version("numpy"),
        "Python": platform.python_version(),
        "options: ccd_tolerance, ccd_iterations": f"{m.opt.ccd_tolerance:g}, "
                                                    f"{m.opt.ccd_iterations}",
    }}
    for key, (p, v, a, duration) in (
            ("the firm ground, 1 m/s from 5 m, 0.3 s", (5.0, 1.0, 0.0, 0.3)),
            ("the crest, braking at 0.5 m/s^2 from 2.12 m/s at 20 m, 1.4 s",
             (20.0, 2.12, -0.5, 1.4))):
        worst = {"position": 0.0, "normal": 0.0, "distance": 0.0}
        steps, unmatched, idle_most, rows_idle = 0, 0, 0, 0

        def sample(sim_, i, v_ref):
            nonlocal steps, unmatched, idle_most, rows_idle
            probe.qpos[:], probe.qvel[:] = d.qpos, d.qvel
            base, _ = contacts(0.0)
            nef = probe.nefc
            gap, idle = contacts(0.1)
            m.geom_gap[:] = 0.0                     # the run itself has no gap
            steps += 1
            idle_most = max(idle_most, idle)
            rows_idle = max(rows_idle, probe.nefc - nef)
            if set(base) != set(gap):
                unmatched += 1
            for pair in set(base) & set(gap):
                (pa, na, da), (pb, nb, db) = base[pair], gap[pair]
                worst["position"] = max(worst["position"], float(numpy.abs(pa - pb).max()))
                worst["normal"] = max(worst["normal"], float(numpy.abs(na - nb).max()))
                worst["distance"] = max(worst["distance"], abs(da - db))

        m.geom_gap[:] = 0.0
        mc.drive(sim, p, v, a, duration, sample)
        facts[key] = {
            "steps": steps,
            "steps in which the gap changes which contacts act": unmatched,
            "inactive contacts with the gap, most in a step": idle_most,
            "constraint rows that the gap adds, most in a step": rows_idle,
            "largest change of a contact that acts: position [m], normal, distance [m]":
                f"{g3(worst['position'])}, {g3(worst['normal'])}, {g3(worst['distance'])}",
        }
    states = []
    for gap in (0.0, 0.1):
        m.geom_gap[:] = gap
        sim.place(5.0, 1.0, 0.0)
        for _ in range(3):
            state = sim.period(0.0, 0.0)
        states.append(state)
    m.geom_gap[:] = 0.0
    dp, dv, dl = (b - a for a, b in zip(*states))
    facts["the motion"] = {
        "three periods at 1 m/s from 5 m, with the gap less without: dp [m], dv [m/s], dl [m]":
            f"{g3(dp)}, {g3(dv)}, {g3(dl)}",
    }
    write_summary(f"mujoco-{mujoco.__version__}-gap", facts, args.check)


# ----------------------------------------------------------------------------
# MuJoCo with the changes that Newton makes around it, one at a time and
# together (Section S12.9.2)
# ----------------------------------------------------------------------------
# Newton's SolverMuJoCo on MuJoCo's C library differs from MuJoCo by more than
# its arithmetic. It disables MuJoCo's multiple contact points between convex
# geoms (mjDSBL_MULTICCD), rebuilds MuJoCo's model from the parameters that
# Newton holds in single precision (gravity and the solver's tolerance among
# them), writes Newton's single-precision state into MuJoCo before every step,
# and the port reads the positions and velocities of the bodies from Newton's
# state, in single precision. Each change is imposed here on MuJoCo itself, in
# the release that Newton 1.7.0 requires, through the script of Section 11.1.
NEWTON_CHANGES = (
    ("multiple contacts off", "mujoco-3.15.0-multiccd-off", dict(multiccd=False)),
    ("parameters in single precision", "mujoco-3.15.0-f32-parameters", dict(parameters=True)),
    ("state in single precision before every step", "mujoco-3.15.0-f32-state",
     dict(state=True)),
    ("readout in single precision", "mujoco-3.15.0-f32-readout", dict(readout=True)),
    ("all four", "mujoco-3.15.0-newton-changes",
     dict(multiccd=False, parameters=True, state=True, readout=True)),
)


def single_precision_text(xml):
    """The MJCF text with every decimal number in an attribute replaced by the
    nearest number in single precision, written in full so that the compiler
    reads exactly that number; integers are kept."""
    import re
    import numpy

    def number(m):
        s = m.group(0)
        if "." not in s and "e" not in s.lower():
            return s
        return repr(float(numpy.float32(float(s))))

    def attribute(m):
        return m.group(1) + re.sub(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", number, m.group(2)) + m.group(3)

    return re.sub(r'(=")([^"]*)(")', attribute, xml)


def newton_like(mc, multiccd=True, parameters=False, state=False, readout=False):
    """The class Sim of Section 11.1 with the given changes that Newton makes
    around MuJoCo."""
    import mujoco
    import numpy

    def f32(x):
        return float(numpy.float32(x))

    class NewtonLike(mc.cm.Sim):
        _changed = False

        def static_sink(self):
            # called at the end of Sim.__init__, before anything is measured
            if not self._changed:
                self._changed = True
                if parameters:
                    self.xml = single_precision_text(self.xml)
                    self.m = mujoco.MjModel.from_xml_string(self.xml)
                    self.data = mujoco.MjData(self.m)
                    o = self.m.opt
                    o.gravity[:] = [f32(g) for g in o.gravity]
                    o.tolerance = f32(o.tolerance)
                if not multiccd:
                    self.m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
            return super().static_sink()

        def _round_state(self):
            d = self.data
            d.qpos[:] = d.qpos.astype(numpy.float32)
            d.qvel[:] = d.qvel.astype(numpy.float32)

        def place(self, p, v, l):
            super().place(p, v, l)
            if state:
                self._round_state()
                mujoco.mj_forward(self.m, self.data)

        def foot(self, wheel="front"):
            if not readout:
                return super().foot(wheel)
            x, _, z = self.data.xpos[self.body[wheel]]
            return self.pr.foot(f32(x), f32(z))

        def _front(self):
            f, vx, vz = super()._front()
            return (f, f32(vx), f32(vz)) if readout else (f, vx, vz)

        def period(self, u, ell, kind="segment", record=None, steps=None):
            """Sim.period, with the state rounded to single precision before
            every step when 'state' is set."""
            m, d = self.m, self.data
            v0 = self.state()[1]
            if abs(v0) < mc.cm.sim.V_REST:
                v0 = 0.0
            if kind == "rest":
                brake = 1 if v0 > 0 else 0
            else:
                brake = 1 if v0 > 0 > u else -1 if v0 < 0 < u else 0
            for i in range(steps or self.steps):
                tau = i * m.opt.timestep
                v_ref, a_ref = v0 + u * tau, u
                if (kind == "rest" and v0 <= 0) or (brake and v_ref * v0 <= 0):
                    v_ref, a_ref = 0.0, 0.0
                if state:
                    self._round_state()
                mujoco.mj_step1(m, d)
                if record is not None:
                    record.append(self.sample(a_ref, v_ref))
                self.controls(a_ref, v_ref, ell)
                mujoco.mj_step2(m, d)
            mujoco.mj_forward(m, d)
            return self.state()

    return NewtonLike


def mujoco_newton_changes(args):
    import mujoco
    sys.path.insert(0, CH11)
    import mujoco_cart as mc

    if mujoco.__version__ != "3.15.0":
        raise SystemExit("run this variant in the Newton environment, with MuJoCo 3.15.0")
    reference, _ = source("mujoco-3.15.0")
    newton, _ = source("newton-mujoco-c")
    facts = {"record": {
        "variant": "mujoco-3.15.0-newton-changes",
        "what": "MuJoCo itself with the changes that Newton makes around it, one at a time "
                "and together, through the per-period comparison of Section 11.1, at 1 ms",
        "version of mujoco": mujoco.__version__,
        "version of numpy": version("numpy"),
        "Python": platform.python_version(),
    }}
    world_sim = mc.cm.world.Sim
    t0 = time.time()
    for what, name, changes in NEWTON_CHANGES:
        mc.cm.world.Sim = newton_like(mc, **changes)
        try:
            rows = mc.agreement(mc.DESIGN)
        finally:
            mc.cm.world.Sim = world_sim
        write_csv(os.path.join(OUT, name, "agreement.csv"), rows)
        mine = {r["k"]: r for r in rows}
        group = {"variant": name}
        for label, other in (("MuJoCo 3.15.0", reference), ("Newton on MuJoCo's C library",
                                                            newton)):
            if other is None:
                group[f"against {label}"] = "results missing"
                continue
            d = {k: norm(*(float(mine[k][c]) - float(other[k][c]) for c in ("dp", "dv", "dl")))
                 for k in sorted(set(mine) & set(other))}
            worst = max(d, key=d.get)
            group[f"against {label}: median, largest [m], period"] = (
                f"{g3(statistics.median(d.values()))}, {g3(d[worst])}, {worst}")
            group[f"against {label}: periods with a difference above 1 um"] = \
                sum(v > 1e-6 for v in d.values())
            group[f"against {label}: identical in every period"] = \
                "yes" if max(d.values()) == 0 else "no"
        facts[what] = group
    facts["record"]["time of the five comparisons [s]"] = f"{time.time() - t0:.0f}"
    write_summary("mujoco-3.15.0-newton-changes", facts, args.check)


# ----------------------------------------------------------------------------
# Genesis, in single precision, and with its rasterizer
# ----------------------------------------------------------------------------
def genesis_versions():
    return {f"version of {p}": version(p) for p in ("genesis-world", "quadrants", "torch",
                                                    "numpy")}


def genesis_f32(args):
    import cart_ports
    import experiment
    t0 = time.time()
    world = cart_ports.CartGenesis(settings="mujoco-f32")
    rows = experiment.agreement(world)
    write_csv(os.path.join(OUT, "genesis-f32", "agreement.csv"), rows)
    rec = world.sim.record()
    facts = {"record": {
        "variant": "genesis-f32",
        "what": "Genesis, the options of the MJCF text, single precision",
        **genesis_versions(),
        "Python": platform.python_version(),
        "settings": rec["settings"],
        "state precision": rec["state precision"],
        "rigid options: integrator, constraint solver, cone, iterations, tolerance":
            rec["rigid options: integrator, constraint solver, cone, iterations, tolerance"],
        "time of the comparison [s]": f"{time.time() - t0:.0f}",
    }, "the run of Chapter 3, period by period": run_facts(rows)}
    write_summary("genesis-f32", facts, args.check)


def genesis_render(args):
    import cart_ports
    import experiment
    t0 = time.time()
    plain = experiment.agreement(cart_ports.CartGenesis(settings="mujoco"),
                                 only=RENDER_PERIODS)
    world = cart_ports.CartGenesis(settings="mujoco", camera=(96, 64))
    sim = world.sim
    step = sim._step
    pixels = []

    def rendered_step():
        step()
        pixels.append(float(sim.render().mean()))

    sim._step = rendered_step
    t1 = time.time()
    rendered = experiment.agreement(world, only=RENDER_PERIODS)
    write_csv(os.path.join(OUT, "genesis-render", "agreement.csv"), rendered)
    same, largest = compare_rows(plain, rendered)
    facts = {"record": {
        "variant": "genesis-render",
        "what": "Genesis, the options of the MJCF text, without and with rendering",
        **genesis_versions(),
        "Python": platform.python_version(),
        "periods": ", ".join(str(k) for k in RENDER_PERIODS),
        "time of the comparison without rendering [s]": f"{t1 - t0:.0f}",
        "time of the comparison with rendering [s]": f"{time.time() - t1:.0f}",
    }, "the renderer": {
        "renderer": f"Genesis's rasterizer through {os.environ['PYOPENGL_PLATFORM']}, "
                    "96 x 64 pixels, a camera beside the chassis",
        "frames, one after every step of the periods": len(pixels),
        "mean pixel value of the frames": g3(statistics.fmean(pixels)),
        "every value of every period identical with and without rendering":
            "yes" if same else "no",
        "largest difference of the error vectors, norm of the state [m]": g3(largest),
    }}
    write_summary("genesis-render", facts, args.check)


# ----------------------------------------------------------------------------
# The comparisons
# ----------------------------------------------------------------------------
def source(name):
    """The per-period errors of a configuration or a variant."""
    for path in (os.path.join(OUT, name, "agreement.csv"),
                 os.path.join(HERE, "results", name, "agreement.csv")):
        if os.path.exists(path):
            return read_rows(path), os.path.relpath(path, HERE)
    if name == "mujoco-3.16.0":
        path = os.path.join(CH11, "results", "agreement-chapter3.csv")
        return read_rows(path), os.path.relpath(path, HERE)
    return None, None


def compare(args):
    lines = ["Comparisons of Section 12.2: per period, the difference of two error vectors "
             "(dp, dv, dl), each the end state of the period less the nominal motion from "
             "the state at which the period began, in the norm of the state; each period "
             "starts afresh, so these are errors over one period, not whole runs", ""]
    for a, b, what in PAIRS:
        ra, pa = source(a)
        rb, pb = source(b)
        lines.append(f"[{a} less {b}]")
        lines.append(f"  changes = {what}")
        if ra is None or rb is None:
            lines.append("  not compared = results missing")
            continue
        ks = sorted(set(ra) & set(rb))
        d = {k: norm(*(float(ra[k][c]) - float(rb[k][c]) for c in ("dp", "dv", "dl")))
             for k in ks}
        worst = max(ks, key=lambda k: d[k])
        lines.append(f"  from = {pa}; {pb}")
        lines.append(f"  periods compared = {len(ks)}")
        lines.append(f"  median, largest [m], period = {g3(statistics.median(d.values()))}, "
                     f"{g3(d[worst])}, {worst}")
        lines.append(f"  periods with a difference above 1 um, 1 mm = "
                     f"{sum(v > 1e-6 for v in d.values())}, {sum(v > 1e-3 for v in d.values())}")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "comparisons.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("variant", nargs="?",
                    choices=("mujoco", "mujoco-gap", "mujoco-newton-changes", "genesis-f32",
                             "genesis-render"))
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    if args.compare:
        compare(args)
    elif args.variant == "mujoco":
        mujoco_variant(args)
    elif args.variant == "mujoco-gap":
        mujoco_gap(args)
    elif args.variant == "mujoco-newton-changes":
        mujoco_newton_changes(args)
    elif args.variant == "genesis-f32":
        genesis_f32(args)
    elif args.variant == "genesis-render":
        genesis_render(args)
    else:
        ap.error("give a variant or --compare")


if __name__ == "__main__":
    main()
