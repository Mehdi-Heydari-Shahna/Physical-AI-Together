#!/usr/bin/env python3
"""
Numerical validation and the reproduction of failures.

Physical AI, Together, Chapter 12 (Section 12.3).

Checks of the numerical model of the declared experiment of Chapter 11, each a
change that leaves the physical model as it is, judged by the criteria in
CRITERIA; and the failures found in Chapters 11 and 12, each with what
reproduces it. Each part runs in the environment of its simulator:

    python numerics.py mujoco     MuJoCo 3.16.0, the environment of ../ch11: the
                                  comparison of Section S11.9.2 at 0.5 and 0.25 ms; the
                                  constraint solver's settings and statistics; the two
                                  collision pipelines against the exact geometry of the
                                  ground; friction and the tool's stop at three time
                                  steps; one period at 1 m/s at three time steps; the
                                  failures that MuJoCo reproduces
    python numerics.py newton     Newton 1.7.0 (requirements-newton.txt): the
                                  comparison at 0.5 and 0.25 ms on MuJoCo's C library,
                                  and with --warp on MuJoCo Warp as well (about 40
                                  minutes on one processor); the contacts of MuJoCo's C
                                  library inside Newton against the exact geometry; one
                                  period at 1 m/s at three time steps; the failures of
                                  Newton's configurations
    python numerics.py genesis    Genesis 1.4.4 (requirements-genesis.txt): the
                                  comparison at 0.5 and 0.25 ms; GJK and MPR against
                                  the exact geometry; one period at 1 m/s at three time
                                  steps; the failures of Genesis's configurations
    python numerics.py usd        an environment with mujoco-usd-converter (0.6.1, the
                                  release that Isaac Sim 6.2.0 bundles, or 0.8.0): the
                                  failure of the MJCF import of Section 11.2
    python numerics.py --report   the numerical validation, the failures, the record's
                                  new entries and the portability report, from the
                                  results present (any Python 3.10 or later, no
                                  simulator)
    python numerics.py --compare-with CODE
                                  a rerun in the code folder CODE (with ch11/results
                                  and ch12/results) against the shipped results, under
                                  the acceptable numerical variation of the record

Options: --check compares every value with the shipped summary first (times and
versions are not compared); --warp (newton) adds MuJoCo Warp at 0.5 and 0.25 ms.

Every comparison is that of Section 12.1.3: in each period of the run of
Chapter 3, the error vector (dp, dv, dl), the end state less the nominal motion
from the state at which the period began; two runs are compared period by
period by the norm of the state of the difference of their vectors,
max(|dp| + |dl|, |dv| * 1 s). Every period starts afresh.

Outputs, in results/numerics/: steps/<configuration>-<step>.csv (one row per
period, 17 significant digits), <part>/summary.txt, and, from --report,
validation.txt, failures.txt and record.txt; from --compare-with,
reproduction.txt; and results/portability-report.txt.
"""
import argparse
import csv
import hashlib
import importlib.metadata as md
import math
import os
import platform
import re
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CH11 = os.path.join(HERE, "..", "ch11")
RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(RESULTS, "numerics")
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True

T = 0.1                       # the period of the governor [s]
STEPS = (0.001, 0.0005, 0.00025)
BOUND = 0.05                  # the bound of Chapter 5 on the disturbance [m/s^2]
COLUMNS = ("k", "t", "p", "v", "l", "u", "ell", "start", "dp", "dv", "dl", "end",
           "within_p", "within_v", "low", "high", "lmax", "w", "w_axle", "w_axle_step")
VOLATILE = ("time of", "time per step", "wall time", "hardware", "Python", "version of")

# The faithful configurations of Section 12.1 and their runs at 1 ms
CONFIGS = (
    ("mujoco", "MuJoCo 3.16.0", os.path.join(CH11, "results", "agreement-chapter3.csv")),
    ("newton-mujoco-c", "Newton 1.7.0 on MuJoCo's C library",
     os.path.join(RESULTS, "newton-mujoco-c", "agreement.csv")),
    ("newton-mujoco-warp", "Newton 1.7.0 on MuJoCo Warp",
     os.path.join(RESULTS, "newton-mujoco-warp", "agreement.csv")),
    ("genesis", "Genesis 1.4.4, the options of the text",
     os.path.join(RESULTS, "genesis", "agreement.csv")),
)
# Other runs at 1 ms whose speed signal the estimation check reads
OTHERS = (
    ("newton-xpbd", "Newton 1.7.0, XPBD", os.path.join(RESULTS, "newton-xpbd", "agreement.csv")),
    ("isaac", "Isaac Sim 6.2.0 (Section 11.2)",
     os.path.join(CH11, "results", "isaac", "agreement-chapter3-isaac.csv")),
)

# The record's criteria for the checks of the numerical model. Tolerances:
# 1 um for positions and 1 um/s for speeds, choices; 0.005 m/s^2 for
# accelerations, a tenth of the bound of Chapter 5 and the resolution of the
# bisection that measures the braking the ground allows; the constraint
# solver and collision detection state their own.
TOLERANCE = {"position": 1e-6, "speed": 1e-6, "acceleration": 0.005}
CRITERIA = {
    "time step": "halving the time step twice: a value of the comparison of Section 11.1.4 "
                 "is resolved at 1 ms if neither halving changes it by more than its "
                 "tolerance (1 um, 1 um/s, 0.005 m/s^2), once the integrator's error u h T/2 "
                 "in the position at the end of a period, which the record predicts, is set "
                 "apart, and a rate read over one step is judged by its change over that "
                 "step, its jump; otherwise the record reports it as unresolved at 1 ms, with "
                 "its values at the three steps",
    "constraint solver": "a tolerance 10^4 times tighter with ten times the iterations, or no "
                         "warm start, changes no period's error by more than 1 nm; no step "
                         "ends at the iteration limit",
    "collision detection": "every contact of a wheel with the ground agrees with the exact "
                           "geometry of the ground and the wheels, as MuJoCo compiled them "
                           "(MuJoCo, and MuJoCo's C library inside Newton) or as the text "
                           "gives them (Genesis), to within 1 um in distance and 1e-4 rad in "
                           "normal",
    "friction": "neither halving of the time step changes the braking that the drive loop "
                "sustains by more than 0.005 m/s^2, or the encoders' creep by more than 1 um/s",
    "limits": "neither halving of the time step changes the tool's reach at the end of period "
              "201, or its largest reach against the stop, by more than 1 um",
    "estimation": "in every period, the position at the end lies within what the speed "
                  "signal allows over the period, to within 1 um (the test of Section 11.2.3, "
                  "which a signal that is the rate of the position passes)",
    "arithmetic": "in single precision, one period at 1 m/s from 5 m ends within 1 um of what "
                  "the spacing of the representable positions predicts",
}

# The acceptable numerical variation of a reproduction, as the record states it
VARIATION = (
    ("a second run, the same environment", "every value identical, bit for bit"),
    ("a new environment from the requirements files, the same kind of machine",
     "every value identical, bit for bit"),
    ("another machine or operating system",
     "every quoted value equal at its quoted precision, or within one unit of its last digit; "
     "and every period's error vector within 1 nm of the shipped one in double precision "
     "(about thirty times the largest change that the perturbations of the constraint solver "
     "produce, a choice) and within two spacings of the representable positions in single "
     "precision (the difference that the emulation of Newton in Section 12.2.2 left), the "
     "spacing taken at the farthest position of the run"),
)


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
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


def hardware():
    """The machine, as the records of Chapters 11 and 12 name it."""
    name = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    name = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return f"{platform.system()} {platform.machine()}, {name}"


def ms(h):
    return f"{h * 1000:g} ms"


def step_path(config, h):
    if h == STEPS[0]:
        return dict((c, p) for c, _, p in CONFIGS)[config]
    return os.path.join(OUT, "steps", f"{config}-{h * 1000:g}ms.csv")


STEP_DIFFERENCES = []        # per-period files that a --check run found changed


def store_rows(path, rows, check):
    """Writes a per-period file; with check, first compares every value with
    the shipped file and notes any that differ."""
    rows = list(rows)
    if check:
        old = read_rows(path)
        if old is None:
            STEP_DIFFERENCES.append(f"{os.path.basename(path)}: no shipped file")
        else:
            new = {r["k"]: {c: float(f"{float(r[c]):.17g}") for c in COLUMNS[1:]} for r in rows}
            differ = sum(new[k][c] != old[k][c] for k in new for c in COLUMNS[1:]
                         if k in old and c in old[k])
            if differ or set(new) != set(old):
                STEP_DIFFERENCES.append(f"{os.path.basename(path)}: {differ} values differ")
    write_rows(path, rows)


def write_rows(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow([r["k"]] + [f"{float(r[c]):.17g}" for c in COLUMNS[1:]])


def read_rows(path):
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8") as fh:
        return {int(r["k"]): {c: float(v) for c, v in r.items() if c != "k" and v != ""}
                for r in csv.DictReader(fh)}


def read_summary(path):
    facts, group = {}, None
    if not os.path.exists(path):
        return facts
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


def write_summary(path, facts, check):
    """Writes a summary; with check, compares every value with the shipped
    summary first and fails on any difference."""
    failures = []
    if check:
        if not os.path.exists(path):
            failures.append("no shipped summary to compare with")
        else:
            for group, items in read_summary(path).items():
                if group == "check":
                    continue
                for key, want in items.items():
                    if key.startswith(VOLATILE):
                        continue
                    got = facts.get(group, {}).get(key)
                    if got is None or str(got) != want:
                        failures.append(f"{group}.{key}: shipped {want!r}, now {got!r}")
    lines = []
    for group, items in facts.items():
        lines.append(f"[{group}]")
        lines += [f"  {k} = {v}" for k, v in items.items()]
    if check:
        failures += STEP_DIFFERENCES
        lines.append("[check] against the shipped summary and per-period files: " +
                     ("all as shipped" if not failures else "DIFFERENT: " + "; ".join(failures)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if failures:                       # keep the shipped summary
        path += ".new"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if failures:
        print(f"[check] the shipped summary is unchanged; this run's summary is {path}")
        raise SystemExit(1)


def compare(a, b):
    """Median, largest and its period, and the periods above 1 nm and 1 um,
    of the norm of the state of the difference of two runs' error vectors."""
    ks = sorted(set(a) & set(b))
    d = {k: norm(*(a[k][c] - b[k][c] for c in ("dp", "dv", "dl"))) for k in ks}
    worst = max(ks, key=lambda k: d[k])
    return (statistics.median(d.values()), d[worst], worst,
            sum(v > 1e-9 for v in d.values()), sum(v > 1e-6 for v in d.values()))


def beyond_speed(r, h):
    """How far the position at the end of a period lies outside what the
    speed signal allows [m], the test of Section S11.10.3: the simulators
    advance positions with the velocity at the end of each step, so if the
    signal v were the rate of the position, the end difference dp would lie
    within [low T, high T] + u h T / 2."""
    lo = r["low"] * T + r["u"] * h * T / 2
    hi = r["high"] * T + r["u"] * h * T / 2
    return max(0.0, r["dp"] - hi, lo - r["dp"])


def spacing(x):
    """The spacing of the numbers in single precision at x."""
    return 2.0 ** (math.floor(math.log2(abs(x))) - 23)


def rounding_prediction(p0, v, h):
    """The end of one period at speed v from p0 in single precision, less the
    exact end, if each step adds v h to a representable position and rounds
    the sum to the nearest representable one."""
    s = spacing(p0)
    n = v * h / s
    return (round(n) - n) * s * round(T / h)


# ----------------------------------------------------------------------------
# The exact geometry of the ground
# ----------------------------------------------------------------------------
def upper_profile(points):
    """The upper boundary of the convex hull of the points (x, z), from left
    to right: the surface on which the wheels roll."""
    hull = []
    for p in sorted(set(points)):
        while len(hull) >= 2 and ((hull[-1][0] - hull[-2][0]) * (p[1] - hull[-2][1])
                                  - (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0])) >= 0:
            hull.pop()
        hull.append(p)
    return hull


def closest_on_profile(profile, c):
    """(distance, closest point) from the point c = (x, z) to the profile."""
    best = (math.inf, None)
    for (x0, z0), (x1, z1) in zip(profile[:-1], profile[1:]):
        ex, ez = x1 - x0, z1 - z0
        t = ((c[0] - x0) * ex + (c[1] - z0) * ez) / (ex * ex + ez * ez)
        t = max(0.0, min(1.0, t))
        q = (x0 + t * ex, z0 + t * ez)
        dist = math.hypot(c[0] - q[0], c[1] - q[1])
        if dist < best[0]:
            best = (dist, q)
    return best


def below_profile(profile, c):
    """Whether the point c = (x, z) lies below the profile, inside the ground."""
    for (x0, z0), (x1, z1) in zip(profile[:-1], profile[1:]):
        if x0 <= c[0] <= x1:
            return c[1] < z0 + (z1 - z0) * (c[0] - x0) / (x1 - x0)
    return False


class ContactErrors:
    """The largest differences between the contacts that a simulator reports
    for a wheel and the exact contact of a sphere of radius r with the
    profile: in signed distance, in the angle of the normal, and in the
    point, the midpoint between the two surfaces."""

    def __init__(self, profile, r, trace=None):
        self.profile, self.r = profile, r
        self.dist = self.angle = self.point = 0.0
        self.count = 0
        self.trace = trace          # a list for (x of the wheel, errors), or None

    def add(self, centre, dist, normal, point, r=None):
        """centre (x, z) of the wheel; the simulator's signed distance
        (negative in penetration), its normal (x, z) from the ground towards
        the wheel or the reverse, and its point (x, z); r, the radius of the
        wheel as the simulator compiled it, if it differs from the design's."""
        r = self.r if r is None else r
        d, q = closest_on_profile(self.profile, centre)
        nx, nz = (centre[0] - q[0]) / d, (centre[1] - q[1]) / d
        if below_profile(self.profile, centre):
            # the centre of the wheel inside the ground: the distance of the
            # centre from the surface is negative, and the outward normal
            # points from the centre to the surface
            d, nx, nz = -d, -nx, -nz
        # the angle between the two normals, whichever way the simulator's points
        cross = abs(normal[0] * nz - normal[1] * nx)
        dot = abs(normal[0] * nx + normal[1] * nz)
        mid = ((q[0] + centre[0] - r * nx) / 2, (q[1] + centre[1] - r * nz) / 2)
        e_dist, e_angle = abs(dist - (d - r)), math.atan2(cross, dot)
        e_point = math.hypot(point[0] - mid[0], point[1] - mid[1])
        self.dist = max(self.dist, e_dist)
        self.angle = max(self.angle, e_angle)
        self.point = max(self.point, e_point)
        self.count += 1
        if self.trace is not None:
            self.trace.append((centre[0], e_dist, e_angle, e_point))

    def facts(self):
        return f"{self.count}; {g3(self.dist)}, {g3(self.angle)}, {g3(self.point)}"

    def passes(self):
        return self.dist <= 1e-6 and self.angle <= 1e-4


CONTACT_KEY = ("contacts of the wheels; largest difference from the exact geometry: "
               "distance [m], normal [rad], point [m]")
SCENARIOS = (
    ("the firm ground, 1 m/s from 5 m for 0.3 s", 5.0, 1.0, 0.0, 0.3),
    ("the crest, braking at 0.5 m/s^2 from 2.12 m/s at 20 m for 1.4 s", 20.0, 2.12, -0.5, 1.4),
    ("the slope, 1 m/s from 23 m for 0.3 s", 23.0, 1.0, 0.0, 0.3),
)


def write_trace(path, trace):
    """One row per contact of a wheel during the motion over the crest: the
    x of the wheel's centre and the three differences from the exact
    geometry."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(("x", "distance", "normal", "point"))
        for row in trace:
            w.writerow([f"{v:.9g}" for v in row])


def profile_from_text(xml):
    """The profile of the ground from the vertices of its mesh in the MJCF text."""
    v = [float(x) for x in re.search(r'mesh name="ground" vertex="([^"]*)"', xml).group(1).split()]
    return upper_profile([(v[i], v[i + 2]) for i in range(0, len(v), 3) if v[i + 1] < 0])


def cart_geoms(m):
    """The geoms of the ground and of the two wheels in a MuJoCo model of the
    cart, by their types: the one mesh, and the two spheres. (Newton names the
    geoms of the model that it builds after their paths.)"""
    import mujoco
    meshes = [i for i in range(m.ngeom) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH]
    spheres = [i for i in range(m.ngeom) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE]
    if len(meshes) != 1 or len(spheres) != 2:
        raise RuntimeError(f"expected one mesh and two spheres, got {len(meshes)} and "
                           f"{len(spheres)}")
    return meshes[0], set(spheres)


def profile_from_mujoco(m, d):
    """The profile of the ground that MuJoCo compiled: the vertices of its
    mesh, which MuJoCo stores in single precision in the mesh's frame, placed
    in the world."""
    import numpy
    g = cart_geoms(m)[0]
    mesh = m.geom_dataid[g]
    a, n = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
    local = m.mesh_vert[a:a + n].astype(numpy.float64)
    world = local @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
    return upper_profile([(float(x), float(z)) for x, y, z in world if y < 0])


def mujoco_contact_errors(m, d, profile, r, errors):
    """Adds every contact of a wheel with the ground in MuJoCo's data."""
    ground, wheels = cart_geoms(m)
    for c in d.contact[:d.ncon]:
        pair = {int(c.geom1), int(c.geom2)}
        if ground not in pair or not (pair & wheels):
            continue
        wheel = (pair - {ground}).pop()
        x, _, z = d.geom_xpos[wheel]
        n = c.frame[:3]
        errors.add((float(x), float(z)), float(c.dist), (float(n[0]), float(n[2])),
                   (float(c.pos[0]), float(c.pos[2])), float(m.geom_size[wheel][0]))


# ----------------------------------------------------------------------------
# Failures: what failed, where, what catches it, and what reproduces it
# ----------------------------------------------------------------------------
FAILURES = (
    dict(id="F1", what="a wheel sinks into the crest of the ground",
         where="Genesis 1.4.4, MPR collision detection (its default)",
         kind="mistake of conversion: the computation of the contacts; a numerical failure of "
              "the algorithm that detects them",
         caught="the check of the crest (Section 12.1.3)", part="genesis"),
    dict(id="F2", what="the state stops being a number while the cart settles at rest",
         where="Newton 1.7.0, Featherstone, Newton's contacts with the importer's defaults",
         kind="another model of contact, whose parameters the text does not carry; the "
              "integration diverges", caught="every port check (Section 12.1.2)",
         part="newton"),
    dict(id="F3", what="the state stops being a number while the cart settles at rest",
         where="Newton 1.7.0, semi-implicit solver, Newton's contacts with the importer's "
               "defaults", kind="the same as F2", caught="every port check (Section 12.1.2)",
         part="newton"),
    dict(id="F4", what="a wheel loses its load on the crest",
         where="Newton 1.7.0 on MuJoCo's C library, the importer's hull of 64 vertices",
         kind="mistake of conversion: the ground of the physical model",
         caught="the check of the crest (Section 12.1.3)", part="newton"),
    dict(id="F5", what="the contacts on the crest are computed differently, and in one period "
                       "the disturbance of the front axle exceeds the bound",
         where="Newton 1.7.0 on MuJoCo's C library, Newton's gap of 0.1 m",
         kind="mistake of conversion: the computation of the contacts (numerical model)",
         caught="the comparison with MuJoCo and the bound of Chapter 5, not the port checks",
         part="newton"),
    dict(id="F6", what="the drive loop sustains almost no braking",
         where="Newton 1.7.0, XPBD with Newton's contacts and the importer's defaults",
         kind="another model of contact and friction", caught="the check of friction",
         part="newton"),
    dict(id="F7", what="too little braking, and a wheel loses its load on the crest",
         where="Genesis 1.4.4 with its own defaults",
         kind="another model of contact and friction (among others, a pyramidal cone and a "
              "contact time constant of 0.01 s), with MPR, whose share the comparison does "
              "not separate",
         caught="the checks of friction and of the crest", part="genesis"),
    dict(id="F8", what="the contacts on the crest are wrong by up to a millimetre in distance, "
                       "the loads there by a sixth, and the disturbance exceeds the bound",
         where="MuJoCo 3.16.0, the libccd pipeline (MPR) instead of the native one",
         kind="numerical failure: the algorithm that detects the contacts",
         caught="the check of collision detection (Section 12.3), the comparison with the "
                "native pipeline and the bound of Chapter 5, not the port checks",
         part="mujoco"),
    dict(id="F9", what="a wheel leaves the ground on the crest while the cart rolls freely "
                       "at 3 m/s",
         where="MuJoCo 3.16.0",
         kind="MuJoCo's law of contact, by our reading of Section 11.1.3, at a speed that the "
              "certified plans do not reach; the time step does not change it",
         caught="the diagnostics of Section 11.1.3", part="mujoco"),
    dict(id="F10", what="the MJCF importer fails on the mesh of the ground, written in the text",
         where="mujoco-usd-converter 0.6.1 (Isaac Sim 6.2.0) and 0.8.0",
         kind="import failure: inline meshes are not supported",
         caught="the import itself (Section 11.2.1)", part="usd"),
    dict(id="F11", what="the reported speed is not the rate of the position",
         where="Isaac Sim 6.2.0, PhysX (Section 11.2)",
         kind="numerical failure: PhysX advances the positions in single precision at each of "
              "the 32 substeps of TGS and rounds every advance, by up to half a spacing, so "
              "that the cart moves faster or slower than the velocity that it adds; traced on "
              "the firm ground and on the slope, where 35 of the 41 periods lie, and over ten "
              "steps the rate of the position stayed within that bound of the reported speed "
              "(Section 11.2.5); with PGS, one advance per step, no period ended beyond the "
              "reported speed by more than 0.01 mm",
         caught="the test of the speed signal (Section 11.2.3), and the port checks of one "
                "period and of braking (Section 12.1.3)", part="isaac",
         observed="in the two runs of Section 11.2, on one machine, whose per-period tables "
                  "are identical, byte for byte",
         traced="traced on the firm ground and on the slope, and its variants run, with python "
                "isaac_trace.py (../ch11), on the same machine; the crest not traced"),
    dict(id="F12", what="the loads at rest exceed the weight, and on the crest a wheel reports "
                        "no load in single steps",
         where="Isaac Sim 6.2.0, PhysX (Section 11.2)",
         kind="at rest, a signal: the contact forces of single steps scatter about the loads "
              "of a rigid cart from step to step, the excess at rest is the reading of one "
              "step, and over 200 steps the mean is the weight (Section 11.2.5); on the crest, "
              "not classified: the unloaded steps were not traced, and their cause is "
              "unresolved",
         caught="the port checks of the loads at rest and of the crest (Section 12.1.3)",
         part="isaac", observed="in the two runs of Section 11.2, on one machine",
         traced="the loads at rest traced with python isaac_trace.py (../ch11), on the same "
                "machine; the crest not traced"),
    dict(id="F13", what="the encoders' reading of the speed changes with the time step",
         where="Isaac Sim 6.2.0, PhysX (Section 11.2)",
         kind="numerical failure: a signal that the time step does not settle; at 1 ms the "
              "encoders follow the motion of the rounded positions of F11, and the readings at "
              "1 and 0.5 ms lie within the bounds of that rounding, the reading at 0.25 ms "
              "1.9 mm/s beyond its bound (Section 11.2.5)",
         caught="the criterion of the check of friction (Section 12.3), applied to the "
                "diagnostic of Section 11.2.3, the check itself not run in Isaac Sim",
         part="isaac", observed="at 1 ms in the two runs of Section 11.2, on one machine; at "
                                "0.5 and 0.25 ms in the second run only",
         traced="traced at 1 ms only, with python isaac_trace.py (../ch11), on the same "
                "machine"),
)

# Where the signatures of the failures observed in Isaac Sim are read, in the
# results of Chapter 11: (failure, file, start of the line, how to quote it)
ISAAC_SIGNATURES = (
    ("F11", "isaac-derived.txt", "  Isaac Sim: periods whose end position lies beyond the "
     "reported speed by more than 0.01, 0.1, 1 mm",
     "periods beyond the reported speed by more than 0.01, 0.1, 1 mm: {}"),
    ("F12", "isaac-derived.txt", "  wheel loads at rest less the weight [N], [%]",
     "loads at rest less the weight [N], [%]: {}"),
    ("F12", os.path.join("isaac", "isaac-summary.txt"),
     "  crest, braking at 0.5 from 2.12 m/s at 20 m",
     "crest, braking at the label from 2.12 m/s: {}"),
    ("F13", os.path.join("isaac", "isaac-summary.txt"),
     "  encoder less cart speed at 2 m/s^2, drive loop, at 1 ms [mm/s]",
     "encoders less the cart's speed at 2 m/s^2, at 1 ms [mm/s]: {}"),
    ("F13", os.path.join("isaac", "isaac-summary.txt"),
     "  drive loop: encoder less cart speed at 0.5 and 0.25 ms [mm/s]",
     "the same at 0.5 and 0.25 ms [mm/s]: {}"),
    # the follow-up traces (Section S11.6), as isaac_check.py derives them
    ("F11", "isaac-derived.txt", "  traces: from 5 m: steps; displacement per step in "
     "single-precision spacings, from, to; steps with 2112, 32 times 66; largest departure from "
     "whole spacings (the nine digits printed)",
     "traced from 5 m at 1 m/s: steps; spacings per step, from, to; steps with 32 x 66; "
     "departure from whole spacings: {}"),
    ("F11", "isaac-derived.txt", "  traces: from 5 m: spacings per step if the advance were "
     "rounded once per step, at the reported speed, from, to; steps with 2097",
     "the same, with one rounding per step: spacings per step, from, to; steps with 2097: {}"),
    ("F11", "isaac-derived.txt", "  traces: period 83, on the slope: pitch [rad]; over the "
     "period, rate of the position less reported speed, mean, RMS [mm/s]; less the rounding of "
     "the horizontal slider's advance, mean, RMS [mm/s]",
     "traced in period 83, on the slope: pitch [rad]; rate of the position less reported "
     "speed, mean, RMS; less the rounding, mean, RMS [mm/s]: {}"),
    ("F11", "isaac-derived.txt", "  traces: PGS solver: position updates per step; periods "
     "beyond the reported speed by more than 0.01 mm; largest [mm]",
     "with PGS: position updates per step; periods beyond by more than 0.01 mm; largest [mm]: "
     "{}"),
    ("F12", "isaac-derived.txt", "  traces: at rest on the firm ground: steps; first step's "
     "loads [N]; total, from, to, mean [N]; mean less the weight [%]",
     "traced at rest on the firm ground: steps; first step's loads [N]; total from, to, mean "
     "[N]; mean less the weight [%]: {}"),
    ("F13", "isaac-derived.txt", "  traces: from 5 m: encoder less reported speed, mean [mm/s]; "
     "over 10 steps, encoder less rate of the position, RMS [mm/s]; per step, RMS [mm/s]",
     "traced from 5 m at 1 ms: encoder less reported speed, mean; over 10 steps, encoder less "
     "rate of the position, RMS; per step, RMS [mm/s]: {}"),
    ("F13", "isaac-derived.txt", "  traces: the same at 2.25 m, the encoders' diagnostic, at 1, "
     "0.5 and 0.25 ms [mm/s]",
     "bounds of the rounding at 2.25 m at 1, 0.5 and 0.25 ms [mm/s]: {}"),
)


# ----------------------------------------------------------------------------
# MuJoCo
# ----------------------------------------------------------------------------
def mujoco_sim_class(mc, options=None, disable=(), stats=None):
    """The class Sim of Section 11.1 with options of MuJoCo changed before
    anything is measured, and, with stats, the constraint solver's
    iterations counted at every step."""
    import mujoco

    class Changed(mc.cm.Sim):
        _changed = False

        def static_sink(self):
            # called at the end of Sim.__init__, before anything is measured
            if not self._changed:
                self._changed = True
                for name, value in (options or {}).items():
                    setattr(self.m.opt, name, value)
                for bit in disable:
                    self.m.opt.disableflags |= int(getattr(mujoco.mjtDisableBit, bit))
            return super().static_sink()

        def period(self, u, ell, kind="segment", record=None, steps=None):
            """Sim.period, counting the solver's iterations after every step."""
            if stats is None:
                return super().period(u, ell, kind, record, steps)
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
                mujoco.mj_step1(m, d)
                if record is not None:
                    record.append(self.sample(a_ref, v_ref))
                self.controls(a_ref, v_ref, ell)
                mujoco.mj_step2(m, d)
                n = int(d.solver_niter[:max(1, d.nisland)].max())
                stats["steps"] += 1
                stats["most"] = max(stats["most"], n)
                stats["capped"] += n >= m.opt.iterations
            mujoco.mj_forward(m, d)
            return self.state()

    return Changed


def mujoco_agreement(mc, design, cls=None):
    """The per-period comparison of Section S11.9.2 with the given class in
    place of Sim."""
    world_sim = mc.cm.world.Sim
    if cls is not None:
        mc.cm.world.Sim = cls
    try:
        rows = mc.agreement(design)
    finally:
        mc.cm.world.Sim = world_sim
    return {r["k"]: r for r in rows}


def folded_disturbance(rows):
    """The largest disturbance read from the front axle with the tool folded,
    over 1 ms or one step, and the periods in which it exceeds the bound."""
    folded = [r for r in rows.values() if r["l"] == 0 and r["ell"] == 0]
    worst = max(max(r["w_axle"], r["w_axle_step"]) for r in folded)
    above = sorted(r_k for r_k, r in rows.items()
                   if r["l"] == 0 and r["ell"] == 0 and max(r["w_axle"], r["w_axle_step"]) > BOUND)
    return worst, above


def ranges(ks):
    out, ks = [], sorted(ks)
    i = 0
    while i < len(ks):
        j = i
        while j + 1 < len(ks) and ks[j + 1] == ks[j] + 1:
            j += 1
        out.append(f"{ks[i]}" if i == j else f"{ks[i]} to {ks[j]}")
        i = j + 1
    return ", ".join(out) if out else "none"


def one_period(sim_place_period, p0=5.0, v=1.0):
    """The end of one period at speed v from p0 with no command, less p0 + v T."""
    place, period = sim_place_period
    place(p0, v, 0.0)
    return period(0.0, 0.0)[0] - p0 - v * T


def part_mujoco(args):
    import dataclasses
    import mujoco
    sys.path.insert(0, CH11)
    import mujoco_cart as mc
    import isaac_check

    t0 = time.time()
    design = mc.DESIGN
    facts = {"record": {
        "part": "mujoco",
        "hardware": hardware(),
        "version of mujoco": mujoco.__version__,
        "version of numpy": version("numpy"),
        "Python": platform.python_version(),
    }}

    # --- the time step -------------------------------------------------------------------
    for h in STEPS[1:]:
        rows = mujoco_agreement(mc, dataclasses.replace(design, timestep=h))
        store_rows(step_path("mujoco", h), rows.values(), args.check)
    # the record's settings, again, in full precision: the reference of the
    # comparisons below (the shipped file of Section 11.1 has nine digits)
    reference = mujoco_agreement(mc, design)
    shipped = read_rows(step_path("mujoco", STEPS[0]))
    same = all(float(f"{reference[k][c]:.9g}") == shipped[k][c]
               for k in shipped for c in ("dp", "dv", "dl", "end", "low", "high"))
    facts["record"]["the record's settings, again: equal to the shipped run of Section 11.1 "
                    "at its nine digits"] = "yes" if same else "no"

    # --- the constraint solver -----------------------------------------------------------
    group = {}
    for label, options, disable in (
            ("tolerance 1e-12, 1000 iterations", dict(tolerance=1e-12, iterations=1000), ()),
            ("no warm start", {}, ("mjDSBL_WARMSTART",)),
            ("tolerance 1e-6", dict(tolerance=1e-6), ()),
            ("islands off", {}, ("mjDSBL_ISLAND",)),
            ("the conjugate-gradient solver", dict(solver=int(mujoco.mjtSolver.mjSOL_CG)), ())):
        stats = {"steps": 0, "most": 0, "capped": 0}
        rows = mujoco_agreement(mc, design, mujoco_sim_class(mc, options, disable, stats))
        med, worst, k, n9, n6 = compare(rows, reference)
        group[f"{label}: against the record's settings: median, largest [m], period; periods "
              f"above 1 nm, 1 um"] = f"{g3(med)}, {g3(worst)}, {k}; {n9}, {n6}"
        group[f"{label}: steps; most iterations in a step; steps at the iteration limit"] = (
            f"{stats['steps']}; {stats['most']}; {stats['capped']}")
    stats = {"steps": 0, "most": 0, "capped": 0}
    mujoco_agreement(mc, design, mujoco_sim_class(mc, None, (), stats))
    group = {"the record's settings: steps; most iterations in a step; steps at the iteration "
             "limit": f"{stats['steps']}; {stats['most']}; {stats['capped']}", **group}
    facts["constraint solver"] = group

    # --- collision detection ---------------------------------------------------------------
    group = {}
    for label, disable in (("native pipeline", ()), ("libccd pipeline", ("mjDSBL_NATIVECCD",))):
        cls = mujoco_sim_class(mc, None, disable)
        sim = cls(design)
        mujoco.mj_forward(sim.m, sim.data)
        profile = profile_from_mujoco(sim.m, sim.data)
        verdict = True
        for name, p, v, a, duration in SCENARIOS:
            trace = [] if name.startswith("the crest") else None
            errors = ContactErrors(profile, design.wheel_radius, trace)
            mc.drive(sim, p, v, a, duration,
                     lambda s, i, v_ref: mujoco_contact_errors(s.m, s.data, profile,
                                                               design.wheel_radius, errors))
            group[f"{label}, {name}: {CONTACT_KEY}"] = errors.facts()
            verdict = verdict and errors.passes()
            if trace is not None:
                write_trace(os.path.join(OUT, f"contacts-crest-mujoco-{label.split()[0]}.csv"),
                            trace)
        group[f"{label}: verdict"] = "pass" if verdict else "fail"
        least, lost, worst = mc.crest_contacts(cls(design), 20.0, 2.12, -0.5, 1.4)
        group[f"{label}: crest, braking at the label: least penetration [mm], steps with a "
              f"wheel without contact, load difference from the rigid cart's [%]"] = (
            f"{g3(1000 * least)}, {lost}, {g3(100 * worst)}")
        # the four port checks of Section 12.1.3
        s = cls(design)
        s.place(5.0, 0.0, 0.0)
        for _ in range(10):
            s.period(0.0, 0.0)
        loads = sum(float(s.data.sensor(w + " load").data[0]) for w in ("front", "rear"))
        weight = s.mass * mc.G
        dp = one_period((s.place, s.period))
        cap = mc.capacity(s, 8.0, 4.0, -1)
        verdicts = (abs(loads - weight) <= 1e-3 * weight, abs(dp) <= 1e-4,
                    abs(cap - 2.4525) <= 0.01 * 2.4525, lost == 0)
        group[f"{label}: port checks: loads at rest less the weight [%]; one period at 1 m/s "
              f"from 5 m, dp [m]; braking on firm ground from 4 m/s [m/s^2]; verdicts"] = (
            f"{g3(100 * (loads - weight) / weight)}; {g3(dp)}; {g3(cap)}; "
            + ", ".join("pass" if v else "fail" for v in verdicts))
        if disable:
            rows = mujoco_agreement(mc, design, cls)
            med, w, k, n9, n6 = compare(rows, reference)
            dist, above = folded_disturbance(rows)
            group[f"{label}: against the native pipeline: median, largest [m], period; periods "
                  f"above 1 um"] = f"{g3(med)}, {g3(w)}, {k}; {n6}"
            group[f"{label}: axle, tool folded: largest disturbance [m/s^2]; periods above "
                  f"the bound"] = f"{g3(dist)}; {len(above)}: {ranges(above)}"
            for tol in (1e-8, 1e-10):
                rows = mujoco_agreement(mc, design, mujoco_sim_class(
                    mc, dict(ccd_tolerance=tol, ccd_iterations=1000), disable))
                group[f"{label}, ccd_tolerance {tol:g}, 1000 iterations: axle, tool folded: "
                      f"largest disturbance [m/s^2]"] = g3(folded_disturbance(rows)[0])
    group["multiple contacts off: against the record's settings: identical in every period"] = (
        "yes" if compare(mujoco_agreement(mc, design, mujoco_sim_class(
            mc, None, ("mjDSBL_MULTICCD",))), reference)[1] == 0 else "no")
    facts["collision detection"] = group

    # --- friction --------------------------------------------------------------------------
    group = {}
    caps, creeps = [], []
    for h in STEPS:
        d_h = dataclasses.replace(design, timestep=h)
        caps.append(mc.capacity(mc.cm.Sim(d_h), 8.0, 4.0, -1))
        creeps.append(mc.encoders(d_h))
    group["braking on firm ground from 4 m/s, at 1, 0.5, 0.25 ms [m/s^2]"] = \
        ", ".join(g3(c) for c in caps)
    group["encoders less the cart's speed at 2 m/s^2, at 1, 0.5, 0.25 ms [mm/s]"] = \
        ", ".join(g3(1000 * c) for c in creeps)
    def halved(xs):
        return max(abs(xs[1] - xs[0]), abs(xs[2] - xs[1]))
    group["verdict"] = ("pass" if halved(caps) <= TOLERANCE["acceleration"]
                        and halved(creeps) <= TOLERANCE["speed"] else "fail")
    facts["friction"] = group

    # --- limits: the tool's last approach, and its reference ------------------------------
    group = {}
    euler = []
    for h in (0.001, 0.0005, 0.00025, 0.000125):
        e = dataclasses.replace(design, timestep=h, integrator="Euler")
        euler.append(-mc.agreement(e, only={201})[0]["dl"])
    group["the tool's last approach, Euler, at 1, 0.5, 0.25, 0.125 ms: short of the stop [m]"] = \
        ", ".join(g3(x) for x in euler)
    reference = isaac_check.tool_shortfall(1e-6)
    group["the tool's equation integrated directly (RK4, 1 us): short of the stop [m]"] = \
        g3(reference)
    group["the same, to four digits [m]"] = f"{reference:.4g}"
    facts["limits"] = group

    # --- arithmetic: one period at 1 m/s, double precision -------------------------------------
    vals = []
    for h in STEPS:
        sim = mc.cm.Sim(dataclasses.replace(design, timestep=h))
        vals.append(one_period((sim.place, sim.period)))
    facts["arithmetic"] = {"one period at 1 m/s from 5 m, at 1, 0.5, 0.25 ms: dp [m]":
                           ", ".join(g3(x) for x in vals)}

    # --- failures that MuJoCo reproduces ----------------------------------------------------------
    group = {}
    sim = mujoco_sim_class(mc, None, ("mjDSBL_NATIVECCD",))(design)
    least, lost, worst = mc.crest_contacts(sim, 20.0, 2.12, -0.5, 1.4)
    group["F8: crest, braking at the label: load difference from the rigid cart's [%]"] = \
        g3(100 * worst)
    lost = []
    for h in STEPS:
        sim = mc.cm.Sim(dataclasses.replace(design, timestep=h))
        lost.append(mc.crest_contacts(sim, 19.5, 3.0, None, 1.0)[1])
    group["F9: crest, rolling freely from 3 m/s: steps with a wheel without contact, at 1, 0.5, "
          "0.25 ms"] = ", ".join(str(n) for n in lost)
    group["F9: the same, as time [ms]"] = ", ".join(f"{n * h * 1000:g}" for n, h in zip(lost, STEPS))
    facts["failures"] = group
    facts["record"]["time of the part [s]"] = f"{time.time() - t0:.0f}"
    write_summary(os.path.join(OUT, "mujoco", "summary.txt"), facts, args.check)


# ----------------------------------------------------------------------------
# Newton
# ----------------------------------------------------------------------------
def part_newton(args):
    import cart_ports
    import experiment
    from cart_ports.base import StateNotFinite

    t0 = time.time()
    facts = {"record": {
        "part": "newton",
        "hardware": hardware(),
        **{f"version of {p}": version(p) for p in ("newton", "warp-lang", "mujoco",
                                                   "mujoco-warp", "numpy")},
        "Python": platform.python_version(),
    }}
    solvers = ["mujoco-c"] + (["mujoco-warp"] if args.warp else [])
    for solver in solvers:
        for h in STEPS[1:]:
            world = cart_ports.CartNewton(solver=solver, timestep=h)
            rows = experiment.agreement(world)
            store_rows(step_path(f"newton-{solver}", h), rows, args.check)

    # the contacts of MuJoCo's C library inside Newton, against the exact geometry
    from cart_ports.design import Design
    world = cart_ports.CartNewton(solver="mujoco-c")
    sim = world.sim
    mm, md_ = sim.solver.mj_model, sim.solver.mj_data
    import mujoco
    mujoco.mj_forward(mm, md_)
    profile = profile_from_mujoco(mm, md_)
    r = Design().wheel_radius
    group = {}
    verdict = True
    for name, p, v, a, duration in SCENARIOS:
        errors = ContactErrors(profile, r)
        sim.place(p, v, 0.0)
        for i in range(int(round(duration / sim.dt))):
            sim.controls(a, v + a * i * sim.dt, 0.0)
            sim.step()
            # after mj_step, MuJoCo's contacts and geom positions are those of
            # the state at the start of the step
            mujoco_contact_errors(mm, md_, profile, r, errors)
        group[f"MuJoCo's C library inside Newton, {name}: {CONTACT_KEY}"] = errors.facts()
        verdict = verdict and errors.passes()
    group["verdict"] = "pass" if verdict else "fail"
    facts["collision detection"] = group

    # one period at 1 m/s from 5 m at three time steps, in single precision
    vals, preds = [], []
    for h in STEPS:
        s = cart_ports.CartNewton(solver="mujoco-c", timestep=h).sim
        vals.append(one_period((s.place, s.period)))
        preds.append(rounding_prediction(5.0, 1.0, h))
    facts["arithmetic"] = {
        "MuJoCo's C library: one period at 1 m/s from 5 m, at 1, 0.5, 0.25 ms: dp [m]":
            ", ".join(g3(x) for x in vals),
        "predicted from the spacing of single precision [m]": ", ".join(g3(x) for x in preds),
    }

    # failures
    group = {}
    for fid, solver in (("F2", "featherstone"), ("F3", "semi-implicit")):
        try:
            cart_ports.CartNewton(solver=solver)
            group[f"{fid}: {solver}: steps until the state stops being a number"] = "none"
        except StateNotFinite as exc:
            group[f"{fid}: {solver}: steps until the state stops being a number"] = \
                re.search(r"after (\d+) steps", str(exc)).group(1)
    s = cart_ports.CartNewton(solver="mujoco-c", hull_vertices=64).sim
    unloaded, worst, n, deepest = experiment.crest_loads(s, 20.0, 2.12, -0.5, 1.4)
    group["F4: the importer's hull: steps with a wheel unloaded, of steps on the crest; deepest "
          "below the wheel's radius [mm]"] = f"{unloaded} of {n}; {g3(1000 * deepest)}"
    gap = cart_ports.CartNewton(solver="mujoco-c", gap=None)
    rows = {r["k"]: r for r in experiment.agreement(gap, only={36, 59})}
    ref = read_rows(step_path("mujoco", STEPS[0]))
    group["F5: Newton's gap: period 36, axle disturbance [m/s^2]; period 59, difference from "
          "MuJoCo [m]"] = (f"{g3(max(rows[36]['w_axle'], rows[36]['w_axle_step']))}; "
                           f"{g3(norm(*(rows[59][c] - ref[59][c] for c in ('dp', 'dv', 'dl'))))}")
    xpbd = cart_ports.CartNewton(solver="xpbd").sim
    group["F6: XPBD: braking on firm ground from 4 m/s [m/s^2]"] = \
        g3(experiment.capacity(xpbd, 8.0, 4.0, -1))
    facts["failures"] = group
    facts["record"]["time of the part [s]"] = f"{time.time() - t0:.0f}"
    write_summary(os.path.join(OUT, "newton", "summary.txt"), facts, args.check)


# ----------------------------------------------------------------------------
# Genesis
# ----------------------------------------------------------------------------
def genesis_contact_errors(world, profile, scenarios):
    """The contacts of the wheels with the ground in Genesis against the exact
    geometry. Genesis detects the contacts of a step for the state at its
    start, as MuJoCo does."""
    import numpy
    sim = world.sim
    e = sim.entity
    collider = sim.scene.sim.rigid_solver.collider
    ground = e.geoms[0].idx
    wheels = {e.geoms[3].idx: "front", e.geoms[4].idx: "rear"}
    out = {}
    for name, p, v, a, duration in scenarios:
        errors = ContactErrors(profile, sim.design.wheel_radius,
                               [] if name.startswith("the crest") else None)
        sim.place(p, v, 0.0)
        for i in range(int(round(duration / sim.dt))):
            k = sim.kin()
            centres = {"front": k.front, "rear": k.rear}
            sim.controls(a, v + a * i * sim.dt, 0.0)
            sim.step()
            c = collider.get_contacts(as_tensor=True, to_torch=False)
            ga, gb = numpy.asarray(c["geom_a"]), numpy.asarray(c["geom_b"])
            pen, nrm, pos = (numpy.asarray(c[x]) for x in ("penetration", "normal", "position"))
            for j in range(len(ga)):
                pair = {int(ga[j]), int(gb[j])}
                if ground not in pair or not (pair & set(wheels)):
                    continue
                wheel = wheels[(pair - {ground}).pop()]
                errors.add(centres[wheel], -float(pen[j]), (float(nrm[j][0]), float(nrm[j][2])),
                           (float(pos[j][0]), float(pos[j][2])))
        out[name] = errors
    return out


def part_genesis(args):
    import cart_ports
    import experiment
    from cart_ports.design import mjcf

    t0 = time.time()
    facts = {"record": {
        "part": "genesis",
        "hardware": hardware(),
        **{f"version of {p}": version(p) for p in ("genesis-world", "quadrants", "torch",
                                                   "numpy")},
        "Python": platform.python_version(),
    }}
    for h in STEPS[1:]:
        world = cart_ports.CartGenesis(settings="mujoco", timestep=h)
        store_rows(step_path("genesis", h), experiment.agreement(world), args.check)

    world = cart_ports.CartGenesis(settings="mujoco")
    profile = profile_from_text(mjcf(world.sim.design))
    group, failures = {}, {}
    for label, settings in (("GJK", "mujoco"), ("MPR", "mujoco-mpr")):
        w = world if settings == "mujoco" else cart_ports.CartGenesis(settings=settings)
        errs = genesis_contact_errors(w, profile, SCENARIOS)
        for name, errors in errs.items():
            group[f"{label}, {name}: {CONTACT_KEY}"] = errors.facts()
            if errors.trace is not None:
                write_trace(os.path.join(OUT, f"contacts-crest-genesis-{label.lower()}.csv"),
                            errors.trace)
        group[f"{label}: verdict"] = "pass" if all(e.passes() for e in errs.values()) else "fail"
        if settings == "mujoco-mpr":
            unloaded, worst, n, deepest = experiment.crest_loads(w.sim, 20.0, 2.12, -0.5, 1.4)
            failures[
                "F1: MPR: steps with a wheel unloaded, of steps on the crest; deepest below the "
                "wheel's radius [mm]"] = f"{unloaded} of {n}; {g3(1000 * deepest)}"
    facts["collision detection"] = group

    vals = []
    for h in STEPS:
        s = cart_ports.CartGenesis(settings="mujoco", timestep=h).sim
        vals.append(one_period((s.place, s.period)))
    facts["arithmetic"] = {"the options of the text, double precision: one period at 1 m/s "
                           "from 5 m, at 1, 0.5, 0.25 ms: dp [m]": ", ".join(g3(x) for x in vals)}
    facts["failures"] = failures
    facts["record"]["time of the part [s]"] = f"{time.time() - t0:.0f}"
    write_summary(os.path.join(OUT, "genesis", "summary.txt"), facts, args.check)


def part_genesis_defaults(args):
    """F7 needs Genesis in single precision, which a process chooses once."""
    import cart_ports
    import experiment

    s = cart_ports.CartGenesis(settings="defaults").sim
    unloaded, worst, n, deepest = experiment.crest_loads(s, 20.0, 2.12, -0.5, 1.4)
    cap = experiment.capacity(s, 8.0, 4.0, -1)
    facts = {"record": {"part": "genesis-defaults", "hardware": hardware(),
                        **{f"version of {p}": version(p) for p in ("genesis-world", "quadrants",
                                                                   "torch", "numpy")},
                        "Python": platform.python_version()},
             "failures": {"F7: its defaults: braking on firm ground from 4 m/s [m/s^2]; crest, "
                          "steps with a wheel unloaded, of steps on the crest":
                          f"{g3(cap)}; {unloaded} of {n}"}}
    write_summary(os.path.join(OUT, "genesis-defaults", "summary.txt"), facts, args.check)


# ----------------------------------------------------------------------------
# The MJCF import of Section 11.2
# ----------------------------------------------------------------------------
def part_usd(args):
    import shutil
    import subprocess
    import tempfile

    # the converter's command, beside this Python or on the PATH
    exe = (shutil.which("mujoco_usd_converter", path=os.path.dirname(sys.executable))
           or shutil.which("mujoco_usd_converter"))
    if exe is None:
        raise SystemExit("mujoco-usd-converter is not installed in this environment")
    text = os.path.join(CH11, "results", "cart.xml")
    with tempfile.TemporaryDirectory() as tmp:
        res = subprocess.run([exe, text, os.path.join(tmp, "out")], capture_output=True,
                             text=True, timeout=600)
        made = [os.path.join(d, f) for d, _, fs in os.walk(os.path.join(tmp, "out"))
                for f in fs]
        size = sum(os.path.getsize(f) for f in made)
        # what the files hold: their prims, and those with children or
        # authored properties (USD, which the converter installs)
        try:
            from pxr import Usd
            stages = [Usd.Stage.Open(f) for f in made]      # kept open while read
            prims = [p for stage in stages for p in stage.TraverseAll()]
            held = (f"{len(prims)}; "
                    f"{sum(bool(p.GetChildren() or p.GetAuthoredProperties()) for p in prims)}")
            del prims, stages
        except ImportError:
            held = "not inspected: USD is not installed"
    log = res.stdout + res.stderr
    m = re.search(r"Python runtime error: ([^'\n]*)", log)
    release = version("mujoco-usd-converter")
    facts = {"record": {"part": f"usd-{release}", "hardware": hardware(),
                        "version of mujoco-usd-converter": release,
                        "version of mujoco": version("mujoco"),
                        "Python": platform.python_version()},
             "failures": {
                 "F10: converting the MJCF text of Section 11.1: error": m.group(1) if m else
                 "none",
                 "F10: exit status of the converter; files it leaves, their bytes":
                     f"{res.returncode}; {len(made)}, {size}",
                 "F10: prims in those files; prims with children or properties": held}}
    write_summary(os.path.join(OUT, f"usd-{release}", "summary.txt"), facts, args.check)


# ----------------------------------------------------------------------------
# The report: validation, failures, the record's new entries, the portability
# report
# ----------------------------------------------------------------------------
def code_hash():
    """SHA-256 of the code of Chapters 10 to 12 (Python files and requirements),
    in a fixed order: the version of the code that produced the results. The
    results, and virtual environments made inside the folders (.venv,
    venv-newton and the like), are left out."""
    h = hashlib.sha256()
    root = os.path.normpath(os.path.join(HERE, ".."))
    for chapter in ("ch10", "ch11", "ch12"):
        base = os.path.join(root, chapter)
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(
                d for d in dirnames
                if d != "__pycache__" and not d.startswith(("results", ".", "venv"))
                and not os.path.exists(os.path.join(dirpath, d, "pyvenv.cfg")))
            for f in sorted(filenames):
                if f.endswith(".py") or f.startswith("requirements"):
                    path = os.path.join(dirpath, f)
                    h.update(os.path.relpath(path, root).replace(os.sep, "/").encode())
                    with open(path, "rb") as fh:
                        h.update(fh.read().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def step_facts():
    """For each faithful configuration and time step: the values that the
    record reports, and the difference from MuJoCo at the same step."""
    out = {}
    for config, label, _ in CONFIGS:
        for h in STEPS:
            rows = read_rows(step_path(config, h))
            if rows is None:
                continue
            ends = sorted(r["end"] for r in rows.values())
            # the error at the end of each period with the integrator's u h T/2,
            # which the record predicts, set apart from the position
            less = {k: norm(r["dp"] - r["u"] * h * T / 2, r["dv"], r["dl"])
                    for k, r in rows.items()}
            ends_less = sorted(less.values())
            dist, above = folded_disturbance(rows)
            folded = [r for r in rows.values() if r["l"] == 0 and r["ell"] == 0]
            over_ms = max(folded, key=lambda r: r["w_axle"])
            over_step = max(folded, key=lambda r: r["w_axle_step"])
            beyond = [beyond_speed(r, h) for r in rows.values()]
            speed = max((r["within_v"], k) for k, r in rows.items() if k != 201)
            f = dict(rows=rows,
                     median=statistics.median(ends),
                     p90=ends[math.ceil(0.9 * len(ends)) - 1],
                     largest=max(ends),
                     median_less=statistics.median(ends_less),
                     p90_less=ends_less[math.ceil(0.9 * len(ends_less)) - 1],
                     accel_less=rows[1]["dp"] - rows[1]["u"] * h * T / 2,
                     brake=rows[40]["end"], brake_less=less[40],
                     speed=speed,
                     accel=rows[1]["dp"],
                     tool=-rows[201]["dl"],
                     lmax=max(r["lmax"] for r in rows.values()),
                     axle=dist, above=len(above), above_k=above,
                     axle_ms=(over_ms["w_axle"], [k for k, r in rows.items() if r is over_ms][0]),
                     axle_step=(over_step["w_axle_step"],
                                [k for k, r in rows.items() if r is over_step][0]),
                     beyond=sum(b > 1e-6 for b in beyond), beyond_max=max(beyond))
            mujoco_rows = read_rows(step_path("mujoco", h))
            if config != "mujoco" and mujoco_rows is not None:
                f["against"] = compare(rows, mujoco_rows)
            out[(config, h)] = f
    return out


def figure_data(facts, parts):
    """The data of Figure S12.4: steps.csv, the difference of each faithful port
    from MuJoCo at the same time step; limits.csv, the tool's last approach
    against the time step; contacts-crest-binned.csv, the largest difference
    of a contact's distance from the exact one in each 10 mm of the wheel's
    path over the crest."""
    with open(os.path.join(OUT, "steps.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        names = ("newton-mujoco-c", "newton-mujoco-warp", "genesis")
        w.writerow(["step_ms"] + [f"{n}_{q}".replace("-", "_") for n in names
                                  for q in ("median", "largest")])
        for h in STEPS:
            row = [f"{h * 1000:g}"]
            for n in names:
                f = facts.get((n, h))
                row += [f"{f['against'][0]:.6g}", f"{f['against'][1]:.6g}"] if f else ["nan", "nan"]
            w.writerow(row)
    # the contacts over the crest: the largest difference from the exact
    # distance in each 10 mm of the wheel's path, for each pipeline
    traces = {}
    for name in ("mujoco-native", "mujoco-libccd", "genesis-gjk"):
        path = os.path.join(OUT, f"contacts-crest-{name}.csv")
        if os.path.exists(path):
            with open(path, newline="", encoding="utf-8") as fh:
                traces[name] = [(float(r["x"]), float(r["distance"])) for r in csv.DictReader(fh)]
    if traces:
        bins = {}
        for name, rows in traces.items():
            for x, d in rows:
                b = math.floor(x / 0.01)
                bins.setdefault(b, {}).setdefault(name, 0.0)
                bins[b][name] = max(bins[b][name], d)
        with open(os.path.join(OUT, "contacts-crest-binned.csv"), "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["x"] + [n.replace("-", "_") for n in traces])
            for b in sorted(bins):
                w.writerow([f"{(b + 0.5) * 0.01:.3f}"] + [
                    f"{bins[b][n]:.6g}" if n in bins[b] else "nan" for n in traces])
    euler = [float(x) for x in parts["mujoco"]["limits"][
        "the tool's last approach, Euler, at 1, 0.5, 0.25, 0.125 ms: short of the stop [m]"
    ].split(", ")]
    reference = float(parts["mujoco"]["limits"][
        "the tool's equation integrated directly (RK4, 1 us): short of the stop [m]"])
    ch11 = read_summary(os.path.join(CH11, "results", "mujoco-summary.txt"))
    small = [float(x) for x in ch11["the same at smaller time steps"][
        "tool unfolds, reach short of the stop, implicitfast at 0.125, 0.0625 ms [m]"
    ].split(", ")]
    with open(os.path.join(OUT, "limits.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(("step_ms", "mujoco", "newton_mujoco_c", "genesis", "mujoco_euler",
                    "reference"))
        for i, h in enumerate(STEPS + (0.000125, 0.0000625)):
            row = [f"{h * 1000:g}"]
            for n in ("mujoco", "newton-mujoco-c", "genesis"):
                f = facts.get((n, h))
                if f is not None:
                    row.append(f"{1000 * f['tool']:.6g}")
                elif n == "mujoco" and h in (0.000125, 0.0000625):
                    row.append(f"{1000 * small[0 if h == 0.000125 else 1]:.6g}")
                else:
                    row.append("nan")
            row.append(f"{1000 * euler[i]:.6g}" if i < len(euler) else "nan")
            row.append(f"{1000 * reference:.6g}")
            w.writerow(row)


# The values of the comparison of Section S11.9.2 that the check of the time
# step judges, and those that the check of limits judges: (name, the value
# in the facts of a step, its kind)
STEP_VALUES = (
    ("the end of a period, median, less u h T/2", lambda f: f["median_less"], "position"),
    ("the end of a period, 90th percentile, less u h T/2", lambda f: f["p90_less"],
     "position"),
    ("accelerating from rest, period 1, dp less u h T/2", lambda f: f["accel_less"],
     "position"),
    ("braking at 2 m/s^2, period 40, the end less u h T/2", lambda f: f["brake_less"],
     "position"),
    ("the speed within a period outside period 201, largest", lambda f: f["speed"][0],
     "speed"),
    ("the axle's disturbance over 1 ms, tool folded, largest", lambda f: f["axle_ms"][0],
     "acceleration"),
    ("the axle's disturbance over one step, tool folded, largest",
     lambda f: f["axle_step"][0], "acceleration"),
)
REACH = 0.8                   # the end of the tool's stroke [m], the design's reach
LIMIT_VALUES = (
    ("the tool's reach at the end of period 201, short of the stop", lambda f: f["tool"],
     "position"),
    ("the tool's largest reach beyond the end of its stroke, against the stop",
     lambda f: f["lmax"] - REACH, "position"),
)
UNIT = {"position": "m", "speed": "m/s", "acceleration": "m/s^2"}


def verdicts(got, values):
    """Each value at the three steps, the larger change of the two halvings,
    and whether it is resolved at 1 ms: neither halving changes it by more
    than its tolerance."""
    out = []
    for name, get, kind in values:
        xs = [get(g) for g in got]
        change = max(abs(xs[1] - xs[0]), abs(xs[2] - xs[1]))
        out.append((name, xs, change, change <= TOLERANCE[kind], kind))
    return out


def verdict_lines(label, judged):
    """The lines of validation.txt that give the verdicts of a configuration."""
    ok = [name for name, _, _, resolved, _ in judged if resolved]
    no = [f"{name} [{UNIT[kind]}]: {', '.join(g3(x) for x in xs)}; {g3(change)}"
          for name, xs, change, resolved, kind in judged if not resolved]
    return [f"  {label}: resolved at 1 ms = " + ("; ".join(ok) if ok else "none"),
            f"  {label}: unresolved at 1 ms: values at 1, 0.5, 0.25 ms; larger change of a "
            f"halving = " + ("; ".join(no) if no else "none")]


def ch11_jump():
    """The jump of the speed signal against the axle in one step at the crest,
    at 1, 0.5 and 0.25 ms [m/s], and the same read over one step [m/s^2], from
    the results of Section 11.1."""
    groups = read_summary(os.path.join(CH11, "results", "mujoco-summary.txt"))
    items = {k: v for g in groups.values() for k, v in g.items()}
    jump = [float(items["jump of the speed signal against the axle in one step: largest "
                        "[m/s], period, at p"].split(",")[0])]
    rate = [float(items["speed signal: acceleration less command over 1 ms, tool folded: "
                        "largest, period, at p"].split(",")[0])]
    for h in ("0.5 ms", "0.25 ms"):
        jump.append(float(items[f"{h}: jump of the speed signal against the axle in one step "
                                f"[m/s], period"].split(",")[0]))
        rate.append(float(items[f"{h}: acceleration less command over one step, tool folded, "
                                f"above 0.05 m/s^2"].split("largest ")[1].split()[0]))
    return jump, rate


def report(args):
    facts = step_facts()
    parts = {p: read_summary(os.path.join(OUT, p, "summary.txt"))
             for p in ("mujoco", "newton", "genesis", "genesis-defaults", "usd-0.6.1",
                       "usd-0.8.0")}
    lines = ["Numerical validation of the declared experiment (Section 12.3): each check "
             "changes the numerical model and leaves the physical model as it is.", ""]

    # the time step
    unresolved = {}
    lines.append("[time step]")
    lines.append(f"  criterion = {CRITERIA['time step']}")
    for config, label, _ in CONFIGS:
        got = [facts.get((config, h)) for h in STEPS]
        if any(g is None for g in got):
            lines.append(f"  {label} = not run at every step")
            continue
        lines.append(f"  {label}: end of period, median; 90th percentile; largest [m], at 1, "
                     f"0.5, 0.25 ms = " + "; ".join(
                         ", ".join(g3(g[key]) for g in got) for key in ("median", "p90",
                                                                        "largest")))
        lines.append(f"  {label}: accelerating from rest (period 1), dp [m] = "
                     + ", ".join(g3(g["accel"]) for g in got))
        lines.append(f"  {label}: axle, tool folded, largest disturbance over 1 ms, period; "
                     f"over one step, period [m/s^2] = " + "; ".join(
                         f"{g3(g['axle_ms'][0])}, {g['axle_ms'][1]}, {g3(g['axle_step'][0])}, "
                         f"{g['axle_step'][1]}" for g in got))
        lines.append(f"  {label}: axle, tool folded, periods above the bound = " + "; ".join(
            f"{g['above']}" + (f" ({ranges(g['above_k'])})" if g['above'] else "")
            for g in got))
        if config != "mujoco":
            lines.append(f"  {label}: against MuJoCo at the same step: median, largest [m], "
                         f"period; periods above 1 um = " + "; ".join(
                             f"{g3(g['against'][0])}, {g3(g['against'][1])}, "
                             f"{g['against'][2]}, {g['against'][4]}" for g in got))
        lines.append(f"  {label}: braking at 2 m/s^2 (period 40), end [m] = "
                     + ", ".join(g3(g["brake"]) for g in got))
        lines.append(f"  {label}: less the integrator's u h T/2: end of period, median; 90th "
                     f"percentile [m] = " + "; ".join(
                         ", ".join(g3(g[key]) for g in got) for key in ("median_less",
                                                                        "p90_less")))
        lines.append(f"  {label}: less the integrator's u h T/2: accelerating from rest "
                     f"(period 1), dp; braking at 2 m/s^2 (period 40), end [m] = " + "; ".join(
                         ", ".join(g3(g[key]) for g in got) for key in ("accel_less",
                                                                        "brake_less")))
        lines.append(f"  {label}: speed within a period outside period 201: largest [m/s], "
                     f"period = " + "; ".join(f"{g3(g['speed'][0])}, {g['speed'][1]}"
                                              for g in got))
        judged = verdicts(got, STEP_VALUES)
        if config == "mujoco":
            jump, rate = ch11_jump()
            lines.append(f"  {label} (Section 11.1): the jump of the speed signal against the "
                         f"axle in one step, at the crest [m/s] = "
                         + ", ".join(g3(x) for x in jump) + "; read over one step [m/s^2] = "
                         + ", ".join(g3(x) for x in rate))
            change = max(abs(jump[1] - jump[0]), abs(jump[2] - jump[1]))
            judged.append(("the jump of the speed signal at the crest, in one step", jump,
                           change, change <= TOLERANCE["speed"], "speed"))
        lines += verdict_lines(label, judged)
        unresolved[config] = [j for j in judged if not j[3]]
    lines.append("")

    # the constraint solver and collision detection: from the parts
    for group in ("constraint solver", "collision detection"):
        lines.append(f"[{group}]")
        lines.append(f"  criterion = {CRITERIA[group]}")
        for part in ("mujoco", "newton", "genesis"):
            for k, v in parts[part].get(group, {}).items():
                lines.append(f"  {part}: {k} = {v}")
        lines.append("")

    # the traces of the motion over the crest
    lines.append("[collision detection over the crest, contact by contact]")
    for name in ("mujoco-native", "mujoco-libccd", "genesis-gjk", "genesis-mpr"):
        path = os.path.join(OUT, f"contacts-crest-{name}.csv")
        if not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            dist = [float(r["distance"]) for r in csv.DictReader(fh)]
        lines.append(f"  {name}: contacts; beyond 1 um, 0.1 mm in distance; largest [m] = "
                     f"{len(dist)}; {sum(d > 1e-6 for d in dist)}, {sum(d > 1e-4 for d in dist)}; "
                     f"{g3(max(dist))}")
    lines.append("")

    # friction
    lines.append("[friction]")
    lines.append(f"  criterion = {CRITERIA['friction']}")
    for k, v in parts["mujoco"].get("friction", {}).items():
        lines.append(f"  mujoco: {k} = {v}")
    lines.append("")

    # limits
    lines.append("[limits]")
    lines.append(f"  criterion = {CRITERIA['limits']}")
    for config, label, _ in CONFIGS:
        got = [facts.get((config, h)) for h in STEPS]
        if any(g is None for g in got):
            continue
        lines.append(f"  {label}: the tool's reach at the end of period 201, short of the "
                     f"stop, at 1, 0.5, 0.25 ms [m] = " + ", ".join(g3(g["tool"]) for g in got))
        lines.append(f"  {label}: extrapolated from 0.5 and 0.25 ms, first order [m] = "
                     + g3(2 * got[2]["tool"] - got[1]["tool"]))
        lines.append(f"  {label}: largest reach against the stop, at 1, 0.5, 0.25 ms [m] = "
                     + ", ".join(f"{g['lmax']:.7f}" for g in got))
        judged = verdicts(got, LIMIT_VALUES)
        lines += verdict_lines(label, judged)
        unresolved.setdefault(config, []).extend(j for j in judged if not j[3])
    for k, v in parts["mujoco"].get("limits", {}).items():
        lines.append(f"  mujoco: {k} = {v}")
    lines.append("")

    # estimation
    lines.append("[estimation]")
    lines.append(f"  criterion = {CRITERIA['estimation']}")
    for config, label, _ in CONFIGS:
        got = [facts.get((config, h)) for h in STEPS]
        if any(g is None for g in got):
            continue
        lines.append(f"  {label}: periods beyond the speed signal by more than 1 um; largest "
                     f"[m], at 1, 0.5, 0.25 ms = " + "; ".join(
                         f"{g['beyond']}, {g3(g['beyond_max'])}" for g in got))
    for config, label, path in OTHERS:
        rows = read_rows(path)
        if rows is None:
            continue
        beyond = [beyond_speed(r, STEPS[0]) for r in rows.values()]
        lines.append(f"  {label}: periods beyond the speed signal by more than 1 um, 10 um; "
                     f"largest [m], at 1 ms = {sum(b > 1e-6 for b in beyond)}, "
                     f"{sum(b > 1e-5 for b in beyond)}; {g3(max(beyond))}")
    lines.append("")

    # arithmetic
    lines.append("[arithmetic]")
    lines.append(f"  criterion = {CRITERIA['arithmetic']}")
    for part in ("mujoco", "newton", "genesis"):
        for k, v in parts[part].get("arithmetic", {}).items():
            lines.append(f"  {part}: {k} = {v}")
    lines.append("")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "validation.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # failures
    sig = {}
    for part, items in parts.items():
        for k, v in items.get("failures", {}).items():
            fid = k.split(":", 1)[0]
            sig.setdefault(fid, []).append((part, k.split(": ", 1)[1], v))
    flines = ["Failures of the declared experiment found in Chapters 11 and 12, and what "
              "reproduces them (Section 12.3)", ""]
    for f in FAILURES:
        flines.append(f"[{f['id']}]")
        for key in ("what", "where", "kind", "caught"):
            flines.append(f"  {key} = {f[key]}")
        if f["part"] == "isaac":
            flines.append("  reproduction = Isaac Sim 6.2.0 on a machine with an RTX GPU: "
                          f"python isaac_cart.py (../ch11); observed {f['observed']}; "
                          f"{f['traced']}")
            for fid, name, start, quote in ISAAC_SIGNATURES:
                if fid != f["id"]:
                    continue
                with open(os.path.join(CH11, "results", name), encoding="utf-8") as fh:
                    found = [line.rstrip("\n").split(" = ", 1)[1] for line in fh
                             if line.startswith(start + " = ")]
                flines.append("  signature (isaac) = " + quote.format(
                    found[0] if len(found) == 1 else "not found"))
        else:
            cmd = {"mujoco": "python numerics.py mujoco (the environment of ../ch11)",
                   "newton": "python numerics.py newton (the Newton environment)",
                   "genesis": "python numerics.py genesis (the Genesis environment)",
                   "usd": "python numerics.py usd (an environment with mujoco-usd-converter)"}
            if f["id"] == "F7":
                flines.append("  reproduction = python numerics.py genesis-defaults (the Genesis "
                              "environment, a process of its own)")
            else:
                flines.append(f"  reproduction = {cmd[f['part']]}")
            for part, key, value in sig.get(f["id"], []):
                flines.append(f"  signature ({part}) = {key}: {value}")
    with open(os.path.join(OUT, "failures.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(flines) + "\n")
    print("\n".join(flines))

    figure_data(facts, parts)

    # the record's new entries
    rlines = ["New entries of the record of the experiment (Section 12.3), for the "
              "configuration of Section 11.1 and the faithful ports", "",
              "numerical validation"]
    for name, text in CRITERIA.items():
        rlines.append(f"  {name}: {text}")
    rlines.append("  collision algorithms: MuJoCo 3.16.0, its native pipeline (GJK and EPA), the "
                  "default; Newton 1.7.0 on MuJoCo's C library, the same; Newton 1.7.0 on MuJoCo "
                  "Warp, MuJoCo Warp's own; Genesis 1.4.4, GJK, which the port sets (its default "
                  "is MPR)")
    for config, label, _ in CONFIGS:
        if unresolved.get(config):
            rlines.append(f"  unresolved at 1 ms, {label}, values at 1, 0.5, 0.25 ms: " + "; ".join(
                f"{name} [{UNIT[kind]}] = {', '.join(g3(x) for x in xs)}"
                for name, xs, _, _, kind in unresolved[config]))
    tools = [facts.get((c, h)) for c in ("mujoco", "genesis") for h in STEPS]
    if all(tools):
        lo = min(2 * facts[(c, STEPS[2])]["tool"] - facts[(c, STEPS[1])]["tool"]
                 for c in ("mujoco", "genesis"))
        hi = max(2 * facts[(c, STEPS[2])]["tool"] - facts[(c, STEPS[1])]["tool"]
                 for c in ("mujoco", "genesis"))
        reference = parts["mujoco"].get("limits", {}).get(
            "the tool's equation integrated directly (RK4, 1 us): short of the stop [m]", "?")
        rlines.append(f"  the tool's last approach: extrapolated from 0.5 and 0.25 ms at first "
                      f"order, {g3(1000 * lo)} to {g3(1000 * hi)} mm short of the stop in "
                      f"MuJoCo and in Genesis; the tool's equation integrated directly, "
                      f"{g3(1000 * float(reference))} mm")
    rlines.append("  outcomes: results/numerics/validation.txt")
    rlines.append("failures")
    rlines += [f"  {f['id']}: {f['what']}; {f['where']}" for f in FAILURES]
    rlines.append("  reproductions and signatures: results/numerics/failures.txt")
    rlines.append("acceptable numerical variation of a reproduction")
    rlines += [f"  {a}: {b}" for a, b in VARIATION]
    with open(os.path.join(OUT, "record.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(rlines) + "\n")

    # the portability report
    plines = ["Portability report of the declared experiment of Chapter 11",
              "Physical AI, Together, Chapter 12",
              f"code of Chapters 10 to 12, SHA-256 {code_hash()}",
              ""]
    for name in ("support-matrix.txt",):
        with open(os.path.join(RESULTS, name), encoding="utf-8") as fh:
            plines += ["== The support matrix (Section 12.1)", fh.read().rstrip(), ""]
    with open(os.path.join(RESULTS, "layers", "comparisons.txt"), encoding="utf-8") as fh:
        plines += ["== What each comparison changes (Section 12.2)", fh.read().rstrip(), ""]
    newton = os.path.join(RESULTS, "layers", "mujoco-3.15.0-newton-changes", "summary.txt")
    if os.path.exists(newton):
        with open(newton, encoding="utf-8") as fh:
            kept = [line.rstrip("\n") for line in fh
                    if not line.startswith(("  time of", "[check]"))]
        plines += ["== The changes that Newton makes around MuJoCo (Section 12.2, Table 12.4)",
                   "\n".join(kept).rstrip(), ""]
    plines += ["== Numerical validation (Section 12.3)", "\n".join(lines).rstrip(), ""]
    plines += ["== Failures (Section 12.3)", "\n".join(flines).rstrip(), ""]
    plines += ["== The record's new entries (Section 12.3)", "\n".join(rlines).rstrip(), ""]
    with open(os.path.join(RESULTS, "portability-report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(plines) + "\n")


# ----------------------------------------------------------------------------
# A reproduction against the shipped results
# ----------------------------------------------------------------------------
SINGLE = ("newton-mujoco-c", "newton-mujoco-warp", "newton-mujoco-c-hull64",
          "newton-mujoco-c-gap", "newton-xpbd", "newton-featherstone", "newton-semi-implicit",
          "genesis-defaults")


def hardware_line(path):
    """The line of a record that names the machine."""
    if not os.path.exists(path):
        return "no record"
    lines = open(path, encoding="utf-8").read().splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "hardware" and i + 1 < len(lines):
            return lines[i + 1].strip()
    return "not recorded"


def compare_files(mine, theirs, single=False):
    """How a rerun's per-period file differs from the shipped one: identical
    bytes; the values that differ at the stored precision; the largest
    difference of the error vectors in the norm of the state; and whether it
    lies within the acceptable variation for another machine."""
    with open(mine, "rb") as fh:
        a_bytes = fh.read()
    with open(theirs, "rb") as fh:
        b_bytes = fh.read()
    a, b = read_rows(mine), read_rows(theirs)
    differ = sum(a[k][c] != b[k][c] for k in a for c in a[k] if c in b.get(k, {}))
    largest, at = 0.0, None
    for k in sorted(set(a) & set(b)):
        d = norm(*(a[k][c] - b[k][c] for c in ("dp", "dv", "dl")))
        if d > largest:
            largest, at = d, k
    if single:
        p = max(abs(a[k].get("p1", a[k]["p"])) for k in a) or 1.0
        bound = 2 * spacing(p)
    else:
        bound = 1e-9
    return (a_bytes == b_bytes, differ, largest, at, largest <= bound, bound)


NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def last_digit(token):
    """One unit of the last digit of a quoted number."""
    mantissa, _, exponent = token.lower().partition("e")
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    return 10.0 ** (int(exponent or 0) - decimals)


def within_last_digit(a, b):
    """Whether two quoted values have the same words and differ, number by
    number, by at most one unit of the last digit of the shipped one."""
    if NUMBER.sub("#", a) != NUMBER.sub("#", b):
        return False
    return all(x == y or abs(float(x) - float(y)) <= last_digit(y) * (1 + 1e-9)
               for x, y in zip(NUMBER.findall(a), NUMBER.findall(b)))


def compare_summaries(mine, theirs):
    """The values of two summaries that differ, times and versions aside:
    (group and key, the rerun's value, the shipped value, whether they lie
    within one unit of the last digit)."""
    a, b = read_summary(mine), read_summary(theirs)
    out = []
    for group in sorted(set(a) | set(b)):
        if group == "check":
            continue
        for key in sorted(set(a.get(group, {})) | set(b.get(group, {}))):
            if key.startswith(VOLATILE):
                continue
            value, other = a.get(group, {}).get(key), b.get(group, {}).get(key)
            if value != other:
                out.append((f"{group}.{key}", value, other, value is not None
                            and other is not None and within_last_digit(value, other)))
    return out


def compare_records(shipped, rerun):
    """The lines of two records that differ, as (shipped, rerun) pairs."""
    import difflib
    if not (os.path.exists(shipped) and os.path.exists(rerun)):
        return None
    with open(shipped, encoding="utf-8") as fh:
        a = fh.read().splitlines()
    with open(rerun, encoding="utf-8") as fh:
        b = fh.read().splitlines()
    return [(" / ".join(x.strip() for x in a[i1:i2]), " / ".join(x.strip() for x in b[j1:j2]))
            for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes()
            if tag != "equal"]


def compare_with(args):
    root = os.path.abspath(args.compare_with)
    where = "/".join(root.replace(os.sep, "/").split("/")[-2:])
    lines = [f"A reproduction against the shipped results: the rerun in .../{where}",
             f"acceptable variation on another machine or operating system: {VARIATION[2][1]}",
             ""]
    items = [("mujoco", os.path.join(CH11, "results"), os.path.join(root, "ch11", "results"),
              "agreement-chapter3.csv", "mujoco-summary.txt", "record.txt")]
    for name in sorted(os.listdir(os.path.join(root, "ch12", "results"))):
        if os.path.exists(os.path.join(root, "ch12", "results", name, "summary.txt")):
            items.append((name, os.path.join(RESULTS, name),
                          os.path.join(root, "ch12", "results", name),
                          "agreement.csv", "summary.txt", "record.txt"))
    for name, shipped, rerun, rows, summary, record in items:
        lines.append(f"[{name}]")
        lines.append(f"  hardware of the shipped run = {hardware_line(os.path.join(shipped, record))}")
        lines.append(f"  hardware of the rerun = {hardware_line(os.path.join(rerun, record))}")
        changed = compare_records(os.path.join(shipped, record), os.path.join(rerun, record))
        if changed is None:
            lines.append("  record = not in both")
        else:
            lines.append(f"  lines of the record that differ = {len(changed)}" + "".join(
                f"; {a} -> {b}" for a, b in changed))
        within = True
        if os.path.exists(os.path.join(rerun, rows)) and os.path.exists(os.path.join(shipped, rows)):
            same, differ, largest, at, within, bound = compare_files(
                os.path.join(shipped, rows), os.path.join(rerun, rows), name in SINGLE)
            lines.append(f"  per-period file identical, byte for byte = {'yes' if same else 'no'}")
            lines.append(f"  values that differ at the stored precision = {differ}")
            lines.append(f"  largest difference of an error vector, norm of the state [m], "
                         f"period = {g3(largest)}, {at}")
            lines.append(f"  error vectors within {g3(bound)} m = {'yes' if within else 'no'}")
        else:
            lines.append("  per-period file = none in the shipped results or the rerun")
        diffs = compare_summaries(os.path.join(rerun, summary), os.path.join(shipped, summary))
        beyond = [d for d in diffs if not d[3]]
        lines.append(f"  values of the summary that differ, times and versions aside = "
                     f"{len(diffs)}; beyond one unit of their last digit = {len(beyond)}" + "".join(
                         f"; {key}: {mine!r} against {theirs!r}"
                         for key, mine, theirs, _ in diffs[:5]))
        lines.append(f"  within the acceptable variation for another machine = "
                     f"{'yes' if within and not beyond else 'no'}")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "reproduction.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("part", nargs="?",
                    choices=("mujoco", "newton", "genesis", "genesis-defaults", "usd"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--warp", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--compare-with", metavar="CODE",
                    help="the code folder of a rerun, with ch11/results and ch12/results")
    args = ap.parse_args()
    if args.compare_with:
        compare_with(args)
    elif args.report:
        report(args)
    elif args.part == "mujoco":
        part_mujoco(args)
    elif args.part == "newton":
        part_newton(args)
    elif args.part == "genesis":
        part_genesis(args)
    elif args.part == "genesis-defaults":
        part_genesis_defaults(args)
    elif args.part == "usd":
        part_usd(args)
    else:
        ap.error("give a part or --report")


if __name__ == "__main__":
    main()
