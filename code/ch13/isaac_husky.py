#!/usr/bin/env python3
"""
The Husky A300 of Section 13.1 in Isaac Sim: the same description, drive
controller, drivers and grounds, and the diagnostics of Section 13.1 that
apply to PhysX.

Physical AI, Together, Chapter 13 (Section 13.1).

Version 5.

Versions 1 to 3. Version 2 corrected four faults of version 1 (the loads
and contacts at rest read from a single step, PhysX's wrap of the wheels'
positions at +-2 pi, the wheels' inertias under the import's principal
axes, the deprecated names of the contact getters). The runs of versions 1
and 2 (27 October 2026, Isaac Sim 6.2.0-rc.14 on an RTX 4060 Laptop GPU,
Windows 11) were not valid: their wheels were cylinders and their ground a
box 80 m wide and 0.1 m thick, which PhysX meets by GJK and EPA with a
third contact point beside the ends of the cylinder's line of contact
(contactConvexCoreConvex, GuContactConvexCoreConvex.cpp, in the PhysX SDK,
release 5.12.0), and the robot hopped (its base up to 67 mm above the
height of touching wheels at 1 m/s). Version 3 made the ground a USD Plane,
which PhysX meets in closed form at the two ends of that line
(contactPlaneConvexCore), screened setups that change one thing at a time,
and judged every diagnostic by the base's envelope. Its run was valid but
for a settling check at 0.25 ms that judged the base's velocities; but at
1 m/s and 1/3 rad/s the robot turned with a radius of 27 to 32 m under TGS,
with the rims at their targets, where Coulomb's friction gives 3.1 to 3.3 m
and MuJoCo's turn (results/mujoco-summary.txt) is within 2 % of its
reference; under PGS it went almost straight. Users of the Husky and Jackal
in Isaac Sim report the same (NVIDIA developer forum, "Wheeled robot
incorrect behavior" and "Skid-Steered behavior for robots", 2023).

Version 4 read PhysX's friction at its anchors (get_friction_data) and
judged it against the slip there, from PhysX's link velocities; it read
the joints' torques that PhysX applies, judged the settling by the base's
motion, repeated the settling, the spin, the turn and the stop at 1, 0.5
and 0.25 ms, and screened five more setups (time steps of 1 ms and 1/600 s
under TGS and PGS, a friction correlation distance of 0.01 mm, and
compliant contacts). Its run (27 October 2026, the same computer):
  * was valid: 14 of its 16 setups, all but the two with cylinders on the
    box, and every diagnostic of its full runs; its check of the friction
    data read +0.999 to +1.004 on the plane: they are the forces on the
    wheels;
  * turned the robot at 1 m/s and 1/3 rad/s with a radius of 24.5 to 47.1 m
    in every setup under TGS with round wheels on the plane, 7.6 to 10.3
    times its Coulomb reference, whatever the time step (0.25 to 1.67 ms),
    the velocity iterations (1 or 4), the correlation distance or the
    compliance of the contacts, and with spheres as with cylinders; under
    PGS the robot went almost straight (534 m with spheres, over 5000 m
    with cylinders);
  * showed what the friction does in that turn: the rims held their
    targets; every anchor slipped, with |F| / (mu N) of median 1.00; PhysX
    kept no anchor from the step before, so the persistence of anchors
    that version 4 suspected is not the cause; and the friction stood 42
    to 44 deg (median) from the opposite of the slip, at every time step
    and on every ground. In the spins with the rims at 0.98 m/s it stood
    0.7 to 3.6 deg from it, and the yaw rate came within 4.4 % of
    Coulomb's (7.7 % at 0.25 ms on asphalt). With the rims at 0.49 m/s,
    the yaw rate came within 8 % of Coulomb's with spheres, and with
    cylinders on gravel and wet grass; with cylinders on asphalt the
    friction stood 42 deg from the opposite of the slip, and the robot spun
    at 0.16 rad/s, 82 % short.
In the PhysX SDK (release 5.12.0, and 5.9.1 as tagged for Kit 110.3), the
solver resolves the friction of a patch of contact along two directions:
t0, along the bodies' relative linear velocity, which for a wheel on the
ground is the velocity of its centre, not that of the contact point
(setupFinalizeExtSolverContactsStep, DyTGSContactPrep.cpp; the same in
DyArticulationContactPrep.cpp for PGS), and t1 across it. Each direction
has its own velocity multiplier: a coefficient common to both, divided by
the contact's response along that direction with the articulation link's
cfm added (cfmScale, 0.025 by default). Under TGS the two impulses of an
anchor are cut together to a circle, of radius mu N for the patch, shared
by its anchors (half each where there are two; solveExtContactStep);
under PGS each is clamped on its own (solveExtContact,
DySolverConstraints.cpp). Where an anchor slips, the friction is then
anti-parallel to D s, where s is the slip and D = diag(1/W_00, 1/W_11) in
the frame (t0, t1), W being the contact point's inverse effective mass
(W_00 + cfm and W_11 + cfm with the cfm); Coulomb's friction is
anti-parallel to s.
At the foot of a wheel of this robot (the base free, the wheels free to
spin, the description's masses and inertias) W gives an effective mass of
1.2 kg along the wheel and 20 kg across it: a sideways slip weighs 16.5
times as much as a forward one. This two-row law, with nothing fitted,
predicts what version 4's traces show: the friction's direction (median
errors of 4 to 12 deg in the turns and below 1 deg in the spin at 0.49 m/s
on asphalt, where Coulomb's are 38 to 47 deg and 42 deg; with t0 along the
wheel's own axis instead, the spins at 0.98 m/s would miss by about
40 deg), a turn of 25 to 26 m (PhysX: 27 to 32 m), and two kinds of steady
spin with the rims at 0.49 m/s, near Coulomb's 0.9 rad/s and at 0.1 to
0.2 rad/s, both of which PhysX reached (the low one on asphalt). With the
link's cfm added to W, as the source reads, the law fits those traces less
well (2.8 deg against 0.5 deg in that spin); why is not settled here.

Version 5 therefore
  * screens three more setups: the default with the wheels' inertia about
    their axles 2, 8 and 32 times the description's. That raises the
    contact's effective mass along the wheel and leaves it across the
    wheel, so that the two-row law predicts turns of about 14, 5.4 and
    3.2 m; at 32 times the two effective masses are about equal, and the
    law becomes Coulomb's. These setups test the law; they are not models
    of the robot, and are never added to the full runs;
  * gives, under TGS, beside every Coulomb reference of the spins and the
    turn, the steady state of the two-row law for the same contacts and
    rims, quasi-static as the reference (from PhysX's motion, and from
    Coulomb's where that leads to another), without the link's cfm and
    with it; and, beside the angle between PhysX's friction and the
    opposite of the slip, the angle between PhysX's friction and the
    law's, per anchor and step;
  * reports the base's drift, its mean forward and sideways speeds, in the
    spins;
  * writes what it prints to console.txt in the results;
  * reads PhysX's friction with get_friction_contact_data, the name that
    omni.physics.tensors 110.4.4 asks for in place of get_friction_data
    (its DeprecationWarning in version 4's run), once the two have returned
    the same arrays at a step with friction; until then, and where the new
    name is missing, fails or returns other data, with get_friction_data,
    which version 4 read and checked (the summary's validity group says
    which, and why);
  * gives the importer a URDF with absolute mesh paths in a temporary
    folder, removed after the import, and keeps in the results a copy
    whose mesh paths are relative to the description's package
    (husky/description/clearpath_platform_description), so that the
    results hold no path of the computer that ran them.

It
  * imports the corrected URDF of husky/asset.py with Isaac Sim's URDF
    importer (URDFImporter), fixed joints merged, the base free; the
    importer leaves absolute mesh paths as they are (it writes such paths
    itself when it relocates a URDF: changelog of
    isaacsim.asset.importer.urdf, 3.9.0);
  * opens the imported asset for every run, selects its PhysX variant and
    adds, in the session layer: a physics scene stepping at the time step
    with the settings of Chapter 11 (TGS unless the setup says PGS, CPU
    dynamics, MBP broadphase, patch friction, no CCD, no stabilization);
    the articulation's settings of Chapter 11 (no
    sleeping, no stabilization threshold, 32 position iterations and the
    setup's velocity iterations, no self-collision) and no damping of the
    links; the ground of the setup with a physics material of the ground's
    friction (static and dynamic, restitution 0), bound to the ground and to
    the wheels' colliders, contact offset of the setup (0.02 m as in Chapter
    11) and rest offset 0; contact reports on the wheels' links, which
    PhysX's contact views need; the wheels' angular drives as velocity
    drives (stiffness 0, damping the drivers' proportional gain of
    40 N m s/rad, written per degree as USD reads it, maximum force 30 N m);
    and the description's masses and inertias (the wheels' inertia about
    the axle scaled where the setup says);
  * records what the importer and PhysX made of the description;
  * screens the setups (--screen) on asphalt: the sign of the friction
    data, at rest, at 1 m/s, the spin with the rims at 0.98 m/s, the turn
    at 1 m/s and 1/3 rad/s, the sideways slide and the stop;
  * runs in full, for the setups of --full and those the screen adds (every
    screened setup that is valid, keeps the description's inertias and
    turned within 25 % of a Coulomb reference without residual; not with
    --no-auto), on asphalt, gravel and wet grass (friction 0.8, 0.6, 0.35),
    the diagnostics of Section 13.1 with the A300's drive controller
    (husky/drive.py) and the drivers' integral loop of husky/sim.py: the
    loads and contacts at rest, averaged over 0.5 s, with their first
    moments beside the weight's; the spin with the rims held at 0.49 and
    0.98 m/s, and the turn commanded at 1 m/s and 1/3 rad/s, against the
    Coulomb steady state for those contacts; the drive at 1 m/s; a sideways
    slide from 1 m/s and a forward one with the wheels held, against
    v^2 / (2 mu g); the creep under a sideways push of 30 % of mu m g at the
    centre of mass, as MuJoCo's xfrc_applied pushes; the stop from a
    commanded 1 m/s; the spin commanded at 1 rad/s; and, on asphalt, the
    speed and the odometry over 10 s;
  * repeats the settling, the spin with the rims at 0.98 m/s, the turn and
    the stop at time steps of 1, 0.5 and 0.25 ms on every ground, for the
    first setup run in full (not with --quick), as husky_mujoco.py repeats
    the stop, the spin and the turn, and reports what a stop that does not
    come to rest still does.
Every diagnostic reports the base's envelope and largest tilt and is judged
valid, marginal or INVALID: in motion by the base's excursion from its
height at rest (valid within 2 mm and 1 deg, marginal within 10 mm and
3 deg), at rest by the base's height relative to touching round wheels, the
spread of that height and the tilt (valid within 1 mm, 0.5 mm and 0.2 deg,
marginal within 5 mm, 2 mm and 1 deg); a pose that is not finite, or a robot
that did not settle within 3 s, is INVALID. Version 2's options (--sweep,
--wheels, --contact-offset, --velocity-iterations, --dt,
--external-forces-every-iteration, --compliant), which the setups replace,
stop the script with an error.

Every step runs on its own: a step that fails is reported in errors.txt
with its traceback, and the others go on.

Usage, with the Python of Isaac Sim 6.2.0 (about 75 minutes with the
defaults on the computer of the earlier runs; version 4's took 70):
    python -u isaac_husky.py                  (pip install of isaacsim)
    <isaac-sim>/python.sh isaac_husky.py      (Linux install)
    <isaac-sim>\\python.bat isaac_husky.py     (Windows install)
Options:
    --list                 print the setups and stop
    --screen NAMES         the setups to screen, comma-separated, or all
                           (the default) or none
    --full NAMES           the setups to run in full, comma-separated, or
                           none (default: cylinder-plane-tgs,
                           cylinder-plane-pgs, sphere-plane-tgs)
    --force-full           run them in full even where the screen found them
                           INVALID
    --no-auto              do not add to the full runs the screened setups
                           that turn within 25 % of Coulomb
    --grounds asphalt      the grounds of the full runs and the time steps
                           (comma-separated)
    --quick                skip the time steps of 1 and 0.25 ms
    --tag NAME             write to results/isaac-NAME (default results/isaac)

Outputs, in results/isaac/ (or results/isaac-NAME/) next to this file:
    isaac-summary.txt     every value, by group, with the envelopes
    record.txt            the record of the experiment
    console.txt           what the script printed, and Python's warnings
    husky_a300.urdf       the URDF that the importer read, its mesh paths
                          relative to the description's package
    usd/                  the importer's USD
    stage-<setup>-<ground>.usda    the session layer that the script added
    trace-screen-<setup>-drive.csv single steps at 1 m/s in the screen
    trace-<setup>-<ground>-<what>.csv  single steps of the full runs: at
                          rest, at 1 m/s, in the slides and in the stop, and
                          the last 0.5 s of the spins (spin-0.49, spin-0.98)
                          and of the turn, with each wheel's friction, slip,
                          joint torque, anchors and the oldest anchor's age
    trace-<setup>-<ground>-stop-<dt>ms.csv  the stops at the time steps
    errors.txt            the steps that failed, with tracebacks
In isaac-summary.txt, record.txt, console.txt and errors.txt, the folder of
this file is written '.' and the user's home folder '~'.
"""
import argparse
import csv
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from collections import deque
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
VERSION = 5


# ----------------------------------------------------------------------------
# The setups: what touches what, and how PhysX solves it
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Setup:
    name: str
    wheels: str = "cylinder"        # cylinder | sphere | convex (the cylinder as a convex mesh)
    ground: str = "plane"           # plane | box (80 m wide, 0.1 m thick: version 2's ground)
    solver: str = "TGS"             # TGS | PGS
    contact_offset: float = 0.02    # m, of the wheels' colliders and of the ground
    velocity_iterations: int = 1
    dt: float = 0.0005              # s
    correlation: float = 0.0        # m, physxScene:frictionCorrelationDistance; 0: not authored (0.025)
    compliant: tuple = ()           # (stiffness [N/m], damping [N s/m]) of compliant contacts; (): rigid
    wheel_inertia: float = 1.0      # the wheels' inertia about their axles, in multiples of the description's
    note: str = ""

    def text(self):
        extra = ""
        if self.correlation:
            extra += f", friction correlation distance {self.correlation * 1e3:g} mm"
        if self.compliant:
            extra += (f", compliant contacts of stiffness {self.compliant[0]:g} N/m and damping "
                      f"{self.compliant[1]:g} N s/m")
        if self.wheel_inertia != 1.0:
            extra += f", the wheels' inertia about their axles {self.wheel_inertia:g} times the description's"
        return (f"wheels' colliders {self.wheels}, ground {self.ground}, solver {self.solver}, "
                f"contact offset {self.contact_offset:g} m, velocity iterations {self.velocity_iterations}, "
                f"time step {self.dt * 1e3:.4g} ms" + extra + (f" ({self.note})" if self.note else ""))


FORUM = "NVIDIA developer forum, 'Wheeled robot incorrect behavior', 2023"
INERTIA_NOTE = ("it raises the contact's effective mass along the wheel and leaves it across: a test of the "
                "two-row law of PhysX's friction, not a model of the robot; never added to the full runs")
SETUPS = (
    Setup("cylinder-box-tgs", ground="box", note="version 2's setup"),
    Setup("cylinder-plane-tgs", note="the default"),
    Setup("sphere-plane-tgs", wheels="sphere"),
    Setup("convex-plane-tgs", wheels="convex"),
    Setup("sphere-box-tgs", wheels="sphere", ground="box"),
    Setup("convex-box-tgs", wheels="convex", ground="box"),
    Setup("cylinder-box-tgs-offset-2mm", ground="box", contact_offset=0.002),
    Setup("cylinder-plane-pgs", solver="PGS"),
    Setup("sphere-plane-pgs", wheels="sphere", solver="PGS"),
    Setup("cylinder-plane-tgs-4-velocity-iterations", velocity_iterations=4),
    Setup("cylinder-plane-tgs-dt-0.25ms", dt=0.00025),
    Setup("cylinder-plane-tgs-dt-1ms", dt=0.001),
    Setup("cylinder-plane-tgs-600hz", dt=1.0 / 600.0, note="the time step of a reply in the " + FORUM),
    Setup("cylinder-plane-pgs-600hz", solver="PGS", dt=1.0 / 600.0, note="the same, with PGS"),
    Setup("cylinder-plane-tgs-correlation-0.01mm", correlation=1e-5,
          note="friction anchors kept from step to step only while they part by less than 0.01 mm along "
               "the normal, which on a rolling wheel drops them after a turn of about 0.011 rad, where "
               "PhysX's test of the normals drops them after 0.045 rad"),
    Setup("cylinder-plane-tgs-compliant", compliant=(4e5, 4e3),
          note="at each of the eight points, a penetration at rest of 0.25 mm (MuJoCo's: 0.23 mm), "
               "near critical damping"),
    Setup("cylinder-plane-tgs-wheel-inertia-x2", wheel_inertia=2.0, note=INERTIA_NOTE),
    Setup("cylinder-plane-tgs-wheel-inertia-x8", wheel_inertia=8.0, note=INERTIA_NOTE),
    Setup("cylinder-plane-tgs-wheel-inertia-x32", wheel_inertia=32.0,
          note=INERTIA_NOTE + "; at 32 times the effective masses along and across the wheel are about equal"),
)
BY_NAME = {s.name: s for s in SETUPS}
FULL_DEFAULT = ("cylinder-plane-tgs", "cylinder-plane-pgs", "sphere-plane-tgs")
TURN_CLOSE = 0.25       # a screened setup whose turn is within 25 % of its Coulomb reference is run in full too

parser = argparse.ArgumentParser(description="Section 13.1: the Husky A300 in Isaac Sim")
parser.add_argument("--list", action="store_true", help="print the setups and stop")
parser.add_argument("--screen", default="all", help="setups to screen on asphalt: names, all or none")
parser.add_argument("--full", default=",".join(FULL_DEFAULT), help="setups to run in full: names or none")
parser.add_argument("--force-full", action="store_true",
                    help="run the setups of --full even where the screen found them invalid")
parser.add_argument("--grounds", default="asphalt,gravel,wet grass",
                    help="grounds of the full runs, comma-separated: asphalt, gravel, wet grass")
parser.add_argument("--quick", action="store_true", help="skip the time steps of 1 and 0.25 ms")
parser.add_argument("--no-auto", action="store_true",
                    help="do not add to the full runs the screened setups that turn within 25 %% of Coulomb")
parser.add_argument("--tag", default=None, help="write to results/isaac-NAME")
args, _unknown = parser.parse_known_args()
REMOVED = ("--sweep", "--wheels", "--contact-offset", "--velocity-iterations", "--dt",
           "--external-forces-every-iteration", "--compliant")
_removed = [a for a in _unknown if a.split("=")[0] in REMOVED]
if _removed:
    parser.error(f"{', '.join(_removed)}: options of version 2, which the setups replace "
                 "(--list prints them; --screen and --full choose them)")


def pick(text, option):
    key = text.strip().lower()
    if key == "none":
        return []
    if key == "all":
        return list(SETUPS)
    names = [n.strip() for n in text.split(",") if n.strip()]
    bad = [n for n in names if n not in BY_NAME]
    if bad:
        parser.error(f"--{option}: unknown setups {bad}; --list prints them")
    return [BY_NAME[n] for n in dict.fromkeys(names)]


if args.list:
    for s in SETUPS:
        print(f"{s.name:42s} {s.text()}")
    sys.exit(0)
SCREEN = pick(args.screen, "screen")
FULL = pick(args.full, "full")
OUT = os.path.join(HERE, "results", "isaac" + (f"-{args.tag}" if args.tag else ""))

from isaacsim import SimulationApp                                    # noqa: E402

APP = SimulationApp({"headless": True})

import numpy as np                                                    # noqa: E402
import omni.kit.app                                                   # noqa: E402
import omni.timeline                                                  # noqa: E402
import omni.usd                                                       # noqa: E402
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade         # noqa: E402

sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
from husky import asset, coulomb, drive, world                       # noqa: E402  (no MuJoCo)

# Isaac Sim 6.2 deprecates the contact getters' old names; the values they
# return are those that version 1 read (choose_friction_reader compares the
# friction getter's new name with the old one before it uses it).
warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*contact.*deprecated.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=r"get_\w+\(\) is deprecated")

APPROXIMATE_CYLINDERS = "/physics/collisionApproximateCylinders"
G = 9.81
KV = 40.0              # N m s/rad, the drivers' proportional gain (husky/model.py)
TI = 0.1               # s, the drivers' integral time
TORQUE = 30.0          # N m
RADIUS = 0.1651        # m, the wheel's radius in the description
WHEEL_Z = 0.02913      # m, the wheels' centres above base_link
MASS = 80.0            # kg, the description's
PLACE_DEPTH = 1e-4     # m
WINDOW = 0.5           # s, the steps over which loads and contacts are averaged at rest
TWO_PI = 2.0 * math.pi
WHEELS = ("front_left", "rear_left", "front_right", "rear_right")
LEFT, RIGHT = WHEELS[:2], WHEELS[2:]
CENTRES = {"front_left": (0.256, 0.2829), "rear_left": (-0.256, 0.2829),
           "front_right": (0.256, -0.2829), "rear_right": (-0.256, -0.2829)}
SIDES = ("inner", "outer")
# The validity of a diagnostic, from the base's envelope: (valid up to,
# marginal up to); beyond, INVALID. At rest, the base's height relative to the
# height of touching round wheels (a polygonal wheel, a convex mesh, rests lower
# by R (1 - cos(pi / n)), 0.8 mm for 32 sides), the spread of its height and
# its tilt; in motion, the base's excursion from its height at rest (the median
# of the setup's last window at rest) and its tilt.
REST_LIMITS = dict(offset=(1.0, 5.0), spread=(0.5, 2.0), tilt=(0.2, 1.0))     # mm, mm, deg
MOTION_LIMITS = dict(excursion=(2.0, 10.0), tilt=(1.0, 3.0))                 # mm, deg
LEVELS = ("valid", "marginal", "INVALID")
SETTLE_MIN, SETTLE_MAX, SETTLE_HOLD = 1.0, 3.0, 0.1   # s
# Steady, over SETTLE_HOLD: the base's height changed by less than 0.05 mm, its
# roll and pitch by less than 0.02 deg and its place on the ground by less
# than 0.1 mm (a tenth of the limits at rest for the spread and the tilt, a
# tenth of the valid offset for the place), whatever the time step.
STEADY = dict(height=0.05, tilt=0.02, drift=0.1)      # mm, deg, mm
SLIP_MIN = 1e-3        # m/s, the slip below which a contact's friction has no direction to be judged against
TRACE_TAIL = 0.5       # s, the last part of a window of the spins and the turn written as a trace
SCREEN_GROUND = ("asphalt", 0.8)
WANTED = [g.strip() for g in args.grounds.split(",") if g.strip()]
_unknown_grounds = [g for g in WANTED if g not in [x.name for x in world.GROUNDS]]
if _unknown_grounds:
    APP.close()
    raise SystemExit(f"unknown grounds: {_unknown_grounds}; the grounds are "
                     f"{', '.join(x.name for x in world.GROUNDS)}")
GROUNDS = [(g.name, g.friction) for g in world.GROUNDS if g.name in WANTED]
MODEL_GROUP = "what the importer and PhysX made of the description"
SCREEN_GROUP = f"screen on {SCREEN_GROUND[0]}, mu {SCREEN_GROUND[1]:g}"
FACTS, ERRORS, INVALID, VERDICTS, RAN = {}, [], [], {}, []
TURNS = {}             # setup: the screen's turn radius against its Coulomb reference, less 1
STARTED = time.time()


CONSOLE = []            # the open console.txt, once main() has opened it


def scrub(text):
    """Text for the results, with no path of the computer that ran the
    script: this file's folder becomes '.', the user's home folder '~'
    (either slash, any case, as Windows and Kit write them)."""
    import re
    for path, mark in ((HERE, "."), (os.path.expanduser("~"), "~")):
        parts = [p for p in re.split(r"[\\/]+", path or "") if p]
        if len(parts) < 2:
            continue
        lead = r"[\\/]*" if path[:1] in ("/", "\\") else ""
        pattern = lead + r"[\\/]+".join(re.escape(p) for p in parts)
        text = re.sub(pattern, lambda _m, mark=mark: mark, text, flags=re.IGNORECASE)
    return text


def to_console(line):
    for fh in CONSOLE:
        try:
            fh.write(scrub(line) + "\n")
            fh.flush()
        except (OSError, ValueError):
            pass


def say(text):
    line = f"[{time.time() - STARTED:7.1f} s] {text}"
    print(line, flush=True)
    to_console(line)


_show_warning = warnings.showwarning


def show_warning(message, category, filename, lineno, file=None, line=None):
    """Python's warnings, as Python shows them, and in console.txt."""
    to_console("warning: " + warnings.formatwarning(message, category, os.path.basename(filename), lineno,
                                                     line).strip())
    _show_warning(message, category, filename, lineno, file, line)


warnings.showwarning = show_warning


def pct(x, digits=2):
    return f"{100.0 * x:+.{digits}f} %"


def share(x):
    return f"{100.0 * x:.0f}%"


ONCE = set()


def report_once(key, exc):
    """A reader that fails, in errors.txt and on the console once per run
    (every stage tries it again)."""
    if key in ONCE:
        return
    ONCE.add(key)
    ERRORS.append((key, traceback.format_exc()))
    say(f"  {key}: {exc!r}")


def attempt(out, key, fn):
    try:
        out[key] = fn()
    except Exception as exc:                       # noqa: BLE001 (report and go on)
        out[key] = f"FAILED: {exc!r}"
        ERRORS.append((key, traceback.format_exc()))
        say(f"  failed: {key}: {exc!r}")


def group(name, fn):
    say(name)
    out = FACTS.setdefault(name, {})
    try:
        fn(out)
    except Exception as exc:                       # noqa: BLE001
        out["FAILED"] = repr(exc)
        ERRORS.append((name, traceback.format_exc()))
        say(f"  failed: {name}: {exc!r}")
    write_outputs()


def write_outputs():
    """The summary and the errors, rewritten after every group; a file that
    cannot be written (one held open by another program, say) is reported on
    the console and written again after the next group."""
    os.makedirs(OUT, exist_ok=True)
    lines = []
    for grp, items in FACTS.items():
        lines.append(f"[{grp}]")
        for k, v in items.items():
            lines.append(f"  {k} = {v}")
    for name, text in (("isaac-summary.txt", "\n".join(lines) + "\n"),
                       ("errors.txt", "\n\n".join(f"== {k}\n{tb}" for k, tb in ERRORS) or "none\n")):
        try:
            with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
                fh.write(scrub(text))
        except OSError as exc:
            say(f"  could not write {name}: {exc!r}")


def write_csv(name, header, rows):
    """A trace; one that cannot be written is reported in errors.txt
    without failing the diagnostic that made it."""
    os.makedirs(OUT, exist_ok=True)
    try:
        with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as fh:
            out = csv.writer(fh)
            out.writerow(header)
            for row in rows:
                out.writerow([f"{x:.6g}" if isinstance(x, float) else x for x in row])
    except OSError:
        ERRORS.append((f"writing {name}", traceback.format_exc()))
        say(f"  could not write {name}")
    return name


def slug(name):
    return name.replace(" ", "-")


def lag1(xs):
    """The correlation of successive values: near -1 when they alternate;
    NaN for a constant series."""
    x = np.asarray(xs, dtype=float)
    floor = 1e-9 * (1.0 + float(np.abs(x).mean())) if len(x) else 0.0
    if len(x) < 3 or float(np.std(x[:-1])) <= floor or float(np.std(x[1:])) <= floor:
        return float("nan")
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def corr(x):
    return "-" if math.isnan(x) else f"{x:+.2f}"


# ----------------------------------------------------------------------------
# Stepping (as in code/ch11/cart_isaac/sim.py, for Isaac Sim 6.x)
# ----------------------------------------------------------------------------
class Physics:
    def __init__(self):
        from isaacsim.core.simulation_manager import SimulationManager
        self.sm = SimulationManager
        self.mode, self.scene_dt, self.dt, self._t = None, None, None, 0.0
        self.own_view = False

    def configure(self, dt):
        self.dt = dt
        try:
            self.scene_dt = float(self.sm.get_physics_dt())
        except Exception:                          # noqa: BLE001
            self.scene_dt = None
        if self.scene_dt is not None and abs(self.scene_dt - dt) < 1e-12:
            self.mode = "simulation manager"
            return
        self.mode = "physics interface"
        import omni.physics.core as core
        self._direct = core.get_physics_simulation_interface()

    def use_cpu(self):
        if hasattr(self.sm, "set_device"):
            self.sm.set_device("cpu")
        else:
            self.sm.set_physics_sim_device("cpu")

    def step(self):
        if self.mode == "simulation manager":
            self.sm.step(steps=1)
            return
        self._direct.simulate(self.dt, self._t)
        self._t += self.dt

    def view(self):
        view = None
        try:
            view = self.sm.get_physics_sim_view()
        except Exception:                          # noqa: BLE001
            view = None
        if view is None or not getattr(view, "is_valid", True):
            import omni.physics.tensors as tensors
            stage_id = omni.usd.get_context().get_stage_id()
            try:
                view = tensors.create_simulation_view("numpy", stage_id=stage_id, backend="physx")
            except TypeError:
                view = tensors.create_simulation_view("numpy", stage_id=stage_id)
            view.set_subspace_roots("/")
            self.own_view = True
        return view


def normal_load(view, dt):
    """The normal force of a wheel's contact view [N]: the vertical component
    of the force matrix (a vector per sensor and filter), or its value if the
    getter returns magnitudes."""
    fn = getattr(view, "get_contact_force_matrix", None) or getattr(view, "get_normal_contact_force_matrix")
    a = np.asarray(fn(dt), dtype=float)
    if a.ndim >= 1 and a.shape[-1] == 3:
        return float(a.reshape(-1, 3)[:, 2].sum())
    return float(a.sum())


def contact_data(view, dt):
    """forces, points, normals, separations, counts and starts of a wheel's
    contact view (normals and separations NaN if they cannot be read)."""
    fn = getattr(view, "get_contact_data", None) or getattr(view, "get_normal_contact_data")
    data = fn(dt)
    forces = np.asarray(data[0], dtype=float).reshape(-1)
    points = np.asarray(data[1], dtype=float).reshape(-1, 3)
    try:
        normals = np.asarray(data[2], dtype=float).reshape(-1, 3)
        seps = np.asarray(data[3], dtype=float).reshape(-1)
        if len(normals) < len(points) or len(seps) < len(points):
            raise ValueError("short buffers")
    except Exception:                              # noqa: BLE001
        normals = np.full((len(points), 3), np.nan)
        seps = np.full(len(points), np.nan)
    return forces, points, normals, seps, np.asarray(data[4]).reshape(-1), np.asarray(data[5]).reshape(-1)


FRICTION_READER = {}    # the getter that friction_data uses, once chosen, and why (choose_friction_reader)
FRICTION_SAME_MIN = 1.0  # N, the largest friction of a step at which the two getters' arrays are compared


def friction_layout(data):
    """(forces, points, counts, starts) of a friction getter's result, if it
    has the layout of get_friction_data: forces and points of three
    components each, as many of one as of the other, and counts and start
    indices that stay within them; None otherwise."""
    try:
        forces = np.asarray(data[0], dtype=float).reshape(-1, 3)
        points = np.asarray(data[1], dtype=float).reshape(-1, 3)
        counts = np.asarray(data[2]).reshape(-1)
        starts = np.asarray(data[3]).reshape(-1)
    except Exception:                              # noqa: BLE001
        return None
    if len(forces) != len(points) or len(counts) != len(starts):
        return None
    if np.any(counts < 0) or np.any(starts < 0) or np.any(starts + counts > len(forces)):
        return None
    return forces, points, counts, starts


def friction_arrays(fn, dt, name):
    """(forces, points, counts, starts) of the friction getter fn, in the
    layout of get_friction_data; RuntimeError for another layout."""
    parsed = friction_layout(fn(dt))
    if parsed is None:
        raise RuntimeError(f"{name} returned another layout")
    return parsed


def read_getter(fn, dt, name):
    """friction_layout of what fn returns, with None; or None, with why,
    where fn fails or returns another layout."""
    try:
        parsed = friction_layout(fn(dt))
    except Exception as exc:                       # noqa: BLE001
        return None, f"{name} failed ({exc!r})"
    return parsed, (None if parsed is not None else f"{name} returned another layout")


def choose_friction_reader(view, dt):
    """This step's friction arrays, and the choice of the getter for the rest
    of the run. get_friction_data, which version 4 read and checked, gives
    them until get_friction_contact_data has returned the same arrays at a
    step whose largest friction is FRICTION_SAME_MIN or more (a check of the
    layout alone cannot tell forces from impulses, or the wheel's side of
    an anchor from the ground's); then get_friction_contact_data alone. The
    old name is kept for the run where the new one fails, or returns
    another layout or other values; the one that exists, where the other
    is missing; the new one where the old one fails."""
    new = getattr(view, "get_friction_contact_data", None)
    old = getattr(view, "get_friction_data", None)
    if new is None or old is None:
        if new is None and old is None:
            raise RuntimeError("omni.physics.tensors has neither get_friction_data nor get_friction_contact_data")
        name, missing = (("get_friction_data", "get_friction_contact_data") if new is None else
                         ("get_friction_contact_data", "get_friction_data"))
        FRICTION_READER.update(name=name, why=f"omni.physics.tensors has no {missing}")
        return friction_arrays(old or new, dt, name)
    FRICTION_READER["compared"] = True
    ref, why_old = read_getter(old, dt, "get_friction_data")
    if ref is None:
        other, why_new = read_getter(new, dt, "get_friction_contact_data")
        if other is None:
            raise RuntimeError(f"{why_old}; {why_new}")
        FRICTION_READER.update(name="get_friction_contact_data", why=why_old)
        return other
    ref = tuple(np.array(a, copy=True) for a in ref)
    other, why_new = read_getter(new, dt, "get_friction_contact_data")
    if other is None:
        FRICTION_READER.update(name="get_friction_data", why=why_new)
        return ref
    if (any(a.shape != b.shape for a, b in zip(ref, other)) or not np.array_equal(ref[2], other[2])
            or not np.array_equal(ref[3], other[3])):
        FRICTION_READER.update(name="get_friction_data",
                               why="get_friction_contact_data returned other counts or start indices")
        return ref
    big = float(np.nanmax(np.abs(ref[0]), initial=0.0))
    if not (np.allclose(other[0], ref[0], rtol=1e-5, atol=1e-6 * max(big, 1.0), equal_nan=True)
            and np.allclose(other[1], ref[1], rtol=0.0, atol=1e-7, equal_nan=True)):
        gaps = [float(np.nanmax(np.abs(other[k] - ref[k]), initial=0.0)) for k in (0, 1)]
        FRICTION_READER.update(name="get_friction_data", why=(
            f"get_friction_contact_data returned other values: forces up to {gaps[0]:.3g} N apart (the largest "
            f"{big:.3g} N), points up to {gaps[1]:.3g} m"))
        return ref
    if big >= FRICTION_SAME_MIN:
        FRICTION_READER.update(name="get_friction_contact_data", why=(
            f"it returned the same arrays as get_friction_data at a step with friction of up to {big:.3g} N"))
    return ref


def friction_data(view, dt):
    """The friction of a wheel's contact view in the last step, as PhysX
    reports it (choose_friction_reader: the impulse of each friction anchor
    over the time step, at the anchor's point, as omni.physx's CPU contact
    view writes it, CpuRigidContactView.cpp): a list of (force, point),
    world frame; entries within 0.01 mm of each other, which would be one
    anchor reported twice, are summed (PhysX keeps a patch's two anchors at
    least 0.1 mm apart, growPatches)."""
    name = FRICTION_READER.get("name")
    if name is None:
        forces, points, counts, starts = choose_friction_reader(view, dt)
    else:
        forces, points, counts, starts = friction_arrays(getattr(view, name), dt, name)
    anchors = []
    for k in range(len(counts)):
        for i in range(int(starts[k]), int(starts[k]) + int(counts[k])):
            for a in anchors:
                if float(np.linalg.norm(a[1] - points[i])) < 1e-5:
                    a[0] += forces[i]
                    break
            else:
                anchors.append([forces[i].copy(), points[i].copy()])
    return [(f, p) for f, p in anchors]


# ----------------------------------------------------------------------------
# The import
# ----------------------------------------------------------------------------
IMPORT = {}


def import_robot(out):
    """The importer reads the URDF with absolute mesh paths from a temporary
    folder, removed after the import (the USD it writes does not refer to
    the URDF; SimulationApp's fast shutdown ends the process without
    Python's exit handlers); the results keep a copy whose mesh paths are
    relative to the description's package."""
    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
    asset.write_urdf(os.path.join(OUT, "husky_a300.urdf"), meshes="package")
    before = sorted(os.listdir(os.path.join(OUT, "usd"))) if os.path.isdir(os.path.join(OUT, "usd")) else []
    folder = tempfile.mkdtemp(prefix="isaac_husky_")
    try:
        urdf = asset.write_urdf(os.path.join(folder, "husky_a300.urdf"), meshes="absolute")
        cfg = URDFImporterConfig()
        cfg.urdf_path = os.path.normpath(urdf)
        cfg.usd_path = os.path.normpath(os.path.join(OUT, "usd"))
        cfg.merge_fixed_joints = True
        cfg.fix_base = False
        cfg.collision_from_visuals = False
        cfg.allow_self_collision = False
        importer = URDFImporter(cfg)
        path = os.path.normpath(importer.import_urdf())
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    IMPORT["usd"] = path
    out["importer config"] = (f"merge_fixed_joints {cfg.merge_fixed_joints}, fix_base {cfg.fix_base}, "
                              f"collision_type {getattr(cfg, 'collision_type', '-')}, "
                              f"joint_drive_type {getattr(cfg, 'joint_drive_type', '-')}, "
                              f"joint_target_type {getattr(cfg, 'joint_target_type', '-')}")
    out["USD written"] = os.path.relpath(path, HERE)
    out["URDF in the results"] = ("husky_a300.urdf, mesh paths relative to husky/description/"
                                  "clearpath_platform_description (the importer read a copy with absolute paths)")
    if before:
        out["usd/ held before the import"] = ", ".join(before) + " (left by an earlier run)"


# ----------------------------------------------------------------------------
# The robot in a stage
# ----------------------------------------------------------------------------
def urdf_inertials():
    """The description's inertials: mass, centre of mass and the diagonal of
    the inertia in the link's frame (the description's inertial origins have
    no rotation and its products of inertia are zero, which is checked)."""
    import xml.etree.ElementTree as ET
    robot = ET.fromstring(asset.urdf_text())
    out = {}
    for link in robot.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        i = inertial.find("inertia").attrib
        o = inertial.find("origin")
        rpy = [float(x) for x in (o.get("rpy", "0 0 0").split() if o is not None else (0, 0, 0))]
        products = [float(i.get(k, 0.0)) for k in ("ixy", "ixz", "iyz")]
        if any(abs(x) > 1e-12 for x in rpy + products):
            raise RuntimeError(f"{link.get('name')}: a rotated inertial or products of inertia, "
                               "which this script does not write")
        out[link.get("name")] = dict(mass=float(inertial.find("mass").get("value")),
                                     com=[float(x) for x in (o.get("xyz", "0 0 0").split() if o is not None else (0, 0, 0))],
                                     diag=[float(i["ixx"]), float(i["iyy"]), float(i["izz"])])
    return out


def inertia_in_link_frame(prim):
    """The inertia tensor of a body as the stage composes it: the diagonal in
    the principal axes, turned into the prim's frame (USD's matrices act on
    row vectors, so the tensor is M^T D M)."""
    m = UsdPhysics.MassAPI(prim)
    d = m.GetDiagonalInertiaAttr().Get()
    q = m.GetPrincipalAxesAttr().Get()
    if d is None:
        return None
    if q is None:
        q = Gf.Quatf(1.0, 0.0, 0.0, 0.0)
    r = Gf.Matrix3d(Gf.Rotation(Gf.Quatd(q.GetReal(), Gf.Vec3d(q.GetImaginary()))).GetQuat())
    return r.GetTranspose() * Gf.Matrix3d(d[0], 0, 0, 0, d[1], 0, 0, 0, d[2]) * r


def all_prims(stage):
    return [p for p in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())]


def rotation(quat_xyzw):
    qx, qy, qz, qw = quat_xyzw
    return np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                     [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                     [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def roll_pitch(quat_xyzw):
    qx, qy, qz, qw = quat_xyzw
    roll = math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx))))
    return roll, pitch


class Envelope:
    """Over the steps of a diagnostic: the base's height relative to the
    height of touching round wheels (lowest, highest, and at rest the median)
    and its largest tilt, judged against REST_LIMITS (kind 'rest') or
    MOTION_LIMITS (kind 'motion', the excursion measured from ref, the base's
    height at rest [mm]); and the drivers' torque, its largest value and the
    share of the wheels' steps at the limit, both as PhysX reports the
    joints' torques (the joint forces projected on the axes) and from the
    drive law clip(KV (target - speed after the step), +-30 N m). flags (such
    as 'not settled', or a pose or a drive law's torque that is not finite)
    make it INVALID; PhysX's torques that are not finite are counted."""

    def __init__(self, kind="motion", ref=0.0):
        self.kind, self.ref = kind, ref
        self.lo, self.hi, self.tilt, self.steps = math.inf, -math.inf, 0.0, 0
        self.heights = []
        self.taus = {}          # source ('PhysX', 'drive law'): [wheel-steps, at the limit, largest, not finite]
        self.flags = []

    def flag(self, text):
        if text not in self.flags:
            self.flags.append(text)

    def add(self, dz_mm, tilt_deg):
        if not (math.isfinite(dz_mm) and math.isfinite(tilt_deg)):
            self.flag("a pose not finite")
            return
        self.lo = min(self.lo, dz_mm)
        self.hi = max(self.hi, dz_mm)
        self.tilt = max(self.tilt, tilt_deg)
        self.steps += 1
        if self.kind == "rest":
            self.heights.append(dz_mm)

    def torques(self, taus, source="drive law"):
        st = self.taus.setdefault(source, [0, 0, 0.0, 0])
        for t in taus:
            if not math.isfinite(t):
                if source == "PhysX":          # PhysX's reading is optional: counted, not judged
                    st[3] += 1
                else:
                    self.flag(f"a torque not finite ({source})")
                continue
            a = abs(t)
            st[0] += 1
            st[2] = max(st[2], a)
            if a >= 0.99 * TORQUE:
                st[1] += 1

    def median(self):
        return float(np.median(self.heights)) if self.heights else float("nan")

    def measures(self):
        """(value, (valid up to, marginal up to)) of each limit."""
        if self.kind == "rest":
            return [(abs(self.median()), REST_LIMITS["offset"]), (self.hi - self.lo, REST_LIMITS["spread"]),
                    (self.tilt, REST_LIMITS["tilt"])]
        return [(max(abs(self.lo - self.ref), abs(self.hi - self.ref)), MOTION_LIMITS["excursion"]),
                (self.tilt, MOTION_LIMITS["tilt"])]

    @property
    def level(self):
        """0 valid, 1 marginal, 2 INVALID."""
        if not self.steps or self.flags:
            return 2
        lvl = 0
        for value, (fine, gross) in self.measures():
            if not value <= gross:
                lvl = 2
            elif value > fine:
                lvl = max(lvl, 1)
        return lvl

    @property
    def valid(self):
        return self.level == 0

    def text(self):
        if not self.steps:
            return "no steps: INVALID" + "".join(f" ({f})" for f in self.flags)
        if self.kind == "rest":
            body = (f"base at rest {self.median():+.2f} mm from the height of touching round wheels, spread "
                    f"{self.hi - self.lo:.2f} mm, tilt {self.tilt:.2f} deg")
        else:
            body = (f"base {self.lo - self.ref:+.2f} to {self.hi - self.ref:+.2f} mm from its height at rest, "
                    f"tilt {self.tilt:.2f} deg")
        return body + ": " + LEVELS[self.level] + "".join(f" ({f})" for f in self.flags)

    def torque_text(self):
        """PhysX's joint torques where they were read, with the drive law's
        estimate beside them; otherwise the estimate."""
        def one(source):
            n, lim, big, bad = self.taus[source]
            return (f"{big:.1f} N m, at the limit in {share(lim / n)}" +
                    (f", {bad} values not finite" if bad else ""))
        physx = self.taus.get("PhysX", [0, 0, 0.0, 0])
        law = self.taus.get("drive law", [0, 0, 0.0, 0])
        if physx[0]:
            text = "largest " + one("PhysX") + " (PhysX's joint torques"
            if law[0]:
                text += "; the drive law's " + one("drive law")
            return text + ")"
        note = f"; PhysX's joint torques: none finite ({physx[3]})" if physx[3] else ""
        if law[0]:
            return "largest " + one("drive law") + " (the drive law's" + note + ")"
        return "-" + note


ENVELOPE_KEY = "the base's envelope and largest tilt: verdict"
LAW_KEY = ("the two-row law's steady motion nearest PhysX's (and any other it has), and with the links' cfm")


def judged(out_key, env, group_name):
    """Records a marginal or invalid envelope for the validity group."""
    if env is not None and env.level > 0:
        INVALID.append((group_name, out_key, env.text(), env.level))


class IsaacHusky:
    """The Husky on a ground of friction mu, in a setup, stepped at dt with
    the A300's controller and the drivers' integral loop."""

    def __init__(self, mu, setup, dt=None):
        self.mu, self.setup = mu, setup
        self.dt = setup.dt if dt is None else dt
        self.physics = Physics()
        self.drive = drive.A300Drive()
        self.steps_per_update = int(round(self.drive.period / self.dt))
        self.command, self.direct = (0.0, 0.0), None
        self.target = {w: 0.0 for w in WHEELS}
        self.z = {w: 0.0 for w in WHEELS}
        self.step_count = 0
        self.notes = {}
        self.q_raw = self.q_unwrapped = None
        self.wraps = 0
        self.push = 0.0
        self.env = None
        self._tgt_prev = None
        self.settled = None
        self.rest_dz = 0.0          # mm, the base's height at rest relative to touching round wheels
        self.friction_sign = None   # +1: PhysX's friction data are the forces on the wheels, -1: on the ground
        self.friction_read = None   # True, or False once PhysX's friction data could not be read
        self.joint_torques_read = None   # True, or False once PhysX's joint torques could not be read
        self.link_motion_read = None     # True, or False once PhysX's link velocities could not be read
        self.friction_scale = None       # |the friction check's ratio|
        self._anchors, self._anchors_step = None, None   # the anchors of the last step read, on the wheels
        self.last_taus = {w: float("nan") for w in WHEELS}
        self.centre_vel = None      # {wheel: (vx, vy)}, the wheels' centres in the base's frame, of the last friction()
        self._model = None
        self._approximate_before = None
        self._open()

    def friction_model(self):
        """The two-row law of PhysX's friction for this setup's masses."""
        if self._model is None:
            self._model = TwoRowModel(self.setup.wheel_inertia)
        return self._model

    # --- the stage ----------------------------------------------------------------
    def _set_approximation(self):
        """omni.physx's setting that approximates every cylinder by a convex
        mesh: on for the setups with convex wheels, off otherwise, before the
        stage is opened; close() restores the value found."""
        try:
            import carb
            settings = carb.settings.get_settings()
            self._approximate_before = settings.get(APPROXIMATE_CYLINDERS)
            settings.set_bool(APPROXIMATE_CYLINDERS, self.setup.wheels == "convex")
            self.notes[f"omni.physx {APPROXIMATE_CYLINDERS}"] = str(settings.get(APPROXIMATE_CYLINDERS))
        except Exception as exc:                   # noqa: BLE001
            if self.setup.wheels == "convex":
                raise
            self.notes[f"omni.physx {APPROXIMATE_CYLINDERS}"] = f"not read ({exc!r})"

    def _restore_approximation(self):
        if self._approximate_before is None:
            return
        try:
            import carb
            carb.settings.get_settings().set_bool(APPROXIMATE_CYLINDERS, bool(self._approximate_before))
        except Exception:                          # noqa: BLE001
            pass

    def _open(self):
        ctx = omni.usd.get_context()
        timeline = omni.timeline.get_timeline_interface()
        if not timeline.is_stopped():
            timeline.stop()
            APP.update()
        self._set_approximation()
        ctx.open_stage(IMPORT["usd"])
        APP.update()
        self.stage = stage = ctx.get_stage()
        stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))
        for prim in stage.GetPseudoRoot().GetChildren():
            vs = prim.GetVariantSets()
            if vs.HasVariantSet("Physics"):
                names = vs.GetVariantSet("Physics").GetVariantNames()
                if "physx" in names:
                    vs.GetVariantSet("Physics").SetVariantSelection("physx")
                self.notes["Physics variants"] = ", ".join(names)
        APP.update()
        self.notes["stage: metersPerUnit, kilogramsPerUnit, upAxis"] = (
            f"{UsdGeom.GetStageMetersPerUnit(stage):g}, {UsdPhysics.GetStageKilogramsPerUnit(stage):g}, "
            f"{UsdGeom.GetStageUpAxis(stage)}")
        prims = all_prims(stage)
        self.root = None
        for p in prims:
            if p.HasAPI(UsdPhysics.ArticulationRootAPI):
                self.root = p
                break
        self.joints = {}
        for p in prims:
            if p.IsA(UsdPhysics.RevoluteJoint) and p.GetName().endswith("_wheel_joint"):
                self.joints[p.GetName()[:-len("_wheel_joint")]] = p
        if sorted(self.joints) != sorted(WHEELS):
            raise RuntimeError(f"wheel joints found: {sorted(self.joints)}")
        self.wheel_link = {w: stage.GetPrimAtPath(UsdPhysics.RevoluteJoint(j).GetBody1Rel().GetTargets()[0])
                           for w, j in self.joints.items()}
        base_path = UsdPhysics.RevoluteJoint(self.joints["front_left"]).GetBody0Rel().GetTargets()[0]
        self.base = stage.GetPrimAtPath(base_path)
        if self.root is None:
            UsdPhysics.ArticulationRootAPI.Apply(self.base)
            self.root = self.base
            self.notes["articulation root"] = "none in the import; applied to " + str(base_path)
        else:
            self.notes["articulation root"] = str(self.root.GetPath())
        self.notes["base link, wheel links"] = f"{base_path}; " + ", ".join(
            str(self.wheel_link[w].GetPath()) for w in WHEELS)

        def joint(w):
            j = UsdPhysics.RevoluteJoint(self.joints[w])
            return f"{w}: {j.GetAxisAttr().Get()}, {j.GetLowerLimitAttr().Get()}, {j.GetUpperLimitAttr().Get()}"
        self.notes["wheel joints: axis, lower and upper limits"] = "; ".join(joint(w) for w in WHEELS)
        rigid = [p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        self.notes["rigid bodies"] = ", ".join(str(p.GetPath()) for p in rigid)
        self._masses(rigid)
        self._colliders(prims)
        self._drives()
        self._bodies(rigid)
        self._scene()
        self._ground()
        self._place_prim()
        APP.update()
        self.physics.use_cpu()
        timeline.play()
        APP.update()
        self.physics.configure(self.dt)
        self.view = self.physics.view()
        self.art = self.view.create_articulation_view(str(self.root.GetPath()))
        if self.art is None or not self.art.check():
            raise RuntimeError(f"no valid articulation view at {self.root.GetPath()}")
        meta = self.art.shared_metatype
        self.dof_names, self.link_names = list(meta.dof_names), list(meta.link_names)
        self.dof = {w: self.dof_names.index(f"{w}_wheel_joint") for w in WHEELS}
        self.link_of = {w: self.link_names.index(self.wheel_link[w].GetName()) if self.wheel_link[w].GetName()
                        in self.link_names else None for w in WHEELS}
        self.ndof = len(self.dof_names)
        self.idx = np.array([0], dtype=np.int32)
        self.q_raw = np.asarray(self.art.get_dof_positions(), dtype=float)[0].copy()
        self.q_unwrapped = self.q_raw.copy()
        self.contacts = {w: self._contact_view(w) for w in WHEELS}
        self.body_view = None
        try:
            self.body_view = self.view.create_rigid_body_view(str(self.base.GetPath()))
        except Exception:                          # noqa: BLE001
            self.body_view = None

    def _masses(self, rigid):
        """The description's masses, centres of mass and inertias, with the
        principal axes of the link's frame; the import's values are recorded
        first."""
        want = urdf_inertials()
        rows = []
        for p in rigid:
            mapi = UsdPhysics.MassAPI(p)
            m = mapi.GetMassAttr().Get() if p.HasAPI(UsdPhysics.MassAPI) else None
            rows.append(f"{p.GetName()} {m}")
        self.notes["masses authored by the import [kg]"] = ", ".join(rows)

        def axle(w):
            inertia = inertia_in_link_frame(self.wheel_link[w]) if self.wheel_link[w].HasAPI(UsdPhysics.MassAPI) else None
            return f"{w} {inertia[1][1]:.5f}" if inertia is not None else f"{w} none"
        self.notes["wheels' inertia about the axle (y) in the import [kg m^2]"] = ", ".join(axle(w) for w in WHEELS)
        fixes = []

        def author(prim, name, ref):
            mapi = UsdPhysics.MassAPI.Apply(prim)
            if mapi.GetMassAttr().Get() != ref["mass"]:
                fixes.append(name)
            mapi.CreateMassAttr(ref["mass"])
            mapi.CreateDiagonalInertiaAttr(Gf.Vec3f(*ref["diag"]))
            mapi.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            mapi.CreateCenterOfMassAttr(Gf.Vec3f(*ref["com"]))

        for w in WHEELS:
            ref = dict(want[f"{w}_wheel_link"])
            if self.setup.wheel_inertia != 1.0:      # the axle is the link's y (the joints' axis Y)
                ref["diag"] = [ref["diag"][0], ref["diag"][1] * self.setup.wheel_inertia, ref["diag"][2]]
            author(self.wheel_link[w], f"{w}_wheel_link", ref)
        ref = want["chassis_link"]          # merged into base_link (chassis_joint at the origin)
        author(self.base, self.base.GetName(), ref)
        self.base_com = tuple(ref["com"])
        self.base_mass = ref["mass"]
        self.wheel_masses = {w: want[f"{w}_wheel_link"]["mass"] for w in WHEELS}
        self.wheel_com = {w: tuple(want[f"{w}_wheel_link"]["com"]) for w in WHEELS}
        self.notes["masses set from the description"] = ", ".join(fixes) or "none differed"
        self.notes["wheels' inertia about the axle (y) as composed [kg m^2] (description "
                   f"{want['front_left_wheel_link']['diag'][1]:.5f})"] = ", ".join(axle(w) for w in WHEELS)
        self.notes[WHEEL_INERTIA_KEY] = ", ".join(axle(w) for w in WHEELS)

    def _colliders(self, prims):
        cols = [p for p in prims if p.HasAPI(UsdPhysics.CollisionAPI)]
        desc = []
        for p in cols:
            approx = ""
            if p.HasAPI(UsdPhysics.MeshCollisionAPI):
                approx = UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get()
            desc.append(f"{p.GetPath()} {p.GetTypeName()} {approx}".strip())
        self.notes["colliders of the import"] = "; ".join(desc)
        self.wheel_colliders = {w: [p for p in cols if str(p.GetPath()).startswith(str(self.wheel_link[w].GetPath()))]
                                for w in WHEELS}
        if self.setup.wheels == "sphere":
            for w in WHEELS:
                for p in self.wheel_colliders[w]:
                    UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr(False)
                sphere = UsdGeom.Sphere.Define(self.stage, str(self.wheel_link[w].GetPath()) + "/sphere_collider")
                sphere.CreateRadiusAttr(RADIUS)
                sphere.CreatePurposeAttr(UsdGeom.Tokens.guide)
                UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
                self.wheel_colliders[w] = [sphere.GetPrim()]
        mat = UsdShade.Material.Define(self.stage, "/World/Looks/ground_material")
        mapi = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        mapi.CreateStaticFrictionAttr(self.mu)
        mapi.CreateDynamicFrictionAttr(self.mu)
        mapi.CreateRestitutionAttr(0.0)
        if self.setup.compliant:
            pm = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
            pm.CreateCompliantContactStiffnessAttr(self.setup.compliant[0])
            pm.CreateCompliantContactDampingAttr(self.setup.compliant[1])
            self.notes["material: compliant contacts, stiffness [N/m], damping [N s/m]"] = (
                f"{pm.GetCompliantContactStiffnessAttr().Get()}, {pm.GetCompliantContactDampingAttr().Get()}")
        self.material = mat
        for w in WHEELS:
            # bound on the wheel's link, which its colliders inherit, and on the
            # colliders themselves where they can be edited
            UsdShade.MaterialBindingAPI.Apply(self.wheel_link[w]).Bind(
                mat, UsdShade.Tokens.weakerThanDescendants, "physics")
            for p in self.wheel_colliders[w]:
                if p.IsInstanceProxy():
                    continue
                UsdShade.MaterialBindingAPI.Apply(p).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
        self.notes["wheel colliders"] = "; ".join(
            f"{w}: " + ", ".join(f"{p.GetTypeName()}" for p in self.wheel_colliders[w]) for w in WHEELS)
        skipped = []
        for w in WHEELS:
            for p in self.wheel_colliders[w]:
                if p.IsInstanceProxy():
                    skipped.append(str(p.GetPath()))
                    continue
                api = PhysxSchema.PhysxCollisionAPI.Apply(p)
                api.CreateContactOffsetAttr(self.setup.contact_offset)
                api.CreateRestOffsetAttr(0.0)
        self.notes["wheel colliders left at the import's offsets (instance proxies)"] = \
            ", ".join(skipped) or "none"

    def _bodies(self, rigid):
        """Chapter 11's settings: the articulation never sleeps and has no
        stabilization threshold (PhysX's defaults are 5e-5 and 1e-5, and the
        importer authors no PhysxArticulationAPI), 32 position iterations and
        the setup's velocity iterations, no self-collision; no damping of the
        links (PhysX's default angular damping is 0.05); contact reports on
        the wheels' links, without which PhysX's rigid contact views report
        nothing."""
        art = PhysxSchema.PhysxArticulationAPI.Apply(self.root)
        art.CreateSleepThresholdAttr(0.0)
        art.CreateStabilizationThresholdAttr(0.0)
        art.CreateSolverPositionIterationCountAttr(32)
        art.CreateSolverVelocityIterationCountAttr(self.setup.velocity_iterations)
        art.CreateEnabledSelfCollisionsAttr(False)
        for p in rigid:
            if p.IsInstanceProxy():
                continue
            rb = PhysxSchema.PhysxRigidBodyAPI.Apply(p)
            rb.CreateLinearDampingAttr(0.0)
            rb.CreateAngularDampingAttr(0.0)
        for w in WHEELS:
            PhysxSchema.PhysxContactReportAPI.Apply(self.wheel_link[w]).CreateThresholdAttr(0.0)
        self.notes["articulation: sleep and stabilization thresholds, iterations; links' damping; "
                   "contact reports"] = (f"0, 0, 32 and {self.setup.velocity_iterations} (authored); 0 (authored); "
                                         "on the four wheel links")

    def _drives(self):
        before = []
        for w, j in self.joints.items():
            d = UsdPhysics.DriveAPI.Get(j, "angular")
            if d:
                before.append(f"{w}: stiffness {d.GetStiffnessAttr().Get()}, damping {d.GetDampingAttr().Get()}, "
                              f"max force {d.GetMaxForceAttr().Get()}")
            d = UsdPhysics.DriveAPI.Apply(j, "angular")
            d.CreateTypeAttr("force")
            d.CreateStiffnessAttr(0.0)
            d.CreateDampingAttr(KV * math.pi / 180.0)        # N m s per degree, as USD reads it
            d.CreateMaxForceAttr(TORQUE)
            d.CreateTargetVelocityAttr(0.0)
        self.notes["wheel drives of the import"] = "; ".join(before) or "none"

    def _scene(self):
        scenes = [p for p in all_prims(self.stage) if p.IsA(UsdPhysics.Scene)]
        if scenes:
            scene = UsdPhysics.Scene(scenes[0])
        else:
            scene = UsdPhysics.Scene.Define(self.stage, "/World/physicsScene")
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr(G)
        px = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        px.CreateTimeStepsPerSecondAttr(int(round(1.0 / self.dt)))
        # the settings of Chapter 11 (code/ch11/cart_isaac/stage.py), the solver
        # of the setup; the friction type is left at its fallback, patch
        # (omni.physx 110.4 deprecates the attribute)
        px.CreateSolverTypeAttr(self.setup.solver)
        px.CreateEnableGPUDynamicsAttr(False)
        px.CreateBroadphaseTypeAttr("MBP")
        px.CreateEnableCCDAttr(False)
        px.CreateEnableStabilizationAttr(False)
        if self.setup.correlation:
            px.CreateFrictionCorrelationDistanceAttr(self.setup.correlation)
        self.notes["physics scene"] = f"{scene.GetPath()} (in the import: {bool(scenes)})"
        corr_attr = px.GetFrictionCorrelationDistanceAttr()
        offs_attr = px.GetFrictionOffsetThresholdAttr()
        self.notes["scene: friction correlation distance, friction offset threshold [m]"] = (
            f"{corr_attr.Get() if corr_attr and corr_attr.HasAuthoredValue() else 'not authored (default 0.025)'}, "
            f"{offs_attr.Get() if offs_attr and offs_attr.HasAuthoredValue() else 'not authored (default 0.04)'}")

        def val(name):
            try:
                return str(getattr(px, f"Get{name}Attr")().Get())
            except Exception as exc:               # noqa: BLE001
                return f"n/a ({type(exc).__name__})"
        self.notes["scene: solver, friction type, CCD, stabilization, GPU dynamics, broadphase, "
                   "time steps per second"] = ", ".join(
            val(n) for n in ("SolverType", "FrictionType", "EnableCCD", "EnableStabilization",
                             "EnableGPUDynamics", "BroadphaseType", "TimeStepsPerSecond"))

    def _ground(self):
        """The setup's ground through z = 0: a USD Plane (axis Z), which PhysX
        makes an unbounded plane, or version 2's box, 80 m wide and 0.1 m
        thick."""
        path = "/World/ground"
        if self.setup.ground == "plane":
            if not hasattr(UsdGeom, "Plane"):
                raise RuntimeError("this USD has no UsdGeom.Plane")
            plane = UsdGeom.Plane.Define(self.stage, path)
            plane.CreateAxisAttr(UsdGeom.Tokens.z)
            plane.CreateWidthAttr(80.0)
            plane.CreateLengthAttr(80.0)
            plane.CreatePurposeAttr(UsdGeom.Tokens.guide)
            prim = plane.GetPrim()
            self.notes["ground"] = "a USD Plane, axis Z, at height 0"
        elif self.setup.ground == "box":
            cube = UsdGeom.Cube.Define(self.stage, path)
            cube.CreateSizeAttr(1.0)
            xf = UsdGeom.XformCommonAPI(cube.GetPrim())
            if not (xf.SetTranslate(Gf.Vec3d(0.0, 0.0, -0.05)) and xf.SetScale(Gf.Vec3f(80.0, 80.0, 0.1))):
                raise RuntimeError("the ground's transform could not be authored")
            prim = cube.GetPrim()
            self.notes["ground"] = "a box 80 m wide and 0.1 m thick, its top at height 0"
        else:
            raise ValueError(self.setup.ground)
        UsdPhysics.CollisionAPI.Apply(prim)
        api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        api.CreateContactOffsetAttr(self.setup.contact_offset)
        api.CreateRestOffsetAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            self.material, UsdShade.Tokens.weakerThanDescendants, "physics")
        self.ground_path = path

    def _place_prim(self):
        """Start the simulation with the wheels just above the ground: the
        robot's top prim raised by the height of touching wheels. Where the
        import's transform operations do not suit XformCommonAPI, a
        translation is added in front of them."""
        top = self.stage.GetPrimAtPath("/" + str(self.base.GetPath()).strip("/").split("/")[0])
        lift = Gf.Vec3d(0.0, 0.0, self.rest_height() + 0.001)
        how = "XformCommonAPI"
        if not UsdGeom.XformCommonAPI(top).SetTranslate(lift):
            xf = UsdGeom.Xformable(top)
            ops = xf.GetOrderedXformOps()
            op = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "place")
            op.Set(lift)
            xf.SetXformOpOrder([op] + list(ops))
            how = "a translation added in front of the import's operations"
        self.notes["robot's top prim, raised by"] = f"{top.GetPath()}, {how}"

    def rest_height(self):
        return RADIUS - WHEEL_Z        # the wheel's radius less its centre's height in base_link

    def _contact_view(self, w):
        """A contact view of the wheel's rigid body against the ground, in the
        forms that Section 11.2's script found to work."""
        path = str(self.wheel_link[w].GetPath())
        error = None
        for pattern, filters in (([path], [[self.ground_path]]), (path, [self.ground_path]),
                                 ([path], [self.ground_path])):
            try:
                view = self.view.create_rigid_contact_view(pattern, filter_patterns=filters,
                                                           max_contact_data_count=64)
                if view is not None and view.check():
                    return view
            except Exception as exc:               # noqa: BLE001
                error = exc
        raise RuntimeError(f"no rigid contact view for {w}: {error!r}")

    def close(self):
        timeline = omni.timeline.get_timeline_interface()
        self.contacts, self.art = {}, None
        if self.physics.own_view and self.view is not None:
            try:
                self.view.invalidate()
            except Exception:                      # noqa: BLE001
                pass
        self.view = None
        timeline.stop()
        APP.update()
        self._restore_approximation()

    def export(self, path):
        self.stage.GetSessionLayer().Export(path)

    # --- placing and stepping -------------------------------------------------------
    def place(self, x=0.0, y=0.0, heading=0.0, speed=0.0, lateral=0.0, spin=None):
        """The robot just into the ground, moving at speed forwards and
        lateral leftwards, its wheels turning at spin [rad/s] (by default the
        spin that rolls it at speed); the controller and the drivers reset."""
        z = self.rest_height() - PLACE_DEPTH
        root = np.array([[x, y, z, 0.0, 0.0, math.sin(heading / 2), math.cos(heading / 2)]], dtype=np.float32)
        c, s = math.cos(heading), math.sin(heading)
        vel = np.array([[speed * c - lateral * s, speed * s + lateral * c, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        self.art.set_root_transforms(root, self.idx)
        self.art.set_root_velocities(vel, self.idx)
        spin = speed / RADIUS if spin is None else spin
        q = np.zeros((1, self.ndof), dtype=np.float32)
        qd = np.zeros((1, self.ndof), dtype=np.float32)
        for w in WHEELS:
            qd[0, self.dof[w]] = spin
            self.target[w], self.z[w] = spin, 0.0
        self.art.set_dof_positions(q, self.idx)
        self.art.set_dof_velocities(qd, self.idx)
        self.art.set_dof_velocity_targets(qd.copy(), self.idx)
        try:
            self.view.update_articulations_kinematic()
        except Exception:                          # noqa: BLE001
            pass
        self.q_raw = np.asarray(self.art.get_dof_positions(), dtype=float)[0].copy()
        self.q_unwrapped = self.q_raw.copy()
        self.drive.reset(*self.side_positions())
        if speed:
            self.drive.previous.clear()
            self.drive.previous.extend([(speed, 0.0), (speed, 0.0)])
        self.command, self.direct, self.step_count = (speed, 0.0), None, 0
        self._anchors, self._anchors_step = None, None
        self._tgt_prev = None
        self.set_push(0.0)

    def wheel_targets(self, left, right):
        self.direct = (left, right)

    def dofs(self):
        """The joints' positions, unwrapped, and their velocities. PhysX wraps
        the position of a revolute joint of type eREVOLUTE by 4 pi when it
        passes +-2 pi; between two readings a step apart the wheels turn far
        less than pi, so any change of more than pi is a wrap, which is
        removed and counted."""
        q = np.asarray(self.art.get_dof_positions(), dtype=float)[0]
        qd = np.asarray(self.art.get_dof_velocities(), dtype=float)[0]
        d = q - self.q_raw
        jumps = np.abs(d) > math.pi
        if jumps.any():
            self.wraps += int(jumps.sum())
            d = d - TWO_PI * np.round(d / TWO_PI)
        self.q_unwrapped = self.q_unwrapped + d
        self.q_raw = q.copy()
        return self.q_unwrapped.copy(), qd

    def side_positions(self):
        q, _ = self.dofs()
        return (0.5 * sum(q[self.dof[w]] for w in LEFT), 0.5 * sum(q[self.dof[w]] for w in RIGHT))

    def watch(self, kind="motion"):
        """Starts an envelope over the steps that follow: 'rest', or 'motion'
        measured from the base's height at rest."""
        self.env = Envelope(kind, self.rest_dz if kind == "motion" else 0.0)
        return self.env

    def unwatch(self):
        env, self.env = self.env, None
        return env

    def step(self, n=1):
        for _ in range(n):
            q, qd = self.dofs()
            if self.env is not None and self._tgt_prev is not None:
                self.env.torques([max(-TORQUE, min(TORQUE, KV * (self._tgt_prev[w] - qd[self.dof[w]])))
                                  for w in WHEELS])
            if self.step_count % self.steps_per_update == 0:
                if self.direct is None:
                    left = 0.5 * sum(q[self.dof[w]] for w in LEFT)
                    right = 0.5 * sum(q[self.dof[w]] for w in RIGHT)
                    lt, rt = self.drive.update(*self.command, left, right)
                else:
                    lt, rt = self.direct
                for w in LEFT:
                    self.target[w] = lt
                for w in RIGHT:
                    self.target[w] = rt
            tgt = np.zeros((1, self.ndof), dtype=np.float32)
            for w in WHEELS:
                e = self.target[w] - qd[self.dof[w]]
                z = self.z[w] + self.dt / TI * e
                room = TORQUE / KV
                self.z[w] = min(max(z, -room - e), room - e)
                tgt[0, self.dof[w]] = self.target[w] + self.z[w]
            self._tgt_prev = {w: float(tgt[0, self.dof[w]]) for w in WHEELS}
            self.art.set_dof_velocity_targets(tgt, self.idx)
            if self.push:
                self._apply_push()
            self.physics.step()
            self.step_count += 1
            if self.env is not None:
                r = np.asarray(self.art.get_root_transforms(), dtype=float)[0]
                roll, pitch = roll_pitch(r[3:7])
                self.env.add((float(r[2]) - self.rest_height()) * 1e3, math.degrees(max(abs(roll), abs(pitch))))
                taus = self._joint_torques()
                if taus is not None:
                    self.env.torques([taus[w] for w in WHEELS], "PhysX")

    def _joint_torques(self):
        """The wheels' joint torques of the last step as PhysX reports them,
        the joints' forces projected on their axes, which for these joints
        without friction or limits are the drives' torques [N m]; None once
        they cannot be read (the drive law's estimate then stands alone)."""
        if self.joint_torques_read is False:
            return None
        try:
            f = np.asarray(self.art.get_dof_projected_joint_forces(), dtype=float).reshape(-1)
            out = {w: float(f[self.dof[w]]) for w in WHEELS}
            self.joint_torques_read = True
        except Exception as exc:                   # noqa: BLE001
            self.joint_torques_read = False
            self.last_taus = {w: float("nan") for w in WHEELS}
            self.notes["PhysX's joint torques"] = f"not read ({exc!r}); the drive law's estimate only"
            report_once("PhysX's joint torques (the drive law's estimate went on alone)", exc)
            return None
        self.last_taus = out
        return out

    def run(self, seconds):
        self.step(int(round(seconds / self.dt)))

    def set_push(self, force_y):
        self.push = force_y

    def _apply_push(self):
        """The push, at the base's centre of mass (the description's, set by
        _masses), where MuJoCo's xfrc_applied acts."""
        if self.body_view is None:
            raise RuntimeError("no rigid body view of the base for the push")
        f = np.array([[0.0, self.push, 0.0]], dtype=np.float32)
        t = np.zeros((1, 3), dtype=np.float32)
        tr = np.asarray(self.body_view.get_transforms(), dtype=float)[0]
        com = tr[:3] + rotation(tr[3:7]) @ np.array(self.base_com)
        self.body_view.apply_forces_and_torques_at_position(
            f, t, com.reshape(1, 3).astype(np.float32), np.array([0], dtype=np.int32), True)

    # --- what the simulator knows ---------------------------------------------------
    def pose(self):
        r = np.asarray(self.art.get_root_transforms(), dtype=float)[0]
        qx, qy, qz, qw = r[3:7]
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        return float(r[0]), float(r[1]), yaw, float(r[2])

    def tilt(self):
        """Roll and pitch of the base [rad]."""
        return roll_pitch(np.asarray(self.art.get_root_transforms(), dtype=float)[0][3:7])

    def body_velocity(self):
        v = np.asarray(self.art.get_root_velocities(), dtype=float)[0]
        yaw = self.pose()[2]
        c, s = math.cos(yaw), math.sin(yaw)
        return float(c * v[0] + s * v[1]), float(-s * v[0] + c * v[1]), float(v[5])

    def out_of_plane_rates(self):
        """The base's vertical speed [m/s] and its angular velocity about the
        world's x and y [rad/s]."""
        v = np.asarray(self.art.get_root_velocities(), dtype=float)[0]
        return float(v[2]), float(v[3]), float(v[4])

    def rim_speeds(self):
        _, qd = self.dofs()
        return {w: RADIUS * float(qd[self.dof[w]]) for w in WHEELS}

    def loads(self):
        """Each wheel's normal load in the last step [N]."""
        return {w: normal_load(view, self.dt) for w, view in self.contacts.items()}

    def contact_points(self):
        """(wheel, x, y, normal force, separation, normal's z) of each contact
        point of the last step, x and y in the frame of the base."""
        x0, y0, yaw, _ = self.pose()
        c, s = math.cos(yaw), math.sin(yaw)
        out = []
        for w, view in self.contacts.items():
            forces, points, normals, seps, counts, starts = contact_data(view, self.dt)
            for k in range(len(counts)):
                for i in range(int(starts[k]), int(starts[k]) + int(counts[k])):
                    px, py = points[i, 0] - x0, points[i, 1] - y0
                    out.append((w, c * px + s * py, -s * px + c * py, abs(float(forces[i])),
                                float(seps[i]), float(normals[i, 2])))
        return out

    def friction_raw(self):
        """{wheel: [(force, point), ...]}, PhysX's friction data of the last
        step in the world frame; None once they cannot be read (the first
        failure is reported in errors.txt, and the diagnostics go on without
        them)."""
        if self.friction_read is False:
            return None
        try:
            out = {w: friction_data(view, self.dt) for w, view in self.contacts.items()}
        except Exception as exc:                   # noqa: BLE001
            self.friction_read = False
            self.notes["PhysX's friction data"] = f"not read ({exc!r})"
            report_once("PhysX's friction data (the diagnostics went on without them)", exc)
            return None
        self.friction_read = True
        return out

    def wheel_motion(self):
        """{wheel: (centre of mass, linear velocity, angular velocity, origin,
        rotation)} of the wheels' links in the world frame, from PhysX's link
        transforms and velocities (the latter of the links' centres of mass,
        which for these wheels lie at the links' origins); None once they
        cannot be read."""
        if self.link_motion_read is False or None in self.link_of.values():
            return None
        try:
            tr = np.asarray(self.art.get_link_transforms(), dtype=float).reshape(-1, 7)
            vel = np.asarray(self.art.get_link_velocities(), dtype=float).reshape(-1, 6)
            out = {}
            for w in WHEELS:
                k = self.link_of[w]
                rot = rotation(tr[k, 3:7])
                com = tr[k, :3] + rot @ np.array(self.wheel_com[w])
                out[w] = (com, vel[k, :3], vel[k, 3:6], tr[k, :3].copy(), rot)
        except Exception as exc:                   # noqa: BLE001
            self.link_motion_read = False
            self.notes["PhysX's link velocities"] = f"not read ({exc!r})"
            report_once("PhysX's link velocities (the slip from the base's twist and the rims instead)", exc)
            return None
        self.link_motion_read = True
        return out

    def friction(self):
        """Each wheel's friction anchors of the last step, in the frame of the
        base: a list of (fx, fy, x, y, slip, age), the force [N] as PhysX
        reports it (multiplied by friction_sign where that is known, so that
        it is the force on the wheel) at the anchor's point [m], the slip
        there [m/s], the velocity of the wheel's material point at the anchor
        along the ground, (sx, sy), from PhysX's link velocities (None where
        they cannot be read), and the anchor's age [steps]. PhysX reports the
        wheel's side of an anchor, the point it stores in the wheel's frame
        placed by the wheel's pose at the start of the step (DyContactPrep.cpp),
        so an anchor kept from the step before lies at the same place on the
        wheel: its age then grows by one, and a new anchor has age 1. The age
        is None unless the two steps before were both read, and for wheels
        that are convex meshes, which pivot on their vertices, so that a new
        anchor can fall on the place of the last one. An anchor point that
        lies more than 0.25 m from the wheel's centre, or is not finite, is
        taken at the centre's foot. Empty lists where the friction data
        cannot be read."""
        raw = self.friction_raw()
        self.centre_vel = None
        if raw is None:
            return {w: [] for w in WHEELS}
        motion = self.wheel_motion()
        x0, y0, yaw, _ = self.pose()
        c, s = math.cos(yaw), math.sin(yaw)
        if motion is not None:
            self.centre_vel = {w: (c * motion[w][1][0] + s * motion[w][1][1], -s * motion[w][1][0] + c * motion[w][1][1])
                               for w in WHEELS}
        sign = self.friction_sign or 1.0
        before = self._anchors if self._anchors_step == self.step_count - 1 else None
        round_wheels = self.setup.wheels in ("cylinder", "sphere")
        memory = {}
        out = {}
        for w in WHEELS:
            rows = []
            memory[w] = []
            for f, p in raw[w]:
                age = None
                if before is not None and round_wheels and math.isfinite(float(np.sum(p))):
                    origin, rot = before["frames"][w]
                    here = rot.T @ (np.asarray(p, dtype=float) - origin)
                    if before["complete"]:
                        age = 1
                        for there, old in before["anchors"][w]:
                            if float(np.linalg.norm(here - there)) < 2e-5:
                                age = old + 1
                                break
                    memory[w].append((here, age if age is not None else 1))
                px, py = p[0] - x0, p[1] - y0
                x, y = c * px + s * py, -s * px + c * py
                if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(p[2])) or \
                        math.hypot(x - CENTRES[w][0], y - CENTRES[w][1]) > 0.25:
                    x, y = CENTRES[w]
                    p = np.array([x0 + c * x - s * y, y0 + s * x + c * y, 0.0])
                slip = None
                if motion is not None:
                    com, v, om = motion[w][:3]
                    sv = v + np.cross(om, np.asarray(p, dtype=float) - com)
                    slip = (c * sv[0] + s * sv[1], -s * sv[0] + c * sv[1])
                rows.append((sign * (c * f[0] + s * f[1]), sign * (-s * f[0] + c * f[1]), x, y, slip, age))
            out[w] = rows
        if motion is not None:
            self._anchors = {"frames": {w: (motion[w][3], motion[w][4]) for w in WHEELS}, "anchors": memory,
                             "complete": before is not None}
            self._anchors_step = self.step_count
        return out


# ----------------------------------------------------------------------------
# Loads over many steps
# ----------------------------------------------------------------------------
LOAD_COLUMNS = [f"load {w} [N]" for w in WHEELS] + ["load, total [N]"]


class Loads:
    """The loads of the single steps of a window."""

    def __init__(self):
        self.rows = []

    def add(self, ld):
        self.rows.append([ld[w] for w in WHEELS])
        return ld

    def stats(self):
        """The mean of each wheel and of the total, the total's least, largest
        and standard deviation, the correlation of successive totals, and the
        shares of steps in which all four wheels carry load and in which
        none does."""
        a = np.asarray(self.rows, dtype=float).reshape(-1, len(WHEELS))
        if len(a) == 0:
            raise RuntimeError("no steps")
        tot = a.sum(axis=1)
        return dict(mean={w: float(a[:, i].mean()) for i, w in enumerate(WHEELS)},
                    total=float(tot.mean()), lo=float(tot.min()), hi=float(tot.max()), std=float(tot.std()),
                    lag1=lag1(tot), all_down=float(np.mean(np.all(a > 0.0, axis=1))),
                    none=float(np.mean(tot <= 0.0)), steps=len(a))


def describe_loads(st):
    return (", ".join(f"{st['mean'][w]:.1f}" for w in WHEELS) +
            f"; total {st['total']:.1f} (weight {MASS * G:.1f})")


def describe_scatter(st):
    return (f"{st['lo']:.1f}, {st['hi']:.1f}, {st['std']:.1f}; {corr(st['lag1'])}; "
            f"{share(st['all_down'])}, {share(st['none'])} of {st['steps']} steps")


# ----------------------------------------------------------------------------
# The Coulomb reference
# ----------------------------------------------------------------------------
def coulomb_twist(contacts, rims, mu):
    """The steady motion (u, v, Omega) of the robot under Coulomb friction
    (husky/coulomb.py) for contacts (wheel, x, y, N) and rim speeds: the twist
    that minimizes the friction's dissipation, sum_k mu N_k |slip_k(q)|, whose
    gradient is minus the friction's force and moment (coulomb.residual), so
    that its minimum, which exists and is the steady motion, is where they
    vanish, or, for contacts that stick, where the others' are balanced
    within those contacts' limits. Iteratively reweighted least squares, the
    weights mu N_k / max(|slip_k|, delta), delta from 1e-2 down to 1e-10 m/s.
    Returns the twist, the largest force [N] or moment [N m] that the
    slipping contacts leave unbalanced, and the number of contacts that stick
    (slip below 1e-7 m/s)."""
    rows = [(np.array([[1.0, 0.0, -y], [0.0, 1.0, x]]), np.array([rims[w], 0.0]), mu * n)
            for w, x, y, n in contacts if n > 0.0]
    if len(rows) < 2:
        raise RuntimeError("fewer than two loaded contacts")
    q = np.zeros(3)
    for delta in 10.0 ** -np.arange(2.0, 11.0):
        for _ in range(500):
            a, b = np.zeros((3, 3)), np.zeros(3)
            for m, r, c in rows:
                s = m @ q - r
                wgt = c / max(math.hypot(s[0], s[1]), delta)
                a += wgt * (m.T @ m)
                b += wgt * (m.T @ r)
            q_new = np.linalg.lstsq(a, b, rcond=None)[0]
            moved = float(np.max(np.abs(q_new - q)))
            q = q_new
            if moved < 1e-13:
                break
    stick = [k for k, (m, r, c) in enumerate(rows) if math.hypot(*(m @ q - r)) < 1e-7]
    slipping = [(w, x, y, n) for k, (w, x, y, n) in enumerate([c for c in contacts if c[3] > 0.0])
                if k not in stick]
    res = coulomb.residual(q, slipping, rims, mu, eps=0.0) if slipping else np.zeros(3)
    return q, float(np.abs(res).max()), len(stick)


class TwoRowModel:
    """PhysX's patch friction at a contact that slips, as its solver resolves
    it, reduced to a law that can be set beside Coulomb's (the docstring,
    version 4's run): the friction is mu N times minus the unit vector of
    T^T D T s, s the slip (along the ground, in the base's frame), T the
    rotation into (t0, t1), t0 along the velocity of the wheel's centre and
    t1 across it (below 0.01 m/s, PhysX's fallback t0, the world's -y,
    DyTGSContactPrep.cpp), and D = diag(1 / (W_00 + cfm), 1 / (W_11 + cfm)),
    W the contact point's inverse effective mass in that frame:
    J M^-1 J^T for the robot with the base free and the four wheels free to
    spin, from the description's masses, centres of mass and inertias (the
    wheels' inertia about the axle scaled as the setup says), the base
    level. cfm: 0 (the law), or that of PhysX's articulation links (cfmScale,
    0.025 by default, times the inverse length scale, 1).
    """
    CFM = 0.025

    def __init__(self, wheel_inertia=1.0):
        ine = urdf_inertials()
        n = 6 + len(WHEELS)
        m = np.zeros((n, n))

        def body(mass, com, inertia, wheel=None):
            jv = np.zeros((3, n))
            jv[:, 0:3] = np.eye(3)
            jv[:, 3:6] = -skew(np.asarray(com, dtype=float))
            jw = np.zeros((3, n))
            jw[:, 3:6] = np.eye(3)
            if wheel is not None:
                jw[1, 6 + wheel] = 1.0
            return mass * jv.T @ jv + jw.T @ inertia @ jw

        base = ine["chassis_link"]          # merged into base_link
        m += body(base["mass"], base["com"], np.diag(base["diag"]))
        self.centre = {}
        for k, w in enumerate(WHEELS):
            ref = ine[f"{w}_wheel_link"]
            diag = [ref["diag"][0], ref["diag"][1] * wheel_inertia, ref["diag"][2]]
            c = np.array([CENTRES[w][0], CENTRES[w][1], WHEEL_Z]) + np.asarray(ref["com"], dtype=float)
            self.centre[w] = c
            m += body(ref["mass"], c, np.diag(diag), wheel=k)
        self.minv = np.linalg.inv(m)
        self.n = n
        self._cache = {}

    def block(self, wheel, x, y):
        """The 2 x 2 tangential block of J M^-1 J^T at (x, y) on the ground
        under the wheel, in the base's frame [1/kg]."""
        k = WHEELS.index(wheel)
        p = np.array([x, y, WHEEL_Z - RADIUS])
        j = np.zeros((3, self.n))
        j[:, 0:3] = np.eye(3)
        j[:, 3:6] = -skew(p)
        j[:, 6 + k] = np.cross([0.0, 1.0, 0.0], p - self.centre[wheel])
        return (j @ self.minv @ j.T)[:2, :2]

    def inverse_mass(self, wheel, x, y):
        """block at (x, y) rounded to 1 mm, kept for the next call (under the
        wheels, the block changes by less than 0.2 % per mm)."""
        key = (wheel, round(x, 3), round(y, 3))
        out = self._cache.get(key)
        if out is None:
            out = self.block(wheel, key[1], key[2])
            if len(self._cache) < 100000:
                self._cache[key] = out
        return out

    def direction(self, wheel, x, y, slip, centre, yaw=0.0, cfm=0.0):
        """The unit vector of the law's friction at (x, y) for the slip and the
        wheel centre's velocity (both in the base's frame); zero without
        slip."""
        s = np.asarray(slip, dtype=float)
        if math.hypot(s[0], s[1]) < 1e-12:
            return np.zeros(2)
        c = np.asarray(centre, dtype=float)
        if math.hypot(c[0], c[1]) > 0.01:
            t0 = c / math.hypot(c[0], c[1])
        else:
            t0 = np.array([-math.sin(yaw), -math.cos(yaw)])
        t = np.array([t0, [-t0[1], t0[0]]])
        w = t @ self.inverse_mass(wheel, x, y) @ t.T
        st = t @ s
        d = t.T @ np.array([st[0] / (w[0, 0] + cfm), st[1] / (w[1, 1] + cfm)])
        nd = math.hypot(d[0], d[1])
        return -d / nd if nd > 0.0 else np.zeros(2)

    def effective_masses(self, wheel):
        """The effective masses along and across the wheel at its centre's
        foot [kg]."""
        w = self.block(wheel, CENTRES[wheel][0], CENTRES[wheel][1])
        return 1.0 / w[0, 0], 1.0 / w[1, 1]


def skew(a):
    return np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])


def two_row_residual(q, contacts, rims, mu, model, cfm):
    """The friction forces of the two-row law and their moment about the
    origin, for the twist q (quasi-static, as coulomb_twist)."""
    u, v, om = q
    out = np.zeros(3)
    for w, x, y, n in contacts:
        s = (u - om * y - rims[w], v + om * x)
        c = (u - om * CENTRES[w][1], v + om * CENTRES[w][0])
        f = mu * n * model.direction(w, x, y, s, c, cfm=cfm)
        out += (f[0], f[1], x * f[1] - y * f[0])
    return out


def two_row_twist(contacts, rims, mu, model, start, cfm=0.0, iterations=200):
    """A steady motion (u, v, Omega) of the two-row law for contacts (wheel,
    x, y, N) and rim speeds, by Newton's method from start with a backtracking
    line search; the law has no potential, so unlike Coulomb's it can have
    several steady motions, and the start picks one. Returns the twist and
    the largest force [N] or moment [N m] left unbalanced."""
    contacts = [c for c in contacts if c[3] > 0.0]
    q = np.array(start, dtype=float)

    def res(x):
        return two_row_residual(x, contacts, rims, mu, model, cfm)

    r = res(q)
    for _ in range(iterations):
        if float(np.abs(r).max()) < 1e-9:
            break
        jac = np.zeros((3, 3))
        for k in range(3):
            dq = np.zeros(3)
            dq[k] = 1e-7
            jac[:, k] = (res(q + dq) - res(q - dq)) / 2e-7
        step = np.linalg.lstsq(jac, -r, rcond=None)[0]
        lam, best = 1.0, None
        while lam > 1e-6:
            trial = q + lam * step
            rt = res(trial)
            if float(np.abs(rt).max()) < float(np.abs(r).max()):
                best = (trial, rt)
                break
            lam *= 0.5
        if best is None:
            break
        q, r = best
    return q, float(np.abs(r).max())


def law_value(q, what):
    """The yaw rate (what = 'yaw') or the radius ('radius') of a twist."""
    if what == "yaw":
        return float(q[2])
    return float(q[0] / q[2]) if abs(q[2]) > 1e-12 else float("inf")


def two_row_states(cs, rims, mu, model, measured, coulomb_q, what, cfm=0.0):
    """The two-row law's steady motions from several starts (PhysX's mean
    twist, the same without its sideways speed, without its forward speed,
    and Coulomb's twist): the one nearest PhysX's motion (or, where none
    converged, the one with the least residual) and the values of the others
    that converged and differ from it by more than 1 %."""
    starts = [tuple(measured), (measured[0], 0.0, measured[2]), (0.0, 0.0, measured[2])]
    if coulomb_q is not None:
        starts.append(tuple(coulomb_q))
    sols = [two_row_twist(cs, rims, mu, model, s, cfm=cfm) for s in starts]
    good = [(q, r) for q, r in sols if r <= 1e-6]
    target = law_value(measured, what)
    if good:
        near = min(good, key=lambda qr: abs(law_value(qr[0], what) - target))
    else:
        near = min(sols, key=lambda qr: qr[1])
    base = law_value(near[0], what)
    others = []
    for q, r in good:
        x = law_value(q, what)
        if abs(x - base) > 0.01 * max(abs(base), 1e-9) and all(abs(x - y) > 0.01 * max(abs(y), 1e-9) for y in others):
            others.append(x)
    return near, others


def model_text(h, cs, rims, measured, coulomb_q, what):
    """'two-row law: ...' for a spin (what = 'yaw') or the turn ('radius'):
    the law's steady motion nearest PhysX's (measured, the mean twist), any
    other it has (two_row_states), and the same with the links' cfm. Only
    under TGS."""
    if h.setup.solver != "TGS":
        return "two-row law: - (PGS clamps each direction on its own)"
    if not cs:
        return "two-row law: no contacts at rest, -"

    def fmt(x):
        return f"{x:.4f}" if what == "yaw" else f"{x:.3f}"

    def one(cfm):
        (q, r), others = two_row_states(cs, rims, h.mu, h.friction_model(), measured, coulomb_q, what, cfm)
        text = fmt(law_value(q, what)) + (f" (residual {r:.1e})" if r > 1e-6 else "")
        if others:
            text += " (another steady motion at " + ", ".join(fmt(x) for x in others) + ")"
        return text

    try:
        return f"two-row law: {one(0.0)}; with the links' cfm {TwoRowModel.CFM:g}: {one(TwoRowModel.CFM)}"
    except Exception as exc:                       # noqa: BLE001 (the measurements stand without the law)
        report_once("the two-row law's steady motion (the measurements went on without it)", exc)
        return f"two-row law: failed ({exc!r})"


def reference_text(q, residual, sticking, measured, what):
    """'reference, difference' for a measured value against the reference's
    yaw rate (what = 'yaw') or radius (what = 'radius')."""
    if what == "yaw":
        if abs(q[2]) < 1e-12:
            return "Coulomb: no yaw rate, -"
        ref = q[2]
        text = f"{ref:.4f}, {pct(measured / ref - 1)}"
    else:
        if abs(q[2]) < 1e-12 or abs(q[0]) < 1e-12:
            return "Coulomb: no turn, -"
        ref = q[0] / q[2]
        text = f"{ref:.3f}, {pct(measured / ref - 1)}"
    if sticking or residual > 1e-6:
        text += f" (residual {residual:.1e} with {sticking} sticking)"
    return text


# ----------------------------------------------------------------------------
# The diagnostics
# ----------------------------------------------------------------------------
def settle(h):
    """Places the robot and holds its wheels until it rests: at least
    SETTLE_MIN, at most SETTLE_MAX, until the base has been steady over the
    last SETTLE_HOLD (its height, roll and pitch, and place on the ground
    changed by less than STEADY) and lies within 5 mm of the height of
    touching round wheels, tilted less than 0.2 deg. Version 3 judged the
    base's velocities instead and, at 0.25 ms, found the robot unsettled
    while the base did not move. Sets h.settled to (rested, seconds)."""
    h.place()
    h.wheel_targets(0.0, 0.0)
    n_min, n_max = int(round(SETTLE_MIN / h.dt)), int(round(SETTLE_MAX / h.dt))
    n_hold = max(2, int(round(SETTLE_HOLD / h.dt)))
    hist = deque(maxlen=n_hold + 1)
    for i in range(1, n_max + 1):
        h.step(1)
        x, y, _, z = h.pose()
        roll, pitch = h.tilt()
        hist.append((x, y, z, roll, pitch))
        if i < n_min or len(hist) <= n_hold:
            continue
        a = np.asarray(hist)
        dz = (z - h.rest_height()) * 1e3
        steady = (np.ptp(a[:, 2]) * 1e3 < STEADY["height"] and
                  math.degrees(max(np.ptp(a[:, 3]), np.ptp(a[:, 4]))) < STEADY["tilt"] and
                  math.hypot(a[-1, 0] - a[0, 0], a[-1, 1] - a[0, 1]) * 1e3 < STEADY["drift"])
        if steady and abs(dz) < 5.0 and math.degrees(max(abs(roll), abs(pitch))) < 0.2:
            h.settled = (True, i * h.dt)
            return h.settled
    h.settled = (False, n_max * h.dt)
    return h.settled


def friction_check(h, v0=1.0, skip=0.005, seconds=0.04):
    """The sign and scale of PhysX's friction data, from the robot's
    momentum in a sideways slide with the wheels held: the impulse of the
    friction that the data report, summed over the wheels and the steps of
    the window, against the change of the robot's momentum over the window,
    its mass times the change of the base's velocity (all its bodies slide
    together, and friction is the only horizontal force on it). The ratio
    is near +1 if the data are the forces on the wheels and near -1 if they
    are those on the ground, and its size tells how complete they are.
    h.friction_sign is set where |ratio| lies between 0.5 and 2, and
    h.friction_scale = |ratio|. Returns the ratio; None where the data
    cannot be read, inf where they are not finite, NaN where the base did
    not slow down."""
    h.friction_sign = h.friction_scale = None
    h.place(lateral=v0)
    h.wheel_targets(0.0, 0.0)
    h.run(skip)
    n = max(4, int(round(seconds / h.dt)))
    v_start = np.asarray(h.art.get_root_velocities(), dtype=float)[0][:2]
    impulse = np.zeros(2)
    for _ in range(n):
        h.step(1)
        raw = h.friction_raw()
        if raw is None:
            return None
        for anchors in raw.values():
            for force, _ in anchors:
                impulse += force[:2] * h.dt
    if not np.all(np.isfinite(impulse)):
        return float("inf")
    dp = MASS * (np.asarray(h.art.get_root_velocities(), dtype=float)[0][:2] - v_start)
    if float(dp @ dp) < 1e-8:
        return float("nan")
    ratio = float(impulse @ dp / (dp @ dp))
    if 0.5 <= abs(ratio) <= 2.0:
        h.friction_sign = 1.0 if ratio > 0 else -1.0
        h.friction_scale = abs(ratio)
    return ratio


FRICTION_CHECK_KEY = ("friction data: the impulse of the friction that PhysX reports, summed over the wheels, "
                      "against the change of the robot's momentum in a sideways slide from 1 m/s with the wheels "
                      "held (+1: the forces on the wheels, -1: on the ground)")


def friction_check_text(ratio):
    if ratio is None:
        return "-; PhysX's friction data could not be read (errors.txt)"
    if ratio == float("inf"):
        return "-; PhysX's friction data are not finite: they are not judged"
    if not math.isfinite(ratio):
        return "nan; the base did not slow down: the friction data are not judged"
    if not 0.5 <= abs(ratio) <= 2.0:
        return f"{ratio:+.3f}; neither: the friction data are not judged"
    text = (f"{ratio:+.3f}; the data are the forces on the wheels" if ratio > 0 else
            f"{ratio:+.3f}; the data are the forces on the ground, and are turned")
    if abs(abs(ratio) - 1.0) > 0.1:
        text += f"; their sizes miss the momentum by {pct(abs(ratio) - 1, 0)}, so |F| / (mu N) below is off as much"
    return text


def settled_text(h):
    rested, t = h.settled
    return f"settled after {t:.2f} s" if rested else f"NOT settled within {t:.1f} s"


def rest_window(h, seconds=WINDOW):
    """At rest after settling: the loads and the contact points of every step
    of the window, binned per wheel and side of its centre plane (inner,
    outer), the envelope, and a trace of the steps."""
    settle(h)
    n = int(round(seconds / h.dt))
    log, rows = Loads(), []
    acc = {(w, s): [0.0, 0.0, 0.0, 0, 0.0, 0.0] for w in WHEELS for s in SIDES}  # f, f x, f y, points, sep, nz
    env = h.watch("rest")
    for i in range(n):
        h.step(1)
        ld = log.add(h.loads())
        counts = {w: 0 for w in WHEELS}
        for w, x, y, f, sep, nz in h.contact_points():
            side = "outer" if abs(y) >= abs(CENTRES[w][1]) else "inner"
            a = acc[(w, side)]
            a[0] += f
            a[1] += f * x
            a[2] += f * y
            a[3] += 1
            a[4] += sep
            a[5] += nz
            counts[w] += 1
        _, _, _, z = h.pose()
        roll, pitch = h.tilt()
        rows.append([round((i + 1) * h.dt, 6), z, roll, pitch] + [ld[w] for w in WHEELS] + [sum(ld.values())] +
                    [counts[w] for w in WHEELS])
    h.unwatch()
    bins = {}
    for key, a in acc.items():
        bins[key] = dict(points=a[3] / n, load=a[0] / n,
                         x=a[1] / a[0] if a[0] > 0 else float("nan"), y=a[2] / a[0] if a[0] > 0 else float("nan"),
                         sep=a[4] / a[3] if a[3] else float("nan"), nz=a[5] / a[3] if a[3] else float("nan"))
    if not h.settled[0]:
        env.flag("not settled")
    if math.isfinite(env.median()):
        h.rest_dz = env.median()
    return log.stats(), bins, env, rows


def reference_contacts(st, bins):
    """The contacts at rest for the Coulomb reference: for each wheel and side
    of its centre plane, the centroid of the contact points weighted by their
    forces, with their mean load over the window; for a wheel without contact
    points, its centre with the mean load of the force matrix. Contacts
    without load are left out; fewer than two loaded ones cannot carry a
    steady motion."""
    out, how = [], []
    for w in WHEELS:
        mine = [(s, bins[(w, s)]) for s in SIDES if bins[(w, s)]["points"] > 0]
        if mine:
            for s, b in mine:
                if b["load"] > 0.0:
                    out.append((w, b["x"], b["y"], b["load"]))
                    how.append(f"{w} {s}")
        elif st["mean"][w] > 0.0:
            out.append((w, CENTRES[w][0], CENTRES[w][1], st["mean"][w]))
            how.append(f"{w} centre")
    if len(out) < 2:
        raise RuntimeError(f"fewer than two loaded contacts at rest: {how}")
    return out, f"{len(out)} contacts: " + ", ".join(how)


class FrictionStats:
    """Over the steps of a window, PhysX's friction against Coulomb's: per
    anchor and step, the angle between the friction on the wheel and the
    opposite of the slip at the anchor, where the slip exceeds SLIP_MIN, the
    slip being the velocity along the ground of the wheel's material point
    at the anchor, from PhysX's link velocities (where those cannot be read,
    that of the base's twist and the wheel's rim speed, (u - Omega y - rim,
    v + Omega x), as husky/coulomb.py has it, which leaves out the base's
    roll and pitch rates); per wheel with anchors and step, the sum of their
    |F| against mu N, for the wheels that carry load; per step the yaw
    moment of all the friction about the base's origin, which in a steady
    motion is near zero; and the share of the anchors that PhysX kept from
    the step before (their age, IsaacHusky.friction). Coulomb's friction has
    the angle 0 and, where it slips, |F| = mu N. Only where the sign of the
    data is known (friction_check). Under TGS, beside it, the angle between
    the friction on the wheel and the two-row law's (TwoRowModel) for the
    same slip and the velocity of the wheel's centre (from PhysX's link
    velocities, or from the base's twist), without the links' cfm and with
    it."""

    def __init__(self, h):
        self.h = h
        self.model_angles, self.model_angles_cfm = [], []
        self.model_off = h.setup.solver != "TGS"
        self.angles, self.ratios, self.moments = [], [], []
        self.steps, self.anchor_steps, self.planar = 0, 0, 0
        self.aged, self.kept, self.oldest = 0, 0, 0

    def add(self, twist, rims, loads, fr):
        if self.h.friction_sign is None:
            return
        mz = 0.0
        model, yaw = None, 0.0
        if not self.model_off:
            try:
                model, yaw = self.h.friction_model(), self.h.pose()[2]
            except Exception as exc:               # noqa: BLE001
                self.model_off = True
                report_once("the two-row law (the friction's statistics went on without it)", exc)
        for w in WHEELS:
            total = 0.0
            if model is not None:
                cv = self.h.centre_vel[w] if self.h.centre_vel is not None else \
                    (twist[0] - twist[2] * CENTRES[w][1], twist[1] + twist[2] * CENTRES[w][0])
            for fx, fy, x, y, slip, age in fr[w]:
                if age is not None:
                    self.aged += 1
                    self.kept += age > 1
                    self.oldest = max(self.oldest, age)
                if slip is None:
                    slip = planar_slip(twist, rims[w], x, y)
                    self.planar += 1
                sx, sy = slip
                nf, ns = math.hypot(fx, fy), math.hypot(sx, sy)
                total += nf
                mz += x * fy - y * fx
                self.anchor_steps += 1
                if nf > 1e-9 and ns > SLIP_MIN:
                    cosang = max(-1.0, min(1.0, -(fx * sx + fy * sy) / (nf * ns)))
                    self.angles.append(math.degrees(math.acos(cosang)))
                    if model is not None:
                        try:
                            pair = []
                            for cfm in (0.0, TwoRowModel.CFM):
                                d = model.direction(w, x, y, (sx, sy), cv, yaw, cfm)
                                cosm = max(-1.0, min(1.0, (fx * d[0] + fy * d[1]) / nf))
                                pair.append(math.degrees(math.acos(cosm)))
                            self.model_angles.append(pair[0])
                            self.model_angles_cfm.append(pair[1])
                        except Exception as exc:   # noqa: BLE001
                            model, self.model_off = None, True
                            report_once("the two-row law (the friction's statistics went on without it)", exc)
            if fr[w] and loads[w] > 0.0:
                self.ratios.append(total / (self.h.mu * loads[w]))
        self.moments.append(mz)
        self.steps += 1

    def text(self):
        if self.h.friction_read is False:
            return "friction against slip: - (PhysX's friction data could not be read)"
        if self.h.friction_sign is None:
            return "friction against slip: - (the sign of PhysX's friction data is not known)"
        if not self.steps:
            return "friction against slip: -"
        if self.ratios:
            r = np.asarray(self.ratios)
            part = f"anchors' |F| / (mu N) median {np.median(r):.2f}, below 0.9 in {share(np.mean(r < 0.9))}"
            scale = self.h.friction_scale
            if scale is not None and abs(scale - 1.0) > 0.1:
                part += f" (the data's sizes miss the momentum by {pct(scale - 1, 0)})"
        else:
            part = "no loaded wheel had anchors"
        part += f"; friction's yaw moment, mean {np.mean(self.moments):+.1f} N m"
        if self.aged:
            part += (f"; anchors kept from the step before in {share(self.kept / self.aged)} of their steps "
                     f"(oldest {self.oldest} steps)")
        if not self.angles:
            return f"friction against slip: no anchor slipped by more than {SLIP_MIN * 1e3:g} mm/s; " + part
        a = np.asarray(self.angles)
        how = " (the slip from the base's twist and the rims)" if self.planar else ""
        law = ""
        if self.model_angles:
            m, mc = np.asarray(self.model_angles), np.asarray(self.model_angles_cfm)
            law = (f"; against the two-row law: angle median {np.median(m):.1f} deg, 90 % below "
                   f"{np.percentile(m, 90):.1f} deg (with the links' cfm {TwoRowModel.CFM:g}: median "
                   f"{np.median(mc):.1f} deg)")
        return (f"friction against slip{how}: angle median {np.median(a):.1f} deg, 90 % below "
                f"{np.percentile(a, 90):.1f} deg (anchors slipping in {share(len(a) / max(1, self.anchor_steps))} "
                f"of their steps){law}; " + part)

    def law_median(self):
        return f"{np.median(self.model_angles):.1f}" if self.model_angles else "-"


def trace_row(h, t, tw, rs, ld, fr):
    """A step of a trace of the spins and the turn (TRACE_SPIN)."""
    u, v, om = tw
    row = [round(t, 6), u, v, om] + [rs[w] for w in WHEELS] + [ld[w] for w in WHEELS]
    for w in WHEELS:
        row += [sum(a[0] for a in fr[w]), sum(a[1] for a in fr[w])]
    for w in WHEELS:
        slips = [a[4] for a in fr[w] if a[4] is not None]
        if slips:
            row += [sum(q[0] for q in slips) / len(slips), sum(q[1] for q in slips) / len(slips)]
            continue
        if fr[w]:
            x, y = sum(a[2] for a in fr[w]) / len(fr[w]), sum(a[3] for a in fr[w]) / len(fr[w])
        else:
            x, y = CENTRES[w]
        row += list(planar_slip(tw, rs[w], x, y))
    row += [h.last_taus[w] for w in WHEELS]
    for w in WHEELS:
        ages = [a[5] for a in fr[w] if a[5] is not None]
        row += [len(fr[w]), max(ages) if ages else float("nan")]
    return row


def planar_slip(twist, rim, x, y):
    """The slip at (x, y) in the base's frame of a wheel with the rim speed
    rim, from the base's twist (u, v, Omega), as husky/coulomb.py has it."""
    u, v, om = twist
    return u - om * y - rim, v + om * x


def mean_rims(h, n, log, fs=None, rows=None, tail=None):
    """Steps n times; the mean twist (u, v, Omega) and rim speeds, the loads
    logged; PhysX's friction judged in fs and, over the last tail steps,
    traced in rows."""
    acc = np.zeros(3)
    rims = {w: 0.0 for w in WHEELS}
    tail = n if tail is None else min(n, tail)
    for i in range(n):
        h.step(1)
        ld = log.add(h.loads())
        tw = h.body_velocity()
        acc += np.array(tw) / n
        rs = h.rim_speeds()
        for w in WHEELS:
            rims[w] += rs[w] / n
        if fs is None and rows is None:
            continue
        fr = h.friction()
        if fs is not None:
            fs.add(tw, rs, ld, fr)
        if rows is not None and i >= n - tail:
            rows.append(trace_row(h, (i + 1) * h.dt, tw, rs, ld, fr))
    return acc, rims


def spin_check(h, rim, seconds=3.0, window=1.0, rows=None):
    h.place()
    env = h.watch()
    h.wheel_targets(-rim / RADIUS, rim / RADIUS)
    h.run(seconds - window)
    log, fs = Loads(), FrictionStats(h)
    acc, rims = mean_rims(h, int(round(window / h.dt)), log, fs, rows, int(round(TRACE_TAIL / h.dt)))
    h.unwatch()
    return acc[2], rims, log.stats(), env, fs, acc


def drift_text(tw):
    """The base's mean forward and sideways speeds in a spin [m/s]."""
    return f"{tw[0]:+.4f}, {tw[1]:+.4f}"


def turn_check(h, v=1.0, w=1.0 / 3.0, seconds=8.0, window=2.0, rows=None):
    h.place()
    env = h.watch()
    h.command = (v, w)
    h.run(seconds - window)
    log, fs = Loads(), FrictionStats(h)
    acc, rims = mean_rims(h, int(round(window / h.dt)), log, fs, rows, int(round(TRACE_TAIL / h.dt)))
    h.unwatch()
    targets = {w: h.target[w] * RADIUS for w in WHEELS}
    return acc, rims, log.stats(), env, targets, fs


def slide_check(h, v0=1.0, forward=False, rows=None):
    """A slide from v0, sideways, or forwards with the wheels' targets at
    zero: the distance until the speed falls below 1 mm/s (at most 5 s),
    v0^2 / (2 mu g), the largest rim speed, and the loads. Forwards, the
    drivers hold the wheels with at most 30 N m, which on asphalt hardly
    exceeds the torque of the friction at a wheel (0.8 x 196 N x 0.1651 m =
    25.9 N m): there the held wheels can turn, and the slide be longer than
    v0^2 / (2 mu g) without any fault of the friction."""
    if forward:
        h.place(speed=v0, spin=0.0)
    else:
        h.place(lateral=v0)
    env = h.watch()
    h.wheel_targets(0.0, 0.0)
    x0, y0, _, _ = h.pose()
    t, largest, log = 0.0, 0.0, Loads()
    while t < 5.0:
        h.step(1)
        t += h.dt
        u, v, om = h.body_velocity()
        rims = h.rim_speeds()
        ld = log.add(h.loads())
        largest = max(largest, max(abs(r) for r in rims.values()))
        if rows is not None:
            x, y, _, z = h.pose()
            rows.append([round(t, 6), math.hypot(x - x0, y - y0), u, v, om, z] + [rims[w] for w in WHEELS] +
                        [ld[w] for w in WHEELS] + [sum(ld.values())])
        if math.hypot(u, v) < 1e-3:
            break
    h.unwatch()
    x1, y1, _, _ = h.pose()
    return math.hypot(x1 - x0, y1 - y0), v0 * v0 / (2.0 * h.mu * G), largest, log.stats(), t, env


def creep_check(h, part=0.3):
    settle(h)
    env = h.watch()
    if not h.settled[0]:
        env.flag("not settled")
    y0 = h.pose()[1]
    h.set_push(part * h.mu * MASS * G)
    log = Loads()
    for _ in range(int(round(2.0 / h.dt))):
        h.step(1)
        log.add(h.loads())
    h.set_push(0.0)
    h.unwatch()
    return abs(h.pose()[1] - y0) / 2.0, log.stats(), env


def brake(h, v0=1.0, rows=None, run_up=3.0):
    """The stop from a commanded v0: the speed before, the distance, the time
    until the speed falls below 1 mm/s (at most 5 s), the largest slip."""
    h.place()
    env = h.watch()
    h.command = (v0, 0.0)
    h.run(run_up)
    x0 = h.pose()[0]
    u0 = h.body_velocity()[0]
    h.command = (0.0, 0.0)
    t, slip, log, stopped, mark = 0.0, 0.0, Loads(), False, None
    while t < 5.0:
        h.step(1)
        t += h.dt
        u = h.body_velocity()[0]
        rims = h.rim_speeds()
        slip = max(slip, u - sum(rims.values()) / 4.0)
        if rows is not None:
            ld = log.add(h.loads())
            rows.append([round(t, 6), h.pose()[0] - x0, u, h.pose()[3]] + [rims[w] for w in WHEELS] +
                        [ld[w] for w in WHEELS] + [sum(ld.values())])
        if t > 0.1 and abs(u) < 1e-3:
            stopped = True
            break
        if mark is None and t >= 4.0:
            mark = (h.pose()[0], t)
    h.unwatch()
    creep = None
    if not stopped and mark is not None and t > mark[1]:
        creep = (h.pose()[0] - mark[0]) / (t - mark[1])
    return u0, h.pose()[0] - x0, t, slip, (log.stats() if log.rows else None), stopped, env, creep


def rest_text(t, stopped, creep):
    """The time to rest of a stop, or what the robot still did."""
    if stopped:
        return f"{t:.3f}"
    return f"not at rest after {t:.1f}" + (f" ({creep * 1e3:.2f} mm/s over its last second)"
                                           if creep is not None else "")


def drive_window(h, run_up=3.0, seconds=2.0):
    """Driving at the command 1 m/s, after run_up: a trace of the steps of the
    window; the speed, the rims' speed, their scatter, the lateral speed's
    and the yaw rate's scatter, and the loads."""
    h.place()
    env = h.watch()
    h.command = (1.0, 0.0)
    h.run(run_up)
    n = int(round(seconds / h.dt))
    rows, log, fs = [], Loads(), FrictionStats(h)
    x0 = h.pose()[0]
    for i in range(n):
        h.step(1)
        u, v, om = h.body_velocity()
        rims = h.rim_speeds()
        ld = log.add(h.loads())
        fs.add((u, v, om), rims, ld, h.friction())
        rows.append([round((i + 1) * h.dt, 6), h.pose()[0] - x0, u, v, om, h.pose()[3]] +
                    [rims[w] for w in WHEELS] + [ld[w] for w in WHEELS] + [sum(ld.values())])
    h.unwatch()
    target = float(np.mean([h.target[w] for w in WHEELS])) * RADIUS
    a = np.asarray(rows, dtype=float)
    fl = a[:, 6]
    return dict(speed=float(a[:, 2].mean()), rim=float(a[:, 6:10].mean()), target=target, u_std=float(a[:, 2].std()),
                v_std=float(a[:, 3].std()), yaw_std=float(a[:, 4].std()),
                rim_std=float(fl.std()), rim_lag1=lag1(fl), loads=log.stats(), rows=rows, env=env, fs=fs)


TRACE_MOTION = (["t [s]", "distance [m]", "u [m/s]", "v [m/s]", "yaw rate [rad/s]", "base height [m]"] +
                [f"rim {w} [m/s]" for w in WHEELS] + LOAD_COLUMNS)
TRACE_REST = (["t [s]", "base height [m]", "roll [rad]", "pitch [rad]"] + LOAD_COLUMNS +
              [f"points {w}" for w in WHEELS])
TRACE_STOP = (["t [s]", "distance [m]", "u [m/s]", "base height [m]"] + [f"rim {w} [m/s]" for w in WHEELS] +
              LOAD_COLUMNS)


def trace_spin_header(h):
    """The columns of trace_row; the friction is the force on the wheel
    where the sign of PhysX's data is known, else as reported."""
    on = "on the wheel" if h.friction_sign is not None else "as reported"
    return (["t [s]", "u [m/s]", "v [m/s]", "yaw rate [rad/s]"] + [f"rim {w} [m/s]" for w in WHEELS] +
            [f"load {w} [N]" for w in WHEELS] +
            [f"friction {on}, {w} {a} [N]" for w in WHEELS for a in ("x", "y")] +
            [f"slip {w} {a} [m/s]" for w in WHEELS for a in ("x", "y")] +
            [f"joint torque {w} (PhysX) [N m]" for w in WHEELS] +
            [c for w in WHEELS for c in (f"anchors {w}", f"oldest anchor {w} [steps]")])


def describe_bins(bins):
    return "; ".join(f"{w} {s} {b['points']:.2f}, {b['x']:.4f}, {b['y']:.4f}, {b['load']:.1f}, "
                     f"{b['sep'] * 1e3:.3f}, {b['nz']:.3f}"
                     for (w, s), b in bins.items() if b["points"] > 0) or "none"


MOMENTS_KEY = ("loads at rest: their first moments about the base's origin, sum N x and sum N y, and those of "
               "the weight at the centre of mass [N m]")


def describe_moments(bins, h):
    """The balance that rigid contacts leave to the loads at rest: with two
    points of contact per cylinder, eight in all, only the total and the
    moments are set by statics, not the share of each wheel."""
    nx = sum(b["load"] * b["x"] for b in bins.values() if b["points"] > 0 and b["load"] > 0)
    ny = sum(b["load"] * b["y"] for b in bins.values() if b["points"] > 0 and b["load"] > 0)
    total = h.base_mass + sum(h.wheel_masses.values())
    com = [(h.base_mass * h.base_com[k] + sum(h.wheel_masses[w] * CENTRES[w][k] for w in WHEELS)) / total
           for k in (0, 1)]
    return f"{nx:+.2f}, {ny:+.2f}; {total * G * com[0]:+.2f}, {total * G * com[1]:+.2f}"


def side_text(speeds):
    """'left, right': the mean of each side's wheels [m/s]."""
    return (f"{0.5 * sum(speeds[w] for w in LEFT):.3f}, {0.5 * sum(speeds[w] for w in RIGHT):.3f}")


def points_per_wheel(bins):
    return ", ".join(f"{sum(bins[(w, s)]['points'] for s in SIDES):.2f}" for w in WHEELS)


WHEEL_INERTIA_KEY = "wheels' inertia about the axle (y) on the stage [kg m^2]"
SETUP_KEYS = ("ground", "wheel colliders", f"omni.physx {APPROXIMATE_CYLINDERS}", WHEEL_INERTIA_KEY,
              "scene: solver, friction type, CCD, stabilization, GPU dynamics, broadphase, time steps per second",
              "scene: friction correlation distance, friction offset threshold [m]",
              "articulation: sleep and stabilization thresholds, iterations; links' damping; contact reports",
              "material: compliant contacts, stiffness [N/m], damping [N s/m]")


def model_facts(h):
    """What the importer and PhysX made of the description, read from the
    first stage opened; the settings of that stage's setup are left to the
    setup's own line (setup_notes)."""
    out = {"read from the stage of the setup": h.setup.name}
    out.update({k: v for k, v in h.notes.items() if k not in SETUP_KEYS})
    attempt(out, "DOF names", lambda: ", ".join(h.dof_names))
    attempt(out, "link names", lambda: ", ".join(h.link_names))
    attempt(out, "link masses [kg]", lambda: ", ".join(
        f"{n} {m:.6g}" for n, m in zip(h.link_names, np.asarray(h.art.get_masses())[0])))
    attempt(out, "total mass [kg]", lambda: f"{float(np.asarray(h.art.get_masses())[0].sum()):.4f}")
    for what in ("stiffnesses", "dampings", "max_forces", "max_velocities"):
        attempt(out, f"DOF {what} (tensor API)", lambda what=what: ", ".join(
            f"{n} {x:.6g}" for n, x in zip(h.dof_names, np.asarray(getattr(h.art, f"get_dof_{what}")())[0])))
    return out


def setup_notes(h):
    """The settings of the setup as the stage has them."""
    out = {k: h.notes[k] for k in SETUP_KEYS if k in h.notes}
    rows = []
    try:
        for w in WHEELS:
            for p in h.wheel_colliders[w]:
                api = PhysxSchema.PhysxCollisionAPI(p)
                rows.append(f"{w} {api.GetContactOffsetAttr().Get():g}, {api.GetRestOffsetAttr().Get():g}")
    except Exception as exc:                       # noqa: BLE001
        rows = [f"not read ({exc!r})"]
    out["wheel colliders' contact and rest offsets (authored) [m]"] = " / ".join(rows) or "not authored"
    out["physics time step, as the simulation manager reports it [s]"] = f"{h.physics.scene_dt}"
    out["physics stepped through"] = h.physics.mode
    try:
        ine = np.asarray(h.art.get_inertias(), dtype=float).reshape(len(h.link_names), 9)
        out["wheels' inertia about the axle (y) as PhysX has it (tensor API) [kg m^2]"] = ", ".join(
            f"{w} {ine[h.link_of[w], 4]:.5f}" for w in WHEELS if h.link_of[w] is not None)
    except Exception as exc:                       # noqa: BLE001
        out["wheels' inertia about the axle (y) as PhysX has it (tensor API) [kg m^2]"] = f"not read ({exc!r})"
    try:
        mx, my = h.friction_model().effective_masses("front_left")
        out["the contact's effective mass at a wheel's foot along and across the wheel, the two-row law's [kg]"] = (
            f"{mx:.2f}, {my:.2f} (ratio {my / mx:.1f})")
    except Exception as exc:                       # noqa: BLE001
        out["the contact's effective mass at a wheel's foot, the two-row law's"] = f"not computed ({exc!r})"
    return "; ".join(f"{k}: {v}" for k, v in out.items())


# ----------------------------------------------------------------------------
# The screen
# ----------------------------------------------------------------------------
def screen_one(s, out):
    """Short versions of the diagnostics for one setup, on asphalt."""
    name, mu = SCREEN_GROUND
    h = IsaacHusky(mu, s)
    marks = {}

    def keep(key, env):
        if env is not None and env.level > 0:
            marks[key] = env.level
        judged(f"{s.name}: {key}", env, SCREEN_GROUP)

    try:
        if not FACTS.get(MODEL_GROUP):
            FACTS.setdefault(MODEL_GROUP, {}).update(model_facts(h))
        out[f"{s.name}: setup"] = s.text() + "; " + setup_notes(h)
        cs = None
        attempt(out, f"{s.name}: {FRICTION_CHECK_KEY}", lambda: friction_check_text(friction_check(h)))

        def rest():
            nonlocal cs
            st, bins, env, _ = rest_window(h)
            keep("rest", env)
            try:
                cs = reference_contacts(st, bins)[0]
            except RuntimeError:
                cs = None
            return (f"{settled_text(h)}; {describe_loads(st)}; single steps: {describe_scatter(st)}; "
                    f"contact points per step and wheel {points_per_wheel(bins)}; {env.text()}")
        attempt(out, f"{s.name}: rest: settling; loads, mean over {WINDOW:g} s [N]; single steps: total least, "
                     "largest, std [N], correlation, all wheels down, none; contact points per step and wheel; "
                     + ENVELOPE_KEY, rest)

        def dr():
            res = drive_window(h, run_up=2.5, seconds=1.0)
            write_csv(f"trace-screen-{s.name}-drive.csv", TRACE_MOTION, res["rows"])
            env = res["env"]
            keep("drive", env)
            return (f"{res['speed']:.4f}, {res['rim']:.4f}, {res['target']:.4f}, {pct(res['rim'] / res['speed'] - 1)}; "
                    f"{res['v_std']:.4f}, {res['yaw_std']:.4f}; {share(res['loads']['none'])}; "
                    f"{env.torque_text()}; {res['fs'].text()}; {env.text()}")
        attempt(out, f"{s.name}: drive at 1 m/s: speed, rims, the rims' target [m/s], rims ahead by; std of the "
                     "lateral speed [m/s] and of the yaw rate [rad/s]; steps without load; drivers' torque; "
                     "friction against slip; " + ENVELOPE_KEY, dr)

        def sp():
            om, rims, _, env, fs, tw = spin_check(h, 0.98)
            keep("spin", env)
            ref, q = "Coulomb: no contacts at rest, -", None
            if cs:
                q, residual, sticking = coulomb_twist(cs, rims, mu)
                ref = reference_text(q, residual, sticking, om, "yaw")
            mean_rim = sum(abs(r) for r in rims.values()) / 4.0
            return (f"{om:.4f}, {ref}; {model_text(h, cs, rims, tw, q, 'yaw')}; {drift_text(tw)}; "
                    f"{mean_rim:.3f}, 0.980; {env.torque_text()}; {fs.text()}; {env.text()}")
        attempt(out, f"{s.name}: spin, rims at 0.98 m/s: yaw rate, Coulomb [rad/s], difference; {LAW_KEY} [rad/s]; "
                     "drift: the base's forward and sideways speeds [m/s]; mean rim speed, "
                     "its target [m/s]; drivers' torque; friction against slip; " + ENVELOPE_KEY, sp)

        def tu():
            acc, rims, _, env, tg, fs = turn_check(h, seconds=5.0, window=1.0)
            keep("turn", env)
            radius = acc[0] / acc[2] if acc[2] else float("inf")
            ref, q = "Coulomb: no contacts at rest, -", None
            if cs:
                q, residual, sticking = coulomb_twist(cs, rims, mu)
                ref = reference_text(q, residual, sticking, radius, "radius")
                if (abs(q[2]) > 1e-12 and abs(q[0]) > 1e-12 and math.isfinite(radius) and residual <= 1e-6
                        and not sticking):
                    TURNS[s.name] = radius / (q[0] / q[2]) - 1.0
            return (f"{radius:.3f}, {ref}; {model_text(h, cs, rims, acc, q, 'radius')}; {side_text(rims)}; "
                    f"{side_text(tg)}; {env.torque_text()}; {fs.text()}; {env.text()}")
        attempt(out, f"{s.name}: turn at 1 m/s and 1/3 rad/s: radius, Coulomb [m], difference; {LAW_KEY} [m]; "
                     "rims left, right [m/s]; their targets [m/s]; drivers' torque; friction against slip; "
                     + ENVELOPE_KEY, tu)

        def sl():
            dist, ideal, largest, _, t, env = slide_check(h)
            keep("sideways slide", env)
            return f"{dist:.4f}, {ideal:.4f}, {pct(dist / ideal - 1)}; {env.text()}"
        attempt(out, f"{s.name}: sideways slide from 1 m/s: distance, v^2/(2 mu g) [m], difference; "
                     + ENVELOPE_KEY, sl)

        def br():
            u0, dist, t, slip, _, stopped, env, creep = brake(h)
            keep("stop", env)
            return f"{u0:.4f}, {dist:.4f}, {rest_text(t, stopped, creep)}; {env.text()}"
        attempt(out, f"{s.name}: stop from the command at 1 m/s: speed before [m/s], distance [m], time to rest "
                     "[s]; " + ENVELOPE_KEY, br)
        failed = [k.split(": ")[1] for k, v in out.items() if k.startswith(f"{s.name}:") and str(v).startswith("FAILED")
                  and not k.startswith(f"{s.name}: friction data")]
        level = max([2 if failed or h.settled is None else 0] + list(marks.values()))
        VERDICTS[s.name] = level
        reasons = ([f"{k} {LEVELS[v]}" for k, v in marks.items()] + [f"{k} failed" for k in failed])
        out[f"{s.name}: verdict"] = LEVELS[level] + (": " + ", ".join(reasons) if reasons else "")
    finally:
        h.close()


def run_screen(out):
    for s in SCREEN:
        say(f"  screen: {s.name}")
        try:
            screen_one(s, out)
        except Exception as exc:                   # noqa: BLE001
            out[f"{s.name}: verdict"] = f"FAILED: {exc!r}"
            VERDICTS[s.name] = 2
            ERRORS.append((f"screen {s.name}", traceback.format_exc()))
            say(f"  failed: screen {s.name}: {exc!r}")
        write_outputs()


# ----------------------------------------------------------------------------
# The full runs
# ----------------------------------------------------------------------------
def ground_facts(s, name, mu, first):
    gname = f"{s.name}: {name}, mu {mu:g}"

    def run(out):
        h = IsaacHusky(mu, s)
        tag = f"{s.name}-{slug(name)}"

        def ok(key, env):
            judged(key, env, gname)
            return env.text()

        try:
            if not FACTS.get(MODEL_GROUP):
                FACTS.setdefault(MODEL_GROUP, {}).update(model_facts(h))
            if first:
                out["setup"] = s.text() + "; " + setup_notes(h)
            attempt(out, FRICTION_CHECK_KEY, lambda: friction_check_text(friction_check(h)))
            rest = {}

            def at_rest():
                st, bins, env, rows = rest_window(h)
                rest.update(st=st, bins=bins, env=env)
                write_csv(f"trace-{tag}-rest.csv", TRACE_REST, rows)
                return describe_loads(st) + f"; {settled_text(h)}; " + ok("rest", env)
            attempt(out, f"loads at rest, mean over {WINDOW:g} s, front and rear, left then right [N]; settling; "
                         + ENVELOPE_KEY, at_rest)
            if "st" in rest:
                st, bins = rest["st"], rest["bins"]
                out["loads of single steps at rest: total least, largest, std [N]; correlation of successive "
                    "totals; all wheels down, none"] = describe_scatter(st)
                out["contacts at rest, per wheel and side of its centre plane: points per step; centroid x, y "
                    "[m]; mean load [N]; mean separation [mm]; normal z"] = describe_bins(bins)
                out[MOMENTS_KEY] = describe_moments(bins, h)
            ref = {}
            if "st" in rest:
                attempt(ref, "contacts", lambda: reference_contacts(rest["st"], rest["bins"]))
            cs = ref["contacts"][0] if isinstance(ref.get("contacts"), tuple) else None
            out["Coulomb reference from"] = ref["contacts"][1] if cs else ref.get("contacts", "no loads at rest")

            for rim in (0.49, 0.98):
                def sp(rim=rim):
                    rows = []
                    om, rims, st, env, fs, tw = spin_check(h, rim, rows=rows)
                    write_csv(f"trace-{tag}-spin-{rim:g}.csv", trace_spin_header(h), rows)
                    mean_rim = sum(abs(r) for r in rims.values()) / 4.0
                    ref_text, q = "Coulomb: no contacts at rest, -", None
                    if cs:
                        q, residual, sticking = coulomb_twist(cs, rims, h.mu)
                        ref_text = reference_text(q, residual, sticking, om, "yaw")
                    return (f"{om:.4f}, {ref_text}, {share(st['all_down'])}; "
                            f"{model_text(h, cs, rims, tw, q, 'yaw')}; {drift_text(tw)}; {mean_rim:.3f}, {rim:.3f}; "
                            f"{env.torque_text()}; {fs.text()}; " + ok(f"spin, rims at {rim} m/s", env))
                attempt(out, f"spin, rims at {rim} m/s: yaw rate, Coulomb [rad/s], difference, all wheels down; "
                             f"{LAW_KEY} [rad/s]; drift: the base's forward and sideways speeds [m/s]; "
                             "mean rim speed, its target [m/s]; drivers' torque; friction against slip; "
                             + ENVELOPE_KEY, sp)

            def tu():
                rows = []
                acc, rims, st, env, tg, fs = turn_check(h, rows=rows)
                write_csv(f"trace-{tag}-turn.csv", trace_spin_header(h), rows)
                radius = acc[0] / acc[2] if acc[2] else float("inf")
                ref_text, q = "Coulomb: no contacts at rest, -", None
                if cs:
                    q, residual, sticking = coulomb_twist(cs, rims, h.mu)
                    ref_text = reference_text(q, residual, sticking, radius, "radius")
                return (f"{radius:.3f}, {ref_text}, {share(st['all_down'])}; "
                        f"{model_text(h, cs, rims, acc, q, 'radius')}; {side_text(rims)}; {side_text(tg)}; "
                        f"{env.torque_text()}; {fs.text()}; " + ok("turn", env))
            attempt(out, "turn at 1 m/s and 1/3 rad/s: radius, Coulomb [m], difference, all wheels down; "
                         f"{LAW_KEY} [m]; "
                         "rims left, right [m/s]; their targets [m/s]; drivers' torque; friction against slip; "
                         + ENVELOPE_KEY, tu)

            def dr():
                res = drive_window(h)
                write_csv(f"trace-{tag}-drive.csv", TRACE_MOTION, res["rows"])
                st, env = res["loads"], res["env"]
                return (f"{res['speed']:.4f}, {res['rim']:.4f}, {res['target']:.4f}, "
                        f"{pct(res['rim'] / res['speed'] - 1)}; "
                        f"{res['u_std']:.4f}, {res['v_std']:.4f}, {res['yaw_std']:.4f}; "
                        f"{res['rim_std']:.4f}, {corr(res['rim_lag1'])}; " +
                        describe_loads(st) + "; " + describe_scatter(st) + f"; {env.torque_text()}; " +
                        f"{res['fs'].text()}; " + ok("driving at 1 m/s", env))
            attempt(out, "driving at the command 1 m/s, 2 s after 3 s: speed, rims, the rims' target [m/s], rims "
                         "ahead by; std of "
                         "the speed, the lateral speed [m/s] and the yaw rate [rad/s]; front left rim's std [m/s], "
                         "correlation of its successive values; mean loads [N]; single steps: total least, largest, "
                         "std [N]; correlation; all wheels down, none; drivers' torque; friction against slip; "
                         + ENVELOPE_KEY, dr)

            for forward in (False, True):
                def sl(forward=forward):
                    rows = []
                    dist, ideal, largest, st, t, env = slide_check(h, forward=forward, rows=rows)
                    write_csv(f"trace-{tag}-slide-{'forward' if forward else 'sideways'}.csv", TRACE_MOTION, rows)
                    return (f"{dist:.4f}, {ideal:.4f}, {pct(dist / ideal - 1)}, {share(st['all_down'])}; "
                            f"{largest:.3f}; " + ok("forward slide" if forward else "sideways slide", env))
                attempt(out, ("forward slide from 1 m/s with the wheels held" if forward else
                              "sideways slide from 1 m/s") +
                        ": distance, v^2/(2 mu g) [m], difference, all wheels down; largest rim speed [m/s]; "
                        + ENVELOPE_KEY, sl)

            def cr():
                speed, st, env = creep_check(h)
                return f"{speed * 1e3:.2f}, {share(st['all_down'])}; {settled_text(h)}; " + ok("creep", env)
            attempt(out, "creep under a sideways push of 30 % of mu m g [mm/s], all wheels down; settling; "
                         + ENVELOPE_KEY, cr)

            def br():
                rows = []
                u0, dist, t, slip, st, stopped, env, creep = brake(h, rows=rows)
                write_csv(f"trace-{tag}-stop.csv", TRACE_STOP, rows)
                return (f"{u0:.4f}, {dist:.4f}, {rest_text(t, stopped, creep)}, {slip:.3f}, "
                        f"{share(st['all_down'])}; " + ok("stop", env))
            attempt(out, "stop from the command at 1 m/s: speed before [m/s], distance [m], time to rest [s], "
                         "largest slip [m/s], all wheels down; " + ENVELOPE_KEY, br)

            def sc():
                h.place()
                env = h.watch()
                h.command = (0.0, 1.0)
                h.run(3.0)
                log, fs = Loads(), FrictionStats(h)
                acc, rims = mean_rims(h, int(round(1.0 / h.dt)), log, fs)
                h.unwatch()
                om = acc[2]
                mean_rim = sum(abs(r) for r in rims.values()) / 4.0
                target = sum(abs(h.target[w]) for w in WHEELS) / 4.0 * RADIUS
                w_eff = f"{h.drive.separation / om:.4f}" if om else "inf"
                ref_text, q = "Coulomb: no contacts at rest, -", None
                if cs:
                    q, residual, sticking = coulomb_twist(cs, rims, h.mu)
                    ref_text = reference_text(q, residual, sticking, om, "yaw")
                return (f"{om:.4f}, {w_eff}, {share(log.stats()['all_down'])}; {ref_text}; "
                        f"{model_text(h, cs, rims, acc, q, 'yaw')}; {drift_text(acc)}; {mean_rim:.3f}, {target:.3f}; "
                        f"{env.torque_text()}; {fs.text()}; " + ok("spin commanded at 1 rad/s", env))
            attempt(out, "spin commanded at 1 rad/s: yaw rate [rad/s], W_eff [m], all wheels down; Coulomb for "
                         f"those rims [rad/s], difference; {LAW_KEY} [rad/s]; drift: the base's forward and sideways "
                         "speeds [m/s]; mean rim speed, its target [m/s]; drivers' torque; friction against slip; "
                         + ENVELOPE_KEY, sc)
            if name == "asphalt":
                def od():
                    h.place()
                    env = h.watch()
                    h.command = (1.0, 0.0)
                    h.run(2.0)
                    x0, o0, w0 = h.pose()[0], h.drive.x, h.wraps
                    h.run(10.0)
                    h.unwatch()
                    true, odo = h.pose()[0] - x0, h.drive.x - o0
                    return (f"{true / 10.0:.4f}; {true:.4f}, {odo:.4f}, {pct(1 - odo / true)}; {h.wraps - w0}; "
                            + ok("odometry", env))
                attempt(out, "speed at the command 1 m/s [m/s]; 10 s: distance, odometry [m], short by; wraps of "
                             "the wheels' positions removed; " + ENVELOPE_KEY, od)
            attempt(out, "session layer", lambda: (h.export(os.path.join(
                OUT, f"stage-{tag}.usda")), f"stage-{tag}.usda")[1])
        finally:
            h.close()
    return gname, run


TIME_STEPS = (0.001, 0.0005, 0.00025)


def timestep_facts(s):
    """The settling, the spin with the rims at 0.98 m/s and the turn against
    their Coulomb references, and the stop, at time steps of 1, 0.5 and
    0.25 ms on every ground of the full runs (husky_mujoco.py repeats the
    stop, the spin and the turn); the check of the friction data at each;
    a trace of every stop."""
    gname = f"time steps: {s.name}"

    def run(out):
        for name, mu in GROUNDS:
            vals = {k: [] for k in ("check", "settled", "spin", "radius", "turn", "angle", "law", "law angle",
                                    "stop")}

            def put(key, fn, h, label):
                try:
                    vals[key].append(fn())
                except Exception as exc:           # noqa: BLE001
                    vals[key].append(f"FAILED {exc!r}")
                    ERRORS.append((f"{gname}: {name} at {label}: {key}", traceback.format_exc()))

            def mark(env, label, what):
                judged(f"{name}, {what} at {label}", env, gname)
                return "" if env.valid else f" ({LEVELS[env.level]})"

            for dt in TIME_STEPS:
                label = f"{dt * 1e3:g} ms"
                h = None
                try:
                    h = IsaacHusky(mu, s, dt)
                except Exception as exc:           # noqa: BLE001
                    for key in vals:
                        vals[key].append(f"FAILED {exc!r}")
                    ERRORS.append((f"{gname}: {name} at {label}", traceback.format_exc()))
                    continue
                try:
                    put("check", lambda: friction_check_text(friction_check(h)).split(";")[0], h, label)
                    rest = {}

                    def settled():
                        st, bins, env, _ = rest_window(h)
                        try:
                            rest["cs"] = reference_contacts(st, bins)[0]
                        except RuntimeError:
                            rest["cs"] = None
                        return (f"{h.settled[1]:.2f}" if h.settled[0] else "not settled") + mark(env, label, "rest")
                    put("settled", settled, h, label)
                    cs = rest.get("cs")

                    def spin():
                        om, rims, _, env, _, _ = spin_check(h, 0.98)
                        if not cs:
                            return "Coulomb: no contacts at rest" + mark(env, label, "spin")
                        q = coulomb_twist(cs, rims, mu)[0]
                        return (pct(om / q[2] - 1) if abs(q[2]) > 1e-12 else "-") + mark(env, label, "spin")
                    put("spin", spin, h, label)

                    def turn():
                        acc, rims, _, env, _, fs = turn_check(h)
                        radius = acc[0] / acc[2] if acc[2] else float("inf")
                        r_text = f"{radius:.3f}" + mark(env, label, "turn")
                        a_text = f"{np.median(fs.angles):.1f}" if fs.angles else "-"
                        vals["radius"].append(r_text)
                        vals["angle"].append(a_text)
                        vals["law angle"].append(fs.law_median())
                        law = "-"
                        if cs and h.setup.solver == "TGS":
                            try:
                                qc = coulomb_twist(cs, rims, mu)[0]
                                (ql, rl), _ = two_row_states(cs, rims, mu, h.friction_model(), acc, qc, "radius")
                                law = f"{law_value(ql, 'radius'):.3f}" + (f" (residual {rl:.1e})" if rl > 1e-6 else "")
                            except Exception as exc:   # noqa: BLE001
                                law = f"failed ({exc!r})"
                                report_once("the two-row law's steady motion (the measurements went on without it)",
                                            exc)
                        vals["law"].append(law)
                        if not cs:
                            return "Coulomb: no contacts at rest"
                        q = coulomb_twist(cs, rims, mu)[0]
                        return pct(radius / (q[0] / q[2]) - 1) if abs(q[2]) > 1e-12 and abs(q[0]) > 1e-12 else "-"
                    n_radius = len(vals["radius"])
                    put("turn", turn, h, label)
                    while len(vals["radius"]) < n_radius + 1:
                        vals["radius"].append("FAILED")
                        vals["angle"].append("-")
                    while len(vals["law"]) < n_radius + 1:
                        vals["law"].append("-")
                    while len(vals["law angle"]) < n_radius + 1:
                        vals["law angle"].append("-")

                    def stop():
                        rows = []
                        u0, dist, t, slip, _, stopped, env, creep = brake(h, rows=rows)
                        write_csv(f"trace-{s.name}-{slug(name)}-stop-{dt * 1e3:g}ms.csv", TRACE_STOP, rows)
                        text = f"{dist:.4f}"
                        if not stopped:
                            text += " (" + rest_text(t, stopped, creep) + ")"
                        return text + mark(env, label, "stop")
                    put("stop", stop, h, label)
                finally:
                    h.close()
            out[f"{name}: friction data's check at 1, 0.5, 0.25 ms (+1: the forces on the wheels)"] = ", ".join(
                vals["check"])
            out[f"{name}: settled after, at 1, 0.5, 0.25 ms [s]"] = ", ".join(vals["settled"])
            out[f"{name}: spin at 0.98 m/s against Coulomb at 1, 0.5, 0.25 ms"] = ", ".join(vals["spin"])
            out[f"{name}: turn at 1, 0.5, 0.25 ms: radius [m]; against Coulomb"] = (
                ", ".join(vals["radius"]) + "; " + ", ".join(vals["turn"]))
            out[f"{name}: turn: friction against slip, median angle at 1, 0.5, 0.25 ms [deg]"] = ", ".join(vals["angle"])
            out[f"{name}: turn: the two-row law's radius for those rims and contacts [m]; friction against the "
                "law, median angle [deg], at 1, 0.5, 0.25 ms"] = (", ".join(vals["law"]) + "; " +
                                                                  ", ".join(vals["law angle"]))
            out[f"{name}: stop from 1 m/s, distance at 1, 0.5, 0.25 ms [m]"] = ", ".join(vals["stop"])
            write_outputs()
    return run


def validity_facts(out):
    out["limits, valid up to and marginal up to: at rest, the base from the height of touching round wheels "
        "[mm], the spread of its height [mm], its tilt [deg]; in motion, its excursion from its height at "
        "rest [mm], its tilt [deg]"] = (
        "; ".join(f"{a:g} and {b:g}" for a, b in REST_LIMITS.values()) + "; " +
        "; ".join(f"{a:g} and {b:g}" for a, b in MOTION_LIMITS.values()))
    if VERDICTS:
        for lvl, word in enumerate(LEVELS):
            out[f"screen: {word} setups"] = ", ".join(n for n, v in VERDICTS.items() if v == lvl) or "none"
    else:
        out["screen"] = "none screened"
    out["full runs"] = ", ".join(RAN) or "none ran"
    if "name" in FRICTION_READER:
        out["PhysX's friction read with"] = FRICTION_READER["name"] + f" ({FRICTION_READER['why']})"
    elif FRICTION_READER:
        out["PhysX's friction read with"] = (
            f"get_friction_data (get_friction_contact_data returned the same arrays, but at no step with "
            f"friction of {FRICTION_SAME_MIN:g} N or more)")
    by_group = {}
    for grp, key, text, lvl in INVALID:
        if grp == SCREEN_GROUP:
            continue
        by_group.setdefault((grp, LEVELS[lvl]), []).append(f"{key.split(':')[0]} ({text})")
    for (grp, word), items in by_group.items():
        out[f"{grp}: {word} diagnostics"] = "; ".join(items)
    if RAN and not by_group:
        out["full runs: marginal or invalid diagnostics"] = "none"


# ----------------------------------------------------------------------------
# The record
# ----------------------------------------------------------------------------
def ext_id(name):
    try:
        return omni.kit.app.get_app().get_extension_manager().get_enabled_extension_id(name) or "-"
    except Exception:                              # noqa: BLE001
        return "-"


def record_facts(out):
    try:
        from isaacsim.core.version import get_version
        v = get_version()
        out["Isaac Sim"] = v[0] + (f"-{v[1]}" if v[1] else "")
    except Exception as exc:                       # noqa: BLE001
        out["Isaac Sim"] = f"unknown ({exc!r})"
    try:
        out["Kit"] = omni.kit.app.get_app().get_build_version()
    except Exception:                              # noqa: BLE001
        out["Kit"] = "-"
    for name in ("omni.physx", "omni.physics.tensors", "isaacsim.core.simulation_manager",
                 "isaacsim.asset.importer.urdf"):
        out[f"extension {name}"] = ext_id(name)
    out["Python, NumPy"] = f"{platform.python_version()}, {np.__version__}"
    out["OS, machine"] = f"{platform.system()} {platform.release()}, {platform.machine()}"
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)
        out["GPU"] = q.stdout.strip().replace("\n", "; ") or "not reported"
    except Exception:                              # noqa: BLE001
        out["GPU"] = "not reported"
    out["script"] = f"isaac_husky.py, version {VERSION}"
    out["command line"] = " ".join(sys.argv[1:]) or "(none)"
    out["setups screened"] = ", ".join(s.name for s in SCREEN) or "none"
    out["setups run in full"] = ", ".join(s.name for s in FULL) or "none"
    out["drivers: gain [N m s/rad], integral time [s], limit [N m]"] = f"{KV:g}, {TI:g}, {TORQUE:g}"
    out["description"] = asset.SOURCE


def write_record():
    rec = FACTS.get("record", {})
    mdl = FACTS.get(MODEL_GROUP, {})
    lines = [
        "Record of the experiment of Section 13.1 in Isaac Sim: the Husky A300 with PhysX",
        "",
        "model and asset provenance",
        "  " + rec.get("description", "-"),
        "  the URDF of husky/asset.py (corrected track), imported with isaacsim.asset.importer.urdf: " +
        FACTS.get("the import", {}).get("importer config", "-"),
        "  masses set from the description: " + str(mdl.get("masses set from the description", "-")) +
        "; inertias with the link's principal axes",
        "simulator and solver",
        f"  Isaac Sim {rec.get('Isaac Sim', '-')}, Kit {rec.get('Kit', '-')}, {rec.get('extension omni.physx', '-')}",
        "  the scene as in Chapter 11 (CPU dynamics, MBP broadphase, patch friction, no CCD, no stabilization), "
        "the solver of each setup; the articulation with no sleeping, no stabilization "
        "threshold, 32 position iterations and the setup's velocity iterations, no self-collision, no damping of "
        "the links",
        "  setups (name: wheels' colliders, ground, solver, contact offset, velocity iterations, time step, and "
        "what else the setup changes):",
    ]
    for s in dict.fromkeys(SCREEN + FULL):
        lines.append(f"    {s.name}: {s.text()}")
    lines += [
        "  screened: " + rec.get("setups screened", "-") + "; run in full: " + rec.get("setups run in full", "-"),
        "  script, command line: " + rec.get("script", "-") + "; " + rec.get("command line", "-"),
        "time steps",
        "  the setup's; the drive controller at 50 Hz; the drivers' integral loop at every step",
        "actuators and sensors",
        "  PhysX angular velocity drives, damping the drivers' gain per degree, maximum force the limit; " +
        rec.get("drivers: gain [N m s/rad], integral time [s], limit [N m]", "-"),
        "  wheels' positions unwrapped from PhysX's wrap at +-2 pi; loads and contacts at rest averaged over "
        f"{WINDOW:g} s; the drivers' torque as PhysX reports the joints' torques and from the drive law with the "
        "velocities after each step; PhysX's friction data at the anchors against the slip there, their sign "
        "from the momentum of a sideways slide",
        "  references: the Coulomb steady state (husky/coulomb.py, by iteratively reweighted least squares) and, "
        "under TGS, the two-row law of PhysX's patch friction (TwoRowModel, from the description's masses), both "
        "quasi-static, for the contacts at rest and the rims measured",
        "  what the script printed, with Python's warnings, in console.txt",
        "frames and units",
        "  SI units; world z up; base x forward, y left",
        "uncertainty bounds",
        "  frictions of the grounds: " + ", ".join(f"{n} {m:g}" for n, m in GROUNDS),
        "  validity of a diagnostic (valid, or marginal, up to): in motion, the base's excursion from its height "
        f"at rest {MOTION_LIMITS['excursion'][0]:g} ({MOTION_LIMITS['excursion'][1]:g}) mm and its tilt "
        f"{MOTION_LIMITS['tilt'][0]:g} ({MOTION_LIMITS['tilt'][1]:g}) deg; at rest, after settling, the base "
        f"{REST_LIMITS['offset'][0]:g} ({REST_LIMITS['offset'][1]:g}) mm from the height of touching round "
        f"wheels, the spread of its height {REST_LIMITS['spread'][0]:g} ({REST_LIMITS['spread'][1]:g}) mm and "
        f"its tilt {REST_LIMITS['tilt'][0]:g} ({REST_LIMITS['tilt'][1]:g}) deg; beyond, INVALID, and not run "
        "in full",
        "constraints, terminal behaviour, split, seeds",
        "  as in results/record.txt of husky_mujoco.py",
        "hardware",
        f"  {rec.get('OS, machine', '-')}; GPU {rec.get('GPU', '-')}",
        "guarantee status",
        "  none: diagnostics only",
    ]
    with open(os.path.join(OUT, "record.txt"), "w", encoding="utf-8") as fh:
        fh.write(scrub("\n".join(lines) + "\n"))


def main():
    os.makedirs(OUT, exist_ok=True)
    try:
        CONSOLE.append(open(os.path.join(OUT, "console.txt"), "w", encoding="utf-8"))
    except OSError as exc:
        print(f"console.txt cannot be written: {exc!r}", flush=True)
    say(f"isaac_husky.py version {VERSION}; command line: {' '.join(sys.argv[1:]) or '(none)'}")
    say(f"results in {os.path.relpath(OUT, HERE)}")
    group("record", record_facts)
    group("the import", import_robot)
    FACTS["record"]["setups run in full"] = "none"
    if "usd" not in IMPORT:
        say("the import failed; nothing else can run")
    else:
        FACTS.setdefault(MODEL_GROUP, {})          # filled by the first stage opened
        if SCREEN:
            group(SCREEN_GROUP, run_screen)
        runs = []
        for s in FULL:
            if VERDICTS.get(s.name, 0) == 2 and not args.force_full:
                FACTS.setdefault("full runs skipped", {})[s.name] = "the screen found it INVALID (--force-full runs it)"
                continue
            runs.append(s)
        if not args.no_auto:
            for s in SCREEN:
                diff = TURNS.get(s.name)
                if (s in runs or VERDICTS.get(s.name) != 0 or diff is None or abs(diff) > TURN_CLOSE
                        or s.wheel_inertia != 1.0):
                    continue
                runs.append(s)
                FACTS.setdefault("full runs added", {})[s.name] = (
                    f"its turn in the screen was within {TURN_CLOSE:.0%} of the Coulomb reference ({pct(diff)}); "
                    "--no-auto leaves it out")
        RAN.extend(s.name for s in runs)
        FACTS["record"]["setups run in full"] = ", ".join(RAN) or "none"
        for i, s in enumerate(runs):
            for j, (name, mu) in enumerate(GROUNDS):
                gname, fn = ground_facts(s, name, mu, first=(j == 0))
                group(gname, fn)
        if runs and not args.quick:
            group(f"time steps: {runs[0].name}", timestep_facts(runs[0]))
    group("validity", validity_facts)
    write_record()
    write_outputs()
    say(f"done; {len(ERRORS) - len(ONCE)} step(s) failed; {len(ONCE)} of PhysX's data or of the two-row law "
        f"unavailable (errors.txt); {len(INVALID)} marginal or invalid diagnostic(s); results in "
        f"{os.path.relpath(OUT, HERE)}")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        to_console(traceback.format_exc())
        raise
    finally:
        for _fh in CONSOLE:
            _fh.close()
        APP.close()
