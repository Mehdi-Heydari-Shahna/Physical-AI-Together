"""
Two of Section 14.1's onset runs of the PC138US-8 as a video, and its
stills.

Physical AI, Together, Chapter 14 (Section 14.1).

The script runs excavator.model.onset() for the entry that Section 14.1
quotes, the hook at the ground and the reach of 4.6 m over the front, with
the book's modules unchanged, and records every run that it simulates at 30
frames per second of simulated time (render.shapes.Recorder, which only
copies the state after each of MuJoCo's steps), with the forces on the four
supports and the force at the lift point. It draws the first two: the runs
that check the bisection's bracket, at 98 and 102 % of the static tipping
load. In each, the machine stands on its four supports with its front fixed
in the lifting pose and settles for 2 s without a load; a downward force at
the lift point, the load's weight, then rises linearly over 5 s to its
share of the static tipping load and is held. At 98 % the machine stands;
at 102 % it tips over the front tipping line. The drawing of the second run
stops when the undercarriage leans 15 degrees: the simulated machine has no
contacts but its four supports, and keeps turning about the front line
through the ground.

The machine is drawn on the model's bodies (the undercarriage on its free
joint, the upper structure, the boom, the arm and the bucket fixed in the
pose) to dimensions of the brochure where it gives them and of ours where
it does not: the model has two boxes for its tracks, a box for its upper
structure and one for its counterweight, and capsules for its boom, arm and
bucket, none of them drawn. The cylinders are drawn at the model's lengths
in this pose (front.boom_length, front.arm_length) between pins placed by
us. The load, which the simulation applies as a force at the lift point
(its inertia left out), is drawn as a test bag that fills with water as the
force rises, hanging below the lift point in a pit, since the hook is at
the ground; the pit, its shoring, the yard and everything else are
scenery. The forces on the supports are drawn as arrows beside the tracks
and the front tipping line on the ground. The drawing simulates nothing.

    python excavator_video.py            # writes results/video/
    python excavator_video.py --quick
    python excavator_video.py --export DIR   # the scene and frames for the
                                             # GPU renderer (render/cycles)

Requires MuJoCo 3.16.0, NumPy, SciPy and Pillow, EGL (Mesa's or a GPU's)
and ffmpeg.

Writes results/video/excavator-onset.mp4, its stills, and
excavator-onset-frames.csv: each frame's run, simulated time, load at the
lift point, the four supports' forces and the undercarriage's lean, the
check that the drawing follows the record.
"""
import argparse
import csv
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

import mujoco                                                     # noqa: E402
import numpy as np                                                # noqa: E402

FPS = 30
G = 9.81                          # [m/s^2] as in excavator.model.onset
REACH, HEIGHT, SIDE = 4.6, 0.0, False          # the entry that Section 14.1 quotes
RUNS = (("stands", 0.98), ("tips", 1.02))      # the bracket's two runs, in onset's order
END_STANDS = 10.0                 # [s] the standing run is drawn to here (its hold lasts to 31 s)
STOP_LEAN = 15.0                  # [deg] the tipping run is drawn to this lean
HOLD = 1.5                        # [s] its last frame held
STILLS = (("stands", 6.0), ("stands", 10.0), ("tips", 8.5), ("tips", "end"))
SUPPORTS = ("support_fl", "support_fr", "support_rl", "support_rr")
PIT_X, PIT_Y, PIT_DEPTH = (2.8, 6.4), (-1.8, 1.8), 4.2         # [m] the drawn pit
WALL_SLOPE = math.radians(58.0)   # the pit's battered wall on the camera's side
WATER = 1000.0                    # [kg/m^3] the bag's water


def record():
    """onset() for the entry, recorded: for each of its runs, in order, the
    times [s], qpos, the supports' normal forces [N] and the force at the
    lift point [N]; and onset's result."""
    from excavator import capacity, model
    from render import shapes

    def extra(m, d):
        f = model.support_forces(m, d)
        return {"support": np.array([f.get(n, 0.0) for n in SUPPORTS]),
                "hook": np.array([-d.qfrc_applied[2]])}

    with shapes.Recorder(FPS, select=lambda m, d: m.nq == 7, extra=extra) as rec:
        fraction, analytic = model.onset(REACH, HEIGHT, SIDE)
    runs = []
    for r in rec.runs.values():
        runs.append(dict(t=np.asarray(r["t"]), qpos=np.asarray(r["qpos"]),
                         support=np.array([e["support"] for e in r["extra"]]),
                         hook=np.array([e["hook"][0] for e in r["extra"]])))
    return dict(runs=runs, fraction=fraction, analytic=analytic,
                pose=capacity.pose(REACH, HEIGHT))


def lean(q):
    """The undercarriage's lean from upright [deg], from its quaternion."""
    w, x, y, z = q[3:7] / np.linalg.norm(q[3:7])
    return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))


def quat_y(angle):
    """A rotation by angle about +y."""
    return [math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0]


def rot_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def capsule_between(body, name, a, b, radius, material):
    from render import shapes
    return shapes.visual(body, name, mujoco.mjtGeom.mjGEOM_CAPSULE, (radius,),
                         fromto=tuple(a) + tuple(b), material=material)


def box_between(body, name, a, b, half_y, half_z, material):
    """A box from a to b in the body's x-z plane (its length along a-b),
    half_y wide and half_z deep."""
    from render import shapes
    a, b = np.asarray(a, float), np.asarray(b, float)
    mid, d = 0.5 * (a + b), b - a
    length = float(np.linalg.norm(d))
    pitch = math.atan2(-d[2], d[0])
    return shapes.visual(body, name, mujoco.mjtGeom.mjGEOM_BOX, (0.5 * length, half_y, half_z),
                         pos=tuple(mid), quat=quat_y(pitch), material=material)


def mesh(spec, name, verts, faces):
    m = spec.add_mesh()
    m.name = name
    m.uservert = np.asarray(verts, float).ravel().tolist()
    m.userface = np.asarray(faces, int).ravel().tolist()
    return m.name


def prism(profile, half_width):
    """A prism of the closed polygon profile (n x 2, in x-z, convex or
    star-shaped about its mean) extruded over y in +-half_width: vertices
    and faces."""
    p = np.asarray(profile, float)
    n = len(p)
    verts = [(x, -half_width, z) for x, z in p] + [(x, half_width, z) for x, z in p]
    c = p.mean(axis=0)
    verts += [(c[0], -half_width, c[1]), (c[0], half_width, c[1])]
    faces = []
    for k in range(n):
        j = (k + 1) % n
        faces += [(k, j, n + j), (k, n + j, n + k), (2 * n, j, k), (2 * n + 1, n + k, n + j)]
    return verts, faces


def sector(r0, r1, a0, a1, z0, z1, n=24):
    """An annular sector between radii r0 and r1, angles a0 and a1 [rad]
    about z, from z0 to z1: vertices and faces."""
    a = np.linspace(a0, a1, n)
    ring = []
    for z in (z0, z1):
        ring += [(r0 * math.cos(t), r0 * math.sin(t), z) for t in a]
        ring += [(r1 * math.cos(t), r1 * math.sin(t), z) for t in a]
    faces = []

    def idx(level, outer, k):
        return level * 2 * n + outer * n + k

    for k in range(n - 1):
        # bottom and top
        faces += [(idx(0, 0, k), idx(0, 1, k + 1), idx(0, 1, k)),
                  (idx(0, 0, k), idx(0, 0, k + 1), idx(0, 1, k + 1)),
                  (idx(1, 0, k), idx(1, 1, k), idx(1, 1, k + 1)),
                  (idx(1, 0, k), idx(1, 1, k + 1), idx(1, 0, k + 1))]
        # inner and outer walls
        faces += [(idx(0, 0, k), idx(1, 0, k), idx(1, 0, k + 1)),
                  (idx(0, 0, k), idx(1, 0, k + 1), idx(0, 0, k + 1)),
                  (idx(0, 1, k), idx(0, 1, k + 1), idx(1, 1, k + 1)),
                  (idx(0, 1, k), idx(1, 1, k + 1), idx(1, 1, k))]
    for k in (0, n - 1):
        faces += [(idx(0, 0, k), idx(0, 1, k), idx(1, 1, k)),
                  (idx(0, 0, k), idx(1, 1, k), idx(1, 0, k))]
    return ring, faces


def track_loop(half, radius):
    """The track's centre line through its shoes in the undercarriage's x-z
    plane: the straight runs between x = -half and +half, at z = 0 and
    2 radius, joined by half circles round the idler and the sprocket;
    returns a function of the arc length s, its pose (x, z, heading)."""
    straight, arc = 2 * half, math.pi * radius
    length = 2 * straight + 2 * arc

    def at(s):
        s %= length
        if s < straight:                                # bottom run, forwards
            return -half + s, 0.0, 0.0
        s -= straight
        if s < arc:                                     # round the idler
            a = s / radius
            return half + radius * math.sin(a), radius - radius * math.cos(a), a
        s -= arc
        if s < straight:                                # top run, backwards
            return half - s, 2 * radius, math.pi
        s -= straight
        a = s / radius                                  # round the sprocket
        return -half - radius * math.sin(a), radius + radius * math.cos(a), math.pi + a

    return at, length


class Shot:
    """The two runs, drawn (render.shoot)."""

    def __init__(self, rec, captions=True):
        self.r = rec
        self.captions = captions
        self.frames = self._frames()
        self._views = None
        self._ids = None

    # --- the frames: (run, index into its record) ----------------------------
    def _frames(self):
        out = []
        stands, tips = self.r["runs"][0], self.r["runs"][1]
        for k, t in enumerate(stands["t"]):
            if t <= END_STANDS + 1e-9:
                out.append((0, k, False))
        last = None
        for k, q in enumerate(tips["qpos"]):
            out.append((1, k, False))
            if lean(q) >= STOP_LEAN:
                last = k
                break
        if last is None:
            raise SystemExit("the tipping run does not reach the drawing's last lean")
        out += [(1, last, True)] * int(round(HOLD * FPS))
        return out

    def run_of(self, i):
        j, k, held = self.frames[i]
        return self.r["runs"][j], k, j, held

    # --- the visual model --------------------------------------------------
    def build(self):
        from excavator import capacity, front, mass, model
        from excavator import published as pub
        from render import shapes, stage, textures as tx
        V = mujoco.mjtGeom
        spec = model.build(load=0.0, pose=self.r["pose"], slew=0.0)
        stage.dress(spec, extent=10.0, shadowclip=2.0, sun=(-0.30, -0.62, -0.72),
                    zenith=(0.27, 0.47, 0.77), horizon=(0.84, 0.88, 0.92), haze=0.5)
        for g in spec.geoms:
            g.group = 3                                  # the model's own shapes, not drawn
        gravel = tx.add(spec, "gravel", tx.gravel(1024, seed=51), size=2.5, specular=0.05,
                        shininess=0.05)
        soil = tx.add(spec, "soil", tx.soil(1024, seed=52), size=2.0, specular=0.04)
        floor = tx.add(spec, "pit_floor", tx.soil(512, seed=53, tone=(96, 76, 58)), size=1.5,
                       specular=0.04)
        concrete = tx.add(spec, "concrete", tx.concrete(1024, seed=54), size=3.0, specular=0.05)
        yellow = tx.plain(spec, "machine_yellow", (0.94, 0.68, 0.07, 1.0), specular=0.45,
                          shininess=0.55)
        dark = tx.plain(spec, "dark_steel", (0.17, 0.18, 0.19, 1.0), specular=0.45, shininess=0.5)
        steel = tx.plain(spec, "steel", (0.50, 0.51, 0.52, 1.0), specular=0.6, shininess=0.7)
        chrome = tx.plain(spec, "chrome", (0.80, 0.81, 0.82, 1.0), specular=0.9, shininess=0.9)
        shoe = tx.plain(spec, "track_shoe", (0.11, 0.11, 0.11, 1.0), specular=0.25, shininess=0.3)
        glass = tx.plain(spec, "glass", (0.20, 0.27, 0.32, 0.6), specular=0.9, shininess=0.95)
        black = tx.plain(spec, "rubber", (0.06, 0.06, 0.06, 1.0), specular=0.08)
        bag = tx.plain(spec, "bag", (0.95, 0.42, 0.06, 1.0), specular=0.35, shininess=0.45)
        strap = tx.plain(spec, "strap", (0.92, 0.80, 0.12, 1.0), specular=0.2)
        shoring = tx.add(spec, "shoring", tx.paint(256, seed=55, tone=(196, 64, 32), wear=0.4),
                         size=1.0, specular=0.2)
        barrier = tx.add(spec, "barrier", tx.chevrons(512, stripes=6, a=(214, 40, 32),
                                                      b=(236, 236, 232)), size=1.0)
        cladding = tx.add(spec, "cladding", tx.corrugated(512, seed=56, tone=(150, 154, 158),
                                                          ribs=16), size=2.0, specular=0.25)
        cont = [tx.add(spec, f"container{k}", tx.corrugated(512, seed=57 + k, tone=c, ribs=10,
                                                           rust=0.3), size=2.6, specular=0.3)
                for k, c in enumerate(((36, 82, 140), (150, 40, 32), (40, 110, 70)))]
        wood = tx.add(spec, "wood", tx.wood(512, seed=60), size=1.2, specular=0.05)
        orange = tx.plain(spec, "cone_orange", (0.93, 0.36, 0.05, 1.0), specular=0.25)
        white = tx.plain(spec, "cone_band", (0.93, 0.93, 0.91, 1.0), specular=0.4)
        roof = tx.plain(spec, "roof", (0.32, 0.34, 0.36, 1.0), specular=0.2)
        world = spec.worldbody

        # --- the yard and the pit (the model's ground plane is the yard's
        # surface outside the pit; the pit is drawn only) ---------------------
        x0, x1 = PIT_X
        y0, y1 = PIT_Y
        h = PIT_DEPTH / 2
        top = y1 + PIT_DEPTH / math.tan(WALL_SLOPE)       # the battered wall's top edge
        far = 150.0
        for name, (cx, cy, hx, hy) in (
                ("yard_back", ((x0 - far) / 2, 0.0, (far + x0) / 2, far)),
                ("yard_front", ((x1 + far) / 2, 0.0, (far - x1) / 2, far)),
                ("yard_left", ((x0 + x1) / 2, (top + far) / 2, (x1 - x0) / 2, (far - top) / 2)),
                ("yard_right", ((x0 + x1) / 2, (y0 - far) / 2, (x1 - x0) / 2, (far + y0) / 2))):
            shapes.visual(world, name, V.mjGEOM_BOX, (hx, hy, h), pos=(cx, cy, -h), material=gravel)
        shapes.visual(world, "pit_floor", V.mjGEOM_BOX, ((x1 - x0) / 2, (top - y0) / 2, 0.05),
                      pos=((x0 + x1) / 2, (top + y0) / 2, -PIT_DEPTH - 0.05), material=floor)
        # the walls' faces of soil, and the battered wall on the left (+y)
        for name, (cx, cy, hx, hy) in (("wall_back", (x0 + 0.004, (y0 + top) / 2, 0.004, (top - y0) / 2)),
                                       ("wall_front", (x1 - 0.004, (y0 + top) / 2, 0.004, (top - y0) / 2)),
                                       ("wall_right", ((x0 + x1) / 2, y0 + 0.004, (x1 - x0) / 2, 0.004))):
            shapes.visual(world, name, V.mjGEOM_BOX, (hx, hy, h), pos=(cx, cy, -h), material=soil)
        wedge = mesh(spec, "mesh_wall_left", *prism([(y1, -PIT_DEPTH), (top, -PIT_DEPTH),
                                                     (top, 0.0)], (x1 - x0) / 2))
        g = shapes.visual(world, "wall_left", V.mjGEOM_MESH, (1, 1, 1),
                          pos=((x0 + x1) / 2, 0.0, 0.0),
                          quat=[math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)],
                          material=soil)
        g.meshname = wedge
        # the shoring: steel panels on the three vertical walls, struts across
        for name, (cx, cy, hx, hy) in (("shore_back", (x0 + 0.05, 0.0, 0.04, (y1 - y0) / 2 - 0.05)),
                                       ("shore_front", (x1 - 0.05, 0.0, 0.04, (y1 - y0) / 2 - 0.05)),
                                       ("shore_right", ((x0 + x1) / 2, y0 + 0.05, (x1 - x0) / 2 - 0.12,
                                                        0.04))):
            shapes.visual(world, name, V.mjGEOM_BOX, (hx, hy, h - 0.3), pos=(cx, cy, -h - 0.25),
                          material=shoring)
        for k, z in enumerate((-1.2, -3.2)):
            for sy in (-1.2, 1.2):
                capsule_between(world, f"strut{k}{sy:+.0f}", (x0 + 0.09, sy, z), (x1 - 0.09, sy, z),
                                0.06, steel)
        # barriers along the pit's far edges (the camera's side left open)
        for k, (a, b) in enumerate((((x0 - 0.6, y0 - 0.8), (x1 + 0.6, y0 - 0.8)),
                                    ((x1 + 0.6, y0 - 0.8), (x1 + 0.6, top + 0.8)))):
            a, b = np.array(a), np.array(b)
            n = max(1, int(round(np.linalg.norm(b - a) / 1.6)))
            for j in range(n):
                p = a + (b - a) * (j + 0.5) / n
                yaw = math.atan2(*(b - a)[::-1])
                body = world.add_body()
                body.name = f"barrier{k}{j}"
                body.pos = [p[0], p[1], 0.0]
                body.quat = shapes.quat_z(yaw).tolist()
                shapes.visual(body, f"barrier{k}{j}_board", V.mjGEOM_BOX, (0.7, 0.02, 0.12),
                              pos=(0, 0, 0.85), material=barrier)
                for sx in (-0.62, 0.62):
                    shapes.visual(body, f"barrier{k}{j}_leg{sx:+.1f}", V.mjGEOM_BOX,
                                  (0.025, 0.025, 0.45), pos=(sx, 0, 0.45), material=white)
                    shapes.visual(body, f"barrier{k}{j}_foot{sx:+.1f}", V.mjGEOM_BOX,
                                  (0.06, 0.25, 0.03), pos=(sx, 0, 0.03), material=black)
        # the yard round it: a site office, containers, pipes, a spoil heap
        office = world.add_body()
        office.name = "office"
        office.pos = [-15.0, 7.0, 0.0]
        office.quat = shapes.quat_z(0.15).tolist()
        shapes.visual(office, "office_box", V.mjGEOM_BOX, (3.0, 1.22, 1.3), pos=(0, 0, 1.3),
                      material=cladding)
        shapes.visual(office, "office_window", V.mjGEOM_BOX, (0.7, 0.01, 0.45),
                      pos=(0.8, -1.23, 1.5), material=glass)
        shapes.visual(office, "office_roof", V.mjGEOM_BOX, (3.1, 1.3, 0.06), pos=(0, 0, 2.66),
                      material=roof)
        for k, (x, y, yaw) in enumerate(((-14.0, -7.0, 0.1), (-17.0, 15.0, 1.2), (18.0, -10.0, 0.4))):
            shapes.container(spec, f"container{k}", (x, y), yaw, cont[k % 3], dark)
        along_x = [math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]
        for k, (y, z) in enumerate(((-6.0, 0.0), (-4.75, 0.0), (-3.5, 0.0), (-5.375, 1.05),
                                    (-4.125, 1.05))):
            shapes.visual(world, f"pipe{k}", V.mjGEOM_CYLINDER, (0.6, 1.25),
                          pos=(-6.5, y, 0.76 + z), quat=along_x, material=concrete)
        for k, x in enumerate((-7.3, -5.7)):
            shapes.visual(world, f"pipe_bearer{k}", V.mjGEOM_BOX, (0.08, 2.0, 0.08),
                          pos=(x, -4.75, 0.08), material=wood)
        heap_r, heap_h = 3.4, 1.8
        a = np.linspace(0.0, 2 * math.pi, 40, endpoint=False)
        rings = []
        for f, z in ((1.0, 0.0), (0.82, 0.45), (0.58, 0.95), (0.33, 1.45), (0.0, heap_h)):
            rings.append([(f * heap_r * math.cos(t) * (1.0 + 0.08 * math.sin(3 * t)),
                           f * heap_r * math.sin(t) * 0.75, z) for t in a])
        verts = [v for ring in rings[:-1] for v in ring] + [(0.0, 0.0, heap_h)]
        n = len(a)
        faces = []
        for r in range(3):
            for k in range(n):
                j = (k + 1) % n
                faces += [(r * n + k, r * n + j, (r + 1) * n + j), (r * n + k, (r + 1) * n + j, (r + 1) * n + k)]
        apex = len(verts) - 1
        faces += [(3 * n + k, 3 * n + (k + 1) % n, apex) for k in range(n)]
        g = shapes.visual(world, "spoil_heap", V.mjGEOM_MESH, (1, 1, 1), pos=(5.2, -7.0, 0.0),
                          material=soil)
        g.meshname = mesh(spec, "mesh_spoil_heap", verts, faces)
        for k, (x, y) in enumerate(((-4.6, 2.6), (-3.5, -4.2), (8.2, 4.8), (8.2, -4.0))):
            shapes.cone(spec, f"cone{k}", (x, y), orange, white, black)
        for k in range(14):
            x = -16.0 + 3.0 * k
            shapes.visual(world, f"fence_post{k}", V.mjGEOM_CYLINDER, (0.025, 1.0),
                          pos=(x, 17.0, 1.0), material=steel)
            shapes.visual(world, f"fence_panel{k}", V.mjGEOM_BOX, (1.48, 0.004, 0.95),
                          pos=(x + 1.5, 17.0, 1.05), rgba=(0.55, 0.57, 0.58, 0.35))
            shapes.visual(world, f"fence_foot{k}", V.mjGEOM_BOX, (0.3, 0.1, 0.06),
                          pos=(x, 17.0, 0.06), material=concrete)
        door = tx.plain(spec, "door", (0.30, 0.32, 0.34, 1.0), specular=0.3)
        for k, (x, y, hx, hy, hz) in enumerate(((-62.0, 12.0, 18.0, 9.0, 4.5), (-25.0, -62.0, 22.0, 10.0, 5.0),
                                               (-70.0, -30.0, 10.0, 18.0, 4.0), (62.0, -48.0, 10.0, 22.0, 5.5))):
            shapes.visual(world, f"hall{k}_walls", V.mjGEOM_BOX, (hx, hy, hz), pos=(x, y, hz),
                          material=cladding)
            shapes.visual(world, f"hall{k}_roof", V.mjGEOM_BOX, (hx + 0.4, hy + 0.4, 0.15),
                          pos=(x, y, 2 * hz + 0.15), material=roof)
            for sx, sy, ax, ay in ((1, 0, 0.02, hy - 1.0), (-1, 0, 0.02, hy - 1.0),
                                   (0, 1, hx - 1.0, 0.02), (0, -1, hx - 1.0, 0.02)):
                shapes.visual(world, f"hall{k}_windows{sx}{sy}", V.mjGEOM_BOX, (ax, ay, 0.35),
                              pos=(x + sx * (hx + 0.01), y + sy * (hy + 0.01), 2 * hz - 0.9),
                              material=glass)
            # two roller doors on the face towards the yard
            if abs(x) > abs(y):
                fx, fy, along = -math.copysign(1.0, x), 0.0, (0.0, 1.0)
            else:
                fx, fy, along = 0.0, -math.copysign(1.0, y), (1.0, 0.0)
            for j, off in enumerate((-0.3, 0.3)):
                half = (hy if along[1] else hx) * off
                px = x + fx * (hx + 0.03) + along[0] * half
                py = y + fy * (hy + 0.03) + along[1] * half
                size = (0.03, 2.0, 0.7 * hz) if along[1] else (2.0, 0.03, 0.7 * hz)
                shapes.visual(world, f"hall{k}_door{j}", V.mjGEOM_BOX, size, pos=(px, py, 0.7 * hz),
                              material=door)

        # --- the machine, on the model's bodies ------------------------------
        lower = next(b for b in spec.bodies if b.name == "lower")
        upper = next(b for b in spec.bodies if b.name == "upper")
        boom = next(b for b in spec.bodies if b.name == "boom")
        arm = next(b for b in spec.bodies if b.name == "arm")
        bucket = next(b for b in spec.bodies if b.name == "bucket")
        z_ground = -model.SPHERE                          # the ground in the undercarriage's frame
        half, radius = capacity.FRONT_LINE, 0.36          # the idler's axis over the front line
        at, length = track_loop(half, radius)
        pitch = length / 46
        for sy in (-1, 1):
            yc = sy * pub.TRACK_GAUGE / 2
            side = "l" if sy > 0 else "r"
            for k in range(46):
                x, z, hd = at(k * pitch)
                q = quat_y(-hd)
                shapes.visual(lower, f"shoe_{side}{k}", V.mjGEOM_BOX, (0.074, pub.SHOE_WIDTH / 2, 0.017),
                              pos=(x, yc, z_ground + 0.035 + z), quat=q, material=shoe)
                nx, nz = math.sin(hd), -math.cos(hd)            # outward, away from the loop
                shapes.visual(lower, f"grouser_{side}{k}", V.mjGEOM_BOX,
                              (0.012, pub.SHOE_WIDTH / 2, 0.012),
                              pos=(x + 0.022 * nx, yc, z_ground + 0.035 + z + 0.022 * nz), quat=q,
                              material=shoe)
            wz = z_ground + 0.035 + radius
            rot = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]
            for name, x, r in (("idler", half, radius - 0.05), ("sprocket", -half, radius - 0.03)):
                shapes.visual(lower, f"{name}_{side}", V.mjGEOM_CYLINDER, (r, 0.11), pos=(x, yc, wz),
                              quat=rot, material=dark)
                shapes.visual(lower, f"{name}_hub_{side}", V.mjGEOM_CYLINDER, (0.12, 0.16),
                              pos=(x, yc, wz), quat=rot, material=steel)
            for k, x in enumerate(np.linspace(-0.95, 0.95, 4)):
                shapes.visual(lower, f"roller_{side}{k}", V.mjGEOM_CYLINDER, (0.085, 0.13),
                              pos=(x, yc, z_ground + 0.035 + 0.12), quat=rot, material=dark)
            shapes.visual(lower, f"frame_{side}", V.mjGEOM_BOX, (half - 0.1, 0.13, 0.17),
                          pos=(0.0, yc, wz), material=yellow)
            shapes.visual(lower, f"carrier_{side}", V.mjGEOM_CYLINDER, (0.07, 0.1),
                          pos=(0.0, yc, wz + radius - 0.08), quat=rot, material=dark)
        shapes.visual(lower, "car_body", V.mjGEOM_BOX, (0.75, pub.TRACK_GAUGE / 2 - 0.1, 0.18),
                      pos=(0.0, 0.0, z_ground + 0.62), material=yellow)
        shapes.visual(lower, "swing_circle", V.mjGEOM_CYLINDER, (0.62, 0.07),
                      pos=(0.0, 0.0, mass.SWING_HEIGHT - model.SPHERE - 0.09), material=dark)
        # the upper structure: frame, counterweight, hood, cab
        shapes.visual(upper, "deck", V.mjGEOM_BOX, (1.0, 1.15, 0.09), pos=(0.02, 0.0, 0.09),
                      material=dark)
        cw = mesh(spec, "mesh_counterweight",
                  *sector(0.95, pub.TAIL_SWING_RADIUS - 0.02, math.radians(118), math.radians(242),
                          pub.COUNTERWEIGHT_CLEARANCE - mass.SWING_HEIGHT,
                          pub.COUNTERWEIGHT_CLEARANCE - mass.SWING_HEIGHT + 0.78))
        g = shapes.visual(upper, "counterweight_shell", V.mjGEOM_MESH, (1, 1, 1), material=yellow)
        g.meshname = cw
        shapes.visual(upper, "hood", V.mjGEOM_BOX, (0.85, 0.44, 0.42), pos=(-0.18, -0.72, 0.60),
                      material=yellow)
        shapes.visual(upper, "hood_grille", V.mjGEOM_BOX, (0.30, 0.005, 0.16), pos=(-0.30, -1.165, 0.62),
                      material=dark)
        shapes.visual(upper, "exhaust", V.mjGEOM_CYLINDER, (0.05, 0.18), pos=(-0.40, -0.55, 1.18),
                      material=dark)
        cab_top = pub.CAB_HEIGHT - mass.SWING_HEIGHT
        cx, cy, chx, chy = 0.18, 0.72, 0.74, 0.46
        shapes.visual(upper, "cab_base", V.mjGEOM_BOX, (chx, chy, 0.22), pos=(cx, cy, 0.40),
                      material=yellow)
        zc0, zc1 = 0.62, cab_top - 0.04
        for sx in (-1, 1):
            for sy in (-1, 1):
                shapes.visual(upper, f"cab_post{sx}{sy}", V.mjGEOM_BOX, (0.035, 0.035, (zc1 - zc0) / 2),
                              pos=(cx + sx * (chx - 0.035), cy + sy * (chy - 0.035), (zc0 + zc1) / 2),
                              material=dark)
        shapes.visual(upper, "cab_roof", V.mjGEOM_BOX, (chx, chy, 0.04), pos=(cx, cy, cab_top - 0.04),
                      material=dark)
        for name, pos, size in (("cab_glass_front", (cx + chx - 0.01, cy, (zc0 + zc1) / 2),
                                 (0.006, chy - 0.04, (zc1 - zc0) / 2 - 0.02)),
                                ("cab_glass_left", (cx, cy + chy - 0.01, (zc0 + zc1) / 2),
                                 (chx - 0.04, 0.006, (zc1 - zc0) / 2 - 0.02)),
                                ("cab_glass_right", (cx, cy - chy + 0.01, (zc0 + zc1) / 2),
                                 (chx - 0.04, 0.006, (zc1 - zc0) / 2 - 0.02)),
                                ("cab_glass_rear", (cx - chx + 0.01, cy, (zc0 + zc1) / 2),
                                 (0.006, chy - 0.04, (zc1 - zc0) / 2 - 0.02))):
            shapes.visual(upper, name, V.mjGEOM_BOX, size, pos=pos, material=glass)
        shapes.visual(upper, "seat", V.mjGEOM_BOX, (0.25, 0.25, 0.08), pos=(cx - 0.15, cy, 0.78),
                      material=black)
        # the boom's foot, and the boom, the arm and the bucket, each in its frame
        b0 = np.array([front.X0, 0.0, front.Z0 - mass.SWING_HEIGHT])      # in the upper's frame
        t1, t2, t3 = self.r["pose"]
        rot = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]
        for sy in (-1, 1):
            shapes.visual(upper, f"foot_plate{sy}", V.mjGEOM_BOX, (0.22, 0.03, 0.26),
                          pos=(b0[0] + 0.05, sy * 0.24, 0.30), material=yellow)
        knee = np.array([2.55, 0.0, 0.40])
        tip = np.array([pub.BOOM_LENGTH, 0.0, 0.0])
        box_between(boom, "boom_lower", (-0.15, 0, 0), knee + (0.12, 0, 0), 0.19, 0.27, yellow)
        box_between(boom, "boom_upper", knee - (0.12, 0, 0), tip + (0.1, 0, 0), 0.18, 0.22, yellow)
        for name, p, r in (("boom_foot_pin", (0, 0, 0), 0.12), ("boom_tip_pin", tip, 0.11)):
            shapes.visual(boom, name, V.mjGEOM_CYLINDER, (r, 0.24), pos=p, quat=rot, material=steel)
        # the arm: its body from the boom's pin to the bucket's, the tail above
        shapes.visual(arm, "arm_body", V.mjGEOM_BOX, (1.30, 0.15, 0.19), pos=(1.15, 0.0, 0.0),
                      material=yellow)
        shapes.visual(arm, "arm_tail", V.mjGEOM_BOX, (0.40, 0.15, 0.33), pos=(-0.32, 0.0, 0.12),
                      material=yellow)
        shapes.visual(arm, "bucket_pin", V.mjGEOM_CYLINDER, (0.08, 0.2), pos=(pub.ARM_LENGTH, 0, 0),
                      quat=rot, material=steel)
        # the bucket, curled: two side plates and the shell between them
        prof = [(0.0, 0.10), (0.18, 0.28), (0.55, 0.34), (0.90, 0.22), (front.R_BUCKET, -0.02),
                (0.92, -0.06), (0.62, -0.02), (0.30, -0.04), (0.05, -0.08)]
        verts, faces = prism(prof, pub.BUCKET_WIDTH / 2)
        g = shapes.visual(bucket, "bucket_shell", V.mjGEOM_MESH, (1, 1, 1), material=yellow)
        g.meshname = mesh(spec, "mesh_bucket", verts, faces)
        for k in range(5):
            y = (k - 2) * pub.BUCKET_WIDTH / 5.4
            shapes.visual(bucket, f"tooth{k}", V.mjGEOM_BOX, (0.07, 0.04, 0.02),
                          pos=(front.R_BUCKET + 0.05, y, -0.02), quat=quat_y(0.3), material=dark)
        # the hook below the bucket's pin
        shapes.visual(arm, "hook_eye", V.mjGEOM_BOX, (0.06, 0.05, 0.03),
                      pos=(pub.ARM_LENGTH + 0.12, 0.0, 0.0), material=dark)
        # the cylinders at the model's lengths, between pins placed by us
        a_b, c_b, beta = front.boom_cylinder()
        boom_len = front.boom_length(t1)
        base_b = b0 + np.array([0.40, 0.0, -0.32])        # on the upper structure (ours)
        d1 = np.array([math.cos(t1), 0.0, math.sin(t1)])    # along the boom, in the upper's frame
        n1 = np.array([-math.sin(t1), 0.0, math.cos(t1)])
        # the rod pin on the boom's underside, at the distance that gives the model's length
        from scipy.optimize import brentq
        under = 0.0                                       # on the boom's axis, below its plates

        def gap(s):
            return np.linalg.norm(b0 + s * d1 + under * n1 - base_b) - boom_len
        s_rod = brentq(gap, 0.8, 4.0)
        rod_b = b0 + s_rod * d1 + under * n1
        for sy in (-1, 1):
            shapes.visual(boom, f"boom_cyl_lug{sy}", V.mjGEOM_BOX, (0.10, 0.025, 0.16),
                          pos=(s_rod, sy * 0.2, under + 0.10), material=yellow)
        for sy in (-1, 1):
            a = base_b + (0, sy * 0.27, 0)
            b = rod_b + (0, sy * 0.27, 0)
            mid = a + (b - a) * 0.56
            capsule_between(upper, f"boom_cyl_barrel{sy}", a, mid, 0.085, dark)
            capsule_between(upper, f"boom_cyl_rod{sy}", mid - (b - a) * 0.05, b, 0.045, chrome)
        a_a, c_a, off = front.ARM_CYLINDER
        arm_len = front.arm_length(t2)
        b1 = b0 + pub.BOOM_LENGTH * d1                    # the arm's pin, in the upper's frame
        back = rot_y(math.radians(20.0)) @ (-d1)          # back along the boom, above its top
        base_a = b1 + a_a * back
        gamma = off - t2                                  # the angle between the arms a and c
        cands = [b1 + c_a * (rot_y(sgn * gamma) @ back) for sgn in (-1, 1)]
        rod_a = max(cands, key=lambda p: p[2])            # the arm's tail, above its pin
        assert abs(np.linalg.norm(rod_a - base_a) - arm_len) < 1e-6
        mid = base_a + (rod_a - base_a) * 0.55
        capsule_between(upper, "arm_cyl_barrel", base_a, mid, 0.09, dark)
        capsule_between(upper, "arm_cyl_rod", mid - (rod_a - base_a) * 0.05, rod_a, 0.05, chrome)
        capsule_between(upper, "arm_tail_link", b1, rod_a, 0.07, yellow)
        # the bucket's cylinder along the arm, to the linkage at the bucket
        d2 = np.array([math.cos(t1 + t2), 0.0, math.sin(t1 + t2)])
        n2 = np.array([-math.sin(t1 + t2), 0.0, math.cos(t1 + t2)])
        p_base = b1 + 0.35 * d2 - 0.28 * n2
        p_link = b1 + (pub.ARM_LENGTH - 0.30) * d2 - 0.30 * n2
        mid = p_base + (p_link - p_base) * 0.55
        capsule_between(upper, "bucket_cyl_barrel", p_base, mid, 0.075, dark)
        capsule_between(upper, "bucket_cyl_rod", mid - (p_link - p_base) * 0.05, p_link, 0.04, chrome)
        pin = b1 + pub.ARM_LENGTH * d2
        d3 = np.array([math.cos(t1 + t2 + t3), 0.0, math.sin(t1 + t2 + t3)])
        p_bucket = pin + 0.32 * d3
        for sy in (-1, 1):
            capsule_between(upper, f"link_arm{sy}", p_link + (0, sy * 0.12, 0),
                            b1 + (pub.ARM_LENGTH - 0.22) * d2 + (0, sy * 0.12, 0), 0.035, dark)
            capsule_between(upper, f"link_bucket{sy}", p_link + (0, sy * 0.12, 0),
                            p_bucket + (0, sy * 0.12, 0), 0.035, dark)
        self.lengths = (boom_len, arm_len)

        # --- the bag: a mocap body that the drawing places below the hook ----
        bag_body = world.add_body()
        bag_body.name = "bag"
        bag_body.mocap = True
        shapes.visual(bag_body, "bag_shell", V.mjGEOM_ELLIPSOID, (0.5, 0.5, 0.3), material=bag)
        shapes.visual(bag_body, "bag_ring", V.mjGEOM_CYLINDER, (0.32, 0.03), pos=(0.0, 0.0, 0.0),
                      material=strap)
        m = spec.compile()
        self._ids = dict(bag=m.body("bag").id, shell=m.geom("bag_shell").id,
                         ring=m.geom("bag_ring").id, lift=m.site("lift_point").id,
                         supports=[m.geom(n).id for n in SUPPORTS], lower=m.body("lower").id)
        return m

    # --- the bag ---------------------------------------------------------------
    @staticmethod
    def bag_size(force):
        """The drawn bag's half sizes (radius, half height) [m] for the water
        that weighs force [N]: its radius grows to 1.2 m with the cube root of
        the volume, its height holds the volume (an ellipsoid)."""
        vol = max(force, 0.0) / (WATER * G)
        full = 1.02 * 5790.1 / WATER                     # the larger run's volume, about 5.9 m^3
        r = 0.45 + 0.75 * (min(vol / full, 1.0)) ** (1.0 / 3.0)
        c = max(0.25, 3.0 * vol / (4.0 * math.pi * r * r))
        return r, c

    def pose(self, m, d, i):
        run, k, _, _ = self.run_of(i)
        d.qpos[:] = run["qpos"][k]
        mujoco.mj_forward(m, d)
        ids = self._ids
        r, c = self.bag_size(run["hook"][k])
        hook = d.site_xpos[ids["lift"]] - np.array([0.0, 0.0, 0.18])
        ring = hook - np.array([0.0, 0.0, 0.75])
        m.geom_size[ids["shell"]] = (r, r, c)
        m.geom_pos[ids["shell"]] = (0.0, 0.0, -c + 0.05)
        m.geom_size[ids["ring"]][0] = min(0.32, 0.6 * r)
        d.mocap_pos[m.body_mocapid[ids["bag"]]] = ring
        mujoco.mj_forward(m, d)
        self._hook = hook

    def focus(self, m, lookat):
        from render import stage
        stage.follow(m, lookat, distance=40.0)

    def views(self):
        if self._views is not None:
            return self._views
        from render import film
        # (time in the run, lookat x, y, z, distance, azimuth, elevation)
        keys = {0: ((0.0, 2.6, 0.0, 0.0, 17.5, 228.0, -24.0), (6.0, 2.4, 0.0, 0.2, 15.5, 246.0, -20.0),
                    (10.0, 2.2, 0.0, 0.3, 15.0, 252.0, -18.0)),
                1: ((0.0, 2.2, 0.0, 0.3, 15.0, 262.0, -18.0), (3.0, 1.0, 0.4, 0.4, 12.0, 285.0, -12.0),
                    (6.0, -0.6, 0.6, 0.4, 9.5, 304.0, -7.0), (7.4, -0.6, 0.6, 0.4, 9.5, 306.0, -7.0),
                    (8.4, 1.0, 0.3, 0.2, 14.0, 282.0, -11.0), (9.6, 2.0, 0.0, -0.3, 16.5, 272.0, -15.0))}
        rows = []
        for i in range(len(self.frames)):
            run, k, j, held = self.run_of(i)
            t = run["t"][k] + (0.0 if not held else 0.0)
            ks = keys[j]
            kt = [x[0] for x in ks]
            rows.append([np.interp(t, kt, [x[c] for x in ks]) for c in range(1, 7)])
        self._views = film.smooth_path(rows, 1.0 / FPS, tau=(0.5, 0.5, 0.5, 0.9, 0.9, 0.9))
        return self._views

    def view(self, i):
        return self.views()[i]

    def overlay(self, i, m, d):
        """The supports' forces as arrows from the ground beside the tracks
        (green), the front tipping line on the ground (yellow), and the bag's
        four straps from the hook."""
        from render import film
        run, k, _, _ = self.run_of(i)
        forces = run["support"][k]
        ids = self._ids
        lower_R = d.xmat[ids["lower"]].reshape(3, 3)
        pts = [d.geom_xpos[g] for g in ids["supports"]]
        hook = self._hook
        r, c = self.bag_size(run["hook"][k])
        ring = hook - np.array([0.0, 0.0, 0.75])

        def draw(scene):
            for p, f, name in zip(pts, forces, SUPPORTS):
                out = 1.0 if name[-1] == "l" else -1.0
                base = p + lower_R @ np.array([0.0, out * 0.42, 0.0])
                base[2] = max(base[2] - 0.02, 0.0) + 0.005
                if f > 50.0:
                    film.arrow(scene, base, base + np.array([0.0, 0.0, 1.5 * f / 1e5]), 0.07,
                               (0.30, 0.85, 0.50, 0.95))
                film.dot(scene, base, 0.09, (0.30, 0.85, 0.50, 0.9) if f > 50.0
                         else (0.95, 0.25, 0.20, 0.9))
            fl, fr = pts[0], pts[1]
            u = (fl - fr) / np.linalg.norm(fl - fr)
            a, b = fr - 0.6 * u, fl + 0.6 * u
            a[2] = b[2] = 0.006
            film.line(scene, a, b, 0.018, (1.0, 0.85, 0.10, 0.95))
            for ang in (0.25, 1.82, 3.39, 4.96):
                film.line(scene, hook, ring + 0.55 * min(0.32, 0.6 * r) *
                          np.array([math.cos(ang), math.sin(ang), 0.0]) * 1.6, 0.022,
                          (0.92, 0.80, 0.12, 1.0), decor=False)
        return draw

    def caption(self, i):
        if not self.captions:
            return None
        from render import film
        run, k, j, held = self.run_of(i)
        share = RUNS[j][1]
        f = run["support"][k] / 1e3
        load = run["hook"][k] / G
        t = run["t"][k]
        state = ("settling, no load" if t < 2.0 else "the load rising" if t < 7.0 else "the load held")
        lines = [f"{100 * share:.0f} % of the static tipping load, {share * self.r['analytic']:.0f} kg "
                 f"· t = {film.number(t, 2, 5)} s · {state}",
                 f"load {film.number(load, 0, 4)} kg · supports: front {film.number(f[0], 1, 4)} and "
                 f"{film.number(f[1], 1, 4)} kN, rear {film.number(f[2], 1, 4)} and "
                 f"{film.number(f[3], 1, 4)} kN · lean {film.number(lean(run['qpos'][k]), 2, 5)}°"]
        if held:
            lines.append("The drawing stops here: the simulated machine, with contacts only at the "
                         "ends of its tipping lines, keeps turning about the front line")
        elif j == 0 and t > 7.0:
            lines.append("The machine stands; the run holds the load to 31 s")
        return ("PC138US-8 model: the load raised slowly at the lift point", lines)

    def caption2(self, i):
        if not self.captions:
            return None
        return (None, ["Simulated in MuJoCo 3.16 (1 ms steps), Section 14.1's onset runs",
                       "Drawn from the record: the machine on the model's bodies,",
                       "the load as a bag that fills; green: the supports' forces"])


def sight(shot, clear=1.5):
    """The frames whose camera comes within clear of something drawn that
    is not the machine, the ground or the pit, or whose line of sight to
    the undercarriage, the cab or the boom's tip meets something else
    drawn: a list of (frame, reason), empty if the camera sees the machine
    throughout."""
    from render import film
    m = shot.build()
    d = mujoco.MjData(m)
    lower = m.body("lower").id
    machine = {b for b in range(m.nbody) if m.body_rootid[b] == lower}
    ground = ("yard_", "pit_floor", "wall_", "shore_", "strut")
    scenery = [g for g in range(m.ngeom) if m.geom_group[g] <= 2
               and m.geom_bodyid[g] not in machine
               and not any(m.geom(g).name.startswith(p) for p in ground)
               and m.geom(g).name not in ("bag_shell", "bag_ring")]
    groups = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)
    hit = np.zeros(1, dtype=np.int32)
    out = []
    for i, view in enumerate(shot.views()):
        shot.pose(m, d, i)
        cam = film.camera_position(*view)
        if cam[2] < 0.5:
            out.append((i, f"camera {cam[2]:.2f} m above the yard"))
        near = film.clearances(m, d, cam, scenery)
        k = int(np.argmin(near))
        if near[k] < clear:
            out.append((i, f"camera {near[k]:.2f} m from {m.geom(scenery[k]).name}"))
        targets = {"undercarriage": d.xpos[lower] + np.array([0.0, 0.0, 0.5]),
                   "cab": d.geom_xpos[m.geom("cab_roof").id],
                   "boom's tip": d.geom_xpos[m.geom("boom_tip_pin").id]}
        for name, target in targets.items():
            ray = target - cam
            length = float(np.linalg.norm(ray))
            dist = mujoco.mj_ray(m, d, cam, ray / length, groups, 1, -1, hit)
            if 0 <= dist < length - 0.5 and m.geom_bodyid[hit[0]] not in machine:
                out.append((i, f"sight to the {name} blocked by {m.geom(hit[0]).name}"))
    return out


def check(rec):
    """onset()'s result against the summary that excavator_mujoco.py wrote,
    and the two drawn runs' loads against their shares."""
    path = os.path.join(HERE, "results", "excavator-summary.txt")
    want = None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if "hook 0 m, reach 4.6 m, over the front: the settled" in line:
                want = line.split("=", 1)[1].split(";")[1].strip()
    got = f"{rec['fraction'] * rec['analytic']:.1f}"
    if want != got:
        raise SystemExit(f"onset() gives {got} kg, the summary {want} kg")
    for (name, share), run in zip(RUNS, rec["runs"]):
        held = run["hook"][run["t"] > 7.5].mean() / G
        if abs(held / (share * rec["analytic"]) - 1.0) > 1e-9:
            raise SystemExit(f"the run '{name}' holds {held:.3f} kg, not {share:g} of the "
                             "static tipping load")
    return got


def main():
    ap = argparse.ArgumentParser(description="Section 14.1's onset runs as a video")
    ap.add_argument("--quick", action="store_true", help="640 x 360, no stills")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--export", metavar="DIR", help="write the scene and frames for render/cycles "
                    "instead of drawing them")
    args = ap.parse_args()
    from render import shoot
    out_dir = os.path.join(HERE, "results", "video")
    os.makedirs(out_dir, exist_ok=True)
    rec = record()
    tips_at = check(rec)
    shot = Shot(rec)
    blocked = sight(shot)
    if blocked:
        raise SystemExit(f"the camera loses the machine in {len(blocked)} frames, first at "
                         f"frame {blocked[0][0]}: {blocked[0][1]}")
    n = len(shot.frames)
    stills = {}
    for name, when in STILLS:
        j = [r[0] for r in RUNS].index(name)
        if when == "end":
            i = max(k for k, f in enumerate(shot.frames) if f[0] == j and not f[2])
        else:
            i = next(k for k, f in enumerate(shot.frames)
                     if f[0] == j and abs(rec["runs"][j]["t"][f[1]] - when) < 0.5 / FPS)
        t = rec["runs"][j]["t"][shot.frames[i][1]]
        stills[i] = f"excavator-onset-{name}-{t:05.2f}s"
    if args.export:
        from render import export
        export.bundle(shot, n, args.export, fps=FPS, stills=stills, name="excavator-onset")
        return
    size = (640, 360) if args.quick else (1920, 1080)
    path = os.path.join(out_dir, "excavator-onset.mp4" if not args.quick else "excavator-quick.mp4")
    shoot.render(shot, n, path, size=size, supersample=1 if args.quick else 2,
                 workers=args.workers)
    frames, rate, w, h = shoot.probe(path)
    last = shot.frames[-1]
    tips = rec["runs"][1]
    print(f"{path}: {frames} frames at {rate:g} per second, {w} x {h}; onset() simulated "
          f"{len(rec['runs'])} runs and tips over from {tips_at} kg (static "
          f"{rec['analytic']:.1f} kg); the tipping run drawn to {tips['t'][last[1]]:.3f} s, lean "
          f"{lean(tips['qpos'][last[1]]):.2f} deg; cylinders drawn at {shot.lengths[0]:.4f} and "
          f"{shot.lengths[1]:.4f} m")
    with open(os.path.join(out_dir, "excavator-onset-frames.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(("frame", "run", "t", "load_kg", "front_l_N", "front_r_N", "rear_l_N",
                     "rear_r_N", "lean_deg"))
        for i in range(n):
            run, k, j, _ = shot.run_of(i)
            f = run["support"][k]
            wr.writerow((i, RUNS[j][0], shoot.fixed(run["t"][k], 4),
                         shoot.fixed(run["hook"][k] / G, 2), *(shoot.fixed(x, 1) for x in f),
                         shoot.fixed(lean(run["qpos"][k]), 4)))
    if not args.quick:
        clean = Shot(rec, captions=False)
        for i, name in stills.items():
            shoot.still(clean, i, os.path.join(out_dir, f"{name}.jpg"), size=(1600, 900))


if __name__ == "__main__":
    main()
