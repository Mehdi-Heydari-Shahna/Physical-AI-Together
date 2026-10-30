"""
Section 13.2's governed crossing of the yard by the Husky A300 as a video,
and its stills.

Physical AI, Together, Chapter 13 (Section 13.2; Figure 13.7).

The script runs the crossing "map" of husky_route.py (the map's labels,
the proposal husky.cruise, the estimator with sideslip) again with the
book's modules unchanged and records the simulated robot at 30 frames per
second of simulated time (render.shapes.Recorder, which only copies the
state after each of MuJoCo's steps), from the end of the estimators'
calibration at rest, where the crossing's clock starts. It then draws the
record, simulating nothing: the robot with the visual meshes of Clearpath's
description of the A300, which the book's model is without (the simulators
discard them), its wheels turning as the simulated wheel joints did, on the
yard of world.py: the asphalt, the gravel and the wet grass where the
simulation has its three grounds, stacks of pallets that fill the boxes of
its obstacles beside the lane, the dock where it has the dock, and the
lane's edges where the task has them. The field, the fence, the building
with its door and canopy, and the cones are scenery. Before it draws, the
script checks that the camera stays inside the fence, away from the
obstacles, and sees the robot in every frame (sight()).

    python husky_video.py --clearpath DIR      # writes results/video/
    python husky_video.py --clearpath DIR --quick
    python husky_video.py --clearpath DIR --export DIR2   # the scene and frames for the
                                                          # GPU renderer (render/cycles)

DIR is a checkout of Clearpath's clearpath_common at the commit of the
record (78e3677), for its clearpath_platform_description/meshes. Requires
MuJoCo 3.16.0, NumPy, SciPy and Pillow, EGL (Mesa's or a GPU's) and
ffmpeg. About forty minutes on two cores at 1920 x 1080.

Writes results/video/husky-map.mp4, its stills, and husky-map-frames.csv:
the frames' times and the robot's position and orientation (a quaternion),
the check that the drawing follows the record.
"""
import argparse
import csv
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "ch10"))
sys.path.insert(0, os.path.join(HERE, ".."))

import mujoco                                                     # noqa: E402
import numpy as np                                                # noqa: E402

FPS = 30
STILLS = (3.0, 11.5, 18.5, 26.0)
KEYS = ((0.0, -35.0, -15.0, 3.6), (3.0, -60.0, -12.0, 3.6), (6.0, -95.0, -11.0, 3.8),
        (9.0, -120.0, -14.0, 4.2), (11.5, -100.0, -24.0, 5.0), (14.0, -80.0, -24.0, 5.0),
        (17.0, -95.0, -14.0, 4.0), (20.0, -80.0, -12.0, 3.8), (23.0, -55.0, -14.0, 4.2),
        (26.0, -45.0, -18.0, 4.6), (30.0, -60.0, -18.0, 4.8), (34.0, -75.0, -20.0, 5.0))
WHEELS = ("front_left", "rear_left", "front_right", "rear_right")


def record():
    """The crossing "map" of husky_route.py, recorded from the crossing's
    start: times [s], qpos, and the yard's trace (time, s, d, speed, ...)."""
    import husky_route as hr
    from render import shapes
    with shapes.Recorder(FPS, select=lambda m, d: m.nq == 11) as rec:
        run, yard, cfg, metrics = hr.crossing("map")
    t, qpos, _, _, _ = rec.longest()
    keep = t >= yard.t0 - 1e-9
    trace = np.array([r[:9] for r in yard.trace])
    return dict(t=t[keep] - yard.t0, qpos=qpos[keep], trace=trace,
                alarms=list(metrics["alarms"]), final_s=metrics["final_s"],
                at_rest=metrics["at_rest_in_zone"])


class Shot:
    """The crossing, drawn (render.shoot)."""

    def __init__(self, rec, clearpath, captions=True):
        self.r, self.clearpath, self.captions = rec, clearpath, captions
        self._views = None

    def build(self):
        from husky import model, world
        from render import collada, shapes, stage, textures as tx, urdf
        spec = model.build()
        for b in spec.bodies:                         # the robot's own shapes, not drawn
            if b.name != "world":
                for g in b.geoms:
                    g.group = 3
        stage.dress(spec, extent=3.0, shadowclip=6.0, sun=(0.55, 0.35, -0.76),
                    zenith=(0.30, 0.50, 0.80), horizon=(0.84, 0.88, 0.92), haze=0.5)
        asphalt = tx.add(spec, "asphalt", tx.asphalt(1024, seed=51), size=3.0, specular=0.08)
        gravel = tx.add(spec, "gravel", tx.gravel(1024, seed=52, stones=12000), size=1.2,
                        specular=0.05)
        grass = tx.add(spec, "grass", tx.grass(1024, seed=53, wet=True), size=2.5,
                       reflectance=0.03, specular=0.25, shininess=0.5)
        field = tx.add(spec, "field", tx.grass(1024, seed=54, wet=False), size=3.0,
                       specular=0.05)
        concrete = tx.add(spec, "concrete", tx.concrete(1024, seed=55), size=2.0, specular=0.05)
        paint = tx.add(spec, "paint", tx.paint(256, seed=56), size=0.5, specular=0.15)
        yellow = tx.add(spec, "yellow_paint", tx.paint(256, seed=57, tone=(226, 180, 30)), size=0.5)
        wood = tx.add(spec, "wood", tx.wood(512, seed=58), size=1.2, specular=0.05)
        carton = tx.plain(spec, "carton", (0.62, 0.48, 0.32, 1.0), specular=0.05)
        wrap = tx.plain(spec, "wrap", (0.80, 0.82, 0.84, 1.0), specular=0.5, shininess=0.6)
        cladding = tx.add(spec, "cladding", tx.corrugated(512, seed=59, tone=(150, 156, 160),
                                                          ribs=16), size=1.5, specular=0.3)
        door = tx.add(spec, "door", tx.corrugated(512, seed=60, tone=(70, 90, 110), ribs=24),
                      size=1.0, specular=0.3)
        dark = tx.plain(spec, "dark_steel", (0.15, 0.16, 0.17, 1.0), specular=0.4)
        fence = tx.plain(spec, "fence", (0.55, 0.57, 0.58, 1.0), specular=0.5)
        orange = tx.plain(spec, "cone_orange", (0.93, 0.36, 0.05, 1.0), specular=0.25)
        white = tx.plain(spec, "cone_band", (0.93, 0.93, 0.91, 1.0), specular=0.4)
        black = tx.plain(spec, "cone_base", (0.10, 0.10, 0.10, 1.0), specular=0.1)
        for g in spec.geoms:
            if g.name == "asphalt (patch)":
                g.material, g.rgba = asphalt, [1, 1, 1, 1]
            elif g.name == "gravel (patch)":
                g.material, g.rgba = gravel, [1, 1, 1, 1]
            elif g.name == "wet grass (patch)":
                g.material, g.rgba = grass, [1, 1, 1, 1]
            elif g.name == "ground":
                g.material, g.rgba = concrete, [1, 1, 1, 1]
            elif g.name in [o.name for o in world.OBSTACLES]:
                g.group = 3                           # drawn below instead
        for light in list(spec.lights):
            if light.name not in ("sun", "skyfill"):
                light.active = False
        stage.ground(spec, field, size=200, z=-0.012)
        # the lane's edges along the route
        ss = np.arange(-1.0, world.DOCK_S + 0.01, 0.05)
        xy = np.array([world.point(s)[:2] for s in ss])
        for side in (-1, 1):
            shapes.ribbon(spec, f"lane{side}", xy, 0.05, 0.0012, paint,
                          offset=side * world.LANE_HALF_WIDTH)
        # the pallets where the simulation has its obstacles: stacks that fill each
        # obstacle's box, its footprint and its height
        for k, o in enumerate(world.OBSTACLES[1:]):
            hx, hy = o.half[0], o.half[1]
            yaw = o.yaw + (0.0 if hx >= hy else 0.5 * math.pi)
            along, across = 2 * max(hx, hy), 2 * min(hx, hy)
            n = max(1, int(round(along / 1.2)))
            for j in range(n):
                off = (j - (n - 1) / 2) * along / n
                cx = o.center[0] + off * math.cos(yaw)
                cy = o.center[1] + off * math.sin(yaw)
                shapes.pallet_stack(spec, f"pallets{k}_{j}", (cx, cy), yaw, wood, carton,
                                    layers=2, wrap=wrap if (j + k) % 2 else None,
                                    size=(along / n, across), height=2 * o.half[2])
        # the dock and the building behind it
        dock = world.OBSTACLES[0]
        b = spec.worldbody.add_body()
        b.name = "dock_body"
        b.pos = [dock.center[0], dock.center[1], 0.0]
        shapes.visual(b, "dock_block", mujoco.mjtGeom.mjGEOM_BOX,
                      (dock.half[0], dock.half[1], dock.half[2]), pos=(0, 0, dock.half[2]),
                      material=concrete)
        shapes.visual(b, "dock_edge", mujoco.mjtGeom.mjGEOM_BOX, (dock.half[0], 0.04, 0.012),
                      pos=(0.0, -dock.half[1] + 0.04, 2 * dock.half[2] + 0.012), material=yellow)
        shapes.visual(b, "building", mujoco.mjtGeom.mjGEOM_BOX, (9.0, 0.3, 3.5),
                      pos=(0.0, dock.half[1] + 0.3, 3.5), material=cladding)
        shapes.visual(b, "door", mujoco.mjtGeom.mjGEOM_BOX, (1.6, 0.02, 1.5),
                      pos=(0.0, dock.half[1] - 0.01, 2 * dock.half[2] + 1.5), material=door)
        shapes.visual(b, "canopy", mujoco.mjtGeom.mjGEOM_BOX, (2.2, 0.6, 0.05),
                      pos=(0.0, dock.half[1] - 0.6, 2 * dock.half[2] + 3.2), material=dark)
        # a fence round the yard
        x0, x1, y0, y1 = world.YARD
        f = spec.worldbody.add_body()
        f.name = "fence"
        for k, (a, c) in enumerate((((x0, y0), (x1, y0)), ((x0, y0), (x0, y1)), ((x1, y0), (x1, y1)))):
            a, c = np.array(a, float), np.array(c, float)
            n = int(np.linalg.norm(c - a) / 2.5)
            for j in range(n + 1):
                p = a + (c - a) * j / n
                shapes.visual(f, f"post{k}_{j}", mujoco.mjtGeom.mjGEOM_CYLINDER, (0.03, 0.9),
                              pos=(p[0], p[1], 0.9), material=fence)
            for z in (0.3, 1.0, 1.7):
                shapes.visual(f, f"rail{k}_{z}", mujoco.mjtGeom.mjGEOM_CAPSULE, (0.012,),
                              fromto=(a[0], a[1], z, c[0], c[1], z), material=fence)
        for k, s0 in enumerate((11.0, 12.4, 13.8)):        # inside the turn, beyond the lane
            x, y, h = world.point(s0)
            shapes.cone(spec, f"cone{k}", (x - 1.1 * math.sin(h), y + 1.1 * math.cos(h)),
                        orange, white, black, scale=0.8)
        # the robot's visual meshes, on the bodies that MuJoCo's import kept
        desc = os.path.join(HERE, "husky", "description", "husky_a300.urdf")
        bodies = ["base_link"] + [f"{w}_wheel_link" for w in WHEELS]
        pkg = {"clearpath_platform_description":
               os.path.join(self.clearpath, "clearpath_platform_description")}
        elements = urdf.visuals(desc, bodies, pkg)
        mats = {}

        def material(rgba, path):
            name = os.path.basename(path)
            key = (tuple(round(c, 3) for c in rgba), "wheel" in path)
            if key not in mats:
                r, g, b_, a = rgba
                bright = max(r, g, b_)
                if "wheel" in path:
                    mats[key] = tx.plain(spec, f"husky_m{len(mats)}", (0.08, 0.08, 0.08, 1.0),
                                         specular=0.15, shininess=0.2)
                elif bright > 0.5:                 # paint, glossy
                    mats[key] = tx.plain(spec, f"husky_m{len(mats)}", (r, g, b_, a),
                                         specular=0.5, shininess=0.6)
                else:                              # dark plastics and metal: lifted a little,
                    k = 1.8                        # since nothing here lights them indirectly
                    mats[key] = tx.plain(spec, f"husky_m{len(mats)}",
                                         (min(1.0, k * r), min(1.0, k * g), min(1.0, k * b_), a),
                                         specular=0.25, shininess=0.35)
            return mats[key]

        urdf.attach(spec, elements, material, collada, name="a300")
        return spec.compile()

    def pose(self, m, d, i):
        d.qpos[:] = self.r["qpos"][i]
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)

    def focus(self, m, lookat):
        from render import stage
        stage.follow(m, lookat, distance=40.0)

    def views(self):
        if self._views is not None:
            return self._views
        from render import film
        # (time, azimuth relative to the heading, elevation, distance) [s, deg, deg, m]:
        # the camera on the robot's left, inside the turn and away from the pallets,
        # which sight() checks
        keys = KEYS
        kt = np.array([k[0] for k in keys])
        rows = []
        for i, t in enumerate(self.r["t"]):
            q = self.r["qpos"][i]
            w, x, y, z = q[3:7]
            heading = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            rows.append([q[0], q[1], q[2] + 0.15, np.interp(t, kt, [k[3] for k in keys]),
                         math.degrees(heading) + np.interp(t, kt, [k[1] for k in keys]),
                         np.interp(t, kt, [k[2] for k in keys])])
        self._views = film.smooth_path(rows, 1.0 / FPS, tau=(0.35, 0.35, 0.35, 1.2, 1.2, 1.2))
        return self._views

    def view(self, i):
        return self.views()[i]

    def overlay(self, i, m, d):
        return None

    def caption(self, i):
        if not self.captions:
            return None
        from husky import world
        from render import film
        t = self.r["t"][i]
        tr = self.r["trace"]
        j = min(int(np.searchsorted(tr[:, 0], t)), len(tr) - 1)
        s, speed = tr[j, 1], tr[j, 3]
        ground = world.ground_at(s).name
        return ("Husky A300 under the governor, crossing the yard to the dock",
                [f"t = {film.number(t, 1, 4)} s · {film.number(speed, 2, 4)} m/s · "
                 f"s = {film.number(s, 2, 5)} m along the route · {ground}"])

    def caption2(self, i):
        if not self.captions:
            return None
        return (None, ["Simulated in MuJoCo 3.16 (0.5 ms steps), Section 13.2, the crossing 'map'",
                       "Drawn from the record with Clearpath's meshes"])


def sight(shot, margin=1.0, clear=1.0):
    """The frames whose camera stands within margin of the yard's fence or
    beyond the dock's face, within clear of an obstacle, or whose line of
    sight to the robot's middle or to a point 0.45 m ahead of it or behind
    it meets something else drawn: a list of (time, reason), empty if the
    camera sees the robot throughout."""
    from husky import world
    from render import film
    m = shot.build()
    d = mujoco.MjData(m)
    robot = {m.body(n).id for n in ["base_link"] + [f"{w}_wheel_link" for w in WHEELS]}
    groups = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)
    x0, x1, y0, y1 = world.YARD
    face = world.OBSTACLES[0].center[1] - world.OBSTACLES[0].half[1]
    hit = np.zeros(1, dtype=np.int32)
    out = []
    for i, view in enumerate(shot.views()):
        shot.pose(m, d, i)
        cam = film.camera_position(*view)
        if not (x0 + margin <= cam[0] <= x1 - margin and y0 + margin <= cam[1] <= face - margin):
            out.append((shot.r["t"][i], f"camera at ({cam[0]:.2f}, {cam[1]:.2f}) m"))
        for o in world.OBSTACLES:
            c, s_ = math.cos(o.yaw), math.sin(o.yaw)
            rel = cam[:2] - np.asarray(o.center[:2])
            local = np.abs([c * rel[0] + s_ * rel[1], -s_ * rel[0] + c * rel[1]]) - o.half[:2]
            if math.hypot(*np.maximum(local, 0.0)) < clear:
                out.append((shot.r["t"][i], f"camera within {clear} m of {o.name}"))
        q = shot.r["qpos"][i]
        w, x, y, z = q[3:7]
        h = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        for off in (-0.45, 0.0, 0.45):
            target = np.array([q[0] + off * math.cos(h), q[1] + off * math.sin(h), q[2] + 0.12])
            ray = target - cam
            length = float(np.linalg.norm(ray))
            dist = mujoco.mj_ray(m, d, cam, ray / length, groups, 1, -1, hit)
            if 0 <= dist < length - 0.05 and m.geom_bodyid[hit[0]] not in robot:
                out.append((shot.r["t"][i], f"sight blocked by {m.geom(hit[0]).name}"))
    return out


def main():
    ap = argparse.ArgumentParser(description="Section 13.2's crossing as a video")
    ap.add_argument("--clearpath", required=True, help="a checkout of clearpath_common")
    ap.add_argument("--quick", action="store_true", help="640 x 360, no stills")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--export", metavar="DIR", help="write the scene and frames for the GPU "
                    "renderer (render/cycles) instead of drawing them")
    args = ap.parse_args()
    from render import shoot
    out_dir = os.path.join(HERE, "results", "video")
    os.makedirs(out_dir, exist_ok=True)
    rec = record()
    shot = Shot(rec, args.clearpath)
    blocked = sight(shot)
    if blocked:
        raise SystemExit(f"the camera loses the robot in {len(blocked)} frames, first at "
                         f"{blocked[0][0]:.2f} s: {blocked[0][1]}")
    n = len(rec["t"])
    if args.export:
        from render import export
        export.bundle(shot, n, args.export, fps=FPS, name="husky-map",
                      stills={int(round(s * FPS)): f"husky-map-{s:04.1f}s" for s in STILLS})
        return
    size = (640, 360) if args.quick else (1920, 1080)
    path = os.path.join(out_dir, "husky-map.mp4" if not args.quick else "husky-quick.mp4")
    shoot.render(shot, n, path, size=size, supersample=1 if args.quick else 2,
                 workers=args.workers)
    frames, rate, w, h = shoot.probe(path)
    print(f"{path}: {frames} frames at {rate:g} per second, {w} x {h}; the record: {n} "
          f"states over {rec['t'][-1]:.2f} s; alarms {rec['alarms'] or 'none'}; at rest in "
          f"the dock zone from {rec['at_rest']:.2f} s")
    with open(os.path.join(out_dir, "husky-map-frames.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(("t", "x", "y", "z", "qw", "qx", "qy", "qz"))
        for k in range(n):
            q = rec["qpos"][k]
            wr.writerow((shoot.fixed(rec['t'][k], 4),) + tuple(shoot.fixed(v, 5) for v in q[:7]))
    if not args.quick:
        clean = Shot(rec, args.clearpath, captions=False)
        for s in STILLS:
            i = int(round(s * FPS))
            shoot.still(clean, i, os.path.join(out_dir, f"husky-map-{s:04.1f}s.jpg"),
                        size=(1600, 900))


if __name__ == "__main__":
    main()
