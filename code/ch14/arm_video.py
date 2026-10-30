"""
Section 14.2's two stops of the loaded UR5e near the post as a video, and
its stills.

Physical AI, Together, Chapter 14 (Section 14.2; Figure 14.6).

The script runs again, with the book's modules unchanged, the two stops of
Table S14.2 from the latest angles, one of which ends 0.0100 m short of the
post and the other in it: the heaviest corner of the declared interval,
4.95 kg with its centre of mass 0.075 m below the flange, swinging at full
speed (180 deg/s) and stopped

  * under the declared interval's label, 56.76 rad/s^2, from the latest
    angle that this label allows, -61.072 deg: it comes to rest 0.0100 m
    from the post, on the floor; and
  * under the empty arm's label, 128.69 rad/s^2, from the latest angle that
    this label allows with the box in place, -63.857 deg: it overruns its
    commanded rest and strikes the post.

It records the simulated state after every step of 0.5 ms
(render.shapes.Recorder, which only copies the state after each of
MuJoCo's steps), with the force of the post's contacts, and draws every
second state: one frame for each millisecond of simulated time, so that
the video, at 30 frames per second, runs at 1/33 of real time; each stop's
first state is held while the camera moves in, and each run's last state
at its end. Each stop begins, as in the book, with the arm already at full
speed at its starting angle; nothing before it is simulated. The arm is drawn with the visual
meshes of UR's description (ur_description 3.6.0, the release whose
configuration files the book's model reads), placed where the description
places them; the post and the payload are the model's own boxes; the
clearance's floor of 0.01 m is drawn as a translucent slab against the
post's face. The drawing simulates nothing. The workbench, the tote at the
pick, the plate under the place, the cell's fence and the hall are
scenery.

    python arm_video.py --ur DIR       # writes results/video/
    python arm_video.py --ur DIR --quick
    python arm_video.py --ur DIR --export DIR2   # the scene and frames for the
                                                 # GPU renderer (render/cycles)

DIR is a checkout of Universal_Robots_ROS2_Description at the commit of
the book's description (7ce98d8), for its meshes/ur5e/visual. Requires
MuJoCo 3.16.0, NumPy, PyYAML, SciPy and Pillow, EGL (Mesa's or a GPU's)
and ffmpeg. About seventeen minutes on two cores at 1920 x 1080.

Writes results/video/ur5e-stops.mp4, its stills ur5e-stops-<stop>-<ms>ms.jpg,
and ur5e-stops-frames.csv: each frame's stop, time, the base's angle and
speed, the clearance from the post and the contacts' force, the check that
the drawing follows the record.
"""
import argparse
import csv
import math
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

import mujoco                                                     # noqa: E402
import numpy as np                                                # noqa: E402

FPS = 30
FRAME = 0.001             # [s] of simulated time per frame: 1/33 of real time at 30 fps
STOPS = (("declared", 0.120), ("empty", 0.250))   # the stops and how much of each is drawn [s]
OPEN = 40                 # frames that hold each stop's first state while the camera moves in
HOLD = 45                 # frames that hold each run's last state
STILLS = (("declared", 0.000), ("declared", "end"), ("empty", 0.000), ("empty", "deepest"))
MESHES = {"base": "base", "shoulder": "shoulder", "upper_arm": "upperarm",
          "forearm": "forearm", "wrist_1": "wrist1", "wrist_2": "wrist2", "wrist_3": "wrist3"}
BODIES = {"base": "base_link_inertia", "shoulder": "shoulder_link",
          "upper_arm": "upper_arm_link", "forearm": "forearm_link", "wrist_1": "wrist_1_link",
          "wrist_2": "wrist_2_link", "wrist_3": "wrist_3_link"}


def _down(x, digits):
    """x rounded towards minus infinity, as the book prints labels."""
    q = 10 ** digits
    return f"{math.floor(round(float(x) * q, 6)) / q + 0.0:.{digits}f}"


def _up(x, digits):
    """x rounded towards plus infinity, as the book prints angles."""
    q = 10 ** digits
    return f"{math.ceil(round(float(x) * q, 6)) / q + 0.0:.{digits}f}"


def post_force(m, d):
    """The normal force of the contacts with the post [N]."""
    post = m.geom("post").id
    f6 = np.zeros(6)
    total = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        if post in (c.geom1, c.geom2):
            mujoco.mj_contactForce(m, d, i, f6)
            total += abs(float(f6[0]))
    return {"force": np.array([total])}


def record():
    """The two stops, recorded after every step: for each, its label
    [rad/s^2], start [rad], the stop's facts (control.stop), and times [s],
    qpos, qvel, the contacts' force [N] and the clearance [m] at each
    recorded state, the stop's start (t = 0) included."""
    from arm import control, labels as al, world
    from render import shapes
    b, _, _ = al.interval_label()
    b_empty = control.capability(control.Arm(payload=0.0), al.FULL, al.SHARE)[0]
    corner = al.arm_for(al.MASS[1], al.OFFSET[1])
    out = {}
    for name, label in (("declared", b), ("empty", b_empty)):
        start = al.latest_brake(corner, label)
        rate = 1.0 / corner.m.opt.timestep
        with shapes.Recorder(rate, select=lambda m, d: d is corner.d, extra=post_force) as rec:
            facts = control.stop(corner, label, al.FULL, start=start)
        t, qpos, qvel = rec.of(corner.d)
        force = np.array([e["force"][0] for e in rec.runs[id(corner.d)]["extra"]])
        q0 = world.turned(corner.pose, start)
        qd0 = np.zeros(corner.m.nv)
        qd0[0] = al.FULL
        t = np.concatenate([[0.0], t])
        qpos = np.vstack([q0, qpos])
        qvel = np.vstack([qd0, qvel])
        force = np.concatenate([[0.0], force])
        d = mujoco.MjData(corner.m)
        clear = []
        for q in qpos:
            d.qpos[:] = q
            mujoco.mj_kinematics(corner.m, d)
            clear.append(world.clearance(corner.m, d))
        out[name] = dict(label=label, start=start, facts=facts, t=t, qpos=qpos, qvel=qvel,
                         force=force, clearance=np.array(clear), offset=float(corner.pose[0]),
                         rest=al.stop_angle(label, start))
    return out


def _ease(u):
    return 0.5 - 0.5 * math.cos(math.pi * min(1.0, max(0.0, u)))


class Shot:
    """The two stops, drawn (render.shoot): each stop's first state held
    while the camera moves in, the stop in slow motion, one state per
    millisecond, and the run's last state held."""

    def __init__(self, rec, ur, overlays=True):
        self.r, self.ur, self.overlays = rec, ur, overlays
        self.frames = []                       # (stop, index into its record)
        self.phases = []                       # ("open" | "play" | "end", progress 0 to 1)
        for name, length in STOPS:
            t = rec[name]["t"]
            idx = [int(np.argmin(np.abs(t - k * FRAME)))
                   for k in range(int(round(length / FRAME)) + 1)]
            self.frames += [(name, idx[0])] * OPEN + [(name, i) for i in idx] + \
                [(name, len(t) - 1)] * HOLD
            self.phases += [("open", j / (OPEN - 1)) for j in range(OPEN)] + \
                [("play", t[i] / length) for i in idx] + [("end", 1.0)] * HOLD
        self._views = None

    def n(self):
        return len(self.frames)

    def index(self, name, at):
        """The first frame of the stop at the time at [s], at its deepest
        point in the post ("deepest") or at the run's end ("end")."""
        t, clear = self.r[name]["t"], self.r[name]["clearance"]
        own = [j for j, f in enumerate(self.frames) if f[0] == name]
        if at == "end":
            return min(j for j in own if self.frames[j][1] == len(t) - 1)
        if at == "deepest":
            return min(own, key=lambda j: (clear[self.frames[j][1]], j))
        return min(own, key=lambda j: (abs(t[self.frames[j][1]] - at), j))

    # --- the visual model --------------------------------------------------
    def build(self):
        from arm import labels as al, model, world
        from render import collada, shapes, stage, textures as tx, urdf
        spec = model.build(payload=al.MASS[1], payload_offset=al.OFFSET[1], box_offset=al.BOX,
                           world=world.add_world)
        spec.geom("table").group = 3                  # the model's reference plane
        stage.dress(spec, extent=1.0, shadowclip=2.2, sun=(-0.32, 0.38, -0.87), sun_power=0.72,
                    fill_power=0.30, ambient=0.36, softness=0.45, haze=0.25,
                    zenith=(0.62, 0.66, 0.70), horizon=(0.80, 0.81, 0.82))
        floor = tx.add(spec, "floor", tx.concrete(1024, seed=31, tone=(128, 132, 128)), size=2.5,
                       specular=0.15, shininess=0.25)
        stage.ground(spec, floor, size=40, z=-0.78)
        steel = tx.add(spec, "bench_top", tx.breadboard(1024, seed=32), size=0.4,
                       reflectance=0.07, specular=0.45, shininess=0.55)    # the one mirror
        frame = tx.plain(spec, "bench_frame", (0.16, 0.17, 0.19, 1.0), specular=0.3,
                         shininess=0.4)
        yellow = tx.plain(spec, "fence_post", (0.93, 0.70, 0.08, 1.0), specular=0.3,
                          shininess=0.35)
        panel = tx.plain(spec, "fence_panel", (0.10, 0.11, 0.12, 0.38), specular=0.5,
                         shininess=0.7)
        wall = tx.add(spec, "hall_wall", tx.floor_tiles(1024, seed=33, tone=(196, 198, 196),
                                                        tiles=2, joint=(160, 161, 160)),
                      size=4.0, specular=0.05)
        tote = tx.plain(spec, "tote", (0.12, 0.30, 0.62, 1.0), specular=0.25, shininess=0.3)
        alu = tx.plain(spec, "nest", (0.36, 0.37, 0.39, 1.0), specular=0.5, shininess=0.5)
        wb = spec.worldbody
        box = mujoco.mjtGeom.mjGEOM_BOX
        # the workbench: its top at the model's table plane, z = 0
        shapes.visual(wb, "bench_top", box, (0.68, 0.82, 0.015), pos=(0.15, 0.0, -0.015),
                      material=steel)
        for k, (x, y) in enumerate(((-0.49, -0.78), (-0.49, 0.78), (0.79, -0.78), (0.79, 0.78))):
            shapes.visual(wb, f"bench_leg{k}", box, (0.03, 0.03, 0.375), pos=(x, y, -0.405),
                          material=frame)
        for k, (y) in enumerate((-0.78, 0.78)):
            shapes.visual(wb, f"bench_rail{k}", box, (0.66, 0.025, 0.03), pos=(0.15, y, -0.62),
                          material=frame)
        shapes.visual(wb, "bench_apron", box, (0.66, 0.80, 0.03), pos=(0.15, 0.0, -0.06),
                      material=frame)
        # the pick's tote and the place's nest on the bench (scenery)
        a = world.PICK
        c = (0.5 * math.cos(a), 0.5 * math.sin(a))
        q = shapes.quat_z(a)
        for k, (pos, size) in enumerate((((0.0, 0.0, 0.004), (0.14, 0.19, 0.004)),
                                         ((0.137, 0.0, 0.06), (0.004, 0.19, 0.056)),
                                         ((-0.137, 0.0, 0.06), (0.004, 0.19, 0.056)),
                                         ((0.0, 0.187, 0.06), (0.14, 0.004, 0.056)),
                                         ((0.0, -0.187, 0.06), (0.14, 0.004, 0.056)))):
            p = np.array(pos)
            rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            xy = rot @ p[:2]
            shapes.visual(wb, f"tote{k}", box, size, pos=(c[0] + xy[0], c[1] + xy[1], p[2]),
                          quat=q, material=tote)
        a = world.PLACE
        shapes.visual(wb, "nest", box, (0.085, 0.085, 0.006),
                      pos=(0.5 * math.cos(a), 0.5 * math.sin(a), 0.006), quat=shapes.quat_z(a),
                      material=alu)
        # the cell's fence and the hall (scenery)
        for k, (x, y) in enumerate(((-1.5, -2.4), (-1.5, 2.4), (2.5, -2.4), (2.5, 2.4),
                                    (0.5, -2.4), (0.5, 2.4), (2.5, 0.0))):
            shapes.visual(wb, f"fence_post{k}", box, (0.03, 0.03, 1.0), pos=(x, y, 0.22),
                          material=yellow)
        for k, (x, y, sx, sy) in enumerate(((-0.5, -2.4, 0.97, 0.008), (1.5, -2.4, 0.97, 0.008),
                                            (-0.5, 2.4, 0.97, 0.008), (1.5, 2.4, 0.97, 0.008),
                                            (2.5, -1.2, 0.008, 1.17), (2.5, 1.2, 0.008, 1.17))):
            shapes.visual(wb, f"fence_panel{k}", box, (sx, sy, 0.85), pos=(x, y, 0.37),
                          material=panel)
            shapes.visual(wb, f"fence_rail{k}", box, (sx + 0.01, sy + 0.012, 0.02),
                          pos=(x, y, 1.22), material=yellow)
        for name, pos, size in (("wall_back", (0.0, 6.0, 1.7), (8.0, 0.1, 2.5)),
                                ("wall_front", (0.0, -6.0, 1.7), (8.0, 0.1, 2.5)),
                                ("wall_left", (-6.0, 0.0, 1.7), (0.1, 8.0, 2.5)),
                                ("wall_right", (6.0, 0.0, 1.7), (0.1, 8.0, 2.5))):
            shapes.visual(wb, name, box, size, pos=pos, material=wall)
        # the model's own post and payload, in paint
        post = tx.plain(spec, "post_paint", (0.62, 0.14, 0.12, 1.0), specular=0.35,
                        shininess=0.45)
        load = tx.plain(spec, "payload_paint", (0.86, 0.52, 0.16, 1.0), specular=0.4,
                        shininess=0.5)
        spec.geom("post").material = post
        spec.geom("payload").material = load
        # UR's visual meshes, where the description places them
        desc = model.description()
        origins = model.mesh_origins(desc)
        visual = os.path.join(self.ur, "meshes", "ur5e", "visual")
        elements = []
        for link, mesh in MESHES.items():
            xyz, rmat = origins[link]
            elements.append((BODIES[link], os.path.join(visual, mesh + ".dae"), xyz,
                             model.quat(rmat), [0.6, 0.6, 0.6, 1.0], [1.0, 1.0, 1.0]))
        mats = {}

        def material(rgba, path):
            key = tuple(round(c, 3) for c in rgba)
            if key not in mats:
                name = f"ur_{len(mats)}"
                r, g, b_ = key[:3]
                if b_ > r + 0.15:                              # UR's blue caps
                    tx.plain(spec, name, (r, g, b_, 1.0), specular=0.35, shininess=0.45)
                elif r > 0.7:                                  # bright metal
                    tx.plain(spec, name, (r, g, b_, 1.0), specular=0.7, shininess=0.7)
                elif r > 0.15:                                 # the joints' grey
                    tx.plain(spec, name, (min(1.0, 1.35 * r),) * 3 + (1.0,), specular=0.4,
                             shininess=0.5)
                else:                                          # black plastics and rubber
                    tx.plain(spec, name, (0.06, 0.06, 0.065, 1.0), specular=0.25,
                             shininess=0.3)
                mats[key] = name
            return mats[key]

        urdf.attach(spec, elements, material, collada, name="ur5e")
        return spec.compile()

    # --- one frame ---------------------------------------------------------
    def pose(self, m, d, i):
        name, k = self.frames[i]
        d.qpos[:] = self.r[name]["qpos"][k]
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)

    def focus(self, m, lookat):
        from render import stage
        stage.follow(m, lookat, distance=6.0)

    def views(self):
        """For each stop: from a view of the whole arm in to the gap between
        the payload and the post while the first state is held, then a slow
        drift closer during the stop, the same for both stops."""
        if self._views is not None:
            return self._views
        from arm import world
        a = world.post_angle() + math.radians(9.0)
        gap = np.array([0.5 * math.cos(a), 0.5 * math.sin(a), 0.19])
        wide = (np.array([0.16, -0.30, 0.32]), 1.75, 128.0, -13.0)
        near = (gap, 0.82, 110.0, -9.5)
        close = (gap, 0.62, 104.0, -8.0)

        def mix(p, q, w):
            return ((1 - w) * p[0] + w * q[0], (1 - w) * p[1] + w * q[1],
                    (1 - w) * p[2] + w * q[2], (1 - w) * p[3] + w * q[3])
        out = []
        for phase, u in self.phases:
            if phase == "open":
                out.append(mix(wide, near, _ease(u)))
            else:
                out.append(mix(near, close, _ease(u)))
        self._views = out
        return out

    def view(self, i):
        return self.views()[i]

    def overlay(self, i, m, d):
        if not self.overlays:
            return None
        from arm import labels as al, world
        from render import film
        a = world.post_angle()
        hx, hy, hz = (s / 2 for s in world.POST_SIZE)
        u = np.array([-math.sin(a), math.cos(a), 0.0])     # towards the place
        c = np.array([0.5 * math.cos(a), 0.5 * math.sin(a), hz]) + (hy + al.FLOOR / 2) * u
        mat = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0],
                        [0.0, 0.0, 1.0]]).ravel()

        def draw(scene):
            if scene.ngeom >= scene.maxgeom:
                return
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX,
                                np.array([hx + 0.004, al.FLOOR / 2, hz + 0.002]), c, mat,
                                np.array([1.0, 0.82, 0.10, 0.30], dtype=np.float32))
            film._mark(g, True)     # the fields that mjv_initGeom leaves unset
            scene.ngeom += 1
        return draw

    def caption(self, i):
        name, k = self.frames[i]
        r = self.r[name]
        t = r["t"][k]
        speed = abs(math.degrees(r["qvel"][k][0]))
        clear = r["clearance"][k]
        force = r["force"][k]
        if name == "declared":
            title = "UR5e with 4.95 kg, stopped from 180 deg/s under the declared label"
        else:
            title = "The same arm and payload under the empty arm's label"
        first = (f"label {_down(r['label'], 2)} rad/s^2, from {_up(math.degrees(r['start']), 3)} "
                 "deg, the latest angle that it allows")
        if clear >= 0:                      # rounded down, as the book prints clearances
            where = f"clearance from the post {_down(1000 * clear, 1)} mm"
        else:
            where = f"in the post by {_up(-1000 * clear, 1)} mm"
        if self.phases[i][0] == "end":
            line = f"t = {1000 * t:5.1f} ms, the run's end: at rest, {where}"
        elif self.phases[i][0] == "open":
            line = f"t = {1000 * t:5.1f} ms, the stop's start: base {speed:5.1f} deg/s, {where}"
        else:
            line = f"t = {1000 * t:5.1f} ms    base {speed:5.1f} deg/s    {where}"
            if force > 0.5:
                line += f"    contact force {force:3.0f} N"
        return (title, [first, line])

    def caption2(self, i):
        return (None, ["Simulated in MuJoCo 3.16 (0.5 ms steps), Section 14.2",
                       "Slow motion: 1 ms per frame, 1/33 of real time",
                       "Drawn from the record with UR's visual meshes",
                       "Yellow: the clearance's floor, 10 mm from the post"])


def main():
    ap = argparse.ArgumentParser(description="Section 14.2's two stops of the UR5e as a video")
    ap.add_argument("--ur", required=True,
                    help="a checkout of Universal_Robots_ROS2_Description")
    ap.add_argument("--quick", action="store_true", help="640 x 360, no stills")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--export", metavar="DIR", help="write the scene and frames for the GPU "
                    "renderer (render/cycles) instead of drawing them")
    args = ap.parse_args()
    from render import shoot
    out_dir = os.path.join(HERE, "results", "video")
    os.makedirs(out_dir, exist_ok=True)
    rec = record()
    for name, r in rec.items():
        f = r["facts"]
        print(f"{name}: label {_down(r['label'], 2)} rad/s^2 from "
              f"{_up(math.degrees(r['start']), 3)} deg; commanded rest "
              f"{_up(math.degrees(r['rest']), 3)} deg; overrun {_up(math.degrees(f['beyond']), 3)} "
              f"deg; least clearance {_down(f['clearance'], 4)} m; largest force on the post "
              f"{_up(r['force'].max(), 0)} N; {len(r['t'])} states (the book's rounding)")
    shot = Shot(rec, args.ur)
    n = shot.n()
    if args.export:
        from render import export
        stills = {}
        for name, at in STILLS:
            i = shot.index(name, at)
            ms = 1000 * rec[name]["t"][shot.frames[i][1]]
            stills[i] = f"ur5e-stops-{name}-{ms:05.1f}ms"
        export.bundle(shot, n, args.export, fps=FPS, name="ur5e-stops", stills=stills)
        return
    size = (640, 360) if args.quick else (1920, 1080)
    path = os.path.join(out_dir, "ur5e-stops.mp4" if not args.quick else "ur5e-stops-quick.mp4")
    shoot.render(shot, n, path, size=size, supersample=1 if args.quick else 2,
                 workers=args.workers)
    frames, rate, w, h = shoot.probe(path)
    print(f"{path}: {frames} frames at {rate:g} per second, {w} x {h}")
    with open(os.path.join(out_dir, "ur5e-stops-frames.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(("frame", "stop", "t", "base_angle_deg", "base_speed_deg_s", "clearance_m",
                     "post_force_n"))
        for i, (name, k) in enumerate(shot.frames):
            r = rec[name]
            wr.writerow((i, name, shoot.fixed(r['t'][k], 4),
                         shoot.fixed(math.degrees(r['qpos'][k][0] - r['offset']), 4),
                         shoot.fixed(math.degrees(r['qvel'][k][0]), 3),
                         shoot.fixed(r['clearance'][k], 5), shoot.fixed(r['force'][k], 1)))
    if not args.quick:
        clean = Shot(rec, args.ur)
        for name, at in STILLS:
            i = clean.index(name, at)
            ms = 1000 * rec[name]["t"][clean.frames[i][1]]
            shoot.still(clean, i, os.path.join(out_dir, f"ur5e-stops-{name}-{ms:05.1f}ms.jpg"),
                        size=(1600, 900))


if __name__ == "__main__":
    main()
