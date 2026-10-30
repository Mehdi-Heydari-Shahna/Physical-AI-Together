"""
Section 15.2's pushed run of the Go2 as a video, and its stills.

Physical AI, Together, Chapter 15 (Section 15.2; Figure 15.5).

The script runs the configuration "pushed" of quad_governed.py with the
book's modules unchanged, records the simulated state at 30 frames per
second of simulated time (render.shapes.Recorder, which only copies the
state after each of MuJoCo's steps), and then draws the record: the robot
with the visual meshes of MuJoCo Menagerie's go2.xml, which the book's
model deletes, on a stage of a concrete yard, with the push on the
trunk drawn as an arrow while it acts, and the triangle of the three other
feet on the floor while FR may carry less than its share. The drawing
simulates nothing: each frame's joint angles and base pose are the
record's.

    python quad_video.py --menagerie DIR     # writes results/video/
    python quad_video.py --menagerie DIR --quick
    python quad_video.py --menagerie DIR --export DIR2   # the scene and frames for the
                                                         # GPU renderer (render/cycles)

DIR is a checkout of MuJoCo Menagerie at the commit of the record
(f3b6013), whose unitree_go2/go2.xml is the book's description/go2.xml with
its meshes. Requires MuJoCo 3.16.0, NumPy, SciPy and Pillow, EGL (Mesa's
or a GPU's) and ffmpeg. About ten minutes on two cores at 1920 x 1080.

Writes results/video/go2-pushed.mp4, its stills go2-pushed-<t>s.jpg, and
go2-pushed-frames.csv: the frames' times and the base's position, the
check that the drawing follows the record.
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
STILLS = (2.3, 4.6, 5.6, 8.0)           # [s] of the run, on the frames: shift, hold, touchdown, home
THREE = (3.2, 6.2)                      # [s] FR may carry less than its share (Table 15.2)
PHASES = ((0.0, "standing at home"), (1.8, "shifting the centre of mass"),
          (3.2, "unloading FR"), (3.6, "lifting FR"), (4.0, "FR held up"),
          (5.0, "lowering FR"), (5.4, "touchdown"), (5.8, "reloading FR"),
          (6.2, "returning home"), (7.6, "home, its command at rest"))


def record():
    """The pushed run of quad_governed.py, recorded: times from the run's
    start [s], qpos, the push on the trunk [N], and the run's record."""
    import allovi
    import quad_governed as qg
    from render import shapes
    kw = dict(pushes=qg.PUSHES)
    with shapes.Recorder(FPS, select=lambda m, d: m.nq == 19,
                         extra=lambda m, d: {"push": d.xfrc_applied[:, :3].sum(axis=0)}) as rec:
        cfg, task, world = qg.configuration("pushed", **kw)
        run = allovi.run(cfg, qg.PERIODS)
    t, qpos, _, extra, _ = rec.longest()
    t0 = world.machine.t0
    keep = t >= t0 - 1e-9
    push = np.array([e["push"] for e in extra])[keep]
    alarms = list(run.alarms)
    return t[keep] - t0, qpos[keep], push, alarms


def phase(t):
    name = PHASES[0][1]
    for t0, n in PHASES:
        if t >= t0 - 1e-9:
            name = n
    return name


class Shot:
    """The pushed run, drawn (render.shoot)."""

    def __init__(self, menagerie, t, qpos, push, overlays=True):
        self.menagerie, self.t, self.qpos, self.push = menagerie, t, qpos, push
        self.overlays = overlays
        self._views = None

    # --- the visual model --------------------------------------------------
    def build(self):
        from render import shapes, stage, textures as tx
        spec = mujoco.MjSpec.from_file(os.path.join(self.menagerie, "unitree_go2", "go2.xml"))
        stage.dress(spec, extent=2.0, shadowclip=6.0, sun=(-0.55, 0.30, -0.78))
        slab = tx.add(spec, "slab", tx.floor_tiles(1024, seed=6, tiles=4), size=3.0,
                      reflectance=0.03, specular=0.12, shininess=0.2)
        stage.ground(spec, slab, size=120)
        mat = tx.add(spec, "mat", tx.asphalt(512, seed=11, tone=(44, 45, 47)), size=1.0,
                     specular=0.05, shininess=0.05)
        g = spec.worldbody.add_geom()
        g.name, g.type, g.material = "rubber_mat", mujoco.mjtGeom.mjGEOM_BOX, mat
        g.size, g.pos = [1.1, 0.8, 0.003], [0.02, 0.0, -0.0028]
        g.contype = g.conaffinity = 0
        wall = tx.add(spec, "wall", tx.floor_tiles(1024, seed=21, tone=(176, 174, 168), tiles=2,
                                                    joint=(140, 139, 135)), size=6.0,
                      specular=0.05)
        for name, pos, size in (("wall_back", (0.0, 9.0, 3.0), (11.2, 0.2, 3.0)),
                                ("wall_front", (0.0, -9.0, 3.0), (11.2, 0.2, 3.0)),
                                ("wall_left", (-11.0, 0.0, 3.0), (0.2, 9.0, 3.0)),
                                ("wall_right", (11.0, 0.0, 3.0), (0.2, 9.0, 3.0))):
            w = spec.worldbody.add_geom()
            w.name, w.type, w.material = name, mujoco.mjtGeom.mjGEOM_BOX, wall
            w.size, w.pos = list(size), list(pos)
            w.contype = w.conaffinity = 0
        orange = tx.plain(spec, "cone_orange", (0.93, 0.36, 0.05, 1.0), specular=0.25,
                          shininess=0.3)
        white = tx.plain(spec, "cone_band", (0.93, 0.93, 0.91, 1.0), specular=0.4,
                         shininess=0.5)
        black = tx.plain(spec, "cone_base", (0.10, 0.10, 0.10, 1.0), specular=0.1)
        for k, (x, y) in enumerate(((1.7, 1.4), (-1.5, 1.6), (2.0, -1.7))):
            shapes.cone(spec, f"cone{k}", (x, y), orange, white, black)
        return spec.compile()

    # --- one frame ---------------------------------------------------------
    def pose(self, m, d, i):
        d.qpos[:] = self.qpos[i]
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)

    def focus(self, m, lookat):
        from render import stage
        stage.follow(m, lookat)

    def views(self):
        """The camera's views for every frame: a critically damped follow of
        keyed views (azimuth, elevation, distance) and of the trunk."""
        if self._views is not None:
            return self._views
        from render import film
        keys = ((0.0, 15.0, -10.0, 2.5), (1.6, 38.0, -12.0, 1.8), (3.2, 58.0, -9.0, 1.4),
                (6.2, 72.0, -10.0, 1.45), (7.6, 100.0, -12.0, 1.7), (10.0, 140.0, -15.0, 2.1))
        kt = np.array([k[0] for k in keys])
        base = self.qpos[:, :3] + np.array([0.0, 0.0, -0.05])
        rows = [[base[i, 0], base[i, 1], base[i, 2], np.interp(t, kt, [k[3] for k in keys]),
                 np.interp(t, kt, [k[1] for k in keys]), np.interp(t, kt, [k[2] for k in keys])]
                for i, t in enumerate(self.t)]
        self._views = film.smooth_path(rows, 1.0 / FPS, tau=(0.5, 0.5, 0.5, 0.9, 0.9, 0.9))
        return self._views

    def view(self, i):
        return self.views()[i]

    def overlay(self, i, m, d):
        if not self.overlays:
            return None
        from render import film
        t = self.t[i]
        f = self.push[i]
        mag = float(np.linalg.norm(f[:2]))
        feet = [d.geom_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)].copy()
                for n in ("FL", "RL", "RR")]
        com = d.subtree_com[1].copy()

        def draw(scene):
            if mag > 0.3:
                u = np.array([f[0], f[1], 0.0]) / mag
                tip = com - 0.16 * u
                film.arrow(scene, tip - (0.04 * mag) * u, tip, 0.02, (1.0, 0.42, 0.05, 1.0))
            if THREE[0] <= t <= THREE[1]:
                for a, b in ((0, 1), (1, 2), (2, 0)):
                    p, q = feet[a].copy(), feet[b].copy()
                    p[2] = q[2] = 0.0015
                    film.line(scene, p, q, 0.0035, (0.30, 0.85, 0.50, 0.75))
                c = com.copy()
                c[2] = 0.002
                film.dot(scene, c, 0.018, (0.98, 0.98, 0.98, 0.95))
        return draw

    def caption(self, i):
        t = self.t[i]
        from render import film
        lines = [f"t = {film.number(t, 1, 4)} s    {phase(t)}"]
        mag = float(np.linalg.norm(self.push[i][:2]))
        if mag > 0.3:
            lines.append(f"push on the trunk: {mag:4.1f} N (the label's bound: 10 N)")
        else:
            lines.append("no push")
        return ("Unitree Go2 under the governor: lifting its front-right foot", lines)

    def caption2(self, i):
        return (None, ["Simulated in MuJoCo 3.16 (1 ms steps), Section 15.2, run 'pushed'",
                       "Drawn from the record with Menagerie's meshes"])


def main():
    ap = argparse.ArgumentParser(description="Section 15.2's pushed run as a video")
    ap.add_argument("--menagerie", required=True, help="a checkout of MuJoCo Menagerie")
    ap.add_argument("--quick", action="store_true", help="640 x 360, no stills")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--export", metavar="DIR", help="write the scene and frames for the GPU "
                    "renderer (render/cycles) instead of drawing them")
    args = ap.parse_args()
    from render import shoot
    out_dir = os.path.join(HERE, "results", "video")
    os.makedirs(out_dir, exist_ok=True)
    t, qpos, push, alarms = record()
    shot = Shot(args.menagerie, t, qpos, push)
    n = len(t)
    if args.export:
        from render import export
        export.bundle(shot, n, args.export, fps=FPS, name="go2-pushed",
                      stills={int(round(s * FPS)): f"go2-pushed-{s:.2f}s" for s in STILLS})
        return
    size = (640, 360) if args.quick else (1920, 1080)
    path = os.path.join(out_dir, "go2-pushed.mp4" if not args.quick else "go2-pushed-quick.mp4")
    shoot.render(shot, n, path, size=size, supersample=1 if args.quick else 2,
                 workers=args.workers)
    frames, rate, w, h = shoot.probe(path)
    print(f"{path}: {frames} frames at {rate:g} per second, {w} x {h}; the record: {n} "
          f"states over {t[-1]:.2f} s; alarms {alarms or 'none'}")
    with open(os.path.join(out_dir, "go2-pushed-frames.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(("t", "base_x", "base_y", "base_z", "push_x", "push_y"))
        for k in range(n):
            wr.writerow((shoot.fixed(t[k], 4), shoot.fixed(qpos[k, 0], 5),
                         shoot.fixed(qpos[k, 1], 5), shoot.fixed(qpos[k, 2], 5),
                         shoot.fixed(push[k, 0], 3), shoot.fixed(push[k, 1], 3)))
    if not args.quick:
        clean = Shot(args.menagerie, t, qpos, push)
        for s in STILLS:
            i = int(round(s * FPS))
            shoot.still(clean, i, os.path.join(out_dir, f"go2-pushed-{s:.2f}s.jpg"),
                        size=(1600, 900))


if __name__ == "__main__":
    main()
