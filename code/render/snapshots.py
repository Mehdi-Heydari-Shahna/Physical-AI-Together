#!/usr/bin/env python3
"""
Snapshots of the book's simulated robots, one of each, for the figure of
the robots in Chapter 1 (Section 1.1, Who this book is for).

Physical AI, Together, Chapter 1 (the figure of the robots); the runs are
those of Chapters 11 and 13 to 15.

Each robot is drawn at one recorded instant of the run that its chapter
draws as a video, with the visual model of the chapter's video script (or,
for the cart, of ../ch11/mujoco_stills.py) and a camera of its own, without
captions or annotations. Nothing is simulated for a snapshot beyond what the
video script records: the Husky's, the UR5e's, the mobile manipulator's,
the excavator's and the Go2's runs are recorded again by their scripts'
record(), which run the chapters' configurations unchanged; the Apollo's is
read from ../ch15/results/humanoid-frames.npz, which humanoid_carry.py
writes; the cart is placed at a state of the exact run of Chapter 3 by the
adapter's own Sim.place. The drawing then calls mj_forward only.

Usage, from this folder (MuJoCo 3.16.0, NumPy, SciPy, Pillow and EGL, as
the video scripts):

    MUJOCO_GL=egl python3 snapshots.py --clearpath DIR --ur DIR \\
        --menagerie DIR --team-impedance DIR          # writes results/snapshots/
    MUJOCO_GL=egl python3 snapshots.py --only cart excavator

--clearpath, --ur, --menagerie and --team-impedance are checkouts of
clearpath_common, Universal_Robots_ROS2_Description, MuJoCo Menagerie and
Team-Impedance-Control, as for the video scripts (their READMEs give the
commits); each is needed only for its robot. The images are 1600 x 900
JPEG files, drawn at twice that size and scaled down (film.Film).
"""
import argparse
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)                                  # the package render

import mujoco                                             # noqa: E402
import numpy as np                                        # noqa: E402

from render import shoot, stage                           # noqa: E402

SIZE = (1600, 900)
ROBOTS = ("cart", "husky", "ur5e", "hdmm", "excavator", "go2", "apollo")


def chapter(*names):
    """Puts the chapters' folders first on the import path."""
    for name in reversed(names):
        path = os.path.join(CODE, name)
        if path not in sys.path:
            sys.path.insert(0, path)


class Hero:
    """A chapter's shot (render.shoot) drawn at one frame, with a camera of
    its own: (lookat [m], distance [m], azimuth [deg], elevation [deg])."""

    def __init__(self, shot, camera, overlay=None):
        self.shot, self.camera, self._overlay = shot, camera, overlay

    def build(self):
        return self.shot.build()

    def pose(self, m, d, i):
        self.shot.pose(m, d, i)

    def view(self, i):
        lookat, dist, az, el = self.camera
        return np.asarray(lookat, float), dist, az, el

    def focus(self, m, lookat):
        if hasattr(self.shot, "focus"):
            self.shot.focus(m, lookat)
        else:
            stage.follow(m, lookat)

    def overlay(self, i, m, d):
        return self._overlay(m, d) if self._overlay is not None else None

    def caption(self, i):
        return None


class CartShot:
    """The cart of Section 11.1 placed at one state (p, v, l) of the exact run
    of Chapter 3, with the visual model of ../ch11/mujoco_stills.py."""

    def __init__(self, state):
        chapter("ch11")
        from cart_mujoco.scene import Design
        from cart_mujoco.sim import Sim
        self.sim, self.state = Sim(Design()), state

    def build(self):
        import mujoco_stills
        return mujoco_stills.visual_model(self.sim.xml)

    def pose(self, m, d, i):
        self.sim.place(*self.state)
        d.qpos[:], d.qvel[:] = self.sim.data.qpos, self.sim.data.qvel
        mujoco.mj_forward(m, d)

    def focus(self, m, lookat):
        stage.follow(m, lookat, distance=30.0)


def cart(args):
    """At the start of period 78 of the run of Chapter 3 (t = 7.8 s): on the
    slope at 1 m/s towards the wall, the tool folded, the rangefinder's ray
    reaching the wall. The ray that MuJoCo draws is a hairline at the
    figure's size, so it is drawn again over it, as a thin blue capsule from
    the rangefinder's site along its axis, as long as the sensor reads."""
    import importlib
    from render import film
    shot = CartShot((23.405, 1.0, 0.0))
    stills = importlib.import_module("mujoco_stills")

    def overlay(m, d):
        base = stills.annotations(m, d)
        site = m.site("tip").id
        start = d.site_xpos[site].copy()
        axis = d.site_xmat[site].reshape(3, 3)[:, 2]
        end = start + float(d.sensor("range").data[0]) * axis

        def draw(scene):
            base(scene)
            film.line(scene, start, end, 0.012, [0.176, 0.353, 0.533, 1.0])
        return draw
    return Hero(shot, ((23.7, 0.0, -0.38), 2.6, 140.0, -24.0), overlay=overlay), 0


def husky(args):
    """The governed crossing of the yard with the map's labels (Section 13.2), at 11.5 s."""
    chapter("ch13", "ch10")
    import husky_video as v
    rec = v.record()
    shot = v.Shot(rec, args.clearpath, captions=False)
    i = 345                                               # 11.5 s at 30 frames per second
    return Hero(shot, (shot.view(i)[0], 4.0, 90.0, -18.0)), i


def ur5e(args):
    """The stop under the declared label with 4.95 kg (Section 14.2), 60 ms
    into the stop."""
    chapter("ch14", "ch10")
    import arm_video as v
    rec = v.record()
    shot = v.Shot(rec, args.ur, overlays=False)
    i = 100
    return Hero(shot, (shot.view(i)[0], 2.0, 210.0, -15.0)), i


def hdmm(args):
    """The heavy-duty mobile manipulator's reference run (Section 14.3), at
    5.0 s: 3 m/s, the arm lifted by 20 degrees."""
    chapter("ch14", "ch10")
    import hdmm_video as v
    rec = v.record()
    shot = v.Shot(rec, captions=False)
    i = 150
    return Hero(shot, (shot.view(i)[0], 11.0, 135.0, -17.0)), i


def excavator(args):
    """The PC138US-8's onset run with the load raised to 102 % of the static
    tipping load (Section 14.1), at 9.43 s of the run: the rear of the tracks
    off the ground, the machine leaning forward by 15 degrees."""
    chapter("ch14")
    import excavator_video as v
    rec = v.record()
    shot = v.Shot(rec, captions=False)
    i = 584
    run, k, _, _ = shot.run_of(i)
    base = np.asarray(run["qpos"][k][:3], float)
    return Hero(shot, (base + (2.0, 0.0, 1.0), 9.0, 90.0, -12.0)), i


def go2(args):
    """The Go2's governed lift of its front-right foot while it is pushed
    with 10 N (Section 15.2), at 2.30 s."""
    chapter("ch15", "ch10")
    import quad_video as v
    t, qpos, push, _ = v.record()
    shot = v.Shot(args.menagerie, t, qpos, push, overlays=False)
    i = 69                                                # 2.30 s at 30 frames per second
    return Hero(shot, (shot.view(i)[0], 1.3, 60.0, -13.0)), i


def apollo(args):
    """One Apollo carrying a beam (Section 15.3), at 33.9 s, stepping with its
    left foot."""
    chapter("ch15")
    import humanoid_video as v
    rec = dict(np.load(os.path.join(CODE, "ch15", "results", "humanoid-frames.npz")))
    shot = v.Shot(rec, args.team_impedance, captions=False, overlays=False)
    i = 1017                                              # 33.9 s at 30 frames per second
    pelvis = np.asarray(rec["qpos"][i][:3], float)
    return Hero(shot, (pelvis + (0.0, 0.4, 0.3), 3.0, 30.0, -15.0)), i


NEEDS = {"husky": "clearpath", "ur5e": "ur", "go2": "menagerie", "apollo": "team_impedance"}


def main():
    ap = argparse.ArgumentParser(description="snapshots of the book's simulated robots")
    ap.add_argument("--clearpath", help="a checkout of clearpath_common (the Husky)")
    ap.add_argument("--ur", help="a checkout of Universal_Robots_ROS2_Description (the UR5e)")
    ap.add_argument("--menagerie", help="a checkout of MuJoCo Menagerie (the Go2)")
    ap.add_argument("--team-impedance", dest="team_impedance",
                    help="a checkout of Team-Impedance-Control (the Apollo)")
    ap.add_argument("--only", nargs="+", choices=ROBOTS, help="draw only these robots")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "snapshots"))
    args = ap.parse_args()
    names = args.only or ROBOTS
    missing = [n for n in names if n in NEEDS and not getattr(args, NEEDS[n])]
    if missing:
        flags = ", ".join("--" + NEEDS[n].replace("_", "-") for n in missing)
        raise SystemExit(f"{', '.join(missing)}: give {flags}, or leave them out with --only")
    os.makedirs(args.out, exist_ok=True)
    for name in names:
        t0 = time.time()
        hero, i = globals()[name](args)
        path = os.path.join(args.out, f"robots-{name}.jpg")
        shoot.still(hero, i, path, size=SIZE, supersample=2, quality=92)
        print(f"{path}: frame {i}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
