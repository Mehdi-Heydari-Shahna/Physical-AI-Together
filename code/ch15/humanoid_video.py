"""
Section 15.3's run of one Apollo carrying a beam as a video, and its stills.

Physical AI, Together, Chapter 15 (Section 15.3; Figure S15.1).

The script draws the record that humanoid_carry.py writes
(results/humanoid-frames.npz: the simulated state at 30 frames per second
of simulated time, with the soles' and palms' contact forces) in the
repository's own scene, built by its package teamimp from the same
checkout (--team-impedance DIR): the construction site, the trestles and
the beam, and the Apollo with MuJoCo Menagerie's meshes as the repository
decimates them. It draws, on the ground, the polygon of the soles that
carry the robot (green; the stance sole alone in single support) and the
centre of pressure of the ground's forces, the zero-moment point (white,
with a pole of 0.3 m over it, since in single support the foot hides it),
and for 0.4 s after each touchdown's first contact a ring round the foot
(orange). The drawing simulates nothing: each frame's state is the
record's. The repository's scene writes its textures into the checkout's
assets/textures, as the repository does when it runs.

    python humanoid_video.py --team-impedance DIR          # results/video/
    python humanoid_video.py --team-impedance DIR --quick
    python humanoid_video.py --team-impedance DIR --part 1/2   # then 2/2,
    python humanoid_video.py --team-impedance DIR --join 2     # and join
    python humanoid_video.py --team-impedance DIR --export DIR2

The video also shows, in its lower right corner, the soles from above with
the zero-moment point and, in single support, its distance inside the
stance sole's edges. Requires MuJoCo 3.16.0, NumPy, SciPy and Pillow, EGL
(Mesa's or a GPU's) and ffmpeg; daqp is not needed for drawing. About ninety
minutes on two cores at 1920 x 1080 (--part splits the drawing).

Writes results/video/apollo-carry.mp4, its stills apollo-carry-<t>s.jpg,
and apollo-carry-frames.csv: each frame's time, phase, the pelvis's and the
beam's positions, the check that the drawing follows the record.
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
RING = 0.4                       # [s] the ring round a foot after its touchdown
POLE = 0.30                      # [m] the pole over the zero-moment point
WORDS = (("settle", "standing"), ("go", "walking to the beam"), ("pre-reach", "crouching to the beam"),
         ("reach-in", "reaching under the beam"), ("contact", "palms to the beam"),
         ("load-transfer", "taking the beam's weight off the trestles"), ("lift", "lifting the beam"),
         ("carry", "carrying the beam"), ("place", "lowering the beam"),
         ("unload", "setting the beam down"), ("release", "letting go"), ("stow", "standing up"),
         ("back", "stepping back"), ("done", "standing"))
STILLS = (("go2-step-l", "single"), ("lift", "last"), ("carry", "edge"), ("back1-step-l", "single"))
EDGE = 0.0005                    # [m] a zero-moment point this close to the sole's edge is at it


def convex(points):
    """The convex hull of points (x, y, ...) in the plane, anticlockwise."""
    xy = np.array([p[:2] for p in points], float)
    xy = xy[np.lexsort((xy[:, 1], xy[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in xy:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in xy[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def words(phase):
    return next(w for p, w in WORDS if phase.startswith(p))


class Shot:
    def __init__(self, rec, checkout, captions=True, overlays=True):
        self.r = rec
        self.checkout = checkout
        self.captions = captions
        self.overlays = overlays
        self._views = None

    def build(self):
        from humanoid import one_robot
        mods = one_robot.load(self.checkout)
        cfg, _ = one_robot.adapt(mods)
        scene = mods["scene"]
        h = scene.PROFILES[cfg.profile][0]
        x_sup, yb = float(self.r["x_sup"][0]), float(self.r["yb"][0])
        sup = [("A0", -x_sup, 0.0, cfg.zA, "wood"), ("A1", x_sup, 0.0, cfg.zA, "wood"),
               ("B0", -x_sup, yb, cfg.zB, "rubber"), ("B1", x_sup, yb, cfg.zB, "rubber")]
        spec = scene.build_spec(1, "plant", supports=sup,
                                beam_pos=(0.0, 0.0, cfg.zA + h / 2 + 0.0005), profile=cfg.profile,
                                beam_len=float(self.r["length"][0]), mass=float(self.r["mass"][0]),
                                background=True)
        spec.visual.quality.shadowsize = 8192
        m = spec.compile()
        if m.nq != self.r["qpos"].shape[1]:
            raise SystemExit(f"the scene has {m.nq} coordinates, the record {self.r['qpos'].shape[1]}")
        self._ids = dict(pelvis=m.body("r0_base_link").id, beam=m.body("beam").id,
                         sole={s: m.site(f"r0_{s}_sole").id for s in "lr"})
        return m

    def pose(self, m, d, i):
        d.qpos[:] = self.r["qpos"][i]
        mujoco.mj_forward(m, d)

    def views(self):
        if self._views is not None:
            return self._views
        from render import film
        t, q = self.r["t"], self.r["qpos"]
        start = {}
        for k, p in enumerate(self.r["phase"]):
            start.setdefault(str(p), t[k])
        ts = lambda name: start[name]                                  # noqa: E731
        # (time, azimuth, elevation, distance, lookat height) [s, deg, deg, m, m]: from
        # behind the robot and to its left (azimuths between 0 and 90 degrees), so that the
        # trestles ahead of it do not stand between the camera and the robot; straight
        # behind it once it steps back from the trestles that it has passed
        keys = ((0.0, 62.0, -16.0, 4.6, 0.90), (ts("go2-step-r") - 2.0, 50.0, -20.0, 4.2, 0.85),
                (ts("pre-reach"), 38.0, -14.0, 3.1, 0.85), (ts("lift"), 34.0, -12.0, 2.9, 0.90),
                (ts("carry1-step-l") + 1.0, 55.0, -26.0, 3.8, 0.80),
                (ts("carry2-step-l"), 70.0, -34.0, 3.6, 0.60), (ts("carry3-step-r"), 75.0, -34.0, 3.6, 0.60),
                (ts("place") - 1.0, 60.0, -16.0, 3.4, 0.90), (ts("release"), 48.0, -14.0, 3.2, 0.90),
                (ts("back1-step-l"), 86.0, -16.0, 4.2, 0.85), (t[-1], 88.0, -15.0, 4.6, 0.85))
        kt = [k[0] for k in keys]
        rows = []
        for i in range(len(t)):
            p = q[i, 0:3]
            rows.append([p[0], p[1] + 0.30, np.interp(t[i], kt, [k[4] for k in keys]),
                         np.interp(t[i], kt, [k[3] for k in keys]),
                         np.interp(t[i], kt, [k[1] for k in keys]),
                         np.interp(t[i], kt, [k[2] for k in keys])])
        self._views = film.smooth_path(rows, 1.0 / FPS, tau=(0.6, 0.6, 0.6, 1.0, 1.0, 1.0))
        return self._views

    def view(self, i):
        return self.views()[i]

    def corners(self, d, s):
        from humanoid_carry import SOLE
        p = d.site_xpos[self._ids["sole"][s]].copy()
        R = d.site_xmat[self._ids["sole"][s]].reshape(3, 3)
        out = []
        for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
            c = p + R @ np.array([a * SOLE[0], b * SOLE[1], 0.0])
            c[2] = 0.004
            out.append(c)
        return out

    def overlay(self, i, m, d):
        if not self.overlays:
            return None
        from render import film
        c = self.r["contacts"][i]
        t = self.r["t"][i]
        fz = {"l": c[0], "r": c[1]}
        cop = {"l": c[2:4], "r": c[4:6]}
        feet = [s for s in "lr" if fz[s] > 1.0]
        pts = [p for s in feet for p in self.corners(d, s)]
        rings = [("l" if int(side) == 0 else "r") for (t0, side) in self.r["touch"]
                 if t0 <= t <= t0 + RING]

        def draw(scene):
            if pts:
                poly = convex(pts)
                for k in range(len(poly)):
                    a = np.r_[poly[k], 0.004]
                    b = np.r_[poly[(k + 1) % len(poly)], 0.004]
                    film.line(scene, a, b, 0.007, (0.30, 0.85, 0.50, 0.9))
                total = sum(fz[s] for s in feet)
                z = sum(fz[s] * cop[s] for s in feet) / total
                film.dot(scene, np.r_[z, 0.005], 0.024, (0.98, 0.98, 0.98, 0.95))
                # a pole over the point, which the stance foot would hide on the ground
                film.line(scene, np.r_[z, 0.005], np.r_[z, POLE], 0.005, (0.98, 0.98, 0.98, 0.8))
                film.dot(scene, np.r_[z, POLE], 0.02, (0.98, 0.98, 0.98, 0.95), height=0.004)
            for s in rings:
                p = d.site_xpos[self._ids["sole"][s]].copy()
                p[2] = 0.002
                film.dot(scene, p, 0.17, (1.0, 0.45, 0.08, 0.45), height=0.0005)
        return draw

    def margin(self, i, d):
        """In single support, the zero-moment point's distance inside the
        stance sole's edges [m] and the stance foot; otherwise None."""
        from humanoid_carry import SOLE
        c = self.r["contacts"][i]
        on = [s for s in "lr" if c["lr".index(s)] > 1.0]
        if len(on) != 1:
            return None
        s = on[0]
        z = np.asarray(c[2:4] if s == "l" else c[4:6])
        sid = self._ids["sole"][s]
        loc = d.site_xmat[sid].reshape(3, 3).T @ (np.r_[z, 0.0] - d.site_xpos[sid])
        return min(SOLE[0] - abs(loc[0]), SOLE[1] - abs(loc[1])), s

    def panel(self, i, m, d):
        """The soles seen from above, the robot's forward up: each sole
        (filled while it carries), the polygon of those that carry (green),
        the zero-moment point (white), and in single support its distance
        inside the stance sole's edges."""
        if not self.captions:
            return None
        from humanoid_carry import SOLE
        c = self.r["contacts"][i]
        fz = {"l": c[0], "r": c[1]}
        cop = {"l": np.asarray(c[2:4]), "r": np.asarray(c[4:6])}
        on = [s for s in "lr" if fz[s] > 1.0]
        corners = {s: [p[:2] for p in self.corners(d, s)] for s in "lr"}
        mid = 0.5 * (np.mean(corners["l"], axis=0) + np.mean(corners["r"], axis=0))
        rel = lambda p: (float(p[0] - mid[0]), float(p[1] - mid[1]))      # noqa: E731
        polys = [([rel(p) for p in corners[s]], (0.85, 0.85, 0.85, 1.0),
                  (0.40, 0.42, 0.45, 1.0) if s in on else None) for s in "lr"]
        dots, text = [], ""
        if on:
            hull = convex([p for s in on for p in corners[s]])
            polys.append(([rel(p) for p in hull], (0.30, 0.85, 0.50, 1.0), None))
            z = sum(fz[s] * cop[s] for s in on) / sum(fz[s] for s in on)
            dots.append((rel(z), (0.98, 0.98, 0.98, 1.0)))
            if len(on) == 1:
                margin, s = self.margin(i, d)
                margin = math.floor(max(margin, 0.0) * 1000)
                foot = "left" if s == "l" else "right"
                text = (f"ZMP {margin:.0f} mm inside the {foot} sole's edges" if margin >= 1
                        else f"ZMP at the edge of the {foot} sole")
            else:
                text = "on both feet"
        return dict(polys=polys, dots=dots, text=text, half=0.25)

    def caption(self, i):
        if not self.captions:
            return None
        from render import film
        c = self.r["contacts"][i]
        t = self.r["t"][i]
        phase = str(self.r["phase"][i])
        feet = [s for s in "lr" if c["lr".index(s)] > 1.0]
        support = ("on both feet" if len(feet) == 2 else
                   f"on the {'left' if feet == ['l'] else 'right'} foot" if feet else "")
        palms = c[6] + c[7]
        w = float(self.r["weight"][0])
        line2 = (f"beam {float(self.r['mass'][0]):.1f} kg, aluminium, HEA 200 section, "
                 f"{float(self.r['length'][0]):g} m")
        if palms > 1.0:
            line2 += f" · the palms carry {film.number(100 * palms / w, 0, 3)} % of its weight"
        return ("One Apollo carries a beam: the repository's two-loop controller, one robot",
                [f"t = {film.number(t, 1, 4)} s · {words(phase)} · {support}", line2])

    def caption2(self, i):
        if not self.captions:
            return None
        return (None, ["Simulated in MuJoCo 3.16 (1 ms steps), Section 15.3",
                       "Drawn from the record; green: the soles that carry the robot,",
                       "white: the zero-moment point, under the pole; orange: a touchdown"])


def stills_of(shot):
    """The frames of the book's stills: in single support in the middle of a
    phase's, the last of a phase, or, for "edge", the middle of the loaded
    frames whose zero-moment point is at the stance sole's edge (the
    nearest if none is)."""
    rec = shot.r
    out = {}
    phase = np.array([str(p) for p in rec["phase"]])
    feet = rec["mode"][:, 0]
    m = shot.build()
    d = mujoco.MjData(m)
    for name, where in STILLS:
        if where == "edge":
            idx = [k for k in range(len(phase)) if phase[k].startswith(name) and feet[k] == 1]
            near = []
            for k in idx:
                shot.pose(m, d, k)
                res = shot.margin(k, d)
                if res is not None:
                    near.append((res[0], k))
            at = [k for g, k in near if g < EDGE]
            i = at[len(at) // 2] if at else min(near)[1]
        else:
            idx = np.nonzero(phase == name)[0]
            if where == "single":
                single = idx[feet[idx] == 1]
                i = int(single[len(single) // 2]) if len(single) else int(idx[len(idx) // 2])
            else:
                i = int(idx[-1])
        out[int(i)] = f"apollo-carry-{rec['t'][i]:04.1f}s"
    return out


def main():
    ap = argparse.ArgumentParser(description="Section 15.3's Apollo as a video")
    ap.add_argument("--team-impedance", required=True, help="a checkout of Team-Impedance-Control")
    ap.add_argument("--quick", action="store_true", help="640 x 360, no stills")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--part", metavar="K/N", help="draw only the K-th of N parts of the video "
                    "(apollo-carry-partK.mp4), for machines on which one run of the whole is too "
                    "long; --join N then joins them and writes the frames file and the stills")
    ap.add_argument("--join", type=int, metavar="N")
    ap.add_argument("--export", metavar="DIR", help="write the scene and frames for the GPU "
                    "renderer (render/cycles) instead of drawing them")
    args = ap.parse_args()
    from render import shoot
    out_dir = os.path.join(HERE, "results", "video")
    os.makedirs(out_dir, exist_ok=True)
    rec = dict(np.load(os.path.join(HERE, "results", "humanoid-frames.npz")))
    shot = Shot(rec, args.team_impedance)
    n = len(rec["t"])
    stills = stills_of(shot)
    if args.export:
        from render import export
        export.bundle(shot, n, args.export, fps=FPS, name="apollo-carry", stills=stills)
        return
    size = (640, 360) if args.quick else (1920, 1080)
    path = os.path.join(out_dir, "apollo-carry.mp4" if not args.quick else "apollo-carry-quick.mp4")
    if args.part:
        k, parts = (int(x) for x in args.part.split("/"))
        frames = np.array_split(np.arange(n), parts)[k - 1]
        part = os.path.join(out_dir, f"apollo-carry-part{k}.mp4")
        shoot.render(shot, n, part, size=size, supersample=1 if args.quick else 2,
                     workers=args.workers, frames=frames)
        print(f"{part}: frames {frames[0]} to {frames[-1]}")
        return
    if args.join:
        parts = [os.path.join(out_dir, f"apollo-carry-part{k}.mp4") for k in range(1, args.join + 1)]
        shoot.join(parts, path)
        for p in parts:
            os.remove(p)
    else:
        shoot.render(shot, n, path, size=size, supersample=1 if args.quick else 2,
                     workers=args.workers)
    frames, rate, w, h = shoot.probe(path)
    if frames != n:
        raise SystemExit(f"{path} holds {frames} frames, not {n}")
    print(f"{path}: {frames} frames at {rate:g} per second, {w} x {h}; the record: {n} states "
          f"over {rec['t'][-1]:.2f} s, run with the QP solver {rec['qp'][0]} and the repository "
          f"at {rec['commit'][0][:7]}")
    with open(os.path.join(out_dir, "apollo-carry-frames.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(("t", "phase", "pelvis_x", "pelvis_y", "pelvis_z", "beam_x", "beam_y", "beam_z"))
        for k in range(n):
            q = rec["qpos"][k]
            wr.writerow((shoot.fixed(rec["t"][k], 4), str(rec["phase"][k]),
                         *(shoot.fixed(v, 5) for v in q[0:3]), *(shoot.fixed(v, 5) for v in q[-7:-4])))
    if not args.quick:
        clean = Shot(rec, args.team_impedance, captions=False)
        for i, name in stills.items():
            shoot.still(clean, i, os.path.join(out_dir, f"{name}.jpg"), size=(1600, 900))


if __name__ == "__main__":
    main()
