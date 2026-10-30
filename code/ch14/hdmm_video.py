"""
Section 14.3's reference run of the heavy-duty mobile manipulator as a
video, and its stills.

Physical AI, Together, Chapter 14 (Section 14.3; Figure 14.9).

The script runs the configuration "reference" of hdmm_governed.py again
with the book's modules unchanged and records the simulated machine at 30
frames per second of simulated time (render.shapes.Recorder, which only
copies the state after each of MuJoCo's steps). The simulation, like
Section 14.3's, holds the machine in the frame of the base's commanded
motion along the route, gravity tilted by its acceleration and the turning
frame's forces on every body: the drawing places that frame on the yard at
the commanded arc length of each frame (route.pose_at) and the simulated
bodies in it, the base with its creep, pitch and roll on its contacts and
the arm on its servos, as the record has them, and simulates nothing.
Every part of the machine is drawn on those bodies to dimensions of ours:
the model has four contact spheres instead of wheels, a box for its base,
capsules for its boom and stick and a cube for its tool, none of them
drawn; the drawn wheels turn and steer as the commanded motion would turn
them (two joints each, which only the drawing has), and the hydraulic
cylinders span their pins. The yard is the route of route.py with its
dashed centre line, its wet patch around the second turn, the wall at the
stop line and the drop zone, within 0.5 m of the line; the edge lines, the
containers, the pallets, the cones and the warehouses are scenery. Before
it draws, the script checks that the camera keeps 2 m from the scenery and
sees the machine in every frame (sight()).

    python hdmm_video.py            # writes results/video/
    python hdmm_video.py --quick
    python hdmm_video.py --export DIR2   # the scene and frames for the
                                         # GPU renderer (render/cycles)

Requires MuJoCo 3.16.0, NumPy, SciPy and Pillow, EGL (Mesa's or a GPU's)
and ffmpeg. About forty minutes on two cores at 1920 x 1080.

Writes results/video/hdmm-reference.mp4, its stills, and
hdmm-reference-frames.csv: the frames' times, the commanded arc length and
the simulated joints, the check that the drawing follows the record.
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
END = 49.0                       # [s] the video ends here; the machine rests from 46.2 s
STILLS = (9.6, 21.0, 35.0, 47.0)          # [s] lifting, the two turns, at rest
WHEEL_R = 0.50                   # [m] the drawn wheels' radius (the model has contact points)


def record():
    """The reference run of hdmm_governed.py, recorded: run times [s], the
    simulated qpos (base in the frame of the commanded motion, then slew,
    lift, tilt), the commanded arc length, speed and planned pose at those
    times, the machine's settled origin and yaw, and the run's alarms."""
    import allovi
    import hdmm_governed as hg
    from render import shapes
    with shapes.Recorder(FPS, select=lambda m, d: m.nq == 10) as rec:
        cfg, task, world = hg.configuration("reference")
        run = allovi.run(cfg, hg.PERIODS)
    t, qpos, _, _, _ = rec.longest()
    mach = world.machine
    log = mach.arrays()
    # the run's time: the simulated time less the settling before the run
    settle = round(float(mach.d.time - mach.t), 9)
    tau = t - settle
    keep = (tau >= -1e-9) & (tau <= END + 1e-9)
    tau, qpos = tau[keep], qpos[keep]
    lt = np.concatenate([[0.0], log["t"]])
    s = np.interp(tau, lt, np.concatenate([[log["s"][0]], log["s"]]))
    v = np.interp(tau, lt, np.concatenate([[0.0], log["v"]]))
    planned = np.stack([np.interp(tau, lt, np.concatenate([[log[k][0]], log[k]]))
                        for k in ("p0", "p1", "p2")], axis=1)
    return dict(t=tau, qpos=qpos, s=s, v=v, planned=planned, origin=mach.origin.copy(),
                yaw0=float(mach.yaw0), alarms=list(run.alarms), held=hg.HELD)


def qz(angle):
    return np.array([math.cos(0.5 * angle), 0.0, 0.0, math.sin(0.5 * angle)])


def qmul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


class Shot:
    """The reference run, drawn (render.shoot)."""

    def __init__(self, rec, captions=True):
        self.r = rec
        self.captions = captions
        self._views = None
        self._world = None

    # --- the visual model --------------------------------------------------
    def build(self):
        from hdmm import model, route
        from render import shapes, stage, textures as tx
        spec = model.build(payload=self.r["held"], pose=None)
        stage.dress(spec, extent=8.0, shadowclip=8.0, sun=(-0.35, -0.55, -0.76),
                    zenith=(0.28, 0.48, 0.78), horizon=(0.84, 0.88, 0.92), haze=0.55)
        for g in spec.geoms:
            if g.name != "ground":
                g.group = 3                          # the model's own shapes, not drawn
        ground = next(g for g in spec.geoms if g.name == "ground")
        asphalt = tx.add(spec, "asphalt", tx.asphalt(2048, seed=31), size=7.0,
                         specular=0.08, shininess=0.1)
        ground.material = asphalt
        ground.size = [400.0, 400.0, 0.1]
        ground.rgba = [1, 1, 1, 1]
        wet = tx.add(spec, "wet", tx.wet_asphalt(1024, seed=32), size=4.0, reflectance=0.28,
                     specular=0.6, shininess=0.85)
        paint = tx.add(spec, "paint", tx.paint(256, seed=33), size=0.6, specular=0.15)
        yellow_paint = tx.add(spec, "yellow_paint", tx.paint(256, seed=34, tone=(226, 180, 30)),
                              size=0.6, specular=0.15)
        concrete = tx.add(spec, "concrete", tx.concrete(1024, seed=35), size=3.0,
                          specular=0.05)
        stripes = tx.add(spec, "chevrons", tx.chevrons(512, stripes=8), size=2.0, specular=0.2)
        wood = tx.add(spec, "wood", tx.wood(512, seed=36), size=1.2, specular=0.05)
        crate = tx.add(spec, "crate", tx.wood(512, seed=37, tone=(150, 112, 70)), size=0.6,
                       specular=0.05)
        wrap = tx.plain(spec, "wrap", (0.80, 0.82, 0.84, 1.0), specular=0.5, shininess=0.6)
        box_brown = tx.plain(spec, "carton", (0.62, 0.48, 0.32, 1.0), specular=0.05)
        steel = tx.plain(spec, "steel", (0.45, 0.46, 0.48, 1.0), specular=0.6, shininess=0.7)
        dark_steel = tx.plain(spec, "dark_steel", (0.18, 0.19, 0.20, 1.0), specular=0.5,
                              shininess=0.6)
        machine_yellow = tx.plain(spec, "machine_yellow", (0.93, 0.66, 0.06, 1.0), specular=0.45,
                                  shininess=0.55)
        rubber = tx.plain(spec, "rubber", (0.07, 0.07, 0.07, 1.0), specular=0.08, shininess=0.1)
        rim = tx.plain(spec, "rim", (0.70, 0.70, 0.68, 1.0), specular=0.6, shininess=0.7)
        glass = tx.plain(spec, "glass", (0.22, 0.30, 0.36, 0.72), specular=0.9, shininess=0.95,
                         reflectance=0.15)
        orange = tx.plain(spec, "cone_orange", (0.93, 0.36, 0.05, 1.0), specular=0.25)
        white = tx.plain(spec, "cone_band", (0.93, 0.93, 0.91, 1.0), specular=0.4)
        black = tx.plain(spec, "cone_base", (0.10, 0.10, 0.10, 1.0), specular=0.1)
        cont = [tx.add(spec, f"container{k}", tx.corrugated(512, seed=40 + k, tone=c, ribs=10,
                                                           rust=0.35), size=2.6, specular=0.3)
                for k, c in enumerate(((36, 82, 140), (150, 40, 32), (40, 110, 70),
                                       (190, 140, 40)))]
        # the route's markings and the wet patch
        ss = np.arange(-5.0, route.S_LINE + 0.01, 0.25)
        xy = np.array([route.pose_at(x)[:2] for x in ss])
        for side in (-1, 1):
            shapes.ribbon(spec, f"edge{side}", xy, 0.15, 0.003, paint, offset=side * 2.6)
        for k, s0 in enumerate(np.arange(-4.0, route.S_LINE - 3.0, 6.0)):
            seg = np.array([route.pose_at(x)[:2] for x in np.arange(s0, s0 + 3.0 + 0.01, 0.25)])
            shapes.ribbon(spec, f"dash{k}", seg, 0.12, 0.003, paint)
        wseg = np.array([route.pose_at(x)[:2]
                         for x in np.arange(route.WET_FROM, route.WET_TO + 0.01, 0.25)])
        shapes.ribbon(spec, "wet_patch", wseg, 7.0, 0.0015, wet)
        # the drop zone, within 0.5 m of the stop line, and the wall at the line
        lx, ly, lh = route.pose_at(route.S_LINE)
        c, sn = math.cos(lh), math.sin(lh)
        across = np.array([[lx - 0.25 * c + 3.2 * sn, ly - 0.25 * sn - 3.2 * c],
                           [lx - 0.25 * c - 3.2 * sn, ly - 0.25 * sn + 3.2 * c]])
        shapes.ribbon(spec, "drop_zone", across, 0.5, 0.004, yellow_paint)
        wall = spec.worldbody.add_body()
        wall.name = "wall"
        wall.pos = [lx + 0.35 * c, ly + 0.35 * sn, 0.0]
        wall.quat = qz(lh).tolist()
        shapes.visual(wall, "wall_block", mujoco.mjtGeom.mjGEOM_BOX, (0.35, 5.0, 1.0),
                      pos=(0.0, 0.0, 1.0), material=concrete)
        shapes.visual(wall, "wall_face", mujoco.mjtGeom.mjGEOM_BOX, (0.005, 3.2, 0.25),
                      pos=(-0.352, 0.0, 0.6), material=stripes)
        # scenery beside the route
        for k, (s0, off) in enumerate(((8.0, 7.5), (22.0, 8.5), (36.0, 7.5), (68.0, -7.5),
                                       (95.0, 7.5))):
            x, y, h = route.pose_at(s0)
            px, py = x - off * math.sin(h), y + off * math.cos(h)
            shapes.container(spec, f"container{k}", (px, py), h + (0.0 if k % 2 else 0.05),
                             cont[k % 4], dark_steel)
        for k, (s0, off) in enumerate(((26.0, 5.2), (30.0, 5.4), (75.0, -5.0), (100.0, 5.2),
                                       (12.0, 5.0))):
            x, y, h = route.pose_at(s0)
            shapes.pallet_stack(spec, f"pallets{k}", (x - off * math.sin(h), y + off * math.cos(h)),
                                h + 0.3 * k, wood, box_brown, layers=2 + k % 2,
                                wrap=wrap if k % 2 else None)
        for k, s0 in enumerate(np.arange(route.S_TURN1 - 2.0, route.TURNS[1].end + 4.0, 4.0)):
            x, y, h = route.pose_at(s0)
            for side in (-1, 1):
                if (route.TURNS[0].start - 3 <= s0 <= route.TURNS[0].end + 3) or \
                        (route.TURNS[1].start - 3 <= s0 <= route.TURNS[1].end + 3):
                    shapes.cone(spec, f"cone{k}{side}",
                                (x - side * 3.0 * math.sin(h), y + side * 3.0 * math.cos(h)),
                                orange, white, black)
        # warehouses round the yard, so that the asphalt does not meet the sky
        cladding = tx.add(spec, "cladding", tx.corrugated(512, seed=46, tone=(156, 160, 164),
                                                          ribs=16), size=2.0, specular=0.25)
        roof = tx.plain(spec, "roof", (0.32, 0.34, 0.36, 1.0), specular=0.2)
        for k, (x, y, hx, hy, hz, yaw) in enumerate(((30.0, -38.0, 25.0, 9.0, 7.0, 0.0),
                                                     (-30.0, 15.0, 10.0, 22.0, 8.0, 0.0),
                                                     (95.0, -10.0, 14.0, 20.0, 9.0, 0.0),
                                                     (40.0, 62.0, 30.0, 10.0, 8.0, 0.0),
                                                     (120.0, 45.0, 10.0, 25.0, 10.0, 0.0),
                                                     (75.0, 75.0, 18.0, 8.0, 7.0, 0.3))):
            b = spec.worldbody.add_body()
            b.name = f"warehouse{k}"
            b.pos = [x, y, 0.0]
            b.quat = qz(yaw).tolist()
            shapes.visual(b, f"warehouse{k}_walls", mujoco.mjtGeom.mjGEOM_BOX, (hx, hy, hz),
                          pos=(0.0, 0.0, hz), material=cladding)
            shapes.visual(b, f"warehouse{k}_roof", mujoco.mjtGeom.mjGEOM_BOX,
                          (hx + 0.3, hy + 0.3, 0.25), pos=(0.0, 0.0, 2 * hz + 0.25), material=roof)
        # the machine's drawn parts: on the base, the turret, the boom, the stick, the tool
        base = next(b for b in spec.bodies if b.name == "base")
        z0 = model.SPHERE
        V = mujoco.mjtGeom
        shapes.visual(base, "frame", V.mjGEOM_BOX, (1.45, 0.58, 0.18), pos=(0.0, 0.0, 0.62 - z0),
                      material=dark_steel)
        shapes.visual(base, "deck", V.mjGEOM_BOX, (1.5, 0.66, 0.12), pos=(0.0, 0.0, 0.88 - z0),
                      material=machine_yellow)
        shapes.visual(base, "hood", V.mjGEOM_BOX, (0.25, 0.62, 0.22), pos=(-1.25, 0.0, 1.22 - z0),
                      material=machine_yellow)
        shapes.visual(base, "grille", V.mjGEOM_BOX, (0.01, 0.48, 0.14), pos=(-1.505, 0.0, 1.18 - z0),
                      material=dark_steel)
        shapes.visual(base, "cab", V.mjGEOM_BOX, (0.35, 0.32, 0.42), pos=(-0.60, -0.33, 1.42 - z0),
                      material=machine_yellow)
        shapes.visual(base, "cab_glass_front", V.mjGEOM_BOX, (0.005, 0.27, 0.24),
                      pos=(-0.245, -0.33, 1.56 - z0), material=glass)
        shapes.visual(base, "cab_glass_side", V.mjGEOM_BOX, (0.28, 0.005, 0.22),
                      pos=(-0.60, -0.655, 1.56 - z0), material=glass)
        shapes.visual(base, "cab_roof", V.mjGEOM_BOX, (0.39, 0.36, 0.02), pos=(-0.60, -0.33, 1.85 - z0),
                      material=dark_steel)
        for sx in (-1, 1):
            shapes.visual(base, f"bumper{sx}", V.mjGEOM_BOX, (0.06, 0.70, 0.08),
                          pos=(sx * 1.52, 0.0, 0.70 - z0), material=dark_steel)
            for sy in (-1, 1):
                shapes.visual(base, f"fender{sx}{sy}", V.mjGEOM_BOX, (0.36, 0.18, 0.02),
                              pos=(sx * model.AXLE, sy * model.HALF_TRACK, 1.02 - z0),
                              material=dark_steel)
        self.wheel_names = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                name = f"{'f' if sx > 0 else 'r'}{'l' if sy > 0 else 'r'}"
                steer = base.add_body()
                steer.name = f"steer_{name}"
                steer.pos = [sx * model.AXLE, sy * model.HALF_TRACK, WHEEL_R - z0]
                j = steer.add_joint()
                j.name, j.type, j.axis = f"steer_{name}", mujoco.mjtJoint.mjJNT_HINGE, [0, 0, 1]
                steer.explicitinertial = True
                steer.mass, steer.inertia = 1.0, [0.1, 0.1, 0.1]
                spin = steer.add_body()
                spin.name = f"spin_{name}"
                j = spin.add_joint()
                j.name, j.type, j.axis = f"spin_{name}", mujoco.mjtJoint.mjJNT_HINGE, [0, 1, 0]
                spin.explicitinertial = True
                spin.mass, spin.inertia = 1.0, [0.1, 0.1, 0.1]
                rot = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]
                shapes.visual(spin, f"tire_{name}", V.mjGEOM_CYLINDER, (WHEEL_R, 0.19),
                              quat=rot, material=rubber)
                shapes.visual(spin, f"rim_{name}", V.mjGEOM_CYLINDER, (0.29, 0.195),
                              quat=rot, material=rim)
                for k in range(6):
                    a = k * math.pi / 3
                    shapes.visual(spin, f"lug_{name}{k}", V.mjGEOM_CYLINDER, (0.035, 0.198),
                                  pos=(0.18 * math.cos(a), 0.0, 0.18 * math.sin(a)), quat=rot,
                                  material=dark_steel)
                for k in range(16):
                    a = k * math.pi / 8
                    th = 0.5 * math.pi - a           # the block's z along the radius
                    r = WHEEL_R - 0.015
                    shapes.visual(spin, f"tread_{name}{k}", V.mjGEOM_BOX, (0.035, 0.192, 0.02),
                                  pos=(r * math.cos(a), 0.0, r * math.sin(a)),
                                  quat=[math.cos(th / 2), 0.0, math.sin(th / 2), 0.0],
                                  material=rubber)
                self.wheel_names.append(name)
        turret = next(b for b in spec.bodies if b.name == "turret")
        shapes.visual(turret, "slew_ring", V.mjGEOM_CYLINDER, (0.55, 0.06), pos=(0.0, 0.0, -0.27),
                      material=dark_steel)
        shapes.visual(turret, "platform", V.mjGEOM_BOX, (0.62, 0.5, 0.16), pos=(-0.12, 0.0, -0.18),
                      material=machine_yellow)
        shapes.visual(turret, "counterweight", V.mjGEOM_BOX, (0.22, 0.5, 0.22),
                      pos=(-0.62, 0.0, -0.08), material=dark_steel)
        shapes.visual(turret, "pivot", V.mjGEOM_CYLINDER, (0.13, 0.26), pos=(0.0, 0.0, 0.0),
                      quat=[math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0], material=steel)
        boom = next(b for b in spec.bodies if b.name == "boom")
        shapes.visual(boom, "boom_beam", V.mjGEOM_BOX, (1.2, 0.16, 0.2), pos=(1.2, 0.0, 0.0),
                      material=machine_yellow)
        shapes.visual(boom, "boom_plate", V.mjGEOM_BOX, (1.1, 0.165, 0.02), pos=(1.15, 0.0, 0.2),
                      material=dark_steel)
        shapes.visual(boom, "elbow", V.mjGEOM_CYLINDER, (0.11, 0.2), pos=(2.4, 0.0, 0.0),
                      quat=[math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0], material=steel)
        stick = next(b for b in spec.bodies if b.name == "stick")
        shapes.visual(stick, "stick_beam", V.mjGEOM_BOX, (0.98, 0.12, 0.15), pos=(0.98, 0.0, 0.0),
                      material=machine_yellow)
        tool = next(b for b in spec.bodies if b.name == "tool")
        shapes.visual(tool, "tool_plate", V.mjGEOM_BOX, (0.03, 0.24, 0.2), pos=(-0.17, 0.0, 0.0),
                      material=dark_steel)
        shapes.visual(tool, "fork_l", V.mjGEOM_BOX, (0.19, 0.03, 0.015), pos=(0.0, 0.14, -0.19),
                      material=dark_steel)
        shapes.visual(tool, "fork_r", V.mjGEOM_BOX, (0.19, 0.03, 0.015), pos=(0.0, -0.14, -0.19),
                      material=dark_steel)
        shapes.visual(tool, "crate", V.mjGEOM_BOX, (0.19, 0.23, 0.17), pos=(0.0, 0.0, -0.005),
                      material=crate)
        m = spec.compile()
        self._ids = {n: (m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"steer_{n}")],
                         m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"spin_{n}")])
                     for n in self.wheel_names}
        return m

    # --- one frame ---------------------------------------------------------
    def world_pose(self, i):
        """The base's pose on the yard at frame i: the commanded frame at the
        arc length s, and the simulated base in it."""
        from hdmm import route
        q = self.r["qpos"][i]
        x, y, h = route.pose_at(self.r["s"][i])
        o, yaw0 = self.r["origin"], self.r["yaw0"]
        dx, dy = q[0] - o[0], q[1] - o[1]
        c0, s0 = math.cos(-yaw0), math.sin(-yaw0)
        dx, dy = c0 * dx - s0 * dy, s0 * dx + c0 * dy
        ch, sh = math.cos(h), math.sin(h)
        pos = np.array([x + ch * dx - sh * dy, y + sh * dx + ch * dy, q[2]])
        quat = qmul(qz(h), qmul(qz(-yaw0), q[3:7]))
        return pos, quat, h

    def pose(self, m, d, i):
        from hdmm import model, route
        pos, quat, h = self.world_pose(i)
        q = self.r["qpos"][i]
        d.qpos[:] = 0.0
        d.qpos[0:3], d.qpos[3:7], d.qpos[7:10] = pos, quat, q[7:10]
        s = self.r["s"][i]
        k, _ = route.curvature_at(s)
        delta = math.atan(model.AXLE * k)
        for n, (a_steer, a_spin) in self._ids.items():
            left = n[1] == "l"
            arc = s + (-1.0 if left else 1.0) * model.HALF_TRACK * h
            d.qpos[a_steer] = delta if n[0] == "f" else -delta
            d.qpos[a_spin] = arc / WHEEL_R
        mujoco.mj_forward(m, d)

    def focus(self, m, lookat):
        from render import stage
        stage.follow(m, lookat, distance=60.0)

    def views(self):
        if self._views is not None:
            return self._views
        from render import film
        # (time, azimuth relative to the heading, elevation, distance) [s, deg, deg, m]
        keys = ((0.0, 35.0, -16.0, 15.0), (5.0, 70.0, -14.0, 13.0), (7.5, 95.0, -10.0, 11.0),
                (12.0, 100.0, -11.0, 11.5), (16.0, 20.0, -20.0, 15.0), (19.0, -20.0, -30.0, 17.0),
                (25.0, -40.0, -26.0, 16.0), (29.0, 30.0, -20.0, 15.0), (32.0, 70.0, -28.0, 16.0),
                (40.0, 60.0, -24.0, 15.0), (43.0, 80.0, -14.0, 13.0), (46.0, 90.0, -12.0, 12.0),
                (49.0, 95.0, -12.0, 12.5))
        kt = np.array([k[0] for k in keys])
        rows = []
        for i, t in enumerate(self.r["t"]):
            pos, _, h = self.world_pose(i)
            look = pos + np.array([0.0, 0.0, 1.2])
            rows.append([look[0], look[1], look[2], np.interp(t, kt, [k[3] for k in keys]),
                         math.degrees(h) + np.interp(t, kt, [k[1] for k in keys]),
                         np.interp(t, kt, [k[2] for k in keys])])
        self._views = film.smooth_path(rows, 1.0 / FPS, tau=(0.4, 0.4, 0.4, 1.5, 1.5, 1.5))
        return self._views

    def view(self, i):
        return self.views()[i]

    def overlay(self, i, m, d):
        """The hydraulic cylinders between their pins, drawn as scene geoms
        (barrel and rod), since they span two bodies."""
        from render import film

        def at(body, local):
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
            return d.xpos[b] + d.xmat[b].reshape(3, 3) @ np.asarray(local, float)

        lift_a, lift_b = at("turret", (0.35, 0.0, -0.22)), at("boom", (1.05, 0.0, -0.2))
        tilt_a, tilt_b = at("boom", (1.6, 0.0, 0.2)), at("stick", (-0.25, 0.0, 0.16))

        def draw(scene):
            for a, b, r in ((lift_a, lift_b, 0.075), (tilt_a, tilt_b, 0.06)):
                mid = a + 0.55 * (b - a)
                film.line(scene, a, mid, r, (0.20, 0.21, 0.22, 1.0), decor=False)
                film.line(scene, mid - 0.05 * (b - a), b, 0.55 * r, (0.80, 0.82, 0.84, 1.0),
                          decor=False)
        return draw

    def caption(self, i):
        if not self.captions:
            return None
        from hdmm import model, route
        from render import film
        t, s, v = self.r["t"][i], self.r["s"][i], self.r["v"][i]
        p = self.r["planned"][i]
        carry, reach = np.array(model.POSES["carry"]), np.array(model.POSES["reach"])
        arm = ("carrying" if np.allclose(p, carry, atol=1e-6) else
               "reaching" if np.allclose(p, reach, atol=1e-6) else "lifting the arm")
        ground = route.ground_at(s).name
        turn = route.turn_on(s - 0.01, s + 0.01)
        where = f"in the {turn.name} turn" if turn is not None else "on a straight"
        rest = " · at rest" if v <= 1e-6 and t > 10.0 else ""
        return ("Heavy-duty mobile manipulator under the governor",
                [f"t = {film.number(t, 1, 4)} s · {film.number(v, 2, 4)} m/s · {arm} · {where}, "
                 f"{ground} ground{rest}",
                 f"payload {self.r['held']:g} kg"])

    def caption2(self, i):
        if not self.captions:
            return None
        return (None, ["Simulated in MuJoCo 3.16 (1 ms steps), Section 14.3, the reference run",
                       "Drawn from the record; the machine's parts drawn on its bodies"])


def sight(shot, clear=2.0):
    """The frames whose camera comes within clear of something drawn that is
    not the machine (the ground and the paint on it aside), or whose line of
    sight to the machine's base, turret or tool meets something else drawn:
    a list of (time, reason), empty if the camera sees the machine
    throughout."""
    from render import film
    m = shot.build()
    d = mujoco.MjData(m)
    base = m.body("base").id
    machine = {b for b in range(m.nbody) if m.body_rootid[b] == base}
    shot.pose(m, d, 0)
    flat = (int(mujoco.mjtGeom.mjGEOM_PLANE), int(mujoco.mjtGeom.mjGEOM_MESH))
    scenery = [g for g in range(m.ngeom) if m.geom_group[g] <= 2
               and m.geom_bodyid[g] not in machine
               and not (int(m.geom_type[g]) in flat and d.geom_xpos[g][2] < 0.05)]
    groups = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)
    hit = np.zeros(1, dtype=np.int32)
    out = []
    for i, view in enumerate(shot.views()):
        shot.pose(m, d, i)
        cam = film.camera_position(*view)
        near = film.clearances(m, d, cam, scenery)
        k = int(np.argmin(near))
        if near[k] < clear:
            out.append((shot.r["t"][i], f"camera {near[k]:.2f} m from {m.geom(scenery[k]).name}"))
        for body in ("base", "turret", "tool"):
            target = d.xpos[m.body(body).id] + (np.array([0.0, 0.0, 0.6]) if body == "base" else 0.0)
            ray = target - cam
            length = float(np.linalg.norm(ray))
            dist = mujoco.mj_ray(m, d, cam, ray / length, groups, 1, -1, hit)
            if 0 <= dist < length - 0.3 and m.geom_bodyid[hit[0]] not in machine:
                out.append((shot.r["t"][i], f"sight to the {body} blocked by {m.geom(hit[0]).name}"))
    return out


def main():
    ap = argparse.ArgumentParser(description="Section 14.3's reference run as a video")
    ap.add_argument("--quick", action="store_true", help="640 x 360, no stills")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--export", metavar="DIR", help="write the scene and frames for the GPU "
                    "renderer (render/cycles) instead of drawing them")
    args = ap.parse_args()
    from render import shoot
    out_dir = os.path.join(HERE, "results", "video")
    os.makedirs(out_dir, exist_ok=True)
    rec = record()
    shot = Shot(rec)
    blocked = sight(shot)
    if blocked:
        raise SystemExit(f"the camera loses the machine in {len(blocked)} frames, first at "
                         f"{blocked[0][0]:.2f} s: {blocked[0][1]}")
    n = len(rec["t"])
    if args.export:
        from render import export
        export.bundle(shot, n, args.export, fps=FPS, name="hdmm-reference",
                      stills={int(round(s * FPS)): f"hdmm-reference-{s:04.1f}s" for s in STILLS})
        return
    size = (640, 360) if args.quick else (1920, 1080)
    path = os.path.join(out_dir, "hdmm-reference.mp4" if not args.quick else "hdmm-quick.mp4")
    shoot.render(shot, n, path, size=size, supersample=1 if args.quick else 2,
                 workers=args.workers)
    frames, rate, w, h = shoot.probe(path)
    print(f"{path}: {frames} frames at {rate:g} per second, {w} x {h}; the record: {n} "
          f"states over {rec['t'][-1]:.2f} s; alarms {rec['alarms'] or 'none'}")
    with open(os.path.join(out_dir, "hdmm-reference-frames.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(("t", "s", "v", "slew", "lift", "tilt", "creep_x", "creep_y"))
        for k in range(n):
            q = rec["qpos"][k]
            wr.writerow((shoot.fixed(rec['t'][k], 4), shoot.fixed(rec['s'][k], 4),
                         shoot.fixed(rec['v'][k], 4), shoot.fixed(q[7], 5), shoot.fixed(q[8], 5),
                         shoot.fixed(q[9], 5), shoot.fixed(q[0] - rec['origin'][0], 5),
                         shoot.fixed(q[1] - rec['origin'][1], 5)))
    if not args.quick:
        clean = Shot(rec, captions=False)
        for s in STILLS:
            i = int(round(s * FPS))
            shoot.still(clean, i, os.path.join(out_dir, f"hdmm-reference-{s:04.1f}s.jpg"),
                        size=(1600, 900))


if __name__ == "__main__":
    main()
