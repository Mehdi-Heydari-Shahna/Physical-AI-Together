"""
Shapes and frames for the videos' scenes: flat ribbons along a path (lane
marks, a wet patch), quaternions, and the recording of a run's states.

Physical AI, Together, Chapters 13 to 15 (the videos of their governed runs).
"""
import math

import mujoco
import numpy as np


def quat_z(angle):
    return np.array((math.cos(0.5 * angle), 0.0, 0.0, math.sin(0.5 * angle)))


def quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def ribbon(spec, name, xy, width, z, material, offset=0.0, group=1):
    """A flat strip of the given width along the polyline xy (n x 2), its
    middle offset sideways (to the left of the direction of travel) by
    offset, at the height z: a mesh geom on the world body, visual only."""
    xy = np.asarray(xy, float)
    keep = np.concatenate([[True], np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-9])
    xy = xy[keep]
    d = np.gradient(xy, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    left = np.stack([-d[:, 1], d[:, 0]], axis=1)
    mid = xy + offset * left
    a = mid + 0.5 * width * left
    b = mid - 0.5 * width * left
    n = len(xy)
    verts = np.zeros((2 * n, 3))
    verts[0::2, :2], verts[1::2, :2] = a, b
    verts[:, 2] = z
    faces = []
    for i in range(n - 1):
        p, q, r, s = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
        faces += [(p, q, r), (r, q, s)]
    mesh = spec.add_mesh()
    mesh.name = f"mesh_{name}"
    mesh.uservert = verts.ravel().tolist()
    mesh.userface = np.asarray(faces).ravel().tolist()
    mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL     # a flat strip has no volume
    geom = spec.worldbody.add_geom()
    geom.name = name
    geom.type = mujoco.mjtGeom.mjGEOM_MESH
    geom.meshname = mesh.name
    geom.material = material
    geom.contype = geom.conaffinity = 0
    geom.group = group
    geom.mass = 0.0
    return geom


def frustum(spec, name, r0, r1, height, n=48):
    """A mesh of a truncated cone along z, radius r0 at z = 0 and r1 at
    z = height (r1 = 0: a cone), closed at both ends."""
    a = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    ring0 = np.stack([r0 * np.cos(a), r0 * np.sin(a), np.zeros(n)], axis=1)
    ring1 = np.stack([r1 * np.cos(a), r1 * np.sin(a), np.full(n, height)], axis=1)
    verts = np.concatenate([ring0, ring1, [[0.0, 0.0, 0.0], [0.0, 0.0, height]]])
    faces = []
    for k in range(n):
        j = (k + 1) % n
        faces += [(k, j, n + j), (k, n + j, n + k), (2 * n, j, k), (2 * n + 1, n + k, n + j)]
    mesh = spec.add_mesh()
    mesh.name = name
    mesh.uservert = verts.ravel().tolist()
    mesh.userface = np.asarray(faces).ravel().tolist()
    return mesh.name


def cone(spec, name, pos, material_body, material_band, material_base, scale=1.0):
    """A traffic cone of 0.45 m (times scale) at pos on the floor: a square
    base, an orange body and a white band; visual only."""
    s = scale
    x, y = pos[0], pos[1]
    body = spec.worldbody.add_body()
    body.name = name
    body.pos = [x, y, 0.0]
    visual(body, f"{name}_base", mujoco.mjtGeom.mjGEOM_BOX, (0.17 * s, 0.17 * s, 0.0125 * s),
           pos=(0, 0, 0.0125 * s), material=material_base)
    m1 = frustum(spec, f"mesh_{name}_body", 0.125 * s, 0.022 * s, 0.43 * s)
    g = visual(body, f"{name}_body", mujoco.mjtGeom.mjGEOM_MESH, (1, 1, 1),
               pos=(0, 0, 0.025 * s), material=material_body)
    g.meshname = m1
    m2 = frustum(spec, f"mesh_{name}_band", 0.078 * s + 0.001, 0.058 * s + 0.001, 0.09 * s)
    g = visual(body, f"{name}_band", mujoco.mjtGeom.mjGEOM_MESH, (1, 1, 1),
               pos=(0, 0, 0.025 * s + 0.20 * s), material=material_band)
    g.meshname = m2
    return body


def pallet_stack(spec, name, center, yaw, wood, cargo, layers=3, cargo_rgba=None, wrap=None,
                 size=(1.2, 0.8), height=None):
    """A stack of loaded pallets on the floor at center (x, y), each pallet
    size[0] by size[1] m (along and across yaw): each a deck, three runners
    and a box of cargo 0.04 m inside the deck's edges; the cargo 0.44 m high,
    or such that the stack is height m high; visual only. The stack fills
    the box of the pallets' size and its height."""
    body = spec.worldbody.add_body()
    body.name = name
    body.pos = [center[0], center[1], 0.0]
    body.quat = quat_z(yaw).tolist()
    hx, hy = size[0] / 2, size[1] / 2
    h = 0.22 if height is None else (height / layers - 0.124) / 2
    z = 0.0
    for k in range(layers):
        for j, y in enumerate((-(hy - 0.05), 0.0, hy - 0.05)):
            visual(body, f"{name}_runner{k}{j}", mujoco.mjtGeom.mjGEOM_BOX, (hx, 0.05, 0.05),
                   pos=(0.0, y, z + 0.05), material=wood)
        visual(body, f"{name}_deck{k}", mujoco.mjtGeom.mjGEOM_BOX, (hx, hy, 0.012),
               pos=(0.0, 0.0, z + 0.112), material=wood)
        visual(body, f"{name}_cargo{k}", mujoco.mjtGeom.mjGEOM_BOX, (hx - 0.04, hy - 0.04, h),
               pos=(0.0, 0.0, z + 0.124 + h), material=cargo if wrap is None else wrap)
        z += 0.124 + 2 * h
    return body


def container(spec, name, center, yaw, material, frame, length=6.06, width=2.44, height=2.59):
    """A shipping container of the given size on the floor: a body of
    corrugated sides and frame corners; visual only."""
    body = spec.worldbody.add_body()
    body.name = name
    body.pos = [center[0], center[1], 0.0]
    body.quat = quat_z(yaw).tolist()
    visual(body, f"{name}_box", mujoco.mjtGeom.mjGEOM_BOX,
           (length / 2 - 0.03, width / 2 - 0.03, height / 2 - 0.03), pos=(0, 0, height / 2),
           material=material)
    for sx in (-1, 1):
        for sy in (-1, 1):
            visual(body, f"{name}_post{sx}{sy}", mujoco.mjtGeom.mjGEOM_BOX, (0.08, 0.08, height / 2),
                   pos=(sx * (length / 2 - 0.08), sy * (width / 2 - 0.08), height / 2),
                   material=frame)
    for sz in (0.08, height - 0.08):
        for sy in (-1, 1):
            visual(body, f"{name}_rail{sz:.1f}{sy}", mujoco.mjtGeom.mjGEOM_BOX,
                   (length / 2, 0.08, 0.08), pos=(0, sy * (width / 2 - 0.08), sz),
                   material=frame)
    return body


def visual(body, name, kind, size, pos=(0, 0, 0), quat=None, material="", rgba=None,
           group=1, fromto=None):
    """A visual geom on body: no mass, no contact."""
    g = body.add_geom()
    g.name = name
    g.type = kind
    g.size = list(size) + [0.0] * (3 - len(size))
    if fromto is not None:
        g.fromto = list(fromto)
    else:
        g.pos = list(pos)
    if quat is not None:
        g.quat = list(quat)
    if material:
        g.material = material
    if rgba is not None:
        g.rgba = list(rgba)
    g.contype = g.conaffinity = 0
    g.group = group
    g.mass = 0.0
    g.density = 0.0
    return g


class Recorder:
    """Records, while installed, the state of every MjData that MuJoCo's
    mj_step advances and that select accepts, at the frame times of a video
    (every 1/fps of simulated time): time, qpos, qvel, and whatever extra(m,
    d) returns (a dict of arrays, such as the forces applied to bodies). It
    wraps mj_step and changes nothing in it: the step runs first, with the
    same arguments, and the record only copies from the data afterwards."""

    def __init__(self, fps, select=lambda m, d: True, extra=None):
        self.dt = 1.0 / fps
        self.select = select
        self.extra = extra
        self.runs = {}
        self._step = None

    def __enter__(self):
        self._step = mujoco.mj_step
        step, runs, dt, select, extra = self._step, self.runs, self.dt, self.select, self.extra

        def recorded(m, d, *args, **kwargs):
            step(m, d, *args, **kwargs)
            if select(m, d):
                r = runs.setdefault(id(d), {"next": 0.0, "t": [], "qpos": [], "qvel": [],
                                            "extra": [], "steps": 0, "data": d})
                r["steps"] += 1
                if d.time >= r["next"] - 1e-9:
                    r["t"].append(float(d.time))
                    r["qpos"].append(d.qpos.copy())
                    r["qvel"].append(d.qvel.copy())
                    if extra is not None:
                        r["extra"].append({k: np.array(v, copy=True)
                                           for k, v in extra(m, d).items()})
                    r["next"] = (round(d.time / dt) + 1) * dt

        mujoco.mj_step = recorded
        return self

    def __exit__(self, *exc):
        mujoco.mj_step = self._step
        return False

    def of(self, data):
        r = self.runs[id(data)]
        return np.asarray(r["t"]), np.asarray(r["qpos"]), np.asarray(r["qvel"])

    def longest(self):
        """The run with the most steps: (t, qpos, qvel, extra, steps)."""
        r = max(self.runs.values(), key=lambda r: r["steps"])
        return (np.asarray(r["t"]), np.asarray(r["qpos"]), np.asarray(r["qvel"]), r["extra"],
                r["steps"])
