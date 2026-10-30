"""
The visual elements of a URDF, placed on the bodies of a MuJoCo model that
was imported from the same URDF without them.

Physical AI, Together, Chapters 13 to 15 (the videos of their governed runs).

MuJoCo's URDF import fuses links joined by fixed joints into their parent
body. visuals(path, bodies) walks the URDF's joints from each body that the
compiled model kept and returns every visual element of the links fused
into it, with its pose relative to that body: (body, mesh file, position,
quaternion (w, x, y, z), rgba). attach(spec, ...) adds them to an MjSpec as
geoms that only draw: STL and OBJ files through MuJoCo, COLLADA files
through collada.py.
"""
import math
import os
import xml.etree.ElementTree as ET

import numpy as np


def _rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), \
        math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _origin(el):
    T = np.eye(4)
    o = el.find("origin") if el is not None else None
    if o is not None:
        xyz = [float(v) for v in o.get("xyz", "0 0 0").split()]
        rpy = [float(v) for v in o.get("rpy", "0 0 0").split()]
        T[:3, :3] = _rpy_matrix(*rpy)
        T[:3, 3] = xyz
    return T


def _quat(R):
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w > 1e-6:
        return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w),
                         (R[1, 0] - R[0, 1]) / (4 * w)])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(max(0.0, 1.0 + R[i, i] - R[j, j] - R[k, k])) * 2.0
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = s / 4.0
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


def visuals(path, bodies, packages):
    """The visual elements of the URDF at path, grouped on the given bodies
    (link names that the compiled model kept): a list of (body, mesh path,
    pos, quat, rgba, scale). packages maps a ROS package's name to its
    folder, for 'package://' file names."""
    root = ET.parse(path).getroot()
    colours = {}
    for mat in root.iterfind("material"):
        c = mat.find("color")
        if c is not None:
            colours[mat.get("name")] = [float(v) for v in c.get("rgba").split()]
    links = {l.get("name"): l for l in root.iterfind("link")}
    children = {}
    for j in root.iterfind("joint"):
        children.setdefault(j.find("parent").get("link"), []).append(
            (j.find("child").get("link"), j.get("type"), _origin(j)))
    out = []

    def mesh_path(name):
        if name.startswith("package://"):
            pkg, rest = name[len("package://"):].split("/", 1)
            return os.path.join(packages[pkg], rest)
        return name

    def walk(body, link, T):
        for v in links[link].iterfind("visual"):
            geo = v.find("geometry/mesh")
            if geo is None:
                continue
            M = T @ _origin(v)
            mat = v.find("material")
            rgba = [0.6, 0.6, 0.6, 1.0]
            if mat is not None:
                c = mat.find("color")
                rgba = [float(x) for x in c.get("rgba").split()] if c is not None else \
                    colours.get(mat.get("name"), rgba)
            scale = [float(s) for s in geo.get("scale", "1 1 1").split()]
            out.append((body, mesh_path(geo.get("filename")), M[:3, 3].copy(), _quat(M[:3, :3]),
                        rgba, scale))
        for child, kind, J in children.get(link, []):
            if child in bodies:
                continue
            if kind == "fixed":
                walk(body, child, T @ J)

    for b in bodies:
        if b in links:
            walk(b, b, np.eye(4))
    return out


def attach(spec, elements, materials, collada_module, name="vis", shell=True, group=1):
    """Adds the elements to spec's bodies as visual geoms. materials(rgba,
    file) gives the material name of a part (to choose paint, rubber and
    so on by colour and file). Returns the number of geoms added."""
    import mujoco
    n = 0
    bodies = {b.name: b for b in spec.bodies}
    cache = {}
    for k, (body, path, pos, quat, rgba, scale) in enumerate(elements):
        parts = []
        ext = os.path.splitext(path)[1].lower()
        if ext == ".dae":
            if path not in cache:
                cache[path] = collada_module.merged(collada_module.load(path))
            for c, (V, F) in cache[path].items():
                parts.append(("user", (V * np.asarray(scale), F), list(c)))
        else:
            parts.append(("file", path, rgba))
        for j, (kind, src, colour) in enumerate(parts):
            mesh = spec.add_mesh()
            mesh.name = f"{name}_{k}_{j}"
            if kind == "file":
                mesh.file = src
                mesh.scale = list(scale)
            else:
                V, F = src
                mesh.uservert = np.asarray(V, float).ravel().tolist()
                mesh.userface = np.asarray(F, int).ravel().tolist()
            if shell:
                mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
            g = bodies[body].add_geom()
            g.name = f"{name}_{k}_{j}"
            g.type = mujoco.mjtGeom.mjGEOM_MESH
            g.meshname = mesh.name
            g.pos, g.quat = list(pos), list(quat)
            g.material = materials(colour, path)
            g.contype = g.conaffinity = 0
            g.group = group
            g.mass = 0.0
            g.density = 0.0
            n += 1
    return n
