"""
A reader of COLLADA (.dae) meshes, enough for the robots' visual meshes,
which MuJoCo does not load: triangles and polygons, node transforms and
the diffuse colours of their materials.

Physical AI, Together, Chapters 13 to 15 (the videos of their governed runs).

load(path) returns the parts of a file as (vertices [n x 3, m], faces
[k x 3], rgba), in the file's own frame converted to metres and to z up, as
ROS's viewers read it. add(spec, ...) puts them into an MjSpec as visual
meshes, one per colour.
"""
import xml.etree.ElementTree as ET

import numpy as np

NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}


def _floats(text):
    return np.array(text.split(), dtype=float) if text and text.strip() else np.zeros(0)


def _node_matrix(node):
    T = np.eye(4)
    for child in node:
        tag = child.tag.split("}")[-1]
        v = _floats(child.text)
        if tag == "matrix":
            T = T @ v.reshape(4, 4)
        elif tag == "translate":
            M = np.eye(4)
            M[:3, 3] = v
            T = T @ M
        elif tag == "rotate":
            axis, ang = v[:3] / np.linalg.norm(v[:3]), np.radians(v[3])
            K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            M = np.eye(4)
            M[:3, :3] = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
            T = T @ M
        elif tag == "scale":
            T = T @ np.diag(list(v) + [1.0])
    return T


def load(path):
    root = ET.parse(path).getroot()
    unit = root.find("c:asset/c:unit", NS)
    scale = float(unit.get("meter", 1.0)) if unit is not None else 1.0
    up = root.find("c:asset/c:up_axis", NS)
    up = up.text.strip() if up is not None and up.text else "Y_UP"

    effects = {}
    for eff in root.iterfind(".//c:library_effects/c:effect", NS):
        col = eff.find(".//c:diffuse/c:color", NS)
        rgba = list(_floats(col.text)) if col is not None else [0.6, 0.6, 0.6, 1.0]
        if len(rgba) == 3:
            rgba.append(1.0)
        tr = eff.find(".//c:transparency/c:float", NS)
        if tr is not None and eff.find(".//c:transparent", NS) is not None:
            a = float(tr.text)
            rgba[3] = rgba[3] * (a if a <= 1.0 else 1.0)
        effects[eff.get("id")] = rgba
    materials = {}
    for mat in root.iterfind(".//c:library_materials/c:material", NS):
        ie = mat.find("c:instance_effect", NS)
        materials[mat.get("id")] = effects.get(ie.get("url")[1:], [0.6, 0.6, 0.6, 1.0])

    geometries = {}
    for geo in root.iterfind(".//c:library_geometries/c:geometry", NS):
        mesh = geo.find("c:mesh", NS)
        if mesh is None:
            continue
        sources = {}
        for src in mesh.iterfind("c:source", NS):
            fa = src.find("c:float_array", NS)
            acc = src.find("c:technique_common/c:accessor", NS)
            if fa is None or acc is None:
                continue
            stride = int(acc.get("stride", "1"))
            sources[src.get("id")] = _floats(fa.text).reshape(-1, stride)
        vert = mesh.find("c:vertices", NS)
        pos_id = next(i.get("source")[1:] for i in vert.iterfind("c:input", NS)
                      if i.get("semantic") == "POSITION")
        positions = sources[pos_id][:, :3]
        prims = []
        for tag in ("triangles", "polylist", "polygons"):
            for prim in mesh.iterfind(f"c:{tag}", NS):
                inputs = prim.findall("c:input", NS)
                offsets = {i.get("semantic"): int(i.get("offset", "0")) for i in inputs}
                stride = max(offsets.values()) + 1
                if tag == "polygons":
                    polys = [(_floats(p.text).astype(int).reshape(-1, stride))
                             for p in prim.iterfind("c:p", NS)]
                    faces = []
                    for poly in polys:
                        v = poly[:, offsets["VERTEX"]]
                        faces += [(v[0], v[j], v[j + 1]) for j in range(1, len(v) - 1)]
                    faces = np.array(faces, dtype=int).reshape(-1, 3)
                else:
                    p = prim.find("c:p", NS)
                    if p is None:
                        continue
                    idx = _floats(p.text).astype(int).reshape(-1, stride)[:, offsets["VERTEX"]]
                    if tag == "triangles":
                        faces = idx.reshape(-1, 3)
                    else:
                        counts = _floats(prim.find("c:vcount", NS).text).astype(int)
                        faces, k = [], 0
                        for c in counts:
                            poly = idx[k:k + c]
                            k += c
                            faces += [(poly[0], poly[j], poly[j + 1]) for j in range(1, c - 1)]
                        faces = np.array(faces, dtype=int).reshape(-1, 3)
                prims.append((prim.get("material"), faces))
        geometries[geo.get("id")] = (positions, prims)

    library_nodes = {n.get("id"): n for n in root.iterfind(".//c:library_nodes//c:node", NS)}
    parts = []

    def walk(node, T):
        T = T @ _node_matrix(node)
        for ig in node.iterfind("c:instance_geometry", NS):
            gid = ig.get("url")[1:]
            if gid not in geometries:
                continue
            binds = {im.get("symbol"): im.get("target")[1:]
                     for im in ig.iterfind(".//c:instance_material", NS)}
            positions, prims = geometries[gid]
            P = positions @ T[:3, :3].T + T[:3, 3]
            for symbol, faces in prims:
                rgba = materials.get(binds.get(symbol, symbol), [0.6, 0.6, 0.6, 1.0])
                parts.append((P, faces, rgba))
        for child in node.iterfind("c:node", NS):
            walk(child, T)
        for inode in node.iterfind("c:instance_node", NS):
            target = library_nodes.get(inode.get("url")[1:])
            if target is not None:
                walk(target, T)

    scene = root.find("c:scene/c:instance_visual_scene", NS)
    vs = None
    for cand in root.iterfind(".//c:library_visual_scenes/c:visual_scene", NS):
        if scene is None or cand.get("id") == scene.get("url")[1:]:
            vs = cand
            break
    if vs is not None:
        for node in vs.iterfind("c:node", NS):
            walk(node, np.eye(4))

    if up == "Y_UP":
        R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    elif up == "X_UP":
        R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    else:
        R = np.eye(3)
    return [((P @ R.T) * scale, F, rgba) for P, F, rgba in parts if len(F)]


def merged(parts, digits=3):
    """The parts merged by colour: {rgba: (vertices, faces)}."""
    groups = {}
    for P, F, rgba in parts:
        key = tuple(round(float(c), digits) for c in rgba)
        groups.setdefault(key, []).append((P, F))
    out = {}
    for key, items in groups.items():
        verts, faces, n = [], [], 0
        for P, F in items:
            used = np.unique(F)
            remap = -np.ones(len(P), dtype=int)
            remap[used] = np.arange(len(used)) + n
            verts.append(P[used])
            faces.append(remap[F])
            n += len(used)
        out[key] = (np.concatenate(verts), np.concatenate(faces))
    return out
