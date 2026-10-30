"""
Runs build.py on a bundle with stand-ins for Blender's modules (bpy,
mathutils), outside Blender, to test its own logic: reading the bundle, the
shapes (every triangle's vertices exist, every corner normal has unit
length), the UV layers, the materials' keys, the tracks shown and hidden
frame by frame, and the frame loop and its file names. Blender's API itself
is not tested: the stand-ins accept every call.

Physical AI, Together, Chapters 13 to 15 (the videos of their runs, path
traced).

    python standin_test.py BUNDLE [--frames SPEC] [--sky 5|4]

--sky 5 offers the sky types of Blender 5.x, 4 those of 4.x (the default).
Writes empty stand-in frames into a temporary folder and removes it.
"""
import argparse
import os
import shutil
import sys
import tempfile
import types
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROBLEMS = []
STATS = dict(meshes=0, objects=0, materials=0, renders=0, loops=0, uv_layers=0, normal_sets=0)

# --- mathutils ---------------------------------------------------------------------


class Matrix:
    def __init__(self):
        self.a = np.eye(4)

    @staticmethod
    def Identity(n):
        return Matrix()

    def __getitem__(self, i):
        return self.a[i]

    def to_4x4(self):
        return self


class _Quat:
    def __init__(self, R):
        self.R = R

    def to_matrix(self):
        M = Matrix()
        M.a[:3, :3] = self.R
        return M


class Vector:
    def __init__(self, v):
        self.v = np.array(v, float)

    def to_track_quat(self, track, up):
        z = self.v / np.linalg.norm(self.v)
        y0 = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.99 else np.array([1.0, 0.0, 0.0])
        x = np.cross(y0, z)
        x /= np.linalg.norm(x)
        return _Quat(np.column_stack([x, np.cross(z, x), z]))


# --- bpy ---------------------------------------------------------------------------


class _Collection(list):
    def foreach_set(self, attr, seq):
        if len(seq) != len(self):
            PROBLEMS.append(f"foreach_set {attr}: {len(seq)} values for {len(self)} items")


class _Mesh:
    def __init__(self, name):
        self.name = name
        self.polygons, self.loops, self.materials = _Collection(), _Collection(), []
        self.uv_layers = types.SimpleNamespace(new=self._uv)
        STATS["meshes"] += 1

    def _uv(self, name):
        def put(attr, seq):
            if len(seq) != 2 * len(self.loops):
                PROBLEMS.append(f"{self.name}: {len(seq)} UV values for {len(self.loops)} corners")
            STATS["uv_layers"] += 1
        return types.SimpleNamespace(data=types.SimpleNamespace(foreach_set=put))

    def from_pydata(self, verts, edges, faces):
        nv = len(verts)
        for f in faces:
            if len(f) != 3 or not all(0 <= int(i) < nv for i in f):
                PROBLEMS.append(f"{self.name}: a face {f} with {nv} vertices")
                break
        self.polygons = _Collection([None] * len(faces))
        self.loops = _Collection([None] * (3 * len(faces)))
        STATS["loops"] += len(self.loops)

    def normals_split_custom_set(self, normals):
        n = np.array(normals, float).reshape(-1, 3)
        if len(n) != len(self.loops):
            PROBLEMS.append(f"{self.name}: {len(n)} normals for {len(self.loops)} corners")
        length = np.linalg.norm(n, axis=1)
        bad = ~np.isfinite(length) | (np.abs(length - 1.0) > 1e-6)
        if bad.any():
            PROBLEMS.append(f"{self.name}: {int(bad.sum())} of {len(n)} corner normals not of "
                            "unit length")
        STATS["normal_sets"] += 1

    def update(self):
        pass

    def clear_geometry(self):
        self.polygons, self.loops = _Collection(), _Collection()


class _Object:
    def __init__(self, name, data):
        self.name, self.data = name, data
        slots = len(data.materials) if isinstance(data, _Mesh) else 0
        self.material_slots = [types.SimpleNamespace(link="DATA", material=None) for _ in range(slots)]
        self.hide_render, self.matrix_world = False, None
        STATS["objects"] += 1


class _Pixels:
    def __len__(self):
        return 4 * 16 * 9

    def foreach_get(self, px):
        px[:] = 0.2


def _nodes_new(kind):
    node = mock.MagicMock()
    if kind == "ShaderNodeTexSky":
        node.bl_rna.properties.__getitem__.return_value.enum_items.keys.return_value = SKY
    return node


def _with_nodes(name):
    thing = mock.MagicMock()
    thing.node_tree.nodes.new.side_effect = _nodes_new
    return thing


def _material(name):
    STATS["materials"] += 1
    return _with_nodes(name)


def _render(write_still=True):
    path = bpy.context.scene.render.filepath
    if not (isinstance(path, str) and path.endswith(".png")):
        PROBLEMS.append(f"a render to {path!r}")
        return
    open(path, "wb").close()
    STATS["renders"] += 1


def _image(path, check_existing=False):
    if not os.path.exists(path):
        PROBLEMS.append(f"an image that does not exist: {path}")
    img = mock.MagicMock()
    img.pixels = _Pixels()
    return img


SKY = []
bpy = mock.MagicMock()


def install(sky):
    SKY[:] = sky
    mu = types.ModuleType("mathutils")
    mu.Matrix, mu.Vector = Matrix, Vector
    bpy.app.version_string = "stand-in"
    bpy.data.meshes.new.side_effect = _Mesh
    bpy.data.objects.new.side_effect = _Object
    bpy.data.materials.new.side_effect = _material
    bpy.data.worlds.new.side_effect = _with_nodes
    bpy.data.images.load.side_effect = _image
    bpy.ops.render.render.side_effect = _render
    sys.modules["mathutils"], sys.modules["bpy"] = mu, bpy


def main():
    ap = argparse.ArgumentParser(description="build.py's logic, with stand-ins for Blender")
    ap.add_argument("bundle")
    ap.add_argument("--frames", default="stills")
    ap.add_argument("--sky", default="4", choices=("4", "5"))
    args = ap.parse_args()
    install(["PREETHAM", "HOSEK_WILKIE", "NISHITA"] if args.sky == "4" else
            ["SINGLE_SCATTERING", "MULTIPLE_SCATTERING", "PREETHAM", "HOSEK_WILKIE"])
    sys.path.insert(0, HERE)
    import build
    out = tempfile.mkdtemp(prefix="standin-")
    try:
        sys.argv = ["blender", "--", "--bundle", args.bundle, "--out", out, "--frames",
                    args.frames, "--overwrite"]
        build.main()
        bundle = build.Bundle(args.bundle)
        frames = bundle.frames(args.frames)
        names = {int(k): v for k, v in bundle.meta["stills"].items()}
        for i in frames:
            name = names[i] if args.frames == "stills" else f"frame_{i:05d}"
            if not os.path.exists(os.path.join(out, f"{name}.png")):
                PROBLEMS.append(f"frame {i} was not written")
        if not os.path.exists(os.path.join(out, "render.json")):
            PROBLEMS.append("no render.json")
    finally:
        shutil.rmtree(out, ignore_errors=True)
    print(", ".join(f"{k} {v}" for k, v in STATS.items()))
    for p in PROBLEMS:
        print("PROBLEM:", p)
    print("no problems" if not PROBLEMS else f"{len(PROBLEMS)} problems")
    raise SystemExit(1 if PROBLEMS else 0)


if __name__ == "__main__":
    main()
