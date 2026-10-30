"""
Renders a bundle of render/export.py with Blender's Cycles path tracer, on
a GPU where there is one. Run it inside Blender, in the background:

    blender -b -P build.py -- --bundle DIR --out DIR [options]

Physical AI, Together, Chapters 13 to 15 (the videos of their runs, path
traced).

The script builds the bundle's scene in an empty Blender file: every geom
that MuJoCo's renderer would draw, as a mesh of its shape (MuJoCo's own
meshes, and boxes, spheres, capsules, cylinders, ellipsoids, planes and
arrows made here to MuJoCo's sizes), with a physically based material
derived from the geom's colour, material and texture; a physical sky whose
sun lies in the direction of the bundle's sun; and the camera of each
frame, with MuJoCo's vertical field of view. It then sets each frame's poses
and renders it, one PNG file a frame, skipping frames already rendered, so
that a render that stops can be resumed. Nothing is simulated: the poses are
the bundle's, which are the record's.

Options
  --bundle DIR      the bundle, written by a video script's --export
  --out DIR         the folder for the frames (frame_00000.png, ...) and stills
  --frames SPEC     all (default) | stills | a:b[:step] | a list 0,30,60
  --size WxH        the image's size (default: the bundle's, 1920x1080)
  --samples N       samples a pixel (default 128; the denoiser removes the rest)
  --device D        OPTIX (default), CUDA, HIP, ONEAPI, METAL or CPU
  --exposure E      stops added to the automatic exposure (default 0)
  --fixed-exposure  no automatic exposure: E stops only
  --sun-lamp        light the scene with a sun lamp, not the sky's own sun disc
  --blend FILE      also save the built scene, at the first frame, as FILE
  --overwrite       render frames whose files already exist

Written against Blender's Python API of the 4.x and 5.x series from its
documentation and source, and not yet run in Blender; its own logic (the
bundle, the shapes and their normals, the materials, the tracks and the
frame loop) runs on every bundle with stand-ins for Blender's modules
(standin_test.py). Run the smoke test of README.md first.
"""
import json
import math
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Matrix, Vector

# MuJoCo's geom types (mjtGeom)
PLANE, HFIELD, SPHERE, CAPSULE, ELLIPSOID, CYLINDER, BOX, MESH, SDF = range(9)
ARROW, ARROW1, ARROW2, LINE, LINEBOX, FLEX, SKIN, LABEL, TRIANGLE = range(100, 109)
DECOR = 4                                   # mjCAT_DECOR


def parse(argv):
    args = dict(bundle=None, out=None, frames="all", size=None, samples=128, device="OPTIX",
                exposure=0.0, fixed_exposure=False, sun_lamp=False, blend=None, overwrite=False)
    k = 0
    while k < len(argv):
        a = argv[k]
        key = a[2:].replace("-", "_")
        if key in ("fixed_exposure", "sun_lamp", "overwrite"):
            args[key] = True
            k += 1
            continue
        if key not in args:
            raise SystemExit(f"unknown option {a}")
        args[key] = argv[k + 1]
        k += 2
    if not args["bundle"] or not args["out"]:
        raise SystemExit("--bundle and --out are needed")
    args["samples"] = int(args["samples"])
    args["exposure"] = float(args["exposure"])
    return args


# --- the bundle ------------------------------------------------------------
class Bundle:
    def __init__(self, path):
        self.path = path
        with open(os.path.join(path, "scene.json"), encoding="utf-8") as fh:
            self.meta = json.load(fh)
        self.g = dict(np.load(os.path.join(path, "geoms.npz")))
        self.meshes = dict(np.load(os.path.join(path, "meshes.npz")))
        self.cam = dict(np.load(os.path.join(path, "camera.npz")))
        self.n = int(self.meta["frames"])

    def frames(self, spec):
        if spec == "all":
            return list(range(self.n))
        if spec == "stills":
            return sorted(int(k) for k in self.meta["stills"])
        if ":" in spec:
            parts = [int(x) for x in spec.split(":")]
            a, b = parts[0], parts[1]
            step = parts[2] if len(parts) > 2 else 1
            return list(range(a, min(b, self.n), step))
        return [int(x) for x in spec.split(",")]


# --- shapes -----------------------------------------------------------------
def _ring(n):
    a = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.cos(a), np.sin(a)


def shape(kind, size, mesh=None):
    """Vertices (V x 3), triangles (F x 3) and corner normals (F x 3 x 3) of a
    geom of MuJoCo's type kind with the scene's size, in its frame."""
    sx, sy, sz = (float(x) for x in size)
    if kind == MESH:
        v, f = mesh["v"].astype(float), mesh["f"].astype(np.int64)
        n = mesh.get("n")
        if n is None:
            e1, e2 = v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]
            fn = np.cross(e1, e2)
            fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
            n = np.repeat(fn[:, None, :], 3, axis=1)
        return v, f, n.astype(float), mesh.get("uv")
    if kind == BOX or kind == LINEBOX:
        v = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)])
        quads = [(0, 2, 6, 4), (1, 5, 7, 3), (0, 4, 5, 1), (2, 3, 7, 6), (0, 1, 3, 2), (4, 6, 7, 5)]
        normals = [(0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0), (-1, 0, 0), (1, 0, 0)]
        verts, faces, cn = [], [], []
        for q, nn in zip(quads, normals):
            b = len(verts)
            verts += [v[i] for i in q]
            faces += [(b, b + 1, b + 2), (b, b + 2, b + 3)]
            cn += [[nn] * 3, [nn] * 3]
        return np.array(verts), np.array(faces), np.array(cn, float), None
    if kind == PLANE:
        hx = sx if sx > 0 else 1000.0
        hy = sy if sy > 0 else 1000.0
        v = np.array([(-hx, -hy, 0), (hx, -hy, 0), (hx, hy, 0), (-hx, hy, 0)], float)
        f = np.array([(0, 1, 2), (0, 2, 3)])
        return v, f, np.tile([0.0, 0.0, 1.0], (2, 3, 1)), None
    if kind in (SPHERE, ELLIPSOID):
        rx, ry, rz = (sx, sx, sx) if kind == SPHERE else (sx, sy, sz)
        nu, nv = 48, 24
        verts, norms = [], []
        for j in range(nv + 1):
            th = math.pi * j / nv
            for i in range(nu):
                ph = 2 * math.pi * i / nu
                u = np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
                verts.append(u * (rx, ry, rz))
                nn = u / np.maximum(np.array([rx, ry, rz]), 1e-9)
                norms.append(nn / np.linalg.norm(nn))
        faces = []
        for j in range(nv):
            for i in range(nu):
                a, b = j * nu + i, j * nu + (i + 1) % nu
                c, d = a + nu, b + nu
                if j > 0:
                    faces.append((a, c, b))
                if j < nv - 1:
                    faces.append((b, c, d))
        verts, norms, faces = np.array(verts), np.array(norms), np.array(faces)
        return verts, faces, norms[faces], None
    if kind in (CYLINDER, CAPSULE):
        r, h = sx, sz                                # the scene's size: (r, r, half length)
        n = 48
        c, s = _ring(n)
        verts, faces, cn = [], [], []
        # side
        for z in (-h, h):
            verts += [(r * c[i], r * s[i], z) for i in range(n)]
        for i in range(n):
            j = (i + 1) % n
            faces += [(i, j, n + j), (i, n + j, n + i)]
            ni, nj = (c[i], s[i], 0.0), (c[j], s[j], 0.0)
            cn += [[ni, nj, nj], [ni, nj, ni]]
        if kind == CYLINDER:
            for sign in (-1, 1):
                b = len(verts)
                z = sign * h
                verts += [(0.0, 0.0, z)] + [(r * c[i], r * s[i], z) for i in range(n)]
                for i in range(n):
                    j = (i + 1) % n
                    faces.append((b, b + 1 + i, b + 1 + j) if sign > 0 else (b, b + 1 + j, b + 1 + i))
                    cn.append([(0.0, 0.0, sign)] * 3)
        else:
            m = 12
            for sign in (-1, 1):
                b = len(verts)
                rings = []
                for k in range(m + 1):
                    a = 0.5 * math.pi * k / m
                    rings.append((math.cos(a), math.sin(a)))
                for (cr, sr) in rings:
                    verts += [(r * cr * c[i], r * cr * s[i], sign * (h + r * sr)) for i in range(n)]
                for k in range(m):
                    for i in range(n):
                        j = (i + 1) % n
                        a0, b0 = b + k * n + i, b + k * n + j
                        a1, b1 = a0 + n, b0 + n
                        tri = [(a0, b0, b1), (a0, b1, a1)]
                        if sign < 0:
                            tri = [(a0, b1, b0), (a0, a1, b1)]
                        for t in tri:
                            faces.append(t)
                            cn.append([None, None, None])
            verts_a = np.array(verts, float)
            centre = lambda p: np.array([0.0, 0.0, math.copysign(h, p[2])])     # noqa: E731
            cn = [[(c_ if c_ is not None else None) for c_ in row] for row in cn]
            faces_a = np.array(faces)
            normals = np.zeros((len(faces_a), 3, 3))
            for fi, (row, f) in enumerate(zip(cn, faces_a)):
                for k in range(3):
                    if row[k] is not None:
                        normals[fi, k] = row[k]
                    else:
                        p = verts_a[f[k]]
                        q = p - centre(p)
                        length = np.linalg.norm(q)
                        normals[fi, k] = (q / length if length > 1e-12      # a zero radius:
                                          else (0.0, 0.0, math.copysign(1.0, p[2])))   # the axis
            return verts_a, faces_a, normals, None
        return np.array(verts, float), np.array(faces), np.array(cn, float), None
    if kind in (ARROW, ARROW1, ARROW2, LINE):
        r, length = max(sx, 1e-4), sz
        if kind == LINE:                             # from the geom's origin along +z
            v, f, n, _ = shape(CAPSULE, (r, r, 0.5 * length))
            return v + (0.0, 0.0, 0.5 * length), f, n, None
        head = min(0.35 * length, 6.0 * r)
        shaft = max(length - head, 1e-4)
        v1, f1, n1, _ = shape(CYLINDER, (r, r, 0.5 * shaft))
        v1 = v1 + (0.0, 0.0, 0.5 * shaft)
        n = 48
        c, s = _ring(n)
        R = 2.0 * r
        verts = [(0.0, 0.0, length)] + [(R * c[i], R * s[i], shaft) for i in range(n)] + [(0.0, 0.0, shaft)]
        faces, cn = [], []
        slope = R / head
        for i in range(n):
            j = (i + 1) % n
            faces.append((0, 1 + i, 1 + j))
            ni = np.array([c[i], s[i], slope])
            nj = np.array([c[j], s[j], slope])
            na = 0.5 * (ni + nj)
            cn.append([na / np.linalg.norm(na), ni / np.linalg.norm(ni), nj / np.linalg.norm(nj)])
            faces.append((n + 1, 1 + j, 1 + i))
            cn.append([(0.0, 0.0, -1.0)] * 3)
        v2, f2 = np.array(verts, float), np.array(faces)
        return (np.concatenate([v1, v2]), np.concatenate([f1, f2 + len(v1)]),
                np.concatenate([n1, np.array(cn, float)]), None)
    raise ValueError(f"geom type {kind} is not drawn")


def make_mesh(name, verts, faces, normals, uv=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, p)) for p in verts], [], [tuple(map(int, f)) for f in faces])
    me.update()
    if uv is not None and len(uv):
        layer = me.uv_layers.new(name="UVMap")
        layer.data.foreach_set("uv", np.asarray(uv, np.float32).reshape(-1))
    me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    if hasattr(me, "use_auto_smooth"):                         # Blender before 4.1
        me.use_auto_smooth = True
    loops = np.asarray(normals, float).reshape(-1, 3)
    if len(loops) == len(me.loops):
        me.normals_split_custom_set([tuple(x) for x in loops])
    me.update()
    return me


def set_mesh(me, verts, faces, normals):
    """Replaces a mesh's geometry (for a geom whose size changes)."""
    me.clear_geometry()
    me.from_pydata([tuple(map(float, p)) for p in verts], [], [tuple(map(int, f)) for f in faces])
    me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    loops = np.asarray(normals, float).reshape(-1, 3)
    if len(loops) == len(me.loops):
        me.normals_split_custom_set([tuple(x) for x in loops])
    me.update()


# --- materials ----------------------------------------------------------------
# Hints by the material's name in the bundle: (roughness, metallic, bump)
NAMED = (("glass", None), ("chrome", (0.12, 1.0, 0.0)), ("rim", (0.30, 1.0, 0.0)),
         ("bolt", (0.35, 1.0, 0.0)), ("alu", (0.35, 1.0, 0.2)), ("steel", (0.45, 0.6, 0.0)),
         ("metal", (0.35, 1.0, 0.0)), ("plate", (0.40, 1.0, 0.2)), ("breadboard", (0.40, 1.0, 0.2)),
         ("rubber", (0.85, 0.0, 0.0)), ("tire", (0.85, 0.0, 0.0)), ("shoe", (0.55, 0.3, 0.0)),
         ("asphalt", (0.85, 0.0, 0.6)), ("wet", (0.18, 0.0, 0.3)), ("concrete", (0.80, 0.0, 0.5)),
         ("gravel", (0.90, 0.0, 1.0)), ("soil", (0.95, 0.0, 0.9)), ("pit_floor", (0.95, 0.0, 0.9)),
         ("grass", (0.80, 0.0, 0.8)), ("wood", (0.75, 0.0, 0.5)), ("crate", (0.75, 0.0, 0.5)),
         ("tiles", (0.45, 0.0, 0.4)), ("floor", (0.45, 0.0, 0.4)), ("paint", (0.55, 0.0, 0.2)),
         ("container", (0.45, 0.4, 0.4)), ("cladding", (0.45, 0.5, 0.4)), ("roof", (0.6, 0.3, 0.0)),
         ("yellow", (0.35, 0.0, 0.0)), ("bag", (0.45, 0.0, 0.1)), ("cone", (0.5, 0.0, 0.0)))


class Materials:
    def __init__(self, bundle, images):
        self.b = bundle
        self.images = images
        self.cache = {}

    def key(self, matid, texid, rgba, emission, specular, shininess, reflectance, texrepeat,
            texuniform, category):
        return (int(matid), int(texid), tuple(np.round(np.asarray(rgba, float), 3)),
                round(float(emission), 3), round(float(specular), 3), round(float(shininess), 3),
                round(float(reflectance), 3), tuple(np.round(np.asarray(texrepeat, float), 4)),
                int(texuniform), int(category == DECOR))

    def get(self, *args):
        k = self.key(*args)
        if k not in self.cache:
            self.cache[k] = self.make(k)
        return self.cache[k]

    def make(self, k):
        matid, texid, rgba, emission, specular, shininess, reflectance, rep, uniform, decor = k
        info = self.b.meta["materials"].get(str(matid), {})
        name = info.get("name", "") or f"geom{len(self.cache)}"
        mat = bpy.data.materials.new(f"{name}.{len(self.cache)}")
        mat.use_nodes = True
        nt = mat.node_tree
        for node in list(nt.nodes):
            nt.nodes.remove(node)
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        lower = name.lower()
        hint = next((h for n, h in NAMED if n in lower), ())
        rough = min(0.95, max(0.08, math.sqrt(2.0 / (128.0 * shininess + 2.0))))
        metal, bump = 0.0, 0.0
        if hint:
            rough, metal, bump = hint
        if reflectance > 0.0:
            rough = min(rough, max(0.05, 0.35 - reflectance))
        inp = bsdf.inputs
        color = (rgba[0], rgba[1], rgba[2], 1.0)
        tex = None
        if texid >= 0 and str(texid) in self.b.meta["textures"]:
            img = self.images[str(texid)]
            tex = nt.nodes.new("ShaderNodeTexImage")
            tex.image = img
            tex.projection = "BOX"
            tex.projection_blend = 0.15
            coord = nt.nodes.new("ShaderNodeTexCoord")
            mapping = nt.nodes.new("ShaderNodeMapping")
            nt.links.new(coord.outputs["Object"], mapping.inputs["Vector"])
            mapping.inputs["Scale"].default_value = (rep[0], rep[1], rep[0])
            nt.links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
            tint = nt.nodes.new("ShaderNodeVectorMath")      # the texture times the colour
            tint.operation = "MULTIPLY"
            nt.links.new(tex.outputs["Color"], tint.inputs[0])
            tint.inputs[1].default_value = color[:3]
            nt.links.new(tint.outputs[0], inp["Base Color"])
            if bump > 0.0:
                b = nt.nodes.new("ShaderNodeBump")
                b.inputs["Strength"].default_value = 0.35 * bump
                b.inputs["Distance"].default_value = 0.01
                nt.links.new(tex.outputs["Color"], b.inputs["Height"])
                nt.links.new(b.outputs["Normal"], inp["Normal"])
        else:
            inp["Base Color"].default_value = color
        inp["Roughness"].default_value = rough
        inp["Metallic"].default_value = metal
        alpha = rgba[3]
        if "glass" in lower:
            for socket, value in (("Transmission Weight", 1.0), ("Transmission", 1.0)):
                if socket in inp:
                    inp[socket].default_value = value
                    break
            inp["Roughness"].default_value = 0.03
            inp["IOR"].default_value = 1.45
            inp["Base Color"].default_value = (min(1, rgba[0] * 1.8), min(1, rgba[1] * 1.8),
                                                min(1, rgba[2] * 1.8), 1.0)
        elif alpha < 0.999:
            inp["Alpha"].default_value = alpha
        if decor:
            for socket in ("Emission Color", "Emission"):
                if socket in inp:
                    inp[socket].default_value = color
                    break
            inp["Emission Strength"].default_value = 1.2 + 2.0 * emission
            inp["Roughness"].default_value = 0.5
        elif emission > 0.0:
            for socket in ("Emission Color", "Emission"):
                if socket in inp:
                    inp[socket].default_value = color
                    break
            inp["Emission Strength"].default_value = 2.0 * emission
        return mat


# --- the scene ------------------------------------------------------------------
def matrix(pos, mat9):
    R = np.asarray(mat9, float).reshape(3, 3)
    M = Matrix.Identity(4)
    for i in range(3):
        for j in range(3):
            M[i][j] = float(R[i, j])
        M[i][3] = float(pos[i])
    return M


def camera_matrix(pos, fwd, up):
    f = np.asarray(fwd, float)
    f /= np.linalg.norm(f)
    u = np.asarray(up, float)
    u = u - f * (u @ f)
    u /= np.linalg.norm(u)
    r = np.cross(f, u)
    M = Matrix.Identity(4)
    for i in range(3):
        M[i][0], M[i][1], M[i][2], M[i][3] = float(r[i]), float(u[i]), float(-f[i]), float(pos[i])
    return M


def setup_render(scene, args, size, fps):
    scene.render.engine = "CYCLES"
    scene.render.resolution_x, scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.fps = int(fps)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    if hasattr(scene.render, "use_persistent_data"):
        scene.render.use_persistent_data = True
    cyc = scene.cycles
    cyc.samples = args["samples"]
    if hasattr(cyc, "use_adaptive_sampling"):
        cyc.use_adaptive_sampling = True
    cyc.use_denoising = True
    device = args["device"].upper()
    if device != "CPU":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        chosen = None
        for kind in [device] + [k for k in ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL") if k != device]:
            try:
                prefs.compute_device_type = kind
            except TypeError:
                continue
            if hasattr(prefs, "refresh_devices"):
                prefs.refresh_devices()
            else:
                prefs.get_devices()
            devs = [d for d in prefs.devices if d.type == kind]
            if devs:
                for d in prefs.devices:
                    d.use = d.type == kind
                chosen = kind
                break
        if chosen is None:
            print("no GPU found for Cycles: rendering on the CPU")
            cyc.device = "CPU"
        else:
            print(f"Cycles renders on {chosen}: " + ", ".join(d.name for d in prefs.devices if d.use))
            cyc.device = "GPU"
            if chosen == "OPTIX":
                try:
                    cyc.denoiser = "OPTIX"
                except TypeError:
                    pass
            if hasattr(cyc, "denoising_use_gpu"):
                cyc.denoising_use_gpu = True
    else:
        cyc.device = "CPU"
    vs = scene.view_settings
    for transform in ("AgX", "Filmic"):
        try:
            vs.view_transform = transform
            break
        except TypeError:
            continue
    vs.look = "None"
    vs.exposure = 0.0


def setup_world(scene, bundle, args):
    sun = next((lt for lt in bundle.meta["lights"] if not lt["headlight"] and lt["castshadow"]),
               None)
    toward = -np.asarray(sun["dir"] if sun else (-0.4, -0.3, -0.85), float)
    toward /= np.linalg.norm(toward)
    world = bpy.data.worlds.new("sky")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    for node in list(nt.nodes):
        nt.nodes.remove(node)
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    sky = nt.nodes.new("ShaderNodeTexSky")
    kinds = sky.bl_rna.properties["sky_type"].enum_items.keys()
    kind = next(k for k in ("MULTIPLE_SCATTERING", "NISHITA", "SINGLE_SCATTERING", "HOSEK_WILKIE",
                            "PREETHAM") if k in kinds)
    sky.sky_type = kind
    lamp = args["sun_lamp"]
    if kind in ("MULTIPLE_SCATTERING", "NISHITA", "SINGLE_SCATTERING"):
        # Cycles puts this sky's sun at (-cos e sin r, cos e cos r, sin e)
        e = math.asin(max(-1.0, min(1.0, toward[2])))
        r = math.atan2(-toward[0], toward[1])
        sky.sun_elevation, sky.sun_rotation = e, r
        sky.sun_disc = not lamp
        strength = 0.12
    else:
        sky.sun_direction = Vector(toward.tolist())
        lamp = True
        strength = 0.6
    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = strength
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    if hasattr(world, "cycles"):
        if hasattr(world.cycles, "sampling_method"):
            world.cycles.sampling_method = "MANUAL"
            world.cycles.sample_map_resolution = 4096
    if lamp:
        data = bpy.data.lights.new("sun", "SUN")
        data.energy = 4.0
        data.angle = math.radians(0.6)
        obj = bpy.data.objects.new("sun", data)
        scene.collection.objects.link(obj)
        z = Vector(toward.tolist())
        obj.matrix_world = z.to_track_quat("Z", "Y").to_matrix().to_4x4()
    return kind


def build(args):
    t0 = time.time()
    bundle = Bundle(args["bundle"])
    meta = bundle.meta
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    size = tuple(meta["size"]) if not args["size"] else tuple(int(x) for x in args["size"].split("x"))
    setup_render(scene, args, size, meta["fps"])
    sky = setup_world(scene, bundle, args)
    images = {}
    for tid, rel in meta["textures"].items():
        img = bpy.data.images.load(os.path.join(bundle.path, rel), check_existing=True)
        img.colorspace_settings.name = "sRGB"
        images[tid] = img
    mats = Materials(bundle, images)
    g = bundle.g
    meshes = {}
    for mid in meta["meshes"]:
        part = {k[len(str(mid)) + 1:]: v for k, v in bundle.meshes.items() if k.startswith(f"{mid}_")}
        meshes[mid] = part
    shared = {}

    def data_for(kind, size_, dataid):
        if kind == MESH:
            key = ("mesh", int(dataid) // 2)
            if key not in shared:
                v, f, n, uv = shape(MESH, (1, 1, 1), meshes[int(dataid) // 2])
                shared[key] = make_mesh(f"mesh{key[1]}", v, f, n, uv)
            return shared[key]
        key = (int(kind), tuple(np.round(np.asarray(size_, float), 5)))
        if key not in shared:
            v, f, n, _ = shape(int(kind), size_)
            shared[key] = make_mesh(f"shape{len(shared)}", v, f, n)
        return shared[key]

    def material_of(kind, j, frame=None):
        rgba = g[f"{kind}_rgba"][j] if frame is None else g["track_rgba"][frame, j]
        return mats.get(g[f"{kind}_matid"][j], g[f"{kind}_texid"][j], rgba, g[f"{kind}_emission"][j],
                        g[f"{kind}_specular"][j], g[f"{kind}_shininess"][j],
                        g[f"{kind}_reflectance"][j], g[f"{kind}_texrepeat"][j],
                        g[f"{kind}_texuniform"][j], g[f"{kind}_category"][j])

    def new_object(name, data, mat, decor):
        if len(data.materials) == 0:                 # one slot on the mesh, shared ...
            data.materials.append(None)
        obj = bpy.data.objects.new(name, data)
        scene.collection.objects.link(obj)
        obj.material_slots[0].link = "OBJECT"        # ... and each object's own material in it
        obj.material_slots[0].material = mat
        if decor:
            for attr in ("visible_shadow", "visible_glossy"):
                if hasattr(obj, attr):
                    setattr(obj, attr, False)
            if hasattr(obj, "cycles_visibility"):
                obj.cycles_visibility.shadow = False
        return obj

    ns = len(g["static_type"])
    for j in range(ns):
        kind = int(g["static_type"][j])
        if kind in (HFIELD, SDF, FLEX, SKIN, LABEL, TRIANGLE):
            continue
        data = data_for(kind, g["static_size"][j], g["static_dataid"][j])
        obj = new_object(f"static{j}", data, material_of("static", j),
                         int(g["static_category"][j]) == DECOR)
        obj.matrix_world = matrix(g["static_pos"][j], g["static_mat"][j])
    tracks = []
    nt = len(g["track_type"])
    for j in range(nt):
        kind = int(g["track_type"][j])
        present = g["track_present"][:, j]
        if kind in (HFIELD, SDF, FLEX, SKIN, LABEL, TRIANGLE) or not present.any():
            continue                                 # not drawn, or in no frame
        sizes = g["track_size"][present, j]
        varies = kind != MESH and np.ptp(sizes, axis=0).max() > 1e-6
        first = int(np.argmax(present))
        if varies:
            v, f, n, _ = shape(kind, g["track_size"][first, j])
            data = make_mesh(f"track{j}", v, f, n)
        else:
            data = data_for(kind, g["track_size"][first, j], g["track_dataid"][j])
        obj = new_object(f"track{j}", data, material_of("track", j, first),
                         int(g["track_category"][j]) == DECOR)
        tracks.append((j, kind, obj, varies))
    cam_data = bpy.data.cameras.new("camera")
    cam_data.sensor_fit = "VERTICAL"
    cam_data.clip_start, cam_data.clip_end = 0.01, 2000.0
    cam = bpy.data.objects.new("camera", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    print(f"built {ns} static geoms and {len(tracks)} of {nt} tracks, {len(shared)} meshes, "
          f"{len(mats.cache)} materials, sky '{sky}', in {time.time() - t0:.1f} s")

    def apply(i):
        for j, kind, obj, varies in tracks:
            on = bool(g["track_present"][i, j])
            obj.hide_render = not on
            if not on:
                continue
            obj.matrix_world = matrix(g["track_pos"][i, j], g["track_mat"][i, j])
            if varies:
                v, f, n, _ = shape(kind, g["track_size"][i, j])
                set_mesh(obj.data, v, f, n)
            obj.material_slots[0].material = material_of("track", j, i)
        cam.matrix_world = camera_matrix(bundle.cam["pos"][i], bundle.cam["forward"][i],
                                         bundle.cam["up"][i])
        cam_data.angle_y = math.radians(float(bundle.cam["fovy"][i]))

    return bundle, scene, apply


def mean_display_value(path):
    img = bpy.data.images.load(path)
    px = np.empty(len(img.pixels), np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(-1, 4)[:, :3]
    bpy.data.images.remove(img)
    return float((px @ np.array([0.2126, 0.7152, 0.0722])).mean())


def auto_exposure(bundle, scene, apply, args, out):
    """Renders the middle frame small, measures its mean display value and
    sets the exposure that brings it to 0.40 (two rounds)."""
    keep = (scene.render.resolution_percentage, scene.cycles.samples)
    scene.render.resolution_percentage = 25
    scene.cycles.samples = 24
    apply(bundle.n // 2)
    path = os.path.join(out, "_exposure.png")
    scene.view_settings.exposure = 0.0
    for _ in range(2):
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        mean = mean_display_value(path)
        step = math.log2(0.40 / max(mean, 1e-4)) * 1.6       # the curve compresses
        scene.view_settings.exposure = max(-6.0, min(6.0, scene.view_settings.exposure + step))
        print(f"exposure: mean {mean:.3f}, set to {scene.view_settings.exposure:+.2f}")
    os.remove(path)
    scene.render.resolution_percentage, scene.cycles.samples = keep


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parse(argv)
    out = os.path.abspath(args["out"])
    os.makedirs(out, exist_ok=True)
    bundle, scene, apply = build(args)
    if args["fixed_exposure"]:
        scene.view_settings.exposure = args["exposure"]
    else:
        auto_exposure(bundle, scene, apply, args, out)
        scene.view_settings.exposure += args["exposure"]
    with open(os.path.join(out, "render.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(blender=bpy.app.version_string, samples=args["samples"],
                       device=scene.cycles.device, exposure=scene.view_settings.exposure,
                       view=scene.view_settings.view_transform), fh, indent=1)
    if args["blend"]:
        apply(0)
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args["blend"]))
    frames = bundle.frames(args["frames"])
    names = {int(k): v for k, v in bundle.meta["stills"].items()}
    t0 = time.time()
    for n, i in enumerate(frames):
        name = names[i] if args["frames"] == "stills" else f"frame_{i:05d}"
        path = os.path.join(out, f"{name}.png")
        if os.path.exists(path) and not args["overwrite"]:
            continue
        apply(i)
        scene.render.filepath = path
        t1 = time.time()
        bpy.ops.render.render(write_still=True)
        print(f"frame {i} ({n + 1}/{len(frames)}): {time.time() - t1:.1f} s, {path}", flush=True)
    print(f"done: {len(frames)} frames in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
