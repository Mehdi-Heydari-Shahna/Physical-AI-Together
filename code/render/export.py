"""
The scene and the frames of a video, written for a path tracer.

Physical AI, Together, Chapters 13 to 15 (the videos of their runs).

bundle() takes the same shot as shoot.render() (render.shoot) and, instead
of drawing its frames, writes what MuJoCo's renderer would draw in each:
every geom of MuJoCo's abstract scene (mjv_updateScene, then the shot's
overlay) with its type, size, pose, colour, material and texture, the
camera, the lights and the captions. Geoms that do not change over the
video are written once; the others, among them the overlays, as tracks with
a value for every frame. render/cycles/build.py reads the bundle in Blender
and renders it with Cycles; check() draws a frame of a bundle again with
MuJoCo's renderer, for comparison with the frame that Film draws.

A bundle is a folder:
  scene.json      the video's name, size, frame rate and frame count, its
                  stills, the materials, textures, meshes and lights, and
                  the tracks' constant fields
  geoms.npz       the static geoms, and the tracks' fields frame by frame
  meshes.npz      the meshes' vertices, triangles and corner normals (and
                  texture coordinates, where a mesh has them), in the frame
                  in which MuJoCo draws them
  camera.npz      the camera's position, forward and up vectors and its
                  vertical field of view, frame by frame
  captions.json   the captions of each frame
  textures/       the 2D textures as PNG files

The bundle holds the meshes of the models it draws; a bundle of a robot
drawn with a maker's meshes holds those meshes, under the maker's terms.
Nothing here takes part in the physics.
"""
import json
import os

import mujoco
import numpy as np

FIELDS = ("type", "dataid", "matid", "texid", "category", "texuniform", "texcoord")
FLOATS = ("emission", "specular", "shininess", "reflectance")


def _scene_geoms(scn):
    """The geoms of an mjvScene as a list of dicts, with a key for each:
    ('g', id) for a model geom, ('a', type, category, k) for the k-th other
    one of its type and category (the overlays, in the order the shot adds
    them)."""
    out = []
    count = {}
    for i in range(scn.ngeom):
        g = scn.geoms[i]
        if g.objtype == int(mujoco.mjtObj.mjOBJ_GEOM):
            key = ("g", int(g.objid))
        else:
            kind = (int(g.type), int(g.category))
            key = ("a", kind[0], kind[1], count.get(kind, 0))
            count[kind] = count.get(kind, 0) + 1
        out.append(dict(key=key, type=int(g.type), dataid=int(g.dataid), matid=int(g.matid),
                        texid=int(g.texid), category=int(g.category), texcoord=int(g.texcoord),
                        texuniform=int(g.texuniform), texrepeat=np.array(g.texrepeat, float),
                        size=np.array(g.size, float), pos=np.array(g.pos, float),
                        mat=np.array(g.mat, float).ravel(), rgba=np.array(g.rgba, float),
                        emission=float(g.emission), specular=float(g.specular),
                        shininess=float(g.shininess), reflectance=float(g.reflectance),
                        transparent=int(g.transparent)))
    return out


def _plain(x):
    """x with its arrays as lists, for JSON."""
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def _camera(scn):
    c0, c1 = scn.camera[0], scn.camera[1]
    pos = 0.5 * (np.array(c0.pos) + np.array(c1.pos))
    fwd = np.array(c0.forward, float)
    up = np.array(c0.up, float)
    fovy = 2.0 * np.degrees(np.arctan2(c0.frustum_top, c0.frustum_near))
    return pos, fwd / np.linalg.norm(fwd), up / np.linalg.norm(up), float(fovy)


def _lights(m, scn):
    out = []
    for i in range(scn.nlight):
        lt = scn.lights[i]
        out.append(dict(headlight=bool(lt.headlight), type=int(lt.type),
                        pos=list(map(float, lt.pos)), dir=list(map(float, lt.dir)),
                        diffuse=list(map(float, lt.diffuse)), ambient=list(map(float, lt.ambient)),
                        specular=list(map(float, lt.specular)), castshadow=bool(lt.castshadow),
                        softness=float(getattr(lt, "softness", 0.0)),
                        name=(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_LIGHT, int(lt.id)) or ""
                              if not lt.headlight and lt.id >= 0 else "")))
    return out


def _mesh(m, mid):
    va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
    v = np.array(m.mesh_vert[va:va + vn], np.float32)
    f = np.array(m.mesh_face[fa:fa + fn], np.int32)
    out = {"v": v, "f": f}
    na = m.mesh_normaladr[mid]
    if m.mesh_normalnum[mid] > 0:
        normals = np.array(m.mesh_normal[na:na + m.mesh_normalnum[mid]], np.float32)
        out["n"] = normals[np.array(m.mesh_facenormal[fa:fa + fn], np.int64)]
    ta = m.mesh_texcoordadr[mid]
    if ta >= 0 and m.mesh_texcoordnum[mid] > 0:
        uv = np.array(m.mesh_texcoord[ta:ta + m.mesh_texcoordnum[mid]], np.float32)
        out["uv"] = uv[np.array(m.mesh_facetexcoord[fa:fa + fn], np.int64)]
    return out


def _texture(m, tid):
    w, h, c = int(m.tex_width[tid]), int(m.tex_height[tid]), int(m.tex_nchannel[tid])
    a = int(m.tex_adr[tid])
    img = np.array(m.tex_data[a:a + w * h * c], np.uint8).reshape(h, w, c)
    return img


def _same(a, b, keys=FIELDS):
    if any(a[k] != b[k] for k in keys):
        return False
    for k in ("size", "pos", "mat", "rgba", "texrepeat"):
        if not np.allclose(a[k], b[k], rtol=0.0, atol=1e-7):
            return False
    return all(abs(a[k] - b[k]) <= 1e-7 for k in FLOATS)


def bundle(shot, n_frames, out, fps=30, stills=None, name="video", size=(1920, 1080),
           groups=(0, 1, 2), look=None):
    """Writes the frames 0 .. n_frames-1 of shot into the folder out (see the
    module's docstring); stills maps frame indices to the names of the
    book's stills, look holds hints for the renderer (exposure, sky)."""
    from PIL import Image
    m = shot.build()
    d = mujoco.MjData(m)
    scn = mujoco.MjvScene(m, maxgeom=20000)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    for g in groups:
        opt.geomgroup[g] = 1
    opt.sitegroup[:] = 0
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    def frame(i):
        shot.pose(m, d, i)
        lookat, dist, az, el = shot.view(i)
        cam.lookat[:] = lookat
        cam.distance, cam.azimuth, cam.elevation = dist, az, el
        mujoco.mjv_updateScene(m, d, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        overlay = shot.overlay(i, m, d)
        if overlay is not None:
            overlay(scn)
        return {g["key"]: g for g in _scene_geoms(scn)}

    # first pass: the geoms that are in every frame and never change are
    # static; the others are tracks
    first, changed, order, count = {}, set(), [], {}
    cams, captions, lights = [], [], None
    for i in range(n_frames):
        geoms_i = frame(i)
        for k, g in geoms_i.items():
            if k not in first:
                first[k] = g
                order.append(k)
            elif k not in changed and not _same(g, first[k]):
                changed.add(k)
            count[k] = count.get(k, 0) + 1
        cams.append(_camera(scn))
        if lights is None:
            lights = _lights(m, scn)
        cap = shot.caption(i)
        extra = getattr(shot, "caption2", None)
        panel = shot.panel(i, m, d) if hasattr(shot, "panel") else None
        captions.append({"caption": None if cap is None else [cap[0], list(cap[1])],
                         "caption2": None if extra is None or extra(i) is None
                         else [extra(i)[0], list(extra(i)[1])],
                         "panel": _plain(panel)})
    tracks = [k for k in order if k in changed or count[k] != n_frames]
    static = [first[k] for k in order if k not in changed and count[k] == n_frames]
    # second pass: the tracks' values frame by frame
    nf, nt = n_frames, len(tracks)
    index = {k: j for j, k in enumerate(tracks)}
    present = np.zeros((nf, nt), bool)
    per = {"size": np.zeros((nf, nt, 3), np.float32), "pos": np.zeros((nf, nt, 3), np.float32),
           "mat": np.tile(np.eye(3, dtype=np.float32).ravel(), (nf, nt, 1)),
           "rgba": np.zeros((nf, nt, 4), np.float32)}
    fixed = [first[k] for k in tracks]
    for i in range(n_frames):
        for k, g in frame(i).items():
            j = index.get(k)
            if j is None:
                continue
            f0 = fixed[j]
            if any(g[c] != f0[c] for c in FIELDS) or any(abs(g[c] - f0[c]) > 1e-6 for c in FLOATS):
                raise ValueError(f"track {k} changes its kind or material at frame {i}")
            present[i, j] = True
            for c in per:
                per[c][i, j] = g[c]
    os.makedirs(os.path.join(out, "textures"), exist_ok=True)
    arrays = {}
    for field in FIELDS + ("transparent",):
        arrays[f"static_{field}"] = np.array([g[field] for g in static], np.int32)
    for field, width in (("size", 3), ("pos", 3), ("mat", 9), ("rgba", 4), ("texrepeat", 2)):
        arrays[f"static_{field}"] = np.array([g[field] for g in static], np.float32).reshape(
            len(static), width)
    for field in FLOATS:
        arrays[f"static_{field}"] = np.array([g[field] for g in static], np.float32)
    arrays["track_present"] = present
    for c, v in per.items():
        arrays[f"track_{c}"] = v
    for field in FIELDS + ("transparent",):
        arrays[f"track_{field}"] = np.array([g[field] for g in fixed], np.int32)
    arrays["track_texrepeat"] = np.array([g["texrepeat"] for g in fixed], np.float32).reshape(nt, 2)
    for field in FLOATS:
        arrays[f"track_{field}"] = np.array([g[field] for g in fixed], np.float32)
    arrays["track_key"] = np.array(["-".join(map(str, k)) for k in tracks])
    np.savez_compressed(os.path.join(out, "geoms.npz"), **arrays)
    # meshes and textures that the geoms use
    used = [g for g in static] + fixed
    mesh_ids = sorted({g["dataid"] // 2 for g in used
                       if g["type"] == int(mujoco.mjtGeom.mjGEOM_MESH) and g["dataid"] >= 0})
    marr = {}
    for mid in mesh_ids:
        for part, val in _mesh(m, mid).items():
            marr[f"{mid}_{part}"] = val
    np.savez_compressed(os.path.join(out, "meshes.npz"), **marr)
    tex_ids = sorted({g["texid"] for g in used if g["texid"] >= 0})
    textures = {}
    for tid in tex_ids:
        if int(m.tex_type[tid]) != int(mujoco.mjtTexture.mjTEXTURE_2D):
            continue
        img = _texture(m, tid)
        tname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TEXTURE, tid) or f"texture{tid}"
        path = os.path.join("textures", f"{tid:03d}_{tname}.png")
        Image.fromarray(img if img.shape[2] != 1 else img[..., 0]).save(os.path.join(out, path))
        textures[str(tid)] = path
    mat_ids = sorted({g["matid"] for g in used if g["matid"] >= 0})
    materials = {str(i): dict(name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, i) or "",
                              rgba=list(map(float, m.mat_rgba[i])),
                              specular=float(m.mat_specular[i]), shininess=float(m.mat_shininess[i]),
                              reflectance=float(m.mat_reflectance[i]),
                              emission=float(m.mat_emission[i]))
                 for i in mat_ids}
    pos = np.array([c[0] for c in cams], np.float32)
    np.savez_compressed(os.path.join(out, "camera.npz"), pos=pos,
                        forward=np.array([c[1] for c in cams], np.float32),
                        up=np.array([c[2] for c in cams], np.float32),
                        fovy=np.array([c[3] for c in cams], np.float32))
    with open(os.path.join(out, "captions.json"), "w", encoding="utf-8") as fh:
        json.dump(captions, fh, ensure_ascii=False)
    meta = dict(name=name, size=list(size), fps=fps, frames=n_frames, units="metres, z up",
                stills={str(int(k)): v for k, v in (stills or {}).items()},
                static=len(static), tracks=len(tracks),
                materials=materials, textures=textures, meshes=[int(i) for i in mesh_ids],
                lights=lights, look=look or {},
                headlight=dict(ambient=list(map(float, m.vis.headlight.ambient)),
                               diffuse=list(map(float, m.vis.headlight.diffuse))),
                haze=dict(rgba=list(map(float, m.vis.rgba.haze)), haze=float(m.vis.map.haze)),
                geom_types={int(t): mujoco.mjtGeom(t).name for t in
                            {int(g["type"]) for g in used}},
                mujoco=mujoco.__version__)
    with open(os.path.join(out, "scene.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    print(f"{out}: {n_frames} frames, {len(static)} static geoms, {len(tracks)} tracks, "
          f"{len(mesh_ids)} meshes, {len(textures)} textures")
    return out


def load(path):
    """A bundle's parts: (meta, geoms, meshes, camera, captions)."""
    with open(os.path.join(path, "scene.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    with open(os.path.join(path, "captions.json"), encoding="utf-8") as fh:
        captions = json.load(fh)
    return (meta, dict(np.load(os.path.join(path, "geoms.npz"))),
            dict(np.load(os.path.join(path, "meshes.npz"))),
            dict(np.load(os.path.join(path, "camera.npz"))), captions)


def check(shot, path, i, size=(960, 540)):
    """Frame i drawn twice with MuJoCo's renderer: by Film from the shot,
    and from the bundle at path (its geoms set into an empty scene and its
    camera); returns both images and the largest difference of a pixel."""
    from . import film
    meta, geoms, _, camera, _ = load(path)
    m = shot.build()
    d = mujoco.MjData(m)
    f = film.Film(m, size[0], size[1], supersample=1)
    shot.pose(m, d, i)
    view = shot.view(i)
    a = f.shoot(d, *view, overlay=shot.overlay(i, m, d))
    scn = f.scene
    scn.ngeom = 0

    def put(kind, j, frame=None):
        g = scn.geoms[scn.ngeom]
        t = geoms[f"{kind}_type"][j]
        if frame is None:
            size_, pos, mat, rgba = (geoms[f"static_{c}"][j] for c in ("size", "pos", "mat", "rgba"))
        else:
            size_, pos, mat, rgba = (geoms[f"track_{c}"][frame, j] for c in ("size", "pos", "mat", "rgba"))
        mujoco.mjv_initGeom(g, int(t), np.asarray(size_, float), np.asarray(pos, float),
                            np.asarray(mat, float), np.asarray(rgba, np.float32))
        g.size[:] = size_                         # the scene's sizes, which initGeom would convert
        for c in FIELDS:
            setattr(g, c, int(geoms[f"{kind}_{c}"][j]))
        for c in FLOATS:
            setattr(g, c, float(geoms[f"{kind}_{c}"][j]))
        g.texrepeat[:] = geoms[f"{kind}_texrepeat"][j]
        g.transparent = int(geoms[f"{kind}_transparent"][j])
        g.camdist = float(np.linalg.norm(np.asarray(pos, float) - camera["pos"][i]))   # the order
        scn.ngeom += 1                                  # in which translucent geoms are drawn

    for j in range(len(geoms["static_type"])):
        put("static", j)
    for j in range(len(geoms["track_type"])):
        if geoms["track_present"][i, j]:
            put("track", j, i)
    for c in (0, 1):
        scn.camera[c].pos[:] = camera["pos"][i]
        scn.camera[c].forward[:] = camera["forward"][i]
        scn.camera[c].up[:] = camera["up"][i]
    mujoco.mjr_render(f.viewport, scn, f.con)
    mujoco.mjr_readPixels(f.rgb, None, f.viewport, f.con)
    from PIL import Image
    b = Image.fromarray(np.flipud(f.rgb))
    diff = int(np.abs(np.asarray(a, int) - np.asarray(b, int)).max())
    return a, b, diff
