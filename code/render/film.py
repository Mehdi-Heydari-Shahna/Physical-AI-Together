"""
The renderer, a camera that follows its target, captions and the video
writer.

Physical AI, Together, Chapters 13 to 15 (the videos of their governed runs).

Film draws a model's state with MuJoCo's own renderer (MjrContext) in an
offscreen framebuffer of supersample times the image's size, with
multisampling, shadows, reflections, the skybox and haze (fog only if asked
for: MuJoCo's fog also covers the skybox), and scales the image down with a
Lanczos filter. Rig moves a free camera towards a
target view along a critically damped path, so that the camera follows the
robot without jerks. Video pipes the frames to ffmpeg (H.264, yuv420p).
"""
import os
import shutil
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import egl

_CONTEXT = None
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def context():
    """The process's one OpenGL context."""
    global _CONTEXT
    if _CONTEXT is None:
        _CONTEXT = egl.Context()
    return _CONTEXT


class Film:
    def __init__(self, model, width=1280, height=720, supersample=2, samples=4, fog=False,
                 groups=(0, 1, 2)):
        self.m = model
        self.width, self.height, self.ss = width, height, supersample
        w, h = width * supersample, height * supersample
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, w)
        model.vis.global_.offheight = max(model.vis.global_.offheight, h)
        model.vis.quality.offsamples = samples
        context()
        self.con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.con)
        self.scene = mujoco.MjvScene(model, maxgeom=20000)
        for flag, on in ((mujoco.mjtRndFlag.mjRND_SHADOW, 1),
                         (mujoco.mjtRndFlag.mjRND_REFLECTION, 1),
                         (mujoco.mjtRndFlag.mjRND_SKYBOX, 1),
                         (mujoco.mjtRndFlag.mjRND_HAZE, 1),
                         (mujoco.mjtRndFlag.mjRND_FOG, 1 if fog else 0)):
            self.scene.flags[flag] = on
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 0
        for g in groups:
            self.opt.geomgroup[g] = 1
        self.opt.sitegroup[:] = 0
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.viewport = mujoco.MjrRect(0, 0, w, h)
        self.rgb = np.zeros((h, w, 3), np.uint8)

    def shoot(self, data, lookat, distance, azimuth, elevation, overlay=None):
        """The image [PIL.Image] of the state in data from the camera;
        overlay(scene), if given, adds geoms to the scene before it is drawn
        (arrows for forces, marks on the floor)."""
        self.cam.lookat[:] = lookat
        self.cam.distance, self.cam.azimuth, self.cam.elevation = distance, azimuth, elevation
        mujoco.mjv_updateScene(self.m, data, self.opt, None, self.cam,
                               mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        if overlay is not None:
            overlay(self.scene)
        mujoco.mjr_render(self.viewport, self.scene, self.con)
        mujoco.mjr_readPixels(self.rgb, None, self.viewport, self.con)
        img = Image.fromarray(np.flipud(self.rgb))
        if self.ss != 1:
            img = img.resize((self.width, self.height), Image.LANCZOS)
        return img


def _mark(g, decor):
    """Sets what mjv_initGeom leaves as the scene's memory had it (MuJoCo
    3.16): the object it belongs to (none), its category (decor, which casts
    no shadow, or dynamic, which does) and its transparent flag (0, as
    mjv_updateScene leaves the model's geoms whatever their alpha; the
    renderer blends by the alpha), so that a frame and its export do not
    depend on that memory."""
    g.objtype = int(mujoco.mjtObj.mjOBJ_UNKNOWN)
    g.objid = -1
    g.category = int(mujoco.mjtCatBit.mjCAT_DECOR if decor else mujoco.mjtCatBit.mjCAT_DYNAMIC)
    g.transparent = 0


def arrow(scene, start, end, width, rgba, decor=True):
    """Adds an arrow from start to end [m] to the scene (visual only); a
    decor geom (the default, for annotations) casts no shadow."""
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                        np.zeros(9), np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, np.asarray(start, float),
                         np.asarray(end, float))
    g.emission = 0.25
    _mark(g, decor)
    scene.ngeom += 1


def line(scene, start, end, width, rgba, decor=True):
    """Adds a capsule from start to end [m] to the scene (visual only); a
    decor geom (the default, for annotations) casts no shadow, a part of a
    machine drawn this way (decor=False) does."""
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                        np.zeros(9), np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(start, float),
                         np.asarray(end, float))
    _mark(g, decor)
    scene.ngeom += 1


def dot(scene, center, radius, rgba, height=0.001):
    """Adds a flat disc at center [m] to the scene (an annotation, no
    shadow)."""
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([radius, radius, height]),
                        np.asarray(center, float), np.eye(3).ravel(),
                        np.asarray(rgba, np.float32))
    _mark(g, True)
    scene.ngeom += 1


def grade(img, contrast=1.06, saturation=1.04, vignette=0.18, warmth=0.0):
    """A mild grade of the image: contrast about mid-grey, saturation, a
    vignette darkening the corners by up to the given share, and a warm or
    cool shift; the look of a camera, not a change of what is shown."""
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = 0.5 + contrast * (a - 0.5)
    grey = a.mean(axis=2, keepdims=True)
    a = grey + saturation * (a - grey)
    if warmth:
        a[..., 0] *= 1.0 + warmth
        a[..., 2] *= 1.0 - warmth
    if vignette:
        h, w = a.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
        a *= (1.0 - vignette * np.clip(r2 - 0.25, 0.0, None) / 1.75)[..., None]
    return Image.fromarray(np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8))


class Rig:
    """A free camera's view (lookat, distance, azimuth, elevation), moved
    towards a target view along a critically damped path of time constant
    tau [s]."""

    def __init__(self, view, tau=0.8):
        self.x = np.array(view, dtype=float)
        self.v = np.zeros_like(self.x)
        self.tau = tau

    def follow(self, target, dt):
        target = np.array(target, dtype=float)
        target[4] = self.x[4] + ((target[4] - self.x[4] + 180.0) % 360.0 - 180.0)  # azimuth
        w = 1.0 / self.tau
        for _ in range(4):                            # four substeps, for stability
            h = dt / 4.0
            a = w * w * (target - self.x) - 2.0 * w * self.v
            self.v += a * h
            self.x += self.v * h
        return self.view()

    def jump(self, view):
        self.x = np.array(view, dtype=float)
        self.v[:] = 0.0

    def view(self):
        """The current view, a copy: (lookat, distance, azimuth, elevation)."""
        x = self.x
        return (x[0:3].copy(), float(x[3]), float(x[4]), float(x[5]))


def number(value, digits, width=0):
    """value printed with digits decimals in at least width characters,
    without a negative zero (for the captions)."""
    text = f"{float(value):.{digits}f}"
    if text.startswith("-") and float(text) == 0.0:
        text = text[1:]
    return text.rjust(width)


def camera_position(lookat, distance, azimuth, elevation):
    """Where MuJoCo puts a free camera that looks at lookat from distance
    at azimuth and elevation [deg]."""
    a, e = np.radians(azimuth), np.radians(elevation)
    return np.asarray(lookat, float) - distance * np.array(
        [np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def clearances(model, data, point, geoms):
    """The distance from point to each of the given geoms, boxes exactly and
    other shapes by their bounding spheres [m]."""
    out = np.empty(len(geoms))
    for k, g in enumerate(geoms):
        rel = np.asarray(point, float) - data.geom_xpos[g]
        if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_BOX):
            local = np.abs(data.geom_xmat[g].reshape(3, 3).T @ rel) - model.geom_size[g]
            out[k] = np.linalg.norm(np.maximum(local, 0.0)) + min(float(local.max()), 0.0)
        else:
            out[k] = np.linalg.norm(rel) - model.geom_rbound[g]
    return out


def smooth_path(targets, dt, tau=(0.25, 0.25, 0.25, 1.0, 1.0, 1.0)):
    """Smooths a camera's path: targets is an array of views, one row per
    frame (lookat x, y, z, distance, azimuth, elevation); each component
    follows its target along a critically damped path of time constant tau
    [s] with the target's own velocity fed forward, so that a target moving
    at constant speed is followed without lag. Azimuths are unwrapped
    first. Returns the views as Film.shoot takes them."""
    X = np.array(targets, dtype=float)
    X[:, 4] = np.degrees(np.unwrap(np.radians(X[:, 4])))
    V = np.gradient(X, dt, axis=0)
    w = 1.0 / np.asarray(tau, dtype=float)
    x, v = X[0].copy(), V[0].copy()
    out = []
    for k in range(len(X)):
        for _ in range(4):
            h = dt / 4.0
            a = w * w * (X[k] - x) + 2.0 * w * (V[k] - v)
            v += a * h
            x += v * h
        out.append((x[0:3].copy(), float(x[3]), float(x[4]), float(x[5])))
    return out


def caption(img, title, lines, corner="bottom-left", scale=1.0):
    """Draws a title and lines of text in a translucent box on the image."""
    w, h = img.size
    s = scale * h / 720.0
    big = ImageFont.truetype(FONT_BOLD, int(22 * s))
    small = ImageFont.truetype(FONT, int(18 * s))
    pad = int(12 * s)
    rows = [(title, big)] if title else []
    rows += [(line, small) for line in lines]
    sizes = [ImageDraw.Draw(img).textbbox((0, 0), t, font=f) for t, f in rows]
    bw = max(b[2] - b[0] for b in sizes) + 2 * pad
    bh = sum(b[3] - b[1] for b in sizes) + pad * (len(rows) + 1)
    x0 = pad if "left" in corner else w - bw - pad
    y0 = h - bh - pad if "bottom" in corner else pad
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.rounded_rectangle((x0, y0, x0 + bw, y0 + bh), radius=int(8 * s), fill=(12, 16, 20, 150))
    y = y0 + pad
    for (text, font), b in zip(rows, sizes):
        d.text((x0 + pad, y - b[1]), text, font=font, fill=(240, 240, 236, 255))
        y += b[3] - b[1] + pad
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def inset(img, panel, corner="bottom-right", scale=1.0):
    """Draws a panel seen from above in a translucent box: polygons (lists of
    (x, y) [m], each with an outline and an optional fill), points, and a
    line of text; panel = dict(polys=[(points, rgba_outline, rgba_fill or
    None)], dots=[((x, y), rgba)], text=str, half=the half width shown [m])."""
    w, h = img.size
    s = scale * h / 720.0
    size, pad = int(190 * s), int(12 * s)
    small = ImageFont.truetype(FONT, int(15 * s))
    text = panel.get("text", "")
    tb = ImageDraw.Draw(img).textbbox((0, 0), text or "Ag", font=small)
    inner = max(size, tb[2] - tb[0])
    bw, bh = inner + 2 * pad, size + 3 * pad + (tb[3] - tb[1])
    x0 = w - bw - pad if "right" in corner else pad
    y0 = h - bh - pad if "bottom" in corner else pad
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.rounded_rectangle((x0, y0, x0 + bw, y0 + bh), radius=int(8 * s), fill=(12, 16, 20, 150))
    half = float(panel.get("half", 0.25))
    cx, cy = x0 + pad + inner / 2, y0 + pad + size / 2

    def px(p):
        return (cx + p[0] / half * size / 2, cy - p[1] / half * size / 2)

    for points, outline, fill in panel.get("polys", []):
        pts = [px(p) for p in points]
        if fill is not None:
            d.polygon(pts, fill=tuple(int(255 * c) for c in fill))
        d.line(pts + [pts[0]], fill=tuple(int(255 * c) for c in outline), width=max(1, int(2 * s)))
    for p, rgba in panel.get("dots", []):
        x, y = px(p)
        r = 4.5 * s
        d.ellipse((x - r, y - r, x + r, y + r), fill=tuple(int(255 * c) for c in rgba))
    if text:
        d.text((x0 + pad, y0 + 2 * pad + size - tb[1]), text, font=small, fill=(240, 240, 236, 255))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


class Video:
    """Frames piped to ffmpeg as H.264 in an MP4 file."""

    def __init__(self, path, width, height, fps=30, crf=18):
        exe = shutil.which("ffmpeg")
        if exe is None:
            raise RuntimeError("ffmpeg is not installed: the videos need it on the PATH")
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.proc = subprocess.Popen(
            [exe, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
             "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", path], stdin=subprocess.PIPE)
        self.frames = 0

    def add(self, img):
        self.proc.stdin.write(np.asarray(img, dtype=np.uint8).tobytes())
        self.frames += 1

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {self.path}")
