"""
The stage of the videos: sky, sun, fill light, haze and the renderer's
quality settings, set on an MjSpec before it is compiled.

Physical AI, Together, Chapters 13 to 15 (the videos of their governed runs).

Everything here is visual: a skybox, directional lights, the headlight's
share, the haze towards the horizon and the size of the shadow map. None
of it reaches the physics, which the chapters' scripts ran on their own
models.
"""
import math

import mujoco
import numpy as np


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def dress(spec, sun=(-0.45, -0.35, -0.82), sun_rgb=(1.0, 0.96, 0.88), sun_power=0.78,
          fill_rgb=(0.62, 0.70, 0.82), fill_power=0.22, ambient=0.30, softness=0.35,
          zenith=(0.30, 0.50, 0.78), horizon=(0.86, 0.89, 0.93), haze=0.35, shadowsize=8192,
          shadowclip=1.5, extent=None, fog=None):
    """Sets the visual stage of spec: a gradient sky from zenith to horizon,
    a sun (directional, casting soft shadows) from the direction sun points
    to, a fill light from the sky opposite it, an ambient share, the haze
    of the horizon, the shadow map's size and the half width of the region
    that it covers around the sun's position (shadowclip, in units of the
    model's extent, which extent sets; follow() moves the sun with the
    camera's focus). fog = (start, end) in units of the extent, or None."""
    vis = spec.visual
    vis.headlight.active = 1
    vis.headlight.ambient = [ambient * 0.6] * 3
    vis.headlight.diffuse = [0.08, 0.08, 0.08]
    vis.headlight.specular = [0.05, 0.05, 0.05]
    vis.quality.shadowsize = shadowsize
    vis.map.haze = haze
    vis.map.shadowclip = shadowclip
    vis.map.znear = 0.005
    vis.map.zfar = 200.0
    vis.rgba.haze = list(horizon) + [1.0]
    if fog is not None:
        vis.map.fogstart, vis.map.fogend = fog
        vis.rgba.fog = list(horizon) + [1.0]
    if extent is not None:
        spec.stat.extent = extent

    sky = spec.add_texture()
    sky.name = "sky"
    sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1 = list(zenith)
    sky.rgb2 = list(horizon)
    sky.width, sky.height = 512, 3072          # the six faces of the box

    d = unit(sun)
    s = spec.worldbody.add_light()
    s.name = "sun"
    s.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    s.dir = d.tolist()
    s.pos = (-40.0 * d).tolist()
    s.diffuse = [sun_power * c for c in sun_rgb]
    s.specular = [0.35 * sun_power * c for c in sun_rgb]
    s.ambient = [ambient * 0.4 * c for c in sun_rgb]
    s.castshadow = True
    s.softness = softness

    f = spec.worldbody.add_light()
    f.name = "skyfill"
    f.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    fd = unit((-d[0], -d[1], -1.6))
    f.dir = fd.tolist()
    f.pos = (-40.0 * fd).tolist()
    f.diffuse = [fill_power * c for c in fill_rgb]
    f.specular = [0.0, 0.0, 0.0]
    f.castshadow = False
    return spec


def ground(spec, material, size=60.0, name="stage_ground", z=0.0, group=1):
    """A visual plane under everything, for the scenery beyond a model's own
    floor: no contact."""
    g = spec.worldbody.add_geom()
    g.name = name
    g.type = mujoco.mjtGeom.mjGEOM_PLANE
    g.size = [size, size, 0.1]
    g.pos = [0.0, 0.0, z]
    g.material = material
    g.contype = g.conaffinity = 0
    g.group = group
    return g


def look(target, distance, azimuth, elevation):
    """A free camera's view as Film.shoot takes it."""
    return (np.asarray(target, float), float(distance), float(azimuth), float(elevation))


def orbit(t, t0, t1, a0, a1):
    """An azimuth moving from a0 to a1 between the times t0 and t1 along a
    half-cosine, still before and after."""
    if t <= t0:
        return a0
    if t >= t1:
        return a1
    w = 0.5 * (1.0 - math.cos(math.pi * (t - t0) / (t1 - t0)))
    return a0 + (a1 - a0) * w


def follow(model, focus, distance=30.0):
    """Moves the sun's position, and so the region of its shadow map, to the
    camera's focus (the light keeps its direction)."""
    for i in range(model.nlight):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_LIGHT, i) == "sun":
            model.light_pos[i] = np.asarray(focus, float) - distance * model.light_dir[i]
