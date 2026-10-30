"""
Procedural textures for the videos: grounds, wood and painted marks, each
generated from a fixed seed so that every run of a script draws the same
images, and each periodic, so that it tiles without seams.

Physical AI, Together, Chapters 13 to 15 (the videos of their governed runs).

A texture is an array of shape (n, n, 3) of uint8. add() puts one into an
MjSpec with a material that repeats it every so many metres.
"""
import numpy as np
from scipy.spatial import cKDTree


def noise(n, beta, seed, aniso=1.0):
    """Periodic fractal noise of spectral slope beta, zero mean, unit
    standard deviation; aniso > 1 stretches it along the image's rows."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((n, n))
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(n)[None, :] / aniso
    f = np.sqrt(fx * fx + fy * fy)
    f[0, 0] = 1.0
    spec = np.fft.fft2(white) / f ** beta
    spec[0, 0] = 0.0
    out = np.real(np.fft.ifft2(spec))
    return (out - out.mean()) / out.std()


def _cells(n, count, seed):
    """Periodic Voronoi cells: for every pixel, the index of its nearest seed
    and the distances to the nearest two (for the cells' edges)."""
    rng = np.random.default_rng(seed)
    pts = rng.random((count, 2)) * n
    tiled = np.concatenate([pts + np.array((dx, dy)) * n
                            for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
    ids = np.tile(np.arange(count), 9)
    yy, xx = np.mgrid[0:n, 0:n]
    q = np.stack([xx.ravel() + 0.5, yy.ravel() + 0.5], axis=1)
    dist, idx = cKDTree(tiled).query(q, k=2)
    return ids[idx[:, 0]].reshape(n, n), dist[:, 0].reshape(n, n), dist[:, 1].reshape(n, n)


def _rgb(base, n, *mods):
    """The colour base over an n x n image, each (field, gain) of mods
    scaling it by 1 + gain * field."""
    out = np.asarray(base, float)[None, None, :] * np.ones((n, n, 1))
    for f, gain in mods:
        out = out * (1.0 + gain * f[..., None])
    return out


def _clip(a):
    return np.clip(np.round(a), 0, 255).astype(np.uint8)


def asphalt(n=1024, seed=1, tone=(58, 59, 61)):
    """Dark asphalt: mottled, grained, with light and dark aggregate."""
    rng = np.random.default_rng(seed)
    out = _rgb(tone, n, (noise(n, 2.2, seed), 0.07), (noise(n, 0.6, seed + 1), 0.08))
    for share, gain in ((0.010, 0.55), (0.012, -0.35)):
        dots = rng.random((n, n)) < share
        out[dots] *= 1.0 + gain * rng.random((int(dots.sum()), 1))
    return _clip(out)


def concrete(n=1024, seed=2, tone=(152, 151, 146)):
    """Light concrete: low mottling, fine grain and small dark pores."""
    rng = np.random.default_rng(seed)
    out = _rgb(tone, n, (noise(n, 2.4, seed), 0.05), (noise(n, 0.8, seed + 1), 0.035))
    pores = rng.random((n, n)) < 0.004
    out[pores] *= 0.75
    return _clip(out)


def gravel(n=1024, seed=3, stones=7000):
    """Gravel: stones of grey and tan shades, each shaded towards its edge,
    darker in the gaps."""
    rng = np.random.default_rng(seed)
    cell, d1, d2 = _cells(n, stones, seed)
    shade = rng.random(stones)
    warm = rng.random(stones)
    grey = np.array((128, 124, 118), float)
    tan = np.array((150, 128, 98), float)
    base = grey[None, :] * (1 - warm[:, None]) + tan[None, :] * warm[:, None]
    base *= (0.70 + 0.55 * shade)[:, None]
    out = base[cell]
    edge = np.clip((d2 - d1) / 3.0, 0.0, 1.0)            # 0 at the gap between two stones
    out *= (0.45 + 0.55 * edge)[..., None]
    out *= 1.0 + 0.10 * noise(n, 1.0, seed + 1)[..., None]
    return _clip(out)


def soil(n=1024, seed=13, tone=(116, 92, 68)):
    """Dug soil: brown clumps, darker moist patches and small pale
    stones."""
    rng = np.random.default_rng(seed)
    out = _rgb(tone, n, (noise(n, 2.0, seed), 0.10), (noise(n, 1.0, seed + 1), 0.10),
               (noise(n, 2.6, seed + 2), 0.06))
    moist = np.clip(noise(n, 2.4, seed + 3), 0.0, None)[..., None]
    out *= 1.0 - 0.18 * moist
    stones = rng.random((n, n)) < 0.006
    out[stones] = out[stones] * 0.4 + 0.6 * np.array((150, 140, 124))
    return _clip(out)


def grass(n=1024, seed=4, wet=True):
    """Short grass, darker and more saturated when wet: blades as noise
    stretched along one direction, over patchy colour."""
    tone = np.array((52, 82, 34) if wet else (78, 110, 48), float)
    blades = noise(n, 0.9, seed, aniso=6.0)
    patches = noise(n, 2.3, seed + 1)
    out = _rgb(tone, n, (blades, 0.22), (patches, 0.12))
    yellow = np.clip(noise(n, 2.0, seed + 2), 0, None)[..., None]
    out = out * (1 - 0.15 * yellow) + 0.15 * yellow * np.array((110, 112, 52))
    return _clip(out)


def wood(n=512, seed=5, tone=(176, 138, 92)):
    """Sawn wood for pallets: grain along the rows, knots, weathering."""
    yy, xx = np.mgrid[0:n, 0:n] / n
    warp = noise(n, 1.8, seed, aniso=8.0)
    grain = np.sin(2 * np.pi * (14 * yy + 0.35 * warp)) * 0.5 + 0.5
    out = _rgb(tone, n, (grain - 0.5, 0.18), (noise(n, 1.2, seed + 1), 0.06))
    return _clip(out)


def floor_tiles(n=1024, seed=6, tone=(168, 168, 164), tiles=4, joint=(118, 118, 116)):
    """A lab's floor of smooth tiles with darker joints, tiles x tiles in the
    texture."""
    out = _rgb(tone, n, (noise(n, 2.4, seed), 0.03), (noise(n, 0.9, seed + 1), 0.02))
    step = n // tiles
    w = max(2, n // 256)
    for k in range(tiles):
        out[k * step:k * step + w, :] = joint
        out[:, k * step:k * step + w] = joint
    return _clip(out)


def corrugated(n=512, seed=7, tone=(40, 86, 140), ribs=12, rust=0.0):
    """The corrugated steel of a shipping container's side: ribs along the
    columns, paint with grime and, optionally, rust streaks."""
    rng = np.random.default_rng(seed)
    xx = np.arange(n)[None, :] / n
    rib = 0.5 + 0.5 * np.cos(2 * np.pi * ribs * xx)
    out = _rgb(tone, n, (np.broadcast_to(rib - 0.5, (n, n)), 0.22),
               (noise(n, 2.0, seed), 0.06), (noise(n, 1.0, seed + 1, aniso=0.2), 0.05))
    if rust:
        streak = np.clip(noise(n, 1.4, seed + 2, aniso=0.08), 0, None)
        streak *= rng.random() * 0.5 + 0.5
        out = out * (1 - rust * streak[..., None]) + rust * streak[..., None] * np.array((120, 64, 30))
    return _clip(out)


def wet_asphalt(n=1024, seed=8):
    """Asphalt soaked dark, with lighter dry islands and darker puddles."""
    base = asphalt(n, seed, tone=(34, 35, 37)).astype(float)
    puddles = noise(n, 2.6, seed + 3)
    shade = 1.0 - 0.25 * np.clip(puddles, 0, None)[..., None]
    return _clip(base * shade)


def paint(n=256, seed=9, tone=(232, 232, 226), wear=0.25):
    """Road paint, worn: the colour over a grain that lets the dark ground
    show through in places."""
    worn = np.clip(noise(n, 1.6, seed), -2.0, 2.0)
    mask = (worn > 1.2 - 2.4 * wear).astype(float)[..., None]
    out = np.asarray(tone, float)[None, None, :] * (1 - 0.25 * mask) * np.ones((n, n, 1))
    out *= 1.0 + 0.05 * noise(n, 0.8, seed + 1)[..., None]
    return _clip(out)


def breadboard(n=1024, seed=10, tone=(150, 153, 157), holes=16, radius=0.11):
    """A machined aluminium plate, such as a robot's workbench: a fine
    brushed grain along the rows and a grid of threaded holes, holes x holes
    in the texture, each of radius (a share of the pitch) with a lighter
    chamfer."""
    grain = noise(n, 0.4, seed, aniso=12.0)
    out = _rgb(tone, n, (grain, 0.025), (noise(n, 2.0, seed + 1), 0.02))
    pitch = n / holes
    yy, xx = np.mgrid[0:n, 0:n] + 0.5
    dx = (xx % pitch) - pitch / 2
    dy = (yy % pitch) - pitch / 2
    r = np.sqrt(dx * dx + dy * dy) / pitch
    hole = r < radius
    ring = (r >= radius) & (r < radius * 1.45)
    out[ring] = out[ring] * 1.12
    out[hole] = np.asarray(tone, float) * 0.22
    return _clip(out)


def chevrons(n=512, stripes=6, a=(236, 186, 32), b=(24, 24, 24)):
    """Yellow and black warning stripes at 45 degrees."""
    yy, xx = np.mgrid[0:n, 0:n] / n
    band = (np.floor((xx + yy) * stripes) % 2).astype(bool)
    out = np.where(band[..., None], np.asarray(a, float), np.asarray(b, float))
    out *= 1.0 + 0.04 * noise(n, 1.2, 12)[..., None]
    return _clip(out)


def add(spec, name, image, size, reflectance=0.0, specular=0.2, shininess=0.2, rgba=None):
    """Adds the image as a 2D texture and a material that repeats it every
    size metres on the geoms that use it (texuniform), and returns the
    material's name."""
    import mujoco
    h, w = image.shape[:2]
    tex = spec.add_texture()
    tex.name = f"tex_{name}"
    tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    tex.width, tex.height, tex.nchannel = w, h, 3
    tex.data = np.ascontiguousarray(image).tobytes()
    mat = spec.add_material()
    mat.name = name
    textures = [""] * len(mat.textures)
    textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
    mat.textures = textures
    mat.texuniform = True
    mat.texrepeat = [1.0 / size, 1.0 / size]
    mat.reflectance = reflectance
    mat.specular = specular
    mat.shininess = shininess
    if rgba is not None:
        mat.rgba = rgba
    return mat.name


def plain(spec, name, rgba, specular=0.3, shininess=0.3, reflectance=0.0):
    """A material without texture: paint, rubber, metal."""
    mat = spec.add_material()
    mat.name = name
    mat.rgba = rgba
    mat.specular, mat.shininess, mat.reflectance = specular, shininess, reflectance
    return mat.name
