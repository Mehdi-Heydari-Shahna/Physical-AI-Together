#!/usr/bin/env python3
"""
Two stills of the cart of Section 11.1 as MuJoCo draws its model.

Physical AI, Together, Chapter 11 (Section 11.1, the figure of the stills).

The model is the MJCF text that cart_mujoco/scene.py writes from its design,
checked against results/cart.xml by its SHA-256. The cart is placed at two
states of the exact run of Chapter 3 (results/agreement-chapter3.csv) with
the adapter's own placement, Sim.place: both wheels on the ground, sunk by
the penetration at rest, which Sim computes when it is built (30 periods at
rest), rolling without slip, the tool still. Nothing of a run is drawn, and
the drawing simulates nothing: it calls mj_forward only.

The visual model is a copy of the simulated one to which ../render adds a
gradient sky, a sun and a fill light; the geoms get the book's colours, the
chassis is drawn translucent to show the wheels inside it, and MuJoCo's ray
of the rangefinder is drawn in the book's blue. Two annotations are added to
the scene: the tool's slide, for which the model has no geom, as a thin
line from the bumper to the tool; and the ground marked as a decoration, so
that it receives shadows but casts none (its long, thin facets on the crest
would shadow one another).

Usage, from this folder (MuJoCo 3.16.0, NumPy, Pillow, EGL):
    MUJOCO_GL=egl python3 mujoco_stills.py          # writes results/video/
    MUJOCO_GL=egl python3 mujoco_stills.py OUT_DIR
"""
import hashlib
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))              # the renderer, ../render
sys.path.insert(0, HERE)                                  # cart_mujoco

import mujoco                                             # noqa: E402
import numpy as np                                        # noqa: E402

from render import film, stage                           # noqa: E402
from cart_mujoco.scene import Design                      # noqa: E402
from cart_mujoco.sim import Sim                           # noqa: E402

SIZE = (1600, 900)


def hexrgba(h, a=1.0):
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)] + [a]


COLOURS = {"ground": hexrgba("A7ADA9"),               # a mid grey
           "wall": hexrgba("A33B3B"),                 # bkred
           "chassis": hexrgba("8A949B"),              # a light bkgrey
           "front": hexrgba("2F3A42"),                # near bkink
           "rear": hexrgba("2F3A42"),
           "tool": hexrgba("B0672A")}                 # bkamber
CHASSIS_ALPHA = 0.55

# name: ((p, v, l) of the exact run, (lookat [m], distance [m], azimuth, elevation [deg]))
SHOTS = {
    # periods 202 to 219: at rest at 23.7 m, the tool unfolded (the state of
    # the rangefinder's diagnostic)
    "operation": ((23.7, 0.0, 0.8), ((24.0, 0.0, -0.47), 3.3, 70.0, -15.0)),
    # the start of period 60: on the crest at 1.9 m/s, braking, the tool folded
    "crest": ((20.795, 1.9, 0.0), ((20.55, 0.0, 0.03), 1.75, 108.0, -8.0)),
}


def visual_model(xml):
    """The simulated model with a stage and colours; no body, joint or
    collision is added or changed."""
    spec = mujoco.MjSpec.from_string(xml)
    stage.dress(spec, sun=(0.35, 0.50, -0.79), sun_power=0.72, fill_power=0.30,
                ambient=0.34, haze=0.20, shadowsize=4096, shadowclip=0.45, extent=8.0,
                zenith=(0.40, 0.56, 0.78), horizon=(0.80, 0.84, 0.88))
    spec.visual.rgba.rangefinder = hexrgba("2D5A88")    # bkblue
    for g in spec.geoms:
        if g.name in COLOURS:
            g.rgba = COLOURS[g.name]
    model = spec.compile()
    model.geom_rgba[model.geom("chassis").id][3] = CHASSIS_ALPHA
    return model


def annotations(model, data):
    """The overlay of Film.shoot: the ground casts no shadow, and the slide
    is drawn from the bumper to the rear face of the tool's box."""
    bumper = data.site_xpos[model.site("bumper").id].copy()
    tip = data.site_xpos[model.site("tip").id].copy()
    axis = data.xmat[model.body("chassis").id].reshape(3, 3)[:, 0]
    end = tip - 0.1 * axis
    ground = model.geom("ground").id

    def overlay(scene):
        for g in scene.geoms[:scene.ngeom]:
            if g.objtype == int(mujoco.mjtObj.mjOBJ_GEOM) and g.objid == ground:
                g.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
        if np.linalg.norm(end - bumper) > 1e-3:
            film.line(scene, bumper, end, 0.006, [0.36, 0.40, 0.43, 1.0])
    return overlay


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "video")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    sim = Sim(Design())
    with open(os.path.join(HERE, "results", "cart.xml"), "rb") as fh:
        recorded = hashlib.sha256(fh.read()).hexdigest()
    built = hashlib.sha256(sim.xml.encode()).hexdigest()
    if built != recorded:
        raise SystemExit(f"the model text differs from results/cart.xml: {built[:16]}")
    print(f"model text as recorded (SHA-256 {built[:16]}); penetration at rest "
          f"{1000 * sim.sink:.3g} mm")
    model = visual_model(sim.xml)
    for name, (state, (lookat, dist, az, el)) in SHOTS.items():
        sim.place(*state)
        data = mujoco.MjData(model)
        data.qpos[:], data.qvel[:] = sim.data.qpos, sim.data.qvel
        mujoco.mj_forward(model, data)
        stage.follow(model, lookat, distance=30.0)
        img = film.Film(model, *SIZE, supersample=2).shoot(
            data, np.array(lookat), dist, az, el, overlay=annotations(model, data))
        img = film.grade(img, contrast=1.04, saturation=1.02, vignette=0.10)
        path = os.path.join(out, f"cart-mujoco-{name}.jpg")
        img.save(path, quality=92, subsampling=0, optimize=True)
        print(f"{path}: (p, v, l) = {state}; rangefinder "
              f"{sim.data.sensor('range').data[0]:.6f} m")
    print(f"{time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
