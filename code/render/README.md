# The videos and stills of the MuJoCo runs (Chapters 11 and 13 to 15)

This package draws recorded runs of the book's MuJoCo models as videos and stills. Nothing in it takes part in the physics: each chapter's video script runs the chapter's own configuration with the chapter's modules unchanged, records the simulated state while MuJoCo steps, and then draws the record.

| Script | Run drawn | Video, in the chapter's `results/video/` | Stills in the book |
|---|---|---|---|
| `../ch11/mujoco_stills.py` | Section 11.1: the cart's model at two placed states of the run of Chapter 3 (no run is drawn) | no video; stills `cart-mujoco-operation.jpg`, `cart-mujoco-crest.jpg` | the figure of the stills in Section 11.1 |
| `snapshots.py` | Chapter 1: one recorded instant of each robot's run (the cart of Section 11.1 and the six runs above), with each video script's visual model and a camera of its own, without captions or annotations | no video; `results/snapshots/robots-*.jpg` | the figure of the robots in Section 1.1 |
| `../ch13/husky_video.py` | Section 13.2: the Husky A300's governed crossing of the yard with the map's labels, dry | `husky-map.mp4`, 34 s | Figure 13.7 |
| `../ch14/arm_video.py` | Section 14.2: the UR5e with 4.95 kg stopped from full speed under the declared label and under the empty arm's label, each from the latest angle that its label allows; slow motion, 1 ms of simulated time per frame | `ur5e-stops.mp4`, 18 s | Figure 14.6 |
| `../ch14/hdmm_video.py` | Section 14.3: the heavy-duty mobile manipulator's reference run through the lift and the two turns to the stop line | `hdmm-reference.mp4`, 49 s | Figure 14.9 |
| `../ch14/excavator_video.py` | Not in the book: two of the PC138US-8's onset runs, the load raised slowly at the lift point to 98 and 102 % of the static tipping load (hook at the ground, reach 4.6 m, over the front) | `excavator-onset.mp4`, 21 s | none |
| `../ch15/quad_video.py` | Section 15.2: the Go2's governed lift of its front-right foot, pushed with 10 N three times | `go2-pushed.mp4`, 10 s | Figure 15.5 |
| `../ch15/humanoid_video.py` | Section 15.3: one Apollo carrying a beam (the record of `humanoid_carry.py`) | `apollo-carry.mp4`, 91 s | Figure S15.1 |

Each script also writes its stills (`*.jpg`, 1600 x 900) and a file of frames (`*-frames.csv`) that lists each frame's simulated time and the drawn state's main coordinates, so that the drawing can be checked against the record. With `--export DIR` a script instead writes the scene and its frames for the path tracer of `cycles/` (see `cycles/README.md`), which renders them with Blender's Cycles on a GPU.

## How a run is recorded and drawn

- `shapes.Recorder` replaces `mujoco.mj_step` while a run executes, by a function that calls the original step with the same arguments and then copies the state (time, `qpos`, `qvel`, and anything else a script asks for, such as the forces applied to bodies) of the `MjData` that it selects, at the frame times of the video, or after every step (the UR5e's stops, drawn in slow motion). It changes nothing in the step, so the run, its decisions and its record are those of the chapter's script, run again with the chapter's modules unchanged.
- The scripts build a visual model with the simulated model's joints, set each frame's recorded state into it and call only `mj_forward` before drawing; the drawing simulates nothing. The visual model adds geoms that do not collide and have no mass: the robots' visual meshes from their descriptions, where the simulated models have collision meshes or none (Clearpath's description of the A300, Universal Robots' of the UR5e, MuJoCo Menagerie's of the Go2, whose model file is the book's), the scenery, and the parts that a model does not have. Each script's docstring and each figure's caption say what is simulated and what is drawn.
- The heavy machine is simulated, as in Section 14.3, in the frame of the base's commanded motion; `hdmm_video.py` places that frame on the yard at each frame's commanded arc length and the simulated bodies in it, and draws every part of the machine on them: its model has four contact spheres for wheels and simple shapes for its base and arm. Its visual model adds a steering and a spinning joint for each drawn wheel, which follow the commanded motion.
- The excavator is drawn the same way, on its simulated bodies (`excavator_video.py`): the model has two boxes for its tracks, a box for its upper structure and one for its counterweight, and capsules for its front. Its load, a force at the lift point in the simulation, is drawn as a bag that fills as the force rises.
- The humanoid's run is the only one that a video script does not run again: `humanoid_carry.py` runs it (with the Team-Impedance-Control repository's package from a checkout) and writes its record, and `humanoid_video.py` draws that record in the repository's own scene.
- The Husky's, the heavy machine's and the excavator's scripts check, before they draw, that the camera keeps away from the scenery and sees the machine in every frame (`sight`), and stop if it does not.

## The modules

| File | Content |
|---|---|
| `egl.py` | A headless OpenGL context through EGL, with `ctypes` only: Mesa's surfaceless platform, a GPU's device platform or the default display. |
| `film.py` | The renderer: MuJoCo's `MjrContext` in an offscreen framebuffer at twice the image's size, with multisampling, shadows, reflections, skybox and haze, scaled down with a Lanczos filter; arrows, lines and discs added to the scene as annotations (they cast no shadow); a light colour grade; the captions; a camera path that follows its targets without lag (`smooth_path`); and the writer that pipes frames to ffmpeg (H.264, yuv420p, CRF 18). |
| `stage.py` | The stage set on a model before it is compiled: a gradient sky, a sun (directional, soft shadows) and a fill light, the ambient share, the haze and the shadow map's size and reach; `follow` moves the shadow map's region with the camera. |
| `shoot.py` | Splits the frames between processes, each with its own OpenGL context, writes one H.264 segment each and joins them without re-encoding (also a part of the frames, for long videos); draws single stills; reads a video's frame count and size with ffprobe. |
| `export.py` | Writes a video's scene and frames as a bundle for the path tracer: every geom that MuJoCo's renderer would draw, its type, size, pose, colour, material and texture, frame by frame where it changes, the camera, the lights and the captions; `check()` draws a frame of a bundle again with MuJoCo's renderer, which gives MuJoCo's frame pixel for pixel. |
| `cycles/` | The path tracer's side, run on a computer with a GPU: `build.py` (in Blender) builds a bundle's scene and renders it with Cycles, `compose.py` adds the captions and writes the video and the stills, `run_all.py` runs the whole chain for the six videos, `standin_test.py` runs `build.py`'s own logic outside Blender; `README.md` says how. |
| `shapes.py` | The recorder, and the shapes of the scenery: ribbons along a route, cones, pallet stacks, shipping containers, and visual geoms. |
| `textures.py` | Procedural textures from fixed seeds (asphalt, concrete, gravel, soil, grass, wood, tiles, corrugated steel, paint, a machined plate with threaded holes, chevrons), each periodic, so that it tiles without seams. |
| `collada.py` | A reader of COLLADA (`.dae`) meshes, which MuJoCo does not load: triangles and polygons, node transforms, the materials' diffuse colours, the file's unit and up axis. |
| `urdf.py` | The visual elements of a URDF placed on the bodies of the MuJoCo model imported from it, through its fixed joints, and their attachment as visual geoms. |

## Requirements

- MuJoCo 3.16.0 and NumPy (the chapters' requirements), SciPy (`textures.py`) and Pillow (captions, colour grade, JPEG stills); the scripts were run with Pillow 12.3.0, NumPy 2.5.3 and SciPy 1.18.1 under CPython 3.13.17.
- EGL and an OpenGL driver: Mesa's (llvmpipe, a software rasterizer, needs no graphics processor) or a GPU's. The videos were drawn with Mesa 25.2.8 on Ubuntu 24.04.
- ffmpeg with libx264 (ffmpeg 6.1.1 was used) and the DejaVu Sans fonts (`/usr/share/fonts/truetype/dejavu/`).
- The makers' visual meshes, which the book's folders do not copy: a checkout of Clearpath's `clearpath_common` at commit `78e3677` for the Husky (`--clearpath DIR`), of `Universal_Robots_ROS2_Description` at commit `7ce98d8`, the commit of the book's description, for the UR5e (`--ur DIR`), of MuJoCo Menagerie at commit `f3b6013` for the Go2 (`--menagerie DIR`), and of Team-Impedance-Control at commit `eb530ea` for the Apollo (`--team-impedance DIR`), whose scene writes its textures into the checkout's `assets/textures`, as the repository does.

```
cd code/ch13 && python3 husky_video.py --clearpath DIR
cd code/ch14 && python3 arm_video.py --ur DIR && python3 hdmm_video.py && python3 excavator_video.py
cd code/ch15 && python3 quad_video.py --menagerie DIR
cd code/ch15 && python3 humanoid_carry.py --team-impedance DIR && python3 humanoid_video.py --team-impedance DIR
```

`--quick` draws a small video without stills, to try the scripts. On two cores of a 2.10 GHz Xeon, with Mesa's software rasterizer, the full videos at 1920 x 1080 took about ten minutes (Go2), forty (Husky), seventeen (UR5e), forty (heavy machine), twenty (excavator) and ninety (Apollo, drawn in three parts with `--part` and joined with `--join`).

## What the drawing is and is not

The images come from MuJoCo's own OpenGL rasterizer: shadows from a shadow map, a mirror reflection on one flat surface at most (the renderer reflects only the first reflective plane or box face of a model), a gradient sky, haze; no ray tracing and no global illumination. The scenery is procedural and ours, but for the Apollo's, which is the Team-Impedance-Control repository's. The videos are engineering renders of simulated states, not photographs, and support no claim beyond those of the runs that they draw. The path-traced versions of `cycles/` change the light and the materials, not the geometry or the poses.
