# The videos path traced on a GPU (Blender Cycles)

This folder renders the book's six videos of its MuJoCo runs again with a path tracer, Blender's Cycles, on an NVIDIA GPU through OptiX (or CUDA, HIP, oneAPI, Metal or the CPU). The physics is not run again here. The book's video scripts write, for every frame, exactly what MuJoCo's renderer draws: each geom with its type, size, pose, colour, material and texture, plus the camera, the lights and the captions. Blender builds that scene and path traces it. The poses are therefore the record's, as in the MuJoCo videos. What changes is the light: the sky, global illumination, soft shadows and reflections. Materials are physically based, derived from MuJoCo's colours and textures with roughness, metalness and bump hints by material name (`build.py`, `NAMED`).

| Video | Script (with `--export`) | Needs | Book's stills |
|---|---|---|---|
| `husky` | `code/ch13/husky_video.py` | `--clearpath DIR` | Figure 13.7 |
| `ur5e` | `code/ch14/arm_video.py` | `--ur DIR` | Figure 14.6 |
| `hdmm` | `code/ch14/hdmm_video.py` | | Figure 14.9 |
| `excavator` | `code/ch14/excavator_video.py` | | none |
| `go2` | `code/ch15/quad_video.py` | `--menagerie DIR` | Figure 15.5 |
| `apollo` | `code/ch15/humanoid_video.py` | `--team-impedance DIR`, and `results/humanoid-frames.npz` from `humanoid_carry.py` | Figure S15.1 |

## What you need

- Blender 4.2 or later (4.5 LTS or a 5.x release), from blender.org. Nothing has to be installed into Blender's Python.
- The book's Python environment (MuJoCo 3.16.0, NumPy, SciPy and Pillow; `code/*/requirements.txt`). The exports need no OpenGL. EGL is used only by the MuJoCo videos.
- ffmpeg on the PATH, and for the videos' captions the DejaVu Sans fonts, as the MuJoCo videos use (matplotlib's copy will do; else `compose.py` takes Liberation Sans or Arial and says so, or the fonts that `run_all.py --font FILE --font-bold FILE` names). The stills have no captions.
- The makers' checkouts that the MuJoCo videos use (`code/render/README.md`): `clearpath_common` at `78e3677`, `Universal_Robots_ROS2_Description` at `7ce98d8`, MuJoCo Menagerie at `f3b6013`, and Team-Impedance-Control at `eb530ea` for the Apollo.

## Step 0: the smoke test (do this first)

```
python code/render/cycles/run_all.py --blender "PATH/TO/blender" --smoke \
    --clearpath DIR --ur DIR --menagerie DIR --team-impedance DIR
```

For each video, this writes the bundle (`cycles-work/<video>/bundle/`, a few seconds to a few minutes each, since the script runs the simulation again) and renders one still at 640 x 360 with 32 samples (`cycles-work/<video>/smoke/frame_NNNNN.png`). Before the long renders, check that:

- the images show the scenes, not black or white frames (the automatic exposure aims at a mean display value of 0.40; `--exposure E` adds E stops);
- `cycles-work/run_all.log` reports that Cycles renders on OPTIX with your GPU's name, not "no GPU found".

`build.py` was written against Blender's Python API from its documentation and source, and has not yet been run in Blender. These parts have been tested without Blender: the exporter (`render/export.py`, `check()`: a bundle drawn again with MuJoCo's renderer gives MuJoCo's frame pixel for pixel, on three to seven frames of each of the six videos; a bundle exported twice, once without any OpenGL, is the same); `build.py`'s own logic, on every bundle, with stand-ins for Blender's modules (`standin_test.py BUNDLE`: the shapes, every corner normal of unit length, the materials, the tracks and the frame loop; Blender's API itself is not tested so); the shapes' windings against their normals; the sun's direction against Cycles' source for its sky models; and `compose.py`. If the smoke test fails, `cycles-work/run_all.log` says where.

## Step 1: the book's stills

```
python code/render/cycles/run_all.py --blender "PATH/TO/blender" --stills-only \
    --clearpath DIR --ur DIR --menagerie DIR --team-impedance DIR
```

This renders the 24 stills of the six videos, those of Figures 13.7, 14.6, 14.9 and 15.5 of the book and Figure S15.1 of the supplement and four of the excavator's video, at 1920 x 1080 with 1024 samples and the OptiX denoiser (`--still-samples`). It writes them as 1600 x 900 JPEGs, the book's size, into `cycles-work/output/`.

## Step 2: the videos

```
python code/render/cycles/run_all.py --blender "PATH/TO/blender" \
    --clearpath DIR --ur DIR --menagerie DIR --team-impedance DIR [--samples 128]
```

The frames are 1920 x 1080 with 128 samples (`--samples`) and the denoiser. The six videos have 630 (excavator), 301 (Go2), 542 (UR5e), 1021 (Husky), 1471 (heavy machine) and 2732 (Apollo) frames. A run that stops can be resumed with the same command: frames already rendered are skipped (`build.py --overwrite` renders them again). `--only excavator,go2` renders some of the videos.

## Output

`cycles-work/cycles-output.zip` holds the stills (`*-cycles.jpg`), the videos (`*-cycles.mp4`), the render settings that each step used (`*-render.json`: Blender's version, the device, the samples and the exposure) and the log. The stills, the `*-render.json` files and the log are what the book's figures need; the videos are for watching.

`--humanoid-run` also runs `code/ch15/humanoid_carry.py` again with daqp, the repository's QP solver (`--qp daqp --check`, which needs daqp installed), into `cycles-work/humanoid-daqp/`, leaving the book's results, from humanoid/qp.py's solver, as they are; its summary and record go into the zip as `humanoid-daqp-summary.txt` and `humanoid-daqp-record.txt`, and the summary's last line says whether every value that the book quotes came out the same.

## Notes

- A bundle holds the meshes of the robots it draws, the makers' meshes among them, under the makers' terms. The bundles are for rendering on your machine, not for distribution.
- `build.py` renders frame by frame from Python, without keyframes, so that it does not depend on Blender's animation API, which changed in the 4.4 and 5.0 series; `--blend FILE` saves the built scene at the first frame for a look in Blender's interface.
- The sky is Blender's physical sky (multiple scattering in 5.x, Nishita in 4.x) with its sun in the direction of the bundle's sun. `--sun-lamp` lights the scene with a sun lamp instead of the sky's sun disc, if the disc's light is noisy on your GPU.
- The overlays of the MuJoCo videos (arrows, polygons, marks on the floor) are drawn as emitting materials that cast no shadow; the captions are added by `compose.py` as `render/film.py` draws them.
