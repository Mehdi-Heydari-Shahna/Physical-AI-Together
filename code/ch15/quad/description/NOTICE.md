# The Go2 description

The folder holds files copied without change from MuJoCo Menagerie's model of Unitree Robotics' Go2 quadruped:

- repository: https://github.com/google-deepmind/mujoco_menagerie
- the folder `unitree_go2/` as it stands at commit `f3b6013629c7b94f8ee93fd977b161fa8350f9bc` (23 October 2026), the repository's head when the files were copied; `go2.xml` last changed in commit `ac6b2b0` (18 May 2026), which reformatted every model's XML file, and the folder last changed in commit `6da8b97` (14 October 2026), which changed only `CHANGELOG.md` and the MJX variant `go2_mjx.xml`
- licence: BSD 3-Clause, copyright Unitree Robotics (`LICENSE`)

Copied:

| File | Content |
|---|---|
| `go2.xml` | the model: 13 bodies, 12 hinge joints and a free joint, 12 torque motors, the feet's contacts and the solver's options |
| `README.md` | Menagerie's description of the model and of its derivation from Unitree's URDF |
| `CHANGELOG.md` | the model's changes |
| `LICENSE` | the licence |

Not copied: the meshes (`assets/`), the scene files and the MJX variant. `quad/model.py` deletes the visual meshes from the model before it compiles it, so that it needs no mesh file; they carry no mass and take part in no contact, and `quad_model.py --menagerie DIR` compares the model compiled with them from a Menagerie checkout with ours: at commit `f3b6013`, the same masses, inertias, centres of mass and collision geometries (`results/model-summary.txt`, which was written with that option).
