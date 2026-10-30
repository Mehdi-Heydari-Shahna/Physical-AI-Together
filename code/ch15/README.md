# Chapter 15: a governed quadruped, and legged robots beyond it

Section 15.1 documents the Go2 of MuJoCo Menagerie as it is published and as we use it, the representations of its state and contacts in MuJoCo, in its stance controller and in the reduced model that the governor's test uses, and compares the reduced model, a cart-table model on a shift line, with the simulated robot (`quad_model.py`). Section 15.2 governs the Go2 with the allovi core of Chapter 10 through a lift of its front-right foot: the lift is a demand on the triangle of the other three feet, the label a bound on the pushes on the trunk; it runs the reference and five configurations that change one thing each, and the operator's proposal without the governor, with and without pushes; sweeps pushes over the reference run; executes every plan in force of two runs unchanged; lowers the foot onto the floor at four speeds on two contact models; and shuts the robot down (`quad_governed.py`); and draws the pushed run as a video (`quad_video.py`). Section 15.3 runs one Apptronik Apollo carrying a beam with the two-loop controller of the Team-Impedance-Control repository, loaded from a checkout, with one robot (`humanoid_carry.py`), and draws the run as a video (`humanoid_video.py`).

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python quad_model.py                # writes results/model-*
.venv/bin/python quad_model.py --check        # also checks every value of its summary
.venv/bin/python quad_governed.py --check     # results/governed-*
.venv/bin/python humanoid_carry.py --team-impedance DIR --check    # results/humanoid-*
.venv/bin/python humanoid_carry.py --team-impedance DIR --without-follow --check   # results/humanoid-without-follow.txt
```

The scripts need MuJoCo 3.16.0, NumPy and SciPy, as `requirements.txt` lists, and the core of Chapter 10 (`../ch10/allovi`), which they import from its place in this repository. They take about 10 s and twelve minutes on the machines that wrote their shipped results (below); `quad_governed.py` runs its sweep on two processes (`--workers`). With `--check`, each script compares every value of its summary but the versions of the libraries with its `EXPECTED`, which holds among them every value that Section 15.1 or 15.2 quotes from it, and exits with an error if one differs; it does not compare the data files of the figures. `quad_model.py --menagerie DIR`, with a checkout of MuJoCo Menagerie at the commit of the record, also compares go2.xml compiled with its meshes with our model.

## The package `quad/`: the Go2

| File | Content |
|---|---|
| `description/` | MuJoCo Menagerie's `unitree_go2/go2.xml`, `README.md`, `CHANGELOG.md` and `LICENSE`, copied without change from commit `f3b6013` (BSD 3-Clause, Unitree Robotics). See `description/NOTICE.md`. |
| `model.py` | The MjSpec of the Go2: go2.xml without its visual meshes, which carry no mass and take part in no contact, a floor plane added, a time step of 1 ms (go2.xml leaves MuJoCo's default of 2 ms); the feet's contact impedance replaced, for a comparison only, by MuJoCo's default (`STIFF_SOLIMP`); the ground's wrench on the robot: each foot's force, the force on other parts, and the centre of pressure of the full wrench, the contacts' torques included, and of the forces alone (`ground_wrench`). |
| `kinematics.py` | The legs' forward and inverse kinematics, the foot sphere's centre against the hip, the thigh and the knee, checked against MuJoCo. |
| `stance.py` | The stance controller (ours): the ground forces on the feet of least weighted norm that give the planned acceleration of the centre of mass with no moment about it, a foot's weight the inverse of its share of the load (`forces`); the trunk's place, level at a given height, at which the model's centre of mass lies where planned with the feet on their places (`base_for_com`); the joint torques, the forces through the legs' Jacobians, the legs' gravity, and joint servos of 120 N m/rad and 3 N m s/rad towards the inverse kinematics, clipped at the motors' limits (`torques`). |
| `sim.py` | The simulated robot (`Machine`): settled at rest, one second on joint servos at Menagerie's home angles and two on the stance controller, the trunk at 0.27 m and the centre of mass held at home by an integral correction (2/s); then run one period at a time on a plan of the centre of mass, its acceleration, FR's load share and height, and the trunk's height, with the motors' torques scaled by a power factor (1 on, 0 off) and horizontal pushes on the trunk (`Push`); it logs every step. The joints' static torques at rest on four feet or on three (`static_shares`). |
| `support.py` | Support polygons: the signed distance from a convex polygon's boundary, concave, positive inside, and a triangle's incentre and inradius. |
| `reduced.py` | The reduced model (declared): the feet's places, home, the centre of mass's height and mass, from the settled stance rounded to 0.1 mm; the rectangle of the four feet and the triangle of FL, RL and RR; the shift line from home towards the triangle's incentre; the cart-table zero-moment point; the period of 0.2 s and the half-cosine motion within it; the operation's profile, FR's load share and target height over its six phases, unload, lift, hold, lower, touchdown and reload, three seconds in all, with the floor's height that the package believes. |
| `governed.py` | Section 15.2: the Go2 as modules of the allovi core. The control package `go2.stance`: segments of one period, the centre of mass's acceleration along the shift line and the operation's profile; plans of (segment, label) pairs and a rest; the funnel's bounds (ours): the centre of mass within 8 mm of its commanded place, FR within 20 mm of its target; two families of constraints, f4 (the rectangle, at its floor) and f3 (the triangle, enlarged by C3 = 60 mm while FR carries its share), with the levels carrying f3 from its floor of 20 mm on a grid of 5 mm; the test: the zero-moment point's least distance over a segment less the funnel's bound, an allowance of 5 mm and 3 mm per newton of the label's push, the acceleration within 0.25 m/s^2, the friction of 0.5 and the static torques counted 1.5 times; the recovery, the fastest motion to rest by a linear programme, at a requirement's level to the place where the rest holds it, during the operation the rest of it; the generator; the rest feedback, which lowers FR at once, and the restart. The label, a bound on the push on the trunk, the global capability 10 N, and the fixed predictor; the operation as a task and its demand predictor; the operator's proposal; the robot and world adapters with the world's oracle for the checks of Section 10.3; and the operator's proposal executed without the governor (`ungoverned`). |

## The package `humanoid/`: one Apollo

| File | Content |
|---|---|
| `one_robot.py` | The repository's package `teamimp` loaded from a checkout, unchanged, and what one robot needs that it does not have: a station for one robot (on the beam's -y side, facing +y); the beam, aluminium with the cross-section of an HEA 200 (EN 10365: 190 x 200 mm, web 6.5 mm, flanges 10 mm, 42.3 kg/m in steel, scaled by the densities 2700 and 7850 kg/m^3 as the repository scales its girder), 1.2 m long, on trestles 0.5 m either side of its middle; the pelvis shifted over the stance foot before a loaded step as far as before an unloaded one (0.95 of the way, where the repository's team shifts a loaded robot's 0.25); and the beam's reference following the pelvis's reference sideways while it is carried (without it, `humanoid_carry.py --without-follow`, the girder is lost in the second loaded step). Everything else, the controller, its gains, the load sharing, the gait and the phases, is the repository's, with `kz = 0` as in its main run. |
| `qp.py` | The load sharing's QP, solved with daqp's interface by SciPy's NNLS as a least-distance problem (Lawson and Hanson, 1974): the same problem, for an environment without daqp; `humanoid_carry.py --qp daqp` uses daqp instead. |

## What the scripts do

`quad_model.py`

- states **the model**: bodies, joints, motors, masses, the motors' limits, the joints' ranges, damping, armature and friction loss, the feet's contacts, the other collision geometries, the solver's options, and what `quad/model.py` changes; with `--menagerie`, the comparison with go2.xml compiled with its meshes;
- settles **the stance** and reports the feet's places, sink and loads, the centre of mass, the declared constants against them, the rectangle, the triangle's incentre and inradius, and home's distances inside them;
- measures **the floor's softness**: a free foot's sink under loads from 1 to 80 N;
- shifts the centre of mass by **55.5 mm** at 0.1, 0.25 and 0.5 m/s^2 and compares the simulated centre of pressure with the planned zero-moment point and with the cart-table point of the simulated motion, the centre of pressure of the forces alone with the full one, and the centre of mass with its plan; and repeats the shift at 0.25 m/s^2 with a time step of 0.5 ms (`results/model-shift.csv`, Figure 15.2).

`quad_governed.py`

- writes **the record** (`results/governed-record.txt`);
- states what the test derives from **the reduced model**: a push's point and its effect per newton, the force that the motion and the push need, the triangle, home's distances, the allowance in all, the rest's level at home and at the requirement's place, and the joints' static torques;
- runs **the governed robot** in six configurations, the reference and five that change one thing, and the operator's proposal without the governor, with the pushes and without, each for 50 periods: when the operation begins and FR leaves the floor, the simulated centre of pressure's least distance inside the triangle and the rectangle and above the level in force, the centre of mass's and FR's deviations, the torques, the stance feet's least load, the touchdown, the governor's interventions, the invalid labels, the alarms and the restarts (`results/governed-trace-*.csv`, Figure 15.4);
- reports **the reference run's preparation**: the requirement, the levels in force, the committed plans and the best candidates of ambition 1;
- runs **the checks of Section 10.3** on the reference and on the shove;
- sweeps **pushes** of 10 N in eight directions over nine phases of the reference run: the deviations and the shortfalls against the funnel's bounds and the allowance, and FR's first contact: its speed against the commanded descent, whether FR leaves the floor again within 0.1 s, and whether the contact comes before the slow descent begins or after it ends;
- executes **every plan in force unchanged** in the reference and in the pushed run: the least simulated distance above each pair's profile entry, and the deviations;
- lowers FR onto the floor from the hold at **four speeds** on Menagerie's feet and on MuJoCo's default impedance: the contact's speed, FR's force, the centre of pressure's jump and its direction, the centre of mass, the other feet's load, and whether FR leaves the floor again;
- runs **the shutdown**: the crouch, the torques cut, and the robot lying on the floor.

`humanoid_carry.py --team-impedance DIR`

- runs **the mission** with one robot (DIR a checkout of Team-Impedance-Control at commit `eb530ea`): two gait cycles of 0.25 m to the beam, the reach under it, the transfer of its weight from the trestles, the lift, four cycles carrying it, the descent onto two other trestles, the release, standing up and one cycle back; it records the plant's state at every step of 1 ms with the soles' and palms' contact forces;
- reports **the run**: whether it completed, its phases, its lift-offs, touchdowns and grasps; in single support, the stance sole's centre of pressure's least distance inside its edges, unloaded and carrying, each loaded step's least, the samples at an edge while carrying, when and at which edge, and how far inside the edges the others between them were; each touchdown's speed at the first contact and the largest force within 50 ms; the actuators' largest share of their torque limits; the beam's distance from its reference while carried, the palms' share of its weight, and where it was set down;
- writes **the record** (`results/humanoid-record.txt`), the summary, a trace every 10 ms and the frames for the video.

It takes about four minutes on one core. With `--check` it compares every value of its summary but the wall time with its `EXPECTED`, which holds every value that Section 15.3 quotes; `--qp daqp` repeats the run with the repository's QP solver, for comparison, and `--out DIR` writes such a run's outputs into DIR, leaving `results/` as it is. `--without-follow` runs the mission without the beam's reference following the pelvis sideways, the third of the adaptations for one robot, and reports in `results/humanoid-without-follow.txt` whether and where it fails and which actuators reached their torque limits before, each from when (about two minutes; `--check` compares it with `EXPECTED_WITHOUT_FOLLOW`). After the girder is lost the mission runs on for about 2.5 s, in which the simulation becomes unstable; MuJoCo says so in its log, `MUJOCO_LOG.TXT`, which it writes in the working folder.

Every distance, level and margin is rounded down, and every deviation, shortfall, share, force and speed rounded up, at the digits printed; but in `humanoid_carry.py`'s summary the largest distance inside the sole's edges of the samples between those at an edge is rounded up, since it bounds how near the edge they were.

## The video

`quad_video.py --menagerie DIR` draws the pushed run of Section 15.2 as a video, with the package `../render`, whose README gives what it needs besides this chapter's requirements: Pillow, EGL and ffmpeg; DIR is a checkout of MuJoCo Menagerie at the commit of the record, `f3b6013`, for the meshes of go2.xml. The script runs the configuration `pushed` of `quad_governed.py` again with this chapter's modules unchanged, records the simulated state at 30 frames per second of simulated time (it copies the state after MuJoCo's steps and changes nothing in them), and draws the record: go2.xml compiled with its visual meshes, set to each recorded state, the push as an arrow while it acts, and from 3.2 to 6.2 s the triangle of FL, RL and RR and the centre of mass over the floor; the floor lies where the simulated floor does, and its tiles, the mat, the walls and the cones are scenery. The drawing simulates nothing, and no result above changes. It takes about ten minutes on two processes at 1920 x 1080 (`--quick` draws a small video, without stills).

```
.venv/bin/pip install pillow
.venv/bin/python quad_video.py --menagerie DIR    # results/video/
.venv/bin/python humanoid_video.py --team-impedance DIR    # after humanoid_carry.py
```

`humanoid_video.py --team-impedance DIR` draws the record that `humanoid_carry.py` writes, in the repository's own scene (the construction site, the trestles and the beam, built by its package from the same checkout, which writes its textures into the checkout's `assets/textures`): the Apollo with MuJoCo Menagerie's meshes as the repository decimates them, set to each recorded state; on the ground the polygon of the soles that carry the robot and the zero-moment point, with a pole over it, and a ring round a foot after its touchdown; and in a corner the soles from above with the zero-moment point and its distance inside the stance sole. It takes about ninety minutes on two processes at 1920 x 1080 (drawn here in three parts); `--part K/N` draws a part of it and `--join N` joins the parts and writes the stills. `--export DIR`, in every video script, writes the scene and frames for the path tracer of `../render/cycles` instead.

## Output (`results/`)

| File | Content |
|---|---|
| `model-summary.txt`, `governed-summary.txt` | Every value, by group; with `--check`, the result of the comparison. |
| `governed-record.txt` | The record of the experiment of Section 15.2 (Table S15.4). |
| `model-shift.csv` | The shift of 55.5 mm at 0.25 m/s^2, every 5 ms (`t`, s): along the shift line from home [m], the planned zero-moment point (`zmp`), the centre of pressure (`cop`), that of the forces alone (`fcop`), the planned and the simulated centre of mass (`c_p`, `c`) (Figure 15.2). |
| `governed-trace-*.csv` | For the reference, the pushed run, the shove and the operator without the governor, every 10 ms: the time (`t`, s), the centre of mass's commanded and measured place on the line (`s_c`, `s_m`, m), FR's height and target (`h`, `h_t`, m), the simulated centre of pressure's distance inside the triangle and the rectangle (`d_tri`, `d_rect`, m), FR's contact force (`f_fr`, N), the push (`push`, N) and the level in force (`level`, m; empty under the rest feedback and without the governor) (Figure 15.4). Figure 15.3 plots the reference run's. |
| `video/go2-pushed-<t>s.jpg` | Four stills of the pushed run's video, at 2.30, 4.60, 5.60 and 8.00 s, 1600 x 900 (Figure 15.5). |
| `video/go2-pushed-frames.csv` | The video's frames: the time (`t`, s), the trunk's position (`base_x`, `base_y`, `base_z`, m) and the push (`push_x`, `push_y`, N) of each, the check that the drawing follows the record. |
| `video/go2-pushed.mp4` | The video, 10 s at 30 frames per second, 1920 x 1080 (H.264). It is not in the repository's history, for its size, but attached to its releases; `quad_video.py` writes it. |
| `humanoid-summary.txt`, `humanoid-record.txt` | The Apollo's run: every value, by group, and its record. |
| `humanoid-without-follow.txt` | The run without the beam's reference following the pelvis: whether and where it failed, and the actuators at their torque limits before it. |
| `humanoid-trace.csv` | The run every 10 ms: the time (`t`, s), the phase, the feet and palms in contact, the soles' normal forces (`f_left`, `f_right`, N), in single support the stance sole's centre of pressure's distance inside its edges (`cop_margin_mm`), the actuators' largest share of their limits, the beam's height (`beam_z`, m) and its distance from its reference while carried (`beam_error_mm`). Figure S15.2 plots the margin on the stance sole. |
| `humanoid-frames.npz` | The state at 30 frames per second, with the soles' and palms' contact forces and centres of pressure, the phase and the mode of each frame, for `humanoid_video.py`. |
| `video/apollo-carry-<t>s.jpg` | Four stills of the Apollo's video, 1600 x 900 (Figure S15.1). |
| `video/apollo-carry-frames.csv` | Its frames: the time, the phase, the pelvis's and the beam's positions. |
| `video/apollo-carry.mp4` | The video, 91 s at 30 frames per second, 1920 x 1080 (H.264), not in the repository's history but attached to its releases. |

## Provenance and reproducibility

The Go2's files are MuJoCo Menagerie's (BSD 3-Clause, Unitree Robotics, `quad/description/LICENSE`); every other file is ours. The Apollo's model and the controller of Section 15.3 are not copied here: they are read from a checkout of the Team-Impedance-Control repository (no project-wide licence; its Apollo model from MuJoCo Menagerie, Apache License 2.0). The Go2 of Menagerie is derived from Unitree's description, not identified from a robot, and no number of this chapter was measured on one.

MuJoCo computes in double precision. The shipped results were produced on one x86-64 Linux machine (a 2.10 GHz Xeon), and those of `quad_governed.py` on a second (a 2.80 GHz Xeon), with MuJoCo 3.16.0 built from its source at the tag 3.16.0 (commit `a892750`) by clang 18.1.3, under CPython 3.13.17 with NumPy 2.5.3 and SciPy 1.18.1; repeated runs wrote identical summaries, and on the second machine `quad_governed.py` wrote every value and data file that an earlier run had written on the first unchanged. They have not been reproduced with other versions, nor `quad_model.py` on another machine; there the floating-point results may differ in the last digits, and the check reports any value that differs at the precision printed.
