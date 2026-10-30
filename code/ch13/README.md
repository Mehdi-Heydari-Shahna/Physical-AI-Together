# Chapter 13: wheeled robots and the Husky

Section 13.1 builds the Husky A300 in MuJoCo and measures what the simulated robot can do (`husky_mujoco.py`); Section 13.2 runs it under the governor of Chapter 10, with an estimator, a feedback, a stop built from labels and a forecast of a shower (`husky_route.py`); Section 13.3 substitutes modules and changes the robot and its world one thing at a time on the same task (`husky_variants.py`). `husky_video.py` draws Section 13.2's crossing with the map's labels as a video.

## Section 13.1: the yard, the Husky and what it can do

This folder holds the Husky A300 of Chapter 13, built in MuJoCo from Clearpath's own description, with the drive controller that the robot runs, and the script `husky_mujoco.py`, which records the experiment and runs the diagnostics of Section 13.1. It needs MuJoCo 3.16.0 and NumPy (`requirements.txt`) and Python 3.10 or later; it does not need ROS or xacro.

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python husky_mujoco.py            # writes results/
.venv/bin/python husky_mujoco.py --check    # also checks every value that Section 13.1 quotes
```

The script takes about a minute.

## The package `husky/`

| File | Content |
|---|---|
| `description/` | Clearpath's description of the A300, copied from `clearpath_common` at commit `78e3677` (release 2.9.18, BSD 3-Clause; see `description/LICENSE` and `description/NOTICE.md`), with our file `husky_a300.urdf.xacro`, which assembles the robot from Clearpath's macros, and the URDF that xacro 2.1.1 expanded from it. |
| `asset.py` | The URDF text that the simulators load: visual elements and the collision meshes of the parts without mass removed, and one correction, declared: the right suspension beam's wheel mounts mirrored, which makes the track 0.5658 m instead of the published 0.5468 m (`published=True` gives the description as published). Also the controller's configuration (`diff_4wd.yaml`), read without a YAML library, and the expansion of the macros again (`python3 -m husky.asset --expand`, with xacro installed). |
| `world.py` | The yard: the route (10 m east on asphalt, a quarter turn of radius 3 m on gravel, north onto wet grass at s = 18 m, the dock's face at s = 25 m), its lane of 1.4 m, the three grounds and their frictions (0.8, 0.6, 0.35, the simulator's choices), the pallets and the dock; `project(x, y)` gives the route coordinates (s, d), and `footprint_offset` the farthest point of the robot's footprint from the centre line. |
| `drive.py` | A port of the A300's drive controller, `diff_drive_controller` of ros2_controllers (branch jazzy, commit `d71703b`) with the rate limiter of control_toolbox (commit `1e54022`): the limits, the wheel speeds and the odometry. Translated from Apache-2.0 sources (`LICENSE-Apache-2.0.txt`). |
| `model.py` | The MjSpec of the robot in its world: MuJoCo's URDF import plus what the description does not hold, each a declared choice: a free joint, no mass for the bumpers (`bumper_mass=True` keeps the 1.238 kg that MuJoCo infers for each), the wheel actuators and sensors, the options of the simulator (0.5 ms, `implicitfast`, Newton, elliptic cones, impratio 100), and the contact pairs of the wheels with the ground: impedance 0.05, normal time constant 8 ms, friction damping 60 1/s (`contacts="default"` gives MuJoCo's defaults). |
| `sim.py` | `Husky`: the model stepped with the drive controller at 50 Hz and the drivers' speed loops at every step (proportional through MuJoCo's velocity actuator, integral through an offset of its control, with anti-windup), the friction of each wheel set from the ground under it, placement 0.1 mm below touching, and what the simulator knows: the pose, the velocities, the wheels, the contacts and the loads. |
| `coulomb.py` | The steady motion of a skid-steered robot under Coulomb friction for given rim speeds and contacts: the reference against which Section 13.1.3 checks MuJoCo's friction. Quasi-static by default (no inertia, the loads at rest); given the mass and the centre of mass, also with the centripetal force of a turn, and with the loads measured in it (`ContactsInMotion`). |
| `estimate.py` | Section 13.2: the gyro's model (ours: bias 0.004 rad/s, noise 0.001 rad/s per reading at 50 Hz, seed 13) and the estimators of the robot's pose: the A300's odometry, dead reckoning (gyro and encoders at the rolling radius 0.16499 m), dead reckoning with the sideslip of a turn to either side (calibrated on each ground), and two variants (the controller's radius; the gyro's bias unestimated). |
| `follow.py` | Section 13.2: the feedback that follows the route at the commanded speed of a plan, on the estimate: a speed loop on the arc length, a path-following yaw rate with an integral on the yaw rate (gains ours). |
| `governed.py` | Section 13.2: the Husky's modules for the allovi core of ../ch10, registered by name: the state (s, u, d, e) and its norm; the control package `husky.route-speed` (segments of a commanded speed, the funnel with its four bounds, the reserves at the dock and in the lane, the test, the recovery continuation, the rest feedback and the restart), the proposal `husky.cruise`, the capability predictors `husky.fixed`, `husky.reactive`, `husky.map` and `husky.forecast`, the robot adapter `husky.robot`, the world adapter `husky.yard`, the evaluator `husky.metrics`, and the shower. |
| `variants.py` | Section 13.3: the substitutes and the changes: the feedback of pure pursuit and the control package `husky.route-speed-pursuit` that uses it; the proposals `husky.hurry` (1.5 m/s, never rest) and `husky.careful` (0.5 m/s); the payload of 51.5 kg (`payload_spec`); the estimator whose readings arrive late, with and without a compensation from the robot's commands (`DelayedDeadReckoning`); and `PeakHusky`, the robot of `sim.py` that also records the drives' torques at every step of the simulator. |

## What the script does

- **The record** of the experiment: versions, the settings of the simulator and those that differ from MuJoCo's defaults, the contact parameters, the size of the model, the description and the hash of the URDF text loaded, the controller and the drivers (`results/record.txt`).
- **The description**: masses, the wheel cylinder, the track of the published description and of the corrected one with the loads at rest of each, the mass that MuJoCo infers for the bumpers, the penetration and the clearance at rest, the contacts of a wheel.
- **The touching start**: the robot placed exactly on the ground, and 1e-12 m lower; a cylinder on a box, with MuJoCo's default convex collision, with a tolerance of 1e-9 and without `nativeccd`; what the reversed contacts do in the first step and while the robot settles, against the placement 0.1 mm lower.
- **The contact of the wheels**: the slip speed at which a sliding contact opens, tested on a point mass pushed at its Coulomb friction (by bisection of the slip speed at which it first loses its contact), with MuJoCo's defaults and with the declared parameters; the same speed for the Husky's own contacts, and, for the declared ones, the slip speed at which a steady load reaches Coulomb's force; the spin, the turn and a sideways slide with MuJoCo's default contact parameters; the same with the declared ones on each ground, against the Coulomb references, and the creep under a sideways push against the creep that MuJoCo's soft friction predicts; and the declared parameters with one at a time changed.
- **What the simulated robot can do**, through its controller: the stop from a commanded 1 m/s on each ground, against wheels that followed the controller's commands exactly and against braking at mu g, with the excess split into the wheels' turning and slip; the spin commanded at 1 and 2 rad/s; the turn commanded at a radius of 3 m; the route driven without feedback and switched on the odometry, with the robot's footprint (the manual's length and width) against the lane; the speed against the command and the odometry over 10 s; the drivers' speed errors while the controller ramps; and the heading odometry over a quarter turn.
- **Time steps**: the stops, the spins at 0.98 m/s and the turns, on each ground, at 1, 0.5 and 0.25 ms.
- **The task's geometry and arithmetic**: the lane's room for the footprint on the straights and in the turn (by bisection with `world.footprint_offset`, and from the outer corners), the torque that the manual's climbing ability asks, the turn against the Coulomb reference with its centripetal force and the loads of the turn, and the decay of a free slip below mu g / b.

## Output (`results/`)

| File | Content |
|---|---|
| `record.txt` | The record of the experiment. |
| `husky_a300.urdf` | The URDF text that MuJoCo loads, with the corrected track. |
| `mujoco-summary.txt` | Every value, by group; with `--check`, the comparison with the values that the book quotes. |
| `contact-onset.csv` | The point mass pushed at its Coulomb friction from each slip speed: the least and the largest load, with MuJoCo's defaults and with the declared parameters (Figure 13.2a). |
| `spin-default.csv` | The spin in place on asphalt with each set of parameters, every 5 ms: the yaw rate and the load on the ground (Figure 13.2b, c). |
| `braking.csv` | The speed of the robot every 5 ms while it stops from 1 m/s on each ground. |

## Numerical precision and reproducibility

MuJoCo computes in double precision and no random numbers are used. On one x86-64 Linux machine, the results were identical, digit for digit, with MuJoCo 3.16.0 under CPython 3.11.15 (NumPy 2.4.6), 3.13.16 (NumPy 2.5.4) and 3.14.6 (NumPy 2.5.4); the group on the task's geometry and arithmetic was added later. On a second x86-64 Linux machine (a 2.80 GHz Xeon), with MuJoCo 3.16.0 built from its source at the tag 3.16.0 (commit a892750) by clang 18.1.3, under CPython 3.13.17 (NumPy 2.5.3), every value and every CSV file was reproduced, and that machine wrote the shipped `record.txt` and `mujoco-summary.txt`, with that group. On a third x86-64 Linux machine (a 2.10 GHz Xeon), with the same build of MuJoCo and the same CPython and NumPy, the chapter's three scripts were run again: every summary and every data file was identical to the shipped ones, and the records differed only in the line that names the hardware; the shipped records remain those of the second machine. `braking.csv` was written again afterwards, when its time column was corrected (it had repeated 0.245 s after the stop on asphalt); its speeds are those of the runs above, digit for digit. Other processors, operating systems and versions of MuJoCo have not been tested; their floating-point results may differ in the last digits, and the check reports any value that differs at the precision quoted.

## Section 13.2: estimation, feedback and a governed stop

```
.venv/bin/python husky_route.py            # writes results/route-*
.venv/bin/python husky_route.py --check    # also checks every value that Section 13.2 quotes
```

The script needs the environment of Section 13.1 and the allovi core of `../ch10` (Python only); it takes about four minutes. It

- **writes the record** of the experiment (`results/route-record.txt`), with the lane's narrowest allowance at the funnel's heading error and the lane's reserve, which it computes and checks against the constants of `governed.py`;
- **calibrates** what the estimators need: the rolling radius, against Section 13.1's value; the sideslip of the turn of radius 3 m at 1 m/s on each ground (the constants `estimate.SIDESLIP`, checked to three digits), and the same turn to the right; the gyro's bias over 2 s at rest;
- **crosses the yard** under the governor with the map's labels and the proposal `husky.cruise`, once with each of the five estimators, and records the estimate's errors and the feedback's distances from the commanded motion in all four components of the state, the state's distance from the commanded motion and from its funnel, the footprint against the lane, the touches and the alarms (`results/route-errors.csv`, Figure 13.5; `results/route-decisions.csv`, the decisions of the crossing with the estimator with sideslip, written by the core);
- **crosses it with each predictor**: fixed, reactive and map in dry weather; map and forecast in the shower (friction 0.15 on the wet grass from 20 s of the crossing, announced at 10 s), with the last 4 s before rest of the map and the shower crossings (`results/route-approach.csv`, Figure 13.8);
- **stops the robot at its labels** from 0.25, 0.5, 0.75 and 1 m/s on asphalt, gravel and wet grass, on wet grass at 3 m/s^2, and in the shower at its label, at the global capability and at the map's label, and checks the bounds that a label promises;
- **repeats the crossing with the map** at 1 and 0.25 ms, and with the gyro's seeds 1 to 20, for the estimator with sideslip and for dead reckoning.

| File | Content |
|---|---|
| `route-record.txt` | The record of the experiment of Section 13.2. |
| `route-summary.txt` | Every value, by group; with `--check`, the comparison with the values that the book quotes. |
| `route-errors.csv` | The crossing with the map, with dead reckoning (`_dr`) and with dead reckoning with sideslip (`_ss`), every 0.1 s: the true arc length and the errors along and across the route. |
| `route-approach.csv` | The last 4 s before rest and 0.2 s after, every 20 ms, dry with the map and in the shower with the map and the forecast: time from rest, commanded and true arc length, estimate, commanded and true speed. |
| `route-decisions.csv` | One row per period of the crossing with the map and the estimator with sideslip, in the format of Chapter 8. |

The gyro's noise is drawn from NumPy's PCG64 generator with the seed 13 (seeds 1 to 20 in the repetition). The shipped results were produced on the second machine named above, where repeated runs wrote identical files; the third machine named above reproduced them identically. They were written with contract 1.0 of the core of Chapter 10, which the summary records (`allovi core, contract`); the core is now at contract 1.1, whose two optional additions this script does not use, so a run of the present code writes the same values and records 1.1, which its `EXPECTED` holds.

## Section 13.3: substitutions and changed assumptions

```
.venv/bin/python husky_variants.py            # writes results/variants-*
.venv/bin/python husky_variants.py --check    # also checks every value of its summary
```

The script needs what `husky_route.py` needs and takes about four minutes. It keeps the task, the yard, the observations and the record of Section 13.2 and changes one thing at a time:

- **substitutes** the feedback by pure pursuit at look-aheads of 0.5, 0.75, 1 and 1.5 m, with the stops at the labels at 0.5 and 1.5 m and, at 0.5 m, the crossing at 1 and 0.25 ms and with the gyro's seeds 1 to 20, the tests of Section 13.2; and the proposal by `husky.hurry` and `husky.careful`, comparing the committed plans of `husky.hurry` with the reference's, period by period;
- **adds the payload**: the rolling radius again, the crossing, and the stops at the labels on each ground;
- **lowers the wet grass's friction** under the map's label, from 0.35 to 0.15, in the stops at the label (`results/variants-friction.csv`, Figure 13.9a) and in the crossing at 0.20 and 0.15, the last compared with the shower of Section 13.2;
- **lowers the drivers' torque limit**, in the stops on asphalt at its label from 30 to 10 N m (`results/variants-torque.csv`, Figure 13.9b) and in the crossing at 20, 15, 12 and 10 N m;
- **delays the estimator's readings** by 0 to 5 updates of the controller, and by 2 and 5 with the compensation (`results/variants-delay.csv`, Figure 13.10).

For every crossing it writes the observations of `husky.metrics`, the components of the estimate's and the feedback's bounds that were not kept, the measurement's largest error at the period boundaries (the quantity of (S5)), the state's distance from its funnel and the episodes in which it left it, the alarm check's largest distance over the periods of normal operation and the boundaries at which the estimate or the state lay outside the initial set, and the drives' largest torque and the steps of the simulator at their limit; for a crossing with an alarm, the same before the alarm, the reasons for which no restart was offered, and what the robot did between the alarm and a contact and after it.

| File | Content |
|---|---|
| `variants-record.txt` | The record of the experiment of Section 13.3. |
| `variants-summary.txt` | Every value, by group; with `--check`, the comparison with `EXPECTED`, which holds every value but the versions of NumPy and Python. |
| `variants-friction.csv` | The stops on the wet grass at the map's label, by the grass's true friction: beyond the commanded point of rest and the estimate's error at rest from each speed, and the largest errors and distances over the four stops [mm, mm/s]. |
| `variants-torque.csv` | The stops on asphalt at its label, by the torque limit, the same. |
| `variants-delay.csv` | The crossing with the map every 0.1 s, on time, 40 and 100 ms late, and 100 ms late with the compensation: the true arc length less its estimate (`es_*`) and less the commanded one (`ds_*`) [mm]. |

The shipped results were produced on the second machine named above; the third machine named above reproduced them identically. As for Section 13.2, they were written with contract 1.0 of the core, and a run of the present code records 1.1 and the same values.

## The video of the crossing

`husky_video.py --clearpath DIR` draws the crossing with the map's labels, dry, of Section 13.2 as a video, with the package `../render`, whose README gives what it needs besides this chapter's requirements: Pillow, SciPy, EGL and ffmpeg; DIR is a checkout of Clearpath's `clearpath_common` at commit `78e3677`, the commit of `husky/description/`, for the A300's visual meshes, which the folder does not copy. The script runs the crossing `map` of `husky_route.py` again with the modules of Section 13.2 unchanged, records the simulated robot at 30 frames per second of simulated time from the end of the estimators' calibration at rest, where the crossing's clock starts (it copies the state after MuJoCo's steps and changes nothing in them), and draws the record: the robot with the visual meshes of Clearpath's description, its wheels turning as the simulated wheel joints did, on the yard of `husky/world.py`, the asphalt, the gravel and the wet grass where the simulation has its three grounds, stacks of pallets that fill the boxes of its obstacles beside the lane, the dock where it has the dock, and the lane's edges where the task has them; the field, the fence, the building with its door and canopy, and the cones are scenery. The drawing simulates nothing, and no result above changes. Before it draws, the script checks that the camera stays inside the fence, away from the obstacles, and sees the robot in every frame. It takes about forty minutes on two processes at 1920 x 1080 (`--quick` draws a small video, without stills).

```
.venv/bin/pip install pillow scipy
.venv/bin/python husky_video.py --clearpath DIR    # results/video/
```

It writes `results/video/husky-map.mp4` (34 s at 30 frames per second, 1920 x 1080, H.264; not in the repository's history, for its size, but attached to its releases), four stills `husky-map-<t>s.jpg` at 3.0, 11.5, 18.5 and 26.0 s (1600 x 900, Figure 13.7), and `husky-map-frames.csv`, each frame's time and the robot's position and orientation (a quaternion), the check that the drawing follows the record.

## In Isaac Sim

`isaac_husky.py` (version 5) runs in Isaac Sim 6.2.0 the diagnostics of Section 13.1 that apply to PhysX: the loads at rest, the spins and the turn against the Coulomb references, the slides, the creep, the stops, the spin commanded at 1 rad/s and the odometry. It imports the corrected URDF with Isaac Sim's URDF importer, gives the wheels PhysX velocity drives with the drivers' proportional gain, the same integral loop, the same drive controller and the same grounds, with the PhysX settings of Chapter 11. It first screens 19 setups on asphalt, each changing one thing (the wheels' colliders, the ground, the solver, the time step, the velocity iterations, the friction correlation distance, compliant contacts, and, as a test of the two-row law of PhysX's friction, the wheels' inertia about their axles), then runs three in full on the three grounds and repeats the default at 1, 0.5 and 0.25 ms. Its docstring describes the four earlier versions and their runs. It runs only inside Isaac Sim, on a computer with an NVIDIA RTX GPU:

```
<isaac-sim>/python.sh isaac_husky.py
python isaac_husky.py                       (pip install of isaacsim)
```

`results/isaac/` holds the run of 28 October 2026 (Isaac Sim 6.2.0-rc.14, omni.physx 110.4.4, Windows 11, an RTX 4060 Laptop GPU; 3996 s, no step failed): `isaac-summary.txt`, every diagnostic with its verdict; `record.txt`; `console.txt` and `errors.txt`; the URDF imported (`husky_a300.urdf`, mesh paths relative to `husky/description/clearpath_platform_description`); the session layers of the full runs (`stage-*.usda`); the traces of the full runs' diagnostics at every step (`trace-*.csv`: up to 2 s of each, the stops and slides to rest) and of the screen's drives (`trace-screen-*.csv`); and the imported asset (`usd/husky_a300_2`, the one this run wrote; `husky_a300` and `husky_a300_1` were left by earlier runs).

What it found, as Section 13.1 and Section S13.4 of the supplement report: the stops agree with MuJoCo's within 6.1 mm and the odometry is 1.46 to 1.58 % short (MuJoCo's 1.51 %), and the sideways slides come within 8 % of v^2/(2 mu g); but the turn commanded at a radius of 3 m comes out at 27.2 to 31.9 m under TGS, Isaac Sim's default solver, 8.6 to 10.0 times its Coulomb reference, and the robot goes almost straight under PGS. Under TGS, PhysX's patch friction weights a sliding contact's slip along and across the motion of the wheel's centre by the contact's effective mass in each direction (1.21 and 20.02 kg at this robot's wheels, without the link's constraint-force mixing), which predicts the direction of its friction to within 5.2 to 6.5 degrees (median) and turns of 25 to 26 m; raising the wheels' inertia 2, 8 and 32 times narrowed the turn to 15.5, 5.8 and 3.6 m, as predicted before the run. The governed runs of the chapter use MuJoCo; this run validates none of their turns.
