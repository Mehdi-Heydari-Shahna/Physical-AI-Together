# Chapter 11: simulator workflows

## Section 11.1: the cart in MuJoCo

This folder holds the first simulator adapter of Chapter 11: the cart of Chapter 3 built in MuJoCo, registered for the allovi core of Chapter 10 as the simulator adapter `cart.mujoco`, and the script `mujoco_cart.py`, which records the experiment and runs its diagnostics and comparisons. It needs MuJoCo 3.16.0 (`requirements.txt`) and Python 3.10 or later, and it imports the packages `allovi` and `cart` from `../ch10`, which in turn imports `../ch03/cart_example.py`.

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python mujoco_cart.py            # writes results/
.venv/bin/python mujoco_cart.py --check    # also checks every value quoted in Section 11.1
```

The script takes about twenty seconds.

`MUJOCO_GL=egl .venv/bin/python mujoco_stills.py` draws the two stills of the cart in Section 11.1 (MuJoCo 3.16.0, NumPy, Pillow and EGL; a few seconds). It places the model at two states of the exact run of Chapter 3 and simulates nothing; it needs `../render`.

## The package `cart_mujoco/`

| File | Content |
|---|---|
| `track.py` | The profile of the track (firm ground to 20 m, a crest of radius 10 m, a slope of 9.5 degrees) and the track coordinate: the arc length of the curve at height r above the ground, to the point of it below the front axle, plus the 0.1 m by which the bumper leads the axle. Beyond the crest, this curve is longer than the ground below it by r times the slope angle, 16.6 mm. |
| `scene.py` | `Design`, every parameter of the simulated cart and track, and `mjcf()`, which writes the MJCF text that MuJoCo compiles. The ground is one convex mesh, the prism under the profile, whose crest consists of 166 chords; the wall stands where the tool tip touches it at p + l = 25 m. |
| `sim.py` | `Sim`: the MuJoCo model and data, the state (p, v, l), placement of the cart at a state, and the drive loop, which realizes a commanded acceleration over one period of 0.1 s in 100 steps of 1 ms: a feedforward force and a speed feedback (gain 100 per second) on the wheels' torque motors, shared by the measured wheel loads. Each step is split into `mj_step1` and `mj_step2`, so that the loop reads the state for which it sets the controls. |
| `world.py` | `CartMuJoCo`, the simulator adapter `cart.mujoco`: it declares the state and the actuation of the cart's robot model, returns the simulator's values as exact rationals, and offers `admits` and `clearance` for the checks of Section 10.3. |

The options `Sim(read="after step")` and `Sim(coordinate="ground")`, and the variants `"wall in radians"` and `"speed servos"` of `mjcf()` and `Sim()`, are the mistakes that the diagnostics of Section S11.9.1 reveal; the variant `"welded wheels"` is the sliding cart of the same section. Nothing else uses them.

## What the script does

- **The record** of the experiment: versions, the settings of the simulator and those that differ from MuJoCo's defaults, the size of the model, the hash of its MJCF text, and the items that every supported experiment records (`results/record.txt`).
- **The diagnostics** of Section S11.9.1: the wheel loads against the weight; the distance of the wheels from the ground along the track, also where each axle reaches and leaves the crest; the rangefinder against the clearance, and with the wall's rotation written in radians where MuJoCo's compiler reads degrees; one period at constant speed, and with the state read after `mj_step`; the steps of each period, and with a loop on MuJoCo's clock; the speed over the crest, and with the ground coordinate; the accelerations that the drive loop sustains on the simulated ground, against the labels and the command bound; the contacts over the crest, braking at the label from 20 m, at constant speeds of 2.3 and 2.5 m/s and rolling freely, and the reference acceleration of the contacts at rest; a box that slides on a plane, and a cart with welded wheels that slides; and the encoders at time steps of 1, 0.5 and 0.25 ms, and with stiff speed servos, also with the Euler integrator.
- **The run of Chapter 3, period by period.** The configuration `chapter3` of `../ch10` is run with the exact track. Then, for each of its 220 periods, the cart is placed 0.05 s before the period starts, on the motion of the previous period, and driven under the previous law until the period starts; the period's law is executed in MuJoCo, and the result is compared with the model's nominal motion from the state at which the period began, at its end and within it. The labels of the exact run are compared with the braking of the simulated ground (`results/agreement-chapter3.csv`). The comparison is repeated at time steps of 0.5 and 0.25 ms, and over the crest with twice the speed gain of the drive loop; the tool's last approach to its stop is followed to time steps of 0.0625 ms with `implicitfast` and 0.125 ms with the Euler integrator.
- **The 26 cases of Section 10.4**, through `allovi.model.motion_agreement`, with tolerances of 0 and 1 mm.
- **The configuration of Chapter 3 with `cart.mujoco`**, through the core: the braking claims of the labels, alarms and physical violations, reported separately.

## Output (`results/`)

| File | Content |
|---|---|
| `record.txt` | The record of the experiment. |
| `cart.xml` | The MJCF text that MuJoCo compiled. |
| `agreement-chapter3.csv` | One row per period of the run of Chapter 3, at 1 ms: the exact state and inputs, the label and the braking of the simulated ground, the distance of the simulated start from the exact state, in the norm and in its parts (position and reach together; speed), the differences at the end of the period and their norm, the largest differences within the period, the largest reach, the realized acceleration less the commanded one, read from the adapter's speed signal and from the speed of the front axle along the ground, over 1 ms and over one step, and the largest jump of the speed signal against the axle's speed in one step. |
| `mujoco-summary.txt` | The values quoted in Section 11.1. |
| `video/cart-mujoco-operation.jpg`, `video/cart-mujoco-crest.jpg` | Two stills of the model at two states of the exact run of Chapter 3, drawn by `mujoco_stills.py` with `../render` (the figure of the stills in Section 11.1); nothing is simulated for them. |

## Numerical precision and reproducibility

MuJoCo computes in double precision and uses no random numbers here. The shipped results were identical, digit for digit, with MuJoCo 3.16.0 under CPython 3.11.15 (NumPy 2.4.6), 3.13.16 (NumPy 2.5.3) and 3.14.6 (NumPy 2.5.4) on one x86-64 Linux machine. Other processors, operating systems and versions of MuJoCo have not been tested; their floating-point results may differ in the last digits, and the check reports any value that differs at the precision quoted.

## Section 11.2: the cart in Isaac Sim

The same cart and track in Isaac Sim, built from the same parameters as a USD stage and simulated with PhysX, connected to the core as the simulator adapter `cart.isaac`, with the drive loop, diagnostics and comparisons of Section 11.1. The script `isaac_cart.py` runs only inside Isaac Sim, on a computer with an NVIDIA RTX GPU; its results for the book were produced once and are included, and `isaac_check.py` reads and checks them with nothing but Python:

```
python3 isaac_check.py            # writes results/isaac-derived.txt, results/isaac-periods.csv
                                  # and results/isaac-rounding-*.csv
python3 isaac_check.py --check    # also checks every value quoted in Section 11.2
```

### The package `cart_isaac/`

| File | Content |
|---|---|
| `design.py` | The design of Section 11.1, read from `cart_mujoco/scene.py` and `cart_mujoco/track.py` without importing MuJoCo, and `IsaacSettings`, the settings of Isaac Sim that the design does not fix: time step, solver, iterations, contact offsets, processor, damping, and the two auxiliary links of 1 g with which a PhysX articulation moves the chassis in its plane. |
| `stage.py` | The USD stage of the cart and its track, written with the schemas of USD Physics and PhysX from the design (the MJCF importer of Isaac Sim 6.2.0 cannot convert `results/cart.xml`, whose ground is a mesh given by its vertices), and the variants with the mistakes that the diagnostics reveal: the wall rotated in radians where USD reads degrees, a stage in centimetres, speed servos with their damping per degree and per radian, PhysX's defaults, and the cart with welded wheels. |
| `sim.py` | `IsaacCart`: the stage, the track coordinate, placement at a state and the drive loop of Section 11.1, line for line; physics stepped one step at a time through the simulation manager; the cart read and written through PhysX's tensor interface on the processor. The few calls that differ between Isaac Sim 5.1 and 6.x are in the class `Physics`. |
| `world.py` | `CartIsaac`, the simulator adapter `cart.isaac`, with the same declarations, oracle and clearance as `cart.mujoco`. |

### Running `isaac_cart.py`

It needs Isaac Sim 6.2.0, either the downloaded build, which brings its own Python, or the pip package in a Python 3.12 environment (`pip install isaacsim[all,extscache]==6.2.0.0 --extra-index-url https://pypi.nvidia.com`, after the PyTorch step of NVIDIA's installation page, and `OMNI_KIT_ACCEPT_EULA=YES` in the environment). From this folder:

```
python isaac_cart.py --quick 2>&1 | tee isaac-console-quick.txt   # without the runs at 0.5 and 0.25 ms
python isaac_cart.py --no-import 2>&1 | tee isaac-console.txt     # the full run, without the MJCF importer
```

(with the downloaded build, `<isaac-sim>/python.sh` on Linux or `<isaac-sim>\python.bat` on Windows in place of `python`). A step that fails is reported in `results/isaac/errors.txt` with its traceback, and the others go on. The script writes only into `results/isaac/`, which it replaces; it uses no rendered output (the application still loads its renderer) and runs PhysX on the processor. The full run took about fifteen minutes on the machine of the record. The docstring of `cart_isaac/__init__.py` refers to `README-isaac.md`, the README of the kit with which the code ran; its content is this part of the present README.

### The shipped results

| Path | Content |
|---|---|
| `results/isaac/` | The full run, with `--no-import`, as `isaac_cart.py` wrote it: `isaac-summary.txt` (every value), `record.txt`, `errors.txt` ("no step failed"), `cart.usda` (the stage, exported after the cart had settled at rest, with the poses and velocities that PhysX wrote back) and `agreement-chapter3-isaac.csv` (one row per period, as `results/agreement-chapter3.csv`); and `pip-freeze.txt`, the packages of the environment, listed after the run. |
| `results/isaac-quick/` | The run with `--quick`, in which the MJCF importer failed (`errors.txt`: "Mesh ground has no file"; its output folder was empty). |
| `results/isaac-logs/` | Everything Isaac Sim printed in the two runs (`console-full.txt`, `console-quick.txt`) and in the follow-up (`console-trace.txt`). |
| `results/isaac-trace/` | The follow-up, as `isaac_trace.py` wrote it (below). |
| `results/isaac-derived.txt` | What `isaac_check.py` derives from these results (and, with `--check`, its verdict). |
| `results/isaac-periods.csv` | One row per period for Figure S11.6: the end differences in both simulators, how far Isaac Sim's end position lies beyond its reported speed, and the reported speed of the front axle less the command, in both simulators. |
| `results/isaac-rounding-5m.csv`, `results/isaac-rounding-period49.csv` | One row per step for Figure S11.2, from the traces of the follow-up: the rate of the position over the step, the rate that rounding each of the 32 substeps' advances to single precision predicts, and the front wheel's encoder speed, each less the speed that PhysX reports after the step [mm/s]. |

The runs were made with the code of this folder unchanged, on Windows 11 with the pip package `isaacsim` 6.2.0.0 under CPython 3.12.15. The text files that the script wrote keep their Windows line endings, except the stages (`cart.usda`), which have Unix line endings; in the logs and in the quick run's `errors.txt` and summary, the user's folder in file paths reads `<user>`, and nothing else was changed.

### What `isaac_check.py` derives

- **The two runs**: every value that both report is the same, except the hash of the whole stage text; the quick run computes some values at one time step only, and those agree; the per-period tables are identical, byte for byte.
- **The stage**: the application writes its renderer's settings (the prims `/Render` and the override of `/OmniverseKit_Persp`) into the stage, and they differed between the runs; without them, the two stages are identical. The hash of the stage that Section 11.2 quotes (Table S11.2) is that of this text (SHA-256 beginning `b8df766ee0f7cf7b`); the summary and `record.txt` of each run hash the whole text (`e197534cfac9a87b` in the full run, `0e401f23395ee388` in the quick run).
- **The loop on the simulation clock**: the clock of Isaac Sim 6.2.0's simulation manager is PhysX's step count divided by the steps per second. With the count that the diagnostics before the loop imply (35,342), the clock's definition gives the 88 periods with 101 or 99 steps. This is a consistency check: it assumes that PhysX's count equals the manager's, and 200 of the start counts from 35,000 to 35,999 give the same pattern.
- **The tool's last approach**: the shortfall of the tool's reach at the end of period 201, from its equation of motion with the servo's saturation, integrated with RK4 (1.15 mm, the cart held or free to recoil) and with the semi-implicit Euler method at four time steps.
- **The reported speed against the position**: if the speed that PhysX reports were the rate of the position, a period's end position would lie within the band that the speed's largest and smallest differences from the nominal motion allow; in Isaac Sim it lies beyond that band by more than 0.01 mm in 41 of the 220 periods, in MuJoCo in none.
- **The run of Chapter 3 by phase**, and the environment of the run from the package list, the record, the console log and the stage.
- **The follow-up** (Section S11.6): that it reproduced the shipped comparison; the cart's advance per step in spacings of single precision, against one rounding per step and against the rounding of each substep's advance; the rate of the position against the reported speed and against that rounding, from 5 m and in period 49 on the firm ground, and in period 83 on the slope, where the horizontal slider's advance is rounded and the cart moves along the slope by that advance divided by the cosine of the slope; the bounds of the rounding by distance from the origin, number of substeps and time step; the encoders against both; each variant of the comparison (Table S11.5); and the loads at rest, rolling, braking and after a placement (a trace's first row reports the forces of the step before it, and after a placement it is left out).

### Known properties of this version of the script

- The hash that `isaac_cart.py` writes into its summary covers the whole stage text, renderer's settings included, and so differs between runs; use the one that `isaac_check.py` computes.
- The PhysX extension `omni.physx` 110.4.4 warns that the scene attribute `frictionType`, which the stage writes as `patch` (the default), is deprecated; the warning is in the logs and changes nothing.
- The bisection of the accelerations that the drive loop sustains accepts an acceleration if the error of the speed changes by no more than 1 mm/s after the first 0.1 s; in Isaac Sim the reported speed jumps by more than that from step to step, and the bisection does not measure the friction there (Section S11.10.3). The sled measures it.
- The adapter's speed signal is the velocity that PhysX reports, which its documentation says need not match the change of position between two steps (Section S11.10.3). For this cart, the follow-up traces the difference, on the firm ground and on the slope, to the rounding of the positions in single precision at each of the 32 substeps of TGS (Section S11.6); it did not trace the crest.
- The loads that the contact views report are the contact forces of single steps, which scatter from step to step about a rigid cart's loads; the diagnostic of the loads at rest reads one step (Section S11.6).

### The follow-up: `isaac_trace.py`

The diagnostics and the comparison of `isaac_cart.py` found a reported speed that is not the rate of the position, loads at rest above the weight and encoders whose reading changes with the time step, and their summaries could only bound these effects. `isaac_trace.py` records them step by step and repeats the comparison with variants. It needs the same Isaac Sim 6.2.0 and runs from this folder:

```
python isaac_trace.py 2>&1 | tee isaac-trace-console.txt   # about 8.5 minutes
python isaac_trace.py --traces-only                          # only the traces
python isaac_trace.py --variants "PGS solver,64 position iterations"
```

It writes only into `results/isaac-trace/`:

| File | Content |
|---|---|
| `summary.txt` | Every value, by group. |
| `trace-one-period-1ms-from-5m.csv` | One row per step: one period at 1 m/s from 5 m without a lead-in, as in the diagnostic of Section S11.10.3, and the two periods after it. |
| `trace-rest-firm-ground.csv`, `trace-rest-slope.csv` | The cart at rest at 5 m and at 24.5 m, each after standing for a second (200 and 500 steps). |
| `trace-braking-2-from-4.csv` | Braking at 2 m/s^2 from 4 m/s at 8 m (0.5 s). |
| `trace-period-83.csv`, `trace-period-49.csv` | Periods 83 and 49 of the run of Chapter 3, with their lead-in of 0.05 s. |
| `periods-*.csv` | The comparison of Section S11.10.4, one row per period, with the settings of Section 11.2 and with six variants: no velocity iteration, four velocity iterations, 64 position iterations, the PGS solver, a contact offset of 2 mm, and a drive loop that shares its force by the wheels' loads filtered over 10 ms. Two columns are new: how far the end position lies beyond the reported speed, and the largest difference between the reported speed and the rate of the position over ten steps. |
| `errors.txt` | The steps that failed ("no step failed"). |

Each row of a trace holds the position, the reported speed, the reach, the commands, the front axle's speed, the wheels' rates, the pitch, the wheels' loads, the least separation of each wheel's contacts and the height of its axle, and, but in the last row, the rate of the position over the step and the reported speed after it.

The shipped `results/isaac-trace/` is the follow-up's single run, on the laptop of the record, with the same release and the same packages as the runs of `isaac_cart.py` (`pip-freeze.txt`, identical to that of `results/isaac/`; `gpu-info.txt`, the output of `nvidia-smi` on that laptop). Its text files keep their Windows line endings; in its console log, `results/isaac-logs/console-trace.txt`, the user's folder in file paths reads `<user>`, and nothing else was changed. With the settings of Section 11.2, it reproduced the shipped comparison: every period, in the 19 columns that the two tables share, to the nine digits printed.
