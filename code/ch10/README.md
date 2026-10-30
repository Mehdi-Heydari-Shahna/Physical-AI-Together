# Chapter 10: the reference implementation

This folder holds the reference implementation of Chapter 10: the package `allovi`, the common core of the governor, the same for every robot, with its check suite and its robot-model adapter; the package `cart`, the modules of the cart of Chapter 3; templates of modules (`templates/`) and two example contributions (`examples/`); an adapter for an export of RoboCompiler, with such an export (`providers/`); the script `cart_allovi.py`, which runs the cart through the core and checks it against Chapter 8; the script `cart_checks.py`, which runs the checks of Section 10.3; and the script `model_providers.py`, which runs the checks of the robot-model adapter of Section 10.4. Everything needs Python 3.8 or later and no other packages.

## The core: `allovi/`

The core implements once what is common to every application: the step of the governor of Algorithm 8.1 (the fallback with its shifted profile, the evaluation of the fresh candidates in the order listed until the budget of the period is spent, admission, selection by rule (4.2) with its ties, and the restart search after an alarm, within the same budget), the closed loop with its clock, the checks of every output that it uses, the defaults for missing or unusable outputs, the alarm check, the registry of modules and the records. It never certifies a plan, never reads a proposal or a label and never changes a candidate; plans, segments, labels, initial sets, proposals, feedback laws and states are opaque to it. By default it does not measure time: a deadline is a budget of evaluations of the test. Version 1.1 of the contract adds two optional features, and runs that do not use them are unchanged: `governor.decide` and `governor.choose_restart` may take an elapsed-time deadline, a clock and an allowance for one evaluation (`deadline`, `clock`, `allowance`), and evaluate only while the clock plus the allowance lies before the deadline, reporting how many candidates they evaluated (`Decision.evaluated`) and whether the deadline cut the evaluation (`Decision.timed_out`); and a control package that sets `INITIAL_FROM_MEASUREMENT = True` receives the measurement and its bound in `initial_set(plan, measured, bound)`. The wall-clock loop that uses the deadline is `../ch16/complete/realtime.py`.

| File | Content |
|---|---|
| `contract.py` | The interfaces: the records of one period (`Observation`, `Requirement`, `Step`, `Candidate`); the protocols of the objects that the control package supplies (plans, the level space, the capability space) and of the eight roles of modules; the methods that the registry requires of each role; and the version of the contract. |
| `schema.py` | Declarations of units, frames, the norm and actuator semantics (`Quantity`, `Schema`, `Bound`, `Actuation`), and their comparison. |
| `levels.py` | Profiles (`Profile`) and the scalar level space (`ScalarLevels`). Vector reserves (Section 4.2.7) are not supported. |
| `governor.py` | The decision of one period (`decide`), the selection rule (`select`), the restart search (`choose_restart`), the checks of the outputs of the modules, and the treatment of a module that fails (`Guard`). |
| `loop.py` | The closed loop (`run`): the order of the calls within a period, defaults, commitment, the alarm check, the rest feedback and the restart. |
| `registry.py` | Registration of modules (`register`, `Registry`) and assembly of configurations (`build`), with their checks. |
| `records.py` | One row per period and one per candidate, and their CSV files. |
| `checks.py` | The check suite of Section 10.3 (`check`): interface checks, consistency checks, falsification checks against an oracle in simulation, and the checks that modules supply; each check reports passed, failed or not applicable, with the number of cases that it examined. A passed check discharges no obligation of Table 4.1. |
| `model.py` | The robot-model adapter of Section 10.4: the record `RobotModel` (coordinates, kinematics, nominal dynamics, actuation, norm, hold and meaning), the status of each item (provided, elsewhere or missing), and the checks `check_model`, `agreement` and `motion_agreement`. |

## The modules of the cart: `cart/`

| File | Content |
|---|---|
| `model.py` | The model of Chapter 3, imported unchanged from `../ch03/cart_example.py`; the declarations of the state, the actuation and the levels of the cart; and the robot model `MODEL`, whose state and actuation the robot adapter, the simulator adapter and the control package declare. |
| `control.py` | The control package `cart.constant-command`: segments of one constant command, the generator and the test of Chapter 3, the rest segment, and the rest feedback of Section 8.3.4 with its reach bound; braking maximally (default) or at the global capability. Its feedback laws (`Law`) can be called with the time and the state, as every law of the contract; the track uses their fields. |
| `modules.py` | The proposal method `cart.naive`; the capability predictors `cart.fixed`, `cart.reactive`, `cart.forecasting`, `cart.persisting` and `cart.survey`; the task `cart.operation` (the tool operation) and the demand predictor `cart.task-plan`; the robot adapter `cart.robot`; the simulator adapter `cart.track`; and the evaluator `cart.metrics`. The track realizes the rest feedback only while it brakes a forward motion on the ramp, as after every alarm of the book; elsewhere it raises `NotImplementedError`. |
| `__init__.py` | Registers the modules and assembles the configuration of one run (`configuration`). |

For the checks of Section 10.3, the control package provides `head(plan)` and a check of its own test (`checks()`), and the track serves as an oracle: `admits(label, segment, k)` and `clearance(state)`.

## Templates, examples and model providers

`templates/` holds one template for each of the eight roles, and one for a robot model; each states the obligations, which no check discharges, and the checks that are run on it (see `templates/README.md` for the path of a contribution). `templates/HELPERS.md` documents the helpers of the cart's code that a contribution may reuse beyond the templates, with examples that run as tests. `examples/eager.py` is a proposal method that asks to accelerate in every period; `examples/survey_margin.py` is a capability predictor, the survey of Section 7.2.3 used with a margin. Both are written from the public interface alone.

`providers/robocompiler.py` builds a robot model from the files that RoboCompiler's compile step writes, reading them with the standard library and calling none of RoboCompiler's code. `providers/robocompiler-stewart/` holds those files for RoboCompiler's six-UPS Stewart platform, unchanged, with the command that produced them, the commit of RoboCompiler and its licence (Apache 2.0; see its `NOTICE.md`).

## What the scripts do

`cart_allovi.py` runs the following.

- **The thirteen runs of Chapter 8.** The forecasting run of Chapter 3; the three ways of meeting the operation of Section 8.2.3 and the two erroneous forecasts of Section 8.2.4; the three time budgets of Section 8.3.3; and the four alarm runs of Section 8.3.4. For each, the core writes the decision file and the trace file in the format of Chapter 8, and the script compares them, byte for byte, with those in `../ch08/results/`. The runs use strict checking (`strict=True`) and the implementation check (`diagnostics=True`).
- **Refusals (Section 10.2).** A control package without the method `restart`, refused when it is registered; a simulator adapter that declares its commanded acceleration in mm/s^2, and a capability predictor that issues friction coefficients, both refused when the configuration is built.
- **Defaults (Section 10.2).** A capability predictor whose labels are never ready: with strict checking, the run stops in period 0; in operation (`strict=False`), the global capability replaces the labels in every period, and the run is that of the fixed predictor.
- **The numerics of a simulator adapter (Section 10.1).** The four alarm runs with the simulator adapter as first written, which kept the floating-point state at the end of the period of the alarm, compared with Chapter 8; and the exact states of the four alarm runs against those of `../ch08/cart_governor.py`, which the script imports.
- **The interfaces (Section 10.2).** The numbers of roles, methods and records, and the version of the contract.

`model_providers.py` runs the checks of the robot-model adapter (Section 10.4) on the cart's model; on a simulator adapter, defined in the script, that declares metres and reports millimetres; and on the export of RoboCompiler in `providers/`, reporting which items of the boundary it provides.

`cart_checks.py` runs the check suite (Section 10.3) on the configuration of the forecasting run of Chapter 3 with the cart's modules; with the survey predictor biased by 0.2 m; with two broken control packages, defined in the script, whose tests forget the tool or accept braking 0.01 m/s^2 beyond the label, each with the naive and the eager proposal method; and with the two examples. It also checks that every template still has every method of its role, and runs the path of a contribution on both examples. With `--module FILE`, it runs the checks on every module that FILE registers, each in its role in the reference configuration, except a task, which the cart's demand predictor and evaluator cannot take.

## Run

```
python3 cart_allovi.py              # writes results/
python3 cart_allovi.py --check      # also checks every number and message quoted in Sections 10.1 and 10.2
python3 cart_checks.py              # writes results/checks-*.txt
python3 cart_checks.py --check      # also checks every number quoted in Section 10.3
python3 cart_checks.py --module FILE   # the checks of the modules that FILE registers
python3 model_providers.py          # writes results/models-summary.txt
python3 model_providers.py --check  # also checks every number quoted in Section 10.4
```

`cart_allovi.py` and `cart_checks.py` take about a minute each with CPython 3.13 or 3.14; with CPython 3.8, about two minutes and two and a half. `model_providers.py` takes a few seconds. `cart_allovi.py` needs `../ch08/cart_governor.py` and its results, which are included.

## Output (`results/`)

| File | Content |
|---|---|
| `summary.txt` | The comparison with Chapter 8, and the numbers and messages quoted in Sections 10.1 and 10.2. |
| `decisions-<run>.csv`, `trace-<run>.csv` | The records of the thirteen runs, written by the core, with the names and columns of the files of Chapter 8. |
| `checks-<case>.txt` | The report of the check suite for one case of Section 10.3. |
| `checks-summary.txt` | The numbers quoted in Section 10.3. |
| `models-summary.txt` | The numbers and the message quoted in Section 10.4. |

## Numerical precision and reproducibility

Every planned and certified quantity is computed with exact rational arithmetic, as in Chapters 3 and 8. In the period of each alarm, the head of the plan brakes harder than the ground allows, and the motion of that period is computed in floating point by the function of Chapter 3; only the four alarm runs contain such a period, one each. The simulator adapter keeps the velocity as computed and corrects the position by its rounding error, which it checks is below 1e-9 m, so that the point of rest that the state implies is exact. The script of Chapter 8 instead keeps the floating-point state and sets the exact point of rest when the rest feedback brings the cart to a stop; the records of the two scripts agree in every printed digit, and their exact positions differ, in the alarm runs, by at most 1.8e-15 m. No random numbers are used. The shipped results were identical, bit for bit, in runs with CPython 3.8.20, 3.13.16 and 3.14.6 on one x86-64 Linux machine; other processors, operating systems and versions have not been tested.
