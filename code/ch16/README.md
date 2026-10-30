# Chapter 16: comparisons that identify the contribution

Two benchmarks, each with its hypotheses written down before its first full run (`HYPOTHESES.md`), and the report of their results. `cart_benchmark.py` runs the anticipation benchmark of Sections 16.1 to 16.3: the cart's tool operation in a family of 100 worlds under fifteen configurations, among them the structural reading of a predictive safety filter, fixed conservatism, reactive preparation, ALLOVI with realistic forecasts, its two ablations, three oracle diagnostics, a model-based preview planner, and a reduced time budget. `cart_substitution.py` runs the substitution grid of Section 16.2: the four control packages of Section 5.3 crossed with the six proposal methods of Chapter 6, each learner trained with the package it is evaluated with. `cart_diagnosis.py`, written after the first run of both, reruns the worlds and pairings behind the hypotheses that failed and reports what they did (Sections 16.2.2, 16.2.4 and 16.3.3); it is a diagnosis, not a test, and changes nothing in the other two scripts.

```
python3 cart_benchmark.py --check       # results/benchmark-*
python3 cart_substitution.py --check    # results/substitution-*
python3 cart_diagnosis.py --check       # results/diagnosis-summary.txt
```

Python 3.8 or later; no other packages. `cart_benchmark.py` imports the core and the cart of Chapter 10 (`../ch10`), and the calibration of Section 7.3 (`../ch07/cart_forecasts.py`), which recomputes the survey's offset, 0.40 m, from its 100 calibration worlds; `cart_substitution.py` imports the protocol of Chapter 6 (`../ch06/cart_learning.py`) and, through it, the control packages of Chapter 5. With `--check`, each script compares every value of its summary with its `EXPECTED` and exits with an error if one differs; `cart_benchmark.py` does not compare the wall-clock times of the decisions, which belong to the machine. All three use up to two processes (`--jobs`); their results do not depend on the number. On the machine that wrote the shipped results (x86_64, two processors, CPython 3.13.17), `cart_benchmark.py` takes 14 to 17 minutes with two processes, `cart_substitution.py` about 40 minutes with one, and `cart_diagnosis.py` about 10 minutes with one.

To run one configuration of the benchmark in one world, for example C11 in world 16001, see section 12 of `../ch10/templates/HELPERS.md`, which gives the call and what it prints.

## The package `bench/`

| File | Content |
|---|---|
| `worlds.py` | The family of worlds of the anticipation benchmark: seeds 16001 to 16100; the ramp's start, the survey's reliability and error (the condition "as calibrated" of Section 7.3), the forecast onset, the lead time and the slip of the true onset, drawn in a fixed order from each seed. |
| `modules.py` | The control package `bench.constant-command`: `cart.constant-command` of Chapter 10 with one more proposal, `retreat`, a command of -0.5 m/s^2; with the proposals of Chapter 3 it is `cart.constant-command`, as the script checks against Chapter 8. The preview planner `bench.preview`, a model-based proposal method that reads the task plan and plans its own back-away, exactly or a number of periods late. The evaluator `bench.metrics`: the task's outcomes, the interventions (by ambition and by the executed command), the executed labels' validity from the world's oracle, alarms, restarts, the true clearance, the test's evaluations. A wrapper that times each decision (`Clock`). |
| `stats.py` | Exact intervals: the two-sided interval of Clopper and Pearson for a proportion, by bisection in exact arithmetic, and the distribution-free interval of order statistics for a median. |
| `hypotheses.py` | The decision rules of hypotheses A1 to A8, as `HYPOTHESES.md` states them. |

## What the scripts do

`cart_benchmark.py`

- checks that `bench.constant-command` with the proposals of Chapter 3 repeats, byte for byte, the decision file of the forecast-informed run of Section 8.2 that the core of Chapter 10 wrote (`../ch10/results/decisions-forecast-informed.csv`);
- runs the fifteen configurations of `HYPOTHESES.md` in the 100 worlds, 320 periods each;
- applies the decision rules of A1 to A8;
- reports each configuration over the valid worlds (those whose survey places the ramp no more than the offset after its true start): the operation on time, late or never begun, the arrival and the time in the work zone, the interventions; and over all worlds the invalid executed labels, the alarms, the restarts, the violations, the least true clearance and the test's evaluations; and every failed run;
- writes the data of Figure 16.2.

`cart_substitution.py`

- checks that its runs with the tube package, and the off-policy learner's with the contraction-metric package, are those of Chapter 6, run for run (`../ch06/results/evaluation.csv`);
- runs every pairing of the four packages and the six methods, training each learner with its package, and applies the decision rules of B1 to B4;
- reports every pairing and its training runs.

## Output (`results/`)

| File | Content |
|---|---|
| `benchmark-summary.txt` | Every value of the anticipation benchmark, by group, the verdicts of A1 to A8, and the decisions' wall-clock times on this machine (not checked). |
| `benchmark-summary-run3.txt` | The summary of the third run, kept for its decision times, which Section 16.3.2 compares with the fourth run's; its values are those of `benchmark-summary.txt`, under the heading that the third run still had (see `HYPOTHESES.md`). |
| `benchmark-runs.csv` | One row per configuration and world: the world's draws, the outcomes, the interventions, the evaluations, the clearance, the labels, the alarms; Figure S16.1 plots ALLOVI's outcomes against lead time and slip. |
| `benchmark-failures.csv` | Every failed run: an invalid executed label, an alarm, a violation, an operation late or never begun, or a cart never in the work zone. |
| `benchmark-traces.csv` | Figure 16.2: in the valid worlds with no slip and a lead time of at least 27 periods, the cart's position about the true onset, every 0.1 s: median, least and largest, for five configurations. |
| `benchmark-record.txt` | The record of the experiment. |
| `substitution-summary.txt` | Every value of the substitution grid, by group, and the verdicts of B1 to B4. |
| `substitution-evaluation.csv` | One row per evaluation run, with the columns of Chapter 6's `evaluation.csv`. |
| `substitution-cells.csv` | One row per pairing: its mean arrival, earliest and latest, and mean intervention (Figure 16.1). |
| `diagnosis-summary.txt` | What the runs behind A5, A6 and B2 to B4 did, by group, and the worlds that are not valid (written after the first run; checked by `cart_diagnosis.py --check`). |

## Numerical precision and reproducibility

The cart's motion and every certified quantity are exact rational numbers, as in Chapters 3, 8 and 10; the substitution grid computes as Chapter 6 does, and repeats Chapter 6's runs exactly where they overlap. Random numbers come only from `random.Random(seed).random()` with integer seeds, as in Chapters 6 and 7. The wall-clock times of `cart_benchmark.py` are measured with `time.perf_counter` and depend on the machine and its load; they never enter a decision.

## The complete case (Sections 11.4 and 16.4)

The cart in MuJoCo under the full contract: a declared model at the simulator's step with bounded disturbances, whose bounds are identified on separate worlds and frozen; two control packages with Lyapunov tubes checked in exact arithmetic; a survey predictor calibrated on separate worlds; five configurations (A-rule, A-preview, A-learner, F-preview, F-rule) with each package; and a real-time phase in which the plant runs on the wall clock and never waits for the governor. The protocol, `complete/PROTOCOL.md`, was written before any run; `complete/FROZEN.json` holds the bounds and the SHA-256 of every frozen file, the protocol included; `complete/DEVIATIONS.md` records every departure.

Requires Python 3.13 (3.10 or later should do), MuJoCo 3.16.0 and NumPy, and the cart of `../ch11/cart_mujoco`. Run the phases in this order; the evaluation, real-time, stress and diagnosis scripts refuse to run if a frozen file has changed:

```
python3 complete_identify.py            # 40 worlds; writes complete/FROZEN.json and results/complete-identification.*
python3 complete_calibrate.py           # 100 worlds; results/complete-calibration.txt (the offset)
python3 complete_train.py               # 300 worlds per package; results/complete-learner-*.json, -training-*.csv
python3 complete_evaluate.py            # 100 worlds, 1000 runs; results/complete-summary.txt, -runs.csv, -failures.csv
python3 complete_realtime.py --part 1   # the wall clock, first 10 worlds (one run at a time, nothing else running)
python3 complete_realtime.py --part 2   # the last 10 worlds
python3 complete_realtime.py --merge    # results/complete-realtime-summary.txt, -runs.csv, -periods.csv
python3 complete_stress.py              # S1 to S5; results/complete-stress-summary.txt, -stress.csv
python3 complete_diagnose.py            # exploratory, after the runs (about 3 minutes): results/complete-diagnosis.txt
python3 complete_figure.py              # prints the coordinates of Figure 16.4 from results/complete-runs.csv
```

Each script with `--worlds n` (or `--episodes n` for the training) runs a smoke test on development worlds with provisional values and writes nothing. On the machine that wrote the shipped results (x86_64, two processors, CPython 3.13.17), the identification took about 15 minutes, the training about 15, the evaluation 35 with two processes, each half of the real-time phase about an hour, and the stress tests about half an hour.

| File | Content |
|---|---|
| `complete/model.py` | The declared model, the bounds (provisional and frozen) and the certificates of the two packages, with the exact check of the Lyapunov inequality. |
| `complete/plant.py` | The MuJoCo cart executing feedback laws at every 1 ms step: the drive loop, the tool's smooth path, the modelled sensor. |
| `complete/packages.py` | The packages `complete.tube` and `complete.mpc`: segments, funnels, rest, the generator and the test. |
| `complete/modules.py` | The world adapter with the labels' oracle, the robot adapter, the survey predictor, the naive rule, the preview planner, the evaluator and the trained learner. |
| `complete/worlds.py` | The phases and their seeds. |
| `complete/config.py` | The ten configurations. |
| `complete/realtime.py` | The plant and the governor in two processes, on `time.monotonic`. |

| Output (`results/`) | Content |
|---|---|
| `complete-identification.txt`, `.csv` | The largest values observed and the bounds frozen from them; one row per identification run. |
| `complete-calibration.txt` | The calibration's scores and the offset. |
| `complete-learner-<pkg>.json`, `complete-training-<pkg>.csv` | The trained tables and the training log. |
| `complete-summary.txt` | The counts by configuration, the verdicts of H1 to H5, the exploratory paired comparisons and the record of the run (the line "periods in which the tool unfolds" of `complete-identification.txt` counts unfoldings, see `complete/DEVIATIONS.md`). |
| `complete-runs.csv`, `complete-failures.csv` | One row per run; every run with a violation, an alarm, a departure, an invalid executed label, a late or missing operation. |
| `complete-realtime-*.csv`, `complete-realtime-summary.txt` | One row per real-time run and per period, the timing distributions and the verdict of H6. |
| `complete-stress.csv`, `complete-stress-summary.txt` | The stress tests S1 to S5 and the verdicts of H7 and H8. |
| `complete-diagnosis.txt` | The exploratory diagnosis, written after the runs: H3, H4 and the paired differences (offline reruns), the real-time periods whose plan was late or whose evaluation was cut (from the records), and the stress test S1 (offline reruns). |
