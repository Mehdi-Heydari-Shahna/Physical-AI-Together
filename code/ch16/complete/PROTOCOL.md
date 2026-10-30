# Protocol of the complete case (Section 16.4)

Written on 29 October 2026, before any run of the identification, calibration, training, evaluation, real-time or stress phase below. The software was written and tested on the development worlds (seeds 42001 to 42050), with one exception: two smoke tests of the learner's training, before this file was written, ran 6 and 8 episodes in the first training worlds (seeds 47001 to 47008) under the provisional bounds and offset; their results were discarded, and nothing from them enters any phase. This file is part of the frozen record: `complete_identify.py` stores its SHA-256, with that of every module, in `FROZEN.json`, and `complete_evaluate.py`, `complete_realtime.py` and `complete_stress.py` refuse to run if any of them has changed. Any departure from this protocol is recorded, with its reason, in `DEVIATIONS.md`, which is not frozen. No external timestamp is supplied with this file; the hash in `FROZEN.json` and in the book identifies it.

## Question

Can one complete task of the cart, in a physics simulator, be executed under the full contract of the book: certified funnels whose assumptions are established on separate data, a terminal rest that is certified, fallible forecasts of capability and of demand together, two control packages crossed with several proposal methods, and execution against a clock that does not wait for the governor? The task is that of Chapter 16: reach the work zone (p >= 24 m), remain available there, prepare for the announced tool operation, and begin it when it is due.

## The world and the model

- **Simulator.** The MuJoCo cart of Section 11.1 (`../../ch11/cart_mujoco`, MuJoCo 3.16.0), every parameter of its design unchanged except the start of the ramp, which each world draws. The drive realizes a commanded acceleration as in Section 11.1.2, with a speed loop that integrates the command, and adds the force of the tool's servo; the tool moves to a commanded reach along a smooth path in 0.35 s (`plant.py`).
- **Sensor.** Position, speed and reach with independent Gaussian errors of standard deviations 0.25 mm, 0.5 mm/s and 0.1 mm, from seeded generators; within each period at every time step (the laws' feedback) and at the period boundaries (the governor).
- **Declared model** (`model.py`): at the simulator's step h = 1 ms, v[i+1] = v[i] + h (u[i] + w[i]) and p[i+1] = p[i] + h v[i+1] + d[i], with |w| <= w_bar and |d| <= d_bar while the command lies within the command bound and, towards the start, within the braking of the label in force; the reach within e_l of its command, except while the tool moves to a new command, which it reaches within four periods (the plant moves it along a smooth path in 0.35 s, which its servo follows without reaching its force limit; during development, the step command of Section 11.1, with which the servo unfolds the tool in about 70 ms at its force limit, was found to lift the wheels off the ground for a few milliseconds when the tool unfolded while the cart backed away); the tool's reaction may add an impulse J beyond w_bar over the periods in which it moves and the next; the sensor's errors within varpi_p, varpi_v and varpi_l.
- **Capabilities.** 'Level ground', 2 m/s^2: the front wheel's contact stays on the level ground before the crest (valid when the bumper stays at most 0.1 m beyond the crest's start). The global capability: 0.5 m/s^2 of braking everywhere, with the declared model, at speeds of at most V_C = 2 m/s (above about 2.4 m/s the wheels leave the crest, Section 11.1.3).

## Phases, in this order, and their seeds

| Phase | Seeds | Script | Output, frozen before the next phase |
|---|---|---|---|
| Identification | 43001 to 43040 | `complete_identify.py` | `FROZEN.json`: the bounds, and the hashes of the frozen modules and of this file |
| Calibration | 44001 to 44100 | `complete_calibrate.py` | `../results/complete-calibration.txt`: the survey's offset |
| Training | 47001 to 47300 | `complete_train.py` | `../results/complete-learner-<package>.json`: the learners' tables |
| Evaluation | 45001 to 45100 | `complete_evaluate.py` | `../results/complete-runs.csv`, `-summary.txt`, `-failures.csv` |
| Real time | 45001 to 45020 | `complete_realtime.py` | `../results/complete-realtime-*.csv`, `-summary.txt` |
| Stress | 46001 to 46030 | `complete_stress.py` | `../results/complete-stress.csv`, `-summary.txt` |

No seed of a later phase is used before that phase. The worlds of every phase are drawn as in the anticipation benchmark (`../bench/worlds.py`): the ramp's start uniform on [17, 21] m on a 1 cm grid; the survey unreliable with probability 0.2, its error uniform within 0.6 m if unreliable and within 0.2 m otherwise, on a 1 cm grid; the forecast onset of the operation uniform on 200 to 240 periods, the lead time uniform on 10 to 60 periods, and a slip of the true onset with probability 0.3, of 1 to 10 periods, later or earlier with probability 1/2. The survey's error and the onset's slip are drawn independently.

### Identification
In each identification world, one scripted excitation run (references accelerating and braking at up to 1.98 m/s^2 on level ground at up to 6.5 m/s, crossing the crest at up to 2.1 m/s with references between -0.48 and 0.2 m/s^2, braking on the slope at 0.48 m/s^2, holding, backing away at 0.45 m/s^2, braking a backward motion at 1.9 m/s^2, moving towards the wall on the slope, and unfolding the tool four times: at rest, while backing away, while stopping a backward motion and while moving towards the wall, folding it in between), and the configurations A-rule, A-preview and F-preview with both packages under the provisional bounds of `model.py`, 320 periods each. Each bound is 1.25 times the largest value observed over every time step of every run, rounded up to its grid: w_bar (periods outside the windows of the tool's impulse, a window being the periods in which the tool moves, to unfold or to fold, and the next; grid 0.005 m/s^2), d_bar (1e-7 m), J (the sum over the window of an unfolding of h max(0, |w| - w_bar); 0.0005 m/s), varpi_p, varpi_v, varpi_l (within the periods and at their boundaries; 0.0001), e_l (periods in which the tool does not move; 0.0001 m). The factor and the grids are fixed here. The bounds are empirical: the identification establishes them on data separate from the evaluation; it does not prove that every world satisfies them, and the evaluation records every departure.

### Calibration
The score of a world is its surveyed start less its true start. The survey predictor with offset q issues correct labels for every segment, period and candidate exactly when the score is at most q. The offset is the 96th smallest of the 100 calibration scores (Proposition 7.1, miscoverage 5 %). A world whose score is at most the offset is called valid.

### Training
For each package, one off-policy learner (`complete.learner`, the learner of Chapter 6 with its hyperparameters, the package's declared constants and the survey's labels), trained with seed 1 for one episode in each of the 300 training worlds, in order, under the frozen bounds and offset, without the operation: an episode is 200 periods, and its task is to reach the work zone. Its table is frozen after the 300th episode.

### Evaluation
Every configuration below runs in every evaluation world for 320 periods (32 s), offline through the core's loop (`allovi.run`), with the frozen bounds, offset and learners, every fresh candidate evaluated in every period:

| Configuration | Proposal method | Demand predictor (task plan) |
|---|---|---|
| A-rule-*p* | the naive rule of Chapter 3 | yes |
| A-preview-*p* | the preview planner (reads the task plan) | yes |
| A-learner-*p* | the trained learner | yes |
| F-preview-*p* | the preview planner (reads the task plan) | no: the filter |
| F-rule-*p* | the naive rule | no: the filter |

with *p* each of `complete.tube` and `complete.mpc`. Every configuration has the same survey predictor, task, operation (postponed by 21 periods after an overload), period, horizon and computation. A-rule and F-preview receive the same information about the operation, through the demand predictor or through the proposal method. The preview planner and the learner are the author's; a comparison with them is internal, not a comparison with a published method.

### Real time
Every configuration runs again in the first 20 evaluation worlds with the plant and the governor in separate processes (`realtime.py`), one run at a time on the machine (two processors), with no other work on it. The plant advances against `time.monotonic` and never waits: at each boundary it executes the plan committed for the period if it arrived before the boundary, and otherwise the shift of the plan in force; it checks the measurement against the initial set of that plan and applies the rest feedback on an alarm. The governor decides with an elapsed-time deadline 6 ms before each boundary, allowing 10 ms for one evaluation of the test, and a proposal not ready 30 ms after the boundary counts as missing.

### Stress tests
In the stress worlds, A-rule with both packages: S1 sliding friction 0.20 (both labels then overstate the braking); S2 an unmodelled payload of 30 kg; S3 a surveyed start 0.6 m beyond the true one in every world; S4 on the wall clock, in the first 10 stress worlds, proposals delayed by U[0, 0.15] s with probability 0.3 and every evaluation of the test delayed by 15 ms, with the allowance for one evaluation raised to 25 ms; S5 as S4 with the nominal allowance of 10 ms.

## Outcomes, reported separately

- **Task.** Arrival (the first boundary with p >= 24 m, true state); the operation begun on time (in the period of the true onset), late (after a postponement), or never; the time in the work zone.
- **Safety.** A violation (true p + l > 25 m at any time step); alarms and restarts; the least true clearance; every label issued, attached to the candidates of every period (evaluated or not, the fallback included), valid or not by the world's oracle, and every invalid executed label; every period with a departure from a frozen bound (w_bar outside the windows of the tool's impulse, J over each window, d_bar, the sensor's errors within the periods and at their boundaries, e_l in the periods in which the tool does not move), a true tracking error outside the funnel's level (V(e) / level > 1), or a clipped command.
- **Timing** (real time). For every period: when the measurement reached the governor, the time of the proposal and of the decision, the slack between sending the plan and the boundary, whether the plan arrived after the boundary (the fallback executed), whether the deadline cut the evaluation, whether a proposal was late; the plant's computation of the period and its lateness in publishing the measurement. A safe run with a late plan is a timing failure; a safe run without the operation is a task failure.

## Hypotheses and decision rules

- **H1** (the guarantee, valid worlds). In every valid world, no configuration has a violation, an alarm, a departure from a frozen bound, or an invalid executed label. Fails on any one.
- **H2** (no violation anywhere). No evaluation run has a violation, miscovered worlds included. Fails on any one. (In a miscovered world the guarantee is only the bound of Proposition 7.1.)
- **H3** (preparation). In every valid world with no early slip (slip >= 0) and lead time of at least N_back + 2 periods, every A-configuration and F-preview begins the operation on time. N_back is the number of periods that the package's declared back-away takes from the floor's rest point to the operation's target, as the preview planner computes it from the frozen bounds (`complete_evaluate.back_periods`). Fails on any one.
- **H4** (preparation does not depend on the proposal method). For each package, A-learner begins the operation on time in every valid world in which A-rule does. Fails on any one.
- **H5** (the information is needed). F-rule begins the operation on time in no valid world in which it arrived before the onset. Fails on any one.
- **H6** (timing). In the real-time runs, no plan arrives after its boundary, no evaluation is cut by the deadline, and every run repeats its offline run exactly, period by period (states, measurements and decisions). Fails on any one.
- **H7** (S4). No plan arrives after its boundary and no run has a violation. Fails on any one.
- **H8** (S5). No run has a violation or an alarm, whether plans arrive late or not. Fails on any one.

S1 to S3 have no hypothesis: they lie outside the declared model or the calibration, and their outcomes are reported as observed. The paired comparisons of 'on time' (A-rule against F-preview, A-preview against F-preview, A-learner against A-rule, A-rule against F-rule, and each configuration between the packages) are exploratory: counts of the worlds in which each or both were on time, with the exact McNemar test, which describes this finite suite and supports no claim of superiority or equivalence.

## What is reported in every case

Every run of every phase, in the CSV files; every failed run; the verdict of every hypothesis, as it falls; the identification's observed maxima beside the bounds; the calibration's scores; the learners' training logs; the record of versions and hashes. Nothing is excluded after the runs.
