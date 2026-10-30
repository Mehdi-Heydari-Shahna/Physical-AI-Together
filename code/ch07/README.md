# Chapter 7: capability and demand predictors for the cart

`cart_forecasts.py` computes every number of Chapter 7, Sections 7.1 to 7.3. It needs Python 3.8 or later and no other packages. It imports the cart, its control method, the governor, the proposal method and the operation from `../ch03/cart_example.py` and uses them unchanged. The governed loop of Chapter 3 is repeated with the predictors as objects, and the script also checks that, with the world, the operation and the predictors of Chapter 3, it reproduces every row of the runs of Chapter 3 (`same_as_chapter_3`).

## Capability predictors (Section 7.1)

A label is a braking β on the positions that a segment covers, for the period in which the segment would be executed; beyond the forecast horizon it is the global capability, 0.5 m/s².

| Predictor | Label of a segment | Provenance and coverage claim |
|---|---|---|
| `Fixed` | 0.5 m/s² | Declared; valid in every period |
| `Reactive` | the braking measured on the segment's region if the cart has traversed all of it, else 0.5 m/s² | Measured; valid if the measurement is exact and the ground does not change |
| `Survey` | 2 m/s² if the region ends before ŝ, the surveyed ramp start moved 0.2 m towards the start, else 0.5 m/s² | Bounded error; valid if the survey error is at most 0.2 m and the ground has not changed since the survey |
| `Map` | the smallest braking of the true map on the region | Exact, in simulation only (the forecasting predictor of Chapters 3 to 6) |
| `Persisting` | the braking measured at the cart's position | Extrapolated; no claim |

**Worlds.** 50 worlds, seeds 1 to 50. Each draws the ramp start from [17, 21] m on a 1 cm grid, as in the world family of Chapter 6, and the survey error from [−0.20, 0.20] m on a 1 cm grid, within the survey's stated accuracy. Each predictor runs the task of Chapter 3, with its operation, in every world.

**Validation records.** The simulation knows the realized world, so it checks every label attached to a candidate pair, which a real predictor cannot do; the labels of trial segments that the generator discards are not counted. For every executed label that claims more than the global capability, it records the region, the label, the executed command, whether the command used more braking than the global capability, the braking that the realized world allows on the region, whether the label was valid, whether the label error showed in the motion, and whether an alarm followed. A segment that reaches the ramp only at the end of its period carries an invalid label whose overstated braking never acts: its motion is the planned one, exactly, and no alarm is raised (this happens in two runs of Section 7.3). A governed run stops at the first alarm, as in Chapter 3. The function of Chapter 3 then calculates, from the measured state at the alarm, where maximal braking brings the cart to rest; on the ramp, where every alarm occurs (the script checks this), it decelerates at the global capability, 0.5 m/s². While the cart moves, no single-trajectory funnel contains every state that maximal braking can reach in the worlds consistent with the global capability, so a restart can be certified only once the cart is at rest, by a plan that holds it there: exactly when it stops at or before the floor, 24.5 m. The restarted loop and the rest of the task are not simulated.

## Demand predictor (Section 7.2)

The operation of Chapter 3, the tool unfolded by 0.8 m, consumes 0.8 m of clearance in p + l ≤ 25 m, so it needs the floor plus 0.8 m, 1.3 m, from its onset period, 200, on; it is announced in period 140. The demand predictor issues the requirement from its announcement until the operation begins; once its predicted onset has passed, it requires its level from pair 0 on. The operation begins at its onset if the plan in force then holds 1.3 m; otherwise an overload is recorded and the operation does not take place, as in Chapter 3.

**Injected errors.** In the world of Chapter 3, with the map predictor: lead times of 3, 2.1, 2, 1 and 0 s; levels of 1.2, 1.25, 1.4 and 1.6 m; predicted onsets 1 s and 0.5 s early, 0.1 s and 0.5 s late. For comparison, the survey predictor without offset, as a point estimate, with surveys biased by −1, −0.5, −0.2, 0, 0.05, 0.1, 0.2, 0.3, 0.35, 0.5 and 1 m; a negative bias places the ramp start too early and leaves every label valid.

## Calibration and validation (Section 7.3)

**Worlds.** The ramp start is drawn as in Section 7.1, so the world with seed i has the same ramp start in both families. The survey is unreliable with probability 0.2, its error drawn from [−0.60, 0.60] m, and precise otherwise, its error drawn from [−0.20, 0.20] m, both on a 1 cm grid; no record says which. The score of a world is the surveyed start minus the true start; the labelling map of a survey rule with offset q is correct for every segment exactly when the score is at most q.

**Rules.** Four rules label as the survey predictor does, with offsets 0 (point estimate), 0.2 m (the stated accuracy as a bound), the calibrated offset, and 0.6 m (the worst-case bound).

**Calibration.** Split conformal prediction on 100 calibration worlds (seeds 1001 to 1100): the offset is the m-th smallest score, m = ⌈101 · 0.95⌉ = 96. The script also computes an upper bound on the probability that the miscoverage given the calibration set exceeds 5 % and 10 % (exact for continuous scores; these scores have ties), and the index that keeps it at most 5 % with probability at least 95 %.

**Held-out evaluation.** 2000 worlds (seeds 2001 to 4000), used for nothing else, under three conditions: as calibrated; after a shift to half unreliable surveys; and stale, with the ground softened over 0.5 m before the ramp since the survey. Each rule's failures are counted without running the robot, because they do not depend on the proposal method, with exact one-sided 95 % upper confidence bounds (Clopper and Pearson).

**Runs.** In the 50 worlds with seeds 1 to 50: the four rules as calibrated, the calibrated rule after the shift and in stale worlds, and the worst-case bound in stale worlds, each with the task and the operation of Chapter 3; and the point estimate as calibrated with two proposal methods, that of Chapter 3 and a cautious one that requests 1.5 m/s until the work zone, both without the operation, until 30 s.

## Run

```
python3 cart_forecasts.py            # all runs; writes results/
python3 cart_forecasts.py --check    # also checks every number quoted in the text
python3 cart_forecasts.py --jobs 1   # one process; the default uses up to two
```

A full run takes about eleven minutes with two processes. The results do not depend on the number of processes.

## Output (`results/`)

| File | Content |
|---|---|
| `summary.txt` | The numbers quoted in Chapter 7. |
| `runs.csv` | One row per run in the family of worlds: predictor, world, ramp start, survey error, ŝ, arrival, operation performed, labels attached to candidate pairs (all, above the global capability, invalid), executed labels above the global capability and invalid, alarm, certified restart, stop, violation, lowest level. |
| `validation.csv` | One validation record per executed label above the global capability, for the survey and persisting predictors. |
| `demand.csv` | One row per run with an injected demand error. |
| `survey-bias.csv` | One row per run with an injected survey error. |
| `calibration.csv` | The 100 calibration worlds of Section 7.3: ramp start, surveyed start, survey error, whether the survey was unreliable, score; Figure 7.2 counts its miscoverage. |
| `held-out.csv` | The 2000 held-out worlds under each of the three conditions; Figure 7.2 counts their miscoverage. |
| `rules.csv` | One row per run of Section 7.3 with the operation: condition, rule, offset, world, score, labels, alarm, certified restart, stop, violation. |
| `loop.csv` | One row per run of the two proposal methods: labels attached to candidate pairs, executed labels above the global capability and how many were invalid and used, alarm, stop, violation; Figure S7.1 plots it. |

## Numerical precision and reproducibility

Every planned and certified quantity is computed with exact rational arithmetic, as in Chapter 3, and so is the realized motion whenever the executed label is valid. Floating point is used only for motion under an invalid label and after an alarm, through the functions of Chapter 3, which use square roots, and for the bound 1 − 0.05^(1/50) on clean worlds, which uses the platform's power function and is printed to one decimal. The calibration and the probabilities of Section 7.3 are computed exactly, with integers and fractions, and its confidence bounds by bisection in integer arithmetic, never below the exact value. Random numbers come only from `random.Random(seed).random()` with integer seeds, as in Chapter 6; Python promises only that `random()` keeps producing the same sequence when the compatible seeder is given the same seed, and the script relies on integer seeding being that seeder. Means are computed exactly and rounded half up. The shipped results were identical, bit for bit, in runs with CPython 3.8.20, 3.13.16 and 3.14.6 on one x86-64 Linux machine; other processors, operating systems and versions have not been tested.
