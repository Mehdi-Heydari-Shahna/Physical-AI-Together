# Chapter 8: the governor of the cart, step by step

`cart_governor.py` computes every number of Chapter 8, Sections 8.1 to 8.3. It needs Python 3.8 or later and no other packages. It imports the cart, its control method (segments, generator and test), the capability predictors, the proposal method and the operation from `../ch03/cart_example.py` and uses them unchanged. The common part of the governor (fallback, profiles, admission, selection, commitment, the alarm check and the order of the response to an alarm) is written out in the script, with the reason for every outcome. The survey predictor of Section 7.2.3 and the rest feedback, maximal braking, are restated in the script; for one consistency check, it also imports `../ch07/cart_forecasts.py`. With the configuration of Chapter 3, the script reproduces every row of the runs of Chapter 3 with the fixed, reactive and forecasting capability predictors (`same_as_chapter_3`), selects in every period the candidate that the selection function of Chapter 3 selects, and checks in every period that evaluating the fallback would give its shifted profile.

## Decision trace (Section 8.1)

For every candidate of every period, fallback included (index 0), the trace records its ambition, its head and the label of the head, its number of pairs, its point of rest, its level and its onset level, each with its reason, whether it is admitted, its two keys in the selection rule, and its outcome. The reason for the level is the first pair, from pair 0 on, that the test does not certify, with the condition that it fails, or else the pair or point of rest with the least clearance, which may lie below the floor; the reason for the onset level is found in the same way from the onset pair on. For every period, the trace records the state, the initial set, the proposal, the requirement, the committed candidate and the reason for the choice. Section 8.1 reads the trace of the forecasting run of Chapter 3, 220 periods.

## Five ways to meet the operation (Section 8.2)

The operation of Chapter 3 needs 1.3 m of reserve from its onset period, 200, on. Five configurations run the cart for 25 s with the forecasting capability predictor; they differ only in what the demand predictor issues. Here "reactive" and "fixed" name ways of meeting the operation, not the capability predictors of Chapter 3.

| Configuration | Requirement |
|---|---|
| `forecast-informed` | from period 140, for the onset in period 200, as in Chapter 3 |
| `reactive-preparation` | nothing before the operation is due |
| `fixed-conservatism` | 1.3 m from the first pair on, in every period until the operation begins |
| `forecast-early` | from period 140, for an onset forecast 1 s early, period 190 |
| `forecast-late` | from period 140, for an onset forecast 1 s late, period 210 |

Once a forecast onset has passed, the demand predictor requires the reserve from the first pair on. If the operation is due and the plan in force does not hold its reserve, an overload is recorded and the operation is postponed: to the forecast onset, if one was forecast and it is still ahead, and otherwise by 21 periods, announced for the new onset. In Chapter 3, such an operation is dropped instead. Twenty-one periods is the shortest postponement that this generator can meet from rest at 24.5 m; the script checks that 20 periods are not enough. The script also checks that the back-away of the reactive-preparation, forecast-early and forecast-late runs is that of the forecast-informed run, shifted in time.

## The governor at run time (Section 8.3)

Algorithm 8.1 adds to the step of the governor its deadline, its commitment and its response to a label alarm. The runs use the configuration of Chapter 3: the forecasting capability predictor, and the operation announced in period 140 for period 200, dropped if its reserve is not held when it is due; each runs until 22 s. The script checks that no alarm is raised in the time-budget runs.

**Time budgets.** The governor evaluates the fresh candidates in the order in which the generator lists them, from ambition 1 to the continuation alone, until the deadline; a candidate not evaluated counts as not certified, and the fallback needs no evaluation.

| Run | Fresh candidates evaluated in a period |
|---|---|
| `full` | every one (the run of Chapter 3; the script checks that its rows are those of Chapter 3) |
| `one-candidate` | only the first, the candidate of ambition 1 |
| `stall` | none in periods 10 to 29, every one otherwise |
| `no-evaluation` | none |

Without a committed fresh candidate in operation mode, the tool stays folded: the operation begins with the first committed plan that unfolds it.

**Alarms and restart.** The survey predictor of Section 7.2.3, used without an offset, with surveys that place the ramp start 0.2 m and 0.35 m too far. After the alarm, the rest feedback brakes maximally, as in Chapter 4, or, for comparison, at the global capability, 0.5 m/s^2 (files ending in `-global`); on the ramp, where both alarms occur, the two move the cart in the same way. The motion of the period of the alarm comes from the floating-point function of Chapter 3, and the point of rest is computed exactly from the speed at which the cart reached the ramp. In every period, the governor searches for a restart candidate, labelled by the global capability, on the set of states that the rest feedback can reach at the next period boundary in the worlds consistent with the global capability. Under maximal braking, while the cart moves towards the wall, that set holds more than one state, and no single-trajectory funnel contains it; at rest, it is the measured state. Braking at 0.5 m/s^2 is realized in every such world, so its set is a single state while the cart moves as well. The restart candidates are those of the generator with the fixed predictor, from that state. An admitted restart candidate is committed at the next period boundary, and the loop restarts with the fixed predictor. The script checks that the alarms and the stops are those that the script of Chapter 7 computes for Section 7.2.3.

## Run

```
python3 cart_governor.py              # all runs; writes results/
python3 cart_governor.py --check      # also checks every number quoted in the text
python3 cart_governor.py --period 179 # prints the trace of one period of the forecasting run
```

A full run takes under a minute.

## Output (`results/`)

| File | Content |
|---|---|
| `summary.txt` | The numbers quoted in Chapter 8. |
| `trace-chapter3.csv` | The decision trace of the forecasting run of Chapter 3: one row per candidate and period. |
| `decisions-chapter3.csv` | One row per period of that run: state, initial set, proposal, requirement, committed candidate and the reason for the choice. |
| `trace-<configuration>.csv`, `decisions-<configuration>.csv` | The same for each of the five runs of Section 8.2. |
| `anticipation.csv` | Position and clearance of the five runs at every period boundary (Figures 8.3 and 8.4). |
| `trace-budget-<run>.csv`, `decisions-budget-<run>.csv` | The same for the time-budget runs of Section 8.3, except `full`, which is the run of Chapter 3. The decisions also record how many fresh candidates were evaluated before the deadline. |
| `trace-alarm-bias-<b>[-global].csv`, `decisions-alarm-bias-<b>[-global].csv` | The same for the alarm runs, with maximal braking or, with `-global`, braking at the global capability as the rest feedback; after the alarm, the decisions record the rest feedback, the restart search and its reason. Figure 8.5 plots the position in the alarm runs and in the time-budget runs, from their decisions files. |

## Numerical precision and reproducibility

Every planned and certified quantity is computed with exact rational arithmetic, as in Chapter 3. In the runs of Sections 8.1 and 8.2 and in the time-budget runs, every label is valid, so the realized motion is exact as well; the script checks that no alarm is raised in them. In the two alarm runs, the motion of the period of the alarm, under an invalid label, is computed in floating point by the function of Chapter 3; the rest feedback continues from the exact value of that floating-point state, and the point of rest is exact. Floating point is otherwise used only to print fractions, through the formatting function of Chapter 3, which prints six decimals; times are rounded half up in exact arithmetic. No random numbers are used. The shipped results were identical, bit for bit, in runs with CPython 3.8.20, 3.13.16 and 3.14.6 on one x86-64 Linux machine; other processors, operating systems and versions have not been tested.
