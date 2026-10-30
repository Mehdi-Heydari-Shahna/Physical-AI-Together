# The hypotheses of the benchmark (Section 16.1)

Written on 25 October 2026, before the first full run of `cart_benchmark.py` and `cart_substitution.py`, and kept in the scripts (`bench/hypotheses.py`), which apply each decision rule to the runs as written here. A change made after a run is listed under *Deviations* at the end, with its reason; the results are reported under the hypotheses as they stood before the change as well.

## The anticipation benchmark (`cart_benchmark.py`)

The task: the cart of Chapter 3 (exact arithmetic, the generator and test of the control package `cart.constant-command` of Chapter 10, with one more proposal, `retreat`, which only the preview planner uses) drives into the work zone (at least 24 m) and holds there; the tool operation of Chapter 3, unfolding the tool by 0.8 m, needs 1.3 m of reserve when it is due. Each world of the family (seeds 16001 to 16100, 100 worlds) draws, in this order: the ramp's start, uniform on [17, 21] m on a 1 cm grid; whether the survey is unreliable (probability 0.2); the survey's error, uniform on [-0.6, 0.6] m if it is unreliable and on [-0.2, 0.2] m otherwise, on a 1 cm grid (the condition "as calibrated" of Section 7.3); the forecast onset, a period uniform on 200 to 240; the lead time, uniform on 10 to 60 periods, so that the operation is announced in the forecast onset less the lead time; whether the onset slips (probability 0.3), by how many periods (1 to 10) and in which direction (later or earlier, each with probability 1/2). The true onset is the forecast onset plus the slip. A run lasts 320 periods (32 s). An operation that cannot begin when it is due is postponed, as in Section 8.2.3: to the forecast onset if that is still ahead, else by 21 periods, announced.

The configurations (capability predictor; demand predictor; proposal method; budget):

| Id | Name | Capability | Demand | Proposal |
|---|---|---|---|---|
| C1 | filter, fixed model | fixed (global capability) | none | naive |
| C2 | filter, measured model | reactive (measured on traversed ground) | none | naive |
| C3 | filter, survey | survey, calibrated offset of Section 7.3 | none | naive |
| C4 | fixed conservatism | survey | the operation's reserve in every period | naive |
| C5 | reactive preparation | survey | the operation once it is due | naive |
| C6 | ALLOVI | survey | the task plan's forecast | naive |
| C7 | ALLOVI without the capability forecast | fixed | forecast | naive |
| C8 | oracle capability (diagnostic) | the true map | forecast | naive |
| C9 | oracle demand (diagnostic) | survey | the true onset | naive |
| C10 | both oracles (diagnostic) | map | true onset | naive |
| C11 | filter, survey, preview planner | survey | none | preview planner, exact |
| C12 | filter, survey, late preview planner | survey | none | preview planner, 6 periods late |
| C13 | ALLOVI, late preview planner | survey | forecast | preview planner, 6 periods late |
| C14 | ALLOVI, preview planner | survey | forecast | preview planner, exact |
| C15 | ALLOVI, one candidate per period | survey | forecast | naive; budget: one fresh candidate |

A world is *valid* if its survey places the ramp's start no more than the calibrated offset, 0.40 m, after the true start: then every label of the survey predictor is valid (Section 7.3). In the other worlds the survey's labels can be invalid, and the theorems of Chapter 9 say nothing about the runs that execute them; the hypotheses below that concern the survey predictor are stated for the valid worlds, and the other worlds are reported apart.

"Demand: none" means that no demand predictor is configured: the governor is the structural reading of a predictive safety filter of Table 9.1, with the test, the generator and the budget of the other configurations. The preview planner reads the task plan, as the demand predictor does, and plans its own back-away from a model of the generator's: it proposes `retreat` from the latest period from which backing away still brings the cart to rest at 23.7 m by the forecast onset (or 6 periods after it, late).

- **A1 (the certificate).** In every run of every configuration whose executed labels are all valid, as the world's oracle assesses them, no alarm is raised and the true clearance never falls below the floor, 0.5 m. Decision: supported if there is no exception.
- **A2 (no demand channel).** With the naive proposal method and no demand predictor (C1 to C3), the operation does not begin in any valid world in which the cart is in the work zone at the true onset. Decision: supported if there is no exception.
- **A3 (preparation from the forecast).** In every valid world with an exact forecast (no slip): with a lead time of at least 21 periods, C6, C7 and C8 begin the operation at its onset; with a lead time of at most 18 periods, they do not; C5 begins it 21 periods late in every world in which the cart is in the work zone at the onset. Decision: supported if every statement holds in every world of its subset.
- **A4 (fixed conservatism).** C4 begins the operation at its onset in every valid world and never brings the cart into the work zone. Decision: supported if both hold in every valid world.
- **A5 (the capability forecast).** In every valid world: the cart arrives in the work zone earlier with C6 than with C7; C8 arrives no later than C6; and C6 and C7 begin the operation in the same period. In every world, C2 arrives in the same period as C1. Decision: supported if every statement holds in every world of its subset.
- **A6 (equal information).** In every valid world: C11 begins the operation in the same period as C6, with the same arrival and the same time in the work zone; C14 does as C11; C13 begins the operation in the same period as C6. In every valid world with no slip and a lead time of at least 27 periods, C12 begins the operation later than C6. Decision: supported if every statement holds in every world of its subset.
- **A7 (errors of the forecast).** In every valid world whose true onset comes at least 21 periods after the announcement, C9 begins the operation at the true onset. In every valid world with a lead time of at least 21 periods: if the onset slips later, C6 begins the operation at the true onset and spends 0.1 s less in the work zone than C9 for every period of slip; if it slips earlier, C6 begins it at the forecast onset. Decision: supported if every statement holds in every world of its subset.
- **A8 (the budget).** C15 begins the operation in the same period as C6 in every valid world. Decision: supported if there is no exception. (A1 covers its alarms and clearances.)

## The substitution grid (`cart_substitution.py`)

The task of Chapter 6: the disturbed cart of Chapter 5 reaches the work zone within 20 s in the world family of Chapter 6, with the forecasting predictor, whose labels are valid by construction. Four control packages of Chapter 5 (`tube`, `qp`, `ccm`, `mpc`) are crossed with six proposal methods of Chapter 6 (the rule, the braking curve, the expert planner, and the three learners, which stand in for SAC, TD-MPC2 and a diffusion policy). Each learner is trained with the package it is evaluated with, by the protocol of Chapter 6 (seeds 1 to 3, 40 training episodes, or five demonstrations and five aggregation episodes for the imitator; the 20 evaluation worlds 1000 to 1019). The `mpc` package receives the adapters for reference and sequence proposals that Chapter 6 gave the other packages; nothing else changes.

- **B1 (the certificate in every pairing).** In every training and evaluation run of every pairing, no alarm is raised and no executed segment leaves its funnel, and every run that arrives comes to rest before the floor. Decision: supported if there is no exception.
- **B2 (the package sets the arrival).** The range of the packages' mean arrival times, each averaged over the six proposal methods, is larger than the range of the proposal methods' mean arrival times, each averaged over the four packages. Decision: supported if it is larger.
- **B3 (no partnership).** The packages rank in the same order of mean arrival for every proposal method. Decision: supported if the order is the same; ties count as agreement.
- **B4 (interventions belong to the proposal method).** Among the methods that propose one reference, and separately among those that propose sequences, the range of the methods' mean interventions, each averaged over the packages, is larger than the range of the packages' mean interventions, each averaged over those methods. Decision: supported if it is larger in both groups.

## Deviations

No hypothesis, decision rule, configuration, module or world was changed after a run.

- 25 October 2026, after the first full run of `cart_benchmark.py`: A1 to A4, A7 and A8 were supported; A5 and A6 were not. Reporting was added, and nothing else: the cases and exceptions of each statement of each hypothesis (`bench/hypotheses.py`, whose rules and verdicts are unchanged), the differences between the runs that A6 compares, and the survey's coverage in the family with its exact interval (`cart_benchmark.py`). The summary's verdicts are those of the first run, which a second, a third and a fourth run repeated value for value; the fourth was made after the last change below, with `--check --jobs 2`, and its decision times are those that the summary reports. The data files of the first, the third and the fourth run were compared byte for byte; those of the second were not kept.
- The same day, after the first run of `cart_substitution.py`: B1 was supported; B2, B3 and B4 were not. Nothing was changed but its `EXPECTED`, filled from that run's summary; the second run, with `--check`, repeated every value and every row of its data files.
- The same day, `cart_diagnosis.py` was written to diagnose the exceptions of A5, A6 and B2 to B4, and the runs in the worlds that are not valid. It reruns the runs concerned and reports what the governor, the proposal methods and the packages did there; it tests no hypothesis, and its explanations were found after the runs, so Sections 16.2 and 16.3 report them as diagnoses, not as results of the benchmarks.
- After the first runs of both scripts, one cross-reference was corrected in the paragraph that explains "Demand: none": Table 9.4 became Table 9.1, the table of established mechanisms read as viability governors. Nothing else above this section was changed.
- After the third run, the heading of the summary's group for the traced worlds was corrected from "Figure 16.1" to "Figure 16.2", the number of the figure that they draw, in `cart_benchmark.py` and its `EXPECTED`; nothing else changed. A fourth run, with `--check`, repeated every value, and its files of runs, failed runs and traces were those of the first and the third, byte for byte. The third run's summary is kept as `results/benchmark-summary-run3.txt` for its decision times.
