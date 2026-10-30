# Chapter 6: learned proposal methods for the cart

`cart_learning.py` computes every number of Chapter 6. It needs Python 3.8 or later and no other packages. It imports the control package, the forecasting predictor and the governor from `../ch05/cart_feedback.py`, and the contraction-metric package from `../ch05/cart_methods.py`, and uses them unchanged. One capability is added: after the rest segment has held the cart at standstill, the initial set is the measured standstill state, which lies in the funnel of the rest segment, so a launch may follow (Section 6.1.1).

## What a proposal means

| Proposal | Meaning | Generator |
|---|---|---|
| reference | one reference acceleration for the next period, in m/s², limited to ±1.875 | blends with the head of the fallback for ambitions 1, 3/4, 1/2, 1/4, 0, as in Section 5.1.6 |
| sequence, chunk | eight reference accelerations for the next 0.8 s | prefixes of 8, 6, 4 and 2 periods with ambitions 1, 3/4, 1/2 and 1/4, each continued by the recovery; the recovery alone with ambition 0 |
| none | no proposal before the deadline | recovery candidates only |

The generator belongs to the control method; the adapters are in this script because Chapter 5 offered only the proposals "accelerate" and "hold". Any proposal is converted to exact fractions before it enters a segment.

The observation of period k, formed at t_k: the measured position and velocity, the label summary (distance, rounded up to whole metres and at most 30 m, to the first region that the labels mark with less braking than firm ground), the level of the plan in execution, the ambition committed at t_k, and the alarm flag. With the ambition α_k, the governor reports the executed reference of period k+1, the head of the committed plan; the learners of Section 6.2 keep it in their records only, and the delay-aware variant of Section 6.3 also uses it to form its features.

## Proposal methods

| Method | Stands in for | Proposal | What it learns |
|---|---|---|---|
| `Rule` | Chapters 3 and 5 | reference: 2 until arrival, then −2 | nothing |
| `BrakingCurve` | a hand-designed controller | reference: tracks v_env − 0.3 m/s with gain 3 /s | nothing |
| `OffPolicyLearner` | SAC | reference: one of eight accelerations | a table of action values (tabular Q-learning) from replayed transitions |
| `Planner` | TD-MPC2 | sequence | a model of the cart's response to references, and a margin ζ from interventions |
| `ChunkImitator` | Diffusion Policy | chunk, kept for up to four periods | the chunks of an expert planner, with one round of dataset aggregation |

The stand-ins keep the adapter of the method they stand in for: the same kind of proposal, the same generator and the same kind of records. They are teaching examples, not the published algorithms.

The folder [`published/`](published/README.md) runs the published implementations (SAC of Stable-Baselines3, the official TD-MPC2, the authors' Diffusion Policy) through the same adapters, on an environment with the Gymnasium API that imports this script's protocol unchanged (Section 6.2.4). Its validation, run on an x86-64 Linux machine, reproduces every training episode and evaluation run of Table 6.3 through that environment; its training runs need PyTorch and, for two of the methods, a GPU. SAC, TD-MPC2 and the diffusion policy have been run and are reported in Section 6.2.4, with SAC's exploratory runs at the stand-in's discount. Its protocol, `published/PROTOCOL.md`, was written before any run; a second protocol, `published/PROTOCOL-R2.md`, for SAC and TD-MPC2 with a reward that charges for time, was written after their results and has no runs yet.

## The protocol

- **Worlds.** Each world draws, from its seed, the start of the ramp in [17, 21] m and a triangle-wave disturbance with amplitude below 0.05 m/s², period in [1.5, 3] s and a random phase. Labels come from each world's map, so they are valid.
- **Training.** The off-policy learner and the planner train for 40 episodes each, in worlds 100 s, ..., 100 s + 39 for seed s. The imitator records 5 expert episodes and 5 aggregation episodes in worlds 100 s + 50, ..., 100 s + 59. Seeds 1, 2 and 3.
- **Evaluation.** Every policy, frozen, runs in the 20 worlds with seeds 1000 to 1019.
- **A run.** The task is to reach 24 m within 20 s. The run then continues until the cart is at rest, at most until 25 s; the proposal method keeps proposing, without exploring, and nothing after arrival enters its records or the mean intervention. A run stops at an alarm (none occurs).
- **Matched conditions.** The control package, the governor, the predictor, the reserve floor, the world family and the evaluation worlds are the same for every policy; one crossed run trains and evaluates the off-policy learner with the contraction-metric package of Section 5.3.
- **Delays.** The off-policy learner of seed 1 is evaluated with 30 % of its proposals missing their deadline; the chunk imitator of seed 1 with new chunks available two periods later than in the main runs (the new chunk is then proposed from its first reference).
- **Metrics.** Arrival time, mean intervention (one minus the committed ambition, averaged over the periods before arrival), fallback commits, alarms, steps outside a funnel, the largest position and the position at rest, for every evaluation and training episode; the share of periods committed in full and, for reference proposals, the share in which the proposal passes unchanged (the generator realizes it and the candidate of ambition 1 is committed).
- **Section 6.3.** Three variants of the off-policy learner, each trained with seeds 1, 2 and 3: one records the executed reference instead of the proposal; one ignores the label summary; one computes its features at the state predicted for t_{k+1} from the reported executed reference of period k. Diagnostics: the mean intervention of the first and the last ten training episodes; pairs of features and proposal recorded at least twice, and those followed by more than one committed ambition; the feature states recorded in training, and those whose greedy action was never recorded there; the share of proposals that ask for the largest acceleration.

## Run

```
python3 cart_learning.py            # the whole protocol; writes results/
python3 cart_learning.py --check    # also checks every number quoted in the text
python3 cart_learning.py --jobs 1   # one process; the default uses up to two
python3 cart_learning.py --quick    # a smoke test with small budgets; numbers differ
```

A full run takes about fifteen minutes with two processes. The results do not depend on the number of processes.

```
python3 discount_check.py           # the off-policy learner with SAC's discount, 0.99
```

`discount_check.py` (needs NumPy, for the import of `published/report.py`) trains and evaluates the off-policy learner exactly as `cart_learning.py` does, once with the book's discount of 0.5, which must reproduce the learner's 60 evaluation runs of `results/evaluation.csv`, and once with 0.99, the discount of SAC as Stable-Baselines3 sets it; it then compares SAC's registered runs with the learner at each discount, world by world, with the sign test of `published/report.py`, and, once `published/train_sac.py --gamma 0.5` has been run, SAC with the learner's discount as well, against the learner at 0.5 and against the rule, the braking curve and the expert planner. It was written after SAC's results were known (Section 6.2.4): its comparisons are exploratory and uncorrected. It writes `results/discount-check.txt` and `results/discount-check.csv`, in about three minutes with two processes.

## Output (`results/`)

| File | Content |
|---|---|
| `summary.txt` | The numbers quoted or summarized in Chapter 6. |
| `trace-rule.csv`, `trace-learner.csv` | Proposed and executed references in the world of Chapter 5, for the rule and the off-policy learner of seed 1, by the period they concern, up to arrival (Figure 6.1). |
| `training.csv` | One row per training episode of every learner; Figure S6.3 plots it, selecting the learners by their row positions, so a change to the rows' order or to the learners' names must change the figure too. |
| `evaluation.csv` | One row per evaluation run of every policy. |

## Numerical precision and reproducibility

The governor and the test use exact rational arithmetic, as in Chapter 5. The simulation and the learners use floating point with only +, −, ×, ÷, square roots and rounding, in a fixed order: floats are summed from left to right, because Python's `sum()` of floats is compensated from version 3.12 on. Means are computed exactly and rounded half up, as in Chapter 5. Two exceptions call the platform's mathematical library, whose last digits may differ between platforms: `math.exp` in the funnel check of the contraction-metric package (`cart_methods.py`), and the sine disturbance of the world of Chapter 5, used only for the trace. Random numbers come only from `random.Random(seed).random()` with integer seeds. Python's documentation promises only that `random()` keeps producing the same sequence when the compatible seeder is given the same seed ("Notes on Reproducibility" in the documentation of the `random` module); the script relies on integer seeding being that seeder. The shipped results were identical, bit for bit, in runs with CPython 3.8.20, 3.13.16 and 3.14.6 on one x86-64 Linux machine; other processors, operating systems and versions have not been tested.
