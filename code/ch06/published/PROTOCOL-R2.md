# The second protocol of the published algorithms: the time reward (Section 6.2.4)

Written on 29 October 2026, after the runs of SAC and TD-MPC2 under `PROTOCOL.md` had been reported (`results/summary.txt`), and before any run under this protocol. The choices below were made knowing those results and the exploratory check `reward_check.py` (`results/reward-check.txt`); the runs that this protocol governs are new, and none of them had been made when it was written. `PROTOCOL.md`, its runs and its decisions stand unchanged: this is a second experiment, reported beside the first, not a revision of it.

## Why

Under `PROTOCOL.md`, SAC arrived later than its stand-in at both budgets, and later at budget L than at budget M; TD-MPC2 arrived in none of its 60 evaluation runs at budget M and in 3 of 60 at budget L (`results/summary.txt`). Both were trained with the registered reward, the off-policy learner's: the executed reference acceleration of the next period divided by 1.875 m/s², minus 0.5 times the intervention. That reward has no term for elapsed time or for arriving. Its first term sums, over the periods before arrival, to about the reference speed reached divided by 0.1875 m/s (plus up to 0.27 for each standstill on the way, because the reward counts no acceleration for a rest segment, which takes over the braking from 0.05 m/s), so without a discount a run that arrives without stopping collects at most 3.27 from it and a run in which the cart stays at rest collects exactly 0; the stand-in's evaluation runs collect 0.74 on average in all, the braking curve's −9.07 and the rule's −28.49 (`reward_check.py`). A discount separates arriving from standing still only through the timing of the accelerations, and at SAC's discount of 0.99 the rule's runs collect less than a run at rest (−6.86 against 0). The check does not explain the runs: with TD-MPC2's discount of 0.975, the stand-in's runs collect 13.22 and the rule's 6.82, more than a run at rest. This protocol asks one question: do SAC and TD-MPC2 arrive earlier when the reward charges for time?

## What changes

1. **The reward of SAC and TD-MPC2** (`cart_env.CartGovernedEnv(reward="time")`): in every period before arrival, −1 minus 0.5 times the intervention; 0 after arrival. The intervention term is the registered one; the acceleration term is replaced by a constant cost per period, so that the return is larger the earlier the cart arrives and the less the governor intervenes, whatever the discount. An episode still ends for the learner at arrival (a termination) or at 20 s (a truncation), and the environment still finishes the run with the current policy acting deterministically, outside the replay buffer.
2. **TD-MPC2's discount: 0.99**, SAC's, set through the official configuration as `discount_min = discount_max = 0.99`. Under `PROTOCOL.md` the official rule derived 0.975 from the 200 periods of an episode. Reason, from `reward_check.py`: with the time reward and the discount 0.975, the rule's runs collect −43.39 and the braking curve's −40.33, less than a run at rest, −40.00 (which pays −1 per period for ever, since a learner bootstraps past the truncation at 20 s), so that a learner could prefer standing still to arriving as these policies arrive; with 0.99, every reference policy that arrives collects more than a run at rest: the stand-in −63.19, the braking curve −69.26, the rule −78.49, at rest −100.00. SAC's discount stays its default, 0.99.

## What does not change

Everything else is that of `PROTOCOL.md`: the implementations and their commits (Stable-Baselines3 2.9.2; TD-MPC2 at `332f19e4aba4bffe2dc736f45ea7f7bdc585903a`), every other hyperparameter, the observation, the action and its scale, the task, the worlds, the governor, the package, the predictor and the reserve floor; seeds 1, 2 and 3; budgets M (the snapshot after 40 training episodes) and L (1000 episodes), in the training worlds 100 s + i mod 40; the evaluation of every frozen policy in the worlds 1000 to 1019 (SAC's deterministic action; TD-MPC2's `act(eval_mode=True)`, torch seeded with the world's seed), which uses no reward; the metrics; the record. The evaluation of a policy trained under this protocol is therefore directly comparable, run for run, with that of the same method, seed and budget under `PROTOCOL.md`.

Not run under this protocol: the diffusion policy, which learns from demonstrations and has no reward, and the stand-ins, whose runs are those of `../results/evaluation.csv`.

## How it is run

`python train_sac.py --reward time` writes `results/sac-time/seed<s>/`; `python train_tdmpc2.py --tdmpc2 PATH --reward time` writes `results/tdmpc2-time/seed<s>/`. Each `record.json` states the reward and the discount. `python report_r2.py` writes `results/summary-r2.txt`; it never changes `results/summary.txt`. Runs made with `--smoke` test the installation and are not results.

## Hypothesis H1: the guarantee during learning

As in `PROTOCOL.md`: in every training episode and every evaluation run of the two methods, at both budgets, the governor commits a plan in every period, no alarm is raised, no executed segment leaves its funnel, the cart never passes 24.5 m, and every run that arrives comes to rest. Decision: supported if there is no exception; an exception would show an error in the environment, the package or the test, and the comparisons would be withheld until its cause is found.

## Comparisons and their decision rule

The unit, the test and the reading are those of `PROTOCOL.md`: for each method and budget and each of the 20 evaluation worlds, the mean over the three trained policies of the arrival time (infinite if one of them does not arrive there) and of the run's mean intervention, as printed with four decimals in the evaluation files; two methods are compared world by world, ties are dropped, and the exact two-sided sign test gives p; "earlier" ("lower") means better in more worlds with the decision's p below 0.05, "later" ("higher") the reverse, otherwise "no difference shown".

**Primary comparisons** (the family of four tests, with Holm's correction at 0.05):

| | Comparison | Quantity |
|---|---|---|
| Q1 | SAC with the time reward, budget M, against SAC under `PROTOCOL.md`, budget M | arrival |
| Q2 | SAC with the time reward, budget L, against SAC under `PROTOCOL.md`, budget L | arrival |
| Q3 | TD-MPC2 with the time reward, budget M, against TD-MPC2 under `PROTOCOL.md`, budget M | arrival |
| Q4 | TD-MPC2 with the time reward, budget L, against TD-MPC2 under `PROTOCOL.md`, budget L | arrival |

For TD-MPC2 the decision compares two changes together, the reward and the discount; this protocol does not separate them.

**Secondary comparisons** (uncorrected p, reported and never quoted as a shown difference), for each method and budget with the time reward: against its stand-in of `PROTOCOL.md` in arrival (SAC: the off-policy learner; TD-MPC2: the planner with a learned model); against the off-policy learner in intervention; against the rule, the braking curve and the expert planner in arrival, and the rule and the braking curve in intervention; against the same method and budget under `PROTOCOL.md` in intervention. Then TD-MPC2 against SAC, both with the time reward, in arrival and intervention, at each budget; and budget L against budget M for each method.

A method arrives in the protocol's sense only if all 60 of its evaluation runs arrive; any run that does not is reported. The comparisons concern these three trained policies of each method in these 20 worlds; they do not estimate the variation between training seeds, and they rank nothing outside the cart, its world family, these budgets and this reward.

## How the results will be reported, whatever they are

`report_r2.py` writes `results/summary-r2.txt`: the table, with the runs of `PROTOCOL.md` beside those of this protocol, the groups of each method and budget, the verdict of H1, the decisions of Q1 to Q4 with their counts and p-values, the secondary comparisons and the record. Section 6.2.4 of the book will report them beside the results of `PROTOCOL.md`, which stand as they are; the results of either protocol are never reported without the other. A run that fails, a method that does not learn to arrive, or a crash is reported as such. No seed is replaced, no budget is extended and no hyperparameter is tuned after a run, unless the change is listed under *Deviations* with its reason, and then the results under both are reported.

## The code

The scripts that implement this protocol are those of the book's code at revision 44: `cart_env.py`, `train_sac.py`, `train_tdmpc2.py`, `protocol.py` and `report_r2.py` (with the functions of `report.py`, version 2), with `../cart_learning.py` and `../../ch05` unchanged. Their SHA-256 hashes when this protocol was written:

| File | SHA-256 |
|---|---|
| `cart_env.py` | `dd33a0689087520371e129095311ebb8727a7ed65271f16d30d8bfa2a2ab9c1e` |
| `protocol.py` | `085f279d55363175714e72c8898f86a72cdd6e9e46249ac61ac80c3d13595810` |
| `train_sac.py` | `1d20e2009c84cc6a907c0f401a5606f92e6ff9f6c5c4d69e4a1895822bda8eed` |
| `train_tdmpc2.py` | `ce4687b1eb11b7d9be705439f3ec186a2bfb532e4f73a4943f59aaab1f783a1b` |
| `report.py` | `e28b476d463ee18f8b1d278162488430825cac9a94d1ea5bbb017ddf9624c293` |
| `report_r2.py` | `035a191608db9e8a812a08b464e82e3c307b9cfe3636954e84c46f8f3832452b` |
| `../cart_learning.py` | `d7d363402335f0d44016b4ba4ce7107efffeb94da9f07e908f4ce1880608deab` |
| `../../ch05/cart_feedback.py` | `f6299817a2e414e1f50a5d71a4e0482cc4a997d34c3589a14eaa644499340ebd` |
| `../../ch05/cart_methods.py` | `91a16792dad6beba31ada4f5943213e5b4564fad37a3c4d8ca68857e1d21e5d2` |

A change to any of them before or during the runs is a deviation, to be listed below with its reason. `test_cart_env.py` checks, in nine test runs (the rule, the braking curve and random proposals, each in three worlds), that the two rewards give the same run for the same proposals and that each period's reward is its formula; `test_harness.py` runs both training scripts with `--reward time` and `report_r2.py`, with stand-ins for the libraries.

## Deviations

None so far.
