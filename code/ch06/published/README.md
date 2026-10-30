# Chapter 6: the published algorithms through the book's adapters

This folder runs three published implementations on the governed cart of Chapter 6, through the adapters of Section 6.2 and under the protocol of the stand-ins: SAC as Stable-Baselines3 implements it, the official TD-MPC2 and the authors' Diffusion Policy. [`PROTOCOL.md`](PROTOCOL.md) states the protocol, the budgets, the hypothesis and the decision rules; it was written before any run, and lists the changes made since under *Deviations*. Section 6.2.4 of the book reports all three, with the eight primary comparisons decided over the complete family, and the exploratory runs of SAC with the stand-in's discount. [`PROTOCOL-R2.md`](PROTOCOL-R2.md) is a second protocol, written after the results of SAC and TD-MPC2 and before any run of its own: the two are trained again with a reward that charges for time. It has no runs yet.

| File | Content |
|---|---|
| `cart_env.py` | The environment, with the Gymnasium API: one world of the protocol per episode, the book's package, predictor and governor from `../cart_learning.py` and `../../ch05`, unchanged. `reward="registered"` (default, `PROTOCOL.md`) or `reward="time"` (`PROTOCOL-R2.md`). |
| `protocol.py` | Worlds, budgets, the episode drivers, the timing of a chunk (`ChunkPolicy`), the metrics files, the summaries and the record. |
| `test_cart_env.py` | The validation: the stand-ins of `../cart_learning.py` trained and evaluated through the environment reproduce every run of Table 6.3 (tab:learning-runs); a random-policy smoke test; the Gymnasium API; in nine test runs, the two rewards give the same run for the same proposals, and every period's reward is its formula. Writes `results/validation.txt` (`--quick`: `results/validation-quick.txt`). |
| `test_harness.py` | A test of the training scripts' control flow with stand-ins for the libraries (for a machine without them), with both rewards, and of the two reports. |
| `train_sac.py`, `train_tdmpc2.py`, `train_diffusion.py` | The three jobs. The first two take `--reward time` for `PROTOCOL-R2.md`; `train_sac.py` also takes `--gamma` (deviation 2). |
| `report.py` | Applies the decision rules of `PROTOCOL.md` and writes `results/summary.txt`; `--pack` writes `results/send-back.zip`. Version 2 (29 October 2026) takes Holm's correction over the registered eight tests also while some are missing, and reports a method only once its three seeds are in (`PROTOCOL.md`, Deviations). |
| `report_r2.py` | Applies the decision rules of `PROTOCOL-R2.md` and writes `results/summary-r2.txt`; it never changes `results/summary.txt`. |
| `reward_check.py` | Exploratory, not a result of either protocol: what the two rewards pay four reference policies, undiscounted and with the discounts of SAC and TD-MPC2; bounds on the published methods' undiscounted returns. Writes `results/reward-check.txt` (about 5 minutes). |
| `training_blocks.py` | The course of a method's training in blocks of episodes, from `training.csv`: the coordinates of the supplement's figures of SAC's and TD-MPC2's training. Not a result of the protocol. |
| `requirements-*.txt` | The exact versions, one environment per job. |

## What has been run, and where

`test_cart_env.py`, `test_harness.py` and `reward_check.py` have been run on an x86-64 Linux machine without a GPU or PyTorch; `results/validation.txt` and `results/reward-check.txt` are the outputs of the first and the third, and on that machine `report.py` and `../discount_check.py` reproduce the laptop's `summary.txt` and `discount-check.txt` exactly. On a Windows laptop with an RTX 4060 Laptop GPU:

- `test_cart_env.py --quick` passed (`results/validation-quick.txt`, Python 3.11.18, gymnasium 1.3.0, whose environment checker passed);
- `train_sac.py`, seeds 1 to 3, on 27 October 2026, on the CPU (`results/sac/seed1` to `seed3`, with `results/sac/console.txt`; the trained policies `M.zip` and `L.zip` are kept by the author, not here);
- `train_tdmpc2.py --compile off`, seeds 1 to 3, finished on 28 and 29 October 2026 (`results/tdmpc2/seed1` to `seed3`, with `results/tdmpc2/console.txt`; the agents `M.pt` and `L.pt` are kept by the author, not here);
- `train_diffusion.py`, seeds 1 to 3, finished on 29 and 30 October 2026 (`results/diffusion/seed1` to `seed3`, with `results/diffusion/console.txt`; the policies `policy.pt` are kept by the author, not here);
- `train_sac.py --gamma 0.5`, seeds 1 to 3, on 30 October 2026, on the CPU (`results/sac-gamma0.5/seed1` to `seed3`, with `results/sac-gamma0.5/console.txt`), and then `../discount_check.py` (`../results/discount-check.txt`): exploratory, deviation 2 of `PROTOCOL.md`;
- `PROTOCOL-R2.md` (`--reward time`): not run. Two attempts stopped at once, before any training: `train_sac.py --reward time` in an environment without Stable-Baselines3, and `train_tdmpc2.py --reward time` given a folder that held no checkout of TD-MPC2 at the pinned commit.

`PROTOCOL.md` and `report.py` cite the tables by the numbering of an earlier draft of the book, in which the stand-ins' table, Table 6.3 (tab:learning-runs) now, was Table 6.4. `results/summary.txt` is the output of `report.py` (version 2) on these files; `results/summary-report-v1.txt` that of version 1 on the SAC runs, kept as the protocol's deviations require. In the `record.json` files of TD-MPC2 and the diffusion policy, the user name in the run machine's paths is replaced by `<user>`, as in the logs of Chapter 11; nothing else in the shipped files is changed, except that the console output of TD-MPC2 is converted from UTF-16 to UTF-8.

## Install

Three environments, because the libraries pin different versions of PyTorch (Stable-Baselines3 2.9.2 needs torch 2.8 or later; the TD-MPC2 commit pins torch 2.7.1). In each, install torch first, from the PyTorch index that matches the machine (the CPU index for SAC; for the GPU jobs the CUDA 12.8 index, which the TD-MPC2 runs above used), then the rest; the comments at the top of each requirements file give the commands.

With Miniforge (in the Miniforge Prompt on Windows):

```
conda create -n ch06-sac python=3.11 -y
conda activate ch06-sac
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-sac.txt

conda create -n ch06-tdmpc2 python=3.11 -y
conda activate ch06-tdmpc2
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-tdmpc2.txt

conda create -n ch06-diffusion python=3.11 -y
conda activate ch06-diffusion
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-diffusion.txt
```

Or with `venv`: `python -m venv .venv-sac` (Python 3.11 to 3.13), `.venv-tdmpc2` (Python 3.11), `.venv-diffusion` (Python 3.11 or 3.12), and the same `pip` commands in each.

The two checkouts, at the commits that the scripts check (they refuse another commit unless given `--allow-other-commit`, and the record keeps the commit either way):

```
git clone https://github.com/nicklashansen/tdmpc2.git
git -C tdmpc2 checkout 332f19e4aba4bffe2dc736f45ea7f7bdc585903a
git clone https://github.com/real-stanford/diffusion_policy.git
git -C diffusion_policy checkout 5ba07ac6661db573af695b419a7947ecb704690f
```

Nothing is installed from the checkouts: the scripts import from them.

## Run

From this folder, in each environment, first the checks (minutes), then the job:

```
python test_cart_env.py --quick                          # the environment, with gymnasium's (and SB3's) checkers
python train_sac.py --smoke                              # ch06-sac
python train_tdmpc2.py --tdmpc2 PATH/tdmpc2 --smoke       # ch06-tdmpc2
python train_diffusion.py --diffusion-policy PATH/diffusion_policy --smoke   # ch06-diffusion

python train_sac.py
python train_tdmpc2.py --tdmpc2 PATH/tdmpc2
python train_diffusion.py --diffusion-policy PATH/diffusion_policy
python report.py --pack
```

`--seeds 1` (or 2, or 3) runs one seed, so that seeds can run in parallel processes; each seed writes its own folder `results/<method>/seed<s>/`. A smoke run writes only under `results/smoke/` and is not a result. `train_tdmpc2.py --compile off` switches off `torch.compile` (on by default under Linux), which changes the speed and not the algorithm; it is needed where Triton is missing, as on Windows.

**The second protocol** (`PROTOCOL-R2.md`): SAC and TD-MPC2 with the time reward, TD-MPC2 with the discount 0.99; every other setting unchanged.

```
python train_sac.py --reward time --smoke                                  # ch06-sac
python train_sac.py --reward time                                          # results/sac-time/
python train_tdmpc2.py --tdmpc2 PATH/tdmpc2 --compile off --reward time --smoke   # ch06-tdmpc2
python train_tdmpc2.py --tdmpc2 PATH/tdmpc2 --compile off --reward time    # results/tdmpc2-time/
python report_r2.py                                                        # results/summary-r2.txt
python report.py --pack
```

A follow-up that the first protocol did not register (deviation 2 of `PROTOCOL.md`): SAC with the discount of the tabular stand-in, every other setting unchanged.

```
python train_sac.py --gamma 0.5 --smoke                 # ch06-sac; writes results/smoke/sac-gamma0.5/
python train_sac.py --gamma 0.5                         # writes results/sac-gamma0.5/seed1 to seed3
python ../discount_check.py                             # compares it with the stand-in at 0.5
```

`report.py` does not read `results/sac-gamma0.5/`; `../discount_check.py` reports the comparison as exploratory.

The inference times are measured on the machine as it is; another job running at the same time can lengthen them. Run one job at a time when those times matter, or note which jobs ran together.

## How long

Measured on the laptop above (the records' `wall_clock_s`). The governor runs in exact arithmetic on one CPU core in every job, and the load of the machine was not controlled.

| Job | Per seed | Three seeds, one after the other |
|---|---|---|
| SAC (CPU, one thread) | training 1.2, 4.1 and 1.5 h (168,174, 160,953 and 161,257 steps); evaluation about 1 min per budget | about 7 h |
| TD-MPC2 (GPU, `--compile off`) | training 13.3, 7.5 and 7.1 h (184,021, 175,539 and 180,652 steps); evaluation 4 to 7 min per budget | about 28 h |
| Diffusion policy (GPU) | training 2.2, 2.7 and 2.0 h (10⁴ and 2 × 10⁴ to 3 × 10⁴ gradient steps of batches of 256); evaluation 31 to 37 min, at a median of 1.4 s per chunk with 100 denoising steps | about 9 h |
| SAC, `--gamma 0.5` (CPU, one thread) | training 0.75, 0.79 and 1.25 h (95,838, 96,332 and 96,428 steps); evaluation 0.5 to 1 min per budget | about 3 h |
| `PROTOCOL-R2.md` | not measured; the number of steps depends on how soon the cart arrives, so plan for the times of the first protocol | about 7 h (SAC) and 28 h (TD-MPC2) |
| `test_cart_env.py` (CPU) | 18 to 40 min with two processes on an x86-64 Linux machine; `--quick`, about 1 to 2 min | |

## Archiving the results

`python report.py --pack` writes `results/send-back.zip`: every `.txt`, `.csv`, `.json` and `.md` file under `results/` except the smoke runs, so `summary.txt`, `summary-r2.txt` when it exists, and for every method and seed `evaluation.csv`, `training.csv` and `record.json` (versions, commits, reward, discount, GPU and CPU, wall-clock times, inference times, `pip freeze`), without the trained policies, which are large (`*.zip`, `*.pt`). Keep the console output of each job with it. If a smoke run fails, its console output says where.

## The environment in brief

`CartGovernedEnv(proposal="reference" | "sequence", package="tube", worlds=[...], after_arrival="continue" | "terminate", post_arrival_policy=None, miss=0.0, normalize_actions=False, reward="registered" | "time")`. `reset(options={"world": seed})` runs the world of that seed; without the option, the next world of the schedule `worlds`; `reset(seed=s)` seeds `np_random` and restarts the schedule. The observation is the vector of Table 6.1 with two features, scaled; `info["obs"]` is the observation dictionary of `../cart_learning.py`. A float action is a reference acceleration (times 1.875 m/s² with `normalize_actions=True`), resolved to 0.001 m/s²; a Fraction passes exactly; `None` is a missed deadline. The registered reward is the off-policy learner's (the executed reference acceleration of the next period over 1.875 m/s², minus 0.5 times the intervention); the time reward is −1 minus 0.5 times the intervention; both are 0 after arrival, and the reward changes nothing else. Mode `"continue"` runs every episode until the cart is at rest, as the protocol's runs do; mode `"terminate"` ends the learner's episode at arrival and, with a `post_arrival_policy`, finishes the run itself. `info["metrics"]` at the end of an episode holds the metrics of `cart_learning.run_metrics`.
