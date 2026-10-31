# Reproducing the results of the book

Every number that *Physical AI, Together* quotes from a computation is produced by a script in `code/`, and most scripts' `--check` option compares what they compute with the values that the book quotes. This file says how to obtain those results again from a fresh copy: the environments, the commands in the order in which their results depend on one another, how long each takes, and what `--check` prints when every value is as quoted. It ends with how to report what you found, whether every value agreed or not. Each chapter folder's `README.md` gives the detail: what each script computes, the files it writes, and the versions with which the shipped results were produced.

## Degrees of a reproduction

Section 17.3.1 of the book distinguishes four degrees, and a report should say which it reached:

1. the script's own check, rerun where it was run;
2. a run in a new environment built from the record's pins;
3. a run on another machine or operating system, within the variation that the record allows (Section S12.8 of the supplement);
4. a run by someone outside the team that produced the results.

No run that this repository records was made outside the team that produced the results. A run by someone outside that team is of the fourth degree, and also of the second or the third if its environment or its machine differs from the record's.

## Before you start

- **A fresh copy.** Clone the repository (`git clone https://github.com/Mehdi-Heydari-Shahna/Physical-AI-Together.git`) and check out the release or commit that you reproduce (`git checkout v1.0`), or unzip its archive, into a new folder; the archive of v1.0 is at Zenodo, DOI [10.5281/zenodo.23823947](https://doi.org/10.5281/zenodo.23823947). Note the commit (`git rev-parse HEAD`) or the archive's identifier for your report.
- **A copy of the shipped results.** Each script overwrites the files that it writes in its folder's `results/`, so keep a copy of the shipped ones before you run anything, and compare afterwards (see "Comparing the files" below):

  ```
  cp -a code code-shipped
  ```

- **Where to run.** Run every command from the folder of its chapter, as below; the scripts find the other chapters' code by relative paths, so keep the folder `code/` whole.
- **What `--check` does.** It compares what the script computes with its table `EXPECTED`, which holds every value that the book quotes from the script's summary and, for most scripts, more; the few values that the book counts from a data file, and wall-clock times, are not checked. It prints the summary, writes it into `results/`, and prints one line that begins with `[check]`: the last line for every script but `ch16/cart_benchmark.py`, which prints the decisions' wall-clock times and its running time after it. The script exits with status 0 if every value is as expected and with status 1 otherwise, and the `[check]` line then lists every value that differed, with the expected and the computed value. The shipped summary file holds the same `[check]` line, so `grep '^\[check\]' code-shipped/chNN/results/<file>` shows what to expect.
- **Running times.** The times below are those that the folders' READMEs state, on the machines that wrote the shipped results (x86-64 Linux; two processors for the scripts that use two). Scripts with `--jobs` use up to two processes by default, and their results do not depend on the number.

## Chapters 3 to 10, 16 and 17: Python alone

These scripts need Python 3.8 or later and no other package. The READMEs of Chapters 6, 7, 8 and 10 record that their shipped results were identical, bit for bit, with CPython 3.8.20, 3.13.16 and 3.14.6 on one x86-64 Linux machine, and those of Chapters 16 and 17 name CPython 3.13.17. Run them in this order: Chapter 5 reads the summary of Chapter 3; Chapter 9 and the first script of Chapter 10 read the records of Chapter 8; Chapter 16 imports the code of Chapters 6, 7 and 10 and compares its runs with results of Chapters 6 and 10; and Chapter 17 imports Chapters 10 and 16. The results that a script reads are shipped, so a later script also runs if you skip an earlier one.

```
cd code/ch03
python3 cart_example.py --check
cd ../ch05
python3 cart_feedback.py --check
python3 cart_methods.py --check
cd ../ch06
python3 cart_learning.py --check
cd ../ch07
python3 cart_forecasts.py --check
cd ../ch08
python3 cart_governor.py --check
cd ../ch09
python3 cart_lead_time.py --check
cd ../ch10
python3 cart_allovi.py --check
python3 cart_checks.py --check
python3 model_providers.py --check
cd ../ch16
python3 cart_benchmark.py --check
python3 cart_substitution.py --check
python3 cart_diagnosis.py --check
cd ../ch17
python3 contribute.py --check
python3 contribute_diagnosis.py --check
```

| Folder | Command | Time (README) | `[check]` line when every value is as quoted | Summary file |
|---|---|---|---|---|
| `ch03` | `python3 cart_example.py --check` | about 15 s | `[check] 119 quoted values: all as quoted` | `results/summary.txt` |
| `ch05` | `python3 cart_feedback.py --check` | about 15 s | `[check] 58 quoted values: all as quoted` | `results/summary.txt` |
| `ch05` | `python3 cart_methods.py --check` | about 70 s | `[check] 111 quoted values: all as quoted` | `results/methods-summary.txt` |
| `ch06` | `python3 cart_learning.py --check` | about 15 min with two processes | `[check] 219 values quoted or summarized in Chapter 6: all as quoted` | `results/summary.txt` |
| `ch07` | `python3 cart_forecasts.py --check` | about 11 min with two processes | `[check] 161 values quoted in Chapter 7: all as quoted` | `results/summary.txt` |
| `ch08` | `python3 cart_governor.py --check` | under a minute | `[check] 196 values quoted in Chapter 8: all as quoted` | `results/summary.txt` |
| `ch09` | `python3 cart_lead_time.py --check` | about 3 min | `[check] 73 values quoted in Chapter 9: all as quoted` | `results/summary.txt` |
| `ch10` | `python3 cart_allovi.py --check` | about a minute (two with CPython 3.8) | `[check] 45 values quoted in Sections 10.1 and 10.2: all as quoted` | `results/summary.txt` |
| `ch10` | `python3 cart_checks.py --check` | about a minute (two and a half with CPython 3.8) | `[check] 45 values quoted in Section 10.3: all as quoted` | `results/checks-summary.txt` |
| `ch10` | `python3 model_providers.py --check` | a few seconds | `[check] 41 values quoted in Section 10.4: all as quoted` | `results/models-summary.txt` |
| `ch16` | `python3 cart_benchmark.py --check` | 14 to 17 min with two processes | `[check] 204 values, every value of the summary but the timing: all as expected` | `results/benchmark-summary.txt` |
| `ch16` | `python3 cart_substitution.py --check` | about 40 min with one process | `[check] 41 values, every value of the summary: all as expected` | `results/substitution-summary.txt` |
| `ch16` | `python3 cart_diagnosis.py --check` | about 10 min with one process | `[check] 49 values: all as expected` | `results/diagnosis-summary.txt` |
| `ch17` | `python3 contribute.py --check` | about 7 min with one process | `[check] 45 values: all as expected` | `results/contribution-summary.txt` |
| `ch17` | `python3 contribute_diagnosis.py --check` | about 4 min with two processes | `[check] 34 values: all as expected` | `results/contribution-diagnosis.txt` |

Some steps of Chapters 6, 11 and 12 also need Python alone, because they read results that were written elsewhere and are shipped:

```
cd code/ch06/published
python3 test_cart_env.py --quick   # the environment of the published learners, through the stand-ins
python3 test_harness.py            # the training scripts' control flow, with stand-ins for the libraries
python3 report.py                  # results/summary.txt from the shipped runs
cd ..
python3 discount_check.py          # results/discount-check.txt
cd ../ch11
python3 isaac_check.py --check     # [check] 238 values quoted in Section 11.2: all as quoted
cd ../ch12
python3 portability.py --matrix    # the support matrix, from the results present
python3 layers.py --compare        # the comparisons of Table S12.4
python3 numerics.py --report       # validation, failures, the record, the portability report
```

`numerics.py --report` writes into `results/portability-report.txt` a SHA-256 of the code of Chapters 10 to 12, which changes with any change to that code, comments and docstrings included; it is the identity of the code that wrote the report, not a value of the book.

## Chapters 11 to 16: the simulators

Each simulator runs in a virtual environment of its own, built from the folder's pinned requirements. The shipped results were produced on x86-64 Linux on the CPU, with the releases that the requirements files pin; each README says on which machines, with which interpreters, and what was reproduced where.

```
cd code/ch11
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt          # MuJoCo 3.16.0; Python 3.10 or later
.venv/bin/python mujoco_cart.py --check            # about 20 s
cd ../ch12                                          # Python 3.10 to 3.13; 3.12 was used
python3.12 -m venv venv-newton
venv-newton/bin/pip install -r requirements-newton.txt
for c in newton-mujoco-warp newton-mujoco-c newton-mujoco-c-hull64 newton-mujoco-c-gap newton-xpbd newton-featherstone newton-semi-implicit; do venv-newton/bin/python portability.py $c --check; done
python3.12 -m venv venv-genesis
venv-genesis/bin/pip install -r requirements-genesis.txt
for c in genesis genesis-mpr genesis-defaults; do venv-genesis/bin/python portability.py $c --check; done
python3 portability.py --matrix
../ch11/.venv/bin/python layers.py mujoco --render --check
venv-newton/bin/python layers.py mujoco --check
venv-genesis/bin/python layers.py genesis-f32 --check
venv-genesis/bin/python layers.py genesis-render --check
../ch11/.venv/bin/python layers.py mujoco-gap --check
venv-newton/bin/python layers.py mujoco-gap --check
venv-newton/bin/python layers.py mujoco-newton-changes --check
python3 layers.py --compare
../ch11/.venv/bin/python numerics.py mujoco --check
venv-newton/bin/python numerics.py newton --warp --check
venv-genesis/bin/python numerics.py genesis --check
venv-genesis/bin/python numerics.py genesis-defaults --check
python3.12 -m venv venv-usd
venv-usd/bin/pip install mujoco-usd-converter==0.6.1
venv-usd/bin/python numerics.py usd --check
python3 numerics.py --report
cd ../ch13
../ch11/.venv/bin/python husky_mujoco.py --check   # about a minute
../ch11/.venv/bin/python husky_route.py --check    # about four minutes
../ch11/.venv/bin/python husky_variants.py --check # about four minutes
cd ../ch14
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt          # MuJoCo 3.16.0, NumPy, PyYAML, SciPy
.venv/bin/python arm_mujoco.py --check             # about 3 s
.venv/bin/python hdmm_mujoco.py --check            # about 75 s
.venv/bin/python excavator_mujoco.py --check       # about 45 s; the excavator is not in the book
.venv/bin/python labels_demand.py --check          # about eighteen minutes
.venv/bin/python hdmm_governed.py --check          # about ten minutes
cd ../ch15
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt          # MuJoCo 3.16.0, NumPy, SciPy
.venv/bin/python quad_model.py --check             # about 10 s; add --menagerie DIR to compare with the meshes too
.venv/bin/python quad_governed.py --check          # about twelve minutes on two processes
.venv/bin/python humanoid_carry.py --team-impedance DIR --check                   # about four minutes
.venv/bin/python humanoid_carry.py --team-impedance DIR --without-follow --check
```

In Chapter 15, `DIR` is a checkout of the Team-Impedance-Control repository at commit `eb530ea`; for `quad_model.py --menagerie DIR`, a checkout of MuJoCo Menagerie at commit `f3b6013`, with which the shipped `model-summary.txt` was written (without it, its line on the model with its meshes is left out). The renderers of Section 12.2 (`--render`, `genesis-render`) need Mesa's OSMesa (on Debian and Ubuntu, the package `libosmesa6`) and PyOpenGL, which pip installs with MuJoCo; in an environment without PyOpenGL, `layers.py mujoco --render` stops at MuJoCo's renderer. On one processor, a configuration of Chapter 12 takes about a minute on MuJoCo's C library, about two in Genesis, about four with XPBD and about ten on MuJoCo Warp, and Featherstone and the semi-implicit solver stop within seconds; the parts of `numerics.py` take about a minute for MuJoCo, about fifty minutes for Newton with `--warp` (ten without) and about ten for Genesis.

| Script | `[check]` line in its shipped summary |
|---|---|
| `ch11/mujoco_cart.py` | `[check] 148 values quoted in Section 11.1: all as quoted` |
| `ch12/portability.py CONFIG`, `ch12/layers.py VARIANT` | `[check] against the shipped summary: all as shipped` |
| `ch12/numerics.py PART` | `[check] against the shipped summary and per-period files: all as shipped` |
| `ch13/husky_mujoco.py` | `[check] 147 values, among them every value that Section 13.1 quotes: all as expected` |
| `ch13/husky_route.py` | `[check] 163 values, among them every value that Section 13.2 quotes: all as expected` |
| `ch13/husky_variants.py` | `[check] 521 values, among them every value that Section 13.3 quotes: all as expected` |
| `ch14/arm_mujoco.py` | `[check] 65 values, among them every value that Section 14.1 quotes: all as expected` |
| `ch14/hdmm_mujoco.py` | `[check] 48 values, among them every value that Section 14.1 quotes: all as expected` |
| `ch14/excavator_mujoco.py` | `[check] 55 values, among them every value that Section 14.1 quotes: all as expected` (the book no longer reports the excavator; the line is the script's as written) |
| `ch14/labels_demand.py` | `[check] 115 values, among them every value that Section 14.2 quotes: all as expected` |
| `ch14/hdmm_governed.py` | `[check] 126 values, among them every value that Section 14.3 quotes: all as expected` |
| `ch15/quad_model.py` | `[check] 22 values, among them every value that Section 15.1 quotes: all as expected` |
| `ch15/quad_governed.py` | `[check] 97 values, among them every value that Section 15.2 quotes: all as expected` |
| `ch15/humanoid_carry.py` | `[check] 22 values: all as expected` |
| `ch15/humanoid_carry.py --without-follow` | `[check] 3 values: all as expected` |

The checks of `ch12/layers.py` and `ch12/portability.py` compare the record of a summary too. With another release of Python than the record's, `layers.py VARIANT --check` prints `DIFFERENT: record.Python: shipped '3.13.16', now ...` and exits with an error even when every computed value is as shipped; a line that names only the record is not a difference of the book's values. MuJoCo computes in double precision, and with another processor, operating system or release of MuJoCo its last digits may differ; the check then lists every value that differs at the precision printed, and Section S12.8 of the supplement states the variation that a reproduction may show. `numerics.py --compare-with PATH` compares a rerun of Chapters 11 and 12 in a copy of `code` at PATH with the shipped results, under that variation (`code/ch12/README.md`).

**The complete case of Chapter 16** (`code/ch16/README.md`): the cart in MuJoCo under the full contract, run in phases under the protocol `code/ch16/complete/PROTOCOL.md`, with MuJoCo 3.16.0 and NumPy (the environment of `code/ch11`). It has no `--check`: the identification freezes the bounds and the hashes of the modules in `complete/FROZEN.json`, the later phases refuse to run if a frozen file has changed, and the shipped files under `code/ch16/results/complete-*` are the record of the runs. Its real-time phase measures the wall clock and asks for a machine with nothing else running; its README gives the order of the phases and their times, and `complete/DEVIATIONS.md` every departure from the protocol.

## What needs a GPU

- `code/ch06/published/` runs the published implementations of SAC, TD-MPC2 and Diffusion Policy through the adapters of Chapter 6, under the protocol `PROTOCOL.md`. Its `README.md` gives its environments (`requirements-*.txt`), its validation and its training jobs, which need PyTorch and, for TD-MPC2 and Diffusion Policy, an NVIDIA GPU, and says which of them have been run; their results are shipped, without the trained policies. `PROTOCOL-R2.md`, a second protocol, has no runs yet.
- `code/ch11/isaac_cart.py` and `isaac_trace.py` run only inside Isaac Sim 6.2.0 on a computer with an NVIDIA RTX GPU. Their results, from the runs that `code/ch11/README.md` describes, are shipped, and `isaac_check.py` checks the values of Section 11.2 from them with Python alone (above).
- `code/ch13/isaac_husky.py` runs only inside Isaac Sim 6.2.0. Its run of 28 October 2026 is shipped in `code/ch13/results/isaac/`, as `code/ch13/README.md` describes.
- The videos of Chapters 13 to 15 and their stills are drawn by scripts with the package `code/render`, whose README lists what they need (Pillow, SciPy, EGL, ffmpeg and checkouts of the robots' descriptions for their visual meshes); the path-traced versions need Blender's Cycles on a GPU (`code/render/cycles/README.md`), and have not yet been rendered. The videos are attached to the releases, not kept in the repository.

## Comparing the files

After the runs, compare the files that the scripts wrote with the shipped ones, for example

```
diff -rq code-shipped/ch08/results code/ch08/results
```

and the same for each folder. A few lines record the machine, the environment or the code rather than a value of the book, and can differ without any value differing:

- in `ch16/results/benchmark-summary.txt`, the group `[timing, this machine; not checked]`, the decisions' wall-clock times;
- in `ch16/results/benchmark-record.txt`, the line that names the interpreter (`Python 3.13.17` in the shipped file);
- in `ch15/results/humanoid-summary.txt`, the wall time;
- in `ch06/published/results/validation-quick.txt`, written on the laptop of the published learners' runs, the interpreter and the packages, and whether gymnasium's checker ran;
- in `ch12/results/portability-report.txt`, the line `code of Chapters 10 to 12, SHA-256 ...`;
- in `ch13/results/route-summary.txt`, `route-record.txt` and `variants-summary.txt`, the version of the core's contract, 1.0 when the shipped results were written and 1.1 now (`code/ch13/README.md`);
- in the simulator chapters, the records (`record.txt` and the like) and the environment files, which name the versions, the processor and the operating system.

On 30 October 2026 the commands of this file that need neither a GPU nor Isaac Sim, Newton, Genesis, the USD converter, Blender or MuJoCo's renderer were run again from a copy of this code, on an x86-64 Linux machine with two processors, under CPython 3.13.17 with MuJoCo 3.16.0, NumPy 2.5.3, SciPy 1.18.1 and PyYAML 6.0.3: the Python-only chapters, the Python-only steps of Chapters 6, 11 and 12, `numerics.py mujoco` and `layers.py mujoco-gap` of Chapter 12, and the MuJoCo scripts of Chapters 11 and 13 to 15 (`quad_model.py` without `--menagerie`); the phases of the complete case of Chapter 16 were not run again. Every check reported every value as quoted, as expected or as shipped, `layers.py mujoco-gap` with the difference of the record's Python release described above, and every file that the runs wrote was identical to the shipped one but for the lines listed above and the line of `ch15/results/model-summary.txt` that needs `--menagerie`.

## Reporting

Report what you ran with the form `code/ch17/templates/REPRODUCTION.md`: the experiment and the release; who reproduced it, and whether from outside the team that produced the results; the degree reached; the environment and how it differs from the record; the commands, exactly as run from a fresh copy; the `[check]` line that each script printed; the values and files compared, and every difference with its cause where you found one; the verdict; and what you did not run, and why. A reproduction in which values differ is as useful as one in which every value agrees: it shows what the record left out.

Open an issue with the form **Reproduction report** (`.github/ISSUE_TEMPLATE/reproduction-report.md`), or add the filled form, as `REPRODUCTION-<script>-<date>.md`, beside the results of the experiment in a pull request.
