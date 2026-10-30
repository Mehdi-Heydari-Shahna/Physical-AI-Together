# Chapter 12: Newton, Genesis and numerical credibility

Section 12.1 runs the declared experiment of Chapter 11, the cart of Chapter 3 on its simulated track, in two more simulators, Newton and Genesis, through simulator adapters with the same contract as `cart.mujoco` of Section 11.1, and publishes the support matrix of the configurations tested. Section 12.2 distinguishes engine, solver and renderer, classifies what each comparison between configurations changes and measures by how much, with controlled variants. Section 12.3 checks the numerical model of the experiment (time steps, the constraint solver, collision detection against the exact geometry, friction, limits, estimation and arithmetic), records the failures found in Chapters 11 and 12 with what reproduces them, and writes the portability report.

## Folder

| File | Content |
|---|---|
| `portability.py` | The script of Section 12.1: one configuration per call, and the support matrix. |
| `layers.py` | The script of Section 12.2: the controlled variants (MuJoCo by itself, with and without a gap on its geoms, and with the changes that Newton makes around it; Genesis in single precision; and MuJoCo and Genesis with a renderer) and the comparisons between configurations. |
| `numerics.py` | The script of Section 12.3: the checks of the numerical model, the reproductions of the failures, the comparison of a rerun with the shipped results, and the portability report. |
| `experiment.py` | The declared experiment, for the ports of the cart: the port checks, the run of Chapter 3 period by period, the cases of Section 10.4 and the configuration of Chapter 3. |
| `cart_ports/design.py` | The design and the MJCF text of Section 11.1, read from `../ch11/cart_mujoco` without importing MuJoCo. |
| `cart_ports/base.py` | The track coordinate, placement and the drive loop of Section 11.1, for any simulator; each port supplies six operations. |
| `cart_ports/newton_cart.py` | The cart in Newton, with a choice of Newton's solvers. |
| `cart_ports/genesis_cart.py` | The cart in Genesis, with the options of the MJCF text (in double or, for Section 12.2, single precision) or Genesis's own defaults, and an optional camera. |
| `cart_ports/world.py` | The simulator adapters `cart.newton` and `cart.genesis`. |
| `requirements-newton.txt`, `requirements-genesis.txt` | The releases with which the results were produced. |

## Running it

Each simulator needs its own Python environment (Python 3.12 was used):

    python3.12 -m venv venv-newton
    venv-newton/bin/pip install -r requirements-newton.txt
    venv-newton/bin/python portability.py newton-mujoco-warp

    python3.12 -m venv venv-genesis
    venv-genesis/bin/pip install -r requirements-genesis.txt
    venv-genesis/bin/python portability.py genesis

The configurations:

| Configuration | What it is |
|---|---|
| `newton-mujoco-warp` | Newton, SolverMuJoCo on MuJoCo Warp, as the port sets it |
| `newton-mujoco-c` | Newton, SolverMuJoCo on MuJoCo's C library, as the port sets it |
| `newton-mujoco-c-hull64` | the same with the importer's convex hull of 64 vertices (a mistake of conversion) |
| `newton-mujoco-c-gap` | the same with Newton's gap of 0.1 m passed to MuJoCo (a mistake of conversion) |
| `newton-xpbd`, `newton-featherstone`, `newton-semi-implicit` | Newton's own solvers, with Newton's contacts and the importer's defaults |
| `genesis` | Genesis, the options of the MJCF text, as the port sets it |
| `genesis-mpr` | the same with Genesis's MPR collision detection, its default (a mistake of conversion) |
| `genesis-defaults` | Genesis with its own defaults |

`python portability.py --matrix` (any Python 3.10 or later; no simulator needed) assembles the support matrix from the results present. Options of a run: `--quick` (the record and the port checks only), `--check` (compare every value with the shipped summary; times and versions are not compared), `--device` (Warp's device for Newton, `cpu` by default).

The variants of Section 12.2, each in the environment of its simulator:

    ../ch11/.venv/bin/python layers.py mujoco --render   MuJoCo 3.16.0, the environment of Chapter 11, without and with a renderer
    ../ch11/.venv/bin/python layers.py mujoco-gap        MuJoCo 3.16.0, the contacts with and without a gap of 0.1 m
    venv-newton/bin/python layers.py mujoco              MuJoCo 3.15.0, the release in Newton's environment
    venv-newton/bin/python layers.py mujoco-gap          MuJoCo 3.15.0, the same
    venv-newton/bin/python layers.py mujoco-newton-changes
                                                         MuJoCo 3.15.0 with the changes that Newton makes around
                                                         it, one at a time and together
    venv-genesis/bin/python layers.py genesis-f32        Genesis, the options of the text, single precision
    venv-genesis/bin/python layers.py genesis-render     Genesis, without and with a renderer, in 13 periods
    python3 layers.py --compare                          the comparisons of Table S12.4 (no simulator needed)

The renderers draw off screen through Mesa's OSMesa: install it (on Debian and Ubuntu, the package `libosmesa6`); the script sets `MUJOCO_GL` and `PYOPENGL_PLATFORM` to `osmesa` unless they are set. `--check` compares every value of a variant's summary with the shipped one (times and the renderer's mean pixel value are not compared).

The checks and reproductions of Section 12.3, each part in the environment of its simulator:

    ../ch11/.venv/bin/python numerics.py mujoco          MuJoCo 3.16.0: the comparison at 0.5 and 0.25 ms, the
                                                         constraint solver, the two collision pipelines against
                                                         the exact geometry, friction and the tool's stop at three
                                                         time steps, and the failures F8 and F9 (about a minute)
    venv-newton/bin/python numerics.py newton --warp     Newton: the comparison at 0.5 and 0.25 ms on MuJoCo's C
                                                         library and on MuJoCo Warp, the contacts inside Newton,
                                                         one period at three time steps, and the failures F2 to F6
                                                         (about fifty minutes on one processor, forty of them on
                                                         MuJoCo Warp; without --warp, about ten)
    venv-genesis/bin/python numerics.py genesis          Genesis: the comparison at 0.5 and 0.25 ms, GJK and MPR
                                                         against the exact geometry, and the failure F1
    venv-genesis/bin/python numerics.py genesis-defaults the failure F7, in single precision (a process of its own)
    venv-usd/bin/python numerics.py usd                  the failure F10, in an environment with
                                                         mujoco-usd-converter (0.6.1 or 0.8.0; Python 3.12)
    python3 numerics.py --report                         validation.txt, failures.txt, record.txt, the data of
                                                         Figure S12.4 and the portability report (no simulator)
    python3 numerics.py --compare-with PATH              a rerun, in the code folder PATH (with ch11/results and
                                                         ch12/results), against the shipped results, under the
                                                         acceptable numerical variation of the record

`--check` compares every value of a part's summary with the shipped one. For `--compare-with`, copy the whole `code` folder elsewhere, rerun there what you want to reproduce (`mujoco_cart.py` in `ch11`, `portability.py` in `ch12`), and pass the path of the copy.

## Output (`results/`)

| File | Content |
|---|---|
| `<configuration>/summary.txt` | Every value of the run, by group. |
| `<configuration>/record.txt` | The record of the experiment. |
| `<configuration>/agreement.csv` | One row per period of the run of Chapter 3, with the columns of `../ch11/results/agreement-chapter3.csv` that the comparison shares, the end state (`p1`, `v1`, `l1`) and `load_jump`, the largest change of a wheel's load in one step. |
| `support-matrix.txt` | The support matrix of Section 12.1. |
| `against-mujoco.csv` | Per period, the norm of the state of the difference between two vectors, each configuration's error over the period (dp, dv, dl) and MuJoCo's in its own run of Section 11.1; every period starts afresh, so these are errors over one period, not whole runs (read by Figure 12.1). |
| `environment-newton.txt`, `environment-genesis.txt` | The complete Python environments (`pip freeze`). |
| `layers/<variant>/summary.txt`, `layers/<variant>/agreement.csv` | Section 12.2: each variant's record and results, and its errors per period to 17 significant digits (`mujoco-3.16.0`, `mujoco-3.15.0`, `genesis-f32`, `genesis-render`); the gap's effect on MuJoCo's contacts (`mujoco-3.16.0-gap`, `mujoco-3.15.0-gap`, summaries only). |
| `layers/comparisons.txt` | Section 12.2: the comparisons of Table S12.4. |
| `layers/mujoco-3.15.0-*/` | Section 12.2: MuJoCo 3.15.0 with each of the changes that Newton makes around it, and with all four (`mujoco-3.15.0-newton-changes`, whose `summary.txt` compares each with MuJoCo 3.15.0 and with Newton: Table S12.1). |
| `numerics/<part>/summary.txt` | Section 12.3: every value of each part, by group. |
| `numerics/steps/*.csv` | Section 12.3: the per-period comparison of each faithful configuration at 0.5 and 0.25 ms, to 17 significant digits (the runs at 1 ms are those of Sections 11.1 and 12.1). |
| `numerics/contacts-crest-*.csv` | Section 12.3: each contact of a wheel with the ground over the crest, against the exact geometry (Figure S12.4c). |
| `numerics/steps.csv`, `numerics/limits.csv` | Section 12.3: the data of Figures S12.4a and 12.5b. |
| `numerics/validation.txt`, `numerics/failures.txt`, `numerics/record.txt` | Section 12.3: the outcomes of the checks, the failures with their signatures, and the record's new entries. |
| `numerics/reproduction.txt` | Section 12.3: a rerun on a machine with another processor, under the same operating system, against the shipped results: the per-period files, the summaries and the records. |
| `portability-report.txt` | Section 12.3: the portability report: the support matrix, the comparisons, the checks, the failures and the record's new entries, with the releases and a hash of the code. |

## Releases and reproducibility

The results were produced on one x86-64 Linux machine, on the CPU, with the releases pinned in the requirements files; Newton's state is held in single precision. A second run of every configuration and of every variant of Section 12.2, in the same environment, reproduced every result in its summary (`--check`); only the times differed, which are those of this machine, with the ports' Python loops, and indicative only. Newton and Genesis change quickly; the support matrix records the releases that were tested, and the release of this code that the printed book cites keeps these pins, so that it remains executable when later releases of the simulators change their interfaces. A later release of the code adds rows to the matrix rather than replacing the tested ones.

The results of Sections 12.1 and 12.2 were produced on a machine whose record names an Intel Xeon processor at 2.10 GHz, except the variant `mujoco-newton-changes` of Section 12.2; that variant and the results of Section 12.3 were produced on one whose record names an Intel Xeon processor at 2.80 GHz. On the second machine, every configuration of the support matrix, rerun in its original environment, and the MuJoCo script of Section 11.1, rerun in a new environment built from `../ch11/requirements.txt`, reproduced the shipped results within the acceptable numerical variation of the record (`results/numerics/reproduction.txt`): every per-period file identical, byte for byte, except that of Genesis with the options of the text, whose error vectors differ by at most 1e-13 m; no value of a summary different; and in the records only the lines that name the processor, and for MuJoCo in its new environment NumPy, different.
