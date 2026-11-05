# Physical AI, Together: code and data

This repository holds the code and data of the book *Physical AI, Together: The ALLOVI Governor: Contracts and Conditional Guarantees* by Mehdi Heydari Shahna, Soyi Jung, Soohyun Park, Joongheon Kim and Jouni Mattila, with the book's online supplement. Every number that the book quotes from a computation is produced by a script here, and the results that it quotes are shipped in each chapter's `results/` folder. Most scripts have a `--check` option that compares what they compute with the values that the book quotes.

The book's file paths, such as `code/ch03/cart_example.py`, are paths in this repository.

## Contents

| Folder | Chapter of the book | What it needs |
|---|---|---|
| `code/ch03` | 3. The smallest complete example | Python 3.8 or later, no other packages |
| `code/ch05` | 5. How a control researcher contributes | Python 3.8 or later, no other packages |
| `code/ch06` | 6. How a learning researcher contributes | Python 3.8 or later, no other packages; `published/` trains the published SAC, TD-MPC2 and Diffusion Policy through the same adapters (PyTorch, and an NVIDIA GPU for TD-MPC2 and Diffusion Policy; see its README) |
| `code/ch07` | 7. Predicting capability and demand | Python 3.8 or later, no other packages |
| `code/ch08` | 8. The ALLOVI governor | Python 3.8 or later, no other packages |
| `code/ch09` | 9. What the complete system can guarantee | Python 3.8 or later, no other packages |
| `code/ch10` | 10. An architecture researchers can extend: the package `allovi`, the cart's modules, templates and the check suite | Python 3.8 or later, no other packages |
| `code/ch11` | 11. MuJoCo and Isaac Sim workflows | MuJoCo 3.16.0 (`requirements.txt`); Isaac Sim 6.2.0 and an NVIDIA GPU for Section 11.2 |
| `code/ch12` | 12. Newton, Genesis and numerical credibility | one environment per simulator (`requirements-newton.txt`, `requirements-genesis.txt`); the support matrix needs Python only |
| `code/ch13` | 13. Wheeled robots and the Husky | MuJoCo 3.16.0 and NumPy (`requirements.txt`); Isaac Sim 6.2.0 for `isaac_husky.py` |
| `code/ch14` | 14. Arms and heavy-duty mobile manipulators | MuJoCo 3.16.0, NumPy, PyYAML and SciPy (`requirements.txt`) |
| `code/ch15` | 15. A governed quadruped, and legged robots beyond it | MuJoCo 3.16.0, NumPy and SciPy (`requirements.txt`); for the humanoid, a checkout of [Team-Impedance-Control](https://github.com/Mehdi-Heydari-Shahna/Team-Impedance-Control) |
| `code/ch16` | 16. Comparisons that identify the contribution | Python 3.8 or later, no other packages; the complete case of the cart in MuJoCo needs MuJoCo 3.16.0 and NumPy |
| `code/ch17` | 17. Contributions and independent verification | Python 3.8 or later, no other packages |
| `code/render` | The videos and stills of the MuJoCo runs (Chapters 1, 11 and 13 to 15), and their path-traced versions (`render/cycles`, Blender Cycles on a GPU) | Pillow, SciPy, EGL and ffmpeg besides each chapter's requirements; Blender 4.2 or later for `render/cycles` |
| `docs` | The online supplement (PDF) | |

Chapters 1, 2, 4 and 18 have no code of their own. Each folder's `README.md` says what its scripts compute, how long they take, what they write into `results/`, and with which versions the shipped results were produced. The comments, docstrings, READMEs and printed results of the code cite the book's sections, tables and figures as the book's version of 31 October 2026 numbers them; `code/NUMBERING.md` gives the current number of every item whose number has changed since.

## Running a chapter

Get a copy of the repository:

```
git clone https://github.com/Mehdi-Heydari-Shahna/Physical-AI-Together.git
cd Physical-AI-Together
```

The examples of Chapters 3 to 10, 16 and 17 run with Python alone:

```
cd code/ch03
python3 cart_example.py            # writes results/
python3 cart_example.py --check    # also compares every value with the book
```

The tests of the core's alarm and restart checks run with Python alone:

```
cd code/ch10
python3 -m unittest discover -s tests -v
```

The simulator chapters use one virtual environment each:

```
cd code/ch13
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python husky_mujoco.py --check
```

On Windows, write `python` for `python3` and `.venv\Scripts\python` for `.venv/bin/python`. With Miniforge, `conda env create -f environment.yml` and `conda activate physical-ai-together` give Python 3.12, which is all that the Python-only chapters need, with Git and the GitHub CLI.

The results that the book quotes are shipped in each chapter's `results/` folder, so they can be read and checked without running anything. Scripts that need a GPU (Isaac Sim in Chapters 11 and 13, the published learners in `code/ch06/published`, and `code/render/cycles`) say so in their folder's README; the results of the Isaac Sim runs and of the published learners are shipped.

## The robots' descriptions

The Husky A300, the UR5e and the Go2 are built from their makers' published descriptions. The files that the scripts need are copied into this repository, unchanged, with their licences (`THIRD_PARTY_NOTICES.md`). The videos and some stills draw the makers' visual meshes, which are not copied: they are read from checkouts of the makers' repositories, at the commits that `code/render/README.md` lists.

## Videos

The videos of the simulated runs in Chapters 13 to 15 are not stored in the repository's history, for their size. They are attached to the release [v1.0](https://github.com/Mehdi-Heydari-Shahna/Physical-AI-Together/releases/tag/v1.0), and the book links to them from their figures: `husky-map.mp4` (Chapter 13), `ur5e-stops.mp4` and `hdmm-reference.mp4` (Chapter 14), `go2-pushed.mp4` and `apollo-carry.mp4` (Chapter 15). `excavator-onset.mp4` shows the excavator of `code/ch14`, which the book does not report. The scripts that draw them are in `code/render` and in the chapters' folders.

## The supplement

`docs/Physical_AI_Together_supplement.pdf` is the book's online supplement, in the book's version of 31 October 2026, whose numbering the code cites (`code/NUMBERING.md`): detailed run records, provenance, release-specific software instructions, diagnostic tables and implementation examples (its chapters, sections, tables and figures are numbered with an `S`, Chapter S13 extending Chapter 13). It is part of the book and is published with it; it is not part of the code's archive (`.gitattributes`). The instructor's manual, with the solutions of the book's exercises, is distributed to instructors separately.

## Reproducibility, releases and citation

`REPRODUCING.md` says how to reproduce every chapter's results from a fresh copy and how to report a reproduction; `CONTRIBUTING.md` says how to contribute a module, and `code/ch10/templates/HELPERS.md` documents the helpers of the cart's code that a contribution may reuse. `code/RELEASES.md` states how an experiment is identified across versions of this code. `CITATION.cff` gives the citation of the book and its code. Cite the release that produced the results you use, or the commit. The book reports the results of release v1.0, archived at Zenodo: DOI [10.5281/zenodo.23823947](https://doi.org/10.5281/zenodo.23823947). Release v1.1 adds the two checks that the book lists as known issues of v1.0, a test of each, and scripts and documentation. With them, one result of the code changes: the run of Chapter 3 in Newton's XPBD port admits none of the restarts that v1.0 admitted (`code/ch12/README.md`). The complete case of Chapter 16, whose frozen record names the files of v1.0, runs from tag v1.0 (`code/RELEASES.md`, `REPRODUCING.md`).

## Licence

The code and the results written for the book are under the MIT License (`LICENSE`). The files of others that this repository contains keep their own licences (`THIRD_PARTY_NOTICES.md`). The online supplement in `docs/` is not under the MIT License: copyright Mehdi Heydari Shahna, all rights reserved.
