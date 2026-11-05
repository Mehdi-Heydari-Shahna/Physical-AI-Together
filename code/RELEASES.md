# Releases, citations and deprecations

This file states how the code of the book keeps the identity of its experiments as the software changes (Section 17.3). It is a policy, not a record: the releases themselves will be listed below as they are made.

## What identifies an experiment

An experiment is identified by
- the release of this repository that contains its script, and the command, with its options;
- the version of the contract (`CONTRACT` in `ch10/allovi/contract.py`, now 1.1) and the name and contract of every module of each configuration, which each record lists;
- the seeds of its worlds, which each record lists;
- the environment: the interpreter and every package, with versions, pinned where the script needs packages (for example `ch11/requirements.txt`, `ch12/requirements-*.txt`, `ch14/requirements.txt`);
- the summary file that its `--check` option verifies, shipped with the release.

The record that each script writes (`results/*record*.txt` or the record tables of the chapters) collects these items. Two runs with the same items give the same values; the README of each chapter states what may differ, for example wall-clock times, and Section S12.8 states the variation allowed on another machine.

## Versions

- **The contract.** A change to the methods of a role, to what they receive or return, or to their meaning, changes the major version (2.0); an addition that leaves every existing module valid, such as a new optional method, changes the minor version (1.1). The registry refuses a module whose contract has another major version.
- **Modules.** A module's `NAME` is never reused for a different behaviour. A module whose behaviour changes is registered under a new name, or with a version in its name, and its method card records the change.
- **Results.** A later release adds results beside the earlier ones instead of replacing them: a new summary file, a new row of a support matrix, a new report. A result that a later release corrects stays in the release that produced it, and the correction says what changed and why.

## Deprecation

A module or script that is to be removed is marked deprecated, in its docstring and in the notes of the release that deprecates it, with the release in which it will be removed, and it stays in at least that release and the next. Its results remain citable through the releases that contain them.

## Citations

Cite the release that produced the results, by its version and its archive identifier (for v1.0, the DOI below); for a contributed module, cite also its method card. `../CITATION.cff` gives the citation of the book and its code.

## Releases

- **v1.0**: the code and the results of the book's version of 31 October 2026, with the online supplement, and the videos of Chapters 13 to 15 attached to the release. Archived at Zenodo, without the videos, on 31 October 2026: DOI [10.5281/zenodo.23823947](https://doi.org/10.5281/zenodo.23823947), the file `Physical-AI-Together-1.0.zip`, whose MD5 checksum the record lists as `ad669a6af7fc2f514c8c43182628e717`. The DOI [10.5281/zenodo.23823946](https://doi.org/10.5281/zenodo.23823946) names every version of the archive and resolves to the latest one.
- **v1.1**: two checks of the alarm rule of Algorithm 8.1 that v1.0 did not make, which the book lists as the known issues of v1.0 (Sections 10.1 and 17.3):
  - a measured distance that is not a finite number, such as NaN, raises the alarm (`ch10/allovi/loop.py`, `within_bound`; `ch16/complete/realtime.py`);
  - a restart plan takes over only if the measurement taken when it is committed passes the check against the set on which it was certified (the same two files).
  Tests of both, `ch10/tests/test_alarm_restart.py`. The second check changes one result of the code: the run of the configuration of Chapter 3 in Newton's XPBD port, which admitted 71 restarts with v1.0, admits none with v1.1; its summary and record, the support matrix and the portability report are those of v1.1, and `ch12/results/v1.0/` keeps those of v1.0 (`ch12/README.md`). Every other `--check` of a script of Chapters 10 to 17 that runs the core's loop gave the values of v1.0 with v1.1, but those of the configurations of Genesis, which were not rerun and which neither check can change (`ch12/README.md`), and those that need a GPU. The complete case's scripts check the files that its frozen record names, which include the two changed files, so the case runs from tag v1.0; outside the protocol, `ch16/complete_recheck.py` reran its offline phases with v1.1 (`ch16/complete/DEVIATIONS.md`, deviation 9). Scripts: `ch16/paired_statistics.py`, Listing S16.1 of the supplement as a file; `ch16/complete_compare.py`, which compares a rerun of the complete case's evaluation with the shipped records; and `ch16/complete_recheck.py`. A failed check of `ch12/portability.py`, `layers.py` or `numerics.py` keeps the shipped summary and writes its own beside it. Documentation: the book's five authors in `CITATION.cff` and the README, the supplement kept out of the code's archive (`.gitattributes`), `NUMBERING.md` for the book's version of 4 November 2026, NumPy and SciPy at the versions of the records in the requirements of Chapter 15, and two entries of the log of deviations of the complete case. The results that v1.0 produced are those that the book reports and stay in v1.0.
