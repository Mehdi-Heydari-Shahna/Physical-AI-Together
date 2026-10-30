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

Cite the release that produced the results, by its version and, once one is assigned, its archive identifier; for a contributed module, cite also its method card. `../CITATION.cff` gives the citation of the book and its code.

## Releases

- **v1.0**: the code and the results of the book's version of 31 October 2026, with the online supplement, and the videos of Chapters 13 to 15 attached to the release. It has no archive identifier yet; one will be added here when it is assigned.
