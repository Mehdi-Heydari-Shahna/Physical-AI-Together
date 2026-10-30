---
name: Reproduction report
about: Report an attempt to reproduce a result of the book or of a contribution, whether every value agreed or not.
title: "Reproduction: <script> (<section of the book or contribution>)"
labels: reproduction
assignees: ''
---

<!--
The fields of code/ch17/templates/REPRODUCTION.md (Section 17.3.1 of the book).
REPRODUCING.md gives the commands of every chapter and the line that each
check prints when every value is as quoted. A reproduction in which values
differ is as useful as one in which every value agrees: report it the same way.
-->

## The experiment

- **Script, options and section of the book:** <!-- e.g. code/ch16/cart_benchmark.py --check, Sections 16.1 to 16.3 -->
- **Summary file reproduced:** <!-- e.g. code/ch16/results/benchmark-summary.txt -->
- **Release or commit used** (`git rev-parse HEAD`, or the archive's identifier), and the release that produced the original results:

## Who, and to which degree

- **Reproduced by:** <!-- name and affiliation -->
- **Outside the team that produced the results?** <!-- yes / no -->
- **Degrees reached** (Section 17.3.1; tick each that applies):
  - [ ] 1. the script's own check, rerun where it was run
  - [ ] 2. a run in a new environment built from the record's pins
  - [ ] 3. a run on another machine or operating system, within the variation that the record allows
  - [ ] 4. a run by someone outside the team that produced the results
- **Date:**

## Environment

<!-- Machine, processor, operating system, interpreter and every package, with
versions (python3 --version; pip freeze in a virtual environment); how it
differs from the record that the chapter's README names. -->

## Commands

<!-- Exactly as run, from a fresh copy of the release. -->

```
```

## Outcome

- **`[check]` line printed by each script**, copied as printed:

```
```

- **Values compared**, and how (the script's `--check`, or by hand):
- **Agreement:** values identical; values within the variation that the record allows (Section S12.8); values outside it, each with the original and the reproduced value:
- **Files compared byte for byte** (for example `diff -rq code-shipped/chNN/results code/chNN/results`), and which differ:
- **Differences explained:** for each difference, its cause if you found one (a package version, the processor, the operating system, the platform's mathematical library), and how you found it:
- **Running time** on your machine:
- **Verdict:** <!-- reproduced / reproduced within the stated variation / not reproduced -->

## What was not reproduced

<!-- The parts of the experiment that you did not run, and why (hardware, licences, running time). -->

## Contact

<!-- How the authors of the original results can reach you. -->
