# Reproduction report: `<experiment>`

A reproduction report records an attempt to obtain the results of an experiment of this book, or of a contribution to it, from its record and its code, by someone who did not produce them. Section 17.3 explains the fields. A failed reproduction is reported in the same form; it is as useful as a successful one. It can be filed as an issue of the book's repository, with the form `reproduction-report`, or added beside the experiment's results in a pull request; the repository's `REPRODUCING.md` gives the commands of every chapter and what each check prints.

| Field | Entry |
|---|---|
| Experiment | the script, its options, the section of the book and the summary file reproduced |
| Release | the release or commit of the code used, and the release that produced the original results |
| Reproduced by | name, affiliation, and whether outside the team that produced the results |
| Degree | the degrees of Section 17.3.1 that the reproduction reached: 1, the script's own check, rerun where it was run; 2, a run in a new environment built from the record's pins; 3, a run on another machine or operating system, within the variation that the record allows; 4, a run by someone outside the team that produced the results |
| Date | of the reproduction |
| Environment | machine, processor, operating system, interpreter and every package, with versions; how it differs from the record |
| Commands | exactly as run, from a fresh copy of the release |
| Check line | the last line that each `--check` printed, copied as printed |
| Running time | on this machine |

## Outcome

- **Values compared.** How many values of the summary were compared, and how (the script's `--check`, or by hand).
- **Agreement.** Values identical; values within the variation that the record allows (Section S12.8); values outside it, each listed with the original and the reproduced value.
- **Files compared.** Which result files were compared byte for byte, and which differ.
- **Differences explained.** For every difference: the cause, if found (a package version, the processor, the operating system, the platform's mathematical library), and how it was found.
- **Verdict.** Reproduced, reproduced within the stated variation, or not reproduced.

## What was not reproduced

The parts of the experiment that were not run, and why (hardware, licences, running time).

## Contact

How the authors of the original results can reach the reproducer.
