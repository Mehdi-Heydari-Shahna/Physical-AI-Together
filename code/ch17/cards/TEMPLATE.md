# Method card: `<NAME>`

A method card says what a module is, what it promises, and on what evidence, so that another researcher can use it, check it and compare with it without asking its author. Fill in every field; write "none" or "not established" where that is the truth. Section 17.2 explains each field, and `reader.full-reverse.md` and `reader.early-window.md` are filled-in examples.

## 1. Identity

| Field | Entry |
|---|---|
| Name and role | `<NAME>`, role `<control / proposal / capability / demand / governor variant / robot / world / task / evaluator>` |
| Contract | the version of the contract it implements (`CONTRACT`), and the core's release it was checked with |
| Version and date | the module's own version, and the date of this card |
| Authors and contact | who wrote it, and how to reach them |
| Dependencies beyond the templates | every module, helper or constant that it uses beyond the interfaces of the templates of Chapter 10, each with where it is documented (`../../ch10/templates/HELPERS.md` for the cart's and the core's), so that a reader without the authors' code knows what to supply |
| Code | the file, and the release or commit that contains it |
| Licence | of the code and of every asset it uses |

## 2. What it does

One paragraph, in the terms of the contract: what the module receives, what it returns, and what it changes compared with the module it replaces.

## 3. Semantics

Every input and output with its physical quantity, unit, frame, sign and period; for a proposal, its horizon and how the generator realizes it; for a label, the region, the period and the worlds it admits; for a requirement, its level, its onset and the operation it serves. State the resolution of every number that the module rounds.

## 4. Obligations and how each is established

The obligations of the module's row of the contract (Table 4.1), one per line, each with how it is established: by a proof (where), by evidence (what, on which worlds), by the specification of the hardware, or "inherited unchanged from `<module>`". An obligation that is not met is listed as not met, with what follows from it.

## 5. Assumptions and scope

The robots, worlds, ranges of parameters and operating conditions for which the obligations are established; what the module assumes of the other modules; what is outside its scope.

## 6. A runnable example

The command that runs the module in a reference configuration, and what it prints.

## 7. Evidence

- **Checks.** The report of the check suite (Section 10.3) with the module in its role, and what a passed check does not establish.
- **Hypotheses.** Where they were registered, and when, before the first run; their decision rules.
- **Comparison.** The configuration it is compared with, the one module that differs, the family of worlds and its seeds, and the budget.
- **Results.** Every outcome with an interval, and the verdict of every hypothesis, the failed ones included.
- **Failed runs.** Every run that failed, with what went wrong, or the file that lists them.

## 8. Measured limitations

Where the module was found not to help, or to hurt; the negative results, with their diagnosis marked as found after the runs.

## 9. Reproducibility

The release of the code, the environment (interpreter and every package, with versions), the seeds, the running time and the machine, and the command that checks every value of the report (`--check`). What a reproduction may differ by.

## 10. Provenance

Every asset, data set and model it uses, with its source, version and licence; what is the authors' own.

## 11. Status

The evidence label of each claim (established certificate, empirical demonstration or research extension; Section 15.3), and whether the results have been reproduced by someone outside the team that wrote the module: by whom, when, and where the reproduction report is.

## 12. How to cite

The citation of the module and of the release that contains it.
