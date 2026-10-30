---
name: Contribution
about: Propose a module for ALLOVI, and register its hypotheses before the first run of its comparison.
title: "Contribution: <NAME> (<role>)"
labels: contribution
assignees: ''
---

<!--
The path of a contribution is in CONTRIBUTING.md and in Chapter 17 of the book.
Open this issue before the first run of the comparison of step 7: its date is
then a date of the hypotheses that others can verify. Add the results later,
in a comment or in the pull request; do not edit the hypotheses below after
the first run, but list any change under "Deviations" with its date and reason.
-->

## Identity

- **NAME and role:** <!-- e.g. yourname.method, role control / proposal / capability / demand / governor variant / robot / world / task / evaluator -->
- **Contract** (`CONTRACT`) and the commit of this repository that the module was built on:
- **Authors and contact:**
- **What it changes**, compared with the module that it replaces, in the terms of the contract:
- **Dependencies beyond the templates:** every module, helper or constant that it uses beyond the interfaces of the templates, with where each is documented (`code/ch10/templates/HELPERS.md` for the cart's and the core's):

## Semantics

<!-- Every input and output with its quantity, unit, frame, sign, period and
resolution; for a proposal its horizon, for a label its region and worlds, for
a requirement its level and onset (Section 17.2.1 of the book). -->

## Obligations and how each is established

<!-- Each obligation of the role (Table 4.1 of the book, and the docstring of
the template), one per line, with how it is established: by a proof (where), by
evidence (what, on which worlds), by the specification of the hardware, or
"inherited unchanged from <module>". A proposal method or a demand forecast has
none for the guarantees; a variant of the governor proves each guarantee that
it claims first. -->

## The check suite (step 5)

<!-- The command, and the report it printed: `python3 cart_checks.py --module FILE`
in code/ch10, or the call of allovi.checks.check of HELPERS.md, section 11. -->

```
```

## Hypotheses (step 6), registered before the first run

<!-- Each hypothesis with its decision rule, the subset of worlds to which the
rule applies, and what an empty subset means. -->

- **Comparator**, which differs from the contribution in this module only:
- **Task and family of worlds**, with the rules that draw them and the seeds (fresh: not 16001 to 16200 for the cart's operation):
- **Date of this registration:**

## Deviations

<!-- After the first run: every change to a hypothesis, a rule, a configuration,
a module or a world, with its date and its reason; "none" otherwise. -->

## After the runs

- **Example:** a runnable configuration, with the command and what it prints:
- **Results:** every configuration with exact intervals; the verdict of every hypothesis, the failed ones included; every failed run:
- **Measured limitations:** where it was found not to help, or to hurt; the negative results (`code/ch17/templates/NEGATIVE-RESULT.md`), with each diagnosis marked as found after the run:
- **Method card** (`code/ch17/cards/TEMPLATE.md`):
- **Reproducibility:** the commit, the environment with versions, the running time, and the command whose `--check` confirms every reported value, with the `[check]` line that it printed:
- **Provenance:** every asset, data set and model, with its source, version and licence:
- **Reproduced by someone outside your team?** <!-- no, or a link to the reproduction report -->
