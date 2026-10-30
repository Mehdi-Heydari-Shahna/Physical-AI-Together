# Contributing a module

A contribution to ALLOVI is one module in one role: a control method, a proposal method, a capability forecast, a demand forecast, or a variant of the governor (Chapter 17 of the book). This file takes a module of your own along the ten steps of the path of a contribution (Figure S17.1 of the supplement), with the files and commands of this repository, and says how to submit it. What a contribution may claim follows from the module that it changes. A proposal method or a demand forecast cannot weaken a guarantee and claims performance only; a control method or a capability forecast must first establish its obligations; and a variant of the governor inherits no guarantee and must prove each one again (Section 17.1).

The code is under the MIT License (`LICENSE`), and a contribution is accepted under the same licence.

## Before you start

- Read Chapters 4 and 10 of the book for the contract and the core, and Chapter 17 for the path; Table 4.1 lists the obligations of each role.
- Work in a fresh copy of the repository, at the release or commit that you build on, and note it (`git rev-parse HEAD`).
- The code of Chapters 3 to 10, 16 and 17 needs Python 3.8 or later and no other package. The cart's quantities are exact, `fractions.Fraction` (`code/ch10/templates/HELPERS.md`, section 2), and a module of the cart should keep them so. If your module needs a package, pin it in a requirements file in your folder, as the simulator chapters do.
- `REPRODUCING.md` says how to run every check of the book; run those of Chapters 10 and 17 once before you change anything, so that you know what they print on your machine.
- From the top of the repository, `python3 -m doctest code/ch10/templates/HELPERS.md` runs every example of the helpers' documentation and prints nothing if each holds.

## The ten steps

**1. Choose the role.** The module that the contribution changes decides what it must establish and what it may claim (Table 17.1 of the book):

| Path | Template | What it must establish | What it may claim |
|---|---|---|---|
| Control method | `code/ch10/templates/control_package.py` | each obligation of its row of Table 4.1: a sound test, the rest segment, its feedback and reach bound, the initial plan, by proof or by measured bounds | certificates under the stated assumptions; performance as measured |
| Proposal method | `code/ch10/templates/proposal_method.py` | nothing for the guarantees; the meaning of its proposals | performance only |
| Capability forecast | `code/ch10/templates/capability_predictor.py` | labels at least as informative as the global capability, and how often they are invalid, on worlds not used to issue or calibrate them | a bound on miscoverage under its assumptions; performance as measured |
| Demand forecast | `code/ch10/templates/demand_predictor.py` | nothing for the guarantees, within operations that can be postponed | performance only |
| Governor variant | a variant of `code/ch10/allovi` | a proof of every guarantee that it claims, before anything else | what it proves again; nothing is inherited |

The other roles, a robot adapter, a simulator adapter, a task and an evaluator, have templates in the same folder.

**2. Copy the template** of the role into a folder of your own (below), rename the class, choose a `NAME` such as `yourname.method`, and fill in every method. The docstrings of the template say what each method receives and returns, and `code/ch10/allovi/contract.py` is the reference. `code/ch10/templates/HELPERS.md` documents, with examples that run as tests, what a module of the cart may reuse beyond the templates: its units and constants, its state and proposals, its task, its segments and plans, the functions of its generator, and its control package, with how to extend it.

**3. Register it** with `@register`, from `allovi`. Registration checks the role, the name, the version of the contract and every method with its number of positional parameters, and refuses the module with a message that names what is missing. A `NAME` is unique within its role and is never reused for a different behaviour (`code/RELEASES.md`).

**4. Establish its obligations.** Write in the module's docstring each obligation of the role, as the template lists it, and how it is established: by a proof (say where), by evidence (say what, on which worlds), by the specification of the hardware, or "inherited unchanged from" a named module. The docstrings of `code/ch17/contrib/full_reverse.py` and `early_window.py` are examples. No check of the software discharges an obligation (Section S10.10.1).

**5. Run the check suite** with the module in its role in the cart's reference configuration, the forecasting run of Chapter 3. From `code/ch10`:

```
python3 cart_checks.py --module path/to/your_module.py
```

prints the report of the 13 checks for each module that the file registers. The command creates each module without arguments and imports the file with only `code/ch10` and `code/ch03` on the path. A module whose constructor needs an argument, such as a demand predictor that reads the cart's task, a file that imports another package, such as the benchmark's `bench`, a task, and a module of another robot are checked instead by calling `allovi.checks.check` with a function that builds the configuration; `code/ch10/templates/HELPERS.md` (section 11) gives the call and what it prints. A passed check shows only that the cases run revealed no mistake; a failed one shows a mistake, or an obligation that does not hold in those cases.

**6. Register the hypotheses** before any run of the comparison. Write each in a file `HYPOTHESES.md` in your folder, dated, with its decision rule, the subset of worlds to which the rule applies, what an empty subset means, the comparator and the family of worlds with its seeds; `code/ch17/HYPOTHESES.md` and `code/ch16/HYPOTHESES.md` are examples. Make the date one that others can verify: commit the file and push it to a public fork, or paste it into the issue that opens the contribution (form **Contribution**), before the first run. After the first run, change no hypothesis, rule, configuration, module or world; if something must change, record it under *Deviations* at the end of the file, with the date and the reason, as those two files do.

**7. Run a matched comparison**: your module replaced in a configuration and nothing else, on a documented task (Table S17.1 of the supplement), in worlds that did not suggest the hypotheses. For the cart's operation, start from a copy of `code/ch17/contribute.py`: `benchmark_configuration` builds ALLOVI as Chapter 16 configured it, C6, with one module replaced; `run` runs one configuration in one world; and `hypotheses` applies each decision rule through the class `Rule`. Draw fresh seeds with the rules of `code/ch16/bench/worlds.py`: seeds 16001 to 16100 are the benchmark's, and 16101 to 16200 those of Chapter 17. Report proportions and medians with the exact intervals of `code/ch16/bench/stats.py`. Give the script a `--check` option and a table of expected values, filled from the first run, so that a second run can confirm every value; Section 17.1.2 and `HELPERS.md` (section 12) say more.

**8. Report every outcome**: every configuration with its intervals, the verdict of every hypothesis, the failed ones included, and every failed run. Mark each diagnosis as found after the run. For each hypothesis that its rule does not support, fill in `code/ch17/templates/NEGATIVE-RESULT.md`.

**9. Write the method card** from `code/ch17/cards/TEMPLATE.md`, with `reader.full-reverse.md` and `reader.early-window.md` beside it as examples. Fill in every field, and write "none" or "not established" where that is the truth. Its field "Dependencies beyond the templates" names every helper that the module uses, with where it is documented.

**10. Invite a reproduction** by someone outside your team, with `REPRODUCING.md` and the form `code/ch17/templates/REPRODUCTION.md`.

## Submitting it as a pull request

Put the contribution in a folder of its own, `contributions/<NAME>/`, for example `contributions/yourname.method/`, with

- the module, in a file of its own, registered, with its obligations in its docstring;
- `README.md`: what the module is, the commands of steps 5 and 7, how long each takes, and the `[check]` line that each prints;
- `HYPOTHESES.md`, as registered in step 6, with its deviations;
- the comparison script of step 7, and in `results/` what it wrote, with the report of the check suite of step 5;
- the method card, `<NAME>.md`, and one filled `NEGATIVE-RESULT.md` for each hypothesis that its rule did not support.

A pull request adds; it does not change what the book reports. It changes no file of `code/ch03` to `code/ch17` and no shipped result: a later release adds results beside the earlier ones (`code/RELEASES.md`). A variant of the governor is the exception, since it changes the core; it says so, and comes with its proofs (Section 17.1.3). Before you open the pull request, run the checks of Chapter 10 and Chapter 17 in your copy, to show that the modules of the book still decide as they did:

```
cd code/ch10
python3 cart_checks.py --check       # [check] 45 values quoted in Section 10.3: all as quoted
cd ../ch17
python3 contribute.py --check        # [check] 45 values: all as expected
```

In the description of the pull request, link the issue that registered the hypotheses, give the commit on which the contribution was built, and go through the checklist of Section 17.2 of the book, saying where the contribution states each item:

- **Semantics**: every input and output with its quantity, unit, frame, sign, period and resolution;
- **Assumptions and obligations**: each obligation of the role and how it is established;
- **Examples**: a runnable configuration, with the command and what it prints;
- **Reproducibility**: the commit, the environment with versions, the seeds, the running time, and the command that checks every reported value;
- **Provenance**: every asset, data set and model, with its source, version and licence;
- **Measured limitations**: where it did not help or hurt, every failed run, the hypotheses that failed;
- **Evaluation**: hypotheses registered before the runs, with a verifiable date and a rule for an empty subset; a comparator that differs in the contributed module only; an open family of worlds; every outcome with an interval.

A review reruns the commands of the folder's README with `--check`, reads the obligations and their evidence, compares the date of the hypotheses with the first run, and may ask for a reproduction before the contribution is cited as reproduced. Chapter 17 of the book will report the first contribution made outside the team.

## Reporting a reproduction or a problem

To report a reproduction of a result of the book or of a contribution, successful or not, use `REPRODUCING.md` and the issue form **Reproduction report**. To report a mistake in the code or in the book, open an issue that names the file, the command and what it printed.
