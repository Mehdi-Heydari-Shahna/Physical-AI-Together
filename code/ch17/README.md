# Chapter 17: contributions and independent verification

The path of a contribution, followed twice to the end, and the forms in which a contribution, a reproduction and a negative result are reported. `contrib/` holds two worked contributions, written against the public interface of the core (`../ch10/allovi`) and reusing the cart's modules (`../ch10/cart`) and the benchmark's (`../ch16/bench`): a control package, `reader.full-reverse`, and a demand predictor, `reader.early-window`. Both use helpers beyond the interfaces of the templates of Chapter 10. The demand predictor reads two constants of `cart.model`, `OPERATION_LEVEL` and `ce.EPS_MIN`, and the schedule of the cart's task `cart.operation` (`pending()`, `always`, `announce`, `predicted`), which it receives as the argument `task`; the control package extends `bench.constant-command` and uses `cart.control._Labels`, which is private, and the plans, the constants and the functions `brake_to_rest` and `backward_step` of the Chapter 3 example, which `cart.model` exposes as `ce`. `../ch10/templates/HELPERS.md` documents each of them, with examples, and how to run and check such a module: `../ch10/cart_checks.py --module` cannot take either file, because it creates each module without arguments and the demand predictor needs the task, and because the control package's file imports the benchmark's package `bench`, which is not on that script's path and whose modules it would otherwise try to check as well; `contribute.py` calls the check suite with a configuration of its own, as HELPERS.md shows. Their hypotheses were written down in `HYPOTHESES.md` before the first run of `contribute.py`, which applies each decision rule as written there. `contribute_diagnosis.py` was written after that run, to examine the hypotheses that failed and what no hypothesis stated; it is a diagnosis, not a test.

```
python3 contribute.py --check              # results/contribution-summary.txt, contribution-runs.csv
python3 contribute_diagnosis.py --check    # results/contribution-diagnosis.txt, contribution-leads.csv
```

Python 3.8 or later; no other packages. Both scripts import the core and the cart of Chapter 10 and the benchmark of Chapter 16 (`../ch16/cart_benchmark.py` and `../ch16/bench`), which imports the calibration of Section 7.3. With `--check`, each compares every value of its report with its `EXPECTED` and exits with an error if one differs. Both use up to two processes (`--jobs`); their results do not depend on the number. On the machine that wrote the shipped results (x86_64, two processors, CPython 3.13.17), `contribute.py` takes about seven minutes with one process, and `contribute_diagnosis.py` about four with two.

## What `contribute.py` does

For each contribution it

- registers the module through `allovi.register` and reports its role, name and contract;
- runs the check suite of Section 10.3 with the module in its role, in the configuration of the forecasting run of Chapter 3;
- runs the module in the world of Chapter 3: for the control package, the operation announced 21, 20, 19 and 18 periods before its onset; for the demand predictor, the schedule of Chapter 3;
- compares it with ALLOVI as Chapter 16 configured it (C6), replacing that one module, in 100 worlds that Chapter 16 did not use, seeds 16101 to 16200, drawn by `../ch16/bench/worlds.py`;
- applies the decision rules of F1 to F4 and E1 to E5, and reports every configuration over the valid worlds with exact intervals, and the paired differences from C6.

## Folders

| Folder | Content |
|---|---|
| `contrib/` | `full_reverse.py`: `bench.constant-command` of Chapter 16 with one more candidate while a requirement is pending, the full reverse of Section 9.3.2. `early_window.py`: the task plan's demand predictor with the requirement's onset 10 periods earlier than the forecast onset. Each docstring states what the module changes, its obligations and how each is established, and what it claims. |
| `cards/` | The method card's template (`TEMPLATE.md`, Section 17.2) and the cards of the two contributions. |
| `templates/` | The forms of a reproduction report (`REPRODUCTION.md`) and of a negative result (`NEGATIVE-RESULT.md`), Section 17.3. |

## Output (`results/`)

| File | Content |
|---|---|
| `contribution-summary.txt` | Every value of the contribution report, by group: registration, the check suite, the world of Chapter 3, the family, each configuration, the verdicts of the hypotheses, the paired differences. |
| `contribution-runs.csv` | One row per configuration and world: the world's draws, the arrival, the time in the work zone, the operation's period and delay, the overloads, the labels, the alarms, the restarts and the least true clearance. |
| `contribution-diagnosis.txt` | What the runs did where F4 and E4 failed, in F2's subset, in the early window's late runs and in the worlds that are not valid (written after the first run). |
| `contribution-leads.csv` | Figure S17.2: for every valid world, its lead time, its slip and the operation's delay under C6, the full reverse and the early window. |

## Numerical precision and reproducibility

As in Chapter 16: the cart's motion and every certified quantity are exact rational numbers, and random numbers come only from `random.Random(seed)` with integer seeds. Nothing in either script depends on the machine's speed.
