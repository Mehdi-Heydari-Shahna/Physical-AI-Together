# Method card: `reader.early-window`

A filled-in example of `TEMPLATE.md` (Section 17.2). Every result below is in `../results/contribution-summary.txt`, `../results/contribution-diagnosis.txt` or `../results/contribution-leads.csv`, and `contribute.py --check` and `contribute_diagnosis.py --check` verify the first two; the back-away's lengths are those of Section 9.3.2, and the running time is that of `../README.md`.

## 1. Identity

| Field | Entry |
|---|---|
| Name and role | `reader.early-window`, role `demand` |
| Contract | 1.0; checked with the core of Chapter 10 (`../../ch10/allovi`) of the book's repository, before its first release |
| Version and date | the first version; this card, 25 October 2026, with its dependencies completed on 27 October 2026 |
| Authors and contact | the author of the book, as a worked example; contact through the book's repository once it is published |
| Dependencies beyond the templates | the constants `OPERATION_LEVEL` and `ce.EPS_MIN` of `cart.model`, and the schedule of the cart's task `cart.operation` (`pending()`, `always`, `announce`, `predicted`), which it receives as the argument `task`; each is documented, with examples, in `../../ch10/templates/HELPERS.md` |
| Code | `../contrib/early_window.py`, in the book's repository before its first release; a citation names the commit |
| Licence | not yet stated; it will be given with the first release (`../../RELEASES.md`) |

## 2. What it does

The task plan's demand predictor `cart.task-plan` of Chapter 10, with the requirement's onset moved `early` periods (here 10) before the onset that the task plan forecasts: the earliest that the task plan states the operation may come. Once that onset has passed, it requires the reserve from the first pair on, as `cart.task-plan` does after its forecast onset; before the announcement, and once the operation has begun, it requires the floor.

## 3. Semantics

A requirement is the operation's reserve, 1.3 m, from an onset pair on: during period k, the pair `max(predicted − early − k − 1, 0)` of the candidates of that period, where `predicted` is the task plan's forecast onset, updated by the task plan when it postpones the operation. The reserve is the clearance from the front to the wall, in metres; a pair lasts one period of 0.1 s. The window applies to every onset that the task plan forecasts, a postponed one included. Nothing is rounded.

## 4. Obligations and how each is established

None for the guarantees. A requirement never enters the test (Section 4.3.1); a wrong requirement changes when the robot prepares, never whether its plans are certified, within the scope of operations that can be postponed without harm (Table 2.1). By Corollary 9.13 (`cor:early-onset`), a reserve held at the early onset is kept until the operation is due.

## 5. Assumptions and scope

The cart's operation of the anticipation benchmark, whose task plan states that the operation may come up to 10 periods before or after its forecast onset, and the postponement rule of Section 8.2.3, which announces a postponed onset 21 periods ahead. It has been run with no other task.

## 6. A runnable example

`python3 contribute.py` runs it with the schedule of Chapter 3 and in the family 16101 to 16200; it prints, among the rest:

```
early-window, the schedule of Chapter 3: begun; the back-away from period; periods in the work zone after the announcement = 200; 170; 45
```

## 7. Evidence

- **Checks.** The check suite of Section 10.3, with the predictor in its role in the configuration of the forecasting run of Chapter 3: 13 passed, 0 failed, 0 not applicable. A passed check discharges no obligation; this module has none. `contribute.py` runs the suite with the task passed to the predictor; `../../ch10/cart_checks.py --module` cannot create the predictor without its task, and `../../ch10/templates/HELPERS.md` gives the call of the suite that runs it alone.
- **Hypotheses.** E1 to E5 of `../HYPOTHESES.md`, written on 25 October 2026 before the first run of `contribute.py`; every decision rule supports a hypothesis only if it holds without exception in its subset.
- **Comparison.** C6 of Chapter 16, with the demand predictor replaced and nothing else; seeds 16101 to 16200, by `../../ch16/bench/worlds.py`; every fresh candidate in every period.
- **Results.** Over the 95 valid worlds: the operation on time in 71 (64.7 to 83.1 %), against C6's 69 (62.5 to 81.3 %); begun earlier than C6 in 11, later in 24, in the same period in 60; late in 24, by a median 6.3 s, against C6's 26 by 2.1 s; median arrival 8.9 s, as C6; median time in the work zone 12.2 s (11.7 to 12.9 s), against 13.0 s. Over all 100: 5 runs with an invalid executed label, 5 alarms, 2 restarts, no violation, as C6. E1 supported (95 cases), E2 supported (11), E3 supported (47), E4 not supported (11 cases, 4 exceptions), E5 supported (100).
- **Failed runs.** `../results/contribution-runs.csv` lists every run. The five with an invalid executed label lie in the five worlds that are not valid; each raised an alarm, two restarted, and in the other three (worlds 16116, 16120 and 16144) no restart could be certified, the true clearance fell to 0.48 or 0.49 m, below the floor, without a violation, and the operation never began, exactly as with C6.

## 8. Measured limitations

Found after the run, by `contribute_diagnosis.py`:

- **A window that cannot be met (E4).** With a lead time of 21 to 26 periods and no slip, the window's onset arrives while the back-away is still accelerating; the requirement then applies from the first pair on, the first key of the selection ties, the proposal "hold" decides, and the cart brakes to rest beyond 23.7 m and waits there through the onset, which it misses. With 27 to 30 periods, the tie comes once the back-away's 16 periods of acceleration and its period of commitment are over; "hold" then brakes as the back-away would, and the cart comes to rest at or before 23.7 m in time. E4 expected lateness in all of 21 to 30. The boundary between 26 and 27 periods is that of these runs; that it holds elsewhere is a new hypothesis. Two more worlds, with lead times of 22 and 21 periods and onsets 2 and 1 periods later than forecast, outside E4's subset, went the same way.
- **Postponed onsets (no hypothesis).** In the 15 valid worlds with a lead time of 20 periods or fewer, the early window begins the operation later than C6 in all 15; in the 12 without a slip it is postponed three times where C6 is postponed once. The task plan announces a postponed onset 21 periods ahead, and the window moves the requirement to 11 periods after the postponement, too few for a back-away from rest, so the same tie stops the cart again. A predictor that moved only the first forecast, and not a postponed onset, which is not uncertain, would avoid this; it is untested.

## 9. Reproducibility

No release yet (a citation names the commit); CPython 3.13.17, no other package; seeds 16101 to 16200; `python3 contribute.py --check`, about seven minutes with one process on an x86_64 machine with two processors, checks every value of the report. Nothing depends on the machine's speed, and a reproduction should repeat every value exactly.

## 10. Provenance

The author's own code, built on `cart.task-plan` of Chapter 10 and the benchmark of Chapter 16; no data or assets.

## 11. Status

The certificate (E1): an established certificate, through the safety theorem of Section 9.2, which holds whatever a demand predictor requires, tested in simulation. The performance: an empirical demonstration, in exact simulation of one cart. Not reproduced outside the team.

## 12. How to cite

The book and its code, as `../../../CITATION.cff` gives them, with the release once one exists, and this card.
