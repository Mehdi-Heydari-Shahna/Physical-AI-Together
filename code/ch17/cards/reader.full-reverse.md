# Method card: `reader.full-reverse`

A filled-in example of `TEMPLATE.md` (Section 17.2). Every result below is in `../results/contribution-summary.txt`, `../results/contribution-diagnosis.txt` or `../results/contribution-leads.csv`, and `contribute.py --check` and `contribute_diagnosis.py --check` verify the first two; the back-away's lengths are those of Section 9.3.2, and the running time is that of `../README.md`.

## 1. Identity

| Field | Entry |
|---|---|
| Name and role | `reader.full-reverse`, role `control` |
| Contract | 1.0; checked with the core of Chapter 10 (`../../ch10/allovi`) of the book's repository, before its first release |
| Version and date | the first version; this card, 25 October 2026, with its dependencies completed on 27 October 2026 |
| Authors and contact | the author of the book, as a worked example; contact through the book's repository once it is published |
| Dependencies beyond the templates | `bench.constant-command` (Chapter 16), which it extends; `cart.control._Labels`, private; the Chapter 3 functions `brake_to_rest` and `backward_step`, its class `Plan` and its constants `EPS_MIN` and `WALL`, through `cart.model.ce`; each is documented, with examples, in `../../ch10/templates/HELPERS.md` |
| Code | `../contrib/full_reverse.py`, in the book's repository before its first release; a citation names the commit |
| Licence | not yet stated; it will be given with the first release (`../../RELEASES.md`) |

## 2. What it does

The control package `bench.constant-command` of Chapter 16 with one more candidate in every period in which a requirement above the floor is pending: the full reverse of Section 9.3.2, which brakes a forward motion to rest, drives towards the start at the strongest command that the labels certify until the front is at or before the requirement's place, and then brakes to rest, offered with ambition 0. The test, the rest segment, the rest feedback and every other candidate are those of `bench.constant-command`.

## 3. Semantics

As `bench.constant-command` (Sections 3.1 and 10.2): the state (p, v, ℓ) in metres, metres per second and metres; the command an acceleration in m/s², positive towards the wall, for one period of 0.1 s; labels a lower bound on braking in m/s²; the requirement a reserve in metres from an onset pair on, whose place is `25 m − level`. The full reverse's segments hold one command each for one period. Nothing is rounded: every quantity is an exact rational number.

## 4. Obligations and how each is established

- Sound test: inherited unchanged from `cart.constant-command`, whose test is exact (Chapter 3); the package adds candidates, not verdicts.
- A certified rest segment, its rest feedback and the reach bound: inherited unchanged.
- Candidates that start at the initial set and chain their segments: by construction, and checked by the check suite (13 passed).
- Candidates for a pending requirement (Assumption 4.13 and the generator's row of Table 4.1): the full reverse is one more such candidate. The guarantees of Chapter 9 hold for any candidates that the test certifies, so the package claims no new guarantee.

## 5. Assumptions and scope

The cart of Chapter 3 and of the anticipation benchmark, on the ramp, with the labels of the survey or of the forecasting predictor. The full reverse assumes nothing beyond what the test checks. It has been run with no other robot.

## 6. A runnable example

`python3 contribute.py` runs it in the world of Chapter 3 with the operation announced 21, 20, 19 and 18 periods before its onset, and in the family 16101 to 16200; it prints, among the rest:

```
full-reverse: lead times in periods at which the operation begins on time; not = 21, 20, 19; 18
```

## 7. Evidence

- **Checks.** The check suite of Section 10.3, with the package in its role in the configuration of the forecasting run of Chapter 3: 13 passed, 0 failed, 0 not applicable. A passed check discharges no obligation. `contribute.py` runs the suite; `../../ch10/cart_checks.py --module` cannot take this file, which imports the benchmark's package `bench`, and `../../ch10/templates/HELPERS.md` gives the call of the suite that runs it alone.
- **Hypotheses.** F1 to F4 of `../HYPOTHESES.md`, written on 25 October 2026 before the first run of `contribute.py`; every decision rule supports a hypothesis only if it holds without exception in its subset.
- **Comparison.** C6 of Chapter 16 (the survey's labels, the task plan's forecast, the naive proposal method, every candidate), with the control package replaced and nothing else; seeds 16101 to 16200, by `../../ch16/bench/worlds.py`; every fresh candidate in every period.
- **Results.** Over the 95 valid worlds: the operation on time in 69 (62.5 to 81.3 %), as C6; late in 26, by a median 2.1 s; median arrival 8.9 s and time in the work zone 13.0 s (12.5 to 13.2 s), as C6. Over all 100: 5 runs with an invalid executed label, 5 alarms, 2 restarts, no violation, as C6. F1 supported (95 cases); F2 supported by its rule but, in substance, untested: its one case is the world of Chapter 3, where Section 9.3.2 found the result, and the family has no valid world without a slip and with a lead time of 19 or 20 periods; F3 supported (95 cases); F4 not supported (58 cases, 58 exceptions).
- **Failed runs.** `../results/contribution-runs.csv` lists every run. The five with an invalid executed label lie in the five worlds that are not valid; each raised an alarm, two restarted, and in the other three (worlds 16116, 16120 and 16144) no restart could be certified, the true clearance fell to 0.48 or 0.49 m, below the floor, without a violation, and the operation never began, exactly as with C6.

## 8. Measured limitations

- The shorter lead time helped in one world of 95: world 16179, with a lead time of 20 periods and an onset 6 periods earlier than forecast, where the full reverse began the operation at the forecast onset and C6 was postponed by 21 periods.
- It did not delay the preparation (F4, found after the run, `contribute_diagnosis.py`): the candidates that follow the proposal end in the generator's own back-away, which needs 21 periods, so the governor stops following the proposal in the same period as with C6. From then on it commits the full reverse at ties, but the full reverse begins with the same command as the back-away, and the executed commands are those of C6 until the operation begins. Delaying the preparation would need the shorter back-away as the continuation of every candidate, a different package and a new hypothesis.
- It rests 0.215 m farther from the wall than the back-away of Chapter 3 when it is used to the end (Section 9.3.2).

## 9. Reproducibility

No release yet (a citation names the commit); CPython 3.13.17, no other package; seeds 16101 to 16200; `python3 contribute.py --check`, about seven minutes with one process on an x86_64 machine with two processors, checks every value of the report. Nothing depends on the machine's speed, and a reproduction should repeat every value exactly.

## 10. Provenance

The author's own code, built on the cart's modules of Chapters 3 and 10 and the benchmark of Chapter 16; no data or assets.

## 11. Status

The certificate (F1): an established certificate, through the safety theorem of Section 9.2, tested in simulation. The lead time of 19 periods: an established result of Section 9.3.2 and an empirical demonstration in the world of Chapter 3. The performance in the family: an empirical demonstration, in exact simulation of one cart. Not reproduced outside the team.

## 12. How to cite

The book and its code, as `../../../CITATION.cff` gives them, with the release once one exists, and this card.
