# The hypotheses of the two worked contributions (Section 17.1)

Written on 25 October 2026, before the first run of `contribute.py`, and kept in the script, which applies each decision rule as written here. A change made after a run is listed under *Deviations* at the end, with its reason.

Both contributions are compared with ALLOVI as Chapter 16 configured it (C6 of `../ch16/HYPOTHESES.md`: the survey's labels with the calibrated offset, the task plan's forecast, the naive proposal method, every candidate), in a family of 100 worlds that Chapter 16 did not use: seeds 16101 to 16200, drawn by the rules of `../ch16/bench/worlds.py`. Each contribution replaces one module of C6 and nothing else. A world is valid as in Chapter 16. Lead time: periods from the announcement to the forecast onset.

## A control package: `reader.full-reverse` (`contrib/full_reverse.py`)

The package of Chapter 16 with one more candidate while a requirement above the floor is pending: the full reverse of Section 9.3.2, which drives towards the start at the strongest certified command until the front is at or before the requirement's place, and then brakes to rest. Section 9.3.2 shows that, in the world of Chapter 3, a generator with this candidate meets the operation with 19 periods of lead time instead of 21.

- **F1 (the certificate).** In every run whose executed labels are all valid, as the world's oracle assesses them, no alarm is raised and the true clearance never falls below the floor. Decision: supported if there is no exception.
- **F2 (the shorter lead time).** In the world of Chapter 3, the operation begins on time when it is announced 19 periods before its onset, and not when it is announced 18 periods before; in every valid world of the family with no slip and a lead time of 19 or 20 periods, the full reverse begins the operation on time, and C6 does not. Decision: supported if every statement holds in every world of its subset.
- **F3 (no loss).** In every valid world, the full reverse begins the operation no later than C6, and arrives in the work zone in the same period. Decision: supported if there is no exception.
- **F4 (later preparation).** In every valid world with no slip and a lead time of 21 periods or more, both begin the operation on time, and the full reverse spends one or two periods more in the work zone than C6. Decision: supported if it holds in every world of the subset.

## A demand predictor: `reader.early-window` (`contrib/early_window.py`)

The task plan's demand predictor with the requirement's onset moved 10 periods earlier than the forecast onset, the earliest that the task plan states the operation may come; once that onset has passed, the requirement applies from the first pair on, as it does for the task plan's forecast. Its predictions come from Corollary 9.13 and, for E4, from the mechanism that Section 16.2.2 found in world 16070: a requirement that applies from the first pair on cannot be raised within a period, so the first key of the selection ties and the proposal decides.

- **E1 (the certificate).** As F1. Decision: supported if there is no exception.
- **E2 (an earlier onset).** In every valid world whose onset comes 1 to 10 periods earlier than forecast, with a lead time of 31 periods or more, the early window begins the operation on time, and C6 does not. Decision: supported if it holds in every world of the subset.
- **E3 (the price).** In every valid world with no slip and a lead time of 31 periods or more, both begin the operation on time, and the early window spends exactly 10 periods less in the work zone than C6. Decision: supported if it holds in every world of the subset.
- **E4 (a window that cannot be met).** In every valid world with no slip and a lead time from 21 to 30 periods, the early window begins the operation later than C6, which begins it on time. Decision: supported if it holds in every world of the subset.
- **E5 (the approach unchanged).** In every world, the early window arrives in the work zone in the same period as C6. Decision: supported if there is no exception.

## Deviations

No hypothesis, decision rule, configuration, module or world was changed after a run.

- 25 October 2026, after the first run of `contribute.py`: F1, F2, F3, E1, E2, E3 and E5 were supported; F4 and E4 were not. F2's subset in the family proved empty, so F2 rests on the world of Chapter 3 alone; its rule, as written, makes it supported. Nothing in `contribute.py` was changed but its `EXPECTED`, filled from the first run's report; the second run, with `--check`, repeated every value.
- The same day, `contribute_diagnosis.py` was written to examine F2's subset, the exceptions of F4 and E4, the early window's runs that began later than C6, which no hypothesis covered, and the five worlds that are not valid. It tests no hypothesis; Section 17.1 reports what it found as diagnoses.
