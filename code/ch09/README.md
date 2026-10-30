# Chapter 9: the lead time of the cart's operation

`cart_lead_time.py` computes the numbers of Section 9.3 that are not quoted from earlier chapters, and the numbers that Chapters 3, 7 and 8 quote from it; with `--check`, it compares each with the value quoted in the text, which the script records. It needs Python 3.8 or later and no other packages. It imports the cart, its control method (segments, generator and test), the capability predictors, the proposal method, the operation and the governor from `../ch03/cart_example.py` and uses them unchanged. Sections 9.1 and 9.2 compute no numbers of their own: they quote those of Chapters 3, 5, 7 and 8.

## What it computes

The cart holds at rest at 24.5 m, with the level 0.5 m, when the operation is announced; the operation needs 1.3 m from its onset, period 200, on. All planned and certified quantities are exact rational numbers.

- **Sufficiency (Theorem 9.12(b)).** The back-away that the generator of Chapter 3 offers from rest at 24.5 m (16 periods at -0.5 m/s^2, then 4 at 2 m/s^2, rest at 23.7 m), its profile and its restoration time, 20 pairs, hence a lead time of 21 periods; the same back-away with the labels of the fixed predictor; and the rate form of the theorem, which gives no positive rate on the grid of levels and 321 periods with exact reserves.
- **Necessity (Theorem 9.12(c)).** The bound 0.5 + 0.0025 j^2 m on the reserve of every plan in force in period k0 + 1 + j, where k0 is the period of the announcement, that is certified on a set containing the state, from Lemma 9.11 and the largest acceleration towards the start on the ramp, 0.5 m/s^2; hence at least 19 periods.
- **The full reverse.** A continuation that drives towards the start at the largest certified acceleration until the front is at or before 23.7 m and then brakes to rest; it attains the bound up to its pair 18 and holds 1.31 m from pair 18 on, hence a lead time of 19 periods.
- **Runs of the governor.** The forecasting run of Chapter 3 with the operation announced 1 to 24 periods before its onset, with the generator of Chapter 3 and with a generator that also offers the full reverse, with ambition 0, whenever a demand is pending. The script checks that the cart holds at 24.5 m at every announcement, that the level in force at the onset is the level of the continuation committed at the announcement from its pair N_a - 1 on, or 1.3 m once the demand is met, and that the smallest lead times that meet the demand are 21 and 19 periods.
- **A requirement missing in one period (Section 9.3.3).** The forecasting run of Chapter 3 (announcement in period 140, onset in period 200), with the requirement replaced by its default, the floor from pair 0 on, in one period from 140 to 199, for generation and selection alike.
- **Retention (Theorem 9.12(a)).** For the nine runs of Chapter 8 whose decision files carry a requirement, every pair of consecutive periods with the same requirement for the same onset period, with the requirement from the first pair on in both, or with an announcement, is checked: the onset level of the committed plan, capped at the requirement, is at least that of the period before. This reads `../ch08/results/decisions-*.csv`; run `../ch08/cart_governor.py` first if they are missing.
- **Proposition 9.16 and Example 9.17.** The arithmetic of the proof that the persisting predictor drives the cart across the wall, and of the plan whose labels outlive their declared horizon.
- **Exercise 9.1.** The reference answer: the cart starts at rest at 0.6 m (and at 0.4 m) while its position sensor reads that much short. The script checks that every command of the fixed run of Chapter 3 is realized exactly from the shifted true state, so that the governor receives the measurements of that run and raises no alarm, and computes the true positions that the answer quotes: rest at 25.095 m, 0.095 m beyond the wall, the tool unfolding with the front at 24.295 m and its tip beyond the wall; with 0.4 m, rest at 24.895 m, inside the constraint but 0.395 m past the floor.

## Run

```
python3 cart_lead_time.py           # writes results/
python3 cart_lead_time.py --check   # also checks every number quoted in the text
```

A full run takes about three minutes.

## Output (`results/`)

| File | Content |
|---|---|
| `summary.txt` | The numbers quoted in the text. |
| `lead-time.csv` | For each lead time from 1 to 24 periods: the level in force at the onset in the runs with each generator, whether the demand is met, and the bound of Theorem 9.12(c) (Figure 9.4). |
| `dropped-requirement.csv` | For each period from 140 to 199: whether the demand is met when the requirement is missing in that period, and the level in force at the onset (Figure 9.5). |
| `retention.csv` | Every pair of consecutive periods checked for retention, with the run, the period, the kind of comparison, the requirement and the two levels compared. |
