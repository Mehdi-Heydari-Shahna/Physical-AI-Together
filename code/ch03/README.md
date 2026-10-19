# Chapter 3: the smallest complete example

`cart_example.py` implements the example of Chapter 3: a cart that approaches a wall over ground whose braking falls from 2 m/s² to 0.5 m/s² on a downhill ramp, with the five modules of ALLOVI and the viability governor. The ground limits every acceleration towards the start (braking a forward motion, or driving backwards); accelerations towards the wall are realized up to the command bound. Each segment also holds an operation mode ℓ, the reach of the tool beyond the front bumper: 0 with the tool folded, 0.8 m with it unfolded. The declared constraint is p + ℓ ≤ 25 m. Every planned and certified quantity is computed with exact rational arithmetic (`fractions.Fraction`), so each number quoted in the chapter can be checked by hand; floating point is used only for crossing instants, for the proposals executed without the governor, for motion under an invalid label (Section 3.3), and for the checks of physical recovery, including the point of no return.

## Run

```
python cart_example.py            # all runs; writes results/
python cart_example.py --check    # also checks every simulation result quoted in Chapter 3
python cart_example.py --period 32
                                   # prints the candidates of one period of the forecasting run
```

Python 3.8 or later; no other packages. A full run takes about fifteen seconds.

## Configurations

| Name | Capability predictor |
|---|---|
| `fixed` | the global capability, braking of at least 0.5 m/s², everywhere |
| `reactive` | braking measured on positions already traversed; the global capability elsewhere |
| `forecasting` | the minimum of a correct terrain map over the positions a segment covers |
| `persisting` | the braking measured at the current position, applied to every segment (Section 3.3); this run stops at its label alarm |

## Output (`results/`)

| File | Content |
|---|---|
| `trace-<configuration>.csv` | One row per period k: the state at the start of the period, the executed command, its operation mode and its label, the proposal, and the plan committed for period k+1 (ambition, selected index, reserve, level, level from the onset on, requirement, point of rest). |
| `plot-<configuration>.csv` | One row per period: the signals of the plan executed in that period; read by Figures 3.2 and 3.3. |
| `track-persisting.csv` | The persisting run: the realized motion every 0.01 s until standstill, and the motion predicted by the plan executed in the period of the alarm; read by Figure 3.4. |
| `candidates-k<period>.csv` | The candidates of selected periods of the forecasting run (Table 3.3 uses period 32). |
| `summary.txt` | The numbers quoted in Chapter 3. |

Columns use SI units: metres, seconds, metres per second and metres per second squared. The reserve is the smallest clearance of a plan, computed from p + ℓ; its level is the reserve rounded down to whole centimetres. A level of `-inf` marks a plan that the test does not certify at the reserve floor.
