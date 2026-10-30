# Chapter 5: control packages for the cart

Two scripts compute every number of Chapter 5. Both need Python 3.8 or later and no other packages.

- `cart_feedback.py` implements the control package of Sections 5.1 and 5.2.
- `cart_methods.py` compares four control packages under matched conditions (Section 5.3). It imports the governor and everything else that is not part of a control method from `cart_feedback.py`, and uses it unchanged; the robust MPC package also supplies the prediction from which the initial set of each plan is chosen.

## `cart_feedback.py`: the package of Sections 5.1 and 5.2

The package controls the cart of Chapter 3 in a world whose motion also carries a disturbance acceleration bounded by 0.05 m/s². The governor, the capability predictors (fixed and forecasting) and the proposal method are those of Chapter 3; only the control method is new:

- **tracking segments** use the feedback u = a − 8 (p − p_ref) − 4 (v − v_ref) around a reference with constant acceleration a.
  - Their certificate is the Lyapunov function V(e) = √(eᵀPe), the square root of a quadratic form, with P = [[6.8, 2], [2, 1]] and decrease rate γ = 1.61 /s.
  - The funnel is the reference surrounded by the tube V ≤ 5/161 m/s.
  - The bounds 0.019 m, 0.049 m/s and 0.125 m/s² hold on the larger set V ≤ 1/32 m/s;
- a **launch segment** from standstill: an open-loop command whose funnel is a box of reachable states;
- a **rest segment** that brakes and holds, with a stoppable set as its funnel;
- a **generator** that blends the proposal with the head of the fallback, continues each blend with braking references down to the handover speed of 0.05 m/s, and ends with the rest segment;
- a **test** that checks, in exact rational arithmetic, the certificate, the input bound, the forward motion, the clearance and every inclusion between funnels. The region of each label holds by construction, because the predictor issues it for the region of its segment;
- a **rest feedback**, the law of the rest segment, which the contract applies only after a label alarm. No alarm occurs in these runs.

The realized motion is integrated in floating point (fourth-order Runge–Kutta, 200 steps per period). Every executed segment is checked against its funnel at every step, in floating point with a tolerance of 1e-9 for rounding. The tolerance matters only where a motion runs along the edge of its funnel, as the launch does under a constant disturbance.

| Run | Content |
|---|---|
| `fixed`, `forecasting` | the governed runs of Section 5.2, with the disturbance w(t) = 0.05 sin(2πt / 2.2 s) m/s² |
| robustness | six further runs, each with the fixed or the forecasting predictor. The disturbance is held at +0.05 or −0.05 m/s², or chosen at every instant to make V grow as fast as possible. The summary reports alarms, steps outside a funnel and the largest V/φ |
| undisturbed | both configurations in a world whose disturbance is zero throughout (Section 5.2.5) |
| Chapter 3 package | the first moving segment of the Chapter 3 control method in this world, which leaves its single-state funnel |

## `cart_methods.py`: four packages under matched conditions (Section 5.3)

| Package | Content |
|---|---|
| `tube` | the package of Sections 5.1 and 5.2 |
| `qp` | the Lyapunov function and rate of `tube`. The feedback is the least input that the hard Lyapunov constraint of a quadratic program allows; its bound, 0.115 m/s², is proved by Sturm's theorem |
| `ccm` | the gains of `tube`, certified by the contraction metric M = [[8, 2], [2, 1]], which contracts at 2 /s |
| `mpc` | the gains, the metric M and the handover speed of `ccm`. Each plan's tube starts from the measured state and grows by s⁺ = ϑ s + (1 − ϑ) φ per period, with ϑ = exp(−0.2) bounded below by a rational; the bounds of each pair are rounded up to millimetres, as in `ccm` |

The world, the disturbance signals, the labels, the predictors, the proposal method, the governor, the generator's structure, the grid of ambitions and the reserve floor are the same in every run. Each package runs eight times: two predictors with the sine signal, plus the six robustness runs.

## Run

```
python3 cart_feedback.py            # all runs; writes results/
python3 cart_feedback.py --check    # also checks every number quoted in Sections 5.1 and 5.2
python3 cart_methods.py --check     # Section 5.3
```

A full run takes about fifteen seconds for `cart_feedback.py` and about seventy seconds for `cart_methods.py`. The comparison with Chapter 3 reads `../ch03/results/summary.txt`, so run the Chapter 3 script first if that file is missing.

## Output (`results/`)

| File | Content |
|---|---|
| `trace-<configuration>.csv` | One row per period k: the realized state at the start of the period, the executed segment and its label, and the proposal. Also the plan committed for period k+1 (ambition, selected index, level, reserve, upper end of its stoppable set), with the largest V/φ and feedback input of the period. |
| `motion-<configuration>.csv` | The realized motion every 0.01 s, with the command and the reference of the executed segment; Figure 5.4 plots the fixed and the forecasting runs against Chapter 3's. |
| `plan-k1.csv`, `plan-k33.csv` | The plans committed in periods 1 and 33 of the forecasting run (Example 5.2 uses period 33). |
| `tube.csv` | The boundaries of the tube and of the set V ≤ 1/32 (Figure 5.1). |
| `errors-forecasting.csv` | The tracking errors of the forecasting run (Figure 5.1). |
| `endphase-*.csv` | The end of the forecasting run in the phase plane (Figure 5.3). |
| `summary.txt` | The numbers quoted in Sections 5.1 and 5.2. |
| `methods-tubes.csv` | Cross-sections of the tubes of the four packages (Figure 5.5). |
| `methods-endphase.csv` | The realized motion of each package from 5 s on, in the forecasting run. |
| `methods-summary.txt` | The numbers quoted in Section 5.3, including the differences and percentages derived from them. |

## Numerical precision

Planned and certified quantities are exact. Quantities of the realized motion are computed in floating point and quoted to two to four significant figures. On another platform, the mathematical library could change the last quoted digit in rare cases; such a change is acceptable numerical variation, not a failed certificate. The certificate checks do not depend on floating point. The plans depend on it only through the measured position at which the proposal method switches to "hold", and, in the `mpc` package, through the measured state from which each tube starts.
