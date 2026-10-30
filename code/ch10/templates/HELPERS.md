# Helpers of the cart's code and of the core

The templates in this folder give the interface of each role. A module written for the cart of Chapter 3 also uses what the templates do not show: the cart's units, constants, state and proposals, its task, its segments and plans, the functions of its generator and its control package; and running or checking it uses the core's configurations, runs, records and check suite. This file documents each of these that the two worked contributions of Chapter 17 (`../../ch17/contrib`) and the reference answer of Exercise 10.1 use: where it lives, what it takes and returns, its units, the invariants that a module must respect, and an example. The docstrings in the files say the same. Every example after a `>>>` prompt runs as a doctest (section 13).

None of these helpers is part of the contract. The registry does not check them, the version of the contract (`CONTRACT`, 1.1, in `../allovi/contract.py`) does not cover them, and a later release may change them; a name with a leading underscore, such as `cart.control._Labels`, is private to its package. A module that uses one names it in the field "Dependencies beyond the templates" of its method card (`../../ch17/cards/TEMPLATE.md`), so that a reader knows what to supply and a change can be traced (`../../RELEASES.md`).

| Helper | Where it lives | Used by |
|---|---|---|
| the units and the state `(p, v, l)` (section 2) | `../cart/model.py` | every module of the cart |
| `cart.model.OPERATION_LEVEL`, `cart.T`, `cart.model.LEVELS`, `cart.fmt` (sections 2 and 3) | `../cart/model.py` | the early window; running and checking a module |
| `cart.model.ce` and its parameters `ce.EPS_MIN`, `ce.WALL`, `ce.RAMP` and the others (section 3) | `../../ch03/cart_example.py` | both contributions |
| the proposals `"accelerate"` and `"hold"`, `ce.proposed_command` (section 4) | `../../ch03/cart_example.py` | Exercise 10.1 |
| the task `cart.operation`, `cart.modules.Operation` (section 5) | `../cart/modules.py` | the early window |
| `ce.Segment` and `ce.Plan` (section 6) | `../../ch03/cart_example.py` | the full reverse |
| `ce.braking_step`, `ce.backward_step`, `ce.brake_to_rest`; `ce.recovery` and `ce.generate`, which the inherited generator calls (section 7) | `../../ch03/cart_example.py` | the full reverse |
| `cart.control._Labels` (section 8) | `../cart/control.py` | the full reverse |
| `cart.constant-command` and `bench.constant-command`, extended by a subclass; what `allovi.Step` and `allovi.Candidate` hold for the cart (section 9) | `../cart/control.py`, `../../ch16/bench/modules.py`, `../allovi/contract.py` | the full reverse |
| `allovi.build`, `allovi.run`, `allovi.RunRecord` and `cart.metrics` (section 10) | `../allovi/registry.py`, `../allovi/loop.py`, `../cart/modules.py` | running any module; Exercise 10.1 |
| `allovi.checks.check`, and what `cart_checks.py --module` cannot take (section 11) | `../allovi/checks.py`, `../cart_checks.py` | checking any module |
| the comparison of `contribute.py`, and a configuration of the benchmark, one world at a time (section 12) | `../../ch17/contribute.py`, `../../ch16/cart_benchmark.py`, `../../ch16/bench/worlds.py` | steps 6 to 9 of Figure S17.1; Exercise 17.1 |

## 1. Importing the cart's code

The core and the cart are the packages `allovi` and `cart` in `code/ch10`. Importing `cart` registers the cart's modules in the default registry of `allovi`, and puts `code/ch03`, whose `cart_example.py` it reads, and `code/ch10` on `sys.path`. The chapters' scripts put the folders they need on the path themselves (`cart_checks.py` puts `code/ch10`; `contribute.py` puts `code/ch16` and `code/ch17` as well), and a script of one's own does the same. From the folder that holds `code/`:

```pycon
>>> import sys
>>> sys.path.insert(0, "code/ch10")
>>> from fractions import Fraction as Fr
>>> import allovi
>>> import cart                              # registers the cart's modules
>>> from cart.model import ce, OPERATION_LEVEL
>>> allovi.REGISTRY.names("control"), allovi.REGISTRY.names("demand")
(['cart.constant-command'], ['cart.task-plan'])

```

Importing a file registers every module that it decorates with `@register`, and those of every package that it imports: `bench.modules` of Chapter 16 registers three (section 9). A name is registered once in its role; registering another class under it raises `allovi.RegistrationError`.

## 2. Units, numbers and the cart's state

Every quantity of the cart is an exact `fractions.Fraction`: positions and the reach in metres, velocities in m/s, commands and labels in m/s², levels and clearances in metres, times in seconds. Periods are whole numbers k, with t_k = k T and T = 1/10 s. Floating point enters only the motion under an invalid label (Section 10.1); a module should compute exactly, since `0.1` is not `Fr(1, 10)`.

The state, and its measurement `observation.measured`, is `(p, v, l)`: the position of the front bumper from the start, positive towards the wall; its velocity; and the reach of the tool beyond the bumper, 0 folded or 4/5 m unfolded. The constraint is p + l <= 25 m, and the reserve of a state is its clearance, 25 m - (p + l). `STATE` and `ACTUATION` in `../cart/model.py` declare them (Section S10.9.2); the measurement is exact (`cart.robot` has the `measurement_bound` 0). The initial set that the cart's control package passes to its generator is a single state of the same form (section 9). `cart.fmt` prints a number as the records do:

```pycon
>>> cart.fmt(Fr(2409, 100)), cart.fmt(Fr(1, 3)), cart.fmt(Fr(25))
('24.09', '0.333333', '25')

```

## 3. Constants

`cart.model` exposes the example of Chapter 3 as the module `ce`, whose parameters (Table S3.1) are its attributes:

| Name | Value | Meaning |
|---|---|---|
| `ce.T`, `cart.T` | 1/10 s | the period of the governor |
| `ce.U_MAX` | 2 m/s² | the command bound |
| `ce.WALL` | 25 m | the wall; the constraint is p + l <= 25 m |
| `ce.RAMP` | 20 m | where the ramp starts in the world of Chapter 3 (see below) |
| `ce.B_FIRM`, `ce.B_RAMP` | 2 and 1/2 m/s² | the braking of firm ground and of the ramp |
| `ce.B_GLOBAL` | 1/2 m/s² | the global capability: braking of at least 0.5 m/s² everywhere |
| `ce.EPS_MIN` | 1/2 m | the reserve floor, the level required when no demand is pending |
| `ce.GRID`, `ce.EPS_TOP` | 1/100 m, 25 m | the grid of levels, from the floor to 25 m |
| `ce.ELL_UNFOLDED` | 4/5 m | the reach of the unfolded tool |
| `ce.WORK_ZONE` | 24 m | the task: reach p >= 24 m and hold there |
| `ce.ALPHAS` | 1, 3/4, 1/2, 1/4, 0 | the ambitions of the generator's blends |
| `ce.HORIZON` | 400 periods | the forecast horizon of the cart's capability predictors |
| `ce.DEMAND_LEVEL`, `cart.model.OPERATION_LEVEL` | 13/10 m | the reserve that the tool operation needs |
| `ce.NEG_INF` | `None` | the level of a plan that the test of Chapter 3 does not certify; the core's `allovi.NEG_INF` is `float('-inf')` |
| `cart.model.LEVELS` | `allovi.ScalarLevels(1/2, 1/100, 25)` | the level space of the cart's test |

```pycon
>>> cart.T, OPERATION_LEVEL, ce.EPS_MIN, ce.WALL - OPERATION_LEVEL
(Fraction(1, 10), Fraction(13, 10), Fraction(1, 2), Fraction(237, 10))
>>> cart.model.LEVELS.contains(Fr(129, 100)), cart.model.LEVELS.contains(Fr(1295, 1000))
(True, False)

```

The functions of `ce` read these parameters when they are called. The scripts of Chapters 7, 16 and 17 set `ce.RAMP` to the ramp of each of their worlds while they run it, and set it back afterwards (section 12); Exercise 3.1 sets `ce.B_GLOBAL`. A module reads `ce.RAMP` when it needs it, never copies it at import, and does not change any of these parameters.

## 4. What the cart's proposals mean

A proposal method of the cart returns `"accelerate"`, `"hold"` or `None`, and the generator of `cart.constant-command` gives a proposal its meaning at the initial set (Section S10.9.4). `ce.proposed_command(q, xhat)`, with xhat = (p, v), returns +2 m/s² for "accelerate"; for "hold", the command that stops the cart within one period if 2 m/s² allows it, at most 2 m/s² in magnitude, and 0 at rest. Every other proposal is read as "hold". `None` means that no proposal is ready, and the generator offers the recovery continuation alone. `bench.constant-command` of Chapter 16 adds "retreat", a command of -1/2 m/s² (section 9).

```pycon
>>> ce.proposed_command("accelerate", (Fr(10), Fr(1)))
Fraction(2, 1)
>>> ce.proposed_command("hold", (Fr(24), Fr(1))), ce.proposed_command("hold", (Fr(24), Fr(0)))
(Fraction(-2, 1), Fraction(0, 1))
>>> ce.proposed_command("wait", (Fr(24), Fr(1)))          # read as "hold"
Fraction(-2, 1)

```

The reference answer of Exercise 10.1 uses nothing else: its method reads `observation.measured[0]`, the position p, and returns "accelerate" or "hold"; section 10 runs it.

## 5. The cart's task, `cart.operation`

`cart.modules.Operation`, registered as the task `cart.operation`, is the tool operation of Chapter 3: due in period `onset`, it needs 1.3 m of reserve, and it begins if the plan in force during that period holds it (Assumption 4.13). Its constructor takes, in periods of 0.1 s: `announce` (140), the period from which the task plan announces it, or `None` if it is never announced; `onset` (200); `predicted`, the onset that the task plan forecasts (by default `onset`); `always` (`False`), which asks a demand predictor to require the reserve in every period; and `postpone` (`None`). With `postpone=None` an overload drops the operation; otherwise the operation is postponed, to the forecast onset if that is still ahead, else by `postpone` periods, announced at once (Section 8.2.3). A run sets `begun`, the period in which the operation began, or `None`; `dropped`; `overloads`, the periods in which it was due and the plan in force did not hold 1.3 m; and `reach`, which `operation(k)` returns: 0 until the operation begins, 4/5 m from then on. `pending()` is true until it has begun or been dropped.

A demand predictor reads these attributes; the early window reads `pending()`, `always`, `announce` and `predicted`. A configuration creates one task and passes the same instance to its demand predictor and to its evaluator, as the argument `task` (section 10); a predictor given another instance would read a schedule that the run never updates.

```pycon
>>> task = cart.modules.Operation()          # announced in period 140, due in 200
>>> task.pending(), task.announce, task.onset, task.predicted, task.always, task.postpone
(True, 140, 200, 200, False, None)
>>> task.begin_period(200, Fr(13, 10))       # the plan in force holds 1.3 m
>>> task.begun, task.pending(), task.operation(200)
(200, False, Fraction(4, 5))
>>> late = cart.modules.Operation(postpone=21)
>>> late.begin_period(200, Fr(1, 2))         # it holds only the floor: an overload
>>> late.overloads, late.onset, late.predicted, late.announce
([200], 221, 221, 200)
>>> demand = allovi.REGISTRY.create("demand", "cart.task-plan", task=late)
>>> o = allovi.Observation(k=205, t=Fr(41, 2), measured=(Fr(49, 2), Fr(0), Fr(0)),
...                        level=Fr(1, 2), ambition=Fr(0), alarm=False)
>>> demand.requirement(o)                    # 1.3 m from pair 221 - 205 - 1 on
Requirement(level=Fraction(13, 10), onset=15)

```

## 6. Segments and plans

`ce.Segment(p0, v0, u)` is one period of the constant command u [m/s²] from (p0, v0). Its attributes are `u`, `p0` and `v0`; `p1` and `v1`, the state at the end of the period if the command is realized; and `lo` and `hi`, the least and the largest position within the period, the region of the segment to which a label refers (the survey predictor of Section 10.2 and `../examples/survey_margin.py` read `segment.hi`). With exact arguments every attribute is exact. `certified(beta)` is condition (i) of Chapter 3 under the label beta, braking of at least beta m/s² on the region: a command of at most 2 m/s² in magnitude, an acceleration towards the start of at most beta, and no reversal of the motion within the period.

`ce.Plan(j0, pairs, p_rest, ell=0)` is a labelled plan: `pairs`, a list of (Segment, label), executed in periods j0, j0 + 1, ..., then the rest segment, which holds the cart at `p_rest`; `ell` is the reach of the tool throughout. `reserve()` is its least clearance if every pair is certified, else `None`; `head_command()` is the command of pair 0 (0 without pairs); `shift()` is the plan without its head, the fallback of the next period; and `profile()` lists the levels certified from pair n on, for n = 0 to N, with `None` where nothing is certified.

The test relies on two conditions that it does not check, and that whoever builds a plan must ensure: each segment starts where the one before it ends, and the last pair brings the cart to rest at `p_rest` (a plan without pairs starts at rest there). The plans of `ce.brake_to_rest`, `ce.recovery` and `ce.generate` meet both; a plan built otherwise, as the full reverse builds one, must meet them too. The package `cart.constant-command` also refuses a plan whose head does not start at the initial set, condition (P1), and turns `None` into the core's `NEG_INF`.

```pycon
>>> seg = ce.Segment(Fr(24), Fr(1), Fr(-2))  # braking at 2 m/s^2 from 1 m/s
>>> seg.p1, seg.v1, seg.lo, seg.hi
(Fraction(2409, 100), Fraction(4, 5), Fraction(24, 1), Fraction(2409, 100))
>>> seg.certified(Fr(2)), seg.certified(Fr(1, 2))
(True, False)
>>> class Firm:                              # every label: firm ground, 2 m/s^2
...     def label(self, k, j, segment):
...         return Fr(2)
>>> pairs, p_rest = ce.brake_to_rest(Firm(), 0, 1, Fr(24), Fr(1))
>>> plan = ce.Plan(1, pairs, p_rest)
>>> len(plan.pairs), p_rest, plan.head_command(), plan.reserve()
(5, Fraction(97, 4), Fraction(-2, 1), Fraction(3, 4))
>>> plan.profile()[0], plan.shift().j0, len(plan.shift().pairs)
(Fraction(3, 4), 2, 4)
>>> wrong = ce.Plan(1, [(seg, Fr(1, 2))] + pairs[1:], p_rest)  # pair 0 brakes beyond its label
>>> wrong.profile()[:2], wrong.reserve()
([None, Fraction(3, 4)], None)
>>> package = allovi.REGISTRY.create("control", "cart.constant-command")
>>> package.profile(plan, (Fr(24), Fr(1), Fr(0))).level
Fraction(3, 4)
>>> package.profile(wrong, (Fr(24), Fr(1), Fr(0))).level, package.profile(plan, (Fr(23), Fr(1), Fr(0))).level
(-inf, -inf)

```

## 7. Building candidates: the functions of the cart's generator

The functions of `ce` that build segments take a `pred`: any object with a method `label(k, j, segment)` that returns the label, in m/s², issued during period k for the segment in period j. During a period of the core, it is `cart.control._Labels(step.labelling)` (section 8). Each function returns exact segments that start at the state it is given.

- `ce.braking_step(pred, k, j, p, v)` returns (segment, label). From (p, v) at the start of period j with v >= 0, it brakes at the strongest of 2 m/s², 0.5 m/s² and the global capability that the label of its own segment certifies, capped at v/T so that the cart stops within the period; with v < 0, it commands min(2, -v/T) m/s² towards the wall, which every world realizes.
- `ce.backward_step(pred, k, j, p, v)` returns (segment, label): the strongest command towards the start, -2 or -0.5 m/s², that the label of its own segment certifies, from rest or during a backward drive (v <= 0).
- `ce.brake_to_rest(pred, k, j, p, v)` returns (pairs, p_rest): `braking_step` repeated until the cart is at rest, and where it is then; no pairs if v = 0.
- `ce.recovery(pred, k, j, p, v, target)` returns (pairs, p_rest): the recovery continuation, braking to rest and, if `target` [m] is given and the cart rests beyond it, the shortest back-away that comes to rest at or before it.
- `ce.generate(pred, k, xhat, u_q, fallback, target, ell)` returns the candidates of period k as a list of (Plan, ambition): the blends of the proposed command u_q with the head of `fallback` for the ambitions of `ce.ALPHAS`, each followed by the recovery continuation, and then the continuation alone with ambition 0, each starting at xhat = (p, v) in period k + 1.

Each raises `AssertionError` if not even the global capability is certified, which labels at least as informative as the global capability (Assumption 4.3) exclude.

```pycon
>>> class Ramp:                              # every label: the ramp, 0.5 m/s^2
...     def label(self, k, j, segment):
...         return Fr(1, 2)
>>> seg, label = ce.braking_step(Ramp(), 0, 1, Fr(24), Fr(1))
>>> seg.u, label
(Fraction(-1, 2), Fraction(1, 2))
>>> seg, label = ce.backward_step(Ramp(), 0, 1, Fr(49, 2), Fr(0))
>>> seg.u, seg.p1, seg.v1
(Fraction(-1, 2), Fraction(9799, 400), Fraction(-1, 20))
>>> pairs, p_rest = ce.brake_to_rest(Ramp(), 0, 1, Fr(24), Fr(1))
>>> len(pairs), p_rest                       # at rest at the wall: not certified
(20, Fraction(25, 1))
>>> pairs, p_rest = ce.recovery(Ramp(), 0, 1, Fr(49, 2), Fr(0), ce.WALL - OPERATION_LEVEL)
>>> [cart.fmt(s.u) for s, _ in pairs].count("-0.5"), len(pairs), p_rest
(16, 20, Fraction(237, 10))

```

The last is the back-away of Section 9.3.2: 16 periods at -0.5 m/s², then 4 at 2 m/s², at rest at 23.7 m.

## 8. Labels in the generator: `cart.control._Labels`

`cart.control._Labels(labelling)` presents the labelling map of the period, `step.labelling`, a function of a period j and a segment, as the `pred` that the functions of section 7 ask: `label(k, j, segment)` returns `labelling(j, segment)` and ignores k. The map is valid only during the period in which the core issued it, so an instance must not be kept for a later period. The class is private to the cart's package; a module that should not depend on it can define the same adapter in a few lines:

```pycon
>>> from cart.control import _Labels
>>> _Labels(lambda j, segment: Fr(1, 2)).label(7, 8, None)
Fraction(1, 2)
>>> class Labels:                            # the same, without the private name
...     def __init__(self, labelling):
...         self.labelling = labelling
...     def label(self, k, j, segment):
...         return self.labelling(j, segment)
>>> Labels(lambda j, segment: Fr(1, 2)).label(7, 8, None)
Fraction(1, 2)

```

## 9. Extending the cart's control package

`cart.control.ConstantCommand`, registered as `cart.constant-command`, is the cart's control package. `bench.modules.Retreating`, registered as `bench.constant-command` in `../../ch16/bench/modules.py`, is a subclass that adds the proposal "retreat", -1/2 m/s²; with the proposals of Chapter 3 it decides as `cart.constant-command` does (`../../ch16/cart_benchmark.py` checks that it repeats, byte for byte, the decision file of the forecast-informed run of Section 8.2). A package that adds candidates, as the full reverse does, subclasses one of them, sets a `NAME` of its own (`ROLE` and `CONTRACT` are inherited), registers the subclass and overrides `generate(step, plan, initial)`:

- `step` is the `allovi.Step` of the period: `step.k`; `step.proposal`; `step.labelling`, the labelling map of the period; `step.requirement`, an `allovi.Requirement` whose `level` is a reserve in metres, the floor `ce.EPS_MIN` when no demand is pending, and whose `onset` is the pair from which the plan should hold it; `step.operation`, the reach that the task asks for, 0 or 4/5 m (`None` without a task); and `step.measured`, the measured state.
- `plan` is the plan in force, a `ce.Plan`; `plan.shift()` is the fallback.
- `initial` is the initial set, a single state (p, v, l), the end of the head of the plan in force, from which every candidate must start in period `step.k + 1`.
- `super().generate(step, plan, initial)` returns the package's candidates, a list of `allovi.Candidate(plan, ambition)` in the order of evaluation: the blends for the ambitions 1, 3/4, 1/2, 1/4 and 0, then the recovery continuation alone with ambition 0; only the continuation if `step.proposal` is `None`. The core evaluates the candidates in this order until the budget of the period is spent (Section 8.3.3), and among candidates that tie on both keys of the selection rule it commits the one listed first (rule (4.2)), so a candidate appended with ambition 0 is committed only when it holds more of the requirement than every candidate before it.

The test, the rest segment, its feedback and `restart` are inherited. The added plans must meet the conditions of section 6, with the labels of `step.labelling` for their own segments; the package's obligations (Table 4.1) remain the contributor's to establish, as the full reverse's docstring does. Importing `bench.modules` needs `code/ch16` on the path, and registers its three modules: `bench.constant-command`, the proposal method `bench.preview` and the evaluator `bench.metrics`.

```pycon
>>> step = allovi.Step(k=180, proposal="hold", labelling=lambda j, segment: Fr(1, 2),
...                    requirement=allovi.Requirement(OPERATION_LEVEL, 19),
...                    operation=Fr(0), measured=(Fr(49, 2), Fr(0), Fr(0)))
>>> at_rest = ce.Plan(180, [], Fr(49, 2))    # the plan in force: at rest at 24.5 m
>>> initial = package.initial_set(at_rest)
>>> initial
(Fraction(49, 2), Fraction(0, 1), Fraction(0, 1))
>>> cands = package.generate(step, at_rest, initial)
>>> [c.ambition for c in cands]
[Fraction(1, 1), Fraction(3, 4), Fraction(1, 2), Fraction(1, 4), Fraction(0, 1), Fraction(0, 1)]
>>> [len(c.plan.pairs) for c in cands], cands[-1].plan.j0, cands[-1].plan.p_rest
([21, 21, 21, 21, 21, 20], 181, Fraction(237, 10))
>>> sys.path.insert(0, "code/ch16")
>>> from bench.modules import Retreating, command_of
>>> Retreating.NAME, Retreating.ROLE, issubclass(Retreating, cart.control.ConstantCommand)
('bench.constant-command', 'control', True)
>>> command_of("retreat", (Fr(24), Fr(0)))
Fraction(-1, 2)

```

The full reverse, `../../ch17/contrib/full_reverse.py`, overrides `generate` of `bench.constant-command` in this way, where `full_reverse` is its own function, which chains `ce.brake_to_rest` and `ce.backward_step`:

```python
    def generate(self, step, plan, initial):
        cands = super().generate(step, plan, initial)
        req = step.requirement.level
        if req > ce.EPS_MIN:
            p, v, _ = initial
            ell = Fr(0) if step.operation is None else step.operation
            labels = _Labels(step.labelling)
            pairs, p_rest = full_reverse(labels, step.k, step.k + 1, p, v, ce.WALL - req)
            if pairs:
                cands.append(Candidate(ce.Plan(step.k + 1, pairs, p_rest, ell), Fr(0)))
        return cands
```

## 10. Running a module in the reference configuration

The reference configuration of the check suite is the forecasting run of Chapter 3: `cart.constant-command`, `cart.naive`, `cart.forecasting`, the demand predictor `cart.task-plan`, the task `cart.operation` announced in period 140 for period 200, `cart.robot`, `cart.track` and `cart.metrics`, for 220 periods. `cart.configuration(name)` builds it with the cart's own modules only. To run another module in it, build the dictionary with the module's name in its role: `allovi.build(name, period, modules)` takes, for each role, a registered name with its arguments, or an instance, as for the task. `allovi.run(config, periods)` runs it with strict checking, which stops at the first failure of a module with an `allovi.ModuleError` that names the period and the call (Section S10.9.4), and returns an `allovi.RunRecord`: `periods`, one `PeriodRecord` per period, whose `state` is the state at t_k; `final_state`; and `alarms` and `restarts`, the instants of each as the index k of t_k. The evaluator `cart.metrics` turns a record into the numbers of the cart's runs: `arrival_in_work_zone` [s], `time_in_work_zone` [s], `tool_unfolds` [s], `least_clearance` [m], `overloads`, `alarms` and `restarts`.

With the class of the reference answer of Exercise 10.1 saved as `mine/patient.py`, beside `code/`, this script runs it, from the folder that holds `code/`:

```python
# run_mine.py: from the folder that holds code/, python3 run_mine.py
import sys
sys.path[:0] = ["code/ch10", "mine"]       # the core and the cart; the module's folder
import allovi
import cart                                # registers the cart's modules
import patient                             # registers reader.patient (Exercise 10.1)

task = cart.modules.Operation()            # announced in period 140, due in period 200
config = allovi.build("patient", cart.T, {
    "control": ("cart.constant-command",), "proposal": ("reader.patient",),
    "capability": ("cart.forecasting",), "demand": ("cart.task-plan", {"task": task}),
    "task": task, "robot": ("cart.robot",), "world": ("cart.track",),
    "evaluator": ("cart.metrics", {"task": task})})
record = allovi.run(config, 220)           # 22 s; strict=True: a failing module stops it
metrics = config.evaluator.evaluate(record)
states = [p.state for p in record.periods] + [record.final_state]
arrival = next(k for k, s in enumerate(states) if s[0] >= 24)
rest = next(k for k in range(arrival, len(states)) if states[k][1] == 0)
print("arrival in period", arrival, "at", cart.fmt(metrics["arrival_in_work_zone"]), "s")
print("first at rest at", cart.fmt(states[rest][0]), "m, in period", rest)
print("operation begun in period", task.begun, "; alarms:", record.alarms)
```

It prints the numbers of the reference answer:

```
arrival in period 86 at 8.6 s
first at rest at 24.405 m, in period 98
operation begun in period 200 ; alarms: []
```

## 11. Checking a module

From `code/ch10`, `python3 cart_checks.py --module FILE` imports FILE, creates every module that it registered without arguments, puts each in its role in the reference configuration, and prints the report of the suite (Section S10.10.4). For the module of Exercise 10.1, its first line is

```
[proposal reader.patient] 13 passed, 0 failed, 0 not applicable
```

It cannot take three kinds of file, and on the first two it stops with a Python error instead of a report:

- a file whose module needs an argument to be created, such as a demand predictor that reads the task: for `../../ch17/contrib/early_window.py` it stops with `TypeError: EarlyWindow.__init__() missing 1 required positional argument: 'task'`;
- a file that imports a package outside `code/ch10` and `code/ch03`, such as the benchmark's `bench`: for `../../ch17/contrib/full_reverse.py` it stops with `ModuleNotFoundError: No module named 'bench'`, and with `code/ch16` on `PYTHONPATH` it tries the three modules that `bench.modules` registers as well and stops at `bench.metrics`, which needs a world and a task;
- a task, which it reports as not run.

For these, and for a module of another robot, call the suite with a function that builds the configuration, as `contribute.py` does for its two contributions:

```python
# check_mine.py: from the folder that holds code/, python3 check_mine.py
import sys
sys.path[:0] = ["code/ch10", "code/ch17"]  # add "code/ch16" for a module that imports bench
import allovi
import cart
from allovi import checks
import contrib.early_window                # registers reader.early-window


def make():                                # called afresh for every run of the suite
    task = cart.modules.Operation()
    return allovi.build("early-window", cart.T, {
        "control": ("cart.constant-command",), "proposal": ("cart.naive",),
        "capability": ("cart.forecasting",),
        "demand": ("reader.early-window", {"task": task, "early": 10}),
        "task": task, "robot": ("cart.robot",), "world": ("cart.track",),
        "evaluator": ("cart.metrics", {"task": task})})


results = checks.check(make, 220, cart.fmt)
print(checks.summary(results))
for r in results:
    print("  " + r.line())
```

It prints

```
13 passed, 0 failed, 0 not applicable
  passed         interface     registration and declarations (8 cases)
  passed         interface     outputs of the contracted kinds (220 cases)
  passed         consistency   fallback keeps its shifted profile (220 cases)
  passed         consistency   labels at least as informative as the global capability (105526 cases)
  passed         consistency   the global capability beyond the horizon (105526 cases)
  passed         consistency   labelling maps stable within their period (105526 cases)
  passed         consistency   inputs within the declared bounds (220 cases)
  passed         consistency   states with the declared components (221 cases)
  passed         consistency   a second run repeats every record (1760 cases)
  passed         falsification executed labels valid (220 cases)
  passed         falsification true reserve at least the level in force (220 cases)
  passed         falsification no alarm while every executed label was valid (220 cases)
  passed         module        own test refuses plans that brake beyond their labels (16 cases)
```

With `"code/ch16"` added to the path, `import contrib.full_reverse`, `"control": ("reader.full-reverse",)` and `"demand": ("cart.task-plan", {"task": task})`, the same script prints `13 passed, 0 failed, 0 not applicable` for the full reverse. `make()` is called once for each run of the suite, so that every run starts with a fresh task. A passed check discharges no obligation (Section S10.10.1).

## 12. Comparing in the benchmark's family

Steps 6 to 9 of Figure S17.1 compare the module with ALLOVI as Chapter 16 configured it, C6, with that one module replaced, in a family of worlds drawn by `../../ch16/bench/worlds.py`: `World(seed)` draws the ramp, the survey, the forecast onset, the lead time and the slip. `../../ch17/contribute.py` does this for the two contributions, and a comparison of one's own can start from a copy of it (Section 17.1.2). Its function `benchmark_configuration(kind, world)` builds a configuration, `run((kind, seed))` runs one configuration in one world, and `hypotheses` applies each decision rule through the class `Rule`. `run` sets `ce.RAMP` to the world's ramp for the run and sets it back afterwards: the functions of `ce`, the track and the oracle of the checks read the ramp from there (section 3). The labels of C6 are those of the survey predictor with the calibrated offset, `capability("survey", world)` of `../../ch16/cart_benchmark.py`, and a world is valid if all of them are, `valid(world)`. A new comparison draws fresh seeds: 16001 to 16100 are the benchmark's, and 16101 to 16200 those of Chapter 17.

```python
# one_world.py: from the folder that holds code/, python3 one_world.py
import sys
sys.path[:0] = ["code/ch17"]               # contribute.py puts code/ch16 and code/ch10 there too
import contribute as cn                    # the comparison of Section 17.1

for kind in ("C6", "early-window"):
    m = cn.run((kind, 16101))              # one configuration in world 16101
    print(kind, "begun", m["begun"], "delay", m["delay"], "in the work zone", m["in_work_zone"])
```

It prints what `../../ch17/results/contribution-runs.csv` records for world 16101:

```
C6 begun 218 delay 0 in the work zone 122
early-window begun 218 delay 0 in the work zone 115
```

A configuration of the benchmark of Chapter 16 runs alone in the same way, through the function `run` of `../../ch16/cart_benchmark.py`, which takes the configuration's id in `../../ch16/HYPOTHESES.md` and a seed, and returns the evaluator's metrics and, for the configurations of Figure 16.2, the positions of the run:

```python
# one_config.py: from the folder that holds code/, python3 one_config.py
import sys
sys.path[:0] = ["code/ch16"]               # cart_benchmark.py puts code/ch10 and code/ch07 there too
import cart_benchmark as cb                # the anticipation benchmark of Chapter 16

m, _ = cb.run(("C11", 16001))              # the preview planner under the filter, world 16001
print("C11, world 16001: begun", m["begun"], "delay", m["delay"], "arrival", m["arrival"],
      "in the work zone", m["in_work_zone"])
```

It prints what `../../ch16/results/benchmark-runs.csv` records for C11 in world 16001:

```
C11, world 16001: begun 207 delay 0 arrival 88 in the work zone 114
```

## 13. Running the examples

Every example after a `>>>` prompt in this file runs as a doctest. From the folder that holds `code/`,

```
python3 -m doctest code/ch10/templates/HELPERS.md
```

prints nothing if every example holds; with `-v` it lists them and ends with `Test passed.` The examples in the docstrings of `../../ch03/cart_example.py`, `../cart/control.py` and `../cart/modules.py` run in the same way:

```
python3 -m doctest code/ch03/cart_example.py
python3 -c "import sys, doctest; sys.path.insert(0, 'code/ch10'); import cart.control, cart.modules; print(doctest.testmod(cart.control), doctest.testmod(cart.modules))"
```

The last prints `TestResults(failed=0, attempted=2) TestResults(failed=0, attempted=8)`. The scripts of sections 10 to 12 printed what is shown above with the code of this folder.
