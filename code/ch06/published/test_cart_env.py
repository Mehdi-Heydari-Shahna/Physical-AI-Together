#!/usr/bin/env python3
"""
Validation of cart_env.py and protocol.py: the environment is the protocol
of Chapter 6.

Physical AI, Together, Chapter 6, Section 6.2.4.

1. The stand-ins through the environment. The proposal methods of
   ../cart_learning.py are trained and evaluated through cart_env.py and
   protocol.py, exactly as cart_learning.job trains and evaluates them: the
   rule, the braking curve and the expert planner; the off-policy learner,
   the planner with a learned model and the chunk imitator, seeds 1 to 3;
   the off-policy learner with the contraction-metric package; and the two
   runs with late proposals or a slow inference. Every training episode and
   every evaluation run must equal its row of ../results/training.csv and
   ../results/evaluation.csv, and every value of their groups of
   ../results/summary.txt, among them the cells of Table 6.3
   (tab:learning-runs), must be reproduced.
2. The paths that the libraries use. The off-policy learner and the planner
   are trained again with after_arrival="terminate", the mode in which
   Stable-Baselines3 and TD-MPC2 train, the run after arrival being finished
   by the environment (with protocol.after_arrival, which records the
   arrival period before the proposals after it, in the order of calls of
   the protocol); their training episodes and evaluation runs must not
   change. The frozen off-policy learner and planner of seed 1 are evaluated
   again with their proposals given as float32 arrays, references
   normalized by 1.875 m/s^2, as the libraries give actions; their runs must
   not change. The chunk imitator is evaluated again through
   protocol.ChunkPolicy, the timing that the diffusion policy uses; its runs
   must not change.
3. A random-policy smoke test: uniformly random proposals (references and
   sequences, with and without missed deadlines, both modes), in the 20
   evaluation worlds and ten training worlds. In every period the governor
   must commit a plan; no run may raise an alarm, leave a funnel or reach
   the reserve floor, and every run that arrives must come to rest before
   it. The Gymnasium API is checked as well, with gymnasium's and
   Stable-Baselines3's own checkers when they are installed.

Usage
    python3 test_cart_env.py              # all of it; writes results/validation.txt
    python3 test_cart_env.py --check      # exits with status 1 on any difference
    python3 test_cart_env.py --quick      # the rule, the braking curve and the smoke test
    python3 test_cart_env.py --jobs N     # processes (default: up to 2)

The full validation takes about half an hour with two processes.
"""
import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cart_env      # noqa: E402
import protocol      # noqa: E402

cl, cf = cart_env.cl, cart_env.cf
BOOK_RESULTS = os.path.join(HERE, "..", "results")
TABLE_ROWS = [   # Table 6.3 (tab:learning-runs): the group of summary.txt per row
    ("Rule", "rule"),
    ("Braking curve, by hand", "braking curve"),
    ("Expert planner, the demonstrator", "expert planner"),
    ("Off-policy learner", "off-policy learner"),
    ("Planner with a learned model", "planner"),
    ("Chunk imitator", "chunk imitator"),
    ("Off-policy learner with the contraction metric", "off-policy learner with ccm"),
    ("Off-policy learner, 30 % of proposals late", "off-policy learner, missed deadlines"),
    ("Chunk imitator, new chunks two periods late", "chunk imitator, latency 2"),
]
TABLE_KEYS = [("Runs arrived", "arrived"), ("Mean arrival", "arrival"),
              ("Earliest, latest", "arrival_range"), ("Mean intervention", "mean_intervention"),
              ("Largest position", "largest_position_all")]


# ----------------------------------------------------------------------------
# 1 and 2: the stand-ins through the environment
# ----------------------------------------------------------------------------
def make_env(proposal, package, after="continue", miss=0.0):
    return cart_env.CartGovernedEnv(proposal=proposal, package=package, after_arrival=after,
                                    miss=miss)


def train_through(env, pm, worlds, phase="training"):
    log = []
    for i, w in enumerate(worlds):
        m = protocol.standin_episode(env, pm, w, train=True)
        m.update(episode=i, phase=phase)
        log.append(m)
    return log


def evaluate_through(env, pm, worlds):
    return [protocol.standin_episode(env, pm, w) for w in worlds]


class AsFloat32(cl.ProposalMethod):
    """A frozen stand-in whose proposals are passed as float32 arrays: a
    reference divided by the declared largest acceleration, a sequence in
    m/s^2."""

    def __init__(self, pm):
        self.pm, self.kind = pm, pm.kind

    def reset(self, train):
        assert not train
        self.pm.reset(False)

    def propose(self, obs):
        q = self.pm.propose(obs)
        if q is None:
            return None
        if self.kind == "reference":
            return np.array([float(q) / float(cl.DECLARED["a_max"])], dtype=np.float32)
        return np.array([float(a) for a in q], dtype=np.float32)

    def record(self, row):
        self.pm.record(row)

    def end_episode(self):
        self.pm.end_episode()


def job(spec):
    """One unit of the protocol, as cart_learning.job, through cart_env."""
    name, seed, package, mode, cfg = spec
    t0 = time.time()
    cart_env.activate(package)
    ev_worlds, n_train = cfg["eval"], cfg["train"]
    log, ev, extra = [], [], {}
    if name in ("rule", "braking curve", "expert planner"):
        pm = {"rule": cl.Rule, "braking curve": cl.BrakingCurve, "expert planner": cl.expert}[name]()
        ev = evaluate_through(make_env(pm.kind if pm.kind == "reference" else "sequence",
                                       package), pm, ev_worlds)
    elif name.startswith("off-policy learner") or name == "planner":
        kind = "sequence" if name == "planner" else "reference"
        tr_env = make_env(kind, package, after=mode)
        pm = cl.Planner() if name == "planner" else cl.OffPolicyLearner(seed)
        if mode == "terminate":
            tr_env.post_arrival_policy = protocol.after_arrival(pm)
        log = train_through(tr_env, pm, protocol.training_worlds(seed, n_train))
        miss = 0.3 if "missed" in name else 0.0
        ev = evaluate_through(make_env(kind, package, miss=miss), pm, ev_worlds)
        if mode == "continue" and package == "tube" and seed == 1 and "," not in name:
            # the same frozen policy, its proposals given as float32 arrays, as the
            # libraries give actions: normalized by the declared largest acceleration
            # for references (SAC, TD-MPC2), in m/s^2 for sequences (diffusion policy)
            f_env = cart_env.CartGovernedEnv(proposal=kind, normalize_actions=(kind == "reference"))
            extra["float_actions"] = evaluate_through(f_env, AsFloat32(pm), ev_worlds)
    elif name.startswith("chunk imitator"):
        env = make_env("sequence", package)
        im = cl.ChunkImitator(latency=2 if "latency" in name else 0)
        teacher = cl.expert()
        rec = cl.Demonstration(teacher, teacher)
        for w in protocol.demonstration_worlds(seed)[:cfg["demos"]]:
            m = protocol.standin_episode(env, rec, w)
            m.update(episode=len(log), phase="demonstration")
            log.append(m)
        im.D = list(rec.data)
        agg = cl.Demonstration(teacher, cl.ChunkImitator())
        agg.act.D = im.D
        for w in protocol.aggregation_worlds(seed)[:cfg["demos"]]:
            m = protocol.standin_episode(env, agg, w)
            m.update(episode=len(log), phase="aggregation")
            log.append(m)
        im.D = im.D + agg.data
        ev = evaluate_through(env, im, ev_worlds)
        extra = dict(demonstrated=len(rec.data), aggregated=len(agg.data),
                     partial=sum(1 for a in rec.ambitions if a < 1),
                     none=sum(1 for a in rec.ambitions if a == 0), periods=len(rec.ambitions))
        if "latency" not in name:      # the timing of protocol.ChunkPolicy
            served = protocol.ChunkPolicy(lambda obs, info: im.infer(info["obs"]))
            extra["chunk_policy"] = []
            for w in ev_worlds:
                served.reset()
                extra["chunk_policy"].append(protocol.run_episode(env, served, w))
    cart_env.activate("tube")
    return dict(name=name, seed=seed, package=package, mode=mode, log=log, eval=ev,
                extra=extra, seconds=round(time.time() - t0, 1))


# ----------------------------------------------------------------------------
# 3: random proposals
# ----------------------------------------------------------------------------
def smoke(spec):
    """Random proposals in one world: (proposal kind, mode, miss, bias, world)."""
    kind, mode, miss, bias, world = spec
    env = cart_env.CartGovernedEnv(proposal=kind, after_arrival=mode, miss=miss,
                                   normalize_actions=(kind == "reference"))
    env.action_space.seed(world)
    lo = 0.0 if not bias else -0.3      # forward-biased: mostly accelerating

    def act(obs, info):
        a = env.action_space.sample()
        return np.maximum(a, lo * env.action_space.high) if bias else a
    if mode == "terminate":
        env.post_arrival_policy = act
    problems, periods, commits = [], 0, 0
    obs, info = env.reset(options={"world": world})
    if not (isinstance(obs, np.ndarray) and obs.dtype == np.float32 and obs.shape == (9,)):
        problems.append("reset: observation is not a float32 vector of 9")
    if not isinstance(info, dict) or "obs" not in info:
        problems.append("reset: info lacks the observation dictionary")
    while True:
        out = env.step(act(obs, info))
        if len(out) != 5:
            problems.append("step: not a 5-tuple")
        obs, r, term, trunc, info = out
        periods += 1
        commits += 1 if info["row"]["sel"] is not None else 0
        if not (isinstance(r, float) and np.isfinite(r)):
            problems.append(f"step: reward {r!r}")
        if not (isinstance(term, bool) and isinstance(trunc, bool)):
            problems.append("step: terminated or truncated is not a bool")
        if obs.shape != (9,) or obs.dtype != np.float32 or not np.all(np.isfinite(obs)):
            problems.append("step: bad observation")
        if term or trunc:
            break
    try:
        env.step(act(obs, info))
        problems.append("step after the end did not raise")
    except RuntimeError:
        pass
    m = info["metrics"]
    return dict(spec=spec, problems=problems, periods=periods, commits=commits,
                rows=len(env.rows), alarms=m["alarms"], out=m["out"], pmax=m["pmax"],
                arrived=m["periods"] is not None, rest=m["rest"], end=m["end"])


def api_checks():
    """The Gymnasium API, with the checkers of gymnasium and
    Stable-Baselines3 when they are installed."""
    notes = []
    env = cart_env.CartGovernedEnv(worlds=[1000, 1001], normalize_actions=True)
    o1, i1 = env.reset(seed=123)
    o2, i2 = env.reset(seed=123)
    o3, i3 = env.reset()
    o4, i4 = env.reset(options={"world": 1005})
    notes.append(f"schedule [1000, 1001]: reset(seed=123) twice, reset(), "
                 f"reset(options={{'world': 1005}}) give the worlds {i1['world']}, "
                 f"{i2['world']}, {i3['world']}, {i4['world']}")
    ok = ([i1["world"], i2["world"], i3["world"], i4["world"]] == [1000, 1000, 1001, 1005]
          and np.array_equal(o1, o2))
    ok = ok and env.action_space.contains(env.action_space.sample())
    notes.append(f"action space {env.action_space}; observation space shape "
                 f"{env.observation_space.shape}")
    # the two rewards (PROTOCOL.md and PROTOCOL-R2.md): the same run for the same
    # proposals, and each period's reward as its formula gives it; the rule, the
    # braking curve and seeded random proposals, in three worlds
    import random as _random

    class RandomRef(cl.ProposalMethod):
        def __init__(self, seed):
            self.rng = _random.Random(seed)

        def propose(self, obs):
            return 2 * float(cl.DECLARED["a_max"]) * self.rng.random() - float(cl.DECLARED["a_max"])

    def reward_run(make, kind, world):
        renv = cart_env.CartGovernedEnv(reward=kind)
        pm = make()
        obs, info = renv.reset(options={"world": world})
        pm.reset(False)
        rows, bad = [], 0
        while True:
            obs, r, term, trunc, info = renv.step(pm.propose(info["obs"]))
            row = info["row"]
            c = 1 - float(row["alpha"])
            a = 0.0 if row["a_next"] is None else float(row["a_next"])
            want = (0.0 if row["after"] else
                    (a / float(cl.DECLARED["a_max"]) - cart_env.CHI * c if kind == "registered"
                     else -1.0 - cart_env.CHI * c))
            bad += abs(r - want) > 1e-12
            rows.append((row["k"], row["q"], row["alpha"], row["sel"], row["p1"], row["v1"]))
            if term or trunc:
                break
        return rows, info["metrics"]["arrival"], bad

    n_runs, n_periods, n_same, n_bad = 0, 0, 0, 0
    for name, make in (("rule", cl.Rule), ("braking curve", cl.BrakingCurve),
                       ("random", None)):
        for world in (1000, 1001, 1002):
            mk = make if make is not None else (lambda w=world: RandomRef(10 ** 6 + w))
            reg = reward_run(mk, "registered", world)
            tim = reward_run(mk, "time", world)
            n_runs += 1
            n_periods += len(tim[0])
            n_same += reg[:2] == tim[:2]
            n_bad += reg[2] + tim[2]
    same = n_same == n_runs
    notes.append(f"rewards: the rule, the braking curve and seeded random proposals in the worlds "
                 f"1000 to 1002, each run with reward='registered' and reward='time': "
                 f"{n_same} of {n_runs} runs the same under both ({n_periods} periods); periods "
                 f"whose reward differs from its formula: {n_bad}")
    ok = ok and same and n_bad == 0
    if cart_env.GYMNASIUM:
        from gymnasium.utils.env_checker import check_env as gym_check
        gym_check(cart_env.CartGovernedEnv(worlds=list(protocol.EVAL_WORLDS),
                                           normalize_actions=True), skip_render_check=True)
        notes.append(f"gymnasium {cart_env.GYMNASIUM}: gymnasium.utils.env_checker.check_env passed")
    else:
        notes.append("gymnasium is not installed: the local stand-in for Box was used, and "
                     "gymnasium's checker was not run")
    try:
        from stable_baselines3.common.env_checker import check_env as sb3_check
        sb3_check(cart_env.CartGovernedEnv(worlds=list(protocol.EVAL_WORLDS),
                                           normalize_actions=True))
        notes.append("Stable-Baselines3: stable_baselines3.common.env_checker.check_env passed")
    except ImportError:
        notes.append("Stable-Baselines3 is not installed: its checker was not run")
    return ok, notes


# ----------------------------------------------------------------------------
# Comparison with the book's results
# ----------------------------------------------------------------------------
def read_summary(path):
    """Groups of summary.txt: {group: {key: (value, line number)}}."""
    groups, g = {}, None
    with open(path) as fh:
        for n, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if line.startswith("["):
                g = line.strip("[]")
                groups[g] = {}
            elif " = " in line and g is not None:
                k, v = line.strip().split(" = ", 1)
                groups[g][k] = (v, n)
    return groups


def read_csv(path):
    with open(path, newline="") as fh:
        r = csv.reader(fh)
        header = next(r)
        return header, [row for row in r]


def run_strings(m):
    """The fields of a run as the csv module writes them (None as '')."""
    return ["" if x is None else str(x) for x in protocol.run_fields(m)]


def compare_runs(results, eval_rows, train_rows):
    """Every evaluation run and training episode against its row of the
    book's CSV files. Returns the numbers compared, by mode ('continue' or
    'terminate') and kind ('evaluation' or 'training'), and the list of
    differences."""
    n, diffs = {}, []
    for r in results:
        name, seed, package, mode = r["name"], str(r["seed"]), r["package"], r["mode"]
        want = [row[4:] for row in eval_rows if row[:3] == [name, package, seed]]
        got = [run_strings(m) for m in r["eval"]]
        worlds = [row[3] for row in eval_rows if row[:3] == [name, package, seed]]
        if [str(m["world"]) for m in r["eval"]] != worlds:
            diffs.append(f"{name}/{package}/{seed} ({mode}): evaluation worlds differ")
        for w, a, b in zip(worlds, got, want):
            n[mode, "evaluation"] = n.get((mode, "evaluation"), 0) + 1
            if a != b:
                diffs.append(f"{name}/{package}/{seed} ({mode}) world {w}: {a} != {b}")
        if len(got) != len(want):
            diffs.append(f"{name}/{package}/{seed}: {len(got)} evaluation runs, book {len(want)}")
        want_t = [row[3:] for row in train_rows if row[:3] == [name, package, seed]]
        got_t = []
        for m in r["log"]:
            f = run_strings(m)
            got_t.append([str(m["episode"]), m["phase"], str(m["world"]),
                          f[0], f[1], f[3], f[4], f[5], f[6]])
        for a, b in zip(got_t, want_t):
            n[mode, "training"] = n.get((mode, "training"), 0) + 1
            if a != b:
                diffs.append(f"{name}/{package}/{seed} ({mode}) training: {a} != {b}")
        if len(got_t) != len(want_t):
            diffs.append(f"{name}/{package}/{seed}: {len(got_t)} training episodes, "
                         f"book {len(want_t)}")
    return n, diffs


def groups_of(results):
    """The groups of summary.txt that the runs through the environment give."""
    by = {}
    for r in results:
        if r["mode"] != "continue":
            continue
        key = r["name"] if r["package"] == "tube" else f"{r['name']} with {r['package']}"
        by.setdefault(key, []).append(r)
    groups = {}
    for key, rs in by.items():
        rs = sorted(rs, key=lambda r: r["seed"])
        trains = [[m for m in r["log"]] for r in rs]
        d = protocol.summarize([r["eval"] for r in rs], trains if any(trains) else None)
        d["seeds"] = ",".join(str(r["seed"]) for r in rs)
        ex = [r["extra"] for r in rs if "demonstrated" in r["extra"]]
        if ex:
            d["demonstrated_states"] = str(sum(e["demonstrated"] for e in ex))
            d["aggregated_states"] = str(sum(e["aggregated"] for e in ex))
            n_dem = sum(e["periods"] for e in ex)
            d["demonstrated_chunks_not_committed_in_full"] = (
                f"{sum(e['partial'] for e in ex)} of {n_dem}")
            d["demonstrated_chunks_not_committed_at_all"] = (
                f"{sum(e['none'] for e in ex)} of {n_dem}")
        if "missed" in key or "imitator" in key:     # as cart_learning.report
            d.pop("training_intervention_first_10", None)
            d.pop("training_intervention_last_10", None)
        groups[key] = d
    return groups


def main():
    ap = argparse.ArgumentParser(description="Validation of the governed cart environment")
    ap.add_argument("--check", action="store_true", help="exit with status 1 on a difference")
    ap.add_argument("--quick", action="store_true", help="the rule, the braking curve, the smoke test")
    ap.add_argument("--jobs", type=int, default=min(2, os.cpu_count() or 1))
    args = ap.parse_args()
    t_start = time.time()
    cfg = dict(eval=list(protocol.EVAL_WORLDS), train=protocol.N_WORLDS, demos=protocol.DEMOS)
    specs = [("rule", 0, "tube", "continue", cfg), ("braking curve", 0, "tube", "continue", cfg)]
    if not args.quick:
        specs += [("expert planner", 0, "tube", "continue", cfg)]
        specs += [(n, s, "tube", "continue", cfg) for n in
                  ("off-policy learner", "planner", "chunk imitator") for s in protocol.SEEDS]
        specs += [("off-policy learner", s, "ccm", "continue", cfg) for s in protocol.SEEDS]
        specs += [("off-policy learner, missed deadlines", 1, "tube", "continue", cfg),
                  ("chunk imitator, latency 2", 1, "tube", "continue", cfg)]
        specs += [(n, s, "tube", "terminate", cfg) for n in ("off-policy learner", "planner")
                  for s in protocol.SEEDS]
    smoke_worlds = list(protocol.EVAL_WORLDS) + [100 + i for i in range(10)]
    smokes = []
    for i, w in enumerate(smoke_worlds):
        smokes += [("reference", "continue", 0.0, False, w), ("reference", "continue", 0.0, True, w),
                   ("sequence", "continue", 0.0, True, w),
                   ("reference", "terminate", 0.0, True, w),
                   ("reference" if i % 2 else "sequence", "continue", 0.3, True, w)]
    if args.quick:
        smokes = smokes[:25]
    order = sorted(range(len(specs)), key=lambda i: -("learner" in specs[i][0] or
                                                      "planner" == specs[i][0]))
    with Pool(args.jobs) as pool:
        out = pool.map(job, [specs[i] for i in order])
        smoke_out = pool.map(smoke, smokes)
    results = [None] * len(specs)
    for i, r in zip(order, out):
        results[i] = r
    api_ok, api_notes = api_checks()

    summary = read_summary(os.path.join(BOOK_RESULTS, "summary.txt"))
    _, eval_rows = read_csv(os.path.join(BOOK_RESULTS, "evaluation.csv"))
    _, train_rows = read_csv(os.path.join(BOOK_RESULTS, "training.csv"))
    n_runs, run_diffs = compare_runs(results, eval_rows, train_rows)
    got = groups_of(results)
    lines, failures = [], []
    lines.append("Validation of code/ch06/published/cart_env.py against code/ch06/results/ "
                 "(written by test_cart_env.py)")
    lines.append(f"python {sys.version.split()[0]}; gymnasium "
                 f"{cart_env.GYMNASIUM or 'not installed (local stand-in for Box)'}; numpy "
                 f"{np.__version__}; {'quick subset' if args.quick else 'full validation'}")
    lines.append("")
    lines.append("[1. Table 6.3 (tab:learning-runs) through the environment]")
    lines.append("  row; column = value through cart_env (value in ../results/summary.txt, line)")
    for label, group in TABLE_ROWS:
        if group not in got:
            continue
        for col, key in TABLE_KEYS:
            v = got[group].get(key)
            want, line = summary[group].get(key, ("missing", 0))
            mark = "" if v == want else "   <-- DIFFERENT"
            if v != want:
                failures.append(f"Table 6.3, {label}, {col}: {v} != {want}")
            lines.append(f"  {label}; {col} = {v} ({want}, summary.txt line {line}){mark}")
    lines.append("")
    lines.append("[1. every value of the same groups of summary.txt]")
    n_vals = 0
    for group, d in got.items():
        for key, v in d.items():
            if key not in summary.get(group, {}):
                continue
            n_vals += 1
            want, line = summary[group][key]
            if v != want:
                failures.append(f"summary.txt [{group}] {key} (line {line}): {v} != {want}")
    lines.append(f"  {n_vals} values of {len(got)} groups compared with summary.txt: "
                 + ("all equal" if not any("summary.txt" in f for f in failures) else "DIFFERENT"))
    cont = [d for d in run_diffs if "(continue)" in d or "(" not in d]
    lines.append(f"  {n_runs.get(('continue', 'evaluation'), 0)} evaluation runs and "
                 f"{n_runs.get(('continue', 'training'), 0)} training episodes compared, row by "
                 f"row, with evaluation.csv and training.csv: "
                 + ("all equal" if not cont else f"{len(cont)} DIFFERENT"))
    failures += run_diffs[:50]
    lines.append("")
    lines.append("[2. the paths that the libraries use]")
    term = [r for r in results if r["mode"] == "terminate"]
    if term:
        lines.append(f"  after_arrival='terminate', the run after arrival finished by the "
                     f"environment: {len(term)} trained policies, "
                     f"{n_runs.get(('terminate', 'training'), 0)} training episodes and "
                     f"{n_runs.get(('terminate', 'evaluation'), 0)} evaluation runs compared, row "
                     f"by row, with the same files: "
                     + ("all equal" if not any("(terminate)" in d for d in run_diffs)
                        else "DIFFERENT"))
    cp = [(r, m, c) for r in results if r["extra"].get("chunk_policy")
          for m, c in zip(r["eval"], r["extra"]["chunk_policy"])]
    if cp:
        same = all(run_strings(m) == run_strings(c) and m["full"] == c["full"]
                   and m["n_pre"] == c["n_pre"] for _, m, c in cp)
        lines.append(f"  protocol.ChunkPolicy serving the chunk imitator's inferences: {len(cp)} "
                     f"evaluation runs equal to the imitator's own: {'yes' if same else 'NO'}")
        if not same:
            failures.append("ChunkPolicy differs from ChunkImitator")
    fl = [(r, m, c) for r in results if r["extra"].get("float_actions")
          for m, c in zip(r["eval"], r["extra"]["float_actions"])]
    if fl:
        keys = ("full", "n_pre", "unchanged", "at_max")
        same = all(run_strings(m) == run_strings(c) and all(m[k] == c[k] for k in keys)
                   for _, m, c in fl)
        who = " and ".join(sorted({r["name"] for r, _, _ in fl}))
        lines.append(f"  float32 actions, as the libraries give them ({who}, seed 1; references "
                     f"normalized by 1.875 m/s^2): {len(fl)} evaluation runs equal to those with "
                     f"exact proposals: {'yes' if same else 'NO'}")
        if not same:
            failures.append("float32 actions differ from exact proposals")
    lines.append("")
    lines.append("[3. random proposals]")
    bad = [s for s in smoke_out if s["problems"] or s["alarms"] or s["out"]
           or s["pmax"] > protocol.FLOOR or (s["arrived"] and s["rest"] is None)
           or s["commits"] != s["periods"]]
    lines.append(f"  {len(smoke_out)} episodes, {sum(s['rows'] for s in smoke_out)} periods "
                 f"({sum(s['periods'] for s in smoke_out)} returned to the policy); a plan "
                 f"committed in every period: "
                 f"{'yes' if all(s['commits'] == s['periods'] for s in smoke_out) else 'NO'}")
    lines.append(f"  alarms {sum(s['alarms'] for s in smoke_out)}; steps outside a funnel "
                 f"{sum(s['out'] for s in smoke_out)}; largest position "
                 f"{max(s['pmax'] for s in smoke_out):.3f} m (floor {protocol.FLOOR} m); arrived "
                 f"{sum(s['arrived'] for s in smoke_out)}, of which at rest "
                 f"{sum(1 for s in smoke_out if s['arrived'] and s['rest'] is not None)}")
    ends = {}
    for s in smoke_out:
        ends[s["end"]] = ends.get(s["end"], 0) + 1
    lines.append("  ends: " + ", ".join(f"{k} {v}" for k, v in sorted(ends.items())))
    lines.append(f"  episodes with a problem: {len(bad)}")
    for s in bad[:20]:
        failures.append(f"smoke {s['spec']}: {s['problems']} alarms {s['alarms']} "
                        f"out {s['out']} pmax {s['pmax']:.3f}")
    lines.append(f"  Gymnasium API: {'as required' if api_ok else 'NOT AS REQUIRED'}")
    for n in api_notes:
        lines.append(f"    {n}")
    if not api_ok:
        failures.append("Gymnasium API")
    lines.append("")
    lines.append("[timing]")
    for r in results:
        lines.append(f"  {r['name']} / {r['package']} / seed {r['seed']} / {r['mode']}: "
                     f"{r['seconds']} s")
    lines.append(f"  total {time.time() - t_start:.0f} s with {args.jobs} processes")
    lines.append("")
    lines.append("[check] " + ("all as in code/ch06/results" if not failures else
                               "FAILED: " + "; ".join(failures[:20])))
    os.makedirs(protocol.RESULTS, exist_ok=True)
    name = "validation-quick.txt" if args.quick else "validation.txt"
    with open(os.path.join(protocol.RESULTS, name), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 1 if (args.check and failures) else 0


if __name__ == "__main__":
    sys.exit(main())
