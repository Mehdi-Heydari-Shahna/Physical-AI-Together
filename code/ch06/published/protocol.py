#!/usr/bin/env python3
"""
The protocol of Section 6.2 for the published algorithms: worlds, budgets,
episode drivers, the timing of a chunk, the metrics files and the record.

Physical AI, Together, Chapter 6, Section 6.2.4. PROTOCOL.md states the
protocol in words; this module is what the scripts run. test_cart_env.py
checks the drivers, the timing of a chunk and the summaries against the
stand-ins' results of ../results/, so that the published algorithms are
evaluated by code that reproduces the chapter's numbers.

Worlds (as in ../cart_learning.py)
    evaluation          the 20 worlds with seeds 1000 to 1019
    training, seed s    the worlds 100 s, ..., 100 s + 39, visited in this
                        order; a budget of more than 40 episodes visits them
                        again in the same order (episode i: 100 s + i mod 40)
    demonstrations      the worlds 100 s + 50, ..., 100 s + 54 (the expert)
    aggregation         the worlds 100 s + 55, ..., 100 s + 59 (the learner
                        acts, the expert labels)
Budgets (PROTOCOL.md)
    SAC, TD-MPC2        M = 40 training episodes, the stand-ins' budget, and
                        L = 1000 training episodes, 25 passes over the same
                        40 worlds; the policy of budget M is the snapshot of
                        the same training run after episode 40
    diffusion policy    5 expert and 5 aggregation episodes, as the chunk
                        imitator
"""
import csv
import datetime
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from fractions import Fraction as Fr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cart_env    # noqa: E402

cl, cf = cart_env.cl, cart_env.cf

SEEDS = (1, 2, 3)
EVAL_WORLDS = tuple(cl.EVAL_SEEDS)                 # 1000, ..., 1019
N_WORLDS = cl.TRAIN_EPISODES                       # 40 training worlds per seed
BUDGETS = {"M": 40, "L": 1000}                     # training episodes, SAC and TD-MPC2
DEMOS = cl.DEMOS                                   # 5 expert, then 5 aggregation episodes
FLOOR = float(cf.WALL - cf.EPS_MIN)                # 24.5 m: the reserve floor
RESULTS = os.path.join(HERE, "results")


def training_worlds(seed, episodes):
    """The worlds of the training episodes of seed s, in order."""
    return [100 * seed + (i % N_WORLDS) for i in range(episodes)]


def demonstration_worlds(seed):
    return [100 * seed + 50 + i for i in range(DEMOS)]


def aggregation_worlds(seed):
    return [100 * seed + 50 + DEMOS + i for i in range(DEMOS)]


# ----------------------------------------------------------------------------
# Drivers
# ----------------------------------------------------------------------------
def run_episode(env, act, world):
    """One episode of env in a world, with act(obs, info) -> action.
    Returns the metrics of cart_learning.run_metrics (with the world)."""
    obs, info = env.reset(options={"world": world})
    while True:
        obs, _, terminated, truncated, info = env.step(act(obs, info))
        if terminated or truncated:
            return info["metrics"]


def no_explore(pm):
    """The proposal of a stand-in after arrival, as cart_learning.episode
    forms it: the method keeps proposing, without exploring."""
    def act(obs, info):
        training = getattr(pm, "train", False)
        pm.train = False
        q = pm.propose(info["obs"])
        pm.train = training
        return q
    return act


def after_arrival(pm):
    """The post-arrival policy of a stand-in in mode 'terminate', in the
    order of calls of cart_learning.episode: the arrival period is recorded
    first, then the method keeps proposing without exploring. In this mode
    the environment makes those proposals before step() returns the arrival
    period, and the stand-ins' record() changes what they propose next (the
    off-policy learner's last proposal, the planner's margin), so the
    wrapper records the arrival period, which the first call receives in
    its info, and standin_episode does not record it again. A library
    learner records the transition that step() returns instead."""
    act = no_explore(pm)

    def f(obs, info):
        row = info.get("row")
        if row is not None and row["done"] and f.recorded is None:
            pm.record(row)
            f.recorded = row["k"]
        return act(obs, info)
    f.recorded = None
    return f


def standin_episode(env, pm, world, train=False):
    """One episode of a proposal method of cart_learning.py through the
    environment, in the order of cart_learning.episode: the method proposes
    from the observation dictionary, records every period up to arrival and
    ends the episode. In mode 'terminate', the environment finishes the run
    after arrival with after_arrival(pm), which must be set as its
    post_arrival_policy."""
    obs, info = env.reset(options={"world": world})
    pm.reset(train)
    after = no_explore(pm)
    post = getattr(env, "post_arrival_policy", None)
    if hasattr(post, "recorded"):
        post.recorded = None
    while True:
        q = after(obs, info) if info["arrived"] else pm.propose(info["obs"])
        obs, _, terminated, truncated, info = env.step(q)
        row = info["row"]
        if not row["after"] and not (post is not None
                                     and getattr(post, "recorded", None) == row["k"]):
            pm.record(row)
        if terminated or truncated:
            break
    pm.end_episode()
    return info["metrics"]


class ChunkPolicy:
    """The timing of a chunk, T_o = 1, T_p = 8, T_a = 4, as the chunk
    imitator of cart_learning.py (ChunkImitator, latency 0) serves its
    chunks: infer(obs, info) returns eight references from the latest
    observation; the policy proposes the remainder of the chunk, padded
    with its last reference to eight, in the following periods, and infers
    again after four periods or after an intervention (alpha_{k-1} < 1).
    Inference times are kept in self.seconds."""

    def __init__(self, infer, n_act=4, horizon=cl.H):
        self.infer, self.n_act, self.H = infer, n_act, horizon
        self.reset()

    def reset(self):
        self.chunk, self.age, self.seconds = None, 0, []

    def __call__(self, obs, info):
        if self.chunk is None or self.age >= self.n_act or info["obs"]["alpha_prev"] < 1:
            t0 = time.perf_counter()
            self.chunk = list(self.infer(obs, info))
            self.seconds.append(time.perf_counter() - t0)
            self.age = 0
        rest = self.chunk[self.age:]
        self.age += 1
        return rest + [rest[-1]] * (self.H - len(rest))


class Timed:
    """A policy act(obs, info) whose calls are timed (inference latency)."""

    def __init__(self, act):
        self.act, self.seconds = act, []

    def __call__(self, obs, info):
        t0 = time.perf_counter()
        a = self.act(obs, info)
        self.seconds.append(time.perf_counter() - t0)
        return a


def latency_summary(seconds):
    """Median and largest inference time, in milliseconds."""
    if not seconds:
        return "none"
    s = sorted(seconds)
    med = s[len(s) // 2] if len(s) % 2 else (s[len(s) // 2 - 1] + s[len(s) // 2]) / 2
    return f"median {1000 * med:.2f} ms, largest {1000 * s[-1]:.2f} ms, {len(s)} calls"


# ----------------------------------------------------------------------------
# Metrics files, in the format of ../results/evaluation.csv and training.csv
# ----------------------------------------------------------------------------
# The first columns are those of ../results/evaluation.csv and training.csv
# (with the budget after the seed in the evaluation file); the last ones hold
# what the summaries need exactly: the counts, the mean intervention as a
# fraction and the positions in full precision.
NA = "not applicable"
EVAL_COLUMNS = ["policy", "package", "seed", "budget", "world", "arrival_s",
                "mean_intervention", "fallbacks", "alarms", "steps_outside", "max_position",
                "rest_position", "periods_before_arrival", "committed_in_full",
                "passed_unchanged", "at_largest_acceleration", "mean_intervention_exact",
                "max_position_exact", "rest_position_exact", "operation"]
TRAIN_COLUMNS = ["learner", "package", "seed", "episode", "phase", "world", "arrival_s",
                 "mean_intervention", "alarms", "steps_outside", "max_position",
                 "rest_position", "periods_before_arrival", "committed_in_full", "fallbacks",
                 "mean_intervention_exact", "max_position_exact", "rest_position_exact"]


def run_fields(m):
    """The fields of one run as ../cart_learning.py writes them."""
    return [m["arrival"], f"{m['load']:.4f}", m["fallbacks"], m["alarms"], m["out"],
            f"{m['pmax']:.4f}", "none" if m["rest"] is None else f"{m['rest']:.4f}"]


def exact_fields(m):
    return [str(m["load_exact"]), repr(m["pmax"]), "none" if m["rest"] is None else repr(m["rest"])]


def write_evaluation(path, policy, package, seed, budget, evals):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(EVAL_COLUMNS)
        for m in evals:
            w.writerow([policy, package, seed, budget, m["world"]] + run_fields(m)
                       + [m["n_pre"], m["full"], NA if m["unchanged"] is None else m["unchanged"],
                          NA if m["at_max"] is None else m["at_max"]]
                       + exact_fields(m) + [NA])


def write_training(path, learner, package, seed, episodes, phase="training", first=0):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(TRAIN_COLUMNS)
        for i, m in enumerate(episodes):
            f = run_fields(m)
            w.writerow([learner, package, seed, first + i, m.get("phase", phase), m["world"],
                        f[0], f[1], f[3], f[4], f[5], f[6], m["n_pre"], m["full"], m["fallbacks"]]
                       + exact_fields(m))


def read_runs(path):
    """The runs of a file written by write_evaluation or write_training, as
    the metrics dictionaries of cart_learning.run_metrics."""
    out = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            arr = row["arrival_s"]
            m = dict(arrival=float(arr) if arr else None,
                     periods=round(float(arr) * 10) if arr else None,
                     load_exact=Fr(row["mean_intervention_exact"]),
                     load=float(Fr(row["mean_intervention_exact"])),
                     fallbacks=int(row["fallbacks"]), alarms=int(row["alarms"]),
                     out=int(row["steps_outside"]), pmax=float(row["max_position_exact"]),
                     rest=(None if row["rest_position_exact"] == "none"
                           else float(row["rest_position_exact"])),
                     n_pre=int(row["periods_before_arrival"]),
                     full=int(row["committed_in_full"]), world=int(row["world"]),
                     seed=int(row["seed"]))
            for k, c in (("unchanged", "passed_unchanged"), ("at_max", "at_largest_acceleration")):
                v = row.get(c, NA)
                m[k] = None if v in (NA, None) else int(v)
            for k in ("budget", "phase", "episode", "policy", "learner"):
                if k in row:
                    m[k] = row[k]
            out.append(m)
    return out


# ----------------------------------------------------------------------------
# Summaries, in the format of ../results/summary.txt
# ----------------------------------------------------------------------------
def summarize(evals_by_seed, trains_by_seed=None):
    """The group of ../results/summary.txt for one policy: evaluation runs
    pooled over the trained policies (a list per seed), training episodes
    (a list per seed, or None), formatted as cart_learning.report formats
    them."""
    evals = [e for ev in evals_by_seed for e in ev]
    d = cl.describe(evals)
    if len(evals_by_seed) > 1:
        d["mean_intervention_by_seed"] = ", ".join(
            cf.r(sum(e["load_exact"] for e in ev) / len(ev), 3) for ev in evals_by_seed)
    tr = [m for t in (trains_by_seed or []) for m in t]
    if tr:
        d["training_episodes"] = str(len(tr))
        d["training_alarms"] = str(sum(m["alarms"] for m in tr))
        d["training_steps_outside_funnel"] = str(sum(m["out"] for m in tr))
        d["training_max_position"] = f"{max(m['pmax'] for m in tr):.3f} m"
        n_arr = sum(1 for m in tr if m["periods"] is not None)
        d["training_arrived"] = f"{n_arr} of {len(tr)}"
        d["training_at_rest_after_arrival"] = (
            f"{sum(1 for m in tr if m['rest'] is not None)} of {n_arr}")
        d["training_periods_before_arrival"] = str(sum(m["n_pre"] for m in tr))
        runs = [t for t in trains_by_seed if t]
        n = min(len(t) for t in runs)
        if n >= 20:
            first = [m["load_exact"] for t in runs for m in t[:10]]
            last = [m["load_exact"] for t in runs for m in t[-10:]]
            d["training_intervention_first_10"] = cf.r(sum(first) / len(first), 3)
            d["training_intervention_last_10"] = cf.r(sum(last) / len(last), 3)
    d["largest_position_all"] = (
        f"{max([e['pmax'] for e in evals] + [m['pmax'] for m in tr]):.3f} m")
    d["operation_timeliness"] = "not applicable (the task of Chapter 6 has no operation)"
    return d


def guarantee_check(evals, trains):
    """What the guarantee predicts for every run whose labels are valid (all
    labels of Chapter 6 are): no alarm, no step outside a funnel, the cart
    never beyond the reserve floor at 24.5 m, and every run that arrives at
    rest. (That the governor commits a plan in every period is checked by
    the environment, which raises an error otherwise.) Returns the counts
    and whether there is no exception."""
    runs = list(evals) + list(trains)
    alarms = sum(m["alarms"] for m in runs)
    out = sum(m["out"] for m in runs)
    pmax = max(m["pmax"] for m in runs)
    arrived = [m for m in runs if m["periods"] is not None]
    not_at_rest = sum(1 for m in arrived if m["rest"] is None)
    beyond = sum(1 for m in runs if m["pmax"] > FLOOR)
    ok = alarms == 0 and out == 0 and not_at_rest == 0 and beyond == 0
    return dict(runs=str(len(runs)), alarms=str(alarms), steps_outside_funnel=str(out),
                arrived_not_at_rest=str(not_at_rest), runs_beyond_floor=str(beyond),
                largest_position=f"{pmax:.3f} m",
                verdict="supported" if ok else "NOT SUPPORTED")


def write_summary(path, groups, header=None):
    lines = list(header or [])
    for group, items in groups.items():
        lines.append(f"[{group}]")
        for k, v in items.items():
            lines.append(f"  {k} = {v}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return lines


# ----------------------------------------------------------------------------
# The record
# ----------------------------------------------------------------------------
def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def cpu_model():
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def git_commit(path):
    """The commit and whether the working tree is clean, of a checkout."""
    try:
        head = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", path, "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return dict(path=os.path.abspath(path), commit=head, clean=not dirty)
    except (OSError, subprocess.CalledProcessError) as e:
        return dict(path=os.path.abspath(path), commit="unknown", error=str(e))


def gpu_record():
    rec = {}
    try:
        import torch
        rec["torch"] = torch.__version__
        rec["torch_cuda"] = torch.version.cuda
        rec["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            rec["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            rec["cudnn"] = torch.backends.cudnn.version()
    except ImportError:
        rec["torch"] = "not installed"
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        if q.returncode == 0:
            rec["nvidia_smi"] = q.stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        pass
    return rec


def pip_freeze():
    try:
        q = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                           text=True, timeout=120)
        return q.stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError) as e:
        return [f"pip freeze failed: {e}"]


def environment_record(packages, checkouts=None):
    """Versions, commits and hardware, for the record of a run."""
    return dict(
        date_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        argv=sys.argv, python=sys.version, platform=platform.platform(),
        machine=platform.machine(), cpu=cpu_model(), cpu_count=os.cpu_count(),
        gymnasium=cart_env.GYMNASIUM or "not installed (local stand-in for Box)",
        packages={p: package_version(p) for p in packages},
        checkouts={k: git_commit(v) for k, v in (checkouts or {}).items()},
        gpu=gpu_record(), pip_freeze=pip_freeze())


def write_record(path, record):
    def default(x):
        if isinstance(x, Fr):
            return str(x)
        if isinstance(x, float) and not math.isfinite(x):
            return str(x)
        return repr(x)
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2, default=default)
        fh.write("\n")


def check_commit(name, path, pinned, allow_other):
    """Refuses a checkout at another commit than the pinned one, unless
    allowed; the record keeps the commit either way."""
    rec = git_commit(path)
    if rec.get("commit") != pinned:
        msg = (f"{name}: the checkout {path} is at {rec.get('commit')}, "
               f"not at the pinned commit {pinned}")
        if not allow_other:
            raise SystemExit(msg + " (use --allow-other-commit to run anyway; "
                             "the record will say so)")
        print("WARNING:", msg)
    return rec
