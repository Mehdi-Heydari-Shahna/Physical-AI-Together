#!/usr/bin/env python3
"""
What the rewards pay for: an exploratory reading of the registered reward,
written after TD-MPC2's results were known (Section 6.2.4; PROTOCOL.md,
Deviations 3), and the reason for the reward of PROTOCOL-R2.md.

Physical AI, Together, Chapter 6. Not part of the protocol: it tests nothing.

The registered reward (cart_env, reward="registered"; the off-policy
learner's, which PROTOCOL.md gives to SAC and TD-MPC2) is, in every period
before arrival,

    r_k = a_k / 1.875 - 0.5 c_k,

a_k the reference acceleration committed for the next period [m/s^2] and
c_k the intervention, one minus the committed ambition; 0 after arrival.
The time reward of PROTOCOL-R2.md (reward="time") is r_k = -1 - 0.5 c_k
before arrival, 0 after. An episode ends for the learner at arrival.

1. Reference policies. The rule, the braking curve and the off-policy
   learner of ../cart_learning.py (seeds 1, 2 and 3, trained as there),
   and a policy that proposes 0 m/s^2 in every period, are run in the 20
   evaluation worlds through cart_env, and the reward of every period
   before arrival is kept. For each, the return of a run, from its first
   period, is given undiscounted and with the discounts of SAC (0.99) and
   of TD-MPC2 under PROTOCOL.md (0.975, derived by its own rule from the
   200 periods of an episode), averaged over the runs. A run that does not
   arrive by 20 s is truncated there, and a learner bootstraps past it: with
   a discount g < 1, its return under the time reward includes the tail
   -g^200 / (1 - g) of a policy that stays where it is (0 under the
   registered reward). The runs are checked against ../results/evaluation.csv:
   arrival and mean intervention must be those of the book.

2. The published methods, from their evaluation files (results/<method>/
   seed*/evaluation.csv). Their rewards are not recorded, so only bounds on
   the undiscounted registered return are given. The first term telescopes
   (sum_k a_k T is the change of the reference speed), except that the
   head of a rest segment counts as no acceleration: the last braking of
   each standstill before arrival, from at most V_H = 0.05 m/s, where the
   rest segment takes over, is not charged, so each such standstill adds up
   to V_H / 0.1875 = 0.27. A run that arrives without stopping on the way
   earns at most V_MAX / (1.875 m/s^2 x 0.1 s) from it, where V_MAX =
   sqrt(2 x 0.375 x 0.5) = 0.612 m/s is the largest reference speed from 24
   m on from which the ramp's declared braking, 0.375 m/s^2, still stops
   the cart by 24.5 m: at most 3.27 - 0.5 S, S the sum of its
   interventions. The evaluation files do not record standstills, so these
   bounds hold for the runs that did not stop on the way. A run that does
   not move (largest position below 1 cm) earns exactly -0.5 S. Part 1
   prints the first term of the reference policies' runs, which do not
   stop.

Usage
    python reward_check.py       # prints; writes results/reward-check.txt
                                 # (about 5 minutes with two processes)
    python reward_check.py --jobs 1
"""
import argparse
import csv
import glob
import math
import os
import statistics
import sys
from fractions import Fraction as Fr
from multiprocessing import Pool

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cart_env    # noqa: E402
import protocol    # noqa: E402

cl = cart_env.cl
BOOK = os.path.join(HERE, "..", "results", "evaluation.csv")
T = 0.1
A_MAX = 1.875
CHI = cart_env.CHI                          # 0.5
V_MAX = math.sqrt(2 * 0.375 * 0.5)          # [m/s]
GAIN_MAX = V_MAX / (A_MAX * T)              # 3.27
DISCOUNTS = (1.0, 0.99, 0.975)
REFERENCE = [("rule", 0), ("braking curve", 0), ("off-policy learner", 1),
             ("off-policy learner", 2), ("off-policy learner", 3), ("at rest", 0)]


class AtRest(cl.ProposalMethod):
    """Proposes 0 m/s^2 in every period."""
    name = "at rest"

    def propose(self, obs):
        return Fr(0)


def episode(env, pm, world):
    """protocol.standin_episode in evaluation, keeping every period's reward
    and intervention before arrival."""
    obs, info = env.reset(options={"world": world})
    pm.reset(False)
    after = protocol.no_explore(pm)
    rewards, iotas, accel = [], [], []
    while True:
        q = after(obs, info) if info["arrived"] else pm.propose(info["obs"])
        obs, r, terminated, truncated, info = env.step(q)
        row = info["row"]
        if not row["after"]:
            pm.record(row)
            rewards.append(r)
            iotas.append(float(1 - row["alpha"]))
            accel.append(0.0 if row["a_next"] is None else float(row["a_next"]) / A_MAX)
        if terminated or truncated:
            break
    pm.end_episode()
    m = info["metrics"]
    return dict(world=world, arrival=m["arrival"], load=m["load"], pmax=m["pmax"],
                rewards=rewards, iotas=iotas, accel=accel)


def job(spec):
    name, seed = spec
    if name == "rule":
        pm = cl.Rule()
    elif name == "braking curve":
        pm = cl.BrakingCurve()
    elif name == "at rest":
        pm = AtRest()
    else:
        pm = cl.OffPolicyLearner(seed)
        tr = cart_env.CartGovernedEnv(proposal="reference", package="tube")
        for w in protocol.training_worlds(seed, protocol.N_WORLDS):
            protocol.standin_episode(tr, pm, w, train=True)
    env = cart_env.CartGovernedEnv(proposal="reference", package="tube")
    return name, seed, [episode(env, pm, w) for w in protocol.EVAL_WORLDS]


def discounted(xs, g):
    return sum(x * g ** k for k, x in enumerate(xs))


def returns(run, g):
    """(registered, time) return of a run from its first period."""
    reg = discounted(run["rewards"], g)
    tim = discounted([-1.0 - CHI * c for c in run["iotas"]], g)
    if run["arrival"] is None and g < 1:
        tim += -(g ** len(run["iotas"])) / (1 - g)
    return reg, tim


def book_rows():
    out = {}
    with open(BOOK, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["package"] == "tube":
                out[(r["policy"], int(r["seed"]), int(r["world"]))] = r
    return out


def check_against_book(name, seed, runs, book):
    """The runs must be those of ../results/evaluation.csv."""
    bad = 0
    for run in runs:
        r = book.get((name, seed, run["world"]))
        if r is None:
            return "not in the book"
        arr = "" if run["arrival"] is None else repr(run["arrival"])
        if arr != r["arrival_s"] or f"{run['load']:.4f}" != r["mean_intervention"]:
            bad += 1
    return "as ../results/evaluation.csv" if bad == 0 else f"{bad} runs DIFFER from the book"


def reference_lines(results):
    book = book_rows()
    groups, checks = {}, []
    for name, seed, runs in results:
        if name != "at rest":
            checks.append(f"{name}" + (f", seed {seed}" if seed else "") + ": "
                          + check_against_book(name, seed, runs, book))
        groups.setdefault(name, []).extend(runs)
    lines = []
    for name, runs in groups.items():
        arrived = [r for r in runs if r["arrival"] is not None]
        s_mean = statistics.mean(sum(r["iotas"]) for r in runs)
        txt = (f"  {name}: runs {len(runs)}; arrived {len(arrived)}; mean S {s_mean:.2f}")
        if arrived:
            gains = [sum(r["accel"]) for r in arrived]
            txt += (f"; first term over the arriving runs from {min(gains):.2f} to "
                    f"{max(gains):.2f} (bound {GAIN_MAX:.2f} for a run that does not stop on "
                    f"the way)")
        lines.append(txt)
        for label, i in (("registered reward", 0), ("time reward", 1)):
            vals = []
            for g in DISCOUNTS:
                xs = [returns(r, g)[i] for r in runs]
                tag = "undiscounted" if g == 1 else f"discount {g:g}"
                vals.append(f"{tag} {statistics.mean(xs):.2f} [{min(xs):.2f}, {max(xs):.2f}]")
            lines.append(f"      {label}: " + "; ".join(vals))
    return lines, checks


def published_lines():
    out = {}
    # the diffusion policy learns from demonstrations, without a reward: not listed
    for folder, label in (("sac", "SAC"), ("tdmpc2", "TD-MPC2"),
                          ("sac-gamma0.5", "SAC, discount 0.5"),
                          ("sac-time", "SAC, R2"), ("tdmpc2-time", "TD-MPC2, R2")):
        for path in sorted(glob.glob(os.path.join(HERE, "results", folder, "seed*",
                                                  "evaluation.csv"))):
            with open(path, newline="") as fh:
                for r in csv.DictReader(fh):
                    exact = r.get("mean_intervention_exact")
                    mean_c = float(Fr(exact)) if exact else float(r["mean_intervention"])
                    periods = int(r["periods_before_arrival"])
                    out.setdefault(f"{label}, {r['budget']}", []).append(
                        dict(arrived=bool(r["arrival_s"]), s=mean_c * periods,
                             still=float(r["max_position"]) < 0.01))
    lines = []
    for name, runs in out.items():
        arr = [r for r in runs if r["arrived"]]
        still = [r for r in runs if r["still"]]
        s_mean = statistics.mean(r["s"] for r in runs)
        txt = (f"  {name}: runs {len(runs)}; arrived {len(arr)}; did not move {len(still)}; "
               f"mean S {s_mean:.2f}")
        if arr:
            ub = statistics.mean(GAIN_MAX - CHI * r["s"] for r in arr)
            txt += (f"; arriving runs, if none stopped on the way: undiscounted registered "
                    f"return at most {ub:.2f} on average")
        if still:
            txt += (f"; runs that did not move: exactly "
                    f"{statistics.mean(-CHI * r['s'] for r in still):.2f}")
        lines.append(txt)
    return lines


def main():
    ap = argparse.ArgumentParser(description="What the rewards pay for (exploratory)")
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            results = pool.map(job, REFERENCE)
    else:
        results = [job(s) for s in REFERENCE]
    ref, checks = reference_lines(results)
    lines = ["What the rewards pay for (reward_check.py): exploratory, written after TD-MPC2's "
             "results; not a result under PROTOCOL.md or PROTOCOL-R2.md",
             f"  registered reward a_k/1.875 - {CHI} c_k; time reward -1 - {CHI} c_k; both 0 "
             f"after arrival. S: the sum of the interventions c_k before arrival. Returns from "
             f"the first period; mean [smallest, largest] over the runs",
             "[reference policies through cart_env, the 20 evaluation worlds]"]
    lines += ref
    lines.append("  the runs: " + "; ".join(checks))
    lines.append("[published methods, from results/<method>/seed*/evaluation.csv: bounds on the "
                 "undiscounted registered return of the runs that did not stop before arrival; "
                 "the files do not record stops, and each stop can add up to "
                 f"{float(cart_env.cl.cf.V_H) / (A_MAX * T):.2f}]")
    lines += published_lines()
    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(HERE, "results", "reward-check.txt"), "w") as fh:
        fh.write(text)


if __name__ == "__main__":
    main()
