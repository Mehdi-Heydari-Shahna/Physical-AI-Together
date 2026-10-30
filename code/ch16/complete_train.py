#!/usr/bin/env python3
"""
The training of the off-policy learner of the complete case.

Physical AI, Together, Chapter 16 (Section 16.4). Protocol: complete/PROTOCOL.md.

For each control package, one learner (complete.learner, the off-policy
learner of Chapter 6 with the package's declared constants) is trained in
the 300 training worlds (seeds 47001 to 47300, one episode each, in order)
under the frozen bounds (complete/FROZEN.json) and the calibrated offset
(results/complete-calibration.txt), with the survey predictor and without
the operation: an episode is 200 periods (20 s) of the governed MuJoCo cart,
and the task of training is to reach the work zone. After each episode the
learner learns from the periods before arrival. Its table is then frozen in
results/complete-learner-<package>.json, with the log of its episodes.

Usage
    python3 complete_train.py               # both packages; writes results/
    python3 complete_train.py --episodes 5  # a smoke test on development worlds,
                                            # provisional values; writes nothing

Requires MuJoCo 3.16.0 and NumPy. About twenty minutes with two processes.
"""
import argparse
import csv
import json
import os
import sys
from multiprocessing import Pool

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import complete                                       # noqa: E402,F401
import allovi                                         # noqa: E402
from complete import config, model as md, worlds      # noqa: E402
from complete.modules import Learner                  # noqa: E402

RESULTS = os.path.join(HERE, "results")
EPISODE = 200


def offset_frozen():
    with open(os.path.join(RESULTS, "complete-calibration.txt")) as fh:
        for line in fh:
            if line.strip().startswith("offset ="):
                return float(line.split("=")[1].split()[0])
    raise RuntimeError("no calibrated offset")


def train(args):
    pkg, n, frozen = args
    bounds = md.bounds(frozen)
    offset = offset_frozen() if frozen else 0.40
    control = config.PACKAGES[pkg](bounds=bounds)
    learner = Learner(control.declared(), seed=1)
    learner.train = True
    log = []
    phase = "training" if frozen else "development"
    for i, seed in enumerate(list(worlds.PHASES[phase])[:n]):
        world = worlds.World(seed)
        cfg, sim = config.configure(world, "A-learner", pkg, bounds, offset,
                                    proposal=learner, task_on=False)
        rec = allovi.run(cfg, EPISODE)
        states = [x.state for x in rec.periods] + [rec.final_state]
        arrival = next((k for k, s in enumerate(states) if s[0] >= md.WORK_ZONE), None)
        before = [x for x in rec.periods if arrival is None or x.k < arrival]
        modified = sum(1 for x in before if x.decision is not None and x.decision.ambition < 1)
        load = sum(1 - float(x.decision.ambition) for x in before if x.decision is not None)
        learner.learn(rec, arrival)
        log.append(dict(episode=i, world=seed, arrival=arrival,
                        mean_intervention=round(load / max(1, len(before)), 6),
                        modified=modified, periods=len(before), alarms=len(rec.alarms)))
    return pkg, learner.table(), log


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--episodes", type=int, default=None,
                    help="a smoke test with that many episodes, provisional bounds; writes nothing")
    args = ap.parse_args()
    n = args.episodes or len(worlds.PHASES["training"])
    frozen = args.episodes is None
    with Pool(2) as pool:
        out = pool.map(train, [(pkg, n, frozen) for pkg in config.PACKAGES])
    for pkg, table, log in out:
        arrived = [r for r in log if r["arrival"] is not None]
        last = log[-50:]
        print(f"{pkg}: episodes {len(log)}, arrived {len(arrived)}, states {len(table)}, "
              f"last 50: mean arrival "
              f"{sum(r['arrival'] for r in last if r['arrival'] is not None) / max(1, sum(1 for r in last if r['arrival'] is not None)) / 10:.2f} s, "
              f"mean intervention {sum(r['mean_intervention'] for r in last) / len(last):.3f}, "
              f"alarms {sum(r['alarms'] for r in log)}")
        if not frozen:
            continue
        os.makedirs(RESULTS, exist_ok=True)
        with open(os.path.join(RESULTS, f"complete-learner-{pkg}.json"), "w") as fh:
            json.dump(dict(package=pkg, episodes=len(log), table=table), fh, indent=1)
            fh.write("\n")
        with open(os.path.join(RESULTS, f"complete-training-{pkg}.csv"), "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(log[0]))
            wr.writeheader()
            wr.writerows(log)


if __name__ == "__main__":
    main()
