#!/usr/bin/env python3
"""
The course of training of a published method, in blocks of episodes.

Physical AI, Together, Chapter 6, Section 6.2.4 (the supplement's figure
of the training of the published SAC, fig:sac-training).

Reads results/<method>/seed<s>/training.csv, written by train_sac.py or
train_tdmpc2.py, and prints for each seed and each block of consecutive
training episodes (20 by default), at the block's middle:
    the exact mean of the episodes' mean interventions, rounded half up to
        three decimals (the column mean_intervention_exact of training.csv);
    the number of episodes that reached the work zone within 20 s;
    the exact mean of their arrival times, in seconds, rounded half up to
        two decimals ("nan" where none arrived).
It also prints the block data as pgfplots coordinates, which the figure
copies, so that the book builds without this tree. Nothing here is a
result of the protocol (PROTOCOL.md): it describes the training episodes
that report.py summarizes.

Usage
    python training_blocks.py                    # SAC, blocks of 20
    python training_blocks.py --method tdmpc2 --block 50
"""
import argparse
import csv
import os
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction as Fr

HERE = os.path.dirname(os.path.abspath(__file__))


def half_up(x, digits):
    """A Fraction rounded half up to the digits given, as text."""
    d = Decimal(x.numerator) / Decimal(x.denominator)
    return str(d.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


def blocks(path, size):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for k in range(0, len(rows), size):
        blk = rows[k:k + size]
        mid = k + len(blk) // 2
        load = sum(Fr(r["mean_intervention_exact"]) for r in blk) / len(blk)
        periods = [int(r["periods_before_arrival"]) for r in blk if r["arrival_s"]]
        mean = half_up(Fr(sum(periods), 10 * len(periods)), 2) if periods else "nan"
        out.append((mid, half_up(load, 3), len(periods), len(blk), mean))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--method", default="sac", help="the results folder: sac or tdmpc2")
    ap.add_argument("--block", type=int, default=20, help="episodes per block")
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    base = os.path.join(args.results, args.method)
    seeds = sorted(int(d[4:]) for d in os.listdir(base) if d.startswith("seed") and d[4:].isdigit()
                   and os.path.exists(os.path.join(base, d, "training.csv")))
    for s in seeds:
        data = blocks(os.path.join(base, f"seed{s}", "training.csv"), args.block)
        print(f"seed {s}: middle, mean intervention, arrived, episodes, mean arrival [s]")
        for mid, load, n, m, mean in data:
            print(f"  {mid}, {load}, {n}, {m}, {mean}")
        print(f"  coordinates, (a): " + " ".join(f"({mid},{load})" for mid, load, *_ in data))
        print(f"  coordinates, (b): " + " ".join(f"({mid},{n})" for mid, _, n, *_ in data))
        print(f"  coordinates, (c): " + " ".join(f"({mid},{mean})" for mid, *_, mean in data))


if __name__ == "__main__":
    main()
