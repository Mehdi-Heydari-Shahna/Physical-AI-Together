#!/usr/bin/env python3
"""
The calibration of the survey's offset for the complete case.

Physical AI, Together, Chapter 16 (Section 16.4). Protocol: complete/PROTOCOL.md.

In each of the 100 calibration worlds (seeds 44001 to 44100, used for
nothing else) the score is the surveyed start of the ramp less the true
one. The labelling map of complete.survey with offset q is correct for every
segment, every period and every candidate exactly when the score is at most
q (complete/modules.py, Survey), so Proposition 7.1 applies: the offset is
the m-th smallest of the n scores, m = ceil((n + 1)(1 - 0.05)) = 96, and the
probability that some label of the evaluation world is invalid is at most
5 % under exchangeability, whatever the proposal method and the package.
Calibration needs no simulation: the scores come from the worlds' draws.
It writes results/complete-calibration.txt.

Usage
    python3 complete_calibrate.py
"""
import math
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from complete import worlds                         # noqa: E402

RESULTS = os.path.join(HERE, "results")
DELTA = 0.05


def main():
    ws = worlds.family("calibration")
    scores = sorted(w.surveyed - w.ramp for w in ws)
    n = len(scores)
    m = math.ceil((n + 1) * (1 - DELTA))
    q = scores[m - 1]
    lines = ["[calibration]",
             f"  worlds = {n} ({ws[0].seed} to {ws[-1].seed}); unreliable surveys = "
             f"{sum(1 for w in ws if w.unreliable)}",
             f"  m = ceil((n + 1)(1 - {DELTA})) = {m}",
             f"  offset = {float(q):.2f} m (the {m}th smallest score)",
             f"  scores above the offset = {sum(1 for s in scores if s > q)}",
             f"  largest score = {float(scores[-1]):.2f} m; smallest = {float(scores[0]):.2f} m"]
    text = "\n".join(lines) + "\n"
    print(text)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "complete-calibration.txt"), "w") as fh:
        fh.write(text)


if __name__ == "__main__":
    main()
