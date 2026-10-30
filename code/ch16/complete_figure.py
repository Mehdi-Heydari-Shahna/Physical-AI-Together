#!/usr/bin/env python3
"""
The coordinates of Figure 16.4: when the governor with the naive rule and
the filter with the preview planner began the operation on time, in the
complete case.

Physical AI, Together, Chapter 16 (Section 16.4). Reads
results/complete-runs.csv (written by complete_evaluate.py) and prints, for
A-rule and F-preview with each package, the valid worlds as pgfplots
coordinates (lead time, slip of the true onset), split into those on time
and those late. The figure's source (figures/fig-16-5-complete-preparation.tex
of the book) holds these coordinates, so that it builds without the code
tree. Worlds with the same lead time and slip coincide.

Usage
    python3 complete_figure.py

Python 3.8 or later; no other package.
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OFFSET = 0.20          # results/complete-calibration.txt


def main():
    with open(os.path.join(RESULTS, "complete-runs.csv")) as fh:
        rows = list(csv.DictReader(fh))
    for pkg in ("tube", "mpc"):
        for cfg in ("A-rule", "F-preview"):
            rs = [r for r in rows if r["config"] == cfg and r["package"] == pkg
                  and float(r["score"]) <= OFFSET + 1e-12]
            for label, keep in (("on time", True), ("late", False)):
                pts = sorted({(int(r["lead"]), int(r["slip"])) for r in rs
                              if (r["on_time"] == "True") == keep})
                n = sum(1 for r in rs if (r["on_time"] == "True") == keep)
                print(f"% {cfg}-{pkg}, {label}: {n} worlds, {len(pts)} points")
                print("coordinates {" + " ".join(f"({a},{b})" for a, b in pts) + "};")


if __name__ == "__main__":
    main()
