#!/usr/bin/env python3
"""
Compares a rerun of the complete case's evaluation with the shipped records.

Physical AI, Together, Chapter 16 (Section 16.4).

The complete case has no --check of its own: its evaluation reruns the
1000 runs of Section 16.4 and writes its records into results/. This script
compares the records of such a rerun with those shipped:

    python3 complete_compare.py RERUN [SHIPPED]

RERUN is the folder results/ of a copy of code/ch16 in which
complete_evaluate.py was run again; SHIPPED is the folder of the shipped
records, by default results/ next to this file. Run the evaluation from tag
v1.0, whose files the frozen record (complete/FROZEN.json) names, in a copy,
so that the shipped files are not overwritten, for example

    git clone --branch v1.0 https://github.com/Mehdi-Heydari-Shahna/Physical-AI-Together.git rerun
    cd rerun/code/ch16
    python3 complete_evaluate.py
    python3 <this repository>/code/ch16/complete_compare.py results

The script compares
  * complete-summary.txt line by line, except the lines of the decisions'
    wall-clock times ("decisions (this machine, ...)") and the section
    [record], which names the time, the machine and the interpreter;
  * complete-runs.csv and complete-failures.csv row by row and cell by cell,
    except the columns of the decisions' wall-clock times (names ending in
    _ms).
It prints each difference and exits with an error if there is one.

Python 3.8 or later; no other packages.
"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = "complete-summary.txt"
TABLES = ("complete-runs.csv", "complete-failures.csv")


def summary_lines(path):
    """The lines of the summary that a rerun must repeat."""
    keep, in_record = [], False
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if line.startswith("["):
                in_record = line.strip() == "[record]"
            if in_record or "decisions (this machine" in line:
                continue
            keep.append(line)
    return keep


def table(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    columns = list(rows[0].keys()) if rows else []
    return columns, rows


def compare_summary(rerun, shipped):
    a, b = summary_lines(rerun), summary_lines(shipped)
    diffs = []
    for i in range(max(len(a), len(b))):
        x = a[i] if i < len(a) else "(missing)"
        y = b[i] if i < len(b) else "(missing)"
        if x != y:
            diffs.append(f"{SUMMARY}, compared line {i + 1}:\n  rerun:   {x}\n  shipped: {y}")
    return len(b), diffs


def compare_table(name, rerun, shipped):
    ca, ra = table(rerun)
    cb, rb = table(shipped)
    diffs = []
    if ca != cb:
        diffs.append(f"{name}: the columns differ")
        return 0, diffs
    cols = [c for c in cb if not c.endswith("_ms")]
    if len(ra) != len(rb):
        diffs.append(f"{name}: {len(ra)} rows in the rerun, {len(rb)} shipped")
    cells = 0
    for i, (x, y) in enumerate(zip(ra, rb)):
        for c in cols:
            cells += 1
            if x[c] != y[c]:
                diffs.append(f"{name}, row {i + 2}, column {c}: rerun {x[c]!r}, shipped {y[c]!r}")
    return cells, diffs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("rerun", help="the folder results/ of the rerun")
    ap.add_argument("shipped", nargs="?", default=os.path.join(HERE, "results"),
                    help="the folder of the shipped records (default: results/)")
    args = ap.parse_args()
    n, diffs = compare_summary(os.path.join(args.rerun, SUMMARY),
                               os.path.join(args.shipped, SUMMARY))
    report = [f"{SUMMARY}: {n} lines compared"]
    for name in TABLES:
        cells, d = compare_table(name, os.path.join(args.rerun, name),
                                 os.path.join(args.shipped, name))
        report.append(f"{name}: {cells} cells compared")
        diffs += d
    print("\n".join(report))
    if diffs:
        print(f"[compare] {len(diffs)} differences:")
        print("\n".join(diffs))
        sys.exit(1)
    print("[compare] the rerun repeats every compared value of the shipped records")


if __name__ == "__main__":
    main()
