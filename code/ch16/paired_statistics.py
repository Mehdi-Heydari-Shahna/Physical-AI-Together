# Paired statistics of the anticipation benchmark, computed after
# the initial analysis from the shipped file of runs. Run in
# code/ch16 with Python 3.8 or later; standard library only.
import collections
import csv
from math import comb

by = collections.defaultdict(dict)
for r in csv.DictReader(open("results/benchmark-runs.csv")):
    by[r["config"]][int(r["world"])] = r
valid = [w for w in sorted(by["C6"])
         if by["C6"][w]["valid"] == "yes"]


def exact(n, k):
    """Two-sided exact binomial test of k of n, probability 1/2."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(comb(n, i) for i in range(k + 1))
    return min(1.0, 2 * tail / 2**n)


def mcnemar(a, b):
    """Exact McNemar test of the operation on time (delay 0)."""
    on = {c: {w for w in valid if by[c][w]["delay"] == "0"}
          for c in (a, b)}
    only_a, only_b = on[a] - on[b], on[b] - on[a]
    p = exact(len(only_a) + len(only_b), len(only_a))
    print(f"{a} v {b}: both {len(on[a] & on[b])}, "
          f"neither {len(valid) - len(on[a] | on[b])}, "
          f"only {a} {len(only_a)}, "
          f"only {b} {len(only_b)} {sorted(only_b)}")
    print(f"  exact McNemar p = {p:.2g}")


def sign(a, b, column="in_work_zone"):
    """Exact sign test of the paired differences of a column."""
    d = [int(by[a][w][column]) - int(by[b][w][column])
         for w in valid]
    pos, neg = sum(x > 0 for x in d), sum(x < 0 for x in d)
    print(f"{column}, {a} - {b}: positive {pos}, "
          f"zero {len(d) - pos - neg}, negative {neg}")
    print(f"  exact sign test p = {exact(pos + neg, pos):.2g}; "
          f"mean {sum(d) / len(d):.2f} periods")


print("valid worlds:", len(valid))
for a, b in (("C6", "C11"), ("C14", "C11"), ("C6", "C12")):
    mcnemar(a, b)
sign("C6", "C11")
