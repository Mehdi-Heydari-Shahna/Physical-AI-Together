"""Exact, distribution-free intervals for the reports of Chapter 16.

Proportions: the two-sided interval of Clopper and Pearson, each end found
by bisection in exact integer arithmetic on the multiples of 2^-24 and
rounded outwards, so that it is never narrower than the exact interval.
Medians: the interval between two order statistics whose coverage, from the
binomial distribution with p = 1/2, is at least the level, for any
continuous distribution (for ties, at least as large). Both treat the
worlds of a family as independent draws from the family's distribution,
which is how the worlds are made.
"""
import math
from fractions import Fraction as Fr

LEVEL = Fr(95, 100)


def _cdf_scaled(n, x, a, b):
    """b^n P(Bin(n, a/b) <= x), exactly, for integers 0 <= a <= b."""
    c = b - a
    if x < 0:
        return 0
    total = 0
    for i in range(x + 1):
        total += math.comb(n, i) * a ** i * c ** (n - i)
    return total


def clopper_pearson(x, n, level=LEVEL, bits=24):
    """The two-sided exact interval for a proportion of x successes in n
    trials, at the given level, as fractions rounded outwards."""
    alpha2 = (1 - level) / 2
    b = 1 << bits
    bn = b ** n

    def p_at_most(a, k):                    # P(Bin(n, a/b) <= k)
        return Fr(_cdf_scaled(n, k, a, b), bn)

    if x == 0:
        lower = Fr(0)
    else:                                    # P(Bin(n, p) >= x) = alpha2
        lo, hi = 0, b
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if 1 - p_at_most(mid, x - 1) <= alpha2:
                lo = mid
            else:
                hi = mid
        lower = Fr(lo, b)
    if x == n:
        upper = Fr(1)
    else:                                    # P(Bin(n, p) <= x) = alpha2
        lo, hi = 0, b
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if p_at_most(mid, x) >= alpha2:
                lo = mid
            else:
                hi = mid
        upper = Fr(hi, b)
    return lower, upper


def median_interval(values, level=LEVEL):
    """The sample median and the interval [x_(j), x_(n+1-j)] with the
    largest j whose coverage 1 - 2 P(Bin(n, 1/2) <= j - 1) is at least the
    level. Returns (median, low, high, coverage)."""
    xs = sorted(values)
    n = len(xs)
    if n == 0:
        return None, None, None, None
    med = xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2
    best = None
    for j in range(1, n // 2 + 1):
        tail = Fr(sum(math.comb(n, i) for i in range(j)), 2 ** n)
        if 1 - 2 * tail >= level:
            best = (j, 1 - 2 * tail)
        else:
            break
    if best is None:
        return med, xs[0], xs[-1], None
    j, cover = best
    return med, xs[j - 1], xs[n - j], cover


def percent(x, n, nd=0):
    """x/n as a percentage, rounded half up, as a string."""
    if n == 0:
        return "-"
    q = Fr(100 * x, n) * 10 ** nd
    r = math.floor(q + Fr(1, 2))
    s = f"{r / 10 ** nd:.{nd}f}" if nd else str(r)
    return s


def interval_percent(lo, hi, nd=1):
    """An interval of proportions in percent, the lower end rounded down and
    the upper end rounded up."""
    lo_p = math.floor(lo * 100 * 10 ** nd) / 10 ** nd
    hi_p = math.ceil(hi * 100 * 10 ** nd) / 10 ** nd
    return f"{lo_p:.{nd}f} to {hi_p:.{nd}f} %"
