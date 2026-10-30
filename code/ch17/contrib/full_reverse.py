"""A reader's control package: the generator of the cart with the full
reverse (Sections 9.3 and 17.1).

Role: control. Name: reader.full-reverse. Contract: 1.0.

What it changes. The package bench.constant-command of Chapter 16 (the cart's
package of Chapter 10 with the proposal 'retreat'), and one more candidate
whenever a requirement above the floor is pending: the full reverse of
Section 9.3, which brakes a forward motion to rest, drives towards the start
at the strongest certified command until the front is at or before the
requirement's place, and brakes to rest beyond it, with ambition 0. The
test, the rest segment, the rest feedback and every other candidate are
those of bench.constant-command.

Obligations (Table 4.1, the control method's row), and how each is
established:
  * sound test: inherited unchanged from cart.constant-command, whose test is
    exact (Chapter 3); this package adds candidates, not verdicts;
  * a certified rest segment, a rest feedback and its reach bound: inherited
    unchanged;
  * the candidates start at the initial set and are well formed: by
    construction (the full reverse starts at the initial set and chains its
    segments), and checked by the check suite of Section 10.3;
  * candidates for a pending requirement (Assumption 4.13 and the generator's
    obligations of Table 4.1): the full reverse is one more such candidate;
    the guarantees of Chapter 9 hold for any candidates that the test
    certifies, so this package claims no new guarantee.

What it claims, as a performance hypothesis: the smallest lead time with
which the cart, holding at 24.5 m, meets the operation falls from 21 periods
to 19, the bound of Theorem 9.12(c) (Section 9.3.2); contribute.py tests the
claim on the benchmark's family of worlds.
"""
from fractions import Fraction as Fr

from allovi import Candidate, register
from bench.modules import Retreating
from cart.control import _Labels
from cart.model import ce


def full_reverse(labels, k, j, p, v, target):
    """The full reverse from (p, v) at the start of period j (Section 9.3):
    maximal certified braking to rest if the cart moves towards the wall,
    then the strongest certified backward command until the front is at or
    before the target, then maximal braking to rest."""
    pairs = []
    if v > 0:
        tail, p = ce.brake_to_rest(labels, k, j, p, v)
        pairs += tail
        j += len(tail)
        v = Fr(0)
    while p > target:
        seg, beta = ce.backward_step(labels, k, j, p, v)
        pairs.append((seg, beta))
        p, v, j = seg.p1, seg.v1, j + 1
    tail, p_stop = ce.brake_to_rest(labels, k, j, p, v)
    return pairs + tail, p_stop


@register
class FullReverse(Retreating):
    """bench.constant-command and the full reverse (module docstring)."""
    NAME = "reader.full-reverse"

    def generate(self, step, plan, initial):
        cands = super().generate(step, plan, initial)
        req = step.requirement.level
        if req > ce.EPS_MIN:
            p, v, _ = initial
            ell = Fr(0) if step.operation is None else step.operation
            labels = _Labels(step.labelling)
            pairs, p_rest = full_reverse(labels, step.k, step.k + 1, p, v, ce.WALL - req)
            if pairs:
                cands.append(Candidate(ce.Plan(step.k + 1, pairs, p_rest, ell), Fr(0)))
        return cands
