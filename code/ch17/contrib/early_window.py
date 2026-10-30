"""A reader's demand predictor: the operation's reserve from the earliest
onset that the task plan allows (Corollary 9.13, Section 17.1).

Role: demand. Name: reader.early-window. Contract: 1.0.

What it changes. The demand predictor cart.task-plan of Chapter 10 requires
the operation's reserve, from the announcement on, for the onset that the
task plan forecasts. This one requires it for an onset 'early' periods
before the forecast, the earliest that the task plan states the operation
may come (in the benchmark's family, 10 periods). Once that onset has passed,
it requires the reserve from the first pair on, as cart.task-plan does after
its forecast onset; before the announcement, and once the operation has
begun, it requires the floor.

Obligations (Table 4.1, the demand predictor's row): none for the
guarantees. A requirement never enters the test (Section 4.3.1), so a wrong
requirement changes when the robot prepares, never whether its plans are
certified; by Corollary 9.13, a reserve held at the early onset is kept
until the operation is due.

What it claims, as a performance hypothesis: an operation that comes up to
'early' periods before its forecast onset, announced with a lead time that
allows the preparation, begins on time; the price is the time in the work
zone given up while the reserve is held early, 'early' periods in every
world in which the robot would otherwise have stayed in the work zone.
"""
from allovi import Requirement, register
from cart.model import OPERATION_LEVEL, ce


@register
class EarlyWindow:
    ROLE, NAME, CONTRACT = "demand", "reader.early-window", "1.0"

    def __init__(self, task, early=10):
        self.task, self.early = task, early

    def requirement(self, observation):
        task, k = self.task, observation.k
        if task.pending():
            if task.always:
                return Requirement(OPERATION_LEVEL, 0)
            if task.announce is not None and k >= task.announce:
                onset = task.predicted - self.early
                return Requirement(OPERATION_LEVEL, max(onset - k - 1, 0))
        return Requirement(ce.EPS_MIN, 0)
