"""Template of a task (Section 10.3; Section 4.3.1).

Obligations, which no check discharges: none for safety; when each operation
is due, and what the generator must begin.

Checks that allovi.checks runs: registration; runs that repeat exactly.

For a module of the cart, HELPERS.md in this folder documents the cart's
task cart.operation, how a configuration passes the same task to the demand
predictor and the evaluator, and how to run and check a task, which
cart_checks.py --module does not take.
"""


class MyTask:
    ROLE, NAME, CONTRACT = "task", "template.task", "1.0"

    def begin_period(self, k, level_in_force):
        """The level of the plan in force at t_k, which decides whether an
        operation due in period k can begin (Assumption 4.13)."""
        raise NotImplementedError

    def operation(self, k):
        """What the generator is asked to begin during period k, or None."""
        raise NotImplementedError
