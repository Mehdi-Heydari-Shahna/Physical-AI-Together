"""Template of an evaluator (Section 10.3; Chapter 16).

Obligations: none; it never influences a decision.

Checks that allovi.checks runs: registration.

For a module of the cart, HELPERS.md in this folder documents the record
of a run (allovi.RunRecord), the cart's evaluator cart.metrics and the
task that it reads, and how to run a configuration.
"""


class MyEvaluator:
    ROLE, NAME, CONTRACT = "evaluator", "template.evaluator", "1.0"

    def evaluate(self, record):
        """A dictionary of metrics of an allovi.RunRecord."""
        raise NotImplementedError
