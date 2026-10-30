"""Template of a demand predictor (Section 10.3; Definition 4.12).

Obligations, which no check discharges: none for safety; the reserve that
each operation consumes, linked to the operations of the task.

Checks that allovi.checks runs: registration; requirements that are None or
an allovi.Requirement whose level is a level of the level space and whose
onset is a whole number of at least 0; runs that repeat exactly.

For a module of the cart, HELPERS.md in this folder documents what the
templates leave out: the cart's task cart.operation, whose schedule a
demand predictor reads and which it receives as the argument 'task', the
constants of the levels, and how to run and check the predictor, which
cart_checks.py --module cannot create without its task.
"""


class MyDemandPredictor:
    ROLE, NAME, CONTRACT = "demand", "template.demand-predictor", "1.0"

    def requirement(self, observation):
        """allovi.Requirement(level, onset) for the plan committed for the
        next period, or None if no demand is pending."""
        raise NotImplementedError
