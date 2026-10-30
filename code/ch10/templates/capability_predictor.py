"""Template of a capability predictor (Section 10.3; Section 4.2.1).

Obligations, which no check discharges:
  * every label at least as informative as the global capability
    (Assumption 4.3), and the global capability beyond the horizon;
  * the worlds that each label admits, and how often labels are invalid,
    with the evidence for it (Chapter 7).

Checks that allovi.checks runs: registration; SPACE equal to the NAME of the
capability space of the test; labels() returning None or a callable; labels
at least as informative as the global capability; the global capability
just beyond the horizon; a labelling map that gives the same label for the
same arguments throughout its period; runs that repeat exactly; and, in
simulation, executed labels that the oracle finds invalid.

For a module of the cart, HELPERS.md in this folder documents what the
templates leave out: the fields of the cart's segments (lo and hi, the
region to which a label refers), the unit of its labels, the parameter
ce.RAMP that the worlds of Chapters 7, 16 and 17 set, and how to run and
check the module in the cart's configurations.
"""


class MyCapabilityPredictor:
    ROLE, CONTRACT = "capability", "1.0"
    NAME = "template.capability-predictor"
    # the NAME of the capability space of the test
    SPACE = "cart.braking"
    # periods; beyond them, every label is the global capability
    horizon = 0

    def labels(self, observation):
        """The labelling map of period observation.k: a function
        labelling(j, segment) that returns the label of the segment
        for period j, used only during period k; or None if no map
        is ready."""
        raise NotImplementedError

    def observe(self, k, before, after):
        """The measured states at the start and at the end of period
        k, reported after the decision of the period."""
        raise NotImplementedError
