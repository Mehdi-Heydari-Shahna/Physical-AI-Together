"""Template of a control package (Section 10.3; Section 4.2, Table 4.1).

Obligations, which no check discharges:
  * a sound test (S3), including the inclusions (P1), (P3) and (P4) and the
    invariance of the rest segment;
  * a rest segment certified under the global capability, a rest feedback
    with a bound on the states that it can reach in one period under it, and
    an initial plan certified on a set that contains every state within the
    measurement-error bound of the first measurement (S4);
  * admissible initial sets, and initial sets, distances and laws computed
    before each deadline (S6); for demands, recovery and operation
    candidates (Theorem 9.12, Assumption 4.13);
  * with the application, a global capability valid in every period (S2).

Checks that allovi.checks runs: registration; candidates, profiles and
answers of reach() of the right kind; profiles that do not decrease; a
fallback that keeps its shifted profile; inputs within the declared bounds;
runs that repeat exactly; in simulation, with head(), the true reserve and
alarms while every executed label was valid; and the package's own checks().

For a module of the cart, HELPERS.md in this folder documents what the
templates leave out: the cart's plans and segments (ce.Plan, ce.Segment)
and the conditions on which its test relies, the functions of its
generator, cart.control._Labels, how to extend cart.constant-command, and
how to run and check the package in the cart's configurations.
"""


class MyControlPackage:
    ROLE, NAME, CONTRACT = "control", "template.control-package", "1.0"
    # capability space: NAME, global_capability, labelling(c); optional,
    # for the checks of Section 10.3: informative(c, d), True if label c is
    # at least as informative as label d
    capabilities = None
    # level space: floor, admits, key, leq, contains; see allovi.ScalarLevels
    levels = None
    STATE = None          # optional: allovi.Schema that the segments assume
    ACTUATION = None      # optional: allovi.Actuation that the segments assume

    def initial_plan(self, measured, bound):
        """(plan, initial set): a plan certified at the floor on a set that
        contains every state within 'bound' of the measurement."""
        raise NotImplementedError

    def initial_set(self, plan):
        """The admissible initial set for the next plan (Definition 4.15)."""
        raise NotImplementedError

    def generate(self, step, plan, initial_set):
        """The candidates of period step.k, a list of allovi.Candidate, in the
        order of evaluation; recovery candidates only if step.proposal is
        None. Plans must provide shift()."""
        raise NotImplementedError

    def profile(self, plan, initial_set):
        """The verdicts of the sound test as an allovi.Profile."""
        raise NotImplementedError

    def law(self, plan):
        """The feedback law of the head of the plan: callable as
        law(tau, state), returning the input in the declared actuation."""
        raise NotImplementedError

    def distance(self, measured, initial_set):
        """The distance from a measured state to an initial set, in the norm
        of the state (a number)."""
        raise NotImplementedError

    def rest_law(self, measured):
        """The rest feedback, a law as law() returns."""
        raise NotImplementedError

    def reach(self, measured, bound):
        """(set, reason): the states that the rest feedback can reach in one
        period under the global capability, or (None, reason)."""
        raise NotImplementedError

    def restart(self, step, reach):
        """Restart candidates on the set 'reach', labelled by the global
        capability; recovery candidates only if step.proposal is None."""
        raise NotImplementedError

    # Optional, for the checks of Section 10.3:
    # def head(self, plan): the pair (segment, label) that the plan executes
    #     in its first period
    # def checks(self): the package's own checks, a list of
    #     allovi.checks.Result
