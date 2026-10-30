"""Template of a proposal method (Section 10.3; Definition 4.11).

Obligations, which no check discharges:
  * causality: the proposal of period k depends on nothing observed after
    t_k; a proposal method carries no obligation for safety;
  * a meaning of its proposals that the generator of the control package
    understands, as the application states it.

Checks that allovi.checks runs: registration; proposals that the generator
can use (a proposal it cannot use makes the generator fail); runs that
repeat exactly, for a deterministic method.

For a module of the cart, HELPERS.md in this folder documents what the
templates leave out: the cart's measured state (p, v, l), the proposals
that its generator understands, and how to run and check the method in
the cart's configurations.
"""


class MyProposalMethod:
    ROLE, NAME, CONTRACT = "proposal", "template.proposal-method", "1.0"

    def propose(self, observation):
        """The proposal q_k from observation o_k (allovi.Observation: k, t,
        measured, level, ambition, alarm), or None if none is ready."""
        raise NotImplementedError
