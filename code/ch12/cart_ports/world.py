"""The simulator adapters of the cart in Newton and Genesis for the allovi
core (Section 12.1): cart.newton and cart.genesis. They declare the state and
the actuation of the cart's robot model, as cart.mujoco of Section 11.1
does, and realize the laws of the control package cart.constant-command in
the same way; only the simulator beneath them differs. Each imports its
simulator when it is created, so that either can be used in an environment
that has only its own simulator."""
import math
from fractions import Fraction as Fr

import allovi
from cart.model import MODEL

from .base import G
from .design import Design


def exact(x):
    """The values of the simulator, as exact rationals: the control package
    of the cart computes in exact arithmetic, and receives them unchanged."""
    return tuple(Fr(float(a)) for a in x)


class PortedWorld:
    """The world of a ported cart: the methods of cart.mujoco, unchanged."""
    ROLE, CONTRACT = "world", "1.0"
    STATE, ACTUATION = MODEL.state, MODEL.actuation

    def _attach(self, sim):
        self.sim, self.last = sim, None

    def reset(self):
        """The cart at rest at the start, its tool folded."""
        self.sim.place(0.0, 0.0, 0.0)
        self.last = exact(self.sim.state())
        return self.last

    def step(self, state, law):
        """One period from 'state' under 'law'. The simulator continues from
        where it is if 'state' is the state that it returned last, and is
        placed at 'state' otherwise, as for the checks of Section 10.4."""
        if state != self.last:
            self.sim.place(*(float(a) for a in state))
        ell = state[2] if law.ell is None else law.ell
        self.last = exact(self.sim.period(float(law.u), float(ell), law.kind))
        return self.last

    # --- the oracle, for the checks of Section 10.3 (simulation only) ---------
    def capacity(self, lo, hi, speed):
        """The braking that the simulated ground allows, in m/s^2, as for
        cart.mujoco: the same ground, built from the same design."""
        d, pr = self.sim.design, self.sim.pr
        mu, lead = d.friction, self.sim.lead
        a_lo = pr.ground(lo - lead - 2 * d.wheel_offset)
        a_hi = pr.ground(hi - lead)
        worst = math.inf
        for s in sorted({a_lo, a_hi, min(max(pr.s0, a_lo), a_hi), min(max(pr.s1, a_lo), a_hi)}):
            ang = pr.point(s)[2]
            b = G * (mu * math.cos(ang) - math.sin(ang))
            if pr.s0 <= s <= pr.s1:
                b -= mu * speed * speed / pr.R
            worst = min(worst, b)
        return worst

    def admits(self, label, segment, k):
        """True if the world admits the label on the region of the segment."""
        speed = float(max(abs(segment.v0), abs(segment.v1)))
        return self.capacity(float(segment.lo), float(segment.hi), speed) >= float(label)

    def clearance(self, state):
        """The reserve of a state in the norm of the state: its distance to
        the states that violate p + l <= 25 m."""
        return Fr(self.sim.design.wall) - (state[0] + state[2])


@allovi.register
class CartNewton(PortedWorld):
    """The world of Section 12.1 in Newton; solver is one of
    newton_cart.SOLVERS."""
    NAME = "cart.newton"

    def __init__(self, design=None, solver="mujoco-warp", device="cpu", timestep=None,
                 **options):
        from .newton_cart import NewtonCart
        self._attach(NewtonCart(design or Design(), solver=solver, device=device,
                                timestep=timestep, **options))


@allovi.register
class CartGenesis(PortedWorld):
    """The world of Section 12.1 in Genesis."""
    NAME = "cart.genesis"

    def __init__(self, design=None, timestep=None, **options):
        from .genesis_cart import GenesisCart
        self._attach(GenesisCart(design or Design(), timestep=timestep, **options))
