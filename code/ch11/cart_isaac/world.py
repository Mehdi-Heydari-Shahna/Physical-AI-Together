"""The simulator adapter of the cart in Isaac Sim for the allovi core
(Section 11.2): cart.isaac. It declares the state and the actuation of the
cart's robot model, as the exact track of Chapter 10 and cart.mujoco of
Section 11.1 do, and realizes the laws of the control package
cart.constant-command from their fields: the commanded acceleration and
reach of a segment, held over the period, or the rest feedback, which brakes
a cart moving towards the wall at its command until the cart is at rest and
then holds it, and holds a cart that is at rest or moving back at once."""
import math
from fractions import Fraction as Fr

import allovi
from cart.model import MODEL

from .design import Design, IsaacSettings
from .sim import G, IsaacCart


def exact(x):
    """The values of the simulator, as exact rationals: the control package
    of the cart computes in exact arithmetic, and receives them unchanged."""
    return tuple(Fr(float(a)) for a in x)


@allovi.register
class CartIsaac:
    """The world of Section 11.2: the track of Chapter 3 built in Isaac Sim."""
    ROLE, NAME, CONTRACT = "world", "cart.isaac", "1.0"
    STATE, ACTUATION = MODEL.state, MODEL.actuation

    def __init__(self, design=None, settings=None):
        self.sim = IsaacCart(design or Design(), settings or IsaacSettings())
        self.last = None

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
    # The same bound as cart.mujoco: both simulators build the same ground.
    def capacity(self, lo, hi, speed):
        """The braking that the simulated ground allows, in m/s^2, when the
        bumper is anywhere in [lo, hi] and the cart is no faster than
        'speed': for each wheel on the ground below it, g (mu cos a - sin a)
        at the angle a of the ground, less mu speed^2 / R on the crest, where
        the wheels are unloaded by the curvature; the least over the region."""
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
