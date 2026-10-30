"""A capability predictor for the cart, written from the public interface
alone (Section 10.3): the survey of Section 7.2.3 used with a margin. A
segment is labelled 2 m/s^2 only if its region ends more than 'margin'
before the surveyed start of the ramp; every other segment, and every period
beyond the horizon, gets the global capability. Its segments are those of
the cart, whose region ends at segment.hi."""
from fractions import Fraction as Fr

from allovi import register

FIRM = Fr(2)          # m/s^2, the braking of firm ground
GLOBAL = Fr(1, 2)     # m/s^2, the global capability


@register
class SurveyMargin:
    ROLE, NAME, CONTRACT = "capability", "example.survey-margin", "1.0"
    SPACE, horizon = "cart.braking", 400

    def __init__(self, surveyed_start=Fr(101, 5), margin=Fr(1, 4)):
        self.threshold = surveyed_start - margin

    def labels(self, observation):
        k = observation.k

        def labelling(j, segment):
            if j > k + 1 + self.horizon or segment.hi >= self.threshold:
                return GLOBAL
            return FIRM
        return labelling

    def observe(self, k, before, after):
        pass
