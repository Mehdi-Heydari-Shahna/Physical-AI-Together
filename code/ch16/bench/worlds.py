"""The family of worlds of the anticipation benchmark (Section 16.1).

Each world draws, from its integer seed, in this order: the start of the
downhill ramp, uniform on [17, 21] m on a 1 cm grid, as in Chapters 6 and 7;
whether the survey of that start is unreliable (probability 0.2); the
survey's error, uniform on [-0.60, 0.60] m if it is unreliable and on
[-0.20, 0.20] m otherwise, on a 1 cm grid, the condition "as calibrated" of
Section 7.3; the forecast onset of the tool operation, a period uniform on
200 to 240; the lead time, uniform on 10 to 60 periods; whether the onset
slips (probability 0.3), by how many periods (1 to 10) and in which
direction (later or earlier, each with probability 1/2). Every draw is made
in every world, so that a world's other draws do not depend on whether its
onset slips.
"""
import random
from fractions import Fraction as Fr

SEEDS = range(16001, 16101)          # the 100 worlds of the benchmark
UNRELIABLE = 0.2                     # share of unreliable surveys
UNRELIABLE_CM, PRECISE_CM = 60, 20   # their error bounds [cm]
ONSET = (200, 240)                   # the forecast onset [periods]
LEAD = (10, 60)                      # the lead time [periods]
SLIP = 0.3                           # share of worlds whose onset slips
SLIP_MAX = 10                        # [periods]


def uniform_int(rng, lo, hi):
    """An integer uniform on lo..hi from one draw of rng.random()."""
    return lo + int((hi - lo + 1) * rng.random())


class World:
    def __init__(self, seed):
        rng = random.Random(seed)
        self.seed = seed
        self.ramp = Fr(round((17 + 4 * rng.random()) * 100), 100)
        self.unreliable = rng.random() < UNRELIABLE
        w = rng.random()
        cm = UNRELIABLE_CM if self.unreliable else PRECISE_CM
        self.survey_error = Fr(round((2 * w - 1) * cm), 100)
        self.surveyed = self.ramp + self.survey_error
        self.onset_forecast = uniform_int(rng, *ONSET)
        self.lead = uniform_int(rng, *LEAD)
        self.announce = self.onset_forecast - self.lead
        slips = rng.random() < SLIP
        size = uniform_int(rng, 1, SLIP_MAX)
        later = rng.random() < 0.5
        self.slip = (size if later else -size) if slips else 0
        self.onset = self.onset_forecast + self.slip

    def describe(self):
        return dict(world=self.seed, ramp=self.ramp, unreliable=self.unreliable,
                    survey_error=self.survey_error, onset_forecast=self.onset_forecast,
                    lead=self.lead, announce=self.announce, slip=self.slip,
                    onset=self.onset)


def family(seeds=SEEDS):
    return [World(s) for s in seeds]
