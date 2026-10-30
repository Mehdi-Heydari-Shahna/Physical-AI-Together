"""The worlds of the complete case (Section 16.4).

Each world is a world of the anticipation benchmark (../bench/worlds.py),
drawn from its seed in the same order: the start of the ramp, uniform on
[17, 21] m on a 1 cm grid, which here is the ground coordinate at which the
simulated crest begins; the survey's reliability (unreliable with
probability 0.2) and error (within 0.6 m or 0.2 m); the forecast onset of
the operation, the lead time and the slip. The sensor's errors are drawn
from two generators seeded from the world's seed: one for the measurements
within each period, which the laws receive, and one for the measurements at
the period boundaries, which the governor receives.

The seeds of each phase are disjoint, and they are fixed here, before any
run of the protocol (PROTOCOL.md):
  development      42001 to 42050   writing and testing the software
  identification   43001 to 43040   the model's bounds (complete_identify.py)
  calibration      44001 to 44100   the survey's offset (complete_calibrate.py)
  training         47001 to 47300   the learner's episodes (complete_train.py)
  evaluation       45001 to 45100   the paired evaluation
  stress           46001 to 46030   the shifted stress tests
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from bench.worlds import World as BenchWorld          # noqa: E402

PHASES = {
    "development": range(42001, 42051),
    "identification": range(43001, 43041),
    "calibration": range(44001, 44101),
    "training": range(47001, 47301),
    "evaluation": range(45001, 45101),
    "stress": range(46001, 46031),
}


class World(BenchWorld):
    """A world of the benchmark's family, with the seeds of its sensor."""

    def __init__(self, seed):
        super().__init__(seed)
        self.sensor_seed = 10 * seed + 1        # measurements within the periods
        self.boundary_seed = 10 * seed + 2      # measurements at the period boundaries

    @property
    def ramp_start(self):
        return float(self.ramp)


def family(phase):
    return [World(s) for s in PHASES[phase]]
