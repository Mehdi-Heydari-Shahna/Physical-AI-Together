"""Tests of the two checks of the core's loop that Algorithm 8.1 makes at
each period boundary: a distance that is not a finite number raises the
alarm, and a restart plan takes over only if the measurement taken when it
is committed passes the check against the set on which it was certified.

    cd code/ch10
    python3 -m unittest discover -s tests -v

Python 3.8 or later; no other packages.
"""
import os
import sys
import unittest
from fractions import Fraction as Fr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.dont_write_bytecode = True
import allovi                                    # noqa: E402
import cart                                      # noqa: E402
from allovi.loop import within_bound             # noqa: E402

ce = cart.ce
ALARM_RUN = dict(capability="cart.survey",
                 capability_args={"ramp_start": ce.RAMP + Fr(1, 5)},
                 rest="global", postpone=None)


class WithinBound(unittest.TestCase):
    def test_exact_numbers(self):
        self.assertTrue(within_bound(Fr(1, 2), Fr(1)))
        self.assertTrue(within_bound(0, 0))
        self.assertFalse(within_bound(Fr(3, 2), Fr(1)))

    def test_floating_point(self):
        self.assertTrue(within_bound(0.5, 1.0))
        self.assertFalse(within_bound(1.5, 1.0))
        self.assertFalse(within_bound(float("nan"), 1.0))
        self.assertFalse(within_bound(float("inf"), 1.0))


class NotANumber(unittest.TestCase):
    """The first six periods of the run of Chapter 3, with a distance that is
    NaN in period 5: the alarm is raised at t_6."""

    def test_nan_raises_the_alarm(self):
        cfg = cart.configuration("chapter3", postpone=None)
        exact, calls = cfg.control.distance, []

        def distance(measured, initial):
            calls.append(1)
            return float("nan") if len(calls) == 6 else exact(measured, initial)
        cfg.control.distance = distance
        rec = allovi.run(cfg, 6)
        self.assertEqual(rec.alarms, [6])

    def test_without_nan_no_alarm(self):
        rec = allovi.run(cart.configuration("chapter3", postpone=None), 6)
        self.assertEqual(rec.alarms, [])


class RestartCheck(unittest.TestCase):
    """The alarm run of Section 8.3.4 with the rest feedback at the global
    capability's braking: it restarts once."""

    def test_the_run_as_reported(self):
        rec = allovi.run(cart.configuration("alarm-bias-0.2-global", **ALARM_RUN), 220)
        self.assertEqual(len(rec.alarms), 1)
        self.assertEqual(len(rec.restarts), 1)

    def test_a_failed_check_refuses_the_restart(self):
        cfg = cart.configuration("alarm-bias-0.2-global", **ALARM_RUN)
        reach0, distance0, sets = cfg.control.reach, cfg.control.distance, []

        def reach(measured, bound):
            s, why = reach0(measured, bound)
            if s is not None:
                sets.append(s)
            return s, why

        def distance(measured, initial):
            if any(initial is s for s in sets):
                return Fr(1)            # the measurement misses the restart's set
            return distance0(measured, initial)
        cfg.control.reach, cfg.control.distance = reach, distance
        rec = allovi.run(cfg, 220)
        self.assertEqual(rec.restarts, [])
        self.assertGreater(len(rec.alarms), 1)


if __name__ == "__main__":
    unittest.main()
