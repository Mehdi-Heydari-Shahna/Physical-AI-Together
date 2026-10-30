"""
The A300's drive controller, as the simulators of Chapter 13 run it.

Physical AI, Together, Chapter 13 (Section 13.1).

On the robot, a velocity command (v, w) goes to platform_velocity_controller,
an instance of diff_drive_controller (ros2_controllers), configured by
description/clearpath_control/config/a300/control/diff_4wd.yaml: at 50 Hz it
limits the command's speed, acceleration and jerk, turns it into one wheel
speed for each side, and integrates odometry from the wheels' positions;
the motor drivers then hold those wheel speeds. This module ports the parts
of diff_drive_controller that the simulators need, for the release of
ros2_controllers for ROS 2 Jazzy (branch jazzy, commit d71703b, of
19 October 2026), with the rate limiter of control_toolbox (branch jazzy,
commit 1e54022):

  * the limiter (SpeedLimiter, RateLimiter::limit): the jerk (not limited in
    the A300's configuration, NaN), then the first derivative with bounds
    that depend on the sign of the previous command, then the value;
  * the wheel speeds: (v -/+ w W / 2) / r, W the wheel separation times its
    multiplier, r the wheel radius times its multiplier, the same for both
    wheels of a side;
  * the odometry (Odometry::update with position feedback, the default):
    the mean of each side's wheel positions, differenced, integrated
    exactly; speeds as rolling means over the window of the configuration.

The command timeout and the publishing are left out: the adapters command
the controller at every update. The configuration also sets
preserve_turning_radius and publish_wheel_data, which the controller of
that branch does not declare; the port has nothing to follow for them.

Licence: this module translates into Python parts of diff_drive_controller
(src/diff_drive_controller.cpp, src/odometry.cpp,
include/diff_drive_controller/speed_limiter.hpp) and of control_toolbox
(include/control_toolbox/rate_limiter.hpp), each "Copyright 2020 PAL
Robotics S.L." and licensed under the Apache License, Version 2.0
(LICENSE-Apache-2.0.txt in this folder). Changes: translated to Python, the
parts listed above only.
"""
import collections
import math

from . import asset


def _limits(cfg, axis):
    g = lambda k: asset.number(cfg[f"{axis}.{k}"])     # noqa: E731
    return dict(min_value=g("min_velocity"), max_value=g("max_velocity"),
                min_neg=g("max_acceleration_reverse"), max_pos=g("max_acceleration"),
                min_pos=g("max_deceleration"), max_neg=g("max_deceleration_reverse"),
                min_jerk=g("min_jerk"), max_jerk=g("max_jerk"))


class RateLimiter:
    """control_toolbox::RateLimiter, as SpeedLimiter constructs it."""

    def __init__(self, min_value, max_value, min_neg, max_pos, min_pos, max_neg,
                 min_jerk, max_jerk):
        nan = math.isnan
        self.has_value = not nan(max_value)
        self.min_value = -max_value if nan(min_value) else min_value
        self.max_value = max_value
        self.has_first = not nan(max_pos)
        min_neg = -max_pos if nan(min_neg) else min_neg
        if self.has_first:
            max_neg = max_pos if nan(max_neg) else max_neg
            min_pos = min_neg if nan(min_pos) else min_pos
        self.min_neg, self.max_pos, self.min_pos, self.max_neg = min_neg, max_pos, min_pos, max_neg
        self.has_second = not nan(max_jerk)
        self.min_jerk = -max_jerk if nan(min_jerk) else min_jerk
        self.max_jerk = max_jerk

    def limit(self, v, v0, v1, dt):
        if self.has_second:
            dv, dv0 = v - v0, v0 - v1
            if (dv - dv0) * (v - v0) > 0:
                dt2 = dt * dt
                da = min(max(dv - dv0, self.min_jerk * dt2), self.max_jerk * dt2)
                v = v0 + dv0 + da
        if self.has_first:
            if v0 > 0.0:
                lo, hi = self.min_pos * dt, self.max_pos * dt
            elif v0 < 0.0:
                lo, hi = self.min_neg * dt, self.max_neg * dt
            else:
                lo, hi = self.min_neg * dt, self.max_pos * dt
            v = v0 + min(max(v - v0, lo), hi)
        if self.has_value:
            v = min(max(v, self.min_value), self.max_value)
        return v


class RollingMean:
    def __init__(self, size):
        self.values = collections.deque(maxlen=size)

    def add(self, x):
        self.values.append(x)
        return sum(self.values) / len(self.values)


class A300Drive:
    """diff_drive_controller with the A300's configuration."""

    def __init__(self, config=None):
        cfg = config or asset.controller()
        self.config = cfg
        self.rate = asset.number(cfg["update_rate"])
        self.period = 1.0 / self.rate
        self.separation = (asset.number(cfg["wheel_separation_multiplier"])
                           * asset.number(cfg["wheel_separation"]))
        self.radius_left = (asset.number(cfg["left_wheel_radius_multiplier"])
                            * asset.number(cfg["wheel_radius"]))
        self.radius_right = (asset.number(cfg["right_wheel_radius_multiplier"])
                             * asset.number(cfg["wheel_radius"]))
        self.linear = RateLimiter(**_limits(cfg, "linear.x"))
        self.angular = RateLimiter(**_limits(cfg, "angular.z"))
        self.window = int(asset.number(cfg["velocity_rolling_window_size"]))
        self.reset()

    def reset(self, left_pos=0.0, right_pos=0.0):
        self.previous = collections.deque([(0.0, 0.0), (0.0, 0.0)], maxlen=2)
        self.x = self.y = self.heading = 0.0
        self.old_left = left_pos * self.radius_left
        self.old_right = right_pos * self.radius_right
        self.lin_mean = RollingMean(self.window)
        self.ang_mean = RollingMean(self.window)
        self.odom_linear = self.odom_angular = 0.0
        self.command = (0.0, 0.0)

    def update(self, v, w, left_pos, right_pos):
        """One update: the command (v, w) and the mean positions of each
        side's wheels [rad]; returns the wheel speeds of the left and right
        side [rad/s]."""
        dt = self.period
        (lin1, ang1), (lin0, ang0) = self.previous[0], self.previous[1]
        v = self.linear.limit(v, lin0, lin1, dt)
        w = self.angular.limit(w, ang0, ang1, dt)
        self.previous.append((v, w))
        self.command = (v, w)
        self._odometry(left_pos, right_pos, dt)
        left = (v - w * self.separation / 2.0) / self.radius_left
        right = (v + w * self.separation / 2.0) / self.radius_right
        return left, right

    def _odometry(self, left_pos, right_pos, dt):
        cur_left, cur_right = left_pos * self.radius_left, right_pos * self.radius_right
        dl, dr = cur_left - self.old_left, cur_right - self.old_right
        self.old_left, self.old_right = cur_left, cur_right
        linear = 0.5 * (dl + dr)
        angular = (dr - dl) / self.separation
        if abs(angular) < 1e-6:
            direction = self.heading + 0.5 * angular
            self.x += linear * math.cos(direction)
            self.y += linear * math.sin(direction)
            self.heading += angular
        else:
            old, r = self.heading, linear / angular
            self.heading += angular
            self.x += r * (math.sin(self.heading) - math.sin(old))
            self.y += -r * (math.cos(self.heading) - math.cos(old))
        self.odom_linear = self.lin_mean.add(linear / dt)
        self.odom_angular = self.ang_mean.add(angular / dt)
