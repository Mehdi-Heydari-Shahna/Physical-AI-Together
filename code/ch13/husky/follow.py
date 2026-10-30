"""
The feedback of Section 13.2.2: following the route at a commanded speed.

Physical AI, Together, Chapter 13 (Section 13.2).

The plans of the governor command a motion along the route: in each period,
a speed u_c(tau) that changes at a constant rate from c0 to c1, so that the
commanded arc length is s_c(tau) = s0 + c0 tau + (c1 - c0) tau^2 / (2 T).
The feedback turns that motion and the estimate of the robot's pose into
the velocity command (v, omega) that the A300's drive controller receives,
once per update of the controller (50 Hz):

  * along the route, v = u_c(tau) + K_S (s_c(tau) - s_hat), never below zero:
    the commanded speed, corrected by the estimate's lag behind the
    commanded arc length;
  * across it, omega = kappa v cos(e) / (1 - kappa d) - K_D v d - K_E w e +
    z, the yaw rate that keeps a robot on the route of curvature kappa,
    less terms in the estimated offset d and heading error e, with
    w = max(v, W_MIN), and z the integral of K_I times the estimated yaw
    rate's shortfall from the yaw rate asked of the robot: in the turns of
    radius 3 m at 1 m/s of the calibration (husky_route.py), the robot turns
    3.5 to 5.0 % less than the controller is asked, and the integral removes
    the offset that this would leave;
  * the speed command is divided by the ratio of the rolling radius to the
    controller's wheel radius, 0.16499 / 0.1625, so that the controller's
    wheel speeds roll the robot at v.

The gains are ours: K_S = 2 1/s, K_D = 1.5 1/m^2, K_E = 3 1/m with
W_MIN = 0.2 m/s, and K_I = 4 1/s. The feedback sees the estimate, never the
simulator's pose: a wrong estimate steers the robot wrongly
(Section 13.2.1).
"""
import math

from . import estimate, world

K_S = 2.0        # 1/s
K_D = 1.5        # 1/m^2
K_E = 3.0        # 1/m
W_MIN = 0.2      # m/s
K_I = 4.0        # 1/s
R_CONTROLLER = 0.1625      # m, the controller's wheel radius (diff_4wd.yaml)


class Motion:
    """The commanded motion of one period: the arc length s0 and the speed
    c0 at its start, the speed c1 at its end, over the period T."""
    __slots__ = ("s0", "c0", "c1", "T")

    def __init__(self, s0, c0, c1, T):
        self.s0, self.c0, self.c1, self.T = s0, c0, c1, T

    def speed(self, tau):
        return self.c0 + (self.c1 - self.c0) * min(max(tau, 0.0), self.T) / self.T

    def arc(self, tau):
        tau = min(max(tau, 0.0), self.T)
        return self.s0 + self.c0 * tau + (self.c1 - self.c0) * tau * tau / (2.0 * self.T)

    def end(self):
        return self.arc(self.T), self.c1


class Follower:
    """The feedback law, with the state of its yaw-rate integral."""

    def __init__(self, k_s=K_S, k_d=K_D, k_e=K_E, w_min=W_MIN, k_i=K_I, rho=estimate.RHO):
        self.k_s, self.k_d, self.k_e, self.w_min, self.k_i = k_s, k_d, k_e, w_min, k_i
        self.scale = R_CONTROLLER / rho
        self.z = 0.0
        self.asked = None

    def reset(self):
        self.z, self.asked = 0.0, None

    def command(self, motion, tau, est, dt):
        """The velocity command (v, omega) for the drive controller at time
        tau within the period, from the estimate est; dt is the time since
        the last call (the integral's step)."""
        if self.asked is not None:
            self.z += self.k_i * (self.asked - est.yaw_rate) * dt
        s, d, e = est.route()
        v = max(0.0, motion.speed(tau) + self.k_s * (motion.arc(tau) - s))
        kappa = world.curvature(s)
        w = kappa * v * math.cos(e) / (1.0 - kappa * d) - self.k_d * v * d \
            - self.k_e * max(v, self.w_min) * e
        self.asked = w
        return v * self.scale, w + self.z
