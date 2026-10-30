"""
What the Husky knows of where it is: its IMU and two estimators of its pose.

Physical AI, Together, Chapter 13 (Section 13.2).

The robot carries the four wheel encoders and the 6-axis IMU of model.py.
MuJoCo's sensors report the simulated quantities exactly; a real gyro does
not. Imu adds to the gyro's yaw rate a constant bias and white noise, ours,
declared: the bias, 0.004 rad/s, and the noise, a standard deviation of
0.001 rad/s per reading at 50 Hz, are values of a modest MEMS gyro, not the
specification of the A300's IMU, which the manual does not give. The noise
is drawn from NumPy's PCG64 generator with a declared seed. The encoders are
read exactly: the manual gives no resolution, and the model has none.

Both estimators update at 50 Hz, with the drive controller, from the wheel
positions and, for the second, the gyro; both start from the pose at which
the robot is placed, as a robot starts from a known pose at its dock.

  * A300Odometry is the odometry that the A300's controller publishes
    (drive.A300Drive, from diff_drive_controller): the mean of each side's
    wheel positions times the controller's wheel radius of 0.1625 m, and the
    heading from their difference over the separation of 0.9835 m. Section 13.1.4 measured both: the distance 1.51 % short, the heading of a turn
    5 % too large.
  * DeadReckoning takes the heading from the gyro, less the bias that it
    estimates while the robot stands before the start, and the distance
    from the mean of the four wheels' positions times the rolling radius
    measured in Section 13.1.4, 0.16499 m. It assumes that the robot moves
    along its heading. A skid-steered robot that turns while it translates
    does not: its centre slides outwards (Section 13.2.1). With
    sideslip=True, the estimator adds a lateral velocity -K u |Omega| Omega
    for its speed u and yaw rate Omega, as if the instantaneous centre of
    rotation lay K u |Omega| ahead of the centre, in proportion to the
    centripetal acceleration u |Omega|, in a turn to either side: outwards,
    to the right in a turn to the left and to the left in a turn to the
    right. K is calibrated for each ground, as
    the rolling radius was, from the turn of radius 3 m at 1 m/s of
    Section 13.1.3, with the simulator's pose as the external reference
    that a calibration of a real robot would need (SIDESLIP; husky_route.py
    computes the values and checks that they are those used here). The
    ground is read from the yard's map at the estimated position.

route() gives the estimate in the coordinates of the route: the arc length
s, the offset d, positive to the left, and the heading error, wrapped to
(-pi, pi].
"""
import math

import numpy as np

from . import model, world

RHO = 0.16499            # m, the rolling radius measured in Section 13.1.4 (asphalt, 1 m/s)
GYRO_BIAS = 0.004        # rad/s, ours
GYRO_NOISE = 0.001       # rad/s per reading at 50 Hz (standard deviation), ours
SEED = 13                # of the gyro's noise
CALIBRATION = 2.0        # s at rest before the start, over which the gyro's bias is estimated

# K [s^2]: the forward offset of the instantaneous centre of rotation over the
# centripetal acceleration, in the steady turn of radius 3 m at 1 m/s of
# Section 13.1.3; husky_route.py computes these values and checks them.
SIDESLIP = {"asphalt": 0.151, "gravel": 0.183, "wet grass": 0.275}


class Imu:
    """The gyro's yaw rate as the estimator reads it: the simulator's, plus
    a constant bias and white noise."""

    def __init__(self, husky, bias=GYRO_BIAS, noise=GYRO_NOISE, seed=None):
        self.h = husky
        self.adr = int(husky.m.sensor("gyro").adr[0])
        self.bias, self.noise = bias, noise
        self.seed = SEED if seed is None else seed
        self.rng = np.random.default_rng(self.seed)

    def yaw_rate(self):
        exact = float(self.h.d.sensordata[self.adr + 2])
        return exact + self.bias + self.noise * float(self.rng.standard_normal())


class Estimate:
    """A pose in the world frame and its rates, with the route coordinates."""

    def __init__(self):
        self.x = self.y = self.heading = 0.0
        self.speed = self.yaw_rate = 0.0

    def route(self):
        s, d = world.project(self.x, self.y)
        e = self.heading - world.point(s)[2]
        e = (e + math.pi) % (2.0 * math.pi) - math.pi
        return s, d, e


class A300Odometry(Estimate):
    """The odometry that the A300's drive controller integrates."""
    NAME = "the A300's odometry"

    def __init__(self, husky):
        super().__init__()
        self.h = husky

    def start(self):
        self.update(0.0)

    def update(self, dt):
        dr = self.h.drive
        self.x, self.y, self.heading = dr.x, dr.y, dr.heading
        self.speed, self.yaw_rate = dr.odom_linear, dr.odom_angular


class DeadReckoning(Estimate):
    """Heading from the gyro, distance from the encoders at the rolling
    radius; with sideslip=True, the lateral slide of the turn as well."""

    def __init__(self, husky, imu=None, rho=RHO, sideslip=False):
        super().__init__()
        self.h, self.imu = husky, imu or Imu(husky)
        self.rho, self.sideslip = rho, sideslip
        self.NAME = "dead reckoning" + (" with sideslip" if sideslip else "")
        self.bias = 0.0
        self.previous = None
        self.skip_calibration = False

    def wheels(self):
        q = self.h.d.qpos
        return [float(q[self.h.qadr[w]]) for w in model.WHEELS]

    def calibrate(self, seconds=CALIBRATION):
        """Stands the robot still for 'seconds' and takes the mean gyro
        reading, one per controller update, as the bias."""
        n = int(round(seconds * self.h.drive.rate))
        total = 0.0
        for _ in range(n):
            self.h.step(self.h.steps_per_update)
            total += self.imu.yaw_rate()
        self.bias = total / n
        return self.bias

    def start(self):
        """From the robot's present pose, which it knows at the start."""
        self.x, self.y, self.heading = self.h.pose()
        self.previous = self.wheels()
        self.speed = self.yaw_rate = 0.0

    def update(self, dt):
        now = self.wheels()
        turned = sum(a - b for a, b in zip(now, self.previous)) / 4.0
        self.previous = now
        ds = self.rho * turned
        self.yaw_rate = self.imu.yaw_rate() - self.bias
        dh = self.yaw_rate * dt
        mid = self.heading + 0.5 * dh
        lateral = 0.0
        if self.sideslip:
            k = SIDESLIP[world.ground_at_xy(self.x, self.y).name]
            w = self.yaw_rate
            lateral = -k * (ds / dt if dt > 0 else 0.0) * abs(w) * w * dt
        self.x += ds * math.cos(mid) - lateral * math.sin(mid)
        self.y += ds * math.sin(mid) + lateral * math.cos(mid)
        self.heading += dh
        self.speed = ds / dt if dt > 0 else 0.0


def estimator(kind, husky):
    """'a300', 'controller radius' (dead reckoning with the sideslip, at the
    controller's wheel radius of 0.1625 m), 'uncalibrated' (with the
    sideslip, the gyro's bias taken as zero), 'dead reckoning' or
    'sideslip'."""
    if kind == "a300":
        return A300Odometry(husky)
    if kind == "controller radius":
        est = DeadReckoning(husky, rho=0.1625, sideslip=True)
        est.NAME = "dead reckoning with sideslip, at the controller's radius"
        return est
    if kind == "uncalibrated":
        est = DeadReckoning(husky, sideslip=True)
        est.NAME = "dead reckoning with sideslip, the gyro's bias not estimated"
        est.skip_calibration = True
        return est
    if kind == "dead reckoning":
        return DeadReckoning(husky)
    if kind == "sideslip":
        return DeadReckoning(husky, sideslip=True)
    raise ValueError(kind)
