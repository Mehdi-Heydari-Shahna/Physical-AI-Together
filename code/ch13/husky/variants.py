"""
Substitutions and changed assumptions for the Husky (Section 13.3).

Physical AI, Together, Chapter 13 (Section 13.3).

Section 13.3 keeps the task, the yard, the observations and the record of
Section 13.2 and changes one thing at a time: a module, or something about
the world that the modules assume. This module holds what it substitutes:

  * PurePursuit: a second feedback law for the control package, with the
    speed loop of follow.Follower and the yaw rate of pure pursuit, which
    steers along the arc through the point of the route LOOKAHEAD ahead of
    the estimate's arc length: the curvature 2 y / l^2 of that arc, for the
    point's lateral offset y in the robot's frame and its distance l from
    the robot (Coulter 1992, who takes the point of the path at the distance
    LOOKAHEAD from the robot; the arc length is our simplification). The
    control package husky.route-speed-pursuit is husky.route-speed with this
    feedback: the same plans, test, bounds and recovery continuation;
  * two proposal methods: husky.hurry, which asks for 1.5 m/s throughout and
    never for rest, and husky.careful, which asks for 0.5 m/s until the
    measured front reaches the dock zone, and for rest from then on;
  * payload_spec: the model of Section 13.1 with a payload on the chassis,
    the manual's all-terrain payload of 51.5 kg by default: a box of
    0.4 m x 0.3 m x 0.1 m, centred on the chassis, with its centre 0.30 m
    above the ground at rest (ours; the manual gives the mass, not where it
    sits), that does not touch anything;
  * DelayedDeadReckoning: dead reckoning with sideslip whose encoder and
    gyro readings arrive a number of controller updates late (20 ms each):
    the encoders' from the first update, the gyro's from the update after
    the delay, its first reading standing in before (the robot starts from
    rest, straight); and, compensated, the same estimate carried forward over
    the delay with the velocity commands of the delayed updates, which the
    robot knows;
  * PeakHusky: the robot of sim.py, which also records, at every step of the
    simulator, the largest torque of a wheel's drive and whether a drive is
    at its limit: the torque that a sample at the controller's updates
    would miss, since the drivers' speed loops answer each new target of
    the controller at once.
"""
import collections
import math

import mujoco

from . import estimate, follow, governed, model, sim, world   # governed puts ../ch10 on the path
from allovi import register                                    # noqa: E402

LOOKAHEAD = 1.0            # m, the default look-ahead of pure pursuit (ours)
PAYLOAD = 51.5             # kg, the manual's payload on all terrain
PAYLOAD_HALF = (0.2, 0.15, 0.05)       # m, half sizes of the payload's box (ours)
PAYLOAD_HEIGHT = 0.30      # m, its centre above the ground at rest (ours)


# ----------------------------------------------------------------------------
# A second feedback law
# ----------------------------------------------------------------------------
class PurePursuit(follow.Follower):
    """The speed loop of follow.Follower; the yaw rate of pure pursuit
    towards the route's point at the estimated arc length plus the
    look-ahead, from the estimated pose. No integral, no feedforward of the
    route's curvature."""

    def __init__(self, lookahead=LOOKAHEAD, **kw):
        super().__init__(**kw)
        self.lookahead = lookahead

    def command(self, motion, tau, est, dt):
        s, d, e = est.route()
        v = max(0.0, motion.speed(tau) + self.k_s * (motion.arc(tau) - s))
        tx, ty, _ = world.point(s + self.lookahead)
        dx, dy = tx - est.x, ty - est.y
        c, sn = math.cos(est.heading), math.sin(est.heading)
        ahead, left = c * dx + sn * dy, -sn * dx + c * dy
        dist2 = ahead * ahead + left * left
        w = 2.0 * v * left / dist2 if dist2 > 1e-12 else 0.0
        self.asked = w
        return v * self.scale, w


@register
class RouteSpeedPursuit(governed.RouteSpeed):
    """husky.route-speed with the feedback of pure pursuit (PurePursuit)."""
    NAME = "husky.route-speed-pursuit"

    def __init__(self, lookahead=LOOKAHEAD):
        super().__init__()
        self.feedback = PurePursuit(lookahead=lookahead)


# ----------------------------------------------------------------------------
# Two proposal methods
# ----------------------------------------------------------------------------
@register
class Hurry:
    """Asks for 1.5 m/s throughout, beyond the route's limit, and never for
    rest."""
    ROLE, NAME, CONTRACT = "proposal", "husky.hurry", "1.0"

    def propose(self, observation):
        return 1.5


@register
class Careful(governed.Cruise):
    """Asks for 0.5 m/s until the measured front first reaches the dock zone,
    and for rest from then on."""
    NAME = "husky.careful"

    def __init__(self):
        super().__init__(speed=0.5)


# ----------------------------------------------------------------------------
# A payload
# ----------------------------------------------------------------------------
def payload_spec(mass=PAYLOAD, **build):
    """The MjSpec of model.build(**build) with a payload of the given mass
    on the chassis (module docstring)."""
    spec = model.build(**build)
    rest = model.rest_height(spec.compile())     # base_link above the ground at rest
    base = spec.body("base_link")
    body = base.add_body(name="payload", pos=[0.0, 0.0, PAYLOAD_HEIGHT - rest])
    g = body.add_geom(name="payload", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=list(PAYLOAD_HALF), mass=mass, rgba=[0.2, 0.3, 0.7, 1.0])
    g.contype, g.conaffinity = 0, 0
    return spec


# ----------------------------------------------------------------------------
# The drives' torques, at every step
# ----------------------------------------------------------------------------
class PeakHusky(sim.Husky):
    """sim.Husky that records, at every step of the simulator, the largest
    torque of a wheel's drive and whether a drive is at its limit: since the
    last call of take() (peak, steps), and over the steps that end later
    than 'after' seconds from the robot's placing (peak_all, steps_all). The
    physics are those of sim.Husky, step for step."""

    def __init__(self, *args, after=0.0, **kw):
        super().__init__(*args, **kw)
        self._ids = [self.act[w] for w in model.WHEELS]
        self._limit = min(self.limit.values())
        self.after = after
        self.peak = self.peak_all = 0.0
        self.steps = self.steps_all = 0

    def step(self, n=1):
        for _ in range(n):
            super().step(1)
            f = max(abs(float(self.d.actuator_force[i])) for i in self._ids)
            at = f >= self._limit - 1e-9
            self.peak = max(self.peak, f)
            self.steps += at
            if self.step_count * self.m.opt.timestep > self.after + 1e-12:
                self.peak_all = max(self.peak_all, f)
                self.steps_all += at

    def take(self):
        """The largest torque and the number of steps at the limit since the
        last call, which it resets."""
        out = (self.peak, self.steps)
        self.peak, self.steps = 0.0, 0
        return out


# ----------------------------------------------------------------------------
# A late sensor
# ----------------------------------------------------------------------------
class _LateImu:
    """The gyro of estimate.Imu, read 'delay' updates late: until 'delay'
    readings are queued, the first reading stands in for the late ones."""

    def __init__(self, imu, delay):
        self.imu = imu
        self.queue = collections.deque()
        self.delay = delay

    def yaw_rate(self):
        self.queue.append(self.imu.yaw_rate())
        if len(self.queue) <= self.delay:
            return self.queue[0]
        return self.queue.popleft()


class DelayedDeadReckoning(estimate.DeadReckoning):
    """Dead reckoning with sideslip on encoder and gyro readings that arrive
    'delay' controller updates late; with compensate=True, the delayed
    estimate carried forward over the delay with the velocity commands of
    the last 'delay' updates, and the speed taken from the latest command.
    The calibration at rest reads the gyro on time: the robot stands still."""

    def __init__(self, husky, delay, compensate=False):
        super().__init__(husky, sideslip=True)
        self.delay, self.compensate = delay, compensate
        self.NAME = (f"dead reckoning with sideslip, {20 * delay} ms late"
                     + (", compensated" if compensate else ""))
        self.on_time = self.imu
        self.wheel_queue = collections.deque()
        self.commands = collections.deque(maxlen=max(delay, 1))
        self.scale = follow.Follower().scale
        self.late = None                 # the delayed pose (x, y, heading)

    def wheels(self):
        now = super().wheels()
        if self.delay == 0:
            return now
        self.wheel_queue.append(now)
        if len(self.wheel_queue) <= self.delay:
            return self.wheel_queue[0]
        return self.wheel_queue.popleft()

    def start(self):
        self.imu = _LateImu(self.on_time, self.delay) if self.delay else self.on_time
        self.wheel_queue.clear()
        now = super(DelayedDeadReckoning, self).wheels()
        for _ in range(self.delay):
            self.wheel_queue.append(now)
        super().start()
        self.late = (self.x, self.y, self.heading)
        self.commands.clear()

    def update(self, dt):
        if self.late is not None:
            self.x, self.y, self.heading = self.late
        super().update(dt)
        self.late = (self.x, self.y, self.heading)
        if not (self.compensate and self.delay):
            return
        self.commands.append((self.h.command[0] / self.scale, self.h.command[1]))
        x, y, hd = self.late
        for v, w in list(self.commands)[-self.delay:]:
            mid = hd + 0.5 * w * dt
            x += v * dt * math.cos(mid)
            y += v * dt * math.sin(mid)
            hd += w * dt
        self.x, self.y, self.heading = x, y, hd
        self.speed = self.commands[-1][0]
