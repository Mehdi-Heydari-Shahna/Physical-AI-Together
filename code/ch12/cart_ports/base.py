"""The simulated cart in a simulator other than MuJoCo (Section 12.1): the
track coordinate, placement at a state, and the drive loop of Section 11.1,
line for line.

What differs between simulators is collected in six operations that each
port supplies:

  _build()            the model and the solver, from the MJCF text of Section 11.1
  _kinematics()       positions and velocities of the axles, pitch, reach and wheel rates
  _contact_loads()    the normal load of each wheel in the last step
  _write(q, qd)       the joint state (x, z, pitch, front, rear, reach) and its rates
  _apply(tf, tr, ell) the wheel torques and the tool's target for the next step
  _step()             one step of the simulator

Everything a port reads is current when it is read: unlike MuJoCo's
mj_step, the step of each simulator here ends with its state, so that the
drive loop reads it before the next step without a split step. The loads
are those of the last step, as the touch sensors of Section 11.1 give them;
right after a placement, before any step, the loop shares its force equally
between the wheels.
"""
import math
from collections import namedtuple

G = 9.81               # gravity of every model [m/s^2]
V_REST = 5e-3          # below 5 mm/s the drive loop counts the cart as at rest [m/s]

Kin = namedtuple("Kin", "front front_vel rear pitch reach front_rate rear_rate")


class StateNotFinite(RuntimeError):
    """The simulator returned a state that is not a number."""

"""front, rear: axle positions (x, z) [m]; front_vel: front axle velocity
(vx, vz) [m/s]; pitch [rad]; reach [m]; wheel rates relative to the chassis
[rad/s]."""


class PortedCart:
    """The cart on its track in a simulator other than MuJoCo; see the module
    docstring. coordinate="ground" is the mistake of Section S11.9.1."""
    BACKEND = "abstract"

    def __init__(self, design, timestep=None, coordinate="axle"):
        self.design = design
        self.dt = float(timestep or design.timestep)
        self.coordinate = coordinate
        self.pr = design.profile()
        self.lead = design.half_length - design.wheel_offset    # bumper ahead of the axle
        self.steps = int(round(0.1 / self.dt))                  # steps per period
        self.mass = design.chassis_mass + 2 * design.wheel_mass + design.tool_mass
        self.wheel_inertia = 0.4 * design.wheel_mass * design.wheel_radius ** 2
        self.clock = 0                                          # steps taken
        self._kin = None
        self._loads = None                                      # none since the placement
        self._build()
        self.sink = 0.0
        self.sink = self.static_sink()

    # --- the operations of each port ----------------------------------------------
    def _build(self):
        raise NotImplementedError

    def _kinematics(self):
        raise NotImplementedError

    def _contact_loads(self):
        raise NotImplementedError

    def _write(self, q, qd):
        raise NotImplementedError

    def _apply(self, torque_front, torque_rear, ell):
        raise NotImplementedError

    def _step(self):
        raise NotImplementedError

    # --- the state of the cart ------------------------------------------------------
    def kin(self):
        if self._kin is None:
            k = self._kinematics()
            if not all(math.isfinite(x) for x in (*k.front, *k.front_vel, *k.rear, k.pitch, k.reach)):
                raise StateNotFinite(f"the state is not a number after {self.clock} steps")
            self._kin = k
        return self._kin

    def foot(self, wheel="front"):
        k = self.kin()
        return self.pr.foot(*(k.front if wheel == "front" else k.rear))

    def state(self, coordinate=None):
        """(p, v, l): the track coordinate of the bumper, its rate, and the
        reach of the tool."""
        k = self.kin()
        f = self.pr.foot(*k.front)
        vx, vz = k.front_vel
        if (coordinate or self.coordinate) == "ground":
            p, v = f.s + self.lead, self.pr.ground_rate(f, vx, vz)
        else:
            p, v = self.pr.axle(f.s) + self.lead, self.pr.axle_rate(f, vx, vz)
        return float(p), float(v), float(k.reach)

    def axle_speed(self):
        """The speed of the front axle along the ground's tangent at its foot."""
        k = self.kin()
        f = self.pr.foot(*k.front)
        vx, vz = k.front_vel
        return vx * math.cos(f.angle) - vz * math.sin(f.angle)

    def loads(self):
        """The normal load of each wheel in the last step [N], or None right
        after a placement."""
        return self._loads

    # --- placement -----------------------------------------------------------------------
    def _axle(self, s):
        x, z, _ = self.pr.point(s)
        nx, nz = self.pr.normal(s)
        h = self.design.wheel_radius - self.sink
        return x + h * nx, z + h * nz

    def _pose(self, p):
        """Chassis position (x, z) and pitch with the bumper at track
        coordinate p and both wheels on the ground (as in Section 11.1)."""
        sf = self.pr.ground(p - self.lead)
        F = self._axle(sf)
        base = 2 * self.design.wheel_offset
        lo, hi = sf - base - 0.2, sf
        for _ in range(80):
            mid = (lo + hi) / 2
            R = self._axle(mid)
            if math.hypot(F[0] - R[0], F[1] - R[1]) > base:
                lo = mid
            else:
                hi = mid
        R = self._axle((lo + hi) / 2)
        pitch = math.atan2(-(F[1] - R[1]), F[0] - R[0])
        wd = self.design.wheel_drop
        return ((F[0] + R[0]) / 2 + wd * math.sin(pitch),
                (F[1] + R[1]) / 2 + wd * math.cos(pitch), pitch)

    def place(self, p, v, l):
        """Put the cart at the state (p, v, l), rolling without slip at speed
        v, its wheels sunk by the static penetration, the tool still."""
        d = self.design
        cx, cz, pitch = self._pose(p)
        eps = 1e-6
        a, b = self._pose(p + eps), self._pose(p - eps)
        rates = [v * (a[i] - b[i]) / (2 * eps) for i in range(3)]
        wheel = v / d.wheel_radius - rates[2]                  # relative to the chassis
        zc = d.wheel_radius + d.wheel_drop
        self._write((cx + d.half_length, cz - zc, pitch, 0.0, 0.0, l),
                    (rates[0], rates[1], rates[2], wheel, wheel, 0.0))
        self._apply(0.0, 0.0, l)
        self._kin, self._loads = None, None

    def static_sink(self):
        """Penetration of the wheels at rest on the firm ground: the depth of
        the lower axle below the wheel radius."""
        self.place(5.0, 0.0, 0.0)
        for _ in range(30):
            self.period(0.0, 0.0)
        k = self.kin()
        return self.design.wheel_radius - min(k.front[1], k.rear[1])

    # --- the drive loop of Section 11.1 -----------------------------------------------
    def step(self):
        self._step()
        self.clock += 1
        self._kin = None
        self._loads = self._contact_loads()

    def controls(self, a_ref, v_ref, ell):
        """The wheels' motors get a force M (a_ref - g sin(pitch)) for the
        acceleration a_ref and for gravity along the chassis, and a feedback
        force M K (v_ref - v) on the cart's speed v, with the speed gain K,
        both shared between the wheels in proportion to their measured loads,
        and each wheel what its own inertia needs. The tool's servo is set to
        the reach ell."""
        d, r = self.design, self.design.wheel_radius
        pitch = self.kin().pitch
        v = self.state()[1]
        force = self.mass * (a_ref - G * math.sin(pitch) + d.speed_gain * (v_ref - v))
        loads = self.loads() or (1.0, 1.0)
        lim = d.drive_limit
        taus = []
        for load in loads:
            share = load / sum(loads) if sum(loads) > 0 else 0.5
            tau = force * r * share + self.wheel_inertia * a_ref / r
            taus.append(max(-lim, min(lim, tau)))
        self._apply(taus[0], taus[1], ell)

    def period(self, u, ell, kind="segment", record=None, steps=None):
        """One period of 0.1 s under the commanded acceleration u and the
        commanded reach ell, held over the period; or its first 'steps'
        steps."""
        v0 = self.state()[1]
        if abs(v0) < V_REST:
            v0 = 0.0                     # at rest: start from zero
        if kind == "rest":               # brake, then hold
            brake = 1 if v0 > 0 else 0
        else:
            brake = 1 if v0 > 0 > u else -1 if v0 < 0 < u else 0
        for i in range(steps or self.steps):
            tau = i * self.dt
            v_ref, a_ref = v0 + u * tau, u
            if (kind == "rest" and v0 <= 0) or (brake and v_ref * v0 <= 0):
                v_ref, a_ref = 0.0, 0.0  # at rest: hold
            if record is not None:
                record.append(self.sample(a_ref, v_ref))
            self.controls(a_ref, v_ref, ell)
            self.step()
        return self.state()

    def sample(self, a_ref, v_ref):
        """The signals at the current step, for the records."""
        p, v, l = self.state()
        k = self.kin()
        loads = self.loads() or (math.nan, math.nan)
        return dict(t=self.clock * self.dt, p=p, v=v, l=l, a_ref=a_ref, v_ref=v_ref,
                    axle=self.axle_speed(), front_rate=k.front_rate, rear_rate=k.rear_rate,
                    pitch=k.pitch, front_load=loads[0], rear_load=loads[1])
