"""The plant of the complete case: the MuJoCo cart of Chapter 11 executing
feedback laws, with a modelled sensor (Section 16.4).

The cart, its track and every number of its MJCF model are those of
../../ch11/cart_mujoco (Design), except the start of the ramp, which each
world draws. What this module adds is what a feedback law needs that the
constant commands of Chapter 11 did not:

  * the law is evaluated at every time step of the simulator (1 ms), on the
    measured state, and returns the commanded acceleration u_a [m/s^2] and
    the commanded reach u_l [m];
  * the drive realizes the commanded acceleration as the drive loop of
    Section 11.1.2 does, with a force M (u_a - g sin(pitch)) shared by the
    wheels in proportion to their measured loads, each wheel's own inertia,
    and a speed loop of gain 100 /s around a speed reference that integrates
    the command (continuous across periods, held within 2 cm/s of the cart's
    speed so that it cannot wind up); it also adds the force of the tool's
    servo, which pushes the chassis back while the tool unfolds;
  * the tool moves to a new commanded reach along a smooth path (a
    polynomial of degree five, from rest to rest) in TOOL_TIME = 0.35 s,
    which its servo follows with the servo's own dynamics as feedforward, so
    that it never reaches its force limit. During development, the step
    command of Section 11.1, with which the servo unfolds the tool in about
    70 ms at its force limit, was found to lift the wheels off the ground for
    a few milliseconds when the tool unfolded while the cart backed away;
  * a sensor measures the position, speed and reach with independent
    Gaussian errors, from a seeded generator.

The speed loop and the tool's force use the simulator's own values, as a
motor controller uses its encoders and its current; the law sees only the
sensor. Nothing here certifies anything: what the realized motion does is
what the identification measures (complete_identify.py).
"""
import math
import random

import mujoco

from cart_mujoco.scene import Design
from cart_mujoco.sim import G, Sim

T = 0.1                  # period of the governor [s]
U_MAX = 2.0              # command bound |u_a| <= 2 m/s^2 (the actuation of Chapter 10)
INNER_GAIN = 100.0       # speed gain of the drive loop [1/s] (Design.speed_gain)
INNER_CLAMP = 0.02       # the drive loop's speed reference stays this close to the speed [m/s]
SETTLE = 200             # time steps of holding still after placement (0.2 s)
TOOL_TIME = 0.35         # the tool reaches a new commanded reach in 0.35 s


class Sensor:
    """The modelled sensor: position, speed and reach with independent
    Gaussian errors of standard deviations sigma = (position [m], speed
    [m/s], reach [m]), drawn from random.Random(seed). The errors are not
    bounded by construction; the bound that the certificates use is
    established on the identification runs (complete_identify.py)."""

    SIGMA = (2.5e-4, 5e-4, 1e-4)

    def __init__(self, seed, sigma=SIGMA):
        self.rng = random.Random(seed)
        self.sigma = tuple(float(s) for s in sigma)

    def __call__(self, p, v, l):
        g, (sp, sv, sl) = self.rng.gauss, self.sigma
        return (p + g(0.0, sp) if sp else p,
                v + g(0.0, sv) if sv else v,
                l + g(0.0, sl) if sl else l)


class Plant(Sim):
    """The simulated cart, executing one law per period.

    A law is any object that can be called as law(tau, measured), with the
    time tau [s] within the period and the measured state (p, v, l), and
    returns (u_a, u_l). run_period(law) executes it for one period and
    returns the true state (p, v, l) at the end, as floats; the measured
    states that the law received are drawn from 'sensor'. With
    record=True, every time step is kept in self.trace (see step_record)."""

    def __init__(self, design=None, sensor=None, record=False, drive_mass=None):
        super().__init__(design or Design())
        self.drive_mass = self.mass if drive_mass is None else float(drive_mass)
        self.sensor = sensor or Sensor(0, (0.0, 0.0, 0.0))
        self.record = record
        self.trace = []
        self.v_inner = None
        self.clipped = 0          # time steps at which |u_a| exceeded U_MAX
        self.time_steps = 0
        self.tool = None          # the tool's path: (start time, start reach, commanded reach)

    # --- placement ---------------------------------------------------------------
    def start(self, p=0.0):
        """The cart at rest at p, its tool folded, after SETTLE time steps in
        which the drive holds it still: the contacts settle before the run
        begins. Returns the true state."""
        self.place(p, 0.0, 0.0)
        self.v_inner = None
        self.tool = (float(self.data.time), 0.0, 0.0)
        m, d = self.m, self.data
        for _ in range(SETTLE):
            mujoco.mj_step1(m, d)
            _, v, _ = self.state()
            self.drive(0.0, 0.0, v)
            mujoco.mj_step2(m, d)
        mujoco.mj_forward(m, d)
        self.v_inner = None
        self.trace = []
        return self.state()

    # --- the drive ---------------------------------------------------------------
    def tool_path(self, ell):
        """The tool's servo target and its path's acceleration at the current
        time, for the commanded reach ell: a new command starts a path from
        the current reach; along it, the target is the path plus the servo's
        own dynamics, (kv l' + m l'') / kp, so that the servo follows it."""
        d, ds = self.data, self.design
        t = float(d.time)
        if self.tool is None or self.tool[2] != ell:
            self.tool = (t, float(d.qpos[self.jq["reach"]]), ell)
        t0, l0, l1 = self.tool
        s = min(1.0, max(0.0, (t - t0) / TOOL_TIME))
        dl = l1 - l0
        pos = l0 + dl * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)
        vel = dl * (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / TOOL_TIME if s < 1 else 0.0
        acc = dl * (60 * s - 180 * s ** 2 + 120 * s ** 3) / TOOL_TIME ** 2 if s < 1 else 0.0
        target = pos + (ds.tool_kv * vel + ds.tool_mass * acc) / ds.tool_kp
        return min(ds.reach, max(0.0, target)), acc

    def tool_force(self, target):
        """The force of the tool's position servo at the current state for the
        target, as the actuator computes it before its limit, then limited."""
        d, ds = self.data, self.design
        q, qd = d.qpos[self.jq["reach"]], d.qvel[self.jv["reach"]]
        f = ds.tool_kp * (target - q) - ds.tool_kv * qd
        return max(-ds.tool_force, min(ds.tool_force, f))

    def drive(self, u, ell, v):
        """Sets the controls for the commanded acceleration u and reach ell,
        at the cart's current speed v."""
        d, r, h = self.data, self.design.wheel_radius, self.m.opt.timestep
        if self.v_inner is None:
            self.v_inner = v
        self.v_inner = min(v + INNER_CLAMP, max(v - INNER_CLAMP, self.v_inner))
        target, _ = self.tool_path(ell)
        pitch = d.qpos[self.jq["pitch"]]
        force = self.drive_mass * (u - G * math.sin(pitch) + INNER_GAIN * (self.v_inner - v))
        force += self.tool_force(target)
        loads = [float(d.sensor(w + " load").data[0]) for w in ("front", "rear")]
        total = loads[0] + loads[1]
        lim = self.design.drive_limit
        for wheel, load in zip(("front", "rear"), loads):
            share = load / total if total > 0 else 0.5
            tau = force * r * share + self.wheel_inertia * u / r
            d.ctrl[self.act["drive " + wheel]] = max(-lim, min(lim, tau))
        d.ctrl[self.act["tool"]] = target
        self.v_inner += u * h

    # --- one period --------------------------------------------------------------------
    def run_period(self, law, wall=25.0):
        """One period of 0.1 s under 'law'; returns the true state at its end
        and leaves in self.summary what the period did (summarize)."""
        m, d = self.m, self.data
        xs, us, ells, errs = [], [], [], []
        clipped = 0
        for i in range(self.steps):
            tau = i * m.opt.timestep
            mujoco.mj_step1(m, d)               # derived quantities of the current state
            x = self.state()
            meas = self.sensor(*x)
            u, ell = law(tau, meas)
            if abs(u) > U_MAX:
                clipped += 1
                u = math.copysign(U_MAX, u)
            xs.append(x)
            us.append(u)
            ells.append(ell)
            errs.append((meas[0] - x[0], meas[1] - x[1], meas[2] - x[2]))
            if self.record:
                self.trace.append(self.step_record(tau, x, meas, u, ell))
            self.drive(u, ell, x[1])
            mujoco.mj_step2(m, d)
            self.time_steps += 1
        mujoco.mj_forward(m, d)
        end = self.state()
        xs.append(end)
        self.clipped += clipped
        self.summary = summarize(xs, us, ells, errs, law, wall, int(d.ncon), clipped,
                                 m.opt.timestep)
        return end

    def step_record(self, tau, x, meas, u, ell):
        """What one time step keeps: the time, the true and the measured
        state, the command, the number of contacts and the wheel loads."""
        d = self.data
        return (float(d.time), tau, float(x[0]), float(x[1]), float(x[2]),
                float(meas[0]), float(meas[1]), float(meas[2]), float(u), float(ell),
                int(d.ncon), float(d.sensor("front load").data[0]),
                float(d.sensor("rear load").data[0]))


TRACE_COLUMNS = ("t", "tau", "p", "v", "l", "p_meas", "v_meas", "l_meas", "u", "ell",
                 "contacts", "front_load", "rear_load")


def summarize(xs, us, ells, errs, law, wall, ncon, clipped, h):
    """What one period did, from the true states at its N + 1 time steps
    (xs, the last one the end), the commands (us, ells) and the sensor's
    errors (errs):
      w      the largest |(v[i+1] - v[i]) / h - u[i]|, and its time step;
      w_sum  the sum over the period of h * |w[i]|, for the tool's impulse;
      w_all  every w[i] (kept for the identification);
      d      the largest |p[i+1] - p[i] - h v[i+1]|;
      n_p, n_v, n_l  the largest sensor errors;
      l_err  the largest |l[i] - ell[i]| over the time steps;
      pl     the largest p + l, at the N + 1 time steps; clear, wall - pl;
      ratio  if the law tracks a segment (law.seg): the largest V(e) / level
             of the error from its reference at the time steps, with the
             level max(s0, s1) and the form law.q (if the law has one);
      contacts at the end; clipped commands."""
    n = len(us)
    ws = [(xs[i + 1][1] - xs[i][1]) / h - us[i] for i in range(n)]
    ds = [abs(xs[i + 1][0] - xs[i][0] - h * xs[i + 1][1]) for i in range(n)]
    iw = max(range(n), key=lambda i: abs(ws[i]))
    pl = max(x[0] + x[2] for x in xs)
    out = dict(w=abs(ws[iw]), w_step=iw, w_sum=sum(h * abs(w) for w in ws), w_all=ws,
               d=max(ds), n_p=max(abs(e[0]) for e in errs), n_v=max(abs(e[1]) for e in errs),
               n_l=max(abs(e[2]) for e in errs),
               l_err=max(abs(xs[i][2] - ells[i]) for i in range(n)),
               pl=pl, clear=wall - pl, contacts=ncon, clipped=clipped, ratio=None)
    seg, q = getattr(law, "seg", None), getattr(law, "q", None)
    if seg is not None and q is not None:
        lv = max(seg.s0, seg.s1)
        worst = 0.0
        for i in range(n + 1):
            pr, vr = seg.ref(i)
            dp, dv = xs[i][0] - pr, xs[i][1] - vr
            worst = max(worst, math.sqrt(max(0.0, q[0] * dp * dp + 2 * q[1] * dp * dv
                                             + q[2] * dv * dv)) / lv)
        out["ratio"] = worst
    return out


def realized(trace, h=0.001):
    """For consecutive time steps of a trace (one period or more, without a
    placement in between): the realized acceleration less the command,
    (v[i+1] - v[i]) / h - u[i], with the index i."""
    out = []
    for i in range(len(trace) - 1):
        out.append(((trace[i + 1][3] - trace[i][3]) / h - trace[i][8], i))
    return out
