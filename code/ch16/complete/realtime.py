"""The complete case on the wall clock: the plant and the governor in two
processes (Section 16.4).

The plant process advances on its own clock and never waits for the
governor. At each period boundary t_k = t_0 + k T (time.monotonic) it
  1. takes the messages that have arrived: initial sets and plans committed
     for coming periods, and restart plans;
  2. fixes what period k executes: the plan committed for period k if it
     arrived before t_k, else the shift of the plan in force (the fallback,
     which needs no new decision); before either, it checks the measurement
     at t_k against the initial set on which that plan was certified (the
     set sent with it, or, for the fallback, the end of the executed funnel)
     and, on an alarm, applies the rest feedback instead; after an alarm it
     applies the rest feedback until a restart plan arrives in time and the
     measurement passes the check against the set on which that plan was
     certified, which the governor sends with it;
  3. publishes the measurement at t_k with what period k executes;
  4. simulates the period (about 4 ms of computation) and sleeps until
     t_{k+1}.
The governor process receives the measurement at t_k and does what the core's
loop (allovi.loop) does within period k, in the same order and with the same
modules, except that the world's step is the plant's: it computes the
initial set of the next plan and sends it at once, decides with an
elapsed-time deadline (governor.decide: evaluation stops while the fallback
can still be committed), and sends the plan, which the plant executes from
t_{k+1} if it arrives in time. A proposal that is not ready by the cutoff
counts as missing (None), and the governor proceeds with recovery
candidates only.

Both processes build the same configuration (config.configure) from the
same world, so that, if no deadline is missed and no evaluation is cut, a
run repeats the offline run of allovi.run exactly.
"""
import gc
import multiprocessing as mp
import threading
import time

import allovi
from allovi import NEG_INF
from allovi.contract import Observation, Requirement, Step
from allovi.governor import (Guard, choose_restart, decide, labelling_problem,
                             profile_problem, requirement_problem)
from allovi.loop import PeriodRecord, RunRecord, within_bound

from . import config
from . import model as md

MARGIN_S = 0.005        # the plan is sent at least 5 ms before the boundary
SELECT_S = 0.001        # kept for selection and sending after the last evaluation
ALLOWANCE_S = 0.010     # the time allowed for one evaluation
CUTOFF_S = 0.030        # a proposal must be ready 30 ms after the boundary
START_S = 0.5           # the clock starts 0.5 s after the initial plan is sent


def _sleep_until(t):
    while True:
        d = t - time.monotonic()
        if d <= 0:
            return
        time.sleep(min(d, 0.002) if d < 0.004 else d - 0.002)


# ----------------------------------------------------------------------------
# The plant process
# ----------------------------------------------------------------------------
def plant_main(conn, spec):
    gc.disable()
    world, name, package, bounds, offset, periods = spec
    cfg, sim = config.configure(world, name, package, bounds, offset)
    control, robot = cfg.control, cfg.robot
    bound = robot.measurement_bound
    x = sim.reset()
    m = robot.measure(x)
    conn.send(("meas0", m, x))
    _, t0, plan, _initial = conn.recv()
    initials, commits, restarts = {}, {}, {}
    mode, prev = "normal", None
    timing = []
    for k in range(periods + 1):
        _sleep_until(t0 + k * md.T)
        t_b = time.monotonic()
        while conn.poll():
            msg = conn.recv()
            if msg[0] == "initial":
                initials[msg[1]] = msg[2]
            elif msg[0] == "commit":
                commits[msg[1]] = (msg[2], time.monotonic())
            elif msg[0] == "restart":
                restarts[msg[1]] = (msg[2], time.monotonic(), msg[3])
        if k == periods:
            conn.send(("meas", k, m, x, None, False, t_b, time.monotonic()))
            break
        alarm, source = False, "commit"
        if k > 0:
            if mode == "normal":
                if k in commits:
                    plan = commits.pop(k)[0]
                    aset = initials.get(k)
                    source = "commit"
                else:
                    aset = control.initial_set(prev) if not getattr(
                        control, "INITIAL_FROM_MEASUREMENT", False) else \
                        control.initial_set(prev, None, None)
                    plan = prev.shift()
                    source = "fallback"
                alarm = not within_bound(control.distance(m, aset), bound)
                if alarm:
                    mode, source = "rest", "rest"
            else:
                if k in restarts:
                    rplan, _, rset = restarts.pop(k)
                    if within_bound(control.distance(m, rset), bound):
                        plan = rplan
                        mode, source = "normal", "restart"
                    else:                     # the restart plan's check fails
                        alarm, source = True, "rest"
                else:
                    source = "rest"
        law = control.law(plan) if mode == "normal" else control.rest_law(m)
        t_pub = time.monotonic()
        conn.send(("meas", k, m, x, source, alarm, t_b, t_pub))
        t_s = time.monotonic()
        x = sim.step(x, law)
        m = robot.measure(x)
        t_e = time.monotonic()
        timing.append((k, t_b - (t0 + k * md.T), t_pub - (t0 + k * md.T), t_e - t_s, source))
        if mode == "normal":
            prev = plan
        for kk in [kk for kk in initials if kk <= k]:
            del initials[kk]
    conn.send(("done", sim.periods, robot.errors, timing))
    conn.close()


# ----------------------------------------------------------------------------
# The governor process (the caller's)
# ----------------------------------------------------------------------------
class _Late:
    pass


LATE = _Late()


def _propose(guard, k, proposal, obs, cutoff):
    """The proposal, or LATE if it is not ready by the cutoff (an instant of
    time.monotonic): the method runs in a thread, whose result is discarded
    if it arrives later."""
    box = {}

    def work():
        box["q"] = guard.call(k, "proposal", proposal.propose, obs)
    th = threading.Thread(target=work, daemon=True)
    th.start()
    th.join(max(0.0, cutoff - time.monotonic()))
    if th.is_alive() or "q" not in box:
        return LATE
    return box["q"]


def run(world, name, package, bounds, offset, periods=config.PERIODS, strict=True,
        allowance=ALLOWANCE_S, wrap_proposal=None, wrap_control=None):
    """One run on the wall clock. Returns (record, metrics, timing): the
    core's record, the evaluator's metrics, and the timing of every period
    (governor and plant). wrap_proposal and wrap_control, if given, wrap the
    governor's proposal method and control package (the stress tests of
    timing); the plant always executes the unwrapped package's laws."""
    ctx = mp.get_context("fork")
    here, there = ctx.Pipe()
    spec = (world, name, package, bounds, offset, periods)
    proc = ctx.Process(target=plant_main, args=(there, spec), daemon=True)
    proc.start()
    try:
        return _govern(here, spec, strict, allowance, wrap_proposal, wrap_control)
    finally:
        proc.join(timeout=10)
        if proc.is_alive():
            proc.terminate()


def _govern(conn, spec, strict, allowance, wrap_proposal, wrap_control):
    gc.disable()
    world, name, package, bounds, offset, periods = spec
    c, sim = config.configure(world, name, package, bounds, offset)
    control = wrap_control(c.control) if wrap_control else c.control
    proposal = wrap_proposal(c.proposal) if (wrap_proposal and c.proposal) else c.proposal
    guard = Guard(strict)
    space, levels = control.capabilities, control.levels
    global_labels = space.labelling(space.global_capability)
    no_demand = Requirement(levels.floor, 0)
    bound = c.robot.measurement_bound
    req_check = requirement_problem(levels)
    uses_meas = getattr(control, "INITIAL_FROM_MEASUREMENT", False)

    _, measured, x = conn.recv()
    plan, initial = control.initial_plan(measured, bound)
    profile = control.profile(plan, initial)
    problem = profile_problem(levels)(profile)
    if problem or not levels.admits(profile.level):
        raise allovi.NoCertifiedInitialPlan(f"{c.name}: initial plan {problem or profile.level}")
    t0 = time.monotonic() + START_S
    conn.send(("start", t0, plan, initial))

    rec = RunRecord(c.name)
    mode, restarted, ambition = "normal", False, 0
    pending = {}                  # plans sent for coming periods: k -> (plan, profile, ambition)
    restart_sent = {}
    timing = []
    for k in range(periods):
        msg = conn.recv()
        _, kk, measured, x, source, alarm, t_b, t_pub = msg
        t_recv = time.monotonic()
        assert kk == k
        tk = t0 + k * md.T
        if k > 0:
            if alarm:
                rec.periods[-1].alarm = True
                rec.alarms.append(k)
                mode, plan, profile = "rest feedback", None, None
                restart_sent.pop(k, None)
            elif source == "commit":
                plan, profile, _amb = pending.pop(k)
            elif source == "fallback":
                plan, profile, ambition = plan.shift(), profile.shift(), 0
            elif source == "restart":
                plan, profile, ambition = restart_sent.pop(k)
                mode, restarted = "normal", True
                rec.restarts.append(k)
            if c.capability is not None:
                guard.call(k - 1, "observe", c.capability.observe, k - 1,
                           rec.periods[-1].measured, measured)
        pending = {j: v for j, v in pending.items() if j > k}
        t = k * c.period
        in_force = profile.level if mode == "normal" else NEG_INF
        if c.task is not None:
            c.task.begin_period(k, in_force)
        obs = Observation(k, t, measured, in_force, ambition, mode != "normal")
        defaults = []
        t_q0 = time.monotonic()
        q = None
        late = False
        if proposal is not None:
            q = _propose(guard, k, proposal, obs, tk + CUTOFF_S)
            if q is LATE:
                q, late = None, True
        t_q1 = time.monotonic()
        if q is None:
            defaults.append("proposal")
        req = None
        if c.demand is not None:
            req = guard.call(k, "demand", c.demand.requirement, obs, check=req_check)
        if req is None:
            req = no_demand
            defaults.append("requirement")
        op = c.task.operation(k) if c.task is not None else None
        deadline = tk + md.T - MARGIN_S - SELECT_S
        t_d0 = time.monotonic()
        if mode == "normal":
            labels, src = None, "global"
            if c.capability is not None and not restarted:
                labels = guard.call(k, "labels", c.capability.labels, obs,
                                    check=labelling_problem)
                src = "predictor" if labels is not None else "default"
                if labels is None:
                    defaults.append("labels")
            labels = labels or global_labels
            step = Step(k, q, labels, req, op, measured)
            initial = control.initial_set(plan, measured, bound) if uses_meas else \
                control.initial_set(plan)
            conn.send(("initial", k + 1, initial))
            dec = decide(control, levels, step, plan, profile, initial, None, guard,
                         False, deadline=deadline, allowance=allowance)
            t_d1 = time.monotonic()
            conn.send(("commit", k + 1, dec.plan))
            t_sent = time.monotonic()
            pending[k + 1] = (dec.plan, dec.profile, dec.ambition)
            rec.periods.append(PeriodRecord(k, t, "normal", x, measured, q, req, op, src,
                                            in_force, plan, initial, dec,
                                            defaults=tuple(defaults)))
            ambition = dec.ambition
        else:
            reach, why = guard.call(k, "reach", control.reach, measured, bound,
                                    default=(None, "no set"))
            dec = None
            if reach is not None:
                step = Step(k, q, global_labels, req, op, measured)
                dec = choose_restart(control, levels, step, reach, None, guard, False,
                                     deadline=deadline, allowance=allowance)
            t_d1 = time.monotonic()
            if dec is not None and dec.plan is not None:
                conn.send(("restart", k + 1, dec.plan, reach))
                restart_sent[k + 1] = (dec.plan, dec.profile, dec.ambition)
            t_sent = time.monotonic()
            rec.periods.append(PeriodRecord(k, t, "rest feedback", x, measured, q, req, op,
                                            "global", in_force, None, None, dec, reach, why,
                                            defaults=tuple(defaults)))
            ambition = dec.ambition if (dec is not None and dec.plan is not None) else 0
        timing.append(dict(k=k, recv=t_recv - tk, proposal=t_q1 - t_q0, late=late,
                           decide=t_d1 - t_d0, slack=(tk + md.T) - t_sent,
                           evaluated=None if dec is None else dec.evaluated,
                           timed_out=False if dec is None else dec.timed_out,
                           source=source))
        gc.collect(0)
    msg = conn.recv()
    _, kk, measured, x, _, _, _, _ = msg
    if c.capability is not None:
        guard.call(periods - 1, "observe", c.capability.observe, periods - 1,
                   rec.periods[-1].measured, measured)
    rec.final_state, rec.final_measured = x, measured
    rec.events = guard.events
    _, periods_summary, errors, plant_timing = conn.recv()
    sim.periods = periods_summary
    c.robot.errors = errors
    gc.enable()
    metrics = c.evaluator.evaluate(rec)
    return rec, metrics, dict(governor=timing, plant=plant_timing)
