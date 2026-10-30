"""The other modules of the complete case (Section 16.4): the simulator and
robot adapters for the MuJoCo cart, the capability predictors, the proposal
methods and the evaluator. The demand predictor and the task are those of
Chapter 10 (cart.task-plan, cart.operation), unchanged.
"""
import math
import random

from allovi import NEG_INF, register

from . import model as md
from .packages import ACTUATION, STATE
from .plant import Plant, Sensor


# ----------------------------------------------------------------------------
# Simulator and robot adapters
# ----------------------------------------------------------------------------
@register
class MuJoCoCart:
    """The world: the cart of Chapter 11 in MuJoCo (plant.Plant), its ramp
    starting at the world's ramp_start. reset() places the cart at rest at
    the start and lets its contacts settle; step(state, law) runs one period
    of the law, which receives the sensor's measurements at every time step,
    and returns the true state. After each period, self.periods holds the
    plant's summary of it (plant.summarize). The oracle methods serve the
    evaluator only."""
    ROLE, NAME, CONTRACT = "world", "complete.mujoco", "1.1"
    STATE, ACTUATION = STATE, ACTUATION

    def __init__(self, design, sensor_seed, sigma=Sensor.SIGMA, record=False, drive_mass=None):
        self.plant = Plant(design, Sensor(sensor_seed, sigma), record, drive_mass)
        self.ramp_start = design.ramp_start
        self.periods = []
        self.last = None

    def reset(self):
        self.last = self.plant.start(0.0)
        self.periods = []
        return self.last

    def step(self, state, law):
        if state != self.last:
            raise RuntimeError("the simulator continues only from the state it returned last")
        self.last = self.plant.run_period(law)
        self.periods.append(self.plant.summary)
        return self.last

    # --- the oracle ---------------------------------------------------------------
    def label_valid(self, label, seg):
        """A label of 'level ground' is valid if the front wheel's contact
        stays before the crest on the segment's region: the bumper at most
        LEAD beyond the crest's start; the global capability is valid by
        assumption (its scope, speeds up to V_C, is checked by the test, its
        model by the identification and by the monitors of the runs)."""
        if label >= md.B_FIRM:
            return seg.f_phi <= self.ramp_start + md.LEAD
        return True


@register
class Robot:
    """The robot: the state (p, v, l) and its measurement at the period
    boundaries, by the sensor of plant.Sensor with its own generator. The
    measurement bound is the norm-bound max(varpi_p + varpi_l, varpi_v * 1 s)
    of the model's bounds (model.bounds)."""
    ROLE, NAME, CONTRACT = "robot", "complete.robot", "1.1"
    STATE, ACTUATION = STATE, ACTUATION

    def __init__(self, seed, bounds, sigma=Sensor.SIGMA):
        self.sensor = Sensor(seed, sigma)
        self.measurement_bound = max(bounds["varpi_p"] + bounds["varpi_l"], bounds["varpi_v"])
        self.errors = []

    def measure(self, state):
        m = self.sensor(*state)
        self.errors.append(tuple(a - b for a, b in zip(m, state)))
        return m


# ----------------------------------------------------------------------------
# Capability predictors
# ----------------------------------------------------------------------------
@register
class Survey:
    """The survey predictor with an offset (Sections 7.2.3 and 7.3): 'level
    ground', 2 m/s^2, for a segment whose region (funnel) has its bumper at
    most LEAD beyond the surveyed start of the ramp less the offset q, and
    the global capability, 0.5 m/s^2, otherwise and beyond the horizon. Its
    labelling map is correct for every segment, every period and every
    candidate exactly when the score (surveyed start) - (true start) is at
    most q."""
    ROLE, NAME, CONTRACT = "capability", "complete.survey", "1.1"
    SPACE, horizon = "complete.braking", md.HORIZON

    def __init__(self, surveyed, offset):
        self.edge = float(surveyed) - float(offset) + md.LEAD

    def labels(self, observation):
        k, edge = observation.k, self.edge

        def labelling(j, seg):
            if j > k + 1 + self.horizon:
                return md.B_GLOBAL
            return md.B_FIRM if seg.f_phi <= edge else md.B_GLOBAL
        return labelling

    def observe(self, k, before, after):
        pass


# ----------------------------------------------------------------------------
# Proposal methods
# ----------------------------------------------------------------------------
@register
class Rule:
    """The naive proposal method of Chapter 3: 'accelerate' until the
    measured position first reaches the work zone, 'hold' from then on."""
    ROLE, NAME, CONTRACT = "proposal", "complete.rule", "1.1"

    def __init__(self):
        self.arrived = False

    def propose(self, observation):
        if observation.measured[0] >= md.WORK_ZONE:
            self.arrived = True
        return "hold" if self.arrived else "accelerate"


@register
class Preview:
    """The preview planner of Section 16.2.2, for the packages of this
    section: a model-based proposal method that reads the task plan. Until
    the measured position first reaches the work zone it proposes
    'accelerate', then 'hold'. Once the operation is announced, it predicts
    the state at the next period boundary from its own last proposal (if the
    governor committed it in full; otherwise as if the cart kept its speed),
    and proposes 'retreat' from the latest period from which the package's
    back-away (the declared retreat, then braking at the declared largest
    acceleration) still brings the reference to rest at the target by the
    forecast onset; it retreats until braking at the declared largest
    acceleration would stop the reference at or before the target, and then
    proposes 'hold'. The target is 25 m less the operation's reserve, the
    declared position error and the reach's error. Once the operation has
    begun, it proposes 'hold'. It reads the package's declared constants
    (packages._Tracking.declared), as a learning researcher could."""
    ROLE, NAME, CONTRACT = "proposal", "complete.preview", "1.1"

    def __init__(self, task, declared, late=0):
        self.task, self.d, self.late = task, declared, late
        self.target = md.WALL - float(md.OPERATION_LEVEL) - declared["e_p"] - declared["e_l"]
        self.arrived, self.retreating, self.last = False, False, None

    def command(self, q, v):
        d = self.d
        if q == "accelerate":
            return d["a_max"]
        if q == "retreat":
            return d["retreat"]
        if v > 0:
            return -min(v / md.T, d["b_firm"])
        if v < 0:
            return min(-v / md.T, d["a_max"])
        return 0.0

    def stop_back(self, p, v):
        """Where braking a backward motion at the declared largest
        acceleration stops the reference, and after how many periods."""
        a, n = self.d["a_max"], 0
        while v < -1e-12:
            step = min(-v / md.T, a)
            p, v = p + v * md.T + step * md.T * md.T / 2, v + step * md.T
            n += 1
        return p, n

    def periods_to_target(self, p):
        if p <= self.target:
            return 0
        r = self.d["retreat"]
        for m in range(1, 300):
            q = p + r * (m * md.T) ** 2 / 2
            w = r * m * md.T
            q, n = self.stop_back(q, w)
            if q <= self.target:
                return m + n
        return 300

    def propose(self, observation):
        k = observation.k
        p, v, ell = observation.measured
        u = self.command(self.last, v) if (self.last is not None and
                                           observation.ambition == 1) else 0.0
        v1 = v + u * md.T
        p1 = p + v * md.T + u * md.T * md.T / 2
        if v > 0 and v1 < 0 or v < 0 and v1 > 0:
            p1, v1 = p - v * v / (2 * u), 0.0
        q = self.decide(k, p, ell, p1, v1)
        self.last = q
        return q

    def decide(self, k, p, ell, p1, v1):
        task = self.task
        if ell > md.ELL / 2 or not task.pending():
            return "hold"
        if not self.arrived:
            if p >= md.WORK_ZONE:
                self.arrived = True
            else:
                return "accelerate"
        if task.announce is None or k < task.announce:
            return "hold"
        at_rest = abs(v1) <= 0.01
        if self.retreating:
            if v1 < -0.01 and self.stop_back(p1, v1)[0] > self.target:
                return "retreat"
            if at_rest and p1 > self.target:
                return "retreat"
            return "hold"
        if at_rest and p1 > self.target:
            if k + 1 + self.periods_to_target(p1) - self.late >= task.predicted:
                self.retreating = True
                return "retreat"
        return "hold"


# ----------------------------------------------------------------------------
# The evaluator
# ----------------------------------------------------------------------------
def windows(moving, n):
    """The windows of the tool's impulse: each run of consecutive periods in
    'moving' (the tool moves), with the period after it, within 0..n-1."""
    out, cur = [], []
    for k in sorted(moving):
        if cur and k != cur[-1] + 1:
            out.append(cur)
            cur = []
        cur.append(k)
    if cur:
        out.append(cur)
    return [set(ev) | ({ev[-1] + 1} if ev[-1] + 1 < n else set()) for ev in out]


@register
class Metrics:
    """What a run did, from the core's record, the world's oracle and the
    plant's summaries of the periods: the task (arrival, time in the work
    zone, when the operation began and by how much late, overloads), the
    interventions, alarms, restarts, the true clearance at every time step,
    the validity of every label issued and of every executed label, and
    every departure from the bounds of the model (bounds)."""
    ROLE, NAME, CONTRACT = "evaluator", "complete.metrics", "1.1"

    def __init__(self, world, task, sim, robot, bounds):
        self.world, self.task, self.sim, self.robot, self.b = world, task, sim, robot, bounds

    def evaluate(self, record):
        w, task, sim, b = self.world, self.task, self.sim, self.b
        states = [x.state for x in record.periods] + [record.final_state]
        arrival = next((i for i, s in enumerate(states) if s[0] >= md.WORK_ZONE), None)
        begun = task.begun
        normal = [x for x in record.periods if x.governor == "normal"]
        until = len(record.periods) if begun is None else begun
        before = [x for x in normal if x.k < (arrival if arrival is not None else until)]
        after = [x for x in normal if arrival is not None and arrival <= x.k < until]
        issued = invalid_issued = invalid_committed = invalid_executed = 0
        first_invalid = None
        for x in record.periods:
            dec = x.decision
            if dec is not None:
                for e in dec.entries:               # evaluated or not, the fallback included
                    for seg, lab in e.plan.pairs:
                        issued += 1
                        if not sim.label_valid(lab, seg):
                            invalid_issued += 1
                            if dec.plan is e.plan:
                                invalid_committed += 1
            if x.governor == "normal" and x.plan.pairs:
                seg, lab = x.plan.pairs[0]
                if not sim.label_valid(lab, seg):
                    invalid_executed += 1
                    first_invalid = x.k if first_invalid is None else first_invalid
        per = sim.periods
        unfold = {x.k for x in normal if x.plan.pairs and x.plan.pairs[0][0].unfold}
        events = windows(unfold, len(per))                 # the tool's impulse
        window = set().union(*events) if events else set()

        def impulse(ev):
            return sum(md.H * max(0.0, abs(x) - b["w_bar"]) for k in ev for x in per[k]["w_all"])
        dep = dict(
            w=sum(1 for k, s in enumerate(per) if k not in window and s["w"] > b["w_bar"]),
            impulse=sum(1 for ev in events if impulse(ev) > b["J"]),
            d=sum(1 for s in per if s["d"] > b["d_bar"]),
            n_p=sum(1 for s in per if s["n_p"] > b["varpi_p"]),
            n_v=sum(1 for s in per if s["n_v"] > b["varpi_v"]),
            n_l=sum(1 for s in per if s["n_l"] > b["varpi_l"]),
            l=sum(1 for k, s in enumerate(per) if k not in unfold and s["l_err"] > b["e_l"]),
            tube=sum(1 for s in per if s["ratio"] is not None and s["ratio"] > 1.0),
            clipped=sum(s["clipped"] for s in per))
        boundary = [e for e in self.robot.errors]
        dep["boundary"] = sum(1 for e in boundary if abs(e[0]) > b["varpi_p"] or
                              abs(e[1]) > b["varpi_v"] or abs(e[2]) > b["varpi_l"])
        ratios = [s["ratio"] for s in per if s["ratio"] is not None]
        least = min(s["clear"] for s in per)
        return dict(
            arrival=arrival, begun=begun, onset=w.onset,
            delay=None if begun is None else begun - w.onset,
            on_time=begun is not None and begun == w.onset,
            overloads=list(task.overloads), dropped=task.dropped,
            in_work_zone=sum(1 for s in states[:-1] if s[0] >= md.WORK_ZONE),
            modified_before_arrival=sum(1 for x in before if x.decision.ambition < 1),
            modified_after_arrival=sum(1 for x in after if x.decision.ambition < 1),
            periods_after_arrival=len(after),
            fallbacks=sum(1 for x in normal if x.decision.selected == 0),
            evaluated=sum(x.decision.evaluated for x in normal),
            alarms=list(record.alarms), restarts=list(record.restarts),
            rest_feedback=sum(1 for x in record.periods if x.governor != "normal"),
            least_clearance=least, violation=least < 0,
            labels_issued=issued, invalid_issued=invalid_issued,
            invalid_committed=invalid_committed, invalid_executed=invalid_executed,
            first_invalid=first_invalid,
            departures=dep, departure_periods=sum(v for k, v in dep.items() if k != "clipped"),
            max_ratio=max(ratios) if ratios else None,
            max_w=max((s["w"] for k, s in enumerate(per) if k not in window), default=0.0),
            final=states[-1])


# ----------------------------------------------------------------------------
# A learned proposal method
# ----------------------------------------------------------------------------
EDGES = (-0.3, -0.1, 0.0, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5)
LOOK = 30                          # reach of the label summary [m]


class Probe:
    """A hypothetical region whose bumper reaches 'f_phi' at most, for
    querying a labelling map (the survey reads only f_phi)."""
    __slots__ = ("f_phi",)

    def __init__(self, x):
        self.f_phi = x


def v_env(p, d_low, dc):
    """A feature, not a certificate: the largest reference speed at p from
    which braking at the declared reference braking, on level ground and,
    from the region that the labels mark with less, at the global
    capability's braking and speed, stops the reference at the floor's rest
    point, 25 m less the floor and the declared errors."""
    x_h = md.WALL - 0.5 - dc["e_p"] - dc["e_l"]
    b1, b2, vc = dc["b_firm"], dc["b_global"], dc["v_c"] - dc["e_v"]
    if d_low is None:
        return math.sqrt(max(0.0, 2 * b1 * (x_h - p)))
    r = p + d_low
    vr = min(vc, math.sqrt(max(0.0, 2 * b2 * (x_h - max(r, p)))))
    if p >= r:
        return vr
    return math.sqrt(vr * vr + 2 * b1 * (r - p))


@register
class Learner:
    """The off-policy learner of Chapter 6 (Section 6.2), for the packages of
    this section: tabular, with replay. Observation features: the margin
    v_env - v in ten bins, whether the labels mark the cart's position with
    less braking than level ground, and whether the previous proposal was
    committed in full. Actions: eight reference accelerations, from the
    package's declared constants. Reward of the proposal of period k: the
    reference acceleration of the head of the plan committed at t_{k+1},
    divided by the declared largest acceleration, less chi times the
    intervention 1 - alpha_k. Hyperparameters as in Chapter 6: chi 0.5,
    discount 0.5, learning rate 0.3, 10 sweeps of replay after each
    episode, exploration 0.2 in training. It reads the labels through the
    capability predictor of its configuration, set as 'survey' (the
    observation may include the labels in force, Section 4.1.5). The task
    of its training is to reach the work zone; it knows nothing of the
    operation."""
    ROLE, NAME, CONTRACT = "proposal", "complete.learner", "1.1"

    def __init__(self, declared, seed=0, q_table=None, chi=0.5, gamma=0.5, lr=0.3,
                 sweeps=10, eps=0.2):
        self.d = declared
        self.rng = random.Random(seed)
        self.chi, self.gamma, self.lr, self.sweeps, self.eps = chi, gamma, lr, sweeps, eps
        self.Q = dict(q_table or {})
        self.buffer, self.trace, self.train, self.survey = [], [], False, None
        d = declared
        self.acts = [-d["b_firm"], -1.0, -d["b_global"], 0.0, 0.5, 1.0, 1.5, d["a_max"]]

    def d_low(self, observation):
        if self.survey is None:
            return None
        lab = self.survey.labels(observation)
        k, p = observation.k, observation.measured[0]
        if lab(k + 1, Probe(p)) < md.B_FIRM:
            return 0
        for dd in range(1, LOOK + 1):
            if lab(k + 1, Probe(p + dd)) < md.B_FIRM:
                return dd
        return None

    def features(self, observation):
        p, v = observation.measured[0], observation.measured[1]
        dl = self.d_low(observation)
        m = v_env(p, dl, self.d) - v
        return (sum(1 for e in EDGES if m >= e), 1 if dl == 0 else 0,
                1 if observation.ambition == 1 else 0)

    def values(self, s):
        if s not in self.Q:
            self.Q[s] = [0.0] * len(self.acts)
        return self.Q[s]

    def greedy(self, s):
        qs = self.values(s)
        best = max(qs)
        return max(i for i, x in enumerate(qs) if x == best)     # ties: faster

    def propose(self, observation):
        s = self.features(observation)
        if self.train and self.rng.random() < self.eps:
            i = int(self.rng.random() * len(self.acts))
        else:
            i = self.greedy(s)
        self.trace.append((observation.k, s, i))
        return self.acts[i]

    def learn(self, record, arrival):
        """After a training episode: the transitions of the periods before
        arrival, from the record of the core, into the replay buffer; then
        the sweeps of replay."""
        dec = {x.k: x.decision for x in record.periods if x.decision is not None}
        steps = [t for t in self.trace if arrival is None or t[0] < arrival]
        for n, (k, s, i) in enumerate(steps):
            d = dec.get(k)
            if d is None:
                continue
            head = d.plan.pairs[0][0] if d.plan.pairs else d.plan.rest
            r = head.a / self.d["a_max"] - self.chi * (1.0 - float(d.ambition))
            done = arrival is not None and k + 1 == arrival
            s2 = steps[n + 1][1] if n + 1 < len(steps) and not done else None
            self.buffer.append((s, i, r, s2, done or s2 is None))
        self.trace = []
        for _ in range(self.sweeps):
            for _j in range(len(self.buffer)):
                s, i, r, s2, done = self.buffer[int(self.rng.random() * len(self.buffer))]
                target = r if done else r + self.gamma * max(self.values(s2))
                qs = self.values(s)
                qs[i] += self.lr * (target - qs[i])

    def table(self):
        return {",".join(str(x) for x in s): v for s, v in self.Q.items()}

    @staticmethod
    def from_table(declared, table):
        q = {tuple(int(x) for x in key.split(",")): list(v) for key, v in table.items()}
        return Learner(declared, q_table=q)
