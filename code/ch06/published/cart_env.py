#!/usr/bin/env python3
"""
The governed cart of Chapter 6 as an environment with the Gymnasium API.

Physical AI, Together, Chapter 6, Section 6.2 (the published algorithms).

One episode is one run of the protocol of Section 6.2 in one world: the
world drawn from its seed by cart_learning.World, the tube package of
Chapter 5 (or the contraction-metric package), the forecasting predictor,
the governor, the reserve floor, the task (reach 24 m within 20 s) and the
period of 0.1 s, all imported from ../cart_learning.py and ../../ch05 and
used unchanged. In every step the governor answers the proposal: it commits
the most ambitious certified candidate or the fallback. Nothing here
certifies anything; the environment only calls the book's governor.

    reset(seed=None, options=None) -> (obs, info)
        The world is options["world"] (an integer seed, or "ch5" for the
        world of Chapter 5), or else the next world of the schedule given to
        the constructor. The argument seed seeds self.np_random, as the
        Gymnasium API asks, and restarts the schedule at its first world, so
        that reset(seed=s) is deterministic; it never selects a world by its
        value, because libraries such as Stable-Baselines3 pass their own
        seed to the first reset.
    step(action) -> (obs, reward, terminated, truncated, info)
        The action is the proposal q_k, formed from the observation at t_k;
        the plan committed at t_{k+1} answers it.

Proposals (the adapters of Section 6.2, in ../cart_learning.py)
    proposal="reference"   one reference acceleration for period k+1, in
                           m/s^2 (shape (1,)); the generator blends it with
                           the head of the fallback on the ambition grid
                           (SAC, TD-MPC2, the off-policy learner, the rule
                           and the braking curve)
    proposal="sequence"    eight reference accelerations (shape (8,)); the
                           generator offers prefixes of 8, 6, 4 and 2
                           periods with ambitions 1, 3/4, 1/2 and 1/4 (the
                           diffusion policy, the planners, the chunk
                           imitator). The timing of a chunk (T_o = 1,
                           T_p = 8, T_a = 4, a new chunk after an
                           intervention) belongs to the proposal method:
                           protocol.ChunkPolicy implements it as the chunk
                           imitator does.
    None                   no proposal before the deadline: recovery
                           candidates only.
    Floats and numpy arrays are agent actions: with normalize_actions=True
    they are first multiplied by the declared largest acceleration (1.875
    m/s^2 for the tube package), and they are then resolved to 0.001 m/s^2
    and made exact rationals, as the braking curve and the planners of
    cart_learning.py resolve their proposals. A non-finite value is a
    missing proposal. Fractions and integers are taken as exact proposals in
    m/s^2 (multiplied by the declared largest acceleration if
    normalize_actions=True), so the book's stand-ins enter unchanged. The
    environment does not clip: the generator applies the package's limits,
    as it does for every proposal method.

The observation o_k (Table 6.1) and one feature, as float32, divided by
fixed scales: position / 25 m, velocity / 10 m/s, label summary / 30 m (1
if none within reach), a flag for none within reach, level of the plan in
execution / 25 m, the ambition alpha_{k-1}, the alarm flag, the feature
v_env / 10 m/s of Section 6.1.2, and a flag for being on the low-braking
region (label summary 0). info["obs"] holds the same observation as the
dictionary that the stand-ins of cart_learning.py read.

Reward of the proposal q_k
    reward="registered" (the default; PROTOCOL.md) as the off-policy
        learner's: the executed reference acceleration of period k+1
        divided by the declared largest acceleration, minus 0.5 times the
        intervention 1 - alpha_k; 0 for a rest head and after arrival.
    reward="time" (PROTOCOL-R2.md) -1 minus 0.5 times the intervention in
        every period before arrival, 0 after arrival: every period before
        arrival costs the same, so the return is larger the earlier the
        cart arrives and the less the governor intervenes.
    The reward changes nothing else: the governor, the plans committed and
    the metrics are the same for the same proposals.

Episode end
    after_arrival="continue"  (evaluation, and the stand-ins' drivers) the
        run continues after arrival until the cart is at rest, as every run
        of the protocol does; terminated at rest or at an alarm; truncated
        if the cart has not arrived by 20 s, or is not at rest by 25 s.
        info["arrived"] says whether the next proposal is formed after
        arrival; info["row"]["after"] whether the period just run was.
    after_arrival="terminate" (training with libraries that reset at the
        end of an episode) terminated at arrival or at an alarm, truncated
        if not arrived by 20 s. If post_arrival_policy (a callable
        (obs, info) -> action) is set, the environment first finishes the
        run with it, without returning to the learner, so that nothing after
        arrival enters the learner's records and the episode's metrics still
        cover the whole run. Its first call receives the observation and the
        info of the arrival step; the run ends as in mode "continue".
    At the end of every episode, info["metrics"] holds the metrics of
    cart_learning.run_metrics, and the episode is appended to self.episodes.

Gymnasium is used if it is installed; otherwise a small local stand-in for
gymnasium.Env and gymnasium.spaces.Box keeps the module usable (numpy is
required). Module state: the book's scripts keep the installed package and
world in module variables, so every reset and step installs its own; several
environments may be used in one process, one step at a time.
"""
import math
import os
import random
import sys
from fractions import Fraction as Fr

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import cart_learning as cl    # noqa: E402  (imports ../../ch05 itself)

cf = cl.cf

try:                                                     # pragma: no cover
    import gymnasium as _gym
    from gymnasium import spaces as _spaces
    GYMNASIUM = _gym.__version__
    _Env = _gym.Env
    Box = _spaces.Box
except ImportError:
    GYMNASIUM = None

    class Box:
        """A minimal stand-in for gymnasium.spaces.Box: low, high, shape,
        dtype, seed(), sample() and contains()."""

        def __init__(self, low, high, shape=None, dtype=np.float32, seed=None):
            self.dtype = np.dtype(dtype)
            if shape is None:
                shape = np.shape(low)
            self.shape = tuple(shape)
            self.low = np.broadcast_to(np.asarray(low, dtype=self.dtype), self.shape).copy()
            self.high = np.broadcast_to(np.asarray(high, dtype=self.dtype), self.shape).copy()
            self._rng = None
            if seed is not None:
                self.seed(seed)

        def seed(self, seed=None):
            self._rng = np.random.default_rng(seed)
            return [seed]

        def sample(self):
            if self._rng is None:
                self.seed()
            lo = np.where(np.isfinite(self.low), self.low, -1.0)
            hi = np.where(np.isfinite(self.high), self.high, 1.0)
            x = self._rng.uniform(lo, hi)
            free = ~(np.isfinite(self.low) & np.isfinite(self.high))
            x[free] = self._rng.normal(size=int(free.sum()))
            return x.astype(self.dtype)

        def contains(self, x):
            x = np.asarray(x)
            return (x.shape == self.shape and bool(np.all(x >= self.low))
                    and bool(np.all(x <= self.high)))

        __contains__ = contains

        def __repr__(self):
            return f"Box({self.low}, {self.high}, {self.shape}, {self.dtype})"

    class _Env:
        """A minimal stand-in for gymnasium.Env."""

        metadata = {"render_modes": []}
        render_mode = None
        _np_random = None

        @property
        def np_random(self):
            if self._np_random is None:
                self._np_random = np.random.default_rng()
            return self._np_random

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                self._np_random = np.random.default_rng(seed)

        def close(self):
            pass


T = cf.T                         # the period, 0.1 s
H = cl.H                         # length of a sequence or chunk: 8 periods
N_PRE = int(cl.T_LIMIT / T)      # 200 periods: the task must be done by 20 s
N_MAX = N_PRE + cl.POST          # 250 periods: at rest by 25 s at the latest
CHI = 0.5                        # weight of the intervention in the reward
SCALES = dict(p=25.0, v=10.0, d=float(cl.LOOK), level=25.0)
OBS_NAMES = ("p/25", "v/10", "label_summary/30", "none_in_reach", "level/25",
             "alpha_prev", "alarm", "v_env/10", "on_low_braking")


def activate(package):
    """Installs a control package of Chapter 5 and reads its declared
    constants. It is called before every period, because the book's scripts
    keep the installed package in module variables; it costs microseconds."""
    cl.use_package(package)


def observation_vector(obs):
    """The observation dictionary as the float32 vector of the environment."""
    d = obs["d_low"]
    lev = obs["level"]
    return np.array([
        obs["p"] / SCALES["p"],
        obs["v"] / SCALES["v"],
        1.0 if d is None else d / SCALES["d"],
        1.0 if d is None else 0.0,
        0.0 if lev is None else float(lev) / SCALES["level"],
        float(obs["alpha_prev"]),
        1.0 if obs["alarm"] else 0.0,
        cl.v_env(obs["p"], d) / SCALES["v"],
        1.0 if d == 0 else 0.0,
    ], dtype=np.float32)


class CartGovernedEnv(_Env):
    """One world of the protocol of Section 6.2 per episode; see the module
    docstring."""

    metadata = {"render_modes": []}
    render_mode = None

    def __init__(self, proposal="reference", package="tube", worlds=None,
                 after_arrival="continue", post_arrival_policy=None, miss=0.0,
                 normalize_actions=False, log_episodes=True, reward="registered"):
        assert proposal in ("reference", "sequence"), proposal
        assert package in ("tube", "ccm"), package
        assert after_arrival in ("continue", "terminate"), after_arrival
        assert reward in ("registered", "time"), reward
        self.proposal, self.package = proposal, package
        self.reward_name = reward
        self.worlds = None if worlds is None else list(worlds)
        self._next_world = 0
        self.after_arrival = after_arrival
        self.post_arrival_policy = post_arrival_policy
        self.miss = miss
        self.normalize_actions = normalize_actions
        self.log_episodes = log_episodes
        self.episodes = []                        # metrics of finished episodes
        activate(package)
        self.declared = {k: float(v) for k, v in cl.DECLARED.items()}
        self._a_max = cl.DECLARED["a_max"]
        n = 1 if proposal == "reference" else H
        hi = 1.0 if normalize_actions else self.declared["a_max"]
        self.action_space = Box(low=-hi, high=hi, shape=(n,), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(len(OBS_NAMES),),
                                     dtype=np.float32)
        self.max_episode_steps = N_PRE            # periods before arrival, at most
        self._world = None
        self._ended = True

    # ------------------------------------------------------------------
    def _activate(self):
        activate(self.package)
        self._world.install()

    def _choose_world(self, options):
        if options and "world" in options:
            w = options["world"]
        elif self.worlds:
            w = self.worlds[self._next_world % len(self.worlds)]
            self._next_world += 1
        else:
            raise ValueError("no world: pass options={'world': seed} or a schedule")
        if w == "ch5":
            return cl.World.chapter5(), 0
        return cl.World(int(w)), int(w)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._next_world = 0
        self._world, wseed = self._choose_world(options)
        self.world_seed = self._world.seed
        self._activate()
        self.pred = cf.Predictor("forecasting")
        # missed deadlines: drawn as in cart_learning.evaluate, from 10^6 + seed
        self._miss_rng = random.Random(10 ** 6 + wseed)
        self.p, self.v = 0.0, 0.0
        self.plan = cf.Plan(0, [], cf.Rest(Fr(0), Fr(0)))
        self.prof = self.plan.profile(("rest", Fr(0), Fr(0)))
        self.alpha_prev, self.alarm = Fr(1), False
        self.rows, self.arrived, self.k = [], None, 0
        self._ended = False
        obs = self._observe()
        return observation_vector(obs), self._info(obs)

    def _observe(self):
        """The observation formed at t_k, exactly as cart_learning.episode
        forms it, and the initial set I_{k+1} that the test will use."""
        k = self.k
        self.seg_ex = self.plan.head()
        self.init = cl.initial_set(self.seg_ex, self.p, self.v)
        seg = self.seg_ex
        self.obs = dict(k=k, t=float(k * T), p=self.p, v=self.v, level=self.prof[0],
                        alpha_prev=self.alpha_prev, alarm=self.alarm,
                        d_low=cl.low_braking_distance(self.pred, k, self.p),
                        kind_now=seg.kind,
                        a_ex_now=float(seg.a) if seg.kind in ("track", "launch") else None)
        return self.obs

    def _info(self, obs):
        return {"obs": obs, "arrived": self.arrived is not None, "world": self.world_seed}

    # ------------------------------------------------------------------
    def _one(self, x):
        """One requested acceleration as an exact rational, or None."""
        if x is None:
            return None
        if isinstance(x, (Fr, int)) and not isinstance(x, bool):
            q = Fr(x)
            return q * self._a_max if self.normalize_actions else q
        x = float(x)
        if not math.isfinite(x):
            return None
        if self.normalize_actions:
            x *= float(self._a_max)
        return cl.grid(x)

    def to_proposal(self, action):
        """The proposal q_k that an action stands for (see the module
        docstring)."""
        if action is None:
            return None
        if self.proposal == "reference":
            if isinstance(action, (list, tuple, np.ndarray)):
                flat = np.asarray(action, dtype=object).reshape(-1)
                assert len(flat) == 1, f"a reference proposal has one value, not {len(flat)}"
                action = flat[0]
            return self._one(action)
        flat = list(np.asarray(action, dtype=object).reshape(-1))
        seq = [self._one(a) for a in flat]
        if not seq or any(a is None for a in seq):
            return None
        return seq

    def step(self, action):
        if self._ended:
            raise RuntimeError("the episode has ended; call reset()")
        self._activate()
        out = self._period(self.to_proposal(action))
        if out["end"] is None and self.after_arrival == "terminate" and out["row"]["done"]:
            out["end"] = "arrival"
            # the run continues unless the cart is at rest or the time is up,
            # the ends of mode 'continue'
            if (self.post_arrival_policy is not None and out["row"]["v1"] != 0.0
                    and self.k < N_MAX):
                self._finish_run(out["vector"], out["info"])
        if out["end"] is not None:
            self._close_episode(out)
        return out["vector"], out["reward"], out["terminated"], out["truncated"], out["info"]

    def _finish_run(self, vec, info):
        """After arrival in mode 'terminate': the rest of the run with the
        post-arrival policy, which never returns to the learner. Its first
        call receives the observation and the info of the arrival step."""
        while True:
            self._activate()
            out = self._period(self.to_proposal(self.post_arrival_policy(vec, info)),
                               end_mode="continue")
            if out["end"] is not None:
                return
            vec, info = out["vector"], out["info"]

    def _period(self, q, end_mode=None):
        """Period k of cart_learning.episode, for the proposal q."""
        end_mode = end_mode or self.after_arrival
        k, init, seg_ex, plan, prof = self.k, self.init, self.seg_ex, self.plan, self.prof
        if self.miss and self._miss_rng.random() < self.miss:
            q = None
        gen = cl.generate_reference if self.proposal == "reference" else cl.generate_prefixes
        fallback = plan.shift()
        fb_prof = prof[1:] if len(prof) > 1 else prof
        cands = gen(self.pred, k, init, q, fallback)
        profiles = [c.profile(init) for c, _ in cands]
        sel = cf.select(cands, profiles, fb_prof)
        if sel is None:            # cannot happen: the fallback is admitted
            raise AssertionError(f"the governor committed no plan in period {k}")
        if sel == 0:
            new_plan, new_prof, alpha = fallback, fb_prof, Fr(0)
        else:
            new_plan, new_prof = cands[sel - 1][0], profiles[sel - 1]
            alpha = cands[sel - 1][1]
        head = new_plan.head()
        p1, v1, st = cf.simulate(seg_ex, self.p, self.v, float(k * T), self._world.w)
        alarm = not cf.inside(init, p1, v1)
        done = self.arrived is None and p1 >= cl.WORK
        after = self.arrived is not None
        row = dict(k=k, t=float(k * T), p=self.p, v=self.v, d_low=self.obs["d_low"], q=q,
                   alpha=alpha, sel=sel, a_now=getattr(seg_ex, "a", None),
                   a_next=head.a if head.kind in ("track", "launch") else None,
                   kind_next=head.kind, alarm=alarm, out=st["out"], pmax=st["pmax"],
                   p1=p1, v1=v1, done=done, after=after,
                   unchanged=(alpha == 1 and cl.realizable(q, init)
                              if self.proposal == "reference" else None))
        self.rows.append(row)
        a_next = float(row["a_next"]) if row["a_next"] is not None else 0.0
        if after:
            reward = 0.0
        elif self.reward_name == "time":
            reward = -1.0 - CHI * float(1 - alpha)
        else:
            reward = a_next / float(self._a_max) - CHI * float(1 - alpha)
        self.p, self.v = p1, v1
        self.plan, self.prof, self.alpha_prev, self.alarm = new_plan, new_prof, alpha, alarm
        self.k = k + 1
        if done:
            self.arrived = k + 1
        end, terminated, truncated = None, False, False
        if alarm:
            end, terminated = "alarm", True
        elif self.arrived is None and self.k >= N_PRE:
            end, truncated = "not arrived by 20 s", True
        elif end_mode == "continue" and self.arrived is not None and v1 == 0.0:
            end, terminated = "at rest", True
        elif self.k >= N_MAX:
            end, truncated = "not at rest by 25 s", True
        obs = self._observe()
        info = self._info(obs)
        info.update(row=row, alpha=float(alpha), intervention=float(1 - alpha),
                    executed=None if row["a_next"] is None else float(row["a_next"]),
                    committed="fallback" if sel == 0 else "candidate")
        return dict(vector=observation_vector(obs), reward=reward, terminated=terminated,
                    truncated=truncated, info=info, row=row, end=end, obs=obs)

    def _close_episode(self, out):
        self._ended = True
        if out["end"] == "arrival":
            out["terminated"], out["truncated"] = True, False
        m = cl.run_metrics(self.rows, self.arrived)
        m.update(world=self.world_seed, end=out["end"], periods_run=len(self.rows))
        out["info"]["metrics"] = m
        if self.log_episodes:
            self.episodes.append(m)
        cl.restore_world()

    def close(self):
        cl.restore_world()


def make(**kwargs):
    """A CartGovernedEnv; the keyword arguments of its constructor."""
    return CartGovernedEnv(**kwargs)
