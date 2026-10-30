#!/usr/bin/env python3
"""
TD-MPC2, the official implementation, on the governed cart of Chapter 6,
through the reference adapter of Section 6.2.

Physical AI, Together, Chapter 6, Section 6.2.4; the protocol is PROTOCOL.md.

    algorithm       the official code of Hansen, Su and Wang (ICLR 2024),
                    github.com/nicklashansen/tdmpc2 at commit
                    332f19e4aba4bffe2dc736f45ea7f7bdc585903a, imported from
                    a checkout and used unchanged: its agent (tdmpc2.TDMPC2),
                    its replay buffer (common.buffer.Buffer), its config.yaml
                    with the default hyperparameters, read by its own
                    common.parser.parse_cfg. Settings that the protocol
                    needs: model_size=5 (the default for single tasks),
                    episodic=true (the task ends at arrival), obs=state, the
                    seed; and, as the official code sets them for its
                    episodic Gymnasium tasks (envs/mujoco.py), rho=0.7 and
                    discount_max=0.99. The settings that the official
                    make_env derives from an environment are derived from
                    this one in the same way: obs_shape, action_dim,
                    episode_length = 200 periods and seed_steps =
                    max(1000, 5 * 200). Logging, videos and checkpoints of
                    the official code are switched off.
    training loop   the loop of trainer/online_trainer.py: random actions for
                    the first seed_steps steps, then the planner; at step
                    seed_steps, seed_steps updates, then one update per step;
                    each finished episode is added to the buffer. Only its
                    end differs: it stops after a number of episodes, not of
                    steps, and it does not evaluate during training.
    action          TD-MPC2 plans over three steps and executes the first
                    action of its plan, as published: one reference per
                    period (proposal="reference", normalize_actions=True, so
                    its action space is [-1, 1] as in the official tasks).
    training        seeds 1, 2 and 3; seed s trains in the worlds 100 s + i
                    mod 40; budget M is the agent saved after 40 episodes,
                    budget L the agent after 1000 episodes of the same run;
                    after arrival the environment finishes the run with the
                    planner in evaluation mode, outside the buffer
    evaluation      each saved agent, act(eval_mode=True), in the 20 worlds
                    1000 to 1019; torch is seeded with the world's seed
                    before each run, because the planner samples

Usage
    python train_tdmpc2.py --tdmpc2 PATH            # PATH: the checkout (holds tdmpc2/)
    python train_tdmpc2.py --tdmpc2 PATH --seeds 2  # one seed
    python train_tdmpc2.py --tdmpc2 PATH --smoke    # 2 and 3 episodes, 2 worlds, and
                                                    # seed_steps 200: a test of the setup
    python train_tdmpc2.py --tdmpc2 PATH --compile off
        torch.compile, on by default in the official config, is used under
        Linux ("auto"); it needs Triton, which other systems usually lack.
        It changes the speed, not the algorithm.
    python train_tdmpc2.py --tdmpc2 PATH --reward time
        the experiment of PROTOCOL-R2.md: the time reward of cart_env (-1
        minus 0.5 times the intervention in every period before arrival)
        and the discount 0.99, SAC's, set through the official config as
        discount_min = discount_max = 0.99 (the official rule would derive
        0.975 from the 200 periods of an episode); every other setting
        unchanged. The evaluation is the same: it uses no reward.

Needs an NVIDIA GPU: the official agent and buffer use cuda:0. Writes
results/tdmpc2/seed<s>/ (evaluation.csv, training.csv, record.json and the
agents M.pt and L.pt), or results/tdmpc2-time/seed<s>/ with --reward time;
results/smoke/... with --smoke.
"""
import argparse
import dataclasses
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cart_env    # noqa: E402
import protocol    # noqa: E402

METHOD = "TD-MPC2"
PINNED = "332f19e4aba4bffe2dc736f45ea7f7bdc585903a"
PACKAGES = ["torch", "tensordict", "torchrl", "hydra-core", "omegaconf", "numpy", "gymnasium",
            "termcolor"]


class TensorCart:
    """The governed cart with the interface of TD-MPC2's environments
    (envs/wrappers/tensor.py over envs/mujoco.py): reset() returns the
    observation, step(action) returns (obs, reward, done, info) as torch
    tensors with info['terminated'] and info['success'], and rand_act()
    samples the action space, which is seeded here."""

    def __init__(self, env, seed, torch):
        self.env, self.torch = env, torch
        self.observation_space, self.action_space = env.observation_space, env.action_space
        self.action_space.seed(seed)
        self.max_episode_steps = env.max_episode_steps

    def rand_act(self):
        return self.torch.from_numpy(self.action_space.sample().astype(np.float32))

    def reset(self):
        obs, _ = self.env.reset()
        return self.torch.from_numpy(obs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action.numpy())
        info = defaultdict(float, info)
        info["success"] = float(info["arrived"])
        info["terminated"] = self.torch.tensor(float(terminated))
        return (self.torch.from_numpy(obs), self.torch.tensor(reward, dtype=self.torch.float32),
                terminated or truncated, info)


def main():
    ap = argparse.ArgumentParser(description="TD-MPC2 (official code) on the governed cart")
    ap.add_argument("--tdmpc2", required=True, help="the checkout of github.com/nicklashansen/tdmpc2")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="default: 1, 2 and 3; with --smoke, 1")
    ap.add_argument("--compile", choices=("auto", "on", "off"), default="auto")
    ap.add_argument("--smoke", action="store_true", help="tiny budgets: a test of the setup")
    ap.add_argument("--allow-other-commit", action="store_true")
    ap.add_argument("--reward", choices=("registered", "time"), default="registered",
                    help="registered: PROTOCOL.md (default); time: PROTOCOL-R2.md, with the "
                         "discount 0.99, writes to results/tdmpc2-time/")
    args = ap.parse_args()
    if args.seeds is None:
        args.seeds = [1] if args.smoke else list(protocol.SEEDS)

    checkout = os.path.abspath(args.tdmpc2)
    src = os.path.join(checkout, "tdmpc2")
    commit = protocol.check_commit("TD-MPC2", checkout, PINNED, args.allow_other_commit)
    sys.path.insert(0, src)
    # as tdmpc2/train.py sets them
    os.environ["LAZY_LEGACY_OP"] = "0"
    os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
    import torch
    import hydra.utils
    from omegaconf import OmegaConf
    from tensordict.tensordict import TensorDict
    from common.parser import parse_cfg
    from common.seed import set_seed
    from common.buffer import Buffer
    from tdmpc2 import TDMPC2
    assert torch.cuda.is_available(), "TD-MPC2 needs a CUDA GPU (its agent and buffer use cuda:0)"
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    use_compile = (args.compile == "on" or (args.compile == "auto" and sys.platform == "linux"))

    budgets = dict(protocol.BUDGETS)
    worlds = list(protocol.EVAL_WORLDS)
    folder = "tdmpc2" if args.reward == "registered" else "tdmpc2-time"
    root = os.path.join(protocol.RESULTS, folder)
    if args.smoke:
        budgets, worlds = {"M": 2, "L": 3}, worlds[:2]
        root = os.path.join(protocol.RESULTS, "smoke", folder)

    def make_cfg(seed, out):
        cfg = OmegaConf.load(os.path.join(src, "config.yaml"))
        for k, v in dict(task="cart-governed", obs="state", episodic=True, model_size=5,
                         steps=budgets["L"] * cart_env.N_PRE, seed=seed, compile=use_compile,
                         enable_wandb=False, save_video=False, save_agent=False,
                         save_csv=False, exp_name="physical-ai-together").items():
            cfg[k] = v
        hydra.utils.get_original_cwd = lambda: out     # parse_cfg asks Hydra for it
        cfg = parse_cfg(cfg)
        cfg.discount_max, cfg.rho = 0.99, 0.7          # as envs/mujoco.py (episodic tasks)
        if args.reward == "time":                      # PROTOCOL-R2.md: the discount 0.99
            cfg.discount_min = 0.99
        cfg.obs_shape = {"state": (len(cart_env.OBS_NAMES),)}    # as envs/__init__.make_env
        cfg.action_dim = 1
        cfg.episode_length = cart_env.N_PRE
        cfg.seed_steps = max(1000, 5 * cfg.episode_length)
        if args.smoke:
            cfg.seed_steps = cfg.episode_length    # so that the updates run in a short test
        return cfg

    def to_td(env, obs, action=None, reward=None, terminated=None):
        """As OnlineTrainer.to_td (trainer/online_trainer.py)."""
        obs = obs.unsqueeze(0).cpu()
        if action is None:
            action = torch.full_like(env.rand_act(), float("nan"))
        if reward is None:
            reward = torch.tensor(float("nan"))
        if terminated is None:
            terminated = torch.tensor(float("nan"))
        return TensorDict(obs=obs, action=action.unsqueeze(0), reward=reward.unsqueeze(0),
                          terminated=terminated.unsqueeze(0), batch_size=(1,))

    for seed in args.seeds:
        out = os.path.join(root, f"seed{seed}")
        os.makedirs(out, exist_ok=True)
        for f in ("evaluation.csv", "training.csv"):
            if os.path.exists(os.path.join(out, f)):
                os.remove(os.path.join(out, f))
        cfg = make_cfg(seed, out)
        set_seed(cfg.seed)
        env = cart_env.CartGovernedEnv(proposal="reference", normalize_actions=True,
                                       after_arrival="terminate", reward=args.reward,
                                       worlds=protocol.training_worlds(seed, budgets["L"]))
        tenv = TensorCart(env, seed, torch)
        agent = TDMPC2(cfg)
        buffer = Buffer(cfg)
        if args.reward == "time":
            assert abs(float(agent.discount) - 0.99) < 1e-9, agent.discount

        def planner(obs, info):
            torch.compiler.cudagraph_mark_step_begin()
            return agent.act(torch.from_numpy(obs), t0=False, eval_mode=True).numpy()
        env.post_arrival_policy = planner

        # the loop of OnlineTrainer.train, ended by the number of episodes
        t0 = time.time()
        step, episodes, n_updates = 0, 0, 0
        while episodes < budgets["L"]:
            obs = tenv.reset()
            tds = [to_td(tenv, obs)]
            done = False
            while not done:
                if step > cfg.seed_steps:
                    action = agent.act(obs, t0=len(tds) == 1)
                else:
                    action = tenv.rand_act()
                obs, reward, done, info = tenv.step(action)
                tds.append(to_td(tenv, obs, action, reward, info["terminated"]))
                if step >= cfg.seed_steps:
                    num_updates = cfg.seed_steps if step == cfg.seed_steps else 1
                    for _ in range(num_updates):
                        agent.update(buffer)
                    n_updates += num_updates
                step += 1
            buffer.add(torch.cat(tds))
            episodes += 1
            if episodes == budgets["M"]:
                agent.save(os.path.join(out, "M.pt"))
        agent.save(os.path.join(out, "L.pt"))
        t_train = time.time() - t0
        assert len(env.episodes) == budgets["L"]
        for i, m in enumerate(env.episodes):
            m["episode"] = i
        protocol.write_training(os.path.join(out, "training.csv"), METHOD, "tube", seed,
                                env.episodes)

        evals, latency, t_eval = {}, {}, {}
        for budget in ("M", "L"):
            t1 = time.time()
            agent.load(os.path.join(out, f"{budget}.pt"))
            ev_env = cart_env.CartGovernedEnv(proposal="reference", normalize_actions=True)
            runs, seconds = [], []
            for w in worlds:
                torch.manual_seed(w)
                state = {"t": 0}

                def act(obs, info):
                    torch.compiler.cudagraph_mark_step_begin()
                    a = agent.act(torch.from_numpy(obs), t0=state["t"] == 0, eval_mode=True)
                    state["t"] += 1
                    return a.numpy()
                timed = protocol.Timed(act)
                runs.append(protocol.run_episode(ev_env, timed, w))
                seconds += timed.seconds
            evals[budget] = runs
            latency[budget] = protocol.latency_summary(seconds)
            protocol.write_evaluation(os.path.join(out, "evaluation.csv"), METHOD, "tube", seed,
                                      budget, runs)
            t_eval[budget] = round(time.time() - t1, 1)
        record = dict(method=METHOD, seed=seed, budgets=budgets, smoke=args.smoke,
                      reward=args.reward, protocol=("PROTOCOL.md" if args.reward == "registered"
                                                    else "PROTOCOL-R2.md"),
                      evaluation_worlds=worlds, training_worlds="100 s + i mod 40",
                      environment_steps=step, updates=n_updates, compile=use_compile,
                      discount=float(agent.discount), parameters=agent.model.total_params,
                      config=dataclasses.asdict(cfg),
                      wall_clock_s=dict(training=round(t_train, 1), evaluation=t_eval),
                      inference_latency=latency, checkout=commit, pinned_commit=PINNED,
                      environment=protocol.environment_record(PACKAGES))
        protocol.write_record(os.path.join(out, "record.json"), record)
        for budget in ("M", "L"):
            d = protocol.summarize([evals[budget]], [env.episodes[:budgets[budget]]])
            print(f"[TD-MPC2, {folder}, seed {seed}, budget {budget}] " +
                  "; ".join(f"{k} = {d[k]}" for k in ("arrived", "arrival", "arrival_range",
                                                      "mean_intervention", "alarms",
                                                      "largest_position_all")))
        print(f"[TD-MPC2, {folder}, seed {seed}] {step} environment steps, {n_updates} updates; "
              f"training {t_train:.0f} s; discount {float(agent.discount):g}")


if __name__ == "__main__":
    main()
