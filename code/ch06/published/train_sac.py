#!/usr/bin/env python3
"""
Soft actor-critic, as Stable-Baselines3 implements it, on the governed cart
of Chapter 6, through the reference adapter of Section 6.2.

Physical AI, Together, Chapter 6, Section 6.2.4; the protocol is PROTOCOL.md.

    algorithm       stable_baselines3.SAC (Raffin et al., JMLR 2021), version
                    pinned in requirements-sac.txt, with its default
                    hyperparameters (policy "MlpPolicy"); the script sets only
                    the seed and the device, and the discount if --gamma is
                    given
    environment     cart_env.CartGovernedEnv(proposal="reference",
                    normalize_actions=True, after_arrival="terminate"): one
                    reference acceleration per period, the action 1 meaning
                    the declared largest acceleration, 1.875 m/s^2; the
                    episode ends for the learner at arrival, and the
                    environment finishes the run with the current
                    deterministic policy, which never enters the replay
                    buffer
    training        seeds 1, 2 and 3; seed s trains in the worlds 100 s + i
                    mod 40, i = 0, 1, ...; budget M is the snapshot after
                    40 episodes, budget L the policy after 1000 episodes of
                    the same run
    evaluation      each frozen policy (predict(deterministic=True)) in the 20
                    evaluation worlds 1000 to 1019, every run continued
                    until the cart is at rest

Usage
    python train_sac.py                    # seeds 1, 2, 3; budgets M and L
    python train_sac.py --seeds 2          # one seed (seeds can run in parallel)
    python train_sac.py --smoke            # 2 and 3 training episodes, 2 worlds:
                                           # a test of the installation
    python train_sac.py --device cuda      # default: cpu, which is enough
    python train_sac.py --gamma 0.5        # the discount of the tabular stand-in,
                                           # every other setting unchanged
    python train_sac.py --reward time      # the reward of PROTOCOL-R2.md, every
                                           # other setting unchanged

Writes results/sac/seed<s>/ (evaluation.csv, training.csv, record.json and
the two policies, M.zip and L.zip); results/smoke/sac/ with --smoke. Then run
report.py. With --gamma g, the runs go to results/sac-gamma<g>/ instead, so
the registered runs are never overwritten; report.py does not read them, and
../discount_check.py compares them with the stand-in of the same discount.
That comparison is not registered in PROTOCOL.md: it answers a question
raised after SAC's results were known (Section 6.2.4), and it is reported as
exploratory. With --reward time, the learner is trained with the time
reward of cart_env (-1 minus 0.5 times the intervention in every period
before arrival), the experiment of PROTOCOL-R2.md, and the runs go to
results/sac-time/ (results/sac-time-gamma<g>/ with --gamma), which
report_r2.py reads. The evaluation is the same: it uses no reward.
"""
import argparse
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cart_env    # noqa: E402
import protocol    # noqa: E402

METHOD = "SAC"
PACKAGES = ["stable-baselines3", "torch", "gymnasium", "numpy", "cloudpickle"]


def main():
    ap = argparse.ArgumentParser(description="SAC (Stable-Baselines3) on the governed cart")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="default: 1, 2 and 3; with --smoke, 1")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1, help="torch threads (default 1)")
    ap.add_argument("--smoke", action="store_true", help="tiny budgets: a test of the installation")
    ap.add_argument("--gamma", type=float, default=None,
                    help="discount factor (default: that of Stable-Baselines3, 0.99); "
                         "writes to results/sac-gamma<g>/")
    ap.add_argument("--reward", choices=("registered", "time"), default="registered",
                    help="registered: PROTOCOL.md (default); time: PROTOCOL-R2.md, "
                         "writes to results/sac-time/")
    args = ap.parse_args()
    if args.seeds is None:
        args.seeds = [1] if args.smoke else list(protocol.SEEDS)

    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    torch.set_num_threads(args.threads)

    budgets = dict(protocol.BUDGETS)
    worlds = list(protocol.EVAL_WORLDS)
    folder = "sac" if args.reward == "registered" else "sac-time"
    if args.gamma is not None:
        folder += f"-gamma{args.gamma:g}"
    extra = {} if args.gamma is None else dict(gamma=args.gamma)
    root = os.path.join(protocol.RESULTS, folder)
    if args.smoke:
        budgets, worlds = {"M": 2, "L": 3}, worlds[:2]
        root = os.path.join(protocol.RESULTS, "smoke", folder)

    class Budget(BaseCallback):
        """Counts finished training episodes; saves the policy of budget M
        after episode budgets['M'] and stops after episode budgets['L']."""

        def __init__(self, save_m):
            super().__init__()
            self.episodes, self.save_m = 0, save_m

        def _on_step(self):
            if self.locals["dones"][0]:
                self.episodes += 1
                if self.episodes == budgets["M"]:
                    self.model.save(self.save_m)
                if self.episodes >= budgets["L"]:
                    return False
            return True

    for seed in args.seeds:
        out = os.path.join(root, f"seed{seed}")
        os.makedirs(out, exist_ok=True)
        for f in ("evaluation.csv", "training.csv"):
            if os.path.exists(os.path.join(out, f)):
                os.remove(os.path.join(out, f))
        t0 = time.time()
        env = cart_env.CartGovernedEnv(proposal="reference", normalize_actions=True,
                                       after_arrival="terminate", reward=args.reward,
                                       worlds=protocol.training_worlds(seed, budgets["L"]))
        model = SAC("MlpPolicy", env, seed=seed, device=args.device, verbose=0, **extra)
        env.post_arrival_policy = lambda obs, info: model.predict(obs, deterministic=True)[0]
        cb = Budget(os.path.join(out, "M.zip"))
        model.learn(total_timesteps=budgets["L"] * cart_env.N_PRE + 1, callback=cb)
        model.save(os.path.join(out, "L.zip"))
        t_train = time.time() - t0
        episodes = env.episodes
        assert len(episodes) == budgets["L"], (len(episodes), budgets["L"])
        for i, m in enumerate(episodes):
            m["episode"] = i
        protocol.write_training(os.path.join(out, "training.csv"), METHOD, "tube", seed, episodes)
        hyper = dict(learning_rate=model.learning_rate, buffer_size=model.buffer_size,
                     learning_starts=model.learning_starts, batch_size=model.batch_size,
                     tau=model.tau, gamma=model.gamma, train_freq=str(model.train_freq),
                     gradient_steps=model.gradient_steps, ent_coef=str(model.ent_coef),
                     target_entropy=float(model.target_entropy),
                     target_update_interval=model.target_update_interval,
                     use_sde=model.use_sde, net_arch=str(model.policy.net_arch),
                     policy=str(model.policy))
        steps = model.num_timesteps

        evals, latency, t_eval = {}, {}, {}
        for budget in ("M", "L"):
            t1 = time.time()
            frozen = SAC.load(os.path.join(out, f"{budget}.zip"), device=args.device)
            act = protocol.Timed(lambda obs, info: frozen.predict(obs, deterministic=True)[0])
            ev_env = cart_env.CartGovernedEnv(proposal="reference", normalize_actions=True)
            evals[budget] = [protocol.run_episode(ev_env, act, w) for w in worlds]
            latency[budget] = protocol.latency_summary(act.seconds)
            protocol.write_evaluation(os.path.join(out, "evaluation.csv"), METHOD, "tube", seed,
                                      budget, evals[budget])
            t_eval[budget] = round(time.time() - t1, 1)
        record = dict(method=METHOD, seed=seed, budgets=budgets, smoke=args.smoke,
                      reward=args.reward, protocol=("PROTOCOL.md" if args.reward == "registered"
                                                    else "PROTOCOL-R2.md"),
                      evaluation_worlds=worlds, training_worlds="100 s + i mod 40",
                      environment_steps=steps, hyperparameters=hyper, device=args.device,
                      torch_threads=args.threads, wall_clock_s=dict(training=round(t_train, 1),
                                                                    evaluation=t_eval),
                      inference_latency=latency,
                      environment=protocol.environment_record(PACKAGES))
        protocol.write_record(os.path.join(out, "record.json"), record)
        for budget in ("M", "L"):
            n = budgets[budget]
            d = protocol.summarize([evals[budget]], [episodes[:n]])
            print(f"[SAC, {folder}, seed {seed}, budget {budget}] " +
                  "; ".join(f"{k} = {d[k]}" for k in ("arrived", "arrival", "arrival_range",
                                                      "mean_intervention", "alarms",
                                                      "largest_position_all")))
        print(f"[SAC, {folder}, seed {seed}] {steps} environment steps; training {t_train:.0f} s")


if __name__ == "__main__":
    main()
