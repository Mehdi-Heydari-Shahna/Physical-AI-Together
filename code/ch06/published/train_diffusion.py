#!/usr/bin/env python3
"""
Diffusion Policy, the authors' implementation, on the governed cart of
Chapter 6, through the chunk adapter of Section 6.2, trained on the expert
planner's demonstrations with one round of dataset aggregation.

Physical AI, Together, Chapter 6, Section 6.2.4; the protocol is PROTOCOL.md.

    algorithm       the low-dimensional U-Net policy of Chi et al. (RSS 2023),
                    github.com/real-stanford/diffusion_policy at commit
                    5ba07ac6661db573af695b419a7947ecb704690f, imported from a
                    checkout and used unchanged:
                    DiffusionUnetLowdimPolicy with ConditionalUnet1D, the
                    DDPM scheduler of diffusers, LinearNormalizer, EMAModel
                    and the learning-rate schedule get_scheduler. Every
                    setting is that of the official configuration
                    train_diffusion_unet_lowdim_workspace.yaml (U-Net
                    256-512-1024, kernel 5, FiLM scale; 100 DDPM steps,
                    squared-cosine schedule, epsilon prediction; 100
                    inference steps; observation as global condition; EMA
                    power 0.75; AdamW, learning rate 1e-4, cosine schedule
                    with 500 warm-up steps; batch 256; 5000 epochs), except
                    what the protocol fixes: T_o = 1, T_p = 8 (horizon) and
                    T_a = 4, as for the chunk imitator, and the seed.
    adapter         proposal="sequence": the chunk of eight references in
                    m/s^2; protocol.ChunkPolicy proposes its remainder,
                    padded with its last reference, and infers a new chunk
                    after four periods or after an intervention, as the
                    chunk imitator does. Observation: the vector of
                    cart_env (T_o = 1). Action labels: the expert's chunk at
                    each recorded state (the demonstrated action, not the
                    executed one, Section 6.3.3).
    data            seed s: the expert planner of cart_learning.py acts
                    through the governor in the worlds 100 s + 50, ..., + 54;
                    the policy trained on these demonstrations acts in the
                    worlds 100 s + 55, ..., + 59, and the expert labels the
                    states it visits (one round of dataset aggregation); the
                    final policy is trained anew on all records, every
                    period before arrival being one record
    evaluation      the final policy (its EMA weights) in the 20 worlds 1000
                    to 1019; torch is seeded with the world's seed before
                    each run, because the sampler draws noise. The official
                    workspace keeps the checkpoints with the best rollout
                    scores; here no evaluation world is seen before the
                    policy is frozen.

Usage
    python train_diffusion.py --diffusion-policy PATH        # PATH: the checkout
    python train_diffusion.py --diffusion-policy PATH --seeds 2
    python train_diffusion.py --diffusion-policy PATH --smoke   # 1 + 1 episodes,
                                                     # 5 epochs, 2 worlds: a test

Needs a CUDA GPU by default (--device cpu works, slowly). Writes
results/diffusion/seed<s>/ (evaluation.csv, training.csv, record.json,
policy.pt); results/smoke/diffusion/ with --smoke.
"""
import argparse
import copy
import os
import random
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cart_env    # noqa: E402
import protocol    # noqa: E402

cl = cart_env.cl
METHOD = "diffusion policy"
PINNED = "5ba07ac6661db573af695b419a7947ecb704690f"
PACKAGES = ["torch", "diffusers", "einops", "zarr", "numpy", "huggingface-hub", "gymnasium"]
# train_diffusion_unet_lowdim_workspace.yaml, with T_o, T_p and T_a of the protocol
CONFIG = dict(horizon=cl.H, n_obs_steps=1, n_action_steps=4, num_inference_steps=100,
              obs_as_global_cond=True, oa_step_convention=True,
              unet=dict(diffusion_step_embed_dim=256, down_dims=[256, 512, 1024], kernel_size=5,
                        n_groups=8, cond_predict_scale=True),
              ddpm=dict(num_train_timesteps=100, beta_start=0.0001, beta_end=0.02,
                        beta_schedule="squaredcos_cap_v2", variance_type="fixed_small",
                        clip_sample=True, prediction_type="epsilon"),
              ema=dict(update_after_step=0, inv_gamma=1.0, power=0.75, min_value=0.0,
                       max_value=0.9999),
              optimizer=dict(lr=1.0e-4, betas=(0.95, 0.999), eps=1.0e-8, weight_decay=1.0e-6),
              lr_scheduler="cosine", lr_warmup_steps=500, num_epochs=5000, batch_size=256)


def collect(env, act, label, world, data):
    """One episode in which act(obs, info) proposes and label(obs dict)
    gives the expert's chunk; every period before arrival adds a record
    (observation vector, expert chunk). Returns the metrics and the
    committed ambitions of the recorded periods."""
    obs, info = env.reset(options={"world": world})
    ambitions = []
    while True:
        chunk = [float(a) for a in label(info["obs"])]
        record = (obs.copy(), chunk)
        obs, _, terminated, truncated, info = env.step(act(obs, info))
        if not info["row"]["after"]:
            data.append(record)
            ambitions.append(info["row"]["alpha"])
        if terminated or truncated:
            return info["metrics"], ambitions


def main():
    ap = argparse.ArgumentParser(description="Diffusion Policy (authors' code) on the governed cart")
    ap.add_argument("--diffusion-policy", required=True, dest="dp",
                    help="the checkout of github.com/real-stanford/diffusion_policy")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="default: 1, 2 and 3; with --smoke, 1")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--smoke", action="store_true", help="tiny budgets: a test of the setup")
    ap.add_argument("--allow-other-commit", action="store_true")
    args = ap.parse_args()
    if args.seeds is None:
        args.seeds = [1] if args.smoke else list(protocol.SEEDS)

    checkout = os.path.abspath(args.dp)
    commit = protocol.check_commit("diffusion_policy", checkout, PINNED, args.allow_other_commit)
    sys.path.insert(0, checkout)
    import torch
    from torch.utils.data import DataLoader, Dataset
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
    from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
    from diffusion_policy.model.common.normalizer import LinearNormalizer
    from diffusion_policy.model.diffusion.ema_model import EMAModel
    from diffusion_policy.model.common.lr_scheduler import get_scheduler
    from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to

    device = torch.device(args.device)
    c = dict(CONFIG)
    demos = protocol.DEMOS
    worlds = list(protocol.EVAL_WORLDS)
    root = os.path.join(protocol.RESULTS, "diffusion")
    if args.smoke:
        c["num_epochs"], demos, worlds = 5, 1, worlds[:2]
        root = os.path.join(protocol.RESULTS, "smoke", "diffusion")
    obs_dim = len(cart_env.OBS_NAMES)

    class Records(Dataset):
        def __init__(self, data):
            self.obs = torch.tensor(np.stack([o for o, _ in data])[:, None, :], dtype=torch.float32)
            self.act = torch.tensor(np.array([a for _, a in data])[:, :, None], dtype=torch.float32)

        def __len__(self):
            return len(self.obs)

        def __getitem__(self, i):
            return {"obs": self.obs[i], "action": self.act[i]}

    def train_policy(data, seed):
        """The training of TrainDiffusionUnetLowdimWorkspace, with the
        settings of CONFIG; returns the EMA policy, the mean loss of the last
        epoch and the number of gradient steps."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        model = ConditionalUnet1D(input_dim=1, local_cond_dim=None,
                                  global_cond_dim=obs_dim * c["n_obs_steps"], **c["unet"])
        policy = DiffusionUnetLowdimPolicy(
            model=model, noise_scheduler=DDPMScheduler(**c["ddpm"]), horizon=c["horizon"],
            obs_dim=obs_dim, action_dim=1, n_action_steps=c["n_action_steps"],
            n_obs_steps=c["n_obs_steps"], num_inference_steps=c["num_inference_steps"],
            obs_as_local_cond=False, obs_as_global_cond=c["obs_as_global_cond"],
            pred_action_steps_only=False, oa_step_convention=c["oa_step_convention"])
        ema_model = copy.deepcopy(policy)
        optimizer = torch.optim.AdamW(policy.parameters(), **c["optimizer"])
        dataset = Records(data)
        gen = torch.Generator().manual_seed(seed)
        loader = DataLoader(dataset, batch_size=c["batch_size"], shuffle=True, num_workers=0,
                            pin_memory=device.type == "cuda", generator=gen)
        normalizer = LinearNormalizer()
        normalizer.fit(data={"obs": dataset.obs, "action": dataset.act}, last_n_dims=1,
                       mode="limits")
        policy.set_normalizer(normalizer)
        ema_model.set_normalizer(normalizer)
        lr_scheduler = get_scheduler(c["lr_scheduler"], optimizer=optimizer,
                                     num_warmup_steps=c["lr_warmup_steps"],
                                     num_training_steps=len(loader) * c["num_epochs"],
                                     last_epoch=-1)
        ema = EMAModel(model=ema_model, **c["ema"])
        policy.to(device)
        ema_model.to(device)
        optimizer_to(optimizer, device)
        steps, last = 0, []
        for epoch in range(c["num_epochs"]):
            losses = []
            for batch in loader:
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                loss = policy.compute_loss(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                ema.step(policy)
                losses.append(loss.item())
                steps += 1
            last = losses
        return ema_model, float(np.mean(last)), steps

    def chunk_policy(ema_model):
        def infer(obs, info):
            x = torch.from_numpy(obs).to(device).reshape(1, 1, obs_dim)
            with torch.no_grad():
                pred = ema_model.predict_action({"obs": x})["action_pred"]
            return pred[0, :, 0].cpu().tolist()
        return protocol.ChunkPolicy(infer, n_act=c["n_action_steps"], horizon=c["horizon"])

    for seed in args.seeds:
        out = os.path.join(root, f"seed{seed}")
        os.makedirs(out, exist_ok=True)
        for f in ("evaluation.csv", "training.csv"):
            if os.path.exists(os.path.join(out, f)):
                os.remove(os.path.join(out, f))
        t0 = time.time()
        env = cart_env.CartGovernedEnv(proposal="sequence")
        teacher = cl.expert()
        teacher.reset(False)
        expert_act = lambda obs, info: teacher.propose(info["obs"])     # noqa: E731
        log, d1, d2, amb = [], [], [], []
        for w in protocol.demonstration_worlds(seed)[:demos]:
            m, a = collect(env, expert_act, teacher.propose, w, d1)
            m.update(phase="demonstration")
            log.append(m)
            amb += a
        pi1, loss1, steps1 = train_policy(d1, seed)
        t_stage1 = time.time() - t0
        acting = chunk_policy(pi1)
        for w in protocol.aggregation_worlds(seed)[:demos]:
            acting.reset()
            torch.manual_seed(w)
            m, _ = collect(env, acting, teacher.propose, w, d2)
            m.update(phase="aggregation")
            log.append(m)
        pi2, loss2, steps2 = train_policy(d1 + d2, seed)
        torch.save(pi2.state_dict(), os.path.join(out, "policy.pt"))
        t_train = time.time() - t0
        protocol.write_training(os.path.join(out, "training.csv"), METHOD, "tube", seed, log)

        t1 = time.time()
        ev_env = cart_env.CartGovernedEnv(proposal="sequence")
        final, evals, seconds = chunk_policy(pi2), [], []
        for w in worlds:
            final.reset()
            torch.manual_seed(w)
            evals.append(protocol.run_episode(ev_env, final, w))
            seconds += final.seconds
        protocol.write_evaluation(os.path.join(out, "evaluation.csv"), METHOD, "tube", seed,
                                  "demonstrations", evals)
        t_eval = time.time() - t1
        record = dict(
            method=METHOD, seed=seed, smoke=args.smoke, config=c, evaluation_worlds=worlds,
            demonstration_worlds=protocol.demonstration_worlds(seed)[:demos],
            aggregation_worlds=protocol.aggregation_worlds(seed)[:demos],
            demonstrated_states=len(d1), aggregated_states=len(d2),
            demonstrated_chunks_not_committed_in_full=f"{sum(1 for a in amb if a < 1)} of {len(amb)}",
            demonstrated_chunks_not_committed_at_all=f"{sum(1 for a in amb if a == 0)} of {len(amb)}",
            gradient_steps=dict(demonstrations=steps1, aggregated=steps2),
            last_epoch_loss=dict(demonstrations=loss1, aggregated=loss2),
            parameters=sum(p.numel() for p in pi2.model.parameters()),
            wall_clock_s=dict(stage1=round(t_stage1, 1), training=round(t_train, 1),
                              evaluation=round(t_eval, 1)),
            inference_latency=protocol.latency_summary(seconds), device=str(device),
            checkout=commit, pinned_commit=PINNED,
            environment=protocol.environment_record(PACKAGES))
        protocol.write_record(os.path.join(out, "record.json"), record)
        d = protocol.summarize([evals], [log])
        print(f"[diffusion policy seed {seed}] " +
              "; ".join(f"{k} = {d[k]}" for k in ("arrived", "arrival", "arrival_range",
                                                  "mean_intervention", "alarms",
                                                  "largest_position_all")))
        print(f"[diffusion policy seed {seed}] {len(d1)} + {len(d2)} records; "
              f"{steps1} + {steps2} gradient steps; training {t_train:.0f} s")


if __name__ == "__main__":
    main()
