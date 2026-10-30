#!/usr/bin/env python3
"""
A test of the control flow of train_sac.py, train_tdmpc2.py,
train_diffusion.py, report.py and report_r2.py, with stand-ins for the
libraries.

Physical AI, Together, Chapter 6, Section 6.2.4.

It runs on a machine without PyTorch, Stable-Baselines3, TD-MPC2 or
Diffusion Policy. This test writes small stand-ins for them into
a temporary folder (never into this one, where they would shadow the real
libraries), with the interfaces that the scripts call, as read from the
sources at the pinned versions; the stand-ins do not learn, and their
"policies" are a crude braking curve so that runs arrive. It runs each
training script with --smoke in a temporary copy of code/ch05 and code/ch06
and checks the files it writes, then runs report.py --smoke and
report_r2.py --smoke on them; train_sac.py and train_tdmpc2.py run twice,
with the registered reward and with the time reward of PROTOCOL-R2.md, and
the records must name the reward and, for TD-MPC2, the discount (0.975 and
0.99, derived as the official code derives it). It tests the scripts' own
code: episode counting, the snapshot of budget M, the run after arrival,
the chunk timing, the files, the record and the reports.
It says nothing about the libraries; the smoke runs on the GPU machine
(README.md) test those.

Usage
    python3 test_harness.py          # a few minutes
    python3 test_harness.py --keep   # keeps the temporary tree, to look at its files
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, "..", ".."))

FAKES = {
    "torch/__init__.py": '''
        """A stand-in for the parts of torch that the scripts call."""
        import contextlib, math, pickle
        import numpy as _np
        __version__ = "stand-in"
        float32 = _np.float32
        class _V: cuda = None
        version = _V()
        class Tensor:
            def __init__(self, a, dtype=None):
                self.a = _np.asarray(a, dtype=dtype)
            def numpy(self): return self.a
            def cpu(self): return self
            def to(self, *args, **kwargs): return self
            def unsqueeze(self, d): return Tensor(_np.expand_dims(self.a, d))
            def reshape(self, *s): return Tensor(self.a.reshape(*s))
            def tolist(self): return self.a.tolist()
            def item(self): return float(self.a)
            def backward(self): pass
            def __getitem__(self, i): return Tensor(self.a[i])
            def __len__(self): return len(self.a)
            @property
            def shape(self): return self.a.shape
        def from_numpy(a): return Tensor(a)
        def tensor(x, dtype=None): return Tensor(_np.asarray(getattr(x, "a", x), dtype=dtype))
        def full_like(t, v): return Tensor(_np.full_like(t.a, v, dtype=_np.float32))
        def cat(xs):
            if hasattr(xs[0], "cat_many"): return xs[0].cat_many(xs)
            return Tensor(_np.concatenate([x.a for x in xs]))
        def manual_seed(s): _np.random.seed(int(s) % 2 ** 32)
        def set_num_threads(n): pass
        def set_float32_matmul_precision(p): pass
        def save(obj, path):
            with open(path, "wb") as fh: pickle.dump("stand-in", fh)
        class _Compiler:
            @staticmethod
            def cudagraph_mark_step_begin(): pass
        compiler = _Compiler()
        class cuda:
            @staticmethod
            def is_available(): return True
            @staticmethod
            def device_count(): return 1
            @staticmethod
            def get_device_name(i): return "stand-in GPU"
            @staticmethod
            def manual_seed_all(s): pass
        class backends:
            class cudnn:
                benchmark = False
                @staticmethod
                def version(): return 0
        class device:
            def __init__(self, s): self.s, self.type = s, s.split(":")[0]
            def __str__(self): return self.s
        @contextlib.contextmanager
        def no_grad(): yield
        class Generator:
            def manual_seed(self, s): self.seed = s; return self
        from . import optim, utils
    ''',
    "torch/optim.py": '''
        class AdamW:
            def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
                self.params, self.state = list(params), {}
            def step(self): pass
            def zero_grad(self): pass
    ''',
    "torch/utils/__init__.py": "from . import data\n",
    "torch/utils/data.py": '''
        import math
        import numpy as _np
        import torch
        class Dataset: pass
        class DataLoader:
            def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False,
                         generator=None):
                self.ds, self.bs, self.shuffle = dataset, batch_size, shuffle
                self.rng = _np.random.default_rng(getattr(generator, "seed", 0))
            def __len__(self): return math.ceil(len(self.ds) / self.bs)
            def __iter__(self):
                idx = _np.arange(len(self.ds))
                if self.shuffle: self.rng.shuffle(idx)
                for i in range(0, len(idx), self.bs):
                    items = [self.ds[j] for j in idx[i:i + self.bs]]
                    yield {k: torch.Tensor(_np.stack([it[k].a for it in items])) for k in items[0]}
    ''',
    "stable_baselines3/__init__.py": '''
        """A stand-in for stable_baselines3.SAC: the interface that train_sac.py calls; the
        learn loop has the semantics of DummyVecEnv (the first reset gets the seed, an
        episode that ends is reset at once); the "policy" is a crude braking curve."""
        import pickle
        import numpy as _np
        __version__ = "stand-in"
        def _policy(obs):
            return _np.clip(_np.array([3.0 * (10 * obs[7] - 0.3 - 10 * obs[1]) / 1.875],
                                      dtype=_np.float32), -1, 1)
        class SAC:
            def __init__(self, policy, env, seed=None, device="auto", verbose=0):
                assert policy == "MlpPolicy"
                self.env, self.seed, self.device = env, seed, device
                self.learning_rate, self.buffer_size, self.learning_starts = 3e-4, 1000000, 100
                self.batch_size, self.tau, self.gamma = 256, 0.005, 0.99
                self.train_freq, self.gradient_steps, self.ent_coef = "TrainFreq(1, step)", 1, "auto"
                self.target_entropy, self.target_update_interval, self.use_sde = -1.0, 1, False
                class P: net_arch = [256, 256]
                self.policy = P()
                self.num_timesteps = 0
                env.action_space.seed(seed)
            def predict(self, obs, deterministic=False):
                return _policy(obs), None
            def learn(self, total_timesteps, callback=None):
                callback.init_callback(self)
                obs, _ = self.env.reset(seed=self.seed)
                while self.num_timesteps < total_timesteps:
                    a = (self.env.action_space.sample() if self.num_timesteps < self.learning_starts
                         else self.predict(obs)[0])
                    obs2, r, term, trunc, info = self.env.step(a)
                    self.num_timesteps += 1
                    callback.locals = {"dones": _np.array([term or trunc]), "infos": [info]}
                    if not callback.on_step():
                        break
                    obs = self.env.reset()[0] if (term or trunc) else obs2
                return self
            def save(self, path):
                with open(path, "wb") as fh: pickle.dump("stand-in", fh)
            @classmethod
            def load(cls, path, device="auto"):
                assert open(path, "rb").read()
                m = cls.__new__(cls)
                return m
    ''',
    "stable_baselines3/common/__init__.py": "",
    "stable_baselines3/common/callbacks.py": '''
        class BaseCallback:
            def __init__(self, verbose=0):
                self.model, self.locals = None, {}
            def init_callback(self, model): self.model = model
            def on_step(self): return self._on_step()
    ''',
    "hydra/__init__.py": "from . import utils\n",
    "hydra/utils.py": '''
        def get_original_cwd():
            raise ValueError("get_original_cwd() must only be used after HydraConfig is initialized")
    ''',
    "omegaconf/__init__.py": '''
        import yaml
        class DictConfig(dict):
            def __getattr__(self, k):
                try: return self[k]
                except KeyError: raise AttributeError(k)
            def __setattr__(self, k, v): self[k] = v
        class OmegaConf:
            @staticmethod
            def load(path):
                with open(path) as fh: return DictConfig(yaml.safe_load(fh))
            @staticmethod
            def to_container(cfg): return dict(cfg)
    ''',
    "tensordict/__init__.py": "from .tensordict import TensorDict\n",
    "tensordict/tensordict.py": '''
        import numpy as _np
        class TensorDict(dict):
            def __init__(self, batch_size=None, **kw):
                super().__init__(**kw); self.batch_size = batch_size
            def cat_many(self, xs): return TensorDict(batch_size=(len(xs),), **{k: None for k in self})
    ''',
    "diffusers/__init__.py": "",
    "diffusers/schedulers/__init__.py": "",
    "diffusers/schedulers/scheduling_ddpm.py": '''
        class DDPMScheduler:
            def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
                         beta_schedule="linear", trained_betas=None, variance_type="fixed_small",
                         clip_sample=True, prediction_type="epsilon", **kwargs):
                self.config = dict(num_train_timesteps=num_train_timesteps,
                                   prediction_type=prediction_type)
    ''',
    # a stand-in checkout of TD-MPC2: the files that train_tdmpc2.py imports
    "tdmpc2-checkout/tdmpc2/config.yaml": '''
        defaults:
            - override hydra/launcher: submitit_local
        task: dog-run
        obs: state
        episodic: false
        steps: 10_000_000
        rho: 0.5
        discount_denom: 5
        discount_min: 0.95
        discount_max: 0.995
        model_size: ???
        compile: true
        seed: 1
        enable_wandb: true
        save_video: true
        save_agent: true
        save_csv: true
        exp_name: default
        obs_shape: ???
        action_dim: ???
        episode_length: ???
        seed_steps: ???
    ''',
    "tdmpc2-checkout/tdmpc2/common/__init__.py": "MODEL_SIZE = {5: {'latent_dim': 512}}\nTASK_SET = {}\n",
    "tdmpc2-checkout/tdmpc2/common/parser.py": '''
        import dataclasses
        from pathlib import Path
        from typing import Any
        import hydra
        from omegaconf import OmegaConf
        from common import MODEL_SIZE, TASK_SET
        def parse_cfg(cfg):
            cfg.work_dir = Path(hydra.utils.get_original_cwd()) / "logs" / cfg.task
            for k, v in MODEL_SIZE[cfg.model_size].items(): cfg[k] = v
            cfg.multitask = cfg.task in TASK_SET
            d = OmegaConf.to_container(cfg)
            fields = [(k, Any, dataclasses.field(default_factory=lambda v=v: v)) for k, v in d.items()]
            return dataclasses.make_dataclass("Config", fields)()
    ''',
    "tdmpc2-checkout/tdmpc2/common/seed.py": '''
        import random
        import numpy as np
        import torch
        def set_seed(seed):
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    ''',
    "tdmpc2-checkout/tdmpc2/common/buffer.py": '''
        class Buffer:
            def __init__(self, cfg):
                assert cfg.episodic and cfg.steps > 0
                self.eps = 0
            def add(self, td):
                self.eps += 1; return self.eps
            def sample(self):
                assert self.eps > 0, "the buffer is empty"
    ''',
    "tdmpc2-checkout/tdmpc2/tdmpc2.py": '''
        import pickle
        import numpy as np
        import torch
        class TDMPC2:
            def __init__(self, cfg):
                assert cfg.action_dim == 1 and cfg.obs_shape == {"state": (9,)}
                frac = cfg.episode_length / cfg.discount_denom      # as TDMPC2._get_discount
                self.cfg = cfg
                self.discount = min(max((frac - 1) / frac, cfg.discount_min), cfg.discount_max)
                class M: total_params = 0
                self.model = M()
            def act(self, obs, t0=False, eval_mode=False, task=None):
                o = obs.numpy()
                a = np.clip(3.0 * (10 * o[7] - 0.3 - 10 * o[1]) / 1.875, -1, 1)
                if not eval_mode: a = float(np.clip(a + 0.1 * np.random.randn(), -1, 1))
                return torch.tensor([a], dtype=torch.float32)
            def update(self, buffer):
                buffer.sample(); return {}
            def save(self, fp):
                with open(fp, "wb") as fh: pickle.dump("stand-in", fh)
            def load(self, fp):
                assert open(fp, "rb").read()
    ''',
    # a stand-in checkout of Diffusion Policy: the modules that train_diffusion.py imports
    "dp-checkout/diffusion_policy/policy/diffusion_unet_lowdim_policy.py": '''
        import numpy as np
        import torch
        class DiffusionUnetLowdimPolicy:
            def __init__(self, model, noise_scheduler, horizon, obs_dim, action_dim, n_action_steps,
                         n_obs_steps, num_inference_steps=None, obs_as_local_cond=False,
                         obs_as_global_cond=False, pred_action_steps_only=False,
                         oa_step_convention=False, **kwargs):
                assert obs_as_global_cond and not pred_action_steps_only
                self.model, self.horizon, self.obs_dim = model, horizon, obs_dim
                self.normalizer = None
            def parameters(self): return []
            def set_normalizer(self, normalizer): self.normalizer = normalizer
            def to(self, device): return self
            def state_dict(self): return {}
            def compute_loss(self, batch):
                assert self.normalizer is not None
                assert batch["obs"].shape[1:] == (1, self.obs_dim), batch["obs"].shape
                assert batch["action"].shape[1:] == (self.horizon, 1), batch["action"].shape
                return torch.Tensor(float(np.mean(batch["action"].a ** 2)))
            def predict_action(self, obs_dict):
                o = obs_dict["obs"].a
                assert o.shape == (1, 1, self.obs_dim), o.shape
                a = np.clip(3.0 * (10 * o[0, 0, 7] - 0.3 - 10 * o[0, 0, 1]), -1.875, 1.875)
                pred = np.full((1, self.horizon, 1), a, dtype=np.float32)
                return {"action": torch.Tensor(pred[:, :4]), "action_pred": torch.Tensor(pred)}
    ''',
    "dp-checkout/diffusion_policy/model/diffusion/conditional_unet1d.py": '''
        class ConditionalUnet1D:
            def __init__(self, input_dim, local_cond_dim=None, global_cond_dim=None,
                         diffusion_step_embed_dim=256, down_dims=[256, 512, 1024], kernel_size=3,
                         n_groups=8, cond_predict_scale=False):
                assert input_dim == 1 and global_cond_dim == 9 and local_cond_dim is None
            def parameters(self): return []
    ''',
    "dp-checkout/diffusion_policy/model/common/normalizer.py": '''
        class LinearNormalizer:
            def fit(self, data, last_n_dims=1, dtype=None, mode="limits", output_max=1.0,
                    output_min=-1.0, range_eps=1e-4, fit_offset=True):
                assert set(data) == {"obs", "action"} and mode == "limits"
    ''',
    "dp-checkout/diffusion_policy/model/diffusion/ema_model.py": '''
        class EMAModel:
            def __init__(self, model, update_after_step=0, inv_gamma=1.0, power=2 / 3,
                         min_value=0.0, max_value=0.9999):
                self.averaged_model = model
            def step(self, new_model): pass
    ''',
    "dp-checkout/diffusion_policy/model/common/lr_scheduler.py": '''
        class _S:
            def step(self): pass
        def get_scheduler(name, optimizer, num_warmup_steps=None, num_training_steps=None, **kwargs):
            assert name == "cosine" and num_training_steps > 0
            return _S()
    ''',
    "dp-checkout/diffusion_policy/common/pytorch_util.py": '''
        def dict_apply(x, func):
            return {k: (dict_apply(v, func) if isinstance(v, dict) else func(v)) for k, v in x.items()}
        def optimizer_to(optimizer, device):
            return optimizer
    ''',
}


def write_fakes(root):
    for rel, text in FAKES.items():
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(textwrap.dedent(text).lstrip("\n"))
    env = dict(os.environ, GIT_AUTHOR_NAME="test", GIT_AUTHOR_EMAIL="test@example.org",
               GIT_COMMITTER_NAME="test", GIT_COMMITTER_EMAIL="test@example.org")
    for repo in ("tdmpc2-checkout", "dp-checkout"):
        p = os.path.join(root, repo)
        for cmd in (["git", "init", "-q"], ["git", "add", "-A"],
                    ["git", "commit", "-q", "-m", "stand-in"]):
            subprocess.run(cmd, cwd=p, env=env, check=True, capture_output=True)


def rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def main():
    tmp = tempfile.mkdtemp(prefix="ch06-harness-")
    fakes, tree = os.path.join(tmp, "fakes"), os.path.join(tmp, "code")
    write_fakes(fakes)
    for ch in ("ch05", "ch06"):
        shutil.copytree(os.path.join(CODE, ch), os.path.join(tree, ch),
                        ignore=shutil.ignore_patterns("__pycache__", "smoke", "sac", "tdmpc2",
                                                      "diffusion", "sac-*", "tdmpc2-*"))
    pub = os.path.join(tree, "ch06", "published")
    env = dict(os.environ, PYTHONPATH=fakes, PYTHONDONTWRITEBYTECODE="1")
    runs = [("SAC", ["train_sac.py", "--smoke"]),
            ("TD-MPC2", ["train_tdmpc2.py", "--smoke", "--allow-other-commit", "--tdmpc2",
                         os.path.join(fakes, "tdmpc2-checkout")]),
            ("diffusion policy", ["train_diffusion.py", "--smoke", "--allow-other-commit",
                                  "--diffusion-policy", os.path.join(fakes, "dp-checkout")]),
            ("SAC, time reward", ["train_sac.py", "--smoke", "--reward", "time"]),
            ("TD-MPC2, time reward", ["train_tdmpc2.py", "--smoke", "--reward", "time",
                                      "--allow-other-commit", "--tdmpc2",
                                      os.path.join(fakes, "tdmpc2-checkout")]),
            ("report", ["report.py", "--smoke", "--pack"]),
            ("report R2", ["report_r2.py", "--smoke"])]
    failures, lines = [], []
    for name, cmd in runs:
        q = subprocess.run([sys.executable] + cmd, cwd=pub, env=env, capture_output=True, text=True)
        tail = (q.stdout + q.stderr).strip().splitlines()[-4:]
        lines.append(f"{name}: exit {q.returncode}")
        lines += [f"    {t}" for t in tail]
        if q.returncode != 0:
            failures.append(f"{name} exited with {q.returncode}")
    smoke = os.path.join(pub, "results", "smoke")
    expect = {"sac": (4, 3), "tdmpc2": (4, 3), "diffusion": (2, 2),   # evaluation, training rows
              "sac-time": (4, 3), "tdmpc2-time": (4, 3)}
    rewards = {"sac": "registered", "tdmpc2": "registered", "sac-time": "time",
               "tdmpc2-time": "time"}
    discounts = {"tdmpc2": 0.975, "tdmpc2-time": 0.99}
    for folder, (n_ev, n_tr) in expect.items():
        d = os.path.join(smoke, folder, "seed1")
        try:
            ev, tr = rows(os.path.join(d, "evaluation.csv")), rows(os.path.join(d, "training.csv"))
            with open(os.path.join(d, "record.json")) as fh:
                rec = json.load(fh)
        except OSError as e:
            failures.append(f"{folder}: {e}")
            continue
        ok = len(ev) == n_ev and len(tr) == n_tr and "environment" in rec
        if folder in rewards and rec.get("reward") != rewards[folder]:
            failures.append(f"{folder}: record says reward {rec.get('reward')}")
        if folder in discounts and abs(rec.get("discount", 0) - discounts[folder]) > 1e-9:
            failures.append(f"{folder}: record says discount {rec.get('discount')}")
        bad = [r for r in ev + tr if r["alarms"] != "0" or r["steps_outside"] != "0"
               or float(r["max_position_exact"]) > 24.5]
        arrived = sum(1 for r in ev if r["arrival_s"])
        lines.append(f"{folder}: {len(ev)} evaluation runs ({arrived} arrived), {len(tr)} training "
                     f"episodes, record with {len(rec)} entries; runs with an alarm, a step outside "
                     f"a funnel or past the floor: {len(bad)}")
        if not ok or bad:
            failures.append(f"{folder}: files not as expected")
    summ = os.path.join(smoke, "summary.txt")
    if os.path.exists(summ):
        text = open(summ).read()
        for g in ("[table]", "[H1]", "[primary]", "[secondary]", "[record]"):
            if g not in text:
                failures.append(f"summary.txt lacks {g}")
        lines.append(f"report: summary.txt with {text.count(chr(10))} lines")
    else:
        failures.append("report wrote no summary.txt")
    summ2 = os.path.join(smoke, "summary-r2.txt")
    if os.path.exists(summ2):
        text2 = open(summ2).read()
        for g in ("[table]", "[H1]", "[primary]", "[secondary]", "[record]", "Q4 = "):
            if g not in text2:
                failures.append(f"summary-r2.txt lacks {g}")
        lines.append(f"report R2: summary-r2.txt with {text2.count(chr(10))} lines")
    else:
        failures.append("report_r2.py wrote no summary-r2.txt")
    if os.path.exists(os.path.join(smoke, "send-back.zip")):
        import zipfile
        with zipfile.ZipFile(os.path.join(smoke, "send-back.zip")) as z:
            names = z.namelist()
        lines.append(f"report --pack: send-back.zip with {len(names)} files, "
                     f"{sum(1 for n in names if n.endswith(('.zip', '.pt')))} policies")
    else:
        failures.append("report --pack wrote no send-back.zip")
    lines.append("[check] " + ("the scripts ran and wrote their files" if not failures
                               else "FAILED: " + "; ".join(failures)))
    print("\n".join(lines))
    if "--keep" in sys.argv:
        print(f"the temporary tree is kept: {tmp}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
