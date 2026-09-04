"""Brax PPO runner for the Cheetah Pup joystick task.

    python -m cheetah_pup.rl.train --smoke                # CPU pipeline check, a few thousand steps
    python -m cheetah_pup.rl.train --num-timesteps 100000000 --impl warp   # GPU run

Starts from Playground's Go1 joystick PPO hyperparameters. Checkpoints (Brax params) are saved
with brax.io.model; convert to ONNX for the robot with cheetah_pup.rl.export (Phase 4).
"""

from __future__ import annotations

import argparse
import functools
import json
import pathlib
import time

import jax

from . import compat  # noqa: F401 — restores jax.device_put_replicated for Brax before it is imported
from brax.io import model as brax_model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from ml_collections import config_dict
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params

from . import joystick, randomize


def ppo_config(smoke: bool, num_timesteps: int | None, num_envs: int | None) -> config_dict.ConfigDict:
    cfg = locomotion_params.brax_ppo_config("Go1JoystickFlatTerrain")
    if smoke:
        cfg.num_timesteps = 2048
        cfg.num_envs = 8
        cfg.num_evals = 1
        cfg.unroll_length = 8
        cfg.batch_size = 32
        cfg.num_minibatches = 2
        cfg.num_updates_per_batch = 1
        cfg.episode_length = 100
        cfg.network_factory.policy_hidden_layer_sizes = (64, 64)
        cfg.network_factory.value_hidden_layer_sizes = (64, 64)
    if num_timesteps:
        cfg.num_timesteps = num_timesteps
    if num_envs:
        cfg.num_envs = num_envs
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="sim/checkpoints")
    ap.add_argument("--num-timesteps", type=int, default=None)
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--impl", default="jax", choices=["jax", "warp"])
    ap.add_argument("--smoke", action="store_true", help="tiny run that only proves the pipeline")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)

    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    overrides = {"impl": args.impl}
    if args.smoke:
        overrides["episode_length"] = 100
    env = joystick.Joystick(config_overrides=overrides)
    eval_env = joystick.Joystick(config_overrides=overrides)
    cfg = ppo_config(args.smoke, args.num_timesteps, args.num_envs)
    train_params = dict(cfg)
    network_factory = functools.partial(ppo_networks.make_ppo_networks, **train_params.pop("network_factory"))
    print(f"obs sizes: {env.observation_size}, action size: {env.action_size}")
    print("ppo:", {k: v for k, v in train_params.items() if not isinstance(v, dict)})

    log = []
    t0 = time.time()

    def progress(num_steps, metrics):
        row = {"steps": int(num_steps), "wall_s": round(time.time() - t0, 1),
               **{k: float(v) for k, v in metrics.items() if "eval/episode_reward" in k or "eval/avg_episode_length" in k}}
        log.append(row)
        print(json.dumps(row))

    train_fn = functools.partial(
        ppo.train, **train_params, network_factory=network_factory,
        randomization_fn=randomize.domain_randomize, progress_fn=progress, seed=args.seed)
    make_inference_fn, params, _ = train_fn(environment=env, eval_env=eval_env, wrap_env_fn=wrapper.wrap_for_brax_training)
    path = out / ("smoke_params" if args.smoke else f"params_{int(cfg.num_timesteps)}")
    brax_model.save_params(str(path), params)
    (out / (path.name + "_log.json")).write_text(json.dumps({"config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in train_params.items()},
                                                             "obs_size": env.observation_size, "log": log}, indent=1, default=str))
    print(f"saved {path} after {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
