"""Run one bounded CPU PPO seed, then compare held-out resets against controls."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import gymnasium
import mujoco
import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from stand_env import ROOT, TASK, StandEnv


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_env(rank, seed):
    def create():
        env = StandEnv()
        env.reset(seed=seed + rank)
        return Monitor(env)
    return create


def verify_environment():
    env = StandEnv()
    check_env(env, warn=True, skip_render_check=True)
    def trajectory():
        obs, _ = env.reset(seed=31337)
        samples = [obs]
        for _ in range(30):
            samples.append(env.step(np.zeros(12))[0])
        return np.stack(samples)
    first, second = trajectory(), trajectory()
    reset_error = float(np.max(np.abs(first - second)))
    assert reset_error == 0, reset_error
    # A body lowered into the ground must not pass on height/upright alone.
    env.reset(seed=31337)
    env.data.qpos[2] = 0.02
    mujoco.mj_forward(env.model, env.data)
    reason = env.failure_reason(env.loaded_bad_contacts())
    assert reason is not None
    env.reset(seed=31337)
    before = env.observation()
    env.data.qpos[:2] += [1.2, -0.8]
    mujoco.mj_forward(env.model, env.data)
    translation_obs_error = float(np.max(np.abs(env.observation() - before)))
    assert translation_obs_error < 1e-7
    start = time.monotonic()
    env.reset(seed=31337)
    steps = 0
    for _ in range(250):
        _, _, terminated, truncated, _ = env.step(np.zeros(12))
        steps += 1
        if terminated or truncated:
            env.reset()
    elapsed = time.monotonic() - start
    result = {
        "gymnasium_sb3_check_env": "passed",
        "repeated_seed_trajectory_max_abs_error": reset_error,
        "body_on_ground_rejected_as": reason,
        "base_xy_translation_observation_max_abs_error": translation_obs_error,
        "single_env_hold_policy_steps": steps,
        "single_env_hold_wall_s": elapsed,
        "single_env_hold_policy_steps_per_wall_s": steps / elapsed,
        "single_env_hold_sim_seconds_per_wall_s": steps * TASK["policy_dt_s"] / elapsed,
    }
    env.close()
    print(json.dumps({"environment_checks": result}), flush=True)
    return result


class BudgetCallback(BaseCallback):
    def __init__(self, seconds):
        super().__init__()
        self.seconds = seconds
        self.rollouts = []

    def _on_training_start(self):
        self.started = time.monotonic()

    def _on_step(self):
        return time.monotonic() - self.started < self.seconds

    def _on_rollout_end(self):
        values = {
            "policy_transitions": self.num_timesteps,
            "wall_s": time.monotonic() - self.started,
            "completed_episode_count_in_window": len(self.model.ep_info_buffer),
            "mean_episode_return_in_window": float(np.mean([v["r"] for v in self.model.ep_info_buffer])) if self.model.ep_info_buffer else None,
            "mean_episode_length_in_window": float(np.mean([v["l"] for v in self.model.ep_info_buffer])) if self.model.ep_info_buffer else None,
        }
        self.rollouts.append(values)
        print(json.dumps({"rollout": values}), flush=True)


def evaluate(policy, seeds):
    env = StandEnv()
    results = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        total, steps, square_error = 0.0, 0, 0.0
        max_tilt, min_height, max_torque = 0.0, float(env.data.qpos[2]), 0.0
        while True:
            action = np.zeros(12) if policy is None else policy.predict(obs, deterministic=True)[0]
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            steps += 1
            square_error += info["height_error_m"] ** 2
            max_tilt = max(max_tilt, info["max_tilt_rad"])
            min_height = min(min_height, info["min_height_m"])
            max_torque = max(max_torque, info["peak_motor_torque_nm"])
            if terminated or truncated:
                results.append({
                    "seed": seed,
                    "success": bool(truncated and not terminated),
                    "return": total,
                    "policy_steps": steps,
                    "duration_s": info["simulation_time_s"],
                    "failure_reason": info["failure_reason"],
                    "height_rmse_m": float(np.sqrt(square_error / steps)),
                    "maximum_tilt_deg": float(np.rad2deg(max_tilt)),
                    "minimum_height_m": min_height,
                    "final_drift_m": info["drift_m"],
                    "peak_motor_torque_nm": max_torque,
                })
                break
    env.close()
    summary = {
        "episodes": len(results),
        "successes": sum(r["success"] for r in results),
        "mean_return": float(np.mean([r["return"] for r in results])),
        "mean_height_rmse_mm": float(np.mean([r["height_rmse_m"] for r in results]) * 1000),
        "worst_tilt_deg": max(r["maximum_tilt_deg"] for r in results),
        "mean_final_drift_mm": float(np.mean([r["final_drift_m"] for r in results]) * 1000),
        "failure_reasons": dict(Counter(r["failure_reason"] for r in results if r["failure_reason"])),
    }
    return {"summary": summary, "episodes": results}


def write_markdown(report):
    train = report["training"]
    text = [
        "# CPU PPO smoke experiment", "",
        "Actual PPO training on the current 12-actuator free-base assembly and pinned BAM controller. "
        "This is a narrow standing task, not a walking policy or stock-servo validation.", "",
        f"Collected **{train['actual_policy_transitions']:,} transitions** in **{train['wall_s']:.1f} seconds** "
        f"({train['policy_transitions_per_wall_s']:.1f} transitions/s), using {train['parallel_envs']} CPU workers. "
        f"PPO applied {train['optimizer_epochs']} training epochs across completed rollout updates.", "",
        "Training uses 50 Hz actions and ten 2 ms BAM physics steps per action, with unchanged 5 V/P400 "
        "settings and 20 ms command delay. Actions are twelve bounded ±0.15 rad offsets from the stand pose. "
        "Both actor and critic receive only 45 proprioceptive/command values; privileged simulator state "
        "appears only in rewards, termination, and evaluation.", "",
        "## Held-out evaluation", "",
        "All three controllers use the same eight held-out reset seeds and five-second horizon. "
        "Initial joints vary by ±0.03 rad, with small joint/base velocities. Deterministic PPO inference "
        "is used for both untrained and trained policies.", "",
        "| Controller | Successful episodes | Mean return | Height RMSE | Worst tilt | Mean drift |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in report["evaluation"].items():
        s = values["summary"]
        text.append(f"| {name} | {s['successes']}/{s['episodes']} | {s['mean_return']:.3f} | {s['mean_height_rmse_mm']:.2f} mm | {s['worst_tilt_deg']:.2f}° | {s['mean_final_drift_mm']:.2f} mm |")
    text += [
        "", report["interpretation"], "",
        "## Scope and reproduction", "",
        "A successful episode must avoid loaded self-contact and non-foot ground contact at every physics "
        "step, keep body height above 105 mm, tilt below 0.35 rad, and drift below 40 mm. The checks also "
        "reject a deliberately lowered body and verify deterministic reset, including BAM friction and "
        "command-delay state. These are deliberately modest smoke-test bounds, not the final stand gate.", "",
        "See [training_cpu/README.md](../training_cpu/README.md) for versions, exact commands, task contract, "
        "checkpoint use, and limitations. [Machine-readable results](cpu-rl-smoke.json) include all "
        "episodes, training configuration, throughput, model/dependency hashes, and environment checks.", "",
        "Primary references: [SB3 2.7.1 PPO CPU guidance](https://stable-baselines3.readthedocs.io/en/v2.7.1/modules/ppo.html), "
        "[official PyTorch CPU wheel instructions](https://pytorch.org/get-started/previous-versions/).", "",
    ]
    (ROOT / "reports/cpu-rl-smoke.md").write_text("\n".join(text))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=16384)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-training-seconds", type=float, default=480)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    checks = verify_environment()
    if args.check_only:
        return
    seed = 20260904
    heldout = list(range(88001, 88009))
    artifacts = ROOT / "models/policies"
    artifacts.mkdir(exist_ok=True)
    vec = SubprocVecEnv([make_env(i, seed) for i in range(args.workers)], start_method="spawn")
    vec.seed(seed)
    kwargs = {
        "learning_rate": 3e-4, "n_steps": 256, "batch_size": 256,
        "n_epochs": 5, "gamma": 0.99, "gae_lambda": 0.95,
        "clip_range": 0.2, "ent_coef": 0.0, "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "policy_kwargs": {"net_arch": {"pi": [64, 64], "vf": [64, 64]}, "log_std_init": -1.5},
        "device": "cpu", "seed": seed, "verbose": 0,
    }
    policy = PPO("MlpPolicy", vec, **kwargs)
    initial = {k: v.detach().clone() for k, v in policy.policy.state_dict().items()}
    policy.save(artifacts / "cpu-stand-smoke-untrained")
    budget = BudgetCallback(args.max_training_seconds)
    started = time.monotonic()
    policy.learn(total_timesteps=args.steps, callback=budget, progress_bar=False)
    elapsed = time.monotonic() - started
    policy.save(artifacts / "cpu-stand-smoke")
    weight_change = float(torch.sqrt(sum(torch.sum((v.detach() - initial[k]) ** 2) for k, v in policy.policy.state_dict().items())))
    vec.close()
    print(json.dumps({"training_completed": {"wall_s": elapsed, "transitions": policy.num_timesteps, "parameter_l2_change": weight_change}}), flush=True)
    untrained = PPO.load(artifacts / "cpu-stand-smoke-untrained.zip", device="cpu")
    eval_started = time.monotonic()
    evaluation = {}
    for name, candidate in (("Fixed P400 home targets", None), ("Untrained PPO", untrained), ("Trained PPO", policy)):
        evaluation[name] = evaluate(candidate, heldout)
        print(json.dumps({"evaluation": name, **evaluation[name]["summary"]}), flush=True)
    robot = StandEnv()
    artifact_config = {"task": TASK, "ppo": kwargs, "heldout_seeds": heldout, "joint_order": list(robot.controller.names)}
    config_path = artifacts / "cpu-stand-smoke-config.json"
    config_path.write_text(json.dumps(artifact_config, indent=2) + "\n")
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in (
        ROOT / "config/robot.json", ROOT / "config/actuator.json", ROOT / "src/cheetah_pup/model.py",
        ROOT / "src/cheetah_pup/actuator.py", ROOT / "src/cheetah_pup/assembly.py",
        Path(__file__), Path(__file__).with_name("stand_env.py"), Path(__file__).with_name("uv.lock"),
        artifacts / "cpu-stand-smoke.zip", artifacts / "cpu-stand-smoke-untrained.zip", config_path,
    )}
    trained = evaluation["Trained PPO"]["summary"]
    baseline = evaluation["Fixed P400 home targets"]["summary"]
    before = evaluation["Untrained PPO"]["summary"]
    interpretation = (
        f"The network weights changed (L2 distance {weight_change:.4f}); this run performed actual PPO "
        f"updates. Held-out mean return changed by {trained['mean_return'] - before['mean_return']:+.3f} "
        f"versus the untrained network and {trained['mean_return'] - baseline['mean_return']:+.3f} versus "
        "fixed home targets. One training seed and eight narrow reset seeds cannot establish "
        "statistically reliable improvement. Standing already works with fixed targets, so CPU "
        "training feasibility is the main result. No walking, motor viability, robustness, or sim-to-real "
        "claim follows from this experiment."
    )
    report = {
        "task": TASK,
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "mujoco": mujoco.__version__, "numpy": np.__version__, "torch": torch.__version__,
                    "gymnasium": gymnasium.__version__, "stable_baselines3": stable_baselines3.__version__,
                    "cuda_available": torch.cuda.is_available(), "torch_threads": torch.get_num_threads(),
                    "cpu_quota": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
                    "memory_limit_bytes": int(Path("/sys/fs/cgroup/memory.max").read_text())},
        "source_commit_before_experiment": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_hashes_sha256": hashes,
        "compiled_mjcf_sha256": hashlib.sha256(robot.xml.encode()).hexdigest(),
        "actuator": robot.controller.config,
        "environment_checks": checks,
        "training": {"algorithm": "Stable-Baselines3 PPO", "seed": seed, "parallel_envs": args.workers,
                     "requested_policy_transitions": args.steps, "actual_policy_transitions": policy.num_timesteps,
                     "wall_budget_s": args.max_training_seconds, "wall_s": elapsed,
                     "policy_transitions_per_wall_s": policy.num_timesteps / elapsed,
                     "nominal_simulation_seconds_collected": policy.num_timesteps * TASK["policy_dt_s"],
                     "optimizer_epochs": policy._n_updates, "parameter_l2_change": weight_change,
                     "hyperparameters": kwargs, "rollouts": budget.rollouts},
        "evaluation_wall_s": time.monotonic() - eval_started,
        "evaluation": evaluation, "interpretation": interpretation,
    }
    robot.close()
    (ROOT / "reports/cpu-rl-smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report)
    print(interpretation, flush=True)


if __name__ == "__main__":
    main()
