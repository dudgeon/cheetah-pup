"""Independent sampled assembly audit for smoke resets and trained evaluation poses."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from stand_env import ROOT, StandEnv
from cheetah_pup.assembly_audit import inspect_poses


def main():
    env = StandEnv()
    reset_seeds = list(range(20260904, 20260928)) + list(range(88001, 88009))
    reset_poses = []
    for seed in reset_seeds:
        env.reset(seed=seed)
        reset_poses.append(env.data.qpos.copy())
    reset_audit = inspect_poses(env.model, reset_poses)
    policy = PPO.load(ROOT / "models/policies/cpu-stand-smoke.zip", device="cpu")
    samples, failures = [], []
    for seed in range(88001, 88009):
        obs, _ = env.reset(seed=seed)
        for step in range(env.max_steps):
            action, _ = policy.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if step % 25 == 0 or terminated or truncated:
                samples.append(env.data.qpos.copy())
            if terminated or truncated:
                if terminated:
                    failures.append({"seed": seed, "reason": info["failure_reason"]})
                break
    rollout_audit = inspect_poses(env.model, samples)
    env.close()
    result = {
        "scope": "Independent SAT/narrow-phase assembly-envelope audit, including extra adjacent/welded bodies and connector reservations. Samples only; neither full action-box clearance nor all training trajectories are certified.",
        "reset_seeds": reset_seeds,
        "reset_pose_audit": reset_audit,
        "trained_evaluation_sample_every_policy_steps": 25,
        "trained_evaluation_pose_audit": rollout_audit,
        "replay_failures": failures,
    }
    path = ROOT / "reports/cpu-rl-smoke.json"
    report = json.loads(path.read_text())
    report["sampled_geometry_audit"] = result
    source = Path(__file__)
    report["source_hashes_sha256"][str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
    path.write_text(json.dumps(report, indent=2) + "\n")
    md_path = path.with_suffix(".md")
    marker = "\n## Additional sampled assembly audit\n"
    text = md_path.read_text().split(marker)[0]
    text += marker + "\n"
    for label, audit in (("Reset", reset_audit), ("Trained evaluation", rollout_audit)):
        text += f"- {label}: {audit['poses_checked']} poses, {audit['solid_interference_pairs']} solid interference pairs, {audit['cable_allowance_interference_pairs']} connector-reservation interference pairs.\n"
    text += "\nThese independent checks include adjacent/welded geometry beyond configured MuJoCo contact pairs. They cover sampled resets and sampled evaluated poses, not the entire action space or all training trajectories.\n"
    md_path.write_text(text)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
