# DR-03 — RL environment for the locked design

**Status**: built and smoke-tested on CPU; ready for a cloud-GPU training run.
Code: `cheetah_pup/rl/` (`base.py`, `joystick.py`, `randomize.py`, `train.py`, `constants.py`).
Model: `sim/cheetah_pup_rl.xml` from `python -m cheetah_pup.mjcf sim/cheetah_pup_rl.xml --rl`.
Tests: `tests/test_rl_env.py`. Stack: MuJoCo Playground (MJX) + Brax PPO, the same lineage as
Open Duck Mini's training; the environment mirrors Playground's Go1 joystick task (Apache-2.0),
which is the same 12-DOF abad/hip/knee quadruped layout.

## Task

Track a body-frame velocity command (vx, vy, yaw rate) on flat ground at 50 Hz control,
2 ms physics. Commands: |vx| ≤ 0.25 m/s, |vy| ≤ 0.10 m/s, |yaw| ≤ 1.0 rad/s, resampled at random
intervals with a chance of zeroing each component (stand still is part of the task).

## What the policy sees and does

| | |
|---|---|
| Observation (policy, 49) | gyro (3), gravity direction in the IMU frame (3), command (3), joint positions − standing pose (12), joint velocities × 0.05 (12), last action (12), foot contacts (4) — only what the real robot measures (BNO055 + servo encoders + foot switches). Noise added to each. |
| Observation (critic) | the above plus accelerometer, base linear/angular velocity, joint state without noise, actuator forces, foot velocities, air times, trunk height |
| Action (12) | offsets from the standing pose, ×0.3 rad; motor targets are slew-limited to the STS3215 firmware cap (5.29 rad/s) and clipped to the joint ranges before reaching the PD servos |
| Termination | trunk up-vector z < 0.3 (~73° tilt) or NaN |
| Episode | 1000 steps (20 s) |

## Rewards (per step, scaled by dt)

Tracking: lin-vel `1.5·exp(−|Δv|²/0.02)`, ang-vel `0.75·exp(−Δω²/0.2)`. Posture: orientation
−5, vertical velocity −0.5, roll/pitch rate −0.05, pose −/+0.3 toward the standing pose, soft joint
limits −1, stand-still −0.5 when the command is zero, termination −1. Effort: torques −0.002,
energy −0.01, action rate −0.01 (all ~10× the Go1 coefficients — our torques are ~10× smaller).
Feet: air time +0.2 (swing ≥ 0.1 s), clearance −1 toward a 30 mm swing height, slip −0.1.

## Domain randomization (per environment)

Floor friction U(0.5, 1.0); joint Coulomb friction ×U(0.8, 1.2); armature ×U(0.9, 1.1); trunk CoM
±10 mm; every body mass ×U(0.9, 1.1) plus ±0.1 kg on the trunk; joint zero offsets ±0.03 rad;
servo kp ×U(0.85, 1.15); torque limit ×U(0.85, 1.05) (battery sag). Pushes are implemented but off
by default. Backlash and the servo's rate-limited internal target come with BAM's actuator model
in the next step.

## Model variant differences from `sim/cheetah_pup.xml`

Feet-only collisions (shell and legs do not collide — Playground's convention for training speed;
falls are caught by the tilt termination), pyramidal friction cone, contact sensors on each foot,
the IMU frame sensor set, and a `home` keyframe alias. Masses, geometry, and the servo model are
identical.

## Running it

```
# CPU pipeline check (a few thousand steps, small networks; ~10 min mostly JIT)
JAX_PLATFORMS=cpu .venv/bin/python -m cheetah_pup.rl.train --smoke

# GPU run (cloud): install jax[cuda12] and mujoco-warp, then
.venv/bin/python -m cheetah_pup.rl.train --impl warp --num-timesteps 100000000 --num-envs 4096
```

Playground's Go1 PPO hyperparameters are the starting point (policy 512-256-128, value the same,
observation normalization on). Expect a first walking policy within tens of millions of steps —
Microduck reports 1–2 hours on one GPU at 4096 environments for a comparable task.

## Smoke test result (CPU, 2026-09-04)

`--smoke`: 8 environments, 100-step episodes, 64-64 networks, 2,048 requested timesteps → 5,120
executed (Brax rounds up to whole batches), 133 s wall time of which nearly all is JIT compile.
Evaluation with the barely-trained policy: episode reward 0.78, average episode length 99.7 of
100 (the robot mostly stays upright for 2 s from the standing pose), every reward term finite and
of the expected sign (tracking +34.8/+20.0, pose +27.5, orientation −8.2, energy −16.2, torques
−3.4, action rate −6.5, feet air time +0.35). Checkpoint `sim/checkpoints/smoke_params` (87 KB)
saved with `brax.io.model`. This proves environment, randomization, PPO, evaluation, and
checkpointing run together; it says nothing about walking quality — that needs the GPU run.

One compatibility fix was needed: Brax 0.14.2 calls `jax.device_put_replicated`, removed in
JAX 0.10; `cheetah_pup/rl/compat.py` restores it with the documented replacement.

## Next steps

1. Cloud-GPU training run; evaluate with `sim/cheetah_pup.xml` (full collisions) and in the DR-02
   playback page (a policy-driven recording is one function away in `validate.py`).
2. Actuator realism: BAM's MuJoCo actuator (rate-limited target, extended friction, backlash).
3. ONNX export of the policy MLP for the Raspberry Pi runtime (Phase 4).
