# Bounded CPU standing experiment

This directory is an isolated Linux CPU training environment. It runs genuine
Stable-Baselines3 PPO on the project's current original quadruped assembly and
the existing `BamPositionController`. It does not change the root dependencies,
actuator fit, firmware gain, motor voltage, mass, geometry, or collision settings.

The purpose is to exercise reset, observation, action, reward, termination,
learning, checkpoint, and evaluation code cheaply before cloud work. This is
not the MuJoCo Warp/mjlab training path and does not establish CPU/GPU parity.

## Reproduce

From the repository root, with the current submodules initialized:

```sh
git submodule update --init vendor/bam_microduck
uv sync --project training_cpu --locked
uv run --project training_cpu --locked python training_cpu/run_smoke.py --check-only
uv run --project training_cpu --locked python training_cpu/run_smoke.py --steps 16384 --workers 8 --max-training-seconds 480
uv run --project training_cpu --locked python training_cpu/check_geometry.py
```

The first command needs access to the public upstream repository. Installation
downloads the pinned CPU PyTorch wheel, about 176 MiB compressed on this machine.
Subsequent runs use the locked environment. The training budget limits collection;
startup, completed PPO updates, and evaluation add wall time. An interrupted final
rollout is not optimized and is still counted in collected transitions.

Versions: Python 3.12, MuJoCo 3.10.0, NumPy 2.4.1, Stable-Baselines3 2.7.1,
Gymnasium 1.2.3, PyTorch 2.9.1+cpu. The lock selects PyTorch only from its official
CPU index, preventing accidental installation of CUDA runtime packages. This
wheel configuration is for Linux CPU execution in the current sandbox; it is
not a tested macOS installation recipe.

## Task contract

`stand_env.py` defines the complete versioned `TASK` dictionary:

- Twelve normalized actions map to home joint angles plus bounded ±0.15 rad
  offsets, in `FL, FR, RL, RR` order, with `hip_roll, hip_pitch, knee` per leg.
- Actions update at 50 Hz. Each action advances ten 2 ms physics steps using
  the actual pinned BAM controller at 5 V, firmware P400, and 20 ms command delay.
- The 45-value float32 observation contains body gyro (3), projected gravity
  (3), zero velocity commands (3), joint offsets (12), joint velocity (12), and
  the preceding normalized action (12). Exact scaling and slices are in `TASK`.
- Neither actor nor critic receives base position, linear velocity, contact
  force, body height, or motor torque. Simulator-only state may shape rewards
  and define termination, as documented in the task configuration.
- Episodes last five seconds. Reset joints vary uniformly by ±0.03 rad with
  small joint/base velocities. The whole robot is moved vertically during reset
  to clear the lowest sole by 0.2 mm, then only forward dynamics move it.
- Every reset clears BAM's previous torque/friction and target-delay queue.
  Every physics step checks for loaded self-contact and non-foot ground contact.
  Failure also occurs below 105 mm body height, above 0.35 rad tilt, beyond 40 mm
  XY drift, or on nonfinite state. Upright resting on the chassis cannot pass.
- PPO uses two 64-unit hidden layers per actor/critic, one training seed, eight
  worker processes, 2,048 transitions per rollout, and five optimization epochs
  per completed rollout. The exact hyperparameters accompany the checkpoints.

The root assembly audit includes additional adjacent/welded solid and reserved
connector-space checks beyond MuJoCo's configured contact pairs. Per-step contact
checks here do not prove clearance across the complete joint range, nor do they
check flexible wires or bracket fasteners.

## Outputs and evaluation

The runner writes:

- `reports/cpu-rl-smoke.json` and `.md`: measured throughput, environment checks,
  PPO updates, weight change, all evaluation episodes, source and artifact hashes.
- `models/policies/cpu-stand-smoke.zip`: trained SB3 weights, optimizer, and model
  metadata. This is a research checkpoint, not a robot deployment artifact.
- `models/policies/cpu-stand-smoke-untrained.zip`: the same network before updates.
- `models/policies/cpu-stand-smoke-config.json`: task and policy contracts.

All three evaluation controllers use the same eight held-out seeds and
five-second horizon: fixed P400 home targets, the untrained deterministic network,
and the trained deterministic network. The baseline is already a competent stand
in this model; failure to beat it is a useful result, not grounds to weaken it.

To load the trusted project-generated checkpoint in this environment:

```python
from stable_baselines3 import PPO

policy = PPO.load("models/policies/cpu-stand-smoke.zip", device="cpu")
action, _ = policy.predict(observation, deterministic=True)
```

The observation must follow the exact saved order, scales, and previous-action
semantics. The environment performs action clipping and radian conversion.

## Practical limits

This is a small standing experiment with one training seed. A few seconds of
survival from small perturbations do not prove recovery, locomotion, motor
thermal margin, battery endurance, carpet/threshold handling, or sim-to-real
success. No observation noise, transmission backlash, structural compliance,
or parameter randomization is included. Perfect simulated IMU signals are an
explicit placeholder. The underlying stock-servo fit-provenance and firmware
uncertainties remain unchanged.

Cloud preparation still requires a reviewed task/reward/termination contract,
GPU actuator and friction parity with this CPU reference, broader reset/contact
validation, and a reproducible capped job. A longer CPU run may be useful for
debugging but is not a substitute for that work.

Primary references: [SB3 2.7.1 PPO, including CPU execution guidance](https://stable-baselines3.readthedocs.io/en/v2.7.1/modules/ppo.html),
[official PyTorch CPU installation recipes](https://pytorch.org/get-started/previous-versions/).
