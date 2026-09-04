# CPU PPO smoke experiment

Actual PPO training on the current 12-actuator free-base assembly and pinned BAM controller. This is a narrow standing task, not a walking policy or stock-servo validation.

Collected **16,384 transitions** in **20.0 seconds** (820.9 transitions/s), using 8 CPU workers. PPO applied 40 training epochs across completed rollout updates.

Training uses 50 Hz actions and ten 2 ms BAM physics steps per action, with unchanged 5 V/P400 settings and 20 ms command delay. Actions are twelve bounded ±0.15 rad offsets from the stand pose. Both actor and critic receive only 45 proprioceptive/command values; privileged simulator state appears only in rewards, termination, and evaluation.

## Held-out evaluation

All three controllers use the same eight held-out reset seeds and five-second horizon. Initial joints vary by ±0.03 rad, with small joint/base velocities. Deterministic PPO inference is used for both untrained and trained policies.

| Controller | Successful episodes | Mean return | Height RMSE | Worst tilt | Mean drift |
|---|---:|---:|---:|---:|---:|
| Fixed P400 home targets | 8/8 | 16.686 | 2.75 mm | 1.11° | 10.35 mm |
| Untrained PPO | 8/8 | 16.692 | 2.74 mm | 1.11° | 10.30 mm |
| Trained PPO | 8/8 | 16.676 | 2.76 mm | 1.11° | 10.51 mm |

The network weights changed (L2 distance 3.0674); this run performed actual PPO updates. Held-out mean return changed by -0.016 versus the untrained network and -0.010 versus fixed home targets. One training seed and eight narrow reset seeds cannot establish statistically reliable improvement. Standing already works with fixed targets, so CPU training feasibility is the main result. No walking, motor viability, robustness, or sim-to-real claim follows from this experiment.

## Scope and reproduction

A successful episode must avoid loaded self-contact and non-foot ground contact at every physics step, keep body height above 105 mm, tilt below 0.35 rad, and drift below 40 mm. The checks also reject a deliberately lowered body and verify deterministic reset, including BAM friction and command-delay state. These are deliberately modest smoke-test bounds, not the final stand gate.

See [training_cpu/README.md](../training_cpu/README.md) for versions, exact commands, task contract, checkpoint use, and limitations. [Machine-readable results](cpu-rl-smoke.json) include all episodes, training configuration, throughput, model/dependency hashes, and environment checks.

Primary references: [SB3 2.7.1 PPO CPU guidance](https://stable-baselines3.readthedocs.io/en/v2.7.1/modules/ppo.html), [official PyTorch CPU wheel instructions](https://pytorch.org/get-started/previous-versions/).

## Additional sampled assembly audit

- Reset: 32 poses, 0 solid interference pairs, 0 connector-reservation interference pairs.
- Trained evaluation: 88 poses, 0 solid interference pairs, 0 connector-reservation interference pairs.

These independent checks include adjacent/welded geometry beyond configured MuJoCo contact pairs. They cover sampled resets and sampled evaluated poses, not the entire action space or all training trajectories.
