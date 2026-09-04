# Design log

Dated record of decisions, milestones, and findings. Newest first. The decision table in
`docs/HANDOFF.md` §3 is the summary; this log keeps the reasoning and the state at the time.

## 2026-09-04 — RL environment built (DR-03, Fable 5.1)

- **Stack decision for Phase 1**: MuJoCo Playground (MJX, JAX backend on CPU here; `--impl warp`
  for GPU) + Brax PPO — the lineage Open Duck Mini trained on, installable in this sandbox. mjlab
  remains the alternative for Phase 5 if Warp on the cloud GPU is preferred.
- **Environment** (`cheetah_pup/rl/`): joystick velocity-tracking task modeled on Playground's
  Go1 env (same 12-DOF layout), with the real robot's sensor set only in the policy observation,
  servo slew limit on motor targets, rewards rescaled for ~1 N·m torques and ~0.15 m/s speeds,
  domain randomization including servo gain and torque-limit spread. Details in
  `docs/design/03-rl-environment.md`.
- **Model variant** `sim/cheetah_pup_rl.xml`: feet-only collisions, IMU frame sensors, foot
  contact sensors (work under the MJX JAX backend), `home` keyframe.
- **Verified on CPU**: 4 env tests pass (obs 49 / privileged 121, finite reset/step, slew limit,
  fall termination). PPO smoke run: 8 envs, 5,120 steps in 133 s (JIT-dominated), all reward
  terms finite, episodes run full length, checkpoint saved. Brax 0.14.2 needed a shim for the
  `jax.device_put_replicated` call JAX 0.10 removed (`cheetah_pup/rl/compat.py`).
- **Licensing note**: Open Duck Playground's env files carry Apache-2.0 headers even though the
  repo lacks a LICENSE file; ours is written against mujoco_playground's own Apache-2.0 code.

## 2026-09-04 — Design locked (A · M) and first MuJoCo validation (Fable 5.1)

- **Decision (owner, in chat)**: "All of the defaults look fine" on the DR-01 page → candidate
  **A · direct drive, size M, knees back**, baseline proportions (thigh 90 / shank 85 / abad link
  40 mm, hips 180 apart, abad axes 70 apart, shell 148 × 62 mm, hip height 120 mm) and gait
  defaults (60 mm step, 1.4 Hz, 25 mm swing). Recorded as `cheetah_pup.design.locked()` and
  `docs/design/locked.json`; every downstream artifact derives from it.
- **Sim model**: `cheetah_pup/mjcf.py` generates `sim/cheetah_pup.xml` — primitives with the
  component masses (1.409 kg total, matching the sizing), joint conventions verified against the
  kinematics library by test, STS3215 as a MuJoCo `general` actuator with kp 18.8 N·m/rad and
  back-EMF damping 0.56 N·m·s/rad from BAM's model, clamped at the datasheet stall (1.91 N·m);
  reflected inertia 0.026 kg·m² and Coulomb friction 0.05 N·m on the joints; 50 Hz control,
  2 ms physics.
- **Validation (open-loop IK targets, no balance feedback)** — `cheetah_pup/validate.py`,
  `sim/validation/`: stands with 0.8 mm sag and 0.75° droop, knee hold torque 0.244 N·m vs
  0.22 N·m quasi-static; walks 6 s without falling (0.10 m, pitch ≤ 10°); trots 6 s without
  falling (0.37 m, 0.062 m/s vs 0.168 commanded, pitch ≤ 8.5°, roll ≤ 10°). Servo torque
  saturates briefly at hip and knee during swing (stiff position loop hitting its 1.91 N·m clamp),
  which is what costs the commanded speed; peak joint speeds 3.4–5.1 rad/s, at the firmware cap.
  Conclusion: the locked geometry is viable on these servos; an RL policy has slack to work with.
- **Bug fixed on the way**: a `<general>` actuator needs `biastype="affine"` or MuJoCo drops the
  position/velocity terms and applies a constant feed-forward torque — the model collapsed at max
  torque until that was set. Keyframe trunk height now includes the foot radius.
- **DR-02 playback page** published: https://claude.ai/code/artifact/2db54b1d-2707-4034-895f-95bec2b86281
  (`docs/design/replay/`, built by `cheetah_pup.build_replay` from `sim/validation/`).
- **Why open-loop is slow**: servo tracking is good (mean error 2–3°, torque clamp hit only 1–4 %
  of samples); the front feet are simply off the ground ~80 % of the trot cycle because the trunk
  pitches under the rear legs' push. A 6 mm stance depth did nothing; a geometric leveling term
  (foot height ∝ hip x · pitch) swept from gain 0.2 to 1.0 buys speed (0.10 m/s at 0.7) at the cost
  of 18–27° attitude swings — proportional leveling through a laggy position loop oscillates. Kept
  gain 0.7 as the comparison run in DR-02. Conclusion for Phase 5: balance is the policy's job; the
  hand-written gait is only a viability probe.

## 2026-09-04 — Phase 1 candidates published (Fable 5.1)

- **Milestone**: DR-01 review page published with three leg-architecture candidates at true scale,
  animated gaits, live sizing, and decision capture. `docs/design/01-candidates.md` has the
  analysis; `cheetah_pup/` is the canonical design library (16 tests passing).
- **Servo geometry** taken from Open Duck Mini v2's STS3215 case meshes (45.22 × 24.72 × 35.7 mm,
  shaft 9.6 mm from the end, Ø20 drive horn, rear idler disc) and BAM's identified 7.4 V model
  (kt 1.18 N·m/A, R 2.48 Ω, armature 0.026 kg·m², velocity cap 5.29 rad/s, ~1 ms delay).
- **Findings**: servo speed cap is the binding gait constraint (trot ≤ ~1.5 Hz); direct-drive knee
  peaks at 40 % of stall in a trot, so no reduction is required; belt ratio limited to ≤ 1.25:1;
  Pi 5 mounts transversely; hip axis needs ≥ 16 mm clearance beyond the shell end; coaxial hip
  clusters add 70 mm of width.
- **Recommendation**: A (direct drive) — best stack reuse and speed margin; B if the owner wants the
  Mini Cheetah hip cluster and accepts the width and belt modeling.
- **Toolchain confirmed** in this sandbox: build123d 0.11.1 (CAD, Phase 2) and MuJoCo 3.12 (Phase 1
  sim) both install and run; Node 22 and headless Chromium available for page snapshots.
- **Open**: the owner's candidate/size selection (gate for the MJCF milestone).

## 2026-09-04 — Research, interview, and architecture decisions (Sonnet 5)

- Research on MIT Cheetah, scaled open-source QDD quadrupeds, the Hugging Face duck family, and
  legged-RL stacks (`docs/research-appendix.md`).
- Decisions from the owner interview recorded in `docs/HANDOFF.md` §3: STS3215-class smart servos
  over QDD; full 12-DOF Mini Cheetah topology; FDM + outsourced PCB assembly; cloud-GPU training from
  a Mac; scriptable CAD (Onshape unreachable from the sandbox); validate kinematics in sim before
  real CAD; $600–1,500 budget; likely open-source release (MIT license on this repo).
- Seven reference repos vendored under `vendor/` with license notes.
