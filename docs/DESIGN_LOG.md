# Design log

Dated record of decisions, milestones, and findings. Newest first. The decision table in
`docs/HANDOFF.md` §3 is the summary; this log keeps the reasoning and the state at the time.

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
