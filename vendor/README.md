# Vendored reference repos

These are git submodules: external repos pinned at a specific commit for local reading,
adaptation reference, and (where license allows) code reuse. They are **not** automatically
part of our build — nothing here is imported/compiled by default. Each one earned its place
in the research pass documented in `docs/research-appendix.md`; see there for full detail.

Clone with submodules:

```
git clone --recurse-submodules <this-repo-url>
# or, if already cloned:
git submodule update --init --recursive
```

## What's here and why

| Path | Upstream | License | Why it's here | Usage rule |
|---|---|---|---|---|
| `open_duck_mini` | [apirrone/Open_Duck_Mini](https://github.com/apirrone/Open_Duck_Mini) | **Apache-2.0** | The open-hardware DIY robot HF/Pollen Robotics sponsored. Our actuator choice (Feetech STS3215) and general power/sensor architecture are drawn from this repo's BOM/CAD. Also ships example ONNX walk policies for reference. | Free to read, adapt, and reuse code per Apache-2.0 (keep attribution/license notice on any copied code). |
| `open_duck_mini_runtime` | [apirrone/Open_Duck_Mini_Runtime](https://github.com/apirrone/Open_Duck_Mini_Runtime) | **⚠️ none found** (no LICENSE file — all-rights-reserved by default) | The on-robot control loop (Pi-side): reads sensors, drives the servo bus, runs the ONNX policy. Best available reference architecture for our own runtime. | **Read for architecture/approach only. Do not copy code verbatim** into our repo until we get explicit licensing terms from the author (Discord: `discord.gg/UtJZsgfQGe`). Reimplement independently, informed by what we read here. |
| `open_duck_playground` | [apirrone/Open_Duck_Playground](https://github.com/apirrone/Open_Duck_Playground) | **⚠️ none found** | The RL training environment (MuJoCo Playground-based, JAX/PPO) used to train Open Duck Mini's policies. Reference for env structure, domain randomization, ONNX export pipeline. | Same caveat as above — reference only pending license clarification. |
| `open_duck_reference_motion_generator` | [apirrone/Open_Duck_reference_motion_generator](https://github.com/apirrone/Open_Duck_reference_motion_generator) | **⚠️ none found** | Procedurally generates reference gaits (via the Placo IK library) used as imitation targets during training. Our quadruped will need an analogous — but new — trot/walk generator; this is a design reference, not directly reusable (bipedal duck gaits don't transfer to a quadruped trot anyway). | Reference only pending license clarification. |
| `odri_solo` | [open-dynamic-robot-initiative/solo](https://github.com/open-dynamic-robot-initiative/solo) | **BSD-3-Clause** | Solo8/Solo12 kinematics and control software. Solo12's 12-DOF (hip ab/ad + hip pitch + knee ×4) leg topology is the closest published prior art to the true Mini Cheetah layout we're using. Good reference for URDF structure, joint limits, and leg IK. | Free to reuse per BSD-3. |
| `odri_actuator_hardware` | [open-dynamic-robot-initiative/open_robot_actuator_hardware](https://github.com/open-dynamic-robot-initiative/open_robot_actuator_hardware) | **BSD-3-Clause** | ODRI's actuator + motor-controller PCB hardware. We are **not** using their QDD actuator (we chose smart servos), but their board architecture — power distribution, connector choices, sensor breakout patterns — is a strong reference for our own custom PCB even though the circuits themselves differ. | Free to reuse per BSD-3. |
| `bam` | [Rhoban/bam](https://github.com/Rhoban/bam) | **Apache-2.0** | "Better Actuator Models" — friction/backlash identification pipeline for servo actuators, built specifically for RL sim-to-real. **Already ships pre-identified parameters for Feetech STS3215 @ 7.4V** (`bam/params/feetech_sts3215_7_4V/`) and a `waveshare` actuator module — i.e. it already models close to our exact hardware. This is what Microduck's own training stack (`microduck_rl`) uses for actuator realism. | Free to reuse per Apache-2.0. This is a direct dependency, not just a reference — plan to `pip install` it or call into it directly during training (Phase 5). |

## Not vendored (and why)

- **`pollen-robotics/microduck` / `microduck_rl`** — Microduck's actual hardware CAD/BOM/PCB were never published, so there's no hardware to reference. Its RL code is Apache-2.0, but the 3D model/asset files are **CC BY-SA-NC (non-commercial)** — since we're using a different (open) hardware base anyway, we skip vendoring this to avoid any accidental dependency on NC-licensed assets. Its training-stack *approach* (mjlab + rsl_rl + BAM) is documented in `docs/research-appendix.md` instead.
- **`google-deepmind/mujoco_playground`, `mujocolab/mjlab`** — these are real Python libraries meant to be installed as pinned dependencies (`pip`/`uv`), not vendored as submodules. See `docs/HANDOFF.md` Phase 1 / Phase 5 for which one to install when.
- **`open-dynamic-robot-initiative/master-board`, `Gepetto/soloRL`** — lower-priority references (ESP32 motor-driver board specific to ODRI's QDD actuator we're not using; a narrower research-paper artifact with unconfirmed licensing). Noted in the research appendix if needed later.
