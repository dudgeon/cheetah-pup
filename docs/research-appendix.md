# Research appendix

Condensed findings from the research pass conducted 2026-09-04, before any design work began.
Four parallel research agents plus one follow-up covered: the original MIT Cheetah/Mini Cheetah,
scaled-down open-source QDD quadruped prior art, the Hugging Face duck robot family, and current
legged-RL training stacks. Everything here was verified via live web search / direct repo fetch,
not model memory — items that couldn't be confirmed are marked **unconfirmed** rather than
stated as fact. Treat unconfirmed items as things to re-verify before a purchasing or
architecture decision leans on them.

---

## 1. MIT Cheetah / Mini Cheetah (the geometry/kinematics reference)

| | Original Cheetah (2013) | Cheetah 3 (2019) | **Mini Cheetah (2019)** |
|---|---|---|---|
| Gear ratio | 5.8:1 custom planetary | 7.67:1 | **6:1 single-stage planetary (confirmed)** |
| DOF | 12 (unconfirmed for this gen specifically) | unconfirmed | **12 total — 3/leg × 4 (confirmed)** |
| Mass | ~32 kg (unconfirmed, single-source) | unconfirmed | **~9 kg / 20 lb (confirmed)** |
| Body dims | ~1m × 23cm × 70cm (unconfirmed) | unconfirmed | **48cm L × 27cm W × 30cm H** |
| Encoder | 13-bit magnetic | unconfirmed | **AMS AS5047 magnetic encoder** |
| Peak torque | ~10 Nm/motor | unconfirmed | **17 Nm peak / 6.9 Nm continuous** |
| Knee drive | Four-bar steel linkage from coaxial hip motor | Chain (single source, unconfirmed) | **Timing belt, 1:1, from hip-stacked actuator** |

**Leg architecture**: each Mini Cheetah leg is a **simple serial chain** — one ab/ad revolute
joint + a serial 2R (hip-pitch, knee-pitch) planar linkage — **not** a parallel/5-bar mechanism.
All three actuators per leg are physically stacked coaxially at the hip; the knee is driven
remotely via a 1:1 timing belt. (The original 2013 Cheetah *did* use a real four-bar parallel
linkage for its knee — this is generation-specific, not a constant Cheetah-family trait.)

**The actuator concept** ("proprioceptive actuator", Seok/Kim/Katz et al.): a high torque-density
motor + a deliberately **low** single-stage planetary reduction (kept low so it stays
backdrivable) + field-oriented current control, so ground-reaction force is inferred from motor
current rather than a force sensor or series-elastic element. This is genuine quasi-direct-drive
(QDD) — the whole reason Mini Cheetah can run, jump, and absorb impacts. Mini Cheetah's
controller is a custom PCB built around an **STM32F446** MCU running FOC, communicating over
**CAN** (also UART).

**Open-source status**: the control *software* stack (`mit-biomimetics/Cheetah-Software`, MIT
license) is open, but contains no actuator CAD/BOM/firmware. The actuator itself **is** open
source, via Ben Katz's own repos: `bgkatz/motorcontrol` (FOC firmware), `bgkatz/3phase_integrated`
(controller PCB — EAGLE + Gerbers), `bgkatz/mc-psu-v3` (power supply board) — all MIT licensed.
Katz's MIT **SM thesis** (not PhD — correcting a common mix-up), *"Low Cost, High Performance
Actuators for Dynamic Robots"* (2018), is the primary written source; his build blog covers the
earlier "HobbyKing Cheetah" DIY predecessor. A commercial derivative exists: CubeMars/T-Motor
**AK-series** actuators (e.g. AK80-6) package essentially the same 6:1/AS5047-class recipe as a
product, with an official MIT licensing relationship **unconfirmed**.

**Why this matters for us**: we adopted the 12-DOF serial-leg topology (ab/ad + hip + knee ×4)
as our kinematic reference, but *not* the QDD actuator technology — see decision log in
`docs/HANDOFF.md`.

---

## 2. Scaled-down open-source QDD quadruped prior art

Ranked by relevance to "MIT-Cheetah-style geometry + modern actuators + RL training":

### 🥇 ODRI Solo8 / Solo12 — strongest overall reference
Org: [open-dynamic-robot-initiative](https://github.com/open-dynamic-robot-initiative)
(LAAS-CNRS / Max Planck Institute / NYU / Naver Labs).

- **Actuator** (fully documented, from-scratch QDD build recipe): T-Motor **Antigravity 4004
  300kV** outrunner + **Broadcom AEDT-9810-Z00** optical encoder + dual-stage timing-belt
  reduction (3:1 × 3:1 = **9:1 total**, Conti Synchroflex AT3 belts), 2.5 Nm @ 12A, 150g/module.
- **DOF**: Solo8 = 8 DOF (hip + knee only, no ab/ad); **Solo12 = 12 DOF** (adds ab/ad) — 2.5 kg
  total mass, 3D-printed legs.
- **Electronics**: central **Master Board v2 is ESP32-based** (not STM32), talks to up to 8
  motor-driver boards over **SPI**. The per-motor driver boards are TI-based: the original
  µDriver v2 uses a **TMS320F28069 C2000 DSP + dual DRV8305** gate drivers; the newer
  `open-motor-driver-initiative` board uses a newer C2000 part (TMS320F2838x) that supports
  CAN-FD.
- **License**: BSD-3-Clause (`solo`, `open_robot_actuator_hardware`), BSD-2-Clause
  (`master-board`) — all confirmed directly from each repo's LICENSE file.
- **RL**: real training code exists and is released — [Gepetto/soloRL](https://github.com/Gepetto/soloRL),
  companion to *"Controlling the Solo12 Quadruped Robot with Deep RL"* (Nature Sci. Reports 2023,
  arXiv 2309.16683). Uses **RaiSim** (not Isaac Gym — an earlier synthesis pass surfaced an
  Isaac-Gym/RTX-4090/Raspberry-Pi-5 claim that could not be verified and appears to be a
  hallucinated mixup; disregard it). Real-robot policy code lives in a companion GitLab repo,
  reported 10μs on-robot inference. Published, real sim-to-real result.

### 🥈 Stanford Pupper v3 — strongest secondary reference (system integration)
Code: [Nate711/pupperv3-monorepo](https://github.com/Nate711/pupperv3-monorepo); docs:
[HandsOnRobotics/pupper-v3-documentation](https://github.com/HandsOnRobotics/pupper-v3-documentation)
(docs site blocked from our sandbox — re-verify directly before relying on it).

- **Actuators**: 12× **Steadywin GIM4305** — an integrated, buy-not-build QDD module (4005 BLDC +
  10:1 planetary + AS5047 14-bit encoder + onboard CAN driver), 24V, ~1 Nm continuous / ~3.5 Nm
  stall, 400W class.
- **Compute**: Raspberry Pi 5 + Hailo AI accelerator + Luxonis depth camera; ROS2 workspace with
  a `neural_controller` package. Claims "RL locomotion out of the box" — **whether full RL
  *training* code (vs. just on-robot inference) is public is unconfirmed.**
- **License**: **GPL-3.0** (confirmed) — copyleft, unlike Pupper v1/Doggo's MIT. Cost ~$1000 BOM.

### Stanford Doggo — kinematics reference only
[Nate711/StanfordDoggoProject](https://github.com/Nate711/StanfordDoggoProject). T-Motor MN5212
gimbal motors (2/leg, coaxial) + AS5047P encoders + ODrive v3.5 + Teensy 3.5. Genuine **5-bar/SCARA
parallel linkage**, 2 DOF/leg (8 total) — the best-documented small QDD leg-linkage math found in
this survey. Fusion 360 CAD, MIT license. Open-loop sinusoidal gaits only, no RL. End-of-life;
Stanford points people to Pupper v3 now.

### Stanford Pupper v1 — superseded
[stanfordroboticsclub/StanfordQuadruped](https://github.com/stanfordroboticsclub/StanfordQuadruped).
12× JX Servo CLS6336HV hobby PWM servos (not QDD), Raspberry Pi, MIT license, ~$600-1000,
end-of-life.

### MangDang Mini Pupper — ROS2/vision reference only
[mangdangroboticsclub](https://github.com/mangdangroboticsclub). Proprietary 12.5g custom PWM
servo (not QDD) via PCA9685, Raspberry Pi 4/CM4 (+ ESP32 co-processor on v2), full ROS1/ROS2 +
Nav2 + YOLO11 stack. Apache-2.0 (current repo) / MIT (legacy repo). No RL training code found
anywhere in the org. ~$550-800 kit.

### Other references (brief)
**OpenQuadruped/"Rex"** ([adham-elarabawy/open-quadruped](https://github.com/adham-elarabawy/open-quadruped),
MIT) — 12-DOF hobby-servo quadruped with a genuinely custom single PCB (Gerbers included) and a
published 3-DOF IK derivation; RL was only a TODO, never implemented. Standalone QDD actuator
reference designs (not full quadrupeds) worth knowing about for future custom-actuator work:
**OpenTorque**, **SpryDrive**, **OpenQDD**, **NOMAD BLDC**.

---

## 3. Hugging Face duck robots — two distinct projects

**Important**: "the Hugging Face mini duck project" is actually two related but distinct things.

### Open Duck Mini v2 — open-hardware DIY project (our actual hardware reference)
Created by **Antoine Pirrone** (`apirrone`, Pollen Robotics R&D engineer, Rhoban RoboCup team
member), sponsored (funded/promoted) by Hugging Face and Pollen Robotics — a community project,
not an official HF-authored repo.

- **Repos** (all vendored — see `vendor/README.md`): hardware/BOM
  ([Open_Duck_Mini](https://github.com/apirrone/Open_Duck_Mini), **Apache-2.0**), runtime
  ([Open_Duck_Mini_Runtime](https://github.com/apirrone/Open_Duck_Mini_Runtime), **no license**),
  RL training ([Open_Duck_Playground](https://github.com/apirrone/Open_Duck_Playground), **no
  license**), reference motion
  ([Open_Duck_reference_motion_generator](https://github.com/apirrone/Open_Duck_reference_motion_generator),
  **no license**).
- **Actuators**: 14× **Feetech STS3215** serial-bus smart servos, 7.4V, ~19 kg·cm stall,
  daisy-chained on a TTL serial bus. Two 5-DOF legs + 4-DOF neck/head, plus small 9g hobby servos
  for expressive parts (exact count/use not fully itemized in sources reviewed).
- **Control board / compute**: **Raspberry Pi Zero 2 W** (Runtime repo also supports Pi 5), 64-bit
  Pi OS Lite. Servos driven via a commercial **Waveshare "Bus Servo Adapter (A)"** breakout — UART
  RX/TX to Pi GPIO. **No dedicated custom PCB in the main repo.**
- **Power**: 2× 18650 Li-ion, **2S pack + 2S BMS** (~7.4V nominal — matches servo voltage
  directly), separate **5V UBEC** for the Pi/logic, XT30 connector, USB-C charge module.
- **Sensors**: single **Bosch BNO055** 9-DOF IMU, Raspberry Pi Camera Module, per-foot contact
  switches, **Adafruit MAX98357 I2S** amp + speaker for expression sounds.
- **Training**: **MuJoCo** via `Open_Duck_Playground`, built on/inspired by DeepMind's
  `mujoco_playground` (JAX/Brax-based PPO). Reference motion is **procedurally generated** (not
  motion-captured) using the **Placo** IK library, with imitation-reward design credited as
  inspired by Disney Research's BDX droid work.
- **Released models**: pretrained ONNX walk policies (`BEST_WALK_ONNX*.onnx`) ship directly in the
  hardware repo.
- **Cost**: BOM ≈ €347–414 (~$375–450), under the project's own "$400" target. Build is
  nontrivial (soldering, servo ID/calibration, RPi setup) — described anecdotally as multiple
  days of work.

### Microduck — commercial product (not directly reusable as a hardware base)
Hugging Face acquired Pollen Robotics in **April 2025**; Microduck (Aug 27 2026, $399 assembled)
is their second post-acquisition product (after Reachy Mini). Same creative lineage as Open Duck
Mini, **different hardware, not published as open hardware**.

- **Repos**: firmware/"brain" in Rust ([pollen-robotics/microduck](https://github.com/pollen-robotics/microduck),
  Apache-2.0), RL training ([pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl),
  **Apache-2.0 code, but 3D model/asset files are CC BY-SA-NC — non-commercial**). No hardware
  CAD/BOM/PCB published — confirmed by the existence of a community "reverse-engineering" repo
  ([fanhao375/microduck-replica](https://github.com/fanhao375/microduck-replica)) built
  specifically because official files don't exist.
- **Actuators**: "15 motors" per press coverage; exact make/model **not officially published**
  and **unconfirmed** (community analysis argues STS3215 is physically too large for Microduck's
  800g/25cm form factor, implying a different/smaller/custom servo).
- **Compute**: **Rockchip RK3566 SoC + AI accelerator**, 1GB RAM, 32GB storage — confirmed from
  the `microduck` repo README. Rust daemon architecture (`robotd`, `mediad`, `tofd`, `configd`,
  `btd`, `padd`, `updaterd`) over Unix-socket JSON-RPC.
- **Power**: removable NP-F550-style camera battery, ~1hr runtime. Internal regulation
  unconfirmed.
- **Sensors**: **two IMUs**, wide-angle front camera (WebRTC), **8×8 ToF array**, mic + speaker, 2
  NFC antennas.
- **Training**: a **different** stack from Open Duck Mini v2 — **mjlab** (Isaac-Lab-style API over
  GPU-accelerated MuJoCo Warp) + **rsl_rl** (PyTorch PPO), with extensive domain randomization
  (battery sag, command latency, friction, ±1° per-joint backlash) via Rhoban's **BAM**. ~1–2
  hours training on one CUDA GPU at 4096 parallel envs. Onboard inference at 50Hz, ONNX export.
- **Released models**: [huggingface.co/pollen-robotics/microduck-policies](https://huggingface.co/pollen-robotics/microduck-policies)
  (Apache-2.0 ONNX policies: walk/sit/stand/kick/grab/roller-skate/self-recovery) and a browser
  sim sandbox at [huggingface.co/spaces/pollen-robotics/microduck-simulator](https://huggingface.co/spaces/pollen-robotics/microduck-simulator).

**Practical takeaway**: we build on Open Duck Mini v2's *open hardware* (servos, power topology,
sensor choices) while adopting Microduck's *newer training approach* (BAM-based actuator
identification; mjlab as a possible upgrade path once real GPU training is underway).

---

## 4. Legged-robot RL training stacks (2025–2026 survey)

| Stack | Physics engine | HW acceleration | Learning curve | Notable users | Relevance here |
|---|---|---|---|---|---|
| **MuJoCo Playground** | MuJoCo via MJX (JAX) | Any JAX-capable GPU, even CPU-runnable | Low — `pip install playground` | Google DeepMind, Berkeley Humanoid, Unitree Go1/G1, Open Duck Mini | **High** — lightest hardware bar, same lineage Open Duck Mini itself uses |
| **Isaac Lab** | Isaac Sim/PhysX, adding Newton/MJWarp backends | Massively parallel NVIDIA GPU (RTX 4080-class officially, RTX 4060 8GB reported working for lighter loads) | Steep — Omniverse/USD, Docker-heavy | Boston Dynamics Spot, Unitree, ANYbotics | Medium — powerful, heavier install than needed here |
| **Genesis** | Custom multi-physics | Single high-end GPU, very fast (claims up to 43M FPS on RTX 4090) | Low-medium, rough edges | Early-stage adopters | Low-Medium — thin real-hardware sim-to-real track record, open stability bugs as of Aug 2026 |
| **legged_gym + rsl_rl** | Isaac Gym Preview (deprecated) | NVIDIA GPU | Medium | ANYmal (origin), `unitree_rl_gym` | The env scaffolding (`legged_gym`) is legacy; **`rsl_rl` the PPO library is still the de facto standard**, integrated into Isaac Lab, Genesis forks, mjlab |
| **Brax / MJX** | MJX = JAX reimplementation of MuJoCo core | GPU/TPU via JAX | N/A (substrate) | Underlies MuJoCo Playground | Physics layer, not a standalone framework |
| **mjlab** (new, early 2026) | MuJoCo Warp (MJWarp) — NVIDIA-Warp-accelerated, co-built by DeepMind/NVIDIA/Disney Research ("Newton" project) | Requires NVIDIA GPU | Isaac-Lab-style ergonomic API | `unitree_rl_mjlab`, **Pollen Robotics' `microduck_rl`** | **This is what Microduck itself actually uses now** — closest "same stack" match if we have GPU access |

**Sim-to-real for cheap small robots**: DeepMind's **Barkour** paper is the gold-standard
methodology reference (specialist RL skills distilled into a generalist transformer) though the
hardware isn't open. **Solo12**'s published result found only *simple* domain randomization
needed for zero-shot transfer, attributed to its low-inertia, high-bandwidth QDD actuators
narrowing the sim gap — this doesn't directly apply to our smart-servo choice, where
**backlash/friction modeling matters more** (which is exactly what BAM addresses). **Berkeley
Humanoid Lite** is another sub-$5k open reference. Standard control pattern across all of these:
PD position-target per joint, policy at 50Hz, faster low-level tracking loop underneath.

**What's reusable across engine choice, regardless of which one we land on**: the BAM
actuator-identification workflow, domain-randomization code structure, ONNX export + on-robot
inference runtime pattern, PPO policy architecture (small MLP, indifferent to leg count beyond
I/O dimensions), and generic reward-shaping utilities. **What must be rebuilt regardless**:
MJCF/URDF geometry, action-space size/order (12-DOF quadruped vs. the duck's bipedal layout),
gait reference motion (quadruped trot ≠ bipedal walk), per-robot reward terms, and
domain-randomization *ranges* (a fresh BAM identification pass on our actual robot, even reusing
their STS3215 baseline as a starting point).

---

## 5. Side research: "NVIDIA AI Router" (low-priority, informational only)

Checked because the user asked whether it could ease Mac → cloud-GPU access for training. The
real product is **NVIDIA PAIR (Personal AI Router)**, announced **2026-09-03** (literally the day
before this research) at IFA — open source, [NVIDIA/Personal-AI-Router](https://github.com/NVIDIA/Personal-AI-Router).

**It doesn't help with our use case.** PAIR pools multiple machines on your **local/home network**
for **inference** requests (Ollama/LM-Studio-compatible endpoints) — it explicitly does not pool
GPU memory or reach cloud GPUs, and it's inference-only, not training. A Mac can be a participant
*node* in your own house, not a client reaching remote NVIDIA GPUs. For actually offloading
training to the cloud, the more relevant (and **not deeply verified today** — flagged for the
low-priority follow-up task) NVIDIA offering would be **DGX Cloud / DGX Cloud Lepton**, a GPU
marketplace — a distinct, older product line unrelated to PAIR.
