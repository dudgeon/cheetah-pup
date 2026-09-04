# Cheetah Pup — Handoff Document

**Status** (2026-09-04): Phase 1 nearly complete. Research done, architecture decided, design
library built and tested, DR-01 candidates reviewed, **design locked (A · M)**, MuJoCo model
generated and validated open-loop (stands, walks, trots). **Next: the RL environment** so training
can start on cloud GPU.

## 0. Progress

| Phase | State | Where |
|---|---|---|
| 0 · Repo & tooling | done | this repo; `vendor/README.md` |
| 1 · Kinematic validation | **design locked; sim model validated (DR-02); RL environment in progress** | `docs/design/locked.json`, `cheetah_pup/mjcf.py`, `sim/cheetah_pup.xml`, `docs/design/02-sim-validation.md`; DR-01 review https://claude.ai/code/artifact/6b9c92f0-98d5-4cf1-928c-98a28d699ba4; DR-02 playback https://claude.ai/code/artifact/2db54b1d-2707-4034-895f-95bec2b86281 |
| 2 · Mechanical CAD | not started (build123d confirmed working here) | — |
| 3 · Electrical / PCB | not started; volumes and power rails fixed in the design library | `cheetah_pup/electronics.py` |
| 4 · Firmware | not started | — |
| 5 · RL training | not started; STS3215 actuator model available from BAM | `vendor/bam` |
| 6 · Bring-up | not started | — |

Keep `docs/DESIGN_LOG.md` current: one dated entry per decision or milestone.

**Purpose of this document**: everything needed to pick up this project cold and start the
detailed spec-and-build phase, without re-deriving decisions already made or re-running research
already done. If you're Fable 5.1 (or any future session) starting fresh: read this document in
full before writing any code, then start at [§8 Immediate next steps](#8-immediate-next-steps).

---

## 1. Project vision

An experimental quadruped robot for reinforcement-learning research:

- **Body geometry and leg kinematics** based on MIT's Mini Cheetah — full 12-DOF topology
  (hip abduction/adduction + hip pitch + knee pitch, ×4 legs), serial 2R leg chain per the real
  robot (not a parallel/5-bar linkage).
- **Actuators and much of the surrounding stack** based on Hugging Face/Pollen Robotics'
  **Open Duck Mini v2** — Feetech STS3215-class serial-bus smart servos, similar power
  architecture, similar sensor stack, and a training approach descended from the same lineage
  (MuJoCo-based RL, with Microduck's newer BAM-based actuator-identification approach folded in).
- A **custom PCB** for motor-bus, power, and sensor breakout (Open Duck Mini itself uses only an
  off-the-shelf Waveshare adapter — there is no existing open custom board for this actuator
  class doing exactly what we need, which is genuinely new work, not just a port).
- Scaled for a **hobbyist build**: FDM-printed structure, outsourced PCB assembly, Raspberry-Pi-class
  compute, cloud-GPU training.

This deliberately does **not** replicate the real Mini Cheetah's actuator technology (custom
quasi-direct-drive BLDC + FOC motor control). See §3 for why, and what that trade-off costs us.

---

## 2. How to use this document

- **§3** is the decision log — what was decided, why, and what alternatives were rejected. Don't
  relitigate these without a good reason; if you find one, update this log rather than silently
  diverging.
- **§4–§7** is the actual plan: architecture, phased build sequence, BOM/budget, safety.
- **§8** is where to start.
- **§9** is what's deliberately left open for you to resolve, with pointers to how.
- Full research backing every claim below lives in `docs/research-appendix.md` — this document
  states conclusions; that one shows the work.
- Reference repos are vendored as git submodules under `vendor/` — see `vendor/README.md` for
  what's there, exact licenses, and usage rules (a few are reference-only pending license
  clarification — don't copy code from those verbatim).

---

## 3. Decision log

These were made through direct interview with the project owner (dudgeon) on 2026-09-04, after
the research pass. Each is a real trade-off, not an arbitrary choice — the "why" matters if you
ever need to revisit one.

| Decision | Chosen | Alternatives considered | Why |
|---|---|---|---|
| **Actuators** | Duck-style smart servos (Feetech STS3215-class, serial bus, position-controlled) | True QDD BLDC (MIT-Cheetah/Solo12-style); hybrid | Matches the literal ask ("same actuators as the HF duck project"), reuses the most existing hardware/software, and is dramatically more tractable for a hobbyist build — no custom FOC motor-controller firmware needed. **Trade-off, stated explicitly and accepted**: not backdrivable, not true torque control. This robot will be a confident, capable *walker*, not a bounding/jumping *runner*. If dynamic gaits become a real goal later, that's a distinct future initiative (see §9), not a retrofit. |
| **Leg DOF** | Full 12-DOF (ab/ad + hip + knee ×4), matching real Mini Cheetah | Simplified 8-DOF (Solo8-style, sagittal only) | Explicitly requested as most faithful to "MIT Cheetah kinematics"; enables turning in place and lateral push recovery that an 8-DOF design can't do well. Costs 4 more servo channels/wiring runs than the simpler option. |
| **Fabrication access** | 3D printer (FDM) at home; PCB assembly outsourced (JLCPCB/PCBWay-style) | Hand-soldering fine-pitch SMD | **PCB design must use assembly-service-friendly footprints** (no 01005/BGA-if-avoidable, standard connectors) since hand-soldering fine-pitch SMD is not a confirmed skill. Structure/legs are FDM-printable — matches nearly every reference project surveyed. |
| **Training compute** | Cloud GPU credits, macOS as the dev/orchestration machine; local GPU explicitly deferred | Local NVIDIA GPU now; CPU-only | Not compute-constrained for the training stack choice — both MuJoCo Playground and the GPU-hungrier mjlab are viable. A **low-priority backlog item** (see §9.5) tracks revisiting local/hybrid GPU augmentation if cloud spend gets high; NVIDIA's brand-new "PAIR" project (announced 2026-09-03) was checked and does **not** help here — it's LAN-only inference pooling, not cloud training offload. |
| **CAD toolchain** | Code-first: **build123d** (or CadQuery as fallback) for parametric geometry + hand-authored **MJCF** for the early kinematics validation | Onshape | **Onshape is confirmed unreachable from this sandbox** (network egress test returned 403 on the CONNECT tunnel — the environment's proxy blocks it, same pattern the research agents hit on most non-GitHub domains). Since the owner wants Claude/Fable operating end-to-end from a Claude Code cloud instance without needing GUI CAD access, and wants STEP files that import cleanly into **Fusion 360** for their own later refinement pass, a scriptable Python CAD library is the only toolchain that satisfies both constraints. **Verify current pip-installability of build123d at the start of Phase 2** — if it has issues, CadQuery is the established fallback; both export STEP. |
| **Milestone sequencing** | Validate kinematics/topology in a *primitive-geometry* MuJoCo sim **before** investing in real CAD or manufacturability | Design real CAD first | The owner's own framing, adopted as-is: "a relatively early milestone should be validating a simple version of the design and kinematics in the RL environment to see if the basic project will work... and we will later refine, think about manufacturability." This is Phase 1 below, and it deliberately needs no CAD-derived meshes — simple boxes/capsules are enough to test whether the topology, proportions, and torque budget are sound. |
| **Experience / division of labor** | Claude/Fable drives ~95% of CAD, 100% of PCB/firmware/RL design; owner has CAD/DFM experience (not advanced linkages) and will do manufacturability refinement in Fusion 360 later | — | Stated directly. Electronics, embedded firmware, and ML/RL were **not** selected as areas of existing hands-on experience — treat those as fully Claude/Fable-driven too, not just CAD. The plan below is written assuming heavy autonomous execution with clear checkpoints for physical assembly (the one thing that must be done by human hands) and design review. |
| **Budget** | $600–$1,500 (excluding the 3D printer already owned) | — | See §6 for a rough BOM sanity check against this. |
| **License / distribution intent** | Personal project, **likely to open-source publicly** | Pure personal use; commercial | Drives §9.2 (unlicensed-repo caveats) and the recommended MIT license on this repo's own code (added — see root `LICENSE`). Since we're not depending on Microduck's CC BY-SA-NC assets at all (different hardware base), that restriction doesn't currently constrain us — but don't pull anything from `pollen-robotics/microduck_rl`'s asset files later without re-checking. |
| **Servo data source** (Phase 1) | STS3215 geometry measured from Open Duck Mini v2's case meshes; electrical/friction model from BAM's identified 7.4 V parameters | datasheet values from memory | Measured meshes and an identified model beat a remembered datasheet; both are vendored (`vendor/open_duck_mini`, `vendor/bam`) and encoded in `cheetah_pup/servo.py`. Design margins use the *datasheet* stall (1.91 N·m), not BAM's more optimistic implied stall. |
| **Sizing design point** (Phase 1) | Trot at 1.4 Hz, 60 mm step, 25 mm swing; loads = ¼ weight standing (≤ 25 % of stall) and ½ weight × 1.5 dynamic at trot peak (≤ 60 % of stall) | 1.6 Hz trot | 1.6 Hz pushed the knee past the STS3215's 5.29 rad/s cap even in direct drive. The cap, not torque, limits gait speed with these servos — accepted as the walker trade-off. |
| **Electronics layout** (Phase 1) | Two layers: abad servos + battery low, Pi 5 (transverse) + PCB high | Pi lengthwise | Transverse Pi saves ~40 mm of body length and keeps hip-to-hip near Mini Cheetah's 1.82× thigh ratio (2.0×). |
| **Knee-drive architecture & size** (locked 2026-09-04) | **A · direct drive, size M**, knees back, baseline proportions and gait — `cheetah_pup.design.locked()`, `docs/design/locked.json` | B coaxial + belt; C coaxial + pushrod; sizes S/L | Owner accepted the DR-01 defaults. A is the lowest-risk path and the best fit for reusing Open Duck Mini's directly-driven-joint sim-to-real pipeline; 40 % of stall at trot peak, 76 % of the speed cap, narrowest hips. |
| **Sim actuator model** (Phase 1) | MuJoCo `general` PD from BAM's electrical model (kp 18.8 N·m/rad, kd 0.56 N·m·s/rad), clamped at the datasheet stall 1.91 N·m; joint armature 0.026 kg·m², Coulomb 0.05 N·m | BAM's implied 3.4 N·m stall; BAM's full stateful model | Conservative clamp for viability; `--bam` flag generates the optimistic variant. The RL environment should move to BAM's MuJoCo integration (rate-limited target, extended friction) for the real training runs. |

---

## 4. Architecture overview

### 4.1 Mechanical
12-DOF quadruped, Mini-Cheetah-proportioned (long body relative to leg length, legs tucked
under the body rather than sprawled like a hobby "SpotMicro"-style bot). Actuators mounted at
the hip where possible (co-axial stacking, like Mini Cheetah/Open Duck Mini) to keep leg swing
mass — and thus reflected inertia the servo has to fight — low, even though we're not chasing
true QDD dynamics. Final scale/mass to be derived from servo torque budget during Phase 1 (see
§5), not fixed in advance — expect something in the rough neighborhood of Open Duck Mini's
footprint (that robot closes the loop on "how big can a 7.4V STS3215-driven robot be"), adapted
for quadruped proportions.

### 4.2 Electrical
- **Servos**: 12–14× Feetech STS3215-class, 7.4V, on a shared serial bus (final servo count and
  whether any joints need a higher-torque variant depends on Phase 1 torque analysis).
- **Custom PCB** (the "motor and sensor breakout" from the original brief): distributes bus +
  power to per-leg connectors (4 runs, not one long daisy chain through everything — more
  robust, easier to service than the stock Waveshare adapter's single chain), regulates 2S
  Li-ion (~7.4V, matches servo voltage directly with no separate boost/buck needed for the
  servo rail) down to 5V for compute and 3.3V for sensors, breaks out I2C for the IMU and room
  for future sensors (ToF, foot contact), includes battery voltage/current sensing, a fuse +
  reverse-polarity protection, and a **software+hardware E-stop** (MOSFET-switched servo power
  rail, killable independently of the Pi) — see §7 Safety.
- **Power**: 2S 18650 Li-ion pack + 2S BMS, mirroring Open Duck Mini v2 exactly (their choice
  already validates this voltage matches STS3215 servos directly).
- **Compute**: Raspberry Pi 5 (more headroom than Open Duck Mini's Pi Zero 2W — Pupper v3 made
  the same call for similar reasons: room for a camera and future sensor/ML work without being
  compute-starved). Pi Zero 2W remains a fallback if size/weight pressure shows up in Phase 1/2.
- **Sensors**: Bosch BNO055 9-DOF IMU at minimum (matches Open Duck Mini's choice, and BAM
  already models this class of setup); camera and/or foot-contact sensing as stretch additions
  once core locomotion works.

### 4.3 Software / training
- **Phase 1 (kinematics validation)**: quickest-to-stand-up MuJoCo setup (plain MuJoCo or MuJoCo
  Playground) on a hand-authored primitive-geometry MJCF. Goal is a yes/no answer on topology
  and rough torque budget, not a polished policy.
- **Phase 5 (production training)**: given cloud GPU is available (not the hobbyist
  hardware-constrained case Playground is optimized for), lean toward **mjlab** — it's the stack
  Microduck itself actually runs today, giving the closest "same stack as the HF project" match,
  and it comes with `rsl_rl` + BAM integration already proven together. Keep MuJoCo Playground as
  the fallback if mjlab setup friction becomes a blocker — both are documented in
  `docs/research-appendix.md` §4.
- **Actuator realism**: `vendor/bam` already ships identified friction parameters for Feetech
  STS3215 @ 7.4V — start from those, then re-identify against our actual robot once hardware
  exists (BAM's pipeline is built for exactly this).
- **Reference motion / reward shaping**: cannot reuse Open Duck Mini's bipedal reference-motion
  generator directly (quadruped trot ≠ bipedal balance), but the *approach* (procedural
  generation via IK rather than mocap, using the Placo library) is worth adapting — quadruped
  trot/walk cycles are kinematically simpler than bipedal balance, so this should be less work,
  not more.
- **Deployment**: ONNX export, ~50Hz on-robot policy execution — matches both Open Duck Mini and
  Microduck's control rate, a reasonable default rather than something to re-derive.

---

## 5. Phased build plan

Each phase lists its goal, key tasks, and what "done" looks like. Phases are sequential but not
rigidly gated — e.g. PCB design (Phase 3) can start once Phase 1 gives rough current/voltage
requirements, without waiting for Phase 2's CAD to be finished.

### Phase 0 — Repo & tooling scaffolding ✅ *done this session*
Research, submodules, this document, initial repo structure, MIT license.

### Phase 1 — Kinematic/topology validation in sim
**Goal**: answer "does the basic idea work" as cheaply as possible, before investing in real CAD.
- Hand-author (or script) a primitive-geometry MJCF: 12-DOF topology, reasonable guessed link
  lengths/masses based on Mini Cheetah's proportions scaled to a target mass, box/capsule
  collision geometry — no real meshes needed yet.
- Stand up a MuJoCo training loop (plain MuJoCo or MuJoCo Playground — whichever is faster to get
  running) and get a policy learning to stand, then walk, even crudely.
- Pull in `vendor/bam`'s STS3215 @ 7.4V parameters early — cheap to include, makes this
  validation more realistic than idealized actuators would.
- Check peak/continuous torque demand from the trained policy against STS3215's ~19 kg·cm rating
  at the chosen leg geometry — this is what actually determines final size/mass, not the other
  way around.
- **Done when**: a policy reliably stands and achieves basic forward locomotion in sim, within
  the chosen servos' torque/speed envelope. If it can't, that's a real finding — iterate on
  proportions/mass before moving on, not after building hardware.

### Phase 2 — Mechanical CAD & manufacturability
- Translate the validated topology into parametric CAD (build123d, scripted), incorporating real
  STS3215 mounting geometry/servo horns, bearing points, wire routing, battery/compute/PCB
  volume, and DFM for FDM printing (wall thickness, overhangs, print orientation, split lines for
  bed size).
- Export STEP; hand off to the project owner for a Fusion 360 refinement pass (aesthetics,
  manufacturability polish, anything requiring their DFM judgment).
- Re-derive MJCF collision geometry and mass properties from the real CAD model; re-run Phase 1's
  validation as a sim-to-sim consistency check before moving on.

### Phase 3 — Electrical: power, sensors, custom PCB
- Schematic + layout (KiCad) for the motor-bus/power/sensor breakout board described in §4.2.
- Design for outsourced assembly: standard footprints, no hand-soldering-only parts.
- Reference `vendor/odri_actuator_hardware` for board architecture patterns (connector choices,
  power distribution) even though the underlying actuator circuit is different.
- Order a small-batch assembled run.

### Phase 4 — Firmware & on-robot integration
- Raspberry Pi 5 control loop: sensor reads (IMU, later foot contact), servo bus driver, ONNX
  policy execution at ~50Hz, safety/torque-limiting, telemetry/logging.
- Architecture informed by reading `vendor/open_duck_mini_runtime` — **reimplement
  independently**, don't copy verbatim (see license caveat in §9.2).

### Phase 5 — Full RL training with real actuator ID + domain randomization
- Re-run BAM identification against the actual assembled robot's servos (starting from the
  vendored STS3215 @ 7.4V parameters).
- Build the full domain-randomization suite (friction, mass, latency, battery sag, backlash) and
  quadruped-specific reward shaping (trot symmetry, foot clearance, etc.).
- Train in mjlab (or MuJoCo Playground — see §4.3) on cloud GPU; export ONNX.

### Phase 6 — Bring-up & iteration
- Assemble, flash, deploy. Tethered/propped-up testing first, progressive gait validation (stand
  → weight shift → step → walk → trot). Diagnose and close the sim-to-real gap.

### Phase 7 — Stretch goals (not in current scope)
Higher-speed gaits, vision-based navigation/obstacle avoidance, additional sensing. **True
dynamic gaits (bounding, jumping) are explicitly out of scope** under the current smart-servo
actuator decision — revisiting that would mean reopening §3's actuator decision, not extending
this phase.

---

## 6. Rough BOM / budget sanity check

Order-of-magnitude only — final part selection happens in Phases 1–3. Excludes the 3D printer
(already owned).

| Category | Rough estimate | Basis |
|---|---|---|
| 12× Feetech STS3215-class servos | $120–160 | Open Duck Mini's own BOM (~$10–13/unit in this quantity) |
| Custom PCB, small-batch assembled | $50–150 | Outsourced assembly, complexity-dependent |
| Raspberry Pi 5 | $60–80 | |
| IMU + misc sensors | $20–40 | BNO055-class + headroom |
| Battery (2S Li-ion) + BMS | $30–50 | Matches Open Duck Mini's power architecture |
| 3D printing filament | $30–50 | Printer already owned |
| Fasteners, bearings, wiring, connectors | $50–100 | |
| **Rough total** | **~$400–650** | |

This lands comfortably under the $600–1,500 budget, leaving real margin for design iteration,
spares, and mistakes (expect some — first-time servo/PCB bring-up rarely goes perfectly). Not a
reason to gold-plate the spec; a reason not to panic about a second PCB revision or a couple of
replacement servos.

---

## 7. Safety considerations

Not optional, even for a small hobbyist robot — 12+ torque-capable servos under an RL policy
that hasn't been validated on hardware yet is a real pinch/impact hazard, and Li-ion packs have
real failure modes.

- **E-stop**: hardware kill switch on the servo power rail (MOSFET, switchable independently of
  Pi software state), plus a software-level "disable torque" path — both should work even if the
  other has failed or the policy is misbehaving.
- **Bring-up protocol**: first policy deployments tethered or with the robot propped up
  off the ground (legs can move freely but can't launch the body), before any free-standing test.
- **Battery**: 2S Li-ion needs a proper BMS (already in the power architecture, §4.2) — charge
  only with a balance charger, never leave charging unattended, standard Li-ion fire-safety
  practice (charge on a non-flammable surface, away from combustibles).
- **Torque limiting**: cap servo torque/speed in firmware below hardware maximums during initial
  bring-up and testing, raise only once behavior is trusted.

---

## 8. Immediate next steps

1. **RL environment** (`cheetah_pup/rl/`): MJX/MuJoCo Playground env over `sim/cheetah_pup.xml`
   — observations (IMU orientation + gyro, joint positions/velocities, last action, velocity
   command), 12-DOF PD position-target actions at 50 Hz, trot/walk reward terms (velocity
   tracking, height, orientation, foot clearance, action smoothness, torque/energy penalties),
   termination on falls, domain randomization hooks (mass, friction, servo gains, latency, BAM
   backlash). Smoke-test on CPU here; train on cloud GPU (§4.3: mjlab is the closer Microduck
   match, Playground the lower-friction fallback).
2. **Actuator realism**: swap the PD approximation for BAM's MuJoCo actuator model
   (`vendor/bam`, native STS3215 @ 7.4 V params) before the real training runs.
3. **Phase 2 kickoff (parallel)**: build123d parametric CAD from `locked()` — servo pockets,
   bearing points, wire routes, print splits — then STEP to the owner for the Fusion 360 pass;
   re-derive MJCF mass properties from CAD.
4. Keep this document, `docs/DESIGN_LOG.md`, and the task list current as decisions get made.

---

## 9. Open questions / deferred decisions

Things deliberately left unresolved rather than guessed at — resolve these as they become
relevant, pulling the owner in only where marked.

### 9.1 Final servo count, and whether any joint needs extra torque
Depends on Phase 1's torque analysis. Hip joints carrying more of the body's weight than knees
might warrant a higher-torque STS3215 variant or a different gear ratio at those specific joints
— don't assume uniform servos across all 12 without checking.

### 9.2 Licensing on the three unlicensed Open Duck repos
`open_duck_mini_runtime`, `open_duck_playground`, and `open_duck_reference_motion_generator` have
no LICENSE file (confirmed by direct inspection of the vendored clones, not just secondhand
research) — legally all-rights-reserved by default despite the project's "open source" framing.
Given the intent to eventually open-source this repo (§3), **reach out to the author (Discord:
`discord.gg/UtJZsgfQGe`) to request explicit licensing terms** before our own repo ships any code
that was directly copied/adapted from those three repos. Until then, treat them as
architecture/approach references only (already reflected in `vendor/README.md`'s usage rules).
**Needs the owner's input** on whether/when to make that outreach.

### 9.3 CAD toolchain — resolved 2026-09-04
build123d 0.11.1 installs and runs in this sandbox (`~/venv-cad`); STEP export is available for the
owner's Fusion 360 pass. CadQuery remains the fallback if a later version breaks.

### 9.4 Training stack: mjlab vs. MuJoCo Playground — partly resolved
MuJoCo 3.12 installs and steps in this sandbox (`~/venv-sim`), so the Phase 1 model validation runs
here on CPU. The choice between mjlab and MuJoCo Playground for GPU training is still open and
belongs to Phase 5; §4.3 leans toward mjlab given cloud GPU access, with Playground as the fallback
if setup friction is high.

### 9.5 [Low priority] Local/hybrid GPU augmentation for training
Tracked as task #9 in this session's task list. Cloud-GPU-first is the current plan; revisit only
if cloud spend becomes a real cost concern. NVIDIA PAIR (checked, §5 of the research appendix)
does not help — it's local-network inference pooling, not cloud training offload. If this becomes
relevant, look at DGX Cloud / DGX Cloud Lepton next (unconfirmed relevance — wasn't deeply
researched, flagged only as the more plausible next thing to check).

### 9.6 Appendages / behaviors beyond core locomotion
No decision was made on whether this robot gets a head, tail, or any non-locomotion
expressiveness (Open Duck Mini has a 4-DOF neck for character; Mini Cheetah has none). Not
mentioned in the original brief — treat as out of scope unless the owner raises it.

### 9.7 Target size/mass — resolved 2026-09-04
Size M locked (thigh 90 mm, 1.41 kg modeled, 120 mm hip height). S/L remain available as presets
if Phase 2 CAD or bring-up argues for a change.

### 9.8 Knee-drive architecture — resolved 2026-09-04
A · direct drive locked (owner accepted the DR-01 defaults). B/C presets stay in the library for
reference.

### 9.9 Open-loop speed loss — carry into the RL environment
Open-loop trot reaches a third of its commanded speed: the trunk pitches under the rear legs and
the front feet float most of the cycle (servo tracking itself is fine, 2–3° mean error). A crude
leveling term recovers speed but oscillates. Both behaviors are what the policy must learn around;
keep torque saturation and BAM's rate-limited target in the training model so the policy does not
learn an unrealistically fast robot, and give it the IMU (orientation + rates) in the observation.
