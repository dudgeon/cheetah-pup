# Research findings and implications

Verified 2026-09-04 against the pinned sources in [REFERENCES.md](REFERENCES.md).
Published facts, engineering deductions, and proposed choices are distinguished below.

## 1. Reuse the appropriate parts of each project

| Source | Reusable contribution | Work this robot still needs |
|---|---|---|
| MIT Mini Cheetah | Three-joint serial leg topology, analytical FK/Jacobian, frame conventions | Original dimensions, component mass distribution, joint limits, CAD and servo dynamics |
| Current Microduck runtime | 50 Hz position-target control architecture, ONNX deployment, telemetry and policy-switching patterns | Twelve-joint mapping, a quadruped observation contract and hardware adapters |
| Current Microduck RL | mjlab/PPO workflow, BAM actuator integration, randomization and export mechanisms | Quadruped task, rewards, contacts, reset states, observations and freshly trained weights |
| Pollen RPI Robot HAT | Open KiCad schematic, PCB and manufacturing files; servo interfaces and sensors | Rail/current sizing, connector choice, power cutoff and packaging for this robot |
| Open Duck Mini v2 | Existing STS3215 hardware integration and accessible DIY construction precedent | It is a different robot from the newly released Microduck; its components are substitutes |
| BAM | Published identified actuator parameters and simulation interfaces | Verify exact actuator, voltage, controller behavior and software compatibility; avoid assuming every motor has every fitted model |

Microduck is publicly described as **open-source software**, not a complete open-hardware
robot. Its press kit states RK3566 compute, 1 GB RAM, 32 GB storage, NP-F550 battery,
two IMUs, camera and 8×8 ToF sensing. The editable production mechanical and electronic
designs are not offered by that disclosure. However, a separate official open robot HAT
does exist, so the earlier blanket conclusion that no relevant Pollen PCB exists is too
broad. The HAT is an electrical reference, not proof of the shipped Microduck circuit.
[Product disclosure](https://pollen-robotics.com/microduck/press-kit/),
[HAT](https://github.com/pollen-robotics/elec_RPI_Robot_HAT).

## 2. Actuator choice should minimize total project work

The new requirement removes a pendulum test rig, measured motor identification, and
fitting new BAM parameters from the baseline. Select an exact supported actuator and
published calibration first; then size the robot conservatively around it. Basic assembly
checks and validation of the complete robot are still necessary, but they should not turn
into a motor-characterization project.

| Candidate | Published evidence | Implication |
|---|---|---|
| Feetech STS3215, specified 7.4 V / 345:1 variant | 55 g; 19.5 kg·cm stall torque; 5 kg·cm rated torque; 52 rpm unloaded in linked specification | Twelve weigh 660 g before structure, battery or compute. Economical integration precedent, but more difficult to make truly tiny. |
| ROBOTIS XL330-M288-T | 18 g; 0.52 N·m stall and 103 rpm unloaded at 5 V; allowed supply 3.7–6 V | Twelve weigh 216 g. Attractive for miniature geometry; smaller torque allowance and component cost must be assessed. |
| Exact Microduck production servo | Code uses XL330/Dynamixel Protocol 2.0 interface; exact purchasable production variant not established | Do not equate an interface name or fitted simulation model with a verified purchasing SKU. |

[Feetech specification](https://www.feetechrc.com/Data/feetechrc/upload/file/20260622/6391772523943436695270694.pdf),
[ROBOTIS specification](https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/),
[Microduck bus implementation](https://github.com/pollen-robotics/microduck/blob/bc41fb5c9a9b39894669c1e022e375cf83800382/duck-control/src/bus.rs).

The Feetech page and its linked specification contain differing figures, so freeze the
supplier's actual variant documentation before purchase. 19.5 kg·cm is approximately
1.91 N·m **stall**, whereas 5 kg·cm is approximately 0.49 N·m **rated**. Neither a stall
rating nor the name of a software torque-limit setting establishes sustainable joint force.
Position control with modeled actuator dynamics remains the working architecture.

Two electrical corrections matter now:

- A 2S pack reaches 8.4 V fully charged. The cited STS3215 specification lists 4–7.4 V;
  a nominal 7.4 V label alone does not validate direct 2S operation. Require an explicitly
  suitable variant or regulation.
- Microduck software describes a roughly 6.6–8.2 V motor supply, while stock XL330 is
  rated only to 6 V. Do not copy that rail into a stock XL330 design. The mismatch must
  remain unresolved until a compatible motor/calibration/power combination is selected.
  [Runtime battery model](https://github.com/pollen-robotics/microduck/blob/bc41fb5c9a9b39894669c1e022e375cf83800382/duck-control/src/model.rs).

BAM contains both STS3215 and XL330 model families. Their availability is not equivalent:
the STS3215 documentation identifies an M1 fit, while Microduck uses an XL330 M6 model.
Do not change only the motor name and assume Microduck's M6 setup still applies.
The selection must trace motor SKU → identified parameter file → controller behavior →
supported simulation implementation. Published calibration reduces work; it does not
guarantee an exact simulation of a new robot or manufacturing batch.
[BAM](https://github.com/Rhoban/bam).

**Provisional recommendation following the new constraint:** evaluate genuine
XL330-M288-T at regulated 5.0 V first. Its smaller size and richer published M6 fit
better support the miniature objective. ROBOTIS lists $27.49 each at this research
date, approximately $330 for twelve before tax/accessories. Its listed 0.10 N·m
continuous figure is explicitly an estimate at 20% of stall, not a measured thermal
guarantee. Size the first simulated robot conservatively around this limit.
[Manufacturer listing](https://www.robotis.us/dynamixel-xl330-m288-t/).

This is not a final motor selection. Inspection of both BAM commits found different
XL330 M6 coefficients; the newer project pin includes a fitted command delay, while
the Microduck lock's actuator constructor includes an explicit 1.75 A maximum.
Both default to 7.5 V / P gain 400, and the fitted JSON does not record measurement
voltage or firmware revision. Use a coherent pinned integration and explicitly set
the physical/simulated voltage and controller settings; do not silently mix fits.
The existing public provenance leaves uncertainty about stock 5 V deployment.
Resolving that through existing documentation and software inspection is a research
gate, not an obligation for the owner to characterize motors. Keep STS3215 as the
stronger/heavier alternative if size and conservative load analysis reject XL330.
[Pinned M6 parameters](https://github.com/Rhoban/bam/blob/62bd8ce12154340be97e06f7f41a0ca8f116d967/bam/params/xl330/m6.json),
[pinned actuator implementation](https://github.com/Rhoban/bam/blob/62bd8ce12154340be97e06f7f41a0ca8f116d967/bam/dynamixel/actuator.py),
[actuator catalog](https://bam.readthedocs.io/en/latest/usage/actuators.html).

## 3. Geometry is a reference, not a uniform scale operation

MIT's `MiniCheetah.h` specifies a 380×98×100 mm body model, 62 mm hip lateral link,
209 mm upper leg and 195 mm lower leg, plus a 4 mm knee lateral offset. These are
model parameters, not the robot's overall exterior dimensions. Its 3.3 kg body mass
excludes the rest of the robot. Its `maxLegLength=0.409` also differs from the current
0.209+0.195 m link sum; derive reach from the actual chain.
[MIT geometry](https://github.com/mit-biomimetics/Cheetah-Software/blob/c71c5a138d3e418cc833e94e25357ceea8955daa/common/include/Dynamics/MiniCheetah.h).

Retain hip roll, hip pitch and knee pitch on each leg. The roll and pitch axes are not
all coaxial. A knee-mounted servo preserves this topology with simpler construction,
but adds distal mass compared with a proximal motor and belt. Decide the transmission
after packaging and simulated load comparison, not from silhouette alone.

MIT's FK uses leg order FR, FL, HR, HL; its dynamics code also applies an extra hip-frame
rotation. Translate the full convention and validate mirrored legs rather than copying
signs from isolated functions.
[FK/Jacobian](https://github.com/mit-biomimetics/Cheetah-Software/blob/c71c5a138d3e418cc833e94e25357ceea8955daa/common/src/Controllers/LegController.cpp),
[frames](https://github.com/mit-biomimetics/Cheetah-Software/blob/c71c5a138d3e418cc833e94e25357ceea8955daa/common/src/Dynamics/Quadruped.cpp).

Engineering deduction: at constant density ideal scaling would give mass ∝ length³,
inertia ∝ length⁵ and gravity torque ∝ length⁴. Purchased servos, boards and batteries
do not scale that way. Build an actual component mass budget and derive loads with
τ=JᵀF, including two-foot support. For illustration, a 1 kg robot equally supported
on two feet at 50 mm effective joint moment arms needs about 0.245 N·m per joint
statically, before accelerations or unequal loading. This is a calculation, not a
proposed robot rating.

## 4. Training and deployment

Microduck's current recipe is mjlab with MuJoCo Warp and PPO. Training requires CUDA;
it offers Hugging Face Jobs submission and CPU MuJoCo policy replay. Keep macOS for
authoring, review and supported local replay; no Metal training claim is made.
Its 14-action, 61-observation biped policies cannot directly control this 12-joint
quadruped. Reuse the training and deployment machinery and train new weights.
[RL workflow](https://github.com/pollen-robotics/microduck_rl).

The inspected project pins Python 3.12, mjlab 1.3.0, Warp 1.12.0 and Torch 2.9.1.
Its lockfile uses BAM commit `62bd8ce12154340be97e06f7f41a0ca8f116d967`, while this
project currently vendors `620a64fe67c1afe94fca81da73b128c7aed17c5f`. Keep those
identities explicit. Do not casually combine dependencies from different branches or
replace a fitted actuator law with ideal PD just to get a successful demo.
[Dependency configuration](https://github.com/pollen-robotics/microduck_rl/blob/29e887ecfbf5d37144759e5a9f8a176dfb83d547/pyproject.toml),
[lockfile](https://github.com/pollen-robotics/microduck_rl/blob/29e887ecfbf5d37144759e5a9f8a176dfb83d547/uv.lock).

Runtime code targets a Radxa Zero 3W; its IMU adapter uses LSM6DSV16X with a custom
Dynamixel endpoint. A Pi/BNO055 combination is an adaptation, not a drop-in clone.
The separate HAT uses a BMI088 and a 5 V/2 A converter. Do not assume this is sufficient
for the prior plan's Pi 5. For initial locomotion, propose one body IMU and reserve
mounting/power/data provisions for camera and ToF; extra sensors do not automatically
improve the first gait. [IMU interface](https://github.com/pollen-robotics/microduck/blob/bc41fb5c9a9b39894669c1e022e375cf83800382/duck-control/src/imu.rs),
[HAT files](https://github.com/pollen-robotics/elec_RPI_Robot_HAT/tree/23eab11927f95ceca0dfa35bf182caeb7db39ea0).
