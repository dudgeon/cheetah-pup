# Cheetah Pup decisions

Updated 2026-09-04 on `codex/microduck-research-20260904`.

This is the active decision record for this branch. Read it with [PLAN.md](PLAN.md).
The original [handoff](../HANDOFF.md) remains historical context; the explicit user
requirements below and the corrected [research](../microduck-review/RESEARCH.md)
take precedence where they differ. No hardware purchase, manufacturing order, or
paid training job is authorized by this plan.

## Confirmed owner requirements

| Requirement | Consequence for implementation |
|---|---|
| First robotics reinforcement-learning project; agent leads the technical work | Small runnable milestones, explanatory notes, and one practical assembly step at a time. The owner should not need to select unfamiliar ML libraries or tune actuator models. |
| MIT Cheetah-inspired geometry and kinematics, scaled down | Four legs, each with hip roll, hip pitch, and knee pitch: 12 actuated joints. Preserve the topology; derive new dimensions and mass properties from purchased components. |
| Duck-style smart servos; reuse the recent Microduck stack where practical | Position-target control; prioritize a published actuator model, mjlab/PPO training, MuJoCo replay, and ONNX deployment. Select components for total integration effort. |
| **No owner servo characterization or motor-model fitting** | No pendulum rig, torque/speed identification campaign, friction fitting, or requirement to generate new BAM parameters. If existing evidence is inadequate, change the design/component or reduce the behavior goal before reopening this requirement. |
| **Smallest size that keeps the project straightforward** | Minimize packaging and mass after torque, thermal, wiring, and assembly needs are satisfied. There is no fixed exterior length limit and no requirement to replicate a scale factor. |
| **Carpet and small doorway thresholds** for the first successful walking version | Train slow, stable walking with clearance and friction variation. Treat compliant carpet and real threshold edges as hardware validation tasks; rigid-floor simulation alone is insufficient evidence. |
| **10–15 minutes of active walking; low weight takes priority** | Begin with a small pack and measured whole-robot energy use when hardware exists. Battery capacity cannot be chosen independently of mass and joint loads. |
| Hardware budget **$600–$1,500**, excluding the owned printer | Keep a revision/spares allowance within the ceiling. Quote the actual basket before purchase. Paid cloud training is a separate explicit budget decision. |
| Cloud CUDA training; macOS for development/orchestration | Use local CPU simulation for geometry, inspection, and policy replay. No dependency on Metal-compatible RL training. |
| Code-first CAD and STEP for owner refinement in Fusion 360 | Prefer build123d, with CadQuery fallback after an installation/export check. Original printable geometry; preserve assembly and mass-property data. |
| Primitive simulation before detailed CAD | Kinematics and load evidence first, then realistic training, then manufacturing design and a CAD-to-simulation update. |
| Home FDM printing; outsourced PCB assembly | Design replaceable printed parts and connectors. Deliver assembly-ready board files, including a specific treatment of through-hole connectors. |
| Likely eventual public open-source release | Preserve upstream notices and asset boundaries; do not import unlicensed reference code or Microduck's restricted mechanical assets into the original design. |
| Another agent is working in the project | Keep changes on the independent branch; do not overwrite its work or advance submodule pins silently. |

## Provisional engineering choices

These are working hypotheses, **not owner-selected hardware or validated performance**.

| Choice | Current position | Evidence required to freeze it |
|---|---|---|
| Actuator | Evaluate stock **ROBOTIS XL330-M288-T** first; **Feetech STS3215** is the heavier alternative | Exact SKU/gear ratio, usable published fit and controller conventions, voltage provenance, thermal assumptions, packaging, load margin, and obtainable parts all agree. |
| Motor rail | Regulated **5.0 V** for stock XL330 | Regulator output tolerances, transients, sustained/peak current, regeneration behavior, and wiring losses remain within the servo's rated limits. |
| Actuator realism | Coherent BAM integration and published parameters, with bounded domain randomization | Trace parameter file → BAM commit → CPU/GPU implementations → motor settings. An unresolved fit voltage is a blocker to calling the simulation hardware-ready. |
| Training framework | Microduck's mjlab/PPO approach with our own quadruped task | Clean install, repeatable smoke job, valid BAM execution, saved checkpoint, and CPU export/replay. A fallback framework needs the same actuator and deployment evidence. |
| Geometry | Small original torso and three-joint serial legs; initially direct-mounted knee servo | Compare packaging and load results before accepting distal mass or adding belts/linkages. No unnecessary transmission fabrication for a first project. |
| Compute | Light Linux board selected by measured ONNX/IO timing and available drivers | Full loop at 50 Hz with sensor/bus traffic and logging; mass, thermal, and regulator budgets close. Pi 5 is not a default commitment. |
| Sensors | One rigidly mounted body IMU, joint feedback, supply monitoring | Supported sensor/driver, known frames, timestamps, usable rate, fault reporting, and observation parity. Reserve space/interface for camera or ToF; neither is required for the initial gait. |
| Battery | Small protected pack with suitable charger; chemistry and cell count open | Fit the final rails, regulator input range, peak discharge requirement, mass allowance, and 10–15 minute target. A nominal pack voltage is not proof of direct servo compatibility. |
| PCB | Four leg bus/power branches, sensor interfaces, protection, and independent motor cutoff | Validate the bus and rails with off-the-shelf prototype electronics before ordering a custom board. A shared data line needs a supported topology and timing analysis. |
| First learned behavior | Stand → weight shift → slow crawl/walk → turn → thresholds | Staged evidence in [PLAN.md](PLAN.md). Bounding, jumping, autonomous navigation, and vision are deferred. |

The XL330's manufacturer specification gives 18 g, 0.52 N·m stall torque and
103 rpm unloaded at 5 V, with a 3.7–6 V supply range. Twelve motors alone therefore
weigh 216 g. The manufacturer's approximate 0.10 N·m continuous figure is a
20%-of-stall estimate, **not a measured thermal guarantee**. Use it as a conservative
screening reference, never as evidence that a given gait is sustainable.
[Specification](https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/),
[manufacturer listing](https://www.robotis.us/dynamixel-xl330-m288-t/).

The existing XL330 M6 fit has incomplete public voltage/firmware provenance, and the
Microduck integration and this repository's BAM gitlink use different commits.
Setting a simulator supply parameter to 5 V does not establish that every fitted
coefficient transfers correctly to a stock motor at 5 V. Resolve this by reading
existing sources and comparing implementations; do not assign the owner a calibration
experiment. See the [research](../microduck-review/RESEARCH.md#2-actuator-choice-should-minimize-total-project-work)
and [pinned references](../microduck-review/REFERENCES.md).

## Explicit corrections to the earlier handoff

| Earlier assumption | Active treatment on this branch |
|---|---|
| STS3215 is the selected exact actuator | Smart servos are selected; the exact SKU is open. Evaluate XL330 first because miniature size and model reuse matter. |
| Re-identify servos using BAM after assembly | Removed from scope by the owner's explicit requirement. |
| 2S batteries directly match the servo rail | Unproven: fully charged 2S is 8.4 V. Stock XL330 must not receive that rail; the exact STS variant also needs a documented allowed range. |
| Pi 5, BNO055 and 18650 pack are fixed | These are candidates from an older Duck design, not committed choices or a faithful Microduck bill of materials. |
| All hip axes can be stacked coaxially | Preserve the actual three-axis chain and offsets. Validate signs and frames rather than copying a silhouette. |
| A stall rating establishes the usable walking load | Use joint-specific gravity and dynamic loads, torque/speed coupling, thermal duty, current/rail limits, and uncertainty. |
| Microduck has nothing useful to vendor | Its runtime and RL code are useful references. A separate open Pollen HAT also exists, but is not proof of the production Microduck PCB. |
| Existing Duck weights provide the base policy | A biped's action/observation spaces do not match this quadruped. Reuse infrastructure; train new weights. |

## Open decisions and how they close

1. **Exact motor and fit:** agent produces a short compatibility record from published
   sources. If no candidate satisfies the no-characterization constraint, present the
   remaining evidence gap and concrete alternative before any purchase.
2. **Dimensions, total mass, and foot geometry:** agent compares parametric variants
   against packaging, joint loads, support stability, and threshold clearance. Prefer
   the smallest variant with useful margin, not the smallest visually plausible model.
3. **Compute/IMU/bus adapter:** agent selects a supported combination and demonstrates
   timing on the selected board before PCB freeze.
4. **Pack/regulation/custom board:** agent closes power and connector budgets, validates
   a prototype, and prepares exact manufacturing outputs. Owner performs assembly and
   normal checks; agent interprets results and updates the design.
5. **Cloud spend and purchases:** agent prepares an exact job or basket, expected cost,
   hard limits, and deliverables. Owner approval is the final step before spending.

Normal assembly checks are still needed: inspect connectors, confirm servo IDs and
joint directions, establish assembly zero positions, confirm IMU orientation, check
rail voltage, and observe whole-robot current/temperature and faults during progressive
operation. These checks establish that the robot was built and operates as intended;
they do not require a test rig or fitting a new actuator model.
