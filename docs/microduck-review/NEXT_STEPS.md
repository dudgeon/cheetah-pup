# Interview and implementation sequence

Status: research baseline saved; requirements interview remains open. This is a proposed
sequence, not the final comprehensive plan or a claim that implementation is complete.

## Questions that still affect the design

1. Maximum acceptable overall robot length / preference for compactness versus easier
   component packaging. STS3215 and XL330 are materially different size and mass choices.
2. Required first-version terrain: smooth indoor floor, carpet/thresholds, or outdoor
   uneven surfaces. This changes foot design, stance, sensors and training objectives.
3. Desired walking runtime per charge. Battery mass feeds directly into motor and geometry
   selection; avoid specifying a battery independently of the load model.

Do not re-interview the owner about motor characterization: they explicitly do not want
to do it. Do not ask them to choose obscure simulation libraries. Make those engineering
recommendations from the evidence. Existing budget, CAD and cloud-compute answers are
recorded in the base handoff; only reopen them for a concrete incompatibility.

## Proposed milestones

| Stage | Deliverable | Completion evidence |
|---|---|---|
| 0. Requirements and motor-model selection | Final requirements; exact motor SKU, supply, published parameter source and compatible software versions; cost/mass budget | Existing calibration is usable without owner characterization; choices fit size and task goals |
| 1. Reproduce and understand the training workflow | Versioned environment, documented commands, a baseline example and short learning notes | Repeatable simulation and export/replay; remote training costs and access are explicit |
| 2. Original primitive quadruped | Parametric 12-DOF MJCF and analytical kinematics, realistic component masses | FK and Jacobian agreement, joint/mirror correctness, collision/workspace sweep, plausible loaded torque/speed budget |
| 3. First learned stand and walk | Quadruped PPO task, fixed evaluation suite, policy and video | Measured command tracking, stability, falls, actuator saturation and energy proxies across held-out seeds and parameter ranges |
| 4. Manufacturing CAD | Scripted original CAD, STEP/STL, assembly instructions and BOM | Mechanical fit review; CAD mass/inertia reconciled into simulation; policy reevaluated |
| 5. Electrical design | Off-the-shelf prototype wiring first; custom KiCad board after requirements stabilize | Exact voltage/pinout/current budgets, logic power capacity, motor cutoff and assembly-service outputs reviewed |
| 6. Robot runtime and bring-up | Position-target inference, joint mapping, IMU orientation, telemetry and fault handling | Simulator/runtime observation parity; correct direction and limits; timely sensor/control loop and reliable stop behavior |
| 7. Sim-to-real iteration | Conservatively trained policy deployed and progressively evaluated | Supported stand then slow walk; logged failures drive geometry/control/randomization changes without custom servo fitting |

Stage 3 confirms viability in a particular simulation; it does not prove physical
sim-to-real success. Published actuator fits plus realistic structure, contacts,
latency and conservative randomization are the baseline path. If that path fails,
first simplify behavior, adjust geometry, or select a better-supported component.
Do not silently turn the project into a servo-identification exercise.

The comprehensive plan after the interview will set numerical acceptance thresholds,
file/module ownership, dependencies, budget ranges, assembly checkpoints and the first
implementation issue. Paid compute, purchases and manufacturing orders are not launched
by this research branch.
