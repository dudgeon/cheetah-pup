# Interview and implementation sequence

Status: the interview below is answered. The active comprehensive plan is
[PLAN.md](../implementation/PLAN.md), and the first primitive model is implemented.
The original proposed sequence below is retained as research history.

## Interview answers

1. Size: smallest that keeps the project straightforward; no fixed length limit.
2. Terrain: carpet and small doorway thresholds.
3. Runtime: 10–15 minutes of active walking; prioritize low weight.

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

The active comprehensive plan sets proposed acceptance thresholds, dependencies,
budget ranges, assembly checkpoints and implementation stages. Paid compute, purchases
and manufacturing orders have not been launched by this research branch.
