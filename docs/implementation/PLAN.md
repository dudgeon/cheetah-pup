# Cheetah Pup implementation plan

Updated 2026-09-04. Active on `codex/microduck-research-20260904`.

Build a small, printable 12-DOF quadruped that learns to walk on carpet and over small
doorway thresholds, using published servo dynamics and as much practical Microduck
infrastructure as possible. The owner is learning robotics RL for the first time;
**motor characterization is not part of the project**. Confirmed requirements and
provisional choices are separated in [DECISIONS.md](DECISIONS.md).

The research and requirements interview are complete, and the first primitive model
and screening tools are implemented. Detailed mechanical CAD, an electrical design, a learned quadruped policy,
and demonstrated hardware performance remain future work. A geometric or quasi-static
result is not proof that the robot can walk or that a published servo fit transfers.

## Proposed success targets

Except for terrain, compactness, budget, and battery duration selected by the owner,
the following numbers are **initial engineering targets**, revisable with evidence.

| Measure | First useful target | Verification |
|---|---|---|
| Size and mass | Smallest package with serviceable wiring and load margin; begin the XL330 study near 0.6 kg total | Actual component budget and dimension sweep, then CAD/assembled measurements. No fixed exterior length yet. |
| Stand | 60 s stationary without falling or persistent saturation | Simulation evaluation across seeds/uncertainty, then supported and free-standing hardware tests. |
| Walk | Begin at 0.05 m/s, then 0.10 m/s, with controllable stop and gentle turning | Held-out command episodes and a measured indoor path. 0.20 m/s and trot are stretch goals after the baseline succeeds. |
| Carpet | Repeatable starts, stops and turns on an ordinary household carpet sample | Hardware video/logs; record pile, underlay and foot snagging. Floor friction variation in sim is only an approximation. |
| Threshold | Start with a 10 mm obstacle, then 15 mm, at 0.10 m/s | Simulated edge/width/friction variations, then at least 9/10 successful traversals each direction on one measured threshold. This does not claim all doorway thresholds. |
| Runtime | 10–15 min active walking, prioritizing low mass | Whole-robot energy/temperature log and remaining usable charge; reserve enough energy for a controlled stop. |
| Control | 50 Hz position-target loop, no stale commands accepted | Full-loop p99 latency below 20 ms and explicit fault behavior under delayed/dropped inputs. |
| Cost | Hardware within $600–$1,500 including planned spares/revision allowance | Exact basket with tax/shipping before order; training spend tracked separately. |

## Milestones and completion evidence

The agent implements and documents each stage. The owner reviews tangible designs and
performs physical printing/assembly/testing with guided steps. Routine engineering
choices do not need another interview.

| Stage | Dependency and agent deliverables | Exit evidence / owner involvement |
|---|---|---|
| **0 — Research and requirements** | Pinned upstreams; source corrections; active decisions and this plan | Completed baseline. Preserve the other agent's work and upstream asset boundaries. |
| **1 — Primitive robot and sizing** | Parametric MJCF, frame/joint contract, analytical kinematics, component masses, support/load calculations, reproducible report | Validate mirrored legs and FK/Jacobian first; workspace sweeps, self-collision and packaging remain required follow-ups. Report feasible and rejected parameter sets with the assumptions that decided them. This is the first implementation slice. |
| **2 — Actuator and training foundation** | From stage 1: exact motor/rail/fit compatibility record; coherent software lock; BAM CPU/GPU parity check; runnable local replay and cloud smoke-job package | Published model is usable without owner identification; no hidden ideal-actuator substitution. Reproduce reset/step/checkpoint/export/replay. Paid job execution waits for the exact capped job to be approved. |
| **3 — Learned stand and slow walk** | From stages 1–2: quadruped task, observation/action contract, reward/curriculum, randomization, PPO training, ONNX export, fixed evaluation suite | Pass the simulation gates below using the selected actuator law; publish configuration, weights, metrics and videos together. Failures drive geometry or control changes before detailed CAD. |
| **4 — Manufacturable design and prototype basket** | From stage 3: original parametric CAD, STEP/STL, mass/inertia update, wiring diagram, selected compute/sensor/power parts, itemized prototype basket | Packaging/DFM review; CAD-based simulation re-evaluation; documented power and bus budgets. Owner can review STEP and print plan. Approve concrete purchases only after evidence is ready. |
| **5 — Off-the-shelf prototype and runtime** | From stage 4: assembly guide, drivers, policy runner, telemetry, motor cutoff, fault handling, progressive bring-up instructions | Owner assembles and records normal checks. Agent verifies mappings, real IO timing, rail behavior, supported stand, then short free walks. No actuator fitting. This validates interfaces before custom PCB manufacture. |
| **6 — Custom breakout PCB and integrated robot** | From stage 5: KiCad schematic/layout, protection/current review, manufacturing/assembly files, board test and installation instructions | Review clean ERC/DRC plus documented waivers; exact assembled-board quote and assembly scope. After order approval, owner installs and checks the board. Re-run fault and gait checks after the electrical change. |
| **7 — Carpet, thresholds and runtime** | From stage 5, and stage 6 if the custom board is used: progressive terrain protocol, log analysis, model uncertainty/geometry/control refinements, updated trained policy and reproducible release | Meet the real-world targets above and document limits. Finish with BOM, source revisions, CAD/PCB files, assembly guide, runtime setup, policy and known issues. |

Stages 4–5 may use an existing adapter indefinitely if it meets the need. The custom
board earns its place through packaging, protection and serviceability; it is not a
prerequisite to the first supported stand. A board revision must not silently change
motor settings, observation timing or supply assumptions used in training.

## Stage 1: the first implementation slice

1. Establish body axes and a fixed leg/joint order. Create a free-floating torso with
   four serial hip-roll/hip-pitch/knee-pitch chains and explicit masses, joint ranges,
   feet and simple collision geometry. Keep primitive geometry independent of CAD.
2. Derive analytical forward kinematics and Jacobians for all mirrored legs; compare
   against an independent numerical/MuJoCo implementation. Proposed tolerances are
   1e-6 m for foot position and 1e-5 for Jacobian elements away from numerical limits.
3. Sweep useful link lengths, stance height, and total mass. Inspect foot workspace,
   joint limits, self-collisions, belly clearance, support polygon and center of mass.
4. Calculate joint torque with `tau = J.T @ F`, including four-, three- and two-foot
   support, uneven load sharing, link/servo gravity, and a stated dynamic margin.
   Solve force and moment balance rather than dividing weight by contact count.
   A centered body over three corners of a rectangular stance is on the support
   triangle's diagonal edge, so two feet can still carry almost the whole load.
   A crawl must shift the center of mass into the triangle before lifting the fourth
   foot; three contacts alone do not guarantee three-way load reduction.
5. Separate geometry checks, quasi-static load screens, and any ideal-controller demo
   from actuator-realistic simulation. Save assumptions and rejected cases alongside
   promising ones. No learned policy or physical feasibility claim follows from this
   slice alone.

The first implemented candidate uses a 160 × 70 × 45 mm torso primitive,
70/75 mm upper/lower links, a 25 mm hip lateral offset, 8 mm foot radius and a
0.613 kg component allowance. These are simulated geometry assumptions, not exterior dimensions
or verified servo packaging. Its nominal hip/knee angles are 0.4/−0.8 rad. The
implementation paths are `config/robot.json`, `src/cheetah_pup/`,
`models/cheetah_pup_flat.xml`, `models/cheetah_pup_threshold.xml`, and
`reports/primitive-validation.{json,md}`. Use the generated report for actual checks
and results; do not mark a gate passed because its file exists. The initial primitive
position controller is an idealized diagnostic tool, not BAM or a learned policy.

**Current evidence:** 26 tests pass, the FK/Jacobian checks agree with MuJoCo,
and 45 combinations of stance, link length and nonmotor mass are screened. The
neutral four-foot pose requires approximately 0.047 N·m peak static torque. Lifting
a front foot without a body shift raises that to about 0.086 N·m, below the 0.10
N·m estimate but short of the proposed 1.5× margin. Lifting a rear foot in that same
pose has no feasible vertical static equilibrium. These are reasons to design and
evaluate body shifts; they do not reject all XL330 gaits or prove a workable one.
Workspace/clearance checks and a realistic actuator integration remain open.

The first result should answer: **what geometry and total mass are plausible enough
to justify integrating the published actuator model?** It may answer that the leading
motor is too weak. In that case shorten moment arms or reduce mass before escalating
to the heavier STS alternative; compare total build effort as well as torque.

## Size, weight, power and cost budgets

These are planning allowances, not a finalized BOM, quotes or validated load ratings.
The implemented 0.613 kg study point uses this provisional mass allocation from
`config/robot.json`:

| Assembly | Initial allowance |
|---|---:|
| 12 XL330 motors | 216 g, manufacturer mass |
| Printed torso, brackets, links and feet | 196 g |
| Compute | 40 g |
| Pack | 100 g |
| Electronics | 35 g |
| Wiring and fasteners | 26 g |
| **Total study point** | **613 g** |

Motor mass and body/leg mass must be allocated exactly once, at their physical
locations. Replace guessed values with vendor masses and then CAD or assembled
measurements. Run sensitivity cases at least ±20% around nonmotor mass assumptions.
Every heavier battery, board or structural reinforcement must re-enter the load model.

For perspective, at 600 g, equal two-foot support is 2.94 N per foot. A 30 mm effective
joint moment arm produces about 0.088 N·m from body support alone; at 40 mm it is about
0.118 N·m. Neither includes link gravity or acceleration. That already straddles the
XL330 manufacturer's approximate 0.10 N·m continuous estimate, so a visually convincing
small model can still have inadequate margin. Stall torque cannot close this gap.
[Motor specification](https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/),
[continuous estimate](https://www.robotis.us/dynamixel-xl330-m288-t/).

For battery sizing, an **unvalidated** 10–20 W average whole-robot load would require
1.7–5 Wh for 10–15 minutes. With 90% conversion efficiency and 80% usable energy,
that becomes approximately 2.3–6.9 Wh of nominal pack capacity. This is an arithmetic
sensitivity range, not a measured power forecast or a pack selection. The 100 g mass
allowance must be checked against the selected pack. Separately size wiring, regulator and protection
for simultaneous current peaks and startup; average watts do not establish peak amperes.
Do not use a blanket simultaneous-stall assumption as normal operating consumption.

| Hardware category | Planning allowance, USD |
|---|---:|
| Twelve XL330-class motors | $330–400 |
| Two spare motors | $55–70 |
| Compute | $45–100 |
| IMU, bus adapter, regulation and prototype protection | $80–160 |
| Pack and appropriate charger | $45–90 |
| Printed parts, fasteners, feet, wiring and connectors | $60–120 |
| Small assembled breakout PCB batch | $80–180 |
| Shipping, taxes and revision contingency | $100–230 |
| **Total allowance** | **$795–1,350** |

The XL330 direction is more likely to occupy the middle of the owner's budget than
its $600 floor. Refresh prices and check connector/horn inclusions before ordering;
avoid spending the remaining margin on optional perception hardware.

## Simulation and training contract

**Reproducibility.** Preserve submodule commits, Python/platform requirements, a full
dependency lock, model/config hashes, seed, actuator parameter hash, commands and
evaluation version with each run. Microduck's reviewed stack uses Python 3.12, mjlab
1.3.0, Warp 1.12.0 and Torch 2.9.1; these are a starting compatibility set, not permission
to combine arbitrary newer packages. Its BAM lock differs from this repository's
gitlink. Record the selected dependency explicitly and verify both local CPU replay
and CUDA execution. [Reviewed dependency configuration](https://github.com/pollen-robotics/microduck_rl/blob/29e887ecfbf5d37144759e5a9f8a176dfb83d547/pyproject.toml),
[lockfile](https://github.com/pollen-robotics/microduck_rl/blob/29e887ecfbf5d37144759e5a9f8a176dfb83d547/uv.lock).

**Actuator contract.** Record motor SKU, gearing, firmware evidence if available,
regulated voltage, control mode/gains/current limit, command delay, update rates and
the exact BAM parameter source. Audit unit/sign conventions and friction/backlash
behavior. Randomization represents a stated uncertainty envelope; it does not repair
an unidentified hardware variant or justify operation above rated voltage.

**Policy interface.** Proposed actor input is 45 scalars: body gyro (3), projected
gravity (3), commanded forward/lateral/yaw velocity (3), 12 joint positions relative to
the nominal stance, 12 joint velocities, and 12 previous actions. Output is 12 bounded
position offsets from that stance. Freeze units, frames, ordering, normalization,
clipping, filtering, action scaling and reset history in a machine-readable schema.
No perfect simulated base linear velocity, terrain map or contact state goes to the
actor unless the real robot can supply an equivalent estimate. The critic may use
privileged information during training; deployment must not depend on it.

**Learning sequence.** Start with joint-limit-respecting standing resets and stable
standing; add commanded weight shift and slow walking; then turns, varied friction,
small perturbations and thresholds. Use quadruped-specific rewards for command
tracking, upright posture and useful foot clearance, with costs for falls, slipping,
body strikes, excessive action changes and actuator saturation. Inspect behavior for
reward loopholes. Existing Microduck biped weights are not a quadruped starting policy.
[Microduck RL reference](https://github.com/pollen-robotics/microduck_rl).

**Evaluation.** Keep evaluation seeds and terrain cases out of training. Start with
20 seeds per fixed flat-ground command case, 60 s episodes, requiring at least 19/20
without falls; proposed steady-state forward-speed error is ≤0.05 m/s. Evaluate
threshold height, edge shape, width and approach direction separately. Report contact
failures, slip, body clearance, joint-limit hits, torque/speed/current saturation,
RMS/peak loads and energy proxies. A numerical success rate is insufficient if it
depends on persistent saturation. Before hardware release, repeat the suite with
CAD-derived masses and a documented actuator/contact/latency uncertainty envelope.

**Cloud jobs.** Prepare a dry-run command and cost estimate before asking to spend.
The job specifies GPU type/rate, maximum runtime and dollar cap, checkpoint interval,
seed/config, artifact destination and cancellation behavior. Begin with a bounded
smoke job before a training campaign; stop failed or nonprogressing runs automatically.
Checkpoint/export/replay correctness comes before larger budgets. No unattended
unbounded training, purchases or subscription changes are part of this turn.

## Mechanical, electrical and runtime execution

**CAD.** Use exact servo/horn envelopes and mounting dimensions; design for fastener
access, replaceable feet, removable battery, supported joints and protected flexing
wires. Keep IMU mounting rigid and away from unnecessary vibration. Document print
material, orientation, supports, tolerances and assembly order. Deliver native scripts,
STEP assembly, printable STLs, exploded views and hardware counts. For each rigid
link, export mass, center of mass and inertia with a declared material/infill assumption;
add purchased hardware separately. Validate positive inertias and frame transforms,
replace visual meshes with efficient collision shapes, and repeat simulation evaluation.

**Power and breakout.** Prefer proven adapters initially. Freeze exact pinouts,
connector polarity/keying, conductor size and current ratings, branch topology,
protocol/baud rate, logic levels and motor/logic ground paths. Provide motor power
cutoff independently of the host, fuse/protection, supply monitoring, test points and
bulk decoupling. Verify regulator heat, transient response and what happens to energy
returned by decelerating motors. Keep compute alive long enough to log a motor cutoff
where practical. The separate Pollen HAT is a useful reference, but its 5 V/2 A supply
does not establish capacity for our whole robot. [HAT reference](https://github.com/pollen-robotics/elec_RPI_Robot_HAT/tree/23eab11927f95ceca0dfa35bf182caeb7db39ea0).

**PCB package.** Deliver KiCad sources, fabrication drawings, Gerbers, drill files,
BOM with exact orderable parts and approved alternates, component placement files,
assembly drawings and test instructions. Check footprint/polarity, board stack-up,
copper/current assumptions, mounting clearances and enclosure fit. Run ERC/DRC and
review all exceptions. Obtain a quote including connectors: confirm explicitly which
through-hole parts are assembled, whether selective/manual soldering is included,
and what the owner must plug in. Do not describe a board with bare connector holes as
fully assembled. Review the first physical board before ordering a larger batch.

**Runtime.** Separate sensor/bus IO, observation construction, ONNX inference,
action limiting, safety state and logging. Use the same schema and reference sample
vectors in simulation and deployment. Benchmark complete loop timing under logging
and communications load. Log timestamps, commands, joint feedback, IMU data, supply
data, action clipping and fault transitions. Validate ONNX outputs against the training
implementation on saved observations within a declared numerical tolerance.

| Fault | Required behavior and validation |
|---|---|
| Stale IMU/joint data, nonfinite observation/action or invalid policy metadata | Reject new motion; enter the documented safe state. Inject these faults in software first. |
| Missed control deadlines or lost host process | Independent watchdog removes motor enable; initial response target ≤100 ms. Verify actual cutoff, not only a log message. |
| Bus timeout or missing servo | Stop advancing the gait; avoid continuing with one uncontrolled leg. Test a disconnected channel while supported. |
| Undervoltage, overcurrent or excessive reported temperature | Inhibit startup or stop before operating limits are exceeded; exact thresholds follow selected hardware documentation. |
| User stop | A physical motor cutoff works without a healthy policy process; software stop is separately tested. |

Motor cutoff can make the robot collapse. Design the chassis and supported bring-up
arrangement for that behavior; do not assume an emergency controlled crouch is possible
after sensor or power failure. Begin physical work with one motor, then one supported
leg, then a suspended robot; confirm IDs, zeros, direction, limits, IMU frame and stop
behavior before supported stance and short free walks. These are normal assembly and
system checks, not a servo-identification campaign.

## If results disagree with simulation

Check assembly, frame/order errors, observations, latency, voltage, geometry/mass and
foot contact assumptions first. Use existing telemetry to diagnose whole-robot behavior.
Reduce speed/step height, adjust stance and clearance, or revise the bounded uncertainty
envelope and retrain. If the motor model remains unsupported, choose a better-supported
actuator or reduce the behavior goal. Do not quietly add custom motor characterization
to the owner's workload or advertise an accurate digital twin from vendor specs alone.

Each milestone ends with a short explanation of what was learned, one reproducible
command or concrete assembly checklist, and a concise record of remaining uncertainty.
The first learning goal is understanding how geometry creates joint load; the next is
how observations, actions and rewards create a policy; the final goal is understanding
which evidence supports transferring that policy to the physical robot.
