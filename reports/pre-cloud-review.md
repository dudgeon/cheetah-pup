# Motor choice and readiness for RL

Updated 2026-09-04. This review combines the current assembly/load results, an actual
CPU training experiment, and two parallel actuator research tracks. It does not
change the chosen motor or commit the project to a larger robot.

## Is the current XL330 viable?

**Plausible for slow experiments, but marginal at the present 613 g allowance.**
The selected crawl needs 0.0934 N·m peak static torque, compared with ROBOTIS's
0.10 N·m continuous estimate. That leaves about 7% reserve before horizontal forces,
contact transients and uncertain physical mass. The estimate itself is calculated
from 20% of stall torque; it is not a measured thermal guarantee for our assembly.
[Manufacturer estimate](https://www.robotis.us/dynamixel-xl330-m288-t/).

The BAM model holds a 60-second stand at the selected settings. The prescribed
crawl stays upright but fails forward progress. Neither proves that a better gait
cannot work. The [mass sweep](motor-viability.md) reaches our proposed 1.5× static
reserve near 440 g, with the motors unchanged. A real component and structural
design at that mass has not been established.

For the intended carpet, thresholds and 10–15 minutes of walking, the next useful
comparison is a lighter small assembly versus a stronger actuator assembly. RL
cannot compensate for unavailable sustained motor output.

## Alternatives with useful public model evidence

These masses retain the present **397 g nonmotor allowance** only to show the
effect of motor choice. They are not finished robot estimates or clearance claims.

| Candidate | Mass of 12 motors | Arithmetic robot mass | Public training evidence | Main tradeoff |
|---|---:|---:|---|---|
| XL330-M288, 5 V baseline | 216 g | 613 g | Published BAM fit; working project adapter and CPU training | Small and affordable; narrow current load reserve and incomplete stock-voltage fit provenance |
| STS3215 C001, 7.4 V, 1:345 | 660 g | 1,057 g | Explicit 7.4 V BAM fit, measured firmware gain mapping and Open Duck Mini V2 deployment | Leading larger candidate for model reuse; new brackets/bus/power and conservative torque bounds needed |
| XC330-T288, higher-voltage variant | 276 g | 673 g | ToddlerBot empirical controller/parameters and hardware precedent | Keeps the small casing envelope; about $1,241 for 12 motors alone, plus a different power rail |
| XL430-W250 | 686.4 g | 1,083.4 g | Open Ant measured stiffness and public MuJoCo/RL implementation | Affordable Dynamixel fallback, but the published motor model is more approximate |

**Recommendation: give the exact 7.4 V STS3215 the first alternative packaging and
load study.** It offers the strongest combination found here of budget, available
fitted dynamics, and related walking deployment. A robot around 1.1 kg may be easier
to build than forcing the smallest servo to operate close to its load estimate.
This is a research priority, not a finalized size or hardware choice. Keep XC330 as
the compact comparison and XL430 as the larger Dynamixel fallback.

The STS3215 fit can justify exploratory training without owner characterization.
Its electrical torque ceiling exceeds the manufacturer's stall figure, so a policy
must also work under conservative output assumptions. Its 12 V namesake and the
STS3250 do not inherit that fit. XC330-T288's model must similarly not be relabeled
as the 5 V M288. No reviewed alternative supplies a complete, independently verified
model of firmware, backlash, contact behavior and thermal endurance.

The detailed notes provide exact SKUs, primary sources, code pins, licenses, prices,
conflicting specifications and integration implications:

- [Feetech and Open Duck evidence](../docs/implementation/ACTUATOR_ALTERNATIVES_FEETECH.md).
- [Dynamixel and Open Ant alternatives](../docs/implementation/ACTUATOR_ALTERNATIVES_DYNAMIXEL.md).
- [XC330/ToddlerBot model provenance](../docs/implementation/XC330_FOLLOWUP.md).

## One more refinement before substantial cloud training

Yes: make it a functional assembly and simulation refinement. The existing
shaft-anchored model is already useful for CPU experiments. More cosmetic detail
will not resolve the important remaining questions.

1. **Compare physically plausible assemblies.** Retain the current small option;
   build a second parametric STS3215 study with real shaft/horn/connector geometry
   and supported joints. Use component-based battery, compute and structure masses.
   Recompute moving mass, inertia, workspace, clearances and gait loads. Do not
   transplant the existing torque demand into a heavier model.
2. **Make the task require useful motion.** Extend the working CPU standing task
   to controlled weight shifts and very slow forward commands. Validate resets,
   contacts, action limits and termination. Compare against fixed targets and the
   failed prescribed gait so staying upright in place cannot count as walking.
3. **Prepare the GPU equivalent.** Match the selected actuator's gain, limits,
   friction, delay and reset state against the CPU reference. Define conservative
   motor/contact uncertainty and held-out evaluations. Then prepare one reproducible
   job with a time/dollar cap for approval.

Exact fastening and wiring design will matter before manufacture; it need not all
be complete before exploratory learning. No owner servo test bench or model-fitting
campaign is added. Normal assembled-robot checks and eventual walking validation
remain necessary to establish physical performance.

## What the sandbox actually ran

The isolated [CPU environment](../training_cpu/README.md) performed genuine PPO on
the unchanged 613 g assembly with the pinned BAM controller at 5 V/P400, 20 ms
command delay, 2 ms physics and 50 Hz actions.

- **16,384 transitions in 19.96 seconds**, eight CPU workers, eight completed
  rollout updates and 40 optimizer epochs. Initial and trained weights are saved.
- Fixed home targets, the untrained network and the trained network each completed
  **8/8 held-out five-second standing episodes**.
- **No learned improvement:** mean return was 16.676 trained versus 16.686 fixed.
  This simple task establishes the learning pipeline, not superior control.
- The independent assembly audit found zero solid or connector-reservation
  interference in 32 reset poses and 88 sampled evaluated poses. This does not
  certify the entire action space.

All 55 existing tests pass, and recorded source/checkpoint hashes match the saved
files. [Measured results and limitations](cpu-rl-smoke.md) include the full comparison.
CPU is useful for the next task-debugging work; cloud compute is unnecessary for
repeating this initial standing experiment. No paid cloud job was launched.
