# Feetech actuator alternatives for cheetah-pup

Research date: 2026-09-04. This is a component shortlist, not a purchasing decision. No actuator or CAD changes were made.

## Recommendation

**Put the Feetech STS3215 7.4 V, 19 kg·cm, 1:345 version on the practical shortlist.** The manufacturer calls this variant **ST-3215-C001**. It is the strongest Feetech choice here because a published fitted actuator model, explicit firmware gain mapping, and demonstrated Open Duck Mini V2 deployment all exist. It can support exploratory RL without requiring the owner to identify servos. A redesign around approximately 1.1 kg may be a more straightforward project than forcing the existing 613 g design to work near the XL330 limit.

It is not a drop-in replacement. Its twelve motors alone weigh more than the whole present robot. The exact 12 V STS3215 and STS3250 do not inherit the 7.4 V model merely because their names or enclosures are similar. Neither qualifies as a better supported alternative on the evidence located in this bounded search.

## Physical and electrical screening

| Candidate | Manufacturer mass | Twelve motors | Mass with present 397 g nonmotor allowance | Manufacturer torque information | Decision |
|---|---:|---:|---:|---|---|
| STS3215 / ST-3215-C001, 7.4 V | 55 ± 1 g | 660 g | 1,057 g | Page lists rated torque 6.5 kg·cm at 6 V, approximately 0.637 N·m; launch article separately lists 19.5 kg·cm stall at 7.4 V, approximately 1.91 N·m | Leading Feetech candidate; redesign and run the fitted model |
| STS3215 / ST-3215-C018, 12 V | 55 ± 1 g | 660 g | 1,057 g | Catalogue lists 30 kg·cm stall at 12 V, approximately 2.94 N·m | No exact fitted model located; do not substitute into C001 fit |
| STS3250 / ST-3250-C001, 12 V | 74.5 ± 1 g | 894 g | 1,291 g | Rated 16 kg·cm at 12 V, approximately 1.57 N·m; separately 50 kg·cm stall, approximately 4.90 N·m | More mass and no exact fitted model located; secondary option |

These robot masses are arithmetic comparisons, not completed BOM estimates: larger supports, wiring and power components can increase them. “Rated torque” is the manufacturer's separate published quantity; it is **not** a continuous rating invented as a fraction of stall. The cited pages do not establish a 10–15 minute walking duty cycle, enclosure cooling, or battery runtime. The current 0.093 N·m crawl requirement cannot simply be reused after replacing motor masses and geometry.

The C001 and C018 catalogue envelopes are 45.2 × 24.7 × 35 mm; STS3250 is 45.22 × 24.72 × 35 mm. They require new shaft origins, brackets, collision volumes, inertias and cable routing in our CAD. The C001 launch article gives a 1:345 gear ratio and 52 rpm at 7.4 V; its text calls this “stall speed,” an internally awkward label, so use the detailed datasheet before freezing a speed limit. This is substantially slower than the XL330's 103 rpm unloaded figure and favors deliberate walking rather than simply speeding up the existing crawl.

Sources: [C001 manufacturer page](https://www.feetechrc.com/74v-19-kgcm-plastic-case-metal-tooth-magnetic-code-double-axis-ttl-series-steering-gear.html), [C001 launch/evaluation article](https://www.feetechrc.com/20210430-56680.html), [STS3215 variant catalogue](https://www.feetechrc.com/products.html?keyword=STS3215), [STS3250 manufacturer page](https://www.feetechrc.com/562636).

The C001 page has conflicting plastic/aluminum and core/coreless wording, and its 6 V stall entry conflicts with the launch article's voltage attribution. Do not use those inconsistent marketing fields as precise simulation evidence. The mass, variant identity and available fit are sufficient for this shortlist; a selected part needs the dated manufacturer drawing/specification.

## What is actually available for training

Our existing `vendor/bam_microduck` checkout already includes the STS3215 model at Rhoban/BAM commit `62bd8ce12154340be97e06f7f41a0ca8f116d967`. No additional submodule is needed to preserve it.

| Evidence | Exact record | Implication |
|---|---|---|
| Fitted model | `bam/params/feetech_sts3215_7_4V/m1.json` | Explicit 7.4 V variant, M1 Coulomb/viscous fit |
| File SHA-256 | `6f2d5c78a5ab86aee1cbb8c330e24076be741a3913ee27b96168bec19d428cfa` | Reproducible parameter input |
| Firmware control class | `bam/feetech/actuator.py`, `STS3215Actuator` | Defaults to 7.4 V and firmware P = 32 |
| PWM mapping | Error gain 0.166; maximum PWM 0.97 | Code says these were determined using an oscilloscope on STS3215 actuators |
| Motor fit | Kt = 1.21164135 N·m/A; R = 2.67616633 ohm | Voltage and back EMF affect torque; more useful than a constant ideal torque cap |
| Mechanical fit | Armature 0.02840336 kg·m²; Coulomb friction 0.05239296 N·m; viscous term 0.05908516 | Effective identified dynamics, not manufacturer rotor CAD inertia |
| Additional behavior | Fitted position offset and an internal target velocity limiter | Reset and command semantics must be preserved in a training adapter |

Source: [pinned parameter file](https://github.com/Rhoban/bam/blob/62bd8ce12154340be97e06f7f41a0ca8f116d967/bam/params/feetech_sts3215_7_4V/m1.json), [pinned actuator implementation](https://github.com/Rhoban/bam/blob/62bd8ce12154340be97e06f7f41a0ca8f116d967/bam/feetech/actuator.py). BAM is Apache-2.0; retain its license and notices when adapting it.

**This is adequate published data to justify an exploratory simulation branch. It is not proof of exact current-production firmware behavior or prolonged thermal capacity.** The JSON does not identify the firmware version, motor serial/batch, exact gain sweep, or held-out validation error. The recorder records supply information but does not itself prove the conditions used to produce this JSON; its `--vin` default even remains a generic 15 V, so do not treat that script default as a safe operating voltage. No raw-data package tied specifically to this exact fit was verified in this pass.

The fit's zero-speed electrical ceiling computes to approximately 3.25 N·m before modeled friction, higher than the approximately 1.91 N·m manufacturer stall figure. Open Duck's XML similarly allows approximately 3.23 N·m. These are fitted/model limits, not physical rated torque. If this option is selected, compare fit behavior with conservative manufacturer bounds and train a bounded uncertainty ensemble; do not allow a learned gait to depend on unexplained excess torque. That is simulation engineering using public data, not owner servo characterization.

Before adapting the class, test reset initialization: `q_target_smooth` is initialized in `load_log`, while a live training environment does not necessarily load a recording. CPU and cloud backends need matching initialization, limits and target updates. The XL330 integration in our project does not automatically validate this different control class.

## Open Duck evidence and reusable stack

The [Open Duck Mini V2 BOM](https://docs.google.com/spreadsheets/d/1gq4iWWHEJVgAA_eemkTEsshXqrYlFxXAPwO515KpCJc/edit) explicitly specifies fourteen **7.4 V STS3215** motors and cautions buyers to choose that version. It also lists a Waveshare bus adapter, Pi Zero 2 W, BNO055 IMU and 2S power arrangement. Its listed servo allowance is 14 per unit in the sheet's project-cost context; it is not a current delivered US quote. This establishes a plausible budget tier within our overall budget, not a purchase-ready BOM.

The [project hub](https://github.com/apirrone/Open_Duck_Mini) identifies BAM as the actuator-identification tool and shows real walking demonstrations. This gives stronger deployment precedent than a bare simulator file. The biped policy does not transfer directly to our 12-joint quadruped; robot morphology, observations and action ordering all change.

Inspected related code pins:

| Repository | Inspected commit | Relevant finding |
|---|---|---|
| [Open Duck Mini Runtime](https://github.com/apirrone/Open_Duck_Mini_Runtime) | `32037347dc43186a017f2116bcfde7c461b81f54` | `scripts/configure_motor.py` sets position mode, P32/I0/D0 and acceleration fields to zero; runtime uses Rustypot Feetech communication |
| [Open Duck Playground](https://github.com/apirrone/Open_Duck_Playground) | `b9be205ac64488c23504ca42e5ec790337adeec3` | JAX/CUDA training; XML contains approximate fitted damping, friction and armature with a position actuator |
| [zeroth-robotics/bam-feetech](https://github.com/zeroth-robotics/bam-feetech) | `61b4d0b0c09b98c02e69deafffb06da0d5ca3f28` | Despite its repository name, inspected default checkout contains no STS3215/STS3250 model or Feetech control implementation; not evidence of an additional fit |

The Playground XML currently uses damping 0.56, friction loss 0.068, armature 0.027 and position gain 13.37, with force limits ±3.23. These differ from the pinned BAM values and should not be mixed into one supposedly calibrated model. Its plain fixed force clipping also does not establish a loaded torque-speed envelope. The code is useful precedent for task design; the existing BAM-based CPU/cloud direction remains the cleaner integration route.

The hub has Apache-2.0 licensing. Runtime and Playground checkouts have no root license file; some Playground source files carry Apache-2.0 notices. Preserve per-file licenses and verify the relevant files before copying. Do not infer blanket licensing of separate repositories from the hub's license. No code from these exploratory clones was incorporated into cheetah-pup.

## Consequences for the project

1. Make a second parametric assembly using C001 manufacturer shaft geometry and 55 g motor masses. Recompute achievable hip spacing, leg workspace, mass distribution and gait loads. Preserve the current XL330 branch as the small option.
2. Use the published 7.4 V/P32 model as the center of an exploratory training ensemble, with explicit uncertainty in torque, supply, delay and backlash. Verify CPU/cloud parity and avoid double-counting passive damping or armature.
3. Treat power as a new design choice. Feetech TTL packets and connectors require a Feetech-compatible bus interface; common 1 Mbps TTL signaling does not make the current Dynamixel protocol interchangeable. A 2S pack reaches 8.4 V, above the C001 page's listed 7.4 V upper range. Resolve the selected dated specification and regulation before a battery design; a project's direct-battery precedent is not a manufacturer voltage rating.
4. Keep normal assembled-robot validation: joint direction/offset checks, power/current/temperature observation and tethered initial walking. These are compatible with the owner's refusal to build a servo test bench or fit models.

Do not select STS3250 merely for its larger stall number: no reusable exact fitted actuator data was located, and twelve motors add 234 g over C001. A first-person STS3250 bench report exists, but spot torque/backlash measurements are not a downloadable identified dynamic model with a matching firmware/control law. No smaller alternative serial servo with a stronger exact fitted-data trail was established in this bounded Feetech-focused search.
