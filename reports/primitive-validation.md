# Primitive simulation validation

Environment: MuJoCo 3.10.0, Python 3.12.13.

This is an original parametric geometry model and load screen. It is not a trained walking robot, manufacturing CAD, or a calibrated digital twin.

The model has **12 joints** and an estimated **613 g** component budget. The torso envelope is **160 × 70 × 45 mm**; that is not the full robot's exterior size.

## Geometry checks

Analytical forward kinematics were checked at 32 asymmetric poses on all four legs, including rotated/transformed floating bases. Maximum foot-position error: 8.33e-17 m. Jacobians agree with MuJoCo within 9.71e-17 m/rad and independent finite differences within 2.31e-11 m/rad.

## Neutral-pose static load screen

Loads satisfy vertical force and roll/pitch moment balance. Equal sharing is not assumed. Includes modeled link gravity, but excludes dynamic gait forces and motor thermals.

| Supports | Static equilibrium | Peak joint torque | Margin to 0.10 N·m estimate |
|---|---|---:|---:|
| four_feet | Yes | 0.0466 N·m | 2.15× |
| three_feet_lift_FL | Yes | 0.0855 N·m | 1.17× |
| three_feet_lift_FR | Yes | 0.0855 N·m | 1.17× |
| three_feet_lift_RL | No; shift body or use dynamics | — | — |
| three_feet_lift_RR | No; shift body or use dynamics | — | — |
| diagonal_FL_RR | No; shift body or use dynamics | — | — |
| diagonal_FR_RL | No; shift body or use dynamics | — | — |

A three-foot crawl needs the center of mass inside its support triangle. Merely lifting one foot does not guarantee three equal loads. The 0.10 N·m motor figure is a manufacturer estimate, and a 1.5× static margin is a proposed screening target, not a proven thermal limit.

## Parameter sensitivity

Screened 45 combinations of nonmotor mass allowances (±20%), upper/lower link lengths (±10 mm), and stance knee flexion. Motor masses and physical envelopes remain fixed. Every variant's load results are in the JSON. These calculations do not select a manufacturable design or establish a complete gait.

## Ideal-PD sanity observation

Over 5.00 s, base height changed from 0.142 to 0.139 m. Any joint reached the torque limit in 0.0% of steps. This uses arbitrary torque-limited PD gains; no inference about real XL330 performance follows.

## Battery allowance

A hypothetical 2S 650 mAh pack stores 4.81 Wh. At an assumed 80% usable energy and 88% conversion efficiency, 10–15 minutes permits approximately 20.3–13.5 W average combined output. This is a sizing calculation, not a selected pack or runtime result; transient current still needs design analysis.

## Open gates

- **geometry implementation:** pass.
- **motor selection:** open: conservative gait load, model provenance and budget must converge.
- **realistic actuator physics:** not implemented: published BAM integration is next.
- **manufacturing and self collision:** not assessed; primitive motor housings are visual-only.
- **rl training:** not started.
- **carpet and threshold traversal:** not demonstrated.
- **battery runtime:** not demonstrated.
- **hardware validation:** not performed.

See [the implementation plan](../docs/implementation/PLAN.md) and [actuator research](../docs/microduck-review/RESEARCH.md). The JSON beside this file contains complete forces, joint torques, configuration hash and numerical results.
