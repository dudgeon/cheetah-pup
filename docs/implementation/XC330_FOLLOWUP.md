# Stronger X330 actuator follow-up

Reviewed 2026-09-04. This is a component research note, not a hardware selection or a validated XC330 simulation.

**XC330 merits the next comparison because it adds relatively little mass and preserves the X330 mechanical envelope. It does not yet solve the requirement for an actuator model with sufficiently clear provenance to avoid owner characterization.** The 5 V M288 is electrically convenient; the higher-voltage T288 has the more relevant published identification work. They are different motors, and neither inherits the XL330 BAM fit.

## Hardware comparison

| Property | XL330-M288-T, current candidate | XC330-M288-T | XC330-T288-T |
|---|---:|---:|---:|
| Reference rail for this table | 5 V | 5 V | 11.1 V |
| Manufacturer mass | 18 g | 23 g | 23 g |
| Advertised overall dimensions | 20 × 34 × 26 mm | 20 × 34 × 26 mm | 20 × 34 × 26 mm |
| Manufacturer estimated continuous torque | 0.10 N·m | 0.186 N·m | 0.184 N·m |
| Unloaded speed from manual | 103 rpm | 81 rpm | 65 rpm |
| Stall torque, **not** a continuous rating | 0.52 N·m | 0.93 N·m | 0.92 N·m |
| Current US list price, each | See current BOM quote | $103.39 | $103.39 |
| Published model lead | BAM XL330 M6, provenance gate open | No unambiguous identified fit found in this bounded search | ToddlerBot XC330 model, exact acquisition settings still unclear |

The continuous figures are ROBOTIS's estimates, calculated from 20% of stall torque; they are not measured thermal endurance guarantees. Sources: [XL330 listing](https://www.robotis.us/dynamixel-xl330-m288-t/), [XC330-M288 listing](https://www.robotis.us/dynamixel-xc330-m288-t/), [XC330-T288 listing](https://www.robotis.us/dynamixel-xc330-t288-t/), and [current ROBOTIS M288 manual and family comparison](https://docs.robotis.com/docs/dxl/model_reference/x_series/xc_series/xc330-m288/). The M288 US store says 65 rpm, conflicting with the manual's 81 rpm at 5 V; use the manual for this comparison and resolve the discrepancy before selection. The T288 listing indicates a lead time.

Twelve XC330s add 60 g to the 613 g assembly allowance, giving **673 g before any power-system changes**, and cost **$1,240.68 before tax, delivery, spares and the rest of the robot**. That presses hard against the existing $600–$1,500 whole-project hardware budget. An eight-pitch/knee upgrade would add 40 g and cost $827.12 for those eight motors alone; it remains only a comparison option, since the roll-joint loads and a mixed actuator model would also need checking.

The M288 operates at 3.7–6 V, recommended 5 V. The T288's published speed/torque points are 9, 11.1 and 12 V; it requires a different servo rail from our 5 V design. ToddlerBot's T288 model must not be relabeled as a 5 V M288 model. Neither stronger candidate makes the current fast-retimed crawl feasible by itself: both have lower unloaded speed than XL330.

## Mechanical reuse is credible, but mass properties must change

The [XC330 manual](https://emanual.robotis.com/docs/en/dxl/x/xc330-m288/#drawings) links the **same manufacturer drawing download 1986 and STEP download 1987** already inspected for our [X330 assembly frame](SERVO_GEOMETRY_SOURCES.md). This is stronger evidence for common stock casing/horn geometry than matching the three advertised dimensions alone. The connector specification also uses JST EHR-03 cable housings and B3B-EH-A headers. The shared [inertia sheet](https://www.robotis.com/service/download.php?no=2136) provides separate mass properties for the motor variants.

A candidate simulation should therefore retain the independently checked shaft/horn convention while replacing each selected motor's mass, COM and inertia with the correct XC330 entries. Repeat the load and clearance audit with the actual selected horn/support accessories. This note does not certify final brackets, fastener engagement, stiffness, mated plugs or flexible cables.

## What ToddlerBot actually provides

The [ToddlerBot paper](https://arxiv.org/html/2502.00893v4) identifies XC330-T288 on a 12 V-class system, on its neck, waist, hip yaw and gripper. It reports transfer between two robot instances without repeating motor identification. Its identification method uses position tracking of chirp signals and a fitted actuation law. It also notes higher tracking error/backlash for XC330 than the larger motors chosen for demanding leg joints. This is useful evidence for model reuse across units, but not a demonstration of our quadruped or the low-voltage M288.

The repository was inspected at **`e337f3b177b4b53abff70b31d1695a7b66cc6d2e`**. It contains two materially different parameter generations:

| Parameter | Current `descriptions/default.yml`, XC330 | Older `descriptions/sysID_XC330/config_dynamics.json` |
|---|---:|---:|
| Passive damping, N·m·s/rad | 0.001 | 0.134 |
| Armature, kg·m² | 0.0048 | 0.0035 |
| Friction loss, N·m | 0.006 | 0.014 |
| Model acceleration torque limit, N·m | 0.68 | 0.99 |
| Speed at torque transition, rad/s | 1.0 | 3.29 |
| High-speed endpoint, rad/s | 6.52 | 10.0 |

These are empirical **model parameters**, not hardware continuous ratings. The current model additionally specifies active damping 0.341 N·m·s/rad, braking limit 1.54 N·m, torque 0.49 N·m at the speed endpoint, hardware-P/simulation-P ratio 150 and passive/active gain ratio 3. It implements asymmetric acceleration/braking behavior. The old configuration instead divides P by 128. Do not combine either parameter set with a controller from the other generation.

Pinned sources: [current parameters](https://github.com/hshi74/toddlerbot/blob/e337f3b177b4b53abff70b31d1695a7b66cc6d2e/toddlerbot/descriptions/default.yml), [older parameters](https://github.com/hshi74/toddlerbot/blob/e337f3b177b4b53abff70b31d1695a7b66cc6d2e/toddlerbot/descriptions/sysID_XC330/config_dynamics.json), [controller implementation](https://github.com/hshi74/toddlerbot/blob/e337f3b177b4b53abff70b31d1695a7b66cc6d2e/toddlerbot/sim/motor_control.py), and [gain conversion](https://github.com/hshi74/toddlerbot/blob/e337f3b177b4b53abff70b31d1695a7b66cc6d2e/toddlerbot/sim/robot.py).

Public [collection code](https://github.com/hshi74/toddlerbot/blob/e337f3b177b4b53abff70b31d1695a7b66cc6d2e/toddlerbot/policies/sysID.py) uses 0.1–10 Hz chirps, amplitudes 0.25/0.5/0.75 rad and multiple P gains; [documentation](https://github.com/hshi74/toddlerbot/blob/e337f3b177b4b53abff70b31d1695a7b66cc6d2e/docs/tools/02_sysID.rst) describes a weighted arm and 2 Mbaud communication. [Optimization code](https://github.com/hshi74/toddlerbot/blob/e337f3b177b4b53abff70b31d1695a7b66cc6d2e/toddlerbot/tools/run_sysID.py) is public. Exact acquisition voltage, firmware version, the raw trials producing the current parameter set and its complete correspondence to those older collection scripts were **not established** in this review. The tiny `sysID_XC330_cache.pkl` contains a robot-name cache, not identified trial data. The repository is MIT licensed; no code or models were copied into our runtime in this step.

## LEAP/Playground is not a clean XC330 model source

At MuJoCo Playground commit **`8a4b4642d8eba8a80ac99ed125cb62c16e1457ad`**, its [LEAP README](https://github.com/google-deepmind/mujoco_playground/blob/8a4b4642d8eba8a80ac99ed125cb62c16e1457ad/mujoco_playground/_src/manipulation/leap_hand/README.md) explicitly names **XL330-M288-T**, while linking to the XC330 manual for the PD table. The [XML](https://github.com/google-deepmind/mujoco_playground/blob/8a4b4642d8eba8a80ac99ed125cb62c16e1457ad/mujoco_playground/_src/manipulation/leap_hand/xmls/leap_rh_mjx.xml) uses `0.600 A × 0.366 N·m/A = 0.2196 N·m`, PD gain 3, damping 0.2 and guessed friction loss 0.02.

The [Playground paper, C.42](https://arxiv.org/html/2502.08844v1) describes rotor measurements and substantial randomization around approximate friction and controller behavior. Those useful techniques do not establish a measured 5 V XC330-M288 model. Preserve the SKU ambiguity rather than treating every LEAP implementation as interchangeable.

## Recommendation for the next refinement

Keep XL330 as the current software baseline. Compare the correctly massed XC330 variants against a feasible, smoother gait before making a purchase decision, while resolving the existing model-source gaps through published material. The most promising reuse lead is ToddlerBot's **T288**, with a power-system and budget tradeoff; the **M288** is the simpler electrical alternative but currently has the weaker identified-model evidence. Neither is ready to replace XL330 merely by increasing a torque limit. Do not add an owner servo-characterization task to close these gaps.
