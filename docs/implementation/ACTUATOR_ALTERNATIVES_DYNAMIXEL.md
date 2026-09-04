# Dynamixel alternatives for the next robot iteration

Research date: 2026-09-04. Scope: public actuator-model evidence and practical component fit for a 12-axis, small quadruped, with a $600–$1,500 total hardware budget, 10–15 minutes of walking, carpet and small thresholds. No owner actuator characterization is proposed. No hardware selection is finalized here.

**Practical shortlist: XL430-W250-T for a somewhat larger, affordable robot; XC330-T288-T for keeping the present compact assembly; XC430-T240BB-T only if the budget grows.** There is enough public modeling work to explore these choices in simulation. None currently establishes the full combination of exact motor behavior, walking endurance and our custom geometry. Missing provenance should become explicit model uncertainty and a bounded source review, rather than an indefinite bar to exploratory RL.

## Hardware cost, mass and size

Prices are current US manufacturer list prices, before tax, shipping and spares. Motor-set mass does not include brackets, controller, battery or cables. The final column is arithmetic using the current **397 g nonmotor allowance**; it is not a new robot mass prediction because larger motors require redesigned parts and power components.

| Exact SKU | Reference rail | Estimated continuous torque | Unloaded speed | Unit mass | Unit price | Set for 12 axes: mass / price | Plus current nonmotor allowance |
|---|---:|---:|---:|---:|---:|---:|---:|
| XL330-M288-T, baseline | 5 V | 0.10 N·m | 103 rpm | 18 g | $27.49 | 216 g / $329.88 | 613 g |
| XC330-M288-T | 5 V | 0.186 N·m | 81 rpm | 23 g | $103.39 | 276 g / $1,240.68 | 673 g |
| XC330-T288-T | 11.1 V | 0.184 N·m | 65 rpm | 23 g | $103.39 | 276 g / $1,240.68 | 673 g |
| XL430-W250-T | 11.1 V | 0.28 N·m | 57 rpm | 57.2 g | $27.50 | 686.4 g / $330.00 | 1,083.4 g |
| XC430-T240BB-T | 12 V | 0.38 N·m | 70 rpm | 65 g | $137.89 | 780 g / $1,654.68 | 1,177 g |
| XM430-W210-T | 12 V | 0.60 N·m | 77 rpm | 82 g | $310.39 | 984 g / $3,724.68 | 1,381 g |
| XM430-W350-T | 12 V | 0.82 N·m | 46 rpm | 82 g | $310.39 | 984 g / $3,724.68 | 1,381 g |
| 2XL430-W250-T, **two axes per module** | 11.1 V | 0.28 N·m per axis | 57 rpm | 98.2 g per module | $149.39 per module | **6 modules:** 589.2 g / $896.34 | 986.2 g |

ROBOTIS calls these continuous figures **estimates derived from 20% of stall torque**. They do not establish a measured temperature-versus-load envelope. Sources: [XL330 store](https://www.robotis.us/dynamixel-xl330-m288-t/), [XC330-M288 store](https://www.robotis.us/dynamixel-xc330-m288-t/), [XC330-T288 store](https://www.robotis.us/dynamixel-xc330-t288-t/), [XL430 store](https://www.robotis.us/dynamixel-xl430-w250-t/), [XC430 store](https://www.robotis.us/dynamixel-xc430-t240bb-t/), [XM430-W210 store](https://www.robotis.us/dynamixel-xm430-w210-t/), [XM430-W350 store](https://www.robotis.us/dynamixel-xm430-w350-t/), [2XL430 store](https://www.robotis.us/dynamixel-2xl430-w250-t/).

The XL430 price is $27.50 on the current manufacturer page and its structured product data. Its store mass says 65 g, while the [manufacturer manual](https://emanual.robotis.com/docs/en/dxl/x/xl430-w250/#specifications) says **57.2 g**; the table uses the manual and retains this discrepancy. XC330-M288 store speed also conflicts with its manual; see [the XC330 note](XC330_FOLLOWUP.md). For XC430 the store combines 11.1 V with torque/speed values that its [manual](https://emanual.robotis.com/docs/en/dxl/x/xc430-t240bb/#specifications) assigns to **12 V**; the table uses the latter.

The single-axis X430 cases are **28.5 × 46.5 × 34 mm**, versus the present X330's 20 × 34 × 26 mm. The [2XL430 module](https://emanual.robotis.com/docs/en/dxl/x/2xl430-w250/#specifications) is 36 × 46.5 × 36 mm. All X430 options require a new assembly layout, shaft offsets, mass properties and clearances. Six dual modules provide twelve axes numerically, but their fixed axis arrangement does **not** reproduce four serial three-axis legs without a new mechanical design. A possible four-dual-hip/four-single-knee combination costs $707.56 in motors and weighs 621.6 g, before proving that layout.

XL430 and 2XL430 permit 6.5–12 V, recommended 11.1 V; a fully charged three-cell pack reaches 12.6 V, so it cannot simply connect directly under that specification. XC430 permits 6.5–14.8 V, recommended 12 V. The candidate power rail must match the model's chosen supply, with a separate compute rail where needed. TTL Protocol 2.0 and position-command software remain reusable, but addresses, supported modes, current/load interpretation and electrical limits are SKU-specific. XL430/XC430 load telemetry is inferred, not measured torque. [XL430 manual](https://emanual.robotis.com/docs/en/dxl/x/xl430-w250/), [XC430 manual](https://emanual.robotis.com/docs/en/dxl/x/xc430-t240bb/).

## Published model and training evidence

### XL430: new, relevant Open Ant evidence

[Open Ant](https://arxiv.org/html/2607.18488v1), released July 2026, provides a quadruped simulation, learning code and physical experiments. It measured XL430 joint stiffness using known angular deflections and a scale. Damping came from a datasheet ratio, velocity gain was tuned, and a one-step observed delay was not explicitly modeled. Its Lite robot overloaded the XL430 in the authors' much longer-leg geometry; the reported approximately 30-minute thermal shutdown occurred in a deliberately stalled heavy-load test, not a 30-minute walking guarantee. Its paper mixes XM430-W210 in the main description with W350 in the identification appendix, so exact variants need care.

Repository inspected at **`a3c5a385197e580943b16c67cc25d2b6e5a35942`**:

- [MuJoCo model](https://github.com/Openmind-Research-Institute/open-ant/blob/a3c5a385197e580943b16c67cc25d2b6e5a35942/sim/assets/ant_with_camera_after_sys_id.xml): hip P = 19.455 N·m/rad, velocity gain = 2.5 N·m·s/rad, damping = 0.235 N·m·s/rad, command filter time constant = 0.12 s and flat torque clamp = ±1.4 N·m. These correspond to the paper's XL430 identification values, even though the newer hardware BOM uses XC430 hips.
- [Hardware driver](https://github.com/Openmind-Research-Institute/open-ant/blob/a3c5a385197e580943b16c67cc25d2b6e5a35942/embodied_ant_env/motor_controller.py): extended-position mode and **50% PWM limit**. It does not explicitly write P/D gains. The simulation's flat force clamp is not a voltage/current/loaded-speed model of that limit.
- [BOM and setup](https://github.com/Openmind-Research-Institute/open-ant/blob/a3c5a385197e580943b16c67cc25d2b6e5a35942/Readme.md): 12 V supplies, 1 Mbaud, and XM430-**W350** knee motors. [SAC](https://github.com/Openmind-Research-Institute/open-ant/blob/a3c5a385197e580943b16c67cc25d2b6e5a35942/agents/sac/sac_cleanrl.py), SARSA and the [simulation environment](https://github.com/Openmind-Research-Institute/open-ant/blob/a3c5a385197e580943b16c67cc25d2b6e5a35942/sim/ant_mujoco.py) are public. Repository license: [MIT](https://github.com/Openmind-Research-Institute/open-ant/blob/a3c5a385197e580943b16c67cc25d2b6e5a35942/LICENSE), with bundled component notices.

**Assessment:** useful evidence for an affordable XL430 exploration, stronger than assuming ideal servos. Resolve the source configuration mismatch in software and represent plausible behavior variation. Raw characterization trials and exact firmware settings were not established here. Do not import the W350 knee fit as a W210 fit, or label the entire current XML as an identified XC430 robot. No existing Open Ant policy is a policy for our different twelve-axis geometry.

### ToddlerBot: richer loaded-speed models for XC/XM and dual XL430

[ToddlerBot's paper](https://arxiv.org/html/2502.00893v4) reports actuator identification and successful policy transfer to a second robot without repeating that identification. It names XC330-T288, XC430-T240BB, XM430-W210, 2XL430-W250 and 2XC430-W250. The current repository additionally contains an explicitly named XM430-W350 model. The system is 12 V-class, but the exact voltage and firmware for each current parameter fit were not established in this bounded review.

At **`e337f3b177b4b53abff70b31d1695a7b66cc6d2e`**, [default.yml](https://github.com/hshi74/toddlerbot/blob/e337f3b177b4b53abff70b31d1695a7b66cc6d2e/toddlerbot/descriptions/default.yml) publishes these empirical model values:

| Model key | Acceleration plateau, N·m | Plateau end, rad/s | High-speed endpoint: rad/s / N·m | Braking clamp, N·m | Armature, kg·m² | Active damping, N·m·s/rad |
|---|---:|---:|---:|---:|---:|---:|
| XC330 | 0.68 | 1.0 | 6.52 / 0.49 | 1.54 | 0.0048 | 0.341 |
| XC430 | 1.47 | 1.2 | 7.00 / 0.19 | 2.00 | 0.0044 | 0.173 |
| XM430-W210 | 1.94 | 0.8 | 7.60 / 0.40 | 2.20 | 0.0022 | 0.183 |
| XM430-W350 | 2.95 | 0.5 | 4.55 / 0.36 | 3.00 | 0.0025 | 0.231 |
| 2XL430 | 0.93 | 2.0 | 5.97 / 0.08 | 1.40 | 0.0083 | 0.162 |

These are fitted **dynamic limits, not continuous torque ratings**. The [controller](https://github.com/hshi74/toddlerbot/blob/e337f3b177b4b53abff70b31d1695a7b66cc6d2e/toddlerbot/sim/motor_control.py) has asymmetric braking, active damping and backdrive behavior; the configuration converts hardware P by 150 and D by 16. Use its matching friction and passive-damping values as well. Do not transfer the 2XL430 fit to a single XL430 merely because their advertised stall ratings match. Older `sysID_*` JSON files differ materially from this current configuration and use a different controller generation.

[MuJoCo/MJX locomotion code](https://github.com/hshi74/toddlerbot/tree/e337f3b177b4b53abff70b31d1695a7b66cc6d2e/toddlerbot/locomotion), models, hardware runtime, chirp-collection/optimization tooling and [MIT license](https://github.com/hshi74/toddlerbot/blob/e337f3b177b4b53abff70b31d1695a7b66cc6d2e/LICENSE) are public. The older logs and full acquisition settings corresponding to the current fitted values were not established here. Existing code and identified parameters can be reused without requiring the owner to operate that collection tooling.

**Assessment:** XC330-T288 has the best combination of compactness and a transferable empirical-model lead. XC430-T240BB offers a useful larger-model comparison, but twelve already exceed the total budget. XM430 variants are technically credible but financially inappropriate for twelve joints in this project. A few upgraded joints are possible only after the load distribution and mixed-motor power/control design justify them.

### BAM search boundary

The project-pinned BAM `62bd8ce12154340be97e06f7f41a0ca8f116d967` and inspected newer `57d13ead53206a6bf0db3d66f86506ae8c2ce01a` provide Dynamixel parameter directories for XL320, XL330, MX64 and MX106, **not XL430, XC430 or XM430**. See [the inspected parameter tree](https://github.com/Rhoban/bam/tree/57d13ead53206a6bf0db3d66f86506ae8c2ce01a/bam/params). Do not present a BAM-family adapter as an identified fit for those absent SKUs. Feetech alternatives are covered separately.

## Recommended next comparison

1. Retain the compact XL330 baseline. The [CPU learning experiment](../../reports/cpu-rl-smoke.md) is complete; its narrow standing result validates the training pipeline.
2. Build one **original X430-sized assembly candidate with XL430-W250-T**, preserving the three-axis leg concept but redesigning the physical installation. Use the public Open Ant identification as an exploratory model with declared uncertainty, a loaded-speed constraint and CPU parity checks. Evaluate the assembled candidate's mass, support loads, motor-speed demands and conservative thermal proxies before a purchase.
3. Compare against the **same-sized-as-current XC330-T288** with its own mass properties and the matching ToddlerBot model. Include its new power rail and approximately $1,241 motor bill in the decision. The 5 V M288 remains interesting if compatible published modeling evidence becomes clear.

This comparison can proceed with modest local or capped cloud training once the software checks pass. A source's missing firmware hash alone need not prevent an exploratory job; conclusions must remain conditional on the modeled uncertainty. None of the sources proves 10–15 minutes on our carpet/threshold task, and detailed manufacturing CAD or a custom PCB should follow the choice that survives this comparison.
