# Motor viability and mass sensitivity

**Judgment: the XL330 is plausible for low-speed experiments, but marginal in the present 613 g study. It is not ready for a hardware commitment aimed at carpet, thresholds and 10–15 minutes of walking.**

The current dynamic model can stand; that result and the failed preset crawl do not establish a fundamental motor yes/no. A poor open-loop gait can fail with adequate motors. Equally, RL cannot provide sustained torque, speed or power the hardware lacks.

## Fixed geometry, unchanged motors

| Total allowance | Nonmotor mass scale | Peak static torque | Margin to 0.10 N·m estimate | 1.5× screen |
|---|---:|---:|---:|---|
| 414 g | 50% | 0.0627 N·m | 1.59× | Pass |
| 454 g | 60% | 0.0689 N·m | 1.45× | Fail |
| 494 g | 70% | 0.0750 N·m | 1.33× | Fail |
| 534 g | 80% | 0.0811 N·m | 1.23× | Fail |
| 613 g | 100% | 0.0934 N·m | 1.07× | Fail |
| 692 g | 120% | 0.1056 N·m | 0.95× | Fail |

For this particular geometry and sampled crawl, a mass allowance near **439 g** reaches the proposed static reserve. This is a sizing result, not a feasible lighter BOM; all 12 motors still contribute 216 g.

At 613 g, the present 0.0934 N·m peak would need about 0.140 N·m of continuous capability to satisfy the chosen reserve without changing motion/mass. Replacing the servo with a heavier one changes that demand; 0.140 N·m is not a sufficient specification for an arbitrary replacement.

The nominal 0.10 N·m figure is [ROBOTIS’s 20%-of-stall continuous estimate](https://www.robotis.us/dynamixel-xl330-m288-t/), not a measured guarantee for this mounting or duty cycle. The [manual](https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/) separately publishes 5 V stall torque and unloaded speed; stall and unloaded speed are distinct endpoints, not simultaneously available walking output.

## Minimum refinement before cloud work

1. Refine the mass budget and mechanically valid workspace. Compare a lighter assembly against one stronger model-supported actuator before committing to hardware. Exact screws and cosmetic CAD can wait; cable/board/battery volumes, moving mass and reliable contacts cannot.
2. Extend the verified standing interface to a useful movement task: 45 proprioceptive observations, 12 bounded targets, 50 Hz commands, BAM physics/delay, collision-free resets, fault/fall detection and useful rewards. Compare learned behavior with the existing fixed-target baseline.
3. Bring the CPU and GPU actuator implementations into agreement and make voltage/gain/delay/friction uncertainty explicit. Small exploratory RL can proceed while stock-servo provenance is open; expensive optimization and hardware transfer should not treat it as resolved.
4. The short CPU learning/checkpoint/evaluation path is now proven: [16,384 PPO transitions](cpu-rl-smoke.md), with no improvement over fixed standing targets. Use CPU for the next task-debugging stage before paying for cloud. This experiment does not validate recovery, walking or terrain transfer.

No additional servo characterization is assigned to the owner. Further model accuracy must come from published evidence, defensible uncertainty bounds, and later normal whole-robot validation.

## Limits

- Motor masses, casing geometry and source tensors never shrink. Only other provisional mass allowances change.
- Uniform allowance scaling is not a new BOM, structural analysis or verified battery/compute package.
- The 1.5× static reserve is a project screening choice against a manufacturer estimate, not a proven thermal boundary.
- No horizontal gait forces, contact transients, loaded speed envelope, actuator tracking, mass-dependent cable clearance or carpet/threshold performance is established.
