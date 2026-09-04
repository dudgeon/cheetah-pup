# Prescribed crawl load and timing screen

This evaluates the illustrated motion with the current rigid-body assembly estimates. It does not demonstrate a trained policy, feasible dynamic gait, or accurate real-servo tracking.

Screened 384 poses, including each body shift and foot swing. Model mass: 613.0 g. Configuration SHA-256: `dd33c5482622efff164d2a07d5c68eb4f882d450b0fbcd53efcfdbeaf6da86a1`.

Selected gait: 140 mm base height, 20 mm advance per cycle and 12 mm peak foot lift. The COM target lies 25% of the way from the four-foot footprint center toward the supporting triangle's centroid.

## Whole-cycle static demand

Every sampled pose has nonnegative vertical support forces balancing the modeled robot's weight and moments. This baseline chooses the smallest summed squared foot forces; joint effort is optimized separately below. Gravity from all moving links is included. Maximum floating-base force residual: 8.88e-15 N; moment residual: 6.29e-16 N·m.

Peak static demand is **0.0934 N·m** at `RR_knee`, giving **1.07×** margin to the configured continuous-torque screen. Proposed 1.5× margin across all sampled poses: **not met**.

| Joint | Static peak, N·m | Static RMS, N·m |
|---|---:|---:|
| FL_hip_roll | 0.0740 | 0.0375 |
| FL_hip_pitch | 0.0596 | 0.0256 |
| FL_knee | 0.0718 | 0.0357 |
| FR_hip_roll | 0.0740 | 0.0375 |
| FR_hip_pitch | 0.0315 | 0.0173 |
| FR_knee | 0.0840 | 0.0452 |
| RL_hip_roll | 0.0740 | 0.0384 |
| RL_hip_pitch | 0.0309 | 0.0172 |
| RL_knee | 0.0807 | 0.0454 |
| RR_hip_roll | 0.0740 | 0.0384 |
| RR_hip_pitch | 0.0463 | 0.0223 |
| RR_knee | 0.0934 | 0.0545 |

The RMS column describes equal-time torque demand, not temperature or electrical current. Passing this static screen does not establish loaded motor speed, actuator tracking or walking.

## Alternative static contact-load allocation

Keep every pose fixed. For four feet, parameterize the one-dimensional null space of vertical force/roll/pitch balance. Nonnegative loads define an interval; enumerate its endpoints and intersections of signed affine joint-torque lines to minimize the maximum absolute joint torque. Three-foot loads are uniquely determined.

Optimizing vertical load sharing during four-foot support gives a full-cycle peak of **0.0934 N·m**, a **1.07×** margin. The limiting joint is `RR_knee` at frame 181. Proposed 1.5× margin: **not met**. Maximum base force/moment residuals remain 8.88e-15 N / 6.29e-16 N·m.

An ideal static force-allocation bound for these fixed poses, not an implemented controller. Position-controlled servos do not automatically realize these contact loads; compliance, contact sensing/estimation and tracking remain unvalidated. This prevents judging the actuator only from the arbitrary minimum-squared-foot-force allocation.

## Timing the same motion

| Case | Cycle | Progress | Peak joint speed | Fraction of 103 rpm unloaded speed | Peak base acceleration | Base force residual | Base moment residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| illustration_timing | 6.400 s | 0.0031 m/s | 1.27 rad/s | 0.12× | 0.40 m/s² | 0.21 N | 0.000572 N·m |
| initial_speed_goal | 0.400 s | 0.0500 m/s | 20.32 rad/s | 1.88× | 101.46 m/s² | 53.9 N | 0.323 N·m |

Periodic central differences include the wrap between cycles with one stride of world translation. Inverse dynamics computes `M(q) qacc + bias(q, qvel)`. Nonnegative vertical foot loads minimize the residual of all six floating-base equations; moments are divided by a documented 0.1 m reference length for this least-squares calculation.

**A nonzero base residual is an unprovided force or moment.** Vertical forces alone cannot create lateral body acceleration. Horizontal contact forces and their joint torques must be solved before claiming a dynamically feasible gait. The JSON includes conditional joint-torque values for diagnostics; these are not validated actuator demands. Accelerating the visual demonstration to the walking target is not a gait controller.

The [manufacturer's XL330 specification](https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/) gives 103 rpm unloaded at 5 V. This is only an upper-reference speed, not a loaded torque-speed envelope. The [manufacturer torque estimate](https://www.robotis.us/dynamixel-xl330-m288-t/) is not a guaranteed continuous thermal rating.

## Limited stance and stride alternatives

Robot geometry, motor envelopes and mass, all mass allowances, joint limits, configured COM-shift fraction and 12 mm foot lift; only gait base height and stride vary.

Sweep validity means reach, joint limits and static support only. Self-interference and assembly clearance were not audited for every variant; the separate assembly audit applies to its explicitly selected trajectory, not this whole sweep.

| Base height | Stride | Geometry | Static peak | Minimax peak | Minimax margin | Target speed / unloaded limit |
|---|---|---|---:|---:|---:|---:|
| 124 mm | 20 mm | Reach/limits pass | 0.1272 N·m | 0.1272 N·m | 0.79× | 1.38× |
| 124 mm | 30 mm | Reach/limits pass | 0.1351 N·m | 0.1351 N·m | 0.74× | 1.02× |
| 124 mm | 40 mm | Reach/limits pass | 0.1418 N·m | 0.1418 N·m | 0.71× | 0.85× |
| 130 mm | 20 mm | Reach/limits pass | 0.1166 N·m | 0.1166 N·m | 0.86× | 1.50× |
| 130 mm | 30 mm | Reach/limits pass | 0.1242 N·m | 0.1242 N·m | 0.81× | 1.10× |
| 130 mm | 40 mm | Reach/limits pass | 0.1305 N·m | 0.1305 N·m | 0.77× | 0.92× |
| 135 mm | 20 mm | Reach/limits pass | 0.1061 N·m | 0.1061 N·m | 0.94× | 1.66× |
| 135 mm | 30 mm | Reach/limits pass | 0.1133 N·m | 0.1133 N·m | 0.88× | 1.21× |
| 135 mm | 40 mm | Reach/limits pass | 0.1191 N·m | 0.1191 N·m | 0.84× | 1.01× |
| 138 mm | 20 mm | Reach/limits pass | 0.0988 N·m | 0.0988 N·m | 1.01× | 1.78× |
| 138 mm | 30 mm | Reach/limits pass | 0.1057 N·m | 0.1057 N·m | 0.95× | 1.30× |
| 138 mm | 40 mm | Reach/limits pass | 0.1109 N·m | 0.1109 N·m | 0.90× | 1.08× |
| 140 mm | 20 mm | Reach/limits pass | 0.0934 N·m | 0.0934 N·m | 1.07× | 1.87× |
| 140 mm | 30 mm | Reach/limits pass | 0.0999 N·m | 0.0999 N·m | 1.00× | 1.37× |
| 140 mm | 40 mm | Reach/limits pass | 0.1047 N·m | 0.1047 N·m | 0.96× | 1.15× |

This limited posture search cannot select or reject the actuator architecture. Passing the two scalar screens is still insufficient for feasible walking; failure calls for gait/contact optimization, geometry/mass changes or a stronger actuator comparison.

## Flat-floor geometry

Minimum modeled non-foot primitive clearance is **6.12 mm**, at `FR_lower_bar`. Physical motor envelopes are included regardless of contact flags; 24 nonphysical cable/port keepouts are excluded. Unhandled geometry types: none.

Physical component geometry against a perfectly flat plane only; group-5 non-colliding cable/port reservations are excluded. A 12 mm peak swing height gives no guaranteed clearance over a 10 mm doorway threshold along the whole foot path. No carpet deformation, obstacle placement, self-interference or manufacturing tolerance is assessed here.

## Remaining limits

- Masses and inertias remain assembly estimates. The derivative screen includes the compiled model's armature, which may still be a provisional value.
- No actuator torque-speed coupling, delay, friction, supply droop, thermal model or tracking controller is included in these demand calculations.
- The smooth IK illustration was designed to explain support transfer, not optimized for torque or speed. Its feasibility must not be equated with quadruped architecture feasibility.
- Faster retiming multiplies joint speed by inverse cycle duration and inertial accelerations approximately by its square. The 0.05 m/s target needs a new gait, a longer stride, or explicit dynamic optimization if this retiming fails.
- No RL training, forward walking rollout, threshold traversal or real hardware validation is performed.

Reproduce from the repository root:

```sh
uv run python -m cheetah_pup.gait_load --config config/robot.json --output reports/gait-load-validation.json
```
