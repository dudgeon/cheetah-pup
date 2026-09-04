# DR-02 — MuJoCo validation of the locked design

**Status**: complete for Phase 1's purpose (viability). Playback page:
https://claude.ai/code/artifact/2db54b1d-2707-4034-895f-95bec2b86281
Source: `docs/design/replay/template.html`, built by `python -m cheetah_pup.build_replay`.
Recordings: `sim/validation/` from `python -m cheetah_pup.validate sim/cheetah_pup.xml`.
Model: `sim/cheetah_pup.xml` from `python -m cheetah_pup.mjcf` (locked design A · M).

## What was tested

| | Result |
|---|---|
| Stand (1 s, hold the keyframe) | 0.8 mm sag, 0.75° droop; knee hold torque **0.244 N·m** (quasi-static estimate 0.22) |
| Open-loop walk, 6 s | stayed up; 0.07 m at 0.012 m/s (commanded 0.062); pitch ≤ 10°, roll ≤ 11° |
| Open-loop trot, 6 s | stayed up; **0.33 m at 0.056 m/s** (commanded 0.168); pitch ≤ 8.5°, roll ≤ 11° |
| Leveled trot (gain 0.7), 6 s | stayed up; 0.62 m at 0.103 m/s; pitch ≤ 18.5°, roll ≤ 19° |
| Peak servo torque in gait | hip and knee touch the 1.91 N·m clamp for 1–4 % of samples during swing; abad ≤ 0.68 N·m |
| Peak joint speed in gait | knee 4.3 rad/s open-loop trot (cap 5.29) |

"Open-loop" = inverse-kinematics joint targets from `cheetah_pup/gait.py` at 50 Hz, no feedback.
"Leveled" = the same plus one geometric term shifting each foot's height by
0.7 × (hip x · sin pitch − hip y · sin roll). Neither balances.

## What it means

1. **The locked geometry is viable on STS3215 servos.** The robot stands with a 10 % duty-cycle
   knee load and survives 6 s of walking and trotting with no controller at all.
2. **Speed loss is posture, not servo capability.** Servo tracking is 2–3° mean error; the front
   feet float ~80 % of the trot cycle because the trunk pitches under the rear legs' push (contact
   duty 21/18/36/40 % vs 50 % commanded). A crude leveling term doubles the speed but oscillates —
   a proportional correction through a laggy position loop has no damping. Balance is the policy's
   job; the hand-written gait is a viability probe, not a controller.
3. **Torque clamp hits are brief and expected**: the firmware loop (kp ≈ 18.8 N·m/rad) saturates
   at ~6° of tracking error during swing reversals. Keep the clamp and BAM's target-rate limit in
   the training model so the policy learns the real robot's limits.

## Model fidelity notes (for the RL environment)

- Primitives with component masses (1.409 kg); no meshes yet. Replace with CAD-derived mass
  properties in Phase 2.
- Servo: PD from BAM's electrical model, clamped at the datasheet stall. Upgrade to BAM's stateful
  MuJoCo actuator (rate-limited target, extended friction) for training.
- Feet: 10 mm spheres, friction 1.0. Contact sensing via MuJoCo touch sensors.
- Sensors present for the observation space: IMU quaternion, gyro, accelerometer; four foot touch
  sensors; joint positions and velocities from the state.
