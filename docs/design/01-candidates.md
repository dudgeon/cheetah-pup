# DR-01 — Leg architecture & proportion candidates

**Status**: awaiting the owner's selection. Review page: https://claude.ai/code/artifact/6b9c92f0-98d5-4cf1-928c-98a28d699ba4
(source: `docs/design/review/template.html`, built by `python -m cheetah_pup.build_artifact`).
Date: 2026-09-04. Analysis code: `cheetah_pup/` (tests in `tests/`).

## What is being decided

All candidates share the Mini Cheetah kinematics fixed in `docs/HANDOFF.md` §3: four legs, each with
abduction/adduction (axis along x, through the hip-pitch axis), hip pitch, and knee pitch, serial 2R
leg, knees pointing backward by default. What differs is **where the knee servo lives and how it
drives the shank** — which changes mass distribution, hip width, build complexity, and how much the
sim has to model beyond the servo itself.

| | **A · Direct drive** | **B · Coaxial hip + belt knee** | **C · Coaxial hip + pushrod knee** |
|---|---|---|---|
| Knee servo location | at the knee, inside the thigh | at the hip, coaxial with the hip servo, outboard of the thigh | same as B |
| Knee transmission | none (horn drives the shank) | GT2 belt, 20T→25T (1.25:1) | printed 4-bar pushrod (~1.2:1 average) |
| Thigh | housing around a 55 g servo | slim printed link with a belt cavity | slim printed link |
| Overall width (size M) | **182 mm** | 253 mm | 253 mm |
| Mass (size M) | 1.41 kg | 1.46 kg | 1.45 kg |
| Knee servo, trot peak | 0.77 N·m (40 % of stall) | 0.65 N·m (34 %) | 0.67 N·m (35 %) |
| Abad / hip, trot peak | 0.52 / 0.31 N·m | 0.54 / 0.32 N·m | 0.53 / 0.32 N·m |
| Peak servo speed, 1.4 Hz trot | 76 % of cap | 94 % of cap | 91 % of cap |
| Electronics fit | yes | yes | yes |
| Sim-to-real modeling | servo only (BAM covers it) | servo + belt compliance/stretch | servo + variable ratio + pin play |

Baseline geometry (size M): thigh 90, shank 85, abad link 40, hips 180 apart (fore-aft), abad axes
70 apart, shell 148 × 62 mm, hip height 120 mm (nominal hip −45°, knee 93° — Mini Cheetah stands at
−46°/92°). Sizes S/L scale the legs by 0.85/1.15.

## Findings that shaped the candidates

1. **The STS3215's speed cap, not its torque, is the binding constraint.** BAM's identification puts
   the firmware's target-rate limit at 5.29 rad/s (≈50 rpm). A 1.4 Hz trot with a 60 mm step and
   25 mm swing needs ~4 rad/s at the knee in direct drive. Any knee reduction multiplies servo speed by
   the same ratio: 1.5:1 (the first belt guess) exceeded the cap; 1.25:1 sits at 94 %. Practical
   trot frequency with these servos is ~1.2–1.5 Hz, i.e. 0.15–0.2 m/s. This is the "confident walker,
   not sprinting cheetah" trade-off from the decision log, now with a number on it.
2. **Direct drive does not need a reduction.** Knee torque at the trot peak (two legs, 1.5× dynamic
   factor, foot swept over the step) is ~0.77 N·m against a 1.91 N·m stall — 40 %, inside the 60 %
   peak allowance, and 12 % when standing on four legs. The belt's torque margin is a nice-to-have
   bought with speed.
3. **The Raspberry Pi 5 mounts transversely.** The shell is ~93 mm wide inside (set by the abad
   servos), so the Pi's 85 mm side fits across the body and it occupies only 56 mm of length. That
   freed ~40 mm and is why the body can stay near Mini Cheetah's length ratio (2.0× thigh vs 1.82).
4. **Abad torque is the second-largest load** (~0.53 N·m at trot peak) because the foot stands
   ~40 mm outboard of the abad axis — the price of a 36 mm-long servo in the hip cluster. Keep the
   abad link as short as the hip servo allows; the slider exposes it.
5. **Coaxial hip clusters cost +70 mm of width.** Two 36 mm servos side by side on the hip axis with
   the thigh between them make B and C nearly as wide as they are long (253 vs 260 mm); A's hip servo
   tucks inboard of the thigh plane.
6. **Hip servo clearance**: with the hip-pitch servo case pointing up from the hip axis, the hip axis
   needs to sit ≥ 16 mm beyond the shell end wall (`hip_x_offset`), or the case collides with the
   body. Baseline set to 16 mm.

## Recommendation

**A · Direct drive.** It is the lowest-risk path to a working RL robot and the best fit for the
"reuse the Hugging Face stack" decision: Open Duck Mini's joints are directly servo-driven, so its BAM
actuator identification, backlash modeling, and sim-to-real pipeline apply without adding a
transmission model. Torque margin is adequate at size M and speed margin is the best of the three.
Its costs — a heavier lower thigh and higher leg inertia — are real but small at this scale (swing
accelerations need ~0.04 N·m at the hip).

**B** is the choice if the owner values the Mini Cheetah look and low leg inertia and accepts the
width, the belt hardware, and belt stretch in the sim. A 1:1 belt (20T→20T) keeps A's torque/speed
numbers while still moving the servo mass to the hip — the ratio slider on the review page shows
that trade directly. **C** is least favored: the linkage bounds the knee range (~100°) and its
variable ratio is the hardest of the three to mirror in simulation.

## How to review and record a decision

The review page draws all three candidates at one scale (1.5 px per mm in every view) with the
servos at their measured size, the electronics volumes in the body, and the legs animated through
stand/crouch, walk, trot, pace, bound (range-of-motion demo), and lateral sway. Sliders change the
proportions and gait; the sizing panels and the "Current proposal" summary update live. **Save
decision** stores the candidate, slider values, tags, and comments to the page's data store
(`feedback/current`, plus a `reviews` log); Claude reads it back with the artifact tools. If the
store is unavailable, **Copy summary** and paste it into chat.

## After selection

1. Add a `LOCKED` preset to `cheetah_pup/design.py` from the saved decision (`docs/design/locked.json`).
2. Generate the MJCF from it: primitive collision geometry, component masses, STS3215 actuator model
   from BAM's parameters (position actuator with the identified kp/velocity cap, friction, armature).
3. Validate standing and IK-driven gait playback in MuJoCo; then build the RL environment.
