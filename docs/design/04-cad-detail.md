# DR-04 — CAD detail of the locked design (Phase 2, first pass)

**Review page**: DR-04 · CAD detail — https://claude.ai/code/artifact/6b9c8170-cae2-4a28-a494-7ba0548bcc4b (WebGL viewer with
gait playback and a joint sweep, part and body tables, hardware checklist). Rebuild with
`python -m cheetah_pup.build_cad_review` after `python -m cad.assembly`.

**Files**: `cad/` (build123d model), `cad/exports/step/*.step` (per part plus
`assembly_nominal_stance.step` and the servo), `cad/exports/stl/*.stl` (millimetres, each part in
its MuJoCo body frame), `cad/exports/mass_properties.json` (per-part and per-body mass, centre of
mass, inertia; servo and electronics placements), `cad/exports/viewer_meshes.json` (viewer data).

## What was built

A parametric build123d model of the locked design, `cheetah_pup.design.locked()`, with every
printed part modelled in the frame of the MuJoCo body that carries it, so the STL files drop into
the simulator without transforms and the per-body mass properties are the simulator's `<inertial>`
elements. The servo is modelled from Open Duck Mini v2's STS3215 case meshes (`cad/servo.py`).

| Part | Qty | Class · effective solid | Each | Purpose |
|---|---|---|---|---|
| Trunk tub | 1 | shell · 95 % | 169.0 g, 143.5 cm³ | Floor, walls to the split, abad servo cradles (windowed shelves, ribs), end-wall bores and M2 holes, battery rails, Pi standoffs, lid bosses |
| Trunk lid | 1 | shell · 95 % | 72.8 g | Split to the top, vent slots, lid screws, PCB bosses hanging from the underside |
| Abad bracket | 4 (each a distinct mirror) | plate · 85 % | 13.6 g | Bolts to the abad horn; carries the hip servo with its horn at the thigh plane (back plate, bar, servo plate with Ø21 bore, gusset) |
| Thigh | 4 (2 L, 2 R) | plate · 85 % | 24.5 g | 3.5 mm plate on the hip horn; knee servo cradle gear-end-down inboard of the plate; knee bore and mounting holes |
| Shank | 4 (2 L, 2 R) | beam · 75 % | 9.3 g | Pad on the knee horn disc; 14 mm-wide beam with a jog inboard so the foot lands 3.25 mm outboard of the knee plane |
| Foot | 4 | TPU · 100 % | 5.0 g | Ø20 sphere at the sim's contact point (placeholder shape) |

Printed total 451 g; robot total **1.391 kg** with 12 servos (660 g), battery (125 g), Pi 5
(65 g), PCB + IMU (30 g), and 60 g of wiring — 1.6 % under the parametric estimate the sizing
used (1.413 kg).

## Servo interface convention (from the measured case)

- Frame: origin on the shaft axis at the horn-seat plane; L along the case (+35.1 to −10.1 mm),
  W ±12.36, A along the shaft (+A toward the horn; case bottom at −32.6).
- Fixed part: a 3 mm plate on the top-face step (A ∈ [1.1, 4.1]), four M2 screws (Ø2.4 clearance)
  into the tapped holes at (L, W) = (8.3, ±10.25) and (29.0, ±10.25); a Ø21 bore in which the
  Ø20 horn disc (A ∈ [2.1, 5.05]) rides as a plain bearing, 0.95 mm proud of the plate.
- Moving part: bolts to the horn's four Ø2.5 holes on r = 7 plus the Ø3.2 centre, seating on the
  proud disc face.
- Pockets: 0.4 mm clearance around the case, including the Ø20 × 2.1 idler disc on the bottom.

## Packaging refinements made while modelling (kinematics unchanged)

| Parameter | Phase 1 | Phase 2 | Why |
|---|---|---|---|
| `abad_link` | 40 mm | **43 mm** | Room for the hip servo case (24.7 wide) plus the bracket's servo plate between the abad axis and the thigh plane |
| `abad_to_abad` | 70 mm | **74 mm** | Abad servo cases (35.7 tall) side by side in the tub with the centre rib |
| `hip_x_offset` | 15 mm | **18 mm** | End wall (3) + step (1.1) + horn disc + bracket back plate before the hip axis |
| `foot_y_offset` | — | **3.25 mm** | The shank sits on the knee horn disc outboard of the thigh plate; the beam jogs back inboard but the foot ends 3.25 mm outboard |
| Layout | Pi 5 and PCB on a top layer, Pi transverse | **Battery on the floor, Pi 5 above it in the centre bay; PCB + IMU hung from the lid over the front abad servos** | The centre bay between the servo cradles is the only 60 mm-tall volume; the PCB fits the 26 mm above the servo tops |

Shell width follows from the servo bay: `shell_width = abad_to_abad + 2 × (10.1 + 0.9 + wall)`
= 102 mm; shell length stays 144 mm.

## In the simulator

`cheetah_pup/mjcf.py` now defaults to the CAD model for the locked design: explicit `<inertial>`
per body from `mass_properties.json`, the STL meshes as visuals (`meshdir` relative to the XML),
servos placed by the recorded frames, feet at `y = ±foot_y_offset`. Collision stays primitive:
foot spheres in both variants; a shell box and leg capsules (hidden geom group 3) in the full
model only. `--no-cad` still generates the Phase 1 primitive model for any preset.

Re-validation on the CAD model (`sim/validation/`, DR-02 rebuilt): stands with 0.9 mm sag and
0.76° droop, knee hold 0.249 N·m; open-loop walk and trot both survive 6 s (walk 0.024 m/s,
trot 0.069 m/s, pitch ≤ 10°); the RL environment's four CPU tests pass on the new model.

## Known gaps and hardware checks

The review page lists what to confirm before printing. In short: the servo hole patterns and
horn fastener size are measured from a mesh, not a datasheet; the Ø21 bore clearance must be
tuned to the printer; joints are single-sided on the horn; hip pitch is limited to about ±60° by
the knee-servo case against the tub corner; battery, Pi cooler, and PCB envelopes are assumptions;
no wire clips, foot sockets, or fork supports are modelled yet. The tub is 169 g against the
143 g parametric shell estimate — acceptable, more windows possible.
