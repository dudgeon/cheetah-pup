# Assembly refinement and clearance audit

The motor shafts, casing offsets and reference mass properties now follow the ROBOTIS XL330 drawing. These are original structural envelopes, not a released printable assembly.

**Geometry change:** pitch shaft24 mm fore/aft outward from the roll shaft, with25 mm lateral offset. Roll motor faces fore/aft; pitch and knee motors face outward left/right. Knee casing tails point toward the foot, providing shoulder clearance during flexion. Stock rear idlers are not installed in this study.

| Check | Poses | Solid interference pairs | Cable allowance interference pairs | Minimum casing separation bound |
|---|---:|---:|---:|---:|
| neutral | 1 | 0 | 0 | 17.00mm |
| prescribed crawl | 192 | 0 | 0 | 16.45mm |
| rejected original crawl | 96 | 6 | 2 | 4.98 mm |
| sampled joint box | 96 | 54 | 26 | -14.34mm |

All12 shafts aligned with their modeled joint axes: **True**. Both physical shaft direction and mathematical joint sign are recorded in JSON; rear roll/right pitch motor signs differ from front/left motors.

The current illustration uses a140 mm body height and25% of the original centroid shift. The rejected comparison uses124 mm and the full centroid shift. Positive support loads are checked for both; only the revised sampled trajectory clears the envelope checks.

The broad random joint-box sample is diagnostic. Any collision there means scalar joint bounds alone do not define a mechanically valid workspace; use collision-aware resets and policy penalties/termination, and refine travel limits after final CAD.

## Remaining assembly gates

- Casing boxes omit rounded corners and small recesses; may conservatively flag actual-clear areas.
- Cable volumes are design allowances around STEP-verified bare socket locations, not mated-plug or flexible-wire CAD.
- Printed cradles have socket slots, but fastener engagement, service access, support bearings and stiffness are not validated.
- Random joint-box sampling is not a proof that every pose within rectangular joint limits is collision-free.
- Reference servo mass is lumped on housing; horn/rotor mass motion is not split from it.

Source dimensions and tensor provenance: [SERVO_GEOMETRY_SOURCES.md](../docs/implementation/SERVO_GEOMETRY_SOURCES.md). Full pairs, distances and pose indices are in the adjacent JSON.
