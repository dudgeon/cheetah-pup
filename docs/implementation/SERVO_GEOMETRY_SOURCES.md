# XL330 physical geometry and assembly frame

Reviewed 2026-09-04. The earlier motor boxes were centered and oriented as generic joint markers. That is inadequate for assembly checks. The output shaft is offset within the housing, the depth includes a protruding horn, and each casing belongs to the body on the fixed side of its joint.

The machine-readable facts and source hashes are in [dimensions.json](../../reference/xl330/dimensions.json). Original manufacturer CAD files are linked rather than redistributed. These are manufacturer **reference** data, not a manufacturing release or measured properties of the future assembled robot.

## Manufacturer coordinate frame

Use a right-handed frame at the **outer mounting face center of the stock output horn**:

| Axis | Direction when viewing the horn with the long casing extending downward |
|---|---|
| X | Right across the 20 mm casing width |
| Y | Up toward the short end above the shaft |
| Z | Outward along the powered output axis, toward the viewer |

This matches the frame pictured on the [manufacturer inertia sheet](https://www.robotis.com/service/download.php?no=2136). It is distinct from the robot frame (forward, left, up). The downloaded STEP uses the same axis directions but places the horn face at Z = +6.5 mm; subtract 6.5 mm from STEP Z before using its coordinates here.

## Dimensions checked against the drawing and STEP

| Feature | Value in the manufacturer frame |
|---|---|
| Main casing | 20 × 34 × **23** mm (X, Y, Z) |
| Casing center relative to horn face | (0, −7.5, −14.5) mm |
| Casing bounds | X ±10; Y −24.5 to +9.5; Z −26 to −3 mm |
| Stock horn exposed disk | Ø16 × 3 mm; center (0, 0, −1.5) mm |
| Overall stock depth | 26 mm, including the horn |
| Shaft center from casing top | 9.5 mm |
| Front/rear frame hole spacing | 16 × 30 mm |
| Frame hole centers in XY | (±8, +7.5) and (±8, −22.5) mm |
| Stock horn attachment holes | Four Ø1.6 mm holes on Ø12 mm pitch circle, 3 mm maximum depth, M2 tapping screws |
| Optional rear idler | Adds 3 mm behind the case, giving 29 mm overall depth |

Sources: [X330 drawing, 2020-05-28](https://www.robotis.com/service/download.php?no=1986) and [manufacturer STEP assembly](https://www.robotis.com/service/download.php?no=1987). The separate rear idler provides passive support; it is not another powered output. Screw engagement, tolerances and access must follow the drawing sections in the later manufacturing CAD.

The 26 mm advertised dimension must **not** become a 26 mm solid casing centered on the shaft. That would place material on the output side, lose the asymmetric casing offset, and misplace its mass.

## Connector access

The official STEP contains two named `B3B-EH` header components. After applying the 6.5 mm origin correction, OpenCascade gives these bare-header bounding boxes:

| Header | X range | Y range | Z range |
|---|---|---|---|
| Positive side | +6.1 to +9.9 mm | −14 to −4 mm | −22.9 to −13.7 mm |
| Negative side | −9.9 to −6.1 mm | −14 to −4 mm | −22.9 to −13.7 mm |

They open through the opposing **width sidewalls**, not through the rear face. The manual identifies JST EHR-03 cable housings and B3B-EH-A PCB headers, with GND/VDD/DATA pin order. [Connector information](https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/#connector-information).

The JSON reserves proposed external spaces X = +10…+20 or −20…−10 mm, Y = −15…−3 mm and Z = −25…−12 mm. These are project allowances. The STEP does not include plugged cables, and these volumes do not establish the required bend radius or unplugging space. A side cradle must leave a window for the socket and plug, or use attachment regions above/below the socket. Printed plates cannot simply cover the entire sidewall.

## Mass properties

The [manufacturer reference sheet released February 2023](https://www.robotis.com/service/download.php?no=2136) gives the XL330-M288 mass as 18 g and its COM as (−0.29100243, −9.2403727, −13.121748) mm in the pictured frame. The full symmetric tensor is transcribed in the JSON, with g·mm² converted to kg·m² using a factor of 10⁻⁹. These values distinguish the M288 from the M077 and XC330 variants.

The PDF does not explicitly name the tensor's reference point. We interpret it as **about the COM**, expressed in the pictured axis directions. This is a physical-consistency inference: treating it as a tensor about the horn origin and shifting it to the COM would give negative Ixx, because the required parallel-axis term is larger than the reported Ixx. That interpretation is impossible for a physical mass distribution.

For placement rotation R and horn-face position p, transform the COM as `p + R @ com`. Transform the tensor as `R @ I_com @ R.T`. Apply the parallel-axis theorem only when aggregating that servo with its attached rigid-body components. Do not use the casing center as the servo COM, or add the full motor mass a second time for the visual horn. The combined reference tensor treats the servo as one component; the small rotating horn's separate inertia is not supplied.

## Proposed serial hip arrangement

The refined assembly uses roll shafts pointing outward along the robot's fore/aft axis, with their casings inside the torso. Each roll carrier holds the pitch motor with its output pointing laterally outward. The current 24 mm fore/aft offset of the pitch shaft puts its nearest casing sidewall 14 mm beyond the roll horn face. This separates the perpendicular casings and leaves room for the connector allowance without increasing the lateral hip lever arm merely to make boxes fit.

These proper rotations preserve the source-frame handedness. Here `front` is +1 for front legs and −1 for rear legs; `side` is +1 for left and −1 for right:

| Motor casing | Source X maps to | Source Y maps to | Source Z / physical output maps to |
|---|---|---|---|
| Roll | front × robot Y | robot Z | front × robot X |
| Pitch | −side × robot X | robot Z | side × robot Y |
| Knee, casing tail extending below knee | −side × upper-link X | upper-link Z | side × upper-link Y |

The roll casing belongs to the torso, the pitch casing to the roll carrier, and the knee casing to the upper leg. The knee casing tail extends below its shaft in upper-link coordinates; it stays fixed to the upper leg as the shank rotates on the outward side. This orientation replaces the earlier tail-toward-hip study, which brought the knee assembly too close to the front shoulder during the reviewed motion. The horn and attached downstream link rotate relative to each casing. Physical shaft direction and the software joint's positive axis need an explicit sign mapping on mirrored legs. A correct rotation axis does not by itself establish an assembled motor's encoder zero.

The 24 mm offset and knee installation direction are our design choices, not ROBOTIS dimensions. Bracket strength, final socket openings, fastener engagement, rear support and cable routing remain to be resolved in manufacturing CAD. Conservative envelopes can check gross interference now, but they cannot prove all of those details.

## Download provenance and reuse

The official download pages redirect to ROBOTIS-hosted public Dropbox links. The downloaded drawing, STEP, inertia sheet and optional accessory drawings were inspected locally; their exact resolved links, SHA-256 hashes and byte lengths are recorded in the JSON. The STEP header identifies a Creo reference dummy assembly dated 2020-07-27. Its external shape includes the optional rear idler and omits the motor's internal mass distribution, so its volume properties must not replace the manufacturer mass sheet.

No explicit open redistribution license for these downloaded CAD assets was established. This repository records factual dimensions and our original assembly model. The [HNX330-N101 aluminum horn](https://www.robotis.com/service/download.php?no=2163) and [FPX330-H101 support frame](https://www.robotis.com/service/download.php?no=2016) are useful later references; neither accessory is assumed installed in the current stock-servo envelope.
