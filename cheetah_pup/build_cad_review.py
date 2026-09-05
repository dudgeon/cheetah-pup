"""Build the DR-04 CAD review page from the CAD exports and the validation recordings.

    python -m cheetah_pup.build_cad_review

Embeds cad/exports/viewer_meshes.json (unique part meshes, their instances per body, and the
kinematic tree), the per-part and per-body mass properties, two seconds of the open-loop walk and
trot recordings for the viewer to play, and the print/assembly notes below, into
docs/design/cad/template.html → docs/design/cad/index.html.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import pathlib
import subprocess

from .design import locked
from .servo import STS3215

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXPORTS = ROOT / "cad" / "exports"
TEMPLATE = ROOT / "docs" / "design" / "cad" / "template.html"
OUTPUT = ROOT / "docs" / "design" / "cad" / "index.html"
VALIDATION = ROOT / "sim" / "validation" / "openloop"

# Printed parts as the printer sees them: one row per unique part, with the orientation and notes.
PART_ROWS = [
    {"key": "trunk_tub", "label": "Trunk tub", "qty": 1, "parts": ["trunk_tub"],
     "orientation": "Floor on the bed, open side up",
     "notes": "Servo shelves and rib windows bridge ≤ 25 mm — no supports. Battery rails, Pi standoffs and lid bosses print vertically. Ø21 abad bores and M2 holes are in the end walls (horizontal holes, small enough to print clean)."},
    {"key": "trunk_lid", "label": "Trunk lid", "qty": 1, "parts": ["trunk_lid"],
     "orientation": "Top face on the bed",
     "notes": "PCB bosses then stand up from the underside; vent slots are through-slots. Four M2 screws into the tub bosses."},
    {"key": "bracket", "label": "Abad bracket", "qty": 4, "parts": ["LF_bracket", "RF_bracket", "LH_bracket", "RH_bracket"],
     "orientation": "Servo-plate face on the bed",
     "notes": "Four distinct mirrors (front/hind × left/right). The Ø21 hip-servo bore and four M2 holes come out vertical; the back plate, bar and gusset stand without supports. Bolts to the abad horn with four screws through the r = 7 pattern."},
    {"key": "thigh", "label": "Thigh", "qty": 4, "parts": ["LF_thigh", "RF_thigh", "LH_thigh", "RH_thigh"],
     "orientation": "Plate outboard face on the bed",
     "notes": "Left and right mirrors, two of each. Knee bore vertical; cradle walls and end stops rise from the plate. Hip horn pattern in the r = 13 pad; the knee servo drops in gear-end-down and is retained by its four M2 top-face screws through the plate."},
    {"key": "shank", "label": "Shank", "qty": 4, "parts": ["LF_shank", "RF_shank", "LH_shank", "RH_shank"],
     "orientation": "Lying on a side face",
     "notes": "Left and right mirrors. The beam is a 14 mm-wide profile in the leg plane: printing it on its side puts the layers in the bending plane. Knee pad bolts to the horn disc; the jog brings the foot back under the thigh plane."},
    {"key": "foot", "label": "Foot (TPU)", "qty": 4, "parts": ["LF_foot", "RF_foot", "LH_foot", "RH_foot"],
     "orientation": "Pole on the bed",
     "notes": "Ø20 sphere placeholder at the sim's contact point. The shank-tip socket (Ø12.4, 8 mm deep) and a flat pole are the next detail to add; print in 95A TPU at 100 %."},
]

VERIFY = [
    "STS3215 top-face M2 holes at (8.3, ±10.25) and (29.0, ±10.25) mm from the horn axis: positions and thread were measured from Open Duck Mini v2's case mesh, not a datasheet — confirm on a real servo before printing the brackets and thighs.",
    "Horn: four Ø2.5 holes on r = 7 plus a Ø3.2 centre. Confirm the screw size (M2 vs M2.5) and that the Ø20 disc stands 2.95 mm proud of the case step.",
    "The horn disc rides in a Ø21 printed bore as a plain bearing (0.5 mm radial clearance in the model). Tune the bore to the printer — 0.15–0.25 mm radial is the target for a snug, low-friction fit.",
    "The Ø20 × 2.1 mm idler disc on the case bottom is assumed by every servo pocket; check it is present on the current STS3215 revision.",
    "Every joint is single-sided (horn side only). Watch horn bearing play under the 1.4 kg robot; the thigh cradle leaves room for an idler-side support (fork) if needed.",
    "Hip pitch is limited to roughly ±60° before the knee-servo case meets the trunk corner — sweep it in Pose mode. A deeper crouch needs a notch in the tub corner or a smaller hip x-offset.",
    "Battery pack envelope 70 × 40 × 22 mm (2S 18650 + BMS) sits between rails 44 mm apart: confirm the actual pack before printing the tub.",
    "Pi 5 with the official active cooler is 25 mm tall on 3 mm standoffs above the battery, 3 mm under the lid: a low-profile cooler would ease this.",
    "PCB + IMU envelope 40 × 60 × 8 mm hung from the lid over the front abad servos (bosses at x = 49 ± 15, y = ±25): revisit when the board is laid out in Phase 3.",
    "Wiring: abad leads pass through the inboard rib windows; hip and knee leads run along the bracket bar and thigh plate. No clips or strain reliefs are modelled yet.",
]


def principal_moments(full: list[float]) -> list[float]:
    """Eigenvalues (ascending) of the symmetric inertia tensor given as Ixx Iyy Izz Ixy Ixz Iyz."""
    a, b, c, d, e, f = full
    p1 = d * d + e * e + f * f
    if p1 < 1e-40:
        return sorted([a, b, c])
    q = (a + b + c) / 3
    p = math.sqrt(((a - q) ** 2 + (b - q) ** 2 + (c - q) ** 2 + 2 * p1) / 6)
    B = [[(a - q) / p, d / p, e / p], [d / p, (b - q) / p, f / p], [e / p, f / p, (c - q) / p]]
    det = (B[0][0] * (B[1][1] * B[2][2] - B[1][2] * B[2][1]) - B[0][1] * (B[1][0] * B[2][2] - B[1][2] * B[2][0])
           + B[0][2] * (B[1][0] * B[2][1] - B[1][1] * B[2][0]))
    phi = math.acos(max(-1.0, min(1.0, det / 2))) / 3
    e1 = q + 2 * p * math.cos(phi)
    e3 = q + 2 * p * math.cos(phi + 2 * math.pi / 3)
    return sorted([e1, 3 * q - e1 - e3, e3])


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "uncommitted"


def gait_segment(gait: str, t0: float = 3.0, t1: float = 5.0) -> dict | None:
    path = VALIDATION / f"{gait}.json"
    if not path.exists():
        return None
    run = json.loads(path.read_text())
    rows = [r for r in run["rows"] if t0 <= r["t"] < t1]
    if len(rows) < 2:
        return None
    x0, y0 = rows[0]["pos"][0], rows[0]["pos"][1]
    return {
        "dt": round(rows[1]["t"] - rows[0]["t"], 4),
        "pos": [[round(r["pos"][0] - x0, 5), round(r["pos"][1] - y0, 5), round(r["pos"][2], 5)] for r in rows],
        "rpy": [[round(v, 5) for v in r["rpy"]] for r in rows],
        "q": [[round(v, 4) for v in r["q"]] for r in rows],
        "stats": run["stats"],
    }


def build_data() -> dict:
    viewer = json.loads((EXPORTS / "viewer_meshes.json").read_text())
    mass = json.loads((EXPORTS / "mass_properties.json").read_text())
    p = locked()
    if viewer["design"] != p.name or mass["design"] != p.name:
        raise ValueError("CAD exports are stale for the locked design: run `python -m cad.assembly`")
    rows = []
    for row in PART_ROWS:
        each = mass["parts"][row["parts"][0]]
        rows.append({**row, "cls": each["class"], "volume_cm3": each["volume_cm3"], "mass_g": each["mass_g"],
                     "infill": mass["infill"][each["class"]]})
    gaits = {g: gait_segment(g) for g in ("walk", "trot")}
    bodies = {name: {**b, "principal": principal_moments(b["fullinertia"])} for name, b in mass["bodies"].items()}
    return {
        "meta": {"generated": dt.date.today().isoformat(), "commit": _git_commit(), "design": p.name, "cad_generated": mass["generated"]},
        "params": p.to_dict(),
        "servo": {"length": STS3215.length, "width": STS3215.width, "height": STS3215.height, "mass": STS3215.mass,
                  "stall": STS3215.stall_torque},
        "viewer": viewer,
        "mass": {"total": mass["total_mass"], "printed": mass["printed_mass"], "bodies": bodies,
                 "infill": mass["infill"], "densities": mass["densities"], "servos": mass["servos"]},
        "parts": rows,
        "gaits": {g: s for g, s in gaits.items() if s},
        "verify": VERIFY,
    }


def main():
    data = json.dumps(build_data(), separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text()
    assert "__DATA_JSON__" in html
    OUTPUT.write_text(html.replace("__DATA_JSON__", data))
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({OUTPUT.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
