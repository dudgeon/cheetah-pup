"""Build the design-review artifact by injecting this package's data into the HTML template.

    python -m cheetah_pup.build_artifact

Reads docs/design/review/template.html, replaces the __DATA_JSON__ marker with the presets,
constants, and reference metrics computed here, and writes docs/design/review/index.html.
The page's JavaScript re-implements the sizing math for live sliders; the embedded reference
metrics let it cross-check itself against this package at load time.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import subprocess
from dataclasses import asdict

from .analysis import metrics
from .design import PRESETS, SIZES, MINI_CHEETAH, preset
from .electronics import PI5, BATTERY_2S, PCB, IMU
from .servo import STS3215, CONTINUOUS_FRACTION, PEAK_FRACTION

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "docs" / "design" / "review" / "template.html"
OUTPUT = ROOT / "docs" / "design" / "review" / "index.html"


def reference(key: str) -> dict:
    m = metrics(preset(key, "M"))
    return {
        "mass": m["mass"],
        "width": m["overall_width"],
        "knee_servo_stand": m["torque"]["stand"]["knee_servo"],
        "knee_servo_trot": m["torque"]["trot_peak"]["knee_servo"],
        "abad_trot": m["torque"]["trot_peak"]["abad"],
        "hip_trot": m["torque"]["trot_peak"]["hip"],
        "speed_knee_servo": m["speed"]["knee_servo"],
        "speed_hip": m["speed"]["hip"],
        "top_needed": m["packaging"]["top_needed"],
    }


def build_data() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        commit = "uncommitted"
    return {
        "meta": {"generated": dt.date.today().isoformat(), "commit": commit},
        "servo": asdict(STS3215),
        "allowances": {"continuous": CONTINUOUS_FRACTION, "peak": PEAK_FRACTION},
        "electronics": {
            "pi": {"name": PI5.name, "size": PI5.size, "mass": PI5.mass},
            "battery": {"name": BATTERY_2S.name, "size": BATTERY_2S.size, "mass": BATTERY_2S.mass},
            "pcb": {"name": PCB.name, "size": PCB.size, "mass": PCB.mass},
            "imu": {"name": IMU.name, "size": IMU.size, "mass": IMU.mass},
        },
        "mini_cheetah": MINI_CHEETAH,
        "sizes": SIZES,
        "presets": {k: PRESETS[k].to_dict() for k in PRESETS},
        "reference": {k: reference(k) for k in PRESETS},
    }


def main():
    html = TEMPLATE.read_text()
    data = json.dumps(build_data(), separators=(",", ":"))
    assert "__DATA_JSON__" in html, "template is missing the __DATA_JSON__ marker"
    OUTPUT.write_text(html.replace("__DATA_JSON__", data))
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({OUTPUT.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
