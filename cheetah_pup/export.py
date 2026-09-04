"""Export the candidate presets, constants, and computed metrics as JSON for the design artifact.

    python -m cheetah_pup.export docs/design/candidates.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict

from .analysis import metrics, summary_line
from .design import PRESETS, SIZES, MINI_CHEETAH, preset
from .electronics import PI5, BATTERY_2S, PCB, IMU
from .servo import STS3215, CONTINUOUS_FRACTION, PEAK_FRACTION


def build() -> dict:
    out = {
        "servo": asdict(STS3215),
        "allowances": {"continuous": CONTINUOUS_FRACTION, "peak": PEAK_FRACTION},
        "electronics": {c.name: {"size": c.size, "mass": c.mass} for c in (PI5, BATTERY_2S, PCB, IMU)},
        "mini_cheetah": MINI_CHEETAH,
        "sizes": SIZES,
        "candidates": {},
    }
    for key in PRESETS:
        for size in SIZES:
            p = preset(key, size)
            out["candidates"][f"{key}-{size}"] = {"params": p.to_dict(), "metrics": metrics(p)}
    return out


def main(argv=None):
    argv = argv or sys.argv[1:]
    data = build()
    if argv:
        with open(argv[0], "w") as f:
            json.dump(data, f, indent=1)
        print(f"wrote {argv[0]}")
    for key in PRESETS:
        print(summary_line(preset(key, "M")))


if __name__ == "__main__":
    main()
