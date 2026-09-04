"""Build the DR-02 sim-playback artifact from the validation recordings.

    python -m cheetah_pup.build_replay [sim/validation]

Reads sim/validation/summary.json and the per-run recordings, embeds a compact copy (time, trunk
pose, joint angles, commanded angles, actuator torques, foot contacts) into
docs/design/replay/template.html, and writes docs/design/replay/index.html.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import subprocess
import sys
from dataclasses import asdict

from .design import locked
from .gait import GAITS
from .mjcf import servo_gains
from .servo import STS3215, CONTINUOUS_FRACTION, PEAK_FRACTION

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "docs" / "design" / "replay" / "template.html"
OUTPUT = ROOT / "docs" / "design" / "replay" / "index.html"


def compact(run: dict) -> dict:
    rows = run["rows"]
    return {
        "stats": run["stats"],
        "t": [r["t"] for r in rows],
        "pos": [r["pos"] for r in rows],
        "rpy": [r["rpy"] for r in rows],
        "q": [r["q"] for r in rows],
        "ctrl": [r["ctrl"] for r in rows],
        "tau": [r["tau"] for r in rows],
        "contact": [r["contact"] for r in rows],
    }


def build_data(val_dir: pathlib.Path) -> dict:
    summary = json.loads((val_dir / "summary.json").read_text())
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        commit = "uncommitted"
    runs = {}
    for mode, gaits in summary["runs"].items():
        runs[mode] = {}
        for gait in gaits:
            runs[mode][gait] = compact(json.loads((val_dir / mode / f"{gait}.json").read_text()))
    p = locked()
    return {
        "meta": {"generated": dt.date.today().isoformat(), "commit": commit, "model": summary["model"]},
        "params": p.to_dict(),
        "servo": {"stall": STS3215.stall_torque, "cap": STS3215.max_velocity, "continuous": CONTINUOUS_FRACTION,
                  "peak": PEAK_FRACTION, "gains": servo_gains("datasheet"), "length": STS3215.length,
                  "width": STS3215.width, "height": STS3215.height, "shaft_from_end": STS3215.shaft_from_end},
        "gaits": {k: {"duty": v["duty"], "phase": v["phase"], "step": v["step"], "freq": v["freq"], "label": v["label"]} for k, v in GAITS.items()},
        "stand": summary["stand"],
        "mass": summary["total_mass_kg"],
        "control_hz": summary["control_hz"],
        "settle": 1.0,
        "runs": runs,
    }


def main(argv=None):
    argv = argv or sys.argv[1:]
    val_dir = pathlib.Path(argv[0]) if argv else ROOT / "sim" / "validation"
    html = TEMPLATE.read_text()
    assert "__DATA_JSON__" in html
    data = json.dumps(build_data(val_dir), separators=(",", ":"))
    OUTPUT.write_text(html.replace("__DATA_JSON__", data))
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({OUTPUT.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
