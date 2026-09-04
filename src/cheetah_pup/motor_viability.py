"""Mass sensitivity for the selected crawl, keeping all twelve motors unchanged.

This is a static engineering screen. Uniformly reducing nonmotor allowances does
not create a manufacturable lighter assembly or establish dynamic/thermal limits.
"""

from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from .analysis import joint_addresses
from .gait_demo import trajectory
from .gait_load import minimum_peak_static_allocation
from .kinematics import LEG_ORDER
from .model import load_config, total_mass


def mass_row(config, nonmotor_scale, frames_per_step=48):
    candidate = copy.deepcopy(config)
    for name in candidate["mass_kg"]:
        if name != "servo":
            candidate["mass_kg"][name] *= nonmotor_scale
    model, frames, _ = trajectory(candidate, frames_per_step=frames_per_step)
    data = mujoco.MjData(model)
    dofs = np.concatenate([joint_addresses(model, leg)[1] for leg in LEG_ORDER])
    torques = []
    for frame in frames:
        data.qpos[:] = frame["qpos"]
        mujoco.mj_forward(model, data)
        result = minimum_peak_static_allocation(model, data, frame["active"])
        if result is None:
            raise ValueError("Unsupported prescribed pose")
        _, required = result
        torques.append(required[dofs])
    torques = np.asarray(torques)
    peak = float(np.max(np.abs(torques)))
    return {
        "nonmotor_mass_scale": nonmotor_scale,
        "mass_kg": total_mass(candidate),
        "motor_mass_kg": 12 * candidate["mass_kg"]["servo"],
        "static_peak_nm": peak,
        "worst_joint_static_rms_nm": float(
            np.max(np.sqrt(np.mean(torques**2, axis=0)))
        ),
        "margin_to_0_10_nm_estimate": 0.10 / peak,
        "continuous_estimate_needed_for_1_5_margin_nm": 1.5 * peak,
        "passes_proposed_static_screen": peak * 1.5 <= 0.10,
    }


def report(config):
    rows = [mass_row(config, s) for s in (0.5, 0.6, 0.7, 0.8, 1.0, 1.2)]
    lo, hi = 0.5, 1.0
    # Interpolate boundary by bisection only if the explicit bracket straddles it.
    boundary = None
    if (
        rows[0]["passes_proposed_static_screen"]
        and not rows[4]["passes_proposed_static_screen"]
    ):
        for _ in range(7):
            mid = (lo + hi) / 2
            r = mass_row(config, mid, frames_per_step=24)
            if r["passes_proposed_static_screen"]:
                lo = mid
            else:
                hi = mid
        # Verify the conservative low side at the same sampling as the main rows.
        candidate = mass_row(config, lo)
        if candidate["passes_proposed_static_screen"]:
            boundary = candidate
    return {
        "kind": "Fixed-geometry mass sensitivity; static hypotheses, not hardware selection",
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
        "method": "Selected 140 mm / 25% shift / 20 mm stride / 12 mm lift crawl, 192 poses; exact static minimax foot-load allocation with link gravity. Recompute COM shifts for every mass allowance.",
        "rows": rows,
        "approximate_passing_mass_boundary": boundary,
        "limitations": [
            "Motor masses, casing geometry and source tensors never shrink. Only other provisional mass allowances change.",
            "Uniform allowance scaling is not a new BOM, structural analysis or verified battery/compute package.",
            "The 1.5× static reserve is a project screening choice against a manufacturer estimate, not a proven thermal boundary.",
            "No horizontal gait forces, contact transients, loaded speed envelope, actuator tracking, mass-dependent cable clearance or carpet/threshold performance is established.",
        ],
    }


def write_report(config, path):
    r = report(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(r, indent=2) + "\n")
    lines = [
        "# Motor viability and mass sensitivity",
        "",
        "**Judgment: the XL330 is plausible for low-speed experiments, but marginal in the present 613 g study. It is not ready for a hardware commitment aimed at carpet, thresholds and 10–15 minutes of walking.**",
        "",
        "The current dynamic model can stand; that result and the failed preset crawl do not establish a fundamental motor yes/no. A poor open-loop gait can fail with adequate motors. Equally, RL cannot provide sustained torque, speed or power the hardware lacks.",
        "",
        "## Fixed geometry, unchanged motors",
        "",
        "| Total allowance | Nonmotor mass scale | Peak static torque | Margin to 0.10 N·m estimate | 1.5× screen |",
        "|---|---:|---:|---:|---|",
    ]
    for row in r["rows"]:
        lines.append(
            f"| {row['mass_kg'] * 1000:.0f} g | {row['nonmotor_mass_scale']:.0%} | {row['static_peak_nm']:.4f} N·m | {row['margin_to_0_10_nm_estimate']:.2f}× | {'Pass' if row['passes_proposed_static_screen'] else 'Fail'} |"
        )
    if r["approximate_passing_mass_boundary"]:
        b = r["approximate_passing_mass_boundary"]
        lines += [
            "",
            f"For this particular geometry and sampled crawl, a mass allowance near **{b['mass_kg'] * 1000:.0f} g** reaches the proposed static reserve. This is a sizing result, not a feasible lighter BOM; all 12 motors still contribute 216 g.",
        ]
    lines += [
        "",
        "At 613 g, the present 0.0934 N·m peak would need about 0.140 N·m of continuous capability to satisfy the chosen reserve without changing motion/mass. Replacing the servo with a heavier one changes that demand; 0.140 N·m is not a sufficient specification for an arbitrary replacement.",
        "",
        "The nominal 0.10 N·m figure is [ROBOTIS’s 20%-of-stall continuous estimate](https://www.robotis.us/dynamixel-xl330-m288-t/), not a measured guarantee for this mounting or duty cycle. The [manual](https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/) separately publishes 5 V stall torque and unloaded speed; stall and unloaded speed are distinct endpoints, not simultaneously available walking output.",
        "",
        "## Minimum refinement before cloud work",
        "",
        "1. Refine the mass budget and mechanically valid workspace. Compare a lighter assembly against one stronger model-supported actuator before committing to hardware. Exact screws and cosmetic CAD can wait; cable/board/battery volumes, moving mass and reliable contacts cannot.",
        "2. Extend the verified standing interface to a useful movement task: 45 proprioceptive observations, 12 bounded targets, 50 Hz commands, BAM physics/delay, collision-free resets, fault/fall detection and useful rewards. Compare learned behavior with the existing fixed-target baseline.",
        "3. Bring the CPU and GPU actuator implementations into agreement and make voltage/gain/delay/friction uncertainty explicit. Small exploratory RL can proceed while stock-servo provenance is open; expensive optimization and hardware transfer should not treat it as resolved.",
        "4. The short CPU learning/checkpoint/evaluation path is now proven: [16,384 PPO transitions](cpu-rl-smoke.md), with no improvement over fixed standing targets. Use CPU for the next task-debugging stage before paying for cloud. This experiment does not validate recovery, walking or terrain transfer.",
        "",
        "No additional servo characterization is assigned to the owner. Further model accuracy must come from published evidence, defensible uncertainty bounds, and later normal whole-robot validation.",
        "",
        "## Limits",
        "",
    ] + ["- " + s for s in r["limitations"]]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return r


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("config/robot.json"))
    p.add_argument("--output", type=Path, default=Path("reports/motor-viability.json"))
    a = p.parse_args()
    r = write_report(load_config(a.config), a.output)
    print(
        json.dumps(
            {"rows": r["rows"], "boundary": r["approximate_passing_mass_boundary"]},
            indent=2,
        )
    )
