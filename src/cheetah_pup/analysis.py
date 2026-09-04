"""Independent geometry checks and explicitly limited static load screening.

No result from this module establishes learned locomotion, real motor behavior,
manufacturing clearance, battery endurance, or carpet traversal.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import platform
import copy
from pathlib import Path

import mujoco
import numpy as np

from .kinematics import JOINT_ORDER, LEG_ORDER, foot_position, leg_jacobian
from .model import build_mjcf, total_mass


def standing_model(config: dict, terrain: str = "flat"):
    model = mujoco.MjModel.from_xml_string(build_mjcf(config, terrain))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
    mujoco.mj_forward(model, data)
    return model, data


def joint_addresses(model, leg):
    ids = [model.joint(f"{leg}_{joint}").id for joint in JOINT_ORDER]
    return model.jnt_qposadr[ids], model.jnt_dofadr[ids]


def kinematic_audit(config: dict, samples: int = 32, seed: int = 731):
    model, data = standing_model(config)
    rng = np.random.default_rng(seed)
    errors = {"fk_max_error_m": 0.0, "jacobian_mujoco_max_error_m_per_rad": 0.0,
              "jacobian_finite_difference_max_error_m_per_rad": 0.0}
    for _ in range(samples):
        data.qpos[:3] = rng.uniform(-0.2, 0.2, 3)
        quat = rng.normal(size=4)
        data.qpos[3:7] = quat / np.linalg.norm(quat)
        for leg in LEG_ORDER:
            qadr, _ = joint_addresses(model, leg)
            data.qpos[qadr] = [rng.uniform(*config["joint_limits_rad"][j]) for j in JOINT_ORDER]
        mujoco.mj_forward(model, data)
        rotation = data.body("base").xmat.reshape(3, 3)
        origin = data.body("base").xpos
        for leg in LEG_ORDER:
            qadr, dadr = joint_addresses(model, leg)
            q = data.qpos[qadr].copy()
            predicted = origin + rotation @ foot_position(config, leg, q)
            site = model.site(f"{leg}_foot").id
            errors["fk_max_error_m"] = max(errors["fk_max_error_m"], float(np.max(np.abs(predicted - data.site_xpos[site]))))
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jp, jr, site)
            analytical = leg_jacobian(config, leg, q)
            errors["jacobian_mujoco_max_error_m_per_rad"] = max(errors["jacobian_mujoco_max_error_m_per_rad"], float(np.max(np.abs(rotation @ analytical - jp[:, dadr]))))
            step = 1e-6
            finite = np.column_stack([(foot_position(config, leg, q + np.eye(3)[j] * step) - foot_position(config, leg, q - np.eye(3)[j] * step)) / (2 * step) for j in range(3)])
            errors["jacobian_finite_difference_max_error_m_per_rad"] = max(errors["jacobian_finite_difference_max_error_m_per_rad"], float(np.max(np.abs(analytical - finite))))
    return {"samples": samples, "legs_per_sample": 4, "seed": seed, **errors,
            "passed": all(v < 1e-8 for v in errors.values())}


def solve_vertical_support(foot_xy, com_xy, weight_n):
    """Find nonnegative vertical loads satisfying force and roll/pitch balance.

    Enumerate active supports (at most four feet), select minimum squared load.
    Return None if COM cannot be supported by this footprint without acceleration.
    No friction, horizontal forces or dynamic stability is claimed.
    """
    foot_xy = np.asarray(foot_xy, dtype=float)
    com_xy = np.asarray(com_xy, dtype=float)
    offsets = foot_xy - com_xy
    matrix = np.vstack((np.ones(len(foot_xy)), offsets.T))
    target = np.array([weight_n, 0.0, 0.0])
    candidates = []
    for count in range(1, len(foot_xy) + 1):
        for subset in itertools.combinations(range(len(foot_xy)), count):
            forces, *_ = np.linalg.lstsq(matrix[:, subset], target, rcond=None)
            residual = matrix[:, subset] @ forces - target
            if np.min(forces) >= -1e-8 and np.max(np.abs(residual)) < 1e-8:
                full = np.zeros(len(foot_xy))
                full[list(subset)] = np.maximum(forces, 0)
                candidates.append(full)
    return min(candidates, key=lambda f: float(f @ f)) if candidates else None


def static_support_screen(config: dict, model=None, data=None):
    """Joint gravity minus J.T*ground force at the neutral pose.

    This includes link gravity from MuJoCo qfrc_bias. Equal mg/n sharing is not
    assumed: load distribution must satisfy whole-robot force/moment balance.
    """
    if model is None:
        model, data = standing_model(config)
    mass = float(np.sum(model.body_mass))
    weight = mass * config["simulation"]["gravity_m_s2"]
    com = data.subtree_com[model.body("base").id]
    foot = {leg: data.site(f"{leg}_foot").xpos.copy() for leg in LEG_ORDER}
    cases = [("four_feet", LEG_ORDER)]
    cases += [(f"three_feet_lift_{leg}", tuple(l for l in LEG_ORDER if l != leg)) for leg in LEG_ORDER]
    cases += [("diagonal_FL_RR", ("FL", "RR")), ("diagonal_FR_RL", ("FR", "RL"))]
    limit = config["actuator"]["torque_limit_nm"]
    results = []
    for name, supports in cases:
        forces = solve_vertical_support([foot[l][:2] for l in supports], com[:2], weight)
        base = {"case": name, "support_feet": list(supports), "vertical_static_equilibrium": forces is not None}
        if forces is None:
            results.append({**base, "reason": "The current COM projection is outside this support polygon/line; body shift or dynamic motion is required."})
            continue
        generalized = data.qfrc_bias.copy()
        for leg, force in zip(supports, forces):
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jp, jr, model.site(f"{leg}_foot").id)
            generalized -= jp.T @ np.array([0, 0, force])
        torques = {f"{leg}_{joint}": float(generalized[model.jnt_dofadr[model.joint(f"{leg}_{joint}").id]]) for leg in LEG_ORDER for joint in JOINT_ORDER}
        peak = max(abs(t) for t in torques.values())
        results.append({**base, "vertical_foot_load_n": dict(zip(supports, map(float, forces))),
                        "joint_torque_nm": torques, "peak_abs_joint_torque_nm": peak,
                        "estimated_continuous_limit_nm": limit,
                        "static_margin_ratio": limit / peak if peak else None,
                        "within_estimated_limit": peak <= limit,
                        "meets_proposed_1_5_static_margin": peak * 1.5 <= limit})
    return {"method": "Neutral-pose vertical force/moment equilibrium and qfrc_bias - J^T F; excludes dynamic gait loads, friction-model fidelity and thermals.",
            "com_world_m": com.tolist(), "foot_centers_world_m": {k: v.tolist() for k, v in foot.items()}, "cases": results}


def pd_sanity_rollout(config: dict, seconds: float = 5.0):
    """Observe limited ideal-PD settling, explicitly not learned locomotion."""
    model, data = standing_model(config)
    initial_height = float(data.qpos[2])
    minimum_height = initial_height
    peak_torque = 0.0
    saturation_steps = 0
    steps = int(round(seconds / model.opt.timestep))
    finite = True
    for _ in range(steps):
        mujoco.mj_step(model, data)
        finite = bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)))
        if not finite:
            break
        minimum_height = min(minimum_height, float(data.qpos[2]))
        torque = float(np.max(np.abs(data.actuator_force)))
        peak_torque = max(peak_torque, torque)
        saturation_steps += int(torque >= config["actuator"]["torque_limit_nm"] * 0.999)
    return {"model_type": "Unidentified ideal position PD; no BAM and no policy",
            "duration_s": float(data.time), "finite": finite,
            "initial_base_height_m": initial_height, "minimum_base_height_m": minimum_height,
            "final_base_height_m": float(data.qpos[2]), "peak_abs_actuator_torque_nm": peak_torque,
            "fraction_steps_any_actuator_at_limit": saturation_steps / max(steps, 1),
            "final_max_joint_speed_rad_s": float(np.max(np.abs(data.qvel[6:]))) if finite else None,
            "interpretation": "Numerical/standing sanity observation only. This does not validate real-motor control, walking or sim-to-real transfer."}


def battery_screen():
    return {"status": "Sizing assumptions, no selected battery or measured endurance",
            "nominal_voltage_v": 7.4, "candidate_capacity_ah": 0.65,
            "nominal_energy_wh": 7.4 * 0.65, "usable_fraction": 0.8,
            "assumed_conversion_efficiency": 0.88,
            "allowable_average_output_power_w": {"10_minutes": 7.4 * .65 * .8 * .88 / (10 / 60), "15_minutes": 7.4 * .65 * .8 * .88 / (15 / 60)},
            "notes": "2S pack must feed a regulated 5V servo rail; 8.4V full charge cannot feed stock XL330 directly. Energy sizing does not establish transient-current capability, rail sizing, or purchased-pack mass."}


def sizing_sweep(config: dict):
    """Screen 45 explicit variants; never scale down the motor mass/envelope."""
    rows = []
    for nonmotor_scale, length_offset, hip_pitch in itertools.product((.8, 1.0, 1.2), (-.01, 0, .01), (.25, .35, .4, .5, .6)):
        candidate = copy.deepcopy(config)
        for key in candidate["mass_kg"]:
            if key != "servo":
                candidate["mass_kg"][key] *= nonmotor_scale
        for key in ("upper_length", "lower_length"):
            candidate["geometry_m"][key] += length_offset
        candidate["home_q_rad"] = [0, hip_pitch, -2 * hip_pitch]
        screen = static_support_screen(candidate)
        four = screen["cases"][0]
        viable_three = [case for case in screen["cases"] if case["case"].startswith("three_") and case["vertical_static_equilibrium"]]
        rows.append({"nonmotor_mass_scale": nonmotor_scale, "mass_kg": total_mass(candidate),
                     "upper_length_m": candidate["geometry_m"]["upper_length"], "lower_length_m": candidate["geometry_m"]["lower_length"],
                     "home_hip_pitch_rad": hip_pitch, "four_foot_peak_torque_nm": four["peak_abs_joint_torque_nm"],
                     "four_foot_static_margin": four["static_margin_ratio"],
                     "worst_statically_supported_three_foot_margin": min(c["static_margin_ratio"] for c in viable_three) if viable_three else None,
                     "three_foot_cases_statically_possible_without_body_shift": len(viable_three)})
    return {"variant_count": len(rows), "notes": "Parameter sensitivity only. Twelve servo masses and envelopes remain unchanged. Nonmotor allowances vary ±20%; link lengths vary ±10mm. No hardware packaging, singularity reserve, or complete gait validation. Three-foot rows use the unchanged body pose, not an optimized crawl.", "variants": rows}


def make_report(config: dict):
    model, data = standing_model(config)
    expected_mass = total_mass(config)
    mass = float(np.sum(model.body_mass))
    kinematics = kinematic_audit(config)
    config_sha = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    foot_soles = [float(data.site(f"{leg}_foot").xpos[2] - config["geometry_m"]["foot_radius"]) for leg in LEG_ORDER]
    return {"stage": "Primitive geometry and load screening, before BAM integration and RL",
            "config_sha256": config_sha, "environment": {"python": platform.python_version(), "mujoco": mujoco.__version__, "numpy": np.__version__},
            "user_requirements": {"size": "smallest that keeps construction straightforward", "terrain": "carpet and small doorway thresholds", "active_walking_minutes": [10, 15], "owner_servo_characterization": False},
            "structure": {"actuators": model.nu, "hinge_joints": model.njnt - 1, "free_joints": 1,
                          "config_mass_kg": expected_mass, "mujoco_mass_kg": mass,
                          "mass_error_kg": abs(expected_mass - mass), "foot_sole_height_m": foot_soles,
                          "positive_body_inertias": bool(np.all(model.body_inertia[1:] > 0)),
                          "body_envelope_m": [config["geometry_m"][k] for k in ("body_length", "body_width", "body_height")]},
            "kinematics": kinematics, "static_support": static_support_screen(config, model, data),
            "sizing_sweep": sizing_sweep(config), "pd_sanity": pd_sanity_rollout(config), "battery_screen": battery_screen(),
            "gates": {"geometry_implementation": "pass" if kinematics["passed"] and abs(expected_mass - mass) < 1e-9 else "fail",
                      "motor_selection": "open: conservative gait load, model provenance and budget must converge",
                      "realistic_actuator_physics": "not implemented: published BAM integration is next",
                      "manufacturing_and_self_collision": "not assessed; primitive motor housings are visual-only",
                      "rl_training": "not started", "carpet_and_threshold_traversal": "not demonstrated",
                      "battery_runtime": "not demonstrated", "hardware_validation": "not performed"}}


def write_report(config: dict, output: Path):
    result = make_report(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    structure = result["structure"]
    kin = result["kinematics"]
    dims = " × ".join(f"{dim * 1000:.0f}" for dim in structure["body_envelope_m"])
    lines = ["# Primitive simulation validation", "", f"Environment: MuJoCo {mujoco.__version__}, Python {platform.python_version()}.", "",
             "This is an original parametric geometry model and load screen. It is not a trained walking robot, manufacturing CAD, or a calibrated digital twin.", "",
             f"The model has **{structure['actuators']} joints** and an estimated **{structure['mujoco_mass_kg']*1000:.0f} g** component budget. The torso envelope is **{dims} mm**; that is not the full robot's exterior size.", "",
             "## Geometry checks", "",
             f"Analytical forward kinematics were checked at {kin['samples']} asymmetric poses on all four legs, including rotated/transformed floating bases. Maximum foot-position error: {kin['fk_max_error_m']:.3g} m. Jacobians agree with MuJoCo within {kin['jacobian_mujoco_max_error_m_per_rad']:.3g} m/rad and independent finite differences within {kin['jacobian_finite_difference_max_error_m_per_rad']:.3g} m/rad.", "",
             "## Neutral-pose static load screen", "", "Loads satisfy vertical force and roll/pitch moment balance. Equal sharing is not assumed. Includes modeled link gravity, but excludes dynamic gait forces and motor thermals.", "",
             "| Supports | Static equilibrium | Peak joint torque | Margin to 0.10 N·m estimate |", "|---|---|---:|---:|"]
    for case in result["static_support"]["cases"]:
        if case["vertical_static_equilibrium"]:
            lines.append(f"| {case['case']} | Yes | {case['peak_abs_joint_torque_nm']:.4f} N·m | {case['static_margin_ratio']:.2f}× |")
        else:
            lines.append(f"| {case['case']} | No; shift body or use dynamics | — | — |")
    lines += ["", "A three-foot crawl needs the center of mass inside its support triangle. Merely lifting one foot does not guarantee three equal loads. The 0.10 N·m motor figure is a manufacturer estimate, and a 1.5× static margin is a proposed screening target, not a proven thermal limit.", "",
              "## Parameter sensitivity", "", f"Screened {result['sizing_sweep']['variant_count']} combinations of nonmotor mass allowances (±20%), upper/lower link lengths (±10 mm), and stance knee flexion. Motor masses and physical envelopes remain fixed. Every variant's load results are in the JSON. These calculations do not select a manufacturable design or establish a complete gait.", "",
              "## Ideal-PD sanity observation", "", f"Over {result['pd_sanity']['duration_s']:.2f} s, base height changed from {result['pd_sanity']['initial_base_height_m']:.3f} to {result['pd_sanity']['final_base_height_m']:.3f} m. Any joint reached the torque limit in {100*result['pd_sanity']['fraction_steps_any_actuator_at_limit']:.1f}% of steps. This uses arbitrary torque-limited PD gains; no inference about real XL330 performance follows.", "",
              "## Battery allowance", "", "A hypothetical 2S 650 mAh pack stores 4.81 Wh. At an assumed 80% usable energy and 88% conversion efficiency, 10–15 minutes permits approximately 20.3–13.5 W average combined output. This is a sizing calculation, not a selected pack or runtime result; transient current still needs design analysis.", "",
              "## Open gates", ""]
    lines += [f"- **{key.replace('_', ' ')}:** {value}." for key, value in result["gates"].items()]
    lines += ["", "See [the implementation plan](../docs/implementation/PLAN.md) and [actuator research](../docs/microduck-review/RESEARCH.md). The JSON beside this file contains complete forces, joint torques, configuration hash and numerical results.", ""]
    output.with_suffix(".md").write_text("\n".join(lines))
    return result
