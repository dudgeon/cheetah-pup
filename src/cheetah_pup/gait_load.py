"""Screen prescribed crawl loads; this does not run or validate locomotion.

The static calculation includes every modeled body's gravity and balances the
floating base. Timing calculations use periodic position differences and
rigid-body inverse dynamics. A nonzero floating-base residual means the
vertical-only ground-force assumption cannot produce the requested motion;
joint torques in that case are conditional diagnostics, not actuator demands
for a feasible gait. Tangential forces, motor dynamics and thermal behavior
remain separate gates.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np

from .analysis import joint_addresses, solve_vertical_support
from .gait_demo import trajectory
from .kinematics import JOINT_ORDER, LEG_ORDER
from .model import load_config


JOINT_NAMES = tuple(f"{leg}_{joint}" for leg in LEG_ORDER for joint in JOINT_ORDER)
NO_LOAD_SPEED_RPM = 103.0
SPEED_SOURCE = "https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/"
TORQUE_SOURCE = "https://www.robotis.us/dynamixel-xl330-m288-t/"


def periodic_derivatives(model, positions, cycle_duration_s, stride_m):
    """Central differences across a translation-periodic fixed-attitude cycle.

    Shift the preceding/following cycle by one stride at the seam. This avoids
    inventing a velocity jump when the last sample returns to the first pose.
    MuJoCo converts quaternion positions into generalized velocities; fixed
    base attitude makes the second difference unambiguous in the same frame.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != model.nq or len(positions) < 4:
        raise ValueError("Need at least four complete configuration samples")
    if not np.isfinite(cycle_duration_s) or cycle_duration_s <= 0:
        raise ValueError("Cycle duration must be finite and positive")
    if not np.all(np.isfinite(positions)) or not np.isfinite(stride_m):
        raise ValueError("Positions and stride must be finite")
    if not np.allclose(positions[:, 3:7], positions[0, 3:7], atol=1e-12, rtol=0):
        raise ValueError("This timing screen requires constant base orientation")
    dt = cycle_duration_s / len(positions)
    velocity = np.zeros((len(positions), model.nv))
    acceleration = np.zeros_like(velocity)
    for index, current in enumerate(positions):
        before = positions[(index - 1) % len(positions)].copy()
        after = positions[(index + 1) % len(positions)].copy()
        if index == 0:
            before[0] -= stride_m
        if index == len(positions) - 1:
            after[0] += stride_m
        backward, forward = np.zeros(model.nv), np.zeros(model.nv)
        mujoco.mj_differentiatePos(model, backward, dt, before, current)
        mujoco.mj_differentiatePos(model, forward, dt, current, after)
        velocity[index] = (forward + backward) / 2
        acceleration[index] = (forward - backward) / dt
    return velocity, acceleration


def vertical_force_map(model, data, supports):
    """Generalized force for unit upward force at each active foot site."""
    columns = []
    for leg in supports:
        jacobian, angular = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacobian, angular, model.site(f"{leg}_foot").id)
        columns.append(jacobian[2])
    return np.column_stack(columns)


def closest_vertical_forces(base_map, required_wrench, moment_length_m=0.1):
    """Nonnegative least-squares vertical forces, including all six base DOFs.

    Enumerate at most 15 positive-force subsets. Divide moments by an explicit
    0.1 m reference length to compare residuals in N. A solution always exists
    as a least-squares diagnostic, but it need not balance the floating base.
    Callers must inspect the unscaled force and moment residuals.
    """
    matrix = np.asarray(base_map, dtype=float)
    target = np.asarray(required_wrench, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != 6 or target.shape != (6,):
        raise ValueError("Expected a 6-by-supports wrench map and six-vector")
    if not 1 <= matrix.shape[1] <= 4 or moment_length_m <= 0:
        raise ValueError("Expected one to four feet and positive moment scale")
    scale = np.array(
        [1.0, 1.0, 1.0, 1 / moment_length_m, 1 / moment_length_m, 1 / moment_length_m]
    )
    weighted = matrix * scale[:, None]
    desired = target * scale
    candidates = [np.zeros(matrix.shape[1])]
    for count in range(1, matrix.shape[1] + 1):
        for subset in itertools.combinations(range(matrix.shape[1]), count):
            loads, *_ = np.linalg.lstsq(weighted[:, subset], desired, rcond=None)
            if np.min(loads) >= -1e-10:
                full = np.zeros(matrix.shape[1])
                full[list(subset)] = np.maximum(loads, 0)
                candidates.append(full)
    # A tiny load-norm tie breaker selects the minimum-norm force distribution
    # if several subsets have effectively identical wrench residuals.
    errors = np.array(
        [np.linalg.norm(weighted @ force - desired) for force in candidates]
    )
    best_error = float(np.min(errors))
    exact_ties = [
        force for force, error in zip(candidates, errors) if error <= best_error + 1e-11
    ]
    return min(exact_ties, key=lambda force: float(force @ force))


def static_frame(model, data, supports):
    """Return vertical loads and generalized gravity-minus-ground demand.

    The data must describe a zero-velocity pose. Gravity is vertical. Returning
    None means the COM is outside the available static support polygon.
    """
    if np.max(np.abs(data.qvel)) > 1e-12:
        raise ValueError("Static screening requires zero velocity")
    if not np.allclose(model.opt.gravity[:2], 0) or model.opt.gravity[2] >= 0:
        raise ValueError("Static screening requires downward vertical gravity")
    com = data.subtree_com[model.body("base").id]
    weight = -float(model.body_mass.sum()) * model.opt.gravity[2]
    loads = solve_vertical_support(
        [data.site(f"{leg}_foot").xpos[:2] for leg in supports], com[:2], weight
    )
    if loads is None:
        return None
    contact_map = vertical_force_map(model, data, supports)
    demand = data.qfrc_bias - contact_map @ loads
    return loads, demand


def minimum_peak_static_allocation(model, data, supports):
    """Minimize peak joint torque over the one redundant vertical-load DOF.

    For four non-collinear coplanar supports, force/roll/pitch balance has one
    null-space direction. Nonnegative forces bound an interval in that scalar.
    Each signed joint torque is affine along it, so the convex piecewise-linear
    maximum attains a minimum at an endpoint or a pairwise line intersection.
    Enumerating those candidates is exact up to floating-point arithmetic.

    This is a static force-allocation bound with the body pose held fixed. It
    says nothing about achieving those loads with position-controlled servos.
    Three non-collinear support feet have no allocation freedom.
    """
    result = static_frame(model, data, supports)
    if result is None:
        return None
    forces, demand = result
    if len(supports) < 4:
        return forces, demand
    contact_map = vertical_force_map(model, data, supports)
    _, singular, right = np.linalg.svd(contact_map[:6])
    rank = int(np.sum(singular > 1e-10))
    if len(supports) - rank != 1:
        raise ValueError(
            "Minimax allocation requires exactly one redundant vertical-load direction"
        )
    direction = right[-1]
    low, high = -float("inf"), float("inf")
    for force, delta in zip(forces, direction):
        if delta > 1e-12:
            low = max(low, -force / delta)
        elif delta < -1e-12:
            high = min(high, -force / delta)
    if not np.isfinite(low) or not np.isfinite(high) or low > high + 1e-10:
        raise ValueError("Could not bound the nonnegative contact-force interval")
    indices = np.concatenate([joint_addresses(model, leg)[1] for leg in LEG_ORDER])
    torque_slope = -(contact_map @ direction)[indices]
    slopes = np.r_[torque_slope, -torque_slope]
    intercepts = np.r_[demand[indices], -demand[indices]]
    candidates = [low, high, 0.0]
    for left, right_index in itertools.combinations(range(len(slopes)), 2):
        difference = slopes[left] - slopes[right_index]
        if abs(difference) > 1e-12:
            crossing = (intercepts[right_index] - intercepts[left]) / difference
            if low <= crossing <= high:
                candidates.append(crossing)
    scalar = min(
        candidates, key=lambda value: float(np.max(intercepts + slopes * value))
    )
    optimized = forces + direction * scalar
    if np.min(optimized) < -1e-9:
        raise ValueError("Minimax allocation produced a negative support load")
    return optimized, data.qfrc_bias - contact_map @ optimized


def geom_floor_clearance(model, data, geom_id):
    """Exact bottom height of a modeled primitive above the z=0 plane.

    This geometric bound includes visual envelopes regardless of collision
    flags. Meshes and other unsupported types return None instead of silently
    claiming clearance. It does not check robot-vs-robot interference.
    """
    kind = model.geom_type[geom_id]
    size = model.geom_size[geom_id]
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        extent = np.abs(rotation[2]) @ size
    elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        extent = size[0] + abs(rotation[2, 2]) * size[1]
    elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        extent = size[0]
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        extent = size[0] * np.sqrt(max(0.0, 1 - rotation[2, 2] ** 2)) + size[1] * abs(
            rotation[2, 2]
        )
    elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        extent = np.linalg.norm(rotation[2] * size)
    else:
        return None
    return float(data.geom_xpos[geom_id, 2] - extent)


def _joint_summary(values):
    return {
        name: {
            "peak_abs": float(np.max(np.abs(values[:, index]))),
            "rms": float(np.sqrt(np.mean(values[:, index] ** 2))),
        }
        for index, name in enumerate(JOINT_NAMES)
    }


def posture_sweep(config, frames_per_step=48, target_speed_m_s=0.05):
    """Compare 15 explicit alternatives without calling any a feasible gait.

    Only stance height and stride change. Motors, mass, links, bounds and the
    12 mm foot lift are held fixed. Reach and limit failures are preserved as
    rejected rows instead of disappearing from the comparison.
    """
    rows = []
    no_load_rad_s = NO_LOAD_SPEED_RPM * 2 * np.pi / 60
    for height, stride in itertools.product(
        (0.124, 0.130, 0.135, 0.138, 0.140), (0.020, 0.030, 0.040)
    ):
        candidate = copy.deepcopy(config)
        candidate.setdefault("gait", {})["base_height_m"] = height
        row = {
            "base_height_m": height,
            "stride_per_cycle_m": stride,
            "foot_lift_m": 0.012,
            "target_speed_m_s": target_speed_m_s,
            "cycle_duration_s": stride / target_speed_m_s,
        }
        try:
            model, frames, _ = trajectory(
                candidate,
                frames_per_step=frames_per_step,
                stride=stride,
                clearance=0.012,
            )
            data = mujoco.MjData(model)
            indices = np.concatenate(
                [joint_addresses(model, leg)[1] for leg in LEG_ORDER]
            )
            torques, optimized_torques, base_residual = [], [], 0.0
            for index, frame in enumerate(frames):
                data.qpos[:] = frame["qpos"]
                mujoco.mj_forward(model, data)
                result = static_frame(model, data, frame["active"])
                if result is None:
                    raise ValueError(f"No static support solution at frame {index}")
                _, generalized = result
                torques.append(generalized[indices])
                _, optimized_demand = minimum_peak_static_allocation(
                    model, data, frame["active"]
                )
                optimized_torques.append(optimized_demand[indices])
                base_residual = max(
                    base_residual, float(np.max(np.abs(generalized[:6])))
                )
            if base_residual > 1e-8:
                raise ValueError("Static support solution has a nonzero base wrench")
            positions = np.array([frame["qpos"] for frame in frames])
            speed, acceleration = periodic_derivatives(
                model, positions, stride / target_speed_m_s, stride
            )
            peak = float(np.max(np.abs(torques)))
            max_rms = float(np.max(np.sqrt(np.mean(np.array(torques) ** 2, axis=0))))
            peak_speed = float(np.max(np.abs(speed[:, indices])))
            margin = candidate["actuator"]["torque_limit_nm"] / peak
            optimized_peak = float(np.max(np.abs(optimized_torques)))
            optimized_margin = candidate["actuator"]["torque_limit_nm"] / optimized_peak
            row.update(
                {
                    "kinematically_valid": True,
                    "static_peak_abs_joint_torque_nm": peak,
                    "largest_joint_static_rms_nm": max_rms,
                    "minimum_margin_to_estimated_continuous": margin,
                    "minimax_allocation_peak_joint_torque_nm": optimized_peak,
                    "minimax_allocation_margin_to_estimated_continuous": optimized_margin,
                    "proposed_1_5_static_margin_met": bool(margin >= 1.5),
                    "target_peak_joint_speed_rad_s": peak_speed,
                    "target_peak_speed_fraction_of_published_no_load": peak_speed
                    / no_load_rad_s,
                    "target_peak_base_translation_acceleration_m_s2": float(
                        np.max(np.linalg.norm(acceleration[:, :3], axis=1))
                    ),
                    "passes_static_margin_and_unloaded_speed_only": bool(
                        margin >= 1.5 and peak_speed <= no_load_rad_s
                    ),
                    "passes_minimax_static_margin_and_unloaded_speed_only": bool(
                        optimized_margin >= 1.5 and peak_speed <= no_load_rad_s
                    ),
                }
            )
        except ValueError as error:
            row.update({"kinematically_valid": False, "rejection_reason": str(error)})
        rows.append(row)
    valid = [row for row in rows if row["kinematically_valid"]]
    best = max(
        valid,
        key=lambda row: row["minimum_margin_to_estimated_continuous"],
        default=None,
    )
    best_minimax = max(
        valid,
        key=lambda row: row["minimax_allocation_margin_to_estimated_continuous"],
        default=None,
    )
    return {
        "variant_count": len(rows),
        "frames_per_step": frames_per_step,
        "support_shift_fraction": config.get("gait", {}).get(
            "support_shift_fraction", 1.0
        ),
        "held_fixed": "Robot geometry, motor envelopes and mass, all mass allowances, joint limits, configured COM-shift fraction and 12 mm foot lift; only gait base height and stride vary.",
        "assembly_clearance_assessed_for_sweep_variants": False,
        "clearance_scope": "Sweep validity means reach, joint limits and static support only. Self-interference and assembly clearance were not audited for every variant; the separate assembly audit applies to its explicitly selected trajectory, not this whole sweep.",
        "method": "Complete sampled cycle gravity-minus-ground joint demands, nonnegative static force/moment balance, and periodic finite-difference speeds at 0.05 m/s. Motor friction, loaded speed, dynamic wrench feasibility and interference are excluded.",
        "valid_variants": len(valid),
        "best_static_margin_variant": best,
        "best_minimax_static_margin_variant": best_minimax,
        "any_pass_static_margin_and_unloaded_speed_only": any(
            row.get("passes_static_margin_and_unloaded_speed_only", False)
            for row in rows
        ),
        "any_pass_minimax_static_margin_and_unloaded_speed_only": any(
            row.get("passes_minimax_static_margin_and_unloaded_speed_only", False)
            for row in rows
        ),
        "interpretation": "This limited posture search cannot select or reject the actuator architecture. Passing the two scalar screens is still insufficient for feasible walking; failure calls for gait/contact optimization, geometry/mass changes or a stronger actuator comparison.",
        "variants": rows,
    }


def screen(
    config,
    frames_per_step=96,
    slow_cycle_s=6.4,
    target_speed_m_s=0.05,
    include_sweep=True,
):
    if frames_per_step < 16:
        raise ValueError("Use at least 16 samples per step for this screen")
    if target_speed_m_s <= 0 or not np.isfinite(target_speed_m_s):
        raise ValueError("Target speed must be finite and positive")
    model, frames, metadata = trajectory(config, frames_per_step=frames_per_step)
    data = mujoco.MjData(model)
    joint_indices = np.concatenate(
        [joint_addresses(model, leg)[1] for leg in LEG_ORDER]
    )
    positions = np.array([frame["qpos"] for frame in frames])
    static_torques, static_loads, static_residuals = [], [], []
    optimized_torques, optimized_loads, optimized_residuals = [], [], []
    lowest_geom, lowest_clearance = None, float("inf")
    unsupported_geoms = set()
    excluded_keepouts = set()
    swing_peak = {leg: 0.0 for leg in LEG_ORDER}
    foot_radius = config["geometry_m"]["foot_radius"]
    foot_geom_ids = {model.geom(f"{leg}_foot_collision").id for leg in LEG_ORDER}
    stance_errors = []
    for index, frame in enumerate(frames):
        data.qpos[:] = frame["qpos"]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        result = static_frame(model, data, frame["active"])
        if result is None:
            raise ValueError(f"No vertical static equilibrium at frame {index}")
        loads, generalized = result
        static_torques.append(generalized[joint_indices])
        static_loads.append(
            {leg: float(force) for leg, force in zip(frame["active"], loads)}
        )
        static_residuals.append(generalized[:6])
        optimized_forces, optimized_demand = minimum_peak_static_allocation(
            model, data, frame["active"]
        )
        optimized_torques.append(optimized_demand[joint_indices])
        optimized_loads.append(
            {leg: float(force) for leg, force in zip(frame["active"], optimized_forces)}
        )
        optimized_residuals.append(optimized_demand[:6])
        for leg in frame["active"]:
            stance_errors.append(
                abs(float(data.site(f"{leg}_foot").xpos[2] - foot_radius))
            )
        if len(frame["active"]) == 3:
            moving = frame["moving"]
            swing_peak[moving] = max(
                swing_peak[moving],
                float(data.site(f"{moving}_foot").xpos[2] - foot_radius),
            )
        for geom_id in range(model.ngeom):
            if model.geom_bodyid[geom_id] == 0 or geom_id in foot_geom_ids:
                continue
            name = model.geom(geom_id).name or f"geom_{geom_id}"
            if model.geom_group[geom_id] == 5 and model.geom_contype[geom_id] == 0:
                excluded_keepouts.add(name)
                continue
            bottom = geom_floor_clearance(model, data, geom_id)
            if bottom is None:
                unsupported_geoms.add(name)
            elif bottom < lowest_clearance:
                lowest_clearance, lowest_geom = bottom, name
    static_torques = np.array(static_torques)
    static_residuals = np.array(static_residuals)
    optimized_torques = np.array(optimized_torques)
    optimized_residuals = np.array(optimized_residuals)
    limit = config["actuator"]["torque_limit_nm"]
    static_peak = float(np.max(np.abs(static_torques)))
    no_load_rad_s = NO_LOAD_SPEED_RPM * 2 * np.pi / 60
    stride = metadata["stride_per_cycle_m"]
    timing = []
    for name, duration in (
        ("illustration_timing", slow_cycle_s),
        ("initial_speed_goal", stride / target_speed_m_s),
    ):
        velocity, acceleration = periodic_derivatives(
            model, positions, duration, stride
        )
        demands, residuals = [], []
        for frame, qvel, qacc in zip(frames, velocity, acceleration):
            data.qpos[:] = frame["qpos"]
            data.qvel[:] = qvel
            mujoco.mj_forward(model, data)
            inertial = np.zeros(model.nv)
            mujoco.mj_mulM(model, data, inertial, qacc)
            required = inertial + data.qfrc_bias
            contact_map = vertical_force_map(model, data, frame["active"])
            loads = closest_vertical_forces(contact_map[:6], required[:6])
            demand = required - contact_map @ loads
            demands.append(demand[joint_indices])
            residuals.append(demand[:6])
        demands, residuals = np.array(demands), np.array(residuals)
        force_residual = np.linalg.norm(residuals[:, :3], axis=1)
        moment_residual = np.linalg.norm(residuals[:, 3:], axis=1)
        balanced = (force_residual < 1e-7) & (moment_residual < 1e-9)
        speed_peak = float(np.max(np.abs(velocity[:, joint_indices])))
        timing.append(
            {
                "case": name,
                "cycle_duration_s": duration,
                "average_progress_m_s": stride / duration,
                "joint_speed_rad_s": _joint_summary(velocity[:, joint_indices]),
                "peak_joint_speed_rad_s": speed_peak,
                "peak_speed_fraction_of_published_no_load": speed_peak / no_load_rad_s,
                "fraction_joint_samples_above_published_no_load": float(
                    np.mean(np.abs(velocity[:, joint_indices]) > no_load_rad_s)
                ),
                "peak_base_translation_acceleration_m_s2": float(
                    np.max(np.linalg.norm(acceleration[:, :3], axis=1))
                ),
                "peak_joint_acceleration_rad_s2": float(
                    np.max(np.abs(acceleration[:, joint_indices]))
                ),
                "vertical_only_dynamic_balance": bool(np.all(balanced)),
                "fraction_frames_vertical_only_balanced": float(np.mean(balanced)),
                "max_floating_base_force_residual_n": float(np.max(force_residual)),
                "max_floating_base_moment_residual_nm": float(np.max(moment_residual)),
                "conditional_joint_torque_nm": _joint_summary(demands),
                "conditional_peak_abs_joint_torque_nm": float(np.max(np.abs(demands))),
                "interpretation": "Inverse-dynamics diagnostic under nonnegative vertical-only foot forces. Any residual is an unprovided floating-base wrench. Conditional joint torques do not constitute a feasible gait; horizontal friction forces and their joint loads must be solved before dynamic torque validation.",
            }
        )
    peak_index = np.unravel_index(
        np.argmax(np.abs(static_torques)), static_torques.shape
    )
    optimized_peak_index = np.unravel_index(
        np.argmax(np.abs(optimized_torques)), optimized_torques.shape
    )
    optimized_peak = float(np.max(np.abs(optimized_torques)))
    return {
        "kind": "Prescribed-crawl rigid-body load and timing screen; no learned policy or forward-dynamics walking",
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
        "mujoco_version": mujoco.__version__,
        "samples": len(frames),
        "frames_per_step": frames_per_step,
        "estimated_mass_kg": float(model.body_mass.sum()),
        "trajectory": {
            key: metadata[key]
            for key in (
                "stride_per_cycle_m",
                "foot_clearance_m",
                "base_height_m",
                "support_shift_fraction",
                "minimum_swing_support_load_fraction",
            )
            if key in metadata
        },
        "manufacturer_references": {
            "candidate": "ROBOTIS XL330-M288-T",
            "supply_v": 5.0,
            "no_load_speed_rpm": NO_LOAD_SPEED_RPM,
            "no_load_speed_rad_s": no_load_rad_s,
            "speed_source": SPEED_SOURCE,
            "torque_source": TORQUE_SOURCE,
            "continuous_torque_screen_nm": limit,
            "notes": "103 rpm is unloaded at 5 V, not a loaded speed guarantee or a torque-speed envelope. The 0.10 N m manufacturer estimate is not a thermal guarantee. Neither is a calibrated actuator model.",
        },
        "quasistatic": {
            "method": "At every pose set qvel=0, solve nonnegative vertical ground loads satisfying total force and roll/pitch moment balance using minimum summed squared foot forces, then compute qfrc_bias - J.T F. Includes all modeled link gravity. The alternative minimax allocation separately optimizes joint demand.",
            "all_frames_static_equilibrium": True,
            "max_floating_base_force_residual_n": float(
                np.max(np.linalg.norm(static_residuals[:, :3], axis=1))
            ),
            "max_floating_base_moment_residual_nm": float(
                np.max(np.linalg.norm(static_residuals[:, 3:], axis=1))
            ),
            "joint_torque_nm": _joint_summary(static_torques),
            "peak_abs_joint_torque_nm": static_peak,
            "peak_frame": int(peak_index[0]),
            "peak_joint": JOINT_NAMES[peak_index[1]],
            "peak_frame_support_loads_n": static_loads[peak_index[0]],
            "minimum_margin_to_estimated_continuous": limit / static_peak,
            "proposed_1_5_margin_met_all_frames": bool(1.5 * static_peak <= limit),
            "fraction_joint_samples_above_estimated_continuous": float(
                np.mean(np.abs(static_torques) > limit)
            ),
            "rms_scope": "RMS is equal-time over the prescribed cycle, not an electrical-current or motor-temperature prediction.",
        },
        "alternative_static_allocation": {
            "method": "Keep every pose fixed. For four feet, parameterize the one-dimensional null space of vertical force/roll/pitch balance. Nonnegative loads define an interval; enumerate its endpoints and intersections of signed affine joint-torque lines to minimize the maximum absolute joint torque. Three-foot loads are uniquely determined.",
            "peak_abs_joint_torque_nm": optimized_peak,
            "joint_torque_nm": _joint_summary(optimized_torques),
            "minimum_margin_to_estimated_continuous": limit / optimized_peak,
            "proposed_1_5_margin_met_all_frames": bool(optimized_peak * 1.5 <= limit),
            "peak_frame": int(optimized_peak_index[0]),
            "peak_joint": JOINT_NAMES[optimized_peak_index[1]],
            "peak_frame_support_loads_n": optimized_loads[optimized_peak_index[0]],
            "max_floating_base_force_residual_n": float(
                np.max(np.linalg.norm(optimized_residuals[:, :3], axis=1))
            ),
            "max_floating_base_moment_residual_nm": float(
                np.max(np.linalg.norm(optimized_residuals[:, 3:], axis=1))
            ),
            "interpretation": "An ideal static force-allocation bound for these fixed poses, not an implemented controller. Position-controlled servos do not automatically realize these contact loads; compliance, contact sensing/estimation and tracking remain unvalidated. This prevents judging the actuator only from the arbitrary minimum-squared-foot-force allocation.",
        },
        "timing": timing,
        "clearance": {
            "minimum_nonfoot_primitive_floor_clearance_m": lowest_clearance,
            "limiting_geom": lowest_geom,
            "unsupported_geom_types": sorted(unsupported_geoms),
            "excluded_nonphysical_keepouts": sorted(excluded_keepouts),
            "maximum_stance_sole_height_error_m": float(max(stance_errors)),
            "peak_swing_sole_height_m_by_leg": swing_peak,
            "scope": "Physical component geometry against a perfectly flat plane only; group-5 non-colliding cable/port reservations are excluded. A 12 mm peak swing height gives no guaranteed clearance over a 10 mm doorway threshold along the whole foot path. No carpet deformation, obstacle placement, self-interference or manufacturing tolerance is assessed here.",
        },
        "posture_sweep": posture_sweep(config, target_speed_m_s=target_speed_m_s)
        if include_sweep
        else None,
        "limitations": [
            "Masses and inertias remain assembly estimates. The derivative screen includes the compiled model's armature, which may still be a provisional value.",
            "No actuator torque-speed coupling, delay, friction, supply droop, thermal model or tracking controller is included in these demand calculations.",
            "The smooth IK illustration was designed to explain support transfer, not optimized for torque or speed. Its feasibility must not be equated with quadruped architecture feasibility.",
            "Faster retiming multiplies joint speed by inverse cycle duration and inertial accelerations approximately by its square. The 0.05 m/s target needs a new gait, a longer stride, or explicit dynamic optimization if this retiming fails.",
            "No RL training, forward walking rollout, threshold traversal or real hardware validation is performed.",
        ],
    }


def write_report(config, output, **kwargs):
    result = screen(config, **kwargs)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    quasi = result["quasistatic"]
    lines = [
        "# Prescribed crawl load and timing screen",
        "",
        "This evaluates the illustrated motion with the current rigid-body assembly estimates. It does not demonstrate a trained policy, feasible dynamic gait, or accurate real-servo tracking.",
        "",
        f"Screened {result['samples']} poses, including each body shift and foot swing. Model mass: {result['estimated_mass_kg'] * 1000:.1f} g. Configuration SHA-256: `{result['config_sha256']}`.",
        "",
        f"Selected gait: {result['trajectory']['base_height_m'] * 1000:.0f} mm base height, {result['trajectory']['stride_per_cycle_m'] * 1000:.0f} mm advance per cycle and {result['trajectory']['foot_clearance_m'] * 1000:.0f} mm peak foot lift. The COM target lies {result['trajectory'].get('support_shift_fraction', 1.0) * 100:.0f}% of the way from the four-foot footprint center toward the supporting triangle's centroid.",
        "",
        "## Whole-cycle static demand",
        "",
        f"Every sampled pose has nonnegative vertical support forces balancing the modeled robot's weight and moments. This baseline chooses the smallest summed squared foot forces; joint effort is optimized separately below. Gravity from all moving links is included. Maximum floating-base force residual: {quasi['max_floating_base_force_residual_n']:.3g} N; moment residual: {quasi['max_floating_base_moment_residual_nm']:.3g} N·m.",
        "",
        f"Peak static demand is **{quasi['peak_abs_joint_torque_nm']:.4f} N·m** at `{quasi['peak_joint']}`, giving **{quasi['minimum_margin_to_estimated_continuous']:.2f}×** margin to the configured continuous-torque screen. Proposed 1.5× margin across all sampled poses: **{'met' if quasi['proposed_1_5_margin_met_all_frames'] else 'not met'}**.",
        "",
        "| Joint | Static peak, N·m | Static RMS, N·m |",
        "|---|---:|---:|",
    ]
    for name, values in quasi["joint_torque_nm"].items():
        lines.append(f"| {name} | {values['peak_abs']:.4f} | {values['rms']:.4f} |")
    allocation = result["alternative_static_allocation"]
    lines.extend(
        [
            "",
            "The RMS column describes equal-time torque demand, not temperature or electrical current. Passing this static screen does not establish loaded motor speed, actuator tracking or walking.",
            "",
            "## Alternative static contact-load allocation",
            "",
            allocation["method"],
            "",
            f"Optimizing vertical load sharing during four-foot support gives a full-cycle peak of **{allocation['peak_abs_joint_torque_nm']:.4f} N·m**, a **{allocation['minimum_margin_to_estimated_continuous']:.2f}×** margin. The limiting joint is `{allocation['peak_joint']}` at frame {allocation['peak_frame']}. Proposed 1.5× margin: **{'met' if allocation['proposed_1_5_margin_met_all_frames'] else 'not met'}**. Maximum base force/moment residuals remain {allocation['max_floating_base_force_residual_n']:.3g} N / {allocation['max_floating_base_moment_residual_nm']:.3g} N·m.",
            "",
            allocation["interpretation"],
            "",
            "## Timing the same motion",
            "",
            "| Case | Cycle | Progress | Peak joint speed | Fraction of 103 rpm unloaded speed | Peak base acceleration | Base force residual | Base moment residual |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["timing"]:
        lines.append(
            f"| {row['case']} | {row['cycle_duration_s']:.3f} s | {row['average_progress_m_s']:.4f} m/s | {row['peak_joint_speed_rad_s']:.2f} rad/s | {row['peak_speed_fraction_of_published_no_load']:.2f}× | {row['peak_base_translation_acceleration_m_s2']:.2f} m/s² | {row['max_floating_base_force_residual_n']:.3g} N | {row['max_floating_base_moment_residual_nm']:.3g} N·m |"
        )
    lines.extend(
        [
            "",
            "Periodic central differences include the wrap between cycles with one stride of world translation. Inverse dynamics computes `M(q) qacc + bias(q, qvel)`. Nonnegative vertical foot loads minimize the residual of all six floating-base equations; moments are divided by a documented 0.1 m reference length for this least-squares calculation.",
            "",
            "**A nonzero base residual is an unprovided force or moment.** Vertical forces alone cannot create lateral body acceleration. Horizontal contact forces and their joint torques must be solved before claiming a dynamically feasible gait. The JSON includes conditional joint-torque values for diagnostics; these are not validated actuator demands. Accelerating the visual demonstration to the walking target is not a gait controller.",
            "",
            f"The [manufacturer's XL330 specification]({SPEED_SOURCE}) gives 103 rpm unloaded at 5 V. This is only an upper-reference speed, not a loaded torque-speed envelope. The [manufacturer torque estimate]({TORQUE_SOURCE}) is not a guaranteed continuous thermal rating.",
            "",
        ]
    )
    sweep = result["posture_sweep"]
    if sweep is not None:
        lines.extend(
            [
                "## Limited stance and stride alternatives",
                "",
                sweep["held_fixed"],
                "",
                sweep["clearance_scope"],
                "",
                "| Base height | Stride | Geometry | Static peak | Minimax peak | Minimax margin | Target speed / unloaded limit |",
                "|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in sweep["variants"]:
            prefix = f"| {row['base_height_m'] * 1000:.0f} mm | {row['stride_per_cycle_m'] * 1000:.0f} mm |"
            if row["kinematically_valid"]:
                lines.append(
                    prefix
                    + f" Reach/limits pass | {row['static_peak_abs_joint_torque_nm']:.4f} N·m | {row['minimax_allocation_peak_joint_torque_nm']:.4f} N·m | {row['minimax_allocation_margin_to_estimated_continuous']:.2f}× | {row['target_peak_speed_fraction_of_published_no_load']:.2f}× |"
                )
            else:
                reason = row["rejection_reason"].replace("\n", " ").replace("|", "/")
                lines.append(prefix + f" Rejected: {reason} | — | — | — | — |")
        lines.extend(["", sweep["interpretation"], ""])
    lines.extend(
        [
            "## Flat-floor geometry",
            "",
            f"Minimum modeled non-foot primitive clearance is **{result['clearance']['minimum_nonfoot_primitive_floor_clearance_m'] * 1000:.2f} mm**, at `{result['clearance']['limiting_geom']}`. Physical motor envelopes are included regardless of contact flags; {len(result['clearance']['excluded_nonphysical_keepouts'])} nonphysical cable/port keepouts are excluded. Unhandled geometry types: {', '.join(result['clearance']['unsupported_geom_types']) or 'none'}.",
            "",
            result["clearance"]["scope"],
            "",
            "## Remaining limits",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in result["limitations"])
    lines.extend(
        [
            "",
            "Reproduce from the repository root:",
            "",
            "```sh",
            "uv run python -m cheetah_pup.gait_load --config config/robot.json --output reports/gait-load-validation.json",
            "```",
            "",
        ]
    )
    output.with_suffix(".md").write_text("\n".join(lines))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/robot.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/gait-load-validation.json")
    )
    parser.add_argument("--frames-per-step", type=int, default=96)
    parser.add_argument("--slow-cycle", type=float, default=6.4)
    parser.add_argument("--target-speed", type=float, default=0.05)
    arguments = parser.parse_args()
    report = write_report(
        load_config(arguments.config),
        arguments.output,
        frames_per_step=arguments.frames_per_step,
        slow_cycle_s=arguments.slow_cycle,
        target_speed_m_s=arguments.target_speed,
    )
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "static_peak_nm": report["quasistatic"]["peak_abs_joint_torque_nm"],
                "static_margin": report["quasistatic"][
                    "minimum_margin_to_estimated_continuous"
                ],
                "timing_speed_fractions": [
                    row["peak_speed_fraction_of_published_no_load"]
                    for row in report["timing"]
                ],
            },
            indent=2,
        )
    )
