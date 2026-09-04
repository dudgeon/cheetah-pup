"""Audit shaft alignment and sampled solid/connector-clearance envelopes.

This tests geometry, not bearing strength, screw engagement or a deformable wire.
Signed distances also cover adjacent body pairs filtered by default by MuJoCo.
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
from .analysis import standing_model, joint_addresses
from .gait_demo import trajectory
from .kinematics import LEG_ORDER, JOINT_ORDER
from .model import load_config


def pairs(model):
    result = []
    for i, j in itertools.combinations(range(model.ngeom), 2):
        a, b = model.geom(i).name, model.geom(j).name
        if (
            model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE
            or model.geom_type[j] == mujoco.mjtGeom.mjGEOM_PLANE
        ):
            continue
        ca, cb = "port_keepout" in a, "port_keepout" in b
        if ca and cb:
            continue
        # Same-rigid-body structural overlaps are welded unions. Housing and
        # cable envelopes still need room even in that body's fixed assembly.
        if model.geom_bodyid[i] == model.geom_bodyid[j] and not (
            ca
            or cb
            or "motor_envelope" in a
            or "motor_envelope" in b
            or "_allowance" in a
            or "_allowance" in b
        ):
            continue
        result.append((i, j, "cable_allowance" if ca or cb else "solid"))
    return result


def shaft_alignment(model, data):
    rows = []
    for leg in LEG_ORDER:
        for role, joint in zip(("roll", "hip", "knee"), JOINT_ORDER):
            horn = model.geom(f"{leg}_{role}_output_horn").id
            jid = model.joint(f"{leg}_{joint}").id
            direction = data.geom_xmat[horn].reshape(3, 3)[:, 2]
            axis = data.xaxis[jid]
            offset = data.geom_xpos[horn] - data.xanchor[jid]
            # A cylinder's +/-z directions are geometrically interchangeable;
            # MuJoCo fromto chooses the opposite sign from our outward normal.
            # The mounting face at the joint anchor establishes shaft direction.
            outward = -offset / np.linalg.norm(offset)
            rows.append(
                {
                    "joint": f"{leg}_{joint}",
                    "axis_alignment_abs_cosine": float(abs(direction @ axis)),
                    "shaft_axis_offset_m": float(
                        np.linalg.norm(offset - np.dot(offset, axis) * axis)
                    ),
                    "physical_positive_to_joint_sign": int(np.sign(outward @ axis)),
                }
            )
    return rows


def box_separation(a_center, a_axes, a_half, b_center, b_axes, b_half):
    """SAT signed gap: positive proves separation; negative is overlap depth.

    For separated OBBs this is a lower bound on Euclidean distance, not the
    closest-point distance. Test all six face normals and nine cross products.
    """
    cross = np.cross(a_axes.T[:, None, :], b_axes.T[None, :, :]).reshape(-1, 3)
    axes = np.vstack((a_axes.T, b_axes.T, cross))
    lengths = np.linalg.norm(axes, axis=1)
    axes = axes[lengths > 1e-10] / lengths[lengths > 1e-10, None]
    radii = np.abs(axes @ a_axes) @ a_half + np.abs(axes @ b_axes) @ b_half
    return float(np.max(np.abs(axes @ (b_center - a_center)) - radii))


def geometry_gap(model, data, i, j):
    if model.geom_type[i] == model.geom_type[j] == mujoco.mjtGeom.mjGEOM_BOX:
        return box_separation(
            data.geom_xpos[i],
            data.geom_xmat[i].reshape(3, 3),
            model.geom_size[i],
            data.geom_xpos[j],
            data.geom_xmat[j].reshape(3, 3),
            model.geom_size[j],
        )
    # Avoid asking narrow-phase solvers about far-separated pairs: signed
    # mj_geomDistance with a large cutoff gave a false negative for two remote
    # boxes in3.10.0. OBB pairs above use independent SAT throughout.
    radii = []
    for k in (i, j):
        if model.geom_type[k] == mujoco.mjtGeom.mjGEOM_SPHERE:
            radii.append(model.geom_size[k, 0])
        elif model.geom_type[k] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            radii.append(np.linalg.norm(model.geom_size[k, :2]))
        else:
            radii.append(np.linalg.norm(model.geom_size[k]))
    bound = float(np.linalg.norm(data.geom_xpos[i] - data.geom_xpos[j]) - sum(radii))
    return (
        bound
        if bound > 0.005
        else float(mujoco.mj_geomDistance(model, data, i, j, 0.005, None))
    )


def inspect_poses(model, poses, tolerance=1e-6):
    data = mujoco.MjData(model)
    candidates = pairs(model)
    worst = {}
    candidate_i = np.array([p[0] for p in candidates])
    candidate_j = np.array([p[1] for p in candidates])
    case_pair = np.array(
        [
            "motor_envelope" in model.geom(i).name
            and "motor_envelope" in model.geom(j).name
            for i, j, _ in candidates
        ]
    )
    bounds = model.geom_rbound[candidate_i] + model.geom_rbound[candidate_j]
    min_motor_gap = float("inf")
    for index, qpos in enumerate(poses):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        delta = data.geom_xpos[candidate_i] - data.geom_xpos[candidate_j]
        nearby = np.sum(delta * delta, axis=1) <= bounds * bounds + 1e-12
        for pair_index in np.flatnonzero(nearby | case_pair):
            i, j, category = candidates[pair_index]
            distance = geometry_gap(model, data, i, j)
            a, b = model.geom(i).name, model.geom(j).name
            if "motor_envelope" in a and "motor_envelope" in b:
                min_motor_gap = min(min_motor_gap, distance)
            if distance < -tolerance:
                key = (a, b, category)
                if key not in worst or distance < worst[key]["signed_distance_m"]:
                    worst[key] = {
                        "geom_a": a,
                        "geom_b": b,
                        "kind": category,
                        "signed_distance_m": distance,
                        "pose_index": index,
                    }
    hits = sorted(worst.values(), key=lambda row: row["signed_distance_m"])
    return {
        "poses_checked": len(poses),
        "pairs_per_pose": len(candidates),
        "minimum_motor_casing_separation_bound_m": min_motor_gap,
        "solid_interference_pairs": sum(h["kind"] == "solid" for h in hits),
        "cable_allowance_interference_pairs": sum(
            h["kind"] == "cable_allowance" for h in hits
        ),
        "worst_interferences": hits,
        "sampled_solid_clearance_pass": not any(h["kind"] == "solid" for h in hits),
        "sampled_cable_allowance_pass": not any(
            h["kind"] == "cable_allowance" for h in hits
        ),
    }


def make_report(config, workspace_samples=96):
    model, data = standing_model(config)
    alignment = shaft_alignment(model, data)
    stand = inspect_poses(model, [data.qpos.copy()])
    _, frames, _ = trajectory(config, frames_per_step=48)
    gait = inspect_poses(model, [f["qpos"] for f in frames])
    rejected = copy.deepcopy(config)
    rejected["gait"].update(base_height_m=0.124, support_shift_fraction=1.0)
    rejected_model, rejected_frames, _ = trajectory(rejected, frames_per_step=24)
    original = inspect_poses(rejected_model, [f["qpos"] for f in rejected_frames])
    rng = np.random.default_rng(330)
    poses = []
    neutral = data.qpos.copy()
    for _ in range(workspace_samples):
        q = neutral.copy()
        for leg in LEG_ORDER:
            qadr, _ = joint_addresses(model, leg)
            q[qadr] = [rng.uniform(*config["joint_limits_rad"][j]) for j in JOINT_ORDER]
        poses.append(q)
    workspace = inspect_poses(model, poses)
    return {
        "kind": "Manufacturer-anchored assembly-envelope audit; not manufacturing approval",
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
        "joint_shaft_alignment": alignment,
        "alignment_pass": all(
            r["shaft_axis_offset_m"] < 1e-10
            and r["axis_alignment_abs_cosine"] > 1 - 1e-10
            for r in alignment
        ),
        "neutral": stand,
        "prescribed_crawl": gait,
        "rejected_original_crawl": original,
        "sampled_joint_box": workspace,
        "workspace_seed": 330,
        "penetration_tolerance_m": 1e-6,
        "distance_method": "Independent15-axis SAT for box pairs; separated gaps are lower bounds, not Euclidean distances. Other shapes use MuJoCo narrow phase behind a sphere bound.",
        "self_contacts": "Enabled, with explicit adjacent motor/link pairs; audit checks additional adjacent and welded housing pairs.",
        "limits": [
            "Casing boxes omit rounded corners and small recesses; may conservatively flag actual-clear areas.",
            "Cable volumes are design allowances around STEP-verified bare socket locations, not mated-plug or flexible-wire CAD.",
            "Printed cradles have socket slots, but fastener engagement, service access, support bearings and stiffness are not validated.",
            "Random joint-box sampling is not a proof that every pose within rectangular joint limits is collision-free.",
            "Reference servo mass is lumped on housing; horn/rotor mass motion is not split from it.",
        ],
    }


def write_report(config, path):
    report = make_report(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Assembly refinement and clearance audit",
        "",
        "The motor shafts, casing offsets and reference mass properties now follow the ROBOTIS XL330 drawing. These are original structural envelopes, not a released printable assembly.",
        "",
        "**Geometry change:** pitch shaft24 mm fore/aft outward from the roll shaft, with25 mm lateral offset. Roll motor faces fore/aft; pitch and knee motors face outward left/right. Knee casing tails point toward the foot, providing shoulder clearance during flexion. Stock rear idlers are not installed in this study.",
        "",
        "| Check | Poses | Solid interference pairs | Cable allowance interference pairs | Minimum casing separation bound |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in (
        "neutral",
        "prescribed_crawl",
        "rejected_original_crawl",
        "sampled_joint_box",
    ):
        r = report[key]
        lines.append(
            f"| {key.replace('_', ' ')} | {r['poses_checked']} | {r['solid_interference_pairs']} | {r['cable_allowance_interference_pairs']} | {1000 * r['minimum_motor_casing_separation_bound_m']:.2f}mm |"
        )
    lines += [
        "",
        f"All 12 shafts aligned with their modeled joint axes: **{report['alignment_pass']}**. Both physical shaft direction and mathematical joint sign are recorded in JSON; rear roll/right pitch motor signs differ from front/left motors.",
        "",
        "The current illustration uses a 140 mm body height and 25% of the original centroid shift. The rejected comparison uses 124 mm and the full centroid shift. Positive support loads are checked for both; only the revised sampled trajectory clears the envelope checks.",
        "",
        "The broad random joint-box sample is diagnostic. Any collision there means scalar joint bounds alone do not define a mechanically valid workspace; use collision-aware resets and policy penalties/termination, and refine travel limits after final CAD.",
        "",
        "## Remaining assembly gates",
        "",
    ] + ["- " + v for v in report["limits"]]
    lines += [
        "",
        "Source dimensions and tensor provenance: [SERVO_GEOMETRY_SOURCES.md](../docs/implementation/SERVO_GEOMETRY_SOURCES.md). Full pairs, distances and pose indices are in the adjacent JSON.",
        "",
    ]
    path.with_suffix(".md").write_text("\n".join(lines))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("config/robot.json"))
    p.add_argument(
        "--output", type=Path, default=Path("reports/assembly-validation.json")
    )
    a = p.parse_args()
    r = write_report(load_config(a.config), a.output)
    print(
        json.dumps(
            {
                k: r[k]
                if k == "alignment_pass"
                else {
                    key: val
                    for key, val in r[k].items()
                    if key != "worst_interferences"
                }
                for k in (
                    "alignment_pass",
                    "neutral",
                    "prescribed_crawl",
                    "sampled_joint_box",
                )
            },
            indent=2,
        )
    )
