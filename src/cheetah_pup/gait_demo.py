"""Animate a prescribed crawl. Kinematic illustration, not RL or dynamics.

Stance feet stay fixed in world coordinates. Each swing is preceded by a body
shift toward the three support feet's centroid, accounting for modeled COM.
Run: uv run python -m cheetah_pup.gait_demo --frames /tmp/cheetah-gait-frames
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mujoco
import numpy as np

from .analysis import joint_addresses, standing_model, solve_vertical_support
from .kinematics import LEG_ORDER, JOINT_ORDER, foot_position
from .model import load_config
from .render import _box, _cylinder


def inverse_leg(config, leg, target):
    g, signs = config["geometry_m"], config["legs"][leg]
    x, y, z = np.asarray(target) - [
        signs["front"] * (g["hip_x"] + g.get("hip_fore_aft_offset", 0)),
        signs["side"] * g["hip_y"],
        g["hip_z"],
    ]
    lateral = signs["side"] * g["hip_offset"]
    if y * y + z * z <= lateral * lateral:
        raise ValueError("Target inside hip-offset exclusion cylinder")
    roll = np.arctan2(z, y) + np.arccos(lateral / np.hypot(y, z))
    planar_z = -np.sqrt(y * y + z * z - lateral * lateral)
    upper, lower = g["upper_length"], g["lower_length"]
    cosine = (x * x + planar_z * planar_z - upper * upper - lower * lower) / (
        2 * upper * lower
    )
    if abs(cosine) > 1 + 1e-10:
        raise ValueError(f"Unreachable foot target for {leg}")
    knee = -np.arccos(np.clip(cosine, -1, 1))
    hip = np.arctan2(-x, -planar_z) - np.arctan2(
        lower * np.sin(knee), upper + lower * np.cos(knee)
    )
    q = np.array([roll, hip, knee])
    limits = np.array([config["joint_limits_rad"][joint] for joint in JOINT_ORDER])
    if np.any(q < limits[:, 0] - 1e-10) or np.any(q > limits[:, 1] + 1e-10):
        raise ValueError(f"Joint limit exceeded for {leg}: {q}")
    return q


def set_pose(config, model, data, base, feet):
    data.qpos[:3] = base
    data.qpos[3:7] = [1, 0, 0, 0]
    for leg in LEG_ORDER:
        qadr, _ = joint_addresses(model, leg)
        data.qpos[qadr] = inverse_leg(config, leg, feet[leg] - base)
    mujoco.mj_forward(model, data)


def smooth(t):
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def trajectory(config, frames_per_step=24, stride=0.020, clearance=0.012):
    model, data = standing_model(config)
    feet = {leg: data.site(f"{leg}_foot").xpos.copy() for leg in LEG_ORDER}
    base = data.qpos[:3].copy()
    base_height = config.get("gait", {}).get("base_height_m", 0.124)
    base[2] = base_height
    sequence = ("RR", "FR", "RL", "FL")
    frames = []
    max_error = 0.0
    min_swing_margin = float("inf")
    # First cycle warms up the periodic footprint/body-shift pattern.
    for cycle in range(2):
        for step, moving in enumerate(sequence):
            supports = tuple(leg for leg in LEG_ORDER if leg != moving)
            centroid = np.mean([feet[leg][:2] for leg in supports], axis=0)
            center = np.mean([xyz[:2] for xyz in feet.values()], axis=0)
            shift_fraction = config.get("gait", {}).get("support_shift_fraction", 1.0)
            centroid = center + shift_fraction * (centroid - center)
            shifted = base.copy()
            shifted[:2] = centroid
            for _ in range(8):
                set_pose(config, model, data, shifted, feet)
                shifted[:2] += centroid - data.subtree_com[model.body("base").id, :2]
            start_base = base.copy()
            start_foot = feet[moving].copy()
            for frame in range(frames_per_step):
                phase = frame / frames_per_step
                current_feet = {leg: xyz.copy() for leg, xyz in feet.items()}
                if phase < 0.375:
                    base_now = start_base + smooth(phase / 0.375) * (
                        shifted - start_base
                    )
                    active = LEG_ORDER
                    label = "Shift weight · all four feet down"
                else:
                    swing = (phase - 0.375) / 0.625
                    base_now = shifted.copy()
                    current_feet[moving][0] += stride * smooth(swing)
                    current_feet[moving][2] += clearance * np.sin(np.pi * swing) ** 2
                    active = supports
                    label = f"Step {moving} · three feet support"
                set_pose(config, model, data, base_now, current_feet)
                for leg in LEG_ORDER:
                    max_error = max(
                        max_error,
                        float(
                            np.max(
                                np.abs(
                                    data.site(f"{leg}_foot").xpos - current_feet[leg]
                                )
                            )
                        ),
                    )
                if len(active) == 3:
                    com = data.subtree_com[model.body("base").id, :2]
                    triangle = np.array([current_feet[leg][:2] for leg in active])
                    loads = solve_vertical_support(triangle, com, 1.0)
                    if loads is None:
                        raise ValueError("COM leaves support triangle during swing")
                    min_swing_margin = min(min_swing_margin, float(np.min(loads)))
                if cycle == 1:
                    frames.append(
                        {
                            "qpos": data.qpos.copy(),
                            "active": active,
                            "moving": moving,
                            "label": label,
                            "progress": (step + phase) / 4,
                            "view_offset": stride * (cycle + (step + phase) / 4),
                        }
                    )
            feet[moving][0] += stride
            base = shifted.copy()
    return (
        model,
        frames,
        {
            "kind": "prescribed kinematic crawl; not learned or dynamically validated",
            "frames": len(frames),
            "stride_per_cycle_m": stride,
            "playback_fps": 15,
            "cycle_duration_s": len(frames) / 15,
            "foot_clearance_m": clearance,
            "base_height_m": base_height,
            "support_shift_fraction": shift_fraction,
            "max_ik_foot_error_m": max_error,
            "minimum_swing_support_load_fraction": min_swing_margin,
            "joint_limits_checked": True,
            "stance_feet_world_locked": True,
            "torque_speed_thermal_validation": False,
            "notes": "Body shifts use the modeled COM; support fractions check vertical static balance only. No contact impulses, actuator dynamics, or terrain traversal are simulated.",
        },
    )


def round_faces(radius, half=0):
    rings = []
    for angle in np.linspace(-np.pi / 2, np.pi / 2, 8):
        z = radius * np.sin(angle) + np.sign(angle) * half
        rings.append(
            [
                [
                    radius * np.cos(angle) * np.cos(phi),
                    radius * np.cos(angle) * np.sin(phi),
                    z,
                ]
                for phi in np.linspace(0, 2 * np.pi, 12, endpoint=False)
            ]
        )
    return [
        [
            rings[i][j],
            rings[i][(j + 1) % 12],
            rings[i + 1][(j + 1) % 12],
            rings[i + 1][j],
        ]
        for i in range(7)
        for j in range(12)
    ]


def render_frames(config, output):
    model, frames, report = trajectory(config)
    data = mujoco.MjData(model)
    local_faces, colors, indices = [], [], []
    for index in range(model.ngeom):
        if model.geom_group[index] == 5:
            continue
        kind, size = model.geom_type[index], model.geom_size[index]
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            faces = _box(size)
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            faces = round_faces(size[0], size[1])
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            faces = round_faces(size[0])
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            faces = _cylinder(size[0], size[1])
        else:
            continue
        faces = [
            [face[0], face[j], face[j + 1]]
            for face in faces
            for j in range(1, len(face) - 1)
        ]
        local_faces.append(np.asarray(faces))
        color = model.geom_rgba[index].copy()
        color[3] = 1
        colors.extend([color] * len(faces))
        indices.append(index)
    output.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 6), dpi=100, facecolor="#f4f3ef")
    ax = fig.add_axes([0.01, 0.14, 0.67, 0.70], projection="3d", computed_zorder=False)
    ax.set_facecolor("#f4f3ef")
    ax.set_axis_off()
    ax.set_xlim(-0.14, 0.14)
    ax.set_ylim(-0.14, 0.14)
    ax.set_zlim(0, 0.21)
    ax.set_box_aspect((0.28, 0.28, 0.21), zoom=1.3)
    ax.set_proj_type("ortho")
    ax.view_init(elev=23, azim=-46)
    robot = Poly3DCollection(
        [],
        facecolors=colors,
        edgecolors=(0.08, 0.13, 0.16, 0.15),
        linewidths=0.2,
        zorder=3,
    )
    ax.add_collection3d(robot)
    floor = Poly3DCollection(
        [[[-0.16, -0.14, 0], [0.16, -0.14, 0], [0.16, 0.14, 0], [-0.16, 0.14, 0]]],
        facecolors="#e1e5e0",
        edgecolors="#c2cbc3",
        linewidths=0.5,
        zorder=0,
    )
    ax.add_collection3d(floor)
    top = fig.add_axes([0.72, 0.26, 0.24, 0.43])
    fig.text(
        0.06,
        0.91,
        "CHEETAH PUP · SLOW CRAWL",
        fontsize=19,
        weight="bold",
        color="#173447",
    )
    phase_text = fig.text(0.06, 0.85, "", fontsize=12, color="#426071")
    fig.text(
        0.06,
        0.08,
        "Planned gait · inverse kinematics · 12 mm foot lift",
        fontsize=11,
        color="#426071",
    )
    fig.text(
        0.06,
        0.035,
        "Illustration only: not a trained policy or validated motor/contact simulation.",
        fontsize=10,
        color="#69777b",
    )
    fig.text(0.73, 0.18, "● Planted", fontsize=9, color="#398360")
    fig.text(0.85, 0.18, "● Swing", fontsize=9, color="#d58b23")
    fig.text(0.76, 0.145, "× Center of mass", fontsize=9, color="#bd5547")
    for n, frame in enumerate(frames):
        data.qpos[:] = frame["qpos"]
        mujoco.mj_forward(model, data)
        offset = np.array([frame["view_offset"], 0, 0])
        verts = []
        for index, faces in zip(indices, local_faces):
            verts.extend(
                faces @ data.geom_xmat[index].reshape(3, 3).T
                + data.geom_xpos[index]
                - offset
            )
        robot.set_verts(verts)
        phase_text.set_text(frame["label"])
        top.clear()
        top.set_facecolor("#f4f3ef")
        pts = {leg: data.site(f"{leg}_foot").xpos - offset for leg in LEG_ORDER}
        active = np.array([pts[leg] for leg in frame["active"]])
        center = active[:, :2].mean(axis=0)
        ordered = active[
            np.argsort(np.arctan2(active[:, 1] - center[1], active[:, 0] - center[0]))
        ]
        top.fill(
            ordered[:, 1] * 1000,
            ordered[:, 0] * 1000,
            color="#dce9df",
            ec="#729781",
            lw=1.2,
        )
        for leg, xyz in pts.items():
            planted = leg in frame["active"]
            top.scatter(
                xyz[1] * 1000,
                xyz[0] * 1000,
                s=70,
                c="#398360" if planted else "#e69a32",
                zorder=4,
            )
            top.text(
                xyz[1] * 1000 + 6, xyz[0] * 1000 + 5, leg, fontsize=9, color="#334c58"
            )
        com = data.subtree_com[model.body("base").id] - offset
        top.scatter(
            com[1] * 1000,
            com[0] * 1000,
            s=60,
            c="#bd5547",
            marker="x",
            zorder=5,
            linewidths=2,
        )
        top.set_xlim(95, -95)
        top.set_ylim(-100, 100)
        top.set_aspect("equal")
        top.set_axis_off()
        top.set_title(
            "Support from above\n↑ Forward", fontsize=11, color="#334c58", pad=9
        )
        fig.savefig(output / f"{n:04d}.png", dpi=100, facecolor=fig.get_facecolor())
        if n % 24 == 0:
            print(f"Rendered {n}/{len(frames)}", flush=True)
    plt.close(fig)
    Path("reports/gait-demo-validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, default=Path("/tmp/cheetah-gait-frames"))
    args = parser.parse_args()
    render_frames(load_config("config/robot.json"), args.frames)
