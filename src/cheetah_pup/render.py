"""Software-render MuJoCo's compiled primitive geometry for inspection.

Uses a static Matplotlib projection, so no system EGL/OpenGL installation is
needed. This image shows the neutral keyframe, not a locomotion result.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mujoco
import numpy as np

from .analysis import standing_model
from .model import total_mass


def _box(size):
    points = np.array(list(itertools.product((-1, 1), repeat=3))) * size
    return [
        [points[i] for i in face]
        for face in (
            (0, 1, 3, 2),
            (4, 6, 7, 5),
            (0, 4, 5, 1),
            (2, 3, 7, 6),
            (0, 2, 6, 4),
            (1, 5, 7, 3),
        )
    ]


def _round(radius, half_length=0):
    rings = []
    for angle in np.linspace(-np.pi / 2, np.pi / 2, 14):
        z = radius * np.sin(angle) + np.sign(angle) * half_length
        rings.append(
            np.array(
                [
                    [
                        radius * np.cos(angle) * np.cos(phi),
                        radius * np.cos(angle) * np.sin(phi),
                        z,
                    ]
                    for phi in np.linspace(0, 2 * np.pi, 20, endpoint=False)
                ]
            )
        )
    return [
        [
            rings[i][j],
            rings[i][(j + 1) % 20],
            rings[i + 1][(j + 1) % 20],
            rings[i + 1][j],
        ]
        for i in range(len(rings) - 1)
        for j in range(20)
    ]


def _cylinder(radius, half_length):
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    bottom = np.column_stack(
        (radius * np.cos(angles), radius * np.sin(angles), np.full(24, -half_length))
    )
    top = bottom.copy()
    top[:, 2] = half_length
    return [bottom[::-1], top] + [
        [bottom[i], bottom[(i + 1) % 24], top[(i + 1) % 24], top[i]] for i in range(24)
    ]


def geom_faces(model, index):
    kind, size = model.geom_type[index], model.geom_size[index]
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        return _box(size)
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return _round(size[0])
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return _round(size[0], size[1])
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return _cylinder(size[0], size[1])
    return []


def add_robot(ax, model, data, indices=None, offset=None):
    faces, colors = [], []
    if indices is None:
        indices = range(model.ngeom)
    if offset is None:
        offset = np.zeros(3)
    for index in indices:
        if model.geom_group[index] == 5:
            continue
        local = geom_faces(model, index)
        transform = data.geom_xmat[index].reshape(3, 3)
        faces.extend(
            [
                np.array(face) @ transform.T + data.geom_xpos[index] - offset
                for face in local
            ]
        )
        colors.extend([model.geom_rgba[index]] * len(local))
    collection = Poly3DCollection(
        faces,
        facecolors=colors,
        edgecolors=(0.08, 0.13, 0.16, 0.25),
        linewidths=0.25,
        shade=True,
    )
    ax.add_collection3d(collection)
    return collection


def render_assembly_review(config, output):
    model, data = standing_model(config)
    fig = plt.figure(figsize=(13, 7.5), dpi=130, facecolor="#f4f3ef")
    main = fig.add_axes([0.02, 0.16, 0.56, 0.66], projection="3d")
    hip = fig.add_axes([0.61, 0.2, 0.37, 0.60], projection="3d")
    for ax in (main, hip):
        ax.set_facecolor("#f4f3ef")
        ax.set_proj_type("ortho")
        ax.set_axis_off()
    add_robot(main, model, data)
    main.set_xlim(-0.115, 0.115)
    main.set_ylim(-0.095, 0.095)
    main.set_zlim(0, 0.18)
    main.set_box_aspect((0.23, 0.19, 0.18), zoom=1.12)
    main.view_init(elev=25, azim=-48)
    # One shoulder isolated from the rest of the frame. No exploded movement:
    # the transform and pose are exactly those in the assembled model.
    indices = [i for i in range(model.ngeom) if model.geom(i).name.startswith("FL_")]
    add_robot(hip, model, data, indices)
    for joint, color, label in [
        ("hip_roll", "#be5548", "Roll · fore/aft shaft"),
        ("hip_pitch", "#217b9d", "Hip pitch · outward shaft"),
    ]:
        jid = model.joint("FL_" + joint).id
        o = data.xanchor[jid]
        direction = data.xaxis[jid]
        line = np.array([o - direction * 0.017, o + direction * 0.021])
        hip.plot(*line.T, color=color, lw=2.6, zorder=5)
        where = o + direction * 0.026
        hip.text(*where, label, color=color, fontsize=9, zorder=6)
    hip.set_xlim(0.025, 0.11)
    hip.set_ylim(0.018, 0.085)
    hip.set_zlim(0.080, 0.16)
    hip.set_box_aspect((0.085, 0.067, 0.080), zoom=1.2)
    hip.view_init(elev=22, azim=42)
    fig.text(
        0.045,
        0.92,
        "CHEETAH PUP · ASSEMBLY STUDY",
        fontsize=22,
        weight="bold",
        color="#173447",
    )
    fig.text(
        0.045,
        0.865,
        "Shaft-anchored XL330 housings, horns and slotted cradles",
        fontsize=13,
        color="#426071",
    )
    fig.text(
        0.66, 0.825, "Front-left shoulder · same assembly", fontsize=12, color="#173447"
    )
    fig.text(
        0.045,
        0.12,
        "Black: servo casing   Silver: output horn   Orange: roll cradle   Blue: frame and legs",
        fontsize=11,
        color="#426071",
    )
    fig.text(
        0.045,
        0.075,
        "23 mm casing + 3 mm horn · pitch shaft offset 24 mm fore/aft and 25 mm laterally",
        fontsize=11,
        color="#426071",
    )
    fig.text(
        0.045,
        0.03,
        "Original clearance-study solids. Fasteners, bearing support and flexible cable routing still need detailed CAD.",
        fontsize=10,
        color="#69777b",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_software(config: dict, output: Path):
    model, data = standing_model(config)
    fig = plt.figure(figsize=(10.8, 7.2), dpi=120, facecolor="#f4f3ef")
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.set_facecolor("#f4f3ef")
    floor = [[[-0.13, -0.12, 0], [0.13, -0.12, 0], [0.13, 0.12, 0], [-0.13, 0.12, 0]]]
    ax.add_collection3d(
        Poly3DCollection(
            floor, facecolors="#e2e4df", edgecolors="#c7ccc7", linewidths=0.5, zorder=0
        )
    )
    all_faces, all_colors = [], []
    for index in range(model.ngeom):
        if model.geom_group[index] == 5:
            continue
        kind = model.geom_type[index]
        size = model.geom_size[index]
        if kind == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            faces = _box(size)
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            faces = _round(size[0])
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            faces = _round(size[0], size[1])
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            faces = _cylinder(size[0], size[1])
        else:
            continue
        transform = data.geom_xmat[index].reshape(3, 3)
        transformed = [
            np.array(face) @ transform.T + data.geom_xpos[index] for face in faces
        ]
        color = model.geom_rgba[index].copy()
        color[3] = 1.0
        all_faces.extend(transformed)
        all_colors.extend([color] * len(transformed))
    ax.add_collection3d(
        Poly3DCollection(
            all_faces,
            facecolors=all_colors,
            edgecolors=(0.08, 0.13, 0.16, 0.20),
            linewidths=0.25,
            shade=True,
            zorder=1,
        )
    )
    ax.set_xlim(-0.13, 0.13)
    ax.set_ylim(-0.13, 0.13)
    ax.set_zlim(0, 0.20)
    ax.set_box_aspect((0.26, 0.26, 0.20), zoom=1.25)
    ax.set_proj_type("ortho")
    ax.view_init(elev=20, azim=-48)
    ax.set_axis_off()
    fig.text(0.07, 0.92, "CHEETAH PUP", fontsize=22, weight="bold", color="#173447")
    fig.text(
        0.07,
        0.87,
        f"Refined assembly study · 12 joints · {total_mass(config) * 1000:.0f} g estimated",
        fontsize=12,
        color="#465961",
    )
    geometry = config["geometry_m"]
    dims = " × ".join(
        f"{geometry[k] * 1000:.0f}"
        for k in ("body_length", "body_width", "body_height")
    )
    fig.text(
        0.07,
        0.09,
        f"Torso {dims} mm · upper/lower links {geometry['upper_length'] * 1000:.0f} / {geometry['lower_length'] * 1000:.0f} mm",
        fontsize=11,
        color="#465961",
    )
    fig.text(
        0.07,
        0.05,
        "Neutral pose. Manufacturer shaft offsets and mass properties. Envelopes are not manufacturing CAD.",
        fontsize=9,
        color="#66747a",
    )
    fig.subplots_adjust(left=0, right=1, bottom=0.11, top=0.83)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
