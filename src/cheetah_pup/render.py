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
    return [[points[i] for i in face] for face in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))]


def _round(radius, half_length=0):
    rings = []
    for angle in np.linspace(-np.pi / 2, np.pi / 2, 14):
        z = radius * np.sin(angle) + np.sign(angle) * half_length
        rings.append(np.array([[radius * np.cos(angle) * np.cos(phi), radius * np.cos(angle) * np.sin(phi), z] for phi in np.linspace(0, 2 * np.pi, 20, endpoint=False)]))
    return [[rings[i][j], rings[i][(j + 1) % 20], rings[i + 1][(j + 1) % 20], rings[i + 1][j]] for i in range(len(rings) - 1) for j in range(20)]


def render_software(config: dict, output: Path):
    model, data = standing_model(config)
    fig = plt.figure(figsize=(10.8, 7.2), dpi=120, facecolor="#f4f3ef")
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.set_facecolor("#f4f3ef")
    floor = [[[-.13, -.12, 0], [.13, -.12, 0], [.13, .12, 0], [-.13, .12, 0]]]
    ax.add_collection3d(Poly3DCollection(floor, facecolors="#e2e4df", edgecolors="#c7ccc7", linewidths=.5, zorder=0))
    all_faces, all_colors = [], []
    for index in range(model.ngeom):
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
        else:
            continue
        transform = data.geom_xmat[index].reshape(3, 3)
        transformed = [np.array(face) @ transform.T + data.geom_xpos[index] for face in faces]
        color = model.geom_rgba[index].copy()
        color[3] = 1.0
        all_faces.extend(transformed)
        all_colors.extend([color] * len(transformed))
    ax.add_collection3d(Poly3DCollection(all_faces, facecolors=all_colors, edgecolors=(.08, .13, .16, .20), linewidths=.25, shade=True, zorder=1))
    ax.set_xlim(-.13, .13)
    ax.set_ylim(-.13, .13)
    ax.set_zlim(0, .20)
    ax.set_box_aspect((.26, .26, .20), zoom=1.25)
    ax.set_proj_type("ortho")
    ax.view_init(elev=20, azim=-48)
    ax.set_axis_off()
    fig.text(.07, .92, "CHEETAH PUP", fontsize=22, weight="bold", color="#173447")
    fig.text(.07, .87, f"First primitive simulation · 12 joints · {total_mass(config)*1000:.0f} g estimated", fontsize=12, color="#465961")
    geometry = config["geometry_m"]
    dims = " × ".join(f"{geometry[k]*1000:.0f}" for k in ("body_length", "body_width", "body_height"))
    fig.text(.07, .09, f"Torso {dims} mm · upper/lower links {geometry['upper_length']*1000:.0f} / {geometry['lower_length']*1000:.0f} mm", fontsize=11, color="#465961")
    fig.text(.07, .05, "Neutral pose. Motor envelopes shown; packaging and actuator physics remain unvalidated.", fontsize=9, color="#66747a")
    fig.subplots_adjust(left=0, right=1, bottom=.11, top=.83)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
