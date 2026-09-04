"""Original assembly envelopes anchored to published XL330 shaft geometry.

Dimensions/COM/tensors come from reference/xl330/dimensions.json. Printed pieces
are original clearance-study solids, not released CAD: holes, screws, cable bend
radii, stiffness and support bearings still need design. Motor mass (including
horn) is lumped on its housing body; the visible child horn adds no second mass.
"""

from __future__ import annotations
from dataclasses import dataclass
import xml.etree.ElementTree as ET
import numpy as np
from .kinematics import LEG_ORDER


@dataclass
class Part:
    name: str
    pos: np.ndarray
    size: np.ndarray  # full box dimensions, or [radius, full cylinder length]
    kind: str = "box"
    axis: np.ndarray | None = None
    color: str = "0.20 0.47 0.61 1"
    collision: bool = True
    group: int = 0


def box(name, pos, size, color="0.20 0.47 0.61 1", collision=True):
    return Part(
        name,
        np.array(pos, float),
        np.array(size, float),
        color=color,
        collision=collision,
    )


def box_component(mass, center, size):
    x, y, z = size
    return (
        mass,
        np.array(center),
        mass / 12 * np.diag([y * y + z * z, x * x + z * z, x * x + y * y]),
    )


def structure_components(parts, mass):
    volumes = np.array([np.prod(p.size) for p in parts])
    return [
        box_component(mass * volume / volumes.sum(), p.pos, p.size)
        for p, volume in zip(parts, volumes)
    ]


def motor_frame(config, leg, role):
    """Source U,V,W axes -> parent coordinates, with W the outward shaft axis."""
    g, signs = config["geometry_m"], config["legs"][leg]
    f, s = signs["front"], signs["side"]
    if role == "roll":
        origin = np.array([f * g["hip_x"], s * g["hip_y"], g["hip_z"]])
        rotation = np.column_stack(([0, f, 0], [0, 0, 1], [f, 0, 0]))
    elif role == "hip":
        origin = np.array([f * g["hip_fore_aft_offset"], s * g["hip_offset"], 0])
        rotation = np.column_stack(([-s, 0, 0], [0, 0, 1], [0, s, 0]))
    elif role == "knee":
        origin = np.array([0, 0, -g["upper_length"]])
        rotation = np.column_stack(([-s, 0, 0], [0, 0, 1], [0, s, 0]))
    else:
        raise ValueError(role)
    return origin, rotation


def motor_parts(config, leg, role):
    spec = config["servo_reference"]
    origin, rotation = motor_frame(config, leg, role)
    case = box(
        f"{leg}_{role}_motor_envelope",
        origin + rotation @ np.array(spec["case_center_m"]),
        np.abs(rotation) @ np.array(spec["case_size_m"]),
        "0.12 0.14 0.16 1",
    )
    return case


def cable_keepouts(config, leg, role):
    """Space allowances around two known socket locations, not mated-plug CAD."""
    origin, rotation = motor_frame(config, leg, role)
    result = []
    for sign in (-1, 1):
        p = box(
            f"{leg}_{role}_port_keepout_{'a' if sign < 0 else 'b'}",
            origin + rotation @ np.array([sign * 0.015, -0.009, -0.0185]),
            np.abs(rotation) @ np.array([0.010, 0.012, 0.013]),
            "0.65 0.35 0.73 0.25",
            False,
        )
        p.group = 5
        result.append(p)
    return result


def servo_component(config, leg, role):
    spec = config["servo_reference"]
    origin, rotation = motor_frame(config, leg, role)
    mass = config["mass_kg"]["servo"]
    return (
        mass,
        origin + rotation @ np.array(spec["center_of_mass_m"]),
        rotation
        @ np.array(spec["inertia_kg_m2"])
        @ rotation.T
        * mass
        / spec["mass_kg"],
    )


def horn_part(config, leg, role):
    # All child body origins are at the corresponding horn front face.
    _, rotation = motor_frame(config, leg, role)
    axis = rotation[:, 2]
    horn = config["servo_reference"]["stock_horn"]
    return Part(
        f"{leg}_{role}_output_horn",
        -axis * horn["depth_m"] / 2,
        np.array([horn["radius_m"], horn["depth_m"]]),
        "cylinder",
        axis,
        "0.72 0.75 0.77 1",
    )


def chassis_parts(config):
    g = config["geometry_m"]
    length, width, height = [g[k] for k in ("body_length", "body_width", "body_height")]
    center_z, thickness = -0.0075, 0.003
    parts = [
        box(
            "chassis_top",
            [0, 0, center_z + (height - thickness) / 2],
            [length, width, thickness],
        ),
        box(
            "chassis_bottom",
            [0, 0, center_z - (height - thickness) / 2],
            [length, width, thickness],
        ),
    ]
    for side, name in ((1, "left"), (-1, "right")):
        parts.append(
            box(
                f"chassis_{name}_center_rail",
                [0, side * 0.021, center_z],
                [0.062, 0.003, height - 0.006],
            )
        )
        for front, end in ((1, "front"), (-1, "rear")):
            parts.append(
                box(
                    f"chassis_{name}_{end}_post",
                    [front * (length / 2 - 0.0035), side * 0.021, center_z],
                    [0.007, 0.003, height - 0.006],
                )
            )
    return parts


def roll_parts(config, leg):
    """Cradle cheek with an open slot for the inward pitch-motor socket.

    Physical bounds are mirrored in x/y. The original bridge passes above the
    socket allowance. Positive-x dimensions below mean fore/aft outward.
    """
    g, signs = config["geometry_m"], config["legs"][leg]
    f, s = signs["front"], signs["side"]
    cheek_x = g["hip_fore_aft_offset"] - 0.0115
    lateral = g["hip_offset"]
    parts = []

    def add(name, x, y, z, dx, dy, dz):
        parts.append(
            box(
                f"{leg}_roll_{name}",
                [f * x, s * y, z],
                [dx, dy, dz],
                "0.82 0.46 0.15 1",
            )
        )

    add("horn_spacer", 0.0015, 0, 0, 0.003, 0.016, 0.016)
    add(
        "bridge",
        (0.003 + cheek_x - 0.0015) / 2,
        0,
        0.006,
        cheek_x - 0.0045,
        0.006,
        0.006,
    )
    # Three solid strips form a socket opening, instead of a solid plate
    # blocking the manufacturer +/-U side port.
    add("cheek_top", cheek_x, lateral - 0.0145, 0.00375, 0.003, 0.023, 0.0115)
    add("cheek_bottom", cheek_x, lateral - 0.0145, -0.02025, 0.003, 0.023, 0.0085)
    add("cheek_front", cheek_x, lateral - 0.007, -0.009, 0.003, 0.008, 0.014)
    return parts


def upper_parts(config, leg):
    """Horn-mounted thigh and slotted knee cheek; shank sweeps on the other side."""
    upper, s = config["geometry_m"]["upper_length"], config["legs"][leg]["side"]
    length = upper - 0.015
    return [
        box(f"{leg}_upper_bar", [0, s * 0.0015, -length / 2], [0.008, 0.003, length]),
        box(
            f"{leg}_upper_bridge",
            [-0.00575, s * 0.0015, -upper + 0.015],
            [0.0115, 0.003, 0.003],
        ),
        box(
            f"{leg}_knee_cheek_top",
            [-0.0115, -s * 0.0115, -upper + 0.007],
            [0.003, 0.029, 0.020],
        ),
        box(
            f"{leg}_knee_cheek_bottom",
            [-0.0115, -s * 0.0115, -upper - 0.0205],
            [0.003, 0.029, 0.011],
        ),
        box(
            f"{leg}_knee_cheek_front",
            [-0.0115, -s * 0.004, -upper - 0.009],
            [0.003, 0.014, 0.012],
        ),
    ]


def lower_parts(config, leg):
    lower, s = config["geometry_m"]["lower_length"], config["legs"][leg]["side"]
    return [box(f"{leg}_lower_bar", [0, s * 0.0015, -lower / 2], [0.008, 0.003, lower])]


def add_parts(body, parts):
    for p in parts:
        attrs = dict(
            name=p.name,
            type=p.kind,
            pos=" ".join(map(str, p.pos)),
            rgba=p.color,
            group=str(p.group),
        )
        if p.kind == "box":
            attrs["size"] = " ".join(map(str, p.size / 2))
        else:
            # Cylinder axis is MuJoCo local z; fromto makes this unambiguous.
            attrs.pop("pos")
            attrs["fromto"] = " ".join(
                map(
                    str,
                    np.r_[
                        p.pos - p.axis * p.size[1] / 2, p.pos + p.axis * p.size[1] / 2
                    ],
                )
            )
            attrs["size"] = str(p.size[0])
        if not p.collision:
            attrs.update(contype="0", conaffinity="0")
        ET.SubElement(body, "geom", **attrs)
