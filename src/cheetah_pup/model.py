"""Generate an original 12-DOF MJCF for kinematics and static load screening.

The actuator is deliberately an ideal, limited PD controller. It is not the
published BAM fit, does not model XL330 thermal limits or voltage behavior, and
must not be presented as a validated sim-to-real training environment.
"""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .kinematics import JOINT_ORDER, LEG_ORDER, foot_position


def load_config(path: str | Path) -> dict:
    """Load and validate the explicit project JSON configuration."""
    with Path(path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    _validate_config(config)
    return config


def total_mass(config: dict) -> float:
    """Return the mass budget in kg, counting all twelve motor assemblies."""
    return float(sum(config["mass_kg"][name] * count
                     for name, count in config["mass_counts"].items()))


def _validate_config(config: dict) -> None:
    geometry = config["geometry_m"]
    for name in ("body_length", "body_width", "body_height", "hip_x", "hip_y",
                 "hip_offset", "upper_length", "lower_length", "foot_radius", "link_radius"):
        if not np.isfinite(geometry[name]) or geometry[name] <= 0:
            raise ValueError(f"geometry_m.{name} must be finite and positive.")
    envelope = np.asarray(geometry["servo_envelope"], dtype=float)
    if envelope.shape != (3,) or not np.all(np.isfinite(envelope)) or np.any(envelope <= 0):
        raise ValueError("servo_envelope must have three finite positive dimensions.")
    for name, mass in config["mass_kg"].items():
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError(f"mass_kg.{name} must be finite and positive.")
    # This model has a fixed component topology; counts cannot silently change
    # the reported budget without changing the generated inertial bodies.
    expected_counts = {"servo": 12, "body_shell": 1, "battery": 1, "compute": 1,
                       "electronics": 1, "wiring_and_fasteners": 1,
                       "roll_bracket": 4, "upper_link": 4, "lower_link": 4, "foot": 4}
    if config["mass_counts"] != expected_counts:
        raise ValueError("mass_counts must match the implemented 12-motor component topology.")
    q = np.asarray(config["home_q_rad"], dtype=float)
    if q.shape != (3,) or not np.all(np.isfinite(q)):
        raise ValueError("home_q_rad must contain three finite angles.")
    for i, name in enumerate(JOINT_ORDER):
        bounds = np.asarray(config["joint_limits_rad"][name], dtype=float)
        if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
            raise ValueError(f"Invalid {name} joint limits.")
        if not bounds[0] <= q[i] <= bounds[1]:
            raise ValueError(f"Home pose violates {name} limits.")
    for leg_name, front, side in (("FL", 1, 1), ("FR", 1, -1),
                                  ("RL", -1, 1), ("RR", -1, -1)):
        if config["legs"][leg_name] != {"front": front, "side": side}:
            raise ValueError(f"{leg_name} must preserve the documented frame convention.")
    actuator = config["actuator"]
    for name in ("torque_limit_nm", "stall_torque_nm", "kp_nm_per_rad", "kv_nm_s_per_rad"):
        if not np.isfinite(actuator[name]) or actuator[name] <= 0:
            raise ValueError(f"actuator.{name} must be finite and positive.")
    if actuator["torque_limit_nm"] > actuator["stall_torque_nm"]:
        raise ValueError("Screening torque limit cannot exceed the stated stall limit.")


def _numbers(values) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def _box_component(mass, center, dimensions):
    x, y, z = np.asarray(dimensions, dtype=float)
    inertia = mass / 12 * np.diag([y * y + z * z, x * x + z * z, x * x + y * y])
    return mass, np.asarray(center, dtype=float), inertia


def _sphere_component(mass, center, radius):
    return mass, np.asarray(center, dtype=float), np.eye(3) * (0.4 * mass * radius**2)


def _inertial(body, components):
    """Combine envelopes at their collective COM using the parallel-axis rule."""
    mass = sum(part[0] for part in components)
    center = sum(part[0] * part[1] for part in components) / mass
    inertia = np.zeros((3, 3))
    for part_mass, part_center, part_inertia in components:
        offset = part_center - center
        inertia += part_inertia + part_mass * (
            np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset))
    if np.any(np.linalg.eigvalsh(inertia) <= 0):
        raise ValueError("Component aggregation produced a non-positive inertia tensor.")
    full = [inertia[0, 0], inertia[1, 1], inertia[2, 2],
            inertia[0, 1], inertia[0, 2], inertia[1, 2]]
    ET.SubElement(body, "inertial", mass=f"{mass:.12g}", pos=_numbers(center),
                  fullinertia=_numbers(full))


def build_mjcf(config: dict, terrain: str = "flat") -> str:
    """Build MJCF with free base, 12 hinges, terrain contacts and stand keyframe.

    Terrain may be flat, threshold, or carpet (a rigid friction placeholder).
    Joint/site names are FL_hip_roll, FL_hip_pitch, FL_knee, FL_foot, etc.
    Leg ordering is FL, FR, RL, RR. Self-collision is disabled; motor housings
    are visible at actual component envelope dimensions, but do not collide.
    """
    _validate_config(config)
    if terrain not in ("flat", "threshold", "carpet"):
        raise ValueError("terrain must be 'flat', 'threshold', or 'carpet'.")
    g, m, a, sim = (config[key] for key in
                    ("geometry_m", "mass_kg", "actuator", "simulation"))
    home = np.asarray(config["home_q_rad"], dtype=float)
    sole_heights = [foot_position(config, leg, home)[2] - g["foot_radius"]
                    for leg in LEG_ORDER]
    if np.ptp(sole_heights) > 1e-9:
        raise ValueError("The stand keyframe requires a symmetric home pose with coplanar feet.")
    base_height = -min(sole_heights)
    if base_height <= g["body_height"] / 2:
        raise ValueError("Home pose places the torso on or below the floor.")
    root = ET.Element("mujoco", model=config["name"])
    root.append(ET.Comment(" Original primitive geometry. Ideal PD only; no BAM or hardware validation. "))
    ET.SubElement(root, "compiler", angle="radian", autolimits="true", inertiafromgeom="false")
    ET.SubElement(root, "option", timestep=str(sim["timestep_s"]), integrator="implicitfast",
                  gravity=f"0 0 {-sim['gravity_m_s2']}")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", azimuth="135", elevation="-25")
    ET.SubElement(visual, "headlight", diffuse="0.8 0.8 0.8", ambient="0.3 0.3 0.3")
    defaults = ET.SubElement(root, "default")
    ET.SubElement(defaults, "joint", type="hinge", limited="true",
                  damping=str(a["joint_damping_nm_s_per_rad"]),
                  armature=str(a["joint_armature_kg_m2"]))
    # Robot bits contact terrain (type 2); they do not contact one another.
    ET.SubElement(defaults, "geom", contype="1", conaffinity="2", condim="3",
                  friction=_numbers(sim["ground_friction"]))
    world = ET.SubElement(root, "worldbody")
    friction = sim["carpet_placeholder_friction"] if terrain == "carpet" else sim["ground_friction"]
    ET.SubElement(world, "geom", name="ground", type="plane", size="2 2 0.1",
                  contype="2", conaffinity="1", friction=_numbers(friction),
                  rgba="0.83 0.85 0.86 1")
    ET.SubElement(world, "light", pos="0 -0.4 1", dir="0 0 -1", directional="true")
    if terrain == "threshold":
        height, depth = sim["threshold_height_m"], sim["threshold_depth_m"]
        center_x = sim["threshold_center_x_m"]
        if height <= 0 or depth <= 0 or center_x - depth / 2 < g["body_length"] / 2 + 0.4:
            raise ValueError("Threshold dimensions must be positive and its near face >=0.4 m beyond the torso.")
        ET.SubElement(world, "geom", name="threshold", type="box",
                      pos=_numbers([center_x, 0, height / 2]),
                      size=_numbers([depth / 2, 0.4, height / 2]),
                      contype="2", conaffinity="1", rgba="0.6 0.35 0.15 1")

    base = ET.SubElement(world, "body", name="base", pos=_numbers([0, 0, base_height]))
    ET.SubElement(base, "freejoint", name="floating_base")
    body_size = [g["body_length"], g["body_width"], g["body_height"]]
    servo_size = g["servo_envelope"]
    # These envelopes place an explicit mass allowance. They do not certify
    # that a specific battery or board will fit the enclosure.
    base_parts = [
        _box_component(m["body_shell"], [0, 0, 0], body_size),
        _box_component(m["battery"], [0, 0, -0.005], [0.060, 0.040, 0.025]),
        _box_component(m["compute"], [0.025, 0, 0.010], [0.045, 0.030, 0.012]),
        _box_component(m["electronics"], [-0.025, 0, 0.010], [0.030, 0.045, 0.015]),
        _box_component(m["wiring_and_fasteners"], [0, 0, 0], body_size),
    ]
    for leg_name in LEG_ORDER:
        leg = config["legs"][leg_name]
        motor_center = [leg["front"] * g["hip_x"],
                        leg["side"] * (g["hip_y"] - servo_size[1] / 2), g["hip_z"]]
        base_parts.append(_box_component(m["servo"], motor_center, servo_size))
        ET.SubElement(base, "geom", name=f"{leg_name}_roll_motor_envelope", type="box",
                      pos=_numbers(motor_center), size=_numbers(np.array(servo_size) / 2),
                      contype="0", conaffinity="0", rgba="0.1 0.12 0.15 1")
    _inertial(base, base_parts)
    ET.SubElement(base, "geom", name="torso_collision", type="box",
                  size=_numbers(np.array(body_size) / 2), rgba="0.15 0.28 0.39 0.65")
    ET.SubElement(base, "site", name="imu", pos="0 0 0", size="0.003", rgba="1 0.4 0 1")
    ET.SubElement(base, "camera", name="follow", mode="trackcom", pos="0.38 -0.43 0.24",
                  xyaxes="0.75 0.66 0 -0.26 0.30 0.92")

    actuator_root = ET.SubElement(root, "actuator")
    for leg_name in LEG_ORDER:
        leg = config["legs"][leg_name]
        lateral = leg["side"] * g["hip_offset"]
        upper, lower, radius = g["upper_length"], g["lower_length"], g["link_radius"]
        origin = [leg["front"] * g["hip_x"], leg["side"] * g["hip_y"], g["hip_z"]]
        roll = ET.SubElement(base, "body", name=f"{leg_name}_roll_link", pos=_numbers(origin))
        ET.SubElement(roll, "joint", name=f"{leg_name}_hip_roll", axis="1 0 0",
                      range=_numbers(config["joint_limits_rad"]["hip_roll"]))
        _inertial(roll, [
            _box_component(m["roll_bracket"], [0, lateral / 2, 0], [0.012, abs(lateral), 0.012]),
            _box_component(m["servo"], [0, lateral, 0], servo_size),
        ])
        ET.SubElement(roll, "geom", name=f"{leg_name}_roll_bar", type="capsule",
                      fromto=_numbers([0, 0, 0, 0, lateral, 0]), size="0.006", rgba="0.8 0.45 0.12 1")
        ET.SubElement(roll, "geom", name=f"{leg_name}_hip_motor_envelope", type="box",
                      pos=_numbers([0, lateral, 0]), size=_numbers(np.array(servo_size) / 2),
                      contype="0", conaffinity="0", rgba="0.1 0.12 0.15 1")
        hip = ET.SubElement(roll, "body", name=f"{leg_name}_upper_link", pos=_numbers([0, lateral, 0]))
        ET.SubElement(hip, "joint", name=f"{leg_name}_hip_pitch", axis="0 1 0",
                      range=_numbers(config["joint_limits_rad"]["hip_pitch"]))
        _inertial(hip, [
            _box_component(m["upper_link"], [0, 0, -upper / 2], [2 * radius, 2 * radius, upper]),
            _box_component(m["servo"], [0, 0, -upper], servo_size),
        ])
        ET.SubElement(hip, "geom", name=f"{leg_name}_upper_bar", type="capsule",
                      fromto=_numbers([0, 0, 0, 0, 0, -upper]), size=str(radius), rgba="0.2 0.47 0.61 1")
        ET.SubElement(hip, "geom", name=f"{leg_name}_knee_motor_envelope", type="box",
                      pos=_numbers([0, 0, -upper]), size=_numbers(np.array(servo_size) / 2),
                      contype="0", conaffinity="0", rgba="0.1 0.12 0.15 1")
        knee = ET.SubElement(hip, "body", name=f"{leg_name}_lower_link", pos=_numbers([0, 0, -upper]))
        ET.SubElement(knee, "joint", name=f"{leg_name}_knee", axis="0 1 0",
                      range=_numbers(config["joint_limits_rad"]["knee"]))
        _inertial(knee, [
            _box_component(m["lower_link"], [0, 0, -lower / 2], [2 * radius, 2 * radius, lower]),
            _sphere_component(m["foot"], [0, 0, -lower], g["foot_radius"]),
        ])
        ET.SubElement(knee, "geom", name=f"{leg_name}_lower_bar", type="capsule",
                      fromto=_numbers([0, 0, 0, 0, 0, -lower]), size=str(radius), rgba="0.2 0.47 0.61 1")
        ET.SubElement(knee, "geom", name=f"{leg_name}_foot_collision", type="sphere",
                      pos=_numbers([0, 0, -lower]), size=str(g["foot_radius"]), rgba="0.08 0.08 0.08 1")
        ET.SubElement(knee, "site", name=f"{leg_name}_foot", pos=_numbers([0, 0, -lower]),
                      size="0.002", rgba="1 0.2 0.1 1")
        for joint_name in JOINT_ORDER:
            ET.SubElement(actuator_root, "position", name=f"{leg_name}_{joint_name}_pd",
                          joint=f"{leg_name}_{joint_name}", kp=str(a["kp_nm_per_rad"]),
                          kv=str(a["kv_nm_s_per_rad"]), ctrllimited="true", forcelimited="true",
                          ctrlrange=_numbers(config["joint_limits_rad"][joint_name]),
                          forcerange=_numbers([-a["torque_limit_nm"], a["torque_limit_nm"]]))

    sensors = ET.SubElement(root, "sensor")
    ET.SubElement(sensors, "accelerometer", name="imu_accel", site="imu")
    ET.SubElement(sensors, "gyro", name="imu_gyro", site="imu")
    ET.SubElement(sensors, "framequat", name="imu_orientation", objtype="site", objname="imu")
    keyframe = ET.SubElement(root, "keyframe")
    joint_home = list(home) * 4
    ET.SubElement(keyframe, "key", name="stand",
                  qpos=_numbers([0, 0, base_height, 1, 0, 0, 0] + joint_home), ctrl=_numbers(joint_home))
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"
