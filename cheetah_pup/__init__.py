"""Cheetah Pup design library: geometry, kinematics, gaits, and sizing analysis.

Units are SI throughout (m, kg, s, rad, N·m). This package is the single source of truth for the
robot's parameters; the design-review artifact and the MJCF generator both derive from it.
"""

from .servo import STS3215, Servo
from .electronics import PI5, BATTERY_2S, PCB, IMU
from .design import DesignParams, MINI_CHEETAH, PRESETS, preset, locked
from .kinematics import planar_fk, planar_ik, leg_fk, leg_ik, planar_jacobian
from .gait import GAITS, foot_trajectory, joint_trajectories
from .analysis import metrics, mass_model, torque_report, speed_report, packaging_report

__all__ = [
    "STS3215", "Servo", "PI5", "BATTERY_2S", "PCB", "IMU",
    "DesignParams", "MINI_CHEETAH", "PRESETS", "preset",
    "planar_fk", "planar_ik", "leg_fk", "leg_ik", "planar_jacobian",
    "GAITS", "foot_trajectory", "joint_trajectories",
    "metrics", "mass_model", "torque_report", "speed_report", "packaging_report",
]
