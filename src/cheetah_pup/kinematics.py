"""Analytical leg kinematics in the body frame (x forward, y left, z up).

This is an original three-joint serial chain. Positive joint rotations use the
right-hand rule about x (hip roll), then local y (hip pitch and knee).
"""

from __future__ import annotations

import numpy as np


LEG_ORDER = ("FL", "FR", "RL", "RR")
JOINT_ORDER = ("hip_roll", "hip_pitch", "knee")


def _chain(config: dict, leg_name: str, q) -> tuple:
    if leg_name not in LEG_ORDER:
        raise ValueError(f"Unknown leg {leg_name!r}; expected one of {LEG_ORDER}.")
    q = np.asarray(q, dtype=float)
    if q.shape != (3,) or not np.all(np.isfinite(q)):
        raise ValueError("q must contain three finite joint angles in radians.")
    geom = config["geometry_m"]
    leg = config["legs"][leg_name]
    a, b, c = q
    ca, sa = np.cos(a), np.sin(a)
    roll = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])

    def pitch(angle):
        co, si = np.cos(angle), np.sin(angle)
        return np.array([[co, 0, si], [0, 1, 0], [-si, 0, co]])

    origin = np.array([leg["front"] * geom["hip_x"],
                       leg["side"] * geom["hip_y"], geom["hip_z"]])
    lateral = np.array([0, leg["side"] * geom["hip_offset"], 0])
    upper = pitch(b) @ np.array([0, 0, -geom["upper_length"]])
    lower = pitch(b + c) @ np.array([0, 0, -geom["lower_length"]])
    return origin, roll, lateral, upper, lower


def foot_position(config: dict, leg_name: str, q) -> np.ndarray:
    """Return the foot sphere center in body coordinates, as a 3-vector.

    The contact sole is one foot radius below the center on a horizontal floor.
    """
    origin, roll, lateral, upper, lower = _chain(config, leg_name, q)
    return origin + roll @ (lateral + upper + lower)


def leg_jacobian(config: dict, leg_name: str, q) -> np.ndarray:
    """Return d(foot_position)/d(q), rows xyz, columns roll/pitch/knee."""
    _, roll, lateral, upper, lower = _chain(config, leg_name, q)
    axis_x = np.array([1.0, 0.0, 0.0])
    axis_y = np.array([0.0, 1.0, 0.0])
    return np.column_stack((
        np.cross(axis_x, roll @ (lateral + upper + lower)),
        roll @ np.cross(axis_y, upper + lower),
        roll @ np.cross(axis_y, lower),
    ))
