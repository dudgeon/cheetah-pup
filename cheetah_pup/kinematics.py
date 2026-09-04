"""3-DOF leg kinematics (abad roll, hip pitch, knee pitch) for a Mini-Cheetah-style serial leg.

Conventions (match Cheetah-Software's Mini Cheetah model):
- Sagittal plane: x forward, z up, origin at the hip-pitch axis.
- q_hip: thigh angle from straight down; positive swings the foot forward.
- q_knee: shank angle relative to the thigh; positive flexes the shank forward, which puts the
  knee *behind* the hip-foot line (the "cheetah" configuration: hip -0.8, knee +1.6 rad).
- q_abad: positive = abduction (the leg swings away from the body) for either side.
- Abad frame: origin where the abad axis (along x) meets the hip-pitch axis; the thigh plane is at
  y = side * abad_link, side = +1 for left legs, -1 for right legs.
"""

from __future__ import annotations

import math


def planar_fk(thigh: float, shank: float, q_hip: float, q_knee: float):
    """Return ((knee_x, knee_z), (foot_x, foot_z)) in the hip frame."""
    kx = thigh * math.sin(q_hip)
    kz = -thigh * math.cos(q_hip)
    a = q_hip + q_knee
    return (kx, kz), (kx + shank * math.sin(a), kz - shank * math.cos(a))


def planar_ik(thigh: float, shank: float, x: float, z: float, knee_sign: int = 1):
    """Return (q_hip, q_knee) placing the foot at (x, z). knee_sign=+1 puts the knee backward."""
    r2 = x * x + z * z
    c = (r2 - thigh * thigh - shank * shank) / (2.0 * thigh * shank)
    if c > 1.0 + 1e-9 or c < -1.0 - 1e-9:
        raise ValueError(f"foot ({x:.4f}, {z:.4f}) unreachable for links {thigh:.4f}/{shank:.4f}")
    c = max(-1.0, min(1.0, c))
    q_knee = knee_sign * math.acos(c)
    q_hip = math.atan2(x, -z) - math.atan2(shank * math.sin(q_knee), thigh + shank * math.cos(q_knee))
    return q_hip, q_knee


def planar_jacobian(thigh: float, shank: float, q_hip: float, q_knee: float):
    """d(foot_x, foot_z)/d(q_hip, q_knee) as ((dx/dhip, dx/dknee), (dz/dhip, dz/dknee))."""
    a = q_hip + q_knee
    return (
        (thigh * math.cos(q_hip) + shank * math.cos(a), shank * math.cos(a)),
        (thigh * math.sin(q_hip) + shank * math.sin(a), shank * math.sin(a)),
    )


def _roll(y: float, z: float, angle: float):
    c, s = math.cos(angle), math.sin(angle)
    return y * c - z * s, y * s + z * c


def leg_fk(thigh, shank, abad_link, q_abad, q_hip, q_knee, side: int):
    """Return dict of hip, knee, foot positions (x, y, z) in the abad frame."""
    (kx, kz), (fx, fz) = planar_fk(thigh, shank, q_hip, q_knee)
    roll = side * q_abad
    y0 = side * abad_link
    hy, hz = _roll(y0, 0.0, roll)
    ky, kz2 = _roll(y0, kz, roll)
    fy, fz2 = _roll(y0, fz, roll)
    return {"hip": (0.0, hy, hz), "knee": (kx, ky, kz2), "foot": (fx, fy, fz2)}


def leg_ik(thigh, shank, abad_link, foot, side: int, knee_sign: int = 1):
    """Return (q_abad, q_hip, q_knee) placing the foot at `foot` (x, y, z) in the abad frame."""
    x, y, z = foot
    d2 = y * y + z * z
    if d2 < abad_link * abad_link:
        raise ValueError("foot inside the abad link radius")
    zp = -math.sqrt(d2 - abad_link * abad_link)
    roll = math.atan2(z, y) - math.atan2(zp, side * abad_link)
    roll = (roll + math.pi) % (2 * math.pi) - math.pi
    q_hip, q_knee = planar_ik(thigh, shank, x, zp, knee_sign)
    return side * roll, q_hip, q_knee


def static_torques(thigh, shank, abad_link, q_hip, q_knee, force_z: float, lateral_offset: float = 0.0):
    """Joint torques balancing a vertical ground reaction `force_z` at the foot (roll = 0).

    Returns (tau_abad, tau_hip, tau_knee). tau_hip = F * foot_x, tau_knee = F * (foot_x - knee_x),
    tau_abad = F * (abad_link + lateral_offset).
    """
    (kx, _), (fx, _) = planar_fk(thigh, shank, q_hip, q_knee)
    return force_z * (abad_link + lateral_offset), force_z * fx, force_z * (fx - kx)
