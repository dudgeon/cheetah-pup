"""Parametric gait generation: foot trajectories in each hip frame and the joint angles via IK.

Legs are ordered LF, RF, LH, RH. A leg's phase offset is the fraction of the cycle at which its
stance begins. `duty` is the stance fraction. Stance moves the foot backward at body speed; swing
returns it forward on a raised arc.
"""

from __future__ import annotations

import math

from .design import DesignParams
from .kinematics import leg_ik

LEGS = ("LF", "RF", "LH", "RH")
LEG_SIDE = {"LF": 1, "RF": -1, "LH": 1, "RH": -1}
LEG_FRONT = {"LF": True, "RF": True, "LH": False, "RH": False}

GAITS = {
    # duty, phase offsets, step-length and stride-frequency multipliers on the design defaults
    "stand":  dict(duty=1.0, phase=dict(LF=0, RF=0, LH=0, RH=0), step=0.0, freq=0.4,
                   label="Stand / crouch", crouch=0.35),
    "walk":   dict(duty=0.75, phase=dict(LH=0.0, LF=0.25, RH=0.5, RF=0.75), step=0.85, freq=0.65,
                   label="Walk (lateral sequence)"),
    "trot":   dict(duty=0.5, phase=dict(LF=0.0, RH=0.0, RF=0.5, LH=0.5), step=1.0, freq=1.0,
                   label="Trot (diagonal pairs)"),
    "pace":   dict(duty=0.5, phase=dict(LF=0.0, LH=0.0, RF=0.5, RH=0.5), step=1.0, freq=0.9,
                   label="Pace (lateral pairs)"),
    "bound":  dict(duty=0.45, phase=dict(LF=0.0, RF=0.0, LH=0.5, RH=0.5), step=1.2, freq=1.1,
                   label="Bound (front/rear pairs) — range-of-motion demo"),
    "sway":   dict(duty=1.0, phase=dict(LF=0, RF=0, LH=0, RH=0), step=0.0, freq=0.5,
                   label="Lateral sway — abad demo", sway=0.03),
}


def foot_trajectory(p: DesignParams, gait: str, leg: str, phase: float):
    """Foot position (x, y, z) in the leg's abad frame at global cycle phase `phase` in [0, 1)."""
    g = GAITS[gait]
    side = LEG_SIDE[leg]
    y = side * p.abad_link
    h = p.stance_height
    if gait == "stand":
        # vertical oscillation between the nominal stance and a crouch
        z = -h * (1.0 - g["crouch"] * 0.5 * (1.0 - math.cos(2 * math.pi * phase)))
        return 0.0, y, z
    if gait == "sway":
        y = side * p.abad_link + g["sway"] * math.sin(2 * math.pi * phase)
        return 0.0, y, -h
    s = p.step_length * g["step"]
    duty = g["duty"]
    local = (phase - g["phase"][leg]) % 1.0
    if local < duty:
        u = local / duty
        return s / 2 - u * s, y, -h
    u = (local - duty) / (1.0 - duty)
    lift = p.swing_height * math.sin(math.pi * u)
    return -s / 2 + u * s, y, -h + lift


def body_speed(p: DesignParams, gait: str) -> float:
    g = GAITS[gait]
    if g["step"] == 0.0:
        return 0.0
    return p.step_length * g["step"] * p.stride_frequency * g["freq"] / g["duty"]


def joint_trajectories(p: DesignParams, gait: str, n: int = 120):
    """Sample joint angles and velocities over one cycle.

    Returns dict leg -> dict(q=[(abad, hip, knee), ...], dq=[...], foot=[...]).
    """
    freq = p.stride_frequency * GAITS[gait]["freq"]
    dt = 1.0 / (freq * n) if freq > 0 else 1.0
    out = {}
    for leg in LEGS:
        side, front = LEG_SIDE[leg], LEG_FRONT[leg]
        ks = p.knee_sign(front)
        q = []
        feet = []
        for i in range(n):
            f = foot_trajectory(p, gait, leg, i / n)
            feet.append(f)
            q.append(leg_ik(p.thigh, p.shank, p.abad_link, f, side, ks))
        dq = []
        for i in range(n):
            a, b = q[i - 1], q[(i + 1) % n]
            dq.append(tuple((b[j] - a[j]) / (2 * dt) for j in range(3)))
        out[leg] = {"q": q, "dq": dq, "foot": feet}
    return out
