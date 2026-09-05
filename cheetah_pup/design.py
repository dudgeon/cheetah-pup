"""Design parameters, the Mini Cheetah reference proportions, and the candidate presets.

Body frame: x forward, y left, z up, origin at the center of the rectangle formed by the four
hip-pitch axes, at hip height. Each leg's abad (roll) axis runs along x and intersects that leg's
hip-pitch axis, exactly as on the MIT Mini Cheetah.
"""

from __future__ import annotations

from dataclasses import dataclass, replace, asdict
from typing import Literal

Architecture = Literal["direct", "belt", "pushrod"]
KneeConfig = Literal["cheetah", "x", "forward"]

# MIT Mini Cheetah, from mit-biomimetics/Cheetah-Software (MiniCheetah.h) and the ICRA 2019 paper.
MINI_CHEETAH = {
    "thigh": 0.209,
    "shank": 0.195,
    "abad_link": 0.062,
    "hip_to_hip": 0.38,
    "abad_to_abad": 0.098,
    "stance_height": 0.29,     # hip height in the nominal stand pose (hip -0.8 rad, knee 1.6 rad)
    "mass": 9.0,
    "body_length": 0.48,
    "body_width": 0.27,
    "body_height": 0.30,
}


@dataclass
class DesignParams:
    name: str
    architecture: Architecture
    # Leg geometry
    thigh: float                 # m, hip-pitch axis to knee axis
    shank: float                 # m, knee axis to foot center
    abad_link: float             # m, abad axis to the thigh plane (lateral)
    # Body geometry
    hip_to_hip: float            # m, fore-aft distance between hip-pitch axes
    abad_to_abad: float          # m, lateral distance between the two abad axes
    hip_x_offset: float          # m, from the abad servo horn face (body end) to the hip-pitch axis
    body_height: float           # m, shell height (centered on the hip axes + body_z_offset)
    body_z_offset: float = 0.0   # m, shell center above the hip axes
    wall: float = 0.003          # m, printed shell wall
    # Posture and gait defaults
    stance_height: float = 0.12  # m, hip axis above ground in the nominal stance
    knee_config: KneeConfig = "cheetah"
    foot_radius: float = 0.010
    step_length: float = 0.06    # m, trot
    stride_frequency: float = 1.4  # Hz, trot — the STS3215 speed cap binds above ~1.5 Hz
    swing_height: float = 0.025  # m
    stance_depth: float = 0.006  # m, stance feet reach below the nominal height so they actually load
    dynamic_factor: float = 1.5  # peak-to-static vertical load factor during trot
    lateral_shift: float = 0.01  # m, abad torque case: foot displaced outward from the thigh plane
    # Transmission
    knee_ratio: float = 1.0      # knee joint torque / knee servo torque (belt or linkage reduction)
    knee_ratio_note: str = ""
    # Phase 2 packaging details (CAD-derived; the kinematics library treats the foot as being in
    # the thigh plane, the sim model uses the offset)
    foot_y_offset: float = 0.0   # m, foot centre outboard of the thigh plane (shank pad + beam jog)

    def knee_sign(self, front: bool) -> int:
        """+1 = knee points backward (shank flexes forward), -1 = knee points forward."""
        if self.knee_config == "cheetah":
            return 1
        if self.knee_config == "forward":
            return -1
        return 1 if front else -1  # "x": front knees back, rear knees forward

    @property
    def leg_length(self) -> float:
        return self.thigh + self.shank

    @property
    def shell_length(self) -> float:
        return self.hip_to_hip - 2 * self.hip_x_offset

    @property
    def shell_width(self) -> float:
        """Outer shell width: the abad servos' short ends (10.1 mm past the axis) plus clearance and walls."""
        return self.abad_to_abad + 2 * (0.0101 + 0.0009 + self.wall)

    def ratios(self) -> dict:
        mc = MINI_CHEETAH
        return {
            "shank/thigh": self.shank / self.thigh,
            "shank/thigh (Mini Cheetah)": mc["shank"] / mc["thigh"],
            "hip_to_hip/thigh": self.hip_to_hip / self.thigh,
            "hip_to_hip/thigh (Mini Cheetah)": mc["hip_to_hip"] / mc["thigh"],
            "abad_link/thigh": self.abad_link / self.thigh,
            "abad_link/thigh (Mini Cheetah)": mc["abad_link"] / mc["thigh"],
            "stance/leg_length": self.stance_height / self.leg_length,
            "stance/leg_length (Mini Cheetah)": mc["stance_height"] / (mc["thigh"] + mc["shank"]),
            "linear scale vs Mini Cheetah": self.thigh / mc["thigh"],
        }

    def to_dict(self) -> dict:
        return asdict(self)


# Baseline proportions shared by the candidates (Mini Cheetah ratios, sized to the STS3215 and the
# electronics bay). Only the knee transmission and the hip cluster differ between A/B/C.
# Phase 2 (CAD) refinements to the DR-01 baseline, all packaging-driven, kinematics unchanged:
# abad_link 40 -> 43 mm so the knee servo case clears the shell side wall; hip_x_offset 16 -> 18 mm
# so the hip servo case clears the abad bracket's back plate; abad_to_abad 70 -> 74 mm so the two
# abad servo cases at one end of the body do not touch; foot 3.25 mm outboard of the thigh plane.
_BASE = dict(
    thigh=0.090,
    shank=0.085,
    abad_link=0.043,
    hip_to_hip=0.180,
    abad_to_abad=0.074,
    hip_x_offset=0.018,
    body_height=0.062,
    stance_height=0.120,
    foot_y_offset=0.00325,
)

PRESETS: dict[str, DesignParams] = {
    "A": DesignParams(
        name="A · Direct drive",
        architecture="direct",
        knee_ratio=1.0,
        knee_ratio_note="knee servo at the knee, drives the shank directly",
        **_BASE,
    ),
    "B": DesignParams(
        name="B · Coaxial hip + belt knee",
        architecture="belt",
        knee_ratio=1.25,
        knee_ratio_note="GT2 belt, 20T at the hip to 25T at the knee; 1.5:1 exceeds the servo speed cap at a 1.4 Hz trot",
        **{**_BASE, "stance_height": 0.117},
    ),
    "C": DesignParams(
        name="C · Coaxial hip + pushrod knee",
        architecture="pushrod",
        knee_ratio=1.2,
        knee_ratio_note="4-bar pushrod, ~1.2:1 average; the ratio varies ±20% over the knee range",
        **{**_BASE, "stance_height": 0.118},
    ),
}

SIZES = {"S": 0.85, "M": 1.0, "L": 1.15}

# Locked on 2026-09-04 (DR-01): the owner accepted the review page defaults — candidate A (direct
# drive), size M, knees back, baseline proportions and gait. Every downstream artifact (MJCF, RL
# environment, CAD) derives from `locked()`; change it only with a new dated design-log entry.
LOCKED_KEY = "A"
LOCKED_SIZE = "M"
LOCKED_NAME = "Cheetah Pup v0.1 · A direct drive · M (locked 2026-09-04, Phase 2 packaging refinements)"


def locked() -> "DesignParams":
    return preset(LOCKED_KEY, LOCKED_SIZE, name=LOCKED_NAME)


def preset(key: str, size: str = "M", **overrides) -> DesignParams:
    """Candidate `key` at size `size`, with any field overridden."""
    p = PRESETS[key]
    s = SIZES[size]
    scaled = replace(
        p,
        name=f"{p.name} · {size}",
        thigh=p.thigh * s,
        shank=p.shank * s,
        abad_link=p.abad_link * (0.5 + 0.5 * s),
        hip_to_hip=max(p.hip_to_hip * s, 0.175),
        abad_to_abad=p.abad_to_abad * (0.5 + 0.5 * s),
        stance_height=p.stance_height * s,
        step_length=p.step_length * s,
        swing_height=p.swing_height * s,
    )
    return replace(scaled, **overrides) if overrides else scaled
