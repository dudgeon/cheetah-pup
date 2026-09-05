"""Dimensions (mm) for the CAD, derived from the locked design plus CAD-only constants."""

from __future__ import annotations

from cheetah_pup.design import locked
from cheetah_pup.electronics import PI5, BATTERY_2S, PCB
from cheetah_pup.analysis import nominal_pose
from . import servo as SV

MM = 1000.0
P = locked()

THIGH = P.thigh * MM
SHANK = P.shank * MM
ABAD_LINK = P.abad_link * MM
HIP_TO_HIP = P.hip_to_hip * MM
ABAD_TO_ABAD = P.abad_to_abad * MM
HIP_X_OFFSET = P.hip_x_offset * MM
BODY_H = P.body_height * MM
SHELL_LEN = P.shell_length * MM
SHELL_W = P.shell_width * MM
WALL = P.wall * MM
FOOT_R = P.foot_radius * MM
FOOT_Y = P.foot_y_offset * MM
STANCE = P.stance_height * MM

# Nominal joint angles (rad) of the stance the assembly is exported in
Q_HIP, Q_KNEE = nominal_pose(P, True)

# Printed-part constants
PLATE = 3.0            # bracket plates on servo top faces
THIGH_PLATE = 3.5      # thigh plate: sits on the hip horn disc, carries the knee servo
PAD = 3.5              # shank pad on the knee horn disc
SPLIT_Z = 14.0         # tub/lid split, above the abad servo tops (12.36)
CLEAR = 0.4            # print clearance around servo cases

# Trunk-frame positions
END_WALL_OUTER = SHELL_LEN / 2                      # 72 mm for the locked design
END_WALL_INNER = END_WALL_OUTER - WALL
ABAD_SEAT_X = END_WALL_INNER - SV.STEP_A            # horn-seat plane of the abad servo (front)
ABAD_Y = ABAD_TO_ABAD / 2

# Electronics (trunk frame, mm)
FLOOR_Z = -BODY_H / 2 + WALL
BATTERY = tuple(v * MM for v in BATTERY_2S.size)   # (x, y, z) as mounted: transverse
BATTERY_C = (0.0, 0.0, FLOOR_Z + BATTERY[2] / 2)
PI = (PI5.size[1] * MM, PI5.size[0] * MM, PI5.size[2] * MM)   # transverse: 56 along x, 85 along y
PI_C = (0.0, 0.0, FLOOR_Z + BATTERY[2] + 3.0 + PI[2] / 2)
PI_HOLES = [(x, y) for x in (-24.5, 24.5) for y in (-29.0, 29.0)]
PCB_ENV = (PCB.size[0] * MM, PCB.size[1] * MM, PCB.size[2] * MM)
PCB_C = (END_WALL_INNER - PCB_ENV[0] / 2, 0.0, BODY_H / 2 - WALL - 4.0 - PCB_ENV[2] / 2)
PCB_HOLES = [(PCB_C[0] + dx, dy) for dx in (-15.0, 15.0) for dy in (-25.0, 25.0)]
LID_BOSSES = [(-31.0, 0.0), (31.0, 0.0), (0.0, -45.5), (0.0, 45.5)]

# Materials
PLA_DENSITY = 1240.0   # kg/m^3
TPU_DENSITY = 1200.0
# Effective solid fraction of printed parts (perimeters + infill) by part class
INFILL = {"shell": 0.95, "plate": 0.85, "beam": 0.75, "tpu": 1.0}
