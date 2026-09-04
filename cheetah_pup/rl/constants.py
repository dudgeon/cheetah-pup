"""Names shared by the model generator and the environment."""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
RL_XML = ROOT / "sim" / "cheetah_pup_rl.xml"

LEGS = ("LF", "RF", "LH", "RH")
FEET_SITES = [f"{leg}_foot" for leg in LEGS]
FEET_GEOMS = [f"{leg}_foot" for leg in LEGS]
FEET_LINVEL_SENSORS = [f"{leg}_foot_global_linvel" for leg in LEGS]
FEET_POS_SENSORS = [f"{leg}_foot_pos" for leg in LEGS]
FEET_CONTACT_SENSORS = [f"{leg}_foot_floor_found" for leg in LEGS]

ROOT_BODY = "trunk"
IMU_SITE = "imu"
FLOOR_GEOM = "floor"

UPVECTOR_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"

# Actuator order in the model: for each leg in LEGS, (abad, hip, knee).
JOINT_NAMES = [f"{leg}_{j}" for leg in LEGS for j in ("abad", "hip", "knee")]
