import math

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from cheetah_pup.analysis import mass_model
from cheetah_pup.design import locked, preset
from cheetah_pup.gait import LEGS, LEG_SIDE, LEG_FRONT
from cheetah_pup.kinematics import leg_fk
from cheetah_pup.mjcf import build_mjcf, servo_gains, joint_names


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_string(build_mjcf(locked()))


def test_model_loads_with_expected_dofs(model):
    assert model.nq == 7 + 12 and model.nv == 6 + 12 and model.nu == 12
    assert [model.actuator(i).name for i in range(12)] == joint_names()


def test_model_mass_matches_sizing(model):
    p = locked()
    assert float(sum(model.body_mass)) == pytest.approx(mass_model(p)["total"], rel=0.02)


def test_forward_kinematics_matches_library(model):
    """Joint conventions in the MJCF reproduce kinematics.leg_fk for every leg."""
    p = locked()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
    q = {"LF": (0.2, -0.6, 1.3), "RF": (-0.1, 0.3, 1.0), "LH": (0.3, -0.9, 1.8), "RH": (0.15, -0.4, 0.9)}
    for i, leg in enumerate(LEGS):
        data.qpos[7 + 3 * i: 10 + 3 * i] = q[leg]
    mujoco.mj_kinematics(model, data)
    for leg in LEGS:
        side, front = LEG_SIDE[leg], LEG_FRONT[leg]
        hx = (1 if front else -1) * p.hip_to_hip / 2
        ay = side * p.abad_to_abad / 2
        expect = leg_fk(p.thigh, p.shank, p.abad_link, *q[leg], side)["foot"]
        got = data.site(f"{leg}_foot").xpos - np.array([0, 0, p.stance_height + p.foot_radius])
        assert got == pytest.approx([hx + expect[0], ay + expect[1], expect[2]], abs=1e-6), leg


def test_stand_keyframe_holds(model):
    p = locked()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
    data.ctrl[:] = model.key("stand").ctrl
    for _ in range(1000):
        mujoco.mj_step(model, data)
    assert data.qpos[2] > 0.95 * (p.stance_height + p.foot_radius)
    assert abs(data.qpos[3]) > 0.995  # near-identity orientation quaternion (w component)
    assert float(np.max(np.abs(data.qpos[7:19] - model.key("stand").ctrl))) < 0.05  # < 3° servo droop


def test_servo_gains_are_in_a_sane_range():
    g = servo_gains("datasheet")
    assert 10 < g["kp"] < 40
    assert 0.3 < g["kd"] < 1.0
    assert g["limit"] == pytest.approx(1.912, abs=0.01)
    assert servo_gains("bam")["limit"] > g["limit"]


def test_other_architectures_build():
    for key in ("B", "C"):
        m = mujoco.MjModel.from_xml_string(build_mjcf(preset(key, "M")))
        assert m.nu == 12
