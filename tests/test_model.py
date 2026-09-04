"""Numerical checks of the provisional model, not hardware validation.

These checks do not establish mechanical clearance, self-collision safety,
actuator fidelity, carpet behavior, or successful sim-to-real transfer.
"""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from cheetah_pup.kinematics import foot_position, leg_jacobian
from cheetah_pup.model import build_mjcf, load_config, total_mass


LEG_NAMES = ("FL", "FR", "RL", "RR")
JOINT_NAMES = ("hip_roll", "hip_pitch", "knee")
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "robot.json"


@pytest.fixture(scope="module")
def config():
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def flat_model(config):
    return mujoco.MjModel.from_xml_string(build_mjcf(config, terrain="flat"))


def _stand(model):
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    assert key_id >= 0, "The model must provide its nominal stand configuration."
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    return data


def _joint_ids(model, leg):
    ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_{joint}")
        for joint in JOINT_NAMES
    ]
    assert min(ids) >= 0
    return np.asarray(ids)


def _pose(model, config, seed, rotated):
    """Populate each leg independently and return an independent base rotation."""
    data = _stand(model)
    base_pos = np.array([0.18, -0.13, 0.33])
    axis = np.array([1.0, -2.0, 3.0])
    axis /= np.linalg.norm(axis)
    angle = 0.62 if rotated else 0.0
    quaternion = np.r_[np.cos(angle / 2), axis * np.sin(angle / 2)]
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    rotation = (
        np.eye(3) * np.cos(angle)
        + (1 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    assert base_id >= 0
    free_id = model.body_jntadr[base_id]
    assert model.jnt_type[free_id] == mujoco.mjtJoint.mjJNT_FREE
    free_adr = model.jnt_qposadr[free_id]
    data.qpos[free_adr : free_adr + 3] = base_pos
    data.qpos[free_adr + 3 : free_adr + 7] = quaternion

    limits = np.array([config["joint_limits_rad"][name] for name in JOINT_NAMES])
    rng = np.random.default_rng(seed)
    poses = {}
    for leg in LEG_NAMES:
        # Stay away from the joint limits while exercising asymmetric poses.
        q = limits[:, 0] + rng.uniform(0.15, 0.85, 3) * np.diff(limits, axis=1)[:, 0]
        data.qpos[model.jnt_qposadr[_joint_ids(model, leg)]] = q
        poses[leg] = q
    mujoco.mj_forward(model, data)
    return data, poses, base_pos, rotation


@pytest.mark.parametrize("seed", [19, 73, 211])
@pytest.mark.parametrize("rotated", [False, True], ids=["translated", "rotated"])
def test_analytical_fk_matches_mujoco_all_legs(flat_model, config, seed, rotated):
    data, poses, base_pos, rotation = _pose(flat_model, config, seed, rotated)
    for leg, q in poses.items():
        site_id = mujoco.mj_name2id(flat_model, mujoco.mjtObj.mjOBJ_SITE, f"{leg}_foot")
        assert site_id >= 0
        expected = base_pos + rotation @ foot_position(config, leg, q)
        np.testing.assert_allclose(
            data.site_xpos[site_id], expected, atol=1e-11, rtol=0
        )


@pytest.mark.parametrize("seed", [19, 73, 211])
@pytest.mark.parametrize("rotated", [False, True], ids=["translated", "rotated"])
def test_leg_jacobians_match_finite_difference_and_mujoco(
    flat_model, config, seed, rotated
):
    data, poses, _, rotation = _pose(flat_model, config, seed, rotated)
    epsilon = 1e-7
    for leg, q in poses.items():
        analytic = leg_jacobian(config, leg, q)
        finite_difference = np.column_stack(
            [
                (
                    foot_position(config, leg, q + step)
                    - foot_position(config, leg, q - step)
                )
                / (2 * epsilon)
                for step in np.eye(3) * epsilon
            ]
        )
        np.testing.assert_allclose(analytic, finite_difference, atol=2e-9, rtol=0)

        site_id = mujoco.mj_name2id(flat_model, mujoco.mjtObj.mjOBJ_SITE, f"{leg}_foot")
        assert site_id >= 0
        translation_jacobian = np.zeros((3, flat_model.nv))
        rotation_jacobian = np.zeros((3, flat_model.nv))
        mujoco.mj_jacSite(
            flat_model, data, translation_jacobian, rotation_jacobian, site_id
        )
        joint_dofs = flat_model.jnt_dofadr[_joint_ids(flat_model, leg)]
        np.testing.assert_allclose(
            translation_jacobian[:, joint_dofs],
            rotation @ analytic,
            atol=1e-11,
            rtol=0,
        )


@pytest.mark.parametrize("seed", [23, 79, 223])
def test_left_right_mirror_in_body_frame(config, seed):
    rng = np.random.default_rng(seed)
    for left, right in (("FL", "FR"), ("RL", "RR")):
        left_q = rng.uniform([-0.5, -0.8, -1.8], [0.5, 0.8, -0.3])
        right_q = left_q * np.array([-1, 1, 1])
        np.testing.assert_allclose(
            foot_position(config, right, right_q),
            foot_position(config, left, left_q) * np.array([1, -1, 1]),
            atol=1e-12,
            rtol=0,
        )


def test_mass_budget_and_physical_inertias(flat_model, config):
    itemized_mass = sum(
        mass * config["mass_counts"][name] for name, mass in config["mass_kg"].items()
    )
    assert itemized_mass == pytest.approx(0.613, abs=1e-12)
    assert total_mass(config) == pytest.approx(itemized_mass, abs=1e-12)
    assert flat_model.body_mass.sum() == pytest.approx(itemized_mass, abs=1e-12)
    assert np.all(flat_model.body_mass[1:] > 0)
    inertias = flat_model.body_inertia[1:]
    assert np.isfinite(inertias).all()
    assert np.all(inertias > 0)
    # Principal inertias must satisfy the rigid-body triangle inequalities.
    assert np.all(2 * inertias <= inertias.sum(axis=1, keepdims=True) + 1e-15)


def test_joint_names_order_and_limits(flat_model, config):
    expected = [f"{leg}_{joint}" for leg in LEG_NAMES for joint in JOINT_NAMES]
    hinge_ids = np.flatnonzero(flat_model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    actual = [
        mujoco.mj_id2name(flat_model, mujoco.mjtObj.mjOBJ_JOINT, int(joint_id))
        for joint_id in hinge_ids
    ]
    assert actual == expected
    assert flat_model.nq == 19
    assert flat_model.nv == 18
    assert np.all(flat_model.jnt_limited[hinge_ids])
    expected_ranges = [
        config["joint_limits_rad"][joint] for _ in LEG_NAMES for joint in JOINT_NAMES
    ]
    np.testing.assert_allclose(flat_model.jnt_range[hinge_ids], expected_ranges)


def test_actuator_torque_limits_and_joint_mapping(flat_model, config):
    expected_joints = [
        int(joint_id) for leg in LEG_NAMES for joint_id in _joint_ids(flat_model, leg)
    ]
    assert flat_model.nu == 12
    np.testing.assert_array_equal(flat_model.actuator_trnid[:, 0], expected_joints)
    assert np.all(flat_model.actuator_forcelimited)
    torque_limit = config["actuator"]["torque_limit_nm"]
    np.testing.assert_allclose(
        flat_model.actuator_forcerange,
        np.tile([-torque_limit, torque_limit], (12, 1)),
    )
    # A force clamp is a joint-torque clamp only with this transmission ratio.
    np.testing.assert_allclose(flat_model.actuator_gear[:, 0], 1.0)
    np.testing.assert_allclose(flat_model.actuator_gear[:, 1:], 0.0)


def test_stand_foot_soles_align_flat_floor(flat_model, config):
    data = _stand(flat_model)
    for leg in LEG_NAMES:
        joint_qpos = flat_model.jnt_qposadr[_joint_ids(flat_model, leg)]
        np.testing.assert_allclose(data.qpos[joint_qpos], config["home_q_rad"])
        site_id = mujoco.mj_name2id(flat_model, mujoco.mjtObj.mjOBJ_SITE, f"{leg}_foot")
        assert site_id >= 0
        assert data.site_xpos[site_id, 2] - config["geometry_m"][
            "foot_radius"
        ] == pytest.approx(0.0, abs=1e-11)
    world_planes = np.flatnonzero(
        (flat_model.geom_bodyid == 0)
        & (flat_model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    )
    assert len(world_planes) == 1
    assert flat_model.geom_pos[world_planes[0], 2] == pytest.approx(0.0)


def test_threshold_is_ten_millimeters_above_floor(config):
    model = mujoco.MjModel.from_xml_string(build_mjcf(config, terrain="threshold"))
    world_boxes = np.flatnonzero(
        (model.geom_bodyid == 0) & (model.geom_type == mujoco.mjtGeom.mjGEOM_BOX)
    )
    assert len(world_boxes) == 1
    geom_id = world_boxes[0]
    center = model.geom_pos[geom_id]
    half_size = model.geom_size[geom_id]
    assert center[2] - half_size[2] == pytest.approx(0.0, abs=1e-12)
    assert center[2] + half_size[2] == pytest.approx(0.01, abs=1e-12)
    assert center[0] == pytest.approx(config["simulation"]["threshold_center_x_m"])
    assert 2 * half_size[0] == pytest.approx(config["simulation"]["threshold_depth_m"])
