"""Mechanical balance and timing checks independent of rendered appearance."""

import copy

import mujoco
import numpy as np
import pytest

from cheetah_pup.analysis import joint_addresses, standing_model
from cheetah_pup.gait_demo import trajectory
from cheetah_pup.gait_load import (
    closest_vertical_forces,
    geom_floor_clearance,
    periodic_derivatives,
    minimum_peak_static_allocation,
    static_frame,
    vertical_force_map,
)
from cheetah_pup.kinematics import LEG_ORDER
from cheetah_pup.model import load_config


@pytest.fixture
def config():
    return load_config("config/robot.json")


def test_loaded_crawl_frames_balance_force_moment_and_link_gravity(config):
    model, frames, _ = trajectory(config)
    data = mujoco.MjData(model)
    for frame in frames:
        data.qpos[:] = frame["qpos"]
        mujoco.mj_forward(model, data)
        forces, demand = static_frame(model, data, frame["active"])
        assert np.min(forces) >= 0
        assert np.linalg.norm(demand[:6]) < 1e-8
        com = data.subtree_com[model.body("base").id]
        moment = sum(
            np.cross(data.site(f"{leg}_foot").xpos - com, [0, 0, force])
            for leg, force in zip(frame["active"], forces)
        )
        np.testing.assert_allclose(moment, 0, atol=1e-8)
        np.testing.assert_allclose(
            sum(forces), -model.body_mass.sum() * model.opt.gravity[2], atol=1e-8
        )
        # Link gravity must be retained: using -J.T F alone is a different
        # calculation for at least one articulated joint in these loaded poses.
        joint_dofs = np.concatenate(
            [joint_addresses(model, leg)[1] for leg in LEG_ORDER]
        )
        assert np.max(np.abs(data.qfrc_bias[joint_dofs])) > 0.0001


def test_static_joint_demand_matches_independent_virtual_work(config):
    model, frames, _ = trajectory(config)
    frame = next(frame for frame in frames[14:] if len(frame["active"]) == 3)
    data = mujoco.MjData(model)
    data.qpos[:] = frame["qpos"]
    mujoco.mj_forward(model, data)
    forces, demand = static_frame(model, data, frame["active"])
    positions = data.qpos.copy()

    # Differentiate gravitational potential minus external work with fixed
    # world vertical forces. This does not use MuJoCo Jacobians or qfrc_bias.
    def potential_minus_ground_work(qpos):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        potential = -np.sum(model.body_mass[:, None] * data.xipos * model.opt.gravity)
        ground_work = sum(
            force * data.site(f"{leg}_foot").xpos[2]
            for leg, force in zip(frame["active"], forces)
        )
        return potential - ground_work

    for leg in LEG_ORDER:
        qadr, dadr = joint_addresses(model, leg)
        for qindex, dof in zip(qadr, dadr):
            before, after = positions.copy(), positions.copy()
            before[qindex] -= 1e-6
            after[qindex] += 1e-6
            derivative = (
                potential_minus_ground_work(after) - potential_minus_ground_work(before)
            ) / 2e-6
            assert derivative == pytest.approx(demand[dof], abs=2e-9)


def test_static_demand_scales_with_all_body_masses(config):
    model, frames, _ = trajectory(config)
    frame = frames[14]
    data = mujoco.MjData(model)
    data.qpos[:] = frame["qpos"]
    mujoco.mj_forward(model, data)
    forces, demand = static_frame(model, data, frame["active"])
    heavier = copy.deepcopy(config)
    for name in heavier["mass_kg"]:
        heavier["mass_kg"][name] *= 2
    heavy_model, heavy_data = standing_model(heavier)
    heavy_data.qpos[:] = frame["qpos"]
    mujoco.mj_forward(heavy_model, heavy_data)
    heavy_forces, heavy_demand = static_frame(heavy_model, heavy_data, frame["active"])
    np.testing.assert_allclose(heavy_forces, 2 * forces, atol=1e-9)
    np.testing.assert_allclose(heavy_demand, 2 * demand, atol=1e-9)


def test_static_rejects_moving_data(config):
    model, data = standing_model(config)
    data.qvel[6] = 1
    with pytest.raises(ValueError, match="zero velocity"):
        static_frame(model, data, LEG_ORDER)


def test_periodic_derivatives_have_correct_seam_and_time_scaling(config):
    model, data = standing_model(config)
    count, duration, stride, amplitude = 256, 2.0, 0.02, 0.2
    times = np.arange(count) / count * duration
    positions = np.tile(data.qpos, (count, 1))
    positions[:, 0] += stride * times / duration
    qadr, dadr = joint_addresses(model, "FL")
    positions[:, qadr[0]] += amplitude * np.sin(2 * np.pi * times / duration)
    velocity, acceleration = periodic_derivatives(model, positions, duration, stride)
    np.testing.assert_allclose(velocity[:, 0], stride / duration, atol=1e-13)
    np.testing.assert_allclose(acceleration[:, :3], 0, atol=1e-11)
    expected_speed = (
        amplitude * 2 * np.pi / duration * np.cos(2 * np.pi * times / duration)
    )
    expected_accel = (
        -amplitude * (2 * np.pi / duration) ** 2 * np.sin(2 * np.pi * times / duration)
    )
    np.testing.assert_allclose(velocity[:, dadr[0]], expected_speed, atol=7e-5)
    np.testing.assert_allclose(acceleration[:, dadr[0]], expected_accel, atol=1.1e-4)
    twice_speed, four_accel = periodic_derivatives(
        model, positions, duration / 2, stride
    )
    np.testing.assert_allclose(twice_speed, 2 * velocity, atol=1e-11)
    np.testing.assert_allclose(four_accel, 4 * acceleration, atol=1e-11)


def test_vertical_solver_exposes_unprovided_horizontal_wrench():
    feet = np.array([[-0.1, -0.05], [0.1, -0.05], [0, 0.05]])
    matrix = np.array([[0, 0, 1, y, -x, 0] for x, y in feet]).T
    desired = np.array([2.0, -3.0, 6.0, 0, 0, 0.04])
    loads = closest_vertical_forces(matrix, desired)
    residual = desired - matrix @ loads
    assert np.min(loads) >= 0
    np.testing.assert_allclose(loads.sum(), 6, atol=1e-12)
    np.testing.assert_allclose(residual[[0, 1, 5]], [2.0, -3.0, 0.04], atol=1e-12)
    np.testing.assert_allclose(residual[[2, 3, 4]], 0, atol=1e-12)


def test_vertical_solver_cannot_generate_ground_suction():
    matrix = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]).T
    desired = np.array([0.0, 0.0, -4.0, 0.0, 0.0, 0.0])
    loads = closest_vertical_forces(matrix, desired)
    np.testing.assert_allclose(loads, 0, atol=1e-12)
    assert np.linalg.norm(desired - matrix @ loads) == 4.0


def test_dynamic_contact_fit_recovers_static_loaded_solution(config):
    model, frames, _ = trajectory(config)
    data = mujoco.MjData(model)
    frame = frames[14]
    data.qpos[:] = frame["qpos"]
    mujoco.mj_forward(model, data)
    static_loads, _ = static_frame(model, data, frame["active"])
    matrix = vertical_force_map(model, data, frame["active"])
    fitted = closest_vertical_forces(matrix[:6], data.qfrc_bias[:6])
    np.testing.assert_allclose(fitted, static_loads, atol=1e-9)


def test_minimax_contact_allocation_is_balanced_and_beats_independent_grid(config):
    model, frames, _ = trajectory(config)
    data = mujoco.MjData(model)
    # Sample a late four-foot shift, where minimum-norm foot loads need not
    # minimize joint effort. Brute-force the feasible load interval using a
    # separately constructed cross-product wrench matrix, not the optimizer's
    # Jacobian SVD or line-intersection candidates.
    frame = [frame for frame in frames if len(frame["active"]) == 4][-2]
    data.qpos[:] = frame["qpos"]
    mujoco.mj_forward(model, data)
    initial_forces, initial_demand = static_frame(model, data, frame["active"])
    optimized_forces, optimized_demand = minimum_peak_static_allocation(
        model, data, frame["active"]
    )
    indices = np.concatenate([joint_addresses(model, leg)[1] for leg in LEG_ORDER])
    initial_peak = np.max(np.abs(initial_demand[indices]))
    optimized_peak = np.max(np.abs(optimized_demand[indices]))
    assert optimized_peak <= initial_peak + 1e-10
    assert np.min(optimized_forces) >= -1e-10
    np.testing.assert_allclose(optimized_demand[:6], 0, atol=1e-8)
    xy = np.array([data.site(f"{leg}_foot").xpos[:2] for leg in frame["active"]])
    explicit_map = np.vstack((np.ones(4), xy[:, 1], -xy[:, 0]))
    direction = np.linalg.svd(explicit_map)[2][-1]
    low = max(
        -force / delta
        for force, delta in zip(initial_forces, direction)
        if delta > 1e-10
    )
    high = min(
        -force / delta
        for force, delta in zip(initial_forces, direction)
        if delta < -1e-10
    )
    generalized_map = vertical_force_map(model, data, frame["active"])
    grid_peak = min(
        np.max(
            np.abs(
                (
                    data.qfrc_bias
                    - generalized_map @ (initial_forces + scalar * direction)
                )[indices]
            )
        )
        for scalar in np.linspace(low, high, 2001)
    )
    assert optimized_peak <= grid_peak + 1e-10
    assert grid_peak - optimized_peak < 1e-4


def test_three_foot_minimax_has_no_spurious_allocation_freedom(config):
    model, frames, _ = trajectory(config)
    data = mujoco.MjData(model)
    frame = next(frame for frame in frames if len(frame["active"]) == 3)
    data.qpos[:] = frame["qpos"]
    mujoco.mj_forward(model, data)
    forces, demand = static_frame(model, data, frame["active"])
    optimized_forces, optimized_demand = minimum_peak_static_allocation(
        model, data, frame["active"]
    )
    np.testing.assert_allclose(optimized_forces, forces, atol=1e-12)
    np.testing.assert_allclose(optimized_demand, demand, atol=1e-12)


def test_clearance_rotated_box_against_independent_vertices(config):
    model, data = standing_model(config)
    data.qpos[3:7] = [np.cos(0.3), np.sin(0.3), 0, 0]
    mujoco.mj_forward(model, data)
    for geom_id in np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_BOX):
        corners = (
            np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
            * model.geom_size[geom_id]
        )
        corners = (
            corners @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
        )
        assert geom_floor_clearance(model, data, geom_id) == pytest.approx(
            np.min(corners[:, 2]), abs=1e-12
        )
