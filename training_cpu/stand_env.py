"""Small proprioceptive stand task; all forward dynamics use the pinned BAM adapter."""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cheetah_pup.actuator import BamPositionController, motor_mjcf
from cheetah_pup.model import build_mjcf, load_config

TASK = {
    "name": "stand-small-reset-perturbations-v1",
    "policy_dt_s": 0.02,
    "episode_s": 5.0,
    "action_scale_rad": 0.15,
    "reset_joint_uniform_rad": 0.03,
    "reset_joint_velocity_uniform_rad_s": 0.05,
    "reset_base_xy_velocity_uniform_m_s": 0.005,
    "reset_base_angular_velocity_uniform_rad_s": 0.05,
    "minimum_height_m": 0.105,
    "maximum_tilt_rad": 0.35,
    "maximum_xy_displacement_m": 0.04,
    "loaded_contact_force_threshold_n": 1e-5,
    "physics": "Current assembly MJCF, full enabled collisions; current pinned BAM config unchanged.",
    "observation": [
        {"name": "body_gyro_rad_s", "slice": [0, 3], "scale": 0.25},
        {"name": "projected_gravity", "slice": [3, 6], "scale": 1.0},
        {"name": "zero_velocity_commands", "slice": [6, 9], "scale": 1.0},
        {"name": "joint_position_minus_home_rad", "slice": [9, 21], "scale": 1.0},
        {"name": "joint_velocity_rad_s", "slice": [21, 33], "scale": 0.05},
        {"name": "previous_action", "slice": [33, 45], "scale": 1.0},
    ],
    "observation_notes": "45 float32 values. No base position/linear velocity/contact/torque/height in actor or critic observations. Perfect simulated orientation and gyro stand in for IMU; hardware noise/latency not identified.",
    "reward": "dt*(1.5*exp(-(height_error/.012)^2) + exp(-(tilt/.15)^2) + .5*exp(-sum(base_velocity_xy^2)/.02^2) + .5*exp(-mean(joint_offset^2)/.1^2) - .02*mean(action_delta^2) - .005*mean((motor_torque/.1)^2)); -1 on failure. Privileged state allowed only in reward and termination.",
    "limitations": "No push recovery, walking, terrain, domain randomization, or stock-servo fidelity validation. Narrow reset perturbations are deliberately an installation/task smoke test.",
}


class StandEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        self.robot_config = load_config(ROOT / "config/robot.json")
        self.xml = motor_mjcf(build_mjcf(self.robot_config))
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.key_id = self.model.key("stand").id
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        self.controller = BamPositionController(self.model, self.data)
        self.home_q = self.data.qpos[self.controller.qadr].copy()
        self.home_height = float(self.data.qpos[2])
        self.substeps = round(TASK["policy_dt_s"] / self.model.opt.timestep)
        assert np.isclose(self.substeps * self.model.opt.timestep, TASK["policy_dt_s"])
        self.max_steps = round(TASK["episode_s"] / TASK["policy_dt_s"])
        self.foot_geoms = {
            self.model.geom(f"{leg}_foot_collision").id for leg in ("FL", "FR", "RL", "RR")
        }
        self.action_space = gym.spaces.Box(-1.0, 1.0, (12,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (45,), np.float32)
        self.previous_action = np.zeros(12)
        self.steps = 0
        self.initial_xy = np.zeros(2)

    def observation(self):
        rotation = self.data.body("base").xmat.reshape(3, 3)
        obs = np.concatenate((
            0.25 * self.data.sensor("imu_gyro").data,
            rotation.T @ np.array([0.0, 0.0, -1.0]),
            np.zeros(3),
            self.data.qpos[self.controller.qadr] - self.home_q,
            0.05 * self.data.qvel[self.controller.dadr],
            self.previous_action,
        )).astype(np.float32)
        assert obs.shape == (45,)
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        self.data.qpos[self.controller.qadr] += self.np_random.uniform(
            -TASK["reset_joint_uniform_rad"], TASK["reset_joint_uniform_rad"], 12
        )
        self.data.qvel[self.controller.dadr] = self.np_random.uniform(
            -TASK["reset_joint_velocity_uniform_rad_s"], TASK["reset_joint_velocity_uniform_rad_s"], 12
        )
        self.data.qvel[:2] = self.np_random.uniform(
            -TASK["reset_base_xy_velocity_uniform_m_s"], TASK["reset_base_xy_velocity_uniform_m_s"], 2
        )
        self.data.qvel[3:6] = self.np_random.uniform(
            -TASK["reset_base_angular_velocity_uniform_rad_s"], TASK["reset_base_angular_velocity_uniform_rad_s"], 3
        )
        # Shift the whole assembly vertically until its lowest foot is just clear,
        # avoiding reset-only floor penetration after randomizing joint angles.
        mujoco.mj_forward(self.model, self.data)
        min_sole = min(self.data.geom_xpos[g, 2] - self.model.geom_size[g, 0] for g in self.foot_geoms)
        self.data.qpos[2] += 0.0002 - min_sole
        # Reset clears upstream torque/friction state AND the 20 ms target queue.
        self.controller.reset()
        self.controller.set_targets(self.home_q)
        self.previous_action[:] = 0
        self.steps = 0
        self.initial_xy = self.data.qpos[:2].copy()
        return self.observation(), {}

    def loaded_bad_contacts(self):
        bad = []
        force = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            b1, b2 = self.model.geom_bodyid[[g1, g2]]
            if b1 == 0 and g2 in self.foot_geoms or b2 == 0 and g1 in self.foot_geoms:
                continue
            mujoco.mj_contactForce(self.model, self.data, i, force)
            if force[0] > TASK["loaded_contact_force_threshold_n"]:
                bad.append([self.model.geom(g1).name, self.model.geom(g2).name])
        return bad

    def pose_metrics(self):
        quat = self.data.qpos[3:7]
        tilt = float(np.arccos(np.clip(1 - 2 * (quat[1] ** 2 + quat[2] ** 2), -1, 1)))
        return float(self.data.qpos[2]), tilt, float(np.linalg.norm(self.data.qpos[:2] - self.initial_xy))

    def failure_reason(self, bad_contacts):
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
            return "nonfinite_state"
        height, tilt, drift = self.pose_metrics()
        if bad_contacts:
            return "loaded_self_or_nonfoot_ground_contact"
        if height < TASK["minimum_height_m"]:
            return "low_body"
        if tilt > TASK["maximum_tilt_rad"]:
            return "excessive_tilt"
        if drift > TASK["maximum_xy_displacement_m"]:
            return "excessive_drift"
        return None

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (12,) or not np.all(np.isfinite(action)):
            raise ValueError("Expected twelve finite normalized action values.")
        action = np.clip(action, -1, 1)
        self.controller.set_targets(self.home_q + TASK["action_scale_rad"] * action)
        bad_contacts, reason = [], None
        peak_torque = 0.0
        max_tilt = 0.0
        min_height = float(self.data.qpos[2])
        for _ in range(self.substeps):
            self.controller.step()
            bad_contacts = self.loaded_bad_contacts()
            reason = self.failure_reason(bad_contacts)
            peak_torque = max(peak_torque, float(np.max(np.abs(self.data.ctrl))))
            height, tilt, _ = self.pose_metrics()
            min_height, max_tilt = min(min_height, height), max(max_tilt, tilt)
            if reason:
                break
        mujoco.mj_forward(self.model, self.data)
        height, tilt, drift = self.pose_metrics()
        height_error = height - self.home_height
        rate = (
            1.5 * np.exp(-(height_error / 0.012) ** 2)
            + np.exp(-(tilt / 0.15) ** 2)
            + 0.5 * np.exp(-np.sum(self.data.qvel[:2] ** 2) / 0.02**2)
            + 0.5 * np.exp(-np.mean((self.data.qpos[self.controller.qadr] - self.home_q) ** 2) / 0.1**2)
            - 0.02 * np.mean((action - self.previous_action) ** 2)
            - 0.005 * np.mean((self.data.ctrl / 0.1) ** 2)
        )
        reward = TASK["policy_dt_s"] * float(rate) - (1.0 if reason else 0.0)
        self.previous_action = action.copy()
        self.steps += 1
        terminated = reason is not None
        truncated = self.steps >= self.max_steps and not terminated
        info = {
            "failure_reason": reason,
            "bad_contacts": bad_contacts,
            "height_m": height,
            "min_height_m": min_height,
            "height_error_m": height_error,
            "tilt_rad": tilt,
            "max_tilt_rad": max_tilt,
            "drift_m": drift,
            "peak_motor_torque_nm": peak_torque,
            "simulation_time_s": float(self.data.time),
            "is_success": bool(truncated),
        }
        return self.observation(), reward, terminated, truncated, info
