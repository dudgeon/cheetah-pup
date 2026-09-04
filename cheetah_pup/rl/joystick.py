"""Joystick task: track a body-frame velocity command (vx, vy, yaw rate) on flat ground.

Adapted from mujoco_playground's Go1 joystick task (Apache-2.0) for a 1.4 kg servo quadruped:
- Policy observations use only what the real robot senses: gyro, gravity direction from the IMU,
  joint positions/velocities, the last action, foot contacts, and the command. Base linear velocity
  is privileged (critic only) — there is no velocity estimate on the real robot.
- Actions are offsets from the standing pose, and motor targets are rate-limited to the STS3215
  firmware cap so the policy cannot ask for motions the servo will not perform.
- Reward scales are re-tuned for our torques (~1 N·m) and speeds (~0.15 m/s).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math

from mujoco_playground._src import mjx_env

from . import base
from . import constants as C
from ..servo import STS3215


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=1000,
        action_repeat=1,
        action_scale=0.3,                 # rad per unit action around the standing pose
        soft_joint_pos_limit_factor=0.95,
        max_motor_velocity=STS3215.max_velocity,  # rad/s, firmware target-rate cap
        impl="jax",                       # "warp" on an NVIDIA GPU
        naconmax=4 * 8192,
        njmax=40,
        noise_config=config_dict.create(
            level=1.0,
            scales=config_dict.create(joint_pos=0.02, joint_vel=1.5, gyro=0.2, gravity=0.05),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                tracking_lin_vel=1.5,
                tracking_ang_vel=0.75,
                lin_vel_z=-0.5,
                ang_vel_xy=-0.05,
                orientation=-5.0,
                dof_pos_limits=-1.0,
                pose=0.3,
                termination=-1.0,
                stand_still=-0.5,
                torques=-0.002,
                action_rate=-0.01,
                energy=-0.01,
                feet_clearance=-1.0,
                feet_height=-0.2,
                feet_slip=-0.1,
                feet_air_time=0.2,
            ),
            tracking_sigma=0.02,          # (m/s)^2 — commands are ~0.1-0.25 m/s
            max_foot_height=0.03,         # m, matches the design swing height
        ),
        pert_config=config_dict.create(
            enable=False,
            velocity_kick=[0.0, 0.5],
            kick_durations=[0.05, 0.2],
            kick_wait_times=[1.0, 3.0],
        ),
        command_config=config_dict.create(
            a=[0.25, 0.10, 1.0],          # |vx| (m/s), |vy| (m/s), |yaw rate| (rad/s)
            b=[0.9, 0.25, 0.5],           # probability each component is non-zero on resample
        ),
        termination_upvector_z=0.3,       # ~73° of tilt ends the episode
    )


class Joystick(base.CheetahPupEnv):
    """Track a joystick command."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
        xml_path: Optional[str] = None,
    ):
        super().__init__(config=config, config_overrides=config_overrides, xml_path=xml_path)
        self._post_init()

    def _post_init(self) -> None:
        key = self._mj_model.keyframe("home")
        self._init_q = jp.array(key.qpos)
        self._default_pose = jp.array(key.qpos[7:])
        self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
        c = (self._lowers + self._uppers) / 2
        r = self._uppers - self._lowers
        self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
        self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor
        self._cmd_a = jp.array(self._config.command_config.a)
        self._cmd_b = jp.array(self._config.command_config.b)

    def _make_data(self, qpos, qvel, ctrl):
        kwargs = {}
        if self._config.impl == "warp":
            kwargs = dict(naconmax=self._config.naconmax, njmax=self._config.njmax)
        return mjx_env.make_data(self.mj_model, qpos=qpos, qvel=qvel, ctrl=ctrl, impl=self._config.impl, **kwargs)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)

        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.2, maxval=0.2)
        qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
        quat = math.axis_angle_to_quat(jp.array([0.0, 0.0, 1.0]), yaw)
        qpos = qpos.at[3:7].set(math.quat_mul(qpos[3:7], quat))
        rng, key = jax.random.split(rng)
        qpos = qpos.at[7:].set(qpos[7:] + jax.random.uniform(key, (12,), minval=-0.1, maxval=0.1))
        rng, key = jax.random.split(rng)
        qvel = qvel.at[0:6].set(jax.random.uniform(key, (6,), minval=-0.2, maxval=0.2))

        data = self._make_data(qpos, qvel, self._default_pose)
        data = mjx.forward(self.mjx_model, data)

        rng, key1, key2, key3 = jax.random.split(rng, 4)
        time_until_next_pert = jax.random.uniform(
            key1, minval=self._config.pert_config.kick_wait_times[0], maxval=self._config.pert_config.kick_wait_times[1])
        pert_duration_seconds = jax.random.uniform(
            key2, minval=self._config.pert_config.kick_durations[0], maxval=self._config.pert_config.kick_durations[1])
        pert_mag = jax.random.uniform(
            key3, minval=self._config.pert_config.velocity_kick[0], maxval=self._config.pert_config.velocity_kick[1])

        rng, key1, key2 = jax.random.split(rng, 3)
        steps_until_next_cmd = jp.round(jax.random.exponential(key1) * 5.0 / self.dt).astype(jp.int32)
        cmd = jax.random.uniform(key2, shape=(3,), minval=-self._cmd_a, maxval=self._cmd_a)

        info = {
            "rng": rng,
            "command": cmd,
            "steps_until_next_cmd": steps_until_next_cmd,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "motor_targets": self._default_pose,
            "feet_air_time": jp.zeros(4),
            "last_contact": jp.zeros(4, dtype=bool),
            "swing_peak": jp.zeros(4),
            "steps_until_next_pert": jp.round(time_until_next_pert / self.dt).astype(jp.int32),
            "pert_duration_seconds": pert_duration_seconds,
            "pert_duration": jp.round(pert_duration_seconds / self.dt).astype(jp.int32),
            "steps_since_last_pert": 0,
            "pert_steps": 0,
            "pert_dir": jp.zeros(3),
            "pert_mag": pert_mag,
        }
        metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward_config.scales.keys()}
        metrics["swing_peak"] = jp.zeros(())
        obs = self._get_obs(data, info, self.get_feet_contact(data))
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        if self._config.pert_config.enable:
            state = self._maybe_apply_perturbation(state)

        motor_targets = self._default_pose + action * self._config.action_scale
        # Firmware slew limit: the STS3215 moves its internal target at most max_motor_velocity.
        prev = state.info["motor_targets"]
        slew = self._config.max_motor_velocity * self.dt
        motor_targets = jp.clip(motor_targets, prev - slew, prev + slew)
        motor_targets = jp.clip(motor_targets, self._lowers, self._uppers)
        data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)
        state.info["motor_targets"] = motor_targets

        contact = self.get_feet_contact(data)
        contact_filt = contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
        state.info["feet_air_time"] += self.dt
        p_fz = data.site_xpos[self._feet_site_id][..., -1]
        state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

        obs = self._get_obs(data, state.info, contact)
        done = self._get_termination(data)

        rewards = self._get_reward(data, action, state.info, done, first_contact, contact)
        rewards = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        state.info["steps_until_next_cmd"] -= 1
        state.info["rng"], key1, key2 = jax.random.split(state.info["rng"], 3)
        state.info["command"] = jp.where(
            state.info["steps_until_next_cmd"] <= 0, self.sample_command(key1, state.info["command"]), state.info["command"])
        state.info["steps_until_next_cmd"] = jp.where(
            done | (state.info["steps_until_next_cmd"] <= 0),
            jp.round(jax.random.exponential(key2) * 5.0 / self.dt).astype(jp.int32),
            state.info["steps_until_next_cmd"])
        state.info["feet_air_time"] *= ~contact
        state.info["last_contact"] = contact
        state.info["swing_peak"] *= ~contact
        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v
        state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

        done = done.astype(reward.dtype)
        return state.replace(data=data, obs=obs, reward=reward, done=done)

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        fell = self.get_upvector(data)[-1] < self._config.termination_upvector_z
        nan = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        return fell | nan

    def _noisy(self, info, x, scale):
        info["rng"], key = jax.random.split(info["rng"])
        return x + (2 * jax.random.uniform(key, shape=x.shape) - 1) * self._config.noise_config.level * scale

    def _get_obs(self, data: mjx.Data, info: dict[str, Any], contact: jax.Array) -> Dict[str, jax.Array]:
        s = self._config.noise_config.scales
        gyro = self.get_gyro(data)
        gravity = self.get_gravity(data)
        joint_angles = data.qpos[7:]
        joint_vel = data.qvel[6:]
        state = jp.hstack([
            self._noisy(info, gyro, s.gyro),                              # 3
            self._noisy(info, gravity, s.gravity),                        # 3
            info["command"],                                              # 3
            self._noisy(info, joint_angles, s.joint_pos) - self._default_pose,  # 12
            self._noisy(info, joint_vel, s.joint_vel) * 0.05,             # 12
            info["last_act"],                                             # 12
            contact.astype(jp.float32),                                   # 4
        ])
        linvel = self.get_local_linvel(data)
        privileged = jp.hstack([
            state,
            gyro, self.get_accelerometer(data), gravity, linvel, self.get_global_angvel(data),
            joint_angles - self._default_pose, joint_vel,
            data.actuator_force,                                          # 12
            info["last_contact"].astype(jp.float32),
            self.get_feet_linvel(data).ravel(),                           # 12
            info["feet_air_time"],
            data.qpos[2:3],                                               # height
        ])
        return {"state": state, "privileged_state": privileged}

    def _get_reward(self, data, action, info, done, first_contact, contact) -> dict[str, jax.Array]:
        cfg = self._config.reward_config
        cmd = info["command"]
        cmd_norm = jp.linalg.norm(cmd)
        local_vel = self.get_local_linvel(data)
        gyro = self.get_gyro(data)
        qpos = data.qpos[7:]
        feet_vel = self.get_feet_linvel(data)
        vel_xy_sq = jp.sum(jp.square(feet_vel[..., :2]), axis=-1)
        foot_z = data.site_xpos[self._feet_site_id][..., -1]
        out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0) + jp.clip(qpos - self._soft_uppers, 0.0, None)
        return {
            "tracking_lin_vel": jp.exp(-jp.sum(jp.square(cmd[:2] - local_vel[:2])) / cfg.tracking_sigma),
            "tracking_ang_vel": jp.exp(-jp.square(cmd[2] - gyro[2]) / (cfg.tracking_sigma * 10.0)),
            "lin_vel_z": jp.square(self.get_global_linvel(data)[2]),
            "ang_vel_xy": jp.sum(jp.square(self.get_global_angvel(data)[:2])),
            "orientation": jp.sum(jp.square(self.get_upvector(data)[:2])),
            "dof_pos_limits": jp.sum(out_of_limits),
            "pose": jp.exp(-jp.sum(jp.square(qpos - self._default_pose) * jp.array([1.0, 0.5, 0.2] * 4))),
            "termination": done,
            "stand_still": jp.sum(jp.abs(qpos - self._default_pose)) * (cmd_norm < 0.01),
            "torques": jp.sqrt(jp.sum(jp.square(data.actuator_force))) + jp.sum(jp.abs(data.actuator_force)),
            "action_rate": jp.sum(jp.square(action - info["last_act"])),
            "energy": jp.sum(jp.abs(data.qvel[6:]) * jp.abs(data.actuator_force)),
            "feet_slip": jp.sum(vel_xy_sq * contact) * (cmd_norm > 0.01),
            "feet_clearance": jp.sum(jp.abs(foot_z - cfg.max_foot_height) * jp.sqrt(jp.sqrt(vel_xy_sq))),
            "feet_height": jp.sum(jp.square(info["swing_peak"] / cfg.max_foot_height - 1.0) * first_contact) * (cmd_norm > 0.01),
            "feet_air_time": jp.sum((info["feet_air_time"] - 0.1) * first_contact) * (cmd_norm > 0.01),
        }

    def _maybe_apply_perturbation(self, state: mjx_env.State) -> mjx_env.State:
        def gen_dir(rng):
            angle = jax.random.uniform(rng, minval=0.0, maxval=jp.pi * 2)
            return jp.array([jp.cos(angle), jp.sin(angle), 0.0])

        def apply_pert(state):
            t = state.info["pert_steps"] * self.dt
            u_t = 0.5 * jp.sin(jp.pi * t / state.info["pert_duration_seconds"])
            force = u_t * self._torso_mass * state.info["pert_mag"] / state.info["pert_duration_seconds"]
            xfrc = jp.zeros((self.mjx_model.nbody, 6)).at[self._torso_body_id, :3].set(force * state.info["pert_dir"])
            state = state.replace(data=state.data.replace(xfrc_applied=xfrc))
            state.info["steps_since_last_pert"] = jp.where(
                state.info["pert_steps"] >= state.info["pert_duration"], 0, state.info["steps_since_last_pert"])
            state.info["pert_steps"] += 1
            return state

        def wait(state):
            state.info["rng"], rng = jax.random.split(state.info["rng"])
            state.info["steps_since_last_pert"] += 1
            xfrc = jp.zeros((self.mjx_model.nbody, 6))
            due = state.info["steps_since_last_pert"] >= state.info["steps_until_next_pert"]
            state.info["pert_steps"] = jp.where(due, 0, state.info["pert_steps"])
            state.info["pert_dir"] = jp.where(due, gen_dir(rng), state.info["pert_dir"])
            return state.replace(data=state.data.replace(xfrc_applied=xfrc))

        return jax.lax.cond(
            state.info["steps_since_last_pert"] >= state.info["steps_until_next_pert"], apply_pert, wait, state)

    def sample_command(self, rng: jax.Array, x_k: jax.Array) -> jax.Array:
        rng, y_rng, w_rng, z_rng = jax.random.split(rng, 4)
        y_k = jax.random.uniform(y_rng, shape=(3,), minval=-self._cmd_a, maxval=self._cmd_a)
        z_k = jax.random.bernoulli(z_rng, self._cmd_b, shape=(3,))
        w_k = jax.random.bernoulli(w_rng, 0.5, shape=(3,))
        return x_k - w_k * (x_k - y_k * z_k)
