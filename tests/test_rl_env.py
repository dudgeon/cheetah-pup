"""CPU checks of the RL environment (JAX/MJX). Slow-ish: the first reset/step JIT takes a minute."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("mujoco_playground")

from cheetah_pup.rl import joystick


@pytest.fixture(scope="module")
def env():
    return joystick.Joystick(config_overrides={"impl": "jax"})


def test_observation_and_action_sizes(env):
    sizes = env.observation_size
    assert env.action_size == 12
    assert sizes["state"] == (3 + 3 + 3 + 12 + 12 + 12 + 4,)
    assert sizes["privileged_state"][0] > sizes["state"][0]


def test_reset_and_step_are_finite(env):
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    state = reset(jax.random.PRNGKey(0))
    assert np.all(np.isfinite(np.asarray(state.obs["state"])))
    for _ in range(5):
        state = step(state, jax.numpy.zeros(env.action_size))
    assert np.all(np.isfinite(np.asarray(state.obs["state"])))
    assert float(state.reward) >= 0.0
    assert float(state.done) == 0.0
    assert float(state.data.qpos[2]) > 0.08  # still standing on the default pose


def test_motor_targets_are_slew_limited(env):
    step = jax.jit(env.step)
    state = jax.jit(env.reset)(jax.random.PRNGKey(1))
    state = step(state, jax.numpy.ones(env.action_size))
    delta = np.abs(np.asarray(state.info["motor_targets"]) - np.asarray(env._default_pose))
    assert np.all(delta <= env._config.max_motor_velocity * env.dt + 1e-6)


def test_fall_terminates(env):
    state = jax.jit(env.reset)(jax.random.PRNGKey(2))
    data = state.data.replace(qpos=state.data.qpos.at[3:7].set(jax.numpy.array([0.0, 1.0, 0.0, 0.0])))
    from mujoco import mjx
    data = mjx.forward(env.mjx_model, data)
    assert bool(env._get_termination(data))
