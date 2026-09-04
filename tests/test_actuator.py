"""Integration gates that distinguish BAM torque motors from ideal PD models."""

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from cheetah_pup.actuator import (
    ACTUATOR_NAMES,
    BamPositionController,
    load_actuator_config,
    motor_mjcf,
    published_model,
    upstream_parity,
    validate_actuator_config,
)
from cheetah_pup.model import build_mjcf, load_config

ROOT = Path(__file__).resolve().parents[1]


def robot():
    return load_config(ROOT / "config/robot.json")


def controller(delay_s=0.0):
    model = mujoco.MjModel.from_xml_string(motor_mjcf(build_mjcf(robot())))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
    config = load_actuator_config()
    config["command_delay_s"] = delay_s
    return model, data, BamPositionController(model, data, config)


@pytest.mark.parametrize("value", [0, 3.7, 6.0, 7.5, float("nan")])
def test_default_or_wrong_rail_is_never_accepted(value):
    config = load_actuator_config()
    config["supply_v"] = value
    with pytest.raises(ValueError, match="5.0 V"):
        validate_actuator_config(config)


def test_explicit_electrical_settings_and_current_limiter_semantics():
    config = load_actuator_config()
    model = published_model(config)
    act = model.actuator
    assert (act.vin, act.kp, act.max_pwm, act.max_current) == (5.0, 400, 1.0, 1.75)
    voltage = act.compute_control(100.0, 0.0, 0.0, 0.002)
    assert act.compute_torque(voltage, True, 0.0, 0.0) == pytest.approx(
        model.kt.value * 1.75
    )
    # At extreme back-driving speed the battery cannot realize the current window.
    # The physical PWM bound takes precedence; post hoc torque clipping is wrong.
    voltage = act.compute_control(100.0, 0.0, 100.0, 0.002)
    assert voltage == 5.0
    assert abs(act.compute_torque(voltage, True, 0.0, 100.0)) > model.kt.value * 1.75


def test_conversion_removes_pd_clamps_and_duplicate_friction():
    original_xml = build_mjcf(robot())
    before = mujoco.MjModel.from_xml_string(original_xml)
    xml = motor_mjcf(original_xml)
    model = mujoco.MjModel.from_xml_string(xml)
    joints = [model.joint(name).id for name in ACTUATOR_NAMES]
    dofs = model.jnt_dofadr[joints]
    assert model.nu == 12
    assert [model.actuator(i).name for i in range(12)] == list(ACTUATOR_NAMES)
    assert np.array_equal(model.body_mass, before.body_mass)
    assert np.array_equal(model.body_inertia, before.body_inertia)
    assert np.array_equal(model.jnt_range, before.jnt_range)
    assert np.all(model.dof_damping[dofs] == 0)
    assert np.all(model.dof_frictionloss[dofs] == 0)
    assert np.all(model.dof_armature[dofs] == 0)
    assert np.all(model.key_ctrl == 0)
    assert not np.any(model.actuator_ctrllimited)
    assert not np.any(model.actuator_forcelimited)
    assert not np.any(model.jnt_actfrclimited[joints])
    assert all(e.tag == "motor" for e in ET.fromstring(xml).find("actuator"))


def test_constructor_preserves_keyframe_and_uses_only_upstream_armature():
    model = mujoco.MjModel.from_xml_string(motor_mjcf(build_mjcf(robot())))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
    data.time = 1.5
    data.qvel[6:] = 0.02
    q, dq = data.qpos.copy(), data.qvel.copy()
    control = BamPositionController(model, data)
    assert np.array_equal(data.qpos, q)
    assert np.array_equal(data.qvel, dq)
    assert data.time == 1.5
    assert control.upstream.last_ts == 1.5
    assert np.all(model.dof_armature[control.dadr] == control.bam_model.armature.value)


def test_controller_rejects_hidden_pd_gain_or_prior_damping():
    model, data, _ = controller()
    model.actuator_gainprm[0, 0] = 1.5
    with pytest.raises(ValueError, match="unit-gear"):
        BamPositionController(model, data)
    model.actuator_gainprm[0, 0] = 1.0
    with pytest.raises(ValueError, match="prior damping/friction/armature"):
        BamPositionController(model, data)


def test_command_queue_is_physics_step_delay_and_does_not_alias_targets():
    model, data, control = controller(0.02)
    assert control.delay_steps == round(0.02 / model.opt.timestep)
    initial = control.targets.copy()
    target = initial.copy()
    target[0] += 0.1
    control.set_targets(target)
    target[0] += 0.1
    for _ in range(control.delay_steps):
        control.step()
        assert np.array_equal(control.applied_targets, initial)
    control.step()
    assert control.applied_targets[0] == pytest.approx(initial[0] + 0.1)


def test_reset_removes_old_commands_and_replays_identically():
    model, data, control = controller(0.02)

    def run():
        target = control.targets.copy()
        target[0] += 0.05
        control.set_targets(target)
        for _ in range(80):
            control.step()
        return data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()

    first = run()
    mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
    control.reset()
    assert control.upstream.last_ts == 0
    for actual, expected in zip(run(), first):
        np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=0)


def test_rejects_nonfinite_incomplete_or_out_of_range_targets_and_bad_clock():
    model, data, control = controller()
    for target in ([0.0] * 3, [float("nan")] * 12, {"FL_hip_roll": 0.0}, [100.0] * 12):
        with pytest.raises(ValueError):
            control.set_targets(target)
    control.update()
    with pytest.raises(RuntimeError, match="once per physics step"):
        control.update()


def test_adapter_matches_direct_upstream_dynamics():
    report = upstream_parity()
    assert report["passed"], report
    assert report["hardware_log_comparison"] is False
