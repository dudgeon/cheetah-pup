"""Pinned upstream BAM CPU adapter, with explicit stock-voltage assumptions.

This executes upstream motor and friction equations. Numerical parity with BAM
does not establish fidelity to a stock XL330-M288-T at 5 V.
"""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Mapping, Sequence
import copy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .kinematics import JOINT_ORDER, LEG_ORDER

BAM_COMMIT = "62bd8ce12154340be97e06f7f41a0ca8f116d967"
ROOT = Path(__file__).resolve().parents[2]
ACTUATOR_NAMES = tuple(f"{leg}_{joint}" for leg in LEG_ORDER for joint in JOINT_ORDER)


def validate_actuator_config(config: dict) -> None:
    if (
        config["candidate"] != "ROBOTIS XL330-M288-T"
        or config["bam_commit"] != BAM_COMMIT
    ):
        raise ValueError(
            "This adapter requires the documented XL330 candidate and exact BAM pin."
        )
    if config["parameter_file"] != "bam/params/xl330/m6.json":
        raise ValueError("This adapter requires the reviewed upstream XL330 M6 fit.")
    # An explicit regulated 5 V contract prevents accidental reuse of 7.5 V defaults.
    if config["supply_v"] != 5.0:
        raise ValueError(
            "This candidate requires an explicit regulated 5.0 V simulation rail."
        )
    gain = config["firmware_p_gain"]
    if isinstance(gain, bool) or not isinstance(gain, int) or not 1 <= gain <= 16383:
        raise ValueError(
            "firmware_p_gain must be an integer control-table value in [1, 16383]."
        )
    for name, upper in (("max_pwm", 1.0), ("max_current_a", 1.75)):
        value = config[name]
        if not np.isfinite(value) or not 0 < value <= upper:
            raise ValueError(f"{name} must be finite and in (0, {upper}].")
    delay = config["command_delay_s"]
    if not np.isfinite(delay) or not 0 <= delay <= 0.1:
        raise ValueError("command_delay_s must be finite and in [0, 0.1].")


def load_actuator_config(path: str | Path | None = None) -> dict:
    config = json.loads(Path(path or ROOT / "config/actuator.json").read_text())
    validate_actuator_config(config)
    return config


def _load_upstream(config: dict):
    """Load the actual pinned source, rejecting mixed/modified BAM installations."""
    validate_actuator_config(config)
    path = (ROOT / config["bam_submodule"]).resolve()
    if not (path / "bam/model.py").is_file():
        raise RuntimeError(
            "Initialize the CPU model with: git submodule update --init vendor/bam_microduck"
        )

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(path), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    if git("rev-parse", "HEAD") != BAM_COMMIT:
        raise RuntimeError(f"BAM source must be pinned to {BAM_COMMIT}.")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError(
            "BAM submodule has tracked modifications; published-source parity is unknown."
        )
    if "bam" not in sys.modules:
        sys.path.insert(0, str(path))
    module = importlib.import_module("bam.model")
    if Path(module.__file__).resolve() != path / "bam/model.py":
        raise RuntimeError(
            "Another BAM installation is already loaded; use a fresh Python process."
        )
    controller_module = importlib.import_module("bam.mujoco")
    return module, controller_module, path


def published_model(config: dict | None = None):
    """Return a new upstream model with every electrical/controller setting explicit."""
    config = config or load_actuator_config()
    module, _, path = _load_upstream(config)
    model = module.load_model(str(path / config["parameter_file"]))
    model.actuator.vin = float(config["supply_v"])
    model.actuator.kp = config["firmware_p_gain"]
    model.actuator.max_pwm = float(config["max_pwm"])
    model.actuator.max_current = float(config["max_current_a"])
    # q_offset is the fitted testbench's zero error, not a robot joint offset.
    # The upstream CPU controller does not apply it; assembly zeros live elsewhere.
    return model


def motor_mjcf(xml: str, joint_names: Sequence[str] = ACTUATOR_NAMES) -> str:
    """Replace this workbench's ideal PD actuators with named unit-gear motors.

    Joint limits and physical geometry are preserved. BAM owns friction, damping
    and reflected rotor inertia; no old ideal-PD torque cap survives. Keyframe
    controls become zero torques; targets must be set through the controller.
    """
    names = tuple(joint_names)
    root = ET.fromstring(xml)
    actuators = root.find("actuator")
    if not names or len(set(names)) != len(names) or actuators is None:
        raise ValueError(
            "Provide unique controlled joint names and an actuator section."
        )
    entries = list(actuators)
    if len(entries) != len(names) or set(e.get("joint") for e in entries) != set(names):
        raise ValueError("MJCF must contain exactly one actuator per requested joint.")
    joints = {j.get("name"): j for j in root.findall(".//worldbody//joint")}
    if not set(names) <= joints.keys():
        raise ValueError("A controlled hinge joint is missing from the MJCF.")
    for name in names:
        joint = joints[name]
        if joint.get("type", "hinge") != "hinge":
            raise ValueError("BAM adapter supports hinge joints only.")
        for attr in ("damping", "frictionloss", "armature", "stiffness"):
            joint.set(attr, "0")
        joint.set("actuatorfrclimited", "false")
    actuators.clear()
    for name in names:
        ET.SubElement(
            actuators,
            "motor",
            name=name,
            joint=name,
            gear="1",
            ctrllimited="false",
            forcelimited="false",
        )
    for key in root.findall("./keyframe/key"):
        key.set("ctrl", " ".join("0" for _ in names))
    root.insert(
        0,
        ET.Comment(
            " BAM motor adapter. Published fit reused at 5 V; hardware fidelity unverified. "
        ),
    )
    ET.indent(root)
    return ET.tostring(root, encoding="unicode") + "\n"


class BamPositionController:
    """12 position targets -> optional command queue -> upstream torque/friction.

    Call set_targets at the desired command rate; call step at every physics
    step. Alternatively call update once immediately before each mj_step.
    After any mj_resetData* call, call reset before stepping again.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: dict | None = None,
        joint_names: Sequence[str] = ACTUATOR_NAMES,
    ):
        self.config = copy.deepcopy(config or load_actuator_config())
        validate_actuator_config(self.config)
        self.model, self.data, self.names = model, data, tuple(joint_names)
        self.dt = float(model.opt.timestep)
        lag = self.config["command_delay_s"] / self.dt
        if not np.isclose(lag, round(lag), atol=1e-9, rtol=0):
            raise ValueError(
                "command_delay_s must be an integer number of physics timesteps."
            )
        self.delay_steps = int(round(lag))
        self.act_ids = np.array([model.actuator(n).id for n in self.names])
        self.joint_ids = np.array([model.joint(n).id for n in self.names])
        if len(set(self.names)) != len(self.names) or len(self.names) != model.nu:
            raise ValueError("This adapter must own all actuators, once each.")
        ids = self.act_ids
        gear = np.zeros((len(ids), 6))
        gear[:, 0] = 1
        gains = np.zeros((len(ids), 10))
        gains[:, 0] = 1
        if (
            np.any(model.jnt_type[self.joint_ids] != mujoco.mjtJoint.mjJNT_HINGE)
            or np.any(model.actuator_trntype[ids] != mujoco.mjtTrn.mjTRN_JOINT)
            or not np.array_equal(model.actuator_trnid[ids, 0], self.joint_ids)
            or not np.array_equal(model.actuator_gear[ids], gear)
            or not np.array_equal(model.actuator_gainprm[ids], gains)
            or np.any(model.actuator_gaintype[ids] != mujoco.mjtGain.mjGAIN_FIXED)
            or np.any(model.actuator_biastype[ids] != mujoco.mjtBias.mjBIAS_NONE)
            or np.any(model.actuator_dyntype[ids] != mujoco.mjtDyn.mjDYN_NONE)
            or np.any(model.actuator_ctrllimited[ids])
            or np.any(model.actuator_forcelimited[ids])
            or np.any(model.jnt_actfrclimited[self.joint_ids])
        ):
            raise ValueError(
                "Use motor_mjcf(): BAM requires unclamped unit-gear torque motors."
            )
        self.qadr = model.jnt_qposadr[self.joint_ids]
        self.dadr = model.jnt_dofadr[self.joint_ids]
        if any(
            np.any(v[self.dadr])
            for v in (model.dof_damping, model.dof_frictionloss, model.dof_armature)
        ):
            raise ValueError(
                "Remove prior damping/friction/armature before attaching BAM."
            )
        self.bam_model = published_model(self.config)
        _, module, _ = _load_upstream(self.config)
        # Upstream calls mj_setConst with this MjData, which resets qpos to qpos0.
        # Preserve the caller's physical state (e.g. the stand keyframe).
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(model, state_spec))
        mujoco.mj_getState(model, data, state, state_spec)
        self.upstream = module.MujocoController(self.bam_model, self.names, model, data)
        mujoco.mj_setState(model, data, state, state_spec)
        self.reset()

    def set_targets(self, targets: Sequence[float] | Mapping[str, float]) -> None:
        if isinstance(targets, Mapping):
            if set(targets) != set(self.names):
                raise ValueError(
                    "Target mapping must name each controlled joint exactly once."
                )
            targets = [targets[n] for n in self.names]
        values = np.asarray(targets, dtype=float)
        if values.shape != (len(self.names),) or not np.all(np.isfinite(values)):
            raise ValueError("Targets must be one finite radian value per joint.")
        limits = self.model.jnt_range[self.joint_ids]
        bounded = self.model.jnt_limited[self.joint_ids].astype(bool)
        if np.any(bounded & ((values < limits[:, 0]) | (values > limits[:, 1]))):
            raise ValueError("A target exceeds the modeled joint limits.")
        self.targets = values.copy()

    def reset(self) -> None:
        """Clear command history and previous-step friction after a simulation reset."""
        self.bam_model.reset()
        self.upstream.reset(self.data.qpos)
        self.upstream.last_ts = (
            self.data.time
        )  # upstream reset does not reset this timestamp
        self.targets = self.data.qpos[self.qadr].copy()
        self.applied_targets = self.targets.copy()
        self._queue = deque(self.targets.copy() for _ in range(self.delay_steps))
        self._last_update_time = None
        self.model.dof_frictionloss[self.dadr] = 0
        self.model.dof_damping[self.dadr] = 0
        self.data.ctrl[self.act_ids] = 0
        mujoco.mj_forward(self.model, self.data)

    def update(self) -> None:
        if not np.isclose(self.model.opt.timestep, self.dt, rtol=0, atol=1e-15):
            raise RuntimeError(
                "Rebuild the controller when the physics timestep changes."
            )
        if self._last_update_time is not None and not np.isclose(
            self.data.time - self._last_update_time, self.dt, rtol=0, atol=1e-10
        ):
            raise RuntimeError(
                "Update once per physics step; call reset after resetting MuJoCo."
            )
        self._queue.append(self.targets.copy())
        self.applied_targets = self._queue.popleft()
        self.upstream.q_target = self.applied_targets.copy()
        self.upstream.update()
        self._last_update_time = float(self.data.time)

    def step(self) -> None:
        self.update()
        mujoco.mj_step(self.model, self.data)


def torque_speed_screen(config: dict | None = None) -> dict:
    """Electrical torque envelope from published coefficients, excluding friction/heat."""
    config = config or load_actuator_config()
    model = published_model(config)
    speeds = np.array([0.0, 2.0, 5.0, 8.0, 10.0, 12.0, 16.0])
    q = np.zeros_like(speeds)
    voltage = model.actuator.compute_control(np.full_like(q, 100.0), q, speeds, 0.002)
    torques = model.actuator.compute_torque(voltage, True, q, speeds)
    return {
        "scope": "Saturated positive position error; upstream motor-side torque before gearbox friction. No continuous rating or battery/thermal prediction.",
        "speed_rad_s": speeds.tolist(),
        "voltage_v": voltage.tolist(),
        "motor_torque_nm": torques.tolist(),
        "fitted_kt_nm_a": model.kt.value,
        "fitted_resistance_ohm": model.R.value,
        "fitted_armature_kg_m2": model.armature.value,
        "zero_speed_current_limited_torque_nm": float(torques[0]),
        "ideal_no_load_speed_before_friction_rad_s": config["supply_v"]
        / model.kt.value,
    }


def upstream_parity(config: dict | None = None) -> dict:
    """Check adapter vs direct upstream on a deterministic loaded pendulum.

    This is a software integration check, not a comparison with a measured log.
    Both instances have identical rigid bodies and separate upstream models.
    """
    config = copy.deepcopy(config or load_actuator_config())
    config["command_delay_s"] = 0
    xml = """<mujoco><compiler angle="radian"/><option timestep="0.002" integrator="implicitfast"/>
    <worldbody><body><joint name="hinge" type="hinge" limited="true" range="-2 2"/>
    <geom type="capsule" fromto="0 0 0 0 0 -0.09" size="0.005" mass="0.012"/>
    <geom type="sphere" pos="0 0 -0.09" size="0.012" mass="0.04"/>
    </body></worldbody><actuator><motor name="hinge" joint="hinge" gear="1"/>
    </actuator></mujoco>"""
    models = [mujoco.MjModel.from_xml_string(xml) for _ in range(2)]
    data = [mujoco.MjData(m) for m in models]
    for m, d in zip(models, data):
        d.qpos[0] = 0.2
        mujoco.mj_forward(m, d)
    wrapper = BamPositionController(models[0], data[0], config, ("hinge",))
    direct_model = published_model(config)
    _, module, _ = _load_upstream(config)
    direct = module.MujocoController(direct_model, ("hinge",), models[1], data[1])
    # Direct BAM also needs its desired initial state restored after mj_setConst.
    data[1].qpos[0] = 0.2
    mujoco.mj_forward(models[1], data[1])
    direct.reset(data[1].qpos)
    errors = {
        "qpos_rad": 0.0,
        "qvel_rad_s": 0.0,
        "torque_nm": 0.0,
        "frictionloss_nm": 0.0,
        "damping_nm_s_per_rad": 0.0,
    }
    for i in range(1500):
        target = 0.30 * np.sin(2 * np.pi * 0.7 * i * 0.002) + (0.12 if i >= 750 else 0)
        wrapper.set_targets([target])
        wrapper.step()
        direct.set_q_target("hinge", target)
        direct.update()
        mujoco.mj_step(models[1], data[1])
        for name, left, right in (
            ("qpos_rad", data[0].qpos, data[1].qpos),
            ("qvel_rad_s", data[0].qvel, data[1].qvel),
            ("torque_nm", data[0].ctrl, data[1].ctrl),
            ("frictionloss_nm", models[0].dof_frictionloss, models[1].dof_frictionloss),
            ("damping_nm_s_per_rad", models[0].dof_damping, models[1].dof_damping),
        ):
            errors[name] = max(errors[name], float(np.max(np.abs(left - right))))
    return {
        "method": "Adapter vs directly invoked pinned upstream controller; gravity-loaded 9cm pendulum, sinusoid then offset, no command delay.",
        "steps": 1500,
        "duration_s": 3.0,
        "max_abs_error": errors,
        "passed": all(e <= 1e-12 for e in errors.values()),
        "hardware_log_comparison": False,
    }


def stand_smoke(
    robot_config: dict, actuator_config: dict | None = None, seconds: float = 5.0
) -> dict:
    """Forward dynamics at constant stand targets; report failures without hiding them."""
    from .model import build_mjcf

    xml = motor_mjcf(build_mjcf(robot_config))
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
    controller = BamPositionController(model, data, actuator_config)
    target = data.qpos[controller.qadr].copy()
    initial_height = float(data.qpos[2])
    minimum_height = initial_height
    peak_torque = peak_velocity = max_error = 0.0
    maximum_tilt = 0.0
    self_contact_pairs = set()
    nonfoot_ground_pairs = set()
    foot_geoms = {model.geom(f"{leg}_foot_collision").id: leg for leg in LEG_ORDER}
    torque_squared = np.zeros(len(controller.names))
    final_foot_loads = dict.fromkeys(LEG_ORDER, 0.0)
    completed_steps = 0
    finite = True
    for _ in range(round(seconds / model.opt.timestep)):
        controller.set_targets(target)
        controller.step()
        finite = bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)))
        if not finite:
            break
        minimum_height = min(minimum_height, float(data.qpos[2]))
        peak_torque = max(peak_torque, float(np.max(np.abs(data.ctrl))))
        torque_squared += data.ctrl[controller.act_ids] ** 2
        completed_steps += 1
        peak_velocity = max(
            peak_velocity, float(np.max(np.abs(data.qvel[controller.dadr])))
        )
        max_error = max(
            max_error, float(np.max(np.abs(data.qpos[controller.qadr] - target)))
        )
        quat = data.qpos[3:7]
        tilt = np.arccos(np.clip(1 - 2 * (quat[1] ** 2 + quat[2] ** 2), -1, 1))
        maximum_tilt = max(maximum_tilt, float(tilt))
        final_foot_loads = dict.fromkeys(LEG_ORDER, 0.0)
        for i in range(data.ncon):
            contact = data.contact[i]
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            if force[0] <= 1e-6:
                continue
            g1, g2 = int(contact.geom1), int(contact.geom2)
            b1, b2 = model.geom_bodyid[[g1, g2]]
            names = tuple(sorted((model.geom(g1).name, model.geom(g2).name)))
            if b1 > 0 and b2 > 0:
                self_contact_pairs.add(names)
            elif (b1 == 0) != (b2 == 0):
                robot_geom = g1 if b1 > 0 else g2
                if robot_geom in foot_geoms:
                    final_foot_loads[foot_geoms[robot_geom]] += float(force[0])
                else:
                    nonfoot_ground_pairs.add(names)
    rms_torque = np.sqrt(torque_squared / max(completed_steps, 1))
    rms_current = rms_torque / controller.bam_model.kt.value
    copper_proxy = controller.bam_model.R.value * rms_current**2
    # These deliberately explicit screening bounds prevent height alone from
    # being called a successful stand when the body has tilted or crept down.
    within_pose_screen = (
        finite
        and initial_height - minimum_height <= 0.01
        and maximum_tilt <= np.deg2rad(10)
        and max_error <= 0.2
        and not self_contact_pairs
        and not nonfoot_ground_pairs
    )
    return {
        "method": "Free-base MuJoCo forward dynamics, fixed stand targets, pinned BAM at5V. No prescribed qpos during stepping and no trained policy.",
        "robot_config_sha256": hashlib.sha256(
            json.dumps(robot_config, sort_keys=True).encode()
        ).hexdigest(),
        "mjcf_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "actuator_config_sha256": hashlib.sha256(
            json.dumps(controller.config, sort_keys=True).encode()
        ).hexdigest(),
        "supply_v": controller.config["supply_v"],
        "firmware_p_gain": controller.config["firmware_p_gain"],
        "duration_s": float(data.time),
        "finite": finite,
        "initial_base_height_m": initial_height,
        "minimum_base_height_m": minimum_height,
        "final_base_height_m": float(data.qpos[2]),
        "final_base_quaternion_wxyz": data.qpos[3:7].tolist(),
        "peak_abs_motor_torque_nm": peak_torque,
        "peak_abs_joint_speed_rad_s": peak_velocity,
        "rms_motor_torque_nm": dict(zip(controller.names, map(float, rms_torque))),
        "rms_winding_current_model_a": dict(
            zip(controller.names, map(float, rms_current))
        ),
        "mean_winding_copper_loss_proxy_w": dict(
            zip(controller.names, map(float, copper_proxy))
        ),
        "power_proxy_scope": "Kt/R-model winding I²R only, from applied motor torque. Excludes electronics, driver losses, regeneration, gearbox frictional heat and thermal paths. Not battery input power, allowable sustained operation, or runtime.",
        "maximum_joint_tracking_error_rad": max_error,
        "maximum_base_tilt_deg": float(np.rad2deg(maximum_tilt)),
        "final_joint_speed_rad_s": dict(
            zip(controller.names, map(float, data.qvel[controller.dadr]))
        ),
        "final_solve_foot_normal_load_n": final_foot_loads,
        "loaded_self_contact_pairs": sorted(self_contact_pairs),
        "loaded_nonfoot_ground_pairs": sorted(nonfoot_ground_pairs),
        "pose_screen_bounds": {
            "maximum_height_loss_m": 0.01,
            "maximum_tilt_deg": 10,
            "maximum_joint_tracking_error_rad": 0.2,
            "loaded_self_or_nonfoot_ground_contact_allowed": False,
        },
        "within_pose_screen": bool(within_pose_screen),
        "delay_s": controller.config["command_delay_s"],
        "delay_steps": controller.delay_steps,
        "hardware_fidelity_validated": False,
        "thermal_validated": False,
    }


def crawl_replay_smoke(robot_config: dict, actuator_config: dict | None = None) -> dict:
    """Replay periodic joint targets with real free-base dynamics for two cycles.

    The reference generator uses a separate kinematic model. Live simulation
    qpos is initialized once from reference frame zero, with zero velocity.
    After that, only twelve position targets enter the live controller; neither
    the base nor any joint is prescribed through qpos while physics runs.
    """
    from .gait_demo import trajectory
    from .model import build_mjcf

    reference_model, frames, metadata = trajectory(robot_config)
    positions = np.array([f["qpos"] for f in frames])
    xml = motor_mjcf(build_mjcf(robot_config))
    model = mujoco.MjModel.from_xml_string(xml)
    if model.nq != reference_model.nq:
        raise ValueError("Gait reference and dynamic model have different coordinates.")
    data = mujoco.MjData(model)
    data.qpos[:] = positions[0]  # the sole prescribed live pose, before initialization
    controller = BamPositionController(model, data, actuator_config)
    for name in controller.names:
        if reference_model.joint(name).qposadr[0] != model.joint(name).qposadr[0]:
            raise ValueError("Gait reference and dynamic joint addresses differ.")
    samples = positions[:, controller.qadr]
    sample_hash = hashlib.sha256(samples.astype("<f8").tobytes()).hexdigest()
    period, cycles, command_dt = float(metadata["cycle_duration_s"]), 2, 0.02
    dt = float(model.opt.timestep)
    decimation = round(command_dt / dt)
    steps = round(cycles * period / dt)
    if not np.isclose(decimation * dt, command_dt) or not np.isclose(
        steps * dt, cycles * period
    ):
        raise ValueError(
            "Command cadence and replay duration must be integral physics steps."
        )
    initial_position = data.qpos[:3].copy()
    minimum_height = float(data.qpos[2])
    maximum_tilt = peak_torque = peak_speed = maximum_error = 0.0
    error_squared = np.zeros(len(controller.names))
    self_pairs, ground_pairs = set(), set()
    first_nonfoot_contact_time = None
    pwm_bound_steps = voltage_clipped_steps = command_updates = completed_steps = 0
    finite = True
    targets = samples[0].copy()
    for step in range(steps):
        if step % decimation == 0:
            # Periodic interpolation includes the last-to-first sample seam.
            phase = (step * dt % period) * len(samples) / period
            index = int(np.floor(phase)) % len(samples)
            fraction = phase - np.floor(phase)
            targets = (1 - fraction) * samples[index] + fraction * samples[
                (index + 1) % len(samples)
            ]
            controller.set_targets(targets)
            command_updates += 1
        controller.update()
        act = controller.bam_model.actuator
        q, dq = data.qpos[controller.qadr], data.qvel[controller.dadr]
        voltage = act.compute_control(controller.applied_targets, q, dq, dt)
        requested = act.vin * act.kp * act.error_gain * (controller.applied_targets - q)
        pwm_bound_steps += int(np.any(np.abs(voltage) >= act.vin * act.max_pwm - 1e-9))
        voltage_clipped_steps += int(np.any(np.abs(voltage - requested) > 1e-9))
        mujoco.mj_step(model, data)  # no qpos, qvel or base overwrite in this loop
        completed_steps += 1
        finite = bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)))
        if not finite:
            break
        error = data.qpos[controller.qadr] - targets
        error_squared += error**2
        maximum_error = max(maximum_error, float(np.max(np.abs(error))))
        minimum_height = min(minimum_height, float(data.qpos[2]))
        quat = data.qpos[3:7]
        maximum_tilt = max(
            maximum_tilt,
            float(np.arccos(np.clip(1 - 2 * (quat[1] ** 2 + quat[2] ** 2), -1, 1))),
        )
        peak_torque = max(peak_torque, float(np.max(np.abs(data.ctrl))))
        peak_speed = max(peak_speed, float(np.max(np.abs(data.qvel[controller.dadr]))))
        for i in range(data.ncon):
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            if force[0] <= 1e-6:
                continue
            g1, g2 = int(data.contact[i].geom1), int(data.contact[i].geom2)
            b1, b2 = model.geom_bodyid[[g1, g2]]
            names = tuple(sorted((model.geom(g1).name, model.geom(g2).name)))
            if b1 > 0 and b2 > 0:
                self_pairs.add(names)
            elif (b1 == 0) != (b2 == 0):
                name = model.geom(g1 if b1 > 0 else g2).name
                if name not in {f"{leg}_foot_collision" for leg in LEG_ORDER}:
                    ground_pairs.add(names)
                    if first_nonfoot_contact_time is None:
                        first_nonfoot_contact_time = float(data.time)
    actual_progress = data.qpos[:3] - initial_position
    desired_progress = cycles * float(metadata["stride_per_cycle_m"])
    bounds = {
        "minimum_forward_progress_fraction": 0.5,
        "maximum_forward_progress_error_m": 0.02,
        "maximum_lateral_drift_m": 0.02,
        "maximum_height_loss_m": 0.01,
        "maximum_base_tilt_deg": 10.0,
        "maximum_joint_tracking_error_rad": 0.2,
        "loaded_self_or_nonfoot_ground_contact_allowed": False,
    }
    checks = {
        "finite_and_complete": finite and completed_steps == steps,
        "forward_progress": actual_progress[0]
        >= desired_progress * bounds["minimum_forward_progress_fraction"],
        "progress_error": abs(actual_progress[0] - desired_progress)
        <= bounds["maximum_forward_progress_error_m"],
        "lateral_drift": abs(actual_progress[1]) <= bounds["maximum_lateral_drift_m"],
        "height": initial_position[2] - minimum_height
        <= bounds["maximum_height_loss_m"],
        "tilt": np.rad2deg(maximum_tilt) <= bounds["maximum_base_tilt_deg"],
        "tracking": maximum_error <= bounds["maximum_joint_tracking_error_rad"],
        "contacts": not self_pairs and not ground_pairs,
    }
    return {
        "method": "Free-base forward dynamics; two prescribed crawl cycles, 50Hz linearly interpolated periodic joint targets, pinned BAM and a single command-delay queue. No learned policy.",
        "initialization": "Live qpos assigned once from reference frame0; qvel starts atzero. After controller initialization only targets and MuJoCo integration change physical state.",
        "manual_pose_overwrites_during_replay": 0,
        "robot_config_sha256": hashlib.sha256(
            json.dumps(robot_config, sort_keys=True).encode()
        ).hexdigest(),
        "mjcf_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "actuator_config_sha256": hashlib.sha256(
            json.dumps(controller.config, sort_keys=True).encode()
        ).hexdigest(),
        "joint_reference_samples_sha256_float64_le": sample_hash,
        "joint_reference_samples": len(samples),
        "cycle_duration_s": period,
        "cycles": cycles,
        "requested_duration_s": cycles * period,
        "duration_s": float(data.time),
        "physics_timestep_s": dt,
        "command_rate_hz": 1 / command_dt,
        "command_updates": command_updates,
        "command_delay_s": controller.config["command_delay_s"],
        "command_delay_steps": controller.delay_steps,
        "firmware_p_gain": controller.config["firmware_p_gain"],
        "supply_v": controller.config["supply_v"],
        "finite": finite,
        "desired_forward_progress_m": desired_progress,
        "actual_base_progress_xyz_m": actual_progress.tolist(),
        "initial_base_position_m": initial_position.tolist(),
        "final_base_position_m": data.qpos[:3].tolist(),
        "minimum_base_height_m": minimum_height,
        "maximum_base_tilt_deg": float(np.rad2deg(maximum_tilt)),
        "peak_abs_motor_torque_nm": peak_torque,
        "peak_abs_joint_speed_rad_s": peak_speed,
        "maximum_joint_tracking_error_rad": maximum_error,
        "rms_joint_tracking_error_rad": dict(
            zip(
                controller.names,
                map(float, np.sqrt(error_squared / max(completed_steps, 1))),
            )
        ),
        "tracking_reference": "Latest commanded 50Hz target, before the command-delay queue; includes intended latency.",
        "fraction_physics_steps_any_motor_at_pwm_bound": pwm_bound_steps
        / max(completed_steps, 1),
        "fraction_physics_steps_any_motor_voltage_clipped": voltage_clipped_steps
        / max(completed_steps, 1),
        "saturation_scope": "Upstream electrical-model PWM/current-window intervention, not a measured thermal or hardware limit.",
        "loaded_self_contact_pairs": sorted(self_pairs),
        "loaded_nonfoot_ground_pairs": sorted(ground_pairs),
        "first_nonfoot_ground_contact_time_s": first_nonfoot_contact_time,
        "screening_bounds": bounds,
        "screening_checks": {k: bool(v) for k, v in checks.items()},
        "failed_checks": [k for k, v in checks.items() if not v],
        "passed_screen": bool(all(checks.values())),
        "learned_policy": False,
        "hardware_fidelity_validated": False,
        "thermal_validated": False,
    }


def validation_report(robot_config: dict, actuator_config: dict | None = None) -> dict:
    config = actuator_config or load_actuator_config()
    _, _, path = _load_upstream(config)
    comparison_config = copy.deepcopy(config)
    comparison_config["firmware_p_gain"] = (
        200 if config["firmware_p_gain"] != 200 else 400
    )
    comparison_config["gain_notes"] = (
        "Diagnostic comparison only; all physical parameters and other controller settings held fixed."
    )
    return {
        "status": "CPU integration tested; actuator selection and sim-to-real gate remain open.",
        "bam_commit": BAM_COMMIT,
        "mujoco_version": mujoco.__version__,
        "config": config,
        "parameter_sha256": hashlib.sha256(
            (path / config["parameter_file"]).read_bytes()
        ).hexdigest(),
        "upstream_parity": upstream_parity(config),
        "torque_speed_screen": torque_speed_screen(config),
        "forward_stand_smoke": stand_smoke(robot_config, config),
        "forward_stand_60s": stand_smoke(robot_config, config, seconds=60.0),
        "firmware_gain_comparison_60s": {
            "varied_setting": "firmware_p_gain",
            "baseline_gain": config["firmware_p_gain"],
            "baseline_result_key": "forward_stand_60s",
            "compared_gain": comparison_config["firmware_p_gain"],
            "compared_result": stand_smoke(
                robot_config, comparison_config, seconds=60.0
            ),
            "scope": "P200 is the Microduck reference; P400 is the XL330 manual and BAM constructor default. This compares software assumptions, not measured firmware tuning.",
        },
        "forward_crawl_replay": crawl_replay_smoke(robot_config, config),
        "hardware_calibration_verified": False,
        "learned_policy": False,
    }


def main():
    from .model import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-config", default=str(ROOT / "config/robot.json"))
    parser.add_argument("--actuator-config", default=str(ROOT / "config/actuator.json"))
    parser.add_argument(
        "--output", default=str(ROOT / "reports/actuator-validation.json")
    )
    args = parser.parse_args()
    report = validation_report(
        load_config(args.robot_config), load_actuator_config(args.actuator_config)
    )
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    fields = (
        "firmware_p_gain",
        "duration_s",
        "minimum_base_height_m",
        "maximum_base_tilt_deg",
        "peak_abs_motor_torque_nm",
        "maximum_joint_tracking_error_rad",
        "within_pose_screen",
    )
    cases = (
        report["forward_stand_smoke"],
        report["forward_stand_60s"],
        report["firmware_gain_comparison_60s"]["compared_result"],
    )
    print(
        json.dumps(
            {
                "output": args.output,
                "parity": report["upstream_parity"]["passed"],
                "stand_cases": [
                    {name: case[name] for name in fields} for case in cases
                ],
                "crawl_replay": {
                    name: report["forward_crawl_replay"][name]
                    for name in (
                        "desired_forward_progress_m",
                        "actual_base_progress_xyz_m",
                        "failed_checks",
                    )
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
