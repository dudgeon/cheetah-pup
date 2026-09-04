"""Validate the generated model in MuJoCo: settle the stand pose, then play gaits open-loop.

    python -m cheetah_pup.validate sim/cheetah_pup.xml --out sim/validation

Open-loop means the IK joint targets from `gait.py` are sent to the position servos at 50 Hz with
no feedback from the body — the same targets the review page animates. A model that stands and
trots forward open-loop is a strong viability signal; the RL policy only has to do better.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

import mujoco
import numpy as np

from .design import locked
from .gait import LEGS, LEG_SIDE, LEG_FRONT, GAITS, foot_trajectory, body_speed
from .kinematics import leg_ik
from .mjcf import CONTROL_HZ, joint_names, servo_gains
from .servo import STS3215


def quat_to_rpy(q):
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def gait_targets(p, gait, t, z_offsets=None):
    """Joint targets (12, in actuator order) at time t for the gait, or None if unreachable.

    `z_offsets` (per leg, metres) is added to each foot's height target — used by the leveling
    baseline to lengthen legs under a low corner of the body.
    """
    freq = p.stride_frequency * GAITS[gait]["freq"]
    phase = (t * freq) % 1.0
    out = []
    for i, leg in enumerate(LEGS):
        x, y, z = foot_trajectory(p, gait, leg, phase)
        if z_offsets is not None:
            z += z_offsets[i]
        try:
            qa, qh, qk = leg_ik(p.thigh, p.shank, p.abad_link, (x, y, z), LEG_SIDE[leg], p.knee_sign(LEG_FRONT[leg]))
        except ValueError:
            return None
        out += [qa, qh, qk]
    return np.array(out)


def leveling_offsets(p, data, gain):
    """Foot-height corrections that push a pitched/rolled body back to level.

    Nose-down (pitch > 0 about +y) lowers the front hips, so the front feet target higher (shorter
    legs) and the rear feet lower. Positive roll about +x lifts the left side, so left feet target
    lower. This is a pure geometric leveling term, not a balance controller.
    """
    roll, pitch, _ = quat_to_rpy(data.qpos[3:7])
    out = []
    for leg in LEGS:
        hx = (1 if LEG_FRONT[leg] else -1) * p.hip_to_hip / 2
        hy = LEG_SIDE[leg] * (p.abad_to_abad / 2 + p.abad_link)
        out.append(gain * (hx * math.sin(pitch) - hy * math.sin(roll)))
    return out


class Recorder:
    def __init__(self, model):
        self.model = model
        self.rows = []

    def record(self, data, ctrl):
        r, pch, yaw = quat_to_rpy(data.qpos[3:7])
        contacts = [float(data.sensor(f"{leg}_touch").data[0] > 1e-4) for leg in LEGS]
        self.rows.append({
            "t": round(float(data.time), 4),
            "pos": [round(float(v), 5) for v in data.qpos[0:3]],
            "rpy": [round(r, 4), round(pch, 4), round(yaw, 4)],
            "q": [round(float(v), 4) for v in data.qpos[7:19]],
            "dq": [round(float(v), 3) for v in data.qvel[6:18]],
            "ctrl": [round(float(v), 4) for v in ctrl],
            "tau": [round(float(v), 4) for v in data.actuator_force],
            "contact": contacts,
        })


def run(model, data, p, gait, seconds, rec=None, settle=1.0, level_gain=0.0):
    """Play `gait` for `seconds` after `settle` seconds of standing. Returns stats.

    level_gain = 0 is pure open-loop; > 0 adds the geometric leveling term (1.0 = full correction).
    """
    dt_ctrl = 1.0 / CONTROL_HZ
    substeps = int(round(dt_ctrl / model.opt.timestep))
    stand = model.key("stand")
    mujoco.mj_resetDataKeyframe(model, data, stand.id)
    stand_ctrl = np.array(stand.ctrl)
    t0 = None
    peak_tau = np.zeros(12)
    peak_dq = np.zeros(12)
    heights, pitches, rolls = [], [], []
    fell = False
    n_ctrl = int(round((settle + seconds) * CONTROL_HZ))
    start_x = None
    for i in range(n_ctrl):
        t = i * dt_ctrl
        if t < settle:
            ctrl = stand_ctrl
        else:
            if t0 is None:
                t0 = t
                start_x = float(data.qpos[0])
            offsets = leveling_offsets(p, data, level_gain) if level_gain > 0 else None
            tgt = gait_targets(p, gait, t - t0, offsets)
            ctrl = tgt if tgt is not None else stand_ctrl
        data.ctrl[:] = ctrl
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        if rec is not None:
            rec.record(data, ctrl)
        if t >= settle:
            peak_tau = np.maximum(peak_tau, np.abs(data.actuator_force))
            peak_dq = np.maximum(peak_dq, np.abs(data.qvel[6:18]))
            r, pch, _ = quat_to_rpy(data.qpos[3:7])
            heights.append(float(data.qpos[2])); pitches.append(pch); rolls.append(r)
            if data.qpos[2] < 0.5 * p.stance_height or abs(pch) > 0.8 or abs(r) > 0.8:
                fell = True
                break
    dist = float(data.qpos[0]) - (start_x or 0.0)
    elapsed = max(1e-6, data.time - settle)
    by_type = lambda arr, k: float(max(arr[i] for i in range(12) if i % 3 == k))
    contact = np.array([r["contact"] for r in rec.rows if r["t"] >= settle]) if rec is not None and rec.rows else None
    return {
        "gait": gait,
        "level_gain": level_gain,
        "seconds": round(elapsed, 2),
        "contact_duty": [round(float(v), 3) for v in contact.mean(axis=0)] if contact is not None and len(contact) else None,
        "fell": fell,
        "distance_m": round(dist, 4),
        "mean_speed_mps": round(dist / elapsed, 4),
        "commanded_speed_mps": round(body_speed(p, gait), 4),
        "height_min": round(min(heights), 4) if heights else None,
        "height_max": round(max(heights), 4) if heights else None,
        "pitch_max_deg": round(math.degrees(max(abs(v) for v in pitches)), 1) if pitches else None,
        "roll_max_deg": round(math.degrees(max(abs(v) for v in rolls)), 1) if rolls else None,
        "peak_torque_Nm": {"abad": by_type(peak_tau, 0), "hip": by_type(peak_tau, 1), "knee": by_type(peak_tau, 2)},
        "peak_speed_rads": {"abad": by_type(peak_dq, 0), "hip": by_type(peak_dq, 1), "knee": by_type(peak_dq, 2)},
        "lateral_drift_m": round(float(data.qpos[1]), 4),
        "yaw_drift_deg": round(math.degrees(quat_to_rpy(data.qpos[3:7])[2]), 1),
    }


def stand_test(model, data, p, seconds=2.0):
    stand = model.key("stand")
    mujoco.mj_resetDataKeyframe(model, data, stand.id)
    data.ctrl[:] = stand.ctrl
    substeps = int(round(1.0 / CONTROL_HZ / model.opt.timestep))
    for _ in range(int(seconds * CONTROL_HZ)):
        for _ in range(substeps):
            mujoco.mj_step(model, data)
    r, pch, _ = quat_to_rpy(data.qpos[3:7])
    tau = np.abs(data.actuator_force)
    target = p.stance_height + p.foot_radius
    return {
        "settled_height_m": round(float(data.qpos[2]), 4),
        "target_height_m": target,
        "sag_mm": round((target - float(data.qpos[2])) * 1000, 1),
        "pitch_deg": round(math.degrees(pch), 2),
        "roll_deg": round(math.degrees(r), 2),
        "hold_torque_Nm": {"abad": round(float(max(tau[0::3])), 3), "hip": round(float(max(tau[1::3])), 3), "knee": round(float(max(tau[2::3])), 3)},
        "joint_error_deg": round(math.degrees(float(np.max(np.abs(data.qpos[7:19] - stand.ctrl)))), 2),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="sim/cheetah_pup.xml")
    ap.add_argument("--out", default="sim/validation")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--gaits", default="stand,walk,trot")
    ap.add_argument("--modes", default="openloop,leveled", help="openloop (gain 0) and/or leveled (gain 1)")
    args = ap.parse_args(argv)
    p = locked()
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": args.model,
        "design": p.name,
        "total_mass_kg": round(float(sum(model.body_mass)), 4),
        "nq": model.nq, "nv": model.nv, "nu": model.nu,
        "servo": servo_gains("datasheet"),
        "control_hz": CONTROL_HZ,
        "timestep": model.opt.timestep,
        "stand": stand_test(model, data, p),
        "runs": {},
    }
    print(f"model mass {summary['total_mass_kg']:.3f} kg, nq={model.nq} nv={model.nv} nu={model.nu}")
    print("stand:", json.dumps(summary["stand"]))
    # 0.7 is the sweep's best speed/attitude compromise; higher gains oscillate (no rate damping)
    gains = {"openloop": 0.0, "leveled": 0.7}
    for mode in args.modes.split(","):
        (out / mode).mkdir(exist_ok=True)
        summary["runs"][mode] = {}
        for gait in args.gaits.split(","):
            rec = Recorder(model)
            stats = run(model, data, p, gait, args.seconds, rec, level_gain=gains[mode])
            summary["runs"][mode][gait] = stats
            (out / mode / f"{gait}.json").write_text(json.dumps({"stats": stats, "joints": joint_names(), "rows": rec.rows}))
            brief = {k: stats[k] for k in ("fell", "distance_m", "mean_speed_mps", "commanded_speed_mps", "pitch_max_deg", "roll_max_deg", "contact_duty")}
            print(f"{mode}/{gait}:", json.dumps(brief), "peak tau", {k: round(v, 2) for k, v in stats["peak_torque_Nm"].items()})
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {out}/summary.json")


if __name__ == "__main__":
    main()
