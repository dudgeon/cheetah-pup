import math

import pytest

from cheetah_pup.analysis import metrics, mass_model, torque_report, speed_report, packaging_report
from cheetah_pup.design import PRESETS, preset
from cheetah_pup.gait import GAITS, joint_trajectories, foot_trajectory, LEGS
from cheetah_pup.servo import STS3215


@pytest.mark.parametrize("key", list(PRESETS))
def test_presets_are_plausible(key):
    p = preset(key, "M")
    m = metrics(p)
    assert 1.0 < m["mass"] < 2.2, m["mass"]
    assert abs(m["com"][0]) < 0.02 and abs(m["com"][1]) < 0.005
    assert m["packaging"]["battery_fits"] and m["packaging"]["top_fits"] and m["packaging"]["height_fits"]
    assert m["torque"]["stand"]["ok"], m["torque"]["stand"]
    assert m["torque"]["trot_peak"]["ok"], m["torque"]["trot_peak"]
    assert m["speed"]["ok"], m["speed"]


def test_belt_reduction_lowers_knee_servo_torque():
    b = preset("B", "M")
    a = torque_report(preset("A", "M", stance_height=b.stance_height), 1.5)
    bb = torque_report(b, 1.5)
    assert bb["trot_peak"]["knee_joint"] == pytest.approx(a["trot_peak"]["knee_joint"])
    assert bb["trot_peak"]["knee_servo"] == pytest.approx(a["trot_peak"]["knee_servo"] / b.knee_ratio)


def test_knee_servo_speed_scales_with_ratio():
    b = preset("B", "M")
    a = speed_report(preset("A", "M", stance_height=b.stance_height))
    assert speed_report(b)["knee_servo"] == pytest.approx(a["knee_servo"] * b.knee_ratio)


def test_transverse_pi_fits_where_longitudinal_does_not():
    from cheetah_pup.analysis import _footprint
    from cheetah_pup.electronics import PI5
    assert _footprint(PI5, 0.070, 0.093) == (0.056, 0.085, True)
    assert _footprint(PI5, 0.070, 0.080) is None


def test_gait_feet_are_periodic_and_stance_is_flat():
    p = preset("A", "M")
    for gait in GAITS:
        for leg in LEGS:
            f0 = foot_trajectory(p, gait, leg, 0.0)
            f1 = foot_trajectory(p, gait, leg, 1.0 - 1e-9)
            assert f0 == pytest.approx(f1, abs=1e-3)
    traj = joint_trajectories(p, "trot", n=60)
    ground = -p.stance_height - p.stance_depth
    for leg in LEGS:
        stance_z = [f[2] for f in traj[leg]["foot"] if f[2] <= ground + 1e-9]
        assert len(stance_z) >= 25
        assert max(traj[leg]["foot"], key=lambda f: f[2])[2] > ground + 0.02


def test_size_presets_scale_geometry():
    s, m, l = (preset("B", k) for k in ("S", "M", "L"))
    assert s.thigh < m.thigh < l.thigh
    assert metrics(s)["mass"] < metrics(l)["mass"]


def test_servo_model_is_more_optimistic_than_datasheet():
    assert STS3215.model_stall_torque() > STS3215.stall_torque
    assert STS3215.available_torque(STS3215.max_velocity) < STS3215.available_torque(0.0)
