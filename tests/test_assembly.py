"""Independent physical-frame and interference regression checks."""

import mujoco
import numpy as np
import pytest
from cheetah_pup.analysis import standing_model
from cheetah_pup.assembly_audit import (
    box_separation,
    geometry_gap,
    pairs,
    shaft_alignment,
)
from cheetah_pup.model import load_config


@pytest.fixture(scope="module")
def assembly():
    return standing_model(load_config("config/robot.json"))


def test_real_case_dimensions_and_off_center_shaft(assembly):
    model, data = assembly
    base = data.body("base").xpos
    # Source drawing:23mm casing depth+3mmhorn; center7.5mm belowshaft.
    expected = {
        "FL": [0.060 - 0.0145, 0.035, -0.0075],
        "FR": [0.060 - 0.0145, -0.035, -0.0075],
        "RL": [-0.060 + 0.0145, 0.035, -0.0075],
        "RR": [-0.060 + 0.0145, -0.035, -0.0075],
    }
    for leg, center in expected.items():
        case = model.geom(f"{leg}_roll_motor_envelope").id
        np.testing.assert_allclose(data.geom_xpos[case] - base, center, atol=1e-12)
        np.testing.assert_allclose(
            2 * model.geom_size[case], [0.023, 0.020, 0.034], atol=1e-12
        )
        horn = model.geom(f"{leg}_roll_output_horn").id
        assert model.geom_size[horn, 0] == pytest.approx(0.008)
        assert 2 * model.geom_size[horn, 1] == pytest.approx(0.003)


def test_all_twelve_joint_axes_match_physical_shafts_and_signs(assembly):
    model, data = assembly
    rows = shaft_alignment(model, data)
    assert len(rows) == 12
    for r in rows:
        assert r["axis_alignment_abs_cosine"] == pytest.approx(1, abs=1e-12)
        assert r["shaft_axis_offset_m"] < 1e-12
        leg, role = r["joint"].split("_", 1)
        expected = (
            (1 if leg.startswith("F") else -1)
            if role == "hip_roll"
            else (1 if leg.endswith("L") else -1)
        )
        assert r["physical_positive_to_joint_sign"] == expected


def test_case_in_parent_horn_in_child_and_keepouts_not_physical(assembly):
    model, _ = assembly
    for leg in ("FL", "FR", "RL", "RR"):
        for role, parent, child in [
            ("roll", "base", f"{leg}_roll_link"),
            ("hip", f"{leg}_roll_link", f"{leg}_upper_link"),
            ("knee", f"{leg}_upper_link", f"{leg}_lower_link"),
        ]:
            case = model.geom(f"{leg}_{role}_motor_envelope").id
            assert model.geom_bodyid[case] == model.body(parent).id
            assert (
                model.geom_bodyid[model.geom(f"{leg}_{role}_output_horn").id]
                == model.body(child).id
            )
            assert model.geom_contype[case] == 1 and model.geom_conaffinity[case] & 1
            for port in ("a", "b"):
                keepout = model.geom(f"{leg}_{role}_port_keepout_{port}").id
                assert model.geom_group[keepout] == 5
                assert (
                    model.geom_contype[keepout] == model.geom_conaffinity[keepout] == 0
                )
    contact_pairs = {
        frozenset((model.geom(i).name, model.geom(j).name))
        for i, j in zip(model.pair_geom1, model.pair_geom2)
    }
    assert frozenset(("FL_hip_motor_envelope", "FL_upper_bar")) in contact_pairs
    audited = {
        frozenset((model.geom(i).name, model.geom(j).name)) for i, j, _ in pairs(model)
    }
    assert frozenset(("battery_allowance", "compute_allowance")) in audited
    assert frozenset(("battery_allowance", "chassis_bottom")) in audited


def test_sat_known_separated_touching_overlapping_and_rotated_boxes():
    eye = np.eye(3)
    zero = np.zeros(3)
    half = np.ones(3)
    assert box_separation(
        zero, eye, half, np.array([3, 0, 0]), eye, half
    ) == pytest.approx(1)
    assert box_separation(
        zero, eye, half, np.array([2, 0, 0]), eye, half
    ) == pytest.approx(0)
    assert box_separation(
        zero, eye, half, np.array([1.5, 0, 0]), eye, half
    ) == pytest.approx(-0.5)
    theta = np.pi / 4
    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    assert box_separation(
        zero, eye, half, np.array([3, 0, 0]), rot, half
    ) == pytest.approx(2 - np.sqrt(2))
    assert box_separation(zero, eye, half, np.array([2, 0, 0]), rot, half) < 0


def test_remote_box_pair_cannot_report_penetration(assembly):
    model, data = assembly
    a = model.geom("FL_knee_motor_envelope").id
    b = model.geom("FR_knee_cheek_front").id
    assert geometry_gap(model, data, a, b) > 0.05
