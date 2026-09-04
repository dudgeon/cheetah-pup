"""Check static balance independently of the robot model's geometry."""
import numpy as np
import pytest

from cheetah_pup.analysis import solve_vertical_support, static_support_screen
from cheetah_pup.model import load_config
from pathlib import Path


def test_rectangle_four_feet_share_centered_load():
    forces = solve_vertical_support([[1, 1], [1, -1], [-1, 1], [-1, -1]], [0, 0], 12)
    np.testing.assert_allclose(forces, [3, 3, 3, 3], atol=1e-12)


def test_three_contacts_on_rectangle_do_not_imply_equal_sharing():
    forces = solve_vertical_support([[1, -1], [-1, 1], [-1, -1]], [0, 0], 12)
    np.testing.assert_allclose(forces, [6, 6, 0], atol=1e-12)


def test_shifted_com_can_share_three_loads():
    forces = solve_vertical_support([[1, -1], [-1, 1], [-1, -1]], [-1 / 3, -1 / 3], 12)
    np.testing.assert_allclose(forces, [4, 4, 4], atol=1e-12)


def test_outside_support_triangle_is_rejected():
    assert solve_vertical_support([[1, -1], [-1, 1], [-1, -1]], [0.2, 0.2], 12) is None


def test_diagonal_support_requires_com_on_line():
    np.testing.assert_allclose(solve_vertical_support([[1, 1], [-1, -1]], [0, 0], 12), [6, 6])
    assert solve_vertical_support([[1, 1], [-1, -1]], [0.1, 0], 12) is None


def test_model_report_does_not_treat_three_feet_as_automatic_margin():
    config = load_config(Path(__file__).resolve().parents[1] / "config/robot.json")
    result = static_support_screen(config)
    cases = {c["case"]: c for c in result["cases"]}
    assert cases["four_feet"]["vertical_static_equilibrium"]
    assert cases["four_feet"]["meets_proposed_1_5_static_margin"]
    for lifted in ("FL", "FR"):
        case = cases[f"three_feet_lift_{lifted}"]
        assert case["vertical_static_equilibrium"]
        assert not case["meets_proposed_1_5_static_margin"]
        assert sum(case["vertical_foot_load_n"].values()) == pytest.approx(0.613 * 9.81)
    assert not cases["three_feet_lift_RL"]["vertical_static_equilibrium"]
