import math

import pytest

from cheetah_pup.kinematics import planar_fk, planar_ik, leg_fk, leg_ik, planar_jacobian, static_torques
from cheetah_pup.design import MINI_CHEETAH


def test_mini_cheetah_stand_pose_round_trips():
    l1, l2 = MINI_CHEETAH["thigh"], MINI_CHEETAH["shank"]
    (kx, kz), (fx, fz) = planar_fk(l1, l2, -0.8, 1.6)
    assert kx < 0, "cheetah configuration puts the knee behind the hip"
    assert abs(fx) < 0.02, "foot lands roughly under the hip"
    assert 0.27 < -fz < 0.30, "stand height matches the published 0.29 m"
    q_hip, q_knee = planar_ik(l1, l2, fx, fz, knee_sign=1)
    assert q_hip == pytest.approx(-0.8, abs=1e-9)
    assert q_knee == pytest.approx(1.6, abs=1e-9)


def test_planar_ik_knee_sign_selects_configuration():
    l1, l2 = 0.09, 0.085
    for sign in (1, -1):
        q_hip, q_knee = planar_ik(l1, l2, 0.02, -0.12, knee_sign=sign)
        (kx, _), (fx, fz) = planar_fk(l1, l2, q_hip, q_knee)
        assert fx == pytest.approx(0.02, abs=1e-9)
        assert fz == pytest.approx(-0.12, abs=1e-9)
        assert (kx < fx) == (sign == 1)


def test_planar_ik_rejects_unreachable():
    with pytest.raises(ValueError):
        planar_ik(0.09, 0.085, 0.0, -0.2)


def test_leg_fk_ik_round_trip_both_sides():
    l1, l2, la = 0.09, 0.085, 0.04
    for side in (1, -1):
        for qa, qh, qk in ((0.0, -0.6, 1.3), (0.3, 0.2, 1.0), (-0.25, -0.9, 1.8)):
            pos = leg_fk(l1, l2, la, qa, qh, qk, side)
            got = leg_ik(l1, l2, la, pos["foot"], side, knee_sign=1)
            assert got == pytest.approx((qa, qh, qk), abs=1e-9)


def test_abduction_moves_foot_outward_for_both_sides():
    l1, l2, la = 0.09, 0.085, 0.04
    for side in (1, -1):
        neutral = leg_fk(l1, l2, la, 0.0, -0.7, 1.4, side)["foot"]
        abducted = leg_fk(l1, l2, la, 0.3, -0.7, 1.4, side)["foot"]
        assert side * abducted[1] > side * neutral[1]
        assert neutral[1] == pytest.approx(side * la)


def test_jacobian_matches_finite_difference():
    l1, l2 = 0.09, 0.085
    q = (-0.6, 1.3)
    J = planar_jacobian(l1, l2, *q)
    eps = 1e-6
    for j in range(2):
        dq = [0.0, 0.0]
        dq[j] = eps
        _, f_plus = planar_fk(l1, l2, q[0] + dq[0], q[1] + dq[1])
        _, f_minus = planar_fk(l1, l2, q[0] - dq[0], q[1] - dq[1])
        assert (f_plus[0] - f_minus[0]) / (2 * eps) == pytest.approx(J[0][j], abs=1e-6)
        assert (f_plus[1] - f_minus[1]) / (2 * eps) == pytest.approx(J[1][j], abs=1e-6)


def test_static_torques_foot_under_hip():
    l1, l2, la = 0.09, 0.085, 0.04
    q_hip, q_knee = planar_ik(l1, l2, 0.0, -0.12)
    ta, th, tk = static_torques(l1, l2, la, q_hip, q_knee, 10.0)
    assert th == pytest.approx(0.0, abs=1e-9), "no hip moment with the foot under the hip"
    assert ta == pytest.approx(10.0 * la)
    (kx, _), _ = planar_fk(l1, l2, q_hip, q_knee)
    assert tk == pytest.approx(10.0 * (0.0 - kx))
    assert tk > 0
