import json
import math

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from cheetah_pup.analysis import mass_model
from cheetah_pup.design import locked, preset
from cheetah_pup.gait import LEGS, LEG_SIDE, LEG_FRONT
from cheetah_pup.kinematics import leg_fk
from cheetah_pup.mjcf import CAD_PROPS, build_mjcf, cad_available, servo_gains, joint_names

HAS_CAD = cad_available(locked())


@pytest.fixture(scope="module", params=["primitive", "cad"])
def model(request):
    """Both fidelity levels of the locked design; the CAD one is what the sim and RL env use."""
    if request.param == "cad":
        if not HAS_CAD:
            pytest.skip("CAD exports missing or stale: run `python -m cad.assembly`")
        return mujoco.MjModel.from_xml_string(build_mjcf(locked(), cad=True))
    return mujoco.MjModel.from_xml_string(build_mjcf(locked(), cad=False))


def _is_cad(model):
    return model.nmesh > 0


def test_model_loads_with_expected_dofs(model):
    assert model.nq == 7 + 12 and model.nv == 6 + 12 and model.nu == 12
    assert [model.actuator(i).name for i in range(12)] == joint_names()


def test_model_mass_matches_source(model):
    p = locked()
    if _is_cad(model):
        cad = json.loads(CAD_PROPS.read_text())
        assert float(sum(model.body_mass)) == pytest.approx(cad["total_mass"], rel=1e-3)
        # the parametric sizing estimate and the CAD should agree to a few percent
        assert cad["total_mass"] == pytest.approx(mass_model(p)["total"], rel=0.05)
    else:
        assert float(sum(model.body_mass)) == pytest.approx(mass_model(p)["total"], rel=0.02)


def test_cad_model_uses_cad_inertials_and_meshes():
    if not HAS_CAD:
        pytest.skip("CAD exports missing or stale")
    cad = json.loads(CAD_PROPS.read_text())
    model = mujoco.MjModel.from_xml_string(build_mjcf(locked(), cad=True))
    assert model.nmesh == 2 + 4 * 3 + 1   # trunk tub/lid, bracket/thigh/shank per leg, servo
    for body, props in cad["bodies"].items():
        b = model.body(body)
        assert float(b.mass[0]) == pytest.approx(props["mass"], rel=1e-4), body
        assert b.ipos == pytest.approx(props["com"], abs=1e-5), body
        # principal moments must be those of the CAD's inertia tensor
        I = np.array([[props["fullinertia"][0], props["fullinertia"][3], props["fullinertia"][4]],
                      [props["fullinertia"][3], props["fullinertia"][1], props["fullinertia"][5]],
                      [props["fullinertia"][4], props["fullinertia"][5], props["fullinertia"][2]]])
        assert sorted(b.inertia) == pytest.approx(sorted(np.linalg.eigvalsh(I)), rel=1e-3), body
    # the RL variant carries the same inertials with feet as the only robot collision geoms
    rl = mujoco.MjModel.from_xml_string(build_mjcf(locked(), rl=True, cad=True))
    assert float(sum(rl.body_mass)) == pytest.approx(cad["total_mass"], rel=1e-3)
    colliding = [rl.geom(i).name for i in range(rl.ngeom) if rl.geom_contype[i] or rl.geom_conaffinity[i]]
    assert sorted(colliding) == sorted(["floor"] + [f"{leg}_foot" for leg in LEGS])


def test_forward_kinematics_matches_library(model):
    """Joint conventions in the MJCF reproduce kinematics.leg_fk for every leg."""
    p = locked()
    foot_y = p.foot_y_offset if _is_cad(model) else 0.0
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
    q = {"LF": (0.2, -0.6, 1.3), "RF": (-0.1, 0.3, 1.0), "LH": (0.3, -0.9, 1.8), "RH": (0.15, -0.4, 0.9)}
    for i, leg in enumerate(LEGS):
        data.qpos[7 + 3 * i: 10 + 3 * i] = q[leg]
    mujoco.mj_kinematics(model, data)
    for leg in LEGS:
        side, front = LEG_SIDE[leg], LEG_FRONT[leg]
        hx = (1 if front else -1) * p.hip_to_hip / 2
        ay = side * p.abad_to_abad / 2
        expect = np.array(leg_fk(p.thigh, p.shank, p.abad_link, *q[leg], side)["foot"])
        # the foot sits `foot_y_offset` outboard in the knee frame, which rolls with the abad joint
        roll = side * q[leg][0]
        expect += side * foot_y * np.array([0.0, math.cos(roll), math.sin(roll)])
        got = data.site(f"{leg}_foot").xpos - np.array([0, 0, p.stance_height + p.foot_radius])
        assert got == pytest.approx([hx + expect[0], ay + expect[1], expect[2]], abs=1e-6), leg


def test_stand_keyframe_holds(model):
    p = locked()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
    data.ctrl[:] = model.key("stand").ctrl
    for _ in range(1000):
        mujoco.mj_step(model, data)
    assert data.qpos[2] > 0.95 * (p.stance_height + p.foot_radius)
    assert abs(data.qpos[3]) > 0.995  # near-identity orientation quaternion (w component)
    assert float(np.max(np.abs(data.qpos[7:19] - model.key("stand").ctrl))) < 0.05  # < 3° servo droop


def test_servo_gains_are_in_a_sane_range():
    g = servo_gains("datasheet")
    assert 10 < g["kp"] < 40
    assert 0.3 < g["kd"] < 1.0
    assert g["limit"] == pytest.approx(1.912, abs=0.01)
    assert servo_gains("bam")["limit"] > g["limit"]


def test_other_architectures_build():
    for key in ("B", "C"):
        m = mujoco.MjModel.from_xml_string(build_mjcf(preset(key, "M")))
        assert m.nu == 12 and m.nmesh == 0   # no CAD for the unlocked candidates: primitive model
