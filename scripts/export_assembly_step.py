#!/usr/bin/env python3
"""Export original MuJoCo assembly envelopes as named STEP solids.

Reproduce from the repository root (Python 3.12):
    uv run --locked --with cadquery-ocp==7.9.3.1.1 python scripts/export_assembly_step.py

cadquery-ocp is an optional export dependency, not part of the RL runtime.
Only the project's original boxes, cylinders and spheres are exported. The
manufacturer STEP is never read. This is an assembly/clearance study, not
manufacturing CAD: no holes, fasteners, cable routing or bearing detail.

Coordinates in the STEP are millimeters. The neutral pose is the compiled
MuJoCo 'stand' keyframe, with feet on Z=0. World terrain and group-5 cable
allowances are excluded. Names and colors are retained using STEPCAF.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.BRepPrimAPI import (
    BRepPrimAPI_MakeBox,
    BRepPrimAPI_MakeCylinder,
    BRepPrimAPI_MakeSphere,
)
from OCP.GProp import GProp_GProps
from OCP.gp import gp_Pnt, gp_Trsf
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCP.STEPCAFControl import STEPCAFControl_Reader, STEPCAFControl_Writer
from OCP.STEPControl import STEPControl_AsIs
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label, TDF_LabelSequence
from OCP.TDocStd import TDocStd_Document
from OCP.TopLoc import TopLoc_Location
from OCP.XCAFDoc import XCAFDoc_ColorGen, XCAFDoc_DocumentTool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cheetah_pup.model import build_mjcf, load_config  # noqa: E402


def shape_measurements(shape):
    bounds = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, bounds, False, False)
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return np.array(bounds.Get()), np.array(props.CentreOfMass().Coord()), props.Mass()


def make_solid(kind, sizes, center, rotation):
    """Use compiled MuJoCo local half-sizes and local-to-world transform."""
    sx, sy, sz = sizes * 1000.0
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        local = BRepPrimAPI_MakeBox(
            gp_Pnt(-sx, -sy, -sz), 2 * sx, 2 * sy, 2 * sz
        ).Shape()
        extent = np.abs(rotation) @ np.array([sx, sy, sz])
        volume = 8 * sx * sy * sz
        type_name = "box"
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        # MuJoCo cylinder: size[0] radius, size[1] half-height; local axis +Z.
        local = BRepPrimAPI_MakeCylinder(sx, 2 * sy).Shape()
        shift = gp_Trsf()
        shift.SetValues(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -sy)
        local = BRepBuilderAPI_Transform(local, shift, True).Shape()
        axis = rotation[:, 2]
        extent = np.abs(axis) * sy + sx * np.sqrt(np.maximum(0, 1 - axis**2))
        volume = np.pi * sx * sx * 2 * sy
        type_name = "cylinder"
    elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        local = BRepPrimAPI_MakeSphere(sx).Shape()
        extent = np.full(3, sx)
        volume = 4 * np.pi * sx**3 / 3
        type_name = "sphere"
    else:
        raise ValueError(f"Unsupported assembly geom type: {kind}")
    transform = gp_Trsf()
    transform.SetValues(*np.column_stack((rotation, center * 1000)).flatten().tolist())
    shape = BRepBuilderAPI_Transform(local, transform, True).Shape()
    expected = {
        "bounds": np.r_[center * 1000 - extent, center * 1000 + extent],
        "center": center * 1000,
        "volume": volume,
        "kind": type_name,
    }
    if not BRepCheck_Analyzer(shape).IsValid():
        raise RuntimeError("Generated an invalid OpenCascade solid.")
    return shape, expected


def label_name(label):
    attribute = TDataStd_Name()
    return (
        attribute.Get().ToExtString()
        if label.FindAttribute(TDataStd_Name.GetID_s(), attribute)
        else None
    )


def read_named_parts(path):
    document = TDocStd_Document(TCollection_ExtendedString("roundtrip"))
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    if reader.ReadFile(str(path)) != IFSelect_RetDone or not reader.Transfer(document):
        raise RuntimeError("STEP readback failed.")
    tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    roots = TDF_LabelSequence()
    tool.GetFreeShapes(roots)
    parts = {}

    def visit(label, parent_location=TopLoc_Location()):
        location = parent_location.Multiplied(tool.GetLocation_s(label))
        referred = TDF_Label()
        target = referred if tool.GetReferredShape_s(label, referred) else label
        children = TDF_LabelSequence()
        tool.GetComponents_s(target, children, False)
        if children.Length():
            for index in range(1, children.Length() + 1):
                visit(children.Value(index), location)
        else:
            name = label_name(target)
            if name in parts:
                raise RuntimeError(f"Duplicate STEP part name: {name}")
            parts[name] = tool.GetShape_s(target).Located(location)

    for index in range(1, roots.Length() + 1):
        visit(roots.Value(index))
    return parts


def audit_named_parts(actual, expected):
    if set(actual) != set(expected):
        raise RuntimeError("STEP part names/count do not match MuJoCo geoms.")
    worst = {"bounds_mm": 0.0, "center_mm": 0.0, "relative_volume": 0.0}
    for name, shape in actual.items():
        bounds, center, volume = shape_measurements(shape)
        reference = expected[name]
        worst["bounds_mm"] = max(
            worst["bounds_mm"], float(np.max(np.abs(bounds - reference["bounds"])))
        )
        worst["center_mm"] = max(
            worst["center_mm"], float(np.max(np.abs(center - reference["center"])))
        )
        worst["relative_volume"] = max(
            worst["relative_volume"], abs(volume / reference["volume"] - 1)
        )
        if not BRepCheck_Analyzer(shape).IsValid():
            raise RuntimeError(f"Invalid STEP solid: {name}")
    if (
        worst["bounds_mm"] > 1e-5
        or worst["center_mm"] > 1e-5
        or worst["relative_volume"] > 1e-7
    ):
        raise RuntimeError(
            f"STEP geometry differs from compiled MuJoCo geometry: {worst}"
        )
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/robot.json")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "models/cheetah_pup_assembly.step"
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "reports/step-export-validation.json"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    xml = build_mjcf(config)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)

    document = TDocStd_Document(TCollection_ExtendedString("assembly"))
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool_s(document.Main())
    root = shape_tool.NewShape()
    TDataStd_Name.Set_s(
        root,
        TCollection_ExtendedString(
            "Cheetah Pup - original assembly envelopes - study only"
        ),
    )
    expected, shapes, excluded = {}, {}, []
    for geom in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
        if model.geom_bodyid[geom] == 0 or model.geom_group[geom] == 5:
            excluded.append(name)
            continue
        shape, reference = make_solid(
            model.geom_type[geom],
            model.geom_size[geom],
            data.geom_xpos[geom],
            data.geom_xmat[geom].reshape(3, 3),
        )
        expected[name], shapes[name] = reference, shape
        part = shape_tool.AddShape(shape, False, False)
        TDataStd_Name.Set_s(part, TCollection_ExtendedString(name))
        color = Quantity_Color(*map(float, model.geom_rgba[geom, :3]), Quantity_TOC_RGB)
        color_tool.SetColor(part, color, XCAFDoc_ColorGen)
        component = shape_tool.AddComponent(root, part, TopLoc_Location())
        TDataStd_Name.Set_s(component, TCollection_ExtendedString(name))
    shape_tool.UpdateAssemblies()
    generated_errors = audit_named_parts(shapes, expected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = STEPCAFControl_Writer()
    writer.SetNameMode(True)
    writer.SetColorMode(True)
    Interface_Static.SetCVal_s("write.step.schema", "AP214IS")
    Interface_Static.SetCVal_s("write.step.unit", "MM")
    if (
        not writer.Transfer(document, STEPControl_AsIs)
        or writer.Write(str(args.output)) != IFSelect_RetDone
    ):
        raise RuntimeError("STEP export failed.")
    readback_errors = audit_named_parts(read_named_parts(args.output), expected)
    all_bounds = np.array([value["bounds"] for value in expected.values()])
    bounds = np.r_[all_bounds[:, :3].min(axis=0), all_bounds[:, 3:].max(axis=0)]
    report = {
        "scope": "Original assembly envelope study; not manufacturing CAD, not a physics rollout.",
        "pose": "MuJoCo stand keyframe",
        "units": "millimeters",
        "mujoco_version": mujoco.__version__,
        "cadquery_ocp_version": importlib.metadata.version("cadquery-ocp"),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "generated_mjcf_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "step_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "step_bytes": args.output.stat().st_size,
        "part_count": len(expected),
        "part_types": dict(Counter(part["kind"] for part in expected.values())),
        "part_names_preserved_on_readback": True,
        "all_solids_valid": True,
        "bounds_min_xyz_max_xyz_mm": bounds.tolist(),
        "size_xyz_mm": (bounds[3:] - bounds[:3]).tolist(),
        "maximum_errors_before_export": generated_errors,
        "maximum_errors_after_step_readback": readback_errors,
        "excluded_world_and_group5_geom_count": len(excluded),
        "excluded_geoms": excluded,
        "manufacturer_cad_copied": False,
        "manufacturing_ready": False,
        "limitations": [
            "Mounting holes, screws, bearings, cable routing and tolerances are absent.",
            "Electronics and battery solids are provisional package allowances.",
            "STEP solids do not encode joint motion or servo reference mass properties.",
            "Geometric agreement with MuJoCo does not validate hardware or actuator behavior.",
        ],
        "reproduce": "uv run --locked --with cadquery-ocp==7.9.3.1.1 python scripts/export_assembly_step.py",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "parts": len(expected),
                "roundtrip_errors": readback_errors,
            }
        )
    )


if __name__ == "__main__":
    main()
