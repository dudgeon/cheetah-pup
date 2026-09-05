"""Assemble the robot at the nominal stance, export STEP/STL, and compute per-body mass properties.

    python -m cad.assembly            # writes cad/exports/{step,stl,mass_properties.json,viewer_meshes.json}

`mass_properties.json` also carries the servo and electronics placements the MJCF generator needs;
`viewer_meshes.json` holds the unique part meshes, their instances per body, and the kinematic tree
for the DR-04 review page's viewer.

Mass properties are computed per MuJoCo body from the printed parts (volume × material density ×
infill factor), the servos (measured 55 g spread over the case solid), and the electronics (measured
masses as uniform boxes), all in the body's own frame — exactly what `<inertial>` needs.
"""

from __future__ import annotations

import json
import math
import pathlib
import time

import numpy as np
from build123d import Box, CenterOf, Compound, Location, Pos, Rot, Shape, export_step, export_stl

from . import params as C
from . import parts as PT
from . import servo as SV

OUT = pathlib.Path(__file__).resolve().parent / "exports"
LEGS = [("LF", 1, True), ("RF", -1, True), ("LH", 1, False), ("RH", -1, False)]
WIRING_MASS = 0.060


def rigid(shape: Shape, mass: float) -> dict:
    """Mass, centre of mass (m), and inertia about the CoM (kg·m²) of a shape carrying `mass`."""
    v = shape.volume
    c = shape.center(CenterOf.MASS)
    I = np.array(shape.matrix_of_inertia, dtype=float) * (mass / v) * 1e-6
    return {"mass": mass, "com": np.array([c.X, c.Y, c.Z]) / 1000.0, "inertia": I, "volume_mm3": v}


def combine(items: list[dict]) -> dict:
    """Combine rigid bodies given in one frame: total mass, CoM, inertia about the total CoM."""
    M = sum(it["mass"] for it in items)
    com = sum(it["mass"] * it["com"] for it in items) / M
    I = np.zeros((3, 3))
    for it in items:
        d = it["com"] - com
        I += it["inertia"] + it["mass"] * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return {"mass": M, "com": com, "inertia": I}


def fullinertia(I: np.ndarray) -> list:
    return [float(I[0, 0]), float(I[1, 1]), float(I[2, 2]), float(I[0, 1]), float(I[0, 2]), float(I[1, 2])]


def rotmat_to_quat(R: np.ndarray) -> list[float]:
    """Unit quaternion (w, x, y, z) of a rotation matrix (Shepperd's method)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    return [float(v) for v in q]


def servo_placement(frame: SV.ServoFrame, body: str) -> dict:
    """Where the servo mesh (exported in its own L, W, A frame) sits in a MuJoCo body: pos (m), quat."""
    R = np.array([frame.l_dir, frame.w_dir, frame.a_dir], dtype=float).T   # columns = frame axes
    return {"body": body, "pos": [round(v / 1000.0, 6) for v in frame.origin], "quat": [round(v, 6) for v in rotmat_to_quat(R)]}


def printed(shape: Shape, cls: str, density: float = C.PLA_DENSITY) -> tuple[dict, float]:
    mass = shape.volume * 1e-9 * density * C.INFILL[cls]
    return rigid(shape, mass), mass


def body_location(leg: str, side: int, front: bool, which: str) -> Location:
    """World location of a body frame at the nominal stance (trunk at z = stance + foot radius)."""
    z0 = C.STANCE + C.FOOT_R
    hx = (1 if front else -1) * C.HIP_TO_HIP / 2
    loc = Pos(0, 0, z0)
    if which == "trunk":
        return loc
    loc = loc * Pos(hx, side * C.ABAD_Y, 0)
    if which == "abad":
        return loc
    loc = loc * Pos(0, side * C.ABAD_LINK, 0) * Rot(0, -math.degrees(C.Q_HIP), 0)
    if which == "hip":
        return loc
    return loc * Pos(0, 0, -C.THIGH) * Rot(0, -math.degrees(C.Q_KNEE), 0)


def tessellate(shape: Shape, tol=0.3, ang=0.4):
    verts, tris = shape.tessellate(tol, ang)
    return [[round(v.X, 2), round(v.Y, 2), round(v.Z, 2)] for v in verts], [list(t) for t in tris]


def build(export: bool = True, viewer: bool = True) -> dict:
    t0 = time.time()
    (OUT / "step").mkdir(parents=True, exist_ok=True)
    (OUT / "stl").mkdir(parents=True, exist_ok=True)
    parts: dict[str, dict] = {}       # printed parts: name -> {shape, cls, body, side...}
    bodies: dict[str, list] = {}      # body -> list of rigid items
    servos: dict[str, dict] = {}      # servo mesh placements per body, for the MJCF
    world: list[tuple[str, str, Shape, str]] = []   # (name, kind, shape in world, body)
    mesh_shapes: dict[str, Shape] = {}              # unique shapes for the viewer, by mesh key
    instances: list[dict] = []                      # viewer instances: mesh key placed in a body (mm)

    def shared(key: str, make):
        if key not in mesh_shapes:
            mesh_shapes[key] = make()
        return mesh_shapes[key]

    def instance(name, kind, body, mesh, pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)):
        instances.append({"name": name, "kind": kind, "body": body, "mesh": mesh,
                          "pos": [round(float(v), 4) for v in pos], "quat": [round(float(v), 6) for v in quat]})

    def add_part(name, shape, cls, body, density=C.PLA_DENSITY):
        props, mass = printed(shape, cls, density)
        parts[name] = {"shape": shape, "cls": cls, "body": body, "mass": mass, "props": props}
        bodies.setdefault(body, []).append(props)

    def add_rigid(name, shape, mass, body):
        bodies.setdefault(body, []).append(rigid(shape, mass))
        return shape

    # ---- trunk
    tub, lid = shared("trunk_tub", PT.trunk_tub), shared("trunk_lid", PT.trunk_lid)
    add_part("trunk_tub", tub, "shell", "trunk")
    add_part("trunk_lid", lid, "shell", "trunk")
    instance("trunk_tub", "print", "trunk", "trunk_tub")
    instance("trunk_lid", "print", "trunk", "trunk_lid")
    servo_solid = shared("servo", SV.servo_solid)
    rigid_trunk = []
    for leg, side, front in LEGS:
        fr = PT.abad_servo_frame(side, front)
        s = SV.placed(servo_solid, fr)
        add_rigid(f"{leg}_abad_servo", s, SV.MASS, "trunk")
        servos[f"{leg}_abad_servo"] = servo_placement(fr, "trunk")
        instance(f"{leg}_abad_servo", "servo", "trunk", "servo", fr.origin, servos[f"{leg}_abad_servo"]["quat"])
        rigid_trunk.append((f"{leg}_abad_servo", "servo", s))
    battery = Pos(*C.BATTERY_C) * shared("battery", lambda: Box(*C.BATTERY))
    pi = Pos(*C.PI_C) * shared("pi5", lambda: Box(*C.PI))
    pcb = Pos(*C.PCB_C) * shared("pcb", lambda: Box(*C.PCB_ENV))
    wiring = Pos(0, 0, 0) * Box(100, 60, 10)
    from cheetah_pup.electronics import PI5, BATTERY_2S, PCB, IMU
    add_rigid("battery", battery, BATTERY_2S.mass, "trunk")
    add_rigid("pi5", pi, PI5.mass, "trunk")
    add_rigid("pcb", pcb, PCB.mass + IMU.mass, "trunk")
    add_rigid("wiring", wiring, WIRING_MASS, "trunk")
    instance("battery", "battery", "trunk", "battery", C.BATTERY_C)
    instance("pi5", "pi", "trunk", "pi5", C.PI_C)
    instance("pcb", "pcb", "trunk", "pcb", C.PCB_C)
    rigid_trunk += [("battery", "battery", battery), ("pi5", "pi", pi), ("pcb", "pcb", pcb)]

    # ---- legs
    leg_shapes = {}
    for leg, side, front in LEGS:
        sd = "L" if side > 0 else "R"
        br = shared(f"bracket_{sd}{'F' if front else 'H'}", lambda: PT.abad_bracket(side, front))
        add_part(f"{leg}_bracket", br, "plate", f"{leg}_abad")
        instance(f"{leg}_bracket", "print", f"{leg}_abad", f"bracket_{sd}{'F' if front else 'H'}")
        hf = PT.hip_servo_frame(side)
        hip_servo = SV.placed(servo_solid, hf)
        add_rigid(f"{leg}_hip_servo", hip_servo, SV.MASS, f"{leg}_abad")
        servos[f"{leg}_hip_servo"] = servo_placement(hf, f"{leg}_abad")
        instance(f"{leg}_hip_servo", "servo", f"{leg}_abad", "servo", hf.origin, servos[f"{leg}_hip_servo"]["quat"])
        th = shared(f"thigh_{sd}", lambda: PT.thigh(side))
        add_part(f"{leg}_thigh", th, "plate", f"{leg}_hip")
        instance(f"{leg}_thigh", "print", f"{leg}_hip", f"thigh_{sd}")
        kf = PT.knee_servo_frame(side)
        knee_servo = SV.placed(servo_solid, kf)
        add_rigid(f"{leg}_knee_servo", knee_servo, SV.MASS, f"{leg}_hip")
        servos[f"{leg}_knee_servo"] = servo_placement(kf, f"{leg}_hip")
        instance(f"{leg}_knee_servo", "servo", f"{leg}_hip", "servo", kf.origin, servos[f"{leg}_knee_servo"]["quat"])
        sh = shared(f"shank_{sd}", lambda: PT.shank(side))
        add_part(f"{leg}_shank", sh, "beam", f"{leg}_knee")
        instance(f"{leg}_shank", "print", f"{leg}_knee", f"shank_{sd}")
        ft = shared(f"foot_{sd}", lambda: PT.foot(side))
        add_part(f"{leg}_foot", ft, "tpu", f"{leg}_knee", C.TPU_DENSITY)
        instance(f"{leg}_foot", "foot", f"{leg}_knee", f"foot_{sd}")
        leg_shapes[leg] = {"abad": [(f"{leg}_bracket", "print", br), (f"{leg}_hip_servo", "servo", hip_servo)],
                           "hip": [(f"{leg}_thigh", "print", th), (f"{leg}_knee_servo", "servo", knee_servo)],
                           "knee": [(f"{leg}_shank", "print", sh), (f"{leg}_foot", "foot", ft)]}

    # ---- world placement at the nominal stance
    L = body_location("", 1, True, "trunk")
    world += [("trunk_tub", "print", L * tub, "trunk"), ("trunk_lid", "print", L * lid, "trunk")]
    world += [(n, k, L * s, "trunk") for (n, k, s) in rigid_trunk]
    for leg, side, front in LEGS:
        for which in ("abad", "hip", "knee"):
            Lb = body_location(leg, side, front, which)
            for (n, k, s) in leg_shapes[leg][which]:
                world.append((n, k, Lb * s, f"{leg}_{which}"))

    # ---- mass properties
    report = {"generated": time.strftime("%Y-%m-%d"), "design": C.P.name, "units": "kg, m, kg·m² (inertia about the body CoM, body frame)",
              "infill": C.INFILL, "densities": {"pla": C.PLA_DENSITY, "tpu": C.TPU_DENSITY}, "parts": {}, "bodies": {}}
    for name, p in parts.items():
        pr = p["props"]
        report["parts"][name] = {"body": p["body"], "class": p["cls"], "volume_cm3": round(pr["volume_mm3"] / 1000, 2),
                                 "mass_g": round(p["mass"] * 1000, 1), "com_m": [round(float(v), 5) for v in pr["com"]]}
    total = 0.0
    for body, items in bodies.items():
        cb = combine(items)
        total += cb["mass"]
        report["bodies"][body] = {"mass": round(float(cb["mass"]), 5), "com": [round(float(v), 5) for v in cb["com"]],
                                  "fullinertia": [float(f"{v:.4e}") for v in fullinertia(cb["inertia"])]}
    report["total_mass"] = round(total, 4)
    printed_total = sum(p["mass"] for p in parts.values())
    report["printed_mass"] = round(printed_total, 4)
    # placements the MJCF generator needs besides the printed parts (which are in their body frames)
    report["servos"] = servos
    report["servo_mesh"] = "servo_sts3215.stl"
    report["electronics"] = {
        "battery": {"body": "trunk", "pos": [v / 1000.0 for v in C.BATTERY_C], "size": [v / 1000.0 for v in C.BATTERY]},
        "pi5": {"body": "trunk", "pos": [v / 1000.0 for v in C.PI_C], "size": [v / 1000.0 for v in C.PI]},
        "pcb": {"body": "trunk", "pos": [v / 1000.0 for v in C.PCB_C], "size": [v / 1000.0 for v in C.PCB_ENV]},
    }
    report["stance"] = {"trunk_z": (C.STANCE + C.FOOT_R) / 1000.0, "q_hip": C.Q_HIP, "q_knee": C.Q_KNEE}

    if export:
        for name, p in parts.items():
            export_step(p["shape"], str(OUT / "step" / f"{name}.step"))
            export_stl(p["shape"], str(OUT / "stl" / f"{name}.stl"), tolerance=0.15, angular_tolerance=0.3)
        export_stl(servo_solid, str(OUT / "stl" / report["servo_mesh"]), tolerance=0.15, angular_tolerance=0.3)
        export_step(servo_solid, str(OUT / "step" / "servo_sts3215.step"))
        assembly = Compound(children=[s for (_, _, s, _) in world])
        export_step(assembly, str(OUT / "step" / "assembly_nominal_stance.step"))
        (OUT / "mass_properties.json").write_text(json.dumps(report, indent=1))
    if viewer:
        tree = [{"name": "trunk", "parent": None, "pos": [0.0, 0.0, 0.0], "axis": None}]
        for leg, side, front in LEGS:
            hx = (1 if front else -1) * C.HIP_TO_HIP / 2
            tree += [{"name": f"{leg}_abad", "parent": "trunk", "pos": [hx, side * C.ABAD_Y, 0.0], "axis": [side, 0, 0]},
                     {"name": f"{leg}_hip", "parent": f"{leg}_abad", "pos": [0.0, side * C.ABAD_LINK, 0.0], "axis": [0, -1, 0]},
                     {"name": f"{leg}_knee", "parent": f"{leg}_hip", "pos": [0.0, 0.0, -C.THIGH], "axis": [0, -1, 0]}]
        meshes = {}
        for key, shape in mesh_shapes.items():
            v, t = tessellate(shape)
            meshes[key] = {"v": [c for p in v for c in p], "t": [i for tri in t for i in tri]}
        viewer_data = {"design": C.P.name, "generated": report["generated"], "units": "mm",
                       "stance": report["stance"], "bodies": tree, "meshes": meshes, "instances": instances,
                       "part_mass_g": {n: pp["mass_g"] for n, pp in report["parts"].items()}}
        (OUT / "viewer_meshes.json").write_text(json.dumps(viewer_data, separators=(",", ":")))
    report["seconds"] = round(time.time() - t0, 1)
    return report


def main():
    r = build()
    print(f"{r['design']}: total {r['total_mass']:.3f} kg, printed {r['printed_mass']*1000:.0f} g, {r['seconds']} s")
    for name, p in r["parts"].items():
        print(f"  {name:14s} {p['mass_g']:6.1f} g  ({p['class']}, {p['volume_cm3']} cm³)")
    for body, b in r["bodies"].items():
        print(f"  body {body:9s} {b['mass']*1000:6.1f} g  com {[round(v*1000,1) for v in b['com']]} mm")


if __name__ == "__main__":
    main()
