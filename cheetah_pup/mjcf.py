"""Generate a MuJoCo MJCF model from a DesignParams.

    python -m cheetah_pup.mjcf sim/cheetah_pup.xml           # locked design: CAD inertias + meshes
    python -m cheetah_pup.mjcf sim/cheetah_pup_rl.xml --rl   # training variant of the same
    python -m cheetah_pup.mjcf out.xml --no-cad              # Phase 1 primitive model (any preset)
    python -m cheetah_pup.mjcf sim/cheetah_pup.xml --bam     # BAM-implied torque limit instead of datasheet

Two levels of fidelity:

* primitive (`cad=False`): the shell, servo cases, and electronics are boxes carrying their component
  masses; thighs and shanks are capsules; feet are spheres. Every mass comes from
  `analysis.structure_masses`, `electronics`, and `servo`, and MuJoCo derives inertia from each geom.
  Works for any candidate/preset, so the sizing studies and the design-review pages use it.
* CAD (`cad=True`, the default whenever `cad/exports/mass_properties.json` matches the design): every
  body carries an explicit `<inertial>` computed by `cad/assembly.py` from the printed parts (volume ×
  density × infill), the servos, and the electronics, and the printed parts and servos are shown as
  the exported STL meshes. Collision geometry stays primitive — foot spheres, plus a shell box and leg
  capsules in the full model (hidden in geom group 3) — which is what the RL environment assumes.

Joint conventions match `kinematics.py`: abad positive = abduction on both sides (axis +x on the
left, -x on the right); hip and knee hinges are about -y so positive angles swing the foot
forward and, for the knee, put the knee behind the hip-foot line.

Actuators are the STS3215's firmware position loop reduced to a MuJoCo `general` actuator:
torque = kp (q_ref - q) - kd q_dot, clamped to the stall torque. kp and the back-EMF damping kd
come from Rhoban BAM's identified electrical model (see `servo.py`); Coulomb friction and the
reflected rotor inertia go on the joint.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib

from .analysis import structure_masses, nominal_pose
from .design import DesignParams, locked
from .electronics import PI5, BATTERY_2S, PCB, IMU, WIRING_MASS
from .gait import LEGS, LEG_SIDE, LEG_FRONT
from .servo import STS3215

CONTROL_HZ = 50
TIMESTEP = 0.002

JOINT_RANGE = {"abad": (-0.8, 0.8), "hip": (-1.6, 1.6)}
KNEE_RANGE = (0.05, 2.7)

ROOT = pathlib.Path(__file__).resolve().parent.parent
CAD_PROPS = ROOT / "cad" / "exports" / "mass_properties.json"
MESH_DIR = ROOT / "cad" / "exports" / "stl"
PRINTED_PARTS = ["trunk_tub", "trunk_lid"] + [f"{leg}_{part}" for leg in LEGS for part in ("bracket", "thigh", "shank")]
ELECTRONICS_RGBA = {"battery": "0.3 0.42 0.63 0.6", "pi5": "0.3 0.55 0.5 0.6", "pcb": "0.37 0.55 0.23 0.6"}


def servo_gains(torque_limit: str = "datasheet") -> dict:
    """kp [N·m/rad], kd [N·m·s/rad], and torque limit [N·m] from the BAM model."""
    sv = STS3215
    duty_per_rad = 0.166 * 32 * 1.0049          # BAM: error_gain * firmware kp * error_gain_ratio
    torque_per_volt = sv.kt / sv.resistance
    kp = duty_per_rad * sv.vin * torque_per_volt
    kd = sv.kt * sv.kt / sv.resistance          # back-EMF
    limit = sv.stall_torque if torque_limit == "datasheet" else sv.model_stall_torque()
    return {"kp": kp, "kd": kd, "limit": limit, "no_load_speed": sv.max_pwm * sv.vin / sv.kt}


def load_cad(p: DesignParams) -> dict:
    """The CAD mass-properties/placement report for `p`, or a clear error about how to make it."""
    if not CAD_PROPS.exists():
        raise FileNotFoundError(f"{CAD_PROPS} is missing: run `python -m cad.assembly` (needs the `cad` extra)")
    cad = json.loads(CAD_PROPS.read_text())
    if cad["design"] != p.name:
        raise ValueError(f"CAD exports are for {cad['design']!r}, not {p.name!r}: rerun `python -m cad.assembly`")
    return cad


def cad_available(p: DesignParams) -> bool:
    try:
        load_cad(p)
        return True
    except (FileNotFoundError, ValueError):
        return False


def _fmt(vals, nd=5) -> str:
    return " ".join(f"{v:.{nd}f}" for v in vals)


def _box(name, pos, size, mass, cls="visual", extra=""):
    return (f'<geom name="{name}" type="box" class="{cls}" pos="{_fmt(pos)}" size="{_fmt(size)}" '
            f'mass="{mass:.5f}" {extra}/>')


def _inertial(body: dict) -> str:
    return (f'<inertial pos="{_fmt(body["com"], 6)}" mass="{body["mass"]:.5f}" '
            f'fullinertia="{" ".join(f"{v:.4e}" for v in body["fullinertia"])}"/>')


def _servo_mesh(name: str, placement: dict) -> str:
    return (f'<geom name="{name}" type="mesh" mesh="sts3215" class="servo_mesh" '
            f'pos="{_fmt(placement["pos"], 6)}" quat="{_fmt(placement["quat"], 6)}"/>')


def build_mjcf(p: DesignParams, torque_limit: str = "datasheet", rl: bool = False,
               cad: bool | None = None, meshdir: str | os.PathLike | None = None) -> str:
    """Build the model. `rl=True` emits the training variant: feet-only collisions (shell and legs
    do not collide, as in Playground's quadruped scenes), the sensor set the RL environment reads
    (IMU frame sensors, per-foot velocity/position, foot-floor contact sensors), and a `home`
    keyframe alias. `cad=None` uses the CAD exports when they exist for this design; `meshdir` is
    written into the compiler element (absolute by default, so the string loads from anywhere)."""
    sv = STS3215
    sm = structure_masses(p)
    g = servo_gains(torque_limit)
    if cad is None:
        cad = cad_available(p)
    cad_data = load_cad(p) if cad else None
    shell_w = p.shell_width
    bz1 = p.body_z_offset - p.body_height / 2 + p.wall
    top_z = bz1 + sv.width + 0.002
    body_cls = "nocollide" if rl else "collision"   # shell, thighs, shanks (primitive model)
    col_extra = 'group="3"'                          # CAD model: collision primitives hidden behind the meshes
    out = []
    w = out.append

    w(f'<mujoco model="cheetah_pup{"_rl" if rl else ""}">')
    compiler = '  <compiler angle="radian" autolimits="true"'
    if cad:
        compiler += f' meshdir="{MESH_DIR if meshdir is None else meshdir}"'
    w(compiler + '/>')
    w(f'  <option timestep="{TIMESTEP}" integrator="implicitfast" cone="{"pyramidal" if rl else "elliptic"}"'
      + (' iterations="4" ls_iterations="8"' if rl else '') + '/>')
    w(f'  <visual><headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/><map znear="0.01"/></visual>')
    w('  <default>')
    w('    <geom density="0" condim="3" friction="0.8 0.02 0.001" solref="0.005 1"/>')
    w(f'    <joint damping="{sv.friction_viscous:.4f}" armature="{sv.armature:.4f}" frictionloss="{sv.friction_base:.4f}"/>')
    w('    <default class="visual"><geom contype="0" conaffinity="0" group="1" rgba="0.17 0.18 0.19 1"/></default>')
    w('    <default class="nocollide"><geom contype="0" conaffinity="0" group="0" rgba="0.85 0.83 0.78 1"/></default>')
    w('    <default class="collision"><geom group="0" rgba="0.85 0.83 0.78 1"/></default>')
    w('    <default class="electronics"><geom contype="0" conaffinity="0" group="1" rgba="0.3 0.55 0.5 0.6"/></default>')
    if cad:
        w('    <default class="print"><geom type="mesh" contype="0" conaffinity="0" group="1" rgba="0.87 0.85 0.80 1"/></default>')
        w('    <default class="servo_mesh"><geom type="mesh" contype="0" conaffinity="0" group="1" rgba="0.17 0.18 0.19 1"/></default>')
    w(f'    <default class="servo"><general dyntype="none" gaintype="fixed" biastype="affine" ctrllimited="true" '
      f'forcelimited="true" gainprm="{g["kp"]:.4f}" biasprm="0 {-g["kp"]:.4f} {-g["kd"]:.4f}" '
      f'forcerange="{-g["limit"]:.4f} {g["limit"]:.4f}"/></default>')
    w('  </default>')
    w('  <asset>')
    w('    <texture name="grid" type="2d" builtin="checker" rgb1="0.9 0.9 0.88" rgb2="0.8 0.8 0.78" width="256" height="256"/>'
      '<material name="grid" texture="grid" texrepeat="8 8" reflectance="0.05"/>')
    if cad:
        # STL files are exported in millimetres, each printed part in its own MuJoCo body frame
        for name in PRINTED_PARTS:
            w(f'    <mesh name="{name}" file="{name}.stl" scale="0.001 0.001 0.001"/>')
        w(f'    <mesh name="sts3215" file="{cad_data["servo_mesh"]}" scale="0.001 0.001 0.001"/>')
    w('  </asset>')
    w('  <worldbody>')
    w('    <light pos="0.5 -0.5 1.2" dir="-0.3 0.3 -1" directional="true"/>')
    w('    <geom name="floor" type="plane" size="0 0 0.05" material="grid" contype="1" conaffinity="1" friction="0.8 0.02 0.001"/>')
    w(f'    <body name="trunk" pos="0 0 {p.stance_height + p.foot_radius:.4f}">')
    w('      <freejoint name="root"/>')
    w('      <site name="imu" pos="0 0 0" size="0.005"/>')
    if cad:
        w("      " + _inertial(cad_data["bodies"]["trunk"]))
        w('      <geom name="trunk_tub" mesh="trunk_tub" class="print"/>')
        w('      <geom name="trunk_lid" mesh="trunk_lid" class="print" rgba="0.87 0.85 0.80 0.55"/>')
        for name, pl in cad_data["servos"].items():
            if pl["body"] == "trunk":
                w("      " + _servo_mesh(name, pl))
        for name, e in cad_data["electronics"].items():
            w(_box(name, e["pos"], [s / 2 for s in e["size"]], 0.0, "electronics", f'rgba="{ELECTRONICS_RGBA[name]}"'))
        if not rl:
            w(_box("shell", (0, 0, p.body_z_offset), (p.shell_length / 2, shell_w / 2, p.body_height / 2), 0.0, "collision", col_extra))
    else:
        w(_box("shell", (0, 0, p.body_z_offset), (p.shell_length / 2, shell_w / 2, p.body_height / 2), sm["shell"] + WIRING_MASS, body_cls))
        # electronics: battery low, Pi 5 transverse and PCB transverse on the top layer, IMU centre
        w(_box("battery", (0, 0, bz1 + BATTERY_2S.size[2] / 2), (BATTERY_2S.size[0] / 2, BATTERY_2S.size[1] / 2, BATTERY_2S.size[2] / 2), BATTERY_2S.mass, "electronics", f'rgba="{ELECTRONICS_RGBA["battery"]}"'))
        pi_x = PI5.size[1]  # transverse: the 56 mm side runs along x
        w(_box("pi5", (-(p.shell_length / 2 - p.wall - pi_x / 2), 0, top_z + PI5.size[2] / 2), (pi_x / 2, PI5.size[0] / 2, PI5.size[2] / 2), PI5.mass, "electronics"))
        pcb_x = PCB.size[1]
        w(_box("pcb", ((p.shell_length / 2 - p.wall - pcb_x / 2), 0, top_z + PCB.size[2] / 2), (pcb_x / 2, PCB.size[0] / 2, PCB.size[2] / 2), PCB.mass, "electronics", f'rgba="{ELECTRONICS_RGBA["pcb"]}"'))
        w(_box("imu_board", (0, 0, top_z + 0.003), (IMU.size[0] / 2, IMU.size[1] / 2, IMU.size[2] / 2), IMU.mass, "electronics", 'rgba="0.48 0.35 0.65 0.7"'))

    for leg in LEGS:
        side, front = LEG_SIDE[leg], LEG_FRONT[leg]
        sx = 1 if front else -1
        hx = sx * p.hip_to_hip / 2
        ay = side * p.abad_to_abad / 2
        ks = p.knee_sign(front)
        fy = side * p.foot_y_offset if cad else 0.0
        if not cad:
            # abad servo case in the trunk corner: shaft along x, long side along y, case extends inboard
            w(_box(f"{leg}_abad_servo", (hx - sx * (p.hip_x_offset + sv.height / 2), ay - side * (sv.length / 2 - sv.shaft_from_end), 0),
                   (sv.height / 2, sv.length / 2, sv.width / 2), sv.mass))
        w(f'      <body name="{leg}_abad" pos="{hx:.5f} {ay:.5f} 0">')
        w(f'        <joint name="{leg}_abad" axis="{side} 0 0" range="{JOINT_RANGE["abad"][0]} {JOINT_RANGE["abad"][1]}"/>')
        if cad:
            w("        " + _inertial(cad_data["bodies"][f"{leg}_abad"]))
            w(f'        <geom name="{leg}_bracket" mesh="{leg}_bracket" class="print"/>')
            w("        " + _servo_mesh(f"{leg}_hip_servo", cad_data["servos"][f"{leg}_hip_servo"]))
        else:
            w("  " + _box(f"{leg}_bracket", (0, side * p.abad_link / 2, -0.004), (0.010, p.abad_link / 2, 0.011), sm["bracket"]))
            # hip-pitch servo: shaft along y at the hip axis, case pointing up, inboard of the thigh plane
            w("  " + _box(f"{leg}_hip_servo", (0, side * (p.abad_link - 0.006 - sv.height / 2), sv.length / 2 - sv.shaft_from_end),
                          (sv.width / 2, sv.height / 2, sv.length / 2), sv.mass))
        w(f'        <body name="{leg}_hip" pos="0 {side * p.abad_link:.5f} 0">')
        w(f'          <joint name="{leg}_hip" axis="0 -1 0" range="{JOINT_RANGE["hip"][0]} {JOINT_RANGE["hip"][1]}"/>')
        if cad:
            w("          " + _inertial(cad_data["bodies"][f"{leg}_hip"]))
            w(f'          <geom name="{leg}_thigh" mesh="{leg}_thigh" class="print"/>')
            w("          " + _servo_mesh(f"{leg}_knee_servo", cad_data["servos"][f"{leg}_knee_servo"]))
            if not rl:
                w(f'          <geom name="{leg}_thigh_col" type="capsule" class="collision" {col_extra} fromto="0 0 0 0 0 {-p.thigh:.5f}" size="0.012"/>')
        else:
            w(f'          <geom name="{leg}_thigh" type="capsule" class="{body_cls}" fromto="0 0 0 0 0 {-p.thigh:.5f}" size="0.010" mass="{sm["thigh"] + sm["leg_hardware"]:.5f}"/>')
            if p.architecture == "direct":
                # knee servo at the knee, shaft along y, case pointing back up the thigh
                w("    " + _box(f"{leg}_knee_servo", (0, 0, -p.thigh + (sv.length / 2 - sv.shaft_from_end)), (sv.width / 2, sv.height / 2, sv.length / 2), sv.mass))
            else:
                w("    " + _box(f"{leg}_knee_servo", (0, side * (0.006 + sv.height / 2), sv.length / 2 - sv.shaft_from_end), (sv.width / 2, sv.height / 2, sv.length / 2), sv.mass))
        knee_lo, knee_hi = (KNEE_RANGE[0], KNEE_RANGE[1]) if ks > 0 else (-KNEE_RANGE[1], -KNEE_RANGE[0])
        w(f'          <body name="{leg}_knee" pos="0 0 {-p.thigh:.5f}">')
        w(f'            <joint name="{leg}_knee" axis="0 -1 0" range="{knee_lo} {knee_hi}"/>')
        if cad:
            w("            " + _inertial(cad_data["bodies"][f"{leg}_knee"]))
            w(f'            <geom name="{leg}_shank" mesh="{leg}_shank" class="print"/>')
            if not rl:
                w(f'            <geom name="{leg}_shank_col" type="capsule" class="collision" {col_extra} fromto="0 {fy:.5f} 0 0 {fy:.5f} {-p.shank:.5f}" size="0.008"/>')
            foot_mass = 0.0
        else:
            w(f'            <geom name="{leg}_shank" type="capsule" class="{body_cls}" fromto="0 0 0 0 0 {-p.shank:.5f}" size="0.007" mass="{sm["shank"]:.5f}"/>')
            foot_mass = sm["foot"]
        w(f'            <geom name="{leg}_foot" type="sphere" class="collision" pos="0 {fy:.5f} {-p.shank:.5f}" size="{p.foot_radius:.4f}" mass="{foot_mass:.5f}" friction="1.0 0.02 0.001" rgba="0.12 0.12 0.12 1"/>')
        w(f'            <site name="{leg}_foot" pos="0 {fy:.5f} {-p.shank:.5f}" size="{p.foot_radius + 0.002:.4f}" type="sphere" rgba="0 0 0 0"/>')
        w('          </body>')
        w('        </body>')
        w('      </body>')
    w('    </body>')
    w('  </worldbody>')

    w('  <actuator>')
    for leg in LEGS:
        ks = p.knee_sign(LEG_FRONT[leg])
        knee_lo, knee_hi = (KNEE_RANGE[0], KNEE_RANGE[1]) if ks > 0 else (-KNEE_RANGE[1], -KNEE_RANGE[0])
        for j, (lo, hi) in (("abad", JOINT_RANGE["abad"]), ("hip", JOINT_RANGE["hip"]), ("knee", (knee_lo, knee_hi))):
            w(f'    <general name="{leg}_{j}" class="servo" joint="{leg}_{j}" ctrlrange="{lo} {hi}"/>')
    w('  </actuator>')
    w('  <sensor>')
    if rl:
        # the set Playground's quadruped envs read, named the same way
        w('    <gyro site="imu" name="gyro"/>')
        w('    <velocimeter site="imu" name="local_linvel"/>')
        w('    <accelerometer site="imu" name="accelerometer"/>')
        w('    <framepos objtype="site" objname="imu" name="position"/>')
        w('    <framezaxis objtype="site" objname="imu" name="upvector"/>')
        w('    <framexaxis objtype="site" objname="imu" name="forwardvector"/>')
        w('    <framelinvel objtype="site" objname="imu" name="global_linvel"/>')
        w('    <frameangvel objtype="site" objname="imu" name="global_angvel"/>')
        w('    <framequat objtype="site" objname="imu" name="orientation"/>')
        for leg in LEGS:
            w(f'    <framelinvel objtype="site" objname="{leg}_foot" name="{leg}_foot_global_linvel"/>')
            w(f'    <framepos objtype="site" objname="{leg}_foot" name="{leg}_foot_pos" reftype="site" refname="imu"/>')
        for leg in LEGS:
            w(f'    <contact name="{leg}_foot_floor_found" geom1="{leg}_foot" geom2="floor" reduce="mindist" num="1" data="found"/>')
    else:
        w('    <framequat name="imu_quat" objtype="site" objname="imu"/>')
        w('    <gyro name="imu_gyro" site="imu"/>')
        w('    <accelerometer name="imu_acc" site="imu"/>')
        for leg in LEGS:
            w(f'    <touch name="{leg}_touch" site="{leg}_foot"/>')
    w('  </sensor>')

    # trunk sits one foot radius above the stance height so the foot spheres rest on the floor
    qpos = ["0", "0", f"{p.stance_height + p.foot_radius:.4f}", "1", "0", "0", "0"]
    ctrl = []
    for leg in LEGS:
        q_hip, q_knee = nominal_pose(p, LEG_FRONT[leg])
        qpos += ["0", f"{q_hip:.5f}", f"{q_knee:.5f}"]
        ctrl += ["0", f"{q_hip:.5f}", f"{q_knee:.5f}"]
    keys = [f'<key name="stand" qpos="{" ".join(qpos)}" ctrl="{" ".join(ctrl)}"/>']
    if rl:
        keys.append(f'<key name="home" qpos="{" ".join(qpos)}" ctrl="{" ".join(ctrl)}"/>')
    w(f'  <keyframe>{"".join(keys)}</keyframe>')
    w('</mujoco>')
    return "\n".join(out) + "\n"


def joint_names():
    return [f"{leg}_{j}" for leg in LEGS for j in ("abad", "hip", "knee")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="sim/cheetah_pup.xml")
    ap.add_argument("--bam", action="store_true", help="use BAM's implied stall torque as the limit")
    ap.add_argument("--rl", action="store_true", help="training variant: feet-only collisions, RL sensor set")
    ap.add_argument("--no-cad", action="store_true", help="primitive geometry and parametric masses (no CAD exports needed)")
    args = ap.parse_args(argv)
    p = locked()
    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    cad = not args.no_cad
    if cad:
        load_cad(p)   # fail loudly on missing/stale exports rather than silently writing the primitive model
    meshdir = os.path.relpath(MESH_DIR, path.resolve().parent)
    xml = build_mjcf(p, "bam" if args.bam else "datasheet", rl=args.rl, cad=cad, meshdir=meshdir)
    path.write_text(xml)
    gains = servo_gains("bam" if args.bam else "datasheet")
    print(f"wrote {path} for {p.name}{' (RL variant)' if args.rl else ''}{' (CAD inertias + meshes)' if cad else ' (primitive)'}: "
          f"kp {gains['kp']:.1f} N·m/rad, kd {gains['kd']:.2f} N·m·s/rad, limit {gains['limit']:.2f} N·m, "
          f"no-load {gains['no_load_speed']:.1f} rad/s")


if __name__ == "__main__":
    main()
