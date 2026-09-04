"""Generate a MuJoCo MJCF model from a DesignParams.

    python -m cheetah_pup.mjcf sim/cheetah_pup.xml          # locked design
    python -m cheetah_pup.mjcf sim/cheetah_pup.xml --bam    # BAM-implied torque limit instead of datasheet

Primitive geometry only (Phase 1): the shell, servo cases, and electronics are boxes with their
component masses; thighs and shanks are capsules; feet are spheres. Every mass comes from
`analysis.structure_masses`, `electronics`, and `servo`, so the model's total mass and inertia
distribution match the sizing analysis. MuJoCo derives inertia from each geom's shape and mass.

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
import math
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


def servo_gains(torque_limit: str = "datasheet") -> dict:
    """kp [N·m/rad], kd [N·m·s/rad], and torque limit [N·m] from the BAM model."""
    sv = STS3215
    duty_per_rad = 0.166 * 32 * 1.0049          # BAM: error_gain * firmware kp * error_gain_ratio
    torque_per_volt = sv.kt / sv.resistance
    kp = duty_per_rad * sv.vin * torque_per_volt
    kd = sv.kt * sv.kt / sv.resistance          # back-EMF
    limit = sv.stall_torque if torque_limit == "datasheet" else sv.model_stall_torque()
    return {"kp": kp, "kd": kd, "limit": limit, "no_load_speed": sv.max_pwm * sv.vin / sv.kt}


def _box(name, pos, size, mass, cls="visual", extra=""):
    return (f'<geom name="{name}" type="box" class="{cls}" pos="{pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}" '
            f'size="{size[0]:.5f} {size[1]:.5f} {size[2]:.5f}" mass="{mass:.5f}" {extra}/>')


def build_mjcf(p: DesignParams, torque_limit: str = "datasheet") -> str:
    sv = STS3215
    sm = structure_masses(p)
    g = servo_gains(torque_limit)
    shell_w = p.abad_to_abad + 0.025
    bz1 = p.body_z_offset - p.body_height / 2 + p.wall
    top_z = bz1 + sv.width + 0.002
    out = []
    w = out.append

    w(f'<mujoco model="cheetah_pup">')
    w(f'  <compiler angle="radian" autolimits="true"/>')
    w(f'  <option timestep="{TIMESTEP}" integrator="implicitfast" cone="elliptic"/>')
    w(f'  <visual><headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/><map znear="0.01"/></visual>')
    w('  <default>')
    w('    <geom density="0" condim="3" friction="0.8 0.02 0.001" solref="0.005 1"/>')
    w(f'    <joint damping="{sv.friction_viscous:.4f}" armature="{sv.armature:.4f}" frictionloss="{sv.friction_base:.4f}"/>')
    w('    <default class="visual"><geom contype="0" conaffinity="0" group="1" rgba="0.17 0.18 0.19 1"/></default>')
    w('    <default class="collision"><geom group="0" rgba="0.85 0.83 0.78 1"/></default>')
    w('    <default class="electronics"><geom contype="0" conaffinity="0" group="1" rgba="0.3 0.55 0.5 0.6"/></default>')
    w(f'    <default class="servo"><general dyntype="none" gaintype="fixed" biastype="affine" ctrllimited="true" '
      f'forcelimited="true" gainprm="{g["kp"]:.4f}" biasprm="0 {-g["kp"]:.4f} {-g["kd"]:.4f}" '
      f'forcerange="{-g["limit"]:.4f} {g["limit"]:.4f}"/></default>')
    w('  </default>')
    w('  <asset><texture name="grid" type="2d" builtin="checker" rgb1="0.9 0.9 0.88" rgb2="0.8 0.8 0.78" width="256" height="256"/>'
      '<material name="grid" texture="grid" texrepeat="8 8" reflectance="0.05"/></asset>')
    w('  <worldbody>')
    w('    <light pos="0.5 -0.5 1.2" dir="-0.3 0.3 -1" directional="true"/>')
    w('    <geom name="floor" type="plane" size="0 0 0.05" material="grid" contype="1" conaffinity="1" friction="0.8 0.02 0.001"/>')
    w(f'    <body name="trunk" pos="0 0 {p.stance_height + p.foot_radius:.4f}">')
    w('      <freejoint name="root"/>')
    w('      <site name="imu" pos="0 0 0" size="0.005"/>')
    w(_box("shell", (0, 0, p.body_z_offset), (p.shell_length / 2, shell_w / 2, p.body_height / 2), sm["shell"] + WIRING_MASS, "collision"))
    # electronics: battery low, Pi 5 transverse and PCB transverse on the top layer, IMU centre
    w(_box("battery", (0, 0, bz1 + BATTERY_2S.size[2] / 2), (BATTERY_2S.size[0] / 2, BATTERY_2S.size[1] / 2, BATTERY_2S.size[2] / 2), BATTERY_2S.mass, "electronics", 'rgba="0.3 0.42 0.63 0.6"'))
    pi_x = PI5.size[1]  # transverse: the 56 mm side runs along x
    w(_box("pi5", (-(p.shell_length / 2 - p.wall - pi_x / 2), 0, top_z + PI5.size[2] / 2), (pi_x / 2, PI5.size[0] / 2, PI5.size[2] / 2), PI5.mass, "electronics"))
    pcb_x = PCB.size[1]
    w(_box("pcb", ((p.shell_length / 2 - p.wall - pcb_x / 2), 0, top_z + PCB.size[2] / 2), (pcb_x / 2, PCB.size[0] / 2, PCB.size[2] / 2), PCB.mass, "electronics", 'rgba="0.37 0.55 0.23 0.6"'))
    w(_box("imu_board", (0, 0, top_z + 0.003), (IMU.size[0] / 2, IMU.size[1] / 2, IMU.size[2] / 2), IMU.mass, "electronics", 'rgba="0.48 0.35 0.65 0.7"'))

    for leg in LEGS:
        side, front = LEG_SIDE[leg], LEG_FRONT[leg]
        sx = 1 if front else -1
        hx = sx * p.hip_to_hip / 2
        ay = side * p.abad_to_abad / 2
        ks = p.knee_sign(front)
        # abad servo case in the trunk corner: shaft along x, long side along y, case extends inboard
        w(_box(f"{leg}_abad_servo", (hx - sx * (p.hip_x_offset + sv.height / 2), ay - side * (sv.length / 2 - sv.shaft_from_end), 0),
               (sv.height / 2, sv.length / 2, sv.width / 2), sv.mass))
        w(f'      <body name="{leg}_abad" pos="{hx:.5f} {ay:.5f} 0">')
        w(f'        <joint name="{leg}_abad" axis="{side} 0 0" range="{JOINT_RANGE["abad"][0]} {JOINT_RANGE["abad"][1]}"/>')
        w("  " + _box(f"{leg}_bracket", (0, side * p.abad_link / 2, -0.004), (0.010, p.abad_link / 2, 0.011), sm["bracket"]))
        # hip-pitch servo: shaft along y at the hip axis, case pointing up, inboard of the thigh plane
        w("  " + _box(f"{leg}_hip_servo", (0, side * (p.abad_link - 0.006 - sv.height / 2), sv.length / 2 - sv.shaft_from_end),
                      (sv.width / 2, sv.height / 2, sv.length / 2), sv.mass))
        w(f'        <body name="{leg}_hip" pos="0 {side * p.abad_link:.5f} 0">')
        w(f'          <joint name="{leg}_hip" axis="0 -1 0" range="{JOINT_RANGE["hip"][0]} {JOINT_RANGE["hip"][1]}"/>')
        w(f'          <geom name="{leg}_thigh" type="capsule" class="collision" fromto="0 0 0 0 0 {-p.thigh:.5f}" size="0.010" mass="{sm["thigh"] + sm["leg_hardware"]:.5f}"/>')
        if p.architecture == "direct":
            # knee servo at the knee, shaft along y, case pointing back up the thigh
            w("    " + _box(f"{leg}_knee_servo", (0, 0, -p.thigh + (sv.length / 2 - sv.shaft_from_end)), (sv.width / 2, sv.height / 2, sv.length / 2), sv.mass))
        else:
            w("    " + _box(f"{leg}_knee_servo", (0, side * (0.006 + sv.height / 2), sv.length / 2 - sv.shaft_from_end), (sv.width / 2, sv.height / 2, sv.length / 2), sv.mass))
        knee_lo, knee_hi = (KNEE_RANGE[0], KNEE_RANGE[1]) if ks > 0 else (-KNEE_RANGE[1], -KNEE_RANGE[0])
        w(f'          <body name="{leg}_knee" pos="0 0 {-p.thigh:.5f}">')
        w(f'            <joint name="{leg}_knee" axis="0 -1 0" range="{knee_lo} {knee_hi}"/>')
        w(f'            <geom name="{leg}_shank" type="capsule" class="collision" fromto="0 0 0 0 0 {-p.shank:.5f}" size="0.007" mass="{sm["shank"]:.5f}"/>')
        w(f'            <geom name="{leg}_foot" type="sphere" class="collision" pos="0 0 {-p.shank:.5f}" size="{p.foot_radius:.4f}" mass="{sm["foot"]:.5f}" friction="1.0 0.02 0.001" rgba="0.12 0.12 0.12 1"/>')
        w(f'            <site name="{leg}_foot" pos="0 0 {-p.shank:.5f}" size="{p.foot_radius + 0.002:.4f}" type="sphere" rgba="0 0 0 0"/>')
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
    w(f'  <keyframe><key name="stand" qpos="{" ".join(qpos)}" ctrl="{" ".join(ctrl)}"/></keyframe>')
    w('</mujoco>')
    return "\n".join(out) + "\n"


def joint_names():
    return [f"{leg}_{j}" for leg in LEGS for j in ("abad", "hip", "knee")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="sim/cheetah_pup.xml")
    ap.add_argument("--bam", action="store_true", help="use BAM's implied stall torque as the limit")
    args = ap.parse_args(argv)
    p = locked()
    xml = build_mjcf(p, "bam" if args.bam else "datasheet")
    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml)
    gains = servo_gains("bam" if args.bam else "datasheet")
    print(f"wrote {path} for {p.name}: kp {gains['kp']:.1f} N·m/rad, kd {gains['kd']:.2f} N·m·s/rad, "
          f"limit {gains['limit']:.2f} N·m, no-load {gains['no_load_speed']:.1f} rad/s")


if __name__ == "__main__":
    main()
