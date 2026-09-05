"""Sizing analysis: mass/CoM model, quasi-static servo torques, joint speeds, electronics packaging."""

from __future__ import annotations

import math

from .design import DesignParams, MINI_CHEETAH
from .electronics import PI5, BATTERY_2S, PCB, IMU, WIRING_MASS
from .gait import LEGS, LEG_SIDE, LEG_FRONT, joint_trajectories, body_speed
from .kinematics import planar_fk, planar_ik, static_torques
from .servo import STS3215, CONTINUOUS_FRACTION, PEAK_FRACTION

G = 9.80665
PLA_DENSITY = 1240.0          # kg/m³
SHELL_THICKNESS = 0.0024      # m, ~3 perimeters
SHELL_OPENING_FACTOR = 0.8    # cutouts, vents, and lids remove some shell area
THIGH_LINEAR_MASS = 0.16      # kg/m, hollow printed box section ~20 x 12 mm
SHANK_LINEAR_MASS = 0.12      # kg/m, slimmer section
HIP_BRACKET_MASS = 0.025
FOOT_MASS = 0.008
LEG_HARDWARE_MASS = 0.012     # bearings, screws, pins per leg
EXTRA_TRANSMISSION_MASS = {"direct": 0.0, "belt": 0.012, "pushrod": 0.010}


def nominal_pose(p: DesignParams, front: bool = True):
    """(q_hip, q_knee) with the foot directly under the hip at the stance height."""
    return planar_ik(p.thigh, p.shank, 0.0, -p.stance_height, p.knee_sign(front))


def structure_masses(p: DesignParams) -> dict:
    shell_area = 2 * (p.shell_length * p.shell_width
                      + p.shell_length * p.body_height
                      + p.shell_width * p.body_height)
    shell = shell_area * SHELL_THICKNESS * PLA_DENSITY * SHELL_OPENING_FACTOR
    thigh = THIGH_LINEAR_MASS * p.thigh + 0.008
    shank = SHANK_LINEAR_MASS * p.shank + 0.005
    return {
        "shell": shell,
        "thigh": thigh,
        "shank": shank,
        "bracket": HIP_BRACKET_MASS,
        "foot": FOOT_MASS,
        "leg_hardware": LEG_HARDWARE_MASS + EXTRA_TRANSMISSION_MASS[p.architecture],
    }


def mass_model(p: DesignParams) -> dict:
    """Point-mass model in the body frame at the nominal stance. Returns components, total, CoM."""
    sv = STS3215
    sm = structure_masses(p)
    comps = []

    def add(name, m, x, y, z):
        comps.append({"name": name, "mass": m, "pos": (x, y, z)})

    # Layout (Phase 2 CAD): battery transverse on the floor between the abad servo cradles, the Pi 5
    # transverse above it on standoffs, the custom PCB (with the IMU) under the lid over the front
    # servos, wiring spread through the body.
    floor = p.body_z_offset - p.body_height / 2 + p.wall
    add("shell", sm["shell"], 0.0, 0.0, p.body_z_offset)
    add(BATTERY_2S.name, BATTERY_2S.mass, 0.0, 0.0, floor + BATTERY_2S.size[2] / 2)
    add(PI5.name, PI5.mass, 0.0, 0.0, floor + BATTERY_2S.size[2] + 0.003 + PI5.size[2] / 2)
    pcb_x = p.shell_length / 2 - p.wall - PCB.size[0] / 2
    pcb_z = p.body_z_offset + p.body_height / 2 - p.wall - 0.004 - PCB.size[2] / 2
    add(PCB.name, PCB.mass, pcb_x, 0.0, pcb_z)
    add(IMU.name, IMU.mass, pcb_x, 0.0, pcb_z)
    add("wiring", WIRING_MASS, 0.0, 0.0, p.body_z_offset)

    for leg in LEGS:
        side, front = LEG_SIDE[leg], LEG_FRONT[leg]
        sx = 1 if front else -1
        hx = sx * p.hip_to_hip / 2
        ay = side * p.abad_to_abad / 2
        ty = side * (p.abad_to_abad / 2 + p.abad_link)          # thigh plane
        q_hip, q_knee = nominal_pose(p, front)
        (kx, kz), (fx, fz) = planar_fk(p.thigh, p.shank, q_hip, q_knee)
        # abad servo inside the body corner, case extends inboard along x from the horn face
        add(f"{leg} abad servo", sv.mass, hx - sx * (p.hip_x_offset + sv.height / 2),
            ay - side * (sv.length / 2 - sv.shaft_from_end), 0.0)
        add(f"{leg} bracket", sm["bracket"], hx, side * (p.abad_to_abad / 2 + p.abad_link / 2), 0.01)
        # hip-pitch servo inboard of the thigh plane, case pointing up from the hip axis
        add(f"{leg} hip servo", sv.mass, hx, ty - side * (0.006 + sv.height / 2), 0.013)
        if p.architecture == "direct":
            # knee servo lives in the thigh, shaft at the knee, case pointing back up the thigh
            ux, uz = -kx / p.thigh, -kz / p.thigh
            add(f"{leg} knee servo", sv.mass, hx + kx + ux * 0.013, ty, kz + uz * 0.013)
        else:
            add(f"{leg} knee servo", sv.mass, hx, ty + side * (0.006 + sv.height / 2), 0.013)
        add(f"{leg} thigh", sm["thigh"], hx + kx / 2, ty, kz / 2)
        add(f"{leg} shank", sm["shank"], hx + (kx + fx) / 2, ty, (kz + fz) / 2)
        add(f"{leg} foot", sm["foot"], hx + fx, ty, fz)
        add(f"{leg} hardware", sm["leg_hardware"], hx + kx / 2, ty, kz / 2)

    total = sum(c["mass"] for c in comps)
    com = tuple(sum(c["mass"] * c["pos"][i] for c in comps) / total for i in range(3))
    return {"components": comps, "total": total, "com": com}


def torque_report(p: DesignParams, mass: float | None = None) -> dict:
    """Quasi-static joint and servo torques for standing (4 legs) and trot peak (2 legs, dynamic)."""
    sv = STS3215
    m = mass if mass is not None else mass_model(p)["total"]
    weight = m * G
    stall = sv.stall_torque
    cases = {}
    for case, force, sweep in (
        ("stand", weight / 4, 0.0),
        ("trot_peak", weight / 2 * p.dynamic_factor, p.step_length / 2),
    ):
        worst = {"abad": 0.0, "hip": 0.0, "knee_joint": 0.0}
        xs = [0.0] if sweep == 0.0 else [sweep * (i / 10 - 1) for i in range(21)]
        for front in (True, False):
            ks = p.knee_sign(front)
            for x in xs:
                q_hip, q_knee = planar_ik(p.thigh, p.shank, x, -p.stance_height, ks)
                ta, th, tk = static_torques(p.thigh, p.shank, p.abad_link, q_hip, q_knee, force,
                                            p.lateral_shift if case == "trot_peak" else 0.0)
                worst["abad"] = max(worst["abad"], abs(ta))
                worst["hip"] = max(worst["hip"], abs(th))
                worst["knee_joint"] = max(worst["knee_joint"], abs(tk))
        knee_servo = worst["knee_joint"] / p.knee_ratio
        limit = stall * (CONTINUOUS_FRACTION if case == "stand" else PEAK_FRACTION)
        cases[case] = {
            "force_per_leg": force,
            "abad": worst["abad"],
            "hip": worst["hip"],
            "knee_joint": worst["knee_joint"],
            "knee_servo": knee_servo,
            "limit": limit,
            "abad_frac": worst["abad"] / stall,
            "hip_frac": worst["hip"] / stall,
            "knee_servo_frac": knee_servo / stall,
            "ok": max(worst["abad"], worst["hip"], knee_servo) <= limit,
        }
    cases["stall_torque"] = stall
    cases["mass"] = m
    return cases


def speed_report(p: DesignParams, gait: str = "trot") -> dict:
    """Peak joint and servo speeds over a gait cycle versus the firmware velocity cap."""
    sv = STS3215
    traj = joint_trajectories(p, gait)
    peak = {"abad": 0.0, "hip": 0.0, "knee_joint": 0.0}
    for leg in LEGS:
        for da, dh, dk in traj[leg]["dq"]:
            peak["abad"] = max(peak["abad"], abs(da))
            peak["hip"] = max(peak["hip"], abs(dh))
            peak["knee_joint"] = max(peak["knee_joint"], abs(dk))
    knee_servo = peak["knee_joint"] * p.knee_ratio
    worst = max(peak["abad"], peak["hip"], knee_servo)
    return {
        "gait": gait,
        "body_speed": body_speed(p, gait),
        "abad": peak["abad"],
        "hip": peak["hip"],
        "knee_joint": peak["knee_joint"],
        "knee_servo": knee_servo,
        "cap": sv.max_velocity,
        "worst_frac": worst / sv.max_velocity,
        "ok": worst <= sv.max_velocity,
    }


def _footprint(item, bay_x: float, bay_y: float):
    """Smallest x-extent orientation of `item` that fits a bay, or None. Returns (x, y, rotated)."""
    l, w = item.size[0], item.size[1]
    options = []
    if l <= bay_x and w <= bay_y:
        options.append((l, w, False))
    if w <= bay_x and l <= bay_y:
        options.append((w, l, True))
    return min(options) if options else None


def packaging_report(p: DesignParams) -> dict:
    """Does the electronics fit in the shell around the four abad servos?

    Layout (Phase 2 CAD): the abad servos sit in the four corners against the end walls; the battery
    lies on the floor in the centre bay between the front and rear servo cradles with the Pi 5
    stacked above it; the custom PCB hangs from the lid over one pair of servos.
    """
    sv = STS3215
    inner_len = p.shell_length - 2 * p.wall
    inner_wid = p.shell_width - 2 * p.wall
    inner_h = p.body_height - 2 * p.wall
    cradle = sv.height + 0.0021 + 0.0025                               # case depth + idler + rib
    bottom_gap = p.shell_length - 2 * p.wall - 2 * cradle              # centre bay length
    battery = _footprint(BATTERY_2S, bottom_gap, inner_wid)
    pi = _footprint(PI5, bottom_gap, inner_wid)
    pcb = _footprint(PCB, cradle + 0.006, inner_wid)                   # may overhang the cradle a little
    stack_needed = BATTERY_2S.size[2] + 0.003 + PI5.size[2] + 0.004    # floor to lid over the centre bay
    pcb_needed = sv.width / 2 + 0.003 + PCB.size[2] + 0.004            # servo top, clearance, PCB, lid bosses
    return {
        "inner_length": inner_len,
        "inner_width": inner_wid,
        "inner_height": inner_h,
        "bottom_gap": bottom_gap,
        "battery_fits": battery is not None,
        "battery_footprint": battery,
        "top_needed": stack_needed,
        "top_fits": pi is not None and stack_needed <= inner_h,
        "pi_footprint": pi,
        "pcb_footprint": pcb,
        "height_needed": pcb_needed,
        "height_fits": pcb is not None and (p.body_height / 2 - p.wall) >= pcb_needed,
    }


def metrics(p: DesignParams) -> dict:
    sv = STS3215
    mm = mass_model(p)
    tq = torque_report(p, mm["total"])
    sp = speed_report(p, "trot")
    pk = packaging_report(p)
    q_hip, q_knee = nominal_pose(p, True)
    outboard = (sv.height + 0.006) if p.architecture != "direct" else 0.006
    overall_width = 2 * (p.abad_to_abad / 2 + p.abad_link + outboard + 0.010)
    return {
        "name": p.name,
        "architecture": p.architecture,
        "mass": mm["total"],
        "com": mm["com"],
        "leg_length": p.leg_length,
        "stance_height": p.stance_height,
        "nominal_hip_deg": math.degrees(q_hip),
        "nominal_knee_deg": math.degrees(q_knee),
        "body_top_height": p.stance_height + p.body_z_offset + p.body_height / 2,
        "overall_length": p.hip_to_hip + 2 * (0.03 + p.foot_radius),
        "overall_width": overall_width,
        "shell_length": p.shell_length,
        "ratios": p.ratios(),
        "torque": tq,
        "speed": sp,
        "packaging": pk,
        "servo_count": 12,
        "servo_mass_fraction": 12 * sv.mass / mm["total"],
        "trot_speed_body_lengths": sp["body_speed"] / p.hip_to_hip,
    }


def summary_line(p: DesignParams) -> str:
    m = metrics(p)
    t = m["torque"]["trot_peak"]
    return (f"{p.name}: mass {m['mass']:.2f} kg, hip {p.stance_height*1000:.0f} mm, "
            f"legs {p.thigh*1000:.0f}/{p.shank*1000:.0f} mm, knee servo {t['knee_servo']:.2f} N·m "
            f"({t['knee_servo_frac']*100:.0f}% stall), hip {t['hip']:.2f} N·m, abad {t['abad']:.2f} N·m, "
            f"speed {m['speed']['worst_frac']*100:.0f}% cap, fits={all([m['packaging']['battery_fits'], m['packaging']['top_fits'], m['packaging']['height_fits']])}")
