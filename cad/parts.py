"""Printed parts of the Cheetah Pup, each built in its MuJoCo body frame (mm).

Naming: `side` is +1 for left legs (+y), -1 for right; `front` selects the x sign of the hip.
Mounting convention at every joint: the fixed part carries a 3 mm plate on the servo's top face
(4 × M2 into the case) with a Ø21 bore in which the Ø20 horn disc rides as a plain bearing; the
moving part bolts to the horn's 4 × Ø2.5 holes at r = 7 and clears the plate by the disc's 0.95 mm.
"""

from __future__ import annotations

from build123d import (Align, Axis, Box, Cylinder, Location, Plane, Polygon, Pos, Rot, Shape, Sphere,
                       extrude, mirror)

from . import params as C
from . import servo as SV


def _box(x0, x1, y0, y1, z0, z1) -> Shape:
    """Axis-aligned box between two corners; the corner order does not matter."""
    x0, x1 = sorted((x0, x1)); y0, y1 = sorted((y0, y1)); z0, z1 = sorted((z0, z1))
    return Pos((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2) * Box(x1 - x0, y1 - y0, z1 - z0)


def _cyl_x(r, x0, x1, y=0.0, z=0.0) -> Shape:
    x0, x1 = sorted((x0, x1))
    return Pos((x0 + x1) / 2, y, z) * Rot(0, 90, 0) * Cylinder(r, x1 - x0)


def _cyl_y(r, y0, y1, x=0.0, z=0.0) -> Shape:
    y0, y1 = sorted((y0, y1))
    return Pos(x, (y0 + y1) / 2, z) * Rot(90, 0, 0) * Cylinder(r, y1 - y0)


def _cyl_z(r, z0, z1, x=0.0, y=0.0) -> Shape:
    z0, z1 = sorted((z0, z1))
    return Pos(x, y, (z0 + z1) / 2) * Cylinder(r, z1 - z0)


def _span(s: int, a: float, b: float):
    """Ordered (min, max) of s*a and s*b."""
    lo, hi = sorted((s * a, s * b))
    return lo, hi


# ---------------------------------------------------------------- trunk

def abad_servo_frame(side: int, front: bool) -> SV.ServoFrame:
    sx = 1 if front else -1
    return SV.ServoFrame(origin=(sx * C.ABAD_SEAT_X, side * C.ABAD_Y, 0.0), l_dir=(0.0, -side, 0.0), a_dir=(sx, 0.0, 0.0))


def trunk_tub() -> Shape:
    """Lower shell: floor, side and end walls up to the split, servo cradles, battery rails, Pi
    standoffs, lid bosses, and the abad servo bores and mounting holes in the end walls."""
    z0, z1 = -C.BODY_H / 2, C.SPLIT_Z
    outer = _box(-C.SHELL_LEN / 2, C.SHELL_LEN / 2, -C.SHELL_W / 2, C.SHELL_W / 2, z0, z1)
    inner = _box(-C.END_WALL_INNER, C.END_WALL_INNER, -C.SHELL_W / 2 + C.WALL, C.SHELL_W / 2 - C.WALL, z0 + C.WALL, z1 + 1)
    tub = outer - inner
    cradle_depth = SV.CASE_A[1] - SV.IDLER_DISC_A[0]   # 34.7: case + idler
    for front in (True, False):
        sx = 1 if front else -1
        x_in = sx * (C.END_WALL_INNER - SV.STEP_A - cradle_depth - 2.0)   # inboard end of the cradle region
        xa, xb = sorted((x_in, sx * C.END_WALL_INNER))
        # shelf under both servos of this end, full width, top 0.5 mm under the case bottom, with a
        # window under each case so only a rim carries the servo
        shelf = _box(xa, xb, -C.SHELL_W / 2 + C.WALL, C.SHELL_W / 2 - C.WALL, -SV.CASE_W - 3.5, -SV.CASE_W - 0.5)
        for side in (1, -1):
            shelf = shelf - _box(xa + 6.0, xb - 6.0, side * 9.0, side * (C.ABAD_Y + 6.0), -SV.CASE_W - 4.0, -SV.CASE_W)
        tub = tub + shelf
        # centre rib between the two cases, floor to split
        tub = tub + _box(xa, xb, -2.6, 2.6, C.FLOOR_Z - 0.1, z1)
        # inboard rib at the cradle end, windowed for wiring and weight
        xi0, xi1 = sorted((x_in, x_in + sx * 2.5))
        rib = _box(xi0, xi1, -C.SHELL_W / 2 + C.WALL, C.SHELL_W / 2 - C.WALL, C.FLOOR_Z - 0.1, z1)
        for side in (1, -1):
            rib = rib - _box(xi0 - 1, xi1 + 1, side * 8.0, side * 40.0, C.FLOOR_Z + 6.0, z1 - 6.0)
        tub = tub + rib
        for side in (1, -1):
            f = abad_servo_frame(side, front)
            # horn bearing bore through the end wall, and the four M2 mounting holes
            tub = tub - _cyl_x(SV.BEARING_BORE_DIA / 2, sx * (C.END_WALL_OUTER + 1), sx * (C.END_WALL_INNER - 1), y=side * C.ABAD_Y, z=0.0)
            for (l, w) in SV.MOUNT_HOLES_TOP:
                pt = f.to_host(l, w, 0.0)
                tub = tub - _cyl_x(SV.M2_CLEARANCE_DIA / 2, sx * (C.END_WALL_OUTER + 1), sx * (C.END_WALL_INNER - 1), y=pt.Y, z=pt.Z)
    # battery rails
    for x in (-22.0, 22.0):
        tub = tub + _box(x - 1.5, x + 1.5, -30.0, 30.0, C.FLOOR_Z - 0.1, C.FLOOR_Z + 6.0)
    # Pi standoffs (M2.5 self-tapping), floor to the Pi board underside
    pi_z = C.PI_C[2] - C.PI[2] / 2
    for (x, y) in C.PI_HOLES:
        tub = tub + _cyl_z(3.0, C.FLOOR_Z - 0.1, pi_z, x, y) - _cyl_z(1.1, pi_z - 8.0, pi_z + 0.1, x, y)
    # lid bosses
    for (x, y) in C.LID_BOSSES:
        tub = tub + _cyl_z(2.75, C.FLOOR_Z - 0.1, z1, x, y) - _cyl_z(1.0, z1 - 10.0, z1 + 0.1, x, y)
    return tub


def trunk_lid() -> Shape:
    """Upper shell from the split to the top, with vent slots, lid screws, and the PCB bosses."""
    z0, z1 = C.SPLIT_Z, C.BODY_H / 2
    outer = _box(-C.SHELL_LEN / 2, C.SHELL_LEN / 2, -C.SHELL_W / 2, C.SHELL_W / 2, z0, z1)
    inner = _box(-C.END_WALL_INNER, C.END_WALL_INNER, -C.SHELL_W / 2 + C.WALL, C.SHELL_W / 2 - C.WALL, z0 - 1, z1 - C.WALL)
    lid = outer - inner
    for (x, y) in C.LID_BOSSES:
        lid = lid - _cyl_z(1.3, z1 - C.WALL - 1, z1 + 1, x, y)
    for i in range(-2, 3):
        lid = lid - _box(-30.0 + i * 12.0 - 1.5, -30.0 + i * 12.0 + 1.5, -30.0, 30.0, z1 - C.WALL - 1, z1 + 1)
    pcb_top = C.PCB_C[2] + C.PCB_ENV[2] / 2
    for (x, y) in C.PCB_HOLES:
        lid = lid + _cyl_z(2.75, pcb_top, z1 - C.WALL + 0.1, x, y) - _cyl_z(1.0, pcb_top - 0.1, pcb_top + 6.0, x, y)
    return lid


# ---------------------------------------------------------------- abad bracket (abad body frame)

def hip_servo_frame(side: int) -> SV.ServoFrame:
    return SV.ServoFrame(origin=(0.0, side * (C.ABAD_LINK - SV.HORN_DISC_A[1]), 0.0), l_dir=(0.0, 0.0, 1.0), a_dir=(0.0, side, 0.0))


def abad_bracket(side: int, front: bool) -> Shape:
    """Rotates about x on the abad horn; carries the hip-pitch servo with its horn at the thigh plane."""
    sx = 1 if front else -1
    s = side
    seat_x = -sx * (C.HIP_X_OFFSET + WALL_TO_SEAT())
    disc_face = seat_x + sx * SV.HORN_DISC_A[1]
    bp0, bp1 = sorted((disc_face, disc_face + sx * C.PLATE))
    # back plate on the abad horn disc: round top, extended down to meet the bar
    back = _cyl_x(13.0, bp0, bp1) + _box(bp0, bp1, -13.0, 13.0, -22.0, 0.0)
    back = back - _cyl_x(SV.HORN_CENTER_DIA / 2 + 0.1, bp0 - 1, bp1 + 1)
    for (dy, dz) in [(SV.HORN_HOLE_R, 0), (-SV.HORN_HOLE_R, 0), (0, SV.HORN_HOLE_R), (0, -SV.HORN_HOLE_R)]:
        back = back - _cyl_x(SV.M2_CLEARANCE_DIA / 2, bp0 - 1, bp1 + 1, y=dy, z=dz)
    hf = hip_servo_frame(s)
    plate_y0, plate_y1 = _span(s, C.ABAD_LINK - SV.HORN_DISC_A[1] + SV.PLATE_A[0], C.ABAD_LINK - SV.HORN_DISC_A[1] + SV.PLATE_A[1])
    far_x = sx * (SV.CASE_W + 2.5)
    x_lo, x_hi = sorted((bp0 if sx > 0 else bp1, far_x))
    # bar under the servo, from the back plate out to the servo plate
    y_lo, y_hi = _span(s, -13.0, C.ABAD_LINK - SV.HORN_DISC_A[1] + SV.PLATE_A[1])
    bar = _box(x_lo, x_hi, y_lo, y_hi, -18.0, -14.0)
    # servo plate on the hip servo's top face
    plate = _box(-(SV.CASE_W + 2.5), SV.CASE_W + 2.5, plate_y0, plate_y1, -18.0, 36.0)
    plate = plate - _cyl_y(SV.BEARING_BORE_DIA / 2, plate_y0 - 1, plate_y1 + 1)
    for (l, w) in SV.MOUNT_HOLES_TOP:
        pt = hf.to_host(l, w, 0.0)
        plate = plate - _cyl_y(SV.M2_CLEARANCE_DIA / 2, plate_y0 - 1, plate_y1 + 1, x=pt.X, z=pt.Z)
    # gusset: the back plate's lower edge thickened along the full bar length
    gusset = _box(bp0, bp1, y_lo, y_hi, -22.0, -14.0)
    return back + bar + plate + gusset


def WALL_TO_SEAT() -> float:
    """Distance from the end wall's outer face back to the abad servo horn seat (wall + step)."""
    return C.WALL + SV.STEP_A


# ---------------------------------------------------------------- thigh (hip body frame)

def knee_servo_frame(side: int) -> SV.ServoFrame:
    # gear end down the leg, horn outboard, case inboard of the thigh plate
    return SV.ServoFrame(origin=(0.0, -side * SV.STEP_A, -C.THIGH), l_dir=(0.0, 0.0, -1.0), a_dir=(0.0, side, 0.0))


def thigh(side: int) -> Shape:
    s = side
    y0, y1 = _span(s, 0.0, C.THIGH_PLATE)
    kf = knee_servo_frame(s)
    cr_top = -C.THIGH - SV.CASE_L[0] + 2.5          # 2.5 mm above the servo's short end
    cr_bot = -C.THIGH - SV.CASE_L[1] - 2.5          # 2.5 mm below the gear end
    cw = SV.CASE_W + C.CLEAR + 2.5                  # cradle half width
    # plate: hip pad, strip, cradle face
    plate = _cyl_y(13.0, y0, y1) + _box(-10.0, 10.0, y0, y1, cr_top - 1.0, 0.0) + _box(-cw, cw, y0, y1, cr_bot, cr_top)
    # hip horn pattern
    plate = plate - _cyl_y(SV.HORN_CENTER_DIA / 2 + 0.1, y0 - 1, y1 + 1)
    for (dx, dz) in [(SV.HORN_HOLE_R, 0), (-SV.HORN_HOLE_R, 0), (0, SV.HORN_HOLE_R), (0, -SV.HORN_HOLE_R)]:
        plate = plate - _cyl_y(SV.M2_CLEARANCE_DIA / 2, y0 - 1, y1 + 1, x=dx, z=dz)
    # knee bearing bore and servo mounting holes
    plate = plate - _cyl_y(SV.BEARING_BORE_DIA / 2, y0 - 1, y1 + 1, z=-C.THIGH)
    for (l, w) in SV.MOUNT_HOLES_TOP:
        pt = kf.to_host(l, w, 0.0)
        plate = plate - _cyl_y(SV.M2_CLEARANCE_DIA / 2, y0 - 1, y1 + 1, x=pt.X, z=pt.Z)
    # cradle walls and end stops around the knee servo case (open inboard for the idler and wiring)
    wy0, wy1 = _span(s, -(SV.CASE_A[1] - SV.IDLER_DISC_A[0]) - SV.STEP_A - 0.5, 0.0)   # plate to just past the idler disc
    walls = _box(-cw, -cw + 2.5, wy0, wy1, cr_bot, cr_top) + _box(cw - 2.5, cw, wy0, wy1, cr_bot, cr_top)
    stops = _box(-cw, cw, wy0, wy1, cr_top - 2.0, cr_top) + _box(-cw, cw, wy0, wy1, cr_bot, cr_bot + 2.0)
    cradle = walls + stops
    cradle = cradle - SV.placed(SV.servo_pocket(C.CLEAR), kf)
    return plate + cradle


# ---------------------------------------------------------------- shank (knee body frame)

def shank(side: int) -> Shape:
    s = side
    pad_in = C.THIGH_PLATE + SV.DISC_PROUD - 0.05    # knee horn disc face, outboard of the thigh plate
    pad_y0, pad_y1 = _span(s, pad_in, pad_in + C.PAD)
    pad = _cyl_y(13.0, pad_y0, pad_y1)
    pad = pad - _cyl_y(SV.HORN_CENTER_DIA / 2 + 0.1, pad_y0 - 1, pad_y1 + 1)
    for (dx, dz) in [(SV.HORN_HOLE_R, 0), (-SV.HORN_HOLE_R, 0), (0, SV.HORN_HOLE_R), (0, -SV.HORN_HOLE_R)]:
        pad = pad - _cyl_y(SV.M2_CLEARANCE_DIA / 2, pad_y0 - 1, pad_y1 + 1, x=dx, z=dz)
    # beam profile in the y-z plane: alongside the thigh cradle for the first 42 mm, then a jog
    # inboard so the foot lands near the thigh plane
    jog0, jog1 = -42.0, -54.0
    ya, yb = pad_in, pad_in + 6.5                    # outboard segment
    yc, yd = C.FOOT_Y - 3.25, C.FOOT_Y + 3.25        # inboard segment, centred on the foot
    pts = [(ya, 8.0), (yb, 8.0), (yb, jog0), (yd, jog1), (yd, -C.SHANK + 2.0), (yc, -C.SHANK + 2.0), (yc, jog1), (ya, jog0)]
    pts = [(s * y, z) for (y, z) in pts]
    beam = extrude(Plane.YZ * Polygon(*pts, align=None), amount=7.0, both=True)
    tip = _cyl_z(6.0, -C.SHANK - 4.0, -C.SHANK + 6.0, y=s * C.FOOT_Y)
    return pad + beam + tip


def foot(side: int) -> Shape:
    """TPU foot: a sphere on the shank tip (modelled at the foot centre used by the sim)."""
    return Pos(0.0, side * C.FOOT_Y, -C.SHANK) * Sphere(C.FOOT_R)


# ---------------------------------------------------------------- helpers

def mirrored_y(shape: Shape) -> Shape:
    return mirror(shape, about=Plane.XZ)


def mirrored_x(shape: Shape) -> Shape:
    return mirror(shape, about=Plane.YZ)
