"""Feetech STS3215 as measured from Open Duck Mini v2's case meshes (see docs/design/04-cad-detail.md).

Servo frame (millimetres): origin on the shaft axis at the horn-seat plane (the case's top face
around the horn); L = case length axis (+L toward the gear end, 35.1 mm; the short end is at
-10.1 mm); W = width axis (±12.36); A = shaft axis (+A toward the horn).

Instances are placed with `place(origin, l_dir, a_dir)`, which builds the frame from where the
gear end and the horn should point.
"""

from __future__ import annotations

from dataclasses import dataclass

from build123d import Box, Cylinder, Location, Plane, Pos, Rot, Vector, Shape

CASE_L = (-10.1, 35.1)      # mm along L
CASE_W = 12.36              # half width
CASE_A = (-32.6, 0.0)       # bottom face to horn seat
STEP_L = (5.45, 35.1)       # raised step on the top face, 1.1 mm
STEP_A = 1.1
BUMP_L = (9.6, 35.1)        # gear cover bump, |W| < 7, to 3.1 mm
BUMP_W = 7.0
BUMP_A = 3.1
HORN_HUB_R, HORN_HUB_A = 3.05, (0.0, 2.1)
HORN_DISC_R, HORN_DISC_A = 10.0, (2.1, 5.05)
IDLER_DISC_R, IDLER_DISC_A = 10.0, (-34.7, -32.6)
HORN_HOLE_R, HORN_HOLE_DIA = 7.0, 2.5      # 4 holes on the disc at (±7, 0) and (0, ±7)
HORN_CENTER_DIA = 3.2
MOUNT_HOLES_TOP = [(8.3, 10.25), (8.3, -10.25), (29.0, 10.25), (29.0, -10.25)]   # (L, W), M2 tapped
MOUNT_HOLES_BOTTOM = [(8.3, 10.25), (8.3, -10.25)]
CASE_SCREWS = [(32.7, 10.0), (32.7, -10.0), (-7.1, 9.35), (-7.1, -9.35)]         # through the case
MASS = 0.055                # kg

# Printed-part interface conventions derived from the above
BEARING_BORE_DIA = 21.0     # the Ø20 horn disc rides in this bore of the fixed part
PLATE_A = (STEP_A, STEP_A + 3.0)   # a 3 mm mounting plate sits on the step: A in [1.1, 4.1]
DISC_PROUD = HORN_DISC_A[1] - PLATE_A[1]   # 0.95 mm of disc beyond the plate for the moving part to seat on
M2_CLEARANCE_DIA = 2.4
M2_TAP_DIA = 1.7


@dataclass(frozen=True)
class ServoFrame:
    origin: tuple      # mm, in the host body frame
    l_dir: tuple       # unit vector of +L
    a_dir: tuple       # unit vector of +A (horn direction)

    @property
    def plane(self) -> Plane:
        return Plane(origin=Vector(*self.origin), x_dir=Vector(*self.l_dir), z_dir=Vector(*self.a_dir))

    @property
    def w_dir(self) -> tuple:
        v = Vector(*self.a_dir).cross(Vector(*self.l_dir))
        return (v.X, v.Y, v.Z)

    def to_host(self, l: float, w: float, a: float) -> Vector:
        return Vector(*self.origin) + Vector(*self.l_dir) * l + Vector(*self.w_dir) * w + Vector(*self.a_dir) * a


def _box(l0, l1, w0, w1, a0, a1) -> Shape:
    return Pos((l0 + l1) / 2, (w0 + w1) / 2, (a0 + a1) / 2) * Box(l1 - l0, w1 - w0, a1 - a0)


def _cyl(r, a0, a1, l=0.0, w=0.0) -> Shape:
    return Pos(l, w, (a0 + a1) / 2) * Cylinder(r, a1 - a0)


def servo_solid(with_horn: bool = True, with_idler: bool = True) -> Shape:
    """The servo in its own frame (L, W, A) → (X, Y, Z)."""
    s = _box(CASE_L[0], CASE_L[1], -CASE_W, CASE_W, CASE_A[0], CASE_A[1])
    s = s + _box(STEP_L[0], STEP_L[1], -CASE_W, CASE_W, 0.0, STEP_A)
    s = s + _box(BUMP_L[0], BUMP_L[1], -BUMP_W, BUMP_W, STEP_A, BUMP_A)
    if with_horn:
        s = s + _cyl(HORN_HUB_R, *HORN_HUB_A) + _cyl(HORN_DISC_R, *HORN_DISC_A)
        for (l, w) in [(HORN_HOLE_R, 0), (-HORN_HOLE_R, 0), (0, HORN_HOLE_R), (0, -HORN_HOLE_R)]:
            s = s - _cyl(HORN_HOLE_DIA / 2, HORN_DISC_A[0] - 0.1, HORN_DISC_A[1] + 0.1, l, w)
    if with_idler:
        s = s + _cyl(IDLER_DISC_R, *IDLER_DISC_A)
    return s


def servo_pocket(clearance: float = 0.4, idler: bool = True) -> Shape:
    """Volume to subtract from a part that cradles the case (no horn: the horn lives in the bore)."""
    c = clearance
    s = _box(CASE_L[0] - c, CASE_L[1] + c, -CASE_W - c, CASE_W + c, CASE_A[0] - c, BUMP_A + c)
    if idler:
        s = s + _cyl(IDLER_DISC_R + c, IDLER_DISC_A[0] - c, IDLER_DISC_A[1] + c)
    return s


def horn_pattern_cutter(a0: float, a1: float, center_dia: float = HORN_CENTER_DIA) -> Shape:
    """Holes matching the horn disc: 4 × M2 clearance at r = 7 plus the centre screw, spanning A in [a0, a1]."""
    s = _cyl(center_dia / 2, a0, a1)
    for (l, w) in [(HORN_HOLE_R, 0), (-HORN_HOLE_R, 0), (0, HORN_HOLE_R), (0, -HORN_HOLE_R)]:
        s = s + _cyl(M2_CLEARANCE_DIA / 2, a0, a1, l, w)
    return s


def mount_pattern_cutter(a0: float, a1: float, holes=MOUNT_HOLES_TOP, dia: float = M2_CLEARANCE_DIA) -> Shape:
    """Clearance holes through a plate on the top face, matching the case's tapped M2 holes."""
    s = None
    for (l, w) in holes:
        c = _cyl(dia / 2, a0, a1, l, w)
        s = c if s is None else s + c
    return s


def bearing_bore_cutter(a0: float, a1: float) -> Shape:
    return _cyl(BEARING_BORE_DIA / 2, a0, a1)


def placed(shape: Shape, frame: ServoFrame) -> Shape:
    """Move a shape built in the servo frame into the host body frame."""
    return frame.plane.location * shape
