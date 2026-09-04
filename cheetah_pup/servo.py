"""Feetech STS3215 (7.4 V) serial-bus servo.

Geometry was measured directly from the STS3215 case meshes shipped in Open Duck Mini v2's MuJoCo
model (vendor/open_duck_mini/mini_bdx/robots/open_duck_mini_v2/wj-wk00-012*.stl, drive_palonier.stl,
passive_palonier.stl). The electrical model is Rhoban BAM's identified parameter set for this servo
at 7.4 V (vendor/bam/bam/params/feetech_sts3215_7_4V/m1.json).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Servo:
    name: str
    length: float           # m, case dimension along the axis the horn sits near one end of
    width: float            # m, case dimension across
    height: float           # m, case dimension along the output shaft, including the shaft boss
    shaft_from_end: float   # m, output-shaft axis distance from the near end, along `length`
    horn_diameter: float    # m, drive horn disc
    horn_thickness: float   # m
    idler_thickness: float  # m, passive disc on the rear shaft (STS3215 supports a rear pivot)
    mass: float             # kg
    stall_torque: float     # N·m, datasheet rating at nominal voltage (design basis)
    max_velocity: float     # rad/s, firmware rate limit on the internal target (BAM)
    kt: float               # N·m/A at the output (BAM)
    resistance: float       # ohm (BAM)
    armature: float         # kg·m², reflected rotor inertia at the output (BAM)
    friction_base: float    # N·m, Coulomb term (BAM)
    friction_viscous: float # N·m·s/rad (BAM)
    command_delay: float    # s (BAM)
    vin: float              # V nominal
    max_pwm: float          # duty-cycle cap (BAM)

    @property
    def shaft_to_far_end(self) -> float:
        return self.length - self.shaft_from_end

    def model_stall_torque(self) -> float:
        """Stall torque implied by BAM's electrical model (optimistic vs. the datasheet rating)."""
        return self.kt * self.max_pwm * self.vin / self.resistance - self.friction_base

    def available_torque(self, omega: float) -> float:
        """Torque available at output speed `omega` (rad/s) per the BAM voltage-controlled model."""
        v = self.max_pwm * self.vin
        return max(0.0, self.kt * (v - self.kt * abs(omega)) / self.resistance
                   - self.friction_base - self.friction_viscous * abs(omega))


STS3215 = Servo(
    name="Feetech STS3215 (7.4 V)",
    length=0.04522,
    width=0.02472,
    height=0.0357,
    shaft_from_end=0.0096,
    horn_diameter=0.020,
    horn_thickness=0.005,
    idler_thickness=0.003,
    mass=0.055,
    stall_torque=19.5 * 9.80665 / 100.0,  # 19.5 kg·cm datasheet -> 1.912 N·m
    max_velocity=5.288,
    kt=1.1776,
    resistance=2.4787,
    armature=0.02608,
    friction_base=0.05152,
    friction_viscous=0.05989,
    command_delay=0.00105,
    vin=7.4,
    max_pwm=0.97,
)

# Design allowances, as fractions of the datasheet stall torque. Continuous is a thermal comfort
# limit for holding poses; peak is for transient loads during gait.
CONTINUOUS_FRACTION = 0.25
PEAK_FRACTION = 0.60
