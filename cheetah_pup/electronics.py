"""Bounding volumes and masses of the control and power hardware carried in the body."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Component:
    name: str
    size: tuple[float, float, float]  # m, (x, y, z) footprint in the body frame
    mass: float                       # kg


# Raspberry Pi 5 board is 85 x 56 mm; height allows the official active cooler.
PI5 = Component("Raspberry Pi 5 + active cooler", (0.085, 0.056, 0.025), 0.065)

# Two 18650 cells side by side (18 x 65 mm each, ~47 g) in a holder with a 2S BMS, XT30 lead.
BATTERY_2S = Component("2S 18650 pack + BMS", (0.070, 0.040, 0.022), 0.125)

# Custom servo-bus / power / sensor breakout board, with connectors.
PCB = Component("Custom bus/power/sensor PCB", (0.060, 0.045, 0.012), 0.025)

# BNO055 breakout, matches Open Duck Mini v2's mesh (20 x 27 x 3 mm).
IMU = Component("BNO055 IMU breakout", (0.027, 0.020, 0.004), 0.005)

WIRING_MASS = 0.060  # kg, servo leads, bus cables, connectors, switch
