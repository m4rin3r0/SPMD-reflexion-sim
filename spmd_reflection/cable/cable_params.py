"""Data structures for cable modeling."""

from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CableParams:
    """Distributed parameters of a transmission line, in SI per meter.
    
    Attributes:
        l_per_m: Series inductance per meter (H/m).
        c_per_m: Shunt capacitance per meter (F/m).
        rdc_per_m: DC series resistance per meter (Ω/m).
        rskin_per_m: Skin effect coefficient per meter (Ω/(m·√Hz)).
            Series resistance scales as rdc + rskin·√f.
    """
    l_per_m: float
    c_per_m: float
    rdc_per_m: float
    rskin_per_m: float

    def z0_at(self, frequency_hz:float) -> complex:
        """Characteristic impedance of the line at a given frequency.
        
        Computes Z₀ = √(Z_series / Y_shunt) from the distributed parameters.
        Returns a complex value in general; at high frequencies where losses
        are negligible, the imaginary part approaches zero.
        """
        omega = 2 * math.pi * frequency_hz
        z_series = self.rdc_per_m + self.rskin_per_m * math.sqrt(frequency_hz) + 1j * omega * self.l_per_m
        y_shunt = 1j * omega * self.c_per_m
        return (z_series / y_shunt) ** 0.5