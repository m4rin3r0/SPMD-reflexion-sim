"""Compute Y-parameters of a lossy transmission line segment.

Uses the standard telegrapher's equations with distributed RLC parameters
(series resistance with skin effect, series inductance, shunt capacitance).
The line is modeled as a reciprocal, symmetric 2-port.
"""

from __future__ import annotations
import numpy as np
from spmd_reflection.cable.cable_params import CableParams


def compute_y_params(length_m:float, cable_params:CableParams, frequency_hz:np.ndarray) -> np.ndarray:
    """Return Y-parameter matrix of a transmission line segment.

    Args:
        length_m: Length of the segment in meters. Must be positive.
        cable_params: Distributed parameters of the line (per meter).
        frequency_hz: 1D array of frequencies in Hz. All values must be positive.

    Returns:
        Complex array of shape (n_freq, 2, 2) with Y-parameters at each frequency.
        Port indexing: 0 = left end, 1 = right end of the segment.

    Raises:
        ValueError: If length is non-positive or any frequency is non-positive.
    """
    if length_m <= 0:
        raise ValueError(f"length_m must be positive, got {length_m}")
    if np.any(frequency_hz <= 0):
        raise ValueError("all frequencies must be positive")

    omega = 2 * np.pi * frequency_hz

    # Distributed impedance and admittance per meter at each frequency.
    z_series = (
        cable_params.rdc_per_m
        + cable_params.rskin_per_m * np.sqrt(frequency_hz)
        + 1j * omega * cable_params.l_per_m
    )
    y_shunt = 1j * omega * cable_params.c_per_m

    # Propagation constant and characteristic impedance at each frequency.
    gamma = np.sqrt(z_series * y_shunt)
    z0 = np.sqrt(z_series / y_shunt)

    # Line Y-parameters from telegrapher's equations.
    gl = gamma * length_m
    sinh_gl = np.sinh(gl)
    cosh_gl = np.cosh(gl)

    y11 = cosh_gl / (z0 * sinh_gl)
    y12 = -1.0 / (z0 * sinh_gl)

    # Assemble (n_freq, 2, 2) array. Line is symmetric and reciprocal,
    # so y22 = y11 and y21 = y12.
    n_freq = len(frequency_hz)
    y = np.empty((n_freq, 2, 2), dtype=complex)
    y[:, 0, 0] = y11
    y[:, 0, 1] = y12
    y[:, 1, 0] = y12
    y[:, 1, 1] = y11

    return y