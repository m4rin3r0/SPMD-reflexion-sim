"""Touchstone S-parameter parsing and normalization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class TouchstoneData:
    frequency: np.ndarray
    s_params: np.ndarray
    z0: float


def _unit_scale(token: str) -> float:
    token = token.lower()
    if token == "hz":
        return 1.0
    if token == "khz":
        return 1e3
    if token == "mhz":
        return 1e6
    if token == "ghz":
        return 1e9
    raise ValueError(f"Unsupported frequency unit: {token}")


def _parse_format(token: str) -> str:
    """ Touchstone S2P format line: "# <freq_unit> S <data_format> R <z0>"
        data_format: RI (real/imag), MA (mag/angle), or DB (dB/angle)."""
    token = token.lower()
    if token in {"ri", "ma", "db"}:
        return token
    raise ValueError(f"Unsupported data format: {token}")

def _to_complex(format: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ Convert (a,b) values into complex S-parameters based on the format:
        RI: a=real, b=imag; MA: a=mag, b=angle deg; DB: a=dB, b=angle deg."""
    if format == "ri":
        return a + 1j * b
    if format == "ma":
        return a * np.exp(1j * np.deg2rad(b))
    if format == "db":
        mag = 10 ** (a / 20.0)
        return mag * np.exp(1j * np.deg2rad(b))
    raise ValueError("Unsupported format")


def parse_s2p(path: str) -> TouchstoneData:
    """Parse a Touchstone .s2p file into frequency, S-parameter matrix, and Z0."""
    freq_unit = "hz"
    data_format = "ri"
    z0 = 50.0

    frequencies: List[float] = []
    values: List[float] = []

    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("!"):
                continue
            if line.startswith("#"):
                # Header line defines frequency unit, data format, and reference impedance.
                parts = line[1:].split()
                if len(parts) >= 4:
                    freq_unit = parts[0]
                    data_format = parts[2]
                    if parts[3].lower() == "r" and len(parts) >= 5:
                        z0 = float(parts[4])
                continue
            if "!" in line:
                # Strip inline comments to keep only numeric data.
                line = line.split("!", 1)[0].strip()
            if not line:
                continue
            tokens = line.split()
            if len(tokens) < 9:
                continue
            # First token is frequency, remaining are S11/S21/S12/S22 pairs.
            frequencies.append(float(tokens[0]))
            values.extend(float(tok) for tok in tokens[1:9])

    if not frequencies:
        raise ValueError("No S-parameter data found.")

    scale = _unit_scale(freq_unit)
    format = _parse_format(data_format)

    # Convert to arrays and reshape to [n_freq, 8] pairs.
    freq_arr = np.array(frequencies, dtype=float) * scale
    raw = np.array(values, dtype=float).reshape(len(freq_arr), 8)

    s11 = _to_complex(format, raw[:, 0], raw[:, 1])
    s21 = _to_complex(format, raw[:, 2], raw[:, 3])
    s12 = _to_complex(format, raw[:, 4], raw[:, 5])
    s22 = _to_complex(format, raw[:, 6], raw[:, 7])

    # s_params[f, i, j] stores the 2-port S-matrix at frequency index f.
    # i=0/1 is output port, j=0/1 is input port: [[S11, S12], [S21, S22]].
    s_params = np.zeros((len(freq_arr), 2, 2), dtype=complex)
    s_params[:, 0, 0] = s11
    s_params[:, 0, 1] = s12
    s_params[:, 1, 0] = s21
    s_params[:, 1, 1] = s22

    return TouchstoneData(frequency=freq_arr, s_params=s_params, z0=float(z0))


def s_to_y(s_params: np.ndarray, z0: float) -> np.ndarray:
    """Convert S-parameters to Y-parameters for a 2-port with scalar Z0."""
    identity = np.eye(2, dtype=complex)
    y_params = []
    for s in s_params:
        denom = identity + s
        inv = np.linalg.inv(denom)
        y = (identity - s) @ inv / z0
        y_params.append(y)
    return np.array(y_params)


def interpolate_s_params(data: TouchstoneData, target_freq: np.ndarray) -> np.ndarray:
    """Interpolate S-parameters to target frequencies (linear on real/imag)."""
    s_interp = np.zeros((len(target_freq), 2, 2), dtype=complex)
    for i in range(2):
        for j in range(2):
            real = np.interp(target_freq, data.frequency, data.s_params[:, i, j].real)
            imag = np.interp(target_freq, data.frequency, data.s_params[:, i, j].imag)
            s_interp[:, i, j] = real + 1j * imag
    return s_interp
