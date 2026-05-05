from __future__ import annotations
from pathlib import Path
import numpy as np
import skrf

from spmd_reflection.drop.models import DropData

# Reference impedance for the simulation (differential).
Z0_REFERENCE = 100.0


def _load_and_validate_touchstone(path:Path, frequency_hz:np.ndarray) -> skrf.Network:
    """Load a Touchstone file and prepare it for the simulation grid.

    Performs:
    1. Loading via skrf.
    2. Validation that it's a 2-port network.
    3. Validation that the file's frequency range covers the simulation grid.
    4. Renormalization to Z0_REFERENCE if the file uses a different impedance.
    5. Interpolation onto the simulation frequency grid.

    Args:
        path: Path to a .s2p Touchstone file.
        frequency_hz: 1D array of simulation frequencies (Hz).

    Returns:
        A skrf.Network on the simulation frequency grid, renormalized to Z0_REFERENCE.

    Raises:
        ValueError: On any structural or range mismatch.
    """
    network = skrf.Network(str(path))
    if network.nports != 2:
        raise ValueError(f"expected a 2-port Touchstone file, but {path.name} has {network.nports} ports")
    file_freq_min = network.f.min()
    file_freq_max = network.f.max()
    file_freq_min = network.f.min()
    file_freq_max = network.f.max()
    if frequency_hz.min() < file_freq_min or frequency_hz.max() > file_freq_max:
        raise ValueError(
            f"simulation range [{frequency_hz.min():.3e}, {frequency_hz.max():.3e}] Hz "
            f"is not covered by Touchstone file {path.name} "
            f"(file range: [{file_freq_min:.3e}, {file_freq_max:.3e}] Hz)")
    if not np.isclose(network.z0[0, 0], Z0_REFERENCE):
        network = network.copy()
        network.renormalize(Z0_REFERENCE)
    target_frequency = skrf.Frequency.from_f(frequency_hz, unit="Hz")
    return network.interpolate(target_frequency)


def load_drop(path: Path, frequency_hz: np.ndarray) -> DropData:
    """Load a jumped-PCB drop measurement as a 2-port Y-matrix.

    The same drop data is used for both TX and RX drops in the solver:
      - For a TX drop, a Norton source is connected at the PHY-side port.
      - For an RX drop, the PHY input impedance (e.g., 20 kΩ) is connected
        at the PHY-side port.

    Args:
        path: Path to the drop's .s2p Touchstone file (jumped measurement).
        frequency_hz: 1D array of simulation frequencies (Hz).

    Returns:
        DropData with Y-parameters of shape (n_freq, 2, 2).
        Port indexing: 0 = PHY side, 1 = trunk side.

    Raises:
        ValueError: If the file is invalid or its frequency range is insufficient.
    """
    network = _load_and_validate_touchstone(path, frequency_hz)
    return DropData(frequency_hz=frequency_hz, y_params=network.y)
