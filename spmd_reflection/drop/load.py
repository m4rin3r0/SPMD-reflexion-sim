from __future__ import annotations
from pathlib import Path
import numpy as np
import skrf

from spmd_reflection.drop.models import RxDropData, TxDropData

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


def load_rx_drop(path:Path, frequency_hz:np.ndarray) -> RxDropData:
    """Load an RX drop measurement and reduce it to a shunt admittance.

    The 2-port measurement of an RX drop has the PHY input impedance present
    at port 1 (PHY side). The reduction to a shunt admittance is valid because
    the drop's insertion loss is constrained by IEEE 802.3da to ≤ 0.16 dB in
    the 1.3-10 MHz band, so the drop acts essentially as a parallel load on
    the trunk.

    Args:
        path: Path to the RX drop's .s2p Touchstone file.
        frequency_hz: 1D array of simulation frequencies (Hz).

    Returns:
        RxDropData with shunt admittance derived from S₂₂.

    Raises:
        ValueError: If the file is invalid or its frequency range is insufficient.
    """
    network = _load_and_validate_touchstone(path, frequency_hz)
    # S₂₂: reflection at the trunk-side port, with port 1 (PHY) terminated
    # in Z₀ during measurement. The PHY input impedance is already part of
    # this reflection coefficient.
    s22 = network.s[:, 1, 1]
    shunt_admittance = (1.0 - s22) / (Z0_REFERENCE * (1.0 + s22))
    return RxDropData(frequency_hz, shunt_admittance)


def load_tx_drop(path: Path, frequency_hz: np.ndarray) -> TxDropData:
    """Load the TX drop measurement and convert to a Y-parameter 2-port.

    Unlike RX drops, the TX drop is kept as a full 2-port because the simulation
    injects a Norton source at port 0 (PHY side). The 2-port couples the source
    to the trunk via port 1 (trunk side).

    Args:
        path: Path to the TX drop's .s2p Touchstone file.
        frequency_hz: 1D array of simulation frequencies (Hz).

    Returns:
        TxDropData with Y-parameters of shape (n_freq, 2, 2).
        Port indexing: 0 = PHY side, 1 = trunk side.

    Raises:
        ValueError: If the file is invalid or its frequency range is insufficient.
    """
    network = _load_and_validate_touchstone(path, frequency_hz)
    return TxDropData(frequency_hz,y_params=network.y)