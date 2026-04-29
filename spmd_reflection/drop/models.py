"""Data structures for drop measurement data, ready for solver consumption."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RxDropData:
    """Measurement data of an RX drop, reduced to a shunt admittance.

    The RX drop is measured as a 2-port (PHY side, trunk side) with the
    PHY input impedance present. From the trunk-side reflection coefficient
    S₂₂, we derive the shunt admittance the drop presents to the trunk:

        Y_shunt(f) = (1 - S₂₂(f)) / (Z₀ · (1 + S₂₂(f)))

    The PHY-side ports of the original 2-port are not used in the simulation.

    Attributes:
        frequency_hz: 1D array of frequency points (Hz).
        shunt_admittance: 1D complex array of shunt admittance per frequency (S).
    """
    frequency_hz: np.ndarray
    shunt_admittance: np.ndarray


@dataclass(frozen=True)
class TxDropData:
    """Measurement data of the TX drop, kept as a full 2-port Y-matrix.

    Unlike RX drops, the TX drop must be modeled as a true 2-port because
    the simulation injects a Norton source at the PHY-side port. The 2-port
    couples the source to the trunk.

    Attributes:
        frequency_hz: 1D array of frequency points (Hz).
        y_params: Complex array of shape (n_freq, 2, 2). Port indexing:
            0 = PHY side, 1 = trunk side.
    """
    frequency_hz: np.ndarray
    y_params: np.ndarray