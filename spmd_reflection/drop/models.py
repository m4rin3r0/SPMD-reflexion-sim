"""Data structure for drop measurement data, used for both TX and RX drops."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DropData:
    """Measurement data of a drop PCB, kept as a 2-port Y-matrix.
    
    The drop is measured with the PHY input impedance bypassed (jumped) so that
    its 2-port S-parameters describe only the passive PCB (CMC, ESD, connectors).
    The same data is used for both TX and RX drops:
      - TX drops: connect a Norton source at the PHY-side port.
      - RX drops: connect the PHY input impedance (20 kΩ) at the PHY-side port.
    
    Attributes:
        frequency_hz: 1D array of frequency points (Hz).
        y_params: Complex array of shape (n_freq, 2, 2). Port indexing:
            0 = PHY side, 1 = trunk side.
    """
    frequency_hz: np.ndarray
    y_params: np.ndarray