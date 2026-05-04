"""Output data structure for AC solver."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SolverResults:
    """Raw output of the AC solver, ready for postprocessing.

    Attributes:
        frequency_hz: 1D array of frequency points (Hz).
        s11_tx: Reflection coefficient at the TX source port. Shape (n_freq,), complex.
        node_voltages: Voltage at each topology node, per frequency.
            Shape (n_freq, n_nodes), complex. Node indexing matches topology.
            Internal PHY nodes (TX-PHY and RX-PHY) are not included here.
        rx_phy_voltages: Voltage at each RX drop's PHY-side port, per frequency.
            Shape (n_freq, n_rx_drops), complex. The RX drops are ordered as
            they appear in topology.drops (excluding the TX drop).
            These voltages represent the signal level reaching each RX PHY,
            which is the basis for insertion loss calculations.
    """
    frequency_hz: np.ndarray
    s11_tx: np.ndarray
    node_voltages: np.ndarray
    rx_phy_voltages: np.ndarray