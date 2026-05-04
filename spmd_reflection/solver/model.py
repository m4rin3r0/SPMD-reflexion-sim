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
            The internal PHY node (Solver-intern) is not included.
    """
    frequency_hz: np.ndarray
    s11_tx: np.ndarray
    node_voltages: np.ndarray