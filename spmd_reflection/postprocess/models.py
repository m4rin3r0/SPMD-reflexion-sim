"""Output data structures for postprocessing."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BusResults:
    """Postprocessed results for one TX-RX simulation run.

    Attributes:
        frequency_hz: 1D array of frequency points (Hz).
        rl_db: Return loss at the TX port. Shape (n_freq,).
            RL = -20*log10(|S11_tx|). Positive = more attenuation of reflection.
        rx_to_tx_db: Ratio of RX Voltage to TX voltage at each RX PHY port. Shape (n_freq, n_rx_drops).
            RX/TX = -20*log10(|V_rx_phy| / 0.5). Positive = signal loss.
            Normalized to matched source voltage (0.5 V for 1 V source into Z0).
        il_tci_db: TCI insertion loss per drop. Shape (n_freq, n_rx_drops).
            IL_TCI = -20*log10(|S21_TCI|). Positive = signal loss.
            Computed from drop shunt admittance (valid for parallel-shunt topology).
        rl_tci_db: TCI return loss per drop. Shape (n_freq, n_rx_drops).
            RL_TCI = -20*log10(|Gamma_TC|). Positive = more attenuation of reflection.
            Gamma_TC computed from trunk-side input admittance with PHY load.
        ms_il_db: Mixing segment insertion loss. Shape (n_freq,).
            IL_MS per IEEE 188.8.1 (Eq. 188-3). Measured between edge termination
            reference planes in 100 Ω with all station loads attached.
    """
    frequency_hz: np.ndarray
    rl_db: np.ndarray
    rx_to_tx_db: np.ndarray
    il_tci_db: np.ndarray
    rl_tci_db: np.ndarray
    il_ms_db: np.ndarray | None