"""Extract physical quantities from solver results and drop data."""

from __future__ import annotations
from typing import Optional

import numpy as np

from spmd_reflection.drop.models import DropData
from spmd_reflection.postprocess.models import BusResults
from spmd_reflection.solver.model import SolverResults
from spmd_reflection.topology.models import Topology

# Reference impedance for the simulation (differential).
Z0_REFERENCE = 100.0

# PHY load for TCI calculations (matched source voltage normalization).
_MATCHED_SOURCE_VOLTAGE = 0.5


def _compute_drop_admittance(drop:DropData, phy_load_ohm:float) -> np.ndarray:
    """Compute the trunk-side input admittance of a drop with PHY load present.

    Reduces the 2-port Y-matrix to a 1-port by terminating port 0 (PHY side)
    with the PHY load admittance Y_L = 1/phy_load_ohm.

    The result is the admittance the drop presents to the trunk (TC1/TC2):

        Y_drop = Y[1,1] - Y[1,0] * Y[0,1] / (Y[0,0] + Y_L)

    This is the standard 2-port input admittance reduction formula.

    Args:
        drop: Drop data with 2-port Y-parameters (Port 0 = PHY, Port 1 = Trunk).
        phy_load_ohm: PHY input impedance (Ω).

    Returns:
        Complex array of shape (n_freq,): trunk-side admittance per frequency.
    """
    y_l = 1.0 / phy_load_ohm
    y = drop.y_params
    return y[:, 1, 1] - (y[:, 1, 0] * y[:, 0, 1]) / (y[:, 0, 0] + y_l)


def _compute_rl(s11_tx:np.ndarray) -> np.ndarray:
    """Return loss at TX port. RL = -20*log10(|S11|). Shape (n_freq,)"""
    return -20.0 * np.log10(np.abs(s11_tx))


def _compute_il_phy(rx_phy_voltages:np.ndarray) -> np.ndarray:
    """IL at each RX PHY port, normalized to matched source voltage
    IL_PHY = -20*log10(|V_rx_phy| / 0.5). Shape (n_freq, n_rx_drops)
    """
    return -20.0 * np.log10(np.abs(rx_phy_voltages) / _MATCHED_SOURCE_VOLTAGE)


def _compute_tci_quantities(drop:DropData, phy_load_ohm:float) -> tuple[np.ndarray, np.ndarray]:
    """TCI IL and RL from trunk-side input admittance.

    For a shunt element between two Z0 transmission lines:
        S21_TCI = 2 / (2 + Z0 * Y_drop)          [transmission]
        S11_TCI = -Z0 * Y_drop / (2 + Z0 * Y_drop) [reflection]

    Returns:
        il_tci_db: shape (n_freq,), IL_TCI = -20*log10(|S21_TCI|).
        rl_tci_db: shape (n_freq,), RL_TCI = -20*log10(|S11_TCI|).
    """
    y_drop = _compute_drop_admittance(drop, phy_load_ohm)
    denominator = 2.0 + Z0_REFERENCE * y_drop
    s21_tci = 2.0 / denominator
    s11_tci = -Z0_REFERENCE * y_drop / denominator
    il_tci_db = -20.0 * np.log10(np.abs(s21_tci))
    rl_tci_db = -20.0 * np.log10(np.abs(s11_tci))
    return il_tci_db, rl_tci_db


def compute_bus_results(results:SolverResults, topology:Topology, drop:DropData, phy_load_ohm:float, il_ms_db=Optional[np.ndarray]) -> BusResults:
    """Extract physical quantities from solver results and drop data.

    Computes:
      - RL at TX port (from S11)
      - IL_PHY at each RX PHY port (from rx_phy_voltages)
      - IL_TCI and RL_TCI for all drops (from drop Y-parameters)

    Mixing segment IL (ms_il_db) is not yet implemented and is set to None.

    Args:
        results: Raw solver output.
        topology: Bus topology (used to determine number of drops).
        drop: Drop measurement data (jumped PCB, same for all drops).
        phy_load_ohm: PHY input impedance (Ω).

    Returns:
        BusResults with all computed quantities.
    """
    n_drops = len(topology.drops)
    rl_db = _compute_rl(results.s11_tx)
    il_phy_db = _compute_il_phy(results.rx_phy_voltages)
    # TCI quantities for all drops (TX and RX alike).
    il_tci_db_single, rl_tci_db_single = _compute_tci_quantities(drop, phy_load_ohm)
    # All drops use the same PCB → broadcast to (n_freq, n_drops).
    il_tci_db = np.tile(il_tci_db_single[:, np.newaxis], (1, n_drops))
    rl_tci_db = np.tile(rl_tci_db_single[:, np.newaxis], (1, n_drops))
    return BusResults(
        frequency_hz=results.frequency_hz,
        rl_db=rl_db,
        il_phy_db=il_phy_db,
        il_tci_db=il_tci_db,
        rl_tci_db=rl_tci_db,
        il_ms_db=il_ms_db)