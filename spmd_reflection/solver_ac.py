"""AC-domain solver for building and reducing network matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np

from .topology import Topology


@dataclass
class SimulationResults:
    frequency: np.ndarray
    s11_db: np.ndarray
    gain_db: np.ndarray
    s21_db: np.ndarray


def _yparams_line(length: float, cable: Dict[str, float], freq: float) -> np.ndarray:
    rdc = cable["rdc"]
    rskin = cable["rskin"]
    l = cable["l"]
    c = cable["c"]

    z_series = rdc + rskin * np.sqrt(freq) + 1j * 2 * np.pi * freq * l
    y_shunt = 1j * 2 * np.pi * freq * c

    gamma = np.sqrt(z_series * y_shunt)
    z0 = np.sqrt(z_series / y_shunt)

    gl = gamma * length
    sinh = np.sinh(gl)
    cosh = np.cosh(gl)

    # Avoid division by tiny values at very low frequency.
    if np.abs(sinh) < 1e-30:
        sinh = 1e-30 + 0j

    y11 = cosh / (z0 * sinh)
    y12 = -1.0 / (z0 * sinh)
    return np.array([[y11, y12], [y12, y11]], dtype=complex)


def run_ac_sim(
    topology: Topology,
    cable_model: Dict[str, float],
    y_s2p: np.ndarray,
    frequency: np.ndarray,
    z0: float,
    rterm: float,
) -> SimulationResults:
    node_count = topology.node_count
    y_network = np.zeros((len(frequency), node_count, node_count), dtype=complex)

    # Trunk segments
    for seg in topology.trunk_segments:
        for idx, freq in enumerate(frequency):
            y = _yparams_line(seg.length, cable_model, freq)
            a = seg.node_a
            b = seg.node_b
            y_network[idx, a, a] += y[0, 0]
            y_network[idx, a, b] += y[0, 1]
            y_network[idx, b, a] += y[1, 0]
            y_network[idx, b, b] += y[1, 1]

    # Drop segments
    for drop in topology.drop_segments:
        for idx, freq in enumerate(frequency):
            y = _yparams_line(drop.length, cable_model, freq)
            a = drop.trunk_node
            b = drop.drop_node
            y_network[idx, a, a] += y[0, 0]
            y_network[idx, a, b] += y[0, 1]
            y_network[idx, b, a] += y[1, 0]
            y_network[idx, b, b] += y[1, 1]

    # Node S2P links (drop -> phy)
    for link in topology.node_links:
        for idx in range(len(frequency)):
            y = y_s2p[idx]
            a = link.drop_node
            b = link.phy_node
            y_network[idx, a, a] += y[0, 0]
            y_network[idx, a, b] += y[0, 1]
            y_network[idx, b, a] += y[1, 0]
            y_network[idx, b, b] += y[1, 1]

    # Terminations to ground at start/end trunk nodes
    if rterm > 0:
        y_term = 1.0 / rterm
        y_network[:, topology.start_node, topology.start_node] += y_term
        y_network[:, topology.end_node, topology.end_node] += y_term

    s11_db = np.zeros(len(frequency))
    s21_db = np.zeros((len(frequency), len(topology.phy_nodes)))
    gain_db = np.zeros((len(frequency), len(topology.phy_nodes)))

    ysrc = 1.0 / z0
    for idx, freq in enumerate(frequency):
        y_total = y_network[idx].copy()
        i_vec = np.zeros(node_count, dtype=complex)

        tx = topology.tx_phy_node
        y_total[tx, tx] += ysrc
        i_vec[tx] = ysrc

        v = np.linalg.solve(y_total, i_vec)
        i_network = y_network[idx] @ v
        i_tx = i_vec[tx] - ysrc * v[tx]

        vin = v[tx]
        a1 = vin + i_tx * z0
        b1 = vin - i_tx * z0
        s11 = a1 / b1
        s11_db[idx] = 20 * np.log10(max(np.abs(s11), 1e-30))

        for n, phy in enumerate(topology.phy_nodes):
            vend = v[phy]
            iend = i_network[phy]
            a2 = vend + iend * z0
            b2 = vend - iend * z0
            s21 = b2 / a1
            gain = vend / vin if vin != 0 else 0.0
            s21_db[idx, n] = 20 * np.log10(max(np.abs(s21), 1e-30))
            gain_db[idx, n] = 20 * np.log10(max(np.abs(gain), 1e-30))

    return SimulationResults(
        frequency=frequency,
        s11_db=s11_db,
        gain_db=gain_db,
        s21_db=s21_db,
    )
