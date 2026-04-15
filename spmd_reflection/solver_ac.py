"""AC-domain solver for trunk networks with inline TX and shunt RX models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np

from .topology import Topology


@dataclass
class SimulationResults:
    frequency:np.ndarray
    s11_db:np.ndarray
    gain_db:np.ndarray
    s21_db:np.ndarray
    tx_node_index:int


def _yparams_line(length:float, cable:Dict[str, float], freq:float) -> np.ndarray:
    """Return a 2-port Y-parameter matrix for a cable segment at one frequency."""
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


def _stamp_two_port(y_matrix:np.ndarray, y_params:np.ndarray, node_a:int, node_b:int) -> None:
    y_matrix[node_a, node_a] += y_params[0, 0]
    y_matrix[node_a, node_b] += y_params[0, 1]
    y_matrix[node_b, node_a] += y_params[1, 0]
    y_matrix[node_b, node_b] += y_params[1, 1]


def _stamp_shunt_termination(y_network:np.ndarray, node:int, z0:float) -> None:
    y_network[:, node, node] += 1.0 / z0


def run_ac_sim(topology:Topology, cable_model:Dict[str, float], rx_shunt_y:np.ndarray, tx_y:np.ndarray, frequency:np.ndarray, z0:float) -> SimulationResults:
    """Run AC simulation across frequency grid and return RL/IL results."""
    node_count = topology.node_count
    tx_node = topology.tx_node
    tx_node_index = tx_node.tx_node_index
    gmin = 1e-12
    # One Y-matrix per frequency point (complex nodal admittance).
    y_network = np.zeros((len(frequency), node_count, node_count), dtype=complex)

    # Stamp trunk segments into the global Y-matrix.
    for seg in topology.trunk_segments:
        for idx, freq in enumerate(frequency):
            _stamp_two_port(y_network[idx], _yparams_line(seg.length, cable_model, freq), seg.node_a, seg.node_b)

    # Stamp the TX 2-port so Touchstone/Y-matrix port 1 sits on the PHY/source side.
    for idx in range(len(frequency)):
        _stamp_two_port(y_network[idx], tx_y[idx], tx_node.tx_phy_node, tx_node.tx_trunk_node)

    # Stamp RX nodes as shunt one-ports directly at their trunk attachment.
    for rx_node in topology.rx_nodes:
        y_network[:, rx_node.trunk_node, rx_node.trunk_node] += rx_shunt_y

    # Terminate the bus with Z0 at both ends.
    for terminal_node in {topology.get_start_node(), topology.get_end_node()}:
        _stamp_shunt_termination(y_network, terminal_node, z0)

    # Keep the reduced nodal system numerically anchored without re-introducing a physical termination.
    diag = np.arange(node_count)
    y_network[:, diag, diag] += gmin

    # Output arrays for return loss and insertion loss.
    s11_db = np.zeros(len(frequency))
    s21_db = np.zeros((len(frequency), len(topology.node_probe_nodes)))
    gain_db = np.zeros((len(frequency), len(topology.node_probe_nodes)))

    # Norton source at TX with reference impedance Z0.
    ysrc = 1.0 / z0
    for idx, freq in enumerate(frequency):
        # Excite TX port 1 on the PHY side with the Norton source.
        y_total = y_network[idx].copy()
        i_vec = np.zeros(node_count, dtype=complex)
        tx_phy = tx_node.tx_phy_node

        y_total[tx_phy, tx_phy] += ysrc
        i_vec[tx_phy] = ysrc

        # Solve nodal voltages for this frequency.
        v = np.linalg.solve(y_total, i_vec)
        i_tx = i_vec[tx_phy] - ysrc * v[tx_phy]

        # Convert port voltage/current to incident/reflected waves.
        vin = v[tx_phy]
        a1 = vin + i_tx * z0
        b1 = vin - i_tx * z0
        s11 = b1 / a1
        s11_db[idx] = 20 * np.log10(max(np.abs(s11), 1e-30))

        # Evaluate receive voltage relative to the launched TX wave.
        for n, probe_node in enumerate(topology.node_probe_nodes):
            if n == tx_node_index:
                s21_db[idx, n] = np.nan
                gain_db[idx, n] = np.nan
                continue

            vend = v[probe_node]
            s21 = vend / a1 if a1 != 0 else 0.0
            gain = vend / vin if vin != 0 else 0.0
            s21_db[idx, n] = 20 * np.log10(max(np.abs(s21), 1e-30))
            gain_db[idx, n] = 20 * np.log10(max(np.abs(gain), 1e-30))

    return SimulationResults(
        frequency=frequency,
        s11_db=s11_db,
        gain_db=gain_db,
        s21_db=s21_db,
        tx_node_index=tx_node_index,
    )
