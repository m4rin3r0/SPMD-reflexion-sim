"""AC solver for multi-drop bus simulation.

Builds a nodal admittance matrix from topology, cable, and drop data, then
solves it with a Norton source at the TX-PHY port. Output is the raw S11
and node voltages ready for postprocessing.
"""

from __future__ import annotations
import numpy as np

from spmd_reflection.cable.model import compute_y_params
from spmd_reflection.cable.cable_params import CableParams
from spmd_reflection.drop.models import RxDropData, TxDropData
from spmd_reflection.solver.model import SolverResults
from spmd_reflection.topology.models import Topology


Z0_REFERENCE = 100.0

def _stamp_two_port(y_matrix:np.ndarray, y_2port:np.ndarray, node_a:int, node_b:int) -> None:
    """Stamp a 2-port Y-matrix into the global nodal admittance matrix.
    
    The 2-port has y_2port shape (2, 2) for a single frequency, or (n_freq, 2, 2)
    for the full frequency grid. The connections are between node_a (port 0)
    and node_b (port 1) of the 2-port. Modifies y_matrix in-place.
    
    Args:
        y_matrix: Global Y-matrix, shape (n_freq, n_nodes, n_nodes), complex.
        y_2port: 2-port admittances, shape (n_freq, 2, 2), complex.
        node_a: Index of the global node connected to port 0 of the 2-port.
        node_b: Index of the global node connected to port 1 of the 2-port.
    """
    y_matrix[:, node_a, node_a] += y_2port[:, 0, 0]
    y_matrix[:, node_a, node_b] += y_2port[:, 0, 1]
    y_matrix[:, node_b, node_a] += y_2port[:, 1, 0]
    y_matrix[:, node_b, node_b] += y_2port[:, 1, 1]


def _assemble_y_matrix(topology:Topology, cable_params:CableParams, rx_drop:RxDropData, tx_drop:TxDropData, frequency_hz:np.ndarray) -> tuple[np.ndarray, int]:
    """Assemble the global Y-matrix from topology, cable, and drop data.

    Conceptually:
      - Trunk segments are stamped as 2-ports between their nodes.
      - RX drops are stamped as shunt admittances at their trunk nodes.
      - The TX drop is stamped as a 2-port between its trunk node and a new internal PHY node.
      - Edge terminations are stamped as shunt admittances at their nodes.

    Args:
        topology: The bus topology.
        cable_params: Distributed cable parameters (per meter).
        rx_drop: RX drop data (shunt admittance per frequency).
        tx_drop: TX drop data (2-port Y-matrix per frequency).
        frequency_hz: Simulation frequency grid (Hz).

    Returns:
        A tuple (y_matrix, tx_phy_node) where:
            y_matrix: Complex array of shape (n_freq, n_total, n_total).
                n_total = topology.n_nodes + 1 (for the internal PHY node).
            tx_phy_node: Index of the internal PHY node where the TX 2-port
                connects (always topology.n_nodes, by convention).
    """
    n_freq = len(frequency_hz)
    n_total = topology.n_nodes + 1   # +1 for the internal PHY node
    tx_phy_node = topology.n_nodes   # the new PHY node, by convention

    y_matrix = np.zeros((n_freq, n_total, n_total), dtype=complex)

    # 1. Trunk segments.
    for segment in topology.trunk_segments:
        y_seg = compute_y_params(segment.length_m, cable_params, frequency_hz)
        _stamp_two_port(y_matrix, y_seg, segment.node_a, segment.node_b)

    # 2. RX and TX drops, identified by role.
    for drop in topology.drops:
        if drop.role == "tx":
            _stamp_two_port(y_matrix, tx_drop.y_params, drop.trunk_node, tx_phy_node)
        else:  # role == "rx"
            y_matrix[:, drop.trunk_node, drop.trunk_node] += rx_drop.shunt_admittance

    # 3. Edge terminations.
    for termination in topology.terminations:
        y_termination = 1.0 / termination.impedance_ohm
        y_matrix[:, termination.node, termination.node] += y_termination

    return y_matrix, tx_phy_node


def _compute_s11_and_voltages(y_matrix:np.ndarray, tx_phy_node:int, n_topology_nodes:int) -> tuple[np.ndarray, np.ndarray]:
    """Solve the nodal system with a Norton source at the TX-PHY node.

    For each frequency:
      1. Add the source admittance ysrc = 1/Z₀ to the (tx_phy_node, tx_phy_node) entry of the Y-matrix.
      2. Inject the Norton current ysrc at the TX-PHY node.
      3. Solve Y·v = i for the node voltages.
      4. Extract S₁₁ from the source-port voltage and current.

    Args:
        y_matrix: Global Y-matrix, shape (n_freq, n_total, n_total), complex.
            Will not be modified — copies are made internally.
        tx_phy_node: Index of the TX-PHY node where the Norton source is placed.
        n_topology_nodes: Number of nodes in the original topology
            (= n_total - 1; everything except the internal PHY node).

    Returns:
        A tuple (s11, node_voltages):
            s11: complex array of shape (n_freq,).
            node_voltages: complex array of shape (n_freq, n_topology_nodes).
                The PHY node voltage is excluded.
    """
    n_freq = y_matrix.shape[0]
    n_total = y_matrix.shape[1]
    ysrc = 1.0 / Z0_REFERENCE

    s11 = np.empty(n_freq, dtype=complex)
    all_voltages = np.empty((n_freq, n_total), dtype=complex)

    for freq_idx in range(n_freq):
        # Add the source admittance to the PHY node (Norton source).
        y_total = y_matrix[freq_idx].copy()
        y_total[tx_phy_node, tx_phy_node] += ysrc

        # Norton current vector: only the TX-PHY node is excited.
        i_vec = np.zeros(n_total, dtype=complex)
        i_vec[tx_phy_node] = ysrc

        # Solve the nodal system for this frequency.
        v = np.linalg.solve(y_total, i_vec)
        all_voltages[freq_idx] = v

        # Extract S11 from voltage and current at the source port.
        v_port = v[tx_phy_node]
        i_port = ysrc - ysrc * v_port  # current flowing into the network

        a1 = v_port + i_port * Z0_REFERENCE
        b1 = v_port - i_port * Z0_REFERENCE
        s11[freq_idx] = b1 / a1

    # Drop the PHY node from voltages — only topology nodes are exposed.
    node_voltages = all_voltages[:, :n_topology_nodes]

    return s11, node_voltages


def run_simulation(topology:Topology, cable_params:CableParams, rx_drop:RxDropData, tx_drop:TxDropData, frequency_hz:np.ndarray) -> SolverResults:
    """Run the AC simulation for one TX-RX configuration.

    Args:
        topology: Bus structure (nodes, segments, drops, terminations).
        cable_params: Distributed parameters of the trunk cable (per meter).
        rx_drop: Shunt admittance for all RX drops (one file shared across drops).
        tx_drop: 2-port Y-matrix for the TX drop.
        frequency_hz: Simulation frequency grid in Hz.

    Returns:
        SolverResults with S11 at the TX source port and node voltages
        at all topology nodes (the internal PHY node is not exposed).
    """
    _validate_inputs(topology, rx_drop, tx_drop, frequency_hz)
    y_matrix, tx_phy_node = _assemble_y_matrix(topology, cable_params, rx_drop, tx_drop, frequency_hz)
    s11, node_voltages = _compute_s11_and_voltages(y_matrix, tx_phy_node, topology.n_nodes)
    return SolverResults(frequency_hz=frequency_hz, s11_tx=s11, node_voltages=node_voltages)


def _validate_inputs(topology:Topology, rx_drop:RxDropData, tx_drop:TxDropData, frequency_hz:np.ndarray) -> None:
    """Validate that all inputs are consistent with each other.

    Checks:
      - Frequency arrays of rx_drop and tx_drop match the simulation grid.
      - Topology has exactly one TX drop.
      - Topology has at least one drop overall (already enforced by build_topology,
        but defensive here).

    Raises:
        ValueError: On any inconsistency.
    """
    if not np.array_equal(rx_drop.frequency_hz, frequency_hz):
        raise ValueError("rx_drop.frequency_hz does not match the simulation frequency grid")
    if not np.array_equal(tx_drop.frequency_hz, frequency_hz):
        raise ValueError("tx_drop.frequency_hz does not match the simulation frequency grid")

    tx_count = sum(1 for drop in topology.drops if drop.role == "tx")
    if tx_count != 1:
        raise ValueError(f"topology must have exactly one TX drop, found {tx_count}")