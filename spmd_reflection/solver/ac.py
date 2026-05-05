"""AC solver for multi-drop bus simulation.

Builds a nodal admittance matrix from topology, cable, and drop data, then
solves it with a Norton source at the TX-PHY port. Output is the raw S11
and node voltages ready for postprocessing.
"""

from __future__ import annotations
import numpy as np

from spmd_reflection.cable.model import compute_y_params
from spmd_reflection.cable.cable_params import CableParams
from spmd_reflection.drop.models import DropData
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


def _assemble_y_matrix_no_terminations(topology:Topology, cable_params:CableParams, drop:DropData, phy_load_ohm:float, frequency_hz:np.ndarray) -> tuple[np.ndarray,int,list[int]]:
    """Assemble the global Y-matrix from topology, cable, and drop data without terminations

    Conceptually:
      - Trunk segments are stamped as 2-ports between their nodes.
      - Each drop is stamped as a 2-port between its trunk node and a new
        internal PHY node.
      - For RX drops: the PHY load (e.g. 20 kΩ) is stamped as a shunt at the
        RX-PHY node.
      - For the TX drop: no shunt is added; the Norton source admittance will
        be added later in _compute_s11_and_voltages.

    Node layout in the Y-matrix:
      [0 .. n_topology_nodes-1]    : topology nodes
      [n_topology_nodes .. -2]     : RX-PHY nodes (one per RX drop)
      [-1]                         : TX-PHY node

    Args:
        topology: The bus topology.
        cable_params: Distributed cable parameters (per meter).
        drop: Drop measurement data (used for both TX and all RX drops).
        phy_load_ohm: PHY input impedance for RX drops (Ω).
        frequency_hz: Simulation frequency grid (Hz).

    Returns:
        A tuple (y_matrix, tx_phy_node, rx_phy_nodes) where:
            y_matrix: Complex array of shape (n_freq, n_total, n_total).
            tx_phy_node: Index of the TX-PHY node (last index).
            rx_phy_nodes: Indices of the RX-PHY nodes, in the order of the
                RX drops as they appear in topology.drops.
    """
    n_freq = len(frequency_hz)
    n_topology_nodes = topology.n_nodes
    # Identify RX drops and assign each one a fresh PHY node.
    rx_drop_indices = [i for i, d in enumerate(topology.drops) if d.role == "rx"]
    n_rx_drops = len(rx_drop_indices)
    rx_phy_nodes = list(range(n_topology_nodes, n_topology_nodes + n_rx_drops))
    tx_phy_node = n_topology_nodes + n_rx_drops
    n_total = tx_phy_node + 1
    y_matrix = np.zeros((n_freq, n_total, n_total), dtype=complex)
    # 1. Trunk segments.
    for segment in topology.trunk_segments:
        y_seg = compute_y_params(segment.length_m, cable_params, frequency_hz)
        _stamp_two_port(y_matrix, y_seg, segment.node_a, segment.node_b)
    # 2. Drops as 2-ports between trunk node and a PHY node.
    rx_iter = iter(rx_phy_nodes)
    for d in topology.drops:
        if d.role == "tx":
            phy_node = tx_phy_node
        else:
            phy_node = next(rx_iter)
            # Add the PHY load as a shunt at the RX-PHY node.
            y_matrix[:, phy_node, phy_node] += 1.0 / phy_load_ohm
        _stamp_two_port(y_matrix, drop.y_params, d.trunk_node, phy_node)
    return y_matrix, tx_phy_node, rx_phy_nodes


def _assemble_y_matrix(topology:Topology, cable_params:CableParams, drop:DropData, phy_load_ohm:float, frequency_hz:np.ndarray) -> tuple[np.ndarray, int, list[int]]:
    """Assemble the global Y-matrix including edge terminations
    
    Conceptually:
      - Everything from _assemble_y_matrix_no_terminations
      - Edge terminations are stamped as shunt admittances at their nodes.

    Args:
        topology: The bus topology.
        cable_params: Distributed cable parameters (per meter).
        drop: Drop measurement data (used for both TX and all RX drops).
        phy_load_ohm: PHY input impedance for RX drops (Ω).
        frequency_hz: Simulation frequency grid (Hz).

    Returns:
        A tuple (y_matrix, tx_phy_node, rx_phy_nodes) where:
            y_matrix: Complex array of shape (n_freq, n_total, n_total).
            tx_phy_node: Index of the TX-PHY node (last index).
            rx_phy_nodes: Indices of the RX-PHY nodes, in the order of the
                RX drops as they appear in topology.drops.
    """
    y_matrix, tx_phy_node, rx_phy_nodes = _assemble_y_matrix_no_terminations(topology, cable_params, drop, phy_load_ohm, frequency_hz)
    # Add edge terminations.
    for termination in topology.terminations:
        y_termination = 1.0 / termination.impedance_ohm
        y_matrix[:, termination.node, termination.node] += y_termination
    return y_matrix, tx_phy_node, rx_phy_nodes


def _compute_s11_and_voltages(y_matrix:np.ndarray, tx_phy_node:int, rx_phy_nodes:list[int], n_topology_nodes:int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the nodal system with a Norton source at the TX-PHY node.

    For each frequency:
    1. Add the source admittance ysrc = 1/Z₀ to the TX-PHY diagonal entry.
    2. Inject the Norton current at the TX-PHY node.
    3. Solve Y·v = i for all node voltages.
    4. Extract S₁₁ from the source-port voltage.
    5. Extract RX-PHY voltages from the solved voltage vector.

    Args:
        y_matrix: Global Y-matrix, shape (n_freq, n_total, n_total).
        tx_phy_node: Index of the TX-PHY node.
        rx_phy_nodes: Indices of the RX-PHY nodes, in RX-drop order.
        n_topology_nodes: Number of topology nodes (for slicing node_voltages).

    Returns:
        A tuple (s11, node_voltages, rx_phy_voltages):
            s11: shape (n_freq,), complex.
            node_voltages: shape (n_freq, n_topology_nodes), complex.
            rx_phy_voltages: shape (n_freq, n_rx_drops), complex.
    """
    n_freq = y_matrix.shape[0]
    n_total = y_matrix.shape[1]
    n_rx = len(rx_phy_nodes)
    ysrc = 1.0 / Z0_REFERENCE
    s11 = np.empty(n_freq, dtype=complex)
    all_voltages = np.empty((n_freq, n_total), dtype=complex)

    for freq_idx in range(n_freq):
        # Add source admittance to TX-PHY diagonal (Norton source).
        y_total = y_matrix[freq_idx].copy()
        y_total[tx_phy_node, tx_phy_node] += ysrc
        # Norton current vector: only TX-PHY node is excited.
        i_vec = np.zeros(n_total, dtype=complex)
        i_vec[tx_phy_node] = ysrc
        # Solve the nodal system.
        v = np.linalg.solve(y_total, i_vec)
        all_voltages[freq_idx] = v
        # Extract S₁₁ from source-port voltage and current.
        v_port = v[tx_phy_node]
        i_port = ysrc - ysrc * v_port
        a1 = v_port + i_port * Z0_REFERENCE
        b1 = v_port - i_port * Z0_REFERENCE
        s11[freq_idx] = b1 / a1

    # Slice out topology nodes (exclude all internal PHY nodes).
    node_voltages = all_voltages[:, :n_topology_nodes]
    # Collect RX-PHY voltages in drop order.
    rx_phy_voltages = np.empty((n_freq, n_rx), dtype=complex)
    for rx_idx, phy_node in enumerate(rx_phy_nodes):
        rx_phy_voltages[:, rx_idx] = all_voltages[:, phy_node]
    return s11, node_voltages, rx_phy_voltages


def _validate_inputs(topology:Topology, drop:DropData, frequency_hz:np.ndarray) -> None:
    """Validate that all inputs are consistent with each other.

    Checks:
      - Drop frequency array matches the simulation grid.
      - Topology has exactly one TX drop.

    Raises:
        ValueError: On any inconsistency.
    """
    if not np.array_equal(drop.frequency_hz, frequency_hz):
        raise ValueError("drop.frequency_hz does not match the simulation frequency grid")
    tx_count = sum(1 for d in topology.drops if d.role == "tx")
    if tx_count != 1:
        raise ValueError(f"topology must have exactly one TX drop, found {tx_count}")


def run_simulation(topology:Topology, cable_params:CableParams, drop:DropData, phy_load_ohm:float, frequency_hz:np.ndarray) -> SolverResults:
    """Run the AC simulation for one TX-RX configuration.

    Args:
        topology: Bus structure (nodes, segments, drops, terminations).
        cable_params: Distributed parameters of the trunk cable (per meter).
        drop: Drop measurement data (jumped PCB, used for all drops).
        phy_load_ohm: PHY input impedance for RX drops (Ω), e.g. 20000.
        frequency_hz: Simulation frequency grid in Hz.

    Returns:
        SolverResults with S11 at the TX source port, node voltages at all
        topology nodes, and RX-PHY voltages at each RX drop's PHY port.
    """
    _validate_inputs(topology, drop, frequency_hz)
    y_matrix, tx_phy_node, rx_phy_nodes = _assemble_y_matrix(topology, cable_params, drop, phy_load_ohm, frequency_hz)
    s11, node_voltages, rx_phy_voltages = _compute_s11_and_voltages(y_matrix, tx_phy_node, rx_phy_nodes, topology.n_nodes)
    return SolverResults(
        frequency_hz=frequency_hz,
        s11_tx=s11,
        node_voltages=node_voltages,
        rx_phy_voltages=rx_phy_voltages)


def run_mixing_segment_simulation(topology:Topology, cable_params:CableParams, drop:DropData, phy_load_ohm:float, frequency_hz:np.ndarray) -> np.ndarray:
    """Compute mixing segment IL per IEEE 802.3da 188.8.1.

    Places a Norton source at the left bus-end node and measure the voltage at the right bus-end node. The edge terminations are replaced
    by the source and load impedances (100 Ω each), as specified by the standard ('substituting the measurement probes for the edge terminators').

    Args:
        topology: Bus topology.
        cable_params: Distributed cable parameters.
        drop: Drop measurement data.
        phy_load_ohm: PHY input impedance for RX drops (Ω).
        frequency_hz: Simulation frequency grid (Hz).

    Returns:
        ms_il_db: shape (n_freq,), mixing segment IL in dB. Positive = loss.
    """
    n_freq = len(frequency_hz)
    # Identify bus-end nodes from terminations.
    left_node  = min(t.node for t in topology.terminations)
    right_node = max(t.node for t in topology.terminations)
    # Build Y-matrix WITHOUT edge terminations (they are replaced by probes).
    y_matrix, _, rx_phy_nodes = _assemble_y_matrix_no_terminations(topology, cable_params, drop, phy_load_ohm, frequency_hz)
    ysrc  = 1.0 / Z0_REFERENCE
    yload = 1.0 / Z0_REFERENCE
    ms_il_db = np.empty(n_freq)
    for freq_idx in range(n_freq):
        y_total = y_matrix[freq_idx].copy()
        # Source impedance at left node.
        y_total[left_node, left_node] += ysrc
        # Load impedance at right node.
        y_total[right_node, right_node] += yload
        # Norton current at left node.
        i_vec = np.zeros(y_total.shape[0], dtype=complex)
        i_vec[left_node] = ysrc
        v = np.linalg.solve(y_total, i_vec)
        # S21 between left and right node.
        v_source = v[left_node]
        v_load   = v[right_node]
        i_source = ysrc - ysrc * v_source
        a1 = v_source + i_source * Z0_REFERENCE
        s21 = 2.0 * v_load / a1   # S21 = 2*V_load / a1 for matched load
        ms_il_db[freq_idx] = -20.0 * np.log10(np.abs(s21))
    return ms_il_db