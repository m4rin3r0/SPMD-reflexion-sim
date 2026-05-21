"""Build a bus topology from user-facing configuration.

Takes drop positions, bus extents, TX selector and termination value,
validates them, and constructs the corresponding Topology graph.
"""

from __future__ import annotations
from spmd_reflection.topology.models import Topology, TrunkSegment, DropAttachment, Termination


def _validate_inputs(drop_positions_m:list[float], bus_start_m:float, bus_end_m:float, tx_drop_index:int, termination_ohm:float) -> None:
    """Raise ValueError if any input violates the topology constraints."""

    # Parameter-level checks first (independent of each other).
    if not drop_positions_m:
        raise ValueError("drop_positions_m must contain at least one drop")

    if bus_start_m >= bus_end_m:
        raise ValueError(
            f"bus_start_m ({bus_start_m}) must be less than "
            f"bus_end_m ({bus_end_m})")

    if termination_ohm <= 0:
        raise ValueError(f"termination_ohm must be positive, got {termination_ohm}")

    # Cross-parameter checks (rely on the above being valid).
    n_drops = len(drop_positions_m)
    if not 0 <= tx_drop_index < n_drops:
        raise ValueError(
            f"tx_drop_index {tx_drop_index} is out of range for "
            f"{n_drops} drops")

    for pos in drop_positions_m:
        if not bus_start_m <= pos <= bus_end_m:
            raise ValueError(
                f"drop_positions_m entry {pos} is outside the bus "
                f"[{bus_start_m}, {bus_end_m}]")

    for prev, curr in zip(drop_positions_m, drop_positions_m[1:]):
        if curr <= prev:
            raise ValueError(
                "drop_positions_m must be strictly increasing")
        

def build_topology(drop_positions_m:list[float], bus_start_m:float, bus_end_m:float, tx_drop_index:int, termination_ohm:float) -> Topology:
    """Build a bus Topology from geometric configuration.

    Args:
        drop_positions_m: Drop positions along the bus, strictly increasing.
        bus_start_m: Left bus end (termination position).
        bus_end_m: Right bus end (termination position).
        tx_drop_index: Index into drop_positions_m of the TX drop.
        termination_ohm: Impedance of both edge terminations.

    Returns:
        A Topology describing the bus graph: nodes, segments, drops, terminations.

    Raises:
        ValueError: If any input violates the constraints.
    """
    _validate_inputs(drop_positions_m, bus_start_m, bus_end_m, tx_drop_index, termination_ohm)

    positions = [bus_start_m, *drop_positions_m, bus_end_m] # Logical positions: bus start, drops in order, bus end.

    # Assign a node index to each logical position.
    # Adjacent equal positions share a node (drop merges with termination).
    node_of_position: list[int] = [0]
    next_node = 1
    for i in range(1, len(positions)):
        if positions[i] == positions[i - 1]:
            node_of_position.append(node_of_position[i - 1])
        else:
            node_of_position.append(next_node)
            next_node += 1

    n_nodes = next_node

    # Build trunk segments between consecutive distinct positions.
    segments: list[TrunkSegment] = []
    for i in range(len(positions) - 1):
        node_a = node_of_position[i]
        node_b = node_of_position[i + 1]
        if node_a != node_b:
            length = positions[i + 1] - positions[i]
            segments.append(TrunkSegment(node_a=node_a, node_b=node_b, length_m=length))

    # Build drop attachments. Position index d+1 in `positions` corresponds
    # to drop d in `drop_positions_m`.
    drops: list[DropAttachment] = []
    for d in range(len(drop_positions_m)):
        trunk_node = node_of_position[d + 1]
        role = "tx" if d == tx_drop_index else "rx"
        drops.append(DropAttachment(trunk_node=trunk_node, role=role))

    # Build terminations at the bus ends.
    terminations = (
        Termination(node=node_of_position[0], impedance_ohm=termination_ohm),
        Termination(node=node_of_position[-1], impedance_ohm=termination_ohm))

    return Topology(n_nodes=n_nodes, trunk_segments=tuple(segments), drops=tuple(drops), terminations=terminations)