"""Topology definitions for trunk segments and inline/shunt node models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class TrunkSegment:
    node_a: int
    node_b: int
    length: float


@dataclass
class RxNode:
    trunk_node: int
    node_index: int


@dataclass
class Topology:
    trunk_segments: List[TrunkSegment]
    rx_nodes: List[RxNode]
    node_probe_nodes: List[int]
    tx_left_node: int
    tx_right_node: int
    tx_node_index: int
    start_node: int
    end_node: int
    node_count: int


def _validate_attach_points(config: dict) -> List[float]:
    attach_points_raw = config.get("attach_points")
    if attach_points_raw is None:
        raise ValueError("'attach_points' is required and must be a list of positions.")
    if not isinstance(attach_points_raw, list):
        raise ValueError("'attach_points' must be a list of positions.")

    attach_points = [float(point) for point in attach_points_raw]
    expected_count = int(config["nodes"])
    if len(attach_points) != expected_count:
        raise ValueError(
            f"'attach_points' must contain exactly {expected_count} values, got {len(attach_points)}."
        )

    length = float(config["length"])
    for idx, point in enumerate(attach_points, start=1):
        if point < 0.0 or point > length:
            raise ValueError(
                f"attach_points[{idx}]={point} is outside the valid range [0, {length}]."
            )

    for idx in range(1, len(attach_points)):
        if attach_points[idx] < attach_points[idx - 1]:
            raise ValueError("'attach_points' must be sorted in non-decreasing order.")

    return attach_points


def build_topology(config: dict) -> Topology:
    attach_points = _validate_attach_points(config)

    tx_node = int(config.get("tx_node", 1))
    if tx_node < 1 or tx_node > config["nodes"]:
        raise ValueError(f"tx_node must be within 1..{config['nodes']}")

    next_node_index = 0
    start_node = next_node_index
    next_node_index += 1
    positions = [0.0] + attach_points
    tx_left_node = start_node
    tx_right_node = start_node

    rx_nodes: List[RxNode] = []
    node_probe_nodes: List[int] = []
    trunk_segments: List[TrunkSegment] = []
    prev_node = start_node

    for n, attach in enumerate(attach_points, start=1):
        segment_length = attach - positions[n - 1]
        if segment_length > 0:
            attach_left_node = next_node_index
            next_node_index += 1
            trunk_segments.append(TrunkSegment(prev_node, attach_left_node, segment_length))
        else:
            attach_left_node = prev_node

        if n == tx_node:
            tx_left_node = attach_left_node
            tx_right_node = next_node_index
            next_node_index += 1
            node_probe_nodes.append(tx_right_node)
            prev_node = tx_right_node
        else:
            rx_nodes.append(RxNode(trunk_node=attach_left_node, node_index=n - 1))
            node_probe_nodes.append(attach_left_node)
            prev_node = attach_left_node

    end_node = next_node_index
    next_node_index += 1
    tail_length = float(config["length"]) - attach_points[-1] if attach_points else float(config["length"])
    if tail_length > 0:
        trunk_segments.append(TrunkSegment(prev_node, end_node, tail_length))
    else:
        end_node = prev_node

    return Topology(
        trunk_segments=trunk_segments,
        rx_nodes=rx_nodes,
        node_probe_nodes=node_probe_nodes,
        tx_left_node=tx_left_node,
        tx_right_node=tx_right_node,
        tx_node_index=tx_node - 1,
        start_node=start_node,
        end_node=end_node,
        node_count=next_node_index,
    )
