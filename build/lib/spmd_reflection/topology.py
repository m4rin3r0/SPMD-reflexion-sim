from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from spmd_reflection.topology_validator import _validate_attach_points, _validate_tx_node


@dataclass
class TrunkSegment:
    node_a:int
    node_b:int
    length:float


@dataclass
class RxNode:
    trunk_node:int
    node_index:int
    left_segment_index:Optional[int] = None
    right_segment_index:Optional[int] = None


@dataclass
class TxNode:
    tx_trunk_node:int
    tx_phy_node:int
    tx_node_index:int


class Topology:
    trunk_segments:List[TrunkSegment]
    rx_nodes:List[RxNode]
    node_probe_nodes:List[int]
    tx_node:TxNode
    node_count:int

    def get_start_node(self) -> int:
        return 0

    def get_end_node(self) -> int:
        if self.trunk_segments:
            return self.trunk_segments[-1].node_b
        return self.tx_node.tx_trunk_node


def build_topology(config:dict) -> Topology:
    attach_points = _validate_attach_points(config)
    tx_node = _validate_tx_node(config)

    next_node_index = 1
    positions = [0.0] + attach_points

    rx_nodes:List[RxNode] = []
    node_probe_nodes:List[int] = []
    trunk_segments:List[TrunkSegment] = []
    prev_node = 0

    for n, attach in enumerate(attach_points, start=1):
        segment_length = attach - positions[n - 1]
        if segment_length > 0:
            attach_left_node = next_node_index
            next_node_index += 1
            trunk_segments.append(TrunkSegment(prev_node, attach_left_node, segment_length))
        else:
            attach_left_node = prev_node

        if n == tx_node:
            tx_trunk_node = attach_left_node
            tx_phy_node = next_node_index
            next_node_index += 1
            node_probe_nodes.append(tx_phy_node)
            prev_node = tx_phy_node
        else:
            rx_nodes.append(RxNode(trunk_node=attach_left_node, node_index=n - 1))
            node_probe_nodes.append(attach_left_node)
            prev_node = attach_left_node

    end_node = next_node_index
    next_node_index += 1
    tail_length = float(config["length"]) - attach_points[-1] if attach_points else float(config["length"])
    if tail_length > 0:
        trunk_segments.append(TrunkSegment(prev_node, end_node, tail_length))
    
    tx_node_index = tx_node -1

    left_segments_by_node = {seg.node_b: idx for idx, seg in enumerate(trunk_segments)}
    right_segments_by_node = {seg.node_a: idx for idx, seg in enumerate(trunk_segments)}
    for rx_node in rx_nodes:
        rx_node.left_segment_index = left_segments_by_node.get(rx_node.trunk_node)
        rx_node.right_segment_index = right_segments_by_node.get(rx_node.trunk_node)

    topology = Topology()
    topology.trunk_segments = trunk_segments
    topology.rx_nodes = rx_nodes
    topology.node_probe_nodes = node_probe_nodes
    topology.tx_node = TxNode(tx_trunk_node,tx_phy_node,tx_node_index)
    topology.node_count = next_node_index

    return topology
