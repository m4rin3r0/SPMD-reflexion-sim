"""Topology definitions for trunk segments and inline/shunt node models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import random


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


class Trunk:
    def __init__(
        self,
        length: float,
        nodes: int,
        separation_min: float,
        start_pad: float,
        end_pad: float,
        start_attach: int,
        end_attach: int,
        attach_error: float,
        attach_points: Optional[List[float]],
    ) -> None:
        self.length = float(length)
        self.nodes = int(nodes)
        self.separation_min = float(separation_min)
        self.start_pad = float(start_pad)
        self.end_pad = float(end_pad)
        self.start_attach = int(start_attach)
        self.end_attach = int(end_attach)
        self.attach_error = float(attach_error)
        self.attach_points = self._build_attach_points(attach_points)

    def _build_attach_points(self, attach_points: Optional[List[float]]) -> List[float]:
        if attach_points is not None:
            return sorted(float(x) for x in attach_points)

        unattached = self.nodes
        attach_start = self.start_pad
        attach_end = self.length - self.end_pad
        points: List[float] = []

        if self.end_attach > 0:
            end_pts = self._end_attach(attach_end, self.end_attach)
            points.extend(end_pts)
            unattached -= self.end_attach
            attach_end = end_pts[0] - self.separation_min

        if self.start_attach > 0:
            start_pts = self._start_attach(attach_start, self.start_attach)
            points.extend(start_pts)
            unattached -= self.start_attach
            attach_start = start_pts[-1] + self.separation_min

        mid_pts: List[float]
        if unattached <= 0:
            mid_pts = []
        else:
            mid_pts = self._distribute_even(attach_start, attach_end, unattached)

        points.extend(mid_pts)
        return sorted(points)

    def _distribute_even(self, start: float, end: float, count: int) -> List[float]:
        if count <= 0:
            return []
        if count == 1:
            return [(start + end) / 2]
        delta = (end - start) / (count - 1)
        points = []
        for i in range(count):
            pt = start + i * delta + random.gauss(0, self.attach_error)
            pt = min(max(pt, 0.0), self.length)
            points.append(pt)
        return points

    def _start_attach(self, start: float, count: int) -> List[float]:
        return [start + i * self.separation_min for i in range(count)]

    def _end_attach(self, end: float, count: int) -> List[float]:
        return [end - ((count - i - 1) * self.separation_min) for i in range(count)]


def build_topology(config: dict) -> Topology:
    trunk = Trunk(
        length=config["length"],
        nodes=config["nodes"],
        separation_min=config["separation_min"],
        start_pad=config["start_pad"],
        end_pad=config["end_pad"],
        start_attach=config["start_attach"],
        end_attach=config["end_attach"],
        attach_error=config.get("attach_error", 0.0),
        attach_points=config.get("attach_points"),
    )

    attach_points = trunk.attach_points
    if len(attach_points) != config["nodes"]:
        raise ValueError("attach_points count must match nodes")

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
