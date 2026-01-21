"""Topology definitions for trunk, drops, nodes, and terminations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import math
import random


@dataclass
class TrunkSegment:
    node_a: int
    node_b: int
    length: float


@dataclass
class DropSegment:
    trunk_node: int
    drop_node: int
    length: float


@dataclass
class NodeLink:
    drop_node: int
    phy_node: int


@dataclass
class Topology:
    trunk_segments: List[TrunkSegment]
    drop_segments: List[DropSegment]
    node_links: List[NodeLink]
    phy_nodes: List[int]
    tx_phy_node: int
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
        random_attach: bool,
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
        self.random_attach = bool(random_attach)
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
        elif self.random_attach:
            mid_pts = self._distribute_random(attach_start, attach_end, unattached)
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

    def _distribute_random(self, start: float, end: float, count: int) -> List[float]:
        available = (end - start) + (2 * self.separation_min)
        if available <= 0:
            return []
        segments = [available]
        for _ in range(count):
            idx = random.randrange(len(segments))
            length = segments.pop(idx)
            if length <= self.separation_min * 2:
                segments.insert(idx, length)
                break
            div = random.uniform(self.separation_min, length - self.separation_min)
            segments.insert(idx, div)
            segments.insert(idx + 1, length - div)
        positions = []
        pos = start
        for seg in segments[:-1]:
            pos += seg
            positions.append(pos)
        return positions

    def _start_attach(self, start: float, count: int) -> List[float]:
        return [start + i * self.separation_min for i in range(count)]

    def _end_attach(self, end: float, count: int) -> List[float]:
        return [end - ((count - i - 1) * self.separation_min) for i in range(count)]


def build_topology(config: dict) -> Topology:
    seed = int(config.get("seed", -1))
    if seed != -1:
        random.seed(seed)

    trunk = Trunk(
        length=config["length"],
        nodes=config["nodes"],
        separation_min=config["separation_min"],
        start_pad=config["start_pad"],
        end_pad=config["end_pad"],
        start_attach=config["start_attach"],
        end_attach=config["end_attach"],
        random_attach=config["random_attach"],
        attach_error=config.get("attach_error", 0.0),
        attach_points=config.get("attach_points"),
    )

    attach_points = trunk.attach_points
    if len(attach_points) != config["nodes"]:
        raise ValueError("attach_points count must match nodes")

    positions = [0.0] + attach_points + [float(config["length"])]
    trunk_nodes = list(range(len(positions)))

    trunk_segments: List[TrunkSegment] = []
    for idx in range(len(positions) - 1):
        length = positions[idx + 1] - positions[idx]
        if length <= 0:
            continue
        trunk_segments.append(TrunkSegment(trunk_nodes[idx], trunk_nodes[idx + 1], length))

    drop_segments: List[DropSegment] = []
    node_links: List[NodeLink] = []
    phy_nodes: List[int] = []

    next_node_index = len(positions)
    for n in range(config["nodes"]):
        trunk_node = trunk_nodes[n + 1]
        drop_node = next_node_index
        next_node_index += 1
        phy_node = next_node_index
        next_node_index += 1

        drop_length = float(config["drop_max"])
        if config.get("random_drop"):
            drop_length = random.uniform(0.0, drop_length)

        drop_segments.append(DropSegment(trunk_node=trunk_node, drop_node=drop_node, length=drop_length))
        node_links.append(NodeLink(drop_node=drop_node, phy_node=phy_node))
        phy_nodes.append(phy_node)

    tx_node = int(config.get("tx_node", 1))
    if tx_node < 1 or tx_node > config["nodes"]:
        raise ValueError("tx_node must be within 1..nodes")

    return Topology(
        trunk_segments=trunk_segments,
        drop_segments=drop_segments,
        node_links=node_links,
        phy_nodes=phy_nodes,
        tx_phy_node=phy_nodes[tx_node - 1],
        start_node=trunk_nodes[0],
        end_node=trunk_nodes[-1],
        node_count=next_node_index,
    )
