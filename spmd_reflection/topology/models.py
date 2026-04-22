"""Data structures for bus topology.

These dataclasses describe the abstract structure of a multi-drop bus:
which nodes exist, how they are connected by cable segments, where drops
attach, and where the edge terminations sit. The classes contain no
electrical parameters — those are added later by the solver.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TrunkSegment:
    """A cable segment connecting two trunk nodes."""
    node_a: int
    node_b: int
    length_m: float


@dataclass(frozen=True)
class DropAttachment:
    """A drop attached to a trunk node, with its role (TX source or RX sink)."""
    trunk_node: int
    role: Literal["tx", "rx"]


@dataclass(frozen=True)
class Termination:
    """An edge termination at a trunk node."""
    node: int
    impedance_ohm: float


@dataclass(frozen=True)
class Topology:
    """
    Attributes:
        n_nodes: Total number of distinct nodes in the bus graph.
        trunk_segments: Cable segments connecting trunk nodes, in geometric order
            from left to right.
        drops: Drops attached to trunk nodes, in the order given by the user's
            drop_positions_m list. `tx_drop_index` from the config refers to
            this ordering.
        terminations: Edge terminations, first at bus start, second at bus end.
    """
    n_nodes: int
    trunk_segments: tuple[TrunkSegment, ...]
    drops: tuple[DropAttachment, ...]
    terminations: tuple[Termination, ...]