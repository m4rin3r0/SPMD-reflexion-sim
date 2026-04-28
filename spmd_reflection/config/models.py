"""Data structures for simulation configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from spmd_reflection.cable.model import CableParams


@dataclass(frozen=True)
class FrequencyGrid:
    """Linear frequency grid for the simulation."""
    start_hz: float
    stop_hz: float
    n_points: int


@dataclass(frozen=True)
class TopologyConfig:
    """User-facing topology configuration.
    
    Validation happens later in build_topology — this dataclass is a plain
    container that mirrors the YAML structure.
    """
    drop_positions_m: tuple[float, ...]
    bus_start_m: float
    bus_end_m: float
    tx_drop_index: int
    termination_ohm: float


@dataclass(frozen=True)
class TouchstonePaths:
    """Absolute paths to Touchstone measurement files."""
    tx: Path
    rx: Path


@dataclass(frozen=True)
class SimConfig:
    """Complete simulation configuration."""
    frequency: FrequencyGrid
    topology: TopologyConfig
    paths: TouchstonePaths
    cable: CableParams