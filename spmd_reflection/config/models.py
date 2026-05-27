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
class DropConfig:
    """Drop configuration: PCB measurement file and PHY input impedance.
    
    The touchstone files are expected to describe only the passive PCB.
    The PHY load is connected externally by the solver (Norton source for TX,
    resistor for RX).
    
    Attributes:
        mpse_touchstone: Path to the MPSE drop's .s2p file.
        mpd_touchstone: Path to the MPD drop's .s2p file.
        mpse_index: position of the MPSE drop in the topology
        phy_load_ohm: PHY input impedance to be connected at the PHY-side port
            of RX drops. Default is 20 kΩ, matching typical 10BASE-T1 PHY chips.
    """
    mpse_touchstone: Path
    mpd_touchstone: Path
    mpse_index: int
    phy_load_ohm: float


@dataclass(frozen=True)
class SimConfig:
    """Complete simulation configuration."""
    frequency: FrequencyGrid
    topology: TopologyConfig
    drop: DropConfig
    cable: CableParams