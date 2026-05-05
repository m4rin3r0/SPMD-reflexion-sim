"""Top-level pipeline: config → solver → postprocessing → BusResults."""

from __future__ import annotations
from pathlib import Path
import numpy as np

from spmd_reflection.cable.cable_params import CableParams
from spmd_reflection.config.load import load_config
from spmd_reflection.drop.load import load_drop
from spmd_reflection.postprocess.extract import compute_bus_results
from spmd_reflection.postprocess.models import BusResults
from spmd_reflection.solver.ac import run_mixing_segment_simulation, run_simulation
from spmd_reflection.topology.build import build_topology


def run(config_path:Path) -> BusResults:
    """Run the full simulation pipeline from config file to postprocessed results.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        BusResults with RL, IL_PHY, IL_TCI, RL_TCI for the configured bus.
    """
    config = load_config(Path(config_path))

    frequency_hz = np.linspace(
        config.frequency.start_hz,
        config.frequency.stop_hz,
        config.frequency.n_points)

    topology = build_topology(
        drop_positions_m=list(config.topology.drop_positions_m),
        bus_start_m=config.topology.bus_start_m,
        bus_end_m=config.topology.bus_end_m,
        tx_drop_index=config.topology.tx_drop_index,
        termination_ohm=config.topology.termination_ohm)

    drop = load_drop(config.drop.touchstone, frequency_hz)

    solver_results = run_simulation(
        topology=topology,
        cable_params=config.cable,
        drop=drop,
        phy_load_ohm=config.drop.phy_load_ohm,
        frequency_hz=frequency_hz)
    
    il_ms_db = run_mixing_segment_simulation(
        topology=topology,
        cable_params=config.cable,
        drop=drop,
        phy_load_ohm=config.drop.phy_load_ohm,
        frequency_hz=frequency_hz)

    return compute_bus_results(
        results=solver_results,
        topology=topology,
        drop=drop,
        phy_load_ohm=config.drop.phy_load_ohm,
        il_ms_db=il_ms_db)