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


def run(config_path:Path) -> tuple[BusResults,np.ndarray,np.ndarray]:
    """Run the full simulation pipeline from config file to postprocessed results.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        BusResults with RL, RX/TX, IL_TCI, RL_TCI for the configured bus.
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

    tx_drop = load_drop(config.drop.tx_touchstone, frequency_hz)
    rx_drop = load_drop(config.drop.rx_touchstone, frequency_hz)

    solver_results = run_simulation(
        topology=topology,
        cable_params=config.cable,
        tx_drop=tx_drop,
        rx_drop=rx_drop,
        phy_load_ohm=config.drop.phy_load_ohm,
        frequency_hz=frequency_hz)
    
    il_ms_db = run_mixing_segment_simulation(
        topology=topology,
        cable_params=config.cable,
        rx_drop=rx_drop,
        phy_load_ohm=config.drop.phy_load_ohm,
        frequency_hz=frequency_hz)

    return compute_bus_results(results=solver_results, topology=topology, tx_drop=tx_drop, rx_drop=rx_drop, phy_load_ohm=config.drop.phy_load_ohm, il_ms_db=il_ms_db), solver_results.tx_phy_voltages, solver_results.rx_phy_voltages


def run_tx_sweep(config_path:Path) -> list[BusResults]:
    """Run the full simulation for each drop as TX in turn.

    Returns:
        List of BusResults, one per TX position.
        Index i corresponds to drop i being the TX.
    """
    config = load_config(Path(config_path))

    frequency_hz = np.linspace(
        config.frequency.start_hz,
        config.frequency.stop_hz,
        config.frequency.n_points)

    tx_drop = load_drop(config.drop.tx_touchstone, frequency_hz)
    rx_drop = load_drop(config.drop.rx_touchstone, frequency_hz)
    n_drops = len(config.topology.drop_positions_m)
    sweep_results = []

    for tx_index in range(n_drops):
        topology = build_topology(
            drop_positions_m=list(config.topology.drop_positions_m),
            bus_start_m=config.topology.bus_start_m,
            bus_end_m=config.topology.bus_end_m,
            tx_drop_index=tx_index,
            termination_ohm=config.topology.termination_ohm,)
        
        solver_results = run_simulation(
            topology=topology,
            cable_params=config.cable,
            tx_drop=tx_drop,
            rx_drop=rx_drop,
            phy_load_ohm=config.drop.phy_load_ohm,
            frequency_hz=frequency_hz)

        il_ms_db = run_mixing_segment_simulation(
            topology=topology,
            cable_params=config.cable,
            rx_drop=rx_drop,
            phy_load_ohm=config.drop.phy_load_ohm,
            frequency_hz=frequency_hz)

        bus_results = compute_bus_results(
            results=solver_results,
            topology=topology,
            tx_drop=tx_drop,
            rx_drop=rx_drop,
            phy_load_ohm=config.drop.phy_load_ohm,
            il_ms_db=il_ms_db)

        sweep_results.append(bus_results)

    return sweep_results