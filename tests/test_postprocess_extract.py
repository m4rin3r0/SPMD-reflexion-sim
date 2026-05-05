"""Tests for postprocess.extract."""

import numpy as np
import pytest
from pathlib import Path

from spmd_reflection.drop.models import DropData
from spmd_reflection.postprocess.extract import compute_bus_results, Z0_REFERENCE
from spmd_reflection.postprocess.models import BusResults
from spmd_reflection.solver.model import SolverResults
from spmd_reflection.topology.build import build_topology
from spmd_reflection.cable.cable_params import CableParams
from spmd_reflection.drop.load import load_drop
from spmd_reflection.solver.ac import run_simulation


def _default_frequency_grid() -> np.ndarray:
    return np.linspace(1e6, 30e6, 50)


def _make_solver_results(frequency_hz:np.ndarray, s11_tx:np.ndarray, rx_phy_voltages:np.ndarray) -> SolverResults:
    """Construct minimal SolverResults for testing."""
    n_freq = len(frequency_hz)
    n_nodes = 2
    return SolverResults(
        frequency_hz=frequency_hz,
        s11_tx=s11_tx,
        node_voltages=np.zeros((n_freq, n_nodes), dtype=complex),
        rx_phy_voltages=rx_phy_voltages)


def _make_drop_with_admittance(frequency_hz:np.ndarray, y_trunk:complex, phy_load_ohm:float=20000.0) -> DropData:
    """Construct a DropData whose trunk-side admittance equals y_trunk.
    
    We want _compute_drop_admittance to return y_trunk. With Y[0,1]=Y[1,0]=0
    (no coupling between PHY and Trunk), the reduction simplifies to:
    
        Y_drop = Y[1,1]
    
    So we just set Y[1,1] = y_trunk directly.
    """
    n_freq = len(frequency_hz)
    y_params = np.zeros((n_freq, 2, 2), dtype=complex)
    y_params[:, 1, 1] = y_trunk
    return DropData(frequency_hz=frequency_hz, y_params=y_params)


def test_rl_is_zero_db_for_zero_reflection():
    """S11 = 0 (perfect fit) -> RL = inf dB – we check a finite case."""
    freqs = _default_frequency_grid()
    n_freq = len(freqs)
    # S11 = 0 -> log10(0) = -inf
    # instead: S11 = 0.001 (very little reflection) -> RL ≈ 60 dB.
    s11 = np.full(n_freq, 0.001, dtype=complex)
    results = _make_solver_results(freqs, s11, np.zeros((n_freq, 0), dtype=complex))
    topology = build_topology(
        drop_positions_m=[3.0],
        bus_start_m=0.0,
        bus_end_m=6.0,
        tx_drop_index=0,
        termination_ohm=100.0)
    drop = _make_drop_with_admittance(freqs, y_trunk=0.0)
    bus = compute_bus_results(results, topology, drop, phy_load_ohm=20000.0)
    expected_rl = -20.0 * np.log10(0.001)  # = 60 dB
    assert np.allclose(bus.rl_db, expected_rl, atol=1e-10)


def test_rl_is_6db_for_half_reflection():
    """S11 = 0.5 -> RL = 6.02 dB."""
    freqs = _default_frequency_grid()
    n_freq = len(freqs)
    s11 = np.full(n_freq, 0.5, dtype=complex)
    results = _make_solver_results(freqs, s11, np.zeros((n_freq, 0), dtype=complex))
    topology = build_topology(
        drop_positions_m=[3.0],
        bus_start_m=0.0,
        bus_end_m=6.0,
        tx_drop_index=0,
        termination_ohm=100.0)
    drop = _make_drop_with_admittance(freqs, y_trunk=0.0)
    bus = compute_bus_results(results, topology, drop, phy_load_ohm=20000.0)
    expected_rl = -20.0 * np.log10(0.5)  # = 6.0206 dB
    assert np.allclose(bus.rl_db, expected_rl, atol=1e-10)


def test_il_phy_is_zero_db_at_matched_source_voltage():
    """V_rx_phy = 0.5 V (matched) -> IL_PHY = 0 dB"""
    freqs = _default_frequency_grid()
    n_freq = len(freqs)
    # one RX-Drop, voltage exactly 0.5 V (= matched source voltage).
    rx_phy_voltages = np.full((n_freq, 1), 0.5, dtype=complex)
    results = _make_solver_results(freqs, np.zeros(n_freq, dtype=complex), rx_phy_voltages)
    topology = build_topology(
        drop_positions_m=[1.0, 3.0],
        bus_start_m=0.0,
        bus_end_m=6.0,
        tx_drop_index=0,
        termination_ohm=100.0)
    drop = _make_drop_with_admittance(freqs, y_trunk=0.0)
    bus = compute_bus_results(results, topology, drop, phy_load_ohm=20000.0)
    # IL_PHY = -20*log10(0.5 / 0.5) = 0 dB.
    assert bus.il_phy_db.shape == (n_freq, 1)
    assert np.allclose(bus.il_phy_db, 0.0, atol=1e-10)


def test_il_phy_is_positive_for_attenuated_signal():
    """V_rx_phy < 0.5 V -> IL_PHY > 0 dB"""
    freqs = _default_frequency_grid()
    n_freq = len(freqs)
    # 2 RX-Drops with different dampings
    v_near = 0.4   # less damping
    v_far  = 0.2   # more damping
    rx_phy_voltages = np.zeros((n_freq, 2), dtype=complex)
    rx_phy_voltages[:, 0] = v_near
    rx_phy_voltages[:, 1] = v_far
    results = _make_solver_results(freqs, np.zeros(n_freq, dtype=complex), rx_phy_voltages)
    topology = build_topology(
        drop_positions_m=[1.0, 3.0, 5.0],
        bus_start_m=0.0,
        bus_end_m=7.0,
        tx_drop_index=0,
        termination_ohm=100.0)
    drop = _make_drop_with_admittance(freqs, y_trunk=0.0)
    bus = compute_bus_results(results, topology, drop, phy_load_ohm=20000.0)
    # IL_PHY > 0 for both Drops (voltage under matched source voltage)
    assert np.all(bus.il_phy_db > 0)
    # more damping -> higher IL
    expected_il_near = -20.0 * np.log10(v_near / 0.5)
    expected_il_far  = -20.0 * np.log10(v_far  / 0.5)
    assert np.allclose(bus.il_phy_db[:, 0], expected_il_near, atol=1e-10)
    assert np.allclose(bus.il_phy_db[:, 1], expected_il_far,  atol=1e-10)
    assert np.all(bus.il_phy_db[:, 1] > bus.il_phy_db[:, 0])


def test_il_tci_is_zero_db_for_invisible_drop():
    """Y_drop = 0 (Drop invisible) -> IL_TCI = 0 dB."""
    freqs = _default_frequency_grid()
    n_freq = len(freqs)
    results = _make_solver_results(
        freqs,
        np.zeros(n_freq, dtype=complex),
        np.zeros((n_freq, 0), dtype=complex))
    topology = build_topology(
        drop_positions_m=[3.0],
        bus_start_m=0.0,
        bus_end_m=6.0,
        tx_drop_index=0,
        termination_ohm=100.0)
    # Y_drop = 0 → S21_TCI = 2 / (2 + 100 * 0) = 1 → IL = 0 dB.
    drop = _make_drop_with_admittance(freqs, y_trunk=0.0)
    bus = compute_bus_results(results, topology, drop, phy_load_ohm=20000.0)
    assert bus.il_tci_db.shape == (n_freq, 1)
    assert np.allclose(bus.il_tci_db, 0.0, atol=1e-10)


def test_rl_tci_is_high_for_matched_drop():
    """Y_drop = Y_0 (perfect match) -> Gamma = 0 → RL_TCI sehr hoch"""
    freqs = _default_frequency_grid()
    n_freq = len(freqs)
    results = _make_solver_results(
        freqs,
        np.zeros(n_freq, dtype=complex),
        np.zeros((n_freq, 0), dtype=complex))
    topology = build_topology(
        drop_positions_m=[3.0],
        bus_start_m=0.0,
        bus_end_m=6.0,
        tx_drop_index=0,
        termination_ohm=100.0)
    # Y_drop = Y0 → Gamma = (Y0 - Y0)/(Y0 + Y0) = 0 → RL → inf
    y0 = 1.0 / 100.0
    drop = _make_drop_with_admittance(freqs, y_trunk=y0)
    bus = compute_bus_results(results, topology, drop, phy_load_ohm=20000.0)
    # RL should be very big
    assert np.all(bus.rl_tci_db > 100.0)


def test_il_tci_and_rl_tci_have_correct_shape():
    """IL_TCI and RL_TCI have Shape (n_freq, n_drops) for all Drops"""
    freqs = _default_frequency_grid()
    n_freq = len(freqs)
    # 3 Drops: TX + 2 RX.
    topology = build_topology(
        drop_positions_m=[1.0, 3.0, 5.0],
        bus_start_m=0.0,
        bus_end_m=7.0,
        tx_drop_index=0,
        termination_ohm=100.0)
    n_drops = len(topology.drops)  # = 3
    results = _make_solver_results(
        freqs,
        np.zeros(n_freq, dtype=complex),
        np.zeros((n_freq, 2), dtype=complex))
    drop = _make_drop_with_admittance(freqs, y_trunk=0.0)
    bus = compute_bus_results(results, topology, drop, phy_load_ohm=20000.0)
    assert bus.il_tci_db.shape == (n_freq, n_drops)
    assert bus.rl_tci_db.shape == (n_freq, n_drops)


def test_ms_il_db_is_none():
    """ms_il_db is None while Mixing Segment IL not implemented"""
    freqs = _default_frequency_grid()
    n_freq = len(freqs)
    ms_il = np.zeros(n_freq)
    topology = build_topology(
        drop_positions_m=[3.0],
        bus_start_m=0.0,
        bus_end_m=6.0,
        tx_drop_index=0,
        termination_ohm=100.0)
    drop = _make_drop_with_admittance(freqs, y_trunk=0.0)
    results = _make_solver_results(
        freqs,
        np.zeros(n_freq, dtype=complex),
        np.zeros((n_freq, 0), dtype=complex))
    bus = compute_bus_results(results, topology, drop, phy_load_ohm=20000.0, il_ms_db=ms_il)
    assert bus.il_ms_db is not None
    assert bus.il_ms_db.shape == (n_freq,)


def test_bus_results_all_shapes_correct():
    """BusResults has correct Shapes"""
    freqs = _default_frequency_grid()
    n_freq = len(freqs)
    # TX + 2 RX Drops.
    topology = build_topology(
        drop_positions_m=[1.0, 3.0, 5.0],
        bus_start_m=0.0,
        bus_end_m=7.0,
        tx_drop_index=0,
        termination_ohm=100.0)
    n_drops = len(topology.drops)        # = 3
    n_rx_drops = n_drops - 1             # = 2
    rx_phy_voltages = np.full((n_freq, n_rx_drops), 0.3, dtype=complex)
    results = _make_solver_results(
        freqs,
        np.full(n_freq, 0.1, dtype=complex),
        rx_phy_voltages)
    drop = _make_drop_with_admittance(freqs, y_trunk=0.0) 
    bus = compute_bus_results(results, topology, drop, phy_load_ohm=20000.0, il_ms_db=np.zeros(n_freq))
    assert isinstance(bus, BusResults)
    assert bus.frequency_hz.shape == (n_freq,)
    assert bus.rl_db.shape == (n_freq,)
    assert bus.il_phy_db.shape == (n_freq, n_rx_drops)
    assert bus.il_tci_db.shape == (n_freq, n_drops)
    assert bus.rl_tci_db.shape == (n_freq, n_drops)
    assert bus.il_ms_db.shape == (n_freq,)


def test_smoke_with_real_solver_output():
    """Smoke Test: compute_bus_results runs with real data"""
    real_file = (Path(__file__).parent.parent/"examples"/"LiteVNA_meas_board2_jumped_differential.s2p")
    if not real_file.is_file():
        pytest.skip(f"real measurement file not found at {real_file}")
    freqs = np.linspace(1e6, 30e6, 100)
    topology = build_topology(
        drop_positions_m=[1.0, 3.0, 5.0],
        bus_start_m=0.0,
        bus_end_m=7.0,
        tx_drop_index=0,
        termination_ohm=100.0)
    cable_params = CableParams(
        l_per_m=413e-9,
        c_per_m=45e-12,
        rdc_per_m=0.19,
        rskin_per_m=5e-7)
    drop = load_drop(real_file, freqs)
    phy_load_ohm = 20000.0 
    solver_results = run_simulation(
        topology=topology,
        cable_params=cable_params,
        drop=drop,
        phy_load_ohm=phy_load_ohm,
        frequency_hz=freqs)
    bus = compute_bus_results(solver_results, topology, drop, phy_load_ohm)
    assert np.all(np.isfinite(bus.rl_db))
    assert np.all(np.isfinite(bus.il_phy_db))
    assert np.all(np.isfinite(bus.il_tci_db))
    assert np.all(np.isfinite(bus.rl_tci_db))
    # Physical plausibility.
    assert np.all(bus.rl_db >= 0)
    assert np.all(bus.il_phy_db >= 0)
    assert np.all(bus.il_tci_db >= 0)
    assert np.all(bus.rl_tci_db >= 0)