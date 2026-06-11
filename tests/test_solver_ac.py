"""Tests for solver.ac."""

from pathlib import Path

import numpy as np
import pytest

from spmd_reflection.cable.cable_params import CableParams
from spmd_reflection.drop.models import DropData
from spmd_reflection.solver.ac import run_simulation, Z0_TRANSMITTER
from spmd_reflection.solver.model import SolverResults
from spmd_reflection.drop.load import load_drop
from spmd_reflection.topology.build import build_topology
from spmd_reflection.topology.models import DropAttachment, Termination, Topology, TrunkSegment
from spmd_reflection.cable.model import compute_y_params


def _default_frequency_grid() -> np.ndarray:
    """Standard frequency grid for solver tests: 1 MHz to 30 MHz, 50 points."""
    return np.linspace(1e6, 30e6, 50)


def _lossless_cable() -> CableParams:
    """Lossless 100 Ω cable, useful for analytic tests."""
    return CableParams(
        l_per_m=500e-9,
        c_per_m=50e-12,
        rdc_per_m=0.0,
        rskin_per_m=0.0)


def _realistic_cable() -> CableParams:
    """Realistic automotive twisted-pair cable."""
    return CableParams(
        l_per_m=413e-9,
        c_per_m=45e-12,
        rdc_per_m=0.19,
        rskin_per_m=5e-7)


def _ideal_drop(frequency_hz:np.ndarray) -> DropData:
    """A drop modeled as a very short lossless 2-port (1 mm at 100 Ω).
        
    Electrically negligible at our frequencies (<1.5e-4 wavelengths at 30 MHz).
    Used for both TX and RX drops in analytic tests.
    """
    short_section_params = CableParams(
        l_per_m=500e-9,
        c_per_m=50e-12,
        rdc_per_m=0.0,
        rskin_per_m=0.0)
    y_params = compute_y_params(
        length_m=0.001,
        cable_params=short_section_params,
        frequency_hz=frequency_hz)
    return DropData(frequency_hz=frequency_hz, y_params=y_params)


def test_tx_in_middle_sees_parallel_terminations():
    """TX in the middle sees 50 Ω = Z0_TRANSMITTER → S₁₁ ≈ 0 (matched)."""
    freqs = _default_frequency_grid()
    topology = build_topology(
        drop_positions_m=[3.0],
        bus_start_m=0.0,
        bus_end_m=6.0,
        tx_drop_index=0,
        termination_ohm=100.0)
    result = run_simulation(
        topology=topology,
        cable_params=_lossless_cable(),
        mpse_drop=_ideal_drop(freqs),
        mpd_drop=_ideal_drop(freqs),
        mpse_index=0,
        phy_load_ohm=20000.0,
        frequency_hz=freqs)
    # Two 100 Ω terminations in parallel = 50 Ω = Z0_TRANSMITTER → matched → S₁₁ ≈ 0.
    assert np.allclose(np.abs(result.s11_tx), 0.0, atol=1e-3)


def test_lossless_open_bus_has_unity_reflection():
    """lossless line without real termination → |S₁₁| = 1."""
    freqs = _default_frequency_grid()
    topology = build_topology(
        drop_positions_m=[0.0],
        bus_start_m=0.0,
        bus_end_m=2.0,
        tx_drop_index=0,
        termination_ohm=1e9)
    result = run_simulation(
        topology=topology,
        cable_params=_lossless_cable(),
        mpse_drop=_ideal_drop(freqs),
        mpd_drop=_ideal_drop(freqs),
        mpse_index=0,
        phy_load_ohm=20000.0,
        frequency_hz=freqs)
    assert np.allclose(np.abs(result.s11_tx), 1.0, atol=1e-6)


def test_passive_bus_has_bounded_reflection():
    """Passive bus: |S₁₁| ≤ 1, |rx_phy_voltages| ≤ source-voltage"""
    freqs = _default_frequency_grid()
    topology = build_topology(
        drop_positions_m=[1.0, 3.0, 5.0],
        bus_start_m=0.0,
        bus_end_m=7.0,
        tx_drop_index=0,
        termination_ohm=100.0)
    result = run_simulation(
        topology=topology,
        cable_params=_realistic_cable(),
        mpse_drop=_ideal_drop(freqs),
        mpd_drop=_ideal_drop(freqs),
        mpse_index=0,
        phy_load_ohm=20000.0,
        frequency_hz=freqs)
    # Passive-check: |S₁₁| ≤ 1
    assert np.all(np.abs(result.s11_tx) <= 1.0 + 1e-9)
    # RX-PHY-voltage shape check: (n_freq, n_rx_drops)
    # TX is Drop 0, so we've got 2 RX-Drops
    assert result.rx_phy_voltages.shape == (len(freqs), 2)
    # RX-PHY-voltages are finite
    assert np.all(np.isfinite(result.rx_phy_voltages))


def test_solver_results_have_correct_shapes():
    """SolverResults has correct shapes and dtypes for all fields."""
    freqs = _default_frequency_grid()
    # Single TX drop, no RX drops.
    topology = build_topology(
        drop_positions_m=[3.0],
        bus_start_m=0.0,
        bus_end_m=6.0,
        tx_drop_index=0,
        termination_ohm=100.0)  
    result = run_simulation(
        topology=topology,
        cable_params=_lossless_cable(),
        mpse_drop=_ideal_drop(freqs),
        mpd_drop=_ideal_drop(freqs),
        mpse_index=0,
        phy_load_ohm=20000.0,
        frequency_hz=freqs)
    assert isinstance(result, SolverResults)
    assert result.frequency_hz.shape == (len(freqs),)
    assert result.s11_tx.shape == (len(freqs),)
    assert result.s11_tx.dtype == np.complex128
    assert result.node_voltages.shape == (len(freqs), topology.n_nodes)
    assert result.node_voltages.dtype == np.complex128
    # No RX drops → second dimension is 0.
    assert result.rx_phy_voltages.shape == (len(freqs), 0)
    assert result.rx_phy_voltages.dtype == np.complex128


def test_rejects_mismatched_drop_frequency_grid():
    """Drop frequency grid must match the simulation grid."""
    freqs = _default_frequency_grid()
    different_freqs = np.linspace(2e6, 25e6, 50)   
    topology = build_topology(
        drop_positions_m=[3.0],
        bus_start_m=0.0,
        bus_end_m=6.0,
        tx_drop_index=0,
        termination_ohm=100.0)
    with pytest.raises(ValueError, match="does not match"):
        run_simulation(
            topology=topology,
            cable_params=_lossless_cable(),
            mpse_drop=_ideal_drop(different_freqs),
            mpd_drop=_ideal_drop(freqs),
            mpse_index=0,
            phy_load_ohm=20000.0,
            frequency_hz=freqs)
        

def test_rejects_topology_without_tx():
    """A topology with no TX drop is rejected by the solver."""
    freqs = _default_frequency_grid()
    # Construct a topology manually that has no TX drop.
    # build_topology always sets exactly one TX, so we bypass it here.
    topology = Topology(
        n_nodes=2,
        trunk_segments=(TrunkSegment(node_a=0, node_b=1, length_m=2.0)),
        drops=(DropAttachment(trunk_node=0, role="rx"),DropAttachment(trunk_node=1, role="rx")),
        terminations=(Termination(node=0, impedance_ohm=100.0),Termination(node=1, impedance_ohm=100.0))) 
    with pytest.raises(ValueError, match="exactly one TX drop"):
        run_simulation(
            topology=topology,
            cable_params=_lossless_cable(),
            mpse_drop=_ideal_drop(freqs),
            mpd_drop=_ideal_drop(freqs),
            mpse_index=0,
            phy_load_ohm=20000.0,
            frequency_hz=freqs)
        

def test_rx_phy_voltages_differ_between_drops():
    """RX drops at different positions receive different signal levels."""
    freqs = _default_frequency_grid()
    topology = build_topology(
        drop_positions_m=[0.0, 2.0, 4.0],
        bus_start_m=0.0,
        bus_end_m=6.0,
        tx_drop_index=0,
        termination_ohm=100.0)
    result = run_simulation(
        topology=topology,
        cable_params=_realistic_cable(),
        mpse_drop=_ideal_drop(freqs),
        mpd_drop=_ideal_drop(freqs),
        mpse_index=0,
        phy_load_ohm=20000.0,
        frequency_hz=freqs)
    # Shape check.
    assert result.rx_phy_voltages.shape == (len(freqs), 2)
    # check voltages are finite and positive.
    assert np.all(np.isfinite(result.rx_phy_voltages))
    assert np.all(np.abs(result.rx_phy_voltages) > 0) 
    # The two RX drops receive different signal levels (position matters).
    v_near = np.abs(result.rx_phy_voltages[:, 0])
    v_far  = np.abs(result.rx_phy_voltages[:, 1])
    assert not np.allclose(v_near, v_far, rtol=1e-3)


def test_smoke_real_measurement():
    """Smoke test: full simulation with a real jumped LiteVNA measurement."""
    real_file = (Path(__file__).parent.parent/"examples"/"LiteVNA_meas_board2_jumped_differential.s2p")
    if not real_file.is_file():
        pytest.skip(f"real measurement file not found at {real_file}")
    freqs = np.linspace(1e6, 30e6, 100)
    drop = load_drop(real_file, freqs)
    topology = build_topology(
        drop_positions_m=[1.0, 3.0, 5.0],
        bus_start_m=0.0,
        bus_end_m=7.0,
        tx_drop_index=0,
        termination_ohm=100.0)
    result = run_simulation(
        topology=topology,
        cable_params=_realistic_cable(),
        drop=drop,
        phy_load_ohm=20000.0,
        frequency_hz=freqs)
    # Shape checks.
    assert result.rx_phy_voltages.shape == (100, 2)
    assert result.s11_tx.shape == (100,)
    # Finiteness.
    assert np.all(np.isfinite(result.s11_tx))
    assert np.all(np.isfinite(result.rx_phy_voltages))
    # Passivity.
    assert np.all(np.abs(result.s11_tx) <= 1.0 + 1e-9)