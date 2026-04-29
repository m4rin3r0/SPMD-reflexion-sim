"""Tests for drop.load."""

from pathlib import Path
import numpy as np
import pytest
import skrf

from spmd_reflection.drop.load import load_rx_drop, load_tx_drop, Z0_REFERENCE
from spmd_reflection.drop.models import RxDropData, TxDropData


def _write_test_touchstone(path:Path, frequency_hz:np.ndarray, s_params:np.ndarray, z0:float=100.0) -> None:
    """Write a synthetic Touchstone file for testing.
    
    Args:
        path: Destination path. Extension implies port count (.s2p, .s4p).
        frequency_hz: 1D array of frequency points.
        s_params: Complex array of shape (n_freq, n_ports, n_ports).
        z0: Reference impedance (default 100 Ω, matching simulation default).
    """
    freq = skrf.Frequency.from_f(frequency_hz, unit="Hz")
    network = skrf.Network(frequency=freq, s=s_params, z0=z0)
    network.write_touchstone(str(path), write_z0=True)


def _default_frequency_grid() -> np.ndarray:
    """Standard simulation frequency grid for tests: 1 MHz to 30 MHz, 50 points."""
    return np.linspace(1e6, 30e6, 50)


def test_loads_and_interpolates_touchstone(tmp_path):
    """A valid 2-port Touchstone is loaded and interpolated onto the simulation grid."""
    # Touchstone file with a coarser frequency grid than the simulation.
    file_freqs = np.linspace(0.5e6, 35e6, 20)
    n_freq = len(file_freqs)
    # Simple S-parameters: Identity-like behavior (no reflection, perfect through).
    # S11 = S22 = 0 (matched), S12 = S21 = 1 (lossless through).
    s_params = np.zeros((n_freq, 2, 2), dtype=complex)
    s_params[:, 0, 1] = 1.0
    s_params[:, 1, 0] = 1.0
    ts_path = tmp_path / "test.s2p"
    _write_test_touchstone(ts_path, file_freqs, s_params)
    # Use a finer simulation grid that fits within the file's range.
    sim_freqs = _default_frequency_grid()
    drop = load_rx_drop(ts_path, sim_freqs)   
    # Returned frequency grid matches the simulation grid.
    assert np.array_equal(drop.frequency_hz, sim_freqs)
    # Shunt admittance should be 1/Z0 because S22 = 0 means matched termination.
    expected_y = np.full(len(sim_freqs), 1.0 / Z0_REFERENCE, dtype=complex)
    assert np.allclose(drop.shunt_admittance, expected_y, atol=1e-12)


def test_rejects_non_2port_touchstone(tmp_path):
    """A 4-port Touchstone file is rejected by the loader."""
    file_freqs = np.linspace(0.5e6, 35e6, 10)
    n_freq = len(file_freqs)
    s_params = np.zeros((n_freq, 4, 4), dtype=complex)
    ts_path = tmp_path / "fourport.s4p"
    _write_test_touchstone(ts_path, file_freqs, s_params)
    sim_freqs = _default_frequency_grid()
    with pytest.raises(ValueError, match="expected a 2-port"):
        load_rx_drop(ts_path, sim_freqs)


def test_rejects_simulation_range_outside_file_range(tmp_path):
    """The simulation grid must lie within the Touchstone's frequency range."""
    # Touchstone covers only 5-20 MHz.
    file_freqs = np.linspace(5e6, 20e6, 10)
    n_freq = len(file_freqs)
    s_params = np.zeros((n_freq, 2, 2), dtype=complex)
    ts_path = tmp_path / "narrow.s2p"
    _write_test_touchstone(ts_path, file_freqs, s_params)  
    # Simulation grid extends from 1 MHz to 30 MHz — wider than the file.
    sim_freqs = np.linspace(1e6, 30e6, 50)  
    with pytest.raises(ValueError, match="not covered"):
        load_rx_drop(ts_path, sim_freqs)


def test_renormalizes_to_reference_impedance(tmp_path):
    """A Touchstone file with non-100Ω reference is renormalized to Z0_REFERENCE."""
    # File uses 50 Ω reference impedance.
    file_freqs = np.linspace(0.5e6, 35e6, 20)
    n_freq = len(file_freqs)  
    # Choose S-parameters so that after renormalization to 100 Ω,
    # the result is predictable. Easiest: matched in 50 Ω → S = 0.
    # In 100 Ω, a 50 Ω termination has Γ = (50-100)/(50+100) = -1/3.
    s_params_in_50ohm = np.zeros((n_freq, 2, 2), dtype=complex)  
    ts_path = tmp_path / "fifty_ohm.s2p"
    _write_test_touchstone(ts_path, file_freqs, s_params_in_50ohm, z0=50.0)  
    sim_freqs = _default_frequency_grid()
    drop = load_rx_drop(ts_path, sim_freqs)
    # After renormalization to 100 Ω, S22 should be -1/3 at all frequencies.
    # Y_shunt = (1 - S22) / (Z0 · (1 + S22))
    #        = (1 - (-1/3)) / (100 · (1 + (-1/3)))
    #        = (4/3) / (100 · 2/3)
    #        = (4/3) / (200/3)
    #        = 4/200 = 1/50
    # This corresponds to a 50 Ω shunt — correct, because the original file
    # had a matched 50 Ω termination.
    expected_y = np.full(len(sim_freqs), 1.0 / 50.0, dtype=complex)
    assert np.allclose(drop.shunt_admittance, expected_y, atol=1e-10)


def test_rx_drop_with_matched_termination_yields_inverse_z0(tmp_path):
    """A drop with S₂₂ = 0 (matched) yields shunt admittance = 1/Z₀."""
    file_freqs = np.linspace(0.5e6, 35e6, 20)
    n_freq = len(file_freqs)
    s_params = np.zeros((n_freq, 2, 2), dtype=complex)
    # S22 = 0 (matched) — already zero in the all-zeros array.
    ts_path = tmp_path / "matched.s2p"
    _write_test_touchstone(ts_path, file_freqs, s_params)
    sim_freqs = _default_frequency_grid()
    drop = load_rx_drop(ts_path, sim_freqs)
    expected_y = np.full(len(sim_freqs), 1.0 / Z0_REFERENCE, dtype=complex)
    assert np.allclose(drop.shunt_admittance, expected_y, atol=1e-12)


def test_rx_drop_with_open_termination_yields_zero_admittance(tmp_path):
    """A drop with S₂₂ = 1 (open) yields shunt admittance = 0."""
    file_freqs = np.linspace(0.5e6, 35e6, 20)
    n_freq = len(file_freqs)
    s_params = np.zeros((n_freq, 2, 2), dtype=complex)
    s_params[:, 1, 1] = 1.0  # S22 = 1 → open termination
    ts_path = tmp_path / "open.s2p"
    _write_test_touchstone(ts_path, file_freqs, s_params) 
    sim_freqs = _default_frequency_grid()
    drop = load_rx_drop(ts_path, sim_freqs)  
    # Y_shunt = (1 - 1) / (Z0 · (1 + 1)) = 0
    expected_y = np.zeros(len(sim_freqs), dtype=complex)
    assert np.allclose(drop.shunt_admittance, expected_y, atol=1e-12)


def test_load_rx_drop_returns_correct_data_structure(tmp_path):
    """load_rx_drop returns RxDropData with correct shapes and types."""
    file_freqs = np.linspace(0.5e6, 35e6, 20)
    n_freq = len(file_freqs)
    s_params = np.zeros((n_freq, 2, 2), dtype=complex)
    ts_path = tmp_path / "test.s2p"
    _write_test_touchstone(ts_path, file_freqs, s_params)
    sim_freqs = _default_frequency_grid()
    drop = load_rx_drop(ts_path, sim_freqs)
    # Type and shape checks.
    assert isinstance(drop, RxDropData)
    assert drop.frequency_hz.shape == (len(sim_freqs),)
    assert drop.shunt_admittance.shape == (len(sim_freqs),)
    assert drop.shunt_admittance.dtype == np.complex128


def test_load_tx_drop_returns_correct_data_structure(tmp_path):
    """load_tx_drop returns TxDropData with Y-parameters of shape (n_freq, 2, 2)."""
    file_freqs = np.linspace(0.5e6, 35e6, 20)
    n_freq = len(file_freqs)
    s_params = np.zeros((n_freq, 2, 2), dtype=complex)
    ts_path = tmp_path / "tx.s2p"
    _write_test_touchstone(ts_path, file_freqs, s_params)
    sim_freqs = _default_frequency_grid()
    drop = load_tx_drop(ts_path, sim_freqs)
    assert isinstance(drop, TxDropData)
    assert drop.frequency_hz.shape == (len(sim_freqs),)
    assert drop.y_params.shape == (len(sim_freqs), 2, 2)
    assert drop.y_params.dtype == np.complex128


def test_load_tx_drop_converts_s_to_y_correctly(tmp_path):
    """Y-parameters from load_tx_drop match the analytical conversion of S-parameters."""
    file_freqs = np.linspace(0.5e6, 35e6, 20)
    n_freq = len(file_freqs)
    # Build a synthetic S-matrix that's a non-trivial 2-port.
    # S11 = 0.2, S22 = 0.3, S12 = S21 = 0.5 (reciprocal, lossy).
    s_params = np.zeros((n_freq, 2, 2), dtype=complex)
    s_params[:, 0, 0] = 0.2
    s_params[:, 1, 1] = 0.3
    s_params[:, 0, 1] = 0.5
    s_params[:, 1, 0] = 0.5
    ts_path = tmp_path / "tx_complex.s2p"
    _write_test_touchstone(ts_path, file_freqs, s_params)
    sim_freqs = _default_frequency_grid()
    drop = load_tx_drop(ts_path, sim_freqs)
    # Build a reference Network with the same S-params at sim_freqs and read its .y.
    ref_freq = skrf.Frequency.from_f(sim_freqs, unit="Hz")
    ref_s = np.zeros((len(sim_freqs), 2, 2), dtype=complex)
    ref_s[:, 0, 0] = 0.2
    ref_s[:, 1, 1] = 0.3
    ref_s[:, 0, 1] = 0.5
    ref_s[:, 1, 0] = 0.5
    ref_network = skrf.Network(frequency=ref_freq, s=ref_s, z0=Z0_REFERENCE)
    assert np.allclose(drop.y_params, ref_network.y, atol=1e-10)


def test_smoke_loads_real_measurement(tmp_path):
    """Smoke test: load a real LiteVNA measurement and verify reasonable output.
        
    This test runs only if the measurement file is available in the project.
    It does not verify specific values — only that the loader handles a real
    file end-to-end without errors and returns plausible orders of magnitude.
    """
    # Path to the project's real measurement file. Adjust if your repo layout differs.
    real_file = Path(__file__).parent.parent / "examples" / "LiteVNA_meas_board2_differential.s2p"
    if not real_file.is_file():
        pytest.skip(f"real measurement file not found at {real_file}")
    sim_freqs = np.linspace(1e6, 30e6, 100)
    drop = load_rx_drop(real_file, sim_freqs) 
    # Sanity: shape and dtype.
    assert drop.shunt_admittance.shape == (100,)
    assert drop.shunt_admittance.dtype == np.complex128
    # Sanity: admittance is finite everywhere (no NaN, no inf).
    assert np.all(np.isfinite(drop.shunt_admittance))
    # Sanity: admittance magnitude is in a plausible range for an RX drop.
    # An ideal high-impedance drop has |Y| ≪ 1/Z₀. We allow a generous bound.
    abs_y = np.abs(drop.shunt_admittance)
    assert np.all(abs_y < 1.0), f"admittance magnitude unexpectedly large: max = {abs_y.max()}"