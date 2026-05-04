"""Tests for drop.load."""

from pathlib import Path
import numpy as np
import pytest
import skrf

from spmd_reflection.drop.load import load_drop, Z0_REFERENCE
from spmd_reflection.drop.models import DropData


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
    file_freqs = np.linspace(0.5e6, 35e6, 20)
    n_freq = len(file_freqs)
    # Simple S-parameters: matched, lossless through.
    s_params = np.zeros((n_freq, 2, 2), dtype=complex)
    s_params[:, 0, 1] = 1.0
    s_params[:, 1, 0] = 1.0
    ts_path = tmp_path / "test.s2p"
    _write_test_touchstone(ts_path, file_freqs, s_params)
    sim_freqs = _default_frequency_grid()
    drop = load_drop(ts_path, sim_freqs)
    # Returned frequency grid matches the simulation grid.
    assert np.array_equal(drop.frequency_hz, sim_freqs)
    # Y-parameters have correct shape.
    assert drop.y_params.shape == (len(sim_freqs), 2, 2)
    # Y-parameters are complex.
    assert drop.y_params.dtype == np.complex128


def test_rejects_non_2port_touchstone(tmp_path):
    """A 4-port Touchstone file is rejected by the loader."""
    file_freqs = np.linspace(0.5e6, 35e6, 10)
    n_freq = len(file_freqs)
    s_params = np.zeros((n_freq, 4, 4), dtype=complex)
    ts_path = tmp_path / "fourport.s4p"
    _write_test_touchstone(ts_path, file_freqs, s_params)
    sim_freqs = _default_frequency_grid()
    with pytest.raises(ValueError, match="expected a 2-port"):
        load_drop(ts_path, sim_freqs)


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
        load_drop(ts_path, sim_freqs)


def test_renormalizes_to_reference_impedance(tmp_path):
    """A Touchstone file with non-100Ω reference is renormalized to Z0_REFERENCE."""
    file_freqs = np.linspace(0.5e6, 35e6, 20)
    n_freq = len(file_freqs)  
    # File with 50 Ω reference, S22 = 0 (matched in 50 Ω).
    # After renormalization to 100 Ω, S22 should become -1/3.
    s_params_in_50ohm = np.zeros((n_freq, 2, 2), dtype=complex)
    ts_path = tmp_path / "fifty_ohm.s2p"
    _write_test_touchstone(ts_path, file_freqs, s_params_in_50ohm, z0=50.0)  
    sim_freqs = _default_frequency_grid()
    drop = load_drop(ts_path, sim_freqs) 
    # After renormalization, recompute S-parameters from the loaded Y-matrix
    # and verify S22 = -1/3 (the matched 50 Ω load seen in 100 Ω reference).
    freq = skrf.Frequency.from_f(sim_freqs, unit="Hz")
    network_check = skrf.Network(frequency=freq, y=drop.y_params, z0=Z0_REFERENCE)
    expected_s22 = -1.0 / 3.0
    assert np.allclose(network_check.s[:, 1, 1], expected_s22, atol=1e-6)


def test_load_drop_returns_correct_data_structure(tmp_path):
    """load_drop returns DropData with correct shapes and types."""
    file_freqs = np.linspace(0.5e6, 35e6, 20)
    n_freq = len(file_freqs)
    s_params = np.zeros((n_freq, 2, 2), dtype=complex)
    ts_path = tmp_path / "test.s2p"
    _write_test_touchstone(ts_path, file_freqs, s_params)
    sim_freqs = _default_frequency_grid()
    drop = load_drop(ts_path, sim_freqs)
    # Type check.
    assert isinstance(drop, DropData)
    # Frequency grid matches simulation.
    assert drop.frequency_hz.shape == (len(sim_freqs),)
    # Y-parameters have shape (n_freq, 2, 2).
    assert drop.y_params.shape == (len(sim_freqs), 2, 2)
    # Y-parameters are complex.
    assert drop.y_params.dtype == np.complex128


def test_y_params_match_skrf_conversion(tmp_path):
    """Y-parameters returned by load_drop match skrf's S→Y conversion."""
    file_freqs = np.linspace(0.5e6, 35e6, 20)
    n_freq = len(file_freqs) 
    # Choose nontrivial S-parameters so the Y-conversion is meaningful.
    s_params = np.zeros((n_freq, 2, 2), dtype=complex)
    s_params[:, 0, 0] = 0.1 + 0.2j     # some reflection at port 0
    s_params[:, 1, 1] = 0.05 - 0.15j   # different reflection at port 1
    s_params[:, 0, 1] = 0.9             # nearly lossless through
    s_params[:, 1, 0] = 0.9             # reciprocal
    ts_path = tmp_path / "nontrivial.s2p"
    _write_test_touchstone(ts_path, file_freqs, s_params)  
    sim_freqs = _default_frequency_grid()
    drop = load_drop(ts_path, sim_freqs)
    # Independently compute the expected Y-parameters and compare.
    network_check = skrf.Network(str(ts_path))
    target_freq = skrf.Frequency.from_f(sim_freqs, unit="Hz")
    network_check = network_check.interpolate(target_freq)    
    assert np.allclose(drop.y_params, network_check.y, atol=1e-10)


def test_smoke_loads_real_jumped_measurement():
    """Smoke test: load a real jumped LiteVNA measurement and verify reasonable output.
        
    This test runs only if the measurement file is available in the project.
    It does not verify specific values — only that the loader handles a real
    jumped measurement end-to-end without errors.
    """
    # Path to the project's real jumped measurement. Adjust if your repo layout differs.
    real_file = (Path(__file__).parent.parent/"examples"/"LiteVNA_meas_board2_jumped_differential.s2p")    
    if not real_file.is_file():
        pytest.skip(f"real jumped measurement file not found at {real_file}")
    sim_freqs = np.linspace(1e6, 30e6, 100)
    drop = load_drop(real_file, sim_freqs)
    # Sanity: shape and dtype.
    assert drop.y_params.shape == (100, 2, 2)
    assert drop.y_params.dtype == np.complex128
    # Sanity: Y-parameters are finite everywhere.
    assert np.all(np.isfinite(drop.y_params))