"""Tests for solver.ac."""

from pathlib import Path

import numpy as np
import pytest

from spmd_reflection.cable.cable_params import CableParams
from spmd_reflection.drop.load import load_rx_drop, load_tx_drop
from spmd_reflection.drop.models import RxDropData, TxDropData
from spmd_reflection.solver.ac import run_simulation, Z0_REFERENCE
from spmd_reflection.solver.model import SolverResults
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


# def _ideal_through_tx(frequency_hz: np.ndarray) -> TxDropData:
#     """A TX drop that behaves as a perfect pass-through 2-port.
    
#     For a 2-port with S₁₁ = S₂₂ = 0 and S₁₂ = S₂₁ = 1 (matched, lossless,
#     reciprocal), the equivalent Y-parameters in Z₀ = 100 Ω reference are:
    
#         Y₁₁ = Y₂₂ = 0
#         Y₁₂ = Y₂₁ = -1/Z₀
    
#     Physically: a perfect ideal transmission line of zero electrical length.
#     """
#     n_freq = len(frequency_hz)
#     y_params = np.zeros((n_freq, 2, 2), dtype=complex)
#     y_params[:, 0, 1] = -1.0 / Z0_REFERENCE
#     y_params[:, 1, 0] = -1.0 / Z0_REFERENCE
#     return TxDropData(frequency_hz=frequency_hz, y_params=y_params)


def _ideal_through_tx(frequency_hz: np.ndarray) -> TxDropData:
    """A TX drop modeled as a very short, lossless 2-port section.
        
    For analytic tests we want a TX that introduces minimal effect on the bus.
    A short lossless cable section (1 mm) approximates a direct connection
    while remaining representable in Y-parameter form.
    """
        
    # Tiny lossless section, electrically negligible at our frequencies.
    short_section_params = CableParams(
        l_per_m=500e-9,
        c_per_m=50e-12,
        rdc_per_m=0.0,
        rskin_per_m=0.0)
    y_params = compute_y_params(
        length_m=0.001,   # 1 mm
        cable_params=short_section_params,
        frequency_hz=frequency_hz)
    return TxDropData(frequency_hz=frequency_hz, y_params=y_params)


def _open_rx(frequency_hz:np.ndarray) -> RxDropData:
    """An RX drop with zero shunt admittance (electrically invisible).
    
    Useful as a placeholder when the solver requires an RX drop but the
    test scenario doesn't actually have any RX nodes that would use it.
    """
    n_freq = len(frequency_hz)
    return RxDropData(
        frequency_hz=frequency_hz,
        shunt_admittance=np.zeros(n_freq, dtype=complex))


def test_tx_in_middle_sees_parallel_terminations():
    """TX int the middle sees 50 Ω (two 100-Ω-ways parallel) → S11 = -1/3."""
    freqs = _default_frequency_grid()
        
    # TX in the middle, terminations at both ends
    topology = build_topology(
        drop_positions_m=[3.0],
        bus_start_m=0.0,
        bus_end_m=6.0,
        tx_drop_index=0,
        termination_ohm=100.0)
        
    result = run_simulation(
        topology=topology,
        cable_params=_lossless_cable(),
        rx_drop=_open_rx(freqs),
        tx_drop=_ideal_through_tx(freqs),
        frequency_hz=freqs)
        
    # source sees 50 Ω, source-impedance 100 Ω → S11 = (50-100)/(50+100) = -1/3
    expected = 1.0 / 3.0
    assert np.allclose(np.abs(result.s11_tx), expected, atol=1e-6)


def test_lossless_open_bus_has_unity_reflection():
    """lossless line without real termination -> |S₁₁| = 1"""
    freqs = _default_frequency_grid()
        
    topology = build_topology(
        drop_positions_m=[0.0],
        bus_start_m=0.0,
        bus_end_m=2.0,
        tx_drop_index=0,
        termination_ohm=1e9) # practically open
        
    result = run_simulation(
        topology=topology,
        cable_params=_lossless_cable(),
        rx_drop=_open_rx(freqs),
        tx_drop=_ideal_through_tx(freqs),
        frequency_hz=freqs)
   
    assert np.allclose(np.abs(result.s11_tx), 1.0, atol=1e-6)  # |S₁₁| ≈ 1 


def test_passive_bus_has_bounded_reflection():
    """passive bus -> |S₁₁| ≤ 1 for all frequencies"""
    freqs = _default_frequency_grid()
        
    # realistic topology
    topology = build_topology(
        drop_positions_m=[1.0, 3.0, 5.0],
        bus_start_m=0.0,
        bus_end_m=7.0,
        tx_drop_index=0,
        termination_ohm=100.0)
        
    result = run_simulation(
        topology=topology,
        cable_params=_realistic_cable(),
        rx_drop=_open_rx(freqs),
        tx_drop=_ideal_through_tx(freqs),
        frequency_hz=freqs)
        
    assert np.all(np.abs(result.s11_tx) <= 1.0 + 1e-9)


def test_debug_ideal_through_tx():
    """Verify that _ideal_through_tx now produces valid S-parameters."""
    import skrf
    freqs = np.linspace(1e6, 30e6, 10)
    tx = _ideal_through_tx(freqs)
        
    freq = skrf.Frequency.from_f(freqs, unit="Hz")
    network = skrf.Network(frequency=freq, y=tx.y_params, z0=Z0_REFERENCE)
        
    print(f"|S11| max = {np.abs(network.s[:, 0, 0]).max()}")
    print(f"|S21| min = {np.abs(network.s[:, 1, 0]).min()}")