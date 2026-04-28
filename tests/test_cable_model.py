"""Tests for cable.model.

Covers Y-parameter computation for lossy transmission lines using analytic
limit cases, physical invariants, and negative tests for input validation.
"""

import numpy as np
import pytest
from spmd_reflection.cable.model import compute_y_params
from spmd_reflection.cable.cable_params import CableParams

"""Table of content
positive tests:
    C1: lossless line -> real Z₀?
    C2: high-frequency LC-limit
    C3: quarter-wavelength lossless line
    C4: short line
    C5: symmetry Y₁₁ = Y₂₂
    C6: reciprocal Y₁₂ = Y₂₁
    C7: passive line non negative input impedance
    C8: correct form and type
negative tests:
    C9-C12: validation
"""

def realistic_params() -> CableParams:
    """Return a set of plausible automotive twisted-pair cable parameters.

    These values correspond to Z₀ ≈ 96 Ω at 10 MHz and a propagation velocity
    factor of about 0.77 — within the IEEE 802.3da spec (100 Ω ± 5 Ω).
    """
    return CableParams(
        l_per_m=413e-9,     # 413 nH/m
        c_per_m=45e-12,     # 45 pF/m
        rdc_per_m=0.19,     # 0.19 Ω/m
        rskin_per_m=5e-7)   # 5×10⁻⁷ Ω/(m·√Hz)


def lossless_params() -> CableParams:
    """Return parameters for a lossless 100 Ω line.
    
    L'/C' ratio gives Z₀ = 100 Ω exactly. No DC resistance, no skin effect.
    Useful for tests against analytic expressions.
    """
    # Z₀ = √(L'/C') = 100 → L'/C' = 10000
    # Choose L' = 500 nH/m → C' = 50 pF/m
    return CableParams(
        l_per_m=500e-9,
        c_per_m=50e-12,
        rdc_per_m=0.0,
        rskin_per_m=0.0)


def test_lossless_line_has_real_z0_and_imaginary_gamma():
    """In a lossless line, Z₀ = √(L'/C') is real and γ = jω√(L'C') is imaginary."""
    params = lossless_params()
    # Test at one specific frequency.
    f = 10e6  # 10 MHz
    omega = 2 * np.pi * f
    # Expected values from analytic formulas.
    expected_z0 = np.sqrt(params.l_per_m / params.c_per_m)  # = 100
    expected_gamma_imag = omega * np.sqrt(params.l_per_m * params.c_per_m)
    z0_computed = params.z0_at(f) # Get the values the way the module computes them.   
    # Z₀ should be essentially real.
    assert z0_computed.real == pytest.approx(expected_z0, rel=1e-12)
    assert abs(z0_computed.imag) < 1e-12  
    # γ is implicit in compute_y_params — we verify it indirectly via the
    # resulting Y-parameters below. Here we just verify the analytic setup.
    assert expected_z0 == pytest.approx(100.0, rel=1e-12)
    assert expected_gamma_imag > 0


def test_z0_approaches_lc_limit_at_high_frequency():
    """At high frequency, Z₀ → √(L'/C') because inductive reactance dominates resistance."""
    params = realistic_params()
    # LC limit (Z₀ at infinite frequency).
    lc_limit = np.sqrt(params.l_per_m / params.c_per_m) 
    # At 40 MHz (upper end of 10BASE-T1M spec), inductive reactance already
    # dominates for typical automotive cable parameters.
    z0_high = params.z0_at(40e6) 
    # Real part close to LC limit, imaginary part small compared to real part.
    assert z0_high.real == pytest.approx(lc_limit, rel=1e-2)
    assert abs(z0_high.imag) < 0.01 * z0_high.real


def test_quarter_wavelength_lossless_line_has_zero_y11():
    """A lossless quarter-wavelength line has Y₁₁ = cot(π/2)/Z₀ = 0."""
    params = lossless_params()
    # Choose a frequency and compute the corresponding quarter wavelength.
    f = 10e6  # 10 MHz
    vp = 1.0 / np.sqrt(params.l_per_m * params.c_per_m)
    wavelength = vp / f
    length_m = wavelength / 4 
    # Compute Y-parameters at exactly this frequency.
    y = compute_y_params(length_m=length_m, cable_params=params, frequency_hz=np.array([f]))  
    # Y₁₁ should be essentially zero at quarter wavelength.
    assert abs(y[0, 0, 0]) < 1e-10


def test_short_line_approximates_series_inductor():
    """When γl ≪ 1, the line approximates a lumped series L + shunt C network.
        
    Specifically: Y₁₂ ≈ -1/(jωL_total), where L_total = L' · length.
    """
    params = lossless_params()
    # Very short line at moderate frequency → electrically short regime.
    length_m = 0.01  # 1 cm
    f = 10e6        # 10 MHz
    # At 10 MHz, λ/4 is 5 m (see Test C3). A 1 cm line is 500× shorter.
    # Expect γl ≈ 2π·10⁷ · √(500e-9 · 50e-12) · 0.01 ≈ 3.1e-3 ≪ 1. 
    y = compute_y_params(length_m=length_m, cable_params=params, frequency_hz=np.array([f])) 
    # For short line: Y₁₂ ≈ -1/(jωL_total)
    omega = 2 * np.pi * f
    l_total = params.l_per_m * length_m
    expected_y12 = -1.0 / (1j * omega * l_total)
    # Accept 1% relative error — short-line approximation is imperfect.
    assert y[0, 0, 1] == pytest.approx(expected_y12, rel=1e-2)


def test_y_matrix_is_symmetric():
    """For a uniform line, Y₁₁ = Y₂₂ at every frequency."""
    params = realistic_params()
    freqs = np.linspace(0.3e6, 40e6, 100)
    y = compute_y_params(length_m=2.0, cable_params=params, frequency_hz=freqs)
    # Y₁₁ should equal Y₂₂ at all frequencies (symmetric 2-port).
    assert np.allclose(y[:, 0, 0], y[:, 1, 1], atol=1e-12)


def test_y_matrix_is_reciprocal():
    """For a passive line without gyrators, Y₁₂ = Y₂₁ at every frequency."""
    params = realistic_params()
    freqs = np.linspace(0.3e6, 40e6, 100)
    y = compute_y_params(length_m=2.0, cable_params=params, frequency_hz=freqs)
    # Y₁₂ should equal Y₂₁ (reciprocity theorem for passive networks).
    assert np.allclose(y[:, 0, 1], y[:, 1, 0], atol=1e-12)


def test_passive_line_has_non_negative_input_resistance():
    """With a passive resistive load at port 2, the input impedance at port 1
    must have a non-negative real part (passivity constraint)."""
    params = realistic_params()
    freqs = np.linspace(0.3e6, 40e6, 100)
    y = compute_y_params(length_m=5.0, cable_params=params, frequency_hz=freqs)
    # Terminate port 2 with Z_L = 100 Ω (matched-ish load).
    z_load = 100.0
    y_load = 1.0 / z_load
    # Input admittance at port 1 with load at port 2:
    # Y_in = Y₁₁ - Y₁₂·Y₂₁ / (Y₂₂ + Y_load)
    y_in = y[:, 0, 0] - y[:, 0, 1] * y[:, 1, 0] / (y[:, 1, 1] + y_load)
    z_in = 1.0 / y_in
    # Passivity: Re(Z_in) ≥ 0 at all frequencies.
    assert np.all(z_in.real >= 0)


def test_output_has_correct_shape_and_dtype():
    """compute_y_params returns a complex array of shape (n_freq, 2, 2)."""
    params = realistic_params()
    freqs = np.linspace(1e6, 40e6, 50)
    y = compute_y_params(length_m=2.0, cable_params=params, frequency_hz=freqs)
    assert y.shape == (50, 2, 2)
    assert y.dtype == np.complex128


def test_rejects_negative_length():
    """length_m must be positive."""
    with pytest.raises(ValueError, match="length_m must be positive"):
        compute_y_params(
            length_m=-1.0,
            cable_params=realistic_params(),
            frequency_hz=np.array([1e6, 10e6]))


def test_rejects_zero_length():
    """length_m must be strictly positive, not zero."""
    with pytest.raises(ValueError, match="length_m must be positive"):
        compute_y_params(
            length_m=0.0,
            cable_params=realistic_params(),
            frequency_hz=np.array([1e6, 10e6]))


def test_rejects_negative_frequency():
    """All frequencies must be positive."""
    with pytest.raises(ValueError, match="frequencies must be positive"):
        compute_y_params(
            length_m=2.0,
            cable_params=realistic_params(),
            frequency_hz=np.array([1e6, -5e6, 10e6]))


def test_rejects_zero_frequency():
    """Zero frequency is not allowed (would cause division by zero)."""
    with pytest.raises(ValueError, match="frequencies must be positive"):
        compute_y_params(
            length_m=2.0,
            cable_params=realistic_params(),
            frequency_hz=np.array([0.0, 1e6, 10e6]))