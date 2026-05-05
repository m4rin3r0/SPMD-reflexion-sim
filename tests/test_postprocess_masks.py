"""Tests for postprocess.masks."""

import numpy as np
import pytest

from spmd_reflection.postprocess.masks import check_mpi_il, check_mpi_rl, mpi_il_mask, mpi_rl_mask, tci_il_mask, tci_rl_mask, mixing_segment_il_mask, check_tci_il, check_tci_rl


def test_tci_il_mask_flat_region():
    """IL_TCI-Mask is 0.16 dB at 1.3-10 MHz"""
    freqs = np.array([1.3e6, 5e6, 9.9e6])
    limit = tci_il_mask(freqs)
    # Standard says 0.16, but floor is 0.2 dB.
    assert np.allclose(limit, 0.2, atol=1e-10)


def test_tci_il_mask_floor_applied():
    """IL_TCI-Mask never goes below 0.2 dB."""
    freqs = np.linspace(0.3e6, 40e6, 1000)
    limit = tci_il_mask(freqs)
    valid = ~np.isnan(limit)
    assert np.all(limit[valid] >= 0.2 - 1e-10)


def test_tci_il_mask_nan_outside_range():
    """frequencies outside 0.3-40 MHz are NaN"""
    freqs = np.array([0.1e6, 0.3e6, 40e6, 41e6])
    limit = tci_il_mask(freqs)
    assert np.isnan(limit[0])   # 0.1 MHz: too low
    assert not np.isnan(limit[1])  # 0.3 MHz: lower limit, valid
    assert not np.isnan(limit[2])  # 40 MHz: upper limit, valid
    assert np.isnan(limit[3])   # 41 MHz: too high


def test_tci_rl_mask_at_boundary():
    """RL_TCI-Mask evalueates both formulas correctly at f = 1.7 MHz"""
    # lower formula (f < 1.7 MHz): -0.3 + 13*f
    f_below = np.array([1.6999e6])
    limit_below = tci_rl_mask(f_below)
    expected_below = -0.3 + 13.0 * 1.6999
    assert np.allclose(limit_below, expected_below, atol=1e-3)
    # upper formula (f >= 1.7 MHz).
    f_at = np.array([1.7e6])
    limit_at = tci_rl_mask(f_at)
    f = 1.7
    expected_at = (-38.55
                - 50.28 * np.log10(f)
                - (3.16 / f)
                + 69.31 * np.sqrt(f)
                - 10.19 * f
                + 0.0636 * f**2)
    assert np.allclose(limit_at, expected_at, atol=1e-3)


def test_tci_rl_mask_low_frequency():
    """RL_TCI-Mask at 0.3 MHz: -0.3 + 13*0.3 = 3.6 dB."""
    freqs = np.array([0.3e6])
    limit = tci_rl_mask(freqs)
    expected = -0.3 + 13.0 * 0.3
    assert np.allclose(limit, expected, atol=1e-10)


def test_mixing_segment_il_mask_at_boundary():
    """Mixing Segment IL-Mask is steady at edge f = 1.5 MHz."""
    f_below = np.array([1.4999e6])
    f_above = np.array([1.5001e6])
    limit_below = mixing_segment_il_mask(f_below)
    limit_above = mixing_segment_il_mask(f_above)
    assert np.abs(limit_below[0] - limit_above[0]) < 0.1


def test_mixing_segment_il_mask_low_frequency():
    """Mixing Segment IL at 0.3 MHz: 35 - 14.54*0.3 = 30.638 dB."""
    freqs = np.array([0.3e6])
    limit = mixing_segment_il_mask(freqs)
    expected = 35.0 - 14.54 * 0.3
    assert np.allclose(limit, expected, atol=1e-10)


def test_check_tci_il_conformant():
    """Values below mask -> True"""
    freqs = np.array([5e6])   # 5 MHz -> limit = 0.2 dB
    il_db = np.array([0.1])   # lower limit -> compliant
    result = check_tci_il(freqs, il_db)
    assert result[0] == True


def test_check_tci_il_nonconformant():
    """Values above mask -> False"""
    freqs = np.array([5e6])   # 5 MHz -> limit = 0.2 dB
    il_db = np.array([0.5])   # above limit -> not compliant
    result = check_tci_il(freqs, il_db)
    assert result[0] == False


def test_check_tci_il_out_of_range_is_conformant():
    """Frequencies outside the range are compliant"""
    freqs = np.array([0.1e6])   # outside 0.3-40 MHz
    il_db = np.array([999.0])   # arbitrary value
    result = check_tci_il(freqs, il_db)
    assert result[0] == True


def test_check_tci_il_2d_input():
    """check_tci_il works with 2D-Input (n_freq, n_drops)."""
    freqs = np.array([5e6, 5e6])
    il_db = np.array([[0.1, 0.5],   # frequency 0: Drop 0 compliant, Drop 1 not
                      [0.1, 0.1]])  # frequency 1: both compliant
    result = check_tci_il(freqs, il_db)
    assert result.shape == (2, 2)
    assert result[0, 0] == True
    assert result[0, 1] == False
    assert result[1, 0] == True
    assert result[1, 1] == True


def test_check_tci_rl_conformant():
    """Values above the rl-mask -> True"""
    freqs = np.array([0.3e6])   # 0.3 MHz -> Limit = 3.6 dB
    rl_db = np.array([10.0])    # above Limit -> compliant
    result = check_tci_rl(freqs, rl_db)
    assert result[0] == True


def test_check_tci_rl_nonconformant():
    """Values below the rl-mask -> False."""
    freqs = np.array([0.3e6])   # 0.3 MHz -> Limit = 3.6 dB
    rl_db = np.array([1.0])     # under Limit -> not compliant
    result = check_tci_rl(freqs, rl_db)
    assert result[0] == False


def test_mpi_rl_mask_at_low_frequency():
    """mpi_rl_mask returns plausible value at 0.3 MHz"""
    freqs = np.array([0.3e6])
    limit = mpi_rl_mask(freqs, n_unit=16)
    # calculation by hand at f = 0.3 MHz, n_unit = 16:
    f = 0.3
    expected = -10.0 * np.log10(
        (10000.0 + (40.194 * f)**2 / 16)
        / (10000.0 + (2010.0 * f / 16)**2)
        + f**2.5 / 480000.0)
    assert np.allclose(limit, expected, atol=1e-6)


def test_mpi_rl_mask_nan_outside_range():
    """frequencies outside 0.3-40 MHz are NaN"""
    freqs = np.array([0.1e6, 0.3e6, 40e6, 41e6])
    limit = mpi_rl_mask(freqs)
    assert np.isnan(limit[0])
    assert not np.isnan(limit[1])
    assert not np.isnan(limit[2])
    assert np.isnan(limit[3])


def test_mpi_il_mask_floor_applied():
    """mpi_il_mask never goes below 0.2 dB"""
    freqs = np.linspace(0.3e6, 40e6, 1000)
    limit = mpi_il_mask(freqs, n_unit=16)
    valid = ~np.isnan(limit)
    assert np.all(limit[valid] >= 0.2 - 1e-10)


def test_mpi_il_mask_above_5mhz_equals_tci_il_mask():
    """mpi_il_mask is identical with tci_il_mask for f >= 5 MHz"""
    freqs = np.linspace(5e6, 40e6, 100)
    limit_mpi = mpi_il_mask(freqs, n_unit=16)
    limit_tci = tci_il_mask(freqs)
    assert np.allclose(limit_mpi, limit_tci, atol=1e-10, equal_nan=True)


def test_check_mpi_rl_conformant():
    """Value above MPI-RL-Mask -> True"""
    freqs = np.array([1e6])
    limit = mpi_rl_mask(freqs, n_unit=16)
    rl_db = limit + 10.0
    result = check_mpi_rl(freqs, rl_db, n_unit=16)
    assert result[0] == True


def test_check_mpi_rl_nonconformant():
    """Value below MPI-RL-Mask -> False"""
    freqs = np.array([1e6])
    limit = mpi_rl_mask(freqs, n_unit=16)
    rl_db = limit - 10.0
    result = check_mpi_rl(freqs, rl_db, n_unit=16)
    assert result[0] == False


def test_check_mpi_il_conformant():
    """Value below MPI-IL-Mask -> True"""
    freqs = np.array([1e6])
    limit = mpi_il_mask(freqs, n_unit=16)
    il_db = limit - 1.0
    result = check_mpi_il(freqs, il_db, n_unit=16)
    assert result[0] == True


def test_check_mpi_il_nonconformant():
    """Value above MPI-IL-Mask -> False"""
    freqs = np.array([1e6])
    limit = mpi_il_mask(freqs, n_unit=16)
    il_db = limit + 1.0
    result = check_mpi_il(freqs, il_db, n_unit=16)
    assert result[0] == False