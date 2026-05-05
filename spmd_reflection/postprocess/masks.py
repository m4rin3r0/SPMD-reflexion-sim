"""IEEE 802.3da Clause 188 mask functions for 10BASE-T1M multi-drop bus.

Each mask function returns the limit values as a function of frequency.
Separate check functions return boolean arrays indicating conformance.

All public functions accept frequency in Hz and return dB values.
Internally, IEEE formulas use frequency in MHz.
"""

from __future__ import annotations
import numpy as np


def tci_il_mask(frequency_hz:np.ndarray) -> np.ndarray:
    """TCI insertion loss mask per IEEE 802.3da Eq. 188-5.

    IL(f) ≤ returned value for conformance.

    Args:
        frequency_hz: 1D array of frequencies in Hz. Valid range: 0.3-40 MHz.

    Returns:
        Array of IL limit values in dB. Points outside 0.3-40 MHz are NaN.
        Note: whenever the formula gives < 0.2 dB, the limit is 0.2 dB.
    """
    f = frequency_hz / 1e6
    result = np.full_like(f, np.nan)
    mask_1 = (f >= 0.3) & (f < 1.3)
    mask_2 = (f >= 1.3) & (f < 10.0)
    mask_3 = (f >= 10.0) & (f < 24.0)
    mask_4 = (f >= 24.0) & (f <= 40.0)
    result[mask_1] = 3.0 - 2.84 * (f[mask_1] - 0.3)
    result[mask_2] = 0.16
    result[mask_3] = (-0.454
                      + (0.22 / f[mask_3])
                      + 0.63 * np.sqrt(f[mask_3])
                      - 0.18 * f[mask_3]
                      + 0.004 * f[mask_3]**2)
    result[mask_4] = 0.145 * f[mask_4] - 2.86
    # Per standard: whenever result < 0.2 dB, limit reverts to 0.2 dB.
    valid = ~np.isnan(result)
    result[valid] = np.maximum(result[valid], 0.2)
    return result


def mpi_il_mask(frequency_hz:np.ndarray, n_unit:int=16) -> np.ndarray:
    """TCI IL mask for MPIs per IEEE 802.3da Eq. 189-2 (f < 5 MHz) and Eq. 188-5 (f >= 5 MHz)

    Args:
        frequency_hz: 1D array of frequencies in Hz. Valid range: 0.3-40 MHz.
        n_unit: MPD unit load value. Default 16 for MPSEs.

    Returns:
        Array of IL limit values in dB. Points outside 0.3-40 MHz are NaN.
    """
    f = frequency_hz / 1e6
    result = np.full_like(f, np.nan)
    mask_low  = (f >= 0.3) & (f < 5.0)
    mask_high = (f >= 5.0) & (f <= 40.0)
    # Below 5 MHz: Eq. 189-2, depends on RL mask.
    rl_limit = mpi_rl_mask(frequency_hz[mask_low], n_unit=n_unit)
    result[mask_low] = (0.16 - 10.0 * np.log10(1.0 - 10.0**(-rl_limit / 10.0)))
    # Above 5 MHz: Eq. 188-5 applies.
    result[mask_high] = tci_il_mask(frequency_hz[mask_high])
    # Floor at 0.2 dB per standard.
    valid = ~np.isnan(result)
    result[valid] = np.maximum(result[valid], 0.2)
    return result


def tci_rl_mask(frequency_hz:np.ndarray) -> np.ndarray:
    """TCI return loss mask per IEEE 802.3da Eq. 188-6.
    RL(f) ≥ returned value for conformance.
    Args:
        frequency_hz: 1D array of frequencies in Hz. Valid range: 0.3-40 MHz.
    Returns:
        Array of RL limit values in dB. Points outside 0.3-40 MHz are NaN.
    """
    f = frequency_hz / 1e6
    result = np.full_like(f, np.nan)
    mask_1 = (f >= 0.3) & (f < 1.7)
    mask_2 = (f >= 1.7) & (f <= 40.0)
    result[mask_1] = -0.3 + 13.0 * f[mask_1]
    result[mask_2] = (-38.55
                      - 50.28 * np.log10(f[mask_2])
                      - (3.16 / f[mask_2])
                      + 69.31 * np.sqrt(f[mask_2])
                      - 10.19 * f[mask_2]
                      + 0.0636 * f[mask_2]**2)
    return result


def mpi_rl_mask(frequency_hz:np.ndarray, n_unit:int=16) -> np.ndarray:
    """TCI RL mask for MPIs per IEEE 802.3da Eq. 189-1.

    Args:
        frequency_hz: 1D array of frequencies in Hz. Valid range: 0.3-40 MHz.
        n_unit: MPD unit load value. Default 16 for MPSEs.
            NOTE: The correct value for a specific configuration requires
            further investigation of Clause 189.

    Returns:
        Array of RL limit values in dB. Points outside 0.3-40 MHz are NaN.
    """
    f = frequency_hz / 1e6
    result = np.full_like(f, np.nan)
    mask_1 = (f >= 0.3) & (f <= 18.0)
    mask_2 = (f > 18.0) & (f <= 40.0)
    result[mask_1] = -10.0 * np.log10(
        (10000.0 + (40.194 * f[mask_1])**2 / n_unit)
        / (10000.0 + (2010.0 * f[mask_1] / n_unit)**2)
        + f[mask_1]**2.5 / 480000.0)
    result[mask_2] = -10.0 * np.log10(
        (10000.0 + (40.192 * f[mask_2])**2 / n_unit)
        / (10000.0 + (2010.0 * f[mask_2] / n_unit)**2)
        + f[mask_2]**5 / 650000000.0)
    return result


def mixing_segment_il_mask(frequency_hz:np.ndarray) -> np.ndarray:
    """Mixing segment insertion loss mask per IEEE 802.3da Eq. 188-3.

    IL(f) ≤ returned value for conformance.

    Args:
        frequency_hz: 1D array of frequencies in Hz. Valid range: 0.3-40 MHz.

    Returns:
        Array of IL limit values in dB. Points outside 0.3-40 MHz are NaN.
    """
    f = frequency_hz / 1e6
    result = np.full_like(f, np.nan)
    mask_1 = (f >= 0.3) & (f < 1.5)
    mask_2 = (f >= 1.5) & (f <= 40.0)
    result[mask_1] = 35.0 - 14.54 * f[mask_1]
    result[mask_2] = (-27.0
                      - 53.0 * np.log10(f[mask_2])
                      - (1.7 / f[mask_2])
                      + 52.0 * np.sqrt(f[mask_2])
                      - 8.9 * f[mask_2]
                      + 0.163 * f[mask_2]**2)
    return result


def check_tci_il(frequency_hz:np.ndarray, il_tci_db:np.ndarray) -> np.ndarray:
    """Check TCI IL conformance against Eq. 188-5 mask.

    Args:
        frequency_hz: 1D array of frequencies in Hz.
        il_tci_db: IL values in dB. Shape (n_freq,) or (n_freq, n_drops).

    Returns:
        Boolean array, same shape as il_tci_db. True = conformant.
        NaN mask points are treated as conformant (out of specified range).
    """
    limit = tci_il_mask(frequency_hz)
    if il_tci_db.ndim == 2:
        limit = limit[:, np.newaxis]
    nan_mask = np.isnan(limit)
    conformant = il_tci_db <= limit
    conformant[np.broadcast_to(nan_mask, conformant.shape)] = True
    return conformant


def check_tci_rl(frequency_hz:np.ndarray, rl_tci_db:np.ndarray) -> np.ndarray:
    """Check TCI RL conformance against Eq. 188-6 mask.

    Args:
        frequency_hz: 1D array of frequencies in Hz.
        rl_tci_db: RL values in dB. Shape (n_freq,) or (n_freq, n_drops).

    Returns:
        Boolean array, same shape as rl_tci_db. True = conformant.
        NaN mask points are treated as conformant (out of specified range).
    """
    limit = tci_rl_mask(frequency_hz)
    if rl_tci_db.ndim == 2:
        limit = limit[:, np.newaxis]
    nan_mask = np.isnan(limit)
    conformant = rl_tci_db >= limit
    conformant[np.broadcast_to(nan_mask, conformant.shape)] = True
    return conformant


def check_mpi_rl(frequency_hz:np.ndarray, rl_tci_db:np.ndarray, n_unit:int=16) -> np.ndarray:
    """Check MPI RL conformance against Eq. 189-1."""
    limit = mpi_rl_mask(frequency_hz, n_unit=n_unit)
    if rl_tci_db.ndim == 2:
        limit = limit[:, np.newaxis]
    nan_mask = np.isnan(limit)
    conformant = rl_tci_db >= limit
    conformant[np.broadcast_to(nan_mask, conformant.shape)] = True
    return conformant


def check_mpi_il(frequency_hz:np.ndarray, il_tci_db:np.ndarray, n_unit:int=16) -> np.ndarray:
    """Check MPI IL conformance against Eq. 189-2 / 188-5."""
    limit = mpi_il_mask(frequency_hz, n_unit=n_unit)
    if il_tci_db.ndim == 2:
        limit = limit[:, np.newaxis]
    nan_mask = np.isnan(limit)
    conformant = il_tci_db <= limit
    conformant[np.broadcast_to(nan_mask, conformant.shape)] = True
    return conformant