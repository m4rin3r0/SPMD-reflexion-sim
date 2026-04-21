"""Plotting helpers for RL/IL results."""

from __future__ import annotations

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

from .solver_ac import SimulationResults


def _return_loss_compliance_limit(frequency_hz:np.ndarray, n_unit:int) -> np.ndarray:
    freq_mhz = frequency_hz / 1e6
    limit_db = np.full_like(freq_mhz, np.nan, dtype=float)

    low_band = (freq_mhz >= 0.3) & (freq_mhz <= 18.0)
    high_band = (freq_mhz >= 18.0) & (freq_mhz <= 40.0)

    if np.any(low_band):
        f = freq_mhz[low_band]
        limit_db[low_band] = 10.0 * np.log10((10000 + ((40.194 * f) ** 2 / n_unit)) / (10000 + ((2010 * f) / n_unit) ** 2) + (f ** 2.5) / 480000.0)

    if np.any(high_band):
        f = freq_mhz[high_band]
        limit_db[high_band] = 10.0 * np.log10((10000 + ((40.192 * f) ** 2 / n_unit)) / (10000 + ((2010 * f) / n_unit) ** 2) + (f ** 5.0) / 6500000000.0)

    return limit_db


def plot_results(results:SimulationResults, output_path:Optional[str] = None, rl_n_unit:Optional[int] = None) -> None:
    freq_mhz = results.frequency / 1e6
    compliance_rl_db = None
    if rl_n_unit is not None:
        compliance_rl_db = _return_loss_compliance_limit(results.frequency, rl_n_unit)

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    ax1, ax2, ax3, ax4 = axes

    ax1.plot(freq_mhz, results.s11_db, label="Bus S11 (RL)")
    if compliance_rl_db is not None:
        ax1.plot(freq_mhz, compliance_rl_db, color="black", linestyle=":", linewidth=1.5, label=f"RL limit (N_UNIT={rl_n_unit})")
    ax1.set_ylabel("RL (dB)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    for idx in range(results.s21_db.shape[1]):
        if idx == results.tx_node_index:
            continue
        ax2.plot(freq_mhz, results.s21_db[:, idx], label=f"Node {idx+1}")
    ax2.set_ylabel("Bus IL (dB)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(ncol=2, fontsize=8)

    for idx, node_index in enumerate(results.drop_node_indices):
        ax3.plot(freq_mhz, results.drop_rl_left_db[:, idx], label=f"Drop {node_index + 1} left")
        ax3.plot(freq_mhz, results.drop_rl_right_db[:, idx], linestyle="--", label=f"Drop {node_index + 1} right")
    if compliance_rl_db is not None:
        ax3.plot(freq_mhz, compliance_rl_db, color="black", linestyle=":", linewidth=1.5, label=f"RL limit (N_UNIT={rl_n_unit})")
    ax3.set_ylabel("Drop RL (dB)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(ncol=2, fontsize=8)

    for idx, node_index in enumerate(results.drop_node_indices):
        ax4.plot(freq_mhz, results.drop_il_lr_db[:, idx], label=f"Drop {node_index + 1} L→R")
        ax4.plot(freq_mhz, results.drop_il_rl_db[:, idx], linestyle="--", label=f"Drop {node_index + 1} R→L")
    ax4.set_xlabel("Frequency (MHz)")
    ax4.set_ylabel("Drop IL (dB)")
    ax4.grid(True, alpha=0.3)
    ax4.legend(ncol=2, fontsize=8)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
    else:
        plt.show()
