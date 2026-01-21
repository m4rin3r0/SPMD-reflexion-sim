"""Plotting helpers for RL/IL results."""

from __future__ import annotations

from typing import Optional
import matplotlib.pyplot as plt

from .solver_ac import SimulationResults


def plot_results(results: SimulationResults, output_path: Optional[str] = None) -> None:
    freq_mhz = results.frequency / 1e6

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(freq_mhz, results.s11_db, label="S11 (RL)")
    ax1.set_ylabel("RL (dB)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    for idx in range(results.gain_db.shape[1]):
        ax2.plot(freq_mhz, results.gain_db[:, idx], label=f"Node {idx+1}")

    ax2.set_xlabel("Frequency (MHz)")
    ax2.set_ylabel("IL (dB)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(ncol=2, fontsize=8)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
    else:
        plt.show()
