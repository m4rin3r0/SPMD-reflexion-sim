"""Plotting helpers for RL/IL results and measurement comparisons."""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np

from .measurement_import import ImportedMeasurementArchive
from .solver_ac import SimulationResults


def plot_results(results: SimulationResults, output_path: Optional[str] = None) -> None:
    freq_mhz = results.frequency / 1e6

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(freq_mhz, results.s11_db, label="S11 (RL)")
    ax1.set_ylabel("RL (dB)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    for idx in range(results.s21_db.shape[1]):
        if idx == results.tx_node_index:
            continue
        ax2.plot(freq_mhz, results.s21_db[:, idx], label=f"Node {idx+1}")

    ax2.set_xlabel("Frequency (MHz)")
    ax2.set_ylabel("IL (dB)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(ncol=2, fontsize=8)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
    else:
        plt.show()


def plot_repeated_measurement_traces(
    archive: ImportedMeasurementArchive,
    output_path: Optional[str] = None,
) -> None:
    repeated = {
        name: traces
        for name, traces in archive.traces.items()
        if len(traces) > 1
    }
    if not repeated:
        raise ValueError("No repeated S-parameter traces found in the measurement archive.")

    names = sorted(repeated)
    ncols = 2
    nrows = ceil(len(names) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.2 * nrows), sharex=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for idx, trace_name in enumerate(names):
        ax = axes[idx // ncols, idx % ncols]
        for trace in repeated[trace_name]:
            freq_mhz = trace.frequency / 1e6
            mag_db = 20.0 * np.log10(np.maximum(np.abs(trace.values), 1e-30))
            label = f"M{trace.measurement_id}: {Path(trace.source_file).name}"
            ax.plot(freq_mhz, mag_db, label=label)

        ax.set_title(trace_name)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for idx in range(len(names), nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle(f"Repeated single-ended measurements ({archive.mapping_name} mapping)")
    fig.tight_layout()

    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
    else:
        plt.show()
