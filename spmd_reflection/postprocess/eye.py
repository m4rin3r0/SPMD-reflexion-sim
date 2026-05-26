"""Generate eye diagrams from simulation results.

Computes the time-domain response of the bus by inverse FFT of the
frequency-domain transfer function H(ω) = V_RX / V_TX, convolves it
with a synthetic PAM3 signal, and segments the result for eye diagram
visualization.

10BASE-T1M uses PAM3 signaling at 7.5 MBaud symbol rate.

Usage:
    from spmd_reflection.postprocess.eye import compute_eye_diagram, plot_eye_diagram

    eye = compute_eye_diagram(
        tx_voltage=solver_results.tx_phy_voltage,
        rx_voltage=solver_results.rx_phy_voltages[:, drop_index],
        frequency_hz=solver_results.frequency_hz,
    )
    fig = plot_eye_diagram(eye)
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


# 10BASE-T1M symbol rate
SYMBOL_RATE_BAUD = 12.5e6

# Default oversampling factor (samples per symbol in the time domain).
DEFAULT_SAMPLES_PER_SYMBOL = 16


@dataclass(frozen=True)
class EyeDiagramData:
    """Data for rendering an eye diagram.

    Attributes:
        time_s: 1D array of time values for one eye window (2 symbol periods).
        segments: 2D array of shape (n_segments, segment_length).
            Each row is one overlaid trace in the eye diagram.
        symbol_period_s: Duration of one symbol in seconds.
    """
    time_s: np.ndarray
    segments: np.ndarray
    symbol_period_s: float


def _generate_baseband_signal(n_symbols:int, samples_per_symbol:int, rng:np.random.Generator) -> np.ndarray:
    """Generate a random binary baseband signal.

    Each symbol is one of {-1, +1}, held for samples_per_symbol samples
    (rectangular pulse shaping).    

    Args:
        n_symbols: Number of symbols to generate.
        samples_per_symbol: Oversampling factor.
        rng: NumPy random generator for reproducibility.

    Returns:
        1D array of length n_symbols * samples_per_symbol.
    """
    symbols = rng.choice([-1.0, 1.0], size=n_symbols)
    return np.repeat(symbols, samples_per_symbol)


def _compute_transfer_function(tx_voltage:np.ndarray, rx_voltage:np.ndarray) -> np.ndarray:
    """Compute H(ω) = V_RX / V_TX with protection against division by zero.

    Args:
        tx_voltage: Complex TX-PHY voltage, shape (n_freq,).
        rx_voltage: Complex RX-PHY voltage, shape (n_freq,).

    Returns:
        Complex transfer function H(ω), shape (n_freq,).
    """
    eps = np.finfo(float).eps
    return rx_voltage / np.where(np.abs(tx_voltage) > eps, tx_voltage, eps)


def _transfer_function_to_impulse_response(h_freq:np.ndarray, n_freq_original:int, fs:float) -> np.ndarray:
    """Convert one-sided H(ω) to a real-valued impulse response h(t).

    Extends H(ω) to a two-sided spectrum suitable for irfft. The DC
    component H(0) is set to H[0] (extrapolated from lowest measured
    frequency). Zero-padding is applied to achieve the target sample rate.

    Args:
        h_freq: One-sided transfer function, shape (n_freq,).
        n_freq_original: Number of original frequency points.
        fs: Target sample rate in Hz.

    Returns:
        Real-valued impulse response h(t).
    """
    df = fs / (2 * len(h_freq))
    n_ifft = int(fs / df)
    n_ifft = max(n_ifft, 2 * len(h_freq))
    h_padded = np.zeros(n_ifft // 2 + 1, dtype=complex)
    h_padded[0] = np.real(h_freq[0])
    h_padded[1:len(h_freq) + 1] = h_freq
    h_t = np.fft.irfft(h_padded, n=n_ifft)
    return h_t


def compute_eye_diagram(tx_voltage:np.ndarray, rx_voltage:np.ndarray, frequency_hz:np.ndarray, n_symbols:int=2000, samples_per_symbol:int=DEFAULT_SAMPLES_PER_SYMBOL, symbol_rate:float=SYMBOL_RATE_BAUD, seed:int=42, snr_db:float=20.0, jitter_fraction:float=0.02) -> EyeDiagramData:
    """Compute eye diagram data from TX and RX voltages.

    Steps:
        1. Compute transfer function H(ω) = V_RX / V_TX.
        2. Convert H(ω) to impulse response h(t) via IFFT.
        3. Generate a random PAM3 signal.
        4. Convolve PAM3 signal with h(t) to get the received signal.
        5. Segment the received signal into 2-symbol-period windows.

    Args:
        tx_voltage: Complex TX-PHY voltage, shape (n_freq,).
        rx_voltage: Complex RX-PHY voltage, shape (n_freq,).
        frequency_hz: Frequency grid in Hz, shape (n_freq,).
        n_symbols: Number of PAM3 symbols to simulate.
        samples_per_symbol: Oversampling factor for time-domain signal.
        symbol_rate: Symbol rate in Baud (default 7.5 MBaud for 10BASE-T1M).
        seed: Random seed for reproducible PAM3 sequence.

    Returns:
        EyeDiagramData with time axis and overlaid signal segments.
    """
    symbol_period = 1.0 / symbol_rate
    fs = samples_per_symbol * symbol_rate
    h_freq = _compute_transfer_function(tx_voltage, rx_voltage)
    h_t = _transfer_function_to_impulse_response(h_freq, len(frequency_hz), fs)
    rng = np.random.default_rng(seed)
    tx_signal = _generate_baseband_signal(n_symbols, samples_per_symbol, rng)
    rx_signal = np.convolve(tx_signal, h_t, mode="full")
    # Adding noise
    signal_power = np.mean(rx_signal**2)
    noise_power = signal_power / (10**(snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), len(rx_signal))
    rx_signal = rx_signal + noise
    segment_length = 2 * samples_per_symbol
    start_offset = 10 * samples_per_symbol
    # Jitter:
    jitter_samples = int(jitter_fraction * samples_per_symbol)
    segments = []
    for i in range((len(rx_signal) - start_offset - segment_length) // samples_per_symbol):
        jitter = rng.integers(-jitter_samples, jitter_samples + 1) if jitter_samples > 0 else 0
        start = start_offset + i * samples_per_symbol + jitter
        end = start + segment_length
        if start < 0 or end > len(rx_signal):
            continue
        segments.append(rx_signal[start:end])
    segments = np.array(segments)
    time_s = np.linspace(0, 2 * symbol_period, segment_length, endpoint=False)
    return EyeDiagramData(time_s=time_s, segments=segments, symbol_period_s=symbol_period)


def plot_eye_diagram(eye:EyeDiagramData, ax=None, title:str="Eye Diagram", color:str="blue", alpha:float=0.03):
    """Plot an eye diagram from precomputed data.

    Args:
        eye: EyeDiagramData from compute_eye_diagram.
        ax: Optional matplotlib axes. If None, a new figure is created.
        title: Plot title.
        color: Line color for the traces.
        alpha: Transparency of individual traces.

    Returns:
        matplotlib Figure object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()
    time_ns = eye.time_s * 1e9
    for segment in eye.segments:
        ax.plot(time_ns, np.real(segment), color=color, alpha=alpha, linewidth=0.5)
    ax.set_xlabel("Zeit (ns)")
    ax.set_ylabel("Amplitude (V)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    # Markierung der Symbolgrenzen.
    t_symbol_ns = eye.symbol_period_s * 1e9
    ax.axvline(x=t_symbol_ns, color="red", linestyle="--", alpha=0.5, label="Symbolgrenze")
    ax.legend()
    return fig