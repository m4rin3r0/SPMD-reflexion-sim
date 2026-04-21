"""Analytic sanity tests for the AC solver core (independent of Topology)."""

from __future__ import annotations

import numpy as np
import pytest

# Wir importieren nur die niedrigsten Bausteine aus dem Solver,
# nicht run_ac_sim, weil wir die Topologie umgehen wollen.
from spmd_reflection.solver_ac import _stamp_two_port


# ---------------------------------------------------------------------------
# Hilfsfunktionen: minimale, analytisch nachvollziehbare Bausteine
# ---------------------------------------------------------------------------

C0 = 2.998e8  # Vakuumlichtgeschwindigkeit, m/s


def _lossless_line_y(length:float, z0_line:float, freq:float, vp_factor:float=0.66) -> np.ndarray:
    """
    Y-Parameter einer idealen, verlustfreien Leitung.
    Analytisch: gamma = j*beta, mit beta = 2*pi*f / (vp_factor*c0)
    Y-Matrix einer verlustfreien Leitung der Länge l mit Wellenwiderstand Z0_line:
        y11 = y22 =  cot(beta*l) / (j*Z0_line)
        y12 = y21 = -1 / (j*Z0_line*sin(beta*l))
    Bewusst NICHT _yparams_line aus dem Solver verwendet, weil wir den Solver testen wollen.
    """
    beta = 2 * np.pi * freq / (vp_factor * C0)
    bl = beta * length
    sin_bl = np.sin(bl)
    cos_bl = np.cos(bl)

    # Schutz gegen exakte Resonanz (sin(bl)=0). Mit krummer Länge bei sinnvollen
    # Frequenzen sollte das nicht passieren; wir fangen es trotzdem ab.
    if abs(sin_bl) < 1e-15:
        sin_bl = 1e-15

    y11 = cos_bl / (1j * z0_line * sin_bl)
    y12 = -1.0 / (1j * z0_line * sin_bl)
    return np.array([[y11, y12], [y12, y11]], dtype=complex)


def _solve_s11(y_network:np.ndarray, source_node:int, z0_ref:float) -> complex:
    """
    Norton-Quelle mit Quellleitwert 1/z0_ref am source_node, S11 aus v und i.
    Identische Konvention wie im Hauptsolver:
        a1 = v + i * z0_ref     (einlaufende Welle)
        b1 = v - i * z0_ref     (reflektierte Welle)
        S11 = b1 / a1
    """
    n = y_network.shape[0] # Anzahl der Knoten
    ysrc = 1.0 / z0_ref

    y_total = y_network.copy()
    y_total[source_node, source_node] += ysrc # Norton-Quele hat Parallel-impedanz von Z0

    i_vec = np.zeros(n, dtype=complex)
    i_vec[source_node] = ysrc  # Norton-Quellstrom = ysrc * v_open mit v_open = 1V

    v = np.linalg.solve(y_total, i_vec)
    v_port = v[source_node]
    # Strom, der wirklich ins Netzwerk fliesst (Quellstrom minus Strom durch Quellleitwert)
    i_port = i_vec[source_node] - ysrc * v_port

    a1 = v_port + i_port * z0_ref
    b1 = v_port - i_port * z0_ref
    return b1 / a1


def _build_one_segment_network(length:float, z0_line:float, term_y:complex, freq:float) -> np.ndarray:
    """
    Baut ein 2-Knoten-Netzwerk:
      Knoten 0 = Source-Port
      Knoten 1 = Leitungsende, mit Shunt-Admittanz term_y nach Masse
    """
    y = np.zeros((2, 2), dtype=complex)
    _stamp_two_port(y, _lossless_line_y(length, z0_line, freq), 0, 1)
    y[1, 1] += term_y
    return y


# ---------------------------------------------------------------------------
# Frequenzraster
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def freqs() -> np.ndarray:
    return np.linspace(1e6, 40e6, 401)


# ---------------------------------------------------------------------------
# Test 1: Verlustfreie, perfekt mit Z0 terminierte Leitung -> S11 = 0
# ---------------------------------------------------------------------------

def test_matched_lossless_line_has_zero_reflection(freqs):
    z0 = 100.0
    length = 5.0  # m, Wert ist egal, weil Reflexion exakt 0 ist
    term_y = 1.0 / z0  # perfekt angepasste Termination

    s11_db = np.empty(len(freqs))
    for idx, f in enumerate(freqs):
        y = _build_one_segment_network(length, z0, term_y, f)
        s11 = _solve_s11(y, source_node=0, z0_ref=z0)
        s11_db[idx] = 20 * np.log10(max(abs(s11), 1e-30))

    # Bei perfekter Anpassung erwarten wir numerisch -250 dB oder besser.
    assert np.all(s11_db < -200), f"max S11 = {s11_db.max():.1f} dB, sollte < -200 dB sein"


# ---------------------------------------------------------------------------
# Test 2: Verlustfreie offene Leitung -> |S11| = 1
# ---------------------------------------------------------------------------

def test_open_lossless_line_has_unity_reflection(freqs):
    z0 = 100.0
    length = 5.0
    term_y = 0.0  # offen

    s11_mag = np.empty(len(freqs))
    for idx, f in enumerate(freqs):
        y = _build_one_segment_network(length, z0, term_y, f)
        s11 = _solve_s11(y, source_node=0, z0_ref=z0)
        s11_mag[idx] = abs(s11)

    # |S11| sollte exakt 1 sein (verlustfrei + offen). Toleranz für Numerik.
    assert np.allclose(s11_mag, 1.0, atol=1e-9), \
        f"|S11| weicht ab: min={s11_mag.min():.6f}, max={s11_mag.max():.6f}"


# ---------------------------------------------------------------------------
# Test 3: Verlustfreie kurzgeschlossene Leitung -> |S11| = 1
# ---------------------------------------------------------------------------

def test_shorted_lossless_line_has_unity_reflection(freqs):
    z0 = 100.0
    length = 5.0
    term_y = 1e9  # quasi-Kurzschluss (sehr hoher Leitwert)

    s11_mag = np.empty(len(freqs))
    for idx, f in enumerate(freqs):
        y = _build_one_segment_network(length, z0, term_y, f)
        s11 = _solve_s11(y, source_node=0, z0_ref=z0)
        s11_mag[idx] = abs(s11)

    assert np.allclose(s11_mag, 1.0, atol=1e-6), \
        f"|S11| weicht ab: min={s11_mag.min():.6f}, max={s11_mag.max():.6f}"


# ---------------------------------------------------------------------------
# Test 4: Phasen-Konsistenz zwischen offener und kurzgeschlossener Leitung
# ---------------------------------------------------------------------------

def test_open_short_phase_difference(freqs):
    """
    Bei einer verlustfreien Leitung sollten offene und kurzgeschlossene
    Termination Reflektionen mit exakt 180 Grad Phasenversatz erzeugen.
    Prüfung: S11_open + S11_short ~ 0 (bei gleichem |S11|).
    """
    z0 = 100.0
    length = 5.0

    diffs = np.empty(len(freqs))
    for idx, f in enumerate(freqs):
        y_open = _build_one_segment_network(length, z0, 0.0, f)
        y_short = _build_one_segment_network(length, z0, 1e9, f)
        s_open = _solve_s11(y_open, 0, z0)
        s_short = _solve_s11(y_short, 0, z0)
        diffs[idx] = abs(s_open + s_short)

    assert np.all(diffs < 1e-5), f"max |S_open + S_short| = {diffs.max():.2e}, sollte ~0 sein"
