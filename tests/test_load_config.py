"""Tests for config.load."""

from pathlib import Path
from typing import Any
import pytest
import yaml

from spmd_reflection.config.load import load_config
from spmd_reflection.config.models import FrequencyGrid, SimConfig, TopologyConfig, TouchstonePaths


"""Table of content:
    CF1: valid config load
    CF2: touchstone paths in subdirectories are correctly resolved
    CF3: missing file → FileNotFoundError
    CF4: missing Top-Level-Section
    CF5: start_hz ≤ 0
    CF6: stop_hz ≤ start_hz
    CF7: n_points < 2
    CF8: missing required field in section
    CF9: drop_positions_m not a list
    CF10: touchstone file missing
    CF11: l_per_m ≤ 0
    CF12: rdc_per_m < 0
    CF13: Z₀ not plausible
"""

def _make_dummy_touchstone(path: Path) -> None:
    """Create a minimal valid-looking touchstone file (empty content suffices for path tests)."""
    path.write_text("# Hz S RI R 100\n")


def _valid_config_dict(tx_filename:str="tx.s2p", rx_filename:str="rx.s2p") -> dict[str,Any]:
    """Return a config dict with all required sections, ready for testing."""
    return {
        "frequency": {
            "start_hz": 300_000,
            "stop_hz": 40_000_000,
            "n_points": 1001,
        },
        "topology": {
            "drop_positions_m": [1.0, 3.0, 5.0],
            "bus_start_m": 0.0,
            "bus_end_m": 7.0,
            "tx_drop_index": 0,
            "termination_ohm": 100.0,
        },
        "paths": {
            "tx_touchstone": tx_filename,
            "rx_touchstone": rx_filename,
        },
        "cable": {
            "l_per_m": 413e-9,
            "c_per_m": 45e-12,
            "rdc_per_m": 0.19,
            "rskin_per_m": 5e-7,
        },
    }


def _write_config(tmp_path:Path, content:dict[str,Any]) -> Path:
    """Write a config dict to a YAML file in tmp_path. Also creates dummy touchstone files."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(content))
    # Also create the touchstone files referenced by the config, if any.
    if "paths" in content:
        if "tx_touchstone" in content["paths"]:
            _make_dummy_touchstone(tmp_path / content["paths"]["tx_touchstone"])
        if "rx_touchstone" in content["paths"]:
            _make_dummy_touchstone(tmp_path / content["paths"]["rx_touchstone"])
    return config_path


def test_loads_complete_valid_config(tmp_path):
    """A complete, valid config is loaded with all fields correctly mapped."""
    config_dict = _valid_config_dict()
    config_path = _write_config(tmp_path, config_dict)
    config = load_config(config_path)

    assert config.frequency == FrequencyGrid(
        start_hz=300_000.0,
        stop_hz=40_000_000.0,
        n_points=1001)
    
    assert config.topology == TopologyConfig(
        drop_positions_m=(1.0, 3.0, 5.0),
        bus_start_m=0.0,
        bus_end_m=7.0,
        tx_drop_index=0,
        termination_ohm=100.0)

    assert config.paths.tx == (tmp_path / "tx.s2p").resolve()
    assert config.paths.rx == (tmp_path / "rx.s2p").resolve()

    assert config.cable.l_per_m == pytest.approx(413e-9)
    assert config.cable.c_per_m == pytest.approx(45e-12)
    assert config.cable.rdc_per_m == pytest.approx(0.19)
    assert config.cable.rskin_per_m == pytest.approx(5e-7)


def test_resolves_paths_in_subdirectory(tmp_path):
    """Touchstone paths in subdirectories are correctly resolved."""
    # Create subdirectory and dummy touchstones inside it.
    subdir = tmp_path / "measurements"
    subdir.mkdir()
    _make_dummy_touchstone(subdir / "tx.s2p")
    _make_dummy_touchstone(subdir / "rx.s2p")

    # Config references them with subdirectory prefix.
    config_dict = _valid_config_dict(
        tx_filename="measurements/tx.s2p",
        rx_filename="measurements/rx.s2p")

    # Write config (but disable auto-creation of touchstones, since we made them ourselves).
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_dict))
    config = load_config(config_path)

    assert config.paths.tx == (subdir / "tx.s2p").resolve()
    assert config.paths.rx == (subdir / "rx.s2p").resolve()


def test_raises_file_not_found_for_missing_config(tmp_path):
    """A non-existent config path raises FileNotFoundError."""
    missing_path = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(missing_path)


def test_rejects_missing_top_level_section(tmp_path):
    """A config missing one of the four top-level sections is rejected."""
    config_dict = _valid_config_dict()
    del config_dict["frequency"]
    config_path = _write_config(tmp_path, config_dict)
    with pytest.raises(ValueError, match="missing required sections"):
        load_config(config_path)


def test_rejects_non_positive_start_hz(tmp_path):
    """frequency.start_hz must be positive."""
    config_dict = _valid_config_dict()
    config_dict["frequency"]["start_hz"] = 0.0
    config_path = _write_config(tmp_path, config_dict)
    with pytest.raises(ValueError, match="start_hz must be positive"):
        load_config(config_path)


def test_rejects_stop_hz_not_greater_than_start_hz(tmp_path):
    """frequency.stop_hz must be strictly greater than start_hz."""
    config_dict = _valid_config_dict()
    config_dict["frequency"]["stop_hz"] = config_dict["frequency"]["start_hz"]
    config_path = _write_config(tmp_path, config_dict)
    with pytest.raises(ValueError, match="must be greater than"):
        load_config(config_path)


def test_rejects_n_points_too_small(tmp_path):
    """frequency.n_points must be at least 2."""
    config_dict = _valid_config_dict()
    config_dict["frequency"]["n_points"] = 1
    config_path = _write_config(tmp_path, config_dict)
    with pytest.raises(ValueError, match="n_points must be at least 2"):
        load_config(config_path)


def test_rejects_missing_field_within_section(tmp_path):
    """A required field missing within a section is rejected with a clear message."""
    config_dict = _valid_config_dict()
    del config_dict["frequency"]["n_points"]
    config_path = _write_config(tmp_path, config_dict)
    with pytest.raises(ValueError, match="frequency section missing required keys"):
        load_config(config_path)


def test_rejects_drop_positions_not_a_list(tmp_path):
    """topology.drop_positions_m must be a list."""
    config_dict = _valid_config_dict()
    config_dict["topology"]["drop_positions_m"] = 3.0
    config_path = _write_config(tmp_path, config_dict)
    with pytest.raises(ValueError, match="drop_positions_m must be a list"):
        load_config(config_path)


def test_rejects_missing_touchstone_file(tmp_path):
    """A touchstone file referenced in the config must exist on disk."""
    config_dict = _valid_config_dict(tx_filename="nonexistent.s2p")
    # Write config but NOT the tx touchstone file.
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_dict))
    # We still need a valid rx file, since the rx check runs only if tx passes.
    _make_dummy_touchstone(tmp_path / "rx.s2p")
    with pytest.raises(ValueError, match="tx_touchstone file not found"):
        load_config(config_path)


def test_rejects_non_positive_l_per_m(tmp_path):
    """cable.l_per_m must be strictly positive."""
    config_dict = _valid_config_dict()
    config_dict["cable"]["l_per_m"] = 0.0
    config_path = _write_config(tmp_path, config_dict)
    with pytest.raises(ValueError, match="l_per_m must be positive"):
        load_config(config_path)


def test_rejects_negative_rdc_per_m(tmp_path):
    """cable.rdc_per_m must be non-negative (zero is allowed for lossless lines)."""
    config_dict = _valid_config_dict()
    config_dict["cable"]["rdc_per_m"] = -0.1
    config_path = _write_config(tmp_path, config_dict)
    with pytest.raises(ValueError, match="rdc_per_m must be non-negative"):
        load_config(config_path)


def test_rejects_implausible_z0(tmp_path):
    """Cable parameters that yield Z₀ far from 100 Ω are rejected as likely unit errors."""
    config_dict = _valid_config_dict()
    # Make L'/C' yield Z₀ ≈ 1000 Ω — clearly wrong for a 100 Ω bus.
    config_dict["cable"]["l_per_m"] = 1e-6
    config_dict["cable"]["c_per_m"] = 1e-12
    config_path = _write_config(tmp_path, config_dict)
    with pytest.raises(ValueError, match="Z.*at 10 MHz"):
        load_config(config_path)