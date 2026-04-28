from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml

from spmd_reflection.cable.cable_params import CableParams
from spmd_reflection.config.models import FrequencyGrid, SimConfig, TopologyConfig, TouchstonePaths


def _read_yaml(path:Path) -> dict[str,Any]:
    """Read a YAML file and return its top-level mapping."""
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config file {path} must contain a mapping at the top level")
    return data

def _parse_frequency(data:dict[str,Any]) -> FrequencyGrid:
    """Parse and validate the frequency section of the config."""
    required = {"start_hz", "stop_hz", "n_points"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"frequency section missing required keys: {sorted(missing)}")
    start_hz = float(data["start_hz"])
    stop_hz = float(data["stop_hz"])
    n_points = int(data["n_points"])
    if start_hz <= 0:
        raise ValueError(f"frequency.start_hz must be positive, got {start_hz}")
    if stop_hz <= start_hz:
        raise ValueError(f"frequency.stop_hz ({stop_hz}) must be greater than start_hz ({start_hz})")
    if n_points < 2:
        raise ValueError(f"frequency.n_points must be at least 2, got {n_points}")
    return FrequencyGrid(start_hz=start_hz, stop_hz=stop_hz, n_points=n_points)

def _parse_topology(data:dict[str,Any]) -> TopologyConfig:
    """Parse the topology section. Semantic validation is delegated to build_topology."""
    required = {"drop_positions_m", "bus_start_m", "bus_end_m", "tx_drop_index", "termination_ohm"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"topology section missing required keys: {sorted(missing)}")
    drop_positions_raw = data["drop_positions_m"]
    if not isinstance(drop_positions_raw, list):
        raise ValueError(f"topology.drop_positions_m must be a list, got {type(drop_positions_raw).__name__}")
    drop_positions = tuple(float(p) for p in drop_positions_raw)
    return TopologyConfig(
        drop_positions_m=drop_positions,
        bus_start_m=float(data["bus_start_m"]),
        bus_end_m=float(data["bus_end_m"]),
        tx_drop_index=int(data["tx_drop_index"]),
        termination_ohm=float(data["termination_ohm"]))

def _parse_paths(data:dict[str,Any], config_dir:Path) -> TouchstonePaths:
    """Parse touchstone paths and resolve them relative to the config directory.
    Args:
        data: The 'paths' section of the YAML config.
        config_dir: Directory containing the config file. Relative paths in
            the config are resolved against this directory.
    Returns:
        TouchstonePaths with absolute paths.
    Raises:
        ValueError: If required keys are missing or files do not exist.
    """
    required = {"tx_touchstone", "rx_touchstone"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"paths section missing required keys: {sorted(missing)}")
    tx_path = (config_dir / data["tx_touchstone"]).resolve()
    rx_path = (config_dir / data["rx_touchstone"]).resolve()
    if not tx_path.is_file():
        raise ValueError(f"tx_touchstone file not found: {tx_path}")
    if not rx_path.is_file():
        raise ValueError(f"rx_touchstone file not found: {rx_path}")
    return TouchstonePaths(tx=tx_path, rx=rx_path)

def _parse_cable(data:dict[str,Any]) -> CableParams:
    """Parse cable parameters and validate physical plausibility."""
    required = {"l_per_m", "c_per_m", "rdc_per_m", "rskin_per_m"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"cable section missing required keys: {sorted(missing)}")
    params = CableParams(
        l_per_m=float(data["l_per_m"]),
        c_per_m=float(data["c_per_m"]),
        rdc_per_m=float(data["rdc_per_m"]),
        rskin_per_m=float(data["rskin_per_m"]))
    if params.l_per_m <= 0:
        raise ValueError(f"cable.l_per_m must be positive, got {params.l_per_m}")
    if params.c_per_m <= 0:
        raise ValueError(f"cable.c_per_m must be positive, got {params.c_per_m}")
    if params.rdc_per_m < 0:
        raise ValueError(f"cable.rdc_per_m must be non-negative, got {params.rdc_per_m}")
    if params.rskin_per_m < 0:
        raise ValueError(f"cable.rskin_per_m must be non-negative, got {params.rskin_per_m}")
    # Sanity check: characteristic impedance should be near 100 Ω at 10 MHz per IEEE 802.3da. 
    # We allow a generous tolerance to permit experiments with non-conformant cables, but warn-or-reject obviously wrong values.
    z0_real = params.z0_at(10e6).real
    if not 50.0 <= z0_real <= 200.0:
        raise ValueError(
            f"cable parameters yield Z₀ ≈ {z0_real:.1f} Ω at 10 MHz; "
            f"expected near 100 Ω. Check l_per_m and c_per_m units (must be H/m and F/m).")
    return params

def load_config(path:Path) -> SimConfig:
    """Load and validate a simulation configuration from a YAML file.
    Args:
        path: Path to the YAML config file.
    Returns:
        A validated SimConfig with all paths resolved to absolute form.
    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
        ValueError: If the config structure or values are invalid.
    """
    path = Path(path).resolve()
    raw = _read_yaml(path)
    required_sections = {"frequency", "topology", "paths", "cable"}
    missing = required_sections - raw.keys()
    if missing:
        raise ValueError(f"config missing required sections: {sorted(missing)}")
    return SimConfig(
        frequency=_parse_frequency(raw["frequency"]),
        topology=_parse_topology(raw["topology"]),
        paths=_parse_paths(raw["paths"], path.parent),
        cable=_parse_cable(raw["cable"]))