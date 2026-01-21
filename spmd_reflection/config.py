"""Configuration loading and validation (JSON + CLI defaults)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


DEFAULTS: Dict[str, Any] = {
    "analysis": "ac",
    "freq_start": 1e5,
    "freq_stop": 4e7,
    "npoints": 400,
    "z0": 100.0,
    "nodes": 16,
    "length": 100.0,
    "attach_points": None,
    "drop_max": 0.02,
    "random_drop": False,
    "random_attach": False,
    "separation_min": 1.0,
    "start_pad": 0.0,
    "end_pad": 0.0,
    "start_attach": 0,
    "end_attach": 0,
    "attach_error": 0.0,
    "seed": -1,
    "tx_node": 1,
    "s2p": None,
    "cable_model": {
        "rdc": 0.0094,
        "l": 20.6435e-9,
        "c": 2.25026e-12,
        "rskin": 1.134268e-5,
        "ref_length": 0.05,
    },
    "termination": {
        "rterm": 100.0,
    },
}


@dataclass
class Config:
    data: Dict[str, Any] = field(default_factory=dict)

    @property
    def freq_start(self) -> float:
        return float(self.data["freq_start"])

    @property
    def freq_stop(self) -> float:
        return float(self.data["freq_stop"])

    @property
    def npoints(self) -> int:
        return int(self.data["npoints"])

    @property
    def z0(self) -> float:
        return float(self.data["z0"])


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(dict(base[key]), value)
        else:
            base[key] = value
    return base


def load_config(json_path: Optional[str], overrides: Optional[Dict[str, Any]] = None) -> Config:
    data = dict(DEFAULTS)
    if json_path:
        with open(json_path, "r") as handle:
            payload = json.load(handle)
        data = _deep_update(data, payload)
    if overrides:
        data = _deep_update(data, overrides)

    _validate(data)
    return Config(data=data)


def _validate(data: Dict[str, Any]) -> None:
    if data["analysis"] != "ac":
        raise ValueError("Only 'ac' analysis is supported in v0.")
    if data["freq_start"] <= 0 or data["freq_stop"] <= 0:
        raise ValueError("Frequency bounds must be > 0.")
    if data["freq_stop"] <= data["freq_start"]:
        raise ValueError("freq_stop must be greater than freq_start.")
    if data["npoints"] < 2:
        raise ValueError("npoints must be >= 2.")
    if not data.get("s2p"):
        raise ValueError("s2p path is required for v0.")
