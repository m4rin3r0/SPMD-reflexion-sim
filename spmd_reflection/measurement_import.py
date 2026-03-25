"""Import helpers for reconstructing node data from measurement archives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
import re
from typing import Dict, List
import zipfile

import numpy as np

from spmd_reflection.touchstone import TouchstoneData, parse_s2p_text


MEASUREMENT_PARAM_MAP: Dict[int, Dict[str, tuple[int, int]]] = {
    1: {"S11": (0, 0), "S21": (1, 0)},
    2: {"S22": (0, 0), "S12": (1, 0)},
    3: {"S11": (0, 0), "S31": (1, 0)},
    4: {"S11": (0, 0), "S41": (1, 0)},
    5: {"S22": (0, 0), "S32": (1, 0)},
    6: {"S22": (0, 0), "S42": (1, 0)},
    7: {"S33": (0, 0), "S43": (1, 0)},
    8: {"S44": (0, 0), "S34": (1, 0)},
    9: {"S33": (0, 0), "S13": (1, 0)},
    10: {"S33": (0, 0), "S23": (1, 0)},
    11: {"S44": (0, 0), "S14": (1, 0)},
    12: {"S44": (0, 0), "S24": (1, 0)},
}

_MEASUREMENT_FILE_PATTERN = re.compile(r"(?<!\d)(0?[1-9]|1[0-2])(?!\d)")


@dataclass
class MeasurementTrace:
    name: str
    measurement_id: int
    source_file: str
    frequency: np.ndarray
    values: np.ndarray


@dataclass
class ImportedMeasurementArchive:
    measurements: Dict[int, TouchstoneData]
    traces: Dict[str, List[MeasurementTrace]]


_TRACE_TARGETS: Dict[str, tuple[int, int]] = {
    "S11": (0, 0),
    "S12": (0, 1),
    "S13": (0, 2),
    "S14": (0, 3),
    "S21": (1, 0),
    "S22": (1, 1),
    "S23": (1, 2),
    "S24": (1, 3),
    "S31": (2, 0),
    "S32": (2, 1),
    "S33": (2, 2),
    "S34": (2, 3),
    "S41": (3, 0),
    "S42": (3, 1),
    "S43": (3, 2),
    "S44": (3, 3),
}


def _extract_measurement_id(path: str) -> int:
    filename = PurePosixPath(path).name
    stem = PurePosixPath(path).stem
    matches = sorted({int(match) for match in _MEASUREMENT_FILE_PATTERN.findall(stem)})
    if not matches:
        raise ValueError(
            f"Could not determine measurement number from filename '{filename}'. "
            "Expected a number between 1 and 12 in the filename."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Filename '{filename}' contains multiple possible measurement numbers: {matches}."
        )
    return matches[0]


def load_measurement_archive(zip_path: str) -> ImportedMeasurementArchive:
    """Load a ZIP archive with the 12 measurement .s2p files."""
    measurements: Dict[int, TouchstoneData] = {}
    source_files: Dict[int, str] = {}

    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            if "/__MACOSX/" in f"/{member.filename}" or PurePosixPath(member.filename).name.startswith("._"):
                continue
            if not member.filename.lower().endswith(".s2p"):
                continue

            measurement_id = _extract_measurement_id(member.filename)
            if measurement_id in measurements:
                raise ValueError(
                    f"Duplicate measurement number {measurement_id} in ZIP archive: "
                    f"'{source_files[measurement_id]}' and '{member.filename}'."
                )

            text = archive.read(member).decode("utf-8")
            measurements[measurement_id] = parse_s2p_text(text)
            source_files[measurement_id] = member.filename

    missing = [measurement_id for measurement_id in range(1, 13) if measurement_id not in measurements]
    if missing:
        raise ValueError(f"ZIP archive is missing measurement files for: {missing}")

    first_measurement = measurements[1]
    for measurement_id, touchstone in measurements.items():
        if touchstone.z0 != first_measurement.z0:
            raise ValueError(
                f"Measurement {measurement_id} has Z0={touchstone.z0}, "
                f"expected {first_measurement.z0}."
            )
        if touchstone.frequency.shape != first_measurement.frequency.shape or not np.allclose(
            touchstone.frequency, first_measurement.frequency
        ):
            raise ValueError(
                f"Measurement {measurement_id} does not match the frequency grid of measurement 1."
            )

    traces: Dict[str, List[MeasurementTrace]] = {}
    for measurement_id, touchstone in measurements.items():
        for param_name, (row, col) in MEASUREMENT_PARAM_MAP[measurement_id].items():
            traces.setdefault(param_name, []).append(
                MeasurementTrace(
                    name=param_name,
                    measurement_id=measurement_id,
                    source_file=source_files[measurement_id],
                    frequency=touchstone.frequency.copy(),
                    values=touchstone.s_params[:, row, col].copy(),
                )
            )

    return ImportedMeasurementArchive(measurements=measurements, traces=traces)


def build_single_ended_4port(archive: ImportedMeasurementArchive) -> TouchstoneData:
    """Reconstruct a 4-port single-ended S-matrix from the imported measurements."""
    reference = archive.measurements[1]
    s_params = np.zeros((len(reference.frequency), 4, 4), dtype=complex)

    for trace_name, (row, col) in _TRACE_TARGETS.items():
        trace_group = archive.traces.get(trace_name)
        if not trace_group:
            raise ValueError(f"Missing trace data for {trace_name}.")
        stacked = np.stack([trace.values for trace in trace_group], axis=0)
        s_params[:, row, col] = np.mean(stacked, axis=0)

    return TouchstoneData(
        frequency=reference.frequency.copy(),
        s_params=s_params,
        z0=reference.z0,
    )


def convert_single_ended_to_differential(data: TouchstoneData) -> TouchstoneData:
    """Convert a 4-port single-ended S-matrix to a 2-port differential S-matrix.

    Ports are paired as (1, 2) and (3, 4).
    """
    if data.s_params.ndim != 3 or data.s_params.shape[1:] != (4, 4):
        raise ValueError("Expected a 4-port single-ended S-matrix for conversion.")

    transform = (1.0 / np.sqrt(2.0)) * np.array(
        [
            [1.0, -1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=complex,
    )
    transform_inv = transform.conj().T

    s_diff = np.zeros((len(data.frequency), 2, 2), dtype=complex)
    for idx, s_single in enumerate(data.s_params):
        s_mixed = transform @ s_single @ transform_inv
        s_diff[idx] = s_mixed[np.ix_([0, 2], [0, 2])]

    return TouchstoneData(
        frequency=data.frequency.copy(),
        s_params=s_diff,
        z0=2.0 * data.z0,
    )


def load_differential_s2p_from_archive(zip_path: str) -> TouchstoneData:
    """Load a measurement ZIP archive and convert it to a differential 2-port."""
    archive = load_measurement_archive(zip_path)
    single_ended = build_single_ended_4port(archive)
    return convert_single_ended_to_differential(single_ended)
