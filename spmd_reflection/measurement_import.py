"""Import helpers for reconstructing node data from measurement archives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
import re
from typing import Dict, List
import zipfile

import numpy as np

from spmd_reflection.touchstone import TouchstoneData, parse_s2p_text


LEGACY_MEASUREMENT_PARAM_MAP: Dict[int, Dict[str, tuple[int, int]]] = {
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

JUMPED_MEASUREMENT_PARAM_MAP: Dict[int, Dict[str, tuple[int, int]]] = {
    1: {"S11": (0, 0), "S31": (1, 0)},
    2: {"S33": (0, 0), "S13": (1, 0)},
    3: {"S11": (0, 0), "S21": (1, 0)},
    4: {"S11": (0, 0), "S41": (1, 0)},
    5: {"S33": (0, 0), "S23": (1, 0)},
    6: {"S33": (0, 0), "S43": (1, 0)},
    7: {"S22": (0, 0), "S42": (1, 0)},
    8: {"S44": (0, 0), "S24": (1, 0)},
    9: {"S22": (0, 0), "S12": (1, 0)},
    10: {"S22": (0, 0), "S32": (1, 0)},
    11: {"S44": (0, 0), "S14": (1, 0)},
    12: {"S44": (0, 0), "S34": (1, 0)},
}

BOARD2_MEASUREMENT_PARAM_MAP: Dict[int, Dict[str, tuple[int, int]]] = {
    1: {"S11": (0, 0), "S21": (1, 0)},
    2: {"S11": (0, 0), "S31": (1, 0)},
    3: {"S11": (0, 0), "S41": (1, 0)},
    4: {"S22": (0, 0), "S42": (1, 0)},
    5: {"S22": (0, 0), "S32": (1, 0)},
    6: {"S22": (0, 0), "S12": (1, 0)},
    7: {"S33": (0, 0), "S13": (1, 0)},
    8: {"S33": (0, 0), "S23": (1, 0)},
    9: {"S33": (0, 0), "S43": (1, 0)},
    10: {"S44": (0, 0), "S34": (1, 0)},
    11: {"S44": (0, 0), "S24": (1, 0)},
    12: {"S44": (0, 0), "S14": (1, 0)},
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
    mapping_name: str


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


def _normalize_mapping_text(text: str) -> str:
    return " ".join(text.lower().split())


def _select_measurement_param_map(zip_path: str, readme_texts: List[str]) -> tuple[str, Dict[int, Dict[str, tuple[int, int]]]]:
    zip_name = PurePosixPath(zip_path).name.lower()
    if "jumped" in zip_name:
        return "jumped", JUMPED_MEASUREMENT_PARAM_MAP

    combined_readme = "\n".join(readme_texts)
    if "Ergibt: S11, S31" in combined_readme and "Ergibt: S22, S42" in combined_readme:
        return "jumped", JUMPED_MEASUREMENT_PARAM_MAP

    normalized_readme = _normalize_mapping_text(combined_readme)
    if (
        "messung 2: vna port a → platine port 1 vna port b → platine port 3" in normalized_readme
        and "messung 4: vna port a → platine port 2 vna port b → platine port 4" in normalized_readme
        and "messung 10: vna port a → platine port 4 vna port b → platine port 3" in normalized_readme
    ):
        return "board2", BOARD2_MEASUREMENT_PARAM_MAP

    return "legacy", LEGACY_MEASUREMENT_PARAM_MAP


def _load_measurements_from_directory(directory_path: str) -> tuple[Dict[int, TouchstoneData], Dict[int, str], List[str]]:
    measurements: Dict[int, TouchstoneData] = {}
    source_files: Dict[int, str] = {}
    readme_texts: List[str] = []
    base_path = Path(directory_path)

    for member in sorted(path for path in base_path.rglob("*") if path.is_file()):
        relative_name = member.relative_to(base_path).as_posix()
        if relative_name.startswith("__MACOSX/") or member.name.startswith("._"):
            continue
        if member.name.lower() in {"readme.txt", "readme.md"}:
            readme_texts.append(member.read_text(encoding="utf-8", errors="replace"))
            continue
        if member.suffix.lower() != ".s2p":
            continue

        measurement_id = _extract_measurement_id(relative_name)
        if measurement_id in measurements:
            raise ValueError(
                f"Duplicate measurement number {measurement_id} in directory: "
                f"'{source_files[measurement_id]}' and '{relative_name}'."
            )

        measurements[measurement_id] = parse_s2p_text(member.read_text(encoding="utf-8"))
        source_files[measurement_id] = relative_name

    return measurements, source_files, readme_texts


def _load_measurements_from_zip(zip_path: str) -> tuple[Dict[int, TouchstoneData], Dict[int, str], List[str]]:
    measurements: Dict[int, TouchstoneData] = {}
    source_files: Dict[int, str] = {}
    readme_texts: List[str] = []

    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            if "/__MACOSX/" in f"/{member.filename}" or PurePosixPath(member.filename).name.startswith("._"):
                continue
            if PurePosixPath(member.filename).name.lower() in {"readme.txt", "readme.md"}:
                readme_texts.append(archive.read(member).decode("utf-8", errors="replace"))
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

    return measurements, source_files, readme_texts


def load_measurement_archive(zip_path: str) -> ImportedMeasurementArchive:
    """Load the 12 measurement .s2p files from a ZIP archive or directory."""
    if Path(zip_path).is_dir():
        measurements, source_files, readme_texts = _load_measurements_from_directory(zip_path)
    else:
        measurements, source_files, readme_texts = _load_measurements_from_zip(zip_path)

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

    mapping_name, measurement_param_map = _select_measurement_param_map(zip_path, readme_texts)
    traces: Dict[str, List[MeasurementTrace]] = {}
    for measurement_id, touchstone in measurements.items():
        for param_name, (row, col) in measurement_param_map[measurement_id].items():
            traces.setdefault(param_name, []).append(
                MeasurementTrace(
                    name=param_name,
                    measurement_id=measurement_id,
                    source_file=source_files[measurement_id],
                    frequency=touchstone.frequency.copy(),
                    values=touchstone.s_params[:, row, col].copy(),
                )
            )

    return ImportedMeasurementArchive(
        measurements=measurements,
        traces=traces,
        mapping_name=mapping_name,
    )


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

    s_diff = np.zeros((len(data.frequency), 2, 2), dtype=complex)
    s11 = data.s_params[:, 0, 0]
    s12 = data.s_params[:, 0, 1]
    s13 = data.s_params[:, 0, 2]
    s14 = data.s_params[:, 0, 3]
    s21 = data.s_params[:, 1, 0]
    s22 = data.s_params[:, 1, 1]
    s23 = data.s_params[:, 1, 2]
    s24 = data.s_params[:, 1, 3]
    s31 = data.s_params[:, 2, 0]
    s32 = data.s_params[:, 2, 1]
    s33 = data.s_params[:, 2, 2]
    s34 = data.s_params[:, 2, 3]
    s41 = data.s_params[:, 3, 0]
    s42 = data.s_params[:, 3, 1]
    s43 = data.s_params[:, 3, 2]
    s44 = data.s_params[:, 3, 3]

    # Differential 2-port reduction for the port pairs (1,2) and (3,4).
    s_diff[:, 0, 0] = 0.5 * (s11 - s12 - s21 + s22)
    s_diff[:, 0, 1] = 0.5 * (s13 - s14 - s23 + s24)
    s_diff[:, 1, 0] = 0.5 * (s31 - s32 - s41 + s42)
    s_diff[:, 1, 1] = 0.5 * (s33 - s34 - s43 + s44)

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
