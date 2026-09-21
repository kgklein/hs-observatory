"""Read HelioSwarm formation positions from mission-summary CDF files.

This adapts the focused CDF-loading logic in
``sparse-field-reconstruction/src/sparse_recon/datasets/helioswarm.py``.  The
expected CDF variables are ``Epoch`` (TT2000), ``Position`` with shape
``(record, spacecraft, 3)``, and ``Spacecraft_Label``. A file or a directory
of ``*.cdf`` files may be supplied.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from hs_observatory.spacecraft.configuration import (
    SpacecraftConfiguration,
    synthetic_configuration,
)

FILL_VALUE_THRESHOLD = -1.0e30


def select_cdf_configuration_sample(
    path: str | Path,
    *,
    requested_time: str | None = None,
) -> dict[str, Any]:
    """Return the nearest valid CDF formation sample and its source context."""

    epochs, positions, labels, sources = read_cdf_positions(path)
    target = epochs[0] if requested_time is None else np.datetime64(requested_time)
    index = int(np.argmin(np.abs(epochs - target)))
    selected = positions[index]
    valid = np.all(selected > FILL_VALUE_THRESHOLD, axis=1)
    valid &= np.asarray([label != "N/A" for label in labels])
    if not np.any(valid):
        raise ValueError("Selected CDF record contains no valid spacecraft positions")
    return {
        "epochs": epochs,
        "positions": positions,
        "labels": labels,
        "source_files": sources,
        "requested_time": requested_time,
        "selected_time": str(epochs[index]),
        "selected_index": index,
        "selected_position": selected,
        "valid": valid,
        "source_file": sources[index],
    }


def _cdf_files(path: str | Path) -> list[Path]:
    source = Path(path).expanduser()
    if source.is_file() and source.suffix.lower() == ".cdf":
        return [source]
    if source.is_dir():
        files = sorted(source.glob("*.cdf"))
        if files:
            return files
        raise FileNotFoundError(f"No CDF files found in directory: {source}")
    raise FileNotFoundError(f"CDF source does not exist: {source}")


def read_cdf_positions(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Return epochs, positions, labels, and one source filename per epoch.

    Epochs are ``datetime64[ns]`` with shape ``(n_record,)``; positions are in
    the CDF's native units (normally km) with shape ``(n_record, n_sc, 3)``.
    """

    try:
        from cdflib import CDF, cdfepoch
    except ImportError as exc:  # pragma: no cover - dependency declared, but friendly failure
        raise ImportError(
            "Reading spacecraft CDFs requires cdflib, included in the base installation"
        ) from exc

    epoch_parts: list[np.ndarray] = []
    position_parts: list[np.ndarray] = []
    source_files: list[str] = []
    labels: list[str] | None = None
    for filename in _cdf_files(path):
        cdf = CDF(str(filename))
        try:
            epoch = np.asarray(cdf.varget("Epoch"), dtype=np.int64)
            position = np.asarray(cdf.varget("Position"), dtype=float)
            file_labels = np.asarray(cdf.varget("Spacecraft_Label")).astype(str).ravel().tolist()
        except ValueError as exc:
            if "No records found" in str(exc):
                continue
            raise
        if epoch.size == 0:
            continue
        if position.shape != (len(epoch), len(file_labels), 3):
            raise ValueError(f"Unexpected Position shape {position.shape} in {filename}")
        if labels is not None and labels != file_labels:
            raise ValueError("Spacecraft labels differ across CDF files")
        labels = file_labels
        epoch_parts.append(epoch)
        position_parts.append(position)
        source_files.extend([str(filename)] * len(epoch))

    if not epoch_parts:
        raise ValueError(f"No CDF Epoch/Position records found in {path}")
    epoch_tt2000 = np.concatenate(epoch_parts)
    positions = np.concatenate(position_parts)
    order = np.argsort(epoch_tt2000)
    epochs = np.asarray(cdfepoch.to_datetime(epoch_tt2000[order]), dtype="datetime64[ns]")
    return epochs, positions[order], labels or [], [source_files[index] for index in order]


def configuration_from_cdf(
    path: str | Path,
    time: np.ndarray,
    *,
    requested_time: str | None = None,
    velocity_km_s: Sequence[float] | None = None,
) -> SpacecraftConfiguration:
    """Select the nearest valid CDF formation and repeat/advect it over ``time``.

    This initial scaffold treats the selected CDF record as a formation
    snapshot. A common translational velocity can optionally advect it. Future
    readers may interpolate full ephemerides without changing downstream APIs.
    """

    sample = select_cdf_configuration_sample(path, requested_time=requested_time)
    selected = sample["selected_position"]
    valid = sample["valid"]

    time = np.asarray(time, dtype=float)
    velocity = (
        np.zeros(3, dtype=float)
        if velocity_km_s is None
        else np.asarray(velocity_km_s, dtype=float)
    )
    if velocity.shape != (3,):
        raise ValueError("spacecraft.velocity_km_s must contain three components")
    base = selected[valid]
    trajectory = base[:, None, :] + time[None, :, None] * velocity[None, None, :]
    selected_labels = [label for label, keep in zip(sample["labels"], valid) if keep]
    metadata: dict[str, Any] = {
        "source": "cdf",
        "cdf_path": str(Path(path).expanduser()),
        "source_file": sample["source_file"],
        "requested_time": requested_time,
        "selected_time": sample["selected_time"],
        "selected_index": sample["selected_index"],
        "position_units": "km",
        "selection": "nearest_record_snapshot",
    }
    return SpacecraftConfiguration(
        selected_labels,
        time,
        trajectory,
        metadata,
        velocity=None if velocity_km_s is None else velocity,
    )


def load_spacecraft_configuration(config: dict[str, Any], time: np.ndarray) -> SpacecraftConfiguration:
    """Load a CDF configuration, optionally falling back to synthetic geometry."""

    source = str(config.get("source", "synthetic")).lower()
    configured_velocity = config.get("velocity_km_s")
    if source in {"synthetic", "synthetic_fallback"}:
        velocity = (
            (0.0, 0.0, 0.0)
            if configured_velocity is None
            else configured_velocity
        )
        return synthetic_configuration(time, velocity_km_s=velocity)
    if source != "cdf":
        raise ValueError(f"Unknown spacecraft source: {source}")

    cdf_path = config.get("cdf_path", config.get("cdf_directory"))
    if not cdf_path:
        if config.get("fallback_synthetic", False):
            velocity = (
                (0.0, 0.0, 0.0)
                if configured_velocity is None
                else configured_velocity
            )
            return synthetic_configuration(time, velocity_km_s=velocity)
        raise ValueError("CDF spacecraft source requires cdf_path or cdf_directory")
    try:
        return configuration_from_cdf(
            cdf_path,
            time,
            requested_time=config.get("configuration_time"),
            velocity_km_s=configured_velocity,
        )
    except (FileNotFoundError, ImportError, ValueError) as exc:
        if not config.get("fallback_synthetic", False):
            raise
        velocity = (
            (0.0, 0.0, 0.0)
            if configured_velocity is None
            else configured_velocity
        )
        fallback = synthetic_configuration(time, velocity_km_s=velocity)
        fallback.metadata["fallback_reason"] = str(exc)
        fallback.metadata["requested_cdf_path"] = str(cdf_path)
        return fallback
