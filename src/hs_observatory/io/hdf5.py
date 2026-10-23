"""HDF5 schema for synthetic observations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from hs_observatory.errors.observatory import relative_alignment_metadata


def _dataset(group: h5py.Group, name: str, value: np.ndarray) -> h5py.Dataset:
    array = np.asarray(value)
    # gzip and automatic chunking keep large Monte Carlo ensembles compact.
    return group.create_dataset(
        name, data=array, compression="gzip", compression_opts=4, chunks=True
    )


def write_observations(
    path: str | Path,
    *,
    time: Mapping[str, np.ndarray],
    spacecraft_ids: list[str] | tuple[str, ...],
    position: Mapping[str, np.ndarray],
    truth: Mapping[str, np.ndarray],
    observed: Mapping[str, np.ndarray],
    errors: Mapping[str, np.ndarray] | None,
    metadata: Mapping[str, Any],
) -> Path:
    """Write the documented observatory schema.

    Each quantity has its own time and position grid. Truth arrays are
    ``n(time_n, sc)``, ``v(time_v, sc, component)``, and
    ``B(time_B, sc, component)``. Generated products may additionally contain
    ``v_spacecraft_frame(time_v, sc, component)`` while retaining ``v`` as
    plasma-frame truth.
    Observations/errors have the same trailing dimensions plus leading ``mc``.
    Each position array is ``r[quantity](sc, time_quantity, component)``.
    """

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    error_config = metadata.get("observatory_error_config", {})
    if isinstance(error_config, (str, bytes)):
        error_config = json.loads(error_config)
    with h5py.File(output, "w") as handle:
        meta_group = handle.create_group("metadata")
        for key, value in metadata.items():
            if value is None:
                value = "null"
            elif not isinstance(value, (str, bytes, int, float, bool, np.number)):
                value = json.dumps(value, sort_keys=True)
            meta_group.attrs[key] = value

        time_group = handle.create_group("time")
        for quantity in ("n", "v", "B"):
            _dataset(time_group, quantity, np.asarray(time[quantity], dtype=float))
        sc_group = handle.create_group("spacecraft")
        sc_group.create_dataset(
            "id", data=np.asarray(spacecraft_ids, dtype=h5py.string_dtype("utf-8"))
        )
        position_group = sc_group.create_group("r")
        for quantity in ("n", "v", "B"):
            _dataset(position_group, quantity, np.asarray(position[quantity], dtype=float))

        for group_name, values in (("truth", truth), ("observed", observed)):
            group = handle.create_group(group_name)
            names = values if group_name == "truth" else ("n", "v", "B")
            for quantity in names:
                _dataset(group, quantity, values[quantity])
        if errors is not None:
            group = handle.create_group("errors")
            for name, values in errors.items():
                dataset = _dataset(group, name, values)
                if name in {"relative_alignment_rotvec", "relative_alignment_matrix"}:
                    dataset.attrs.update(relative_alignment_metadata(error_config["alignment"]))
                    vector = name.endswith("rotvec")
                    dataset.attrs["units"] = "rad" if vector else "1"
                    dataset.attrs["representation"] = (
                        "rotation_vector" if vector else "rotation_matrix"
                    )
                    dataset.attrs["axes"] = (
                        "x,y,z" if vector else "output_component,input_component"
                    )
                if name == "time_offset":
                    dataset.attrs["units"] = "s"
                elif name == "position_offset":
                    dataset.attrs["units"] = "km"
                elif name in {"magnetic_dc_offset", "magnetic_ac_noise"}:
                    dataset.attrs["units"] = "nT"
                elif name in {
                    "velocity_alignment_rotvec",
                    "magnetic_alignment_rotvec",
                    "velocity_angle_alpha",
                    "velocity_angle_beta",
                }:
                    dataset.attrs["units"] = "rad"
                for component in ("density", "velocity_speed"):
                    if name in {
                        f"{component}_bias",
                        f"{component}_log_error",
                        f"{component}_fractional_error",
                    }:
                        section = error_config.get(component, {})
                        interpretation = section.get(
                            "error_interpretation",
                            "fractional_error"
                            if section.get("model") == "gaussian_fractional_bias_plus_ou"
                            else "log_multiplier",
                        )
                        if name.endswith("_fractional_error"):
                            interpretation = "fractional_error"
                        elif name.endswith("_log_error"):
                            interpretation = "log_multiplier"
                        dataset.attrs["units"] = "1"
                        dataset.attrs["interpretation"] = interpretation + (
                            "_bias" if name.endswith("_bias") else ""
                        )
                        if "model" in section:
                            dataset.attrs["model"] = section["model"]
                if name in {"velocity_angle_alpha", "velocity_angle_beta"}:
                    section = error_config.get("velocity_angle", {})
                    if "model" in section:
                        dataset.attrs["model"] = section["model"]
                        dataset.attrs["requirement_interpretation"] = section.get(
                            "requirement_interpretation",
                            "total_angle_confidence_radius"
                            if section["model"] == "transverse_ou_rotation_total_angle"
                            else "per_transverse_component",
                        )
    return output


def read_observations(path: str | Path) -> dict[str, Any]:
    """Read an observatory HDF5 file into ordinary dictionaries and arrays."""

    with h5py.File(path, "r") as handle:
        metadata = {}
        for key, value in handle["metadata"].attrs.items():
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            metadata[key] = value.item() if isinstance(value, np.generic) else value
        result: dict[str, Any] = {
            "metadata": metadata,
            "time": {q: handle[f"time/{q}"][...] for q in ("n", "v", "B")},
            "spacecraft": {
                "id": [
                    value.decode("utf-8") if isinstance(value, bytes) else str(value)
                    for value in handle["spacecraft/id"][...]
                ],
                "r": {q: handle[f"spacecraft/r/{q}"][...] for q in ("n", "v", "B")},
            },
            "truth": {q: handle[f"truth/{q}"][...] for q in ("n", "v", "B")},
            "observed": {q: handle[f"observed/{q}"][...] for q in ("n", "v", "B")},
        }
        if "truth/v_spacecraft_frame" in handle:
            result["truth"]["v_spacecraft_frame"] = handle["truth/v_spacecraft_frame"][...]
        if "errors" in handle:
            result["errors"] = {name: handle[f"errors/{name}"][...] for name in handle["errors"]}
    return result
