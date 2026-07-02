"""Selectable Elsasser-variable analysis built on the common derived layer."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from hs_observatory.analysis.base import AnalysisMethod
from hs_observatory.analysis.derived.elsasser import (
    compute_density_normalization,
    compute_elsasser,
    magnetic_field_to_alfven_units,
)
from hs_observatory.analysis.registry import register_analysis
from hs_observatory.visualization import SPACECRAFT_COLORS


def _settings(config: dict[str, Any]) -> dict[str, Any]:
    data_group = str(config.get("data_group", "observed")).lower()
    if data_group not in {"truth", "observed"}:
        raise ValueError("elsasser.data_group must be 'truth' or 'observed'")
    mc_index = config.get("mc_index", 0)
    if mc_index == "all":
        raise NotImplementedError(
            "mc_index: all is not implemented; select one observed realization"
        )
    mc_index = int(mc_index)
    if mc_index < 0:
        raise ValueError("elsasser.mc_index must be non-negative")

    velocity = config.get("velocity", {})
    density = config.get("density_normalization", {})
    mean = config.get("mean_subtraction", {})
    timebase = config.get("timebase", {})
    plot = config.get("plot", {})
    for name, value in (
        ("velocity", velocity),
        ("density_normalization", density),
        ("mean_subtraction", mean),
        ("timebase", timebase),
        ("plot", plot),
    ):
        if not isinstance(value, dict):
            raise ValueError(f"elsasser.{name} must be a mapping")

    frame = str(velocity.get("frame", "auto")).lower()
    if frame not in {"auto", "plasma", "spacecraft"}:
        raise ValueError("elsasser.velocity.frame must be auto, plasma, or spacecraft")
    correction = str(velocity.get("spacecraft_correction", "auto")).lower()
    if correction not in {"auto", "apply"}:
        raise ValueError("elsasser.velocity.spacecraft_correction must be 'auto' or 'apply'")
    density_method = str(density.get("method", "global_per_spacecraft"))
    if density_method not in {"instantaneous", "global_per_spacecraft"}:
        raise ValueError(
            "elsasser density normalization must be instantaneous or global_per_spacecraft"
        )
    species = str(density.get("species", "proton")).lower()
    if species != "proton":
        # TODO: Add alpha-particle and multi-species mass-density corrections.
        raise NotImplementedError("Only proton mass density is currently supported")
    mean_method = str(mean.get("method", "global_per_spacecraft"))
    if mean_method != "global_per_spacecraft":
        raise NotImplementedError(
            "Only global_per_spacecraft mean subtraction is currently supported"
        )
    timebase_method = str(timebase.get("method", "slower_of_n_and_v"))
    interpolation = str(timebase.get("interpolation", "linear"))
    if timebase_method != "slower_of_n_and_v":
        raise NotImplementedError("Only timebase.method: slower_of_n_and_v is supported")
    if interpolation != "linear":
        raise NotImplementedError("Only linear timebase interpolation is supported")
    dpi = int(plot.get("dpi", 150))
    if dpi <= 0:
        raise ValueError("elsasser.plot.dpi must be positive")
    return {
        "data_group": data_group,
        "mc_index": mc_index,
        "velocity_frame": frame,
        "spacecraft_correction": correction,
        "density_method": density_method,
        "species": species,
        "mean_method": mean_method,
        "timebase_method": timebase_method,
        "interpolation": interpolation,
        "plot": {
            "enabled": bool(plot.get("enabled", False)),
            "path": plot.get("path"),
            "dpi": dpi,
        },
    }


def _time_axis(handle: h5py.File, variable: str) -> np.ndarray:
    if "time" not in handle:
        raise KeyError("Input file is missing /time")
    node = handle["time"]
    if isinstance(node, h5py.Group):
        if variable not in node:
            raise KeyError(f"Input file is missing /time/{variable}")
        time = np.asarray(node[variable], dtype=float)
    else:
        time = np.asarray(node, dtype=float)
    _uniform_cadence(time, variable)
    return time


def _uniform_cadence(time: np.ndarray, variable: str) -> float:
    if time.ndim != 1 or time.size < 2 or not np.all(np.isfinite(time)):
        raise ValueError(f"Time for {variable} must be finite, one-dimensional, and nonempty")
    differences = np.diff(time)
    dt = float(np.mean(differences))
    tolerance = max(1.0e-12, abs(dt) * 1.0e-8)
    if dt <= 0.0 or not np.allclose(differences, dt, rtol=1.0e-7, atol=tolerance):
        raise ValueError(f"Time for {variable} must be uniformly sampled")
    return dt


def _read_selected(
    handle: h5py.File,
    variable: str,
    settings: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    dataset_name = variable
    if (
        variable == "v"
        and settings["data_group"] == "truth"
        and settings["velocity_frame"] == "spacecraft"
    ):
        dataset_name = "v_spacecraft_frame"
    path = f"{settings['data_group']}/{dataset_name}"
    if path not in handle:
        raise KeyError(f"Input file is missing /{path}")
    values = np.asarray(handle[path], dtype=float)
    if settings["data_group"] == "observed":
        if values.ndim not in {3, 4}:
            raise ValueError(f"/{path} must have a leading Monte Carlo dimension")
        if settings["mc_index"] >= values.shape[0]:
            raise IndexError(
                f"mc_index {settings['mc_index']} is outside /{path} with "
                f"{values.shape[0]} realizations"
            )
        values = values[settings["mc_index"]]
    expected_ndim = 2 if variable == "n" else 3
    if values.ndim != expected_ndim or (variable != "n" and values.shape[-1] != 3):
        shape = "(time, spacecraft)" if variable == "n" else "(time, spacecraft, 3)"
        raise ValueError(f"/{path} must have shape {shape} after selection")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"/{path} contains non-finite values")
    time = _time_axis(handle, variable)
    if time.size != values.shape[0]:
        raise ValueError(f"/{path} does not match its time axis")
    return values, time


def _metadata_string(handle: h5py.File, key: str) -> str | None:
    if "metadata" not in handle or key not in handle["metadata"].attrs:
        return None
    value = handle["metadata"].attrs[key]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def _input_velocity_frame(handle: h5py.File, settings: dict[str, Any]) -> str:
    requested = settings["velocity_frame"]
    if requested != "auto":
        return requested
    if settings["data_group"] == "truth":
        # /truth/v is the package's stable plasma-frame contract, including
        # older files that predate explicit frame metadata.
        return "plasma"
    frame = _metadata_string(handle, "velocity_observed_frame")
    if frame is None and "observed/v" in handle:
        frame = handle["observed/v"].attrs.get("frame")
        if isinstance(frame, bytes):
            frame = frame.decode("utf-8")
    if frame not in {"plasma", "spacecraft"}:
        raise ValueError(
            "Could not determine the observed velocity frame; set "
            "elsasser.velocity.frame explicitly to plasma or spacecraft"
        )
    return str(frame)


def _stored_spacecraft_velocity(
    handle: h5py.File,
    time_v: np.ndarray,
    n_spacecraft: int,
) -> np.ndarray | None:
    for path in ("spacecraft/velocity", "spacecraft/v", "spacecraft/V_sc"):
        if path not in handle:
            continue
        values = np.asarray(handle[path], dtype=float)
        target = (time_v.size, n_spacecraft, 3)
        if values.shape == (3,):
            return np.broadcast_to(values, target).copy()
        if values.shape == (n_spacecraft, 3):
            return np.broadcast_to(values[None, ...], target).copy()
        if values.shape == target:
            return values
        if values.shape == (n_spacecraft, time_v.size, 3):
            return np.swapaxes(values, 0, 1)
        raise ValueError(f"/{path} has an unsupported spacecraft-velocity shape")

    # Schema v2 stores the time-aligned trajectory. Its derivative recovers
    # the translation used by the current generator without changing that schema.
    if "spacecraft/r/v" in handle:
        position = np.asarray(handle["spacecraft/r/v"], dtype=float)
        if position.shape != (n_spacecraft, time_v.size, 3):
            raise ValueError("/spacecraft/r/v does not match the velocity time grid")
        return np.swapaxes(np.gradient(position, time_v, axis=1), 0, 1)

    config_yaml = _metadata_string(handle, "config_yaml")
    if config_yaml:
        parsed = yaml.safe_load(config_yaml)
        configured = (parsed or {}).get("spacecraft", {}).get("velocity_km_s")
        if configured is not None:
            velocity = np.asarray(configured, dtype=float)
            if velocity.shape != (3,):
                raise ValueError("Saved spacecraft.velocity_km_s must have three components")
            return np.broadcast_to(velocity, (time_v.size, n_spacecraft, 3)).copy()
    return None


def _interpolate(
    values: np.ndarray,
    source_time: np.ndarray,
    output_time: np.ndarray,
    variable: str,
) -> np.ndarray:
    if np.array_equal(source_time, output_time):
        return values.copy()
    tolerance = max(1.0e-12, _uniform_cadence(source_time, variable) * 1.0e-8)
    if output_time[0] < source_time[0] - tolerance or output_time[-1] > source_time[-1] + tolerance:
        raise ValueError(f"The selected output time grid extends beyond the {variable} time grid")
    flattened = values.reshape(values.shape[0], -1)
    interpolated = np.empty((output_time.size, flattened.shape[1]), dtype=float)
    for column in range(flattened.shape[1]):
        interpolated[:, column] = np.interp(output_time, source_time, flattened[:, column])
    return interpolated.reshape((output_time.size, *values.shape[1:]))


def compute_elsasser_analysis(
    input_path: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Read one data selection, align native grids, and compute Elsasser products."""

    settings = _settings(config)
    with h5py.File(input_path, "r") as handle:
        n, time_n = _read_selected(handle, "n", settings)
        v, time_v = _read_selected(handle, "v", settings)
        B, time_B = _read_selected(handle, "B", settings)
        n_spacecraft = n.shape[1]
        if v.shape[1] != n_spacecraft or B.shape[1] != n_spacecraft:
            raise ValueError("n, v, and B must contain the same spacecraft")
        frame_input = _input_velocity_frame(handle, settings)
        if frame_input == "spacecraft":
            spacecraft_velocity = _stored_spacecraft_velocity(handle, time_v, n_spacecraft)
            if spacecraft_velocity is None:
                raise ValueError(
                    "Spacecraft-frame velocity correction was requested, but spacecraft "
                    "velocity is unavailable in datasets or saved generation metadata"
                )
            v = v + spacecraft_velocity

        if "spacecraft/id" in handle:
            spacecraft_ids = [
                item.decode("utf-8") if isinstance(item, bytes) else str(item)
                for item in handle["spacecraft/id"][...]
            ]
        else:
            spacecraft_ids = [f"S{index}" for index in range(n_spacecraft)]
    if len(spacecraft_ids) != n_spacecraft:
        raise ValueError("/spacecraft/id does not match the selected data")

    dt_n = _uniform_cadence(time_n, "n")
    dt_v = _uniform_cadence(time_v, "v")
    output_time = time_v if np.isclose(dt_n, dt_v, rtol=1.0e-7) or dt_v > dt_n else time_n
    n_aligned = _interpolate(n, time_n, output_time, "n")
    v_plasma = _interpolate(v, time_v, output_time, "v")
    # TODO: Add anti-alias filtering before large-ratio magnetic downsampling.
    B_aligned = _interpolate(B, time_B, output_time, "B")

    n_normalization = compute_density_normalization(n_aligned, settings["density_method"])
    b = magnetic_field_to_alfven_units(B_aligned, n_normalization)
    products = compute_elsasser(v_plasma, b, settings["mean_method"])
    metadata = {
        "analysis_method": "elsasser",
        "input_path": str(Path(input_path)),
        "data_group": settings["data_group"],
        "mc_index": settings["mc_index"],
        "velocity_frame_input": frame_input,
        "velocity_frame_used": "plasma",
        "density_normalization_method": settings["density_method"],
        "density_species": settings["species"],
        "mean_subtraction": settings["mean_method"],
        "timebase_method": settings["timebase_method"],
        "interpolation_method": settings["interpolation"],
        "units_n": "cm^-3",
        "units_v": "km/s",
        "units_B": "nT",
        "units_b": "km/s",
        "z_plus_convention": "delta_v + delta_b",
        "z_minus_convention": "delta_v - delta_b",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    return {
        "metadata": metadata,
        "settings": settings,
        "time": output_time,
        "spacecraft_ids": spacecraft_ids,
        "n": n_aligned,
        "n_normalization": n_normalization,
        "v_plasma": v_plasma,
        "B": B_aligned,
        "b": b,
        **products,
    }


def plot_elsasser(
    output_h5_path: str | Path,
    plot_path: str | Path,
    *,
    dpi: int = 150,
) -> Path:
    """Create the 3-by-2 component diagnostic from an Elsasser HDF5 product."""

    import matplotlib.pyplot as plt

    with h5py.File(output_h5_path, "r") as handle:
        time = np.asarray(handle["time"], dtype=float)
        z_plus = np.asarray(handle["elsasser/z_plus"], dtype=float)
        z_minus = np.asarray(handle["elsasser/z_minus"], dtype=float)
        ids = [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in handle["spacecraft/id"][...]
        ]
    figure, axes = plt.subplots(3, 2, figsize=(11.0, 8.0), sharex=True, constrained_layout=True)
    fallback = plt.cm.tab10(np.linspace(0.0, 0.9, len(ids)))
    for row, component in enumerate("xyz"):
        for column, (values, label) in enumerate(((z_plus, r"$z^+$"), (z_minus, r"$z^-$"))):
            axis = axes[row, column]
            for spacecraft, spacecraft_id in enumerate(ids):
                axis.plot(
                    time,
                    values[:, spacecraft, row],
                    color=SPACECRAFT_COLORS.get(spacecraft_id, fallback[spacecraft]),
                    linewidth=0.8,
                    label=spacecraft_id,
                )
            axis.set_title(f"{label}_{component}")
            axis.set_ylabel("[km/s]")
            axis.grid(True, alpha=0.22)
    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Time [s]")
    if len(ids) <= 12:
        axes[0, 1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    output = Path(plot_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return output


def _write_metadata(group: h5py.Group, metadata: dict[str, Any]) -> None:
    for key, value in metadata.items():
        group.attrs[key] = (
            value
            if isinstance(value, (str, int, float, bool))
            else json.dumps(value, sort_keys=True)
        )


def _write_dataset(group: h5py.Group, name: str, values: np.ndarray, units: str) -> None:
    dataset = group.create_dataset(
        name, data=values, compression="gzip", compression_opts=4, chunks=True
    )
    dataset.attrs["units"] = units


@register_analysis
class ElsasserAnalysis(AnalysisMethod):
    """Analysis-registry adapter for common Elsasser derived quantities."""

    name = "elsasser"
    config_section = "elsasser"

    def run(self, input_path: str | Path) -> dict[str, Any]:
        return compute_elsasser_analysis(input_path, self.config)

    def write_result(self, result: dict[str, Any], output_path: str | Path) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        plot_settings = result["settings"]["plot"]
        plot_path = Path(plot_settings["path"] or output.with_name(f"{output.stem}.png"))
        result["metadata"]["plot_enabled"] = plot_settings["enabled"]
        result["metadata"]["plot_path"] = str(plot_path) if plot_settings["enabled"] else "null"
        with h5py.File(output, "w") as handle:
            _write_metadata(handle.create_group("metadata"), result["metadata"])
            time = handle.create_dataset("time", data=result["time"])
            time.attrs["units"] = "s"
            spacecraft = handle.create_group("spacecraft")
            spacecraft.create_dataset(
                "id",
                data=np.asarray(result["spacecraft_ids"], dtype=h5py.string_dtype("utf-8")),
            )
            elsasser = handle.create_group("elsasser")
            _write_dataset(elsasser, "z_plus", result["z_plus"], "km/s")
            _write_dataset(elsasser, "z_minus", result["z_minus"], "km/s")
            alfven = handle.create_group("alfven")
            _write_dataset(alfven, "b", result["b"], "km/s")
            _write_dataset(alfven, "delta_b", result["delta_b"], "km/s")
            velocity = handle.create_group("velocity")
            _write_dataset(velocity, "v_plasma", result["v_plasma"], "km/s")
            _write_dataset(velocity, "delta_v", result["delta_v"], "km/s")
            density = handle.create_group("density")
            _write_dataset(density, "n", result["n"], "cm^-3")
            _write_dataset(
                density,
                "n_normalization",
                result["n_normalization"],
                "cm^-3",
            )
        if plot_settings["enabled"]:
            plot_elsasser(output, plot_path, dpi=plot_settings["dpi"])
        return output
