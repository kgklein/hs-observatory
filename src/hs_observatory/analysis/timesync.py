"""Timesync magnetic-field reconstruction from spacecraft time series.

Timesync follows the Broeren, Klein, & TenBarge (2024) idea of reconstructing
a 3D magnetic-field volume as a sequence of 2D transverse planes. For each
axial plane, each spacecraft time series is synchronized to the plane's axial
location, then the synchronized transverse samples are interpolated with 2D
inverse-distance weighting.

The fitter uses only HDF5 spacecraft measurements: time, unwrapped physical
spacecraft positions, and magnetic-field samples. It does not load or query a
simulation cube; validation against truth remains a separate layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import warnings

import h5py
import numpy as np

from hs_observatory.analysis.base import AnalysisMethod
from hs_observatory.analysis.registry import register_analysis


COMPONENT_NAMES = ("Bx", "By", "Bz")
PLOT_CMAP_NAMES = {"batlow", "lajolla", "magma", "inferno", "plasma", "viridis", "cividis"}
ERROR_CMAP_NAMES = {"imola", "lajolla", "devon", "magma", "inferno", "plasma", "YlOrRd"}
BACKGROUND_MODES = {"metadata", "actual-volume-mean", "reconstructed-volume-mean", "user"}
COMPONENT_MODES = {"raw", "delta"}


@dataclass(frozen=True)
class TimesyncGrid:
    points: np.ndarray
    full_points: np.ndarray
    local_coordinates: np.ndarray
    axial: np.ndarray
    transverse_u: np.ndarray
    transverse_v: np.ndarray
    mask: np.ndarray
    barycenter_path: np.ndarray
    center: np.ndarray
    e_parallel: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    axial_values: np.ndarray
    transverse_values: np.ndarray
    radius: float
    requested_radius: float
    radius_padding: float
    max_spacecraft_transverse_radius: float
    axial_coverage_min: float
    axial_coverage_max: float
    n_axial: int
    n_transverse: int


def _normalize(vector: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError(f"Cannot normalize {name}")
    return np.asarray(vector, dtype=float) / norm


def _perpendicular_basis(e_parallel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    trial = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(e_parallel, trial))) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    e1 = _normalize(np.cross(e_parallel, trial), "first perpendicular basis vector")
    e2 = _normalize(np.cross(e_parallel, e1), "second perpendicular basis vector")
    return e1, e2


def _vector_from_config(value: Any, name: str) -> np.ndarray:
    if isinstance(value, str):
        parts = value.replace(",", " ").split()
        vector = np.asarray([float(part) for part in parts], dtype=float)
    else:
        vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return vector


def _settings(config: dict[str, Any]) -> dict[str, Any]:
    t0 = config.get("t0")
    t0_index = config.get("t0_index")
    if (t0 is None) == (t0_index is None):
        raise ValueError("timesync requires exactly one of t0 or t0_index")
    delta_t = float(config["delta_t"])
    if delta_t < 0.0:
        raise ValueError("timesync.delta_t must be non-negative")
    data_group = str(config.get("data_group", "truth")).lower()
    if data_group in {"mc", "monte_carlo", "measurement", "measurements", "noisy"}:
        data_group = "observed"
    if data_group not in {"truth", "observed"}:
        raise ValueError("timesync.data_group must be 'truth' or 'observed'")
    mc_index_config = config.get("mc_index", 0)
    if data_group == "observed" and str(mc_index_config).lower() == "all":
        raise NotImplementedError(
            "Timesync reconstruction requires one observed realization; set mc_index to an integer"
        )

    grid = config.get("grid", config)
    if not isinstance(grid, dict):
        raise ValueError("timesync.grid must be a mapping")
    plot = config.get("plot", {})
    if not isinstance(plot, dict):
        raise ValueError("timesync.plot must be a mapping")
    reference = config.get("reference", {})
    if not isinstance(reference, dict):
        raise ValueError("timesync.reference must be a mapping")
    averaged_error_plot = config.get("averaged_error_plot", config.get("broeren_error_plot", {}))
    if not isinstance(averaged_error_plot, dict):
        raise ValueError("timesync.averaged_error_plot must be a mapping")
    time_interp = str(config.get("time_interp", "linear")).lower()
    if time_interp not in {"linear", "nearest"}:
        raise ValueError("timesync.time_interp must be 'linear' or 'nearest'")
    axis_source = str(config.get("axis_source", "barycenter_path")).lower()
    if axis_source not in {"barycenter_path", "user_vector", "flow_velocity"}:
        raise ValueError(
            "timesync.axis_source must be barycenter_path, user_vector, or flow_velocity"
        )
    if axis_source == "flow_velocity":
        raise NotImplementedError(
            "timesync.axis_source: flow_velocity is reserved until a flow-vector "
            "HDF5 convention is added"
        )
    user_axis = None
    if axis_source == "user_vector":
        if "axis_vector" not in config:
            raise ValueError("timesync.axis_source: user_vector requires axis_vector")
        user_axis = _vector_from_config(config["axis_vector"], "timesync.axis_vector")
    magnitude_cmap = str(plot.get("magnitude_cmap", plot.get("cmap", "batlow")))
    error_cmap = str(plot.get("error_cmap", "imola"))
    component_mode = str(plot.get("component_mode", "raw"))
    background_mode = str(plot.get("background_mode", "metadata"))
    if magnitude_cmap not in PLOT_CMAP_NAMES:
        raise ValueError(f"timesync.plot.magnitude_cmap must be one of {sorted(PLOT_CMAP_NAMES)}")
    if error_cmap not in ERROR_CMAP_NAMES:
        raise ValueError(f"timesync.plot.error_cmap must be one of {sorted(ERROR_CMAP_NAMES)}")
    if component_mode not in COMPONENT_MODES:
        raise ValueError("timesync.plot.component_mode must be raw or delta")
    if background_mode not in BACKGROUND_MODES:
        raise ValueError(f"timesync.plot.background_mode must be one of {sorted(BACKGROUND_MODES)}")
    averaged_error_cmap = str(
        averaged_error_plot.get("error_cmap", plot.get("error_cmap", "imola"))
    )
    if averaged_error_cmap not in ERROR_CMAP_NAMES:
        raise ValueError(
            f"timesync.averaged_error_plot.error_cmap must be one of {sorted(ERROR_CMAP_NAMES)}"
        )
    return {
        "data_group": data_group,
        "mc_index": None if data_group == "truth" else int(mc_index_config),
        "t0": None if t0 is None else float(t0),
        "t0_index": None if t0_index is None else int(t0_index),
        "delta_t": delta_t,
        "sample_time_stride": int(config.get("sample_time_stride", 1)),
        "time_interp": time_interp,
        "allow_nearest_fallback": bool(config.get("allow_nearest_fallback", True)),
        "idw_power": float(config.get("idw_power", 2.0)),
        "idw_eps": float(config.get("idw_eps", 1.0e-12)),
        "axis_source": axis_source,
        "axis_vector": user_axis,
        "min_spacecraft_per_plane": int(config.get("min_spacecraft_per_plane", 7)),
        "coordinate_units": str(config.get("coordinate_units", "km")),
        "grid": {
            "radius": float(grid.get("radius", grid.get("radius_km"))),
            "n_axial": int(grid["n_axial"]),
            "n_transverse": int(grid.get("n_transverse", grid.get("n_radial", 11))),
            "axial_padding": float(grid.get("axial_padding", grid.get("axial_padding_km", 0.0))),
            "radius_padding": float(
                grid.get("radius_padding", grid.get("transverse_padding", 0.0))
            ),
        },
        "plot": {
            "enabled": bool(plot.get("enabled", False)),
            "path": plot.get("path"),
            "dpi": int(plot.get("dpi", 150)),
            "magnitude_cmap": magnitude_cmap,
            "error_cmap": error_cmap,
            "component_mode": component_mode,
            "background_mode": background_mode,
            "background_vector": plot.get("background_vector"),
        },
        "reference": {
            "enabled": bool(reference.get("enabled", False)),
            "generation_config": reference.get("generation_config"),
            "observation_path": reference.get("observation_path"),
        },
        "averaged_error_plot": {
            "enabled": bool(averaged_error_plot.get("enabled", False)),
            "path": averaged_error_plot.get("path"),
            "dpi": int(averaged_error_plot.get("dpi", plot.get("dpi", 150))),
            "error_cmap": averaged_error_cmap,
            "capped": bool(averaged_error_plot.get("capped", True)),
        },
    }


def _read_measurements(
    input_path: str | Path,
    *,
    data_group: str,
    mc_index: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    with h5py.File(input_path, "r") as handle:
        if "time/B" in handle:
            times = np.asarray(handle["time/B"], dtype=float)
        elif "time" in handle:
            times = np.asarray(handle["time"], dtype=float)
        else:
            raise KeyError("Input file is missing /time/B")
        path = f"{data_group}/B"
        if path not in handle:
            raise KeyError(f"Input file is missing /{path}")
        magnetic_source = np.asarray(handle[path], dtype=float)
        if data_group == "observed":
            if magnetic_source.ndim != 4 or magnetic_source.shape[-1] != 3:
                raise ValueError("/observed/B must have shape (mc, time, spacecraft, 3)")
            index = 0 if mc_index is None else int(mc_index)
            if not 0 <= index < magnetic_source.shape[0]:
                raise IndexError(
                    f"mc_index {index} is outside /observed/B ensemble axis "
                    f"with length {magnetic_source.shape[0]}"
                )
            magnetic = magnetic_source[index]
        else:
            magnetic = magnetic_source
        if "spacecraft/r/B" in handle:
            positions = np.asarray(handle["spacecraft/r/B"], dtype=float)
        elif "spacecraft/r" in handle:
            positions = np.asarray(handle["spacecraft/r"], dtype=float)
        else:
            raise KeyError("Input file is missing /spacecraft/r/B")
        if "spacecraft/id" in handle:
            spacecraft_ids = [
                item.decode("utf-8") if isinstance(item, bytes) else str(item)
                for item in handle["spacecraft/id"][...]
            ]
        else:
            spacecraft_ids = [f"S{index}" for index in range(positions.shape[0])]

    if magnetic.ndim != 3 or magnetic.shape[-1] != 3:
        raise ValueError(f"/{data_group}/B must have shape (time, spacecraft, 3)")
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("/spacecraft/r/B must have shape (spacecraft, time, 3)")
    if magnetic.shape[:2] != (positions.shape[1], positions.shape[0]):
        raise ValueError(
            f"/{data_group}/B shape must match /spacecraft/r/B as (time, spacecraft, 3)"
        )
    if magnetic.shape[0] != times.size:
        raise ValueError(f"/{data_group}/B time axis length does not match /time/B")
    if len(spacecraft_ids) != positions.shape[0]:
        raise ValueError("/spacecraft/id length does not match /spacecraft/r/B")
    if not (
        np.all(np.isfinite(times))
        and np.all(np.isfinite(positions))
        and np.all(np.isfinite(magnetic))
    ):
        raise ValueError("Timesync inputs contain non-finite values")
    return times, positions, magnetic, spacecraft_ids


def _window_mask(
    times: np.ndarray,
    *,
    t0: float | None,
    t0_index: int | None,
    delta_t: float,
) -> tuple[float, np.ndarray]:
    if t0_index is not None:
        if not 0 <= t0_index < times.size:
            raise IndexError(f"t0_index {t0_index} is outside the B time axis")
        center_time = float(times[t0_index])
    else:
        assert t0 is not None
        center_time = float(t0)
    half_width = 0.5 * float(delta_t)
    mask = (times >= center_time - half_width) & (times <= center_time + half_width)
    if not np.any(mask):
        raise ValueError("Selected Timesync time window contains no samples")
    return center_time, mask


def _select_window(
    times: np.ndarray,
    positions_sc_time: np.ndarray,
    magnetic_time_sc: np.ndarray,
    *,
    t0: float | None,
    t0_index: int | None,
    delta_t: float,
    time_stride: int,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    time_stride = int(time_stride)
    if time_stride < 1:
        raise ValueError("timesync.sample_time_stride must be at least 1")
    t0_value, mask = _window_mask(times, t0=t0, t0_index=t0_index, delta_t=delta_t)
    indices = np.nonzero(mask)[0][::time_stride]
    if indices.size < 2:
        raise ValueError("Timesync requires at least two selected times")
    positions_time_sc = np.swapaxes(positions_sc_time, 0, 1)
    return (
        t0_value,
        np.asarray(times[indices], dtype=float),
        np.asarray(positions_time_sc[indices], dtype=float),
        np.asarray(magnetic_time_sc[indices], dtype=float),
    )


def _axis_from_source(
    barycenter_path: np.ndarray,
    axis_source: str,
    axis_vector: np.ndarray | None,
) -> np.ndarray:
    if axis_source == "user_vector":
        assert axis_vector is not None
        return _normalize(axis_vector, "user Timesync axis")
    displacement = barycenter_path[-1] - barycenter_path[0]
    if np.linalg.norm(displacement) <= 1.0e-12:
        warnings.warn(
            "Barycenter displacement is too small for a Timesync axis; using x-axis",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.array([1.0, 0.0, 0.0])
    return _normalize(displacement, "barycenter displacement")


def build_timesync_grid(
    spacecraft_positions: np.ndarray,
    times: np.ndarray,
    *,
    radius: float,
    n_axial: int,
    n_transverse: int,
    axial_padding: float = 0.0,
    radius_padding: float = 0.0,
    axis_source: str = "barycenter_path",
    axis_vector: np.ndarray | None = None,
) -> TimesyncGrid:
    """Build the local cylindrical grid used by Timesync.

    Coordinates are unwrapped physical coordinates. No periodic wrapping is
    applied here or in the Timesync fitter.
    """

    positions = np.asarray(spacecraft_positions, dtype=float)
    selected_times = np.asarray(times, dtype=float)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("spacecraft_positions must have shape (time, spacecraft, 3)")
    if positions.shape[0] != selected_times.size:
        raise ValueError("times length must match spacecraft_positions")
    requested_radius = float(radius)
    if requested_radius <= 0.0:
        raise ValueError("timesync cylinder radius must be positive")
    n_axial = int(n_axial)
    n_transverse = int(n_transverse)
    if n_axial < 1 or n_transverse < 2:
        raise ValueError("n_axial must be at least 1 and n_transverse at least 2")
    axial_padding = float(axial_padding)
    radius_padding = float(radius_padding)
    if axial_padding < 0.0 or radius_padding < 0.0:
        raise ValueError("Timesync grid padding values must be non-negative")

    barycenter_path = np.mean(positions, axis=1)
    e_parallel = _axis_from_source(barycenter_path, axis_source, axis_vector)
    e1, e2 = _perpendicular_basis(e_parallel)
    center = np.mean(barycenter_path, axis=0)
    basis = np.stack((e_parallel, e1, e2), axis=0)
    spacecraft_local = np.einsum("tsc,lc->tsl", positions - center, basis)
    spacecraft_axial = spacecraft_local[..., 0]
    transverse_radius = np.linalg.norm(spacecraft_local[..., 1:3], axis=-1)
    max_spacecraft_transverse_radius = float(np.max(transverse_radius))
    actual_radius = max(requested_radius, max_spacecraft_transverse_radius + radius_padding)
    axial_min = float(np.min(spacecraft_axial)) - axial_padding
    axial_max = float(np.max(spacecraft_axial)) + axial_padding
    if axial_min == axial_max:
        axial_min -= max(axial_padding, 1.0)
        axial_max += max(axial_padding, 1.0)
    axial_values = np.linspace(axial_min, axial_max, n_axial)
    transverse_values = np.linspace(-actual_radius, actual_radius, n_transverse)
    axial, transverse_u, transverse_v = np.meshgrid(
        axial_values, transverse_values, transverse_values, indexing="ij"
    )
    mask = np.sqrt(transverse_u**2 + transverse_v**2) <= actual_radius
    full_points = (
        center
        + axial[..., None] * e_parallel
        + transverse_u[..., None] * e1
        + transverse_v[..., None] * e2
    )
    local_coordinates = np.stack((axial, transverse_u, transverse_v), axis=-1)
    return TimesyncGrid(
        points=full_points[mask],
        full_points=full_points,
        local_coordinates=local_coordinates,
        axial=axial[mask],
        transverse_u=transverse_u[mask],
        transverse_v=transverse_v[mask],
        mask=mask,
        barycenter_path=barycenter_path,
        center=center,
        e_parallel=e_parallel,
        e1=e1,
        e2=e2,
        axial_values=axial_values,
        transverse_values=transverse_values,
        radius=actual_radius,
        requested_radius=requested_radius,
        radius_padding=radius_padding,
        max_spacecraft_transverse_radius=max_spacecraft_transverse_radius,
        axial_coverage_min=float(np.max(np.min(spacecraft_axial, axis=0))),
        axial_coverage_max=float(np.min(np.max(spacecraft_axial, axis=0))),
        n_axial=n_axial,
        n_transverse=n_transverse,
    )


def _is_monotonic(values: np.ndarray) -> bool:
    differences = np.diff(values)
    return bool(np.all(differences >= 0.0) or np.all(differences <= 0.0))


def _nearest_sync(
    axial_series: np.ndarray,
    local_series: np.ndarray,
    B_series: np.ndarray,
    target_axial: float,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], float, float]:
    nearest = int(np.argmin(np.abs(axial_series - target_axial)))
    mismatch = float(axial_series[nearest] - target_axial)
    return (
        local_series[nearest, 1:3],
        B_series[nearest],
        (nearest, nearest),
        np.nan,
        mismatch,
    )


def _linear_sync(
    axial_series: np.ndarray,
    local_series: np.ndarray,
    B_series: np.ndarray,
    target_axial: float,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], float, float] | None:
    if target_axial < np.min(axial_series) or target_axial > np.max(axial_series):
        return None
    for index in range(axial_series.size - 1):
        a0 = axial_series[index]
        a1 = axial_series[index + 1]
        if (a0 <= target_axial <= a1) or (a1 <= target_axial <= a0):
            denominator = a1 - a0
            if abs(float(denominator)) <= 1.0e-14:
                if abs(float(target_axial - a0)) > 1.0e-14:
                    continue
                fraction = 0.0
            else:
                fraction = float((target_axial - a0) / denominator)
            local = (1.0 - fraction) * local_series[index] + fraction * local_series[index + 1]
            magnetic = (1.0 - fraction) * B_series[index] + fraction * B_series[index + 1]
            return local[1:3], magnetic, (index, index + 1), fraction, 0.0
    return None


def synchronize_to_planes(
    local_positions: np.ndarray,
    magnetic: np.ndarray,
    axial_values: np.ndarray,
    *,
    time_interp: str = "linear",
    allow_nearest_fallback: bool = True,
) -> dict[str, np.ndarray]:
    """Synchronize every spacecraft to each target axial plane."""

    n_time, n_spacecraft, _ = local_positions.shape
    n_axial = axial_values.size
    sync_uv = np.full((n_axial, n_spacecraft, 2), np.nan, dtype=float)
    sync_B = np.full((n_axial, n_spacecraft, 3), np.nan, dtype=float)
    indices = np.full((n_axial, n_spacecraft, 2), -1, dtype=int)
    fractions = np.full((n_axial, n_spacecraft), np.nan, dtype=float)
    axial_mismatch = np.full((n_axial, n_spacecraft), np.nan, dtype=float)
    used = np.zeros((n_axial, n_spacecraft), dtype=bool)
    warned_nonmonotonic = False
    for spacecraft in range(n_spacecraft):
        axial_series = local_positions[:, spacecraft, 0]
        monotonic = _is_monotonic(axial_series)
        if not monotonic and time_interp == "linear":
            if not allow_nearest_fallback:
                raise ValueError(
                    f"Spacecraft {spacecraft} axial coordinate is not monotonic "
                    "over the selected Timesync window"
                )
            if not warned_nonmonotonic:
                warnings.warn(
                    "At least one spacecraft axial coordinate is not monotonic; "
                    "using nearest-neighbor synchronization for those series",
                    RuntimeWarning,
                    stacklevel=2,
                )
                warned_nonmonotonic = True
        for plane, target_axial in enumerate(axial_values):
            if target_axial < np.min(axial_series) or target_axial > np.max(axial_series):
                continue
            if time_interp == "nearest" or (not monotonic and allow_nearest_fallback):
                uv, field, pair, fraction, mismatch = _nearest_sync(
                    axial_series,
                    local_positions[:, spacecraft],
                    magnetic[:, spacecraft],
                    float(target_axial),
                )
            else:
                synced = _linear_sync(
                    axial_series,
                    local_positions[:, spacecraft],
                    magnetic[:, spacecraft],
                    float(target_axial),
                )
                if synced is None:
                    if not allow_nearest_fallback:
                        continue
                    uv, field, pair, fraction, mismatch = _nearest_sync(
                        axial_series,
                        local_positions[:, spacecraft],
                        magnetic[:, spacecraft],
                        float(target_axial),
                    )
                else:
                    uv, field, pair, fraction, mismatch = synced
            sync_uv[plane, spacecraft] = uv
            sync_B[plane, spacecraft] = field
            indices[plane, spacecraft] = pair
            fractions[plane, spacecraft] = fraction
            axial_mismatch[plane, spacecraft] = mismatch
            used[plane, spacecraft] = True
    del n_time
    return {
        "uv": sync_uv,
        "B": sync_B,
        "indices": indices,
        "fractions": fractions,
        "axial_mismatch": axial_mismatch,
        "used": used,
    }


def idw_interpolate_plane(
    target_uv: np.ndarray,
    source_uv: np.ndarray,
    source_B: np.ndarray,
    *,
    power: float = 2.0,
    eps: float = 1.0e-12,
) -> np.ndarray:
    """Vector 2D inverse-distance-weighted interpolation for one plane."""

    targets = np.asarray(target_uv, dtype=float)
    uv = np.asarray(source_uv, dtype=float)
    values = np.asarray(source_B, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("target_uv must have shape (N, 2)")
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("source_uv must have shape (M, 2)")
    if values.ndim != 2 or values.shape != (uv.shape[0], 3):
        raise ValueError("source_B must have shape (M, 3)")
    if power <= 0.0:
        raise ValueError("IDW power must be positive")
    result = np.full((targets.shape[0], 3), np.nan, dtype=float)
    if uv.shape[0] == 0:
        return result
    distances = np.linalg.norm(targets[:, None, :] - uv[None, :, :], axis=-1)
    exact = distances <= eps
    for index in range(targets.shape[0]):
        exact_sources = np.nonzero(exact[index])[0]
        if exact_sources.size:
            result[index] = values[exact_sources[0]]
            continue
        weights = 1.0 / np.power(distances[index], power)
        weight_sum = float(np.sum(weights))
        if weight_sum > 0.0 and np.isfinite(weight_sum):
            result[index] = weights @ values / weight_sum
    return result


def reconstruct_timesync(
    positions: np.ndarray,
    magnetic: np.ndarray,
    grid: TimesyncGrid,
    *,
    time_interp: str = "linear",
    allow_nearest_fallback: bool = True,
    idw_power: float = 2.0,
    idw_eps: float = 1.0e-12,
    min_spacecraft_per_plane: int = 4,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Synchronize spacecraft samples and IDW-interpolate the Timesync volume."""

    basis = np.stack((grid.e_parallel, grid.e1, grid.e2), axis=0)
    local_positions = np.einsum("tsc,lc->tsl", positions - grid.center, basis)
    sync = synchronize_to_planes(
        local_positions,
        magnetic,
        grid.axial_values,
        time_interp=time_interp,
        allow_nearest_fallback=allow_nearest_fallback,
    )
    full_B = np.full(grid.mask.shape + (3,), np.nan, dtype=float)
    n_spacecraft_used = np.sum(sync["used"], axis=1).astype(int)
    insufficient = n_spacecraft_used < int(min_spacecraft_per_plane)
    if np.any(insufficient):
        warnings.warn(
            "Timesync left one or more axial planes masked because too few "
            "spacecraft were available",
            RuntimeWarning,
            stacklevel=2,
        )
    transverse_grid = grid.transverse_values
    transverse_u, transverse_v = np.meshgrid(transverse_grid, transverse_grid, indexing="ij")
    target_uv = np.column_stack((transverse_u.reshape(-1), transverse_v.reshape(-1)))
    for plane in range(grid.n_axial):
        if insufficient[plane]:
            continue
        valid = sync["used"][plane]
        plane_values = idw_interpolate_plane(
            target_uv,
            sync["uv"][plane, valid],
            sync["B"][plane, valid],
            power=idw_power,
            eps=idw_eps,
        ).reshape(grid.n_transverse, grid.n_transverse, 3)
        full_B[plane, grid.mask[plane]] = plane_values[grid.mask[plane]]
    mismatch_abs = np.abs(sync["axial_mismatch"])
    max_mismatch = np.full(grid.n_axial, np.nan, dtype=float)
    mean_mismatch = np.full(grid.n_axial, np.nan, dtype=float)
    for plane in range(grid.n_axial):
        finite = mismatch_abs[plane, np.isfinite(mismatch_abs[plane])]
        if finite.size:
            max_mismatch[plane] = float(np.max(finite))
            mean_mismatch[plane] = float(np.mean(finite))
    diagnostics = {
        "local_positions": local_positions,
        "synchronized_uv": sync["uv"],
        "synchronized_B": sync["B"],
        "nearest_or_bracketing_indices": sync["indices"],
        "interpolation_fraction": sync["fractions"],
        "axial_mismatch": sync["axial_mismatch"],
        "n_spacecraft_used_per_plane": n_spacecraft_used,
        "max_axial_mismatch_per_plane": max_mismatch,
        "mean_axial_mismatch_per_plane": mean_mismatch,
    }
    return full_B[grid.mask], diagnostics


def compute_timesync_reconstruction(
    input_path: str | Path, config: dict[str, Any]
) -> dict[str, Any]:
    """Compute Timesync reconstruction from HDF5 spacecraft measurements."""

    settings = _settings(config)
    times, positions_sc_time, magnetic_time_sc, spacecraft_ids = _read_measurements(
        input_path,
        data_group=settings["data_group"],
        mc_index=settings["mc_index"],
    )
    t0, selected_times, selected_positions, selected_B = _select_window(
        times,
        positions_sc_time,
        magnetic_time_sc,
        t0=settings["t0"],
        t0_index=settings["t0_index"],
        delta_t=settings["delta_t"],
        time_stride=settings["sample_time_stride"],
    )
    grid = build_timesync_grid(
        selected_positions,
        selected_times,
        radius=settings["grid"]["radius"],
        n_axial=settings["grid"]["n_axial"],
        n_transverse=settings["grid"]["n_transverse"],
        axial_padding=settings["grid"]["axial_padding"],
        radius_padding=settings["grid"]["radius_padding"],
        axis_source=settings["axis_source"],
        axis_vector=settings["axis_vector"],
    )
    reconstructed, diagnostics = reconstruct_timesync(
        selected_positions,
        selected_B,
        grid,
        time_interp=settings["time_interp"],
        allow_nearest_fallback=settings["allow_nearest_fallback"],
        idw_power=settings["idw_power"],
        idw_eps=settings["idw_eps"],
        min_spacecraft_per_plane=settings["min_spacecraft_per_plane"],
    )
    return {
        "metadata": {
            "method": "timesync",
            "input_path": str(Path(input_path)),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "assumption": "static magnetic field during selected time window",
            "data_group": settings["data_group"],
            "mc_index": "null" if settings["mc_index"] is None else settings["mc_index"],
            "fitter_uses": "spacecraft times, positions, and selected magnetic-field samples only",
            "coordinate_policy": "unwrapped physical coordinates; no periodic wrapping",
        },
        "settings": settings,
        "t0": t0,
        "delta_t": settings["delta_t"],
        "selected_times": selected_times,
        "spacecraft_ids": spacecraft_ids,
        "sample_trajectory": selected_positions,
        "sample_positions": selected_positions.reshape(-1, 3),
        "sample_B": selected_B.reshape(-1, 3),
        "grid": grid,
        "B_reconstructed": reconstructed,
        "diagnostics": diagnostics,
    }


def _write_metadata(group: h5py.Group, metadata: dict[str, Any]) -> None:
    for key, value in metadata.items():
        if isinstance(value, dict):
            _write_metadata(group.create_group(key), value)
        elif isinstance(value, np.ndarray):
            group.create_dataset(key, data=value)
        elif value is None:
            group.attrs[key] = "null"
        else:
            group.attrs[key] = value


def _grid_metadata(grid: TimesyncGrid) -> dict[str, Any]:
    return {
        "center": grid.center,
        "e_parallel": grid.e_parallel,
        "e1": grid.e1,
        "e2": grid.e2,
        "barycenter_start": grid.barycenter_path[0],
        "barycenter_end": grid.barycenter_path[-1],
        "radius": grid.radius,
        "requested_radius": grid.requested_radius,
        "radius_padding": grid.radius_padding,
        "max_spacecraft_transverse_radius": grid.max_spacecraft_transverse_radius,
        "axial_coverage_min": grid.axial_coverage_min,
        "axial_coverage_max": grid.axial_coverage_max,
        "n_axial": grid.n_axial,
        "n_transverse": grid.n_transverse,
    }


def write_timesync_reconstruction(result: dict[str, Any], output_path: str | Path) -> Path:
    """Write Timesync output in the common reconstruction HDF5 layout."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    settings = result["settings"]
    grid: TimesyncGrid = result["grid"]
    metadata = {
        **result["metadata"],
        "time_interp": settings["time_interp"],
        "allow_nearest_fallback": settings["allow_nearest_fallback"],
        "idw_power": settings["idw_power"],
        "axis_source": settings["axis_source"],
        "sample_time_stride": settings["sample_time_stride"],
        "coordinate_units": settings["coordinate_units"],
        "min_spacecraft_per_plane": settings["min_spacecraft_per_plane"],
        "grid": _grid_metadata(grid),
    }
    with h5py.File(output, "w") as handle:
        _write_metadata(handle.create_group("metadata"), metadata)
        samples = handle.create_group("samples")
        samples.create_dataset("selected_times", data=result["selected_times"])
        samples.create_dataset("time", data=result["selected_times"])
        samples.create_dataset("positions", data=result["sample_positions"], compression="gzip")
        samples.create_dataset("position", data=result["sample_positions"], compression="gzip")
        samples.create_dataset("trajectory", data=result["sample_trajectory"], compression="gzip")
        samples.create_dataset("B", data=result["sample_B"], compression="gzip")
        samples.create_dataset(
            "spacecraft_id",
            data=np.asarray(result["spacecraft_ids"], dtype=h5py.string_dtype("utf-8")),
        )

        grid_group = handle.create_group("grid")
        grid_group.create_dataset("points", data=grid.points, compression="gzip")
        grid_group.create_dataset("full_points", data=grid.full_points, compression="gzip")
        grid_group.create_dataset(
            "dimensions",
            data=np.asarray((grid.n_axial, grid.n_transverse, grid.n_transverse)),
        )
        grid_group.create_dataset(
            "local_coordinates", data=grid.local_coordinates, compression="gzip"
        )
        grid_group.create_dataset("local_axial", data=grid.axial, compression="gzip")
        grid_group.create_dataset("local_u", data=grid.transverse_u, compression="gzip")
        grid_group.create_dataset("local_v", data=grid.transverse_v, compression="gzip")
        grid_group.create_dataset("cylinder_mask", data=grid.mask, compression="gzip")
        grid_group.create_dataset("axial_values", data=grid.axial_values)
        grid_group.create_dataset("transverse_values", data=grid.transverse_values)
        grid_group.create_dataset("barycenter_path", data=grid.barycenter_path)
        grid_group.create_dataset("center", data=grid.center)
        grid_group.create_dataset(
            "basis_vectors", data=np.stack((grid.e_parallel, grid.e1, grid.e2), axis=0)
        )

        reconstruction = handle.create_group("reconstruction")
        reconstruction.create_dataset("B", data=result["B_reconstructed"], compression="gzip")
        reconstruction.create_dataset(
            "Bmag",
            data=np.linalg.norm(result["B_reconstructed"], axis=-1),
            compression="gzip",
        )
        reconstruction.attrs["components"] = np.asarray(
            COMPONENT_NAMES, dtype=h5py.string_dtype("utf-8")
        )

        timesync = handle.create_group("timesync")
        timesync.attrs["time_interp"] = settings["time_interp"]
        timesync.attrs["idw_power"] = settings["idw_power"]
        timesync.attrs["axis_source"] = settings["axis_source"]
        timesync.create_dataset("e_parallel", data=grid.e_parallel)
        timesync.create_dataset("e1", data=grid.e1)
        timesync.create_dataset("e2", data=grid.e2)
        timesync.create_dataset("origin", data=grid.center)
        for name, values in result["diagnostics"].items():
            timesync.create_dataset(name, data=values, compression="gzip")
        handle.attrs["t0"] = result["t0"]
        handle.attrs["delta_t"] = result["delta_t"]
    return output


@register_analysis
class TimesyncAnalysis(AnalysisMethod):
    """Analysis-registry adapter for Timesync magnetic reconstruction."""

    name = "timesync"
    config_section = "timesync"

    def run(self, input_path: str | Path) -> dict[str, Any]:
        return compute_timesync_reconstruction(input_path, self.config)

    def write_result(self, result: dict[str, Any], output_path: str | Path) -> Path:
        output = write_timesync_reconstruction(result, output_path)
        reference_settings = result["settings"]["reference"]
        if reference_settings["enabled"]:
            generation_config = reference_settings["generation_config"]
            if generation_config is None:
                raise ValueError(
                    "timesync.reference.generation_config is required when "
                    "reference.enabled is true"
                )
            from hs_observatory.analysis.rbf_reconstruction import (
                add_simulation_reference_to_rbf_output,
            )

            observation_path = (
                reference_settings["observation_path"] or result["metadata"]["input_path"]
            )
            add_simulation_reference_to_rbf_output(output, observation_path, generation_config)
        plot_settings = result["settings"]["plot"]
        if plot_settings["enabled"]:
            from hs_observatory.analysis.rbf_reconstruction import plot_rbf_reconstruction

            plot_path = Path(plot_settings["path"] or output.with_name(f"{output.stem}.png"))
            plot_rbf_reconstruction(
                output,
                plot_path,
                dpi=plot_settings["dpi"],
                magnitude_cmap=plot_settings["magnitude_cmap"],
                error_cmap=plot_settings["error_cmap"],
                component_mode=plot_settings["component_mode"],
                background_mode=plot_settings["background_mode"],
                background_vector=plot_settings["background_vector"],
            )
        averaged_error_plot = result["settings"]["averaged_error_plot"]
        if averaged_error_plot["enabled"]:
            from hs_observatory.analysis.rbf_reconstruction import (
                plot_rbf_broeren_uv_error,
            )

            plot_path = Path(
                averaged_error_plot["path"] or output.with_name(f"{output.stem}_uv_error.png")
            )
            plot_rbf_broeren_uv_error(
                output,
                plot_path,
                dpi=averaged_error_plot["dpi"],
                error_cmap=averaged_error_plot["error_cmap"],
                capped=averaged_error_plot["capped"],
            )
        return output
