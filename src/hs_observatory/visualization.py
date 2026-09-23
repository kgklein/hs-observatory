"""Default visual checks for generated truth data and spacecraft geometry."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence
import warnings

import h5py
import matplotlib

# Generation is commonly run on headless systems and batch nodes.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np

from hs_observatory.analysis.derived.elsasser import (
    compute_density_normalization,
    compute_elsasser,
)
from hs_observatory.analysis.derived.elsasser import magnetic_field_to_alfven_units
from hs_observatory.io.hdf5 import read_observations


# Palette adapted from sparse-field-reconstruction. Mapping by name preserves
# the established colors even though hs-observatory stores the Hub first.
SPACECRAFT_COLORS = {
    "H": "#CC79A7",
    "N1": "#56B4E9",
    "N2": "#E69F00",
    "N3": "#8E4D4D",
    "N4": "#888888",
    "N5": "#F0E442",
    "N6": "#D55E00",
    "N7": "#009E73",
    "N8": "#8D00B2",
}

EARTH_RADIUS_KM = 6371.0
REGULAR_POLYHEDRON_Q_LIMIT = 0.6
KINETIC_SCALE_KM = 100.0
MHD_SCALE_KM = 1200.0
LAMBDA_SCALE_COLORS = {
    "kinetic": "#f6e58d",
    "transition": "#d7bde2",
    "mhd": "#d7ccc8",
}
FORMISANO_COEFFICIENTS = {
    "a11": 0.016,
    "a22": 1.0,
    "a33": 0.840,
    "a12": 0.171,
    "a14": 39.269,
    "a24": -1.675,
    "a44": -564.29,
}


def spacecraft_color(spacecraft_id: str) -> str:
    try:
        return SPACECRAFT_COLORS[spacecraft_id]
    except KeyError as exc:
        raise ValueError(f"No hardcoded HelioSwarm color for spacecraft {spacecraft_id!r}") from exc


def plot_truth_timeseries(
    time: Mapping[str, np.ndarray],
    truth: Mapping[str, np.ndarray],
    spacecraft_ids: Sequence[str],
    velocity_frame: str = "plasma",
):
    """Return a seven-panel figure of true n, v, and B measurements."""

    fig, axes = plt.subplots(4, 2, figsize=(15, 12), constrained_layout=True)
    panel_specs = (
        (axes[0, 0], "n", None, r"$n$ (cm$^{-3}$)"),
        (axes[1, 0], "v", 0, r"$v_x$ (km s$^{-1}$)"),
        (axes[2, 0], "v", 1, r"$v_y$ (km s$^{-1}$)"),
        (axes[3, 0], "v", 2, r"$v_z$ (km s$^{-1}$)"),
        (axes[1, 1], "B", 0, r"$B_x$ (nT)"),
        (axes[2, 1], "B", 1, r"$B_y$ (nT)"),
        (axes[3, 1], "B", 2, r"$B_z$ (nT)"),
    )
    for ax, quantity, component, ylabel in panel_specs:
        values = np.asarray(truth[quantity], dtype=float)
        if component is not None:
            values = values[..., component]
        for index, spacecraft_id in enumerate(spacecraft_ids):
            ax.plot(
                time[quantity],
                values[:, index],
                color=spacecraft_color(spacecraft_id),
                label=spacecraft_id,
                linewidth=1.4,
            )
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    axes[3, 0].set_xlabel("Time (s)")
    axes[3, 1].set_xlabel("Time (s)")
    axes[0, 1].axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 1].legend(handles, labels, loc="center", ncol=3, frameon=False)
    fig.suptitle(
        "HelioSwarm Synthetic Observatory: True Measurements "
        f"({velocity_frame.capitalize()}-Frame Velocity)"
    )
    return fig, axes


def plot_observed_timeseries(
    time: Mapping[str, np.ndarray],
    truth: Mapping[str, np.ndarray],
    observed: Mapping[str, np.ndarray],
    spacecraft_ids: Sequence[str],
    mc_index: int,
):
    """Plot one observed realization in color over nominal truth in grayscale."""

    fig, axes = plt.subplots(4, 2, figsize=(15, 12), constrained_layout=True)
    panel_specs = (
        (axes[0, 0], "n", None, r"$n$ (cm$^{-3}$)"),
        (axes[1, 0], "v", 0, r"$v_x$ (km s$^{-1}$)"),
        (axes[2, 0], "v", 1, r"$v_y$ (km s$^{-1}$)"),
        (axes[3, 0], "v", 2, r"$v_z$ (km s$^{-1}$)"),
        (axes[1, 1], "B", 0, r"$B_x$ (nT)"),
        (axes[2, 1], "B", 1, r"$B_y$ (nT)"),
        (axes[3, 1], "B", 2, r"$B_z$ (nT)"),
    )
    gray_colors = plt.cm.Greys(np.linspace(0.35, 0.75, len(spacecraft_ids)))
    for ax, quantity, component, ylabel in panel_specs:
        true_values = np.asarray(truth[quantity], dtype=float)
        measured_values = np.asarray(observed[quantity][mc_index], dtype=float)
        if component is not None:
            true_values = true_values[..., component]
            measured_values = measured_values[..., component]
        for index, _spacecraft_id in enumerate(spacecraft_ids):
            ax.plot(
                time[quantity],
                true_values[:, index],
                color=gray_colors[index],
                linewidth=2.0,
                alpha=0.65,
                zorder=1,
            )
        for index, spacecraft_id in enumerate(spacecraft_ids):
            ax.plot(
                time[quantity],
                measured_values[:, index],
                color=spacecraft_color(spacecraft_id),
                label=spacecraft_id,
                linewidth=1.1,
                zorder=2,
            )
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    axes[3, 0].set_xlabel("Nominal time (s)")
    axes[3, 1].set_xlabel("Nominal time (s)")
    axes[0, 1].axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="#777777", linewidth=2.0, label="Nominal truth"))
    labels.append("Nominal truth")
    axes[0, 1].legend(handles, labels, loc="center", ncol=3, frameon=False)
    fig.suptitle(f"Observed Measurements with Nominal Truth: MC {mc_index}")
    return fig, axes


def _density_on_target_time(
    source_time: np.ndarray,
    density: np.ndarray,
    target_time: np.ndarray,
) -> np.ndarray:
    """Interpolate density per spacecraft onto the magnetic-field time grid."""

    source_time = np.asarray(source_time, dtype=float)
    target_time = np.asarray(target_time, dtype=float)
    density = np.asarray(density, dtype=float)
    if density.ndim != 2 or density.shape[0] != len(source_time):
        raise ValueError("density must have shape (time_n, spacecraft)")
    return np.stack(
        [
            np.interp(target_time, source_time, density[:, spacecraft_index])
            for spacecraft_index in range(density.shape[1])
        ],
        axis=1,
    )


def _alfven_velocity_units(
    time: Mapping[str, np.ndarray],
    values: Mapping[str, np.ndarray],
) -> np.ndarray:
    density_at_B = _density_on_target_time(time["n"], values["n"], time["B"])
    return magnetic_field_to_alfven_units(values["B"], density_at_B)


def _component_ylim(*arrays: np.ndarray) -> tuple[float, float]:
    finite = np.concatenate([np.ravel(array[np.isfinite(array)]) for array in arrays])
    if finite.size == 0:
        return -1.0, 1.0
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if lo == hi:
        pad = max(abs(lo) * 0.05, 1.0)
    else:
        pad = 0.06 * (hi - lo)
    return lo - pad, hi + pad


def _select_observed_member(values: np.ndarray, truth_ndim: int, member_index: int) -> np.ndarray:
    """Return one observed member, accepting legacy files without an MC axis."""

    array = np.asarray(values, dtype=float)
    if array.ndim == truth_ndim + 1:
        if not 0 <= member_index < array.shape[0]:
            raise IndexError(
                f"member_index {member_index} is outside observed data with "
                f"{array.shape[0]} members"
            )
        return array[member_index]
    if array.ndim == truth_ndim:
        if member_index != 0:
            raise IndexError("member_index must be 0 when observed data has no MC axis")
        return array
    raise ValueError("observed data has an unsupported shape")


def _figure2_convert_values(
    values: Mapping[str, np.ndarray],
    *,
    input_units: str,
) -> dict[str, np.ndarray]:
    """Convert HDF5 variables to plotting/helper units.

    ``input_units='si'`` assumes density in m^-3, velocity in m/s, and magnetic
    field in T. ``input_units='repo'`` uses the package-native generated
    observation units: cm^-3, km/s, and nT.
    """

    if input_units == "si":
        return {
            "n": np.asarray(values["n"], dtype=float) * 1.0e-6,
            "v": np.asarray(values["v"], dtype=float) * 1.0e-3,
            "B": np.asarray(values["B"], dtype=float) * 1.0e9,
        }
    if input_units == "repo":
        return {name: np.asarray(values[name], dtype=float) for name in ("n", "v", "B")}
    raise ValueError("input_units must be 'si' or 'repo'")


def _figure2_velocity_frame(input_path: str | Path) -> str:
    """Return the stored observed velocity frame, defaulting old files to plasma."""

    with h5py.File(input_path, "r") as handle:
        if "metadata" in handle and "velocity_observed_frame" in handle["metadata"].attrs:
            value = handle["metadata"].attrs["velocity_observed_frame"]
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            frame = str(value).lower()
            if frame in {"plasma", "spacecraft"}:
                return frame
        if "observed/v" in handle:
            value = handle["observed/v"].attrs.get("frame")
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            frame = str(value).lower()
            if frame in {"plasma", "spacecraft"}:
                return frame
    return "plasma"


def _figure2_spacecraft_velocity(
    input_path: str | Path,
    time_v: np.ndarray,
    n_spacecraft: int,
) -> np.ndarray | None:
    """Return spacecraft velocity in km/s on the velocity time grid when available."""

    from hs_observatory.analysis.elsasser import _stored_spacecraft_velocity

    with h5py.File(input_path, "r") as handle:
        return _stored_spacecraft_velocity(handle, np.asarray(time_v, dtype=float), n_spacecraft)


def _figure2_velocity_in_frame(
    input_path: str | Path,
    velocity_km_s: np.ndarray,
    time_v: np.ndarray,
    *,
    source_frame: str,
    target_frame: str,
) -> np.ndarray:
    """Convert velocity between plasma and spacecraft frames for plotting."""

    if source_frame == target_frame:
        return np.asarray(velocity_km_s, dtype=float)
    if source_frame not in {"plasma", "spacecraft"} or target_frame not in {"plasma", "spacecraft"}:
        raise ValueError("velocity frames must be 'plasma' or 'spacecraft'")
    velocity = np.asarray(velocity_km_s, dtype=float)
    spacecraft_velocity = _figure2_spacecraft_velocity(input_path, time_v, velocity.shape[1])
    if spacecraft_velocity is None:
        raise ValueError(
            f"Cannot convert velocity from {source_frame} to {target_frame}; "
            "spacecraft velocity is unavailable in the HDF5 file or metadata"
        )
    if source_frame == "spacecraft" and target_frame == "plasma":
        return velocity + spacecraft_velocity
    return velocity - spacecraft_velocity


def _figure2_elsasser_plus(
    time: Mapping[str, np.ndarray],
    values: Mapping[str, np.ndarray],
    *,
    density_method: str = "global_per_spacecraft",
    mean_method: str = "global_per_spacecraft",
) -> tuple[np.ndarray, np.ndarray]:
    """Return z+ and its time grid using the established Elsasser convention."""

    from hs_observatory.analysis.elsasser import _interpolate

    time_n = np.asarray(time["n"], dtype=float)
    time_v = np.asarray(time["v"], dtype=float)
    time_B = np.asarray(time["B"], dtype=float)
    n = np.asarray(values["n"], dtype=float)
    v = np.asarray(values["v"], dtype=float)
    B = np.asarray(values["B"], dtype=float)
    if n.ndim != 2 or v.ndim != 3 or B.ndim != 3 or v.shape[-1] != 3 or B.shape[-1] != 3:
        raise ValueError("n, v, and B must have shapes (time, sc), (time, sc, 3), (time, sc, 3)")
    dt_n = float(np.mean(np.diff(time_n))) if time_n.size > 1 else np.inf
    dt_v = float(np.mean(np.diff(time_v))) if time_v.size > 1 else np.inf
    output_time = time_v if np.isclose(dt_n, dt_v, rtol=1.0e-7) or dt_v > dt_n else time_n
    n_aligned = _interpolate(n, time_n, output_time, "n")
    v_aligned = _interpolate(v, time_v, output_time, "v")
    B_aligned = _interpolate(B, time_B, output_time, "B")
    normalization = compute_density_normalization(n_aligned, density_method)
    b = magnetic_field_to_alfven_units(B_aligned, normalization)
    return compute_elsasser(v_aligned, b, mean_method)["z_plus"], output_time


def _relative_time(time: np.ndarray) -> np.ndarray:
    values = np.asarray(time, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("time arrays must be one-dimensional and nonempty")
    return values - values[0]


def _read_figure2_column(
    input_path: str | Path,
    *,
    member_index: int,
    input_units: str,
    velocity_frame: str,
) -> dict[str, Any]:
    """Read one Figure 2 column from a generated-observation HDF5 file."""

    data = read_observations(input_path)
    time = {name: np.asarray(values, dtype=float) for name, values in data["time"].items()}
    truth = _figure2_convert_values(data["truth"], input_units=input_units)
    observed_raw = {
        name: _select_observed_member(
            np.asarray(data["observed"][name], dtype=float),
            np.asarray(data["truth"][name]).ndim,
            member_index,
        )
        for name in ("n", "v", "B")
    }
    observed = _figure2_convert_values(observed_raw, input_units=input_units)
    observed_frame = _figure2_velocity_frame(input_path)
    truth["v"] = _figure2_velocity_in_frame(
        input_path,
        truth["v"],
        time["v"],
        source_frame="plasma",
        target_frame=velocity_frame,
    )
    observed["v"] = _figure2_velocity_in_frame(
        input_path,
        observed["v"],
        time["v"],
        source_frame=observed_frame,
        target_frame=velocity_frame,
    )
    truth_z_plus, truth_z_time = _figure2_elsasser_plus(time, truth)
    observed_z_plus, observed_z_time = _figure2_elsasser_plus(time, observed)
    return {
        "time": time,
        "spacecraft_ids": list(data["spacecraft"]["id"]),
        "truth": truth,
        "observed": observed,
        "truth_z_plus": truth_z_plus,
        "truth_z_time": truth_z_time,
        "observed_z_plus": observed_z_plus,
        "observed_z_time": observed_z_time,
        "velocity_frame": velocity_frame,
        "observed_velocity_frame_input": observed_frame,
    }


def _visible_values(
    time: np.ndarray,
    values: np.ndarray,
    *,
    start_time_s: float | None,
    stop_time_s: float | None,
) -> np.ndarray:
    relative = _relative_time(time)
    mask = np.ones(relative.shape, dtype=bool)
    if start_time_s is not None:
        mask &= relative >= start_time_s
    if stop_time_s is not None:
        mask &= relative <= stop_time_s
    selected = np.asarray(values, dtype=float)[mask]
    return selected if selected.size else np.asarray(values, dtype=float)


def _set_shared_pair_limits(
    axis_a,
    axis_b,
    time_a: np.ndarray,
    values_a: np.ndarray,
    time_b: np.ndarray,
    values_b: np.ndarray,
    *,
    start_time_s: float | None,
    stop_time_s: float | None,
) -> None:
    limits = _component_ylim(
        _visible_values(time_a, values_a, start_time_s=start_time_s, stop_time_s=stop_time_s),
        _visible_values(time_b, values_b, start_time_s=start_time_s, stop_time_s=stop_time_s),
    )
    axis_a.set_ylim(*limits)
    axis_b.set_ylim(*limits)


def _plot_figure2_column(
    axes: np.ndarray,
    column: dict[str, Any],
    title: str,
    *,
    panel_start: int,
    start_time_s: float | None,
    stop_time_s: float | None,
) -> None:
    panel_labels = "abcdefghijkl"
    specs = (
        ("truth", "v", column["time"]["v"], column["truth"]["v"], r"Truth $v_x$ (km s$^{-1}$)"),
        (
            "observed",
            "v",
            column["time"]["v"],
            column["observed"]["v"],
            r"Measured $v_x$ (km s$^{-1}$)",
        ),
        ("truth", "B", column["time"]["B"], column["truth"]["B"], r"Truth $B_x$ (nT)"),
        (
            "observed",
            "B",
            column["time"]["B"],
            column["observed"]["B"],
            r"Measured $B_x$ (nT)",
        ),
        (
            "truth",
            "z_plus",
            column["truth_z_time"],
            column["truth_z_plus"],
            r"Truth $z_x^+$ (km s$^{-1}$)",
        ),
        (
            "observed",
            "z_plus",
            column["observed_z_time"],
            column["observed_z_plus"],
            r"Measured $z_x^+$ (km s$^{-1}$)",
        ),
    )
    fallback = plt.cm.tab10(np.linspace(0.0, 0.9, len(column["spacecraft_ids"])))
    for row, (_kind, _quantity, time, values, ylabel) in enumerate(specs):
        axis = axes[row]
        component = np.asarray(values, dtype=float)[:, :, 0]
        for spacecraft_index, spacecraft_id in enumerate(column["spacecraft_ids"]):
            axis.plot(
                _relative_time(time),
                component[:, spacecraft_index],
                color=SPACECRAFT_COLORS.get(spacecraft_id, fallback[spacecraft_index]),
                linewidth=0.95,
            )
        axis.set_ylabel(ylabel, fontsize=14)
        axis.tick_params(axis="both", labelsize=14)
        axis.grid(alpha=0.22)
        axis.text(
            0.015,
            0.88,
            f"({panel_labels[panel_start + row]})",
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
        )
        if row != len(specs) - 1:
            axis.tick_params(labelbottom=False)
    axes[0].set_title(title, fontsize=16)
    axes[-1].set_xlabel(r"$t - t_0$ (s)", fontsize=14)
    if start_time_s is not None or stop_time_s is not None:
        axes[-1].set_xlim(left=start_time_s, right=stop_time_s)
    _set_shared_pair_limits(
        axes[0],
        axes[1],
        specs[0][2],
        specs[0][3][:, :, 0],
        specs[1][2],
        specs[1][3][:, :, 0],
        start_time_s=start_time_s,
        stop_time_s=stop_time_s,
    )
    _set_shared_pair_limits(
        axes[2],
        axes[3],
        specs[2][2],
        specs[2][3][:, :, 0],
        specs[3][2],
        specs[3][3][:, :, 0],
        start_time_s=start_time_s,
        stop_time_s=stop_time_s,
    )
    _set_shared_pair_limits(
        axes[4],
        axes[5],
        specs[4][2],
        specs[4][3][:, :, 0],
        specs[5][2],
        specs[5][3][:, :, 0],
        start_time_s=start_time_s,
        stop_time_s=stop_time_s,
    )


def plot_figure2_timeseries(
    malara_path: str | Path,
    simulation_path: str | Path,
    output_path: str | Path,
    *,
    member_index: int = 0,
    malara_title: str = "Prescribed Intermittent Turbulence",
    simulation_title: str = "Gkyell Five-Moment Simulation",
    input_units: str = "si",
    velocity_frame: str = "plasma",
    start_time_s: float | None = None,
    stop_time_s: float | None = None,
    dpi: int = 300,
) -> Path:
    """Write the two-column paper Figure 2 time-series summary.

    The inputs are generated-observation HDF5 files with ``time``, ``truth``,
    ``observed``, and ``spacecraft/id`` entries compatible with
    :func:`read_observations`. The measured panels use one observed
    MC realization, default member 0. ``z_x^+`` is constructed from the shared
    Elsasser helpers and therefore uses the package mean-subtraction convention.

    Example:
    ``hsobs-plot-figure2 --malara malara.h5 --simulation sim.h5 --output figure2.png``
    """

    if velocity_frame not in {"plasma", "spacecraft"}:
        raise ValueError("velocity_frame must be 'plasma' or 'spacecraft'")
    if start_time_s is not None and stop_time_s is not None and stop_time_s <= start_time_s:
        raise ValueError("stop_time_s must be greater than start_time_s")
    left = _read_figure2_column(
        malara_path,
        member_index=member_index,
        input_units=input_units,
        velocity_frame=velocity_frame,
    )
    right = _read_figure2_column(
        simulation_path,
        member_index=member_index,
        input_units=input_units,
        velocity_frame=velocity_frame,
    )
    figure, axes = plt.subplots(
        6,
        2,
        figsize=(12.5, 13.0),
        sharex="col",
        constrained_layout=True,
    )
    _plot_figure2_column(
        axes[:, 0],
        left,
        malara_title,
        panel_start=0,
        start_time_s=start_time_s,
        stop_time_s=stop_time_s,
    )
    _plot_figure2_column(
        axes[:, 1],
        right,
        simulation_title,
        panel_start=6,
        start_time_s=start_time_s,
        stop_time_s=stop_time_s,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return output


def plot_intermittency_timeseries(
    malara_path: str | Path,
    output_path: str | Path,
    *,
    member_index: int = 0,
    input_units: str = "repo",
    velocity_frame: str = "plasma",
    start_time_s: float = 0.0,
    stop_time_s: float = 300.0,
    dpi: int = 300,
) -> tuple[Path, Path]:
    """Save a Malara-only 4-by-2 figure, with measured panels below truth.

    Interval bounds are seconds from the earliest native time-grid start.
    Elsasser normalization and mean subtraction use the full stored interval,
    as in Figure 2; measured quantities use one MC member without averaging.
    """

    if not np.isfinite([start_time_s, stop_time_s]).all() or stop_time_s <= start_time_s:
        raise ValueError("Finite interval bounds must satisfy stop_time_s > start_time_s")
    if velocity_frame not in {"plasma", "spacecraft"}:
        raise ValueError("velocity_frame must be 'plasma' or 'spacecraft'")
    data = _read_figure2_column(
        malara_path,
        member_index=member_index,
        input_units=input_units,
        velocity_frame=velocity_frame,
    )
    origin = min(float(data["time"][name][0]) for name in ("n", "v", "B"))
    t0 = origin + start_time_s
    t1 = origin + stop_time_s
    specs = (
        (0, 0, "v", r"$v_x$ (km s$^{-1}$)"),
        (0, 1, "B", r"$B_x$ (nT)"),
        (2, 0, "n", r"$n$ (cm$^{-3}$)"),
        (2, 1, "z_plus", r"$z_x^+$ (km s$^{-1}$)"),
    )
    with plt.rc_context({"font.size": 12, "axes.labelsize": 13, "axes.linewidth": 0.7}):
        figure, axes = plt.subplots(
            4, 2, figsize=(13.0, 9.0), sharex=True, constrained_layout=True
        )
        figure.get_layout_engine().set(h_pad=0.025, w_pad=0.04, hspace=0.025, wspace=0.04)
        try:
            for row, col, quantity, label in specs:
                pair_values = []
                for offset, (kind, prefix) in enumerate((("truth", "True"),
                                                       ("observed", "Measured"))):
                    if quantity == "z_plus":
                        time = data[f"{kind}_z_time"]
                        values = data[f"{kind}_z_plus"][:, :, 0]
                    else:
                        time = data["time"][quantity]
                        values = data[kind][quantity]
                        if quantity != "n":
                            values = values[:, :, 0]
                    mask = (time >= t0) & (time <= t1)
                    if not np.any(mask):
                        raise ValueError(f"Selected interval contains no {quantity} samples")
                    selected = values[mask]
                    pair_values.append(selected)
                    axis = axes[row + offset, col]
                    for index, spacecraft_id in enumerate(data["spacecraft_ids"]):
                        axis.plot(
                            time[mask] - t0,
                            selected[:, index],
                            color=spacecraft_color(spacecraft_id),
                            linewidth=0.7,
                        )
                    axis.set_ylabel(f"{prefix} {label}")
                    axis.grid(alpha=0.18, linewidth=0.5)
                    panel = "abcdefgh"[2 * (row + offset) + col]
                    axis.text(
                        0.015, 0.9, f"({panel})", transform=axis.transAxes,
                        fontsize=11, fontweight="bold", va="top",
                    )
                limits = _component_ylim(*pair_values)
                axes[row, col].set_ylim(limits)
                axes[row + 1, col].set_ylim(limits)
            axes[0, 0].set_xlim(0.0, stop_time_s - start_time_s)
            for axis in axes[-1]:
                axis.set_xlabel(r"$t - t_0$ (s)")
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            paths = (output.with_suffix(".pdf"), output.with_suffix(".png"))
            for path in paths:
                figure.savefig(path, dpi=dpi)
        finally:
            plt.close(figure)
    return paths


def plot_velocity_alfven_timeseries(
    time: Mapping[str, np.ndarray],
    values: Mapping[str, np.ndarray],
    spacecraft_ids: Sequence[str],
    velocity_frame: str = "plasma",
    fluctuations: bool = False,
):
    """Plot velocity and magnetic field in Alfvén-speed units.

    The diagnostic converts ``B`` from nT to km/s as
    ``b = B / sqrt(mu0 m_p n)`` using instantaneous density interpolated onto
    the magnetic-field cadence. This is intentionally a normalization check and
    does not change the Elsasser analysis density-normalization defaults.
    """

    velocity = np.asarray(values["v"], dtype=float)
    alfven = _alfven_velocity_units(time, values)
    symbol_prefix = r"\delta " if fluctuations else ""
    title_prefix = "Fluctuating " if fluctuations else ""
    if fluctuations:
        velocity = velocity - np.mean(velocity, axis=0, keepdims=True)
        alfven = alfven - np.mean(alfven, axis=0, keepdims=True)
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), constrained_layout=True)
    labels = ("x", "y", "z")
    for component, label in enumerate(labels):
        u_ax = axes[component, 0]
        b_ax = axes[component, 1]
        for spacecraft_id_index, spacecraft_id in enumerate(spacecraft_ids):
            color = spacecraft_color(spacecraft_id)
            u_ax.plot(
                time["v"],
                velocity[:, spacecraft_id_index, component],
                color=color,
                label=spacecraft_id,
                linewidth=1.3,
            )
            b_ax.plot(
                time["B"],
                alfven[:, spacecraft_id_index, component],
                color=color,
                label=spacecraft_id,
                linewidth=1.3,
            )
        ylim = _component_ylim(
            velocity[:, :, component],
            alfven[:, :, component],
        )
        u_ax.set_ylim(*ylim)
        b_ax.set_ylim(*ylim)
        u_ax.set_ylabel(rf"${symbol_prefix}U_{label}$ (km s$^{{-1}}$)")
        b_ax.set_ylabel(rf"${symbol_prefix}b_{label}$ (km s$^{{-1}}$)")
        u_ax.grid(alpha=0.3)
        b_ax.grid(alpha=0.3)

    axes[0, 0].set_title(r"Velocity $U$")
    axes[0, 1].set_title(r"Magnetic Field $b$ in Velocity Units")
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 1].legend(handles, labels, loc="best", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(
        f"{title_prefix}Velocity and Alfvén-Speed Magnetic Field "
        f"({velocity_frame.capitalize()}-Frame Velocity)"
    )
    return fig, axes


def plot_observed_velocity_alfven_timeseries(
    time: Mapping[str, np.ndarray],
    truth: Mapping[str, np.ndarray],
    observed: Mapping[str, np.ndarray],
    spacecraft_ids: Sequence[str],
    mc_index: int,
    fluctuations: bool = False,
):
    """Overlay one observed U/b realization over nominal truth in grayscale."""

    truth_u = np.asarray(truth["v"], dtype=float)
    truth_b = _alfven_velocity_units(time, truth)
    observed_values = {
        "n": np.asarray(observed["n"][mc_index], dtype=float),
        "v": np.asarray(observed["v"][mc_index], dtype=float),
        "B": np.asarray(observed["B"][mc_index], dtype=float),
    }
    observed_u = observed_values["v"]
    observed_b = _alfven_velocity_units(time, observed_values)
    symbol_prefix = r"\delta " if fluctuations else ""
    title_prefix = "Fluctuating " if fluctuations else ""
    if fluctuations:
        truth_u = truth_u - np.mean(truth_u, axis=0, keepdims=True)
        truth_b = truth_b - np.mean(truth_b, axis=0, keepdims=True)
        observed_u = observed_u - np.mean(observed_u, axis=0, keepdims=True)
        observed_b = observed_b - np.mean(observed_b, axis=0, keepdims=True)

    fig, axes = plt.subplots(3, 2, figsize=(15, 10), constrained_layout=True)
    gray_colors = plt.cm.Greys(np.linspace(0.35, 0.75, len(spacecraft_ids)))
    labels = ("x", "y", "z")
    for component, label in enumerate(labels):
        u_ax = axes[component, 0]
        b_ax = axes[component, 1]
        for spacecraft_index, _spacecraft_id in enumerate(spacecraft_ids):
            u_ax.plot(
                time["v"],
                truth_u[:, spacecraft_index, component],
                color=gray_colors[spacecraft_index],
                linewidth=2.0,
                alpha=0.65,
                zorder=1,
            )
            b_ax.plot(
                time["B"],
                truth_b[:, spacecraft_index, component],
                color=gray_colors[spacecraft_index],
                linewidth=2.0,
                alpha=0.65,
                zorder=1,
            )
        for spacecraft_index, spacecraft_id in enumerate(spacecraft_ids):
            color = spacecraft_color(spacecraft_id)
            u_ax.plot(
                time["v"],
                observed_u[:, spacecraft_index, component],
                color=color,
                label=spacecraft_id,
                linewidth=1.1,
                zorder=2,
            )
            b_ax.plot(
                time["B"],
                observed_b[:, spacecraft_index, component],
                color=color,
                label=spacecraft_id,
                linewidth=1.1,
                zorder=2,
            )
        ylim = _component_ylim(
            truth_u[:, :, component],
            truth_b[:, :, component],
            observed_u[:, :, component],
            observed_b[:, :, component],
        )
        u_ax.set_ylim(*ylim)
        b_ax.set_ylim(*ylim)
        u_ax.set_ylabel(rf"${symbol_prefix}U_{label}$ (km s$^{{-1}}$)")
        b_ax.set_ylabel(rf"${symbol_prefix}b_{label}$ (km s$^{{-1}}$)")
        u_ax.grid(alpha=0.3)
        b_ax.grid(alpha=0.3)

    axes[0, 0].set_title(r"Velocity $U$")
    axes[0, 1].set_title(r"Magnetic Field $b$ in Velocity Units")
    axes[-1, 0].set_xlabel("Nominal time (s)")
    axes[-1, 1].set_xlabel("Nominal time (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="#777777", linewidth=2.0, label="Nominal truth"))
    labels.append("Nominal truth")
    axes[0, 1].legend(handles, labels, loc="best", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(
        f"Observed {title_prefix}Velocity and Alfvén-Speed Magnetic Field: MC {mc_index}"
    )
    return fig, axes


def _plot_relative_projection(
    ax,
    relative_position: np.ndarray,
    spacecraft_ids: Sequence[str],
    x_index: int,
    y_index: int,
    x_label: str,
    y_label: str,
    title: str,
) -> None:
    extent = float(np.max(np.abs(relative_position[:, [x_index, y_index]])))
    limit = 1.08 * extent if extent > 0 else 1.0
    for spacecraft_id, point in zip(spacecraft_ids, relative_position):
        is_hub = spacecraft_id == "H"
        ax.scatter(
            point[x_index],
            point[y_index],
            s=90 if is_hub else 70,
            facecolors=spacecraft_color(spacecraft_id),
            edgecolors="#111111",
            linewidths=1.3 if is_hub else 1.0,
            zorder=3,
        )
        ax.annotate(spacecraft_id, (point[x_index], point[y_index]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="#444444", linewidth=0.8, alpha=0.35)
    ax.axvline(0, color="#444444", linewidth=0.8, alpha=0.35)
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel=x_label, ylabel=y_label, title=title)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)


def plot_spacecraft_geometry(
    time: np.ndarray,
    position: np.ndarray,
    spacecraft_ids: Sequence[str],
):
    """Return hub-relative formation projections and an X-Z trajectory plot."""

    position = np.asarray(position, dtype=float)
    if position.shape != (len(spacecraft_ids), len(time), 3):
        raise ValueError("position must have shape (n_spacecraft, n_time, 3)")
    try:
        hub_index = list(spacecraft_ids).index("H")
    except ValueError as exc:
        raise ValueError("Geometry visualization requires spacecraft H") from exc

    initial = position[:, 0, :]
    relative = initial - initial[hub_index]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    _plot_relative_projection(
        axes[0, 0], relative, spacecraft_ids, 0, 2,
        r"$\Delta X$ (km)", r"$\Delta Z$ (km)", r"Hub-Relative $\Delta X$-$\Delta Z$",
    )
    _plot_relative_projection(
        axes[1, 0], relative, spacecraft_ids, 0, 1,
        r"$\Delta X$ (km)", r"$\Delta Y$ (km)", r"Hub-Relative $\Delta X$-$\Delta Y$",
    )
    _plot_relative_projection(
        axes[1, 1], relative, spacecraft_ids, 2, 1,
        r"$\Delta Z$ (km)", r"$\Delta Y$ (km)", r"Hub-Relative $\Delta Z$-$\Delta Y$",
    )

    path_ax = axes[0, 1]
    for index, spacecraft_id in enumerate(spacecraft_ids):
        path_ax.plot(
            position[index, :, 0],
            position[index, :, 2],
            color=spacecraft_color(spacecraft_id),
            label=spacecraft_id,
            linewidth=1.8,
        )
        path_ax.scatter(
            position[index, 0, 0],
            position[index, 0, 2],
            s=70 if spacecraft_id == "H" else 50,
            facecolors=spacecraft_color(spacecraft_id),
            edgecolors="#111111",
            linewidths=1.0,
            zorder=3,
        )
    path_ax.set(xlabel="X (km)", ylabel="Z (km)", title="Spacecraft Paths: X-Z Projection")
    path_ax.grid(alpha=0.25)
    path_ax.set_aspect("equal", adjustable="datalim")
    path_ax.legend(loc="best", ncol=3, fontsize=8)
    fig.suptitle("HelioSwarm Synthetic Observatory Geometry")
    return fig, axes


def _plot_relative_overlay(
    ax,
    true_relative: np.ndarray,
    measured_relative: np.ndarray,
    spacecraft_ids: Sequence[str],
    x_index: int,
    y_index: int,
    x_label: str,
    y_label: str,
    title: str,
) -> None:
    combined = np.concatenate((true_relative, measured_relative), axis=0)
    extent = float(np.max(np.abs(combined[:, [x_index, y_index]])))
    limit = 1.08 * extent if extent > 0 else 1.0
    gray_colors = plt.cm.Greys(np.linspace(0.35, 0.75, len(spacecraft_ids)))
    for index, (spacecraft_id, true_point, measured_point) in enumerate(
        zip(spacecraft_ids, true_relative, measured_relative)
    ):
        is_hub = spacecraft_id == "H"
        size = 90 if is_hub else 70
        ax.scatter(
            true_point[x_index],
            true_point[y_index],
            s=size,
            facecolors=gray_colors[index],
            edgecolors="#666666",
            linewidths=1.0,
            alpha=0.7,
            zorder=1,
        )
        ax.scatter(
            measured_point[x_index],
            measured_point[y_index],
            s=size,
            facecolors=spacecraft_color(spacecraft_id),
            edgecolors="#111111",
            linewidths=1.3 if is_hub else 1.0,
            zorder=3,
        )
        ax.plot(
            [true_point[x_index], measured_point[x_index]],
            [true_point[y_index], measured_point[y_index]],
            color="#888888",
            linewidth=0.8,
            alpha=0.6,
            zorder=2,
        )
        ax.annotate(
            spacecraft_id,
            (measured_point[x_index], measured_point[y_index]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axhline(0, color="#444444", linewidth=0.8, alpha=0.35)
    ax.axvline(0, color="#444444", linewidth=0.8, alpha=0.35)
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel=x_label, ylabel=y_label, title=title)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)


def plot_observed_spacecraft_geometry(
    time: np.ndarray,
    true_position: np.ndarray,
    position_offset: np.ndarray,
    spacecraft_ids: Sequence[str],
    mc_index: int,
):
    """Plot measured positions in color over nominal positions in grayscale."""

    true_position = np.asarray(true_position, dtype=float)
    if true_position.shape != (len(spacecraft_ids), len(time), 3):
        raise ValueError("true_position must have shape (n_spacecraft, n_time, 3)")
    if not 0 <= mc_index < len(position_offset):
        raise ValueError("mc_index is outside the position-offset ensemble")
    offsets = np.asarray(position_offset[mc_index], dtype=float)
    if offsets.shape != (len(spacecraft_ids), 3):
        raise ValueError("position_offset must have shape (n_mc, n_spacecraft, 3)")
    measured_position = true_position + offsets[:, None, :]
    try:
        hub_index = list(spacecraft_ids).index("H")
    except ValueError as exc:
        raise ValueError("Geometry visualization requires spacecraft H") from exc
    true_initial = true_position[:, 0, :]
    measured_initial = measured_position[:, 0, :]
    true_relative = true_initial - true_initial[hub_index]
    measured_relative = measured_initial - measured_initial[hub_index]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    _plot_relative_overlay(
        axes[0, 0], true_relative, measured_relative, spacecraft_ids, 0, 2,
        r"$\Delta X$ (km)", r"$\Delta Z$ (km)", r"Hub-Relative $\Delta X$-$\Delta Z$",
    )
    _plot_relative_overlay(
        axes[1, 0], true_relative, measured_relative, spacecraft_ids, 0, 1,
        r"$\Delta X$ (km)", r"$\Delta Y$ (km)", r"Hub-Relative $\Delta X$-$\Delta Y$",
    )
    _plot_relative_overlay(
        axes[1, 1], true_relative, measured_relative, spacecraft_ids, 2, 1,
        r"$\Delta Z$ (km)", r"$\Delta Y$ (km)", r"Hub-Relative $\Delta Z$-$\Delta Y$",
    )

    path_ax = axes[0, 1]
    gray_colors = plt.cm.Greys(np.linspace(0.35, 0.75, len(spacecraft_ids)))
    for index, spacecraft_id in enumerate(spacecraft_ids):
        path_ax.plot(
            true_position[index, :, 0],
            true_position[index, :, 2],
            color=gray_colors[index],
            linewidth=2.6,
            alpha=0.65,
            zorder=1,
        )
        path_ax.plot(
            measured_position[index, :, 0],
            measured_position[index, :, 2],
            color=spacecraft_color(spacecraft_id),
            label=spacecraft_id,
            linewidth=1.8,
            zorder=2,
        )
        path_ax.scatter(
            measured_position[index, 0, 0],
            measured_position[index, 0, 2],
            s=70 if spacecraft_id == "H" else 50,
            facecolors=spacecraft_color(spacecraft_id),
            edgecolors="#111111",
            linewidths=1.0,
            zorder=3,
        )
    handles, labels = path_ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="#777777", linewidth=2.6, label="Nominal truth"))
    labels.append("Nominal truth")
    path_ax.set(xlabel="X (km)", ylabel="Z (km)", title="Spacecraft Paths: X-Z Projection")
    path_ax.grid(alpha=0.25)
    path_ax.set_aspect("equal", adjustable="datalim")
    path_ax.legend(handles, labels, loc="best", ncol=3, fontsize=8)
    fig.suptitle(f"Observed Geometry with Nominal Truth: MC {mc_index}")
    return fig, axes


def unique_baselines(positions: np.ndarray) -> np.ndarray:
    """Return all unique pairwise baseline vectors from one spacecraft sample."""

    points = np.asarray(positions, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("positions must have shape (n_spacecraft, 3)")
    if not np.all(np.isfinite(points)):
        raise ValueError("positions contains non-finite values")
    return np.asarray(
        [points[j] - points[i] for i, j in combinations(range(len(points)), 2)],
        dtype=float,
    )


def _shape_metrics_for_subset(points: np.ndarray) -> dict[str, float]:
    centered = points - np.mean(points, axis=0, keepdims=True)
    tensor = centered.T @ centered / float(len(points))
    eigenvalues = np.linalg.eigvalsh(tensor)
    semiaxes = np.sqrt(np.clip(eigenvalues, 0.0, None))[::-1]
    a, b, c = (float(value) for value in semiaxes)
    eps = np.finfo(float).eps
    elongation = 0.0 if a <= eps else 1.0 - b / a
    planarity = 0.0 if b <= eps else 1.0 - c / b
    elongation = float(np.clip(elongation, 0.0, 1.0))
    planarity = float(np.clip(planarity, 0.0, 1.0))
    return {
        "a": a,
        "b": b,
        "c": c,
        "L": 2.0 * a,
        "E": elongation,
        "P": planarity,
        "Q": float(np.hypot(elongation, planarity)),
    }


def polyhedron_metrics(positions: np.ndarray) -> list[dict[str, Any]]:
    """Return volumetric-tensor metrics for all 4-through-N spacecraft subsets."""

    points = np.asarray(positions, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("positions must have shape (n_spacecraft, 3)")
    if len(points) < 4:
        raise ValueError("at least four spacecraft are required for polyhedra")
    if not np.all(np.isfinite(points)):
        raise ValueError("positions contains non-finite values")

    records: list[dict[str, Any]] = []
    for size in range(4, len(points) + 1):
        for subset in combinations(range(len(points)), size):
            metrics = _shape_metrics_for_subset(points[np.asarray(subset)])
            metrics.update({"indices": subset, "n_spacecraft": size})
            records.append(metrics)
    return records


def formisano_bow_shock_value(position_km: np.ndarray) -> np.ndarray:
    """Evaluate the Formisano bow-shock surface function in GSE Earth radii."""

    point = np.asarray(position_km, dtype=float) / EARTH_RADIUS_KM
    x = point[..., 0]
    y = point[..., 1]
    z = point[..., 2]
    c = FORMISANO_COEFFICIENTS
    return (
        c["a11"] * x**2
        + c["a22"] * y**2
        + c["a33"] * z**2
        + c["a12"] * x * y
        + c["a14"] * x
        + c["a24"] * y
        + c["a44"]
    )


def _connection_line_intersects_bow_shock(position_km: np.ndarray) -> bool:
    return _formisano_line_intersects_bow_shock(position_km, y_sign=-1.0)


def _formisano_line_intersects_bow_shock(position_km: np.ndarray, *, y_sign: float) -> bool:
    point = np.asarray(position_km, dtype=float) / EARTH_RADIUS_KM
    c = FORMISANO_COEFFICIENTS
    direction = np.array([1.0, y_sign, 0.0]) / np.sqrt(2.0)
    x0, y0, z0 = point
    dx, dy, dz = direction
    quadratic = (
        c["a11"] * dx**2
        + c["a22"] * dy**2
        + c["a33"] * dz**2
        + c["a12"] * dx * dy
    )
    linear = (
        2.0 * c["a11"] * x0 * dx
        + 2.0 * c["a22"] * y0 * dy
        + 2.0 * c["a33"] * z0 * dz
        + c["a12"] * (x0 * dy + y0 * dx)
        + c["a14"] * dx
        + c["a24"] * dy
    )
    constant = float(formisano_bow_shock_value(position_km))
    if abs(quadratic) < np.finfo(float).eps:
        return abs(linear) > np.finfo(float).eps
    return linear**2 - 4.0 * quadratic * constant >= 0.0


def classify_gse_region(position_km: np.ndarray) -> str:
    """Classify a GSE position using the Formisano bow-shock model."""

    if float(formisano_bow_shock_value(position_km)) < 0.0:
        return "Magnetosphere / Magnetosheath"
    if _connection_line_intersects_bow_shock(position_km):
        return "Connected Foreshock"
    return "Solar Wind"


# Okabe–Ito region colors, shared by all orbit-context panels.
REGION_COLORS = {
    "Solar Wind": "#D55E00",
    "Connected Foreshock": "#009E73",
    "Magnetosphere / Magnetosheath": "#0072B2",
}


def _discrete_polyhedron_cmap() -> ListedColormap:
    try:
        from cmcrameri import cm as cmcrameri_cm  # type: ignore

        colors = cmcrameri_cm.batlow(np.linspace(0.12, 0.88, 6))
    except ImportError:
        colors = plt.cm.cividis(np.linspace(0.12, 0.88, 6))
    return ListedColormap(colors, name="polyhedron_size")


def _single_sample_positions(position: np.ndarray) -> np.ndarray:
    array = np.asarray(position, dtype=float)
    if array.ndim == 3:
        return array[:, 0, :]
    if array.ndim == 2 and array.shape[1] == 3:
        return array
    raise ValueError("position must have shape (n_spacecraft, 3) or (n_spacecraft, n_time, 3)")


def _selected_time_label(metadata: Mapping[str, Any] | None) -> str:
    if not metadata:
        return "unknown-time"
    return str(
        metadata.get("selected_time")
        or metadata.get("requested_time")
        or metadata.get("configuration_time")
        or "unknown-time"
    )


def _selected_time_title_label(metadata: Mapping[str, Any] | None) -> str:
    label = _selected_time_label(metadata)
    if label == "unknown-time":
        return label
    try:
        dt = np.datetime64(label)
        return np.datetime_as_string(dt, unit="s")
    except ValueError:
        return label.split(".")[0]


def geometry_summary_paths(
    output_path: str | Path,
    metadata: Mapping[str, Any] | None = None,
    *,
    variant: str = "summary",
) -> tuple[Path, Path]:
    """Return PDF and PNG paths for a geometry summary variant."""

    if variant not in {"summary", "polyhedra", "formation"}:
        raise ValueError(f"Unknown geometry summary variant: {variant}")

    output = Path(output_path)
    tag = _selected_time_label(metadata)
    safe_tag = "".join(char if char.isalnum() else "-" for char in tag).strip("-")
    safe_tag = "-".join(part for part in safe_tag.split("-") if part) or "unknown-time"
    directory = output.parent / "geometry_overviews"
    base = directory / f"{output.stem}_geometry_{variant}_{safe_tag}"
    return base.with_suffix(".pdf"), base.with_suffix(".png")


def configuration_3d_paths(
    output_path: str | Path,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Return time-tagged PDF and PNG paths for the 3D configuration plot."""

    output = Path(output_path)
    tag = _selected_time_label(metadata)
    safe_tag = "".join(char if char.isalnum() else "-" for char in tag).strip("-")
    safe_tag = "-".join(part for part in safe_tag.split("-") if part) or "unknown-time"
    base = output.parent / "geometry_overviews" / f"{output.stem}_3d_configuration_{safe_tag}"
    return base.with_suffix(".pdf"), base.with_suffix(".png")


def helio_swarm_mesocenter(positions: np.ndarray) -> np.ndarray:
    """Return the arithmetic mean of the nine GSE spacecraft positions."""

    sample = _single_sample_positions(positions)
    if sample.shape != (9, 3):
        raise ValueError(f"3D configuration requires positions for 9 spacecraft; got {sample.shape}")
    if not np.all(np.isfinite(sample)):
        raise ValueError("3D configuration positions contain non-finite values")
    return np.mean(sample, axis=0)


def default_configuration_connections(
    spacecraft_ids: Sequence[str] = ("H", "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8"),
) -> tuple[tuple[str, str], ...]:
    """Return only the eight default Hub-to-Node spokes."""

    labels = tuple("Hub" if value == "H" else value for value in spacecraft_ids)
    if set(labels) != {"Hub", *(f"N{index}" for index in range(1, 9))}:
        raise ValueError("3D configuration requires spacecraft H (Hub) and N1 through N8")
    return tuple(("Hub", f"N{index}") for index in range(1, 9))


def parse_configuration_connections(
    specifications: Sequence[str] | str | None,
) -> tuple[tuple[str, str], ...]:
    """Parse extra pairs written as ``Hub-N3`` or ``N2-N5``.

    A YAML string or list of strings is accepted. Comma-separated strings are
    also supported for convenient CLI use.
    """

    if specifications is None:
        return ()
    raw = [specifications] if isinstance(specifications, str) else list(specifications)
    items = [item.strip() for value in raw for item in str(value).split(",") if item.strip()]
    valid = {"Hub", *(f"N{index}" for index in range(1, 9))}
    pairs: list[tuple[str, str]] = []
    for item in items:
        parts = tuple(part.strip() for part in item.split("-"))
        if len(parts) != 2 or not all(parts):
            raise ValueError(
                f"Invalid 3D connection {item!r}; expected LABEL-LABEL, e.g. N2-N5"
            )
        first, second = parts
        invalid = [label for label in parts if label not in valid]
        if invalid:
            raise ValueError(
                f"Invalid spacecraft label(s) {', '.join(invalid)} in connection {item!r}; "
                "valid labels are Hub and N1 through N8"
            )
        if first == second:
            raise ValueError(f"Invalid 3D connection {item!r}; endpoints must differ")
        pair = (first, second)
        if pair not in pairs and pair[::-1] not in pairs:
            pairs.append(pair)
    return tuple(pairs)


def plot_3d_configuration(
    positions: np.ndarray,
    spacecraft_ids: Sequence[str],
    *,
    metadata: Mapping[str, Any] | None = None,
    extra_connections: Sequence[str] | str | None = None,
):
    """Plot the mesocentric HelioSwarm configuration in actual GSE axes."""

    sample = _single_sample_positions(positions)
    if len(spacecraft_ids) != 9 or sample.shape != (9, 3):
        raise ValueError("3D configuration requires exactly 9 spacecraft positions and labels")
    display_labels = tuple("Hub" if label == "H" else label for label in spacecraft_ids)
    defaults = default_configuration_connections(spacecraft_ids)
    extras = parse_configuration_connections(extra_connections)
    index_by_label = {label: index for index, label in enumerate(display_labels)}
    mesocenter = helio_swarm_mesocenter(sample)
    relative = sample - mesocenter
    extent = float(np.max(np.abs(relative)))
    axis_limit = 1.12 * extent if extent > 0.0 else 1.0

    with plt.rc_context({
        "font.size": 9.0,
        "axes.titlesize": 11.0,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "savefig.bbox": "tight",
    }):
        figure = plt.figure(figsize=(6.8, 6.25), constrained_layout=True)
        layout_engine = figure.get_layout_engine()
        if layout_engine is not None:
            layout_engine.set(rect=(0.0, 0.14, 1.0, 0.82))
        axis = figure.add_subplot(111, projection="3d")
        low = -axis_limit
        high = axis_limit
        for index, (spacecraft_id, label) in enumerate(zip(spacecraft_ids, display_labels)):
            color = spacecraft_color(spacecraft_id)
            x, y, z = relative[index]
            # The X-Z projection terminates on the +Y wall; the other two
            # projections remain on the negative X and Z bounding planes.
            axis.plot([x, low], [y, y], [z, z], "--", color=color, alpha=0.22, lw=0.65)
            axis.plot([x, x], [y, high], [z, z], "--", color=color, alpha=0.22, lw=0.65)
            axis.plot([x, x], [y, y], [z, low], "--", color=color, alpha=0.22, lw=0.65)
            axis.scatter(
                [low, x, x], [y, high, y], [z, z, low], s=18, c=color,
                alpha=0.48, edgecolors="none", depthshade=False,
            )
            axis.scatter(
                x, y, z, s=76 if label == "Hub" else 55, c=color,
                edgecolors="#202020", linewidths=0.55, depthshade=False, zorder=6,
            )

        connections = defaults + tuple(pair for pair in extras if pair not in defaults and pair[::-1] not in defaults)
        for first, second in connections:
            left, right = index_by_label[first], index_by_label[second]
            color = spacecraft_color(spacecraft_ids[right])
            axis.plot(*relative[[left, right]].T, color=color, lw=1.65, alpha=0.88)

        axis.set(xlim=(-axis_limit, axis_limit), ylim=(-axis_limit, axis_limit),
                 zlim=(-axis_limit, axis_limit), xlabel=r"$X_{GSE}$ (km)",
                 ylabel=r"$Y_{GSE}$ (km)", zlabel=r"$Z_{GSE}$ (km)")
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=24, azim=-56)
        axis.grid(True, alpha=0.24)
        axis.set_title(f"HelioSwarm Configuration\n{_selected_time_title_label(metadata)}", pad=13)
        figure.text(
            0.5, 0.064,
            r"GSE coordinates; Sunward is $-X_{GSE}$  |  Origin at HelioSwarm mesocenter",
            ha="center", va="bottom", fontsize=8.0, color="#444444",
        )
        legend_handles = [
            Line2D(
                [0], [0], marker="o", linestyle="none", markersize=6.0,
                markerfacecolor=spacecraft_color(spacecraft_id),
                markeredgecolor="#202020", markeredgewidth=0.4, label=label,
            )
            for spacecraft_id, label in zip(spacecraft_ids, display_labels)
        ]
        figure.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.012),
            ncol=9,
            frameon=False,
            fontsize=7.8,
            handletextpad=0.3,
            columnspacing=0.75,
            borderaxespad=0.0,
        )
    diagnostics = {
        "requested_configuration_time": (metadata or {}).get("requested_time")
        or (metadata or {}).get("configuration_time"),
        "matched_cdf_sample_time": (metadata or {}).get("selected_time"),
        "matched_sample_index": (metadata or {}).get("selected_index"),
        "mesocenter_km": mesocenter.tolist(),
        "axis_limit_km": axis_limit,
        "default_connections": defaults,
        "extra_connections": extras,
    }
    return figure, diagnostics


def save_3d_configuration(
    *, output_path: str | Path, config: dict, positions: np.ndarray,
    spacecraft_ids: Sequence[str], metadata: Mapping[str, Any] | None = None,
    extra_connections: Sequence[str] | str | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Write the reusable 3D configuration plot as PDF and PNG."""

    pdf_path, png_path = configuration_3d_paths(output_path, metadata)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    figure, diagnostics = plot_3d_configuration(
        positions, spacecraft_ids, metadata=metadata, extra_connections=extra_connections,
    )
    figure.savefig(pdf_path)
    figure.savefig(png_path, dpi=int(config.get("dpi", 150)))
    plt.close(figure)
    diagnostics.update(output_pdf_path=str(pdf_path), output_png_path=str(png_path))
    return pdf_path, png_path, diagnostics


def _axis_limits(values: np.ndarray, pad: float = 0.08) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if lo == hi:
        width = max(abs(lo), 1.0)
        return lo - width * 0.5, hi + width * 0.5
    margin = (hi - lo) * pad
    return lo - margin, hi + margin


def _plot_summary_relative_projection(
    ax,
    relative: np.ndarray,
    spacecraft_ids: Sequence[str],
    x_index: int,
    y_index: int,
    title: str,
) -> None:
    labels = ("X", "Y", "Z")
    extent = float(np.max(np.abs(relative[:, [x_index, y_index]])))
    limit = 1.12 * extent if extent > 0 else 1.0
    for index, spacecraft_id in enumerate(spacecraft_ids):
        is_hub = index == 0
        ax.scatter(
            relative[index, x_index],
            relative[index, y_index],
            s=192 if is_hub else 120,
            c=spacecraft_color(spacecraft_id),
            edgecolors="#111111",
            linewidths=0.55 if is_hub else 0.45,
            zorder=4 if is_hub else 3,
        )
    ax.axhline(0.0, color="#333333", linewidth=0.6, alpha=0.35)
    ax.axvline(0.0, color="#333333", linewidth=0.6, alpha=0.35)
    ax.set(
        xlim=(-limit, limit),
        ylim=(-limit, limit),
        xlabel=rf"$\Delta {labels[x_index]}_{{GSE}}$ (km)",
        ylabel=rf"$\Delta {labels[y_index]}_{{GSE}}$ (km)",
        title=title,
    )
    ax.set_xticks(np.arange(-2000.0, 2000.1, 1000.0))
    ax.set_aspect("equal")
    ax.grid(alpha=0.22, linewidth=0.5)


def _plot_context_projection(
    ax,
    trajectory: np.ndarray,
    selected: np.ndarray,
    x_index: int,
    y_index: int,
    title: str,
    moon_position: np.ndarray | None = None,
) -> None:
    labels = ("X", "Y", "Z")
    trajectory_re = trajectory / EARTH_RADIUS_KM
    selected_re = selected / EARTH_RADIUS_KM
    ax.add_patch(
        plt.Circle(
            (0.0, 0.0),
            1.0,
            color="#3a86ff",
            alpha=0.95,
            zorder=3,
        )
    )
    ax.plot(
        trajectory_re[:, x_index],
        trajectory_re[:, y_index],
        color="#777777",
        linewidth=0.55,
        alpha=0.45,
        zorder=1,
    )
    for start, stop in zip(trajectory[:-1], trajectory[1:]):
        segment_region = classify_gse_region((start + stop) * 0.5)
        start_re = start / EARTH_RADIUS_KM
        stop_re = stop / EARTH_RADIUS_KM
        ax.plot(
            [start_re[x_index], stop_re[x_index]],
            [start_re[y_index], stop_re[y_index]],
            color=REGION_COLORS[segment_region],
            linewidth=0.85,
            alpha=0.9,
            zorder=2,
        )
    if moon_position is not None:
        moon = np.asarray(moon_position, dtype=float) / EARTH_RADIUS_KM
        ax.scatter(
            moon[x_index],
            moon[y_index],
            s=72,
            c="#a8a8a8",
            edgecolors="#444444",
            linewidths=0.4,
            zorder=3,
        )
    ax.scatter(
        selected_re[x_index],
        selected_re[y_index],
        s=136,
        c="#d62728",
        edgecolors="#111111",
        linewidths=0.45,
        zorder=4,
    )
    ax.set(
        xlabel=rf"${labels[x_index]}_{{GSE}}$ ($R_E$)",
        ylabel=rf"${labels[y_index]}_{{GSE}}$ ($R_E$)",
        title=title,
    )
    extra_x = [0.0, selected_re[x_index]]
    extra_y = [0.0, selected_re[y_index]]
    if moon_position is not None:
        moon = np.asarray(moon_position, dtype=float) / EARTH_RADIUS_KM
        extra_x.append(float(moon[x_index]))
        extra_y.append(float(moon[y_index]))
    all_x = np.concatenate((trajectory_re[:, x_index], extra_x))
    all_y = np.concatenate((trajectory_re[:, y_index], extra_y))
    ax.set_xlim(*_axis_limits(all_x, pad=0.12))
    ax.set_ylim(*_axis_limits(all_y, pad=0.12))
    ax.grid(alpha=0.22, linewidth=0.5)
    ax.set_aspect("equal", adjustable="datalim")


def _context_legend_handles(moon_position: np.ndarray | None = None) -> list[Line2D]:
    handles = [
        *(
            Line2D([0], [0], color=color, linewidth=1.3, label=name)
            for name, color in REGION_COLORS.items()
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#d62728",
            markeredgecolor="#111111",
            label="Observatory",
        ),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#3a86ff", label="Earth"),
    ]
    if moon_position is not None:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#a8a8a8",
                label="Moon",
            )
        )
    return handles


def _positive_component_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    positive = finite[finite > 0.0]
    hi = float(np.max(positive)) if positive.size else MHD_SCALE_KM
    lo = float(np.min(positive)) if positive.size else 1.0
    lo = min(lo, KINETIC_SCALE_KM) * 0.6
    hi = max(hi, MHD_SCALE_KM) * 1.8
    return max(lo, 1.0e-6), hi


def _baseline_component_limits(components: np.ndarray) -> tuple[float, float]:
    values = np.maximum(np.abs(np.asarray(components, dtype=float)), 1.0e-9)
    return _positive_component_limits(values)


def _plot_baseline_components(
    ax,
    components: np.ndarray,
    x_index: int,
    y_index: int,
    title: str,
    limits: tuple[float, float],
    *,
    baseline_colors: np.ndarray | None = None,
) -> None:
    labels = ("X", "Y", "Z")
    values = np.maximum(np.abs(components), 1.0e-9)
    lo, hi = limits
    ax.axvspan(
        lo, KINETIC_SCALE_KM, color=LAMBDA_SCALE_COLORS["kinetic"], alpha=0.24, zorder=0
    )
    ax.axvspan(
        KINETIC_SCALE_KM,
        MHD_SCALE_KM,
        color=LAMBDA_SCALE_COLORS["transition"],
        alpha=0.2,
        zorder=0,
    )
    ax.axvspan(MHD_SCALE_KM, hi, color=LAMBDA_SCALE_COLORS["mhd"], alpha=0.22, zorder=0)
    ax.axhspan(
        lo, KINETIC_SCALE_KM, color=LAMBDA_SCALE_COLORS["kinetic"], alpha=0.12, zorder=0
    )
    ax.axhspan(
        KINETIC_SCALE_KM,
        MHD_SCALE_KM,
        color=LAMBDA_SCALE_COLORS["transition"],
        alpha=0.1,
        zorder=0,
    )
    ax.axhspan(MHD_SCALE_KM, hi, color=LAMBDA_SCALE_COLORS["mhd"], alpha=0.11, zorder=0)
    ax.scatter(
        values[:, x_index],
        values[:, y_index],
        s=64,
        c="#252525" if baseline_colors is None else baseline_colors,
        alpha=0.82,
        edgecolors="white" if baseline_colors is None else "#333333",
        linewidths=0.25,
        zorder=3,
    )
    ax.axvline(KINETIC_SCALE_KM, color="#777777", linewidth=0.7, linestyle=":")
    ax.axvline(MHD_SCALE_KM, color="#777777", linewidth=0.7, linestyle=":")
    ax.axhline(KINETIC_SCALE_KM, color="#777777", linewidth=0.7, linestyle=":")
    ax.axhline(MHD_SCALE_KM, color="#777777", linewidth=0.7, linestyle=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set(
        xlabel=rf"$|\lambda_{{{labels[x_index]},GSE}}|$ (km)",
        ylabel=rf"$|\lambda_{{{labels[y_index]},GSE}}|$ (km)",
        title=title,
    )
    ax.grid(alpha=0.18, linewidth=0.5, which="both")


def _label_geometry_summary_panels(axes: Sequence[Any], *, grid_order: bool = False) -> None:
    ordered = sorted(
        axes,
        key=lambda axis: (
            (axis.get_subplotspec().rowspan.start, axis.get_subplotspec().colspan.start)
            if grid_order else (-axis.get_position().y0, axis.get_position().x0)
        ),
    )
    for label, axis in zip("abcdefgh", ordered):
        axis.text(
            0.035,
            0.94,
            f"({label})",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=24,
            fontweight="bold",
            color="#111111",
        )


def _regular_size_ratio_exists(metrics: Sequence[Mapping[str, Any]]) -> bool:
    regular_sizes = sorted(
        float(item["L"])
        for item in metrics
        if float(item["Q"]) <= REGULAR_POLYHEDRON_Q_LIMIT and float(item["L"]) > 0.0
    )
    if len(regular_sizes) < 2:
        return False
    return regular_sizes[-1] / regular_sizes[0] >= 3.0


def geometry_summary_diagnostics(
    positions: np.ndarray,
    *,
    metadata: Mapping[str, Any] | None = None,
    hub_index: int = 0,
) -> dict[str, Any]:
    """Compute counts and status flags reported by the geometry summary."""

    sample = _single_sample_positions(positions)
    if len(sample) != 9:
        raise ValueError(f"Geometry summary requires 9 spacecraft; got {len(sample)}")
    baselines = unique_baselines(sample)
    metrics = polyhedron_metrics(sample)
    lengths = np.linalg.norm(baselines, axis=1)
    mesocenter = np.mean(sample, axis=0)
    regular_count = sum(
        1 for item in metrics if float(item["Q"]) <= REGULAR_POLYHEDRON_Q_LIMIT
    )
    if len(baselines) != 36:
        warnings.warn(f"Expected 36 baselines; got {len(baselines)}", stacklevel=2)
    if len(metrics) != 382:
        warnings.warn(f"Expected 382 polyhedra; got {len(metrics)}", stacklevel=2)
    metadata = metadata or {}
    return {
        "requested_configuration_time": metadata.get("requested_time")
        or metadata.get("configuration_time"),
        "matched_cdf_sample_time": metadata.get("selected_time"),
        "matched_sample_index": metadata.get("selected_index"),
        "hub_index": hub_index,
        "n_spacecraft": len(sample),
        "n_baselines": len(baselines),
        "baseline_min_km": float(np.min(lengths)),
        "baseline_max_km": float(np.max(lengths)),
        "n_polyhedra": len(metrics),
        "n_regular_polyhedra": regular_count,
        "regular_size_ratio_at_least_3": _regular_size_ratio_exists(metrics),
        "region": classify_gse_region(mesocenter),
    }


POLYHEDRON_POINT_AREA = 33  # 1.5 times the previous 22 pt².
POLYHEDRON_HIGHLIGHT_AREA = 285  # 1.5 times the previous 190 pt².
POLYHEDRON_EXTRA_Q_GUIDES = (0.4, 0.8)
POLYHEDRON_Q_SHADE_COLOR = "#009E73"


def _shade_polyhedron_quality(ax, *, shape: bool, base_alpha: float) -> None:
    """Shade disjoint quality bands so their opacities do not accumulate."""
    from matplotlib.patches import Wedge

    for low, high, multiplier in ((0.0, 0.4, 1.5), (0.4, 0.6, 1.0), (0.6, 0.8, 0.5)):
        style = dict(facecolor=POLYHEDRON_Q_SHADE_COLOR, edgecolor="none",
                     alpha=base_alpha * multiplier, zorder=0)
        if shape:
            ax.add_patch(Wedge((0, 0), high, 0, 90, width=high - low, **style))
        else:
            ax.axvspan(low, high, **style)


def _polyhedron_length_cmap():
    """Use Crameri's sequential lajolla map, retaining the optional fallback."""
    try:
        from cmcrameri import cm
    except ImportError:
        return plt.get_cmap("cividis")
    return cm.bam


def _selected_plot_polyhedra(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """Match gradient small/best_multi/all_nine choices using the plot catalog."""

    selected = {}
    regular = [
        m for m in metrics if m["n_spacecraft"] == 4 and m["Q"] <= REGULAR_POLYHEDRON_Q_LIMIT
    ]
    if regular:
        selected["smallest regular"] = min(regular, key=lambda m: (m["L"], m["Q"], m["indices"]))

    def multi_key(m):
        eigenvalues = np.square([m["a"], m["b"], m["c"]])
        tolerance = np.finfo(float).eps * max(m["a"], 1.0) ** 2 * 32
        condition = eigenvalues[0] / eigenvalues[2] if np.all(eigenvalues > tolerance) else np.inf
        return m["Q"], condition, m["L"], m["indices"]

    multi = [m for m in metrics if 4 < m["n_spacecraft"] < 9]
    if multi:
        selected["most regular N>4"] = min(multi, key=multi_key)
    all_nine = [m for m in metrics if m["n_spacecraft"] == 9]
    if all_nine:
        selected["all N=9"] = all_nine[0]
    return selected


def _highlight_plot_polyhedra(ax, selected, *, shape=False):
    markers = {"smallest regular": "o", "most regular N>4": "s", "all N=9": "D"}
    for label, record in selected.items():
        x, y = (record["E"], record["P"]) if shape else (record["Q"], max(record["L"], 1e-9))
        ax.scatter(
            x,
            y,
            s=POLYHEDRON_HIGHLIGHT_AREA,
            marker=markers[label],
            facecolors="none",
            edgecolors="#151515",
            linewidths=1.8,
            zorder=6,
            label=label,
        )


def _plot_polyhedron_shape(ax, metrics: Sequence[Mapping[str, Any]]) -> None:
    """Project elongation and planarity, coloring positive sizes logarithmically."""

    elongation = np.asarray([item["E"] for item in metrics], dtype=float)
    planarity = np.asarray([item["P"] for item in metrics], dtype=float)
    lengths = np.asarray([item["L"] for item in metrics], dtype=float)
    positive = lengths > 0
    if np.any(positive):
        low, high = float(lengths[positive].min()), float(lengths[positive].max())
        # A constant-size formation still needs a nonzero logarithmic color range.
        if low == high:
            low, high = low / np.sqrt(10), high * np.sqrt(10)
        points = ax.scatter(
            elongation[positive],
            planarity[positive],
            c=lengths[positive],
            cmap=_polyhedron_length_cmap(),
            norm=LogNorm(vmin=low, vmax=high),
            s=POLYHEDRON_POINT_AREA,
            alpha=0.5,
            edgecolors="none",
        )
        ax.figure.colorbar(points, ax=ax, pad=0.01, fraction=0.03).set_label("L (km)")
    else:
        ax.text(
            0.5,
            0.5,
            "No positive L values for logarithmic colors",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=16,
        )
    if np.any(~positive):
        ax.scatter(
            elongation[~positive],
            planarity[~positive],
            marker="x",
            color="#777777",
            label="L = 0 km",
        )
        ax.legend(loc="upper right", fontsize=16)
    ax.set(
        xlabel="Elongation E",
        ylabel="Planarity P",
        title="Polyhedral Shape",
        xlim=(-0.02, 1.02),
        ylim=(-0.02, 1.02),
    )
    theta = np.linspace(0, np.pi / 2, 181)
    e = REGULAR_POLYHEDRON_Q_LIMIT * np.cos(theta)
    p = REGULAR_POLYHEDRON_Q_LIMIT * np.sin(theta)
    _shade_polyhedron_quality(ax, shape=True, base_alpha=0.06)
    ax.plot(e, p, "--", color="#555555", lw=1.1)
    for radius in POLYHEDRON_EXTRA_Q_GUIDES:
        ax.plot(radius * np.cos(theta), radius * np.sin(theta),
                linestyle=":", color="#777777", lw=1.0, zorder=2)
    _highlight_plot_polyhedra(ax, _selected_plot_polyhedra(metrics), shape=True)
    ax.set_aspect("auto")
    ax.grid(alpha=0.2, linewidth=0.5)


def _plot_summary_formation(
    ax,
    relative: np.ndarray,
    spacecraft_ids: Sequence[str],
    baseline_pairs: Sequence[tuple[int, int]],
    baseline_colors: np.ndarray,
) -> None:
    """Draw hub-relative GSE sampling volume, without an orbit inset."""

    for (i, j), color in zip(baseline_pairs, baseline_colors):
        ax.plot(
            *relative[[i, j]].T,
            color=color,
            lw=0.8,
            alpha=0.55,
            zorder=1,
        )
    #for point in relative[1:]:
    #    ax.plot(*np.stack((relative[0], point)).T, color="#888888", lw=1.2, alpha=0.4)
    # The hull is a visual aid only; the figure also works without optional SciPy
    # or for a coplanar/coincident formation where no 3D hull exists.
    try:
        from scipy.spatial import ConvexHull, QhullError
    except ImportError:
        pass
    else:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        try:
            hull = ConvexHull(relative)
        except QhullError:
            pass
        else:
            ax.add_collection3d(
                Poly3DCollection(
                    relative[hull.simplices],
                    facecolor="#56B4E9",
                    edgecolor="#888888",
                    linewidth=0.55,
                    alpha=0.075,
                )
            )
    for index, (point, spacecraft_id) in enumerate(zip(relative, spacecraft_ids)):
        ax.scatter(
            *point,
            s=140 if index == 0 else 80,
            color=spacecraft_color(spacecraft_id),
            edgecolor="#202020",
            linewidth=0.7,
            depthshade=False,
            label="Hub" if spacecraft_id == "H" else spacecraft_id,
        )
    span = np.ptp(relative, axis=0)
    margin = np.maximum(0.08 * span, 20.0)
    low, high = relative.min(axis=0) - margin, relative.max(axis=0) + margin
    ax.set(
        xlim=(low[0], high[0]),
        ylim=(low[1], high[1]),
        zlim=(low[2], high[2]),
        xlabel=r"$\Delta X_{GSE}$ (km)",
        ylabel=r"$\Delta Y_{GSE}$ (km)",
        zlabel=r"$\Delta Z_{GSE}$ (km)",
        title="Hub-relative formation",
    )
    ax.set_box_aspect(high - low)
    ax.set_proj_type("ortho")
    ax.view_init(elev=22, azim=-58)
    ax.tick_params(labelsize=18)
    ax.xaxis.labelpad = ax.yaxis.labelpad = ax.zaxis.labelpad = 18
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.015), ncol=5, fontsize=18, frameon=False)


def _style_publication_geometry(fig, variant, relative, metadata):
    """Apply physical aspect, common scales, and typography to existing panels."""
    from matplotlib.ticker import MaxNLocator
    from matplotlib.patches import Patch

    fig.suptitle("HelioSwarm Design Reference Geometry", x=0.055, y=0.985, ha="left", fontsize=27)
    epoch = _selected_time_title_label(metadata)
    fig.text(0.055, 0.94, f"Epoch: {epoch} UTC", fontsize=24, color="#555555")
    extent = max(float(np.max(np.abs(relative))) * 1.12, 1.0)
    ticks = MaxNLocator(nbins=4, symmetric=True).tick_values(-extent, extent)
    ticks = ticks[(ticks >= -extent) & (ticks <= extent)]
    limit = extent
    axes = {
        (spec.rowspan.start, spec.colspan.start): ax
        for ax in fig.axes
        if (spec := ax.get_subplotspec()) is not None
    }
    context = axes[0, 1 if variant == "polyhedra" else 3]
    context.set_aspect("equal", adjustable="box")
    if variant == "polyhedra":
        for key in ((0, 0), (1, 0), (1, 1)):
            ax = axes[key]
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set(xlim=(-limit, limit), ylim=(-limit, limit))
            ax.set_aspect("equal", adjustable="box")
    else:
        for key in ((0, 2), (1, 2), (1, 3)):
            ax = axes[key]
            ax.set_aspect("equal", adjustable="box")
            for patch in ax.patches:
                patch.set_alpha(0.08)
            for line in ax.lines:
                line.set_color("#777777")
                line.set_linewidth(0.9)
        axes[0, 2].legend(
            handles=[
                Patch(facecolor=LAMBDA_SCALE_COLORS[key], alpha=0.45, label=label)
                for key, label in (
                    ("kinetic", "ion <100 km"),
                    ("transition", "transition 100–1200 km"),
                    ("mhd", "MHD >1200 km"),
                )
            ],
            loc="upper left",
            bbox_to_anchor=(0.035, 0.85),
            fontsize=15,
            frameon=False,
        )
    for ax in fig.axes:
        ax.set_title(ax.get_title().replace("GSE", r"\mathrm{GSE}"), fontsize=22)
        ax.tick_params(labelsize=18)
        for getter, setter in ((ax.get_xlabel, ax.set_xlabel), (ax.get_ylabel, ax.set_ylabel)):
            setter(getter().replace("GSE", r"\mathrm{GSE}"), fontsize=22)
        if ax.name == "3d":
            ax.set_zlabel(ax.get_zlabel().replace("GSE", r"\mathrm{GSE}"), fontsize=22)
        for text in ax.texts:
            if text.get_text() in {f"({c})" for c in "abcdefgh"}:
                text.set_position((0.035, 0.94))
                text.set_fontsize(22)


def plot_geometry_summary(
    position: np.ndarray,
    spacecraft_ids: Sequence[str],
    *,
    time: np.ndarray | None = None,
    metadata: Mapping[str, Any] | None = None,
    context_trajectory: np.ndarray | None = None,
    moon_position: np.ndarray | None = None,
    hub_index: int = 0,
    variant: str = "summary",
):
    """Return the original summary, a polyhedra view, or a 3D formation view."""

    if variant not in {"summary", "polyhedra", "formation"}:
        raise ValueError(f"Unknown geometry summary variant: {variant}")

    sample = _single_sample_positions(position)
    trajectory_source = (
        np.asarray(position, dtype=float)
        if context_trajectory is None
        else np.asarray(context_trajectory, dtype=float)
    )
    if trajectory_source.ndim == 3:
        trajectory = np.mean(trajectory_source, axis=0)
    elif trajectory_source.ndim == 2 and trajectory_source.shape[1] == 3:
        trajectory = trajectory_source
    else:
        raise ValueError(
            "context_trajectory must have shape (n_time, 3) or "
            "(n_spacecraft, n_time, 3)"
        )
    if len(sample) != len(spacecraft_ids):
        raise ValueError("spacecraft_ids length must match positions")
    if len(sample) != 9:
        raise ValueError(f"Geometry summary requires 9 spacecraft; got {len(sample)}")
    if hub_index != 0:
        raise ValueError("Geometry summary expects hub_index=0")

    baselines = unique_baselines(sample)
    metrics = polyhedron_metrics(sample)
    diagnostics = geometry_summary_diagnostics(sample, metadata=metadata, hub_index=hub_index)
    relative = sample - sample[hub_index]
    baseline_colors = None
    if variant == "formation":
        from cmcrameri import cm

        # Keep the unique_baselines ordering for both lines and component points.
        baseline_pairs = list(combinations(range(len(relative)), 2))
        baseline_vectors = np.asarray([relative[j] - relative[i] for i, j in baseline_pairs])
        baseline_magnitudes = np.linalg.norm(baseline_vectors, axis=1)
        positive_lengths = baseline_magnitudes[baseline_magnitudes > 0.0]
        baseline_cmap = cm.batlow
        baseline_norm = LogNorm(vmin=positive_lengths.min(), vmax=positive_lengths.max())
        baseline_colors = baseline_cmap(baseline_norm(baseline_magnitudes))
    mesocenter = np.mean(sample, axis=0)
    title_time = _selected_time_title_label(metadata)

    with plt.rc_context(
        {
            "font.size": 24.6,
            "axes.titlesize": 27.0,
            "axes.labelsize": 25.5,
            "xtick.labelsize": 22.5,
            "ytick.labelsize": 22.5,
            "legend.fontsize": 22.5,
            "figure.titlesize": 33.0,
            "savefig.bbox": "tight",
        }
    ):
        fig = plt.figure(figsize=(24.0 if variant == "polyhedra" else 31.0, 16.0), constrained_layout=True)
        layout_engine = fig.get_layout_engine()
        if layout_engine is not None:
            layout_engine.set(
                rect=(0.0, 0.0, 1.0, 0.94),
                h_pad=0.035,
                w_pad=0.035,
                hspace=0.08,
                wspace=0.035,
            )
        if variant == "polyhedra":
            grid = fig.add_gridspec(2, 3, width_ratios=(1.0, 1.0, 1.18))
        else:
            grid = fig.add_gridspec(2, 4, width_ratios=(1.0, 1.12, 1.0, 1.18))

        if variant != "formation":
            _plot_summary_relative_projection(
                fig.add_subplot(grid[0, 0]),
                relative,
                spacecraft_ids,
                0,
                1,
                r"$\Delta Y_{GSE}$ vs $\Delta X_{GSE}$",
            )

        else:
            formation_ax = fig.add_subplot(grid[:, :2], projection="3d")
            _plot_summary_formation(
                formation_ax, relative, spacecraft_ids, baseline_pairs, baseline_colors
            )

        context_ax = fig.add_subplot(grid[0, 1 if variant == "polyhedra" else 3])
        _plot_context_projection(
            context_ax,
            trajectory,
            mesocenter,
            0,
            1,
            r"",
            moon_position=moon_position,
        )
        context_ax.legend(
            handles=_context_legend_handles(moon_position),
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            fontsize=19.8,
            frameon=False,
            borderaxespad=0.25,
        )
        #context_ax.set_title(r"Orbit Context:")

        if variant != "polyhedra":
            baseline_limits = _baseline_component_limits(baselines)
            lambda_y_x_ax = fig.add_subplot(grid[0, 2])
            _plot_baseline_components(
                lambda_y_x_ax,
                baselines,
                0,
                1,
                r"$|\lambda_Y|$ vs $|\lambda_X|$",
                baseline_limits,
                baseline_colors=baseline_colors,
            )

        if variant != "formation":
            poly_ax = fig.add_subplot(grid[0, 2 if variant == "polyhedra" else 1])
            cmap = _discrete_polyhedron_cmap()
            norm = BoundaryNorm(np.arange(3.5, 10.5, 1.0), cmap.N)
            q_values = np.asarray([item["Q"] for item in metrics], dtype=float)
            l_values = np.asarray([item["L"] for item in metrics], dtype=float)
            sizes = np.asarray([item["n_spacecraft"] for item in metrics], dtype=float)
            if variant == "polyhedra":
                _shade_polyhedron_quality(poly_ax, shape=False, base_alpha=0.14)
            else:
                poly_ax.axvspan(
                    0.0, REGULAR_POLYHEDRON_Q_LIMIT, color="#90be6d",
                    alpha=0.14, label="regular region",
                )
            scatter = poly_ax.scatter(
                q_values,
                np.maximum(l_values, 1.0e-9),
                c=sizes,
                cmap=cmap,
                norm=norm,
                s=48,
                alpha=0.78,
                edgecolors="none",
            )
            poly_ax.axvline(
                REGULAR_POLYHEDRON_Q_LIMIT,
                color="#555555",
                linewidth=0.9,
                linestyle="--",
            )
            poly_ax.set_yscale("log")
            poly_ax.set_xlim(left=0.0)
            poly_ax.set(
                xlabel=r"$\sqrt{E^2 + P^2}$",
                ylabel=r"$L$ (km)",
                title="Polyhedral Configuration Summary",
            )
            poly_ax.grid(alpha=0.2, linewidth=0.5, which="both")
            colorbar = fig.colorbar(
                scatter,
                ax=poly_ax,
                ticks=np.arange(4, 10),
                pad=0.01,
                fraction=0.045,
            )
            colorbar.set_label("# spacecraft")
            colorbar.ax.tick_params(labelsize=24)

        if variant != "formation":
            _plot_summary_relative_projection(
                fig.add_subplot(grid[1, 0]),
                relative,
                spacecraft_ids,
                0,
                2,
                r"$\Delta Z_{GSE}$ vs $\Delta X_{GSE}$",
            )
            _plot_summary_relative_projection(
                fig.add_subplot(grid[1, 1]),
                relative,
                spacecraft_ids,
                1,
                2,
                r"$\Delta Z_{GSE}$ vs $\Delta Y_{GSE}$",
            )
        if variant != "polyhedra":
            lambda_z_x_ax = fig.add_subplot(grid[1, 2])
            _plot_baseline_components(
                lambda_z_x_ax,
                baselines,
                0,
                2,
                r"$|\lambda_Z|$ vs $|\lambda_X|$",
                baseline_limits,
                baseline_colors=baseline_colors,
            )
            lambda_z_y_ax = fig.add_subplot(grid[1, 3])
            _plot_baseline_components(
                lambda_z_y_ax,
                baselines,
                1,
                2,
                r"$|\lambda_Z|$ vs $|\lambda_Y|$",
                baseline_limits,
                baseline_colors=baseline_colors,
            )
        else:
            _plot_polyhedron_shape(fig.add_subplot(grid[1, 2]), metrics)

        if variant == "formation":
            # Letter the five publication panels consecutively.
            for label, axis in zip(
                ("a", "b", "c", "d", "e"),
                (formation_ax, lambda_y_x_ax, context_ax, lambda_z_x_ax, lambda_z_y_ax),
            ):
                text = axis.text2D if axis is formation_ax else axis.text
                text(0.035, 0.94, f"({label})", transform=axis.transAxes,
                     ha="left", va="top", fontsize=24, fontweight="bold", color="#111111")
        else:
            _label_geometry_summary_panels(
                [axis for axis in fig.axes if axis.get_subplotspec() is not None],
                grid_order=variant == "polyhedra",
            )
        fig.suptitle(
            f"HelioSwarm Design Reference Mission Geometry: {title_time}",
            x=0.055,
            y=0.965,
            ha="left",
        )
    if variant != "summary":
        _style_publication_geometry(fig, variant, relative, metadata)
        if variant == "formation":
            # Use spare header space without resizing any of the existing panels.
            colorbar_ax = fig.add_axes((0.52, 0.955, 0.15, 0.012))
            colorbar = fig.colorbar(
                plt.cm.ScalarMappable(norm=baseline_norm, cmap=baseline_cmap),
                cax=colorbar_ax,
                orientation="horizontal",
            )
            colorbar.set_label(r"$|\lambda|$ (km)", fontsize=16)
            colorbar.ax.xaxis.set_label_position("top")
            colorbar.ax.tick_params(labelsize=14)
        if variant == "polyhedra":
            selected = _selected_plot_polyhedra(metrics)
            _highlight_plot_polyhedra(poly_ax, selected)
            scatter.set_sizes([POLYHEDRON_POINT_AREA])
            scatter.set_alpha(0.38)
            for boundary in POLYHEDRON_EXTRA_Q_GUIDES:
                poly_ax.axvline(boundary, linestyle=":", color="#777777", lw=1.0, zorder=2)
            poly_ax.legend(loc="lower right", fontsize=16, frameon=True,
                           framealpha=0.9, edgecolor="none", handletextpad=0.4)
            poly_ax.text(REGULAR_POLYHEDRON_Q_LIMIT - 0.04, 0.05, r"regular $Q\leq0.6$",
                         transform=poly_ax.get_xaxis_transform(), ha="right", va="bottom",
                         fontsize=14, color="#555555", rotation=90)
            if "smallest regular" not in selected:
                poly_ax.text(0.03, 0.03, "No regular N=4 subset", transform=poly_ax.transAxes,
                             fontsize=14, color="#555555")
            diagnostics = dict(diagnostics)
            diagnostics["selected_polyhedra"] = {
                name: {key: item[key] for key in ("indices", "n_spacecraft", "E", "P", "Q", "L")}
                for name, item in selected.items()
            }
    return fig, diagnostics


def visualization_paths(output_path: str | Path, config: dict) -> tuple[Path, Path]:
    """Resolve explicit paths or derive them beside the HDF5 output."""

    output = Path(output_path)
    truth_path = Path(config.get("truth_path", output.with_name(f"{output.stem}_truth.png")))
    geometry_path = Path(
        config.get("geometry_path", output.with_name(f"{output.stem}_geometry.png"))
    )
    return truth_path, geometry_path


def observed_visualization_paths(
    output_path: str | Path, config: dict, mc_index: int
) -> tuple[Path, Path]:
    output = Path(output_path)
    observed_path = Path(
        config.get(
            "observed_path",
            output.with_name(f"{output.stem}_observed_mc{mc_index:03d}.png"),
        )
    )
    observed_geometry_path = Path(
        config.get(
            "observed_geometry_path",
            output.with_name(f"{output.stem}_geometry_observed_mc{mc_index:03d}.png"),
        )
    )
    return observed_path, observed_geometry_path


def alfven_visualization_path(output_path: str | Path, config: dict) -> Path:
    """Resolve the truth U/b diagnostic path."""

    output = Path(output_path)
    return Path(
        config.get("alfven_path", output.with_name(f"{output.stem}_velocity_alfven.png"))
    )


def observed_alfven_visualization_path(
    output_path: str | Path, config: dict, mc_index: int
) -> Path:
    """Resolve the observed U/b diagnostic path."""

    output = Path(output_path)
    return Path(
        config.get(
            "observed_alfven_path",
            output.with_name(f"{output.stem}_velocity_alfven_observed_mc{mc_index:03d}.png"),
        )
    )


def save_geometry_summary(
    *,
    output_path: str | Path,
    config: dict,
    time: np.ndarray,
    position: np.ndarray,
    spacecraft_ids: Sequence[str],
    metadata: Mapping[str, Any] | None = None,
    context_trajectory: np.ndarray | None = None,
    moon_position: np.ndarray | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Write all three geometry views as PDF/PNG; return the original summary paths."""

    dpi = int(config.get("dpi", 150))
    variant_paths = {}
    for variant in ("summary", "polyhedra", "formation"):
        pdf_path, png_path = geometry_summary_paths(output_path, metadata, variant=variant)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        figure, diagnostics = plot_geometry_summary(
            position,
            spacecraft_ids,
            time=time,
            metadata=metadata,
            context_trajectory=context_trajectory,
            moon_position=moon_position,
            hub_index=0,
            variant=variant,
        )
        try:
            figure.savefig(pdf_path)
            figure.savefig(png_path, dpi=dpi)
        finally:
            plt.close(figure)
        variant_paths[variant] = {"pdf": str(pdf_path), "png": str(png_path)}
    pdf_path, png_path = geometry_summary_paths(output_path, metadata)
    diagnostics = dict(diagnostics)
    diagnostics["output_pdf_path"] = str(pdf_path)
    diagnostics["output_png_path"] = str(png_path)
    diagnostics["variant_paths"] = variant_paths
    return pdf_path, png_path, diagnostics


def save_default_visualizations(
    *,
    output_path: str | Path,
    config: dict,
    time: Mapping[str, np.ndarray],
    truth: Mapping[str, np.ndarray],
    position: Mapping[str, np.ndarray],
    spacecraft_ids: Sequence[str],
    velocity_frame: str = "plasma",
) -> tuple[Path, Path]:
    """Write the default truth and geometry PNGs and close their figures."""

    truth_path, geometry_path = visualization_paths(output_path, config)
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    geometry_path.parent.mkdir(parents=True, exist_ok=True)
    dpi = int(config.get("dpi", 150))

    truth_figure, _ = plot_truth_timeseries(
        time, truth, spacecraft_ids, velocity_frame=velocity_frame
    )
    truth_figure.savefig(truth_path, dpi=dpi)
    plt.close(truth_figure)

    path_quantity = max(("n", "v", "B"), key=lambda quantity: len(time[quantity]))
    geometry_figure, _ = plot_spacecraft_geometry(
        time[path_quantity], position[path_quantity], spacecraft_ids
    )
    geometry_figure.savefig(geometry_path, dpi=dpi)
    plt.close(geometry_figure)
    if bool(config.get("alfven_enabled", True)):
        alfven_path = alfven_visualization_path(output_path, config)
        alfven_path.parent.mkdir(parents=True, exist_ok=True)
        alfven_figure, _ = plot_velocity_alfven_timeseries(
            time,
            truth,
            spacecraft_ids,
            velocity_frame=velocity_frame,
            fluctuations=bool(config.get("alfven_fluctuations", True)),
        )
        alfven_figure.savefig(alfven_path, dpi=dpi)
        plt.close(alfven_figure)
    return truth_path, geometry_path


def save_observed_visualizations(
    *,
    output_path: str | Path,
    config: dict,
    time: Mapping[str, np.ndarray],
    truth: Mapping[str, np.ndarray],
    observed: Mapping[str, np.ndarray],
    position: Mapping[str, np.ndarray],
    position_offset: np.ndarray,
    spacecraft_ids: Sequence[str],
    mc_index: int,
) -> tuple[Path, Path]:
    """Write observed-versus-truth overlays for one Monte Carlo realization."""

    observed_path, geometry_path = observed_visualization_paths(output_path, config, mc_index)
    observed_path.parent.mkdir(parents=True, exist_ok=True)
    geometry_path.parent.mkdir(parents=True, exist_ok=True)
    dpi = int(config.get("dpi", 150))

    timeseries_figure, _ = plot_observed_timeseries(
        time, truth, observed, spacecraft_ids, mc_index
    )
    timeseries_figure.savefig(observed_path, dpi=dpi)
    plt.close(timeseries_figure)

    path_quantity = max(("n", "v", "B"), key=lambda quantity: len(time[quantity]))
    geometry_figure, _ = plot_observed_spacecraft_geometry(
        time[path_quantity],
        position[path_quantity],
        position_offset,
        spacecraft_ids,
        mc_index,
    )
    geometry_figure.savefig(geometry_path, dpi=dpi)
    plt.close(geometry_figure)
    if bool(config.get("alfven_enabled", True)):
        alfven_path = observed_alfven_visualization_path(output_path, config, mc_index)
        alfven_path.parent.mkdir(parents=True, exist_ok=True)
        alfven_figure, _ = plot_observed_velocity_alfven_timeseries(
            time,
            truth,
            observed,
            spacecraft_ids,
            mc_index,
            fluctuations=bool(config.get("alfven_fluctuations", True)),
        )
        alfven_figure.savefig(alfven_path, dpi=dpi)
        plt.close(alfven_figure)
    return observed_path, geometry_path
