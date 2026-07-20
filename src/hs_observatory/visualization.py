"""Default visual checks for generated truth data and spacecraft geometry."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence
import warnings

import matplotlib

# Generation is commonly run on headless systems and batch nodes.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np

from hs_observatory.analysis.derived.elsasser import magnetic_field_to_alfven_units


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


REGION_COLORS = {
    "Solar Wind": "#d73027",
    "Connected Foreshock": "#1a9850",
    "Magnetosphere / Magnetosheath": "#4575b4",
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
        return str(np.datetime64(label, "s"))
    except ValueError:
        return label.split(".")[0]


def geometry_summary_paths(
    output_path: str | Path,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Return default PDF and PNG paths for the publication geometry summary."""

    output = Path(output_path)
    tag = _selected_time_label(metadata)
    safe_tag = "".join(char if char.isalnum() else "-" for char in tag).strip("-")
    safe_tag = "-".join(part for part in safe_tag.split("-") if part) or "unknown-time"
    directory = output.parent / "geometry_overviews"
    base = directory / f"{output.stem}_geometry_summary_{safe_tag}"
    return base.with_suffix(".pdf"), base.with_suffix(".png")


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
    x_index: int,
    y_index: int,
    title: str,
) -> None:
    labels = ("X", "Y", "Z")
    extent = float(np.max(np.abs(relative[:, [x_index, y_index]])))
    limit = 1.12 * extent if extent > 0 else 1.0
    ax.scatter(
        relative[1:, x_index],
        relative[1:, y_index],
        s=120,
        c="#111111",
        edgecolors="white",
        linewidths=0.45,
        zorder=3,
    )
    ax.scatter(
        relative[0, x_index],
        relative[0, y_index],
        s=192,
        c="#d62728",
        edgecolors="#111111",
        linewidths=0.55,
        zorder=4,
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
    handles.extend(
        Line2D([0], [0], color=color, linewidth=1.3, label=name)
        for name, color in REGION_COLORS.items()
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
        c="#252525",
        alpha=0.82,
        edgecolors="white",
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


def plot_geometry_summary(
    position: np.ndarray,
    spacecraft_ids: Sequence[str],
    *,
    time: np.ndarray | None = None,
    metadata: Mapping[str, Any] | None = None,
    context_trajectory: np.ndarray | None = None,
    moon_position: np.ndarray | None = None,
    hub_index: int = 0,
):
    """Return a compact publication-style geometry summary figure."""

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
    mesocenter = np.mean(sample, axis=0)
    title_time = _selected_time_title_label(metadata)
    region = diagnostics["region"]
    regular_exists = diagnostics["n_regular_polyhedra"] > 0
    ratio_exists = diagnostics["regular_size_ratio_at_least_3"]

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
        fig = plt.figure(figsize=(31.0, 16.0), constrained_layout=True)
        layout_engine = fig.get_layout_engine()
        if layout_engine is not None:
            layout_engine.set(
                rect=(0.0, 0.0, 1.0, 0.91),
                h_pad=0.035,
                w_pad=0.035,
                hspace=0.08,
                wspace=0.035,
            )
        grid = fig.add_gridspec(2, 4, width_ratios=(1.0, 1.12, 1.0, 1.18))

        _plot_summary_relative_projection(
            fig.add_subplot(grid[0, 0]),
            relative,
            0,
            1,
            r"$\Delta Y_{GSE}$ vs $\Delta X_{GSE}$",
        )

        context_ax = fig.add_subplot(grid[0, 1])
        _plot_context_projection(
            context_ax,
            trajectory,
            mesocenter,
            0,
            1,
            r"Orbit Context: $Y_{GSE}$ vs $X_{GSE}$ ($R_E$)",
            moon_position=moon_position,
        )
        context_ax.legend(
            handles=_context_legend_handles(moon_position),
            loc="lower left",
            fontsize=19.8,
            frameon=False,
            borderaxespad=0.25,
        )

        baseline_limits = _baseline_component_limits(baselines)
        _plot_baseline_components(
            fig.add_subplot(grid[0, 2]),
            baselines,
            0,
            1,
            r"$|\lambda_Y|$ vs $|\lambda_X|$",
            baseline_limits,
        )

        poly_ax = fig.add_subplot(grid[0, 3])
        cmap = _discrete_polyhedron_cmap()
        norm = BoundaryNorm(np.arange(3.5, 10.5, 1.0), cmap.N)
        q_values = np.asarray([item["Q"] for item in metrics], dtype=float)
        l_values = np.asarray([item["L"] for item in metrics], dtype=float)
        sizes = np.asarray([item["n_spacecraft"] for item in metrics], dtype=float)
        poly_ax.axvspan(
            0.0,
            REGULAR_POLYHEDRON_Q_LIMIT,
            color="#90be6d",
            alpha=0.14,
            label="regular region",
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
            xlabel=r"$Q = \sqrt{E^2 + P^2}$",
            ylabel=r"$L = 2a$ (km)",
            title="Polyhedral Configuration Summary",
        )
        poly_ax.grid(alpha=0.2, linewidth=0.5, which="both")
        status_text = (
            f"Region: {region}\n"
            f"Regular: {'yes' if regular_exists else 'no'}\n"
            f"Size ratio >= 3: {'yes' if ratio_exists else 'no'}"
        )
        poly_ax.text(
            0.02,
            0.96,
            status_text,
            va="top",
            ha="left",
            fontsize=20.4,
            color="#222222",
            transform=poly_ax.transAxes,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 2.0},
        )
        colorbar = fig.colorbar(
            scatter,
            ax=poly_ax,
            ticks=np.arange(4, 10),
            pad=0.01,
            fraction=0.045,
        )
        colorbar.set_label("# spacecraft")
        colorbar.ax.tick_params(labelsize=24)

        _plot_summary_relative_projection(
            fig.add_subplot(grid[1, 0]),
            relative,
            0,
            2,
            r"$\Delta Z_{GSE}$ vs $\Delta X_{GSE}$",
        )
        _plot_summary_relative_projection(
            fig.add_subplot(grid[1, 1]),
            relative,
            1,
            2,
            r"$\Delta Z_{GSE}$ vs $\Delta Y_{GSE}$",
        )
        lambda_z_x_ax = fig.add_subplot(grid[1, 2])
        _plot_baseline_components(
            lambda_z_x_ax,
            baselines,
            0,
            2,
            r"$|\lambda_Z|$ vs $|\lambda_X|$",
            baseline_limits,
        )
        lambda_z_y_ax = fig.add_subplot(grid[1, 3])
        _plot_baseline_components(
            lambda_z_y_ax,
            baselines,
            1,
            2,
            r"$|\lambda_Z|$ vs $|\lambda_Y|$",
            baseline_limits,
        )
        lambda_z_y_ax.text(
            0.04,
            0.96,
            f"Kinetic: < {KINETIC_SCALE_KM:g} km\n"
            f"Transition: {KINETIC_SCALE_KM:g}-{MHD_SCALE_KM:g} km\n"
            f"MHD: > {MHD_SCALE_KM:g} km",
            va="top",
            ha="left",
            fontsize=20.4,
            color="#222222",
            transform=lambda_z_y_ax.transAxes,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.68, "pad": 2.0},
        )

        fig.suptitle(
            f"HelioSwarm Geometry Summary - Matched configuration: {title_time}",
            y=0.985,
        )
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
    """Write the publication-style geometry summary as PDF and PNG."""

    pdf_path, png_path = geometry_summary_paths(output_path, metadata)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    dpi = int(config.get("dpi", 150))
    figure, diagnostics = plot_geometry_summary(
        position,
        spacecraft_ids,
        time=time,
        metadata=metadata,
        context_trajectory=context_trajectory,
        moon_position=moon_position,
        hub_index=0,
    )
    figure.savefig(pdf_path)
    figure.savefig(png_path, dpi=dpi)
    plt.close(figure)
    diagnostics = dict(diagnostics)
    diagnostics["output_pdf_path"] = str(pdf_path)
    diagnostics["output_png_path"] = str(png_path)
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
