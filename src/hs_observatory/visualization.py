"""Default visual checks for generated truth data and spacecraft geometry."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

# Generation is commonly run on headless systems and batch nodes.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
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
