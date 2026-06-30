"""Simultaneous spatial decorrelation and second-order pair differences."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from hs_observatory.analysis.base import AnalysisMethod
from hs_observatory.analysis.registry import register_analysis

VARIABLES = ("n", "v", "B")
VARIABLE_COLORS = {"n": "#333333", "v": "#0072B2", "B": "#D55E00"}
COMPONENT_COLORS = ("#56B4E9", "#009E73", "#CC79A7")


def _settings(config: dict[str, Any]) -> dict[str, Any]:
    variables = list(config.get("variables", VARIABLES))
    if not variables or any(variable not in VARIABLES for variable in variables):
        raise ValueError("decorrelation.variables must contain one or more of: n, v, B")
    data_group = str(config.get("data_group", "observed"))
    if data_group not in {"observed", "truth"}:
        raise ValueError("decorrelation.data_group must be 'observed' or 'truth'")
    mc_index = config.get("mc_index", 0)
    if mc_index == "all":
        raise NotImplementedError("mc_index: all is not implemented; select one realization")
    mc_index = int(mc_index)
    if mc_index < 0:
        raise ValueError("decorrelation.mc_index must be non-negative")

    mean_config = config.get("mean", {})
    mean_method = str(mean_config.get("method", "per_time"))
    if mean_method not in {"per_time", "global"}:
        raise ValueError("decorrelation.mean.method must be 'per_time' or 'global'")

    separation = config.get("separation", {})
    if separation.get("units", "km") != "km":
        raise ValueError("Only kilometer separation units are currently supported")
    binning = str(separation.get("binning", "log"))
    if binning not in {"linear", "log"}:
        raise ValueError("decorrelation.separation.binning must be 'linear' or 'log'")
    n_bins = int(separation.get("n_bins", 30))
    if n_bins < 1:
        raise ValueError("decorrelation.separation.n_bins must be positive")

    correlation = config.get("correlation", {})
    if correlation.get("vector_mode", "dot") != "dot":
        raise ValueError("Only correlation.vector_mode: dot is currently supported")
    structure = config.get("structure_function", {})
    if int(structure.get("order", 2)) != 2:
        raise NotImplementedError("Only second-order structure functions are implemented")

    return {
        "variables": variables,
        "data_group": data_group,
        "mc_index": mc_index,
        "mean_method": mean_method,
        "separation_units": "km",
        "binning": binning,
        "n_bins": n_bins,
        "r_min": separation.get("r_min"),
        "r_max": separation.get("r_max"),
        "normalize": bool(correlation.get("normalize", True)),
        "component_correlations": bool(correlation.get("component_correlations", True)),
        "compute_structure_function": bool(structure.get("compute", True)),
        "structure_function_order": 2,
    }


def _read_variable(
    handle: h5py.File, variable: str, settings: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    data_path = f"{settings['data_group']}/{variable}"
    if data_path not in handle:
        raise KeyError(f"Requested variable '{variable}' is missing from /{settings['data_group']}")
    values = np.asarray(handle[data_path], dtype=float)
    if settings["data_group"] == "observed":
        if values.ndim < 3:
            raise ValueError(f"/{data_path} does not contain a leading Monte Carlo dimension")
        if settings["mc_index"] >= values.shape[0]:
            raise IndexError(
                f"mc_index {settings['mc_index']} is outside /{data_path} with "
                f"{values.shape[0]} realizations"
            )
        values = values[settings["mc_index"]]

    position_node = handle.get("spacecraft/r")
    if position_node is None:
        raise KeyError("Input file is missing /spacecraft/r")
    if isinstance(position_node, h5py.Group):
        if variable not in position_node:
            raise KeyError(f"Input file is missing /spacecraft/r/{variable}")
        position = np.asarray(position_node[variable], dtype=float)
    else:  # Legacy shared-coordinate schema.
        position = np.asarray(position_node, dtype=float)

    if position.ndim != 3 or position.shape[-1] != 3:
        raise ValueError(f"Spacecraft positions for {variable} must have shape (spacecraft, time, 3)")
    expected_value_shape = (position.shape[1], position.shape[0])
    if variable in {"v", "B"}:
        expected_value_shape = (*expected_value_shape, 3)
    if values.shape != expected_value_shape:
        raise ValueError(
            f"/{data_path} has shape {values.shape}, but positions require {expected_value_shape}"
        )
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(position)):
        raise ValueError(f"Variable {variable} or its positions contain non-finite values")
    return values, position


def _fluctuations(values: np.ndarray, method: str) -> np.ndarray:
    if method == "per_time":
        return values - np.mean(values, axis=1, keepdims=True)
    axes = (0, 1)
    return values - np.mean(values, axis=axes, keepdims=True)


def _pair_products(
    values: np.ndarray,
    position: np.ndarray,
    variable: str,
    settings: dict[str, Any],
) -> dict[str, np.ndarray]:
    n_spacecraft = position.shape[0]
    if n_spacecraft < 2:
        raise ValueError("Static decorrelation requires at least two spacecraft")
    left, right = np.triu_indices(n_spacecraft, k=1)
    # separation: (time, unique_pair), with unique pairs satisfying i < j.
    separation = np.linalg.norm(position[left] - position[right], axis=-1).T
    fluctuation = _fluctuations(values, settings["mean_method"])
    result: dict[str, np.ndarray] = {"separation": separation.reshape(-1)}

    if variable == "n":
        correlation = fluctuation[:, left] * fluctuation[:, right]
        if settings["normalize"]:
            denominator = float(np.mean(fluctuation**2))
            correlation = correlation / denominator if denominator > 0 else np.full_like(correlation, np.nan)
        result["correlation"] = correlation.reshape(-1)
        difference = values[:, left] - values[:, right]
        result["structure_function"] = (difference**2).reshape(-1)
        return result

    left_fluctuation = fluctuation[:, left, :]
    right_fluctuation = fluctuation[:, right, :]
    dot_correlation = np.sum(left_fluctuation * right_fluctuation, axis=-1)
    if settings["normalize"]:
        denominator = float(np.mean(np.sum(fluctuation**2, axis=-1)))
        dot_correlation = (
            dot_correlation / denominator
            if denominator > 0
            else np.full_like(dot_correlation, np.nan)
        )
    result["correlation_dot"] = dot_correlation.reshape(-1)

    if settings["component_correlations"]:
        component_correlation = left_fluctuation * right_fluctuation
        if settings["normalize"]:
            denominator = np.mean(fluctuation**2, axis=(0, 1))
            component_correlation = np.divide(
                component_correlation,
                denominator,
                out=np.full_like(component_correlation, np.nan),
                where=denominator > 0,
            )
        result["correlation_components"] = component_correlation.reshape(-1, 3)

    difference = values[:, left, :] - values[:, right, :]
    result["structure_function"] = np.sum(difference**2, axis=-1).reshape(-1)
    return result


def _bin_edges(separations: list[np.ndarray], settings: dict[str, Any]) -> np.ndarray:
    all_separations = np.concatenate(separations)
    if not np.all(np.isfinite(all_separations)):
        raise ValueError("Pair separations contain non-finite values")
    if settings["binning"] == "log" and np.any(all_separations <= 0):
        raise ValueError("Logarithmic separation bins require strictly positive separations")

    r_min = float(np.min(all_separations)) if settings["r_min"] is None else float(settings["r_min"])
    r_max = float(np.max(all_separations)) if settings["r_max"] is None else float(settings["r_max"])
    if settings["binning"] == "log" and r_min <= 0:
        raise ValueError("r_min must be positive for logarithmic bins")
    if r_max < r_min:
        raise ValueError("r_max must be greater than or equal to r_min")
    if r_max == r_min:
        if settings["binning"] == "log":
            r_min *= 0.9
            r_max *= 1.1
        else:
            padding = max(abs(r_min) * 0.1, 0.5)
            r_min -= padding
            r_max += padding
    if settings["binning"] == "log":
        return np.geomspace(r_min, r_max, settings["n_bins"] + 1)
    return np.linspace(r_min, r_max, settings["n_bins"] + 1)


def _binned_statistics(
    separation: np.ndarray, values: np.ndarray, edges: np.ndarray
) -> dict[str, np.ndarray]:
    n_bins = len(edges) - 1
    trailing_shape = values.shape[1:]
    mean = np.full((n_bins, *trailing_shape), np.nan, dtype=float)
    std = np.full_like(mean, np.nan)
    stderr = np.full_like(mean, np.nan)
    counts = np.zeros(n_bins, dtype=np.int64)
    bin_index = np.searchsorted(edges, separation, side="right") - 1
    bin_index[separation == edges[-1]] = n_bins - 1

    for index in range(n_bins):
        selected = values[bin_index == index]
        counts[index] = len(selected)
        if len(selected) == 0:
            continue
        if trailing_shape:
            for component in np.ndindex(trailing_shape):
                component_values = selected[(slice(None), *component)]
                finite = component_values[np.isfinite(component_values)]
                if len(finite):
                    mean[(index, *component)] = np.mean(finite)
                    std[(index, *component)] = np.std(finite)
                    stderr[(index, *component)] = std[(index, *component)] / np.sqrt(len(finite))
        else:
            finite = selected[np.isfinite(selected)]
            if len(finite):
                mean[index] = np.mean(finite)
                std[index] = np.std(finite)
                stderr[index] = std[index] / np.sqrt(len(finite))
    return {"count": counts, "mean": mean, "std": std, "stderr": stderr}


def compute_static_decorrelation(
    input_path: str | Path, config: dict[str, Any]
) -> dict[str, Any]:
    settings = _settings(config)
    pair_data: dict[str, dict[str, np.ndarray]] = {}
    with h5py.File(input_path, "r") as handle:
        for variable in settings["variables"]:
            values, position = _read_variable(handle, variable, settings)
            pair_data[variable] = _pair_products(values, position, variable, settings)

    edges = _bin_edges(
        [products["separation"] for products in pair_data.values()], settings
    )
    result: dict[str, Any] = {
        "metadata": {
            "method": "static_decorrelation",
            "input_path": str(Path(input_path)),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": settings,
            "assumption": "simultaneous spatial snapshots; no temporal decorrelation",
            "std_ddof": 0,
        },
        "bin_edges": edges,
        "bin_lower": edges[:-1],
        "bin_upper": edges[1:],
        "bin_centers": (
            np.sqrt(edges[:-1] * edges[1:])
            if settings["binning"] == "log"
            else 0.5 * (edges[:-1] + edges[1:])
        ),
        "counts": {},
        "correlation": {},
        "structure_function": {},
    }
    for variable, products in pair_data.items():
        separation = products["separation"]
        correlation_name = variable if variable == "n" else f"{variable}_dot"
        correlation_values = (
            products["correlation"] if variable == "n" else products["correlation_dot"]
        )
        correlation_stats = _binned_statistics(separation, correlation_values, edges)
        result["counts"][variable] = correlation_stats.pop("count")
        result["correlation"][correlation_name] = correlation_stats
        if variable != "n" and settings["component_correlations"]:
            component_stats = _binned_statistics(
                separation, products["correlation_components"], edges
            )
            component_stats.pop("count")
            result["correlation"][f"{variable}_components"] = component_stats
        if settings["compute_structure_function"]:
            structure_stats = _binned_statistics(
                separation, products["structure_function"], edges
            )
            structure_stats.pop("count")
            result["structure_function"][variable] = structure_stats
    return result


def plot_static_decorrelation(result: dict[str, Any]):
    """Create correlation and structure-function diagnostics from a result."""

    import matplotlib.pyplot as plt

    configuration = result["metadata"]["configuration"]
    variables = configuration["variables"]
    figure, axes = plt.subplots(
        len(variables),
        2,
        figsize=(13, 3.6 * len(variables)),
        squeeze=False,
        constrained_layout=True,
    )
    separation = result["bin_centers"]
    component_labels = ("x", "y", "z")

    for row, variable in enumerate(variables):
        correlation_axis = axes[row, 0]
        structure_axis = axes[row, 1]
        correlation_name = variable if variable == "n" else f"{variable}_dot"
        statistics = result["correlation"][correlation_name]
        mean = statistics["mean"]
        stderr = statistics["stderr"]
        correlation_axis.plot(
            separation,
            mean,
            marker="o",
            markersize=3.5,
            color=VARIABLE_COLORS[variable],
            label=f"{variable} " + ("scalar" if variable == "n" else "dot"),
        )
        correlation_axis.fill_between(
            separation,
            mean - stderr,
            mean + stderr,
            color=VARIABLE_COLORS[variable],
            alpha=0.18,
            linewidth=0,
        )
        component_name = f"{variable}_components"
        if component_name in result["correlation"]:
            component_mean = result["correlation"][component_name]["mean"]
            for component, label, color in zip(
                range(3), component_labels, COMPONENT_COLORS
            ):
                correlation_axis.plot(
                    separation,
                    component_mean[:, component],
                    linestyle="--",
                    linewidth=1.2,
                    color=color,
                    label=f"{variable}{label}",
                )
        correlation_axis.axhline(0.0, color="#777777", linewidth=0.8, alpha=0.6)
        correlation_axis.set_ylabel("Normalized correlation" if configuration["normalize"] else "Pair product")
        correlation_axis.set_title(f"{variable}: simultaneous correlation")
        correlation_axis.legend(fontsize=8, ncol=2)
        correlation_axis.grid(alpha=0.25)

        if variable in result["structure_function"]:
            structure_stats = result["structure_function"][variable]
            structure_mean = structure_stats["mean"]
            structure_stderr = structure_stats["stderr"]
            structure_axis.plot(
                separation,
                structure_mean,
                marker="o",
                markersize=3.5,
                color=VARIABLE_COLORS[variable],
                label=rf"$S_2({variable})$",
            )
            structure_axis.fill_between(
                separation,
                np.maximum(0.0, structure_mean - structure_stderr),
                structure_mean + structure_stderr,
                color=VARIABLE_COLORS[variable],
                alpha=0.18,
                linewidth=0,
            )
            structure_axis.set_ylabel(r"Second-order structure function")
            structure_axis.set_title(f"{variable}: pair differences")
            structure_axis.legend(fontsize=8)
            structure_axis.grid(alpha=0.25)
        else:
            structure_axis.axis("off")
            structure_axis.text(
                0.5,
                0.5,
                "Structure function disabled",
                ha="center",
                va="center",
                transform=structure_axis.transAxes,
            )

        for axis in (correlation_axis, structure_axis):
            if axis.axison:
                axis.set_xscale(configuration["binning"])
                axis.set_xlabel("Separation (km)")

    data_label = configuration["data_group"]
    if data_label == "observed":
        data_label += f" MC {configuration['mc_index']}"
    figure.suptitle(f"Static Spatial Decorrelation — {data_label}")
    return figure, axes


def _diagnostic_plot_settings(
    output_path: str | Path, config: dict[str, Any]
) -> tuple[bool, Path, int]:
    visualization = config.get("visualization", {})
    if not isinstance(visualization, dict):
        raise ValueError("decorrelation.visualization must be a mapping")
    enabled = bool(visualization.get("enabled", True))
    output = Path(output_path)
    path = Path(
        visualization.get(
            "path", output.with_name(f"{output.stem}_diagnostic.png")
        )
    )
    dpi = int(visualization.get("dpi", 150))
    if dpi <= 0:
        raise ValueError("decorrelation.visualization.dpi must be positive")
    return enabled, path, dpi


@register_analysis
class StaticDecorrelationAnalysis(AnalysisMethod):
    name = "static_decorrelation"
    config_section = "decorrelation"

    def run(self, input_path: str | Path) -> dict[str, Any]:
        return compute_static_decorrelation(input_path, self.config)

    def write_result(self, result: dict[str, Any], output_path: str | Path) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        plot_enabled, plot_path, plot_dpi = _diagnostic_plot_settings(output, self.config)
        result["metadata"]["diagnostic_plot_enabled"] = plot_enabled
        result["metadata"]["diagnostic_plot_path"] = str(plot_path) if plot_enabled else "null"
        with h5py.File(output, "w") as handle:
            metadata = handle.create_group("metadata")
            for key, value in result["metadata"].items():
                metadata.attrs[key] = (
                    value
                    if isinstance(value, (str, int, float, bool))
                    else json.dumps(value, sort_keys=True)
                )
            for name in ("bin_edges", "bin_lower", "bin_upper", "bin_centers"):
                dataset = handle.create_dataset(name, data=result[name])
                dataset.attrs["units"] = result["metadata"]["configuration"][
                    "separation_units"
                ]
            counts = handle.create_group("counts")
            for variable, values in result["counts"].items():
                counts.create_dataset(variable, data=values)
            for root_name in ("correlation", "structure_function"):
                root = handle.create_group(root_name)
                for product_name, statistics in result[root_name].items():
                    group = root.create_group(product_name)
                    for statistic, values in statistics.items():
                        group.create_dataset(statistic, data=values)
                    if product_name.endswith("_components"):
                        group.attrs["components"] = np.asarray(
                            ["x", "y", "z"], dtype=h5py.string_dtype("utf-8")
                        )
        if plot_enabled:
            import matplotlib.pyplot as plt

            plot_path.parent.mkdir(parents=True, exist_ok=True)
            figure, _ = plot_static_decorrelation(result)
            figure.savefig(plot_path, dpi=plot_dpi)
            plt.close(figure)
        return output
