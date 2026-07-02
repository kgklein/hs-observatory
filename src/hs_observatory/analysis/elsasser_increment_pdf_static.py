"""Static all-point-pair PDFs of Elsasser increment amplitudes."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any
import warnings

import h5py
import numpy as np

from hs_observatory.analysis.base import AnalysisMethod
from hs_observatory.analysis.derived.pairs import (
    local_pairwise_directions,
    pair_chunks,
    perpendicular_projection,
    possible_pair_count,
    selected_pair_count,
    separation_magnitude,
)
from hs_observatory.analysis.elsasser import (
    _interpolate,
    _time_axis,
    compute_elsasser_analysis,
)
from hs_observatory.analysis.registry import register_analysis

FIELDS = ("z_plus", "z_minus")


def geometric_mean_positive(values: np.ndarray) -> float:
    """Return ``exp(mean(log(x)))`` over finite, strictly positive values."""

    array = np.asarray(values, dtype=float)
    selected = array[np.isfinite(array) & (array > 0.0)]
    return float(np.exp(np.mean(np.log(selected)))) if selected.size else np.nan


def pdf_density(values: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return histogram counts and a density integrating to one over the edges."""

    counts, _ = np.histogram(np.asarray(values, dtype=float), bins=edges)
    widths = np.diff(edges)
    if np.sum(counts) == 0:
        return counts, np.full(widths.shape, np.nan)
    return counts, counts / (float(np.sum(counts)) * widths)


def _mapping(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"elsasser_increment_pdf_static.{name} must be a mapping")
    return value


def _settings(config: dict[str, Any]) -> dict[str, Any]:
    datasets = config.get(
        "datasets",
        [
            {"name": "truth", "data_group": "truth"},
            {"name": "observed_mc0", "data_group": "observed", "mc_index": 0},
        ],
    )
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("elsasser_increment_pdf_static.datasets must be a nonempty list")
    parsed_datasets = []
    names: set[str] = set()
    for dataset in datasets:
        if not isinstance(dataset, dict):
            raise ValueError("Every configured dataset must be a mapping")
        name = str(dataset.get("name", ""))
        if not name or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError("Dataset names may contain only letters, numbers, '.', '_', and '-'")
        if name in names:
            raise ValueError(f"Duplicate dataset name: {name}")
        names.add(name)
        data_group = str(dataset.get("data_group", ""))
        if data_group not in {"truth", "observed"}:
            raise ValueError("Dataset data_group must be 'truth' or 'observed'")
        mc_index = dataset.get("mc_index", 0)
        if mc_index == "all":
            raise NotImplementedError("MC ensemble aggregation is not implemented")
        mc_index = int(mc_index)
        if mc_index < 0:
            raise ValueError("Dataset mc_index must be non-negative")
        parsed_datasets.append({"name": name, "data_group": data_group, "mc_index": mc_index})

    elsasser = _mapping(config, "elsasser")
    increments = _mapping(config, "increments")
    if str(increments.get("field", "both")) != "both":
        raise NotImplementedError("Both z_plus and z_minus are always computed")
    plotted_field = str(increments.get("plotted_field", "z_plus"))
    if plotted_field not in FIELDS:
        raise ValueError("increments.plotted_field must be z_plus or z_minus")
    component = str(increments.get("component", "perpendicular"))
    if component not in {"perpendicular", "full_vector"}:
        raise ValueError("increments.component must be perpendicular or full_vector")
    if str(increments.get("statistic", "magnitude")) != "magnitude":
        raise NotImplementedError("Only increment statistic: magnitude is supported")

    mean_field = _mapping(config, "mean_field")
    if str(mean_field.get("method", "local_pairwise")) != "local_pairwise":
        raise NotImplementedError("Only mean_field.method: local_pairwise is supported")
    if str(mean_field.get("fallback", "global_mean_B")) != "global_mean_B":
        raise NotImplementedError("Only mean_field.fallback: global_mean_B is supported")
    min_norm = float(mean_field.get("min_norm", 1.0e-12))
    if min_norm < 0.0:
        raise ValueError("mean_field.min_norm must be non-negative")

    scale = _mapping(config, "scale_bins")
    separation = str(scale.get("separation", "perpendicular"))
    if separation not in {"perpendicular", "full"}:
        raise ValueError("scale_bins.separation must be perpendicular or full")
    scale_binning = str(scale.get("binning", "log"))
    if scale_binning not in {"log", "linear"}:
        raise ValueError("scale_bins.binning must be log or linear")
    n_scale_bins = int(scale.get("n_bins", 12))
    min_samples = int(scale.get("min_samples_per_bin", 50))
    if n_scale_bins < 1 or min_samples < 1:
        raise ValueError("Scale bin counts and minimum samples must be positive")

    pdf = _mapping(config, "pdf_bins")
    if str(pdf.get("normalized_by", "geometric_mean")) != "geometric_mean":
        raise NotImplementedError("Only geometric-mean PDF normalization is supported")
    if str(pdf.get("binning", "linear")) != "linear":
        raise NotImplementedError("Only linear PDF bins are supported")
    n_pdf_bins = int(pdf.get("n_bins", 100))
    x_min = float(pdf.get("x_min", 0.0))
    x_max = pdf.get("x_max", "auto_percentile")
    if x_max != "auto_percentile":
        x_max = float(x_max)
        if x_max <= x_min:
            raise ValueError("pdf_bins.x_max must exceed x_min")
    percentile = float(pdf.get("x_max_percentile", 99.5))
    if not 0.0 < percentile <= 100.0 or n_pdf_bins < 1:
        raise ValueError("PDF bin count must be positive and percentile in (0, 100]")

    sampling = _mapping(config, "pair_sampling")
    mode = str(sampling.get("mode", "all"))
    if mode not in {"all", "random"}:
        raise ValueError("pair_sampling.mode must be all or random")
    chunk_size = int(sampling.get("chunk_size", 1_000_000))
    n_pairs = sampling.get("n_pairs")
    n_pairs = None if n_pairs is None else int(n_pairs)
    max_pairs = sampling.get("max_pairs")
    max_pairs = None if max_pairs is None else int(max_pairs)
    random_seed = sampling.get("random_seed", 12345)
    random_seed = None if random_seed is None else int(random_seed)
    if chunk_size < 1:
        raise ValueError("pair_sampling.chunk_size must be positive")
    if mode == "random" and (n_pairs is None or n_pairs < 1):
        raise ValueError("random pair sampling requires pair_sampling.n_pairs")
    if mode == "random" and random_seed is None:
        raise ValueError("random pair sampling requires a non-null random_seed across passes")
    if max_pairs is not None and max_pairs < 1:
        raise ValueError("pair_sampling.max_pairs must be positive or null")

    output_samples = _mapping(config, "output_samples")
    max_samples = int(output_samples.get("max_samples", 100_000))
    if max_samples < 1:
        raise ValueError("output_samples.max_samples must be positive")
    plot = _mapping(config, "plot")
    if str(plot.get("yscale", "log")) not in {"linear", "log"}:
        raise ValueError("plot.yscale must be linear or log")
    if str(plot.get("xscale", "linear")) not in {"linear", "log"}:
        raise ValueError("plot.xscale must be linear or log")
    dpi = int(plot.get("dpi", 150))
    if dpi < 1:
        raise ValueError("plot.dpi must be positive")

    alignment = _mapping(config, "alignment")
    return {
        "datasets": parsed_datasets,
        "elsasser": elsasser,
        "increment_component": component,
        "plotted_field": plotted_field,
        "mean_field_method": "local_pairwise",
        "mean_field_fallback": "global_mean_B",
        "min_norm": min_norm,
        "separation": separation,
        "scale_binning": scale_binning,
        "n_scale_bins": n_scale_bins,
        "lambda_min": scale.get("lambda_min", "auto"),
        "lambda_max": scale.get("lambda_max", "auto"),
        "min_samples_per_bin": min_samples,
        "n_pdf_bins": n_pdf_bins,
        "x_min": x_min,
        "x_max": x_max,
        "x_max_percentile": percentile,
        "pair_mode": mode,
        "chunk_size": chunk_size,
        "n_pairs": n_pairs,
        "max_pairs": max_pairs,
        "random_seed": random_seed,
        "alignment_compute": bool(alignment.get("compute", True)),
        "save_samples": bool(output_samples.get("save", False)),
        "max_samples": max_samples,
        "plot": {
            "enabled": bool(plot.get("enabled", True)),
            "directory": plot.get("directory"),
            "yscale": str(plot.get("yscale", "log")),
            "xscale": str(plot.get("xscale", "linear")),
            "dpi": dpi,
        },
    }


def _read_positions(input_path: str | Path, output_time: np.ndarray) -> np.ndarray:
    """Return positions on the Elsasser grid as ``(time, spacecraft, 3)``."""

    with h5py.File(input_path, "r") as handle:
        if "spacecraft/r" not in handle:
            raise KeyError("Input file is missing /spacecraft/r")
        node = handle["spacecraft/r"]
        if isinstance(node, h5py.Group):
            candidates = []
            for variable in ("v", "n", "B"):
                if variable in node:
                    source_time = _time_axis(handle, variable)
                    candidates.append((variable, source_time))
            if not candidates:
                raise KeyError("Input file has no variable-specific spacecraft positions")
            exact = next(
                (item for item in candidates if np.array_equal(item[1], output_time)),
                candidates[0],
            )
            variable, source_time = exact
            position = np.asarray(node[variable], dtype=float)
        else:
            position = np.asarray(node, dtype=float)
            time_node = handle["time"]
            if isinstance(time_node, h5py.Group):
                source_time = _time_axis(handle, "v")
            else:
                source_time = np.asarray(time_node, dtype=float)
    if position.ndim != 3 or position.shape[-1] != 3:
        raise ValueError("Spacecraft positions must have shape (spacecraft, time, 3)")
    time_first = np.swapaxes(position, 0, 1)
    if time_first.shape[0] != source_time.size:
        raise ValueError("Spacecraft positions do not match their source time grid")
    return _interpolate(time_first, source_time, output_time, "spacecraft position")


def _chunks(n_points: int, settings: dict[str, Any]):
    return pair_chunks(
        n_points,
        settings["chunk_size"],
        mode=settings["pair_mode"],
        n_pairs=settings["n_pairs"],
        random_seed=settings["random_seed"],
        max_pairs=settings["max_pairs"],
    )


def _pair_values(
    left: np.ndarray,
    right: np.ndarray,
    positions: np.ndarray,
    z_plus: np.ndarray,
    z_minus: np.ndarray,
    B: np.ndarray,
    global_B: np.ndarray,
    settings: dict[str, Any],
) -> dict[str, np.ndarray]:
    directions, valid_direction = local_pairwise_directions(
        B[left], B[right], global_B, min_norm=settings["min_norm"]
    )
    displacement = positions[right] - positions[left]
    separation = separation_magnitude(displacement, directions, settings["separation"])
    deltas = {
        "z_plus": z_plus[right] - z_plus[left],
        "z_minus": z_minus[right] - z_minus[left],
    }
    perpendicular = {
        name: perpendicular_projection(values, directions) for name, values in deltas.items()
    }
    selected = perpendicular if settings["increment_component"] == "perpendicular" else deltas
    increments = {name: np.linalg.norm(selected[name], axis=1) for name in FIELDS}
    cross = np.linalg.norm(np.cross(perpendicular["z_plus"], perpendicular["z_minus"]), axis=1)
    denominator = np.linalg.norm(perpendicular["z_plus"], axis=1) * np.linalg.norm(
        perpendicular["z_minus"], axis=1
    )
    sin_theta = np.divide(
        cross,
        denominator,
        out=np.full(cross.shape, np.nan),
        where=denominator > 0.0,
    )
    return {
        "lambda": separation,
        "z_plus": increments["z_plus"],
        "z_minus": increments["z_minus"],
        "sin_theta": sin_theta,
        "valid_direction": valid_direction,
    }


def _bin_indices(values: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.searchsorted(edges, values, side="right") - 1
    indices[values == edges[-1]] = len(edges) - 2
    valid = np.isfinite(values) & (indices >= 0) & (indices < len(edges) - 1)
    return indices, valid


def _priority_sample_update(
    current_values: np.ndarray,
    current_keys: np.ndarray,
    new_values: np.ndarray,
    rng: np.random.Generator,
    limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    if new_values.size == 0:
        return current_values, current_keys
    new_keys = rng.random(new_values.size)
    values = np.concatenate((current_values, new_values))
    keys = np.concatenate((current_keys, new_keys))
    if values.size > limit:
        keep = np.argpartition(keys, limit - 1)[:limit]
        values = values[keep]
        keys = keys[keep]
    return values, keys


def compute_increment_pdf_for_points(
    positions: np.ndarray,
    z_plus: np.ndarray,
    z_minus: np.ndarray,
    B_used: np.ndarray,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Compute scale-conditioned PDFs from time-first point arrays.

    All four inputs have shape ``(time, spacecraft, 3)`` and are flattened in
    time-major order before unique unordered pair streaming.
    """

    arrays = [np.asarray(value, dtype=float) for value in (positions, z_plus, z_minus, B_used)]
    if any(value.shape != arrays[0].shape for value in arrays):
        raise ValueError("positions, z_plus, z_minus, and B_used must have matching shapes")
    if arrays[0].ndim != 3 or arrays[0].shape[-1] != 3:
        raise ValueError("Point arrays must have shape (time, spacecraft, 3)")
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("Point arrays must contain only finite values")
    n_time, n_spacecraft, _ = arrays[0].shape
    flat_position, flat_plus, flat_minus, flat_B = [value.reshape(-1, 3) for value in arrays]
    n_points = flat_position.shape[0]
    if n_points < 2:
        raise ValueError("At least two spacetime points are required")
    global_B = np.mean(flat_B, axis=0)
    pair_count = selected_pair_count(
        n_points,
        mode=settings["pair_mode"],
        n_pairs=settings["n_pairs"],
        max_pairs=settings["max_pairs"],
    )

    inferred_min = np.inf
    inferred_max = -np.inf
    skipped_direction = 0
    skipped_nonpositive = 0
    for left, right in _chunks(n_points, settings):
        values = _pair_values(
            left, right, flat_position, flat_plus, flat_minus, flat_B, global_B, settings
        )
        valid = values["valid_direction"] & np.isfinite(values["lambda"])
        skipped_direction += int(np.sum(~values["valid_direction"]))
        if settings["scale_binning"] == "log":
            skipped_nonpositive += int(np.sum(valid & (values["lambda"] <= 0.0)))
            valid &= values["lambda"] > 0.0
        selected_lambda = values["lambda"][valid]
        if selected_lambda.size:
            inferred_min = min(inferred_min, float(np.min(selected_lambda)))
            inferred_max = max(inferred_max, float(np.max(selected_lambda)))
    if not np.isfinite(inferred_min) or not np.isfinite(inferred_max):
        raise ValueError("No valid pair separations are available for scale binning")
    lambda_min = (
        inferred_min if settings["lambda_min"] in {"auto", None} else float(settings["lambda_min"])
    )
    lambda_max = (
        inferred_max if settings["lambda_max"] in {"auto", None} else float(settings["lambda_max"])
    )
    if settings["scale_binning"] == "log" and lambda_min <= 0.0:
        raise ValueError("lambda_min must be positive for logarithmic bins")
    if lambda_max < lambda_min:
        raise ValueError("lambda_max must be at least lambda_min")
    if lambda_max == lambda_min:
        if settings["scale_binning"] == "log":
            lambda_min *= 0.9
            lambda_max *= 1.1
        else:
            padding = max(abs(lambda_min) * 0.1, 0.5)
            lambda_min -= padding
            lambda_max += padding
    if settings["scale_binning"] == "log":
        lambda_edges = np.geomspace(lambda_min, lambda_max, settings["n_scale_bins"] + 1)
        lambda_centers = np.sqrt(lambda_edges[:-1] * lambda_edges[1:])
    else:
        lambda_edges = np.linspace(lambda_min, lambda_max, settings["n_scale_bins"] + 1)
        lambda_centers = 0.5 * (lambda_edges[:-1] + lambda_edges[1:])

    n_bins = settings["n_scale_bins"]
    lambda_counts = np.zeros(n_bins, dtype=np.int64)
    positive_counts = {name: np.zeros(n_bins, dtype=np.int64) for name in FIELDS}
    zero_counts = {name: np.zeros(n_bins, dtype=np.int64) for name in FIELDS}
    sum_log = {name: np.zeros(n_bins) for name in FIELDS}
    sum_second = {name: np.zeros(n_bins) for name in FIELDS}
    sum_fourth = {name: np.zeros(n_bins) for name in FIELDS}
    alignment_sum = np.zeros(n_bins)
    alignment_counts = np.zeros(n_bins, dtype=np.int64)
    outside_scale_bins = 0
    for left, right in _chunks(n_points, settings):
        values = _pair_values(
            left, right, flat_position, flat_plus, flat_minus, flat_B, global_B, settings
        )
        indices, valid_bin = _bin_indices(values["lambda"], lambda_edges)
        valid_bin &= values["valid_direction"]
        outside_scale_bins += int(np.sum(values["valid_direction"] & ~valid_bin))
        for index in range(n_bins):
            selected = valid_bin & (indices == index)
            count = int(np.sum(selected))
            lambda_counts[index] += count
            if count == 0:
                continue
            for name in FIELDS:
                increments = values[name][selected]
                positive = np.isfinite(increments) & (increments > 0.0)
                positive_values = increments[positive]
                positive_counts[name][index] += positive_values.size
                zero_counts[name][index] += int(np.sum(increments == 0.0))
                sum_log[name][index] += float(np.sum(np.log(positive_values)))
                finite = increments[np.isfinite(increments)]
                sum_second[name][index] += float(np.sum(finite**2))
                sum_fourth[name][index] += float(np.sum(finite**4))
            if settings["alignment_compute"]:
                angles = values["sin_theta"][selected]
                finite_angles = angles[np.isfinite(angles)]
                alignment_sum[index] += float(np.sum(finite_angles))
                alignment_counts[index] += finite_angles.size

    typical: dict[str, np.ndarray] = {}
    rms: dict[str, np.ndarray] = {}
    fourth_root: dict[str, np.ndarray] = {}
    for name in FIELDS:
        typical[name] = np.exp(
            np.divide(
                sum_log[name],
                positive_counts[name],
                out=np.full(n_bins, np.nan),
                where=positive_counts[name] > 0,
            )
        )
        rms[name] = np.sqrt(
            np.divide(
                sum_second[name],
                lambda_counts,
                out=np.full(n_bins, np.nan),
                where=lambda_counts > 0,
            )
        )
        fourth_root[name] = np.power(
            np.divide(
                sum_fourth[name],
                lambda_counts,
                out=np.full(n_bins, np.nan),
                where=lambda_counts > 0,
            ),
            0.25,
        )

    percentile_samples: dict[str, np.ndarray] = {name: np.empty(0) for name in FIELDS}
    percentile_keys: dict[str, np.ndarray] = {name: np.empty(0) for name in FIELDS}
    rngs = {
        name: np.random.default_rng(
            None if settings["random_seed"] is None else int(settings["random_seed"]) + index + 1
        )
        for index, name in enumerate(FIELDS)
    }
    if settings["x_max"] == "auto_percentile":
        reservoir_limit = min(settings["max_samples"], 100_000)
        for left, right in _chunks(n_points, settings):
            values = _pair_values(
                left, right, flat_position, flat_plus, flat_minus, flat_B, global_B, settings
            )
            indices, valid_bin = _bin_indices(values["lambda"], lambda_edges)
            valid_bin &= values["valid_direction"]
            for name in FIELDS:
                valid = (
                    valid_bin
                    & (values[name] > 0.0)
                    & np.isfinite(typical[name][np.clip(indices, 0, n_bins - 1)])
                )
                normalized = values[name][valid] / typical[name][indices[valid]]
                percentile_samples[name], percentile_keys[name] = _priority_sample_update(
                    percentile_samples[name],
                    percentile_keys[name],
                    normalized,
                    rngs[name],
                    reservoir_limit,
                )

    pdf_edges: dict[str, np.ndarray] = {}
    for name in FIELDS:
        if settings["x_max"] == "auto_percentile":
            if percentile_samples[name].size:
                x_max = float(np.percentile(percentile_samples[name], settings["x_max_percentile"]))
            else:
                warnings.warn(
                    f"No positive {name} increments for percentile estimation; using x_max=10",
                    RuntimeWarning,
                    stacklevel=2,
                )
                x_max = 10.0
        else:
            x_max = float(settings["x_max"])
        if not np.isfinite(x_max) or x_max <= settings["x_min"]:
            x_max = settings["x_min"] + 10.0
        pdf_edges[name] = np.linspace(settings["x_min"], x_max, settings["n_pdf_bins"] + 1)

    histogram_counts = {
        name: np.zeros((n_bins, settings["n_pdf_bins"]), dtype=np.int64) for name in FIELDS
    }
    sample_lists: dict[str, list[np.ndarray]] = {
        "lambda": [],
        "increment_z_plus": [],
        "increment_z_minus": [],
        "normalized_increment_z_plus": [],
        "normalized_increment_z_minus": [],
        "sin_theta": [],
    }
    saved_count = 0
    for left, right in _chunks(n_points, settings):
        values = _pair_values(
            left, right, flat_position, flat_plus, flat_minus, flat_B, global_B, settings
        )
        indices, valid_bin = _bin_indices(values["lambda"], lambda_edges)
        valid_bin &= values["valid_direction"]
        normalized_by_field = {}
        for name in FIELDS:
            normalized = np.full(values[name].shape, np.nan)
            normalizable = valid_bin & (values[name] > 0.0)
            normalized[normalizable] = (
                values[name][normalizable] / typical[name][indices[normalizable]]
            )
            normalized_by_field[name] = normalized
            for index in range(n_bins):
                selected = normalized[valid_bin & (indices == index)]
                histogram_counts[name][index] += np.histogram(
                    selected[np.isfinite(selected)], bins=pdf_edges[name]
                )[0]
        if settings["save_samples"] and saved_count < settings["max_samples"]:
            available = np.flatnonzero(valid_bin)
            take = available[: settings["max_samples"] - saved_count]
            if take.size:
                sample_lists["lambda"].append(values["lambda"][take])
                sample_lists["increment_z_plus"].append(values["z_plus"][take])
                sample_lists["increment_z_minus"].append(values["z_minus"][take])
                sample_lists["normalized_increment_z_plus"].append(
                    normalized_by_field["z_plus"][take]
                )
                sample_lists["normalized_increment_z_minus"].append(
                    normalized_by_field["z_minus"][take]
                )
                sample_lists["sin_theta"].append(values["sin_theta"][take])
                saved_count += take.size

    pdf_results = {}
    for name in FIELDS:
        widths = np.diff(pdf_edges[name])
        totals = np.sum(histogram_counts[name], axis=1)
        density = np.divide(
            histogram_counts[name],
            totals[:, None] * widths[None, :],
            out=np.full(histogram_counts[name].shape, np.nan, dtype=float),
            where=totals[:, None] > 0,
        )
        pdf_results[name] = {
            "bin_edges": pdf_edges[name],
            "bin_centers": 0.5 * (pdf_edges[name][:-1] + pdf_edges[name][1:]),
            "density": density,
            "counts": histogram_counts[name],
            "typical_increment": typical[name],
            "rms_increment": rms[name],
            "fourth_root_increment": fourth_root[name],
            "positive_increment_counts": positive_counts[name],
            "zero_increment_counts": zero_counts[name],
        }
    samples = {
        name: np.concatenate(parts) if parts else np.empty(0)
        for name, parts in sample_lists.items()
    }
    alignment_mean = np.divide(
        alignment_sum,
        alignment_counts,
        out=np.full(n_bins, np.nan),
        where=alignment_counts > 0,
    )
    valid_bin = lambda_counts >= settings["min_samples_per_bin"]
    return {
        "n_time": n_time,
        "n_spacecraft": n_spacecraft,
        "n_points": n_points,
        "n_pairs_possible": possible_pair_count(n_points),
        "n_pairs_processed": pair_count,
        "n_pairs_skipped": pair_count - int(np.sum(lambda_counts)),
        "n_pairs_skipped_direction": skipped_direction,
        "n_pairs_skipped_nonpositive_lambda": skipped_nonpositive,
        "n_pairs_outside_scale_bins": outside_scale_bins,
        "lambda": {
            "bin_edges": lambda_edges,
            "bin_centers": lambda_centers,
            "counts": lambda_counts,
            "valid_bin": valid_bin,
        },
        "pdf": pdf_results,
        "alignment": {
            "mean_sin_theta": alignment_mean,
            "counts": alignment_counts,
        },
        "samples": samples,
    }


def compute_elsasser_increment_pdf_static(
    input_path: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Compute configured truth/observed static Elsasser increment PDFs."""

    settings = _settings(config)
    results = {}
    for dataset in settings["datasets"]:
        elsasser_config = dict(settings["elsasser"])
        elsasser_config.update(
            {"data_group": dataset["data_group"], "mc_index": dataset["mc_index"]}
        )
        elsasser_config["plot"] = {"enabled": False}
        derived = compute_elsasser_analysis(input_path, elsasser_config)
        positions = _read_positions(input_path, derived["time"])
        pair_result = compute_increment_pdf_for_points(
            positions,
            derived["z_plus"],
            derived["z_minus"],
            derived["B"],
            settings,
        )
        pair_result["selection"] = dataset
        pair_result["elsasser_metadata"] = derived["metadata"]
        results[dataset["name"]] = pair_result
    return {
        "metadata": {
            "analysis_method": "elsasser_increment_pdf_static",
            "sampling_assumption": "static_all_point_pairs",
            "input_path": str(Path(input_path)),
            "elsasser_convention": "z_plus = delta_v + delta_b; z_minus = delta_v - delta_b",
            "increment_component": settings["increment_component"],
            "scale_separation": settings["separation"],
            "mean_field_method": settings["mean_field_method"],
            "pair_sampling_mode": settings["pair_mode"],
            "chunk_size": settings["chunk_size"],
            "lambda_binning": settings["scale_binning"],
            "pdf_binning": "linear",
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
        "settings": settings,
        "results": results,
    }


def _field_label(field: str) -> str:
    return r"z^+" if field == "z_plus" else r"z^-"


def plot_increment_pdf(
    result: dict[str, Any],
    dataset_name: str,
    field: str,
    output_path: str | Path,
    *,
    xscale: str = "linear",
    yscale: str = "log",
    dpi: int = 150,
) -> Path:
    """Plot one normalized-increment PDF curve per sufficiently sampled scale."""

    import matplotlib.pyplot as plt

    dataset = result["results"][dataset_name]
    pdf = dataset["pdf"][field]
    valid = dataset["lambda"]["valid_bin"]
    centers = dataset["lambda"]["bin_centers"]
    figure, axis = plt.subplots(figsize=(8.5, 6.0), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(centers)))
    plotted_count = 0
    for index, center in enumerate(centers):
        density = pdf["density"][index]
        selected = np.isfinite(density) & (density > 0.0)
        if not valid[index] or not np.any(selected):
            continue
        axis.plot(
            pdf["bin_centers"][selected],
            density[selected],
            color=colors[index],
            linewidth=1.3,
            label=f"{center:.3g} km",
        )
        plotted_count += 1
    axis.set_xscale(xscale)
    axis.set_yscale(yscale)
    component = result["metadata"]["increment_component"]
    suffix = r"_\perp" if component == "perpendicular" else ""
    label = _field_label(field)
    axis.set_xlabel(
        rf"$\delta {label}{suffix} / \exp(\langle\ln \delta {label}{suffix}\mid\lambda\rangle)$"
    )
    axis.set_ylabel(rf"$P(\delta {label}{suffix}\mid\lambda)$")
    selection = dataset["selection"]
    title_dataset = (
        "truth" if selection["data_group"] == "truth" else f"observed MC {selection['mc_index']}"
    )
    axis.set_title(f"Elsasser increment PDF, {title_dataset}, {field}")
    axis.grid(True, which="both", alpha=0.22)
    if 0 < plotted_count <= 12:
        axis.legend(title=r"$\lambda$", fontsize=8, title_fontsize=8)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return output


def _write_metadata(group: h5py.Group, metadata: dict[str, Any]) -> None:
    for key, value in metadata.items():
        group.attrs[key] = (
            value
            if isinstance(value, (str, int, float, bool, np.number))
            else json.dumps(value, sort_keys=True)
        )


def _write_array(group: h5py.Group, name: str, values: np.ndarray) -> None:
    group.create_dataset(name, data=values, compression="gzip", compression_opts=4, chunks=True)


@register_analysis
class ElsasserIncrementPDFStaticAnalysis(AnalysisMethod):
    """Registry adapter for static all-spacetime-point increment PDFs."""

    name = "elsasser_increment_pdf_static"
    config_section = "elsasser_increment_pdf_static"

    def run(self, input_path: str | Path) -> dict[str, Any]:
        return compute_elsasser_increment_pdf_static(input_path, self.config)

    def write_result(self, result: dict[str, Any], output_path: str | Path) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        settings = result["settings"]
        plot_directory = Path(settings["plot"]["directory"] or output.parent)
        plot_paths = {}
        if settings["plot"]["enabled"]:
            for name in result["results"]:
                compact_field = settings["plotted_field"].replace("_", "")
                plot_paths[name] = plot_directory / f"{output.stem}_{name}_{compact_field}.png"
        result["metadata"]["plot_enabled"] = settings["plot"]["enabled"]
        result["metadata"]["plot_paths"] = {name: str(path) for name, path in plot_paths.items()}
        with h5py.File(output, "w") as handle:
            _write_metadata(handle.create_group("metadata"), result["metadata"])
            root = handle.create_group("results")
            for name, dataset in result["results"].items():
                group = root.create_group(name)
                selection = dataset["selection"]
                dataset_metadata = {
                    "data_group": selection["data_group"],
                    "mc_index": selection["mc_index"],
                    "density_normalization_method": dataset["elsasser_metadata"][
                        "density_normalization_method"
                    ],
                    "mean_subtraction_method": dataset["elsasser_metadata"]["mean_subtraction"],
                    "n_time": dataset["n_time"],
                    "n_spacecraft": dataset["n_spacecraft"],
                    "n_points": dataset["n_points"],
                    "n_pairs_possible": dataset["n_pairs_possible"],
                    "n_pairs_processed": dataset["n_pairs_processed"],
                    "n_pairs_skipped": dataset["n_pairs_skipped"],
                    "pair_sampling_mode": settings["pair_mode"],
                    "chunk_size": settings["chunk_size"],
                    "increment_component": settings["increment_component"],
                    "scale_separation": settings["separation"],
                    "mean_field_method": settings["mean_field_method"],
                    "lambda_binning": settings["scale_binning"],
                    "pdf_binning": "linear",
                }
                _write_metadata(group, dataset_metadata)
                lambda_group = group.create_group("lambda")
                for key, values in dataset["lambda"].items():
                    _write_array(lambda_group, key, values)
                pdf_root = group.create_group("pdf")
                for field, products in dataset["pdf"].items():
                    field_group = pdf_root.create_group(field)
                    for key, values in products.items():
                        _write_array(field_group, key, values)
                if settings["alignment_compute"]:
                    alignment = group.create_group("alignment")
                    for key, values in dataset["alignment"].items():
                        _write_array(alignment, key, values)
                if settings["save_samples"]:
                    samples = group.create_group("samples")
                    for key, values in dataset["samples"].items():
                        _write_array(samples, key, values)
        for name, path in plot_paths.items():
            plot_increment_pdf(
                result,
                name,
                settings["plotted_field"],
                path,
                xscale=settings["plot"]["xscale"],
                yscale=settings["plot"]["yscale"],
                dpi=settings["plot"]["dpi"],
            )
        return output
