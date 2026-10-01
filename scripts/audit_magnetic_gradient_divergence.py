#!/usr/bin/env python3
"""Reproduce finite-scale divergence diagnostics for a saved gradient product.

This is intentionally separate from the production analysis.  It reads saved
products and the source magnetic grid, but does not alter or constrain the
multi-spacecraft estimator.

Requires SciPy: install with ``python3 -m pip install -e ".[analysis]"``.
For BP input, also include the ``simulation-bp`` extra.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import yaml
from scipy.spatial import Delaunay

from hs_observatory.analysis.magnetic_gradient import (
    CONFIGURATION_ORDER,
    estimate_gradient_least_squares,
    geometry_diagnostics,
    gradient_curl_divergence,
)
from hs_observatory.io.simulation_bp import _read_selected_components
from hs_observatory.io.simulation_npy import PeriodicTrilinearInterpolator

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load_generation(observation_path: Path) -> tuple[dict[str, Any], np.ndarray]:
    with h5py.File(observation_path, "r") as handle:
        raw = handle["metadata"].attrs["config_yaml"]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        generation = yaml.safe_load(str(raw))
        initial_hub = np.asarray(handle["spacecraft/r/B"][0, 0], dtype=float)
    return generation, initial_hub


def _load_magnetic_grid(
    generation: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    simulation = generation["field"]["simulation"]
    if str(simulation["format"]).lower() != "bp":
        raise ValueError("this audit script currently expects a BP simulation source")
    layout = simulation["layout"]
    indices = tuple(int(value) for value in layout["B_components"])
    source = Path(simulation["sources"]["field"]).expanduser()
    components, source_shape, _variables = _read_selected_components(
        source, str(layout["field_variable"]), indices
    )
    scale = float(simulation["scale_factors"]["B"])
    magnetic = np.stack([components[index] for index in indices], axis=-1) * scale
    coordinates = simulation["coordinates"]
    units = str(coordinates["units"]).lower()
    if units == "rho_p":
        box = np.asarray(coordinates["box_size_rho_p"], dtype=float) * float(
            coordinates["rho_p_km"]
        )
    elif units == "km":
        box = np.asarray(coordinates["box_size_km"], dtype=float)
    else:
        box = np.asarray(coordinates["box_size_code"], dtype=float) * float(
            coordinates["code_length_to_km"]
        )
    metadata = {
        "source": str(source.resolve()),
        "source_shape": list(source_shape),
        "B_components": list(indices),
        "B_scale_nT_per_code": scale,
        "grid_shape": list(magnetic.shape[:3]),
        "box_size_km": box.tolist(),
        "spacing_km": (box / np.asarray(magnetic.shape[:3])).tolist(),
        "axis_order": "array axes (0,1,2) = configured simulation (x,y,z)",
        "component_order": "trailing B components = configured (Bx,By,Bz)",
    }
    return magnetic, box, metadata


def _gradient_summary(gradient: np.ndarray) -> dict[str, Any]:
    curl, magnitude, divergence = gradient_curl_divergence(gradient)
    return {
        "gradient": np.asarray(gradient).tolist(),
        "divergence": float(divergence),
        "curl": np.asarray(curl).tolist(),
        "curl_magnitude": float(magnitude),
        "absolute_divergence_to_curl": (
            float(abs(divergence) / magnitude) if magnitude > 0.0 else None
        ),
    }


def _coordinate_sample_check(
    observation_path: Path,
    magnetic: np.ndarray,
    box: np.ndarray,
    initial_hub: np.ndarray,
    *,
    requested_time: float,
) -> dict[str, Any]:
    with h5py.File(observation_path, "r") as handle:
        time = np.asarray(handle["time/B"], dtype=float)
        index = int(np.argmin(np.abs(time - requested_time)))
        positions = np.asarray(handle["spacecraft/r/B"][:, index], dtype=float)
        saved_field = np.asarray(handle["truth/B"][index], dtype=float)
        source = handle["metadata"].attrs.get("spacecraft_source", "{}")
        if isinstance(source, bytes):
            source = source.decode("utf-8")
        source_metadata = json.loads(str(source))
    mapped = np.mod(positions - initial_hub + 0.5 * box, box)
    reproduced = PeriodicTrilinearInterpolator(box).sample(magnetic, mapped)
    difference = reproduced - saved_field
    return {
        "time_s": float(time[index]),
        "spacecraft_position_input_units": source_metadata.get("position_units", "not recorded"),
        "spacecraft_position_shape": list(positions.shape),
        "spacecraft_position_min_km": np.min(positions, axis=0).tolist(),
        "spacecraft_position_max_km": np.max(positions, axis=0).tolist(),
        "mapped_simulation_position_min_km": np.min(mapped, axis=0).tolist(),
        "mapped_simulation_position_max_km": np.max(mapped, axis=0).tolist(),
        "mapping": (
            "componentwise translation plus periodic wrap; identity axis map; "
            "Jacobian determinant +1"
        ),
        "saved_B_units": "nT (generation scale factor applied before HDF5 write)",
        "maximum_reproduction_error_nT": float(np.max(np.abs(difference))),
        "rms_reproduction_error_nT": float(np.sqrt(np.mean(difference**2))),
    }


def _dense_affine_fit(points: np.ndarray, field: np.ndarray) -> tuple[np.ndarray, float]:
    estimate = estimate_gradient_least_squares(points, field)
    return estimate.gradient, estimate.rms_residual


def _sphere_offsets(count: int, radius: float, rng: np.random.Generator) -> np.ndarray:
    directions = rng.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = radius * rng.random(count) ** (1.0 / 3.0)
    return directions * radii[:, None]


def _dense_sphere_series(
    interpolator: PeriodicTrilinearInterpolator,
    magnetic: np.ndarray,
    centers: np.ndarray,
    radius: float,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = _sphere_offsets(count, radius, rng)
    points = centers[:, None, :] + offsets[None, :, :]
    sampled = interpolator.sample(magnetic, points)
    centered_points = offsets - np.mean(offsets, axis=0)
    centered_field = sampled - np.mean(sampled, axis=1, keepdims=True)
    inverse = np.linalg.inv(centered_points.T @ centered_points)
    gradient = np.einsum("tpi,pj,jk->tik", centered_field, centered_points, inverse)
    fitted = np.mean(sampled, axis=1, keepdims=True) + np.einsum(
        "pj,tij->tpi", centered_points, gradient
    )
    residual = np.sqrt(np.mean(np.sum((sampled - fitted) ** 2, axis=-1), axis=1))
    return gradient, residual


def _sample_convex_hull(vertices: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    triangulation = Delaunay(vertices)
    tetrahedra = vertices[triangulation.simplices]
    determinants = np.linalg.det(
        np.stack(
            (
                tetrahedra[:, 1] - tetrahedra[:, 0],
                tetrahedra[:, 2] - tetrahedra[:, 0],
                tetrahedra[:, 3] - tetrahedra[:, 0],
            ),
            axis=-1,
        )
    )
    weights = np.abs(determinants)
    weights /= np.sum(weights)
    chosen = rng.choice(len(tetrahedra), size=count, p=weights)
    barycentric = rng.exponential(size=(count, 4))
    barycentric /= np.sum(barycentric, axis=1, keepdims=True)
    return np.einsum("ni,nij->nj", barycentric, tetrahedra[chosen])


def _grid_derivative_statistics(
    magnetic: np.ndarray,
    box: np.ndarray,
) -> dict[str, Any]:
    spacing = box / np.asarray(magnetic.shape[:3], dtype=float)
    divergence = np.zeros(magnetic.shape[:3], dtype=float)
    curl = np.zeros((*magnetic.shape[:3], 3), dtype=float)
    component_rms = np.empty((3, 3), dtype=float)
    derivatives: dict[tuple[int, int], np.ndarray] = {}
    for field_component in range(3):
        for derivative_axis in range(3):
            derivative = (
                np.roll(magnetic[..., field_component], -1, axis=derivative_axis)
                - np.roll(magnetic[..., field_component], 1, axis=derivative_axis)
            ) / (2.0 * spacing[derivative_axis])
            component_rms[field_component, derivative_axis] = np.sqrt(np.mean(derivative**2))
            if field_component == derivative_axis:
                divergence += derivative
            if (field_component, derivative_axis) in {
                (2, 1),
                (1, 2),
                (0, 2),
                (2, 0),
                (1, 0),
                (0, 1),
            }:
                derivatives[field_component, derivative_axis] = derivative
    curl[..., 0] = derivatives.pop((2, 1)) - derivatives.pop((1, 2))
    curl[..., 1] = derivatives.pop((0, 2)) - derivatives.pop((2, 0))
    curl[..., 2] = derivatives.pop((1, 0)) - derivatives.pop((0, 1))
    magnitude = np.linalg.norm(curl, axis=-1)
    rms_divergence = float(np.sqrt(np.mean(divergence**2)))
    rms_curl = float(np.sqrt(np.mean(magnitude**2)))
    return {
        "component_derivative_rms_nT_per_km": component_rms.tolist(),
        "rms_divergence_nT_per_km": rms_divergence,
        "rms_curl_magnitude_nT_per_km": rms_curl,
        "rms_divergence_to_curl": rms_divergence / rms_curl,
        "rms_curl_to_divergence": rms_curl / rms_divergence,
        "absolute_divergence_percentiles": np.percentile(
            np.abs(divergence), (50, 90, 95, 99, 99.9)
        ).tolist(),
        "curl_magnitude_percentiles": np.percentile(magnitude, (50, 90, 95, 99, 99.9)).tolist(),
    }


def _actual_scale_test(
    product_path: Path,
    magnetic: np.ndarray,
    box: np.ndarray,
    *,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    interpolator = PeriodicTrilinearInterpolator(box)
    spacing = box / np.asarray(magnetic.shape[:3])
    reference_spacing = float(np.cbrt(np.prod(spacing)))
    records: list[dict[str, Any]] = []
    with h5py.File(product_path, "r") as handle:
        for category in ("small", "best_multi", "all_nine"):
            positions = np.asarray(
                handle[f"configurations/{category}/true_estimate/positions"][0],
                dtype=float,
            )
            positions -= np.mean(positions, axis=0)
            positions /= geometry_diagnostics(positions).characteristic_size
            for multiple in (0.5, 1, 2, 4, 8, 16):
                scale = multiple * reference_spacing
                divergences = []
                curls = []
                conditions = []
                for _ in range(12):
                    center = rng.uniform(0.0, box)
                    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
                    if np.linalg.det(rotation) < 0.0:
                        rotation[:, 0] *= -1.0
                    points = center + scale * (positions @ rotation.T)
                    estimate = estimate_gradient_least_squares(
                        points, interpolator.sample(magnetic, points)
                    )
                    _curl, curl_magnitude, divergence = gradient_curl_divergence(estimate.gradient)
                    divergences.append(float(divergence))
                    curls.append(float(curl_magnitude))
                    conditions.append(float(estimate.condition_number))
                divergences_array = np.asarray(divergences)
                curls_array = np.asarray(curls)
                records.append(
                    {
                        "configuration": category,
                        "scale_multiple_of_geometric_mean_spacing": multiple,
                        "scale_km": scale,
                        "rms_absolute_divergence_nT_per_km": float(
                            np.sqrt(np.mean(divergences_array**2))
                        ),
                        "rms_curl_magnitude_nT_per_km": float(np.sqrt(np.mean(curls_array**2))),
                        "median_absolute_divergence_to_curl": float(
                            np.median(
                                np.abs(divergences_array)
                                / np.maximum(curls_array, np.finfo(float).tiny)
                            )
                        ),
                        "median_condition_number": float(np.median(conditions)),
                        "reference_spacing_km": reference_spacing,
                    }
                )
    return records


def _analytic_interpolation_test(
    product_path: Path,
    *,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    grid_count = 64
    box = np.full(3, float(grid_count))
    coordinates = np.arange(grid_count, dtype=float)
    xx, yy, zz = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    interpolator = PeriodicTrilinearInterpolator(box)
    shapes: dict[str, np.ndarray] = {}
    with h5py.File(product_path, "r") as handle:
        for category in ("small", "best_multi", "all_nine"):
            points = np.asarray(
                handle[f"configurations/{category}/true_estimate/positions"][0],
                dtype=float,
            )
            points -= np.mean(points, axis=0)
            shapes[category] = points / geometry_diagnostics(points).characteristic_size

    records: list[dict[str, Any]] = []
    for wavelength in (8.0, 16.0, 32.0):
        wave_number = 2.0 * np.pi / wavelength
        grid = np.stack(
            (
                np.sin(wave_number * zz),
                np.sin(wave_number * xx),
                np.sin(wave_number * yy),
            ),
            axis=-1,
        )

        def direct(points: np.ndarray) -> np.ndarray:
            return np.stack(
                (
                    np.sin(wave_number * points[:, 2]),
                    np.sin(wave_number * points[:, 0]),
                    np.sin(wave_number * points[:, 1]),
                ),
                axis=-1,
            )

        for category, normalized_points in shapes.items():
            for scale in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
                direct_divergence = []
                interpolated_divergence = []
                direct_ratio = []
                interpolated_ratio = []
                for _ in range(24):
                    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
                    if np.linalg.det(rotation) < 0.0:
                        rotation[:, 0] *= -1.0
                    center = rng.uniform(0.0, box)
                    points = center + scale * (normalized_points @ rotation.T)
                    direct_gradient = estimate_gradient_least_squares(
                        points, direct(points)
                    ).gradient
                    interpolated_gradient = estimate_gradient_least_squares(
                        points, interpolator.sample(grid, points)
                    ).gradient
                    for gradient, divergences, ratios in (
                        (direct_gradient, direct_divergence, direct_ratio),
                        (
                            interpolated_gradient,
                            interpolated_divergence,
                            interpolated_ratio,
                        ),
                    ):
                        _curl, magnitude, divergence = gradient_curl_divergence(gradient)
                        divergences.append(float(divergence))
                        ratios.append(
                            float(abs(divergence) / magnitude)
                            if magnitude > np.finfo(float).tiny
                            else np.nan
                        )
                direct_array = np.asarray(direct_divergence)
                interpolated_array = np.asarray(interpolated_divergence)
                records.append(
                    {
                        "configuration": category,
                        "wavelength_grid_spacings": wavelength,
                        "configuration_scale_grid_spacings": scale,
                        "direct_rms_divergence": float(np.sqrt(np.mean(direct_array**2))),
                        "interpolated_rms_divergence": float(
                            np.sqrt(np.mean(interpolated_array**2))
                        ),
                        "interpolation_contribution_rms": float(
                            np.sqrt(np.mean((interpolated_array - direct_array) ** 2))
                        ),
                        "direct_median_absolute_divergence_to_curl": float(
                            np.nanmedian(direct_ratio)
                        ),
                        "interpolated_median_absolute_divergence_to_curl": float(
                            np.nanmedian(interpolated_ratio)
                        ),
                        "analytic_field": (
                            "B=(sin(kz),sin(kx),sin(ky)); divergence identically zero"
                        ),
                    }
                )
    return records


def _representative_indices(
    time: np.ndarray, ratio: np.ndarray, start: float, stop: float
) -> list[tuple[str, int]]:
    candidates = np.flatnonzero((time >= start) & (time <= stop) & np.isfinite(ratio))
    if not len(candidates):
        return []
    local = ratio[candidates]
    choices = (
        ("small_ratio", int(candidates[np.argmin(local)])),
        ("near_unity", int(candidates[np.argmin(np.abs(local - 1.0))])),
        ("largest_peak", int(candidates[np.argmax(local)])),
    )
    return list(dict.fromkeys(choices))


def _interval_diagnostics(
    product_path: Path,
    magnetic: np.ndarray,
    box: np.ndarray,
    initial_hub: np.ndarray,
    *,
    start: float,
    stop: float,
    dense_samples: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    interpolator = PeriodicTrilinearInterpolator(box)
    table: list[dict[str, Any]] = []
    hull_point_sets: list[np.ndarray] = []
    series: dict[str, dict[str, np.ndarray]] = {}
    with h5py.File(product_path, "r") as handle:
        time = np.asarray(handle["time/analysis_time"], dtype=float)
        interval = (time >= start) & (time <= stop)
        interval_indices = np.flatnonzero(interval)
        for category in CONFIGURATION_ORDER:
            path = f"configurations/{category}"
            if path not in handle:
                continue
            group = handle[path]
            truth = group["truth"]
            estimate = group["true_estimate"]
            positions = np.asarray(estimate["positions"], dtype=float)
            centers = np.mod(
                np.mean(positions, axis=1) - initial_hub + 0.5 * box,
                box,
            )
            size = np.asarray(estimate["characteristic_size"], dtype=float)
            dense_gradient, dense_residual = _dense_sphere_series(
                interpolator,
                magnetic,
                centers[interval],
                0.5 * float(np.median(size)),
                dense_samples,
                rng,
            )
            ideal_gradient = np.asarray(estimate["gradient"], dtype=float)[interval]
            coarse_gradient = np.asarray(truth["coarse_gradient"], dtype=float)[interval]
            ideal_curl, ideal_curl_magnitude, ideal_divergence = gradient_curl_divergence(
                ideal_gradient
            )
            coarse_curl, coarse_curl_magnitude, coarse_divergence = gradient_curl_divergence(
                coarse_gradient
            )
            dense_curl, dense_curl_magnitude, dense_divergence = gradient_curl_divergence(
                dense_gradient
            )
            ratio = np.asarray(estimate["divergence_curl_ratio"], dtype=float)
            series[category] = {
                "time": time[interval],
                "ideal_curl_magnitude": ideal_curl_magnitude,
                "coarse_curl_magnitude": coarse_curl_magnitude,
                "ideal_divergence": ideal_divergence,
                "coarse_divergence": coarse_divergence,
                "ideal_ratio": np.abs(ideal_divergence)
                / np.maximum(ideal_curl_magnitude, np.finfo(float).tiny),
                "coarse_ratio": np.abs(coarse_divergence)
                / np.maximum(coarse_curl_magnitude, np.finfo(float).tiny),
                "divergence_to_gradient_norm": np.abs(ideal_divergence)
                / np.maximum(
                    np.linalg.norm(ideal_gradient, axis=(-2, -1)),
                    np.finfo(float).tiny,
                ),
                "curl_error": ideal_curl - coarse_curl,
                "condition_number": np.asarray(estimate["condition_number"], dtype=float)[interval],
                "dense_residual": dense_residual,
                "coarse_cell_count": np.asarray(truth["coarse_cell_count"], dtype=float)[interval],
                "dense_divergence": dense_divergence,
                "dense_curl_magnitude": dense_curl_magnitude,
            }
            for label, index in _representative_indices(time, ratio, start, stop):
                local_index = int(np.searchsorted(interval_indices, index))
                vertices = positions[index] - np.mean(positions[index], axis=0)
                hull_offsets = _sample_convex_hull(vertices, dense_samples, rng)
                hull_points = centers[index] + hull_offsets
                hull_gradient, hull_residual = _dense_affine_fit(
                    hull_points,
                    interpolator.sample(magnetic, hull_points),
                )
                geometry_record = geometry_diagnostics(positions[index])
                record: dict[str, Any] = {
                    "configuration": category,
                    "selection": label,
                    "time_s": float(time[index]),
                    "barycenter_km": np.mean(positions[index], axis=0).tolist(),
                    "mapped_barycenter_km": centers[index].tolist(),
                    "L_km": geometry_record.characteristic_size,
                    "E": geometry_record.elongation,
                    "P": geometry_record.planarity,
                    "Q": geometry_record.chi,
                    "volumetric_tensor_eigenvalues_km2": geometry_record.eigenvalues.tolist(),
                    "condition_number": geometry_record.condition_number,
                    "affine_residual_nT": float(np.asarray(estimate["affine_rms_residual"])[index]),
                    "dense_sphere_affine_residual_nT": float(dense_residual[local_index]),
                    "dense_hull_affine_residual_nT": hull_residual,
                    "coarse_cell_count": int(np.asarray(truth["coarse_cell_count"])[index]),
                    "ideal": _gradient_summary(np.asarray(estimate["gradient"])[index]),
                    "point_truth": _gradient_summary(np.asarray(truth["point_gradient"])[index]),
                    "coarse_truth": _gradient_summary(np.asarray(truth["coarse_gradient"])[index]),
                    "dense_sphere_fit": _gradient_summary(dense_gradient[local_index]),
                    "dense_convex_hull_fit": _gradient_summary(hull_gradient),
                }
                table.append(record)
                hull_point_sets.append(hull_points)
    if hull_point_sets:
        combined_points = np.concatenate(hull_point_sets, axis=0)
        hull_average_gradients = np.empty((len(hull_point_sets), 3, 3))
        spacing = box / np.asarray(magnetic.shape[:3], dtype=float)
        for field_component in range(3):
            for derivative_axis in range(3):
                derivative = (
                    np.roll(magnetic[..., field_component], -1, axis=derivative_axis)
                    - np.roll(magnetic[..., field_component], 1, axis=derivative_axis)
                ) / (2.0 * spacing[derivative_axis])
                sampled = interpolator.sample(derivative, combined_points)
                start_index = 0
                for record_index, points in enumerate(hull_point_sets):
                    stop_index = start_index + len(points)
                    hull_average_gradients[record_index, field_component, derivative_axis] = (
                        np.mean(sampled[start_index:stop_index])
                    )
                    start_index = stop_index
        for record, gradient in zip(table, hull_average_gradients):
            record["convex_hull_average_gradient"] = _gradient_summary(gradient)
    return table, series


def _plot_debug(series: dict[str, dict[str, np.ndarray]], output: Path, dpi: int) -> None:
    categories = [name for name in CONFIGURATION_ORDER if name in series]
    figure, axes = plt.subplots(
        8,
        len(categories),
        figsize=(4.2 * len(categories), 20.0),
        sharex=True,
        squeeze=False,
    )
    component_colors = ("#d1495b", "#2a9d60", "#3777bb")
    for column, category in enumerate(categories):
        data = series[category]
        time = data["time"]
        axes[0, column].plot(time, data["ideal_curl_magnitude"], label="ideal")
        axes[0, column].plot(time, data["coarse_curl_magnitude"], label="coarse")
        axes[0, column].plot(time, data["dense_curl_magnitude"], label="dense sphere", alpha=0.8)
        axes[1, column].plot(time, data["ideal_divergence"], label="ideal")
        axes[1, column].plot(time, data["coarse_divergence"], label="coarse")
        axes[1, column].plot(time, data["dense_divergence"], label="dense sphere", alpha=0.8)
        axes[1, column].axhline(0.0, color="black", linewidth=0.7)
        axes[2, column].plot(time, data["ideal_ratio"], label="ideal")
        axes[2, column].plot(time, data["coarse_ratio"], label="coarse")
        axes[3, column].plot(time, data["divergence_to_gradient_norm"])
        for component, color in enumerate(component_colors):
            axes[4, column].plot(
                time,
                data["curl_error"][:, component],
                color=color,
                label=("x", "y", "z")[component],
            )
        axes[5, column].plot(time, data["condition_number"])
        axes[6, column].plot(time, data["dense_residual"])
        axes[7, column].plot(time, data["coarse_cell_count"])
        median_truth = np.nanmedian(data["coarse_curl_magnitude"])
        for row in range(8):
            axis = axes[row, column]
            for fraction, color in ((0.10, "#f6bd60"), (0.05, "#f28482"), (0.01, "#9d4edd")):
                mask = data["coarse_curl_magnitude"] < fraction * median_truth
                axis.fill_between(
                    time,
                    0.0,
                    1.0,
                    where=mask,
                    transform=axis.get_xaxis_transform(),
                    color=color,
                    alpha=0.08,
                    linewidth=0,
                )
            axis.grid(alpha=0.2)
        axes[0, column].set_title(category.replace("_", " "))
        axes[-1, column].set_xlabel("Time (s)")
    labels = (
        r"$|\nabla\times B|$ (nT/km)",
        r"$\nabla\cdot B$ (nT/km)",
        r"$|\nabla\cdot B|/|\nabla\times B|$",
        r"$|\nabla\cdot B|/\|\nabla B\|_F$",
        r"$\Delta(\nabla\times B)$ (nT/km)",
        "Condition number",
        "Dense-sphere affine residual (nT)",
        "Cells in coarse sphere",
    )
    for row, label in enumerate(labels):
        axes[row, 0].set_ylabel(label)
    axes[0, 0].legend(frameon=False, fontsize=7)
    axes[1, 0].legend(frameon=False, fontsize=7)
    axes[4, 0].legend(frameon=False, fontsize=7, ncol=3)
    figure.suptitle("Magnetic-gradient divergence audit (debugging only)")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _series_statistics(series: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, float]]:
    statistics: dict[str, dict[str, float]] = {}
    for category, data in series.items():
        ideal_ratio = np.asarray(data["ideal_ratio"], dtype=float)
        statistics[category] = {
            "ideal_ratio_median": float(np.nanmedian(ideal_ratio)),
            "ideal_ratio_p90": float(np.nanpercentile(ideal_ratio, 90.0)),
            "ideal_ratio_p99": float(np.nanpercentile(ideal_ratio, 99.0)),
            "ideal_ratio_maximum": float(np.nanmax(ideal_ratio)),
            "ideal_ratio_fraction_at_least_one": float(np.nanmean(ideal_ratio >= 1.0)),
            "ideal_rms_divergence_nT_per_km": float(
                np.sqrt(np.nanmean(data["ideal_divergence"] ** 2))
            ),
            "coarse_rms_divergence_nT_per_km": float(
                np.sqrt(np.nanmean(data["coarse_divergence"] ** 2))
            ),
            "dense_sphere_rms_divergence_nT_per_km": float(
                np.sqrt(np.nanmean(data["dense_divergence"] ** 2))
            ),
            "ideal_rms_curl_magnitude_nT_per_km": float(
                np.sqrt(np.nanmean(data["ideal_curl_magnitude"] ** 2))
            ),
            "dense_sphere_affine_residual_median_nT": float(np.nanmedian(data["dense_residual"])),
            "condition_number_median": float(np.nanmedian(data["condition_number"])),
        }
    return statistics


def _write_csv(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for record in records for key in record})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, (dict, list))
                        else value
                    )
                    for key, value in record.items()
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--product",
        type=Path,
        default=Path("outputs/sim-prod-400_magnetic_gradients.h5"),
    )
    parser.add_argument(
        "--observation",
        type=Path,
        default=Path("outputs/sim-prod-400.h5"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/magnetic_gradient_divergence_audit"),
    )
    parser.add_argument("--start-time", type=float, default=150.0)
    parser.add_argument("--stop-time", type=float, default=200.0)
    parser.add_argument("--dense-samples", type=int, default=256)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    if args.stop_time <= args.start_time:
        parser.error("--stop-time must be greater than --start-time")
    if args.dense_samples < 32:
        parser.error("--dense-samples must be at least 32")

    rng = np.random.default_rng(20260726)
    generation, initial_hub = _load_generation(args.observation)
    magnetic, box, coordinate_metadata = _load_magnetic_grid(generation)
    coordinate_sample = _coordinate_sample_check(
        args.observation,
        magnetic,
        box,
        initial_hub,
        requested_time=0.5 * (args.start_time + args.stop_time),
    )
    grid_statistics = _grid_derivative_statistics(magnetic, box)
    scale_records = _actual_scale_test(args.product, magnetic, box, rng=rng)
    analytic_records = _analytic_interpolation_test(args.product, rng=rng)
    interval_table, series = _interval_diagnostics(
        args.product,
        magnetic,
        box,
        initial_hub,
        start=args.start_time,
        stop=args.stop_time,
        dense_samples=args.dense_samples,
        rng=rng,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "coordinate_metadata": coordinate_metadata,
        "coordinate_sample": coordinate_sample,
        "grid_derivative_statistics": grid_statistics,
        "interval": [args.start_time, args.stop_time],
        "interval_statistics": _series_statistics(series),
        "dense_samples": args.dense_samples,
        "representative_samples": interval_table,
    }
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(interval_table, args.output_dir / "representative_samples.csv")
    _write_csv(scale_records, args.output_dir / "actual_scale_dependence.csv")
    _write_csv(
        analytic_records,
        args.output_dir / "analytic_interpolation_scale_dependence.csv",
    )
    _plot_debug(series, args.output_dir / "debug_timeseries.png", args.dpi)
    print(args.output_dir / "audit_summary.json")
    print(args.output_dir / "representative_samples.csv")
    print(args.output_dir / "actual_scale_dependence.csv")
    print(args.output_dir / "analytic_interpolation_scale_dependence.csv")
    print(args.output_dir / "debug_timeseries.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
