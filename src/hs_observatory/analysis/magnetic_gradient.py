"""Multi-spacecraft magnetic-gradient analysis.

The least-squares estimator follows Equation 2 of Broeren & Klein (2024),
``Multi-Point Gradient Estimation in Turbulence`` (arXiv:2405.14889).  The
pair sum is over the N(N-1)/2 independent (unordered) spacecraft pairs.

Gradient arrays use ``gradient[field_component, spatial_derivative]``;
for example, ``gradient[0, 1]`` is dBx/dy.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence
import warnings

import h5py
import matplotlib
import numpy as np
import yaml

from hs_observatory import __version__
from hs_observatory.analysis.base import AnalysisMethod
from hs_observatory.analysis.registry import register_analysis
from hs_observatory.fields.simulation_sampled import SimulationSampledField
from hs_observatory.io.simulation_npy import PeriodicTrilinearInterpolator

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from hs_observatory.visualization import SPACECRAFT_COLORS  # noqa: E402


CONFIGURATION_ORDER = ("small", "intermediate", "best_multi", "large", "all_nine")


@dataclass(frozen=True)
class GeometryDiagnostics:
    """Volumetric-tensor diagnostics for one configuration sample."""

    barycenter: np.ndarray
    volumetric_tensor: np.ndarray
    eigenvalues: np.ndarray
    semiaxes: np.ndarray
    characteristic_size: float
    elongation: float
    planarity: float
    chi: float
    condition_number: float
    rank: int


@dataclass(frozen=True)
class GradientEstimate:
    """Complete result of one N-point affine magnetic-field fit."""

    barycenter: np.ndarray
    mean_field: np.ndarray
    volumetric_tensor: np.ndarray
    inverse_volumetric_tensor: np.ndarray
    eigenvalues: np.ndarray
    singular_values: np.ndarray
    gradient: np.ndarray
    fitted_field: np.ndarray
    residuals: np.ndarray
    component_rms_residual: np.ndarray
    rms_residual: float
    condition_number: float
    rank: int
    used_pseudoinverse: bool
    valid: bool


@dataclass(frozen=True)
class TimeAlignment:
    """Samples synchronized from spacecraft-specific clocks to one time grid."""

    field: np.ndarray
    positions: np.ndarray
    valid: np.ndarray
    interpolation_required: np.ndarray
    fractions: np.ndarray
    brackets: np.ndarray


def geometry_diagnostics(positions: np.ndarray) -> GeometryDiagnostics:
    """Compute Broeren-style geometry metrics for ``(spacecraft, xyz)`` points."""

    points = np.asarray(positions, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError("positions must have shape (n_spacecraft, 3)")
    barycenter = np.mean(points, axis=0)
    centered = points - barycenter
    tensor = centered.T @ centered / float(len(points))
    eigenvalues = np.linalg.eigvalsh(tensor)[::-1]
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    semiaxes = np.sqrt(eigenvalues)
    a, b, c = semiaxes
    scale = max(float(a), 1.0)
    tolerance = np.finfo(float).eps * scale * scale * 32.0
    rank = int(np.count_nonzero(eigenvalues > tolerance))
    elongation = 0.0 if a <= 0.0 else float(np.clip(1.0 - b / a, 0.0, 1.0))
    planarity = 0.0 if b <= 0.0 else float(np.clip(1.0 - c / b, 0.0, 1.0))
    condition = float(np.linalg.cond(tensor)) if rank == 3 else float("inf")
    return GeometryDiagnostics(
        barycenter=barycenter,
        volumetric_tensor=tensor,
        eigenvalues=eigenvalues,
        semiaxes=semiaxes,
        characteristic_size=float(2.0 * a),
        elongation=elongation,
        planarity=planarity,
        chi=float(np.hypot(elongation, planarity)),
        condition_number=condition,
        rank=rank,
    )


def geometry_is_static(
    positions: np.ndarray,
    *,
    rtol: float = 1.0e-10,
    atol_km: float = 1.0e-8,
) -> bool:
    """Return whether relative spacecraft geometry is constant through time."""

    points = np.asarray(positions, dtype=float)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("positions must have shape (time, spacecraft, 3)")
    centered = points - np.mean(points, axis=1, keepdims=True)
    return bool(np.allclose(centered, centered[0], rtol=rtol, atol=atol_km))


def estimate_gradient_least_squares(
    positions: np.ndarray,
    field: np.ndarray,
    *,
    rcond: float | None = None,
    condition_number_max: float = 1.0e12,
    allow_pseudoinverse: bool = True,
) -> GradientEstimate:
    """Estimate ``dB_i/dx_j`` with the general N-point least-squares method."""

    points = np.asarray(positions, dtype=float)
    values = np.asarray(field, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise ValueError("positions must have shape (n_spacecraft >= 4, 3)")
    if values.shape != points.shape:
        raise ValueError("field must have the same (n_spacecraft, 3) shape as positions")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(values)):
        return _invalid_estimate(points, values)

    geometry = geometry_diagnostics(points)
    n_spacecraft = len(points)
    left, right = np.triu_indices(n_spacecraft, k=1)
    field_difference = values[left] - values[right]
    position_difference = points[left] - points[right]
    covariance = field_difference.T @ position_difference / float(n_spacecraft**2)
    use_pseudoinverse = (
        geometry.rank < 3
        or not np.isfinite(geometry.condition_number)
        or geometry.condition_number > float(condition_number_max)
    )
    if use_pseudoinverse and not allow_pseudoinverse:
        return _invalid_estimate(points, values, geometry=geometry)

    if use_pseudoinverse:
        effective_rcond = 1.0e-15 if rcond is None else float(rcond)
        inverse = np.linalg.pinv(geometry.volumetric_tensor, rcond=effective_rcond)
        gradient = covariance @ inverse
    else:
        inverse = np.linalg.solve(geometry.volumetric_tensor, np.eye(3))
        gradient = np.linalg.solve(geometry.volumetric_tensor.T, covariance.T).T

    mean_field = np.mean(values, axis=0)
    fitted = mean_field + (points - geometry.barycenter) @ gradient.T
    residuals = values - fitted
    component_rms = np.sqrt(np.mean(residuals**2, axis=0))
    rms = float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))
    finite = bool(
        np.all(np.isfinite(gradient))
        and np.all(np.isfinite(residuals))
        and (geometry.rank == 3 or allow_pseudoinverse)
    )
    return GradientEstimate(
        barycenter=geometry.barycenter,
        mean_field=mean_field,
        volumetric_tensor=geometry.volumetric_tensor,
        inverse_volumetric_tensor=inverse,
        eigenvalues=geometry.eigenvalues,
        singular_values=np.linalg.svd(geometry.volumetric_tensor, compute_uv=False),
        gradient=gradient,
        fitted_field=fitted,
        residuals=residuals,
        component_rms_residual=component_rms,
        rms_residual=rms,
        condition_number=geometry.condition_number,
        rank=geometry.rank,
        used_pseudoinverse=use_pseudoinverse,
        valid=finite,
    )


def gradient_response_coefficients(
    positions: np.ndarray, **settings: Any
) -> np.ndarray:
    """Build reusable linear gradient responses using the existing estimator.

    Returns ``(geometry_time, spacecraft, derivative_axis)`` coefficients;
    ``geometry_time`` is one when relative geometry is static. Contract with
    ``B(time, spacecraft, component)`` using ``tnj,tni->tij``. Invalid geometry
    produces NaNs, following :func:`estimate_gradient_least_squares`.
    """

    points = np.asarray(positions, dtype=float)
    if points.ndim != 3 or points.shape[1] < 4 or points.shape[2] != 3 or not len(points):
        raise ValueError("positions must have shape (time >= 1, spacecraft >= 4, 3)")
    samples = points[:1] if geometry_is_static(points) else points
    response = np.full(samples.shape, np.nan)
    for time_index, sample in enumerate(samples):
        for member in range(sample.shape[0]):
            basis = np.zeros_like(sample)
            basis[member, 0] = 1.0
            estimate = estimate_gradient_least_squares(sample, basis, **settings)
            response[time_index, member] = estimate.gradient[0]
    return response


def apply_gradient_response(response: np.ndarray, field: np.ndarray) -> np.ndarray:
    """Evaluate cached estimator responses on a new magnetic-field series."""

    values = np.asarray(field, dtype=float)
    # Remove the uniform field exactly, as the estimator's pair differences do.
    return np.einsum("tnj,tni->tij", response, values - values[:, :1])


def _invalid_estimate(
    positions: np.ndarray,
    field: np.ndarray,
    *,
    geometry: GeometryDiagnostics | None = None,
) -> GradientEstimate:
    points = np.asarray(positions, dtype=float)
    values = np.asarray(field, dtype=float)
    n_spacecraft = len(points)
    if geometry is None and np.all(np.isfinite(points)):
        geometry = geometry_diagnostics(points)
    nan3 = np.full(3, np.nan)
    nan33 = np.full((3, 3), np.nan)
    barycenter = np.mean(points, axis=0) if points.size else nan3.copy()
    return GradientEstimate(
        barycenter=barycenter if geometry is None else geometry.barycenter,
        mean_field=np.mean(values, axis=0) if values.size else nan3.copy(),
        volumetric_tensor=nan33 if geometry is None else geometry.volumetric_tensor,
        inverse_volumetric_tensor=nan33.copy(),
        eigenvalues=nan3 if geometry is None else geometry.eigenvalues,
        singular_values=nan3.copy(),
        gradient=nan33.copy(),
        fitted_field=np.full((n_spacecraft, 3), np.nan),
        residuals=np.full((n_spacecraft, 3), np.nan),
        component_rms_residual=nan3.copy(),
        rms_residual=float("nan"),
        condition_number=float("inf") if geometry is None else geometry.condition_number,
        rank=0 if geometry is None else geometry.rank,
        used_pseudoinverse=False,
        valid=False,
    )


def gradient_curl_divergence(gradient: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return curl vector, curl magnitude, and divergence from gradient tensors."""

    value = np.asarray(gradient, dtype=float)
    if value.shape[-2:] != (3, 3):
        raise ValueError("gradient must end with shape (3, 3)")
    curl = np.stack(
        (
            value[..., 2, 1] - value[..., 1, 2],
            value[..., 0, 2] - value[..., 2, 0],
            value[..., 1, 0] - value[..., 0, 1],
        ),
        axis=-1,
    )
    curl_magnitude = np.linalg.norm(curl, axis=-1)
    divergence = np.trace(value, axis1=-2, axis2=-1)
    return curl, curl_magnitude, divergence


def divergence_curl_ratio(
    divergence: np.ndarray,
    curl_magnitude: np.ndarray,
    floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return |div B|/|curl B| and a mask; small denominators become NaN."""

    div = np.asarray(divergence, dtype=float)
    curl = np.asarray(curl_magnitude, dtype=float)
    valid = np.isfinite(div) & np.isfinite(curl) & (curl >= float(floor)) & (curl > 0.0)
    ratio = np.full(np.broadcast_shapes(div.shape, curl.shape), np.nan, dtype=float)
    np.divide(np.abs(div), curl, out=ratio, where=valid)
    return ratio, valid


def align_spacecraft_samples(
    common_time: np.ndarray,
    time_offsets: np.ndarray,
    positions: np.ndarray,
    field: np.ndarray,
    *,
    allow_extrapolation: bool = False,
) -> TimeAlignment:
    """Linearly align paired position/field samples to ``common_time``.

    ``positions`` is ``(spacecraft,time,3)`` and ``field`` is
    ``(time,spacecraft,3)``.  Each spacecraft source grid is
    ``common_time + time_offsets[spacecraft]``.
    """

    time = np.asarray(common_time, dtype=float)
    offsets = np.asarray(time_offsets, dtype=float)
    source_positions = np.asarray(positions, dtype=float)
    source_field = np.asarray(field, dtype=float)
    n_time = len(time)
    n_spacecraft = len(offsets)
    if time.ndim != 1 or n_time < 2 or np.any(np.diff(time) <= 0.0):
        raise ValueError("common_time must be strictly increasing with at least two samples")
    if source_positions.shape != (n_spacecraft, n_time, 3):
        raise ValueError("positions must have shape (spacecraft, time, 3)")
    if source_field.shape != (n_time, n_spacecraft, 3):
        raise ValueError("field must have shape (time, spacecraft, 3)")

    aligned_field = np.full_like(source_field, np.nan)
    aligned_positions = np.full_like(source_positions, np.nan)
    valid = np.zeros((n_time, n_spacecraft), dtype=bool)
    interpolation = np.zeros_like(valid)
    fractions = np.full((n_time, n_spacecraft), np.nan)
    brackets = np.full((n_time, n_spacecraft, 2), -1, dtype=np.int64)
    for spacecraft in range(n_spacecraft):
        source_time = time + offsets[spacecraft]
        for target_index, target in enumerate(time):
            insertion = int(np.searchsorted(source_time, target, side="left"))
            if insertion < n_time and np.isclose(
                source_time[insertion], target, rtol=0.0, atol=1.0e-12
            ):
                left = right = insertion
                fraction = 0.0
            else:
                right = insertion
                left = right - 1
                if left < 0 or right >= n_time:
                    if not allow_extrapolation:
                        continue
                    left, right = ((0, 1) if target < source_time[0] else (n_time - 2, n_time - 1))
                width = source_time[right] - source_time[left]
                if not np.isfinite(width) or width <= 0.0:
                    continue
                fraction = float((target - source_time[left]) / width)
            if left == right:
                magnetic = source_field[left, spacecraft]
                position = source_positions[spacecraft, left]
            else:
                magnetic = (
                    (1.0 - fraction) * source_field[left, spacecraft]
                    + fraction * source_field[right, spacecraft]
                )
                position = (
                    (1.0 - fraction) * source_positions[spacecraft, left]
                    + fraction * source_positions[spacecraft, right]
                )
                interpolation[target_index, spacecraft] = True
            if not np.all(np.isfinite(magnetic)) or not np.all(np.isfinite(position)):
                continue
            aligned_field[target_index, spacecraft] = magnetic
            aligned_positions[spacecraft, target_index] = position
            valid[target_index, spacecraft] = True
            fractions[target_index, spacecraft] = fraction
            brackets[target_index, spacecraft] = (left, right)
    return TimeAlignment(
        field=aligned_field,
        positions=aligned_positions,
        valid=valid,
        interpolation_required=interpolation,
        fractions=fractions,
        brackets=brackets,
    )


def periodic_grid_gradient(
    magnetic_grid: np.ndarray,
    box_size: Sequence[float],
) -> np.ndarray:
    """Second-order centered periodic derivative of ``B(nx,ny,nz,3)``."""

    magnetic = np.asarray(magnetic_grid, dtype=float)
    box = np.asarray(box_size, dtype=float)
    if magnetic.ndim != 4 or magnetic.shape[-1] != 3:
        raise ValueError("magnetic_grid must have shape (nx, ny, nz, 3)")
    if box.shape != (3,) or np.any(box <= 0.0):
        raise ValueError("box_size must contain three positive lengths")
    spacing = box / np.asarray(magnetic.shape[:3], dtype=float)
    gradient = np.empty((*magnetic.shape[:3], 3, 3), dtype=float)
    for derivative_axis in range(3):
        gradient[..., :, derivative_axis] = (
            np.roll(magnetic, -1, axis=derivative_axis)
            - np.roll(magnetic, 1, axis=derivative_axis)
        ) / (2.0 * spacing[derivative_axis])
    return gradient


def periodic_spherical_average(
    grid: np.ndarray,
    center: np.ndarray,
    radius: float,
    box_size: Sequence[float],
) -> tuple[np.ndarray, int, bool]:
    """Average grid nodes in a periodic sphere using minimum-image distance."""

    values = np.asarray(grid, dtype=float)
    box = np.asarray(box_size, dtype=float)
    point = np.mod(np.asarray(center, dtype=float), box)
    if values.ndim < 3 or box.shape != (3,) or radius < 0.0:
        raise ValueError("invalid periodic grid, center, or radius")
    shape = np.asarray(values.shape[:3], dtype=int)
    spacing = box / shape
    if radius == 0.0:
        index = np.mod(np.rint(point / spacing).astype(int), shape)
        return values[tuple(index)], 1, False

    index_sets: list[np.ndarray] = []
    wrapped = False
    for axis in range(3):
        if radius >= 0.5 * box[axis]:
            indices = np.arange(shape[axis], dtype=int)
            wrapped = True
        else:
            center_index = int(np.floor(point[axis] / spacing[axis]))
            extent = int(np.ceil(radius / spacing[axis])) + 1
            raw = np.arange(center_index - extent, center_index + extent + 1)
            wrapped |= bool(np.any((raw < 0) | (raw >= shape[axis])))
            indices = np.unique(np.mod(raw, shape[axis]))
        index_sets.append(indices)
    ix, iy, iz = np.meshgrid(*index_sets, indexing="ij")
    coordinates = np.stack((ix * spacing[0], iy * spacing[1], iz * spacing[2]), axis=-1)
    displacement = (coordinates - point + 0.5 * box) % box - 0.5 * box
    mask = np.sum(displacement**2, axis=-1) <= radius**2 * (1.0 + 32.0 * np.finfo(float).eps)
    selected = values[ix[mask], iy[mask], iz[mask]]
    if not len(selected):
        return np.full(values.shape[3:], np.nan), 0, wrapped
    return np.mean(selected, axis=0), int(len(selected)), wrapped


def periodic_spherical_average_fft(
    grid: np.ndarray,
    centers: np.ndarray,
    radius: float,
    box_size: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a periodic spherical top-hat convolution at arbitrary centers.

    The convolution is exact for spheres centered on grid nodes.  Values at
    off-grid centers are obtained by periodic trilinear interpolation of that
    convolved grid.
    """

    values = np.asarray(grid, dtype=float)
    points = np.asarray(centers, dtype=float)
    box = np.asarray(box_size, dtype=float)
    if values.ndim < 3 or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("grid and centers must have shapes (nx,ny,nz,...) and (n,3)")
    if box.shape != (3,) or np.any(box <= 0.0) or radius < 0.0:
        raise ValueError("invalid periodic grid, centers, radius, or box size")
    if not np.all(np.isfinite(points)):
        raise ValueError("centers contain non-finite values")

    shape = np.asarray(values.shape[:3], dtype=int)
    spacing = box / shape
    offsets_squared = []
    for count, step in zip(shape, spacing):
        indices = np.arange(count)
        offsets_squared.append((np.minimum(indices, count - indices) * step) ** 2)
    xy_squared = offsets_squared[0][:, None] + offsets_squared[1][None, :]
    mask = np.empty(tuple(shape), dtype=bool)
    threshold = radius**2 * (1.0 + 32.0 * np.finfo(float).eps)
    for z_index, z_squared in enumerate(offsets_squared[2]):
        mask[:, :, z_index] = xy_squared + z_squared <= threshold
    count = int(np.count_nonzero(mask))
    if count == 0:
        output = np.full((len(points), *values.shape[3:]), np.nan)
        return output, np.zeros(len(points), dtype=np.int64), np.zeros(len(points), dtype=bool)

    kernel = mask.astype(float)
    kernel /= float(count)
    kernel_spectrum = np.fft.rfftn(kernel, axes=(0, 1, 2))
    flat_values = values.reshape((*shape, -1))
    sampled = np.empty((len(points), flat_values.shape[-1]), dtype=float)
    interpolator = PeriodicTrilinearInterpolator(box)
    for component in range(flat_values.shape[-1]):
        spectrum = np.fft.rfftn(flat_values[..., component], axes=(0, 1, 2))
        spectrum *= kernel_spectrum
        convolved = np.fft.irfftn(spectrum, s=tuple(shape), axes=(0, 1, 2))
        sampled[:, component] = interpolator.sample(convolved, points)

    mapped = np.mod(points, box)
    wrapped = np.any((mapped < radius) | (mapped + radius >= box), axis=1)
    return (
        sampled.reshape((len(points), *values.shape[3:])),
        np.full(len(points), count, dtype=np.int64),
        wrapped,
    )


def select_configurations(
    positions: np.ndarray,
    *,
    max_q: float = 0.6,
    intermediate_l_min_km: float = 800.0,
    intermediate_l_max_km: float = 1200.0,
    large_l_min_km: float = 1500.0,
    enabled: Mapping[str, bool] | None = None,
) -> dict[str, dict[str, Any]]:
    """Select deterministic tetrahedra, best 5--8 point subset, and all nine."""

    points = np.asarray(positions, dtype=float)
    if points.shape != (9, 3):
        raise ValueError("configuration selection requires exactly nine spacecraft")
    use = {name: name != "best_multi" for name in CONFIGURATION_ORDER}
    use.update(dict(enabled or {}))
    candidates = []
    for indices in combinations(range(9), 4):
        geometry = geometry_diagnostics(points[np.asarray(indices)])
        candidates.append((indices, geometry))

    selected: dict[str, dict[str, Any]] = {}
    specifications = {
        "small": (
            [item for item in candidates if item[1].chi <= max_q],
            lambda item: (item[1].characteristic_size, item[1].chi, item[0]),
            f"minimum L among tetrahedra with Q <= {max_q:g}",
        ),
        "intermediate": (
            [
                item
                for item in candidates
                if intermediate_l_min_km
                <= item[1].characteristic_size
                <= intermediate_l_max_km
            ],
            lambda item: (item[1].chi, abs(item[1].characteristic_size - 1000.0), item[0]),
            f"minimum Q with {intermediate_l_min_km:g} <= L <= {intermediate_l_max_km:g} km",
        ),
        "large": (
            [item for item in candidates if item[1].characteristic_size >= large_l_min_km],
            lambda item: (item[1].chi, item[1].characteristic_size, item[0]),
            f"minimum Q with L >= {large_l_min_km:g} km",
        ),
    }
    for category, (eligible, key, criterion) in specifications.items():
        if not use[category]:
            continue
        if not eligible:
            raise ValueError(f"No tetrahedron satisfies the {category} selection: {criterion}")
        indices, geometry = min(eligible, key=key)
        selected[category] = {
            "indices": np.asarray(indices, dtype=int),
            "geometry": geometry,
            "criterion": criterion,
        }
    if use["best_multi"]:
        multi_candidates = []
        for count in range(5, 9):
            for indices in combinations(range(9), count):
                geometry = geometry_diagnostics(points[np.asarray(indices)])
                multi_candidates.append((indices, geometry))
        indices, geometry = min(
            multi_candidates,
            key=lambda item: (
                item[1].chi,
                item[1].condition_number,
                item[1].characteristic_size,
                item[0],
            ),
        )
        selected["best_multi"] = {
            "indices": np.asarray(indices, dtype=int),
            "geometry": geometry,
            "criterion": "minimum Q among all subsets with 5 <= N <= 8",
        }
    if use["all_nine"]:
        selected["all_nine"] = {
            "indices": np.arange(9, dtype=int),
            "geometry": geometry_diagnostics(points),
            "criterion": "all nine spacecraft",
        }
    return selected


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    gradient = dict(config.get("gradient", {}))
    time_sync = dict(config.get("time_sync", {}))
    selection = dict(config.get("configuration_selection", {}))
    truth = dict(config.get("truth", {}))
    ratio = dict(config.get("divergence_ratio", {}))
    monte_carlo = dict(config.get("monte_carlo", {}))
    storage = dict(config.get("storage", config.get("output", {})))
    diagnostics = dict(config.get("diagnostics", {}))
    method = str(gradient.get("method", "least_squares"))
    if method != "least_squares":
        raise ValueError("magnetic_gradient.gradient.method must be least_squares")
    sync_method = str(time_sync.get("method", "linear"))
    if sync_method != "linear":
        raise ValueError("magnetic_gradient.time_sync.method must be linear")
    shape_metric = str(selection.get("shape_metric", "q")).lower()
    if shape_metric not in {"q", "chi"}:
        raise ValueError("Only the repository's Q/chi shape metric is currently available")
    if int(truth.get("derivative_order", 2)) != 2:
        raise ValueError("magnetic_gradient.truth.derivative_order must be 2")
    coarse_average_method = str(truth.get("coarse_average_method", "fft")).lower()
    if coarse_average_method not in {"direct", "fft"}:
        raise ValueError("magnetic_gradient.truth.coarse_average_method must be direct or fft")
    if str(ratio.get("denominator", "curl_magnitude")) != "curl_magnitude":
        raise ValueError("divergence_ratio.denominator must be curl_magnitude")
    floor_fraction = float(ratio.get("floor_fraction_of_median_truth", 0.01))
    if not np.isfinite(floor_fraction) or floor_fraction < 0.0:
        raise ValueError("divergence ratio floor fraction must be finite and non-negative")
    percentiles = np.asarray(monte_carlo.get("percentiles", (25.0, 50.0, 75.0)), dtype=float)
    if percentiles.shape != (3,) or not np.array_equal(percentiles, (25.0, 50.0, 75.0)):
        raise ValueError("magnetic_gradient currently stores percentiles [25, 50, 75]")
    compression = storage.get("compression", "gzip")
    if compression in {"none", "null", None, False}:
        compression = None
    if compression not in {None, "gzip", "lzf"}:
        raise ValueError("storage.compression must be gzip, lzf, or none")
    enabled = {}
    for name in CONFIGURATION_ORDER:
        section = selection.get(name, {})
        default_enabled = name != "best_multi"
        enabled[name] = (
            bool(section.get("enabled", default_enabled))
            if isinstance(section, Mapping)
            else bool(section)
        )
    return {
        "use_true": bool(config.get("use_true", True)),
        "use_measured": bool(config.get("use_measured", True)),
        "gradient": {
            "rcond": gradient.get("rcond"),
            "condition_number_max": float(gradient.get("condition_number_max", 1.0e12)),
            "allow_pseudoinverse": bool(gradient.get("allow_pseudoinverse", True)),
        },
        "time_sync": {"allow_extrapolation": bool(time_sync.get("allow_extrapolation", False))},
        "selection": {
            "max_q": float(selection.get("max_q", selection.get("max_chi", 0.6))),
            "intermediate_l_min_km": float(
                dict(selection.get("intermediate", {})).get("l_min_km", 800.0)
            ),
            "intermediate_l_max_km": float(
                dict(selection.get("intermediate", {})).get("l_max_km", 1200.0)
            ),
            "large_l_min_km": float(
                dict(selection.get("large", {})).get("l_min_km", 1500.0)
            ),
            "enabled": enabled,
        },
        "truth": {
            "pointwise_barycenter": bool(truth.get("pointwise_barycenter", True)),
            "derivative_order": int(truth.get("derivative_order", 2)),
            "coarse_average_method": coarse_average_method,
        },
        "ratio_floor_fraction": floor_fraction,
        "store_all_realizations": bool(monte_carlo.get("store_all_realizations", True)),
        "compression": compression,
        "compression_level": int(storage.get("compression_level", 4)),
        "chunk_time": int(storage.get("chunk_time", 256)),
        "verbose": bool(diagnostics.get("verbose", False)),
        "verbose_time_index": int(diagnostics.get("time_index", 0)),
        "verbose_mc_index": int(diagnostics.get("mc_index", 0)),
        "simulation_override": dict(config.get("simulation_override", {})),
        "analysis_config_yaml": str(config.get("_analysis_config_yaml", "")),
        "analysis_config_path": str(config.get("_analysis_config_path", "")),
    }


def _geometry_series(positions: np.ndarray) -> dict[str, np.ndarray | bool]:
    points = np.asarray(positions, dtype=float)
    records = [geometry_diagnostics(sample) for sample in points]
    is_static = geometry_is_static(points)
    return {
        "barycenter": np.asarray([item.barycenter for item in records]),
        "volumetric_tensor": np.asarray([item.volumetric_tensor for item in records]),
        "eigenvalues": np.asarray([item.eigenvalues for item in records]),
        "semiaxes": np.asarray([item.semiaxes for item in records]),
        "characteristic_size": np.asarray([item.characteristic_size for item in records]),
        "elongation": np.asarray([item.elongation for item in records]),
        "planarity": np.asarray([item.planarity for item in records]),
        "chi": np.asarray([item.chi for item in records]),
        "d_ep": np.full(len(points), np.nan),
        "condition_number": np.asarray([item.condition_number for item in records]),
        "rank": np.asarray([item.rank for item in records], dtype=np.int8),
        "is_static": is_static,
    }


def _estimate_series(
    positions: np.ndarray,
    field: np.ndarray,
    settings: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    n_time, n_spacecraft, _ = positions.shape
    result = {
        "positions": np.asarray(positions, dtype=float).copy(),
        "gradient": np.full((n_time, 3, 3), np.nan),
        "barycenter": np.full((n_time, 3), np.nan),
        "volumetric_tensor": np.full((n_time, 3, 3), np.nan),
        "inverse_volumetric_tensor": np.full((n_time, 3, 3), np.nan),
        "eigenvalues": np.full((n_time, 3), np.nan),
        "singular_values": np.full((n_time, 3), np.nan),
        "fitted_field": np.full((n_time, n_spacecraft, 3), np.nan),
        "affine_residual": np.full((n_time, n_spacecraft, 3), np.nan),
        "component_rms_residual": np.full((n_time, 3), np.nan),
        "affine_rms_residual": np.full(n_time, np.nan),
        "condition_number": np.full(n_time, np.nan),
        "characteristic_size": np.full(n_time, np.nan),
        "elongation": np.full(n_time, np.nan),
        "planarity": np.full(n_time, np.nan),
        "chi": np.full(n_time, np.nan),
        "d_ep": np.full(n_time, np.nan),
        "rank": np.zeros(n_time, dtype=np.int8),
        "used_pseudoinverse": np.zeros(n_time, dtype=bool),
        "valid": np.zeros(n_time, dtype=bool),
    }
    for time_index in range(n_time):
        estimate = estimate_gradient_least_squares(
            positions[time_index], field[time_index], **settings
        )
        for name in (
            "gradient",
            "barycenter",
            "volumetric_tensor",
            "inverse_volumetric_tensor",
            "eigenvalues",
            "singular_values",
            "fitted_field",
        ):
            result[name][time_index] = getattr(estimate, name)
        result["affine_residual"][time_index] = estimate.residuals
        result["component_rms_residual"][time_index] = estimate.component_rms_residual
        result["affine_rms_residual"][time_index] = estimate.rms_residual
        result["condition_number"][time_index] = estimate.condition_number
        semiaxes = np.sqrt(np.clip(estimate.eigenvalues, 0.0, None))
        a, b, c = semiaxes
        result["characteristic_size"][time_index] = 2.0 * a
        result["elongation"][time_index] = 0.0 if a <= 0.0 else 1.0 - b / a
        result["planarity"][time_index] = 0.0 if b <= 0.0 else 1.0 - c / b
        result["chi"][time_index] = np.hypot(
            result["elongation"][time_index], result["planarity"][time_index]
        )
        result["rank"][time_index] = estimate.rank
        result["used_pseudoinverse"][time_index] = estimate.used_pseudoinverse
        result["valid"][time_index] = estimate.valid
    return result


def _add_derived(products: dict[str, np.ndarray], floor: float) -> None:
    curl, curl_magnitude, divergence = gradient_curl_divergence(products["gradient"])
    ratio, ratio_valid = divergence_curl_ratio(divergence, curl_magnitude, floor)
    products.update(
        curl=curl,
        curl_magnitude=curl_magnitude,
        divergence=divergence,
        divergence_curl_ratio=ratio,
        divergence_curl_ratio_valid=ratio_valid,
    )


def _safe_normalized(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    finite = denominator[np.isfinite(denominator) & (denominator > 0.0)]
    robust = float(np.median(finite)) if finite.size else 1.0
    floor = np.sqrt(np.finfo(float).eps) * max(robust, 1.0)
    result = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan)
    np.divide(
        numerator,
        denominator,
        out=result,
        where=np.isfinite(denominator) & (denominator > floor),
    )
    return result


def _add_errors(products: dict[str, np.ndarray], truth: Mapping[str, np.ndarray]) -> None:
    gradient_delta = products["gradient"] - truth["gradient"]
    gradient_norm = np.linalg.norm(gradient_delta, axis=(-2, -1))
    truth_gradient_norm = np.linalg.norm(truth["gradient"], axis=(-2, -1))
    curl_delta = products["curl"] - truth["curl"]
    curl_error = np.linalg.norm(curl_delta, axis=-1)
    magnitude_delta = products["curl_magnitude"] - truth["curl_magnitude"]
    products.update(
        gradient_delta=gradient_delta,
        gradient_frobenius_error=gradient_norm,
        gradient_relative_frobenius_error=_safe_normalized(gradient_norm, truth_gradient_norm),
        curl_delta=curl_delta,
        curl_l2_error=curl_error,
        curl_relative_l2_error=_safe_normalized(curl_error, truth["curl_magnitude"]),
        curl_magnitude_delta=magnitude_delta,
        curl_magnitude_absolute_error=np.abs(magnitude_delta),
        curl_magnitude_relative_error=_safe_normalized(
            np.abs(magnitude_delta), truth["curl_magnitude"]
        ),
    )


def _load_simulation(
    metadata: Mapping[str, Any], settings: Mapping[str, Any]
) -> tuple[SimulationSampledField, dict[str, Any]]:
    config_yaml = metadata.get("config_yaml")
    if isinstance(config_yaml, bytes):
        config_yaml = config_yaml.decode("utf-8")
    if not config_yaml:
        raise ValueError("Observation metadata is missing the embedded generation config_yaml")
    generation = yaml.safe_load(str(config_yaml))
    if not isinstance(generation, Mapping) or not isinstance(generation.get("field"), Mapping):
        raise ValueError("Embedded generation YAML has no field mapping")
    field_config = dict(generation["field"])
    if str(field_config.get("model")) != "simulation_sampled":
        raise ValueError(
            "magnetic_gradient requires observations from field.model: simulation_sampled"
        )
    if settings["simulation_override"]:
        simulation = _deep_merge(
            dict(field_config.get("simulation", {})), settings["simulation_override"]
        )
        field_config["simulation"] = simulation
    return SimulationSampledField(field_config), generation


def _truth_products(
    field_model: SimulationSampledField,
    gradient_grid: np.ndarray,
    true_positions: np.ndarray,
    geometry: Mapping[str, Any],
    average_method: str,
) -> dict[str, np.ndarray]:
    barycenter = np.asarray(geometry["barycenter"])
    initial_hub = np.asarray(true_positions[0, 0], dtype=float)
    mapped = np.mod(
        barycenter - initial_hub + 0.5 * field_model.box_size_km,
        field_model.box_size_km,
    )
    interpolator = PeriodicTrilinearInterpolator(field_model.box_size_km)
    point_gradient = interpolator.sample(
        gradient_grid.reshape((*gradient_grid.shape[:3], 9)), mapped
    ).reshape((len(mapped), 3, 3))
    coarse_gradient = np.full_like(point_gradient, np.nan)
    cell_count = np.zeros(len(mapped), dtype=np.int64)
    wrapped = np.zeros(len(mapped), dtype=bool)
    sizes = np.asarray(geometry["characteristic_size"], dtype=float)
    if average_method == "fft":
        if not np.allclose(sizes, sizes[0], rtol=1.0e-10, atol=1.0e-8):
            raise ValueError(
                "FFT coarse truth requires static configuration size; "
                "use truth.coarse_average_method: direct for time-varying geometry"
            )
        coarse_gradient, cell_count, wrapped = periodic_spherical_average_fft(
            gradient_grid, mapped, 0.5 * float(sizes[0]), field_model.box_size_km
        )
    else:
        for time_index, (center, size) in enumerate(zip(mapped, sizes)):
            coarse_gradient[time_index], cell_count[time_index], wrapped[time_index] = (
                periodic_spherical_average(
                    gradient_grid, center, 0.5 * float(size), field_model.box_size_km
                )
            )
    return {
        "coarse_gradient": coarse_gradient,
        "point_gradient": point_gradient,
        "coarse_cell_count": cell_count,
        "periodic_wrap": wrapped,
    }


def _truth_derived(products: dict[str, np.ndarray], floor_fraction: float) -> float:
    coarse_curl, coarse_magnitude, coarse_divergence = gradient_curl_divergence(
        products["coarse_gradient"]
    )
    finite = coarse_magnitude[np.isfinite(coarse_magnitude)]
    median = float(np.median(finite)) if finite.size else float("nan")
    floor = float(floor_fraction * median) if np.isfinite(median) else float("inf")
    point_curl, point_magnitude, point_divergence = gradient_curl_divergence(
        products["point_gradient"]
    )
    coarse_ratio, coarse_valid = divergence_curl_ratio(
        coarse_divergence, coarse_magnitude, floor
    )
    point_ratio, point_valid = divergence_curl_ratio(point_divergence, point_magnitude, floor)
    products.update(
        coarse_curl=coarse_curl,
        coarse_curl_magnitude=coarse_magnitude,
        coarse_divergence=coarse_divergence,
        coarse_divergence_curl_ratio=coarse_ratio,
        coarse_divergence_curl_ratio_valid=coarse_valid,
        point_curl=point_curl,
        point_curl_magnitude=point_magnitude,
        point_divergence=point_divergence,
        point_divergence_curl_ratio=point_ratio,
        point_divergence_curl_ratio_valid=point_valid,
        coarse_valid=np.all(np.isfinite(products["coarse_gradient"]), axis=(-2, -1))
        & (products["coarse_cell_count"] > 0),
        point_valid=np.all(np.isfinite(products["point_gradient"]), axis=(-2, -1)),
    )
    products["valid"] = products["coarse_valid"]
    return floor


def _coordinate_offsets(
    handle: h5py.File,
    metadata: Mapping[str, Any],
    n_mc: int,
    n_spacecraft: int,
) -> tuple[np.ndarray, np.ndarray]:
    error_models: dict[str, Any] = {}
    try:
        error_models = json.loads(str(metadata.get("error_models", "{}")))
    except (TypeError, json.JSONDecodeError):
        pass
    time_active = str(error_models.get("time", "none")) not in {"none", "no_error"}
    position_active = str(error_models.get("position", "none")) not in {"none", "no_error"}
    if "errors/time_offset" in handle:
        time_offset = np.asarray(handle["errors/time_offset"], dtype=float)
    elif time_active:
        raise ValueError("Active timing errors require /errors/time_offset in the observation file")
    else:
        time_offset = np.zeros((n_mc, n_spacecraft))
    if "errors/position_offset" in handle:
        position_offset = np.asarray(handle["errors/position_offset"], dtype=float)
    elif position_active:
        raise ValueError(
            "Active position errors require /errors/position_offset in the observation file"
        )
    else:
        position_offset = np.zeros((n_mc, n_spacecraft, 3))
    if time_offset.shape != (n_mc, n_spacecraft):
        raise ValueError("/errors/time_offset has an incompatible shape")
    if position_offset.shape != (n_mc, n_spacecraft, 3):
        raise ValueError("/errors/position_offset has an incompatible shape")
    return time_offset, position_offset


def compute_magnetic_gradient_analysis(
    input_path: str | Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Calculate truth, true-sample, and measured magnetic-gradient products."""

    settings = _settings(config)
    source = Path(input_path)
    with h5py.File(source, "r") as handle:
        for required in ("time/B", "spacecraft/r/B", "truth/B", "spacecraft/id"):
            if required not in handle:
                raise KeyError(f"Observation file is missing /{required}")
        metadata = {
            key: value.decode("utf-8") if isinstance(value, bytes) else value
            for key, value in handle["metadata"].attrs.items()
        }
        time = np.asarray(handle["time/B"], dtype=float)
        nominal_positions = np.asarray(handle["spacecraft/r/B"], dtype=float)
        true_field = np.asarray(handle["truth/B"], dtype=float)
        spacecraft_ids = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in handle["spacecraft/id"][...]
        ]
        if len(spacecraft_ids) != 9:
            raise ValueError("magnetic_gradient requires the full nine-spacecraft observatory")
        if nominal_positions.shape != (9, len(time), 3):
            raise ValueError("/spacecraft/r/B must have shape (9, time, 3)")
        if true_field.shape != (len(time), 9, 3):
            raise ValueError("/truth/B must have shape (time, 9, 3)")
        observed = None
        time_offset = position_offset = None
        if settings["use_measured"]:
            if "observed/B" not in handle:
                raise KeyError("Measured analysis requires /observed/B")
            observed = np.asarray(handle["observed/B"], dtype=float)
            if observed.ndim != 4 or observed.shape[1:] != (len(time), 9, 3):
                raise ValueError("/observed/B must have shape (mc, time, 9, 3)")
            time_offset, position_offset = _coordinate_offsets(
                handle, metadata, observed.shape[0], 9
            )

    if len(time) < 2:
        raise ValueError("magnetic_gradient requires at least two magnetic-field samples")
    true_positions = np.swapaxes(nominal_positions, 0, 1)
    selected = select_configurations(true_positions[0], **settings["selection"])
    simulation, generation_config = _load_simulation(metadata, settings)
    magnetic_grid = (
        np.asarray(simulation.snapshot.B, dtype=float) * simulation.scale_factors["B"]
    )
    simulation_gradient = periodic_grid_gradient(magnetic_grid, simulation.box_size_km)
    configurations: dict[str, dict[str, Any]] = {}
    truth_metadata: dict[str, Any] = {
        "simulation_snapshots_used": 1,
        "derivative_method": "second-order centered finite difference",
        "coarse_average_method": settings["truth"]["coarse_average_method"],
        "coarse_average_interpolation": (
            "periodic trilinear after grid-node convolution"
            if settings["truth"]["coarse_average_method"] == "fft"
            else "none; direct grid-node selection at each barycenter"
        ),
        "native_gradient_used": False,
        "simulation_grid_shape": list(magnetic_grid.shape[:3]),
        "box_size_km": simulation.box_size_km.tolist(),
    }

    for category in CONFIGURATION_ORDER:
        if category not in selected:
            continue
        indices = selected[category]["indices"]
        positions = true_positions[:, indices]
        fields = true_field[:, indices]
        geometry = _geometry_series(positions)
        geometry["pseudoinverse_required"] = (
            (np.asarray(geometry["rank"]) < 3)
            | (
                np.asarray(geometry["condition_number"])
                > settings["gradient"]["condition_number_max"]
            )
        )
        truth = _truth_products(
            simulation,
            simulation_gradient,
            true_positions,
            geometry,
            settings["truth"]["coarse_average_method"],
        )
        floor = _truth_derived(truth, settings["ratio_floor_fraction"])
        true_estimate = _estimate_series(positions, fields, settings["gradient"])
        _add_derived(true_estimate, floor)
        _add_errors(
            true_estimate,
            {
                "gradient": truth["coarse_gradient"],
                "curl": truth["coarse_curl"],
                "curl_magnitude": truth["coarse_curl_magnitude"],
            },
        )
        configurations[category] = {
            "indices": indices,
            "spacecraft_ids": [spacecraft_ids[index] for index in indices],
            "criterion": selected[category]["criterion"],
            "geometry": geometry,
            "truth": truth,
            "curl_floor": floor,
            "true_estimate": true_estimate,
        }

    if settings["use_measured"]:
        assert observed is not None and time_offset is not None and position_offset is not None
        for mc_index in range(len(observed)):
            measured_positions = nominal_positions + position_offset[mc_index, :, None, :]
            aligned = align_spacecraft_samples(
                time,
                time_offset[mc_index],
                measured_positions,
                observed[mc_index],
                allow_extrapolation=settings["time_sync"]["allow_extrapolation"],
            )
            aligned_positions = np.swapaxes(aligned.positions, 0, 1)
            for category, product in configurations.items():
                indices = product["indices"]
                estimate = _estimate_series(
                    aligned_positions[:, indices],
                    aligned.field[:, indices],
                    settings["gradient"],
                )
                _add_derived(estimate, product["curl_floor"])
                truth = product["truth"]
                _add_errors(
                    estimate,
                    {
                        "gradient": truth["coarse_gradient"],
                        "curl": truth["coarse_curl"],
                        "curl_magnitude": truth["coarse_curl_magnitude"],
                    },
                )
                synchronization = {
                    "effective_time": time[:, None]
                    + time_offset[mc_index, indices][None, :],
                    "valid": aligned.valid[:, indices],
                    "interpolation_required": aligned.interpolation_required[:, indices],
                    "fractions": aligned.fractions[:, indices],
                    "brackets": aligned.brackets[:, indices],
                }
                if "measured" not in product:
                    product["measured"] = {
                        key: np.empty((len(observed), *value.shape), dtype=value.dtype)
                        for key, value in estimate.items()
                    }
                    product["synchronization"] = {
                        key: np.empty((len(observed), *value.shape), dtype=value.dtype)
                        for key, value in synchronization.items()
                    }
                for key, value in estimate.items():
                    product["measured"][key][mc_index] = value
                for key, value in synchronization.items():
                    product["synchronization"][key][mc_index] = value
            report_interval = max(1, len(observed) // 10)
            if (mc_index + 1) % report_interval == 0 or mc_index + 1 == len(observed):
                print(
                    f"Magnetic-gradient MC progress: {mc_index + 1}/{len(observed)}",
                    flush=True,
                )

        for product in configurations.values():
            product["mc_summary"] = _mc_summary(product["measured"])

    diagnostics = _diagnostic_counts(configurations)
    result = {
        "metadata": {
            "method": "magnetic_gradient",
            "source_file": str(source.resolve()),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "code_version": __version__,
            "git_commit": _git_commit(),
            "coordinate_system": (
                "observation/native; simulation mapping from embedded generation YAML"
            ),
            "gradient_index_convention": "gradient[field_component, spatial_derivative]",
            "spacecraft_index_convention": "zero-based HDF5 array order; IDs are authoritative",
            "position_units": "km",
            "magnetic_field_units": "nT",
            "gradient_units": "nT/km",
            "time_units": "s",
            "analysis_config": settings["analysis_config_yaml"] or json.dumps(config, default=str),
            "analysis_config_path": settings["analysis_config_path"],
            "source_generation_config": yaml.safe_dump(generation_config, sort_keys=True),
            "shape_metric": "Q = chi = sqrt(E^2 + P^2)",
            "selection_time_index": 0,
            "simulation": truth_metadata,
            "simulation_source": simulation.metadata.get(
                "simulation_sources", simulation.metadata.get("simulation_path")
            ),
            "normalized_error_denominator_rule": (
                "denominator > sqrt(machine_epsilon) * max(median_positive, 1)"
            ),
            "diagnostics": diagnostics,
        },
        "settings": settings,
        "time": time,
        "simulation_time": np.full(len(time), np.nan),
        "spacecraft_ids": spacecraft_ids,
        "configurations": configurations,
    }
    _print_diagnostics(result)
    if settings["verbose"]:
        _print_verbose(result)
    return result


SUMMARY_QUANTITIES = (
    "gradient",
    "curl",
    "curl_magnitude",
    "divergence",
    "divergence_curl_ratio",
    "affine_rms_residual",
    "condition_number",
    "chi",
    "d_ep",
    "gradient_delta",
    "gradient_frobenius_error",
    "gradient_relative_frobenius_error",
    "curl_delta",
    "curl_l2_error",
    "curl_relative_l2_error",
    "curl_magnitude_delta",
    "curl_magnitude_absolute_error",
    "curl_magnitude_relative_error",
)


def _mc_summary(measured: Mapping[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    summaries: dict[str, dict[str, np.ndarray]] = {}
    for name in SUMMARY_QUANTITIES:
        values = np.asarray(measured[name], dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            q25, median, q75 = np.nanpercentile(values, (25.0, 50.0, 75.0), axis=0)
        count = np.count_nonzero(np.isfinite(values), axis=0)
        summaries[name] = {
            "q25": q25,
            "median": median,
            "q75": q75,
            "valid_count": count,
            "valid_fraction": count / float(values.shape[0]),
        }
    return summaries


def _diagnostic_counts(configurations: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    invalid_geometry = 0
    pseudoinverse = 0
    interpolation_failures = 0
    small_curl = 0
    wrapped = 0
    cell_counts: list[np.ndarray] = []
    for product in configurations.values():
        true = product["true_estimate"]
        invalid_geometry += int(np.count_nonzero(~true["valid"]))
        pseudoinverse += int(np.count_nonzero(true["used_pseudoinverse"]))
        truth = product["truth"]
        small_curl += int(np.count_nonzero(~truth["coarse_divergence_curl_ratio_valid"]))
        wrapped += int(np.count_nonzero(truth["periodic_wrap"]))
        cell_counts.append(truth["coarse_cell_count"])
        if "measured" in product:
            invalid_geometry += int(np.count_nonzero(~product["measured"]["valid"]))
            pseudoinverse += int(np.count_nonzero(product["measured"]["used_pseudoinverse"]))
            interpolation_failures += int(
                np.count_nonzero(~product["synchronization"]["valid"])
            )
            small_curl += int(
                np.count_nonzero(~product["measured"]["divergence_curl_ratio_valid"])
            )
    cells = np.concatenate(cell_counts) if cell_counts else np.asarray([], dtype=int)
    return {
        "invalid_geometry_samples": invalid_geometry,
        "pseudoinverse_uses": pseudoinverse,
        "timing_interpolation_failures": interpolation_failures,
        "small_curl_ratio_exclusions": small_curl,
        "periodic_wrap_cases": wrapped,
        "coarse_cells_min": int(np.min(cells)) if cells.size else 0,
        "coarse_cells_median": float(np.median(cells)) if cells.size else 0.0,
        "coarse_cells_max": int(np.max(cells)) if cells.size else 0,
    }


def _git_commit() -> str:
    try:
        return subprocess.run(
            ("git", "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _print_diagnostics(result: Mapping[str, Any]) -> None:
    print("Selected configurations:")
    for category, product in result["configurations"].items():
        geometry = product["geometry"]
        print(
            f"  {category:12s} spacecraft {product['spacecraft_ids']}, "
            f"L={np.nanmedian(geometry['characteristic_size']):.6g} km, "
            f"dEP=unavailable, chi={np.nanmedian(geometry['chi']):.6g}"
        )
    first = next(iter(result["configurations"].values()))
    n_mc = first.get("measured", {}).get("gradient", np.empty((0,))).shape[0]
    counts = result["metadata"]["diagnostics"]
    print("Gradient calculation:")
    print(f"  time samples: {len(result['time'])}")
    print(f"  MC realizations: {n_mc}")
    for label, key in (
        ("invalid geometry samples", "invalid_geometry_samples"),
        ("pseudoinverse uses", "pseudoinverse_uses"),
        ("timing interpolation failures", "timing_interpolation_failures"),
        ("small-curl ratio exclusions", "small_curl_ratio_exclusions"),
    ):
        print(f"  {label}: {counts[key]}")
    print("Truth calculation:")
    print("  simulation snapshots used: 1")
    print(
        "  coarse-graining cells per configuration: "
        f"{counts['coarse_cells_min']}/{counts['coarse_cells_median']:.6g}/"
        f"{counts['coarse_cells_max']} (min/median/max)"
    )
    print(f"  periodic-wrap cases: {counts['periodic_wrap_cases']}")


def _print_verbose(result: Mapping[str, Any]) -> None:
    settings = result["settings"]
    time_index = settings["verbose_time_index"]
    mc_index = settings["verbose_mc_index"]
    for category, product in result["configurations"].items():
        if not 0 <= time_index < len(result["time"]):
            raise IndexError("diagnostics.time_index is outside the analysis time grid")
        estimate = product["true_estimate"]
        if "measured" in product:
            if not 0 <= mc_index < len(product["measured"]["gradient"]):
                raise IndexError("diagnostics.mc_index is outside the measured ensemble")
            estimate = {key: value[mc_index] for key, value in product["measured"].items()}
        print(f"Verbose magnetic-gradient diagnostic: {category}, time={time_index}, MC={mc_index}")
        for name in (
            "positions",
            "barycenter",
            "volumetric_tensor",
            "eigenvalues",
            "condition_number",
            "gradient",
            "curl",
            "divergence",
            "divergence_curl_ratio",
            "fitted_field",
            "affine_residual",
        ):
            print(f"  {name}:\n{estimate[name][time_index]}")
        print(f"  coarse truth:\n{product['truth']['coarse_gradient'][time_index]}")
        print(f"  point truth:\n{product['truth']['point_gradient'][time_index]}")


UNIT_BY_NAME = {
    "effective_time": "s",
    "positions": "km",
    "barycenter": "km",
    "characteristic_size": "km",
    "semiaxes": "km",
    "volumetric_tensor": "km^2",
    "inverse_volumetric_tensor": "km^-2",
    "eigenvalues": "km^2",
    "singular_values": "km^2",
    "gradient": "nT/km",
    "coarse_gradient": "nT/km",
    "point_gradient": "nT/km",
    "gradient_delta": "nT/km",
    "gradient_frobenius_error": "nT/km",
    "curl": "nT/km",
    "coarse_curl": "nT/km",
    "point_curl": "nT/km",
    "curl_delta": "nT/km",
    "curl_l2_error": "nT/km",
    "curl_magnitude": "nT/km",
    "coarse_curl_magnitude": "nT/km",
    "point_curl_magnitude": "nT/km",
    "curl_magnitude_delta": "nT/km",
    "curl_magnitude_absolute_error": "nT/km",
    "divergence": "nT/km",
    "coarse_divergence": "nT/km",
    "point_divergence": "nT/km",
    "fitted_field": "nT",
    "affine_residual": "nT",
    "component_rms_residual": "nT",
    "affine_rms_residual": "nT",
}


def _dataset(
    group: h5py.Group,
    name: str,
    values: np.ndarray,
    settings: Mapping[str, Any],
) -> h5py.Dataset:
    array = np.asarray(values)
    kwargs: dict[str, Any] = {}
    if array.ndim > 0 and array.size > 1 and array.dtype.kind not in {"O", "U", "S"}:
        compression = settings["compression"]
        if compression is not None:
            kwargs.update(compression=compression, chunks=True)
            if compression == "gzip":
                kwargs["compression_opts"] = settings["compression_level"]
    dataset = group.create_dataset(name, data=array, **kwargs)
    dataset.attrs["units"] = UNIT_BY_NAME.get(name, "1")
    if name in {"gradient", "coarse_gradient", "point_gradient", "gradient_delta"}:
        dataset.attrs["index_convention"] = "[field_component, spatial_derivative]"
    if name == "d_ep":
        dataset.attrs["available"] = False
    return dataset


def _write_array_mapping(
    group: h5py.Group,
    values: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> None:
    for name, value in values.items():
        if isinstance(value, np.ndarray):
            _dataset(group, name, value, settings)


def _write_metadata(group: h5py.Group, metadata: Mapping[str, Any]) -> None:
    for key, value in metadata.items():
        if value is None:
            value = "null"
        elif not isinstance(value, (str, bytes, bool, int, float, np.number)):
            value = json.dumps(value, sort_keys=True)
        group.attrs[key] = value


def write_magnetic_gradient_product(result: Mapping[str, Any], output_path: str | Path) -> Path:
    """Write the calculation product consumed by the standalone plotter."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    settings = result["settings"]
    with h5py.File(output, "w") as handle:
        _write_metadata(handle.create_group("metadata"), result["metadata"])
        time_group = handle.create_group("time")
        analysis_time = time_group.create_dataset("analysis_time", data=result["time"])
        analysis_time.attrs["units"] = "s"
        simulation_time = time_group.create_dataset(
            "simulation_time", data=result["simulation_time"]
        )
        simulation_time.attrs.update(units="simulation/native", available=False)
        valid = np.ones(len(result["time"]), dtype=bool)
        for product in result["configurations"].values():
            valid &= product["true_estimate"]["valid"]
        time_group.create_dataset("validity", data=valid)
        spacecraft = handle.create_group("spacecraft")
        spacecraft.create_dataset(
            "id", data=np.asarray(result["spacecraft_ids"], dtype=h5py.string_dtype("utf-8"))
        )
        configurations = handle.create_group("configurations")
        for category, product in result["configurations"].items():
            group = configurations.create_group(category)
            group.attrs.update(
                selection_category=category,
                selection_criterion=product["criterion"],
                number_of_spacecraft=len(product["indices"]),
                curl_floor=product["curl_floor"],
                curl_floor_units="nT/km",
            )
            group.create_dataset("spacecraft_indices", data=product["indices"])
            group.create_dataset(
                "spacecraft_ids",
                data=np.asarray(product["spacecraft_ids"], dtype=h5py.string_dtype("utf-8")),
            )
            geometry_group = group.create_group("geometry")
            geometry = product["geometry"]
            geometry_group.attrs.update(
                static_intrinsic_geometry=bool(geometry["is_static"]),
                d_ep_available=False,
                shape_metric="Q = chi",
            )
            for name, value in geometry.items():
                if not isinstance(value, np.ndarray):
                    continue
                stored = value
                if geometry["is_static"] and name != "barycenter":
                    stored = value[0]
                _dataset(geometry_group, name, stored, settings)
            truth_group = group.create_group("truth")
            _write_array_mapping(truth_group, product["truth"], settings)
            true_group = group.create_group("true_estimate")
            _write_array_mapping(true_group, product["true_estimate"], settings)
            if "measured" in product:
                measured_group = group.create_group("measured")
                if settings["store_all_realizations"]:
                    _write_array_mapping(measured_group, product["measured"], settings)
                sync_group = measured_group.create_group("synchronization")
                _write_array_mapping(sync_group, product["synchronization"], settings)
                summary_group = group.create_group("mc_summary")
                for quantity, summary in product["mc_summary"].items():
                    _write_array_mapping(summary_group.create_group(quantity), summary, settings)
    return output


def _read_scalar_or_series(dataset: h5py.Dataset, n_time: int) -> np.ndarray:
    values = np.asarray(dataset, dtype=float)
    return np.full(n_time, float(values)) if values.ndim == 0 else values


def _curl_error_decomposition(
    group: h5py.Group, time_mask: np.ndarray
) -> dict[str, np.ndarray | float]:
    """Return finite-scale and measurement curl-vector errors.

    ``C_C``, ``C_T``, and ``C_M`` denote the coarse-truth, ideal-estimate,
    and measured-realization curl vectors, respectively.  Both error terms
    are normalized by the median coarse-truth curl magnitude over the
    selected plotting interval.
    """

    coarse_curl = np.asarray(group["truth/coarse_curl"], dtype=float)[time_mask]
    ideal_curl = np.asarray(group["true_estimate/curl"], dtype=float)[time_mask]
    coarse_magnitude = np.linalg.norm(coarse_curl, axis=-1)
    finite_coarse_magnitude = coarse_magnitude[np.isfinite(coarse_magnitude)]
    if not finite_coarse_magnitude.size:
        raise ValueError(
            "curl-error decomposition requires a finite coarse-truth curl "
            "magnitude in the selected interval"
        )
    normalization = float(np.nanmedian(finite_coarse_magnitude))
    if normalization <= 0.0:
        raise ValueError(
            "curl-error decomposition requires a positive median coarse-truth "
            "curl magnitude in the selected interval"
        )
    finite_scale_error = np.linalg.norm(ideal_curl - coarse_curl, axis=-1)

    if "measured/curl" not in group:
        raise ValueError(
            "curl-error decomposition requires stored measured curl realizations; "
            "rerun the analysis with monte_carlo.store_all_realizations: true"
        )
    measured_curl = np.asarray(group["measured/curl"], dtype=float)[:, time_mask, :]
    measurement_error = np.linalg.norm(
        measured_curl - ideal_curl[np.newaxis, :, :], axis=-1
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        q25, median, q75 = np.nanpercentile(
            measurement_error, (25.0, 50.0, 75.0), axis=0
        )
    return {
        "normalization": normalization,
        "finite_scale_error": finite_scale_error / normalization,
        "measurement_error_q25": q25 / normalization,
        "measurement_error_median": median / normalization,
        "measurement_error_q75": q75 / normalization,
    }


def _plot_curl_error_decomposition(
    handle: h5py.File,
    categories: list[str],
    full_time: np.ndarray,
    time_mask: np.ndarray,
    output_path: str | Path,
    *,
    start_time_s: float | None,
    stop_time_s: float | None,
    dpi: int,
) -> Path:
    """Plot the compact curl finite-scale/measurement-error decomposition."""

    time = full_time[time_mask]
    figure, axes = plt.subplots(
        3,
        len(categories),
        figsize=(4.4 * len(categories), 9.5),
        squeeze=False,
        sharex=True,
    )
    row_values: list[list[np.ndarray]] = [[] for _ in range(3)]
    truth_color = "#111111"
    ideal_color = SPACECRAFT_COLORS["H"]
    measured_color = SPACECRAFT_COLORS["N1"]

    for column, category in enumerate(categories):
        group = handle[f"configurations/{category}"]
        truth = group["truth"]
        ideal = group["true_estimate"]
        summary = group.get("mc_summary")
        if summary is None:
            raise ValueError(
                "curl-error decomposition requires Monte Carlo summary datasets"
            )
        indices = np.asarray(group["spacecraft_indices"], dtype=int).tolist()
        ids = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in group["spacecraft_ids"][...]
        ]
        size = _read_scalar_or_series(
            group["geometry/characteristic_size"], len(full_time)
        )[time_mask]
        chi = _read_scalar_or_series(group["geometry/chi"], len(full_time))[time_mask]
        axes[0, column].set_title(
            f"{category.replace('_', ' ')}\n{ids} / {indices}; N={len(indices)}\n"
            f"median L={np.nanmedian(size):.3g} km, Q={np.nanmedian(chi):.3g}",
            fontsize=9,
        )

        coarse_magnitude = np.asarray(
            truth["coarse_curl_magnitude"], dtype=float
        )[time_mask]
        ideal_magnitude = np.asarray(ideal["curl_magnitude"], dtype=float)[time_mask]
        measured_magnitude_q25 = np.asarray(
            summary["curl_magnitude/q25"], dtype=float
        )[time_mask]
        measured_magnitude_median = np.asarray(
            summary["curl_magnitude/median"], dtype=float
        )[time_mask]
        measured_magnitude_q75 = np.asarray(
            summary["curl_magnitude/q75"], dtype=float
        )[time_mask]
        axis = axes[0, column]
        axis.plot(
            time,
            coarse_magnitude,
            color=truth_color,
            linewidth=1.5,
            label="coarse truth",
        )
        axis.plot(
            time,
            ideal_magnitude,
            color=ideal_color,
            linewidth=1.1,
            label="ideal estimate",
        )
        axis.fill_between(
            time,
            measured_magnitude_q25,
            measured_magnitude_q75,
            color=measured_color,
            alpha=0.23,
            linewidth=0,
        )
        axis.plot(
            time,
            measured_magnitude_median,
            color=measured_color,
            linewidth=1.2,
            label="measured median/IQR",
        )
        row_values[0].extend(
            (
                coarse_magnitude,
                ideal_magnitude,
                measured_magnitude_q25,
                measured_magnitude_median,
                measured_magnitude_q75,
            )
        )

        ideal_total_error = np.asarray(ideal["curl_l2_error"], dtype=float)[time_mask]
        measured_total_q25 = np.asarray(
            summary["curl_l2_error/q25"], dtype=float
        )[time_mask]
        measured_total_median = np.asarray(
            summary["curl_l2_error/median"], dtype=float
        )[time_mask]
        measured_total_q75 = np.asarray(
            summary["curl_l2_error/q75"], dtype=float
        )[time_mask]
        axis = axes[1, column]
        axis.plot(
            time,
            ideal_total_error,
            color=ideal_color,
            linewidth=1.1,
            label=r"ideal $|C_T-C_C|$",
        )
        axis.fill_between(
            time,
            measured_total_q25,
            measured_total_q75,
            color=measured_color,
            alpha=0.23,
            linewidth=0,
        )
        axis.plot(
            time,
            measured_total_median,
            color=measured_color,
            linewidth=1.2,
            label=r"measured $|C_M-C_C|$ median/IQR",
        )
        row_values[1].extend(
            (
                ideal_total_error,
                measured_total_q25,
                measured_total_median,
                measured_total_q75,
            )
        )

        decomposition = _curl_error_decomposition(group, time_mask)
        finite_scale_error = np.asarray(
            decomposition["finite_scale_error"], dtype=float
        )
        measurement_median = np.asarray(
            decomposition["measurement_error_median"], dtype=float
        )
        axis = axes[2, column]
        axis.plot(
            time,
            finite_scale_error,
            color=ideal_color,
            linewidth=1.1,
            label=r"$|C_T-C_C|/\mathrm{median}_t|C_C|$",
        )
        axis.plot(
            time,
            measurement_median,
            color=measured_color,
            linewidth=1.2,
            label=(
                r"$\mathrm{median}_{MC}|C_M-C_T|/"
                r"\mathrm{median}_t|C_C|$"
            ),
        )
        row_values[2].extend((finite_scale_error, measurement_median))

        for row, ylabel in enumerate(
            (
                r"$|\nabla\times B|$ (nT/km)",
                r"$|\Delta(\nabla\times B)|$ (nT/km)",
                "Normalized curl-vector error",
            )
        ):
            axes[row, column].set_ylabel(ylabel)
            axes[row, column].grid(alpha=0.22)
            if column == 0:
                axes[row, column].legend(fontsize=7, frameon=False)
        axes[-1, column].set_xlabel("Time (s)")
        axes[-1, column].set_xlim(
            left=start_time_s if start_time_s is not None else time[0],
            right=stop_time_s if stop_time_s is not None else time[-1],
        )

    for row, values in enumerate(row_values):
        finite_parts = [
            value[np.isfinite(value)] for value in values if np.any(np.isfinite(value))
        ]
        finite = np.concatenate(finite_parts) if finite_parts else np.asarray([])
        if not finite.size:
            continue
        low, high = np.nanmin(finite), np.nanmax(finite)
        if not np.isfinite(low) or not np.isfinite(high):
            continue
        if high <= low:
            high = low + max(abs(low) * 0.05, 1.0)
        padding = 0.05 * (high - low)
        lower = max(0.0, low - padding)
        upper = high + padding
        for axis in axes[row]:
            axis.set_ylim(lower, upper)

    figure.suptitle("HelioSwarm Magnetic-Gradient Curl Error Decomposition")
    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_magnetic_gradient_product(
    product_path: str | Path,
    output_path: str | Path,
    *,
    layout: str = "standard",
    row4: str = "divergence",
    ratio_scale: str = "auto",
    start_time_s: float | None = None,
    stop_time_s: float | None = None,
    dpi: int = 300,
) -> Path:
    """Plot a saved product without accessing observations or simulation data."""

    if layout not in {"standard", "error_decomposition"}:
        raise ValueError("layout must be standard or error_decomposition")
    row4_aliases = {
        "divergence": "divergence",
        "affine_rms_residual": "affine_rms_residual",
        "condition_number": "condition_number",
        "chi": "chi",
        "d_ep": "d_ep",
    }
    if row4 not in row4_aliases:
        raise ValueError(f"row4 must be one of {sorted(row4_aliases)}")
    if ratio_scale not in {"auto", "linear", "log"}:
        raise ValueError("ratio_scale must be auto, linear, or log")
    if start_time_s is not None and not np.isfinite(start_time_s):
        raise ValueError("start_time_s must be finite")
    if stop_time_s is not None and not np.isfinite(stop_time_s):
        raise ValueError("stop_time_s must be finite")
    if (
        start_time_s is not None
        and stop_time_s is not None
        and stop_time_s <= start_time_s
    ):
        raise ValueError("stop_time_s must be greater than start_time_s")
    with h5py.File(product_path, "r") as handle:
        full_time = np.asarray(handle["time/analysis_time"], dtype=float)
        time_mask = np.ones(full_time.shape, dtype=bool)
        if start_time_s is not None:
            time_mask &= full_time >= start_time_s
        if stop_time_s is not None:
            time_mask &= full_time <= stop_time_s
        if not np.any(time_mask):
            available = (
                f"{np.nanmin(full_time):g}–{np.nanmax(full_time):g} s"
                if full_time.size
                else "empty"
            )
            raise ValueError(
                "requested time range contains no analysis samples; "
                f"available range is {available}"
            )
        time = full_time[time_mask]
        categories = [name for name in CONFIGURATION_ORDER if name in handle["configurations"]]
        if not categories:
            raise ValueError("Gradient product contains no selected configurations")
        if layout == "error_decomposition":
            return _plot_curl_error_decomposition(
                handle,
                categories,
                full_time,
                time_mask,
                output_path,
                start_time_s=start_time_s,
                stop_time_s=stop_time_s,
                dpi=dpi,
            )
        figure, axes = plt.subplots(
            5,
            len(categories),
            figsize=(4.4 * len(categories), 15.0),
            squeeze=False,
            sharex=True,
        )
        row_values: list[list[np.ndarray]] = [[] for _ in range(5)]
        # Values used to choose display limits.  In particular, do not let
        # near-zero coarse-truth divergence/curl ratios compress the estimates.
        row_scale_values: list[list[np.ndarray]] = [[] for _ in range(5)]
        truth_color = "#111111"
        true_color = SPACECRAFT_COLORS["H"]
        measured_color = SPACECRAFT_COLORS["N1"]
        for column, category in enumerate(categories):
            group = handle[f"configurations/{category}"]
            truth = group["truth"]
            true = group["true_estimate"]
            summary = group.get("mc_summary")
            indices = np.asarray(group["spacecraft_indices"], dtype=int).tolist()
            ids = [
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in group["spacecraft_ids"][...]
            ]
            size = _read_scalar_or_series(
                group["geometry/characteristic_size"], len(full_time)
            )[time_mask]
            chi = _read_scalar_or_series(
                group["geometry/chi"], len(full_time)
            )[time_mask]
            axes[0, column].set_title(
                f"{category.replace('_', ' ')}\n{ids} / {indices}; N={len(indices)}\n"
                f"median L={np.nanmedian(size):.3g} km, Q={np.nanmedian(chi):.3g}",
                fontsize=9,
            )
            specifications = (
                ("curl_magnitude", "coarse_curl_magnitude", r"$|\nabla\times B|$ (nT/km)"),
                ("curl_l2_error", None, r"$|\Delta(\nabla\times B)|$ (nT/km)"),
                (
                    "divergence_curl_ratio",
                    "coarse_divergence_curl_ratio",
                    r"$|\nabla\cdot B|/|\nabla\times B|$",
                ),
                (
                    row4_aliases[row4],
                    "coarse_divergence" if row4 == "divergence" else None,
                    _row4_label(row4),
                ),
                ("curl", "coarse_curl", r"$\nabla\times B$ (nT/km)"),
            )
            for row, (quantity, truth_quantity, ylabel) in enumerate(specifications):
                axis = axes[row, column]
                is_vector = quantity == "curl"
                component_colors = ("#d1495b", "#2a9d60", "#3777bb")
                if truth_quantity is not None:
                    values = np.asarray(truth[truth_quantity], dtype=float)[time_mask]
                    if is_vector:
                        for component, color in enumerate(component_colors):
                            axis.plot(
                                time,
                                values[:, component],
                                color=color,
                                linewidth=1.2,
                                linestyle=":",
                                label=f"coarse {('x', 'y', 'z')[component]}",
                            )
                    else:
                        axis.plot(
                            time,
                            values,
                            color=truth_color,
                            linewidth=1.5,
                            label="coarse truth",
                        )
                    row_values[row].append(values)
                if row == 3 and row4 in {"chi", "d_ep"}:
                    values = _read_scalar_or_series(
                        group[f"geometry/{row4}"], len(full_time)
                    )[time_mask]
                else:
                    values = np.asarray(true[quantity], dtype=float)[time_mask]
                if is_vector:
                    for component, color in enumerate(component_colors):
                        axis.plot(
                            time,
                            values[:, component],
                            color=color,
                            linewidth=1.0,
                            linestyle="--",
                            label=f"estimate {('x', 'y', 'z')[component]}",
                        )
                else:
                    axis.plot(
                        time,
                        values,
                        color=true_color,
                        linewidth=1.1,
                        label="true estimate",
                    )
                row_values[row].append(values)
                row_scale_values[row].append(values)
                if summary is not None and quantity in summary:
                    q25 = np.asarray(summary[f"{quantity}/q25"], dtype=float)[time_mask]
                    median = np.asarray(
                        summary[f"{quantity}/median"], dtype=float
                    )[time_mask]
                    q75 = np.asarray(summary[f"{quantity}/q75"], dtype=float)[time_mask]
                    if is_vector:
                        for component, color in enumerate(component_colors):
                            axis.fill_between(
                                time,
                                q25[:, component],
                                q75[:, component],
                                color=color,
                                alpha=0.12,
                                linewidth=0,
                            )
                            axis.plot(
                                time,
                                median[:, component],
                                color=color,
                                linewidth=1.15,
                                label=f"measured {('x', 'y', 'z')[component]}",
                            )
                    else:
                        axis.fill_between(
                            time,
                            q25,
                            q75,
                            color=measured_color,
                            alpha=0.23,
                            linewidth=0,
                        )
                        axis.plot(
                            time,
                            median,
                            color=measured_color,
                            linewidth=1.2,
                            label="measured median/IQR",
                        )
                    row_values[row].extend((q25, median, q75))
                    row_scale_values[row].extend((q25, median, q75))
                axis.set_ylabel(ylabel)
                axis.grid(alpha=0.22)
                if column == 0 and row == 0:
                    axis.legend(fontsize=8, frameon=False)
                elif column == 0 and is_vector:
                    axis.legend(fontsize=6, frameon=False, ncol=3)
            axes[-1, column].set_xlabel("Time (s)")
            axes[-1, column].set_xlim(
                left=start_time_s if start_time_s is not None else time[0],
                right=stop_time_s if stop_time_s is not None else time[-1],
            )
        for row in range(5):
            values_for_limits = row_scale_values[row] or row_values[row]
            finite_parts = [
                value[np.isfinite(value)]
                for value in values_for_limits
                if np.any(np.isfinite(value))
            ]
            finite = np.concatenate(finite_parts) if finite_parts else np.asarray([])
            if not finite.size:
                continue
            if row == 2:
                positive = finite[finite > 0.0]
                use_log = ratio_scale == "log" or (
                    ratio_scale == "auto"
                    and positive.size > 1
                    and np.nanpercentile(positive, 99) / np.nanpercentile(positive, 1) >= 100.0
                )
                if use_log:
                    for axis in axes[row]:
                        axis.set_yscale("log")
                    finite = positive
            elif row == 3 and row4 in {"affine_rms_residual", "condition_number"}:
                positive = finite[finite > 0.0]
                if positive.size:
                    for axis in axes[row]:
                        axis.set_yscale("log")
                    finite = positive
            is_log = axes[row, 0].get_yscale() == "log"
            limit_values = np.log10(finite) if is_log else finite
            low, high = np.nanpercentile(limit_values, (0.5, 99.5))
            if np.isfinite(low) and np.isfinite(high) and high > low:
                padding = 0.05 * (high - low)
                lower, upper = low - padding, high + padding
                if is_log:
                    lower, upper = 10.0**lower, 10.0**upper
                for axis in axes[row]:
                    axis.set_ylim(lower, upper)
        figure.suptitle("HelioSwarm Multi-Spacecraft Magnetic-Gradient Diagnostics")
        figure.tight_layout()
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
    return output


def _row4_label(row4: str) -> str:
    return {
        "divergence": r"$\nabla\cdot B$ (nT/km)",
        "affine_rms_residual": "Affine RMS residual (nT)",
        "condition_number": "Condition number",
        "chi": r"$Q=\sqrt{E^2+P^2}$",
        "d_ep": r"$d_{EP}$ (unavailable)",
    }[row4]


@register_analysis
class MagneticGradientAnalysis(AnalysisMethod):
    """Registry adapter for the calculation-only magnetic-gradient product."""

    name = "magnetic_gradient"
    config_section = "magnetic_gradient"

    def run(self, input_path: str | Path) -> dict[str, Any]:
        return compute_magnetic_gradient_analysis(input_path, self.config)

    def write_result(self, result: dict[str, Any], output_path: str | Path) -> Path:
        return write_magnetic_gradient_product(result, output_path)
