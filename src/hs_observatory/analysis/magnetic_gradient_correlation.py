"""Correlation diagnostics for saved multi-spacecraft magnetic gradients.

The Pearson coefficient is named ``pearson_r`` throughout this module to
distinguish it from the volumetric tensor ``R_jk`` used by the gradient
estimator. Gradient arrays follow
``gradient[field_component, derivative_direction]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import warnings

import h5py
import matplotlib
import numpy as np
import yaml

from hs_observatory import __version__
from hs_observatory.analysis.base import AnalysisMethod
from hs_observatory.analysis.registry import register_analysis

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402


CORRELATION_SELECTION_ORDER = (
    "most_regular_n4",
    "second_regular_n4",
    "most_regular_n_gt_4",
    "all_nine",
)
CORRELATION_SELECTION_TITLES = {
    "most_regular_n4": "Most regular N=4",
    "second_regular_n4": "Second-most regular N=4",
    "most_regular_n_gt_4": "Most regular N>4",
    "all_nine": "All N=9",
}
GRADIENT_INDEX_CONVENTION = "gradient[field_component, derivative_direction]"
INVALID_REASONS = (
    "valid",
    "insufficient_samples",
    "reference_near_zero_variance",
    "estimate_near_zero_variance",
)


@dataclass(frozen=True)
class CorrelationResult:
    """Scalar paired-series correlation and amplitude diagnostics."""

    pearson_r: float
    valid_count: int
    valid_fraction: float
    reference_standard_deviation: float
    estimate_standard_deviation: float
    valid: bool
    reason: str
    normalized_rmse: float
    regression_slope: float
    regression_intercept: float


@dataclass(frozen=True)
class ComponentCorrelationResult:
    """Component-wise correlation results with a shared component shape."""

    pearson_r: np.ndarray
    valid_count: np.ndarray
    valid_fraction: np.ndarray
    reference_standard_deviation: np.ndarray
    estimate_standard_deviation: np.ndarray
    valid: np.ndarray
    invalid_reason: np.ndarray
    normalized_rmse: np.ndarray
    regression_slope: np.ndarray
    regression_intercept: np.ndarray


def pearson_correlation_1d(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    minimum_valid_samples: int = 30,
    minimum_standard_deviation: float = 0.0,
) -> CorrelationResult:
    """Calculate a Pearson coefficient for paired finite samples."""

    x = np.asarray(reference, dtype=float)
    y = np.asarray(estimate, dtype=float)
    if x.ndim != 1 or y.shape != x.shape:
        raise ValueError("reference and estimate must be one-dimensional and equal length")
    if minimum_valid_samples < 2:
        raise ValueError("minimum_valid_samples must be at least 2")
    if minimum_standard_deviation < 0.0:
        raise ValueError("minimum_standard_deviation must be nonnegative")
    use = np.isfinite(x) & np.isfinite(y)
    if valid is not None:
        validity = np.asarray(valid, dtype=bool)
        if validity.shape != x.shape:
            raise ValueError("valid mask must match the input series")
        use &= validity
    count = int(np.count_nonzero(use))
    fraction = count / float(x.size) if x.size else 0.0
    if count < minimum_valid_samples:
        return CorrelationResult(
            np.nan,
            count,
            fraction,
            np.nan,
            np.nan,
            False,
            "insufficient_samples",
            np.nan,
            np.nan,
            np.nan,
        )

    x_used = x[use]
    y_used = y[use]
    x_mean = float(np.mean(x_used))
    y_mean = float(np.mean(y_used))
    x_centered = x_used - x_mean
    y_centered = y_used - y_mean
    x_std = float(np.sqrt(np.mean(x_centered**2)))
    y_std = float(np.sqrt(np.mean(y_centered**2)))
    if x_std <= minimum_standard_deviation:
        reason = "reference_near_zero_variance"
    elif y_std <= minimum_standard_deviation:
        reason = "estimate_near_zero_variance"
    else:
        reason = "valid"
    if reason != "valid":
        return CorrelationResult(
            np.nan,
            count,
            fraction,
            x_std,
            y_std,
            False,
            reason,
            np.nan,
            np.nan,
            np.nan,
        )

    denominator = np.sqrt(float(np.sum(x_centered**2)) * float(np.sum(y_centered**2)))
    pearson_r = float(np.sum(x_centered * y_centered) / denominator)
    pearson_r = float(np.clip(pearson_r, -1.0, 1.0))
    normalized_rmse = float(np.sqrt(np.mean((y_used - x_used) ** 2)) / x_std)
    regression_slope = float(np.sum(x_centered * y_centered) / np.sum(x_centered**2))
    regression_intercept = y_mean - regression_slope * x_mean
    return CorrelationResult(
        pearson_r,
        count,
        fraction,
        x_std,
        y_std,
        True,
        reason,
        normalized_rmse,
        regression_slope,
        regression_intercept,
    )


def _component_correlation(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    valid: np.ndarray | None,
    minimum_valid_samples: int,
    minimum_standard_deviation: float,
) -> ComponentCorrelationResult:
    reference_array = np.asarray(reference, dtype=float)
    estimate_array = np.asarray(estimate, dtype=float)
    if reference_array.shape != estimate_array.shape or reference_array.ndim < 2:
        raise ValueError("reference and estimate must have equal (time, components...) shapes")
    component_shape = reference_array.shape[1:]
    output_float = {
        name: np.full(component_shape, np.nan, dtype=float)
        for name in (
            "pearson_r",
            "valid_fraction",
            "reference_standard_deviation",
            "estimate_standard_deviation",
            "normalized_rmse",
            "regression_slope",
            "regression_intercept",
        )
    }
    valid_count = np.zeros(component_shape, dtype=np.int64)
    valid_output = np.zeros(component_shape, dtype=bool)
    invalid_reason = np.empty(component_shape, dtype=object)
    for component in np.ndindex(component_shape):
        result = pearson_correlation_1d(
            reference_array[(slice(None), *component)],
            estimate_array[(slice(None), *component)],
            valid=valid,
            minimum_valid_samples=minimum_valid_samples,
            minimum_standard_deviation=minimum_standard_deviation,
        )
        output_float["pearson_r"][component] = result.pearson_r
        valid_count[component] = result.valid_count
        output_float["valid_fraction"][component] = result.valid_fraction
        output_float["reference_standard_deviation"][component] = (
            result.reference_standard_deviation
        )
        output_float["estimate_standard_deviation"][component] = result.estimate_standard_deviation
        valid_output[component] = result.valid
        invalid_reason[component] = result.reason
        output_float["normalized_rmse"][component] = result.normalized_rmse
        output_float["regression_slope"][component] = result.regression_slope
        output_float["regression_intercept"][component] = result.regression_intercept
    return ComponentCorrelationResult(
        pearson_r=output_float["pearson_r"],
        valid_count=valid_count,
        valid_fraction=output_float["valid_fraction"],
        reference_standard_deviation=output_float["reference_standard_deviation"],
        estimate_standard_deviation=output_float["estimate_standard_deviation"],
        valid=valid_output,
        invalid_reason=invalid_reason,
        normalized_rmse=output_float["normalized_rmse"],
        regression_slope=output_float["regression_slope"],
        regression_intercept=output_float["regression_intercept"],
    )


def gradient_component_correlation(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    minimum_valid_samples: int = 30,
    minimum_standard_deviation: float = 0.0,
) -> ComponentCorrelationResult:
    """Correlate corresponding ``G[field, derivative]`` time series."""

    if np.asarray(reference).shape[1:] != (3, 3):
        raise ValueError("gradient inputs must have shape (time, 3, 3)")
    return _component_correlation(
        reference,
        estimate,
        valid=valid,
        minimum_valid_samples=minimum_valid_samples,
        minimum_standard_deviation=minimum_standard_deviation,
    )


def _result_mapping(result: ComponentCorrelationResult) -> dict[str, np.ndarray]:
    return {
        "pearson_r": result.pearson_r,
        "valid_count": result.valid_count,
        "valid_fraction": result.valid_fraction,
        "reference_standard_deviation": result.reference_standard_deviation,
        "estimate_standard_deviation": result.estimate_standard_deviation,
        "valid": result.valid,
        "invalid_reason": result.invalid_reason,
        "normalized_rmse": result.normalized_rmse,
        "regression_slope": result.regression_slope,
        "regression_intercept": result.regression_intercept,
    }


def _scalar_result_mapping(result: CorrelationResult) -> dict[str, Any]:
    return {
        "pearson_r": result.pearson_r,
        "valid_count": result.valid_count,
        "valid_fraction": result.valid_fraction,
        "reference_standard_deviation": result.reference_standard_deviation,
        "estimate_standard_deviation": result.estimate_standard_deviation,
        "valid": result.valid,
        "invalid_reason": result.reason,
        "normalized_rmse": result.normalized_rmse,
        "regression_slope": result.regression_slope,
        "regression_intercept": result.regression_intercept,
    }


def _geometry_value(group: h5py.Group, name: str) -> float:
    values = np.asarray(group[f"geometry/{name}"], dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")


def _configuration_metadata(name: str, group: h5py.Group) -> dict[str, Any]:
    indices = np.asarray(group["spacecraft_indices"], dtype=np.int64)
    names = np.asarray(
        [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in group["spacecraft_ids"][...]
        ],
        dtype=object,
    )
    return {
        "source_category": name,
        "spacecraft_indices": indices,
        "spacecraft_names": names,
        "number_of_spacecraft": len(indices),
        "characteristic_size": _geometry_value(group, "characteristic_size"),
        "elongation": _geometry_value(group, "elongation"),
        "planarity": _geometry_value(group, "planarity"),
        "chi": _geometry_value(group, "chi"),
        "d_ep": _geometry_value(group, "d_ep"),
        "condition_number": _geometry_value(group, "condition_number"),
    }


def select_correlation_configurations(
    configurations: Mapping[str, Mapping[str, Any]],
    *,
    shape_metric: str = "chi",
) -> dict[str, dict[str, Any]]:
    """Select four distinct saved configurations with deterministic tie-breaking."""

    if shape_metric not in {"chi", "elongation", "planarity", "d_ep"}:
        raise ValueError("shape_metric must be chi, elongation, planarity, or d_ep")

    def key(item: tuple[str, Mapping[str, Any]]) -> tuple[Any, ...]:
        name, metadata = item
        metric = float(metadata[shape_metric])
        condition = float(metadata["condition_number"])
        size = float(metadata["characteristic_size"])
        indices = tuple(int(value) for value in metadata["spacecraft_indices"])
        return (
            metric if np.isfinite(metric) else np.inf,
            condition if np.isfinite(condition) else np.inf,
            size if np.isfinite(size) else np.inf,
            indices,
            name,
        )

    items = list(configurations.items())
    if not any(np.isfinite(float(metadata[shape_metric])) for _, metadata in items):
        raise ValueError(
            f"shape metric {shape_metric!r} is unavailable for every saved configuration"
        )
    n4 = sorted(
        (item for item in items if int(item[1]["number_of_spacecraft"]) == 4),
        key=key,
    )
    if len(n4) < 2:
        raise ValueError("correlation selection requires two saved N=4 configurations")
    intermediate_multi = sorted(
        (item for item in items if 4 < int(item[1]["number_of_spacecraft"]) < 9),
        key=key,
    )
    if not intermediate_multi:
        raise ValueError("correlation selection requires a saved configuration with 4 < N < 9")
    all_nine = sorted(
        (item for item in items if int(item[1]["number_of_spacecraft"]) == 9),
        key=key,
    )
    if not all_nine:
        raise ValueError("correlation selection requires the full N=9 configuration")
    return {
        "most_regular_n4": dict(n4[0][1]),
        "second_regular_n4": dict(n4[1][1]),
        "most_regular_n_gt_4": dict(intermediate_multi[0][1]),
        "all_nine": dict(all_nine[0][1]),
    }


def _comparison_metrics(
    reference_gradient: np.ndarray,
    estimate_gradient: np.ndarray,
    reference_curl: np.ndarray,
    estimate_curl: np.ndarray,
    reference_curl_magnitude: np.ndarray,
    estimate_curl_magnitude: np.ndarray,
    *,
    valid: np.ndarray,
    minimum_valid_samples: int,
    minimum_standard_deviation: float,
) -> dict[str, Any]:
    gradient = gradient_component_correlation(
        reference_gradient,
        estimate_gradient,
        valid=valid,
        minimum_valid_samples=minimum_valid_samples,
        minimum_standard_deviation=minimum_standard_deviation,
    )
    curl = _component_correlation(
        reference_curl,
        estimate_curl,
        valid=valid,
        minimum_valid_samples=minimum_valid_samples,
        minimum_standard_deviation=minimum_standard_deviation,
    )
    curl_magnitude = pearson_correlation_1d(
        reference_curl_magnitude,
        estimate_curl_magnitude,
        valid=valid,
        minimum_valid_samples=minimum_valid_samples,
        minimum_standard_deviation=minimum_standard_deviation,
    )
    gradient_valid = np.isfinite(gradient.pearson_r)
    curl_valid = np.isfinite(curl.pearson_r)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        summary = {
            "median_gradient_pearson_r": float(np.nanmedian(gradient.pearson_r)),
            "minimum_gradient_pearson_r": float(np.nanmin(gradient.pearson_r)),
            "median_curl_component_pearson_r": float(np.nanmedian(curl.pearson_r)),
            "valid_gradient_component_count": int(np.count_nonzero(gradient_valid)),
            "valid_curl_component_count": int(np.count_nonzero(curl_valid)),
        }
    return {
        "gradient": _result_mapping(gradient),
        "curl_component": _result_mapping(curl),
        "curl_magnitude": _scalar_result_mapping(curl_magnitude),
        "summary": summary,
    }


def _realization_comparison(
    reference_gradient: np.ndarray,
    estimate_gradient: np.ndarray,
    reference_curl: np.ndarray,
    estimate_curl: np.ndarray,
    reference_curl_magnitude: np.ndarray,
    estimate_curl_magnitude: np.ndarray,
    *,
    reference_valid: np.ndarray,
    estimate_valid: np.ndarray,
    minimum_valid_samples: int,
    minimum_standard_deviation: float,
    percentiles: tuple[float, float, float],
) -> dict[str, Any]:
    n_mc = estimate_gradient.shape[0]
    comparisons = []
    for member in range(n_mc):
        comparisons.append(
            _comparison_metrics(
                reference_gradient,
                estimate_gradient[member],
                reference_curl,
                estimate_curl[member],
                reference_curl_magnitude,
                estimate_curl_magnitude[member],
                valid=reference_valid & estimate_valid[member],
                minimum_valid_samples=minimum_valid_samples,
                minimum_standard_deviation=minimum_standard_deviation,
            )
        )

    result: dict[str, Any] = {}
    for quantity in ("gradient", "curl_component"):
        result[quantity] = {}
        for metric in (
            "pearson_r",
            "valid_count",
            "valid_fraction",
            "reference_standard_deviation",
            "estimate_standard_deviation",
            "valid",
            "invalid_reason",
            "normalized_rmse",
            "regression_slope",
            "regression_intercept",
        ):
            result[quantity][metric] = np.asarray(
                [comparison[quantity][metric] for comparison in comparisons]
            )
        pearson_r = result[quantity]["pearson_r"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lower, median, upper = np.nanpercentile(pearson_r, percentiles, axis=0)
        result[quantity]["pearson_r_lower"] = lower
        result[quantity]["pearson_r_median"] = median
        result[quantity]["pearson_r_upper"] = upper
        for percentile, values in zip(percentiles, (lower, median, upper)):
            label = f"q{int(percentile):02d}" if percentile.is_integer() else f"q{percentile:g}"
            result[quantity][f"pearson_r_{label}"] = values
        valid_realizations = np.isfinite(pearson_r)
        result[quantity]["valid_realization_count"] = np.count_nonzero(valid_realizations, axis=0)
        result[quantity]["valid_realization_fraction"] = np.mean(valid_realizations, axis=0)

    result["curl_magnitude"] = {}
    for metric in (
        "pearson_r",
        "valid_count",
        "valid_fraction",
        "reference_standard_deviation",
        "estimate_standard_deviation",
        "valid",
        "invalid_reason",
        "normalized_rmse",
        "regression_slope",
        "regression_intercept",
    ):
        result["curl_magnitude"][metric] = np.asarray(
            [comparison["curl_magnitude"][metric] for comparison in comparisons]
        )
    curl_magnitude_r = result["curl_magnitude"]["pearson_r"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lower, median, upper = np.nanpercentile(curl_magnitude_r, percentiles)
    result["curl_magnitude"].update(
        pearson_r_lower=lower,
        pearson_r_median=median,
        pearson_r_upper=upper,
        valid_realization_count=int(np.count_nonzero(np.isfinite(curl_magnitude_r))),
        valid_realization_fraction=float(np.mean(np.isfinite(curl_magnitude_r))),
    )
    for percentile, value in zip(percentiles, (lower, median, upper)):
        label = f"q{int(percentile):02d}" if percentile.is_integer() else f"q{percentile:g}"
        result["curl_magnitude"][f"pearson_r_{label}"] = value

    result["summary_by_realization"] = {
        name: np.asarray([comparison["summary"][name] for comparison in comparisons])
        for name in (
            "median_gradient_pearson_r",
            "minimum_gradient_pearson_r",
            "median_curl_component_pearson_r",
            "valid_gradient_component_count",
            "valid_curl_component_count",
        )
    }
    result["summary"] = {}
    for name in (
        "median_gradient_pearson_r",
        "minimum_gradient_pearson_r",
        "median_curl_component_pearson_r",
    ):
        values = result["summary_by_realization"][name]
        finite = np.isfinite(values)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lower, median, upper = np.nanpercentile(values, percentiles)
        result["summary"][name] = {
            "lower": lower,
            "median": median,
            "upper": upper,
            "valid_count": int(np.count_nonzero(finite)),
            "valid_fraction": float(np.mean(finite)),
        }
    return result


def compute_magnetic_gradient_correlations(
    gradient_product_path: str | Path,
    settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Calculate correlations from a saved magnetic-gradient product."""

    options = dict(settings or {})
    selection = dict(options.get("selection", {}))
    correlation = dict(options.get("correlation", {}))
    bootstrap = dict(options.get("bootstrap", {}))
    if bootstrap.get("enabled", False):
        raise NotImplementedError("moving-block bootstrap is not implemented")
    shape_metric = str(selection.get("shape_metric", "chi"))
    scale_warning = float(selection.get("max_pair_scale_ratio_warning", 2.0))
    minimum_valid_samples = int(correlation.get("minimum_valid_samples", 30))
    minimum_standard_deviation = float(correlation.get("minimum_standard_deviation", 1.0e-15))
    percentiles = tuple(
        float(value) for value in correlation.get("measured_percentiles", (5.0, 50.0, 95.0))
    )
    if len(percentiles) != 3 or not 0 <= percentiles[0] < percentiles[1] < percentiles[2] <= 100:
        raise ValueError("measured_percentiles must contain three increasing values")
    if percentiles[1] != 50.0:
        raise ValueError("the middle measured percentile must be 50")
    compute_measured_vs_coarse = bool(correlation.get("compute_measured_vs_coarse", True))

    source = Path(gradient_product_path)
    with h5py.File(source, "r") as handle:
        available_metadata = {
            name: _configuration_metadata(name, group)
            for name, group in handle["configurations"].items()
        }
        selected = select_correlation_configurations(available_metadata, shape_metric=shape_metric)
        first_size = selected["most_regular_n4"]["characteristic_size"]
        second_size = selected["second_regular_n4"]["characteristic_size"]
        pair_scale_ratio = max(first_size, second_size) / min(first_size, second_size)
        warning_messages: list[str] = []
        if pair_scale_ratio > scale_warning:
            message = (
                "The two selected N=4 configurations differ in characteristic "
                f"size by a factor {pair_scale_ratio:.3g}, exceeding "
                f"max_pair_scale_ratio_warning={scale_warning:g}; their "
                "performance is not a pure geometry comparison."
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            warning_messages.append(message)

        time = np.asarray(handle["time/analysis_time"], dtype=float)
        global_valid = np.asarray(handle["time/validity"], dtype=bool)
        configurations: dict[str, Any] = {}
        for selection_name in CORRELATION_SELECTION_ORDER:
            metadata = selected[selection_name]
            group = handle[f"configurations/{metadata['source_category']}"]
            coarse_gradient = np.asarray(group["truth/coarse_gradient"], dtype=float)
            ideal_gradient = np.asarray(group["true_estimate/gradient"], dtype=float)
            measured_gradient = np.asarray(group["measured/gradient"], dtype=float)
            coarse_curl = np.asarray(group["truth/coarse_curl"], dtype=float)
            ideal_curl = np.asarray(group["true_estimate/curl"], dtype=float)
            measured_curl = np.asarray(group["measured/curl"], dtype=float)
            coarse_curl_magnitude = np.asarray(group["truth/coarse_curl_magnitude"], dtype=float)
            ideal_curl_magnitude = np.asarray(group["true_estimate/curl_magnitude"], dtype=float)
            measured_curl_magnitude = np.asarray(group["measured/curl_magnitude"], dtype=float)
            coarse_valid = global_valid & np.asarray(group["truth/coarse_valid"], dtype=bool)
            ideal_valid = global_valid & np.asarray(group["true_estimate/valid"], dtype=bool)
            measured_valid = np.asarray(group["measured/valid"], dtype=bool)
            measured_valid &= global_valid[None, :]

            ideal_vs_coarse = _comparison_metrics(
                coarse_gradient,
                ideal_gradient,
                coarse_curl,
                ideal_curl,
                coarse_curl_magnitude,
                ideal_curl_magnitude,
                valid=coarse_valid & ideal_valid,
                minimum_valid_samples=minimum_valid_samples,
                minimum_standard_deviation=minimum_standard_deviation,
            )
            measured_vs_ideal = _realization_comparison(
                ideal_gradient,
                measured_gradient,
                ideal_curl,
                measured_curl,
                ideal_curl_magnitude,
                measured_curl_magnitude,
                reference_valid=ideal_valid,
                estimate_valid=measured_valid,
                minimum_valid_samples=minimum_valid_samples,
                minimum_standard_deviation=minimum_standard_deviation,
                percentiles=percentiles,
            )
            product: dict[str, Any] = {
                "metadata": metadata,
                "ideal_vs_coarse": ideal_vs_coarse,
                "measured_vs_ideal": measured_vs_ideal,
            }
            if compute_measured_vs_coarse:
                product["measured_vs_coarse"] = _realization_comparison(
                    coarse_gradient,
                    measured_gradient,
                    coarse_curl,
                    measured_curl,
                    coarse_curl_magnitude,
                    measured_curl_magnitude,
                    reference_valid=coarse_valid,
                    estimate_valid=measured_valid,
                    minimum_valid_samples=minimum_valid_samples,
                    minimum_standard_deviation=minimum_standard_deviation,
                    percentiles=percentiles,
                )
            configurations[selection_name] = product

        diagnostic_options = dict(options.get("diagnostic", {}))
        diagnostic_selection = str(diagnostic_options.get("configuration", "most_regular_n4"))
        if diagnostic_selection not in configurations:
            raise ValueError(f"unknown diagnostic configuration {diagnostic_selection!r}")
        component = tuple(int(value) for value in diagnostic_options.get("component", (0, 0)))
        if len(component) != 2 or any(value not in range(3) for value in component):
            raise ValueError("diagnostic component must contain two indices in [0, 2]")
        realization = int(diagnostic_options.get("realization", 0))
        metadata = configurations[diagnostic_selection]["metadata"]
        group = handle[f"configurations/{metadata['source_category']}"]
        measured_dataset = group["measured/gradient"]
        if realization < 0 or realization >= measured_dataset.shape[0]:
            raise ValueError("diagnostic realization is out of range")
        diagnostic = {
            "configuration": diagnostic_selection,
            "component": np.asarray(component, dtype=np.int8),
            "realization": realization,
            "time": time,
            "coarse": np.asarray(group["truth/coarse_gradient"])[:, component[0], component[1]],
            "ideal": np.asarray(group["true_estimate/gradient"])[:, component[0], component[1]],
            "measured": np.asarray(measured_dataset)[realization, :, component[0], component[1]],
            "coarse_valid": global_valid & np.asarray(group["truth/coarse_valid"], dtype=bool),
            "ideal_valid": global_valid & np.asarray(group["true_estimate/valid"], dtype=bool),
            "measured_valid": global_valid
            & np.asarray(group["measured/valid"][realization], dtype=bool),
        }

    return {
        "metadata": {
            "source_gradient_file": str(source.resolve()),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "code_version": __version__,
            "comparison_definitions": (
                "T,C=ideal sparse estimate versus coarse truth; "
                "M,T=each measured realization versus ideal estimate; "
                "M,C=each measured realization versus coarse truth"
            ),
            "tensor_index_convention": GRADIENT_INDEX_CONVENTION,
            "shape_metric": shape_metric,
            "measured_percentiles": np.asarray(percentiles),
            "minimum_valid_samples": minimum_valid_samples,
            "minimum_standard_deviation": minimum_standard_deviation,
            "max_pair_scale_ratio_warning": scale_warning,
            "n_time": len(time),
            "n_mc": next(iter(configurations.values()))["measured_vs_ideal"]["gradient"][
                "pearson_r"
            ].shape[0],
            "warnings": warning_messages,
        },
        "configurations": configurations,
        "diagnostic": diagnostic,
    }


def _create_dataset(
    group: h5py.Group,
    name: str,
    value: Any,
    *,
    compression: str | None,
    attrs: Mapping[str, Any] | None = None,
) -> h5py.Dataset:
    array = np.asarray(value)
    if array.dtype == object or array.dtype.kind in {"U", "S"}:
        array = np.asarray(array, dtype=h5py.string_dtype("utf-8"))
    kwargs = {"compression": compression} if compression and array.ndim else {}
    dataset = group.create_dataset(name, data=array, **kwargs)
    if attrs:
        dataset.attrs.update(attrs)
    return dataset


def _write_nested(
    group: h5py.Group,
    mapping: Mapping[str, Any],
    *,
    compression: str | None,
) -> None:
    for name, value in mapping.items():
        if isinstance(value, Mapping):
            _write_nested(group.create_group(name), value, compression=compression)
        else:
            attrs: dict[str, Any] = {}
            if name in {"pearson_r", "pearson_r_lower", "pearson_r_median", "pearson_r_upper"}:
                attrs = {
                    "definition": "Pearson product-moment correlation coefficient",
                    "units": "1",
                }
            elif name == "normalized_rmse":
                attrs = {"definition": "RMSE(estimate-reference) / std(reference)", "units": "1"}
            elif name == "regression_slope":
                attrs = {
                    "definition": "slope b in estimate = intercept + b * reference",
                    "units": "1",
                }
            elif name == "regression_intercept":
                attrs = {
                    "definition": "intercept a in estimate = a + slope * reference",
                    "units": "nT/km",
                }
            _create_dataset(group, name, value, compression=compression, attrs=attrs)


def write_magnetic_gradient_correlation_product(
    result: Mapping[str, Any],
    output_path: str | Path,
    *,
    compression: str | None = "gzip",
) -> Path:
    """Write a standalone correlation product."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        metadata_group = handle.create_group("metadata")
        for name, value in result["metadata"].items():
            if name == "measured_percentiles":
                _create_dataset(metadata_group, name, value, compression=None, attrs={"units": "%"})
            elif name == "warnings":
                _create_dataset(metadata_group, name, value, compression=None)
            else:
                metadata_group.attrs[name] = value
        metadata_group.attrs["pearson_symbol"] = (
            "Pearson correlation R; distinct from volumetric tensor R_jk"
        )
        metadata_group.attrs["gradient_units"] = "nT/km"

        configurations_group = handle.create_group("configurations")
        for selection_name in CORRELATION_SELECTION_ORDER:
            product = result["configurations"][selection_name]
            group = configurations_group.create_group(selection_name)
            metadata = group.create_group("metadata")
            for name, value in product["metadata"].items():
                if name in {"spacecraft_indices", "spacecraft_names"}:
                    _create_dataset(metadata, name, value, compression=None)
                else:
                    metadata.attrs[name] = value
            metadata.attrs["characteristic_size_units"] = "km"
            for comparison in (
                "ideal_vs_coarse",
                "measured_vs_ideal",
                "measured_vs_coarse",
            ):
                if comparison in product:
                    _write_nested(
                        group.create_group(comparison),
                        product[comparison],
                        compression=compression,
                    )

        diagnostic = handle.create_group("diagnostic")
        diagnostic.attrs["configuration"] = result["diagnostic"]["configuration"]
        diagnostic.attrs["realization"] = result["diagnostic"]["realization"]
        diagnostic.attrs["tensor_index_convention"] = GRADIENT_INDEX_CONVENTION
        for name, value in result["diagnostic"].items():
            if name in {"configuration", "realization"}:
                continue
            attrs = {"units": "s"} if name == "time" else {}
            if name in {"coarse", "ideal", "measured"}:
                attrs = {"units": "nT/km"}
            _create_dataset(diagnostic, name, value, compression=compression, attrs=attrs)
    return output


def _format_statistic(value: float) -> str:
    return "NaN" if not np.isfinite(value) else f"{value:.4f}"


def print_correlation_summary(result: Mapping[str, Any]) -> None:
    """Print selection metadata and scalar correlation summaries."""

    print("Selected gradient-correlation configurations:")
    for selection_name in CORRELATION_SELECTION_ORDER:
        product = result["configurations"][selection_name]
        metadata = product["metadata"]
        print(f"  {CORRELATION_SELECTION_TITLES[selection_name]}:")
        print(
            "    spacecraft="
            f"{metadata['spacecraft_names'].tolist()} indices="
            f"{metadata['spacecraft_indices'].tolist()}"
        )
        print(
            f"    N={metadata['number_of_spacecraft']} "
            f"L={metadata['characteristic_size']:.6g} km "
            f"E={metadata['elongation']:.6g} P={metadata['planarity']:.6g} "
            f"chi={metadata['chi']:.6g} d_EP={metadata['d_ep']:.6g} "
            f"condition={metadata['condition_number']:.6g}"
        )
    metadata = result["metadata"]
    invalid_ideal = 0
    invalid_measured = 0
    zero_variance = 0
    for product in result["configurations"].values():
        invalid_ideal += int(np.count_nonzero(~product["ideal_vs_coarse"]["gradient"]["valid"]))
        invalid_measured += int(
            np.count_nonzero(~product["measured_vs_ideal"]["gradient"]["valid"])
        )
        reasons = product["measured_vs_ideal"]["gradient"]["invalid_reason"]
        zero_variance += int(
            np.count_nonzero(
                (reasons == "reference_near_zero_variance")
                | (reasons == "estimate_near_zero_variance")
            )
        )
    print("Correlation calculation:")
    print(
        f"  time samples={metadata['n_time']} MC realizations={metadata['n_mc']} "
        f"minimum valid samples={metadata['minimum_valid_samples']}"
    )
    print(
        f"  invalid ideal/coarse tensor cells={invalid_ideal} "
        f"invalid measured/ideal realization-cells={invalid_measured} "
        f"zero-variance cells={zero_variance}"
    )
    for selection_name in CORRELATION_SELECTION_ORDER:
        product = result["configurations"][selection_name]
        ideal = product["ideal_vs_coarse"]["summary"]
        measured = product["measured_vs_ideal"]["summary"]
        print(f"{CORRELATION_SELECTION_TITLES[selection_name]}:")
        print(
            "  ideal vs coarse: median gradient R="
            f"{_format_statistic(ideal['median_gradient_pearson_r'])}, "
            f"minimum gradient R={_format_statistic(ideal['minimum_gradient_pearson_r'])}, "
            "median curl-component R="
            f"{_format_statistic(ideal['median_curl_component_pearson_r'])}"
        )
        labels = (
            ("median gradient R", "median_gradient_pearson_r"),
            ("minimum gradient R", "minimum_gradient_pearson_r"),
            ("median curl-component R", "median_curl_component_pearson_r"),
        )
        print("  measured vs ideal:")
        for label, key in labels:
            summary = measured[key]
            print(
                f"    {label}={_format_statistic(summary['median'])} "
                f"[{_format_statistic(summary['lower'])}, "
                f"{_format_statistic(summary['upper'])}]"
            )


def _decode_strings(dataset: h5py.Dataset) -> list[str]:
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in dataset[...]]


def _configuration_title(group: h5py.Group, selection_name: str) -> str:
    metadata = group["metadata"].attrs
    return (
        f"{CORRELATION_SELECTION_TITLES[selection_name]}\n"
        f"N={int(metadata['number_of_spacecraft'])}, "
        f"L={float(metadata['characteristic_size']):.3g} km\n"
        rf"$\chi$={float(metadata['chi']):.3g}, "
        rf"$\kappa_R$={float(metadata['condition_number']):.3g}"
    )


def _draw_heatmap_grid(
    figure: plt.Figure,
    axes: np.ndarray,
    handle: h5py.File,
    *,
    color_limits: tuple[float, float] = (-1.0, 1.0),
    annotate_values: bool = True,
    colorbar_rectangle: tuple[float, float, float, float] = (
        0.925,
        0.16,
        0.018,
        0.68,
    ),
) -> None:
    if tuple(float(value) for value in color_limits) != (-1.0, 1.0):
        raise ValueError("Pearson correlation heat maps require color_limits [-1, 1]")
    colormap = plt.get_cmap("coolwarm_r").copy()
    colormap.set_bad("#dedede")
    normalization = Normalize(vmin=-1.0, vmax=1.0)
    image = None
    for column, selection_name in enumerate(CORRELATION_SELECTION_ORDER):
        group = handle[f"configurations/{selection_name}"]
        matrices = (
            np.asarray(group["ideal_vs_coarse/gradient/pearson_r"], dtype=float),
            np.asarray(group["measured_vs_ideal/gradient/pearson_r_median"], dtype=float),
        )
        for row, matrix in enumerate(matrices):
            axis = axes[row, column]
            image = axis.imshow(
                np.ma.masked_invalid(matrix),
                cmap=colormap,
                norm=normalization,
                origin="upper",
                aspect="equal",
            )
            axis.set_xticks(range(3), (r"$\partial_x$", r"$\partial_y$", r"$\partial_z$"))
            axis.set_yticks(range(3), (r"$B_x$", r"$B_y$", r"$B_z$"))
            if column:
                axis.tick_params(labelleft=False)
            if row == 0:
                axis.set_title(_configuration_title(group, selection_name), fontsize=9)
            if annotate_values:
                for field_component in range(3):
                    for derivative_direction in range(3):
                        value = matrix[field_component, derivative_direction]
                        text = "--" if not np.isfinite(value) else f"{value:.2f}"
                        color = "white" if np.isfinite(value) and abs(value) > 0.55 else "black"
                        axis.text(
                            derivative_direction,
                            field_component,
                            text,
                            ha="center",
                            va="center",
                            fontsize=8,
                            color=color,
                        )
    axes[0, 0].set_ylabel("Ideal vs coarse\nField component")
    axes[1, 0].set_ylabel("Measured vs ideal\nField component")
    for axis in axes[1]:
        axis.set_xlabel("Derivative direction")
    if image is not None:
        colorbar_axis = figure.add_axes(colorbar_rectangle)
        colorbar = figure.colorbar(image, cax=colorbar_axis)
        colorbar.set_label(r"Pearson correlation, $\mathcal{R}$")


def plot_magnetic_gradient_correlation_heatmaps(
    product_path: str | Path,
    output_path: str | Path,
    *,
    color_limits: tuple[float, float] = (-1.0, 1.0),
    annotate_values: bool = True,
    dpi: int = 300,
) -> Path:
    """Plot the 2x4 tensor-component Pearson-correlation heat maps."""

    with h5py.File(product_path, "r") as handle:
        figure, axes = plt.subplots(2, 4, figsize=(13.8, 7.0), squeeze=False)
        _draw_heatmap_grid(
            figure,
            axes,
            handle,
            color_limits=color_limits,
            annotate_values=annotate_values,
        )
        figure.suptitle(
            "Magnetic-Gradient Component Recovery\n"
            r"$G_{jk}=\partial B_j/\partial x_k$"
        )
        figure.subplots_adjust(
            left=0.07, right=0.91, bottom=0.09, top=0.84, wspace=0.18, hspace=0.28
        )
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
    return output


def plot_magnetic_gradient_measured_vs_ideal_heatmaps(
    product_path: str | Path,
    output_path: str | Path,
    *,
    dpi: int = 300,
) -> Path:
    """Plot measured-versus-ideal heat maps and their MC summary."""

    with h5py.File(product_path, "r") as handle:
        figure = plt.figure(figsize=(13.8, 7.2))
        grid = figure.add_gridspec(2, 4, height_ratios=(1.0, 0.72))
        axes = np.asarray([[figure.add_subplot(grid[0, column]) for column in range(4)]])
        summary_axis = figure.add_subplot(grid[1, :])
        colormap = plt.get_cmap("coolwarm_r").copy()
        colormap.set_bad("#dedede")
        normalization = Normalize(vmin=-1.0, vmax=1.0)
        image = None
        for column, selection_name in enumerate(CORRELATION_SELECTION_ORDER):
            group = handle[f"configurations/{selection_name}"]
            matrix = np.asarray(group["measured_vs_ideal/gradient/pearson_r_median"], dtype=float)
            axis = axes[0, column]
            image = axis.imshow(
                np.ma.masked_invalid(matrix),
                cmap=colormap,
                norm=normalization,
                origin="upper",
                aspect="equal",
            )
            axis.set_title(_configuration_title(group, selection_name), fontsize=9)
            axis.set_xticks(range(3), (r"$\partial_x$", r"$\partial_y$", r"$\partial_z$"))
            axis.set_yticks(range(3), (r"$B_x$", r"$B_y$", r"$B_z$"))
            if column:
                axis.tick_params(labelleft=False)
            else:
                axis.set_ylabel("Field component")
            axis.set_xlabel("Derivative direction")
            for field_component in range(3):
                for derivative_direction in range(3):
                    value = matrix[field_component, derivative_direction]
                    text = "--" if not np.isfinite(value) else f"{value:.3f}"
                    color = "white" if np.isfinite(value) and abs(value) > 0.55 else "black"
                    axis.text(
                        derivative_direction,
                        field_component,
                        text,
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=color,
                    )
        if image is not None:
            colorbar_axis = figure.add_axes((0.925, 0.53, 0.018, 0.34))
            colorbar = figure.colorbar(image, cax=colorbar_axis)
            colorbar.set_label(r"Pearson correlation, $\mathcal{R}$")
        _draw_summary_panel(summary_axis, handle, include_ideal=False, y_limits=(0.7, 1.0))
        figure.suptitle(
            "Measured-versus-Ideal Magnetic-Gradient Component Recovery\n"
            r"$\mathrm{median}_m\left(\mathcal{R}^{M,T,(m)}_{jk}\right)$, "
            r"$G_{jk}=\partial B_j/\partial x_k$"
        )
        figure.subplots_adjust(
            left=0.07,
            right=0.91,
            bottom=0.08,
            top=0.80,
            wspace=0.18,
            hspace=0.58,
        )
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
    return output


SUMMARY_METRICS = (
    ("median_gradient_pearson_r", r"median $G_{jk}$", "o", "#2d6a9f"),
    ("minimum_gradient_pearson_r", r"minimum $G_{jk}$", "s", "#d05a8c"),
    ("median_curl_component_pearson_r", r"median curl component", "^", "#5c9f5c"),
)


def _draw_summary_panel(
    axis: plt.Axes,
    handle: h5py.File,
    *,
    include_ideal: bool = True,
    y_limits: tuple[float, float] = (-1.05, 1.05),
) -> None:
    positions = np.arange(len(CORRELATION_SELECTION_ORDER), dtype=float)
    for metric_index, (name, label, marker, color) in enumerate(SUMMARY_METRICS):
        offset = (metric_index - 1) * 0.12
        ideal_values = []
        measured_median = []
        measured_lower = []
        measured_upper = []
        for selection_name in CORRELATION_SELECTION_ORDER:
            group = handle[f"configurations/{selection_name}"]
            ideal_values.append(float(group[f"ideal_vs_coarse/summary/{name}"][()]))
            summary = group[f"measured_vs_ideal/summary/{name}"]
            measured_median.append(float(summary["median"][()]))
            measured_lower.append(float(summary["lower"][()]))
            measured_upper.append(float(summary["upper"][()]))
        ideal_array = np.asarray(ideal_values)
        median_array = np.asarray(measured_median)
        if include_ideal:
            axis.plot(
                positions + offset - 0.035,
                ideal_array,
                linestyle="none",
                marker=marker,
                markersize=6,
                markerfacecolor="none",
                markeredgecolor=color,
                markeredgewidth=1.3,
                label=f"{label}: T,C",
            )
        axis.errorbar(
            positions + offset + (0.035 if include_ideal else 0.0),
            median_array,
            yerr=np.vstack(
                (
                    median_array - np.asarray(measured_lower),
                    np.asarray(measured_upper) - median_array,
                )
            ),
            linestyle="none",
            marker=marker,
            markersize=5,
            color=color,
            capsize=3,
            label=f"{label}: M,T",
        )
    axis.axhline(0.0, color="#444444", linewidth=1.0)
    axis.axhline(1.0, color="#999999", linewidth=0.8, linestyle=":")
    axis.set_ylim(*y_limits)
    axis.set_ylabel(r"Pearson correlation, $\mathcal{R}$")
    axis.set_xticks(
        positions,
        [CORRELATION_SELECTION_TITLES[name] for name in CORRELATION_SELECTION_ORDER],
    )
    axis.grid(alpha=0.2, axis="y")
    axis.legend(frameon=False, fontsize=7, ncol=3, loc="lower left")


def plot_magnetic_gradient_correlation_summary(
    product_path: str | Path,
    output_path: str | Path,
    *,
    dpi: int = 300,
) -> Path:
    """Plot scalar correlation summaries for the four selected configurations."""

    with h5py.File(product_path, "r") as handle:
        figure, axis = plt.subplots(figsize=(11.5, 4.5))
        _draw_summary_panel(axis, handle)
        figure.suptitle("Magnetic-Gradient Correlation Summary")
        figure.tight_layout()
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
    return output


def plot_magnetic_gradient_correlation_combined(
    product_path: str | Path,
    output_path: str | Path,
    *,
    color_limits: tuple[float, float] = (-1.0, 1.0),
    annotate_values: bool = True,
    dpi: int = 300,
) -> Path:
    """Plot the heat maps and scalar summary in one publication layout."""

    with h5py.File(product_path, "r") as handle:
        figure = plt.figure(figsize=(14.0, 10.2))
        grid = figure.add_gridspec(3, 4, height_ratios=(1.0, 1.0, 0.85))
        axes = np.asarray(
            [[figure.add_subplot(grid[row, column]) for column in range(4)] for row in range(2)]
        )
        summary_axis = figure.add_subplot(grid[2, :])
        _draw_heatmap_grid(
            figure,
            axes,
            handle,
            color_limits=color_limits,
            annotate_values=annotate_values,
            colorbar_rectangle=(0.925, 0.42, 0.018, 0.43),
        )
        _draw_summary_panel(summary_axis, handle)
        figure.suptitle(
            "Magnetic-Gradient Component Recovery\n"
            r"$G_{jk}=\partial B_j/\partial x_k$",
            y=0.98,
        )
        figure.subplots_adjust(
            left=0.07,
            right=0.91,
            bottom=0.07,
            top=0.88,
            wspace=0.22,
            hspace=0.48,
        )
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
    return output


def _scatter_with_regression(
    axis: plt.Axes,
    reference: np.ndarray,
    estimate: np.ndarray,
    valid: np.ndarray,
    *,
    pearson_r: float,
    slope: float,
    intercept: float,
    reference_label: str,
    estimate_label: str,
    color: str,
) -> None:
    use = valid & np.isfinite(reference) & np.isfinite(estimate)
    axis.scatter(reference[use], estimate[use], s=6, alpha=0.22, color=color)
    if np.any(use):
        low = float(min(np.min(reference[use]), np.min(estimate[use])))
        high = float(max(np.max(reference[use]), np.max(estimate[use])))
        axis.plot((low, high), (low, high), color="#777777", linestyle=":", linewidth=1)
        if np.isfinite(slope) and np.isfinite(intercept):
            x_line = np.asarray((low, high))
            axis.plot(
                x_line,
                intercept + slope * x_line,
                color="black",
                linewidth=1.2,
            )
    axis.set_xlabel(f"{reference_label} (nT/km)")
    axis.set_ylabel(f"{estimate_label} (nT/km)")
    axis.text(
        0.03,
        0.97,
        rf"$\mathcal{{R}}={pearson_r:.3f}$"
        "\n"
        rf"$b={slope:.3f}$, $a={intercept:.2e}$",
        transform=axis.transAxes,
        va="top",
        fontsize=9,
    )
    axis.grid(alpha=0.2)


def plot_magnetic_gradient_correlation_diagnostic(
    product_path: str | Path,
    output_path: str | Path,
    *,
    dpi: int = 200,
) -> Path:
    """Plot stored component time series and regression diagnostics."""

    with h5py.File(product_path, "r") as handle:
        diagnostic = handle["diagnostic"]
        selection_name = str(diagnostic.attrs["configuration"])
        realization = int(diagnostic.attrs["realization"])
        component = np.asarray(diagnostic["component"], dtype=int)
        time = np.asarray(diagnostic["time"], dtype=float)
        coarse = np.asarray(diagnostic["coarse"], dtype=float)
        ideal = np.asarray(diagnostic["ideal"], dtype=float)
        measured = np.asarray(diagnostic["measured"], dtype=float)
        coarse_valid = np.asarray(diagnostic["coarse_valid"], dtype=bool)
        ideal_valid = np.asarray(diagnostic["ideal_valid"], dtype=bool)
        measured_valid = np.asarray(diagnostic["measured_valid"], dtype=bool)
        group = handle[f"configurations/{selection_name}"]
        ideal_metrics = group["ideal_vs_coarse/gradient"]
        measured_metrics = group["measured_vs_ideal/gradient"]
        index = (int(component[0]), int(component[1]))
        ideal_r = float(ideal_metrics["pearson_r"][index])
        ideal_slope = float(ideal_metrics["regression_slope"][index])
        ideal_intercept = float(ideal_metrics["regression_intercept"][index])
        measured_r = float(measured_metrics["pearson_r"][(realization, *index)])
        measured_slope = float(measured_metrics["regression_slope"][(realization, *index)])
        measured_intercept = float(measured_metrics["regression_intercept"][(realization, *index)])

    figure = plt.figure(figsize=(11.5, 7.8))
    grid = figure.add_gridspec(2, 2)
    time_axis = figure.add_subplot(grid[0, :])
    coarse_plot = np.where(coarse_valid, coarse, np.nan)
    ideal_plot = np.where(ideal_valid, ideal, np.nan)
    measured_plot = np.where(measured_valid, measured, np.nan)
    time_axis.plot(time, coarse_plot, color="black", linewidth=1.3, label="coarse")
    time_axis.plot(time, ideal_plot, color="#cc5c99", linewidth=1.0, label="ideal")
    time_axis.plot(
        time, measured_plot, color="#48a9e6", linewidth=0.9, alpha=0.85, label="measured"
    )
    time_axis.set_xlabel("Time (s)")
    time_axis.set_ylabel(r"$G_{jk}$ (nT/km)")
    time_axis.grid(alpha=0.2)
    time_axis.legend(frameon=False, ncol=3)
    _scatter_with_regression(
        figure.add_subplot(grid[1, 0]),
        coarse,
        ideal,
        coarse_valid & ideal_valid,
        pearson_r=ideal_r,
        slope=ideal_slope,
        intercept=ideal_intercept,
        reference_label="Coarse",
        estimate_label="Ideal",
        color="#cc5c99",
    )
    _scatter_with_regression(
        figure.add_subplot(grid[1, 1]),
        ideal,
        measured,
        ideal_valid & measured_valid,
        pearson_r=measured_r,
        slope=measured_slope,
        intercept=measured_intercept,
        reference_label="Ideal",
        estimate_label=f"Measured MC {realization}",
        color="#48a9e6",
    )
    labels = ("x", "y", "z")
    figure.suptitle(
        f"Gradient-Correlation Diagnostic — {CORRELATION_SELECTION_TITLES[selection_name]}\n"
        rf"$G_{{{labels[index[0]]}{labels[index[1]]}}}"
        rf"=\partial B_{labels[index[0]]}/\partial {labels[index[1]]}$"
    )
    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_magnetic_gradient_correlations_from_config(
    config_path: str | Path,
) -> dict[str, Path]:
    """Render enabled figures from an existing correlation HDF5 product."""

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("magnetic-gradient correlation config must be a mapping")
    product_path = Path(config["output"]["path"])
    if not product_path.is_file():
        raise FileNotFoundError(
            f"correlation product not found: {product_path}; run hsobs-analyze first"
        )
    plot = config.get("plot", {})
    if not isinstance(plot, Mapping):
        raise ValueError("plot must be a mapping")
    outputs: dict[str, Path] = {}
    heatmaps = dict(plot.get("heatmaps", {}))
    if heatmaps.get("enabled", True):
        outputs["heatmaps"] = plot_magnetic_gradient_correlation_heatmaps(
            product_path,
            heatmaps.get("output", "gradient_correlation_heatmaps.png"),
            color_limits=tuple(heatmaps.get("color_limits", (-1.0, 1.0))),
            annotate_values=bool(heatmaps.get("annotate_values", True)),
            dpi=int(heatmaps.get("dpi", 300)),
        )
    measured_heatmaps = dict(plot.get("measured_vs_ideal_heatmaps", {}))
    if measured_heatmaps.get("enabled", False):
        outputs["measured_vs_ideal_heatmaps"] = plot_magnetic_gradient_measured_vs_ideal_heatmaps(
            product_path,
            measured_heatmaps.get("output", "gradient_correlation_measured_vs_ideal_heatmaps.png"),
            dpi=int(measured_heatmaps.get("dpi", 300)),
        )
    summary = dict(plot.get("summary", {}))
    if summary.get("enabled", True):
        outputs["summary"] = plot_magnetic_gradient_correlation_summary(
            product_path,
            summary.get("output", "gradient_correlation_summary.png"),
            dpi=int(summary.get("dpi", 300)),
        )
    combined = dict(plot.get("combined", {}))
    if combined.get("enabled", True):
        outputs["combined"] = plot_magnetic_gradient_correlation_combined(
            product_path,
            combined.get("output", "gradient_correlation_combined.png"),
            color_limits=tuple(combined.get("color_limits", (-1.0, 1.0))),
            annotate_values=bool(combined.get("annotate_values", True)),
            dpi=int(combined.get("dpi", 300)),
        )
    diagnostic = dict(plot.get("diagnostic", {}))
    if diagnostic.get("enabled", True):
        outputs["diagnostic"] = plot_magnetic_gradient_correlation_diagnostic(
            product_path,
            diagnostic.get("output", "gradient_correlation_diagnostic.png"),
            dpi=int(diagnostic.get("dpi", 200)),
        )
    return outputs


@register_analysis
class MagneticGradientCorrelationAnalysis(AnalysisMethod):
    """Registry adapter for standalone magnetic-gradient correlation products."""

    name = "magnetic_gradient_correlation"
    config_section = "magnetic_gradient_correlation"

    def run(self, input_path: str | Path) -> dict[str, Any]:
        result = compute_magnetic_gradient_correlations(input_path, self.config)
        if bool(self.config.get("diagnostics", {}).get("verbose", True)):
            print_correlation_summary(result)
        return result

    def write_result(self, result: dict[str, Any], output_path: str | Path) -> Path:
        storage = dict(self.config.get("storage", {}))
        compression = storage.get("compression", "gzip")
        if compression in {"none", None, False}:
            compression = None
        return write_magnetic_gradient_correlation_product(
            result, output_path, compression=compression
        )
