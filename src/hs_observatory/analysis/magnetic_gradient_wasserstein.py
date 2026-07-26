"""Distributional error diagnostics for saved magnetic-gradient products."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping
import warnings

import h5py
import matplotlib
import numpy as np
import yaml

from hs_observatory.analysis.magnetic_gradient import CONFIGURATION_ORDER

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def empirical_wasserstein_1d(first: np.ndarray, second: np.ndarray) -> float:
    """Return exact W1 between two finite one-dimensional empirical samples."""

    a = np.sort(np.asarray(first, dtype=float).ravel())
    b = np.sort(np.asarray(second, dtype=float).ravel())
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if not a.size or not b.size:
        return float("nan")
    support = np.sort(np.concatenate((a, b)))
    if support.size < 2:
        return 0.0
    widths = np.diff(support)
    a_cdf = np.searchsorted(a, support[:-1], side="right") / float(a.size)
    b_cdf = np.searchsorted(b, support[:-1], side="right") / float(b.size)
    return float(np.sum(np.abs(a_cdf - b_cdf) * widths))


def _validate_interval_percentiles(
    interval_percentiles: tuple[float, float],
) -> tuple[float, float]:
    lower, upper = (float(value) for value in interval_percentiles)
    if not 0.0 <= lower < 50.0 < upper <= 100.0:
        raise ValueError(
            "interval_percentiles must satisfy 0 <= lower < 50 < upper <= 100"
        )
    return lower, upper


def _summary(
    values: np.ndarray,
    interval_percentiles: tuple[float, float],
) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size:
        q25, median, q75 = np.percentile(finite, (25.0, 50.0, 75.0))
        lower, upper = np.percentile(finite, interval_percentiles)
    else:
        q25 = median = q75 = np.nan
        lower = upper = np.nan
    return {
        "by_mc": array.tolist(),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
        "interval_lower": float(lower),
        "interval_upper": float(upper),
        "n_valid": int(finite.size),
        "n_total": int(array.size),
    }


def _empirical_cdf(sample: np.ndarray, x: np.ndarray) -> np.ndarray:
    finite = np.sort(np.asarray(sample, dtype=float)[np.isfinite(sample)])
    if not finite.size:
        return np.full_like(x, np.nan)
    return np.searchsorted(finite, x, side="right") / float(finite.size)


def _case_specification(value: Any) -> tuple[Path, str | None]:
    if isinstance(value, Mapping):
        if "path" not in value:
            raise ValueError("case mappings must contain path")
        return Path(value["path"]), None if value.get("label") is None else str(value["label"])
    return Path(value), None


def _read_configuration(group: h5py.Group) -> dict[str, np.ndarray]:
    required = (
        "truth/coarse_curl_magnitude",
        "truth/coarse_valid",
        "true_estimate/curl_magnitude",
        "true_estimate/valid",
        "measured/curl_magnitude",
        "measured/valid",
    )
    missing = [name for name in required if name not in group]
    if missing:
        raise KeyError(f"missing datasets: {', '.join(missing)}")
    return {
        "coarse": np.asarray(group["truth/coarse_curl_magnitude"], dtype=float),
        "coarse_valid": np.asarray(group["truth/coarse_valid"], dtype=bool),
        "true": np.asarray(group["true_estimate/curl_magnitude"], dtype=float),
        "true_valid": np.asarray(group["true_estimate/valid"], dtype=bool),
        "measured": np.asarray(group["measured/curl_magnitude"], dtype=float),
        "measured_valid": np.asarray(group["measured/valid"], dtype=bool),
    }


def _configuration_diagnostics(
    data: Mapping[str, np.ndarray],
    interval_percentiles: tuple[float, float],
) -> dict[str, Any]:
    coarse = data["coarse"]
    true = data["true"]
    measured = data["measured"]
    base_valid = (
        data["coarse_valid"]
        & data["true_valid"]
        & np.isfinite(coarse)
        & np.isfinite(true)
    )
    scale_values = coarse[base_valid & (coarse > 0.0)]
    scale = float(np.median(scale_values)) if scale_values.size else float("nan")
    canonical_method = empirical_wasserstein_1d(true[base_valid], coarse[base_valid])

    n_mc = measured.shape[0]
    method_common = np.full(n_mc, np.nan)
    measurement = np.full(n_mc, np.nan)
    total = np.full(n_mc, np.nan)
    pointwise = np.full(n_mc, np.nan)
    triangle_slack = np.full(n_mc, np.nan)
    valid_samples = np.zeros(n_mc, dtype=int)
    for member in range(n_mc):
        valid = (
            base_valid
            & data["measured_valid"][member]
            & np.isfinite(measured[member])
        )
        valid_samples[member] = int(np.count_nonzero(valid))
        if not valid_samples[member]:
            continue
        c = coarse[valid]
        t = true[valid]
        m = measured[member, valid]
        method_common[member] = empirical_wasserstein_1d(t, c)
        measurement[member] = empirical_wasserstein_1d(m, t)
        total[member] = empirical_wasserstein_1d(m, c)
        pointwise[member] = float(np.median(np.abs(m - t)))
        triangle_slack[member] = method_common[member] + measurement[member] - total[member]

    def normalized(values: np.ndarray) -> np.ndarray:
        if not np.isfinite(scale) or scale <= 0.0:
            return np.full_like(values, np.nan)
        return values / scale

    return {
        "reference_scale": scale,
        "method_w1_raw": canonical_method,
        "method_w1_normalized": (
            canonical_method / scale if np.isfinite(scale) and scale > 0.0 else np.nan
        ),
        "method_common_raw": _summary(method_common, interval_percentiles),
        "method_common_normalized": _summary(
            normalized(method_common), interval_percentiles
        ),
        "measurement_w1_raw": _summary(measurement, interval_percentiles),
        "measurement_w1_normalized": _summary(
            normalized(measurement), interval_percentiles
        ),
        "total_w1_raw": _summary(total, interval_percentiles),
        "total_w1_normalized": _summary(normalized(total), interval_percentiles),
        "pointwise_median_absolute_error_raw": _summary(
            pointwise, interval_percentiles
        ),
        "pointwise_median_absolute_error_normalized": _summary(
            normalized(pointwise), interval_percentiles
        ),
        "triangle_slack_raw": _summary(triangle_slack, interval_percentiles),
        "valid_time_samples_by_mc": valid_samples.tolist(),
        "_cdf_source": {
            "coarse": coarse[base_valid],
            "true": true[base_valid],
            "measured": measured,
            "measured_valid": data["measured_valid"] & base_valid[None, :],
        },
    }


def _cdf_diagnostics(
    source: Mapping[str, np.ndarray],
    scale: float,
    n_points: int,
    interval_percentiles: tuple[float, float],
) -> dict[str, Any]:
    if not np.isfinite(scale) or scale <= 0.0:
        return {}
    coarse = source["coarse"] / scale
    true = source["true"] / scale
    measured_rows = []
    for row, valid in zip(source["measured"], source["measured_valid"]):
        measured_rows.append(row[valid] / scale)
    finite_measured = [row[np.isfinite(row)] for row in measured_rows if np.any(np.isfinite(row))]
    pooled = np.concatenate((coarse, true, *finite_measured))
    upper = float(np.percentile(pooled[np.isfinite(pooled)], 99.75))
    x = np.linspace(0.0, max(upper, np.finfo(float).eps), n_points)
    measured_cdf = np.asarray([_empirical_cdf(row, x) for row in measured_rows])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lower, median, upper = np.nanpercentile(
            measured_cdf,
            (interval_percentiles[0], 50.0, interval_percentiles[1]),
            axis=0,
        )
    return {
        "x_normalized": x.tolist(),
        "coarse_cdf": _empirical_cdf(coarse, x).tolist(),
        "true_cdf": _empirical_cdf(true, x).tolist(),
        "measured_cdf_lower": lower.tolist(),
        "measured_cdf_median": median.tolist(),
        "measured_cdf_upper": upper.tolist(),
    }


def compute_magnetic_gradient_wasserstein_diagnostics(
    results_by_case: Mapping[str, str | Path | Mapping[str, Any]],
    *,
    all_errors_case: str = "200",
    cdf_points: int = 256,
    interval_percentiles: tuple[float, float] = (25.0, 75.0),
) -> dict[str, Any]:
    """Compute curl-magnitude distributional errors from saved gradient products."""

    if cdf_points < 16:
        raise ValueError("cdf_points must be at least 16")
    interval_percentiles = _validate_interval_percentiles(interval_percentiles)
    messages: list[str] = []
    configurations: dict[str, dict[str, Any]] = {}
    labels: dict[str, str] = {}
    case_order: list[str] = []
    for raw_case, specification in results_by_case.items():
        case = str(raw_case)
        path, label = _case_specification(specification)
        labels[case] = label or case
        if not path.is_file():
            message = f"Skipping case {case}: gradient product not found: {path}"
            messages.append(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            continue
        try:
            with h5py.File(path, "r") as handle:
                available = [
                    name for name in CONFIGURATION_ORDER
                    if f"configurations/{name}" in handle
                ]
                if not available:
                    raise ValueError("product contains no recognized configurations")
                for category in available:
                    data = _read_configuration(handle[f"configurations/{category}"])
                    entry = _configuration_diagnostics(data, interval_percentiles)
                    cdf_source = entry.pop("_cdf_source")
                    bucket = configurations.setdefault(category, {"cases": {}})
                    bucket["cases"][case] = entry
                    if case == all_errors_case:
                        bucket["cdf"] = _cdf_diagnostics(
                            cdf_source,
                            entry["reference_scale"],
                            cdf_points,
                            interval_percentiles,
                        )
            case_order.append(case)
        except (OSError, KeyError, TypeError, ValueError) as error:
            message = f"Skipping case {case}: {error}"
            messages.append(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)

    if not case_order:
        raise ValueError("No usable magnetic-gradient products were found")
    if all_errors_case not in case_order:
        all_errors_case = case_order[0]
        messages.append(f"Using case {all_errors_case} for CDF panels")
        # Re-read only the selected fallback to construct its CDF data.
        path, _label = _case_specification(results_by_case[all_errors_case])
        with h5py.File(path, "r") as handle:
            for category in configurations:
                data = _read_configuration(handle[f"configurations/{category}"])
                entry = _configuration_diagnostics(data, interval_percentiles)
                source = entry.pop("_cdf_source")
                configurations[category]["cdf"] = _cdf_diagnostics(
                    source,
                    entry["reference_scale"],
                    cdf_points,
                    interval_percentiles,
                )

    return {
        "quantity": "curl_magnitude",
        "quantity_label": "|curl B|",
        "units": "nT/km",
        "normalization": "median positive coarse-truth curl magnitude per configuration",
        "interval_percentiles": list(interval_percentiles),
        "all_errors_case": all_errors_case,
        "case_order": case_order,
        "case_labels": {case: labels[case] for case in case_order},
        "configurations": configurations,
        "warnings": messages,
    }


def _errorbar(
    axis: plt.Axes,
    x: np.ndarray,
    entries: list[Mapping[str, Any]],
    key: str,
    *,
    offset: float,
    label: str,
    color: str,
) -> None:
    median = np.asarray([entry[key]["median"] for entry in entries])
    lower = median - np.asarray(
        [entry[key].get("interval_lower", entry[key]["q25"]) for entry in entries]
    )
    upper = np.asarray(
        [entry[key].get("interval_upper", entry[key]["q75"]) for entry in entries]
    ) - median
    axis.errorbar(
        x + offset, median, yerr=np.vstack((lower, upper)), fmt="o",
        color=color, capsize=3, markersize=4, label=label,
    )


def plot_magnetic_gradient_wasserstein(
    diagnostics: Mapping[str, Any],
    output_path: str | Path,
    *,
    dpi: int = 300,
) -> Path:
    """Plot CDFs and distributional error decomposition."""

    categories = [
        category for category in CONFIGURATION_ORDER
        if category in diagnostics["configurations"]
    ]
    cases = diagnostics["case_order"]
    labels = diagnostics["case_labels"]
    interval_percentiles = diagnostics.get("interval_percentiles", (25.0, 75.0))
    interval_label = (
        f"{float(interval_percentiles[0]):g}th–"
        f"{float(interval_percentiles[1]):g}th"
    )
    figure, axes = plt.subplots(
        3, len(categories), figsize=(4.3 * len(categories), 10.5), squeeze=False
    )
    wasserstein_limits: list[float] = []
    pointwise_limits: list[float] = []
    for column, category in enumerate(categories):
        product = diagnostics["configurations"][category]
        axis = axes[0, column]
        cdf = product.get("cdf", {})
        if cdf:
            x = np.asarray(cdf["x_normalized"])
            axis.plot(x, cdf["coarse_cdf"], color="black", lw=1.6, label="coarse truth")
            axis.plot(x, cdf["true_cdf"], color="#cc5c99", lw=1.3, label="true estimate")
            axis.fill_between(
                x,
                cdf.get("measured_cdf_lower", cdf.get("measured_cdf_q25")),
                cdf.get("measured_cdf_upper", cdf.get("measured_cdf_q75")),
                color="#48a9e6", alpha=0.24, linewidth=0,
            )
            axis.plot(
                x, cdf["measured_cdf_median"], color="#48a9e6", lw=1.3,
                label=f"measured median/{interval_label}",
            )
        axis.set_title(category.replace("_", " "))
        axis.set_xlabel(r"$|\nabla\times B|/\mathrm{median}(|\nabla\times B|_{\rm coarse})$")
        axis.set_ylabel("Empirical CDF")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.22)

        entries = [product["cases"][case] for case in cases if case in product["cases"]]
        local_cases = [case for case in cases if case in product["cases"]]
        positions = np.arange(len(local_cases), dtype=float)
        metric_axis = axes[1, column]
        _errorbar(
            metric_axis, positions, entries, "measurement_w1_normalized",
            offset=-0.12, label=r"$W_1(M,T)$", color="#48a9e6",
        )
        _errorbar(
            metric_axis, positions, entries, "total_w1_normalized",
            offset=0.12, label=r"$W_1(M,C)$", color="#e181b4",
        )
        method = float(entries[0]["method_w1_normalized"]) if entries else np.nan
        if np.isfinite(method):
            wasserstein_limits.append(method)
            metric_axis.axhline(
                method,
                color="black",
                linestyle="--",
                linewidth=1.2,
                label=r"$W_1(T,C)$",
            )
        for entry in entries:
            for key in ("measurement_w1_normalized", "total_w1_normalized"):
                wasserstein_limits.extend(
                    (
                        entry[key].get("interval_lower", entry[key]["q25"]),
                        entry[key].get("interval_upper", entry[key]["q75"]),
                    )
                )
        metric_axis.set_yscale("log")
        metric_axis.set_ylabel(r"Normalized $W_1$")
        metric_axis.set_xticks(positions, [labels[case] for case in local_cases], rotation=25, ha="right")
        metric_axis.grid(alpha=0.22, axis="y")

        point_axis = axes[2, column]
        _errorbar(
            point_axis, positions, entries,
            "pointwise_median_absolute_error_normalized",
            offset=0.0, label="pointwise", color="#725ac1",
        )
        for entry in entries:
            pointwise = entry["pointwise_median_absolute_error_normalized"]
            pointwise_limits.extend(
                (
                    pointwise.get("interval_lower", pointwise["q25"]),
                    pointwise.get("interval_upper", pointwise["q75"]),
                )
            )
        point_axis.set_yscale("log")
        point_axis.set_ylabel("Normalized pointwise median |M−T|")
        point_axis.set_xticks(
            positions, [labels[case] for case in local_cases], rotation=25, ha="right"
        )
        point_axis.grid(alpha=0.22, axis="y")
    positive_limits = np.asarray(
        [value for value in wasserstein_limits if np.isfinite(value) and value > 0.0]
    )
    if positive_limits.size:
        low, high = np.min(np.log10(positive_limits)), np.max(np.log10(positive_limits))
        padding = 0.06 * max(high - low, 1.0)
        for axis in axes[1]:
            axis.set_ylim(10.0 ** (low - padding), 10.0 ** (high + padding))
    positive_pointwise = np.asarray(
        [value for value in pointwise_limits if np.isfinite(value) and value > 0.0]
    )
    if positive_pointwise.size:
        low, high = np.min(np.log10(positive_pointwise)), np.max(np.log10(positive_pointwise))
        padding = 0.06 * max(high - low, 1.0)
        for axis in axes[2]:
            axis.set_ylim(10.0 ** (low - padding), 10.0 ** (high + padding))
    if categories:
        axes[0, 0].legend(frameon=False, fontsize=8)
        axes[1, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("Magnetic-Gradient Curl-Magnitude Distributional Errors")
    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output


def _most_regular_tetrahedron(
    handle: h5py.File, requested: str | None
) -> tuple[str, float]:
    tetrahedra: list[tuple[float, str]] = []
    for category in CONFIGURATION_ORDER:
        path = f"configurations/{category}"
        if path not in handle:
            continue
        group = handle[path]
        if len(group["spacecraft_indices"]) != 4:
            continue
        chi = np.asarray(group["geometry/chi"], dtype=float)
        finite = chi[np.isfinite(chi)]
        median_chi = float(np.median(finite)) if finite.size else float("inf")
        tetrahedra.append((median_chi, category))
    if not tetrahedra:
        raise ValueError("gradient product contains no four-spacecraft configuration")
    if requested not in {None, "most_regular", "most_regular_tetrahedron"}:
        matches = [item for item in tetrahedra if item[1] == requested]
        if not matches:
            raise ValueError(
                f"requested configuration {requested!r} is not an available tetrahedron"
            )
        return matches[0][1], matches[0][0]
    median_chi, category = min(tetrahedra)
    return category, median_chi


def _set_linear_limits(
    axis: plt.Axes, values: list[np.ndarray], *, nonnegative: bool = False
) -> None:
    finite_parts = [
        value[np.isfinite(value)] for value in values if np.any(np.isfinite(value))
    ]
    if not finite_parts:
        return
    finite = np.concatenate(finite_parts)
    low, high = float(np.min(finite)), float(np.max(finite))
    if high <= low:
        high = low + max(abs(low) * 0.05, 1.0)
    padding = 0.05 * (high - low)
    lower = max(0.0, low - padding) if nonnegative else low - padding
    axis.set_ylim(lower, high + padding)


def plot_magnetic_gradient_combined(
    diagnostics: Mapping[str, Any],
    product_path: str | Path,
    output_path: str | Path,
    *,
    configuration: str | None = "most_regular",
    start_time_s: float | None = None,
    stop_time_s: float | None = None,
    dpi: int = 300,
) -> tuple[Path, str]:
    """Plot time-domain and distributional curl diagnostics in a 2x3 figure."""

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
        category, median_chi = _most_regular_tetrahedron(handle, configuration)
        if category not in diagnostics["configurations"]:
            raise ValueError(
                f"configuration {category!r} is absent from Wasserstein diagnostics"
            )
        group = handle[f"configurations/{category}"]
        if "measured/curl" not in group:
            raise ValueError(
                "combined plot requires stored measured curl realizations; rerun "
                "with monte_carlo.store_all_realizations: true"
            )
        time = np.asarray(handle["time/analysis_time"], dtype=float)
        time_mask = np.ones(time.shape, dtype=bool)
        if start_time_s is not None:
            time_mask &= time >= start_time_s
        if stop_time_s is not None:
            time_mask &= time <= stop_time_s
        if not np.any(time_mask):
            raise ValueError("requested time range contains no analysis samples")
        selected_time = time[time_mask]

        truth = group["truth"]
        ideal = group["true_estimate"]
        summary = group.get("mc_summary")
        if summary is None:
            raise ValueError("combined plot requires Monte Carlo summary datasets")
        coarse_magnitude = np.asarray(
            truth["coarse_curl_magnitude"], dtype=float
        )[time_mask]
        ideal_magnitude = np.asarray(ideal["curl_magnitude"], dtype=float)[time_mask]
        measured_magnitude = {
            statistic: np.asarray(
                summary[f"curl_magnitude/{statistic}"], dtype=float
            )[time_mask]
            for statistic in ("q25", "median", "q75")
        }
        ideal_error = np.asarray(ideal["curl_l2_error"], dtype=float)[time_mask]
        measured_error = {
            statistic: np.asarray(
                summary[f"curl_l2_error/{statistic}"], dtype=float
            )[time_mask]
            for statistic in ("q25", "median", "q75")
        }
        ideal_curl = np.asarray(ideal["curl"], dtype=float)[time_mask]
        measured_curl = np.asarray(group["measured/curl"], dtype=float)[:, time_mask, :]
        coarse_curl_x = np.asarray(truth["coarse_curl"], dtype=float)[time_mask, 0]
        ideal_curl_x = ideal_curl[:, 0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            curl_x_q25, curl_x_median, curl_x_q75 = np.nanpercentile(
                measured_curl[:, :, 0], (25.0, 50.0, 75.0), axis=0
            )
        indices = np.asarray(group["spacecraft_indices"], dtype=int).tolist()
        ids = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in group["spacecraft_ids"][...]
        ]
        size = np.asarray(group["geometry/characteristic_size"], dtype=float)
        median_size = float(np.nanmedian(size))

    figure, axes = plt.subplots(2, 3, figsize=(15.2, 8.6), squeeze=False)
    truth_color = "#111111"
    ideal_color = "#cc5c99"
    measured_color = "#48a9e6"

    axis = axes[0, 1]
    axis.plot(
        selected_time,
        coarse_magnitude,
        color=truth_color,
        linewidth=1.5,
        label="coarse truth",
    )
    axis.plot(
        selected_time,
        ideal_magnitude,
        color=ideal_color,
        linewidth=1.1,
        label="ideal estimate",
    )
    axis.fill_between(
        selected_time,
        measured_magnitude["q25"],
        measured_magnitude["q75"],
        color=measured_color,
        alpha=0.23,
        linewidth=0,
    )
    axis.plot(
        selected_time,
        measured_magnitude["median"],
        color=measured_color,
        linewidth=1.2,
        label="measured median/IQR",
    )
    axis.set_ylabel(r"$|\nabla\times B|$ (nT/km)")
    axis.legend(frameon=False, fontsize=8)
    _set_linear_limits(
        axis,
        [
            coarse_magnitude,
            ideal_magnitude,
            measured_magnitude["q25"],
            measured_magnitude["q75"],
        ],
        nonnegative=True,
    )

    axis = axes[0, 2]
    axis.plot(
        selected_time,
        ideal_error,
        color=ideal_color,
        linewidth=1.1,
        label=r"ideal $|C_T-C_C|$",
    )
    axis.fill_between(
        selected_time,
        measured_error["q25"],
        measured_error["q75"],
        color=measured_color,
        alpha=0.23,
        linewidth=0,
    )
    axis.plot(
        selected_time,
        measured_error["median"],
        color=measured_color,
        linewidth=1.2,
        label=r"measured $|C_M-C_C|$ median/IQR",
    )
    axis.set_ylabel(r"$|\Delta(\nabla\times B)|$ (nT/km)")
    axis.legend(frameon=False, fontsize=8)
    _set_linear_limits(
        axis,
        [
            ideal_error,
            measured_error["q25"],
            measured_error["median"],
            measured_error["q75"],
        ],
        nonnegative=True,
    )

    axis = axes[0, 0]
    axis.plot(
        selected_time,
        coarse_curl_x,
        color=truth_color,
        linewidth=1.5,
        label="coarse truth",
    )
    axis.plot(
        selected_time,
        ideal_curl_x,
        color=ideal_color,
        linewidth=1.1,
        label="ideal estimate",
    )
    axis.fill_between(
        selected_time,
        curl_x_q25,
        curl_x_q75,
        color=measured_color,
        alpha=0.23,
        linewidth=0,
    )
    axis.plot(
        selected_time,
        curl_x_median,
        color=measured_color,
        linewidth=1.2,
        label="measured median/IQR",
    )
    axis.set_ylabel(r"$(\nabla\times B)_x$ (nT/km)")
    _set_linear_limits(
        axis,
        [coarse_curl_x, ideal_curl_x, curl_x_q25, curl_x_median, curl_x_q75],
    )
    axis.legend(frameon=False, fontsize=8)

    for axis in axes[0]:
        axis.set_xlabel("Time (s)")
        axis.set_xlim(
            start_time_s if start_time_s is not None else selected_time[0],
            stop_time_s if stop_time_s is not None else selected_time[-1],
        )
        axis.grid(alpha=0.22)

    product = diagnostics["configurations"][category]
    cdf = product.get("cdf", {})
    axis = axes[1, 0]
    if cdf:
        x = np.asarray(cdf["x_normalized"])
        axis.plot(x, cdf["coarse_cdf"], color=truth_color, lw=1.6, label="coarse truth")
        axis.plot(x, cdf["true_cdf"], color=ideal_color, lw=1.3, label="ideal estimate")
        interval_percentiles = diagnostics.get(
            "interval_percentiles", (25.0, 75.0)
        )
        interval_label = (
            f"{float(interval_percentiles[0]):g}th–"
            f"{float(interval_percentiles[1]):g}th"
        )
        axis.fill_between(
            x,
            cdf.get("measured_cdf_lower", cdf.get("measured_cdf_q25")),
            cdf.get("measured_cdf_upper", cdf.get("measured_cdf_q75")),
            color=measured_color,
            alpha=0.24,
            linewidth=0,
        )
        axis.plot(
            x,
            cdf["measured_cdf_median"],
            color=measured_color,
            lw=1.3,
            label=f"measured median/{interval_label}",
        )
    axis.set_xlabel(
        r"$|\nabla\times B|/\mathrm{median}(|\nabla\times B|_{\rm coarse})$"
    )
    axis.set_ylabel("Empirical CDF")
    axis.set_ylim(0.0, 1.0)
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.22)

    cases = diagnostics["case_order"]
    labels = diagnostics["case_labels"]
    local_cases = [case for case in cases if case in product["cases"]]
    entries = [product["cases"][case] for case in local_cases]
    positions = np.arange(len(local_cases), dtype=float)

    axis = axes[1, 1]
    _errorbar(
        axis,
        positions,
        entries,
        "measurement_w1_normalized",
        offset=-0.12,
        label=r"$W_1(M,T)$",
        color=measured_color,
    )
    _errorbar(
        axis,
        positions,
        entries,
        "total_w1_normalized",
        offset=0.12,
        label=r"$W_1(M,C)$",
        color="#e181b4",
    )
    method = float(entries[0]["method_w1_normalized"]) if entries else np.nan
    if np.isfinite(method):
        axis.axhline(
            method,
            color=truth_color,
            linestyle="--",
            linewidth=1.2,
            label=r"$W_1(T,C)$",
        )
    wasserstein_limits = [method]
    for entry in entries:
        for key in ("measurement_w1_normalized", "total_w1_normalized"):
            wasserstein_limits.extend(
                (
                    entry[key].get("interval_lower", entry[key]["q25"]),
                    entry[key].get("interval_upper", entry[key]["q75"]),
                )
            )
    axis.set_yscale("log")
    axis.set_ylabel(r"Normalized $W_1$")
    axis.set_xticks(
        positions, [labels[case] for case in local_cases], rotation=25, ha="right"
    )
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.22, axis="y")

    axis = axes[1, 2]
    _errorbar(
        axis,
        positions,
        entries,
        "pointwise_median_absolute_error_normalized",
        offset=0.0,
        label="pointwise",
        color="#725ac1",
    )
    pointwise_limits: list[float] = []
    for entry in entries:
        pointwise = entry["pointwise_median_absolute_error_normalized"]
        pointwise_limits.extend(
            (
                pointwise.get("interval_lower", pointwise["q25"]),
                pointwise.get("interval_upper", pointwise["q75"]),
            )
        )
    axis.set_yscale("log")
    axis.set_ylabel("Normalized pointwise median |M−T|")
    axis.set_xticks(
        positions, [labels[case] for case in local_cases], rotation=25, ha="right"
    )
    axis.grid(alpha=0.22, axis="y")

    for axis, limits in (
        (axes[1, 1], wasserstein_limits),
        (axes[1, 2], pointwise_limits),
    ):
        positive = np.asarray(
            [value for value in limits if np.isfinite(value) and value > 0.0]
        )
        if positive.size:
            low = float(np.min(np.log10(positive)))
            high = float(np.max(np.log10(positive)))
            padding = 0.06 * max(high - low, 1.0)
            axis.set_ylim(10.0 ** (low - padding), 10.0 ** (high + padding))

    for label, axis in zip(("a", "b", "c", "d", "e", "f"), axes.ravel()):
        axis.text(
            -0.12,
            1.04,
            f"({label})",
            transform=axis.transAxes,
            fontsize=11,
            fontweight="bold",
            va="bottom",
        )
    figure.suptitle(
        f"Magnetic-Gradient Curl Diagnostics — {category.replace('_', ' ')} "
        f"tetrahedron\n{ids} / {indices}; median L={median_size:.3g} km, "
        f"Q={median_chi:.3g}"
    )
    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output, category


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def write_magnetic_gradient_wasserstein_json(
    diagnostics: Mapping[str, Any], output_path: str | Path
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_json_safe(diagnostics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def write_magnetic_gradient_wasserstein_csv(
    diagnostics: Mapping[str, Any], output_path: str | Path
) -> Path:
    columns = [
        "configuration", "case", "label", "reference_scale_nT_per_km",
        "method_w1_raw", "method_w1_normalized",
        "measurement_w1_raw_q25", "measurement_w1_raw_median", "measurement_w1_raw_q75",
        "measurement_w1_normalized_q25", "measurement_w1_normalized_median",
        "measurement_w1_normalized_q75", "total_w1_raw_q25", "total_w1_raw_median",
        "total_w1_raw_q75", "total_w1_normalized_q25", "total_w1_normalized_median",
        "total_w1_normalized_q75", "pointwise_normalized_q25",
        "pointwise_normalized_median", "pointwise_normalized_q75", "n_mc_valid", "n_mc_total",
    ]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for category, product in diagnostics["configurations"].items():
            for case, entry in product["cases"].items():
                row = {
                    "configuration": category,
                    "case": case,
                    "label": diagnostics["case_labels"][case],
                    "reference_scale_nT_per_km": entry["reference_scale"],
                    "method_w1_raw": entry["method_w1_raw"],
                    "method_w1_normalized": entry["method_w1_normalized"],
                    "n_mc_valid": entry["measurement_w1_raw"]["n_valid"],
                    "n_mc_total": entry["measurement_w1_raw"]["n_total"],
                }
                for prefix, key in (
                    ("measurement_w1_raw", "measurement_w1_raw"),
                    ("measurement_w1_normalized", "measurement_w1_normalized"),
                    ("total_w1_raw", "total_w1_raw"),
                    ("total_w1_normalized", "total_w1_normalized"),
                    ("pointwise_normalized", "pointwise_median_absolute_error_normalized"),
                ):
                    for statistic in ("q25", "median", "q75"):
                        row[f"{prefix}_{statistic}"] = entry[key][statistic]
                writer.writerow(row)
    return output


def magnetic_gradient_wasserstein_from_config(
    config_path: str | Path,
) -> tuple[Path, dict[str, Any]]:
    """Read YAML, compute diagnostics, and write configured artifacts."""

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("magnetic-gradient Wasserstein config must be a mapping")
    cases = config.get("cases")
    if not isinstance(cases, Mapping) or not cases:
        raise ValueError("config requires a nonempty cases mapping")
    diagnostics = compute_magnetic_gradient_wasserstein_diagnostics(
        cases,
        all_errors_case=str(config.get("all_errors_case", "200")),
        cdf_points=int(config.get("cdf_points", 256)),
        interval_percentiles=tuple(
            float(value)
            for value in config.get("interval_percentiles", (25.0, 75.0))
        ),
    )
    output = config.get("output", {})
    if not isinstance(output, Mapping):
        raise ValueError("output must be a mapping")
    figure = plot_magnetic_gradient_wasserstein(
        diagnostics,
        output.get("figure", "magnetic_gradient_wasserstein.png"),
        dpi=int(config.get("dpi", 300)),
    )
    if output.get("json"):
        write_magnetic_gradient_wasserstein_json(diagnostics, output["json"])
    if output.get("csv"):
        write_magnetic_gradient_wasserstein_csv(diagnostics, output["csv"])
    return figure, diagnostics


def magnetic_gradient_combined_from_config(
    config_path: str | Path,
    *,
    output_override: str | Path | None = None,
    configuration_override: str | None = None,
    start_time_override: float | None = None,
    stop_time_override: float | None = None,
    dpi_override: int | None = None,
) -> tuple[Path, dict[str, Any], str]:
    """Compute diagnostics and write the configured six-panel combined figure."""

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("magnetic-gradient combined config must be a mapping")
    cases = config.get("cases")
    if not isinstance(cases, Mapping) or not cases:
        raise ValueError("config requires a nonempty cases mapping")
    all_errors_case = str(config.get("all_errors_case", "200"))
    diagnostics = compute_magnetic_gradient_wasserstein_diagnostics(
        cases,
        all_errors_case=all_errors_case,
        cdf_points=int(config.get("cdf_points", 256)),
        interval_percentiles=tuple(
            float(value)
            for value in config.get("interval_percentiles", (25.0, 75.0))
        ),
    )
    selected_case = str(diagnostics["all_errors_case"])
    product_path, _label = _case_specification(cases[selected_case])
    output = config.get("output", {})
    if not isinstance(output, Mapping):
        raise ValueError("output must be a mapping")
    output_path = output_override or output.get(
        "figure", "magnetic_gradient_combined.png"
    )
    configuration = (
        configuration_override
        if configuration_override is not None
        else str(config.get("configuration", "most_regular"))
    )
    start_time_s = (
        start_time_override
        if start_time_override is not None
        else config.get("start_time_s")
    )
    stop_time_s = (
        stop_time_override
        if stop_time_override is not None
        else config.get("stop_time_s")
    )
    figure, selected_configuration = plot_magnetic_gradient_combined(
        diagnostics,
        product_path,
        output_path,
        configuration=configuration,
        start_time_s=None if start_time_s is None else float(start_time_s),
        stop_time_s=None if stop_time_s is None else float(stop_time_s),
        dpi=int(dpi_override if dpi_override is not None else config.get("dpi", 300)),
    )
    return figure, diagnostics, selected_configuration
