"""Cheap Wasserstein diagnostics for saved Elsasser increment-PDF products."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping
import warnings

import numpy as np

from hs_observatory.analysis.elsasser_increment_pdf_static import (
    _load_publication_pdf_product,
)


def _bin_geometry(bins: np.ndarray, n_pdf: int) -> tuple[np.ndarray, np.ndarray]:
    """Return bin centers and widths from edges or centers."""

    values = np.asarray(bins, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("PDF bins must be a one-dimensional finite array")
    if values.size == n_pdf + 1:
        edges = values
        centers = 0.5 * (edges[:-1] + edges[1:])
    elif values.size == n_pdf:
        if n_pdf < 2:
            raise ValueError("At least two bin centers are needed to infer bin widths")
        centers = values
        edges = np.empty(n_pdf + 1, dtype=float)
        edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
        edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
        edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    else:
        raise ValueError("PDF bins must contain N centers or N+1 edges")
    widths = np.diff(edges)
    if np.any(widths <= 0.0):
        raise ValueError("PDF bins must be strictly increasing")
    return centers, widths


def normalize_pdf(pdf: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Normalize a binned density over its finite bin range using actual widths."""

    density = np.asarray(pdf, dtype=float)
    if density.ndim != 1:
        raise ValueError("PDF must be one-dimensional")
    _centers, widths = _bin_geometry(bins, density.size)
    if np.any(np.isfinite(density) & (density < 0.0)):
        raise ValueError("PDF density cannot be negative")
    clean = np.where(np.isfinite(density), density, 0.0)
    integral = float(np.sum(clean * widths))
    if not np.isfinite(integral) or integral <= 0.0:
        raise ValueError("PDF has no positive finite integral")
    return clean / integral


def symmetric_pdf_difference(
    pdf_mc: np.ndarray,
    pdf_true: np.ndarray,
    bins: np.ndarray,
) -> np.ndarray:
    """Return ``|PDF_MC - PDF_true| / |PDF_MC + PDF_true|`` bin by bin.

    Both inputs are first normalized over the common finite histogram range.
    Bins where both normalized densities are zero are returned as NaN.
    """

    mc_density = normalize_pdf(pdf_mc, bins)
    truth_density = normalize_pdf(pdf_true, bins)
    numerator = np.abs(mc_density - truth_density)
    denominator = np.abs(mc_density + truth_density)
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan),
        where=denominator > 0.0,
    )


def pdf_to_cdf(pdf: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Convert a binned density to its normalized right-edge CDF values."""

    normalized = normalize_pdf(pdf, bins)
    _centers, widths = _bin_geometry(bins, normalized.size)
    return np.cumsum(normalized * widths)


def wasserstein_from_binned_pdfs(pdf_a: np.ndarray, pdf_b: np.ndarray, bins: np.ndarray) -> float:
    """Approximate one-dimensional W1 from shared binned-PDF CDFs."""

    first = np.asarray(pdf_a, dtype=float)
    second = np.asarray(pdf_b, dtype=float)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("PDFs must be one-dimensional arrays with identical shapes")
    _centers, widths = _bin_geometry(bins, first.size)
    return float(np.sum(np.abs(pdf_to_cdf(first, bins) - pdf_to_cdf(second, bins)) * widths))


def geometric_mean_normalized_rayleigh_pdf_on_bins(
    bins: np.ndarray, *, bins_are_edges: bool = True
) -> np.ndarray:
    """Return the finite-range-normalized Rayleigh density on histogram bins.

    The saved perpendicular-increment PDFs use the magnitude of a two-component
    vector divided by its geometric mean.  For isotropic zero-mean Gaussian
    components, that normalized magnitude has density

    ``2 exp(-gamma) x exp[-exp(-gamma) x**2]`` for ``x >= 0``,

    where ``gamma`` is the Euler-Mascheroni constant.  Bin-average densities are
    computed from the analytic CDF before normalization over the saved finite
    histogram range.
    """

    values = np.asarray(bins, dtype=float)
    n_pdf = values.size - 1 if bins_are_edges else values.size
    centers, widths = _bin_geometry(values, n_pdf)
    edges = np.concatenate(([centers[0] - 0.5 * widths[0]], centers + 0.5 * widths))
    positive_edges = np.maximum(edges, 0.0)
    rate = np.exp(-np.euler_gamma)
    cdf = 1.0 - np.exp(-rate * positive_edges**2)
    density = np.diff(cdf) / widths
    return normalize_pdf(density, values)


def rayleigh_normalized_wasserstein(
    pdf_case: np.ndarray,
    pdf_true: np.ndarray,
    bins: np.ndarray,
    *,
    denominator_tolerance: float = 1.0e-12,
) -> float:
    """Return W1(case, truth) / W1(normalized Rayleigh, truth)."""

    if denominator_tolerance < 0.0:
        raise ValueError("denominator_tolerance must be non-negative")
    rayleigh = geometric_mean_normalized_rayleigh_pdf_on_bins(np.asarray(bins, dtype=float))
    denominator = wasserstein_from_binned_pdfs(rayleigh, pdf_true, bins)
    if denominator <= denominator_tolerance:
        warnings.warn(
            "Rayleigh-to-truth Wasserstein distance is at or below the configured "
            "tolerance; returning NaN",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.nan
    return wasserstein_from_binned_pdfs(pdf_case, pdf_true, bins) / denominator


def _case_value(value: Any) -> tuple[Any, str | None]:
    if isinstance(value, Mapping) and "path" in value:
        return value["path"], None if value.get("label") is None else str(value["label"])
    return value, None


def _available_lag_indices(product: Mapping[str, np.ndarray]) -> np.ndarray:
    valid = np.asarray(product["valid"], dtype=bool)
    centers = np.asarray(product["lambda_centers"], dtype=float)
    usable = valid[0] & np.any(valid[1:], axis=0) & np.isfinite(centers)
    return np.flatnonzero(usable)


def _warn(message: str, messages: list[str]) -> None:
    messages.append(message)
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def _diagnostic_number(value: float) -> float:
    return float(value)


def compute_increment_pdf_wasserstein_diagnostics(
    results_by_case: Mapping[str, str | Path | Mapping[str, Any]],
    selected_lag: float | None = None,
    default_lag: str = "second_largest",
    quantity: str = "z_plus",
    *,
    denominator_tolerance: float = 1.0e-12,
    reference_case: str = "001",
) -> dict[str, Any]:
    """Compute per-MC Rayleigh-normalized W1 diagnostics from saved products.

    Missing products and unusable cases are warned about and skipped. The target
    lag comes from ``reference_case`` (or the first available case), while each
    case independently selects its nearest usable saved lag.
    """

    if quantity != "z_plus":
        raise ValueError("this diagnostic currently supports quantity='z_plus' only")
    if default_lag != "second_largest":
        raise ValueError("default_lag must be 'second_largest'")
    if denominator_tolerance < 0.0:
        raise ValueError("denominator_tolerance must be non-negative")
    if selected_lag is not None and (not np.isfinite(selected_lag) or selected_lag <= 0.0):
        raise ValueError("selected_lag must be a finite positive value in km")

    messages: list[str] = []
    loaded: dict[str, dict[str, Any]] = {}
    labels: dict[str, str] = {}
    for raw_case, value in results_by_case.items():
        case = str(raw_case)
        source, label = _case_value(value)
        labels[case] = label or case
        try:
            if isinstance(source, Mapping):
                product = dict(source)
            else:
                path = Path(source)
                if not path.is_file():
                    _warn(f"Skipping case {case}: saved PDF product not found: {path}", messages)
                    continue
                product = _load_publication_pdf_product(path, quantity)
            available = _available_lag_indices(product)
            if available.size < 2 and selected_lag is None:
                _warn(
                    f"Skipping case {case}: fewer than two lag bins have truth and MC PDFs",
                    messages,
                )
                continue
            if available.size == 0:
                _warn(f"Skipping case {case}: no usable lag bins", messages)
                continue
            loaded[case] = product
        except (OSError, KeyError, TypeError, ValueError) as error:
            _warn(f"Skipping case {case}: {error}", messages)

    if not loaded:
        raise ValueError("No usable saved increment-PDF products were found")
    reference = reference_case if reference_case in loaded else next(iter(loaded))
    reference_product = loaded[reference]
    reference_available = _available_lag_indices(reference_product)
    target_lag = (
        float(selected_lag)
        if selected_lag is not None
        else float(reference_product["lambda_centers"][reference_available[-2]])
    )

    cases: dict[str, Any] = {}
    selected_values: list[float] = []
    for case, product in loaded.items():
        available = _available_lag_indices(product)
        centers = np.asarray(product["lambda_centers"], dtype=float)
        lag_index = int(available[np.argmin(np.abs(centers[available] - target_lag))])
        lag_km = float(centers[lag_index])
        selected_values.append(lag_km)
        bins = np.asarray(product["x_edges"], dtype=float)
        truth = np.asarray(product["truth_density"][lag_index], dtype=float)
        if np.any(~np.isfinite(truth)) and np.any(np.isfinite(truth)):
            _warn(
                f"Case {case}: non-finite truth PDF bins at {lag_km:g} km are treated as zero",
                messages,
            )
        rayleigh = geometric_mean_normalized_rayleigh_pdf_on_bins(bins)
        try:
            denominator = wasserstein_from_binned_pdfs(rayleigh, truth, bins)
        except ValueError as error:
            _warn(f"Skipping case {case}: invalid truth PDF at {lag_km:g} km ({error})", messages)
            continue

        distances: list[float] = []
        member_warnings: list[str] = []
        if denominator <= denominator_tolerance:
            message = (
                f"Case {case}: Rayleigh-to-truth Wasserstein distance {denominator:.3g} "
                f"is at or below tolerance {denominator_tolerance:.3g}; normalized "
                "distances are NaN"
            )
            _warn(message, messages)
            member_warnings.append(message)
        for member, density in enumerate(np.asarray(product["mc_density"])[:, lag_index]):
            if not bool(product["valid"][member + 1, lag_index]):
                member_warnings.append(f"MC member {member} is invalid at the selected lag")
                distances.append(np.nan)
                continue
            if np.any(~np.isfinite(density)) and np.any(np.isfinite(density)):
                member_warnings.append(
                    f"MC member {member} has non-finite PDF bins treated as zero"
                )
            try:
                numerator = wasserstein_from_binned_pdfs(density, truth, bins)
                distances.append(
                    np.nan if denominator <= denominator_tolerance else numerator / denominator
                )
            except ValueError as error:
                member_warnings.append(f"MC member {member} skipped: {error}")
                distances.append(np.nan)
        finite = np.asarray(distances, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            p25, p50, p75 = np.percentile(finite, [25.0, 50.0, 75.0])
        else:
            p25 = p50 = p75 = np.nan
            _warn(f"Case {case}: no usable MC members at {lag_km:g} km", messages)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median_pdf = np.nanmedian(np.asarray(product["mc_density"])[:, lag_index], axis=0)
        try:
            median_pdf_distance = (
                np.nan
                if denominator <= denominator_tolerance
                else wasserstein_from_binned_pdfs(median_pdf, truth, bins) / denominator
            )
        except ValueError as error:
            median_pdf_distance = np.nan
            member_warnings.append(f"Median MC PDF skipped: {error}")
        cases[case] = {
            "label": labels[case],
            "selected_lag_km": lag_km,
            "selected_lag_index": lag_index,
            "rayleigh_truth_w1": float(denominator),
            "DW_R_by_mc": [_diagnostic_number(value) for value in distances],
            "DW_R_p25": _diagnostic_number(p25),
            "DW_R_p50": _diagnostic_number(p50),
            "DW_R_p75": _diagnostic_number(p75),
            "DW_R_median_pdf": _diagnostic_number(median_pdf_distance),
            "n_mc_total": len(distances),
            "n_mc_valid": int(finite.size),
            "warnings": member_warnings,
        }

    if not cases:
        raise ValueError("No cases had a usable truth PDF at the selected lag")
    if reference not in cases:
        replacement = next(iter(cases))
        _warn(
            f"Reference case {reference} was unusable at the selected lag; using {replacement}",
            messages,
        )
        reference = replacement
    if selected_values and not np.allclose(
        selected_values, selected_values[0], rtol=1e-6, atol=1e-6
    ):
        _warn(
            "Selected lag grids differ across cases; nearest lags were selected independently: "
            + ", ".join(
                f"{case}={entry['selected_lag_km']:.6g} km" for case, entry in cases.items()
            ),
            messages,
        )
    return {
        "quantity": quantity,
        "lag_selection": {
            "requested_lag_km": None if selected_lag is None else float(selected_lag),
            "default": default_lag,
            "target_lag_km": target_lag,
            "reference_case": reference,
        },
        "denominator_tolerance": float(denominator_tolerance),
        "cases": cases,
        "warnings": messages,
    }


def _case_order(diagnostics: Mapping[str, Any], order: str) -> list[str]:
    cases = list(diagnostics["cases"])
    if order == "supplied":
        return cases
    if order == "median":
        return sorted(
            cases,
            key=lambda case: (
                not np.isfinite(diagnostics["cases"][case]["DW_R_p50"]),
                (
                    np.inf
                    if not np.isfinite(diagnostics["cases"][case]["DW_R_p50"])
                    else diagnostics["cases"][case]["DW_R_p50"]
                ),
            ),
        )
    raise ValueError("case_order must be 'supplied' or 'median'")


def plot_increment_pdf_wasserstein_publication(
    results_by_case: Mapping[str, str | Path | Mapping[str, Any]],
    labels_by_case: Mapping[str, str] | None = None,
    p_value: float = 0.7,
    selected_lag: float | None = None,
    output_path: str | Path = "increment_pdf_wasserstein_p07.png",
    show_individual_bands: bool = False,
    *,
    diagnostics: dict[str, Any] | None = None,
    diagnostic_json_path: str | Path | None = None,
    diagnostic_csv_path: str | Path | None = None,
    denominator_tolerance: float = 1.0e-12,
    case_order: str = "supplied",
    all_errors_case: str = "001",
    difference_y_min: float = 1.0e-4,
    wasserstein_x_min: float = 1.0e-3,
    wasserstein_x_max: float | None = 0.2,
    show_difference_panel: bool = True,
    append_lag_to_filename: bool = False,
    dpi: int = 300,
) -> tuple[Path, dict[str, Any]]:
    """Create the publication figure entirely from saved increment-PDF products."""

    import matplotlib.pyplot as plt

    if wasserstein_x_max is not None and (
        not np.isfinite(wasserstein_x_max) or wasserstein_x_max <= 0.0
    ):
        raise ValueError("wasserstein_x_max must be a finite positive value or null")
    if not np.isfinite(wasserstein_x_min) or wasserstein_x_min <= 0.0:
        raise ValueError("wasserstein_x_min must be a finite positive value")
    if wasserstein_x_max is not None and wasserstein_x_min >= wasserstein_x_max:
        raise ValueError("wasserstein_x_min must be less than wasserstein_x_max")
    if not np.isfinite(difference_y_min) or difference_y_min <= 0.0:
        raise ValueError("difference_y_min must be a finite positive value")
    configured: dict[str, Any] = {}
    for raw_case, value in results_by_case.items():
        case = str(raw_case)
        source, embedded_label = _case_value(value)
        label = (labels_by_case or {}).get(case, embedded_label or case)
        configured[case] = {"path": source, "label": label}
    if diagnostics is None:
        diagnostics = compute_increment_pdf_wasserstein_diagnostics(
            configured,
            selected_lag=selected_lag,
            denominator_tolerance=denominator_tolerance,
            reference_case=all_errors_case,
        )

    loaded = {
        case: _load_publication_pdf_product(configured[case]["path"], "z_plus")
        for case in diagnostics["cases"]
        if not isinstance(configured[case]["path"], Mapping)
    }
    for case in diagnostics["cases"]:
        if isinstance(configured[case]["path"], Mapping):
            loaded[case] = dict(configured[case]["path"])

    if show_difference_panel:
        figure, (pdf_axis, difference_axis, distance_axis) = plt.subplots(
            1,
            3,
            figsize=(16.3, 5.2),
            gridspec_kw={"width_ratios": [1.45, 1.2, 1.0]},
        )
        figure.subplots_adjust(left=0.055, right=0.985, bottom=0.16, top=0.88, wspace=0.32)
    else:
        figure, (pdf_axis, distance_axis) = plt.subplots(
            1,
            2,
            figsize=(11.2, 5.2),
            gridspec_kw={"width_ratios": [1.45, 1.0]},
        )
        difference_axis = None
        figure.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.88, wspace=0.34)
    colors = plt.cm.tab10(np.linspace(0.0, 0.9, max(1, len(loaded))))
    color_by_case = {case: colors[index] for index, case in enumerate(loaded)}

    reference = diagnostics["lag_selection"]["reference_case"]
    reference_entry = diagnostics["cases"][reference]
    reference_product = loaded[reference]
    reference_index = reference_entry["selected_lag_index"]
    x_edges = reference_product["x_edges"]
    x = reference_product["x_centers"]
    truth = normalize_pdf(reference_product["truth_density"][reference_index], x_edges)
    rayleigh = geometric_mean_normalized_rayleigh_pdf_on_bins(x_edges)
    pdf_axis.plot(
        x,
        rayleigh,
        color="0.45",
        linestyle="--",
        linewidth=1.2,
        label="normalized Rayleigh",
        zorder=0,
    )
    pdf_axis.plot(
        x, truth, color="black", linestyle="-", linewidth=2.3, label="sampled truth", zorder=5
    )

    for case, product in loaded.items():
        entry = diagnostics["cases"][case]
        index = entry["selected_lag_index"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            lower, median, upper = np.nanpercentile(
                product["mc_density"][:, index], [25.0, 50.0, 75.0], axis=0
            )
        color = color_by_case[case]
        emphasized = case == all_errors_case
        if emphasized or show_individual_bands:
            selected = np.isfinite(lower) & np.isfinite(upper) & (upper > 0.0)
            pdf_axis.fill_between(
                product["x_centers"],
                lower,
                upper,
                where=selected,
                color=color,
                alpha=0.22 if emphasized else 0.09,
                linewidth=0,
                zorder=1,
            )
        selected = np.isfinite(median) & (median > 0.0)
        pdf_axis.plot(
            product["x_centers"][selected],
            median[selected],
            color=color,
            linestyle="--" if emphasized else "-",
            linewidth=2.2 if emphasized else 1.15,
            alpha=1.0 if emphasized else 0.82,
            label=entry["label"],
            zorder=4 if emphasized else 2,
        )
        if difference_axis is not None:
            case_truth = product["truth_density"][index]
            difference = symmetric_pdf_difference(median, case_truth, product["x_edges"])
            difference_selected = np.isfinite(difference) & (difference > 0.0)
            difference_axis.plot(
                product["x_centers"][difference_selected],
                difference[difference_selected],
                color=color,
                linestyle="--" if emphasized else "-",
                linewidth=2.2 if emphasized else 1.15,
                alpha=1.0 if emphasized else 0.82,
                zorder=4 if emphasized else 2,
            )

    lag_km = reference_entry["selected_lag_km"]
    lag_label = f"{lag_km:,.0f}" if lag_km >= 1_000 else f"{lag_km:.3g}"
    pdf_axis.set_title(rf"(a) $p={p_value:g}$, $\lambda_\perp={lag_label}\,\mathrm{{km}}$")
    pdf_axis.set_xlabel(
        r"$\delta z^+_\perp / \exp[\langle\ln(\delta z^+_\perp)\mid"
        r"\lambda_\perp\rangle]$"
    )
    pdf_axis.set_ylabel(r"PDF of normalized $\delta z^+_\perp$")
    pdf_axis.set_yscale("log")
    pdf_axis.set_xscale("linear")
    pdf_axis.grid(True, which="both", alpha=0.17)
    pdf_axis.legend(frameon=False, fontsize=8)

    if difference_axis is not None:
        difference_axis.set_title("(b) Symmetric PDF difference")
        difference_axis.set_xlabel(
            r"$\delta z^+_\perp / \exp[\langle\ln(\delta z^+_\perp)\mid"
            r"\lambda_\perp\rangle]$"
        )
        difference_axis.set_ylabel(
            r"$|\mathrm{PDF}_{MC}-\mathrm{PDF}_{true}|/"
            r"|\mathrm{PDF}_{MC}+\mathrm{PDF}_{true}|$"
        )
        difference_axis.set_xscale("linear")
        difference_axis.set_yscale("log")
        difference_axis.set_ylim(bottom=difference_y_min)
        difference_axis.grid(True, which="both", alpha=0.17)

    ordered = _case_order(diagnostics, case_order)
    for position, case in enumerate(ordered):
        entry = diagnostics["cases"][case]
        values = [entry["DW_R_p25"], entry["DW_R_p50"], entry["DW_R_p75"]]
        if not np.all(np.isfinite(values)):
            continue
        lower, median, upper = (float(value) for value in values)
        emphasized = case == all_errors_case
        if median <= 0.0:
            distance_axis.plot(
                wasserstein_x_min,
                position,
                marker="<",
                linestyle="none",
                color=color_by_case[case],
                markersize=7 if emphasized else 6,
                clip_on=False,
                zorder=3,
            )
            continue
        distance_axis.errorbar(
            median,
            position,
            xerr=[[median - lower], [upper - median]],
            fmt="o",
            color=color_by_case[case],
            markersize=7 if emphasized else 5,
            elinewidth=2.2 if emphasized else 1.4,
            capsize=3,
            zorder=3,
        )
    if wasserstein_x_max is None or wasserstein_x_max >= 1.0:
        distance_axis.axvline(1.0, color="0.35", linestyle="--", linewidth=1.2, zorder=0)
    distance_axis.set_yticks(range(len(ordered)))
    distance_axis.set_yticklabels([diagnostics["cases"][case]["label"] for case in ordered])
    distance_axis.invert_yaxis()
    distance_axis.set_xlabel(r"$D_W^R$")
    distance_panel_label = "c" if show_difference_panel else "b"
    distance_axis.set_title(f"({distance_panel_label}) PDF mismatch by error case")
    distance_axis.set_xscale("log")
    if wasserstein_x_max is not None:
        distance_axis.set_xlim(wasserstein_x_min, wasserstein_x_max)
    else:
        distance_axis.set_xlim(left=wasserstein_x_min)
    distance_axis.grid(True, axis="x", which="both", alpha=0.2)

    output = Path(output_path)
    if append_lag_to_filename:
        filename_lag = lag_label.replace(",", "")
        output = output.with_name(f"{output.stem}-lambda-{filename_lag}{output.suffix}")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    if diagnostic_json_path is not None:
        write_increment_pdf_wasserstein_json(diagnostics, diagnostic_json_path)
    if diagnostic_csv_path is not None:
        write_increment_pdf_wasserstein_csv(diagnostics, diagnostic_csv_path)
    return output, diagnostics


def write_increment_pdf_wasserstein_json(
    diagnostics: Mapping[str, Any], output_path: str | Path
) -> Path:
    def json_safe(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        return value

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(json_safe(diagnostics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def write_increment_pdf_wasserstein_csv(
    diagnostics: Mapping[str, Any], output_path: str | Path
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "case",
        "label",
        "lag_km",
        "DW_R_p25",
        "DW_R_p50",
        "DW_R_p75",
        "DW_R_median_pdf",
        "rayleigh_truth_w1",
        "n_mc_valid",
        "n_mc_total",
    ]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for case, entry in diagnostics["cases"].items():
            writer.writerow(
                {
                    "case": case,
                    "label": entry["label"],
                    "lag_km": entry["selected_lag_km"],
                    "DW_R_p25": entry["DW_R_p25"],
                    "DW_R_p50": entry["DW_R_p50"],
                    "DW_R_p75": entry["DW_R_p75"],
                    "DW_R_median_pdf": entry["DW_R_median_pdf"],
                    "rayleigh_truth_w1": entry["rayleigh_truth_w1"],
                    "n_mc_valid": entry["n_mc_valid"],
                    "n_mc_total": entry["n_mc_total"],
                }
            )
    return output


def plot_increment_pdf_wasserstein_from_config(
    config_path: str | Path,
    *,
    panels_a_c: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Read a lightweight YAML plot config and create figure plus diagnostics."""

    import yaml

    path = Path(config_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("Wasserstein publication config must be a YAML mapping")
    cases = config.get("cases")
    if not isinstance(cases, Mapping) or not cases:
        raise ValueError("Wasserstein publication config requires a nonempty 'cases' mapping")
    resolved_cases: dict[str, Any] = {}
    for raw_case, raw_entry in cases.items():
        case = str(raw_case)
        if not isinstance(raw_entry, Mapping) or "path" not in raw_entry:
            raise ValueError(f"cases.{case} must be a mapping containing path")
        product_path = Path(str(raw_entry["path"])).expanduser()
        if not product_path.is_absolute():
            # Repository configs conventionally use paths relative to the working directory.
            product_path = Path.cwd() / product_path
        resolved_cases[case] = {
            "path": product_path,
            "label": str(raw_entry.get("label", case)),
        }
    output = config.get("output", {})
    if not isinstance(output, Mapping):
        raise ValueError("output must be a mapping")
    figure_path = output.get("figure", "increment_pdf_wasserstein_p07.png")
    json_path = output.get("json")
    csv_path = output.get("csv")
    return plot_increment_pdf_wasserstein_publication(
        resolved_cases,
        p_value=float(config.get("p_value", 0.7)),
        selected_lag=(
            None if config.get("selected_lag_km") is None else float(config["selected_lag_km"])
        ),
        output_path=figure_path,
        show_individual_bands=bool(config.get("show_individual_bands", False)),
        diagnostic_json_path=json_path,
        diagnostic_csv_path=csv_path,
        denominator_tolerance=float(config.get("denominator_tolerance", 1.0e-12)),
        case_order=str(config.get("case_order", "supplied")),
        all_errors_case=str(config.get("all_errors_case", "001")),
        difference_y_min=float(config.get("difference_y_min", 1.0e-4)),
        wasserstein_x_min=float(config.get("wasserstein_x_min", 1.0e-3)),
        wasserstein_x_max=(
            None
            if config.get("wasserstein_x_max", 0.2) is None
            else float(config.get("wasserstein_x_max", 0.2))
        ),
        show_difference_panel=not panels_a_c,
        append_lag_to_filename=bool(config.get("append_lag_to_filename", True)),
        dpi=int(config.get("dpi", 300)),
    )
