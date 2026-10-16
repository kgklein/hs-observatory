"""Local logarithmic slopes from saved structure functions; no upstream evaluation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import warnings

import h5py
import numpy as np
import yaml

from hs_observatory.analysis.structure_function import (
    P_MODEL_RUN_COLORS,
    _ORDER_MARKERS,
    _ensemble_interval,
    _summary_result,
)

CASCADE_CAPTION = (
    "Vertical marks: nominal cell sizes L₀/2ᵐ; compact support widths are twice these sizes. "
    "These are contextual scales, not asserted inertial-range boundaries."
)


def p_model_exponents(orders: np.ndarray, p: float) -> np.ndarray:
    """Analytic reference exponents, independent of measured local slopes."""
    if not np.isfinite(p) or not 0.5 <= p <= 1.0:
        raise ValueError("p must lie in [0.5, 1.0]")
    orders = np.asarray(orders, dtype=float)
    return 1.0 - np.log2(p ** (orders / 3.0) + (1.0 - p) ** (orders / 3.0))


def local_log_slopes(
    centers: np.ndarray,
    values: np.ndarray,
    *,
    counts: np.ndarray | None = None,
    method: str = "three-point",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return coordinates, slopes, and source validity; preserve all local gaps.

    Values have shape (member, order, bin); counts have shape (member, bin).
    A neighborhood must contain only valid samples and strictly increasing centers.
    """
    centers = np.asarray(centers, dtype=float)
    values = np.asarray(values, dtype=float)
    if centers.ndim != 1 or values.ndim != 3 or values.shape[-1] != centers.size:
        raise ValueError("expected centers (bin,) and S_n (member, order, bin)")
    if method not in {"three-point", "adjacent"}:
        raise ValueError("local slope method must be three-point or adjacent")
    valid_centers = np.isfinite(centers) & (centers > 0)
    valid = np.isfinite(values) & (values > 0) & valid_centers
    if counts is not None:
        counts = np.asarray(counts, dtype=float)
        if counts.shape != (values.shape[0], centers.size):
            raise ValueError("counts must have shape (member, bin)")
        valid &= (np.isfinite(counts) & (counts > 0))[:, None, :]
    log_x = np.full(centers.shape, np.nan)
    np.log(centers, out=log_x, where=valid_centers)
    log_y = np.full(values.shape, np.nan)
    np.log(values, out=log_y, where=valid)
    if method == "adjacent":
        coordinates = np.exp((log_x[:-1] + log_x[1:]) / 2)
        slopes = np.full((*values.shape[:-1], max(0, centers.size - 1)), np.nan)
        for k in range(centers.size - 1):
            if not np.all(valid_centers[k : k + 2]) or centers[k + 1] <= centers[k]:
                coordinates[k] = np.nan
                continue
            good = np.all(valid[..., k : k + 2], axis=-1)
            difference = (log_y[..., k + 1] - log_y[..., k]) / (log_x[k + 1] - log_x[k])
            slopes[..., k] = np.where(good, difference, np.nan)
    else:
        coordinates = np.where(valid_centers, centers, np.nan)
        slopes = np.full(values.shape, np.nan)
        for k in range(1, centers.size - 1):
            window = slice(k - 1, k + 2)
            if not np.all(valid_centers[window]) or not np.all(np.diff(centers[window]) > 0):
                continue
            x = log_x[window]
            centered_x = x - x.mean()
            good = np.all(valid[..., window], axis=-1)
            slope = np.sum(log_y[..., window] * centered_x, axis=-1) / np.sum(centered_x**2)
            slopes[..., k] = np.where(good, slope, np.nan)
    return coordinates, slopes, valid


def derive_product(product: dict, quantity: str, orders: list[int], method: str) -> dict:
    """Derive slopes for one stored product, retaining member-level arrays."""
    source_orders = list(product["orders"])
    values = np.asarray(product[quantity]["S_n"], dtype=float)
    if values.ndim != 3 or values.shape[0] < 1 or values.shape[1] != len(source_orders):
        raise ValueError("individual S_n curves must have shape (member, order, bin)")
    indices = [source_orders.index(order) for order in orders]
    centers = np.asarray(product["lambda_bin_centers_km"], dtype=float)
    coordinates, slopes, valid = local_log_slopes(
        centers,
        values[:, indices, :],
        counts=np.asarray(product["counts"]),
        method=method,
    )
    return {
        "source_centers_km": centers,
        "source_edges_km": np.asarray(product.get("lambda_bin_edges_km", []), dtype=float),
        "source_valid": valid,
        "coordinates_km": coordinates,
        "slopes": slopes,
        "slope_valid": np.isfinite(slopes),
    }


def cascade_context(path: str | Path) -> dict:
    """Read explicit field parameters only; never instantiate a field model."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    field = config["field"]
    domain, turbulence = field["domain"], field["turbulence"]
    lengths = np.asarray(domain["L"], dtype=float)
    injection = int(turbulence["injection_scale_index"])
    maximum = int(turbulence["n_scales"])
    h = float(turbulence["spectral_exponent_h"])
    if (
        field["model"] != "malara_turbulence"
        or turbulence.get("geometry", "isotropic") != "isotropic"
        or domain.get("units", "km") != "km"
        or lengths.shape != (3,)
        or not np.all(np.isfinite(lengths) & (lengths > 0))
        or not np.all(lengths == lengths[0])
        or injection != turbulence["injection_scale_index"]
        or maximum != turbulence["n_scales"]
        or not 0 <= injection <= maximum
        or not np.isfinite(h)
    ):
        raise ValueError(f"{path}: cascade marks require valid isotropic Malara parameters in km")
    indices = sorted(set(range(injection, maximum + 1, 3)) | {maximum})
    return {
        "generation_config": str(Path(path).resolve()),
        "domain_lengths_km": lengths.tolist(),
        "injection_scale_index": injection,
        "maximum_scale_index": maximum,
        "spectral_exponent_h": h,
        "mark_indices": indices,
        "cell_sizes_km": [float(lengths[0] / 2.0**m) for m in indices],
        "caption": CASCADE_CAPTION,
    }


def load_local_cases(config: dict, quantity: str, method: str, show_scales: bool) -> list[dict]:
    entries = config.get("cases")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("local diagnostic requires a nonempty cases mapping")
    cases = []
    seen = set()
    for name, entry in entries.items():
        p = float(entry["p_value"])
        if p in seen:
            raise ValueError("duplicate p_value in local diagnostic")
        seen.add(p)
        path = Path(entry["path"]).expanduser().resolve()
        result = _summary_result(path)
        truth = result["products"]["truth"]
        observed = result["products"].get("observed", {})
        has_mc = (
            isinstance(observed.get(quantity), dict)
            and "S_n" in observed[quantity]
            and np.asarray(observed[quantity]["S_n"]).ndim == 3
        )
        orders = list(truth["orders"])
        if has_mc:
            orders = [n for n in orders if n in observed["orders"]]
        maximum = config.get("max_order", 6)
        orders = [n for n in orders if maximum is None or n <= int(maximum)]
        if not orders:
            raise ValueError(f"case {name}: no requested orders available")
        case = {
            "name": str(name),
            "p_value": p,
            "orders": np.asarray(orders, dtype=int),
            "source_path": str(path),
            "theory": p_model_exponents(orders, p),
            "truth": derive_product(truth, quantity, orders, method),
        }
        if case["truth"]["slopes"].shape[0] != 1:
            raise ValueError(f"case {name}: sampled truth must contain one curve per order")
        if has_mc:
            mc = derive_product(observed, quantity, orders, method)
            mc["p05"], mc["median"], mc["p95"] = _ensemble_interval(mc["slopes"])
            mc["finite_members"] = np.sum(mc["slope_valid"], axis=0)
            case["mc"] = mc
        else:
            case["mc_limitation"] = "Individual MC curves unavailable; MC slopes/bands omitted."
            warnings.warn(f"case {name}: {case['mc_limitation']}", stacklevel=2)
        if show_scales:
            if not entry.get("generation_config"):
                raise ValueError(f"case {name}: --show-cascade-scales requires generation_config")
            case["cascade"] = cascade_context(Path(entry["generation_config"]).expanduser())
        cases.append(case)
    return sorted(cases, key=lambda case: case["p_value"], reverse=True)


def write_local_results(path: Path, cases: list[dict], metadata: dict) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = 1
        handle.attrs["metadata_json"] = json.dumps(metadata)
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                handle.attrs[key] = value
        for index, case in enumerate(cases):
            group = handle.create_group(f"cases/{index}")
            for key in ("name", "p_value", "source_path", "mc_limitation"):
                if key in case:
                    group.attrs[key] = case[key]
            for key in ("orders", "theory"):
                group.create_dataset(key, data=case[key])
            if "cascade" in case:
                group.attrs["cascade_json"] = json.dumps(case["cascade"])
            for source in ("truth", "mc"):
                if source in case:
                    product = group.create_group(source)
                    for key, value in case[source].items():
                        product.create_dataset(key, data=value, compression="gzip")


def _mark_scales(axis, case: dict, limits: tuple[float, float], *, label: bool = True) -> None:
    if "cascade" not in case:
        return
    context = case["cascade"]
    for m, scale in zip(context["mark_indices"], context["cell_sizes_km"], strict=True):
        if limits[0] <= scale <= limits[1]:
            axis.axvline(scale, color="0.5", alpha=0.18, linewidth=0.7, zorder=0)
            if label:
                axis.text(
                    scale,
                    0.98,
                    f"m={m}",
                    transform=axis.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=7,
                    color="0.4",
                )


def _plot_order(axis, case: dict, index: int, color: str, *, show_theory: bool) -> None:
    order = int(case["orders"][index])
    truth = case["truth"]
    if "sampling_sd" in case:
        spread = case["sampling_sd"][index]
        axis.fill_between(
            truth["coordinates_km"],
            truth["slopes"][0, index] - spread,
            truth["slopes"][0, index] + spread,
            color=color,
            alpha=0.12,
            linewidth=0,
        )
    axis.plot(
        truth["coordinates_km"],
        truth["slopes"][0, index],
        color=color,
        marker=_ORDER_MARKERS[(order - 1) % len(_ORDER_MARKERS)],
        markersize=4,
        linewidth=1.4,
    )
    if "mc" in case:
        mc = case["mc"]
        axis.fill_between(
            mc["coordinates_km"],
            mc["p05"][index],
            mc["p95"][index],
            color=color,
            alpha=0.12,
            linewidth=0,
        )
        axis.plot(
            mc["coordinates_km"], mc["median"][index], color=color, linestyle="--", linewidth=1.2
        )
    if show_theory:
        axis.axhline(
            case["theory"][index], color=color, linestyle="--", alpha=0.45, linewidth=0.8, zorder=0
        )


def plot_local_diagnostics(
    cases: list[dict],
    prefix: Path,
    *,
    dpi: int,
    show_theory: bool,
    curve_label: str = "sampled truth",
    focused: bool = True,
    caption_extra: str = "",
) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    centers = np.concatenate(
        [
            case[source]["source_centers_km"]
            for case in cases
            for source in ("truth", "mc")
            if source in case
        ]
    )
    valid = centers[np.isfinite(centers) & (centers > 0)]
    if valid.size < 2 or valid.min() == valid.max():
        raise ValueError("local diagnostics require at least two distinct positive separations")
    limits = (float(valid.min()), float(valid.max()))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    orders = sorted({int(n) for case in cases for n in case["orders"]})
    order_colors = {n: colors[i % len(colors)] for i, n in enumerate(orders)}
    styles = [Line2D([], [], color="0.2", marker="o", label=curve_label)]
    if any("sampling_sd" in case for case in cases):
        from matplotlib.patches import Patch

        styles.append(
            Patch(color="0.5", alpha=0.12, label="spatial sampling variability: ±1 batch SD")
        )
    if any("mc" in case for case in cases):
        styles.append(Line2D([], [], color="0.2", linestyle="--", label="MC median (bands: 5–95%)"))
    if show_theory:
        styles.append(
            Line2D([], [], color="0.5", linestyle="--", linewidth=0.8, label="p-model reference")
        )
    styles.append(Line2D([], [], color="0.4", linestyle=":", label="slope = 1"))
    figure, axes = plt.subplots(
        len(cases), 1, figsize=(10, 3.2 * len(cases) + 1), sharex=True, squeeze=False
    )
    for axis, case in zip(axes[:, 0], cases, strict=True):
        for i, order in enumerate(case["orders"]):
            _plot_order(axis, case, i, order_colors[int(order)], show_theory=show_theory)
        axis.set_title(f"p = {case['p_value']:g}", loc="left")
        axis.set_ylabel(r"$\xi_n^{\mathrm{local}}$")
        axis.set_xscale("log")
        axis.set_xlim(limits)
        axis.axhline(1, color="0.4", linestyle=":", linewidth=1)
        axis.grid(alpha=0.15)
        _mark_scales(axis, case, limits)
        if "mc_limitation" in case:
            axis.text(
                0.99,
                0.03,
                "MC curves unavailable",
                transform=axis.transAxes,
                ha="right",
                fontsize=8,
            )
    order_handles = [
        Line2D(
            [],
            [],
            color=order_colors[n],
            marker=_ORDER_MARKERS[(n - 1) % len(_ORDER_MARKERS)],
            label=f"n = {n}",
        )
        for n in orders
    ]
    figure.legend(
        handles=order_handles,
        ncols=min(6, len(orders)),
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
    )
    axes[-1, 0].set_xlabel(r"$\lambda$ [km]")
    figure.legend(handles=styles, loc="upper center", ncols=len(styles), fontsize=8)
    caption = CASCADE_CAPTION if any("cascade" in case for case in cases) else ""
    caption = "\n".join(text for text in (caption, caption_extra) if text)
    figure.text(0.5, 0.012, caption, ha="center", fontsize=8, wrap=True)
    figure.tight_layout(rect=(0, 0.05 if caption else 0.01, 1, 0.93))
    outputs = []
    try:
        for suffix in (".png", ".pdf"):
            path = Path(f"{prefix}{suffix}")
            figure.savefig(path, dpi=dpi)
            outputs.append(path)
    finally:
        plt.close(figure)

    if not focused:
        return outputs

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, order in zip(axes.flat, (2, 3, 4, 6), strict=True):
        for ci, case in enumerate(cases):
            if order in case["orders"]:
                i = list(case["orders"]).index(order)
                _plot_order(
                    axis,
                    case,
                    i,
                    P_MODEL_RUN_COLORS.get(case["p_value"], colors[ci % len(colors)]),
                    show_theory=show_theory,
                )
        axis.set_title(f"n = {order}")
        axis.set_xscale("log")
        axis.set_xlim(limits)
        axis.axhline(1, color="0.4", linestyle=":", linewidth=1)
        axis.grid(alpha=0.15)
        # Shared marks are valid only when the supplied cascade contexts agree.
        contexts = [case.get("cascade") for case in cases]
        if all(context is not None for context in contexts):
            geometries = [(c["domain_lengths_km"], c["mark_indices"]) for c in contexts]
            if all(geometry == geometries[0] for geometry in geometries):
                _mark_scales(axis, cases[0], limits)
        axis.set_ylabel(r"$\xi_n^{\mathrm{local}}$")
    for axis in axes[-1]:
        axis.set_xlabel(r"$\lambda$ [km]")
    case_handles = [
        Line2D(
            [],
            [],
            color=P_MODEL_RUN_COLORS.get(c["p_value"], colors[i % len(colors)]),
            label=f"p = {c['p_value']:g}",
        )
        for i, c in enumerate(cases)
    ]
    figure.legend(handles=case_handles + styles, loc="upper center", ncols=4, fontsize=8)
    figure.text(0.5, 0.012, caption, ha="center", fontsize=8, wrap=True)
    figure.tight_layout(rect=(0, 0.06 if caption else 0.01, 1, 0.91))
    try:
        for suffix in (".png", ".pdf"):
            path = Path(f"{prefix}_focused{suffix}")
            figure.savefig(path, dpi=dpi)
            outputs.append(path)
    finally:
        plt.close(figure)
    return outputs


def local_slopes_from_config(
    config_path: str | Path,
    *,
    method: str = "three-point",
    output_prefix: str | Path | None = None,
    show_cascade_scales: bool = False,
    show_p_model_override: bool | None = None,
) -> list[Path]:
    """Write derived HDF5 and diagnostic figures from saved publication JSONs."""
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("local diagnostic config must be a YAML mapping")
    quantity = str(config.get("quantity", "z_plus"))
    cases = load_local_cases(config, quantity, method, show_cascade_scales)
    default = Path(config.get("output", {}).get("figure", "structure_function.png"))
    prefix = (
        Path(output_prefix).expanduser()
        if output_prefix is not None
        else default.with_name(default.stem + "_local_slopes")
    )
    prefix = prefix.resolve()
    targets = [Path(f"{prefix}{suffix}") for suffix in (".h5", ".png", ".pdf")]
    targets += [Path(f"{prefix}_focused{suffix}") for suffix in (".png", ".pdf")]
    sources = {Path(case["source_path"]) for case in cases} | {Path(config_path).resolve()}
    sources.add(default.expanduser().resolve())
    sources.update(
        Path(case["cascade"]["generation_config"]) for case in cases if "cascade" in case
    )
    if any(target in sources for target in targets):
        raise ValueError("output prefix would overwrite an original input or publication figure")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    show_theory = (
        bool(config.get("show_p_model", True))
        if show_p_model_override is None
        else show_p_model_override
    )
    metadata = {
        "estimator": method,
        "quantity": quantity,
        "config_path": str(Path(config_path).resolve()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "show_p_model": show_theory,
        "ensemble_policy": "slopes per realization, then finite-member 5th/50th/95th percentiles",
        "edge_policy": "NaN endpoints for three-point; no extrapolation or gap bridging",
        "show_cascade_scales": show_cascade_scales,
    }
    outputs = plot_local_diagnostics(
        cases, prefix, dpi=int(config.get("dpi", 300)), show_theory=show_theory
    )
    write_local_results(targets[0], cases, metadata)
    return [targets[0], *outputs]
