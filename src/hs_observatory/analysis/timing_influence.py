"""Polyhedron influence diagnostics for absolute-divergence, zero-clock-error fits."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from hs_observatory.analysis.magnetic_gradient import (
    apply_gradient_response,
    geometry_diagnostics,
    geometry_is_static,
)
from hs_observatory.analysis.timing_calibrate import (
    _store,
    solve_timing_with_variance,
    timing_jacobian,
)


def decompose_timing_influence(
    ideal_divergence: np.ndarray, jacobian: np.ndarray, variance: np.ndarray, sigma_t: float
) -> dict[str, Any]:
    """Resolve J.T W d into signed per-polyhedron clock contributions.

    Use the unchanged absolute-divergence solver for row selection and a reference
    correction. Solve the normal operator through an SVD of the augmented QR
    factor, avoiding an explicit inverse and an SVD of the squared-condition H.
    """
    fit = solve_timing_with_variance(ideal_divergence, jacobian, variance, sigma_t)
    valid = fit["fit_valid"]
    sigma = np.broadcast_to(np.sqrt(variance), ideal_divergence.shape)[valid]
    weighted_j = jacobian[valid] / sigma[:, None]
    weighted_d = ideal_divergence[valid] / sigma
    g = weighted_j.T @ weighted_d
    g_polyhedron = np.zeros((jacobian.shape[-1], ideal_divergence.shape[1]))
    for polyhedron in range(ideal_divergence.shape[1]):
        mask = valid[:, polyhedron]
        if np.any(mask):
            if np.ndim(variance) == 1:
                g_polyhedron[:, polyhedron] = (
                    jacobian[mask, polyhedron].T
                    @ ideal_divergence[mask, polyhedron]
                    / variance[polyhedron]
                )
            else:
                g_polyhedron[:, polyhedron] = jacobian[mask, polyhedron].T @ (
                    ideal_divergence[mask, polyhedron] / variance[mask, polyhedron]
                )
    n_parameters = jacobian.shape[-1]
    hessian = weighted_j.T @ weighted_j + np.eye(n_parameters) / sigma_t**2
    design = np.vstack((weighted_j, np.eye(n_parameters) / sigma_t))
    triangular = np.linalg.qr(design, mode="r")
    _, singular, right = np.linalg.svd(triangular, full_matrices=False)
    keep = singular > np.finfo(float).eps * max(design.shape) * singular[0]
    projected = right @ np.column_stack((g, g_polyhedron))
    projected[keep] /= singular[keep, None] ** 2
    projected[~keep] = 0
    contributions = -right.T @ projected
    predicted, per_polyhedron = contributions[:, 0], contributions[:, 1:]
    # Scale absolute tolerance by the sum of magnitudes to allow cancellation.
    g_tolerance = 1e-12 * max(1.0, float(np.max(np.abs(g_polyhedron).sum(axis=1))))
    c_tolerance = 1e-12 * max(1.0, float(np.max(np.abs(per_polyhedron).sum(axis=1))))
    checks = {
        "g_sum_matches": bool(
            np.allclose(g_polyhedron.sum(axis=1), g, rtol=1e-9, atol=g_tolerance)
        ),
        "c_sum_matches": bool(
            np.allclose(per_polyhedron.sum(axis=1), predicted, rtol=1e-9, atol=c_tolerance)
        ),
        "solver_matches": bool(
            np.allclose(predicted, fit["correction_s"], rtol=1e-8, atol=c_tolerance)
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"Timing influence numerical consistency checks failed: {checks}")
    ranking = np.argsort(-np.abs(per_polyhedron), axis=1, kind="stable")
    ranked = np.take_along_axis(per_polyhedron, ranking, axis=1)
    return {
        "g": g,
        "g_polyhedron": g_polyhedron,
        "H": hessian,
        "predicted_correction_s": predicted,
        "polyhedron_correction_s": per_polyhedron,
        "solver_correction_s": fit["correction_s"],
        "fit_valid": valid,
        "ranked_polyhedron_indices": ranking,
        "cumulative_signed_s": np.cumsum(ranked, axis=1),
        "cumulative_absolute_s": np.cumsum(np.abs(ranked), axis=1),
        "g_sum_error": g_polyhedron.sum(axis=1) - g,
        "c_sum_error_s": per_polyhedron.sum(axis=1) - predicted,
        "solver_error_s": predicted - fit["correction_s"],
        **checks,
    }


def influence_geometry(result: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Use repository geometry and affine-fit RMS conventions, on ideal samples."""
    series = {}
    counts, scales, conditions, residuals = [], [], [], []
    for (name, indices), response in zip(result["selected"].items(), result["responses"]):
        points = result["positions"][:, indices]
        static = geometry_is_static(points)
        records = [geometry_diagnostics(sample) for sample in (points[:1] if static else points)]
        scale = np.array([record.characteristic_size for record in records])
        condition = np.array([record.condition_number for record in records])
        entry = {
            "volumetric_tensor": np.array([record.volumetric_tensor for record in records]),
            "characteristic_size": scale,
            "condition_number": condition,
            "static_geometry": static,
        }
        residual_rms = np.nan
        if len(indices) > 4:
            field = result["ideal_B"][:, indices]
            gradient = apply_gradient_response(response, field)
            centered = points - points.mean(axis=1, keepdims=True)
            fitted = field.mean(axis=1, keepdims=True) + np.einsum(
                "tnj,tij->tni", centered, gradient
            )
            # Same sqrt(mean_spacecraft(sum_component(residual**2))) as GradientEstimate.
            rms = np.sqrt(np.mean(np.sum((field - fitted) ** 2, axis=-1), axis=1))
            entry["affine_rms_residual"] = rms
            finite = rms[np.isfinite(rms)]
            if finite.size:
                residual_rms = float(np.sqrt(np.mean(finite**2)))
        series[name] = entry
        counts.append(len(indices))
        scales.append(np.median(scale))
        conditions.append(np.median(condition))
        residuals.append(residual_rms)
    return series, {
        "number_of_spacecraft": np.asarray(counts),
        "characteristic_size_km": np.asarray(scales),
        "condition_number": np.asarray(conditions),
        "affine_rms_residual_nT": np.asarray(residuals),
        "sigma_d": (
            np.sqrt(result["variance"])
            if np.ndim(result["variance"]) == 1
            else np.median(np.sqrt(result["variance"]), axis=0)
        ),
    }


def _aggregate(values: np.ndarray, contributions: np.ndarray, *, discrete: bool = False) -> dict:
    finite = np.isfinite(values)
    if discrete:
        edges = np.arange(3.5, 10.0, 1.0)
    elif np.any(finite):
        # Four equal-population ranges in the selected geometry; duplicate edges collapse.
        edges = np.unique(np.quantile(values[finite], [0, 0.25, 0.5, 0.75, 1]))
        if len(edges) == 1:
            edges = np.array([edges[0], edges[0]])
    else:
        edges = np.array([0.0, 0.0])
    n_bins = len(edges) - 1
    membership = np.full(len(values), n_bins, dtype=int)  # Last bin: nonfinite / not applicable.
    membership[finite] = np.searchsorted(edges[1:-1], values[finite], side="right")
    signed = np.zeros((contributions.shape[0], n_bins + 1))
    absolute = np.zeros_like(signed)
    counts = np.zeros(n_bins + 1, dtype=int)
    for group in range(n_bins + 1):
        selected = membership == group
        signed[:, group] = contributions[:, selected].sum(axis=1)
        absolute[:, group] = np.abs(contributions[:, selected]).sum(axis=1)
        counts[group] = selected.sum()
    return {
        "edges": edges,
        "polyhedron_bin": membership,
        "counts": counts,
        "signed_s": signed,
        "absolute_s": absolute,
    }


def compute_timing_influence(result: Mapping[str, Any]) -> dict[str, Any]:
    jacobian = timing_jacobian(
        result["time"],
        result["ideal_B"],
        list(result["selected"].values()),
        result["responses"],
        result["node_indices"],
    )
    influence = decompose_timing_influence(
        result["ideal_divergence"], jacobian, result["variance"], result["sigma_t"]
    )
    del jacobian
    rms_offset_s = float(np.sqrt(np.mean(influence["predicted_correction_s"] ** 2)))
    print(
        f"Timing influence RMS recovered offset (across Nodes; Hub excluded): "
        f"{rms_offset_s:.6g} s ({rms_offset_s * 1000:.6g} ms)",
        flush=True,
    )
    geometry, summary = influence_geometry(result)
    influence["geometry"] = geometry
    influence["polyhedra"] = summary
    influence["aggregates"] = {
        name: _aggregate(
            summary[name],
            influence["polyhedron_correction_s"],
            discrete=name == "number_of_spacecraft",
        )
        for name in (
            "number_of_spacecraft",
            "characteristic_size_km",
            "condition_number",
            "affine_rms_residual_nT",
        )
    }
    return influence


def write_timing_influence(handle: h5py.File, result: Mapping[str, Any]) -> None:
    influence = result["influence"]
    settings = result["settings"]
    group = handle.create_group("timing_influence")
    group.attrs.update(
        target="absolute ideal divergence",
        timing_offsets="zero; Hub fixed",
        axes="Node, polyhedron",
        geometry_summary="median over time; L=2*sqrt(max eigenvalue(R))",
        affine_summary="sqrt(mean_time(affine_rms_residual**2)); only N>4",
        bins="quartile edges; last bin is nonfinite/not applicable; N bins are integers 4..9",
        decomposition="additive contributions with the full H fixed; not leave-one-out refits",
    )
    group.create_dataset(
        "polyhedron_id", data=np.asarray(list(result["selected"]), dtype=h5py.string_dtype())
    )
    for name, value in influence.items():
        if name in {"geometry", "polyhedra", "aggregates"}:
            continue
        units = (
            "s"
            if name.endswith("_s")
            else {"g": "1/s", "g_polyhedron": "1/s", "g_sum_error": "1/s", "H": "1/s^2"}.get(
                name, "1"
            )
        )
        _store(group, name, value, settings, units)
    polyhedra = group.create_group("polyhedra")
    for name, value in influence["polyhedra"].items():
        units = {
            "characteristic_size_km": "km",
            "affine_rms_residual_nT": "nT",
            "sigma_d": "nT/km",
        }.get(name, "1")
        _store(polyhedra, name, value, settings, units)
    for name, entry in influence["geometry"].items():
        geometry = handle[f"configurations/{name}"].create_group("influence_geometry")
        for key, value in entry.items():
            units = {
                "volumetric_tensor": "km^2",
                "characteristic_size": "km",
                "affine_rms_residual": "nT",
            }.get(key, "1")
            _store(geometry, key, value, settings, units)
    aggregates = group.create_group("aggregates")
    for name, entry in influence["aggregates"].items():
        target = aggregates.create_group(name)
        for key, value in entry.items():
            units = "s" if key.endswith("_s") else "1"
            if key == "edges":
                units = {"characteristic_size_km": "km", "affine_rms_residual_nT": "nT"}.get(
                    name, "1"
                )
            _store(target, key, value, settings, units)


def plot_timing_influence(
    product_path: str | Path, output_path: str | Path, *, dpi: int = 300
) -> list[Path]:
    """Export heat maps, cumulative curves, binned summaries, and top-ten CSV."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm

    from hs_observatory.visualization import SPACECRAFT_COLORS

    with h5py.File(product_path, "r") as handle:
        if "timing_influence" not in handle:
            return []
        if not handle["metadata"].attrs.get("complete", False):
            raise ValueError("Timing-calibration product is incomplete")
        source = handle["timing_influence"]
        names = source["polyhedron_id"].asstr()[...]
        nodes = handle["spacecraft/node_id"].asstr()[...]
        g = source["g_polyhedron"][...]
        c = source["polyhedron_correction_s"][...] * 1000
        ranking = source["ranked_polyhedron_indices"][...]
        summaries = {name: ds[...] for name, ds in source["polyhedra"].items()}
        aggregates = {
            name: {key: ds[...] for key, ds in entry.items()}
            for name, entry in source["aggregates"].items()
        }
        members = [
            ", ".join(handle[f"configurations/{name}/spacecraft_ids"].asstr()[...])
            for name in names
        ]
    base = Path(output_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs = []

    def save(fig, suffix):
        path = base.with_name(f"{base.stem}_influence_{suffix}{base.suffix}")
        try:
            fig.savefig(path, dpi=dpi)
        finally:
            plt.close(fig)
        outputs.append(path)

    conditions = summaries["condition_number"]
    valid_conditions = np.isfinite(conditions) & (conditions > 0)
    finite_conditions = conditions[valid_conditions]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    if finite_conditions.size:
        lower, upper = finite_conditions.min(), finite_conditions.max()
        if lower == upper:
            lower, upper = lower / np.sqrt(10), upper * np.sqrt(10)
        bins = np.geomspace(lower, upper, 31)
        ax.hist(finite_conditions, bins=bins, edgecolor="black", linewidth=0.5)
        ax.set_xscale("log")
    ax.set(
        xlabel="cond(R) (median over time)",
        ylabel="Number of polyhedra",
        title=f"Condition numbers of retained polyhedra (N={len(conditions)})",
    )
    omitted = int(np.count_nonzero(~valid_conditions))
    if omitted:
        ax.text(
            0.98,
            0.98,
            f"Nonfinite or nonpositive: {omitted}",
            transform=ax.transAxes,
            ha="right",
            va="top",
        )
    ax.grid(axis="y", alpha=0.2)
    save(fig, "condition_numbers")

    order = np.argsort(summaries["number_of_spacecraft"], kind="stable")
    n_order = summaries["number_of_spacecraft"][order]
    fig, axes = plt.subplots(2, 1, figsize=(15, 7), constrained_layout=True)
    for ax, values, label in zip(axes, (g, c), (r"$g_{\alpha,P}$ [1/s]", r"$c_{\alpha,P}$ [ms]")):
        maximum = max(float(np.max(np.abs(values))), np.finfo(float).tiny)
        norm = SymLogNorm(linthresh=maximum * 1e-3, vmin=-maximum, vmax=maximum)
        display = ax.imshow(values[:, order], aspect="auto", cmap="RdBu_r", norm=norm)
        ax.set_yticks(np.arange(len(nodes)), nodes)
        centers, labels = [], []
        for size in np.unique(n_order):
            locations = np.flatnonzero(n_order == size)
            centers.append(locations.mean())
            labels.append(f"N={size} ({len(locations)})")
            ax.axvline(locations[0] - 0.5, color="black", lw=0.5)
        ax.set_xticks(centers, labels, rotation=90)
        ax.set_xlabel("Polyhedra grouped by spacecraft count")
        ticks = maximum * np.array([-1, -0.1, -0.01, 0, 0.01, 0.1, 1])
        colorbar = fig.colorbar(display, ax=ax, ticks=ticks, label=label + " (signed symlog)")
        colorbar.ax.set_yticklabels([f"{value:.2g}" for value in ticks])
    save(fig, "heatmaps")

    fig, axes = plt.subplots(4, 2, figsize=(13, 12), constrained_layout=True)
    for row, (node, ax) in enumerate(zip(nodes, axes.flat)):
        ranked = c[row, ranking[row]]
        rank = np.arange(1, len(ranked) + 1)
        ax.plot(rank, np.cumsum(ranked), label="Signed sum")
        ax.plot(rank, np.cumsum(np.abs(ranked)), label="Sum of absolute contributions")
        ax.axhline(c[row].sum(), color="black", ls=":", label="Predicted correction")
        ax.set(title=node, xlabel="Polyhedron rank by |contribution|", ylabel="Cumulative [ms]")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    save(fig, "cumulative")

    fig, axes = plt.subplots(4, 2, figsize=(15, 14), constrained_layout=True)
    titles = {
        "number_of_spacecraft": "N",
        "characteristic_size_km": "L [km]",
        "condition_number": "cond(R)",
        "affine_rms_residual_nT": "Affine residual [nT]",
    }
    for row, name in enumerate(
        (
            "number_of_spacecraft",
            "characteristic_size_km",
            "condition_number",
            "affine_rms_residual_nT",
        )
    ):
        entry = aggregates[name]
        edges = entry["edges"]
        labels = (
            [str(n) for n in range(4, 10)]
            if name == "number_of_spacecraft"
            else [f"{a:.3g}–{b:.3g}" for a, b in zip(edges[:-1], edges[1:])]
        )
        labels.append("N/A / nonfinite")
        for column, kind in enumerate(("signed_s", "absolute_s")):
            ax = axes[row, column]
            for node_index, node in enumerate(nodes):
                ax.plot(
                    np.arange(len(labels)),
                    entry[kind][node_index] * 1000,
                    "o-",
                    ms=3,
                    color=SPACECRAFT_COLORS.get(node),
                    label=node,
                )
            ax.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
            ax.set(
                title=f"{titles[name]}: {'signed' if column == 0 else 'absolute'} sum",
                ylabel="Timing contribution [ms]",
            )
            ax.grid(alpha=0.2)
    axes[0, 0].legend(ncol=4, fontsize=8)
    save(fig, "aggregates")

    report = base.with_name(f"{base.stem}_influence_top.csv")
    with report.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "node",
                "rank",
                "polyhedron",
                "members",
                "N",
                "L_km_median",
                "cond_R_median",
                "sigma_d_nT_per_km",
                "affine_rms_nT",
                "g_per_s",
                "signed_c_ms",
                "absolute_c_ms",
            ]
        )
        for node_index, node in enumerate(nodes):
            for rank, polyhedron in enumerate(ranking[node_index, :10], 1):
                writer.writerow(
                    [
                        node,
                        rank,
                        names[polyhedron],
                        members[polyhedron],
                        *[
                            summaries[key][polyhedron]
                            for key in (
                                "number_of_spacecraft",
                                "characteristic_size_km",
                                "condition_number",
                                "sigma_d",
                                "affine_rms_residual_nT",
                            )
                        ],
                        g[node_index, polyhedron],
                        c[node_index, polyhedron],
                        abs(c[node_index, polyhedron]),
                    ]
                )
    outputs.append(report)
    return outputs
