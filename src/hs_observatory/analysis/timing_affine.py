"""All-nine polynomial residual bases for the shared timing-calibration pipeline."""

from __future__ import annotations

import numpy as np

from hs_observatory.analysis.magnetic_gradient import (
    align_spacecraft_samples,
    geometry_diagnostics,
    geometry_is_static,
)


def affine_projection_bases(positions: np.ndarray) -> dict[str, np.ndarray]:
    """Orthonormal residual columns in spacecraft-major (Bx, By, Bz) order.

    A common spatial scale conditions the designs without changing their column
    spaces or the trace constraint. Common scalar whitening likewise leaves these
    bases unchanged. Static relative geometry is factored only once.
    """
    points = np.asarray(positions, dtype=float)
    if points.ndim != 3 or points.shape[1:] != (9, 3) or not np.isfinite(points).all():
        raise ValueError("Projection positions must have shape (time, 9, 3) and be finite")
    # Parameters are B0 (3), followed by G[field_component, spatial_derivative].
    trace = np.zeros((1, 12))
    trace[0, [3, 7, 11]] = 1
    _, _, right = np.linalg.svd(trace, full_matrices=True)
    trace_free = right[1:].T
    records = {key: [] for key in ("non_affine_basis", "divergence_basis", "df_basis",
                                  "affine_singular_values", "df_singular_values")}
    samples = points[:1] if geometry_is_static(points) else points
    for sample in samples:
        geometry = geometry_diagnostics(sample)
        if geometry.rank != 3:
            raise ValueError("All-nine affine projection requires nondegenerate 3D geometry")
        centered = (sample - geometry.barycenter) / geometry.characteristic_size
        design = np.zeros((9, 3, 12))
        design[:, :, :3] = np.eye(3)
        for component in range(3):
            design[:, component, 3 + 3*component:6 + 3*component] = centered
        design = design.reshape(27, 12)
        u_aff, s_aff, _ = np.linalg.svd(design, full_matrices=True)
        u_df, s_df, _ = np.linalg.svd(design @ trace_free, full_matrices=True)
        for singular, rank in ((s_aff, 12), (s_df, 11)):
            tolerance = np.finfo(float).eps * 27 * singular[0]
            if np.count_nonzero(singular > tolerance) != rank:
                raise ValueError("Rank-deficient all-nine affine design")
        non_affine = u_aff[:, 12:]
        # Extract the one affine direction orthogonal to the trace-free space.
        remainder = u_aff[:, :12] - u_df[:, :11] @ (u_df[:, :11].T @ u_aff[:, :12])
        u_div, s_div, _ = np.linalg.svd(remainder, full_matrices=False)
        if not np.isclose(s_div[0], 1, atol=1e-10) or np.any(s_div[1:] > 1e-10):
            raise ValueError("Divergence subspace must have rank one")
        divergence = u_div[:, :1]
        full = np.column_stack((non_affine, divergence))
        if (not np.allclose(full.T @ full, np.eye(16), atol=1e-10, rtol=0)
                or not np.allclose(full.T @ (design @ trace_free), 0, atol=1e-10)):
            raise ValueError("Affine residual bases failed orthogonality checks")
        for key, value in zip(records, (non_affine, divergence, full, s_aff, s_df)):
            records[key].append(value)
    return {key: np.asarray(value) for key, value in records.items()}


def quadratic_parameter_basis() -> np.ndarray:
    """Map 26 independent coefficients to B0(3), G(9), symmetric H(18).

    H pairs are xx, xy, xz, yy, yz, zz for each field component. Coefficients
    refer to uniformly scaled coordinates; the four zero constraints retain
    their form under this scaling.
    """
    pairs = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
    constraints = np.zeros((4, 30))
    constraints[0, [3, 7, 11]] = 1
    for k in range(3):
        for i in range(3):
            pair = pairs.index(tuple(sorted((i, k))))
            constraints[k + 1, 12 + 6*i + pair] = 1
    _, singular, right = np.linalg.svd(constraints, full_matrices=True)
    if np.count_nonzero(singular > 1e-12) != 4:
        raise ValueError("Quadratic divergence constraints must have rank four")
    return right[4:].T


def quadratic_design_matrix(positions: np.ndarray) -> np.ndarray:
    """Return the exact divergence-free 27 x 26 design in scaled coordinates."""
    points = np.asarray(positions, dtype=float)
    if points.shape != (9, 3) or not np.isfinite(points).all():
        raise ValueError("Quadratic positions must have shape (9, 3) and be finite")
    geometry = geometry_diagnostics(points)
    if geometry.characteristic_size <= 0:
        raise ValueError("Quadratic design requires a positive observatory length")
    x = (points - geometry.barycenter) / geometry.characteristic_size
    quadratic = np.column_stack((.5*x[:, 0]**2, x[:, 0]*x[:, 1], x[:, 0]*x[:, 2],
                                  .5*x[:, 1]**2, x[:, 1]*x[:, 2], .5*x[:, 2]**2))
    design = np.zeros((9, 3, 30))
    design[:, :, :3] = np.eye(3)
    for i in range(3):
        design[:, i, 3 + 3*i:6 + 3*i] = x
        design[:, i, 12 + 6*i:18 + 6*i] = quadratic
    return design.reshape(27, 30) @ quadratic_parameter_basis()


def quadratic_projection_bases(positions: np.ndarray, *,
                               condition_number_max: float = 1e10) -> dict[str, np.ndarray]:
    """Cache a single residual direction for static relative geometry.

    Reject rank loss or severe design conditioning; never discard extra singular
    modes or regularize quadratic coefficients to manufacture a rank-one fit.
    """
    points = np.asarray(positions, dtype=float)
    if points.ndim != 3 or points.shape[1:] != (9, 3) or not np.isfinite(points).all():
        raise ValueError("Quadratic positions must have shape (time, 9, 3) and be finite")
    if not np.isfinite(condition_number_max) or condition_number_max < 1:
        raise ValueError("quadratic_condition_number_max must be finite and at least one")
    samples = points[:1] if geometry_is_static(points) else points
    records = {key: [] for key in ("quadratic_basis", "quadratic_singular_values",
                                  "quadratic_condition_number", "quadratic_rank")}
    for index, sample in enumerate(samples):
        design = quadratic_design_matrix(sample)
        left, singular, _ = np.linalg.svd(design, full_matrices=True)
        tolerance = np.finfo(float).eps * 27 * singular[0]
        rank = int(np.count_nonzero(singular > tolerance))
        condition = float(singular[0] / singular[-1]) if rank == 26 else np.inf
        if rank != 26 or condition > condition_number_max:
            raise ValueError(
                f"Quadratic design at geometry sample {index}: rank={rank}/26, "
                f"condition_number={condition:.6g}, limit={condition_number_max:.6g}"
            )
        q = left[:, 26:]
        if (not np.allclose(q.T @ q, np.eye(1), atol=1e-12, rtol=0)
                or not np.allclose(q.T @ design, 0, atol=1e-12*singular[0], rtol=0)):
            raise ValueError(f"Quadratic basis orthogonality failed at geometry sample {index}")
        for key, value in zip(records, (q, singular, condition, rank)):
            records[key].append(value)
    return {key: np.asarray(value) for key, value in records.items()}


def projection_components(bases):
    """Residual components stored for the selected polynomial model."""
    return ("quadratic",) if "quadratic_basis" in bases else ("non_affine", "divergence")


def project_field(basis: np.ndarray, field: np.ndarray) -> np.ndarray:
    """Residual coordinates; their squared norm is the projector residual power."""
    return np.einsum("tqr,tq->tr", basis, field.reshape(len(field), 27))


def calibrate_affine_realization(time, positions, measured, node_indices, sigma_t,
                                 bases, method, sigma_B):
    """Use the shared solver and interpolation, fitting measured data only."""
    from hs_observatory.analysis.timing_calibrate import solve_timing

    basis_key = {"all_nine_divergence": "divergence_basis",
                 "all_nine_df_affine": "df_basis", "all_nine_df_quadratic": "quadratic_basis"}[method]
    basis = bases[basis_key]
    before = project_field(basis, measured)
    derivative = np.gradient(measured, time, axis=0, edge_order=1)
    sampling = np.zeros((len(time), 9, 3, len(node_indices)))
    for column, node in enumerate(node_indices):
        sampling[:, node, :, column] = derivative[:, node]
    jacobian = np.einsum("tqr,tqn->trn", basis, sampling.reshape(len(time), 27, -1))
    result = solve_timing(before, jacobian, np.full(basis.shape[-1], sigma_B**2), sigma_t)
    # Generic solver's legacy divergence keys denote the entire data term here.
    for label in ("before", "after"):
        result[f"linear_data_chi2_{label}"] = result.pop(f"linear_divergence_chi2_{label}")
    offsets = np.zeros(9)
    offsets[node_indices] = -result["correction_s"]
    aligned = align_spacecraft_samples(time, offsets, positions.swapaxes(0, 1), measured)
    after = project_field(basis, aligned.field)
    common = np.all(result["fit_valid"], axis=1) & np.all(np.isfinite(after), axis=1)
    if not np.any(common):
        raise ValueError("No common valid rows remain after timing correction")
    residuals = {}
    for label, field in (("before", measured), ("after", aligned.field)):
        for component in projection_components(bases):
            coordinates = project_field(bases[f"{component}_basis"], field)
            residuals[f"{component}_residual_{label}"] = coordinates
            result[f"{component}_chi2_{label}"] = float(
                np.sum((coordinates[common] / sigma_B)**2))
        target = "quadratic" if method == "all_nine_df_quadratic" else "divergence"
        data = result[f"{target}_chi2_{label}"]
        if method == "all_nine_df_affine":
            data += result[f"non_affine_chi2_{label}"]
        result[f"data_chi2_{label}"] = data
        result[f"chi2_{label}"] = data + result[f"prior_chi2_{label}"]
        result[f"normalized_residual_rms_{label}"] = np.sqrt(
            data / (common.sum() * basis.shape[-1]))
    residuals.update(fit_valid=result["fit_valid"], comparison_valid=common)
    result.update(
        projection_residuals=residuals,
        measured_B=measured, corrected_B=aligned.field, corrected_B_valid=aligned.valid,
        comparison_row_count=int(common.sum() * basis.shape[-1]),
        fit_row_count=int(result["fit_valid"].sum()),
        chi2_increased=bool(result["chi2_after"] > result["chi2_before"]
                            + 1e-12 * max(1., result["chi2_before"])),
    )
    return result


def plot_affine_product(product_path, output_path, *, mc_index=0, dpi=300,
                        start_time_s=None, stop_time_s=None):
    """Projection diagnostics, using the same MC layout and Node colors."""
    from pathlib import Path

    import h5py
    import matplotlib.pyplot as plt

    from hs_observatory.analysis.timing_calibrate import _plot_time_mask
    from hs_observatory.visualization import SPACECRAFT_COLORS

    if dpi < 1:
        raise ValueError("dpi must be positive")
    with h5py.File(product_path, "r") as handle:
        if not handle["metadata"].attrs.get("complete", False):
            raise ValueError("Timing-calibration product is incomplete")
        injected = handle["timing/injected_offset_s"][...] * 1000
        recovered = handle["timing/recovered_offset_s"][...] * 1000
        if not 0 <= mc_index < len(injected):
            raise ValueError("mc_index is outside the saved ensemble")
        nodes = handle["spacecraft/node_id"].asstr()[...]
        time = handle["time/analysis_time"][...]
        mask = _plot_time_mask(time, start_time_s, stop_time_s)
        method = handle["metadata"].attrs["calibration_method"]
        diagnostics = {key: value[...] for key, value in handle["diagnostics"].items()
                       if isinstance(value, h5py.Dataset)}
        singular = handle["diagnostics/weighted_jacobian/singular_values"][mc_index]
        components = projection_components(handle["projection"])
        if components == ("quadratic",):
            design_singular = handle["projection/quadratic_singular_values"][...]
            design_condition = handle["projection/quadratic_condition_number"][...]
        curves = {}
        for component in components:
            curves[component] = [handle[f"projection/ideal_{component}_residual"][...],
                                 handle[f"projection/{component}_residual_before"][mc_index],
                                 handle[f"projection/{component}_residual_after"][mc_index]]
    fig, axes = plt.subplots(4, 2, figsize=(12, 13), constrained_layout=True)
    fig.suptitle(method)
    for column, node in enumerate(nodes):
        axes[0, 0].scatter(injected[:, column], recovered[:, column], s=12,
                           color=SPACECRAFT_COLORS.get(node), label=node)
    bounds = [min(injected.min(), recovered.min()), max(injected.max(), recovered.max())]
    axes[0, 0].plot(bounds, bounds, "k--")
    axes[0, 0].set(xlabel="Injected offset [ms]", ylabel="Recovered offset [ms]")
    axes[0, 0].legend(ncol=4, fontsize=8)
    axes[0, 1].boxplot(recovered - injected)
    axes[0, 1].set_xticks(np.arange(1, 9), nodes)
    axes[0, 1].set(ylabel="Recovered − injected [ms]", xlabel="Node")
    for label in ("before", "after"):
        axes[1, 0].plot(diagnostics[f"chi2_{label}"], "o-", label=label)
    axes[1, 0].set(xlabel="MC realization", ylabel="Objective (data + prior)")
    axes[1, 0].legend()
    axes[1, 1].plot(np.arange(1, 9), singular, "o-")
    if np.any(singular > 0):
        axes[1, 1].set_yscale("log")
    axes[1, 1].set(xlabel="Singular-value index", ylabel="Weighted Jacobian [1/s]")
    for column, component in enumerate(components):
        for label in ("before", "after"):
            axes[2, column].plot(diagnostics[f"{component}_chi2_{label}"], "o-", label=label)
        axes[2, column].set(xlabel="MC realization", ylabel="Residual power / σB²",
                            title=component.replace("_", " "))
        axes[2, column].legend()
        for label, residual in zip(("Ideal", "Measured", "Corrected"), curves[component]):
            axes[3, column].plot(time[mask], np.linalg.norm(residual[mask], axis=1), label=label)
        axes[3, column].set(xlabel="Time [s]", ylabel="Residual norm [nT]",
                            title=f"MC {mc_index}")
        axes[3, column].legend()
    if components == ("quadratic",):
        axes[2, 1].semilogy(np.arange(1, 27), np.median(design_singular, axis=0), "o-")
        axes[2, 1].set(xlabel="Design singular-value index", ylabel="Scaled design singular value",
                        title="Median quadratic design spectrum")
        condition = np.broadcast_to(design_condition, time.shape)
        axes[3, 1].semilogy(time[mask], condition[mask])
        axes[3, 1].set(xlabel="Time [s]", ylabel="Quadratic design condition number")
    for ax in axes.flat:
        ax.grid(alpha=.2)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output, dpi=dpi)
    finally:
        plt.close(fig)
    return output


def plot_timing_method_comparison(product_paths, output_path, *, dpi=300):
    """Compare recovery on identical input ensembles; objectives have different weights."""
    from pathlib import Path

    import h5py
    import matplotlib.pyplot as plt

    if len(product_paths) < 2:
        raise ValueError("Comparison requires at least two timing products")
    if dpi < 1:
        raise ValueError("dpi must be positive")
    records = []
    with h5py.File(product_paths[0], "r") as reference:
        for path in product_paths:
            with h5py.File(path, "r") as handle:
                if not handle["metadata"].attrs.get("complete", False):
                    raise ValueError("Timing-calibration product is incomplete")
                for key in ("spacecraft/id", "spacecraft/node_id", "spacecraft/positions",
                            "time/analysis_time", "timing/injected_offset_s", "magnetic_field/ideal"):
                    if not np.array_equal(handle[key][...], reference[key][...]):
                        raise ValueError(f"Comparison requires identical inputs: {key}")
                measured, baseline = handle["magnetic_field/measured"], reference["magnetic_field/measured"]
                if measured.shape != baseline.shape or any(
                    not np.array_equal(measured[mc], baseline[mc], equal_nan=True)
                    for mc in range(len(measured))
                ):
                    raise ValueError("Comparison requires identical measured MC ensembles")
                attrs = handle["metadata"].attrs
                if attrs["sigma_t_s"] != reference["metadata"].attrs["sigma_t_s"]:
                    raise ValueError("Comparison requires identical timing priors")
                if "projection" in handle:
                    scales = [r["sigma_B"] for r in records if r["sigma_B"] is not None]
                    if scales and attrs["sigma_B_nT"] != scales[0]:
                        raise ValueError("Projection comparison requires identical sigma_B_nT")
                records.append({
                    "method": attrs.get("calibration_method", "polyhedron"),
                    "sigma_B": attrs.get("sigma_B_nT"),
                    "bias": handle["mc_summary/bias_s"][...] * 1000,
                    "rmse": handle["mc_summary/rmse_s"][...] * 1000,
                    "rms": np.sqrt(np.mean(handle["timing/recovered_offset_s"][...]**2, axis=1))*1000,
                    "singular": handle["diagnostics/weighted_jacobian/singular_values"][...],
                })
        nodes = reference["spacecraft/node_id"].asstr()[...]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for record in records:
        for ax, key in zip(axes[0], ("bias", "rmse")):
            ax.plot(nodes, record[key], "o-", label=record["method"])
            ax.set(xlabel="Node", ylabel=f"Recovery {key} [ms]")
        axes[1, 0].plot(record["rms"], "o-", label=record["method"])
        axes[1, 1].plot(np.arange(1, 9), np.median(record["singular"], axis=0),
                        "o-", label=record["method"])
    axes[1, 0].set(xlabel="MC realization", ylabel="RMS recovered offset across Nodes [ms]")
    axes[1, 1].set(xlabel="Singular-value index", ylabel="Median weighted Jacobian [1/s]")
    if any(np.any(record["singular"] > 0) for record in records):
        axes[1, 1].set_yscale("log")
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output, dpi=dpi)
    finally:
        plt.close(fig)
    return output
