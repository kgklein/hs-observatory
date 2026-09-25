"""Exact solenoidal quadratic design, projection, and timing integration."""

import h5py
import numpy as np
import pytest

from hs_observatory.analysis.magnetic_gradient import geometry_diagnostics
from hs_observatory.analysis.timing_affine import (
    calibrate_affine_realization,
    project_field,
    quadratic_design_matrix,
    quadratic_parameter_basis,
    quadratic_projection_bases,
)
from hs_observatory.analysis.timing_calibrate import (
    _settings,
    compute_timing_calibration,
    write_timing_calibration_product,
)
from hs_observatory.cli import plot_timing_calibrate_main
from test_timing_affine import turbulent_field, write_source

METHOD = "all_nine_df_quadratic"


def quadratic_field(time, points):
    """Independent construction: transverse vector times a univariate quadratic."""
    rng = np.random.default_rng(72)
    k = rng.normal(size=(32, 3))
    k /= np.linalg.norm(k, axis=1)[:, None]
    a = rng.normal(size=k.shape)
    a -= np.sum(a*k, axis=1)[:, None]*k
    np.testing.assert_allclose(np.sum(a*k, axis=1), 0, atol=1e-14)
    coordinate = points @ k.T
    spatial = (coordinate**2 + .7*coordinate + .3).T[:, :, None]*a[:, None, :]
    phase = time[..., None]*np.linspace(.5, 8, 32) + rng.uniform(0, 6, 32)
    return np.einsum("tnm,mni->tni", np.sin(phase), spatial)


def test_quadratic_constraints_and_design():
    transform = quadratic_parameter_basis()
    assert transform.shape == (30, 26)
    np.testing.assert_allclose(transform.T @ transform, np.eye(26), atol=1e-14)
    # Interpret every column as an actual polynomial, independently of construction.
    g = transform[3:12].reshape(3, 3, 26)
    h = np.zeros((3, 3, 3, 26))
    for component in range(3):
        for pair, (j, k) in enumerate(((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))):
            h[component, j, k] = h[component, k, j] = transform[12+6*component+pair]
    np.testing.assert_allclose(np.einsum("iia->a", g), 0, atol=1e-14)
    np.testing.assert_allclose(np.einsum("iika->ka", h), 0, atol=1e-14)
    points = np.random.default_rng(19).normal(size=(9, 3))
    geometry = geometry_diagnostics(points)
    x = (points - geometry.barycenter)/geometry.characteristic_size
    expected = (transform[:3][None] + np.einsum("ija,nj->nia", g, x)
                + .5*np.einsum("ijka,nj,nk->nia", h, x, x))
    design = quadratic_design_matrix(points)
    np.testing.assert_allclose(design, expected.reshape(27, 26), atol=1e-14)
    assert np.linalg.matrix_rank(design) == 26


@pytest.mark.parametrize("moving", [False, True])
def test_quadratic_projector_and_exact_field(moving):
    rng = np.random.default_rng(19)
    points = rng.normal(size=(9, 3))
    positions = np.stack([points, points+2, points-3])
    if moving:
        positions[1:] += rng.normal(size=(2, 9, 3))*.2
    bases = quadratic_projection_bases(positions)
    assert bases["quadratic_basis"].shape == (3 if moving else 1, 27, 1)
    for index, sample in enumerate(positions):
        q = bases["quadratic_basis"][index if moving else 0]
        design = quadratic_design_matrix(sample)
        np.testing.assert_allclose(q.T @ design, 0, atol=1e-14)
        np.testing.assert_allclose(q.T @ q, np.eye(1), atol=1e-14)
        projector = q @ q.T
        assert np.linalg.matrix_rank(projector) == 1
        np.testing.assert_allclose(projector.T, projector, atol=1e-14)
        np.testing.assert_allclose(projector @ projector, projector, atol=1e-14)
        field = quadratic_field(np.full((1, 9), .2), sample)
        np.testing.assert_allclose(project_field(q[None], field), 0, atol=1e-12)
    # Translation and common length-unit changes preserve the projection.
    q_scaled = quadratic_projection_bases(positions*1000)["quadratic_basis"]
    q = bases["quadratic_basis"]
    np.testing.assert_allclose(q_scaled @ q_scaled.swapaxes(-1, -2),
                               q @ q.swapaxes(-1, -2), atol=1e-12)


def test_quadratic_geometry_failures_report_sample_rank_and_condition():
    points = np.random.default_rng(19).normal(size=(9, 3))
    with pytest.raises(ValueError, match=r"sample 0: rank=26/26, condition_number="):
        quadratic_projection_bases(points[None], condition_number_max=1.1)
    bad = points.copy()
    bad[1] = bad[0]
    with pytest.raises(ValueError, match=r"sample 1: rank=\d+/26, condition_number=inf"):
        quadratic_projection_bases(np.stack([points, bad]))


@pytest.mark.parametrize("maximum", [0, -1, np.inf, np.nan, True, "bad"])
def test_quadratic_condition_setting(maximum):
    with pytest.raises(ValueError, match="quadratic_condition_number_max"):
        _settings({"calibration_method": METHOD, "sigma_B_nT": .1,
                   "quadratic_condition_number_max": maximum})


def test_quadratic_timing_recovery_hub_and_resampled_objective():
    time = np.linspace(-3, 3, 2001)
    points = np.random.default_rng(19).normal(size=(9, 3))
    positions = np.broadcast_to(points, (len(time), 9, 3))
    bases = quadratic_projection_bases(positions)
    for offsets in (np.zeros(9), np.r_[0, np.linspace(-.0001, .0001, 8)]):
        measured = quadratic_field(time[:, None]+offsets, points)
        fit = calibrate_affine_realization(time, positions, measured, np.arange(1, 9),
                                           .1, bases, METHOD, .001)
        assert fit["jacobian"]["rank"] == 8
        tolerance = 1e-6 if np.any(offsets) else 1e-12
        np.testing.assert_allclose(fit["recovered_offset_s"], offsets[1:], atol=tolerance)
        np.testing.assert_array_equal(fit["correction_s"], -fit["recovered_offset_s"])
        np.testing.assert_array_equal(fit["corrected_B"][:, 0], measured[:, 0])
        common = fit["projection_residuals"]["comparison_valid"]
        for label, field in (("before", measured), ("after", fit["corrected_B"])):
            residual = project_field(bases["quadratic_basis"], field)
            actual = np.sum((residual[common]/.001)**2)
            assert fit[f"quadratic_chi2_{label}"] == pytest.approx(actual, abs=1e-18)
            assert fit[f"chi2_{label}"] == pytest.approx(actual+fit[f"prior_chi2_{label}"])
        assert fit["chi2_after"] <= fit["chi2_before"] + 1e-15
        assert np.linalg.eigvalsh(fit["covariance_s2"]).min() > 0


def test_quadratic_hdf5_mc_truth_independence_and_four_method_cli(tmp_path):
    time = np.linspace(-2, 2, 121)
    points = np.random.default_rng(19).normal(size=(9, 3))
    positions = np.broadcast_to(points, (len(time), 9, 3))
    shifts = np.stack([np.zeros(9), np.r_[0, np.linspace(-.0001, .0001, 8)]])
    measured = np.stack([quadratic_field(time[:, None]+shift, points) for shift in shifts])
    source = write_source(tmp_path / "source.h5", time, positions, measured[0], measured, shifts,
                          order=np.array([3, 0, 2, 1, 4, 5, 6, 7, 8]))
    methods = ("polyhedron", "all_nine_divergence", "all_nine_df_affine", METHOD)
    products = []
    for method in methods:
        config = {"calibration_method": method, "sigma_t_s": .0125, "sigma_B_nT": .1}
        products.append(write_timing_calibration_product(
            compute_timing_calibration(source, config), tmp_path / f"{method}.h5"))
    with h5py.File(products[-1]) as handle:
        assert handle["metadata"].attrs["complete"]
        assert handle["projection/quadratic_basis"].shape == (1, 27, 1)
        assert handle["projection/quadratic_singular_values"].shape == (1, 26)
        assert handle["projection/quadratic_condition_number"][0] > 1
        assert handle["projection/quadratic_rank"][0] == 26
        assert handle["projection/quadratic_residual_after"].shape == (2, 121, 1)
        assert handle["projection/quadratic_residual_after"].attrs["units"] == "nT"
        np.testing.assert_allclose(handle["mc_summary/rmse_s"], np.sqrt(np.mean(
            handle["timing/recovery_error_s"][...]**2, axis=0)))
        np.testing.assert_array_equal(handle["magnetic_field/corrected"][:, :, 1],
                                       handle["magnetic_field/measured"][:, :, 1])
    with h5py.File(source, "r+") as handle:
        handle["truth/B"][...] = 12345
    changed = write_timing_calibration_product(compute_timing_calibration(source, config),
                                               tmp_path / "changed.h5")
    with h5py.File(changed) as actual, h5py.File(products[-1]) as expected:
        for key in ("timing/correction_s", "timing/covariance_s2", "diagnostics/chi2_after"):
            np.testing.assert_array_equal(actual[key], expected[key])
    source.unlink()
    plot = tmp_path / "quadratic.png"
    assert plot_timing_calibrate_main([
        str(products[-1]), "--output", str(plot), "--dpi", "40",
        "--compare-products", *map(str, products[:-1]),
    ]) == 0
    assert plot.stat().st_size > 1000
    assert (tmp_path / "quadratic_comparison.png").stat().st_size > 1000


@pytest.mark.parametrize("seed", [47, 48, 49])
def test_quadratic_turbulent_zero_offset(seed):
    time = np.linspace(0, 8, 241)
    points = np.random.default_rng(19).normal(size=(9, 3))
    positions = np.broadcast_to(points, (len(time), 9, 3))
    field = turbulent_field(time, points, seed)
    bases = quadratic_projection_bases(positions)
    fit = calibrate_affine_realization(time, positions, field, np.arange(1, 9),
                                       .0125, bases, METHOD, .1)
    rms = np.sqrt(np.mean(fit["correction_s"]**2))*1000
    assert np.isfinite(rms)
    assert fit["linear_chi2_after"] <= fit["linear_chi2_before"] + 1e-10
    print(f"seed={seed} quadratic: RMS={rms:.6g} ms; "
          f"max={np.max(np.abs(fit['correction_s']))*1000:.6g} ms; "
          f"timing rank={fit['jacobian']['rank']}; "
          f"design condition={bases['quadratic_condition_number'][0]:.6g}")
