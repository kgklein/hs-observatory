"""Affine projection identities, flight-like recovery, and turbulent-field limits."""

import h5py
import numpy as np
import pytest
import yaml

from hs_observatory.analysis.magnetic_gradient import gradient_response_coefficients
from hs_observatory.analysis.timing_affine import (
    affine_projection_bases,
    calibrate_affine_realization,
    plot_timing_method_comparison,
    project_field,
)
from hs_observatory.analysis.timing_calibrate import (
    _settings,
    calibrate_realization,
    divergence_series,
    divergence_variance,
    select_polyhedra,
    compute_timing_calibration,
    write_timing_calibration_product,
)
from hs_observatory.cli import plot_timing_calibrate_main
from hs_observatory.io.hdf5 import write_observations

METHODS = ("all_nine_divergence", "all_nine_df_affine")
IDS = ["H", *(f"N{i}" for i in range(1, 9))]


def affine_field(time, points):
    rng = np.random.default_rng(123)
    gradients = rng.normal(size=(16, 3, 3))
    gradients -= np.trace(gradients, axis1=1, axis2=2)[:, None, None] * np.eye(3) / 3
    offsets = rng.normal(size=(16, 3))
    spatial = offsets[:, None, :] + np.einsum("mij,nj->mni", gradients, points)
    phases = np.asarray(time)[..., None] * np.linspace(0.5, 8, 16) + np.arange(16)
    return np.einsum("tnm,mni->tni", np.sin(phases), spatial)


def turbulent_field(time, points, seed=47):
    """Random-phase solenoidal Fourier surrogate, not an MHD simulation."""
    rng = np.random.default_rng(seed)
    k = rng.normal(size=(24, 3))
    k /= np.linalg.norm(k, axis=1)[:, None]
    k *= np.geomspace(0.2, 3, len(k))[:, None]
    amplitude = rng.normal(size=k.shape)
    amplitude -= (np.sum(amplitude * k, axis=1) / np.sum(k * k, axis=1))[:, None] * k
    amplitude /= np.linalg.norm(amplitude, axis=1)[:, None]
    amplitude *= np.linalg.norm(k, axis=1)[:, None] ** (-5 / 6) / np.sqrt(len(k))
    np.testing.assert_allclose(np.sum(k * amplitude, axis=1), 0, atol=1e-14)
    phase = points @ k.T + time[:, None, None] * (k @ np.array([0.7, -0.3, 0.2]))
    phase += rng.uniform(0, 2 * np.pi, len(k))
    return np.einsum("tnm,mi->tni", np.sin(phase), amplitude)


@pytest.mark.parametrize("moving", [False, True])
def test_projector_identities_and_power(moving):
    rng = np.random.default_rng(9)
    points = rng.normal(size=(9, 3))
    positions = np.stack([points + i for i in range(3)])
    if moving:
        positions[1:] += rng.normal(size=(2, 9, 3)) * 0.1
    bases = affine_projection_bases(positions)
    assert len(bases["df_basis"]) == (3 if moving else 1)
    projectors = {}
    for name, rank in (("non_affine", 15), ("df", 16), ("divergence", 1)):
        q = bases[f"{name}_basis"]
        p = q @ q.swapaxes(-1, -2)
        projectors[name] = p
        np.testing.assert_allclose(p, p.swapaxes(-1, -2), atol=1e-13)
        np.testing.assert_allclose(p @ p, p, atol=1e-13)
        np.testing.assert_array_equal(np.linalg.matrix_rank(p), np.full(len(p), rank))
    np.testing.assert_allclose(projectors["divergence"] @ projectors["non_affine"], 0, atol=1e-13)
    np.testing.assert_allclose(
        projectors["df"] - projectors["non_affine"], projectors["divergence"], atol=1e-13
    )
    field = rng.normal(size=positions.shape)
    power = {
        key: np.sum(project_field(bases[f"{key}_basis"], field) ** 2, axis=1) for key in projectors
    }
    np.testing.assert_allclose(power["df"], power["non_affine"] + power["divergence"])
    # Independently construct the unconstrained and constrained design projectors.
    for i, sample in enumerate(positions[: len(bases["df_basis"])]):
        centered = sample - sample.mean(axis=0)
        design = np.column_stack((np.ones(9), centered))
        scalar_q = np.linalg.qr(design)[0]
        affine_perp = np.eye(27) - np.kron(scalar_q @ scalar_q.T, np.eye(3))
        np.testing.assert_allclose(projectors["non_affine"][i], affine_perp, atol=1e-13)
    df_field = affine_field(np.broadcast_to(np.arange(3)[:, None], (3, 9)), points)
    if not moving:
        # Uniform translations preserve the space of trace-free affine fields.
        np.testing.assert_allclose(project_field(bases["df_basis"], df_field), 0, atol=1e-13)


def test_degenerate_geometry_rejected():
    with pytest.raises(ValueError, match="nondegenerate"):
        affine_projection_bases(np.zeros((2, 9, 3)))


@pytest.mark.parametrize("method", METHODS)
def test_affine_zero_and_known_offsets(method):
    time = np.linspace(-3, 3, 3001)
    points = np.random.default_rng(19).normal(size=(9, 3))
    positions = np.broadcast_to(points, (len(time), 9, 3))
    bases = affine_projection_bases(positions)
    recovery_errors = []
    for offsets in (
        np.zeros(9),
        np.r_[0.0, np.linspace(-0.0003, 0.0003, 8)],
        np.r_[0.0, np.linspace(-0.00003, 0.00003, 8)],
    ):
        measured = affine_field(time[:, None] + offsets, points)
        fit = calibrate_affine_realization(
            time, positions, measured, np.arange(1, 9), 0.1, bases, method, 0.001
        )
        assert fit["jacobian"]["rank"] == 8
        # First-order timing and sampled derivatives have truncation error.
        tolerance = 0.01 * np.max(np.abs(offsets)) if np.any(offsets) else 1e-12
        if np.any(offsets):
            recovery_errors.append(np.max(np.abs(fit["recovered_offset_s"] - offsets[1:])))
        np.testing.assert_allclose(fit["recovered_offset_s"], offsets[1:], atol=tolerance)
        np.testing.assert_array_equal(fit["correction_s"], -fit["recovered_offset_s"])
        np.testing.assert_array_equal(fit["corrected_B"][:, 0], measured[:, 0])
        assert np.linalg.eigvalsh(fit["covariance_s2"]).min() > 0
        common = fit["projection_residuals"]["comparison_valid"]
        basis = bases["df_basis" if method == METHODS[1] else "divergence_basis"]
        actual = np.sum((project_field(basis, fit["corrected_B"])[common] / 0.001) ** 2)
        assert fit["data_chi2_after"] == pytest.approx(actual, rel=1e-12, abs=1e-18)
        assert fit["chi2_after"] <= fit["chi2_before"] + 1e-15
        if method == METHODS[1]:
            for label in ("before", "after"):
                assert fit[f"data_chi2_{label}"] == pytest.approx(
                    fit[f"non_affine_chi2_{label}"] + fit[f"divergence_chi2_{label}"]
                )
    assert recovery_errors[1] < 0.15 * recovery_errors[0]


@pytest.mark.parametrize(
    "config",
    [
        {"calibration_method": "unknown"},
        *[{"calibration_method": METHODS[0], "sigma_B_nT": v} for v in (None, 0, -1, np.nan)],
        {"calibration_method": METHODS[0], "sigma_B_nT": 1, "use_ideal_divergence": True},
        {"calibration_method": METHODS[1], "sigma_B_nT": 1, "polyhedron_filter": {}},
        {
            "calibration_method": METHODS[1],
            "sigma_B_nT": 1,
            "diagnostics": {"polyhedron_influence": True},
        },
    ],
)
def test_settings_reject_invalid_projection(config):
    with pytest.raises(ValueError):
        _settings(config)


def write_source(path, time, positions, ideal, measured, offsets, order=None):
    order = np.arange(9) if order is None else order
    n = np.ones((len(time), 9))
    v = np.zeros_like(ideal)
    write_observations(
        path,
        time={key: time for key in ("n", "v", "B")},
        spacecraft_ids=[IDS[i] for i in order],
        position={key: positions[:, order].swapaxes(0, 1) for key in ("n", "v", "B")},
        truth={"n": n, "v": v, "B": ideal[:, order]},
        observed={
            "n": np.broadcast_to(n, (len(measured), *n.shape)),
            "v": np.broadcast_to(v, (len(measured), *v.shape)),
            "B": measured[:, :, order],
        },
        errors={"time_offset": offsets[:, order]},
        metadata={"config_yaml": yaml.safe_dump({"errors": {"time": {"model": "none"}}})},
    )
    return path


def test_products_comparison_truth_independence_and_legacy(tmp_path):
    time = np.linspace(-2, 2, 121)
    points = np.random.default_rng(19).normal(size=(9, 3))
    positions = np.broadcast_to(points, (len(time), 9, 3))
    ideal = affine_field(np.broadcast_to(time[:, None], (len(time), 9)), points)
    offsets = np.stack([np.zeros(9), np.r_[0, np.linspace(-0.0002, 0.0002, 8)]])
    measured = np.stack([affine_field(time[:, None] + shift, points) for shift in offsets])
    order = np.array([3, 2, 0, 1, 4, 5, 6, 7, 8])
    source = write_source(tmp_path / "source.h5", time, positions, ideal, measured, offsets, order)
    products = []
    for method in ("polyhedron", *METHODS):
        config = {"calibration_method": method, "sigma_t_s": 0.1, "sigma_B_nT": 0.01}
        result = compute_timing_calibration(source, config)
        output = write_timing_calibration_product(result, tmp_path / f"{method}.h5")
        products.append(output)
        with h5py.File(output) as handle:
            assert handle["metadata"].attrs["complete"]
            assert "H" not in handle["spacecraft/node_id"].asstr()[...]
            np.testing.assert_allclose(
                handle["timing/recovery_error_s"],
                handle["timing/recovered_offset_s"][...] - handle["timing/injected_offset_s"][...],
            )
            if method != "polyhedron":
                assert handle["projection/df_basis"].shape == (1, 27, 16)
                assert handle["projection/non_affine_residual_after"].shape == (2, 121, 15)
                assert handle["projection/divergence_residual_after"].shape == (2, 121, 1)
                assert handle["diagnostics/jacobian/singular_values"].attrs["units"] == "nT/s"
    # Default retains precisely the explicit legacy fit.
    legacy = write_timing_calibration_product(
        compute_timing_calibration(source, {"sigma_t_s": 0.1}), tmp_path / "default.h5"
    )
    with h5py.File(legacy) as a, h5py.File(products[0]) as b:
        for key in ("timing/correction_s", "timing/covariance_s2", "diagnostics/chi2_after"):
            np.testing.assert_array_equal(a[key], b[key])
    # Changing synthetic truth cannot influence either new fit or its normalization.
    with h5py.File(source, "r+") as handle:
        handle["truth/B"][...] = 12345
    for method, reference in zip(METHODS, products[1:]):
        output = write_timing_calibration_product(
            compute_timing_calibration(
                source, {"calibration_method": method, "sigma_t_s": 0.1, "sigma_B_nT": 0.01}
            ),
            tmp_path / f"changed_{method}.h5",
        )
        with h5py.File(output) as a, h5py.File(reference) as b:
            np.testing.assert_array_equal(a["timing/correction_s"], b["timing/correction_s"])
    source.unlink()
    plot = tmp_path / "projection.png"
    assert (
        plot_timing_calibrate_main(
            [
                str(products[2]),
                "--output",
                str(plot),
                "--dpi",
                "40",
                "--compare-products",
                *map(str, products[:2]),
            ]
        )
        == 0
    )
    assert plot.stat().st_size > 1000
    assert (tmp_path / "projection_comparison.png").stat().st_size > 1000
    with h5py.File(products[1], "r+") as handle:
        handle["magnetic_field/measured"][0, 10, 3, 0] += 0.1
    with pytest.raises(ValueError, match="identical measured"):
        plot_timing_method_comparison(products, tmp_path / "invalid.png")


@pytest.mark.parametrize("seed", [47, 48, 49])
def test_turbulent_zero_offset_spurious_corrections(seed):
    time = np.linspace(0, 8, 241)
    points = np.random.default_rng(19).normal(size=(9, 3))
    positions = np.broadcast_to(points, (len(time), 9, 3))
    field = turbulent_field(time, points, seed)
    bases = affine_projection_bases(positions)
    polyhedra = list(select_polyhedra(points, IDS, {}).values())
    responses = [gradient_response_coefficients(positions[:, indices]) for indices in polyhedra]
    divergence = np.column_stack(
        [
            divergence_series(response, field[:, indices])
            for indices, response in zip(polyhedra, responses)
        ]
    )
    _, variance = divergence_variance(divergence, 1e-12)
    legacy = calibrate_realization(
        time, positions, field, polyhedra, responses, np.arange(1, 9), variance, 0.0125
    )
    print(
        f"seed={seed} polyhedron: spurious RMS="
        f"{np.sqrt(np.mean(legacy['correction_s'] ** 2)) * 1000:.6g} ms; "
        f"max={np.max(np.abs(legacy['correction_s'])) * 1000:.6g} ms; "
        f"rank={legacy['jacobian']['rank']}"
    )
    for method in METHODS:
        fit = calibrate_affine_realization(
            time, positions, field, np.arange(1, 9), 0.0125, bases, method, 0.1
        )
        rms = np.sqrt(np.mean(fit["correction_s"] ** 2))
        assert np.isfinite(rms)
        assert fit["linear_chi2_after"] <= fit["linear_chi2_before"] + 1e-10
        print(
            f"seed={seed} {method}: spurious RMS={rms * 1000:.6g} ms; "
            f"max={np.max(np.abs(fit['correction_s'])) * 1000:.6g} ms; "
            f"rank={fit['jacobian']['rank']}"
        )
