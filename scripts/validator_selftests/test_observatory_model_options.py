"""Explicit-only error-model statistics and validator compatibility; no fields evaluated."""

# ruff: noqa: E402 -- private scripts package is not installed as a public API.
import json
import math
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from measurement_error_validation.model import angular_sigma, normalize
from measurement_error_validation.reporting import model_interpretations, plot_figures
from measurement_error_validation.runner import Validator
from test_measurement_error_validation import args

from hs_observatory.errors.observatory import (
    _apply_gaussian_fractional_bias_ou,
    apply_observatory_measurement_errors,
    apply_transverse_angular_ou,
    normalized_observatory_config,
    transverse_component_sigma_rad,
)
from hs_observatory.io.hdf5 import read_observations, write_observations


def only(**sections):
    cfg = {
        q: {"enabled": False}
        for q in (
            "timing",
            "position",
            "density",
            "velocity_speed",
            "velocity_angle",
            "magnetic_dc",
            "magnetic_ac",
            "alignment",
        )
    }
    for name, section in sections.items():
        cfg[name] = {"enabled": True, **section}
    return cfg


def normal_moments(x, sigma):
    # Independent draws only. Per-check tail probability 1e-6 accommodates the
    # predeclared collection of moment/coverage checks without tight MC tolerances.
    x = np.asarray(x).ravel()
    assert abs(x.mean()) <= stats.norm.ppf(1 - 5e-7) * sigma / np.sqrt(len(x))
    q = (len(x) - 1) * x.var(ddof=1) / sigma**2
    assert stats.chi2.ppf(5e-7, len(x) - 1) <= q <= stats.chi2.ppf(1 - 5e-7, len(x) - 1)


def alignment_product(path, *, relative=True, m=40, sigma_requirement=3, level=3):
    model = (
        "relative_constant_rotation_B_to_V" if relative else "independent_constant_rotation_B_and_V"
    )
    cfg = only(
        alignment=dict(
            model=model,
            requirement_deg_per_axis=sigma_requirement,
            sigma_level=level,
        )
    )
    times = {q: np.arange(3.0) for q in ("n", "v", "B")}
    truth = {
        "n": np.ones((3, 2)),
        "v": np.broadcast_to([400.0, -10.0, 5.0], (3, 2, 3)),
        "B": np.broadcast_to([5.0, 2.0, -1.0], (3, 2, 3)),
    }
    sampled = {q: np.broadcast_to(v, (m, *v.shape)) for q, v in truth.items()}
    result = apply_observatory_measurement_errors(sampled, times, cfg, np.random.default_rng(719))
    write_observations(
        path,
        time=times,
        spacecraft_ids=["S0", "S1"],
        position={q: np.zeros((2, 3, 3)) for q in times},
        truth=truth,
        observed=result.observed,
        errors=result.errors,
        metadata={**result.metadata, "n_mc": m, "random_seed": 719, "schema_version": "2"},
    )
    supplied = dict(
        errors={**cfg, "n_mc": m, "random_seed": 719},
        time=dict(t_start_s=0, t_stop_s=2, cadence_s=1),
    )
    return supplied, result


@pytest.mark.parametrize("relative", [False, True])
def test_alignment_hdf5_validator_and_distribution_pages(tmp_path, relative):
    from unittest.mock import patch
    from matplotlib.figure import Figure

    path = tmp_path / "alignment.h5"
    supplied, result = alignment_product(path, relative=relative)
    before = path.read_bytes()
    with h5py.File(path) as handle:
        validator = Validator(handle, supplied, args())
        rows = [r for r in validator.run() if r["component"] == "alignment"]
        assert not validator.discrepancies
        assert not any(r["status"] in ("FAIL", "SKIPPED") for r in rows)
        prefix = "relative" if relative else "velocity"
        assert any(r["statistic"] == prefix + ".0.sigma" for r in rows)
        assert any(r["statistic"] == prefix + ".rodrigues" for r in rows)
        if relative:
            assert not any("B_V_independence" in r["statistic"] for r in rows)
            assert "FC Cartesian frame" in " ".join(model_interpretations(validator.cfg))
        else:
            assert any("B_V_independence" in r["statistic"] for r in rows)
        captured = {}
        original_save = Figure.savefig

        def capture(fig, filename, **kwargs):
            captured[Path(filename).name] = (
                fig._suptitle.get_text(),
                [axis.get_title() for axis in fig.axes[::2]],
            )
            original_save(fig, filename, **kwargs)

        with patch.object(Figure, "savefig", capture):
            figures = plot_figures(validator.plots, tmp_path)
        assert figures == [
            f"alignment_distributions_{i}.{ext}"
            for i in ((1,) if relative else (1, 2))
            for ext in ("png", "pdf")
        ]
        for name, (title, panels) in captured.items():
            field = "relative" if relative else ("velocity" if "_1." in name else "magnetic")
            assert len(panels) == 4 and all(p.startswith(field) for p in panels)
            assert field in title
            if relative:
                assert "FC reference" in title
            assert (tmp_path / name).stat().st_size > 0
    loaded = read_observations(path)
    for key, values in result.errors.items():
        np.testing.assert_array_equal(loaded["errors"][key], values)
    assert path.read_bytes() == before


def test_relative_alignment_gaussian_maxwell_isotropy_and_independence(tmp_path):
    path = tmp_path / "statistics.h5"
    supplied, result = alignment_product(path, m=10000)
    omega = result.errors["relative_alignment_rotvec"]
    sigma = np.deg2rad(1.0)
    for axis in range(3):
        normal_moments(omega[..., axis], sigma)
    radius = np.linalg.norm(omega, axis=-1).ravel()
    assert stats.kstest(radius, stats.maxwell(scale=sigma).cdf).pvalue > 1e-6
    with h5py.File(path) as handle:
        validator = Validator(handle, supplied, args())
        rows = [r for r in validator.run() if r["component"] == "alignment"]
        assert not any(r["status"] != "PASS" for r in rows)
        names = {r["statistic"] for r in rows}
        assert {
            "relative.coverage.0",
            "relative.direction_cosine.0",
            "relative.mc_independence.S0.0_0",
            "relative.sc_independence.0_1.0_1",
        } <= names
        assert next(r for r in rows if r["statistic"] == "relative.0.sigma")["expected"] == sigma


@pytest.mark.parametrize(
    "corruption",
    [
        "units",
        "convention",
        "metadata",
        "malformed_metadata",
        "missing",
        "missing_matrix",
        "shape",
        "nonfinite",
        "sigma",
        "transpose",
        "determinant",
        "shared",
        "legacy_alias",
        "legacy_provenance",
        "yaml",
    ],
)
def test_relative_alignment_validator_rejects_corruption(tmp_path, corruption):
    from measurement_error_validation.runner import InvalidInput

    path = tmp_path / "corrupt.h5"
    supplied, _ = alignment_product(path)
    with h5py.File(path, "a") as handle:
        vec = handle["errors/relative_alignment_rotvec"]
        mat = handle["errors/relative_alignment_matrix"]
        if corruption == "units":
            vec.attrs["units"] = "deg"
        elif corruption == "convention":
            mat.attrs["reference_instrument"] = "MAG"
        elif corruption == "metadata":
            del handle["metadata"].attrs["relative_alignment"]
        elif corruption == "malformed_metadata":
            handle["metadata"].attrs["relative_alignment"] = "{"
        elif corruption == "missing":
            del handle[vec.name]
        elif corruption == "missing_matrix":
            del handle[mat.name]
        elif corruption == "shape":
            del handle[vec.name]
            handle.create_dataset("errors/relative_alignment_rotvec", data=np.zeros((40, 3)))
        elif corruption == "nonfinite":
            vec[0, 0, 0] = np.nan
        elif corruption == "sigma":
            vec[...] *= 2
        elif corruption == "transpose":
            mat[...] = np.swapaxes(mat[...], -1, -2)
        elif corruption == "determinant":
            mat[0, 0] = np.diag([1.0, 1.0, -1.0])
        elif corruption == "shared":
            vec[:, 1] = vec[:, 0]
            mat[:, 1] = mat[:, 0]
        elif corruption == "legacy_alias":
            handle.create_dataset("errors/magnetic_alignment_rotvec", data=vec[...])
        elif corruption == "legacy_provenance":
            cfg = json.loads(handle["metadata"].attrs["observatory_error_config"])
            cfg["alignment"]["model"] = "independent_constant_rotation_B_and_V"
            handle["metadata"].attrs["observatory_error_config"] = json.dumps(cfg)
        elif corruption == "yaml":
            supplied["errors"]["alignment"]["model"] = "independent_constant_rotation_B_and_V"
    with h5py.File(path) as handle:
        validator = Validator(handle, supplied, args())
        if corruption in ("shape", "nonfinite", "malformed_metadata"):
            with pytest.raises(InvalidInput):
                validator.run()
        else:
            rows = [r for r in validator.run() if r["component"] == "alignment"]
            if corruption == "yaml":
                assert validator.cfg["alignment"]["model"] == "relative_constant_rotation_B_to_V"
                assert validator.discrepancies
            else:
                assert any(r["status"] in ("FAIL", "SKIPPED") for r in rows)
                if corruption == "sigma":
                    assert (
                        next(r for r in rows if r["statistic"] == "relative.0.sigma")["status"]
                        == "FAIL"
                    )
                if corruption == "shared":
                    assert (
                        next(r for r in rows if r["statistic"] == "relative.duplicate_paths")[
                            "status"
                        ]
                        == "FAIL"
                    )


def test_relative_alignment_zero_sigma_validator(tmp_path):
    path = tmp_path / "zero.h5"
    supplied, _ = alignment_product(path, sigma_requirement=0)
    with h5py.File(path) as handle:
        rows = [
            r for r in Validator(handle, supplied, args()).run() if r["component"] == "alignment"
        ]
        assert all(r["status"] == "PASS" for r in rows)
        assert any(r["statistic"] == "relative.0.zero" for r in rows)
        assert not any("direction_cosine" in r["statistic"] for r in rows)


@pytest.mark.parametrize("level", [2, 3])
def test_relative_alignment_small_selection_reports_insufficient_data(tmp_path, level):
    path = tmp_path / "selection.h5"
    supplied, _ = alignment_product(path, level=level)
    with h5py.File(path) as handle:
        validator = Validator(handle, supplied, args(mc="0,1", spacecraft="S1"))
        rows = [r for r in validator.run() if r["component"] == "alignment"]
        assert any(r["status"] == "INSUFFICIENT_DATA" for r in rows)
        assert next(r for r in rows if r["statistic"] == "relative.0.sigma")[
            "expected"
        ] == pytest.approx(np.deg2rad(3 / level))
        assert not any(r["statistic"].startswith("relative.sc_independence") for r in rows)


@pytest.mark.parametrize("component,requirement", [("density", 0.06), ("velocity_speed", 0.03)])
def test_fractional_gaussian_bias_ou_stationarity_coverage_and_symmetry(component, requirement):
    cfg = normalized_observatory_config(
        {
            component: {
                "model": "gaussian_fractional_bias_plus_ou",
                "requirement_fraction": requirement,
                "sigma_level": 2,
                "bias_variance_fraction": 2,
                "ou_variance_fraction": 2,
                "ou_correlation_time_s": 300,
            }
        }
    )[component]
    t = np.array([0.0, 300.0, 600.0])
    x = np.full((60000, 3, 1), 5.0)
    observed, bias, error = _apply_gaussian_fractional_bias_ou(
        x, t, cfg, component, np.random.default_rng(14)
    )
    u = error - bias[:, None, :]
    sigma = requirement / 2
    normal_moments(bias, sigma / np.sqrt(2))
    for i in range(3):
        normal_moments(u[:, i], sigma / np.sqrt(2))
        normal_moments(error[:, i], sigma)
    initial = error[:, 0, 0]
    assert stats.kstest(initial, stats.norm(scale=sigma).cdf).pvalue > 1e-6
    assert (
        stats.binomtest(
            int(np.sum(np.abs(initial) <= requirement)), len(initial), math.erf(np.sqrt(2))
        ).pvalue
        > 1e-6
    )
    assert stats.binomtest(int(np.sum(initial > 0)), len(initial), 0.5).pvalue > 1e-6
    assert stats.pearsonr(bias[:, 0], u[:, 0, 0]).pvalue > 1e-6
    correlation = np.corrcoef(u[:, 0, 0], u[:, 1, 0])[0, 1]
    assert abs(np.arctanh(correlation) - np.arctanh(np.exp(-1))) < 6 / np.sqrt(len(x) - 3)
    innovations = (u[:, 1:] - np.exp(-1) * u[:, :-1]) / (
        sigma / np.sqrt(2) * np.sqrt(1 - np.exp(-2))
    )
    normal_moments(innovations, 1.0)
    np.testing.assert_array_equal(observed, x * (1 + error))
    second = _apply_gaussian_fractional_bias_ou(x, t, cfg, component, np.random.default_rng(14))
    for a, b in zip((observed, bias, error), second):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("level", [2, 3])
@pytest.mark.parametrize("radius", [0.0, 0.25, 1.7])
def test_dynamic_rayleigh_radius_conversion(level, radius):
    raw = {
        "velocity_angle": {
            "model": "transverse_ou_rotation_total_angle",
            "requirement_deg": radius,
            "sigma_level": level,
        }
    }
    cfg = normalized_observatory_config(raw)["velocity_angle"]
    sigma = transverse_component_sigma_rad(cfg)
    independent = angular_sigma(normalize(raw)["velocity_angle"])
    assert sigma == pytest.approx(independent, rel=2e-15)
    if radius:
        assert stats.rayleigh.cdf(np.deg2rad(radius), scale=sigma) == pytest.approx(
            math.erf(level / math.sqrt(2))
        )
    else:
        assert sigma == 0


@pytest.mark.parametrize("level", [2, 3])
def test_total_angle_quantile_isotropy_ou_and_exact_rotation(level):
    cfg = normalized_observatory_config(
        {
            "velocity_angle": {
                "model": "transverse_ou_rotation_total_angle",
                "requirement_deg": 0.25,
                "sigma_level": level,
            }
        }
    )["velocity_angle"]
    sigma = transverse_component_sigma_rad(cfg)
    # Fixed x direction has deterministic tangent basis y,z. Three timestamps
    # give 20,000 independent paths and 40,000 independent innovations per axis.
    vectors = np.broadcast_to([400.0, 0.0, 0.0], (20000, 3, 1, 3))
    observed, a, b = apply_transverse_angular_ou(
        vectors, sigma, 300.0, 300.0, np.random.default_rng(88)
    )
    angle = np.hypot(a, b)
    p = math.erf(level / math.sqrt(2))
    assert (
        stats.binomtest(int(np.sum(angle[:, 0, 0] <= np.deg2rad(0.25))), len(vectors), p).pvalue
        > 1e-6
    )
    for coordinate in (a, b):
        for i in range(3):
            normal_moments(coordinate[:, i], sigma)
        z = (coordinate[:, 1:] - np.exp(-1) * coordinate[:, :-1]) / (
            sigma * np.sqrt(1 - np.exp(-2))
        )
        normal_moments(z, 1.0)
        rho = np.corrcoef(coordinate[:, 0, 0], coordinate[:, 1, 0])[0, 1]
        assert abs(np.arctanh(rho) - np.arctanh(np.exp(-1))) < 6 / np.sqrt(len(vectors) - 3)
    assert stats.pearsonr(a[:, 0, 0], b[:, 0, 0]).pvalue > 1e-6
    azimuth = (np.arctan2(b[:, 0, 0], a[:, 0, 0]) + np.pi) / (2 * np.pi)
    assert stats.kstest(azimuth, "uniform").pvalue > 1e-6
    # Exact rotation, independent of the production rotation/basis helpers.
    sinc = np.sinc(angle / np.pi)
    expected = np.stack((400 * np.cos(angle), 400 * sinc * a, 400 * sinc * b), axis=-1)
    np.testing.assert_allclose(observed, expected, rtol=3e-15, atol=2e-13)
    np.testing.assert_allclose(np.linalg.norm(observed, axis=-1), 400, rtol=5e-16)
    actual = np.arctan2(np.linalg.norm(observed[..., 1:], axis=-1), observed[..., 0])
    np.testing.assert_allclose(actual, angle, rtol=1e-14, atol=1e-16)
    # Quantile tested through exact binomial inversion rather than a fixed percent tolerance.
    ordered = np.sort(actual[:, 0, 0])
    lo = int(stats.binom.ppf(5e-7, len(vectors), p))
    hi = int(stats.binom.ppf(1 - 5e-7, len(vectors), p)) + 1
    assert ordered[lo - 1] <= np.deg2rad(0.25) <= ordered[min(hi, len(vectors)) - 1]


def product(path, fractional, total_angle, *, disabled=False, historical=False):
    raw = only(
        density={
            "model": "gaussian_fractional_bias_plus_ou"
            if fractional
            else "multiplicative_bias_plus_ou",
            "sigma_level": 2,
            "ou_correlation_time_s": 300,
        },
        velocity_speed={
            "model": "gaussian_fractional_bias_plus_ou"
            if fractional
            else "multiplicative_bias_plus_ou",
            "sigma_level": 2,
            "ou_correlation_time_s": 300,
        },
        velocity_angle={
            "model": "transverse_ou_rotation_total_angle"
            if total_angle
            else "transverse_ou_rotation",
            "sigma_level": 2,
            "ou_correlation_time_s": 300,
        },
    )
    if disabled:
        for section in raw.values():
            section["enabled"] = False
    times = {q: np.array([0.0, 300.0, 600.0]) for q in ("n", "v", "B")}
    truth = {
        "n": np.full((3, 1), 5.0),
        "v": np.broadcast_to([400.0, 0.0, 0.0], (3, 1, 3)),
        "B": np.broadcast_to([0.0, 0.0, 5.0], (3, 1, 3)),
    }
    sampled = {q: np.broadcast_to(v, (40, *v.shape)) for q, v in truth.items()}
    result = apply_observatory_measurement_errors(sampled, times, raw, np.random.default_rng(17))
    metadata = {**result.metadata, "n_mc": 40, "random_seed": 17}
    if historical:
        for section in metadata["observatory_error_config"].values():
            section.pop("error_interpretation", None)
            section.pop("requirement_interpretation", None)
    write_observations(
        path,
        time=times,
        spacecraft_ids=["S0"],
        position={q: np.zeros((1, 3, 3)) for q in times},
        truth=truth,
        observed=result.observed,
        errors=result.errors,
        metadata=metadata,
    )
    if historical:
        with h5py.File(path, "a") as f:
            for ds in f["errors"].values():
                if "bias" in ds.name or "log_error" in ds.name or "angle_" in ds.name:
                    for key in list(ds.attrs):
                        if key != "units" or ds.attrs[key] == "1":
                            del ds.attrs[key]
    return {
        "errors": {**raw, "n_mc": 40, "random_seed": 17},
        "time": {"t_start_s": 0, "t_stop_s": 600, "cadence_s": 300},
    }, result


@pytest.mark.parametrize(
    "fractional,total_angle", [(False, False), (True, False), (False, True), (True, True)]
)
def test_hdf5_schema_and_independent_validator_families(tmp_path, fractional, total_angle):
    p = tmp_path / "errors.h5"
    supplied, result = product(
        p, fractional, total_angle, historical=not fractional and not total_angle
    )
    before = p.read_bytes()
    with h5py.File(p, "r") as f:
        v = Validator(f, supplied, args())
        assert not v.discrepancies
        rows = v.run()
        cfg = json.loads(f["metadata"].attrs["observatory_error_config"])
        suffix = "fractional_error" if fractional else "log_error"
        for name, requirement in (("density", 0.06), ("velocity_speed", 0.03)):
            ds = f[f"errors/{name}_{suffix}"]
            assert ds.shape == (40, 3, 1)
            assert f"errors/{name}_{'log_error' if fractional else 'fractional_error'}" not in f
            if fractional:
                assert ds.attrs["units"] == "1"
                assert ds.attrs["interpretation"] == "fractional_error"
                assert f[f"errors/{name}_bias"].attrs["interpretation"] == "fractional_error_bias"
                assert ds.attrs["model"] == "gaussian_fractional_bias_plus_ou"
                row = next(
                    r
                    for r in rows
                    if r["statistic"] == f"{name}_fractional_error.initial_fractional.sigma"
                )
                assert row["expected"] == requirement / 2 and row["units"] == "fraction"
                row = next(
                    r for r in rows if r["statistic"] == f"{name}_fractional_error.fractional_mean"
                )
                assert row["expected"] == 0
            else:
                assert v.cfg[name]["error_interpretation"] == "log_multiplier"
                row = next(r for r in rows if r["statistic"] == f"{name}_log_error.fractional_mean")
                assert row["expected"] == pytest.approx(np.expm1((requirement / 2) ** 2 / 2))
        row = next(r for r in rows if r["statistic"] == "velocity_angle_alpha.initial_ou.sigma")
        assert row["expected"] == pytest.approx(angular_sigma(v.cfg["velocity_angle"]))
        row = next(r for r in rows if r["statistic"] == "total_angle.requirement_coverage")
        assert row["expected"] == pytest.approx(
            math.erf(np.sqrt(2)) if total_angle else 1 - np.exp(-2)
        )
        if total_angle:
            assert (
                cfg["velocity_angle"]["requirement_interpretation"]
                == "total_angle_confidence_radius"
            )
            assert (
                f["errors/velocity_angle_alpha"].attrs["requirement_interpretation"]
                == "total_angle_confidence_radius"
            )
        notes = " ".join(model_interpretations(v.cfg))
        assert ("zero expected fractional mean" in notes) == fractional
        assert ("a total-angle Rayleigh confidence radius" in notes) == total_angle
    loaded = read_observations(p)
    for name, values in result.errors.items():
        np.testing.assert_array_equal(loaded["errors"][name], values)
    assert p.read_bytes() == before


def test_provenance_wins_over_supplied_yaml_and_rejects_contradictory_metadata(tmp_path):
    p = tmp_path / "errors.h5"
    supplied, _ = product(p, True, True)
    supplied["errors"]["density"]["model"] = "multiplicative_bias_plus_ou"
    with h5py.File(p, "r") as f:
        v = Validator(f, supplied, args())
        assert v.discrepancies
        assert v.cfg["density"]["model"] == "gaussian_fractional_bias_plus_ou"
        rows = v.run()
        assert any(r["statistic"].startswith("density_fractional_error") for r in rows)
    with pytest.raises(ValueError, match="contradicts"):
        normalize(
            {
                "density": {
                    "model": "gaussian_fractional_bias_plus_ou",
                    "error_interpretation": "log_multiplier",
                }
            }
        )
    with h5py.File(p, "a") as f:
        f["errors/density_fractional_error"].attrs["interpretation"] = "log_multiplier"
    with h5py.File(p, "r") as f:
        rows = Validator(f, supplied, args()).run()
    assert (
        next(r for r in rows if r["statistic"] == "density_fractional_error.interpretation")[
            "status"
        ]
        == "FAIL"
    )


def test_disabled_new_model_dataset_names_and_zero_values(tmp_path):
    p = tmp_path / "disabled.h5"
    product(p, True, True, disabled=True)
    with h5py.File(p, "r") as f:
        for component in ("density", "velocity_speed"):
            assert f"errors/{component}_log_error" not in f
            np.testing.assert_array_equal(f[f"errors/{component}_fractional_error"][:], 0)


def test_wrapping_probability_is_reported_without_changing_angle_model(tmp_path):
    p = tmp_path / "large_angle.h5"
    supplied, _ = product(p, False, True)
    with h5py.File(p, "r") as f:
        v = Validator(f, supplied, args())
        v.cfg["velocity_angle"]["requirement_deg"] = 180
        rows = v.run()
    row = next(r for r in rows if r["statistic"] == "total_angle.wrapping_probability")
    assert row["expected"] == pytest.approx(1 - math.erf(np.sqrt(2)))
    assert not row["primary"]
