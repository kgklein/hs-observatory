"""Noise-spectrum integration, geometry propagation, and timing compatibility."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from hs_observatory.analysis.magnetic_gradient import gradient_response_coefficients
from hs_observatory.analysis.timing_calibrate import (
    _settings,
    compute_timing_calibration,
    solve_timing,
    solve_timing_with_variance,
    write_timing_calibration_product,
)
from hs_observatory.analysis.timing_influence import decompose_timing_influence
from hs_observatory.analysis.timing_noise import (
    integrate_noise,
    interpolate_asd,
    noise_divergence_variance,
)
from hs_observatory.cli import plot_timing_calibrate_main
from test_timing_affine import affine_field, write_source

SPECTRUM = Path(__file__).resolve().parents[1] / "configs/helioswarm_mag_scm_noise.yaml"


def model(frequency=(1.0, 10.0), asd=(2.0, 2.0)):
    return {
        "frequency_units": "Hz",
        "asd_units": "nT/sqrt(Hz)",
        "interpretation": "per_component",
        "spectrum": list(zip(frequency, asd)),
    }


def options(**kwargs):
    return {
        "below_range": "flat",
        "above_range": "flat",
        "processing": {"sampling": "ideal_antialias", "averaging_duration_s": 0.0},
        **kwargs,
    }


def test_log_interpolation_psd_and_explicit_extension():
    # ASD=f^-1 at geometric midpoint; power must be squared, not doubled.
    assert interpolate_asd([np.sqrt(10)], [1, 10], [1, 0.1])[0] == pytest.approx(1 / np.sqrt(10))
    np.testing.assert_allclose(
        interpolate_asd([0, 0.1, 100], [1, 10], [1, 0.1], below_range="flat", above_range="flat"),
        [1, 1, 0.1],
    )
    with pytest.raises(ValueError, match="below supplied"):
        interpolate_asd([0], [1, 10], [1, 0.1])
    with pytest.raises(ValueError, match="above supplied"):
        interpolate_asd([11], [1, 10], [1, 0.1])
    actual = integrate_noise(np.arange(20) * 0.1, model(), options())
    np.testing.assert_allclose(actual["component_psd_nT2_per_Hz"], [4, 4])
    assert actual["sigma_B_squared_nT2"] == pytest.approx(4 * 5)
    assert actual["sigma_B_nT"] == pytest.approx(np.sqrt(20))
    # No hidden lower limit 1/duration: changing record length must not change variance.
    longer = integrate_noise(np.arange(200) * 0.1, model(), options())
    assert longer["sigma_B_squared_nT2"] == pytest.approx(20)
    vector = integrate_noise(np.arange(20) * 0.1, model(), options(interpretation="vector_total"))
    assert vector["sigma_B_squared_nT2"] == pytest.approx(20 / 3)


def test_powerlaw_integration_and_declared_filter():
    time = np.arange(20) * 0.05  # Nyquist=10 Hz.
    actual = integrate_noise(time, model(asd=(1, 0.1)), options())
    # Flat extension over 0..1, then integral_1^10 f^-2 df.
    assert actual["sigma_B_squared_nT2"] == pytest.approx(1.9, rel=1e-12)
    processing = {
        "sampling": "ideal_antialias",
        "applied_to_input": True,
        "transfer_power": [[0, 0], [1, 0], [2, 1], [10, 1]],
    }
    filtered = integrate_noise(time, model(), options(below_range="error", processing=processing))
    assert filtered["sigma_B_squared_nT2"] == pytest.approx(4 * (0.5 + 8))
    averaged = integrate_noise(
        time,
        model(),
        options(
            processing={
                "sampling": "ideal_antialias",
                "applied_to_input": True,
                "averaging_duration_s": 0.4,
            }
        ),
    )
    frequency = np.linspace(0, 10, 100_001)
    expected = np.trapezoid(4 * np.sinc(0.4 * frequency) ** 2, frequency)
    assert averaged["sigma_B_squared_nT2"] == pytest.approx(expected, rel=1e-8)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"below_range": "error"}, "below supplied"),
        ({"above_range": "error"}, "above supplied"),
        ({"processing": {}}, "ideal_antialias"),
        (
            {"processing": {"sampling": "ideal_antialias", "averaging_duration_s": 0.1}},
            "applied_to_input",
        ),
        ({"interpretation": "unknown"}, "interpretation"),
    ],
)
def test_noise_invalid_processing(overrides, match):
    with pytest.raises(ValueError, match=match):
        integrate_noise(np.arange(10) * 0.01, model(), options(**overrides))
    with pytest.raises(ValueError, match="uniformly"):
        integrate_noise(np.array([0.0, 0.1, 0.21]), model(), options())


def test_geometry_scale_condition_floor_and_response_equivalence():
    points = np.random.default_rng(19).normal(size=(9, 3))
    subsets = [np.arange(4), np.arange(9)]
    variance, _ = noise_divergence_variance(points[None], subsets, 0.25)
    scaled, _ = noise_divergence_variance((points * 10)[None], subsets, 0.25)
    np.testing.assert_allclose(np.sqrt(scaled), np.sqrt(variance) / 10)
    squeezed, _ = noise_divergence_variance((points * [1, 1, 0.01])[None], subsets, 0.25)
    assert np.all(squeezed > variance)
    for column, indices in enumerate(subsets):
        response = gradient_response_coefficients(points[None, indices])[0]
        assert variance[column] == pytest.approx(0.25 * np.sum(response**2), rel=1e-10)
    moving = np.stack([points, points * 2, points * 3])
    raw, floored = noise_divergence_variance(moving, subsets, 0.25, floor=1.0)
    np.testing.assert_allclose(raw, variance[None] / np.arange(1, 4)[:, None] ** 2)
    np.testing.assert_array_equal(floored, np.maximum(raw, 1.0))
    flat = points.copy()
    flat[:, 2] = 0
    invalid, _ = noise_divergence_variance(flat[None], subsets, 0.25)
    assert np.isinf(invalid).all()


def test_time_dependent_weights_without_changing_solver():
    rng = np.random.default_rng(71)
    residual = rng.normal(size=(17, 3))
    jacobian = rng.normal(size=(17, 3, 8))
    variance = rng.uniform(0.1, 3, size=residual.shape)
    variance[0, 0] = np.inf
    actual = solve_timing_with_variance(residual, jacobian, variance, 0.0125)
    expected = solve_timing(
        residual.reshape(1, -1), jacobian.reshape(1, -1, 8), variance.ravel(), 0.0125
    )
    np.testing.assert_array_equal(actual["correction_s"], expected["correction_s"])
    assert actual["fit_valid"].shape == residual.shape
    assert not actual["fit_valid"][0, 0]
    influence = decompose_timing_influence(residual, jacobian, variance, 0.0125)
    np.testing.assert_allclose(
        influence["predicted_correction_s"], actual["correction_s"], atol=1e-14
    )


def make_config():
    return {
        "sigma_t_s": 0.0125,
        "polyhedron_filter": {"allowed_n": [9]},
        "divergence_normalization": "instrument_noise",
        "instrument_noise": {"spectrum_file": str(SPECTRUM), **options()},
    }


def test_hdf5_units_moving_geometry_truth_independence_cli_and_default(tmp_path):
    time = np.linspace(-2, 2, 61)
    points = np.random.default_rng(19).normal(size=(9, 3))
    positions = points[None] * (1 + 0.01 * time[:, None, None])
    shifts = np.stack([np.zeros(9), np.r_[0, np.linspace(-0.0001, 0.0001, 8)]])
    field = np.stack([affine_field(time[:, None] + shift, points) for shift in shifts])
    source = write_source(tmp_path / "source.h5", time, positions, field[0], field, shifts)
    config = make_config()
    prepared = compute_timing_calibration(source, config)
    assert prepared["variance"].shape == (len(time), 1)
    noise = write_timing_calibration_product(prepared, tmp_path / "noise.h5")
    legacy_config = {"sigma_t_s": 0.0125, "polyhedron_filter": {"allowed_n": [9]}}
    legacy = write_timing_calibration_product(
        compute_timing_calibration(source, legacy_config), tmp_path / "legacy.h5"
    )
    explicit = write_timing_calibration_product(
        compute_timing_calibration(
            source, {**legacy_config, "divergence_normalization": "ideal_divergence"}
        ),
        tmp_path / "explicit.h5",
    )
    with h5py.File(legacy) as a, h5py.File(explicit) as b:
        for key in ("timing/correction_s", "timing/covariance_s2", "diagnostics/chi2_after"):
            np.testing.assert_array_equal(a[key], b[key])
    with h5py.File(noise) as handle:
        assert handle["instrument_noise/sigma_B_nT"].attrs["units"] == "nT"
        assert handle["instrument_noise/sigma_B_squared_nT2"].attrs["units"] == "nT^2"
        assert handle["instrument_noise/component_psd_nT2_per_Hz"].attrs["units"] == "nT^2/Hz"
        name = next(iter(handle["configurations"]))
        group = handle[f"configurations/{name}"]
        assert group["sigma_d_squared"].shape == (61,)
        assert group["sigma_d"].attrs["units"] == "nT/km"
        np.testing.assert_allclose(group["sigma_d"][...] ** 2, group["sigma_d_squared"])
        np.testing.assert_allclose(
            handle["diagnostics/divergence_objective_fraction_after"][...]
            + handle["diagnostics/prior_objective_fraction_after"][...],
            1,
        )
    with h5py.File(source, "r+") as handle:
        handle["truth/B"][...] = 12345
    changed = write_timing_calibration_product(
        compute_timing_calibration(source, config), tmp_path / "changed.h5"
    )
    with h5py.File(noise) as a, h5py.File(changed) as b:
        for key in ("timing/correction_s", "timing/covariance_s2", "diagnostics/chi2_after"):
            np.testing.assert_array_equal(a[key], b[key])
    output = tmp_path / "noise.png"
    assert (
        plot_timing_calibrate_main(
            [str(noise), "--output", str(output), "--dpi", "40", "--compare-products", str(legacy)]
        )
        == 0
    )
    assert (tmp_path / "noise_noise_weights.png").stat().st_size > 1000
    assert (tmp_path / "noise_comparison.png").stat().st_size > 1000


def test_editable_spectrum_and_normalization_validation():
    spectrum = yaml.safe_load(SPECTRUM.read_text())
    assert len(spectrum["spectrum"]) == 12
    actual = integrate_noise(np.arange(10) * 0.03125, spectrum, options())
    assert actual["integration_limits_hz"].tolist() == [0, 16]
    assert actual["sigma_B_nT"] > 0
    assert _settings({})["divergence_normalization"] == "ideal_divergence"
    with pytest.raises(ValueError, match="polyhedron method only"):
        _settings(
            {
                "calibration_method": "all_nine_df_quadratic",
                "divergence_normalization": "instrument_noise",
            }
        )
