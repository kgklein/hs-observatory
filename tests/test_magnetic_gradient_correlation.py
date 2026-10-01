from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest

from hs_observatory.analysis.magnetic_gradient_correlation import (
    CORRELATION_SELECTION_ORDER,
    _draw_heatmap_grid,
    compute_magnetic_gradient_correlations,
    gradient_component_correlation,
    pearson_correlation_1d,
    plot_magnetic_gradient_correlation_combined,
    plot_magnetic_gradient_correlation_diagnostic,
    plot_magnetic_gradient_correlation_heatmaps,
    plot_magnetic_gradient_correlation_summary,
    plot_magnetic_gradient_measured_vs_ideal_heatmaps,
    select_correlation_configurations,
    write_magnetic_gradient_correlation_product,
)


def test_pearson_correlation_affine_sign_missing_and_validity():
    x = np.arange(10.0)
    identical = pearson_correlation_1d(x, x, minimum_valid_samples=3)
    negative_copy = pearson_correlation_1d(x, -x, minimum_valid_samples=3)
    positive = pearson_correlation_1d(x, 3.0 * x + 7.0, minimum_valid_samples=3)
    negative = pearson_correlation_1d(x, -2.0 * x + 4.0, minimum_valid_samples=3)
    assert identical.pearson_r == pytest.approx(1.0)
    assert identical.normalized_rmse == pytest.approx(0.0)
    assert negative_copy.pearson_r == pytest.approx(-1.0)
    assert positive.pearson_r == pytest.approx(1.0)
    assert negative.pearson_r == pytest.approx(-1.0)
    assert positive.regression_slope == pytest.approx(3.0)
    assert positive.regression_intercept == pytest.approx(7.0)

    with_missing = pearson_correlation_1d(
        np.array([0.0, 1.0, np.nan, 3.0]),
        np.array([0.0, 1.0, 2.0, np.inf]),
        minimum_valid_samples=2,
    )
    assert with_missing.pearson_r == pytest.approx(1.0)
    assert with_missing.valid_count == 2
    assert with_missing.valid_fraction == pytest.approx(0.5)

    constant = pearson_correlation_1d(
        np.ones(10), x, minimum_valid_samples=3, minimum_standard_deviation=1.0e-15
    )
    assert not constant.valid
    assert np.isnan(constant.pearson_r)
    assert constant.reason == "reference_near_zero_variance"

    too_few = pearson_correlation_1d(
        x,
        x,
        valid=np.arange(10) < 2,
        minimum_valid_samples=3,
    )
    assert not too_few.valid
    assert np.isnan(too_few.pearson_r)
    assert too_few.reason == "insufficient_samples"


def test_gradient_component_correlations_preserve_tensor_orientation():
    rng = np.random.default_rng(42)
    expected = np.array([[-0.9, -0.65, -0.4], [-0.15, 0.1, 0.35], [0.6, 0.8, 0.95]])
    reference = np.empty((128, 3, 3))
    estimate = np.empty_like(reference)
    for component in np.ndindex((3, 3)):
        x = rng.normal(size=128)
        x -= np.mean(x)
        x /= np.sqrt(np.mean(x**2))
        z = rng.normal(size=128)
        z -= np.mean(z)
        z -= np.dot(z, x) / np.dot(x, x) * x
        z /= np.sqrt(np.mean(z**2))
        correlation = expected[component]
        reference[(slice(None), *component)] = x
        estimate[(slice(None), *component)] = correlation * x + np.sqrt(1.0 - correlation**2) * z
    result = gradient_component_correlation(reference, estimate, minimum_valid_samples=30)
    np.testing.assert_allclose(result.pearson_r, expected, atol=1.0e-14)
    assert result.pearson_r[0, 1] == pytest.approx(expected[0, 1])


def _metadata(
    indices: list[int],
    *,
    chi: float,
    condition: float,
    size: float,
) -> dict[str, object]:
    return {
        "spacecraft_indices": np.asarray(indices),
        "number_of_spacecraft": len(indices),
        "chi": chi,
        "elongation": chi,
        "planarity": chi,
        "d_ep": chi,
        "condition_number": condition,
        "characteristic_size": size,
    }


def test_configuration_selection_and_deterministic_tie_breaking():
    configurations = {
        "n4_lex_later": _metadata([0, 2, 3, 4], chi=0.1, condition=2.0, size=50.0),
        "n4_condition": _metadata([0, 1, 2, 3], chi=0.1, condition=1.0, size=70.0),
        "n4_second": _metadata([1, 2, 3, 4], chi=0.2, condition=1.0, size=60.0),
        "multi_size_larger": _metadata([0, 1, 2, 3, 4], chi=0.15, condition=2.0, size=90.0),
        "multi_size_smaller": _metadata([0, 1, 2, 3, 5], chi=0.15, condition=2.0, size=80.0),
        "nine": _metadata(list(range(9)), chi=0.4, condition=4.0, size=200.0),
    }
    selected = select_correlation_configurations(configurations)
    assert selected["most_regular_n4"]["spacecraft_indices"].tolist() == [0, 1, 2, 3]
    assert selected["second_regular_n4"]["spacecraft_indices"].tolist() == [0, 2, 3, 4]
    assert selected["most_regular_n_gt_4"]["spacecraft_indices"].tolist() == [
        0,
        1,
        2,
        3,
        5,
    ]
    assert selected["all_nine"]["spacecraft_indices"].tolist() == list(range(9))


def _curl(gradient: np.ndarray) -> np.ndarray:
    return np.stack(
        (
            gradient[..., 2, 1] - gradient[..., 1, 2],
            gradient[..., 0, 2] - gradient[..., 2, 0],
            gradient[..., 1, 0] - gradient[..., 0, 1],
        ),
        axis=-1,
    )


def _write_configuration(
    parent: h5py.Group,
    name: str,
    indices: list[int],
    *,
    chi: float,
    condition: float,
    size: float,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    n_time = 64
    coarse = rng.normal(size=(n_time, 3, 3))
    coarse[:, 2, 2] = 1.0
    ideal = 0.85 * coarse + rng.normal(scale=0.25, size=coarse.shape)
    ideal[:, 2, 2] = 2.0
    measured = np.stack(
        (
            ideal,
            -ideal,
            ideal + rng.normal(scale=0.3, size=ideal.shape),
        )
    )
    group = parent.create_group(name)
    group.create_dataset("spacecraft_indices", data=indices)
    group.create_dataset(
        "spacecraft_ids",
        data=np.asarray([f"S{index}" for index in indices], dtype=h5py.string_dtype()),
    )
    geometry = group.create_group("geometry")
    for key, value in (
        ("characteristic_size", size),
        ("elongation", chi / 2.0),
        ("planarity", chi / 2.0),
        ("chi", chi),
        ("d_ep", np.nan),
        ("condition_number", condition),
    ):
        geometry.create_dataset(key, data=value)
    truth = group.create_group("truth")
    truth.create_dataset("coarse_gradient", data=coarse)
    truth.create_dataset("coarse_curl", data=_curl(coarse))
    truth.create_dataset("coarse_curl_magnitude", data=np.linalg.norm(_curl(coarse), axis=-1))
    truth.create_dataset("coarse_valid", data=np.ones(n_time, dtype=bool))
    true = group.create_group("true_estimate")
    true.create_dataset("gradient", data=ideal)
    true.create_dataset("curl", data=_curl(ideal))
    true.create_dataset("curl_magnitude", data=np.linalg.norm(_curl(ideal), axis=-1))
    true.create_dataset("valid", data=np.ones(n_time, dtype=bool))
    observed = group.create_group("measured")
    observed.create_dataset("gradient", data=measured)
    observed.create_dataset("curl", data=_curl(measured))
    observed.create_dataset("curl_magnitude", data=np.linalg.norm(_curl(measured), axis=-1))
    observed.create_dataset("valid", data=np.ones((3, n_time), dtype=bool))


def _write_gradient_product(path: Path) -> Path:
    with h5py.File(path, "w") as handle:
        time = handle.create_group("time")
        time.create_dataset("analysis_time", data=np.arange(64, dtype=float))
        time.create_dataset("validity", data=np.ones(64, dtype=bool))
        configurations = handle.create_group("configurations")
        _write_configuration(
            configurations,
            "tetra_best",
            [0, 1, 2, 3],
            chi=0.1,
            condition=1.2,
            size=50.0,
            seed=1,
        )
        _write_configuration(
            configurations,
            "tetra_second",
            [1, 2, 3, 4],
            chi=0.2,
            condition=1.5,
            size=60.0,
            seed=2,
        )
        _write_configuration(
            configurations,
            "multi",
            [0, 1, 2, 3, 4],
            chi=0.15,
            condition=1.4,
            size=80.0,
            seed=3,
        )
        _write_configuration(
            configurations,
            "nine",
            list(range(9)),
            chi=0.3,
            condition=3.0,
            size=150.0,
            seed=4,
        )
    return path


def test_realization_correlations_are_summarized_after_calculation(tmp_path):
    source = _write_gradient_product(tmp_path / "gradient.h5")
    result = compute_magnetic_gradient_correlations(
        source,
        {
            "correlation": {
                "minimum_valid_samples": 10,
                "measured_percentiles": [5, 50, 95],
            }
        },
    )
    gradient = result["configurations"]["most_regular_n4"]["measured_vs_ideal"]["gradient"]
    assert gradient["pearson_r"].shape == (3, 3, 3)
    assert gradient["pearson_r"][0, 0, 0] == pytest.approx(1.0)
    assert gradient["pearson_r"][1, 0, 0] == pytest.approx(-1.0)
    assert gradient["pearson_r_median"][0, 0] > 0.8
    assert np.isnan(gradient["pearson_r"][:, 2, 2]).all()


def test_hdf5_product_and_all_plots_are_source_independent(tmp_path):
    source = _write_gradient_product(tmp_path / "gradient.h5")
    result = compute_magnetic_gradient_correlations(
        source,
        {
            "correlation": {
                "minimum_valid_samples": 10,
                "measured_percentiles": [5, 50, 95],
            },
            "diagnostic": {
                "configuration": "most_regular_n4",
                "component": [0, 1],
                "realization": 2,
            },
        },
    )
    product = write_magnetic_gradient_correlation_product(result, tmp_path / "correlations.h5")
    with h5py.File(product) as handle:
        assert set(handle["configurations"]) == set(CORRELATION_SELECTION_ORDER)
        assert (
            handle["metadata"].attrs["tensor_index_convention"]
            == "gradient[field_component, derivative_direction]"
        )
        measured = handle["configurations/most_regular_n4/measured_vs_ideal/gradient/pearson_r"]
        assert measured.shape == (3, 3, 3)
        assert measured.compression == "gzip"
        assert np.isnan(measured[:, 2, 2]).all()
        figure, axes = plt.subplots(2, 4)
        _draw_heatmap_grid(figure, axes, handle)
        assert all(
            image.get_clim() == (-1.0, 1.0) for axis in axes.ravel() for image in axis.images
        )
        assert axes[0, 0].images[0].cmap.name == "coolwarm_r"
        assert [tick.get_text() for tick in axes[0, 0].get_yticklabels()] == [
            "$B_x$",
            "$B_y$",
            "$B_z$",
        ]
        assert [tick.get_text() for tick in axes[0, 0].get_xticklabels()] == [
            "$\\partial_x$",
            "$\\partial_y$",
            "$\\partial_z$",
        ]
        plt.close(figure)

    source.unlink()
    outputs = (
        plot_magnetic_gradient_correlation_heatmaps(product, tmp_path / "heatmaps.png", dpi=60),
        plot_magnetic_gradient_correlation_summary(product, tmp_path / "summary.png", dpi=60),
        plot_magnetic_gradient_measured_vs_ideal_heatmaps(
            product, tmp_path / "measured_heatmaps.png", dpi=60
        ),
        plot_magnetic_gradient_correlation_combined(product, tmp_path / "combined.png", dpi=60),
        plot_magnetic_gradient_correlation_diagnostic(product, tmp_path / "diagnostic.png", dpi=60),
    )
    assert all(output.stat().st_size > 0 for output in outputs)
    requested_cmaps: list[str] = []
    original_get_cmap = plt.get_cmap

    def recording_get_cmap(name, *args, **kwargs):
        requested_cmaps.append(name)
        return original_get_cmap(name, *args, **kwargs)

    plt.get_cmap = recording_get_cmap
    try:
        plot_magnetic_gradient_measured_vs_ideal_heatmaps(
            product, tmp_path / "measured_heatmaps_check.png", dpi=60
        )
    finally:
        plt.get_cmap = original_get_cmap
    assert "coolwarm_r" in requested_cmaps
    with pytest.raises(ValueError, match="require color_limits"):
        plot_magnetic_gradient_correlation_heatmaps(
            product,
            tmp_path / "invalid.png",
            color_limits=(-0.5, 1.0),
            dpi=60,
        )
