from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from hs_observatory.analysis.magnetic_gradient_wasserstein import (
    compute_magnetic_gradient_wasserstein_diagnostics,
    empirical_wasserstein_1d,
    plot_magnetic_gradient_combined,
    plot_magnetic_gradient_wasserstein,
    write_magnetic_gradient_wasserstein_csv,
    write_magnetic_gradient_wasserstein_json,
)


def _write_product(path: Path) -> Path:
    coarse = np.array([1.0, 2.0, 3.0, 4.0])
    true = np.array([1.0, 2.0, 4.0, 5.0])
    measured = np.stack((true, coarse))
    with h5py.File(path, "w") as handle:
        time = handle.create_group("time")
        time.create_dataset("analysis_time", data=np.arange(4, dtype=float))
        group = handle.create_group("configurations/small")
        group.create_dataset("spacecraft_indices", data=[0, 1, 2, 3])
        group.create_dataset(
            "spacecraft_ids",
            data=np.asarray(["H", "N1", "N2", "N3"], dtype=h5py.string_dtype()),
        )
        geometry = group.create_group("geometry")
        geometry.create_dataset("chi", data=0.2)
        geometry.create_dataset("characteristic_size", data=100.0)
        truth = group.create_group("truth")
        truth.create_dataset("coarse_curl_magnitude", data=coarse)
        truth.create_dataset(
            "coarse_curl",
            data=np.column_stack((coarse, np.zeros((4, 2)))),
        )
        truth.create_dataset("coarse_valid", data=np.ones(4, dtype=bool))
        estimate = group.create_group("true_estimate")
        estimate.create_dataset("curl_magnitude", data=true)
        ideal_curl = np.column_stack((np.zeros(4), true, np.zeros(4)))
        estimate.create_dataset("curl", data=ideal_curl)
        ideal_error = np.sqrt(coarse**2 + true**2)
        estimate.create_dataset("curl_l2_error", data=ideal_error)
        estimate.create_dataset("valid", data=np.ones(4, dtype=bool))
        observed = group.create_group("measured")
        observed.create_dataset("curl_magnitude", data=measured)
        measured_curl = np.stack(
            (
                ideal_curl,
                np.column_stack((coarse, np.zeros((4, 2)))),
            )
        )
        observed.create_dataset("curl", data=measured_curl)
        observed.create_dataset("valid", data=np.ones((2, 4), dtype=bool))
        summary = group.create_group("mc_summary")
        magnitude_summary = summary.create_group("curl_magnitude")
        magnitude_summary.create_dataset("q25", data=np.percentile(measured, 25, axis=0))
        magnitude_summary.create_dataset("median", data=np.median(measured, axis=0))
        magnitude_summary.create_dataset("q75", data=np.percentile(measured, 75, axis=0))
        total_errors = np.linalg.norm(
            measured_curl - np.asarray(truth["coarse_curl"])[np.newaxis, :, :],
            axis=-1,
        )
        error_summary = summary.create_group("curl_l2_error")
        error_summary.create_dataset("q25", data=np.percentile(total_errors, 25, axis=0))
        error_summary.create_dataset("median", data=np.median(total_errors, axis=0))
        error_summary.create_dataset("q75", data=np.percentile(total_errors, 75, axis=0))
    return path


def test_empirical_wasserstein_handles_equal_and_unequal_samples():
    assert empirical_wasserstein_1d([0.0, 2.0], [1.0, 3.0]) == pytest.approx(1.0)
    assert empirical_wasserstein_1d([0.0], [1.0, 2.0]) == pytest.approx(1.5)
    assert empirical_wasserstein_1d([np.nan], [1.0]) != empirical_wasserstein_1d([np.nan], [1.0])


def test_distributional_decomposition_and_normalization(tmp_path):
    product = _write_product(tmp_path / "gradient.h5")
    diagnostics = compute_magnetic_gradient_wasserstein_diagnostics(
        {"200": {"path": product, "label": "all errors"}}
    )
    small = diagnostics["configurations"]["small"]
    entry = small["cases"]["200"]
    assert entry["reference_scale"] == pytest.approx(2.5)
    assert entry["method_w1_raw"] == pytest.approx(0.5)
    assert entry["method_w1_normalized"] == pytest.approx(0.2)
    np.testing.assert_allclose(entry["measurement_w1_raw"]["by_mc"], [0.0, 0.5])
    np.testing.assert_allclose(entry["total_w1_raw"]["by_mc"], [0.5, 0.0])
    np.testing.assert_allclose(entry["pointwise_median_absolute_error_raw"]["by_mc"], [0.0, 0.5])
    assert min(entry["triangle_slack_raw"]["by_mc"]) >= -1.0e-14
    assert len(small["cdf"]["x_normalized"]) == 256


def test_configurable_interval_uses_fifth_and_ninety_fifth_percentiles(tmp_path):
    product = _write_product(tmp_path / "gradient.h5")
    diagnostics = compute_magnetic_gradient_wasserstein_diagnostics(
        {"200": {"path": product, "label": "all errors"}},
        interval_percentiles=(5.0, 95.0),
    )
    summary = diagnostics["configurations"]["small"]["cases"]["200"]["measurement_w1_raw"]
    assert diagnostics["interval_percentiles"] == [5.0, 95.0]
    assert summary["interval_lower"] == pytest.approx(0.025)
    assert summary["interval_upper"] == pytest.approx(0.475)


def test_configurable_interval_rejects_invalid_percentiles(tmp_path):
    product = _write_product(tmp_path / "gradient.h5")
    with pytest.raises(ValueError, match="interval_percentiles"):
        compute_magnetic_gradient_wasserstein_diagnostics(
            {"200": product},
            interval_percentiles=(95.0, 5.0),
        )


def test_missing_cases_warn_and_outputs_are_written(tmp_path):
    product = _write_product(tmp_path / "gradient.h5")
    with pytest.warns(RuntimeWarning, match="not found"):
        diagnostics = compute_magnetic_gradient_wasserstein_diagnostics(
            {
                "200": {"path": product, "label": "all errors"},
                "210": {"path": tmp_path / "missing.h5", "label": "timing"},
            }
        )
    assert diagnostics["case_order"] == ["200"]
    figure = plot_magnetic_gradient_wasserstein(diagnostics, tmp_path / "figure.png", dpi=60)
    json_path = write_magnetic_gradient_wasserstein_json(diagnostics, tmp_path / "diagnostics.json")
    csv_path = write_magnetic_gradient_wasserstein_csv(diagnostics, tmp_path / "diagnostics.csv")
    assert figure.stat().st_size > 0
    assert json.loads(json_path.read_text())["quantity"] == "curl_magnitude"
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 1
    assert rows[0]["configuration"] == "small"


def test_combined_figure_selects_tetrahedron_and_is_written(tmp_path):
    product = _write_product(tmp_path / "gradient.h5")
    diagnostics = compute_magnetic_gradient_wasserstein_diagnostics(
        {"400": {"path": product, "label": "all errors"}},
        all_errors_case="400",
        interval_percentiles=(5.0, 95.0),
    )
    output, category = plot_magnetic_gradient_combined(
        diagnostics,
        product,
        tmp_path / "combined.png",
        start_time_s=1.0,
        stop_time_s=3.0,
        dpi=60,
    )
    assert category == "small"
    assert output.stat().st_size > 0
