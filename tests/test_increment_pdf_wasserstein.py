from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from hs_observatory.analysis.increment_pdf_wasserstein import (
    compute_increment_pdf_wasserstein_diagnostics,
    geometric_mean_normalized_rayleigh_pdf_on_bins,
    normalize_pdf,
    pdf_to_cdf,
    plot_increment_pdf_wasserstein_publication,
    rayleigh_normalized_wasserstein,
    symmetric_pdf_difference,
    wasserstein_from_binned_pdfs,
)


def _write_product(path: Path, lag_centers: np.ndarray | None = None) -> Path:
    lag_centers = np.asarray(
        [100.0, 200.0, 400.0, 800.0] if lag_centers is None else lag_centers,
        dtype=float,
    )
    lag_edges = np.concatenate(
        (
            [lag_centers[0] * 0.75],
            0.5 * (lag_centers[:-1] + lag_centers[1:]),
            [lag_centers[-1] * 1.25],
        )
    )
    x_edges = np.array([0.0, 0.3, 0.9, 1.8, 3.0])
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    truth_base = np.array([0.45, 0.7, 0.3, 0.06])
    with h5py.File(path, "w") as handle:
        root = handle.create_group("results")
        for name, data_group, mc_index, offset in (
            ("truth", "truth", 0, 0.0),
            ("observed_mc000", "observed", 0, -0.08),
            ("observed_mc001", "observed", 1, 0.04),
            ("observed_mc002", "observed", 2, 0.12),
        ):
            group = root.create_group(name)
            group.attrs["data_group"] = data_group
            group.attrs["mc_index"] = mc_index
            lag = group.create_group("lambda")
            lag.create_dataset("bin_edges", data=lag_edges)
            lag.create_dataset("bin_centers", data=lag_centers)
            lag.create_dataset("valid_bin", data=np.ones(lag_centers.size, dtype=bool))
            pdf = group.create_group("pdf").create_group("z_plus")
            pdf.create_dataset("bin_edges", data=x_edges)
            pdf.create_dataset("bin_centers", data=x_centers)
            rows = []
            for index in range(lag_centers.size):
                row = truth_base.copy()
                if data_group == "observed":
                    row += offset * np.array([1.0, -0.5, 0.3, -0.1])
                rows.append(row * (1.0 + 0.1 * index))
            pdf.create_dataset("density", data=np.stack(rows))
    return path


def test_binned_pdf_utilities_use_nonuniform_widths_and_normalize():
    edges = np.array([0.0, 1.0, 3.0, 6.0])
    density = np.array([1.0, 1.0, 1.0])
    normalized = normalize_pdf(density, edges)
    np.testing.assert_allclose(normalized, np.full(3, 1.0 / 6.0))
    np.testing.assert_allclose(pdf_to_cdf(density, edges), [1.0 / 6.0, 0.5, 1.0])
    shifted = np.array([0.0, 0.0, 1.0])
    assert wasserstein_from_binned_pdfs(density, shifted, edges) == pytest.approx(7.0 / 6.0)
    assert wasserstein_from_binned_pdfs(density * 17.0, density, edges) == pytest.approx(0.0)


def test_symmetric_pdf_difference_normalizes_inputs_and_omits_zero_denominator():
    edges = np.array([0.0, 1.0, 2.0])
    np.testing.assert_allclose(
        symmetric_pdf_difference(np.array([1.0, 3.0]), np.array([1.0, 1.0]), edges),
        [1.0 / 3.0, 0.2],
    )
    three_edges = np.array([0.0, 1.0, 2.0, 3.0])
    difference = symmetric_pdf_difference(
        np.array([0.0, 1.0, 1.0]),
        np.array([0.0, 1.0, 1.0]),
        three_edges,
    )
    assert np.isnan(difference[0])
    np.testing.assert_allclose(difference[1:], 0.0)


def test_rayleigh_reference_is_geometric_mean_and_finite_range_normalized():
    edges = np.array([0.0, 0.2, 1.0, 2.5])
    rayleigh = geometric_mean_normalized_rayleigh_pdf_on_bins(edges)
    assert np.sum(rayleigh * np.diff(edges)) == pytest.approx(1.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    rayleigh_on_centers = geometric_mean_normalized_rayleigh_pdf_on_bins(
        centers, bins_are_edges=False
    )
    inferred_widths = np.array([0.5, 0.825, 1.15])
    assert np.sum(rayleigh_on_centers * inferred_widths) == pytest.approx(1.0)

    fine_edges = np.linspace(0.0, 8.0, 40_001)
    fine_centers = 0.5 * (fine_edges[:-1] + fine_edges[1:])
    fine_density = geometric_mean_normalized_rayleigh_pdf_on_bins(fine_edges)
    log_mean = np.sum(np.log(fine_centers) * fine_density * np.diff(fine_edges))
    assert np.exp(log_mean) == pytest.approx(1.0, rel=1.0e-4)


def test_rayleigh_normalized_wasserstein_tolerance_warns():
    edges = np.array([0.0, 0.2, 1.0, 2.5])
    rayleigh = geometric_mean_normalized_rayleigh_pdf_on_bins(edges)
    with pytest.warns(RuntimeWarning, match="at or below"):
        value = rayleigh_normalized_wasserstein(
            rayleigh, rayleigh, edges, denominator_tolerance=1.0e-10
        )
    assert np.isnan(value)


def test_diagnostics_select_second_largest_per_case_and_compute_each_mc(tmp_path):
    first = _write_product(tmp_path / "001.h5")
    second = _write_product(tmp_path / "011.h5", np.array([101.0, 201.0, 401.0, 801.0]))
    missing = tmp_path / "021.h5"
    with pytest.warns(RuntimeWarning, match="not found"):
        result = compute_increment_pdf_wasserstein_diagnostics(
            {
                "001": {"path": first, "label": "all errors"},
                "011": {"path": second, "label": "editable source"},
                "021": {"path": missing, "label": "unfinished"},
            }
        )
    assert result["lag_selection"]["target_lag_km"] == 400.0
    assert result["cases"]["001"]["selected_lag_index"] == 2
    assert result["cases"]["011"]["selected_lag_km"] == 401.0
    assert "021" not in result["cases"]
    distances = result["cases"]["001"]["DW_R_by_mc"]
    assert len(distances) == 3 and all(value is not None for value in distances)
    np.testing.assert_allclose(
        [result["cases"]["001"][key] for key in ("DW_R_p25", "DW_R_p50", "DW_R_p75")],
        np.percentile(distances, [25.0, 50.0, 75.0]),
    )


def test_publication_plot_and_diagnostic_files_use_saved_products(tmp_path):
    product = _write_product(tmp_path / "001.h5")
    figure_path = tmp_path / "figure.png"
    json_path = tmp_path / "diagnostic.json"
    csv_path = tmp_path / "diagnostic.csv"
    output, diagnostics = plot_increment_pdf_wasserstein_publication(
        {"001": {"path": product, "label": "all errors"}},
        output_path=figure_path,
        diagnostic_json_path=json_path,
        diagnostic_csv_path=csv_path,
        dpi=60,
    )
    assert output == figure_path and output.stat().st_size > 0
    assert json.loads(json_path.read_text())["cases"]["001"]["selected_lag_km"] == 400.0
    header = csv_path.read_text().splitlines()[0]
    assert header == (
        "case,label,lag_km,DW_R_p25,DW_R_p50,DW_R_p75,DW_R_median_pdf,"
        "rayleigh_truth_w1,n_mc_valid,n_mc_total"
    )
    assert diagnostics["cases"]["001"]["n_mc_valid"] == 3

    with pytest.raises(ValueError, match="wasserstein_x_max"):
        plot_increment_pdf_wasserstein_publication(
            {"001": {"path": product, "label": "all errors"}},
            output_path=tmp_path / "invalid.png",
            wasserstein_x_max=0.0,
            dpi=60,
        )
    with pytest.raises(ValueError, match="wasserstein_x_min"):
        plot_increment_pdf_wasserstein_publication(
            {"001": {"path": product, "label": "all errors"}},
            output_path=tmp_path / "invalid.png",
            wasserstein_x_min=0.0,
            dpi=60,
        )
    with pytest.raises(ValueError, match="difference_y_min"):
        plot_increment_pdf_wasserstein_publication(
            {"001": {"path": product, "label": "all errors"}},
            output_path=tmp_path / "invalid.png",
            difference_y_min=0.0,
            dpi=60,
        )


def test_publication_plot_can_show_only_current_panels_a_and_c(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    product = _write_product(tmp_path / "001.h5")
    real_close = plt.close
    monkeypatch.setattr(plt, "close", lambda _figure: None)

    output, _diagnostics = plot_increment_pdf_wasserstein_publication(
        {"001": {"path": product, "label": "all errors"}},
        output_path=tmp_path / "panels-a-c.png",
        show_difference_panel=False,
        append_lag_to_filename=True,
        dpi=60,
    )

    figure = plt.gcf()
    assert output.name == "panels-a-c-lambda-400.png"
    assert output.stat().st_size > 0
    assert len(figure.axes) == 2
    assert [axis.get_title() for axis in figure.axes] == [
        r"(a) $p=0.7$, $\lambda_\perp=400\,\mathrm{km}$",
        "(b) PDF mismatch by error case",
    ]
    real_close(figure)
