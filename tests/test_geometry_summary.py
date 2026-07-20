import matplotlib.pyplot as plt
import numpy as np

from hs_observatory.spacecraft.configuration import synthetic_configuration
from hs_observatory.visualization import (
    plot_geometry_summary,
    polyhedron_metrics,
    unique_baselines,
)


def test_unique_baselines_for_nine_spacecraft():
    positions = np.arange(27.0).reshape(9, 3)

    baselines = unique_baselines(positions)

    assert baselines.shape == (36, 3)


def test_polyhedron_metrics_enumerates_four_through_nine_spacecraft():
    positions = np.arange(27.0).reshape(9, 3)

    metrics = polyhedron_metrics(positions)

    assert len(metrics) == 382
    assert {item["n_spacecraft"] for item in metrics} == {4, 5, 6, 7, 8, 9}


def test_regular_tetrahedron_has_small_elongation_and_planarity():
    positions = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]
    )

    metric = polyhedron_metrics(positions)[0]

    assert metric["a"] >= metric["b"] >= metric["c"]
    assert metric["E"] < 1.0e-12
    assert metric["P"] < 1.0e-12
    assert metric["Q"] < 1.0e-12


def test_planar_configuration_has_large_planarity():
    positions = np.array(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ]
    )

    metric = polyhedron_metrics(positions)[0]

    assert metric["a"] >= metric["b"] >= metric["c"]
    assert metric["P"] > 0.99


def test_elongated_configuration_has_large_elongation():
    positions = np.array(
        [
            [-10.0, 0.0, 0.0],
            [-3.0, 0.05, 0.0],
            [4.0, -0.05, 0.0],
            [11.0, 0.0, 0.02],
        ]
    )

    metric = polyhedron_metrics(positions)[0]

    assert metric["a"] >= metric["b"] >= metric["c"]
    assert metric["E"] > 0.98


def test_degenerate_polyhedron_metrics_are_finite():
    positions = np.zeros((4, 3))

    metric = polyhedron_metrics(positions)[0]

    assert metric["a"] == 0.0
    assert metric["b"] == 0.0
    assert metric["c"] == 0.0
    assert np.isfinite(metric["E"])
    assert np.isfinite(metric["P"])
    assert np.isfinite(metric["Q"])


def test_geometry_summary_plot_writes_pdf_and_png(tmp_path):
    spacecraft = synthetic_configuration(np.array([0.0, 1.0]), velocity_km_s=(1.0, 0.0, 0.25))

    figure, diagnostics = plot_geometry_summary(
        spacecraft.position,
        spacecraft.spacecraft_ids,
        time=spacecraft.time,
        metadata={"selected_time": "2030-06-01T00:00:00", "selected_index": 7},
    )
    pdf_path = tmp_path / "summary.pdf"
    png_path = tmp_path / "summary.png"
    figure.savefig(pdf_path)
    figure.savefig(png_path, dpi=100)
    plt.close(figure)

    assert diagnostics["n_baselines"] == 36
    assert diagnostics["n_polyhedra"] == 382
    assert pdf_path.stat().st_size > 0
    assert png_path.stat().st_size > 0
