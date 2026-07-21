import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import pytest

from hs_observatory.spacecraft.configuration import synthetic_configuration
from hs_observatory.visualization import (
    EARTH_RADIUS_KM,
    SPACECRAFT_COLORS,
    classify_gse_region,
    formisano_bow_shock_value,
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


def test_formisano_region_classification_uses_bow_shock_surface():
    inside = np.array([0.0, 0.0, 0.0])
    connected = np.array([-50.0 * 6371.0, -50.0 * 6371.0, 0.0])
    solar_wind = np.array([80.0 * 6371.0, 0.0, 0.0])
    positive_line_only = np.array([-40.0 * 6371.0, 80.0 * 6371.0, 0.0])

    assert formisano_bow_shock_value(inside) < 0.0
    assert classify_gse_region(inside) == "Magnetosphere / Magnetosheath"
    assert classify_gse_region(connected) == "Connected Foreshock"
    assert classify_gse_region(solar_wind) == "Solar Wind"
    assert classify_gse_region(positive_line_only) == "Solar Wind"


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


def test_geometry_summary_omits_moon_and_uses_mesocenter_context_marker():
    spacecraft = synthetic_configuration(np.array([0.0, 1.0]), velocity_km_s=(1.0, 0.0, 0.25))
    sample = spacecraft.position[:, 0, :]
    mesocenter = sample.mean(axis=0)

    figure, _diagnostics = plot_geometry_summary(
        spacecraft.position,
        spacecraft.spacecraft_ids,
        time=spacecraft.time,
        metadata={"selected_time": "2030-06-01T00:00:00"},
    )

    legend_labels = [
        text.get_text()
        for axis in figure.axes
        if axis.get_legend() is not None
        for text in axis.get_legend().get_texts()
    ]
    assert "Moon" not in legend_labels
    context_axis = figure.axes[1]
    selected_marker = context_axis.collections[-1].get_offsets()[0]
    assert np.allclose(selected_marker, mesocenter[[0, 1]] / EARTH_RADIUS_KM)
    plt.close(figure)


def test_geometry_summary_baseline_panels_share_limits():
    spacecraft = synthetic_configuration(np.array([0.0, 1.0]))

    figure, _diagnostics = plot_geometry_summary(
        spacecraft.position,
        spacecraft.spacecraft_ids,
        time=spacecraft.time,
    )

    baseline_axes = [figure.axes[index] for index in (2, 7, 8)]
    x_limits = [axis.get_xlim() for axis in baseline_axes]
    y_limits = [axis.get_ylim() for axis in baseline_axes]
    assert x_limits[0] == x_limits[1] == x_limits[2]
    assert y_limits[0] == y_limits[1] == y_limits[2]
    assert x_limits[0] == y_limits[0]
    plt.close(figure)


def test_geometry_summary_uses_requested_two_by_four_panel_order():
    spacecraft = synthetic_configuration(np.array([0.0, 1.0]))

    figure, _diagnostics = plot_geometry_summary(
        spacecraft.position,
        spacecraft.spacecraft_ids,
        time=spacecraft.time,
    )

    top_axes = sorted(
        [axis for axis in figure.axes if axis.get_title() and axis.get_position().y0 > 0.45],
        key=lambda axis: axis.get_position().x0,
    )
    top_titles = [axis.get_title() for axis in top_axes]
    assert top_titles == [
        r"$\Delta Y_{GSE}$ vs $\Delta X_{GSE}$",
        "Polyhedral Configuration Summary",
        r"$|\lambda_Y|$ vs $|\lambda_X|$",
        "Orbit Context:",
    ]
    context_axis = top_axes[3]
    assert "$X_{GSE}$ ($R_E$)" in context_axis.get_xlabel()
    assert "$Y_{GSE}$ ($R_E$)" in context_axis.get_ylabel()

    titles = [axis.get_title() for axis in figure.axes]
    assert titles[5] == r"$\Delta Z_{GSE}$ vs $\Delta X_{GSE}$"
    assert titles[6] == r"$\Delta Z_{GSE}$ vs $\Delta Y_{GSE}$"
    assert titles[7] == r"$|\lambda_Z|$ vs $|\lambda_X|$"
    assert titles[8] == r"$|\lambda_Z|$ vs $|\lambda_Y|$"
    plt.close(figure)


def test_geometry_summary_title_truncates_fractional_seconds():
    spacecraft = synthetic_configuration(np.array([0.0, 1.0]))

    figure, _diagnostics = plot_geometry_summary(
        spacecraft.position,
        spacecraft.spacecraft_ids,
        time=spacecraft.time,
        metadata={"selected_time": "2030-05-31T23:50:35.810100000"},
    )

    assert "2030-05-31T23:50:35" in figure._suptitle.get_text()
    assert "810100000" not in figure._suptitle.get_text()
    assert figure._suptitle.get_ha() == "left"
    plt.close(figure)


def test_geometry_summary_delta_panels_use_sparse_x_ticks():
    spacecraft = synthetic_configuration(np.array([0.0, 1.0]))

    figure, _diagnostics = plot_geometry_summary(
        spacecraft.position,
        spacecraft.spacecraft_ids,
        time=spacecraft.time,
    )

    for axis_index in (0, 5, 6):
        assert np.array_equal(
            figure.axes[axis_index].get_xticks(),
            np.array([-2000.0, -1000.0, 0.0, 1000.0, 2000.0]),
        )
    plt.close(figure)


def test_geometry_summary_delta_panels_use_spacecraft_palette():
    spacecraft = synthetic_configuration(np.array([0.0, 1.0]))

    figure, _diagnostics = plot_geometry_summary(
        spacecraft.position,
        spacecraft.spacecraft_ids,
        time=spacecraft.time,
    )

    first_delta_axis = figure.axes[0]
    assert first_delta_axis.collections[0].get_facecolor()[0] == pytest.approx(
        matplotlib.colors.to_rgba(SPACECRAFT_COLORS["H"])
    )
    assert first_delta_axis.collections[1].get_facecolor()[0] == pytest.approx(
        matplotlib.colors.to_rgba(SPACECRAFT_COLORS["N1"])
    )
    assert first_delta_axis.collections[1].get_edgecolor()[0] == pytest.approx(
        matplotlib.colors.to_rgba("#111111")
    )
    plt.close(figure)
