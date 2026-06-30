import matplotlib.pyplot as plt
import numpy as np

from hs_observatory.spacecraft.configuration import synthetic_configuration
from hs_observatory.visualization import (
    SPACECRAFT_COLORS,
    plot_observed_spacecraft_geometry,
    plot_observed_timeseries,
    plot_spacecraft_geometry,
    plot_truth_timeseries,
)


def test_truth_plot_has_seven_panels_and_nine_spacecraft_lines():
    ids = ("H", "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8")
    time = {"n": np.arange(3.0), "v": np.arange(2.0), "B": np.arange(4.0)}
    truth = {
        "n": np.zeros((3, 9)),
        "v": np.zeros((2, 9, 3)),
        "B": np.zeros((4, 9, 3)),
    }

    figure, axes = plot_truth_timeseries(time, truth, ids)

    active_axes = [axis for axis in axes.ravel() if axis.axison]
    assert len(active_axes) == 7
    assert all(len(axis.lines) == 9 for axis in active_axes)
    assert not axes[0, 1].axison
    assert axes[1, 0].get_ylabel().startswith("$v_x$")
    assert axes[1, 1].get_ylabel().startswith("$B_x$")
    assert axes[0, 0].lines[0].get_color() == SPACECRAFT_COLORS["H"]
    assert axes[0, 0].lines[1].get_color() == SPACECRAFT_COLORS["N1"]
    plt.close(figure)


def test_geometry_plot_contains_three_relative_views_and_path_view():
    time = np.arange(3.0)
    spacecraft = synthetic_configuration(time, velocity_km_s=(1.0, 0.0, 0.5))

    figure, axes = plot_spacecraft_geometry(
        spacecraft.time, spacecraft.position, spacecraft.spacecraft_ids
    )

    assert axes.shape == (2, 2)
    assert len(axes[0, 1].lines) >= 9
    hub_offset = axes[0, 0].collections[0].get_offsets()[0]
    assert np.allclose(hub_offset, (0.0, 0.0))
    plt.close(figure)


def test_observed_timeseries_overlays_color_on_grayscale_truth():
    ids = ("H", "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8")
    time = {"n": np.arange(3.0), "v": np.arange(2.0), "B": np.arange(4.0)}
    truth = {
        "n": np.zeros((3, 9)),
        "v": np.zeros((2, 9, 3)),
        "B": np.zeros((4, 9, 3)),
    }
    observed = {
        quantity: np.stack((values + 1.0, values + 2.0))
        for quantity, values in truth.items()
    }

    figure, axes = plot_observed_timeseries(time, truth, observed, ids, mc_index=1)

    active_axes = [axis for axis in axes.ravel() if axis.axison]
    assert len(active_axes) == 7
    assert all(len(axis.lines) == 18 for axis in active_axes)
    assert axes[0, 0].lines[9].get_color() == SPACECRAFT_COLORS["H"]
    assert not axes[0, 1].axison
    plt.close(figure)


def test_observed_geometry_overlays_offsets_on_nominal_positions():
    time = np.arange(3.0)
    spacecraft = synthetic_configuration(time, velocity_km_s=(1.0, 0.0, 0.5))
    offsets = np.zeros((2, 9, 3))
    offsets[1, 1] = [2.0, 3.0, 4.0]

    figure, axes = plot_observed_spacecraft_geometry(
        time, spacecraft.position, offsets, spacecraft.spacecraft_ids, mc_index=1
    )

    assert len(axes[0, 1].lines) == 18
    true_n1 = axes[0, 0].collections[2].get_offsets()[0]
    measured_n1 = axes[0, 0].collections[3].get_offsets()[0]
    assert np.allclose(measured_n1 - true_n1, (2.0, 4.0))
    plt.close(figure)
