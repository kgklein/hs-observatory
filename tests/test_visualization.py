import matplotlib.pyplot as plt
import h5py
import numpy as np

from hs_observatory.cli import plot_figure2_main
from hs_observatory.spacecraft.configuration import synthetic_configuration
from hs_observatory.visualization import (
    SPACECRAFT_COLORS,
    plot_figure2_timeseries,
    plot_observed_spacecraft_geometry,
    plot_observed_timeseries,
    plot_observed_velocity_alfven_timeseries,
    plot_spacecraft_geometry,
    plot_truth_timeseries,
    plot_velocity_alfven_timeseries,
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
        quantity: np.stack((values + 1.0, values + 2.0)) for quantity, values in truth.items()
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


def test_velocity_alfven_plot_converts_b_to_velocity_units():
    ids = ("H", "N1")
    time = {"n": np.array([0.0, 2.0]), "v": np.array([0.0, 1.0]), "B": np.array([0.0, 1.0, 2.0])}
    values = {
        "n": np.full((2, 2), 5.0),
        "v": np.zeros((2, 2, 3)),
        "B": np.zeros((3, 2, 3)),
    }
    values["v"][:, :, 0] = 100.0
    values["B"][:, :, 0] = 5.0

    figure, axes = plot_velocity_alfven_timeseries(time, values, ids)

    assert axes.shape == (3, 2)
    assert len(axes[0, 0].lines) == 2
    assert len(axes[0, 1].lines) == 2
    assert axes[0, 0].get_ylabel().startswith("$U_x$")
    assert axes[0, 1].get_ylabel().startswith("$b_x$")
    assert np.all(np.asarray(axes[0, 1].lines[0].get_ydata()) > 40.0)
    assert np.all(np.asarray(axes[0, 1].lines[0].get_ydata()) < 60.0)
    plt.close(figure)


def test_velocity_alfven_plot_can_show_fluctuations():
    ids = ("H",)
    time = {
        "n": np.array([0.0, 1.0, 2.0]),
        "v": np.array([0.0, 1.0, 2.0]),
        "B": np.array([0.0, 1.0, 2.0]),
    }
    values = {
        "n": np.full((3, 1), 5.0),
        "v": np.array([[[10.0, 0.0, 0.0]], [[11.0, 0.0, 0.0]], [[12.0, 0.0, 0.0]]]),
        "B": np.zeros((3, 1, 3)),
    }

    figure, axes = plot_velocity_alfven_timeseries(time, values, ids, fluctuations=True)

    assert np.allclose(axes[0, 0].lines[0].get_ydata(), [-1.0, 0.0, 1.0])
    assert axes[0, 0].get_ylabel().startswith("$\\delta U_x$")
    plt.close(figure)


def test_observed_velocity_alfven_plot_overlays_truth_and_observed():
    ids = ("H", "N1")
    time = {"n": np.array([0.0, 2.0]), "v": np.array([0.0, 1.0]), "B": np.array([0.0, 1.0, 2.0])}
    truth = {
        "n": np.full((2, 2), 5.0),
        "v": np.ones((2, 2, 3)),
        "B": np.ones((3, 2, 3)),
    }
    observed = {
        "n": np.stack((truth["n"], truth["n"] * 1.1)),
        "v": np.stack((truth["v"], truth["v"] * 2.0)),
        "B": np.stack((truth["B"], truth["B"] * 2.0)),
    }

    figure, axes = plot_observed_velocity_alfven_timeseries(time, truth, observed, ids, mc_index=1)

    assert axes.shape == (3, 2)
    assert len(axes[0, 0].lines) == 4
    assert len(axes[0, 1].lines) == 4
    assert axes[0, 0].lines[2].get_color() == SPACECRAFT_COLORS["H"]
    plt.close(figure)


def _write_figure2_input(path, *, scale: float = 1.0, observed_frame: str = "plasma"):
    ids = ("H", "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8")
    time = np.arange(5.0)
    spacecraft_offsets = np.arange(len(ids), dtype=float)[None, :]
    density = np.full((time.size, len(ids)), 5.0e6)
    velocity = np.zeros((time.size, len(ids), 3))
    magnetic = np.zeros_like(velocity)
    velocity[..., 0] = scale * (1000.0 * time[:, None] + 10.0 * spacecraft_offsets)
    magnetic[..., 0] = scale * (5.0e-9 + 0.1e-9 * time[:, None] + 0.01e-9 * spacecraft_offsets)
    magnetic[..., 2] = 4.0e-9
    spacecraft_velocity_km_s = np.zeros((time.size, len(ids), 3))
    spacecraft_velocity_km_s[..., 0] = 0.05
    observed_velocity = velocity + 20.0
    if observed_frame == "spacecraft":
        observed_velocity = observed_velocity - spacecraft_velocity_km_s * 1000.0
    observed = {
        "n": np.stack((density * 1.01, density * 1.02)),
        "v": np.stack((observed_velocity, observed_velocity + 20.0)),
        "B": np.stack((magnetic + 0.05e-9, magnetic + 0.1e-9)),
    }
    with h5py.File(path, "w") as handle:
        handle.create_group("metadata").attrs["velocity_observed_frame"] = observed_frame
        time_group = handle.create_group("time")
        spacecraft = handle.create_group("spacecraft")
        spacecraft.create_dataset("id", data=np.asarray(ids, dtype=h5py.string_dtype("utf-8")))
        spacecraft.create_dataset("velocity", data=spacecraft_velocity_km_s)
        position = spacecraft.create_group("r")
        truth = handle.create_group("truth")
        observed_group = handle.create_group("observed")
        for name, values in (("n", density), ("v", velocity), ("B", magnetic)):
            time_group.create_dataset(name, data=time)
            position.create_dataset(name, data=np.zeros((len(ids), time.size, 3)))
            truth.create_dataset(name, data=values)
            observed_group.create_dataset(name, data=observed[name])


def test_plot_figure2_timeseries_uses_nine_spacecraft_and_no_legend(tmp_path):
    malara = tmp_path / "malara.h5"
    simulation = tmp_path / "simulation.h5"
    output = tmp_path / "figure2.png"
    _write_figure2_input(malara, observed_frame="spacecraft")
    _write_figure2_input(simulation, scale=2.0)

    path = plot_figure2_timeseries(
        malara,
        simulation,
        output,
        velocity_frame="plasma",
        start_time_s=1.0,
        stop_time_s=3.0,
    )

    assert path == output
    assert output.is_file() and output.stat().st_size > 0


def test_plot_figure2_cli_smoke(tmp_path, capsys):
    malara = tmp_path / "malara.h5"
    simulation = tmp_path / "simulation.h5"
    output = tmp_path / "figure2_cli.png"
    _write_figure2_input(malara)
    _write_figure2_input(simulation, scale=2.0)

    assert (
        plot_figure2_main(
            [
                "--malara",
                str(malara),
                "--simulation",
                str(simulation),
                "--output",
                str(output),
                "--start-time",
                "1.0",
                "--stop-time",
                "3.0",
            ]
        )
        == 0
    )
    assert str(output) in capsys.readouterr().out
    assert output.is_file() and output.stat().st_size > 0
