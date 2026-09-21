import matplotlib.pyplot as plt
import numpy as np
import pytest

from hs_observatory.spacecraft.configuration import synthetic_configuration
from hs_observatory.visualization import (
    default_configuration_connections,
    helio_swarm_mesocenter,
    parse_configuration_connections,
    plot_3d_configuration,
    save_3d_configuration,
)


def test_helio_swarm_mesocenter_is_arithmetic_mean():
    positions = np.arange(27.0).reshape(9, 3)

    assert np.allclose(helio_swarm_mesocenter(positions), positions.mean(axis=0))


def test_default_connections_are_only_hub_to_node():
    connections = default_configuration_connections()

    assert connections == tuple(("Hub", f"N{index}") for index in range(1, 9))


def test_parse_additional_connections_and_reject_invalid_labels():
    assert parse_configuration_connections(["Hub-N3", "N2-N5", "N5-N2"]) == (
        ("Hub", "N3"),
        ("N2", "N5"),
    )
    with pytest.raises(ValueError, match="Invalid spacecraft label.*N9"):
        parse_configuration_connections(["N2-N9"])


def test_xz_projection_is_on_positive_y_wall_and_legend_lists_spacecraft():
    spacecraft = synthetic_configuration(np.array([0.0]))

    figure, diagnostics = plot_3d_configuration(
        spacecraft.position, spacecraft.spacecraft_ids
    )

    axis = figure.axes[0]
    projected_x, projected_y, projected_z = axis.collections[0]._offsets3d
    assert projected_y[1] == pytest.approx(diagnostics["axis_limit_km"])
    assert projected_x[1] == pytest.approx(projected_x[2])
    assert projected_z[1] == pytest.approx(projected_z[0])
    assert [text.get_text() for text in figure.legends[0].get_texts()] == [
        "Hub", "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8"
    ]
    plt.close(figure)


def test_save_3d_configuration_writes_nonempty_pdf_and_png(tmp_path):
    spacecraft = synthetic_configuration(np.array([0.0]))
    output = tmp_path / "run.h5"

    pdf_path, png_path, diagnostics = save_3d_configuration(
        output_path=output,
        config={"dpi": 80},
        positions=spacecraft.position,
        spacecraft_ids=spacecraft.spacecraft_ids,
        metadata={"selected_time": "2030-06-01T00:00:00", "selected_index": 4},
        extra_connections=["N2-N5"],
    )

    assert pdf_path.stat().st_size > 0
    assert png_path.stat().st_size > 0
    assert diagnostics["extra_connections"] == (("N2", "N5"),)
    assert pdf_path.parent.name == "geometry_overviews"
    plt.close("all")
