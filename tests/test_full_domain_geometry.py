"""Periodic Malara box coordinates and opt-in generation plumbing."""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml

from hs_observatory import generate as generation
from hs_observatory.cli import generate_main
from hs_observatory.config import load_generation_config
from hs_observatory.visualization import (
    plot_malara_full_domain_geometry,
    save_malara_full_domain_geometry,
    spacecraft_color,
)


@pytest.mark.parametrize("last_x", [12.0, 32.0])
def test_box_coordinates_wrapping_and_fixed_limits_after_draw(last_x):
    origin = np.array([17.0, -29.0, 41.0])
    lengths = np.array([10.0, 20.0, 30.0])
    position = origin + np.array([[[9, 2, 3], [11, 4, 5], [last_x, 6, 7]]])
    wrapped = origin + np.mod(position[0] - origin, lengths)
    figure = plot_malara_full_domain_geometry(
        position, ["H"], domain_size=lengths, origin=origin, time=np.arange(3.0)
    )
    try:
        figure.canvas.draw()
        assert len(figure.axes) == 4
        assert [text.get_text() for text in figure.legends[0].get_texts()] == ["Hub"]
        for axis, (x, y) in zip(figure.axes[:3], ((0, 1), (0, 2), (1, 2))):
            assert axis.get_xlim() == (origin[x], origin[x] + lengths[x])
            assert axis.get_ylim() == (origin[y], origin[y] + lengths[y])
            assert axis.get_aspect() == 1 and axis.get_adjustable() == "box"
            assert not axis.get_autoscale_on()
            path = axis.lines[0]
            assert path.get_color() == spacecraft_color("H")
            data = path.get_xydata()
            # First crossing breaks the path, with the actual next point retained.
            assert np.isnan(data[1]).all()
            np.testing.assert_allclose(data[np.isfinite(data[:, 0])], wrapped[:, [x, y]])
            np.testing.assert_allclose(axis.collections[0].get_offsets(), wrapped[[0]][:, [x, y]])
            if last_x > 30:
                # Also break a multi-box jump, even when wrapped positions are close.
                assert np.isnan(data[3]).all()
        axis3d = figure.axes[3]
        np.testing.assert_array_equal(
            [axis3d.get_xlim(), axis3d.get_ylim(), axis3d.get_zlim()],
            np.column_stack((origin, origin + lengths)),
        )
        ratio = axis3d.get_box_aspect() / lengths
        np.testing.assert_allclose(ratio, ratio[0])
        assert not axis3d.get_autoscale_on()
        np.testing.assert_allclose(np.array(axis3d.collections[0]._offsets3d).ravel(), wrapped[0])
        assert sum(line.get_color() == "0.5" for line in axis3d.lines) == 12
    finally:
        plt.close(figure)


def test_snapshot_and_save_pdf_png_close_figure(tmp_path):
    paths = save_malara_full_domain_geometry(
        output_path=tmp_path / "run.h5",
        config={"dpi": 40},
        position=np.array([[-1, 2, 3], [4, 5, 6]]),
        spacecraft_ids=["H", "N1"],
        domain_size=[10, 20, 30],
        origin=[0, 0, 0],
        metadata={"selected_time": "2030-06-01T00:00:00"},
    )
    for path, suffix in zip(paths, ("pdf", "png")):
        assert path.name == f"run_geometry_full_domain_2030-06-01T00-00-00.{suffix}"
        assert path.parent == tmp_path / "geometry_overviews"
        assert path.stat().st_size > 1000
    assert not plt.get_fignums()


@pytest.fixture
def recipe(tmp_path):
    config = {
        "spacecraft": {"source": "synthetic", "velocity_km_s": [100, 0, 250]},
        "time": {"t_start_s": 0, "t_stop_s": 1, "cadence_s": {"n": 0.5, "v": 1, "B": 0.25}},
        "field": {
            "model": "malara_turbulence",
            "domain": {"L": [2000, 3000, 4000], "origin": [17, -29, 41]},
            "turbulence": {"n_scales": 1, "injection_scale_index": 1},
            "output_fields": {
                "n": {"background": 5, "fluctuation": "constant"},
                "v": {"background": [400, 0, 0], "fluctuation": "none"},
                "B": {"background": [0, 0, 5], "fluctuation": "none"},
            },
        },
        "errors": {"n_mc": 0},
        "output": {"path": str(tmp_path / "tiny.h5")},
        "visualization": {"enabled": True, "alfven_enabled": False, "dpi": 30},
    }
    path = tmp_path / "generation.yaml"

    def write(**visualization):
        config["visualization"].update(visualization)
        path.write_text(yaml.safe_dump(config))
        return path

    return config, write


@pytest.mark.parametrize(
    "yaml_flag,override,master,expected",
    [
        (None, None, True, False),
        (True, None, True, True),
        (False, True, True, True),
        (True, False, True, False),
        (True, None, False, False),
    ],
)
def test_generation_flag_precedence_and_master_switch(
    recipe, monkeypatch, yaml_flag, override, master, expected
):
    config, write = recipe
    if yaml_flag is not None:
        config["visualization"]["plot_full_domain_geometry"] = yaml_flag
    path = write(enabled=master)
    save = generation.save_malara_full_domain_geometry
    calls = []

    def capture(**kwargs):
        calls.append(kwargs)
        return save(**kwargs)

    monkeypatch.setattr(generation, "save_malara_full_domain_geometry", capture)
    output = generation.generate_from_config(path, plot_full_domain_geometry=override)
    with h5py.File(output) as handle:
        metadata = handle["metadata"].attrs
        for extension in ("pdf", "png"):
            key = f"full_domain_geometry_{extension}_path"
            assert (key in metadata) == expected
            if expected:
                assert Path(metadata[key]).stat().st_size > 1000
        assert len(calls) == int(expected)
        if expected:
            # B supplies the highest-cadence nominal trajectory here.
            np.testing.assert_array_equal(calls[0]["position"], handle["spacecraft/r/B"])
            np.testing.assert_array_equal(calls[0]["time"], handle["time/B"])
            np.testing.assert_array_equal(calls[0]["domain_size"], [2000, 3000, 4000])
            np.testing.assert_array_equal(calls[0]["origin"], [17, -29, 41])
    assert bool(list(output.parent.glob("geometry_overviews/*full_domain*"))) == expected
    if master:
        assert (output.parent / "tiny_geometry.png").is_file()
        assert list(output.parent.glob("geometry_overviews/*geometry_summary*.png"))


@pytest.mark.parametrize("value", ["true", 1, None])
def test_config_requires_boolean(recipe, value):
    _, write = recipe
    with pytest.raises(ValueError, match="plot_full_domain_geometry must be a boolean"):
        load_generation_config(write(plot_full_domain_geometry=value))


def test_non_malara_request_rejected_before_field_creation(recipe, monkeypatch):
    config, write = recipe
    config["field"] = {"model": "sine_sum", "modes": []}
    path = write()

    def forbidden(*args):
        pytest.fail("unsupported request reached field creation")

    monkeypatch.setattr(generation, "field_from_config", forbidden)
    with pytest.raises(ValueError, match="supported only for malara_turbulence"):
        generation.generate_from_config(path, plot_full_domain_geometry=True)
    assert not Path(config["output"]["path"]).exists()


@pytest.mark.parametrize(
    "flag,expected",
    [
        (None, None),
        ("--plot-full-domain-geometry", True),
        ("--no-plot-full-domain-geometry", False),
    ],
)
def test_cli_boolean_override(monkeypatch, flag, expected):
    captured = {}

    def generate(path, **kwargs):
        captured.update(kwargs)
        return Path("tiny.h5")

    monkeypatch.setattr(generation, "generate_from_config", generate)
    assert generate_main(["generation.yaml", *([] if flag is None else [flag])]) == 0
    assert captured["plot_full_domain_geometry"] is expected
