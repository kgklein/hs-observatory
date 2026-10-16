"""Truth-only generation never enters the Monte Carlo pipeline."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

import hs_observatory.generate as generation
from hs_observatory.config import load_generation_config
from hs_observatory.io.hdf5 import read_observations


def recipe(tmp_path, *, n_mc=0, write_errors=True, visualization=False, model="observatory"):
    config = {
        "spacecraft": {"source": "synthetic", "velocity_km_s": [105.0, 2.0, 245.0]},
        "time": {"t_start_s": 0.0, "t_stop_s": 1.0, "cadence_s": {"n": 0.5, "v": 1.0, "B": 0.25}},
        "field": {
            "model": "sine_sum",
            "background": {"n": 5.0, "v": [400.0, 0.0, 0.0], "B": [0.0, 0.0, 5.0]},
            "modes": [],
        },
        "errors": {
            "n_mc": n_mc,
            "random_seed": 42,
            "model": model,
            "variables": {"n": {"model": "gaussian_relative", "sigma": 0.01}},
        },
        "output": {"path": str(tmp_path / "truth-only.h5"), "write_errors": write_errors},
        "visualization": {"enabled": visualization},
    }
    path = tmp_path / "generation.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def fail_mc(*args, **kwargs):
    raise AssertionError("truth-only generation must not enter the MC pipeline")


@pytest.mark.parametrize("model", ["observatory", "legacy"])
@pytest.mark.parametrize("write_errors", [False, True])
def test_truth_only_skips_mc_and_roundtrips(tmp_path, monkeypatch, model, write_errors):
    config = recipe(tmp_path, model=model, write_errors=write_errors)
    field = generation.field_from_config(load_generation_config(config).data["field"])
    evaluate = field.evaluate_quantity
    calls = []

    def record(quantity, time, position):
        calls.append(quantity)
        return evaluate(quantity, time, position)

    monkeypatch.setattr(field, "evaluate_quantity", record)
    monkeypatch.setattr(generation, "field_from_config", lambda cfg: field)
    for name in (
        "time_offset_model_from_config",
        "position_offset_model_from_config",
        "error_model_from_config",
        "normalized_observatory_config",
        "apply_observatory_measurement_errors",
        "_trajectory_positions_at_times",
        "_trajectory_velocities_at_times",
        "save_observed_visualizations",
    ):
        monkeypatch.setattr(generation, name, fail_mc)
    monkeypatch.setattr(generation.np.random, "default_rng", fail_mc)
    output = generation.generate_from_config(config)
    assert calls == ["n", "v", "B"]
    loaded = read_observations(output)
    for quantity, shape in {"n": (3, 9), "v": (2, 9, 3), "B": (5, 9, 3)}.items():
        assert loaded["truth"][quantity].shape == shape
        assert loaded["observed"][quantity].shape == (0, *shape)
    assert (loaded.get("errors") == {}) if write_errors else ("errors" not in loaded)
    metadata = loaded["metadata"]
    assert metadata["n_mc"] == 0
    assert not metadata["mc_generation_enabled"]
    assert metadata["config_yaml"] == config.read_text()
    error_models = json.loads(metadata["error_models"])
    assert error_models == {
        "measurements": {},
        "time": "not_generated",
        "position": "not_generated",
        "observatory": False,
    }
    for name in ("observed_plot_path", "observed_geometry_plot_path", "observed_alfven_plot_path"):
        assert metadata[name] == "null"
    with h5py.File(output) as handle:
        assert handle["observed/n"].compression == "gzip"
        assert handle["metadata"].attrs["schema_version"] == "2"


def test_truth_visualizations_survive_without_observed_members(tmp_path, monkeypatch):
    config = recipe(tmp_path, visualization=True)
    data = yaml.safe_load(config.read_text())
    # Even an explicit observed request/index cannot enable plots without members.
    data["visualization"].update(observed_enabled=True, mc_index=7, alfven_enabled=True)
    config.write_text(yaml.safe_dump(data))
    monkeypatch.setattr(generation, "save_observed_visualizations", fail_mc)
    output = generation.generate_from_config(config)
    metadata = read_observations(output)["metadata"]
    assert metadata["visualization_enabled"]
    assert metadata["alfven_visualization_enabled"]
    assert not metadata["observed_visualization_enabled"]
    assert not metadata["observed_alfven_visualization_enabled"]
    for name in (
        "truth_plot_path",
        "geometry_plot_path",
        "alfven_plot_path",
        "geometry_summary_pdf_path",
        "geometry_summary_png_path",
        "configuration_3d_pdf_path",
        "configuration_3d_png_path",
    ):
        assert Path(metadata[name]).is_file()
    assert not list(tmp_path.glob("*observed*.png"))


@pytest.mark.parametrize("frame", ["plasma", "spacecraft"])
def test_truth_matches_positive_mc_run_and_default_remains_one(tmp_path, frame):
    config = recipe(tmp_path)
    data = yaml.safe_load(config.read_text())
    data["measurement"] = {"velocity_frame": frame}
    config.write_text(yaml.safe_dump(data))
    zero = read_observations(generation.generate_from_config(config))
    # Omitting n_mc exercises the unchanged default and positive-count path.
    del data["errors"]["n_mc"]
    config.write_text(yaml.safe_dump(data))
    one = read_observations(generation.generate_from_config(config))
    assert one["metadata"]["n_mc"] == 1
    assert one["metadata"]["mc_generation_enabled"]
    for name in ("n", "v", "B", "v_spacecraft_frame"):
        np.testing.assert_array_equal(zero["truth"][name], one["truth"][name])
    for quantity in ("n", "v", "B"):
        np.testing.assert_array_equal(
            zero["spacecraft"]["r"][quantity], one["spacecraft"]["r"][quantity]
        )
        assert one["observed"][quantity].shape[0] == 1
    # A repeated positive run with fixed seeds preserves the existing random stream.
    repeated = read_observations(generation.generate_from_config(config))
    for quantity in ("n", "v", "B"):
        np.testing.assert_array_equal(one["observed"][quantity], repeated["observed"][quantity])


def test_negative_mc_count_rejected(tmp_path):
    config = recipe(tmp_path, n_mc=-1)
    with pytest.raises(ValueError, match="n_mc must be non-negative"):
        load_generation_config(config)
