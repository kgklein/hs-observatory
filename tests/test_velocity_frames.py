from pathlib import Path

import h5py
import numpy as np
import pytest

from hs_observatory.generate import generate_from_config
from hs_observatory.cli import generate_main
from hs_observatory.spacecraft.configuration import SpacecraftConfiguration


def _write_config(
    path: Path,
    output: Path,
    *,
    frame: str | None,
    velocity_error: str = "{model: none}",
) -> Path:
    measurement = "" if frame is None else f"measurement:\n  velocity_frame: {frame}\n"
    path.write_text(
        f"""
spacecraft:
  source: synthetic
  velocity_km_s: [100.0, 0.0, 250.0]
time:
  t_start_s: 0.0
  t_stop_s: 2.0
  cadence_s: 1.0
field:
  model: sine_sum
  background:
    n: 5.0
    v: [0.0, 0.0, 0.0]
    B: [5.0, 0.0, 0.0]
  modes: []
{measurement}errors:
  random_seed: 1234
  n_mc: 1
  variables:
    n: {{model: none}}
    v: {velocity_error}
    B: {{model: none}}
output:
  path: {output}
  write_errors: true
visualization:
  enabled: false
""",
        encoding="utf-8",
    )
    return path


def test_velocity_measurement_frame_defaults_to_spacecraft(tmp_path):
    output = tmp_path / "default.h5"
    config = _write_config(tmp_path / "default.yaml", output, frame=None)

    generate_from_config(config)

    with h5py.File(output) as handle:
        assert handle["metadata"].attrs["velocity_observed_frame"] == "spacecraft"
        assert handle["metadata"].attrs["truth_visualization_velocity_frame"] == "spacecraft"
        assert np.array_equal(handle["observed/v"][0], handle["truth/v_spacecraft_frame"][...])


def test_truth_visualization_uses_spacecraft_frame_by_default(tmp_path, monkeypatch):
    output = tmp_path / "visualization-default.h5"
    config = _write_config(tmp_path / "visualization-default.yaml", output, frame="spacecraft")
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "visualization:\n  enabled: false",
            "visualization:\n  enabled: true",
        ),
        encoding="utf-8",
    )
    captured = {}

    def capture_visualization(**kwargs):
        captured.update(kwargs)
        return tmp_path / "truth.png", tmp_path / "geometry.png"

    monkeypatch.setattr(
        "hs_observatory.generate.save_default_visualizations",
        capture_visualization,
    )
    generate_from_config(config)

    assert captured["velocity_frame"] == "spacecraft"
    assert np.all(captured["truth"]["v"] == [-100.0, 0.0, -250.0])


def test_cli_can_override_truth_visualization_to_plasma(tmp_path, capsys):
    output = tmp_path / "visualization-plasma.h5"
    config = _write_config(tmp_path / "visualization-plasma.yaml", output, frame="spacecraft")

    assert generate_main([str(config), "--truth-velocity-frame", "plasma"]) == 0
    assert str(output) in capsys.readouterr().out
    with h5py.File(output) as handle:
        assert handle["metadata"].attrs["truth_visualization_velocity_frame"] == "plasma"


@pytest.mark.parametrize("frame", ["spacecraft", "plasma"])
def test_constant_spacecraft_velocity_truth_observed_frame_and_metadata(tmp_path, frame):
    output = tmp_path / f"{frame}.h5"
    config = _write_config(tmp_path / f"{frame}.yaml", output, frame=frame)

    generate_from_config(config)

    with h5py.File(output) as handle:
        plasma = handle["truth/v"][...]
        spacecraft = handle["truth/v_spacecraft_frame"][...]
        observed = handle["observed/v"][0]
        assert np.all(plasma == [0.0, 0.0, 0.0])
        assert np.all(spacecraft == [-100.0, 0.0, -250.0])
        expected = spacecraft if frame == "spacecraft" else plasma
        assert np.array_equal(observed, expected)
        metadata = handle["metadata"].attrs
        assert metadata["velocity_truth_frame"] == "plasma"
        assert metadata["velocity_observed_frame"] == frame
        assert metadata["velocity_error_applied_frame"] == frame


def test_velocity_error_is_added_to_selected_spacecraft_frame(tmp_path):
    output = tmp_path / "error.h5"
    config = _write_config(
        tmp_path / "error.yaml",
        output,
        frame="spacecraft",
        velocity_error="{model: gaussian_vector_component, sigma: [1.0, 2.0, 3.0]}",
    )

    generate_from_config(config)

    with h5py.File(output) as handle:
        observed = handle["observed/v"][...]
        selected_truth = handle["truth/v_spacecraft_frame"][...]
        realized_error = handle["errors/v"][...]
        np.testing.assert_allclose(observed - selected_truth[np.newaxis, ...], realized_error)
        assert not np.allclose(observed, handle["truth/v"][...])


def test_spacecraft_configuration_broadcasts_supported_velocity_shapes():
    ids = ("H", "N1")
    time = np.arange(3.0)
    position = np.zeros((2, 3, 3))
    constant = SpacecraftConfiguration(ids, time, position, velocity=[1.0, 2.0, 3.0])
    canonical_values = np.arange(18, dtype=float).reshape(2, 3, 3)
    canonical = SpacecraftConfiguration(ids, time, position, velocity=canonical_values)
    time_first_values = np.arange(18, dtype=float).reshape(3, 2, 3)
    time_first = SpacecraftConfiguration(ids, time, position, velocity=time_first_values)

    assert constant.velocity.shape == (2, 3, 3)
    assert np.all(constant.velocity == [1.0, 2.0, 3.0])
    assert np.array_equal(canonical.velocity, canonical_values)
    assert np.array_equal(time_first.velocity, np.swapaxes(time_first_values, 0, 1))


def test_spacecraft_frame_requires_available_spacecraft_velocity(tmp_path, monkeypatch):
    output = tmp_path / "missing.h5"
    config = _write_config(tmp_path / "missing.yaml", output, frame="spacecraft")

    def configuration_without_velocity(_config, time):
        return SpacecraftConfiguration(
            ("H", "N1"),
            time,
            np.zeros((2, len(time), 3)),
            velocity=None,
        )

    monkeypatch.setattr(
        "hs_observatory.generate.load_spacecraft_configuration",
        configuration_without_velocity,
    )
    with pytest.raises(ValueError, match="no spacecraft velocity is available"):
        generate_from_config(config)


def test_time_dependent_spacecraft_velocity_is_aligned_with_field_output(tmp_path, monkeypatch):
    output = tmp_path / "time-dependent.h5"
    config = _write_config(tmp_path / "time-dependent.yaml", output, frame="spacecraft")

    def time_dependent_configuration(_config, time):
        ids = ("H", "N1")
        velocity = np.zeros((2, len(time), 3))
        velocity[0, :, 0] = 10.0 + time
        velocity[1, :, 2] = 20.0 + 2.0 * time
        return SpacecraftConfiguration(
            ids,
            time,
            np.zeros((2, len(time), 3)),
            velocity=velocity,
        )

    monkeypatch.setattr(
        "hs_observatory.generate.load_spacecraft_configuration",
        time_dependent_configuration,
    )
    generate_from_config(config)

    with h5py.File(output) as handle:
        transformed = handle["truth/v_spacecraft_frame"][...]
        assert np.array_equal(transformed[:, 0, 0], [-10.0, -11.0, -12.0])
        assert np.array_equal(transformed[:, 1, 2], [-20.0, -22.0, -24.0])
