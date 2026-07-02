from pathlib import Path

import h5py
import numpy as np
import pytest

from hs_observatory.fields.simulation_sampled import (
    SimulationSampledField,
    coordinate_box_size_km,
    hub_centered_coordinates,
)
from hs_observatory.generate import generate_from_config
from hs_observatory.io.simulation_npy import (
    PeriodicTrilinearInterpolator,
    load_simulation_npy,
)


def _grid_points(shape, box):
    axes = [np.arange(count, dtype=float) * length / count for count, length in zip(shape, box)]
    return np.meshgrid(*axes, indexing="ij")


def _write_snapshot(path: Path, shape=(4, 5, 6)) -> tuple[Path, np.ndarray]:
    box = np.array([8.0, 10.0, 12.0])
    x, y, z = _grid_points(shape, box)
    values = np.empty((*shape, 7), dtype=float)
    values[..., 0] = 1.0 + x + 2.0 * y + 3.0 * z
    values[..., 1] = x
    values[..., 2] = y
    values[..., 3] = z
    values[..., 4] = 2.0 * x
    values[..., 5] = 3.0 * y
    values[..., 6] = 4.0 * z
    np.save(path, values)
    return path, box


def test_npy_reader_default_and_configurable_fields_first_layout(tmp_path):
    path, _box = _write_snapshot(tmp_path / "snapshot.npy", shape=(2, 3, 4))

    snapshot = load_simulation_npy(path)

    assert snapshot.n.shape == (2, 3, 4)
    assert snapshot.v.shape == (2, 3, 4, 3)
    assert snapshot.B.shape == (2, 3, 4, 3)
    source = np.load(path)
    np.testing.assert_array_equal(snapshot.n, source[..., 0])
    np.testing.assert_array_equal(snapshot.v, source[..., 1:4])
    np.testing.assert_array_equal(snapshot.B, source[..., 4:7])

    fields_first_path = tmp_path / "fields-first.npy"
    np.save(fields_first_path, np.moveaxis(source, -1, 0))
    fields_first = load_simulation_npy(
        fields_first_path,
        {
            "array_order": "fields_first",
            "variables": {"n": 0, "v": [1, 2, 3], "B": [4, 5, 6]},
        },
    )
    np.testing.assert_array_equal(fields_first.v, snapshot.v)


def test_npy_reader_rejects_wrong_shape_and_component_mapping(tmp_path):
    wrong_shape = tmp_path / "wrong.npy"
    np.save(wrong_shape, np.zeros((3, 4, 5)))
    with pytest.raises(ValueError, match="four-dimensional"):
        load_simulation_npy(wrong_shape)

    path, _box = _write_snapshot(tmp_path / "snapshot.npy")
    with pytest.raises(ValueError, match="exceed the component axis"):
        load_simulation_npy(
            path,
            {"variables": {"n": 0, "v": [1, 2, 3], "B": [4, 5, 20]}},
        )


def test_reference_vector_layout_uses_background_for_omitted_fields(tmp_path):
    path = tmp_path / "legacy-vector.npy"
    magnetic = np.zeros((3, 3, 3, 3))
    magnetic[...] = [1.0, 2.0, 3.0]
    np.save(path, magnetic)

    snapshot = load_simulation_npy(
        path, {"array_order": "fields_last", "variables": {"B": [0, 1, 2]}}
    )
    assert snapshot.available_fields == ("B",)
    np.testing.assert_array_equal(snapshot.n, 0.0)
    np.testing.assert_array_equal(snapshot.v, 0.0)

    field = SimulationSampledField(
        {
            "background": {
                "n": 5.0,
                "v": [10.0, 20.0, 30.0],
                "B": [0.0, 0.0, 0.0],
            },
            "simulation": {
                "path": str(path),
                "layout": {"variables": {"B": [0, 1, 2]}},
                "coordinates": {
                    "units": "km",
                    "box_size_km": [3.0, 3.0, 3.0],
                },
                "fields_are": {
                    "n": "fluctuation",
                    "v": "fluctuation",
                    "B": "total",
                },
            },
        }
    )
    output = field.evaluate(np.arange(2.0), np.zeros((2, 2, 3)))
    np.testing.assert_allclose(output["n"], 5.0)
    np.testing.assert_allclose(output["v"], np.broadcast_to([10.0, 20.0, 30.0], output["v"].shape))
    np.testing.assert_allclose(output["B"], np.broadcast_to([1.0, 2.0, 3.0], output["B"].shape))


def test_trilinear_scalar_and_vector_interpolation_is_exact_inside_cells(tmp_path):
    path, box = _write_snapshot(tmp_path / "linear.npy")
    snapshot = load_simulation_npy(path)
    interpolator = PeriodicTrilinearInterpolator(box)
    points = np.array([[0.7, 1.4, 2.2], [3.1, 4.2, 5.3]])

    density = interpolator.interpolate_scalar(snapshot.n, points)
    velocity = interpolator.interpolate_vector(snapshot.v, points)
    magnetic = interpolator.interpolate_vector(snapshot.B, points)

    np.testing.assert_allclose(
        density, 1.0 + points[:, 0] + 2.0 * points[:, 1] + 3.0 * points[:, 2]
    )
    np.testing.assert_allclose(velocity, points)
    np.testing.assert_allclose(magnetic, points * np.array([2.0, 3.0, 4.0]))


def test_periodic_wrapping_gives_identical_samples(tmp_path):
    path, box = _write_snapshot(tmp_path / "periodic.npy")
    snapshot = load_simulation_npy(path)
    interpolator = PeriodicTrilinearInterpolator(box)
    point = np.array([[1.2, 2.3, 3.4]])
    equivalent = np.concatenate((point, point + [box[0], 0.0, 0.0], point - [0.0, box[1], 0.0]))

    sampled = interpolator.sample(snapshot.B, equivalent)

    np.testing.assert_allclose(sampled, np.broadcast_to(sampled[0], sampled.shape))


def test_hub_centered_mapping_and_long_track_wrapping():
    box = np.array([10.0, 20.0, 30.0])
    positions = np.array(
        [
            [[100.0, 200.0, 300.0], [150.0, 250.0, 350.0]],
            [[103.0, 196.0, 305.0], [173.0, 226.0, 415.0]],
        ]
    )

    mapped = hub_centered_coordinates(positions, box, hub_spacecraft=0)

    np.testing.assert_allclose(mapped[0, 0], box / 2.0)
    expected = np.mod(box / 2.0 + positions - positions[0, 0], box)
    np.testing.assert_allclose(mapped, expected)
    assert not np.array_equal(mapped[0, 0], mapped[0, 1])


def test_common_formation_translation_produces_time_varying_samples(tmp_path):
    path = tmp_path / "translated.npy"
    shape = (8, 2, 2)
    values = np.zeros((*shape, 7))
    values[..., 0] = np.arange(shape[0])[:, None, None]
    values[..., 1] = np.arange(shape[0])[:, None, None]
    np.save(path, values)
    field = SimulationSampledField(
        {
            "simulation": {
                "path": str(path),
                "coordinates": {"units": "km", "box_size_km": [8.0, 2.0, 2.0]},
                "fields_are": {"n": "total", "v": "total", "B": "total"},
            }
        }
    )
    time = np.arange(4.0)
    position = np.zeros((2, 4, 3))
    position[:, :, 0] = time[None, :]
    position[1, :, 1] = 0.25

    density = field.evaluate_quantity("n", time, position)

    np.testing.assert_allclose(density[:, 0], [4.0, 5.0, 6.0, 7.0])
    np.testing.assert_allclose(density[:, 1], density[:, 0])
    assert np.ptp(density[:, 0]) > 0.0


def test_coordinate_scaling_supports_km_code_and_reference_rho_p_convention():
    km, _ = coordinate_box_size_km({"units": "km", "box_size_km": [1.0, 2.0, 3.0]})
    code, _ = coordinate_box_size_km(
        {"units": "code", "box_size_code": [1.0, 2.0, 3.0], "code_length_to_km": 10.0}
    )
    rho_p, metadata = coordinate_box_size_km(
        {"units": "rho_p", "box_size_rho_p": [4.0, 5.0, 6.0], "rho_p_km": 20.0}
    )

    np.testing.assert_allclose(km, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(code, [10.0, 20.0, 30.0])
    np.testing.assert_allclose(rho_p, [80.0, 100.0, 120.0])
    assert metadata["rho_p_km"] == 20.0


def test_field_background_scale_factors_and_output_shapes(tmp_path):
    path = tmp_path / "constant.npy"
    values = np.empty((3, 3, 3, 7))
    values[..., 0] = 2.0
    values[..., 1:4] = [1.0, 2.0, 3.0]
    values[..., 4:7] = [4.0, 5.0, 6.0]
    np.save(path, values)
    field = SimulationSampledField(
        {
            "background": {"n": 5.0, "v": [10.0, 20.0, 30.0], "B": [7.0, 8.0, 9.0]},
            "simulation": {
                "path": str(path),
                "coordinates": {"units": "km", "box_size_km": [9.0, 9.0, 9.0]},
                "fields_are": {"n": "fluctuation", "v": "fluctuation", "B": "fluctuation"},
                "scale_factors": {"n": 2.0, "v": 3.0, "B": 0.5},
            },
        }
    )
    position = np.zeros((2, 4, 3))
    position[1, :, 0] = [1.0, 11.0, 21.0, 31.0]
    time = np.arange(4.0)

    output = field.evaluate(time, position)

    assert output["n"].shape == (4, 2)
    assert output["v"].shape == (4, 2, 3)
    assert output["B"].shape == (4, 2, 3)
    np.testing.assert_allclose(output["n"], 9.0)
    np.testing.assert_allclose(output["v"], np.broadcast_to([13.0, 26.0, 39.0], output["v"].shape))
    np.testing.assert_allclose(output["B"], np.broadcast_to([9.0, 10.5, 12.0], output["B"].shape))
    assert field.metadata["simulation_shape"] == [3, 3, 3, 7]
    assert field.metadata["boundary"] == "periodic"


def test_full_generation_pipeline_and_velocity_frame(tmp_path):
    snapshot_path = tmp_path / "snapshot.npy"
    values = np.empty((3, 3, 3, 7))
    values[..., 0] = 5.0
    values[..., 1:4] = [10.0, 20.0, 30.0]
    values[..., 4:7] = [4.0, 5.0, 6.0]
    np.save(snapshot_path, values)
    output_path = tmp_path / "observations.h5"
    config_path = tmp_path / "generate.yaml"
    config_path.write_text(
        f"""
spacecraft:
  source: synthetic
  velocity_km_s: [1.0, 0.0, 0.0]
time:
  t_start_s: 0.0
  t_stop_s: 2.0
  cadence_s: 1.0
field:
  model: simulation_sampled
  background:
    n: 0.0
    v: [0.0, 0.0, 0.0]
    B: [0.0, 0.0, 0.0]
  simulation:
    path: {snapshot_path}
    format: npy
    coordinates:
      units: km
      box_size_km: [100.0, 100.0, 100.0]
    coordinate_mapping: {{method: hub_centered, hub_spacecraft: 0}}
    boundary: periodic
    fields_are: {{n: total, v: total, B: total}}
    scale_factors: {{n: 1.0, v: 1.0, B: 1.0}}
measurement:
  velocity_frame: spacecraft
errors:
  random_seed: 12
  n_mc: 1
  variables:
    n: {{model: none}}
    v: {{model: none}}
    B: {{model: none}}
output:
  path: {output_path}
  write_errors: true
visualization:
  enabled: false
""",
        encoding="utf-8",
    )

    assert generate_from_config(config_path) == output_path

    with h5py.File(output_path) as handle:
        for path in ("truth/n", "truth/v", "truth/B", "observed/n", "observed/v", "observed/B"):
            assert path in handle
        np.testing.assert_allclose(handle["truth/n"], 5.0)
        truth_v = handle["truth/v"][...]
        truth_B = handle["truth/B"][...]
        np.testing.assert_allclose(truth_v, np.broadcast_to([10.0, 20.0, 30.0], truth_v.shape))
        np.testing.assert_allclose(truth_B, np.broadcast_to([4.0, 5.0, 6.0], truth_B.shape))
        observed_v = handle["observed/v"][0]
        spacecraft_v = handle["truth/v_spacecraft_frame"][...]
        expected_spacecraft_v = np.broadcast_to([9.0, 20.0, 30.0], observed_v.shape)
        np.testing.assert_allclose(observed_v, expected_spacecraft_v)
        np.testing.assert_allclose(spacecraft_v, expected_spacecraft_v)
        assert handle["metadata"].attrs["field_model"] == "simulation_sampled"
        field_metadata = handle["metadata"].attrs["field_metadata"]
        assert "simulation_sampled" in field_metadata
        assert str(snapshot_path.resolve()) in field_metadata
