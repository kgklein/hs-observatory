import sys
from types import SimpleNamespace

import numpy as np
import pytest

from hs_observatory.fields.simulation_sampled import SimulationSampledField
from hs_observatory.io.simulation_bp import load_simulation_bp


def _install_fake_adios2(monkeypatch, arrays: dict[str, np.ndarray]):
    class FakeStream:
        def __init__(self, path, mode):
            assert mode == "r"
            self.array = arrays[str(path)]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def available_variables(self):
            shape = ",".join(str(value) for value in self.array.shape)
            return {"CartGridField": {"Shape": shape, "Type": "double"}, "time": {}}

        def read(self, name, start=None, count=None):
            assert name == "CartGridField"
            if start is None:
                return self.array.copy()
            slices = tuple(slice(begin, begin + length) for begin, length in zip(start, count))
            return self.array[slices].copy()

    monkeypatch.setitem(sys.modules, "adios2", SimpleNamespace(Stream=FakeStream))


def _bp_arrays(shape=(2, 3, 4)):
    field = np.zeros((*shape, 8))
    moments = np.zeros((*shape, 5))
    indices = np.indices(shape)
    density = 2.0 + 0.1 * indices[0]
    velocity = np.stack((1.0 + indices[0], 2.0 + indices[1], 3.0 + indices[2]), axis=-1)
    magnetic = np.stack((4.0 + indices[0], 5.0 + indices[1], 6.0 + indices[2]), axis=-1)
    moments[..., 0] = density
    moments[..., 1:4] = density[..., None] * velocity
    field[..., 3:6] = magnetic
    return field, moments, density, velocity, magnetic


def test_split_bp_loader_extracts_density_bulk_velocity_and_magnetic_field(tmp_path, monkeypatch):
    field_path = tmp_path / "field.bp"
    moments_path = tmp_path / "ion.bp"
    field_path.touch()
    moments_path.touch()
    field, moments, density, velocity, magnetic = _bp_arrays()
    _install_fake_adios2(monkeypatch, {str(field_path): field, str(moments_path): moments})

    snapshot = load_simulation_bp(field_path, moments_path)

    np.testing.assert_allclose(snapshot.n, density)
    np.testing.assert_allclose(snapshot.v, velocity)
    np.testing.assert_allclose(snapshot.B, magnetic)
    assert snapshot.layout["schema"] == "split_gkeyll_bp"
    assert snapshot.layout["B_components"] == [3, 4, 5]
    assert snapshot.layout["momentum_components"] == [1, 2, 3]
    assert snapshot.layout["velocity_representation"] == "momentum_density"


def test_bp_loader_supports_configurable_components_and_direct_velocity(tmp_path, monkeypatch):
    field_path = tmp_path / "field.bp"
    moments_path = tmp_path / "ion.bp"
    field_path.touch()
    moments_path.touch()
    field = np.zeros((2, 2, 2, 5))
    moments = np.zeros((2, 2, 2, 5))
    field[..., [0, 2, 4]] = [7.0, 8.0, 9.0]
    moments[..., 4] = 6.0
    moments[..., [3, 2, 1]] = [1.0, 2.0, 3.0]
    _install_fake_adios2(monkeypatch, {str(field_path): field, str(moments_path): moments})

    snapshot = load_simulation_bp(
        field_path,
        moments_path,
        {
            "B_components": [0, 2, 4],
            "density_component": 4,
            "momentum_components": [3, 2, 1],
            "velocity_representation": "velocity",
        },
    )

    np.testing.assert_allclose(snapshot.n, 6.0)
    np.testing.assert_allclose(snapshot.v, np.broadcast_to([1.0, 2.0, 3.0], snapshot.v.shape))
    np.testing.assert_allclose(snapshot.B, np.broadcast_to([7.0, 8.0, 9.0], snapshot.B.shape))


def test_bp_loader_can_convert_mass_density_to_number_density(tmp_path, monkeypatch):
    field_path = tmp_path / "field.bp"
    moments_path = tmp_path / "ion.bp"
    field_path.touch()
    moments_path.touch()
    field, moments, _density, velocity, _magnetic = _bp_arrays(shape=(2, 2, 2))
    moments[..., 0] = 100.0
    moments[..., 1:4] = moments[..., 0, None] * velocity
    _install_fake_adios2(monkeypatch, {str(field_path): field, str(moments_path): moments})

    snapshot = load_simulation_bp(
        field_path,
        moments_path,
        {
            "density_representation": "mass_density",
            "density_mass": 100.0,
            "velocity_representation": "momentum_density",
        },
    )

    np.testing.assert_allclose(snapshot.n, 1.0)
    np.testing.assert_allclose(snapshot.v, velocity)
    assert snapshot.layout["density_representation"] == "mass_density"
    assert snapshot.layout["density_mass"] == 100.0


def test_bp_loader_rejects_zero_density_for_momentum_conversion(tmp_path, monkeypatch):
    field_path = tmp_path / "field.bp"
    moments_path = tmp_path / "ion.bp"
    field_path.touch()
    moments_path.touch()
    field, moments, _density, _velocity, _magnetic = _bp_arrays()
    moments[0, 0, 0, 0] = 0.0
    _install_fake_adios2(monkeypatch, {str(field_path): field, str(moments_path): moments})

    with pytest.raises(ValueError, match="finite and nonzero"):
        load_simulation_bp(field_path, moments_path)


def test_simulation_sampled_field_accepts_split_bp_sources(tmp_path, monkeypatch):
    field_path = tmp_path / "field.bp"
    moments_path = tmp_path / "ion.bp"
    field_path.touch()
    moments_path.touch()
    field, moments, _density, _velocity, _magnetic = _bp_arrays(shape=(2, 2, 2))
    _install_fake_adios2(monkeypatch, {str(field_path): field, str(moments_path): moments})
    sampled_field = SimulationSampledField(
        {
            "background": {"n": 0.0, "v": [0.0, 0.0, 0.0], "B": [0.0, 0.0, 0.0]},
            "simulation": {
                "format": "bp",
                "sources": {"field": str(field_path), "moments": str(moments_path)},
                "coordinates": {"units": "km", "box_size_km": [2.0, 2.0, 2.0]},
                "fields_are": {"n": "total", "v": "total", "B": "total"},
            },
        }
    )
    positions = np.zeros((2, 2, 3))

    output = sampled_field.evaluate(np.arange(2.0), positions)

    assert output["n"].shape == (2, 2)
    assert output["v"].shape == (2, 2, 3)
    assert output["B"].shape == (2, 2, 3)
    assert sampled_field.metadata["simulation_format"] == "bp"
    assert sampled_field.metadata["simulation_sources"] == {
        "field": str(field_path.resolve()),
        "moments": str(moments_path.resolve()),
    }
