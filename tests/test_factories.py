"""Factory facades preserve the canonical selectors and configuration routing."""

import importlib
import subprocess
import sys
from unittest.mock import patch

import pytest

from hs_observatory import errors
from hs_observatory.errors import factory as error_factory
from hs_observatory.errors import gaussian, offsets, ornstein_uhlenbeck
from hs_observatory.fields.factory import field_from_config
from hs_observatory.fields.sine_sum import SineSumField
from hs_observatory.fields.sine_sum import field_from_config as legacy_field_from_config


def test_facades_preserve_legacy_and_subpackage_exports():
    assert field_from_config is legacy_field_from_config
    for name, module in (
        ("error_model_from_config", gaussian),
        ("position_offset_model_from_config", offsets),
        ("time_offset_model_from_config", offsets),
    ):
        assert getattr(error_factory, name) is getattr(module, name)
        assert getattr(errors, name) is getattr(module, name)


def test_sine_sum_construction():
    field = field_from_config({"model": "sine_sum", "background": {"n": 5}})
    assert type(field) is SineSumField
    assert field.background_n == 5


@pytest.mark.parametrize(
    "model,module_name,class_name",
    [
        ("malara_turbulence", "malara", "MalaraTurbulenceField"),
        ("plume_single_mode", "plume_single_mode", "PlumeSingleModeField"),
        ("simulation_sampled", "simulation_sampled", "SimulationSampledField"),
        ("shock", "shock", "ShockField"),
    ],
)
def test_field_routes_forward_configuration(model, module_name, class_name):
    # Exercise selection without loading simulation files or constructing turbulence.
    module = importlib.import_module(f"hs_observatory.fields.{module_name}")
    config = {"model": model}
    with patch.object(module, class_name) as constructor:
        assert field_from_config(config) is constructor.return_value
        constructor.assert_called_once_with(config)


@pytest.mark.parametrize(
    "aliases,parameters,expected",
    [
        (("none", "no_error"), {}, gaussian.NoError),
        (("gaussian_absolute", "absolute_gaussian"), {"sigma": 1}, gaussian.AbsoluteGaussianError),
        (
            ("gaussian_relative", "relative_gaussian"),
            {"sigma": 0.1},
            gaussian.RelativeGaussianError,
        ),
        (
            ("gaussian_vector_component", "vector_component_gaussian"),
            {"sigma": [1, 1, 1]},
            gaussian.VectorComponentGaussianError,
        ),
    ],
)
def test_measurement_error_choices(aliases, parameters, expected):
    for model in aliases:
        assert (
            type(error_factory.error_model_from_config({"model": model, **parameters})) is expected
        )


@pytest.mark.parametrize("quantity", ["n", "v", "B"])
def test_ou_choices(quantity):
    parameters = (
        {"sigma": 1, "tau_s": 2}
        if quantity == "n"
        else {"magnitude_sigma": 1, "magnitude_tau_s": 2}
    )
    expected = (
        ornstein_uhlenbeck.ScalarOrnsteinUhlenbeckError
        if quantity == "n"
        else ornstein_uhlenbeck.VectorOrnsteinUhlenbeckError
    )
    for model in ("ou", "ornstein_uhlenbeck"):
        assert (
            type(
                error_factory.error_model_from_config(
                    {"model": model, **parameters}, quantity=quantity
                )
            )
            is expected
        )


@pytest.mark.parametrize(
    "selector,aliases,parameters,expected",
    [
        (
            error_factory.time_offset_model_from_config,
            ("none", "no_error"),
            {},
            offsets.NoTimeOffset,
        ),
        (
            error_factory.time_offset_model_from_config,
            ("gaussian_constant", "gaussian_time_offset"),
            {"sigma_s": 0.1},
            offsets.GaussianTimeOffset,
        ),
        (
            error_factory.time_offset_model_from_config,
            ("constant_per_spacecraft", "observatory_timing"),
            {},
            offsets.GaussianTimeOffset,
        ),
        (
            error_factory.position_offset_model_from_config,
            ("none", "no_error"),
            {},
            offsets.NoPositionOffset,
        ),
        (
            error_factory.position_offset_model_from_config,
            ("gaussian_constant", "gaussian_position_offset"),
            {"sigma_km": 0.1},
            offsets.GaussianPositionOffset,
        ),
        (
            error_factory.position_offset_model_from_config,
            ("isotropic_3d_gaussian_position_error", "observatory_position"),
            {},
            offsets.GaussianPositionOffset,
        ),
    ],
)
def test_coordinate_error_choices(selector, aliases, parameters, expected):
    for model in aliases:
        assert type(selector({"model": model, **parameters})) is expected


def test_defaults_and_disabled_offsets():
    assert type(error_factory.error_model_from_config(None)) is gaussian.NoError
    for selector, expected in (
        (error_factory.time_offset_model_from_config, offsets.NoTimeOffset),
        (error_factory.position_offset_model_from_config, offsets.NoPositionOffset),
    ):
        assert type(selector(None)) is expected
        assert type(selector({"model": "gaussian_constant", "enabled": False})) is expected


def test_facade_imports_do_not_require_optional_dependencies():
    # A fresh process avoids dependencies loaded during collection of other tests.
    code = """
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'scipy', 'adios2', 'cmcrameri'}:
            raise ImportError('Optional dependency blocked for facade import test')
sys.meta_path.insert(0, BlockOptional())
from hs_observatory.fields.factory import field_from_config
from hs_observatory.errors.factory import error_model_from_config
assert callable(field_from_config)
assert callable(error_model_from_config)
assert 'matplotlib.pyplot' not in sys.modules
"""
    subprocess.run([sys.executable, "-B", "-c", code], check=True, capture_output=True, text=True)
