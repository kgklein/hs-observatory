import h5py
import numpy as np
import yaml

from hs_observatory.fields.malara import MalaraTurbulenceField
from hs_observatory.fields.sine_sum import field_from_config
from hs_observatory.generate import generate_from_config


def _field_config(*, velocity_fluctuation="none", realization_offset=None):
    velocity = {
        "background": [400.0, 0.0, 0.0],
        "fluctuation": velocity_fluctuation,
        "amplitude": 2.0,
    }
    if realization_offset is not None:
        velocity["realization_offset"] = realization_offset
    return {
        "model": "malara_turbulence",
        "domain": {"L": [100.0, 100.0, 100.0], "units": "km", "origin": [10, 20, 30]},
        "turbulence": {
            "geometry": "isotropic",
            "n_scales": 3,
            "injection_scale_index": 2,
            "spectral_exponent_h": 1.0 / 3.0,
            "intermittency_p": 0.7,
            "base_seed": 42,
            "realization": 3,
        },
        "output_fields": {
            "n": {"background": 5.0, "fluctuation": "none"},
            "v": velocity,
            "B": {
                "background": [5.0, 0.0, 0.0],
                "fluctuation": "malara",
                "amplitude": 0.5,
            },
        },
    }


def test_malara_field_constructs_from_yaml_and_returns_standard_shapes():
    parsed = yaml.safe_load(yaml.safe_dump({"field": _field_config()}))
    field = field_from_config(parsed["field"])
    time = np.arange(4.0)
    position = np.random.default_rng(2).uniform(0.0, 100.0, size=(9, 4, 3))

    values = field.evaluate(time, position)

    assert isinstance(field, MalaraTurbulenceField)
    assert values["n"].shape == (4, 9)
    assert values["v"].shape == (4, 9, 3)
    assert values["B"].shape == (4, 9, 3)
    assert np.all(values["n"] == 5.0)
    assert np.allclose(values["v"], [400.0, 0.0, 0.0])
    assert not np.allclose(values["B"], [5.0, 0.0, 0.0])


def test_velocity_and_magnetic_fields_share_or_offset_realizations():
    shared = MalaraTurbulenceField(_field_config(velocity_fluctuation="malara"))
    independent = MalaraTurbulenceField(
        _field_config(velocity_fluctuation="malara", realization_offset=1000)
    )
    assert shared._quantity_evaluators["v"] is shared._quantity_evaluators["B"]
    assert independent._quantity_evaluators["v"] is not independent._quantity_evaluators["B"]

    time = np.arange(2.0)
    position = np.random.default_rng(7).uniform(0.0, 100.0, size=(3, 2, 3))
    shared_v = shared.evaluate_quantity("v", time, position) - [400.0, 0.0, 0.0]
    shared_B = shared.evaluate_quantity("B", time, position) - [5.0, 0.0, 0.0]
    independent_v = independent.evaluate_quantity("v", time, position) - [400.0, 0.0, 0.0]

    assert np.allclose(shared_v / 2.0, shared_B / 0.5)
    assert not np.allclose(independent_v / 2.0, shared_B / 0.5)


def test_generation_workflow_writes_malara_truth_and_observations(tmp_path):
    output = tmp_path / "malara.h5"
    config = tmp_path / "malara.yaml"
    field_yaml = yaml.safe_dump({"field": _field_config()}, sort_keys=False)
    config.write_text(
        f"""
spacecraft:
  source: synthetic
time:
  t_start_s: 0
  t_stop_s: 0.1
  cadence_s: 0.1
{field_yaml}errors:
  random_seed: 1
  n_mc: 1
  variables:
    n: {{model: none}}
    v: {{model: none}}
    B: {{model: none}}
output:
  path: {output}
visualization:
  enabled: false
""",
        encoding="utf-8",
    )

    assert generate_from_config(config) == output

    with h5py.File(output) as handle:
        assert handle["truth/n"].shape == (2, 9)
        assert handle["truth/v"].shape == (2, 9, 3)
        assert handle["truth/B"].shape == (2, 9, 3)
        assert handle["observed/n"].shape == (1, 2, 9)
        assert handle["observed/v"].shape == (1, 2, 9, 3)
        assert handle["observed/B"].shape == (1, 2, 9, 3)
        assert handle["metadata"].attrs["field_model"] == "malara_turbulence"
        assert np.any(handle["truth/B"][..., 1:] != 0.0)
