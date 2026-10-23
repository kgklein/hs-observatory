import h5py
import numpy as np
import pytest

from hs_observatory.errors.observatory import (
    apply_observatory_measurement_errors,
    component_sigma_from_3d_radial_requirement,
    component_sigma_from_3d_radial_3sigma,
    estimate_asd,
    generate_colored_noise_from_asd,
    magnetic_component_asd_1sigma_nT,
    normalized_observatory_config,
    perturbed_pair_baselines,
    rotate_vectors,
    rotation_matrix_from_rotvec,
    sigma_from_requirement,
)
from hs_observatory.errors.offsets import (
    position_offset_model_from_config,
    time_offset_model_from_config,
)
from hs_observatory.generate import generate_from_config


def test_observatory_timing_requirement_sigma():
    cfg = normalized_observatory_config({"model": "observatory"})
    offsets = time_offset_model_from_config(cfg["timing"]).draw(
        np.random.default_rng(1), n_mc=20_000, n_spacecraft=1
    )
    assert np.std(offsets) == pytest.approx(12.5e-3 / 3.0, rel=0.03)


def test_observatory_position_radial_requirement():
    cfg = normalized_observatory_config({"model": "observatory"})
    offsets = position_offset_model_from_config(cfg["position"]).draw(
        np.random.default_rng(2), n_mc=120_000, n_spacecraft=1
    )
    magnitude = np.linalg.norm(offsets[:, 0, :], axis=-1)
    assert np.quantile(magnitude, 0.9973) == pytest.approx(2.5, rel=0.04)
    assert component_sigma_from_3d_radial_3sigma(2.5) == pytest.approx(2.5 / 3.762)


def test_two_sigma_position_radial_requirement():
    cfg = normalized_observatory_config(
        {
            "position": {
                "requirement_magnitude_km": 2.5,
                "sigma_level": 2,
            }
        }
    )
    offsets = position_offset_model_from_config(cfg["position"]).draw(
        np.random.default_rng(22), n_mc=120_000, n_spacecraft=1
    )
    magnitude = np.linalg.norm(offsets[:, 0, :], axis=-1)
    assert np.quantile(magnitude, 0.9545) == pytest.approx(2.5, rel=0.03)
    assert component_sigma_from_3d_radial_requirement(2.5, 2) == pytest.approx(
        offsets.std(axis=(0, 1)).mean(), rel=0.02
    )


def test_equivalent_two_and_three_sigma_scalar_requirements():
    assert sigma_from_requirement(2.0, 2) == sigma_from_requirement(3.0, 3)
    cfg_2 = normalized_observatory_config(
        {
            "timing": {"requirement_ms": 2.0, "sigma_level": 2},
            "magnetic_dc": {"requirement_nT_per_axis": 2.0, "sigma_level": 2},
            "magnetic_ac": {"sigma_level": 2},
        }
    )
    cfg_3 = normalized_observatory_config(
        {
            "timing": {"requirement_ms": 3.0, "sigma_level": 3},
            "magnetic_dc": {"requirement_nT_per_axis": 3.0, "sigma_level": 3},
            "magnetic_ac": {
                "sigma_level": 3,
                "anchors": [
                    {"frequency_Hz": 0.001, "asd_pT_per_sqrtHz": 2250.0},
                    {"frequency_Hz": 1.0, "asd_pT_per_sqrtHz": 22.5},
                    {"frequency_Hz": 10.0, "asd_pT_per_sqrtHz": 2.25},
                ],
            },
        }
    )
    assert time_offset_model_from_config(cfg_2["timing"]).sigma_s == pytest.approx(
        time_offset_model_from_config(cfg_3["timing"]).sigma_s
    )
    assert sigma_from_requirement(
        cfg_2["magnetic_dc"]["requirement_nT_per_axis"], 2
    ) == pytest.approx(sigma_from_requirement(cfg_3["magnetic_dc"]["requirement_nT_per_axis"], 3))
    frequency = np.array([0.001, 1.0, 10.0])
    assert np.allclose(
        magnetic_component_asd_1sigma_nT(frequency, cfg_2["magnetic_ac"]),
        magnetic_component_asd_1sigma_nT(frequency, cfg_3["magnetic_ac"]),
    )


def test_legacy_requirements_normalize_and_conflicts_are_rejected():
    cfg = normalized_observatory_config({"timing": {"requirement_3sigma_ms": 12.5}})
    assert cfg["timing"]["requirement_ms"] == 12.5
    assert cfg["timing"]["sigma_level"] == 3
    assert "requirement_3sigma_ms" not in cfg["timing"]

    with pytest.raises(ValueError, match="both"):
        normalized_observatory_config(
            {
                "timing": {
                    "requirement_ms": 12.5,
                    "requirement_3sigma_ms": 12.5,
                }
            }
        )
    with pytest.raises(ValueError, match="implies sigma_level 3"):
        normalized_observatory_config(
            {
                "timing": {
                    "requirement_3sigma_ms": 12.5,
                    "sigma_level": 2,
                }
            }
        )
    with pytest.raises(ValueError, match="must be 2 or 3"):
        normalized_observatory_config({"density": {"sigma_level": 1}})


def test_density_and_velocity_log_error_variance_and_metadata():
    time = {"n": np.arange(4000.0), "v": np.arange(4000.0), "B": np.arange(8.0)}
    sampled = {
        "n": np.ones((32, len(time["n"]), 3)) * 5.0,
        "v": np.broadcast_to(
            np.array([[[[400.0, 30.0, 10.0]]]]),
            (32, len(time["v"]), 3, 3),
        ).copy(),
        "B": np.ones((32, len(time["B"]), 3, 3)) * np.array([5.0, 0.1, 0.2]),
    }
    cfg = normalized_observatory_config(
        {
            "model": "observatory",
            "velocity_angle": {"enabled": False},
            "magnetic_ac": {"enabled": False},
            "alignment": {"enabled": False},
        }
    )
    result = apply_observatory_measurement_errors(sampled, time, cfg, np.random.default_rng(3))
    assert np.var(result.errors["density_log_error"]) == pytest.approx((0.06 / 3.0) ** 2, rel=0.15)
    assert np.var(result.errors["velocity_speed_log_error"]) == pytest.approx(
        (0.03 / 3.0) ** 2, rel=0.15
    )
    assert result.errors["density_bias"].shape == (32, 3)
    assert result.errors["velocity_speed_bias"].shape == (32, 3)


def test_velocity_angle_and_alignment_rotations_preserve_magnitudes():
    time = {"n": np.arange(4.0), "v": np.arange(128.0), "B": np.arange(4.0)}
    sampled = {
        "n": np.ones((4, len(time["n"]), 2)),
        "v": np.broadcast_to(
            np.array([[[[400.0, 20.0, 5.0], [50.0, 200.0, 10.0]]]]),
            (4, len(time["v"]), 2, 3),
        ).copy(),
        "B": np.ones((4, len(time["B"]), 2, 3)),
    }
    cfg = normalized_observatory_config(
        {
            "model": "observatory",
            "density": {"enabled": False},
            "velocity_speed": {"enabled": False},
            "magnetic_dc": {"enabled": False},
            "magnetic_ac": {"enabled": False},
        }
    )
    result = apply_observatory_measurement_errors(sampled, time, cfg, np.random.default_rng(4))
    assert np.allclose(
        np.linalg.norm(result.observed["v"], axis=-1),
        np.linalg.norm(sampled["v"], axis=-1),
    )

    vectors = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]])
    rotated = rotate_vectors(vectors, rotation_matrix_from_rotvec([0.1, -0.2, 0.3]))
    assert np.allclose(np.linalg.norm(rotated, axis=-1), np.linalg.norm(vectors, axis=-1))


def test_magnetic_dc_sigma_and_ac_asd():
    cfg = normalized_observatory_config({"model": "observatory"})
    dc_sigma = sigma_from_requirement(
        cfg["magnetic_dc"]["requirement_nT_per_axis"],
        cfg["magnetic_dc"]["sigma_level"],
    )
    dc = np.random.default_rng(5).normal(scale=dc_sigma, size=(80_000, 3))
    assert np.std(dc[:, 0]) == pytest.approx(0.05, rel=0.03)

    n_times = 4096
    dt = 0.03125
    rng = np.random.default_rng(6)
    psd_accum = None
    for _ in range(48):
        noise = generate_colored_noise_from_asd(
            n_times,
            dt,
            lambda f: magnetic_component_asd_1sigma_nT(f, cfg["magnetic_ac"]),
            rng,
        )
        freq, asd = estimate_asd(noise, dt)
        psd = asd**2
        psd_accum = psd if psd_accum is None else psd_accum + psd
    mean_asd = np.sqrt(psd_accum / 48)
    target = magnetic_component_asd_1sigma_nT(freq, cfg["magnetic_ac"])
    band = (freq >= 0.05) & (freq <= 8.0)
    ratio = np.median(mean_asd[band] / target[band])
    assert ratio == pytest.approx(1.0, rel=0.2)


def test_perturbed_baselines_are_position_differences():
    position = np.array([[0.0, 0.0, 0.0], [2.0, -1.0, 0.5], [-3.0, 0.0, 1.0]])
    baselines = perturbed_pair_baselines(position)
    assert np.allclose(baselines[1, 2], position[1] - position[2])
    assert np.allclose(baselines[2, 1], -baselines[1, 2])


def test_generate_observatory_model_writes_metadata(tmp_path):
    output = tmp_path / "observatory.h5"
    config = tmp_path / "observatory.yaml"
    config.write_text(
        f"""
spacecraft:
  source: synthetic
  velocity_km_s: [10, 0, 0]
time:
  t_start_s: 0
  t_stop_s: 1
  cadence_s:
    n: 0.5
    v: 0.5
    B: 0.25
field:
  model: sine_sum
  background:
    n: 5
    v: [400, 0, 0]
    B: [5, 0, 0]
  modes: []
measurement:
  velocity_frame: spacecraft
errors:
  model: observatory
  random_seed: 7
  n_mc: 2
  magnetic_ac:
    enabled: false
output:
  path: {output}
visualization:
  enabled: false
""",
        encoding="utf-8",
    )
    generate_from_config(config)

    with h5py.File(output) as handle:
        assert handle["observed/n"].shape == (2, 3, 9)
        assert handle["errors/density_bias"].shape == (2, 9)
        assert handle["errors/velocity_alignment_rotvec"].attrs["units"] == "rad"
        assert handle["errors/magnetic_dc_offset"].attrs["units"] == "nT"
        assert "observatory_error_config" in handle["metadata"].attrs
        metadata = handle["metadata"].attrs["observatory_error_config"]
        assert '"requirement_ms": 12.5' in metadata
        assert '"sigma_level": 3' in metadata


def _legacy_reference():
    import json
    from pathlib import Path

    path = Path(__file__).parent / "data/observatory_legacy_reference.json"
    return json.loads(path.read_text())


def _run_reference(config):
    from hs_observatory.errors.offsets import (
        position_offset_model_from_config,
        time_offset_model_from_config,
    )

    reference = _legacy_reference()
    rng = np.random.default_rng(reference["seed"])
    cfg = normalized_observatory_config(config)
    coordinates = {
        "time_offset": time_offset_model_from_config(cfg["timing"]).draw(rng, 2, 2),
        "position_offset": position_offset_model_from_config(cfg["position"]).draw(rng, 2, 2),
    }
    result = apply_observatory_measurement_errors(
        {k: np.asarray(v) for k, v in reference["sampled"].items()},
        {k: np.asarray(v) for k, v in reference["time"].items()},
        config,
        rng,
    )
    return result, coordinates, rng.bit_generator.state


@pytest.mark.parametrize("explicit", [False, True])
def test_legacy_reference_arrays_and_rng_are_unchanged(explicit):
    reference = _legacy_reference()
    config = reference["config"]
    if explicit:
        for component in ("density", "velocity_speed"):
            config[component]["model"] = "multiplicative_bias_plus_ou"
        config["velocity_angle"]["model"] = "transverse_ou_rotation"
    result, coordinates, state = _run_reference(config)
    for name, actual in result.observed.items():
        expected = np.asarray(reference["observed"][name])
        if name == "v":
            # Norms, exp and rotations can round differently across CPU/NumPy paths.
            assert actual.shape == expected.shape
            np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-12)
        else:
            np.testing.assert_array_equal(actual, expected)
    for name, actual in {**coordinates, **result.errors}.items():
        expected = np.asarray(reference["errors"][name])
        if name == "v":
            # This derived residual inherits the observed velocity's rounding.
            assert actual.shape == expected.shape
            np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-12)
        else:
            np.testing.assert_array_equal(actual, expected)
    assert state == reference["rng_state"]


def test_new_options_preserve_draws_and_unrelated_errors():
    reference = _legacy_reference()
    config = reference["config"]
    for component in ("density", "velocity_speed"):
        config[component]["model"] = "gaussian_fractional_bias_plus_ou"
    config["velocity_angle"]["model"] = "transverse_ou_rotation_total_angle"
    result, coordinates, state = _run_reference(config)
    assert state == reference["rng_state"]
    for name in ("density", "velocity_speed"):
        assert name + "_log_error" not in result.errors
        np.testing.assert_array_equal(
            result.errors[name + "_fractional_error"], reference["errors"][name + "_log_error"]
        )
        np.testing.assert_array_equal(
            result.errors[name + "_bias"], reference["errors"][name + "_bias"]
        )
    for name in (
        "B",
        "magnetic_dc_offset",
        "magnetic_ac_noise",
        "magnetic_alignment_matrix",
        "magnetic_alignment_rotvec",
        "velocity_alignment_matrix",
        "velocity_alignment_rotvec",
    ):
        np.testing.assert_array_equal(result.errors[name], reference["errors"][name])
    for name, actual in coordinates.items():
        np.testing.assert_array_equal(actual, reference["errors"][name])
    np.testing.assert_array_equal(result.observed["B"], reference["observed"]["B"])


def _only_options(**sections):
    cfg = {
        name: {"enabled": False}
        for name in (
            "timing",
            "position",
            "density",
            "velocity_speed",
            "velocity_angle",
            "alignment",
            "magnetic_dc",
            "magnetic_ac",
        )
    }
    for name, value in sections.items():
        cfg[name] = {"enabled": True, **value}
    return cfg


RELATIVE_ALIGNMENT = "relative_constant_rotation_B_to_V"


def _relative_inputs(n_mc=4):
    time = {"n": np.arange(3.0), "v": np.arange(5.0), "B": np.arange(8.0)}
    sampled = {
        "n": np.ones((n_mc, 3, 2)),
        "v": np.broadcast_to([400.0, -20.0, 5.0], (n_mc, 5, 2, 3)).copy(),
        "B": np.broadcast_to(np.arange(48.0).reshape(8, 2, 3), (n_mc, 8, 2, 3)).copy(),
    }
    sampled["v"][:, 0] = 0
    return sampled, time


@pytest.mark.parametrize("enabled,requirement", [(True, 3.0), (False, 3.0), (True, 0.0)])
def test_relative_alignment_application_and_velocity_direction_independence(enabled, requirement):
    from scipy.spatial.transform import Rotation

    sampled, time = _relative_inputs()
    cfg = _only_options(
        alignment=dict(
            model=RELATIVE_ALIGNMENT,
            enabled=enabled,
            requirement_deg_per_axis=requirement,
        )
    )
    rng = np.random.default_rng(72)
    before = rng.bit_generator.state
    result = apply_observatory_measurement_errors(sampled, time, cfg, rng)
    vec = result.errors["relative_alignment_rotvec"]
    matrices = result.errors["relative_alignment_matrix"]
    assert vec.shape == (4, 2, 3) and matrices.shape == (4, 2, 3, 3)
    assert not any(
        name.startswith(("velocity_alignment", "magnetic_alignment")) for name in result.errors
    )
    np.testing.assert_array_equal(result.observed["v"], sampled["v"])
    np.testing.assert_array_equal(result.errors["v"], 0)
    np.testing.assert_allclose(
        matrices,
        Rotation.from_rotvec(vec.reshape(-1, 3)).as_matrix().reshape(matrices.shape),
        atol=1e-15,
    )
    np.testing.assert_allclose(
        np.swapaxes(matrices, -1, -2) @ matrices,
        np.broadcast_to(np.eye(3), matrices.shape),
        atol=1e-15,
    )
    np.testing.assert_allclose(np.linalg.det(matrices), 1, atol=1e-15)
    expected = np.einsum("msij,mtsj->mtsi", matrices, sampled["B"])
    np.testing.assert_allclose(result.observed["B"], expected, atol=2e-14)
    np.testing.assert_allclose(
        np.linalg.norm(result.observed["B"], axis=-1),
        np.linalg.norm(sampled["B"], axis=-1),
        atol=2e-14,
    )
    if not enabled or requirement == 0:
        np.testing.assert_array_equal(vec, 0)
        np.testing.assert_array_equal(result.observed["B"], sampled["B"])
    if not enabled:
        assert rng.bit_generator.state == before
    else:
        assert not np.array_equal(vec[0], vec[1]) or requirement == 0
        assert not np.array_equal(vec[:, 0], vec[:, 1]) or requirement == 0
    # Rotation is defined in FC axes, independent of instantaneous V direction.
    sampled["v"] = np.flip(sampled["v"], axis=-1).copy()
    other = apply_observatory_measurement_errors(sampled, time, cfg, np.random.default_rng(72))
    np.testing.assert_array_equal(other.errors["relative_alignment_matrix"], matrices)
    np.testing.assert_array_equal(other.observed["B"], result.observed["B"])


def test_relative_alignment_signed_rotation_and_axis_parallel_vector():
    matrix = rotation_matrix_from_rotvec(np.array([0, 0, np.pi / 2]))
    np.testing.assert_allclose(rotate_vectors([1, 0, 0], matrix), [0, 1, 0], atol=2e-16)
    np.testing.assert_array_equal(rotate_vectors([0, 0, 2], matrix), [0, 0, 2])
    np.testing.assert_allclose(rotate_vectors([0, 1, 0], matrix.T), [1, 0, 0], atol=2e-16)


def test_relative_alignment_rng_compatibility_and_magnetic_noise_order():
    from copy import deepcopy

    config = normalized_observatory_config(deepcopy(_legacy_reference()["config"]))
    legacy, coordinates, state = _run_reference(config)
    config["alignment"]["model"] = RELATIVE_ALIGNMENT
    relative, rel_coordinates, rel_state = _run_reference(config)
    assert state == rel_state
    for name in coordinates:
        np.testing.assert_array_equal(coordinates[name], rel_coordinates[name])
    for name in legacy.errors:
        if name not in {
            "v",
            "velocity_alignment_rotvec",
            "velocity_alignment_matrix",
            "magnetic_alignment_rotvec",
            "magnetic_alignment_matrix",
        }:
            np.testing.assert_array_equal(relative.errors[name], legacy.errors[name])
    for suffix in ("rotvec", "matrix"):
        np.testing.assert_array_equal(
            relative.errors["relative_alignment_" + suffix],
            legacy.errors["magnetic_alignment_" + suffix],
        )
    np.testing.assert_array_equal(relative.observed["B"], legacy.observed["B"])
    reference = _legacy_reference()
    before_rotation = (
        np.asarray(reference["sampled"]["B"])
        + relative.errors["magnetic_dc_offset"][:, None]
        + relative.errors["magnetic_ac_noise"]
    )
    matrix = relative.errors["relative_alignment_matrix"]
    np.testing.assert_allclose(
        relative.observed["B"], np.einsum("msij,mtsj->mtsi", matrix, before_rotation), atol=1e-14
    )
    wrong_order = (
        np.einsum("msij,mtsj->mtsi", matrix, reference["sampled"]["B"])
        + relative.errors["magnetic_dc_offset"][:, None]
        + relative.errors["magnetic_ac_noise"]
    )
    assert not np.allclose(relative.observed["B"], wrong_order, rtol=1e-10, atol=1e-12)
    config["alignment"]["enabled"] = False
    disabled, _, disabled_state = _run_reference(config)
    np.testing.assert_array_equal(relative.observed["v"], disabled.observed["v"])
    config["alignment"]["model"] = "independent_constant_rotation_B_and_V"
    legacy_disabled, _, legacy_disabled_state = _run_reference(config)
    assert disabled_state == legacy_disabled_state
    np.testing.assert_array_equal(disabled.observed["B"], legacy_disabled.observed["B"])


@pytest.mark.parametrize("level", [2, 3])
def test_relative_alignment_normalization_alias_and_uncertainty(level):
    cfg = normalized_observatory_config(
        {
            "alignment": {
                "model": RELATIVE_ALIGNMENT,
                "requirement_deg_per_axis": 3,
                "sigma_level": level,
            }
        }
    )
    assert cfg["alignment"]["sigma_level"] == level
    alias = normalized_observatory_config(
        {
            "alignment": {
                "model": RELATIVE_ALIGNMENT,
                "requirement_3sigma_deg_per_axis": 3,
            }
        }
    )
    assert alias["alignment"]["requirement_deg_per_axis"] == 3
    assert alias["alignment"]["sigma_level"] == 3


@pytest.mark.parametrize("value", [-1, np.inf, np.nan])
def test_relative_alignment_rejects_invalid_requirements(value):
    with pytest.raises(ValueError, match="finite and non-negative"):
        normalized_observatory_config(
            {
                "alignment": {
                    "model": RELATIVE_ALIGNMENT,
                    "requirement_deg_per_axis": value,
                }
            }
        )


def test_alignment_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unsupported alignment.model"):
        normalized_observatory_config({"alignment": {"model": "misspelled"}})


@pytest.mark.parametrize("frame", ["spacecraft", "plasma"])
@pytest.mark.parametrize("n_mc,write_errors", [(2, True), (0, True), (2, False)])
def test_generate_relative_alignment_frame_and_truth_only(tmp_path, frame, n_mc, write_errors):
    import json
    import yaml
    from hs_observatory.io.hdf5 import read_observations

    output = tmp_path / "relative.h5"
    path = tmp_path / "relative.yaml"
    path.write_text(
        yaml.safe_dump(
            dict(
                spacecraft=dict(source="synthetic", velocity_km_s=[10, 0, 0]),
                time=dict(t_start_s=0, t_stop_s=2, cadence_s=1),
                field=dict(
                    model="sine_sum", background=dict(n=5, v=[400, 20, 5], B=[5, 1, 2]), modes=[]
                ),
                measurement=dict(velocity_frame=frame),
                errors=dict(
                    model="observatory",
                    n_mc=n_mc,
                    random_seed=7,
                    **_only_options(alignment=dict(model=RELATIVE_ALIGNMENT)),
                ),
                output=dict(path=str(output), write_errors=write_errors),
                visualization=dict(enabled=False),
            )
        )
    )
    generate_from_config(path)
    loaded = read_observations(output)
    assert loaded["metadata"]["schema_version"] == "2"
    assert loaded["metadata"]["velocity_observed_frame"] == frame
    if n_mc:
        expected = loaded["truth"]["v_spacecraft_frame" if frame == "spacecraft" else "v"]
        np.testing.assert_array_equal(
            loaded["observed"]["v"], np.broadcast_to(expected, loaded["observed"]["v"].shape)
        )
        convention = json.loads(loaded["metadata"]["relative_alignment"])
        assert convention["reference_instrument"] == "FC"
        assert convention["attitude_assumption"] == "constant"
    if n_mc and write_errors:
        assert loaded["errors"]["relative_alignment_rotvec"].shape == (2, 9, 3)
        with h5py.File(output) as handle:
            for suffix, units in (("rotvec", "rad"), ("matrix", "1")):
                ds = handle["errors/relative_alignment_" + suffix]
                assert ds.attrs["units"] == units
                assert ds.attrs["reference_instrument"] == "FC"
                assert ds.attrs["matrix_direction"] == convention["matrix_direction"]
        assert "velocity_alignment_rotvec" not in loaded["errors"]
        assert "magnetic_alignment_rotvec" not in loaded["errors"]
    elif n_mc == 0:
        assert not loaded.get("errors")
    else:
        assert "errors" not in loaded


@pytest.mark.parametrize("component", ["density", "velocity_speed"])
def test_fractional_application_and_zero_vectors(component):
    cfg = _only_options(**{component: {"model": "gaussian_fractional_bias_plus_ou"}})
    reference = _legacy_reference()
    sampled = {k: np.asarray(v) for k, v in reference["sampled"].items()}
    result = apply_observatory_measurement_errors(
        sampled,
        {k: np.asarray(v) for k, v in reference["time"].items()},
        cfg,
        np.random.default_rng(45),
    )
    epsilon = result.errors[component + "_fractional_error"]
    q = "n" if component == "density" else "v"
    expected = sampled[q] * (1 + epsilon if q == "n" else (1 + epsilon)[..., None])
    np.testing.assert_allclose(result.observed[q], expected, rtol=2e-15, atol=0)
    if q == "v":
        np.testing.assert_allclose(np.cross(sampled[q], result.observed[q]), 0, atol=1e-11)
        np.testing.assert_array_equal(result.observed[q][0, 0, 0], 0)


@pytest.mark.parametrize("component", ["density", "velocity_speed"])
@pytest.mark.parametrize("value", [-1.0, -2.0, float("nan"), float("inf")])
def test_nonpositive_or_nonfinite_fractional_factors_raise_even_for_zero_inputs(
    monkeypatch, component, value
):
    import hs_observatory.errors.observatory as model

    def invalid_ou(n_times, dt, sigma, tau, rng, sample_shape=()):
        return np.full((n_times, *sample_shape), value)

    monkeypatch.setattr(model, "generate_ou_process", invalid_ou)
    cfg = _only_options(
        **{
            component: {
                "model": "gaussian_fractional_bias_plus_ou",
                "bias_variance_fraction": 0,
                "ou_variance_fraction": 1,
            }
        }
    )
    sampled = {"n": np.zeros((1, 2, 1)), "v": np.zeros((1, 2, 1, 3)), "B": np.zeros((1, 2, 1, 3))}
    with pytest.raises(
        ValueError,
        match=component
        + r".*gaussian_fractional_bias_plus_ou.*factor=.*\(mc, time, spacecraft\)=\(0, 0, 0\)",
    ):
        model.apply_observatory_measurement_errors(
            sampled, {q: np.arange(2.0) for q in sampled}, cfg, np.random.default_rng(1)
        )


@pytest.mark.parametrize(
    "component,model,key,interpretation",
    [
        ("density", "gaussian_fractional_bias_plus_ou", "error_interpretation", "fractional_error"),
        (
            "velocity_speed",
            "gaussian_fractional_bias_plus_ou",
            "error_interpretation",
            "fractional_error",
        ),
        (
            "velocity_angle",
            "transverse_ou_rotation_total_angle",
            "requirement_interpretation",
            "total_angle_confidence_radius",
        ),
    ],
)
def test_model_interpretations_and_disabled_storage(component, model, key, interpretation):
    cfg = normalized_observatory_config({component: {"model": model}})
    assert cfg[component][key] == interpretation
    with pytest.raises(ValueError, match="conflicts"):
        normalized_observatory_config({component: {"model": model, key: "incorrect"}})
    with pytest.raises(ValueError, match="Unsupported"):
        normalized_observatory_config({component: {"model": "typo"}})
    reference = _legacy_reference()
    cfg = _only_options()
    cfg[component]["model"] = model
    result = apply_observatory_measurement_errors(
        {k: np.asarray(v) for k, v in reference["sampled"].items()},
        {k: np.asarray(v) for k, v in reference["time"].items()},
        cfg,
        np.random.default_rng(4),
    )
    if component != "velocity_angle":
        assert component + "_log_error" not in result.errors
        np.testing.assert_array_equal(result.errors[component + "_fractional_error"], 0)


@pytest.mark.parametrize("radius", [0.0, 0.25])
def test_public_total_angle_dispatch_and_zero_requirement(radius):
    import math
    from hs_observatory.errors.observatory import apply_transverse_angular_ou

    cfg = _only_options(
        velocity_angle={
            "model": "transverse_ou_rotation_total_angle",
            "requirement_deg": radius,
            "sigma_level": 2,
            "ou_correlation_time_s": 300,
        }
    )
    reference = _legacy_reference()
    sampled = {k: np.asarray(v) for k, v in reference["sampled"].items()}
    times = {k: np.asarray(v) for k, v in reference["time"].items()}
    result = apply_observatory_measurement_errors(sampled, times, cfg, np.random.default_rng(71))
    sigma = np.deg2rad(radius) / np.sqrt(-2 * math.log1p(-math.erf(np.sqrt(2))))
    expected, a, b = apply_transverse_angular_ou(
        sampled["v"], sigma, 300, np.diff(times["v"]), np.random.default_rng(71)
    )
    np.testing.assert_allclose(result.observed["v"], expected, rtol=2e-15, atol=1e-14)
    np.testing.assert_allclose(result.errors["velocity_angle_alpha"], a, rtol=2e-15, atol=0)
    np.testing.assert_allclose(result.errors["velocity_angle_beta"], b, rtol=2e-15, atol=0)
    if radius == 0:
        np.testing.assert_array_equal(result.observed["v"], sampled["v"])


@pytest.mark.parametrize(
    "component,key,model",
    [
        ("density", "requirement_fraction", "gaussian_fractional_bias_plus_ou"),
        ("velocity_speed", "requirement_fraction", "gaussian_fractional_bias_plus_ou"),
        ("velocity_angle", "requirement_deg", "transverse_ou_rotation_total_angle"),
    ],
)
@pytest.mark.parametrize("invalid", [-1.0, float("nan"), float("inf")])
def test_invalid_new_requirements_are_rejected(component, key, model, invalid):
    with pytest.raises(ValueError, match="finite and non-negative"):
        normalized_observatory_config({component: {"model": model, key: invalid}})
