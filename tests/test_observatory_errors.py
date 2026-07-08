import h5py
import numpy as np
import pytest

from hs_observatory.errors.observatory import (
    apply_observatory_measurement_errors,
    component_sigma_from_3d_radial_3sigma,
    estimate_asd,
    generate_colored_noise_from_asd,
    magnetic_component_asd_1sigma_nT,
    normalized_observatory_config,
    perturbed_pair_baselines,
    rotate_vectors,
    rotation_matrix_from_rotvec,
    sigma_from_3sigma,
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
    assert component_sigma_from_3d_radial_3sigma(2.5) == pytest.approx(
        2.5 / 3.762
    )


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
    result = apply_observatory_measurement_errors(
        sampled, time, cfg, np.random.default_rng(3)
    )
    assert np.var(result.errors["density_log_error"]) == pytest.approx(
        (0.06 / 3.0) ** 2, rel=0.15
    )
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
    result = apply_observatory_measurement_errors(
        sampled, time, cfg, np.random.default_rng(4)
    )
    assert np.allclose(
        np.linalg.norm(result.observed["v"], axis=-1),
        np.linalg.norm(sampled["v"], axis=-1),
    )

    vectors = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]])
    rotated = rotate_vectors(vectors, rotation_matrix_from_rotvec([0.1, -0.2, 0.3]))
    assert np.allclose(np.linalg.norm(rotated, axis=-1), np.linalg.norm(vectors, axis=-1))


def test_magnetic_dc_sigma_and_ac_asd():
    cfg = normalized_observatory_config({"model": "observatory"})
    dc_sigma = sigma_from_3sigma(cfg["magnetic_dc"]["requirement_3sigma_nT_per_axis"])
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
