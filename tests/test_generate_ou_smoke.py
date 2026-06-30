import h5py

from hs_observatory.generate import generate_from_config


def test_generation_with_ou_errors_and_native_cadences(tmp_path):
    output = tmp_path / "ou-observations.h5"
    config = tmp_path / "ou.yaml"
    config.write_text(
        f"""
spacecraft:
  source: synthetic
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
errors:
  random_seed: 19
  n_mc: 2
  variables:
    n:
      model: ornstein_uhlenbeck
      sigma: 0.2
      tau_s: 1.0
    v:
      model: ornstein_uhlenbeck
      magnitude_sigma: 0.01
      magnitude_tau_s: 1.0
      direction_sigma_deg: 0.5
    B:
      model: ornstein_uhlenbeck
      magnitude_sigma: 0.02
      magnitude_tau_s: 0.5
      direction_sigma_deg: 1.0
      direction_tau_s: 0.25
output:
  path: {output}
visualization:
  enabled: true
  mc_index: 1
""",
        encoding="utf-8",
    )

    generate_from_config(config)

    with h5py.File(output) as handle:
        assert handle["observed/n"].shape == (2, 3, 9)
        assert handle["observed/v"].shape == (2, 3, 9, 3)
        assert handle["observed/B"].shape == (2, 5, 9, 3)
        assert handle["errors/n"].shape == (2, 3, 9)
        assert handle["errors/v"].shape == (2, 3, 9, 3)
        assert handle["errors/B"].shape == (2, 5, 9, 3)
        assert "ornstein_uhlenbeck" in handle["metadata"].attrs["error_models"]
        assert handle["metadata"].attrs["observed_visualization_mc_index"] == 1
    assert (tmp_path / "ou-observations_observed_mc001.png").is_file()
    assert (tmp_path / "ou-observations_geometry_observed_mc001.png").is_file()
