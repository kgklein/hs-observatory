import h5py
import numpy as np

from hs_observatory.generate import generate_from_config


def test_generator_samples_field_at_hidden_coordinate_offsets(tmp_path):
    output = tmp_path / "coordinate-errors.h5"
    config = tmp_path / "coordinate-errors.yaml"
    config.write_text(
        f"""
spacecraft:
  source: synthetic
time:
  t_start_s: 0
  t_stop_s: 1
  cadence_s: 0.5
field:
  model: sine_sum
  background:
    n: 0
    v: [0, 0, 0]
    B: [0, 0, 0]
  modes:
    - quantity: n
      amplitude: 1
      k: [0.1, 0, 0]
      omega: 0.5
      phase: 0
errors:
  random_seed: 81
  n_mc: 2
  time:
    model: gaussian_constant
    sigma_s: 0.2
  position:
    model: gaussian_constant
    sigma_km: [0.4, 0, 0]
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

    generate_from_config(config)

    with h5py.File(output) as handle:
        nominal_time = handle["time/n"][...]
        nominal_x = handle["spacecraft/r/n"][..., 0]
        time_offset = handle["errors/time_offset"][...]
        x_offset = handle["errors/position_offset"][..., 0]
        observed = handle["observed/n"][...]
        truth = handle["truth/n"][...]

    expected = np.empty_like(observed)
    for mc_index in range(2):
        phase = (
            0.1 * (nominal_x.T + x_offset[mc_index][None, :])
            - 0.5 * (nominal_time[:, None] + time_offset[mc_index][None, :])
        )
        expected[mc_index] = np.sin(phase)

    assert np.allclose(observed, expected)
    assert not np.allclose(observed[0], truth)
