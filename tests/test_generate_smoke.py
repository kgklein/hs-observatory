import json

import h5py
import numpy as np

from hs_observatory.analyze import analyze_from_config
from hs_observatory.generate import _evaluate_quantity_with_progress, generate_from_config


class _ProgressField:
    name = "malara_turbulence"

    def evaluate_quantity(self, quantity, time, position):
        del quantity
        values = np.asarray(time, dtype=float)
        if values.ndim == 2:
            values = values[:, 0]
        return np.broadcast_to(values[:, None], (len(values), position.shape[0])).copy()


def test_malara_progress_prints_every_ten_measurement_seconds(capsys):
    time = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
    position = np.zeros((2, len(time), 3))

    result = _evaluate_quantity_with_progress(
        _ProgressField(),
        "n",
        time,
        position,
        label="truth n",
    )

    assert result.shape == (5, 2)
    assert np.array_equal(result[:, 0], time)
    output = capsys.readouterr().out
    assert "truth n: generated through t=10 measurement s" in output
    assert "truth n: generated through t=20 measurement s" in output


def test_generation_and_analysis_with_synthetic_fallback(tmp_path):
    h5_path = tmp_path / "generated.h5"
    generation_config = tmp_path / "generate.yaml"
    generation_config.write_text(
        f"""
spacecraft:
  source: cdf
  cdf_path: {tmp_path / 'missing-cdf'}
  fallback_synthetic: true
time:
  t_start_s: 0
  t_stop_s: 1
  cadence_s:
    n: 0.5
    v: 1.0
    B: 0.25
field:
  model: sine_sum
  background:
    n: 5
    v: [400, 0, 0]
    B: [5, 0, 0]
  modes: []
errors:
  random_seed: 42
  n_mc: 2
  variables:
    n: {{model: gaussian_relative, sigma: 0.01}}
    v: {{model: none}}
    B: {{model: gaussian_absolute, sigma: 0.1}}
output:
  path: {h5_path}
""",
        encoding="utf-8",
    )
    assert generate_from_config(generation_config) == h5_path
    assert (tmp_path / "generated_truth.png").is_file()
    assert (tmp_path / "generated_geometry.png").is_file()
    overview_dir = tmp_path / "geometry_overviews"
    assert any(overview_dir.glob("generated_geometry_summary_*.pdf"))
    assert any(overview_dir.glob("generated_geometry_summary_*.png"))
    assert any(overview_dir.glob("generated_3d_configuration_*.pdf"))
    assert any(overview_dir.glob("generated_3d_configuration_*.png"))
    assert (tmp_path / "generated_velocity_alfven.png").is_file()
    assert (tmp_path / "generated_observed_mc000.png").is_file()
    assert (tmp_path / "generated_geometry_observed_mc000.png").is_file()
    assert (tmp_path / "generated_velocity_alfven_observed_mc000.png").is_file()

    with h5py.File(h5_path) as handle:
        assert handle["time/n"].shape == (3,)
        assert handle["time/v"].shape == (2,)
        assert handle["time/B"].shape == (5,)
        assert handle["spacecraft/r/n"].shape == (9, 3, 3)
        assert handle["spacecraft/r/v"].shape == (9, 2, 3)
        assert handle["spacecraft/r/B"].shape == (9, 5, 3)
        assert handle["truth/n"].shape == (3, 9)
        assert handle["truth/v"].shape == (2, 9, 3)
        assert handle["truth/B"].shape == (5, 9, 3)
        assert handle["observed/n"].shape == (2, 3, 9)
        assert handle["errors/time_offset"].shape == (2, 9)
        assert handle["errors/position_offset"].shape == (2, 9, 3)
        assert handle["errors/time_offset"].attrs["units"] == "s"
        assert handle["errors/position_offset"].attrs["units"] == "km"
        assert not handle["errors/time_offset"][...].any()
        assert not handle["errors/position_offset"][...].any()
        assert handle["metadata"].attrs["schema_version"] == "2"
        assert "synthetic_fallback" in handle["metadata"].attrs["spacecraft_source"]

    json_path = tmp_path / "diagnostics.json"
    analysis_config = tmp_path / "analyze.yaml"
    analysis_config.write_text(
        f"""
input:
  path: {h5_path}
analysis:
  method: diagnostics
output:
  path: {json_path}
""",
        encoding="utf-8",
    )
    assert analyze_from_config(analysis_config) == json_path
    result = json.loads(json_path.read_text(encoding="utf-8"))
    assert result["spacecraft_separation"]["n"]["n_pairs"] == 36
    assert result["spacecraft_separation"]["B"]["n_pairs"] == 36
