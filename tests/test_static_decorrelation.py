from pathlib import Path

import h5py
import numpy as np

from hs_observatory.analyze import analyze_from_config
from hs_observatory.analysis.static_decorrelation import (
    StaticDecorrelationAnalysis,
    compute_static_decorrelation,
    plot_static_decorrelation,
)
from hs_observatory.cli import analyze_main
from hs_observatory.io.hdf5 import write_observations


def _write_test_observations(
    path: Path,
    x_positions: np.ndarray,
    n_values: np.ndarray,
    *,
    n_mc: int = 2,
) -> Path:
    n_time, n_spacecraft = n_values.shape
    time_axis = np.arange(n_time, dtype=float)
    position = np.zeros((n_spacecraft, n_time, 3))
    position[:, :, 0] = x_positions[:, None]
    v = np.stack((n_values, 0.5 * n_values, -0.25 * n_values), axis=-1)
    B = np.stack((-0.2 * n_values, n_values, 0.3 * n_values), axis=-1)
    truth = {"n": n_values, "v": v, "B": B}
    observed = {
        variable: np.broadcast_to(values, (n_mc, *values.shape)).copy()
        for variable, values in truth.items()
    }
    return write_observations(
        path,
        time={variable: time_axis for variable in ("n", "v", "B")},
        spacecraft_ids=tuple(f"S{index}" for index in range(n_spacecraft)),
        position={variable: position for variable in ("n", "v", "B")},
        truth=truth,
        observed=observed,
        errors=None,
        metadata={"schema_version": "2"},
    )


def _base_config(**overrides):
    config = {
        "variables": ["n"],
        "data_group": "truth",
        "mean": {"method": "global"},
        "separation": {
            "units": "km",
            "binning": "linear",
            "n_bins": 2,
            "r_min": 0.0,
            "r_max": 11.0,
        },
        "correlation": {
            "vector_mode": "dot",
            "component_correlations": True,
            "normalize": True,
        },
        "structure_function": {"compute": True, "order": 2},
    }
    config.update(overrides)
    return config


def test_nearby_scalar_spacecraft_are_more_correlated_than_distant_spacecraft(tmp_path):
    x = np.array([0.0, 1.0, 10.0])
    phase = np.linspace(0.0, 20.0 * np.pi, 1000, endpoint=False)
    density = np.sin(phase[:, None] + 0.2 * x[None, :])
    input_path = _write_test_observations(tmp_path / "known.h5", x, density)

    result = compute_static_decorrelation(input_path, _base_config())

    correlation = result["correlation"]["n"]["mean"]
    structure = result["structure_function"]["n"]["mean"]
    assert np.all(np.isfinite(correlation))
    assert correlation[0] > correlation[1]
    assert structure[0] < structure[1]
    assert np.array_equal(result["counts"]["n"], [1000, 2000])


def test_pair_count_uses_unique_pairs_at_every_time(tmp_path):
    x = np.array([0.0, 1.0, 3.0, 7.0])
    density = np.arange(20, dtype=float).reshape(5, 4)
    input_path = _write_test_observations(tmp_path / "pairs.h5", x, density)
    config = _base_config(
        separation={
            "units": "km",
            "binning": "linear",
            "n_bins": 1,
            "r_min": 0.0,
            "r_max": 8.0,
        }
    )

    result = compute_static_decorrelation(input_path, config)

    assert result["counts"]["n"].sum() == 5 * 6  # time samples * C(4, 2)


def test_empty_bins_have_zero_count_and_nan_statistics(tmp_path):
    x = np.array([0.0, 1.0])
    density = np.tile([1.0, -1.0], (3, 1))
    input_path = _write_test_observations(tmp_path / "empty.h5", x, density)
    config = _base_config(
        mean={"method": "per_time"},
        separation={
            "units": "km",
            "binning": "linear",
            "n_bins": 4,
            "r_min": 0.0,
            "r_max": 4.0,
        },
    )

    result = compute_static_decorrelation(input_path, config)

    empty = result["counts"]["n"] == 0
    assert np.any(empty)
    assert np.all(np.isnan(result["correlation"]["n"]["mean"][empty]))
    assert np.all(np.isnan(result["structure_function"]["n"]["std"][empty]))


def test_hdf5_output_groups_and_component_shapes(tmp_path):
    x = np.array([0.0, 1.0, 4.0])
    phase = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    density = np.sin(phase[:, None] + 0.3 * x[None, :])
    input_path = _write_test_observations(tmp_path / "all-variables.h5", x, density)
    output_path = tmp_path / "decorrelation.h5"
    config = _base_config(
        variables=["n", "v", "B"],
        data_group="observed",
        mc_index=1,
        separation={
            "units": "km",
            "binning": "log",
            "n_bins": 5,
            "r_min": None,
            "r_max": None,
        },
    )
    method = StaticDecorrelationAnalysis(config)
    result = method.run(input_path)
    figure, axes = plot_static_decorrelation(result)
    assert axes.shape == (3, 2)
    method.write_result(result, output_path)

    with h5py.File(output_path) as handle:
        assert handle["bin_edges"].shape == (6,)
        assert handle["bin_lower"].shape == (5,)
        assert handle["counts/n"].shape == (5,)
        assert handle["correlation/n/mean"].shape == (5,)
        assert handle["correlation/v_dot/std"].shape == (5,)
        assert handle["correlation/B_components/mean"].shape == (5, 3)
        assert handle["structure_function/v/stderr"].shape == (5,)
        assert handle["metadata"].attrs["method"] == "static_decorrelation"
        assert handle["metadata"].attrs["diagnostic_plot_enabled"]
    assert (tmp_path / "decorrelation_diagnostic.png").is_file()
    figure.clf()


def test_static_decorrelation_analysis_config_and_cli_smoke(tmp_path, capsys):
    x = np.array([0.0, 2.0, 5.0])
    density = np.arange(18, dtype=float).reshape(6, 3)
    input_path = _write_test_observations(tmp_path / "input.h5", x, density)
    output_path = tmp_path / "output.h5"
    config_path = tmp_path / "analysis.yaml"
    config_path.write_text(
        f"""
input:
  path: {input_path}
analysis:
  method: static_decorrelation
output:
  path: {output_path}
decorrelation:
  variables: [n, v, B]
  data_group: observed
  mc_index: 0
  mean: {{method: per_time}}
  separation:
    units: km
    binning: linear
    n_bins: 4
    r_min: 1.0
    r_max: 6.0
  correlation:
    vector_mode: dot
    component_correlations: true
    normalize: true
  structure_function: {{compute: true, order: 2}}
""",
        encoding="utf-8",
    )

    assert analyze_from_config(config_path) == output_path
    assert output_path.is_file()
    assert analyze_main([str(config_path)]) == 0
    assert str(output_path) in capsys.readouterr().out
