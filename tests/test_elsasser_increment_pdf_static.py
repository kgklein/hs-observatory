from pathlib import Path

import h5py
import numpy as np
import pytest

from hs_observatory.analyze import analyze_from_config
from hs_observatory.analysis.derived.pairs import (
    local_pairwise_directions,
    pair_chunks,
    perpendicular_projection,
    possible_pair_count,
    separation_magnitude,
)
from hs_observatory.analysis.elsasser_increment_pdf_static import (
    _read_positions,
    _settings,
    compute_increment_pdf_for_points,
    geometric_mean_positive,
    pdf_density,
)
from hs_observatory.analysis.registry import list_analyses
from hs_observatory.cli import analyze_main


def _pair_settings(**overrides):
    config = {
        "datasets": [{"name": "truth", "data_group": "truth"}],
        "increments": {"component": "perpendicular"},
        "scale_bins": {
            "separation": "perpendicular",
            "binning": "linear",
            "n_bins": 2,
            "lambda_min": 0.5,
            "lambda_max": 4.0,
            "min_samples_per_bin": 1,
        },
        "pdf_bins": {"n_bins": 10, "x_min": 0.0, "x_max": 10.0},
        "pair_sampling": {"mode": "all", "chunk_size": 2},
        "plot": {"enabled": False},
    }
    config.update(overrides)
    return _settings(config)


def test_pair_enumeration_is_unique_unordered_and_includes_all_point_categories():
    pairs = []
    for left, right in pair_chunks(4, 2):
        pairs.extend(zip(left.tolist(), right.tolist()))

    assert possible_pair_count(4) == 6
    assert len(pairs) == 6
    assert len(set(pairs)) == 6
    assert all(left < right for left, right in pairs)
    assert all(left != right for left, right in pairs)

    # point = time * n_spacecraft + spacecraft, with n_spacecraft = 2.
    decoded = [((a // 2, a % 2), (b // 2, b % 2)) for a, b in pairs]
    assert any(a[0] == b[0] and a[1] != b[1] for a, b in decoded)
    assert any(a[0] != b[0] and a[1] == b[1] for a, b in decoded)
    assert any(a[0] != b[0] and a[1] != b[1] for a, b in decoded)


def test_random_pair_enumeration_is_reproducible_and_bounded():
    first = list(pair_chunks(20, 3, mode="random", n_pairs=7, random_seed=4))
    second = list(pair_chunks(20, 3, mode="random", n_pairs=7, random_seed=4))
    first_pairs = np.column_stack(
        (np.concatenate([item[0] for item in first]), np.concatenate([item[1] for item in first]))
    )
    second_pairs = np.column_stack(
        (
            np.concatenate([item[0] for item in second]),
            np.concatenate([item[1] for item in second]),
        )
    )
    np.testing.assert_array_equal(first_pairs, second_pairs)
    assert len(np.unique(first_pairs, axis=0)) == 7


def test_perpendicular_projection_and_separation_have_known_geometry():
    direction = np.array([[0.0, 0.0, 1.0]])
    vector = np.array([[3.0, 4.0, 12.0]])

    np.testing.assert_allclose(perpendicular_projection(vector, direction), [[3.0, 4.0, 0.0]])
    assert separation_magnitude(vector, direction, "perpendicular")[0] == pytest.approx(5.0)
    assert separation_magnitude(vector, direction, "full")[0] == pytest.approx(13.0)


def test_local_pairwise_direction_uses_global_fallback():
    directions, valid = local_pairwise_directions(
        np.array([[0.0, 0.0, 2.0]]),
        np.array([[0.0, 0.0, -2.0]]),
        np.array([0.0, 5.0, 0.0]),
    )

    assert valid[0]
    np.testing.assert_allclose(directions[0], [0.0, 1.0, 0.0])


def test_geometric_mean_excludes_zero_and_pdf_is_a_density():
    values = np.array([0.0, 1.0, 4.0, 16.0])
    assert geometric_mean_positive(values) == pytest.approx(4.0)

    edges = np.linspace(0.0, 5.0, 6)
    counts, density = pdf_density(np.array([0.2, 0.8, 1.2, 3.4]), edges)
    assert np.sum(counts) == 4
    assert np.sum(density * np.diff(edges)) == pytest.approx(1.0)


def test_chunked_statistics_match_naive_pair_calculation():
    positions = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 2.0, 0.0], [3.0, 2.0, 0.0]],
        ]
    )
    z_plus = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 2.0, 0.0]],
            [[2.0, 1.0, 0.0], [4.0, 3.0, 0.0]],
        ]
    )
    z_minus = 0.5 * z_plus + np.array([0.0, 1.0, 0.0])
    B = np.zeros_like(positions)
    B[..., 2] = 5.0
    settings = _pair_settings()

    result = compute_increment_pdf_for_points(positions, z_plus, z_minus, B, settings)

    flat_r = positions.reshape(-1, 3)
    flat_plus = z_plus.reshape(-1, 3)
    left, right = np.triu_indices(4, k=1)
    lam = np.linalg.norm(flat_r[right] - flat_r[left], axis=1)
    increment = np.linalg.norm(flat_plus[right] - flat_plus[left], axis=1)
    edges = result["lambda"]["bin_edges"]
    indices = np.searchsorted(edges, lam, side="right") - 1
    indices[lam == edges[-1]] = len(edges) - 2
    expected_counts = np.array([np.sum(indices == index) for index in range(2)])
    expected_typical = np.array(
        [geometric_mean_positive(increment[indices == index]) for index in range(2)]
    )

    np.testing.assert_array_equal(result["lambda"]["counts"], expected_counts)
    np.testing.assert_allclose(result["pdf"]["z_plus"]["typical_increment"], expected_typical)
    assert result["n_pairs_processed"] == 6


def test_alignment_angle_is_one_for_orthogonal_perpendicular_increments():
    positions = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    z_plus = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    z_minus = np.array([[[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
    B = np.array([[[0.0, 0.0, 5.0], [0.0, 0.0, 5.0]]])
    settings = _pair_settings(
        scale_bins={
            "separation": "perpendicular",
            "binning": "linear",
            "n_bins": 1,
            "lambda_min": "auto",
            "lambda_max": "auto",
            "min_samples_per_bin": 1,
        }
    )

    result = compute_increment_pdf_for_points(positions, z_plus, z_minus, B, settings)

    assert result["alignment"]["counts"][0] == 1
    assert result["alignment"]["mean_sin_theta"][0] == pytest.approx(1.0)


def test_static_pdf_observed_positions_use_selected_mc_position_offset(tmp_path):
    path = tmp_path / "positions.h5"
    time = np.array([0.0, 1.0])
    positions = np.zeros((2, 2, 3))
    positions[1, :, 0] = 1.0
    offsets = np.zeros((2, 2, 3))
    offsets[1, 1, 0] = 4.0

    with h5py.File(path, "w") as handle:
        handle.create_group("metadata")
        times = handle.create_group("time")
        spacecraft = handle.create_group("spacecraft")
        spacecraft.create_group("r").create_dataset("v", data=positions)
        spacecraft["r"].create_dataset("n", data=positions)
        spacecraft["r"].create_dataset("B", data=positions)
        errors = handle.create_group("errors")
        errors.create_dataset("position_offset", data=offsets)
        for name in ("n", "v", "B"):
            times.create_dataset(name, data=time)

    truth_positions, truth_metadata = _read_positions(
        path,
        time,
        data_group="truth",
        mc_index=0,
    )
    observed_positions, observed_metadata = _read_positions(
        path,
        time,
        data_group="observed",
        mc_index=1,
    )

    np.testing.assert_allclose(truth_positions[:, 1, 0], [1.0, 1.0])
    np.testing.assert_allclose(observed_positions[:, 1, 0], [5.0, 5.0])
    assert truth_metadata["coordinate_policy"] == "nominal_spacecraft_positions"
    assert observed_metadata["coordinate_policy"] == "nominal_plus_observed_position_offset"


def _write_observations(path: Path) -> Path:
    n_time, n_spacecraft = 4, 2
    time = np.arange(n_time, dtype=float)
    density = 5.0 + 0.2 * time[:, None] + np.array([[0.0, 1.0]])
    velocity = np.empty((n_time, n_spacecraft, 3))
    velocity[..., 0] = time[:, None] ** 2 + np.array([[0.0, 1.0]])
    velocity[..., 1] = time[:, None] * np.array([[1.0, 2.0]])
    velocity[..., 2] = 0.5 * time[:, None]
    magnetic = np.empty_like(velocity)
    magnetic[..., 0] = 0.2 * time[:, None]
    magnetic[..., 1] = np.array([[0.1, -0.1]])
    magnetic[..., 2] = 5.0
    positions = np.empty((n_spacecraft, n_time, 3))
    for spacecraft in range(n_spacecraft):
        positions[spacecraft, :, 0] = 3.0 * time + spacecraft
        positions[spacecraft, :, 1] = 2.0 * spacecraft
        positions[spacecraft, :, 2] = 0.0

    with h5py.File(path, "w") as handle:
        times = handle.create_group("time")
        position_group = handle.create_group("spacecraft").create_group("r")
        handle["spacecraft"].create_dataset(
            "id", data=np.asarray(["H", "N1"], dtype=h5py.string_dtype("utf-8"))
        )
        truth = handle.create_group("truth")
        observed = handle.create_group("observed")
        for name, values in (("n", density), ("v", velocity), ("B", magnetic)):
            times.create_dataset(name, data=time)
            position_group.create_dataset(name, data=positions)
            truth.create_dataset(name, data=values)
            first = values.copy()
            second = values.copy()
            if name == "v":
                second = values + 0.3 * values**2
            observed.create_dataset(name, data=np.stack((first, second)))
        metadata = handle.create_group("metadata")
        metadata.attrs["velocity_observed_frame"] = "plasma"
    return path


def test_truth_observed_output_plot_and_cli_smoke(tmp_path, capsys):
    input_path = _write_observations(tmp_path / "input.h5")
    output_path = tmp_path / "elsasser_increment_pdf_static.h5"
    plot_directory = tmp_path / "plots"
    config_path = tmp_path / "analysis.yaml"
    config_path.write_text(
        f"""
analysis:
  method: elsasser_increment_pdf_static
input:
  path: {input_path}
output:
  path: {output_path}
elsasser_increment_pdf_static:
  datasets:
    - {{name: truth, data_group: truth}}
    - {{name: observed_mc0, data_group: observed, mc_index: 1}}
  elsasser:
    velocity: {{frame: auto, spacecraft_correction: auto}}
    density_normalization: {{method: global_per_spacecraft, species: proton}}
    mean_subtraction: {{method: global_per_spacecraft}}
    timebase: {{method: slower_of_n_and_v, interpolation: linear}}
  increments: {{field: both, plotted_field: z_plus, component: perpendicular, statistic: magnitude}}
  mean_field: {{method: local_pairwise, fallback: global_mean_B, min_norm: 1.0e-12}}
  scale_bins:
    separation: perpendicular
    binning: log
    n_bins: 3
    lambda_min: auto
    lambda_max: auto
    min_samples_per_bin: 1
  pdf_bins:
    normalized_by: geometric_mean
    binning: linear
    n_bins: 12
    x_min: 0.0
    x_max: auto_percentile
    x_max_percentile: 99.5
  pair_sampling: {{mode: all, chunk_size: 3, max_pairs: null, random_seed: 7}}
  alignment: {{compute: true}}
  output_samples: {{save: true, max_samples: 5}}
  plot: {{enabled: true, directory: {plot_directory}, yscale: log, xscale: linear}}
""",
        encoding="utf-8",
    )

    assert "elsasser_increment_pdf_static" in list_analyses()
    assert analyze_from_config(config_path) == output_path
    with h5py.File(output_path) as handle:
        for dataset in ("truth", "observed_mc0"):
            assert f"results/{dataset}/pdf/z_plus/density" in handle
            assert f"results/{dataset}/pdf/z_minus/density" in handle
            assert f"results/{dataset}/alignment/mean_sin_theta" in handle
            assert handle[f"results/{dataset}/samples/lambda"].shape[0] <= 5
        assert handle["results/truth"].attrs["mc_index"] == 0
        assert handle["results/observed_mc0"].attrs["mc_index"] == 1
        assert handle["results/truth"].attrs["coordinate_policy"] == "nominal_spacecraft_positions"
        assert (
            handle["results/observed_mc0"].attrs["coordinate_policy"]
            == "nominal_spacecraft_positions"
        )
        assert handle["metadata"].attrs["sampling_assumption"] == "static_all_point_pairs"
        truth_typical = handle["results/truth/pdf/z_plus/typical_increment"][...]
        observed_typical = handle["results/observed_mc0/pdf/z_plus/typical_increment"][...]
        assert not np.allclose(truth_typical, observed_typical, equal_nan=True)

    truth_plot = plot_directory / f"{output_path.stem}_truth_zplus.png"
    observed_plot = plot_directory / f"{output_path.stem}_observed_mc0_zplus.png"
    assert truth_plot.is_file() and truth_plot.stat().st_size > 0
    assert observed_plot.is_file() and observed_plot.stat().st_size > 0
    assert analyze_main([str(config_path)]) == 0
    assert str(output_path) in capsys.readouterr().out
