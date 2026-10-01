from pathlib import Path

import h5py
import numpy as np
import pytest

from hs_observatory.analysis.derived.pairs import adaptive_stratified_pairs
from hs_observatory.analysis.registry import list_analyses
from hs_observatory.analysis.structure_function import (
    StructureFunctionAnalysis,
    _ensemble_interval,
    _fit_scaling_exponents,
    compute_all_spacetime_structure_functions,
    compute_pairwise_structure_functions,
    plot_structure_function_diagnostic,
    plot_structure_function_overlay,
    plot_structure_function_overlay_from_config,
    plot_structure_function_summary,
    plot_structure_function_summary_row,
    structure_function_kurtosis_ylim,
    structure_function_summary_ylims,
)


def test_vector_structure_function_preserves_members_and_empty_bins():
    positions = np.array(
        [
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            ],
        ]
    )
    z_plus = np.array(
        [
            [
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            ],
        ]
    )
    z_minus = 2.0 * z_plus

    result = compute_pairwise_structure_functions(
        positions,
        z_plus,
        z_minus,
        [2, 4],
        np.array([0.5, 1.5, 2.5]),
        min_fit_bins=2,
    )

    np.testing.assert_array_equal(result["counts"], [[2, 0], [2, 0]])
    np.testing.assert_allclose(result["z_plus"]["S_n"][:, 0, 0], [10.0, 1.0])
    np.testing.assert_allclose(result["z_minus"]["S_n"][:, 0, 0], [40.0, 4.0])
    assert np.isnan(result["z_plus"]["S_n"][0, 0, 1])
    np.testing.assert_allclose(result["z_plus"]["flatness"][:, 0], [136.0 / 100.0, 1.0])


def test_flatness_is_none_without_second_and_fourth_orders():
    positions = np.array([[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]])
    z_plus = np.array([[[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]])

    result = compute_pairwise_structure_functions(
        positions,
        z_plus,
        z_plus,
        [1, 3],
        np.array([0.5, 1.5]),
        min_fit_bins=2,
    )

    assert result["z_plus"]["flatness"] is None


def test_all_spacetime_structure_function_matches_direct_pairs_and_sampling():
    positions = np.array(
        [
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            ],
        ]
    )
    z_plus = np.zeros_like(positions)
    z_plus[0, ..., 0] = positions[0, ..., 0] * 2.0
    z_plus[1, ..., 0] = positions[1, ..., 0] * 3.0
    z_minus = 0.5 * z_plus

    result = compute_all_spacetime_structure_functions(
        positions,
        z_plus,
        z_minus,
        [2],
        np.array([0.5, 1.5, 3.0, 5.0]),
        pair_sampling={
            "mode": "all",
            "chunk_size": 2,
            "n_pairs": None,
            "max_pairs": None,
            "random_seed": 7,
        },
        min_fit_bins=2,
    )

    np.testing.assert_array_equal(result["counts"], [[2, 2, 2], [2, 2, 2]])
    np.testing.assert_allclose(result["z_plus"]["S_n"][0, 0], [4.0, 16.0, 50.0])
    np.testing.assert_allclose(result["z_plus"]["S_n"][1, 0], [9.0, 36.0, 112.5])
    np.testing.assert_allclose(result["z_minus"]["S_n"][0, 0], [1.0, 4.0, 12.5])
    assert result["spacetime_pair_counts"]["possible_per_member"] == 6

    sampled = compute_all_spacetime_structure_functions(
        positions[:1],
        z_plus[:1],
        z_minus[:1],
        [2],
        np.array([0.5, 1.5, 3.0, 5.0]),
        pair_sampling={
            "mode": "all",
            "chunk_size": 2,
            "n_pairs": None,
            "max_pairs": 3,
            "random_seed": 7,
        },
        min_fit_bins=2,
    )

    np.testing.assert_array_equal(sampled["counts"], [[1, 1, 1]])
    assert sampled["spacetime_pair_counts"]["selected_per_member"][0] == 3

    limited = compute_all_spacetime_structure_functions(
        positions[:1],
        z_plus[:1],
        z_minus[:1],
        [2],
        np.array([0.5, 1.5, 3.0, 5.0]),
        pair_sampling={
            "mode": "all",
            "chunk_size": 2,
            "n_pairs": None,
            "max_pairs": None,
            "random_seed": 7,
        },
        min_fit_bins=2,
        max_pair_lambda_km=2.5,
    )

    np.testing.assert_array_equal(limited["counts"], [[2, 2, 0]])
    np.testing.assert_allclose(limited["z_plus"]["S_n"][0, 0, :2], [4.0, 16.0])
    assert np.isnan(limited["z_plus"]["S_n"][0, 0, 2])
    assert limited["spacetime_pair_counts"]["skipped_above_max_lambda"][0] == 2

    lower_limited = compute_all_spacetime_structure_functions(
        positions[:1],
        z_plus[:1],
        z_minus[:1],
        [2],
        np.array([0.5, 1.5, 3.0, 5.0]),
        pair_sampling={
            "mode": "all",
            "chunk_size": 2,
            "n_pairs": None,
            "max_pairs": None,
            "random_seed": 7,
        },
        min_fit_bins=2,
        min_pair_lambda_km=1.5,
    )

    np.testing.assert_array_equal(lower_limited["counts"], [[0, 2, 2]])
    assert lower_limited["spacetime_pair_counts"]["skipped_below_min_lambda"][0] == 2


def test_same_time_structure_function_respects_minimum_pair_lambda():
    positions = np.array([[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]])
    result = compute_pairwise_structure_functions(
        positions,
        positions,
        positions,
        [1],
        np.array([0.5, 1.5, 2.5, 3.5]),
        min_fit_bins=2,
        min_pair_lambda_km=1.5,
    )

    np.testing.assert_array_equal(result["counts"], [[0, 1, 1]])


def test_adaptive_stratified_pairs_fill_each_available_bin_reproducibly():
    positions = np.column_stack((np.arange(6.0), np.zeros((6, 2))))
    edges = np.array([0.5, 1.5, 2.5, 6.0])
    kwargs = {
        "target_pairs_per_bin": 2,
        "max_candidates": 1000,
        "candidate_chunk_size": 7,
        "random_seed": 42,
    }

    left, right, diagnostics = adaptive_stratified_pairs(positions, edges, **kwargs)
    repeated_left, repeated_right, repeated_diagnostics = adaptive_stratified_pairs(
        positions, edges, **kwargs
    )

    separations = np.linalg.norm(positions[right] - positions[left], axis=1)
    counts, _ = np.histogram(separations, bins=edges)
    np.testing.assert_array_equal(counts, [2, 2, 2])
    np.testing.assert_array_equal(left, repeated_left)
    np.testing.assert_array_equal(right, repeated_right)
    assert diagnostics["target_reached"] is True
    assert diagnostics["candidates_examined"] < kwargs["max_candidates"]
    np.testing.assert_array_equal(
        diagnostics["retained_per_bin"], repeated_diagnostics["retained_per_bin"]
    )


def test_adaptive_stratified_structure_function_reports_shortfall():
    positions = np.array([[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]])
    result = compute_all_spacetime_structure_functions(
        positions,
        positions,
        positions,
        [1],
        np.array([0.5, 1.5, 2.5, 3.5]),
        pair_sampling={
            "mode": "adaptive_stratified",
            "chunk_size": 4,
            "n_pairs": None,
            "max_pairs": None,
            "random_seed": 9,
            "target_pairs_per_bin": 2,
            "max_candidates": 40,
            "binning_n_pairs": 10,
        },
        min_fit_bins=2,
    )

    # Only one unique length-2 pair exists, and no pair has length 3.
    np.testing.assert_array_equal(result["counts"], [[2, 1, 0]])
    assert not result["spacetime_pair_counts"]["sampling_target_reached"][0]
    assert result["spacetime_pair_counts"]["candidates_examined"][0] == 40


def test_scaling_exponents_match_synthetic_power_laws():
    centers = np.array([1.0, 2.0, 4.0, 8.0])
    orders = [1, 2, 3]
    S_n = np.array([[centers ** (0.5 * order) for order in orders]])
    counts = np.ones((1, centers.size), dtype=int)

    result = _fit_scaling_exponents(
        S_n,
        counts,
        centers,
        orders,
        {"total": [1.0, 8.0]},
        min_fit_bins=3,
    )

    np.testing.assert_allclose(result["raw"][0, 0], [0.5, 1.0, 1.5])
    np.testing.assert_allclose(result["normalized_by_order2"][0, 0], [0.5, 1.0, 1.5])


def _write_structure_input(path: Path) -> Path:
    n_time, n_spacecraft, n_mc = 4, 3, 2
    time = np.arange(n_time, dtype=float)
    positions = np.zeros((n_spacecraft, n_time, 3))
    positions[1, :, 0] = 1.0
    positions[2, :, 0] = 3.0
    density = np.full((n_time, n_spacecraft), 5.0)
    velocity = np.zeros((n_time, n_spacecraft, 3))
    velocity[..., 0] = time[:, None] * np.array([[1.0, 2.0, 4.0]])
    magnetic = np.zeros_like(velocity)
    magnetic[..., 2] = 5.0 + 0.1 * time[:, None]
    observed = {
        "n": np.broadcast_to(density, (n_mc, *density.shape)).copy(),
        "v": np.stack((velocity, 1.5 * velocity)),
        "B": np.broadcast_to(magnetic, (n_mc, *magnetic.shape)).copy(),
    }
    position_offset = np.zeros((n_mc, n_spacecraft, 3))
    position_offset[0, 1, 0] = 10.0
    position_offset[1, 2, 1] = 20.0

    with h5py.File(path, "w") as handle:
        metadata = handle.create_group("metadata")
        metadata.attrs["velocity_observed_frame"] = "plasma"
        times = handle.create_group("time")
        spacecraft = handle.create_group("spacecraft")
        spacecraft.create_dataset(
            "id", data=np.asarray(["S0", "S1", "S2"], dtype=h5py.string_dtype("utf-8"))
        )
        position_group = spacecraft.create_group("r")
        truth = handle.create_group("truth")
        observed_group = handle.create_group("observed")
        errors = handle.create_group("errors")
        errors.create_dataset("position_offset", data=position_offset)
        for name, values in (("n", density), ("v", velocity), ("B", magnetic)):
            times.create_dataset(name, data=time)
            position_group.create_dataset(name, data=positions)
            truth.create_dataset(name, data=values)
            observed_group.create_dataset(name, data=observed[name])
    return path


def test_run_computes_observed_truth_and_uses_observed_position_offsets(tmp_path):
    input_path = _write_structure_input(tmp_path / "input.h5")
    analysis = StructureFunctionAnalysis(
        {
            "min_order": 2,
            "max_order": 4,
            "lambda_bins_km": [0.1, 2.0, 5.0, 50.0],
            "min_fit_bins": 2,
            "scale_ranges_km": {"total": [0.1, 50.0]},
            "elsasser": {"velocity": {"frame": "auto"}},
            "plot": {
                "enabled": True,
                "path": str(tmp_path / "sf.png"),
                "publication": {
                    "enabled": True,
                    "path": str(tmp_path / "sf_publication.png"),
                    "p_value": 0.9,
                    "quantity": "z_plus",
                    "max_order": 4,
                    "fit_range": "total",
                    "dpi": 80,
                },
            },
        }
    )

    result = analysis.run(input_path)
    output = analysis.write_result(result, tmp_path / "sf.json")

    assert "structure_function" in list_analyses()
    assert output.is_file()
    assert (tmp_path / "sf.png").is_file()
    assert (tmp_path / "sf_publication.png").is_file()
    assert set(result["products"]) == {"observed", "truth"}
    assert len(result["products"]["observed"]["counts"]) == 2
    truth_separations = np.asarray(result["products"]["truth"]["pair_separations_km"])
    observed_separations = np.asarray(result["products"]["observed"]["pair_separations_km"])
    assert truth_separations[0, 0, 0] == pytest.approx(1.0)
    assert observed_separations[0, 0, 0] == pytest.approx(11.0)
    assert observed_separations[1, 1, 0] == pytest.approx(np.hypot(3.0, 20.0))


def test_run_all_spacetime_uses_observed_position_offsets(tmp_path):
    input_path = _write_structure_input(tmp_path / "input.h5")
    analysis = StructureFunctionAnalysis(
        {
            "min_order": 2,
            "max_order": 2,
            "lambda_bins_km": [0.1, 2.0, 5.0, 50.0],
            "min_fit_bins": 2,
            "scale_ranges_km": {"total": [0.1, 50.0]},
            "increment_mode": "all_spacetime_pairs",
            "pair_sampling": {"mode": "all", "chunk_size": 5},
            "elsasser": {"velocity": {"frame": "auto"}},
        }
    )

    result = analysis.run(input_path)

    truth_counts = np.asarray(result["products"]["truth"]["counts"])
    observed_counts = np.asarray(result["products"]["observed"]["counts"])
    np.testing.assert_array_equal(truth_counts[0], [16, 32, 0])
    np.testing.assert_array_equal(observed_counts[0], [0, 16, 32])
    assert (
        result["products"]["observed"]["spacetime_pair_counts"]["skipped_nonpositive_separation"][0]
        == 18
    )
    assert result["metadata"]["increment_mode"] == "all_spacetime_pairs"
    assert (
        "observed positions use /errors/position_offset on nominal r" in result["metadata"]["notes"]
    )


def test_plot_helper_accepts_serialized_result(tmp_path):
    result = {
        "products": {
            "observed": {
                "orders": [2, 4],
                "lambda_bin_edges_km": [0.75, 1.5, 3.0, 6.0],
                "lambda_bin_centers_km": [1.0, 2.0, 4.0],
                "counts": [[2, 4, 8]],
                "z_plus": {
                    "S_n": [[[1.0, 4.0, 16.0], [3.0, 48.0, 768.0]]],
                    "flatness": [[3.0, 3.0, 3.0]],
                    "scaling_exponents": {
                        "raw": [[[2.0, 4.0]]],
                        "normalized_by_order2": [[[1.0, 2.0]]],
                        "range_names": ["total"],
                        "scale_ranges_km": {"total": [1.0, 4.0]},
                    },
                },
            }
        }
    }

    output = plot_structure_function_diagnostic(result, tmp_path / "plot.png")

    assert output.is_file() and output.stat().st_size > 0


def _summary_plot_result():
    orders = list(range(1, 8))
    centers = np.array([1.0, 2.0, 4.0])
    truth_sf = np.array([[centers ** (order / 3.0) for order in orders]])
    # Deliberately keep truth far outside the MC ensemble so a truth member
    # accidentally included in the percentiles changes the expected medians.
    truth_sf *= 100.0
    mc_scales = np.array([1.0, 2.0, 3.0, 20.0])
    mc_sf = np.stack(
        [[scale * centers ** (order / 3.0) for order in orders] for scale in mc_scales]
    )

    def product(values):
        second = values[:, orders.index(2)]
        fourth = values[:, orders.index(4)]
        flatness = fourth / second**2
        member_offsets = np.arange(values.shape[0], dtype=float)[:, None]
        xi = np.asarray(orders, dtype=float)[None, :] / 3.0 + member_offsets
        return {
            "orders": orders,
            "lambda_bin_edges_km": [0.75, 1.5, 3.0, 6.0],
            "lambda_bin_centers_km": centers.tolist(),
            "counts": np.ones((values.shape[0], 3)).tolist(),
            "z_plus": {
                "S_n": values.tolist(),
                "flatness": flatness.tolist(),
                "scaling_exponents": {
                    "raw": xi[:, None, :].tolist(),
                    "normalized_by_order2": xi[:, None, :].tolist(),
                    "range_names": ["total"],
                    "scale_ranges_km": {"total": [1.0, 4.0]},
                },
            },
        }

    return {"products": {"truth": product(truth_sf), "observed": product(mc_sf)}}


def test_ensemble_interval_uses_5th_and_95th_percentiles():
    lower, median, upper = _ensemble_interval(np.arange(101.0)[:, None])

    np.testing.assert_allclose(lower, [5.0])
    np.testing.assert_allclose(median, [50.0])
    np.testing.assert_allclose(upper, [95.0])


def test_summary_row_uses_sampled_truth_and_mc_only_percentiles():
    import matplotlib.pyplot as plt

    result = _summary_plot_result()
    figure, axes = plot_structure_function_summary_row(result, p_value=0.9)

    assert len(axes) == 3
    # Six orders, with alternating sampled-truth and MC-median curves.
    assert len(axes[0].lines) == 12
    np.testing.assert_allclose(
        axes[0].lines[0].get_ydata(), [100.0, 100.0 * 2 ** (1 / 3), 100.0 * 4 ** (1 / 3)]
    )
    np.testing.assert_allclose(
        axes[0].lines[1].get_ydata(), 2.5 * np.array([1.0, 2 ** (1 / 3), 4 ** (1 / 3)])
    )
    # The exponent panel uses all seven fitted orders, independent of max_order=6.
    np.testing.assert_array_equal(axes[1].lines[0].get_xdata(), np.arange(1, 8))
    np.testing.assert_allclose(
        axes[1].lines[-1].get_ydata(),
        1.0 - np.log2(0.9 ** (np.arange(1, 8) / 3.0) + 0.1 ** (np.arange(1, 8) / 3.0)),
    )
    gaussian = axes[2].lines[-1]
    assert gaussian.get_linestyle() == "--"
    np.testing.assert_allclose(gaussian.get_ydata(), [5.0 / 3.0, 5.0 / 3.0])
    plt.close(figure)


def test_summary_plot_accepts_json_and_supplied_axes(tmp_path):
    import json
    import matplotlib.pyplot as plt

    result_path = tmp_path / "sf.json"
    result_path.write_text(json.dumps(_summary_plot_result()), encoding="utf-8")
    output = plot_structure_function_summary(
        result_path, p_value=0.9, output_path=tmp_path / "summary.png", dpi=80
    )
    assert output.is_file() and output.stat().st_size > 0

    figure, axes = plt.subplots(1, 3)
    returned_figure, returned_axes = plot_structure_function_summary_row(
        result_path, p_value=0.7, axes=axes, run_color="#123456"
    )
    assert returned_figure is figure
    np.testing.assert_array_equal(returned_axes, axes)
    assert axes[0].lines[0].get_color() == "#123456"
    plt.close(figure)


def test_structure_function_overlay_writes_two_panel_figure(tmp_path):
    output = plot_structure_function_overlay(
        {0.9: _summary_plot_result(), 0.7: _summary_plot_result(), 0.5: _summary_plot_result()},
        tmp_path / "combined.png",
        dpi=80,
    )

    assert output.is_file() and output.stat().st_size > 0


def test_structure_function_overlay_from_config(tmp_path, monkeypatch):
    import json

    for case in (400, 401, 402):
        (tmp_path / f"{case}.json").write_text(json.dumps(_summary_plot_result()), encoding="utf-8")
    config = tmp_path / "combined.yaml"
    config.write_text(
        """cases:
  '400': {path: 400.json, p_value: 0.9}
  '401': {path: 401.json, p_value: 0.7}
  '402': {path: 402.json, p_value: 0.5}
output: {figure: combined.png}
dpi: 80
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    output = plot_structure_function_overlay_from_config(config)

    assert output == Path("combined.png")
    assert output.is_file() and output.stat().st_size > 0


def test_structure_function_overlay_config_accepts_fit_overrides(tmp_path, monkeypatch):
    import json

    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(_summary_plot_result()), encoding="utf-8")
    config = tmp_path / "combined.yaml"
    config.write_text(
        """cases:
  run: {path: result.json, p_value: 0.9}
output: {figure: ignored.png}
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    output = plot_structure_function_overlay_from_config(
        config,
        output_override="overridden.png",
        fit_lambda_range_override=(1.0, 2.0),
        min_fit_bins_override=2,
        show_p_model_override=False,
    )

    assert output == Path("overridden.png")
    assert output.is_file() and output.stat().st_size > 0


def test_summary_row_optionally_adds_pair_count_panel():
    import matplotlib.pyplot as plt

    figure, axes = plot_structure_function_summary_row(
        _summary_plot_result(), p_value=0.9, show_pair_counts=True
    )

    assert len(axes) == 4
    assert axes[3].get_title() == "pairs per bin"
    assert axes[3].get_ylabel() == "number of baseline pairs"
    # Sampled truth and MC median are both plotted against the lambda bins.
    assert len(axes[3].lines) == 2
    np.testing.assert_allclose(axes[3].lines[0].get_ydata(), [1.0, 1.0, 1.0])
    np.testing.assert_allclose(axes[3].lines[1].get_ydata(), [1.0, 1.0, 1.0])
    plt.close(figure)


def test_summary_row_can_omit_kurtosis_with_or_without_pair_counts():
    import matplotlib.pyplot as plt

    figure, axes = plot_structure_function_summary_row(
        _summary_plot_result(), p_value=0.9, show_kurtosis=False
    )
    assert len(axes) == 2
    assert [axis.get_title() for axis in axes] == [
        "structure functions",
        "scaling exponents",
    ]
    plt.close(figure)

    figure, axes = plot_structure_function_summary_row(
        _summary_plot_result(),
        p_value=0.9,
        show_kurtosis=False,
        show_pair_counts=True,
    )
    assert len(axes) == 3
    assert [axis.get_title() for axis in axes] == [
        "structure functions",
        "scaling exponents",
        "pairs per bin",
    ]
    plt.close(figure)


def test_summary_row_can_refit_saved_structure_functions_and_hide_p_model():
    import matplotlib.pyplot as plt

    figure, axes = plot_structure_function_summary_row(
        _summary_plot_result(),
        p_value=0.9,
        fit_lambda_range_km=(1.0, 2.0),
        min_fit_bins=2,
        show_p_model=False,
    )

    # Sampled truth and the MC errorbar remain, but the p-model curve is absent.
    assert not any("model" in line.get_label() for line in axes[1].lines)
    np.testing.assert_allclose(axes[1].lines[0].get_ydata(), np.arange(1, 8) / 3.0)
    assert len(axes[0].patches) == 1
    fit_box = axes[0].patches[0]
    display_vertices = fit_box.get_transform().transform(fit_box.get_path().vertices)
    fit_box_x = axes[0].transData.inverted().transform(display_vertices)[:, 0]
    np.testing.assert_allclose([fit_box_x.min(), fit_box_x.max()], [1.0, 2.0])
    plt.close(figure)


def test_shared_kurtosis_limits_can_be_applied_to_multiple_rows():
    import matplotlib.pyplot as plt

    result = _summary_plot_result()
    limits = structure_function_kurtosis_ylim([result, result])
    first, first_axes = plot_structure_function_summary_row(result, kurtosis_ylim=limits)
    second, second_axes = plot_structure_function_summary_row(result, kurtosis_ylim=limits)

    np.testing.assert_allclose(first_axes[2].get_ylim(), limits)
    np.testing.assert_allclose(second_axes[2].get_ylim(), limits)
    plt.close(first)
    plt.close(second)


def test_shared_summary_limits_cover_and_align_all_three_panels():
    import matplotlib.pyplot as plt

    result = _summary_plot_result()
    limits = structure_function_summary_ylims(
        [result, result], p_values=[0.9, 0.7], show_p_model=False
    )
    figures_and_axes = [
        plot_structure_function_summary_row(
            result,
            p_value=p_value,
            structure_ylim=limits[0],
            exponent_ylim=limits[1],
            kurtosis_ylim=limits[2],
            show_p_model=False,
        )
        for p_value in (0.9, 0.7)
    ]

    for figure, axes in figures_and_axes:
        for panel_index, expected in enumerate(limits):
            np.testing.assert_allclose(axes[panel_index].get_ylim(), expected)
        plt.close(figure)


def test_shared_summary_limits_can_exclude_kurtosis():
    result = _summary_plot_result()

    limits = structure_function_summary_ylims(
        [result, result],
        p_values=[0.9, 0.7],
        show_p_model=False,
        show_kurtosis=False,
    )

    assert len(limits) == 2
