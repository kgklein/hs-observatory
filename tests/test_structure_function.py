from pathlib import Path

import h5py
import numpy as np
import pytest

from hs_observatory.analysis.registry import list_analyses
from hs_observatory.analysis.structure_function import (
    StructureFunctionAnalysis,
    _fit_scaling_exponents,
    compute_pairwise_structure_functions,
    plot_structure_function_diagnostic,
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
    np.testing.assert_allclose(result["z_plus"]["flatness"][:, 0], [272.0 / 100.0, 1.0])


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
            "plot": {"enabled": True, "path": str(tmp_path / "sf.png")},
        }
    )

    result = analysis.run(input_path)
    output = analysis.write_result(result, tmp_path / "sf.json")

    assert "structure_function" in list_analyses()
    assert output.is_file()
    assert (tmp_path / "sf.png").is_file()
    assert set(result["products"]) == {"observed", "truth"}
    assert len(result["products"]["observed"]["counts"]) == 2
    truth_separations = np.asarray(result["products"]["truth"]["pair_separations_km"])
    observed_separations = np.asarray(result["products"]["observed"]["pair_separations_km"])
    assert truth_separations[0, 0, 0] == pytest.approx(1.0)
    assert observed_separations[0, 0, 0] == pytest.approx(11.0)
    assert observed_separations[1, 1, 0] == pytest.approx(np.hypot(3.0, 20.0))


def test_plot_helper_accepts_serialized_result(tmp_path):
    result = {
        "products": {
            "observed": {
                "orders": [2, 4],
                "lambda_bin_centers_km": [1.0, 2.0, 4.0],
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
