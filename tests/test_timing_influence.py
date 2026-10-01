"""Additive timing influence identities, geometry reuse, and opt-in integration."""

import h5py
import numpy as np
import pytest
import yaml

from hs_observatory.analysis.magnetic_gradient import estimate_gradient_least_squares
from hs_observatory.analysis.timing_calibrate import compute_timing_calibration, solve_timing
from hs_observatory.analysis.timing_influence import _aggregate, decompose_timing_influence
from hs_observatory.analyze import analyze_from_config
from hs_observatory.io.hdf5 import write_observations
from hs_observatory.spacecraft.configuration import synthetic_configuration


@pytest.mark.parametrize("rank_deficient", [False, True])
def test_influence_sum_identities_and_existing_solver(rank_deficient):
    rng = np.random.default_rng(42)
    jacobian = rng.normal(size=(15, 12, 8))
    if rank_deficient:
        jacobian[..., -1] = jacobian[..., 0]
    ideal = rng.normal(size=(15, 12)) + 0.4
    variance = np.mean(ideal**2, axis=0)
    # The diagnostic must use exactly the solver's row mask, including invalid rows.
    ideal[0, 2] = np.nan
    jacobian[1, 4, 3] = np.nan
    variance[7] = np.nan
    actual = decompose_timing_influence(ideal, jacobian, variance, 0.0125)
    reference = solve_timing(ideal, jacobian, variance, 0.0125)
    valid = reference["fit_valid"]
    weighted = jacobian[valid] / np.broadcast_to(np.sqrt(variance), ideal.shape)[valid, None]
    expected_h = weighted.T @ weighted + np.eye(8) / 0.0125**2
    np.testing.assert_allclose(actual["H"], expected_h)
    np.testing.assert_allclose(actual["g_polyhedron"].sum(axis=1), actual["g"], atol=1e-12)
    np.testing.assert_allclose(
        actual["polyhedron_correction_s"].sum(axis=1), actual["predicted_correction_s"], atol=1e-14
    )
    np.testing.assert_allclose(
        actual["predicted_correction_s"], reference["correction_s"], atol=1e-14
    )
    np.testing.assert_allclose(
        actual["H"] @ actual["polyhedron_correction_s"], -actual["g_polyhedron"], atol=1e-12
    )
    assert actual["g_sum_matches"] and actual["c_sum_matches"] and actual["solver_matches"]
    assert np.any(actual["polyhedron_correction_s"] < 0)
    assert np.any(actual["polyhedron_correction_s"] > 0)
    np.testing.assert_allclose(
        actual["cumulative_signed_s"][:, -1], reference["correction_s"], atol=1e-14
    )
    np.testing.assert_allclose(
        actual["cumulative_absolute_s"][:, -1],
        np.abs(actual["polyhedron_correction_s"]).sum(axis=1),
    )
    for metric in (np.arange(12.0), np.full(12, 2.0), np.full(12, np.nan)):
        aggregate = _aggregate(metric, actual["polyhedron_correction_s"])
        np.testing.assert_allclose(
            aggregate["signed_s"].sum(axis=1), reference["correction_s"], atol=1e-14
        )
        np.testing.assert_allclose(
            aggregate["absolute_s"].sum(axis=1),
            np.abs(actual["polyhedron_correction_s"]).sum(axis=1),
        )
        assert aggregate["counts"].sum() == 12


@pytest.mark.parametrize("allowed_n, retained", [(None, 3), ([5, 9], 2)])
def test_zero_offset_influence_hdf5_and_existing_geometry(tmp_path, allowed_n, retained):
    time = np.linspace(0, 1, 12)
    formation = synthetic_configuration(time)
    positions = formation.position.swapaxes(0, 1)
    x, y = positions[..., 0], positions[..., 1]
    field = np.stack(
        (
            1e-4 * x * y + np.sin(time[:, None]),
            -0.5e-4 * y * y + np.cos(time[:, None]),
            np.broadcast_to(np.sin(2 * time[:, None]), x.shape),
        ),
        axis=-1,
    )
    source = tmp_path / "observations.h5"
    zeros = np.zeros((len(time), 9))
    write_observations(
        source,
        time={q: time for q in ("n", "v", "B")},
        spacecraft_ids=formation.spacecraft_ids,
        position={q: formation.position for q in ("n", "v", "B")},
        truth={"n": zeros, "v": np.zeros_like(field), "B": field},
        observed={"n": zeros[None], "v": np.zeros_like(field)[None], "B": field[None]},
        errors={"time_offset": np.zeros((1, 9))},
        metadata={"config_yaml": yaml.safe_dump({"errors": {"time": {"model": "none"}}})},
    )
    options = {
        "sigma_t_s": 0.0125,
        "diagnostics": {"polyhedron_influence": True},
        "polyhedron_filter": {"allowed_n": allowed_n},
        "polyhedra": [
            ["H", "N1", "N2", "N3"],
            ["H", "N1", "N2", "N3", "N4"],
            list(formation.spacecraft_ids),
        ],
    }
    output = tmp_path / "influence.h5"
    config = tmp_path / "analysis.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "input": {"path": str(source)},
                "analysis": {"method": "timing-calibrate"},
                "output": {"path": str(output)},
                "timing_calibrate": options,
            }
        )
    )
    analyze_from_config(config)
    with h5py.File(output) as handle:
        group = handle["timing_influence"]
        assert group["g_polyhedron"].shape == (8, retained)
        assert list(group["polyhedron_id"].asstr()[...]) == list(
            handle["configurations"].attrs["order"]
        )
        assert group["g_polyhedron"].attrs["units"] == "1/s"
        assert group["polyhedron_correction_s"].attrs["units"] == "s"
        np.testing.assert_allclose(
            group["predicted_correction_s"][...], handle["timing/correction_s"][0], atol=1e-12
        )
        for column, name in enumerate(group["polyhedron_id"].asstr()[...]):
            entry = handle[f"configurations/{name}"]
            indices = entry["spacecraft_indices"][...]
            assert allowed_n is None or len(indices) in allowed_n
            geometry = entry["influence_geometry"]
            expected = estimate_gradient_least_squares(positions[0, indices], field[0, indices])
            np.testing.assert_allclose(geometry["volumetric_tensor"][0], expected.volumetric_tensor)
            np.testing.assert_allclose(geometry["condition_number"][0], expected.condition_number)
            assert entry["condition_number"][()] == group["polyhedra/condition_number"][column]
            np.testing.assert_allclose(
                geometry["characteristic_size"][0], 2 * np.sqrt(expected.eigenvalues[0])
            )
            assert group["polyhedra/sigma_d"][column] == pytest.approx(
                np.sqrt(entry["sigma_d_squared"][()])
            )
            if len(indices) > 4:
                expected_rms = [
                    estimate_gradient_least_squares(p[indices], b[indices]).rms_residual
                    for p, b in zip(positions, field)
                ]
                np.testing.assert_allclose(
                    geometry["affine_rms_residual"][...], expected_rms, atol=1e-14
                )
            else:
                assert "affine_rms_residual" not in geometry
        for entry in group["aggregates"].values():
            np.testing.assert_allclose(
                entry["signed_s"][...].sum(axis=1), group["predicted_correction_s"][...], atol=1e-12
            )
    with pytest.raises(ValueError, match="use_ideal_divergence: false"):
        compute_timing_calibration(source, {**options, "use_ideal_divergence": True})
    with h5py.File(source, "r+") as handle:
        handle["errors/time_offset"][0, 1] = 0.001
    with pytest.raises(ValueError, match="zero injected"):
        compute_timing_calibration(source, options)
