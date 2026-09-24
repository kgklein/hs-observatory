"""Timing-only calibration contracts and small synthetic integration fixtures."""

from itertools import combinations
import json

import h5py
import numpy as np
import pytest
import yaml

from hs_observatory.analysis.magnetic_gradient import (
    align_spacecraft_samples,
    apply_gradient_response,
    estimate_gradient_least_squares,
    geometry_diagnostics,
    gradient_response_coefficients,
)
from hs_observatory.analysis.timing_calibrate import (
    _filter_polyhedra,
    _settings,
    calibrate_realization,
    compute_timing_calibration,
    divergence_variance,
    divergence_series,
    plot_timing_calibration_product,
    select_polyhedra,
    solve_timing,
    timing_jacobian,
    write_timing_calibration_product,
)
from hs_observatory.analyze import analyze_from_config
from hs_observatory.cli import plot_timing_calibrate_main
from hs_observatory.generate import generate_from_config
from hs_observatory.io.hdf5 import write_observations
from hs_observatory.spacecraft.configuration import synthetic_configuration


IDS = ["H", *(f"N{i}" for i in range(1, 9))]


@pytest.mark.parametrize("options", [None, {}, {"allowed_n": None},
                                   {"max_condition_number": None},
                                   {"allowed_n": None, "max_condition_number": None}])
def test_filter_no_cut_preserves_candidates(options):
    points = np.random.default_rng(20).normal(size=(9, 3))
    candidates = select_polyhedra(points, IDS, {})
    settings = _settings({"polyhedron_filter": options})
    selected, conditions = _filter_polyhedra(points[None], candidates, settings["polyhedron_filter"])
    assert list(selected) == list(candidates)
    assert len(selected) == 382
    for name in selected:
        np.testing.assert_array_equal(selected[name], candidates[name])
        assert conditions[name] == geometry_diagnostics(points[selected[name]]).condition_number


@pytest.mark.parametrize("sizes, count", [([4], 126), ([9], 1), ([4, 4], 126)])
def test_filter_spacecraft_counts(sizes, count):
    points = np.random.default_rng(20).normal(size=(9, 3))
    candidates = select_polyhedra(points, IDS, {})
    selected, _ = _filter_polyhedra(points[None], candidates, {
        "allowed_n": sizes, "max_condition_number": None,
    })
    assert len(selected) == count
    assert all(len(indices) in sizes for indices in selected.values())
    if sizes == [9]:
        np.testing.assert_array_equal(next(iter(selected.values())), np.arange(9))


@pytest.mark.parametrize("sizes", [None, [6, 7, 8, 9]])
def test_filter_condition_and_combined_cuts(sizes):
    points = np.random.default_rng(20).normal(size=(9, 3))
    candidates = select_polyhedra(points, IDS, {})
    expected = {name: geometry_diagnostics(points[indices]).condition_number
                for name, indices in candidates.items()}
    # Select an actual condition to exercise inclusive equality at the boundary.
    boundary = next(name for name, indices in candidates.items() if len(indices) == 6)
    threshold = expected[boundary]
    selected, conditions = _filter_polyhedra(points[None], candidates, {
        "allowed_n": sizes, "max_condition_number": threshold,
    })
    assert list(selected) == [name for name, indices in candidates.items()
                              if (sizes is None or len(indices) in sizes)
                              and expected[name] <= threshold]
    assert boundary in selected
    assert 0 < len(selected) < len(candidates)
    assert all(value <= threshold for value in conditions.values())


def test_filter_moving_and_translated_geometry():
    points = np.random.default_rng(20).normal(size=(9, 3))
    candidates = {"all_nine": np.arange(9)}
    moving = np.stack([points * [scale, 1, 1] for scale in (1, 2, 10)])
    expected = np.median([geometry_diagnostics(sample).condition_number for sample in moving])
    assert expected != geometry_diagnostics(moving[0]).condition_number
    _, conditions = _filter_polyhedra(moving, candidates, {
        "allowed_n": None, "max_condition_number": expected,
    })
    assert conditions["all_nine"] == expected
    with pytest.raises(ValueError, match="polyhedron_filter.*no polyhedra"):
        _filter_polyhedra(moving, candidates, {
            "allowed_n": None, "max_condition_number": np.nextafter(expected, 0),
        })
    translated = points[None] + np.arange(3)[:, None, None]
    _, conditions = _filter_polyhedra(translated, candidates, {
        "allowed_n": None, "max_condition_number": None,
    })
    assert conditions["all_nine"] == geometry_diagnostics(points).condition_number
    flat = np.zeros((1, 9, 3))
    _, conditions = _filter_polyhedra(flat, candidates, {
        "allowed_n": None, "max_condition_number": None,
    })
    assert conditions["all_nine"] == np.inf
    with pytest.raises(ValueError, match="polyhedron_filter.*no polyhedra"):
        _filter_polyhedra(flat, candidates, {"allowed_n": None, "max_condition_number": 100})


@pytest.mark.parametrize("options", [
    False, [], {"allowed_n": 4}, {"allowed_n": [True]}, {"allowed_n": [4.0]},
    {"allowed_n": [3]}, {"allowed_n": [10]}, {"allowed_n": ["4"]},
    *[{"max_condition_number": value} for value in (True, "100", 0, -1, np.inf, np.nan)],
])
def test_filter_invalid_settings(options):
    with pytest.raises(ValueError, match="polyhedron_filter"):
        _settings({"polyhedron_filter": options})


@pytest.mark.parametrize("options", [{"allowed_n": []}, {"max_condition_number": 0.5}])
def test_filter_empty_before_response_construction(tmp_path, synthetic, monkeypatch, options):
    source = _observation(tmp_path, synthetic)

    def unexpected_response(*args, **kwargs):
        pytest.fail("Responses must not be constructed for an empty selection")

    monkeypatch.setattr("hs_observatory.analysis.timing_calibrate.gradient_response_coefficients",
                        unexpected_response)
    with pytest.raises(ValueError, match="polyhedron_filter.*no polyhedra"):
        compute_timing_calibration(source, {"polyhedron_filter": options})


def test_filter_intersects_existing_selection(tmp_path, synthetic):
    source = _observation(tmp_path, synthetic)
    explicit = {"polyhedra": [IDS[:4], IDS], "polyhedron_filter": {"allowed_n": [9]}}
    result = compute_timing_calibration(source, explicit)
    assert list(result["selected"]) == ["_".join(IDS)]
    assert result["metadata"]["polyhedron_filter"]["candidate_count"] == 2
    legacy = {"configuration_selection": {
        "small": False, "intermediate": False, "large": False, "all_nine": True,
    }, "polyhedron_filter": {"allowed_n": [9]}}
    result = compute_timing_calibration(source, legacy)
    assert list(result["selected"]) == ["all_nine"]
    legacy["polyhedron_filter"] = {"allowed_n": [4]}
    with pytest.raises(ValueError, match="polyhedron_filter.*no polyhedra"):
        compute_timing_calibration(source, legacy)


@pytest.mark.parametrize("options", [None, {}, {"allowed_n": None, "max_condition_number": None}])
def test_no_filter_preserves_numerical_fit(tmp_path, synthetic, options):
    source = _observation(tmp_path, synthetic)
    time, positions, ideal, _, measured, polyhedra, responses = synthetic
    result = compute_timing_calibration(source, {
        "sigma_d_floor": 1e-6, "polyhedron_filter": options,
    })
    original_divergence = np.column_stack([
        divergence_series(response, ideal[:, indices])
        for indices, response in zip(polyhedra, responses)
    ])
    _, variance = divergence_variance(original_divergence, 1e-6)
    original = calibrate_realization(time, positions, measured, polyhedra, responses,
                                     np.arange(1, 9), variance, .01)
    actual = calibrate_realization(time, positions, measured, list(result["selected"].values()),
                                   result["responses"], result["node_indices"],
                                   result["variance"], result["sigma_t"])
    for key in ("correction_s", "covariance_s2", "fit_valid"):
        np.testing.assert_array_equal(actual[key], original[key])


def test_filtered_product_matches_fresh_explicit_fit(tmp_path, synthetic):
    source = _observation(tmp_path, synthetic)
    options = {"allowed_n": [8, 9, 9], "max_condition_number": 100}
    filtered = compute_timing_calibration(source, {
        "sigma_d_floor": 1e-6, "polyhedron_filter": options,
    })
    explicit = compute_timing_calibration(source, {
        "sigma_d_floor": 1e-6,
        "polyhedra": [[IDS[i] for i in indices] for indices in filtered["selected"].values()],
    })
    output = write_timing_calibration_product(filtered, tmp_path / "filtered.h5")
    reference = write_timing_calibration_product(explicit, tmp_path / "explicit.h5")
    with h5py.File(output) as actual, h5py.File(reference) as expected:
        for name in ("correction_s", "covariance_s2"):
            np.testing.assert_array_equal(actual[f"timing/{name}"], expected[f"timing/{name}"])
        metadata = json.loads(actual["metadata"].attrs["polyhedron_filter"])
        assert metadata["allowed_n"] == options["allowed_n"]
        assert metadata["max_condition_number"] == 100
        assert metadata["candidate_count"] == 382
        assert metadata["retained_count"] == len(filtered["selected"])
        assert list(actual["configurations"].attrs["order"]) == list(filtered["selected"])
        for name, indices in filtered["selected"].items():
            group = actual[f"configurations/{name}"]
            assert group.attrs["number_of_spacecraft"] == len(indices)
            assert group["condition_number"][()] == filtered["condition_numbers"][name]
            assert group["condition_number"].attrs["units"] == "1"
        unset = json.loads(expected["metadata"].attrs["polyhedron_filter"])
        assert unset["allowed_n"] is None and unset["max_condition_number"] is None
    source.unlink()
    plot = plot_timing_calibration_product(output, tmp_path / "filtered.png", dpi=40)
    assert plot.stat().st_size > 1000


def _field(time):
    return np.stack((np.sin(time), np.cos(time), np.sin(2 * time)), axis=-1)


@pytest.fixture
def synthetic():
    time = np.linspace(-1, 1, 101)
    points = np.random.default_rng(20).normal(size=(9, 3))
    positions = np.broadcast_to(points, (len(time), 9, 3)).copy()
    ideal = _field(np.broadcast_to(time[:, None], (len(time), 9)))
    offsets = np.r_[0.0, np.linspace(-0.0008, 0.0008, 8)]
    measured = _field(time[:, None] + offsets)
    polyhedra = list(select_polyhedra(points, IDS, {}).values())
    responses = [gradient_response_coefficients(positions[:, indices]) for indices in polyhedra]
    return time, positions, ideal, offsets, measured, polyhedra, responses


def _observation(tmp_path, synthetic, *, order=None):
    time, positions, ideal, offsets, measured, _, _ = synthetic
    order = np.arange(9) if order is None else order
    path = tmp_path / "observations.h5"
    n = np.ones((len(time), 9))
    v = np.zeros((len(time), 9, 3))
    generation = {"errors": {"time": {"model": "gaussian_constant", "sigma_s": 0.01}}}
    write_observations(
        path, time={q: time for q in ("n", "v", "B")},
        spacecraft_ids=[IDS[i] for i in order],
        position={q: positions[:, order].swapaxes(0, 1) for q in ("n", "v", "B")},
        truth={"n": n, "v": v, "B": ideal[:, order]},
        observed={"n": np.stack([n, n]), "v": np.stack([v, v]),
                  "B": np.stack([ideal[:, order], measured[:, order]])},
        errors={"time_offset": np.stack([np.zeros(9), offsets[order]])},
        metadata={"config_yaml": yaml.safe_dump(generation)},
    )
    return path


def test_cached_gradient_matches_scalar_for_static_and_moving_geometry():
    rng = np.random.default_rng(10)
    for moving in (False, True):
        points = rng.normal(size=(9, 3))
        positions = np.broadcast_to(points, (4, 9, 3)).copy()
        positions += np.arange(4)[:, None, None] * np.array([0.4, -0.3, 0.2])
        if moving:
            positions += rng.normal(size=positions.shape) * 0.05
        field = rng.normal(size=positions.shape)
        response = gradient_response_coefficients(positions)
        assert response.shape[0] == (4 if moving else 1)
        actual = apply_gradient_response(response, field)
        expected = np.array([estimate_gradient_least_squares(p, b).gradient
                             for p, b in zip(positions, field)])
        np.testing.assert_allclose(actual, expected, atol=2e-14)
    flat = np.zeros((2, 4, 3))
    rejected = gradient_response_coefficients(flat, allow_pseudoinverse=False)
    assert np.isnan(rejected).all()


def test_selection_and_variance(synthetic):
    _, positions, _, _, _, polyhedra, _ = synthetic
    assert len(polyhedra) == 382
    assert [tuple(p) for p in polyhedra] == [p for n in range(4, 10) for p in combinations(range(9), n)]
    selected = select_polyhedra(positions[0], IDS, {"polyhedra": [["N3", "H", "N2", "N1"]]})
    assert list(selected) == ["H_N1_N2_N3"]
    raw, variance = divergence_variance(np.array([[2., 0.], [2., 0.]]), 0.1)
    np.testing.assert_allclose(raw, [4, 0])
    np.testing.assert_allclose(variance, [4, 0.01])
    with pytest.raises(ValueError, match="not both"):
        select_polyhedra(positions[0], IDS, {"polyhedra": [], "configuration_selection": {}})
    with pytest.raises(ValueError, match="unique"):
        select_polyhedra(positions[0], IDS, {"polyhedra": [["H"] * 4]})
    legacy = select_polyhedra(positions[0], IDS, {"configuration_selection": {
        "small": False, "intermediate": False, "large": False, "all_nine": True,
    }})
    assert list(legacy) == ["all_nine"]


def test_zero_offset_and_known_offsets(synthetic):
    time, positions, ideal, offsets, measured, polyhedra, responses = synthetic
    variance = np.full(len(polyhedra), 1e-12)
    for values, injected in ((ideal, np.zeros(9)), (measured, offsets)):
        result = calibrate_realization(time, positions, values, polyhedra, responses,
                                       np.arange(1, 9), variance, 0.01)
        assert result["correction_s"].shape == (8,)
        # One-sided endpoint derivatives and nonlinear offsets introduce truncation error.
        tolerance = 2e-7 if np.any(injected) else 1e-12
        np.testing.assert_allclose(result["correction_s"], -injected[1:], atol=tolerance, rtol=0.01)
        np.testing.assert_allclose(result["corrected_B"][:, 0], values[:, 0])
        assert result["corrected_B"].shape == ideal.shape
        assert result["measured_divergence"].shape == (len(time), 382)
        assert result["chi2_after"] <= result["chi2_before"] + 1e-10
        assert result["linear_chi2_after"] <= result["linear_chi2_before"] + 1e-10
        assert result["prior_chi2_before"] == 0
        assert result["chi2_after"] == pytest.approx(
            result["divergence_chi2_after"] + result["prior_chi2_after"])
        if np.any(injected):
            assert not result["corrected_B_valid"].all()
            assert np.isnan(result["corrected_B"]).any()
            common = result["comparison_valid"]
            expected = np.sum((result["measured_divergence"][common] / 1e-6) ** 2)
            assert result["divergence_chi2_before"] == pytest.approx(expected)


def test_jacobian_membership_sign_and_finite_difference(synthetic):
    time, _, _, _, measured, polyhedra, responses = synthetic
    jacobian = timing_jacobian(time, measured, polyhedra, responses, np.arange(1, 9))
    derivative = np.gradient(measured, time, axis=0)
    assert jacobian.shape == (len(time), 382, 8)
    for column, indices in enumerate(polyhedra):
        for node in range(1, 9):
            if node not in indices:
                assert np.array_equal(jacobian[:, column, node - 1], np.zeros(len(time)))
    indices = polyhedra[0]
    step = 1e-5
    perturbation = np.zeros_like(measured)
    perturbation[:, 1] = derivative[:, 1] * step
    before = apply_gradient_response(responses[0], measured[:, indices])
    after = apply_gradient_response(responses[0], (measured + perturbation)[:, indices])
    np.testing.assert_allclose(np.trace(after - before, axis1=1, axis2=2) / step,
                               jacobian[:, 0, 0], atol=1e-9)


def test_augmented_solver_covariance_and_rank_deficiency():
    rng = np.random.default_rng(15)
    jacobian = rng.normal(size=(20, 3, 8))
    divergence = rng.normal(size=(20, 3))
    variance = np.array([0.1, 2., 3.])
    result = solve_timing(divergence, jacobian, variance, 0.2)
    weighted = (jacobian / np.sqrt(variance)[None, :, None]).reshape(-1, 8)
    normal = weighted.T @ weighted + np.eye(8) / 0.2**2
    np.testing.assert_allclose(normal @ result["covariance_s2"], np.eye(8), atol=1e-13)
    assert result["weighted_jacobian"]["rank"] == 8
    assert np.all(np.linalg.eigvalsh(result["covariance_s2"]) > 0)
    zero = solve_timing(divergence, np.zeros_like(jacobian), variance, 0.2)
    assert zero["jacobian"]["condition_number"] == np.inf
    assert zero["jacobian"]["rank"] == 0
    np.testing.assert_array_equal(zero["correction_s"], np.zeros(8))
    np.testing.assert_allclose(zero["covariance_s2"], np.eye(8) * 0.2**2)
    with pytest.raises(ValueError, match="No valid"):
        solve_timing(np.full_like(divergence, np.nan), jacobian, variance, 0.2)


def test_hdf5_cli_roundtrip_ordering_and_product_only_plot(tmp_path, synthetic):
    order = np.array([3, 2, 0, 1, 4, 5, 6, 7, 8])
    source = _observation(tmp_path, synthetic, order=order)
    config = tmp_path / "analysis.yaml"
    output = tmp_path / "calibrated.h5"
    config.write_text(yaml.safe_dump({
        "input": {"path": str(source)}, "analysis": {"method": "timing-calibrate"},
        "output": {"path": str(output)}, "timing_calibrate": {"sigma_d_floor": 1e-6},
    }))
    assert analyze_from_config(config) == output
    with h5py.File(output) as handle:
        assert handle["metadata"].attrs["complete"]
        assert "timing-calibrate" in handle["metadata"].attrs["analysis_config_yaml"]
        assert handle["spacecraft/id"].asstr()[...].tolist() == [IDS[i] for i in order]
        assert "H" not in handle["spacecraft/node_id"].asstr()[...]
        assert handle["magnetic_field/ideal"].shape == (101, 9, 3)
        assert handle["magnetic_field/measured"].shape == (2, 101, 9, 3)
        assert handle["magnetic_field/corrected"].shape == (2, 101, 9, 3)
        assert handle["timing/covariance_s2"].shape == (2, 8, 8)
        assert handle["timing/covariance_s2"].attrs["units"] == "s^2"
        assert len(handle["configurations"]) == 382
        assert handle["magnetic_field/measured"].compression == "gzip"
        recovered = handle["timing/recovered_offset_s"][...]
        injected = handle["timing/injected_offset_s"][...]
        np.testing.assert_allclose(handle["mc_summary/bias_s"][...], (recovered - injected).mean(0))
        np.testing.assert_allclose(handle["mc_summary/std_recovered_offset_s"][...], recovered.std(0))
        assert handle["mc_summary/chi2_before"].id == handle["diagnostics/chi2_before"].id
    source.unlink()  # Plotting must not reopen the source observations.
    plot = tmp_path / "diagnostics.png"
    assert plot_timing_calibrate_main([str(output), "--output", str(plot), "--dpi", "40"]) == 0
    assert plot.stat().st_size > 1000
    with pytest.raises(ValueError, match="mc_index"):
        plot_timing_calibration_product(output, plot, mc_index=2)


def test_input_rejection_and_explicit_selection(tmp_path, synthetic):
    source = _observation(tmp_path, synthetic)
    config = {"polyhedra": [["H", "N1", "N2", "N3"]], "sigma_d_floor": 1e-6}
    result = compute_timing_calibration(source, config)
    assert len(result["selected"]) == 1
    with pytest.raises(ValueError, match="overwrite"):
        write_timing_calibration_product(result, source)
    with h5py.File(source, "r+") as handle:
        handle["errors/time_offset"][0, 0] = 0.1
    with pytest.raises(ValueError, match="regenerate"):
        compute_timing_calibration(source, config)
    with h5py.File(source, "r+") as handle:
        handle["errors/time_offset"][0, 0] = 0
        handle["metadata"].attrs["config_yaml"] = yaml.safe_dump({"errors": {
            "model": "observatory", "timing": {"enabled": True},
        }})
    with pytest.raises(ValueError, match="timing-only"):
        compute_timing_calibration(source, config)


def test_reference_generation_preserves_node_draws_and_default(tmp_path):
    config = {
        "spacecraft": {"source": "synthetic", "velocity_km_s": [2, 0, 0]},
        "time": {"t_start_s": 0, "t_stop_s": 0.5, "cadence_s": 0.25},
        "field": {"model": "sine_sum", "background": {"n": 1, "v": [0, 0, 0], "B": [0, 0, 1]},
                  "modes": [{"quantity": "B", "amplitude": [1, 0, 0], "k": [0, 0, 0], "omega": 1}]},
        "errors": {"random_seed": 12, "n_mc": 2,
                   "time": {"model": "gaussian_constant", "sigma_s": 0.01}},
        "output": {"path": str(tmp_path / "observations.h5")},
        "visualization": {"enabled": False},
    }
    path = tmp_path / "generate.yaml"
    path.write_text(yaml.safe_dump(config))
    generate_from_config(path)
    with h5py.File(config["output"]["path"]) as handle:
        original = handle["errors/time_offset"][...]
    assert np.any(original[:, 0] != 0)
    config["errors"]["time"]["reference_spacecraft"] = "H"
    path.write_text(yaml.safe_dump(config))
    generate_from_config(path)
    with h5py.File(config["output"]["path"]) as handle:
        actual = handle["errors/time_offset"][...]
        np.testing.assert_array_equal(actual[:, 0], [0, 0])
        np.testing.assert_array_equal(actual[:, 1:], original[:, 1:])
        np.testing.assert_allclose(handle["observed/B"][:, :, 0],
                                   np.broadcast_to(handle["truth/B"][:, 0], (2, 3, 3)))
    config["errors"]["time"]["reference_spacecraft"] = "unknown"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="existing spacecraft"):
        generate_from_config(path)


@pytest.fixture
def linear_timing_toy():
    """Existing formation and recoverable offsets for the rank-six Bx-only toy."""
    time = np.linspace(0.0, 1.0, 11)
    positions = synthetic_configuration(time).position.swapaxes(0, 1)
    polyhedra = list(select_polyhedra(positions[0], IDS, {}).values())
    responses = [gradient_response_coefficients(positions[:, indices]) for indices in polyhedra]
    # tau = b*(y-y_H) + c*(z-z_H) yields only transverse gradients of Bx,
    # hence zero divergence. A finite isotropic prior selects zero in this nullspace.
    transverse = positions[0, 1:, 1:] - positions[0, 0, 1:]
    seed = np.array([2, -1, 3, -4, 2, -3, 1, -2], dtype=float) * 1e-4
    multi = seed - transverse @ np.linalg.lstsq(transverse, seed, rcond=None)[0]
    one_node = np.zeros(8)
    one_node[0] = 2e-4  # N1 lies on the x axis: this offset is orthogonal to both null modes.
    cases = [np.r_[0.0, nodes] for nodes in (one_node, -one_node, multi, -0.5 * multi)]
    return time, positions, polyhedra, responses, transverse, cases


def test_exact_divergence_free_linear_timing_toy(linear_timing_toy):
    time, positions, polyhedra, responses, transverse, cases = linear_timing_toy
    slope = 2.0
    ideal = np.zeros_like(positions)
    ideal[:, :, 0] = slope * time[:, None]
    ideal_d = np.column_stack([
        divergence_series(response, ideal[:, indices])
        for indices, response in zip(polyhedra, responses)
    ])
    np.testing.assert_allclose(ideal_d, 0, atol=1e-14)
    _, variance = divergence_variance(ideal_d, floor=1e-8)
    jacobian = timing_jacobian(time, ideal, polyhedra, responses, np.arange(1, 9))
    assert np.linalg.matrix_rank(jacobian.reshape(-1, 8)) == 6
    null_offsets = 1e-4 * transverse / np.linalg.norm(transverse, axis=0)
    np.testing.assert_allclose(jacobian @ null_offsets, 0, atol=1e-14)
    for injected in cases:
        np.testing.assert_allclose(transverse.T @ injected[1:], 0, atol=1e-16)
        measured = ideal.copy()
        measured[:, :, 0] = slope * (time[:, None] + injected[None, :])
        result = calibrate_realization(
            time, positions, measured, polyhedra, responses, np.arange(1, 9), variance, 0.0125
        )
        assert np.max(np.abs(result["measured_divergence"])) > 1e-6
        np.testing.assert_allclose(result["correction_s"], -injected[1:], atol=1e-11, rtol=0)
        np.testing.assert_allclose(result["recovered_offset_s"], injected[1:], atol=1e-11, rtol=0)
        np.testing.assert_allclose(result["corrected_B"][:, 0], ideal[:, 0], atol=1e-14)
        valid = result["corrected_B_valid"]
        np.testing.assert_allclose(result["corrected_B"][valid], ideal[valid], atol=3e-11, rtol=0)
        valid_d = result["comparison_valid"]
        np.testing.assert_allclose(result["corrected_divergence"][valid_d],
                                   ideal_d[valid_d], atol=1e-12, rtol=0)

        # Alignment's argument is a source-clock offset, not a sampling correction.
        correction = np.r_[0.0, result["correction_s"]]
        aligned = align_spacecraft_samples(time, -correction, positions.swapaxes(0, 1), measured)
        np.testing.assert_allclose(aligned.field, result["corrected_B"], equal_nan=True)
        known_alignment = align_spacecraft_samples(
            time, injected, positions.swapaxes(0, 1), measured
        )
        np.testing.assert_allclose(known_alignment.field[known_alignment.valid],
                                   ideal[known_alignment.valid], atol=1e-14, rtol=0)
        wrong_sign = align_spacecraft_samples(
            time, correction, positions.swapaxes(0, 1), measured
        )
        doubled = ideal.copy()
        doubled[:, :, 0] += 2 * slope * injected[None, :]
        np.testing.assert_allclose(wrong_sign.field[wrong_sign.valid],
                                   doubled[wrong_sign.valid], atol=3e-11, rtol=0)
        assert np.max(np.abs(wrong_sign.field[wrong_sign.valid] - ideal[wrong_sign.valid])) > 1e-4


def test_timing_fit_ideal_divergence_residual(linear_timing_toy):
    time, positions, polyhedra, responses, _, cases = linear_timing_toy
    slope = 2.0
    # A deliberately nonzero ideal divergence isolates the target subtraction.
    # The static Bx = b*x term does not change the exact linear timing derivative.
    ideal = np.zeros_like(positions)
    ideal[:, :, 0] = slope * time[:, None] + 1e-6 * positions[:, :, 0]
    ideal_d = np.column_stack([
        divergence_series(response, ideal[:, indices])
        for indices, response in zip(polyhedra, responses)
    ])
    np.testing.assert_allclose(ideal_d, 1e-6, atol=1e-14, rtol=0)
    raw, variance = divergence_variance(ideal_d, floor=1e-12)
    np.testing.assert_array_equal(raw, np.mean(ideal_d**2, axis=0))
    np.testing.assert_array_equal(variance, raw)  # No reweighting or mean subtraction.
    for injected in [np.zeros(9), *cases]:
        measured = ideal.copy()
        measured[:, :, 0] += slope * injected[None, :]
        measured_d = np.column_stack([
            divergence_series(response, measured[:, indices])
            for indices, response in zip(polyhedra, responses)
        ])
        jacobian = timing_jacobian(time, measured, polyhedra, responses, np.arange(1, 9))
        result = solve_timing(measured_d, jacobian, variance, 0.0125, ideal_divergence=ideal_d)
        if not np.any(injected):
            np.testing.assert_array_equal(measured_d, ideal_d)
            np.testing.assert_allclose(result["correction_s"], 0, atol=1e-15, rtol=0)
            assert result["linear_chi2_before"] == result["linear_chi2_after"] == 0.0
            absolute = solve_timing(measured_d, jacobian, variance, 0.0125)
            assert np.max(np.abs(absolute["correction_s"])) > 1e-6
        # A finite prior shrinks nonzero offsets; these weights make that bias negligible.
        np.testing.assert_allclose(result["correction_s"], -injected[1:], atol=1e-10, rtol=0)
        np.testing.assert_allclose(result["recovered_offset_s"], injected[1:], atol=1e-10, rtol=0)
        residual = measured_d - ideal_d
        predicted = residual + jacobian @ result["correction_s"]
        prior = np.sum((result["correction_s"] / 0.0125)**2)
        np.testing.assert_allclose(result["linear_chi2_before"], np.sum(residual**2 / variance))
        np.testing.assert_allclose(result["linear_chi2_after"],
                                   np.sum(predicted**2 / variance) + prior, atol=1e-12)
        aligned = align_spacecraft_samples(
            time, np.r_[0.0, -result["correction_s"]], positions.swapaxes(0, 1), measured
        )
        np.testing.assert_allclose(aligned.field[aligned.valid], ideal[aligned.valid],
                                   atol=3e-10, rtol=0)
        corrected_d = np.column_stack([
            divergence_series(response, aligned.field[:, indices])
            for indices, response in zip(polyhedra, responses)
        ])
        valid = np.isfinite(corrected_d)
        np.testing.assert_allclose(corrected_d[valid], ideal_d[valid], atol=1e-11, rtol=0)


def test_yaml_ideal_reference_reaches_fit_and_saved_objectives(tmp_path, synthetic):
    source = _observation(tmp_path, synthetic)
    with h5py.File(source, "r+") as handle:
        positions = handle["spacecraft/r/B"][...].swapaxes(0, 1)
        ideal = handle["truth/B"][...]
        ideal[:, :, 0] += 1e-3 * positions[:, :, 0]
        handle["truth/B"][...] = ideal
        handle["observed/B"][...] = np.stack([ideal, ideal])
        handle["errors/time_offset"][...] = 0
    output = tmp_path / "residual.h5"
    config = tmp_path / "residual.yaml"
    config.write_text(yaml.safe_dump({
        "input": {"path": str(source)}, "analysis": {"method": "timing-calibrate"},
        "output": {"path": str(output)}, "timing_calibrate": {
            "use_ideal_divergence": True, "sigma_t_s": 0.0125,
            "polyhedra": [["H", "N1", "N2", "N3"]],
        },
    }))
    analyze_from_config(config)
    with h5py.File(output) as handle:
        assert handle["metadata"].attrs["use_ideal_divergence"]
        assert handle["metadata"].attrs["chi2_target"] == "measured_minus_ideal"
        np.testing.assert_array_equal(handle["timing/correction_s"][...], 0)
        for label in ("before", "after"):
            for prefix in ("chi2", "linear_chi2", "divergence_chi2"):
                np.testing.assert_array_equal(handle[f"diagnostics/{prefix}_{label}"][...], 0)
            np.testing.assert_allclose(
                handle[f"diagnostics/normalized_divergence_rms_{label}"][...], 1.0
            )
    default = compute_timing_calibration(source, {"polyhedra": [["H", "N1", "N2", "N3"]]})
    assert not default["settings"]["use_ideal_divergence"]
