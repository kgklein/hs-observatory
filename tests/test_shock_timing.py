from itertools import combinations
import json

import h5py
import numpy as np
import pytest

from hs_observatory.analysis.registry import list_analyses
from hs_observatory.analysis.shock_timing import (
    DETECTION_STATUS,
    INVERSION_STATUS,
    _settings,
    _truth,
    compute_shock_timing,
    detect_transition,
    fit_combined_transition,
    solve_planar_timing,
    write_shock_timing_product,
)
from hs_observatory.analyze import analyze_from_config


def settings():
    value = _settings({})
    return value["detection"], value["fitting"]


@pytest.mark.parametrize("jump", [4.0, -4.0])
def test_detects_positive_and_negative_tanh_crossings(jump):
    time = np.arange(0.0, 20.0, 0.05)
    rng = np.random.default_rng(4)
    values = 2.0 + jump * 0.5 * (1 + np.tanh(2.197224577 * (time - 9.25) / 0.4))
    values += rng.normal(0.0, 0.03, len(time))

    result = detect_transition(time, values, *settings())

    assert result.success
    assert result.crossing_time == pytest.approx(9.25, abs=0.02)
    assert result.width == pytest.approx(0.4, abs=0.05)
    assert np.sign(result.downstream - result.upstream) == np.sign(jump)


def test_rejects_flat_and_ambiguous_channels():
    time = np.arange(0.0, 20.0, 0.05)
    rng = np.random.default_rng(8)
    flat = rng.normal(0.0, 0.1, len(time))
    assert (
        detect_transition(time, flat, *settings()).status
        == DETECTION_STATUS["no_detectable_shock"]
    )
    multiple = flat + 3.0 * (time >= 6.0) - 3.0 * (time >= 14.0)
    assert (
        detect_transition(time, multiple, *settings()).status
        == DETECTION_STATUS["ambiguous_transition"]
    )


def test_incomplete_transition_reports_missing_precoverage():
    time = np.arange(0.0, 20.0, 0.05)
    values = 1.0 + 4.0 * 0.5 * (1 + np.tanh(2.197224577 * (time - 0.4) / 0.2))

    result = detect_transition(time, values, *settings())

    assert result.status == DETECTION_STATUS["insufficient_pre_samples"]


def test_combined_fit_uses_native_cadences_and_shared_time():
    detection_settings, fitting_settings = settings()
    channels = []
    for cadence, jump in ((0.125, 4.0), (0.03125, -2.0)):
        time = np.arange(0.0, 20.0, cadence)
        values = 1.0 + jump * 0.5 * (1 + np.tanh(2.197224577 * (time - 10.2) / 0.3))
        detection = detect_transition(time, values, detection_settings, fitting_settings)
        assert detection.success
        channels.append((time, values, detection))

    combined = fit_combined_transition(channels, fitting_settings)

    assert combined.success
    assert combined.crossing_time == pytest.approx(10.2, abs=1.0e-3)
    assert combined.width == pytest.approx(0.3, abs=1.0e-3)


def test_exact_timing_and_equal_weight_wls_agree():
    rng = np.random.default_rng(2)
    initial = rng.normal(size=(9, 3)) * [300.0, 500.0, 700.0]
    velocity = np.array([1.0, -0.2, 0.1])
    q = np.array([-0.01, 0.001, -0.0005])
    reference = initial.mean(axis=0)
    true_tref = 20.0
    crossing = np.array([
        (true_tref + q @ (point - reference)) / (1.0 - q @ velocity)
        for point in initial
    ])
    positions = initial + crossing[:, None] * velocity

    ols = solve_planar_timing(crossing, positions, reference, 20.0)
    wls = solve_planar_timing(
        crossing, positions, reference, 20.0, uncertainties=np.ones(9) * 0.1
    )

    assert ols.success and wls.success
    np.testing.assert_allclose(ols.q, q, atol=1.0e-12)
    np.testing.assert_allclose(wls.q, q, atol=1.0e-12)
    assert ols.reference_time == pytest.approx(true_tref)
    assert len(list(combinations(range(9), 4))) == 126


def test_timing_rejects_missing_rank_and_condition():
    times = np.arange(4.0)
    flat = np.column_stack((np.arange(4.0), np.zeros(4), np.zeros(4)))
    result = solve_planar_timing(times, flat, np.zeros(3), 0.0)
    assert result.status == INVERSION_STATUS["degenerate_geometry"]
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0e-9]])
    result = solve_planar_timing(times, points, np.zeros(3), 0.0, max_condition_number=10.0)
    assert result.status == INVERSION_STATUS["poor_conditioning"]
    result = solve_planar_timing(times[:3], points[:3], np.zeros(3), 0.0)
    assert result.status == INVERSION_STATUS["insufficient_spacecraft"]


def test_truth_converts_generator_normal_and_moving_crossings():
    metadata = {
        "field_model": "shock",
        "field_metadata": json.dumps({
            "normal_gse": [1, 0, 0], "velocity_gse_km_s": [-100, 0, 0],
            "mesocenter_gse_km": [0, 0, 0], "reference_time_s": 0,
            "distance_from_mesocenter_km": 2000,
        }),
    }
    time = np.linspace(0, 30, 301)
    trajectory = np.zeros((2, len(time), 3))
    trajectory[0, :, 0] = time
    trajectory[1, :, 0] = 100 + 2 * time

    truth = _truth(metadata, np.zeros(3), time, trajectory)

    np.testing.assert_array_equal(truth["propagation_normal"], [-1, 0, 0])
    np.testing.assert_allclose(truth["q"], [-0.01, 0, 0])
    assert truth["speed"] == 100
    assert truth["reference_time"] == 20
    np.testing.assert_allclose(truth["arrival_times"], [2000 / 101, 1900 / 102])


def _observation(path, *, n_mc=2, include_truth=True):
    ids = ["H", *(f"N{i}" for i in range(1, 9))]
    offsets = np.array([
        [0, 0, 0], [250, 0, 0], [-200, 80, 30], [50, 350, -40],
        [20, -300, 100], [120, 80, 450], [-80, -50, -400],
        [550, 200, 150], [-500, -180, -120],
    ], dtype=float)
    offsets -= offsets.mean(axis=0)
    grids = {"n": np.arange(0, 40.0001, 0.125),
             "v": np.arange(0, 40.0001, 0.125),
             "B": np.arange(0, 40.0001, 0.03125)}
    crossings = 20.0 - offsets[:, 0] / 100.0
    rng = np.random.default_rng(11)
    with h5py.File(path, "w") as handle:
        meta = handle.create_group("metadata")
        meta.attrs["field_model"] = "shock" if include_truth else "unknown"
        if include_truth:
            meta.attrs["field_metadata"] = json.dumps({
                "normal_gse": [1, 0, 0], "velocity_gse_km_s": [-100, 0, 0],
                "mesocenter_gse_km": [0, 0, 0], "reference_time_s": 0,
                "distance_from_mesocenter_km": 2000,
            })
        time_group = handle.create_group("time")
        spacecraft = handle.create_group("spacecraft")
        spacecraft.create_dataset("id", data=np.asarray(ids, dtype=h5py.string_dtype()))
        positions = spacecraft.create_group("r")
        observed = handle.create_group("observed")
        for quantity, time in grids.items():
            time_group.create_dataset(quantity, data=time)
            positions.create_dataset(quantity, data=np.broadcast_to(
                offsets[:, None, :], (9, len(time), 3)
            ))
            fraction = 0.5 * (1 + np.tanh(
                2.197224577 * (time[:, None] - crossings[None, :]) / 0.25
            ))
            if quantity == "n":
                base = 5 + 12 * fraction
                values = base[None] + rng.normal(0, 0.08, (n_mc, *base.shape))
            elif quantity == "v":
                base = np.zeros((len(time), 9, 3))
                base[..., 0] = 400 - 300 * fraction
                values = base[None] + rng.normal(0, 0.5, (n_mc, *base.shape))
            else:
                base = np.zeros((len(time), 9, 3))
                base[..., 0] = 0.5
                base[..., 1] = 5 + 12 * fraction
                base[..., 2] = 1 + 2.5 * fraction
                values = base[None] + rng.normal(0, 0.03, (n_mc, *base.shape))
            observed.create_dataset(quantity, data=values)
    return path


def test_end_to_end_mc_hdf5_and_truth_optional(tmp_path):
    source = _observation(tmp_path / "observations.h5")
    result = compute_shock_timing(source, {"modes": "both", "plot": {"enabled": False}})

    assert "shock_timing" in list_analyses()
    assert len(result["combinations"]) == 126
    for name in ("n", "Vx", "By", "Bz", "combined"):
        solution = result["solutions"][name]["ols"]
        assert np.all(solution["status"] == 0)
        np.testing.assert_allclose(
            solution["normal"], np.broadcast_to([-1, 0, 0], (2, 3)), atol=2.0e-2
        )
        np.testing.assert_allclose(solution["speed"], 100, atol=2.0)
    for name in ("Vy", "Vz", "Bx"):
        assert not np.any(result["independent"]["status"][result["observables"].index(name)] == 0)

    output = write_shock_timing_product(result, tmp_path / "timing.h5")
    with h5py.File(output) as handle:
        assert handle["detection/independent/status"].shape == (7, 2, 9)
        assert handle["reconstruction/combined/four_spacecraft/status"].shape == (2, 126)
        assert handle["reconstruction/combined/ols/q"].shape == (2, 3)
        assert handle["truth/arrival_times"].shape == (9,)
        assert bool(handle["metadata"].attrs["truth_available"])
        highlighted = handle["four_spacecraft_combinations/highlighted"]
        assert highlighted.attrs["selection_source"] == "combined"
        assert highlighted.attrs["selection_metric"] == "angular_error_rmse_deg"
        assert highlighted["best/indices"].shape == (4,)
        assert highlighted["worst/indices"].shape == (4,)
        assert highlighted["best"].attrs["rmse"] <= highlighted["worst"].attrs["rmse"]

    result["settings"]["plot"].update({
        "enabled": True,
        "detection_base": str(tmp_path / "detection"),
        "accuracy_base": str(tmp_path / "accuracy"),
        "geometry_base": str(tmp_path / "geometry"),
        "tetrahedra_geometry_base": str(tmp_path / "tetrahedra_geometry"),
        "dpi": 30,
    })
    write_shock_timing_product(result, tmp_path / "timing_with_plots.h5")
    for stem in ("detection", "accuracy", "geometry", "tetrahedra_geometry"):
        assert (tmp_path / f"{stem}.pdf").stat().st_size > 100
        assert (tmp_path / f"{stem}.png").stat().st_size > 100

    flight = compute_shock_timing(
        _observation(tmp_path / "flight.h5", n_mc=1, include_truth=False),
        {"modes": "combined", "truth_validation": "auto", "plot": {"enabled": False}},
    )
    assert flight["truth"] is None
    assert np.all(flight["solutions"]["combined"]["ols"]["status"] == 0)


def test_analysis_yaml_routes_through_existing_cli_workflow(tmp_path):
    source = _observation(tmp_path / "observations.h5", n_mc=1)
    output = tmp_path / "analysis.h5"
    config = tmp_path / "analysis.yaml"
    config.write_text(
        f"""
analysis:
  method: shock_timing
input:
  path: {source}
output:
  path: {output}
shock_timing:
  modes: combined
  observables: [n, Vx, By, Bz]
  truth_validation: required
  plot:
    enabled: false
""",
        encoding="utf-8",
    )

    assert analyze_from_config(config) == output
    with h5py.File(output) as handle:
        assert handle["metadata"].attrs["method"] == "shock_timing"
        assert handle["metadata"].attrs["analysis_config_path"] == str(config.resolve())
        assert handle["reconstruction/combined/ols/status"][0] == 0
