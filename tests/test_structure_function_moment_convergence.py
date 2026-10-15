"""Small synthetic tests only; never use production observations."""

import json

import h5py
import numpy as np
import pytest
import yaml

from hs_observatory.analysis import structure_function_moment_convergence as study
from hs_observatory.analysis.derived.geometry_pairs import build_geometry_pair_plan


def records(amplitudes, midpoints=None):
    result = np.zeros(len(amplitudes), dtype=study.RECORD_DTYPE)
    result["amplitude_km_s"] = amplitudes
    result["midpoint_s"] = np.arange(len(amplitudes)) if midpoints is None else midpoints
    return result


def test_moments_tails_concentration_and_cumulative():
    x = np.arange(1., 1001.)
    result = study.moment_diagnostics(records(x), (0, 1000))
    for i, n in enumerate(study.ORDERS):
        weights = x**n
        assert result["S_n"][i] == pytest.approx(weights.mean())
        assert result["N_conc"][i] == pytest.approx(weights.sum()**2 / np.sum(weights**2))
        for j, count in enumerate((1, 10, 50)):
            assert result["tail_contribution"][i, j] == pytest.approx(
                weights[-count:].sum()/weights.sum())
    normalized, cumulative = study.cumulative_distribution(x)
    np.testing.assert_allclose(normalized, x / np.sqrt(np.mean(x*x)))
    np.testing.assert_allclose(cumulative[:, -1], 1)
    np.testing.assert_allclose(cumulative[1], np.cumsum(x**4)/np.sum(x**4))
    assert result["hist_counts"].sum() == len(x)
    assert np.sum(result["hist_pdf"] * np.diff(result["hist_edges_km_s"])) == pytest.approx(1)
    _, tied = study.cumulative_distribution(np.array([1., 1., 2.]))
    np.testing.assert_allclose(tied[:, 0], tied[:, 1])


def test_numerical_stability_and_uniform_weights():
    result = study.moment_diagnostics(records([1e100, 2e100]), (0, 1))
    assert np.isfinite(result["log_S_n"]).all()
    assert np.isfinite(result["tail_contribution"]).all()
    assert result["N_conc"][2] == pytest.approx(65**2 / (1 + 64**2))
    uniform = study.moment_diagnostics(records(np.ones(20)), (0, 20))
    np.testing.assert_allclose(uniform["N_conc"], 20)
    np.testing.assert_allclose(uniform["tail_contribution"], .05)


def test_temporal_localization_and_canonical_lag_groups():
    data = records([1., 2., 10., 3.], [0., 29.99, 30., 61.])
    data["spacecraft_left"] = [0, 1, 0, 0]
    data["spacecraft_right"] = [1, 0, 1, 1]
    data["lag_samples"] = [2, -2, 2, 3]
    result = study.moment_diagnostics(data, (0, 61), block_width=30, top_count=1)
    expected = np.array([1 + 2**6, 10**6, 3**6], dtype=float)
    np.testing.assert_allclose(result["block_contribution"][2], expected / expected.sum())
    np.testing.assert_array_equal(result["block_extreme_counts"][0], [0, 1, 0])
    np.testing.assert_array_equal(result["group_keys"], [[0, 1, 2], [0, 1, 3]])
    np.testing.assert_array_equal(result["group_counts"], [3, 1])
    assert result["top_records"]["midpoint_s"][0] == 30
    assert result["top_block_contribution"][2, 0] == pytest.approx(expected.max()/expected.sum())
    assert result["top_block_contribution"][2, 1] == pytest.approx(1)


@pytest.mark.parametrize("amplitude", [[], [0.], [7.]])
def test_empty_zero_and_short_bins(amplitude):
    result = study.moment_diagnostics(records(amplitude), (0, 1), top_count=0)
    assert result["count"] == len(amplitude)
    assert result["group_keys"].shape[1] == 3
    if not amplitude:
        assert np.isnan(result["S_n"]).all()
        assert result["block_counts"].sum() == 0
    elif amplitude[0] == 0:
        np.testing.assert_array_equal(result["S_n"], 0)
        assert np.isnan(result["tail_contribution"]).all()
    else:
        np.testing.assert_allclose(result["tail_contribution"], 1)
        np.testing.assert_allclose(result["N_conc"], 1)


def test_sampler_census_matches_bruteforce_and_partial_prefixes():
    times = np.arange(7.)
    positions = np.zeros((7, 2, 3))
    positions[:, :, 0] = 100*times[:, None] + [0, 80]
    z_plus = np.zeros_like(positions)
    z_plus[:, :, 0] = times[:, None]**2 + [0, 3]
    edges = np.array([70., 250., 700., 800.])
    plan = build_geometry_pair_plan(positions, edges)
    direct = []
    flat_r, flat_z = positions.reshape(-1, 3), z_plus.reshape(-1, 3)
    for a in range(len(flat_r)):
        for b in range(a+1, len(flat_r)):
            separation = np.linalg.norm(flat_r[b]-flat_r[a])
            if edges[0] <= separation < edges[1]:
                direct.append(np.linalg.norm(flat_z[b]-flat_z[a]))
    census = study.sample_records(plan, positions, z_plus, times, 0, 10000, 1, 3)
    assert len(census) == plan.populations[0] == len(direct)
    np.testing.assert_allclose(np.sort(census["amplitude_km_s"]), np.sort(direct))
    for n in (2, 4, 6):
        result = study.moment_diagnostics(census, (0, 6))
        assert result["S_n"][list(study.ORDERS).index(n)] == pytest.approx(np.mean(np.array(direct)**n))
    small = study.sample_records(plan, positions, z_plus, times, 0, 5, 21, 2)
    larger = study.sample_records(plan, positions, z_plus, times, 0, 10, 21, 4)
    np.testing.assert_array_equal(small, larger[:5])
    identities = np.column_stack([larger[name] for name in
                                 ("spacecraft_left", "spacecraft_right", "time_left_s", "time_right_s")])
    assert len(np.unique(identities, axis=0)) == len(larger)
    assert not len(study.sample_records(plan, positions, z_plus, times, 2, 10, 1, 2))


def test_saved_results_and_fingerprint(tmp_path):
    result = study.moment_diagnostics(records([1., 2., 3.]), (0, 3))
    path = tmp_path / "cache.h5"
    metadata = {"case": "synthetic"}
    metadata["fingerprint"] = study.fingerprint(metadata)
    study.save_product(path, metadata, result)
    saved_metadata, saved = study.read_product(path, metadata["fingerprint"])
    assert saved_metadata == metadata
    for key in result:
        np.testing.assert_array_equal(saved[key], result[key])
    np.testing.assert_array_equal(study.cumulative_distribution(saved["sorted_amplitude_km_s"])[1],
                                  study.cumulative_distribution(np.array([1., 2., 3.]))[1])
    with pytest.raises(ValueError, match="incompatible"):
        study.read_product(path, "different")
    assert study.fingerprint({"block_width": 30}) != study.fingerprint({"block_width": 60})
    assert "sorted_amplitude_km_s" not in study.read_product(path, distribution=False)[1]


def make_input(tmp_path):
    (tmp_path / "generation").mkdir()
    config = dict(output=dict(path="synthetic.h5"), errors=dict(n_mc=0),
                  field=dict(turbulence=dict(intermittency_p=.9)))
    generation = tmp_path / "generation/synthetic-t6.yaml"
    generation.write_text(yaml.safe_dump(config))
    source = tmp_path / "synthetic.h5"
    time = np.arange(7.)
    positions = np.zeros((2, 7, 3))
    positions[:, :, 0] = 100*time[None] + np.array([0, 80])[:, None]
    with h5py.File(source, "w") as h:
        h.create_group("metadata").attrs["config_yaml"] = yaml.safe_dump(config)
        h.create_dataset("spacecraft/id", data=np.array(["S0", "S1"], dtype=h5py.string_dtype()))
        for variable in ("n", "v", "B"):
            h.create_dataset(f"time/{variable}", data=time)
            h.create_dataset(f"spacecraft/r/{variable}", data=positions)
            values = np.full((7, 2), 5.) if variable == "n" else np.zeros((7, 2, 3))
            if variable == "v":
                values[:, :, 0] = time[:, None]**2 * [[1., 2.]]
            h.create_dataset(f"truth/{variable}", data=values)
        # There are deliberately no observed/error datasets.
    return source, generation


def test_cli_preflight_resume_and_plot_only_without_inputs(tmp_path, monkeypatch, capsys):
    source, generation = make_input(tmp_path)
    output = tmp_path / "diagnostics"
    common = ["--root", str(tmp_path), "--cases", "synthetic", "--duration", "6",
              "--separations", "300", "--budgets", "5", "10", "--seeds", "7", "8",
              "--block-width", "2", "--output", str(output)]
    study.main(common + ["--dry-run"])
    assert "eligible=" in capsys.readouterr().out
    assert not output.exists()
    # Exercise actual plotting once, after deleting input sources below.
    original_plot = study.plot_saved
    monkeypatch.setattr(study, "plot_saved", lambda *args: None)
    study.main(common)
    paths = sorted(output.glob("*.h5"))
    assert len(paths) == 4
    mtimes = [path.stat().st_mtime_ns for path in paths]
    monkeypatch.setattr(study, "sample_records", lambda *args: pytest.fail("unexpected resampling"))
    study.main(common + ["--resume"])
    assert mtimes == [path.stat().st_mtime_ns for path in paths]
    source.unlink()
    generation.unlink()
    monkeypatch.setattr(study, "plot_saved", original_plot)
    study.main(["--output", str(output), "--plot-only"])
    for name in ("A_cumulative", "B_tail_concentration", "C_temporal", "D_budget_convergence"):
        for extension in ("png", "pdf"):
            assert (output / f"figures/{name}.{extension}").stat().st_size > 0
    summary = json.loads((output / "figures/summary.json").read_text())
    assert len(summary["rows"]) == 4
    assert "correlation-corrected" in summary["caution"]
    for row in summary["rows"]:
        if row["budget"] == 10:
            np.testing.assert_allclose(row["relative_to_largest_budget"], 0)


def test_selection_and_input_validation(tmp_path):
    bins = study.select_bins()
    assert len(bins) == 5
    assert study.select_bins([500], [0]) == sorted({0, study.select_bins([500])[0]})
    with pytest.raises(ValueError):
        study.select_bins([np.nan])
    with pytest.raises(ValueError):
        study.select_bins(bins=[40])
    for arguments in (["--budgets", "0"], ["--block-width", "nan"], ["--seeds", "1", "1"]):
        with pytest.raises(SystemExit):
            study.main(arguments)
    source, generation = make_input(tmp_path)
    config = yaml.safe_load(generation.read_text())
    config["errors"]["n_mc"] = 2
    generation.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="n_mc"):
        study.prepare_case(tmp_path, "synthetic", 6)
    config["errors"]["n_mc"] = 0
    config["field"]["turbulence"]["intermittency_p"] = .5
    generation.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="differs"):
        study.prepare_case(tmp_path, "synthetic", 6)
    assert source.exists()
