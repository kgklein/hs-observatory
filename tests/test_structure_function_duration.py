"""Small saved-truth tests; never generate or analyze production fields."""

import json

import h5py
import numpy as np
import pytest
import yaml

import hs_observatory.analysis.structure_function_duration as study


def inputs(tmp_path, durations=(1.0, 2.0)):
    root = tmp_path
    (root / "generation").mkdir()
    (root / "data").mkdir()
    # Keep synthetic saved-truth tests independent of production recipes/budgets.
    config = {
        "analysis": {"method": "structure_function_duration_convergence"},
        "output": {"directory": "results", "report": "report.md"},
        "duration_study": {
            "generation_glob": "generation/malara-ts-502-t*.yaml",
            "durations_s": list(durations),
            "seeds": [12345, 23456, 34567, 45678, 56789],
            "fit_pairs": 8,
            "outside_pairs": 4,
            "peak_interval_km": [700.0, 1500.0],
            "coverage_cells_per_axis": 64,
        },
        "structure_function": {
            "increment_mode": "all_spacetime_pairs",
            "min_pair_lambda_km": 70.0,
            "max_pair_lambda_km": 20000.0,
            "pair_sampling": {
                "mode": "geometry_stratified",
                "chunk_size": 100000,
                "target_pairs_per_bin": 8,
            },
            "min_order": 1,
            "max_order": 6,
            "lambda_bins_km": np.geomspace(70.0, 20000.0, 41).tolist(),
            "scale_ranges_km": {"primary": [150.0, 1200.0], "sensitivity": [200.0, 1000.0]},
            "min_fit_bins": 3,
            "elsasser": {
                "velocity": {"frame": "auto", "spacecraft_correction": "auto"},
                "density_normalization": {"method": "global_per_spacecraft", "species": "proton"},
                "mean_subtraction": {"method": "global_per_spacecraft"},
                "timebase": {"method": "slower_of_n_and_v", "interpolation": "linear"},
            },
            "plot": {"enabled": False, "publication": {"enabled": False}},
        },
    }
    for duration in durations:
        times = np.arange(round(duration * 8) + 1) / 8
        starts = np.array([[0.0, 0.0, 0.0], [90.0, 130.0, 0.0], [300.0, 350.0, 0.0]])
        positions = starts[None] + times[:, None, None] * [105.0, 2.0, 245.0]
        generation = {
            "time": {"t_start_s": 0.0, "t_stop_s": duration, "cadence_s": 0.125},
            "field": {
                "model": "malara_turbulence",
                "domain": {"L": [100000.0] * 3, "units": "km"},
                "turbulence": {
                    "intermittency_p": 0.5,
                    "base_seed": 12345,
                    "realization": 1,
                    "n_scales": 14,
                    "injection_scale_index": 2,
                },
            },
            "output": {"path": f"data/duration-{duration:g}.observation"},
            "spacecraft": {"velocity_km_s": [105.0, 2.0, 245.0]},
            "measurement": {"velocity_frame": "spacecraft"},
            "errors": {"n_mc": 0},
        }
        recipe = root / "generation" / f"malara-ts-502-t{duration:g}.yaml"
        recipe.write_text(yaml.safe_dump(generation))
        with h5py.File(root / generation["output"]["path"], "w") as h:
            h.create_group("metadata").attrs.update(
                config_yaml=yaml.safe_dump(generation),
                n_mc=0,
                velocity_truth_frame="plasma",
                velocity_observed_frame="spacecraft",
            )
            h.create_dataset("spacecraft/id", data=np.array(["A", "B", "C"], dtype="S"))
            for q in ("n", "v", "B"):
                h.create_dataset(f"time/{q}", data=times)
                h.create_dataset(f"spacecraft/r/{q}", data=positions.swapaxes(0, 1))
                if q == "n":
                    values = np.full(positions.shape[:2], 5.0)
                else:
                    background = [400.0, 0.0, 0.0] if q == "v" else [0.0, 0.0, 5.0]
                    values = np.array(background) + np.sin(positions / 200.0)
                h.create_dataset(f"truth/{q}", data=values)
            # Deliberately omit /observed entirely: no MC members may be required.
    return config


def test_truth_only_discovery_and_exact_prefix(tmp_path):
    config = inputs(tmp_path)
    validation = study.validate_inputs(config, tmp_path, [1.0, 2.0])
    assert validation["valid_nested_sequence"]
    for record in validation["inputs"].values():
        assert record["exact_prefix"]
        assert record["n_mc_in_file"] == 0
        assert record["source"].endswith(".observation")
        assert all(c["max_abs_difference"] == 0 for c in record["prefix_comparisons"].values())


@pytest.mark.parametrize("change", ["truth", "position", "time", "realization"])
def test_non_nested_inputs_flagged(tmp_path, change):
    config = inputs(tmp_path)
    with h5py.File(tmp_path / "data/duration-1.observation", "r+") as h:
        if change == "realization":
            generation = yaml.safe_load(h["metadata"].attrs["config_yaml"])
            generation["field"]["turbulence"]["realization"] = 2
            h["metadata"].attrs["config_yaml"] = yaml.safe_dump(generation)
        elif change == "truth":
            h["truth/B"][0, 0, 0] += 1
        elif change == "time":
            for q in ("n", "v", "B"):
                h[f"time/{q}"][...] += 0.1
        else:
            for q in ("n", "v", "B"):
                h[f"spacecraft/r/{q}"][...] += [0.0, 5.0, 0.0]
    validation = study.validate_inputs(config, tmp_path, [1.0, 2.0])
    assert not validation["valid_nested_sequence"]
    assert validation["mismatched_durations"] == [1.0]


def test_seed_aggregation_and_peak_interval():
    summary = study.seed_summary(np.array([[1.0, 4.0, np.nan], [3.0, 6.0, np.nan]]))
    np.testing.assert_allclose(summary["mean"][:2], [2.0, 5.0])
    np.testing.assert_allclose(summary["std"][:2], np.sqrt(2))
    assert np.isnan(summary["mean"][2])
    assert np.isnan(study.seed_summary(np.array([[1.0]]))["std"][0])
    x = np.array([500.0, 750.0, 1000.0, 1400.0, 1800.0])
    slopes = np.array([[[99.0, 2.0, 4.0, 3.0, 100.0]], [[0.0, 3.0, 2.0, 1.0, 0.0]]])
    peaks = study.local_peaks(x, slopes, [700.0, 1500.0])
    np.testing.assert_array_equal(peaks["amplitude"][:, 0], [4.0, 3.0])
    np.testing.assert_array_equal(peaks["separation_km"][:, 0], [1000.0, 750.0])


def test_periodic_coverage_is_not_unwrapped_distance():
    positions = np.array([[[0.0, 0.0, 0.0]], [[1000.0, 0.0, 0.0]], [[2000.0, 0.0, 0.0]]])
    coverage = study.coverage_diagnostics(
        positions, {"field": {"domain": {"L": [1000.0] * 3}}}, cells=4
    )
    assert coverage["distance_traveled_km_per_sc"][0] == 2000
    assert coverage["all_samples"]["unique_voxels"] == 1
    np.testing.assert_allclose(coverage["all_samples"]["repeat_fraction"], 2 / 3)


def test_preflight_never_samples(tmp_path, monkeypatch):
    config = inputs(tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.chdir(tmp_path)

    def fail(*args, **kwargs):
        raise AssertionError("preflight cannot sample")

    monkeypatch.setattr(study, "sample", fail)
    assert study.main(["--config", str(path), "--dry-run"]) == 0
    assert (tmp_path / "results/validation.json").is_file()
    assert not (tmp_path / "results/moments").exists()


def test_cached_moments_masks_budgets_and_plot_only(tmp_path, monkeypatch):
    config = inputs(tmp_path)
    config["duration_study"]["verification"] = {
        "durations_s": [2.0],
        "fit_pairs": 16,
        "outside_pairs": 4,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.chdir(tmp_path)
    args = ["--config", str(path), "--seed-count", "2", "--verify-budget"]
    assert study.main([*args, "--mode", "sample"]) == 0
    output = tmp_path / "results"
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["records"]) == 6
    for record in manifest["records"]:
        saved = json.loads((output / record["path"]).read_text())
        diag = saved["product"]["spacetime_pair_counts"]["geometry_sampling"]["members"][0]
        x = np.array(saved["product"]["lambda_bin_centers_km"])
        budget = 8 if record["budget"] == "baseline" else 16
        expected = np.where((x >= 150) & (x <= 1200), budget, 4)
        np.testing.assert_array_equal(diag["requested_per_bin"], expected)
        np.testing.assert_array_equal(
            saved["product"]["counts"][0], np.minimum(expected, diag["eligible_per_bin"])
        )
        np.testing.assert_allclose(
            np.diff(np.log(saved["product"]["lambda_bin_edges_km"])), np.log(20000 / 70) / 40
        )

    def fail(*args, **kwargs):
        raise AssertionError("cached operation cannot access inputs or evaluate increments")

    monkeypatch.setattr(study, "compute_all_spacetime_structure_functions", fail)
    # Resume only opens valid cached seed moments.
    assert study.main([*args, "--mode", "sample"]) == 0
    # Config-only fit-window changes must not invalidate moment caches.
    config["structure_function"]["scale_ranges_km"]["sensitivity"] = [220.0, 950.0]
    path.write_text(yaml.safe_dump(config))
    assert study.main([*args, "--mode", "sample"]) == 0
    for source in (tmp_path / "data").iterdir():
        source.unlink()
    monkeypatch.setattr(study, "validate_inputs", fail)
    assert (
        study.main(["--config", str(path), "--plot-only", "--sensitivity-fit", "220", "950"]) == 0
    )
    result = json.loads((output / "derived.json").read_text())
    assert len(result["groups"]) == 3
    higher = next(g for g in result["groups"] if g["budget"] == "verification")
    assert "delta_baseline_budget" in higher["fields"]["z_plus"]
    for group in result["groups"]:
        x = np.array(group["centers_km"])
        f = group["fields"]["z_plus"]
        for w, limits in enumerate(result["windows"].values()):
            expected = np.sum(
                (x >= limits[0]) & (x <= limits[1]) & (np.array(group["counts"][0]) > 0)
            )
            assert f["fits"]["bins_used"][0][w][0] == expected
    assert len(list(output.glob("*.png"))) == 4
    assert len(list(output.glob("*.pdf"))) == 4
    assert (tmp_path / "report.md").is_file()
    assert "p=0.5" in (tmp_path / "report.md").read_text()
    assert (tmp_path / "report.csv").is_file()


def test_smoke_output_override_and_non_nested_refusal(tmp_path, monkeypatch):
    config = inputs(tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.chdir(tmp_path)
    assert (
        study.main(
            [
                "--config",
                str(path),
                "--dry-run",
                "--generation-glob",
                "generation/malara-ts-502-t*.yaml",
                "--output-dir",
                "other-study",
            ]
        )
        == 0
    )
    assert (tmp_path / "other-study/validation.json").is_file()
    with h5py.File(tmp_path / "data/duration-1.observation", "r+") as h:
        h["truth/B"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="INPUT MISMATCH"):
        study.main(["--config", str(path), "--mode", "sample"])
    assert not (tmp_path / "results/manifest.json").exists()


def test_one_seed_single_duration_smoke(tmp_path, monkeypatch):
    config = inputs(tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.chdir(tmp_path)
    assert (
        study.main(
            [
                "--config",
                str(path),
                "--durations",
                "1",
                "--seed-count",
                "1",
                "--output-dir",
                "smoke",
            ]
        )
        == 0
    )
    result = json.loads((tmp_path / "smoke/derived.json").read_text())
    assert len(result["groups"]) == 1
    assert result["groups"][0]["seeds"] == [12345]
    assert np.all(np.isnan(result["groups"][0]["fields"]["z_plus"]["xi_summary"]["std"]))
    assert (tmp_path / "smoke/report.md").is_file()
    assert len(list((tmp_path / "smoke").glob("*.png"))) == 4
