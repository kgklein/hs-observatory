import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from hs_observatory.analysis.derived.geometry_pairs import build_geometry_pair_plan
from hs_observatory.analysis.structure_function import (
    FIELDS,
    compute_all_spacetime_structure_functions,
    structure_function_moment_means,
)
from hs_observatory.analysis.structure_function_convergence import (
    fit_saved,
    prefix_moments,
    run_study,
    stage2_edges,
)


def test_explicit_bin_design():
    edges = stage2_edges()
    assert len(edges) == 27
    np.testing.assert_array_equal(edges[[0, 4, 18, 26]], [70, 150, 1200, 20000])
    assert np.all(np.diff(edges) > 0)


def test_prefixes_equal_separate_fixed_budget_calls_including_census():
    p = np.array([[[0.0, 0.0, 0.0], [100.0, 200.0, 0.0]]]) + np.arange(12)[:, None, None] * [
        100.0,
        0.0,
        0.0,
    ]
    edges = [70.0, 150.0, 300.0, 600.0, 1200.0, 20000.0]
    plan = build_geometry_pair_plan(p, edges, min_pair_lambda_km=70.0, max_pair_lambda_km=20000.0)
    rng = np.random.default_rng(19)
    fields = {name: rng.normal(size=p.shape) for name in FIELDS}
    budgets = [3, 7, 500]
    # Largest budget exceeds populations: smaller means must not be based on
    # the Stage 1 sampler's deterministic census prefixes.
    for chunk in [1, 11, 1000]:
        sums, counts, requested, _ = prefix_moments(plan, p, fields, budgets, 4, 123, chunk)
        for b in range(len(budgets)):
            reference = compute_all_spacetime_structure_functions(
                p[None],
                fields["z_plus"][None],
                fields["z_minus"][None],
                list(range(1, 7)),
                edges,
                pair_sampling=dict(
                    mode="geometry_stratified",
                    target_pairs_per_bin=requested[b],
                    random_seed=123,
                    chunk_size=13,
                    max_pairs=None,
                ),
                min_pair_lambda_km=70.0,
                max_pair_lambda_km=20000.0,
            )
            np.testing.assert_array_equal(counts[b], reference["counts"][0])
            for f, name in enumerate(FIELDS):
                means = structure_function_moment_means(sums[:, f], counts)
                np.testing.assert_allclose(means[b], reference[name]["S_n"][0], equal_nan=True)


def test_refit_saved_power_law_and_local_slopes():
    centers = np.sqrt(stage2_edges()[:-1] * stage2_edges()[1:])
    values = np.array([centers ** (n / 3) for n in range(1, 7)])[None, None]
    counts = np.ones((1, 1, len(centers)), dtype=int)
    exponents, slopes = fit_saved(values, counts, centers)
    np.testing.assert_allclose(exponents[0, 0], np.tile(np.arange(1, 7) / 3, (2, 1)))
    np.testing.assert_allclose(
        slopes[0, 0, :, 1:-1], np.broadcast_to(np.arange(1, 7)[:, None] / 3, (6, 24))
    )


def test_driver_selects_only_truth_and_saves_refittable_arrays(tmp_path, monkeypatch):
    import hs_observatory.analysis.structure_function_convergence as module

    source = tmp_path / "observations.h5"
    times = np.linspace(0, 300, 2401)
    p = np.array([[[0.0, 0.0, 0.0], [100.0, 200.0, 0.0]]]) + np.arange(2401)[:, None, None] * [
        100.0,
        0.0,
        0.0,
    ]
    with h5py.File(source, "w") as h:
        h.create_group("metadata").attrs["config_yaml"] = yaml.safe_dump(
            dict(field=dict(turbulence=dict(intermittency_p=0.9)))
        )
        h.create_dataset("time/v", data=times)
        h.create_dataset("spacecraft/r/v", data=p.swapaxes(0, 1))

    calls = []

    def truth_only(path, config):
        calls.append(config["data_group"])
        assert config["data_group"] == "truth"
        return dict(time=times, spacecraft_ids=["A", "B"], z_plus=np.sin(p), z_minus=np.cos(p))

    monkeypatch.setattr(module, "compute_elsasser_analysis", truth_only)
    output = tmp_path / "convergence.h5"
    recipe = dict(
        input=dict(path=str(source)),
        output=dict(path=str(output)),
        analysis=dict(method="structure_function_truth_convergence"),
        structure_function=dict(
            increment_mode="all_spacetime_pairs",
            lambda_bins_km=stage2_edges().tolist(),
            scale_ranges_km=dict(primary=[150.0, 1200.0], sensitivity=[200.0, 1000.0]),
            pair_sampling=dict(mode="geometry_stratified", target_pairs_per_bin=4),
        ),
        stage2=dict(
            data_group="truth",
            run_id=400,
            p_value=0.9,
            seeds=[1, 2, 3, 4, 5],
            budgets=[1, 2, 3, 4],
            outside_target=2,
        ),
    )
    config = tmp_path / "pilot.yaml"
    config.write_text(yaml.safe_dump(recipe))
    run_study(config)
    assert calls == ["truth"]
    with h5py.File(output) as h:
        assert h["z_plus/S_n"].shape == (5, 4, 6, 26)
        assert h["z_plus/xi_n"].shape == (5, 4, 2, 6)
        assert h["z_plus/local_slopes"].shape == (5, 4, 6, 26)
        assert json.loads(h.attrs["metadata_json"])["data_group"] == "truth"
    reference = tmp_path / "reference.h5"
    with h5py.File(reference, "w") as h:
        h.create_dataset("time/v", data=times)
        h.create_dataset("spacecraft/r/v", data=(p + [1.0, 0.0, 0.0]).swapaxes(0, 1))
    recipe["stage2"]["geometry_reference"] = str(reference)
    config.write_text(yaml.safe_dump(recipe))
    with pytest.raises(ValueError, match="trajectory differs"):
        run_study(config)


def test_pilot_recipes_use_explicit_edges_and_windows():
    from hs_observatory.analysis.structure_function import _settings

    root = Path(__file__).resolve().parents[1]
    for run in [400, 401, 402]:
        recipe = yaml.safe_load(
            (root / f"analysis/malara-prod-{run}_sf-stage2-pilot.yaml").read_text()
        )
        assert recipe["analysis"]["method"] == "structure_function_truth_convergence"
        assert recipe["stage2"]["data_group"] == "truth"
        assert len(recipe["stage2"]["seeds"]) >= 5
        settings = _settings(recipe["structure_function"])
        np.testing.assert_array_equal(settings["lambda_bins_km"], stage2_edges())
        assert settings["scale_ranges_km"] == {
            "primary": [150.0, 1200.0],
            "sensitivity": [200.0, 1000.0],
        }
        assert len(settings["pair_sampling"]["target_pairs_per_bin"]) == 26
