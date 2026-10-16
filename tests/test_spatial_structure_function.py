"""Small deterministic checks of direct sampling and cached statistical products."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pytest
import yaml

matplotlib.use("Agg")

from hs_observatory.analysis import spatial_structure_function as spatial  # noqa: E402
from hs_observatory.analysis import structure_function as sf  # noqa: E402
from hs_observatory.analysis import structure_function_local as local  # noqa: E402
from hs_observatory.analyze import analyze_from_config  # noqa: E402
from hs_observatory.cli_spatial_structure_function import plot_main, sample_main  # noqa: E402
from hs_observatory.fields.malara import MalaraConfig, MalaraTurbulence  # noqa: E402


def analytic_field(points):
    return np.sin(np.asarray(points) * 2 * np.pi / 64)


class AnalyticEvaluator:
    calls = 0
    clears = 0

    def __init__(self, config):
        self.config = config

    def evaluate_vector(self, points):
        type(self).calls += 1
        return analytic_field(points)

    evaluate_vector_cached = evaluate_vector

    def clear_cache(self):
        type(self).clears += 1


def configuration(tmp_path):
    field = {
        "model": "malara_turbulence",
        "domain": {"L": [64, 64, 64], "units": "km"},
        "turbulence": {
            "n_scales": 2,
            "injection_scale_index": 1,
            "spectral_exponent_h": 1 / 3,
            "base_seed": 17,
            "realization": 3,
            "backend": "cached",
            "intermittency_p": 0.7,
        },
        "output_fields": {
            "n": {"background": 5, "fluctuation": "constant"},
            "v": {"background": [0, 0, 0], "fluctuation": "malara", "amplitude": 10},
            "B": {"background": [0, 0, 0], "fluctuation": "malara", "amplitude": 0.5},
        },
    }
    generation = tmp_path / "field.yaml"
    generation.write_text(yaml.safe_dump({"field": field}))
    options = {
        "n_pairs": 16,
        "n_batches": 4,
        "p_values": [0.5, 0.7, 0.9],
        "seed": 91,
        "chunk_size": 3,
        "lambda_centers_km": [1, 2, 4, 8, 16],
        "scale_ranges_km": {"candidate": [1, 8], "comparison": [2, 16]},
        "fit_range": "candidate",
        "plot": {"enabled": False, "dpi": 35},
    }
    cfg = {
        "input": {"path": str(generation)},
        "analysis": {"method": "spatial_structure_function"},
        "output": {"path": str(tmp_path / "spatial.h5")},
        "spatial_structure_function": options,
    }
    path = tmp_path / "analysis.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path, cfg


def test_pair_magnitudes_isotropy_and_uniform_positions():
    left, right = spatial.sample_spatial_pairs(
        spatial.spatial_pair_rng(32, 7), 30000, (64, 64, 64), 7
    )
    directions = (right - left) / 7
    np.testing.assert_allclose(np.linalg.norm(right - left, axis=1), 7, atol=1e-14)
    assert np.all((left >= 0) & (left < 64))
    np.testing.assert_allclose(left.mean(axis=0), 32, atol=0.3)
    np.testing.assert_allclose(directions.mean(axis=0), 0, atol=0.015)
    np.testing.assert_allclose(directions.T @ directions / len(left), np.eye(3) / 3, atol=0.01)


def test_pair_prefix_independent_of_chunk_size_and_p():
    domain = (64, 64, 64)
    a = spatial.sample_spatial_pairs(spatial.spatial_pair_rng(15, 3), 5000, domain, 3)
    b = spatial.sample_spatial_pairs(spatial.spatial_pair_rng(15, 3), 2000, domain, 3)
    rng = spatial.spatial_pair_rng(15, 3)
    chunks = [spatial.sample_spatial_pairs(rng, n, domain, 3) for n in [7, 50, 1943]]
    for i in range(2):
        np.testing.assert_array_equal(a[i][:2000], b[i])
        np.testing.assert_array_equal(np.concatenate([pair[i] for pair in chunks]), b[i])
    visited = []
    for p in [0.5, 0.7, 0.9]:
        points = []

        def evaluate(x):
            points.append(x.copy())
            return analytic_field(x)

        spatial.sample_case(
            MalaraConfig(domain_size=domain, intermittency_p=p),
            [3],
            n_pairs=16,
            n_batches=4,
            seed=15,
            chunk_size=3,
            evaluator=evaluate,
        )
        visited.append(np.concatenate(points))
    for points in visited[1:]:
        np.testing.assert_array_equal(points, visited[0])


def test_periodic_wrapping_with_unwrapped_pairs():
    model = MalaraTurbulence(
        MalaraConfig(domain_size=(64, 64, 64), n_scales=2, injection_scale_index=1)
    )
    points = np.array([[65, -2, 130], [63, 64.5, 1.0]])
    np.testing.assert_allclose(
        model.evaluate_vector_cached(points), model.evaluate_vector(points % 64), atol=1e-12
    )


def test_moments_batches_checkpoints_and_chunk_invariance():
    config = MalaraConfig(domain_size=(64, 64, 64))
    centers = np.array([1.0, 3.0, 7.0])
    case = spatial.sample_case(
        config, centers, n_pairs=17, n_batches=4, seed=8, chunk_size=3, evaluator=analytic_field
    )
    other = spatial.sample_case(
        config, centers, n_pairs=17, n_batches=4, seed=8, chunk_size=100, evaluator=analytic_field
    )
    np.testing.assert_allclose(case["q"]["S_n"], other["q"]["S_n"], rtol=1e-14)
    for si, separation in enumerate(centers):
        left, right = spatial.sample_spatial_pairs(
            spatial.spatial_pair_rng(8, separation), 17, config.domain_size, separation
        )
        magnitudes = np.linalg.norm(analytic_field(right) - analytic_field(left), axis=1)
        expected = [np.mean(magnitudes**n) for n in spatial.ORDERS]
        np.testing.assert_allclose(case["q"]["S_n"][0, :, si], expected, rtol=1e-14)
        start = 0
        for bi, count in enumerate(case["batch_counts"][:, si]):
            sample = magnitudes[start : start + count]
            np.testing.assert_allclose(
                case["batch_S_n"][bi, :, si], [np.mean(sample**n) for n in spatial.ORDERS]
            )
            start += count
        assert start == 17
        for ci, count in enumerate(case["checkpoint_counts"][:, si]):
            np.testing.assert_allclose(
                case["checkpoint_S_n"][ci, :, si],
                [np.mean(magnitudes[:count] ** n) for n in spatial.ORDERS],
            )
    np.testing.assert_allclose(case["batch_sd"], np.std(case["batch_S_n"], axis=0, ddof=1))
    np.testing.assert_array_equal(case["batch_counts"].sum(axis=0), case["counts"][0])
    np.testing.assert_allclose(case["batch_moment_sums"].sum(axis=0), case["moment_sums"][0])


def test_bounded_parallel_queue_and_order(monkeypatch):
    live = 0
    peak = 0

    class Future:
        def __init__(self, task):
            self.task = task

        def result(self):
            nonlocal live
            live -= 1
            return self.task

    class Executor:
        def submit(self, fn, task):
            nonlocal live, peak
            live += 1
            peak = max(peak, live)
            return Future(task)

    results = list(spatial._bounded_results(Executor(), ((i, i) for i in range(100)), 4))
    assert results == [(i, i) for i in range(100)]
    assert peak == 4 and live == 0


def test_parallel_chunk_matches_serial():
    config = MalaraConfig(domain_size=(64, 64, 64), n_scales=2, injection_scale_index=1)
    kwargs = dict(n_pairs=4, n_batches=2, seed=10, chunk_size=2)
    serial = spatial.sample_case(config, [1, 4, 8], **kwargs)
    with ThreadPoolExecutor(max_workers=2) as pool:
        parallel = spatial.sample_case(
            config, [1, 4, 8], executor=pool, n_workers=2, backend="parallel", **kwargs
        )
    np.testing.assert_array_equal(serial["q"]["S_n"], parallel["q"]["S_n"])


def test_sampling_reuses_field_configuration_and_avoids_observatory(tmp_path, monkeypatch):
    _, cfg = configuration(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("observatory workflow called")

    import hs_observatory.generate as generate
    import hs_observatory.fields.malara as malara

    monkeypatch.setattr(generate, "generate_from_config", forbidden)
    monkeypatch.setattr(generate, "_spacecraft_on_time_axes", forbidden)
    monkeypatch.setattr(generate, "error_model_from_config", forbidden)
    monkeypatch.setattr(generate, "apply_observatory_measurement_errors", forbidden)
    monkeypatch.setattr(malara.MalaraTurbulenceField, "evaluate_quantity", forbidden)
    monkeypatch.setattr(sf, "compute_elsasser_analysis", forbidden)
    monkeypatch.setattr(spatial, "MalaraTurbulence", AnalyticEvaluator)
    AnalyticEvaluator.calls = AnalyticEvaluator.clears = 0
    method = spatial.SpatialStructureFunctionAnalysis(cfg["spatial_structure_function"])
    result = method.run(cfg["input"]["path"])
    assert AnalyticEvaluator.calls == AnalyticEvaluator.clears > 0
    for case in result["cases"]:
        field = case["field_config"]
        assert field["amplitude"] == pytest.approx(
            64
        )  # adapter default; not 1 or output-field amplitude
        assert field["base_seed"] == 17 and field["realization"] == 3
        assert field["epsilon_exponent"] == pytest.approx(1 / 3)
        np.testing.assert_array_equal(case["counts"], [[16] * 5])
    output = method.write_result(result, cfg["output"]["path"])
    loaded = spatial.read_spatial_result(output)
    assert loaded["metadata"]["separation_mode"] == "exact"
    assert "lambda_bin_edges_km" not in loaded["cases"][0]
    assert "z_plus" not in loaded["cases"][0]
    with h5py.File(output) as handle:
        assert handle["cases/0/local/slope_valid"].dtype == np.dtype(bool)


def test_global_local_analysis_uses_shared_functions(monkeypatch):
    centers = np.array([1.0, 2.0, 3.0, 7.0, 12.0])
    exponents = np.arange(1, 7) / 3
    values = 5 * centers[None, None, :] ** exponents[None, :, None]
    case = {
        "p_value": 0.5,
        "orders": np.arange(1, 7),
        "lambda_bin_centers_km": centers,
        "q": {"S_n": values},
        "counts": np.ones((1, 5)),
        "batch_S_n": np.repeat(values, 4, axis=0),
        "batch_counts": np.ones((4, 5)),
    }
    called = []
    original = spatial._fit_scaling_exponents

    def traced(*args, **kwargs):
        called.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(spatial, "_fit_scaling_exponents", traced)
    for method in ("three-point", "adjacent"):
        spatial.derive_spatial_case(case, {"all": [1, 12]}, method=method)
        fit = case["q"]["scaling_exponents"]
        np.testing.assert_allclose(fit["raw"][0, 0], exponents)
        np.testing.assert_allclose(fit["r_squared"], 1)
        np.testing.assert_allclose(fit["log_rms"], 0, atol=1e-14)
        np.testing.assert_allclose(fit["intercept"], np.log(5))
        coords, expected, valid = local.local_log_slopes(centers, values, method=method)
        np.testing.assert_allclose(case["local"]["slopes"], expected, equal_nan=True)
        np.testing.assert_array_equal(case["local"]["coordinates_km"], coords)
        np.testing.assert_array_equal(case["local"]["source_valid"], valid)
    assert len(called) == 2


def test_cached_plotting_never_constructs_field_or_changes_source(tmp_path, monkeypatch):
    cfg_path, cfg = configuration(tmp_path)
    monkeypatch.setattr(spatial, "MalaraTurbulence", AnalyticEvaluator)
    output = analyze_from_config(cfg_path)
    original_bytes = output.read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("field sampling invoked during cached post-processing")

    monkeypatch.setattr(spatial, "MalaraTurbulence", forbidden)
    monkeypatch.setattr(spatial, "MalaraTurbulenceField", forbidden)
    monkeypatch.setattr(spatial, "sample_case", forbidden)
    monkeypatch.setattr(spatial, "_evaluate_malara_chunk", forbidden)
    prefix = tmp_path / "refit"
    assert (
        plot_main(
            [
                str(output),
                "--output-prefix",
                str(prefix),
                "--fit-lambda-range",
                "2",
                "16",
                "--fit-window",
                "small",
                "1",
                "4",
                "--local-slope-method",
                "adjacent",
                "--show-cascade-scales",
            ]
        )
        == 0
    )
    assert output.read_bytes() == original_bytes
    result = spatial.read_spatial_result(f"{prefix}.h5")
    assert result["metadata"]["source_spatial_product"] == str(output)
    assert result["config"]["local_slope_method"] == "adjacent"
    for case in result["cases"]:
        assert case["local"]["slopes"].shape == (1, 6, 4)
        product = spatial.spatial_plot_product(case)
        np.testing.assert_array_equal(product["q"]["S_n"], case["q"]["S_n"])
    for name in ("structure_functions", "local_slopes"):
        for suffix in ("png", "pdf"):
            assert Path(f"{prefix}_{name}.{suffix}").stat().st_size > 0
    with pytest.raises(ValueError, match="separate"):
        spatial.postprocess_spatial_result(output, output_prefix=output.with_suffix(""))
    with pytest.raises(ValueError, match="overwrite"):
        spatial.postprocess_spatial_result(f"{prefix}.h5", output_prefix=output.with_suffix(""))


def test_comparison_fits_actual_points_and_saves_provenance(tmp_path, monkeypatch):
    _, cfg = configuration(tmp_path)
    monkeypatch.setattr(spatial, "MalaraTurbulence", AnalyticEvaluator)
    method = spatial.SpatialStructureFunctionAnalysis(cfg["spatial_structure_function"])
    result = method.run(cfg["input"]["path"])
    centers = np.array([1.5, 3, 6, 9, 12.0])
    values = centers[None, None, :] ** (np.arange(1, 7)[None, :, None] / 3)
    source = tmp_path / "hs.json"
    source.write_text(
        json.dumps(
            {
                "products": {
                    "truth": {
                        "orders": list(range(1, 7)),
                        "lambda_bin_centers_km": centers.tolist(),
                        "counts": np.ones((1, 5)).tolist(),
                        "z_plus": {"S_n": values.tolist()},
                    }
                }
            }
        )
    )
    config = tmp_path / "hs.yaml"
    config.write_text(yaml.safe_dump({"cases": {"hs": {"path": str(source), "p_value": 0.5}}}))
    result["config"].update(comparison_config=str(config))
    result["config"]["plot"]["enabled"] = True
    output = method.write_result(result, tmp_path / "compare.h5")
    loaded = spatial.read_spatial_result(output)
    comparison = loaded["comparisons"]["0.5"]
    assert comparison["path"] == str(source)
    np.testing.assert_array_equal(comparison["centers_km"], centers)
    np.testing.assert_array_equal(comparison["fit"]["bins_used"][0, 0], [3] * 6)
    np.testing.assert_allclose(comparison["fit"]["raw"][0, 0], np.arange(1, 7) / 3)
    assert (tmp_path / "compare_comparison.png").exists()
    config.write_text("cases: {}")
    with pytest.raises(ValueError, match="nonempty"):
        spatial._comparison_products(result)
    config.unlink()
    with pytest.warns(UserWarning, match="unavailable"):
        assert spatial._comparison_products(result) == {}


def test_cli_overrides_and_hs_centers(tmp_path, monkeypatch):
    cfg_path, cfg = configuration(tmp_path)
    monkeypatch.setattr(spatial, "MalaraTurbulence", AnalyticEvaluator)
    prefix = tmp_path / "override"
    assert (
        sample_main(
            [
                str(cfg_path),
                "--n-pairs",
                "8",
                "--p",
                "0.9",
                "--seed",
                "12",
                "--lambda-centers-km",
                "1",
                "3",
                "8",
                "--output-prefix",
                str(prefix),
                "--smoke",
            ]
        )
        == 0
    )
    loaded = spatial.read_spatial_result(f"{prefix}.h5")
    assert len(loaded["cases"]) == 1
    assert loaded["metadata"]["smoke"]
    assert loaded["config"]["seed"] == 12
    np.testing.assert_array_equal(loaded["cases"][0]["counts"], [[8] * 3])
    hs = tmp_path / "centers.json"
    hs.write_text(json.dumps({"products": {"truth": {"lambda_bin_centers_km": [1, 5, 7]}}}))
    np.testing.assert_array_equal(spatial._centers({"centers_from_hs": str(hs)}), [1, 5, 7])
    with pytest.raises(ValueError, match="choose"):
        spatial._centers({"centers_from_hs": str(hs), "lambda_centers_km": [1, 2, 3]})


@pytest.mark.parametrize(
    "change",
    [
        {"n_pairs": 0},
        {"n_pairs": 2},
        {"chunk_size": 0},
        {"seed": -1},
        {"p_values": [0.4]},
        {"lambda_centers_km": [1, 2, 2]},
    ],
)
def test_invalid_options_rejected_before_field_evaluation(tmp_path, monkeypatch, change):
    _, cfg = configuration(tmp_path)
    options = {**cfg["spatial_structure_function"], **change}
    monkeypatch.setattr(spatial, "sample_case", lambda *a, **kw: pytest.fail("sampling started"))
    with pytest.raises(ValueError):
        spatial.SpatialStructureFunctionAnalysis(options).run(cfg["input"]["path"])


def test_nonfinite_and_overflow_evaluations_are_errors():
    with pytest.raises(ValueError, match="nonfinite"):
        spatial._evaluated_moments(np.full((1, 2, 3), np.nan))
    values = np.zeros((1, 2, 3))
    values[:, 1] = 1e100
    with pytest.raises(ValueError, match="overflow"):
        spatial._evaluated_moments(values)
    with pytest.raises(ValueError, match="overflow"):
        spatial._kahan_add(np.array([1e308]), np.array([0.0]), np.array([1e308]))


def test_output_protects_other_hdf5_products(tmp_path, monkeypatch):
    _, cfg = configuration(tmp_path)
    monkeypatch.setattr(spatial, "MalaraTurbulence", AnalyticEvaluator)
    method = spatial.SpatialStructureFunctionAnalysis(cfg["spatial_structure_function"])
    result = method.run(cfg["input"]["path"])
    target = tmp_path / "observations.h5"
    with h5py.File(target, "w") as handle:
        handle.create_dataset("sentinel", data=[1, 2, 3])
    before = target.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        method.write_result(result, target)
    assert target.read_bytes() == before
