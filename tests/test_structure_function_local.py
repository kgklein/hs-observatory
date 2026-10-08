"""Focused numerical and post-processing integration tests for local slopes."""

import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pytest
import yaml

matplotlib.use("Agg")

from hs_observatory.analysis import structure_function as sf  # noqa: E402
from hs_observatory.analysis import structure_function_local as local  # noqa: E402
from hs_observatory.cli import plot_structure_function_overlay_main  # noqa: E402


@pytest.mark.parametrize("method", ["three-point", "adjacent"])
def test_power_law_nonuniform_spacing(method):
    centers = np.exp([0, 0.3, 1.2, 2.7, 4.0, 4.2])
    alpha = np.array([0.2, 1.0, 1.7])
    values = 3 * centers[None, None, :] ** alpha[None, :, None]
    x, slopes, valid = local.local_log_slopes(centers, values, method=method)
    interior = slopes[..., 1:-1] if method == "three-point" else slopes
    np.testing.assert_allclose(interior, np.broadcast_to(alpha[None, :, None], interior.shape))
    assert valid.all()
    if method == "three-point":
        np.testing.assert_array_equal(x, centers)
        assert np.isnan(slopes[..., [0, -1]]).all()
    else:
        np.testing.assert_allclose(x, np.sqrt(centers[:-1] * centers[1:]))


def test_p_model_identities():
    orders = np.arange(1, 7)
    np.testing.assert_allclose(local.p_model_exponents(orders, 0.5), orders / 3)
    for p in (0.5, 0.7, 0.9):
        np.testing.assert_allclose(local.p_model_exponents([3], p), [1])


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf])
@pytest.mark.parametrize("method", ["three-point", "adjacent"])
def test_invalid_values_do_not_bridge_gaps(bad, method):
    centers = np.geomspace(1, 64, 7)
    values = centers[None, None, :].copy()
    values[..., 3] = bad
    _, slopes, valid = local.local_log_slopes(centers, values, method=method)
    assert not valid[..., 3].any()
    gap = slice(2, 5) if method == "three-point" else slice(2, 4)
    assert np.isnan(slopes[..., gap]).all()
    assert np.isfinite(slopes[..., 1]).all()
    assert np.isfinite(slopes[..., -2]).all()


@pytest.mark.parametrize("bad", [0, np.nan, np.inf, 2])
@pytest.mark.parametrize("method", ["three-point", "adjacent"])
def test_invalid_or_nonincreasing_separations(bad, method):
    centers = np.array([1, 2, 4, 8, 16, 32, 64.], dtype=float)
    values = centers[None, None, :].copy()
    centers[3] = bad
    _, slopes, _ = local.local_log_slopes(centers, values, method=method)
    index = 2 if method == "adjacent" else 3
    assert np.isnan(slopes[..., index]).all()
    if not np.isfinite(bad) or bad <= 0:
        assert np.isnan(slopes[..., 2]).all()


@pytest.mark.parametrize("method", ["three-point", "adjacent"])
def test_empty_bin_masks_neighborhood(method):
    centers = np.geomspace(1, 64, 7)
    counts = np.ones((2, 7))
    counts[1, 3] = 0
    values = np.broadcast_to(centers, (2, 1, 7))
    _, slopes, valid = local.local_log_slopes(centers, values, counts=counts, method=method)
    assert valid[0].all() and not valid[1, 0, 3]
    assert np.isfinite(slopes[0, 0, 2:4]).all()
    assert np.isnan(slopes[1, 0, 2:4]).all()


def saved_config(tmp_path, *, summary_only=False):
    centers = np.array([1., 2., 4., 8., 16.])
    orders = [1, 2, 3, 4, 5, 6]
    truth_values = centers[None, None, :] ** (np.array(orders)[None, :, None] / 3)
    # Curves cross; the ensemble median curve changes membership with separation.
    member_curves = np.array([[1, 4, 4, 4, 4],
                              [2, 8, 8, 8, 8],
                              [1024, 1, 1, 1, 1]], dtype=float)
    mc_values = np.broadcast_to(member_curves[:, None, :], (3, 6, 5)).copy()

    def product(values):
        return {"orders": orders, "lambda_bin_centers_km": centers.tolist(),
                "lambda_bin_edges_km": [0.7, 1.4, 2.8, 5.6, 11.2, 22.4],
                "counts": np.ones((len(values), 5)).tolist(),
                "z_plus": {"S_n": values.tolist()}}

    result = {"products": {"truth": product(truth_values), "observed": product(mc_values)}}
    if summary_only:
        result["products"]["observed"]["z_plus"] = {"median": np.median(mc_values, axis=0).tolist()}
    path = tmp_path / "saved.json"
    path.write_text(json.dumps(result))
    generation = tmp_path / "generation.yaml"
    generation.write_text(yaml.safe_dump({"field": {
        "model": "malara_turbulence", "domain": {"L": [64, 64, 64], "units": "km"},
        "turbulence": {"geometry": "isotropic", "injection_scale_index": 2,
                       "n_scales": 8, "spectral_exponent_h": 1/3},
    }}))
    config = {"cases": {"test": {"path": str(path), "p_value": 0.7,
                                 "generation_config": str(generation)}},
              "quantity": "z_plus", "dpi": 40, "fit_lambda_range_km": [4, 8],
              "output": {"figure": str(tmp_path / "global.png")}}
    cfg_path = tmp_path / "overlay.yaml"
    cfg_path.write_text(yaml.safe_dump(config))
    return cfg_path, config, result


@pytest.mark.parametrize("method", ["three-point", "adjacent"])
def test_mc_statistics_are_from_member_slopes(tmp_path, method):
    _, config, result = saved_config(tmp_path)
    case = local.load_local_cases(config, "z_plus", method, False)[0]
    values = np.asarray(result["products"]["observed"]["z_plus"]["S_n"])
    centers = result["products"]["observed"]["lambda_bin_centers_km"]
    _, raw, _ = local.local_log_slopes(centers, values, method=method)
    p05, median, p95 = sf._ensemble_interval(raw)
    for key, expected in (("p05", p05), ("median", median), ("p95", p95)):
        np.testing.assert_allclose(case["mc"][key], expected)
    _, summary_slopes, _ = local.local_log_slopes(
        centers, np.median(values, axis=0)[None, ...], method=method,
    )
    valid = np.isfinite(median)
    assert not np.allclose(median[valid], summary_slopes[0][valid])


def test_mc_invalid_member_is_excluded_per_location(tmp_path):
    _, config, result = saved_config(tmp_path)
    result["products"]["observed"]["z_plus"]["S_n"][0][0][2] = 0
    Path(config["cases"]["test"]["path"]).write_text(json.dumps(result))
    case = local.load_local_cases(config, "z_plus", "three-point", False)[0]
    np.testing.assert_array_equal(case["mc"]["finite_members"][0], [0, 2, 2, 2, 0])
    assert np.isfinite(case["mc"]["median"][0, 1:4]).all()


def test_summary_only_omits_mc_and_roundtrips(tmp_path):
    cfg_path, _, _ = saved_config(tmp_path, summary_only=True)
    with pytest.warns(UserWarning, match="Individual MC curves unavailable"):
        outputs = local.local_slopes_from_config(cfg_path)
    with h5py.File(outputs[0]) as handle:
        case = handle["cases/0"]
        assert "mc" not in case
        assert "unavailable" in case.attrs["mc_limitation"]
        assert case["truth/slopes"].shape == (1, 6, 5)


@pytest.mark.parametrize("method", ["three-point", "adjacent"])
def test_cli_postprocessing_roundtrip_never_calls_upstream(tmp_path, monkeypatch, method):
    cfg_path, _, result = saved_config(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("upstream or global plotting invoked")

    import hs_observatory.generate as generate
    from hs_observatory.fields.malara import MalaraTurbulenceField

    monkeypatch.setattr(generate, "generate_from_config", forbidden)
    monkeypatch.setattr(MalaraTurbulenceField, "evaluate_quantity", forbidden)
    for name in ("compute_all_spacetime_structure_functions", "compute_pairwise_structure_functions",
                 "compute_elsasser_analysis", "plot_structure_function_overlay_from_config"):
        monkeypatch.setattr(sf, name, forbidden)
    monkeypatch.setattr(sf.StructureFunctionAnalysis, "run", forbidden)
    prefix = tmp_path / "derived"
    assert plot_structure_function_overlay_main([
        str(cfg_path), "--local-slopes", "--local-slope-method", method,
        "--show-cascade-scales", "--output-prefix", str(prefix),
    ]) == 0
    for suffix in (".h5", ".png", ".pdf", "_focused.png", "_focused.pdf"):
        assert Path(f"{prefix}{suffix}").stat().st_size > 0
    with h5py.File(f"{prefix}.h5") as handle:
        assert handle.attrs["estimator"] == method
        case = handle["cases/0"]
        np.testing.assert_array_equal(case["orders"], [1, 2, 3, 4, 5, 6])
        shape = (1, 6, 5 if method == "three-point" else 4)
        assert case["truth/slopes"].shape == shape
        assert case["mc/slopes"].shape == (3, *shape[1:])
        assert case["mc/source_valid"].dtype == np.dtype(bool)
        assert case["mc/slope_valid"].dtype == np.dtype(bool)
        context = json.loads(case.attrs["cascade_json"])
        assert context["cell_sizes_km"] == [16, 2, 0.25]
        np.testing.assert_allclose(case["truth/slopes"][0, :, 1:-1],
                                   np.broadcast_to(np.arange(1, 7)[:, None] / 3,
                                                   case["truth/slopes"][0, :, 1:-1].shape))
    assert json.loads((tmp_path / "saved.json").read_text()) == result
    assert not (tmp_path / "global.png").exists()


def test_annotations_and_missing_config(tmp_path):
    _, config, _ = saved_config(tmp_path)
    context = local.cascade_context(config["cases"]["test"]["generation_config"])
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    local._mark_scales(axis, {"cascade": context}, (1, 16))
    assert len(axis.lines) == 2
    assert [text.get_text() for text in axis.texts] == ["m=2", "m=5"]
    plt.close(figure)
    del config["cases"]["test"]["generation_config"]
    with pytest.raises(ValueError, match="generation_config"):
        local.load_local_cases(config, "z_plus", "three-point", True)
    # Ordinary post-processing must not read a generation config.
    local.load_local_cases(config, "z_plus", "three-point", False)


@pytest.mark.parametrize("change", ["anisotropic", "fractional_index", "missing_parameter"])
def test_cascade_requires_explicit_valid_parameters(tmp_path, change):
    _, config, _ = saved_config(tmp_path)
    path = Path(config["cases"]["test"]["generation_config"])
    generation = yaml.safe_load(path.read_text())
    field = generation["field"]
    if change == "anisotropic":
        field["domain"]["L"][1] = 32
    elif change == "fractional_index":
        field["turbulence"]["injection_scale_index"] = 2.5
    else:
        del field["turbulence"]["n_scales"]
    path.write_text(yaml.safe_dump(generation))
    with pytest.raises((ValueError, KeyError)):
        local.cascade_context(path)


@pytest.mark.parametrize("option", [["--output", "bad.png"], ["--fit-range", "total"],
                                    ["--fit-lambda-range", "1", "4"], ["--min-fit-bins", "3"]])
def test_local_cli_rejects_global_overrides(option):
    with pytest.raises(SystemExit) as exc:
        plot_structure_function_overlay_main(["unused.yaml", "--local-slopes", *option])
    assert exc.value.code == 2


def test_default_cli_routes_to_existing_overlay(monkeypatch):
    called = []
    monkeypatch.setattr(sf, "plot_structure_function_overlay_from_config",
                        lambda *args, **kwargs: called.append((args, kwargs)) or Path("plot.png"))
    assert plot_structure_function_overlay_main(["existing.yaml", "--fit-range", "total"]) == 0
    assert called[0][0] == ("existing.yaml",)
    assert called[0][1]["fit_range_override"] == "total"


def test_output_prefix_cannot_overwrite_publication(tmp_path):
    cfg_path, _, _ = saved_config(tmp_path)
    with pytest.raises(ValueError, match="overwrite"):
        local.local_slopes_from_config(cfg_path, output_prefix=tmp_path / "global")
