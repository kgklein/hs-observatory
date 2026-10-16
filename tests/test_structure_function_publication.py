import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import yaml

from hs_observatory.analysis import structure_function_publication as publication
from hs_observatory.analysis.structure_function import _fit_scaling_exponents, _settings
from hs_observatory.analysis.structure_function_convergence import stage2_edges


def result_fixture():
    centers = np.sqrt(stage2_edges()[:-1] * stage2_edges()[1:])
    orders = list(range(1, 7))
    products = {}
    for group, offsets, amplitude in [
        ("truth", [0.2], 100.0),
        ("observed", [0.01, 0.02, 0.04, 0.08], 1.0),
    ]:
        values = np.array(
            [[amplitude * centers ** (n / 3 + offset) for n in orders] for offset in offsets]
        )
        counts = np.ones((len(offsets), 26), dtype=int)
        field = dict(
            S_n=values.tolist(),
            scaling_exponents=_fit_scaling_exponents(
                values,
                counts,
                centers,
                orders,
                {"primary": [150.0, 1200.0], "sensitivity": [200.0, 1000.0]},
                3,
            ),
        )
        products[group] = dict(
            orders=orders,
            counts=counts.tolist(),
            lambda_bin_edges_km=stage2_edges().tolist(),
            lambda_bin_centers_km=centers.tolist(),
            z_plus=field,
            z_minus=dict(field),
        )
    return dict(
        products=products,
        metadata=dict(
            stage3=dict(
                provenance=dict(
                    generation_config=dict(
                        field=dict(
                            domain=dict(L=[100000.0] * 3), turbulence=dict(injection_scale_index=2)
                        )
                    )
                )
            )
        ),
    )


def test_saved_percentiles_use_only_individual_mc_members():
    result = result_fixture()
    publication.add_saved_summaries(result)
    observed = result["products"]["observed"]["z_plus"]
    np.testing.assert_allclose(
        observed["ensemble_summary"]["S_n"], np.percentile(observed["S_n"], [5, 50, 95], axis=0)
    )
    np.testing.assert_allclose(
        observed["ensemble_summary"]["xi_raw"],
        np.percentile(observed["scaling_exponents"]["raw"], [5, 50, 95], axis=0),
    )
    assert "ensemble_summary" not in result["products"]["truth"]["z_plus"]
    assert observed["local_slopes"]["slopes"].shape == (4, 6, 26)


@pytest.mark.parametrize("run", [400, 401, 402])
def test_two_panel_styles_theory_physical_annotation_and_raw_exponents(run):
    import matplotlib.pyplot as plt

    result = result_fixture()
    fig = publication.publication_figure(result, run, [150.0, 1200.0], ((1.0, 1e10), (0.0, 3.0)))
    assert len(fig.axes) == 2
    left, right = fig.axes
    assert left.get_xlim() == (70.0, 20000.0)
    assert len(left.lines) == 12
    assert left.lines[0].get_markerfacecolor() == "white"
    assert left.lines[1].get_linestyle() == "--"
    p = publication.P_VALUES[run]
    n = np.arange(1, 7)
    np.testing.assert_allclose(
        right.lines[-1].get_ydata(), 1 - np.log2(p ** (n / 3) + (1 - p) ** (n / 3))
    )
    # xi3 stays 1.2, and is not forced to 1.
    assert right.lines[0].get_ydata()[2] == pytest.approx(1.2)
    assert any("25{,}000" in text.get_text() for text in fig.texts)
    plt.close(fig)


def test_replot_never_dispatches_analysis_and_opt_in_does(monkeypatch):
    called = []
    monkeypatch.setattr(publication, "run_analysis", lambda run: called.append(("analysis", run)))
    monkeypatch.setattr(publication, "plot_saved", lambda *args: called.append(("plot", args[0])))
    publication.main(["--cases", "400"])
    assert called == [("plot", [400])]
    called.clear()
    publication.main(["--cases", "401", "--recalculate", "--analysis-only"])
    assert called == [("analysis", 401)]


def test_pilot_schema_outputs_and_budgets():
    root = Path(__file__).resolve().parents[1]
    for run in [400, 401, 402]:
        path = root / f"analysis/malara-prod-{run}_sf-stage3-publication.yaml"
        config = yaml.safe_load(path.read_text())
        settings = _settings(config["structure_function"])
        assert config["analysis"]["method"] == "structure_function"
        assert "structure_function_stage3" in config["output"]["path"]
        np.testing.assert_array_equal(settings["lambda_bins_km"], stage2_edges())
        assert (
            settings["pair_sampling"]["target_pairs_per_bin"]
            == [10000] * 4 + [100000] * 14 + [10000] * 8
        )
        assert settings["scale_ranges_km"] == {
            "primary": [150.0, 1200.0],
            "sensitivity": [200.0, 1000.0],
        }
        assert not settings["plot"]["publication"]["enabled"]


def test_saved_only_replot_produces_both_formats_without_input_hdf5(tmp_path, monkeypatch):
    from hs_observatory.analysis.structure_function import _to_serializable

    monkeypatch.setattr(publication, "DIRECTORY", tmp_path)
    path = publication.result_path(402)
    path.write_text(json.dumps(_to_serializable(result_fixture())))
    monkeypatch.setattr(publication, "run_analysis", lambda _: pytest.fail("unexpected analysis"))
    publication.plot_saved([402])
    assert (tmp_path / "malara-prod-402_sf_publication.png").is_file()
    assert (tmp_path / "malara-prod-402_sf_publication.pdf").is_file()
    publication.plot_saved([402], window="sensitivity")
    assert (tmp_path / "malara-prod-402_sf_publication_fit_200_1000.pdf").is_file()
    assert json.loads((tmp_path / "plot_summary_fit_200_1000.json").read_text())["402"][
        "fit_bounds"
    ] == [200.0, 1000.0]


def test_shell_recalculation_requires_stage3_and_legacy_help_still_works():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/structure_function_publication.sh"
    rejected = subprocess.run(
        ["bash", str(script), "--recalculate", "400"], capture_output=True, text=True
    )
    assert rejected.returncode == 2
    assert "require --stage3" in rejected.stderr
    help_result = subprocess.run(["bash", str(script), "--help"], capture_output=True, text=True)
    assert help_result.returncode == 0
    assert "100|101|102|400|401|402" in help_result.stderr
