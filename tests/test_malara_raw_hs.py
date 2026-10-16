"""Tiny shared-realization fixtures only; never read production observations."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from hs_observatory.analysis import malara_raw_hs as study
from hs_observatory.analysis.derived.geometry_pairs import GeometryPlanCache, geometry_pair_chunks
from hs_observatory.analysis.structure_function import compute_all_spacetime_structure_functions
from hs_observatory.fields.malara import MalaraTurbulenceField


@pytest.fixture
def workspace(tmp_path):
    generation = yaml.safe_load(Path("generation/malara-ts-500a-t1500.yaml").read_text())
    generation["field"]["turbulence"].update(n_scales=2, injection_scale_index=1, backend="cached")
    generation["field"]["domain"].update(L=[4000.0] * 3, origin=[17.0, -29.0, 41.0])
    generation["time"] = {"t_start_s": 0.0, "t_stop_s": 10.0, "cadence_s": 1.0}
    generation["output"]["path"] = "tiny.h5"
    time = np.arange(11.0)
    # Includes positions outside the periodic domain; separation must stay unwrapped.
    positions = np.array([[3990.0, 0.0, 10.0], [4110.0, 50.0, 0.0], [4400.0, 200.0, 20.0]])[
        None
    ] + time[:, None, None] * [105.0, 145.0, 205.0]
    adapter = MalaraTurbulenceField(generation["field"])
    with h5py.File(tmp_path / "tiny.h5", "w") as h:
        h.create_group("metadata").attrs.update(
            config_yaml=yaml.safe_dump(generation),
            n_mc=0,
            velocity_truth_frame="plasma",
            spacecraft_source=json.dumps({"position_units": "km"}),
        )
        h.create_dataset("spacecraft/id", data=np.array(["A", "B", "C"], dtype="S"))
        for q in ("n", "v", "B"):
            h.create_dataset(f"time/{q}", data=time)
            h.create_dataset(f"spacecraft/r/{q}", data=positions.swapaxes(0, 1))
            h.create_dataset(
                f"truth/{q}", data=adapter.evaluate_quantity(q, time, positions.swapaxes(0, 1))
            )
    (tmp_path / "generation.yaml").write_text(yaml.safe_dump(generation))
    config = {
        "generation": "generation.yaml",
        "output": {"directory": "results"},
        "sampling": {"fit_pairs": 8, "outside_pairs": 4, "seeds": [12345, 23456, 34567]},
    }
    (tmp_path / "analysis.yaml").write_text(yaml.safe_dump(config))
    return tmp_path


def test_reconstruction_frames_origin_wrapping(workspace):
    entry, positions = study.preflight(workspace, "generation.yaml")
    quantities, checks = study.reconstruct(workspace, entry, positions, chunk_size=7)
    assert max(checks.values()) < 1e-9
    np.testing.assert_allclose(
        quantities["ideal_z_plus"] / entry["K"], quantities["centered_q"], atol=1e-10
    )
    adapter = MalaraTurbulenceField(entry["generation"]["field"])
    raw = adapter._quantity_evaluators["v"].evaluate_vector(positions - adapter.origin)
    np.testing.assert_allclose(quantities["raw_q"], raw)
    assert entry["field_config"]["realization"] == 1
    assert entry["origin_km"].tolist() == [17.0, -29.0, 41.0]


@pytest.mark.parametrize(
    "change", ["realization", "origin", "frame", "units", "positions", "time", "offset"]
)
def test_consistency_rejects_changes(workspace, change):
    with h5py.File(workspace / "tiny.h5", "r+") as h:
        config = yaml.safe_load(h["metadata"].attrs["config_yaml"])
        if change == "realization":
            config["field"]["turbulence"]["realization"] += 1
        elif change == "origin":
            config["field"]["domain"]["origin"][0] += 1
        elif change == "frame":
            h["metadata"].attrs["velocity_truth_frame"] = "spacecraft"
        elif change == "units":
            h["metadata"].attrs["spacecraft_source"] = json.dumps({"position_units": "m"})
        elif change == "positions":
            h["spacecraft/r/B"][0, 0, 0] += 1
        elif change == "time":
            h["time/B"][0] += 0.01
        else:
            config["field"]["output_fields"]["B"]["realization_offset"] = 1
            (workspace / "generation.yaml").write_text(yaml.safe_dump(config))
        h["metadata"].attrs["config_yaml"] = yaml.safe_dump(config)
    with pytest.raises(ValueError):
        study.preflight(workspace, "generation.yaml")


def test_corrupt_ideal_truth_rejected(workspace):
    with h5py.File(workspace / "tiny.h5", "r+") as h:
        h["truth/v"][0, 0, 0] += 1
    entry, positions = study.preflight(workspace, "generation.yaml")
    with pytest.raises(ValueError, match="ideal v"):
        study.reconstruct(workspace, entry, positions, chunk_size=7)


def test_one_pair_stream_and_existing_estimator(workspace):
    entry, positions = study.preflight(workspace, "generation.yaml")
    values, _ = study.reconstruct(workspace, entry, positions, chunk_size=7)
    plan, _ = GeometryPlanCache().get(positions, study.EDGES, 70.0, 20000.0)
    targets = np.full(40, 8)
    pairs = list(geometry_pair_chunks(plan, targets, random_seed=12345, chunk_size=7))
    visited = []

    def stream():
        for left, right in pairs:
            visited.append(np.column_stack((left, right)))
            yield left, right

    result = study.accumulate(positions, values, study.EDGES, stream(), entry["K"])
    expected_digest = hashlib.sha256(np.concatenate(visited).astype("<i8").tobytes()).hexdigest()
    assert result["pair_index_sha256"] == expected_digest
    assert len(visited) == len(pairs)
    assert max(result["increment_checks"].values()) < 1e-10
    counts = result["contributions"]["all"]["counts"]
    np.testing.assert_array_equal(counts[0], np.minimum(plan.populations, targets))
    all_q = result["contributions"]["all"]["quantities"]
    same = result["contributions"]["same_spacecraft"]["quantities"]
    np.testing.assert_allclose(same["raw_q"]["S_n"], same["centered_q"]["S_n"], atol=1e-12)
    cross = result["contributions"]["cross_spacecraft"]["quantities"]
    assert not np.allclose(cross["raw_q"]["S_n"], cross["centered_q"]["S_n"], equal_nan=True)
    np.testing.assert_allclose(all_q["centered_q"]["S_n"], all_q["ideal_z_plus"]["S_n"], atol=1e-12)
    native = all_q["ideal_z_plus"]["moment_sums_native"]
    powers = abs(entry["K"]) ** np.array(study.ORDERS)
    expected = (
        np.divide(
            native,
            counts[:, None, :],
            out=np.full_like(native, np.nan),
            where=counts[:, None, :] > 0,
        )
        / powers[None, :, None]
    )
    np.testing.assert_allclose(expected, all_q["ideal_z_plus"]["S_n"])
    # Independently check agreement with the pre-existing estimator (not used as
    # evidence of strict matching; the single stream above provides that guarantee).
    old = compute_all_spacetime_structure_functions(
        positions[None],
        values["raw_q"][None],
        values["centered_q"][None],
        study.ORDERS,
        study.EDGES,
        pair_sampling={
            "mode": "geometry_stratified",
            "target_pairs_per_bin": targets,
            "random_seed": 12345,
            "chunk_size": 7,
        },
        scale_ranges_km=study.WINDOWS,
        min_pair_lambda_km=70.0,
        max_pair_lambda_km=20000.0,
    )
    np.testing.assert_allclose(all_q["raw_q"]["S_n"], old["z_plus"]["S_n"])
    np.testing.assert_allclose(all_q["centered_q"]["S_n"], old["z_minus"]["S_n"])
    other_chunks = study.accumulate(
        positions,
        values,
        study.EDGES,
        geometry_pair_chunks(plan, targets, random_seed=12345, chunk_size=19),
        entry["K"],
    )
    assert other_chunks["pair_index_sha256"] == result["pair_index_sha256"]


def test_preflight_does_not_evaluate_or_sample(workspace, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("preflight dispatched evaluation or sampling")

    monkeypatch.setattr(study, "reconstruct", forbidden)
    monkeypatch.setattr(study, "geometry_pair_chunks", forbidden)
    assert study.main(["--root", str(workspace), "--config", "analysis.yaml", "--dry-run"]) == 0
    assert not (workspace / "results").exists()


def test_truth_only_fit_plot_resume(workspace, monkeypatch):
    args = ["--root", str(workspace), "--config", "analysis.yaml", "--field-chunk-size", "7"]
    from hs_observatory.analysis.spatial_structure_function import write_spatial_result

    entry, _ = study.preflight(workspace, "generation.yaml")
    centers = np.sqrt(study.EDGES[:-1] * study.EDGES[1:])
    case = {
        "p_value": 0.9,
        "field_config": entry["field_config"],
        "lambda_bin_centers_km": centers,
        "counts": np.ones((1, 40), dtype=int),
        "q": {"S_n": centers[None, None] ** (np.arange(1, 7)[None, :, None] / 3)},
    }
    direct = workspace / "direct.h5"
    write_spatial_result({"metadata": {}, "config": {}, "cases": [case]}, direct)
    assert study.main(args + ["--direct3d", str(direct)]) == 0
    output = workspace / "results"
    assert "isotropic exact-separation estimates" in (output / "report.md").read_text()
    assert len(study.direct_cases([direct], entry)) == 1
    mismatched = deepcopy(entry)
    mismatched["field_config"]["realization"] += 1
    with pytest.raises(ValueError, match="configuration differs"):
        study.direct_cases([direct], mismatched)
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["records"]) == 3
    assert len(manifest["edges_km"]) == 41
    for record in manifest["records"]:
        saved = json.loads((output / record["path"]).read_text())
        fits = saved["contributions"]["all"]["quantities"]["raw_q"]["fits"]
        assert np.asarray(fits["raw"]).shape == (1, 2, 6)
        assert np.isfinite(fits["raw"]).any()
    for suffix in ("png", "pdf"):
        assert (output / f"diagnostics.{suffix}").stat().st_size > 1000

    def forbidden(*args, **kwargs):
        pytest.fail("resume/plot-only evaluated field")

    monkeypatch.setattr(study, "reconstruct", forbidden)
    assert study.main(args) == 0
    # Plot-only must also work when original observation and recipe are unavailable.
    (workspace / "tiny.h5").unlink()
    (workspace / "generation.yaml").unlink()
    assert study.main(args + ["--plot-only"]) == 0
    record = manifest["records"][0]
    path = output / record["path"]
    saved = json.loads(path.read_text())
    saved["contributions"]["all"]["quantities"]["raw_q"]["S_n"][0][0][1] = 999
    study.save_json(path, saved)
    assert (
        study.validate_completed(path, record["signature"], np.array(manifest["expected_counts"]))
        is None
    )
    with pytest.raises(ValueError, match="invalid completed"):
        study.main(args + ["--plot-only"])


def test_fit_power_law_with_empty_bins():
    centers = np.sqrt(study.EDGES[:-1] * study.EDGES[1:])
    moments = centers[None, None] ** (np.arange(1, 7)[None, :, None] / 3)
    counts = np.ones((1, 40), dtype=int)
    counts[0, -1] = 0
    moments[..., -1] = np.nan
    record = {
        "contributions": {
            group: {
                "counts": counts,
                "quantities": {q: {"S_n": deepcopy(moments)} for q in study.QUANTITIES},
            }
            for group in study.CONTRIBUTIONS
        }
    }
    study.derive(record, study.EDGES)
    fits = record["contributions"]["all"]["quantities"]["raw_q"]["fits"]
    np.testing.assert_allclose(fits["raw"][0], np.tile(np.arange(1, 7) / 3, (2, 1)))
