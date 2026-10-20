"""Lightweight orchestration tests; fake CLI workloads, no production SF evaluation."""

from datetime import datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import pytest
import yaml


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/benchmark_sf_parallelism.py"
spec = importlib.util.spec_from_file_location("benchmark_sf_parallelism", MODULE_PATH)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


@pytest.fixture
def workload(tmp_path, monkeypatch):
    source = tmp_path / "input.h5"
    source.write_bytes(b"fake input; never opened as HDF5")
    production = tmp_path / "production.json"
    production.write_text("preserve production")
    diagnostic = tmp_path / "production.png"
    diagnostic.write_bytes(b"preserve figure")
    config = {
        "input": {"path": str(source)},
        "analysis": {"method": "structure_function"},
        "output": {"path": str(production)},
        "structure_function": {
            "data_groups": ["truth"],
            "lambda_bins_km": None,
            "scale_ranges_km": {"total": [100, 1000]},
            "pair_sampling": {"random_seed": 3, "chunk_size": 7},
            "plot": {
                "enabled": True,
                "path": str(diagnostic),
                "publication": {"enabled": False, "path": str(tmp_path / "pub.png")},
            },
        },
    }
    config_path = tmp_path / "analysis.yaml"
    config_path.write_text(yaml.safe_dump(config))
    executable = tmp_path / "hsobs-analyze"
    executable.write_text(
        f"#!{sys.executable}\n"
        + """
import argparse
import json
import os
from pathlib import Path
import time
import yaml
parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("--workers", type=int)
args = parser.parse_args()
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    assert os.environ[name] == "1"
config = yaml.safe_load(Path(args.config).read_text())
time.sleep(0.03)
if os.environ.get("TEST_FAIL_WORKER") == str(args.workers):
    raise SystemExit(7)
value = 2.0 if os.environ.get("TEST_MISMATCH_WORKER") == str(args.workers) else 1.0
result = {"method": "structure_function", "products": {"truth": {"counts": [[2]],
          "z_plus": {"S_n": [[[value, float("nan")]]], "flatness": [[1.5]]}}},
          "metadata": {"created_utc": str(time.time()), "seed": 3}}
Path(config["output"]["path"]).write_text(json.dumps(result))
Path(config["structure_function"]["plot"]["path"]).write_bytes(b"isolated figure")
"""
    )
    executable.chmod(0o755)
    monkeypatch.setattr(bench, "available_cpus", lambda: 8)
    output = tmp_path / "benchmark"
    args = [
        str(config_path),
        "--cli",
        str(executable),
        "--output-dir",
        str(output),
        "--workers",
        "1",
        "2",
        "--repetitions",
        "1",
        "--sample-interval",
        "0.005",
    ]
    return args, output, config, production, diagnostic


@pytest.mark.parametrize(
    "args",
    [
        ["--workers", "0"],
        ["--workers", "2", "2"],
        ["--repetitions", "0"],
        ["--sample-interval", "0"],
        ["--sample-interval", "nan"],
        ["--max-differences", "-1"],
    ],
)
def test_invalid_arguments(args):
    with pytest.raises(SystemExit):
        bench.arguments(args)


def test_defaults_and_worker_subset():
    args = bench.arguments([])
    assert args.workers == [1, 2, 4, 8] and args.repetitions == 3
    assert bench.planned_workers([4], 8) == [1, 4]
    with pytest.warns(UserWarning, match="skipping"):
        assert bench.planned_workers([1, 4], 2) == [1]
    with pytest.warns(UserWarning, match="explicitly allowed"):
        assert bench.planned_workers([4], 2, allow=True) == [1, 4]


def test_cpu_affinity_and_slurm_limit(monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(8)))
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "3")
    assert bench.available_cpus() == 3


def test_dry_run_does_not_launch_hash_or_write(workload, monkeypatch):
    args, root, *_ = workload

    def forbidden(*args, **kwargs):
        pytest.fail("dry run must not hash or launch analyses")

    monkeypatch.setattr(bench, "sha256", forbidden)
    monkeypatch.setattr(bench, "run_process", forbidden)
    assert bench.main([*args, "--dry-run"]) == 0
    assert not root.exists()


def test_output_isolation_and_scientific_settings(workload):
    _, root, config, production, _ = workload
    rewritten = bench.isolated_config(config, Path(config["input"]["path"]), root)
    assert (
        rewritten["structure_function"]["pair_sampling"]
        == config["structure_function"]["pair_sampling"]
    )
    assert rewritten["structure_function"]["lambda_bins_km"] is None
    assert (
        rewritten["structure_function"]["scale_ranges_km"]
        == config["structure_function"]["scale_ranges_km"]
    )
    assert config["output"]["path"] == str(production)
    assert Path(rewritten["output"]["path"]).is_relative_to(root)
    for path in (
        rewritten["structure_function"]["plot"]["path"],
        rewritten["structure_function"]["plot"]["publication"]["path"],
    ):
        assert Path(path).is_relative_to(root)


def test_exact_comparison_nan_masks_schema_order_and_metadata():
    baseline = {
        "products": {"observed": {"S_n": [[1.0, math.nan]], "indices": [3, 1]}},
        "metadata": {"created_utc": "first", "seed": 3},
    }
    actual = json.loads(json.dumps(baseline))
    actual["metadata"]["created_utc"] = "second"
    assert list(bench.differences(baseline, actual)) == []
    actual["products"]["observed"]["S_n"][0][1] = 2.0
    actual["products"]["observed"]["indices"] = [1, 3]
    actual["metadata"]["seed"] = 4
    paths = [diff["path"] for diff in bench.differences(baseline, actual)]
    assert "/products/observed/S_n/0/1" in paths
    assert "/products/observed/indices/0" in paths
    assert "/metadata/seed" in paths
    assert list(bench.differences({"x": 1}, {"x": 1.0}))[0]["kind"] == "type/schema"
    assert list(bench.differences({"x": 1}, {}))[0]["actual"] == "<missing>"
    assert list(bench.differences([1], [1, 2]))[0]["kind"] == "length"
    assert list(bench.differences({"a/b": 1}, {"a/b": 2}))[0]["path"] == "/a~1b"
    assert list(bench.differences({"x": 1.0}, {"x": math.nextafter(1.0, 2.0)}))


def test_sampled_memory_includes_children_and_excludes_other_groups(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    for pid, group, pages in [(1, 10, 20), (2, 10, 30), (3, 99, 1000)]:
        path = tmp_path / str(pid)
        path.mkdir()
        fields = ["S", "0", str(group), *(["0"] * 18), str(pages)]
        (path / "stat").write_text(f"{pid} (name with ) parens) " + " ".join(fields))
    assert bench.process_group_rss(10, tmp_path) == 50 * os.sysconf("SC_PAGE_SIZE")
    monkeypatch.setattr(sys, "platform", "darwin")
    assert bench.process_group_rss(10, tmp_path) is None


def test_summary_calculations():
    rows = [{"workers": 1, "wall_seconds": value, "equivalence": "exact"} for value in (10, 14)] + [
        {"workers": 2, "wall_seconds": 3, "equivalence": "exact"},
        {"workers": 2, "wall_seconds": 0.1, "equivalence": "mismatch"},
    ]
    assert bench.summary(rows) == [
        {"workers": 1, "valid_runs": 2, "median_seconds": 12, "speedup": 1, "efficiency": 1},
        {"workers": 2, "valid_runs": 1, "median_seconds": 3, "speedup": 4, "efficiency": 2},
    ]


def test_sequential_cli_runs_cache_and_rerun_preserve_artifacts(workload):
    args, root, config, production, diagnostic = workload
    assert bench.main(args) == 0
    assert production.read_text() == "preserve production"
    assert diagnostic.read_bytes() == b"preserve figure"
    for name in ("benchmark_results.csv", "benchmark_report.md", "benchmark_scaling.png"):
        assert (root / name).stat().st_size > 0
    first_runs = list((root / "runs").glob("**/execution.json"))
    assert len(first_runs) == 2
    started = [json.loads(path.read_text()) for path in sorted(first_runs)]
    assert started[0]["command"][-2:] == ["--workers", "1"]
    assert datetime.fromisoformat(started[1]["started_utc"]).timestamp() >= (
        datetime.fromisoformat(started[0]["started_utc"]).timestamp() + started[0]["wall_seconds"]
    )
    for path in first_runs:
        record = json.loads(path.read_text())
        assert record["wall_seconds"] > 0 and record["exit_status"] == 0
        if sys.platform.startswith("linux"):
            assert record["peak_aggregate_rss_bytes"] > 0
    assert bench.main(args) == 0
    assert len(list((root / "runs").glob("**/execution.json"))) == 2
    assert bench.main([*args, "--rerun"]) == 0
    assert len(list((root / "runs").glob("**/execution.json"))) == 4
    assert all(path.exists() for path in first_runs)
    rewritten = yaml.safe_load(next((root / "runs").glob("**/analysis.yaml")).read_text())
    assert (
        rewritten["structure_function"]["pair_sampling"]
        == config["structure_function"]["pair_sampling"]
    )


@pytest.mark.parametrize(
    "environment,expected",
    [
        ("TEST_FAIL_WORKER", "analysis exited with status 7"),
        ("TEST_MISMATCH_WORKER", "/products/truth/z_plus/S_n/0/0/0"),
    ],
)
def test_failure_and_mismatch_reports_preserve_completed_results(
    workload, monkeypatch, environment, expected
):
    args, root, *_ = workload
    monkeypatch.setenv(environment, "2")
    assert bench.main(args) == 1
    report = (root / "benchmark_report.md").read_text()
    assert expected in report
    serial = root / "runs/workers-1/rep-1/attempt-000001/result.json"
    assert serial.is_file()
    assert "exact" in (serial.parent / "validation.json").read_text()


def test_manifest_changes_require_new_output_directory(workload):
    args, root, config, *_ = workload
    assert bench.main(args) == 0
    Path(config["input"]["path"]).write_bytes(b"changed input")
    assert bench.main(args) == 2
    assert len(list((root / "runs").glob("**/execution.json"))) == 2


def test_production_root_and_symlink_escape_rejected(workload, tmp_path):
    args, _, _, production, _ = workload
    assert bench.main([*args, "--output-dir", str(production.parent), "--dry-run"]) == 2
    root = tmp_path / "isolated"
    root.mkdir()
    (root / "runs").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        bench.new_attempt(root, 1, 1)


def test_comparison_also_checks_embedded_percentiles():
    assert (
        list(bench.differences({"mc_percentiles": [1, 2]}, {"mc_percentiles": [1, 3]}))[0]["path"]
        == "/mc_percentiles/1"
    )


def test_derived_percentiles_keep_nan_masks_and_member_axis():
    import numpy as np

    observed = {"counts": [[1], [3], [9]]}
    for field in ("z_plus", "z_minus"):
        observed[field] = {
            "S_n": [[[1.0, math.nan]], [[3.0, math.nan]], [[9.0, math.nan]]],
            "flatness": [[1.0], [3.0], [9.0]],
            "scaling_exponents": {
                "raw": [[[1.0]], [[3.0]], [[9.0]]],
                "normalized_by_order2": [[[1.0]], [[1.0]], [[1.0]]],
            },
        }
    stats = bench.ensemble_statistics({"products": {"observed": observed}})
    np.testing.assert_array_equal(
        stats["counts"], np.percentile([[1], [3], [9]], [5, 50, 95], axis=0)
    )
    assert all(math.isnan(curve[0][1]) for curve in stats["z_plus/S_n"])
    assert stats["z_plus/normalized_by_order2"] == [[[1.0]], [[1.0]], [[1.0]]]


def test_missing_baseline_fails_validation(workload, monkeypatch):
    args, root, *_ = workload
    monkeypatch.setenv("TEST_FAIL_WORKER", "1")
    assert bench.main(args) == 1
    assert "baseline_unavailable" in (root / "benchmark_report.md").read_text()


def test_completed_result_corruption_is_reported_without_rerun(workload):
    args, root, *_ = workload
    assert bench.main(args) == 0
    path = root / "runs/workers-2/rep-1/attempt-000001/result.json"
    path.write_text("{}")
    assert bench.main(args) == 1
    assert "saved result changed after completion" in (root / "benchmark_report.md").read_text()
    assert len(list((root / "runs").glob("**/execution.json"))) == 2


def test_memory_unavailable_is_not_reported_as_parent_only(workload, monkeypatch):
    args, root, *_ = workload
    monkeypatch.setattr(bench, "process_group_rss", lambda pid: None)
    assert bench.main(args) == 0
    record = json.loads(next((root / "runs").glob("**/execution.json")).read_text())
    assert record["peak_aggregate_rss_bytes"] is None
    assert record["memory_method"] == "unavailable"


def test_malformed_observed_schema_still_reports_json_paths(tmp_path):
    original = {"method": "structure_function", "products": {"observed": {"counts": [[1]]}}}
    changed = {"method": "structure_function", "products": {"observed": {"counts": [[2]]}}}
    records = []
    for workers, result in [(1, original), (2, changed)]:
        directory = tmp_path / str(workers)
        directory.mkdir()
        bench.write_json(directory / "result.json", result)
        bench.write_json(directory / "ensemble_statistics.json", bench.statistics_payload(result))
        records.append(
            (
                directory,
                {
                    "workers": workers,
                    "repetition": 1,
                    "exit_status": 0,
                    "wall_seconds": 1,
                    "error": "",
                    "result_sha256": bench.sha256(directory / "result.json"),
                    "ensemble_statistics_sha256": bench.sha256(
                        directory / "ensemble_statistics.json"
                    ),
                },
            )
        )
    rows = bench.evaluate_runs(records, 20)
    assert all(row["equivalence"] == "mismatch" for row in rows)
    assert "/products/observed/counts/0/0" in [diff["path"] for diff in rows[1]["differences"]]
    assert bench.summary(rows) == []
