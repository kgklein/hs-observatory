#!/usr/bin/env python3
"""Isolated CLI-based SF equivalence checks and sequential process benchmarks."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
from itertools import islice
import json
import math
import os
from pathlib import Path
import platform
import shutil
import shlex
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import warnings

import yaml


ROOT = Path(__file__).resolve().parents[1]
THREADS = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
# Exact paths only: do not suppress arbitrary keys named time, seed, runtime, etc.
EXCLUDED_PATHS = {("metadata", "created_utc")}
SOURCE_FILES = (
    "src/hs_observatory/cli.py",
    "src/hs_observatory/analyze.py",
    "src/hs_observatory/config.py",
    "src/hs_observatory/analysis/structure_function.py",
    "src/hs_observatory/analysis/structure_function_parallel.py",
    "src/hs_observatory/analysis/elsasser.py",
    "src/hs_observatory/analysis/derived/elsasser.py",
    "src/hs_observatory/analysis/derived/pairs.py",
    "src/hs_observatory/analysis/derived/geometry_pairs.py",
)
MISSING = object()


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="analysis/malara-prod-600_sf-short-mc-v00.yaml")
    parser.add_argument("--workers", nargs="+", type=positive_int, default=[1, 2, 4, 8])
    parser.add_argument("--repetitions", type=positive_int, default=3)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/sf_parallel_benchmark")
    parser.add_argument(
        "--cli", type=Path, help="hsobs-analyze executable (defaults to project venv/PATH)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print plans without writing or launching"
    )
    parser.add_argument(
        "--rerun", action="store_true", help="make new attempts; retain previous artifacts"
    )
    parser.add_argument("--allow-oversubscription", action="store_true")
    parser.add_argument("--sample-interval", type=float, default=0.05, metavar="SECONDS")
    parser.add_argument("--max-differences", type=positive_int, default=20)
    args = parser.parse_args(argv)
    if len(set(args.workers)) != len(args.workers):
        parser.error("--workers must not contain duplicates")
    if not math.isfinite(args.sample_interval) or args.sample_interval <= 0:
        parser.error("--sample-interval must be finite and positive")
    return args


def resolve_project(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def inside(root, path):
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"benchmark path escapes output directory: {path}")
    return path


def load_config(path):
    config = yaml.safe_load(path.read_text())
    if (
        not isinstance(config, dict)
        or config.get("analysis", {}).get("method") != "structure_function"
    ):
        raise ValueError("benchmark requires an analysis.method: structure_function YAML")
    input_path = resolve_project(config["input"]["path"])
    if not input_path.is_file():
        raise ValueError(f"input does not exist: {input_path}")
    return config, input_path


def isolated_config(config, input_path, directory):
    result = deepcopy(config)
    result["input"]["path"] = str(input_path)
    result["output"]["path"] = str(directory / "result.json")
    # Match the workflow's shallow options/section merge, including plots in options.
    settings = deepcopy(result["analysis"].get("options", {}))
    settings.update(result.get("structure_function", {}))
    plot = settings.setdefault("plot", {})
    plot["path"] = str(directory / "diagnostic.png")
    plot.setdefault("publication", {})["path"] = str(directory / "publication.png")
    result["structure_function"] = settings
    return result


def protect_production_paths(root, config_path, input_path, config):
    settings = dict(config["analysis"].get("options", {}))
    settings.update(config.get("structure_function", {}))
    plot = settings.get("plot", {})
    protected = [config_path, input_path, resolve_project(config["output"]["path"])]
    protected += [
        resolve_project(plot.get("path", "structure_function_diagnostic.png")),
        resolve_project(plot.get("publication", {}).get("path", "sf_summary_p09.png")),
    ]
    if any(path.is_relative_to(root) for path in protected):
        raise ValueError(
            "benchmark output directory contains a production input/config/output path"
        )
    if root.exists() and any(root.iterdir()) and not (root / "manifest.json").is_file():
        raise ValueError("refusing nonempty output directory without a benchmark manifest")


def available_cpus():
    affinity = (
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    )
    allocation = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocation:
        try:
            return min(affinity, max(1, int(allocation)))
        except ValueError:
            warnings.warn("invalid SLURM_CPUS_PER_TASK; using CPU affinity", stacklevel=2)
    return affinity


def planned_workers(requested, capacity, allow=False):
    result = []
    for workers in sorted(set([1, *requested])):
        if workers > capacity:
            warnings.warn(
                f"{workers} workers exceed detected allocation of {capacity} CPUs; "
                + ("oversubscription explicitly allowed" if allow else "skipping"),
                stacklevel=2,
            )
            if not allow:
                continue
        result.append(workers)
    return result


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_signature(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def analysis_environment(cli):
    """Probe the entrypoint's Python, which may differ from this runner's Python."""
    with cli.open("rb") as stream:
        shebang = stream.readline(4096).decode(errors="replace").strip()
    words = shlex.split(shebang[2:]) if shebang.startswith("#!") else []
    if not words or "python" not in Path(words[0]).name:
        return {"status": "unidentified interpreter", "shebang": shebang}
    code = (
        "import sys, json, importlib.metadata as m; "
        "names=('numpy','scipy','h5py','PyYAML','matplotlib'); "
        "print(json.dumps({'python':sys.version,'executable':sys.executable,"
        "'packages':{n:m.version(n) for n in names}}))"
    )
    try:
        result = subprocess.run(
            [*words, "-c", code],
            env={**os.environ, **THREADS},
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)
    except (OSError, ValueError, subprocess.CalledProcessError):
        return {"status": "interpreter environment probe unavailable", "shebang": shebang}


def manifest(config, input_path, cli):
    versions = {}
    for package in ("numpy", "scipy", "h5py", "PyYAML", "matplotlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable in runner environment"
    return {
        "schema_version": 1,
        "config": config,
        "input_path": str(input_path),
        "input_sha256": sha256(input_path),
        "cli": str(cli),
        "cli_sha256": sha256(cli),
        "analysis_environment": analysis_environment(cli),
        "runner_sha256": sha256(Path(__file__)),
        "source_sha256": {name: sha256(ROOT / name) for name in SOURCE_FILES},
        "runner_python": sys.version,
        "runner_executable": sys.executable,
        "runner_package_versions": versions,
        "threads": THREADS,
    }


def atomic_text(path, value):
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}-")
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(value)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


@contextmanager
def benchmark_directory(root, expected_manifest):
    if root.exists() and any(root.iterdir()) and not (root / "manifest.json").is_file():
        raise ValueError("refusing nonempty output directory without a benchmark manifest")
    root.mkdir(parents=True, exist_ok=True)
    lock = inside(root, root / ".benchmark.lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise ValueError(
            "benchmark directory is locked; inspect .benchmark.lock before retrying"
        ) from error
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(f"{os.getpid()}\n")
        path = root / "manifest.json"
        if path.exists():
            if json.loads(path.read_text()) != expected_manifest:
                raise ValueError("benchmark manifest changed; choose a new --output-dir")
        else:
            write_json(path, expected_manifest)
        yield
    finally:
        lock.unlink(missing_ok=True)


def pointer(path):
    return "/" + "/".join(str(part).replace("~", "~0").replace("/", "~1") for part in path)


def displayed(value):
    if value is MISSING:
        return "<missing>"
    text = repr(value)
    return text if len(text) <= 160 else text[:157] + "..."


def differences(baseline, actual, path=()):
    if path in EXCLUDED_PATHS:
        return
    mismatch = None
    if type(baseline) is not type(actual):
        mismatch = "type/schema"
    elif isinstance(baseline, dict):
        for key in sorted(set(baseline) | set(actual)):
            yield from differences(
                baseline.get(key, MISSING), actual.get(key, MISSING), (*path, key)
            )
        return
    elif isinstance(baseline, list):
        if len(baseline) != len(actual):
            yield {
                "path": pointer(path),
                "kind": "length",
                "baseline": len(baseline),
                "actual": len(actual),
            }
        for index, (left, right) in enumerate(zip(baseline, actual)):
            yield from differences(left, right, (*path, index))
        return
    elif isinstance(baseline, float) and math.isnan(baseline) and math.isnan(actual):
        return
    elif baseline != actual:
        mismatch = "value"
    if mismatch:
        yield {
            "path": pointer(path),
            "kind": mismatch,
            "baseline": displayed(baseline),
            "actual": displayed(actual),
        }


def ensemble_statistics(result):
    """Validate plotting statistics from saved members, outside timed analysis."""
    import numpy as np

    observed = result["products"].get("observed")
    if not observed:
        return {}
    arrays = {"counts": observed["counts"]}
    for field in ("z_plus", "z_minus"):
        product = observed[field]
        arrays[f"{field}/S_n"] = product["S_n"]
        if product["flatness"] is not None:
            arrays[f"{field}/flatness"] = product["flatness"]
        for fit in ("raw", "normalized_by_order2"):
            arrays[f"{field}/{fit}"] = product["scaling_exponents"][fit]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return {
            key: np.nanpercentile(np.asarray(value, dtype=float), [5, 50, 95], axis=0).tolist()
            for key, value in arrays.items()
        }


def statistics_payload(result):
    try:
        return ensemble_statistics(result)
    except (KeyError, TypeError, ValueError, IndexError) as error:
        # Keep the raw result comparable so schema failures still get JSON paths.
        return {"_derivation_error": f"{type(error).__name__}: {error}"}


def process_group_rss(pgid, proc=Path("/proc")):
    """Sample simultaneous group RSS, including spawned descendants (Linux)."""
    if not sys.platform.startswith("linux") or not proc.is_dir():
        return None
    total, seen = 0, False
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            # stat field 5 is pgrp; field 24 is resident pages. Names can contain ')'.
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == pgid:
                total += int(fields[21]) * os.sysconf("SC_PAGE_SIZE")
                seen = True
        except (OSError, ValueError, IndexError):
            continue
    return total if seen else None


def stop_group(process):
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.terminate()


def run_process(command, directory, interval):
    environment = os.environ.copy()
    environment.update(THREADS)
    environment.update(MPLBACKEND="Agg", MPLCONFIGDIR=str(directory / "matplotlib-cache"))
    start = time.perf_counter()
    measurements = {"peak": None, "samples": 0, "error": ""}
    stop = threading.Event()
    process = None
    monitor = None

    def sample():
        try:
            while not stop.is_set():
                measured = process_group_rss(process.pid)
                if measured is not None:
                    measurements["peak"] = max(measurements["peak"] or 0, measured)
                    measurements["samples"] += 1
                stop.wait(interval)
        except OSError as error:
            measurements["error"] = str(error)

    try:
        with (directory / "analysis.log").open("wb") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=os.name == "posix",
            )
            monitor = threading.Thread(target=sample, daemon=True)
            monitor.start()
            exit_status = process.wait()
            elapsed = time.perf_counter() - start
    finally:
        stop.set()
        if monitor is not None:
            monitor.join()
        if process is not None:
            stop_group(process)
            process.wait()
    return {
        "wall_seconds": elapsed,
        "exit_status": exit_status,
        "peak_aggregate_rss_bytes": measurements["peak"],
        "memory_samples": measurements["samples"],
        "memory_error": measurements["error"],
        "memory_method": "linux_procfs_sampled_process_group_rss"
        if measurements["samples"]
        else "unavailable",
        "sample_interval_seconds": interval,
    }


def latest_attempt(root, workers, repetition):
    parent = inside(root, root / "runs" / f"workers-{workers}" / f"rep-{repetition}")
    attempts = sorted(parent.glob("attempt-*")) if parent.exists() else []
    for directory in reversed(attempts):
        inside(root, directory)
        record = inside(root, directory / "execution.json")
        if record.is_file():
            return directory, json.loads(record.read_text())
    return None


def new_attempt(root, workers, repetition):
    parent = inside(root, root / "runs" / f"workers-{workers}" / f"rep-{repetition}")
    parent.mkdir(parents=True, exist_ok=True)
    number = 1
    while (directory := parent / f"attempt-{number:06d}").exists():
        number += 1
    directory.mkdir()
    return directory


def execute_run(args, config, input_path, cli, workers, repetition, signature):
    directory = new_attempt(args.output_dir, workers, repetition)
    run_config = isolated_config(config, input_path, directory)
    config_path = directory / "analysis.yaml"
    config_path.write_text(yaml.safe_dump(run_config, sort_keys=False))
    command = [str(cli), str(config_path), "--workers", str(workers)]
    record = {
        "workers": workers,
        "repetition": repetition,
        "command": command,
        "cwd": str(ROOT),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "allocated_cpus": available_cpus(),
        "exit_status": None,
        "wall_seconds": None,
        "error": "",
        "peak_aggregate_rss_bytes": None,
        "memory_method": "unavailable",
    }
    try:
        if file_signature(input_path) != signature:
            raise ValueError("input file changed since manifest hashing")
        record.update(run_process(command, directory, args.sample_interval))
        if file_signature(input_path) != signature:
            raise ValueError("input file changed during analysis")
        if record["exit_status"] == 0:
            output = directory / "result.json"
            result = json.loads(output.read_text())
            if result.get("method") != "structure_function" or not result.get("products"):
                raise ValueError("CLI did not produce an SF result with member products")
            stats = statistics_payload(result)
            write_json(directory / "ensemble_statistics.json", stats)
            record["result_sha256"] = sha256(output)
            record["ensemble_statistics_sha256"] = sha256(directory / "ensemble_statistics.json")
        else:
            record["error"] = (
                f"analysis exited with status {record['exit_status']}; see analysis.log"
            )
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
    write_json(directory / "execution.json", record)
    return directory, record


def load_result(directory, record):
    if record.get("error") or record["exit_status"] != 0:
        raise ValueError(record.get("error") or "analysis failed")
    path = directory / "result.json"
    if sha256(path) != record["result_sha256"]:
        raise ValueError("saved result changed after completion")
    if sha256(directory / "ensemble_statistics.json") != record["ensemble_statistics_sha256"]:
        raise ValueError("saved ensemble statistics changed after completion")
    return json.loads(path.read_text())


def evaluate_runs(records, max_differences):
    loaded, rows = {}, []
    for directory, record in records:
        row = dict(record, run_directory=str(directory), equivalence="failed", differences=[])
        row["speedup"] = row["efficiency"] = None
        try:
            loaded[directory] = load_result(directory, record)
        except (OSError, ValueError, KeyError) as error:
            row["error"] = str(error)
        rows.append(row)
    baseline = next(
        (row for row in rows if row["workers"] == 1 and Path(row["run_directory"]) in loaded), None
    )
    if baseline is not None:
        reference = loaded[Path(baseline["run_directory"])]
        reference_stats = statistics_payload(reference)
        for row in rows:
            directory = Path(row["run_directory"])
            if directory not in loaded:
                continue
            result = loaded[directory]
            actual_stats = statistics_payload(result)
            comparison = list(islice(differences(reference, result), max_differences + 1))
            if "_derivation_error" in actual_stats or "_derivation_error" in reference_stats:
                comparison.append(
                    {
                        "path": "/mc_ensemble_statistics",
                        "kind": "derivation",
                        "baseline": displayed(reference_stats),
                        "actual": displayed(actual_stats),
                    }
                )
            if len(comparison) <= max_differences:
                comparison.extend(
                    islice(
                        differences(
                            reference_stats,
                            actual_stats,
                            ("mc_ensemble_statistics",),
                        ),
                        max_differences + 1 - len(comparison),
                    )
                )
            row["differences_truncated"] = len(comparison) > max_differences
            row["differences"] = comparison[:max_differences]
            row["equivalence"] = "mismatch" if comparison else "exact"
    else:
        for row in rows:
            if Path(row["run_directory"]) in loaded:
                row["equivalence"] = "baseline_unavailable"
    for row in rows:
        matched = next(
            (
                candidate
                for candidate in rows
                if candidate["workers"] == 1
                and candidate["repetition"] == row["repetition"]
                and candidate["equivalence"] == "exact"
            ),
            None,
        )
        if matched and row["equivalence"] == "exact" and row["wall_seconds"]:
            row["speedup"] = matched["wall_seconds"] / row["wall_seconds"]
            row["efficiency"] = row["speedup"] / row["workers"]
    return rows


def summary(rows):
    times = {}
    for row in rows:
        if row["equivalence"] == "exact":
            times.setdefault(row["workers"], []).append(row["wall_seconds"])
    baseline = statistics.median(times[1]) if times.get(1) else None
    result = []
    for workers, values in sorted(times.items()):
        median = statistics.median(values)
        speedup = baseline / median if baseline is not None and median else None
        result.append(
            {
                "workers": workers,
                "valid_runs": len(values),
                "median_seconds": median,
                "speedup": speedup,
                "efficiency": speedup / workers if speedup is not None else None,
            }
        )
    return result


def write_reports(root, rows):
    columns = [
        "workers",
        "repetition",
        "wall_seconds",
        "peak_aggregate_rss_bytes",
        "memory_method",
        "memory_samples",
        "sample_interval_seconds",
        "memory_error",
        "exit_status",
        "equivalence",
        "speedup",
        "efficiency",
        "error",
        "run_directory",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(root / "benchmark_results.csv", buffer.getvalue())
    aggregates = summary(rows)
    report = [
        "# SF parallelism benchmark",
        "",
        "Times are end-to-end CLI wall seconds, including startup, preparation, evaluation, "
        "fitting, serialization and enabled plotting. No phase timings are instrumented. "
        "JSON comparison and benchmark summary/plot generation are outside that timer.",
        "",
        "Memory is the maximum sampled sum of RSS across the analysis process group, including "
        "spawned workers and helpers, excluding this runner. Linux /proc samples may miss short "
        "peaks; shared pages are counted per process and reads are not atomic. This is sampled "
        "aggregate RSS, not cgroup memory. Unavailable measurements remain blank, never parent-only RSS. "
        "The sampling interval and sample count are in each execution.json.",
        "",
        "Runs are sequential in repetition then worker order. Input hashing warms filesystem caches; "
        "no cache flushing is performed. Summaries use only exact, successful runs. Speedup is the "
        "ratio of median serial time to median worker time; efficiency is speedup/workers. CSV ratios "
        "use the serial run of the same repetition.",
        "",
        "Comparison is recursive and exact, including types, array order, NaN masks and deterministic "
        "metadata. The only excluded result path is `/metadata/created_utc`. Separate observed "
        "5th/50th/95th percentile checks use saved counts, S_n, flatness and exponents; these are "
        "validation postprocessing, not new SF estimators. Full provenance is in manifest.json.",
        "",
        "| Workers | Exact runs | Median seconds | Speedup | Efficiency |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in aggregates:
        report.append(
            f"| {item['workers']} | {item['valid_runs']} | {item['median_seconds']:.6g} "
            f"| {item['speedup']:.4g} | {item['efficiency']:.4g} |"
        )
    report += ["", "## Individual runs", ""]
    for row in rows:
        report.append(
            f"- workers={row['workers']}, repetition={row['repetition']}: "
            f"{row['equivalence']}, exit={row['exit_status']}; `{row['run_directory']}`"
        )
        if row.get("error"):
            report.append(f"  Error: {row['error']}")
        if row.get("memory_error"):
            report.append(
                f"  Memory sampling error (measurement incomplete): {row['memory_error']}"
            )
        for difference in row["differences"]:
            report.append(
                f"  - `{difference['path']}` ({difference['kind']}): "
                f"baseline={difference['baseline']!r}; actual={difference['actual']!r}"
            )
        if row.get("differences_truncated"):
            report.append("  Additional differences omitted; increase --max-differences.")
        write_json(
            Path(row["run_directory"]) / "validation.json",
            {key: row[key] for key in ("workers", "repetition", "equivalence", "differences")},
        )
    atomic_text(root / "benchmark_report.md", "\n".join(report) + "\n")
    scaling_plot(root, aggregates, rows)


def scaling_plot(root, aggregates, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    axes[0].plot(
        [a["workers"] for a in aggregates], [a["median_seconds"] for a in aggregates], "o-"
    )
    axes[0].set_ylabel("Median end-to-end wall time [s]")
    axes[1].plot([a["workers"] for a in aggregates], [a["speedup"] for a in aggregates], "o-")
    axes[1].set_ylabel("Speedup (ratio of median times)")
    for row in rows:
        if row.get("peak_aggregate_rss_bytes") is not None:
            axes[2].plot(row["workers"], row["peak_aggregate_rss_bytes"] / 2**20, "o", color="C0")
    axes[2].set_ylabel("Sampled peak aggregate RSS [MiB]")
    if not aggregates:
        axes[0].text(0.5, 0.5, "No validated runs", ha="center", transform=axes[0].transAxes)
    for axis in axes:
        axis.set_xlabel("Workers")
        axis.grid(alpha=0.25)
    descriptor, temporary = tempfile.mkstemp(dir=root, suffix=".png")
    os.close(descriptor)
    try:
        figure.savefig(temporary, dpi=150)
        os.replace(temporary, root / "benchmark_scaling.png")
    finally:
        plt.close(figure)
        Path(temporary).unlink(missing_ok=True)


def main(argv=None):
    args = arguments(argv)
    try:
        args.output_dir = args.output_dir.expanduser().resolve()
        config_path = resolve_project(args.config)
        config, input_path = load_config(config_path)
        protect_production_paths(args.output_dir, config_path, input_path, config)
        cli = args.cli or ROOT / ".venv/bin/hsobs-analyze"
        if not cli.is_file() and args.cli is None:
            cli = Path(shutil.which("hsobs-analyze") or str(cli))
        cli = cli.expanduser().absolute()  # Preserve the venv entrypoint/shebang.
        if not cli.is_file() or not os.access(cli, os.X_OK):
            raise ValueError(f"hsobs-analyze executable not found: {cli}; use --cli")
        capacity = available_cpus()
        workers = planned_workers(args.workers, capacity, args.allow_oversubscription)
        if config.get("structure_function", {}).get("data_groups") == ["truth"]:
            warnings.warn(
                "truth-only selection has one task; member parallelism cannot speed it up",
                stacklevel=2,
            )
        print(f"Config: {config_path}; input: {input_path}; CLI: {cli}")
        for repetition in range(1, args.repetitions + 1):
            for count in workers:
                print(
                    f"workers={count} repetition={repetition}: {args.output_dir}/runs/workers-{count}/rep-{repetition}"
                )
        if args.dry_run:
            print(
                "Dry run: no writes, input hashing, or analysis launches; baseline 1 is always included. "
                "Completed attempts will be reused unless --rerun; manifest compatibility is checked on execution."
            )
            return 0
        signature = file_signature(input_path)
        expected = manifest(config, input_path, cli)
        if file_signature(input_path) != signature:
            raise ValueError("input changed during manifest hashing")
        records = []
        with benchmark_directory(args.output_dir, expected):
            try:
                for repetition in range(1, args.repetitions + 1):
                    for count in workers:
                        existing = latest_attempt(args.output_dir, count, repetition)
                        if existing and not args.rerun:
                            print(f"Reusing workers={count} repetition={repetition}: {existing[0]}")
                            records.append(existing)
                        else:
                            records.append(
                                execute_run(
                                    args, config, input_path, cli, count, repetition, signature
                                )
                            )
                        rows = evaluate_runs(records, args.max_differences)
                        write_reports(args.output_dir, rows)
            except KeyboardInterrupt:
                if records:
                    write_reports(args.output_dir, evaluate_runs(records, args.max_differences))
                raise
        failed = any(row["equivalence"] != "exact" for row in rows)
        for row in rows:
            for difference in row["differences"]:
                print(
                    f"MISMATCH workers={row['workers']} repetition={row['repetition']} "
                    f"{difference['path']}: {difference['baseline']} != {difference['actual']}",
                    file=sys.stderr,
                )
        return 1 if failed else 0
    except (OSError, ValueError, KeyError) as error:
        print(f"Benchmark error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
