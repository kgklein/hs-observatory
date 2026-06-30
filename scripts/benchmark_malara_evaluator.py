#!/usr/bin/env python3
"""Benchmark the low-level serial Malara turbulence evaluator."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from hs_observatory.fields.malara import MalaraConfig, MalaraTurbulence


def make_trajectory_like_positions(
    n_points: int,
    rng: np.random.Generator,
    n_spacecraft: int = 9,
) -> np.ndarray:
    """Return a smooth normalized trajectory sampled by a compact formation.

    Points are ordered time-major, with all spacecraft at one time adjacent in
    the flattened array. The first spacecraft is the hub at the formation
    center; the other spacecraft retain fixed small offsets while the center
    follows a straight path through the periodic unit domain. This is a cache
    locality surrogate, not a reconstruction of a particular HelioSwarm orbit.
    """

    if n_points <= 0:
        raise ValueError("n_points must be positive")
    if n_spacecraft <= 0:
        raise ValueError("n_spacecraft must be positive")

    n_times = int(np.ceil(n_points / n_spacecraft))
    start = rng.uniform(0.0, 1.0, size=3)
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    distance = np.linspace(0.0, 0.75, n_times)
    centers = start + distance[:, np.newaxis] * direction

    offsets = rng.normal(size=(n_spacecraft, 3))
    offsets[0] = 0.0  # Hub
    maximum_offset = float(np.max(np.linalg.norm(offsets, axis=1)))
    if maximum_offset > 0.0:
        offsets *= 0.02 / maximum_offset

    positions = centers[:, np.newaxis, :] + offsets[np.newaxis, :, :]
    return np.mod(positions, 1.0).reshape(-1, 3)[:n_points]


def _comma_separated_numbers(value: str, converter: type) -> list[Any]:
    try:
        return [converter(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid comma-separated list: {value}") from error


def _print_evaluations(rows: list[dict[str, Any]]) -> None:
    print("\nEvaluator backends:")
    print(
        "backend             points scales   p  chunk workers  seconds "
        " points/s  us/point   max_abs_diff entries hit_rate"
    )
    for row in rows:
        chunk = "-" if row["chunk_size"] is None else str(row["chunk_size"])
        workers = "-" if row["n_workers"] is None else str(row["n_workers"])
        entries = "-" if row["cache_entries"] is None else str(row["cache_entries"])
        hit_rate = (
            "-" if row["cache_hit_rate"] is None else f"{row['cache_hit_rate']:.3f}"
        )
        print(
            f"{row['backend']:<18s} {row['n_points']:>6d} "
            f"{row['n_scales']:>6d} {row['intermittency_p']:>3.1f} "
            f"{chunk:>6s} {workers:>7s} "
            f"{row['elapsed_time_seconds']:>8.4f} "
            f"{row['points_per_second']:>9.1f} "
            f"{row['microseconds_per_point']:>9.1f} "
            f"{row['max_abs_diff_from_direct']:>14.3e} "
            f"{entries:>7s} {hit_rate:>8s}"
        )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "benchmark",
        "backend",
        "position_mode",
        "n_points",
        "n_scales",
        "intermittency_p",
        "chunk_size",
        "n_workers",
        "elapsed_time_seconds",
        "points_per_second",
        "microseconds_per_point",
        "max_abs_diff_from_direct",
        "cache_entries",
        "cache_hit_rate",
        "total_eddy_touches",
        "n_unique_eddies",
        "reuse_factor",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-points",
        type=int,
        default=100,
        help="largest case to run (default: 100; use 10000 for the full suite)",
    )
    parser.add_argument(
        "--point-counts",
        default="100,1000,10000,100000",
        help="comma-separated point counts, filtered by --max-points",
    )
    parser.add_argument(
        "--scales",
        default="4,6,8,10",
        help="comma-separated maximum zero-based scale indices",
    )
    parser.add_argument(
        "--p-values",
        default="0.5,0.7",
        help="comma-separated intermittency values",
    )
    parser.add_argument(
        "--backends",
        default="direct,cached,parallel",
        help=(
            "comma-separated direct, chunked, cached, parallel, or "
            "parallel-uncached"
        ),
    )
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument(
        "--chunk-sizes",
        help="optional comma-separated chunk sizes for chunked/parallel backends",
    )
    parser.add_argument("--n-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument(
        "--trajectory-like",
        action="store_true",
        help=(
            "sample a smooth nine-spacecraft formation trajectory instead of "
            "independent random points"
        ),
    )
    parser.add_argument("--csv", type=Path, help="optionally write all rows to CSV")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.max_points <= 0:
        raise SystemExit("--max-points must be positive")

    requested_points = _comma_separated_numbers(args.point_counts, int)
    point_counts = sorted(
        {value for value in requested_points if 0 < value <= args.max_points}
    )
    if not point_counts:
        point_counts = [args.max_points]
    scales = sorted(set(_comma_separated_numbers(args.scales, int)))
    p_values = sorted(set(_comma_separated_numbers(args.p_values, float)))
    backends = [item.strip() for item in args.backends.split(",") if item.strip()]
    supported_backends = {
        "direct",
        "chunked",
        "cached",
        "parallel",
        "parallel-uncached",
    }
    unknown_backends = sorted(set(backends) - supported_backends)
    if not backends or unknown_backends:
        raise SystemExit(
            "--backends contains unsupported values: " + ", ".join(unknown_backends)
        )
    chunk_sizes = [args.chunk_size]
    if args.chunk_sizes:
        chunk_sizes = sorted(
            set(_comma_separated_numbers(args.chunk_sizes, int))
        )
    if not scales or min(scales) < 2:
        raise SystemExit("--scales must contain values >= injection scale 2")
    if not p_values or any(not 0.5 <= value <= 1.0 for value in p_values):
        raise SystemExit("--p-values must contain values in [0.5, 1.0]")
    if any(value <= 0 for value in chunk_sizes):
        raise SystemExit("chunk sizes must be positive integers")
    if args.n_workers <= 0:
        raise SystemExit("--n-workers must be positive")

    rng = np.random.default_rng(args.seed)
    position_mode = "trajectory-like" if args.trajectory_like else "random"
    if args.trajectory_like:
        all_positions = make_trajectory_like_positions(max(point_counts), rng)
    else:
        all_positions = rng.uniform(0.0, 1.0, size=(max(point_counts), 3))
    evaluation_rows: list[dict[str, Any]] = []
    diagnostic_case: tuple[MalaraTurbulence, np.ndarray] | None = None

    print("Malara evaluator benchmark")
    print("--------------------------")
    print(f"Position sampling: {position_mode}")
    for p_value in p_values:
        for n_scales in scales:
            config = MalaraConfig(
                n_scales=n_scales,
                injection_scale_index=2,
                intermittency_p=p_value,
                base_seed=12345,
                realization=1,
            )
            evaluator = MalaraTurbulence(config)
            for n_points in point_counts:
                positions = all_positions[:n_points]
                start = perf_counter()
                direct_values = evaluator.evaluate_vector(positions)
                direct_elapsed = perf_counter() - start

                for backend in backends:
                    sizes = (
                        chunk_sizes
                        if backend in {"chunked", "parallel", "parallel-uncached"}
                        else [None]
                    )
                    for chunk_size in sizes:
                        cache_entries = None
                        cache_hit_rate = None
                        if backend == "direct":
                            values = direct_values
                            elapsed = direct_elapsed
                        else:
                            optimized = MalaraTurbulence(config)
                            start = perf_counter()
                            if backend == "cached":
                                values = optimized.evaluate_vector_cached(positions)
                            elif backend == "chunked":
                                values = optimized.evaluate_vector_chunked(
                                    positions, chunk_size
                                )
                            else:
                                values = optimized.evaluate_vector_parallel(
                                    positions,
                                    chunk_size=chunk_size,
                                    n_workers=args.n_workers,
                                    use_cache=backend == "parallel",
                                )
                            elapsed = perf_counter() - start
                            if backend == "cached":
                                cache = optimized.cache_info()
                                cache_entries = cache["n_entries"]
                                cache_hit_rate = cache["hit_rate"]

                        evaluation_rows.append(
                            {
                                "benchmark": "evaluation",
                                "backend": backend,
                                "position_mode": position_mode,
                                "n_points": n_points,
                                "n_scales": n_scales,
                                "intermittency_p": p_value,
                                "chunk_size": chunk_size,
                                "n_workers": (
                                    args.n_workers
                                    if backend in {"parallel", "parallel-uncached"}
                                    else None
                                ),
                                "elapsed_time_seconds": elapsed,
                                "points_per_second": n_points / elapsed,
                                "microseconds_per_point": 1.0e6 * elapsed / n_points,
                                "max_abs_diff_from_direct": float(
                                    np.max(np.abs(values - direct_values))
                                ),
                                "cache_entries": cache_entries,
                                "cache_hit_rate": cache_hit_rate,
                            }
                        )
                diagnostic_case = (evaluator, positions)

    _print_evaluations(evaluation_rows)
    assert diagnostic_case is not None
    evaluator, positions = diagnostic_case
    statistics = evaluator.collect_touched_eddy_statistics(positions)
    reuse_row = {
        "benchmark": "eddy_reuse",
        "backend": "diagnostic",
        "position_mode": position_mode,
        "n_points": statistics["n_points"],
        "n_scales": statistics["n_scales"],
        "intermittency_p": evaluator.config.intermittency_p,
        "total_eddy_touches": statistics["total_eddy_touches"],
        "n_unique_eddies": statistics["n_unique_eddies"],
        "reuse_factor": statistics["reuse_factor"],
    }
    print("\nEddy reuse diagnostics (largest requested case):")
    print("n_points  n_scales  total_touches  unique_eddies  reuse_factor")
    print(
        f"{reuse_row['n_points']:>8d}  {reuse_row['n_scales']:>8d}  "
        f"{reuse_row['total_eddy_touches']:>13d}  "
        f"{reuse_row['n_unique_eddies']:>13d}  "
        f"{reuse_row['reuse_factor']:>12.2f}"
    )

    if args.csv:
        _write_csv(args.csv, [*evaluation_rows, reuse_row])
        print(f"\nWrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
