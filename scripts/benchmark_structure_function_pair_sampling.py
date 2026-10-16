"""Bounded, fresh-process comparison; never runs generation or the MC ensemble.

Run from the repository root with PYTHONPATH=src. Only sampled truth is used.
The JSON output contains timings, memory, counts and accuracy against a census.
"""

import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np

from hs_observatory.analysis import structure_function as sf
from hs_observatory.analysis.derived.geometry_pairs import GeometryPlanCache
from hs_observatory.analysis.elsasser import compute_elsasser_analysis


def worker(args):
    elsasser = compute_elsasser_analysis(args.input, {"data_group": "truth"})
    p, _, _ = sf._positions_for_group_member(
        args.input,
        "truth",
        0,
        elsasser["time"],
        len(elsasser["spacecraft_ids"]),
    )
    p = p[None, : args.n_times]
    zp = elsasser["z_plus"][None, : args.n_times]
    zm = elsasser["z_minus"][None, : args.n_times]
    if p.shape[1] != args.n_times:
        raise ValueError("n-times exceeds the available aligned time series")
    edges = np.geomspace(70.0, 5000.0, 10)
    settings = dict(
        mode=args.worker,
        chunk_size=50000,
        n_pairs=None,
        max_pairs=None,
        target_pairs_per_bin=args.target,
        max_candidates=args.max_candidates,
        binning_n_pairs=100000,
        random_seed=args.seed,
    )

    def calculate(seed):
        return sf.compute_all_spacetime_structure_functions(
            p,
            zp,
            zm,
            list(range(1, 7)),
            edges,
            pair_sampling=dict(settings, random_seed=seed),
        )

    baseline_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    durations = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        result = calculate(args.seed)
        durations.append(time.perf_counter() - start)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    # Separate instrumentation pass: avoid charging timer-wrapper overhead to
    # the above kernel benchmark. Lazy sampling time is charged at next().
    # Exhaustive iterator timing is not instrumented by these selector wrappers.
    stages = dict(
        pair_selection_seconds=None if args.worker == "all" else 0.0, moment_power_sum_seconds=0.0
    )
    adaptive, geometry = sf.adaptive_stratified_pairs, sf.geometry_pair_chunks
    moment, get_plan = sf.structure_function_moment_sums, GeometryPlanCache.get

    def timed(fn, key):
        def call(*a, **kw):
            start = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                stages[key] += time.perf_counter() - start

        return call

    def timed_chunks(*a, **kw):
        iterator = geometry(*a, **kw)
        while True:
            start = time.perf_counter()
            try:
                chunk = next(iterator)
            except StopIteration:
                stages["pair_selection_seconds"] += time.perf_counter() - start
                return
            stages["pair_selection_seconds"] += time.perf_counter() - start
            yield chunk

    sf.adaptive_stratified_pairs = timed(adaptive, "pair_selection_seconds")
    sf.geometry_pair_chunks = timed_chunks
    sf.structure_function_moment_sums = timed(moment, "moment_power_sum_seconds")
    GeometryPlanCache.get = timed(get_plan, "pair_selection_seconds")
    try:
        start = time.perf_counter()
        calculate(args.seed)
        stages["total_seconds"] = time.perf_counter() - start
    finally:
        sf.adaptive_stratified_pairs, sf.geometry_pair_chunks = adaptive, geometry
        sf.structure_function_moment_sums, GeometryPlanCache.get = moment, get_plan
    estimates = [result["z_plus"]["S_n"][0]]
    if args.worker != "all":
        estimates.extend(
            calculate(args.seed + i)["z_plus"]["S_n"][0] for i in range(1, args.accuracy_seeds)
        )
    return sf._to_serializable(
        dict(
            mode=args.worker,
            seconds=durations,
            median_seconds=float(np.median(durations)),
            baseline_peak_rss_mib=baseline_rss,
            peak_rss_mib=peak_rss,
            stages=stages,
            counts=result["counts"],
            diagnostics=result["spacetime_pair_counts"],
            lambda_bin_edges_km=edges,
            z_plus_estimates=estimates,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="outputs/malara-prod-400.h5")
    parser.add_argument("--n-times", type=int, default=256)
    parser.add_argument("--target", type=int, default=2000)
    parser.add_argument("--max-candidates", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--accuracy-seeds", type=int, default=5)
    parser.add_argument("--output", default="/tmp/structure-function-sampling-benchmark.json")
    parser.add_argument(
        "--worker",
        choices=["all", "adaptive_stratified", "geometry_stratified"],
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if (
        not 2 <= args.n_times <= 512
        or args.target < 1
        or args.max_candidates < 1
        or not 1 <= args.repeats <= 10
        or not 1 <= args.accuracy_seeds <= 20
    ):
        parser.error(
            "bounded benchmark requires 2–512 times, positive budgets, "
            "1–10 repeats and 1–20 accuracy seeds"
        )
    if args.worker:
        print(json.dumps(worker(args)))
        return
    records = {}
    for mode in ("all", "adaptive_stratified", "geometry_stratified"):
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker", mode],
            check=True,
            capture_output=True,
            text=True,
        )
        records[mode] = json.loads(completed.stdout)
        print(
            f"{mode}: {records[mode]['median_seconds']:.4f} s; "
            f"{records[mode]['peak_rss_mib']:.1f} MiB peak RSS",
            flush=True,
        )
    reference = np.array(records["all"]["z_plus_estimates"])[0]
    for mode in ("adaptive_stratified", "geometry_stratified"):
        relative = np.array(records[mode]["z_plus_estimates"]) / reference - 1
        records[mode]["relative_rmse_by_order"] = np.sqrt(np.mean(relative**2, axis=(0, 2)))
    report = dict(
        n_times=args.n_times,
        target_per_bin=args.target,
        seed=args.seed,
        accuracy_seeds=args.accuracy_seeds,
        numpy_version=np.__version__,
        records=records,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(sf._to_serializable(report), indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
