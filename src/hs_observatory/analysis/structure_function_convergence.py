"""Truth-only fixed-budget convergence study using existing sampled fields.

No generation or observed-member reads occur. Saved moments can be refitted and
replotted without sampling again. Sampling-seed variability is never labeled MC.
"""

import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import h5py
import numpy as np
import yaml

from hs_observatory.analysis.derived.geometry_pairs import (
    ALGORITHM_VERSION,
    build_geometry_pair_plan,
    geometry_pair_chunks,
)
from hs_observatory.analysis.elsasser import compute_elsasser_analysis
from hs_observatory.analysis.structure_function import (
    FIELDS,
    P_MODEL_RUN_COLORS,
    _bin_indices,
    _fit_scaling_exponents,
    _positions_for_group_member,
    _settings,
    _to_serializable,
    compute_all_spacetime_structure_functions,
    plot_structure_function_curve,
    structure_function_moment_means,
    structure_function_moment_sums,
)
from hs_observatory.analysis.structure_function_local import local_log_slopes, p_model_exponents

ORDERS = list(range(1, 7))
FIT_WINDOWS = {"primary": [150.0, 1200.0], "sensitivity": [200.0, 1000.0]}


def stage2_edges():
    return np.r_[
        np.geomspace(70.0, 150.0, 5),
        np.geomspace(150.0, 1200.0, 15)[1:],
        np.geomspace(1200.0, 20000.0, 9)[1:],
    ]


def prefix_moments(plan, positions, fields, budgets, outside_target, seed, chunk_size):
    """Fixed-budget means from shared random prefixes; census handled separately.

    Returns sums (budget, field, order, bin), counts (budget, bin), and timings.
    Sampling never uses an increment-dependent stopping rule. If a bin reaches
    census, smaller targets use a separate random partial stream, not its
    deterministic ordered census prefix.
    """
    budgets = np.asarray(budgets, dtype=np.int64)
    if (
        budgets.ndim != 1
        or not len(budgets)
        or np.any(budgets <= 0)
        or np.any(np.diff(budgets) <= 0)
        or outside_target <= 0
    ):
        raise ValueError("positive increasing budgets and a positive outside target are required")
    centers = np.sqrt(plan.edges[:-1] * plan.edges[1:])
    inside = (centers >= 150) & (centers <= 1200)
    requested = np.where(inside[None], budgets[:, None], outside_target)
    desired = np.minimum(requested, plan.populations[None])
    partial = np.max(np.where(desired < plan.populations[None], desired, 0), axis=0)
    census = (plan.populations > 0) & np.any(desired == plan.populations[None], axis=0)
    sums = np.zeros((len(budgets), len(FIELDS), len(ORDERS), len(centers)))
    flat_positions = positions.reshape(-1, 3)
    values = [np.asarray(fields[name]).reshape(-1, 3) for name in FIELDS]
    timings = dict(sampling_seconds=0.0, accumulation_seconds=0.0, selected_unique_pairs=0)

    def consume(targets, is_census):
        running = np.zeros((len(FIELDS), len(ORDERS), len(centers)))
        seen = np.zeros(len(centers), dtype=np.int64)
        iterator = geometry_pair_chunks(plan, targets, random_seed=seed, chunk_size=chunk_size)
        while True:
            start = time.perf_counter()
            try:
                left, right = next(iterator)
            except StopIteration:
                timings["sampling_seconds"] += time.perf_counter() - start
                break
            timings["sampling_seconds"] += time.perf_counter() - start
            start = time.perf_counter()
            d = np.linalg.norm(flat_positions[right] - flat_positions[left], axis=1)
            bins, valid = _bin_indices(d, plan.edges)
            if not np.all(valid):
                raise RuntimeError("geometry plan emitted a pair outside its bins")
            for q in np.unique(bins):
                mask = bins == q
                magnitudes = [
                    np.linalg.norm(v[right[mask]] - v[left[mask]], axis=1) for v in values
                ]
                checkpoints = np.unique(desired[:, q]) if not is_census else [targets[q]]
                checkpoints = [int(c) for c in checkpoints if seen[q] < c <= seen[q] + mask.sum()]
                begin = 0
                for endpoint in [*checkpoints, int(seen[q] + mask.sum())]:
                    end = endpoint - int(seen[q])
                    if end > begin:
                        for field, magnitude in enumerate(magnitudes):
                            running[field, :, q] += structure_function_moment_sums(
                                magnitude[begin:end], ORDERS
                            )
                    if endpoint in checkpoints:
                        output = desired[:, q] == endpoint
                        if is_census:
                            output &= desired[:, q] == plan.populations[q]
                        sums[output, :, :, q] = running[None, :, :, q]
                    begin = end
                seen[q] += int(mask.sum())
            timings["accumulation_seconds"] += time.perf_counter() - start

    consume(partial, False)
    if np.any(census):
        consume(np.where(census, plan.populations, 0), True)
    timings["selected_unique_pairs"] = int(np.sum(np.max(desired, axis=0)))
    timings["evaluated_pairs"] = int(partial.sum() + plan.populations[census].sum())
    return sums, desired, requested, timings


def fit_saved(values, counts, centers, windows=FIT_WINDOWS):
    """Values (seed, budget, order, bin); output (seed, budget, window, order)."""
    exponents, slopes = [], []
    for b in range(values.shape[1]):
        result = _fit_scaling_exponents(values[:, b], counts[:, b], centers, ORDERS, windows, 3)
        exponents.append(result["raw"])
        slopes.append(local_log_slopes(centers, values[:, b], counts=counts[:, b])[1])
    return np.stack(exponents, axis=1), np.stack(slopes, axis=1)


def run_study(config_path, legacy_candidates=0, benchmark_only=False, benchmark_budget=None):
    started = time.perf_counter()
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())
    if config["analysis"]["method"] != "structure_function_truth_convergence":
        raise ValueError("use a dedicated structure_function_truth_convergence pilot config")
    study = config["stage2"]
    if study["data_group"] != "truth":
        raise ValueError("Stage 2 only permits ideal sampled truth")
    source = Path(config["input"]["path"])
    if not source.is_file():
        raise FileNotFoundError(f"Existing observations required; no generation fallback: {source}")
    settings = _settings(config["structure_function"])
    if settings["pair_sampling"]["mode"] != "geometry_stratified":
        raise ValueError("Stage 2 requires geometry_stratified")
    with h5py.File(source) as handle:
        generation = yaml.safe_load(handle["metadata"].attrs["config_yaml"])
        p = generation["field"]["turbulence"]["intermittency_p"]
        if not np.isclose(p, study["p_value"]):
            raise ValueError("input intermittency_p does not match the requested case")
    elsasser = compute_elsasser_analysis(source, dict(settings["elsasser"], data_group="truth"))
    positions, position_source, notes = _positions_for_group_member(
        source, "truth", 0, elsasser["time"], len(elsasser["spacecraft_ids"])
    )
    if study.get("geometry_reference"):
        reference, _, _ = _positions_for_group_member(
            study["geometry_reference"],
            "truth",
            0,
            elsasser["time"],
            len(elsasser["spacecraft_ids"]),
        )
        if not np.array_equal(positions, reference):
            raise ValueError("pilot trajectory differs from the configured geometry_reference")
    times = elsasser["time"]
    if len(times) != 2401 or not np.isclose(times[-1] - times[0], 300):
        raise ValueError("Stage 2 requires the original 2401 timestamps / 300 s duration")
    preparation = time.perf_counter() - started
    edges = np.array(settings["lambda_bins_km"])
    if not np.array_equal(edges, stage2_edges()):
        raise ValueError("Stage 2 requires the explicit 26-bin design")
    build_started = time.perf_counter()
    plan = build_geometry_pair_plan(
        positions, edges, min_pair_lambda_km=70.0, max_pair_lambda_km=20000.0
    )
    plan_seconds = time.perf_counter() - build_started
    seeds, budgets = study["seeds"], study["budgets"]
    if len(seeds) < 5 or len(set(seeds)) != len(seeds):
        raise ValueError("at least five distinct sampling seeds are required")
    if benchmark_only:
        seeds, budgets = seeds[:1], [benchmark_budget or budgets[-1]]
    sums, counts, timing = [], [], []
    for seed in seeds:
        s, c, requested, t = prefix_moments(
            plan,
            positions,
            elsasser,
            budgets,
            study["outside_target"],
            seed,
            settings["pair_sampling"]["chunk_size"],
        )
        sums.append(s)
        counts.append(c)
        timing.append(t)
        print(
            f"{study['run_id']} seed {seed}: selection {t['sampling_seconds']:.3f}s; "
            f"accumulation {t['accumulation_seconds']:.3f}s",
            flush=True,
        )
    sums, counts = np.array(sums), np.array(counts)
    centers = np.sqrt(edges[:-1] * edges[1:])
    products = {}
    for f, name in enumerate(FIELDS):
        values = np.stack(
            [structure_function_moment_means(s[:, f], c) for s, c in zip(sums, counts)]
        )
        xi, slopes = fit_saved(values, counts, centers, settings["scale_ranges_km"])
        products[name] = dict(S_n=values, xi_n=xi, local_slopes=slopes)
    legacy = None
    if legacy_candidates:
        start = time.perf_counter()
        old = compute_all_spacetime_structure_functions(
            positions[None],
            elsasser["z_plus"][None],
            elsasser["z_minus"][None],
            ORDERS,
            edges,
            pair_sampling=dict(
                mode="adaptive_stratified",
                target_pairs_per_bin=10000,
                max_candidates=legacy_candidates,
                random_seed=seeds[0],
                chunk_size=100000,
            ),
            min_pair_lambda_km=70.0,
            max_pair_lambda_km=20000.0,
            scale_ranges_km=settings["scale_ranges_km"],
        )
        legacy = dict(
            seconds=time.perf_counter() - start,
            counts=old["counts"],
            diagnostics=old["spacetime_pair_counts"],
        )
    output = Path(config["output"]["path"])
    if benchmark_only:
        suffix = f"_benchmark_{benchmark_budget}" if benchmark_budget else "_benchmark"
        output = output.with_name(output.stem + suffix + ".h5")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(
        run_id=study["run_id"],
        p_value=p,
        source=str(source.resolve()),
        benchmark_only=benchmark_only,
        source_size_bytes=source.stat().st_size,
        source_mtime_ns=source.stat().st_mtime_ns,
        truth_positions_sha256=hashlib.sha256(positions.tobytes()).hexdigest(),
        truth_fields_sha256={
            name: hashlib.sha256(elsasser[name].tobytes()).hexdigest() for name in FIELDS
        },
        data_group="truth",
        uncertainty="pair-subsampling seeds, not measurement-error MC",
        algorithm_version=ALGORITHM_VERSION,
        numpy_version=np.__version__,
        n_times=len(times),
        duration_s=float(times[-1] - times[0]),
        position_source=position_source,
        notes=notes,
        preparation_seconds=preparation,
        geometry_reference=study.get("geometry_reference"),
        plan_seconds=plan_seconds,
        candidate_groups=plan.candidate_groups,
        boundary_pairs_examined=plan.boundary_pairs_examined,
        seed_timings=timing,
        fit_windows=settings["scale_ranges_km"],
        legacy=legacy,
        config_yaml=config_path.read_text(),
    )
    with h5py.File(output, "w") as handle:
        for name, data in dict(
            edges_km=edges,
            centers_km=centers,
            orders=ORDERS,
            seeds=seeds,
            budgets=budgets,
            eligible=plan.populations,
            counts=counts,
            requested=requested,
            shortfalls=np.maximum(requested - plan.populations, 0),
            groups_per_bin=[len(b.groups) for b in plan.bins],
            time_s=times,
        ).items():
            handle.create_dataset(name, data=data)
        for name, fields in products.items():
            for key, data in fields.items():
                handle.create_dataset(f"{name}/{key}", data=data)
        metadata["total_study_seconds"] = time.perf_counter() - started
        metadata["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        handle.attrs["metadata_json"] = json.dumps(_to_serializable(metadata))
    print(output, flush=True)
    return output


def read_case(path):
    with h5py.File(path) as h:
        result = {
            name: h[name][:]
            for name in [
                "edges_km",
                "centers_km",
                "budgets",
                "seeds",
                "counts",
                "eligible",
                "requested",
                "groups_per_bin",
            ]
        }
        result["metadata"] = json.loads(h.attrs["metadata_json"])
        result["S_n"] = h["z_plus/S_n"][:]
    result["xi_n"], result["local_slopes"] = fit_saved(
        result["S_n"], result["counts"], result["centers_km"]
    )
    return result


def summarize(case):
    values, xi = case["S_n"], case["xi_n"]
    centers = case["centers_km"]
    fit = (centers >= 150) & (centers <= 1200)
    cv = values.std(axis=0, ddof=1) / values.mean(axis=0)
    relative_to_max = values / values[:, -1:] - 1
    return dict(
        metadata=case["metadata"],
        budgets=case["budgets"],
        eligible=case["eligible"],
        centers_km=centers,
        selected_per_bin=case["counts"][0],
        sampled_fraction=case["counts"][0] / case["eligible"],
        moment_cv_per_bin=cv,
        moment_cv_fit_median=np.median(cv[..., fit], axis=-1),
        moment_cv_fit_max=np.max(cv[..., fit], axis=-1),
        relative_to_max_fit_rms=np.sqrt(np.mean(relative_to_max[..., fit] ** 2, axis=(0, 3))),
        xi_mean=xi.mean(axis=0),
        xi_sd=xi.std(axis=0, ddof=1),
        xi_window_difference_mean=(xi[:, :, 1] - xi[:, :, 0]).mean(axis=0),
        xi_difference_to_max_mean=(xi - xi[:, -1:]).mean(axis=0),
        local_slope_fit_median=np.nanmedian(case["local_slopes"][..., fit], axis=(0, 3)),
    )


def plot_cases(paths, directory):
    import matplotlib.pyplot as plt

    cases = [read_case(path) for path in paths]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    summary = {str(c["metadata"]["run_id"]): summarize(c) for c in cases}
    (directory / "summary.json").write_text(json.dumps(_to_serializable(summary), indent=2) + "\n")
    figures = []
    fig, axes = plt.subplots(1, len(cases), figsize=(6 * len(cases), 4), squeeze=False)
    for ax, c in zip(axes[0], cases):
        xi = c["xi_n"]
        for n, color in zip([4, 5, 6], ["#0072B2", "#D55E00", "#009E73"]):
            for w, style in [(0, "-"), (1, "--")]:
                ax.errorbar(
                    c["budgets"],
                    xi[:, :, w, n - 1].mean(axis=0),
                    yerr=xi[:, :, w, n - 1].std(axis=0, ddof=1),
                    color=color,
                    linestyle=style,
                    marker="o",
                    capsize=3,
                    label=f"n={n}, {'150–1200' if w == 0 else '200–1000'} km",
                )
        ax.set(
            xscale="log",
            xlabel="Pairs per fitting bin",
            ylabel=r"$\xi_n$",
            title=f"{c['metadata']['run_id']}, p={c['metadata']['p_value']}; mean ± seed SD",
        )
        ax.grid(alpha=0.2)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        fontsize=8,
    )
    figures.append(("xi_convergence", fig))
    fig, axes = plt.subplots(len(cases), 3, figsize=(13, 3.8 * len(cases)), squeeze=False)
    for row, c in zip(axes, cases):
        cv = c["S_n"].std(axis=0, ddof=1) / c["S_n"].mean(axis=0)
        for ax, n in zip(row, [4, 5, 6]):
            for b, budget in enumerate(c["budgets"]):
                ax.plot(c["centers_km"], 100 * cv[b, n - 1], marker="o", ms=3, label=f"{budget:,}")
            ax.axvspan(150, 1200, color="grey", alpha=0.1)
            ax.set(
                xscale="log",
                xlabel=r"$\lambda$ [km]",
                ylabel="Between-seed CV [%]",
                title=f"{c['metadata']['run_id']}, S{n}",
            )
            ax.grid(alpha=0.2)
            ax.legend(title="Pairs / fit bin", fontsize=8)
    figures.append(("high_order_convergence", fig))
    fig, axes = plt.subplots(1, len(cases), figsize=(7 * len(cases), 4.5), squeeze=False)
    for ax, c in zip(axes[0], cases):
        mean = c["S_n"][:, -1].mean(axis=0)
        sd = c["S_n"][:, -1].std(axis=0, ddof=1)
        p = c["metadata"]["p_value"]
        color = P_MODEL_RUN_COLORS[p]
        centers = c["centers_km"]
        fit = (centers >= 150) & (centers <= 1200)
        reference = p_model_exponents(np.array(ORDERS), p)
        for i, marker in enumerate(["o", "s", "^", "D", "v", "P"]):
            plot_structure_function_curve(
                ax, centers, mean[i], color=color, marker=marker, label=f"n={i + 1}"
            )
            ax.fill_between(centers, mean[i] - sd[i], mean[i] + sd[i], color=color, alpha=0.1)
            anchor = np.flatnonzero(fit)[0]
            ax.loglog(
                centers[fit],
                mean[i, anchor] * (centers[fit] / centers[anchor]) ** reference[i],
                color="black",
                linestyle="--",
                linewidth=0.8,
                label="p-model, anchored shape" if i == 0 else None,
            )
        ax.axvspan(150, 1200, color="grey", alpha=0.12, label="150–1200 km fit")
        ax.axvline(200, color="grey", linestyle=":")
        ax.axvline(1000, color="grey", linestyle=":", label="200–1000 sensitivity")
        ax.set(
            xscale="log",
            yscale="log",
            xlabel=r"$\lambda$ [km]",
            ylabel=r"$S_n(\lambda)$ [$(km/s)^n$]",
            title=f"{c['metadata']['run_id']}, p={p}; sampled truth only",
        )
        ax.grid(alpha=0.2)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        fontsize=8,
    )
    figures.append(("truth_structure_functions", fig))
    for name, fig in figures:
        fig.tight_layout()
        for extension in ["png", "pdf"]:
            fig.savefig(directory / f"{name}.{extension}", dpi=200, bbox_inches="tight")
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--legacy-candidates", type=int, default=0)
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="fresh-process, single-seed maximum-budget benchmark",
    )
    parser.add_argument(
        "--benchmark-budget",
        type=int,
        help="override the fitting-bin budget for a single-seed benchmark",
    )
    parser.add_argument("--plot", nargs="+")
    parser.add_argument("--figure-directory", default="outputs/structure_function_stage2/figures")
    args = parser.parse_args(argv)
    if args.legacy_candidates < 0 or args.legacy_candidates > 2000000:
        parser.error("legacy comparison is capped at two million candidate draws")
    if args.benchmark_budget is not None and (
        not args.benchmark_only or args.benchmark_budget <= 0
    ):
        parser.error("a positive --benchmark-budget requires --benchmark-only")
    if args.config:
        run_study(args.config, args.legacy_candidates, args.benchmark_only, args.benchmark_budget)
    if args.plot:
        plot_cases(args.plot, args.figure_directory)
    if not args.config and not args.plot:
        parser.error("provide --config or --plot")


if __name__ == "__main__":
    main()
