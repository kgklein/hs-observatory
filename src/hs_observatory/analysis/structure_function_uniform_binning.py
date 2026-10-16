"""Truth-only bin-grid comparison; saved results can be refitted without sampling."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import yaml

from hs_observatory.analysis.derived.geometry_pairs import (
    ALGORITHM_VERSION,
    GeometryPlanCache,
    geometry_pair_chunks,
)
from hs_observatory.analysis.elsasser import compute_elsasser_analysis
from hs_observatory.analysis.structure_function import (
    P_MODEL_RUN_COLORS,
    _bin_indices,
    _fit_scaling_exponents,
    _positions_for_group_member,
    _settings,
    _to_serializable,
    compute_all_spacetime_structure_functions,
    plot_structure_function_curve,
)
from hs_observatory.analysis.structure_function_convergence import FIT_WINDOWS, ORDERS, stage2_edges
from hs_observatory.analysis.structure_function_local import local_log_slopes

ROOT = Path("outputs/structure_function_uniform_binning")
SEEDS = (12345, 23456)


def grids():
    """Fixed endpoints; shifted grid has two half-width edge bins (41 total)."""
    step = np.log(20000 / 70) / 40
    return {
        "A": stage2_edges(),
        "B": np.geomspace(70.0, 20000.0, 41),
        "C": np.r_[70.0, 70 * np.exp((np.arange(40) + 0.5) * step), 20000.0],
    }


def targets(edges):
    centers = np.sqrt(edges[:-1] * edges[1:])
    return np.where((centers >= 150) & (centers <= 1200), 100000, 10000)


def separation_statistics(plan, positions, requested, seed):
    """Population log means compressed by affine lag; sampled means from exact coordinates.

    Affine-group distances differ across starts only by coordinate roundoff. Boundary
    split groups use their explicit starts. Sample statistics replay the same stream,
    without any increment-dependent selection or observed-member processing.
    """
    population_logs = []
    for group_bin in plan.bins:
        g = group_bin.groups
        if not len(g):
            population_logs.append(np.nan)
            continue
        starts = g[:, 3].copy()
        for row, explicit in group_bin.explicit_starts.items():
            starts[row] = explicit[0]
        d = np.linalg.norm(
            positions[starts + g[:, 2], g[:, 1]] - positions[starts, g[:, 0]], axis=1
        )
        population_logs.append(np.average(np.log(d), weights=g[:, 4]))
    sums = np.zeros(len(plan.bins))
    counts = np.zeros(len(plan.bins), dtype=np.int64)
    flat = positions.reshape(-1, 3)
    for left, right in geometry_pair_chunks(plan, requested, random_seed=seed, chunk_size=100000):
        d = np.linalg.norm(flat[right] - flat[left], axis=1)
        bins, valid = _bin_indices(d, plan.edges)
        if not np.all(valid):
            raise AssertionError("sample outside physical limits")
        sums += np.bincount(bins, weights=np.log(d), minlength=len(counts))
        counts += np.bincount(bins, minlength=len(counts))
    means = np.divide(sums, counts, out=np.full(len(counts), np.nan), where=counts > 0)
    return np.exp(population_logs), np.exp(means), counts


def diagnostics(product, population_centers, sampled_centers):
    centers = np.asarray(product["lambda_bin_centers_km"])
    counts = np.asarray(product["counts"])
    values = np.asarray(product["z_plus"]["S_n"])
    result = {}
    for name, x in [
        ("geometric", centers),
        ("population", population_centers),
        ("sampled", sampled_centers),
    ]:
        result[name] = {
            "centers": x,
            "fits": _fit_scaling_exponents(values, counts, x, ORDERS, FIT_WINDOWS, 3),
            "local_slopes": local_log_slopes(x, values, counts=counts)[1],
        }
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    ROOT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    protected = [
        p
        for case in (400, 402)
        for p in (
            Path(f"analysis/malara-prod-{case}_sf-stage3-publication.yaml"),
            *Path("outputs/structure_function_stage3").glob(f"malara-prod-{case}_sf_publication.*"),
        )
    ]
    before = {str(p): digest(p) for p in protected}
    output = {
        "edges": grids(),
        "seeds": SEEDS,
        "algorithm": ALGORITHM_VERSION,
        "windows": FIT_WINDOWS,
        "cases": {},
        "protected_hashes": before,
    }
    cache = GeometryPlanCache(max_entries=3)
    for case in (400, 402):
        config = yaml.safe_load(
            Path(f"analysis/malara-prod-{case}_sf-stage3-publication.yaml").read_text()
        )
        settings = _settings(config["structure_function"])
        source = Path(config["input"]["path"])
        original = json.loads(
            Path(
                f"outputs/structure_function_stage3/malara-prod-{case}_sf_publication.json"
            ).read_text()
        )
        expected = original["metadata"]["stage3"]["provenance"]["input_sha256"]
        if digest(source) != expected:
            raise ValueError("input differs from the corrected Stage 3 observation")
        truth = compute_elsasser_analysis(source, dict(settings["elsasser"], data_group="truth"))
        positions, _, _ = _positions_for_group_member(
            source, "truth", 0, truth["time"], len(truth["spacecraft_ids"])
        )
        if len(truth["time"]) != 2401 or not np.isclose(np.ptp(truth["time"]), 300):
            raise ValueError("expected original 300-second observations")
        records = {}
        for name, edges in grids().items():
            requested = targets(edges)
            plan, _ = cache.get(positions, edges, 70.0, 20000.0)
            records[name] = {}
            for seed in SEEDS:
                tick = time.perf_counter()
                if name == "A" and seed == 12345:
                    product = original["products"]["truth"]
                    np.testing.assert_array_equal(product["lambda_bin_edges_km"], edges)
                    origin = "saved Stage 3 truth"
                else:
                    product = compute_all_spacetime_structure_functions(
                        positions[None],
                        np.asarray(truth["z_plus"])[None],
                        np.asarray(truth["z_minus"])[None],
                        ORDERS,
                        edges,
                        pair_sampling=dict(
                            settings["pair_sampling"],
                            random_seed=seed,
                            target_pairs_per_bin=requested,
                        ),
                        scale_ranges_km=FIT_WINDOWS,
                        min_pair_lambda_km=70.0,
                        max_pair_lambda_km=20000.0,
                        geometry_plan_cache=cache,
                    )
                    origin = "truth-only existing accumulator"
                pop_x, sample_x, counts = separation_statistics(plan, positions, requested, seed)
                np.testing.assert_array_equal(counts, np.minimum(requested, plan.populations))
                np.testing.assert_array_equal(counts, np.asarray(product["counts"])[0])
                records[name][str(seed)] = {
                    "product": product,
                    "diagnostics": diagnostics(product, pop_x, sample_x),
                    "eligible": plan.populations,
                    "requested": requested,
                    "selected": counts,
                    "groups": [len(b.groups) for b in plan.bins],
                    "shortfall": np.maximum(requested - counts, 0),
                    "origin": origin,
                    "seconds": time.perf_counter() - tick,
                }
                print(
                    f"{case} {name} seed {seed}: {records[name][str(seed)]['seconds']:.2f}s",
                    flush=True,
                )
        output["cases"][str(case)] = {"input_sha256": expected, "grids": records}
    if before != {str(p): digest(p) for p in protected}:
        raise AssertionError("protected Stage 3 artifacts changed")
    output["seconds"] = time.perf_counter() - started
    output["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    (ROOT / "results.json").write_text(json.dumps(_to_serializable(output), indent=2))
    write_tables(output)
    plot(output)


def write_tables(data):
    """Compact counts/fits tables and a standalone exact-edge recipe."""
    (ROOT / "bin_edges.json").write_text(json.dumps(_to_serializable(data["edges"]), indent=2))
    with (ROOT / "bin_counts.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case",
                "grid",
                "seed",
                "lower_km",
                "upper_km",
                "center_km",
                "population_log_center_km",
                "sample_log_center_km",
                "eligible",
                "requested",
                "selected",
                "eligible_groups",
                "shortfall",
            ]
        )
        for case, item in data["cases"].items():
            for grid, records in item["grids"].items():
                edges = data["edges"][grid]
                for seed, record in records.items():
                    d = record["diagnostics"]
                    for q in range(len(edges) - 1):
                        writer.writerow(
                            [
                                case,
                                grid,
                                seed,
                                edges[q],
                                edges[q + 1],
                                d["geometric"]["centers"][q],
                                d["population"]["centers"][q],
                                d["sampled"]["centers"][q],
                                *[
                                    record[key][q]
                                    for key in (
                                        "eligible",
                                        "requested",
                                        "selected",
                                        "groups",
                                        "shortfall",
                                    )
                                ],
                            ]
                        )
    with (ROOT / "fitted_exponents.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case", "grid", "seed", "centers", "window", "order", "xi"])
        for case, item in data["cases"].items():
            for grid, records in item["grids"].items():
                for seed, record in records.items():
                    for centers, d in record["diagnostics"].items():
                        for w, window in enumerate(FIT_WINDOWS):
                            for n in range(6):
                                writer.writerow(
                                    [
                                        case,
                                        grid,
                                        seed,
                                        centers,
                                        window,
                                        n + 1,
                                        d["fits"]["raw"][0][w][n],
                                    ]
                                )


def plot(data):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    colors = {grid: P_MODEL_RUN_COLORS[p] for grid, p in zip("ABC", (0.9, 0.5, 0.7))}
    labels = {"A": "Piecewise", "B": "Uniform", "C": "Shifted uniform"}
    figures = [
        plt.subplots(2, 3, figsize=(10, 5), constrained_layout=True),
        plt.subplots(2, 2, figsize=(8, 5), constrained_layout=True),
        plt.subplots(2, 2, figsize=(8, 5), constrained_layout=True),
    ]
    for row, case in enumerate(("400", "402")):
        p = 0.9 if case == "400" else 0.5
        for name in ("A", "B", "C"):
            for seed in SEEDS:
                r = data["cases"][case]["grids"][name][str(seed)]
                d = r["diagnostics"]["geometric"]
                x = np.asarray(d["centers"])
                style = dict(
                    color=colors[name],
                    alpha=1 if seed == 12345 else 0.3,
                    linewidth=1.3,
                    linestyle="-" if seed == 12345 else ":",
                )
                label = labels[name] if seed == 12345 else None
                for col, order in enumerate((3, 4, 6)):
                    ax = figures[0][1][row, col]
                    ax.plot(
                        x,
                        np.asarray(d["local_slopes"])[0, order - 1],
                        label=label,
                        marker="o" if seed == 12345 else None,
                        markersize=3,
                        **style,
                    )
                    ax.set(
                        xscale="log",
                        xlim=(500, 2000),
                        title=f"p={p}, n={order}",
                        xlabel="Separation (km)",
                        ylabel=r"Local $\xi_n$",
                    )
                for col, order in enumerate((4, 6)):
                    ax = figures[1][1][row, col]
                    values = np.asarray(r["product"]["z_plus"]["S_n"])[0, order - 1]
                    if seed == 12345:
                        plot_structure_function_curve(
                            ax, x, values, color=colors[name], marker="o", label=label
                        )
                    else:
                        ax.plot(x, values, **style)
                    ax.set(
                        xscale="log",
                        yscale="log",
                        xlim=(500, 2000),
                        title=f"p={p}, n={order}",
                        xlabel="Separation (km)",
                        ylabel=r"$S_n^+$",
                    )
                for col, window in enumerate(FIT_WINDOWS):
                    ax = figures[2][1][row, col]
                    ax.plot(
                        ORDERS,
                        np.asarray(d["fits"]["raw"])[0, col],
                        label=label,
                        marker="o" if seed == 12345 else None,
                        markersize=3,
                        **style,
                    )
                    ax.set(
                        title=f"p={p}, {FIT_WINDOWS[window]} km",
                        xlabel="Order n",
                        ylabel=r"Fitted $\xi_n$",
                        xticks=ORDERS,
                    )
    for index, (fig, axes) in enumerate(figures):
        fig.suptitle("Solid: seed 12345; faint dotted: seed 23456", fontsize=9)
        for ax in axes.flat:
            if index < 2:
                visible = []
                for line in ax.lines:
                    x, y = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
                    visible.extend(y[(x >= 500) & (x <= 2000) & np.isfinite(y)])
                lo, hi = min(visible), max(visible)
                if index == 1:
                    ax.set_ylim(lo / 1.25, hi * 1.25)
                else:
                    padding = 0.12 * max(hi - lo, 0.1)
                    ax.set_ylim(lo - padding, hi + padding)
                ax.axvline(1200, color=".5", linestyle="--", linewidth=0.8)
            ax.grid(alpha=0.15)
        axes[0, 0].legend(fontsize=8)
        name = ("local_slopes", "structure_functions", "fitted_exponents")[index]
        for suffix in ("png", "pdf"):
            fig.savefig(ROOT / f"{name}.{suffix}", dpi=200)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        data = json.loads((ROOT / "results.json").read_text())
        write_tables(data)
        plot(data)
    else:
        run()


if __name__ == "__main__":
    main()
