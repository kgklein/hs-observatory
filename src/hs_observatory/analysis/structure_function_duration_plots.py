"""Saved-moment duration-study figures and report; no observation-file access."""

import csv
from pathlib import Path

import numpy as np

from hs_observatory.analysis.structure_function import (
    P_MODEL_RUN_COLORS,
    plot_structure_function_curve,
)
from hs_observatory.analysis.structure_function_convergence import ORDERS


def plot_duration(data, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    groups = [g for g in data["groups"] if g["budget"] == "baseline"]
    if not groups:
        raise ValueError("no baseline results to plot")
    groups.sort(key=lambda g: g["duration_s"])
    duration = np.array([g["duration_s"] for g in groups])
    colors = [
        P_MODEL_RUN_COLORS[0.5],
        P_MODEL_RUN_COLORS[0.7],
        "#CC79A7",
        "#E69F00",
        P_MODEL_RUN_COLORS[0.9],
    ]
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    primary = list(data["windows"]).index("primary")
    first = groups[0]
    theory = np.asarray(first["theory"])
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.8), constrained_layout=True)
    for n, ax in enumerate(axes.flat):
        for w, (name, limits) in enumerate(data["windows"].items()):
            mean = [g["fields"]["z_plus"]["xi_summary"]["mean"][w][n] for g in groups]
            sd = [g["fields"]["z_plus"]["xi_summary"]["std"][w][n] for g in groups]
            ax.errorbar(
                duration,
                mean,
                yerr=sd,
                color=colors[w % len(colors)],
                marker="o" if name == "primary" else "s",
                linestyle="-" if name == "primary" else "--",
                capsize=3,
                label=f"{name}: {limits[0]:g}–{limits[1]:g} km",
            )
        ax.axhline(theory[n], color=".35", ls=":", lw=1, label="p-model reference")
        ax.set(
            title=f"n={n + 1}" + (" (focus)" if n + 1 in (3, 4, 6) else ""),
            xlabel="Observation duration (s)",
            ylabel=r"$\xi_n$",
        )
        ax.grid(alpha=0.15)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(
        f"p={first['p']:g}; scaling exponents; bars = between-pair-seed sample SD", fontsize=10
    )
    _save(fig, output / "exponents_vs_duration", plt)

    for kind in ("local_slopes", "structure_functions"):
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), constrained_layout=True)
        orders = (3, 4, 6) if kind == "local_slopes" else (2, 4, 6)
        for n, ax in zip(orders, axes):
            for index, g in enumerate(groups):
                color = colors[index % len(colors)]
                x = np.asarray(g["centers_km"])
                field = g["fields"]["z_plus"]
                statistic = field["slope_summary" if kind == "local_slopes" else "S_n_summary"]
                mean = np.asarray(statistic["mean"])[n - 1]
                sd = np.asarray(statistic["std"])[n - 1]
                if kind == "structure_functions":
                    plot_structure_function_curve(
                        ax, x, mean, color=color, marker="o", label=f"{g['duration_s']:g} s"
                    )
                else:
                    ax.plot(x, mean, color=color, lw=1.5, label=f"{g['duration_s']:g} s")
                lower = mean - sd
                if kind == "structure_functions":
                    lower = np.where(lower > 0, lower, np.nan)
                ax.fill_between(x, lower, mean + sd, color=color, alpha=0.10)
            ax.set_xscale("log")
            ax.set(title=f"n={n}", xlabel="Separation (km)")
            ax.grid(alpha=0.15)
            if kind == "local_slopes":
                ax.set(xlim=(300, 2000), ylabel=r"Local $\xi_n$")
                ax.axhline(theory[n - 1], color=".35", ls=":", lw=1)
                # Ignore off-panel slope excursions when selecting y limits.
                visible = []
                for line in ax.lines[:-1]:
                    x, y = np.array(line.get_xdata()), np.array(line.get_ydata())
                    visible.extend(y[(x >= 300) & (x <= 2000) & np.isfinite(y)])
                if visible:
                    lo, hi = min(*visible, theory[n - 1]), max(*visible, theory[n - 1])
                    pad = max(0.1, 0.12 * (hi - lo))
                    ax.set_ylim(lo - pad, hi + pad)
            else:
                ax.set(yscale="log", xlim=(70, 20000), ylabel=r"$S_n^+$")
                ax.axvspan(150, 1200, color=".5", alpha=0.10)
            ax.axvline(1000, color=".6", ls="--", lw=0.7)
        axes[0].legend(fontsize=7)
        reference_label = "; dotted: p-model" if kind == "local_slopes" else ""
        fig.suptitle(
            f"p={first['p']:g}; means across pair seeds; shading = ±1 sample SD" + reference_label,
            fontsize=10,
        )
        _save(fig, output / kind, plt)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    for n, color in zip((3, 4, 6), colors):
        delta = [g["fields"]["z_plus"]["delta_longest"]["mean"][primary][n - 1] for g in groups]
        sd = [g["fields"]["z_plus"]["delta_longest"]["std"][primary][n - 1] for g in groups]
        axes[0].errorbar(
            duration, delta, yerr=sd, marker="o", color=color, capsize=3, label=f"n={n}; Δ seed SD"
        )
        sampling_sd = np.array(
            [g["fields"]["z_plus"]["xi_summary"]["std"][primary][n - 1] for g in groups]
        )
        axes[0].fill_between(duration, -sampling_sd, sampling_sd, color=color, alpha=0.07)
    axes[0].axhline(0, color=".5", lw=0.7)
    axes[0].set(
        xlabel="Observation duration (s)",
        ylabel=r"$\xi_n(T)-\xi_n(T_{\mathrm{longest}})$",
        title=f"Primary {data['windows']['primary']} km",
    )
    axes[0].legend(fontsize=7)
    for axis, color in enumerate(colors[:3]):
        fraction = [
            np.mean(np.asarray(g["coverage"]["coordinate_domain_fractions_per_sc"])[:, axis])
            for g in groups
        ]
        axes[1].plot(
            duration, fraction, marker="o", color=color, label=f"{'xyz'[axis]} traversal/L"
        )
    axes[1].set(
        xlabel="Observation duration (s)",
        ylabel="Nominal domain lengths traversed",
        title="Unwrapped displacement; not independent coverage",
    )
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.15)
    fig.suptitle(
        f"p={first['p']:g}; longest selected duration is a reference, not ground truth; "
        "bands = individual xi seed SD",
        fontsize=9,
    )
    _save(fig, output / "duration_summary", plt)


def _save(fig, prefix, plt):
    for extension in ("png", "pdf"):
        fig.savefig(f"{prefix}.{extension}", dpi=200)
    plt.close(fig)


def write_report(data, path):
    """Write complete fitted-exponent/peak/coverage tables plus a compact CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    groups = data["groups"]
    validation = data["validation"]
    lines = [
        "# Observation-duration structure-function convergence",
        "",
        "**Generated from saved truth moments; no measurement-error MC statistics.**",
        "",
        f"Nested input sequence within tolerance: **{validation['valid_nested_sequence']}**.",
        "Inputs, full generation YAML, hashes, exact-prefix comparisons and tolerances are",
        "in validation.json and manifest.json. Sampling settings and exact bin edges are",
        "stored with each seed. Geometric centers define the main fits; population",
        "mean-log-center fits are separate diagnostics.",
        "",
        "All uncertainties below are sample SD across deterministic pair seeds, conditional",
        "on one field and duration. Seed ranges and individual values remain in derived.json.",
        "The longest selected duration is a reference, not exact truth. Fewer than five",
        "seeds or a single selected duration is a pilot, not a completed convergence study.",
        "",
        "## Input checks",
        "",
        "| Duration s | p | Field seed / realization | Levels (inclusive) | Exact prefix | Max truth difference |",
        "|---:|---:|---|---|---|---:|",
    ]
    for _, record in sorted(validation["inputs"].items(), key=lambda v: v[1]["duration_s"]):
        differences = [
            c["max_abs_difference"]
            for k, c in record["prefix_comparisons"].items()
            if k.endswith("_truth") and c["max_abs_difference"] is not None
        ]
        lines.append(
            f"| {record['duration_s']:g} | {record['p']} | {record['base_seed']} / "
            f"{record['realization']} | {record['injection_level']}–"
            f"{record['maximum_level_inclusive']} | {record['exact_prefix']} | "
            f"{max(differences, default=float('nan')):.3g} |"
        )
    if all(g["p"] == 0.5 for g in groups):
        lines += [
            "",
            "**This sequence has p=0.5. It cannot directly settle the stronger",
            "p=0.9 anomaly; a matched intermittent duration sequence would be needed.**",
        ]
    lines += [
        "",
        "## Fitted exponents",
        "",
        "| Budget | Duration s | Window km | Seeds | "
        + " | ".join(f"xi{n} mean ± SD" for n in ORDERS)
        + " |",
        "|---|---:|---|---:|" + "---:|" * 6,
    ]
    csv_path = path.with_suffix(".csv")
    with csv_path.open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "budget",
                "duration_s",
                "window",
                "order",
                "mean",
                "std",
                "min",
                "max",
                "minus_theory",
                "delta_previous",
                "delta_longest",
                "delta_longest_std",
            ]
        )
        for g in groups:
            f = g["fields"]["z_plus"]
            for w, (name, limits) in enumerate(data["windows"].items()):
                summary = f["xi_summary"]
                values = [
                    f"{summary['mean'][w][n]:.4f} ± {summary['std'][w][n]:.4f}" for n in range(6)
                ]
                lines.append(
                    f"| {g['budget']} | {g['duration_s']:g} | {limits} | "
                    f"{len(g['seeds'])} | " + " | ".join(values) + " |"
                )
                for n in range(6):
                    previous = f["delta_previous"]
                    writer.writerow(
                        [
                            g["budget"],
                            g["duration_s"],
                            name,
                            n + 1,
                            *[summary[key][w][n] for key in ("mean", "std", "min", "max")],
                            summary["mean"][w][n] - g["theory"][n],
                            previous["mean"][w][n] if previous else np.nan,
                            f["delta_longest"]["mean"][w][n],
                            f["delta_longest"]["std"][w][n],
                        ]
                    )
    lines += [
        "",
        "## Duration changes (primary fits)",
        "",
        "| Budget | Duration s | Order | Δ previous | Δ longest ± paired-seed SD |",
        "|---|---:|---:|---:|---:|",
    ]
    primary = list(data["windows"]).index("primary")
    for g in groups:
        f = g["fields"]["z_plus"]
        for n in (3, 4, 6):
            previous = f["delta_previous"]
            delta = previous["mean"][primary][n - 1] if previous else np.nan
            longest = f["delta_longest"]
            lines.append(
                f"| {g['budget']} | {g['duration_s']:g} | {n} | {delta:.4f} | "
                f"{longest['mean'][primary][n - 1]:.4f} ± "
                f"{longest['std'][primary][n - 1]:.4f} |"
            )
    if any("delta_baseline_budget" in g["fields"]["z_plus"] for g in groups):
        lines += [
            "",
            "## Higher-budget verification (primary fits)",
            "",
            "| Duration s | Order | Higher minus baseline xi ± paired-seed SD |",
            "|---:|---:|---:|",
        ]
        for g in groups:
            delta = g["fields"]["z_plus"].get("delta_baseline_budget")
            if delta is not None:
                for n in (3, 4, 6):
                    lines.append(
                        f"| {g['duration_s']:g} | {n} | "
                        f"{delta['mean'][primary][n - 1]:.4f} ± "
                        f"{delta['std'][primary][n - 1]:.4f} |"
                    )
    lines += [
        "",
        "## Local-slope peaks",
        "",
        f"Maximum over geometric centers in {data['peak_interval_km']} km, separately per seed.",
        "Reported location ranges are bin centers; a maximum need not be an interior peak.",
        "",
        "| Budget | Duration s | Order | Amplitude mean ± SD | Location range km |",
        "|---|---:|---:|---:|---:|",
    ]
    for g in groups:
        f = g["fields"]["z_plus"]
        for n in (3, 4, 6):
            locations = np.asarray(f["peaks"]["separation_km"])[:, n - 1]
            finite = locations[np.isfinite(locations)]
            lower, upper = (min(finite), max(finite)) if len(finite) else (np.nan, np.nan)
            lines.append(
                f"| {g['budget']} | {g['duration_s']:g} | {n} | "
                f"{f['peak_summary']['mean'][n - 1]:.4f} ± "
                f"{f['peak_summary']['std'][n - 1]:.4f} | {lower:.1f}–{upper:.1f} |"
            )
    lines += [
        "",
        "## Spatial coverage and population",
        "",
        "| Budget | Duration s | Mean path km | Axis traversals x/y/z | "
        "Decimated coarse revisits | Eligible / selected |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for g in groups:
        c = g["coverage"]
        traversal = np.mean(c["coordinate_domain_fractions_per_sc"], axis=0)
        lines.append(
            f"| {g['budget']} | {g['duration_s']:g} | "
            f"{np.mean(c['distance_traveled_km_per_sc']):.1f} | "
            + "/".join(f"{x:.3f}" for x in traversal)
            + f" | {c['decimated_samples']['repeat_fraction']:.3f} | "
            f"{sum(g['geometry']['eligible_per_bin']):,} / "
            f"{sum(g['counts'][0]):,} |"
        )
    lines += [
        "",
        "Coverage uses unwrapped path length and sparse periodic voxels (64 cells/axis",
        "by default). Decimation is approximately one voxel length of travel. Repeated",
        "voxels include formation overlap and correlated samples; this is not a count",
        "of independent eddies. Eligible spacecraft-pair/signed-lag counts, population",
        "direction histograms (absolute cosine to flow) and all-bin counts are saved.",
        "",
        "## Interpretation",
        "",
        "Assess changes against both seed variability and the optional higher-budget",
        "results. Agreement with p-model theory is a separate comparison, not a",
        "convergence criterion. Persistent peaks do not alone identify physical scales.",
        "Global density normalization is evaluated per duration using the unchanged",
        "Elsasser machinery; this can matter if mean density differs between records.",
        "No automatic physical-convergence claim is made. Independent fields, other p",
        "values and longer independent spatial coverage remain separate tests.",
    ]
    lines += [
        "",
        "## Reproduction and saved results",
        "",
        "```bash",
        "# Validate only (no pair sampling):",
        "bash scripts/structure_function_duration.sh --dry-run",
        "# Complete baseline plus optional endpoint higher-budget verification:",
        "bash scripts/structure_function_duration.sh --verify-budget",
        "# One-duration, one-seed bounded smoke test, separate output directory:",
        "bash scripts/structure_function_duration.sh --durations 300 --seed-count 1 "
        "--fit-pairs 1000 --outside-pairs 1000 "
        "--output-dir outputs/structure_function_duration_smoke",
        "# Refit and plot saved moments; no HDF5 or increment evaluation:",
        "bash scripts/structure_function_duration.sh --plot-only",
        "```",
        "",
        "Defaults above use the supplied p=0.5 recipe. For another p sequence, set",
        "`--generation-glob` to its generation YAML pattern and `--output-dir` to a",
        "distinct directory; use that same output directory for cached commands.",
        "The selected run configuration is embedded in manifest.json.",
        "",
        "The dedicated output directory holds validation.json, manifest.json, atomic",
        "per-duration/budget/seed moments/*.json files, derived.json and four PNG/PDF",
        "figure pairs. The report CSV contains all xi means/SD/ranges and changes.",
        "Each seed product saves both Elsasser variables, exact edges, counts,",
        "sampler diagnostics, empirical population centers, spatial coverage, timing",
        "and peak RSS. Input and statistical-code fingerprints control cache reuse.",
        "Fit windows are excluded from moment-cache keys. Figures and report are",
        "rebuilt from saved moments; the manifest describes the latest selected run.",
        "Incomplete runs can be resumed with the same command. No automatic full",
        "census is requested, although small bins are capped at their population.",
        "",
        "Use `--mode sample`, `--mode derive`, and `--mode plot` for separate stages.",
        "`--mode plot` reads derived.json; `--plot-only` first refits saved moments.",
        "Selected seeds/durations apply to cached refitting when explicitly provided.",
        "Error bars across duration changes use common seed labels, with differences",
        "computed per seed; identities are not forced equal across populations.",
        "Periodic wrapping is used only for coverage, never to replace the existing",
        "unwrapped-separation structure-function estimator.",
    ]
    path.write_text("\n".join(lines) + "\n")
