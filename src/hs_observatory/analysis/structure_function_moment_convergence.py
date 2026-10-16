"""Truth-only rare-increment diagnostics using the geometry_stratified sampler.

N_conc measures weight concentration, NOT a correlation-corrected effective
sample size. Shared observations, pair groups and temporal blocks are dependent.
No fields or measurement-error realizations are generated.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re

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
    P_MODEL_RUN_COLORS,
    _positions_for_group_member,
    _settings,
    _to_serializable,
)

VERSION = "moment_convergence_v1"
EDGES = np.geomspace(70.0, 20000.0, 41)
ORDERS = np.array([2, 4, 6])
TAILS = np.array([0.001, 0.01, 0.05])
RECORD_DTYPE = np.dtype(
    [
        (name, dtype)
        for name, dtype in (
            ("spacecraft_left", "i4"),
            ("spacecraft_right", "i4"),
            ("time_left_s", "f8"),
            ("time_right_s", "f8"),
            ("midpoint_s", "f8"),
            ("lag_samples", "i8"),
            ("separation_km", "f8"),
            ("amplitude_km_s", "f8"),
        )
    ]
)
CAUTION = (
    "N_conc is a concentration count, not a correlation-corrected effective sample size. "
    "Pairs, lag groups and midpoint-attributed blocks can share observations/structures; "
    "they are not independent events. Seed variability is pair sampling, not error MC."
)


def fingerprint(value):
    return hashlib.sha256(json.dumps(_to_serializable(value), sort_keys=True).encode()).hexdigest()


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_bins(separations=None, bins=None):
    """Nearest geometric centers in log distance; explicit bins are zero based."""
    centers = np.sqrt(EDGES[:-1] * EDGES[1:])
    separations = [300, 500, 700, 1000, 1500] if separations is None else separations
    if any(not np.isfinite(s) or s < EDGES[0] or s > EDGES[-1] for s in separations):
        raise ValueError("separations must be finite and within 70–20000 km")
    bins = [] if bins is None else list(bins)
    if any(b < 0 or b >= 40 for b in bins):
        raise ValueError("bin indices must be in 0..39")
    return sorted(set(bins + [int(np.argmin(abs(np.log(centers / s)))) for s in separations]))


def sample_records(plan, positions, z_plus, times, bin_index, budget, seed, chunk_size):
    """Retain only selected identities, bounded by budget; no population enumeration.

    Each budget uses the sampler directly: partial draws share reproducible random
    prefixes, but census uses its own ordered stream, never a biased census prefix.
    """
    count = min(int(plan.populations[bin_index]), budget)
    records = np.empty(count, dtype=RECORD_DTYPE)
    targets = np.zeros(len(plan.bins), dtype=np.int64)
    targets[bin_index] = budget
    flat_r, flat_z = positions.reshape(-1, 3), z_plus.reshape(-1, 3)
    offset = 0
    for left, right in geometry_pair_chunks(plan, targets, random_seed=seed, chunk_size=chunk_size):
        a, b = left // plan.n_spacecraft, right // plan.n_spacecraft
        row = records[offset : offset + len(left)]
        row["spacecraft_left"] = left % plan.n_spacecraft
        row["spacecraft_right"] = right % plan.n_spacecraft
        row["time_left_s"], row["time_right_s"] = times[a], times[b]
        row["midpoint_s"] = 0.5 * (times[a] + times[b])
        row["lag_samples"] = b - a
        row["separation_km"] = np.linalg.norm(flat_r[right] - flat_r[left], axis=1)
        row["amplitude_km_s"] = np.linalg.norm(flat_z[right] - flat_z[left], axis=1)
        offset += len(left)
    if offset != count:
        raise RuntimeError("sampler count differs from exact eligible population cap")
    return records


def moment_diagnostics(records, time_bounds, block_width=30.0, top_count=100):
    """Exact empirical tails use ceil(f*N) ranks (ties broken by stable input order).

    All powers use amplitudes divided by their maximum. Log moments remain usable
    even when the dimensional moment exceeds float64. Zero-moment fractions and
    empty-bin moments are undefined (NaN); zero data have N_conc=0.
    """
    if not np.isfinite(block_width) or block_width <= 0 or top_count < 0:
        raise ValueError("positive finite block width and nonnegative top count required")
    amplitude = records["amplitude_km_s"]
    if np.any(~np.isfinite(amplitude)) or np.any(amplitude < 0):
        raise ValueError("increment amplitudes must be finite and nonnegative")
    start, stop = time_bounds
    n_blocks = max(1, int(np.ceil((stop - start) / block_width)))
    block_edges = start + np.arange(n_blocks + 1) * block_width
    block_edges[-1] = max(stop, start + block_width) if n_blocks == 1 else stop
    block = np.clip(((records["midpoint_s"] - start) / block_width).astype(int), 0, n_blocks - 1)
    ranking = np.argsort(amplitude, kind="stable")
    sorted_x = amplitude[ranking]
    count = len(amplitude)
    scale = float(sorted_x[-1]) if count else 0.0
    scaled = amplitude / scale if scale else np.zeros(count)
    # Canonical spacecraft order with a signed lag in that orientation.
    left, right = records["spacecraft_left"], records["spacecraft_right"]
    reverse = left > right
    keys = np.column_stack(
        (
            np.minimum(left, right),
            np.maximum(left, right),
            np.where(reverse, -records["lag_samples"], records["lag_samples"]),
        )
    )
    groups, inverse = np.unique(keys, axis=0, return_inverse=True)
    tail_counts = np.minimum(count, np.ceil(TAILS * count).astype(int))
    tail_masks = np.zeros((len(TAILS), count), dtype=bool)
    for mask, size in zip(tail_masks, tail_counts):
        if size:
            mask[ranking[-size:]] = True
    result = dict(
        count=count,
        orders=ORDERS,
        tail_fractions=TAILS,
        tail_counts=tail_counts,
        sorted_amplitude_km_s=sorted_x,
        amplitude_scale_km_s=scale,
        block_edges_s=block_edges,
        block_counts=np.bincount(block, minlength=n_blocks),
        group_keys=groups,
        group_counts=np.bincount(inverse, minlength=len(groups)),
        top_records=records[ranking[-min(count, top_count) :][::-1]] if top_count else records[:0],
    )
    for name in ("block", "group"):
        labels, size = (block, n_blocks) if name == "block" else (inverse, len(groups))
        result[f"{name}_extreme_counts"] = np.array(
            [np.bincount(labels[mask], minlength=size) for mask in tail_masks]
        )
    moments, logs, concentration, tails, blocks, group_weights, top_blocks = ([] for _ in range(7))
    for order in ORDERS:
        weights = scaled**order
        total = weights.sum()
        fractions = weights / total if total else np.full(count, np.nan)
        log_moment = (
            order * np.log(scale) + np.log(total / count)
            if total
            else (-np.inf if count else np.nan)
        )
        with np.errstate(over="ignore", under="ignore"):
            moments.append(np.exp(log_moment))
        logs.append(log_moment)
        concentration.append(total * total / np.dot(weights, weights) if total else 0.0)
        tails.append([fractions[mask].sum() if total else np.nan for mask in tail_masks])
        block_fraction = (
            np.bincount(block, weights=weights, minlength=n_blocks) / total
            if total
            else np.full(n_blocks, np.nan)
        )
        blocks.append(block_fraction)
        group_weights.append(
            np.bincount(inverse, weights=weights, minlength=len(groups)) / total
            if total
            else np.full(len(groups), np.nan)
        )
        top_blocks.append([np.sort(block_fraction)[-k:].sum() for k in (1, 3, 5)])
    result.update(
        S_n=np.array(moments),
        log_S_n=np.array(logs),
        N_conc=np.array(concentration),
        tail_contribution=np.array(tails),
        block_contribution=np.array(blocks),
        group_contribution=np.array(group_weights),
        top_block_counts=np.array([1, 3, 5]),
        top_block_contribution=np.array(top_blocks),
    )
    # Log-spaced positive-amplitude PDF includes the entire observed tail; zeros separate.
    positive = sorted_x[sorted_x > 0]
    if len(positive):
        lo, hi = positive[0], positive[-1]
        hist_edges = np.geomspace(lo, hi, 129) if hi > lo else np.array([lo * 0.5, hi * 1.5])
    else:
        hist_edges = np.array([0.0, 1.0])
    hist, _ = np.histogram(positive, hist_edges)
    result.update(
        hist_edges_km_s=hist_edges,
        hist_counts=hist,
        hist_pdf=hist / (max(count, 1) * np.diff(hist_edges)),
        zero_count=int(np.sum(amplitude == 0)),
    )
    return result


def cumulative_distribution(amplitudes):
    """Regenerate exact C_n(x)=sum_{a: amplitude<=x} amplitude**n / total from cache."""
    x = np.asarray(amplitudes)
    if not len(x) or not x[-1]:
        return x, np.full((3, len(x)), np.nan)
    scaled = x / x[-1]
    normalized = scaled / np.sqrt(np.mean(scaled**2))
    curves = []
    for n in ORDERS:
        weights = scaled**n
        cumulative = np.cumsum(weights) / weights.sum()
        # At tied amplitudes C includes all increments with that amplitude.
        ends = np.searchsorted(x, x, side="right") - 1
        curves.append(cumulative[ends])
    return normalized, np.array(curves)


def save_product(path, metadata, diagnostics):
    """Atomic completion: interrupted products cannot be mistaken for cache hits."""
    path = Path(path)
    temporary = path.with_suffix(".h5.tmp")
    with h5py.File(temporary, "w") as h:
        h.attrs["metadata_json"] = json.dumps(_to_serializable(metadata), sort_keys=True)
        for key, value in diagnostics.items():
            array = np.asarray(value)
            h.create_dataset(
                key, data=array, compression="gzip" if array.size and array.ndim else None
            )
        h.attrs["complete"] = True
    temporary.replace(path)


def read_product(path, expected=None, distribution=True):
    with h5py.File(path) as h:
        metadata = json.loads(h.attrs["metadata_json"])
        if not h.attrs.get("complete") or (expected and metadata["fingerprint"] != expected):
            raise ValueError(f"incomplete or incompatible cache: {path}")
        return metadata, {
            key: h[key][()] for key in h if distribution or key != "sorted_amplitude_km_s"
        }


def prepare_case(root, case, duration):
    """Resolve output.path relative to project root, as generation commands do."""
    case = re.sub(r"-t\d+(?:\.\d+)?$", "", case) + f"-t{duration:g}"
    config_path = root / "generation" / f"{case}.yaml"
    config = yaml.safe_load(config_path.read_text())
    source = Path(config["output"]["path"])
    source = source if source.is_absolute() else root / source
    if not source.is_file():
        raise FileNotFoundError(f"saved observations required (no generation fallback): {source}")
    with h5py.File(source) as h:
        saved = yaml.safe_load(h["metadata"].attrs["config_yaml"])
        if "truth" not in h:
            raise ValueError(f"missing truth datasets: {source}")
        for cfg in (config, saved):
            if int(cfg["errors"]["n_mc"]) != 0:
                raise ValueError("Study A requires truth-only n_mc: 0 generation")
        if saved != config:
            raise ValueError("saved generation YAML differs from the requested generation YAML")
    settings = dict(_settings({})["elsasser"], data_group="truth")
    elsasser = compute_elsasser_analysis(source, settings)
    times = np.asarray(elsasser["time"])
    if len(times) < 2 or not np.isclose(times[-1] - times[0], duration):
        raise ValueError("aligned input duration differs from requested duration")
    positions, position_source, notes = _positions_for_group_member(
        source, "truth", 0, times, len(elsasser["spacecraft_ids"])
    )
    z_plus = np.asarray(elsasser["z_plus"])
    if np.any(~np.isfinite(z_plus)):
        raise ValueError("nonfinite truth Elsasser fields")
    plan = build_geometry_pair_plan(
        positions, EDGES, min_pair_lambda_km=70.0, max_pair_lambda_km=20000.0
    )
    metadata = dict(
        case=case,
        source=str(source.resolve()),
        source_sha256=digest_file(source),
        generation_sha256=digest_file(config_path),
        p_value=config["field"]["turbulence"]["intermittency_p"],
        spacecraft_ids=elsasser["spacecraft_ids"],
        elsasser_settings=settings,
        elsasser_metadata={k: v for k, v in elsasser["metadata"].items() if k != "created_utc"},
        position_source=position_source,
        notes=notes,
        duration_s=duration,
        version=VERSION,
        sampler_version=ALGORITHM_VERSION,
        numpy_version=np.__version__,
        implementation_sha256=digest_file(__file__),
        dependency_sha256={
            name: digest_file(Path(__file__).parent / name)
            for name in (
                "elsasser.py",
                "derived/elsasser.py",
                "derived/geometry_pairs.py",
                "structure_function.py",
            )
        },
        caution=CAUTION,
    )
    return metadata, plan, positions, z_plus, times


def summarize_products(products, output):
    """Save actual counts and differences to largest budget / seed mean in log space."""
    rows = []
    for metadata, data in products:
        peers = [
            (m, d)
            for m, d in products
            if m["case"] == metadata["case"] and m["bin"] == metadata["bin"]
        ]
        same_seed = [(m, d) for m, d in peers if m["seed"] == metadata["seed"]]
        largest = max(same_seed, key=lambda pair: pair[0]["budget"])[1]
        seed_data = [d for m, d in peers if m["budget"] == metadata["budget"]]
        seed_logs = np.array([d["log_S_n"] for d in seed_data])
        with np.errstate(invalid="ignore", over="ignore"):
            delta = np.expm1(data["log_S_n"] - largest["log_S_n"])
            seed_delta = data["log_S_n"] - seed_logs.mean(axis=0)
        rows.append(
            dict(
                case=metadata["case"],
                bin=metadata["bin"],
                seed=metadata["seed"],
                budget=metadata["budget"],
                count=data["count"],
                eligible=metadata["eligible"],
                S_n=data["S_n"],
                log_S_n=data["log_S_n"],
                relative_to_largest_budget=delta,
                log_difference_to_seed_mean=seed_delta,
                between_seed_log_sd=seed_logs.std(axis=0),
                tail_difference_to_largest_budget=(
                    data["tail_contribution"] - largest["tail_contribution"]
                ),
                between_seed_tail_sd=np.array([d["tail_contribution"] for d in seed_data]).std(
                    axis=0
                ),
                tail_contribution=data["tail_contribution"],
                N_conc=data["N_conc"],
            )
        )
    Path(output).write_text(
        json.dumps(_to_serializable(dict(caution=CAUTION, rows=rows)), indent=2) + "\n"
    )


def plot_saved(paths, output):
    """Only cached diagnostic files are read; source HDF5/YAML are unnecessary."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    products = [read_product(path, distribution=False) for path in paths]
    locations = {m["fingerprint"]: path for path, (m, _) in zip(paths, products)}
    if not products:
        raise ValueError("no saved products selected")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summarize_products(products, output / "summary.json")
    bins = sorted({m["bin"] for m, _ in products})
    cases = sorted({m["case"] for m, _ in products})

    def finish(fig, name):
        for ax in fig.axes:
            ax.grid(alpha=0.2)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=6)
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{name}.{extension}", dpi=200, bbox_inches="tight")
        plt.close(fig)

    def primary(case, q):
        subset = [(m, d) for m, d in products if m["case"] == case and m["bin"] == q]
        if not subset:
            return []
        budget = max(m["budget"] for m, _ in subset)
        return [(m, d) for m, d in subset if m["budget"] == budget]

    fig, axes = plt.subplots(len(bins), 3, figsize=(13, 3 * len(bins)), squeeze=False)
    for row, q in zip(axes, bins):
        for case in cases:
            for m, d in primary(case, q):
                with h5py.File(locations[m["fingerprint"]]) as handle:
                    x, curves = cumulative_distribution(handle["sorted_amplitude_km_s"][:])
                # Plot all tails; decimate only rendering, never saved distribution data.
                indices = (
                    np.unique(
                        np.r_[
                            np.linspace(0, max(0, len(x) - 1), 1500).astype(int),
                            np.arange(max(0, len(x) - 1000), len(x)),
                        ]
                    )
                    if len(x)
                    else []
                )
                color = P_MODEL_RUN_COLORS.get(m["p_value"], "grey")
                for ax, n, curve in zip(row, ORDERS, curves):
                    ax.plot(
                        x[indices],
                        curve[indices],
                        color=color,
                        alpha=0.6,
                        label=f"{case}, seed {m['seed']}",
                    )
                    ax.set(
                        xlabel=r"$x/\sqrt{S_2}$",
                        ylabel=f"C{n}(x)",
                        title=f"{m['center_km']:.0f} km, n={n}",
                        xscale="symlog",
                    )
    finish(fig, "A_cumulative")

    fig, axes = plt.subplots(3, 2, figsize=(12, 11), squeeze=False)
    for case in cases:
        for i, n in enumerate(ORDERS):
            for seed in sorted({m["seed"] for m, _ in products if m["case"] == case}):
                selected = [
                    pair for q in bins for pair in primary(case, q) if pair[0]["seed"] == seed
                ]
                x = [m["center_km"] for m, _ in selected]
                for t, style in enumerate((":", "--", "-")):
                    axes[i, 0].plot(
                        x,
                        [d["tail_contribution"][i, t] for _, d in selected],
                        style,
                        color=P_MODEL_RUN_COLORS.get(selected[0][0]["p_value"], "grey")
                        if selected
                        else None,
                        label=f"{case}, s{seed}, top {100 * TAILS[t]:g}%",
                    )
                axes[i, 1].plot(
                    x,
                    [d["N_conc"][i] / max(1, d["count"]) for _, d in selected],
                    "o-",
                    label=f"{case}, s{seed}",
                    color=P_MODEL_RUN_COLORS.get(selected[0][0]["p_value"], "grey")
                    if selected
                    else None,
                )
            for ax in axes[i]:
                ax.set(xscale="log", xlabel="Separation [km]", title=f"n={n}")
            axes[i, 0].set_ylabel("Largest-increment contribution fraction")
            axes[i, 1].set_ylabel("N_conc / N (concentration only)")
    finish(fig, "B_tail_concentration")

    fig, axes = plt.subplots(
        len(bins), len(cases), figsize=(7 * len(cases), 3 * len(bins)), squeeze=False
    )
    for row, q in zip(axes, bins):
        for ax, case in zip(row, cases):
            for m, d in primary(case, q):
                centers = 0.5 * (d["block_edges_s"][:-1] + d["block_edges_s"][1:])
                ax.plot(
                    centers,
                    d["block_contribution"][2],
                    "o-",
                    ms=3,
                    alpha=0.6,
                    label=f"seed {m['seed']}",
                    color=P_MODEL_RUN_COLORS.get(m["p_value"], "grey"),
                )
                extreme = d["block_extreme_counts"][0] > 0
                ax.scatter(
                    centers[extreme],
                    d["block_contribution"][2, extreme],
                    marker="x",
                    color="black",
                    s=20,
                    label="top 0.1% present",
                )
                ax.set(
                    title=f"{case}, {m['center_km']:.0f} km",
                    xlabel="Pair midpoint [s]",
                    ylabel="S6 block contribution fraction",
                )
    finish(fig, "C_temporal")

    fig, axes = plt.subplots(len(bins), 6, figsize=(24, 3 * len(bins)), squeeze=False)
    for row, q in zip(axes, bins):
        for case in cases:
            for seed in sorted({m["seed"] for m, _ in products if m["case"] == case}):
                selected = sorted(
                    [
                        (m, d)
                        for m, d in products
                        if m["case"] == case and m["bin"] == q and m["seed"] == seed
                    ],
                    key=lambda pair: pair[0]["budget"],
                )
                for i, n in enumerate(ORDERS):
                    for j in range(2):
                        ax = row[2 * i + j]
                        if selected:
                            for tail in [None] if j == 0 else range(len(TAILS)):
                                values = [
                                    d["S_n"][i] if tail is None else d["tail_contribution"][i, tail]
                                    for _, d in selected
                                ]
                                style = "-" if tail is None else (":", "--", "-")[tail]
                                label = f"{case}, s{seed}"
                                if tail is not None:
                                    label += f", top {100 * TAILS[tail]:g}%"
                                ax.plot(
                                    [m["budget"] for m, _ in selected],
                                    values,
                                    marker="o",
                                    linestyle=style,
                                    alpha=0.6,
                                    color=P_MODEL_RUN_COLORS.get(selected[0][0]["p_value"], "grey"),
                                    label=label,
                                )
                        ax.set(
                            xscale="log",
                            xlabel="Requested pairs / bin (capped at census)",
                            ylabel=f"S{n} [(km/s)^{n}]"
                            if j == 0
                            else f"S{n}: largest-event fractions",
                            title=f"{np.sqrt(EDGES[q] * EDGES[q + 1]):.0f} km",
                        )
                        if j == 0:
                            ax.set_yscale("log")
    finish(fig, "D_budget_convergence")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--cases", nargs="+", default=["malara-ts-500a", "malara-ts-502a"])
    parser.add_argument(
        "--duration", type=float, help="duration [s]; default: case suffix, otherwise 1500"
    )
    parser.add_argument(
        "--separations", nargs="+", type=float, help="replace default target separations [km]"
    )
    parser.add_argument("--bins", nargs="+", type=int, help="additional zero-based bin indices")
    parser.add_argument("--budgets", nargs="+", type=int, default=[500000])
    parser.add_argument("--seeds", nargs="+", type=int, default=[12345, 23456, 34567])
    parser.add_argument("--block-width", type=float, default=30.0)
    parser.add_argument("--top-count", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=100000)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/structure_function_moment_convergence")
    )
    parser.add_argument("--dry-run", "--preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args(argv)
    if (
        (args.duration is not None and (not np.isfinite(args.duration) or args.duration <= 0))
        or not np.isfinite(args.block_width)
        or args.block_width <= 0
        or args.top_count < 0
        or args.chunk_size <= 0
        or any(b <= 0 for b in args.budgets)
        or any(s < 0 for s in args.seeds)
        or len(set(args.budgets)) != len(args.budgets)
        or len(set(args.seeds)) != len(args.seeds)
    ):
        parser.error(
            "require positive duration/block width/budgets/chunk size, unique budgets/seeds, "
            "and nonnegative seeds/top count"
        )
    args.output = args.output.resolve()
    manifest_path = args.output / "manifest.json"
    if args.plot_only:
        manifest = json.loads(manifest_path.read_text())
        plot_saved([args.output / p for p in manifest["products"]], args.output / "figures")
        return
    selected = select_bins(args.separations, args.bins)
    paths = []
    for case in args.cases:
        suffix = re.search(r"-t(\d+(?:\.\d+)?)$", case)
        duration = (
            args.duration if args.duration is not None else (float(suffix[1]) if suffix else 1500.0)
        )
        metadata, plan, positions, fields, times = prepare_case(args.root, case, duration)
        for q in selected:
            print(
                f"{metadata['case']} bin={q} [{EDGES[q]:.3f}, {EDGES[q + 1]:.3f}] km; "
                f"eligible={plan.populations[q]}; "
                f"selected={[min(b, int(plan.populations[q])) for b in args.budgets]}",
                flush=True,
            )
            if args.dry_run:
                continue
            args.output.mkdir(parents=True, exist_ok=True)
            for seed in args.seeds:
                for budget in sorted(args.budgets):
                    meta = dict(
                        metadata,
                        bin=q,
                        edges_km=EDGES[q : q + 2],
                        center_km=float(np.sqrt(EDGES[q] * EDGES[q + 1])),
                        seed=seed,
                        budget=budget,
                        eligible=int(plan.populations[q]),
                        block_width_s=args.block_width,
                        top_count=args.top_count,
                    )
                    key = fingerprint(meta)
                    meta["fingerprint"] = key
                    path = args.output / f"{metadata['case']}_b{q}_s{seed}_N{budget}_{key[:16]}.h5"
                    if args.resume and path.exists():
                        read_product(path, key, distribution=False)
                        print(f"resume: {path.name}", flush=True)
                    else:
                        records = sample_records(
                            plan, positions, fields, times, q, budget, seed, args.chunk_size
                        )
                        diagnostics = moment_diagnostics(
                            records, (times[0], times[-1]), args.block_width, args.top_count
                        )
                        save_product(path, meta, diagnostics)
                        del records, diagnostics
                        print(f"saved: {path.name}", flush=True)
                    paths.append(path)
    if not args.dry_run:
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                dict(version=VERSION, caution=CAUTION, products=[p.name for p in paths]), indent=2
            )
            + "\n"
        )
        temporary.replace(manifest_path)
        plot_saved(paths, args.output / "figures")


if __name__ == "__main__":
    main()
