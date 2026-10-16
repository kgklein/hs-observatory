"""Matched raw-Malara and ideal-Elsasser moments on saved nominal HS trajectories.

No trajectory generation, measurement MC, or production work at import/preflight.
Pair indices are streamed once and shared by every quantity and contribution.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
import yaml

from hs_observatory.analysis.derived.elsasser import magnetic_field_to_alfven_units
from hs_observatory.analysis.derived.geometry_pairs import (
    ALGORITHM_VERSION,
    GeometryPlanCache,
    _affine_model,
    geometry_pair_chunks,
)
from hs_observatory.analysis.elsasser import compute_elsasser_analysis
from hs_observatory.analysis.structure_function import (
    _bin_indices,
    _fit_scaling_exponents,
    structure_function_moment_means,
    structure_function_moment_sums,
)
from hs_observatory.analysis.structure_function_duration import (
    file_hash,
    fingerprint,
    physical_field,
    save_json,
)
from hs_observatory.analysis.structure_function_local import local_log_slopes, p_model_exponents
from hs_observatory.fields.malara import MalaraTurbulenceField

VERSION = 1
ORDERS = list(range(1, 7))
QUANTITIES = ("raw_q", "centered_q", "ideal_z_plus")
CONTRIBUTIONS = ("all", "same_spacecraft", "cross_spacecraft")
EDGES = np.geomspace(70.0, 20000.0, 41)
WINDOWS = {"primary": [150.0, 1200.0], "sensitivity": [200.0, 1000.0]}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_equal(actual, expected, name, *, atol=1e-9, rtol=1e-10):
    require(
        actual.shape == expected.shape
        and np.all(np.isfinite(actual))
        and np.allclose(actual, expected, atol=atol, rtol=rtol),
        f"{name} does not match generation metadata / reconstructed field",
    )
    return float(np.max(np.abs(actual - expected), initial=0))


def preflight(root, recipe):
    """Validate saved coordinate/field contracts without evaluating the Malara field."""
    recipe = root / recipe
    generation = yaml.safe_load(recipe.read_text())
    source = root / generation["output"]["path"]
    with h5py.File(source) as h:
        metadata = dict(h["metadata"].attrs)
        embedded = yaml.safe_load(metadata["config_yaml"])
        require(
            physical_field(generation) == physical_field(embedded),
            "generation recipe and embedded physical field differ",
        )
        for key in ("time", "spacecraft", "measurement"):
            require(generation.get(key) == embedded.get(key), f"{key} metadata differs from recipe")
        require(embedded["field"]["model"] == "malara_turbulence", "expected Malara field")
        require(
            metadata.get("velocity_truth_frame", "plasma") == "plasma",
            "/truth/v must use the plasma-frame contract",
        )
        spacecraft_metadata = json.loads(metadata.get("spacecraft_source", "{}"))
        require(
            spacecraft_metadata.get("position_units", "km") == "km",
            "nominal spacecraft coordinates must be km",
        )
        adapter = MalaraTurbulenceField(embedded["field"])
        outputs = embedded["field"]["output_fields"]
        offsets = [int(outputs[q].get("realization_offset", 0)) for q in ("v", "B")]
        require(offsets[0] == offsets[1], "v and B must share a field realization")
        require(
            all(adapter.fluctuation_modes[q] == "malara" for q in ("v", "B")),
            "both vector outputs must use Malara fluctuations",
        )
        require(adapter.background_n > 0, "constant positive density required")
        t = np.asarray(h["time/v"])
        require(
            t.size >= 2 and np.all(np.isfinite(t)) and np.all(np.diff(t) > 0),
            "invalid time coordinates",
        )
        positions = np.asarray(h["spacecraft/r/v"]).swapaxes(0, 1)
        require(
            positions.ndim == 3
            and positions.shape[0] == len(t)
            and positions.shape[-1] == 3
            and np.all(np.isfinite(positions)),
            "invalid nominal position coordinates",
        )
        # Initial cases have identical cadences. Reject interpolation rather than
        # asserting an algebraic identity across nonlinear interpolated fields.
        for q in ("n", "v", "B"):
            check_equal(np.asarray(h[f"time/{q}"]), t, f"{q} time", atol=0, rtol=0)
            check_equal(
                np.asarray(h[f"spacecraft/r/{q}"]).swapaxes(0, 1),
                positions,
                f"{q} nominal coordinates",
                atol=0,
                rtol=0,
            )
            shape = positions.shape[:2] if q == "n" else positions.shape
            values = np.asarray(h[f"truth/{q}"])
            require(values.shape == shape and np.all(np.isfinite(values)), f"invalid truth/{q}")
        check_equal(
            np.asarray(h["truth/n"]),
            np.full(positions.shape[:2], adapter.background_n),
            "constant density",
        )
        configured = embedded["time"]
        check_equal(
            t[[0, -1]], np.array([configured["t_start_s"], configured["t_stop_s"]]), "time extent"
        )
        cadence = configured["cadence_s"]
        for q in ("n", "v", "B"):
            dt = cadence[q] if isinstance(cadence, dict) else cadence
            check_equal(np.diff(t), np.full(len(t) - 1, dt), f"{q} cadence")
        ids = [x.decode() if isinstance(x, bytes) else str(x) for x in h["spacecraft/id"][...]]
        require(len(ids) == positions.shape[1], "spacecraft IDs / position shape mismatch")
        _affine_model(positions)
        effective = asdict(adapter._quantity_evaluators["v"].config)
    conversion = magnetic_field_to_alfven_units(
        np.ones((1, 1, 3)), np.array([adapter.background_n])
    )[0, 0, 0]
    k = adapter.fluctuation_amplitudes["v"] + conversion * adapter.fluctuation_amplitudes["B"]
    require(np.isfinite(k) and k != 0, "nonzero finite K required")
    return {
        "recipe": str(recipe.relative_to(root)),
        "source": str(source.relative_to(root)),
        "recipe_yaml": recipe.read_text(),
        "generation": embedded,
        "input_sha256": file_hash(source),
        "field_config": effective,
        "origin_km": adapter.origin,
        "spacecraft_metadata": spacecraft_metadata,
        "spacecraft_ids": ids,
        "metadata": metadata,
        "K": k,
        "coordinate_convention": "nominal r in km; q evaluated at (r-origin) modulo L; "
        "pair separation is unwrapped Euclidean |r_b-r_a|",
        "shape": positions.shape,
    }, positions


def reconstruct(root, entry, positions, *, backend=None, chunk_size=1000, workers=None):
    """Evaluate the saved realization in bounded chunks and verify native ideal truth."""
    field = deepcopy(entry["generation"]["field"])
    turbulence = field["turbulence"]
    if backend is not None:
        turbulence["backend"] = backend
    if workers is not None:
        turbulence["n_workers"] = workers
    turbulence["chunk_size"] = chunk_size
    adapter = MalaraTurbulenceField(field)
    evaluator = adapter._quantity_evaluators["v"]
    flat = positions.reshape(-1, 3)
    raw = np.empty_like(flat)
    # Bound parallel dispatch as well as cached-eddy storage, regardless of backend.
    batch = chunk_size * max(1, adapter.n_workers or 1)
    for start in range(0, len(flat), batch):
        raw[start : start + batch] = adapter._evaluate_turbulence(
            evaluator, flat[start : start + batch] - adapter.origin
        )
        evaluator.clear_cache()
    raw = raw.reshape(positions.shape)
    checks = {}
    with h5py.File(root / entry["source"]) as h:
        for quantity in ("v", "B"):
            expected = (
                adapter.background_vectors[quantity]
                + adapter.fluctuation_amplitudes[quantity] * raw
            )
            checks[quantity] = check_equal(
                np.asarray(h[f"truth/{quantity}"]), expected, f"reconstructed ideal {quantity}"
            )
    truth = compute_elsasser_analysis(root / entry["source"], {"data_group": "truth"})
    centered = raw - raw.mean(axis=0, keepdims=True)
    checks["z_plus_over_K"] = check_equal(
        truth["z_plus"] / entry["K"], centered, "z_plus/K versus q-mean(q)"
    )
    return {"raw_q": raw, "centered_q": centered, "ideal_z_plus": truth["z_plus"]}, checks


def accumulate(positions, quantities, edges, pair_iterator, k):
    """One selected pair stream for all quantities; never replay an RNG per quantity."""
    n_sc = positions.shape[1]
    flat = {key: np.asarray(quantities[key]).reshape(-1, 3) for key in QUANTITIES}
    require(
        all(value.shape == positions.reshape(-1, 3).shape for value in flat.values()),
        "all quantities must use identical spacecraft-time nodes",
    )
    nb = len(edges) - 1
    counts = {group: np.zeros((1, nb), dtype=np.int64) for group in CONTRIBUTIONS}
    sums = {group: {q: np.zeros((1, 6, nb)) for q in QUANTITIES} for group in CONTRIBUTIONS}
    digest = hashlib.sha256()
    position = positions.reshape(-1, 3)
    residual = {"centered_vs_rescaled_increment_max_abs": 0.0, "same_sc_cancellation_max_abs": 0.0}
    for left, right in pair_iterator:
        # Interleaved, canonical bytes make the digest independent of chunk boundaries.
        digest.update(np.column_stack((left, right)).astype("<i8").tobytes())
        index, valid = _bin_indices(np.linalg.norm(position[right] - position[left], axis=1), edges)
        same = left % n_sc == right % n_sc
        increments = {q: values[right] - values[left] for q, values in flat.items()}
        rescaled = increments["ideal_z_plus"] / k
        residual["centered_vs_rescaled_increment_max_abs"] = max(
            residual["centered_vs_rescaled_increment_max_abs"],
            check_equal(rescaled, increments["centered_q"], "matched centered increments"),
        )
        residual["same_sc_cancellation_max_abs"] = max(
            residual["same_sc_cancellation_max_abs"],
            check_equal(
                increments["raw_q"][same],
                increments["centered_q"][same],
                "same-spacecraft mean cancellation",
            ),
        )
        for group, selection in (
            ("all", valid),
            ("same_spacecraft", valid & same),
            ("cross_spacecraft", valid & ~same),
        ):
            for bi in np.unique(index[selection]):
                mask = selection & (index == bi)
                counts[group][0, bi] += np.count_nonzero(mask)
                for q in QUANTITIES:
                    sums[group][q][0, :, bi] += structure_function_moment_sums(
                        np.linalg.norm(increments[q][mask], axis=1), ORDERS
                    )
    powers = np.abs(k) ** np.asarray(ORDERS)
    products = {}
    for group in CONTRIBUTIONS:
        products[group] = {"counts": counts[group], "quantities": {}}
        for q in QUANTITIES:
            moments = structure_function_moment_means(sums[group][q], counts[group])
            if q == "ideal_z_plus":
                moments = moments / powers[None, :, None]
            products[group]["quantities"][q] = {
                "S_n": moments,
                "moment_sums_native": sums[group][q],
            }
        a = products[group]["quantities"]
        require(
            np.allclose(
                a["centered_q"]["S_n"],
                a["ideal_z_plus"]["S_n"],
                rtol=1e-9,
                atol=1e-12,
                equal_nan=True,
            ),
            "rescaled moments differ",
        )
    return {
        "contributions": products,
        "pair_index_sha256": digest.hexdigest(),
        "increment_checks": residual,
    }


def derive(record, edges):
    centers = np.sqrt(edges[:-1] * edges[1:])
    for product in record["contributions"].values():
        counts = np.asarray(product["counts"])
        for values in product["quantities"].values():
            moments = np.asarray(values["S_n"])
            values["fits"] = _fit_scaling_exponents(moments, counts, centers, ORDERS, WINDOWS, 3)
            _, slopes, _ = local_log_slopes(centers, moments, counts=counts)
            values["local_slopes"] = slopes
        q = product["quantities"]
        product["raw_minus_centered_exponents"] = np.asarray(
            q["raw_q"]["fits"]["raw"]
        ) - np.asarray(q["centered_q"]["fits"]["raw"])
    return record


def validate_completed(path, signature, expected):
    """Validate checksum, counts, and moment shapes before skipping a completed seed."""
    try:
        record = json.loads(path.read_text())
        payload = {key: value for key, value in record.items() if key != "payload_sha256"}
        require(record["payload_sha256"] == fingerprint(payload), "moment checksum mismatch")
        require(record["signature"] == signature, "signature mismatch")
        groups = record["contributions"]
        np.testing.assert_array_equal(groups["all"]["counts"], expected[None])
        np.testing.assert_array_equal(
            np.asarray(groups["all"]["counts"]),
            np.asarray(groups["same_spacecraft"]["counts"]) + groups["cross_spacecraft"]["counts"],
        )
        for product in groups.values():
            count = np.asarray(product["counts"])
            for q in QUANTITIES:
                moments = np.asarray(product["quantities"][q]["S_n"])
                require(moments.shape == (1, 6, len(expected)), "invalid moment shape")
                require(
                    np.all(np.isfinite(moments[..., count[0] > 0])), "nonfinite populated moments"
                )
        return record
    except (OSError, ValueError, KeyError, TypeError, AssertionError):
        return None


def direct_cases(paths, entry):
    """Load only physically matching isotropic exact-separation raw-q results."""
    from hs_observatory.analysis.spatial_structure_function import read_spatial_result

    found = []
    for path in paths:
        result = read_spatial_result(path)
        for case in result["cases"]:
            if float(case["p_value"]) != entry["field_config"]["intermittency_p"]:
                continue
            require(
                fingerprint(case["field_config"]) == fingerprint(entry["field_config"]),
                f"{path}: direct-3D field configuration differs",
            )
            found.append((path, case))
    require(not paths or found, "no direct-3D case matches this p value")
    return found


def plot(records, entry, output, edges, spatial=()):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers = np.sqrt(edges[:-1] * edges[1:])
    theory = p_model_exponents(np.asarray(ORDERS), entry["field_config"]["intermittency_p"])
    labels = ("raw q", "q minus spacecraft mean", "ideal z+ / |K|^n")
    styles = ("-", "--", ":")
    colors = ("C0", "C1", "C2")
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for qi, q in enumerate(QUANTITIES):
        products = [r["contributions"]["all"]["quantities"][q] for r in records]
        moments = np.mean([p["S_n"] for p in products], axis=0)[0]
        for ni, n in enumerate(ORDERS):
            axes[0, 0].loglog(
                centers, moments[ni], styles[qi], color=f"C{ni}", label=f"n={n}, {labels[qi]}"
            )
        fits = np.array([p["fits"]["raw"][0] for p in products])
        for wi, name in enumerate(WINDOWS):
            # Seed variation is pair-selection variation, not independent field uncertainty.
            mean = np.mean(fits[:, wi], axis=0)
            axes[0, 1].plot(
                ORDERS,
                mean,
                styles[qi],
                color=colors[qi],
                marker="o" if wi == 0 else "x",
                label=f"{labels[qi]}, {name}",
            )
            for seed_fit in fits[:, wi]:
                axes[0, 1].plot(ORDERS, seed_fit, styles[qi], color=colors[qi], alpha=0.15)
        slopes = np.mean([p["local_slopes"] for p in products], axis=0)[0]
        for axis, n in zip(axes[1], (3, 4, 6)):
            axis.semilogx(centers, slopes[n - 1], styles[qi], color=colors[qi], label=labels[qi])
            axis.set_title(f"Local slope n={n}")
    for wi, name in enumerate(WINDOWS):
        differences = np.array(
            [r["contributions"]["all"]["raw_minus_centered_exponents"][0][wi] for r in records]
        )
        axes[0, 2].plot(ORDERS, np.mean(differences, axis=0), marker="o", label=name)
        for row in differences:
            axes[0, 2].plot(ORDERS, row, alpha=0.2)
    for path, case in spatial:
        x = case["lambda_bin_centers_km"]
        values = np.asarray(case["q"]["S_n"])
        counts = np.asarray(case["counts"])
        fit = _fit_scaling_exponents(values, counts, x, ORDERS, WINDOWS, 3)
        label = f"Isotropic exact-separation ({Path(path).stem})"
        axes[0, 1].plot(ORDERS, fit["raw"][0, 0], "-.", label=label)
        _, slopes, _ = local_log_slopes(x, values, counts=counts)
        for axis, n in zip(axes[1], (3, 4, 6)):
            axis.semilogx(x, slopes[0, n - 1], "-.", label=label)
        for ni in range(6):
            axes[0, 0].loglog(x, values[0, ni], "-.", color=f"C{ni}")
    axes[0, 0].set(
        title="Finite-width HS bins: mean moments across selection seeds",
        ylabel="S_n",
        xlabel="Nominal separation [km]",
    )
    axes[0, 1].plot(ORDERS, theory, "k-", label="Analytic p-model")
    axes[0, 1].set(
        title="Fitted exponents (individual seeds faint)", xlabel="Order n", ylabel="xi_n"
    )
    axes[0, 2].axhline(0, color="k", lw=0.7)
    axes[0, 2].set(
        title="Raw minus mean-subtracted exponent", xlabel="Order n", ylabel="Delta xi_n"
    )
    for axis, n in zip(axes[1], (3, 4, 6)):
        axis.axhline(theory[n - 1], color="k", lw=0.7)
        axis.axvspan(300, 2000, color="grey", alpha=0.08)
        axis.set(
            xlim=(300, 2000), xlabel="Nominal separation [km]", ylabel="d log S_n / d log lambda"
        )
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=6)
    figure.suptitle(
        f"p={entry['field_config']['intermittency_p']:g}; "
        "one realization; seed spread measures pair selection only\n"
        "Direct-3D overlays, if present, are isotropic exact-separation estimates"
    )
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"diagnostics.{suffix}", dpi=160)
    plt.close(figure)


def write_report(records, manifest, output, direct_paths):
    """Expose conditional pair populations and mean-subtraction effects for review."""
    entry = manifest["input"]
    lines = [
        f"# Raw Malara HS comparison: p={entry['field_config']['intermittency_p']:g}",
        "",
        f"Input: `{entry['source']}`; git commit: `{manifest['git_commit']}`.",
        f"Realization: `{entry['field_config']}`; origin [km]: `{entry['origin_km']}`.",
        f"K = {entry['K']:.12g}. Ideal z+ moments are divided by |K|^n.",
        "",
        entry["coordinate_convention"],
        "",
        "All three quantities share identical selected spacecraft-time indices. "
        "Stored ideal truth is reprocessed with the existing Elsasser routine; "
        "older independently sampled SF caches are not treated as matched.",
        "Seed variation reflects pair selection in one realization, not independent "
        "realization uncertainty. Each bin is capped at its eligible population.",
        "",
        "| Seed | All pairs | Same spacecraft | Cross spacecraft |",
        "| --- | ---: | ---: | ---: |",
    ]
    for record in records:
        counts = [int(np.sum(record["contributions"][group]["counts"])) for group in CONTRIBUTIONS]
        lines.append(f"| {record['seed']} | {counts[0]} | {counts[1]} | {counts[2]} |")
    lines += [
        "",
        "Per-bin counts, native moment sums, rescaled moments, local slopes, both fit "
        "windows, reconstruction residuals and the shared pair-index digest are in each "
        "seed JSON. The manifest records requested and eligible bin populations.",
        "",
        "| Contribution | Fit [km] | n | Raw xi | Centered xi | Rescaled ideal xi | Raw - centered |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in CONTRIBUTIONS:
        for wi, (name, bounds) in enumerate(WINDOWS.items()):
            fits = np.mean(
                [
                    [
                        record["contributions"][group]["quantities"][q]["fits"]["raw"][0][wi]
                        for q in QUANTITIES
                    ]
                    for record in records
                ],
                axis=0,
            )
            for ni, order in enumerate(ORDERS):
                raw, centered, ideal = fits[:, ni]
                lines.append(
                    f"| {group} | {name} {bounds[0]:g}–{bounds[1]:g} | {order} | "
                    f"{raw:.6g} | {centered:.6g} | {ideal:.6g} | {raw - centered:.6g} |"
                )
    if direct_paths:
        lines += [
            "",
            "Direct-3D overlays are **isotropic exact-separation estimates**. "
            "HS estimates average finite-width separation bins along restricted trajectories. "
            "These estimators do not use matched pairs.",
            "",
            "Direct-3D inputs: " + ", ".join(f"`{path}`" for path in direct_paths),
        ]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="analysis/malara-raw-hs-500a.yaml")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--fit-pairs", type=int)
    parser.add_argument("--outside-pairs", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--chunk-size", type=int, default=100000, help="pair stream chunk size")
    parser.add_argument("--field-chunk-size", type=int, default=500)
    parser.add_argument("--backend", choices=("direct", "chunked", "cached", "parallel"))
    parser.add_argument("--workers", type=int)
    parser.add_argument("--dry-run", "--preflight", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--direct3d", type=Path, nargs="+", default=[])
    args = parser.parse_args(argv)
    root = args.root.resolve()
    config = yaml.safe_load((root / args.config).read_text())
    output = root / config["output"]["directory"]
    if args.plot_only:
        manifest = json.loads((output / "manifest.json").read_text())
        edges = np.asarray(manifest["edges_km"])
        records = []
        for record in manifest["records"]:
            saved = validate_completed(
                output / record["path"],
                record["signature"],
                np.asarray(manifest["expected_counts"]),
            )
            require(saved is not None, f"invalid completed result: {record['path']}")
            records.append(derive(saved, edges))
        entry = manifest["input"]
    else:
        edges = EDGES
        fit_pairs = (
            args.fit_pairs if args.fit_pairs is not None else config["sampling"]["fit_pairs"]
        )
        outside = (
            args.outside_pairs
            if args.outside_pairs is not None
            else config["sampling"]["outside_pairs"]
        )
        seeds = args.seeds if args.seeds is not None else config["sampling"]["seeds"]
        require(
            all(x > 0 for x in (fit_pairs, outside, args.chunk_size, args.field_chunk_size)),
            "budgets and chunks must be positive",
        )
        require(args.workers is None or args.workers > 0, "workers must be positive")
        require(
            bool(seeds) and len(set(seeds)) == len(seeds) and all(s >= 0 for s in seeds),
            "seeds must be nonnegative and unique",
        )
        entry, positions = preflight(root, config["generation"])
        print(f"Validated {entry['source']}: {entry['shape']}; K={entry['K']:.12g}", flush=True)
        if args.dry_run:
            print(
                f"No field evaluation or pair selection. Seeds={seeds}, fit={fit_pairs}, outside={outside}"
            )
            return 0
        centers = np.sqrt(EDGES[:-1] * EDGES[1:])
        targets = np.where((centers >= 150) & (centers <= 1200), fit_pairs, outside)
        plan, _ = GeometryPlanCache(max_entries=1).get(positions, EDGES, 70.0, 20000.0)
        expected = np.minimum(plan.populations, targets)
        code = [
            Path(__file__),
            Path(__file__).with_name("structure_function.py"),
            Path(__file__).parent / "derived/geometry_pairs.py",
            Path(__file__).parent / "derived/elsasser.py",
            Path(__file__).with_name("elsasser.py"),
            Path(__file__).with_name("structure_function_duration.py"),
            Path(__file__).with_name("structure_function_local.py"),
            Path(__file__).parents[1] / "fields/malara.py",
        ]
        hashes = {str(p.relative_to(Path(__file__).parents[1])): file_hash(p) for p in code}
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
        manifest = {
            "version": VERSION,
            "input": entry,
            "config": config,
            "git_commit": git.stdout.strip() if git.returncode == 0 else "unavailable",
            "code_hashes": hashes,
            "edges_km": EDGES,
            "orders": ORDERS,
            "windows_km": WINDOWS,
            "numpy_version": np.__version__,
            "sampler": ALGORITHM_VERSION,
            "seeds": seeds,
            "requested_counts": targets,
            "expected_counts": expected,
            "eligible_counts": plan.populations,
            "records": [],
            "evaluation": {
                "backend": args.backend,
                "workers": args.workers,
                "field_chunk_size": args.field_chunk_size,
            },
            "comparison": "All quantities use one identical pair-index stream; ideal truth "
            "is reprocessed, never assumed matched to older sampled products.",
        }
        values = None
        records = []
        output.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            signature = fingerprint(
                {
                    "version": VERSION,
                    "input": entry,
                    "code": hashes,
                    "edges": EDGES,
                    "targets": targets,
                    "seed": seed,
                    "sampler": ALGORITHM_VERSION,
                    "numpy": np.__version__,
                }
            )
            path = output / f"seed{seed}_{signature[:16]}.json"
            saved = None if args.force else validate_completed(path, signature, expected)
            if saved is None:
                if values is None:
                    values, checks = reconstruct(
                        root,
                        entry,
                        positions,
                        backend=args.backend,
                        chunk_size=args.field_chunk_size,
                        workers=args.workers,
                    )
                stream = geometry_pair_chunks(
                    plan, targets, random_seed=seed, chunk_size=args.chunk_size
                )
                saved = accumulate(positions, values, EDGES, stream, entry["K"])
                np.testing.assert_array_equal(saved["contributions"]["all"]["counts"][0], expected)
                require(
                    file_hash(root / entry["source"]) == entry["input_sha256"],
                    "input changed during computation",
                )
                saved.update(signature=signature, seed=seed, reconstruction_checks=checks)
                derive(saved, EDGES)
                saved["payload_sha256"] = fingerprint(saved)
                save_json(path, saved)
                print(f"Saved {path}", flush=True)
            else:
                print(f"Reuse validated {path}", flush=True)
            records.append(derive(saved, EDGES))
            manifest["records"].append({"path": path.name, "signature": signature, "seed": seed})
            save_json(output / "manifest.json", manifest)
    require(bool(records), "no completed seed records")
    spatial = direct_cases([root / p for p in args.direct3d], entry)
    plot(records, entry, output, edges, spatial)
    write_report(records, manifest, output, args.direct3d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
