"""Truth-only observation-duration study with independently cached seed moments."""

import argparse
from copy import deepcopy
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
    GeometryPlanCache,
    _affine_model,
)
from hs_observatory.analysis.elsasser import compute_elsasser_analysis
from hs_observatory.analysis.structure_function import (
    FIELDS,
    _fit_scaling_exponents,
    _positions_for_group_member,
    _settings,
    _to_serializable,
    compute_all_spacetime_structure_functions,
)
from hs_observatory.analysis.structure_function_convergence import FIT_WINDOWS, ORDERS
from hs_observatory.analysis.structure_function_local import local_log_slopes, p_model_exponents
from hs_observatory.analysis.structure_function_uniform_binning import separation_statistics

VERSION = "duration_convergence_v1"
DEFAULT_CONFIG = "analysis/malara-ts-502_sf-duration.yaml"


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(_to_serializable(value), sort_keys=True).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_to_serializable(value), indent=2))
    temporary.replace(path)


def discover_inputs(config, root, durations):
    """Resolve HDF5 paths from generation YAML output declarations, not filename suffixes."""
    found = {}
    for recipe in sorted(root.glob(config["duration_study"]["generation_glob"])):
        generation = yaml.safe_load(recipe.read_text())
        duration = float(generation["time"]["t_stop_s"]) - float(generation["time"]["t_start_s"])
        if duration not in durations:
            continue
        if duration in found:
            raise ValueError(f"ambiguous generation recipes for {duration:g} s")
        source = (root / generation["output"]["path"]).resolve()
        if not source.is_file() or not h5py.is_hdf5(source):
            raise ValueError(f"missing/non-HDF5 generation output declared by {recipe}: {source}")
        found[duration] = (recipe, source)
    missing = set(durations) - found.keys()
    if missing:
        raise ValueError(f"no generation recipe for durations: {sorted(missing)}")
    return found


def compare_prefix(short, long, *, atol, rtol):
    """Numerical comparison including shape, exact equality and explicit tolerance."""
    if short.shape != long.shape:
        return {
            "within_tolerance": False,
            "exact": False,
            "max_abs_difference": None,
            "atol": atol,
            "rtol": rtol,
            "shape_mismatch": [short.shape, long.shape],
        }
    difference = np.abs(short - long)
    return {
        "within_tolerance": bool(np.allclose(short, long, atol=atol, rtol=rtol)),
        "exact": bool(np.array_equal(short, long)),
        "max_abs_difference": float(np.max(difference, initial=0)),
        "atol": atol,
        "rtol": rtol,
    }


def physical_field(config):
    field = deepcopy(config["field"])
    for key in ("backend", "chunk_size", "n_workers", "use_cache"):
        field.get("turbulence", {}).pop(key, None)
    return field


def validate_inputs(config, root, durations):
    """Read native truth only; compare every shorter record to the longest input."""
    found = discover_inputs(config, root, durations)
    records, arrays = {}, {}
    for duration, (recipe, source) in sorted(found.items()):
        with h5py.File(source) as handle:
            generation_yaml = handle["metadata"].attrs["config_yaml"]
            generation = yaml.safe_load(generation_yaml)
            if generation["field"]["model"] != "malara_turbulence":
                raise ValueError(f"{source}: expected Malara observations")
            field = generation["field"]
            domain = field["domain"]
            if domain.get("units", "km") != "km" or np.any(np.asarray(domain["L"]) <= 0):
                raise ValueError("coverage diagnostics require positive domain lengths in km")
            turbulence = field["turbulence"]
            p_model_exponents(np.array(ORDERS), turbulence["intermittency_p"])
            axes, data, native = {}, {}, {}
            for q in ("n", "v", "B"):
                t = np.asarray(handle[f"time/{q}"])
                r = np.asarray(handle[f"spacecraft/r/{q}"])
                z = np.asarray(handle[f"truth/{q}"])
                if (
                    len(t) < 2
                    or not np.all(np.isfinite(t))
                    or np.any(np.diff(t) <= 0)
                    or not np.all(np.isfinite(r))
                    or not np.all(np.isfinite(z))
                ):
                    raise ValueError(f"{source}: invalid native {q} samples")
                cadence = float(np.median(np.diff(t)))
                configured = generation["time"]["cadence_s"]
                configured = configured[q] if isinstance(configured, dict) else configured
                if (
                    not np.isclose(np.ptp(t), duration, atol=1e-9, rtol=0)
                    or not np.isclose(cadence, configured, atol=1e-10, rtol=1e-10)
                    or not np.allclose(np.diff(t), cadence, atol=1e-10, rtol=1e-10)
                ):
                    raise ValueError(f"{source}: duration/cadence differs from recipe at {q}")
                expected_ndim = 2 if q == "n" else 3
                if (
                    z.ndim != expected_ndim
                    or (q != "n" and z.shape[-1] != 3)
                    or r.shape != (z.shape[1], len(t), 3)
                    or z.shape[0] != len(t)
                ):
                    raise ValueError(f"{source}: inconsistent native array shapes")
                _affine_model(r.swapaxes(0, 1))
                axes[q], data[q] = t, (r, z)
                native[q] = {
                    "duration_s": float(np.ptp(t)),
                    "cadence_s": cadence,
                    "timestamps": len(t),
                    "start_s": float(t[0]),
                    "start_positions_km": r[:, 0],
                }
            ids = [
                x.decode() if isinstance(x, bytes) else str(x) for x in handle["spacecraft/id"][...]
            ]
            records[str(duration)] = {
                "duration_s": duration,
                "source": str(source),
                "input_sha256": file_hash(source),
                "recipe": str(recipe),
                "recipe_sha256": file_hash(recipe),
                "recipe_yaml": recipe.read_text(),
                "generation_yaml": generation_yaml,
                "generation_config": generation,
                "native": native,
                "spacecraft_ids": ids,
                "hdf5_metadata": {
                    key: value.decode() if isinstance(value, bytes) else value
                    for key, value in handle["metadata"].attrs.items()
                },
                "n_mc_in_file": int(handle["metadata"].attrs.get("n_mc", -1)),
                "field_parameters": physical_field(generation),
                "recipe_matches_embedded_field": physical_field(yaml.safe_load(recipe.read_text()))
                == physical_field(generation),
                "p": float(turbulence["intermittency_p"]),
                "base_seed": turbulence["base_seed"],
                "realization": turbulence["realization"],
                "injection_level": turbulence["injection_scale_index"],
                "maximum_level_inclusive": turbulence["n_scales"],
            }
            arrays[duration] = (axes, data)
    longest = max(durations)
    reference = records[str(longest)]
    mismatches = []
    for duration in sorted(durations):
        record = records[str(duration)]
        same_field = record["field_parameters"] == reference["field_parameters"]
        comparisons = {}
        for q in ("n", "v", "B"):
            t = arrays[duration][0][q]
            r, z = arrays[duration][1][q]
            lt = arrays[longest][0][q][: len(t)]
            lr, lz = arrays[longest][1][q]
            for name, short, long, atol in (
                ("time", t, lt, 1e-10),
                ("positions", r, lr[:, : len(t)], 1e-8),
                ("truth", z, lz[: len(t)], 1e-10),
            ):
                comparisons[f"{q}_{name}"] = compare_prefix(short, long, atol=atol, rtol=1e-12)
        record["prefix_comparisons"] = comparisons
        record["same_field_parameters"] = same_field
        record["same_spacecraft_ids"] = record["spacecraft_ids"] == reference["spacecraft_ids"]
        record["nested_within_tolerance"] = (
            same_field
            and record["same_spacecraft_ids"]
            and all(c["within_tolerance"] for c in comparisons.values())
        )
        record["exact_prefix"] = record["nested_within_tolerance"] and all(
            c["exact"] for c in comparisons.values()
        )
        if not record["nested_within_tolerance"] or not record["recipe_matches_embedded_field"]:
            mismatches.append(duration)
    return {
        "version": VERSION,
        "reference_duration_s": longest,
        "inputs": records,
        "valid_nested_sequence": not mismatches,
        "mismatched_durations": mismatches,
        "interpretation": "Conditional on one field realization; no measurement MC is read.",
    }


def coverage_diagnostics(positions, generation, cells=64):
    """Sparse periodic voxels are a resolution-dependent revisitation proxy, not independence."""
    domain = generation["field"]["domain"]
    lengths = np.asarray(domain["L"], dtype=float)
    origin = np.asarray(domain.get("origin", [0, 0, 0]), dtype=float)
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=-1)
    median_step = float(np.median(steps))
    stride = max(1, int(np.ceil(np.min(lengths / cells) / median_step))) if median_step else 1

    def occupancy(values):
        wrapped = np.mod(values - origin, lengths)
        indices = np.floor(wrapped / lengths * cells).astype(np.int64)
        unique = len(np.unique(indices.reshape(-1, 3), axis=0))
        return {
            "samples": int(indices.size // 3),
            "unique_voxels": unique,
            "repeat_fraction": 1 - unique / (indices.size // 3),
        }

    return {
        "domain_km": lengths,
        "distance_traveled_km_per_sc": steps.sum(axis=0),
        "coordinate_domain_fractions_per_sc": np.abs(positions[-1] - positions[0]) / lengths,
        "voxel_cells_per_axis": cells,
        "voxel_size_km": lengths / cells,
        "all_samples": occupancy(positions),
        "decimation_stride": stride,
        "decimated_samples": occupancy(positions[::stride]),
        "warning": "Coarse voxel revisits include adjacent/correlated samples; not independence.",
    }


def geometry_diagnostics(plan, positions):
    population_centers, _, _ = separation_statistics(
        plan, positions, np.zeros(len(plan.bins), dtype=int), 0
    )
    flow = positions[-1, 0] - positions[0, 0]
    flow_norm = np.linalg.norm(flow)
    histograms = []
    for b in plan.bins:
        g = b.groups
        if not len(g) or not flow_norm:
            histograms.append(np.zeros(10))
            continue
        starts = g[:, 3].copy()
        for row, explicit in b.explicit_starts.items():
            starts[row] = explicit[0]
        vectors = positions[starts + g[:, 2], g[:, 1]] - positions[starts, g[:, 0]]
        cosine = np.clip(
            np.abs(vectors @ flow / (np.linalg.norm(vectors, axis=1) * flow_norm)), 0, 1
        )
        histograms.append(np.histogram(cosine, bins=np.linspace(0, 1, 11), weights=g[:, 4])[0])
    return {
        "population_log_centers_km": population_centers,
        "eligible_groups_per_bin": [len(b.groups) for b in plan.bins],
        "eligible_per_bin": plan.populations,
        "abs_cos_flow_edges": np.linspace(0, 1, 11),
        "population_direction_histograms": histograms,
    }


def seed_summary(values):
    """Finite-aware mean, sample SD, min/max; a single seed has undefined SD."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    count = finite.sum(axis=0)
    mean = np.divide(
        np.where(finite, values, 0).sum(axis=0),
        count,
        out=np.full(values.shape[1:], np.nan),
        where=count > 0,
    )
    variance = np.divide(
        np.where(finite, (values - mean) ** 2, 0).sum(axis=0),
        count - 1,
        out=np.full_like(mean, np.nan),
        where=count > 1,
    )
    return {
        "mean": mean,
        "std": np.sqrt(variance),
        "n": count,
        "min": np.where(count, np.min(np.where(finite, values, np.inf), axis=0), np.nan),
        "max": np.where(count, np.max(np.where(finite, values, -np.inf), axis=0), np.nan),
    }


def local_peaks(centers, slopes, interval):
    slopes = np.asarray(slopes)
    locations = np.full(slopes.shape[:-1], np.nan)
    amplitudes = np.full_like(locations, np.nan)
    for index in np.ndindex(slopes.shape[:-1]):
        valid = (centers >= interval[0]) & (centers <= interval[1]) & np.isfinite(slopes[index])
        if np.any(valid):
            q = np.flatnonzero(valid)[np.argmax(slopes[index][valid])]
            locations[index], amplitudes[index] = centers[q], slopes[index][q]
    return {"separation_km": locations, "amplitude": amplitudes}


def sample(config, root, validation, seeds, budgets, force=False):
    output = root / config["output"]["directory"]
    settings = _settings(config["structure_function"])
    edges = np.asarray(settings["lambda_bins_km"])
    if (
        settings["increment_mode"] != "all_spacetime_pairs"
        or settings["pair_sampling"]["mode"] != "geometry_stratified"
    ):
        raise ValueError("duration study requires all_spacetime_pairs / geometry_stratified")
    centers = np.sqrt(edges[:-1] * edges[1:])
    inside = (centers >= 150) & (centers <= 1200)
    source_files = [
        Path(__file__),
        Path(__file__).with_name("structure_function.py"),
        Path(__file__).with_name("elsasser.py"),
        Path(__file__).with_name("structure_function_uniform_binning.py"),
        Path(__file__).parent / "derived/geometry_pairs.py",
        Path(__file__).parent / "derived/elsasser.py",
    ]
    code_hashes = {str(p.relative_to(Path(__file__).parent)): file_hash(p) for p in source_files}
    manifest = {
        "version": VERSION,
        "validation": validation,
        "records": [],
        "config": config,
        "seeds": seeds,
        "code_hashes": code_hashes,
    }
    cache = GeometryPlanCache(max_entries=1)
    for duration, entry in validation["inputs"].items():
        relevant = [
            b for b in budgets if b["durations"] is None or entry["duration_s"] in b["durations"]
        ]
        truth = positions = spatial = geometry = None
        for budget in relevant:
            requested = np.where(inside, budget["fit_pairs"], budget["outside_pairs"])
            for seed in seeds:
                signature = fingerprint(
                    {
                        "input": entry["input_sha256"],
                        "code": code_hashes,
                        "elsasser": settings["elsasser"],
                        "edges": edges,
                        "targets": requested,
                        "orders": ORDERS,
                        "seed": seed,
                        "sampler": ALGORITHM_VERSION,
                    }
                )
                path = (
                    output
                    / "moments"
                    / f"t{entry['duration_s']:g}"
                    / (
                        f"{budget['name']}_fit{budget['fit_pairs']}_out{budget['outside_pairs']}"
                        f"_seed{seed}_{signature[:16]}.json"
                    )
                )
                if not path.exists() or force:
                    tick = time.perf_counter()
                    if truth is None:
                        truth = compute_elsasser_analysis(
                            entry["source"], dict(settings["elsasser"], data_group="truth")
                        )
                        positions, _, _ = _positions_for_group_member(
                            entry["source"], "truth", 0, truth["time"], len(truth["spacecraft_ids"])
                        )
                        plan, _ = cache.get(positions, edges, 70.0, 20000.0)
                        spatial = coverage_diagnostics(
                            positions,
                            entry["generation_config"],
                            config["duration_study"]["coverage_cells_per_axis"],
                        )
                        geometry = geometry_diagnostics(plan, positions)
                    product = compute_all_spacetime_structure_functions(
                        positions[None],
                        truth["z_plus"][None],
                        truth["z_minus"][None],
                        ORDERS,
                        edges,
                        pair_sampling=dict(
                            settings["pair_sampling"],
                            random_seed=seed,
                            target_pairs_per_bin=requested,
                        ),
                        scale_ranges_km=settings["scale_ranges_km"],
                        min_pair_lambda_km=70.0,
                        max_pair_lambda_km=20000.0,
                        geometry_plan_cache=cache,
                    )
                    expected = np.minimum(requested, geometry["eligible_per_bin"])
                    np.testing.assert_array_equal(product["counts"][0], expected)
                    if file_hash(entry["source"]) != entry["input_sha256"]:
                        raise ValueError("input changed during sampling; result not saved")
                    save_json(
                        path,
                        {
                            "signature": signature,
                            "seed": seed,
                            "duration_s": float(duration),
                            "budget": budget,
                            "product": product,
                            "geometry": geometry,
                            "coverage": spatial,
                            "seconds": time.perf_counter() - tick,
                            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                            / 1024,
                        },
                    )
                    print(f"Saved {path}", flush=True)
                else:
                    if json.loads(path.read_text())["signature"] != signature:
                        raise ValueError(f"invalid cache signature: {path}")
                    print(f"Reuse {path}", flush=True)
                manifest["records"].append(
                    {
                        "path": str(path.relative_to(output)),
                        "signature": signature,
                        "duration_s": float(duration),
                        "seed": seed,
                        "budget": budget["name"],
                    }
                )
                # A new completed seed is immediately resumable after interruption.
                save_json(output / "manifest.json", manifest)
    return manifest


def derive(manifest, output, windows, interval, durations=None, seeds=None):
    """No HDF5 access; recompute fits and local slopes using saved individual moments."""
    groups = {}
    for record in manifest["records"]:
        if (durations and record["duration_s"] not in durations) or (
            seeds is not None and record["seed"] not in seeds
        ):
            continue
        saved = json.loads((output / record["path"]).read_text())
        if saved["signature"] != record["signature"]:
            raise ValueError("saved moments do not match manifest")
        groups.setdefault((record["budget"], record["duration_s"]), []).append(saved)
    derived = {
        "version": VERSION,
        "validation": manifest["validation"],
        "windows": windows,
        "peak_interval_km": interval,
        "groups": [],
    }
    for (budget, duration), records in sorted(groups.items()):
        records.sort(key=lambda r: r["seed"])
        first = records[0]
        centers = np.asarray(first["product"]["lambda_bin_centers_km"])
        counts = np.array([r["product"]["counts"][0] for r in records])
        item = {
            "budget": budget,
            "duration_s": duration,
            "seeds": [r["seed"] for r in records],
            "centers_km": centers,
            "edges_km": first["product"]["lambda_bin_edges_km"],
            "coverage": first["coverage"],
            "geometry": first["geometry"],
            "counts": counts,
            "sampling": first["product"]["spacetime_pair_counts"]["geometry_sampling"]["members"][
                0
            ],
            "fields": {},
        }
        p = manifest["validation"]["inputs"][str(float(duration))]["p"]
        theory = p_model_exponents(np.array(ORDERS), p)
        item["p"], item["theory"] = p, theory
        for field in FIELDS:
            values = np.array([r["product"][field]["S_n"][0] for r in records])
            fit = _fit_scaling_exponents(values, counts, centers, ORDERS, windows, 3)
            slopes = local_log_slopes(centers, values, counts=counts)[1]
            empirical = np.asarray(first["geometry"]["population_log_centers_km"])
            peaks = local_peaks(centers, slopes, interval)
            item["fields"][field] = {
                "S_n": values,
                "S_n_summary": seed_summary(values),
                "fits": fit,
                "xi_summary": seed_summary(fit["raw"]),
                "local_slopes": slopes,
                "slope_summary": seed_summary(slopes),
                "peaks": peaks,
                "peak_summary": seed_summary(peaks["amplitude"]),
                "minus_theory": fit["raw"] - theory,
                "population_center_fits": _fit_scaling_exponents(
                    values, counts, empirical, ORDERS, windows, 3
                ),
            }
        derived["groups"].append(item)
    if not derived["groups"]:
        raise ValueError("no saved seed moments match the selected durations")
    for budget in sorted({g["budget"] for g in derived["groups"]}):
        sequence = [g for g in derived["groups"] if g["budget"] == budget]
        for index, item in enumerate(sequence):
            item["reference_duration_s"] = sequence[-1]["duration_s"]
            for field in FIELDS:
                result = item["fields"][field]
                for label, ref in [
                    ("longest", sequence[-1]),
                    ("previous", sequence[index - 1] if index else None),
                ]:
                    if ref is None:
                        result[f"delta_{label}"] = None
                        continue
                    common = sorted(set(item["seeds"]) & set(ref["seeds"]))
                    left = [item["seeds"].index(s) for s in common]
                    right = [ref["seeds"].index(s) for s in common]
                    xi = np.asarray(result["fits"]["raw"])[left]
                    xi_ref = np.asarray(ref["fields"][field]["fits"]["raw"])[right]
                    result[f"delta_{label}"] = seed_summary(xi - xi_ref)
                    a = np.asarray(result["S_n"])[left]
                    b = np.asarray(ref["fields"][field]["S_n"])[right]
                    result[f"S_n_relative_{label}"] = seed_summary(a / b - 1)
    for item in derived["groups"]:
        baseline = next(
            (
                g
                for g in derived["groups"]
                if g["budget"] == "baseline" and g["duration_s"] == item["duration_s"]
            ),
            None,
        )
        if item["budget"] == "baseline" or baseline is None:
            continue
        common = sorted(set(item["seeds"]) & set(baseline["seeds"]))
        left = [item["seeds"].index(seed) for seed in common]
        right = [baseline["seeds"].index(seed) for seed in common]
        for field in FIELDS:
            xi = np.asarray(item["fields"][field]["fits"]["raw"])[left]
            reference = np.asarray(baseline["fields"][field]["fits"]["raw"])[right]
            item["fields"][field]["delta_baseline_budget"] = seed_summary(xi - reference)
    return derived


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    parser.add_argument(
        "--mode", choices=["preflight", "sample", "derive", "plot", "run"], default="run"
    )
    parser.add_argument("--dry-run", action="store_true", help="validate inputs; no pair sampling")
    parser.add_argument(
        "--plot-only", action="store_true", help="refit and plot saved moments only"
    )
    parser.add_argument("--durations", nargs="+", type=float)
    parser.add_argument("--generation-glob", help="generation recipes for another p sequence")
    parser.add_argument(
        "--output-dir", type=Path, help="separate study/cache directory; report saved there"
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--seed-count", type=int)
    parser.add_argument("--fit-pairs", type=int)
    parser.add_argument("--outside-pairs", type=int)
    parser.add_argument(
        "--verify-budget", action="store_true", help="also sample selected higher budgets"
    )
    parser.add_argument("--higher-fit-pairs", type=int)
    parser.add_argument("--force", action="store_true", help="recompute existing seed caches")
    parser.add_argument("--primary-fit", nargs=2, type=float)
    parser.add_argument("--sensitivity-fit", nargs=2, type=float)
    args = parser.parse_args(argv)
    root = Path.cwd()
    config = yaml.safe_load(args.config.read_text())
    if config["analysis"]["method"] != "structure_function_duration_convergence":
        parser.error("use a dedicated duration-convergence configuration")
    study = config["duration_study"]
    if args.generation_glob:
        study["generation_glob"] = args.generation_glob
    if args.output_dir:
        config["output"]["directory"] = str(args.output_dir)
        config["output"]["report"] = str(args.output_dir / "report.md")
    durations = sorted(set(args.durations or study["durations_s"]))
    durations = [float(t) for t in durations]
    seeds = args.seeds if args.seeds is not None else study["seeds"]
    if args.seed_count is not None:
        if not 1 <= args.seed_count <= len(seeds):
            parser.error("--seed-count must be between 1 and the available seed-list length")
        seeds = seeds[: args.seed_count]
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        parser.error("seeds must be distinct nonnegative integers")
    if len(seeds) < 5:
        print("WARNING: fewer than five seeds; pilot only, not the default convergence study.")
    budgets = [
        {
            "name": "baseline",
            "durations": None,
            "fit_pairs": args.fit_pairs if args.fit_pairs is not None else study["fit_pairs"],
            "outside_pairs": args.outside_pairs
            if args.outside_pairs is not None
            else study["outside_pairs"],
        }
    ]
    if args.verify_budget:
        verification = study["verification"]
        budgets.append(
            {
                "name": "verification",
                "durations": verification["durations_s"],
                "fit_pairs": args.higher_fit_pairs
                if args.higher_fit_pairs is not None
                else verification["fit_pairs"],
                "outside_pairs": verification["outside_pairs"],
            }
        )
    if any(b["fit_pairs"] < 1 or b["outside_pairs"] < 1 for b in budgets):
        parser.error("pair budgets must be positive")
    settings = _settings(config["structure_function"])
    windows = dict(settings["scale_ranges_km"])
    for label in FIT_WINDOWS:
        override = getattr(args, f"{label}_fit")
        if override:
            if not 0 < override[0] < override[1]:
                parser.error("fit limits must be positive and increasing")
            windows[label] = override
    output = root / config["output"]["directory"]
    mode = "preflight" if args.dry_run else args.mode
    if args.plot_only:
        mode = "derive"
    if mode in ("preflight", "sample", "run"):
        validation = validate_inputs(config, root, durations)
        save_json(output / "validation.json", validation)
        print(f"Nested sequence within tolerance: {validation['valid_nested_sequence']}")
        if not validation["valid_nested_sequence"]:
            raise ValueError("INPUT MISMATCH: see validation.json; not a nested duration sequence")
        if mode == "preflight":
            print(f"No sampling. Seeds: {seeds}; budgets: {budgets}")
            return 0
        manifest = sample(config, root, validation, seeds, budgets, force=args.force)
    elif mode == "derive":
        manifest = json.loads((output / "manifest.json").read_text())
    if mode in ("derive", "run"):
        selected_seeds = seeds if args.seeds is not None or args.seed_count is not None else None
        derived = derive(
            manifest, output, windows, study["peak_interval_km"], args.durations, selected_seeds
        )
        save_json(output / "derived.json", derived)
        from hs_observatory.analysis.structure_function_duration_plots import write_report

        write_report(derived, root / config["output"]["report"])
    if mode in ("plot", "run") or args.plot_only:
        from hs_observatory.analysis.structure_function_duration_plots import plot_duration

        plot_duration(json.loads((output / "derived.json").read_text()), output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
