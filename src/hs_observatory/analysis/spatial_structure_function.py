"""Streaming exact-separation structure functions of the raw periodic Malara field."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import json
from pathlib import Path
import warnings

import h5py
import numpy as np
import yaml

from hs_observatory.analysis.base import AnalysisMethod
from hs_observatory.analysis.registry import register_analysis
from hs_observatory.analysis.structure_function import (
    P_MODEL_RUN_COLORS,
    _ORDER_MARKERS,
    _fit_scaling_exponents,
    _summary_result,
    plot_structure_function_curve,
    structure_function_moment_means,
    structure_function_moment_sums,
)
from hs_observatory.analysis.structure_function_local import (
    CASCADE_CAPTION,
    _mark_scales,
    local_log_slopes,
    p_model_exponents,
    plot_local_diagnostics,
)
from hs_observatory.fields.malara import (
    MalaraTurbulence,
    MalaraTurbulenceField,
    _evaluate_malara_chunk,
)

ORDERS = list(range(1, 7))
DEFAULT_WINDOWS = {"candidate": [200.0, 1000.0], "comparison": [100.0, 1300.0]}
BATCH_NOTE = (
    "Spatial batch variability is a sampling diagnostic, not independent-realization uncertainty."
)


def _positive_int(value, name):
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def spatial_pair_rng(seed: int, separation: float) -> np.random.Generator:
    """An independent per-separation stream, unaffected by p, grid size or chunking."""
    words = np.asarray([separation], dtype="<f8").view("<u4")
    return np.random.default_rng(np.random.SeedSequence([seed, *map(int, words)]))


def sample_spatial_pairs(rng, n_pairs: int, domain, separation: float):
    """Five draws per pair guarantee deterministic prefixes across chunk sizes."""
    draws = rng.random((n_pairs, 5))
    left = draws[:, :3] * np.asarray(domain)
    azimuth = 2 * np.pi * draws[:, 3]
    cosine = 2 * draws[:, 4] - 1
    sine = np.sqrt(np.maximum(0, 1 - cosine**2))
    direction = np.column_stack((sine * np.cos(azimuth), sine * np.sin(azimuth), cosine))
    return left, left + separation * direction


def _moment_task(task):
    config, left, right, backend, use_cache = task
    positions = np.stack((left, right), axis=1)
    values = _evaluate_malara_chunk(
        (config, positions, backend in {"cached", "parallel"} and use_cache)
    )
    return _evaluated_moments(values)


def _evaluated_moments(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 3 or values.shape[1:] != (2, 3):
        raise ValueError("field values must have shape (pair, 2, 3)")
    if not np.all(np.isfinite(values)):
        raise ValueError("raw field returned nonfinite values")
    with np.errstate(over="raise", invalid="raise"):
        try:
            sums = structure_function_moment_sums(
                np.linalg.norm(values[:, 1] - values[:, 0], axis=1), ORDERS
            )
        except FloatingPointError as exc:
            raise ValueError(
                "increment moments overflowed; field normalization is unchanged"
            ) from exc
    if not np.all(np.isfinite(sums)):
        raise ValueError("nonfinite increment moment sums")
    return sums


def _bounded_results(executor, tasks, limit):
    """Preserve reduction order and bound both submitted coordinates and results."""
    pending = deque()
    iterator = iter(tasks)
    for _ in range(limit):
        item = next(iterator, None)
        if item is None:
            break
        descriptor, task = item
        pending.append((descriptor, executor.submit(_moment_task, task)))
    while pending:
        descriptor, future = pending.popleft()
        yield descriptor, future.result()
        item = next(iterator, None)
        if item is not None:
            descriptor, task = item
            pending.append((descriptor, executor.submit(_moment_task, task)))


def _kahan_add(total, correction, value):
    with np.errstate(over="raise", invalid="raise"):
        try:
            increment = value - correction
            updated = total + increment
            correction[:] = (updated - total) - increment
            total[:] = updated
        except FloatingPointError as exc:
            raise ValueError("accumulated increment moments overflowed") from exc


def _batch_sd(values):
    valid = np.isfinite(values)
    finite = valid.sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sd = np.nanstd(values, axis=0, ddof=1)
    return sd, finite


def sample_case(
    config,
    centers,
    *,
    n_pairs,
    n_batches,
    seed,
    chunk_size,
    backend="cached",
    n_workers=1,
    use_cache=True,
    executor=None,
    evaluator=None,
):
    """Stream all scales for one p; evaluator injection supports inexpensive tests."""
    batch_edges = np.linspace(0, n_pairs, n_batches + 1, dtype=int)
    checkpoints = np.unique(np.ceil(n_pairs * np.array([0.25, 0.5, 0.75, 1])).astype(int))
    n_scales = len(centers)
    batch_sums = np.zeros((n_batches, 6, n_scales))
    batch_correction = np.zeros_like(batch_sums)
    total = np.zeros((6, n_scales))
    correction = np.zeros_like(total)
    checkpoint_sums = np.zeros((len(checkpoints), 6, n_scales))

    def tasks():
        for si, separation in enumerate(centers):
            rng = spatial_pair_rng(seed, float(separation))
            boundaries = sorted(set(batch_edges[1:].tolist() + checkpoints.tolist()))
            start = 0
            for boundary in boundaries:
                while start < boundary:
                    stop = min(start + chunk_size, boundary)
                    batch = int(np.searchsorted(batch_edges[1:], start, side="right"))
                    left, right = sample_spatial_pairs(
                        rng, stop - start, config.domain_size, separation
                    )
                    yield (si, batch, stop), (config, left, right, backend, use_cache)
                    start = stop

    if executor is not None:
        results = _bounded_results(executor, tasks(), max(1, 2 * n_workers))
    else:
        model = None if evaluator is not None else MalaraTurbulence(config)

        def serial_results():
            for descriptor, task in tasks():
                _, left, right, _, _ = task
                points = np.stack((left, right), axis=1)
                try:
                    if evaluator is not None:
                        values = evaluator(points)
                    elif backend in {"direct", "chunked"} or not use_cache:
                        values = model.evaluate_vector(points)
                    else:
                        values = model.evaluate_vector_cached(points)
                    yield descriptor, _evaluated_moments(values)
                finally:
                    if model is not None:
                        model.clear_cache()

        results = serial_results()
    for (si, batch, stop), sums in results:
        _kahan_add(total[:, si], correction[:, si], sums)
        _kahan_add(batch_sums[batch, :, si], batch_correction[batch, :, si], sums)
        matches = np.flatnonzero(checkpoints == stop)
        if matches.size:
            checkpoint_sums[matches[0], :, si] = total[:, si]
    counts = np.full((1, n_scales), n_pairs, dtype=np.int64)
    batch_counts = np.broadcast_to(np.diff(batch_edges)[:, None], (n_batches, n_scales)).copy()
    checkpoint_counts = np.broadcast_to(checkpoints[:, None], (len(checkpoints), n_scales)).copy()
    values = structure_function_moment_means(total[None], counts)
    batch_values = structure_function_moment_means(batch_sums, batch_counts)
    sd, _ = _batch_sd(batch_values)
    return {
        "orders": np.asarray(ORDERS),
        "lambda_bin_centers_km": np.asarray(centers),
        "counts": counts,
        "q": {"S_n": values},
        "moment_sums": total[None],
        "batch_counts": batch_counts,
        "batch_S_n": batch_values,
        "batch_moment_sums": batch_sums,
        "batch_sd": sd,
        "batch_cv": np.divide(sd, values[0], out=np.full_like(sd, np.nan), where=values[0] > 0),
        "checkpoint_counts": checkpoint_counts,
        "checkpoint_S_n": structure_function_moment_means(checkpoint_sums, checkpoint_counts),
    }


def validate_windows(windows):
    if not isinstance(windows, dict) or not windows:
        raise ValueError("scale_ranges_km must be a nonempty mapping")
    parsed = {}
    for name, bounds in windows.items():
        values = np.asarray(bounds, dtype=float)
        if values.shape != (2,) or not np.all(np.isfinite(values)) or not 0 < values[0] < values[1]:
            raise ValueError(f"fit window {name!r} requires increasing positive finite bounds")
        parsed[str(name)] = values.tolist()
    return parsed


def fit_with_diagnostics(values, counts, centers, orders, windows, min_bins):
    fit = _fit_scaling_exponents(values, counts, centers, orders, windows, min_bins)
    shape = fit["raw"].shape
    diagnostics = {key: np.full(shape, np.nan) for key in ("intercept", "log_rms", "r_squared")}
    for mi in range(len(values)):
        for wi, name in enumerate(fit["range_names"]):
            lo, hi = windows[name]
            for oi in range(len(orders)):
                slope = fit["raw"][mi, wi, oi]
                if not np.isfinite(slope):
                    continue
                valid = (
                    (centers >= lo)
                    & (centers <= hi)
                    & (counts[mi] > 0)
                    & np.isfinite(values[mi, oi])
                    & (values[mi, oi] > 0)
                )
                x, y = np.log(centers[valid]), np.log(values[mi, oi, valid])
                intercept = y.mean() - slope * x.mean()
                residual = y - (intercept + slope * x)
                variance = np.sum((y - y.mean()) ** 2)
                diagnostics["intercept"][mi, wi, oi] = intercept
                diagnostics["log_rms"][mi, wi, oi] = np.sqrt(np.mean(residual**2))
                if variance > 0:
                    diagnostics["r_squared"][mi, wi, oi] = 1 - np.sum(residual**2) / variance
    fit.update(diagnostics)
    return fit


def derive_spatial_case(case, windows, min_bins=3, method="three-point"):
    centers, orders = case["lambda_bin_centers_km"], case["orders"].tolist()
    values = case["q"]["S_n"]
    case["q"]["scaling_exponents"] = fit_with_diagnostics(
        values, case["counts"], centers, orders, windows, min_bins
    )
    coords, slopes, valid = local_log_slopes(centers, values, counts=case["counts"], method=method)
    _, batch_slopes, _ = local_log_slopes(
        centers, case["batch_S_n"], counts=case["batch_counts"], method=method
    )
    sd, finite = _batch_sd(batch_slopes)
    case["local"] = {
        "coordinates_km": coords,
        "slopes": slopes,
        "source_valid": valid,
        "slope_valid": np.isfinite(slopes),
        "batch_slopes": batch_slopes,
        "batch_sd": sd,
        "finite_batches": finite,
    }
    case["theory"] = p_model_exponents(orders, case["p_value"])
    return case


def _centers(options):
    if options.get("lambda_centers_km") is not None and options.get("centers_from_hs"):
        raise ValueError("choose explicit centers or centers_from_hs, not both")
    if options.get("centers_from_hs"):
        product = _summary_result(options["centers_from_hs"])["products"]["truth"]
        centers = np.asarray(product["lambda_bin_centers_km"], dtype=float)
    elif options.get("lambda_centers_km") is not None:
        centers = np.asarray(options["lambda_centers_km"], dtype=float)
    else:
        lo, hi = float(options.get("lambda_min_km", 50)), float(options.get("lambda_max_km", 20000))
        if not np.isfinite(lo) or not np.isfinite(hi) or not 0 < lo < hi:
            raise ValueError("lambda bounds must be increasing positive finite values")
        centers = np.geomspace(
            lo, hi, _positive_int(options.get("n_separations", 25), "n_separations")
        )
    if (
        centers.ndim != 1
        or len(centers) < 3
        or not np.all(np.isfinite(centers) & (centers > 0))
        or not np.all(np.diff(centers) > 0)
    ):
        raise ValueError("at least three positive increasing finite separations are required")
    return centers


def _cascade_from_config(config, source):
    indices = sorted(
        set(range(config.injection_scale_index, config.n_scales + 1, 3)) | {config.n_scales}
    )
    return {
        "generation_config": source,
        "domain_lengths_km": list(config.domain_size),
        "injection_scale_index": config.injection_scale_index,
        "maximum_scale_index": config.n_scales,
        "spectral_exponent_h": config.spectral_exponent_h,
        "mark_indices": indices,
        "cell_sizes_km": [config.domain_size[0] / 2**m for m in indices],
        "caption": CASCADE_CAPTION,
    }


@register_analysis
class SpatialStructureFunctionAnalysis(AnalysisMethod):
    name = "spatial_structure_function"
    config_section = "spatial_structure_function"

    def run(self, input_path):
        options = dict(self.config)
        source = Path(input_path).expanduser().resolve()
        source_yaml = source.read_text(encoding="utf-8")
        generation = yaml.safe_load(source_yaml)
        field = generation["field"]
        if field["model"] != "malara_turbulence":
            raise ValueError("direct spatial analysis requires malara_turbulence")
        base = MalaraTurbulenceField(field).base_config
        if not np.all(np.asarray(base.domain_size) == base.domain_size[0]):
            raise ValueError("initial direct spatial analysis requires an isotropic cubic domain")
        centers = _centers(options)
        explicit_grid = options.get("lambda_centers_km") is not None or options.get(
            "centers_from_hs"
        )
        if centers[-1] >= min(base.domain_size) / 2:
            if not explicit_grid:
                raise ValueError(
                    "default/logarithmic separation grid must stay below half the domain; use explicit centers to override"
                )
            warnings.warn(
                "explicit separations reach half the domain: periodic-image ambiguity", stacklevel=2
            )
        p_values = list(map(float, options.get("p_values", [0.5, 0.7, 0.9])))
        if not p_values or len(set(p_values)) != len(p_values):
            raise ValueError("p_values must be nonempty and unique")
        for p in p_values:
            p_model_exponents(ORDERS, p)
        n_pairs = _positive_int(options.get("n_pairs", 2000), "n_pairs")
        n_batches = _positive_int(options.get("n_batches", 8), "n_batches")
        if n_batches > n_pairs:
            raise ValueError("n_batches cannot exceed n_pairs")
        seed = options.get("seed", 12345)
        if isinstance(seed, bool) or int(seed) != seed or seed < 0:
            raise ValueError("sampling seed must be a nonnegative integer")
        seed = int(seed)
        chunk_size = _positive_int(options.get("chunk_size", 128), "chunk_size")
        backend = options.get("backend", field["turbulence"].get("backend", "cached"))
        if backend not in {"cached", "parallel", "direct", "chunked"}:
            raise ValueError("backend must be cached, parallel, direct or chunked")
        worker_setting = options.get("n_workers", field["turbulence"].get("n_workers", 1))
        workers = _positive_int(1 if worker_setting is None else worker_setting, "n_workers")
        use_cache = bool(options.get("use_cache", field["turbulence"].get("use_cache", True)))
        windows = validate_windows(options.get("scale_ranges_km", DEFAULT_WINDOWS))
        minimum = _positive_int(options.get("min_fit_bins", 3), "min_fit_bins")
        if minimum < 2:
            raise ValueError("min_fit_bins must be at least two")
        method = options.get("local_slope_method", "three-point")
        if method not in {"three-point", "adjacent"}:
            raise ValueError("invalid local_slope_method")
        selected = options.get("fit_range", next(iter(windows)))
        if selected not in windows:
            raise ValueError("fit_range must name a configured window")
        options.update(
            scale_ranges_km=windows,
            min_fit_bins=minimum,
            local_slope_method=method,
            fit_range=selected,
            n_pairs=n_pairs,
            n_batches=n_batches,
            seed=seed,
            chunk_size=chunk_size,
            backend=backend,
            n_workers=workers,
            use_cache=use_cache,
        )
        cases = []
        pool = ProcessPoolExecutor(max_workers=workers) if backend == "parallel" else None
        try:
            for p in sorted(p_values, reverse=True):
                effective = replace(base, intermittency_p=p)
                print(
                    f"Direct spatial p={p:g}: {len(centers)} separations × {n_pairs} pairs",
                    flush=True,
                )
                case = sample_case(
                    effective,
                    centers,
                    n_pairs=n_pairs,
                    n_batches=n_batches,
                    seed=seed,
                    chunk_size=chunk_size,
                    backend=backend,
                    n_workers=workers,
                    use_cache=use_cache,
                    executor=pool,
                )
                case.update(
                    p_value=p,
                    field_config=asdict(effective),
                    cascade=_cascade_from_config(effective, str(source)),
                )
                cases.append(derive_spatial_case(case, windows, minimum, method))
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
        return {
            "method": self.name,
            "metadata": {
                "field_model": "malara_turbulence",
                "source_field_config": str(source),
                "source_field_yaml": source_yaml,
                "sampling_seed": seed,
                "coordinate_convention": "uniform domain-relative x; unwrapped x+lambda*r; evaluator wraps",
                "separation_mode": "exact",
                "quantity": "q",
                "units": "raw evaluator units; no velocity/B/Elsasser conversion",
                "batch_note": BATCH_NOTE,
                "smoke": bool(options.get("smoke", False)),
                "sampling_rng": "PCG64",
                "sample_stream_version": 1,
                "numpy_version": np.__version__,
            },
            "config": options,
            "cases": cases,
        }

    def write_result(self, result, output_path):
        output = Path(output_path).expanduser().resolve()
        protected = [Path(result["metadata"]["source_field_config"])]
        if result["metadata"].get("source_spatial_product"):
            protected.append(Path(result["metadata"]["source_spatial_product"]))
            if result["metadata"].get("sampling_product"):
                protected.append(Path(result["metadata"]["sampling_product"]))
        else:
            result["metadata"]["sampling_product"] = str(output)
        if output in protected:
            raise ValueError("output cannot overwrite source data")
        if output.exists():
            # Allow regeneration of our own products, but protect observations and other analyses.
            try:
                read_spatial_result(output)
            except (OSError, ValueError, KeyError) as exc:
                raise ValueError(
                    "output would overwrite a file that is not a direct spatial product"
                ) from exc
        result["comparisons"] = {
            str(p): product for p, product in _comparison_products(result).items()
        }
        write_spatial_result(result, output)
        if result["config"].get("plot", {}).get("enabled", True):
            for path in plot_spatial_result(result, output.with_suffix("")):
                print(path)
        return output


def _write_tree(group, mapping):
    for key, value in mapping.items():
        if isinstance(value, dict):
            _write_tree(group.create_group(key), value)
        elif isinstance(value, (str, bool, int, float)):
            group.attrs[key] = value
        elif key in {"range_names", "notes"}:
            group.create_dataset(key, data=np.asarray(value, dtype=h5py.string_dtype()))
        else:
            array = np.asarray(value)
            group.create_dataset(
                key, data=array, compression="gzip" if array.ndim and array.size else None
            )


def _read_tree(group):
    result = dict(group.attrs)
    for key, value in group.items():
        if isinstance(value, h5py.Group):
            result[key] = _read_tree(value)
        elif key in {"range_names", "notes"}:
            result[key] = value.asstr()[:].tolist()
        else:
            result[key] = value[()]
    return result


def write_spatial_result(result, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        handle.attrs["schema_version"] = 1
        handle.attrs["method"] = "spatial_structure_function"
        handle.attrs["metadata_json"] = json.dumps(result["metadata"])
        handle.attrs["config_json"] = json.dumps(result["config"])
        for i, case in enumerate(result["cases"]):
            _write_tree(handle.create_group(f"cases/{i}"), case)
        _write_tree(handle.create_group("comparisons"), result.get("comparisons", {}))


def read_spatial_result(path):
    with h5py.File(path) as handle:
        if (
            handle.attrs.get("method") != "spatial_structure_function"
            or handle.attrs.get("schema_version") != 1
        ):
            raise ValueError("unsupported direct spatial structure-function product")
        return {
            "method": "spatial_structure_function",
            "metadata": json.loads(handle.attrs["metadata_json"]),
            "config": json.loads(handle.attrs["config_json"]),
            "comparisons": _read_tree(handle["comparisons"]),
            "cases": [_read_tree(handle["cases"][key]) for key in sorted(handle["cases"], key=int)],
        }


def spatial_plot_product(case):
    """Adapt exact raw-q statistics to the shared product convention without MC aliases."""
    return {
        "orders": case["orders"].tolist(),
        "lambda_bin_centers_km": case["lambda_bin_centers_km"],
        "counts": case["counts"],
        "q": case["q"],
        "separation_mode": "exact",
    }


def _comparison_products(result):
    path = result["config"].get("comparison_config")
    if not path:
        return {}
    path = Path(path).expanduser()
    if not path.exists():
        warnings.warn(f"HS comparison config unavailable: {path}", stacklevel=2)
        return {}
    config = yaml.safe_load(path.read_text())
    if (
        not isinstance(config, dict)
        or not isinstance(config.get("cases"), dict)
        or not config["cases"]
    ):
        raise ValueError("HS comparison config requires nonempty cases")
    products = {}
    for entry in config["cases"].values():
        p = float(entry["p_value"])
        if p in products:
            raise ValueError("duplicate HS comparison p_value")
        source = Path(entry["path"]).expanduser().resolve()
        if not source.exists():
            warnings.warn(f"HS comparison unavailable: {source}", stacklevel=2)
            continue
        quantity = config.get("quantity", "z_plus")
        product = _summary_result(source)["products"]["truth"]
        x = np.asarray(product["lambda_bin_centers_km"], dtype=float)
        y = np.asarray(product[quantity]["S_n"], dtype=float)
        counts = np.asarray(product["counts"])
        orders = list(product["orders"])
        if (
            x.ndim != 1
            or not np.all(np.isfinite(x) & (x > 0))
            or not np.all(np.diff(x) > 0)
            or y.shape != (1, len(orders), len(x))
            or counts.shape != (1, len(x))
        ):
            raise ValueError(f"malformed HS comparison product: {source}")
        fit = fit_with_diagnostics(
            y,
            counts,
            x,
            orders,
            result["config"]["scale_ranges_km"],
            result["config"]["min_fit_bins"],
        )
        products[p] = {
            "path": str(source),
            "quantity": quantity,
            "orders": np.asarray(orders),
            "fit": fit,
            "centers_km": x,
        }
    return products


def _save_figure(figure, prefix, dpi):
    import matplotlib.pyplot as plt

    outputs = []
    try:
        for suffix in (".png", ".pdf"):
            path = Path(f"{prefix}{suffix}")
            figure.savefig(path, dpi=dpi, bbox_inches="tight")
            outputs.append(path)
    finally:
        plt.close(figure)
    return outputs


def plot_spatial_result(result, prefix):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    options = result["config"]
    plot = options.get("plot", {})
    dpi = int(plot.get("dpi", 150))
    show_scales = bool(plot.get("show_cascade_scales", False))
    cases = result["cases"]
    windows = options["scale_ranges_km"]
    selected = options["fit_range"]
    smoke = "SMOKE TEST — not a scientific result. " if result["metadata"].get("smoke") else ""
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.axvspan(*windows[selected], color="0.6", alpha=0.15)
    for case in cases:
        product = spatial_plot_product(case)
        x = product["lambda_bin_centers_km"]
        color = P_MODEL_RUN_COLORS.get(case["p_value"], "0.3")
        for oi, order in enumerate(case["orders"]):
            plot_structure_function_curve(
                axis,
                x,
                product["q"]["S_n"][0, oi],
                color=color,
                marker=_ORDER_MARKERS[oi % len(_ORDER_MARKERS)],
            )
    if show_scales:
        x = cases[0]["lambda_bin_centers_km"]
        _mark_scales(axis, cases[0], (float(x.min()), float(x.max())))
        figure.text(0.5, 0.01, CASCADE_CAPTION, ha="center", fontsize=8)
    axis.set(
        xscale="log",
        yscale="log",
        xlabel=r"$\lambda$ [km] (exact separations)",
        ylabel=r"$S_n^{3D}(\lambda)$ [raw field units$^n$]",
        title=smoke + "Direct 3D raw Malara structure functions",
    )
    axis.grid(alpha=0.15)
    handles = [
        Line2D(
            [], [], color=P_MODEL_RUN_COLORS.get(c["p_value"], "0.3"), label=f"p = {c['p_value']:g}"
        )
        for c in cases
    ]
    handles += [
        Line2D([], [], color="0.3", linestyle="none", marker=_ORDER_MARKERS[i], label=f"n = {n}")
        for i, n in enumerate(ORDERS)
    ]
    axis.legend(handles=handles, ncols=3, fontsize=8)
    figure.tight_layout(rect=(0, 0.04 if show_scales else 0, 1, 1))
    outputs = _save_figure(figure, f"{prefix}_structure_functions", dpi)
    local_cases = []
    for case in cases:
        data = case["local"]
        local_case = {
            "orders": case["orders"],
            "p_value": case["p_value"],
            "theory": case["theory"],
            "truth": {**data, "source_centers_km": case["lambda_bin_centers_km"]},
            "sampling_sd": np.where(data["finite_batches"] >= 3, data["batch_sd"], np.nan),
        }
        if show_scales:
            local_case["cascade"] = case["cascade"]
        local_cases.append(local_case)
    outputs += plot_local_diagnostics(
        local_cases,
        Path(f"{prefix}_local_slopes"),
        dpi=dpi,
        show_theory=True,
        curve_label="direct 3D full sample",
        focused=False,
        caption_extra=smoke + BATCH_NOTE,
    )
    comparisons = {float(p): product for p, product in result.get("comparisons", {}).items()}
    matching = [case for case in cases if case["p_value"] in comparisons]
    if matching:
        figure, axes = plt.subplots(1, len(windows), figsize=(6 * len(windows), 5), squeeze=False)
        for wi, (name, bounds) in enumerate(windows.items()):
            axis = axes[0, wi]
            notes = []
            for case in matching:
                orders = case["orders"]
                color = P_MODEL_RUN_COLORS.get(case["p_value"], "0.3")
                axis.plot(orders, case["theory"], color=color, linestyle="--")
                axis.plot(
                    orders, case["q"]["scaling_exponents"]["raw"][0, wi], color=color, marker="o"
                )
                hs = comparisons[case["p_value"]]
                hs_slopes = hs["fit"]["raw"][0, wi]
                axis.plot(hs["orders"], hs_slopes, color=color, marker="s", linestyle=":")
                if not np.all(np.isfinite(hs_slopes)):
                    bins = hs["fit"]["bins_used"][0, wi]
                    notes.append(
                        f"HS p={case['p_value']:g}: {int(bins.min())} points; fit unavailable"
                    )
            axis.set(
                xlabel="order n", ylabel=r"$\xi_n$", title=f"{name}: {bounds[0]:g}–{bounds[1]:g} km"
            )
            axis.grid(alpha=0.15)
            if notes:
                axis.text(
                    0.03,
                    0.97,
                    "\n".join(notes),
                    transform=axis.transAxes,
                    fontsize=8,
                    va="top",
                    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
                )
        style = [
            Line2D([], [], color="0.3", linestyle="--", label="p-model"),
            Line2D([], [], color="0.3", marker="o", label="direct raw q"),
            Line2D([], [], color="0.3", marker="s", linestyle=":", label="ideal HS z (processed)"),
        ]
        figure.legend(
            handles=handles[: len(cases)] + style, loc="upper center", ncols=3, fontsize=8
        )
        figure.text(
            0.5,
            0.01,
            smoke
            + "Raw q and processed HS Elsasser fields differ; this comparison does not isolate sampling alone.",
            ha="center",
            fontsize=8,
        )
        figure.tight_layout(rect=(0, 0.06, 1, 0.88))
        outputs += _save_figure(figure, f"{prefix}_comparison", dpi)
    return outputs


def postprocess_spatial_result(
    source,
    *,
    output_prefix=None,
    windows=None,
    method=None,
    show_scales=None,
    comparison_config=None,
):
    source = Path(source).expanduser().resolve()
    result = read_spatial_result(source)
    options = result["config"]
    if windows is not None:
        options["scale_ranges_km"] = validate_windows(windows)
        if options["fit_range"] not in options["scale_ranges_km"]:
            options["fit_range"] = next(iter(options["scale_ranges_km"]))
    if method is not None:
        options["local_slope_method"] = method
    if comparison_config is not None:
        options["comparison_config"] = comparison_config
    if show_scales is not None:
        options.setdefault("plot", {})["show_cascade_scales"] = show_scales
    options.setdefault("plot", {})["enabled"] = True
    for case in result["cases"]:
        derive_spatial_case(
            case, options["scale_ranges_km"], options["min_fit_bins"], options["local_slope_method"]
        )
    result["metadata"].setdefault(
        "sampling_product", result["metadata"].get("source_spatial_product", str(source))
    )
    result["metadata"]["source_spatial_product"] = str(source)
    prefix = (
        Path(output_prefix).expanduser().resolve()
        if output_prefix
        else source.with_name(source.stem + "_postprocess").with_suffix("")
    )
    target = Path(f"{prefix}.h5")
    if target == source:
        raise ValueError("cached post-processing must use a separate output prefix")
    return SpatialStructureFunctionAnalysis(options).write_result(result, target)
