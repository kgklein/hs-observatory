"""Pairwise Elsasser vector-increment structure functions."""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from hs_observatory.analysis.base import AnalysisMethod
from hs_observatory.analysis.derived.pairs import (
    pair_chunks,
    possible_pair_count,
    selected_pair_count,
)
from hs_observatory.analysis.elsasser import (
    _interpolate,
    _time_axis,
    compute_elsasser_analysis,
)
from hs_observatory.analysis.registry import register_analysis
from hs_observatory.io.hdf5 import read_observations

FIELDS = ("z_plus", "z_minus")
POSITION_VARIABLES = ("v", "n", "B")


def _mapping(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"structure_function.{name} must be a mapping")
    return value


def _settings(config: dict[str, Any]) -> dict[str, Any]:
    min_order = int(config.get("min_order", 1))
    max_order = int(config.get("max_order", 6))
    if min_order < 1 or max_order < min_order:
        raise ValueError("structure_function orders must satisfy 1 <= min_order <= max_order")
    n_lambda_bins = int(config.get("n_lambda_bins", 20))
    if n_lambda_bins < 1:
        raise ValueError("structure_function.n_lambda_bins must be positive")
    lambda_bins_km = config.get("lambda_bins_km")
    if lambda_bins_km is not None:
        lambda_bins_km = [float(value) for value in lambda_bins_km]
        if len(lambda_bins_km) < 2 or any(
            right <= left for left, right in zip(lambda_bins_km, lambda_bins_km[1:])
        ):
            raise ValueError("structure_function.lambda_bins_km must be increasing bin edges")
    scale_ranges_km = config.get("scale_ranges_km")
    if scale_ranges_km is None:
        scale_ranges_km = {}
    if not isinstance(scale_ranges_km, dict):
        raise ValueError("structure_function.scale_ranges_km must be a mapping")
    parsed_ranges = {}
    for name, bounds in scale_ranges_km.items():
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError("Each scale range must be [min_km, max_km]")
        lower, upper = (float(bounds[0]), float(bounds[1]))
        if upper <= lower:
            raise ValueError("Scale range upper bound must exceed lower bound")
        parsed_ranges[str(name)] = [lower, upper]

    elsasser = _mapping(config, "elsasser")
    density = dict(elsasser.get("density_normalization", {}))
    mean = dict(elsasser.get("mean_subtraction", {}))
    timebase = dict(elsasser.get("timebase", {}))
    velocity = dict(elsasser.get("velocity", {}))
    plot = _mapping(config, "plot")
    fit = _mapping(config, "fit")
    min_fit_bins = int(config.get("min_fit_bins", fit.get("min_bins", 3)))
    if min_fit_bins < 2:
        raise ValueError("structure_function.min_fit_bins must be at least 2")
    max_pair_lambda_km = config.get("max_pair_lambda_km", config.get("max_lambda_km"))
    if max_pair_lambda_km is not None:
        max_pair_lambda_km = float(max_pair_lambda_km)
        if max_pair_lambda_km <= 0.0:
            raise ValueError("structure_function.max_pair_lambda_km must be positive or null")
    increment_mode = str(config.get("increment_mode", "same_time_pairs"))
    if increment_mode not in {"same_time_pairs", "all_spacetime_pairs"}:
        raise ValueError(
            "structure_function.increment_mode must be "
            "'same_time_pairs' or 'all_spacetime_pairs'"
        )
    pair_sampling = _mapping(config, "pair_sampling")
    pair_mode = str(pair_sampling.get("mode", "all"))
    if pair_mode not in {"all", "random"}:
        raise ValueError("structure_function.pair_sampling.mode must be 'all' or 'random'")
    chunk_size = int(pair_sampling.get("chunk_size", 1_000_000))
    if chunk_size < 1:
        raise ValueError("structure_function.pair_sampling.chunk_size must be positive")
    n_pairs = pair_sampling.get("n_pairs")
    n_pairs = None if n_pairs is None else int(n_pairs)
    max_pairs = pair_sampling.get("max_pairs")
    max_pairs = None if max_pairs is None else int(max_pairs)
    random_seed = pair_sampling.get("random_seed", 12345)
    random_seed = None if random_seed is None else int(random_seed)
    if pair_mode == "random" and (n_pairs is None or n_pairs < 1):
        raise ValueError("random structure-function pair sampling requires pair_sampling.n_pairs")
    if pair_mode == "random" and random_seed is None:
        raise ValueError("random structure-function pair sampling requires pair_sampling.random_seed")
    if max_pairs is not None and max_pairs < 1:
        raise ValueError("structure_function.pair_sampling.max_pairs must be positive or null")
    return {
        "orders": list(range(min_order, max_order + 1)),
        "n_lambda_bins": n_lambda_bins,
        "lambda_bins_km": lambda_bins_km,
        "scale_ranges_km": parsed_ranges,
        "min_fit_bins": min_fit_bins,
        "max_pair_lambda_km": max_pair_lambda_km,
        "increment_mode": increment_mode,
        "pair_sampling": {
            "mode": pair_mode,
            "chunk_size": chunk_size,
            "n_pairs": n_pairs,
            "max_pairs": max_pairs,
            "random_seed": random_seed,
        },
        "elsasser": {
            "velocity": velocity,
            "density_normalization": density,
            "mean_subtraction": mean,
            "timebase": timebase,
            "plot": {"enabled": False},
        },
        "plot": {
            "enabled": bool(plot.get("enabled", False)),
            "path": plot.get("path", "structure_function_diagnostic.png"),
            "field": str(plot.get("field", "z_plus")),
            "member": int(plot.get("member", 0)),
            "data_group": str(plot.get("data_group", "observed")),
            "normalized_exponents": bool(plot.get("normalized_exponents", False)),
            "dpi": int(plot.get("dpi", 150)),
        },
    }


def _to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    return value


def _bin_indices(values: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.searchsorted(edges, values, side="right") - 1
    indices[values == edges[-1]] = len(edges) - 2
    valid = np.isfinite(values) & (indices >= 0) & (indices < len(edges) - 1)
    return indices, valid


def _lambda_edges_km(separations_km: np.ndarray, settings: dict[str, Any]) -> np.ndarray:
    configured = settings["lambda_bins_km"]
    if configured is not None:
        return np.asarray(configured, dtype=float)
    positive = separations_km[np.isfinite(separations_km) & (separations_km > 0.0)]
    max_pair_lambda_km = settings["max_pair_lambda_km"]
    if max_pair_lambda_km is not None:
        positive = positive[positive <= max_pair_lambda_km]
    if positive.size == 0:
        raise ValueError("No positive pair separations are available for structure functions")
    lower = float(np.min(positive))
    upper = float(np.max(positive))
    if upper == lower:
        lower *= 0.9
        upper *= 1.1
    return np.geomspace(lower, upper, settings["n_lambda_bins"] + 1)


def _lambda_edges_from_bounds_km(
    lower: float,
    upper: float,
    settings: dict[str, Any],
) -> np.ndarray:
    configured = settings["lambda_bins_km"]
    if configured is not None:
        return np.asarray(configured, dtype=float)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0.0:
        raise ValueError("No positive pair separations are available for structure functions")
    max_pair_lambda_km = settings["max_pair_lambda_km"]
    if max_pair_lambda_km is not None:
        upper = min(upper, max_pair_lambda_km)
        if upper < lower:
            raise ValueError("No positive pair separations remain below max_pair_lambda_km")
    if upper == lower:
        lower *= 0.9
        upper *= 1.1
    return np.geomspace(lower, upper, settings["n_lambda_bins"] + 1)


def _scale_ranges_km(
    centers_km: np.ndarray,
    counts: np.ndarray,
    settings: dict[str, Any],
) -> dict[str, list[float]]:
    configured = settings["scale_ranges_km"]
    if configured:
        return {name: [bounds[0], bounds[1]] for name, bounds in configured.items()}
    finite = np.isfinite(centers_km) & (np.sum(counts, axis=0) > 0)
    if not np.any(finite):
        return {"total": [float(np.nanmin(centers_km)), float(np.nanmax(centers_km))]}
    return {
        "total": [
            float(np.min(centers_km[finite])),
            float(np.max(centers_km[finite])),
        ]
    }


def _flatness(S_n: np.ndarray, orders: list[int]) -> np.ndarray | None:
    if 2 not in orders or 4 not in orders:
        return None
    second = S_n[:, orders.index(2), :]
    fourth = S_n[:, orders.index(4), :]
    return np.divide(
        fourth,
        second**2,
        out=np.full(second.shape, np.nan),
        where=np.isfinite(second) & (second != 0.0) & np.isfinite(fourth),
    )


def _fit_scaling_exponents(
    S_n: np.ndarray,
    counts: np.ndarray,
    centers_km: np.ndarray,
    orders: list[int],
    scale_ranges_km: dict[str, list[float]],
    min_fit_bins: int,
) -> dict[str, Any]:
    n_members, n_orders, _n_bins = S_n.shape
    range_names = list(scale_ranges_km)
    raw = np.full((n_members, len(range_names), n_orders), np.nan)
    bins_used = np.zeros((n_members, len(range_names), n_orders), dtype=int)
    notes: list[str] = []
    log_lambda = np.log(centers_km)
    for range_index, name in enumerate(range_names):
        lower, upper = scale_ranges_km[name]
        in_range = (centers_km >= lower) & (centers_km <= upper) & (counts > 0)
        for member in range(n_members):
            for order_index in range(n_orders):
                valid = (
                    in_range[member]
                    & np.isfinite(log_lambda)
                    & np.isfinite(S_n[member, order_index])
                    & (S_n[member, order_index] > 0.0)
                )
                bins_used[member, range_index, order_index] = int(np.sum(valid))
                if np.sum(valid) < min_fit_bins:
                    notes.append(
                        f"{name}: member {member}, order {orders[order_index]} used "
                        f"{int(np.sum(valid))} fit bins; need {min_fit_bins}"
                    )
                    continue
                slope, _intercept = np.polyfit(
                    log_lambda[valid], np.log(S_n[member, order_index, valid]), 1
                )
                raw[member, range_index, order_index] = slope
    normalized = np.full_like(raw, np.nan)
    if 2 in orders:
        order2 = raw[:, :, orders.index(2)]
        normalized = np.divide(
            raw,
            order2[:, :, None],
            out=np.full_like(raw, np.nan),
            where=np.isfinite(order2[:, :, None]) & (order2[:, :, None] != 0.0),
        )
    return {
        "raw": raw,
        "normalized_by_order2": normalized,
        "range_names": range_names,
        "scale_ranges_km": scale_ranges_km,
        "bins_used": bins_used,
        "notes": notes,
    }


def compute_pairwise_structure_functions(
    positions_km: np.ndarray,
    z_plus: np.ndarray,
    z_minus: np.ndarray,
    orders: list[int],
    lambda_edges_km: np.ndarray,
    *,
    scale_ranges_km: dict[str, list[float]] | None = None,
    min_fit_bins: int = 3,
    max_pair_lambda_km: float | None = None,
) -> dict[str, Any]:
    """Compute same-time pairwise vector-increment structure functions.

    Arrays are member-first and time-first: ``(member, time, spacecraft, 3)``.
    Each member is fitted independently; no ensemble average is performed.
    """

    arrays = [np.asarray(value, dtype=float) for value in (positions_km, z_plus, z_minus)]
    if any(value.shape != arrays[0].shape for value in arrays):
        raise ValueError("positions, z_plus, and z_minus must have matching shapes")
    if arrays[0].ndim != 4 or arrays[0].shape[-1] != 3:
        raise ValueError("Inputs must have shape (member, time, spacecraft, 3)")
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("Structure-function inputs must contain only finite values")
    edges = np.asarray(lambda_edges_km, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("lambda_edges_km must be increasing one-dimensional bin edges")

    n_members, _n_time, n_spacecraft, _ = arrays[0].shape
    pairs = list(combinations(range(n_spacecraft), 2))
    n_bins = edges.size - 1
    centers = np.sqrt(edges[:-1] * edges[1:])
    counts = np.zeros((n_members, n_bins), dtype=int)
    sums = {field: np.zeros((n_members, len(orders), n_bins)) for field in FIELDS}
    pair_separations = np.full((n_members, len(pairs), arrays[0].shape[1]), np.nan)

    for member in range(n_members):
        for pair_index, (left, right) in enumerate(pairs):
            separation = np.linalg.norm(
                arrays[0][member, :, left] - arrays[0][member, :, right],
                axis=1,
            )
            pair_separations[member, pair_index] = separation
            bin_index, valid = _bin_indices(separation, edges)
            valid &= separation > 0.0
            if max_pair_lambda_km is not None:
                valid &= separation <= max_pair_lambda_km
            for index in range(n_bins):
                selected = valid & (bin_index == index)
                count = int(np.sum(selected))
                if count == 0:
                    continue
                counts[member, index] += count
                for field, values in (("z_plus", arrays[1]), ("z_minus", arrays[2])):
                    increment = values[member, selected, left] - values[member, selected, right]
                    magnitude = np.linalg.norm(increment, axis=1)
                    for order_index, order in enumerate(orders):
                        sums[field][member, order_index, index] += float(np.sum(magnitude**order))

    products: dict[str, Any] = {
        "orders": orders,
        "lambda_bin_edges_km": edges,
        "lambda_bin_centers_km": centers,
        "spacecraft_pairs": pairs,
        "pair_separations_km": pair_separations,
        "counts": counts,
    }
    if scale_ranges_km is None:
        finite = np.isfinite(centers) & (np.sum(counts, axis=0) > 0)
        scale_ranges_km = (
            {"total": [float(np.min(centers[finite])), float(np.max(centers[finite]))]}
            if np.any(finite)
            else {"total": [float(edges[0]), float(edges[-1])]}
        )
    products["scale_ranges_km"] = scale_ranges_km
    for field in FIELDS:
        S_n = np.divide(
            sums[field],
            counts[:, None, :],
            out=np.full(sums[field].shape, np.nan),
            where=counts[:, None, :] > 0,
        )
        products[field] = {
            "S_n": S_n,
            "flatness": _flatness(S_n, orders),
            "scaling_exponents": _fit_scaling_exponents(
                S_n, counts, centers, orders, scale_ranges_km, min_fit_bins
            ),
        }
    return products


def _sampling_kwargs(pair_sampling: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": pair_sampling["mode"],
        "n_pairs": pair_sampling["n_pairs"],
        "random_seed": pair_sampling["random_seed"],
        "max_pairs": pair_sampling["max_pairs"],
    }


def _selection_count_kwargs(pair_sampling: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": pair_sampling["mode"],
        "n_pairs": pair_sampling["n_pairs"],
        "max_pairs": pair_sampling["max_pairs"],
    }


def _spacetime_lambda_bounds_km(
    positions_km: np.ndarray,
    pair_sampling: dict[str, Any],
    max_pair_lambda_km: float | None,
) -> tuple[float, float]:
    lower = np.inf
    upper = -np.inf
    flat_position = np.asarray(positions_km, dtype=float).reshape(-1, 3)
    for left, right in pair_chunks(
        flat_position.shape[0],
        pair_sampling["chunk_size"],
        **_sampling_kwargs(pair_sampling),
    ):
        separation = np.linalg.norm(flat_position[right] - flat_position[left], axis=1)
        positive = separation[np.isfinite(separation) & (separation > 0.0)]
        if max_pair_lambda_km is not None:
            positive = positive[positive <= max_pair_lambda_km]
        if positive.size:
            lower = min(lower, float(np.min(positive)))
            upper = max(upper, float(np.max(positive)))
    return lower, upper


def compute_all_spacetime_structure_functions(
    positions_km: np.ndarray,
    z_plus: np.ndarray,
    z_minus: np.ndarray,
    orders: list[int],
    lambda_edges_km: np.ndarray,
    *,
    pair_sampling: dict[str, Any],
    scale_ranges_km: dict[str, list[float]] | None = None,
    min_fit_bins: int = 3,
    max_pair_lambda_km: float | None = None,
) -> dict[str, Any]:
    """Compute all-spacetime vector-increment structure functions.

    Arrays are member-first and time-first: ``(member, time, spacecraft, 3)``.
    Each member is flattened in time-major order and streamed through unique
    unordered point pairs.  Lambda is the actual point-to-point position
    separation for the selected member and data group.
    """

    arrays = [np.asarray(value, dtype=float) for value in (positions_km, z_plus, z_minus)]
    if any(value.shape != arrays[0].shape for value in arrays):
        raise ValueError("positions, z_plus, and z_minus must have matching shapes")
    if arrays[0].ndim != 4 or arrays[0].shape[-1] != 3:
        raise ValueError("Inputs must have shape (member, time, spacecraft, 3)")
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("Structure-function inputs must contain only finite values")
    edges = np.asarray(lambda_edges_km, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("lambda_edges_km must be increasing one-dimensional bin edges")

    n_members, n_time, n_spacecraft, _ = arrays[0].shape
    n_points = n_time * n_spacecraft
    if n_points < 2:
        raise ValueError("At least two spacetime points are required")
    n_bins = edges.size - 1
    centers = np.sqrt(edges[:-1] * edges[1:])
    counts = np.zeros((n_members, n_bins), dtype=np.int64)
    sums = {field: np.zeros((n_members, len(orders), n_bins)) for field in FIELDS}
    possible_pairs = possible_pair_count(n_points)
    selected_pairs = selected_pair_count(n_points, **_selection_count_kwargs(pair_sampling))
    pair_count_by_member = np.full(n_members, selected_pairs, dtype=np.int64)
    skipped_nonpositive = np.zeros(n_members, dtype=np.int64)
    skipped_above_max_lambda = np.zeros(n_members, dtype=np.int64)
    outside_lambda_bins = np.zeros(n_members, dtype=np.int64)

    for member in range(n_members):
        flat_position = arrays[0][member].reshape(-1, 3)
        flat_values = {
            "z_plus": arrays[1][member].reshape(-1, 3),
            "z_minus": arrays[2][member].reshape(-1, 3),
        }
        for left, right in pair_chunks(
            n_points,
            pair_sampling["chunk_size"],
            **_sampling_kwargs(pair_sampling),
        ):
            separation = np.linalg.norm(flat_position[right] - flat_position[left], axis=1)
            bin_index, valid = _bin_indices(separation, edges)
            nonpositive = np.isfinite(separation) & (separation <= 0.0)
            skipped_nonpositive[member] += int(np.sum(nonpositive))
            valid &= separation > 0.0
            above_max = np.zeros(separation.shape, dtype=bool)
            if max_pair_lambda_km is not None:
                above_max = np.isfinite(separation) & (separation > max_pair_lambda_km)
                skipped_above_max_lambda[member] += int(np.sum(above_max))
                valid &= ~above_max
            outside_lambda_bins[member] += int(
                np.sum(np.isfinite(separation) & (separation > 0.0) & ~above_max & ~valid)
            )
            for index in range(n_bins):
                selected = valid & (bin_index == index)
                count = int(np.sum(selected))
                if count == 0:
                    continue
                counts[member, index] += count
                for field, values in flat_values.items():
                    magnitude = np.linalg.norm(values[right[selected]] - values[left[selected]], axis=1)
                    for order_index, order in enumerate(orders):
                        sums[field][member, order_index, index] += float(np.sum(magnitude**order))

    products: dict[str, Any] = {
        "orders": orders,
        "lambda_bin_edges_km": edges,
        "lambda_bin_centers_km": centers,
        "spacecraft_pairs": [],
        "pair_separations_km": None,
        "counts": counts,
        "spacetime_pair_counts": {
            "possible_per_member": possible_pairs,
            "selected_per_member": pair_count_by_member,
            "skipped_nonpositive_separation": skipped_nonpositive,
            "skipped_above_max_lambda": skipped_above_max_lambda,
            "outside_lambda_bins": outside_lambda_bins,
        },
    }
    if scale_ranges_km is None:
        finite = np.isfinite(centers) & (np.sum(counts, axis=0) > 0)
        scale_ranges_km = (
            {"total": [float(np.min(centers[finite])), float(np.max(centers[finite]))]}
            if np.any(finite)
            else {"total": [float(edges[0]), float(edges[-1])]}
        )
    products["scale_ranges_km"] = scale_ranges_km
    for field in FIELDS:
        S_n = np.divide(
            sums[field],
            counts[:, None, :],
            out=np.full(sums[field].shape, np.nan),
            where=counts[:, None, :] > 0,
        )
        products[field] = {
            "S_n": S_n,
            "flatness": _flatness(S_n, orders),
            "scaling_exponents": _fit_scaling_exponents(
                S_n, counts, centers, orders, scale_ranges_km, min_fit_bins
            ),
        }
    return products


def _observed_member_count(data: dict[str, Any]) -> int:
    observed = np.asarray(data["observed"]["n"])
    truth = np.asarray(data["truth"]["n"])
    return int(observed.shape[0]) if observed.ndim == truth.ndim + 1 else 1


def _interpolate_positions_with_offsets(
    position_sc_time_km: np.ndarray,
    source_time: np.ndarray,
    output_time: np.ndarray,
) -> np.ndarray:
    time_first = np.swapaxes(position_sc_time_km, 0, 1)
    return _interpolate(time_first, source_time, output_time, "spacecraft position")


def _candidate_position_nodes(
    handle: h5py.File,
    group_name: str,
) -> list[tuple[str, h5py.Dataset, np.ndarray, bool]]:
    candidates: list[tuple[str, h5py.Dataset, np.ndarray, bool]] = []
    group_specific_roots = (
        f"{group_name}/spacecraft/r",
        f"{group_name}/r",
        f"{group_name}/positions",
        f"{group_name}/position",
    )
    for root in group_specific_roots:
        if root in handle and isinstance(handle[root], h5py.Group):
            for variable in POSITION_VARIABLES:
                path = f"{root}/{variable}"
                if path in handle:
                    candidates.append((path, handle[path], _time_axis(handle, variable), True))
        elif root in handle and isinstance(handle[root], h5py.Dataset):
            candidates.append((root, handle[root], _time_axis(handle, "v"), True))
    if "spacecraft/r" in handle:
        node = handle["spacecraft/r"]
        if isinstance(node, h5py.Group):
            for variable in POSITION_VARIABLES:
                if variable in node:
                    candidates.append(
                        (
                            f"spacecraft/r/{variable}",
                            node[variable],
                            _time_axis(handle, variable),
                            False,
                        )
                    )
        elif isinstance(node, h5py.Dataset):
            candidates.append(("spacecraft/r", node, _time_axis(handle, "v"), False))
    return candidates


def _positions_for_group_member(
    input_path: str | Path,
    group_name: str,
    member_index: int,
    output_time: np.ndarray,
    n_spacecraft: int,
) -> tuple[np.ndarray, str, list[str]]:
    notes: list[str] = []
    with h5py.File(input_path, "r") as handle:
        candidates = _candidate_position_nodes(handle, group_name)
        if not candidates:
            raise KeyError("Input file is missing spacecraft positions")
        exact = next(
            (item for item in candidates if np.array_equal(item[2], output_time)),
            candidates[0],
        )
        path, dataset, source_time, group_specific = exact
        position = np.asarray(dataset, dtype=float)
        if position.ndim == 4:
            if member_index >= position.shape[0]:
                raise IndexError(f"member {member_index} is outside /{path}")
            position = position[member_index]
        if position.shape != (n_spacecraft, source_time.size, 3):
            raise ValueError(f"/{path} must have shape (spacecraft, time, 3)")
        position_offset = None
        # Current schema stores hidden realized spatial offsets under /errors
        # while nominal /spacecraft/r remains unchanged. For observed products,
        # reconstruct the measured spatial coordinates from position offsets
        # when no group-specific measured-position dataset is available. Timing
        # offsets affect field sampling time, but the generator does not advect
        # spacecraft positions by those clock offsets.
        if group_name == "observed" and not group_specific:
            if "errors/position_offset" in handle:
                offsets = np.asarray(handle["errors/position_offset"], dtype=float)
                position_offset = offsets[member_index] if offsets.ndim == 3 else offsets
                notes.append("observed positions use /errors/position_offset on nominal r")
        positions_km = _interpolate_positions_with_offsets(
            position,
            source_time,
            output_time,
        )
        if position_offset is not None:
            positions_km = positions_km + position_offset[None, :, :]
        source = path
        if group_name == "observed" and not group_specific and position_offset is None:
            notes.append("observed positions fell back to nominal /spacecraft/r")
        if group_name == "truth" and not group_specific:
            notes.append("truth positions fell back to nominal /spacecraft/r")
    return positions_km, source, notes


def _analysis_config(
    settings: dict[str, Any],
    group_name: str,
    member_index: int,
) -> dict[str, Any]:
    config = dict(settings["elsasser"])
    config["data_group"] = group_name
    config["mc_index"] = member_index
    return config


@register_analysis
class StructureFunctionAnalysis(AnalysisMethod):
    """Compute binned pairwise Elsasser vector-increment structure functions."""

    name = "structure_function"
    config_section = "structure_function"

    def run(self, input_path: str | Path) -> dict[str, Any]:
        data = read_observations(input_path)
        settings = _settings(self.config)
        groups = ["observed"]
        if "truth" in data:
            groups.append("truth")
        n_observed = _observed_member_count(data)
        pairs = list(combinations(range(len(data["spacecraft"]["id"])), 2))

        first_pass: dict[str, list[dict[str, Any]]] = {}
        all_separations: list[np.ndarray] = []
        lambda_lower_km = np.inf
        lambda_upper_km = -np.inf
        notes: list[str] = []
        field_names_used: dict[str, Any] = {}
        for group_name in groups:
            n_members = n_observed if group_name == "observed" else 1
            first_pass[group_name] = []
            field_names_used[group_name] = {"positions": []}
            for member_index in range(n_members):
                elsasser = compute_elsasser_analysis(
                    input_path,
                    _analysis_config(settings, group_name, member_index),
                )
                positions_km, position_source, position_notes = _positions_for_group_member(
                    input_path,
                    group_name,
                    member_index,
                    np.asarray(elsasser["time"], dtype=float),
                    len(elsasser["spacecraft_ids"]),
                )
                first_pass[group_name].append({"elsasser": elsasser, "positions_km": positions_km})
                field_names_used[group_name]["positions"].append(position_source)
                notes.extend(position_notes)
                if settings["increment_mode"] == "same_time_pairs":
                    for left, right in pairs:
                        all_separations.append(
                            np.linalg.norm(positions_km[:, left] - positions_km[:, right], axis=1)
                        )
                elif settings["lambda_bins_km"] is None:
                    lower, upper = _spacetime_lambda_bounds_km(
                        positions_km,
                        settings["pair_sampling"],
                        settings["max_pair_lambda_km"],
                    )
                    lambda_lower_km = min(lambda_lower_km, lower)
                    lambda_upper_km = max(lambda_upper_km, upper)

        if settings["increment_mode"] == "same_time_pairs":
            lambda_edges_km = _lambda_edges_km(np.concatenate(all_separations), settings)
        else:
            lambda_edges_km = _lambda_edges_from_bounds_km(
                lambda_lower_km,
                lambda_upper_km,
                settings,
            )
        products: dict[str, Any] = {}
        for group_name, members in first_pass.items():
            positions = np.stack([member["positions_km"] for member in members], axis=0)
            z_plus = np.stack([member["elsasser"]["z_plus"] for member in members], axis=0)
            z_minus = np.stack([member["elsasser"]["z_minus"] for member in members], axis=0)
            if settings["increment_mode"] == "same_time_pairs":
                group_result = compute_pairwise_structure_functions(
                    positions,
                    z_plus,
                    z_minus,
                    settings["orders"],
                    lambda_edges_km,
                    scale_ranges_km=settings["scale_ranges_km"] or None,
                    min_fit_bins=settings["min_fit_bins"],
                    max_pair_lambda_km=settings["max_pair_lambda_km"],
                )
            else:
                group_result = compute_all_spacetime_structure_functions(
                    positions,
                    z_plus,
                    z_minus,
                    settings["orders"],
                    lambda_edges_km,
                    pair_sampling=settings["pair_sampling"],
                    scale_ranges_km=settings["scale_ranges_km"] or None,
                    min_fit_bins=settings["min_fit_bins"],
                    max_pair_lambda_km=settings["max_pair_lambda_km"],
                )
            group_result["n_members"] = positions.shape[0]
            group_result["time_s"] = np.stack(
                [member["elsasser"]["time"] for member in members], axis=0
            )
            products[group_name] = group_result
            for field in FIELDS:
                notes.extend(group_result[field]["scaling_exponents"]["notes"])

        result = {
            "method": self.name,
            "config": {
                "min_order": settings["orders"][0],
                "max_order": settings["orders"][-1],
                "n_lambda_bins": settings["n_lambda_bins"],
                "lambda_bins_km": settings["lambda_bins_km"],
                "scale_ranges_km": settings["scale_ranges_km"],
                "min_fit_bins": settings["min_fit_bins"],
                "max_pair_lambda_km": settings["max_pair_lambda_km"],
                "increment_mode": settings["increment_mode"],
                "pair_sampling": settings["pair_sampling"],
            },
            "spacecraft_pairs": [list(pair) for pair in pairs],
            "products": products,
            "metadata": {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "notes": sorted(set(notes)),
                "units": {
                    "lambda_bin_edges_km": "km",
                    "lambda_bin_centers_km": "km",
                    "pair_separations_km": "km",
                    "S_n": "(km/s)^n",
                    "time_s": "s",
                },
                "field_names_used": field_names_used,
                "increment_type": (
                    "same-time spacecraft-pair vector magnitude"
                    if settings["increment_mode"] == "same_time_pairs"
                    else "all-spacetime point-pair vector magnitude"
                ),
                "increment_mode": settings["increment_mode"],
                "pair_sampling": settings["pair_sampling"],
                "max_pair_lambda_km": settings["max_pair_lambda_km"],
                "ensemble_policy": "members are processed independently and not averaged",
            },
        }
        return _to_serializable(result)

    def write_result(self, result: dict[str, Any], output_path: str | Path) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        settings = _settings(self.config)
        if settings["plot"]["enabled"]:
            plot_structure_function_diagnostic(
                result,
                settings["plot"]["path"],
                data_group=settings["plot"]["data_group"],
                field=settings["plot"]["field"],
                member=settings["plot"]["member"],
                normalized_exponents=settings["plot"]["normalized_exponents"],
                dpi=settings["plot"]["dpi"],
            )
        return output


def plot_structure_function_diagnostic(
    result: dict[str, Any],
    output_path: str | Path = "structure_function_diagnostic.png",
    *,
    data_group: str = "observed",
    field: str = "z_plus",
    member: int = 0,
    normalized_exponents: bool = False,
    dpi: int = 150,
) -> Path:
    """Create a three-panel structure-function, exponent, and flatness summary."""

    import matplotlib.pyplot as plt

    if field not in FIELDS:
        raise ValueError("field must be 'z_plus' or 'z_minus'")
    products = result["products"][data_group]
    orders = np.asarray(products["orders"], dtype=int)
    edges_km = np.asarray(products["lambda_bin_edges_km"], dtype=float)
    centers_km = np.asarray(products["lambda_bin_centers_km"], dtype=float)
    counts = np.asarray(products["counts"], dtype=float)
    values = np.asarray(products[field]["S_n"], dtype=float)
    if not 0 <= member < values.shape[0]:
        raise IndexError(f"member {member} is outside {data_group}/{field}")
    exponents = products[field]["scaling_exponents"]
    exponent_key = "normalized_by_order2" if normalized_exponents else "raw"
    xi = np.asarray(exponents[exponent_key], dtype=float)
    ranges = exponents["scale_ranges_km"]
    range_names = list(exponents["range_names"])
    flatness_values = products[field]["flatness"]
    flatness = None if flatness_values is None else np.asarray(flatness_values, dtype=float)

    figure, axes = plt.subplots(1, 4, figsize=(18.0, 4.2), constrained_layout=True)
    for axis in (axes[0], axes[2], axes[3]):
        for name, bounds in ranges.items():
            axis.axvspan(bounds[0], bounds[1], alpha=0.08, label=name)
    for order_index, order in enumerate(orders):
        axes[0].loglog(centers_km, values[member, order_index], marker="o", label=f"n={order}")
    axes[0].set_xlabel(r"$\lambda$ [km]")
    axes[0].set_ylabel(r"$S_n(\lambda)$")
    axes[0].set_title(field.replace("_", " "))
    axes[0].grid(True, which="both", alpha=0.22)
    axes[0].legend(fontsize=8, ncols=2)

    for range_index, name in enumerate(range_names):
        axes[1].plot(orders, xi[member, range_index], marker="o", label=name)
    if not normalized_exponents:
        axes[1].plot(orders, orders / 3.0, color="0.35", linestyle="--", label="K41 n/3")
        axes[1].set_ylabel(r"$\xi_n$")
    else:
        axes[1].axhline(1.0, color="0.35", linestyle="--", linewidth=1.0)
        axes[1].set_ylabel(r"$\xi_n/\xi_2$")
    axes[1].set_xlabel("order n")
    axes[1].set_title("scaling exponents")
    axes[1].grid(True, alpha=0.22)
    axes[1].legend(fontsize=8)

    if flatness is not None:
        axes[2].loglog(centers_km, flatness[member], marker="o", label=field.replace("_", " "))
    axes[2].axhline(
        5.0 / 3.0,
        color="0.35",
        linestyle="--",
        linewidth=1.0,
        label="3D Gaussian",
    )
    axes[2].set_xlabel(r"$\lambda$ [km]")
    axes[2].set_ylabel(r"$S_4/S_2^2$")
    axes[2].set_title("flatness")
    axes[2].grid(True, which="both", alpha=0.22)
    axes[2].legend(fontsize=8)

    axes[3].stairs(counts[member], edges_km, color="#2f6f9f", linewidth=1.5)
    axes[3].scatter(
        centers_km,
        counts[member],
        color="#2f6f9f",
        s=18,
        zorder=3,
    )
    axes[3].set_xscale("log")
    if np.any(counts[member] > 0):
        axes[3].set_yscale("log")
    axes[3].set_xlabel(r"$\lambda$ [km]")
    increment_mode = result.get("metadata", {}).get("increment_mode", "same_time_pairs")
    if increment_mode == "all_spacetime_pairs":
        axes[3].set_ylabel("spacetime point pairs")
    else:
        axes[3].set_ylabel("pair-time samples")
    axes[3].set_title(r"$\lambda$ coverage")
    axes[3].grid(True, which="both", alpha=0.22)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return output
