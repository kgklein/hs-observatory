"""Process scheduling for SF members; scientific kernels remain in structure_function."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
import hashlib
import multiprocessing
from pathlib import Path
from typing import Any

import numpy as np

from hs_observatory.analysis import structure_function as sf


@dataclass(frozen=True)
class MemberTask:
    path: str
    group: str
    index: int
    slot: int
    settings: dict[str, Any]
    edges: np.ndarray
    shape: tuple[int, ...]
    position_source: str
    position_notes: tuple[str, ...]
    plan_reused: bool

    @property
    def identity(self) -> tuple[str, int, int]:
        return self.group, self.index, self.slot


def _read_member(path, settings, group, index):
    fields = sf.compute_elsasser_analysis(path, sf._analysis_config(settings, group, index))
    positions, source, notes = sf._positions_for_group_member(
        path, group, index, np.asarray(fields["time"], dtype=float), len(fields["spacecraft_ids"])
    )
    if positions.shape != fields["z_plus"].shape or positions.shape != fields["z_minus"].shape:
        raise ValueError("positions, z_plus, and z_minus must have matching shapes")
    return fields, positions, source, tuple(notes)


def evaluate_member(task: MemberTask):
    """Open this member independently and run the unchanged singleton evaluator."""
    fields, positions, source, notes = _read_member(
        task.path, task.settings, task.group, task.index
    )
    if positions.shape != task.shape or (source, notes) != (
        task.position_source,
        task.position_notes,
    ):
        raise ValueError("member shape or position provenance changed after preparation")
    settings = task.settings
    kwargs = dict(
        scale_ranges_km=settings["scale_ranges_km"] or None,
        min_fit_bins=settings["min_fit_bins"],
        min_pair_lambda_km=settings["min_pair_lambda_km"],
        max_pair_lambda_km=settings["max_pair_lambda_km"],
    )
    evaluator = sf.compute_pairwise_structure_functions
    if settings["increment_mode"] == "all_spacetime_pairs":
        evaluator = sf.compute_all_spacetime_structure_functions
        kwargs["pair_sampling"] = settings["pair_sampling"]
    product = evaluator(
        positions[None],
        fields["z_plus"][None],
        fields["z_minus"][None],
        settings["orders"],
        task.edges,
        **kwargs,
    )
    product["time_s"] = np.asarray(fields["time"])[None]
    return task.identity, product


def _validate_member(task, product):
    expected_keys = {
        "orders",
        "lambda_bin_edges_km",
        "lambda_bin_centers_km",
        "spacecraft_pairs",
        "pair_separations_km",
        "counts",
        "scale_ranges_km",
        "time_s",
        *sf.FIELDS,
    }
    if task.settings["increment_mode"] == "all_spacetime_pairs":
        expected_keys.add("spacetime_pair_counts")
    if not isinstance(product, dict) or set(product) != expected_keys:
        raise ValueError(f"incomplete or inconsistent member product for {task.identity}")
    n_bins = len(task.edges) - 1
    n_orders = len(task.settings["orders"])
    shapes = {"counts": (1, n_bins), "time_s": (1, task.shape[0])}
    for key, shape in shapes.items():
        if np.shape(product[key]) != shape:
            raise ValueError(f"invalid {key} shape for {task.identity}")
    if product["orders"] != task.settings["orders"] or not np.array_equal(
        product["lambda_bin_edges_km"], task.edges
    ):
        raise ValueError(f"inconsistent orders or edges for {task.identity}")
    if not np.array_equal(
        product["lambda_bin_centers_km"], np.sqrt(task.edges[:-1] * task.edges[1:])
    ):
        raise ValueError(f"inconsistent bin centers for {task.identity}")
    for field in sf.FIELDS:
        if np.shape(product[field]["S_n"]) != (1, n_orders, n_bins):
            raise ValueError(f"invalid {field} moment shape for {task.identity}")
        flatness = product[field]["flatness"]
        if flatness is not None and np.shape(flatness) != (1, n_bins):
            raise ValueError(f"invalid {field} flatness shape for {task.identity}")
    if task.settings["increment_mode"] == "same_time_pairs":
        n_pairs = task.shape[1] * (task.shape[1] - 1) // 2
        if np.shape(product["pair_separations_km"]) != (1, n_pairs, task.shape[0]):
            raise ValueError(f"invalid pair separation shape for {task.identity}")
    elif product["pair_separations_km"] is not None:
        raise ValueError(f"unexpected pair separations for {task.identity}")


def _submit_member(executor, task):
    try:
        return executor.submit(evaluate_member, task)
    except Exception as error:
        raise RuntimeError(
            f"SF worker submission failed for {task.group} member {task.index}"
        ) from error


def _collect_members(tasks, workers):
    """Bound submitted work and reject missing, duplicate, or misidentified results."""
    results = [None] * len(tasks)
    iterator = iter(tasks)
    executor = ProcessPoolExecutor(
        max_workers=min(workers, len(tasks)), mp_context=multiprocessing.get_context("spawn")
    )
    pending = {}
    try:
        for _ in range(min(workers, len(tasks))):
            task = next(iterator)
            pending[_submit_member(executor, task)] = task
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                task = pending.pop(future)
                try:
                    identity, product = future.result()
                    if identity != task.identity or results[task.slot] is not None:
                        raise ValueError("missing, duplicate, or inconsistent member identity")
                    _validate_member(task, product)
                    results[task.slot] = product
                except Exception as error:
                    raise RuntimeError(
                        f"SF worker failed for {task.group} member {task.index}"
                    ) from error
                following = next(iterator, None)
                if following is not None:
                    pending[_submit_member(executor, following)] = following
        if any(result is None for result in results):
            raise RuntimeError("SF workers returned an incomplete ensemble")
        return results
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def _assemble_members(tasks, members, settings):
    if not tasks or len(tasks) != len(members) or any(item is None for item in members):
        raise RuntimeError("cannot assemble an incomplete SF ensemble")
    first = members[0]
    result = {}
    for key in ("orders", "lambda_bin_edges_km", "lambda_bin_centers_km", "spacecraft_pairs"):
        if any(not np.array_equal(first[key], item[key]) for item in members):
            raise ValueError(f"inconsistent member {key}")
        result[key] = first[key]
    for key in ("counts", "pair_separations_km", "time_s"):
        values = [item[key] for item in members]
        if values[0] is None:
            if any(value is not None for value in values):
                raise ValueError(f"inconsistent member {key}")
            result[key] = None
        else:
            result[key] = np.concatenate(values, axis=0)
    if "spacetime_pair_counts" in first:
        diagnostics = {}
        for key, value in first["spacetime_pair_counts"].items():
            values = [item["spacetime_pair_counts"][key] for item in members]
            if key == "geometry_sampling":
                invariants = {k: v for k, v in value.items() if k != "members"}
                if any(
                    {k: v for k, v in item.items() if k != "members"} != invariants
                    or len(item["members"]) != 1
                    for item in values
                ):
                    raise ValueError("inconsistent geometry diagnostics")
                geometry_members = []
                for task, item in zip(tasks, values):
                    geometry_members.append(dict(item["members"][0], plan_reused=task.plan_reused))
                diagnostics[key] = dict(invariants, members=geometry_members)
            elif isinstance(value, np.ndarray):
                if any(item.shape != (1,) for item in values):
                    raise ValueError(f"invalid member diagnostic {key}")
                diagnostics[key] = np.concatenate(values)
            else:
                if any(item != value for item in values):
                    raise ValueError(f"inconsistent diagnostic {key}")
                diagnostics[key] = value
        result["spacetime_pair_counts"] = diagnostics
    centers = result["lambda_bin_centers_km"]
    edges = result["lambda_bin_edges_km"]
    ranges = settings["scale_ranges_km"]
    if not ranges:
        finite = np.isfinite(centers) & (np.sum(result["counts"], axis=0) > 0)
        ranges = (
            {"total": [float(np.min(centers[finite])), float(np.max(centers[finite]))]}
            if np.any(finite)
            else {"total": [float(edges[0]), float(edges[-1])]}
        )
    result["scale_ranges_km"] = ranges
    for field in sf.FIELDS:
        moments = np.concatenate([item[field]["S_n"] for item in members], axis=0)
        result[field] = {
            "S_n": moments,
            "flatness": sf._flatness(moments, settings["orders"]),
            "scaling_exponents": sf._fit_scaling_exponents(
                moments,
                result["counts"],
                centers,
                settings["orders"],
                ranges,
                settings["min_fit_bins"],
            ),
        }
    result["n_members"] = len(members)
    if tasks[0].group == "observed" and settings["mc_indices"] != "all":
        result["mc_indices"] = [task.index for task in tasks]
    return result


def run_members(input_path, settings, groups, observed_indices, pairs, workers):
    """Prepare common geometry without retaining prepared ensemble field arrays."""
    path = str(Path(input_path).resolve())
    prepared = []
    group_shapes = {}
    sources = {group: {"positions": []} for group in groups}
    notes = []
    lower, upper = np.inf, -np.inf
    logical_cache = OrderedDict()
    geometry = settings["pair_sampling"]["mode"] == "geometry_stratified"
    for group in groups:
        for index in observed_indices if group == "observed" else [0]:
            try:
                fields, positions, source, member_notes = _read_member(path, settings, group, index)
                shape = positions.shape
                if group in group_shapes and group_shapes[group] != shape:
                    raise ValueError("all input arrays must have the same shape within a group")
                group_shapes[group] = shape
                reused = False
                if geometry:
                    # Edges and limits are common to every member in this run. These
                    # remaining key components reproduce the serial two-entry LRU.
                    key = (
                        shape,
                        hashlib.sha256(np.ascontiguousarray(positions).tobytes()).digest(),
                    )
                    reused = key in logical_cache
                    logical_cache[key] = None
                    logical_cache.move_to_end(key)
                    if len(logical_cache) > 2:
                        logical_cache.popitem(last=False)
                if settings["lambda_bins_km"] is None:
                    if settings["increment_mode"] == "same_time_pairs":
                        for left, right in pairs:
                            distances = np.linalg.norm(
                                positions[:, left] - positions[:, right], axis=1
                            )
                            valid = np.isfinite(distances) & (distances > 0)
                            if settings["min_pair_lambda_km"] is not None:
                                valid &= distances >= settings["min_pair_lambda_km"]
                            if settings["max_pair_lambda_km"] is not None:
                                valid &= distances <= settings["max_pair_lambda_km"]
                            if np.any(valid):
                                lower = min(lower, float(np.min(distances[valid])))
                                upper = max(upper, float(np.max(distances[valid])))
                    else:
                        member_lower, member_upper = sf._spacetime_lambda_bounds_km(
                            positions,
                            settings["pair_sampling"],
                            settings["min_pair_lambda_km"],
                            settings["max_pair_lambda_km"],
                        )
                        lower, upper = min(lower, member_lower), max(upper, member_upper)
                prepared.append((group, index, shape, source, member_notes, reused))
                sources[group]["positions"].append(source)
                notes.extend(member_notes)
                del fields, positions
            except Exception as error:
                raise RuntimeError(f"SF preparation failed for {group} member {index}") from error
    if settings["increment_mode"] == "same_time_pairs":
        edges = sf._lambda_edges_km(np.asarray([lower, upper]), settings)
    else:
        edges = sf._lambda_edges_from_bounds_km(lower, upper, settings)
    tasks = [
        MemberTask(path, group, index, slot, settings, edges, shape, source, member_notes, reused)
        for slot, (group, index, shape, source, member_notes, reused) in enumerate(prepared)
    ]
    results = _collect_members(tasks, workers)
    products = {}
    for group in groups:
        group_tasks = [task for task in tasks if task.group == group]
        product = _assemble_members(
            group_tasks, [results[task.slot] for task in group_tasks], settings
        )
        products[group] = product
        for field in sf.FIELDS:
            notes.extend(product[field]["scaling_exponents"]["notes"])
    return products, sources, notes
