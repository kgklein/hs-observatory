"""Exact stratified identity sampling for common-velocity affine trajectories.

Positions are (time, spacecraft, xyz). A lag group has T-|k| starts; sampling
uniform integer ranks over their concatenation weights groups automatically.
Near numerical boundaries, actual coordinates resolve each start individually.
"""

from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
import hashlib

import numpy as np

ALGORITHM_VERSION = "geometry_stratified_v1"


def geometry_targets(value, n_bins: int) -> np.ndarray:
    """Accept a nonnegative integer target or one integer per bin."""
    values = np.asarray(value)
    if values.dtype.kind not in "iu" or values.ndim > 1:
        raise ValueError("geometry_stratified targets must be nonnegative integers")
    if values.ndim == 0:
        values = np.repeat(values[None], n_bins)
    if values.shape != (n_bins,) or np.any(values < 0):
        raise ValueError("geometry_stratified requires one nonnegative target per separation bin")
    if np.any(values > np.iinfo(np.int64).max):
        raise ValueError("geometry_stratified targets exceed int64 capacity")
    return values.astype(np.int64)


def _affine_model(positions: np.ndarray) -> tuple[np.ndarray, float]:
    if (positions.ndim != 3 or positions.shape[-1] != 3
            or positions.shape[0] < 1 or positions.shape[1] < 1
            or not np.all(np.isfinite(positions))):
        raise ValueError("geometry_stratified positions must be finite (time, spacecraft, 3)")
    n_time = len(positions)
    step = ((positions[-1, 0] - positions[0, 0]) / (n_time - 1)
            if n_time > 1 else np.zeros(3))
    predicted = positions[0][None] + np.arange(n_time)[:, None, None] * step
    residual = float(np.max(np.linalg.norm(positions - predicted, axis=-1)))
    # Allow floating-point construction/offset roundoff, not formation evolution.
    rounding = 64 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(positions))))
    if residual > rounding:
        raise ValueError(
            "geometry_stratified requires common-velocity affine trajectories on the "
            "aligned time grid; use pair_sampling.mode='all' for unsupported geometry"
        )
    return step, 2 * residual + rounding


def _lag_groups(positions, step):
    n_time, n_spacecraft, _ = positions.shape
    for i in range(n_spacecraft):
        for j in range(i, n_spacecraft):
            k = np.arange(1, n_time) if i == j else np.arange(1 - n_time, n_time)
            d = np.linalg.norm(positions[0, j] - positions[0, i] + k[:, None] * step, axis=1)
            yield i, j, k, d


def _classify(d, edges, lower, upper):
    q = np.searchsorted(edges, d, side="right") - 1
    q[d == edges[-1]] = len(edges) - 2
    valid = (d > 0) & (q >= 0) & (q < len(edges) - 1)
    if lower is not None:
        valid &= d >= lower
    if upper is not None:
        valid &= d <= upper
    return np.where(valid, q, -1)


@dataclass
class GeometryBin:
    # Columns: spacecraft i, spacecraft j, signed lag, first start, multiplicity.
    groups: np.ndarray
    cumulative: np.ndarray
    explicit_starts: dict[int, np.ndarray]

    @property
    def population(self) -> int:
        return int(self.cumulative[-1]) if len(self.cumulative) else 0

    def identities(self, ranks, n_spacecraft):
        g = np.searchsorted(self.cumulative, ranks, side="right")
        previous = np.zeros(len(ranks), dtype=np.int64)
        nonzero = g > 0
        previous[nonzero] = self.cumulative[g[nonzero] - 1]
        offsets = ranks - previous
        start = self.groups[g, 3] + offsets
        for row, starts in self.explicit_starts.items():
            mask = g == row
            start[mask] = starts[offsets[mask]]
        a = start * n_spacecraft + self.groups[g, 0]
        b = (start + self.groups[g, 2]) * n_spacecraft + self.groups[g, 1]
        return np.minimum(a, b), np.maximum(a, b)


@dataclass
class GeometryPairPlan:
    bins: tuple[GeometryBin, ...]
    edges: np.ndarray
    n_spacecraft: int
    candidate_groups: int
    boundary_pairs_examined: int

    @property
    def populations(self):
        return np.array([b.population for b in self.bins], dtype=np.int64)


def build_geometry_pair_plan(
    positions_km, lambda_edges_km, *, min_pair_lambda_km=None, max_pair_lambda_km=None,
) -> GeometryPairPlan:
    """Enumerate discrete lags, preserving the existing exact norm/bin predicate.

    Roundoff-close groups are expanded only to classify their starting times.
    Uniform groups retain implicit consecutive starts; only split groups store
    explicit starts. No field values enter this plan or the random selection.
    """
    positions = np.asarray(positions_km, dtype=float)
    step, error = _affine_model(positions)
    edges = np.asarray(lambda_edges_km, dtype=float)
    if (edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges))
            or np.any(np.diff(edges) <= 0)):
        raise ValueError("geometry_stratified requires finite increasing separation bin edges")
    lower, upper = min_pair_lambda_km, max_pair_lambda_km
    boundaries = np.unique([0.0, *edges, *([] if lower is None else [lower]),
                            *([] if upper is None else [upper])])
    rows: list[list[np.ndarray]] = [[] for _ in edges[:-1]]
    row_counts = np.zeros(len(edges) - 1, dtype=np.int64)
    explicit: list[dict[int, np.ndarray]] = [{} for _ in edges[:-1]]
    examined = boundary_pairs = 0
    for i, j, lags, distances in _lag_groups(positions, step):
        examined += len(lags)
        classification = _classify(distances, edges, lower, upper)
        insertion = np.searchsorted(boundaries, distances)
        gap = np.minimum(
            abs(distances - boundaries[np.clip(insertion, 0, len(boundaries) - 1)]),
            abs(distances - boundaries[np.clip(insertion - 1, 0, len(boundaries) - 1)]),
        )
        near = gap <= error
        # The usual case stays vectorized over lags; expand only numerical
        # boundary exceptions. Never loop over every pair-time combination.
        for q in np.unique(classification[(classification >= 0) & ~near]):
            selected_lags = lags[(classification == q) & ~near]
            size = len(selected_lags)
            rows[q].append(np.column_stack((
                np.full(size, i), np.full(size, j), selected_lags,
                np.maximum(0, -selected_lags), len(positions) - abs(selected_lags),
            )))
            row_counts[q] += size
        for k in lags[near]:
            start0, weight = max(0, -int(k)), len(positions) - abs(int(k))
            starts = np.arange(start0, start0 + weight)
            actual = np.linalg.norm(positions[starts + k, j] - positions[starts, i], axis=1)
            labels = _classify(actual, edges, lower, upper)
            boundary_pairs += weight
            for label in np.unique(labels[labels >= 0]):
                selected = starts[labels == label]
                row = row_counts[label]
                rows[label].append(np.array([[i, j, k, start0, len(selected)]]))
                row_counts[label] += 1
                if len(selected) != weight:
                    explicit[label][row] = selected
    bins = []
    for row, starts in zip(rows, explicit):
        groups = np.concatenate(row).astype(np.int64) if row else np.empty((0, 5), dtype=np.int64)
        bins.append(GeometryBin(groups, np.cumsum(groups[:, 4]), starts))
    return GeometryPairPlan(tuple(bins), edges.copy(), positions.shape[1], examined, boundary_pairs)


class GeometryPlanCache:
    """Run-local bounded LRU; only byte-identical coordinates reuse exact plans."""

    def __init__(self, max_entries=2):
        self.max_entries = max_entries
        self._plans = OrderedDict()

    def get(self, positions, edges, lower=None, upper=None):
        positions = np.ascontiguousarray(positions, dtype=float)
        edges = np.ascontiguousarray(edges, dtype=float)
        digest = hashlib.sha256(positions.tobytes()).digest()
        key = (positions.shape, digest, edges.tobytes(), lower, upper)
        if key in self._plans:
            self._plans.move_to_end(key)
            return self._plans[key], True
        plan = build_geometry_pair_plan(
            positions, edges, min_pair_lambda_km=lower, max_pair_lambda_km=upper,
        )
        self._plans[key] = plan
        if len(self._plans) > self.max_entries:
            self._plans.popitem(last=False)
        return plan, False


def geometry_pair_chunks(
    plan: GeometryPairPlan, target_pairs_per_bin, *, random_seed: int, chunk_size: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Sample uniform ranks without replacement using sparse Fisher–Yates.

    PCG64 streams are keyed by seed and the two bin-edge float64 bit patterns.
    Scalar draws give chunk-independent sequences, with no MC-order input.
    A census streams ordered ranks without allocating a permutation or hash map.
    """
    targets = geometry_targets(target_pairs_per_bin, len(plan.bins))
    if chunk_size < 1 or random_seed is None or random_seed < 0:
        raise ValueError("geometry_stratified requires a positive chunk size and nonnegative seed")
    for q, (bin_plan, target) in enumerate(zip(plan.bins, targets)):
        population = bin_plan.population
        count = min(int(target), population)
        words = np.asarray(plan.edges[q:q + 2], dtype="<f8").view("<u4")
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
            [int(random_seed), *map(int, words)]
        )))
        swaps: dict[int, int] = {}
        for start in range(0, count, chunk_size):
            stop = min(count, start + chunk_size)
            ranks = np.empty(stop - start, dtype=np.int64)
            if count == population:
                ranks[:] = np.arange(start, stop)
            else:
                for offset, draw in enumerate(range(start, stop)):
                    remaining = population - draw
                    chosen = int(rng.integers(remaining))
                    ranks[offset] = swaps.get(chosen, chosen)
                    last = remaining - 1
                    replacement = swaps.pop(last, last)
                    if chosen != last:
                        swaps[chosen] = replacement
            yield bin_plan.identities(ranks, plan.n_spacecraft)


def geometry_lambda_bounds(positions_km, lower=None, upper=None):
    """Deterministic conservative automatic bounds, without random pair probes."""
    positions = np.asarray(positions_km, dtype=float)
    step, error = _affine_model(positions)
    minimum, maximum = np.inf, -np.inf
    for i, j, lags, d in _lag_groups(positions, step):
        valid = d > error
        if lower is not None:
            valid &= d + error >= lower
        if upper is not None:
            valid &= d - error <= upper
        if np.any(valid):
            minimum = min(minimum, float(np.min(d[valid])) - error)
            maximum = max(maximum, float(np.max(d[valid])) + error)
        # A modeled zero can hide tiny positive stored-coordinate differences;
        # resolve these and pair-limit tangencies rather than discarding them.
        near = d <= error
        for limit in (lower, upper):
            if limit is not None:
                near |= abs(d - limit) <= error
        for k in lags[near]:
            starts = np.arange(max(0, -int(k)), min(len(positions), len(positions) - int(k)))
            actual = np.linalg.norm(positions[starts + k, j] - positions[starts, i], axis=1)
            keep = actual > 0
            if lower is not None:
                keep &= actual >= lower
            if upper is not None:
                keep &= actual <= upper
            if np.any(keep):
                minimum = min(minimum, float(np.min(actual[keep])))
                maximum = max(maximum, float(np.max(actual[keep])))
    return minimum, maximum
