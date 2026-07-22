"""Memory-bounded pair enumeration and vector projection utilities."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def possible_pair_count(n_points: int) -> int:
    """Return the number of unique unordered pairs from ``n_points``."""

    if n_points < 0:
        raise ValueError("n_points must be non-negative")
    return n_points * (n_points - 1) // 2


def selected_pair_count(
    n_points: int,
    *,
    mode: str = "all",
    n_pairs: int | None = None,
    max_pairs: int | None = None,
) -> int:
    """Return the number of pairs that the configured iterator will emit."""

    total = possible_pair_count(n_points)
    if mode == "all":
        selected = total
    elif mode == "random":
        if n_pairs is None or n_pairs < 1:
            raise ValueError("random pair sampling requires a positive n_pairs")
        selected = min(n_pairs, total)
    else:
        raise ValueError("pair sampling mode must be 'all' or 'random'")
    if max_pairs is not None:
        if max_pairs < 1:
            raise ValueError("max_pairs must be positive or null")
        selected = min(selected, max_pairs)
    return selected


def _ordinal_pairs(ordinals: np.ndarray, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Map lexicographic upper-triangle ordinals to point indices."""

    row_ends = np.cumsum(np.arange(n_points - 1, 0, -1, dtype=np.int64))
    left = np.searchsorted(row_ends, ordinals, side="right")
    previous_end = np.where(left == 0, 0, row_ends[left - 1])
    right = left + 1 + (ordinals - previous_end)
    return left.astype(np.int64), right.astype(np.int64)


def pair_chunks(
    n_points: int,
    chunk_size: int,
    *,
    mode: str = "all",
    n_pairs: int | None = None,
    random_seed: int | None = None,
    max_pairs: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield unique unordered point-index pairs in bounded chunks.

    ``mode='all'`` streams the upper triangle without constructing it.
    ``mode='random'`` samples pair ordinals uniformly without replacement.
    Reusing the same seed reproduces the same selection across streaming passes.
    """

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    count = selected_pair_count(n_points, mode=mode, n_pairs=n_pairs, max_pairs=max_pairs)
    if mode == "random":
        rng = np.random.default_rng(random_seed)
        ordinals = rng.choice(possible_pair_count(n_points), size=count, replace=False)
        for start in range(0, count, chunk_size):
            yield _ordinal_pairs(ordinals[start : start + chunk_size], n_points)
        return

    remaining = count
    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []
    buffered = 0
    for left_index in range(max(0, n_points - 1)):
        if remaining == 0:
            break
        row_count = min(n_points - left_index - 1, remaining)
        row_start = left_index + 1
        while row_count:
            take = min(row_count, chunk_size - buffered)
            left_parts.append(np.full(take, left_index, dtype=np.int64))
            right_parts.append(np.arange(row_start, row_start + take, dtype=np.int64))
            buffered += take
            row_start += take
            row_count -= take
            remaining -= take
            if buffered == chunk_size:
                yield np.concatenate(left_parts), np.concatenate(right_parts)
                left_parts.clear()
                right_parts.clear()
                buffered = 0
    if buffered:
        yield np.concatenate(left_parts), np.concatenate(right_parts)


def adaptive_stratified_pairs(
    positions_km: np.ndarray,
    lambda_edges_km: np.ndarray,
    *,
    target_pairs_per_bin: int,
    max_candidates: int,
    candidate_chunk_size: int,
    random_seed: int | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Retain random unique pairs until every separation bin reaches its target.

    Candidate ordinals are drawn uniformly in bounded batches. Draws may repeat,
    but a pair can be retained only once. This avoids storing every examined
    ordinal while keeping the retained sample uniform within each lambda bin.
    """

    positions = np.asarray(positions_km, dtype=float)
    edges = np.asarray(lambda_edges_km, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions_km must have shape (point, 3)")
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("lambda_edges_km must be increasing one-dimensional edges")
    if target_pairs_per_bin < 1:
        raise ValueError("target_pairs_per_bin must be positive")
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    if candidate_chunk_size < 1:
        raise ValueError("candidate_chunk_size must be positive")

    total = possible_pair_count(positions.shape[0])
    if total == 0:
        raise ValueError("at least two positions are required")
    n_bins = edges.size - 1
    retained_by_bin: list[list[int]] = [[] for _ in range(n_bins)]
    retained_ordinals: set[int] = set()
    rng = np.random.default_rng(random_seed)
    candidates_examined = 0

    while candidates_examined < max_candidates and any(
        len(values) < target_pairs_per_bin for values in retained_by_bin
    ):
        draw_count = min(candidate_chunk_size, max_candidates - candidates_examined)
        ordinals = rng.integers(0, total, size=draw_count, dtype=np.int64)
        candidates_examined += draw_count
        ordinals = np.unique(ordinals)
        rng.shuffle(ordinals)
        left, right = _ordinal_pairs(ordinals, positions.shape[0])
        separations = np.linalg.norm(positions[right] - positions[left], axis=1)
        bin_indices = np.searchsorted(edges, separations, side="right") - 1
        bin_indices[separations == edges[-1]] = n_bins - 1

        for ordinal, bin_index in zip(ordinals.tolist(), bin_indices.tolist()):
            if bin_index < 0 or bin_index >= n_bins or ordinal in retained_ordinals:
                continue
            retained = retained_by_bin[bin_index]
            if len(retained) < target_pairs_per_bin:
                retained.append(ordinal)
                retained_ordinals.add(ordinal)

    retained_counts = np.asarray([len(values) for values in retained_by_bin], dtype=np.int64)
    selected = np.asarray(
        [ordinal for values in retained_by_bin for ordinal in values], dtype=np.int64
    )
    if selected.size:
        rng.shuffle(selected)
        left, right = _ordinal_pairs(selected, positions.shape[0])
    else:
        left = np.empty(0, dtype=np.int64)
        right = np.empty(0, dtype=np.int64)
    diagnostics: dict[str, object] = {
        "candidates_examined": candidates_examined,
        "retained_per_bin": retained_counts,
        "target_pairs_per_bin": target_pairs_per_bin,
        "target_reached": bool(np.all(retained_counts >= target_pairs_per_bin)),
    }
    return left, right, diagnostics


def perpendicular_projection(vectors: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Remove each vector's component along its corresponding unit direction."""

    values = np.asarray(vectors, dtype=float)
    unit = np.asarray(directions, dtype=float)
    if values.shape != unit.shape or values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("vectors and directions must have matching shape (sample, 3)")
    return values - np.sum(values * unit, axis=1)[:, None] * unit


def local_pairwise_directions(
    B_left: np.ndarray,
    B_right: np.ndarray,
    global_mean_B: np.ndarray,
    *,
    min_norm: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Return local mean-field unit vectors with a global-direction fallback."""

    left = np.asarray(B_left, dtype=float)
    right = np.asarray(B_right, dtype=float)
    global_field = np.asarray(global_mean_B, dtype=float)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 3:
        raise ValueError("B_left and B_right must have matching shape (sample, 3)")
    if global_field.shape != (3,):
        raise ValueError("global_mean_B must have shape (3,)")
    if min_norm < 0.0:
        raise ValueError("min_norm must be non-negative")

    local = 0.5 * (left + right)
    norms = np.linalg.norm(local, axis=1)
    valid_local = np.isfinite(norms) & (norms >= min_norm) & (norms > 0.0)
    directions = np.zeros_like(local)
    directions[valid_local] = local[valid_local] / norms[valid_local, None]

    fallback = ~valid_local
    global_norm = float(np.linalg.norm(global_field))
    global_valid = np.isfinite(global_norm) and global_norm >= min_norm and global_norm > 0.0
    if global_valid:
        directions[fallback] = global_field / global_norm
    valid = valid_local | (fallback & global_valid)
    return directions, valid


def separation_magnitude(
    displacement: np.ndarray,
    directions: np.ndarray,
    component: str,
) -> np.ndarray:
    """Return full or field-perpendicular separation magnitude."""

    values = np.asarray(displacement, dtype=float)
    if component == "full":
        return np.linalg.norm(values, axis=1)
    if component == "perpendicular":
        return np.linalg.norm(perpendicular_projection(values, directions), axis=1)
    raise ValueError("separation component must be 'perpendicular' or 'full'")
