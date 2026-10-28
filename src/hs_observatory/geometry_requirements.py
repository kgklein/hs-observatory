"""Nominal HS_Hours baseline coverage, independent of plotting and Monte Carlo.

Categories are ordered by scale, then quantity (X, Y, Z, D). Bounds are
lower-inclusive and upper-exclusive. Inputs must be unrounded vectors in km.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


SPACECRAFT_IDS = ("H", *(f"N{i}" for i in range(1, 9)))
CATEGORY_LABELS = tuple(
    f"{quantity}:{scale}"
    for scale in ("small", "intermediate", "large")
    for quantity in ("X", "Y", "Z", "D")
)
REFERENCE_WITNESS = (("H", "N1"), ("H", "N5"), ("H", "N6"), ("N2", "N4"))


@dataclass(frozen=True)
class BaselineCoverage:
    """Available-baseline memberships and an independently verified witness."""

    pairs: tuple[tuple[str, str], ...]
    category_labels: tuple[str, ...]
    membership: np.ndarray  # (baseline, category), unavailable endpoints excluded
    counts: np.ndarray  # (scale, quantity)
    category_covered: np.ndarray
    all_categories_covered: bool
    hub_available: bool
    satisfies_requirement: bool
    contributing_pairs: dict[str, tuple[tuple[str, str], ...]]
    witness_pairs: tuple[tuple[str, str], ...]
    witness_category_covered: np.ndarray


def _canonical_pair(pair: Sequence[str]) -> tuple[str, str]:
    if len(pair) != 2 or pair[0] == pair[1] or any(p not in SPACECRAFT_IDS for p in pair):
        raise ValueError(f"Invalid baseline endpoints: {pair!r}")
    return tuple(sorted(pair, key=SPACECRAFT_IDS.index))


def evaluate_baseline_coverage(
    vectors: np.ndarray,
    pairs: Sequence[tuple[str, str]],
    *,
    availability: Mapping[str, bool] | None = None,
    witness_pairs: Sequence[tuple[str, str]] | None = None,
) -> BaselineCoverage:
    """Evaluate all twelve categories, optionally verifying an explicit witness.

    With no explicit witness, greedily cover uncovered categories, breaking ties
    by canonical spacecraft order; then prune redundant edges in reverse order.
    This is deterministic, not a claim of mathematical minimality. Incomplete
    configurations return no automatic witness. Explicit invalid witnesses raise.
    An availability mapping must specify H and every supplied endpoint.
    """
    vectors = np.asarray(vectors, dtype=float)
    pairs = tuple(_canonical_pair(pair) for pair in pairs)
    if vectors.shape != (len(pairs), 3) or not np.all(np.isfinite(vectors)):
        raise ValueError("vectors must be finite with shape (n_baselines, 3)")
    if len(set(pairs)) != len(pairs):
        raise ValueError("Baseline endpoint pairs must be unique")
    required_ids = {"H", *(name for pair in pairs for name in pair)}
    if availability is not None and not required_ids.issubset(availability):
        raise ValueError("availability must specify H and every baseline endpoint")
    available = {name: True for name in required_ids} if availability is None else availability
    active = np.asarray([available[a] and available[b] for a, b in pairs], dtype=bool)
    quantities = np.column_stack((np.abs(vectors), np.linalg.norm(vectors, axis=1)))
    membership = np.concatenate(
        [
            (quantities >= lower) & (quantities < upper)
            for lower, upper in (([2.5, 2.5, 2.5, 10.0], 100.0), (100.0, 1200.0), (1200.0, 5000.0))
        ],
        axis=1,
    )
    membership &= active[:, None]
    covered = membership.any(axis=0)
    hub_available = bool(available["H"])
    success = bool(covered.all() and hub_available)
    canonical_order = sorted(
        range(len(pairs)), key=lambda i: tuple(SPACECRAFT_IDS.index(p) for p in pairs[i])
    )
    chosen: list[int] = []
    if witness_pairs is not None:
        requested = tuple(_canonical_pair(pair) for pair in witness_pairs)
        if len(set(requested)) != len(requested):
            raise ValueError("Witness pairs must be unique")
        unknown = [pair for pair in requested if pair not in pairs]
        if unknown:
            raise ValueError(f"Witness pairs absent from baseline catalog: {unknown}")
        chosen = [pairs.index(pair) for pair in requested]
        witness_covered = membership[chosen].any(axis=0)
        unavailable_pairs = [pairs[i] for i in chosen if not active[i]]
        if not hub_available or not witness_covered.all() or unavailable_pairs:
            missing = [label for label, ok in zip(CATEGORY_LABELS, witness_covered) if not ok]
            raise ValueError(
                f"Witness does not satisfy baseline requirement; missing categories: {missing}; "
                f"hub available: {hub_available}; unavailable pairs: {unavailable_pairs}"
            )
    elif success:
        remaining = np.ones(12, dtype=bool)
        while remaining.any():
            index = max(canonical_order, key=lambda i: np.count_nonzero(membership[i] & remaining))
            chosen.append(index)
            remaining &= ~membership[index]
        for index in reversed(chosen.copy()):
            reduced = [i for i in chosen if i != index]
            if membership[reduced].any(axis=0).all():
                chosen = reduced
    return BaselineCoverage(
        pairs=pairs,
        category_labels=CATEGORY_LABELS,
        membership=membership,
        counts=membership.sum(axis=0).reshape(3, 4),
        category_covered=covered,
        all_categories_covered=bool(covered.all()),
        hub_available=hub_available,
        satisfies_requirement=success,
        contributing_pairs={
            label: tuple(pairs[i] for i in np.flatnonzero(membership[:, column]))
            for column, label in enumerate(CATEGORY_LABELS)
        },
        witness_pairs=tuple(pairs[i] for i in chosen),
        witness_category_covered=membership[chosen].any(axis=0),
    )
