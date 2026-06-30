"""Minimal pairwise second-order structure-function scaffold."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from hs_observatory.analysis.base import AnalysisMethod
from hs_observatory.analysis.registry import register_analysis
from hs_observatory.io.hdf5 import read_observations


@register_analysis
class StructureFunctionAnalysis(AnalysisMethod):
    """Compute unbinned pairwise second moments as an extensible placeholder."""

    name = "structure_function"

    def run(self, input_path: str | Path) -> dict[str, Any]:
        data = read_observations(input_path)
        observed = data["observed"]
        pairs = list(combinations(range(len(data["spacecraft"]["id"])), 2))
        products: dict[str, Any] = {}
        for quantity, values in observed.items():
            moments = []
            for left, right in pairs:
                increment = values[:, :, left, ...] - values[:, :, right, ...]
                if increment.ndim == 3:  # vector: sum component increments squared
                    squared = np.sum(increment**2, axis=-1)
                else:
                    squared = increment**2
                moments.append(float(np.mean(squared)))
            products[quantity] = {"pair_second_moment": moments}
        # TODO: bin by separation/direction and add uncertainty estimates.
        return {"method": self.name, "spacecraft_pairs": [list(p) for p in pairs], "products": products}
