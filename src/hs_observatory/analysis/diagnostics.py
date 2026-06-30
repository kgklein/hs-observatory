"""Basic numerical sanity checks for generated observations."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from hs_observatory.analysis.base import AnalysisMethod
from hs_observatory.analysis.registry import register_analysis
from hs_observatory.io.hdf5 import read_observations


def _stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values)
    # Vector products retain a component axis; everything else becomes scalar.
    axes = tuple(range(values.ndim - 1)) if values.ndim >= 4 and values.shape[-1] == 3 else None
    mean = np.mean(values, axis=axes)
    std = np.std(values, axis=axes)
    return {"mean": np.asarray(mean).tolist(), "std": np.asarray(std).tolist()}


@register_analysis
class DiagnosticsAnalysis(AnalysisMethod):
    name = "diagnostics"

    def run(self, input_path: str | Path) -> dict[str, Any]:
        data = read_observations(input_path)
        result: dict[str, Any] = {
            "method": self.name,
            "input": str(input_path),
            "observed": {q: _stats(data["observed"][q]) for q in ("n", "v", "B")},
        }
        if "truth" in data:
            residual_truth = {q: data["truth"][q] for q in ("n", "v", "B")}
            if (
                data["metadata"].get("velocity_observed_frame") == "spacecraft"
                and "v_spacecraft_frame" in data["truth"]
            ):
                residual_truth["v"] = data["truth"]["v_spacecraft_frame"]
            result["residual"] = {
                q: _stats(data["observed"][q] - residual_truth[q][None, ...])
                for q in ("n", "v", "B")
            }

        result["spacecraft_separation"] = {}
        for quantity in ("n", "v", "B"):
            r = data["spacecraft"]["r"][quantity]  # (spacecraft, time_quantity, component)
            separations = [
                np.linalg.norm(r[i] - r[j], axis=-1)
                for i, j in combinations(range(len(r)), 2)
            ]
            separation_array = np.asarray(separations)
            result["spacecraft_separation"][quantity] = {
                "units": "km",
                "n_pairs": len(separations),
                "minimum": float(np.min(separation_array)),
                "mean": float(np.mean(separation_array)),
                "maximum": float(np.max(separation_array)),
            }
        return result
