#!/usr/bin/env python3
"""Refit saved structure functions while progressively trimming either scale end.

Run with .venv/bin/python scripts/structure_function_fit_sweep.py.
Relative config, input, and output paths are resolved from the project root.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
matplotlib.use("Agg")

from hs_observatory.analysis.structure_function import (  # noqa: E402
    _ensemble_interval,
    _fit_scaling_exponents,
    plot_structure_function_overlay,
)


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_cases(config: dict, quantity: str, max_order: int | None) -> tuple[list, np.ndarray]:
    """Load each input once and validate every scale grid before writing outputs."""
    entries = config.get("cases")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("config requires a nonempty cases mapping")
    cases = []
    reference = None
    p_values = set()
    for run_id, entry in entries.items():
        p_value = float(entry["p_value"])
        if not np.isfinite(p_value) or p_value in p_values:
            raise ValueError(f"case {run_id}: p_value must be finite and unique")
        p_values.add(p_value)
        with project_path(entry["path"]).open(encoding="utf-8") as stream:
            result = json.load(stream)
        for source in ("truth", "observed"):
            product = result["products"][source]
            centers = np.asarray(product["lambda_bin_centers_km"], dtype=float)
            if (
                centers.ndim != 1
                or centers.size < 2
                or not np.all(np.isfinite(centers) & (centers > 0))
                or not np.all(np.diff(centers) > 0)
            ):
                raise ValueError(f"case {run_id} {source}: invalid scale-bin centers")
            if reference is None:
                reference = centers
            elif not np.array_equal(centers, reference):
                raise ValueError(f"case {run_id} {source}: scale-bin centers differ")
            values = np.asarray(product[quantity]["S_n"], dtype=float)
            counts = np.asarray(product["counts"], dtype=float)
            orders = product["orders"]
            if (
                values.ndim != 3
                or values.shape[0] < 1
                or values.shape[1:] != (len(orders), centers.size)
                or counts.shape != (values.shape[0], centers.size)
                or (source == "truth" and values.shape[0] != 1)
            ):
                raise ValueError(f"case {run_id} {source}: incompatible product shapes")
        truth_orders = result["products"]["truth"]["orders"]
        mc_orders = result["products"]["observed"]["orders"]
        selected = [
            order for order in truth_orders
            if order in mc_orders and (max_order is None or order <= max_order)
        ]
        if not selected:
            raise ValueError(f"case {run_id}: no requested orders in common")
        cases.append((str(run_id), p_value, result, selected))
    return cases, reference


def fit_product(product: dict, quantity: str, bounds: tuple, min_bins: int) -> dict:
    return _fit_scaling_exponents(
        np.asarray(product[quantity]["S_n"], dtype=float),
        np.asarray(product["counts"], dtype=float),
        np.asarray(product["lambda_bin_centers_km"], dtype=float),
        product["orders"],
        {"sweep": list(bounds)},
        min_bins,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="analysis/malara-prod-combined_sf-publication.yaml")
    parser.add_argument("--output-dir", default="outputs/malara-prod-combined_sf_fit_sweep")
    parser.add_argument("--min-fit-bins", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        with project_path(args.config).open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        if not isinstance(config, dict):
            raise ValueError("config must be a YAML mapping")
        quantity = config.get("quantity", "z_plus")
        max_order = config.get("max_order", 6)
        cases, centers = load_cases(config, quantity, max_order)
        if not 2 <= args.min_fit_bins <= centers.size:
            raise ValueError(f"min-fit-bins must be between 2 and {centers.size}")
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        parser.error(str(exc))

    ranges = [("all", 0, 0)]
    ranges.extend((f"drop_large_{k}", 0, k) for k in range(1, len(centers) - args.min_fit_bins + 1))
    ranges.extend((f"drop_small_{k}", k, 0) for k in range(1, len(centers) - args.min_fit_bins + 1))
    output_dir = project_path(args.output_dir)
    rows = []
    results_by_p = {p: result for _, p, result, _ in cases}
    for name, drop_small, drop_large in ranges:
        retained = centers[drop_small:len(centers) - drop_large]
        bounds = (float(retained[0]), float(retained[-1]))
        missing_truth = missing_mc = 0
        for run_id, p_value, result, orders in cases:
            truth = result["products"]["truth"]
            observed = result["products"]["observed"]
            truth_fit = fit_product(truth, quantity, bounds, args.min_fit_bins)
            mc_fit = fit_product(observed, quantity, bounds, args.min_fit_bins)
            for order in orders:
                ti = truth["orders"].index(order)
                mi = observed["orders"].index(order)
                truth_xi = truth_fit["raw"][0, 0, ti]
                mc_xi = mc_fit["raw"][:, 0, mi]
                lower, median, upper = _ensemble_interval(mc_xi)
                mc_bins = mc_fit["bins_used"][:, 0, mi]
                finite_mc = int(np.count_nonzero(np.isfinite(mc_xi)))
                missing_truth += int(not np.isfinite(truth_xi))
                missing_mc += len(mc_xi) - finite_mc
                rows.append({
                    "range": name,
                    "drop_small": drop_small,
                    "drop_large": drop_large,
                    "retained_bins": len(retained),
                    "lambda_min_km": bounds[0],
                    "lambda_max_km": bounds[1],
                    "run_id": run_id,
                    "p_value": p_value,
                    "quantity": quantity,
                    "order": order,
                    "truth_exponent": float(truth_xi),
                    "truth_fit_bins": int(truth_fit["bins_used"][0, 0, ti]),
                    "mc_median": float(median),
                    "mc_p05": float(lower),
                    "mc_p95": float(upper),
                    "mc_finite_fits": finite_mc,
                    "mc_fit_bins_min": int(mc_bins.min()),
                    "mc_fit_bins_max": int(mc_bins.max()),
                })
        figure_path = plot_structure_function_overlay(
            results_by_p,
            output_dir / f"{name}.png",
            quantity=quantity,
            max_order=max_order,
            fit_lambda_range_km=bounds,
            min_fit_bins=args.min_fit_bins,
            show_p_model=config.get("show_p_model", True),
            dpi=int(config.get("dpi", 300)),
        )
        print(
            f"{figure_path.name}: {len(retained)} bins, {bounds[0]:.6g}–{bounds[1]:.6g} km; "
            f"missing fits: truth={missing_truth}, MC={missing_mc}"
        )
    csv_path = output_dir / "fits.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(ranges)} figures and {len(rows)} fit rows to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
