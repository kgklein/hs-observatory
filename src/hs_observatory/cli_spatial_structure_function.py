"""Direct spatial sampling and independent cached post-processing commands.

Module fallback: python -m hs_observatory.cli_spatial_structure_function sample|plot ...
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

from hs_observatory.analysis.spatial_structure_function import (
    DEFAULT_WINDOWS,
    SpatialStructureFunctionAnalysis,
    postprocess_spatial_result,
    read_spatial_result,
)
from hs_observatory.config import load_analysis_config


def _fit_options(parser):
    parser.add_argument("--fit-lambda-range", nargs=2, type=float, metavar=("MIN_KM", "MAX_KM"))
    parser.add_argument(
        "--fit-window", nargs=3, action="append", metavar=("NAME", "MIN_KM", "MAX_KM")
    )
    parser.add_argument("--local-slope-method", choices=("three-point", "adjacent"))
    parser.add_argument("--show-cascade-scales", action="store_true", default=None)
    parser.add_argument("--comparison-config", help="existing virtual-HS overlay YAML")
    parser.add_argument("--output-prefix")


def _windows(args, current):
    windows = dict(current)
    if args.fit_lambda_range:
        windows["candidate"] = args.fit_lambda_range
    for name, lo, hi in args.fit_window or []:
        windows[name] = [float(lo), float(hi)]
    return windows


def sample_main(argv=None):
    parser = argparse.ArgumentParser(description="Sample raw periodic Malara spatial increments")
    parser.add_argument("config", help="analysis YAML (input.path is the generation YAML)")
    parser.add_argument("--field-config")
    parser.add_argument("--p", nargs="+", type=float, dest="p_values")
    for flag in ("n-pairs", "n-batches", "seed", "n-workers", "chunk-size", "n-separations"):
        parser.add_argument(f"--{flag}", type=int)
    parser.add_argument("--backend", choices=("cached", "parallel", "direct", "chunked"))
    parser.add_argument("--lambda-min-km", type=float)
    parser.add_argument("--lambda-max-km", type=float)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--lambda-centers-km", nargs="+", type=float)
    group.add_argument("--centers-from-hs")
    parser.add_argument(
        "--smoke", action="store_true", help="mark deliberately tiny diagnostic products"
    )
    _fit_options(parser)
    args = parser.parse_args(argv)
    try:
        loaded = load_analysis_config(args.config)
        config = loaded.data
        if config["analysis"]["method"] != "spatial_structure_function":
            raise ValueError("analysis.method must be spatial_structure_function")
        options = dict(config["analysis"].get("options", {}))
        options.update(config.get("spatial_structure_function", {}))
        for key in (
            "p_values",
            "n_pairs",
            "n_batches",
            "seed",
            "n_workers",
            "chunk_size",
            "n_separations",
            "backend",
            "lambda_min_km",
            "lambda_max_km",
            "local_slope_method",
            "comparison_config",
        ):
            value = getattr(args, key)
            if value is not None:
                options[key] = value
        if args.lambda_centers_km is not None or args.centers_from_hs:
            options.pop("lambda_centers_km", None)
            options.pop("centers_from_hs", None)
            options[
                "lambda_centers_km" if args.lambda_centers_km is not None else "centers_from_hs"
            ] = (
                args.lambda_centers_km
                if args.lambda_centers_km is not None
                else args.centers_from_hs
            )
        elif any(
            v is not None for v in (args.n_separations, args.lambda_min_km, args.lambda_max_km)
        ):
            options.pop("lambda_centers_km", None)
            options.pop("centers_from_hs", None)
        options["scale_ranges_km"] = _windows(args, options.get("scale_ranges_km", DEFAULT_WINDOWS))
        if args.show_cascade_scales:
            options.setdefault("plot", {})["show_cascade_scales"] = True
        if args.smoke:
            options["smoke"] = True
        method = SpatialStructureFunctionAnalysis(options)
        result = method.run(args.field_config or config["input"]["path"])
        output = (
            Path(f"{args.output_prefix}.h5")
            if args.output_prefix
            else Path(config["output"]["path"])
        )
        print(method.write_result(result, output))
    except (ValueError, TypeError, OSError, KeyError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    return 0


def plot_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Refit and plot saved direct spatial moments; no field evaluation"
    )
    parser.add_argument("product")
    _fit_options(parser)
    args = parser.parse_args(argv)
    try:
        current = read_spatial_result(args.product)["config"]["scale_ranges_km"]
        output = postprocess_spatial_result(
            args.product,
            output_prefix=args.output_prefix,
            windows=_windows(args, current),
            method=args.local_slope_method,
            show_scales=args.show_cascade_scales,
            comparison_config=args.comparison_config,
        )
        print(output)
    except (ValueError, TypeError, OSError, KeyError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in {"sample", "plot"}:
        raise SystemExit(
            "usage: python -m hs_observatory.cli_spatial_structure_function sample|plot ..."
        )
    raise SystemExit((sample_main if sys.argv[1] == "sample" else plot_main)(sys.argv[2:]))
