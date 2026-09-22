"""Regenerate geometry figures from saved observations without rerunning generation.

Run from an installed source checkout, e.g. ``python -m pip install -e .``.
Relative CDF paths are resolved from the working directory, as during generation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import yaml

from hs_observatory.generate import _geometry_context_trajectory
from hs_observatory.visualization import geometry_summary_paths, plot_geometry_summary

# Import the project plotting code first so its headless backend is configured.
import matplotlib.pyplot as plt


def regenerate_geometry(
    source: Path,
    *,
    variants: tuple[str, ...] = ("polyhedra", "formation"),
    output_dir: Path | None = None,
    dpi: int = 150,
    cdf_path: Path | None = None,
    no_context: bool = False,
) -> list[Path]:
    """Read nominal positions and metadata, then save selected PDF/PNG views."""
    source = Path(source).expanduser()
    with h5py.File(source, "r") as handle:
        attrs = handle["metadata"].attrs
        metadata = json.loads(attrs.get("spacecraft_source", "{}"))
        config = yaml.safe_load(attrs.get("config_yaml", "{}")) or {}
        if not isinstance(metadata, dict) or not isinstance(config, dict):
            raise ValueError("Observation spacecraft_source/config_yaml must contain mappings")
        # Match generation's choice of the longest native time grid. No truth or
        # Monte Carlo measurement arrays are loaded.
        quantity = max(("n", "v", "B"), key=lambda name: len(handle[f"time/{name}"]))
        position = handle[f"spacecraft/r/{quantity}"][:]
        time = handle[f"time/{quantity}"][:]
        ids = handle["spacecraft/id"].asstr()[:].tolist()

    spacecraft_config = dict(config.get("spacecraft") or {})
    if cdf_path is not None:
        spacecraft_config["cdf_path"] = str(Path(cdf_path).expanduser())
    elif not (spacecraft_config.get("cdf_path") or spacecraft_config.get("cdf_directory")):
        spacecraft_config["cdf_path"] = metadata.get("cdf_path")
    # Keep the saved selection epoch when reloading external trajectory context.
    spacecraft_config["configuration_time"] = (
        metadata.get("selected_time")
        or metadata.get("requested_time")
        or spacecraft_config.get("configuration_time")
    )
    context = None
    if not no_context:
        try:
            context = _geometry_context_trajectory(spacecraft_config, metadata)
        except (OSError, ValueError, RuntimeError) as error:
            raise ValueError(
                f"Could not load orbit context: {error}. Use --cdf-path to locate the "
                "mission CDF inputs, or --no-context to plot only saved positions."
            ) from error
    if context is None:
        print(
            "Orbit context uses saved positions only; no external trajectory loaded.",
            file=sys.stderr,
        )

    paths = []
    for variant in dict.fromkeys(variants):
        figure, _ = plot_geometry_summary(
            position,
            ids,
            time=time,
            metadata=metadata,
            context_trajectory=context,
            variant=variant,
        )
        try:
            pdf, png = geometry_summary_paths(source, metadata, variant=variant)
            if output_dir is not None:
                directory = Path(output_dir).expanduser()
                pdf, png = directory / pdf.name, directory / png.name
            pdf.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(pdf)
            figure.savefig(png, dpi=dpi)
        finally:
            plt.close(figure)
        paths.extend((pdf, png))
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=Path, help="Saved observation HDF5 product")
    parser.add_argument(
        "--variant",
        choices=("polyhedra", "formation", "summary"),
        action="append",
        help="View to render; repeat to select multiple (default: polyhedra and formation)",
    )
    parser.add_argument("--output-dir", type=Path, help="Override the geometry_overviews directory")
    parser.add_argument("--dpi", type=int, default=150, help="PNG resolution (default: 150)")
    context = parser.add_mutually_exclusive_group()
    context.add_argument("--cdf-path", type=Path, help="Override the saved mission CDF path")
    context.add_argument("--no-context", action="store_true", help="Use saved positions only")
    args = parser.parse_args(argv)
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    try:
        paths = regenerate_geometry(
            args.observations,
            variants=tuple(args.variant or ("polyhedra", "formation")),
            output_dir=args.output_dir,
            dpi=args.dpi,
            cdf_path=args.cdf_path,
            no_context=args.no_context,
        )
    except (OSError, ValueError, KeyError, RuntimeError, yaml.YAMLError) as error:
        parser.exit(1, f"Could not regenerate geometry figures: {error}\n")
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
