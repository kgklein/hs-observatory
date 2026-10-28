"""Render purple total-separation examples with tick and physical-scale-bar insets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hs_observatory.geometry_figure import (
    PURPLE_SCALE_COLORS,
    PURPLE_SCALE_PAIRS,
    PURPLE_VIEW,
    REFERENCE_EPOCH,
    intermediate_projection_diagnostics,
    load_science_hour_geometry,
    plot_baseline_configuration,
    plot_science_hour_geometry,
    total_scale_examples,
)

import matplotlib.pyplot as plt


def _save(figure, base: Path, formats):
    paths = []
    try:
        with plt.rc_context({"pdf.fonttype": 42}):
            for extension in formats:
                path = base.with_suffix(f".{extension}")
                figure.savefig(path, dpi=300, facecolor="white")
                paths.append(str(path))
    finally:
        plt.close(figure)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-path", type=Path, required=True)
    parser.add_argument("--epoch", default=REFERENCE_EPOCH)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--elev", type=float, default=PURPLE_VIEW[0])
    parser.add_argument("--azim", type=float, default=PURPLE_VIEW[1])
    parser.add_argument("--intermediate-pair", default="-".join(PURPLE_SCALE_PAIRS[1]))
    args = parser.parse_args(argv)
    pairs = (PURPLE_SCALE_PAIRS[0], tuple(args.intermediate_pair.split("-")), PURPLE_SCALE_PAIRS[2])
    try:
        record = load_science_hour_geometry(args.trajectory_path, epoch=args.epoch)
        examples = total_scale_examples(record, pairs, colors=PURPLE_SCALE_COLORS)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        diagnostics = None
        for variant in ("ticks", "scalebar"):
            options = dict(
                view=(args.elev, args.azim),
                zoom_inset=True,
                baseline_mode="hybrid",
                scale_pairs=pairs,
                baseline_style="purple",
                inset_variant=variant,
            )
            figure, axes = plot_science_hour_geometry(record, **options)
            if diagnostics is None:
                diagnostics = intermediate_projection_diagnostics(record, axes[2])
            paths.extend(
                _save(
                    figure, args.output_dir / f"science_hour_geometry_v03_{variant}", ("pdf", "png")
                )
            )
            figure, _ = plot_baseline_configuration(record, **options)
            paths.extend(_save(figure, args.output_dir / f"baseline_v03_{variant}", ("png",)))
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                **record.diagnostics(),
                "visualization": {
                    "meaning": "Illustrative total-length categories, not a twelve-category witness selection",
                    "camera": [args.elev, args.azim],
                    "examples": examples,
                    "intermediate_candidates_full_panel": diagnostics,
                },
                "outputs": paths,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
