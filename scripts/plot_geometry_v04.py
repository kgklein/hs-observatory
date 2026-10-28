"""Compare teal and six-level spacecraft-count geometry figures without recomputing geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hs_observatory.geometry_figure import (
    PURPLE_SCALE_PAIRS,
    PURPLE_VIEW,
    REFERENCE_EPOCH,
    load_science_hour_geometry,
    plot_science_hour_geometry,
)

import matplotlib.pyplot as plt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-path", type=Path, required=True)
    parser.add_argument("--epoch", default=REFERENCE_EPOCH)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--panel-b-previews", action="store_true")
    args = parser.parse_args(argv)
    try:
        record = load_science_hour_geometry(args.trajectory_path, epoch=args.epoch)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for coloring in ("teal", "nsc"):
            figure, axes = plot_science_hour_geometry(
                record,
                view=PURPLE_VIEW,
                zoom_inset=True,
                baseline_mode="hybrid",
                scale_pairs=PURPLE_SCALE_PAIRS,
                baseline_style="purple",
                inset_variant="scalebar",
                layout_style="v04",
                polyhedron_coloring=coloring,
            )
            try:
                with plt.rc_context({"pdf.fonttype": 42}):
                    for extension in ("pdf", "png"):
                        path = args.output_dir / f"science_hour_geometry_v04_{coloring}.{extension}"
                        figure.savefig(path, dpi=300, facecolor="white")
                        paths.append(str(path))
                    if args.panel_b_previews:
                        figure.canvas.draw()
                        bounds = axes[1].get_tightbbox(figure.canvas.get_renderer())
                        crop = bounds.transformed(figure.dpi_scale_trans.inverted()).padded(0.04)
                        path = args.output_dir / f"polyhedra_v04_{coloring}.png"
                        figure.savefig(path, dpi=300, facecolor="white", bbox_inches=crop)
                        paths.append(str(path))
            finally:
                plt.close(figure)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                **record.diagnostics(),
                "outputs": paths,
                "visualization": {
                    "camera": PURPLE_VIEW,
                    "baseline_pairs": PURPLE_SCALE_PAIRS,
                    "membership_colormap": "cividis, six discrete levels",
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
