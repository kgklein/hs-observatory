"""Render total-separation examples, independent of the twelve-category witness.

Example::

    python scripts/plot_baseline_hybrid.py \
        --trajectory-path ../HelioSwarm/HS-RT/PhB_SRD8_0x78f --output-dir outputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hs_observatory.geometry_figure import (
    HYBRID_SCALE_PAIRS,
    HYBRID_VIEW,
    REFERENCE_EPOCH,
    load_science_hour_geometry,
    plot_baseline_configuration,
    save_science_hour_geometry,
    total_scale_examples,
)

import matplotlib.pyplot as plt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--trajectory-path", type=Path, required=True)
    parser.add_argument("--epoch", default=REFERENCE_EPOCH)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--elev", type=float, default=HYBRID_VIEW[0])
    parser.add_argument("--azim", type=float, default=HYBRID_VIEW[1])
    parser.add_argument("--alt-elev", type=float, default=25)
    parser.add_argument("--alt-azim", type=float, default=-45)
    for scale, pair in zip(("small", "intermediate", "large"), HYBRID_SCALE_PAIRS):
        parser.add_argument(f"--{scale}-pair", default="-".join(pair))
    args = parser.parse_args(argv)
    pairs = tuple(
        tuple(getattr(args, f"{scale}_pair").split("-"))
        for scale in ("small", "intermediate", "large")
    )
    try:
        record = load_science_hour_geometry(args.trajectory_path, epoch=args.epoch)
        examples = total_scale_examples(record, pairs)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        paths = save_science_hour_geometry(
            record,
            args.output_dir / "science_hour_geometry_hybrid_v01",
            output_format="both",
            view=(args.elev, args.azim),
            zoom_inset=True,
            baseline_mode="hybrid",
            scale_pairs=pairs,
        )
        for name, view in (
            ("baseline_hybrid_v01", (args.elev, args.azim)),
            ("baseline_hybrid_v01_view_B", (args.alt_elev, args.alt_azim)),
        ):
            figure, _ = plot_baseline_configuration(
                record,
                view=view,
                zoom_inset=True,
                baseline_mode="hybrid",
                scale_pairs=pairs,
            )
            try:
                path = args.output_dir / f"{name}.png"
                figure.savefig(path, dpi=300, facecolor="white")
                paths.append(path)
            finally:
                plt.close(figure)
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                **record.diagnostics(),
                "visualization": {
                    "meaning": "Total-separation scale examples; not selected as a twelve-category witness",
                    "examples": examples,
                    "camera": [args.elev, args.azim],
                    "alternate_camera": [args.alt_elev, args.alt_azim],
                },
                "outputs": [str(path) for path in paths],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
