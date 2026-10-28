"""Plot nominal science-hour prerequisites directly from mission CDF positions.

Example (from an installed source checkout)::

    python scripts/plot_science_hour_geometry.py \
        --trajectory-path ../HelioSwarm/HS-RT/PhB_SRD8_0x78f \
        --epoch 2030-08-25T11:50:35.801000000 \
        --output outputs/science_hour_geometry --format both
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hs_observatory.geometry_figure import (
    DEFAULT_BASELINE_VIEW,
    load_science_hour_geometry,
    save_baseline_previews,
    save_science_hour_geometry,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--trajectory-path", type=Path, required=True, help="Mission CDF file or directory"
    )
    parser.add_argument("--epoch", required=True, help="Requested native epoch (ISO UTC)")
    parser.add_argument(
        "--output", type=Path, required=True, help="Output stem or PDF/PNG filename"
    )
    parser.add_argument("--format", choices=("pdf", "png", "both"), default="both")
    parser.add_argument("--epoch-tolerance-s", type=float, default=0.001)
    parser.add_argument("--elev", type=float, default=DEFAULT_BASELINE_VIEW[0])
    parser.add_argument("--azim", type=float, default=DEFAULT_BASELINE_VIEW[1])
    parser.add_argument(
        "--zoom-inset", action="store_true", help="Add an equal-scale near-hub X-Y inset"
    )
    parser.add_argument(
        "--baseline-previews",
        action="store_true",
        help="Also save camera trials A/B/C and baseline previews with/without inset",
    )
    parser.add_argument(
        "--witness",
        action="append",
        metavar="H-N1",
        help="Witness pair; repeat to replace the default four-pair reference witness",
    )
    args = parser.parse_args(argv)
    options = {"epoch": args.epoch, "epoch_tolerance_s": args.epoch_tolerance_s}
    if args.witness is not None:
        pairs = [tuple(value.split("-")) for value in args.witness]
        if any(len(pair) != 2 for pair in pairs):
            parser.error("--witness must be a pair such as H-N1")
        options["witness_pairs"] = pairs
    try:
        record = load_science_hour_geometry(args.trajectory_path, **options)
        paths = save_science_hour_geometry(
            record,
            args.output,
            output_format=args.format,
            view=(args.elev, args.azim),
            zoom_inset=args.zoom_inset,
        )
        if args.baseline_previews:
            paths.extend(save_baseline_previews(record, args.output))
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps({**record.diagnostics(), "outputs": [str(path) for path in paths]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
