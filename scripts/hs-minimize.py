#!/usr/bin/env python3
r"""Find the recorded HS formation with minimum and maximum RMS pair separation within ±10 days.

Run from an installed source checkout (python -m pip install -e .):

    python scripts/hs-minimize.py --cdf-path /path/to/cdf-directory \
        --time 2030-06-01T00:00:00 --output outputs/hs-minimum.csv

Times without a timezone are UTC. Positions follow the mission reader's km
convention. All 36 pairs among nine spacecraft are used, without interpolation.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np

from hs_observatory.io.cdf_positions import FILL_VALUE_THRESHOLD, read_cdf_positions


WINDOW = np.timedelta64(10, "D")


def parse_time(value: str) -> np.datetime64:
    """Normalize an ISO date/time to UTC within the reader's nanosecond range."""
    try:
        instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
        instant = instant.replace(tzinfo=timezone.utc) if instant.tzinfo is None else instant
        utc = instant.astimezone(timezone.utc).replace(tzinfo=None)
        microseconds = np.datetime64(utc, "us")
        target = microseconds.astype("datetime64[ns]")
        if (
            np.isnat(target)
            or target.astype("datetime64[us]") != microseconds
            or target - WINDOW > target
            or target + WINDOW < target
        ):
            raise ValueError("time and its ±10-day window exceed the supported CDF epoch range")
        return target
    except (ValueError, OverflowError) as error:
        raise ValueError(f"Invalid ISO time {value!r}: {error}") from error


def utc_string(epoch: np.datetime64) -> str:
    return str(np.datetime_as_string(epoch, unit="ns", timezone="UTC"))


def find_minimum(cdf_path: Path, target: np.datetime64) -> dict[str, Any]:
    """Read real CDF records, retaining complete nine-spacecraft window samples."""
    epochs, positions, labels, sources = read_cdf_positions(cdf_path)
    epochs = np.asarray(epochs, dtype="datetime64[ns]")
    positions = np.asarray(positions, dtype=float)
    keep = [index for index, label in enumerate(labels) if label != "N/A"]
    names = [labels[index] for index in keep]
    if len(names) != 9 or len(set(names)) != 9:
        raise ValueError("Expected nine distinct spacecraft labels after ignoring N/A slots")
    if (
        epochs.ndim != 1
        or positions.shape != (len(epochs), len(labels), 3)
        or len(sources) != len(epochs)
    ):
        raise ValueError(
            "CDF epochs, positions, labels, and source filenames have inconsistent shapes"
        )
    valid_epoch = ~np.isnat(epochs)
    if not np.any(valid_epoch):
        raise ValueError("No valid CDF epochs")
    start, stop = target - WINDOW, target + WINDOW
    window = valid_epoch & (epochs >= start) & (epochs <= stop)
    indices = np.flatnonzero(window)
    if not len(indices):
        raise ValueError(f"No CDF samples in {utc_string(start)} through {utc_string(stop)}")
    # Sort stably even if a future reader returns files in a different order.
    indices = indices[np.argsort(epochs[indices], kind="stable")]
    points = positions[indices][:, keep]
    valid = np.all(np.isfinite(points) & (points > FILL_VALUE_THRESHOLD), axis=(1, 2))
    excluded = int(np.count_nonzero(~valid))
    window_epochs = epochs[indices]
    indices, points = indices[valid], points[valid]
    if not len(indices):
        raise ValueError(
            f"No valid nine-spacecraft samples in the ±10-day window ({excluded} excluded)"
        )
    first, second = np.triu_indices(9, k=1)
    differences = points[:, first] - points[:, second]
    rms = np.sqrt(np.mean(np.sum(differences**2, axis=2), axis=1))
    if not np.isfinite(rms).all():
        raise ValueError("CDF separations exceed the supported numerical range")
    best = int(np.argmin(rms))  # Chronological order breaks equal minima by earliest epoch.
    return {
        "epochs": epochs[indices],
        "rms": rms,
        "offset_days": (epochs[indices] - target) / np.timedelta64(1, "D"),
        "sources": [sources[index] for index in indices],
        "best": best,
        "maximum": int(np.argmax(rms)),  # Equal maxima also select the earliest epoch.
        "excluded": excluded,
        "invalid_epochs": int(np.count_nonzero(~valid_epoch)),
        "start": start,
        "stop": stop,
        "coverage_start": epochs[valid_epoch].min(),
        "coverage_stop": epochs[valid_epoch].max(),
        "window_start": window_epochs[0],
        "window_stop": window_epochs[-1],
    }


def write_csv(result: dict[str, Any], output: Path) -> None:
    if any(output.resolve() == Path(source).resolve() for source in result["sources"]):
        raise ValueError("CSV output must not overwrite a source CDF")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_utc", "offset_days", "rms_separation_km", "source_cdf"])
        for epoch, offset, rms, source in zip(
            result["epochs"],
            result["offset_days"],
            result["rms"],
            result["sources"],
        ):
            writer.writerow([utc_string(epoch), float(offset), float(rms), source])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--cdf-path", required=True, type=Path, help="Mission CDF file or directory"
    )
    parser.add_argument(
        "--time", required=True, help="Designated ISO timestamp (default timezone: UTC)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hs-minimize.csv"),
        help="Output CSV path (default: hs-minimize.csv)",
    )
    args = parser.parse_args(argv)
    try:
        target = parse_time(args.time)
        result = find_minimum(args.cdf_path.expanduser(), target)
        output = args.output.expanduser()
        write_csv(result, output)
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as error:
        parser.exit(1, f"hs-minimize: {error}\n")
    print(f"Designated UTC time: {utc_string(target)}")
    print(f"Requested window: {utc_string(result['start'])} through {utc_string(result['stop'])}")
    print(
        f"CDF coverage: {utc_string(result['coverage_start'])} through {utc_string(result['coverage_stop'])}"
    )
    print(
        f"Recorded window coverage: {utc_string(result['window_start'])} through {utc_string(result['window_stop'])}"
    )
    if result["coverage_start"] > result["start"] or result["coverage_stop"] < result["stop"]:
        print(
            "Warning: CDF coverage does not span the full requested ±10-day interval; "
            "the extrema cover available valid records only.",
            file=sys.stderr,
        )
    for label, index in (("Minimum", result["best"]), ("Maximum", result["maximum"])):
        print(f"{label} UTC time: {utc_string(result['epochs'][index])}")
        print(f"RMS separation (36 pairs): {result['rms'][index]:.12g} km")
        print(f"Offset from designated time: {result['offset_days'][index]:+.12g} days")
        print(f"Source CDF: {result['sources'][index]}")
    print(f"Window samples: {len(result['epochs'])} valid, {result['excluded']} excluded")
    if result["invalid_epochs"]:
        print(
            f"Records with invalid epochs (not assigned to the window): {result['invalid_epochs']}"
        )
    print(f"CSV: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
