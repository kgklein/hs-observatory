#!/usr/bin/env python3
"""Validate saved HIVE latent measurement errors without regenerating observations."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import yaml

from measurement_error_validation.reporting import write_reports
from measurement_error_validation.runner import InvalidInput, Validator


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mc", default="all", help="all or comma-separated zero-based MC indices")
    parser.add_argument(
        "--spacecraft", default="all", help="all or comma-separated stored spacecraft IDs"
    )
    parser.add_argument("--confidence", type=float, default=0.99)
    parser.add_argument("--chunk-times", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args(argv)
    if not np.isfinite(args.confidence) or not 0 < args.confidence < 1:
        parser.error("confidence must be strictly between zero and one")
    if args.chunk_times < 1 or args.seed < 0:
        parser.error("chunk-times must be positive; seed must be nonnegative")
    sources = {args.input.resolve(), args.config.resolve()}
    if any(
        (args.output_dir / name).resolve() in sources
        for name in ("validation_results.json", "validation_results.csv", "validation_report.md")
    ):
        parser.error("output paths must not replace inputs")
    return args


def revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def main(argv=None):
    args = arguments(argv)
    provenance = dict(
        input=str(args.input.resolve()),
        config=str(args.config.resolve()),
        created_utc=datetime.now(timezone.utc).isoformat(),
        source_revision=revision(),
        selections=dict(mc=args.mc, spacecraft=args.spacecraft),
        confidence=args.confidence,
        seed=args.seed,
        chunk_times=args.chunk_times,
        estimator=dict(
            window="periodic Hann",
            overlap_fraction=0.5,
            detrend=False,
            broadband_segment_limit=8192,
            low_frequency_segment="full record",
            correction="Holm within input",
            approximate_path_minimum=30,
        ),
        dependencies={
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "h5py", "PyYAML", "matplotlib")
        },
    )
    payload = dict(
        status="FAIL",
        exit_code=2,
        provenance=provenance,
        results=[],
        discrepancies=[],
        configuration_checked=False,
    )
    validator = None
    try:
        try:
            supplied = yaml.safe_load(args.config.read_text())
        except (OSError, yaml.YAMLError) as exc:
            raise InvalidInput(f"cannot read supplied YAML: {exc}") from exc
        if not isinstance(supplied, dict):
            raise InvalidInput("supplied YAML must be a mapping")
        provenance["supplied_output_path"] = supplied.get("output", {}).get("path")
        provenance["output_path_note"] = (
            "Input location is explicit; output-path relocation is informational."
        )
        initial = args.input.stat()
        provenance.update(size_bytes=initial.st_size, mtime_ns=initial.st_mtime_ns)
        with h5py.File(args.input, "r") as handle:
            validator = Validator(handle, supplied, args)
            provenance.update(
                embedded_configuration=validator.cfg,
                embedded_metadata=validator.meta,
                selected_mc=validator.mcs,
                selected_spacecraft=[validator.ids[i] for i in validator.scs],
                n_paths=validator.k,
                time_counts={q: len(t) for q, t in validator.times.items()},
            )
            b_time = validator.times["B"]
            n, dt = len(b_time), float(b_time[1] - b_time[0])
            lengths = sorted(set((min(8192, n), n)))
            provenance["spectral_resolution"] = dict(
                cadence_s=dt,
                nyquist_Hz=1 / (2 * dt),
                fundamental_Hz=1 / (n * dt),
                segments=[
                    dict(
                        length=length,
                        resolution_Hz=1 / (length * dt),
                        overlap=length // 2,
                        segments_per_path=1 + (n - length) // (length - length // 2),
                    )
                    for length in lengths
                ],
            )
            payload.update(
                discrepancies=validator.discrepancies,
                configuration_checked=True,
                results=validator.run(),
            )
        final = args.input.stat()
        if (initial.st_size, initial.st_mtime_ns) != (final.st_size, final.st_mtime_ns):
            raise OSError("input changed during reading (possibly still downloading)")
        rows = payload["results"]
        failures = any(r["status"] == "FAIL" for r in rows)
        gaps = any(
            r["primary"]
            and r["status"] in ("SKIPPED", "INSUFFICIENT_DATA")
            and validator.cfg.get(r["component"], {}).get("enabled", True)
            for r in rows
        )
        payload["exit_code"] = int(bool(failures or gaps or payload["discrepancies"]))
        payload["status"] = (
            "FAIL"
            if failures or payload["discrepancies"]
            else ("INSUFFICIENT_DATA" if gaps else "PASS")
        )
    except (OSError, RuntimeError) as exc:
        payload.update(status="INCOMPLETE_INPUT", exit_code=2)
        if validator is not None:
            payload["results"] = validator.results.finalize()
        payload["results"].append(
            dict(
                component="input",
                statistic="readability",
                expected=None,
                measured=None,
                status="INCOMPLETE_INPUT",
                note=str(exc),
                primary=True,
            )
        )
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        payload.update(status="FAIL", exit_code=2)
        if validator is not None:
            payload["results"] = validator.results.finalize()
        payload["results"].append(
            dict(
                component="input",
                statistic="validity",
                expected=None,
                measured=None,
                status="FAIL",
                note=str(exc),
                primary=True,
            )
        )
    write_reports(args.output_dir, payload, [] if validator is None else validator.plots)
    print(f"{payload['status']}: {args.output_dir / 'validation_report.md'}")
    return payload["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
