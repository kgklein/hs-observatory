"""Console entry points."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path


def _run(description: str, workflow: Callable[[str], Path], argv: list[str] | None) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("config", help="YAML configuration file")
    args = parser.parse_args(argv)
    output = workflow(args.config)
    print(output)
    return 0


def generate_main(argv: list[str] | None = None) -> int:
    from hs_observatory.generate import generate_from_config

    parser = argparse.ArgumentParser(
        description="Generate synthetic HelioSwarm observations"
    )
    parser.add_argument("config", help="YAML configuration file")
    parser.add_argument(
        "--truth-velocity-frame",
        choices=("spacecraft", "plasma"),
        help="override the velocity frame used only in the truth visualization",
    )
    args = parser.parse_args(argv)
    output = generate_from_config(
        args.config,
        truth_velocity_frame=args.truth_velocity_frame,
    )
    print(output)
    return 0


def analyze_main(argv: list[str] | None = None) -> int:
    from hs_observatory.analyze import analyze_from_config

    return _run("Analyze synthetic HelioSwarm observations", analyze_from_config, argv)
