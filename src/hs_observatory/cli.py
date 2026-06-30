"""Console entry points."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from hs_observatory.analyze import analyze_from_config
from hs_observatory.generate import generate_from_config


def _run(description: str, workflow: Callable[[str], Path], argv: list[str] | None) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("config", help="YAML configuration file")
    args = parser.parse_args(argv)
    output = workflow(args.config)
    print(output)
    return 0


def generate_main(argv: list[str] | None = None) -> int:
    return _run("Generate synthetic HelioSwarm observations", generate_from_config, argv)


def analyze_main(argv: list[str] | None = None) -> int:
    return _run("Analyze synthetic HelioSwarm observations", analyze_from_config, argv)
