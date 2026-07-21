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


def plot_figure2_main(argv: list[str] | None = None) -> int:
    from hs_observatory.visualization import plot_figure2_timeseries

    parser = argparse.ArgumentParser(
        description="Plot the two-column paper Figure 2 observatory time-series summary"
    )
    parser.add_argument("--malara", required=True, help="Malara-model observation HDF5 file")
    parser.add_argument(
        "--simulation",
        required=True,
        help="turbulent-simulation observation HDF5 file",
    )
    parser.add_argument("--output", required=True, help="output figure path, e.g. figure2.png")
    parser.add_argument(
        "--member-index",
        type=int,
        default=0,
        help="observed Monte Carlo realization to plot; default: 0",
    )
    parser.add_argument(
        "--input-units",
        choices=("si", "repo"),
        default="si",
        help="units stored in input HDF5 variables; default: si",
    )
    parser.add_argument(
        "--velocity-frame",
        choices=("plasma", "spacecraft"),
        default="plasma",
        help="common velocity frame for truth, measured, and z+ panels; default: plasma",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=None,
        help="relative start time in seconds for the plotted x-axis",
    )
    parser.add_argument(
        "--stop-time",
        type=float,
        default=None,
        help="relative stop time in seconds for the plotted x-axis",
    )
    parser.add_argument("--malara-title", default=None)
    parser.add_argument("--simulation-title", default=None)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)
    output = plot_figure2_timeseries(
        args.malara,
        args.simulation,
        args.output,
        member_index=args.member_index,
        **({} if args.malara_title is None else {"malara_title": args.malara_title}),
        **(
            {}
            if args.simulation_title is None
            else {"simulation_title": args.simulation_title}
        ),
        input_units=args.input_units,
        velocity_frame=args.velocity_frame,
        start_time_s=args.start_time,
        stop_time_s=args.stop_time,
        dpi=args.dpi,
    )
    print(output)
    return 0
