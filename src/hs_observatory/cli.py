"""Console entry points."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
import math


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


def plot_structure_function_summary_main(argv: list[str] | None = None) -> int:
    """Plot a publication-style structure-function summary from analysis JSON."""

    from hs_observatory.analysis.structure_function import plot_structure_function_summary

    parser = argparse.ArgumentParser(
        description="Plot sampled truth and MC structure-function ensemble statistics"
    )
    parser.add_argument("result", help="structure-function analysis JSON")
    parser.add_argument("--output", required=True, help="output figure path")
    parser.add_argument("--p-value", type=float, default=0.9, help="Malara p-model value")
    parser.add_argument(
        "--quantity", choices=("z_plus", "z_minus"), default="z_plus"
    )
    parser.add_argument("--run-color", default=None, help="Matplotlib color for this run")
    parser.add_argument("--max-order", type=int, default=6)
    parser.add_argument("--fit-range", default=None, help="named exponent fit range")
    parser.add_argument(
        "--pair-counts", action="store_true",
        help="add a fourth panel showing the number of baseline pairs per bin",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)
    output = plot_structure_function_summary(
        args.result,
        p_value=args.p_value,
        output_path=args.output,
        quantity=args.quantity,
        run_color=args.run_color,
        max_order=args.max_order,
        fit_range=args.fit_range,
        show_pair_counts=args.pair_counts,
        dpi=args.dpi,
    )
    print(output)
    return 0


def plot_increment_pdf_publication_main(argv: list[str] | None = None) -> int:
    """Plot three Malara cases from saved increment-PDF analysis HDF5 files."""

    from hs_observatory.analysis.elsasser_increment_pdf_static import (
        plot_increment_pdf_publication,
    )

    parser = argparse.ArgumentParser(
        description="Plot sampled truth and MC normalized Elsasser increment PDFs"
    )
    parser.add_argument(
        "--result", action="append", required=True, metavar="P=HDF5",
        help="saved result for one p case; repeat for p=0.9, 0.7, and 0.5",
    )
    parser.add_argument("--output", required=True, help="output publication figure path")
    parser.add_argument(
        "--lags", type=float, nargs=3, default=None, metavar=("SMALL", "MEDIUM", "LARGE"),
        help="three requested perpendicular lags in km; nearest saved bins are used",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)
    results_by_p = {}
    for specification in args.result:
        try:
            p_text, result_path = specification.split("=", 1)
            p_value = float(p_text)
        except (TypeError, ValueError) as error:
            parser.error(f"invalid --result {specification!r}; expected P=HDF5 ({error})")
        if p_value in results_by_p:
            parser.error(f"duplicate --result for p={p_value:g}")
        results_by_p[p_value] = result_path
    output = plot_increment_pdf_publication(
        results_by_p,
        selected_lags=args.lags,
        quantity="z_plus",
        output_path=args.output,
        dpi=args.dpi,
    )
    print(output)
    return 0


def plot_increment_pdf_wasserstein_main(argv: list[str] | None = None) -> int:
    """Plot Gaussian-normalized W1 diagnostics from saved increment-PDF products."""

    from hs_observatory.analysis.increment_pdf_wasserstein import (
        plot_increment_pdf_wasserstein_from_config,
    )

    parser = argparse.ArgumentParser(
        description="Plot PDF comparisons and Gaussian-normalized Wasserstein diagnostics"
    )
    parser.add_argument("config", help="lightweight publication plotting YAML")
    args = parser.parse_args(argv)
    output, diagnostics = plot_increment_pdf_wasserstein_from_config(args.config)
    print(output)
    print("case,label,lag_km,DW_G_p25,DW_G_p50,DW_G_p75,DW_G_median_pdf")
    for case, entry in diagnostics["cases"].items():
        values = [
            case,
            entry["label"],
            f"{entry['selected_lag_km']:.8g}",
            *("nan" if not math.isfinite(entry[key]) else f"{entry[key]:.8g}" for key in (
                "DW_G_p25", "DW_G_p50", "DW_G_p75", "DW_G_median_pdf"
            )),
        ]
        print(",".join(values))
    return 0


def plot_magnetic_gradient_main(argv: list[str] | None = None) -> int:
    """Plot a saved magnetic-gradient product without recalculation."""

    from hs_observatory.analysis.magnetic_gradient import plot_magnetic_gradient_product

    parser = argparse.ArgumentParser(
        description="Plot a calculated HelioSwarm magnetic-gradient HDF5 product"
    )
    parser.add_argument("product", help="magnetic-gradient analysis HDF5 file")
    parser.add_argument("--output", required=True, help="output figure path")
    parser.add_argument(
        "--row4",
        choices=("affine_rms_residual", "condition_number", "chi", "d_ep"),
        default="affine_rms_residual",
    )
    parser.add_argument(
        "--ratio-scale", choices=("auto", "linear", "log"), default="auto"
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)
    output = plot_magnetic_gradient_product(
        args.product,
        args.output,
        row4=args.row4,
        ratio_scale=args.ratio_scale,
        dpi=args.dpi,
    )
    print(output)
    return 0
