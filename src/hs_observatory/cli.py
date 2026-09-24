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
    parser.add_argument(
        "--3d-connection",
        action="append",
        dest="configuration_3d_extra_connections",
        metavar="LABEL-LABEL",
        help=(
            "additional solid pair in the 3D geometry plot; repeat as needed "
            "(valid labels: Hub, N1 through N8; example: N2-N5)"
        ),
    )
    args = parser.parse_args(argv)
    output = generate_from_config(
        args.config,
        truth_velocity_frame=args.truth_velocity_frame,
        configuration_3d_extra_connections=args.configuration_3d_extra_connections,
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
        "--fit-lambda-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="refit the saved structure functions over these lambda bounds in km",
    )
    parser.add_argument(
        "--min-fit-bins",
        type=int,
        default=None,
        help="minimum number of populated lambda bins required for a refit",
    )
    parser.add_argument(
        "--no-p-model",
        action="store_false",
        dest="show_p_model",
        help="hide the p-model reference curve",
    )
    parser.add_argument(
        "--no-kurtosis",
        action="store_false",
        dest="show_kurtosis",
        help="omit the kurtosis panel",
    )
    parser.add_argument(
        "--structure-ylim",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="set common structure-function y-axis bounds",
    )
    parser.add_argument(
        "--exponent-ylim",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="set common scaling-exponent y-axis bounds",
    )
    parser.add_argument(
        "--kurtosis-ylim",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="set common kurtosis y-axis bounds",
    )
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
        fit_lambda_range_km=args.fit_lambda_range,
        min_fit_bins=args.min_fit_bins,
        show_p_model=args.show_p_model,
        show_kurtosis=args.show_kurtosis,
        show_pair_counts=args.pair_counts,
        structure_ylim=args.structure_ylim,
        exponent_ylim=args.exponent_ylim,
        kurtosis_ylim=args.kurtosis_ylim,
        dpi=args.dpi,
    )
    print(output)
    return 0


def plot_structure_function_overlay_main(argv: list[str] | None = None) -> int:
    """Overlay several saved Malara structure-function publication results."""

    from hs_observatory.analysis.structure_function import (
        plot_structure_function_overlay_from_config,
    )

    parser = argparse.ArgumentParser(
        description="Overlay Malara structure functions and fitted scaling exponents"
    )
    parser.add_argument("config", help="structure-function overlay YAML")
    parser.add_argument("--output", default=None, help="override output figure path")
    parser.add_argument(
        "--fit-range",
        default=None,
        help="use a named exponent fit range saved in the analysis JSON",
    )
    parser.add_argument(
        "--fit-lambda-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN_KM", "MAX_KM"),
        help="refit the saved structure functions over these lambda bounds in km",
    )
    parser.add_argument(
        "--min-fit-bins",
        type=int,
        default=None,
        help="minimum number of populated lambda bins required for the refit",
    )
    parser.add_argument(
        "--no-p-model",
        action="store_const",
        const=False,
        default=None,
        dest="show_p_model",
        help="hide the theoretical p-model curves",
    )
    args = parser.parse_args(argv)
    output = plot_structure_function_overlay_from_config(
        args.config,
        output_override=args.output,
        fit_range_override=args.fit_range,
        fit_lambda_range_override=args.fit_lambda_range,
        min_fit_bins_override=args.min_fit_bins,
        show_p_model_override=args.show_p_model,
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
        "--lags", type=float, nargs="+", default=None, metavar="LAG",
        help="requested perpendicular lags in km; three for 1x3 or four for 2x2",
    )
    parser.add_argument(
        "--layout",
        choices=("1x3", "2x2"),
        default="1x3",
        help="panel layout; default: 1x3",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)
    expected_lags = 4 if args.layout == "2x2" else 3
    if args.lags is not None and len(args.lags) != expected_lags:
        parser.error(
            f"--layout {args.layout} requires exactly {expected_lags} --lags values"
        )
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
        layout=args.layout,
        dpi=args.dpi,
    )
    print(output)
    return 0


def plot_increment_pdf_wasserstein_main(argv: list[str] | None = None) -> int:
    """Plot Rayleigh-normalized W1 diagnostics from saved increment-PDF products."""

    from hs_observatory.analysis.increment_pdf_wasserstein import (
        plot_increment_pdf_wasserstein_from_config,
    )

    parser = argparse.ArgumentParser(
        description="Plot PDF comparisons and Rayleigh-normalized Wasserstein diagnostics"
    )
    parser.add_argument("config", help="lightweight publication plotting YAML")
    parser.add_argument(
        "--panels-a-c",
        action="store_true",
        help="plot only the current PDF (a) and Wasserstein (c) panels",
    )
    args = parser.parse_args(argv)
    output, diagnostics = plot_increment_pdf_wasserstein_from_config(
        args.config,
        panels_a_c=args.panels_a_c,
    )
    print(output)
    print("case,label,lag_km,DW_R_p25,DW_R_p50,DW_R_p75,DW_R_median_pdf")
    for case, entry in diagnostics["cases"].items():
        values = [
            case,
            entry["label"],
            f"{entry['selected_lag_km']:.8g}",
            *("nan" if not math.isfinite(entry[key]) else f"{entry[key]:.8g}" for key in (
                "DW_R_p25", "DW_R_p50", "DW_R_p75", "DW_R_median_pdf"
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
        "--layout",
        choices=("standard", "error_decomposition"),
        default="standard",
        help="standard diagnostics or compact curl-error decomposition",
    )
    parser.add_argument(
        "--row4",
        choices=("divergence", "affine_rms_residual", "condition_number", "chi", "d_ep"),
        default="divergence",
    )
    parser.add_argument(
        "--ratio-scale", choices=("auto", "linear", "log"), default="auto"
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=None,
        help="first analysis time to plot, in seconds",
    )
    parser.add_argument(
        "--stop-time",
        type=float,
        default=None,
        help="last analysis time to plot, in seconds",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)
    output = plot_magnetic_gradient_product(
        args.product,
        args.output,
        layout=args.layout,
        row4=args.row4,
        ratio_scale=args.ratio_scale,
        start_time_s=args.start_time,
        stop_time_s=args.stop_time,
        dpi=args.dpi,
    )
    print(output)
    return 0


def plot_magnetic_gradient_wasserstein_main(argv: list[str] | None = None) -> int:
    """Plot curl-magnitude Wasserstein diagnostics from saved gradient products."""

    from hs_observatory.analysis.magnetic_gradient_wasserstein import (
        magnetic_gradient_wasserstein_from_config,
    )

    parser = argparse.ArgumentParser(
        description="Plot magnetic-gradient curl-magnitude Wasserstein diagnostics"
    )
    parser.add_argument("config", help="lightweight Wasserstein comparison YAML")
    args = parser.parse_args(argv)
    output, diagnostics = magnetic_gradient_wasserstein_from_config(args.config)
    print(output)
    print(
        "configuration,case,method_W1_norm,"
        "measurement_W1_norm_median,total_W1_norm_median,pointwise_norm_median"
    )
    for category, product in diagnostics["configurations"].items():
        for case, entry in product["cases"].items():
            print(
                f"{category},{case},{entry['method_w1_normalized']:.8g},"
                f"{entry['measurement_w1_normalized']['median']:.8g},"
                f"{entry['total_w1_normalized']['median']:.8g},"
                f"{entry['pointwise_median_absolute_error_normalized']['median']:.8g}"
            )
    return 0


def plot_magnetic_gradient_combined_main(argv: list[str] | None = None) -> int:
    """Plot combined time-domain and distributional curl diagnostics."""

    from hs_observatory.analysis.magnetic_gradient_wasserstein import (
        magnetic_gradient_combined_from_config,
    )

    parser = argparse.ArgumentParser(
        description="Plot six-panel magnetic-gradient curl diagnostics"
    )
    parser.add_argument("config", help="combined magnetic-gradient comparison YAML")
    parser.add_argument("--output", default=None, help="override output figure path")
    parser.add_argument(
        "--configuration",
        default=None,
        help="tetrahedron category or most_regular",
    )
    parser.add_argument("--start-time", type=float, default=None)
    parser.add_argument("--stop-time", type=float, default=None)
    parser.add_argument("--dpi", type=int, default=None)
    args = parser.parse_args(argv)
    output, _diagnostics, configuration = magnetic_gradient_combined_from_config(
        args.config,
        output_override=args.output,
        configuration_override=args.configuration,
        start_time_override=args.start_time,
        stop_time_override=args.stop_time,
        dpi_override=args.dpi,
    )
    print(output)
    print(f"configuration={configuration}")
    return 0


def plot_magnetic_gradient_correlation_main(argv: list[str] | None = None) -> int:
    """Plot saved magnetic-gradient correlation products without recalculation."""

    from hs_observatory.analysis.magnetic_gradient_correlation import (
        plot_magnetic_gradient_correlations_from_config,
    )

    parser = argparse.ArgumentParser(
        description="Plot magnetic-gradient Pearson-correlation diagnostics"
    )
    parser.add_argument("config", help="magnetic-gradient correlation YAML")
    args = parser.parse_args(argv)
    outputs = plot_magnetic_gradient_correlations_from_config(args.config)
    for name, output in outputs.items():
        print(f"{name}={output}")
    return 0


def plot_timing_calibrate_main(argv: list[str] | None = None) -> int:
    """Plot a saved timing-calibration HDF5 product."""

    from hs_observatory.analysis.timing_calibrate import (
        plot_timing_calibration_product,
        plot_timing_calibration_timeseries,
    )

    parser = argparse.ArgumentParser(description="Plot HelioSwarm timing calibration diagnostics")
    parser.add_argument("product", help="timing-calibration analysis HDF5 file")
    parser.add_argument("--output", required=True, help="output figure path")
    parser.add_argument("--mc-index", type=int, default=0, help="representative MC realization")
    parser.add_argument(
        "--timeseries-component", choices=("x", "y", "z", "magnitude", "all"), default="x",
        help="magnetic-field quantity in the additional eight-Node time-series figure (default: x)",
    )
    parser.add_argument("--start-time", type=float, help="first time-series sample to plot, in seconds")
    parser.add_argument("--stop-time", type=float, help="last time-series sample to plot, in seconds")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--compare-products", nargs="+", default=[], metavar="HDF5",
        help="additional timing products with identical inputs; write a method comparison figure",
    )
    args = parser.parse_args(argv)
    print(plot_timing_calibration_product(
        args.product, args.output, mc_index=args.mc_index, dpi=args.dpi,
        start_time_s=args.start_time, stop_time_s=args.stop_time,
    ))
    output = Path(args.output)
    timeseries_output = output.with_name(f"{output.stem}_timeseries{output.suffix}")
    print(plot_timing_calibration_timeseries(
        args.product, timeseries_output, mc_index=args.mc_index,
        component=args.timeseries_component, dpi=args.dpi,
        start_time_s=args.start_time, stop_time_s=args.stop_time,
    ))
    from hs_observatory.analysis.timing_influence import plot_timing_influence

    for influence_output in plot_timing_influence(args.product, args.output, dpi=args.dpi):
        print(influence_output)
    if args.compare_products:
        from hs_observatory.analysis.timing_affine import plot_timing_method_comparison

        comparison_output = output.with_name(f"{output.stem}_comparison{output.suffix}")
        print(plot_timing_method_comparison(
            [args.product, *args.compare_products], comparison_output, dpi=args.dpi,
        ))
    return 0
