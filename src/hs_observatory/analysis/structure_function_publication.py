"""Refined 40X publication workflow; recalculation is always explicit."""

import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import h5py
import numpy as np
import yaml

from hs_observatory.analyze import analyze_from_config
from hs_observatory.analysis.derived.geometry_pairs import build_geometry_pair_plan
from hs_observatory.analysis.elsasser import compute_elsasser_analysis
from hs_observatory.analysis.structure_function import (
    FIELDS,
    P_MODEL_RUN_COLORS,
    _ensemble_interval,
    _fit_scaling_exponents,
    _positions_for_group_member,
    _settings,
    _to_serializable,
    plot_structure_function_summary_row,
)
from hs_observatory.analysis.structure_function_convergence import stage2_edges
from hs_observatory.analysis.structure_function_local import (
    derive_product,
    load_local_cases,
    plot_local_diagnostics,
    write_local_results,
)

P_VALUES = {400: .9, 401: .7, 402: .5}
DIRECTORY = Path("outputs/structure_function_stage3")


def digest_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_path(run):
    return Path(f"analysis/malara-prod-{run}_sf-stage3-publication.yaml")


def result_path(run):
    return DIRECTORY / f"malara-prod-{run}_sf_publication.json"


def verify_input(run, config):
    """Read existing inputs only; verify all member geometries before analysis."""
    source = Path(config["input"]["path"])
    if not source.is_file():
        raise FileNotFoundError(f"Existing 40X observations required: {source}")
    settings = _settings(config["structure_function"])
    if (settings["increment_mode"] != "all_spacetime_pairs"
            or settings["pair_sampling"]["mode"] != "geometry_stratified"
            or not np.array_equal(settings["lambda_bins_km"], stage2_edges())
            or settings["scale_ranges_km"] != {"primary": [150., 1200.], "sensitivity": [200., 1000.]}):
        raise ValueError("refined publication requires the Stage 2 sampler, edges and windows")
    with h5py.File(source) as handle:
        generation_yaml = handle["metadata"].attrs["config_yaml"]
        generation = yaml.safe_load(generation_yaml)
        field = generation["field"]
        turbulence = field["turbulence"]
        if (field["model"] != "malara_turbulence"
                or turbulence["intermittency_p"] != P_VALUES[run]
                or turbulence["injection_scale_index"] != 2
                or turbulence["n_scales"] != 14
                or turbulence["geometry"] != "isotropic"
                or not np.isclose(turbulence["spectral_exponent_h"], 1/3)
                or field["domain"]["L"] != [100000., 100000., 100000.]
                or field["domain"]["units"] != "km"):
            raise ValueError(f"{run}: incompatible field configuration")
        formation = json.loads(handle["metadata"].attrs["spacecraft_source"])
        if formation["source"] != "cdf":
            raise ValueError(f"{run}: CDF formation required; synthetic fallback is not a publication input")
        for variable in ("n", "v", "B"):
            if handle[f"observed/{variable}"].shape[0] != 50:
                raise ValueError(f"{run}: require all 50 existing MC realizations")
        for name, shape in [("position_offset", (50, 9, 3)), ("time_offset", (50, 9))]:
            values = handle[f"errors/{name}"][:]
            if values.shape != shape or not np.all(np.isfinite(values)):
                raise ValueError(f"{run}: invalid stored {name}")
    truth = compute_elsasser_analysis(source, dict(settings["elsasser"], data_group="truth"))
    times = truth["time"]
    if len(times) != 2401 or not np.isclose(times[-1]-times[0], 300):
        raise ValueError(f"{run}: require 2401 aligned timestamps / 300 seconds")
    nominal, _, _ = _positions_for_group_member(source, "truth", 0, times, 9)
    reference, _, _ = _positions_for_group_member("outputs/malara-prod-402.h5", "truth", 0, times, 9)
    if not np.array_equal(nominal, reference):
        raise ValueError(f"{run}: trajectory differs from the verified 402 formation")
    eligible = []
    for member in range(51):
        positions = nominal if member == 50 else _positions_for_group_member(
            source, "observed", member, times, 9)[0]
        plan = build_geometry_pair_plan(positions, stage2_edges(),
                                       min_pair_lambda_km=70., max_pair_lambda_km=20000.)
        if np.any(plan.populations == 0):
            raise ValueError(f"{run}: unexpected empty bin in member {member}")
        eligible.append(plan.populations)
    return dict(
        input_path=str(source.resolve()), input_sha256=digest_file(source),
        generation_config_sha256=hashlib.sha256(generation_yaml.encode()).hexdigest(),
        generation_config=generation, formation=formation,
        nominal_position_sha256=hashlib.sha256(nominal.tobytes()).hexdigest(),
        time_sha256=hashlib.sha256(np.asarray(times).tobytes()).hexdigest(),
        n_times=len(times), duration_s=float(times[-1]-times[0]), mc_members=50,
        eligible_min=np.min(eligible, axis=0), eligible_max=np.max(eligible, axis=0),
        truth_eligible=eligible[-1],
        position_policy="observed nominal r plus stored position_offset, unless group-specific positions",
        timing_policy="preserve clock-shifted field samples; do not add hidden timing displacement to labels",
    )


def add_saved_summaries(result):
    """Keep individual realizations; derive percentiles and local slopes after reduction."""
    for group, product in result["products"].items():
        for name in FIELDS:
            field = product[name]
            field["local_slopes"] = derive_product(product, name, product["orders"], "three-point")
            if group == "observed":
                field["ensemble_summary"] = dict(
                    percentiles=[5, 50, 95],
                    S_n=np.nanpercentile(field["S_n"], [5, 50, 95], axis=0),
                    xi_raw=np.nanpercentile(field["scaling_exponents"]["raw"], [5, 50, 95], axis=0),
                    fit_range_names=field["scaling_exponents"]["range_names"],
                )


def run_analysis(run):
    started = time.perf_counter()
    path = config_path(run)
    config_text = path.read_text()
    config = yaml.safe_load(config_text)
    print(f"{run}: verifying existing field, duration, 50 MC members and all bin populations", flush=True)
    provenance = verify_input(run, config)
    provenance["analysis_config_yaml"] = config_text
    provenance["analysis_config_sha256"] = hashlib.sha256(config_text.encode()).hexdigest()
    base = Path(__file__).parent
    provenance["analysis_code_sha256"] = {
        name: digest_file(base/name) for name in ["structure_function.py", "elsasser.py",
                                                 "derived/geometry_pairs.py"]}
    preflight_seconds = time.perf_counter()-started
    print(f"{run}: input verified; calculating all 50 MC members and ideal sampled truth", flush=True)
    analysis_started = time.perf_counter()
    output = analyze_from_config(path)
    analysis_seconds = time.perf_counter()-analysis_started
    result = json.loads(output.read_text())
    add_saved_summaries(result)
    provenance["input_unchanged"] = digest_file(config["input"]["path"]) == provenance["input_sha256"]
    if not provenance["input_unchanged"]:
        raise RuntimeError("input changed during analysis")
    result["metadata"]["stage3"] = dict(
        run_id=run, p_value=P_VALUES[run], provenance=provenance,
        preflight_seconds=preflight_seconds, analysis_seconds=analysis_seconds,
        total_seconds=time.perf_counter()-started,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
    )
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(_to_serializable(result), indent=2)+"\n")
    temporary.replace(output)
    print(f"{run}: saved {output}; analysis {analysis_seconds:.1f}s", flush=True)


def refit(result, bounds, min_bins=3):
    """Derive raw exponents from saved individual curves, never percentiles."""
    fits = {}
    for group in ("truth", "observed"):
        product = result["products"][group]
        fits[group] = _fit_scaling_exponents(
            np.asarray(product["z_plus"]["S_n"]), np.asarray(product["counts"]),
            np.asarray(product["lambda_bin_centers_km"]), product["orders"],
            {"plot": list(bounds)}, min_bins)["raw"][:, 0]
    return fits


def publication_figure(result, run, bounds, limits, min_fit_bins=3):
    """Two equal panels, with common scales and raw, unnormalized xi_n."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    figure, axes = plt.subplots(1, 2, figsize=(10., 4.6))
    figure.subplots_adjust(left=.075, right=.985, top=.87, bottom=.25, wspace=.28)
    plot_structure_function_summary_row(
        result, p_value=P_VALUES[run], axes=axes, show_kurtosis=False,
        fit_lambda_range_km=tuple(bounds), show_legends=False, show_run_label=False,
        structure_ylim=limits[0], exponent_ylim=limits[1], p_model_color="0.25",
        min_fit_bins=min_fit_bins)
    color = P_MODEL_RUN_COLORS[P_VALUES[run]]
    axes[0].set(xlim=(70, 20000), ylabel=r"$S_n^+(\lambda)$ [$(\mathrm{km\,s}^{-1})^n$]",
                title=rf"(a) HelioSwarm structure functions, $p={P_VALUES[run]:g}$")
    axes[1].set_title(rf"(b) Scaling exponents, {bounds[0]:g}–{bounds[1]:g} km")
    for axis in axes:
        axis.tick_params(labelsize=9)
        axis.xaxis.label.set_size(10)
        axis.yaxis.label.set_size(10)
        axis.title.set_size(10)
    # Existing row routines retain open truth markers and filled MC markers.
    orders = result["products"]["truth"]["orders"]
    order_handles = [Line2D([], [], color=color, linestyle="none", marker=marker,
                            markerfacecolor="white", markersize=4, label=f"n={n}")
                     for n, marker in zip(orders, ["o", "s", "^", "D", "v", "P"])]
    axes[0].legend(handles=order_handles, ncols=3, loc="upper left", fontsize=8,
                   framealpha=.9, handletextpad=.4, columnspacing=1)
    provenance = [
        Line2D([], [], color=color, marker="o", markerfacecolor="white", label="ideal sampled truth"),
        Line2D([], [], color=color, marker="o", linestyle="--", label="MC median"),
        Patch(facecolor=color, alpha=.12, label="MC 5th–95th"),
        Patch(facecolor="0.7", alpha=.25, label=f"fit {bounds[0]:g}–{bounds[1]:g} km"),
    ]
    figure.legend(handles=provenance, ncols=4, loc="lower center", bbox_to_anchor=(.5, .10),
                  fontsize=8, frameon=False, columnspacing=1)
    axes[1].legend(fontsize=8, loc="upper left", framealpha=.9)
    axes[1].lines[-1].set_linewidth(1.)
    physical = result["metadata"]["stage3"]["provenance"]["generation_config"]["field"]
    box = physical["domain"]["L"][0]
    injection = physical["turbulence"]["injection_scale_index"]
    cell = box/2**injection
    cell_label = f"{cell:,.0f}".replace(",", "{,}")
    figure.text(.5, .025,
                rf"$L_0=10^5$ km; $\ell_{{\rm inj}}=L_0/2^{injection}={cell_label}$ km"
                " — largest nominal cascade cell; no measured spectral break is implied.",
                ha="center", fontsize=8, color="0.4")
    return figure


def plot_saved(runs, window="primary", bounds=None, diagnostics=False, min_fit_bins=3):
    import matplotlib.pyplot as plt

    records = {run: json.loads(result_path(run).read_text()) for run in runs}
    selected = bounds or {"primary": [150., 1200.], "sensitivity": [200., 1000.]}[window]
    suffix = "" if bounds is None and window == "primary" else f"_fit_{selected[0]:g}_{selected[1]:g}"
    structure_min, structure_max, exponent_min, exponent_max = np.inf, -np.inf, np.inf, -np.inf
    derived = {}
    for run, result in records.items():
        fits = refit(result, selected, min_fit_bins)
        lower, median, upper = _ensemble_interval(fits["observed"])
        theory = 1-np.log2(P_VALUES[run]**(np.arange(1, 7)/3)
                          + (1-P_VALUES[run])**(np.arange(1, 7)/3))
        exponent_min = min(exponent_min, float(np.min([fits["truth"].min(), lower.min(), theory.min()])))
        exponent_max = max(exponent_max, float(np.max([fits["truth"].max(), upper.max(), theory.max()])))
        for group in ["truth", "observed"]:
            curves = np.asarray(result["products"][group]["z_plus"]["S_n"])
            visible = curves if group == "truth" else np.nanpercentile(curves, [5, 95], axis=0)
            structure_min = min(structure_min, float(visible.min()))
            structure_max = max(structure_max, float(visible.max()))
        derived[str(run)] = dict(fit_bounds=selected, truth=fits["truth"][0],
                                  mc_p05=lower, mc_median=median, mc_p95=upper,
                                  mc_minus_truth=median-fits["truth"][0], theory=theory)
    margin = .08*(exponent_max-exponent_min)
    limits = ((structure_min/1.6, structure_max*1.6), (exponent_min-margin, exponent_max+margin))
    DIRECTORY.mkdir(exist_ok=True)
    for run, result in records.items():
        figure = publication_figure(result, run, selected, limits, min_fit_bins)
        for extension in ["png", "pdf"]:
            target = DIRECTORY/f"malara-prod-{run}_sf_publication{suffix}.{extension}"
            figure.savefig(target, dpi=300)
            print(target, flush=True)
        plt.close(figure)
    derived["metadata"] = dict(
        fit_bounds=selected, min_fit_bins=min_fit_bins,
        result_sha256={str(run): digest_file(result_path(run)) for run in runs},
        renderer_sha256=digest_file(__file__), common_y_limits=limits,
    )
    (DIRECTORY/f"plot_summary{suffix}.json").write_text(json.dumps(_to_serializable(derived), indent=2)+"\n")
    if diagnostics:
        sensitivity_figure(records)
        local_config = dict(cases={str(run): dict(path=str(result_path(run)), p_value=P_VALUES[run])
                                  for run in runs}, max_order=6)
        cases = load_local_cases(local_config, "z_plus", "three-point", False)
        prefix = DIRECTORY/"local_slopes"
        plot_local_diagnostics(cases, prefix, dpi=200, show_theory=True)
        write_local_results(prefix.with_suffix(".h5"), cases, dict(
            estimator="three-point", quantity="z_plus", uncertainty="individual MC members"))


def sensitivity_figure(records):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(records), figsize=(4.2*len(records), 3.8), squeeze=False,
                                constrained_layout=True)
    saved = {}
    for axis, (run, result) in zip(axes[0], records.items()):
        saved[str(run)] = {}
        for name, bounds, style in [("primary", [150., 1200.], "-"),
                                    ("sensitivity", [200., 1000.], "--")]:
            fit = refit(result, bounds)
            lo, median, hi = _ensemble_interval(fit["observed"])
            color = P_MODEL_RUN_COLORS[P_VALUES[run]]
            x = np.arange(1, 7)
            axis.plot(x, fit["truth"][0], color=color, marker="o", markerfacecolor="white",
                      linestyle=style, label=f"truth {bounds[0]:g}–{bounds[1]:g} km")
            axis.errorbar(x+.06 if name == "primary" else x-.06, median,
                          yerr=[median-lo, hi-median], color=color, marker="s", markersize=3,
                          linestyle="none", capsize=2, alpha=.8,
                          label="MC median and 5th–95th" if name == "primary" else None)
            saved[str(run)][name] = dict(truth=fit["truth"][0], mc_p05=lo, mc_median=median, mc_p95=hi)
        axis.set(xlabel="order n", ylabel=r"$\xi_n$", title=f"{run}, p={P_VALUES[run]:g}")
        axis.grid(alpha=.18)
        axis.legend(fontsize=7)
    for extension in ["png", "pdf"]:
        figure.savefig(DIRECTORY/f"fit_window_sensitivity.{extension}", dpi=250)
    plt.close(figure)
    (DIRECTORY/"fit_window_sensitivity.json").write_text(json.dumps(_to_serializable(saved), indent=2)+"\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", type=int, choices=list(P_VALUES), default=list(P_VALUES))
    parser.add_argument("--recalculate", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--fit-window", choices=["primary", "sensitivity"], default="primary")
    parser.add_argument("--fit-lambda-range", nargs=2, type=float)
    parser.add_argument("--min-fit-bins", type=int, default=3)
    args = parser.parse_args(argv)
    if args.analysis_only and not args.recalculate:
        parser.error("--analysis-only requires --recalculate")
    if args.recalculate:
        for run in args.cases:
            run_analysis(run)
    if not args.analysis_only:
        plot_saved(args.cases, args.fit_window, args.fit_lambda_range, args.diagnostics,
                   args.min_fit_bins)


if __name__ == "__main__":
    main()
