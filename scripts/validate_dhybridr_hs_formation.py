"""Recover the production Malara CDF formation and diagnose frozen shock sampling.

Run from repository root with .venv/bin/python -B scripts/validate_dhybridr_hs_formation.py.
Only new ignored outputs/dhybridr/hs_formation products are written. Reuses the
previous diagnostic's independent native-HDF5 interpolation reference; never
regenerates its outputs. Provenance assertions run before observation generation.
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hive-hs-formation-mpl")
os.environ.setdefault("MPLBACKEND", "Agg")

from cdflib import CDF
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
import yaml

from hs_observatory.analysis.magnetic_gradient import geometry_diagnostics, select_configurations
from hs_observatory.analysis.shock_timing import INVERSION_STATUS, solve_planar_timing
from hs_observatory.generate import _spacecraft_on_time_axes, _time_axes, generate_from_config
from hs_observatory.io.cdf_positions import load_spacecraft_configuration
from hs_observatory.io.hdf5 import read_observations
from hs_observatory.visualization import (
    geometry_summary_diagnostics,
    plot_3d_configuration,
    polyhedron_metrics,
    spacecraft_color,
    unique_baselines,
)
from validate_dhybridr_shock import (
    differences,
    reference_data,
    reference_sample,
    save_csv,
    spatial_crossings,
    upward_crossings,
)


def timing_record(label, indices, arrivals, points, ids):
    subset = np.asarray(indices)
    solution = solve_planar_timing(
        arrivals[subset],
        points[subset],
        points[subset].mean(axis=0),
        float(arrivals[subset].mean()),
    )
    return {
        "subset": label,
        "ids": ",".join(ids[i] for i in subset),
        "status": next(k for k, v in INVERSION_STATUS.items() if v == solution.status),
        "condition": solution.condition_number,
        "speed_km_s": solution.speed,
        "nx": solution.normal[0],
        "ny": solution.normal[1],
        "nz": solution.normal[2],
        "angle_from_x_deg": float(np.degrees(np.arccos(np.clip(solution.normal[0], -1, 1)))),
        "rms_s": solution.rms_residual,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/example_generate_dhybridr_hs_formation.yaml")
    )
    parser.add_argument("--production", type=Path, default=Path("generation/malara-prod-900.yaml"))
    parser.add_argument("--release", type=Path, default=Path("../dhybridr/release"))
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse output only if its embedded configuration matches",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    production = yaml.safe_load(args.production.read_text())
    baseline = yaml.safe_load(Path("configs/example_generate_dhybridr.yaml").read_text())
    assert config["field"] == baseline["field"]
    assert config["time"] == baseline["time"] and config["errors"] == baseline["errors"]
    assert config["measurement"] == baseline["measurement"]
    for key in ("source", "cdf_directory", "configuration_time"):
        assert config["spacecraft"][key] == production["spacecraft"][key]
    assert config["spacecraft"]["fallback_synthetic"] is False
    assert config["spacecraft"]["velocity_km_s"] == [-200.0, 0.0, 0.0]
    one = load_spacecraft_configuration(config["spacecraft"], np.array([0.0]))
    ids = list(one.spacecraft_ids)
    assert ids == ["H", *[f"N{i}" for i in range(1, 9)]]
    assert one.metadata["source"] == "cdf" and "fallback_reason" not in one.metadata
    expected_time = "2030-08-25T11:50:35.801000000"
    assert one.metadata["selected_time"] == expected_time
    points = one.position[:, 0]
    offsets = points - points[0]
    stats = geometry_summary_diagnostics(points, metadata=one.metadata)
    assert stats["n_baselines"] == 36 and stats["n_polyhedra"] == 382
    assert stats["n_regular_polyhedra"] == 46
    np.testing.assert_allclose(
        [stats["baseline_min_km"], stats["baseline_max_km"]], [71.3933, 3033.75], rtol=2e-6
    )
    # Load the production selector too; its translation does not change the t=0 formation.
    prod_geometry = load_spacecraft_configuration(production["spacecraft"], np.array([0.0]))
    np.testing.assert_array_equal(points, prod_geometry.position[:, 0])
    cdf = CDF(one.metadata["source_file"])
    position_attrs = cdf.varattsget("Position")
    assert position_attrs["UNITS"] == "km" and position_attrs["LABLAXIS"] == "GSE Pos"
    global_attrs = cdf.globalattsget()
    print("Recovered and verified actual production formation:", stats, flush=True)
    output_path = Path(config["output"]["path"])
    output = output_path.parent
    output.mkdir(parents=True, exist_ok=True)

    save_csv(
        output / "formation_coordinates.csv",
        [
            {
                "spacecraft": ids[i],
                "x_gse_km": points[i, 0],
                "y_gse_km": points[i, 1],
                "z_gse_km": points[i, 2],
                "dx_from_hub_km": offsets[i, 0],
                "dy_from_hub_km": offsets[i, 1],
                "dz_from_hub_km": offsets[i, 2],
            }
            for i in range(9)
        ],
    )
    # Optional verified reuse avoids another NPZ decompression for figure edits.
    if args.reuse and output_path.exists():
        stored = read_observations(output_path)
        assert yaml.safe_load(stored["metadata"]["config_yaml"]) == config
        print("Reusing verified existing real-formation product", flush=True)
    else:
        print("Generating unchanged-physics CDF-formation observations", flush=True)
        generate_from_config(args.config)
    data = read_observations(output_path)
    assert data["metadata"]["schema_version"] == "2"
    assert data["spacecraft"]["id"] == ids
    source_metadata = json.loads(data["metadata"]["spacecraft_source"])
    assert source_metadata["source"] == "cdf"
    assert source_metadata["selected_time"] == expected_time
    assert "fallback_reason" not in source_metadata
    metadata = json.loads(data["metadata"]["field_metadata"])
    di, va = metadata["normalization"]["di_km"], metadata["normalization"]["vA_km_s"]
    time = _time_axes(config["time"])
    geometry = _spacecraft_on_time_axes(config["spacecraft"], time)
    hub_di = np.array(config["field"]["simulation"]["initial_hub_position_di"])
    velocity = np.array(config["spacecraft"]["velocity_km_s"])
    axes, raw = reference_data(args.release, 10000)
    periods = np.array([(a[1] - a[0]) * len(a) for a in axes])
    origin = np.array([a[0] for a in axes])
    coordinates, wrapped = {}, {}
    compare = []
    for q in time:
        np.testing.assert_array_equal(data["time"][q], time[q])
        np.testing.assert_array_equal(data["spacecraft"]["r"][q], geometry[q].position)
        np.testing.assert_allclose(
            data["spacecraft"]["r"][q] - points[:, None],
            np.broadcast_to(time[q][None, :, None] * velocity, geometry[q].position.shape),
            atol=1e-10,
        )
        for k in (0, len(time[q]) // 2, len(time[q]) - 1):
            np.testing.assert_allclose(
                unique_baselines(data["spacecraft"]["r"][q][:, k]),
                unique_baselines(points),
                atol=1e-9,
            )
        coords = hub_di + (data["spacecraft"]["r"][q] - points[0]) / di
        assert np.all((coords[..., 0] >= axes[0][0]) & (coords[..., 0] <= axes[0][-1]))
        coordinates[q] = coords
        wrapped[q] = coords.copy()
        wrapped[q][..., 1:] = (coords[..., 1:] - origin[1:]) % periods[1:] + origin[1:]
        names = ["n"] if q == "n" else [f"{'V' if q == 'v' else 'B'}{c}" for c in "xyz"]
        references = []
        for component, name in enumerate(names):
            scale = 5.0 if q in ("n", "B") else va
            independent = reference_sample(raw[name], coords.swapaxes(0, 1), axes) * scale
            references.append(independent)
            actual = data["truth"][q] if q == "n" else data["truth"][q][..., component]
            compare.append(differences(name, actual, independent, "package units", 1e-8))
        predicted = references[0] if q == "n" else np.stack(references, axis=-1)
        if q == "B":
            compare.append(
                differences(
                    "Bmag",
                    np.linalg.norm(data["truth"]["B"], axis=-1),
                    np.linalg.norm(predicted, axis=-1),
                    "nT",
                    1e-8,
                )
            )
        if q == "v":
            predicted -= velocity
            compare.append(
                differences(
                    "v_spacecraft_frame",
                    data["truth"]["v_spacecraft_frame"],
                    predicted,
                    "km/s",
                    1e-6,
                )
            )
        compare.append(
            differences(
                f"observed_{q}_zero_error", data["observed"][q][0], predicted, "package units", 1e-8
            )
        )
        assert np.all(data["errors"][q] == 0)
    assert np.all(data["errors"]["time_offset"] == 0)
    assert np.all(data["errors"]["position_offset"] == 0)
    save_csv(output / "source_comparison.csv", compare)

    pair_lookup, pairs = {}, []
    for i, j in combinations(range(9), 2):
        delta = offsets[j] - offsets[i]
        minimum = delta.copy()
        images = np.floor(minimum[1:] / (10 * di) + 0.5).astype(int)
        minimum[1:] -= images * (10 * di)
        row = {
            "pair": f"{ids[i]}-{ids[j]}",
            "i": i,
            "j": j,
            "dx_km": delta[0],
            "dy_km": delta[1],
            "dz_km": delta[2],
            "length_km": np.linalg.norm(delta),
            "length_di": np.linalg.norm(delta) / di,
            "dx_di": delta[0] / di,
            "dy_di": delta[1] / di,
            "dz_di": delta[2] / di,
            "transverse_km": np.linalg.norm(delta[1:]),
            "min_dy_km": minimum[1],
            "min_dz_km": minimum[2],
            "minimum_image_length_km": np.linalg.norm(minimum),
            "minimum_transverse_km": np.linalg.norm(minimum[1:]),
            "transverse_reduction_km": np.linalg.norm(delta[1:]) - np.linalg.norm(minimum[1:]),
            "length_reduction_fraction": 1 - np.linalg.norm(minimum) / np.linalg.norm(delta),
            "y_image_count": images[0],
            "z_image_count": images[1],
            "aliased": bool(np.any(images)),
            "grid_cells": np.linalg.norm(delta) / (0.5 * di),
            "planar_delay_s": delta[0] / 200,
        }
        pair_lookup[i, j] = row
        pairs.append(row)
    save_csv(output / "baselines.csv", pairs)
    selected = select_configurations(points, enabled={"best_multi": True})
    records, polyhedra = polyhedron_metrics(points), []
    for r in records:
        idx = r["indices"]
        aliases = sum(pair_lookup[i, j]["aliased"] for i, j in combinations(idx, 2))
        g = geometry_diagnostics(points[list(idx)])
        row = {
            "ids": ",".join(ids[i] for i in idx),
            "n": r["n_spacecraft"],
            "L_km": r["L"],
            "L_di": r["L"] / di,
            "Q": r["Q"],
            "E": r["E"],
            "P": r["P"],
            "tensor_condition": g.condition_number,
            "rank": g.rank,
            "regular": r["Q"] <= 0.6,
            "aliased_pairs": aliases,
            "all_baselines_unaliased": aliases == 0,
        }
        polyhedra.append(row)
    save_csv(output / "polyhedra.csv", polyhedra)
    selected_rows = []
    for name, r in selected.items():
        key = ",".join(ids[i] for i in r["indices"])
        row = dict(next(r for r in polyhedra if r["ids"] == key))
        row["selection"] = name
        row["criterion"] = r["criterion"]
        selected_rows.append(row)
    save_csv(output / "selected_polyhedra.csv", selected_rows)

    x = axes[0]
    profile = raw["n"].mean(axis=(0, 1))
    downstream = profile[(x >= 64) & (x <= 67)].mean()
    precursor = profile[(x >= 72) & (x <= 75)].mean()
    threshold = 0.5 * (downstream + precursor)
    mean_ramp = float(spatial_crossings(x, gaussian_filter1d(profile, 1), threshold)[0])
    crossings, planar, paths = [], [], []
    for sc, name in enumerate(ids):
        xx = coordinates["n"][sc, :, 0]
        series = data["truth"]["n"][:, sc] / 5
        filtered = gaussian_filter1d(series, 0.25 / 0.125)
        roots = upward_crossings(time["n"], filtered, threshold, xx)
        # Preserve ambiguity/failure if any: do not force a crossing or change the track.
        measured = float(roots[0]) if len(roots) == 1 else np.nan
        predicted = (mean_ramp - xx[0]) * di / velocity[0]
        raw_roots = upward_crossings(time["n"], series, threshold, xx)
        crossings.append(measured)
        planar.append(predicted)
        a, w = coordinates["n"][sc], wrapped["n"][sc]
        paths.append(
            {
                "spacecraft": name,
                "x_start_di": a[0, 0],
                "x_end_di": a[-1, 0],
                "x_margin_di": min(a[:, 0].min() - 0.25, 499.75 - a[:, 0].max()),
                "y_unwrapped_di": a[0, 1],
                "z_unwrapped_di": a[0, 2],
                "y_wrapped_di": w[0, 1],
                "z_wrapped_di": w[0, 2],
                "start_n_n0": series[0],
                "end_n_n0": series[-1],
                "gated_crossings": len(roots),
                "raw_gated_crossings": len(raw_roots),
                "crossing_time_s": measured,
                "planar_time_s": predicted,
                "residual_s": measured - predicted,
            }
        )
    save_csv(output / "spacecraft_crossings.csv", paths)
    crossings, planar = np.array(crossings), np.array(planar)
    accepted = np.flatnonzero(np.isfinite(crossings))
    fits = []
    if len(accepted) >= 4:
        fits.append(
            timing_record("all_crossing_formation_frame", accepted, crossings, offsets, ids)
        )
        fits.append(timing_record("exact_planar_formation_frame", accepted, planar, offsets, ids))
    for name, record in selected.items():
        idx = record["indices"]
        if all(i in accepted for i in idx):
            fits.append(timing_record(name, idx, crossings, offsets, ids))
    # Existing solver, evaluated for every existing tetrahedron; no new selection algorithm.
    tetra_fits = []
    for r in records:
        idx = r["indices"]
        if len(idx) == 4 and all(i in accepted for i in idx):
            row = timing_record("tetrahedron", idx, crossings, offsets, ids)
            row["Q"] = r["Q"]
            row["aliased_pairs"] = sum(
                pair_lookup[i, j]["aliased"] for i, j in combinations(idx, 2)
            )
            tetra_fits.append(row)
    if fits:
        save_csv(output / "timing_fits.csv", fits)
    if tetra_fits:
        save_csv(output / "tetrahedron_timing.csv", tetra_fits)
    delays = []
    for row in pairs:
        i, j = row["i"], row["j"]
        delays.append(
            {
                "pair": row["pair"],
                "measured_delay_s": crossings[j] - crossings[i],
                "planar_delay_s": planar[j] - planar[i],
                "residual_delay_s": (crossings[j] - crossings[i]) - (planar[j] - planar[i]),
            }
        )
    save_csv(output / "pair_timing_delays.csv", delays)

    colors = [spacecraft_color(name) for name in ids]
    fig, figstats = plot_3d_configuration(points, ids, metadata=one.metadata)
    # Retain existing plotting/projection conventions; add the tensor principal dimensions.
    g = geometry_diagnostics(points)
    for text in fig.texts:
        if "Sunward" in text.get_text():
            text.set_text("CDF labels: GSE positions in km | origin at HelioSwarm mesocenter")
    fig.text(
        0.5,
        0.98,
        "2×tensor semiaxes [km]: " + ", ".join(f"{2 * s:.1f}" for s in g.semiaxes),
        fontsize=9,
        ha="center",
        va="top",
    )
    fig.savefig(output / "figure1_real_formation.png", dpi=180)
    plt.close(fig)
    fig, panels = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for sc, name in enumerate(ids):
        a = coordinates["B"][sc]
        w = wrapped["B"][sc]
        panels[0].plot(a[:, 0], time["B"], color=colors[sc], label=name)
        panels[1].scatter(a[0, 1], a[0, 2], color=colors[sc], marker="o")
        panels[1].annotate(name, a[0, 1:], xytext=(4, 4), textcoords="offset points")
        panels[2].scatter(w[0, 1], w[0, 2], color=colors[sc], marker="s")
        panels[2].annotate(name, w[0, 1:], xytext=(4, 4), textcoords="offset points")
    panels[0].axvspan(64, 74, color="tab:red", alpha=0.15, label="downstream ramp")
    panels[0].axvspan(85, 94, color="tab:orange", alpha=0.15, label="upstream compression")
    for bound in (0.25, 499.75):
        panels[0].axvline(bound, color="black", ls="--", lw=1)
    panels[0].set(
        xlabel=r"Unwrapped $x/d_i$",
        ylabel="Observation time [s]",
        xlim=(-5, 505),
        title="Tracks and nonperiodic x limits",
    )
    panels[0].legend(ncol=2, fontsize=7)
    for panel in panels[1:]:
        for bound in (0, 10):
            panel.axvline(bound, color="black", ls="--", lw=0.7)
            panel.axhline(bound, color="black", ls="--", lw=0.7)
        panel.set(xlabel=r"$y/d_i$", ylabel=r"$z/d_i$")
        panel.set_aspect("equal")
        panel.grid(alpha=0.2)
    panels[1].set_title("Actual relative geometry (unwrapped)")
    panels[2].set(title="Periodically wrapped sampling locations", xlim=(0, 10), ylim=(0, 10))
    fig.suptitle("Figure 2 — Real formation in the frozen dHybridR domain; identity axis alignment")
    fig.savefig(output / "figure2_domain_tracks.png", dpi=180)
    plt.close(fig)
    fig, panels = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    for sc, name in enumerate(ids):
        for panel, q, values in [
            (panels[0], "n", data["truth"]["n"]),
            (panels[1], "B", np.linalg.norm(data["truth"]["B"], axis=-1)),
            (panels[2], "v", data["truth"]["v"][..., 0]),
        ]:
            panel.plot(time[q], values[:, sc], color=colors[sc], lw=1.0, label=name)
        if np.isfinite(crossings[sc]):
            panels[0].scatter(
                crossings[sc], threshold * 5, color=colors[sc], s=25, edgecolor="black"
            )
    panels[0].axhline(threshold * 5, color="0.5", ls="--", lw=0.7)
    panels[0].legend(ncol=9, fontsize=8)
    for panel in panels:
        panel.grid(alpha=0.2)
    panels[0].set_ylabel(r"$n$ [cm$^{-3}$]")
    panels[1].set_ylabel(r"$|B|$ [nT]")
    panels[2].set_ylabel("Vx [km/s], simulation frame")
    panels[2].set_xlabel("Observation time [s]")
    fig.suptitle(
        "Figure 3 — Nine actual-formation traces; zero errors; gated ramp crossings marked"
    )
    fig.savefig(output / "figure3_real_crossing.png", dpi=180)
    plt.close(fig)
    fig, panels = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    panels[0, 0].hist([r["length_di"] for r in pairs], bins=12, color="0.5")
    panels[0, 0].axvline(0.5, color="black", ls="--", label="one cell")
    panels[0, 0].set(
        xlabel="Physical baseline [di]", ylabel="Pair count", title="36 mission baselines"
    )
    panels[0, 0].legend()
    alias_colors = ["tab:red" if r["aliased"] else "tab:blue" for r in pairs]
    panels[0, 1].scatter(
        [r["transverse_km"] / di for r in pairs],
        [r["minimum_transverse_km"] / di for r in pairs],
        c=alias_colors,
    )
    panels[0, 1].plot([0, 30], [0, 30], color="0.5", ls="--")
    panels[0, 1].set(
        xlabel="Physical transverse separation [di]",
        ylabel="Minimum image [di]",
        title="Red: periodic-image reduction; blue: unchanged",
    )
    tetras = [r for r in polyhedra if r["n"] == 4]
    panels[1, 0].scatter(
        [r["L_di"] for r in tetras],
        [r["Q"] for r in tetras],
        c=["tab:red" if r["aliased_pairs"] else "tab:blue" for r in tetras],
        s=15,
    )
    panels[1, 0].axhline(0.6, color="black", ls="--")
    panels[1, 0].set(
        xlabel="Tetrahedron L [di]", ylabel="Physical Q", title="126 existing tetrahedra"
    )
    panels[1, 1].scatter(
        [abs(r["planar_delay_s"]) for r in delays],
        [abs(r["residual_delay_s"]) for r in delays],
        c=alias_colors,
    )
    panels[1, 1].set(
        xlabel="Absolute planar pair delay [s]",
        ylabel="Absolute delay departure [s]",
        title="Timing departures include transverse structure",
    )
    fig.suptitle("Figure 4 — Geometric suitability and periodic-image limitations")
    fig.savefig(output / "figure4_suitability.png", dpi=180)
    plt.close(fig)

    # Orientation opportunity: projections of the actual baselines along existing
    # tensor principal directions, not a new alignment or geometry selection.
    eig, vectors = np.linalg.eigh(g.volumetric_tensor)
    orientation = []
    baseline_vectors = unique_baselines(points)
    for d in range(3):
        direction = vectors[:, 2 - d]
        projected = abs(baseline_vectors @ direction)
        orientation.append(
            {
                "principal_axis": d + 1,
                "direction": direction.tolist(),
                "abs_projection_min_km": projected.min(),
                "abs_projection_median_km": np.median(projected),
                "abs_projection_max_km": projected.max(),
            }
        )
    metrics = {
        "production_config": str(args.production),
        "formation_metadata": one.metadata,
        "verification": stats,
        "cdf_position_attributes": {k: position_attrs[k] for k in ("UNITS", "LABLAXIS", "CATDESC")},
        "cdf_design_description": global_attrs["TEXT"],
        "normalization": metadata["normalization"],
        "ids": ids,
        "offsets_km": offsets.tolist(),
        "spans_di": (np.ptp(points, axis=0) / di).tolist(),
        "principal_full_dimensions_km": (2 * g.semiaxes).tolist(),
        "x_abs_baseline_percentiles_km": np.percentile(
            abs(baseline_vectors[:, 0]), [0, 25, 50, 75, 100]
        ).tolist(),
        "orientation_projections": orientation,
        "aliased_pair_count": sum(r["aliased"] for r in pairs),
        "near_identical_transverse_pairs": [
            r for r in pairs if r["minimum_transverse_km"] < 0.5 * di
        ],
        "polyhedron_counts": [
            {
                "n": n,
                "total": sum(r["n"] == n for r in polyhedra),
                "regular": sum(r["n"] == n and r["regular"] for r in polyhedra),
                "unaliased": sum(r["n"] == n and r["all_baselines_unaliased"] for r in polyhedra),
                "regular_unaliased": sum(
                    r["n"] == n and r["regular"] and r["all_baselines_unaliased"] for r in polyhedra
                ),
            }
            for n in range(4, 10)
        ],
        "selected": selected_rows,
        "paths": paths,
        "source_comparison": compare,
        "midpoint_level_n_n0": threshold,
        "planar_surface_x_di": mean_ramp,
        "timing_fits": fits,
        "crossing_count": len(accepted),
        "delay_abs_percentiles_s": np.nanpercentile(
            [abs(r["measured_delay_s"]) for r in delays], [0, 25, 50, 75, 100]
        ).tolist(),
        "planar_delay_abs_percentiles_s": np.percentile(
            [abs(r["planar_delay_s"]) for r in delays], [0, 25, 50, 75, 100]
        ).tolist(),
        "max_pair_delay_departure_s": float(
            np.nanmax([abs(r["residual_delay_s"]) for r in delays])
        ),
        "figure1_diagnostics": figstats,
    }

    def encode(value):
        if isinstance(value, dict):
            return {str(k): encode(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, np.ndarray)):
            return [encode(v) for v in value]
        if isinstance(value, np.generic):
            return encode(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    (output / "metrics.json").write_text(json.dumps(encode(metrics), indent=2) + "\n")
    print(
        json.dumps(
            encode(
                {
                    key: metrics[key]
                    for key in ("crossing_count", "aliased_pair_count", "timing_fits", "paths")
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
