"""Scientific diagnostics for the unchanged iteration-10000 dHybridR example.

Run from the repository root with .venv/bin/python scripts/validate_dhybridr_shock.py.
Writes only ignored outputs/dhybridr/validation products. The report is a separate
reviewed artifact. No production configuration or implementation is changed.
The independent reference reads only seven *single-snapshot HDF5* datasets, keeps
their native [z,y,x] order, uses searchsorted interpolation and SI constants, and
never calls HIVE's interpolation for reference values.
"""

from __future__ import annotations

import argparse
import copy
import csv
from itertools import combinations, product
import json
import os
from pathlib import Path
import shutil
import tempfile

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hive-dhybridr-validation-mpl")
os.environ.setdefault("MPLBACKEND", "Agg")

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import elementary_charge, mu_0, proton_mass
from scipy.ndimage import gaussian_filter1d
import yaml

from hs_observatory.analysis.shock_timing import (
    DETECTION_STATUS,
    INVERSION_STATUS,
    compute_shock_timing,
    solve_planar_timing,
)
from hs_observatory.fields.simulation_sampled import (
    SimulationSampledField,
    hub_centered_coordinates,
)
from hs_observatory.generate import (
    _spacecraft_on_time_axes,
    _time_axes,
    _trajectory_positions_at_times,
    generate_from_config,
)
from hs_observatory.io.hdf5 import read_observations


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reference_data(release, iteration):
    """Read native source order and coordinate ranges independently of NPZ loader."""
    quantities = {"n": ("Phase/x3x2x1/Sp01", "dens_sp01")}
    quantities.update({f"B{c}": (f"Fields/Magnetic/Total/{c}", "Bfld") for c in "xyz"})
    quantities.update({f"V{c}": (f"Phase/FluidVel/Sp01/{c}", "Vfld") for c in "xyz"})
    arrays = {}
    axes = None
    for name, (directory, stem) in quantities.items():
        path = release / "Output" / directory / f"{stem}_{iteration:08d}.h5"
        with h5py.File(path) as handle:
            value = handle["DATA"][()].astype(float)
            this_axes = []
            for axis, count in enumerate(value.shape[::-1], 1):
                low, high = handle[f"AXIS/X{axis} AXIS"][()]
                this_axes.append(
                    np.linspace(low, high, count, endpoint=False) + (high - low) / (2 * count)
                )
            assert int(handle.attrs["ITER"][0]) == iteration
            assert float(handle.attrs["TIME"][0]) == 50.0
        if axes is not None:
            for a, b in zip(axes, this_axes):
                np.testing.assert_array_equal(a, b)
        axes = this_axes
        arrays[name] = value
    return axes, arrays


def reference_sample(array, coordinates, axes):
    """Independent scalar reference: native-axis searches, periodic y/z only."""
    shape = coordinates.shape[:-1]
    points = np.reshape(coordinates, (-1, 3)).copy()
    indices, fractions = [], []
    for dim, nodes in enumerate(axes):
        count = len(nodes)
        if dim == 0:
            if np.any((points[:, dim] < nodes[0] - 1e-12) | (points[:, dim] > nodes[-1] + 1e-12)):
                raise ValueError("reference x query outside cell centers")
            # Last center belongs to the final valid interior interpolation cell.
            lower = np.minimum(np.searchsorted(nodes, points[:, dim], side="right") - 1, count - 2)
            lower = np.maximum(lower, 0)
            upper = lower + 1
            fraction = (points[:, dim] - nodes[lower]) / (nodes[upper] - nodes[lower])
        else:
            period = (nodes[1] - nodes[0]) * count
            points[:, dim] = (points[:, dim] - nodes[0]) % period + nodes[0]
            extended = np.r_[nodes, nodes[0] + period]
            lower = np.searchsorted(extended, points[:, dim], side="right") - 1
            upper = (lower + 1) % count
            fraction = (points[:, dim] - extended[lower]) / (extended[lower + 1] - extended[lower])
        indices.append((lower, upper))
        fractions.append(fraction)
    result = np.zeros(len(points))
    for bits in product((0, 1), repeat=3):
        weight = np.ones(len(points))
        for axis, bit in enumerate(bits):
            weight *= fractions[axis] if bit else 1 - fractions[axis]
        ix, iy, iz = [indices[d][bit] for d, bit in enumerate(bits)]
        result += weight * array[iz, iy, ix]
    return result.reshape(shape)


def differences(name, actual, reference, units, floor):
    error = np.abs(np.asarray(actual) - reference)
    denominator = np.maximum(np.abs(reference), floor)
    assert np.all(np.isfinite(error))
    np.testing.assert_allclose(actual, reference, rtol=2e-10, atol=2e-10)
    return {
        "quantity": name,
        "units": units,
        "count": error.size,
        "max_absolute": float(error.max()),
        "max_relative_floored": float((error / denominator).max()),
        "relative_floor": floor,
    }


def upward_crossings(time, values, level, x, gate=(64.0, 74.0)):
    indices = np.flatnonzero(
        (values[:-1] < level) & (values[1:] >= level) & (x[:-1] >= gate[0]) & (x[1:] <= gate[1])
    )
    return np.array(
        [
            time[k] + (level - values[k]) / (values[k + 1] - values[k]) * (time[k + 1] - time[k])
            for k in indices
        ]
    )


def spatial_crossings(x, profile, level, gate=(64.0, 74.0)):
    # Reverse spatial ordering to follow the upstream-to-downstream traversal.
    return upward_crossings(-x[::-1], profile[::-1], level, x[::-1], gate) * -1


def solution_record(label, times, points):
    result = solve_planar_timing(times, points, points.mean(axis=0), float(times.mean()))
    return {
        "case": label,
        "status": next(k for k, v in INVERSION_STATUS.items() if v == result.status),
        "normal": result.normal.tolist(),
        "speed_km_s": result.speed,
        "rank": result.rank,
        "condition_number": result.condition_number,
        "rms_residual_s": result.rms_residual,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/example_generate_dhybridr.yaml")
    )
    parser.add_argument("--release", type=Path, default=Path("../dhybridr/release"))
    parser.add_argument("--output", type=Path, default=Path("outputs/dhybridr/validation"))
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.config.read_text())
    sim = config["field"]["simulation"]
    assert sim["iteration"] == 10000
    print("Loading one NPZ snapshot and independent HDF5 reference", flush=True)
    field = SimulationSampledField(config["field"])
    axes, raw = reference_data(args.release, sim["iteration"])
    n0, B0 = sim["normalization"]["n0_cm3"], sim["normalization"]["B0_nT"]
    di = np.sqrt(proton_mass / (mu_0 * n0 * 1e6 * elementary_charge**2)) / 1000
    va = B0 * 1e-9 / np.sqrt(mu_0 * proton_mass * n0 * 1e6) / 1000
    # SciPy's bundled CODATA version is independent of the loader's fixed constants.
    # Record its small scale offset instead of hiding it in interpolation errors.
    source_comparison = []
    for name, a in raw.items():
        expected = (
            field.snapshot.n
            if name == "n"
            else getattr(field.snapshot, "B" if name[0] == "B" else "v")[..., "xyz".index(name[-1])]
        )
        np.testing.assert_array_equal(a.transpose(2, 1, 0), expected)
        source_comparison.append({"quantity": name, "max_code_value_difference": 0.0})
    independent_scales = {
        "di_km": float(di),
        "vA_km_s": float(va),
        "Omega_ci_s_inv": elementary_charge * B0 * 1e-9 / proton_mass,
    }
    scale_offsets = {
        key: value / field.normalization[key] - 1 for key, value in independent_scales.items()
    }
    # Use independently evaluated expressions with the documented CODATA-2018 values
    # for the strict numerical comparison, so constants-version shifts are separate.
    di = np.sqrt(1.67262192369e-27 / (1.25663706212e-6 * n0 * 1e6 * 1.602176634e-19**2)) / 1000
    va = B0 * 1e-9 / np.sqrt(1.25663706212e-6 * 1.67262192369e-27 * n0 * 1e6) / 1000
    scales = {"n": n0, **{f"B{c}": B0 for c in "xyz"}, **{f"V{c}": va for c in "xyz"}}
    time = _time_axes(config["time"])
    spacecraft = _spacecraft_on_time_axes(config["spacecraft"], time)
    ids = list(spacecraft["n"].spacecraft_ids)
    assert ids == ["H", *[f"N{i}" for i in range(1, 9)]] and len(set(ids)) == 9
    reference = spacecraft["n"].position[field.hub_spacecraft, 0]
    field.set_coordinate_mapping_reference(reference)
    origin = np.array([axis[0] for axis in axes])
    periods = np.array([(axis[1] - axis[0]) * len(axis) for axis in axes])
    coordinates, unwrapped = {}, {}
    for q in time:
        actual_map = (
            hub_centered_coordinates(
                spacecraft[q].position,
                field.box_size_km,
                field.hub_spacecraft,
                reference_position_km=reference,
                initial_hub_position_km=field.initial_hub_position_km,
                wrap=False,
            )
            / di
            + origin
        )
        independent_map = sim["initial_hub_position_di"] + (spacecraft[q].position - reference) / di
        np.testing.assert_allclose(actual_map, independent_map, atol=1e-12, rtol=1e-14)
        unwrapped[q] = independent_map
        coordinates[q] = independent_map.copy()
        coordinates[q][..., 1:] = coordinates[q][..., 1:] % periods[1:]
    baseline = output / "baseline_observations.h5"
    candidates = [
        Path(config["output"]["path"]),
        Path("/tmp/hive-dhybridr-stage1-smoke/synthetic_observations_simulation_sampled.h5"),
    ]
    reused = None
    for candidate in candidates:
        if candidate.exists():
            data = read_observations(candidate)
            original = yaml.safe_load(data["metadata"]["config_yaml"])
            if all(original[k] == config[k] for k in config if k != "output"):
                shutil.copy2(candidate, baseline)
                reused = str(candidate)
                break
    if reused is None:
        fresh = copy.deepcopy(config)
        fresh["output"]["path"] = str(baseline)
        with tempfile.TemporaryDirectory(prefix="hive-dhybridr-baseline-") as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(fresh))
            generate_from_config(path)
    data = read_observations(baseline)
    assert data["metadata"]["schema_version"] == "2"
    assert data["spacecraft"]["id"] == ids
    comparison = []
    predictions = {}
    for q in time:
        np.testing.assert_array_equal(data["time"][q], time[q])
        np.testing.assert_array_equal(data["spacecraft"]["r"][q], spacecraft[q].position)
        names = ["n"] if q == "n" else [f"{'V' if q == 'v' else 'B'}{c}" for c in "xyz"]
        samples = [
            reference_sample(raw[name], unwrapped[q].swapaxes(0, 1), axes) * scales[name]
            for name in names
        ]
        predictions[q] = samples[0] if q == "n" else np.stack(samples, axis=-1)
        assert data["truth"][q].shape == predictions[q].shape
        for component, name in enumerate(names):
            actual = data["truth"][q] if q == "n" else data["truth"][q][..., component]
            comparison.append(
                differences(
                    name,
                    actual,
                    samples[component],
                    {"n": "cm^-3", "v": "km/s", "B": "nT"}[q],
                    {"n": 1e-8, "v": 1e-6, "B": 1e-8}[q],
                )
            )
        expected_observed = predictions[q].copy()
        if q == "v":
            expected_observed -= spacecraft[q].velocity.swapaxes(0, 1)
        comparison.append(
            differences(
                f"observed_{q}_zero_error",
                data["observed"][q][0],
                expected_observed,
                "package units",
                1e-8,
            )
        )
    comparison.append(
        differences(
            "Bmag",
            np.linalg.norm(data["truth"]["B"], axis=-1),
            np.linalg.norm(predictions["B"], axis=-1),
            "nT",
            1e-8,
        )
    )
    comparison.append(
        differences(
            "v_spacecraft_frame",
            data["truth"]["v_spacecraft_frame"],
            predictions["v"] - spacecraft["v"].velocity.swapaxes(0, 1),
            "km/s",
            1e-6,
        )
    )
    save_csv(output / "source_comparison.csv", comparison)
    save_csv(output / "npz_hdf5_grid_comparison.csv", source_comparison)

    x = axes[0]
    bmag = np.sqrt(sum(raw[f"B{c}"] ** 2 for c in "xyz"))
    fields = {
        "n": raw["n"],
        "Bmag": bmag,
        **{f"B{c}": raw[f"B{c}"] for c in "xyz"},
        "Vx": raw["Vx"],
    }
    profiles = {key: value.mean(axis=(0, 1)) for key, value in fields.items()}
    # A native grid-center cut through the actual initial Hub y/z, not a transverse average.
    jy, jz = [int(np.argmin(abs(axes[d] - sim["initial_hub_position_di"][d]))) for d in (1, 2)]
    plateau = []
    for label, lo, hi in [
        ("downstream", 20, 40),
        ("upstream_reference", 170, 220),
        ("local_downstream_ramp", 64, 67),
        ("local_precursor_ramp", 72, 75),
        ("traversal_start", 110, 120),
    ]:
        mask = (x >= lo) & (x <= hi)
        row = {"region": label, "xlo_di": lo, "xhi_di": hi}
        for key, a in fields.items():
            row[f"{key}_mean"] = float(a[..., mask].mean())
            row[f"{key}_std"] = float(a[..., mask].std())
        plateau.append(row)
    save_csv(output / "plateau_states.csv", plateau)
    level = 0.5 * (plateau[2]["n_mean"] + plateau[3]["n_mean"])
    mean_roots = spatial_crossings(x, gaussian_filter1d(profiles["n"], 1), level)
    assert len(mean_roots) == 1, mean_roots
    shock_x = float(mean_roots[0])
    jump = plateau[2]["n_mean"] - plateau[3]["n_mean"]
    ramp_edges = [
        float(
            spatial_crossings(
                x, gaussian_filter1d(profiles["n"], 1), plateau[3]["n_mean"] + fraction * jump
            )[-1]
        )
        for fraction in (0.1, 0.9)
    ]
    front = np.full(raw["n"].shape[:2], np.nan)
    for iz, iy in np.ndindex(front.shape):
        roots = spatial_crossings(x, gaussian_filter1d(raw["n"][iz, iy], 1), level)
        if len(roots):
            front[iz, iy] = roots[-1]  # Final sustained downstream entry within the gate.
    save_csv(
        output / "shock_surface.csv",
        [
            {"y_di": axes[1][iy], "z_di": axes[2][iz], "front_x_di": front[iz, iy]}
            for iz, iy in np.ndindex(front.shape)
        ],
    )

    geometry_rows = []
    for sc, name in enumerate(ids):
        a = unwrapped["B"][sc]
        wrapped = coordinates["B"][sc]
        geometry_rows.append(
            {
                "spacecraft": name,
                "x_start_di": a[0, 0],
                "x_end_di": a[-1, 0],
                "dx_di": a[-1, 0] - a[0, 0],
                "left_margin_di": a[:, 0].min() - x[0],
                "right_margin_di": x[-1] - a[:, 0].max(),
                "y_unwrapped_di": a[0, 1],
                "z_unwrapped_di": a[0, 2],
                "y_wrapped_di": wrapped[0, 1],
                "z_wrapped_di": wrapped[0, 2],
                "crosses_ramp": bool(a[:, 0].min() < 64 and a[:, 0].max() > 74),
            }
        )
    save_csv(output / "spacecraft_geometry.csv", geometry_rows)
    # Full positions at three times; common translation should cancel in every pair.
    pairs = []
    for index in (0, len(time["B"]) // 2, len(time["B"]) - 1):
        p = unwrapped["B"][:, index]
        for i, j in combinations(range(9), 2):
            vector = p[j] - p[i]
            minimum = vector.copy()
            minimum[1:] -= periods[1:] * np.floor(minimum[1:] / periods[1:] + 0.5)
            transverse = np.linalg.norm(vector[1:])
            effective = np.linalg.norm(minimum[1:])
            pairs.append(
                {
                    "time_s": time["B"][index],
                    "pair": f"{ids[i]}-{ids[j]}",
                    "dx_di": vector[0],
                    "dy_di": vector[1],
                    "dz_di": vector[2],
                    "dy_min_di": minimum[1],
                    "dz_min_di": minimum[2],
                    "physical_distance_km": np.linalg.norm(vector) * di,
                    "minimum_image_distance_km": np.linalg.norm(minimum) * di,
                    "transverse_distance_di": transverse,
                    "transverse_min_di": effective,
                    "transverse_reduction_di": transverse - effective,
                }
            )
    save_csv(output / "pair_geometry.csv", pairs)

    # Counterfactual variation: keep each spacecraft's x but put all at Hub y/z.
    variance = []
    for key, q in [("n", "n"), ("Vx", "v"), ("Bmag", "B")]:
        coord = unwrapped[q].swapaxes(0, 1).copy()
        coord[..., 1:] = coord[:, :1, 1:]
        if key == "Bmag":
            actual = np.linalg.norm(data["truth"][q], axis=-1) / B0
            xonly = np.sqrt(sum(reference_sample(raw[f"B{c}"], coord, axes) ** 2 for c in "xyz"))
        else:
            actual = data["truth"][q] / n0 if key == "n" else data["truth"][q][..., 0] / va
            xonly = reference_sample(raw[key], coord, axes)
        full = actual - actual.mean(axis=1, keepdims=True)
        xx = xonly - xonly.mean(axis=1, keepdims=True)
        residual = full - xx
        variance.append(
            {
                "quantity": key,
                "full_rms_code": float(np.sqrt(np.mean(full**2))),
                "x_only_rms_code": float(np.sqrt(np.mean(xx**2))),
                "transverse_residual_rms_code": float(np.sqrt(np.mean(residual**2))),
                "x_only_to_full_variance": float(np.mean(xx**2) / np.mean(full**2)),
                "residual_to_full_variance": float(np.mean(residual**2) / np.mean(full**2)),
                "twice_covariance_to_full": float(2 * np.mean(xx * residual) / np.mean(full**2)),
            }
        )
    save_csv(output / "variation_decomposition.csv", variance)

    # Density timing uses a declared spatial gate, a common local midpoint, and
    # 0.25-s Gaussian sigma. The stronger final ramp is distinct from x~89 compression.
    timing = []
    measured, planar = [], []
    velocity = np.asarray(config["spacecraft"]["velocity_km_s"])
    for sc, name in enumerate(ids):
        xx = unwrapped["n"][sc, :, 0]
        density = data["truth"]["n"][:, sc] / n0
        filtered = gaussian_filter1d(density, 0.25 / np.median(np.diff(time["n"])))
        roots = upward_crossings(time["n"], filtered, level, xx)
        assert len(roots) == 1, (name, roots)
        tc = float(roots[0])
        pred = (shock_x - xx[0]) * di / velocity[0]
        direct = upward_crossings(time["n"], density, level, xx)
        sensitivity = []
        for multiplier in (-0.1, 0.1):
            trial = level + multiplier * (plateau[2]["n_mean"] - plateau[3]["n_mean"])
            rr = upward_crossings(time["n"], filtered, trial, xx)
            sensitivity.append(float(rr[-1]) if len(rr) else np.nan)
        measured.append(tc)
        planar.append(pred)
        timing.append(
            {
                "spacecraft": name,
                "midpoint_n_n0": level,
                "measured_time_s": tc,
                "planar_time_s": pred,
                "measured_minus_planar_s": tc - pred,
                "crossing_x_di": float(np.interp(tc, time["n"], xx)),
                "raw_crossing_time_s": float(direct[-1]) if len(direct) else np.nan,
                "filtered_crossings_in_gate": len(roots),
                "raw_crossings_in_gate": len(direct),
                "low_threshold_time_s": sensitivity[0],
                "high_threshold_time_s": sensitivity[1],
            }
        )
    save_csv(output / "crossing_times.csv", timing)
    measured, planar = np.asarray(measured), np.asarray(planar)
    offsets = spacecraft["n"].position[:, 0]
    at_crossing = np.array(
        [
            [np.interp(tc, time["n"], spacecraft["n"].position[sc, :, d]) for d in range(3)]
            for sc, tc in enumerate(measured)
        ]
    )
    exact_at_crossing = offsets + planar[:, None] * velocity
    # Analytic plane intersection is exactly constant x. Avoid a spurious tiny
    # fourth SVD mode caused solely by cancellation of offset + velocity*time.
    exact_at_crossing[:, 0] = (shock_x - sim["initial_hub_position_di"][0]) * di + reference[0]
    timing_models = [
        solution_record("frozen_frame_exact_planar", planar, exact_at_crossing),
        solution_record("frozen_frame_measured", measured, at_crossing),
        solution_record("formation_frame_exact_planar", planar, offsets),
        solution_record("formation_frame_measured", measured, offsets),
    ]
    subsets = list(combinations(range(9), 4))

    # Best-conditioned fixed formation tetrahedron, selected without crossing times.
    def design_condition(sub):
        centered = offsets[list(sub)] - offsets[list(sub)].mean(axis=0)
        scale = np.sqrt(np.mean(np.sum(centered**2, axis=1)))
        return np.linalg.cond(np.column_stack((np.ones(4), centered / scale)))

    best = min(subsets, key=design_condition)
    four_rows = []
    for sub in subsets:
        record = solution_record("formation_frame_four", measured[list(sub)], offsets[list(sub)])
        four_rows.append(
            {
                "ids": ",".join(ids[i] for i in sub),
                "status": record["status"],
                "condition_number": record["condition_number"],
                "speed_km_s": record["speed_km_s"],
                "normal_angle_from_plus_x_deg": float(
                    np.degrees(np.arccos(np.clip(record["normal"][0], -1, 1)))
                ),
            }
        )
    save_csv(output / "four_spacecraft_timing.csv", four_rows)
    for label, tt, pp in [
        ("frozen_frame_measured", measured, at_crossing),
        ("formation_frame_measured", measured, offsets),
        ("frozen_frame_exact_planar", planar, exact_at_crossing),
    ]:
        timing_models.append(solution_record(label + "_four", tt[list(best)], pp[list(best)]))
    # Run the actual existing detector/inversion pipeline without changing settings.
    auto = compute_shock_timing(
        baseline,
        {
            "observables": ["n"],
            "modes": "independent",
            "truth_validation": "disabled",
            "plot": {"enabled": False},
        },
    )
    status_lookup = {v: k for k, v in DETECTION_STATUS.items()}
    auto_detections = [status_lookup[int(s)] for s in auto["independent"]["status"][0, 0]]

    # Controlled legacy Gaussian errors: MC=2, clock 10ms, position 0.5km,
    # density 0.05 cm^-3, velocity 1km/s per component, magnetic 0.02nT/component.
    error_config = copy.deepcopy(config)
    error_config["errors"] = {
        "n_mc": 2,
        "random_seed": 12345,
        "time": {"model": "gaussian_constant", "sigma_s": 0.01},
        "position": {"model": "gaussian_constant", "sigma_km": [0.5, 0.5, 0.5]},
        "variables": {
            "n": {"model": "gaussian_absolute", "sigma": 0.05},
            "v": {"model": "gaussian_vector_component", "sigma": [1.0, 1.0, 1.0]},
            "B": {"model": "gaussian_vector_component", "sigma": [0.02, 0.02, 0.02]},
        },
    }
    error_config["output"]["path"] = str(output / "error_observations.h5")
    print("Generating two error realizations with unchanged production API", flush=True)
    with tempfile.TemporaryDirectory(prefix="hive-dhybridr-errors-") as directory:
        path = Path(directory) / "config.yaml"
        path.write_text(yaml.safe_dump(error_config))
        generate_from_config(path)
    error_data = read_observations(error_config["output"]["path"])
    assert error_data["metadata"]["schema_version"] == "2"
    assert error_data["spacecraft"]["id"] == ids
    error_rows, error_margin = [], np.inf
    for mc in range(2):
        for q in time:
            shifted_time = time[q][:, None] + error_data["errors"]["time_offset"][mc]
            shifted_position = _trajectory_positions_at_times(spacecraft[q], shifted_time)
            shifted_position += error_data["errors"]["position_offset"][mc, :, None, :]
            coord = sim["initial_hub_position_di"] + (shifted_position - reference) / di
            error_margin = min(
                error_margin, float(np.minimum(coord[..., 0] - x[0], x[-1] - coord[..., 0]).min())
            )
            names = ["n"] if q == "n" else [f"{'V' if q == 'v' else 'B'}{c}" for c in "xyz"]
            sampled = [
                reference_sample(raw[name], coord.swapaxes(0, 1), axes) * scales[name]
                for name in names
            ]
            expected = sampled[0] if q == "n" else np.stack(sampled, axis=-1)
            if q == "v":
                expected -= spacecraft[q].velocity.swapaxes(0, 1)
            expected += error_data["errors"][q][mc]
            row = differences(
                f"mc{mc}_{q}", error_data["observed"][q][mc], expected, "package units", 1e-8
            )
            row["measurement_noise_rms"] = float(np.sqrt(np.mean(error_data["errors"][q][mc] ** 2)))
            nominal = data["observed"][q][0]
            row["observed_minus_nominal_rms"] = float(
                np.sqrt(np.mean((error_data["observed"][q][mc] - nominal) ** 2))
            )
            error_rows.append(row)
    save_csv(output / "error_smoke.csv", error_rows)
    # Explicit out-of-bounds query using a realized error offset and same nominal reference.
    bad_position = spacecraft["n"].position[:, :1].copy()
    bad_position += error_data["errors"]["position_offset"][0, :, None, :]
    bad_position[..., 0] += 600 * di
    try:
        field.evaluate_quantity("n", np.array([0.0]), bad_position)
    except ValueError as exc:
        guard_message = str(exc)
        assert "x wrapping is forbidden" in guard_message
    else:
        raise AssertionError("Boundary guard did not reject invalid perturbed x")

    # Three diagnostic figures: unlike the standard visualization, include source
    # profiles, simulation-native coordinates, independent overlays and timing gates.
    colors = plt.get_cmap("tab10")(np.arange(9))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, panels = plt.subplots(4, 1, figsize=(11, 11), sharex=True, constrained_layout=True)
    for panel, key, label in zip(
        panels[[0, 1, 3]], ["n", "Bmag", "Vx"], [r"$n/n_0$", r"$|B|/B_0$", r"$V_x/v_A$"]
    ):
        panel.plot(x, fields[key][jz, jy], color="0.55", lw=0.9, label="Hub y/z grid cut")
        panel.plot(x, profiles[key], color="black", lw=1.5, label="transverse mean")
        panel.set_ylabel(label)
    for c, color in zip("xyz", ["tab:red", "tab:green", "tab:blue"]):
        panels[2].plot(x, profiles[f"B{c}"], color=color, label=f"mean B{c}")
        panels[2].plot(x, raw[f"B{c}"][jz, jy], color=color, alpha=0.35, lw=0.7)
    panels[2].set_ylabel(r"$B_i/B_0$")
    for panel in panels:
        panel.axvspan(64, 74, alpha=0.15, color="tab:red", label="downstream ramp gate")
        panel.axvspan(85, 94, alpha=0.15, color="tab:orange", label="upstream compression")
        panel.axvline(120.25, ls="--", color="tab:blue", lw=1, label="initial Hub")
        panel.axvline(
            geometry_rows[0]["x_end_di"], ls="--", color="tab:purple", lw=1, label="final Hub"
        )
        panel.axvspan(
            min(r["x_end_di"] for r in geometry_rows),
            max(r["x_start_di"] for r in geometry_rows),
            color="tab:blue",
            alpha=0.035,
        )
        panel.grid(alpha=0.2)
    panels[0].legend(ncol=2, fontsize=8, loc="upper right")
    panels[2].legend(ncol=3, fontsize=8)
    panels[-1].set(xlabel=r"Source $x/d_i$ (downstream left, upstream right)", xlim=(15, 230))
    fig.suptitle("Figure 1 — Frozen dHybridR shock structure, iteration 10000")
    fig.savefig(output / "figure1_shock_structure.png", dpi=180)
    plt.close(fig)

    fig, panels = plt.subplots(
        1, 3, figsize=(15, 5), constrained_layout=True, gridspec_kw={"width_ratios": [1.5, 1, 1]}
    )
    for sc, name in enumerate(ids):
        a = unwrapped["B"][sc]
        panels[0].plot(a[:, 0], time["B"], color=colors[sc], label=name)
        panels[0].scatter([a[0, 0], a[-1, 0]], [0, 40], color=colors[sc], s=15)
        for panel in panels[1:]:
            panel.scatter(a[0, 1], a[0, 2], color=colors[sc])
            panel.annotate(name, a[0, 1:3], xytext=(3, 3), textcoords="offset points", fontsize=8)
    panels[0].axvspan(64, 74, color="tab:red", alpha=0.15)
    panels[0].axvspan(85, 94, color="tab:orange", alpha=0.15)
    panels[0].set(
        xlabel=r"Source $x/d_i$", ylabel="Observation time [s]", title="Nine prescribed tracks"
    )
    panels[0].legend(ncol=3, fontsize=8)
    panels[1].set(
        xlim=(0, 10),
        ylim=(0, 10),
        xlabel=r"$y/d_i$",
        ylabel=r"$z/d_i$",
        title="Full periodic cross section",
    )
    panels[2].set(xlabel=r"$y/d_i$", ylabel=r"$z/d_i$", title="Formation zoom (same points)")
    for panel in panels[1:]:
        panel.set_aspect("equal")
        panel.grid(alpha=0.2)
    for value in (0, 10):
        panels[1].axhline(value, color="black", ls="--")
        panels[1].axvline(value, color="black", ls="--")
    fig.suptitle("Figure 2 — HelioSwarm sampling geometry; no transverse wrapping on these tracks")
    fig.savefig(output / "figure2_sampling_geometry.png", dpi=180)
    plt.close(fig)

    fig, panels = plt.subplots(3, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
    for sc, name in enumerate(ids):
        for panel, q, values in [
            (panels[0], "n", data["truth"]["n"]),
            (panels[1], "B", np.linalg.norm(data["truth"]["B"], axis=-1)),
            (panels[2], "v", data["truth"]["v"][..., 0]),
        ]:
            panel.plot(time[q], values[:, sc], color=colors[sc], lw=0.8, label=name)
        panels[0].axvline(measured[sc], color=colors[sc], lw=0.5, alpha=0.5)
    panels[0].scatter(
        time["n"][::16],
        predictions["n"][::16, 0],
        s=18,
        marker="x",
        color="black",
        label="independent HDF5 reference (Hub)",
    )
    for panel in panels:
        panel.grid(alpha=0.2)
    panels[0].set_ylabel(r"$n$ [cm$^{-3}$]")
    panels[1].set_ylabel(r"$|B|$ [nT]")
    panels[2].set_ylabel(r"$V_x$ [km/s], simulation frame")
    panels[2].set_xlabel("Observation time [s]")
    panels[0].legend(ncol=5, fontsize=8)
    fig.suptitle("Figure 3 — Truth-level nine-spacecraft crossing (zero errors)")
    fig.savefig(output / "figure3_synthetic_crossing.png", dpi=180)
    plt.close(fig)

    fig, panels = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    mesh = panels[0].pcolormesh(axes[1], axes[2], front, shading="nearest", cmap="viridis")
    fig.colorbar(mesh, ax=panels[0], label=r"Midpoint surface $x/d_i$")
    for sc, name in enumerate(ids):
        p = coordinates["n"][sc, 0]
        panels[0].scatter(p[1], p[2], color=colors[sc], edgecolor="white", s=30)
    panels[0].set(xlabel=r"$y/d_i$", ylabel=r"$z/d_i$", title="Principal ramp surface (smoothed)")
    panels[1].scatter(np.arange(9), measured - planar, c=colors)
    panels[1].axhline(0, color="black", lw=0.8)
    panels[1].set(
        xticks=np.arange(9),
        xticklabels=ids,
        ylabel="Measured − planar arrival [s]",
        title="Surface/estimator departures; zero measurement errors",
    )
    panels[1].grid(alpha=0.2)
    fig.suptitle("Figure 4 — Transverse structure and gated crossing timing")
    fig.savefig(output / "figure4_surface_timing.png", dpi=180)
    plt.close(fig)

    metrics = {
        "baseline_config": str(args.config),
        "reused_baseline": reused,
        "normalization": field.normalization,
        "independent_CODATA_relative_offsets": scale_offsets,
        "source_shape_zyx": raw["n"].shape,
        "ids": ids,
        "plateaus": plateau,
        "density_compression_ratio": plateau[0]["n_mean"] / plateau[1]["n_mean"],
        "shock_gate_di": [64, 74],
        "secondary_gate_di": [85, 94],
        "midpoint_level_n_n0": level,
        "mean_ramp_crossing_x_di": shock_x,
        "ramp_10_90_edges_di": ramp_edges,
        "ramp_10_90_width_di": abs(ramp_edges[1] - ramp_edges[0]),
        "surface_x_percentiles_di": np.nanpercentile(front, [0, 5, 50, 95, 100]).tolist(),
        "surface_valid_columns": int(np.isfinite(front).sum()),
        "source_comparisons": comparison,
        "geometry": geometry_rows,
        "variation": variance,
        "timing": timing,
        "timing_models": timing_models,
        "best_four_ids": [ids[i] for i in best],
        "default_HIVE_density_detection": auto_detections,
        "formation_frame_four_status_counts": {
            key: sum(r["status"] == key for r in four_rows) for key in INVERSION_STATUS
        },
        "default_HIVE_ols_status": int(auto["solutions"]["n"]["ols"]["status"][0]),
        "default_HIVE_four_status_counts": {
            str(code): int(
                np.count_nonzero(auto["solutions"]["n"]["four_spacecraft"]["status"] == code)
            )
            for code in INVERSION_STATUS.values()
        },
        "maximum_pair_reduction_di": max(p["transverse_reduction_di"] for p in pairs),
        "minimum_transverse_pair_di": min(p["transverse_min_di"] for p in pairs),
        "error_minimum_x_boundary_margin_di": error_margin,
        "error_checks": error_rows,
        "boundary_guard_message": guard_message,
        "sample_counts": {key: len(value) for key, value in time.items()},
        "cell_distance_scales": {
            "density_samples_per_cell": 0.5 * di / 200 / 0.125,
            "B_samples_per_cell": 0.5 * di / 200 / 0.03125,
        },
    }

    # Nonfinite timing values represent explicitly unavailable/degenerate solutions.
    def serializable(value):
        if isinstance(value, dict):
            return {str(k): serializable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [serializable(v) for v in value]
        if isinstance(value, np.generic):
            return serializable(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    (output / "metrics.json").write_text(json.dumps(serializable(metrics), indent=2) + "\n")
    print(
        json.dumps(
            serializable(
                {
                    key: metrics[key]
                    for key in [
                        "density_compression_ratio",
                        "mean_ramp_crossing_x_di",
                        "surface_x_percentiles_di",
                        "timing_models",
                        "default_HIVE_density_detection",
                        "error_minimum_x_boundary_margin_di",
                    ]
                }
            ),
            indent=2,
        ),
        flush=True,
    )
    print(f"Wrote diagnostic products to {output}", flush=True)


if __name__ == "__main__":
    main()
