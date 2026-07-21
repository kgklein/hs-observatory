"""Synthetic-observation generation workflow."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from hs_observatory.config import load_generation_config
from hs_observatory.errors.gaussian import error_model_from_config
from hs_observatory.errors.offsets import (
    position_offset_model_from_config,
    time_offset_model_from_config,
)
from hs_observatory.errors.observatory import (
    apply_observatory_measurement_errors,
    normalized_observatory_config,
    observatory_errors_enabled,
)
from hs_observatory.fields.sine_sum import field_from_config
from hs_observatory.io.cdf_positions import (
    FILL_VALUE_THRESHOLD,
    load_spacecraft_configuration,
    select_cdf_configuration_sample,
)
from hs_observatory.io.hdf5 import write_observations
from hs_observatory.spacecraft.configuration import SpacecraftConfiguration
from hs_observatory.visualization import (
    alfven_visualization_path,
    geometry_summary_paths,
    observed_alfven_visualization_path,
    observed_visualization_paths,
    save_default_visualizations,
    save_geometry_summary,
    save_observed_visualizations,
    visualization_paths,
)

PROGRESS_INTERVAL_S = 10.0


def _time_axes(config: dict) -> dict[str, np.ndarray]:
    start = float(config["t_start_s"])
    stop = float(config["t_stop_s"])
    configured_cadence = config["cadence_s"]
    cadences = (
        {quantity: float(configured_cadence[quantity]) for quantity in ("n", "v", "B")}
        if isinstance(configured_cadence, dict)
        else {quantity: float(configured_cadence) for quantity in ("n", "v", "B")}
    )
    # Include the requested stop when it lies on each instrument's cadence grid.
    return {
        quantity: np.arange(start, stop + cadence * 0.5, cadence, dtype=float)
        for quantity, cadence in cadences.items()
    }


def _spacecraft_on_time_axes(
    config: dict, time: dict[str, np.ndarray]
) -> dict[str, SpacecraftConfiguration]:
    """Load the trajectory once, then select each native measurement grid."""

    master_time = np.unique(np.concatenate(list(time.values())))
    master = load_spacecraft_configuration(config, master_time)
    result = {}
    for quantity, quantity_time in time.items():
        indices = np.searchsorted(master_time, quantity_time)
        if not np.array_equal(master_time[indices], quantity_time):
            raise RuntimeError(f"Could not align spacecraft positions for {quantity}")
        result[quantity] = SpacecraftConfiguration(
            master.spacecraft_ids,
            quantity_time,
            master.position[:, indices, :],
            metadata=dict(master.metadata),
            velocity=(
                None
                if master.velocity is None
                else master.velocity[:, indices, :]
            ),
        )
    return result


def _has_configured_errors(config: dict) -> bool:
    if observatory_errors_enabled(config):
        return True

    def enabled(model_config: dict | None) -> bool:
        model = str((model_config or {}).get("model", "none")).lower()
        return model not in {"none", "no_error"}

    return (
        enabled(config.get("time"))
        or enabled(config.get("position"))
        or any(enabled(value) for value in config.get("variables", {}).values())
    )


def _geometry_context_trajectory(config: dict, metadata: dict) -> np.ndarray | None:
    """Return full CDF observatory trajectory context when it is available."""

    if metadata.get("source") != "cdf":
        return None
    cdf_path = config.get("cdf_path", config.get("cdf_directory"))
    if not cdf_path:
        return None
    sample = select_cdf_configuration_sample(
        cdf_path,
        requested_time=config.get("configuration_time"),
    )
    valid = np.asarray(sample["valid"], dtype=bool)
    positions = np.asarray(sample["positions"], dtype=float)[:, valid, :]
    keep = np.all(positions > FILL_VALUE_THRESHOLD, axis=(1, 2))
    keep &= np.all(np.isfinite(positions), axis=(1, 2))
    if not np.any(keep):
        return None
    return np.swapaxes(positions[keep], 0, 1)


def _evaluate_quantity_with_progress(
    field,
    quantity: str,
    time: np.ndarray,
    position: np.ndarray,
    *,
    progress_time: np.ndarray | None = None,
    label: str,
) -> np.ndarray:
    """Evaluate a field quantity, reporting Malara progress every 10 seconds."""

    nominal_time = np.asarray(time if progress_time is None else progress_time, dtype=float)
    if (
        getattr(field, "name", "") != "malara_turbulence"
        or nominal_time.ndim != 1
        or nominal_time.size == 0
        or nominal_time[-1] - nominal_time[0] < PROGRESS_INTERVAL_S
    ):
        return field.evaluate_quantity(quantity, time, position)

    time_array = np.asarray(time, dtype=float)
    position_array = np.asarray(position, dtype=float)
    chunks = []
    start = 0
    elapsed_stop = nominal_time[-1] - nominal_time[0]
    report = PROGRESS_INTERVAL_S
    while report <= elapsed_stop + 1.0e-9:
        stop = int(np.searchsorted(nominal_time - nominal_time[0], report, side="right"))
        if stop > start:
            chunks.append(
                field.evaluate_quantity(
                    quantity,
                    time_array[start:stop] if time_array.ndim == 1 else time_array[start:stop, :],
                    position_array[:, start:stop, :],
                )
            )
            print(f"{label}: generated through t={report:g} measurement s", flush=True)
            start = stop
        report += PROGRESS_INTERVAL_S
    if start < nominal_time.size:
        chunks.append(
            field.evaluate_quantity(
                quantity,
                time_array[start:] if time_array.ndim == 1 else time_array[start:, :],
                position_array[:, start:, :],
            )
        )
    return np.concatenate(chunks, axis=0)


def generate_from_config(
    config_path: str | Path,
    *,
    truth_velocity_frame: str | None = None,
) -> Path:
    loaded = load_generation_config(config_path)
    config = loaded.data
    time = _time_axes(config["time"])
    spacecraft = _spacecraft_on_time_axes(config["spacecraft"], time)
    field = field_from_config(config["field"])
    velocity_frame = str(
        config.get("measurement", {}).get("velocity_frame", "spacecraft")
    ).lower()
    truth = {
        quantity: _evaluate_quantity_with_progress(
            field,
            quantity,
            time[quantity],
            spacecraft[quantity].position,
            label=f"truth {quantity}",
        )
        for quantity in ("n", "v", "B")
    }
    spacecraft_velocity = spacecraft["v"].velocity
    if spacecraft_velocity is None:
        if velocity_frame == "spacecraft":
            raise ValueError(
                "measurement.velocity_frame is 'spacecraft', but no spacecraft "
                "velocity is available; provide spacecraft.velocity_km_s or a "
                "trajectory source with velocity"
            )
        spacecraft_velocity_time_first = None
    else:
        spacecraft_velocity_time_first = np.swapaxes(spacecraft_velocity, 0, 1)
        truth["v_spacecraft_frame"] = (
            truth["v"] - spacecraft_velocity_time_first
        )

    errors_config = config["errors"]
    n_mc = int(errors_config.get("n_mc", 1))
    seed = errors_config.get("random_seed")
    rng = np.random.default_rng(seed)
    n_spacecraft = len(spacecraft["n"].spacecraft_ids)
    use_observatory_errors = observatory_errors_enabled(errors_config)
    observatory_config = (
        normalized_observatory_config(errors_config) if use_observatory_errors else None
    )
    time_config = (
        observatory_config["timing"]
        if use_observatory_errors
        else errors_config.get("time")
    )
    position_config = (
        observatory_config["position"]
        if use_observatory_errors
        else errors_config.get("position")
    )
    time_offset_model = time_offset_model_from_config(time_config)
    position_offset_model = position_offset_model_from_config(position_config)
    time_offset = time_offset_model.draw(rng, n_mc, n_spacecraft)  # (mc, spacecraft), seconds
    position_offset = position_offset_model.draw(
        rng, n_mc, n_spacecraft
    )  # (mc, spacecraft, xyz), km

    observed: dict[str, np.ndarray] = {}
    realized_errors: dict[str, np.ndarray] = {
        "time_offset": time_offset,
        "position_offset": position_offset,
    }
    variable_configs = errors_config.get("variables", {})
    measurement_error_names = {}
    sampled_by_quantity: dict[str, np.ndarray] = {}
    for quantity in ("n", "v", "B"):
        sampled = np.empty((n_mc, *truth[quantity].shape), dtype=float)
        for mc_index in range(n_mc):
            # The coordinate offsets are constant over this spacecraft's whole
            # time series. Nominal time/position remain unchanged in the file.
            sampled_time = time[quantity][:, None] + time_offset[mc_index][None, :]
            sampled_position = (
                spacecraft[quantity].position
                + position_offset[mc_index][:, None, :]
            )
            sampled[mc_index] = _evaluate_quantity_with_progress(
                field,
                quantity,
                sampled_time,
                sampled_position,
                progress_time=time[quantity],
                label=f"sampled {quantity} MC {mc_index + 1}/{n_mc}",
            )
            if quantity == "v" and velocity_frame == "spacecraft":
                assert spacecraft_velocity_time_first is not None
                sampled[mc_index] -= spacecraft_velocity_time_first
        sampled_by_quantity[quantity] = sampled

    if use_observatory_errors:
        result = apply_observatory_measurement_errors(
            sampled_by_quantity, time, errors_config, rng
        )
        observed.update(result.observed)
        realized_errors.update(result.errors)
        measurement_error_names = {
            "n": "observatory_density",
            "v": "observatory_velocity_speed_angle_alignment",
            "B": "observatory_magnetic_dc_ac_alignment",
        }
        observatory_metadata = result.metadata
    else:
        observatory_metadata = {}
        for quantity in ("n", "v", "B"):
            model = error_model_from_config(variable_configs.get(quantity), quantity=quantity)
            observed[quantity], realized_errors[quantity] = model.apply_ensemble(
                sampled_by_quantity[quantity], rng, time=time[quantity]
            )
            measurement_error_names[quantity] = model.name

    output_path = Path(config["output"]["path"])
    visualization_config = config.get("visualization", {})
    visualization_enabled = bool(visualization_config.get("enabled", True))
    requested_truth_velocity_frame = str(
        truth_velocity_frame
        or visualization_config.get("truth_velocity_frame", "spacecraft")
    ).lower()
    if requested_truth_velocity_frame not in {"spacecraft", "plasma"}:
        raise ValueError(
            "truth visualization velocity frame must be 'spacecraft' or 'plasma'"
        )
    truth_visualization_velocity_frame = requested_truth_velocity_frame
    if (
        truth_visualization_velocity_frame == "spacecraft"
        and "v_spacecraft_frame" not in truth
    ):
        truth_visualization_velocity_frame = "plasma"
    truth_for_visualization = dict(truth)
    if truth_visualization_velocity_frame == "spacecraft":
        truth_for_visualization["v"] = truth["v_spacecraft_frame"]
    observed_visualization_enabled = (
        visualization_enabled
        and bool(visualization_config.get("observed_enabled", True))
        and _has_configured_errors(errors_config)
    )
    observed_mc_index = int(visualization_config.get("mc_index", 0))
    if observed_visualization_enabled and not 0 <= observed_mc_index < n_mc:
        raise ValueError(
            f"visualization.mc_index must be between 0 and {n_mc - 1}; "
            f"got {observed_mc_index}"
        )
    truth_plot_path, geometry_plot_path = visualization_paths(output_path, visualization_config)
    geometry_summary_pdf_path, geometry_summary_png_path = geometry_summary_paths(
        output_path, spacecraft["n"].metadata
    )
    alfven_plot_path = alfven_visualization_path(output_path, visualization_config)
    observed_plot_path, observed_geometry_plot_path = observed_visualization_paths(
        output_path, visualization_config, observed_mc_index
    )
    observed_alfven_plot_path = observed_alfven_visualization_path(
        output_path, visualization_config, observed_mc_index
    )
    alfven_visualization_enabled = visualization_enabled and bool(
        visualization_config.get("alfven_enabled", True)
    )
    alfven_visualization_fluctuations = bool(
        visualization_config.get("alfven_fluctuations", True)
    )
    observed_alfven_visualization_enabled = (
        observed_visualization_enabled
        and bool(visualization_config.get("alfven_enabled", True))
    )
    metadata = {
        "config_yaml": loaded.yaml_text,
        "config_path": str(loaded.path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": "null" if seed is None else int(seed),
        "n_mc": n_mc,
        "field_model": field.name,
        "field_metadata": json.dumps(
            getattr(field, "metadata", {}), sort_keys=True
        ),
        "error_models": json.dumps(
            {
                "measurements": measurement_error_names,
                "time": time_offset_model.name,
                "position": position_offset_model.name,
                "observatory": bool(use_observatory_errors),
            },
            sort_keys=True,
        ),
        **observatory_metadata,
        "spacecraft_source": json.dumps(spacecraft["n"].metadata, sort_keys=True),
        "schema_version": "2",
        "velocity_truth_frame": "plasma",
        "velocity_observed_frame": velocity_frame,
        "velocity_error_applied_frame": velocity_frame,
        "truth_visualization_velocity_frame": truth_visualization_velocity_frame,
        "visualization_enabled": visualization_enabled,
        "truth_plot_path": str(truth_plot_path) if visualization_enabled else "null",
        "geometry_plot_path": str(geometry_plot_path) if visualization_enabled else "null",
        "geometry_summary_pdf_path": (
            str(geometry_summary_pdf_path) if visualization_enabled else "null"
        ),
        "geometry_summary_png_path": (
            str(geometry_summary_png_path) if visualization_enabled else "null"
        ),
        "alfven_visualization_enabled": alfven_visualization_enabled,
        "alfven_visualization_fluctuations": alfven_visualization_fluctuations,
        "alfven_plot_path": (
            str(alfven_plot_path) if alfven_visualization_enabled else "null"
        ),
        "observed_visualization_enabled": observed_visualization_enabled,
        "observed_visualization_mc_index": observed_mc_index,
        "observed_plot_path": (
            str(observed_plot_path) if observed_visualization_enabled else "null"
        ),
        "observed_geometry_plot_path": (
            str(observed_geometry_plot_path) if observed_visualization_enabled else "null"
        ),
        "observed_alfven_visualization_enabled": observed_alfven_visualization_enabled,
        "observed_alfven_plot_path": (
            str(observed_alfven_plot_path)
            if observed_alfven_visualization_enabled
            else "null"
        ),
    }
    output = write_observations(
        output_path,
        time=time,
        spacecraft_ids=spacecraft["n"].spacecraft_ids,
        position={quantity: spacecraft[quantity].position for quantity in ("n", "v", "B")},
        truth=truth,
        observed=observed,
        errors=realized_errors if config["output"].get("write_errors", True) else None,
        metadata=metadata,
    )
    if visualization_enabled:
        save_default_visualizations(
            output_path=output,
            config=visualization_config,
            time=time,
            truth=truth_for_visualization,
            position={quantity: spacecraft[quantity].position for quantity in ("n", "v", "B")},
            spacecraft_ids=spacecraft["n"].spacecraft_ids,
            velocity_frame=truth_visualization_velocity_frame,
        )
        path_quantity = max(("n", "v", "B"), key=lambda quantity: len(time[quantity]))
        pdf_path, png_path, geometry_diagnostics = save_geometry_summary(
            output_path=output,
            config=visualization_config,
            time=time[path_quantity],
            position=spacecraft[path_quantity].position,
            spacecraft_ids=spacecraft["n"].spacecraft_ids,
            metadata=spacecraft["n"].metadata,
            context_trajectory=_geometry_context_trajectory(
                config["spacecraft"], spacecraft["n"].metadata
            ),
        )
        print(
            "Geometry summary diagnostics:\n"
            f"  requested configuration_time: "
            f"{geometry_diagnostics['requested_configuration_time']}\n"
            f"  matched CDF sample time: "
            f"{geometry_diagnostics['matched_cdf_sample_time']}\n"
            f"  matched sample index: {geometry_diagnostics['matched_sample_index']}\n"
            f"  hub index: {geometry_diagnostics['hub_index']}\n"
            f"  number of spacecraft: {geometry_diagnostics['n_spacecraft']}\n"
            f"  number of unique baselines: {geometry_diagnostics['n_baselines']}\n"
            f"  baseline magnitude range: "
            f"{geometry_diagnostics['baseline_min_km']:.6g} to "
            f"{geometry_diagnostics['baseline_max_km']:.6g} km\n"
            f"  number of polyhedra: {geometry_diagnostics['n_polyhedra']}\n"
            f"  regular polyhedra with Q<=0.6: "
            f"{geometry_diagnostics['n_regular_polyhedra']}\n"
            f"  regular size ratio >= 3: "
            f"{geometry_diagnostics['regular_size_ratio_at_least_3']}\n"
            f"  output PDF path: {pdf_path}\n"
            f"  output PNG path: {png_path}"
        )
    if observed_visualization_enabled:
        observed_truth = dict(truth)
        if velocity_frame == "spacecraft":
            observed_truth["v"] = truth["v_spacecraft_frame"]
        save_observed_visualizations(
            output_path=output,
            config=visualization_config,
            time=time,
            truth=observed_truth,
            observed=observed,
            position={quantity: spacecraft[quantity].position for quantity in ("n", "v", "B")},
            position_offset=position_offset,
            spacecraft_ids=spacecraft["n"].spacecraft_ids,
            mc_index=observed_mc_index,
        )
    return output
