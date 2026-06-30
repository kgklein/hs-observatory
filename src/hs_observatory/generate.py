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
from hs_observatory.fields.sine_sum import field_from_config
from hs_observatory.io.cdf_positions import load_spacecraft_configuration
from hs_observatory.io.hdf5 import write_observations
from hs_observatory.spacecraft.configuration import SpacecraftConfiguration
from hs_observatory.visualization import (
    observed_visualization_paths,
    save_default_visualizations,
    save_observed_visualizations,
    visualization_paths,
)


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
        )
    return result


def _has_configured_errors(config: dict) -> bool:
    def enabled(model_config: dict | None) -> bool:
        model = str((model_config or {}).get("model", "none")).lower()
        return model not in {"none", "no_error"}

    return (
        enabled(config.get("time"))
        or enabled(config.get("position"))
        or any(enabled(value) for value in config.get("variables", {}).values())
    )


def generate_from_config(config_path: str | Path) -> Path:
    loaded = load_generation_config(config_path)
    config = loaded.data
    time = _time_axes(config["time"])
    spacecraft = _spacecraft_on_time_axes(config["spacecraft"], time)
    field = field_from_config(config["field"])
    truth = {
        quantity: field.evaluate_quantity(
            quantity, time[quantity], spacecraft[quantity].position
        )
        for quantity in ("n", "v", "B")
    }

    errors_config = config["errors"]
    n_mc = int(errors_config.get("n_mc", 1))
    seed = errors_config.get("random_seed")
    rng = np.random.default_rng(seed)
    n_spacecraft = len(spacecraft["n"].spacecraft_ids)
    time_offset_model = time_offset_model_from_config(errors_config.get("time"))
    position_offset_model = position_offset_model_from_config(errors_config.get("position"))
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
            sampled[mc_index] = field.evaluate_quantity(
                quantity, sampled_time, sampled_position
            )
        model = error_model_from_config(variable_configs.get(quantity), quantity=quantity)
        observed[quantity], realized_errors[quantity] = model.apply_ensemble(
            sampled, rng, time=time[quantity]
        )
        measurement_error_names[quantity] = model.name

    output_path = Path(config["output"]["path"])
    visualization_config = config.get("visualization", {})
    visualization_enabled = bool(visualization_config.get("enabled", True))
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
    observed_plot_path, observed_geometry_plot_path = observed_visualization_paths(
        output_path, visualization_config, observed_mc_index
    )
    metadata = {
        "config_yaml": loaded.yaml_text,
        "config_path": str(loaded.path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": "null" if seed is None else int(seed),
        "n_mc": n_mc,
        "field_model": field.name,
        "error_models": json.dumps(
            {
                "measurements": measurement_error_names,
                "time": time_offset_model.name,
                "position": position_offset_model.name,
            },
            sort_keys=True,
        ),
        "spacecraft_source": json.dumps(spacecraft["n"].metadata, sort_keys=True),
        "schema_version": "2",
        "visualization_enabled": visualization_enabled,
        "truth_plot_path": str(truth_plot_path) if visualization_enabled else "null",
        "geometry_plot_path": str(geometry_plot_path) if visualization_enabled else "null",
        "observed_visualization_enabled": observed_visualization_enabled,
        "observed_visualization_mc_index": observed_mc_index,
        "observed_plot_path": (
            str(observed_plot_path) if observed_visualization_enabled else "null"
        ),
        "observed_geometry_plot_path": (
            str(observed_geometry_plot_path) if observed_visualization_enabled else "null"
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
            truth=truth,
            position={quantity: spacecraft[quantity].position for quantity in ("n", "v", "B")},
            spacecraft_ids=spacecraft["n"].spacecraft_ids,
        )
    if observed_visualization_enabled:
        save_observed_visualizations(
            output_path=output,
            config=visualization_config,
            time=time,
            truth=truth,
            observed=observed,
            position={quantity: spacecraft[quantity].position for quantity in ("n", "v", "B")},
            position_offset=position_offset,
            spacecraft_ids=spacecraft["n"].spacecraft_ids,
            mc_index=observed_mc_index,
        )
    return output
