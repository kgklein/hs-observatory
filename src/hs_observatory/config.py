"""Small, explicit YAML configuration loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


@dataclass(frozen=True)
class LoadedConfig:
    """A parsed configuration together with its exact serialized source."""

    data: dict[str, Any]
    yaml_text: str
    path: Path


def load_config(path: str | Path, required_sections: Iterable[str]) -> LoadedConfig:
    path = Path(path)
    yaml_text = path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(yaml_text)
    if not isinstance(parsed, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {path}")

    missing = [section for section in required_sections if section not in parsed]
    if missing:
        raise ValueError(f"Missing required configuration sections: {', '.join(missing)}")
    for section in required_sections:
        if not isinstance(parsed[section], dict):
            raise ValueError(f"Configuration section '{section}' must be a mapping")
    return LoadedConfig(data=parsed, yaml_text=yaml_text, path=path.resolve())


def load_generation_config(path: str | Path) -> LoadedConfig:
    loaded = load_config(path, ("spacecraft", "time", "field", "errors", "output"))
    cfg = loaded.data
    for key in ("cadence_s", "t_start_s", "t_stop_s"):
        if key not in cfg["time"]:
            raise ValueError(f"Missing required time setting: {key}")
    cadence = cfg["time"]["cadence_s"]
    if isinstance(cadence, dict):
        missing_quantities = [quantity for quantity in ("n", "v", "B") if quantity not in cadence]
        if missing_quantities:
            raise ValueError(
                "time.cadence_s is missing quantities: " + ", ".join(missing_quantities)
            )
        cadence_values = [float(cadence[quantity]) for quantity in ("n", "v", "B")]
    else:
        # A scalar remains supported for older configurations and means that
        # every instrument uses the same cadence.
        cadence_values = [float(cadence)]
    if any(value <= 0 for value in cadence_values):
        raise ValueError("All time.cadence_s values must be positive")
    if float(cfg["time"]["t_stop_s"]) < float(cfg["time"]["t_start_s"]):
        raise ValueError("time.t_stop_s must be at least time.t_start_s")
    if "model" not in cfg["field"]:
        raise ValueError("Missing required field.model setting")
    if int(cfg["errors"].get("n_mc", 1)) < 1:
        raise ValueError("errors.n_mc must be at least one")
    measurement = cfg.get("measurement", {})
    if not isinstance(measurement, dict):
        raise ValueError("Configuration section 'measurement' must be a mapping")
    velocity_frame = str(measurement.get("velocity_frame", "spacecraft")).lower()
    if velocity_frame not in {"spacecraft", "plasma"}:
        raise ValueError("measurement.velocity_frame must be 'spacecraft' or 'plasma'")
    for coordinate_error in ("time", "position"):
        value = cfg["errors"].get(coordinate_error, {})
        if not isinstance(value, dict):
            raise ValueError(f"errors.{coordinate_error} must be a mapping")
    if "path" not in cfg["output"]:
        raise ValueError("Missing required output.path setting")
    visualization = cfg.get("visualization", {})
    if not isinstance(visualization, dict):
        raise ValueError("Configuration section 'visualization' must be a mapping")
    if int(visualization.get("dpi", 150)) <= 0:
        raise ValueError("visualization.dpi must be positive")
    if int(visualization.get("mc_index", 0)) < 0:
        raise ValueError("visualization.mc_index must be non-negative")
    truth_velocity_frame = str(visualization.get("truth_velocity_frame", "spacecraft")).lower()
    if truth_velocity_frame not in {"spacecraft", "plasma"}:
        raise ValueError("visualization.truth_velocity_frame must be 'spacecraft' or 'plasma'")
    return loaded


def load_analysis_config(path: str | Path) -> LoadedConfig:
    loaded = load_config(path, ("input", "analysis", "output"))
    if "path" not in loaded.data["input"]:
        raise ValueError("Missing required input.path setting")
    if "method" not in loaded.data["analysis"]:
        raise ValueError("Missing required analysis.method setting")
    if "path" not in loaded.data["output"]:
        raise ValueError("Missing required output.path setting")
    return loaded
