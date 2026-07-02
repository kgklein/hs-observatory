"""Field model that samples a static numerical simulation snapshot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hs_observatory.fields.base import FieldModel
from hs_observatory.io.simulation_bp import load_simulation_bp
from hs_observatory.io.simulation_npy import (
    PeriodicTrilinearInterpolator,
    SimulationSnapshot,
    load_simulation_npy,
)
from hs_observatory.utils.units import vector3


def coordinate_box_size_km(config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Resolve configured km, generic code-unit, or proton-gyroradius boxes."""

    units = str(config.get("units", "km")).lower()
    if units == "km":
        box = np.asarray(config.get("box_size_km"), dtype=float)
        metadata = {"units": "km", "box_size_km": box.tolist() if box.shape == (3,) else None}
    elif units == "code":
        box_code = np.asarray(config.get("box_size_code"), dtype=float)
        scale = float(config.get("code_length_to_km", 0.0))
        box = box_code * scale
        metadata = {
            "units": "code",
            "box_size_code": box_code.tolist() if box_code.shape == (3,) else None,
            "code_length_to_km": scale,
            "box_size_km": box.tolist() if box.shape == (3,) else None,
        }
        if scale <= 0.0 or not np.isfinite(scale):
            raise ValueError("simulation.coordinates.code_length_to_km must be positive")
    elif units == "rho_p":
        box_rho_p = np.asarray(config.get("box_size_rho_p"), dtype=float)
        scale = float(config.get("rho_p_km", 0.0))
        box = box_rho_p * scale
        metadata = {
            "units": "rho_p",
            "box_size_rho_p": box_rho_p.tolist() if box_rho_p.shape == (3,) else None,
            "rho_p_km": scale,
            "box_size_km": box.tolist() if box.shape == (3,) else None,
        }
        if scale <= 0.0 or not np.isfinite(scale):
            raise ValueError("simulation.coordinates.rho_p_km must be positive")
    else:
        raise ValueError("simulation.coordinates.units must be km, code, or rho_p")
    if box.shape != (3,) or not np.all(np.isfinite(box)) or np.any(box <= 0.0):
        raise ValueError("Simulation coordinate box must contain three positive lengths")
    return box, metadata


def hub_centered_coordinates(
    position_km: np.ndarray,
    box_size_km: np.ndarray,
    hub_spacecraft: int = 0,
) -> np.ndarray:
    """Center the Hub initially, then preserve the formation's absolute motion.

    Input and output use shape ``(spacecraft, time, 3)``. The first Hub sample
    maps to the box center. Every later point retains its displacement from
    that initial Hub position before periodic wrapping.
    """

    position = np.asarray(position_km, dtype=float)
    box = np.asarray(box_size_km, dtype=float)
    if position.ndim != 3 or position.shape[-1] != 3:
        raise ValueError("position must have shape (spacecraft, time, 3)")
    if not 0 <= hub_spacecraft < position.shape[0]:
        raise IndexError(
            f"hub_spacecraft {hub_spacecraft} is outside {position.shape[0]} spacecraft"
        )
    if box.shape != (3,) or np.any(box <= 0.0):
        raise ValueError("box_size_km must contain three positive lengths")
    initial_hub = position[hub_spacecraft, 0]
    relative = position - initial_hub[None, None, :]
    # TODO: Add an explicit simulation-to-spacecraft coordinate rotation here.
    return np.mod(relative + 0.5 * box, box)


class SimulationSampledField(FieldModel):
    """Sample one static, periodic simulation cube along spacecraft positions."""

    name = "simulation_sampled"

    def __init__(self, config: dict[str, Any]) -> None:
        simulation = config.get("simulation", {})
        if not isinstance(simulation, dict):
            raise ValueError("field.simulation must be a mapping")
        simulation_format = str(simulation.get("format", "npy")).lower()
        layout = simulation.get("layout", {})
        if not isinstance(layout, dict):
            raise ValueError("field.simulation.layout must be a mapping")
        if simulation_format == "npy":
            path = simulation.get("path")
            if not path:
                raise ValueError("field.simulation.path is required for format: npy")
            self.snapshot: SimulationSnapshot = load_simulation_npy(path, layout)
            source_paths = {"combined": str(Path(path).expanduser().resolve())}
        elif simulation_format == "bp":
            sources = simulation.get("sources", {})
            if not isinstance(sources, dict):
                raise ValueError("field.simulation.sources must be a mapping for format: bp")
            field_path = sources.get("field", simulation.get("path"))
            moments_path = sources.get("moments", simulation.get("moments_path"))
            if not field_path or not moments_path:
                raise ValueError(
                    "format: bp requires simulation.sources.field and simulation.sources.moments"
                )
            self.snapshot = load_simulation_bp(field_path, moments_path, layout)
            source_paths = {
                "field": str(Path(field_path).expanduser().resolve()),
                "moments": str(Path(moments_path).expanduser().resolve()),
            }
        else:
            raise NotImplementedError("simulation_sampled supports only format: npy or format: bp")

        coordinates = simulation.get("coordinates", {})
        if not isinstance(coordinates, dict):
            raise ValueError("field.simulation.coordinates must be a mapping")
        self.box_size_km, coordinate_metadata = coordinate_box_size_km(coordinates)
        self.interpolator = PeriodicTrilinearInterpolator(self.box_size_km)
        if str(simulation.get("boundary", "periodic")).lower() != "periodic":
            raise NotImplementedError(
                "simulation_sampled currently supports only periodic boundaries"
            )

        mapping = simulation.get("coordinate_mapping", {})
        if not isinstance(mapping, dict):
            raise ValueError("field.simulation.coordinate_mapping must be a mapping")
        if str(mapping.get("method", "hub_centered")) != "hub_centered":
            raise NotImplementedError("Only coordinate_mapping.method: hub_centered is supported")
        if "hub_spacecraft_id" in mapping:
            raise NotImplementedError(
                "hub_spacecraft_id requires spacecraft-ID injection; use hub_spacecraft index"
            )
        self.hub_spacecraft = int(mapping.get("hub_spacecraft", 0))
        if self.hub_spacecraft < 0:
            raise ValueError("coordinate_mapping.hub_spacecraft must be non-negative")

        background = config.get("background", {})
        if not isinstance(background, dict):
            raise ValueError("field.background must be a mapping")
        self.background = {
            "n": float(background.get("n", 0.0)),
            "v": vector3(background.get("v", (0.0, 0.0, 0.0)), "background.v"),
            "B": vector3(background.get("B", (0.0, 0.0, 0.0)), "background.B"),
        }
        fields_are = simulation.get("fields_are", {})
        scale_factors = simulation.get("scale_factors", {})
        if not isinstance(fields_are, dict) or not isinstance(scale_factors, dict):
            raise ValueError("simulation.fields_are and scale_factors must be mappings")
        self.fields_are = {
            quantity: str(fields_are.get(quantity, "total")).lower() for quantity in ("n", "v", "B")
        }
        if any(value not in {"total", "fluctuation"} for value in self.fields_are.values()):
            raise ValueError("simulation.fields_are values must be total or fluctuation")
        missing_fields = set(("n", "v", "B")) - set(self.snapshot.available_fields)
        invalid_missing = sorted(
            quantity for quantity in missing_fields if self.fields_are[quantity] != "fluctuation"
        )
        if invalid_missing:
            raise ValueError(
                "Snapshot fields omitted by simulation.layout.variables must be marked "
                f"as fluctuations so their backgrounds are used: {invalid_missing}"
            )
        self.scale_factors = {
            quantity: float(scale_factors.get(quantity, 1.0)) for quantity in ("n", "v", "B")
        }
        if not all(np.isfinite(value) for value in self.scale_factors.values()):
            raise ValueError("simulation.scale_factors must be finite")
        self.metadata = {
            "field_model": self.name,
            "simulation_path": source_paths.get("combined", source_paths.get("field")),
            "simulation_sources": source_paths,
            "simulation_format": simulation_format,
            "simulation_shape": list(self.snapshot.source_shape),
            "simulation_grid_shape": list(self.snapshot.grid_shape),
            "simulation_layout": self.snapshot.layout,
            "simulation_available_fields": list(self.snapshot.available_fields),
            "box_size_km": self.box_size_km.tolist(),
            "coordinate_units": coordinate_metadata,
            "coordinate_mapping_method": "hub_centered",
            "coordinate_mapping_reference": "hub_position_at_first_sample",
            "hub_spacecraft": self.hub_spacecraft,
            "boundary": "periodic",
            "grid_coordinates": "uniform_endpoint_excluded_[0,L)",
            "interpolation": "trilinear",
            "fields_are": self.fields_are,
            "scale_factors": self.scale_factors,
            "output_units": {"n": "cm^-3", "v": "km/s", "B": "nT"},
        }

    def evaluate_quantity(
        self, quantity: str, time: np.ndarray, position: np.ndarray
    ) -> np.ndarray:
        if quantity not in {"n", "v", "B"}:
            raise ValueError(f"Unknown field quantity: {quantity}")
        position = np.asarray(position, dtype=float)
        if position.ndim != 3 or position.shape[-1] != 3:
            raise ValueError("position must have shape (spacecraft, time, 3)")
        n_spacecraft, n_time, _ = position.shape
        time_array = np.asarray(time, dtype=float)
        if time_array.shape not in {(n_time,), (n_time, n_spacecraft)}:
            raise ValueError("time must have shape (time,) or (time, spacecraft)")
        mapped = hub_centered_coordinates(position, self.box_size_km, self.hub_spacecraft)
        points_time_first = np.swapaxes(mapped, 0, 1)
        grid = getattr(self.snapshot, quantity)
        sampled = self.interpolator.sample(grid, points_time_first)
        sampled *= self.scale_factors[quantity]
        if self.fields_are[quantity] == "fluctuation":
            sampled += self.background[quantity]
        return sampled

    # TODO: Add time-dependent snapshots, coordinate rotations, and advecting boxes.
