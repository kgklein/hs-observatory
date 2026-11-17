"""Field model that samples a static numerical simulation snapshot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hs_observatory.fields.base import FieldModel
from hs_observatory.io.simulation_bp import load_simulation_bp
from hs_observatory.io.simulation_dhybridr import load_simulation_dhybridr, proton_normalization
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
    reference_position_km: np.ndarray | None = None,
    initial_hub_position_km: np.ndarray | None = None,
    wrap: bool = True,
) -> np.ndarray:
    """Center the Hub initially, then preserve the formation's absolute motion.

    Input and output use shape ``(spacecraft, time, 3)``. The first Hub sample
    maps to the box center. Every later point retains its displacement from
    that initial Hub position before periodic wrapping. An optional target is
    expressed in rebased grid km. ``wrap=False`` permits source-specific domain
    validation before any periodic wrapping; defaults retain the original map.
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
    initial_hub = (
        position[hub_spacecraft, 0]
        if reference_position_km is None
        else np.asarray(reference_position_km, dtype=float)
    )
    if initial_hub.shape != (3,) or not np.all(np.isfinite(initial_hub)):
        raise ValueError("reference_position_km must contain three finite coordinates")
    relative = position - initial_hub[None, None, :]
    # TODO: Add an explicit simulation-to-spacecraft coordinate rotation here.
    target = (
        0.5 * box
        if initial_hub_position_km is None
        else np.asarray(initial_hub_position_km, dtype=float)
    )
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("initial_hub_position_km must contain three finite coordinates")
    mapped = relative + target
    return np.mod(mapped, box) if wrap else mapped


class SimulationSampledField(FieldModel):
    """Sample one frozen volume along prescribed simulation-frame trajectories."""

    name = "simulation_sampled"

    def __init__(self, config: dict[str, Any]) -> None:
        simulation = config.get("simulation", {})
        if not isinstance(simulation, dict):
            raise ValueError("field.simulation must be a mapping")
        simulation_format = str(simulation.get("format", "npy")).lower()
        self.is_dhybridr = simulation_format == "dhybridr_npz"
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
        elif self.is_dhybridr:
            sources = simulation.get("sources", {})
            if not isinstance(sources, dict) or not all(
                sources.get(q) for q in ("fields", "moments")
            ):
                raise ValueError("dhybridr_npz requires simulation.sources.fields and .moments")
            if layout:
                raise ValueError("dhybridr_npz uses the documented release layout; omit layout")
            normalization = simulation.get("normalization", {})
            if not isinstance(normalization, dict) or not all(
                key in normalization for key in ("n0_cm3", "B0_nT")
            ):
                raise ValueError("dhybridr_npz requires normalization.n0_cm3 and .B0_nT")
            self.normalization = proton_normalization(
                normalization["n0_cm3"], normalization["B0_nT"]
            )
            self.snapshot = load_simulation_dhybridr(
                sources["fields"], sources["moments"], simulation.get("iteration")
            )
            source_paths = {
                key: str(Path(sources[key]).expanduser().resolve()) for key in ("fields", "moments")
            }
            simulation = self._dhybridr_configuration(simulation)
        else:
            raise NotImplementedError("simulation_sampled supports npy, bp, or dhybridr_npz")

        coordinates = simulation.get("coordinates", {})
        if not isinstance(coordinates, dict):
            raise ValueError("field.simulation.coordinates must be a mapping")
        self.box_size_km, coordinate_metadata = coordinate_box_size_km(coordinates)
        self.interpolator = PeriodicTrilinearInterpolator(self.box_size_km)
        boundary = "interior_x_periodic_yz" if self.is_dhybridr else "periodic"
        if str(simulation.get("boundary", boundary)).lower() != boundary:
            raise NotImplementedError(
                f"simulation format {simulation_format} requires boundary: {boundary}"
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
        self.coordinate_mapping_reference_km: np.ndarray | None = None

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
            "simulation_path": source_paths.get(
                "combined", source_paths.get("field", source_paths.get("fields"))
            ),
            "simulation_sources": source_paths,
            "simulation_format": simulation_format,
            "simulation_shape": list(self.snapshot.source_shape),
            "simulation_grid_shape": list(self.snapshot.grid_shape),
            "simulation_layout": self.snapshot.layout,
            "simulation_available_fields": list(self.snapshot.available_fields),
            "box_size_km": self.box_size_km.tolist(),
            "coordinate_units": coordinate_metadata,
            "coordinate_mapping_method": "hub_centered",
            "coordinate_mapping_reference": "nominal_hub_position_at_first_sample",
            "hub_spacecraft": self.hub_spacecraft,
            "boundary": boundary,
            "grid_coordinates": "uniform_endpoint_excluded_[0,L)",
            "interpolation": "trilinear",
            "fields_are": self.fields_are,
            "scale_factors": self.scale_factors,
            "output_units": {"n": "cm^-3", "v": "km/s", "B": "nT"},
        }
        if self.is_dhybridr:
            self.metadata.update(
                {
                    "normalization": self.normalization,
                    "initial_hub_position_di": self.initial_hub_position_di.tolist(),
                    "grid_origin_di": self.snapshot.layout["origin_di"],
                    "frozen_snapshot": True,
                    "trajectory_frame": "simulation_downstream_rest",
                    "axis_alignment": "identity",
                    "snapshot_time_s": self.snapshot.layout["snapshot_time_Omega_ci_inv"]
                    / self.normalization["Omega_ci_s_inv"],
                }
            )

    def _dhybridr_configuration(self, simulation: dict[str, Any]) -> dict[str, Any]:
        """Fill derived settings locally and reject contradictory user scales."""
        simulation = dict(simulation)
        scale = self.normalization["di_km"]
        box = np.asarray(self.snapshot.layout["box_size_di"])
        coordinates = simulation.get("coordinates", {})
        if not isinstance(coordinates, dict):
            raise ValueError("simulation.coordinates must be a mapping")
        coordinates = dict(coordinates)
        units = str(coordinates.get("units", "code")).lower()
        if units == "code":
            expected = {"units": "code", "box_size_code": box.tolist(), "code_length_to_km": scale}
        elif units == "km":
            expected = {"units": "km", "box_size_km": (box * scale).tolist()}
        else:
            raise ValueError("dHybridR coordinates.units must be code (d_i) or km")
        for key, value in coordinates.items():
            if key not in expected or (
                key != "units"
                and (
                    np.shape(value) != np.shape(expected[key])
                    or not np.allclose(
                        np.asarray(value, dtype=float), expected[key], rtol=1e-10, atol=0.0
                    )
                )
            ):
                raise ValueError(
                    f"dHybridR coordinates.{key} conflicts with the grid/normalization"
                )
        simulation["coordinates"] = expected
        derived = {
            "n": self.normalization["n0_cm3"],
            "v": self.normalization["vA_km_s"],
            "B": self.normalization["B0_nT"],
        }
        configured = simulation.get("scale_factors", {})
        if not isinstance(configured, dict):
            raise ValueError("simulation.scale_factors must be a mapping")
        for key, value in configured.items():
            if (
                key not in derived
                or isinstance(value, (bool, np.bool_))
                or np.ndim(value) != 0
                or not np.isclose(float(value), derived[key], rtol=1e-10, atol=0)
            ):
                raise ValueError(f"dHybridR scale_factors.{key} conflicts with normalization")
        simulation["scale_factors"] = derived
        fields_are = simulation.get("fields_are", {})
        if not isinstance(fields_are, dict) or any(
            key not in derived or str(value).lower() != "total" for key, value in fields_are.items()
        ):
            raise ValueError("dHybridR n, v, B are total quantities; fields_are must be total")
        origin = np.asarray(self.snapshot.layout["origin_di"])
        self.initial_hub_position_di = np.asarray(
            simulation.get("initial_hub_position_di", origin + box / 2), dtype=float
        )
        if self.initial_hub_position_di.shape != (3,) or not np.all(
            np.isfinite(self.initial_hub_position_di)
        ):
            raise ValueError("initial_hub_position_di must contain three finite coordinates")
        lo, hi = self.snapshot.layout["cell_center_interval_di"][0]
        if not lo <= self.initial_hub_position_di[0] <= hi:
            raise ValueError(
                f"Initial dHybridR x={self.initial_hub_position_di[0]} d_i "
                f"outside supported cell-center interval [{lo}, {hi}] d_i"
            )
        # Keep the exact legacy half-box calculation when placement is omitted.
        self.initial_hub_position_km = (
            (self.initial_hub_position_di - origin) * scale
            if "initial_hub_position_di" in simulation
            else None
        )
        return simulation

    def set_coordinate_mapping_reference(self, position_km: np.ndarray) -> None:
        """Fix the simulation-box origin to the nominal initial Hub position."""

        reference = np.asarray(position_km, dtype=float)
        if reference.shape != (3,) or not np.all(np.isfinite(reference)):
            raise ValueError("Simulation mapping reference must contain three finite coordinates")
        self.coordinate_mapping_reference_km = reference.copy()

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
        mapped = hub_centered_coordinates(
            position,
            self.box_size_km,
            self.hub_spacecraft,
            reference_position_km=self.coordinate_mapping_reference_km,
            initial_hub_position_km=(self.initial_hub_position_km if self.is_dhybridr else None),
            wrap=not self.is_dhybridr,
        )
        if self.is_dhybridr:
            # Guard unwrapped x, including every MC query, before the periodic sampler.
            scale = self.normalization["di_km"]
            origin = self.snapshot.layout["origin_di"][0]
            lo, hi = self.snapshot.layout["cell_center_interval_di"][0]
            xmax = (hi - lo) * scale
            x = mapped[..., 0]
            invalid = ~np.isfinite(x) | (x < 0) | (x > xmax)
            if np.any(invalid):
                bad = float(x[invalid][0] / scale + origin)
                raise ValueError(
                    f"dHybridR x={bad:.17g} d_i outside supported cell-center "
                    f"interval [{lo}, {hi}] d_i; x wrapping is forbidden"
                )
        points_time_first = np.swapaxes(mapped, 0, 1)
        grid = getattr(self.snapshot, quantity)
        if self.is_dhybridr:
            # At the final center, floating arithmetic may yield index nx-1+epsilon.
            # Sample that plane with a singleton x axis: no first-cell contribution,
            # no clamping of queries, and the same y/z interpolator and weights.
            last = points_time_first[..., 0] * grid.shape[0] / self.box_size_km[0]
            last = last >= grid.shape[0] - 1
            sampled = np.empty((*last.shape, *grid.shape[3:]))
            sampled[~last] = self.interpolator.sample(grid, points_time_first[~last])
            sampled[last] = self.interpolator.sample(grid[-1:], points_time_first[last])
        else:
            sampled = self.interpolator.sample(grid, points_time_first)
        sampled *= self.scale_factors[quantity]
        if self.fields_are[quantity] == "fluctuation":
            sampled += self.background[quantity]
        return sampled

    # TODO: Add time-dependent snapshots, coordinate rotations, and advecting boxes.
