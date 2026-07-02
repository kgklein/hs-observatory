"""Read split Gkeyll/ADIOS2 field and ion-moment simulation snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hs_observatory.io.simulation_npy import SimulationSnapshot


DEFAULT_BP_LAYOUT = {
    "field_variable": "CartGridField",
    "moments_variable": "CartGridField",
    "B_components": (3, 4, 5),
    "density_component": 0,
    "momentum_components": (1, 2, 3),
    "velocity_representation": "momentum_density",
}


def _adios2_module():
    try:
        import adios2
    except ImportError as exc:
        raise ImportError(
            "Reading simulation format 'bp' requires ADIOS2. Install the optional "
            "dependency with: python -m pip install 'hs-observatory[simulation-bp]'"
        ) from exc
    return adios2


def _shape_from_metadata(metadata: dict[str, Any]) -> tuple[int, ...] | None:
    raw = metadata.get("Shape")
    if raw is None:
        return None
    if isinstance(raw, str):
        stripped = raw.strip().strip("[]()")
        return tuple(int(value.strip()) for value in stripped.split(",") if value.strip())
    array = np.asarray(raw, dtype=int)
    return tuple(int(value) for value in array.ravel())


def _read_selected_components(
    path: Path,
    variable_name: str,
    component_indices: tuple[int, ...],
) -> tuple[dict[int, np.ndarray], tuple[int, ...], list[str]]:
    adios2 = _adios2_module()
    with adios2.Stream(str(path), "r") as stream:
        available = stream.available_variables()
        if variable_name not in available:
            raise ValueError(
                f"Variable {variable_name!r} is missing from {path}; "
                f"available variables: {sorted(available)}"
            )
        metadata_shape = _shape_from_metadata(available[variable_name])
        if metadata_shape is not None and len(metadata_shape) != 4:
            raise ValueError(
                f"{path}:{variable_name} must be 4D with trailing components; "
                f"metadata reports {metadata_shape}"
            )
        if metadata_shape is not None and any(
            index < 0 or index >= metadata_shape[-1] for index in component_indices
        ):
            raise ValueError(
                f"Requested components {component_indices} exceed {path}:{variable_name} "
                f"shape {metadata_shape}"
            )

        selected: dict[int, np.ndarray] = {}
        if metadata_shape is not None:
            spatial_shape = metadata_shape[:3]
            for index in sorted(set(component_indices)):
                start = [0, 0, 0, index]
                count = [*spatial_shape, 1]
                values = np.asarray(
                    stream.read(variable_name, start=start, count=count), dtype=float
                )
                if values.shape == (*spatial_shape, 1):
                    values = values[..., 0]
                elif values.shape != spatial_shape:
                    raise ValueError(
                        f"Selected component {index} from {path}:{variable_name} returned "
                        f"shape {values.shape}, expected {spatial_shape} or {(*spatial_shape, 1)}"
                    )
                selected[index] = values
            source_shape = metadata_shape
        else:
            packed = np.asarray(stream.read(variable_name), dtype=float)
            if packed.ndim != 4:
                raise ValueError(
                    f"{path}:{variable_name} must have shape (nx, ny, nz, components); "
                    f"got {packed.shape}"
                )
            if any(index < 0 or index >= packed.shape[-1] for index in component_indices):
                raise ValueError(
                    f"Requested components {component_indices} exceed "
                    f"{path}:{variable_name} shape {packed.shape}"
                )
            selected = {index: packed[..., index] for index in set(component_indices)}
            source_shape = tuple(int(value) for value in packed.shape)
        return selected, source_shape, sorted(str(name) for name in available)


def _vector_indices(value: Any, name: str) -> tuple[int, int, int]:
    array = np.asarray(value)
    if array.shape != (3,) or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"simulation.layout.{name} must contain three component indices")
    return tuple(int(index) for index in array)


def load_simulation_bp(
    field_path: str | Path,
    moments_path: str | Path,
    layout: dict[str, Any] | None = None,
) -> SimulationSnapshot:
    """Load density, bulk velocity, and magnetic field from two packed BP files."""

    field_source = Path(field_path).expanduser()
    moments_source = Path(moments_path).expanduser()
    for source, label in ((field_source, "field"), (moments_source, "moments")):
        if not source.is_file():
            raise FileNotFoundError(f"Simulation {label} BP file does not exist: {source}")
        if source.suffix.lower() != ".bp":
            raise ValueError(f"Simulation {label} source must use .bp format: {source}")

    config = dict(DEFAULT_BP_LAYOUT)
    if layout:
        config.update(layout)
    field_variable = str(config["field_variable"])
    moments_variable = str(config["moments_variable"])
    B_indices = _vector_indices(config["B_components"], "B_components")
    momentum_indices = _vector_indices(config["momentum_components"], "momentum_components")
    density_index = int(config["density_component"])
    representation = str(config["velocity_representation"]).lower()
    if representation not in {"momentum_density", "velocity"}:
        raise ValueError(
            "simulation.layout.velocity_representation must be momentum_density or velocity"
        )

    field_components, field_shape, field_variables = _read_selected_components(
        field_source, field_variable, B_indices
    )
    moment_requested = (density_index, *momentum_indices)
    moment_components, moment_shape, moment_variables = _read_selected_components(
        moments_source, moments_variable, moment_requested
    )
    if field_shape[:3] != moment_shape[:3]:
        raise ValueError(
            "Field and ion-moment BP grids must match; got "
            f"{field_shape[:3]} and {moment_shape[:3]}"
        )
    n = np.asarray(moment_components[density_index], dtype=float)
    momentum_or_velocity = np.stack(
        [moment_components[index] for index in momentum_indices], axis=-1
    )
    if representation == "momentum_density":
        if np.any(~np.isfinite(n)) or np.any(n == 0.0):
            raise ValueError(
                "Ion density must be finite and nonzero to convert momentum density to velocity"
            )
        v = momentum_or_velocity / n[..., None]
    else:
        v = momentum_or_velocity
    B = np.stack([field_components[index] for index in B_indices], axis=-1)
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(B)):
        raise ValueError("Selected BP density, velocity, or magnetic values are non-finite")

    normalized_layout = {
        "schema": "split_gkeyll_bp",
        "field_variable": field_variable,
        "moments_variable": moments_variable,
        "B_components": list(B_indices),
        "density_component": density_index,
        "momentum_components": list(momentum_indices),
        "velocity_representation": representation,
        "field_source_shape": list(field_shape),
        "moments_source_shape": list(moment_shape),
        "field_available_variables": field_variables,
        "moments_available_variables": moment_variables,
    }
    return SimulationSnapshot(
        n=n,
        v=v,
        B=B,
        source_shape=field_shape,
        layout=normalized_layout,
        available_fields=("n", "v", "B"),
    )
