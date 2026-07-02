"""Self-contained loading of uniform-grid simulation snapshots from NumPy files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_VARIABLES: dict[str, int | tuple[int, int, int]] = {
    "n": 0,
    "v": (1, 2, 3),
    "B": (4, 5, 6),
}


@dataclass(frozen=True)
class SimulationSnapshot:
    """Parsed scalar/vector fields on a periodic uniform grid.

    Scalar ``n`` has shape ``(nx, ny, nz)``. Vector ``v`` and ``B`` have
    shape ``(nx, ny, nz, 3)``. Values remain in snapshot/code units; the field
    adapter applies configured conversions to package units.
    """

    n: np.ndarray
    v: np.ndarray
    B: np.ndarray
    source_shape: tuple[int, ...]
    layout: dict[str, Any]
    available_fields: tuple[str, ...]

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.n.shape)


def _component_selection(value: Any, quantity: str) -> int | tuple[int, int, int]:
    if quantity == "n":
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError("simulation.layout.variables.n must be one component index")
        return int(value)
    array = np.asarray(value)
    if array.shape != (3,) or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(
            f"simulation.layout.variables.{quantity} must contain three component indices"
        )
    return tuple(int(index) for index in array)


def load_simulation_npy(
    path: str | Path,
    layout: dict[str, Any] | None = None,
) -> SimulationSnapshot:
    """Load a combined ``n, v, B`` snapshot with a configurable component axis.

    The default layout is ``(nx, ny, nz, 7)`` ordered as
    ``n, vx, vy, vz, Bx, By, Bz``. ``fields_first`` accepts the equivalent
    ``(7, nx, ny, nz)`` arrangement. Component indices can be overridden by
    ``layout.variables``.
    """

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Simulation snapshot does not exist: {source}")
    if source.suffix.lower() != ".npy":
        raise ValueError(f"Simulation snapshot must use .npy format: {source}")
    config = dict(layout or {})
    array_order = str(config.get("array_order", "fields_last")).lower()
    if array_order not in {"fields_last", "fields_first"}:
        raise ValueError("simulation.layout.array_order must be fields_last or fields_first")
    variables_config = config.get("variables", DEFAULT_VARIABLES)
    if not isinstance(variables_config, dict):
        raise ValueError("simulation.layout.variables must be a mapping")
    unknown = sorted(set(variables_config) - {"n", "v", "B"})
    if unknown:
        raise ValueError(f"Unknown simulation.layout.variables entries: {unknown}")
    if not variables_config:
        raise ValueError("simulation.layout.variables must select at least one field")
    variables = {
        quantity: _component_selection(variables_config[quantity], quantity)
        for quantity in ("n", "v", "B")
        if quantity in variables_config
    }

    source_array = np.load(source, allow_pickle=False)
    if source_array.ndim != 4:
        raise ValueError(
            "Simulation .npy array must be four-dimensional with one component axis; "
            f"got shape {source_array.shape}"
        )
    values = source_array if array_order == "fields_last" else np.moveaxis(source_array, 0, -1)
    component_count = values.shape[-1]
    indices: list[int] = []
    for quantity, selection in variables.items():
        indices.extend([int(selection)] if quantity == "n" else selection)
    if any(index < 0 or index >= component_count for index in indices):
        raise ValueError(
            f"Configured component indices {indices} exceed the component axis of "
            f"shape {source_array.shape}"
        )
    if len(set(indices)) != len(indices):
        raise ValueError("Configured n, v, and B component indices must be distinct")
    grid_shape = values.shape[:3]
    if any(length < 1 for length in grid_shape):
        raise ValueError(f"Simulation spatial axes must be nonempty; got {grid_shape}")
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("Simulation snapshot must contain numeric values")

    n = (
        np.asarray(values[..., variables["n"]], dtype=float)
        if "n" in variables
        else np.zeros(grid_shape, dtype=float)
    )
    v = (
        np.asarray(values[..., list(variables["v"])], dtype=float)
        if "v" in variables
        else np.zeros((*grid_shape, 3), dtype=float)
    )
    B = (
        np.asarray(values[..., list(variables["B"])], dtype=float)
        if "B" in variables
        else np.zeros((*grid_shape, 3), dtype=float)
    )
    if not np.all(np.isfinite(n)) or not np.all(np.isfinite(v)) or not np.all(np.isfinite(B)):
        raise ValueError("Simulation snapshot contains non-finite selected field values")
    normalized_layout = {
        "array_order": array_order,
        "variables": {
            quantity: int(selection) if quantity == "n" else list(selection)
            for quantity, selection in variables.items()
        },
        "missing_fields": [quantity for quantity in ("n", "v", "B") if quantity not in variables],
    }
    return SimulationSnapshot(
        n,
        v,
        B,
        tuple(source_array.shape),
        normalized_layout,
        tuple(quantity for quantity in ("n", "v", "B") if quantity in variables),
    )


class PeriodicTrilinearInterpolator:
    """Trilinear sampler for endpoint-excluded grids on ``[0, L)``."""

    def __init__(self, box_size_km: np.ndarray | tuple[float, float, float]) -> None:
        box = np.asarray(box_size_km, dtype=float)
        if box.shape != (3,) or not np.all(np.isfinite(box)) or np.any(box <= 0.0):
            raise ValueError("box_size_km must contain three positive finite lengths")
        self.box_size_km = box

    def sample(self, grid: np.ndarray, points_km: np.ndarray) -> np.ndarray:
        """Sample a scalar ``(nx,ny,nz)`` or vector ``(nx,ny,nz,c)`` grid."""

        values = np.asarray(grid, dtype=float)
        if values.ndim not in {3, 4}:
            raise ValueError("grid must have shape (nx, ny, nz) or (nx, ny, nz, components)")
        points = np.asarray(points_km, dtype=float)
        if points.ndim < 1 or points.shape[-1] != 3:
            raise ValueError("points_km must have shape (..., 3)")
        if not np.all(np.isfinite(points)):
            raise ValueError("points_km contains non-finite values")
        leading_shape = points.shape[:-1]
        flat = np.mod(points.reshape(-1, 3), self.box_size_km)

        lower = []
        upper = []
        weights = []
        for axis, (length, count) in enumerate(zip(self.box_size_km, values.shape[:3])):
            if count == 1:
                lower.append(np.zeros(flat.shape[0], dtype=np.int64))
                upper.append(np.zeros(flat.shape[0], dtype=np.int64))
                weights.append(np.zeros(flat.shape[0]))
                continue
            scaled = flat[:, axis] * count / length
            floor = np.floor(scaled)
            lower.append(floor.astype(np.int64) % count)
            upper.append((lower[-1] + 1) % count)
            weights.append(scaled - floor)

        lx, ly, lz = lower
        ux, uy, uz = upper
        wx, wy, wz = weights
        trailing_shape = values.shape[3:]
        result = np.zeros((flat.shape[0], *trailing_shape), dtype=float)
        for ix, x_weight in ((lx, 1.0 - wx), (ux, wx)):
            for iy, y_weight in ((ly, 1.0 - wy), (uy, wy)):
                for iz, z_weight in ((lz, 1.0 - wz), (uz, wz)):
                    corner_weight = x_weight * y_weight * z_weight
                    if trailing_shape:
                        corner_weight = corner_weight[:, None]
                    result += corner_weight * values[ix, iy, iz]
        return result.reshape((*leading_shape, *trailing_shape))

    def interpolate_scalar(self, grid: np.ndarray, points_km: np.ndarray) -> np.ndarray:
        return self.sample(grid, points_km)

    def interpolate_vector(self, grid: np.ndarray, points_km: np.ndarray) -> np.ndarray:
        if np.asarray(grid).ndim != 4:
            raise ValueError("Vector grid must have shape (nx, ny, nz, components)")
        return self.sample(grid, points_km)
