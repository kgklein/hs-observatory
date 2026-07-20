"""Component-wise scalar RBF reconstruction of magnetic-field measurements.

This prototype assumes the magnetic field is static during the selected time
window and fits only to synthetic spacecraft measurements: unwrapped physical
spacecraft positions and sampled ``B`` values. The fitter deliberately has no
path for loading or querying the original simulation cube. Reference fields may
be added by external validation code later, but they are not part of the model.

Divergence-free vector RBF reconstruction is intentionally left for a future
implementation; this module reconstructs ``Bx``, ``By``, and ``Bz``
independently with the Broeren et al. (2024) multiquadric RBF kernel

    phi(d; sigma) = sqrt(1 + (d / sigma)^2).

The tunable ``sigma`` is a physical length scale in the same unwrapped units as
the spacecraft positions. Rippa-style leave-one-out cross-validation can select
``sigma`` from spacecraft measurements only; simulation-cube access remains in
separate validation helpers outside the fitter.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import warnings
from typing import Any

import h5py
import numpy as np

from hs_observatory.analysis.base import AnalysisMethod
from hs_observatory.analysis.registry import register_analysis
from hs_observatory.config import load_generation_config
from hs_observatory.fields.simulation_sampled import SimulationSampledField
from hs_observatory.visualization import SPACECRAFT_COLORS

COMPONENT_NAMES = ("Bx", "By", "Bz")
MAGNITUDE_CMAP_NAMES = {"batlow", "lajolla", "magma", "inferno", "plasma", "viridis", "cividis"}
ERROR_CMAP_NAMES = {"imola", "lajolla", "devon", "magma", "inferno", "plasma", "YlOrRd"}
COMPONENT_MODES = {"raw", "delta"}
BACKGROUND_MODES = {
    "metadata",
    "actual-volume-mean",
    "reconstructed-volume-mean",
    "user",
}


@dataclass(frozen=True)
class CylindricalGrid:
    """Local Cartesian grid clipped to a cylinder around the barycenter path."""

    points: np.ndarray
    axial: np.ndarray
    transverse_u: np.ndarray
    transverse_v: np.ndarray
    mask: np.ndarray
    full_points: np.ndarray
    barycenter_path: np.ndarray
    center: np.ndarray
    e_parallel: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    axial_values: np.ndarray
    transverse_values: np.ndarray
    radius: float
    requested_radius: float
    radius_padding: float
    max_spacecraft_transverse_radius: float
    require_all_spacecraft_coverage: bool
    axial_coverage_min: float
    axial_coverage_max: float
    n_axial: int
    n_transverse: int


def _as_2d_vector_array(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _normalize(vector: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError(f"Cannot normalize {name}")
    return vector / norm


def _perpendicular_basis(e_parallel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    trial = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(e_parallel, trial))) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    e1 = _normalize(np.cross(e_parallel, trial), "first perpendicular basis vector")
    e2 = _normalize(np.cross(e_parallel, e1), "second perpendicular basis vector")
    return e1, e2


def _automatic_length_scale(positions: np.ndarray) -> float:
    if positions.shape[0] < 2:
        raise ValueError("At least two unique samples are required to estimate an RBF length scale")
    offsets = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(offsets, axis=-1)
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    positive = nearest[np.isfinite(nearest) & (nearest > 0.0)]
    if positive.size == 0:
        raise ValueError("RBF sample positions must contain at least two distinct points")
    return float(np.median(positive))


def _pairwise_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.linalg.norm(left[:, None, :] - right[None, :, :], axis=-1)


def _multiquadric_matrix(
    left: np.ndarray,
    right: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Return Broeren/Rippa multiquadric phi(d)=sqrt(1+(d/sigma)^2)."""

    if sigma <= 0.0 or not np.isfinite(sigma):
        raise ValueError("RBF sigma must be positive and finite")
    return np.sqrt(1.0 + (_pairwise_distances(left, right) / sigma) ** 2)


def _solve_coefficients(matrix: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, str]:
    try:
        return np.linalg.solve(matrix, values), "solve"
    except np.linalg.LinAlgError:
        warnings.warn(
            "RBF interpolation matrix is singular; falling back to least squares",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.linalg.lstsq(matrix, values, rcond=None)[0], "lstsq"


def _sigma_candidates_from_config(
    positions: np.ndarray,
    *,
    sigma_min: float | None,
    sigma_max: float | None,
    sigma_num: int,
    sigma_grid: str,
) -> np.ndarray:
    characteristic = _automatic_length_scale(positions)
    lower = characteristic / 4.0 if sigma_min is None else float(sigma_min)
    upper = characteristic * 4.0 if sigma_max is None else float(sigma_max)
    count = int(sigma_num)
    if count < 1:
        raise ValueError("sigma_num must be positive")
    if lower <= 0.0 or upper <= 0.0 or upper < lower:
        raise ValueError("sigma_min and sigma_max must be positive with sigma_max >= sigma_min")
    if sigma_grid == "linear":
        return np.linspace(lower, upper, count)
    if sigma_grid == "log":
        return np.geomspace(lower, upper, count)
    raise ValueError("sigma_grid must be 'log' or 'linear'")


def _component_indices(components: str | list[str] | tuple[str, ...]) -> list[int]:
    labels = ("bx", "by", "bz")
    if isinstance(components, str):
        requested = labels if components.lower() == "all" else (components.lower(),)
    else:
        requested = tuple(str(component).lower() for component in components)
    unknown = sorted(set(requested) - set(labels))
    if unknown:
        raise ValueError(f"Unknown Rippa component labels: {unknown}")
    return [labels.index(component) for component in requested]


def rippa_loocv_cost(
    positions: np.ndarray,
    values: np.ndarray,
    sigma: float,
    *,
    smoothing: float = 0.0,
) -> float:
    """Return Rippa (1999) RMS leave-one-out residual estimate.

    For a candidate sigma, solve ``A c = f`` and estimate leave-one-out
    residuals with ``e_i = c_i / (A^{-1})_ii``. With nonzero smoothing, this
    uses the regularized matrix ``A + smoothing I`` consistently for both
    coefficients and diagonal inverse terms.
    """

    sample_positions = _as_2d_vector_array(positions, "positions")
    sample_values = np.asarray(values, dtype=float)
    if sample_values.ndim != 1 or sample_values.shape[0] != sample_positions.shape[0]:
        raise ValueError("values must have shape (N,)")
    if not np.all(np.isfinite(sample_values)):
        raise ValueError("values contains non-finite entries")
    matrix = _multiquadric_matrix(sample_positions, sample_positions, float(sigma))
    if smoothing < 0.0:
        raise ValueError("smoothing must be non-negative")
    if smoothing:
        matrix = matrix + float(smoothing) * np.eye(matrix.shape[0])
    coefficients, _solver = _solve_coefficients(matrix, sample_values)
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        warnings.warn(
            "RBF interpolation matrix inverse failed for Rippa cost; using pseudo-inverse",
            RuntimeWarning,
            stacklevel=2,
        )
        inverse = np.linalg.pinv(matrix)
    denominator = np.diag(inverse)
    residual = np.divide(
        coefficients,
        denominator,
        out=np.full_like(coefficients, np.nan),
        where=np.abs(denominator) > 0.0,
    )
    finite = residual[np.isfinite(residual)]
    if finite.size == 0:
        return float("inf")
    return float(np.sqrt(np.mean(finite**2)))


def choose_sigma_rippa(
    positions: np.ndarray,
    B: np.ndarray,
    sigma_candidates: np.ndarray,
    *,
    smoothing: float = 0.0,
    mode: str = "field",
    components: str | list[str] | tuple[str, ...] = "all",
) -> dict[str, Any]:
    """Choose sigma from candidates using field Rippa cost or geometry heuristic."""

    sample_positions = _as_2d_vector_array(positions, "positions")
    sample_B = _as_2d_vector_array(B, "B")
    candidates = np.asarray(sigma_candidates, dtype=float)
    if candidates.ndim != 1 or candidates.size == 0:
        raise ValueError("sigma_candidates must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(candidates)) or np.any(candidates <= 0.0):
        raise ValueError("sigma_candidates must be positive and finite")
    normalized_mode = str(mode).lower()
    if normalized_mode == "geometry":
        characteristic = _automatic_length_scale(sample_positions)
        costs = np.abs(np.log(candidates / characteristic))
        method = "geometry_nearest_neighbor_heuristic"
    elif normalized_mode == "field":
        indices = _component_indices(components)
        component_costs = np.empty((candidates.size, len(indices)), dtype=float)
        for sigma_index, sigma in enumerate(candidates):
            for column, component_index in enumerate(indices):
                component_costs[sigma_index, column] = rippa_loocv_cost(
                    sample_positions,
                    sample_B[:, component_index],
                    float(sigma),
                    smoothing=smoothing,
                )
        costs = np.sqrt(np.mean(component_costs**2, axis=1))
        method = "rippa_loocv_field"
    else:
        raise ValueError("rippa_mode must be 'field' or 'geometry'")
    best_index = int(np.nanargmin(costs))
    return {
        "selected_sigma": float(candidates[best_index]),
        "sigma_candidates": candidates,
        "costs": costs,
        "mode": normalized_mode,
        "components": components,
        "method": method,
        "best_index": best_index,
    }
    

def explicit_loocv_cost(
    positions: np.ndarray,
    B: np.ndarray,
    sigma: float,
    *,
    smoothing: float = 0.0,
) -> float:
    """Return Teddy/Broeren-style explicit vector leave-one-out residual norm."""

    sample_positions = _as_2d_vector_array(positions, "positions")
    sample_B = _as_2d_vector_array(B, "B")
    if sample_positions.shape[0] < 4:
        raise ValueError("explicit LOOCV requires at least four samples")
    residual = np.empty_like(sample_B)
    for index in range(sample_positions.shape[0]):
        keep = np.ones(sample_positions.shape[0], dtype=bool)
        keep[index] = False
        matrix = _multiquadric_matrix(
            sample_positions[keep], sample_positions[keep], float(sigma)
        )
        if smoothing:
            matrix = matrix + float(smoothing) * np.eye(matrix.shape[0])
        coefficients, _solver = _solve_coefficients(matrix, sample_B[keep])
        phi = _multiquadric_matrix(
            sample_positions[index : index + 1], sample_positions[keep], float(sigma)
        )
        residual[index] = sample_B[index] - (phi @ coefficients)[0]
    return float(np.linalg.norm(residual))


def choose_sigma_explicit_loocv(
    positions: np.ndarray,
    B: np.ndarray,
    *,
    sigma_initial: float = 100.0,
    sigma_min: float | None = None,
    sigma_max: float | None = None,
    smoothing: float = 0.0,
    max_iter: int = 40,
    tolerance: float = 1.0e-3,
) -> dict[str, Any]:
    """Choose sigma by continuous explicit LOOCV optimization."""

    sample_positions = _as_2d_vector_array(positions, "positions")
    sample_B = _as_2d_vector_array(B, "B")
    initial = float(sigma_initial)
    if initial <= 0.0 or not np.isfinite(initial):
        raise ValueError("sigma_initial must be positive and finite")
    lower = initial / 100.0 if sigma_min is None else float(sigma_min)
    upper = initial * 100.0 if sigma_max is None else float(sigma_max)
    if lower <= 0.0 or upper <= lower:
        raise ValueError("explicit LOOCV sigma bounds must satisfy 0 < min < max")

    evaluated: dict[float, float] = {}

    def cost(sigma: float) -> float:
        key = float(sigma)
        if key <= 0.0 or not np.isfinite(key):
            return float("inf")
        if key not in evaluated:
            evaluated[key] = explicit_loocv_cost(
                sample_positions, sample_B, key, smoothing=smoothing
            )
        return evaluated[key]

    optimizer = "scipy_minimize_log_sigma"
    try:
        from scipy.optimize import minimize
    except ModuleNotFoundError:
        optimizer = "fallback_log_grid"
        for sigma in np.geomspace(lower, upper, max(8, int(max_iter))):
            cost(float(sigma))
    else:

        def objective(log_sigma: np.ndarray) -> float:
            sigma = float(np.exp(np.ravel(log_sigma)[0]))
            return cost(sigma)

        log_initial = np.array([np.log(initial)], dtype=float)
        options = {"maxiter": int(max_iter), "gtol": float(tolerance)}
        if sigma_min is None and sigma_max is None:
            result = minimize(objective, log_initial, method="CG", options=options)
            optimizer = "scipy_cg_log_sigma"
        else:
            bounds = [(np.log(lower), np.log(upper))]
            result = minimize(
                objective,
                log_initial,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": int(max_iter), "ftol": float(tolerance)},
            )
            optimizer = "scipy_lbfgsb_log_sigma"
        cost(float(np.exp(np.ravel(result.x)[0])))
    cost(initial)
    sigmas = np.asarray(sorted(evaluated), dtype=float)
    costs = np.asarray([evaluated[float(sigma)] for sigma in sigmas], dtype=float)
    best_index = int(np.nanargmin(costs))
    return {
        "selected_sigma": float(sigmas[best_index]),
        "sigma_candidates": sigmas,
        "costs": costs,
        "mode": "explicit_loocv",
        "components": "all",
        "method": "explicit_loocv_vector_norm",
        "best_index": best_index,
        "sigma_initial": initial,
        "sigma_min": lower,
        "sigma_max": upper,
        "optimizer": optimizer,
    }


def _subsample_sigma_training_data(
    selected_positions: np.ndarray,
    selected_B: np.ndarray,
    *,
    max_samples: int | None,
    random_seed: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subsample by time while keeping all spacecraft for selected times."""

    positions = np.asarray(selected_positions, dtype=float)
    magnetic = np.asarray(selected_B, dtype=float)
    if positions.shape != magnetic.shape or positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("selected_positions and selected_B must both have shape (time, sc, 3)")
    n_time, n_spacecraft, _components = positions.shape
    if max_samples is None or n_time * n_spacecraft <= int(max_samples):
        indices = np.arange(n_time, dtype=int)
    else:
        max_times = max(1, int(max_samples) // n_spacecraft)
        rng = np.random.default_rng(random_seed)
        indices = np.sort(rng.choice(n_time, size=min(max_times, n_time), replace=False))
    return indices, positions[indices].reshape(-1, 3), magnetic[indices].reshape(-1, 3)


SIGMA_CACHE_CODE_VERSION = "multiquadric_sigma_selection_v3"


def _normalized_components(value: str | list[str] | tuple[str, ...]) -> str:
    if isinstance(value, str):
        return value.lower()
    return ",".join(str(item).lower() for item in value)


def _hash_array(values: np.ndarray, tolerance: float | None = None) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if tolerance is not None:
        tolerance = float(tolerance)
        if tolerance <= 0.0:
            raise ValueError("hash_tolerance must be positive when provided")
        array = np.round(array / tolerance) * tolerance
        array = np.ascontiguousarray(array, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _hash_json(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_sigma_cache_key(
    sample_positions: np.ndarray,
    *,
    selected_times: np.ndarray | None = None,
    spacecraft_ids: list[str] | tuple[str, ...] | None = None,
    t0: float | None = None,
    delta_t: float | None = None,
    kernel: str = "multiquadric",
    smoothing: float = 0.0,
    sigma_candidates: np.ndarray | None = None,
    sigma_selection: str = "rippa_grid",
    sigma_initial: float | None = None,
    sigma_min: float | None = None,
    sigma_max: float | None = None,
    sigma_subsample_max: int | None = None,
    sigma_random_seed: int | None = None,
    rippa_mode: str = "field",
    rippa_components: str | list[str] | tuple[str, ...] = "all",
    coordinate_units: str = "km",
    hash_tolerance: float | None = None,
) -> dict[str, Any]:
    """Build a stable cache key from unwrapped sample positions and settings."""

    positions = _as_2d_vector_array(sample_positions, "sample_positions")
    candidates = (
        np.asarray([], dtype=float)
        if sigma_candidates is None
        else np.asarray(sigma_candidates, dtype=float)
    )
    if candidates.ndim != 1:
        raise ValueError("sigma_candidates must be a one-dimensional array")
    position_hash = _hash_array(positions, hash_tolerance)
    time_hash = (
        None
        if selected_times is None
        else _hash_array(np.asarray(selected_times, dtype=float), hash_tolerance)
    )
    settings = {
        "spacecraft_ids": list(spacecraft_ids or ()),
        "selected_time_hash": time_hash,
        "t0": None if t0 is None else float(t0),
        "delta_t": None if delta_t is None else float(delta_t),
        "kernel": str(kernel).lower(),
        "smoothing": float(smoothing),
        "sigma_candidates": candidates.tolist(),
        "sigma_selection": str(sigma_selection).lower(),
        "sigma_initial": None if sigma_initial is None else float(sigma_initial),
        "sigma_min": None if sigma_min is None else float(sigma_min),
        "sigma_max": None if sigma_max is None else float(sigma_max),
        "sigma_subsample_max": (
            None if sigma_subsample_max is None else int(sigma_subsample_max)
        ),
        "sigma_random_seed": (
            None if sigma_random_seed is None else int(sigma_random_seed)
        ),
        "rippa_mode": str(rippa_mode).lower(),
        "rippa_components": _normalized_components(rippa_components),
        "coordinate_units": str(coordinate_units),
        "code_version": SIGMA_CACHE_CODE_VERSION,
        "hash_tolerance": hash_tolerance,
    }
    settings_hash = _hash_json(settings)
    combined_hash = _hash_json(
        {"position_hash": position_hash, "settings_hash": settings_hash}
    )
    return {
        "cache_key": combined_hash,
        "position_hash": position_hash,
        "settings_hash": settings_hash,
        "settings": settings,
    }


def _cache_group(handle: h5py.File, cache_key: str) -> h5py.Group | None:
    path = f"sigma_cache/{cache_key}"
    return handle[path] if path in handle else None


def load_sigma_from_cache(
    cache_path: str | Path,
    cache_key_info: dict[str, Any],
) -> dict[str, Any] | None:
    """Load a matching sigma cache entry after validating key metadata."""

    path = Path(cache_path)
    if not path.is_file():
        return None
    cache_key = str(cache_key_info["cache_key"])
    with h5py.File(path, "r") as handle:
        group = _cache_group(handle, cache_key)
        if group is None:
            return None
        expected = cache_key_info["settings"]
        checks = {
            "position_hash": cache_key_info["position_hash"],
            "settings_hash": cache_key_info["settings_hash"],
            "kernel": expected["kernel"],
            "sigma_selection": expected["sigma_selection"],
            "rippa_mode": expected["rippa_mode"],
            "rippa_components": expected["rippa_components"],
            "coordinate_units": expected["coordinate_units"],
            "code_version": expected["code_version"],
        }
        for key, expected_value in checks.items():
            actual = group.attrs.get(key)
            if actual != expected_value:
                return None
        if int(group.attrs.get("n_samples", -1)) != int(
            np.asarray(cache_key_info["sample_positions"]).shape[0]
        ):
            return None
        if float(group.attrs.get("smoothing", np.nan)) != expected["smoothing"]:
            return None
        cached_candidates = np.asarray(group["sigma_candidates"], dtype=float)
        expected_candidates = np.asarray(expected["sigma_candidates"], dtype=float)
        if expected["sigma_selection"] == "rippa_grid" and not np.array_equal(
            cached_candidates, expected_candidates
        ):
            return None
        return {
            "selected_sigma": float(group.attrs["selected_sigma"]),
            "sigma_candidates": cached_candidates,
            "costs": np.asarray(group["loocv_costs"], dtype=float),
            "mode": str(group.attrs["rippa_mode"]),
            "components": str(group.attrs["rippa_components"]),
            "method": str(group.attrs["method"]),
            "best_index": int(group.attrs["best_index"]),
            "source": "cache",
            "cache_key": cache_key,
            "cache_path": str(path),
        }


def save_sigma_to_cache(
    cache_path: str | Path,
    cache_key_info: dict[str, Any],
    selection: dict[str, Any],
    metadata: dict[str, Any],
) -> Path:
    """Write selected sigma and diagnostic arrays to an HDF5 sigma cache."""

    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cache_key = str(cache_key_info["cache_key"])
    with h5py.File(path, "a") as handle:
        root = handle.require_group("sigma_cache")
        if cache_key in root:
            del root[cache_key]
        group = root.create_group(cache_key)
        group.create_dataset(
            "sigma_candidates",
            data=np.asarray(selection["sigma_candidates"], dtype=float),
        )
        group.create_dataset("loocv_costs", data=np.asarray(selection["costs"], dtype=float))
        sample_positions = np.asarray(metadata["sample_positions"], dtype=float)
        group.create_dataset("sample_position_min", data=np.min(sample_positions, axis=0))
        group.create_dataset("sample_position_max", data=np.max(sample_positions, axis=0))
        for key, value in {
            "selected_sigma": float(selection["selected_sigma"]),
            "kernel": cache_key_info["settings"]["kernel"],
            "smoothing": float(cache_key_info["settings"]["smoothing"]),
            "sigma_selection": cache_key_info["settings"]["sigma_selection"],
            "sigma_initial": cache_key_info["settings"]["sigma_initial"],
            "sigma_min": cache_key_info["settings"]["sigma_min"],
            "sigma_max": cache_key_info["settings"]["sigma_max"],
            "sigma_subsample_max": cache_key_info["settings"]["sigma_subsample_max"],
            "sigma_random_seed": cache_key_info["settings"]["sigma_random_seed"],
            "rippa_mode": cache_key_info["settings"]["rippa_mode"],
            "rippa_components": cache_key_info["settings"]["rippa_components"],
            "t0": metadata.get("t0"),
            "delta_t": metadata.get("delta_t"),
            "n_samples": sample_positions.shape[0],
            "n_spacecraft": metadata.get("n_spacecraft"),
            "position_hash": cache_key_info["position_hash"],
            "settings_hash": cache_key_info["settings_hash"],
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "code_version": SIGMA_CACHE_CODE_VERSION,
            "coordinate_units": cache_key_info["settings"]["coordinate_units"],
            "method": selection.get("method"),
            "best_index": selection.get("best_index"),
            "cache_key": cache_key,
            "notes": metadata.get("notes", ""),
        }.items():
            if value is not None:
                group.attrs[key] = value
        for name in ("barycenter_start", "barycenter_end"):
            if name in metadata:
                group.create_dataset(name, data=np.asarray(metadata[name], dtype=float))
        if "sigma_time_indices" in metadata:
            group.create_dataset(
                "sigma_time_indices", data=np.asarray(metadata["sigma_time_indices"], dtype=int)
            )
    return path


def get_or_compute_sigma(
    sample_positions: np.ndarray,
    sample_B: np.ndarray,
    sigma_candidates: np.ndarray | None,
    *,
    selection_positions: np.ndarray | None = None,
    selection_B: np.ndarray | None = None,
    cache_path: str | Path | None = None,
    use_cache: bool = True,
    force_recompute: bool = False,
    save_cache: bool = True,
    selected_times: np.ndarray | None = None,
    spacecraft_ids: list[str] | tuple[str, ...] | None = None,
    t0: float | None = None,
    delta_t: float | None = None,
    kernel: str = "multiquadric",
    smoothing: float = 0.0,
    sigma_selection: str = "rippa_grid",
    sigma_initial: float | None = None,
    sigma_min: float | None = None,
    sigma_max: float | None = None,
    sigma_subsample_max: int | None = None,
    sigma_random_seed: int | None = None,
    rippa_mode: str = "field",
    rippa_components: str | list[str] | tuple[str, ...] = "all",
    coordinate_units: str = "km",
    hash_tolerance: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return cached sigma when possible, otherwise compute and optionally save it."""

    positions = _as_2d_vector_array(sample_positions, "sample_positions")
    fit_positions = (
        positions
        if selection_positions is None
        else _as_2d_vector_array(selection_positions, "selection_positions")
    )
    fit_B = (
        _as_2d_vector_array(sample_B, "sample_B")
        if selection_B is None
        else _as_2d_vector_array(selection_B, "selection_B")
    )
    if fit_positions.shape[0] != fit_B.shape[0]:
        raise ValueError("selection_positions and selection_B must have the same sample count")
    cache_key_info = compute_sigma_cache_key(
        positions,
        selected_times=selected_times,
        spacecraft_ids=spacecraft_ids,
        t0=t0,
        delta_t=delta_t,
        kernel=kernel,
        smoothing=smoothing,
        sigma_candidates=sigma_candidates,
        sigma_selection=sigma_selection,
        sigma_initial=sigma_initial,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sigma_subsample_max=sigma_subsample_max,
        sigma_random_seed=sigma_random_seed,
        rippa_mode=rippa_mode,
        rippa_components=rippa_components,
        coordinate_units=coordinate_units,
        hash_tolerance=hash_tolerance,
    )
    cache_key_info["sample_positions"] = positions
    if cache_path is not None and use_cache and not force_recompute:
        cached = load_sigma_from_cache(cache_path, cache_key_info)
        if cached is not None:
            print(
                "Using cached RBF sigma "
                f"{cached['selected_sigma']} from {cache_path} "
                f"(key {cached['cache_key'][:12]})"
            )
            return cached

    normalized_selection = str(sigma_selection).lower()
    if normalized_selection == "explicit_loocv":
        selection = choose_sigma_explicit_loocv(
            fit_positions,
            fit_B,
            sigma_initial=100.0 if sigma_initial is None else float(sigma_initial),
            sigma_min=None if sigma_min is None else float(sigma_min),
            sigma_max=None if sigma_max is None else float(sigma_max),
            smoothing=smoothing,
        )
    elif normalized_selection == "rippa_grid":
        selection = choose_sigma_rippa(
            fit_positions,
            fit_B,
            np.asarray(sigma_candidates, dtype=float),
            smoothing=smoothing,
            mode=rippa_mode,
            components=rippa_components,
        )
    else:
        raise ValueError("sigma_selection must be 'rippa_grid' or 'explicit_loocv'")
    selection.update(
        {
            "source": (
                "computed_explicit_loocv"
                if normalized_selection == "explicit_loocv"
                else "computed_rippa"
            ),
            "cache_key": cache_key_info["cache_key"],
            "cache_path": None if cache_path is None else str(cache_path),
        }
    )
    print(
        "Computed RBF sigma "
        f"{selection['selected_sigma']} (key {selection['cache_key'][:12]})"
    )
    if cache_path is not None and save_cache:
        cache_metadata = dict(metadata or {})
        cache_metadata["sample_positions"] = positions
        save_sigma_to_cache(cache_path, cache_key_info, selection, cache_metadata)
    return selection


class ComponentwiseRBFReconstructor:
    """Fit independent scalar RBF interpolants for ``Bx``, ``By``, and ``Bz``.

    Parameters
    ----------
    kernel:
        Currently only ``multiquadric`` is implemented, with
        ``phi(d)=sqrt(1+(d/sigma)^2)`` from Broeren et al. (2024).
    sigma:
        Shape/length-scale parameter in the same physical units as positions.
        If omitted, a geometry-based median-nearest-neighbor heuristic is used
        and recorded in metadata.
    smoothing:
        Non-negative diagonal regularization. ``0`` performs exact interpolation
        when the system is well-conditioned.
    """

    def __init__(
        self,
        kernel: str = "multiquadric",
        sigma: float | None = None,
        smoothing: float = 0.0,
        auto_sigma: bool = False,
        sigma_candidates: np.ndarray | None = None,
        rippa_mode: str = "field",
        rippa_components: str | list[str] | tuple[str, ...] = "all",
        length_scale: float | None = None,
    ) -> None:
        self.kernel = str(kernel).lower()
        if self.kernel != "multiquadric":
            raise NotImplementedError("Only kernel='multiquadric' is currently implemented")
        if sigma is None and length_scale is not None:
            warnings.warn(
                "rbf_reconstruction.length_scale is deprecated; use sigma instead",
                DeprecationWarning,
                stacklevel=2,
            )
            sigma = length_scale
        self.sigma = None if sigma is None else float(sigma)
        self.smoothing = float(smoothing)
        self.auto_sigma = bool(auto_sigma)
        self.sigma_candidates = sigma_candidates
        self.rippa_mode = rippa_mode
        self.rippa_components = rippa_components
        self.selected_sigma_: float | None = None
        self.condition_number_: float | None = None
        self.solver_: str | None = None
        self.sigma_selection_: dict[str, Any] | None = None
        self._positions: np.ndarray | None = None
        self._coefficients: np.ndarray | None = None

    def fit(self, positions: np.ndarray, B: np.ndarray) -> "ComponentwiseRBFReconstructor":
        positions = _as_2d_vector_array(positions, "positions")
        magnetic = _as_2d_vector_array(B, "B")
        if positions.shape[0] != magnetic.shape[0]:
            raise ValueError("positions and B must contain the same number of samples")
        if positions.shape[0] < 4:
            raise ValueError("RBF reconstruction requires at least four samples")
        if np.unique(positions, axis=0).shape[0] < 4:
            raise ValueError("RBF reconstruction requires at least four distinct sample positions")
        if self.smoothing < 0.0:
            raise ValueError("RBF smoothing must be non-negative")
        if self.auto_sigma:
            if self.sigma_candidates is None:
                raise ValueError("auto_sigma requires sigma_candidates")
            self.sigma_selection_ = choose_sigma_rippa(
                positions,
                magnetic,
                np.asarray(self.sigma_candidates, dtype=float),
                smoothing=self.smoothing,
                mode=self.rippa_mode,
                components=self.rippa_components,
            )
            sigma = self.sigma_selection_["selected_sigma"]
        elif self.sigma is None:
            sigma = _automatic_length_scale(positions)
            warnings.warn(
                f"No RBF sigma supplied; using geometry heuristic sigma={sigma:.6g}. "
                "Sigma is important and should be checked.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.sigma_selection_ = {
                "selected_sigma": sigma,
                "method": "geometry_nearest_neighbor_heuristic",
                "mode": "geometry",
            }
        else:
            sigma = self.sigma
        if sigma <= 0.0 or not np.isfinite(sigma):
            raise ValueError("RBF sigma must be positive and finite")

        matrix = _multiquadric_matrix(positions, positions, float(sigma))
        if self.smoothing:
            matrix = matrix + self.smoothing * np.eye(matrix.shape[0])
        self.condition_number_ = float(np.linalg.cond(matrix))
        self._coefficients, self.solver_ = _solve_coefficients(matrix, magnetic)
        self._positions = positions
        self.selected_sigma_ = float(sigma)
        return self

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        points = _as_2d_vector_array(points, "points")
        if self._positions is None or self._coefficients is None or self.selected_sigma_ is None:
            raise RuntimeError("ComponentwiseRBFReconstructor must be fit before evaluate")
        phi = _multiquadric_matrix(points, self._positions, self.selected_sigma_)
        return phi @ self._coefficients

    def get_metadata(self) -> dict[str, Any]:
        return {
            "kernel": self.kernel,
            "sigma": self.selected_sigma_,
            "smoothing": self.smoothing,
            "condition_number": self.condition_number_,
            "solver": self.solver_,
            "implementation": "direct_multiquadric_solver",
            "sigma_selection": self.sigma_selection_,
        }


def build_cylindrical_grid(
    spacecraft_positions: np.ndarray,
    times: np.ndarray,
    *,
    radius: float,
    n_axial: int,
    n_transverse: int,
    axial_padding: float = 0.0,
    radius_padding: float = 0.0,
    require_all_spacecraft_coverage: bool = False,
) -> CylindricalGrid:
    """Build a local grid in a cylinder around the selected barycenter path.

    ``spacecraft_positions`` must use unwrapped physical coordinates with shape
    ``(time, spacecraft, 3)``. No periodic wrapping or modulo operation is
    applied here or in the RBF fitter.
    """

    positions = np.asarray(spacecraft_positions, dtype=float)
    selected_times = np.asarray(times, dtype=float)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("spacecraft_positions must have shape (time, spacecraft, 3)")
    if positions.shape[0] != selected_times.size:
        raise ValueError("times length must match the position time axis")
    if positions.shape[0] < 1 or positions.shape[1] < 1:
        raise ValueError("At least one time and one spacecraft are required")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(selected_times)):
        raise ValueError("Grid inputs contain non-finite values")
    requested_radius = float(radius)
    if requested_radius <= 0.0:
        raise ValueError("cylinder radius must be positive")
    n_axial = int(n_axial)
    n_transverse = int(n_transverse)
    if n_axial < 1 or n_transverse < 2:
        raise ValueError("n_axial must be at least 1 and n_transverse at least 2")
    axial_padding = float(axial_padding)
    if axial_padding < 0.0:
        raise ValueError("axial_padding must be non-negative")
    radius_padding = float(radius_padding)
    if radius_padding < 0.0:
        raise ValueError("radius_padding must be non-negative")

    barycenter_path = np.mean(positions, axis=1)
    displacement = barycenter_path[-1] - barycenter_path[0]
    if np.linalg.norm(displacement) <= 1.0e-12:
        warnings.warn(
            "Barycenter displacement is too small for a cylinder axis; using x-axis",
            RuntimeWarning,
            stacklevel=2,
        )
        e_parallel = np.array([1.0, 0.0, 0.0])
    else:
        e_parallel = _normalize(displacement, "barycenter displacement")
    e1, e2 = _perpendicular_basis(e_parallel)

    center = np.mean(barycenter_path, axis=0)
    basis = np.stack((e_parallel, e1, e2), axis=0)
    spacecraft_local = np.einsum("tsc,lc->tsl", positions - center, basis)
    spacecraft_axial = spacecraft_local[..., 0]
    transverse_radius = np.linalg.norm(spacecraft_local[..., 1:3], axis=-1)
    max_spacecraft_transverse_radius = float(np.max(transverse_radius))
    radius = max(
        requested_radius,
        max_spacecraft_transverse_radius + radius_padding,
    )
    axial_coverage_min = float(np.max(np.min(spacecraft_axial, axis=0)))
    axial_coverage_max = float(np.min(np.max(spacecraft_axial, axis=0)))
    if axial_coverage_max < axial_coverage_min:
        raise ValueError(
            "Selected spacecraft trajectories have no common axial coverage; "
            "disable require_all_spacecraft_coverage or widen the time window"
        )
    if require_all_spacecraft_coverage:
        axial_min = axial_coverage_min
        axial_max = axial_coverage_max
    else:
        axial_min = float(np.min(spacecraft_axial)) - axial_padding
        axial_max = float(np.max(spacecraft_axial)) + axial_padding
    if axial_min == axial_max:
        axial_min -= axial_padding
        axial_max += axial_padding
    axial_values = np.linspace(axial_min, axial_max, n_axial)
    transverse_values = np.linspace(-radius, radius, n_transverse)
    axial, transverse_u, transverse_v = np.meshgrid(
        axial_values, transverse_values, transverse_values, indexing="ij"
    )
    mask = np.sqrt(transverse_u**2 + transverse_v**2) <= radius
    full_grid = (
        center
        + axial[..., None] * e_parallel
        + transverse_u[..., None] * e1
        + transverse_v[..., None] * e2
    )
    return CylindricalGrid(
        points=full_grid[mask],
        axial=axial[mask],
        transverse_u=transverse_u[mask],
        transverse_v=transverse_v[mask],
        mask=mask,
        full_points=full_grid,
        barycenter_path=barycenter_path,
        center=center,
        e_parallel=e_parallel,
        e1=e1,
        e2=e2,
        axial_values=axial_values,
        transverse_values=transverse_values,
        radius=radius,
        requested_radius=requested_radius,
        radius_padding=radius_padding,
        max_spacecraft_transverse_radius=max_spacecraft_transverse_radius,
        require_all_spacecraft_coverage=bool(require_all_spacecraft_coverage),
        axial_coverage_min=axial_coverage_min,
        axial_coverage_max=axial_coverage_max,
        n_axial=n_axial,
        n_transverse=n_transverse,
    )


def _settings(config: dict[str, Any]) -> dict[str, Any]:
    t0 = config.get("t0")
    t0_index = config.get("t0_index")
    if (t0 is None) == (t0_index is None):
        raise ValueError("rbf_reconstruction requires exactly one of t0 or t0_index")
    delta_t = float(config["delta_t"])
    if delta_t < 0.0:
        raise ValueError("rbf_reconstruction.delta_t must be non-negative")
    data_group = str(config.get("data_group", "truth")).lower()
    if data_group in {"mc", "monte_carlo", "measurement", "measurements", "noisy"}:
        data_group = "observed"
    if data_group not in {"truth", "observed"}:
        raise ValueError(
            "rbf_reconstruction.data_group must be 'truth' or 'observed'"
        )
    mc_index_config = config.get("mc_index", 0)
    if data_group == "observed" and str(mc_index_config).lower() == "all":
        raise NotImplementedError(
            "RBF reconstruction requires one observed realization; set mc_index to an integer"
        )
    grid = config.get("grid", {})
    if not isinstance(grid, dict):
        raise ValueError("rbf_reconstruction.grid must be a mapping")
    plot = config.get("plot", {})
    if not isinstance(plot, dict):
        raise ValueError("rbf_reconstruction.plot must be a mapping")
    broeren_error_plot = config.get("broeren_error_plot", {})
    if not isinstance(broeren_error_plot, dict):
        raise ValueError("rbf_reconstruction.broeren_error_plot must be a mapping")
    reference = config.get("reference", {})
    if not isinstance(reference, dict):
        raise ValueError("rbf_reconstruction.reference must be a mapping")
    reference_plot = reference.get("plot", {})
    if not isinstance(reference_plot, dict):
        raise ValueError("rbf_reconstruction.reference.plot must be a mapping")
    sigma = config.get("sigma", config.get("length_scale"))
    configured_sigma_selection = config.get("sigma_selection")
    auto_sigma = bool(config.get("auto_sigma_rippa", config.get("auto_sigma", False)))
    if configured_sigma_selection is None:
        sigma_selection = "rippa_grid" if auto_sigma else "manual"
    else:
        sigma_selection = str(configured_sigma_selection).lower()
        auto_sigma = sigma_selection != "manual"
    if sigma_selection not in {"manual", "rippa_grid", "explicit_loocv"}:
        raise ValueError(
            "rbf_reconstruction.sigma_selection must be manual, rippa_grid, "
            "or explicit_loocv"
        )
    sigma_grid = str(config.get("sigma_grid", "log")).lower()
    if sigma_grid not in {"log", "linear"}:
        raise ValueError("rbf_reconstruction.sigma_grid must be 'log' or 'linear'")
    rippa_plot = config.get("rippa_plot", {})
    if not isinstance(rippa_plot, dict):
        raise ValueError("rbf_reconstruction.rippa_plot must be a mapping")
    magnitude_cmap = str(plot.get("magnitude_cmap", plot.get("cmap", "batlow")))
    error_cmap = str(plot.get("error_cmap", "imola"))
    if magnitude_cmap not in MAGNITUDE_CMAP_NAMES:
        raise ValueError(
            "rbf_reconstruction.plot.magnitude_cmap must be one of "
            f"{sorted(MAGNITUDE_CMAP_NAMES)}"
        )
    if error_cmap not in ERROR_CMAP_NAMES:
        raise ValueError(
            "rbf_reconstruction.plot.error_cmap must be one of "
            f"{sorted(ERROR_CMAP_NAMES)}"
        )
    reference_magnitude_cmap = str(
        reference_plot.get("magnitude_cmap", reference_plot.get("cmap", magnitude_cmap))
    )
    reference_error_cmap = str(reference_plot.get("error_cmap", error_cmap))
    broeren_error_cmap = str(
        broeren_error_plot.get("error_cmap", error_cmap)
    )
    component_mode = str(plot.get("component_mode", "raw"))
    background_mode = str(plot.get("background_mode", "metadata"))
    background_vector = plot.get("background_vector")
    reference_component_mode = str(reference_plot.get("component_mode", component_mode))
    reference_background_mode = str(reference_plot.get("background_mode", background_mode))
    reference_background_vector = reference_plot.get(
        "background_vector", background_vector
    )
    if reference_magnitude_cmap not in MAGNITUDE_CMAP_NAMES:
        raise ValueError(
            "rbf_reconstruction.reference.plot.magnitude_cmap must be one of "
            f"{sorted(MAGNITUDE_CMAP_NAMES)}"
        )
    if reference_error_cmap not in ERROR_CMAP_NAMES:
        raise ValueError(
            "rbf_reconstruction.reference.plot.error_cmap must be one of "
            f"{sorted(ERROR_CMAP_NAMES)}"
        )
    if broeren_error_cmap not in ERROR_CMAP_NAMES:
        raise ValueError(
            "rbf_reconstruction.broeren_error_plot.error_cmap must be one of "
            f"{sorted(ERROR_CMAP_NAMES)}"
        )
    if component_mode not in COMPONENT_MODES:
        raise ValueError(
            "rbf_reconstruction.plot.component_mode must be raw or delta"
        )
    if reference_component_mode not in COMPONENT_MODES:
        raise ValueError(
            "rbf_reconstruction.reference.plot.component_mode must be raw or delta"
        )
    if background_mode not in BACKGROUND_MODES:
        raise ValueError(
            "rbf_reconstruction.plot.background_mode must be one of "
            f"{sorted(BACKGROUND_MODES)}"
        )
    if reference_background_mode not in BACKGROUND_MODES:
        raise ValueError(
            "rbf_reconstruction.reference.plot.background_mode must be one of "
            f"{sorted(BACKGROUND_MODES)}"
        )
    background_vector = (
        None if background_vector is None else _vector_from_value(background_vector)
    )
    reference_background_vector = (
        None
        if reference_background_vector is None
        else _vector_from_value(reference_background_vector)
    )
    if background_mode == "user" and background_vector is None:
        raise ValueError(
            "rbf_reconstruction.plot.background_mode: user requires background_vector"
        )
    if reference_background_mode == "user" and reference_background_vector is None:
        raise ValueError(
            "rbf_reconstruction.reference.plot.background_mode: user requires "
            "background_vector"
        )
    return {
        "data_group": data_group,
        "mc_index": None if data_group == "truth" else int(mc_index_config),
        "t0": None if t0 is None else float(t0),
        "t0_index": None if t0_index is None else int(t0_index),
        "delta_t": delta_t,
        "sample_time_stride": int(config.get("sample_time_stride", 1)),
        "kernel": str(config.get("kernel", "multiquadric")),
        "sigma": None if sigma is None else float(sigma),
        "auto_sigma": auto_sigma,
        "sigma_selection": sigma_selection,
        "sigma_min": config.get("sigma_min"),
        "sigma_max": config.get("sigma_max"),
        "sigma_num": int(config.get("sigma_num", 25)),
        "sigma_grid": sigma_grid,
        "sigma_initial": float(config.get("sigma_initial", 100.0)),
        "sigma_subsample_max": config.get("sigma_subsample_max"),
        "sigma_random_seed": config.get("sigma_random_seed"),
        "rippa_mode": str(config.get("rippa_mode", "field")).lower(),
        "rippa_components": config.get("rippa_components", "all"),
        "sigma_cache": config.get("sigma_cache", "outputs/rbf_sigma_cache.h5"),
        "use_sigma_cache": bool(config.get("use_sigma_cache", True)),
        "save_sigma_cache": bool(config.get("save_sigma_cache", True)),
        "force_recompute_sigma": bool(config.get("force_recompute_sigma", False)),
        "sigma_cache_hash_tolerance": config.get("sigma_cache_hash_tolerance"),
        "coordinate_units": str(config.get("coordinate_units", "km")),
        "smoothing": float(config.get("smoothing", 0.0)),
        "min_samples": int(config.get("min_samples", 4)),
        "grid": {
            "radius": float(grid["radius"]),
            "n_axial": int(grid["n_axial"]),
            "n_transverse": int(grid.get("n_transverse", grid.get("n_radial", 11))),
            "axial_padding": float(grid.get("axial_padding", 0.0)),
            "radius_padding": float(
                grid.get(
                    "radius_padding",
                    grid.get("transverse_padding", 0.0),
                )
            ),
            "require_all_spacecraft_coverage": bool(
                grid.get(
                    "require_all_spacecraft_coverage",
                    grid.get("all_spacecraft_coverage", False),
                )
            ),
        },
        "plot": {
            "enabled": bool(plot.get("enabled", False)),
            "path": plot.get("path"),
            "dpi": int(plot.get("dpi", 150)),
            "magnitude_cmap": magnitude_cmap,
            "error_cmap": error_cmap,
            "component_mode": component_mode,
            "background_mode": background_mode,
            "background_vector": background_vector,
        },
        "rippa_plot": {
            "enabled": bool(rippa_plot.get("enabled", True)),
            "path": rippa_plot.get("path"),
            "dpi": int(rippa_plot.get("dpi", plot.get("dpi", 150))),
        },
        "broeren_error_plot": {
            "enabled": bool(broeren_error_plot.get("enabled", False)),
            "path": broeren_error_plot.get("path"),
            "dpi": int(broeren_error_plot.get("dpi", plot.get("dpi", 150))),
            "error_cmap": broeren_error_cmap,
            "capped": bool(broeren_error_plot.get("capped", True)),
        },
        "reference": {
            "enabled": bool(reference.get("enabled", False)),
            "generation_config": reference.get("generation_config"),
            "observation_path": reference.get("observation_path"),
            "plot": {
                "enabled": bool(reference_plot.get("enabled", True)),
                "path": reference_plot.get("path"),
                "dpi": int(reference_plot.get("dpi", plot.get("dpi", 150))),
                "magnitude_cmap": reference_magnitude_cmap,
                "error_cmap": reference_error_cmap,
                "component_mode": reference_component_mode,
                "background_mode": reference_background_mode,
                "background_vector": reference_background_vector,
            },
        },
    }


def _read_magnetic_measurements(
    input_path: str | Path,
    *,
    data_group: str = "truth",
    mc_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    with h5py.File(input_path, "r") as handle:
        if "time/B" in handle:
            times = np.asarray(handle["time/B"], dtype=float)
        elif "time" in handle:
            times = np.asarray(handle["time"], dtype=float)
        else:
            raise KeyError("Input file is missing /time/B")
        dataset_path = f"{data_group}/B"
        if dataset_path not in handle:
            raise KeyError(f"Input file is missing /{dataset_path}")
        magnetic_source = np.asarray(handle[dataset_path], dtype=float)
        if data_group == "observed":
            if magnetic_source.ndim != 4 or magnetic_source.shape[-1] != 3:
                raise ValueError(
                    "/observed/B must have shape (mc, time, spacecraft, 3)"
                )
            index = 0 if mc_index is None else int(mc_index)
            if not 0 <= index < magnetic_source.shape[0]:
                raise IndexError(
                    f"mc_index {index} is outside /observed/B ensemble axis "
                    f"with length {magnetic_source.shape[0]}"
                )
            magnetic = magnetic_source[index]
        else:
            magnetic = magnetic_source
        if "spacecraft/r/B" in handle:
            positions = np.asarray(handle["spacecraft/r/B"], dtype=float)
        elif "spacecraft/r" in handle:
            positions = np.asarray(handle["spacecraft/r"], dtype=float)
        else:
            raise KeyError("Input file is missing /spacecraft/r/B")
        if "spacecraft/id" in handle:
            spacecraft_ids = [
                item.decode("utf-8") if isinstance(item, bytes) else str(item)
                for item in handle["spacecraft/id"][...]
            ]
        else:
            spacecraft_ids = [f"S{index}" for index in range(positions.shape[0])]

    if magnetic.ndim != 3 or magnetic.shape[-1] != 3:
        raise ValueError(f"/{data_group}/B must have shape (time, spacecraft, 3)")
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("/spacecraft/r/B must have shape (spacecraft, time, 3)")
    if magnetic.shape[:2] != (positions.shape[1], positions.shape[0]):
        raise ValueError(
            f"/{data_group}/B shape must match /spacecraft/r/B as "
            "(time, spacecraft, 3)"
        )
    if magnetic.shape[0] != times.size:
        raise ValueError(f"/{data_group}/B time axis length does not match /time/B")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(magnetic)):
        raise ValueError("Magnetic-field times or samples contain non-finite values")
    if not np.all(np.isfinite(positions)):
        raise ValueError("Spacecraft positions contain non-finite values")
    if len(spacecraft_ids) != positions.shape[0]:
        raise ValueError("/spacecraft/id length does not match /spacecraft/r/B")
    return times, positions, magnetic, spacecraft_ids


def _window_mask(
    times: np.ndarray,
    *,
    t0: float | None,
    t0_index: int | None,
    delta_t: float,
) -> tuple[float, np.ndarray]:
    if t0_index is not None:
        index = t0_index
        if not 0 <= index < times.size:
            raise IndexError(f"t0_index {index} is outside the B time axis")
        center_time = float(times[index])
    else:
        assert t0 is not None
        center_time = float(t0)
    half_width = 0.5 * float(delta_t)
    mask = (times >= center_time - half_width) & (times <= center_time + half_width)
    if not np.any(mask):
        raise ValueError("Selected RBF time window contains no samples")
    return center_time, mask


def collect_samples_for_window(
    times: np.ndarray,
    positions: np.ndarray,
    B: np.ndarray,
    *,
    t0: float | None = None,
    t0_index: int | None = None,
    delta_t: float,
    time_stride: int = 1,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect unwrapped spacecraft samples for the selected time window."""

    time_axis = np.asarray(times, dtype=float)
    selected_positions_source = np.asarray(positions, dtype=float)
    selected_B_source = np.asarray(B, dtype=float)
    if time_axis.ndim != 1:
        raise ValueError("times must have shape (time,)")
    if selected_positions_source.ndim != 3 or selected_positions_source.shape[-1] != 3:
        raise ValueError("positions must have shape (time, spacecraft, 3)")
    if selected_B_source.shape != selected_positions_source.shape:
        raise ValueError("B must have shape (time, spacecraft, 3)")
    if selected_positions_source.shape[0] != time_axis.size:
        raise ValueError("positions and B time axes must match times")
    if not (
        np.all(np.isfinite(time_axis))
        and np.all(np.isfinite(selected_positions_source))
        and np.all(np.isfinite(selected_B_source))
    ):
        raise ValueError("Window inputs contain non-finite values")
    time_stride = int(time_stride)
    if time_stride < 1:
        raise ValueError("time_stride must be at least 1")
    center_time, mask = _window_mask(
        time_axis, t0=t0, t0_index=t0_index, delta_t=delta_t
    )
    window_indices = np.nonzero(mask)[0][::time_stride]
    if window_indices.size == 0:
        raise ValueError("Selected RBF time window contains no samples after time_stride")
    selected_times = time_axis[window_indices]
    selected_positions = selected_positions_source[window_indices]
    selected_B = selected_B_source[window_indices]
    return (
        center_time,
        selected_times,
        selected_positions,
        selected_positions.reshape(-1, 3),
        selected_B.reshape(-1, 3),
    )


def compute_rbf_reconstruction(input_path: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    """Compute component-wise RBF reconstruction from selected magnetic samples."""

    settings = _settings(config)
    times, positions_sc_time, magnetic_time_sc, spacecraft_ids = _read_magnetic_measurements(
        input_path,
        data_group=settings["data_group"],
        mc_index=settings["mc_index"],
    )
    # Convert positions from file layout (spacecraft, time, xyz) to the grid and
    # flattening layout (time, spacecraft, xyz), preserving unwrapped coordinates.
    positions_time_sc = np.swapaxes(positions_sc_time, 0, 1)
    (
        t0,
        selected_times,
        selected_positions,
        sample_positions,
        sample_B,
    ) = collect_samples_for_window(
        times,
        positions_time_sc,
        magnetic_time_sc,
        t0=settings["t0"],
        t0_index=settings["t0_index"],
        delta_t=settings["delta_t"],
        time_stride=settings["sample_time_stride"],
    )
    if sample_positions.shape[0] < settings["min_samples"]:
        raise ValueError(
            f"RBF reconstruction requires at least {settings['min_samples']} samples; "
            f"selected {sample_positions.shape[0]}"
        )

    grid = build_cylindrical_grid(
        selected_positions,
        selected_times,
        radius=settings["grid"]["radius"],
        n_axial=settings["grid"]["n_axial"],
        n_transverse=settings["grid"]["n_transverse"],
        axial_padding=settings["grid"]["axial_padding"],
        radius_padding=settings["grid"]["radius_padding"],
        require_all_spacecraft_coverage=settings["grid"][
            "require_all_spacecraft_coverage"
        ],
    )
    sigma_candidates = None
    sigma_selection = None
    sigma_source = "manual" if settings["sigma"] is not None else "geometry"
    if settings["auto_sigma"]:
        selected_B = sample_B.reshape(selected_positions.shape)
        sigma_selection_positions = sample_positions
        sigma_selection_B = sample_B
        sigma_time_indices = np.arange(selected_positions.shape[0], dtype=int)
        if settings["sigma_selection"] == "explicit_loocv":
            sigma_time_indices, sigma_selection_positions, sigma_selection_B = (
                _subsample_sigma_training_data(
                    selected_positions,
                    selected_B,
                    max_samples=(
                        None
                        if settings["sigma_subsample_max"] is None
                        else int(settings["sigma_subsample_max"])
                    ),
                    random_seed=(
                        None
                        if settings["sigma_random_seed"] is None
                        else int(settings["sigma_random_seed"])
                    ),
                )
            )
        else:
            sigma_candidates = _sigma_candidates_from_config(
                sample_positions,
                sigma_min=(
                    None
                    if settings["sigma_min"] is None
                    else float(settings["sigma_min"])
                ),
                sigma_max=(
                    None
                    if settings["sigma_max"] is None
                    else float(settings["sigma_max"])
                ),
                sigma_num=settings["sigma_num"],
                sigma_grid=settings["sigma_grid"],
            )
        sigma_selection = get_or_compute_sigma(
            sample_positions,
            sample_B,
            sigma_candidates,
            selection_positions=sigma_selection_positions,
            selection_B=sigma_selection_B,
            cache_path=settings["sigma_cache"],
            use_cache=settings["use_sigma_cache"],
            force_recompute=settings["force_recompute_sigma"],
            save_cache=settings["save_sigma_cache"],
            selected_times=selected_times,
            spacecraft_ids=spacecraft_ids,
            t0=t0,
            delta_t=settings["delta_t"],
            kernel=settings["kernel"],
            smoothing=settings["smoothing"],
            sigma_selection=settings["sigma_selection"],
            sigma_initial=settings["sigma_initial"],
            sigma_min=(
                None
                if settings["sigma_min"] is None
                else float(settings["sigma_min"])
            ),
            sigma_max=(
                None
                if settings["sigma_max"] is None
                else float(settings["sigma_max"])
            ),
            sigma_subsample_max=(
                None
                if settings["sigma_subsample_max"] is None
                else int(settings["sigma_subsample_max"])
            ),
            sigma_random_seed=(
                None
                if settings["sigma_random_seed"] is None
                else int(settings["sigma_random_seed"])
            ),
            rippa_mode=settings["rippa_mode"],
            rippa_components=settings["rippa_components"],
            coordinate_units=settings["coordinate_units"],
            hash_tolerance=(
                None
                if settings["sigma_cache_hash_tolerance"] is None
                else float(settings["sigma_cache_hash_tolerance"])
            ),
            metadata={
                "t0": t0,
                "delta_t": settings["delta_t"],
                "sample_time_stride": settings["sample_time_stride"],
                "n_spacecraft": selected_positions.shape[1],
                "barycenter_start": grid.barycenter_path[0],
                "barycenter_end": grid.barycenter_path[-1],
                "sigma_time_indices": sigma_time_indices,
                "notes": "RBF sigma cache keyed to unwrapped sample positions",
            },
        )
        sigma_source = sigma_selection["source"]
    reconstructor = ComponentwiseRBFReconstructor(
        kernel=settings["kernel"],
        sigma=(
            settings["sigma"]
            if sigma_selection is None
            else float(sigma_selection["selected_sigma"])
        ),
        smoothing=settings["smoothing"],
        auto_sigma=False,
        sigma_candidates=None,
        rippa_mode=settings["rippa_mode"],
        rippa_components=settings["rippa_components"],
    ).fit(sample_positions, sample_B)
    reconstructed = reconstructor.evaluate(grid.points)
    rbf_metadata = reconstructor.get_metadata()
    if sigma_selection is not None:
        rbf_metadata["sigma_selection"] = sigma_selection
    rbf_metadata["sigma_source"] = sigma_source
    rbf_metadata["sigma_cache_key"] = (
        None if sigma_selection is None else sigma_selection["cache_key"]
    )
    rbf_metadata["sigma_cache_path"] = (
        None if sigma_selection is None else sigma_selection["cache_path"]
    )
    return {
        "metadata": {
            "method": "rbf_reconstruction",
            "input_path": str(Path(input_path)),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "assumption": "static magnetic field during selected time window",
            "data_group": settings["data_group"],
            "mc_index": "null" if settings["mc_index"] is None else settings["mc_index"],
            "fitter_uses": "spacecraft positions and selected magnetic-field samples only",
            "coordinate_policy": "unwrapped physical coordinates; no periodic wrapping",
        },
        "settings": settings,
        "t0": t0,
        "delta_t": settings["delta_t"],
        "selected_times": selected_times,
        "spacecraft_ids": spacecraft_ids,
        "sample_trajectory": selected_positions,
        "sample_positions": sample_positions,
        "sample_B": sample_B,
        "grid": grid,
        "B_reconstructed": reconstructed,
        "rbf_metadata": rbf_metadata,
    }


def _metadata_dict(grid: CylindricalGrid) -> dict[str, np.ndarray | float | int]:
    return {
        "center": grid.center,
        "e_parallel": grid.e_parallel,
        "e1": grid.e1,
        "e2": grid.e2,
        "barycenter_start": grid.barycenter_path[0],
        "barycenter_end": grid.barycenter_path[-1],
        "radius": grid.radius,
        "requested_radius": grid.requested_radius,
        "radius_padding": grid.radius_padding,
        "max_spacecraft_transverse_radius": grid.max_spacecraft_transverse_radius,
        "require_all_spacecraft_coverage": grid.require_all_spacecraft_coverage,
        "axial_coverage_min": grid.axial_coverage_min,
        "axial_coverage_max": grid.axial_coverage_max,
        "n_axial": grid.n_axial,
        "n_transverse": grid.n_transverse,
    }


def _write_metadata(group: h5py.Group, metadata: dict[str, Any]) -> None:
    for key, value in metadata.items():
        if isinstance(value, dict):
            subgroup = group.create_group(key)
            _write_metadata(subgroup, value)
        elif isinstance(value, np.ndarray):
            group.create_dataset(key, data=value)
        elif value is None:
            group.attrs[key] = "null"
        else:
            if isinstance(value, (list, tuple)):
                value = ",".join(str(item) for item in value)
            group.attrs[key] = value


def write_rbf_reconstruction(result: dict[str, Any], output_path: str | Path) -> Path:
    """Write the RBF reconstruction product using the project HDF5 convention."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    grid: CylindricalGrid = result["grid"]
    settings = result["settings"]
    metadata = {
        **result["metadata"],
        **{
            key: value
            for key, value in result["rbf_metadata"].items()
            if key != "sigma_selection"
        },
        "auto_sigma": settings["auto_sigma"],
        "sigma_grid": settings["sigma_grid"],
        "sigma_selection": settings["sigma_selection"],
        "sigma_initial": settings["sigma_initial"],
        "sigma_subsample_max": settings["sigma_subsample_max"],
        "sigma_random_seed": settings["sigma_random_seed"],
        "sigma_num": settings["sigma_num"],
        "sigma_min": settings["sigma_min"],
        "sigma_max": settings["sigma_max"],
        "rippa_mode": settings["rippa_mode"],
        "rippa_components": settings["rippa_components"],
        "sigma_cache": settings["sigma_cache"],
        "use_sigma_cache": settings["use_sigma_cache"],
        "save_sigma_cache": settings["save_sigma_cache"],
        "force_recompute_sigma": settings["force_recompute_sigma"],
        "sigma_cache_hash_tolerance": settings["sigma_cache_hash_tolerance"],
        "sample_time_stride": settings["sample_time_stride"],
        "coordinate_units": settings["coordinate_units"],
        "grid": _metadata_dict(grid),
    }
    with h5py.File(output, "w") as handle:
        _write_metadata(handle.create_group("metadata"), metadata)
        samples = handle.create_group("samples")
        samples.create_dataset("position", data=result["sample_positions"], compression="gzip")
        samples.create_dataset(
            "trajectory", data=result["sample_trajectory"], compression="gzip"
        )
        samples.create_dataset("B", data=result["sample_B"], compression="gzip")
        samples.create_dataset("time", data=result["selected_times"])
        samples.create_dataset(
            "spacecraft_id",
            data=np.asarray(
                result["spacecraft_ids"], dtype=h5py.string_dtype("utf-8")
            ),
        )

        grid_group = handle.create_group("grid")
        grid_group.create_dataset("points", data=grid.points, compression="gzip")
        grid_group.create_dataset(
            "full_points", data=grid.full_points, compression="gzip"
        )
        grid_group.create_dataset(
            "dimensions",
            data=np.asarray((grid.n_axial, grid.n_transverse, grid.n_transverse)),
        )
        grid_group.create_dataset(
            "local_coordinates",
            data=np.stack((grid.axial, grid.transverse_u, grid.transverse_v), axis=-1),
            compression="gzip",
        )
        grid_group.create_dataset("local_axial", data=grid.axial, compression="gzip")
        grid_group.create_dataset("local_u", data=grid.transverse_u, compression="gzip")
        grid_group.create_dataset("local_v", data=grid.transverse_v, compression="gzip")
        grid_group.create_dataset("cylinder_mask", data=grid.mask, compression="gzip")
        grid_group.create_dataset("axial_values", data=grid.axial_values)
        grid_group.create_dataset("transverse_values", data=grid.transverse_values)
        grid_group.create_dataset("barycenter_path", data=grid.barycenter_path)
        grid_group.create_dataset("center", data=grid.center)
        grid_group.create_dataset(
            "basis_vectors", data=np.stack((grid.e_parallel, grid.e1, grid.e2), axis=0)
        )

        reconstruction = handle.create_group("reconstruction")
        reconstruction.create_dataset(
            "B", data=result["B_reconstructed"], compression="gzip"
        )
        reconstruction.create_dataset(
            "Bmag",
            data=np.linalg.norm(result["B_reconstructed"], axis=-1),
            compression="gzip",
        )
        reconstruction.attrs["components"] = np.asarray(
            COMPONENT_NAMES, dtype=h5py.string_dtype("utf-8")
        )
        rbf_group = handle.create_group("rbf")
        rbf_group.attrs["sigma"] = result["rbf_metadata"]["sigma"]
        rbf_group.attrs["sigma_source"] = result["rbf_metadata"]["sigma_source"]
        if result["rbf_metadata"].get("sigma_cache_key") is not None:
            rbf_group.attrs["sigma_cache_key"] = result["rbf_metadata"][
                "sigma_cache_key"
            ]
        if result["rbf_metadata"].get("sigma_cache_path") is not None:
            rbf_group.attrs["sigma_cache_path"] = result["rbf_metadata"][
                "sigma_cache_path"
            ]
        rbf_group.attrs["kernel"] = result["rbf_metadata"]["kernel"]
        rbf_group.attrs["smoothing"] = result["rbf_metadata"]["smoothing"]
        rbf_group.attrs["condition_number"] = result["rbf_metadata"][
            "condition_number"
        ]
        rbf_group.attrs["solver"] = result["rbf_metadata"]["solver"]
        handle.attrs["t0"] = result["t0"]
        handle.attrs["delta_t"] = result["delta_t"]
        sigma_selection = result["rbf_metadata"].get("sigma_selection")
        if sigma_selection is not None:
            selection = handle.create_group("sigma_selection")
            for key, value in sigma_selection.items():
                if isinstance(value, np.ndarray):
                    selection.create_dataset(key, data=value)
                elif value is not None:
                    if isinstance(value, (list, tuple)):
                        value = ",".join(str(item) for item in value)
                    selection.attrs[key] = value
            if "sigma_candidates" in sigma_selection:
                rbf_group.create_dataset(
                    "sigma_candidates",
                    data=np.asarray(sigma_selection["sigma_candidates"], dtype=float),
                )
            if "costs" in sigma_selection:
                rbf_group.create_dataset(
                    "loocv_costs",
                    data=np.asarray(sigma_selection["costs"], dtype=float),
                )
    return output


def _values_to_full_grid(
    values: np.ndarray,
    mask: np.ndarray,
    n_axial: int,
    n_transverse: int,
    *,
    name: str = "values",
) -> np.ndarray:
    value_array = np.asarray(values, dtype=float)
    if value_array.shape[0] != int(np.sum(mask)):
        raise ValueError(f"{name} length does not match the cylinder mask")
    full = np.full((n_axial, n_transverse, n_transverse), np.nan, dtype=float)
    full[mask] = value_array
    return full


def _slice_to_image(
    values: np.ndarray,
    mask: np.ndarray,
    n_axial: int,
    n_transverse: int,
    *,
    mode: str,
) -> np.ndarray:
    full = _values_to_full_grid(values, mask, n_axial, n_transverse)
    if mode == "axial":
        return full[:, :, n_transverse // 2].T
    if mode == "transverse":
        return full[n_axial // 2]
    raise ValueError(f"Unknown slice mode {mode!r}")


def _is_relative_error_label(label: str) -> bool:
    normalized = label.lower()
    return "relative error" in normalized or "error percent" in normalized


def _error_contour_levels(image: np.ndarray) -> np.ndarray:
    finite = np.asarray(image, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.array([], dtype=float)
    lower = float(np.nanmin(finite))
    upper = float(np.nanmax(finite))
    levels = np.arange(5.0, upper + 0.1, 5.0)
    return levels[(levels > lower) & (levels < upper)]


def _add_error_contours(axis: Any, x: np.ndarray, y: np.ndarray, image: np.ndarray) -> None:
    levels = _error_contour_levels(image)
    if levels.size == 0:
        return
    masked = np.ma.masked_invalid(image)
    axis.contour(
        x,
        y,
        masked,
        levels=levels,
        colors="black",
        linewidths=1.25,
        alpha=0.65,
        zorder=4,
    )
    axis.contour(
        x,
        y,
        masked,
        levels=levels,
        colors="white",
        linewidths=0.6,
        alpha=0.95,
        zorder=4.1,
    )


def get_scientific_cmap(name: str, fallback: str = "magma") -> Any:
    """Return a Matplotlib or optional Crameri colormap with a clear fallback."""

    import matplotlib

    try:
        return matplotlib.colormaps[name]
    except KeyError:
        pass

    try:
        import cmcrameri.cm as cmc  # type: ignore[import-not-found]

        if hasattr(cmc, name):
            return getattr(cmc, name)
    except ImportError:
        pass

    print(f"Requested colormap {name} not available; using {fallback}.")
    return matplotlib.colormaps[fallback]


def _component_cmap() -> Any:
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "custom_blue_grey_red",
        (
            (0.0, "#2C6BB2"),
            (0.5, "#D9D9D9"),
            (1.0, "#B2182B"),
        ),
    )


def _plot_product_kind(label: str) -> str:
    if label.startswith(("Bx ", "By ", "Bz ", "δBx ", "δBy ", "δBz ")):
        return "component"
    if label.startswith("|B|"):
        return "magnitude"
    if _is_relative_error_label(label) or "error magnitude" in label.lower():
        return "error"
    return "scalar"


def _vector_from_value(value: Any) -> np.ndarray | None:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        value = value.replace(",", " ").strip()
        if not value:
            return None
        parts = value.split()
        if len(parts) != 3:
            return None
        vector = np.asarray([float(part) for part in parts], dtype=float)
    else:
        vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        return None
    return vector


def _background_from_metadata(handle: h5py.File) -> np.ndarray | None:
    dataset_paths = (
        "metadata/B0",
        "simulation/B0",
        "simulation/background_field",
        "metadata/background_field",
    )
    attr_locations = (
        (handle.attrs, ("B0", "background_field")),
        (handle["metadata"].attrs, ("B0", "background_field"))
        if "metadata" in handle
        else (None, ()),
        (handle["simulation"].attrs, ("B0", "background_field"))
        if "simulation" in handle
        else (None, ()),
    )
    for path in dataset_paths:
        if path in handle:
            vector = _vector_from_value(handle[path][...])
            if vector is not None:
                return vector
    for attrs, names in attr_locations:
        if attrs is None:
            continue
        for name in names:
            if name in attrs:
                vector = _vector_from_value(attrs[name])
                if vector is not None:
                    return vector
    return None


def _valid_field_mean(field: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(field, dtype=float)
    if values.ndim == 4 and values.shape[-1] == 3:
        if mask is not None:
            values = values[np.asarray(mask, dtype=bool)]
        else:
            values = values.reshape(-1, 3)
    elif values.ndim == 2 and values.shape[1] == 3:
        pass
    else:
        raise ValueError("Background mean field requires shape (N, 3) or (..., 3)")
    finite = np.all(np.isfinite(values), axis=1)
    if not np.any(finite):
        raise ValueError("Cannot determine background field from non-finite values")
    return np.mean(values[finite], axis=0)


def _read_background_field(
    handle: h5py.File,
    source: str,
    mask: np.ndarray | None,
) -> np.ndarray:
    if source == "actual-volume-mean":
        for path in ("validation/B_actual", "validation/B_true"):
            if path in handle:
                return _valid_field_mean(np.asarray(handle[path], dtype=float), mask)
        raise ValueError("background-mode actual-volume-mean requires /validation/B_actual")
    if source == "reconstructed-volume-mean":
        if "reconstruction/B" not in handle:
            raise ValueError(
                "background-mode reconstructed-volume-mean requires /reconstruction/B"
            )
        return _valid_field_mean(np.asarray(handle["reconstruction/B"], dtype=float), mask)
    raise ValueError(f"Unsupported background mean source {source!r}")


def determine_background_field(
    h5_file: h5py.File,
    field_kind: str,
    background_mode: str = "metadata",
    user_background: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Determine a common guide/background field for delta-component plots."""

    del field_kind  # Reserved for future side-by-side actual/reconstructed plots.
    mode = str(background_mode)
    if mode not in BACKGROUND_MODES:
        raise ValueError(f"background_mode must be one of {sorted(BACKGROUND_MODES)}")
    if mode == "user":
        vector = None if user_background is None else _vector_from_value(user_background)
        if vector is None:
            raise ValueError("background-mode user requires a finite 3-vector")
        return vector, {"source": "user"}
    if mode == "metadata":
        vector = _background_from_metadata(h5_file)
        if vector is not None:
            return vector, {"source": "metadata"}
        if "validation/B_actual" in h5_file or "validation/B_true" in h5_file:
            return _read_background_field(
                h5_file, "actual-volume-mean", mask
            ), {"source": "actual-volume-mean"}
        return _read_background_field(
            h5_file, "reconstructed-volume-mean", mask
        ), {"source": "reconstructed-volume-mean"}
    return _read_background_field(h5_file, mode, mask), {"source": mode}


def compute_component_plot_field(
    B: np.ndarray,
    B0: np.ndarray | None = None,
    mode: str = "raw",
) -> np.ndarray:
    """Return raw or fluctuating component field for diagnostic plots."""

    values = np.asarray(B, dtype=float)
    if mode == "raw":
        return values
    if mode == "delta":
        if B0 is None:
            raise ValueError("component-mode delta requires a background field")
        return values - np.asarray(B0, dtype=float).reshape(1, 3)
    raise ValueError("component mode must be 'raw' or 'delta'")


def _mean_over_axial(
    values: np.ndarray,
    mask: np.ndarray,
    n_axial: int,
    n_transverse: int,
) -> np.ndarray:
    full = _values_to_full_grid(values, mask, n_axial, n_transverse)
    finite = np.isfinite(full)
    counts = np.sum(finite, axis=0)
    sums = np.nansum(full, axis=0)
    return np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)


def _magnetic_products(
    values: np.ndarray,
    label: str,
    *,
    component_mode: str = "raw",
    background: np.ndarray | None = None,
) -> list[tuple[str, np.ndarray]]:
    component_values = compute_component_plot_field(
        values, B0=background, mode=component_mode
    )
    component_labels = (
        ("δBx", "δBy", "δBz") if component_mode == "delta" else ("Bx", "By", "Bz")
    )
    return [
        (f"{component_labels[0]} {label}", component_values[:, 0]),
        (f"{component_labels[1]} {label}", component_values[:, 1]),
        (f"{component_labels[2]} {label}", component_values[:, 2]),
        (f"|B| {label}", np.linalg.norm(values, axis=-1)),
    ]


def _trajectory_local_coordinates(
    trajectory: np.ndarray,
    center: np.ndarray,
    basis_vectors: np.ndarray,
) -> np.ndarray:
    """Project trajectory positions onto the local axial/u/v cylinder basis."""

    return np.einsum("tsc,lc->tsl", trajectory - center, basis_vectors)


def _overlay_trajectories(
    axis: Any,
    local_trajectory: np.ndarray,
    mode: str,
    spacecraft_ids: list[str] | None = None,
) -> None:
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt

    n_spacecraft = local_trajectory.shape[1]
    if spacecraft_ids is None:
        spacecraft_ids = [f"S{index}" for index in range(n_spacecraft)]
    fallback_colors = plt.cm.tab10(np.linspace(0.0, 0.9, max(n_spacecraft, 1)))
    effects = [path_effects.withStroke(linewidth=1.2, foreground="black", alpha=0.45)]
    for spacecraft_index in range(n_spacecraft):
        spacecraft_id = spacecraft_ids[spacecraft_index]
        color = SPACECRAFT_COLORS.get(
            spacecraft_id, fallback_colors[spacecraft_index % len(fallback_colors)]
        )
        if mode == "axial":
            x = local_trajectory[:, spacecraft_index, 0]
            y = local_trajectory[:, spacecraft_index, 1]
        elif mode == "transverse":
            x = local_trajectory[:, spacecraft_index, 2]
            y = local_trajectory[:, spacecraft_index, 1]
        else:
            raise ValueError(f"Unknown trajectory overlay mode {mode!r}")
        axis.plot(
            x,
            y,
            color=color,
            linewidth=0.75,
            marker="o",
            markersize=1.8,
            markeredgewidth=0.25,
            markeredgecolor="black",
            alpha=0.9,
            label=spacecraft_id if axis.get_legend() is None else None,
            path_effects=effects,
            zorder=5,
        )


def _read_plot_context(
    hdf5_path: str | Path,
    field_path: str,
    label: str,
    *,
    component_mode: str = "raw",
    background_mode: str = "metadata",
    background_vector: np.ndarray | None = None,
) -> dict[str, Any]:
    with h5py.File(hdf5_path, "r") as data:
        values = np.asarray(data[field_path], dtype=float)
        mask = np.asarray(data["grid/cylinder_mask"], dtype=bool)
        axial_values = np.asarray(data["grid/axial_values"], dtype=float)
        transverse_values = np.asarray(data["grid/transverse_values"], dtype=float)
        local_trajectory = None
        spacecraft_ids = None
        if "samples/trajectory" in data:
            local_trajectory = _trajectory_local_coordinates(
                np.asarray(data["samples/trajectory"], dtype=float),
                np.asarray(data["grid/center"], dtype=float),
                np.asarray(data["grid/basis_vectors"], dtype=float),
            )
            if "samples/spacecraft_id" in data:
                spacecraft_ids = [
                    item.decode("utf-8") if isinstance(item, bytes) else str(item)
                    for item in data["samples/spacecraft_id"][...]
                ]
        background = None
        background_metadata: dict[str, Any] | None = None
        if component_mode == "delta":
            field_kind = "actual" if field_path.startswith("validation/") else "reconstructed"
            background, background_metadata = determine_background_field(
                data,
                field_kind,
                background_mode=background_mode,
                user_background=background_vector,
                mask=mask,
            )
            print(
                "Using B0 for delta component plots "
                f"({background_metadata['source']}): {background}"
            )
        products = _magnetic_products(
            values,
            label,
            component_mode=component_mode,
            background=background,
        )
        if field_path == "reconstruction/B" and "validation/error_percent_capped" in data:
            products.append(
                (
                    "Relative error capped at 100%",
                    np.asarray(data["validation/error_percent_capped"], dtype=float),
                )
            )
        elif field_path == "reconstruction/B" and "validation/error_mag" in data:
            products.append(
                ("Error magnitude", np.asarray(data["validation/error_mag"], dtype=float))
            )
        return {
            "mask": mask,
            "axial_values": axial_values,
            "transverse_values": transverse_values,
            "products": products,
            "local_trajectory": local_trajectory,
            "spacecraft_ids": spacecraft_ids,
            "component_mode": component_mode,
            "background": background,
            "background_metadata": background_metadata,
        }


def _plot_slice_products(
    context: dict[str, Any],
    plot_path: str | Path,
    *,
    dpi: int = 150,
    magnitude_cmap: str = "batlow",
    error_cmap: str = "imola",
    title: str | None = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    mask = context["mask"]
    axial_values = context["axial_values"]
    transverse_values = context["transverse_values"]
    products = context["products"]
    local_trajectory = context["local_trajectory"]
    spacecraft_ids = context["spacecraft_ids"]
    n_axial = axial_values.size
    n_transverse = transverse_values.size
    component_cmap = _component_cmap()
    magnitude_colormap = get_scientific_cmap(magnitude_cmap, fallback="magma")
    error_colormap = get_scientific_cmap(error_cmap, fallback="inferno")

    figure, axes = plt.subplots(
        len(products),
        2,
        figsize=(10.5, 3.0 * len(products)),
        constrained_layout=True,
        squeeze=False,
    )
    axial_extent = [
        float(axial_values[0]),
        float(axial_values[-1]),
        float(transverse_values[0]),
        float(transverse_values[-1]),
    ]
    transverse_extent = [
        float(transverse_values[0]),
        float(transverse_values[-1]),
        float(transverse_values[0]),
        float(transverse_values[-1]),
    ]
    for row, (label, values) in enumerate(products):
        axial_image = _slice_to_image(
            values, mask, n_axial, n_transverse, mode="axial"
        )
        transverse_image = _slice_to_image(
            values, mask, n_axial, n_transverse, mode="transverse"
        )
        product_kind = _plot_product_kind(label)
        image_kwargs: dict[str, Any] = {}
        if product_kind == "component":
            finite = np.concatenate(
                (
                    axial_image[np.isfinite(axial_image)],
                    transverse_image[np.isfinite(transverse_image)],
                )
            )
            symmetric_limit = (
                1.0 if finite.size == 0 else max(float(np.max(np.abs(finite))), 1.0e-12)
            )
            image_kwargs["cmap"] = component_cmap
            image_kwargs["norm"] = TwoSlopeNorm(
                vmin=-symmetric_limit, vcenter=0.0, vmax=symmetric_limit
            )
        elif product_kind == "magnitude":
            image_kwargs["cmap"] = magnitude_colormap
        elif product_kind == "error":
            image_kwargs["cmap"] = error_colormap
        else:
            image_kwargs["cmap"] = magnitude_colormap
        for column, (slice_title, image, extent, xlabel, ylabel) in enumerate(
            (
                ("central axial slice", axial_image, axial_extent, "axial [km]", "u [km]"),
                (
                    "central transverse slice",
                    transverse_image,
                    transverse_extent,
                    "v [km]",
                    "u [km]",
                ),
            )
        ):
            axis = axes[row, column]
            image_artist = axis.imshow(
                image,
                origin="lower",
                extent=extent,
                aspect="auto" if column == 0 else "equal",
                **image_kwargs,
            )
            axis.set_title(f"{label}: {slice_title}")
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)
            if column == 1:
                axis.set_aspect("equal", adjustable="box")
            if _is_relative_error_label(label):
                contour_x = axial_values if column == 0 else transverse_values
                contour_y = transverse_values
                _add_error_contours(axis, contour_x, contour_y, image)
            if local_trajectory is not None:
                _overlay_trajectories(
                    axis,
                    local_trajectory,
                    "axial" if column == 0 else "transverse",
                    spacecraft_ids,
                )
            colorbar = figure.colorbar(image_artist, ax=axis)
            colorbar.set_label(label)
    if title is not None:
        figure.suptitle(title)
    output = Path(plot_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return output


def plot_rbf_reconstruction(
    hdf5_path: str | Path,
    plot_path: str | Path,
    *,
    dpi: int = 150,
    magnitude_cmap: str = "batlow",
    error_cmap: str = "imola",
    component_mode: str = "raw",
    background_mode: str = "metadata",
    background_vector: np.ndarray | None = None,
) -> Path:
    """Plot central axial and transverse slices of reconstructed magnetic field."""

    return _plot_slice_products(
        _read_plot_context(
            hdf5_path,
            "reconstruction/B",
            "reconstructed",
            component_mode=component_mode,
            background_mode=background_mode,
            background_vector=background_vector,
        ),
        plot_path,
        dpi=dpi,
        magnitude_cmap=magnitude_cmap,
        error_cmap=error_cmap,
    )


def _simulation_reference_B(
    rbf_hdf5_path: str | Path,
    observation_path: str | Path,
    generation_config_path: str | Path,
) -> np.ndarray:
    """Evaluate the original simulation snapshot on the saved RBF grid.

    This helper is intentionally outside the RBF fitter. It reads the original
    simulation only for validation/plotting, maps the unwrapped physical grid
    points into the same periodic simulation coordinates used during synthetic
    measurement generation, and returns ``B_actual`` in the generated output
    units.
    """

    loaded = load_generation_config(generation_config_path)
    field_config = loaded.data["field"]
    if str(field_config.get("model")) != "simulation_sampled":
        raise ValueError("Direct simulation reference requires field.model: simulation_sampled")
    field = SimulationSampledField(field_config)
    with h5py.File(rbf_hdf5_path, "r") as reconstruction_handle:
        grid_points = np.asarray(reconstruction_handle["grid/points"], dtype=float)
    with h5py.File(observation_path, "r") as observation_handle:
        if "spacecraft/r/B" in observation_handle:
            spacecraft_position = np.asarray(
                observation_handle["spacecraft/r/B"], dtype=float
            )
        elif "spacecraft/r" in observation_handle:
            spacecraft_position = np.asarray(observation_handle["spacecraft/r"], dtype=float)
        else:
            raise KeyError("Observation file is missing /spacecraft/r/B")
    if field.hub_spacecraft >= spacecraft_position.shape[0]:
        raise IndexError(
            f"Configured hub spacecraft {field.hub_spacecraft} is outside "
            f"{spacecraft_position.shape[0]} spacecraft"
        )
    initial_hub = spacecraft_position[field.hub_spacecraft, 0]
    mapped = np.mod(grid_points - initial_hub + 0.5 * field.box_size_km, field.box_size_km)
    actual = field.interpolator.sample(field.snapshot.B, mapped)
    actual *= field.scale_factors["B"]
    if field.fields_are["B"] == "fluctuation":
        actual += field.background["B"]
    return actual


def add_simulation_reference_to_rbf_output(
    rbf_hdf5_path: str | Path,
    observation_path: str | Path,
    generation_config_path: str | Path,
) -> Path:
    """Save direct simulation ``B_actual`` and RBF error magnitude in an RBF file."""

    output = Path(rbf_hdf5_path)
    actual = _simulation_reference_B(output, observation_path, generation_config_path)
    with h5py.File(output, "r+") as handle:
        reconstructed = np.asarray(handle["reconstruction/B"], dtype=float)
        if "reconstruction/Bmag" not in handle:
            handle["reconstruction"].create_dataset(
                "Bmag", data=np.linalg.norm(reconstructed, axis=-1), compression="gzip"
            )
        component_error = reconstructed - actual
        error_mag = np.linalg.norm(actual - reconstructed, axis=-1)
        actual_mag = np.linalg.norm(actual, axis=-1)
        error_percent = np.divide(
            100.0 * error_mag,
            actual_mag,
            out=np.full_like(error_mag, np.nan),
            where=actual_mag > 0.0,
        )
        error_percent_capped = np.minimum(error_percent, 100.0)
        if "validation" in handle:
            del handle["validation"]
        validation = handle.create_group("validation")
        validation.create_dataset("B_actual", data=actual, compression="gzip")
        validation.create_dataset("Bmag_actual", data=actual_mag, compression="gzip")
        validation.create_dataset(
            "component_error", data=component_error, compression="gzip"
        )
        validation.create_dataset("error_mag", data=error_mag, compression="gzip")
        validation.create_dataset(
            "error_magnitude", data=error_mag, compression="gzip"
        )
        validation.create_dataset(
            "error_percent", data=error_percent, compression="gzip"
        )
        validation.create_dataset(
            "error_percent_capped", data=error_percent_capped, compression="gzip"
        )
        validation.attrs["source"] = "direct_simulation_read"
        validation.attrs["error_percent_cap"] = 100.0
        validation.attrs["generation_config"] = str(Path(generation_config_path))
        validation.attrs["observation_path"] = str(Path(observation_path))
    return output


def plot_rbf_simulation_reference(
    hdf5_path: str | Path,
    plot_path: str | Path,
    *,
    dpi: int = 150,
    magnitude_cmap: str = "batlow",
    error_cmap: str = "imola",
    component_mode: str = "raw",
    background_mode: str = "metadata",
    background_vector: np.ndarray | None = None,
) -> Path:
    """Plot direct simulation values on the same slices as the RBF reconstruction."""

    return _plot_slice_products(
        _read_plot_context(
            hdf5_path,
            "validation/B_actual",
            "direct simulation",
            component_mode=component_mode,
            background_mode=background_mode,
            background_vector=background_vector,
        ),
        plot_path,
        dpi=dpi,
        magnitude_cmap=magnitude_cmap,
        error_cmap=error_cmap,
    )


def plot_rbf_broeren_uv_error(
    hdf5_path: str | Path,
    plot_path: str | Path,
    *,
    dpi: int = 150,
    error_cmap: str = "imola",
    capped: bool = True,
) -> Path:
    """Plot a Broeren-style u-v error map averaged along the cylinder axis.

    The map is a validation diagnostic, not part of the RBF fit. It averages
    point-wise relative error over the local axial direction, leaving a 2D
    transverse ``u-v`` view comparable to the paper's projected spacecraft
    geometry plots.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colormap = get_scientific_cmap(error_cmap, fallback="inferno")

    with h5py.File(hdf5_path, "r") as handle:
        if "validation" not in handle:
            raise ValueError("Broeren-style u-v error plot requires validation data")
        preferred = (
            "validation/error_percent_capped"
            if capped
            else "validation/error_percent"
        )
        fallback = "validation/error_percent"
        if preferred in handle:
            values = np.asarray(handle[preferred], dtype=float)
            title = (
                "Mean relative error capped at 100%"
                if preferred.endswith("capped")
                else "Mean relative error"
            )
        elif fallback in handle:
            values = np.asarray(handle[fallback], dtype=float)
            title = "Mean relative error"
        else:
            raise ValueError("RBF output does not contain validation/error_percent")
        mask = np.asarray(handle["grid/cylinder_mask"], dtype=bool)
        axial_values = np.asarray(handle["grid/axial_values"], dtype=float)
        transverse_values = np.asarray(handle["grid/transverse_values"], dtype=float)
        local_trajectory = _trajectory_local_coordinates(
            np.asarray(handle["samples/trajectory"], dtype=float),
            np.asarray(handle["grid/center"], dtype=float),
            np.asarray(handle["grid/basis_vectors"], dtype=float),
        )

    image = _mean_over_axial(
        values, mask, axial_values.size, transverse_values.size
    )
    extent = [
        float(transverse_values[0]),
        float(transverse_values[-1]),
        float(transverse_values[0]),
        float(transverse_values[-1]),
    ]
    figure, axis = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
    image_artist = axis.imshow(
        np.ma.masked_invalid(image),
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap=colormap,
    )
    _add_error_contours(axis, transverse_values, transverse_values, image)
    projected = local_trajectory[..., 1:3].reshape(-1, 2)
    axis.scatter(
        projected[:, 1],
        projected[:, 0],
        s=8,
        c="black",
        alpha=0.55,
        linewidths=0.0,
        zorder=5,
    )
    axis.scatter(
        [0.0],
        [0.0],
        marker="x",
        c="#D55E00",
        s=36,
        linewidths=1.2,
        zorder=6,
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("v [km]")
    axis.set_ylabel("u [km]")
    axis.set_title(f"{title}: axial average")
    colorbar = figure.colorbar(image_artist, ax=axis)
    colorbar.set_label("relative error [%]")
    output = Path(plot_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return output


def plot_rippa_sigma_selection(
    hdf5_path: str | Path,
    plot_path: str | Path,
    *,
    dpi: int = 150,
) -> Path:
    """Plot Rippa/geometry sigma-selection cost versus candidate sigma."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with h5py.File(hdf5_path, "r") as handle:
        if "sigma_selection/sigma_candidates" not in handle:
            raise ValueError("RBF output does not contain sigma-selection candidates")
        candidates = np.asarray(handle["sigma_selection/sigma_candidates"], dtype=float)
        costs = np.asarray(handle["sigma_selection/costs"], dtype=float)
        selected_sigma = float(handle["metadata"].attrs["sigma"])
        sigma_grid = str(handle["metadata"].attrs.get("sigma_grid", "linear"))
        method = str(handle["sigma_selection"].attrs.get("method", "sigma selection"))

    figure, axis = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    axis.plot(candidates, costs, marker="o", linewidth=1.2, markersize=3.0)
    axis.axvline(selected_sigma, color="#D55E00", linewidth=1.4, label="selected sigma")
    if sigma_grid == "log":
        axis.set_xscale("log")
    axis.set_xlabel("sigma [position units]")
    if "rippa" in method or "loocv" in method:
        axis.set_ylabel("LOOCV cost")
    else:
        axis.set_ylabel("geometry cost")
    axis.set_title(f"RBF sigma selection: {method}")
    axis.grid(True, alpha=0.25)
    axis.legend()
    output = Path(plot_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return output


@register_analysis
class RBFReconstructionAnalysis(AnalysisMethod):
    """Analysis-registry adapter for component-wise magnetic RBF reconstruction."""

    name = "rbf_reconstruction"
    config_section = "rbf_reconstruction"

    def run(self, input_path: str | Path) -> dict[str, Any]:
        return compute_rbf_reconstruction(input_path, self.config)

    def write_result(self, result: dict[str, Any], output_path: str | Path) -> Path:
        output = write_rbf_reconstruction(result, output_path)
        reference_settings = result["settings"]["reference"]
        if reference_settings["enabled"]:
            generation_config = reference_settings["generation_config"]
            if generation_config is None:
                raise ValueError(
                    "rbf_reconstruction.reference.generation_config is required "
                    "when reference.enabled is true"
                )
            observation_path = reference_settings["observation_path"] or result[
                "metadata"
            ]["input_path"]
            add_simulation_reference_to_rbf_output(
                output, observation_path, generation_config
            )
        plot_settings = result["settings"]["plot"]
        if plot_settings["enabled"]:
            plot_path = Path(
                plot_settings["path"] or output.with_name(f"{output.stem}.png")
            )
            plot_rbf_reconstruction(
                output,
                plot_path,
                dpi=plot_settings["dpi"],
                magnitude_cmap=plot_settings["magnitude_cmap"],
                error_cmap=plot_settings["error_cmap"],
                component_mode=plot_settings["component_mode"],
                background_mode=plot_settings["background_mode"],
                background_vector=plot_settings["background_vector"],
            )
        if reference_settings["enabled"] and reference_settings["plot"]["enabled"]:
            reference_plot = reference_settings["plot"]
            plot_path = Path(
                reference_plot["path"]
                or output.with_name(f"{output.stem}_simulation_reference.png")
            )
            plot_rbf_simulation_reference(
                output,
                plot_path,
                dpi=reference_plot["dpi"],
                magnitude_cmap=reference_plot["magnitude_cmap"],
                error_cmap=reference_plot["error_cmap"],
                component_mode=reference_plot["component_mode"],
                background_mode=reference_plot["background_mode"],
                background_vector=reference_plot["background_vector"],
            )
        broeren_error_plot = result["settings"]["broeren_error_plot"]
        if broeren_error_plot["enabled"]:
            plot_path = Path(
                broeren_error_plot["path"]
                or output.with_name(f"{output.stem}_broeren_uv_error.png")
            )
            plot_rbf_broeren_uv_error(
                output,
                plot_path,
                dpi=broeren_error_plot["dpi"],
                error_cmap=broeren_error_plot["error_cmap"],
                capped=broeren_error_plot["capped"],
            )
        if (
            result["settings"]["auto_sigma"]
            and result["settings"]["rippa_plot"]["enabled"]
        ):
            rippa_plot = result["settings"]["rippa_plot"]
            plot_path = Path(
                rippa_plot["path"] or output.with_name(f"{output.stem}_sigma.png")
            )
            plot_rippa_sigma_selection(output, plot_path, dpi=rippa_plot["dpi"])
        return output
