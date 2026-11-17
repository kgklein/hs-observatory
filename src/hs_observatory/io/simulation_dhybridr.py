"""Single frozen snapshot from the documented proton dHybridR NPZ release.

Arrays stay in code units. Spatial nodes are rebased from exported cell centers
to zero for HIVE's existing interpolator; vector components are never rotated.
NPZ members decompress in full, so copy one snapshot and release each history
before reading the next member. No external release loader is required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hs_observatory.io.simulation_npy import SimulationSnapshot


ARRAY_ORDER = "data[t, z, y, x]; coordinates are cell centres"
MU0 = 1.25663706212e-6  # N/A^2 (CODATA 2018)
PROTON_MASS_KG = 1.67262192369e-27
ELEMENTARY_CHARGE_C = 1.602176634e-19


def proton_normalization(n0_cm3: float, B0_nT: float) -> dict[str, float]:
    """Derive linked proton scales; reference parameters have no defaults."""
    if any(isinstance(value, (bool, np.bool_)) for value in (n0_cm3, B0_nT)):
        raise ValueError("n0_cm3 and B0_nT must be positive finite numbers")
    n0, B0 = float(n0_cm3), float(B0_nT)
    if not np.all(np.isfinite([n0, B0])) or min(n0, B0) <= 0:
        raise ValueError("n0_cm3 and B0_nT must be positive finite numbers")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        density = np.float64(n0) * 1e6
        magnetic = np.float64(B0) * 1e-9
        vA = magnetic / np.sqrt(MU0 * PROTON_MASS_KG * density) / 1000
        di = np.sqrt(PROTON_MASS_KG / (MU0 * density * ELEMENTARY_CHARGE_C**2)) / 1000
        omega = ELEMENTARY_CHARGE_C * magnetic / PROTON_MASS_KG
    if not np.all(np.isfinite([vA, di, omega])) or min(vA, di, omega) <= 0:
        raise ValueError("n0_cm3 and B0_nT produce invalid physical scales")
    return {
        "n0_cm3": n0,
        "B0_nT": B0,
        "vA_km_s": float(vA),
        "di_km": float(di),
        "Omega_ci_s_inv": float(omega),
    }


def _real_array(value: np.ndarray, name: str) -> np.ndarray:
    if value.dtype.kind not in "fiu" or not np.all(np.isfinite(value)):
        raise ValueError(f"dHybridR {name} must contain finite real numeric values")
    return value


def _metadata(archive: Any) -> dict[str, np.ndarray]:
    if archive["array_order"].shape != () or str(archive["array_order"]) != ARRAY_ORDER:
        raise ValueError(f"dHybridR array_order must declare {ARRAY_ORDER!r}")
    metadata = {key: archive[key] for key in ("iter", "time", "x", "y", "z")}
    iterations = metadata["iter"]
    if (
        iterations.ndim != 1
        or not iterations.size
        or iterations.dtype.kind not in "iu"
        or np.any(iterations < 0)
        or np.unique(iterations).size != iterations.size
    ):
        raise ValueError("dHybridR iter must contain unique nonnegative integer iterations")
    time = _real_array(metadata["time"], "time")
    if time.shape != iterations.shape or np.any(time < 0):
        raise ValueError("dHybridR time must match iter and be nonnegative")
    for name in ("x", "y", "z"):
        axis = _real_array(metadata[name], name)
        if axis.ndim != 1 or axis.size < 2:
            raise ValueError(f"dHybridR {name} must be a 1D cell-center axis with >=2 cells")
        step = float(axis[1] - axis[0])
        if step <= 0 or not np.allclose(np.diff(axis), step, rtol=1e-10, atol=1e-12):
            raise ValueError(f"dHybridR {name} coordinates must be increasing and uniform")
        # This release exports centers of a zero-edge-origin domain, in d_i.
        if not np.isclose(axis[0], step / 2, rtol=1e-10, atol=1e-12):
            raise ValueError(f"dHybridR {name} must start at a half-cell center")
    return metadata


def _selected_member(
    archive: Any, name: str, unit: str, index: int, shape: tuple[int, ...]
) -> np.ndarray:
    units = archive[f"units_{name}"]
    if units.shape != () or str(units) != unit:
        raise ValueError(f"dHybridR units_{name} must be {unit!r}")
    history = archive[name]
    if history.shape != shape or history.dtype.kind not in "fiu":
        raise ValueError(f"dHybridR {name} must be real [time,z,y,x] with shape {shape}")
    selected = np.array(history[index].transpose(2, 1, 0), dtype=float, copy=True)
    del history
    return _real_array(selected, name)


def load_simulation_dhybridr(
    fields_path: str | Path,
    moments_path: str | Path,
    iteration: int,
) -> SimulationSnapshot:
    """Read seven total quantities at one explicit iteration, in code units.

    The strict release contract supplies cell-center axes in d_i, time in
    Omega_ci^-1, proton number density, and direct fluid velocity in v_A.
    Boundary/frame conventions absent from NPZ are recorded from release docs.
    """
    if isinstance(iteration, (bool, np.bool_)) or not isinstance(iteration, (int, np.integer)):
        raise ValueError("dHybridR iteration must be an explicit integer iteration number")
    paths = [Path(path).expanduser() for path in (fields_path, moments_path)]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"dHybridR archive does not exist: {path}")
        if path.suffix.lower() != ".npz":
            raise ValueError(f"dHybridR archive must use .npz format: {path}")
    with (
        np.load(paths[0], allow_pickle=False) as fields,
        np.load(paths[1], allow_pickle=False) as moments,
    ):
        try:
            fmeta, mmeta = _metadata(fields), _metadata(moments)
            for key in fmeta:
                if not np.array_equal(fmeta[key], mmeta[key]):
                    raise ValueError(f"dHybridR fields/moments {key} arrays must agree")
            matches = np.flatnonzero(fmeta["iter"] == iteration)
            if matches.size != 1:
                raise ValueError(f"dHybridR iteration {iteration} is not present in both archives")
            index = int(matches[0])
            axes = [fmeta[name] for name in ("x", "y", "z")]
            shape = (len(fmeta["iter"]), *(len(axis) for axis in axes[::-1]))
            # Only selected snapshots survive each call; no list of histories.
            n = _selected_member(moments, "n", "n_0 e", index, shape)
            v = np.empty((*n.shape, 3))
            B = np.empty_like(v)
            for component, name in enumerate(("Vx", "Vy", "Vz")):
                v[..., component] = _selected_member(moments, name, "v_A", index, shape)
            for component, name in enumerate(("Bx", "By", "Bz")):
                B[..., component] = _selected_member(fields, name, "B_0", index, shape)
        except KeyError as exc:
            raise ValueError(f"Missing required dHybridR NPZ member: {exc}") from exc
    spacing = [float(axis[1] - axis[0]) for axis in axes]
    layout = {
        "schema": "dhybridr_npz",
        "array_order": ARRAY_ORDER,
        "iteration": int(iteration),
        "snapshot_index": index,
        "snapshot_time_Omega_ci_inv": float(fmeta["time"][index]),
        "origin_di": [float(axis[0]) for axis in axes],
        "spacing_di": spacing,
        "box_size_di": [len(axis) * step for axis, step in zip(axes, spacing)],
        "cell_center_interval_di": [[float(axis[0]), float(axis[-1])] for axis in axes],
        "source_boundaries": {"x": ["reflecting", "open"], "y": "periodic", "z": "periodic"},
        "source_frame": "downstream_rest",
        "spatial_units": "d_i",
        "source_units": {"n": "n_0 e", "v": "v_A", "B": "B_0"},
        "fields_are": "total",
        "species": "proton_charge_state_1",
        "grid_location": "cell_centers",
        "axis_alignment": "identity",
    }
    return SimulationSnapshot(n, v, B, shape, layout, ("n", "v", "B"))
