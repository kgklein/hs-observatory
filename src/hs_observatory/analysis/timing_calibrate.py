"""Linear timing calibration of synthetic magnetic measurements, Hub clock fixed.

Timing corrections c sample measured B at t+c; recovered injected offsets are -c.
By default the ideal series determines only fixed divergence weights.
The use_ideal_divergence option instead fits measured minus ideal divergence.
"""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterator, Mapping
import warnings

import h5py
import numpy as np
import yaml

from hs_observatory import __version__
from hs_observatory.analysis.base import AnalysisMethod
from hs_observatory.analysis.magnetic_gradient import (
    _git_commit,
    _settings as gradient_settings,
    _write_metadata,
    align_spacecraft_samples,
    apply_gradient_response,
    geometry_diagnostics,
    geometry_is_static,
    gradient_curl_divergence,
    gradient_response_coefficients,
    select_configurations,
)
from hs_observatory.analysis.registry import register_analysis
from hs_observatory.errors.observatory import (
    normalized_observatory_config,
    observatory_errors_enabled,
)
from hs_observatory.errors.offsets import time_offset_model_from_config


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    # Normalize legacy boolean category switches before sharing gradient settings.
    selection = dict(config.get("configuration_selection", {}))
    for name in ("small", "intermediate", "large", "best_multi", "all_nine"):
        if name in selection and isinstance(selection[name], bool):
            selection[name] = {"enabled": selection[name]}
    settings = gradient_settings({**config, "configuration_selection": selection})
    polyhedron_filter = config.get("polyhedron_filter")
    if polyhedron_filter is None:
        polyhedron_filter = {}
    if not isinstance(polyhedron_filter, Mapping):
        raise ValueError("polyhedron_filter must be a mapping or null")
    allowed_n = polyhedron_filter.get("allowed_n")
    if allowed_n is not None and (
        not isinstance(allowed_n, list)
        or any(type(n) is not int or not 4 <= n <= 9 for n in allowed_n)
    ):
        raise ValueError(
            "polyhedron_filter.allowed_n must be a list of integers from 4 to 9 or null"
        )
    maximum = polyhedron_filter.get("max_condition_number")
    if maximum is not None and (
        isinstance(maximum, bool)
        or not isinstance(maximum, (int, float))
        or not np.isfinite(maximum)
        or maximum <= 0
    ):
        raise ValueError(
            "polyhedron_filter.max_condition_number must be finite and positive or null"
        )
    settings["polyhedron_filter"] = {
        "allowed_n": allowed_n,
        "max_condition_number": maximum,
    }
    use_ideal = config.get("use_ideal_divergence", False)
    if not isinstance(use_ideal, bool):
        raise ValueError("use_ideal_divergence must be a boolean")
    settings["use_ideal_divergence"] = use_ideal
    influence = dict(config.get("diagnostics", {})).get("polyhedron_influence", False)
    if not isinstance(influence, bool):
        raise ValueError("diagnostics.polyhedron_influence must be a boolean")
    if influence and use_ideal:
        raise ValueError("Polyhedron influence requires use_ideal_divergence: false")
    settings["polyhedron_influence"] = influence
    method = config.get("calibration_method", "polyhedron")
    if method not in {
        "polyhedron",
        "all_nine_divergence",
        "all_nine_df_affine",
        "all_nine_df_quadratic",
    }:
        raise ValueError("Unknown timing calibration_method")
    settings["calibration_method"] = method
    normalization = config.get("divergence_normalization", "ideal_divergence")
    if normalization not in {"ideal_divergence", "instrument_noise"}:
        raise ValueError("divergence_normalization must be ideal_divergence or instrument_noise")
    if normalization == "instrument_noise" and method != "polyhedron":
        raise ValueError("Divergence normalization applies to the polyhedron method only")
    settings["divergence_normalization"] = normalization
    if method != "polyhedron":
        if use_ideal or influence:
            raise ValueError(
                "Polynomial projection forbids use_ideal_divergence and polyhedron_influence"
            )
        if any(
            config.get(key) is not None
            for key in ("polyhedra", "configuration_selection", "polyhedron_filter")
        ):
            raise ValueError("All-nine projection does not support polyhedron selection/filtering")
        sigma_B = config.get("sigma_B_nT")
        if (
            isinstance(sigma_B, bool)
            or not isinstance(sigma_B, (int, float))
            or not np.isfinite(sigma_B)
            or sigma_B <= 0
            or sigma_B > np.sqrt(np.finfo(float).max)
            or sigma_B < np.sqrt(np.nextafter(0.0, 1.0))
        ):
            raise ValueError("Polynomial projection requires finite positive sigma_B_nT")
        settings["sigma_B_nT"] = float(sigma_B)
        if method == "all_nine_df_quadratic":
            maximum = config.get("quadratic_condition_number_max", 1e10)
            if (
                isinstance(maximum, bool)
                or not isinstance(maximum, (int, float))
                or not np.isfinite(maximum)
                or maximum < 1
            ):
                raise ValueError("quadratic_condition_number_max must be finite and at least one")
            settings["quadratic_condition_number_max"] = float(maximum)
    if not settings["store_all_realizations"]:
        raise ValueError("timing-calibrate requires monte_carlo.store_all_realizations: true")
    if settings["time_sync"]["allow_extrapolation"]:
        raise ValueError("timing-calibrate does not extrapolate corrected fields")
    return settings


def select_polyhedra(
    positions: np.ndarray, spacecraft_ids: list[str], config: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    """Select all subsets by default, explicit members, or legacy categories."""

    if "polyhedra" in config and "configuration_selection" in config:
        raise ValueError("Specify polyhedra or configuration_selection, not both")
    if "configuration_selection" in config:
        selected = select_configurations(positions, **_settings(config)["selection"])
        return {name: value["indices"] for name, value in selected.items()}
    if "polyhedra" in config:
        members = config["polyhedra"]
        if not isinstance(members, list) or not members:
            raise ValueError("polyhedra must be a nonempty list of spacecraft-ID lists")
        subsets = []
        for group in members:
            if not isinstance(group, list) or not 4 <= len(group) <= 9:
                raise ValueError("Each polyhedron requires 4 to 9 spacecraft IDs")
            if len(set(group)) != len(group) or any(item not in spacecraft_ids for item in group):
                raise ValueError("Polyhedron members must be unique existing spacecraft IDs")
            subsets.append(tuple(sorted(spacecraft_ids.index(item) for item in group)))
        if len(set(subsets)) != len(subsets):
            raise ValueError("Duplicate polyhedra are not supported")
        subsets.sort(key=lambda item: (len(item), item))
    else:
        subsets = [group for size in range(4, 10) for group in combinations(range(9), size)]
    return {"_".join(spacecraft_ids[i] for i in group): np.asarray(group) for group in subsets}


def _filter_polyhedra(
    positions: np.ndarray,
    candidates: Mapping[str, np.ndarray],
    options: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Intersect candidates using the same representative geometry as timing influence."""

    selected, conditions = {}, {}
    allowed_n = options["allowed_n"]
    maximum = options["max_condition_number"]
    for name, indices in candidates.items():
        if allowed_n is not None and len(indices) not in allowed_n:
            continue
        points = positions[:, indices]
        samples = points[:1] if geometry_is_static(points) else points
        condition = float(
            np.median([geometry_diagnostics(sample).condition_number for sample in samples])
        )
        if maximum is not None and not condition <= maximum:
            continue
        selected[name] = indices
        conditions[name] = condition
    if not selected:
        raise ValueError("polyhedron_filter retained no polyhedra from the candidate selection")
    return selected, conditions


def divergence_series(response: np.ndarray, field: np.ndarray) -> np.ndarray:
    return gradient_curl_divergence(apply_gradient_response(response, field))[2]


def divergence_variance(divergence: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray]:
    """Fixed mean-square about zero, with a floor on standard deviation."""

    floor_squared = floor * floor
    if not np.isfinite(floor_squared) or floor <= 0 or floor_squared == 0:
        raise ValueError("sigma_d_floor must be finite, positive, and safely representable")
    valid = np.isfinite(divergence)
    count = valid.sum(axis=0)
    raw = np.full(divergence.shape[1], np.nan)
    np.divide((np.where(valid, divergence, 0.0) ** 2).sum(axis=0), count, out=raw, where=count > 0)
    return raw, np.maximum(raw, floor_squared)


def timing_jacobian(
    time: np.ndarray,
    measured: np.ndarray,
    polyhedra: list[np.ndarray],
    responses: list[np.ndarray],
    node_indices: np.ndarray,
) -> np.ndarray:
    """Divergence derivative with respect to field-sampling corrections (seconds)."""

    derivative = np.gradient(measured, time, axis=0, edge_order=1)
    jacobian = np.zeros((len(time), len(polyhedra), len(node_indices)))
    columns = {int(member): column for column, member in enumerate(node_indices)}
    for polyhedron, (indices, response) in enumerate(zip(polyhedra, responses)):
        for local, member in enumerate(indices):
            if int(member) in columns:
                jacobian[:, polyhedron, columns[int(member)]] = np.sum(
                    response[:, local] * derivative[:, member], axis=-1
                )
    return jacobian


def _spectrum(matrix: np.ndarray) -> dict[str, Any]:
    values = np.linalg.svd(matrix, compute_uv=False)
    values = np.pad(values, (0, matrix.shape[1] - len(values)))
    tolerance = np.finfo(float).eps * max(matrix.shape) * values[0]
    rank = int(np.count_nonzero(values > tolerance))
    condition = float(values[0] / values[-1]) if rank == matrix.shape[1] else np.inf
    return {"singular_values": values, "condition_number": condition, "rank": rank}


def solve_timing(
    divergence: np.ndarray,
    jacobian: np.ndarray,
    variance: np.ndarray,
    sigma_t: float,
    *,
    ideal_divergence: np.ndarray | None = None,
) -> dict[str, Any]:
    """Solve the explicitly augmented linear least-squares system, not normal equations.

    An optional ideal reference changes the target to measured minus ideal
    divergence. Callers omitting it retain the absolute-divergence objective.
    Returned linear chi-square values
    describe the chosen target; the parameter and alignment signs are unchanged.
    """

    if not np.isfinite(sigma_t) or sigma_t <= 0:
        raise ValueError("sigma_t_s must be finite and positive")
    if ideal_divergence is not None:
        if np.shape(ideal_divergence) != np.shape(divergence):
            raise ValueError("ideal_divergence must have the same shape as measured divergence")
        divergence = np.asarray(divergence) - np.asarray(ideal_divergence)
    valid = (
        np.isfinite(divergence)
        & np.all(np.isfinite(jacobian), axis=-1)
        & np.isfinite(variance)[None, :]
        & (variance[None, :] > 0)
    )
    if not np.any(valid):
        raise ValueError("No valid timing-calibration fit rows")
    rows = jacobian[valid]
    unweighted = _spectrum(rows)
    sigma = np.broadcast_to(np.sqrt(variance), divergence.shape)[valid]
    rows /= sigma[:, None]
    weighted = _spectrum(rows)
    n_parameters = jacobian.shape[-1]
    design = np.empty((len(rows) + n_parameters, n_parameters))
    design[:-n_parameters] = rows
    design[-n_parameters:] = np.eye(n_parameters) / sigma_t
    del rows
    target = np.concatenate((-divergence[valid] / sigma, np.zeros(n_parameters)))
    correction, _, rank, singular_values = np.linalg.lstsq(design, target, rcond=None)
    # QR avoids allocating a tall U; SVD of R gives the same normal-matrix covariance.
    triangular = np.linalg.qr(design, mode="r")
    _, singular, right = np.linalg.svd(triangular, full_matrices=False)
    retained = singular > np.finfo(float).eps * max(design.shape) * singular[0]
    inverse = np.zeros_like(singular)
    inverse[retained] = 1.0 / singular[retained] ** 2
    covariance = (right.T * inverse) @ right
    residual = design @ correction - target
    prior = float(np.sum((correction / sigma_t) ** 2))
    before = float(target @ target)
    after_divergence = float(residual[:-n_parameters] @ residual[:-n_parameters])
    return {
        "correction_s": correction,
        "recovered_offset_s": -correction,
        "covariance_s2": covariance,
        "fit_valid": valid,
        "jacobian": unweighted,
        "weighted_jacobian": weighted,
        "augmented_singular_values": singular_values,
        "augmented_rank": int(rank),
        "linear_chi2_before": before,
        "linear_chi2_after": after_divergence + prior,
        "linear_divergence_chi2_before": before,
        "linear_divergence_chi2_after": after_divergence,
        "prior_chi2_before": 0.0,
        "prior_chi2_after": prior,
    }


def solve_timing_with_variance(divergence, jacobian, variance, sigma_t, *, ideal_divergence=None):
    """Adapt time-dependent row variances to the unchanged least-squares solver."""
    if np.ndim(variance) == 1:
        return solve_timing(
            divergence, jacobian, variance, sigma_t, ideal_divergence=ideal_divergence
        )
    grid = np.broadcast_to(variance, divergence.shape)
    result = solve_timing(
        divergence.reshape(1, -1),
        jacobian.reshape(1, -1, jacobian.shape[-1]),
        grid.ravel(),
        sigma_t,
        ideal_divergence=None if ideal_divergence is None else ideal_divergence.reshape(1, -1),
    )
    result["fit_valid"] = result["fit_valid"].reshape(divergence.shape)
    return result


def calibrate_realization(
    time: np.ndarray,
    positions: np.ndarray,
    measured: np.ndarray,
    polyhedra: list[np.ndarray],
    responses: list[np.ndarray],
    node_indices: np.ndarray,
    variance: np.ndarray,
    sigma_t: float,
    *,
    ideal_divergence: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit once, resample B, then independently evaluate corrected divergence."""

    before = np.column_stack(
        [
            divergence_series(response, measured[:, indices])
            for indices, response in zip(polyhedra, responses)
        ]
    )
    jacobian = timing_jacobian(time, measured, polyhedra, responses, node_indices)
    result = solve_timing_with_variance(
        before, jacobian, variance, sigma_t, ideal_divergence=ideal_divergence
    )
    del jacobian
    offsets = np.zeros(9)
    offsets[node_indices] = -result["correction_s"]
    aligned = align_spacecraft_samples(time, offsets, positions.swapaxes(0, 1), measured)
    after = np.column_stack(
        [
            divergence_series(response, aligned.field[:, indices])
            for indices, response in zip(polyhedra, responses)
        ]
    )
    valid = result["fit_valid"] & np.isfinite(after)
    if not np.any(valid):
        raise ValueError("No common valid rows remain after timing correction")
    for label, divergence in (("before", before), ("after", after)):
        normalized_squared = np.zeros_like(divergence)
        np.divide(divergence**2, variance, out=normalized_squared, where=valid)
        contribution = float(normalized_squared.sum())
        objective = contribution
        if ideal_divergence is not None:
            residual = divergence - ideal_divergence
            objective = float(np.sum((residual**2 / variance)[valid]))
        result[f"divergence_chi2_{label}"] = objective
        result[f"chi2_{label}"] = objective + result[f"prior_chi2_{label}"]
        total = result[f"chi2_{label}"]
        result[f"divergence_objective_fraction_{label}"] = (
            objective / total if total > 0 else np.nan
        )
        result[f"prior_objective_fraction_{label}"] = (
            result[f"prior_chi2_{label}"] / total if total > 0 else np.nan
        )
        result[f"normalized_divergence_rms_{label}"] = np.sqrt(contribution / valid.sum())
        per_polyhedron = np.full(len(polyhedra), np.nan)
        np.divide(
            normalized_squared.sum(axis=0),
            valid.sum(axis=0),
            out=per_polyhedron,
            where=valid.sum(axis=0) > 0,
        )
        result[f"polyhedron_rms_{label}"] = np.sqrt(per_polyhedron)
    result.update(
        measured_B=measured,
        corrected_B=aligned.field,
        corrected_B_valid=aligned.valid,
        measured_divergence=before,
        corrected_divergence=after,
        comparison_valid=valid,
        comparison_row_count=int(valid.sum()),
        fit_row_count=int(result["fit_valid"].sum()),
        chi2_increased=bool(result["chi2_after"] > result["chi2_before"]),
    )
    return result


def _validate_errors(generation: Mapping[str, Any], handle: h5py.File) -> Mapping[str, Any]:
    errors = generation.get("errors", {})
    if observatory_errors_enabled(errors):
        normalized = normalized_observatory_config(errors)
        for name, section in normalized.items():
            if name != "timing" and isinstance(section, Mapping) and section.get("enabled", True):
                raise ValueError(f"timing-calibrate requires timing-only errors; {name} is enabled")
        timing = normalized["timing"]
    else:
        timing = errors.get("time", {})
        for name, section in errors.get("variables", {}).items():
            if section.get("model", "none") not in {"none", "no_error"}:
                raise ValueError(f"timing-calibrate does not support {name} measurement errors")
        if errors.get("position", {}).get("model", "none") not in {"none", "no_error"}:
            raise ValueError("timing-calibrate requires no position errors")
    if "errors/position_offset" in handle and np.any(handle["errors/position_offset"][...] != 0):
        raise ValueError("timing-calibrate requires zero position offsets")
    # Also validates the supported constant-per-spacecraft timing model.
    time_offset_model_from_config(dict(timing))
    return timing


def compute_timing_calibration(input_path: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Prepare shared products; MC calculations stream during product writing."""

    source = Path(input_path).resolve()
    settings = _settings(config)
    floor = float(config.get("sigma_d_floor", 1.0e-12))
    with h5py.File(source, "r") as handle:
        metadata = dict(handle["metadata"].attrs)
        generation = yaml.safe_load(metadata.get("config_yaml", ""))
        if not isinstance(generation, dict) or "errors" not in generation:
            raise ValueError("Embedded generation config_yaml with errors is required")
        timing = _validate_errors(generation, handle)
        ids = list(handle["spacecraft/id"].asstr()[...])
        if len(ids) != 9 or set(ids) != {"H", *(f"N{i}" for i in range(1, 9))}:
            raise ValueError("Expected H and N1 through N8, each exactly once")
        time = np.asarray(handle["time/B"], dtype=float)
        positions = np.asarray(handle["spacecraft/r/B"], dtype=float).swapaxes(0, 1)
        ideal = np.asarray(handle["truth/B"], dtype=float)
        observed_shape = handle["observed/B"].shape
        offsets = np.asarray(handle["errors/time_offset"], dtype=float)
        if settings["polyhedron_influence"]:
            if np.any(offsets != 0):
                raise ValueError("Polyhedron influence requires zero injected timing offsets")
            for measured in handle["observed/B"]:
                if not np.array_equal(measured, ideal, equal_nan=True):
                    raise ValueError("Polyhedron influence requires measured B equal to ideal B")
    if time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("time/B must contain at least two finite, strictly increasing samples")
    if ideal.shape != (len(time), 9, 3) or positions.shape != ideal.shape:
        raise ValueError("Expected B(time, 9, 3) and positions(9, time, 3)")
    if not np.isfinite(positions).all():
        raise ValueError("Nominal positions must be finite")
    if len(observed_shape) != 4 or observed_shape[1:] != ideal.shape or observed_shape[0] < 1:
        raise ValueError("observed/B must have shape (mc >= 1, time, 9, 3)")
    if offsets.shape != (observed_shape[0], 9) or not np.isfinite(offsets).all():
        raise ValueError("errors/time_offset must have shape (mc, 9) with finite constant offsets")
    if np.any(offsets[:, ids.index("H")] != 0):
        raise ValueError("Hub offsets must be zero; regenerate with timing.reference_spacecraft: H")
    model = time_offset_model_from_config(dict(timing))
    sigma_t = float(config.get("sigma_t_s", getattr(model, "sigma_s", np.nan)))
    if not np.isfinite(sigma_t) or sigma_t <= 0:
        raise ValueError("Supply positive sigma_t_s or a positive generation timing uncertainty")
    projection = None
    noise = None
    if settings["calibration_method"] != "polyhedron":
        from hs_observatory.analysis.timing_affine import affine_projection_bases

        if settings["calibration_method"] == "all_nine_df_quadratic":
            from hs_observatory.analysis.timing_affine import quadratic_projection_bases

            projection = quadratic_projection_bases(
                positions,
                condition_number_max=settings["quadratic_condition_number_max"],
            )
        else:
            projection = affine_projection_bases(positions)
        selected, conditions, responses = {}, {}, []
        candidate_count = 0
        ideal_divergence = np.empty((len(time), 0))
        raw = variance = np.empty(0)
    else:
        selected = select_polyhedra(positions[0], ids, config)
        if not selected:
            raise ValueError("At least one polyhedron must be selected")
        candidate_count = len(selected)
        selected, conditions = _filter_polyhedra(positions, selected, settings["polyhedron_filter"])
        responses = [
            gradient_response_coefficients(positions[:, indices], **settings["gradient"])
            for indices in selected.values()
        ]
        ideal_divergence = np.column_stack(
            [
                divergence_series(response, ideal[:, indices])
                for indices, response in zip(selected.values(), responses)
            ]
        )
        raw, variance = divergence_variance(ideal_divergence, floor)
        ideal_reference_variance = variance
        if settings["divergence_normalization"] == "instrument_noise":
            from hs_observatory.analysis.timing_noise import prepare_noise

            noise, raw, variance = prepare_noise(time, positions, list(selected.values()), config)
            noise["ideal_reference_variance"] = ideal_reference_variance
    nodes = np.asarray([i for i, name in enumerate(ids) if name != "H"])
    result = {
        "source": source,
        "time": time,
        "positions": positions,
        "ideal_B": ideal,
        "spacecraft_ids": ids,
        "node_indices": nodes,
        "selected": selected,
        "condition_numbers": conditions,
        "responses": responses,
        "ideal_divergence": ideal_divergence,
        "raw_variance": raw,
        "variance": variance,
        "sigma_t": sigma_t,
        "offsets": offsets[:, nodes],
        "n_mc": observed_shape[0],
        "settings": settings,
        "metadata": {
            "method": "timing-calibrate",
            "source_file": str(source),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "code_version": __version__,
            "git_commit": _git_commit(),
            "schema_version": "1",
            "generation_config_yaml": metadata["config_yaml"],
            "analysis_config_yaml": config.get(
                "_analysis_config_yaml", yaml.safe_dump(dict(config))
            ),
            "analysis_config_path": config.get("_analysis_config_path", ""),
            "sigma_t_s": sigma_t,
            "sigma_d_floor_nT_per_km": floor,
            "use_ideal_divergence": settings["use_ideal_divergence"],
            "chi2_target": (
                "measured_minus_ideal"
                if settings["use_ideal_divergence"]
                else "absolute_divergence"
            ),
            "normalized_divergence_rms_target": "absolute_divergence",
            "sign_convention": "measured=B_ideal(t+tau); corrected=B_measured(t+c); recovered_tau=-c",
            "hub_id": "H",
            "shared_ideal_applies_to_all_mc": True,
            "covariance": "SVD of augmented design via QR; no residual variance rescaling",
            "comparison": "before/after use identical comparison_valid rows; fit uses fit_valid",
            "settings": settings,
            "polyhedron_filter": {
                **settings["polyhedron_filter"],
                "candidate_count": candidate_count,
                "retained_count": len(selected),
                "condition_number_convention": "first sample for static relative geometry; "
                "median over analysis time otherwise",
            },
        },
    }
    if noise is not None:
        result["noise"] = noise
        result["metadata"].update(
            divergence_normalization="instrument_noise",
            sigma_B_nT=noise["sigma_B_nT"],
            schema_version="2",
        )
        result["metadata"].pop("sigma_d_floor_nT_per_km")
    if projection is not None:
        result["projection"] = projection
        result["metadata"].update(
            schema_version="2",
            calibration_method=settings["calibration_method"],
            sigma_B_nT=settings["sigma_B_nT"],
            chi2_target=settings["calibration_method"],
            residual_space="common scalar whitening: B / sigma_B_nT",
        )
        result["metadata"].pop("normalized_divergence_rms_target")
        result["metadata"].pop("polyhedron_filter")
    if settings["polyhedron_influence"]:
        from hs_observatory.analysis.timing_influence import compute_timing_influence

        result["influence"] = compute_timing_influence(result)
    return result


def _realizations(result: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    with h5py.File(result["source"], "r") as handle:
        for mc in range(result["n_mc"]):
            if "projection" in result:
                from hs_observatory.analysis.timing_affine import calibrate_affine_realization

                yield calibrate_affine_realization(
                    result["time"],
                    result["positions"],
                    handle["observed/B"][mc],
                    result["node_indices"],
                    result["sigma_t"],
                    result["projection"],
                    result["settings"]["calibration_method"],
                    result["settings"]["sigma_B_nT"],
                )
                continue
            yield calibrate_realization(
                result["time"],
                result["positions"],
                handle["observed/B"][mc],
                list(result["selected"].values()),
                result["responses"],
                result["node_indices"],
                result["variance"],
                result["sigma_t"],
                ideal_divergence=(
                    result["ideal_divergence"]
                    if result["settings"]["use_ideal_divergence"]
                    else None
                ),
            )


def _store(
    group: h5py.Group, name: str, value: Any, settings: Mapping[str, Any], units: str = "1"
) -> h5py.Dataset:
    array = np.asarray(value)
    options: dict[str, Any] = {}
    if array.ndim and array.size > 1 and settings["compression"]:
        options.update(compression=settings["compression"], chunks=True)
        if settings["compression"] == "gzip":
            options["compression_opts"] = settings["compression_level"]
    dataset = group.create_dataset(name, data=array, **options)
    dataset.attrs["units"] = units
    return dataset


def _store_mc(
    group: h5py.Group,
    name: str,
    value: Any,
    mc: int,
    count: int,
    settings: Mapping[str, Any],
    units: str = "1",
) -> None:
    array = np.asarray(value)
    if name not in group:
        options: dict[str, Any] = {"chunks": (1, *array.shape)}
        if array.ndim:
            options["chunks"] = (1, min(array.shape[0], settings["chunk_time"]), *array.shape[1:])
        if settings["compression"]:
            options["compression"] = settings["compression"]
            if settings["compression"] == "gzip":
                options["compression_opts"] = settings["compression_level"]
        dataset = group.create_dataset(
            name, shape=(count, *array.shape), dtype=array.dtype, **options
        )
        dataset.attrs["units"] = units
    group[name][mc] = array


def write_timing_calibration_product(result: Mapping[str, Any], output_path: str | Path) -> Path:
    """Write all realizations incrementally, keeping only one MC fit in memory."""

    output = Path(output_path)
    if output.resolve() == result["source"]:
        raise ValueError("Analysis output must not overwrite the observation input")
    output.parent.mkdir(parents=True, exist_ok=True)
    settings = result["settings"]
    if settings["chunk_time"] < 1:
        raise ValueError("storage.chunk_time must be positive")
    with h5py.File(output, "w") as handle:
        metadata = handle.create_group("metadata")
        _write_metadata(metadata, result["metadata"])
        metadata.attrs["complete"] = False
        _store(handle.create_group("time"), "analysis_time", result["time"], settings, "s")
        spacecraft = handle.create_group("spacecraft")
        for name, values in (
            ("id", result["spacecraft_ids"]),
            ("node_id", [result["spacecraft_ids"][i] for i in result["node_indices"]]),
        ):
            spacecraft.create_dataset(name, data=np.asarray(values, dtype=h5py.string_dtype()))
        _store(spacecraft, "node_indices", result["node_indices"], settings)
        _store(spacecraft, "positions", result["positions"], settings, "km")
        fields = handle.create_group("magnetic_field")
        _store(fields, "ideal", result["ideal_B"], settings, "nT")
        configurations = handle.create_group("configurations")
        names = list(result["selected"])
        configurations.attrs["order"] = np.asarray(names, dtype=h5py.string_dtype())
        for column, (name, indices) in enumerate(result["selected"].items()):
            group = configurations.create_group(name)
            group.attrs["number_of_spacecraft"] = len(indices)
            _store(group, "condition_number", result["condition_numbers"][name], settings)
            _store(group, "spacecraft_indices", indices, settings)
            group.create_dataset(
                "spacecraft_ids",
                data=np.asarray(
                    [result["spacecraft_ids"][i] for i in indices], dtype=h5py.string_dtype()
                ),
            )
            _store(
                group, "ideal_divergence", result["ideal_divergence"][:, column], settings, "nT/km"
            )
            for key, values in (
                ("sigma_d_squared_raw", result["raw_variance"]),
                ("sigma_d_squared", result["variance"]),
            ):
                _store(group, key, values[..., column], settings, "(nT/km)^2")
            if "noise" in result:
                _store(
                    group, "sigma_d", np.sqrt(result["variance"][..., column]), settings, "nT/km"
                )
        if "noise" in result:
            from hs_observatory.analysis.timing_noise import write_noise

            write_noise(handle, result)
        if "influence" in result:
            from hs_observatory.analysis.timing_influence import write_timing_influence

            write_timing_influence(handle, result)
        if "projection" in result:
            from hs_observatory.analysis.timing_affine import project_field, projection_components

            projection = handle.create_group("projection")
            components = projection_components(result["projection"])
            quadratic = components == ("quadratic",)
            projection.attrs.update(
                axes="geometry sample, measurement component, residual coordinate",
                static_geometry=len(
                    result["projection"]["quadratic_basis" if quadratic else "df_basis"]
                )
                == 1,
            )
            if quadratic:
                projection.attrs.update(
                    quadratic_design_rank=26,
                    quadratic_residual_rank=1,
                    quadratic_condition_number_max=settings["quadratic_condition_number_max"],
                )
            else:
                projection.attrs.update(
                    affine_rank=12,
                    df_rank=11,
                    non_affine_rank=15,
                    divergence_rank=1,
                    df_residual_rank=16,
                )
            for key, value in result["projection"].items():
                _store(projection, key, value, settings)
            for component in components:
                _store(
                    projection,
                    f"ideal_{component}_residual",
                    project_field(result["projection"][f"{component}_basis"], result["ideal_B"]),
                    settings,
                    "nT",
                )
        timing = handle.create_group("timing")
        _store(timing, "injected_offset_s", result["offsets"], settings, "s")
        diagnostics = handle.create_group("diagnostics")
        for mc, realization in enumerate(_realizations(result)):
            for key in ("correction_s", "recovered_offset_s", "covariance_s2"):
                _store_mc(
                    timing,
                    key,
                    realization[key],
                    mc,
                    result["n_mc"],
                    settings,
                    "s^2" if key == "covariance_s2" else "s",
                )
            _store_mc(
                timing,
                "recovery_error_s",
                realization["recovered_offset_s"] - result["offsets"][mc],
                mc,
                result["n_mc"],
                settings,
                "s",
            )
            if "projection_residuals" in realization:
                for key, value in realization["projection_residuals"].items():
                    _store_mc(
                        projection,
                        key,
                        value,
                        mc,
                        result["n_mc"],
                        settings,
                        "nT" if "residual" in key else "1",
                    )
            for name, key, units in (
                ("measured", "measured_B", "nT"),
                ("corrected", "corrected_B", "nT"),
                ("corrected_valid", "corrected_B_valid", "1"),
            ):
                _store_mc(fields, name, realization[key], mc, result["n_mc"], settings, units)
            for column, name in enumerate(names):
                group = configurations[name]
                for key in (
                    "measured_divergence",
                    "corrected_divergence",
                    "fit_valid",
                    "comparison_valid",
                ):
                    units = "nT/km" if "divergence" in key else "1"
                    _store_mc(
                        group, key, realization[key][:, column], mc, result["n_mc"], settings, units
                    )
                for label in ("before", "after"):
                    _store_mc(
                        group,
                        f"normalized_divergence_rms_{label}",
                        realization[f"polyhedron_rms_{label}"][column],
                        mc,
                        result["n_mc"],
                        settings,
                    )
            for key, value in realization.items():
                if key in {"jacobian", "weighted_jacobian"}:
                    subgroup = diagnostics.require_group(key)
                    for diagnostic, item in value.items():
                        units = (
                            "1/s"
                            if key == "weighted_jacobian"
                            else "nT/s"
                            if "projection" in result
                            else "nT/(km s)"
                        )
                        _store_mc(
                            subgroup,
                            diagnostic,
                            item,
                            mc,
                            result["n_mc"],
                            settings,
                            units if diagnostic == "singular_values" else "1",
                        )
                elif np.isscalar(value) or key == "augmented_singular_values":
                    _store_mc(
                        diagnostics,
                        key,
                        value,
                        mc,
                        result["n_mc"],
                        settings,
                        "1/s" if key == "augmented_singular_values" else "1",
                    )
            if realization["chi2_increased"]:
                warnings.warn(
                    f"MC {mc}: resampled corrected chi-square exceeds baseline", stacklevel=2
                )
            if (mc + 1) % max(1, result["n_mc"] // 10) == 0:
                print(f"Timing-calibration MC progress: {mc + 1}/{result['n_mc']}", flush=True)
        summary = handle.create_group("mc_summary")
        recovered = timing["recovered_offset_s"][...]
        error = recovered - result["offsets"]
        for name, values in (
            ("mean_recovered_offset_s", recovered.mean(axis=0)),
            ("bias_s", error.mean(axis=0)),
            ("rmse_s", np.sqrt(np.mean(error**2, axis=0))),
            ("std_recovered_offset_s", recovered.std(axis=0)),
        ):
            _store(summary, name, values, settings, "s")
        rms_key = (
            "normalized_residual_rms" if "projection" in result else "normalized_divergence_rms"
        )
        before = diagnostics[f"{rms_key}_before"][...]
        after = diagnostics[f"{rms_key}_after"][...]
        reduction = np.full_like(before, np.nan)
        np.divide(before - after, before, out=reduction, where=before > 0)
        _store(summary, "fractional_rms_reduction", reduction, settings)
        _store(summary, "percentiles", [25.0, 50.0, 75.0], settings, "%")
        finite = reduction[np.isfinite(reduction)]
        _store(
            summary,
            "fractional_rms_reduction_percentiles",
            np.percentile(finite, [25, 50, 75]) if finite.size else np.full(3, np.nan),
            settings,
        )
        for key in ("chi2_before", "chi2_after"):
            summary[key] = diagnostics[key]  # Full empirical distributions, without duplication.
        metadata.attrs["complete"] = True
    return output


@register_analysis
class TimingCalibrateAnalysis(AnalysisMethod):
    name = "timing-calibrate"
    config_section = "timing_calibrate"

    def run(self, input_path: str | Path) -> dict[str, Any]:
        return compute_timing_calibration(input_path, self.config)

    def write_result(self, result: dict[str, Any], output_path: str | Path) -> Path:
        return write_timing_calibration_product(result, output_path)


def _plot_time_mask(
    time: np.ndarray, start_time_s: float | None, stop_time_s: float | None
) -> np.ndarray:
    """Select an inclusive plotting interval on the saved analysis time axis."""

    for name, value in (("start_time_s", start_time_s), ("stop_time_s", stop_time_s)):
        if value is not None and not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if start_time_s is not None and stop_time_s is not None and stop_time_s <= start_time_s:
        raise ValueError("stop_time_s must be greater than start_time_s")
    mask = np.ones(len(time), dtype=bool)
    if start_time_s is not None:
        mask &= time >= start_time_s
    if stop_time_s is not None:
        mask &= time <= stop_time_s
    if not np.any(mask):
        raise ValueError("Requested time range contains no analysis samples")
    return mask


def plot_timing_calibration_product(
    product_path: str | Path,
    output_path: str | Path,
    *,
    mc_index: int = 0,
    dpi: int = 300,
    start_time_s: float | None = None,
    stop_time_s: float | None = None,
) -> Path:
    """Plot a compact diagnostic figure using only the saved analysis product."""

    with h5py.File(product_path, "r") as handle:
        is_projection = "projection" in handle
    if is_projection:
        from hs_observatory.analysis.timing_affine import plot_affine_product

        return plot_affine_product(
            product_path,
            output_path,
            mc_index=mc_index,
            dpi=dpi,
            start_time_s=start_time_s,
            stop_time_s=stop_time_s,
        )

    import matplotlib.pyplot as plt

    from hs_observatory.visualization import SPACECRAFT_COLORS

    if dpi < 1:
        raise ValueError("dpi must be positive")
    with h5py.File(product_path, "r") as handle:
        if not handle["metadata"].attrs.get("complete", False):
            raise ValueError("Timing-calibration product is incomplete")
        injected = handle["timing/injected_offset_s"][...] * 1000.0
        recovered = handle["timing/recovered_offset_s"][...] * 1000.0
        if not 0 <= mc_index < len(injected):
            raise ValueError("mc_index is outside the saved ensemble")
        ids = handle["spacecraft/node_id"].asstr()[...]
        time = handle["time/analysis_time"][...]
        time_mask = _plot_time_mask(time, start_time_s, stop_time_s)
        time = time[time_mask]
        before = handle["diagnostics/normalized_divergence_rms_before"][...]
        after = handle["diagnostics/normalized_divergence_rms_after"][...]
        singular = handle["diagnostics/weighted_jacobian/singular_values"][mc_index]
        configurations = handle["configurations"]
        names = list(configurations.attrs["order"])
        selected = [names[i] for i in np.unique(np.linspace(0, len(names) - 1, 3).astype(int))]
        examples = []
        for name in selected:
            group = configurations[name]
            examples.append(
                (
                    ", ".join(group["spacecraft_ids"].asstr()[...]),
                    float(np.median(np.sqrt(group["sigma_d_squared"][()]))),
                    [
                        group["ideal_divergence"][...][time_mask],
                        group["measured_divergence"][mc_index][time_mask],
                        group["corrected_divergence"][mc_index][time_mask],
                    ],
                )
            )
    fig, axes = plt.subplots(4, 2, figsize=(12, 13), constrained_layout=True)
    ax = axes[0, 0]
    for column, name in enumerate(ids):
        ax.scatter(
            injected[:, column],
            recovered[:, column],
            s=12,
            alpha=0.65,
            color=SPACECRAFT_COLORS.get(name),
            label=name,
        )
    bounds = [min(injected.min(), recovered.min()), max(injected.max(), recovered.max())]
    ax.plot(bounds, bounds, "k--", linewidth=1)
    ax.set(xlabel="Injected offset [ms]", ylabel="Recovered offset [ms]")
    ax.legend(ncol=4, fontsize=8)
    axes[0, 1].boxplot(recovered - injected, showfliers=True)
    axes[0, 1].set_xticks(np.arange(1, 9), ids)
    axes[0, 1].axhline(0, color="k", linestyle="--", linewidth=1)
    axes[0, 1].set(ylabel="Recovered − injected [ms]", xlabel="Node")
    axes[1, 0].plot(np.arange(len(before)), before, "o-", markersize=3, label="Before")
    axes[1, 0].plot(np.arange(len(after)), after, "o-", markersize=3, label="After")
    axes[1, 0].set(xlabel="MC realization", ylabel="Normalized divergence RMS")
    axes[1, 0].legend()
    axes[1, 1].plot(np.arange(1, 9), singular, "o-")
    if np.any(singular > 0):
        axes[1, 1].set_yscale("log")
    axes[1, 1].set(
        xlabel="Singular-value index",
        ylabel="Weighted Jacobian singular value [1/s]",
        title=f"MC {mc_index}",
    )
    for ax, (members, sigma, curves) in zip(axes[2:].flat, examples):
        for label, curve in zip(("Ideal", "Measured", "Corrected"), curves):
            ax.plot(time, curve, linewidth=0.9, label=label)
        ax.set(
            xlabel="Time [s]",
            ylabel="Divergence [nT/km]",
            title=f"{members}\nσd = {sigma:.3g} nT/km; MC {mc_index}",
        )
        ax.legend(fontsize=8)
        if len(time) > 1:
            ax.set_xlim(time[0], time[-1])
    for ax in list(axes[2:].flat)[len(examples) :]:
        ax.set_visible(False)
    for ax in axes.flat:
        if ax.get_visible():
            ax.grid(alpha=0.2)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output, dpi=dpi)
    finally:
        plt.close(fig)
    return output


def plot_timing_calibration_timeseries(
    product_path: str | Path,
    output_path: str | Path,
    *,
    mc_index: int = 0,
    component: str = "x",
    dpi: int = 300,
    start_time_s: float | None = None,
    stop_time_s: float | None = None,
) -> Path:
    """Compare ideal, measured, and shifted B in one panel per Node."""

    import matplotlib.pyplot as plt

    from hs_observatory.visualization import SPACECRAFT_COLORS

    if component not in {"x", "y", "z", "magnitude", "all"}:
        raise ValueError("component must be x, y, z, magnitude, or all")
    if dpi < 1:
        raise ValueError("dpi must be positive")
    with h5py.File(product_path, "r") as handle:
        if not handle["metadata"].attrs.get("complete", False):
            raise ValueError("Timing-calibration product is incomplete")
        fields = handle["magnetic_field"]
        if not 0 <= mc_index < fields["measured"].shape[0]:
            raise ValueError("mc_index is outside the saved ensemble")
        time = handle["time/analysis_time"][...]
        time_mask = _plot_time_mask(time, start_time_s, stop_time_s)
        time = time[time_mask]
        ids = handle["spacecraft/id"].asstr()[...].tolist()
        nodes = [ids.index(f"N{i}") for i in range(1, 9)]
        curves = {
            "Ideal": fields["ideal"][...][time_mask][:, nodes],
            "Measured": fields["measured"][mc_index][time_mask][:, nodes],
            "Shifted": fields["corrected"][mc_index][time_mask][:, nodes],
        }

    labels = {
        "x": r"$B_x$",
        "y": r"$B_y$",
        "z": r"$B_z$",
        "magnitude": r"$|B|$",
        "all": r"$B_x, B_y, B_z$",
    }
    styles = {"Ideal": "-", "Measured": ":", "Shifted": "--"}
    colors = {"Ideal": "black", "Measured": "tab:orange", "Shifted": "tab:blue"}
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True, constrained_layout=True)
    fig.suptitle(f"Node magnetic-field time series — MC {mc_index}")
    for node, ax in enumerate(axes.flat):
        name = f"N{node + 1}"
        for label, values in curves.items():
            if component == "all":
                component_colors = ("tab:blue", "tab:orange", "tab:green")
                for index, (axis, color) in enumerate(zip("xyz", component_colors)):
                    ax.plot(
                        time,
                        values[:, node, index],
                        color=color,
                        linestyle=styles[label],
                        linewidth=1.1,
                        label=f"{label} B{axis}",
                    )
            else:
                series = (
                    np.linalg.norm(values[:, node], axis=-1)
                    if component == "magnitude"
                    else values[:, node, "xyz".index(component)]
                )
                ax.plot(
                    time,
                    series,
                    color=colors[label],
                    linestyle=styles[label],
                    linewidth=1.1,
                    label=label,
                )
        ax.set_title(name, color=SPACECRAFT_COLORS[name])
        ax.set_ylabel(f"{labels[component]} [nT]")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8, ncol=3 if component == "all" else 1)
    if len(time) > 1:
        axes[0, 0].set_xlim(time[0], time[-1])
    for ax in axes[-1]:
        ax.set_xlabel("Time [s]")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output, dpi=dpi)
    finally:
        plt.close(fig)
    return output
