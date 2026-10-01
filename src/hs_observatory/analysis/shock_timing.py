"""Planar-shock arrival detection and multi-spacecraft timing reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
from typing import Any, Mapping
import warnings

import h5py
import numpy as np

from hs_observatory.analysis.base import AnalysisMethod
from hs_observatory.analysis.registry import register_analysis

OBSERVABLES = ("n", "Vx", "Vy", "Vz", "Bx", "By", "Bz")
OBSERVABLE_SOURCE = {
    "n": ("n", None),
    "Vx": ("v", 0), "Vy": ("v", 1), "Vz": ("v", 2),
    "Bx": ("B", 0), "By": ("B", 1), "Bz": ("B", 2),
}
DETECTION_STATUS = {
    "success": 0,
    "no_detectable_shock": 1,
    "insufficient_pre_samples": 2,
    "insufficient_post_samples": 3,
    "ambiguous_transition": 4,
    "profile_fit_failed": 5,
    "insignificant_jump": 6,
    "poor_fit_quality": 7,
    "invalid_timing_uncertainty": 8,
}
INVERSION_STATUS = {
    "success": 0,
    "insufficient_spacecraft": 1,
    "degenerate_geometry": 2,
    "poor_conditioning": 3,
    "invalid_weights": 4,
    "timing_inversion_failure": 5,
}
TANH_COEFFICIENT = 2.197224577


@dataclass
class TransitionDetection:
    status: int
    crossing_time: float = np.nan
    uncertainty: float = np.nan
    width: float = np.nan
    upstream: float = np.nan
    downstream: float = np.nan
    noise: float = np.nan
    significance: float = np.nan
    normalized_rms: float = np.nan
    candidate_time: float = np.nan

    @property
    def success(self) -> bool:
        return self.status == DETECTION_STATUS["success"]


@dataclass
class TimingSolution:
    status: int
    q: np.ndarray
    normal: np.ndarray
    speed: float
    reference_time: float
    residuals: np.ndarray
    rms_residual: float
    rank: int
    singular_values: np.ndarray
    condition_number: float
    reduced_chi_square: float
    geometry_condition_number: float
    characteristic_size: float
    elongation: float
    planarity: float

    @property
    def success(self) -> bool:
        return self.status == INVERSION_STATUS["success"]


def _mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    center = np.median(values)
    return float(1.4826 * np.median(np.abs(values - center)))


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    detection = dict(config.get("detection", {}))
    fitting = dict(config.get("fitting", {}))
    timing = dict(config.get("timing", {}))
    storage = dict(config.get("storage", {}))
    plot = dict(config.get("plot", {}))
    modes = str(config.get("modes", "both")).lower()
    if modes not in {"independent", "combined", "both"}:
        raise ValueError("shock_timing.modes must be independent, combined, or both")
    configured_observables = config.get("observables", "auto")
    if isinstance(configured_observables, str):
        if configured_observables.lower() != "auto":
            raise ValueError("shock_timing.observables must be auto or a list")
        observables = list(OBSERVABLES)
    elif isinstance(configured_observables, list):
        observables = [str(value) for value in configured_observables]
        if not observables or len(set(observables)) != len(observables):
            raise ValueError("shock_timing.observables must be a nonempty unique list")
        unknown = set(observables) - set(OBSERVABLES)
        if unknown:
            raise ValueError(f"Unknown shock-timing observables: {sorted(unknown)}")
    else:
        raise ValueError("shock_timing.observables must be auto or a list")
    truth_mode = str(config.get("truth_validation", "auto")).lower()
    if truth_mode not in {"auto", "required", "disabled"}:
        raise ValueError("truth_validation must be auto, required, or disabled")
    result = {
        "modes": modes,
        "observables": observables,
        "truth_validation": truth_mode,
        "detection": {
            "min_jump_significance": float(detection.get("min_jump_significance", 5.0)),
            "candidate_window_s": float(detection.get("candidate_window_s", 1.0)),
            "transition_guard_s": float(detection.get("transition_guard_s", 0.25)),
            "quiet_window_s": float(detection.get("quiet_window_s", 2.0)),
            "min_quiet_samples": int(detection.get("min_quiet_samples", 8)),
            "min_side_samples": int(detection.get("min_side_samples", 16)),
        },
        "fitting": {
            "fit_half_window_s": float(fitting.get("fit_half_window_s", 3.0)),
            "min_width_cadence_fraction": float(
                fitting.get("min_width_cadence_fraction", 0.25)
            ),
            "max_width_s": float(fitting.get("max_width_s", 2.0)),
            "max_normalized_rms": float(fitting.get("max_normalized_rms", 3.0)),
            "max_nfev": int(fitting.get("max_nfev", 2000)),
        },
        "timing": {
            "max_condition_number": float(timing.get("max_condition_number", 1.0e8)),
            "rcond": timing.get("rcond"),
            "uncertainty_floor_s": float(timing.get("uncertainty_floor_s", 0.001)),
            "uncertainty_floor_cadence_fraction": float(
                timing.get("uncertainty_floor_cadence_fraction", 0.25)
            ),
        },
        "storage": {
            "compression": str(storage.get("compression", "gzip")),
            "compression_level": int(storage.get("compression_level", 4)),
        },
        "plot": {
            "enabled": bool(plot.get("enabled", True)),
            "detection_base": plot.get("detection_base"),
            "accuracy_base": plot.get("accuracy_base"),
            "geometry_base": plot.get("geometry_base"),
            "tetrahedra_geometry_base": plot.get("tetrahedra_geometry_base"),
            "representative_mc_index": int(plot.get("representative_mc_index", 0)),
            "dpi": int(plot.get("dpi", 150)),
        },
    }
    for section in ("detection", "fitting", "timing"):
        for key, value in result[section].items():
            if key != "rcond" and isinstance(value, (float, int)) and value <= 0:
                raise ValueError(f"shock_timing.{section}.{key} must be positive")
    return result


def _tanh(time: np.ndarray, upstream: float, downstream: float, center: float, width: float):
    fraction = 0.5 * (1.0 + np.tanh(TANH_COEFFICIENT * (time - center) / width))
    return upstream + (downstream - upstream) * fraction


def _candidate_clusters(time: np.ndarray, values: np.ndarray, settings: Mapping[str, Any]):
    cadence = float(np.median(np.diff(time)))
    diff_noise = _mad(np.diff(values)) / np.sqrt(2.0)
    noise_floor = max(diff_noise, np.finfo(float).eps * max(1.0, np.max(np.abs(values))))
    half = settings["candidate_window_s"]
    guard = settings["transition_guard_s"]
    minimum = max(3, int(np.ceil(0.25 * half / cadence)))
    scores = np.full(len(time), np.nan)
    for index, center in enumerate(time):
        left = values[(time >= center - guard - half) & (time <= center - guard)]
        right = values[(time >= center + guard) & (time <= center + guard + half)]
        if len(left) >= minimum and len(right) >= minimum:
            scores[index] = abs(float(np.median(right) - np.median(left))) / noise_floor
    selected = np.flatnonzero(scores >= settings["min_jump_significance"])
    if not len(selected):
        return [], scores, noise_floor
    clusters = np.split(selected, np.flatnonzero(np.diff(selected) > 1) + 1)
    return clusters, scores, noise_floor


def detect_transition(
    time: np.ndarray,
    values: np.ndarray,
    detection_settings: Mapping[str, Any],
    fitting_settings: Mapping[str, Any],
) -> TransitionDetection:
    """Detect and fit one transition without using simulation truth."""

    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    if time.ndim != 1 or values.shape != time.shape or len(time) < 4:
        return TransitionDetection(DETECTION_STATUS["no_detectable_shock"])
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(values)):
        return TransitionDetection(DETECTION_STATUS["profile_fit_failed"])
    clusters, scores, global_noise = _candidate_clusters(time, values, detection_settings)
    if not clusters:
        return TransitionDetection(
            DETECTION_STATUS["no_detectable_shock"], noise=global_noise
        )
    # A single physical transition creates one contiguous score cluster.
    if len(clusters) > 1:
        return TransitionDetection(DETECTION_STATUS["ambiguous_transition"], noise=global_noise)
    cluster = clusters[0]
    candidate_index = int(cluster[np.nanargmax(scores[cluster])])
    candidate_time = float(time[candidate_index])
    if candidate_index < detection_settings["min_side_samples"]:
        return TransitionDetection(
            DETECTION_STATUS["insufficient_pre_samples"],
            candidate_time=candidate_time, noise=global_noise,
        )
    if len(time) - candidate_index - 1 < detection_settings["min_side_samples"]:
        return TransitionDetection(
            DETECTION_STATUS["insufficient_post_samples"],
            candidate_time=candidate_time, noise=global_noise,
        )
    guard = detection_settings["transition_guard_s"]
    quiet = detection_settings["quiet_window_s"]
    left_mask = (time >= candidate_time - guard - quiet) & (time <= candidate_time - guard)
    right_mask = (time >= candidate_time + guard) & (time <= candidate_time + guard + quiet)
    if np.count_nonzero(left_mask) < detection_settings["min_quiet_samples"]:
        return TransitionDetection(DETECTION_STATUS["insufficient_pre_samples"])
    if np.count_nonzero(right_mask) < detection_settings["min_quiet_samples"]:
        return TransitionDetection(DETECTION_STATUS["insufficient_post_samples"])
    left, right = values[left_mask], values[right_mask]
    upstream0, downstream0 = float(np.median(left)), float(np.median(right))
    noise_left, noise_right = _mad(left), _mad(right)
    noise = max(
        float(np.hypot(noise_left, noise_right) / np.sqrt(2.0)),
        global_noise,
        np.finfo(float).eps,
    )
    preliminary_significance = abs(downstream0 - upstream0) / noise
    if preliminary_significance < detection_settings["min_jump_significance"]:
        return TransitionDetection(
            DETECTION_STATUS["insignificant_jump"], noise=noise,
            significance=preliminary_significance, candidate_time=candidate_time,
        )
    fit_mask = np.abs(time - candidate_time) <= fitting_settings["fit_half_window_s"]
    fit_time, fit_values = time[fit_mask], values[fit_mask]
    cadence = float(np.median(np.diff(time)))
    minimum_width = fitting_settings["min_width_cadence_fraction"] * cadence
    maximum_width = fitting_settings["max_width_s"]
    try:
        from scipy.optimize import least_squares
    except ImportError as exc:
        raise ImportError(
            "Shock timing requires SciPy; reinstall hs-observatory with its dependencies"
        ) from exc

    try:
        initial = np.array([
            upstream0, downstream0, candidate_time,
            min(max(2.0 * cadence, minimum_width), maximum_width),
        ])
        scale = max(abs(downstream0 - upstream0), noise)
        lower = [upstream0 - 2 * scale, downstream0 - 2 * scale,
                 float(fit_time[0]), minimum_width]
        upper = [upstream0 + 2 * scale, downstream0 + 2 * scale,
                 float(fit_time[-1]), maximum_width]
        fit = least_squares(
            lambda p: (_tanh(fit_time, *p) - fit_values) / noise,
            initial, bounds=(lower, upper), loss="linear", x_scale="jac",
            max_nfev=fitting_settings["max_nfev"],
        )
    except (ValueError, RuntimeError, FloatingPointError):
        return TransitionDetection(DETECTION_STATUS["profile_fit_failed"], noise=noise)
    if not fit.success or fit.jac.shape[1] != 4 or np.linalg.matrix_rank(fit.jac) < 4:
        return TransitionDetection(DETECTION_STATUS["profile_fit_failed"], noise=noise)
    upstream, downstream, crossing, width = (float(value) for value in fit.x)
    residual = (_tanh(fit_time, *fit.x) - fit_values) / noise
    normalized_rms = float(np.sqrt(np.mean(residual**2)))
    significance = abs(downstream - upstream) / noise
    if significance < detection_settings["min_jump_significance"]:
        status = DETECTION_STATUS["insignificant_jump"]
    elif normalized_rms > fitting_settings["max_normalized_rms"]:
        status = DETECTION_STATUS["poor_fit_quality"]
    else:
        dof = max(1, len(fit_time) - 4)
        covariance = np.linalg.pinv(fit.jac.T @ fit.jac)
        covariance *= max(1.0, float(np.sum(residual**2) / dof))
        uncertainty = float(np.sqrt(max(0.0, covariance[2, 2])))
        status = (
            DETECTION_STATUS["success"]
            if np.isfinite(uncertainty) and uncertainty > 0.0
            else DETECTION_STATUS["invalid_timing_uncertainty"]
        )
        return TransitionDetection(
            status, crossing, uncertainty, width, upstream, downstream,
            noise, significance, normalized_rms, candidate_time,
        )
    return TransitionDetection(
        status, crossing, np.nan, width, upstream, downstream,
        noise, significance, normalized_rms, candidate_time,
    )


def fit_combined_transition(
    channels: list[tuple[np.ndarray, np.ndarray, TransitionDetection]],
    fitting_settings: Mapping[str, Any],
) -> TransitionDetection:
    """Fit native-cadence channels with a common center and temporal width."""

    accepted = [item for item in channels if item[2].success]
    if not accepted:
        return TransitionDetection(DETECTION_STATUS["no_detectable_shock"])
    try:
        from scipy.optimize import least_squares
    except ImportError as exc:
        raise ImportError(
            "Shock timing requires SciPy; reinstall hs-observatory with its dependencies"
        ) from exc

    try:
        centers = np.array([item[2].crossing_time for item in accepted])
        widths = np.array([item[2].width for item in accepted])
        parameters = []
        for _, _, detection in accepted:
            parameters.extend([detection.upstream, detection.downstream])
        parameters.extend([float(np.median(centers)), float(np.median(widths))])
        lower, upper = [], []
        for _, values, detection in accepted:
            scale = max(abs(detection.downstream - detection.upstream), detection.noise)
            lower.extend([detection.upstream - 2 * scale, detection.downstream - 2 * scale])
            upper.extend([detection.upstream + 2 * scale, detection.downstream + 2 * scale])
        cadence_min = min(float(np.median(np.diff(item[0]))) for item in accepted)
        minimum_width = fitting_settings["min_width_cadence_fraction"] * cadence_min
        lower.extend([max(item[0][0] for item in accepted), minimum_width])
        upper.extend([min(item[0][-1] for item in accepted), fitting_settings["max_width_s"]])

        def residuals(p: np.ndarray, equal_channels: bool = True) -> np.ndarray:
            result = []
            center, width = p[-2:]
            for index, (time, values, detection) in enumerate(accepted):
                mask = np.abs(time - center) <= fitting_settings["fit_half_window_s"]
                residual = (
                    _tanh(time[mask], p[2 * index], p[2 * index + 1], center, width)
                    - values[mask]
                ) / detection.noise
                if equal_channels:
                    residual = residual / np.sqrt(max(1, len(residual)))
                result.append(residual)
            return np.concatenate(result)

        fit = least_squares(
            residuals, parameters, bounds=(lower, upper), loss="linear", x_scale="jac",
            max_nfev=fitting_settings["max_nfev"],
        )
        if not fit.success:
            raise RuntimeError
        center, width = (float(value) for value in fit.x[-2:])
        # Physical-noise curvature retains the information supplied by cadence.
        eps = np.sqrt(np.finfo(float).eps)
        base = residuals(fit.x, equal_channels=False)
        jacobian = np.empty((len(base), len(fit.x)))
        for column in range(len(fit.x)):
            step = eps * max(1.0, abs(float(fit.x[column])))
            shifted = fit.x.copy()
            shifted[column] += step
            jacobian[:, column] = (residuals(shifted, False) - base) / step
        covariance = np.linalg.pinv(jacobian.T @ jacobian)
        dof = max(1, len(base) - len(fit.x))
        covariance *= max(1.0, float(np.sum(base**2) / dof))
        uncertainty = float(np.sqrt(max(0.0, covariance[-2, -2])))
        rms = float(np.sqrt(np.mean(base**2)))
        if rms > fitting_settings["max_normalized_rms"]:
            status = DETECTION_STATUS["poor_fit_quality"]
        else:
            status = (
                DETECTION_STATUS["success"]
                if np.isfinite(uncertainty) and uncertainty > 0
                else DETECTION_STATUS["invalid_timing_uncertainty"]
            )
        return TransitionDetection(
            status, center, uncertainty, width,
            noise=float(np.median([item[2].noise for item in accepted])),
            significance=float(min(item[2].significance for item in accepted)),
            normalized_rms=rms,
        )
    except (ValueError, RuntimeError, FloatingPointError):
        return TransitionDetection(DETECTION_STATUS["profile_fit_failed"])


def solve_planar_timing(
    crossing_times: np.ndarray,
    positions: np.ndarray,
    reference_position: np.ndarray,
    time_origin: float,
    *,
    uncertainties: np.ndarray | None = None,
    max_condition_number: float = 1.0e8,
    rcond: float | None = None,
) -> TimingSolution:
    """Solve the intercept/slowness system by scaled SVD least squares."""

    times = np.asarray(crossing_times, dtype=float)
    points = np.asarray(positions, dtype=float)
    empty = TimingSolution(
        INVERSION_STATUS["timing_inversion_failure"], np.full(3, np.nan),
        np.full(3, np.nan), np.nan, np.nan, np.full(len(times), np.nan),
        np.nan, 0, np.full(4, np.nan), np.inf, np.nan,
        np.inf, np.nan, np.nan, np.nan,
    )
    if len(times) < 4:
        empty.status = INVERSION_STATUS["insufficient_spacecraft"]
        return empty
    centered = points - np.asarray(reference_position, dtype=float)
    geometry_centered = points - np.mean(points, axis=0)
    geometry_tensor = geometry_centered.T @ geometry_centered / len(points)
    eigenvalues = np.clip(np.linalg.eigvalsh(geometry_tensor)[::-1], 0.0, None)
    axes = np.sqrt(eigenvalues)
    geometry_rank = np.linalg.matrix_rank(geometry_tensor)
    empty.geometry_condition_number = (
        float(np.linalg.cond(geometry_tensor)) if geometry_rank == 3 else np.inf
    )
    empty.characteristic_size = float(2.0 * axes[0])
    empty.elongation = 0.0 if axes[0] == 0 else float(1.0 - axes[1] / axes[0])
    empty.planarity = 0.0 if axes[1] == 0 else float(1.0 - axes[2] / axes[1])
    length_scale = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    if not np.isfinite(length_scale) or length_scale <= 0.0:
        empty.status = INVERSION_STATUS["degenerate_geometry"]
        return empty
    design = np.column_stack((np.ones(len(times)), centered / length_scale))
    if uncertainties is not None:
        sigma = np.asarray(uncertainties, dtype=float)
        if sigma.shape != times.shape or np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
            empty.status = INVERSION_STATUS["invalid_weights"]
            return empty
        weight = 1.0 / sigma
    else:
        sigma = None
        weight = np.ones(len(times))
    weighted_design = design * weight[:, None]
    weighted_target = (times - time_origin) * weight
    singular = np.linalg.svd(weighted_design, compute_uv=False)
    tolerance = np.finfo(float).eps * max(weighted_design.shape) * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    condition = float(singular[0] / singular[-1]) if rank == 4 else np.inf
    empty.rank, empty.singular_values, empty.condition_number = rank, singular, condition
    if rank < 4:
        empty.status = INVERSION_STATUS["degenerate_geometry"]
        return empty
    if condition > max_condition_number:
        empty.status = INVERSION_STATUS["poor_conditioning"]
        return empty
    try:
        beta = np.linalg.lstsq(weighted_design, weighted_target, rcond=rcond)[0]
    except np.linalg.LinAlgError:
        return empty
    q = beta[1:] / length_scale
    q_norm = float(np.linalg.norm(q))
    if not np.isfinite(q_norm) or q_norm <= 0.0:
        return empty
    reference_time = float(time_origin + beta[0])
    residuals = times - (reference_time + centered @ q)
    rms = float(np.sqrt(np.mean(residuals**2)))
    chi2 = np.nan
    if sigma is not None and len(times) > 4:
        chi2 = float(np.sum((residuals / sigma) ** 2) / (len(times) - 4))
    return TimingSolution(
        INVERSION_STATUS["success"], q, q / q_norm, 1.0 / q_norm,
        reference_time, residuals, rms, rank, singular, condition, chi2,
        empty.geometry_condition_number, empty.characteristic_size,
        empty.elongation, empty.planarity,
    )


def _read_input(path: str | Path, observables: list[str]) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        ids = [value.decode() if isinstance(value, bytes) else str(value)
               for value in handle["spacecraft/id"][...]]
        metadata = {key: (value.decode() if isinstance(value, bytes) else value)
                    for key, value in handle["metadata"].attrs.items()}
        data = {}
        n_mc = None
        for name in observables:
            quantity, component = OBSERVABLE_SOURCE[name]
            time = np.asarray(handle[f"time/{quantity}"], dtype=float)
            values = np.asarray(handle[f"observed/{quantity}"], dtype=float)
            if component is not None:
                values = values[..., component]
            n_mc = values.shape[0] if n_mc is None else n_mc
            if values.shape != (n_mc, len(time), len(ids)):
                raise ValueError(f"Observed {name} has an unexpected shape")
            data[name] = {"time": time, "values": values,
                          "cadence": float(np.median(np.diff(time)))}
        trajectory_candidates = []
        for quantity in ("n", "v", "B"):
            time = np.asarray(handle[f"time/{quantity}"], dtype=float)
            positions = np.asarray(handle[f"spacecraft/r/{quantity}"], dtype=float)
            trajectory_candidates.append((float(np.median(np.diff(time))), time, positions))
        _, trajectory_time, trajectory = min(trajectory_candidates, key=lambda item: item[0])
    return {"ids": ids, "metadata": metadata, "data": data, "n_mc": int(n_mc or 0),
            "trajectory_time": trajectory_time, "trajectory": trajectory}


def _positions_at_times(trajectory_time: np.ndarray, trajectory: np.ndarray, times: np.ndarray):
    positions = np.empty((len(times), 3))
    for spacecraft, crossing in enumerate(times):
        for component in range(3):
            positions[spacecraft, component] = np.interp(
                crossing, trajectory_time, trajectory[spacecraft, :, component]
            )
    return positions


def _truth(metadata: Mapping[str, Any], reference_position: np.ndarray,
           trajectory_time: np.ndarray, trajectory: np.ndarray) -> dict[str, Any] | None:
    if str(metadata.get("field_model", "")) != "shock":
        return None
    try:
        field = json.loads(str(metadata["field_metadata"]))
        normal_ud = np.asarray(field["normal_gse"], dtype=float)
        velocity = np.asarray(field["velocity_gse_km_s"], dtype=float)
        center = np.asarray(field["mesocenter_gse_km"], dtype=float)
        t0 = float(field["reference_time_s"])
        distance = float(field["distance_from_mesocenter_km"])
        signed_speed = float(np.dot(normal_ud, velocity))
        if signed_speed == 0.0:
            return None
        q = normal_ud / signed_speed
        propagation_normal = np.sign(signed_speed) * normal_ud
        reference_time = t0 + (
            np.dot(normal_ud, reference_position - center) - distance
        ) / signed_speed
        arrivals = np.full(trajectory.shape[0], np.nan)
        for spacecraft in range(trajectory.shape[0]):
            xi = (
                (trajectory[spacecraft] - center) @ normal_ud
                - distance - signed_speed * (trajectory_time - t0)
            )
            exact = np.flatnonzero(xi == 0.0)
            if len(exact):
                arrivals[spacecraft] = trajectory_time[exact[0]]
                continue
            crossing = np.flatnonzero(xi[:-1] * xi[1:] < 0.0)
            if len(crossing):
                index = int(crossing[0])
                arrivals[spacecraft] = trajectory_time[index] - xi[index] * (
                    trajectory_time[index + 1] - trajectory_time[index]
                ) / (xi[index + 1] - xi[index])
        return {"normal_upstream_to_downstream": normal_ud,
                "propagation_normal": propagation_normal,
                "signed_normal_velocity": signed_speed,
                "speed": abs(signed_speed), "q": q,
                "reference_time": float(reference_time), "arrival_times": arrivals}
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _detection_arrays(shape: tuple[int, ...]) -> dict[str, np.ndarray]:
    return {
        "status": np.full(shape, DETECTION_STATUS["no_detectable_shock"], dtype=np.int16),
        **{name: np.full(shape, np.nan) for name in (
            "crossing_time", "uncertainty", "width", "upstream", "downstream",
            "noise", "significance", "normalized_rms", "candidate_time")},
    }


def _put_detection(
    arrays: dict[str, np.ndarray], index: tuple[int, ...], item: TransitionDetection
):
    arrays["status"][index] = item.status
    for name in arrays:
        if name != "status":
            arrays[name][index] = getattr(item, name)


def _solution_arrays(shape: tuple[int, ...], n_spacecraft: int) -> dict[str, np.ndarray]:
    return {
        "status": np.full(shape, INVERSION_STATUS["timing_inversion_failure"], dtype=np.int16),
        "q": np.full((*shape, 3), np.nan), "normal": np.full((*shape, 3), np.nan),
        "speed": np.full(shape, np.nan), "reference_time": np.full(shape, np.nan),
        "rms_residual": np.full(shape, np.nan), "rank": np.zeros(shape, dtype=np.int16),
        "condition_number": np.full(shape, np.nan),
        "reduced_chi_square": np.full(shape, np.nan),
        "geometry_condition_number": np.full(shape, np.nan),
        "characteristic_size": np.full(shape, np.nan),
        "elongation": np.full(shape, np.nan), "planarity": np.full(shape, np.nan),
        "n_spacecraft": np.zeros(shape, dtype=np.int16),
        "singular_values": np.full((*shape, 4), np.nan),
        "residuals": np.full((*shape, n_spacecraft), np.nan),
        "participant_mask": np.zeros((*shape, n_spacecraft), dtype=bool),
    }


def _put_solution(arrays: dict[str, np.ndarray], index: tuple[int, ...], solution: TimingSolution,
                  members: np.ndarray):
    for name in ("status", "q", "normal", "speed", "reference_time", "rms_residual",
                 "rank", "condition_number", "reduced_chi_square", "singular_values"):
        arrays[name][index] = getattr(solution, name)
    for name in ("geometry_condition_number", "characteristic_size", "elongation", "planarity"):
        arrays[name][index] = getattr(solution, name)
    arrays["n_spacecraft"][index] = len(members)
    arrays["participant_mask"][index][members] = True
    if len(solution.residuals) == len(members):
        arrays["residuals"][index][members] = solution.residuals


def _add_errors(solution: dict[str, np.ndarray], truth: dict[str, Any] | None):
    shape = solution["status"].shape
    errors = {
        "angular_error_deg": np.full(shape, np.nan), "speed_error": np.full(shape, np.nan),
        "fractional_speed_error": np.full(shape, np.nan),
        "q_error": np.full((*shape, 3), np.nan), "q_error_norm": np.full(shape, np.nan),
        "reference_time_error": np.full(shape, np.nan),
    }
    if truth is not None:
        valid = solution["status"] == INVERSION_STATUS["success"]
        dots = np.sum(solution["normal"] * truth["propagation_normal"], axis=-1)
        errors["angular_error_deg"][valid] = np.degrees(
            np.arccos(np.clip(dots[valid], -1, 1))
        )
        errors["speed_error"][valid] = solution["speed"][valid] - truth["speed"]
        errors["fractional_speed_error"][valid] = (
            np.abs(errors["speed_error"][valid]) / truth["speed"]
        )
        errors["q_error"][valid] = solution["q"][valid] - truth["q"]
        errors["q_error_norm"][valid] = np.linalg.norm(errors["q_error"][valid], axis=-1)
        errors["reference_time_error"][valid] = (
            solution["reference_time"][valid] - truth["reference_time"]
        )
    solution["errors"] = errors


def compute_shock_timing(input_path: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    settings = _settings(config)
    source = _read_input(input_path, settings["observables"])
    ids, observables = source["ids"], settings["observables"]
    n_mc, n_sc, n_obs = source["n_mc"], len(ids), len(observables)
    time_origin = 0.5 * (source["trajectory_time"][0] + source["trajectory_time"][-1])
    reference_positions = _positions_at_times(
        source["trajectory_time"], source["trajectory"], np.full(n_sc, time_origin)
    )
    reference_position = np.mean(reference_positions, axis=0)
    truth = None if settings["truth_validation"] == "disabled" else _truth(
        source["metadata"], reference_position, source["trajectory_time"], source["trajectory"]
    )
    if settings["truth_validation"] == "required" and truth is None:
        raise ValueError("Shock truth validation was required but compatible metadata are absent")

    independent = _detection_arrays((n_obs, n_mc, n_sc))
    detections: list[list[list[TransitionDetection]]] = [
        [[TransitionDetection(1) for _ in range(n_sc)] for _ in range(n_mc)]
        for _ in range(n_obs)
    ]
    for observable_index, name in enumerate(observables):
        item = source["data"][name]
        for mc in range(n_mc):
            for spacecraft in range(n_sc):
                detection = detect_transition(
                    item["time"], item["values"][mc, :, spacecraft],
                    settings["detection"], settings["fitting"],
                )
                detections[observable_index][mc][spacecraft] = detection
                _put_detection(independent, (observable_index, mc, spacecraft), detection)

    combined = _detection_arrays((n_mc, n_sc))
    contributors = np.zeros((n_mc, n_sc, n_obs), dtype=bool)
    if settings["modes"] in {"combined", "both"}:
        for mc in range(n_mc):
            for spacecraft in range(n_sc):
                channels = []
                for observable_index, name in enumerate(observables):
                    detection = detections[observable_index][mc][spacecraft]
                    contributors[mc, spacecraft, observable_index] = detection.success
                    item = source["data"][name]
                    channels.append((item["time"], item["values"][mc, :, spacecraft], detection))
                _put_detection(
                    combined, (mc, spacecraft),
                    fit_combined_transition(channels, settings["fitting"]),
                )
    if truth is not None:
        for observable_index in range(n_obs):
            independent.setdefault("arrival_time_error", np.full((n_obs, n_mc, n_sc), np.nan))
            valid = independent["status"][observable_index] == DETECTION_STATUS["success"]
            independent["arrival_time_error"][observable_index][valid] = (
                independent["crossing_time"][observable_index][valid]
                - np.broadcast_to(truth["arrival_times"], (n_mc, n_sc))[valid]
            )
        combined["arrival_time_error"] = np.full((n_mc, n_sc), np.nan)
        valid = combined["status"] == DETECTION_STATUS["success"]
        combined["arrival_time_error"][valid] = (
            combined["crossing_time"][valid]
            - np.broadcast_to(truth["arrival_times"], (n_mc, n_sc))[valid]
        )

    combinations4 = np.asarray(list(combinations(range(n_sc), 4)), dtype=int)
    groups: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    if settings["modes"] in {"independent", "both"}:
        for observable_index, name in enumerate(observables):
            groups[name] = (
                independent["crossing_time"][observable_index],
                independent["uncertainty"][observable_index],
                independent["status"][observable_index],
            )
    if settings["modes"] in {"combined", "both"}:
        groups["combined"] = (
            combined["crossing_time"], combined["uncertainty"], combined["status"]
        )
    solutions = {}
    for label, (crossings, uncertainties, statuses) in groups.items():
        global_ols = _solution_arrays((n_mc,), n_sc)
        global_wls = _solution_arrays((n_mc,), n_sc)
        four = _solution_arrays((n_mc, len(combinations4)), n_sc)
        cadence = (
            min(source["data"][name]["cadence"] for name in observables)
            if label == "combined"
            else source["data"][label]["cadence"]
        )
        floor = max(
            settings["timing"]["uncertainty_floor_s"],
            settings["timing"]["uncertainty_floor_cadence_fraction"] * cadence,
        )
        for mc in range(n_mc):
            members = np.flatnonzero(statuses[mc] == DETECTION_STATUS["success"])
            if len(members) >= 4:
                positions = _positions_at_times(
                    source["trajectory_time"], source["trajectory"], crossings[mc]
                )
                common = dict(
                    reference_position=reference_position, time_origin=time_origin,
                    max_condition_number=settings["timing"]["max_condition_number"],
                    rcond=settings["timing"]["rcond"],
                )
                _put_solution(global_ols, (mc,), solve_planar_timing(
                    crossings[mc, members], positions[members], **common
                ), members)
                sigma = np.maximum(uncertainties[mc, members], floor)
                _put_solution(global_wls, (mc,), solve_planar_timing(
                    crossings[mc, members], positions[members], uncertainties=sigma, **common
                ), members)
                member_set = set(members.tolist())
                for combination_index, subset in enumerate(combinations4):
                    if set(subset.tolist()) <= member_set:
                        _put_solution(four, (mc, combination_index), solve_planar_timing(
                            crossings[mc, subset], positions[subset], **common
                        ), subset)
            else:
                global_ols["status"][mc] = INVERSION_STATUS["insufficient_spacecraft"]
                global_wls["status"][mc] = INVERSION_STATUS["insufficient_spacecraft"]
                four["status"][mc] = INVERSION_STATUS["insufficient_spacecraft"]
        for solution in (global_ols, global_wls, four):
            _add_errors(solution, truth)
        solutions[label] = {"ols": global_ols, "wls": global_wls, "four_spacecraft": four}
    return {
        "source": Path(input_path).resolve(), "settings": settings, "metadata": source["metadata"],
        "analysis_config_yaml": config.get("_analysis_config_yaml"),
        "analysis_config_path": config.get("_analysis_config_path"),
        "ids": ids, "observables": observables, "data": source["data"],
        "n_mc": n_mc, "reference_position": reference_position,
        "time_origin": float(time_origin), "trajectory_time": source["trajectory_time"],
        "trajectory": source["trajectory"], "truth": truth,
        "independent": independent, "combined": combined, "contributors": contributors,
        "combinations": combinations4, "solutions": solutions,
    }


def _store(group: h5py.Group, name: str, value: Any, settings: Mapping[str, Any], units=None):
    array = np.asarray(value)
    kwargs = {}
    if array.ndim and array.size:
        kwargs = {"compression": settings["storage"]["compression"],
                  "compression_opts": settings["storage"]["compression_level"]}
    dataset = group.create_dataset(name, data=array, **kwargs)
    if units:
        dataset.attrs["units"] = units
    return dataset


def _write_detection(group: h5py.Group, values: Mapping[str, np.ndarray], settings):
    units = {"crossing_time": "s", "uncertainty": "s", "width": "s",
             "arrival_time_error": "s"}
    for name, value in values.items():
        _store(group, name, value, settings, units.get(name))
    group["status"].attrs["codes"] = json.dumps(DETECTION_STATUS, sort_keys=True)


def _percentiles(values: np.ndarray, success: np.ndarray) -> np.ndarray:
    masked = np.where(success[(...,) + (None,) * (values.ndim - success.ndim)], values, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(masked, [16, 50, 84], axis=0)


def _write_solution(group: h5py.Group, solution: Mapping[str, Any], settings):
    units = {"q": "s/km", "speed": "km/s", "reference_time": "s",
             "rms_residual": "s", "residuals": "s"}
    errors = solution["errors"]
    for name, value in solution.items():
        if name != "errors":
            _store(group, name, value, settings, units.get(name))
    labels = np.full(solution["n_spacecraft"].shape, "unavailable", dtype=object)
    for index in np.ndindex(labels.shape):
        count = int(solution["n_spacecraft"][index])
        if count:
            labels[index] = "nine-spacecraft" if count == 9 else f"{count}-spacecraft"
    group.create_dataset("fit_label", data=np.asarray(labels, dtype=h5py.string_dtype()))
    group["status"].attrs["codes"] = json.dumps(INVERSION_STATUS, sort_keys=True)
    validation = group.create_group("errors")
    error_units = {"angular_error_deg": "deg", "speed_error": "km/s",
                   "fractional_speed_error": "1", "q_error": "s/km",
                   "q_error_norm": "s/km", "reference_time_error": "s"}
    for name, value in errors.items():
        _store(validation, name, value, settings, error_units[name])
    success = solution["status"] == INVERSION_STATUS["success"]
    summary = group.create_group("mc_summary")
    counts = np.sum(success, axis=0)
    failures = success.shape[0] - counts
    if np.ndim(counts) == 0:
        summary.attrs["success_count"] = int(counts)
        summary.attrs["failure_count"] = int(failures)
        summary.attrs["success_fraction"] = float(np.mean(success))
    else:
        _store(summary, "success_count", counts, settings)
        _store(summary, "failure_count", failures, settings)
        _store(summary, "success_fraction", np.mean(success, axis=0), settings)
    _store(summary, "percentiles", [16.0, 50.0, 84.0], settings, "%")
    status_counts = np.stack([
        np.sum(solution["status"] == code, axis=0)
        for code in sorted(INVERSION_STATUS.values())
    ])
    _store(summary, "status_count", status_counts, settings)
    summary["status_count"].attrs["codes"] = json.dumps(INVERSION_STATUS, sort_keys=True)
    for name in ("q", "normal", "speed", "reference_time", "rms_residual"):
        _store(
            summary, f"{name}_percentiles",
            _percentiles(solution[name], success), settings, units.get(name),
        )
    for name, value in errors.items():
        _store(
            summary, f"{name}_percentiles", _percentiles(value, success),
            settings, error_units[name],
        )


def _plot_paths(output: Path, plot: Mapping[str, Any], key: str):
    configured = plot[f"{key}_base"]
    base = Path(configured) if configured else output.with_name(f"{output.stem}_{key}")
    return base.with_suffix(".pdf"), base.with_suffix(".png")


def _best_worst_tetrahedra(result: Mapping[str, Any]):
    """Return combined-fit tetrahedra selected by empirical MC RMSE."""

    solutions = result["solutions"]
    if not solutions:
        return None
    source_label = "combined" if "combined" in solutions else next(iter(solutions))
    four = solutions[source_label]["four_spacecraft"]
    if result["truth"] is not None:
        values = four["errors"]["angular_error_deg"]
    else:
        values = four["rms_residual"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rmse = np.sqrt(np.nanmean(values**2, axis=0))
    valid = np.flatnonzero(np.isfinite(rmse))
    if not len(valid):
        return None
    best = int(valid[np.argmin(rmse[valid])])
    worst = int(valid[np.argmax(rmse[valid])])
    return source_label, best, worst, rmse


def _save_tetrahedra_geometry(result: Mapping[str, Any], output: Path, plt) -> None:
    """Compose existing mesocentric and polyhedral geometry conventions."""

    selected = _best_worst_tetrahedra(result)
    if selected is None:
        return
    from hs_observatory.visualization import polyhedron_metrics, spacecraft_color

    source_label, best_index, worst_index, rmse = selected
    combinations4 = result["combinations"]
    best, worst = combinations4[best_index], combinations4[worst_index]
    positions = _positions_at_times(
        result["trajectory_time"], result["trajectory"],
        np.full(len(result["ids"]), result["time_origin"]),
    )
    metrics = [item for item in polyhedron_metrics(positions) if item["n_spacecraft"] == 4]
    hub_index = next(
        (index for index, spacecraft_id in enumerate(result["ids"])
         if spacecraft_id in {"H", "Hub"}),
        None,
    )
    regular_hub = (
        min(
            (item for item in metrics if hub_index in item["indices"]),
            key=lambda item: item["Q"],
        )
        if hub_index is not None else None
    )
    mesocenter = np.mean(positions, axis=0)
    relative = positions - mesocenter
    extent = float(np.max(np.abs(relative)))
    limit = 1.12 * extent if extent > 0 else 1.0
    colors = {"Best RMSE": "#0072B2", "Worst RMSE": "#D55E00"}
    subsets = {"Best RMSE": best, "Worst RMSE": worst}
    if regular_hub is not None:
        colors = {"Most regular + Hub": "#009E73", **colors}
        subsets = {
            "Most regular + Hub": np.asarray(regular_hub["indices"]),
            **subsets,
        }

    fig = plt.figure(figsize=(13.5, 7.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=(1.25, 1.25, 1.0))
    axis = fig.add_subplot(grid[:, :2], projection="3d")
    low, high = -limit, limit
    for index, spacecraft_id in enumerate(result["ids"]):
        x, y, z = relative[index]
        color = spacecraft_color(spacecraft_id)
        axis.plot([x, low], [y, y], [z, z], "--", color=color, alpha=0.16, lw=0.55)
        axis.plot([x, x], [y, high], [z, z], "--", color=color, alpha=0.16, lw=0.55)
        axis.plot([x, x], [y, y], [z, low], "--", color=color, alpha=0.16, lw=0.55)
        axis.scatter([low, x, x], [y, high, y], [z, z, low], s=14, c=color,
                     alpha=0.38, edgecolors="none", depthshade=False)
        axis.scatter(x, y, z, s=52, c=color, edgecolors="#202020",
                     linewidths=0.5, depthshade=False, zorder=6)
        axis.text(x, y, z, f" {spacecraft_id}", fontsize=7)
    for label, subset in subsets.items():
        for left, right in combinations(subset.tolist(), 2):
            axis.plot(*relative[[left, right]].T, color=colors[label], lw=2.2,
                      alpha=0.92, zorder=5)
    axis.scatter(0.0, 0.0, 0.0, marker="x", s=70, color="black",
                 linewidths=1.5, depthshade=False, zorder=8)
    axis.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit),
             xlabel=r"$\Delta X_{GSE}$ (km)", ylabel=r"$\Delta Y_{GSE}$ (km)",
             zlabel=r"$\Delta Z_{GSE}$ (km)", title="Mesocentric Formation and Projections")
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=24, azim=-56)

    metric_by_subset = {tuple(item["indices"]): item for item in metrics}
    ql = fig.add_subplot(grid[0, 2])
    ep = fig.add_subplot(grid[1, 2])
    ql.scatter([item["Q"] for item in metrics], [item["L"] for item in metrics],
               s=10, color="0.72", alpha=0.5)
    ep.scatter([item["E"] for item in metrics], [item["P"] for item in metrics],
               s=10, color="0.72", alpha=0.5)
    for label, subset in subsets.items():
        metric = metric_by_subset[tuple(subset.tolist())]
        ql.scatter(metric["Q"], metric["L"], s=85, color=colors[label],
                   edgecolor="white", linewidth=0.7, label=label, zorder=4)
        ep.scatter(metric["E"], metric["P"], s=85, color=colors[label],
                   edgecolor="white", linewidth=0.7, label=label, zorder=4)
    ql.set(xlabel=r"$Q=\sqrt{E^2+P^2}$", ylabel=r"$L$ (km)",
           title="Tetrahedron Size and Shape")
    ql.set_yscale("log")
    ep.set(xlabel="Elongation E", ylabel="Planarity P", title="Tetrahedron Shape",
           xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    for panel in (ql, ep):
        panel.grid(alpha=0.22, which="both")
    handles, legend_labels = ql.get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc="upper right", bbox_to_anchor=(0.705, 0.88),
        fontsize=8,
    )
    metric_name = "angular" if result["truth"] is not None else "timing-residual"
    regular_suffix = (
        f", regular+Hub Q={regular_hub['Q']:.3g}"
        if regular_hub is not None else ""
    )
    fig.suptitle(
        f"Selected Four-Spacecraft Geometry by {source_label} "
        f"{metric_name} RMSE\n"
        f"best={rmse[best_index]:.3g}, worst={rmse[worst_index]:.3g}"
        f"{regular_suffix}"
    )
    for path in _plot_paths(output, result["settings"]["plot"], "tetrahedra_geometry"):
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=result["settings"]["plot"]["dpi"])
    plt.close(fig)


def _save_figures(result: Mapping[str, Any], output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plot = result["settings"]["plot"]
    mc = plot["representative_mc_index"]
    if not 0 <= mc < result["n_mc"]:
        raise ValueError("plot.representative_mc_index is outside the MC ensemble")
    # Detection: earliest, median, and latest successful combined/first-channel crossings.
    crossings = result["combined"]["crossing_time"][mc]
    valid = np.flatnonzero(result["combined"]["status"][mc] == 0)
    if not len(valid):
        crossings = result["independent"]["crossing_time"][0, mc]
        valid = np.flatnonzero(result["independent"]["status"][0, mc] == 0)
    representatives = (
        valid[np.argsort(crossings[valid])[[0, len(valid) // 2, -1]]]
        if len(valid) else np.arange(min(3, len(result["ids"])))
    )
    fig, axes = plt.subplots(
        len(result["observables"]), len(representatives),
        figsize=(4 * len(representatives), 2.1 * len(result["observables"])),
        squeeze=False,
    )
    for row, name in enumerate(result["observables"]):
        item = result["data"][name]
        for column, spacecraft in enumerate(representatives):
            ax = axes[row, column]
            ax.plot(item["time"], item["values"][mc, :, spacecraft], lw=0.8, color="0.35")
            status = int(result["independent"]["status"][row, mc, spacecraft])
            if status == 0:
                p = [result["independent"][key][row, mc, spacecraft]
                     for key in ("upstream", "downstream", "crossing_time", "width")]
                ax.plot(item["time"], _tanh(item["time"], *p), color="tab:red", lw=1.2)
                ax.axvline(p[2], color="tab:red", ls="--", lw=0.8)
            else:
                reason = next(key for key, value in DETECTION_STATUS.items() if value == status)
                ax.text(0.02, 0.92, reason.replace("_", " "), transform=ax.transAxes,
                        va="top", fontsize=7)
            if column == 0:
                ax.set_ylabel(name)
            if row == 0:
                ax.set_title(result["ids"][spacecraft])
    axes[-1, 0].set_xlabel("Time (s)")
    fig.tight_layout()
    for path in _plot_paths(output, plot, "detection"):
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=plot["dpi"])
    plt.close(fig)

    labels = list(result["solutions"])
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    truth_available = result["truth"] is not None
    from hs_observatory.visualization import polyhedron_metrics

    formation_positions = _positions_at_times(
        result["trajectory_time"], result["trajectory"],
        np.full(len(result["ids"]), result["time_origin"]),
    )
    hub_index = next(
        (index for index, spacecraft_id in enumerate(result["ids"])
         if spacecraft_id in {"H", "Hub"}),
        None,
    )
    regular_hub_index = None
    if hub_index is not None:
        formation_metrics = [
            item for item in polyhedron_metrics(formation_positions)
            if item["n_spacecraft"] == 4
        ]
        regular_hub = min(
            (item for item in formation_metrics
             if item["n_spacecraft"] == 4 and hub_index in item["indices"]),
            key=lambda item: item["Q"],
        )
        matches = np.flatnonzero(np.all(
            result["combinations"] == np.asarray(regular_hub["indices"]), axis=1
        ))
        if len(matches):
            regular_hub_index = int(matches[0])
    else:
        formation_metrics = [
            item for item in polyhedron_metrics(formation_positions)
            if item["n_spacecraft"] == 4
        ]
    metric_by_subset = {tuple(item["indices"]): item for item in formation_metrics}
    configuration_size = np.asarray([
        metric_by_subset[tuple(subset.tolist())]["L"]
        for subset in result["combinations"]
    ])
    metrics = (
        (("angular_error_deg", "Angular error (deg)"),
         ("speed_error", "Speed error (km/s)"),
         ("q_error_norm", "Slowness error (s/km)"),
         ("reference_time_error", "Reference-time error (s)"))
        if truth_available else
        (("speed", "Normal speed (km/s)"),
         ("q_norm", "Slowness magnitude (s/km)"),
         ("reference_time", "Reference time (s)"),
         ("rms_residual", "Timing RMS residual (s)"))
    )
    for ax, (metric, ylabel) in zip(axes.ravel(), metrics):
        method_specs = [("ols", None, "OLS"), ("wls", None, "WLS")]
        # Select representative tetrahedra by per-combination angular-error RMSE
        # (or timing-residual RMSE when truth is unavailable), without pooling them.
        representative_by_label = {}
        for label in labels:
            four = result["solutions"][label]["four_spacecraft"]
            selection_values = (
                four["errors"]["angular_error_deg"]
                if truth_available else four["rms_residual"]
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                performance = np.sqrt(np.nanmean(selection_values**2, axis=0))
            finite = np.flatnonzero(np.isfinite(performance))
            if len(finite):
                ordered = finite[np.argsort(performance[finite])]
                representative_by_label[label] = (
                    int(ordered[0]), int(ordered[len(ordered) // 2]), int(ordered[-1])
                )
        method_specs.extend(("four_spacecraft", slot, name) for slot, name in enumerate(
            ("4SC best", "4SC median", "4SC worst")
        ))
        if regular_hub_index is not None:
            method_specs.append(("four_spacecraft", "regular_hub", "4SC regular + Hub"))
        for offset, (method, slot, display) in enumerate(method_specs):
            rmses, percentile_low, percentile_high = [], [], []
            for label in labels:
                solution = result["solutions"][label][method]
                if truth_available:
                    values = solution["errors"][metric]
                elif metric == "q_norm":
                    values = np.linalg.norm(solution["q"], axis=-1)
                else:
                    values = solution[metric]
                if slot is not None:
                    if slot == "regular_hub":
                        values = values[:, regular_hub_index]
                    else:
                        chosen = representative_by_label.get(label)
                        values = np.asarray([]) if chosen is None else values[:, chosen[slot]]
                finite = values[np.isfinite(values)]
                rmse = float(np.sqrt(np.mean(finite**2))) if len(finite) else np.nan
                rmses.append(rmse)
                absolute = np.abs(finite)
                positive = absolute[absolute > 0.0]
                if len(positive):
                    low, high = np.percentile(positive, [16, 84])
                else:
                    low, high = np.nan, np.nan
                percentile_low.append(low)
                percentile_high.append(high)
            center = 0.5 * (len(method_specs) - 1)
            x = np.arange(len(labels)) + (offset - center) * 0.11
            marker, = ax.plot(x, rmses, "o", ms=4, label=display)
            color = marker.get_color()
            ax.vlines(x, percentile_low, percentile_high, color=color,
                      linewidth=1.0, alpha=0.82)
            cap = 0.025
            ax.hlines(percentile_low, x - cap, x + cap, color=color,
                      linewidth=1.0, alpha=0.82)
            ax.hlines(percentile_high, x - cap, x + cap, color=color,
                      linewidth=1.0, alpha=0.82)
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_ylabel(ylabel.replace("error", "RMSE"))
        ax.set_yscale("log")
        ax.grid(alpha=0.25, which="both")
    axes[0, 0].legend(fontsize=7)
    spread_name = "absolute error" if truth_available else "absolute quantity"
    fig.suptitle(
        f"Markers: MC RMSE; whiskers: 16th–84th percentiles of {spread_name}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    for path in _plot_paths(output, plot, "accuracy"):
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=plot["dpi"])
    plt.close(fig)

    selected_outlines = {}
    selected_tetrahedra = _best_worst_tetrahedra(result)
    if selected_tetrahedra is not None:
        _, best_index, worst_index, _ = selected_tetrahedra
        selected_outlines.update({
            best_index: ("Best RMSE", "#0072B2"),
            worst_index: ("Worst RMSE", "#D55E00"),
        })
    if regular_hub_index is not None:
        selected_outlines[regular_hub_index] = ("Most regular + Hub", "#009E73")

    fig, axes_grid = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes_grid.ravel()
    for label in labels:
        four = result["solutions"][label]["four_spacecraft"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            condition = np.nanmedian(four["condition_number"], axis=0)
            metric = four["errors"]["angular_error_deg"] if truth_available else four["speed"]
            error = np.nanmedian(metric, axis=0)
        axes[0].scatter(condition, error, s=10, alpha=0.55, label=label)
        axes[1].scatter(configuration_size, error, s=10, alpha=0.55, label=label)
        marker_size = 10.0 + 3.0 * np.clip(error, 0.0, None)
        edgecolors = np.full(len(error), "none", dtype=object)
        linewidths = np.zeros(len(error))
        for combination_index, (_, color) in selected_outlines.items():
            edgecolors[combination_index] = color
            linewidths[combination_index] = 1.6
        valid_geometry = (
            np.isfinite(configuration_size) & np.isfinite(condition)
            & np.isfinite(marker_size)
        )
        axes[3].scatter(
            configuration_size[valid_geometry], condition[valid_geometry],
            s=marker_size[valid_geometry], alpha=0.5,
            edgecolors=edgecolors[valid_geometry], linewidths=linewidths[valid_geometry],
            label="_nolegend_",
        )
        rates = [np.mean(result["solutions"][label][method]["status"] != 0)
                 for method in ("four_spacecraft", "ols", "wls")]
        axes[2].plot((0, 1, 2), rates, marker="o", label=label)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Scaled design condition number")
    axes[0].set_ylabel("Median angular error (deg)" if truth_available else "Median speed (km/s)")
    axes[0].set_yscale("log")
    axes[0].set_title("Conditioning and Accuracy")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Configuration size L (km)")
    axes[1].set_ylabel(
        "Median angular error (deg)" if truth_available else "Median speed (km/s)"
    )
    axes[1].set_title("Size and Accuracy")
    axes[2].set_xticks((0, 1, 2), ("Four-point", "OLS", "WLS"))
    axes[2].set_ylabel("Failure fraction")
    axes[2].set_title("Reconstruction Failures")
    axes[3].set_xscale("log")
    axes[3].set_yscale("log")
    axes[3].set_xlabel("Configuration size L (km)")
    axes[3].set_ylabel("Scaled design condition number")
    axes[3].set_title("Joint Geometry Reliability")
    size_values = (0.1, 1.0, 10.0, 50.0) if truth_available else (1.0, 10.0, 100.0)
    size_handles = [
        axes[3].scatter(
            [], [], s=10.0 + 3.0 * value, facecolor="none", edgecolor="0.3",
            label=f"{value:g}",
        )
        for value in size_values
    ]
    size_title = "Median angular error (deg)" if truth_available else "Median speed (km/s)"
    size_legend = axes[3].legend(
        handles=size_handles, title=size_title, fontsize=7, title_fontsize=7,
        loc="upper left",
    )
    selection_handles = [
        Line2D(
            [], [], marker="o", markersize=np.sqrt(55), markerfacecolor="none",
            markeredgecolor=color, markeredgewidth=1.6, linestyle="none", label=label,
        )
        for label, color in selected_outlines.values()
    ]
    if selection_handles:
        axes[3].add_artist(size_legend)
        axes[3].legend(
            handles=selection_handles, title="Selected tetrahedra",
            fontsize=7, title_fontsize=7, loc="center left",
        )
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    for path in _plot_paths(output, plot, "geometry"):
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=plot["dpi"])
    plt.close(fig)
    _save_tetrahedra_geometry(result, output, plt)


def write_shock_timing_product(result: Mapping[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    if output.resolve() == result["source"]:
        raise ValueError("Analysis output must not overwrite the observation input")
    output.parent.mkdir(parents=True, exist_ok=True)
    settings = result["settings"]
    with h5py.File(output, "w") as handle:
        metadata = handle.create_group("metadata")
        metadata.attrs.update(method="shock_timing", source_path=str(result["source"]),
                              settings=json.dumps(settings, sort_keys=True),
                              truth_available=result["truth"] is not None,
                              n_mc=result["n_mc"], complete=False)
        if result.get("analysis_config_yaml") is not None:
            metadata.attrs["analysis_config_yaml"] = result["analysis_config_yaml"]
            metadata.attrs["analysis_config_path"] = str(result["analysis_config_path"])
        spacecraft = handle.create_group("spacecraft")
        spacecraft.create_dataset(
            "id", data=np.asarray(result["ids"], dtype=h5py.string_dtype())
        )
        _store(spacecraft, "reference_position", result["reference_position"], settings, "km")
        spacecraft.attrs["time_origin_s"] = result["time_origin"]
        observable_group = handle.create_group("observables")
        observable_group.create_dataset(
            "name", data=np.asarray(result["observables"], dtype=h5py.string_dtype())
        )
        detection = handle.create_group("detection")
        _write_detection(detection.create_group("independent"), result["independent"], settings)
        combined = detection.create_group("combined")
        _write_detection(combined, result["combined"], settings)
        _store(combined, "contributors", result["contributors"], settings)
        detection_summary = detection.create_group("mc_summary")
        independent_counts = np.stack([
            np.sum(result["independent"]["status"] == code, axis=1)
            for code in sorted(DETECTION_STATUS.values())
        ])
        combined_counts = np.stack([
            np.sum(result["combined"]["status"] == code, axis=0)
            for code in sorted(DETECTION_STATUS.values())
        ])
        _store(detection_summary, "independent_status_count", independent_counts, settings)
        _store(detection_summary, "combined_status_count", combined_counts, settings)
        detection_summary.attrs["status_codes"] = json.dumps(
            DETECTION_STATUS, sort_keys=True
        )
        combinations_group = handle.create_group("four_spacecraft_combinations")
        _store(combinations_group, "indices", result["combinations"], settings)
        combinations_group.create_dataset("ids", data=np.asarray([
            [result["ids"][index] for index in subset] for subset in result["combinations"]
        ], dtype=h5py.string_dtype()))
        selected = _best_worst_tetrahedra(result)
        if selected is not None:
            source_label, best_index, worst_index, rmse = selected
            highlighted = combinations_group.create_group("highlighted")
            highlighted.attrs["selection_source"] = source_label
            highlighted.attrs["selection_metric"] = (
                "angular_error_rmse_deg"
                if result["truth"] is not None else "timing_residual_rmse_s"
            )
            for name, index in (("best", best_index), ("worst", worst_index)):
                subset = result["combinations"][index]
                group = highlighted.create_group(name)
                group.attrs["combination_index"] = index
                group.attrs["rmse"] = rmse[index]
                _store(group, "indices", subset, settings)
                group.create_dataset(
                    "ids",
                    data=np.asarray(
                        [result["ids"][member] for member in subset],
                        dtype=h5py.string_dtype(),
                    ),
                )
        if result["truth"] is not None:
            truth_group = handle.create_group("truth")
            for name, value in result["truth"].items():
                units = (
                    "km/s" if name in {"signed_normal_velocity", "speed"}
                    else "s/km" if name == "q"
                    else "s" if "time" in name else "1"
                )
                _store(truth_group, name, value, settings, units)
        reconstruction = handle.create_group("reconstruction")
        failures = []
        for observable_index, observable in enumerate(result["observables"]):
            for mc, spacecraft in np.argwhere(
                result["independent"]["status"][observable_index] != 0
            ):
                failures.append(
                    f"detection|independent|{observable}|{mc}|{spacecraft}|"
                    f"{result['independent']['status'][observable_index, mc, spacecraft]}"
                )
        for mc, spacecraft in np.argwhere(result["combined"]["status"] != 0):
            failures.append(
                f"detection|combined|combined|{mc}|{spacecraft}|"
                f"{result['combined']['status'][mc, spacecraft]}"
            )
        for label, methods in result["solutions"].items():
            label_group = reconstruction.create_group(label)
            for method, solution in methods.items():
                _write_solution(label_group.create_group(method), solution, settings)
                for index in np.argwhere(solution["status"] != 0):
                    failures.append("|".join(map(str, (
                        "reconstruction", method, label, *index.tolist(),
                        int(solution["status"][tuple(index)]),
                    ))))
            paired = label_group.create_group("paired_success")
            ols_ok = methods["ols"]["status"] == 0
            wls_ok = methods["wls"]["status"] == 0
            paired.attrs["ols_wls_count"] = int(np.count_nonzero(ols_ok & wls_ok))
            four_ok = methods["four_spacecraft"]["status"] == 0
            _store(paired, "four_ols_count", np.sum(four_ok & ols_ok[:, None], axis=0), settings)
            _store(paired, "four_wls_count", np.sum(four_ok & wls_ok[:, None], axis=0), settings)
        failure_group = handle.create_group("failure_records")
        failure_group.create_dataset("record", data=np.asarray(
            failures, dtype=h5py.string_dtype()
        ))
        failure_group.attrs["format"] = (
            "detection|mode|observable|mc|spacecraft|status OR "
            "reconstruction|method|analysis|mc|combination_if_present|status"
        )
        failure_group.attrs["status_codes"] = json.dumps(INVERSION_STATUS, sort_keys=True)
        failure_group.attrs["detection_status_codes"] = json.dumps(
            DETECTION_STATUS, sort_keys=True
        )
        metadata.attrs["complete"] = True
    if settings["plot"]["enabled"]:
        _save_figures(result, output)
    return output


@register_analysis
class ShockTimingAnalysis(AnalysisMethod):
    name = "shock_timing"
    config_section = "shock_timing"

    def run(self, input_path: str | Path) -> dict[str, Any]:
        return compute_shock_timing(input_path, self.config)

    def write_result(self, result: dict[str, Any], output_path: str | Path) -> Path:
        return write_shock_timing_product(result, output_path)
