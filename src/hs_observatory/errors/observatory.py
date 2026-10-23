"""Requirement-motivated HelioSwarm-like observatory error model.

The functions in this module operate in the package's native units:
time in seconds, position in km, density in cm^-3, velocity in km/s, and
magnetic field in nT. Requirement values carry a section-local ``sigma_level``
of either 2 or 3.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping

import numpy as np


DEFAULT_OBSERVATORY_ERROR_CONFIG: dict[str, Any] = {
    "timing": {
        "enabled": True,
        "requirement_ms": 12.5,
        "sigma_level": 3,
        "model": "constant_per_spacecraft",
    },
    "position": {
        "enabled": True,
        "requirement_magnitude_km": 2.5,
        "sigma_level": 3,
        "model": "isotropic_3d_gaussian_position_error",
    },
    "density": {
        "enabled": True,
        "requirement_fraction": 0.06,
        "sigma_level": 3,
        "model": "multiplicative_bias_plus_ou",
        "bias_variance_fraction": 0.5,
        "ou_variance_fraction": 0.5,
        "ou_correlation_time_s": 1.0,
    },
    "velocity_speed": {
        "enabled": True,
        "requirement_fraction": 0.03,
        "sigma_level": 3,
        "model": "multiplicative_bias_plus_ou",
        "bias_variance_fraction": 0.5,
        "ou_variance_fraction": 0.5,
        "ou_correlation_time_s": 1.0,
    },
    "velocity_angle": {
        "enabled": True,
        "requirement_deg": 0.25,
        "sigma_level": 3,
        "model": "transverse_ou_rotation",
        "ou_correlation_time_s": 1.0,
    },
    "magnetic_dc": {
        "enabled": True,
        "requirement_nT_per_axis": 0.15,
        "sigma_level": 3,
        "model": "constant_per_axis_offset",
    },
    "magnetic_ac": {
        "enabled": True,
        "model": "colored_gaussian_noise_merged_product",
        "requirement_interpretation": "total_vector_asd",
        "sigma_level": 3,
        "component_distribution": "uniform",
        "anchors": [
            {"frequency_Hz": 0.001, "asd_pT_per_sqrtHz": 1500.0},
            {"frequency_Hz": 1.0, "asd_pT_per_sqrtHz": 15.0},
            {"frequency_Hz": 10.0, "asd_pT_per_sqrtHz": 1.5},
        ],
        "extrapolate_above_10Hz": "continue_last_loglog_slope",
    },
    "alignment": {
        "enabled": True,
        "requirement_deg_per_axis": 3.0,
        "sigma_level": 3,
        "model": "independent_constant_rotation_B_and_V",
    },
}

_LEGACY_REQUIREMENT_KEYS = {
    "timing": ("requirement_3sigma_ms", "requirement_ms"),
    "position": ("requirement_3sigma_magnitude_km", "requirement_magnitude_km"),
    "density": ("requirement_3sigma_fraction", "requirement_fraction"),
    "velocity_speed": ("requirement_3sigma_fraction", "requirement_fraction"),
    "velocity_angle": ("requirement_3sigma_deg", "requirement_deg"),
    "magnetic_dc": ("requirement_3sigma_nT_per_axis", "requirement_nT_per_axis"),
    "alignment": ("requirement_3sigma_deg_per_axis", "requirement_deg_per_axis"),
}


def _deep_merge(defaults: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key, value in defaults.items():
        if isinstance(value, Mapping):
            merged[key] = _deep_merge(value, {})
        elif isinstance(value, list):
            merged[key] = [dict(item) if isinstance(item, Mapping) else item for item in value]
        else:
            merged[key] = value
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def observatory_errors_enabled(config: Mapping[str, Any]) -> bool:
    """Return true when the new observatory model is explicitly requested."""

    model = str(config.get("model", "")).lower()
    return model in {"observatory", "helioswarm_observatory"} or "observatory" in config


def normalized_observatory_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Merge user settings with observatory defaults.

    The preferred YAML shape places the sections directly under ``errors`` with
    ``errors.model: observatory``.  A nested ``errors.observatory`` mapping is
    also accepted.  Legacy ``errors.time`` is accepted as an alias for
    ``timing`` so older coordinate settings can be migrated incrementally.
    """

    user = dict(config or {})
    nested = user.get("observatory")
    if isinstance(nested, Mapping):
        user = {**user, **nested}
    if "time" in user and "timing" not in user:
        user["timing"] = user["time"]
    relevant = {
        key: value for key, value in user.items() if key in DEFAULT_OBSERVATORY_ERROR_CONFIG
    }
    canonical: dict[str, Any] = {}
    for section_name, section_value in relevant.items():
        if not isinstance(section_value, Mapping):
            canonical[section_name] = section_value
            continue
        section = dict(section_value)
        if section_name in _LEGACY_REQUIREMENT_KEYS:
            legacy_key, canonical_key = _LEGACY_REQUIREMENT_KEYS[section_name]
            if legacy_key in section:
                if canonical_key in section:
                    raise ValueError(
                        f"{section_name} cannot specify both {legacy_key} and {canonical_key}"
                    )
                if "sigma_level" in section and int(section["sigma_level"]) != 3:
                    raise ValueError(f"{section_name}.{legacy_key} implies sigma_level 3")
                section[canonical_key] = section.pop(legacy_key)
                section["sigma_level"] = 3
        if section_name == "magnetic_ac":
            interpretation = section.get("requirement_interpretation")
            if interpretation == "3sigma_total_vector_asd":
                if "sigma_level" in section and int(section["sigma_level"]) != 3:
                    raise ValueError("magnetic_ac.3sigma_total_vector_asd implies sigma_level 3")
                section["requirement_interpretation"] = "total_vector_asd"
                section["sigma_level"] = 3
            elif interpretation not in {None, "total_vector_asd"}:
                raise ValueError(
                    "magnetic_ac.requirement_interpretation must be 'total_vector_asd'"
                )
        canonical[section_name] = section
    merged = _deep_merge(DEFAULT_OBSERVATORY_ERROR_CONFIG, canonical)
    for section_name, section in merged.items():
        sigma_level = int(section["sigma_level"])
        if sigma_level not in {2, 3} or sigma_level != section["sigma_level"]:
            raise ValueError(f"{section_name}.sigma_level must be 2 or 3")
        section["sigma_level"] = sigma_level
        if section_name == "alignment":
            if section["model"] not in {
                "independent_constant_rotation_B_and_V",
                "relative_constant_rotation_B_to_V",
            }:
                raise ValueError(f"Unsupported alignment.model: {section['model']!r}")
            if section["model"] == "relative_constant_rotation_B_to_V":
                requirement = float(section["requirement_deg_per_axis"])
                if not np.isfinite(requirement) or requirement < 0:
                    raise ValueError("alignment requirement must be finite and non-negative")
        if section_name in {"density", "velocity_speed", "velocity_angle"}:
            if section_name == "velocity_angle":
                interpretations = {
                    "transverse_ou_rotation": "per_transverse_component",
                    "transverse_ou_rotation_total_angle": "total_angle_confidence_radius",
                }
                key = "requirement_interpretation"
                requirement = float(section["requirement_deg"])
            else:
                interpretations = {
                    "multiplicative_bias_plus_ou": "log_multiplier",
                    "gaussian_fractional_bias_plus_ou": "fractional_error",
                }
                key = "error_interpretation"
                requirement = float(section["requirement_fraction"])
            model = section["model"]
            if model not in interpretations:
                raise ValueError(f"Unsupported {section_name}.model: {model!r}")
            interpretation = interpretations[model]
            if section.get(key, interpretation) != interpretation:
                raise ValueError(f"{section_name}.{key} conflicts with model {model!r}")
            section[key] = interpretation
            if not np.isfinite(requirement) or requirement < 0:
                raise ValueError(f"{section_name} requirement must be finite and non-negative")
    return merged


def relative_alignment_metadata(section: Mapping[str, Any]) -> dict[str, Any]:
    """Describe the residual MAG-to-FC error in the nominal FC Cartesian axes."""
    return {
        "model": "relative_constant_rotation_B_to_V",
        "reference_instrument": "FC",
        "interpretation": "residual_MAG_to_FC_reconstruction_uncertainty",
        "perturbed_product": "merged_MAG_SCM_B",
        "requirement_deg_per_axis": float(section["requirement_deg_per_axis"]),
        "sigma_level": section["sigma_level"],
        "sigma_axis_rad": float(
            np.deg2rad(section["requirement_deg_per_axis"] / section["sigma_level"])
        ),
        "requirement_interpretation": "gaussian_rotvec_per_axis",
        "temporal_behavior": "constant_per_mc_spacecraft",
        "component_basis": "common_cartesian_nominal_FC",
        "attitude_assumption": "constant",
        "matrix_direction": "ideal_FC_referenced_B_to_reported_FC_referenced_B",
        "rotation_convention": "active_right_handed_column_vector",
        "application": "B_out=R@(B_sampled+DC+AC)",
        "rng_policy": "legacy_alignment_two_normal_blocks_relative_uses_second",
    }


def sigma_from_requirement(value: float, sigma_level: int) -> float:
    """Convert a symmetric requirement at 2 or 3 sigma to one deviation."""

    level = int(sigma_level)
    if level not in {2, 3} or level != sigma_level:
        raise ValueError("sigma_level must be 2 or 3")
    return float(value) / level


def sigma_from_3sigma(value: float) -> float:
    """Convert a symmetric 3-sigma requirement to one standard deviation."""

    return sigma_from_requirement(value, 3)


def component_sigma_from_3d_radial_requirement(radius: float, sigma_level: int) -> float:
    """Return component sigma for a 2σ or 3σ isotropic radial contour."""

    level = int(sigma_level)
    if level not in {2, 3} or level != sigma_level:
        raise ValueError("sigma_level must be 2 or 3")
    probability = 0.9545 if level == 2 else 0.9973
    return _component_sigma_from_3d_radial_probability(radius, probability)


def component_sigma_from_3d_radial_3sigma(
    radius_3sigma: float, probability: float = 0.9973
) -> float:
    """Per-component sigma for an isotropic 3D Gaussian radial requirement.

    For the default 99.73% probability this uses the Maxwell radius factor
    3.762, so a 2.5 km radial requirement gives about 0.665 km per component.
    A small bisection fallback is used for other probabilities.
    """

    return _component_sigma_from_3d_radial_probability(radius_3sigma, probability)


def _component_sigma_from_3d_radial_probability(radius: float, probability: float) -> float:
    radius = float(radius)
    probability = float(probability)
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must lie between 0 and 1")
    if np.isclose(probability, 0.9973):
        return radius / 3.762

    def cdf(x: float) -> float:
        return float(math.erf(x / np.sqrt(2.0)) - np.sqrt(2.0 / np.pi) * x * np.exp(-0.5 * x * x))

    lo, hi = 0.0, 16.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < probability:
            lo = mid
        else:
            hi = mid
    return radius / (0.5 * (lo + hi))


def generate_ou_process(
    n_times: int,
    dt: float | np.ndarray,
    sigma: float,
    tau: float,
    rng: np.random.Generator,
    sample_shape: tuple[int, ...] = (),
) -> np.ndarray:
    """Draw a stationary OU process with exact discrete transitions.

    ``dt`` may be a scalar cadence or the ``n_times - 1`` intervals between
    samples.  The returned array has shape ``(n_times, *sample_shape)``.
    """

    n_times = int(n_times)
    sigma = float(sigma)
    tau = float(tau)
    if n_times < 0:
        raise ValueError("n_times must be non-negative")
    if sigma < 0:
        raise ValueError("OU sigma must be non-negative")
    if tau <= 0:
        raise ValueError("OU tau must be positive")
    values = np.empty((n_times, *sample_shape), dtype=float)
    if n_times == 0:
        return values
    values[0] = rng.normal(scale=sigma, size=sample_shape)
    intervals = (
        np.full(n_times - 1, float(dt), dtype=float)
        if np.ndim(dt) == 0
        else np.asarray(dt, dtype=float)
    )
    if intervals.shape != (n_times - 1,):
        raise ValueError(f"dt must be scalar or shape ({n_times - 1},)")
    if np.any(intervals <= 0):
        raise ValueError("OU time intervals must be positive")
    for index, interval in enumerate(intervals, start=1):
        phi = float(np.exp(-interval / tau))
        values[index] = phi * values[index - 1] + rng.normal(
            scale=sigma * np.sqrt(1.0 - phi * phi), size=sample_shape
        )
    return values


def rotation_matrix_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    """Construct a right-handed Rodrigues rotation matrix from a rotation vector."""

    vec = np.asarray(rotvec, dtype=float)
    if vec.shape != (3,):
        raise ValueError("rotvec must have shape (3,)")
    angle = float(np.linalg.norm(vec))
    matrix = np.eye(3)
    if angle == 0.0:
        return matrix
    axis = vec / angle
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return matrix + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def rotate_vectors(vectors: np.ndarray, rotvec_or_matrix: np.ndarray) -> np.ndarray:
    """Rotate vectors with final dimension three by a rotation vector or matrix."""

    matrix = np.asarray(rotvec_or_matrix, dtype=float)
    if matrix.shape == (3,):
        matrix = rotation_matrix_from_rotvec(matrix)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must have shape (3,) or (3, 3)")
    values = np.asarray(vectors, dtype=float)
    if values.shape[-1] != 3:
        raise ValueError("vectors must have a final dimension of length three")
    return values @ matrix.T


def orthonormal_basis_perp_to_vector(vhat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two stable unit vectors perpendicular to a unit vector."""

    unit = np.asarray(vhat, dtype=float)
    if unit.shape != (3,):
        raise ValueError("vhat must have shape (3,)")
    norm = float(np.linalg.norm(unit))
    if norm == 0.0:
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    unit = unit / norm
    helper = np.array([1.0, 0.0, 0.0]) if abs(unit[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = helper - np.dot(helper, unit) * unit
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(unit, e1)
    return e1, e2


def _rotate_unit_by_transverse(unit: np.ndarray, transverse: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(transverse))
    if angle == 0.0:
        return unit
    tangent = transverse / angle
    return np.cos(angle) * unit + np.sin(angle) * tangent


def apply_transverse_angular_ou(
    vectors: np.ndarray,
    sigma_angle: float,
    tau: float,
    dt: float | np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply two OU angular errors in the plane transverse to each vector.

    The vector magnitudes are preserved exactly up to floating-point roundoff.
    Returns ``(rotated_vectors, alpha, beta)``.
    """

    values = np.asarray(vectors, dtype=float)
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError("vectors must have shape (n_mc, n_time, n_spacecraft, 3)")
    n_mc, n_times, n_spacecraft, _ = values.shape
    alpha = generate_ou_process(
        n_times, dt, sigma_angle, tau, rng, sample_shape=(n_mc, n_spacecraft)
    )
    beta = generate_ou_process(
        n_times, dt, sigma_angle, tau, rng, sample_shape=(n_mc, n_spacecraft)
    )
    alpha = np.moveaxis(alpha, 0, 1)
    beta = np.moveaxis(beta, 0, 1)
    magnitude = np.linalg.norm(values, axis=-1)
    output = values.copy()
    for mc_index in range(n_mc):
        for time_index in range(n_times):
            for sc_index in range(n_spacecraft):
                mag = magnitude[mc_index, time_index, sc_index]
                if mag == 0.0:
                    output[mc_index, time_index, sc_index] = 0.0
                    continue
                unit = values[mc_index, time_index, sc_index] / mag
                e1, e2 = orthonormal_basis_perp_to_vector(unit)
                transverse = (
                    alpha[mc_index, time_index, sc_index] * e1
                    + beta[mc_index, time_index, sc_index] * e2
                )
                output[mc_index, time_index, sc_index] = mag * _rotate_unit_by_transverse(
                    unit, transverse
                )
    return output, alpha, beta


def _time_intervals(time: np.ndarray) -> np.ndarray:
    values = np.asarray(time, dtype=float)
    if values.ndim != 1:
        raise ValueError("time must be one-dimensional")
    if len(values) > 1 and np.any(np.diff(values) <= 0):
        raise ValueError("time must be strictly increasing")
    return np.diff(values)


def _variance_partition(section: Mapping[str, Any], sigma_total: float) -> tuple[float, float]:
    bias_fraction = float(section.get("bias_variance_fraction", 0.5))
    ou_fraction = float(section.get("ou_variance_fraction", 1.0 - bias_fraction))
    if bias_fraction < 0 or ou_fraction < 0:
        raise ValueError("variance fractions must be non-negative")
    total = bias_fraction + ou_fraction
    if total <= 0:
        raise ValueError("at least one variance fraction must be positive")
    bias_fraction /= total
    ou_fraction /= total
    return sigma_total * np.sqrt(bias_fraction), sigma_total * np.sqrt(ou_fraction)


def _apply_multiplicative_bias_ou(
    sampled: np.ndarray,
    time: np.ndarray,
    section: Mapping[str, Any],
    requirement_key: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sigma_total = sigma_from_requirement(
        float(section[requirement_key]), int(section["sigma_level"])
    )
    sigma_bias, sigma_ou = _variance_partition(section, sigma_total)
    n_mc, n_times, n_spacecraft = sampled.shape
    bias = rng.normal(scale=sigma_bias, size=(n_mc, n_spacecraft))
    ou = generate_ou_process(
        n_times,
        _time_intervals(time),
        sigma_ou,
        float(section["ou_correlation_time_s"]),
        rng,
        sample_shape=(n_mc, n_spacecraft),
    )
    log_error = bias[:, None, :] + np.moveaxis(ou, 0, 1)
    return sampled * np.exp(log_error), bias, log_error


def _apply_gaussian_fractional_bias_ou(
    sampled: np.ndarray,
    time: np.ndarray,
    section: Mapping[str, Any],
    component: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply 1 + Gaussian bias + stationary OU, without clipping or resampling.

    Draw order matches the legacy lognormal helper, so later error components
    retain their seeded realizations when this option is selected.
    """
    sigma_total = sigma_from_requirement(
        float(section["requirement_fraction"]), int(section["sigma_level"])
    )
    sigma_bias, sigma_ou = _variance_partition(section, sigma_total)
    n_mc, n_times, n_spacecraft = sampled.shape
    bias = rng.normal(scale=sigma_bias, size=(n_mc, n_spacecraft))
    ou = generate_ou_process(
        n_times,
        _time_intervals(time),
        sigma_ou,
        float(section["ou_correlation_time_s"]),
        rng,
        sample_shape=(n_mc, n_spacecraft),
    )
    fractional_error = bias[:, None, :] + np.moveaxis(ou, 0, 1)
    factor = 1.0 + fractional_error
    invalid = ~np.isfinite(factor) | (factor <= 0)
    if np.any(invalid):
        index = tuple(int(i) for i in np.unravel_index(np.argmax(invalid), invalid.shape))
        raise ValueError(
            f"{component} gaussian_fractional_bias_plus_ou requires finite positive "
            f"multiplicative factors; factor={factor[index]!r} at "
            f"(mc, time, spacecraft)={index}"
        )
    return sampled * factor, bias, fractional_error


def transverse_component_sigma_rad(section: Mapping[str, Any]) -> float:
    """Convert the selected angular requirement, preserving legacy arithmetic."""
    if section["model"] == "transverse_ou_rotation":
        return np.deg2rad(
            sigma_from_requirement(float(section["requirement_deg"]), section["sigma_level"])
        )
    if section["model"] != "transverse_ou_rotation_total_angle":
        raise ValueError(f"Unsupported velocity_angle.model: {section['model']!r}")
    # erfc is 1 - erf but avoids subtractive cancellation in the tail.
    tail = math.erfc(section["sigma_level"] / np.sqrt(2.0))
    return np.deg2rad(float(section["requirement_deg"])) / np.sqrt(-2.0 * math.log(tail))


def magnetic_asd_total(
    frequency: np.ndarray,
    anchors: list[Mapping[str, Any]],
    extrapolation: str = "continue_last_loglog_slope",
) -> np.ndarray:
    """Evaluate the configured total-vector magnetic ASD in pT/sqrt(Hz)."""

    freq = np.asarray(frequency, dtype=float)
    if np.any(freq < 0):
        raise ValueError("frequency must be non-negative")
    points = sorted(
        (float(item["frequency_Hz"]), float(item["asd_pT_per_sqrtHz"])) for item in anchors
    )
    if len(points) < 2:
        raise ValueError("at least two ASD anchors are required")
    anchor_f = np.array([item[0] for item in points])
    anchor_a = np.array([item[1] for item in points])
    if np.any(anchor_f <= 0) or np.any(anchor_a <= 0):
        raise ValueError("ASD anchors must be positive")
    log_f = np.log(anchor_f)
    log_a = np.log(anchor_a)
    safe_freq = np.maximum(freq, anchor_f[0])
    result = np.exp(np.interp(np.log(safe_freq), log_f, log_a))
    below = freq < anchor_f[0]
    if np.any(below):
        first_slope = (log_a[1] - log_a[0]) / (log_f[1] - log_f[0])
        result[below] = (
            anchor_a[0] * (np.maximum(freq[below], anchor_f[0]) / anchor_f[0]) ** first_slope
        )
    above = freq > anchor_f[-1]
    if np.any(above):
        if extrapolation != "continue_last_loglog_slope":
            raise ValueError(f"Unknown magnetic ASD extrapolation: {extrapolation}")
        last_slope = (log_a[-1] - log_a[-2]) / (log_f[-1] - log_f[-2])
        result[above] = anchor_a[-1] * (freq[above] / anchor_f[-1]) ** last_slope
    return result


def magnetic_component_asd_1sigma_nT(
    frequency: np.ndarray, section: Mapping[str, Any]
) -> np.ndarray:
    """Convert total-vector requirement ASD to one-sigma component nT ASD."""

    total_pT = magnetic_asd_total(
        frequency,
        list(section["anchors"]),
        str(section.get("extrapolate_above_10Hz", "continue_last_loglog_slope")),
    )
    interpretation = section.get("requirement_interpretation", "3sigma_total_vector_asd")
    legacy = interpretation == "3sigma_total_vector_asd"
    if interpretation not in {"total_vector_asd", "3sigma_total_vector_asd"}:
        raise ValueError("requirement_interpretation must be 'total_vector_asd'")
    if legacy and section.get("sigma_level", 3) != 3:
        raise ValueError("3sigma_total_vector_asd implies sigma_level 3")
    sigma_level = 3 if legacy else section.get("sigma_level", 3)
    return total_pT / (float(sigma_level) * np.sqrt(3.0)) * 1.0e-3


def generate_colored_noise_from_asd(
    n_times: int,
    dt: float,
    asd_func: Callable[[np.ndarray], np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate real colored Gaussian noise from a one-sided ASD.

    ``asd_func`` returns the desired one-sigma one-sided amplitude spectral
    density in time-series units per sqrt(Hz).  The Fourier coefficients are
    scaled so the simple one-sided periodogram has expectation ``ASD**2``.
    """

    n_times = int(n_times)
    dt = float(dt)
    if n_times < 1:
        raise ValueError("n_times must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")
    freqs = np.fft.rfftfreq(n_times, dt)
    psd = np.asarray(asd_func(freqs), dtype=float) ** 2
    coeff = np.zeros(len(freqs), dtype=complex)
    if len(freqs) > 1:
        stop = -1 if n_times % 2 == 0 else None
        positive = slice(1, stop)
        scale = np.sqrt(n_times * psd[positive] / (4.0 * dt))
        coeff[positive] = scale * (rng.normal(size=scale.shape) + 1j * rng.normal(size=scale.shape))
    if n_times % 2 == 0 and len(freqs) > 1:
        coeff[-1] = rng.normal(scale=np.sqrt(n_times * psd[-1] / (2.0 * dt)))
    coeff[0] = 0.0
    return np.fft.irfft(coeff, n=n_times)


def estimate_asd(time_series: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a one-sided ASD from a uniformly sampled real time series."""

    values = np.asarray(time_series, dtype=float)
    if values.ndim != 1:
        raise ValueError("time_series must be one-dimensional")
    centered = values - np.mean(values)
    freqs = np.fft.rfftfreq(len(centered), float(dt))
    coeff = np.fft.rfft(centered)
    psd = (np.abs(coeff) ** 2) * (float(dt) / len(centered))
    if len(psd) > 1 and len(centered) % 2 == 0:
        psd[1:-1] *= 2.0
    elif len(psd) > 1:
        psd[1:] *= 2.0
    return freqs, np.sqrt(psd)


def magnetic_ac_asd_diagnostic(
    noise: np.ndarray, dt: float, section: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return estimated and target component ASD for generated magnetic noise."""

    freqs, estimated = estimate_asd(np.asarray(noise, dtype=float), dt)
    target = magnetic_component_asd_1sigma_nT(freqs, section)
    return freqs, estimated, target


def perturbed_pair_baselines(position: np.ndarray) -> np.ndarray:
    """Compute pair baselines self-consistently from spacecraft positions."""

    values = np.asarray(position, dtype=float)
    if values.ndim < 2 or values.shape[-1] != 3:
        raise ValueError("position must end with (spacecraft, 3) or include time before 3")
    return values[..., :, None, :] - values[..., None, :, :]


@dataclass(frozen=True)
class ObservatoryErrorResult:
    observed: dict[str, np.ndarray]
    errors: dict[str, np.ndarray]
    metadata: dict[str, str]


def apply_observatory_measurement_errors(
    sampled: Mapping[str, np.ndarray],
    time: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    rng: np.random.Generator,
) -> ObservatoryErrorResult:
    """Apply density, velocity, magnetic, and alignment errors to sampled truth."""

    cfg = normalized_observatory_config(config)
    observed: dict[str, np.ndarray] = {}
    errors: dict[str, np.ndarray] = {}

    n_sampled = np.asarray(sampled["n"], dtype=float)
    n_fractional = cfg["density"]["model"] == "gaussian_fractional_bias_plus_ou"
    if cfg["density"].get("enabled", True):
        n_apply = (
            _apply_gaussian_fractional_bias_ou if n_fractional else _apply_multiplicative_bias_ou
        )
        n_obs, n_bias, n_error = n_apply(
            n_sampled,
            np.asarray(time["n"], dtype=float),
            cfg["density"],
            "density" if n_fractional else "requirement_fraction",
            rng,
        )
    else:
        n_obs = n_sampled.copy()
        n_bias = np.zeros((n_sampled.shape[0], n_sampled.shape[2]))
        n_error = np.zeros_like(n_sampled)
    observed["n"] = n_obs
    errors["n"] = n_obs - n_sampled
    errors["density_bias"] = n_bias
    errors["density_fractional_error" if n_fractional else "density_log_error"] = n_error

    v_sampled = np.asarray(sampled["v"], dtype=float)
    v_work = v_sampled.copy()
    v_fractional = cfg["velocity_speed"]["model"] == "gaussian_fractional_bias_plus_ou"
    if cfg["velocity_speed"].get("enabled", True):
        speed = np.linalg.norm(v_work, axis=-1)
        v_apply = (
            _apply_gaussian_fractional_bias_ou if v_fractional else _apply_multiplicative_bias_ou
        )
        speed_obs, v_bias, v_error = v_apply(
            speed,
            np.asarray(time["v"], dtype=float),
            cfg["velocity_speed"],
            "velocity_speed" if v_fractional else "requirement_fraction",
            rng,
        )
        scale = np.divide(speed_obs, speed, out=np.ones_like(speed), where=speed > 0.0)
        v_work = v_work * scale[..., None]
    else:
        v_bias = np.zeros((v_sampled.shape[0], v_sampled.shape[2]))
        v_error = np.zeros(v_sampled.shape[:-1])
    errors["velocity_speed_bias"] = v_bias
    errors["velocity_speed_fractional_error" if v_fractional else "velocity_speed_log_error"] = (
        v_error
    )
    if cfg["velocity_angle"].get("enabled", True):
        sigma_angle = transverse_component_sigma_rad(cfg["velocity_angle"])
        v_work, alpha, beta = apply_transverse_angular_ou(
            v_work,
            sigma_angle,
            float(cfg["velocity_angle"]["ou_correlation_time_s"]),
            _time_intervals(np.asarray(time["v"], dtype=float)),
            rng,
        )
    else:
        alpha = np.zeros(v_sampled.shape[:-1])
        beta = np.zeros(v_sampled.shape[:-1])
    errors["velocity_angle_alpha"] = alpha
    errors["velocity_angle_beta"] = beta

    n_mc = v_sampled.shape[0]
    n_spacecraft = v_sampled.shape[2]
    relative = cfg["alignment"]["model"] == "relative_constant_rotation_B_to_V"
    align_sigma = np.deg2rad(
        sigma_from_requirement(
            float(cfg["alignment"]["requirement_deg_per_axis"]),
            cfg["alignment"]["sigma_level"],
        )
    )
    if cfg["alignment"].get("enabled", True):
        if relative:
            # Reserve the legacy V draw block without introducing an FC rotation.
            rng.normal(scale=align_sigma, size=(n_mc, n_spacecraft, 3))
        else:
            v_align_rotvec = rng.normal(scale=align_sigma, size=(n_mc, n_spacecraft, 3))
        b_align_rotvec = rng.normal(scale=align_sigma, size=(n_mc, n_spacecraft, 3))
    else:
        v_align_rotvec = np.zeros((n_mc, n_spacecraft, 3))
        b_align_rotvec = np.zeros((n_mc, n_spacecraft, 3))
    if not relative:
        v_matrices = np.empty((n_mc, n_spacecraft, 3, 3), dtype=float)
    b_matrices = np.empty((n_mc, n_spacecraft, 3, 3), dtype=float)
    for mc_index in range(n_mc):
        for sc_index in range(n_spacecraft):
            if not relative:
                v_matrices[mc_index, sc_index] = rotation_matrix_from_rotvec(
                    v_align_rotvec[mc_index, sc_index]
                )
            b_matrices[mc_index, sc_index] = rotation_matrix_from_rotvec(
                b_align_rotvec[mc_index, sc_index]
            )
            if not relative:
                v_work[mc_index, :, sc_index, :] = rotate_vectors(
                    v_work[mc_index, :, sc_index, :], v_matrices[mc_index, sc_index]
                )
    observed["v"] = v_work
    errors["v"] = v_work - v_sampled
    if not relative:
        errors["velocity_alignment_rotvec"] = v_align_rotvec
        errors["velocity_alignment_matrix"] = v_matrices

    b_sampled = np.asarray(sampled["B"], dtype=float)
    b_work = b_sampled.copy()
    if cfg["magnetic_dc"].get("enabled", True):
        dc_sigma = sigma_from_requirement(
            float(cfg["magnetic_dc"]["requirement_nT_per_axis"]),
            cfg["magnetic_dc"]["sigma_level"],
        )
        dc = rng.normal(scale=dc_sigma, size=(b_sampled.shape[0], b_sampled.shape[2], 3))
        b_work = b_work + dc[:, None, :, :]
    else:
        dc = np.zeros((b_sampled.shape[0], b_sampled.shape[2], 3))
    errors["magnetic_dc_offset"] = dc
    if cfg["magnetic_ac"].get("enabled", True):
        b_time = np.asarray(time["B"], dtype=float)
        if len(b_time) < 2:
            raise ValueError("magnetic AC noise requires at least two B samples")
        dt_values = np.diff(b_time)
        if not np.allclose(dt_values, dt_values[0]):
            raise ValueError("magnetic AC spectral noise requires uniform B cadence")
        dt = float(dt_values[0])
        ac = np.empty_like(b_work)
        section = cfg["magnetic_ac"]
        for mc_index in range(b_work.shape[0]):
            for sc_index in range(b_work.shape[2]):
                for component in range(3):
                    ac[mc_index, :, sc_index, component] = generate_colored_noise_from_asd(
                        b_work.shape[1],
                        dt,
                        lambda f, section=section: magnetic_component_asd_1sigma_nT(f, section),
                        rng,
                    )
        b_work = b_work + ac
    else:
        ac = np.zeros_like(b_work)
    errors["magnetic_ac_noise"] = ac
    for mc_index in range(b_work.shape[0]):
        for sc_index in range(b_work.shape[2]):
            b_work[mc_index, :, sc_index, :] = rotate_vectors(
                b_work[mc_index, :, sc_index, :], b_matrices[mc_index, sc_index]
            )
    observed["B"] = b_work
    errors["B"] = b_work - b_sampled
    prefix = "relative" if relative else "magnetic"
    errors[prefix + "_alignment_rotvec"] = b_align_rotvec
    errors[prefix + "_alignment_matrix"] = b_matrices

    metadata = {
        "observatory_error_config": cfg,
        "observatory_error_model": "helioswarm_requirement_observatory",
    }
    if relative:
        metadata["relative_alignment"] = relative_alignment_metadata(cfg["alignment"])
    return ObservatoryErrorResult(observed=observed, errors=errors, metadata=metadata)
