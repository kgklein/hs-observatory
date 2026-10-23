"""Independent configuration interpretation and mathematical expectations.

No production generators or requirement-conversion functions are imported.
The rounded 3-sigma radial factor and FFT endpoint convention intentionally
match the implemented model documented in the Stage 1 audit.
"""

from copy import deepcopy
import math

import numpy as np
from scipy import signal, stats

DEFAULTS = {
    "timing": dict(
        enabled=True, requirement_ms=12.5, sigma_level=3, model="constant_per_spacecraft"
    ),
    "position": dict(
        enabled=True,
        requirement_magnitude_km=2.5,
        sigma_level=3,
        model="isotropic_3d_gaussian_position_error",
    ),
    "density": dict(
        enabled=True,
        requirement_fraction=0.06,
        sigma_level=3,
        model="multiplicative_bias_plus_ou",
        bias_variance_fraction=0.5,
        ou_variance_fraction=0.5,
        ou_correlation_time_s=1.0,
    ),
    "velocity_speed": dict(
        enabled=True,
        requirement_fraction=0.03,
        sigma_level=3,
        model="multiplicative_bias_plus_ou",
        bias_variance_fraction=0.5,
        ou_variance_fraction=0.5,
        ou_correlation_time_s=1.0,
    ),
    "velocity_angle": dict(
        enabled=True,
        requirement_deg=0.25,
        sigma_level=3,
        model="transverse_ou_rotation",
        ou_correlation_time_s=1.0,
    ),
    "magnetic_dc": dict(
        enabled=True, requirement_nT_per_axis=0.15, sigma_level=3, model="constant_per_axis_offset"
    ),
    "magnetic_ac": dict(
        enabled=True,
        model="colored_gaussian_noise_merged_product",
        requirement_interpretation="total_vector_asd",
        sigma_level=3,
        component_distribution="uniform",
        anchors=[
            dict(frequency_Hz=0.001, asd_pT_per_sqrtHz=1500.0),
            dict(frequency_Hz=1.0, asd_pT_per_sqrtHz=15.0),
            dict(frequency_Hz=10.0, asd_pT_per_sqrtHz=1.5),
        ],
        extrapolate_above_10Hz="continue_last_loglog_slope",
    ),
    "alignment": dict(
        enabled=True,
        requirement_deg_per_axis=3.0,
        sigma_level=3,
        model="independent_constant_rotation_B_and_V",
    ),
}
REQUIREMENTS = {
    "timing": ("requirement_ms", "requirement_3sigma_ms"),
    "position": ("requirement_magnitude_km", "requirement_3sigma_magnitude_km"),
    "density": ("requirement_fraction", "requirement_3sigma_fraction"),
    "velocity_speed": ("requirement_fraction", "requirement_3sigma_fraction"),
    "velocity_angle": ("requirement_deg", "requirement_3sigma_deg"),
    "magnetic_dc": ("requirement_nT_per_axis", "requirement_3sigma_nT_per_axis"),
    "alignment": ("requirement_deg_per_axis", "requirement_3sigma_deg_per_axis"),
}


def normalize(config):
    """Resolve documented defaults/aliases independently, rejecting unsupported models."""
    if not isinstance(config, dict):
        raise ValueError("error configuration must be a mapping")
    user = dict(config)
    if user.get("model", "observatory") not in ("observatory", "helioswarm_observatory"):
        raise ValueError("only observatory error configurations are supported")
    if "observatory" in user:
        user.update(user["observatory"])
    if "time" in user and "timing" not in user:
        user["timing"] = user["time"]
    result = deepcopy(DEFAULTS)
    for name, default in DEFAULTS.items():
        section = dict(user.get(name, {}))
        if name in REQUIREMENTS:
            current, legacy = REQUIREMENTS[name]
            if legacy in section:
                if current in section or section.get("sigma_level", 3) != 3:
                    raise ValueError(f"conflicting legacy requirement in {name}")
                section[current] = section.pop(legacy)
                section["sigma_level"] = 3
        if section.get("requirement_interpretation") == "3sigma_total_vector_asd":
            if section.get("sigma_level", 3) != 3:
                raise ValueError("legacy magnetic ASD implies sigma_level=3")
            section.update(requirement_interpretation="total_vector_asd", sigma_level=3)
        allowed = set(default) | {"reference_spacecraft", "radial_probability"}
        if name in ("density", "velocity_speed"):
            allowed.add("error_interpretation")
        if name == "velocity_angle":
            allowed.add("requirement_interpretation")
        unknown = set(section) - allowed
        if unknown:
            raise ValueError(f"unsupported {name} parameters: {sorted(unknown)}")
        result[name].update(section)
        s = result[name]
        if not isinstance(s["enabled"], bool) or s["sigma_level"] not in (2, 3):
            raise ValueError(f"invalid enabled/sigma_level in {name}")
        options = {default["model"]}
        if name == "alignment":
            options.add("relative_constant_rotation_B_to_V")
        if name in ("density", "velocity_speed"):
            options.add("gaussian_fractional_bias_plus_ou")
        if name == "velocity_angle":
            options.add("transverse_ou_rotation_total_angle")
        if s["model"] not in options:
            raise ValueError(f"unsupported {name} model: {s['model']}")
        if name in ("density", "velocity_speed", "velocity_angle"):
            key = (
                "requirement_interpretation" if name == "velocity_angle" else "error_interpretation"
            )
            meaning = (
                {
                    "transverse_ou_rotation": "per_transverse_component",
                    "transverse_ou_rotation_total_angle": "total_angle_confidence_radius",
                }[s["model"]]
                if name == "velocity_angle"
                else "fractional_error"
                if s["model"] == "gaussian_fractional_bias_plus_ou"
                else "log_multiplier"
            )
            if s.get(key, meaning) != meaning:
                raise ValueError(f"{name}.{key} contradicts embedded model {s['model']}")
            s[key] = meaning
        for key, value in s.items():
            if key.startswith("requirement_") and key != "requirement_interpretation":
                if not np.isfinite(value) or value < 0:
                    raise ValueError(f"invalid {name}.{key}")
        if "ou_correlation_time_s" in s:
            if not np.isfinite(s["ou_correlation_time_s"]) or s["ou_correlation_time_s"] <= 0:
                raise ValueError(f"invalid {name} OU correlation time")
        if name in ("density", "velocity_speed"):
            fractions = np.array([s["bias_variance_fraction"], s["ou_variance_fraction"]])
            if not np.all(np.isfinite(fractions)) or np.any(fractions < 0) or fractions.sum() <= 0:
                raise ValueError(f"invalid {name} variance fractions")
            fractions = fractions / fractions.sum()
            s["bias_variance_fraction"], s["ou_variance_fraction"] = fractions.tolist()
        if name == "position" and "radial_probability" in s:
            if not 0 < s["radial_probability"] < 1:
                raise ValueError("radial_probability must be strictly between zero and one")
        if name == "magnetic_ac":
            if (
                s["requirement_interpretation"] != "total_vector_asd"
                or s["component_distribution"] != "uniform"
                or s["extrapolate_above_10Hz"] != "continue_last_loglog_slope"
            ):
                raise ValueError("unsupported magnetic ASD convention")
            a = np.array([[v["frequency_Hz"], v["asd_pT_per_sqrtHz"]] for v in s["anchors"]])
            a = a[np.argsort(a[:, 0])]
            if len(a) < 2 or not np.all(np.isfinite(a)) or np.any(a <= 0):
                raise ValueError("magnetic anchors must be finite and positive")
            if np.any(np.diff(a[:, 0]) <= 0):
                raise ValueError("magnetic anchor frequencies must be distinct")
            s["anchors"] = [dict(frequency_Hz=float(f), asd_pT_per_sqrtHz=float(v)) for f, v in a]
    return result


def differences(expected, actual, prefix=""):
    """Return dotted, structured discrepancies without ignoring missing keys."""
    out = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(expected.keys() | actual.keys()):
            out.extend(
                differences(expected.get(key), actual.get(key), f"{prefix}.{key}".strip("."))
            )
    elif expected != actual:
        out.append(dict(parameter=prefix, supplied=expected, embedded=actual))
    return out


def position_sigma(section):
    p = section.get("radial_probability", 0.9545 if section["sigma_level"] == 2 else 0.9973)
    factor = 3.762 if np.isclose(p, 0.9973) else stats.chi.ppf(p, 3)
    return section["requirement_magnitude_km"] / factor, p


def split_sigma(section):
    total = section["requirement_fraction"] / section["sigma_level"]
    return (
        total,
        total * np.sqrt(section["bias_variance_fraction"]),
        total * np.sqrt(section["ou_variance_fraction"]),
    )


def angular_sigma(section):
    """Independent angular conversion; no production conversion helpers."""
    radius = np.deg2rad(section["requirement_deg"])
    if section["model"] == "transverse_ou_rotation_total_angle":
        probability = math.erf(section["sigma_level"] / math.sqrt(2))
        return radius / math.sqrt(-2 * math.log1p(-probability))
    return radius / section["sigma_level"]


def component_asd(frequency, section):
    anchors = section["anchors"]
    f = np.array([a["frequency_Hz"] for a in anchors])
    a = np.array([a["asd_pT_per_sqrtHz"] for a in anchors])
    x = np.maximum(np.asarray(frequency), f[0])
    log_a = np.interp(np.log(x), np.log(f), np.log(a))
    slope = np.log(a[-1] / a[-2]) / np.log(f[-1] / f[-2])
    log_a = np.where(x > f[-1], np.log(a[-1]) + slope * np.log(x / f[-1]), log_a)
    return np.exp(log_a) * 1e-3 / (section["sigma_level"] * np.sqrt(3))


def spectral_target(n, dt, section, length):
    """Exact expected Hann one-sided PSD of a segment of the periodic finite record.

    C[l] = irfft(a_k**2/(2 dt)). This includes zero DC and the production
    even-N half-power Nyquist bin. Folding the linear taper autocorrelation
    times C onto segment FFT lags accounts for leakage without simulation.
    """
    if not 2 <= length <= n:
        raise ValueError("spectral segment must have 2 <= length <= n")
    a = component_asd(np.fft.rfftfreq(n, dt), section)
    a[0] = 0
    covariance = np.fft.irfft(a**2 / (2 * dt), n=n)
    window = signal.windows.hann(length, sym=False)
    ac = signal.fftconvolve(window, window[::-1], mode="full")[length - 1 :]
    folded = ac * covariance[:length]
    folded[1:] += (ac[1:] * covariance[1:length])[::-1]
    expected = np.fft.rfft(folded).real * dt / np.dot(window, window)
    expected[1 : -1 if length % 2 == 0 else None] *= 2
    return np.fft.rfftfreq(length, dt), np.maximum(expected, 0), covariance[0]


def rodrigues(vector):
    theta = np.linalg.norm(vector)
    if theta == 0:
        return np.eye(3)
    x, y, z = vector / theta
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(theta) * cross + (1 - np.cos(theta)) * cross @ cross
