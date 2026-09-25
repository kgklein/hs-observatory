"""Flight-accessible MAG noise weights for the existing divergence timing fit."""

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from hs_observatory.analysis.magnetic_gradient import geometry_diagnostics, geometry_is_static


def interpolate_asd(frequency, knots, asd, *, below_range="error", above_range="error"):
    """Log-log ASD interpolation; endpoint extension requires an explicit 'flat'."""
    frequency = np.asarray(frequency, dtype=float)
    knots, asd = np.asarray(knots, dtype=float), np.asarray(asd, dtype=float)
    if (knots.ndim != 1 or len(knots) < 2 or asd.shape != knots.shape
            or not np.isfinite(knots).all() or not np.isfinite(asd).all()
            or np.any(knots <= 0) or np.any(asd <= 0) or np.any(np.diff(knots) <= 0)):
        raise ValueError("Noise spectrum requires increasing positive frequencies and positive ASD")
    if not np.isfinite(frequency).all() or np.any(frequency < 0):
        raise ValueError("Noise frequencies must be finite and nonnegative")
    if below_range not in {"error", "flat"} or above_range not in {"error", "flat"}:
        raise ValueError("Noise range policies must be error or flat")
    if below_range == "error" and np.any(frequency < knots[0]):
        raise ValueError("Noise below supplied spectrum: explicitly set below_range or supply a filter/model")
    if above_range == "error" and np.any(frequency > knots[-1]):
        raise ValueError("Noise above supplied spectrum: explicitly set above_range or extend the model")
    return np.exp(np.interp(np.log(np.clip(frequency, knots[0], knots[-1])),
                           np.log(knots), np.log(asd)))


def integrate_noise(time, model: Mapping[str, Any], options: Mapping[str, Any]) -> dict[str, Any]:
    """Integrate one-sided PSD with the declared processing of the input samples.

    Processing declarations describe the already supplied time series; this
    routine never applies a filter or changes calibration measurements.
    """
    time = np.asarray(time, dtype=float)
    if time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all():
        raise ValueError("Instrument noise requires at least two finite sample times")
    cadence = float(np.mean(np.diff(time)))
    if cadence <= 0 or not np.allclose(np.diff(time), cadence, rtol=1e-7, atol=1e-12):
        raise ValueError("Instrument noise currently requires uniformly sampled calibration data")
    if model.get("frequency_units") != "Hz" or model.get("asd_units") != "nT/sqrt(Hz)":
        raise ValueError("Noise model units must be Hz and nT/sqrt(Hz)")
    spectrum = np.asarray(model.get("spectrum"), dtype=float)
    if spectrum.ndim != 2 or spectrum.shape[1] != 2:
        raise ValueError("Noise spectrum must contain [frequency, ASD] pairs")
    frequency, asd = spectrum.T
    # Validate knots independently of the integration range.
    interpolate_asd(frequency, frequency, asd)
    interpretation = options.get("interpretation", model.get("interpretation", "per_component"))
    if interpretation not in {"per_component", "vector_total"}:
        raise ValueError("ASD interpretation must be per_component or vector_total")
    component_asd = asd / (np.sqrt(3) if interpretation == "vector_total" else 1)
    below = options.get("below_range", "error")
    above = options.get("above_range", "error")
    processing = dict(options.get("processing", {}))
    if processing.get("sampling") != "ideal_antialias":
        raise ValueError("Explicit processing.sampling: ideal_antialias is required")
    duration = float(processing.get("averaging_duration_s", 0))
    if not np.isfinite(duration) or duration < 0:
        raise ValueError("averaging_duration_s must be finite and nonnegative")
    response = processing.get("transfer_power")
    if (duration > 0 or response is not None) and processing.get("applied_to_input") is not True:
        raise ValueError("Averaging/filter response must describe processing already applied_to_input")
    nyquist = .5 / cadence
    response_f = response_p = None
    if response is not None:
        table = np.asarray(response, dtype=float)
        if (table.ndim != 2 or table.shape[1] != 2 or len(table) < 2
                or not np.isfinite(table).all() or np.any(np.diff(table[:, 0]) <= 0)
                or table[0, 0] != 0 or table[-1, 0] < nyquist or np.any(table[:, 1] < 0)):
            raise ValueError("transfer_power must cover 0 through Nyquist with nonnegative |H|^2")
        response_f, response_p = table.T
    # Split at spectral and response knots, with extra panels to resolve boxcar lobes.
    anchors = np.unique(np.r_[0., frequency[frequency < nyquist], nyquist])
    panels = []
    for low, high in zip(anchors[:-1], anchors[1:]):
        panels.extend(np.linspace(low, high, 17) if low == 0 else np.geomspace(low, high, 17))
    if response_f is not None:
        panels.extend(response_f[(response_f >= 0) & (response_f <= nyquist)])
    if duration > 0:
        count = int(np.ceil(nyquist*duration*16)) + 1
        if count > 100_000:
            raise ValueError("Averaging response requires too many integration panels")
        panels.extend(np.linspace(0, nyquist, max(2, count)))
    edges = np.unique(panels)
    abscissa, weights = np.polynomial.legendre.leggauss(16)
    widths = np.diff(edges)
    grid = (edges[:-1, None] + .5*widths[:, None]*(1+abscissa)).ravel()
    quadrature = (.5*widths[:, None]*weights).ravel()
    power = np.sinc(grid*duration)**2
    if response_f is not None:
        power *= np.interp(grid, response_f, response_p)
    supported = power > 0
    grid_asd = np.full_like(grid, np.nan)
    grid_asd[supported] = interpolate_asd(grid[supported], frequency, component_asd,
                                         below_range=below, above_range=above)
    effective_psd = np.zeros_like(grid)
    effective_psd[supported] = grid_asd[supported]**2 * power[supported]
    variance = float(np.sum(quadrature*effective_psd))
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("Integrated per-component magnetic variance must be finite and positive")
    return {
        "frequency_hz": frequency, "input_asd_nT_sqrtHz": asd,
        "component_asd_nT_sqrtHz": component_asd, "component_psd_nT2_per_Hz": component_asd**2,
        "integration_frequency_hz": grid, "integration_weights_hz": quadrature,
        "transfer_power": power, "effective_psd_nT2_per_Hz": effective_psd,
        "sigma_B_nT": np.sqrt(variance), "sigma_B_squared_nT2": variance,
        "integration_limits_hz": np.array([0., nyquist]), "cadence_s": cadence,
        "interpretation": interpretation, "below_range": below, "above_range": above,
        "processing": processing,
    }


def noise_divergence_variance(positions, polyhedra, sigma_B_squared, floor=0.):
    """sigma_d^2 = sigma_B^2 Tr(R^-1)/N; singular geometry gets invalid infinite variance."""
    if not np.isfinite(floor) or floor < 0:
        raise ValueError("variance_floor_nT2_per_km2 must be finite and nonnegative")
    if not np.isfinite(sigma_B_squared) or sigma_B_squared <= 0:
        raise ValueError("sigma_B_squared must be finite and positive")
    samples = positions[:1] if geometry_is_static(positions) else positions
    raw = np.full((len(samples), len(polyhedra)), np.inf)
    for column, indices in enumerate(polyhedra):
        for time_index, sample in enumerate(samples):
            geometry = geometry_diagnostics(sample[indices])
            if geometry.rank == 3 and np.all(geometry.eigenvalues > 0):
                raw[time_index, column] = sigma_B_squared*np.sum(1/geometry.eigenvalues)/len(indices)
    raw = raw[0] if len(samples) == 1 else raw
    return raw, np.maximum(raw, floor)


def prepare_noise(time, positions, polyhedra, config):
    options = dict(config.get("instrument_noise", {}))
    if not options.get("spectrum_file"):
        raise ValueError("instrument_noise.spectrum_file is required")
    path = Path(options["spectrum_file"]).expanduser()
    if not path.is_absolute() and config.get("_analysis_config_path"):
        path = Path(config["_analysis_config_path"]).parent / path
    model_text = path.read_text()
    model = yaml.safe_load(model_text)
    if not isinstance(model, Mapping):
        raise ValueError("Noise spectrum file must contain a mapping")
    noise = integrate_noise(time, model, options)
    floor = float(options.get("variance_floor_nT2_per_km2", 0.))
    raw, variance = noise_divergence_variance(positions, polyhedra, noise["sigma_B_squared_nT2"], floor)
    noise.update(spectrum_file=str(path.resolve()), spectrum_yaml=model_text,
                 variance_floor_nT2_per_km2=floor)
    return noise, raw, variance


def write_noise(handle, result):
    from hs_observatory.analysis.timing_calibrate import _store

    group = handle.create_group("instrument_noise")
    settings = result["settings"]
    for key, value in result["noise"].items():
        if isinstance(value, (str, dict)):
            group.attrs[key] = yaml.safe_dump(value) if isinstance(value, dict) else value
        else:
            units = ("nT/sqrt(Hz)" if "asd" in key else "nT^2/Hz" if "psd" in key else
                     "Hz" if key.endswith("_hz") else "s" if key == "cadence_s" else
                     "nT" if key == "sigma_B_nT" else "nT^2" if key == "sigma_B_squared_nT2" else
                     "(nT/km)^2" if key in {"variance_floor_nT2_per_km2", "ideal_reference_variance"}
                     else "1")
            _store(group, key, value, settings, units)
    group.attrs["assumptions"] = (
        "independent spacecraft and isotropic uncorrelated components; one-sided PSD; "
        "declared processing already describes input samples; no filter applied here"
    )


def plot_noise_weights(product_path, output_path, *, dpi=300):
    """Plot the implemented spectrum and weights against diagnostic ideal weights."""
    import h5py
    import matplotlib.pyplot as plt

    with h5py.File(product_path, "r") as handle:
        if "instrument_noise" not in handle:
            return []
        if not handle["metadata"].attrs.get("complete", False):
            raise ValueError("Timing-calibration product is incomplete")
        noise = handle["instrument_noise"]
        frequency = noise["frequency_hz"][...]
        component_asd = noise["component_asd_nT_sqrtHz"][...]
        grid = noise["integration_frequency_hz"][...]
        processed_asd = np.sqrt(noise["effective_psd_nT2_per_Hz"][...])
        sigma = float(noise["sigma_B_nT"][()])
        ideal_weights = 1/noise["ideal_reference_variance"][...]
        names = handle["configurations"].attrs["order"]
        instrument_weights = np.array([
            np.median(1/handle[f"configurations/{name}/sigma_d_squared"][...]) for name in names
        ])
        fractions = {key: handle[f"diagnostics/{key}_objective_fraction_after"][...]
                     for key in ("divergence", "prior")}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].loglog(frequency, component_asd, "o-", label="Tabulated component ASD")
    positive = processed_asd > 0
    axes[0, 0].loglog(grid[positive], processed_asd[positive], label="Integrated ASD × |H|")
    axes[0, 0].set(xlabel="Frequency [Hz]", ylabel="ASD [nT/√Hz]",
                   title=f"Per-component σB = {sigma:.5g} nT")
    axes[0, 0].legend()
    valid = np.isfinite(ideal_weights) & (ideal_weights > 0) & (instrument_weights > 0)
    axes[0, 1].loglog(ideal_weights[valid], instrument_weights[valid], ".")
    if np.any(valid):
        limits = [min(ideal_weights[valid].min(), instrument_weights[valid].min()),
                  max(ideal_weights[valid].max(), instrument_weights[valid].max())]
        axes[0, 1].plot(limits, limits, "k--", label="Equal weight")
    axes[0, 1].set(xlabel="Ideal-divergence weight [(km/nT)²]",
                   ylabel="Instrument weight [(km/nT)²]", title="Median over time when geometry changes")
    axes[1, 0].plot(np.flatnonzero(valid), instrument_weights[valid]/ideal_weights[valid], ".")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set(xlabel="Retained polyhedron index", ylabel="Instrument / ideal weight")
    for key, values in fractions.items():
        axes[1, 1].plot(np.arange(len(values)), values, "o-", label=key)
    axes[1, 1].set(xlabel="MC realization", ylabel="Fraction of corrected objective")
    axes[1, 1].legend()
    for ax in axes.flat:
        ax.grid(alpha=.2)
    base = Path(output_path)
    output = base.with_name(f"{base.stem}_noise_weights{base.suffix}")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output, dpi=dpi)
    finally:
        plt.close(fig)
    return [output]
