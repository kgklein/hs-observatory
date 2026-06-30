"""Single-spacecraft, one-sided FFT power spectra for observation time series."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from hs_observatory.analysis.base import AnalysisMethod
from hs_observatory.analysis.registry import register_analysis
from hs_observatory.visualization import SPACECRAFT_COLORS

VARIABLES = ("n", "v", "B")


def _settings(config: dict[str, Any]) -> dict[str, Any]:
    variables = list(config.get("variables", VARIABLES))
    if not variables or any(variable not in VARIABLES for variable in variables):
        raise ValueError("fft.variables must contain one or more of: n, v, B")
    data_group = str(config.get("data_group", "observed"))
    if data_group not in {"observed", "truth"}:
        raise ValueError("fft.data_group must be 'observed' or 'truth'")
    mc_index = config.get("mc_index", 0)
    if mc_index == "all":
        raise NotImplementedError(
            "mc_index: all is not implemented; ensemble FFT statistics can be added later"
        )
    mc_index = int(mc_index)
    if mc_index < 0:
        raise ValueError("fft.mc_index must be non-negative")

    detrend = str(config.get("detrend", "mean")).lower()
    if detrend not in {"mean", "none"}:
        raise ValueError("fft.detrend must be 'mean' or 'none'")
    window = str(config.get("window", "hann")).lower()
    if window not in {"hann", "none"}:
        raise ValueError("fft.window must be 'hann' or 'none'")
    normalization = str(config.get("normalization", "psd")).lower()
    if normalization != "psd":
        raise ValueError("Only fft.normalization: psd is currently supported")

    plot = config.get("plot", {})
    if not isinstance(plot, dict):
        raise ValueError("fft.plot must be a mapping")
    dpi = int(plot.get("dpi", 150))
    if dpi <= 0:
        raise ValueError("fft.plot.dpi must be positive")
    return {
        "variables": variables,
        "data_group": data_group,
        "mc_index": mc_index,
        "detrend": detrend,
        "window": window,
        "normalization": normalization,
        "plot": {
            "enabled": bool(plot.get("enabled", True)),
            "path": plot.get("path"),
            "truth_enabled": bool(plot.get("truth_enabled", True)),
            "truth_path": plot.get("truth_path"),
            "loglog": bool(plot.get("loglog", True)),
            "overlay_spacecraft": bool(plot.get("overlay_spacecraft", True)),
            "show_components": bool(plot.get("show_components", False)),
            "dpi": dpi,
        },
    }


def _read_time(handle: h5py.File, variable: str) -> np.ndarray:
    if "time" not in handle:
        raise KeyError("Input file is missing /time")
    time_node = handle["time"]
    if isinstance(time_node, h5py.Group):
        if variable not in time_node:
            raise KeyError(f"Input file is missing /time/{variable}")
        return np.asarray(time_node[variable], dtype=float)
    return np.asarray(time_node, dtype=float)


def _uniform_cadence(time: np.ndarray, variable: str) -> float:
    if time.ndim != 1 or time.size < 2:
        raise ValueError(f"Time for {variable} must be one-dimensional with at least two samples")
    if not np.all(np.isfinite(time)):
        raise ValueError(f"Time for {variable} contains non-finite values")
    differences = np.diff(time)
    dt = float(np.mean(differences))
    tolerance = max(1.0e-12, abs(dt) * 1.0e-8)
    if dt <= 0.0 or not np.allclose(differences, dt, rtol=1.0e-7, atol=tolerance):
        raise ValueError(
            f"Time for {variable} must be uniformly sampled with a positive cadence"
        )
    return dt


def _read_values(
    handle: h5py.File,
    variable: str,
    settings: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, float]:
    dataset_name = variable
    if (
        settings["data_group"] == "truth"
        and variable == "v"
        and settings.get("velocity_truth_frame") == "spacecraft"
    ):
        dataset_name = "v_spacecraft_frame"
    path = f"{settings['data_group']}/{dataset_name}"
    if path not in handle:
        raise KeyError(f"Requested variable '{variable}' is missing from /{settings['data_group']}")
    values = np.asarray(handle[path], dtype=float)
    if settings["data_group"] == "observed":
        if values.ndim < 3:
            raise ValueError(f"/{path} does not contain a leading Monte Carlo dimension")
        if settings["mc_index"] >= values.shape[0]:
            raise IndexError(
                f"mc_index {settings['mc_index']} is outside /{path} with "
                f"{values.shape[0]} realizations"
            )
        values = values[settings["mc_index"]]

    time = _read_time(handle, variable)
    expected_dimensions = 2 if variable == "n" else 3
    if values.ndim != expected_dimensions:
        expected = "(time, spacecraft)" if variable == "n" else "(time, spacecraft, 3)"
        raise ValueError(f"/{path} must have shape {expected} after data selection")
    if values.shape[0] != time.size:
        raise ValueError(
            f"/{path} has {values.shape[0]} time samples but its time axis has {time.size}"
        )
    if variable != "n" and values.shape[-1] != 3:
        raise ValueError(f"/{path} must have exactly three vector components")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"/{path} contains non-finite values")
    return values, time, _uniform_cadence(time, variable)


def _window_values(name: str, n_time: int) -> np.ndarray:
    if name == "none":
        return np.ones(n_time, dtype=float)
    window = np.hanning(n_time)
    if not np.any(window):
        raise ValueError("The Hann window requires at least three time samples")
    return window


def _one_sided_psd(
    values: np.ndarray,
    dt: float,
    *,
    detrend: str,
    window: str,
) -> np.ndarray:
    """Return PSD with frequency first using ``dt/sum(w**2) * |rFFT|**2``."""

    working = np.asarray(values, dtype=float)
    if detrend == "mean":
        working = working - np.mean(working, axis=0, keepdims=True)
    weights = _window_values(window, working.shape[0])
    reshape = (weights.size,) + (1,) * (working.ndim - 1)
    transform = np.fft.rfft(working * weights.reshape(reshape), axis=0)
    psd = dt / float(np.sum(weights**2)) * np.abs(transform) ** 2
    if working.shape[0] % 2 == 0:
        psd[1:-1] *= 2.0
    else:
        psd[1:] *= 2.0
    return psd


def _psd_products(
    variable: str,
    values: np.ndarray,
    dt: float,
    settings: dict[str, Any],
) -> dict[str, np.ndarray]:
    psd = _one_sided_psd(
        values,
        dt,
        detrend=settings["detrend"],
        window=settings["window"],
    )
    if variable == "n":
        return {"n": np.transpose(psd, (1, 0))}
    components = np.transpose(psd, (1, 0, 2))
    return {
        f"{variable}_components": components,
        f"{variable}_trace": np.sum(components, axis=-1),
    }


def compute_timeseries_fft(
    input_path: str | Path, config: dict[str, Any]
) -> dict[str, Any]:
    """Compute independent spacecraft spectra for selected observation variables."""

    settings = _settings(config)
    frequencies: dict[str, np.ndarray] = {}
    spectra: dict[str, np.ndarray] = {}
    truth_spectra: dict[str, np.ndarray] = {}
    cadences: dict[str, float] = {}
    sample_counts: dict[str, int] = {}

    with h5py.File(input_path, "r") as handle:
        spacecraft_ids = None
        if "spacecraft/id" in handle:
            spacecraft_ids = [
                item.decode("utf-8") if isinstance(item, bytes) else str(item)
                for item in handle["spacecraft/id"][...]
            ]
        n_spacecraft = None
        for variable in settings["variables"]:
            values, time, dt = _read_values(handle, variable, settings)
            if n_spacecraft is None:
                n_spacecraft = values.shape[1]
            elif values.shape[1] != n_spacecraft:
                raise ValueError("All requested variables must use the same spacecraft count")
            frequencies[variable] = np.fft.rfftfreq(time.size, d=dt)
            cadences[variable] = dt
            sample_counts[variable] = time.size
            spectra.update(_psd_products(variable, values, dt, settings))

        compute_truth = (
            settings["data_group"] == "observed"
            and settings["plot"]["enabled"]
            and settings["plot"]["truth_enabled"]
        )
        if compute_truth:
            truth_settings = dict(settings)
            truth_settings["data_group"] = "truth"
            observed_velocity_frame = "plasma"
            if "metadata" in handle:
                observed_velocity_frame = str(
                    handle["metadata"].attrs.get(
                        "velocity_observed_frame", "plasma"
                    )
                )
            if (
                observed_velocity_frame == "spacecraft"
                and "truth/v_spacecraft_frame" in handle
            ):
                truth_settings["velocity_truth_frame"] = "spacecraft"
            for variable in settings["variables"]:
                values, time, dt = _read_values(handle, variable, truth_settings)
                truth_frequency = np.fft.rfftfreq(time.size, d=dt)
                if not np.array_equal(truth_frequency, frequencies[variable]):
                    raise ValueError(
                        f"Truth and observed frequency grids differ for {variable}"
                    )
                truth_spectra.update(
                    _psd_products(variable, values, dt, settings)
                )

    assert n_spacecraft is not None
    if spacecraft_ids is None:
        spacecraft_ids = [f"S{index}" for index in range(n_spacecraft)]
    if len(spacecraft_ids) != n_spacecraft:
        raise ValueError("/spacecraft/id length does not match the selected variables")

    first_frequency = frequencies[settings["variables"][0]]
    shared_frequency = all(
        frequency.shape == first_frequency.shape
        and np.array_equal(frequency, first_frequency)
        for frequency in frequencies.values()
    )
    scalar_dt: float | dict[str, float] = (
        next(iter(cadences.values())) if shared_frequency else cadences
    )
    scalar_n_time: int | dict[str, int] = (
        next(iter(sample_counts.values())) if shared_frequency else sample_counts
    )
    return {
        "metadata": {
            "method": "timeseries_fft",
            "input_path": str(Path(input_path)),
            "data_group": settings["data_group"],
            "mc_index": settings["mc_index"],
            "variables": settings["variables"],
            "dt": scalar_dt,
            "n_time": scalar_n_time,
            "window": settings["window"],
            "detrend": settings["detrend"],
            "normalization": settings["normalization"],
            "frequency_layout": "shared" if shared_frequency else "per_variable",
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
        "settings": settings,
        "frequency": first_frequency if shared_frequency else frequencies,
        "spacecraft_ids": spacecraft_ids,
        "psd": spectra,
        "truth_psd": truth_spectra,
    }


def _frequency_for_product(handle: h5py.File, variable: str) -> np.ndarray:
    node = handle["frequency"]
    if isinstance(node, h5py.Group):
        return np.asarray(node[variable], dtype=float)
    return np.asarray(node, dtype=float)


def _common_positive_frequency_limits(
    frequencies: list[np.ndarray],
) -> tuple[float, float]:
    positive = [frequency[frequency > 0.0] for frequency in frequencies]
    if not positive or any(frequency.size == 0 for frequency in positive):
        raise ValueError("FFT output has no positive frequencies to plot")
    return (
        min(float(np.min(frequency)) for frequency in positive),
        max(float(np.max(frequency)) for frequency in positive),
    )


def _matched_truth_power_limits(
    handle: h5py.File,
) -> dict[str, tuple[float, float]]:
    """Return shared observed/truth limits where truth has positive power."""

    if "psd" not in handle or "truth_psd" not in handle:
        return {}
    limits: dict[str, tuple[float, float]] = {}
    for variable, product in (("n", "n"), ("v", "v_trace"), ("B", "B_trace")):
        selected_path = f"psd/{product}"
        truth_path = f"truth_psd/{product}"
        if selected_path not in handle or truth_path not in handle:
            continue
        selected = np.asarray(handle[selected_path], dtype=float)
        truth = np.asarray(handle[truth_path], dtype=float)
        positive_truth = truth[np.isfinite(truth) & (truth > 0.0)]
        if positive_truth.size == 0:
            continue
        positive_selected = selected[np.isfinite(selected) & (selected > 0.0)]
        combined = (
            np.concatenate((positive_truth, positive_selected))
            if positive_selected.size
            else positive_truth
        )
        lower = float(np.min(combined))
        upper = float(np.max(combined))
        if lower == upper:
            lower /= 1.25
            upper *= 1.25
        else:
            log_padding = 0.05 * (np.log10(upper) - np.log10(lower))
            lower /= 10.0**log_padding
            upper *= 10.0**log_padding
        limits[variable] = (lower, upper)
    return limits


def plot_timeseries_fft(
    output_h5_path: str | Path,
    plot_path: str | Path,
    *,
    loglog: bool = True,
    overlay_spacecraft: bool = True,
    show_components: bool = False,
    dpi: int = 150,
    psd_group: str = "psd",
    title: str = "Single-Spacecraft Time-Series Power Spectra",
) -> Path:
    """Regenerate the spacecraft-overlay diagnostic from an FFT HDF5 product."""

    import matplotlib.pyplot as plt

    if not overlay_spacecraft:
        raise NotImplementedError(
            "Non-overlay FFT plotting is not implemented; set overlay_spacecraft: true"
        )
    with h5py.File(output_h5_path, "r") as handle:
        ids = [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in handle["spacecraft/id"][...]
        ]
        products = []
        if f"{psd_group}/n" in handle:
            products.append(("n", "n", r"$P_n$ [quantity$^2$/Hz]"))
        if f"{psd_group}/v_trace" in handle:
            products.append(("v", "v_trace", r"Trace $P_v$ [quantity$^2$/Hz]"))
        if f"{psd_group}/B_trace" in handle:
            products.append(("B", "B_trace", r"Trace $P_B$ [quantity$^2$/Hz]"))
        if not products:
            raise ValueError("FFT output contains no plottable spectra")

        figure, axes = plt.subplots(
            len(products),
            1,
            figsize=(9.5, 3.4 * len(products)),
            squeeze=False,
            sharex=True,
            constrained_layout=True,
        )
        frequency_limits = _common_positive_frequency_limits(
            [
                _frequency_for_product(handle, variable)
                for variable, _product, _ylabel in products
            ]
        )
        matched_power_limits = _matched_truth_power_limits(handle)
        fallback_colors = plt.cm.tab10(np.linspace(0.0, 0.9, len(ids)))
        for row, (variable, product, ylabel) in enumerate(products):
            axis = axes[row, 0]
            frequency = _frequency_for_product(handle, variable)
            trace = np.asarray(handle[f"{psd_group}/{product}"], dtype=float)
            positive_frequency = frequency > 0.0
            has_positive_power = False
            for index, spacecraft_id in enumerate(ids):
                color = SPACECRAFT_COLORS.get(spacecraft_id, fallback_colors[index])
                valid = positive_frequency & np.isfinite(trace[index]) & (trace[index] > 0.0)
                has_positive_power = has_positive_power or bool(np.any(valid))
                axis.plot(
                    frequency[valid],
                    trace[index, valid],
                    color=color,
                    linewidth=1.0,
                    label=spacecraft_id,
                )
                component_path = f"{psd_group}/{variable}_components"
                if show_components and component_path in handle:
                    components = np.asarray(handle[component_path][index], dtype=float)
                    for component, linestyle in enumerate((":", "--", "-.")):
                        valid_component = (
                            positive_frequency
                            & np.isfinite(components[:, component])
                            & (components[:, component] > 0.0)
                        )
                        axis.plot(
                            frequency[valid_component],
                            components[valid_component, component],
                            color=color,
                            linestyle=linestyle,
                            linewidth=0.65,
                            alpha=0.35,
                        )
            if loglog:
                axis.set_xscale("log")
                axis.set_yscale("log")
                if not has_positive_power:
                    axis.set_ylim(1.0e-12, 1.0)
                    axis.text(
                        0.5,
                        0.5,
                        "No positive spectral power",
                        ha="center",
                        va="center",
                        transform=axis.transAxes,
                        color="#666666",
                    )
            if variable in matched_power_limits:
                axis.set_ylim(matched_power_limits[variable])
            axis.set_xlim(frequency_limits)
            axis.set_ylabel(ylabel)
            axis.grid(True, which="both", alpha=0.22)
            if row == 0:
                axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
        axes[-1, 0].set_xlabel("Frequency [Hz]")
        figure.suptitle(title)

    output = Path(plot_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return output


def _write_metadata(group: h5py.Group, metadata: dict[str, Any]) -> None:
    for key, value in metadata.items():
        group.attrs[key] = (
            value
            if isinstance(value, (str, int, float, bool))
            else json.dumps(value, sort_keys=True)
        )


def _write_psd_group(
    handle: h5py.File,
    name: str,
    spectra: dict[str, np.ndarray],
) -> None:
    group = handle.create_group(name)
    for product, values in spectra.items():
        dataset = group.create_dataset(
            product,
            data=values,
            compression="gzip",
            compression_opts=4,
            chunks=True,
        )
        dataset.attrs["normalization"] = "one-sided PSD"
        dataset.attrs["units"] = "quantity^2/Hz"
        if product.endswith("_components"):
            dataset.attrs["components"] = np.asarray(
                ["x", "y", "z"], dtype=h5py.string_dtype("utf-8")
            )


@register_analysis
class TimeseriesFFTAnalysis(AnalysisMethod):
    """Analysis-registry adapter for independent spacecraft FFT spectra."""

    name = "timeseries_fft"
    config_section = "fft"

    def run(self, input_path: str | Path) -> dict[str, Any]:
        return compute_timeseries_fft(input_path, self.config)

    def write_result(self, result: dict[str, Any], output_path: str | Path) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        plot_settings = result["settings"]["plot"]
        plot_path = Path(
            plot_settings["path"]
            or output.with_name(f"{output.stem}.png")
        )
        truth_plot_enabled = bool(
            plot_settings["enabled"]
            and plot_settings["truth_enabled"]
            and result["metadata"]["data_group"] == "observed"
        )
        truth_plot_path = Path(
            plot_settings["truth_path"]
            or plot_path.with_name(f"{plot_path.stem}_truth{plot_path.suffix}")
        )
        result["metadata"]["plot_enabled"] = plot_settings["enabled"]
        result["metadata"]["plot_path"] = (
            str(plot_path) if plot_settings["enabled"] else "null"
        )
        result["metadata"]["truth_plot_enabled"] = truth_plot_enabled
        result["metadata"]["truth_plot_path"] = (
            str(truth_plot_path) if truth_plot_enabled else "null"
        )
        with h5py.File(output, "w") as handle:
            _write_metadata(handle.create_group("metadata"), result["metadata"])
            frequency = result["frequency"]
            if isinstance(frequency, dict):
                group = handle.create_group("frequency")
                for variable, values in frequency.items():
                    dataset = group.create_dataset(variable, data=values)
                    dataset.attrs["units"] = "Hz"
            else:
                dataset = handle.create_dataset("frequency", data=frequency)
                dataset.attrs["units"] = "Hz"
            spacecraft = handle.create_group("spacecraft")
            spacecraft.create_dataset(
                "id",
                data=np.asarray(
                    result["spacecraft_ids"], dtype=h5py.string_dtype("utf-8")
                ),
            )
            _write_psd_group(handle, "psd", result["psd"])
            if result["truth_psd"]:
                _write_psd_group(handle, "truth_psd", result["truth_psd"])
        if plot_settings["enabled"]:
            selected_label = result["metadata"]["data_group"]
            if selected_label == "observed":
                selected_label += f" MC {result['metadata']['mc_index']}"
            plot_timeseries_fft(
                output,
                plot_path,
                loglog=plot_settings["loglog"],
                overlay_spacecraft=plot_settings["overlay_spacecraft"],
                show_components=plot_settings["show_components"],
                dpi=plot_settings["dpi"],
                title=f"Single-Spacecraft Time-Series Power Spectra — {selected_label}",
            )
        if truth_plot_enabled:
            plot_timeseries_fft(
                output,
                truth_plot_path,
                loglog=plot_settings["loglog"],
                overlay_spacecraft=plot_settings["overlay_spacecraft"],
                show_components=plot_settings["show_components"],
                dpi=plot_settings["dpi"],
                psd_group="truth_psd",
                title="Single-Spacecraft Time-Series Power Spectra — truth",
            )
        return output
