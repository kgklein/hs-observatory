from pathlib import Path

import h5py
import numpy as np
import pytest

from hs_observatory.analyze import analyze_from_config
from hs_observatory.analysis.registry import list_analyses
from hs_observatory.analysis.timeseries_fft import (
    TimeseriesFFTAnalysis,
    _common_positive_frequency_limits,
    _matched_truth_power_limits,
    _one_sided_psd,
    compute_timeseries_fft,
    plot_timeseries_fft,
)
from hs_observatory.cli import analyze_main


def _write_fft_input(
    path: Path,
    *,
    n_time: int = 128,
    n_spacecraft: int = 9,
    n_mc: int = 2,
    dt: float = 0.125,
    nonuniform: bool = False,
) -> tuple[Path, float]:
    time = np.arange(n_time, dtype=float) * dt
    if nonuniform:
        time[n_time // 2] += 0.02 * dt
    frequency = 8.0 / (n_time * dt)
    phase = 2.0 * np.pi * frequency * time[:, None]
    amplitudes = 1.0 + 0.1 * np.arange(n_spacecraft)[None, :]
    density = amplitudes * np.sin(phase)
    velocity = np.stack((0.5 * density, -density, 0.25 * density), axis=-1)
    magnetic = np.stack((density, np.zeros_like(density), np.zeros_like(density)), axis=-1)
    truth = {"n": density, "v": velocity, "B": magnetic}

    with h5py.File(path, "w") as handle:
        handle.create_dataset("time", data=time)
        spacecraft = handle.create_group("spacecraft")
        spacecraft.create_dataset(
            "id",
            data=np.asarray(
                ["H", *[f"N{index}" for index in range(1, n_spacecraft)]],
                dtype=h5py.string_dtype("utf-8"),
            ),
        )
        truth_group = handle.create_group("truth")
        observed_group = handle.create_group("observed")
        for variable, values in truth.items():
            truth_group.create_dataset(variable, data=values)
            ensemble = np.stack([(mc_index + 1.0) * values for mc_index in range(n_mc)], axis=0)
            observed_group.create_dataset(variable, data=ensemble)
    return path, frequency


def _config(**overrides):
    config = {
        "data_group": "truth",
        "mc_index": 0,
        "variables": ["n", "v", "B"],
        "detrend": "mean",
        "window": "hann",
        "normalization": "psd",
        "plot": {"enabled": False},
    }
    config.update(overrides)
    return config


def test_known_scalar_sine_has_peak_at_expected_frequency(tmp_path):
    input_path, expected_frequency = _write_fft_input(tmp_path / "sine.h5")

    result = compute_timeseries_fft(input_path, _config(variables=["n"]))

    frequency = result["frequency"]
    spectrum = result["psd"]["n"][0]
    peak_index = 1 + np.argmax(spectrum[1:])
    assert frequency[peak_index] == pytest.approx(expected_frequency)


def test_vector_trace_matches_components_and_peaks_at_sine_frequency(tmp_path):
    input_path, expected_frequency = _write_fft_input(tmp_path / "vector.h5")

    result = compute_timeseries_fft(input_path, _config(variables=["B"]))

    components = result["psd"]["B_components"]
    trace = result["psd"]["B_trace"]
    np.testing.assert_allclose(trace, np.sum(components, axis=-1))
    peak_index = 1 + np.argmax(trace[0, 1:])
    assert result["frequency"][peak_index] == pytest.approx(expected_frequency)


@pytest.mark.parametrize("n_time", [127, 128])
def test_one_sided_psd_preserves_power_for_odd_and_even_lengths(n_time):
    dt = 0.2
    sample = np.arange(n_time)
    signal = np.sin(2.0 * np.pi * 8.0 * sample / n_time)

    psd = _one_sided_psd(signal, dt, detrend="none", window="none")
    frequency_spacing = 1.0 / (n_time * dt)

    assert np.sum(psd) * frequency_spacing == pytest.approx(np.mean(signal**2))


def test_plot_frequency_limits_span_all_native_cadences():
    density_frequency = np.array([0.0, 0.1, 1.0, 4.0])
    magnetic_frequency = np.array([0.0, 0.05, 8.0, 16.0])

    assert _common_positive_frequency_limits([density_frequency, magnetic_frequency]) == (
        0.05,
        16.0,
    )


def test_output_hdf5_shapes_and_metadata(tmp_path):
    input_path, _ = _write_fft_input(tmp_path / "input.h5")
    output_path = tmp_path / "fft.h5"
    method = TimeseriesFFTAnalysis(_config(data_group="observed", mc_index=1))

    method.write_result(method.run(input_path), output_path)

    with h5py.File(output_path) as handle:
        assert handle["frequency"].shape == (65,)
        assert handle["psd/n"].shape == (9, 65)
        assert handle["psd/v_trace"].shape == (9, 65)
        assert handle["psd/B_trace"].shape == (9, 65)
        assert handle["psd/v_components"].shape == (9, 65, 3)
        assert handle["psd/B_components"].shape == (9, 65, 3)
        assert handle["spacecraft/id"].shape == (9,)
        assert handle["metadata"].attrs["data_group"] == "observed"
        assert handle["metadata"].attrs["mc_index"] == 1
        assert handle["metadata"].attrs["frequency_layout"] == "shared"


def test_truth_and_observed_mc_selection_produce_different_spectra(tmp_path):
    input_path, _ = _write_fft_input(tmp_path / "selection.h5")

    truth = compute_timeseries_fft(input_path, _config(variables=["n"], data_group="truth"))
    observed = compute_timeseries_fft(
        input_path,
        _config(variables=["n"], data_group="observed", mc_index=1),
    )

    np.testing.assert_allclose(observed["psd"]["n"], 4.0 * truth["psd"]["n"])
    assert truth["metadata"]["data_group"] == "truth"
    assert observed["metadata"]["data_group"] == "observed"
    assert observed["metadata"]["mc_index"] == 1


def test_nonuniform_cadence_and_ensemble_selection_errors(tmp_path):
    input_path, _ = _write_fft_input(tmp_path / "nonuniform.h5", nonuniform=True)

    with pytest.raises(ValueError, match="uniformly sampled"):
        compute_timeseries_fft(input_path, _config(variables=["n"]))
    with pytest.raises(NotImplementedError, match="ensemble FFT statistics"):
        compute_timeseries_fft(
            input_path,
            _config(variables=["n"], data_group="observed", mc_index="all"),
        )


def test_plot_and_cli_smoke(tmp_path, capsys):
    input_path, _ = _write_fft_input(tmp_path / "plot-input.h5")
    output_path = tmp_path / "fft-output.h5"
    plot_path = tmp_path / "fft-plot.png"
    truth_plot_path = tmp_path / "fft-truth-plot.png"
    config_path = tmp_path / "fft.yaml"
    config_path.write_text(
        f"""
analysis:
  method: timeseries_fft
input:
  path: {input_path}
output:
  path: {output_path}
fft:
  data_group: observed
  mc_index: 0
  variables: [n, v, B]
  detrend: mean
  window: hann
  normalization: psd
  plot:
    enabled: true
    path: {plot_path}
    truth_enabled: true
    truth_path: {truth_plot_path}
    loglog: true
    overlay_spacecraft: true
    show_components: false
""",
        encoding="utf-8",
    )

    assert "timeseries_fft" in list_analyses()
    assert analyze_from_config(config_path) == output_path
    assert output_path.is_file()
    assert plot_path.is_file() and plot_path.stat().st_size > 0
    assert truth_plot_path.is_file() and truth_plot_path.stat().st_size > 0
    with h5py.File(output_path) as handle:
        assert "truth_psd/n" in handle
        assert handle["metadata"].attrs["truth_plot_enabled"]
        assert handle["metadata"].attrs["truth_plot_path"] == str(truth_plot_path)
        matched_limits = _matched_truth_power_limits(handle)
        assert set(matched_limits) == {"n", "v", "B"}
        assert all(lower > 0.0 and upper > lower for lower, upper in matched_limits.values())

    regenerated = tmp_path / "regenerated.png"
    assert plot_timeseries_fft(output_path, regenerated) == regenerated
    assert regenerated.stat().st_size > 0
    regenerated_truth = tmp_path / "regenerated-truth.png"
    assert (
        plot_timeseries_fft(
            output_path,
            regenerated_truth,
            psd_group="truth_psd",
            title="Truth spectra",
        )
        == regenerated_truth
    )
    assert regenerated_truth.stat().st_size > 0

    with h5py.File(output_path, "r+") as handle:
        for dataset in handle["truth_psd"].values():
            dataset[...] = 0.0
        assert _matched_truth_power_limits(handle) == {}
    zero_truth_plot = tmp_path / "zero-truth.png"
    plot_timeseries_fft(
        output_path,
        zero_truth_plot,
        psd_group="truth_psd",
        title="Constant truth spectra",
    )
    assert zero_truth_plot.stat().st_size > 0

    assert analyze_main([str(config_path)]) == 0
    assert str(output_path) in capsys.readouterr().out


def test_native_cadences_use_per_variable_frequency_datasets(tmp_path):
    path = tmp_path / "native.h5"
    n_spacecraft = 3
    with h5py.File(path, "w") as handle:
        time_group = handle.create_group("time")
        truth = handle.create_group("truth")
        spacecraft = handle.create_group("spacecraft")
        spacecraft.create_dataset(
            "id",
            data=np.asarray(["H", "N1", "N2"], dtype=h5py.string_dtype("utf-8")),
        )
        for variable, n_time, dt in (("n", 64, 0.125), ("B", 128, 0.03125)):
            time = np.arange(n_time) * dt
            time_group.create_dataset(variable, data=time)
            scalar = np.sin(2.0 * np.pi * time)[:, None]
            values = np.broadcast_to(scalar, (n_time, n_spacecraft))
            if variable == "B":
                values = np.stack((values, values, values), axis=-1)
            truth.create_dataset(variable, data=values)

    output = tmp_path / "native-fft.h5"
    method = TimeseriesFFTAnalysis(_config(variables=["n", "B"]))
    method.write_result(method.run(path), output)

    with h5py.File(output) as handle:
        assert isinstance(handle["frequency"], h5py.Group)
        assert handle["frequency/n"].shape == (33,)
        assert handle["frequency/B"].shape == (65,)
        assert handle["metadata"].attrs["frequency_layout"] == "per_variable"


def test_observed_fft_truth_plot_uses_matching_spacecraft_velocity_frame(tmp_path):
    input_path, _ = _write_fft_input(tmp_path / "frames.h5")
    with h5py.File(input_path, "r+") as handle:
        metadata = handle.create_group("metadata")
        metadata.attrs["velocity_observed_frame"] = "spacecraft"
        plasma_velocity = handle["truth/v"][...]
        spacecraft_velocity = 3.0 * plasma_velocity
        handle["truth"].create_dataset("v_spacecraft_frame", data=spacecraft_velocity)
        handle["observed/v"][0] = spacecraft_velocity

    result = compute_timeseries_fft(
        input_path,
        _config(
            variables=["v"],
            data_group="observed",
            plot={"enabled": True, "truth_enabled": True},
        ),
    )

    np.testing.assert_allclose(result["truth_psd"]["v_trace"], result["psd"]["v_trace"])
