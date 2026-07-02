from pathlib import Path

import h5py
import numpy as np
import pytest

from hs_observatory.analyze import analyze_from_config
from hs_observatory.analysis.derived.elsasser import (
    MU0_SI,
    PROTON_MASS_KG,
    compute_density_normalization,
    compute_elsasser,
    magnetic_field_to_alfven_units,
)
from hs_observatory.analysis.elsasser import (
    ElsasserAnalysis,
    compute_elsasser_analysis,
)
from hs_observatory.analysis.registry import list_analyses
from hs_observatory.cli import analyze_main


def test_magnetic_field_to_alfven_units_matches_si_definition():
    B = np.array([[[5.0, -2.0, 1.0]]])
    n = np.array([[4.0]])

    actual = magnetic_field_to_alfven_units(B, n)
    expected = B * 1.0e-9 / np.sqrt(MU0_SI * n[..., None] * 1.0e6 * PROTON_MASS_KG) / 1000.0

    np.testing.assert_allclose(actual, expected)


def test_density_normalization_methods_preserve_spacecraft_axis():
    n = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])

    np.testing.assert_array_equal(compute_density_normalization(n, "instantaneous"), n)
    np.testing.assert_array_equal(
        compute_density_normalization(n, "global_per_spacecraft"), [3.0, 20.0]
    )


def test_mean_subtraction_and_elsasser_sign_convention():
    v = np.array([[[1.0, 2.0, 3.0]], [[3.0, 4.0, 5.0]]])
    b = np.array([[[10.0, 20.0, 30.0]], [[30.0, 40.0, 50.0]]])

    result = compute_elsasser(v, b)

    np.testing.assert_allclose(result["delta_v"], v - np.mean(v, axis=0))
    np.testing.assert_allclose(result["delta_b"], b - np.mean(b, axis=0))
    np.testing.assert_allclose(result["z_plus"], result["delta_v"] + result["delta_b"])
    np.testing.assert_allclose(result["z_minus"], result["delta_v"] - result["delta_b"])


def test_reusable_utility_shapes():
    rng = np.random.default_rng(4)
    n = 4.0 + rng.random((16, 9))
    v = rng.normal(size=(16, 9, 3))
    B = rng.normal(size=(16, 9, 3))
    normalization = compute_density_normalization(n)
    b = magnetic_field_to_alfven_units(B, normalization)

    products = compute_elsasser(v, b)

    assert b.shape == (16, 9, 3)
    for name in ("z_plus", "z_minus", "delta_v", "delta_b"):
        assert products[name].shape == (16, 9, 3)


def _write_input(
    path: Path,
    *,
    variable_times: bool = True,
    observed_frame: str = "plasma",
    spacecraft_velocity: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_spacecraft = 2
    time_nv = np.arange(8, dtype=float) * 0.125
    time_B = np.arange(29, dtype=float) * 0.03125
    n = 5.0 + time_nv[:, None] * np.array([[1.0, 2.0]])
    v = np.empty((time_nv.size, n_spacecraft, 3))
    v[..., 0] = time_nv[:, None] + np.array([[0.0, 1.0]])
    v[..., 1] = 2.0 * time_nv[:, None]
    v[..., 2] = -time_nv[:, None]
    B = np.empty((time_B.size, n_spacecraft, 3))
    B[..., 0] = 2.0 + 4.0 * time_B[:, None]
    B[..., 1] = -3.0 * time_B[:, None]
    B[..., 2] = np.array([[1.0, 2.0]])
    observed_v = v.copy()
    if observed_frame == "spacecraft":
        assert spacecraft_velocity is not None
        observed_v -= spacecraft_velocity

    with h5py.File(path, "w") as handle:
        if variable_times:
            times = handle.create_group("time")
            times.create_dataset("n", data=time_nv)
            times.create_dataset("v", data=time_nv)
            times.create_dataset("B", data=time_B)
        else:
            handle.create_dataset("time", data=time_nv)
            B = B[::4]
        metadata = handle.create_group("metadata")
        metadata.attrs["velocity_observed_frame"] = observed_frame
        spacecraft = handle.create_group("spacecraft")
        spacecraft.create_dataset(
            "id", data=np.asarray(["H", "N1"], dtype=h5py.string_dtype("utf-8"))
        )
        if spacecraft_velocity is not None:
            spacecraft.create_dataset("velocity", data=spacecraft_velocity)
        truth = handle.create_group("truth")
        observed = handle.create_group("observed")
        for name, truth_values, observed_values in (
            ("n", n, n),
            ("v", v, observed_v),
            ("B", B, B),
        ):
            truth.create_dataset(name, data=truth_values)
            observed.create_dataset(
                name,
                data=np.stack((observed_values, 2.0 * observed_values), axis=0),
            )
    return time_nv, v, B


def _config(**overrides):
    config = {
        "data_group": "truth",
        "density_normalization": {"method": "global_per_spacecraft"},
        "mean_subtraction": {"method": "global_per_spacecraft"},
        "timebase": {
            "method": "slower_of_n_and_v",
            "interpolation": "linear",
        },
        "plot": {"enabled": False},
    }
    config.update(overrides)
    return config


def test_truth_analysis_smoke_and_multicadence_interpolation(tmp_path):
    input_path = tmp_path / "truth.h5"
    time, _v, B = _write_input(input_path)

    result = compute_elsasser_analysis(input_path, _config())
    output = tmp_path / "elsasser.h5"
    ElsasserAnalysis(_config()).write_result(result, output)

    np.testing.assert_array_equal(result["time"], time)
    np.testing.assert_allclose(
        result["b"][:, :, 0] / result["b"][:, :, 2], B[::4, :, 0] / B[::4, :, 2]
    )
    with h5py.File(output) as handle:
        for path in (
            "elsasser/z_plus",
            "elsasser/z_minus",
            "alfven/b",
            "alfven/delta_b",
            "velocity/v_plasma",
            "velocity/delta_v",
            "density/n_normalization",
        ):
            assert path in handle
        assert handle["elsasser/z_plus"].shape == (8, 2, 3)
        assert handle["density/n_normalization"].shape == (2,)
        assert handle["metadata"].attrs["analysis_method"] == "elsasser"
        assert handle["metadata"].attrs["z_plus_convention"] == "delta_v + delta_b"


def test_observed_mc_selection_uses_requested_realization(tmp_path):
    input_path = tmp_path / "observed.h5"
    _write_input(input_path)
    config = _config(
        data_group="observed",
        mc_index=1,
        velocity={"frame": "auto", "spacecraft_correction": "auto"},
    )

    result = compute_elsasser_analysis(input_path, config)

    with h5py.File(input_path) as handle:
        np.testing.assert_allclose(result["n"], handle["observed/n"][1])
        np.testing.assert_allclose(result["v_plasma"], handle["observed/v"][1])
    assert result["metadata"]["mc_index"] == 1


def test_spacecraft_frame_velocity_is_corrected_before_fluctuations(tmp_path):
    input_path = tmp_path / "spacecraft-frame.h5"
    velocity = np.broadcast_to(np.array([100.0, -20.0, 7.0]), (8, 2, 3)).copy()
    _time, plasma, _B = _write_input(
        input_path,
        observed_frame="spacecraft",
        spacecraft_velocity=velocity,
    )
    config = _config(
        data_group="observed",
        mc_index=0,
        velocity={"frame": "auto", "spacecraft_correction": "auto"},
    )

    result = compute_elsasser_analysis(input_path, config)

    np.testing.assert_allclose(result["v_plasma"], plasma)
    np.testing.assert_allclose(result["delta_v"], plasma - np.mean(plasma, axis=0, keepdims=True))
    assert result["metadata"]["velocity_frame_input"] == "spacecraft"


def test_legacy_shared_time_layout_is_supported(tmp_path):
    input_path = tmp_path / "legacy.h5"
    time, _v, _B = _write_input(input_path, variable_times=False)

    result = compute_elsasser_analysis(input_path, _config())

    np.testing.assert_array_equal(result["time"], time)


def test_plot_registry_and_cli_smoke(tmp_path, capsys):
    input_path = tmp_path / "input.h5"
    output_path = tmp_path / "output.h5"
    plot_path = tmp_path / "plot.png"
    config_path = tmp_path / "elsasser.yaml"
    _write_input(input_path)
    config_path.write_text(
        f"""
analysis:
  method: elsasser
input:
  path: {input_path}
output:
  path: {output_path}
elsasser:
  data_group: truth
  density_normalization:
    method: global_per_spacecraft
  plot:
    enabled: true
    path: {plot_path}
""",
        encoding="utf-8",
    )

    assert "elsasser" in list_analyses()
    assert analyze_from_config(config_path) == output_path
    assert plot_path.is_file() and plot_path.stat().st_size > 0
    assert analyze_main([str(config_path)]) == 0
    assert str(output_path) in capsys.readouterr().out


def test_all_mc_and_ambiguous_observed_frame_fail_clearly(tmp_path):
    input_path = tmp_path / "input.h5"
    _write_input(input_path)
    with pytest.raises(NotImplementedError, match="select one observed realization"):
        compute_elsasser_analysis(input_path, _config(data_group="observed", mc_index="all"))
    with h5py.File(input_path, "r+") as handle:
        del handle["metadata"].attrs["velocity_observed_frame"]
    with pytest.raises(ValueError, match="set elsasser.velocity.frame explicitly"):
        compute_elsasser_analysis(
            input_path,
            _config(data_group="observed", velocity={"frame": "auto"}),
        )
