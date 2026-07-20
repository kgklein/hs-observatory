from pathlib import Path
import inspect

import h5py
import numpy as np

from hs_observatory.analyze import analyze_from_config
from hs_observatory.analysis.registry import list_analyses
from hs_observatory.analysis.rbf_reconstruction import plot_rbf_reconstruction
from hs_observatory.analysis.timesync import (
    build_timesync_grid,
    compute_timesync_reconstruction,
    idw_interpolate_plane,
    reconstruct_timesync,
)
from visualization.export_reconstruction_to_paraview import export_reconstruction_to_paraview


def _write_timesync_input(
    path: Path,
    *,
    field: str = "constant",
    n_time: int = 6,
) -> Path:
    time = np.arange(n_time, dtype=float)
    offsets = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    positions = np.empty((4, n_time, 3), dtype=float)
    for spacecraft in range(4):
        positions[spacecraft, :, 0] = 10.0 * time
        positions[spacecraft, :, 1] = offsets[spacecraft, 0]
        positions[spacecraft, :, 2] = offsets[spacecraft, 1]
    positions_time_sc = np.swapaxes(positions, 0, 1)
    if field == "constant":
        magnetic = np.broadcast_to([1.0, -2.0, 5.0], positions_time_sc.shape).copy()
    elif field == "linear_x":
        magnetic = np.zeros_like(positions_time_sc)
        magnetic[..., 0] = positions_time_sc[..., 0]
        magnetic[..., 1] = 2.0
        magnetic[..., 2] = -1.0
    else:
        raise ValueError(field)
    with h5py.File(path, "w") as handle:
        time_group = handle.create_group("time")
        time_group.create_dataset("B", data=time)
        spacecraft = handle.create_group("spacecraft")
        spacecraft.create_dataset(
            "id",
            data=np.asarray(["H", "N1", "N2", "N3"], dtype=h5py.string_dtype("utf-8")),
        )
        r_group = spacecraft.create_group("r")
        r_group.create_dataset("B", data=positions)
        truth = handle.create_group("truth")
        truth.create_dataset("B", data=magnetic)
        observed = handle.create_group("observed")
        observed.create_dataset("B", data=magnetic[None, ...] + 0.25)
    return path


def _config(**overrides):
    config = {
        "data_group": "truth",
        "t0": 2.5,
        "delta_t": 5.0,
        "time_interp": "linear",
        "idw_power": 2.0,
        "axis_source": "barycenter_path",
        "min_spacecraft_per_plane": 4,
        "grid": {
            "radius": 1.0,
            "n_axial": 4,
            "n_transverse": 5,
            "axial_padding": 0.0,
        },
        "plot": {"enabled": False},
    }
    config.update(overrides)
    return config


def test_constant_field_reconstructs_everywhere_inside_cylinder(tmp_path):
    input_path = _write_timesync_input(tmp_path / "constant.h5", field="constant")

    result = compute_timesync_reconstruction(input_path, _config())

    np.testing.assert_allclose(
        result["B_reconstructed"],
        np.broadcast_to([1.0, -2.0, 5.0], result["B_reconstructed"].shape),
    )
    assert result["grid"].mask.shape == (4, 5, 5)
    assert result["B_reconstructed"].shape == (np.count_nonzero(result["grid"].mask), 3)


def test_linear_axial_field_uses_linear_synchronization(tmp_path):
    input_path = _write_timesync_input(tmp_path / "linear.h5", field="linear_x")

    result = compute_timesync_reconstruction(input_path, _config())

    np.testing.assert_allclose(result["B_reconstructed"][:, 0], result["grid"].points[:, 0])
    np.testing.assert_allclose(result["B_reconstructed"][:, 1], 2.0)
    np.testing.assert_allclose(result["B_reconstructed"][:, 2], -1.0)
    np.testing.assert_allclose(
        result["diagnostics"]["axial_mismatch"][
            np.isfinite(result["diagnostics"]["axial_mismatch"])
        ],
        0.0,
    )


def test_idw_returns_exact_spacecraft_values_at_matching_point():
    source_uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    source_B = np.array([[10.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 30.0]])
    target_uv = np.array([[1.0, 0.0], [0.5, 0.5]])

    result = idw_interpolate_plane(target_uv, source_uv, source_B, power=2.0)

    np.testing.assert_allclose(result[0], source_B[1])
    assert np.all(np.isfinite(result[1]))


def test_timesync_masked_planes_are_nan_when_too_few_spacecraft(tmp_path):
    input_path = _write_timesync_input(tmp_path / "masked.h5", field="constant")

    result = compute_timesync_reconstruction(
        input_path,
        _config(min_spacecraft_per_plane=5),
    )

    assert np.isnan(result["B_reconstructed"]).all()
    np.testing.assert_array_equal(
        result["diagnostics"]["n_spacecraft_used_per_plane"],
        np.full(4, 4),
    )


def test_timesync_defaults_to_seven_spacecraft_per_plane(tmp_path):
    input_path = _write_timesync_input(tmp_path / "default-threshold.h5", field="constant")
    config = _config()
    del config["min_spacecraft_per_plane"]

    result = compute_timesync_reconstruction(input_path, config)

    assert result["settings"]["min_spacecraft_per_plane"] == 7
    assert np.isnan(result["B_reconstructed"]).all()


def test_user_axis_grid_and_reconstruction_shape():
    time = np.arange(3.0)
    positions = np.zeros((3, 4, 3), dtype=float)
    positions[..., 1] = time[:, None]
    magnetic = np.broadcast_to([1.0, 2.0, 3.0], positions.shape).copy()

    grid = build_timesync_grid(
        positions,
        time,
        radius=1.0,
        n_axial=3,
        n_transverse=5,
        axis_source="user_vector",
        axis_vector=np.array([0.0, 1.0, 0.0]),
    )
    reconstructed, diagnostics = reconstruct_timesync(
        positions,
        magnetic,
        grid,
        min_spacecraft_per_plane=1,
    )

    np.testing.assert_allclose(grid.e_parallel, [0.0, 1.0, 0.0])
    assert reconstructed.shape == (np.count_nonzero(grid.mask), 3)
    assert diagnostics["nearest_or_bracketing_indices"].shape == (3, 4, 2)


def test_analysis_cli_writes_common_hdf5_and_paraview_can_read(tmp_path, monkeypatch):
    input_path = _write_timesync_input(tmp_path / "input.h5", field="constant")
    output_path = tmp_path / "timesync.h5"
    plot_path = tmp_path / "timesync_cli.png"
    averaged_error_path = tmp_path / "timesync_uv_error.png"
    config_path = tmp_path / "timesync.yaml"

    def fake_add_reference(reconstruction_h5, observation_path, generation_config):
        assert Path(observation_path) == input_path
        assert Path(generation_config).name == "generation.yaml"
        with h5py.File(reconstruction_h5, "r+") as handle:
            reconstructed = np.asarray(handle["reconstruction/B"], dtype=float)
            validation = handle.create_group("validation")
            validation.create_dataset("B_actual", data=reconstructed + 1.0)
            validation.create_dataset(
                "Bmag_actual", data=np.linalg.norm(reconstructed + 1.0, axis=-1)
            )
            validation.create_dataset("component_error", data=np.full_like(reconstructed, -1.0))
            validation.create_dataset(
                "error_magnitude", data=np.linalg.norm(np.ones_like(reconstructed), axis=-1)
            )
            validation.create_dataset(
                "error_percent_capped", data=np.full(reconstructed.shape[0], 10.0)
            )
        return Path(reconstruction_h5)

    monkeypatch.setattr(
        "hs_observatory.analysis.rbf_reconstruction.add_simulation_reference_to_rbf_output",
        fake_add_reference,
    )
    config_path.write_text(
        f"""
analysis:
  method: timesync
input:
  path: {input_path}
output:
  path: {output_path}
timesync:
  data_group: observed
  mc_index: 0
  t0_index: 2
  delta_t: 4.0
  time_interp: linear
  idw_power: 2.0
  min_spacecraft_per_plane: 4
  grid:
    radius: 1.0
    n_axial: 4
    n_transverse: 5
  plot:
    enabled: true
    path: {plot_path}
  averaged_error_plot:
    enabled: true
    path: {averaged_error_path}
  reference:
    enabled: true
    generation_config: {tmp_path / "generation.yaml"}
""",
        encoding="utf-8",
    )

    assert "timesync" in list_analyses()
    assert analyze_from_config(config_path) == output_path
    with h5py.File(output_path) as handle:
        assert handle["grid/full_points"].shape == (4, 5, 5, 3)
        np.testing.assert_array_equal(handle["grid/dimensions"][...], [4, 5, 5])
        assert handle["reconstruction/B"].shape[1] == 3
        assert handle["reconstruction/Bmag"].shape == (handle["reconstruction/B"].shape[0],)
        assert handle["timesync/nearest_or_bracketing_indices"].shape == (4, 4, 2)
        assert handle["timesync/interpolation_fraction"].shape == (4, 4)
        assert handle["timesync/n_spacecraft_used_per_plane"].shape == (4,)
        assert handle["validation/error_percent_capped"].shape == (
            handle["reconstruction/B"].shape[0],
        )
        assert handle["metadata"].attrs["method"] == "timesync"
        assert handle["metadata"].attrs["data_group"] == "observed"
        assert handle["metadata"].attrs["mc_index"] == 0

    products = export_reconstruction_to_paraview(output_path, tmp_path / "paraview")
    assert any(path.name == "reconstruction_structured.vts" for path in products)
    assert plot_path.is_file() and plot_path.stat().st_size > 0
    assert averaged_error_path.is_file() and averaged_error_path.stat().st_size > 0
    plot_path = tmp_path / "timesync_regenerated.png"
    assert plot_rbf_reconstruction(output_path, plot_path) == plot_path
    assert plot_path.stat().st_size > 0


def test_timesync_module_does_not_import_simulation_sampling_code():
    import hs_observatory.analysis.timesync as timesync

    source = inspect.getsource(timesync)
    assert "SimulationSampledField" not in source
    assert "simulation_sampled" not in source
