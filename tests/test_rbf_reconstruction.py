from pathlib import Path

import h5py
import numpy as np
import pytest

from hs_observatory.analyze import analyze_from_config
from hs_observatory.analysis.rbf_reconstruction import (
    ComponentwiseRBFReconstructor,
    add_simulation_reference_to_rbf_output,
    build_cylindrical_grid,
    choose_sigma_explicit_loocv,
    choose_sigma_rippa,
    collect_samples_for_window,
    compute_component_plot_field,
    compute_sigma_cache_key,
    compute_rbf_reconstruction,
    determine_background_field,
    get_or_compute_sigma,
    plot_rbf_broeren_uv_error,
    plot_rbf_reconstruction,
    plot_rippa_sigma_selection,
    plot_rbf_simulation_reference,
    explicit_loocv_cost,
    rippa_loocv_cost,
)
from hs_observatory.analysis.registry import list_analyses


def _linear_B(points: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (
            1.0 + 0.1 * points[:, 0],
            -2.0 + 0.2 * points[:, 1],
            0.5 + 0.3 * points[:, 2],
        )
    )


def _write_rbf_input(
    path: Path,
    *,
    n_time: int = 6,
    n_spacecraft: int = 4,
) -> Path:
    time = np.arange(n_time, dtype=float)
    positions = np.empty((n_spacecraft, n_time, 3), dtype=float)
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    for spacecraft in range(n_spacecraft):
        positions[spacecraft, :, :] = np.column_stack(
            (
                10.0 * time,
                np.full(n_time, offsets[spacecraft, 1]),
                np.full(n_time, offsets[spacecraft, 2]),
            )
        )
    magnetic = _linear_B(np.swapaxes(positions, 0, 1).reshape(-1, 3)).reshape(
        n_time, n_spacecraft, 3
    )
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
        observed.create_dataset(
            "B",
            data=np.stack(
                (
                    magnetic + np.array([100.0, 0.0, 0.0]),
                    magnetic + np.array([200.0, 0.0, 0.0]),
                ),
                axis=0,
            ),
        )
    return path


def _config(**overrides):
    config = {
        "data_group": "truth",
        "t0": 2.0,
        "delta_t": 2.0,
        "kernel": "multiquadric",
        "sigma": 10.0,
        "smoothing": 1.0e-10,
        "grid": {
            "radius": 1.5,
            "radius_padding": 0.0,
            "n_axial": 5,
            "n_transverse": 5,
            "axial_padding": 0.0,
        },
        "plot": {"enabled": False},
    }
    config.update(overrides)
    return config


def test_componentwise_rbf_evaluate_shape_and_known_points():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    magnetic = _linear_B(positions)

    model = ComponentwiseRBFReconstructor(sigma=1.0, smoothing=0.0).fit(positions, magnetic)
    reconstructed = model.evaluate(positions[:3])

    assert reconstructed.shape == (3, 3)
    np.testing.assert_allclose(reconstructed, magnetic[:3], atol=1.0e-10)


def test_componentwise_rbf_rejects_too_few_samples():
    with pytest.raises(ValueError, match="at least four samples"):
        ComponentwiseRBFReconstructor(sigma=1.0).fit(np.zeros((3, 3)), np.zeros((3, 3)))


def test_component_plot_field_raw_and_delta_modes():
    magnetic = np.array([[1.0, 2.0, 10.0], [2.0, 4.0, 12.0]])
    background = np.array([1.5, 3.0, 11.0])

    np.testing.assert_allclose(compute_component_plot_field(magnetic, mode="raw"), magnetic)
    np.testing.assert_allclose(
        compute_component_plot_field(magnetic, background, mode="delta"),
        [[-0.5, -1.0, -1.0], [0.5, 1.0, 1.0]],
    )


def test_determine_background_field_uses_metadata_then_means(tmp_path):
    path = tmp_path / "background.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs["B0"] = [0.1, 0.2, 5.0]
        reconstruction = handle.create_group("reconstruction")
        reconstruction.create_dataset("B", data=np.array([[1.0, 2.0, 10.0], [3.0, 4.0, 12.0]]))
        validation = handle.create_group("validation")
        validation.create_dataset("B_actual", data=np.array([[2.0, 4.0, 20.0], [4.0, 6.0, 22.0]]))

    with h5py.File(path) as handle:
        background, metadata = determine_background_field(handle, "reconstructed")
        np.testing.assert_allclose(background, [0.1, 0.2, 5.0])
        assert metadata["source"] == "metadata"
        background, metadata = determine_background_field(
            handle, "reconstructed", background_mode="actual-volume-mean"
        )
        np.testing.assert_allclose(background, [3.0, 5.0, 21.0])
        assert metadata["source"] == "actual-volume-mean"


def test_rippa_cost_is_finite_and_selection_returns_candidate():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    magnetic = _linear_B(positions)
    candidates = np.array([0.5, 1.0, 2.0])

    cost = rippa_loocv_cost(positions, magnetic[:, 0], 1.0)
    selected = choose_sigma_rippa(positions, magnetic, candidates)

    assert np.isfinite(cost)
    assert selected["selected_sigma"] in candidates
    assert selected["costs"].shape == candidates.shape


def test_explicit_loocv_cost_and_selection_are_finite():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    magnetic = _linear_B(positions)

    cost = explicit_loocv_cost(positions, magnetic, 1.0)
    selected = choose_sigma_explicit_loocv(
        positions,
        magnetic,
        sigma_initial=1.0,
        sigma_min=0.5,
        sigma_max=2.0,
        max_iter=5,
    )

    assert np.isfinite(cost)
    assert 0.5 <= selected["selected_sigma"] <= 2.0
    assert selected["method"] == "explicit_loocv_vector_norm"
    assert selected["costs"].shape == selected["sigma_candidates"].shape


def test_sigma_cache_reuses_and_keys_change(tmp_path):
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    magnetic = _linear_B(positions)
    candidates = np.array([0.5, 1.0, 2.0])
    cache_path = tmp_path / "sigma-cache.h5"

    first = get_or_compute_sigma(
        positions,
        magnetic,
        candidates,
        cache_path=cache_path,
        metadata={"sample_positions": positions, "n_spacecraft": 5},
    )
    second = get_or_compute_sigma(
        positions,
        magnetic,
        candidates,
        cache_path=cache_path,
        metadata={"sample_positions": positions, "n_spacecraft": 5},
    )
    forced = get_or_compute_sigma(
        positions,
        magnetic,
        candidates,
        cache_path=cache_path,
        force_recompute=True,
        metadata={"sample_positions": positions, "n_spacecraft": 5},
    )

    assert cache_path.is_file()
    assert first["source"] == "computed_rippa"
    assert second["source"] == "cache"
    assert forced["source"] == "computed_rippa"
    assert second["selected_sigma"] == first["selected_sigma"]

    base = compute_sigma_cache_key(positions, sigma_candidates=candidates)
    changed_smoothing = compute_sigma_cache_key(
        positions, smoothing=0.1, sigma_candidates=candidates
    )
    changed_grid = compute_sigma_cache_key(positions, sigma_candidates=np.array([0.5, 2.0]))
    changed_positions = compute_sigma_cache_key(
        positions + np.array([0.1, 0.0, 0.0]), sigma_candidates=candidates
    )
    assert base["cache_key"] != changed_smoothing["cache_key"]
    assert base["cache_key"] != changed_grid["cache_key"]
    assert base["cache_key"] != changed_positions["cache_key"]


def test_explicit_loocv_sigma_cache_reuses_and_records_source(tmp_path):
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    magnetic = _linear_B(positions)
    cache_path = tmp_path / "explicit-cache.h5"

    first = get_or_compute_sigma(
        positions,
        magnetic,
        None,
        cache_path=cache_path,
        sigma_selection="explicit_loocv",
        sigma_initial=1.0,
        sigma_min=0.5,
        sigma_max=2.0,
        metadata={"sample_positions": positions, "n_spacecraft": 5},
    )
    second = get_or_compute_sigma(
        positions,
        magnetic,
        None,
        cache_path=cache_path,
        sigma_selection="explicit_loocv",
        sigma_initial=1.0,
        sigma_min=0.5,
        sigma_max=2.0,
        metadata={"sample_positions": positions, "n_spacecraft": 5},
    )

    assert first["source"] == "computed_explicit_loocv"
    assert second["source"] == "cache"
    assert second["selected_sigma"] == first["selected_sigma"]


def test_cylindrical_grid_shape_mask_and_unwrapped_axis():
    time = np.array([0.0, 1.0, 2.0])
    positions = np.zeros((3, 2, 3), dtype=float)
    positions[:, :, 0] = np.array([0.0, 10.0, 20.0])[:, None]
    positions[:, 1, 0] += 5.0
    positions[:, 1, 1] = 1.0

    grid = build_cylindrical_grid(positions, time, radius=0.25, n_axial=4, n_transverse=5)

    assert grid.mask.shape == (4, 5, 5)
    assert grid.points.shape[1] == 3
    assert grid.points.shape[0] == int(np.count_nonzero(grid.mask))
    np.testing.assert_allclose(grid.e_parallel, [1.0, 0.0, 0.0])
    assert grid.requested_radius == 0.25
    assert grid.radius_padding == 0.0
    assert grid.radius >= 0.5
    np.testing.assert_allclose(grid.max_spacecraft_transverse_radius, 0.5)
    assert np.max(grid.points[:, 0]) >= np.max(positions[..., 0])


def test_cylindrical_grid_adds_radius_padding_beyond_spacecraft_distance():
    time = np.array([0.0, 1.0])
    positions = np.zeros((2, 2, 3), dtype=float)
    positions[:, 0, 0] = [0.0, 10.0]
    positions[:, 1, 0] = [0.0, 10.0]
    positions[:, 1, 1] = 500.0

    grid = build_cylindrical_grid(
        positions,
        time,
        radius=100.0,
        radius_padding=200.0,
        n_axial=3,
        n_transverse=5,
    )

    np.testing.assert_allclose(grid.max_spacecraft_transverse_radius, 250.0)
    np.testing.assert_allclose(grid.radius, 450.0)
    assert grid.requested_radius == 100.0
    assert grid.radius_padding == 200.0


def test_cylindrical_grid_can_require_all_spacecraft_axial_coverage():
    time = np.array([0.0, 1.0, 2.0])
    positions = np.zeros((3, 2, 3), dtype=float)
    positions[:, 0, 0] = np.array([0.0, 10.0, 20.0])
    positions[:, 1, 0] = np.array([5.0, 15.0, 25.0])

    grid = build_cylindrical_grid(
        positions,
        time,
        radius=1.0,
        n_axial=4,
        n_transverse=5,
        require_all_spacecraft_coverage=True,
    )

    np.testing.assert_allclose(grid.e_parallel, [1.0, 0.0, 0.0])
    assert grid.require_all_spacecraft_coverage
    np.testing.assert_allclose(grid.axial_values + grid.center[0], [5.0, 10.0, 15.0, 20.0])
    assert np.min(grid.points[:, 0]) >= 5.0
    assert np.max(grid.points[:, 0]) <= 20.0


def test_compute_rbf_reconstruction_collects_window_samples(tmp_path):
    input_path = _write_rbf_input(tmp_path / "input.h5")

    result = compute_rbf_reconstruction(input_path, _config(t0=2.0, delta_t=2.0))

    assert result["sample_positions"].shape == (12, 3)
    assert result["sample_trajectory"].shape == (3, 4, 3)
    assert result["sample_B"].shape == (12, 3)
    np.testing.assert_allclose(result["selected_times"], [1.0, 2.0, 3.0])
    assert np.max(result["sample_positions"][:, 0]) == 30.0
    assert result["B_reconstructed"].shape == (result["grid"].points.shape[0], 3)
    assert result["metadata"]["coordinate_policy"].startswith("unwrapped")
    assert result["rbf_metadata"]["sigma"] == 10.0
    assert result["rbf_metadata"]["implementation"] == "direct_multiquadric_solver"


def test_compute_rbf_reconstruction_supports_time_stride(tmp_path):
    input_path = _write_rbf_input(tmp_path / "input.h5")

    result = compute_rbf_reconstruction(
        input_path,
        _config(t0=2.0, delta_t=4.0, sample_time_stride=2),
    )

    np.testing.assert_allclose(result["selected_times"], [0.0, 2.0, 4.0])
    assert result["sample_trajectory"].shape == (3, 4, 3)
    assert result["sample_positions"].shape == (12, 3)
    assert result["settings"]["sample_time_stride"] == 2


def test_compute_rbf_reconstruction_selects_observed_mc_member(tmp_path):
    input_path = _write_rbf_input(tmp_path / "input.h5")

    result = compute_rbf_reconstruction(
        input_path,
        _config(data_group="observed", mc_index=0, t0=2.0, delta_t=2.0),
    )

    with h5py.File(input_path) as handle:
        expected = handle["observed/B"][0, 1:4].reshape(-1, 3)
        truth = handle["truth/B"][1:4].reshape(-1, 3)
    np.testing.assert_allclose(result["sample_B"], expected)
    assert not np.allclose(result["sample_B"], truth)
    assert result["settings"]["data_group"] == "observed"
    assert result["settings"]["mc_index"] == 0
    assert result["metadata"]["data_group"] == "observed"
    assert result["metadata"]["mc_index"] == 0


def test_compute_rbf_reconstruction_rejects_observed_mc_all(tmp_path):
    input_path = _write_rbf_input(tmp_path / "input.h5")

    with pytest.raises(NotImplementedError, match="one observed realization"):
        compute_rbf_reconstruction(input_path, _config(data_group="observed", mc_index="all"))


def test_collect_samples_for_window_preserves_unwrapped_positions():
    times = np.array([0.0, 1.0, 2.0])
    positions = np.arange(18.0).reshape(3, 2, 3)
    magnetic = _linear_B(positions.reshape(-1, 3)).reshape(3, 2, 3)

    _t0, selected_times, selected_positions, sample_positions, sample_B = (
        collect_samples_for_window(times, positions, magnetic, t0=1.0, delta_t=2.0)
    )

    np.testing.assert_allclose(selected_times, times)
    np.testing.assert_allclose(selected_positions, positions)
    np.testing.assert_allclose(sample_positions, positions.reshape(-1, 3))
    np.testing.assert_allclose(sample_B, magnetic.reshape(-1, 3))


def test_collect_samples_for_window_applies_time_stride():
    times = np.arange(6.0)
    positions = np.arange(36.0).reshape(6, 2, 3)
    magnetic = _linear_B(positions.reshape(-1, 3)).reshape(6, 2, 3)

    _t0, selected_times, selected_positions, sample_positions, sample_B = (
        collect_samples_for_window(
            times,
            positions,
            magnetic,
            t0=2.5,
            delta_t=5.0,
            time_stride=2,
        )
    )

    np.testing.assert_allclose(selected_times, [0.0, 2.0, 4.0])
    np.testing.assert_allclose(selected_positions, positions[[0, 2, 4]])
    np.testing.assert_allclose(sample_positions, positions[[0, 2, 4]].reshape(-1, 3))
    np.testing.assert_allclose(sample_B, magnetic[[0, 2, 4]].reshape(-1, 3))


def test_compute_rbf_reconstruction_rejects_too_few_window_samples(tmp_path):
    input_path = _write_rbf_input(tmp_path / "input.h5")

    with pytest.raises(ValueError, match="requires at least 20 samples"):
        compute_rbf_reconstruction(input_path, _config(min_samples=20))


def test_analysis_cli_writes_hdf5_and_plot(tmp_path):
    input_path = _write_rbf_input(tmp_path / "input.h5")
    output_path = tmp_path / "rbf.h5"
    plot_path = tmp_path / "rbf.png"
    config_path = tmp_path / "rbf.yaml"
    config_path.write_text(
        f"""
analysis:
  method: rbf_reconstruction
input:
  path: {input_path}
output:
  path: {output_path}
rbf_reconstruction:
  data_group: truth
  t0_index: 2
  delta_t: 2.0
  sigma: 10.0
  smoothing: 1.0e-10
  grid:
    radius: 1.5
    n_axial: 5
    n_transverse: 5
  plot:
    enabled: true
    path: {plot_path}
    dpi: 100
""",
        encoding="utf-8",
    )

    assert "rbf_reconstruction" in list_analyses()
    assert analyze_from_config(config_path) == output_path
    assert output_path.is_file()
    assert plot_path.is_file() and plot_path.stat().st_size > 0
    with h5py.File(output_path) as data:
        assert data["grid/points"].shape[1] == 3
        assert data["grid/full_points"].shape == (5, 5, 5, 3)
        np.testing.assert_array_equal(data["grid/dimensions"][...], [5, 5, 5])
        assert data["reconstruction/B"].shape[1] == 3
        assert data["reconstruction/Bmag"].shape == (data["reconstruction/B"].shape[0],)
        assert data["samples/position"].shape == (12, 3)
        assert data["samples/trajectory"].shape == (3, 4, 3)
        assert data["samples/spacecraft_id"].shape == (4,)
        assert data["grid/center"].shape == (3,)
        assert data["grid/cylinder_mask"].shape == (5, 5, 5)
        assert data["metadata"].attrs["kernel"] == "multiquadric"
        assert data["metadata"].attrs["sigma"] == 10.0
        assert data["metadata"].attrs["sample_time_stride"] == 1
        assert data["metadata"].attrs["condition_number"] > 0.0
        assert not data["metadata/grid"].attrs["require_all_spacecraft_coverage"]
        assert data["metadata"].attrs["coordinate_policy"].startswith("unwrapped")
        assert data["rbf"].attrs["sigma_source"] == "manual"
        assert data["rbf"].attrs["sigma"] == 10.0

    regenerated = tmp_path / "regenerated.png"
    assert plot_rbf_reconstruction(output_path, regenerated) == regenerated
    assert regenerated.stat().st_size > 0


def test_analysis_cli_auto_sigma_writes_selection_and_plot(tmp_path):
    input_path = _write_rbf_input(tmp_path / "input.h5")
    output_path = tmp_path / "rbf-auto.h5"
    plot_path = tmp_path / "sigma.png"
    cache_path = tmp_path / "sigma-cache.h5"
    config_path = tmp_path / "rbf-auto.yaml"
    config_path.write_text(
        f"""
analysis:
  method: rbf_reconstruction
input:
  path: {input_path}
output:
  path: {output_path}
rbf_reconstruction:
  data_group: truth
  t0_index: 2
  delta_t: 2.0
  auto_sigma_rippa: true
  sigma_min: 2.0
  sigma_max: 20.0
  sigma_num: 4
  sigma_grid: linear
  sigma_cache: {cache_path}
  rippa_mode: field
  rippa_components: all
  smoothing: 1.0e-10
  grid:
    radius: 1.5
    n_axial: 5
    n_transverse: 5
  plot:
    enabled: false
  rippa_plot:
    enabled: true
    path: {plot_path}
""",
        encoding="utf-8",
    )

    assert analyze_from_config(config_path) == output_path
    with h5py.File(output_path) as handle:
        assert handle["sigma_selection/sigma_candidates"].shape == (4,)
        assert handle["sigma_selection/costs"].shape == (4,)
        selected_sigma = handle["metadata"].attrs["sigma"]
        candidates = handle["sigma_selection/sigma_candidates"][...]
        assert selected_sigma in candidates
        assert handle["rbf"].attrs["sigma_source"] == "computed_rippa"
        assert handle["rbf"].attrs["sigma_cache_path"] == str(cache_path)
        assert "rbf/sigma_candidates" in handle
        assert "rbf/loocv_costs" in handle
    assert cache_path.is_file()

    assert plot_path.is_file() and plot_path.stat().st_size > 0
    regenerated = tmp_path / "sigma-regenerated.png"
    assert plot_rippa_sigma_selection(output_path, regenerated) == regenerated
    assert regenerated.stat().st_size > 0

    cached_output_path = tmp_path / "rbf-auto-cached.h5"
    cached_plot_path = tmp_path / "sigma-cached.png"
    cached_config_path = tmp_path / "rbf-auto-cached.yaml"
    cached_config_path.write_text(
        config_path.read_text(encoding="utf-8")
        .replace(str(output_path), str(cached_output_path))
        .replace(str(plot_path), str(cached_plot_path)),
        encoding="utf-8",
    )
    assert analyze_from_config(cached_config_path) == cached_output_path
    with h5py.File(cached_output_path) as handle:
        assert handle["rbf"].attrs["sigma_source"] == "cache"


def test_analysis_cli_explicit_loocv_sigma_selection_uses_cache(tmp_path):
    input_path = _write_rbf_input(tmp_path / "input.h5")
    output_path = tmp_path / "rbf-explicit.h5"
    cache_path = tmp_path / "explicit-cache.h5"
    config_path = tmp_path / "rbf-explicit.yaml"
    config_path.write_text(
        f"""
analysis:
  method: rbf_reconstruction
input:
  path: {input_path}
output:
  path: {output_path}
rbf_reconstruction:
  data_group: truth
  t0_index: 2
  delta_t: 2.0
  sigma_selection: explicit_loocv
  sigma_initial: 5.0
  sigma_min: 1.0
  sigma_max: 20.0
  sigma_subsample_max: 8
  sigma_random_seed: 4
  sigma_cache: {cache_path}
  smoothing: 1.0e-10
  grid:
    radius: 1.5
    n_axial: 5
    n_transverse: 5
  plot:
    enabled: false
  rippa_plot:
    enabled: false
""",
        encoding="utf-8",
    )

    assert analyze_from_config(config_path) == output_path
    with h5py.File(output_path) as handle:
        assert handle["rbf"].attrs["sigma_source"] == "computed_explicit_loocv"
        assert handle["sigma_selection"].attrs["method"] == "explicit_loocv_vector_norm"
        assert handle["metadata"].attrs["sigma_selection"] == "explicit_loocv"

    cached_output_path = tmp_path / "rbf-explicit-cached.h5"
    cached_config_path = tmp_path / "rbf-explicit-cached.yaml"
    cached_config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(str(output_path), str(cached_output_path)),
        encoding="utf-8",
    )
    assert analyze_from_config(cached_config_path) == cached_output_path
    with h5py.File(cached_output_path) as handle:
        assert handle["rbf"].attrs["sigma_source"] == "cache"


def test_direct_simulation_reference_helper_writes_validation_and_plot(tmp_path):
    observation_path = _write_rbf_input(tmp_path / "input.h5")
    output_path = tmp_path / "rbf.h5"
    method_result = compute_rbf_reconstruction(observation_path, _config())
    from hs_observatory.analysis.rbf_reconstruction import write_rbf_reconstruction

    write_rbf_reconstruction(method_result, output_path)

    snapshot = np.zeros((4, 4, 4, 7), dtype=float)
    snapshot[..., 4] = 1.0
    snapshot[..., 5] = 2.0
    snapshot[..., 6] = 3.0
    snapshot_path = tmp_path / "snapshot.npy"
    np.save(snapshot_path, snapshot)
    generation_config = tmp_path / "generate.yaml"
    generation_config.write_text(
        f"""
spacecraft:
  source: synthetic
time:
  t_start_s: 0.0
  t_stop_s: 1.0
  cadence_s: 1.0
field:
  model: simulation_sampled
  background:
    n: 0.0
    v: [0.0, 0.0, 0.0]
    B: [0.0, 0.0, 0.0]
  simulation:
    format: npy
    path: {snapshot_path}
    coordinates:
      units: km
      box_size_km: [100.0, 100.0, 100.0]
    coordinate_mapping:
      method: hub_centered
      hub_spacecraft: 0
    fields_are:
      n: total
      v: total
      B: total
    scale_factors:
      n: 1.0
      v: 1.0
      B: 10.0
errors:
  n_mc: 1
output:
  path: {tmp_path / "unused.h5"}
""",
        encoding="utf-8",
    )

    add_simulation_reference_to_rbf_output(output_path, observation_path, generation_config)

    with h5py.File(output_path) as handle:
        assert handle["validation/B_actual"].shape == handle["reconstruction/B"].shape
        assert handle["validation/error_mag"].shape == (handle["reconstruction/B"].shape[0],)
        assert handle["validation/error_magnitude"].shape == handle["validation/error_mag"].shape
        assert handle["validation/error_percent"].shape == handle["validation/error_mag"].shape
        assert handle["validation/error_percent_capped"].shape == (
            handle["reconstruction/B"].shape[0],
        )
        assert handle["validation/Bmag_actual"].shape == handle["validation/error_mag"].shape
        assert handle["validation/component_error"].shape == handle["reconstruction/B"].shape
        assert np.nanmax(handle["validation/error_percent_capped"][...]) <= 100.0
        assert handle["validation"].attrs["error_percent_cap"] == 100.0
        np.testing.assert_allclose(handle["validation/B_actual"][0], [10.0, 20.0, 30.0])

    reference_plot = tmp_path / "reference.png"
    assert plot_rbf_simulation_reference(output_path, reference_plot) == reference_plot
    assert reference_plot.stat().st_size > 0

    uv_error_plot = tmp_path / "uv-error.png"
    assert plot_rbf_broeren_uv_error(output_path, uv_error_plot) == uv_error_plot
    assert uv_error_plot.stat().st_size > 0
