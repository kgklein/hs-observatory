import h5py
import numpy as np
import yaml

from hs_observatory.analyze import analyze_from_config
from hs_observatory.analysis.magnetic_gradient import (
    align_spacecraft_samples,
    divergence_curl_ratio,
    estimate_gradient_least_squares,
    geometry_diagnostics,
    geometry_is_static,
    gradient_curl_divergence,
    periodic_grid_gradient,
    periodic_spherical_average,
    periodic_spherical_average_fft,
    select_configurations,
    plot_magnetic_gradient_product,
)
from hs_observatory.fields.simulation_sampled import SimulationSampledField
from hs_observatory.io.hdf5 import write_observations


def _points(n, seed=14):
    rng = np.random.default_rng(seed)
    while True:
        points = rng.normal(size=(n, 3))
        if np.linalg.cond(geometry_diagnostics(points).volumetric_tensor) < 100.0:
            return points


def _affine(points):
    gradient = np.array([[1.2, -0.4, 0.8], [0.3, -2.0, 0.1], [1.1, 0.7, -0.6]])
    field = np.array([4.0, -3.0, 2.0]) + points @ gradient.T
    return gradient, field


def test_exact_affine_field_is_recovered_for_four_and_nine_points():
    for n_spacecraft in (4, 9):
        points = _points(n_spacecraft)
        known, field = _affine(points)
        estimate = estimate_gradient_least_squares(points, field)
        assert estimate.valid
        assert np.allclose(estimate.gradient, known, rtol=1.0e-12, atol=1.0e-12)
        assert estimate.rms_residual < 1.0e-12


def test_gradient_is_translation_and_uniform_field_invariant():
    points = _points(9)
    known, field = _affine(points)
    translated = estimate_gradient_least_squares(points + [100.0, -50.0, 7.0], field)
    shifted = estimate_gradient_least_squares(points, field + [9.0, -2.0, 4.0])
    assert np.allclose(translated.gradient, known)
    assert np.allclose(shifted.gradient, known)


def test_four_point_result_matches_independent_affine_system():
    points = _points(4)
    _, field = _affine(points)
    design = np.column_stack((np.ones(4), points))
    reciprocal_reference = np.linalg.solve(design, field).T[:, 1:]
    estimate = estimate_gradient_least_squares(points, field)
    assert np.allclose(estimate.gradient, reciprocal_reference, atol=1.0e-12)


def test_degenerate_geometry_uses_or_rejects_pseudoinverse():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    field = np.column_stack((points[:, 0], points[:, 1], points[:, 0] + points[:, 1]))
    accepted = estimate_gradient_least_squares(points, field, allow_pseudoinverse=True)
    rejected = estimate_gradient_least_squares(points, field, allow_pseudoinverse=False)
    assert accepted.valid and accepted.used_pseudoinverse
    assert not rejected.valid
    assert np.isnan(rejected.gradient).all()


def test_curl_divergence_indexing_and_ratio_floor():
    gradient = np.array([[1.0, 2.0, 3.0], [5.0, 7.0, 11.0], [13.0, 17.0, 19.0]])
    curl, magnitude, divergence = gradient_curl_divergence(gradient)
    assert np.array_equal(curl, [6.0, -10.0, 3.0])
    assert np.isclose(magnitude, np.sqrt(145.0))
    assert divergence == 27.0
    ratio, valid = divergence_curl_ratio(np.array([2.0, 2.0]), np.array([0.01, 4.0]), 0.1)
    assert not valid[0] and np.isnan(ratio[0])
    assert valid[1] and ratio[1] == 0.5


def test_linear_time_alignment_records_interpolation_and_no_extrapolation():
    time = np.arange(4.0)
    offsets = np.array([0.2, 0.0, -0.2, 0.0])
    positions = np.zeros((4, 4, 3))
    field = np.zeros((4, 4, 3))
    for spacecraft, offset in enumerate(offsets):
        source_time = time + offset
        field[:, spacecraft, 0] = source_time
        positions[spacecraft, :, 0] = source_time
    aligned = align_spacecraft_samples(time, offsets, positions, field)
    assert not aligned.valid[0, 0]
    assert not aligned.valid[-1, 2]
    assert np.allclose(aligned.field[1:3, 0, 0], time[1:3])
    assert aligned.interpolation_required[1, 0]
    assert not aligned.interpolation_required[1, 1]


def test_periodic_derivative_and_boundary_crossing_sphere():
    shape = (32, 24, 20)
    box = np.array([2.0 * np.pi, 2.0 * np.pi, 2.0 * np.pi])
    x = np.arange(shape[0]) * box[0] / shape[0]
    y = np.arange(shape[1]) * box[1] / shape[1]
    z = np.arange(shape[2]) * box[2] / shape[2]
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    magnetic = np.stack((np.sin(xx), np.sin(yy), np.sin(zz)), axis=-1)
    gradient = periodic_grid_gradient(magnetic, box)
    assert np.max(np.abs(gradient[..., 0, 0] - np.cos(xx))) < 0.01
    values = np.arange(np.prod(shape), dtype=float).reshape(shape)
    average, count, wrapped = periodic_spherical_average(
        values, np.array([0.01, 1.0, 1.0]), 0.4, box
    )
    assert np.isfinite(average) and count > 0 and wrapped


def test_fft_spherical_average_matches_direct_at_grid_centers():
    rng = np.random.default_rng(42)
    shape = (12, 10, 8)
    box = np.array([6.0, 5.0, 8.0])
    values = rng.normal(size=(*shape, 2))
    indices = np.array([[0, 2, 3], [5, 7, 1], [11, 9, 7]])
    centers = indices * box / np.asarray(shape)
    fft_average, fft_count, fft_wrapped = periodic_spherical_average_fft(
        values, centers, 1.6, box
    )
    for sample, center in enumerate(centers):
        direct, count, wrapped = periodic_spherical_average(values, center, 1.6, box)
        assert np.allclose(fft_average[sample], direct, rtol=1.0e-12, atol=1.0e-12)
        assert fft_count[sample] == count
        assert fft_wrapped[sample] == wrapped


def test_fft_spherical_average_preserves_constant_field_off_grid():
    shape = (9, 8, 7)
    box = np.array([9.0, 8.0, 14.0])
    values = np.broadcast_to([2.0, -4.0], (*shape, 2))
    centers = np.array([[4.25, 4.0, 7.0], [8.8, 7.9, 13.7]])
    average, count, wrapped = periodic_spherical_average_fft(values, centers, 2.2, box)
    assert np.allclose(average, [2.0, -4.0])
    assert np.all(count > 0)
    assert not wrapped[0] and wrapped[1]


def test_configuration_selection_uses_q_quality_and_expected_scale_ranges():
    positions = np.array(
        [
            [0, 0, 0], [50, 0, 0], [0, 50, 0], [0, 0, 50],
            [900, 0, 0], [0, 900, 0], [0, 0, 900],
            [2200, 0, 0], [0, 2200, 2200],
        ],
        dtype=float,
    )
    selected = select_configurations(positions)
    assert np.array_equal(selected["small"]["indices"], [0, 1, 2, 3])
    assert 800.0 <= selected["intermediate"]["geometry"].characteristic_size <= 1200.0
    assert selected["large"]["geometry"].characteristic_size >= 1500.0
    assert np.array_equal(selected["all_nine"]["indices"], np.arange(9))


def test_static_geometry_detection_ignores_translation_but_not_shape_change():
    points = _points(4)
    translated = np.stack((points, points + [10.0, -3.0, 4.0]))
    changed = translated.copy()
    changed[1, 2, 0] += 0.1
    assert geometry_is_static(translated)
    assert not geometry_is_static(changed)


def test_end_to_end_hdf5_analysis_and_product_only_plot(tmp_path):
    shape = (12, 12, 12)
    box = np.array([8000.0, 8000.0, 8000.0])
    coordinates = [np.arange(count) * length / count for count, length in zip(shape, box)]
    xx, yy, zz = np.meshgrid(*coordinates, indexing="ij")
    source = np.zeros((*shape, 7))
    source[..., 0] = 1.0
    source[..., 4] = np.sin(2.0 * np.pi * yy / box[1])
    source[..., 5] = np.sin(2.0 * np.pi * zz / box[2])
    source[..., 6] = np.sin(2.0 * np.pi * xx / box[0])
    simulation_path = tmp_path / "simulation.npy"
    np.save(simulation_path, source)
    field_config = {
        "model": "simulation_sampled",
        "background": {"n": 0.0, "v": [0.0, 0.0, 0.0], "B": [0.0, 0.0, 0.0]},
        "simulation": {
            "format": "npy",
            "path": str(simulation_path),
            "coordinates": {"units": "km", "box_size_km": box.tolist()},
            "coordinate_mapping": {"method": "hub_centered", "hub_spacecraft": 0},
            "boundary": "periodic",
            "fields_are": {"n": "total", "v": "total", "B": "total"},
            "scale_factors": {"n": 1.0, "v": 1.0, "B": 5.0},
        },
    }
    points = np.array(
        [
            [0, 0, 0], [50, 0, 0], [0, 50, 0], [0, 0, 50],
            [900, 0, 0], [0, 900, 0], [0, 0, 900],
            [2200, 0, 0], [0, 2200, 2200],
        ],
        dtype=float,
    )
    time = np.arange(5.0)
    nominal_positions = points[:, None, :] + time[None, :, None] * [20.0, 3.0, 5.0]
    field_model = SimulationSampledField(field_config)
    true_b = field_model.evaluate_quantity("B", time, nominal_positions)
    n_mc = 2
    time_offset = np.zeros((n_mc, 9))
    time_offset[1] = np.linspace(-0.1, 0.1, 9)
    position_offset = np.zeros((n_mc, 9, 3))
    position_offset[1, :, 0] = np.linspace(-0.5, 0.5, 9)
    observed_b = np.empty((n_mc, *true_b.shape))
    for member in range(n_mc):
        observed_b[member] = field_model.evaluate_quantity(
            "B",
            time[:, None] + time_offset[member][None, :],
            nominal_positions + position_offset[member, :, None, :],
        )
    zeros_n = np.zeros((len(time), 9))
    zeros_v = np.zeros((len(time), 9, 3))
    observation_path = tmp_path / "observations.h5"
    generation = {"field": field_config}
    write_observations(
        observation_path,
        time={name: time for name in ("n", "v", "B")},
        spacecraft_ids=["H", *[f"N{index}" for index in range(1, 9)]],
        position={name: nominal_positions for name in ("n", "v", "B")},
        truth={"n": zeros_n, "v": zeros_v, "B": true_b},
        observed={
            "n": np.broadcast_to(zeros_n, (n_mc, *zeros_n.shape)),
            "v": np.broadcast_to(zeros_v, (n_mc, *zeros_v.shape)),
            "B": observed_b,
        },
        errors={"time_offset": time_offset, "position_offset": position_offset},
        metadata={
            "config_yaml": yaml.safe_dump(generation),
            "error_models": '{"time":"gaussian_constant_time_offset",'
            '"position":"gaussian_constant_position_offset"}',
            "schema_version": "2",
        },
    )
    product_path = tmp_path / "gradients.h5"
    config_path = tmp_path / "gradient.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "analysis": {"method": "magnetic_gradient"},
                "input": {"path": str(observation_path)},
                "output": {"path": str(product_path)},
                "magnetic_gradient": {
                    "gradient": {"condition_number_max": 1.0e12},
                    "configuration_selection": {"shape_metric": "q", "max_q": 0.6},
                    "time_sync": {"method": "linear", "allow_extrapolation": False},
                    "monte_carlo": {"store_all_realizations": True},
                    "storage": {"compression": "gzip", "chunk_time": 2},
                },
            }
        ),
        encoding="utf-8",
    )
    assert analyze_from_config(config_path) == product_path
    with h5py.File(product_path) as handle:
        assert list(handle["configurations"]) == ["all_nine", "intermediate", "large", "small"]
        small = handle["configurations/small"]
        assert np.array_equal(small["spacecraft_indices"], [0, 1, 2, 3])
        assert small["true_estimate/gradient"].shape == (5, 3, 3)
        assert small["measured/gradient"].shape == (2, 5, 3, 3)
        assert small["mc_summary/gradient/median"].shape == (5, 3, 3)
        assert small["truth/coarse_gradient"].attrs["units"] == "nT/km"
        assert small["measured/gradient"].compression == "gzip"
        assert not small["measured/synchronization/valid"][1].all()
    observation_path.unlink()
    simulation_path.unlink()
    figure_path = tmp_path / "gradient.png"
    assert plot_magnetic_gradient_product(product_path, figure_path, dpi=80) == figure_path
    assert figure_path.stat().st_size > 0
