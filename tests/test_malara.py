import numpy as np

from hs_observatory.fields.malara import (
    F,
    MalaraConfig,
    MalaraTurbulence,
    compact_basis,
    compact_basis_derivative,
    dF_dt,
)


def test_compact_basis_values_and_analytic_derivative():
    points = np.array([-0.75, -0.5, -0.2, 0.0, 0.2, 0.5, 0.75])
    values = compact_basis(points)
    derivative = compact_basis_derivative(points)

    assert np.all(values[[0, 1, 5, 6]] == 0.0)
    assert values[3] == 1.0
    assert np.all(derivative[[0, 1, 3, 5, 6]] == 0.0)
    assert np.allclose(values, values[::-1])
    assert np.allclose(derivative, -derivative[::-1])

    interior = np.linspace(-0.45, 0.45, 11)
    step = 1.0e-7
    finite_difference = (compact_basis(interior + step) - compact_basis(interior - step)) / (
        2.0 * step
    )
    assert np.allclose(compact_basis_derivative(interior), finite_difference, atol=2.0e-7)
    assert np.array_equal(F(points), values)
    assert np.array_equal(dF_dt(points), derivative)


def test_evaluate_vector_preserves_arbitrary_leading_shapes():
    model = MalaraTurbulence()
    points = np.random.default_rng(3).random((7, 3))
    trajectories = np.random.default_rng(4).random((3, 5, 3))

    assert model.evaluate_vector(points).shape == (7, 3)
    assert model.evaluate_vector(trajectories).shape == (3, 5, 3)


def test_reproducibility_and_realization_selection():
    points = np.random.default_rng(7).random((8, 3))
    first = MalaraTurbulence(MalaraConfig(base_seed=91, realization=2))
    repeated = MalaraTurbulence(MalaraConfig(base_seed=91, realization=2))
    different = MalaraTurbulence(MalaraConfig(base_seed=91, realization=3))

    first_values = first.evaluate_vector(points)
    assert np.array_equal(first_values, repeated.evaluate_vector(points))
    assert not np.array_equal(first_values, different.evaluate_vector(points))


def test_cell_parameters_and_eight_local_support_candidates_are_deterministic():
    model = MalaraTurbulence(MalaraConfig(n_scales=3, injection_scale_index=2))
    first = model._eddy_parameters(2, 0, 1, 3)
    repeated = model._eddy_parameters(2, 0, 1, 3)

    assert np.array_equal(first.gamma, repeated.gamma)
    assert np.array_equal(first.q_hat, repeated.q_hat)
    assert first.sign == repeated.sign
    assert first.high_octants == repeated.high_octants
    assert np.all(np.abs(first.gamma) <= 1.0)
    assert first.sign in {-1.0, 1.0}
    assert np.isclose(np.linalg.norm(first.q_hat), 1.0)
    assert len(first.high_octants) == 4

    point = np.array([0.10, 0.11, 0.12])
    cells = model._contributing_cells(point, scale=2)
    assert len(cells) == 8
    cell_size = np.asarray(model.config.domain_size) / 4.0
    for cell in cells:
        center = (np.asarray(cell) + 0.5) * cell_size
        displacement = (point - center + 0.5) % 1.0 - 0.5
        assert np.all(np.abs(displacement) <= cell_size)


def test_random_positions_are_finite_and_field_is_periodic():
    config = MalaraConfig(domain_size=(2.0, 3.0, 5.0), n_scales=5)
    model = MalaraTurbulence(config)
    points = np.random.default_rng(12).uniform(-10.0, 10.0, size=(10, 3))
    values = model.evaluate_vector(points)

    assert np.all(np.isfinite(values))
    for axis, length in enumerate(config.domain_size):
        offset = np.zeros(3)
        offset[axis] = length
        assert np.allclose(values, model.evaluate_vector(points + offset), atol=2.0e-12)


def test_field_is_approximately_divergence_free_on_diagnostic_grid():
    model = MalaraTurbulence(MalaraConfig(n_scales=4, injection_scale_index=2))
    axis = np.array([0.13, 0.37, 0.71])
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    step_size = 1.0e-6
    divergence = np.zeros(len(points))

    for component in range(3):
        step = np.zeros(3)
        step[component] = step_size
        forward = model.evaluate_vector(points + step)[:, component]
        backward = model.evaluate_vector(points - step)[:, component]
        divergence += (forward - backward) / (2.0 * step_size)

    assert np.max(np.abs(divergence)) < 1.0e-4


def test_nonintermittent_p_model_has_unit_transfer_and_finite_values():
    model = MalaraTurbulence(MalaraConfig(intermittency_p=0.5))
    for scale in range(5):
        n_cells = 2**scale
        cell = (n_cells - 1, n_cells // 2, 0)
        assert model.energy_transfer_factor(scale, cell) == 1.0
    values = model.evaluate_vector(np.random.default_rng(5).random((6, 3)))
    assert np.all(np.isfinite(values))


def test_increasing_scale_count_preserves_api_and_shape():
    positions = np.random.default_rng(18).random((2, 4, 3))
    coarse = MalaraTurbulence(MalaraConfig(n_scales=4))
    fine = MalaraTurbulence(MalaraConfig(n_scales=7))

    coarse_values = coarse.evaluate_vector(positions)
    fine_values = fine.evaluate_vector(positions)
    assert coarse_values.shape == fine_values.shape == positions.shape
    assert not np.array_equal(coarse_values, fine_values)
