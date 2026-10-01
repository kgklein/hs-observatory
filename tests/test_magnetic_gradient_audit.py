from __future__ import annotations

import numpy as np
import pytest

from hs_observatory.analysis.magnetic_gradient import (
    estimate_gradient_least_squares,
    gradient_curl_divergence,
    periodic_grid_gradient,
)


def _configurations() -> dict[str, np.ndarray]:
    regular = np.array(
        (
            (1.0, 1.0, 1.0),
            (1.0, -1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
        )
    )
    irregular = np.array(
        (
            (0.0, 0.0, 0.0),
            (2.0, 0.1, 0.2),
            (0.3, 1.7, 0.4),
            (0.2, 0.5, 1.3),
        )
    )
    five = np.vstack((regular, (0.2, -0.1, 0.3)))
    nine = np.random.default_rng(90210).normal(size=(9, 3))
    return {
        "regular_tetrahedron": regular,
        "irregular_tetrahedron": irregular,
        "five_point": five,
        "nine_point": nine,
    }


@pytest.mark.parametrize("n_spacecraft", (4, 5, 9))
def test_unordered_pairwise_form_matches_centered_covariance(n_spacecraft):
    rng = np.random.default_rng(1000 + n_spacecraft)
    maximum_difference = 0.0
    for _ in range(8):
        positions = rng.normal(size=(n_spacecraft, 3))
        field = rng.normal(size=(n_spacecraft, 3))
        centered_positions = positions - np.mean(positions, axis=0)
        centered_field = field - np.mean(field, axis=0)
        volumetric_tensor = centered_positions.T @ centered_positions / n_spacecraft
        left, right = np.triu_indices(n_spacecraft, k=1)
        pair_covariance = (
            (field[left] - field[right]).T @ (positions[left] - positions[right]) / n_spacecraft**2
        )
        pairwise = pair_covariance @ np.linalg.inv(volumetric_tensor)
        centered = (
            centered_field.T
            @ centered_positions
            @ np.linalg.inv(centered_positions.T @ centered_positions)
        )
        maximum_difference = max(maximum_difference, float(np.max(np.abs(pairwise - centered))))
    print(f"N={n_spacecraft}: max pairwise-centered difference {maximum_difference:.3e}")
    assert maximum_difference < 5.0e-12


def test_literal_ordered_pair_sum_is_twice_the_centered_estimator():
    rng = np.random.default_rng(42)
    positions = rng.normal(size=(9, 3))
    field = rng.normal(size=(9, 3))
    centered_positions = positions - np.mean(positions, axis=0)
    centered_field = field - np.mean(field, axis=0)
    volumetric_tensor = centered_positions.T @ centered_positions / len(positions)
    field_difference = field[:, None, :] - field[None, :, :]
    position_difference = positions[:, None, :] - positions[None, :, :]
    ordered = (
        np.einsum("abi,abj->ij", field_difference, position_difference)
        / len(positions) ** 2
        @ np.linalg.inv(volumetric_tensor)
    )
    centered = (
        centered_field.T
        @ centered_positions
        @ np.linalg.inv(centered_positions.T @ centered_positions)
    )
    np.testing.assert_allclose(ordered, 2.0 * centered, rtol=2.0e-14, atol=2.0e-14)


@pytest.mark.parametrize(
    ("name", "known", "expected_divergence", "expected_curl"),
    (
        (
            "divergence_and_curl_free",
            np.diag((1.0, -1.0, 0.0)),
            0.0,
            np.zeros(3),
        ),
        ("pure_divergence", np.eye(3), 3.0, np.zeros(3)),
        (
            "pure_positive_z_curl",
            np.array(((0.0, -1.7, 0.0), (1.7, 0.0, 0.0), (0.0, 0.0, 0.0))),
            0.0,
            np.array((0.0, 0.0, 3.4)),
        ),
        (
            "dense_nonsymmetric",
            np.array(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 10.0))),
            16.0,
            np.array((2.0, -4.0, 2.0)),
        ),
    ),
)
def test_required_affine_fields_recover_every_tensor_component(
    name, known, expected_divergence, expected_curl
):
    offset = np.array((0.4, -0.7, 2.1))
    for configuration_name, positions in _configurations().items():
        field = offset + positions @ known.T
        estimate = estimate_gradient_least_squares(positions, field)
        np.testing.assert_allclose(
            estimate.gradient,
            known,
            rtol=2.0e-14,
            atol=1.0e-14,
            err_msg=f"{name}, {configuration_name}",
        )
        curl, _magnitude, divergence = gradient_curl_divergence(estimate.gradient)
        assert divergence == pytest.approx(expected_divergence, abs=1.0e-14)
        np.testing.assert_allclose(curl, expected_curl, rtol=2.0e-14, atol=1.0e-14)
        np.testing.assert_allclose(
            estimate.fitted_field,
            estimate.mean_field + (positions - estimate.barycenter) @ estimate.gradient.T,
            rtol=2.0e-14,
            atol=1.0e-14,
        )


def test_four_point_estimator_matches_independent_affine_system_for_any_field():
    rng = np.random.default_rng(719)
    for name in ("regular_tetrahedron", "irregular_tetrahedron"):
        positions = _configurations()[name]
        field = rng.normal(size=(4, 3))
        generalized = estimate_gradient_least_squares(positions, field).gradient
        design = np.column_stack((np.ones(4), positions))
        reciprocal = np.linalg.solve(design, field).T[:, 1:]
        generalized_curl, _magnitude, generalized_divergence = gradient_curl_divergence(generalized)
        reciprocal_curl, _magnitude, reciprocal_divergence = gradient_curl_divergence(reciprocal)
        print(
            name,
            "max dG",
            np.max(np.abs(generalized - reciprocal)),
            "max dcurl",
            np.max(np.abs(generalized_curl - reciprocal_curl)),
            "ddiv",
            abs(generalized_divergence - reciprocal_divergence),
        )
        np.testing.assert_allclose(generalized, reciprocal, rtol=2.0e-14, atol=2.0e-14)
        np.testing.assert_allclose(generalized_curl, reciprocal_curl, rtol=2.0e-14, atol=2.0e-14)
        assert generalized_divergence == pytest.approx(reciprocal_divergence, abs=2.0e-14)


def test_periodic_grid_gradient_matches_transparent_centered_difference():
    rng = np.random.default_rng(26)
    magnetic = rng.normal(size=(5, 6, 7, 3))
    box = np.array((10.0, 18.0, 28.0))
    spacing = box / magnetic.shape[:3]
    expected = np.empty((*magnetic.shape[:3], 3, 3))
    for ix in range(magnetic.shape[0]):
        for iy in range(magnetic.shape[1]):
            for iz in range(magnetic.shape[2]):
                index = [ix, iy, iz]
                for field_component in range(3):
                    for derivative_axis in range(3):
                        lower = index.copy()
                        upper = index.copy()
                        lower[derivative_axis] = (lower[derivative_axis] - 1) % magnetic.shape[
                            derivative_axis
                        ]
                        upper[derivative_axis] = (upper[derivative_axis] + 1) % magnetic.shape[
                            derivative_axis
                        ]
                        expected[ix, iy, iz, field_component, derivative_axis] = (
                            magnetic[tuple(upper)][field_component]
                            - magnetic[tuple(lower)][field_component]
                        ) / (2.0 * spacing[derivative_axis])
    np.testing.assert_array_equal(periodic_grid_gradient(magnetic, box), expected)
