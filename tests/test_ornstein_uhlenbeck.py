import numpy as np
import pytest

from hs_observatory.errors.gaussian import error_model_from_config
from hs_observatory.errors.ornstein_uhlenbeck import (
    ScalarOrnsteinUhlenbeckError,
    VectorOrnsteinUhlenbeckError,
)


def test_scalar_ou_is_reproducible_stationary_and_correlated():
    time = np.arange(2000, dtype=float) * 0.1
    truth = np.zeros((len(time), 4))
    model = ScalarOrnsteinUhlenbeckError(sigma=2.0, tau_s=1.5)

    observed_a, error_a = model.apply(truth, np.random.default_rng(31), 16, time=time)
    observed_b, error_b = model.apply(truth, np.random.default_rng(31), 16, time=time)

    assert observed_a.shape == (16, len(time), 4)
    assert np.array_equal(observed_a, observed_b)
    assert np.array_equal(error_a, error_b)
    assert np.isclose(np.std(error_a), 2.0, rtol=0.03)
    lag_correlation = np.corrcoef(error_a[:, :-1].ravel(), error_a[:, 1:].ravel())[0, 1]
    assert np.isclose(lag_correlation, np.exp(-0.1 / 1.5), atol=0.015)
    assert not np.array_equal(error_a[:, :, 0], error_a[:, :, 1])


def test_scalar_ou_uses_irregular_time_intervals():
    time = np.array([0.0, 0.25, 1.25])
    truth = np.zeros((len(time), 1))
    model = ScalarOrnsteinUhlenbeckError(sigma=1.0, tau_s=2.0)

    _, error = model.apply(truth, np.random.default_rng(8), 30000, time=time)

    first_correlation = np.corrcoef(error[:, 0, 0], error[:, 1, 0])[0, 1]
    second_correlation = np.corrcoef(error[:, 1, 0], error[:, 2, 0])[0, 1]
    assert np.isclose(first_correlation, np.exp(-0.25 / 2.0), atol=0.015)
    assert np.isclose(second_correlation, np.exp(-1.0 / 2.0), atol=0.015)


def test_vector_ou_preserves_direction_for_magnitude_only_and_clamps_at_zero():
    time = np.arange(20, dtype=float)
    truth = np.broadcast_to([3.0, 4.0, 0.0], (len(time), 2, 3)).copy()
    model = VectorOrnsteinUhlenbeckError(
        magnitude_sigma=3.0,
        magnitude_tau_s=2.0,
        direction_sigma_deg=0.0,
    )

    observed, error = model.apply(truth, np.random.default_rng(4), 40, time=time)

    assert error.shape == observed.shape == (40, len(time), 2, 3)
    cross_product = np.cross(observed, truth[None, ...])
    dot_product = np.sum(observed * truth[None, ...], axis=-1)
    assert np.allclose(cross_product, 0.0)
    assert np.all(dot_product >= 0.0)
    assert np.any(np.linalg.norm(observed, axis=-1) == 0.0)


def test_vector_direction_sigma_is_rms_total_angle_and_preserves_magnitude():
    time = np.arange(1500, dtype=float) * 0.1
    truth = np.zeros((len(time), 8, 3))
    truth[..., 0] = 1.0
    model = VectorOrnsteinUhlenbeckError(
        magnitude_sigma=0.0,
        magnitude_tau_s=1.0,
        direction_sigma_deg=5.0,
        direction_tau_s=0.7,
    )

    observed, _ = model.apply(truth, np.random.default_rng(12), 8, time=time)
    observed_magnitude = np.linalg.norm(observed, axis=-1)
    angle_deg = np.rad2deg(np.arccos(np.clip(observed[..., 0], -1.0, 1.0)))

    assert np.allclose(observed_magnitude, 1.0)
    assert np.isclose(np.sqrt(np.mean(angle_deg**2)), 5.0, rtol=0.06)


def test_vector_ou_preserves_zero_vectors_and_direction_tau_falls_back():
    time = np.arange(5, dtype=float)
    truth = np.zeros((len(time), 2, 3))
    model = VectorOrnsteinUhlenbeckError(
        magnitude_sigma=0.2,
        magnitude_tau_s=3.0,
        direction_sigma_deg=4.0,
    )

    observed, error = model.apply(truth, np.random.default_rng(5), 3, time=time)

    assert model.direction_tau_s == model.magnitude_tau_s
    assert not np.any(observed)
    assert not np.any(error)


def test_ou_factory_selects_scalar_and_vector_models():
    scalar = error_model_from_config(
        {"model": "ornstein_uhlenbeck", "sigma": 0.2, "tau_s": 1.0},
        quantity="n",
    )
    vector = error_model_from_config(
        {
            "model": "ornstein_uhlenbeck",
            "magnitude_sigma": 0.01,
            "magnitude_tau_s": 2.0,
            "direction_sigma_deg": 0.5,
        },
        quantity="B",
    )
    assert isinstance(scalar, ScalarOrnsteinUhlenbeckError)
    assert isinstance(vector, VectorOrnsteinUhlenbeckError)


def test_ou_rejects_missing_or_nonincreasing_time():
    model = ScalarOrnsteinUhlenbeckError(sigma=1.0, tau_s=2.0)
    truth = np.zeros((3, 1))
    with pytest.raises(ValueError, match="require a time axis"):
        model.apply(truth, np.random.default_rng(1), 1)
    with pytest.raises(ValueError, match="strictly increasing"):
        model.apply(truth, np.random.default_rng(1), 1, time=np.array([0.0, 1.0, 1.0]))
