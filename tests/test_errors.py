import numpy as np
import pytest

from hs_observatory.errors.gaussian import (
    AbsoluteGaussianError,
    NoError,
    RelativeGaussianError,
    VectorComponentGaussianError,
)
from hs_observatory.errors.offsets import GaussianPositionOffset, GaussianTimeOffset


@pytest.mark.parametrize(
    "model,truth",
    [
        (AbsoluteGaussianError(0.2), np.ones((4, 9))),
        (RelativeGaussianError(0.1), np.ones((4, 9))),
        (VectorComponentGaussianError([0.1, 0.2, 0.3]), np.ones((4, 9, 3))),
    ],
)
def test_gaussian_errors_are_reproducible(model, truth):
    observed_a, errors_a = model.apply(truth, np.random.default_rng(17), n_mc=3)
    observed_b, errors_b = model.apply(truth, np.random.default_rng(17), n_mc=3)
    assert observed_a.shape == (3, *truth.shape)
    assert np.array_equal(observed_a, observed_b)
    assert np.array_equal(errors_a, errors_b)


def test_no_error_still_adds_monte_carlo_dimension():
    truth = np.arange(6).reshape(2, 3)
    observed, error = NoError().apply(truth, np.random.default_rng(1), n_mc=1)
    assert observed.shape == (1, 2, 3)
    assert np.array_equal(observed[0], truth)
    assert not np.any(error)


def test_coordinate_offsets_are_reproducible_and_constant_over_time_by_design():
    time_model = GaussianTimeOffset(0.02)
    position_model = GaussianPositionOffset([1.0, 2.0, 3.0])
    time_a = time_model.draw(np.random.default_rng(9), n_mc=4, n_spacecraft=9)
    time_b = time_model.draw(np.random.default_rng(9), n_mc=4, n_spacecraft=9)
    position_a = position_model.draw(np.random.default_rng(12), n_mc=4, n_spacecraft=9)
    position_b = position_model.draw(np.random.default_rng(12), n_mc=4, n_spacecraft=9)

    assert time_a.shape == (4, 9)
    assert position_a.shape == (4, 9, 3)
    assert np.array_equal(time_a, time_b)
    assert np.array_equal(position_a, position_b)
