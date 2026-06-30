"""Tests for benchmark position sampling modes."""

import numpy as np

from scripts.benchmark_malara_evaluator import make_trajectory_like_positions


def test_trajectory_like_positions_are_reproducible_and_keep_formation() -> None:
    first = make_trajectory_like_positions(27, np.random.default_rng(42))
    second = make_trajectory_like_positions(27, np.random.default_rng(42))

    assert first.shape == (27, 3)
    assert np.array_equal(first, second)
    assert np.all((first >= 0.0) & (first < 1.0))

    trajectory = first.reshape(3, 9, 3)
    relative = trajectory[:, 1:, :] - trajectory[:, :1, :]
    relative = (relative + 0.5) % 1.0 - 0.5
    np.testing.assert_allclose(relative, np.broadcast_to(relative[0], relative.shape))


def test_trajectory_like_positions_honor_nonmultiple_point_count() -> None:
    positions = make_trajectory_like_positions(11, np.random.default_rng(7))

    assert positions.shape == (11, 3)
