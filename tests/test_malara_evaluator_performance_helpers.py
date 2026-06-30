"""Correctness guards for Malara performance-oriented evaluator helpers."""

import numpy as np
import pytest

from hs_observatory.fields.malara import MalaraConfig, MalaraTurbulence


def _evaluator(**overrides: object) -> MalaraTurbulence:
    values: dict[str, object] = {
        "domain_size": (2.0, 3.0, 5.0),
        "n_scales": 4,
        "injection_scale_index": 2,
        "base_seed": 1357,
        "realization": 4,
    }
    values.update(overrides)
    return MalaraTurbulence(MalaraConfig(**values))


def test_chunked_matches_direct_for_flat_positions() -> None:
    evaluator = _evaluator()
    positions = np.random.default_rng(8).uniform(0.0, 1.0, size=(17, 3))

    direct = evaluator.evaluate_vector(positions)
    chunked = evaluator.evaluate_vector_chunked(positions, chunk_size=6)

    assert np.array_equal(chunked, direct)


def test_chunked_preserves_multidimensional_shape_and_values() -> None:
    evaluator = _evaluator(n_scales=3)
    positions = np.random.default_rng(9).uniform(0.0, 1.0, size=(12, 9, 3))

    direct = evaluator.evaluate_vector(positions)
    chunked = evaluator.evaluate_vector_chunked(positions, chunk_size=17)

    assert chunked.shape == (12, 9, 3)
    assert np.array_equal(chunked, direct)


def test_touched_eddy_statistics_are_consistent() -> None:
    evaluator = _evaluator()
    positions = np.random.default_rng(10).uniform(0.0, 1.0, size=(5, 3))

    statistics = evaluator.collect_touched_eddy_statistics(positions)

    assert statistics["n_points"] == 5
    assert statistics["total_eddy_touches"] > 0
    assert statistics["n_unique_eddies"] > 0
    assert statistics["reuse_factor"] >= 1.0
    assert set(statistics["touches_by_scale"]) == {2, 3, 4}
    assert set(statistics["unique_eddies_by_scale"]) == {2, 3, 4}
    assert sum(statistics["touches_by_scale"].values()) == statistics[
        "total_eddy_touches"
    ]


def test_repeated_positions_expose_eddy_reuse() -> None:
    evaluator = _evaluator()
    positions = np.repeat(np.array([[0.37, 0.81, 1.23]]), 20, axis=0)

    statistics = evaluator.collect_touched_eddy_statistics(positions)

    assert statistics["reuse_factor"] > 1.0


def test_chunked_evaluation_preserves_periodicity() -> None:
    evaluator = _evaluator()
    point = np.array([0.31, 1.17, 4.2])
    domain = np.array(evaluator.config.domain_size)
    positions = np.vstack(
        [point, point + [domain[0], 0.0, 0.0], point + [0.0, domain[1], 0.0],
         point + [0.0, 0.0, domain[2]]]
    )

    direct = evaluator.evaluate_vector(positions)
    chunked = evaluator.evaluate_vector_chunked(positions, chunk_size=2)

    assert np.array_equal(chunked, direct)
    np.testing.assert_allclose(direct, np.broadcast_to(direct[0], direct.shape))


def test_chunk_size_validation_and_single_worker_parallel_path() -> None:
    evaluator = _evaluator()
    positions = np.zeros((2, 3))

    with pytest.raises(ValueError, match="positive"):
        evaluator.evaluate_vector_chunked(positions, chunk_size=0)
    reference = evaluator.evaluate_vector(positions)
    parallel = evaluator.evaluate_vector_parallel(
        positions, chunk_size=1, n_workers=1
    )
    assert np.array_equal(parallel, reference)
