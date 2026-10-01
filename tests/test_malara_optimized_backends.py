"""Scientific equivalence tests for optimized Malara evaluator backends."""

import numpy as np

from hs_observatory.fields.malara import MalaraConfig, MalaraTurbulence


def _evaluator() -> MalaraTurbulence:
    return MalaraTurbulence(
        MalaraConfig(
            domain_size=(2.0, 3.0, 5.0),
            n_scales=4,
            injection_scale_index=2,
            intermittency_p=0.7,
            base_seed=2468,
            realization=3,
        )
    )


def test_cached_evaluation_matches_reference_and_preserves_shapes() -> None:
    evaluator = _evaluator()
    flat = np.random.default_rng(11).uniform(-2.0, 5.0, size=(20, 3))
    trajectory = flat[:18].reshape(2, 9, 3)

    np.testing.assert_allclose(
        evaluator.evaluate_vector_cached(flat),
        evaluator.evaluate_vector(flat),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    cached_trajectory = evaluator.evaluate_vector_cached(trajectory)
    reference_trajectory = evaluator.evaluate_vector(trajectory)
    assert cached_trajectory.shape == (2, 9, 3)
    np.testing.assert_allclose(cached_trajectory, reference_trajectory, rtol=1.0e-12, atol=1.0e-12)


def test_cache_hits_increase_and_clear_resets_cache() -> None:
    evaluator = _evaluator()
    positions = np.repeat(np.array([[0.37, 0.81, 1.23]]), 20, axis=0)

    evaluator.evaluate_vector_cached(positions)
    first = evaluator.cache_info()
    assert first["n_entries"] > 0
    assert first["hits"] > 0
    assert first["misses"] > 0
    assert first["hit_rate"] > 0.0

    evaluator.evaluate_vector_cached(positions)
    second = evaluator.cache_info()
    assert second["hits"] > first["hits"]
    assert second["n_entries"] == first["n_entries"]

    evaluator.clear_cache()
    assert evaluator.cache_info() == {
        "enabled": True,
        "n_entries": 0,
        "n_parameter_entries": 0,
        "hits": 0,
        "misses": 0,
        "hit_rate": 0.0,
    }


def test_parallel_matches_reference_across_chunks_and_worker_counts() -> None:
    evaluator = _evaluator()
    positions = np.random.default_rng(12).uniform(-3.0, 6.0, size=(12, 3))
    reference = evaluator.evaluate_vector(positions)

    one_worker = evaluator.evaluate_vector_parallel(
        positions, chunk_size=5, n_workers=1, use_cache=True
    )
    two_workers_small = evaluator.evaluate_vector_parallel(
        positions, chunk_size=5, n_workers=2, use_cache=True
    )
    two_workers_large = evaluator.evaluate_vector_parallel(
        positions, chunk_size=100, n_workers=2, use_cache=True
    )
    two_workers_uncached = evaluator.evaluate_vector_parallel(
        positions, chunk_size=5, n_workers=2, use_cache=False
    )

    for result in (
        one_worker,
        two_workers_small,
        two_workers_large,
        two_workers_uncached,
    ):
        np.testing.assert_allclose(result, reference, rtol=1.0e-12, atol=1.0e-12)


def test_cached_and_parallel_backends_preserve_periodicity() -> None:
    evaluator = _evaluator()
    point = np.array([0.31, 1.17, 4.2])
    domain = np.asarray(evaluator.config.domain_size)
    positions = np.vstack(
        [
            point,
            point + [domain[0], 0.0, 0.0],
            point + [0.0, domain[1], 0.0],
            point + [0.0, 0.0, domain[2]],
        ]
    )

    cached = evaluator.evaluate_vector_cached(positions)
    parallel = evaluator.evaluate_vector_parallel(positions, chunk_size=2, n_workers=2)
    expected = np.broadcast_to(cached[0], cached.shape)
    np.testing.assert_allclose(cached, expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(parallel, expected, rtol=1.0e-12, atol=1.0e-12)
