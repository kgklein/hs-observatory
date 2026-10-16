"""Small exhaustive checks for the truth-only grid diagnostic."""

import numpy as np

from hs_observatory.analysis.derived.geometry_pairs import build_geometry_pair_plan
from hs_observatory.analysis.structure_function_uniform_binning import (
    diagnostics,
    grids,
    separation_statistics,
    targets,
)
from hs_observatory.analysis.structure_function_local import local_log_slopes


def test_grid_coverage_and_half_width_edges():
    schemes = grids()
    assert [len(e) - 1 for e in schemes.values()] == [26, 40, 41]
    for edges in schemes.values():
        assert edges[0] == 70 and edges[-1] == 20000
        assert np.all(np.diff(edges) > 0)
        centers = np.sqrt(edges[:-1] * edges[1:])
        np.testing.assert_array_equal(targets(edges),
                                      np.where((centers >= 150) & (centers <= 1200),
                                               100000, 10000))
    step = np.diff(np.log(schemes['B']))[0]
    np.testing.assert_allclose(np.diff(np.log(schemes['B'])), step)
    np.testing.assert_allclose(np.diff(np.log(schemes['C'])),
                               np.r_[step / 2, np.repeat(step, 39), step / 2])


def test_population_and_sample_log_centers_against_exhaustive():
    positions = np.arange(5)[:, None, None] * np.array([[[2., 0., 0.]]])
    positions = positions + np.array([[[0., 0., 0.], [0., 3., 0.]]])
    edges = np.array([1., 3., 5., 10., 20.])
    plan = build_geometry_pair_plan(positions, edges)
    population, sample, counts = separation_statistics(plan, positions, plan.populations, 12345)
    flat = positions.reshape(-1, 3)
    a, b = np.triu_indices(len(flat), 1)
    distances = np.linalg.norm(flat[b] - flat[a], axis=1)
    for q in range(len(edges) - 1):
        mask = (distances >= edges[q]) & (distances < edges[q + 1])
        assert counts[q] == mask.sum()
        if counts[q]:
            expected = np.exp(np.mean(np.log(distances[mask])))
            np.testing.assert_allclose([population[q], sample[q]], expected, rtol=1e-14)
        else:
            assert np.isnan(population[q]) and np.isnan(sample[q])


def test_physical_fit_windows_use_representative_centers():
    centers = np.array([100., 160., 210., 400., 800., 1100., 1400.])
    values = centers[None, None, :] ** np.arange(1, 7)[None, :, None]
    product = {'lambda_bin_centers_km': centers, 'counts': np.ones((1, 7), dtype=int),
               'z_plus': {'S_n': values}}
    result = diagnostics(product, centers, centers)
    for entry in result.values():
        np.testing.assert_allclose(entry['fits']['raw'][0],
                                   np.tile(np.arange(1, 7), (2, 1)), atol=1e-13)
        np.testing.assert_array_equal(entry['fits']['bins_used'][0],
                                      np.tile([5, 3], (6, 1)).T)


def test_empirical_center_crossing_window_boundary_changes_mask():
    centers = np.array([100., 160., 210., 400., 800., 1100., 1400.])
    empirical = centers.copy()
    empirical[2] = 199.
    product = {'lambda_bin_centers_km': centers, 'counts': np.ones((1, 7), dtype=int),
               'z_plus': {'S_n': centers[None, None, :] ** np.arange(1, 7)[None, :, None]}}
    result = diagnostics(product, empirical, empirical)
    assert result['geometric']['fits']['bins_used'][0, 1, 0] == 3
    assert result['population']['fits']['bins_used'][0, 1, 0] == 2
    assert np.all(np.isnan(result['population']['fits']['raw'][0, 1]))


def test_unequal_spacing_does_not_create_power_law_slope_spike():
    centers = np.sqrt(grids()['A'][:-1] * grids()['A'][1:])
    values = centers[None, None, :] ** np.arange(1, 7)[None, :, None]
    _, slopes, _ = local_log_slopes(centers, values)
    np.testing.assert_allclose(slopes[0, :, 1:-1],
                               np.repeat(np.arange(1, 7)[:, None], len(centers) - 2, axis=1),
                               atol=1e-12)
