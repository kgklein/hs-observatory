"""Focused exhaustive and statistical checks for affine lag sampling."""

import h5py
import numpy as np
import pytest

from hs_observatory.analysis.derived.geometry_pairs import (
    GeometryPlanCache,
    build_geometry_pair_plan,
    geometry_lambda_bounds,
    geometry_pair_chunks,
)
from hs_observatory.analysis.derived.pairs import pair_chunks
from hs_observatory.analysis.structure_function import (
    StructureFunctionAnalysis,
    _bin_indices,
    _settings,
    compute_all_spacetime_structure_functions,
)


def positions(n_time=7, step=(1.0, 0.5, 0.0)):
    base = np.array([[0.0, 0.0, 0.0], [0.5, 1.0, 0.0], [-2.0, 0.0, 1.0]])
    return base[None] + np.arange(n_time)[:, None, None] * np.array(step)


def sampling(mode="geometry_stratified", target=8, chunk=17, seed=7):
    return dict(
        mode=mode,
        target_pairs_per_bin=target,
        random_seed=seed,
        chunk_size=chunk,
        n_pairs=None,
        max_pairs=None,
    )


def exhaustive_bins(p, edges, lower=None, upper=None):
    flat = p.reshape(-1, 3)
    a, b = next(pair_chunks(len(flat), len(flat) ** 2))
    d = np.linalg.norm(flat[b] - flat[a], axis=1)
    q, valid = _bin_indices(d, np.array(edges))
    valid &= d > 0
    if lower is not None:
        valid &= d >= lower
    if upper is not None:
        valid &= d <= upper
    return [set(zip(a[valid & (q == i)], b[valid & (q == i)])) for i in range(len(edges) - 1)]


def selected(plan, target, seed=7, chunk=17):
    chunks = list(geometry_pair_chunks(plan, target, random_seed=seed, chunk_size=chunk))
    return np.concatenate([np.column_stack(c) for c in chunks]) if chunks else np.empty((0, 2))


@pytest.mark.parametrize("p", [positions(), positions(1), positions(step=(0, 0, 0))])
@pytest.mark.parametrize("limits", [(None, None), (1.0, 4.0)])
def test_populations_and_census_identities_match_exhaustive(p, limits):
    edges = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 100.0])
    lower, upper = limits
    reference = exhaustive_bins(p, edges, lower, upper)
    plan = build_geometry_pair_plan(p, edges, min_pair_lambda_km=lower, max_pair_lambda_km=upper)
    np.testing.assert_array_equal(plan.populations, list(map(len, reference)))
    pairs = selected(plan, 100000)
    assert set(map(tuple, pairs)) == set.union(*reference)
    assert len(pairs) == len(set(map(tuple, pairs)))
    assert np.all(pairs[:, 0] < pairs[:, 1])
    if len(p) > 1 and np.any(np.diff(p, axis=0)):
        groups = np.concatenate([b.groups for b in plan.bins])
        assert np.any(groups[:, 2] < 0) and np.any(groups[:, 2] > 0)
        assert np.any((groups[:, 0] != groups[:, 1]) & (groups[:, 2] == 0))
        assert np.any(groups[:, 0] == groups[:, 1])


def test_boundary_roundoff_uses_stored_coordinates_and_split_starts():
    # Exact affine construction plus permitted floating-point offset roundoff.
    p = positions(9, step=(0.1, 0.0, 0.0)) + 100000.0
    p[:, 1, 0] += 0.03
    flat = p.reshape(-1, 3)
    edge = np.linalg.norm(flat[7] - flat[0])
    edges = np.array([0.0, np.nextafter(edge, 0), edge, np.nextafter(edge, np.inf), 10.0])
    plan = build_geometry_pair_plan(p, edges)
    expected = exhaustive_bins(p, edges)
    np.testing.assert_array_equal(plan.populations, list(map(len, expected)))
    assert plan.boundary_pairs_examined > 0
    assert any(b.explicit_starts for b in plan.bins)
    assert set(map(tuple, selected(plan, 10000))) == set.union(*expected)


def test_uniform_identity_selection_weights_unequal_lag_multiplicities():
    p = positions(4)
    plan = build_geometry_pair_plan(p, [0.0, 100.0])
    population = exhaustive_bins(p, [0.0, 100.0])[0]
    assert len(np.unique(plan.bins[0].groups[:, 4])) > 1
    frequency = dict.fromkeys(population, 0)
    trials, target = 1024, 5
    for seed in range(trials):
        pairs = selected(plan, target, seed)
        assert len(set(map(tuple, pairs))) == target
        for pair in map(tuple, pairs):
            frequency[pair] += 1
    probability = target / len(population)
    sigma = np.sqrt(trials * probability * (1 - probability))
    assert max(abs(count - trials * probability) for count in frequency.values()) < 6 * sigma


def test_streams_independent_of_chunking_order_and_other_bin_targets():
    p = positions(8)
    edges = [0.01, 2.0, 4.0, 20.0]
    plan = build_geometry_pair_plan(p, edges)
    targets = [7, 11, 13]
    np.testing.assert_array_equal(
        selected(plan, targets, chunk=1), selected(plan, targets, chunk=100)
    )
    # Increasing a different bin's budget cannot move this bin's stream.
    np.testing.assert_array_equal(
        selected(plan, targets)[7:18], selected(plan, [10, 11, 13])[10:21]
    )
    moved = build_geometry_pair_plan(p + np.array([0.0, 0.0, 2.0]), edges)
    np.testing.assert_array_equal(selected(plan, targets), selected(moved, targets))


def test_full_moments_and_multiseed_sample_means_match_exhaustive():
    p = positions(6)[None]
    z = np.random.default_rng(19).normal(size=p.shape)
    edges = [0.01, 2.0, 4.0, 20.0, 100.0]
    orders = list(range(1, 7))

    def run(settings):
        return compute_all_spacetime_structure_functions(
            p, z, -2 * z, orders, edges, pair_sampling=settings, min_fit_bins=2
        )

    reference = run(sampling("all"))
    census = run(sampling(target=[10000] * 4))
    np.testing.assert_array_equal(census["counts"], reference["counts"])
    for field in ("z_plus", "z_minus"):
        np.testing.assert_allclose(census[field]["S_n"], reference[field]["S_n"], equal_nan=True)
        np.testing.assert_allclose(
            census[field]["scaling_exponents"]["raw"],
            reference[field]["scaling_exponents"]["raw"],
            equal_nan=True,
        )
    samples = np.array(
        [run(sampling(target=5, seed=seed))["z_plus"]["S_n"][0, :, :3] for seed in range(384)]
    )
    se = samples.std(axis=0, ddof=1) / np.sqrt(len(samples))
    assert np.all(abs(samples.mean(axis=0) - reference["z_plus"]["S_n"][0, :, :3]) < 5 * se)
    d = census["spacetime_pair_counts"]["geometry_sampling"]["members"][0]
    np.testing.assert_array_equal(d["eligible_per_bin"], reference["counts"][0])
    assert d["eligible_per_bin"][-1] == 0
    assert d["population_shortfall_per_bin"][-1] == 10000
    assert np.isnan(census["z_plus"]["S_n"][0, :, -1]).all()


def test_position_offsets_cache_and_mc_order():
    p = positions()[None]
    z = np.random.default_rng(8).normal(size=p.shape)
    shifted = p.copy()
    shifted[:, :, 1] += [2.0, 1.0, 0.0]
    many = np.concatenate([p, shifted, p])
    values = np.repeat(z, 3, axis=0)
    edges = [0.01, 2.0, 4.0, 20.0]
    cache = GeometryPlanCache()
    result = compute_all_spacetime_structure_functions(
        many,
        values,
        values,
        [2, 4],
        edges,
        pair_sampling=sampling(target=5),
        geometry_plan_cache=cache,
    )
    metadata = result["spacetime_pair_counts"]["geometry_sampling"]["members"]
    assert [m["plan_reused"] for m in metadata] == [False, False, True]
    for m, coords in enumerate(many):
        np.testing.assert_array_equal(
            metadata[m]["eligible_per_bin"], list(map(len, exhaustive_bins(coords, edges)))
        )
    order = np.array([1, 2, 0])
    reordered = compute_all_spacetime_structure_functions(
        many[order],
        values[order],
        values[order],
        [2, 4],
        edges,
        pair_sampling=sampling(target=5, chunk=1),
        geometry_plan_cache=cache,
    )
    np.testing.assert_allclose(
        result["z_plus"]["S_n"], reordered["z_plus"]["S_n"][np.argsort(order)]
    )


def test_non_affine_geometry_fails_and_automatic_bounds_cover_population():
    p = positions()
    p[3, 1, 0] += 0.01
    with pytest.raises(ValueError, match="common-velocity affine"):
        build_geometry_pair_plan(p, [0.0, 20.0])
    p = positions()
    low, high = geometry_lambda_bounds(p, 1.0, 4.0)
    reference = exhaustive_bins(p, [1.0, 4.0])[0]
    flat = p.reshape(-1, 3)
    actual = [np.linalg.norm(flat[b] - flat[a]) for a, b in reference]
    assert low <= min(actual) and high >= max(actual)


@pytest.mark.parametrize("targets", [None, [1], [-1, 2], [1.0, 2.0], [True, False]])
def test_invalid_config_targets(targets):
    with pytest.raises(ValueError, match="target"):
        _settings(
            dict(
                increment_mode="all_spacetime_pairs",
                lambda_bins_km=[0.1, 1.0, 10.0],
                pair_sampling=dict(mode="geometry_stratified", target_pairs_per_bin=targets),
            )
        )


def test_hdf5_analysis_reconstructs_member_offsets_and_serializes(tmp_path):
    from test_structure_function import _write_structure_input

    path = _write_structure_input(tmp_path / "input.h5")
    config = dict(
        increment_mode="all_spacetime_pairs",
        lambda_bins_km=[0.1, 2.0, 5.0, 50.0],
        min_order=2,
        max_order=4,
        pair_sampling=sampling(target=1000),
    )
    result = StructureFunctionAnalysis(config).run(path)
    expected = StructureFunctionAnalysis(dict(config, pair_sampling=sampling("all"))).run(path)
    for group in ["truth", "observed"]:
        np.testing.assert_array_equal(
            result["products"][group]["counts"], expected["products"][group]["counts"]
        )
        np.testing.assert_allclose(
            result["products"][group]["z_plus"]["S_n"],
            expected["products"][group]["z_plus"]["S_n"],
            equal_nan=True,
        )
    assert result["products"]["observed"]["counts"][0] == [0, 16, 32]
    assert result["products"]["truth"]["counts"][0] == [16, 32, 0]
    out = StructureFunctionAnalysis(config).write_result(result, tmp_path / "result.json")
    assert '"algorithm_version": "geometry_stratified_v1"' in out.read_text()
    # Also exercise automatic geometry bounds through the full workflow.
    del config["lambda_bins_km"]
    automatic = StructureFunctionAnalysis(dict(config, n_lambda_bins=3)).run(path)
    assert len(automatic["products"]["truth"]["counts"][0]) == 3
    with h5py.File(path, "a") as f:
        f["spacecraft/r/v"][1, 1, 0] += 0.1
    with pytest.raises(ValueError, match="common-velocity affine"):
        StructureFunctionAnalysis(config).run(path)
