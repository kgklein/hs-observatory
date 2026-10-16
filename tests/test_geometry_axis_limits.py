"""Fixed movie ranges, including draw-time aspect and colorbar behavior."""

from dataclasses import replace
from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pytest

from hs_observatory import geometry_movie as movie
from hs_observatory import visualization as visual
from hs_observatory.spacecraft.configuration import synthetic_configuration


@pytest.fixture
def snapshots():
    spacecraft = synthetic_configuration(np.array([0.]))
    small = spacecraft.position[:, 0].copy()
    large = small * [-40., 65., 90.] + [130000., -170000., 20000.]
    return spacecraft.spacecraft_ids, [small, large]


def panel_axes(figure):
    return {(axis.get_subplotspec().rowspan.start, axis.get_subplotspec().colspan.start): axis
            for axis in figure.axes if axis.get_subplotspec() is not None}


def axis_ranges(figure):
    return [(axis.get_xlim(), axis.get_ylim(), axis.get_zlim() if axis.name == '3d' else None)
            for axis in figure.axes]


@pytest.mark.parametrize('variant', ['summary', 'polyhedra', 'formation'])
def test_all_panel_and_colorbar_limits_identical_after_draw(snapshots, variant):
    ids, samples = snapshots
    context = np.stack(samples).swapaxes(0, 1)
    fixed = visual.geometry_axis_limits(iter(samples), context_trajectory=context)
    ranges = []
    for sample in samples:
        figure, _ = visual.plot_geometry_summary(sample, ids, variant=variant,
            context_trajectory=context, axis_limits=fixed)
        try:
            figure.set_dpi(32)
            figure.canvas.draw()
            ranges.append(axis_ranges(figure))
            axes = panel_axes(figure)
            orbit = axes[0, 1 if variant == 'polyhedra' else 3]
            assert orbit.get_xlim() == fixed.context_x
            assert orbit.get_ylim() == fixed.context_y
            assert orbit.get_aspect() == 1 and orbit.get_adjustable() == 'box'
            if variant != 'formation':
                for key in ((0, 0), (1, 0), (1, 1)):
                    axis = axes[key]
                    assert axis.get_xlim() == axis.get_ylim() == (-fixed.spatial_extent, fixed.spatial_extent)
                    assert axis.get_aspect() == 1 and axis.get_adjustable() == 'box'
                    assert all(-fixed.spatial_extent <= tick <= fixed.spatial_extent for tick in axis.get_xticks())
                poly = axes[0, 2 if variant == 'polyhedra' else 1]
                assert poly.get_xlim() == fixed.polyhedron_quality
                assert poly.get_ylim() == fixed.polyhedron_size
                assert poly.get_yscale() == 'log'
                metrics = visual.polyhedron_metrics(sample)
                assert max(record['Q'] for record in metrics) <= poly.get_xlim()[1]
                sizes = [max(record['L'], 1e-9) for record in metrics]
                assert min(sizes) >= poly.get_ylim()[0] and max(sizes) <= poly.get_ylim()[1]
            if variant != 'polyhedra':
                for key in ((0, 2), (1, 2), (1, 3)):
                    axis = axes[key]
                    assert axis.get_xlim() == axis.get_ylim() == fixed.baseline_components
                    assert axis.get_xscale() == axis.get_yscale() == 'log'
                positive = np.abs(visual.unique_baselines(sample))
                positive = positive[positive >= fixed.baseline_components[0]]
                assert positive.max() <= fixed.baseline_components[1]
            if variant == 'formation':
                axis = axes[0, 0]
                bounds = [axis.get_xlim(), axis.get_ylim(), axis.get_zlim()]
                np.testing.assert_array_equal(bounds, fixed.formation_bounds)
                spans = np.ptp(bounds, axis=1)
                ratio = axis.get_box_aspect() / spans
                np.testing.assert_allclose(ratio, np.repeat(ratio[0], 3))
                relative = sample - sample[0]
                assert np.all(relative >= np.asarray(bounds)[:, 0])
                assert np.all(relative <= np.asarray(bounds)[:, 1])
                np.testing.assert_allclose(figure.axes[-1].get_xlim(), fixed.baseline_colors)
            elif variant == 'polyhedra':
                shape = axes[1, 2]
                assert shape.get_xlim() == shape.get_ylim() == fixed.shape
                assert (shape.collections[0].norm.vmin, shape.collections[0].norm.vmax) == fixed.polyhedron_colors
        finally:
            plt.close(figure)
    assert ranges[0] == ranges[1]  # Includes every colorbar axis, as well as the main panels.


def test_scan_no_figures_global_data_containment_and_repeated_records(snapshots, monkeypatch):
    _, samples = snapshots
    context = np.stack(samples).swapaxes(0, 1)
    monkeypatch.setattr(plt, 'figure', Mock(side_effect=AssertionError('first pass created a figure')))
    real_metrics = visual.polyhedron_metrics
    metrics = Mock(wraps=real_metrics)
    monkeypatch.setattr(visual, 'polyhedron_metrics', metrics)
    frames = [{'selected_index': index, 'position': sample} for index, sample in enumerate(samples)]
    fixed = movie.movie_axis_limits([frames[0], frames[0], frames[1]], context)
    assert metrics.call_count == 2
    baseline_lengths = np.concatenate([np.linalg.norm(visual.unique_baselines(sample), axis=1) for sample in samples])
    sizes = [record['L'] for sample in samples for record in real_metrics(sample)]
    assert fixed.baseline_colors[0] <= min(baseline_lengths) <= max(baseline_lengths) <= fixed.baseline_colors[1]
    assert fixed.polyhedron_colors[0] <= min(sizes) <= max(sizes) <= fixed.polyhedron_colors[1]
    assert fixed.spatial_extent >= max(abs(sample - sample[0]).max() for sample in samples)
    centers = context.mean(axis=0) / visual.EARTH_RADIUS_KM
    assert fixed.context_x[0] <= min(-1., centers[:, 0].min())
    assert fixed.context_y[1] >= max(1., centers[:, 1].max())
    single = visual.geometry_axis_limits([samples[0]], context_trajectory=context)
    repeated = visual.geometry_axis_limits([samples[0], samples[0]], context_trajectory=context)
    assert single == repeated
    assert single.spatial_extent < fixed.spatial_extent
    assert visual._geometry_log_range(2., 2.) == (2 / np.sqrt(10), 2 * np.sqrt(10))


@pytest.mark.parametrize('variant', ['summary', 'polyhedra', 'formation'])
def test_coincident_frames_keep_axes_and_colorbars(snapshots, variant):
    ids, samples = snapshots
    zero = np.zeros((9, 3))
    context = np.stack([samples[0], zero]).swapaxes(0, 1)
    fixed = visual.geometry_axis_limits([samples[0], zero], context_trajectory=context)
    ranges = []
    for sample in (samples[0], zero):
        figure, _ = visual.plot_geometry_summary(sample, ids, variant=variant,
            context_trajectory=context, axis_limits=fixed)
        try:
            figure.set_dpi(32)
            figure.canvas.draw()
            ranges.append(axis_ranges(figure))
            if sample is zero and variant != 'summary':
                text = [text.get_text() for axis in figure.axes for text in axis.texts]
                labels = [axis.get_xlabel() for axis in figure.axes]
                assert any('No positive L' in value or 'no positive lengths' in value for value in text + labels)
        finally:
            plt.close(figure)
    assert ranges[0] == ranges[1]
    all_zero = visual.geometry_axis_limits([zero], context_trajectory=context)
    assert all_zero.polyhedron_colors == all_zero.baseline_colors == (1., 10.)
    assert not all_zero.positive_polyhedron_lengths and not all_zero.positive_baseline_lengths
    # Reuse the all-zero fallback in the real renderer, not just in the calculator.
    figure, _ = visual.plot_geometry_summary(zero, ids, variant=variant,
        context_trajectory=context, axis_limits=all_zero)
    try:
        figure.set_dpi(32)
        figure.canvas.draw()
        if variant == 'formation':
            assert figure.axes[-1].get_xlim() == (1., 10.)
        elif variant == 'polyhedra':
            assert figure.axes[-1].get_ylim() == (1., 10.)
    finally:
        plt.close(figure)


def test_constant_positive_ranges_and_configuration_validation(snapshots, monkeypatch):
    _, samples = snapshots
    monkeypatch.setattr(visual, 'polyhedron_metrics', lambda sample: [{'L': 2.}])
    monkeypatch.setattr(visual, 'unique_baselines', lambda sample: np.array([[1., 0., 0.]]))
    fixed = visual.geometry_axis_limits([samples[0]], context_trajectory=samples[0])
    np.testing.assert_allclose(fixed.polyhedron_colors, (2 / np.sqrt(10), 2 * np.sqrt(10)))
    np.testing.assert_allclose(fixed.baseline_colors, (1 / np.sqrt(10), np.sqrt(10)))
    with pytest.raises(ValueError, match='logarithmic'):
        replace(fixed, baseline_colors=(0., 1.))
    with pytest.raises(ValueError, match='increasing'):
        replace(fixed, context_x=(1., 1.))
    with pytest.raises(ValueError, match='at least one'):
        visual.geometry_axis_limits([], context_trajectory=samples[0])


def test_static_call_does_not_scan_or_share_limits(snapshots, monkeypatch):
    ids, samples = snapshots
    monkeypatch.setattr(visual, 'geometry_axis_limits', Mock(side_effect=AssertionError('static scan')))
    bounds = []
    for sample in samples:
        figure, _ = visual.plot_geometry_summary(sample, ids, variant='formation')
        try:
            bounds.append(figure.axes[0].get_xlim())
        finally:
            plt.close(figure)
    assert bounds[0] != bounds[1]
