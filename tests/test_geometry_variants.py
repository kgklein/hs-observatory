"""Layout and data checks for the additional geometry summary views."""

import builtins

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pytest

from hs_observatory.spacecraft.configuration import synthetic_configuration
from hs_observatory.visualization import (
    SPACECRAFT_COLORS,
    geometry_summary_paths,
    plot_geometry_summary,
    polyhedron_metrics,
    save_geometry_summary,
)


def panels(figure):
    return {
        (spec.rowspan.start, spec.colspan.start): axis
        for axis in figure.axes
        if (spec := axis.get_subplotspec()) is not None
    }


def test_polyhedra_layout_and_independent_color_scales():
    spacecraft = synthetic_configuration(np.array([0.0, 1.0]))
    figure, diagnostics = plot_geometry_summary(
        spacecraft.position, spacecraft.spacecraft_ids, variant="polyhedra"
    )
    try:
        axes = panels(figure)
        assert set(axes) == {(r, c) for r in range(2) for c in range(3)}
        assert axes[0, 1].get_title() == ""
        assert "R_E" in axes[0, 1].get_xlabel()
        assert axes[0, 2].get_title() == "Polyhedral Configuration Summary"
        assert "Delta Z" in axes[1, 0].get_title()
        assert "Delta Y" in axes[1, 1].get_title()
        assert axes[1, 2].get_xlabel() == "Elongation E"
        assert axes[1, 2].get_ylabel() == "Planarity P"
        summary = axes[0, 2].collections[0]
        shape = axes[1, 2].collections[0]
        assert isinstance(summary.norm, colors.BoundaryNorm)
        assert isinstance(shape.norm, colors.LogNorm)
        assert summary.colorbar.ax.get_ylabel() == "# spacecraft"
        assert shape.colorbar.ax.get_ylabel() == "L (km)"
        assert summary.colorbar is not shape.colorbar
        metrics = polyhedron_metrics(spacecraft.position[:, 0])
        np.testing.assert_allclose(shape.get_offsets(), [[m["E"], m["P"]] for m in metrics])
        np.testing.assert_allclose(shape.get_array(), [m["L"] for m in metrics])
        assert diagnostics["n_polyhedra"] == 382
        for label, key in zip("abcdef", sorted(axes)):
            assert f"({label})" in [t.get_text() for t in axes[key].texts]
        figure.canvas.draw()
    finally:
        plt.close(figure)


def test_formation_colors_coordinates_and_preserved_right_panels():
    spacecraft = synthetic_configuration(np.array([0.0, 1.0]))
    original, _ = plot_geometry_summary(spacecraft.position, spacecraft.spacecraft_ids)
    figure, _ = plot_geometry_summary(
        spacecraft.position, spacecraft.spacecraft_ids, variant="formation"
    )
    try:
        axes, old = panels(figure), panels(original)
        assert set(axes) == {(0, 0), (0, 2), (0, 3), (1, 2), (1, 3)}
        formation = axes[0, 0]
        spec = formation.get_subplotspec()
        assert list(spec.rowspan) == [0, 1] and list(spec.colspan) == [0, 1]
        assert formation.name == "3d"
        assert not formation.child_axes  # No orbit inset in the 3D panel.
        nodes = [c for c in formation.collections if hasattr(c, "_offsets3d")]
        relative = spacecraft.position[:, 0] - spacecraft.position[0, 0]
        hub_connections = [
            np.column_stack(line.get_data_3d())
            for line in formation.lines
            if np.allclose(np.column_stack(line.get_data_3d())[0], relative[0])
        ]
        assert len(hub_connections) == 8
        for point in relative[1:]:
            assert sum(np.allclose(connection[1], point) for connection in hub_connections) == 1
        for node, point, name in zip(nodes, relative, spacecraft.spacecraft_ids):
            np.testing.assert_allclose(np.asarray(node._offsets3d).ravel(), point)
            np.testing.assert_allclose(
                node.get_facecolors()[0], colors.to_rgba(SPACECRAFT_COLORS[name])
            )
        assert len(nodes) == 9
        for key in ((0, 2), (0, 3), (1, 2), (1, 3)):
            assert axes[key].get_title() == old[key].get_title()
            assert axes[key].get_xlim() == old[key].get_xlim()
            # Equal-aspect orbit limits adapt to panel size; lambda limits do not.
            if key != (0, 3):
                assert axes[key].get_ylim() == old[key].get_ylim()
            assert axes[key].get_xlabel().replace(r"\mathrm{GSE}", "GSE") == old[key].get_xlabel()
            assert axes[key].get_ylabel().replace(r"\mathrm{GSE}", "GSE") == old[key].get_ylabel()
            for new_points, old_points in zip(axes[key].collections, old[key].collections):
                np.testing.assert_allclose(new_points.get_offsets(), old_points.get_offsets())
        figure.canvas.draw()
    finally:
        plt.close(original)
        plt.close(figure)


@pytest.mark.parametrize("variant", ["polyhedra", "formation"])
def test_degenerate_formation_renders(variant):
    figure, _ = plot_geometry_summary(
        np.zeros((9, 3)), ["H", *[f"N{i}" for i in range(1, 9)]], variant=variant
    )
    try:
        figure.canvas.draw()
    finally:
        plt.close(figure)


def test_save_writes_all_variants_and_preserves_original_return(tmp_path):
    spacecraft = synthetic_configuration(np.array([0.0, 1.0]))
    output = tmp_path / "sampled_observatory.h5"
    metadata = {"selected_time": "2030-08-25T11:50:35"}
    pdf, png, diagnostics = save_geometry_summary(
        output_path=output,
        config={"dpi": 30},
        time=spacecraft.time,
        position=spacecraft.position,
        spacecraft_ids=spacecraft.spacecraft_ids,
        metadata=metadata,
    )
    assert (pdf, png) == geometry_summary_paths(output, metadata)
    assert set(diagnostics["variant_paths"]) == {"summary", "polyhedra", "formation"}
    for variant in diagnostics["variant_paths"]:
        for path in geometry_summary_paths(output, metadata, variant=variant):
            assert path.is_file() and path.stat().st_size > 0
    assert len(list((tmp_path / "geometry_overviews").iterdir())) == 6


def test_publication_selection_matches_gradient_analysis():
    from hs_observatory.analysis.magnetic_gradient import select_configurations
    from hs_observatory.visualization import _selected_plot_polyhedra

    # Deterministic nondegenerate configuration with regular tetrahedra.
    points = np.random.default_rng(17).normal(size=(9, 3)) * 300
    catalog = polyhedron_metrics(points)
    chosen = _selected_plot_polyhedra(catalog)
    analysis = select_configurations(
        points,
        enabled={
            "small": True,
            "best_multi": True,
            "all_nine": True,
            "intermediate": False,
            "large": False,
        },
    )
    for label, key in (
        ("smallest regular", "small"),
        ("most regular N>4", "best_multi"),
        ("all N=9", "all_nine"),
    ):
        assert chosen[label]["indices"] == tuple(analysis[key]["indices"])
    assert chosen["smallest regular"]["Q"] <= 0.6
    assert 4 < chosen["most regular N>4"]["n_spacecraft"] < 9


@pytest.mark.parametrize("variant", ["formation", "polyhedra"])
def test_publication_aspect_boundaries_and_highlights(variant):
    spacecraft = synthetic_configuration(np.array([0.0, 1.0]))
    figure, diagnostics = plot_geometry_summary(
        spacecraft.position, spacecraft.spacecraft_ids, variant=variant
    )
    try:
        axes = panels(figure)
        assert "2030" not in figure._suptitle.get_text()
        assert figure._suptitle.get_text() == "HelioSwarm Design Reference Geometry"
        epoch_text = next(text for text in figure.texts if text.get_text().startswith("Epoch:"))
        assert epoch_text.get_fontsize() == figure._suptitle.get_fontsize() - 3
        context = axes[0, 1 if variant == "polyhedra" else 3]
        assert context.get_aspect() == 1 and context.get_adjustable() == "box"
        if variant == "formation":
            formation = axes[0, 0]
            spans = np.array(
                [
                    np.ptp(formation.get_xlim()),
                    np.ptp(formation.get_ylim()),
                    np.ptp(formation.get_zlim()),
                ]
            )
            ratio = formation.get_box_aspect() / spans
            np.testing.assert_allclose(ratio, np.full(3, ratio[0]))
            limits = []
            for key in ((0, 2), (1, 2), (1, 3)):
                ax = axes[key]
                assert ax.get_aspect() == 1
                assert ax.get_adjustable() == "box"
                assert ax.get_xscale() == ax.get_yscale() == "log"
                limits.extend([ax.get_xlim(), ax.get_ylim()])
                for boundary in (100, 1200):
                    assert any(
                        np.array_equal(line.get_xdata(), [boundary, boundary]) for line in ax.lines
                    )
                    assert any(
                        np.array_equal(line.get_ydata(), [boundary, boundary]) for line in ax.lines
                    )
            assert all(limit == limits[0] for limit in limits)
        else:
            physical = [axes[k] for k in ((0, 0), (1, 0), (1, 1))]
            for ax in physical:
                assert ax.get_aspect() == 1
                assert ax.get_xlim() == ax.get_ylim() == physical[0].get_xlim()
                np.testing.assert_array_equal(ax.get_xticks(), physical[0].get_xticks())
                np.testing.assert_array_equal(ax.get_yticks(), physical[0].get_xticks())
            shape = axes[1, 2]
            assert "regular region" not in [
                text.get_text() for text in axes[0, 2].get_legend().get_texts()
            ]
            assert shape.get_aspect() == axes[0, 2].get_aspect() == "auto"
            try:
                from cmcrameri import cm
            except ImportError:
                assert shape.collections[0].cmap.name == "cividis"
            else:
                samples = np.linspace(0.0, 1.0, 5)
                np.testing.assert_allclose(shape.collections[0].cmap(samples), cm.lajolla(samples))
            for axis in (shape, axes[0, 2]):
                np.testing.assert_array_equal(axis.collections[0].get_sizes(), [33])
            arc = shape.lines[0]
            np.testing.assert_allclose(
                np.square(arc.get_xdata()) + np.square(arc.get_ydata()), 0.6**2
            )
            for arc, radius in zip(shape.lines[1:], (0.4, 0.8)):
                np.testing.assert_allclose(
                    np.square(arc.get_xdata()) + np.square(arc.get_ydata()), radius**2
                )
                assert any(
                    np.array_equal(line.get_xdata(), [radius, radius]) for line in axes[0, 2].lines
                )
            assert len(shape.lines) == 3
            for label, record in diagnostics["selected_polyhedra"].items():
                for axis, expected in (
                    (shape, [record["E"], record["P"]]),
                    (axes[0, 2], [record["Q"], max(record["L"], 1e-9)]),
                ):
                    marker = next(c for c in axis.collections if c.get_label() == label)
                    np.testing.assert_allclose(marker.get_offsets(), [expected])
                    np.testing.assert_array_equal(marker.get_sizes(), [285])
    finally:
        plt.close(figure)


def test_length_colormap_falls_back_without_cmcrameri(monkeypatch):
    from hs_observatory.visualization import _polyhedron_length_cmap

    original_import = builtins.__import__

    def without_cmcrameri(name, *args, **kwargs):
        if name == "cmcrameri":
            raise ImportError("Unavailable for fallback test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_cmcrameri)
    assert _polyhedron_length_cmap().name == "cividis"


def test_quality_shading_uses_disjoint_bands_with_decreasing_opacity():
    from matplotlib.patches import Wedge
    from hs_observatory.visualization import _shade_polyhedron_quality

    figure, axes = plt.subplots(1, 2)
    try:
        for axis, shape, base in zip(axes, (False, True), (0.14, 0.06)):
            _shade_polyhedron_quality(axis, shape=shape, base_alpha=base)
            assert len(axis.patches) == 3
            np.testing.assert_allclose(
                [p.get_alpha() for p in axis.patches], np.array([1.5, 1, 0.5]) * base
            )
            for patch, low, high in zip(axis.patches, (0, 0.4, 0.6), (0.4, 0.6, 0.8)):
                np.testing.assert_allclose(patch.get_facecolor()[:3], colors.to_rgb("#009E73"))
                if shape:
                    assert isinstance(patch, Wedge)
                    assert patch.r == high
                    assert patch.width == high - low
                    assert (patch.theta1, patch.theta2) == (0, 90)
                else:
                    assert patch.get_x() == low
                    assert patch.get_width() == high - low
    finally:
        plt.close(figure)
