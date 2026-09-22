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
        assert len(formation.lines) == 8
        nodes = [c for c in formation.collections if hasattr(c, "_offsets3d")]
        relative = spacecraft.position[:, 0] - spacecraft.position[0, 0]
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
            assert axes[key].get_xlabel() == old[key].get_xlabel()
            assert axes[key].get_ylabel() == old[key].get_ylabel()
            for new_points, old_points in zip(axes[key].collections, old[key].collections):
                np.testing.assert_allclose(new_points.get_offsets(), old_points.get_offsets())
        figure.canvas.draw()
    finally:
        plt.close(original)
        plt.close(figure)


@pytest.mark.parametrize("variant", ["polyhedra", "formation"])
def test_degenerate_formation_renders_without_scipy(monkeypatch, variant):
    real_import = builtins.__import__

    def no_scipy(name, *args, **kwargs):
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError("SciPy unavailable for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_scipy)
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
