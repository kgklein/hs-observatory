"""Publication figure of nominal science-hour geometry from one native CDF record."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from hs_observatory.geometry_requirements import (
    REFERENCE_WITNESS,
    SPACECRAFT_IDS,
    BaselineCoverage,
    evaluate_baseline_coverage,
)
from hs_observatory.io.cdf_positions import (
    FILL_VALUE_THRESHOLD,
    read_cdf_positions,
    select_cdf_record,
)
from hs_observatory.visualization import (
    REGION_COLORS,
    REGULAR_POLYHEDRON_Q_LIMIT,
    _plot_context_projection,
    classify_gse_region,
    polyhedron_metrics,
    spacecraft_color,
    unique_baselines,
)

# Import visualization first to select the existing headless plotting backend.
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import proj3d


REFERENCE_EPOCH = "2030-08-25T11:50:35.801000000"


@dataclass(frozen=True)
class ScienceHourGeometry:
    """Shared native geometry and evidence consumed by all three panels."""

    requested_epoch: str
    matched_epoch: str
    selected_index: int
    source_file: str
    spacecraft_ids: tuple[str, ...]
    positions: np.ndarray  # (9, 3), absolute native GSE positions, km
    context_epochs: np.ndarray
    context_positions: np.ndarray  # mesocenters; invalid rows are NaN, not removed
    segment_validity: np.ndarray
    baselines: np.ndarray
    baseline_indices: tuple[tuple[int, int], ...]
    coverage: BaselineCoverage
    polyhedra: list[dict[str, Any]]
    small: dict[str, Any]
    large: dict[str, Any]
    scale_ratio: float

    def diagnostics(self) -> dict[str, Any]:
        def polyhedron(item):
            return {
                "members": [self.spacecraft_ids[i] for i in item["indices"]],
                "L_km": item["L"],
                "chi": item["Q"],
            }

        return {
            "requested_epoch": self.requested_epoch,
            "matched_epoch": self.matched_epoch,
            "selected_index": self.selected_index,
            "source_file": self.source_file,
            "spacecraft_ids": list(self.spacecraft_ids),
            "positions_km": self.positions.tolist(),
            "region": classify_gse_region(self.positions.mean(axis=0)),
            "n_baselines": len(self.baselines),
            "n_polyhedra": len(self.polyhedra),
            "n_acceptable_polyhedra": sum(
                m["Q"] <= REGULAR_POLYHEDRON_Q_LIMIT for m in self.polyhedra
            ),
            "category_order": self.coverage.category_labels,
            "category_counts_scale_by_quantity": self.coverage.counts.tolist(),
            "baseline_requirement_satisfied": self.coverage.satisfies_requirement,
            "witness_pairs": self.coverage.witness_pairs,
            "witness_covers_all_twelve_categories": bool(
                self.coverage.witness_category_covered.all()
            ),
            "small": polyhedron(self.small),
            "large": polyhedron(self.large),
            "scale_ratio": self.scale_ratio,
        }


def prepare_science_hour_geometry(
    epochs: np.ndarray,
    positions: np.ndarray,
    labels: Sequence[str],
    sources: Sequence[str],
    *,
    epoch: str,
    epoch_tolerance_s: float = 0.001,
    witness_pairs: Sequence[tuple[str, str]] | None = REFERENCE_WITNESS,
) -> ScienceHourGeometry:
    """Select one complete native record, preserving invalid context rows and gaps.

    Passing witness_pairs=None selects a deterministic automatic witness instead
    of the default reference witness. No synthetic time series is constructed.
    """
    epochs = np.asarray(epochs, dtype="datetime64[ns]")
    positions = np.asarray(positions, dtype=float)
    if epochs.ndim != 1 or not len(epochs) or np.any(np.isnat(epochs)):
        raise ValueError("Trajectory requires a nonempty finite epoch array")
    if np.any(np.diff(epochs) <= np.timedelta64(0, "ns")):
        raise ValueError("Trajectory epochs must be unique and strictly increasing")
    if len(labels) != 9 or set(labels) != set(SPACECRAFT_IDS):
        raise ValueError("Trajectory must identify H and N1 through N8 exactly once")
    if positions.shape != (len(epochs), 9, 3) or len(sources) != len(epochs):
        raise ValueError("Trajectory positions/source records do not align with epochs")
    if not np.isfinite(epoch_tolerance_s) or epoch_tolerance_s < 0:
        raise ValueError("epoch_tolerance_s must be finite and nonnegative")
    requested = np.datetime64(epoch, "ns")
    if np.isnat(requested) or requested < epochs[0] or requested > epochs[-1]:
        raise ValueError("Requested epoch is outside trajectory coverage")
    sample = select_cdf_record(epochs, positions, list(labels), list(sources), requested_time=epoch)
    offset = abs((epochs[sample["selected_index"]] - requested) / np.timedelta64(1, "s"))
    if offset > epoch_tolerance_s:
        raise ValueError(
            f"Requested epoch {epoch} matched {sample['selected_time']} "
            f"({offset:g} s offset), exceeding tolerance {epoch_tolerance_s:g} s"
        )
    order = [list(labels).index(label) for label in SPACECRAFT_IDS]
    selected = sample["selected_position"][order].copy()
    if not np.all(np.isfinite(selected) & (selected > FILL_VALUE_THRESHOLD)):
        raise ValueError("Selected record must contain nine finite, non-fill spacecraft positions")
    ordered = positions[:, order, :]
    valid = np.all(np.isfinite(ordered) & (ordered > FILL_VALUE_THRESHOLD), axis=(1, 2))
    context = np.full((len(epochs), 3), np.nan)
    context[valid] = ordered[valid].mean(axis=1)
    cadence_s = np.diff(epochs) / np.timedelta64(1, "s")
    segments = valid[:-1] & valid[1:]
    if len(cadence_s):
        segments &= cadence_s <= 1.5 * np.median(cadence_s)
    baseline_indices = tuple(combinations(range(9), 2))
    baselines = unique_baselines(selected)
    pairs = tuple((SPACECRAFT_IDS[i], SPACECRAFT_IDS[j]) for i, j in baseline_indices)
    coverage = evaluate_baseline_coverage(baselines, pairs, witness_pairs=witness_pairs)
    if not coverage.satisfies_requirement:
        missing = [
            name for name, ok in zip(coverage.category_labels, coverage.category_covered) if not ok
        ]
        raise ValueError(f"Configuration fails baseline coverage; missing categories: {missing}")
    metrics = polyhedron_metrics(selected)
    acceptable = sorted(
        (m for m in metrics if m["Q"] <= REGULAR_POLYHEDRON_Q_LIMIT and m["L"] > 0),
        key=lambda m: (m["L"], m["Q"], m["indices"]),
    )
    if len(acceptable) < 2 or acceptable[-1]["L"] / acceptable[0]["L"] < 3:
        raise ValueError("Configuration fails polyhedral multiscale ratio >= 3")
    return ScienceHourGeometry(
        requested_epoch=epoch,
        matched_epoch=sample["selected_time"],
        selected_index=sample["selected_index"],
        source_file=sample["source_file"],
        spacecraft_ids=SPACECRAFT_IDS,
        positions=selected,
        context_epochs=epochs.copy(),
        context_positions=context,
        segment_validity=segments,
        baselines=baselines,
        baseline_indices=baseline_indices,
        coverage=coverage,
        polyhedra=metrics,
        small=acceptable[0],
        large=acceptable[-1],
        scale_ratio=acceptable[-1]["L"] / acceptable[0]["L"],
    )


def load_science_hour_geometry(trajectory_path: str | Path, **kwargs) -> ScienceHourGeometry:
    """Read native CDFs once and prepare a shared qualifying record."""
    return prepare_science_hour_geometry(*read_cdf_positions(trajectory_path), **kwargs)


FIGURE_STYLE = {
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6,
    "axes.linewidth": 0.6,
}
BASELINE_VIEWS = {"A": (20, -65), "B": (25, -45), "C": (30, -60)}
DEFAULT_BASELINE_VIEW = BASELINE_VIEWS["A"]
HYBRID_SCALE_PAIRS = (("H", "N6"), ("H", "N1"), ("N2", "N4"))
HYBRID_SCALE_COLORS = ("#0072B2", "#009E73", "#D55E00")
HYBRID_VIEW = (25, -60)
PURPLE_SCALE_COLORS = ("#B39DDB", "#7E57C2", "#4527A0")
PURPLE_SCALE_PAIRS = (("H", "N6"), ("N1", "N5"), ("N2", "N4"))
PURPLE_VIEW = (25, -65)


def total_scale_examples(
    record: ScienceHourGeometry, pairs=HYBRID_SCALE_PAIRS, *, colors=HYBRID_SCALE_COLORS
):
    """Validate three display examples against existing D-category masks.

    Pairs are supplied in small/intermediate/large order. These total-separation
    examples are independent of the full twelve-category witness selection.
    """
    if len(pairs) != 3:
        raise ValueError("Hybrid display requires one pair per total-length scale (three pairs)")
    if len(colors) != 3:
        raise ValueError("Hybrid display requires three scale colors")
    examples = []
    for column, (pair, scale, interval, color) in enumerate(
        zip(
            pairs,
            ("Small", "Intermediate", "Large"),
            ("10–100", "100–1200", "1200–5000"),
            colors,
        )
    ):
        if len(pair) != 2 or any(name not in record.spacecraft_ids for name in pair):
            raise ValueError(f"Unknown representative baseline: {pair}")
        pair = tuple(sorted(pair, key=record.spacecraft_ids.index))
        if pair not in record.coverage.pairs:
            raise ValueError(f"Representative pair absent from baseline catalog: {pair}")
        index = record.coverage.pairs.index(pair)
        if not record.coverage.membership[index, 4 * column + 3]:
            raise ValueError(f"{pair} does not satisfy the {scale.lower()} total-length bin")
        examples.append(
            {
                "pair": pair,
                "scale": scale,
                "color": color,
                "length_km": float(np.linalg.norm(record.baselines[index])),
                "legend": f"{scale}: {interval} km",
            }
        )
    return tuple(examples)


def intermediate_projection_diagnostics(record: ScienceHourGeometry, axis):
    """Measure all D-intermediate candidates using the rendered camera and axes.

    Lengths/clearances are display points (1/72 inch), independent of save DPI.
    Interior marker clearances exclude the first/last 10% of the segment, where
    near-hub endpoints inevitably cluster. This is a layout diagnostic only.
    """
    axis.figure.canvas.draw()
    relative = record.positions - record.positions[0]
    projected = np.column_stack(proj3d.proj_transform(*relative.T, axis.get_proj())[:2])
    display = axis.transData.transform(projected) * 72 / axis.figure.dpi
    result = []
    for index in np.flatnonzero(record.coverage.membership[:, 7]):
        i, j = record.baseline_indices[index]
        start, stop = display[[i, j]]
        vector = stop - start
        length = float(np.linalg.norm(vector))
        clearances = []
        if length > 0:
            for k, point in enumerate(display):
                if k in (i, j):
                    continue
                along = float(np.dot(point - start, vector) / length**2)
                if 0.1 < along < 0.9:
                    distance = float(np.linalg.norm(point - (start + along * vector)))
                    # Same scatter-area hierarchy and outline used by this plot.
                    marker_radius = np.sqrt(18 if k == 0 else 10) / 2 + 0.15
                    clearances.append(
                        {
                            "spacecraft": record.spacecraft_ids[k],
                            "distance_pt": distance,
                            "overlaps_highlight": bool(distance <= marker_radius + 1.1),
                        }
                    )
        result.append(
            {
                "pair": record.coverage.pairs[index],
                "length_km": float(np.linalg.norm(record.baselines[index])),
                "projected_length_pt": length,
                "interior_markers": clearances,
                "minimum_marker_clearance_pt": min(
                    (c["distance_pt"] for c in clearances), default=None
                ),
            }
        )
    return result


def _baseline_highlights(record, mode, scale_pairs, colors=HYBRID_SCALE_COLORS):
    if mode == "witness":
        return [(pair, "#303030", 2.0) for pair in record.coverage.witness_pairs]
    if mode == "hybrid":
        return [
            (item["pair"], item["color"], 2.2)
            for item in total_scale_examples(record, scale_pairs, colors=colors)
        ]
    raise ValueError("baseline_mode must be witness or hybrid")


def near_hub_window(record: ScienceHourGeometry) -> tuple[np.ndarray, np.ndarray]:
    """Return a display-only near-hub group and square X-Y bounds in km.

    Split at the largest radial distance ratio (at least three) between adjacent
    non-hub spacecraft. This selects a magnification window, never science-hour
    eligibility. Bounds enclose the selected positions with 20% padding per side.
    """
    relative = record.positions - record.positions[0]
    distances = np.linalg.norm(relative, axis=1)
    ordered = np.argsort(distances[1:], kind="stable") + 1
    positive = ordered[distances[ordered] > 0]
    if len(positive) < 2:
        raise ValueError("No separated near-hub group is available for an inset")
    ratios = distances[positive[1:]] / distances[positive[:-1]]
    split = int(np.argmax(ratios))
    if ratios[split] < 3:
        raise ValueError("No distinct near-hub radial gap (ratio >= 3) for an inset")
    indices = np.flatnonzero(distances <= distances[positive[split]])
    points = relative[indices, :2]
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    half_width = 0.7 * float(np.ptp(points, axis=0).max())
    if half_width == 0:
        raise ValueError("Near-hub group has no X-Y extent for a projection inset")
    return indices, np.column_stack((center - half_width, center + half_width))


def _draw_hub_inset(
    axis,
    record: ScienceHourGeometry,
    *,
    highlights=None,
    baseline_style="classic",
    inset_variant="ticks",
    refined=False,
):
    indices, bounds = near_hub_window(record)
    relative = record.positions - record.positions[0]
    purple = baseline_style == "purple"
    inset = axis.inset_axes([0.57, 0.68, 0.38, 0.34] if purple else [0.62, 0.72, 0.38, 0.34])
    inset.set_facecolor("white")
    for i, j in combinations(indices, 2):
        inset.plot(*relative[[i, j], :2].T, color="#B8B8B8", alpha=0.4, lw=0.6)
    included = set(indices)
    if highlights is None:
        highlights = _baseline_highlights(record, "witness", HYBRID_SCALE_PAIRS)
    for (first, second), color, width in highlights:
        i, j = record.spacecraft_ids.index(first), record.spacecraft_ids.index(second)
        if i in included and j in included:
            inset.plot(*relative[[i, j], :2].T, color=color, lw=1.6 if purple else 0.6 * width)
    for i in indices:
        name = record.spacecraft_ids[i]
        inset.scatter(
            *relative[i, :2],
            s=18 if name == "H" else 10,
            color=spacecraft_color(name),
            edgecolors="black",
            linewidths=0.3,
            zorder=4,
        )
        inset.annotate(
            name, relative[i, :2], xytext=(3, 3), textcoords="offset points", fontsize=5, zorder=5
        )
    inset.set(xlim=bounds[0], ylim=bounds[1])
    inset.set_aspect("equal", adjustable="box")
    inset.set_title("Hub region: X–Y (km)", fontsize=5.5, pad=3)
    inset.tick_params(labelsize=5, length=2, pad=1)
    inset.xaxis.set_major_locator(MaxNLocator(3))
    inset.yaxis.set_major_locator(MaxNLocator(3))
    inset.grid(color="#ededed", lw=0.35)
    for spine in inset.spines.values():
        spine.set_color("#aaaaaa")
        spine.set_linewidth(0.4)
    if purple:
        inset.set_title(r"Near-hub ($X$–$Y$)" if refined else "Near-hub detail", fontsize=5, pad=3)
        if inset_variant == "ticks":
            inset.text(
                0.97,
                0.02,
                "X–Y (km)",
                transform=inset.transAxes,
                ha="right",
                va="bottom",
                fontsize=4.5,
            )
        elif inset_variant == "scalebar":
            inset.set_xticks([])
            inset.set_yticks([])
            inset.grid(False)
            span = bounds[0, 1] - bounds[0, 0]
            if span < 50:
                raise ValueError("Inset window is too narrow for a physical 50 km scale bar")
            x0 = bounds[0, 0] + 0.12 * span
            x1 = x0 + 50.0  # actual km in hub-relative X, not a display approximation
            if x1 > bounds[0, 1]:
                x0, x1 = bounds[0, 0], bounds[0, 0] + 50.0
            y = bounds[1, 0] + (0.17 if refined else 0.22) * span
            inset.plot(
                [x0, x1], [y, y], color="#444444", lw=0.8 if refined else 1, gid="scale-bar-50-km"
            )
            for x in (x0, x1):
                inset.plot(
                    [x, x],
                    [y - 0.015 * span, y + 0.015 * span],
                    color="#444444",
                    lw=0.45 if refined else 0.6,
                )
            inset.annotate(
                "50 km",
                ((x0 + x1) / 2, y),
                xytext=(0, -1.2 if refined else -2),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=4.5,
            )
            if not refined:
                inset.text(0.03, 0.96, "X–Y", transform=inset.transAxes, va="top", fontsize=4.5)
        else:
            raise ValueError("inset_variant must be ticks or scalebar")
    return inset


def _draw_baseline_configuration(
    axis,
    record: ScienceHourGeometry,
    *,
    view,
    zoom_inset,
    baseline_mode="witness",
    scale_pairs=HYBRID_SCALE_PAIRS,
    baseline_style="classic",
    inset_variant="ticks",
    refined=False,
):
    if baseline_style not in {"classic", "purple"}:
        raise ValueError("baseline_style must be classic or purple")
    purple = baseline_style == "purple"
    colors = PURPLE_SCALE_COLORS if purple else HYBRID_SCALE_COLORS
    highlights = _baseline_highlights(record, baseline_mode, scale_pairs, colors)
    relative = record.positions - record.positions[0]
    for i, j in record.baseline_indices:
        axis.plot(*relative[[i, j]].T, color="#B8B8B8", lw=0.6, alpha=0.25)
    for (first, second), color, width in highlights:
        i, j = record.spacecraft_ids.index(first), record.spacecraft_ids.index(second)
        axis.plot(*relative[[i, j]].T, color=color, lw=width)
    for i, name in enumerate(record.spacecraft_ids):
        axis.scatter(
            *relative[i],
            s=18 if name == "H" else 10,
            color=spacecraft_color(name),
            edgecolors="black",
            linewidths=0.3,
            depthshade=False,
        )
    span = np.ptp(relative, axis=0)
    margin = np.maximum(0.04 * span, 10)
    low, high = relative.min(axis=0) - margin, relative.max(axis=0) + margin
    axis.set(
        xlim=(low[0], high[0]),
        ylim=(low[1], high[1]),
        zlim=(low[2], high[2]),
        xlabel=r"$\Delta X_{GSE}$ (km)",
        ylabel=r"$\Delta Y_{GSE}$ (km)",
        zlabel=r"$\Delta Z_{GSE}$ (km)",
    )
    axis.set_box_aspect(high - low, zoom=0.98)
    axis.set_proj_type("ortho")
    axis.view_init(elev=view[0], azim=view[1])
    axis.tick_params(labelsize=6, pad=-1, length=2)
    for direction in (axis.xaxis, axis.yaxis, axis.zaxis):
        lower, upper = direction.get_view_interval()
        ticks = MaxNLocator(2).tick_values(lower, upper)
        inside = [float(tick) for tick in ticks if lower <= tick <= upper]
        # Keep the origin and one scale tick; avoid colliding corner labels in
        # a journal-sized 3-D panel. This changes labels, never axis limits.
        nonzero = [tick for tick in inside if tick != 0]
        if 0 in inside and nonzero:
            inside = sorted([0, max(nonzero, key=lambda tick: (abs(tick), tick))])
        direction.set_ticks(inside)
        direction.labelpad = -1
        direction.pane.set_facecolor((1, 1, 1, 0))
        direction.pane.set_edgecolor((1, 1, 1, 0))
        direction.pane.set_alpha(0)
        direction.line.set_color("#aaaaaa")
        direction.line.set_linewidth(0.45)
        direction._axinfo["grid"].update(color="#e8e8e8", linewidth=0.4)
        direction._axinfo["tick"].update(linewidth={True: 0.4, False: 0.3})
        if purple:
            direction.set_ticks([tick for tick in inside if tick != 0])
            direction.label.set_fontsize(7)  # 12.5% smaller than the existing 8 pt
            direction.line.set_color("#c8c8c8")
            direction.line.set_linewidth(0.35)
            direction._axinfo["grid"].update(color="#f1f1f1", linewidth=0.3)
    if zoom_inset:
        _draw_hub_inset(
            axis,
            record,
            highlights=highlights,
            baseline_style=baseline_style,
            inset_variant=inset_variant,
            refined=refined,
        )
    if baseline_mode == "hybrid":
        examples = total_scale_examples(record, scale_pairs, colors=colors)
        axis.legend(
            handles=[
                Line2D([], [], color=item["color"], lw=2.2, label=item["legend"])
                for item in examples
            ],
            title=None if purple else "Total separation D (examples)",
            title_fontsize=6,
            fontsize=6,
            loc="upper center" if purple else "lower center",
            bbox_to_anchor=(
                axis.get_position().x0 + axis.get_position().width / 2,
                0.13 if purple else 0.04,
            ),
            bbox_transform=axis.figure.transFigure,
            frameon=False,
            handlelength=1.8,
            labelspacing=0.2,
            borderaxespad=0,
        )


def plot_baseline_configuration(
    record: ScienceHourGeometry,
    *,
    view=DEFAULT_BASELINE_VIEW,
    zoom_inset: bool = False,
    baseline_mode="witness",
    scale_pairs=HYBRID_SCALE_PAIRS,
    baseline_style="classic",
    inset_variant="ticks",
):
    """Return a standalone camera/inset preview of the same baseline geometry.

    baseline_mode='hybrid' colors one total-length example per scale; it does
    not select or present those edges as a twelve-category witness.
    """
    with plt.rc_context(FIGURE_STYLE):
        hybrid = baseline_mode == "hybrid"
        purple = baseline_style == "purple"
        figure = plt.figure(figsize=(3.3, 3.2 if purple else 3.4 if hybrid else 3.0))
        axis = figure.add_axes(
            [0.10, 0.21, 0.79, 0.70]
            if purple
            else [0.10, 0.25, 0.79, 0.66]
            if hybrid
            else [0.10, 0.10, 0.79, 0.79],
            projection="3d",
        )
        _draw_baseline_configuration(
            axis,
            record,
            view=view,
            zoom_inset=zoom_inset,
            baseline_mode=baseline_mode,
            scale_pairs=scale_pairs,
            baseline_style=baseline_style,
            inset_variant=inset_variant,
        )
    return figure, axis


def plot_science_hour_geometry(
    record: ScienceHourGeometry,
    *,
    view=DEFAULT_BASELINE_VIEW,
    zoom_inset: bool = False,
    baseline_mode="witness",
    scale_pairs=HYBRID_SCALE_PAIRS,
    baseline_style="classic",
    inset_variant="ticks",
    layout_style="original",
    polyhedron_coloring="teal",
):
    """Return three panels from one record, optionally with a near-hub inset.

    The default witness mode preserves v02. Hybrid mode illustrates total-length
    examples separately from the unchanged ensemble eligibility calculations.
    """
    if layout_style not in {"original", "v04"}:
        raise ValueError("layout_style must be original or v04")
    if polyhedron_coloring not in {"teal", "nsc"}:
        raise ValueError("polyhedron_coloring must be teal or nsc")
    with plt.rc_context(FIGURE_STYLE):
        hybrid = baseline_mode == "hybrid"
        figure = plt.figure(figsize=(7.2, 3.0 if hybrid else 2.7))
        grid = figure.add_gridspec(
            1,
            3,
            width_ratios=(1, 0.9, 1.2),
            left=0.08,
            right=0.94,
            top=0.91,
            bottom=0.35 if hybrid else 0.29,
            wspace=0.38,
        )
        orbit = figure.add_subplot(grid[0, 0])
        poly = figure.add_subplot(grid[0, 1])
        formation = figure.add_subplot(grid[0, 2], projection="3d")
        if layout_style == "v04":
            box = formation.get_position()
            formation.set_position([box.x0 - 0.045, box.y0, box.width, box.height])
        axes = (orbit, poly, formation)
        _plot_context_projection(
            orbit,
            record.context_positions,
            record.positions.mean(axis=0),
            0,
            1,
            "",
            segment_validity=record.segment_validity,
        )
        orbit.collections[0].set_linewidth(0.45)  # 18% thinner; same classified segments
        marker = orbit.collections[-1]
        marker.set_sizes([20])
        marker.set_facecolors("white")
        marker.set_edgecolors("black")
        marker.set_linewidths(0.8)
        orbit.set_aspect("equal", adjustable="box")
        orbit.set_anchor("N")
        region_labels = {
            "Solar Wind": "Solar wind",
            "Connected Foreshock": "Connected foreshock",
            "Magnetosphere / Magnetosheath": "Magnetosheath / magnetosphere",
        }
        orbit.legend(
            handles=[
                Line2D([], [], color=color, lw=0.8, label=region_labels[name])
                for name, color in REGION_COLORS.items()
            ],
            loc="upper left",
            bbox_to_anchor=(0, -0.27),
            frameon=False,
            borderaxespad=0,
            handlelength=1.3,
            labelspacing=0.2,
            fontsize=6,
        )
        orbit.xaxis.set_major_locator(MaxNLocator(4))
        orbit.yaxis.set_major_locator(MaxNLocator(4))
        orbit.xaxis.labelpad = 2
        orbit.yaxis.labelpad = 2
        orbit.grid(alpha=0.15, linewidth=0.35)

        q = np.asarray([m["Q"] for m in record.polyhedra])
        lengths = np.asarray([m["L"] for m in record.polyhedra])
        positive = lengths > 0
        good = positive & (q <= REGULAR_POLYHEDRON_Q_LIMIT)
        teal = "#56877b"
        poly.scatter(
            q[positive], lengths[positive], s=5, color="#b8b8b8", alpha=0.5, edgecolors="none"
        )
        counts = np.asarray([m["n_spacecraft"] for m in record.polyhedra])
        if polyhedron_coloring == "nsc":
            cmap = plt.get_cmap("cividis", 6)
            boundaries = np.arange(3.5, 10, 1)
            norm = BoundaryNorm(boundaries, cmap.N)
            acceptable = poly.scatter(
                q[good],
                lengths[good],
                s=8,
                c=counts[good],
                cmap=cmap,
                norm=norm,
                alpha=0.8,
                edgecolors="none",
            )
            # An inset color bar uses otherwise empty data space; it never
            # steals width from the parent axes or shifts neighboring panels.
            color_axis = poly.inset_axes([0.49, 0.11, 0.46, 0.028])
            color_axis.set_gid("polyhedron-membership-colorbar")
            colorbar = figure.colorbar(
                acceptable,
                cax=color_axis,
                orientation="horizontal",
                boundaries=boundaries,
                ticks=np.arange(4, 10),
                spacing="uniform",
            )
            colorbar.set_label(r"$N_{\rm sc}$", fontsize=6, labelpad=1)
            colorbar.ax.xaxis.set_label_position("top")
            colorbar.ax.tick_params(labelsize=5, length=1.5, pad=1)
            colorbar.outline.set_linewidth(0.3)
        else:
            poly.scatter(q[good], lengths[good], s=8, color=teal, alpha=0.8, edgecolors="none")
        poly.axvline(REGULAR_POLYHEDRON_Q_LIMIT, ls="--", color="#555555", lw=0.7)
        for item in (record.small, record.large):
            poly.scatter(
                item["Q"],
                item["L"],
                s=34,
                marker="o",
                facecolors=cmap(norm(item["n_spacecraft"]))
                if polyhedron_coloring == "nsc"
                else teal,
                edgecolors="black",
                linewidths=0.9,
                zorder=5,
            )
        poly.set(xlabel=r"$\chi=\sqrt{E^2+P^2}$", ylabel=r"$L$ (km)", yscale="log", xlim=(0, 1.45))
        poly.text(
            0.95,
            0.27,
            rf"$L_{{\rm large}}/L_{{\rm small}}={record.scale_ratio:.1f}$",
            transform=poly.transAxes,
            ha="right",
            fontsize=6,
        )
        poly.text(
            REGULAR_POLYHEDRON_Q_LIMIT + 0.04,
            0.58,
            r"$\chi_{\max}=0.6$",
            transform=poly.get_xaxis_transform(),
            fontsize=6,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1, "alpha": 0.85},
        )
        if np.any(~positive):
            poly.text(
                0.03,
                0.97,
                f"{np.count_nonzero(~positive)} zero-L candidates omitted",
                transform=poly.transAxes,
                va="top",
                fontsize=6,
            )
        poly.grid(alpha=0.15, lw=0.35, which="both")
        poly.set_axisbelow(True)
        _draw_baseline_configuration(
            formation,
            record,
            view=view,
            zoom_inset=zoom_inset,
            baseline_mode=baseline_mode,
            scale_pairs=scale_pairs,
            baseline_style=baseline_style,
            inset_variant=inset_variant,
            refined=layout_style == "v04",
        )
        # Equal-aspect orbit may shrink within its slot; align the other 2-D
        # plotting region and all panel letters to its actual top and bottom.
        figure.canvas.draw()
        orbit_box, poly_box = orbit.get_position(), poly.get_position()
        poly.set_position([poly_box.x0, orbit_box.y0, poly_box.width, orbit_box.height])
        if baseline_style == "purple" and formation.get_legend() is not None:
            # Align the two external legends without changing panels (a)/(b).
            environment_box = (
                orbit.get_legend()
                .get_window_extent(figure.canvas.get_renderer())
                .transformed(figure.transFigure.inverted())
            )
            formation_box = formation.get_position()
            formation.get_legend().set_bbox_to_anchor(
                (formation_box.x0 + formation_box.width / 2, environment_box.y1),
                transform=figure.transFigure,
            )
        label_y = 0.94
        for label, axis in zip("abc", axes):
            figure.text(
                axis.get_position().x0, label_y, f"({label})", fontweight="bold", fontsize=8
            )
    return figure, axes


def save_baseline_previews(record: ScienceHourGeometry, output: str | Path) -> list[Path]:
    """Save three camera trials, the recommended view, and both inset variants."""
    base = Path(output)
    if base.suffix.lower() in {".pdf", ".png"}:
        base = base.with_suffix("")
    variants = [(f"baseline_view_{name}", view, False) for name, view in BASELINE_VIEWS.items()]
    variants += [
        ("baseline_no_inset", DEFAULT_BASELINE_VIEW, False),
        ("baseline_with_inset", DEFAULT_BASELINE_VIEW, True),
    ]
    paths = []
    for name, view, inset in variants:
        figure, _ = plot_baseline_configuration(record, view=view, zoom_inset=inset)
        try:
            path = base.parent / f"{base.name}_{name}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=300, facecolor="white")
            paths.append(path)
        finally:
            plt.close(figure)
    return paths


def save_science_hour_geometry(
    record: ScienceHourGeometry,
    output: str | Path,
    *,
    output_format: str = "both",
    view=DEFAULT_BASELINE_VIEW,
    zoom_inset: bool = False,
    baseline_mode="witness",
    scale_pairs=HYBRID_SCALE_PAIRS,
    baseline_style="classic",
    inset_variant="ticks",
) -> list[Path]:
    """Save vector PDF and/or a 300 dpi PNG preview; always close the figure."""
    if output_format not in {"pdf", "png", "both"}:
        raise ValueError("output_format must be pdf, png, or both")
    base = Path(output)
    if base.suffix.lower() in {".pdf", ".png"}:
        base = base.with_suffix("")
    formats = ("pdf", "png") if output_format == "both" else (output_format,)
    paths = [base.parent / f"{base.name}.{extension}" for extension in formats]
    figure, _ = plot_science_hour_geometry(
        record,
        view=view,
        zoom_inset=zoom_inset,
        baseline_mode=baseline_mode,
        scale_pairs=scale_pairs,
        baseline_style=baseline_style,
        inset_variant=inset_variant,
    )
    try:
        with plt.rc_context({"pdf.fonttype": 42}):
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                figure.savefig(path, dpi=300, facecolor="white")
    finally:
        plt.close(figure)
    return paths
