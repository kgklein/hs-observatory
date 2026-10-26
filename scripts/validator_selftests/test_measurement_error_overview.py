"""Explicit-only overview smoke tests; no external validation products required."""

# ruff: noqa: E402 -- validator scripts are not an installed package.
from pathlib import Path
import sys
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy import special, stats
from matplotlib.colors import to_rgba

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from measurement_error_validation.model import angular_sigma, normalize, position_sigma
from measurement_error_validation.overview import build_overview, plot_overview
from measurement_error_validation.reporting import write_reports
from measurement_error_validation.runner import Validator
from test_measurement_error_validation import args, fixture


def example():
    cfg = normalize(
        {
            "timing": {"requirement_ms": 15.625},
            "density": {"model": "gaussian_fractional_bias_plus_ou"},
            "velocity_angle": {"model": "transverse_ou_rotation_total_angle"},
            "alignment": {"model": "relative_constant_rotation_B_to_V"},
        }
    )
    records = []

    def add(component, key, values):
        records.append(
            dict(component=component, record_id=key, kind="distribution", values=np.array(values))
        )

    add("timing", "offset.0", [-0.001, 0, 0.001])
    add("position", "radius", [0.1, 0.2, 0.3])
    add("density", "initial_full", [-0.02, 0, 0.02])
    add("velocity_speed", "initial_full", [-0.01, 0, 0.01])
    add("velocity_angle", "total_angle", [0.001, 0.002, 0.003])
    for component, stem in (("magnetic_dc", "offset"), ("alignment", "relative")):
        for index in range(3):
            add(component, f"{stem}.{index}", [-0.001, 0, 0.001])
    for n in (3, 5):
        records.append(
            dict(
                component="magnetic_ac",
                kind="spectrum",
                frequency=np.arange(n),
                psd=np.ones((3, n)) * 4,
                target=np.ones(n) * 9,
            )
        )
    return cfg, records


def test_overview_units_targets_and_layout():
    cfg, records = example()
    # Capture histogram inputs to check transformations independently of bin placement.
    from matplotlib.axes import Axes

    original = Axes.hist
    samples = []

    def capture(ax, values, *positional, **kwargs):
        samples.append(np.asarray(values))
        return original(ax, values, *positional, **kwargs)

    with patch.object(Axes, "hist", capture):
        fig = build_overview(records, cfg)
    try:
        assert len(fig.axes) == 8
        assert fig.axes[0].get_subplotspec().get_gridspec().get_geometry() == (4, 2)
        assert samples[0] == pytest.approx([-1, 0, 1])
        assert samples[2] == pytest.approx([-0.02, 0, 0.02])
        assert samples[3] == pytest.approx(np.expm1([-0.01, 0, 0.01]))
        assert samples[4] == pytest.approx(np.rad2deg([0.001, 0.002, 0.003]))
        assert samples[-1] == pytest.approx(np.rad2deg([-0.001, 0, 0.001]))
        for level, style in ((2, "--"), (3, "-.")):
            probability = special.erf(level / np.sqrt(2))
            expected_bounds = {
                0: [-level * 15.625 / 3, level * 15.625 / 3],
                1: [stats.maxwell.ppf(probability, scale=position_sigma(cfg["position"])[0])],
                2: [-level * 0.06 / 3, level * 0.06 / 3],
                3: np.expm1([-level * 0.03 / 3, level * 0.03 / 3]),
                4: [
                    stats.rayleigh.ppf(
                        probability, scale=np.rad2deg(angular_sigma(cfg["velocity_angle"]))
                    )
                ],
                5: [-level * 0.15 / 3, level * 0.15 / 3],
                7: [-level, level],
            }
            for index, bounds in expected_bounds.items():
                ax = fig.axes[index]
                lines = [line for line in ax.lines if line.get_label() == f"{level}σ"]
                assert len(lines) == len(bounds)
                for line, expected in zip(lines, bounds):
                    assert line.get_xdata() == pytest.approx([expected, expected])
                    assert line.get_linestyle() == style
                    assert ax.get_xlim()[0] < expected < ax.get_xlim()[1]
        for label, ax in zip("abcdefgh", fig.axes):
            text = next(text for text in ax.texts if text.get_text() == f"({label})")
            assert text.get_position() == (0.97, 0.96)
            assert text.get_ha() == "right"
            assert text.get_va() == "top"
        assert all(ax.get_legend() is None for i, ax in enumerate(fig.axes) if i not in (5, 6))
        assert [text.get_text() for text in fig.axes[5].get_legend().get_texts()] == ["x", "y", "z"]
        assert [text.get_text() for text in fig.axes[6].get_legend().get_texts()] == [
            "Saved total ASD"
        ]
        assert len(fig.legends) == 1
        assert [text.get_text() for text in fig.legends[0].get_texts()] == [
            "Saved realizations",
            "Configured model",
            "2σ",
            "3σ",
        ]
        assert len(fig.axes[6].lines) == 2
        assert to_rgba(fig.axes[6].lines[0].get_color()) == to_rgba("0.15")
        for index in (0, 1, 2, 3, 4):
            assert fig.axes[index].patches[0].get_facecolor() == to_rgba("tab:blue", 0.4)
        for ax in fig.axes:
            for line in ax.lines:
                if line.get_label() in ("Configured PDF", "Configured total ASD"):
                    assert to_rgba(line.get_color()) == to_rgba("tab:orange")
        for index, color in enumerate(("tab:red", "tab:green", "tab:purple")):
            for panel in (5, 7):
                assert fig.axes[panel].containers[index].patches[0].get_facecolor() == to_rgba(
                    color, 0.4
                )
        assert fig.axes[6].lines[0].get_xdata() == pytest.approx([1, 2, 3, 4])
        assert fig.axes[6].lines[0].get_ydata() == pytest.approx(np.full(4, np.sqrt(12)))
        assert fig.axes[6].lines[1].get_ydata() == pytest.approx(np.full(4, np.sqrt(27)))
        for panel, distribution in (
            (0, stats.norm(scale=15.625 / 3)),
            (1, stats.maxwell(scale=position_sigma(cfg["position"])[0])),
            (3, stats.lognorm(s=0.03 / 3, loc=-1)),
            (4, stats.rayleigh(scale=np.rad2deg(angular_sigma(cfg["velocity_angle"])))),
        ):
            curve = fig.axes[panel].lines[0]
            assert curve.get_ydata() == pytest.approx(distribution.pdf(curve.get_xdata()))
    finally:
        plt.close(fig)


def test_missing_disabled_legacy_and_zero():
    cfg, records = example()
    cfg["timing"]["enabled"] = False
    cfg["alignment"]["model"] = "independent_constant_rotation_B_and_V"
    cfg["density"]["requirement_fraction"] = 0
    for record in records:
        if record["component"] == "density":
            record["values"][:] = 0
    records = [r for r in records if r["component"] != "position"]
    fig = build_overview(records, cfg)
    try:
        assert any("Disabled" in text.get_text() for text in fig.axes[0].texts)
        assert any("unavailable" in text.get_text() for text in fig.axes[1].texts)
        assert any("legacy" in text.get_text() for text in fig.axes[7].texts)
        assert any(line.get_label() == "Configured point mass" for line in fig.axes[2].lines)
    finally:
        plt.close(fig)


def test_files_and_report_registration(tmp_path):
    cfg, records = example()
    names = plot_overview(records, cfg, tmp_path)
    for name in names:
        assert (tmp_path / name).stat().st_size > 1000
    payload = dict(
        status="PASS",
        exit_code=0,
        results=[],
        provenance={
            "input": "synthetic.h5",
            "embedded_configuration": cfg,
        },
    )
    with patch("measurement_error_validation.reporting.plot_figures", return_value=[]):
        write_reports(tmp_path, payload, records)
    assert payload["figures"] == names + [
        "measurement_error_validation_ou_summary.png",
        "measurement_error_validation_ou_summary.pdf",
    ]
    assert names[0] in (tmp_path / "validation_report.md").read_text()
    assert names[1] in (tmp_path / "validation_results.json").read_text()


def test_reference_clock_and_zero_records(tmp_path):
    import h5py

    cfg = normalize({})
    for section in cfg.values():
        section["enabled"] = False
    cfg["timing"].update(enabled=True, reference_spacecraft="S0")
    cfg["position"].update(enabled=True, requirement_magnitude_km=0)
    cfg["velocity_angle"].update(enabled=True, requirement_deg=0)
    path = tmp_path / "tiny.h5"
    supplied = fixture(path, cfg)
    with h5py.File(path, "a") as handle:
        handle["errors/time_offset"][:, 0] = 0
        for name, shape, units in (
            ("position_offset", (6, 5, 3), "km"),
            ("velocity_angle_alpha", (6, 33, 5), "rad"),
            ("velocity_angle_beta", (6, 33, 5), "rad"),
        ):
            handle.create_dataset("errors/" + name, data=np.zeros(shape)).attrs["units"] = units
    with h5py.File(path, "r") as handle:
        validator = Validator(handle, supplied, args())
        validator.run()
    for component, key in (
        ("timing", "offset.0"),
        ("position", "radius"),
        ("velocity_angle", "total_angle"),
    ):
        record = next(
            r for r in validator.plots if r["component"] == component and r.get("record_id") == key
        )
        assert len(record["values"]) == (24 if component == "timing" else 30)
    fig = build_overview(validator.plots, cfg)
    try:
        assert any("excluded" in text.get_text() for text in fig.axes[0].texts)
        for index in (1, 4):
            assert any(
                line.get_label() == "Configured point mass" for line in fig.axes[index].lines
            )
    finally:
        plt.close(fig)


@pytest.mark.parametrize("sigma_level", [2, 3])
def test_configured_requirement_matches_statistical_level(sigma_level):
    cfg, records = example()
    cfg["velocity_speed"]["model"] = "gaussian_fractional_bias_plus_ou"
    for section in cfg.values():
        section["sigma_level"] = sigma_level
    fig = build_overview(records, cfg)
    try:
        for panel, requirement in (
            (0, 15.625),
            (1, 2.5),
            (2, 0.06),
            (3, 0.03),
            (4, 0.25),
            (5, 0.15),
            (7, 3),
        ):
            lines = [
                line for line in fig.axes[panel].lines if line.get_label() == f"{sigma_level}σ"
            ]
            # Position's existing 3σ conversion uses rounded probability/factor.
            assert lines[-1].get_xdata()[0] == pytest.approx(requirement, rel=2e-4)
    finally:
        plt.close(fig)


def test_legacy_and_custom_radial_notes():
    cfg, records = example()
    cfg["position"]["radial_probability"] = 0.9
    cfg["velocity_angle"]["model"] = "transverse_ou_rotation"
    fig = build_overview(records, cfg)
    try:
        for panel, note in (
            (1, "radial coverage"),
            (3, "asymmetric"),
            (4, "legacy per-coordinate"),
        ):
            assert any(note in text.get_text() for text in fig.axes[panel].texts)
        # Custom coverage changes the Cartesian scale, not the 2σ coverage definition.
        target = stats.maxwell.ppf(
            special.erf(2 / np.sqrt(2)), scale=position_sigma(cfg["position"])[0]
        )
        line = next(line for line in fig.axes[1].lines if line.get_label() == "2σ")
        assert line.get_xdata()[0] == pytest.approx(target)
    finally:
        plt.close(fig)
