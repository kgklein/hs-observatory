"""Explicit-only OU summary tests with small synthetic saved paths."""

# ruff: noqa: E402 -- validator scripts are not an installed package.
from pathlib import Path
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from measurement_error_validation.model import angular_sigma, normalize, split_sigma
from measurement_error_validation.ou_summary import build_ou_summary, plot_ou_summary
from measurement_error_validation.runner import OU_SUMMARY_LAGS, Validator
from test_measurement_error_validation import args, fixture


def synthetic_paths(tmp_path, *, m=6, n=1001, dt=1, zero=False):
    cfg = normalize({})
    for section in cfg.values():
        section["enabled"] = False
    for component in ("density", "velocity_speed"):
        cfg[component].update(
            enabled=True,
            model="gaussian_fractional_bias_plus_ou",
            ou_correlation_time_s=300,
            error_interpretation="fractional_error",
        )
        if zero:
            cfg[component]["requirement_fraction"] = 0
    cfg["velocity_angle"].update(
        enabled=True,
        model="transverse_ou_rotation_total_angle",
        ou_correlation_time_s=300,
        requirement_interpretation="total_angle_confidence_radius",
    )
    if zero:
        cfg["velocity_angle"]["requirement_deg"] = 0
    path = tmp_path / "tiny-ou.h5"
    supplied = fixture(path, cfg, m=m, n=n, dt=dt)
    rng = np.random.default_rng(9)
    expected = {}
    with h5py.File(path, "a") as handle:
        for component in ("density", "velocity_speed", "velocity_angle"):
            sigma = (
                angular_sigma(cfg[component])
                if component == "velocity_angle"
                else split_sigma(cfg[component])[2]
            )
            names = (
                ["velocity_angle_alpha", "velocity_angle_beta"]
                if component == "velocity_angle"
                else [component + "_fractional_error"]
            )
            for i, name in enumerate(names):
                # A nonzero path mean checks that empirical demeaning is never used.
                u = sigma * (rng.normal(size=(m, n, 5)) + 0.5) * (i + 1)
                bias = rng.normal(scale=0.01, size=(m, 5)) if not zero else np.zeros((m, 5))
                saved = u if component == "velocity_angle" else u + bias[:, None, :]
                handle.create_dataset("errors/" + name, data=saved).attrs["units"] = (
                    "rad" if component == "velocity_angle" else "1"
                )
                if component != "velocity_angle":
                    handle.create_dataset("errors/" + component + "_bias", data=bias)
                expected[name] = (u, sigma)
    with h5py.File(path, "r") as handle:
        validator = Validator(handle, supplied, args(confidence=0.99))
        validator.run()
    return cfg, validator, expected


def test_path_covariance_bias_removal_and_fixed_95_percent_intervals(tmp_path):
    cfg, validator, expected = synthetic_paths(tmp_path)
    records = [r for r in validator.plots if r["kind"] == "ou_summary"]
    assert len(records) == 4
    for record in records:
        u, sigma = expected[record["record_id"]]
        assert record["lag"].tolist() == list(OU_SUMMARY_LAGS)
        for j, lag in enumerate(OU_SUMMARY_LAGS):
            products = u * u if lag == 0 else u[:, :-lag] * u[:, lag:]
            per_path = products.mean(axis=1).ravel() / sigma**2
            estimate = per_path.mean()
            margin = stats.t.ppf(0.975, 29) * per_path.std(ddof=1) / np.sqrt(30)
            assert record["measured"][j] == pytest.approx(estimate)
            assert record["lower"][j] == pytest.approx(estimate - margin)
            assert record["upper"][j] == pytest.approx(estimate + margin)
    # Extra plotted lags must not change the original statistical test family.
    acf_names = [r["statistic"] for r in validator.results.rows if ".acf_" in r["statistic"]]
    assert len(acf_names) == 12
    assert all(name.rsplit(".acf_", 1)[1] in ("30s", "300s", "600s") for name in acf_names)
    fig = build_ou_summary(records, cfg)
    try:
        assert len(fig.axes) == 3
        assert not fig.legends
        assert [text.get_text() for text in fig.axes[0].get_legend().get_texts()] == [
            "Empirical (95% CI)",
            "Configured OU model",
        ]
        assert fig.axes[1].get_legend() is None
        assert [text.get_text() for text in fig.axes[2].get_legend().get_texts()] == [
            "α (95% CI)",
            "β (95% CI)",
        ]
        assert fig.axes[0].get_subplotspec().get_gridspec().get_geometry() == (3, 1)
        for label, ax in zip("abc", fig.axes):
            assert any(t.get_text() == f"({label})" for t in ax.texts)
            assert chr(9) not in ax.get_ylabel()
            assert chr(92) + "tau" in ax.get_ylabel()
            curve = ax.lines[0]
            assert curve.get_ydata() == pytest.approx(np.exp(-curve.get_xdata() / 300))
        assert len(fig.axes[2].containers) == 2
        assert fig.axes[2].containers[0].lines[0].get_marker() == "o"
        assert fig.axes[2].containers[1].lines[0].get_marker() == "s"
    finally:
        plt.close(fig)
    names = plot_ou_summary(records, cfg, tmp_path)
    assert names == [
        "measurement_error_validation_ou_summary.png",
        "measurement_error_validation_ou_summary.pdf",
    ]
    assert all((tmp_path / name).stat().st_size > 1000 for name in names)


def test_unresolved_lags_small_ensemble_and_zero_variance(tmp_path):
    cfg, validator, _ = synthetic_paths(tmp_path, m=1, n=20, dt=3)
    records = [r for r in validator.plots if r["kind"] == "ou_summary"]
    for record in records:
        assert record["lag"].tolist() == [0]
        assert np.isnan(record["lower"]).all()
    fig = build_ou_summary(records, cfg)
    try:
        assert any("fewer than 30" in t.get_text() for t in fig.axes[0].texts)
    finally:
        plt.close(fig)
    cfg, validator, _ = synthetic_paths(tmp_path, zero=True)
    records = [r for r in validator.plots if r["kind"] == "ou_summary"]
    fig = build_ou_summary(records, cfg)
    try:
        assert all(
            any("normalization undefined" in t.get_text() for t in ax.texts) for ax in fig.axes
        )
    finally:
        plt.close(fig)
    cfg["density"]["enabled"] = False
    fig = build_ou_summary([], cfg)
    try:
        assert any("Disabled" in t.get_text() for t in fig.axes[0].texts)
        assert any("unavailable" in t.get_text() for t in fig.axes[1].texts)
    finally:
        plt.close(fig)
