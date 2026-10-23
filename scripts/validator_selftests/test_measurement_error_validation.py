"""Explicit-only self-tests: pytest scripts/validator_selftests -q.

Fixtures exercise validator math and I/O, never production generators/fields.
This directory is outside pyproject.toml's default testpaths.
"""

# ruff: noqa: E402 -- explicit-only tests import the sibling scripts package.

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from scipy import signal, stats

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from measurement_error_validation.model import (
    component_asd,
    differences,
    normalize,
    position_sigma,
    rodrigues,
    spectral_target,
    split_sigma,
)
from measurement_error_validation.reporting import clean, write_reports
from measurement_error_validation.runner import InvalidInput, Validator, read_series, selection
from measurement_error_validation.statistics import Results

spec = importlib.util.spec_from_file_location(
    "validator_cli", SCRIPTS / "validate_measurement_errors.py"
)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def config(**overrides):
    c = normalize({})
    for value in c.values():
        value["enabled"] = False
    for name, value in overrides.items():
        c[name].update(value, enabled=True)
    return c


def fixture(path, cfg, m=6, s=5, n=33, dt=1.0):
    rng = np.random.default_rng(41)
    with h5py.File(path, "w") as f:
        a = f.create_group("metadata").attrs
        a["observatory_error_config"] = json.dumps(cfg)
        a["n_mc"] = m
        a["random_seed"] = 17
        f.create_dataset("spacecraft/id", data=np.array([f"S{i}" for i in range(s)], dtype="S"))
        for quantity in ("n", "v", "B"):
            f.create_dataset("time/" + quantity, data=np.arange(n) * dt)
        errors = f.create_group("errors")
        if cfg["timing"]["enabled"]:
            errors.create_dataset(
                "time_offset",
                data=rng.normal(
                    0, cfg["timing"]["requirement_ms"] / cfg["timing"]["sigma_level"] * 1e-3, (m, s)
                ),
            ).attrs["units"] = "s"
    supplied = dict(
        errors={**cfg, "n_mc": m, "random_seed": 17},
        time=dict(t_start_s=0, t_stop_s=(n - 1) * dt, cadence_s=dt),
    )
    return supplied


def args(**kwargs):
    defaults = dict(mc="all", spacecraft="all", confidence=0.99, chunk_times=7, seed=12)
    return SimpleNamespace(**(defaults | kwargs))


def test_requirement_conventions_and_aliases():
    cfg = normalize(
        {
            "position": {"sigma_level": 2},
            "density": {"sigma_level": 2},
            "velocity_speed": {"sigma_level": 2},
        }
    )
    sigma, p = position_sigma(cfg["position"])
    assert sigma == pytest.approx(0.8825114286769669)
    assert stats.maxwell.cdf(2.5, scale=sigma) == pytest.approx(p)
    assert position_sigma(normalize({})["position"])[0] == pytest.approx(2.5 / 3.762)
    assert split_sigma(cfg["density"]) == pytest.approx(
        [0.03, 0.03 / np.sqrt(2), 0.03 / np.sqrt(2)]
    )
    assert split_sigma(cfg["velocity_speed"])[0] == pytest.approx(0.015)
    assert (
        normalize({"density": {"bias_variance_fraction": 2, "ou_variance_fraction": 2}})["density"][
            "bias_variance_fraction"
        ]
        == 0.5
    )
    assert normalize({"time": {"requirement_3sigma_ms": 9}})["timing"]["requirement_ms"] == 9
    with pytest.raises(ValueError):
        normalize({"timing": {"requirement_ms": 9, "requirement_3sigma_ms": 9}})
    with pytest.raises(ValueError):
        normalize({"density": {"ou_correlation_time_s": 0}})
    with pytest.raises(ValueError):
        normalize({"magnetic_ac": {"component_distribution": "random"}})
    c = normalize({"position": {"radial_probability": 0.8}})
    assert stats.maxwell.cdf(2.5, scale=position_sigma(c["position"])[0]) == pytest.approx(0.8)


def test_asd_anchors_plateau_and_conversion():
    c = normalize({})["magnetic_ac"]
    f = np.array([0, 0.0001, 0.001, 1, 10, 100])
    a = component_asd(f, c)
    assert a == pytest.approx(np.array([1500, 1500, 1500, 15, 1.5, 0.15]) * 1e-3 / (3 * np.sqrt(3)))


@pytest.mark.parametrize("n,length", [(15, 7), (16, 8), (17, 17), (16, 16)])
def test_window_target_against_dense_covariance_oracle(n, length):
    """Direct E[|FFT(w*x)|²] matrix calculation independent of FFT-lag implementation."""
    dt = 0.125
    c = normalize({})["magnetic_ac"]
    f, expected, variance = spectral_target(n, dt, c, length)
    all_f = np.fft.fftfreq(n, dt)
    a = component_asd(np.abs(all_f), c)
    a[0] = 0
    j = np.arange(length)
    covariance = np.array(
        [
            [
                np.sum(a * a * np.cos(2 * np.pi * np.arange(n) * (i - k) / n)) / (2 * n * dt)
                for k in j
            ]
            for i in j
        ]
    )
    w = signal.windows.hann(length, sym=False)
    oracle = []
    for k in range(length // 2 + 1):
        transform = w * np.exp(-2j * np.pi * k * j / length)
        p = (transform @ covariance @ transform.conj()).real * dt / np.dot(w, w)
        if k and not (length % 2 == 0 and k == length // 2):
            p *= 2
        oracle.append(p)
    assert expected == pytest.approx(oracle, rel=1e-11, abs=1e-15)
    assert variance == pytest.approx(covariance[0, 0])
    assert f[-1] <= 1 / (2 * dt)


@pytest.mark.parametrize("n", [16, 17])
def test_finite_record_integrated_variance_and_nyquist(n):
    c = normalize({})["magnetic_ac"]
    dt = 0.125
    a = component_asd(np.fft.rfftfreq(n, dt), c)
    expected = np.sum(a[1:] ** 2) / (n * dt)
    if n % 2 == 0:
        expected -= a[-1] ** 2 / (2 * n * dt)
    assert spectral_target(n, dt, c, n)[2] == pytest.approx(expected)


def test_exact_gaussian_holm_and_deliberate_amplitude_error():
    # Quantile grid is deterministic; this tests formulas, not nominal random PASS guarantees.
    x = stats.norm.ppf((np.arange(500) + 0.5) / 500)
    result = Results()
    result.gaussian("unit", "nominal", x, 1)
    sigma_row = next(r for r in result.rows if r["statistic"] == "nominal.sigma")
    assert sigma_row["ci_low"] < 1 < sigma_row["ci_high"]
    result.gaussian("unit", "wrong", x * 4, 1)
    rows = result.finalize()
    assert next(r for r in rows if r["statistic"] == "wrong.sigma")["status"] == "FAIL"
    holm = Results(0.95)
    for p in [0.01, 0.04, 0.03]:
        holm.add("test", str(p), p=p)
    out = holm.finalize()
    assert [r["adjusted_p"] for r in out] == pytest.approx([0.03, 0.06, 0.06])
    assert [r["status"] for r in out] == [
        "PASS" if p >= 0.05 else "FAIL" for p in [0.03, 0.06, 0.06]
    ]


def test_maxwell_rayleigh_gaussian_expectations():
    r = Results()
    p = (np.arange(100) + 0.5) / 100
    for dist in [stats.maxwell(scale=0.8), stats.rayleigh(scale=0.1), stats.norm(scale=0.03)]:
        r.distribution("distribution", "grid", dist.ppf(p), dist)
    assert all(row["raw_p"] > 0.9 for row in r.rows)
    r.coverage("position", "radial", np.arange(100), 94.5, 0.9545)
    assert r.rows[-1]["measured"] == 0.95
    r.path_mean("ou", "short", np.ones(10), 1)
    assert r.rows[-1]["status"] == "INSUFFICIENT_DATA"


def test_selection_and_bounded_reads():
    assert selection("S2,S0", ["S0", "S1", "S2"]) == [2, 0]
    assert selection("2,0", list(range(3)), numeric=True) == [2, 0]
    for text in ["-1", "3", "0,0"]:
        with pytest.raises(InvalidInput):
            selection(text, list(range(3)), numeric=True)

    class Guarded:
        shape = (4, 23, 3)
        name = "guarded"

        def __getitem__(self, key):
            assert isinstance(key[0], int) and isinstance(key[2], int)
            assert key[1].stop - key[1].start <= 7
            return np.arange(key[1].start, key[1].stop, dtype=float)

    assert read_series(Guarded(), 1, 2, 7) == pytest.approx(np.arange(23))


def test_config_crosscheck_and_readonly_fixture(tmp_path):
    p = tmp_path / "small.h5"
    supplied = fixture(p, config(timing={}))
    original = p.read_bytes()
    with h5py.File(p, "r") as f:
        validator = Validator(f, supplied, args())
        assert not validator.discrepancies
        rows = validator.run()
        assert any(r["component"] == "timing" and r["raw_p"] is not None for r in rows)
        supplied["errors"]["timing"]["requirement_ms"] = 20
        assert Validator(f, supplied, args()).discrepancies
    assert p.read_bytes() == original
    assert differences({"sigma": 1}, {"sigma": 2})[0]["parameter"] == "sigma"


def test_ou_saved_bias_subtraction_and_innovations(tmp_path):
    p = tmp_path / "ou.h5"
    cfg = config(density={"sigma_level": 2, "ou_correlation_time_s": 10})
    supplied = fixture(p, cfg, n=41, dt=1)
    rng = np.random.default_rng(42)
    m, s, n = 6, 5, 41
    total, sb, su = split_sigma(cfg["density"])
    b = rng.normal(0, sb, (m, s))
    u = np.empty((m, n, s))
    u[:, 0] = rng.normal(0, su, (m, s))
    z = rng.normal(size=(m, n - 1, s))
    phi = np.exp(-0.1)
    for i in range(1, n):
        u[:, i] = phi * u[:, i - 1] + su * np.sqrt(1 - phi**2) * z[:, i - 1]
    with h5py.File(p, "a") as f:
        f["errors/density_bias"] = b
        f["errors/density_log_error"] = u + b[:, None, :]
    with h5py.File(p, "r") as f:
        v = Validator(f, supplied, args())
        rows = v.run()
    mean = next(r for r in rows if r["statistic"] == "density_log_error.innovations.mean")
    assert mean["measured"] == pytest.approx(z.mean(), abs=1e-14)
    assert mean["n_paths"] == 30 and mean["n_time"] == 40
    assert mean["n_independent_draws"] == 1200
    second = next(r for r in rows if r["statistic"] == "density_log_error.ou_variance")
    assert second["measured"] == pytest.approx(np.mean(u * u))
    # A wrong tau must be visible in the calculated innovation variance.
    with h5py.File(p, "r") as f:
        v = Validator(f, supplied, args())
        v.cfg["density"]["ou_correlation_time_s"] = 0.01
        wrong = v.run()
    wrong_var = next(r for r in wrong if r["statistic"] == "density_log_error.innovations.variance")
    assert wrong_var["status"] == "FAIL"


def test_alignment_invariants_without_application_data(tmp_path):
    p = tmp_path / "rotation.h5"
    cfg = config(alignment={})
    supplied = fixture(p, cfg, m=2, s=2)
    vectors = np.array([[[0.01, 0.02, 0.03], [0, 0, 0]], [[0.04, 0, 0.01], [-0.02, 0.01, 0.03]]])
    with h5py.File(p, "a") as f:
        for field in ("magnetic", "velocity"):
            d = f.create_dataset(f"errors/{field}_alignment_rotvec", data=vectors)
            d.attrs["units"] = "rad"
            f[f"errors/{field}_alignment_matrix"] = np.array(
                [[rodrigues(v) for v in row] for row in vectors]
            )
    with h5py.File(p, "r") as f:
        v = Validator(f, supplied, args())
        rows = v.run()
        assert all(r["status"] == "PASS" for r in rows if r["statistic"].endswith("rodrigues"))
    with h5py.File(p, "a") as f:
        f["errors/velocity_alignment_matrix"][0, 0, 0, 0] = 2
    with h5py.File(p, "r") as f:
        rows = Validator(f, supplied, args()).run()
    assert any(r["status"] == "FAIL" for r in rows if r["statistic"].endswith("orthogonality"))


def test_ac_diagnostics_tiny_fixture_and_bandwidth(tmp_path):
    p = tmp_path / "ac.h5"
    cfg = config(magnetic_ac={})
    supplied = fixture(p, cfg, m=2, s=2, n=33, dt=0.125)
    rng = np.random.default_rng(7)
    x = rng.normal(size=(2, 33, 2, 3))
    x -= x.mean(axis=1, keepdims=True)
    with h5py.File(p, "a") as f:
        f.create_dataset("errors/magnetic_ac_noise", data=x).attrs["units"] = "nT"
    with h5py.File(p, "r") as f:
        v = Validator(f, supplied, args())
        rows = v.run()
    assert next(r for r in rows if r["statistic"] == "anchor_10Hz")["status"] == "INSUFFICIENT_DATA"
    assert (
        next(r for r in rows if r["statistic"] == "zero_mean_relative_to_rms")["status"] == "PASS"
    )
    assert all(np.isfinite(plot["target"]).all() for plot in v.plots if plot["kind"] == "spectrum")


def test_serialization_and_incomplete_input_cli(tmp_path):
    supplied = tmp_path / "config.yaml"
    supplied.write_text("errors: {}\n")
    p = tmp_path / "partial.h5"
    p.write_bytes(b"partial transfer")
    out = tmp_path / "out"
    assert cli.main(["--input", str(p), "--config", str(supplied), "--output-dir", str(out)]) == 2
    report = json.loads((out / "validation_results.json").read_text())
    assert report["status"] == "INCOMPLETE_INPUT"
    assert (out / "validation_results.csv").exists()
    assert (out / "validation_report.md").exists()
    assert clean({"x": np.nan, "y": np.array([1.0])}) == {"x": None, "y": [1.0]}


def test_missing_components_and_config_mismatch_exit(tmp_path):
    import yaml

    p = tmp_path / "fixture.h5"
    cfg = config(timing={})
    supplied = fixture(p, cfg)
    supplied["errors"]["timing"]["requirement_ms"] = 99
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(yaml.safe_dump(supplied))
    out = tmp_path / "results"
    # No plots are needed to exercise orchestration/exit/report behavior.
    original = cli.write_reports

    def no_figures(output, payload, records):
        return write_reports(output, payload, [])

    cli.write_reports = no_figures
    try:
        assert (
            cli.main(["--input", str(p), "--config", str(yaml_file), "--output-dir", str(out)]) == 1
        )
    finally:
        cli.write_reports = original
    report = json.loads((out / "validation_results.json").read_text())
    assert report["configuration_checked"] and report["discrepancies"]
    assert report["provenance"]["embedded_configuration"]["timing"]["requirement_ms"] == 12.5
    with h5py.File(p, "a") as f:
        del f["errors/time_offset"]
    with h5py.File(p, "r") as f:
        rows = Validator(f, supplied, args()).run()
    assert next(r for r in rows if r["statistic"] == "time_offset")["status"] == "SKIPPED"


def test_nominal_timing_exit_zero_and_nonfinite_p_guard(tmp_path, monkeypatch):
    import yaml

    p = tmp_path / "nominal.h5"
    supplied = fixture(p, config(timing={}), m=450, s=1)
    with h5py.File(p, "a") as f:
        f["errors/time_offset"][:, 0] = stats.norm.ppf((np.arange(450) + 0.5) / 450) * (0.0125 / 3)
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(yaml.safe_dump(supplied))
    monkeypatch.setattr(
        cli, "write_reports", lambda out, payload, records: write_reports(out, payload, [])
    )
    out = tmp_path / "results"
    assert cli.main(["--input", str(p), "--config", str(yaml_file), "--output-dir", str(out)]) == 0
    payload = json.loads((out / "validation_results.json").read_text())
    assert payload["status"] == "PASS"
    assert payload["provenance"]["spectral_resolution"]["nyquist_Hz"] == 0.5
    r = Results()
    r.add("numeric", "invalid", p=float("nan"))
    assert r.finalize()[0]["status"] == "FAIL"


def test_single_innovation_is_insufficient_not_division_by_zero(tmp_path):
    p = tmp_path / "single.h5"
    cfg = config(velocity_angle={})
    supplied = fixture(p, cfg, m=1, s=1, n=2)
    with h5py.File(p, "a") as f:
        for name in ("alpha", "beta"):
            f.create_dataset("errors/velocity_angle_" + name, data=np.zeros((1, 2, 1))).attrs[
                "units"
            ] = "rad"
    with h5py.File(p, "r") as f:
        rows = Validator(f, supplied, args()).run()
    assert (
        next(r for r in rows if r["statistic"] == "velocity_angle_alpha.innovations")["status"]
        == "INSUFFICIENT_DATA"
    )


def test_figures_smoke_all_rendering_paths(tmp_path):
    from measurement_error_validation.reporting import plot_figures

    f = np.linspace(0, 4, 9)
    records = [
        dict(
            kind="distribution",
            component="density",
            title="Bias distribution",
            values=stats.norm.ppf((np.arange(20) + 0.5) / 20),
            distribution=stats.norm(),
            units="log fraction",
        ),
        dict(
            kind="acf",
            component="density",
            title="OU ACF",
            values=[(30.0, 0.9, 0.9, 0.8, 1.0), (300.0, 0.3, 0.367, 0.2, 0.4)],
        ),
        dict(
            kind="spectrum",
            component="magnetic_ac",
            title="Tiny smoke fixture",
            frequency=f,
            psd=np.ones((3, 9)),
            target=np.ones(9),
            raw_target=np.ones(9),
            lower=np.ones((3, 9)) * 0.8,
            upper=np.ones((3, 9)) * 1.2,
            n_paths=30,
        ),
    ]
    names = plot_figures(records, tmp_path)
    assert len(names) == 6
    assert all((tmp_path / name).stat().st_size > 0 for name in names)
