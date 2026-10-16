"""Campaign orchestration and reporting with fabricated moments; no field evaluations."""

import importlib.util
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
import pytest
import yaml

from hs_observatory.analysis.spatial_structure_function import (
    derive_spatial_case,
    write_spatial_result,
)

SPEC = importlib.util.spec_from_file_location(
    "campaign", Path(__file__).resolve().parents[1] / "scripts/malara_direct3d_convergence.py"
)
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


def product(generation_text, centers, budget=5000, seed=12345, options=None, p_values=None):
    generation = yaml.safe_load(generation_text)
    fields = campaign.field_configs(generation["field"])
    options = dict(
        options or {},
        n_pairs=budget,
        seed=seed,
        n_batches=8,
        scale_ranges_km=campaign.WINDOW,
        min_fit_bins=3,
        fit_range="candidate",
    )
    cases = []
    for p in campaign.P_VALUES if p_values is None else p_values:
        theory = campaign.p_model_exponents(campaign.ORDERS, p)
        # Explicitly fabricated power laws; changes across seed and budget are known.
        factor = (1 + (seed % 100) / 1000) * {5000: 1.0, 50000: 1.2, 500000: 1.32}[budget]
        values = factor * (centers[None] / 300) ** theory[:, None]
        batches = values[None] * np.linspace(0.8, 1.2, 8)[:, None, None]
        counts = np.full((1, 7), budget)
        batch_counts = np.broadcast_to(
            np.diff(np.linspace(0, budget, 9, dtype=int))[:, None], (8, 7)
        ).copy()
        cp_counts = np.broadcast_to(
            np.ceil(budget * np.array([0.25, 0.5, 0.75, 1])).astype(int)[:, None], (4, 7)
        ).copy()
        case = dict(
            p_value=p,
            field_config=fields[str(p)],
            orders=campaign.ORDERS,
            lambda_bin_centers_km=centers,
            counts=counts,
            q=dict(S_n=values[None]),
            moment_sums=values[None] * budget,
            batch_counts=batch_counts,
            batch_S_n=batches,
            batch_moment_sums=batches * batch_counts[:, None],
            checkpoint_counts=cp_counts,
            checkpoint_S_n=values[None] * np.array([1.25, 1.1, 0.95, 1])[:, None, None],
        )
        derive_spatial_case(case, campaign.WINDOW)
        cases.append(case)
    return dict(
        method="spatial_structure_function",
        config=options,
        comparisons={},
        metadata=dict(
            field_model="malara_turbulence",
            quantity="q",
            separation_mode="exact",
            sampling_seed=seed,
            sampling_rng="PCG64",
            sample_stream_version=1,
            source_field_yaml=generation_text,
            smoke=False,
        ),
        cases=cases,
    )


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "repo"
    output = root / "outputs/malara_direct3d_convergence"
    (root / "generation").mkdir(parents=True)
    (root / "outputs").mkdir()
    (root / "configs").mkdir()
    (root / ".venv/bin").mkdir(parents=True)
    executable = root / ".venv/bin/hsobs-analyze-spatial-structure-function"
    executable.write_text("#!/bin/sh\nexit 99\n")  # Never executed.
    executable.chmod(0o755)
    generation = dict(
        field=dict(
            model="malara_turbulence",
            domain=dict(L=[100000.0] * 3, units="km", origin=[0.0, 0.0, 0.0]),
            turbulence=dict(
                n_scales=2,
                injection_scale_index=1,
                intermittency_p=0.9,
                base_seed=12345,
                realization=1,
            ),
            output_fields=dict(
                n=dict(background=5.0, fluctuation="constant"),
                v=dict(background=[400.0, 0.0, 0.0], fluctuation="malara", amplitude=10.0),
                B=dict(background=[0.0, 0.0, 5.0], fluctuation="malara", amplitude=0.5),
            ),
        )
    )
    text = yaml.safe_dump(generation)
    (root / "generation/malara-ts-500a-t1500.yaml").write_text(text)
    centers = np.linspace(220.0, 1000.0, 7)
    centers[2] = np.nextafter(centers[2], np.inf)  # Ensure precision matters.
    pilot = product(text, centers)
    write_spatial_result(pilot, root / "outputs/direct-spatial-sf-5000.h5")
    (root / "configs/example_spatial_structure_function.yaml").write_text(
        yaml.safe_dump(
            dict(
                analysis=dict(method="spatial_structure_function"),
                input=dict(path="unused.yaml"),
                output=dict(path="outputs/direct-spatial-sf.h5"),
                spatial_structure_function=dict(
                    comparison_config="unrelated.yaml", plot=dict(enabled=True)
                ),
            )
        )
    )
    for name in (
        "analysis/spatial_structure_function.py",
        "analysis/structure_function.py",
        "analysis/structure_function_local.py",
        "cli_spatial_structure_function.py",
        "fields/malara.py",
    ):
        path = root / "src/hs_observatory" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic dependency fingerprint\n")
    return root, output, text, centers


def fake_cli(monkeypatch, fail_at=None):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if fail_at is not None and len(commands) == fail_at:
            raise subprocess.CalledProcessError(1, command)
        budget = int(command[command.index("--n-pairs") + 1])
        seed = int(command[command.index("--seed") + 1])
        source = Path(command[command.index("--field-config") + 1])
        first = command.index("--lambda-centers-km") + 1
        last = command.index("--n-pairs")
        centers = np.array([float(x) for x in command[first:last]])
        target = Path(command[command.index("--output-prefix") + 1] + ".h5")
        options = yaml.safe_load(Path(command[1]).read_text())["spatial_structure_function"]
        p_values = list(
            map(float, command[command.index("--p") + 1 : command.index("--lambda-centers-km")])
        )
        for flag in ("backend", "chunk-size"):
            if "--" + flag in command:
                value = command[command.index("--" + flag) + 1]
                options[flag.replace("-", "_")] = int(value) if flag == "chunk-size" else value
        write_spatial_result(
            product(source.read_text(), centers, budget, seed, options, p_values), target
        )

    monkeypatch.setattr(campaign.subprocess, "run", run)
    return commands


def test_discovery_precision_unique_names_and_dry_run(workspace, monkeypatch, capsys):
    root, output, _, centers = workspace
    plan = campaign.discover(root)
    assert len(plan["jobs"]) == len({j["prefix"] for j in plan["jobs"]}) == 6
    assert plan["inputs"]["config"]["spatial_structure_function"]["plot"] == {"enabled": False}
    assert "comparison_config" not in plan["inputs"]["config"]["spatial_structure_function"]
    job = plan["jobs"][0]
    command = campaign.command_for(root, root / "runtime.yaml", output / job["prefix"], plan, job)
    first, last = command.index("--lambda-centers-km") + 1, command.index("--n-pairs")
    np.testing.assert_array_equal(
        np.array(list(map(float, command[first:last]))).view("u8"), centers.view("u8")
    )
    commands = fake_cli(monkeypatch)
    campaign.run_campaign(root, output, dry_run=True)
    assert not commands and not output.exists()
    assert "4,620,000" in capsys.readouterr().out


def test_resume_requires_receipt_hash_and_complete_moments(workspace, monkeypatch):
    root, output, _, _ = workspace
    commands = fake_cli(monkeypatch)
    campaign.run_campaign(root, output, budgets=[5000], seeds=[12345])
    plan = campaign.discover(root)
    job = plan["jobs"][0]
    assert len(commands) == 1 and campaign.complete(output, plan, job)
    campaign.run_campaign(root, output, budgets=[5000], seeds=[12345], resume=True)
    assert len(commands) == 1
    with pytest.raises(ValueError, match="--resume"):
        campaign.run_campaign(root, output, budgets=[5000], seeds=[12345])
    receipt_path = output / (job["prefix"] + ".complete.json")
    receipt = json.loads(receipt_path.read_text())
    assert receipt["elapsed_s"] >= 0 and receipt["status"] == "complete"
    with h5py.File(output / (job["prefix"] + ".h5"), "a") as h:
        h["cases/0/counts"][0, 0] = 1
    assert not campaign.complete(output, plan, job)
    # Even updating the hash cannot make a numerically incomplete product complete.
    receipt["product_sha256"] = campaign.sha256(output / (job["prefix"] + ".h5"))
    campaign.atomic_json(receipt_path, receipt)
    assert not campaign.complete(output, plan, job)
    campaign.run_campaign(root, output, budgets=[5000], seeds=[12345], resume=True)
    assert len(commands) == 2 and campaign.complete(output, plan, job)
    assert list((output / ".incomplete").glob("*/previous.h5"))
    receipt_path.unlink()
    assert not campaign.complete(output, plan, job)


def test_failure_preserves_success_and_can_resume(workspace, monkeypatch):
    root, output, _, _ = workspace
    commands = fake_cli(monkeypatch, fail_at=2)
    with pytest.raises(RuntimeError, match="preserved diagnostics"):
        campaign.run_campaign(root, output, budgets=[5000], seeds=[12345, 23456])
    plan = campaign.discover(root)
    assert campaign.complete(output, plan, plan["jobs"][0])
    assert not campaign.complete(output, plan, plan["jobs"][1])
    success = output / (plan["jobs"][0]["prefix"] + ".h5")
    before = success.read_bytes()
    campaign.run_campaign(root, output, budgets=[5000], seeds=[12345, 23456], resume=True)
    assert len(commands) == 3 and success.read_bytes() == before
    assert campaign.complete(output, plan, plan["jobs"][1])


def test_changed_inputs_and_exact_centers_rejected(workspace, monkeypatch):
    root, output, _, _ = workspace
    fake_cli(monkeypatch)
    campaign.run_campaign(root, output, budgets=[5000], seeds=[12345])
    plan = campaign.discover(root)
    job = plan["jobs"][0]
    result, _ = campaign.validated_job(output, plan, job)
    result["cases"][0]["lambda_bin_centers_km"][0] = np.nextafter(
        result["cases"][0]["lambda_bin_centers_km"][0], np.inf
    )
    with pytest.raises(ValueError, match="exact separation bits"):
        campaign.validate_result(
            result,
            plan["inputs"]["fields"],
            centers=[float.fromhex(x) for x in plan["inputs"]["centers_hex"]],
        )
    field_path = root / "generation/malara-ts-500a-t1500.yaml"
    cfg = yaml.safe_load(field_path.read_text())
    cfg["field"]["turbulence"]["realization"] += 1
    field_path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="physical field configuration"):
        campaign.run_campaign(root, output, resume=True)


def test_report_aggregation_and_no_sampling(workspace, monkeypatch):
    root, output, _, _ = workspace
    fake_cli(monkeypatch)
    campaign.run_campaign(root, output)
    for source in (root / "generation").glob("*.yaml"):
        source.unlink()  # Saved-product reporting has no live field-config dependency.
    monkeypatch.setattr(campaign.subprocess, "run", lambda *a, **k: pytest.fail("sampling called"))
    records = campaign.report(output)
    assert len(records) == 12
    np.testing.assert_allclose(campaign.budget_changes(records, 0.9, "moments"), 0.2)
    np.testing.assert_allclose(campaign.budget_changes(records, 0.5, "xi"), 0, atol=1e-12)
    moments = campaign.aggregate(records, 0.9, 5000, "moments")
    assert moments.shape == (3, 6, 7)
    scale = (220 / 300) ** campaign.p_model_exponents([1], 0.9)[0]
    assert moments[:, 0, 0].mean() == pytest.approx(1.056 * scale)
    assert moments[:, 0, 0].std(ddof=1) == pytest.approx(0.011 * scale)
    assert campaign.stats(np.array([1.0, 2.0, 3.0])) == "2 ± 1 [1, 3]"
    report = (output / "convergence_report.md").read_text()
    assert "not variability between independent field realizations" in report
    assert "CP50000" in report and "budget" in report
    assert "20 ±" in report  # paired 20% moment change
    assert "Max CP4" in report and "ζ_n" in report
    for extension in ("png", "pdf"):
        assert (output / f"convergence_exponents.{extension}").stat().st_size > 0


def test_partial_report_never_claims_convergence(workspace, monkeypatch):
    root, output, _, _ = workspace
    fake_cli(monkeypatch)
    campaign.run_campaign(root, output, budgets=[5000], seeds=[12345])
    monkeypatch.setattr(campaign, "plot_report", lambda *a: None)
    campaign.report(output, threshold=1.0)
    text = (output / "convergence_report.md").read_text()
    assert "unavailable/incomplete" in text and "insufficient jobs" in text
    assert "single seed; SD unavailable" in text


def test_truncated_output_and_bad_pilot_metadata(workspace, monkeypatch):
    root, output, _, _ = workspace
    output.mkdir()
    path = output / (campaign.prefix(5000, 12345) + ".h5")
    path.write_bytes(b"incomplete HDF5")
    commands = fake_cli(monkeypatch)
    campaign.run_campaign(root, output, budgets=[5000], seeds=[12345], resume=True)
    assert len(commands) == 1
    plan = campaign.discover(root)
    result, _ = campaign.validated_job(output, plan, plan["jobs"][0])
    assert result["metadata"]["sampling_product"] == str(path)
    with h5py.File(root / "outputs/direct-spatial-sf-5000.h5", "a") as h:
        meta = json.loads(h.attrs["metadata_json"])
        meta["quantity"] = "z_plus"
        h.attrs["metadata_json"] = json.dumps(meta)
    with pytest.raises(ValueError, match="raw Malara"):
        campaign.discover(root)


def test_500k_extension_preserves_legacy_and_single_seed_report(workspace, monkeypatch, capsys):
    root, output, _, centers = workspace
    commands = fake_cli(monkeypatch)
    campaign.run_campaign(root, output)
    saved = json.loads((output / "campaign.json").read_text())
    campaign.run_campaign(
        root, output, budgets=[5000, 50000], seeds=[12345], p_values=[0.9], resume=True
    )
    assert len(commands) == 6  # Reuse the completed combined-p products.
    old_files = {
        p: p.read_bytes()
        for p in output.iterdir()
        if p.suffix == ".h5" or p.name.endswith(".complete.json")
    }
    original_sha = campaign.sha256
    monkeypatch.setattr(
        campaign,
        "sha256",
        lambda path: "a" * 64 if Path(path) == Path(campaign.__file__) else original_sha(path),
    )
    current = campaign.discover(root)
    assert current["inputs"]["wrapper_sha256"] != saved["inputs"]["wrapper_sha256"]
    assert all(campaign.complete(output, saved, job) for job in saved["jobs"])
    campaign.run_campaign(
        root, output, budgets=[500000], seeds=[12345], p_values=[0.9], resume=True, dry_run=True
    )
    assert "7,000,000" in capsys.readouterr().out
    assert len(commands) == 6
    assert json.loads((output / "campaign.json").read_text()) == saved
    campaign.run_campaign(
        root, output, budgets=[500000], seeds=[12345], p_values=[0.9], resume=True
    )
    manifest = json.loads((output / "campaign.json").read_text())
    assert manifest["inputs"] == saved["inputs"] and manifest["fingerprint"] == saved["fingerprint"]
    assert manifest["jobs"][:-1] == saved["jobs"]
    job = manifest["jobs"][-1]
    assert job["prefix"] == "direct3d_p09_N500000_seed12345" and job["p_values"] == [0.9]
    result, receipt = campaign.validated_job(output, manifest, job)
    assert receipt["executed_wrapper_sha256"] == "a" * 64
    assert len(result["cases"]) == 1
    np.testing.assert_array_equal(
        result["cases"][0]["lambda_bin_centers_km"].view("u8"), centers.view("u8")
    )
    assert commands[-1][commands[-1].index("--backend") + 1] == "parallel"
    assert commands[-1][commands[-1].index("--chunk-size") + 1] == "128"
    assert result["cases"][0]["local"]["slopes"].shape == (1, 6, 7)
    assert len(commands) == 7
    campaign.run_campaign(
        root, output, budgets=[500000], seeds=[12345], p_values=[0.9], resume=True
    )
    assert len(commands) == 7
    assert all(path.read_bytes() == before for path, before in old_files.items())
    with pytest.raises(ValueError, match="--resume"):
        campaign.run_campaign(root, output, budgets=[500000], seeds=[12345], p_values=[0.9])
    monkeypatch.setattr(campaign.subprocess, "run", lambda *a, **k: pytest.fail("sampling called"))
    old_reports = {
        name: b"previous two-budget report"
        for name in (
            "convergence_report.md",
            "convergence_exponents.png",
            "convergence_exponents.pdf",
        )
    }
    for name, data in old_reports.items():
        (output / name).write_bytes(data)
    records = campaign.report(output)
    assert all(
        (output / "reports_before_500k" / name).read_bytes() == data
        for name, data in old_reports.items()
    )
    assert len(records) == 13
    np.testing.assert_allclose(campaign.budget_changes(records, 0.9, "moments", 50000, 500000), 0.1)
    np.testing.assert_allclose(campaign.budget_changes(records, 0.9, "moments", 5000, 500000), 0.32)
    assert campaign.aggregate(records, 0.5, 500000, "moments") is None
    text = (output / "convergence_report.md").read_text()
    assert "50k→500k" in text and "5k→500k" in text and "S_n, 500k" in text
    assert "10 (single seed; SD unavailable)" in text
    assert "32 (single seed; SD unavailable)" in text
    assert "CP500000" in text and "insufficient seed repeats" in text
    p05 = text.split("## p=0.5")[1]
    assert "S_n, 500k" not in p05  # p=0.5 was not requested at this budget.
    for extension in ("png", "pdf"):
        assert (output / f"convergence_exponents.{extension}").stat().st_size > 0


def test_500k_case_restriction_and_restart(workspace, monkeypatch):
    root, output, _, _ = workspace
    commands = fake_cli(monkeypatch, fail_at=1)
    with pytest.raises(ValueError, match="requires --p 0.9"):
        campaign.run_campaign(root, output, budgets=[500000])
    with pytest.raises(RuntimeError, match="preserved diagnostics"):
        campaign.run_campaign(root, output, budgets=[500000], seeds=[12345], p_values=[0.9])
    campaign.run_campaign(
        root, output, budgets=[500000], seeds=[12345], p_values=[0.9], resume=True
    )
    assert len(commands) == 2
    manifest = json.loads((output / "campaign.json").read_text())
    assert campaign.complete(output, manifest, manifest["jobs"][-1])
    current = campaign.discover(root)
    current["inputs"]["implementation_sha256"]["src/hs_observatory/fields/malara.py"] = "changed"
    with pytest.raises(ValueError, match="scientific inputs changed"):
        campaign.extend_campaign(manifest, current, [])


def test_500k_cli_selects_one_seed_and_one_p_by_default(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(campaign, "run_campaign", lambda *a, **k: calls.append((a, k)))
    campaign.main(["run", "--budget", "500000", "--output", str(tmp_path), "--dry-run"])
    args, kwargs = calls[0]
    assert args[2] == [500000] and args[3] == (12345,)
    assert kwargs["p_values"] == (0.9,) and kwargs["dry_run"]
