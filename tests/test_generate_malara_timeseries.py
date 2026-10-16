"""Orchestration checks without generating fields."""

import importlib.util
from pathlib import Path

import pytest
import yaml

spec = importlib.util.spec_from_file_location(
    "generate_malara_timeseries",
    Path(__file__).resolve().parents[1] / "scripts/generate_malara_timeseries.py",
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def recipes(root):
    (root / "generation").mkdir()
    for duration in (300, 600, 900, 1200, 1500):
        path = root / "generation" / f"malara-ts-502-t{duration}.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "time": {"t_start_s": 0, "t_stop_s": duration},
                    "output": {"path": f"outputs/t{duration}.h5"},
                }
            )
        )


def test_dry_run_orders_pairs_and_skips_existing(tmp_path, monkeypatch, capsys):
    recipes(tmp_path)
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs/t300.h5").touch()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: pytest_fail())
    assert runner.main(["--dry-run"]) == 0
    text = capsys.readouterr().out
    assert "Batch 1: malara-ts-502-t1500.yaml, malara-ts-502-t1200.yaml" in text
    assert "Batch 2: malara-ts-502-t900.yaml, malara-ts-502-t600.yaml" in text
    assert "Batch 3:" not in text
    assert len(runner.discover_cases(tmp_path, overwrite=True)) == 5


def pytest_fail():
    raise AssertionError("dry run must not start processes")


def test_failed_pair_stops_remaining_batch(tmp_path, monkeypatch):
    recipes(tmp_path)
    binary = tmp_path / ".venv/bin/hsobs-generate"
    binary.parent.mkdir(parents=True)
    binary.touch()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    calls = []

    def pair(cases, root, logs):
        calls.append([duration for duration, _ in cases])
        return False

    monkeypatch.setattr(runner, "run_batch", pair)
    assert runner.main([]) == 1
    assert calls == [[1500, 1200]]


@pytest.mark.parametrize("jobs", [1, 2, 3])
def test_run_batch_starts_all_before_waiting(tmp_path, monkeypatch, jobs):
    events = []

    class Process:
        pid = 123

        def __init__(self, command, **kwargs):
            events.append(("start", Path(command[1]).stem))
            assert kwargs["env"]["OPENBLAS_NUM_THREADS"] == "1"
            assert kwargs["start_new_session"]

        def wait(self):
            events.append(("wait", None))
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    batch = [(duration, Path(f"t{duration}.yaml")) for duration in (1500, 1200, 900)[:jobs]]
    assert runner.run_batch(batch, tmp_path, tmp_path)
    assert [event[0] for event in events] == ["start"] * jobs + ["wait"] * jobs


def test_three_jobs_runs_longest_three_then_remaining_two(tmp_path, monkeypatch):
    recipes(tmp_path)
    binary = tmp_path / ".venv/bin/hsobs-generate"
    binary.parent.mkdir(parents=True)
    binary.touch()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    batches = []

    def batch(cases, root, logs):
        batches.append([duration for duration, _ in cases])
        return True

    monkeypatch.setattr(runner, "run_batch", batch)
    assert runner.main(["--jobs", "3"]) == 0
    assert batches == [[1500, 1200, 900], [600, 300]]


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "invalid"])
def test_invalid_jobs_rejected(value, capsys):
    with pytest.raises(SystemExit) as error:
        runner.main(["--dry-run", "--jobs", value])
    assert error.value.code == 2
    assert "--jobs" in capsys.readouterr().err
