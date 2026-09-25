"""Focused tests for the standalone CDF minimum-separation search."""

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "hs-minimize.py"
    spec = importlib.util.spec_from_file_location("hs_minimize", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mock_records(script, monkeypatch, days, scales=None):
    target = script.parse_time("2030-06-01T00:00:00Z")
    epochs = target + np.asarray(days).astype("timedelta64[D]")
    points = np.zeros((len(days), 9, 3))
    points[:, :, 0] = np.arange(9)
    if scales is not None:
        points *= np.asarray(scales)[:, None, None]
    labels = ["H", *(f"N{i}" for i in range(1, 9))]
    sources = [f"record-{index}.cdf" for index in range(len(days))]
    records = [epochs, points, labels, sources]
    monkeypatch.setattr(script, "read_cdf_positions", lambda path: tuple(records))
    return target, records


def test_pairs_minimum_boundaries_tie_and_chronological_sort(script, monkeypatch):
    target, records = mock_records(script, monkeypatch, [11, 10, 0, -10, -11], [0, 1, 3, 1, 0])
    result = script.find_minimum(Path("cdf-dir"), target)
    np.testing.assert_array_equal(result["offset_days"], [-10, 0, 10])
    # For points x=0..8, the mean square across the 36 pairs is 15.
    np.testing.assert_allclose(result["rms"], np.sqrt(15)*np.array([1, 3, 1]))
    assert result["best"] == 0
    assert result["maximum"] == 1
    assert result["sources"] == [records[3][3], records[3][2], records[3][1]]
    assert result["excluded"] == 0


def test_invalid_records_and_na_slots(script, monkeypatch):
    target, records = mock_records(script, monkeypatch, [-3, -2, -1, 0, 1, 2])
    records[1][0, 2, 0] = np.nan
    records[1][1, 3, 1] = np.inf
    records[1][2, 4, 2] = script.FILL_VALUE_THRESHOLD
    records[1][3, 5, 0] = -1e31
    records[1] = np.concatenate([records[1], np.full((6, 1, 3), np.nan)], axis=1)
    records[2].append("N/A")
    result = script.find_minimum(Path("cdf"), target)
    np.testing.assert_array_equal(result["offset_days"], [1, 2])
    assert result["excluded"] == 4


@pytest.mark.parametrize("text", ["2030-06-01", "2030-06-01T00:00:00",
                                  "2030-06-01T00:00:00Z", "2030-06-01T02:00:00+02:00",
                                  "2030-05-31T17:00:00-07:00"])
def test_utc_normalization(script, text):
    assert script.parse_time(text) == np.datetime64("2030-06-01T00:00:00", "ns")


@pytest.mark.parametrize("text", ["not-a-time", "2030-02-30", "NaT", "3000-01-01"])
def test_invalid_time(script, text):
    with pytest.raises(ValueError, match="Invalid ISO time"):
        script.parse_time(text)


@pytest.mark.parametrize("labels", [["H"]*9, ["H", *[f"N{i}" for i in range(1, 8)]],
                                    ["H", *[f"N{i}" for i in range(1, 10)]]])
def test_invalid_labels(script, monkeypatch, labels):
    target, records = mock_records(script, monkeypatch, [0])
    records[2] = labels
    with pytest.raises(ValueError, match="nine distinct"):
        script.find_minimum(Path("cdf"), target)


def test_empty_window_and_no_valid_positions(script, monkeypatch):
    target, records = mock_records(script, monkeypatch, [11])
    with pytest.raises(ValueError, match="No CDF samples"):
        script.find_minimum(Path("cdf"), target)
    records[0][0] = target
    records[1][:] = np.nan
    with pytest.raises(ValueError, match="No valid nine-spacecraft samples"):
        script.find_minimum(Path("cdf"), target)


def test_csv_and_summary_partial_coverage(script, monkeypatch, tmp_path, capsys):
    mock_records(script, monkeypatch, [-1, 0, 1], [2, 1, 3])
    output = tmp_path / "nested" / "minimum.csv"
    assert script.main(["--cdf-path", "cdf-directory", "--time", "2030-06-01",
                        "--output", str(output)]) == 0
    captured = capsys.readouterr()
    assert "Minimum UTC time: 2030-06-01T00:00:00.000000000Z" in captured.out
    assert "Offset from designated time: +0 days" in captured.out
    assert "Source CDF: record-1.cdf" in captured.out
    assert "Maximum UTC time: 2030-06-02T00:00:00.000000000Z" in captured.out
    assert "Offset from designated time: +1 days" in captured.out
    assert "Source CDF: record-2.cdf" in captured.out
    assert "3 valid, 0 excluded" in captured.out
    assert "Recorded window coverage:" in captured.out
    assert "does not span the full requested" in captured.err
    with output.open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["time_utc", "offset_days", "rms_separation_km", "source_cdf"]
        rows = list(reader)
    assert len(rows) == 3
    assert rows[0]["offset_days"] == "-1.0"
    assert float(rows[1]["rms_separation_km"]) == pytest.approx(np.sqrt(15))
    assert rows[1]["source_cdf"] == "record-1.cdf"


def test_default_output_and_full_coverage(script, monkeypatch, tmp_path, capsys):
    mock_records(script, monkeypatch, [-10, 0, 10])
    monkeypatch.chdir(tmp_path)
    assert script.main(["--cdf-path", "mission.cdf", "--time", "2030-06-01"]) == 0
    assert (tmp_path / "hs-minimize.csv").exists()
    assert "Warning" not in capsys.readouterr().err


def test_cli_error_and_help(script, monkeypatch, capsys):
    def missing(path):
        raise FileNotFoundError("CDF source does not exist")
    monkeypatch.setattr(script, "read_cdf_positions", missing)
    with pytest.raises(SystemExit) as failure:
        script.main(["--cdf-path", "missing.cdf", "--time", "2030-06-01"])
    assert failure.value.code == 1
    assert "CDF source does not exist" in capsys.readouterr().err
    with pytest.raises(SystemExit) as help_exit:
        script.main(["--help"])
    assert help_exit.value.code == 0
    assert "--cdf-path" in capsys.readouterr().out


def test_maximum_tie_selects_earliest_and_single_sample(script, monkeypatch):
    target, _ = mock_records(script, monkeypatch, [9, -9, 0], [3, 3, 1])
    result = script.find_minimum(Path("cdf"), target)
    assert result["offset_days"][result["maximum"]] == -9
    target, _ = mock_records(script, monkeypatch, [0])
    result = script.find_minimum(Path("cdf"), target)
    assert result["best"] == result["maximum"] == 0
