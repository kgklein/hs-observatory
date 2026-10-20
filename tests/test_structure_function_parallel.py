"""Small, real-spawn regressions against the legacy serial SF path."""

from contextlib import contextmanager
from pathlib import Path
import pickle
import time

import h5py
import numpy as np
import pytest
import yaml

from hs_observatory.analysis import structure_function as sf
from hs_observatory.analysis import structure_function_local as local
from hs_observatory.analysis import structure_function_parallel as parallel
from hs_observatory.analyze import analyze_from_config
from hs_observatory.cli import analyze_main
from test_structure_function import _selection_config, _write_structure_input


_ORIGINAL_WORKER = parallel.evaluate_member


@pytest.fixture(autouse=True)
def worker_environment(monkeypatch, tmp_path):
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))


def assert_results_equal(serial, actual):
    assert serial["metadata"]["created_utc"]
    assert actual["metadata"]["created_utc"]
    for result in (serial, actual):
        result["metadata"] = {k: v for k, v in result["metadata"].items() if k != "created_utc"}
    np.testing.assert_equal(serial, actual)


def sampling_config(mode="random", **updates):
    sampling = {"mode": mode, "chunk_size": 7, "random_seed": 12345}
    if mode == "random":
        sampling["n_pairs"] = 23
    if mode == "adaptive_stratified":
        sampling.update(target_pairs_per_bin=3, max_candidates=100, binning_n_pairs=20)
    if mode == "geometry_stratified":
        sampling["target_pairs_per_bin"] = [3, 7, 2]
    return _selection_config(
        **{"increment_mode": "all_spacetime_pairs", "pair_sampling": sampling, **updates}
    )


@pytest.mark.parametrize(
    "config",
    [
        _selection_config(),
        _selection_config(lambda_bins_km=None, scale_ranges_km={}),
        sampling_config("all"),
        sampling_config("random", mc_indices=[1, 0]),
        sampling_config("random", lambda_bins_km=None, scale_ranges_km={}),
        sampling_config("adaptive_stratified", scale_ranges_km={}),
        sampling_config("adaptive_stratified", lambda_bins_km=None),
        sampling_config("geometry_stratified", scale_ranges_km={}),
        sampling_config("geometry_stratified", lambda_bins_km=None, n_lambda_bins=3),
        sampling_config("random", data_groups=["truth"]),
        sampling_config("all", data_groups=["observed"], mc_indices=[1]),
    ],
)
def test_real_spawn_exactly_matches_serial(tmp_path, config):
    path = _write_structure_input(tmp_path / "input.h5")
    analysis = sf.StructureFunctionAnalysis(config)
    assert_results_equal(analysis.run(path), analysis.run(path, workers=2))


@contextmanager
def record_calculations():
    """Capture kernel inputs/order through module aliases, including in children."""
    records = []
    originals = {}

    def wrap_iterator(name, original):
        def traced(*args, **kwargs):
            for left, right in original(*args, **kwargs):
                records.append((name, left.copy(), right.copy()))
                yield left, right

        return traced

    for name in ("pair_chunks", "geometry_pair_chunks"):
        originals[name] = getattr(sf, name)
        setattr(sf, name, wrap_iterator(name, originals[name]))
    originals["adaptive_stratified_pairs"] = sf.adaptive_stratified_pairs
    originals["structure_function_moment_sums"] = sf.structure_function_moment_sums

    def adaptive(*args, **kwargs):
        left, right, diagnostics = originals["adaptive_stratified_pairs"](*args, **kwargs)
        records.append(("adaptive", left.copy(), right.copy()))
        return left, right, diagnostics

    def moments(*args, **kwargs):
        result = originals["structure_function_moment_sums"](*args, **kwargs)
        records.append(("moments", result.copy()))
        return result

    sf.adaptive_stratified_pairs = adaptive
    sf.structure_function_moment_sums = moments
    try:
        yield records
    finally:
        for name, original in originals.items():
            setattr(sf, name, original)


def traced_worker(task):
    with record_calculations() as records:
        result = _ORIGINAL_WORKER(task)
    trace_path = Path(task.path).parent / f"trace-{task.slot}.pickle"
    trace_path.write_bytes(pickle.dumps(records))
    return result


@pytest.mark.parametrize("mode", ["all", "random", "adaptive_stratified", "geometry_stratified"])
def test_ordered_pair_identities_chunks_and_moment_sums(tmp_path, monkeypatch, mode):
    path = _write_structure_input(tmp_path / "input.h5")
    analysis = sf.StructureFunctionAnalysis(sampling_config(mode))
    with record_calculations() as serial_records:
        serial = analysis.run(path)
    monkeypatch.setattr(parallel, "evaluate_member", traced_worker)
    actual = analysis.run(path, workers=2)
    parallel_records = []
    for slot in range(3):
        parallel_records.extend(pickle.loads((tmp_path / f"trace-{slot}.pickle").read_bytes()))
    np.testing.assert_equal(serial_records, parallel_records)
    assert_results_equal(serial, actual)


def delayed_worker(task):
    if task.slot == 0:
        time.sleep(0.8)
    result = _ORIGINAL_WORKER(task)
    with (Path(task.path).parent / "completion.txt").open("a") as stream:
        stream.write(f"{task.slot}\n")
    return result


def failing_worker(task):
    if task.group == "observed" and task.index == 1:
        raise ValueError("injected member failure")
    return _ORIGINAL_WORKER(task)


def wrong_identity_worker(task):
    _, result = _ORIGINAL_WORKER(task)
    return (task.group, task.index, 999), result


def incomplete_worker(task):
    return task.identity, None


def test_completion_order_does_not_change_member_order(tmp_path, monkeypatch):
    path = _write_structure_input(tmp_path / "input.h5")
    analysis = sf.StructureFunctionAnalysis(sampling_config(mc_indices=[1, 0]))
    monkeypatch.setattr(parallel, "evaluate_member", delayed_worker)
    serial = analysis.run(path)
    actual = analysis.run(path, workers=2)
    assert (tmp_path / "completion.txt").read_text().splitlines()[0] != "0"
    assert actual["products"]["observed"]["mc_indices"] == [1, 0]
    assert_results_equal(serial, actual)


@pytest.mark.parametrize("worker", [failing_worker, wrong_identity_worker, incomplete_worker])
def test_worker_failure_preserves_successful_outputs(tmp_path, monkeypatch, worker):
    path = _write_structure_input(tmp_path / "input.h5")
    output = tmp_path / "result.json"
    figure = tmp_path / "result.png"
    output.write_text("previous successful JSON")
    figure.write_bytes(b"previous successful figure")
    config = {
        "input": {"path": str(path)},
        "output": {"path": str(output)},
        "analysis": {"method": "structure_function"},
        "structure_function": sampling_config(
            plot={"publication": {"enabled": True, "path": str(figure)}}
        ),
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(parallel, "evaluate_member", worker)
    with pytest.raises(RuntimeError, match=r"SF worker failed for observed member \d") as error:
        analyze_from_config(config_path, workers=2)
    assert error.value.__cause__ is not None
    assert output.read_text() == "previous successful JSON"
    assert figure.read_bytes() == b"previous successful figure"


def test_missing_result_refuses_assembly():
    with pytest.raises(RuntimeError, match="incomplete SF ensemble"):
        parallel._assemble_members([object()], [], {})


def test_zero_mc_and_single_stored_observation(tmp_path):
    path = _write_structure_input(tmp_path / "input.h5")
    with h5py.File(path, "a") as handle:
        for variable in ("n", "v", "B"):
            values = handle[f"observed/{variable}"][0]
            del handle[f"observed/{variable}"]
            handle.create_dataset(f"observed/{variable}", data=values)
    analysis = sf.StructureFunctionAnalysis(_selection_config())
    assert_results_equal(analysis.run(path), analysis.run(path, workers=2))
    with h5py.File(path, "a") as handle:
        for variable in ("n", "v", "B"):
            shape = handle[f"observed/{variable}"].shape
            del handle[f"observed/{variable}"]
            handle.create_dataset(f"observed/{variable}", shape=(0, *shape), dtype=float)
    truth = sf.StructureFunctionAnalysis(_selection_config(data_groups=["truth"]))
    assert_results_equal(truth.run(path), truth.run(path, workers=2))
    for workers in (1, 2):
        with pytest.raises(ValueError, match="selects no observed realizations"):
            analysis.run(path, workers=workers)


def test_cli_workers_validation_and_forwarding(monkeypatch, capsys):
    import hs_observatory.analyze as workflow

    calls = []

    def run(config, *, workers=1):
        calls.append((config, workers))
        return Path("result.json")

    monkeypatch.setattr(workflow, "analyze_from_config", run)
    assert analyze_main(["input.yaml"]) == 0
    assert analyze_main(["input.yaml", "--workers", "2"]) == 0
    assert calls == [("input.yaml", 1), ("input.yaml", 2)]
    for value in ("0", "-1", "1.5", "abc"):
        with pytest.raises(SystemExit) as error:
            analyze_main(["input.yaml", "--workers", value])
        assert error.value.code == 2
    assert "positive integer" in capsys.readouterr().err


def test_other_analysis_rejects_parallel_workers(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "input": {"path": "unused.h5"},
                "output": {"path": "unused.json"},
                "analysis": {"method": "elsasser"},
            }
        )
    )
    with pytest.raises(ValueError, match="supported only for structure_function"):
        analyze_from_config(path, workers=2)


@pytest.mark.parametrize(
    "mode,targets",
    [
        ("adaptive_stratified", 1),
        ("geometry_stratified", [1000, 1000, 1000]),
        ("geometry_stratified", [0, 0, 0]),
    ],
)
def test_sampling_success_census_and_empty_fallback(tmp_path, mode, targets):
    path = _write_structure_input(tmp_path / "input.h5")
    config = sampling_config(mode, scale_ranges_km={})
    config["pair_sampling"]["target_pairs_per_bin"] = targets
    if mode == "adaptive_stratified":
        config["lambda_bins_km"] = [0.1, 50.0]
    analysis = sf.StructureFunctionAnalysis(config)
    serial = analysis.run(path)
    assert_results_equal(serial, analysis.run(path, workers=2))
    if mode == "adaptive_stratified":
        for product in serial["products"].values():
            assert all(product["spacetime_pair_counts"]["sampling_target_reached"])
    if targets == [0, 0, 0]:
        assert serial["products"]["observed"]["scale_ranges_km"] == {"total": [0.1, 50.0]}


def test_logical_geometry_cache_reuse_eviction_and_cross_group(tmp_path):
    path = _write_structure_input(tmp_path / "input.h5")
    with h5py.File(path, "a") as handle:
        for variable in ("n", "v", "B"):
            values = handle[f"observed/{variable}"][0]
            del handle[f"observed/{variable}"]
            handle.create_dataset(f"observed/{variable}", data=np.repeat(values[None], 6, axis=0))
        offsets = np.zeros((6, 3, 3))
        offsets[:, 1, 0] = [0, 1, 0, 2, 1, 0]
        del handle["errors/position_offset"]
        handle.create_dataset("errors/position_offset", data=offsets)
    analysis = sf.StructureFunctionAnalysis(sampling_config("geometry_stratified"))
    serial = analysis.run(path)
    actual = analysis.run(path, workers=2)
    assert_results_equal(serial, actual)
    observed = actual["products"]["observed"]["spacetime_pair_counts"]["geometry_sampling"]
    assert [m["plan_reused"] for m in observed["members"]] == [
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    truth = actual["products"]["truth"]["spacetime_pair_counts"]["geometry_sampling"]
    assert truth["members"][0]["plan_reused"] is True


def test_downstream_local_slopes_percentiles_and_hdf5(tmp_path):
    import json

    path = _write_structure_input(tmp_path / "input.h5")
    analysis = sf.StructureFunctionAnalysis(sampling_config("geometry_stratified"))
    results = [analysis.run(path), analysis.run(path, workers=2)]
    for field in sf.FIELDS:
        for key in ("S_n", "flatness"):
            np.testing.assert_equal(
                sf._ensemble_interval(np.asarray(results[0]["products"]["observed"][field][key])),
                sf._ensemble_interval(np.asarray(results[1]["products"]["observed"][field][key])),
            )
        np.testing.assert_equal(
            sf._ensemble_interval(
                np.asarray(results[0]["products"]["observed"][field]["scaling_exponents"]["raw"])
            ),
            sf._ensemble_interval(
                np.asarray(results[1]["products"]["observed"][field]["scaling_exponents"]["raw"])
            ),
        )
        for method in ("adjacent", "three-point"):
            cases_by_run = []
            for index, result in enumerate(results):
                source = tmp_path / f"result-{index}.json"
                source.write_text(json.dumps(result))
                cases = local.load_local_cases(
                    {"cases": {"fixture": {"p_value": 0.9, "path": str(source)}}},
                    field,
                    method,
                    False,
                )
                cases[0]["source_path"] = "fixture.json"
                cases_by_run.append(cases)
                local.write_local_results(tmp_path / f"local-{index}.h5", cases, {"test": True})
            np.testing.assert_equal(cases_by_run[0], cases_by_run[1])
            with (
                h5py.File(tmp_path / "local-0.h5") as first,
                h5py.File(tmp_path / "local-1.h5") as other,
            ):

                def compare(name, node):
                    peer = other[name]
                    assert type(node) is type(peer)
                    np.testing.assert_equal(dict(node.attrs), dict(peer.attrs))
                    if isinstance(node, h5py.Dataset):
                        assert node.shape == peer.shape and node.dtype == peer.dtype
                        np.testing.assert_array_equal(node[...], peer[...])

                first.visititems(compare)
                np.testing.assert_equal(dict(first.attrs), dict(other.attrs))


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_python_api_rejects_invalid_workers(value):
    with pytest.raises(ValueError, match="positive integer"):
        sf.StructureFunctionAnalysis({}).run("unused.h5", workers=value)
    with pytest.raises(ValueError, match="positive integer"):
        analyze_from_config("unused.yaml", workers=value)


@contextmanager
def guard_selected_hdf5_reads():
    original_array = h5py.Dataset.__array__
    original_getitem = h5py.Dataset.__getitem__

    def asarray(dataset, *args, **kwargs):
        assert not dataset.name.startswith("/truth/")
        assert dataset.name not in ("/observed/n", "/observed/v", "/observed/B")
        return original_array(dataset, *args, **kwargs)

    def getitem(dataset, key, *args, **kwargs):
        assert not dataset.name.startswith("/truth/")
        if dataset.name in ("/observed/n", "/observed/v", "/observed/B", "/errors/position_offset"):
            assert key == 1
        return original_getitem(dataset, key, *args, **kwargs)

    h5py.Dataset.__array__ = asarray
    h5py.Dataset.__getitem__ = getitem
    try:
        yield
    finally:
        h5py.Dataset.__array__ = original_array
        h5py.Dataset.__getitem__ = original_getitem


def guarded_worker(task):
    with guard_selected_hdf5_reads():
        return _ORIGINAL_WORKER(task)


def test_preparation_releases_fields_and_workers_read_only_selected_member(tmp_path, monkeypatch):
    import weakref

    path = _write_structure_input(tmp_path / "input.h5")
    analysis = sf.StructureFunctionAnalysis(
        sampling_config(data_groups=["observed"], mc_indices=[1])
    )
    serial = analysis.run(path)
    references = []
    original_read = parallel._read_member
    original_collect = parallel._collect_members

    def read(*args, **kwargs):
        fields, positions, source, notes = original_read(*args, **kwargs)
        references.extend([weakref.ref(positions), weakref.ref(fields["z_plus"])])
        return fields, positions, source, notes

    def collect(tasks, workers):
        assert all(reference() is None for reference in references)
        assert len(tasks) == 1 and workers == 2
        for task in tasks:
            assert task.path == str(path.resolve())
            # Only small common bin edges, not fields, occur in task descriptors.
            arrays = [value for value in vars(task).values() if isinstance(value, np.ndarray)]
            assert len(arrays) == 1 and arrays[0].shape == (4,)
        return original_collect(tasks, workers)

    monkeypatch.setattr(parallel, "_read_member", read)
    monkeypatch.setattr(parallel, "_collect_members", collect)
    monkeypatch.setattr(parallel, "evaluate_member", guarded_worker)
    with guard_selected_hdf5_reads():
        actual = analysis.run(path, workers=2)
    assert_results_equal(serial, actual)


def test_real_cli_parallel_output_matches_serial(tmp_path):
    import json

    source = _write_structure_input(tmp_path / "input.h5")
    output = tmp_path / "output.json"
    options = sampling_config(data_groups=["truth", "observed"], mc_indices=[1, 0])
    config = {
        "input": {"path": str(source)},
        "output": {"path": str(output)},
        "analysis": {"method": "structure_function"},
        "structure_function": options,
    }
    config_path = tmp_path / "input.yaml"
    config_path.write_text(yaml.safe_dump(config))
    assert analyze_main([str(config_path), "--workers", "2"]) == 0
    actual = json.loads(output.read_text())
    assert "workers" not in actual["config"] and "workers" not in actual["metadata"]
    assert_results_equal(sf.StructureFunctionAnalysis(options).run(source), actual)
