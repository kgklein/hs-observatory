import h5py
import numpy as np

from hs_observatory.io.hdf5 import read_observations, write_observations


def test_hdf5_roundtrip_has_expected_groups_and_shapes(tmp_path):
    n_mc, n_time, n_sc = 2, 5, 9
    truth = {
        "n": np.ones((n_time, n_sc)),
        "v": np.ones((n_time, n_sc, 3)),
        "B": np.ones((n_time, n_sc, 3)),
    }
    observed = {
        name: np.broadcast_to(value, (n_mc, *value.shape)).copy() for name, value in truth.items()
    }
    errors = {name: np.zeros_like(value) for name, value in observed.items()}
    path = tmp_path / "observations.h5"

    time = {quantity: np.arange(n_time) for quantity in ("n", "v", "B")}
    position = {quantity: np.zeros((n_sc, n_time, 3)) for quantity in ("n", "v", "B")}
    write_observations(
        path,
        time=time,
        spacecraft_ids=[f"S{i}" for i in range(n_sc)],
        position=position,
        truth=truth,
        observed=observed,
        errors=errors,
        metadata={"config_yaml": "test: true", "random_seed": 1},
    )

    with h5py.File(path) as handle:
        for name in ("metadata", "time", "spacecraft", "truth", "observed", "errors"):
            assert name in handle
        assert handle["time/n"].shape == (n_time,)
        assert handle["spacecraft/r/B"].shape == (n_sc, n_time, 3)
        assert handle["observed/v"].shape == (n_mc, n_time, n_sc, 3)
        assert handle["observed/v"].compression == "gzip"
    loaded = read_observations(path)
    assert loaded["spacecraft"]["id"][0] == "S0"
    assert np.array_equal(loaded["time"]["B"], time["B"])
    assert np.array_equal(loaded["truth"]["n"], truth["n"])
