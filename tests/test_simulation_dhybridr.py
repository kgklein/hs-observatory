import copy
import json
import weakref

import h5py
import numpy as np
import pytest
import yaml

from hs_observatory.fields.simulation_sampled import (
    SimulationSampledField,
    hub_centered_coordinates,
)
from hs_observatory.generate import generate_from_config
from hs_observatory.io.hdf5 import read_observations
from hs_observatory.io.simulation_dhybridr import (
    ARRAY_ORDER,
    load_simulation_dhybridr,
    proton_normalization,
)
from hs_observatory.spacecraft.configuration import synthetic_configuration


@pytest.fixture
def archives(tmp_path):
    # Asymmetric grid and nonuniform snapshot cadence expose axis/index mistakes.
    x, y, z = [0.25 + 0.5 * np.arange(n) for n in (8, 4, 6)]
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    spatial = xx + 10 * yy + 100 * zz
    values = np.stack([spatial + offset for offset in (0, 1000, 2000)]).astype(np.float32)
    meta = {
        "iter": np.array([0, 1, 200]),
        "time": np.array([0.0, 0.005, 1.0]),
        "x": x,
        "y": y,
        "z": z,
        "array_order": np.array(ARRAY_ORDER),
    }
    fields = dict(meta)
    moments = dict(meta)
    for i, name in enumerate(("Bx", "By", "Bz")):
        fields[name] = values + 100 * i
        fields[f"units_{name}"] = np.array("B_0")
    for i, name in enumerate(("n", "Vx", "Vy", "Vz")):
        moments[name] = values + 10 * i
        moments[f"units_{name}"] = np.array("n_0 e" if name == "n" else "v_A")
    # Unused products must never be read, even when malformed.
    fields["Ex"] = np.array("not a field")
    moments["Pxx"] = np.array("not a pressure")
    paths = (tmp_path / "fields.npz", tmp_path / "moments.npz")
    np.savez_compressed(paths[0], **fields)
    np.savez_compressed(paths[1], **moments)
    return paths, fields, moments


def config_for(archives):
    paths, _, _ = archives
    return {
        "model": "simulation_sampled",
        "background": {"n": 999.0, "v": [999.0, 999.0, 999.0], "B": [999.0, 999.0, 999.0]},
        "simulation": {
            "format": "dhybridr_npz",
            "sources": {"fields": str(paths[0]), "moments": str(paths[1])},
            "iteration": 1,
            "normalization": {"n0_cm3": 5.0, "B0_nT": 5.0},
        },
    }


def test_loader_selects_iteration_and_preserves_components(archives):
    paths, fields, moments = archives
    snapshot = load_simulation_dhybridr(*paths, iteration=1)
    assert snapshot.grid_shape == (8, 4, 6)
    assert snapshot.source_shape == (3, 6, 4, 8)
    assert snapshot.available_fields == ("n", "v", "B")
    assert snapshot.layout["snapshot_time_Omega_ci_inv"] == 0.005
    assert snapshot.layout["snapshot_index"] == 1
    np.testing.assert_array_equal(snapshot.n, moments["n"][1].transpose(2, 1, 0))
    for i, name in enumerate(("Vx", "Vy", "Vz")):
        np.testing.assert_array_equal(snapshot.v[..., i], moments[name][1].transpose(2, 1, 0))
    for i, name in enumerate(("Bx", "By", "Bz")):
        np.testing.assert_array_equal(snapshot.B[..., i], fields[name][1].transpose(2, 1, 0))
    # V is direct velocity, not momentum/n. Non-unit density makes this distinguishable.
    assert snapshot.v[0, 0, 0, 0] > snapshot.n[0, 0, 0]
    np.testing.assert_array_equal(snapshot.layout["origin_di"], [0.25] * 3)
    np.testing.assert_array_equal(snapshot.layout["box_size_di"], [4.0, 2.0, 3.0])


def test_loader_releases_histories_and_reads_only_required_members(archives, monkeypatch):
    paths, _, _ = archives
    original = np.load
    previous = None
    accessed = []
    large = {"Bx", "By", "Bz", "n", "Vx", "Vy", "Vz"}

    class Archive:
        def __init__(self, path, **kwargs):
            self.archive = original(path, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.archive.close()

        def __getitem__(self, key):
            nonlocal previous
            accessed.append(key)
            assert key not in {"Ex", "Pxx"}
            if key in large:
                assert previous is None or previous() is None
            value = self.archive[key]
            if key in large:
                previous = weakref.ref(value)
            return value

    monkeypatch.setattr(np, "load", Archive)
    snapshot = load_simulation_dhybridr(*paths, 200)
    assert previous() is None
    assert set(accessed) & large == large
    assert all(accessed.count(key) == 1 for key in large)
    assert snapshot.n.base is None


@pytest.mark.parametrize("iteration", [2, -1, 1.0, True, None])
def test_loader_rejects_missing_or_noninteger_iteration(archives, iteration):
    with pytest.raises(ValueError, match="iteration"):
        load_simulation_dhybridr(*archives[0], iteration)


@pytest.mark.parametrize(
    "member,value,match",
    [
        ("iter", np.array([0, 1, 201]), "iter arrays must agree"),
        ("time", np.array([0.0, 0.006, 1.0]), "time arrays must agree"),
        ("x", 0.25 + 0.5 * np.arange(9), "x arrays must agree"),
        ("iter", np.array([0, 1, 1]), "unique"),
        ("iter", np.array([0.0, 1.0, 200.0]), "integer"),
        ("time", np.array([0.0, np.nan, 1.0]), "finite"),
        ("array_order", np.array("data[t,x,y,z]"), "array_order"),
        ("x", np.array([0.25, 0.75, 1.4]), "uniform"),
        ("x", np.array([0.0, 0.5, 1.0]), "half-cell"),
        ("x", np.array([0.25]), ">=2 cells"),
        ("units_Vx", np.array("momentum_density"), "units_Vx"),
        ("n", np.zeros((3, 6, 4, 7)), "shape"),
        ("n", np.full((3, 6, 4, 8), np.nan), "finite"),
        ("n", np.zeros((3, 6, 4, 8), dtype=complex), "real"),
    ],
)
def test_loader_rejects_invalid_archive(archives, member, value, match):
    paths, _, moments = archives
    moments[member] = value
    np.savez_compressed(paths[1], **moments)
    with pytest.raises(ValueError, match=match):
        load_simulation_dhybridr(*paths, 1)


def test_missing_member(archives):
    paths, fields, _ = archives
    del fields["Bx"]
    np.savez(paths[0], **fields)
    with pytest.raises(ValueError, match="Missing required.*Bx"):
        load_simulation_dhybridr(*paths, 1)


def test_physical_scales_and_total_quantities(archives):
    field = SimulationSampledField(config_for(archives))
    scales = field.metadata["normalization"]
    # Independent SI calculation (constants match documented CODATA precision).
    mu, mp, e = 1.25663706212e-6, 1.67262192369e-27, 1.602176634e-19
    expected_v = 5e-9 / np.sqrt(mu * mp * 5e6) / 1000
    expected_d = np.sqrt(mp / (mu * 5e6 * e**2)) / 1000
    expected_o = e * 5e-9 / mp
    np.testing.assert_allclose(
        [scales["vA_km_s"], scales["di_km"], scales["Omega_ci_s_inv"]],
        [expected_v, expected_d, expected_o],
        rtol=1e-14,
    )
    assert scales["di_km"] * scales["Omega_ci_s_inv"] == pytest.approx(expected_v)
    result = field.evaluate(np.array([0.0]), np.zeros((1, 1, 3)))
    # Default mapped node is [4,2,3], corresponding to source [2.25,1.25,1.75].
    n_code = 1000 + 2.25 + 10 * 1.25 + 100 * 1.75
    assert result["n"][0, 0] == pytest.approx(n_code * 5)
    np.testing.assert_allclose(result["v"][0, 0], (n_code + np.array([10, 20, 30])) * expected_v)
    np.testing.assert_allclose(result["B"][0, 0], (n_code + np.array([0, 100, 200])) * 5)


@pytest.mark.parametrize(
    "n0,B0", [(0, 5), (-1, 5), (5, np.nan), (np.inf, 5), (5, 0), (True, 5), (1e308, 5)]
)
def test_invalid_normalization(n0, B0):
    with pytest.raises(ValueError, match="positive finite|invalid physical scales"):
        proton_normalization(n0, B0)


@pytest.mark.parametrize(
    "setting,value,match",
    [
        ("normalization", {}, "requires normalization"),
        ("fields_are", {"B": "fluctuation"}, "total"),
        ("scale_factors", {"v": 1.0}, "conflicts"),
        ("scale_factors", {"n": np.nan}, "conflicts"),
        ("coordinates", {"units": "code", "code_length_to_km": 1.0}, "conflicts"),
        ("coordinates", {"units": "km", "box_size_km": [1, 2, 3]}, "conflicts"),
        ("coordinates", {"units": "code", "box_size_code": 4.0}, "conflicts"),
        ("coordinates", {"units": "rho_p"}, "units"),
        ("boundary", "periodic", "requires boundary"),
        ("initial_hub_position_di", [0, 1, 1], "Initial dHybridR x"),
        ("initial_hub_position_di", [4.0, 1, 1], "Initial dHybridR x"),
        ("initial_hub_position_di", [1, np.nan, 1], "three finite"),
    ],
)
def test_rejects_inconsistent_configuration(archives, setting, value, match):
    config = config_for(archives)
    config["simulation"][setting] = value
    with pytest.raises((ValueError, NotImplementedError), match=match):
        SimulationSampledField(config)


def test_accepts_consistent_explicit_scales_without_mutating_config(archives):
    config = config_for(archives)
    scales = proton_normalization(5.0, 5.0)
    config["simulation"]["scale_factors"] = {"n": 5.0, "B": 5.0, "v": scales["vA_km_s"]}
    config["simulation"]["coordinates"] = {
        "units": "code",
        "box_size_code": [4.0, 2.0, 3.0],
        "code_length_to_km": scales["di_km"],
    }
    before = copy.deepcopy(config)
    field = SimulationSampledField(config)
    assert config == before
    np.testing.assert_allclose(field.box_size_km, np.array([4, 2, 3]) * scales["di_km"])


def test_default_mapping_is_exactly_legacy_mapping(archives):
    field = SimulationSampledField(config_for(archives))
    position = np.array(
        [
            [[100.0, 200.0, 300.0], [101.0, 202.0, 303.0]],
            [[103.0, 204.0, 305.0], [104.0, 206.0, 308.0]],
        ]
    )
    box = field.box_size_km
    expected = np.mod(position - position[0, 0] + 0.5 * box, box)
    np.testing.assert_array_equal(hub_centered_coordinates(position, box), expected)
    # Compare against the unchanged generic NPY source using the same code arrays/scales.
    npy = archives[0][0].with_suffix(".npy")
    np.save(
        npy,
        np.concatenate((field.snapshot.n[..., None], field.snapshot.v, field.snapshot.B), axis=-1),
    )
    legacy = SimulationSampledField(
        {
            "simulation": {
                "path": str(npy),
                "coordinates": {"units": "km", "box_size_km": box.tolist()},
                "scale_factors": field.scale_factors,
            }
        }
    )
    for q in ("n", "v", "B"):
        np.testing.assert_array_equal(
            field.evaluate_quantity(q, np.array([0.0, 1.0]), position),
            legacy.evaluate_quantity(q, np.array([0.0, 1.0]), position),
        )
    assert legacy.metadata["boundary"] == "periodic"


@pytest.mark.parametrize("x", [0.25, 0.75, 1.0, 3.75])
def test_custom_hub_centers_and_midcell_interpolation(archives, x):
    config = config_for(archives)
    config["simulation"]["initial_hub_position_di"] = [x, 0.75, 1.25]
    field = SimulationSampledField(config)
    result = field.evaluate(np.array([0.0]), np.zeros((1, 1, 3)))
    expected = 1000 + x + 10 * 0.75 + 100 * 1.25
    assert result["n"][0, 0] == pytest.approx(expected * 5)
    np.testing.assert_allclose(result["B"][0, 0], (expected + np.array([0, 100, 200])) * 5)
    assert field.metadata["initial_hub_position_di"] == [x, 0.75, 1.25]


def test_eight_corner_interpolation_and_exact_final_plane(archives):
    config = config_for(archives)
    config["simulation"]["initial_hub_position_di"] = [1.0, 1.0, 1.0]
    field = SimulationSampledField(config)
    actual = field.evaluate_quantity("B", np.array([0.0]), np.zeros((1, 1, 3)))
    expected = field.snapshot.B[1:3, 1:3, 1:3].mean(axis=(0, 1, 2)) * 5
    np.testing.assert_allclose(actual[0, 0], expected)

    config["simulation"]["initial_hub_position_di"] = [3.75, 1.0, 1.0]
    field = SimulationSampledField(config)
    # A tiny wrapped weight would be catastrophic and expose endpoint leakage.
    field.snapshot.B[0] = 1e300
    expected = field.snapshot.B[-1, 1:3, 1:3].mean(axis=(0, 1)) * 5
    actual = field.evaluate_quantity("B", np.array([0.0]), np.zeros((1, 1, 3)))
    np.testing.assert_allclose(actual[0, 0], expected)


@pytest.mark.parametrize("side", [-1, 1])
def test_rejects_immediately_outside_cell_centers(archives, side):
    config = config_for(archives)
    center = 0.25 if side < 0 else 3.75
    config["simulation"]["initial_hub_position_di"] = [center, 0.75, 1.25]
    field = SimulationSampledField(config)
    field.set_coordinate_mapping_reference(np.zeros(3))
    extent = (3.75 - 0.25) * field.normalization["di_km"]
    # Smallest representable displacement past the represented km endpoints.
    delta = -np.spacing(extent) if side < 0 else np.nextafter(extent, np.inf) - extent
    with pytest.raises(ValueError, match=r"dHybridR x=.*interval.*wrapping is forbidden"):
        field.evaluate_quantity("n", np.array([0.0]), np.array([[[delta, 0.0, 0.0]]]))


def test_periodic_yz_and_forbidden_whole_box_x_excursion(archives):
    config = config_for(archives)
    config["simulation"]["initial_hub_position_di"] = [1.25, -3.25, 7.25]
    field = SimulationSampledField(config)
    field.set_coordinate_mapping_reference(np.zeros(3))
    box = field.box_size_km
    positions = np.array([[[0, 0, 0]], [[0, box[1], -box[2]]]])
    result = field.evaluate_quantity("n", np.array([0.0]), positions)
    np.testing.assert_allclose(result[0], [result[0, 0]] * 2)
    for sign in (-1, 1):
        with pytest.raises(ValueError, match="wrapping is forbidden"):
            field.evaluate_quantity("n", np.array([0.0]), np.array([[[sign * box[0], 0, 0]]]))


def test_fixed_reference_across_cadences_and_perturbations(archives):
    field = SimulationSampledField(config_for(archives))
    field.set_coordinate_mapping_reference(np.array([100.0, 200.0, 300.0]))
    time = np.array([0.0, 0.5, 1.0])
    formation = synthetic_configuration(time, velocity_km_s=[1.0, 0.0, 0.0])
    position = formation.position + np.array([100.0, 200.0, 300.0])
    baseline = field.evaluate_quantity("n", time, position)
    altered = position + np.array([0.25, 0.5, 0.75])
    perturbed = field.evaluate_quantity("n", time[:, None] + np.zeros((3, 9)), altered)
    gradient = np.array([1.0, 10.0, 100.0]) * 5 / field.normalization["di_km"]
    np.testing.assert_allclose(perturbed - baseline, np.dot([0.25, 0.5, 0.75], gradient))
    np.testing.assert_allclose(baseline[0] - baseline[0, 0], formation.position[:, 0] @ gradient)
    np.testing.assert_array_equal(
        field.evaluate_quantity("n", time[::2], position[:, ::2]), baseline[::2]
    )


def generation_config(archives):
    return {
        "spacecraft": {"source": "synthetic", "velocity_km_s": [-1.0, 0.0, 0.0]},
        "time": {"t_start_s": 0.0, "t_stop_s": 1.0, "cadence_s": {"n": 0.5, "v": 0.5, "B": 0.25}},
        "field": config_for(archives),
        "measurement": {"velocity_frame": "spacecraft"},
        "errors": {
            "random_seed": 29,
            "n_mc": 1,
            "variables": {q: {"model": "none"} for q in ("n", "v", "B")},
        },
        "output": {"path": str(archives[0][0].parent / "observations.h5")},
        "visualization": {"enabled": False},
    }


@pytest.mark.parametrize("perturb", [False, True])
def test_nine_spacecraft_schema_and_hidden_coordinate_sampling(archives, tmp_path, perturb):
    config = generation_config(archives)
    if perturb:
        config["errors"].update(
            {
                "time": {"model": "gaussian_constant", "sigma_s": 0.1},
                "position": {"model": "gaussian_constant", "sigma_km": [0.1] * 3},
            }
        )
    path = tmp_path / "generate.yaml"
    path.write_text(yaml.safe_dump(config))
    output = generate_from_config(path)
    data = read_observations(output)
    assert data["metadata"]["schema_version"] == "2"
    assert data["spacecraft"]["id"] == ["H", *[f"N{i}" for i in range(1, 9)]]
    for q, count in (("n", 3), ("v", 3), ("B", 5)):
        trailing = () if q == "n" else (3,)
        assert data["truth"][q].shape == (count, 9, *trailing)
        assert data["observed"][q].shape == (1, count, 9, *trailing)
        assert data["spacecraft"]["r"][q].shape == (9, count, 3)
    metadata = json.loads(data["metadata"]["field_metadata"])
    assert metadata["frozen_snapshot"] is True
    assert metadata["simulation_layout"]["iteration"] == 1
    assert metadata["boundary"] == "interior_x_periodic_yz"
    norm = metadata["normalization"]
    displacement = data["errors"]["position_offset"][0] + data["errors"]["time_offset"][
        0, :, None
    ] * [-1.0, 0.0, 0.0]
    expected_shift = displacement @ (np.array([1.0, 10.0, 100.0]) * 5 / norm["di_km"])
    np.testing.assert_allclose(
        data["observed"]["n"][0] - data["truth"]["n"], np.broadcast_to(expected_shift, (3, 9))
    )
    # Same known coordinate gradient for V components, then subtract trajectory velocity.
    vshift = expected_shift * norm["vA_km_s"] / 5
    expected_v = data["truth"]["v"] + vshift[None, :, None] - [-1.0, 0.0, 0.0]
    np.testing.assert_allclose(data["observed"]["v"][0], expected_v)
    with h5py.File(output) as handle:
        assert handle["observed/B"].compression == "gzip"
        assert handle["errors/time_offset"].attrs["units"] == "s"


def test_mc_out_of_bounds_raises_instead_of_wrapping(archives, tmp_path):
    config = generation_config(archives)
    config["errors"]["position"] = {"model": "gaussian_constant", "sigma_km": [1e6, 0, 0]}
    path = tmp_path / "bad-mc.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="wrapping is forbidden"):
        generate_from_config(path)
