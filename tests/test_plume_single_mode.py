import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from hs_observatory.fields.plume_single_mode import (
    ELEMENTARY_CHARGE_C,
    PROTON_MASS_KG,
    PlumeSingleModeField,
    complex_from_pair,
    compute_proton_reference_scales,
    get_plume_column_map,
    load_plume_row,
    parse_plume_eigenmode,
)
from hs_observatory.fields.sine_sum import field_from_config
from hs_observatory.generate import generate_from_config


def _fake_row(
    n_components=2,
    *,
    k_perp_rho=0.0,
    k_parallel_rho=0.0,
    omega_ratio=0.0,
    gamma_ratio=-0.1,
):
    columns = get_plume_column_map(n_components)
    row = np.zeros(int(columns["minimum_columns"]))
    row[0] = k_perp_rho
    row[1] = k_parallel_rho
    row[4] = omega_ratio
    row[5] = gamma_ratio

    def set_pair(name, value):
        real_column, imaginary_column = columns[name]
        row[real_column] = np.real(value)
        row[imaginary_column] = np.imag(value)

    set_pair("B_x", 1.0 + 0.0j)
    set_pair("B_y", 0.0 + 1.0j)
    set_pair("B_z", 0.0 + 0.0j)
    for species in range(1, n_components + 1):
        set_pair(f"U_{species}_x", species * (0.5 + 0.0j))
        set_pair(f"U_{species}_y", species * (1.0 + 0.0j))
        set_pair(f"U_{species}_z", species * (0.0 + 0.25j))
        set_pair(f"n_{species}", species * (1.0 - 0.5j))
    return row


def _write_mode(path: Path, rows):
    np.savetxt(path, np.asarray(rows))
    return path


def _field_config(path, *, B_rms_nT=np.sqrt(2.0), phase0=0.0):
    return {
        "model": "plume_single_mode",
        "background": {"n": 0.0, "v": [0.0, 0.0, 0.0], "B": [0.0, 0.0, 0.0]},
        "plume": {
            "path": str(path),
            "n_components": 2,
            "row_index": 0,
            "row_index_base": 0,
            "T_ref_eV": 10.0,
            "B_ref_nT": 5.0,
            "output_species": "ref",
            "phase0": phase0,
            "include_growth": False,
            "geometry": {"k_perp_axis": "x", "k_parallel_axis": "z"},
            "amplitude": {"B_rms_nT": B_rms_nT},
        },
    }


def test_row_loading_supports_zero_and_one_based_indices(tmp_path):
    rows = np.arange(3 * 34, dtype=float).reshape(3, 34)
    path = _write_mode(tmp_path / "tiny.mode1", rows)

    assert np.array_equal(load_plume_row(path, 1, 0), rows[1])
    assert np.array_equal(load_plume_row(path, 2, 1), rows[1])
    with pytest.raises(IndexError, match="outside"):
        load_plume_row(path, 3, 0)
    with pytest.raises(ValueError, match="row_index_base"):
        load_plume_row(path, 1, 2)


def test_column_maps_and_complex_pair_parser_for_supported_component_counts():
    assert get_plume_column_map(2)["minimum_columns"] == 34
    assert get_plume_column_map(3)["minimum_columns"] == 42
    assert get_plume_column_map(4)["minimum_columns"] == 50
    assert get_plume_column_map(4)["n_4"] == (48, 49)
    assert complex_from_pair(np.array([2.5, -3.0]), 0, 1) == 2.5 - 3.0j

    mode = parse_plume_eigenmode(_fake_row(3), 3, output_species=2)
    assert np.array_equal(mode.v_complex, [1.0 + 0.0j, 2.0 + 0.0j, 0.0 + 0.5j])
    assert mode.n_complex == 2.0 - 1.0j


def test_proton_reference_scales_use_angular_gyrofrequency_without_two_pi():
    temperature_eV = 10.0
    magnetic_nT = 5.0
    scales = compute_proton_reference_scales(temperature_eV, magnetic_nT)
    expected_omega = ELEMENTARY_CHARGE_C * magnetic_nT * 1.0e-9 / PROTON_MASS_KG
    expected_vth = np.sqrt(2.0 * temperature_eV * ELEMENTARY_CHARGE_C / PROTON_MASS_KG)

    assert scales.omega_ref_rad_s == pytest.approx(expected_omega)
    assert scales.vth_ref_m_s == pytest.approx(expected_vth)
    assert scales.rho_ref_m == pytest.approx(expected_vth / expected_omega)
    assert scales.omega_ref_rad_s != pytest.approx(expected_omega / (2.0 * np.pi))


def test_field_shapes_real_plane_wave_and_relative_phase(tmp_path):
    scales = compute_proton_reference_scales(10.0, 5.0)
    desired_k = 0.4
    desired_omega = 0.7
    row = _fake_row(
        k_perp_rho=desired_k * scales.rho_ref_km,
        omega_ratio=desired_omega / scales.omega_ref_rad_s,
    )
    path = _write_mode(tmp_path / "phase.mode1", [row])
    phase0 = 0.3
    field = PlumeSingleModeField(_field_config(path, phase0=phase0))
    time = np.array([0.0, 0.4, 0.8])
    position = np.zeros((2, 3, 3))
    position[0, :, 0] = [0.0, 0.5, 1.0]
    position[1, :, 0] = [1.5, 2.0, 2.5]

    values = field.evaluate(time, position)
    phase = desired_k * position[:, :, 0].T - desired_omega * time[:, None] + phase0
    assert values["n"].shape == (3, 2)
    assert values["v"].shape == (3, 2, 3)
    assert values["B"].shape == (3, 2, 3)
    np.testing.assert_allclose(values["B"][..., 0], np.cos(phase), atol=1.0e-13)
    np.testing.assert_allclose(values["B"][..., 1], -np.sin(phase), atol=1.0e-13)
    np.testing.assert_allclose(values["v"][..., 1], np.cos(phase), atol=1.0e-13)
    np.testing.assert_allclose(values["n"], np.cos(phase) + 0.5 * np.sin(phase), atol=1.0e-13)


def test_magnetic_normalization_applies_one_scale_to_all_eigenfunctions(tmp_path):
    row = _fake_row()
    columns = get_plume_column_map(2)
    for name, value in (("B_x", 3.0 + 4.0j), ("B_y", 0.0j), ("B_z", 12.0j)):
        real_column, imaginary_column = columns[name]
        row[real_column] = value.real
        row[imaginary_column] = value.imag
    path = _write_mode(tmp_path / "normalization.mode1", [row])
    field = PlumeSingleModeField(_field_config(path, B_rms_nT=0.26))

    assert np.linalg.norm(field.B_complex_scaled) == pytest.approx(0.26)
    assert field.scale == pytest.approx(0.02)  # sqrt(3^2 + 4^2 + 12^2) = 13
    np.testing.assert_allclose(field.v_complex_scaled, field.scale * field.mode.v_complex)
    assert field.n_complex_scaled == pytest.approx(field.scale * field.mode.n_complex)


def test_field_constructs_by_name_and_generation_writes_mode_metadata(tmp_path):
    path = _write_mode(tmp_path / "workflow.mode1", [_fake_row()])
    field_config = _field_config(path, B_rms_nT=0.1)
    assert isinstance(field_from_config(field_config), PlumeSingleModeField)
    output = tmp_path / "plume.h5"
    config = tmp_path / "plume.yaml"
    field_yaml = yaml.safe_dump({"field": field_config}, sort_keys=False)
    config.write_text(
        f"""
spacecraft:
  source: synthetic
time:
  t_start_s: 0
  t_stop_s: 2
  cadence_s: 1
{field_yaml}errors:
  random_seed: 7
  n_mc: 1
  variables:
    n: {{model: none}}
    v: {{model: none}}
    B: {{model: none}}
output:
  path: {output}
visualization:
  enabled: false
""",
        encoding="utf-8",
    )

    assert generate_from_config(config) == output
    with h5py.File(output) as handle:
        assert handle["truth/n"].shape == (3, 9)
        assert handle["truth/v"].shape == (3, 9, 3)
        assert handle["truth/B"].shape == (3, 9, 3)
        assert handle["observed/B"].shape == (1, 3, 9, 3)
        assert handle["metadata"].attrs["field_model"] == "plume_single_mode"
        metadata = json.loads(handle["metadata"].attrs["field_metadata"])
        assert metadata["field_model"] == "plume_single_mode"
        assert metadata["n_components"] == 2
        assert metadata["row_index"] == 0
        assert metadata["gamma_rad_s"] < 0.0
        assert metadata["T_ref_eV"] == 10.0
        assert metadata["B_ref_nT"] == 5.0
