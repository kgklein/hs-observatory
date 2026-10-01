import numpy as np
import pytest

from hs_observatory.fields.shock import ShockField


def shock_config(*, magnetic_field=(0.0, 5.0, 0.0), **overrides):
    shock = {
        "normal_gse": [2.0, 0.0, 0.0],
        "velocity_gse_km_s": [-150.0, 0.0, 0.0],
        "reference_time": "start",
        "distance_from_mesocenter_km": 3000.0,
        "thickness_km": 500.0,
        "upstream": {
            "density_cm3": 5.0,
            "velocity_km_s": [450.0, 0.0, 0.0],
            "magnetic_field_nt": list(magnetic_field),
            "total_pressure_npa": 0.02,
        },
        "gamma": 5.0 / 3.0,
    }
    shock.update(overrides)
    return {"model": "shock", "shock": shock}


def initialized_field(**kwargs):
    field = ShockField(shock_config(**kwargs))
    time = np.array([0.0, 20.0, 40.0])
    offsets = np.array([[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    position = offsets[:, None, :] + time[None, :, None] * np.array([1.0, 0.0, 0.25])
    field.set_nominal_configuration(time, position)
    return field


def test_normalization_units_and_perpendicular_downstream_state():
    field = initialized_field()

    np.testing.assert_allclose(field.normal, [1.0, 0.0, 0.0])
    assert field.compression_ratio == pytest.approx(3.7658529994)
    assert field.downstream_n == pytest.approx(field.upstream_n * field.compression_ratio)
    assert field.downstream_pressure_npa > field.upstream_pressure_npa
    assert field.fast_mach_upstream > 1.0
    assert field.fast_mach_downstream < 1.0
    np.testing.assert_allclose(field.downstream_B, [0.0, 18.829264997, 0.0])
    np.testing.assert_allclose(field.downstream_v, [9.326452756, 0.0, 0.0])
    assert max(field.jump_residuals.values()) < 1.0e-10


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"normal_gse": [0.0, 0.0, 0.0]}, "nonzero"),
        ({"velocity_gse_km_s": [-150.0, 1.0, 0.0]}, "tangential"),
        ({"thickness_km": 0.0}, "positive"),
        ({"gamma": 1.0}, "greater than one"),
    ],
)
def test_invalid_shock_configuration_is_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        ShockField(shock_config(**overrides))


def test_upstream_normal_flow_must_enter_shock():
    with pytest.raises(ValueError, match="normal flow"):
        ShockField(
            shock_config(
                velocity_gse_km_s=[150.0, 0.0, 0.0],
                upstream={
                    "density_cm3": 5.0,
                    "velocity_km_s": [100.0, 0.0, 0.0],
                    "magnetic_field_nt": [0.0, 5.0, 0.0],
                    "total_pressure_npa": 0.02,
                },
            )
        )


def test_fixed_mesocenter_offset_propagation_and_crossing_time():
    field = initialized_field()
    np.testing.assert_allclose(field.mesocenter_km, [0.0, 0.0, 0.0])

    center_path = np.zeros((1, 3, 3))
    center_path[0, :, :] = np.array([[0.0, 0.0, 0.0], [3000.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    times = np.array([0.0, 10.0, 20.0])
    xi = field.signed_distance(times, center_path)[:, 0]
    np.testing.assert_allclose(xi, [-3000.0, 1500.0, 0.0])

    # A nominal mesocenter translating at +1 km/s meets a -150 km/s shock.
    crossing_time = 3000.0 / 151.0
    crossing_position = np.array([[[crossing_time, 0.0, 0.25 * crossing_time]]])
    assert field.signed_distance(np.array([crossing_time]), crossing_position)[
        0, 0
    ] == pytest.approx(0.0)

    original_center = field.mesocenter_km.copy()
    positions = np.full((1, 1, 3), 1.0e6)
    field.evaluate_quantity("n", np.array([1.0e6]), positions)
    np.testing.assert_array_equal(field.mesocenter_km, original_center)


def test_endpoints_midpoint_and_ten_to_ninety_width():
    field = initialized_field()
    xi = np.array([-1.0e9, -250.0, 0.0, 250.0, 1.0e9])
    fraction = field.transition_fraction(xi)

    np.testing.assert_allclose(fraction[[0, 2, 4]], [0.0, 0.5, 1.0], atol=1.0e-14)
    np.testing.assert_allclose(fraction[[1, 3]], [0.1, 0.9], atol=1.0e-10)
    assert xi[3] - xi[1] == field.thickness_km

    position = np.zeros((1, len(xi), 3))
    position[0, :, 0] = xi + field.distance_km
    density = field.evaluate_quantity("n", np.zeros(len(xi)), position)[:, 0]
    assert density[0] == pytest.approx(field.upstream_n)
    assert density[-1] == pytest.approx(field.downstream_n)
    assert density[2] == pytest.approx(0.5 * (field.upstream_n + field.downstream_n))


def test_oblique_rankine_hugoniot_solution_and_analytic_divergence():
    field = initialized_field(magnetic_field=(2.0, 5.0, 1.0))

    assert field.compression_ratio > 1.0
    assert max(field.jump_residuals.values()) < 1.0e-10
    assert np.dot(field.downstream_B, field.normal) == pytest.approx(
        np.dot(field.upstream_B, field.normal), abs=1.0e-13
    )
    assert np.dot(field.downstream_B - field.upstream_B, field.normal) == pytest.approx(
        0.0, abs=1.0e-13
    )
    np.testing.assert_allclose(
        field.analytic_magnetic_divergence(np.array([-100.0, 0.0, 100.0])),
        0.0,
        atol=1.0e-15,
    )


def test_numeric_reference_time_uses_interpolated_nominal_mesocenter():
    field = ShockField(shock_config(reference_time=5.0))
    time = np.array([0.0, 10.0])
    position = np.array(
        [
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [12.0, 0.0, 0.0]],
        ]
    )
    field.set_nominal_configuration(time, position)

    assert field.reference_time_s == 5.0
    np.testing.assert_allclose(field.mesocenter_km, [6.0, 0.0, 0.0])
