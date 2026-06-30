import numpy as np

from hs_observatory.fields.sine_sum import SineSumField


def test_sine_sum_returns_expected_shapes_and_backgrounds():
    time = np.arange(5.0)
    position = np.zeros((9, len(time), 3))
    field = SineSumField(
        {"n": 5.0, "v": [400.0, 0.0, 0.0], "B": [5.0, 0.0, 0.0]},
        [
            {"quantity": "n", "amplitude": 1.0, "k": [0, 0, 0], "omega": 1, "phase": 0},
            {
                "quantity": "B",
                "amplitude": [0, 1, 0],
                "k": [0, 0, 0],
                "omega": 1,
                "phase": 0,
            },
        ],
    )
    values = field.evaluate(time, position)

    assert values["n"].shape == (5, 9)
    assert values["v"].shape == (5, 9, 3)
    assert values["B"].shape == (5, 9, 3)
    assert np.allclose(values["v"], [400.0, 0.0, 0.0])
    assert not np.allclose(values["n"], 5.0)
