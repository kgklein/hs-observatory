"""Configurable sums of plane sine waves."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from hs_observatory.fields.base import FieldModel
from hs_observatory.utils.units import vector3


class SineSumField(FieldModel):
    """Background plasma values plus independent sine-wave perturbations."""

    name = "sine_sum"

    def __init__(self, background: dict[str, Any], modes: Sequence[dict[str, Any]] = ()) -> None:
        self.background_n = float(background.get("n", 0.0))
        self.background_v = vector3(background.get("v", (0.0, 0.0, 0.0)), "background.v")
        self.background_B = vector3(background.get("B", (0.0, 0.0, 0.0)), "background.B")
        self.modes = list(modes)
        for number, mode in enumerate(self.modes):
            quantity = mode.get("quantity")
            if quantity not in {"n", "v", "B"}:
                raise ValueError(f"Mode {number} has unknown quantity {quantity!r}")
            vector3(mode.get("k", ()), f"modes[{number}].k")
            amplitude = np.asarray(mode.get("amplitude"), dtype=float)
            expected = () if quantity == "n" else (3,)
            if amplitude.shape != expected:
                raise ValueError(f"Mode {number} amplitude must have shape {expected}")

    def evaluate_quantity(
        self, quantity: str, time: np.ndarray, position: np.ndarray
    ) -> np.ndarray:
        if quantity not in {"n", "v", "B"}:
            raise ValueError(f"Unknown field quantity: {quantity}")
        time = np.asarray(time, dtype=float)
        position = np.asarray(position, dtype=float)
        if position.ndim != 3 or position.shape[2] != 3:
            raise ValueError("position must have shape (n_spacecraft, n_time, 3)")

        n_sc, n_time, _ = position.shape
        if time.shape == (n_time,):
            time_by_spacecraft = time[:, None]
        elif time.shape == (n_time, n_sc):
            time_by_spacecraft = time
        else:
            raise ValueError(
                "time must have shape (n_time,) or (n_time, n_spacecraft)"
            )
        if quantity == "n":
            values = np.full((n_time, n_sc), self.background_n, dtype=float)
        else:
            background = self.background_v if quantity == "v" else self.background_B
            values = np.broadcast_to(background, (n_time, n_sc, 3)).copy()
        positions_time_first = np.swapaxes(position, 0, 1)  # (n_time, n_sc, 3)

        for mode in self.modes:
            if mode["quantity"] != quantity:
                continue
            k = np.asarray(mode["k"], dtype=float)
            phase = (
                np.einsum("tsd,d->ts", positions_time_first, k)
                - float(mode.get("omega", 0.0)) * time_by_spacecraft
                + float(mode.get("phase", 0.0))
            )
            wave = np.sin(phase)
            amplitude = np.asarray(mode["amplitude"], dtype=float)
            if quantity == "n":
                values += float(amplitude) * wave
            else:
                values += wave[..., None] * amplitude
        return values


def field_from_config(config: dict[str, Any]) -> FieldModel:
    model = str(config.get("model", "")).lower()
    if model == "sine_sum":
        return SineSumField(config.get("background", {}), config.get("modes", []))
    raise ValueError(f"Unknown field model: {model}")
