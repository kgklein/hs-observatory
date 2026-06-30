"""Spacecraft trajectory data structures and test fallback geometry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass
class SpacecraftConfiguration:
    """Positions of a spacecraft formation over time.

    ``position`` always has shape ``(n_spacecraft, n_time, 3)``.
    """

    spacecraft_ids: Sequence[str]
    time: np.ndarray
    position: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.spacecraft_ids = tuple(str(value) for value in self.spacecraft_ids)
        self.time = np.asarray(self.time, dtype=float)
        self.position = np.asarray(self.position, dtype=float)
        expected = (len(self.spacecraft_ids), len(self.time), 3)
        if self.time.ndim != 1:
            raise ValueError("time must be one-dimensional")
        if self.position.shape != expected:
            raise ValueError(f"position must have shape {expected}; got {self.position.shape}")
        if len(set(self.spacecraft_ids)) != len(self.spacecraft_ids):
            raise ValueError("spacecraft_ids must be unique")
        if not np.all(np.isfinite(self.position)):
            raise ValueError("position contains non-finite values")


def synthetic_configuration(
    time: np.ndarray,
    *,
    velocity_km_s: Sequence[float] = (0.0, 0.0, 0.0),
) -> SpacecraftConfiguration:
    """Return a deterministic nine-spacecraft formation for tests and examples."""

    time = np.asarray(time, dtype=float)
    ids = ("H", "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8")
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [25.0, 0.0, 0.0],
            [-20.0, 8.0, 3.0],
            [5.0, 35.0, -4.0],
            [2.0, -30.0, 10.0],
            [12.0, 8.0, 45.0],
            [-8.0, -5.0, -40.0],
            [55.0, 20.0, 15.0],
            [-50.0, -18.0, -12.0],
        ]
    )
    velocity = np.asarray(velocity_km_s, dtype=float)
    if velocity.shape != (3,):
        raise ValueError("velocity_km_s must have three components")
    position = offsets[:, None, :] + time[None, :, None] * velocity[None, None, :]
    return SpacecraftConfiguration(
        ids,
        time,
        position,
        metadata={"source": "synthetic_fallback", "position_units": "km"},
    )
