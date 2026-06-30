"""Minimal array coercion helpers; unit libraries can be added later."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def vector3(value: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must contain exactly three components")
    return array
