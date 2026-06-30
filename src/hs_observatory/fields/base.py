"""Interfaces for underlying plasma fields."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class FieldModel(ABC):
    """A field sampled at spacecraft positions and times."""

    name: str

    @abstractmethod
    def evaluate_quantity(
        self, quantity: str, time: np.ndarray, position: np.ndarray
    ) -> np.ndarray:
        """Evaluate one quantity on ``position[n_sc, n_time, 3]``.

        ``time`` may be shared with shape ``(n_time,)`` or spacecraft-specific
        with shape ``(n_time, n_spacecraft)``.
        """

    def evaluate(self, time: np.ndarray, position: np.ndarray) -> dict[str, np.ndarray]:
        """Evaluate all quantities on one shared grid.

        This convenience method remains useful for tests and field models with
        a common grid. Generation uses :meth:`evaluate_quantity` so each
        instrument can retain its native cadence.
        """

        return {
            quantity: self.evaluate_quantity(quantity, time, position)
            for quantity in ("n", "v", "B")
        }
