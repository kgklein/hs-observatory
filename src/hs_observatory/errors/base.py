"""Interfaces for instrument and measurement errors."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ErrorModel(ABC):
    """Transform truth into a Monte Carlo ensemble of observations."""

    name: str

    @abstractmethod
    def apply(
        self,
        truth: np.ndarray,
        rng: np.random.Generator,
        n_mc: int,
        *,
        time: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return observed values and errors, both shaped ``(n_mc, *truth.shape)``."""

    def apply_ensemble(
        self,
        sampled: np.ndarray,
        rng: np.random.Generator,
        *,
        time: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply measurement error to an existing leading-MC ensemble."""

        observed, error = self.apply(sampled, rng, n_mc=1, time=time)
        return observed[0], error[0]
