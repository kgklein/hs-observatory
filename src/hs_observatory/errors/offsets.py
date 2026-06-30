"""Constant-per-realization coordinate uncertainty models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class CoordinateOffsetModel(ABC):
    """Draw offsets that remain fixed over an entire spacecraft time series."""

    name: str

    @abstractmethod
    def draw(
        self, rng: np.random.Generator, n_mc: int, n_spacecraft: int
    ) -> np.ndarray:
        """Draw an offset for each Monte Carlo realization and spacecraft."""


class NoTimeOffset(CoordinateOffsetModel):
    name = "none"

    def draw(self, rng: np.random.Generator, n_mc: int, n_spacecraft: int) -> np.ndarray:
        del rng
        return np.zeros((n_mc, n_spacecraft), dtype=float)


class GaussianTimeOffset(CoordinateOffsetModel):
    """One normally distributed clock offset per realization and spacecraft."""

    name = "gaussian_constant_time_offset"

    def __init__(self, sigma_s: float) -> None:
        self.sigma_s = float(sigma_s)
        if self.sigma_s < 0:
            raise ValueError("Time-offset sigma_s must be non-negative")

    def draw(self, rng: np.random.Generator, n_mc: int, n_spacecraft: int) -> np.ndarray:
        return rng.normal(scale=self.sigma_s, size=(n_mc, n_spacecraft))


class NoPositionOffset(CoordinateOffsetModel):
    name = "none"

    def draw(self, rng: np.random.Generator, n_mc: int, n_spacecraft: int) -> np.ndarray:
        del rng
        return np.zeros((n_mc, n_spacecraft, 3), dtype=float)


class GaussianPositionOffset(CoordinateOffsetModel):
    """One normally distributed XYZ offset per realization and spacecraft."""

    name = "gaussian_constant_position_offset"

    def __init__(self, sigma_km: float | list[float]) -> None:
        self.sigma_km = np.asarray(sigma_km, dtype=float)
        if self.sigma_km.shape not in {(), (3,)}:
            raise ValueError("Position-offset sigma_km must be a scalar or three components")
        if np.any(self.sigma_km < 0):
            raise ValueError("Position-offset sigma_km values must be non-negative")

    def draw(self, rng: np.random.Generator, n_mc: int, n_spacecraft: int) -> np.ndarray:
        return rng.normal(size=(n_mc, n_spacecraft, 3)) * self.sigma_km


def time_offset_model_from_config(config: dict[str, Any] | None) -> CoordinateOffsetModel:
    config = config or {"model": "none"}
    model = str(config.get("model", "none")).lower()
    if model in {"none", "no_error"}:
        return NoTimeOffset()
    if model in {"gaussian_constant", "gaussian_time_offset"}:
        return GaussianTimeOffset(config["sigma_s"])
    raise ValueError(f"Unknown time-offset model: {model}")


def position_offset_model_from_config(config: dict[str, Any] | None) -> CoordinateOffsetModel:
    config = config or {"model": "none"}
    model = str(config.get("model", "none")).lower()
    if model in {"none", "no_error"}:
        return NoPositionOffset()
    if model in {"gaussian_constant", "gaussian_position_offset"}:
        return GaussianPositionOffset(config["sigma_km"])
    raise ValueError(f"Unknown position-offset model: {model}")
