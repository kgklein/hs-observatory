"""Constant-per-realization coordinate uncertainty models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from hs_observatory.errors.observatory import (
    component_sigma_from_3d_radial_requirement,
    component_sigma_from_3d_radial_3sigma,
    sigma_from_requirement,
)


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
    if not bool(config.get("enabled", True)):
        return NoTimeOffset()
    if model in {"none", "no_error"}:
        return NoTimeOffset()
    if model in {"gaussian_constant", "gaussian_time_offset"}:
        return GaussianTimeOffset(config["sigma_s"])
    if model in {"constant_per_spacecraft", "observatory_timing"}:
        if (
            "requirement_ms" in config
            and "requirement_3sigma_ms" in config
        ):
            raise ValueError(
                "timing cannot specify both requirement_ms and "
                "requirement_3sigma_ms"
            )
        legacy = "requirement_3sigma_ms" in config
        if legacy and config.get("sigma_level", 3) != 3:
            raise ValueError(
                "timing.requirement_3sigma_ms implies sigma_level 3"
            )
        requirement_ms = float(
            config.get(
                "requirement_ms",
                config.get("requirement_3sigma_ms", 12.5),
            )
        )
        return GaussianTimeOffset(
            sigma_from_requirement(
                requirement_ms, 3 if legacy else config.get("sigma_level", 3)
            )
            * 1.0e-3
        )
    raise ValueError(f"Unknown time-offset model: {model}")


def position_offset_model_from_config(config: dict[str, Any] | None) -> CoordinateOffsetModel:
    config = config or {"model": "none"}
    model = str(config.get("model", "none")).lower()
    if not bool(config.get("enabled", True)):
        return NoPositionOffset()
    if model in {"none", "no_error"}:
        return NoPositionOffset()
    if model in {"gaussian_constant", "gaussian_position_offset"}:
        return GaussianPositionOffset(config["sigma_km"])
    if model in {
        "isotropic_3d_gaussian_position_error",
        "observatory_position",
    }:
        if (
            "requirement_magnitude_km" in config
            and "requirement_3sigma_magnitude_km" in config
        ):
            raise ValueError(
                "position cannot specify both requirement_magnitude_km and "
                "requirement_3sigma_magnitude_km"
            )
        legacy = "requirement_3sigma_magnitude_km" in config
        if legacy and config.get("sigma_level", 3) != 3:
            raise ValueError(
                "position.requirement_3sigma_magnitude_km implies sigma_level 3"
            )
        radius = float(
            config.get(
                "requirement_magnitude_km",
                config.get("requirement_3sigma_magnitude_km", 2.5),
            )
        )
        if "radial_probability" in config:
            probability = float(config["radial_probability"])
            component_sigma = component_sigma_from_3d_radial_3sigma(
                radius, probability
            )
        else:
            component_sigma = component_sigma_from_3d_radial_requirement(
                radius, 3 if legacy else config.get("sigma_level", 3)
            )
        return GaussianPositionOffset(
            component_sigma
        )
    raise ValueError(f"Unknown position-offset model: {model}")
