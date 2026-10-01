"""Initial independent Gaussian measurement-error models."""

from __future__ import annotations

from typing import Any

import numpy as np

from hs_observatory.errors.base import ErrorModel
from hs_observatory.errors.ornstein_uhlenbeck import (
    ScalarOrnsteinUhlenbeckError,
    VectorOrnsteinUhlenbeckError,
)


class NoError(ErrorModel):
    name = "none"

    def apply(self, truth: np.ndarray, rng: np.random.Generator, n_mc: int, *, time=None):
        del time
        del rng
        observed = np.broadcast_to(truth, (n_mc, *truth.shape)).copy()
        return observed, np.zeros_like(observed)


class AbsoluteGaussianError(ErrorModel):
    name = "gaussian_absolute"

    def __init__(self, sigma: float | list[float]) -> None:
        self.sigma = np.asarray(sigma, dtype=float)
        if np.any(self.sigma < 0):
            raise ValueError("Gaussian sigma values must be non-negative")

    def apply(self, truth: np.ndarray, rng: np.random.Generator, n_mc: int, *, time=None):
        del time
        shape = (n_mc, *truth.shape)
        error = rng.normal(size=shape) * self.sigma
        return truth[None, ...] + error, error


class RelativeGaussianError(ErrorModel):
    name = "gaussian_relative"

    def __init__(self, sigma: float) -> None:
        self.sigma = float(sigma)
        if self.sigma < 0:
            raise ValueError("Relative Gaussian sigma must be non-negative")

    def apply(self, truth: np.ndarray, rng: np.random.Generator, n_mc: int, *, time=None):
        del time
        error = rng.normal(size=(n_mc, *truth.shape)) * (self.sigma * np.abs(truth))[None, ...]
        return truth[None, ...] + error, error


class VectorComponentGaussianError(AbsoluteGaussianError):
    name = "gaussian_vector_component"

    def __init__(self, sigma: list[float]) -> None:
        super().__init__(sigma)
        if self.sigma.shape != (3,):
            raise ValueError("Vector-component sigma must contain three values")

    def apply(self, truth: np.ndarray, rng: np.random.Generator, n_mc: int, *, time=None):
        if truth.shape[-1] != 3:
            raise ValueError("Vector-component error requires a final dimension of length three")
        return super().apply(truth, rng, n_mc, time=time)


def error_model_from_config(
    config: dict[str, Any] | None, *, quantity: str | None = None
) -> ErrorModel:
    config = config or {"model": "none"}
    model = str(config.get("model", "none")).lower()
    if model in {"none", "no_error"}:
        return NoError()
    if model in {"gaussian_absolute", "absolute_gaussian"}:
        return AbsoluteGaussianError(config["sigma"])
    if model in {"gaussian_relative", "relative_gaussian"}:
        return RelativeGaussianError(config["sigma"])
    if model in {"gaussian_vector_component", "vector_component_gaussian"}:
        return VectorComponentGaussianError(config["sigma"])
    if model in {"ornstein_uhlenbeck", "ou"}:
        if quantity == "n":
            return ScalarOrnsteinUhlenbeckError(
                sigma=config["sigma"],
                tau_s=config["tau_s"],
            )
        if quantity in {"v", "B"}:
            return VectorOrnsteinUhlenbeckError(
                magnitude_sigma=config["magnitude_sigma"],
                magnitude_tau_s=config["magnitude_tau_s"],
                direction_sigma_deg=config.get("direction_sigma_deg", 0.0),
                direction_tau_s=config.get("direction_tau_s"),
            )
        raise ValueError("OU error configuration requires quantity 'n', 'v', or 'B'")
    raise ValueError(f"Unknown error model: {model}")
