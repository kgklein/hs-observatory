"""Cadence-aware Ornstein-Uhlenbeck measurement errors."""

from __future__ import annotations

import numpy as np

from hs_observatory.errors.base import ErrorModel


def _validated_time(time: np.ndarray | None, n_time: int) -> np.ndarray:
    if time is None:
        raise ValueError("Ornstein-Uhlenbeck errors require a time axis")
    values = np.asarray(time, dtype=float)
    if values.shape != (n_time,):
        raise ValueError(f"time must have shape ({n_time},); got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("OU time values must be finite")
    if n_time > 1 and np.any(np.diff(values) <= 0):
        raise ValueError("OU time values must be strictly increasing")
    return values


def _stationary_ou(
    time: np.ndarray,
    *,
    sigma: float,
    tau_s: float,
    rng: np.random.Generator,
    sample_shape: tuple[int, ...],
) -> np.ndarray:
    """Draw a zero-mean OU process using its exact discrete transition."""

    values = np.empty((sample_shape[0], len(time), *sample_shape[1:]), dtype=float)
    if len(time) == 0:
        return values
    values[:, 0, ...] = rng.normal(scale=sigma, size=sample_shape)
    for index, dt in enumerate(np.diff(time), start=1):
        alpha = float(np.exp(-dt / tau_s))
        innovation_sigma = sigma * np.sqrt(1.0 - alpha * alpha)
        values[:, index, ...] = (
            alpha * values[:, index - 1, ...]
            + rng.normal(scale=innovation_sigma, size=sample_shape)
        )
    return values


class ScalarOrnsteinUhlenbeckError(ErrorModel):
    """Absolute additive OU error for a scalar measurement."""

    name = "ornstein_uhlenbeck_scalar"

    def __init__(self, *, sigma: float, tau_s: float) -> None:
        self.sigma = float(sigma)
        self.tau_s = float(tau_s)
        if self.sigma < 0:
            raise ValueError("OU sigma must be non-negative")
        if self.tau_s <= 0:
            raise ValueError("OU tau_s must be positive")

    def _apply_sampled(
        self, sampled: np.ndarray, rng: np.random.Generator, time: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        if sampled.ndim != 3:
            raise ValueError("Scalar OU input must have shape (n_mc, n_time, n_spacecraft)")
        time_values = _validated_time(time, sampled.shape[1])
        error = _stationary_ou(
            time_values,
            sigma=self.sigma,
            tau_s=self.tau_s,
            rng=rng,
            sample_shape=(sampled.shape[0], sampled.shape[2]),
        )
        return sampled + error, error

    def apply(self, truth, rng, n_mc, *, time=None):
        truth = np.asarray(truth, dtype=float)
        if truth.ndim != 2:
            raise ValueError("Scalar OU truth must have shape (n_time, n_spacecraft)")
        sampled = np.broadcast_to(truth, (n_mc, *truth.shape)).copy()
        return self._apply_sampled(sampled, rng, time)

    def apply_ensemble(self, sampled, rng, *, time=None):
        return self._apply_sampled(np.asarray(sampled, dtype=float), rng, time)


class VectorOrnsteinUhlenbeckError(ErrorModel):
    """Relative-magnitude and isotropic directional OU errors for vectors."""

    name = "ornstein_uhlenbeck_vector"

    def __init__(
        self,
        *,
        magnitude_sigma: float,
        magnitude_tau_s: float,
        direction_sigma_deg: float = 0.0,
        direction_tau_s: float | None = None,
    ) -> None:
        self.magnitude_sigma = float(magnitude_sigma)
        self.magnitude_tau_s = float(magnitude_tau_s)
        self.direction_sigma_deg = float(direction_sigma_deg)
        self.direction_tau_s = (
            self.magnitude_tau_s if direction_tau_s is None else float(direction_tau_s)
        )
        if self.magnitude_sigma < 0:
            raise ValueError("OU magnitude_sigma must be non-negative")
        if self.magnitude_tau_s <= 0:
            raise ValueError("OU magnitude_tau_s must be positive")
        if self.direction_sigma_deg < 0:
            raise ValueError("OU direction_sigma_deg must be non-negative")
        if self.direction_tau_s <= 0:
            raise ValueError("OU direction_tau_s must be positive")

    def _apply_sampled(
        self, sampled: np.ndarray, rng: np.random.Generator, time: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        if sampled.ndim != 4 or sampled.shape[-1] != 3:
            raise ValueError(
                "Vector OU input must have shape (n_mc, n_time, n_spacecraft, 3)"
            )
        time_values = _validated_time(time, sampled.shape[1])
        process_shape = (sampled.shape[0], sampled.shape[2])
        magnitude_error = _stationary_ou(
            time_values,
            sigma=self.magnitude_sigma,
            tau_s=self.magnitude_tau_s,
            rng=rng,
            sample_shape=process_shape,
        )

        # Three Cartesian latent components projected onto the local tangent
        # plane are equivalent to two isotropic tangent components. Giving each
        # tangent component sigma_total/sqrt(2) makes direction_sigma_deg the
        # RMS total angular displacement.
        direction_component_sigma = np.deg2rad(self.direction_sigma_deg) / np.sqrt(2.0)
        angular_latent = _stationary_ou(
            time_values,
            sigma=direction_component_sigma,
            tau_s=self.direction_tau_s,
            rng=rng,
            sample_shape=(*process_shape, 3),
        )

        magnitude = np.linalg.norm(sampled, axis=-1)
        nonzero = magnitude > 0.0
        unit = np.divide(
            sampled,
            magnitude[..., None],
            out=np.zeros_like(sampled),
            where=nonzero[..., None],
        )
        tangent = angular_latent - np.sum(angular_latent * unit, axis=-1)[..., None] * unit
        angle = np.linalg.norm(tangent, axis=-1)
        tangent_scale = np.divide(
            np.sin(angle),
            angle,
            out=np.ones_like(angle),
            where=angle > 0.0,
        )
        rotated_unit = np.cos(angle)[..., None] * unit + tangent_scale[..., None] * tangent
        rotated_unit[~nonzero] = 0.0

        magnitude_scale = np.maximum(0.0, 1.0 + magnitude_error)
        observed = rotated_unit * (magnitude * magnitude_scale)[..., None]
        error = observed - sampled
        return observed, error

    def apply(self, truth, rng, n_mc, *, time=None):
        truth = np.asarray(truth, dtype=float)
        if truth.ndim != 3 or truth.shape[-1] != 3:
            raise ValueError("Vector OU truth must have shape (n_time, n_spacecraft, 3)")
        sampled = np.broadcast_to(truth, (n_mc, *truth.shape)).copy()
        return self._apply_sampled(sampled, rng, time)

    def apply_ensemble(self, sampled, rng, *, time=None):
        return self._apply_sampled(np.asarray(sampled, dtype=float), rng, time)
