"""Fast point-wise isotropic intermittent turbulence inspired by Malara et al.

This module evaluates a time-independent, periodic, solenoidal vector field at
arbitrary points. It constructs only the eddies whose compact supports contain
each requested point; no spatial grid or full eddy hierarchy is stored. See
Phys. Rev. E 94, 053109 (2016), https://doi.org/10.1103/PhysRevE.94.053109.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
import hashlib
from itertools import product
from typing import Any, Sequence

import numpy as np

from hs_observatory.fields.base import FieldModel
from hs_observatory.utils.units import vector3

DEFAULT_EPSILON_EXPONENT = 1.0 / 3.0


@dataclass(frozen=True)
class MalaraConfig:
    """Configuration for the isotropic Malara-style turbulence evaluator.

    ``n_scales`` is the largest zero-based scale index. The evaluator includes
    every scale from ``injection_scale_index`` through ``n_scales`` inclusive.
    Paper cell indices are commonly written one-based; internal indices here
    run from 0 through ``2**m - 1`` along each axis at scale ``m``.
    """

    domain_size: tuple[float, float, float] = (1.0, 1.0, 1.0)
    n_scales: int = 6
    injection_scale_index: int = 2
    spectral_exponent_h: float = 1.0 / 3.0
    intermittency_p: float = 0.7
    amplitude: float = 1.0
    base_seed: int = 12345
    realization: int = 1
    epsilon_exponent: float = DEFAULT_EPSILON_EXPONENT

    def __post_init__(self) -> None:
        domain = tuple(float(value) for value in self.domain_size)
        if len(domain) != 3 or any(not np.isfinite(value) or value <= 0 for value in domain):
            raise ValueError("domain_size must contain three positive finite lengths")
        object.__setattr__(self, "domain_size", domain)
        if self.n_scales < 0:
            raise ValueError("n_scales must be non-negative")
        if not 0 <= self.injection_scale_index <= self.n_scales:
            raise ValueError("injection_scale_index must lie between 0 and n_scales")
        if not np.isfinite(self.spectral_exponent_h):
            raise ValueError("spectral_exponent_h must be finite")
        if not 0.5 <= self.intermittency_p <= 1.0:
            raise ValueError("intermittency_p must lie in [0.5, 1.0]")
        if not np.isfinite(self.amplitude):
            raise ValueError("amplitude must be finite")
        if self.base_seed < 0 or self.realization < 0:
            raise ValueError("base_seed and realization must be non-negative")
        if not np.isfinite(self.epsilon_exponent) or self.epsilon_exponent <= 0:
            raise ValueError("epsilon_exponent must be positive and finite")


@dataclass(frozen=True)
class _EddyParameters:
    gamma: np.ndarray
    sign: float
    q_hat: np.ndarray
    high_octants: frozenset[int]


@dataclass(frozen=True)
class _CachedEddyData:
    """Identity-dependent values reused by the cached evaluation backend."""

    parameters: _EddyParameters
    epsilon: float
    amplitude: float


def _evaluate_malara_chunk(
    task: tuple[MalaraConfig, np.ndarray, bool],
) -> np.ndarray:
    """Evaluate one process-safe chunk with a worker-local evaluator/cache."""

    config, positions, use_cache = task
    evaluator = MalaraTurbulence(config)
    if use_cache:
        return evaluator.evaluate_vector_cached(positions)
    return evaluator.evaluate_vector(positions)


def F(t: np.ndarray | float) -> np.ndarray:
    """Compact polynomial F(t), zero outside ``[-1/2, 1/2]``."""

    values = np.asarray(t, dtype=float)
    polynomial = (
        256.0 * values**8
        - 256.0 * values**6
        + 96.0 * values**4
        - 16.0 * values**2
        + 1.0
    )
    return np.where(np.abs(values) <= 0.5, polynomial, 0.0)


def dF_dt(t: np.ndarray | float) -> np.ndarray:
    """Analytic derivative dF/dt, zero outside ``[-1/2, 1/2]``."""

    values = np.asarray(t, dtype=float)
    derivative = (
        2048.0 * values**7
        - 1536.0 * values**5
        + 384.0 * values**3
        - 32.0 * values
    )
    return np.where(np.abs(values) <= 0.5, derivative, 0.0)


# Descriptive aliases are convenient at call sites; F and dF_dt preserve the
# notation used in the model definition and paper.
compact_basis = F
compact_basis_derivative = dF_dt


class MalaraTurbulence:
    """Evaluate a deterministic isotropic Malara-style vector field."""

    def __init__(self, config: MalaraConfig | None = None) -> None:
        self.config = config or MalaraConfig()
        self._domain = np.asarray(self.config.domain_size, dtype=float)
        self._parameter_cache: dict[
            tuple[int, int, int, int], _EddyParameters
        ] = {}
        self._eddy_cache: dict[tuple[int, int, int, int], _CachedEddyData] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _stable_seed(self, scale: int, i: int, j: int, k: int) -> int:
        identity = (
            f"{self.config.base_seed}|{self.config.realization}|"
            f"{scale}|{i}|{j}|{k}"
        ).encode("ascii")
        digest = hashlib.blake2b(identity, digest_size=8, person=b"hsobsMT").digest()
        return int.from_bytes(digest, byteorder="little", signed=False)

    def _eddy_parameters(self, scale: int, i: int, j: int, k: int) -> _EddyParameters:
        """Regenerate one eddy's parameters from its stable cell identity."""

        rng = np.random.default_rng(self._stable_seed(scale, i, j, k))
        gamma = rng.uniform(-1.0, 1.0, size=3)
        sign = float(rng.choice((-1.0, 1.0)))
        q_hat = rng.normal(size=3)
        q_norm = float(np.linalg.norm(q_hat))
        while q_norm == 0.0:  # Defensively handle an effectively impossible draw.
            q_hat = rng.normal(size=3)
            q_norm = float(np.linalg.norm(q_hat))
        q_hat = q_hat / q_norm
        high_octants = frozenset(int(value) for value in rng.choice(8, size=4, replace=False))
        return _EddyParameters(gamma, sign, q_hat, high_octants)

    def energy_transfer_factor(
        self, scale: int, cell_index: Sequence[int]
    ) -> float:
        """Walk the deterministic p-model heritage from the root to one cell."""

        if scale < 0:
            raise ValueError("scale must be non-negative")
        if len(cell_index) != 3:
            raise ValueError("cell_index must contain i, j, and k")
        target = tuple(int(value) for value in cell_index)
        n_cells = 2**scale
        if any(value < 0 or value >= n_cells for value in target):
            raise ValueError(f"cell indices at scale {scale} must lie in [0, {n_cells})")

        epsilon = 1.0
        p = self.config.intermittency_p
        for child_scale in range(1, scale + 1):
            shift_to_parent = scale - child_scale + 1
            parent = tuple(value >> shift_to_parent for value in target)
            bit_shift = scale - child_scale
            child_bits = tuple((value >> bit_shift) & 1 for value in target)
            octant = child_bits[0] + 2 * child_bits[1] + 4 * child_bits[2]
            heritage = self._eddy_parameters(child_scale - 1, *parent).high_octants
            epsilon *= 2.0 * p if octant in heritage else 2.0 * (1.0 - p)
        return epsilon

    def _cached_parameters(
        self, identity: tuple[int, int, int, int]
    ) -> _EddyParameters:
        """Return lazily cached random parameters for one eddy identity."""

        parameters = self._parameter_cache.get(identity)
        if parameters is None:
            parameters = self._eddy_parameters(*identity)
            self._parameter_cache[identity] = parameters
        return parameters

    def _cached_energy_transfer_factor(
        self, scale: int, cell_index: tuple[int, int, int]
    ) -> float:
        """Calculate p-model transfer while caching parent heritage patterns."""

        epsilon = 1.0
        p = self.config.intermittency_p
        for child_scale in range(1, scale + 1):
            shift_to_parent = scale - child_scale + 1
            parent = tuple(value >> shift_to_parent for value in cell_index)
            bit_shift = scale - child_scale
            child_bits = tuple((value >> bit_shift) & 1 for value in cell_index)
            octant = child_bits[0] + 2 * child_bits[1] + 4 * child_bits[2]
            identity = (child_scale - 1, *parent)
            heritage = self._cached_parameters(identity).high_octants
            epsilon *= 2.0 * p if octant in heritage else 2.0 * (1.0 - p)
        return epsilon

    def _cached_eddy_data(
        self, scale: int, cell_index: tuple[int, int, int]
    ) -> _CachedEddyData:
        """Return all deterministic data needed for one local eddy."""

        identity = (scale, *cell_index)
        cached = self._eddy_cache.get(identity)
        if cached is not None:
            self._cache_hits += 1
            return cached

        self._cache_misses += 1
        parameters = self._cached_parameters(identity)
        epsilon = self._cached_energy_transfer_factor(scale, cell_index)
        scale_ratio = 2.0 ** (-scale)
        amplitude = (
            parameters.sign
            * self.config.amplitude
            * epsilon ** self.config.epsilon_exponent
            * scale_ratio ** self.config.spectral_exponent_h
        )
        cached = _CachedEddyData(parameters, epsilon, amplitude)
        self._eddy_cache[identity] = cached
        return cached

    def clear_cache(self) -> None:
        """Remove cached eddy data and reset hit/miss counters."""

        self._parameter_cache.clear()
        self._eddy_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def cache_info(self) -> dict[str, int | float | bool]:
        """Return lazy-cache size and touched-eddy hit statistics."""

        requests = self._cache_hits + self._cache_misses
        return {
            "enabled": True,
            "n_entries": len(self._eddy_cache),
            "n_parameter_entries": len(self._parameter_cache),
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": self._cache_hits / requests if requests else 0.0,
        }

    @staticmethod
    def _nearby_indices(coordinate: float, cell_size: float, n_cells: int) -> tuple[int, ...]:
        containing = int(np.floor(coordinate / cell_size)) % n_cells
        center = (containing + 0.5) * cell_size
        adjacent = containing - 1 if coordinate < center else containing + 1
        # dict preserves order while removing the duplicate that occurs for m=0.
        return tuple(dict.fromkeys((containing, adjacent % n_cells)))

    def _eddy_contribution(
        self,
        point: np.ndarray,
        scale: int,
        cell_index: tuple[int, int, int],
    ) -> np.ndarray:
        n_cells = 2**scale
        cell_size = self._domain / n_cells
        center = (np.asarray(cell_index, dtype=float) + 0.5) * cell_size
        displacement = (point - center + 0.5 * self._domain) % self._domain - 0.5 * self._domain

        # The support width is 2*cell_size. Thus X=d_x/(2*ell_x), etc.,
        # and support membership is |X| <= 1/2 in every direction.
        local = displacement / (2.0 * cell_size)
        if np.any(np.abs(local) > 0.5):
            return np.zeros(3, dtype=float)

        parameters = self._eddy_parameters(scale, *cell_index)
        distorted = local + parameters.gamma * (local**2 - 0.25)
        distorted_derivative = 1.0 + 2.0 * parameters.gamma * local
        basis = compact_basis(distorted)
        basis_derivative = compact_basis_derivative(distorted)

        scale_ratio = 2.0 ** (-scale)  # ell_m / ell_0 and ell_m / L0 isotropically.
        gradient = np.empty(3, dtype=float)
        for axis in range(3):
            other_axes = [index for index in range(3) if index != axis]
            gradient[axis] = (
                scale_ratio
                * basis_derivative[axis]
                * distorted_derivative[axis]
                / (2.0 * cell_size[axis])
                * basis[other_axes[0]]
                * basis[other_axes[1]]
            )

        epsilon = self.energy_transfer_factor(scale, cell_index)
        eddy_amplitude = (
            parameters.sign
            * self.config.amplitude
            * epsilon ** self.config.epsilon_exponent
            * scale_ratio ** self.config.spectral_exponent_h
        )
        # A = a*q_hat*phi, so curl(A) = a*grad(phi) x q_hat.
        return eddy_amplitude * np.cross(gradient, parameters.q_hat)

    def _eddy_contribution_cached(
        self,
        point: np.ndarray,
        scale: int,
        cell_index: tuple[int, int, int],
    ) -> np.ndarray:
        """Evaluate one contribution using only identity-dependent cached data."""

        n_cells = 2**scale
        cell_size = self._domain / n_cells
        center = (np.asarray(cell_index, dtype=float) + 0.5) * cell_size
        displacement = (
            (point - center + 0.5 * self._domain) % self._domain
            - 0.5 * self._domain
        )
        local = displacement / (2.0 * cell_size)
        if np.any(np.abs(local) > 0.5):
            return np.zeros(3, dtype=float)

        cached = self._cached_eddy_data(scale, cell_index)
        parameters = cached.parameters
        distorted = local + parameters.gamma * (local**2 - 0.25)
        distorted_derivative = 1.0 + 2.0 * parameters.gamma * local
        basis = compact_basis(distorted)
        basis_derivative = compact_basis_derivative(distorted)

        scale_ratio = 2.0 ** (-scale)
        gradient = np.empty(3, dtype=float)
        for axis in range(3):
            other_axes = [index for index in range(3) if index != axis]
            gradient[axis] = (
                scale_ratio
                * basis_derivative[axis]
                * distorted_derivative[axis]
                / (2.0 * cell_size[axis])
                * basis[other_axes[0]]
                * basis[other_axes[1]]
            )
        return cached.amplitude * np.cross(gradient, parameters.q_hat)

    def _contributing_cells(
        self, point: np.ndarray, scale: int
    ) -> tuple[tuple[int, int, int], ...]:
        """Return only nearby cell identities whose supports can contain a point."""

        n_cells = 2**scale
        cell_size = self._domain / n_cells
        nearby = [
            self._nearby_indices(point[axis], cell_size[axis], n_cells)
            for axis in range(3)
        ]
        return tuple(product(*nearby))

    def _evaluate_point(self, point: np.ndarray) -> np.ndarray:
        value = np.zeros(3, dtype=float)
        for scale in range(self.config.injection_scale_index, self.config.n_scales + 1):
            for cell_index in self._contributing_cells(point, scale):
                value += self._eddy_contribution(point, scale, cell_index)
        return value

    def _evaluate_point_cached(self, point: np.ndarray) -> np.ndarray:
        value = np.zeros(3, dtype=float)
        for scale in range(self.config.injection_scale_index, self.config.n_scales + 1):
            for cell_index in self._contributing_cells(point, scale):
                value += self._eddy_contribution_cached(point, scale, cell_index)
        return value

    def evaluate_vector(self, positions: np.ndarray) -> np.ndarray:
        """Evaluate the field with output shape identical to ``positions``.

        ``positions`` may have arbitrary leading dimensions and must have final
        dimension 3. Coordinates are wrapped into the periodic domain before
        the local eddy sum is evaluated.
        """

        positions = np.asarray(positions, dtype=float)
        if positions.ndim < 1 or positions.shape[-1] != 3:
            raise ValueError("positions must have final dimension 3")
        if not np.all(np.isfinite(positions)):
            raise ValueError("positions must contain only finite values")
        wrapped = np.mod(positions, self._domain)
        flat_positions = wrapped.reshape(-1, 3)
        flat_values = np.empty_like(flat_positions)
        for index, point in enumerate(flat_positions):
            flat_values[index] = self._evaluate_point(point)
        return flat_values.reshape(positions.shape)

    def evaluate_vector_cached(self, positions: np.ndarray) -> np.ndarray:
        """Evaluate with a lazy cache of deterministic per-eddy quantities.

        Point-dependent coordinates, basis values, gradients, and curl
        contributions are always recomputed. The summation order is identical
        to :meth:`evaluate_vector`.
        """

        positions = np.asarray(positions, dtype=float)
        if positions.ndim < 1 or positions.shape[-1] != 3:
            raise ValueError("positions must have final dimension 3")
        if not np.all(np.isfinite(positions)):
            raise ValueError("positions must contain only finite values")
        wrapped = np.mod(positions, self._domain)
        flat_positions = wrapped.reshape(-1, 3)
        flat_values = np.empty_like(flat_positions)
        for index, point in enumerate(flat_positions):
            flat_values[index] = self._evaluate_point_cached(point)
        return flat_values.reshape(positions.shape)

    def evaluate_vector_chunked(
        self,
        positions: np.ndarray,
        chunk_size: int = 10_000,
    ) -> np.ndarray:
        """Evaluate positions serially in bounded-size chunks.

        Positions may have arbitrary leading dimensions and a final dimension of
        three. The point ordering and per-point arithmetic are identical to
        :meth:`evaluate_vector`; chunking only bounds each call and prepares the
        API for a future process-based backend.
        """

        position_array = np.asarray(positions, dtype=float)
        if position_array.ndim < 1 or position_array.shape[-1] != 3:
            raise ValueError("positions must have final dimension 3")
        if not np.all(np.isfinite(position_array)):
            raise ValueError("positions must contain only finite values")
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)):
            raise TypeError("chunk_size must be an integer")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        flat_positions = position_array.reshape(-1, 3)
        flat_values = np.empty_like(flat_positions)
        for start in range(0, flat_positions.shape[0], int(chunk_size)):
            stop = min(start + int(chunk_size), flat_positions.shape[0])
            flat_values[start:stop] = self.evaluate_vector(flat_positions[start:stop])
        return flat_values.reshape(position_array.shape)

    def collect_touched_eddy_statistics(self, positions: np.ndarray) -> dict[str, object]:
        """Count eddy identities that would be touched during evaluation.

        This diagnostic uses the same periodic wrapping and local-eddy selection
        as :meth:`evaluate_vector`, but does not reconstruct parameters or
        calculate field contributions. Eddy identities follow the internal
        zero-based convention ``(scale, i, j, k)``.
        """

        position_array = np.asarray(positions, dtype=float)
        if position_array.ndim < 1 or position_array.shape[-1] != 3:
            raise ValueError("positions must have final dimension 3")
        if not np.all(np.isfinite(position_array)):
            raise ValueError("positions must contain only finite values")

        wrapped = np.mod(position_array, self._domain)
        flat_positions = wrapped.reshape(-1, 3)
        scales = range(
            self.config.injection_scale_index,
            self.config.n_scales + 1,
        )
        touches_by_scale = {scale: 0 for scale in scales}
        unique_by_scale: dict[int, set[tuple[int, int, int, int]]] = {
            scale: set() for scale in scales
        }

        for point in flat_positions:
            for scale in scales:
                cells = self._contributing_cells(point, scale)
                touches_by_scale[scale] += len(cells)
                unique_by_scale[scale].update(
                    (scale, i, j, k) for i, j, k in cells
                )

        total_eddy_touches = sum(touches_by_scale.values())
        unique_eddies = set().union(*unique_by_scale.values())
        n_unique_eddies = len(unique_eddies)
        reuse_factor = (
            total_eddy_touches / n_unique_eddies if n_unique_eddies else 0.0
        )
        return {
            "n_points": flat_positions.shape[0],
            # ``n_scales`` retains its public meaning: maximum zero-based scale.
            "n_scales": self.config.n_scales,
            "n_evaluated_scales": len(touches_by_scale),
            "total_eddy_touches": total_eddy_touches,
            "n_unique_eddies": n_unique_eddies,
            "reuse_factor": reuse_factor,
            "unique_eddies_by_scale": {
                scale: len(identities)
                for scale, identities in unique_by_scale.items()
            },
            "touches_by_scale": touches_by_scale,
        }

    def evaluate_vector_parallel(
        self,
        positions: np.ndarray,
        chunk_size: int = 10_000,
        n_workers: int | None = None,
        use_cache: bool = True,
    ) -> np.ndarray:
        """Evaluate position chunks in independent worker processes.

        Workers construct evaluators from the serializable configuration and
        maintain separate lazy caches. ``ProcessPoolExecutor.map`` preserves
        chunk order, so output is deterministic across worker counts.
        """

        position_array = np.asarray(positions, dtype=float)
        if position_array.ndim < 1 or position_array.shape[-1] != 3:
            raise ValueError("positions must have final dimension 3")
        if not np.all(np.isfinite(position_array)):
            raise ValueError("positions must contain only finite values")
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)):
            raise TypeError("chunk_size must be an integer")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if n_workers is not None:
            if isinstance(n_workers, bool) or not isinstance(n_workers, int):
                raise TypeError("n_workers must be an integer or None")
            if n_workers <= 0:
                raise ValueError("n_workers must be positive")

        flat_positions = position_array.reshape(-1, 3)
        if flat_positions.shape[0] == 0:
            return np.empty_like(position_array)
        chunks = [
            flat_positions[start : start + int(chunk_size)]
            for start in range(0, flat_positions.shape[0], int(chunk_size))
        ]
        tasks = [(self.config, chunk, use_cache) for chunk in chunks]
        if n_workers == 1:
            results = [_evaluate_malara_chunk(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                results = list(executor.map(_evaluate_malara_chunk, tasks))
        return np.concatenate(results, axis=0).reshape(position_array.shape)


class MalaraTurbulenceField(FieldModel):
    """Observation-level field adapter for Malara vector fluctuations.

    Density is constant in this first implementation. Velocity and magnetic
    field fluctuations may share one deterministic realization or select an
    independent realization through ``realization_offset``.
    """

    name = "malara_turbulence"

    def __init__(self, config: dict[str, Any]) -> None:
        domain = config.get("domain", {})
        if domain.get("units", "km") != "km":
            raise ValueError("Malara field domain units must currently be 'km'")
        domain_size = vector3(domain.get("L", ()), "field.domain.L")
        if np.any(domain_size <= 0):
            raise ValueError("field.domain.L values must be positive")
        self.origin = vector3(domain.get("origin", (0.0, 0.0, 0.0)), "field.domain.origin")

        turbulence = config.get("turbulence", {})
        geometry = str(turbulence.get("geometry", "isotropic"))
        if geometry != "isotropic":
            raise NotImplementedError("Only isotropic Malara turbulence is implemented")
        self.backend = str(turbulence.get("backend", "cached"))
        if self.backend not in {"direct", "chunked", "cached", "parallel"}:
            raise ValueError(
                "field.turbulence.backend must be direct, chunked, cached, or parallel"
            )
        self.chunk_size = int(turbulence.get("chunk_size", 10_000))
        if self.chunk_size <= 0:
            raise ValueError("field.turbulence.chunk_size must be positive")
        configured_workers = turbulence.get("n_workers")
        self.n_workers = (
            None if configured_workers is None else int(configured_workers)
        )
        if self.n_workers is not None and self.n_workers <= 0:
            raise ValueError("field.turbulence.n_workers must be positive or null")
        self.parallel_use_cache = bool(turbulence.get("use_cache", True))
        # grad(phi) carries 1/L0. Default the vector-potential amplitude to L0
        # so this adapter exposes dimensionless fluctuations and the output
        # amplitudes retain their requested physical units.
        characteristic_length = float(np.cbrt(np.prod(domain_size)))
        self.base_config = MalaraConfig(
            domain_size=tuple(domain_size),
            n_scales=int(turbulence.get("n_scales", 6)),
            injection_scale_index=int(turbulence.get("injection_scale_index", 2)),
            spectral_exponent_h=float(turbulence.get("spectral_exponent_h", 1.0 / 3.0)),
            intermittency_p=float(turbulence.get("intermittency_p", 0.7)),
            amplitude=float(turbulence.get("amplitude", characteristic_length)),
            base_seed=int(turbulence.get("base_seed", 12345)),
            realization=int(turbulence.get("realization", 1)),
            epsilon_exponent=float(
                turbulence.get("epsilon_exponent", DEFAULT_EPSILON_EXPONENT)
            ),
        )

        output_fields = config.get("output_fields")
        if not isinstance(output_fields, dict):
            raise ValueError("Malara field requires an output_fields mapping")
        missing = [quantity for quantity in ("n", "v", "B") if quantity not in output_fields]
        if missing:
            raise ValueError("Malara output_fields is missing: " + ", ".join(missing))

        density = output_fields["n"]
        self.background_n = float(density.get("background", 0.0))
        if not np.isfinite(self.background_n):
            raise ValueError("field.output_fields.n.background must be finite")
        density_fluctuation = str(density.get("fluctuation", "constant"))
        if density_fluctuation not in {"constant", "none"}:
            raise NotImplementedError(
                "Malara density fluctuation must be 'constant' (or legacy 'none'); "
                "compressive density turbulence is not implemented"
            )
        self.density_fluctuation = "constant"

        self.background_vectors: dict[str, np.ndarray] = {}
        self.fluctuation_modes: dict[str, str] = {}
        self.fluctuation_amplitudes: dict[str, float] = {}
        self._quantity_evaluators: dict[str, MalaraTurbulence | None] = {}
        evaluators_by_realization: dict[int, MalaraTurbulence] = {}
        for quantity in ("v", "B"):
            field_config = output_fields[quantity]
            self.background_vectors[quantity] = vector3(
                field_config.get("background", (0.0, 0.0, 0.0)),
                f"field.output_fields.{quantity}.background",
            )
            fluctuation = str(field_config.get("fluctuation", "none"))
            if fluctuation not in {"none", "malara"}:
                raise ValueError(
                    f"field.output_fields.{quantity}.fluctuation must be 'none' or 'malara'"
                )
            amplitude = float(field_config.get("amplitude", 0.0))
            if not np.isfinite(amplitude):
                raise ValueError(f"field.output_fields.{quantity}.amplitude must be finite")
            self.fluctuation_modes[quantity] = fluctuation
            self.fluctuation_amplitudes[quantity] = amplitude
            if fluctuation == "none":
                self._quantity_evaluators[quantity] = None
                continue

            realization_offset = int(field_config.get("realization_offset", 0))
            effective_realization = self.base_config.realization + realization_offset
            if effective_realization < 0:
                raise ValueError("Effective Malara realization must be non-negative")
            if effective_realization not in evaluators_by_realization:
                evaluator_config = replace(
                    self.base_config, realization=effective_realization
                )
                evaluators_by_realization[effective_realization] = MalaraTurbulence(
                    evaluator_config
                )
            self._quantity_evaluators[quantity] = evaluators_by_realization[
                effective_realization
            ]

    def evaluate_quantity(
        self, quantity: str, time: np.ndarray, position: np.ndarray
    ) -> np.ndarray:
        if quantity not in {"n", "v", "B"}:
            raise ValueError(f"Unknown field quantity: {quantity}")
        position = np.asarray(position, dtype=float)
        time = np.asarray(time, dtype=float)
        if position.ndim != 3 or position.shape[-1] != 3:
            raise ValueError("position must have shape (n_spacecraft, n_time, 3)")
        n_spacecraft, n_time, _ = position.shape
        if time.shape not in {(n_time,), (n_time, n_spacecraft)}:
            raise ValueError(
                "time must have shape (n_time,) or (n_time, n_spacecraft)"
            )
        if quantity == "n":
            return np.full((n_time, n_spacecraft), self.background_n, dtype=float)

        background = self.background_vectors[quantity]
        values = np.broadcast_to(background, (n_time, n_spacecraft, 3)).copy()
        evaluator = self._quantity_evaluators[quantity]
        if evaluator is not None and self.fluctuation_amplitudes[quantity] != 0.0:
            positions_time_first = np.swapaxes(position, 0, 1)
            relative_positions = positions_time_first - self.origin
            values += self.fluctuation_amplitudes[quantity] * self._evaluate_turbulence(
                evaluator, relative_positions
            )
        return values

    def _evaluate_turbulence(
        self, evaluator: MalaraTurbulence, positions: np.ndarray
    ) -> np.ndarray:
        """Dispatch one vector fluctuation evaluation to the configured backend."""

        if self.backend == "direct":
            return evaluator.evaluate_vector(positions)
        if self.backend == "chunked":
            return evaluator.evaluate_vector_chunked(positions, self.chunk_size)
        if self.backend == "cached":
            return evaluator.evaluate_vector_cached(positions)
        return evaluator.evaluate_vector_parallel(
            positions,
            chunk_size=self.chunk_size,
            n_workers=self.n_workers,
            use_cache=self.parallel_use_cache,
        )
