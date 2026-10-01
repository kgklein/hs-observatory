"""Propagating planar ideal-MHD shock field."""

from __future__ import annotations

from typing import Any

import numpy as np

from hs_observatory.fields.base import FieldModel
from hs_observatory.utils.units import vector3

MU0 = 4.0e-7 * np.pi
PROTON_MASS_KG = 1.67262192369e-27
TRANSITION_COEFFICIENT = 2.197224577
JUMP_RESIDUAL_TOLERANCE = 1.0e-8


def _finite_vector(value: Any, name: str) -> np.ndarray:
    result = vector3(value, name)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _fast_speed(
    rho: float,
    sound_squared: float,
    magnetic_field: np.ndarray,
    normal: np.ndarray,
) -> float:
    """Return the fast magnetosonic speed along ``normal`` in SI units."""

    magnetic_squared = float(np.dot(magnetic_field, magnetic_field)) / (MU0 * rho)
    normal_alfven_squared = float(np.dot(magnetic_field, normal)) ** 2 / (MU0 * rho)
    discriminant = max(
        0.0,
        (sound_squared + magnetic_squared) ** 2 - 4.0 * sound_squared * normal_alfven_squared,
    )
    return float(np.sqrt(0.5 * (sound_squared + magnetic_squared + np.sqrt(discriminant))))


class ShockField(FieldModel):
    """A finite-width observation model joining two ideal-MHD shock states.

    The uniform endpoints satisfy the Rankine--Hugoniot conditions. The tanh
    interpolation between them is synthetic and is not claimed to resolve a
    physical shock structure.
    """

    name = "shock"

    def __init__(self, config: dict[str, Any]) -> None:
        shock = config.get("shock")
        if not isinstance(shock, dict):
            raise ValueError("field.shock must be a mapping")
        upstream = shock.get("upstream")
        if not isinstance(upstream, dict):
            raise ValueError("field.shock.upstream must be a mapping")

        supplied_normal = _finite_vector(shock.get("normal_gse", ()), "shock.normal_gse")
        normal_magnitude = float(np.linalg.norm(supplied_normal))
        if normal_magnitude == 0.0:
            raise ValueError("shock.normal_gse must be nonzero")
        self.normal = supplied_normal / normal_magnitude
        self.shock_velocity = _finite_vector(
            shock.get("velocity_gse_km_s", ()), "shock.velocity_gse_km_s"
        )
        normal_shock_speed = float(np.dot(self.shock_velocity, self.normal))
        tangential_shock_velocity = self.shock_velocity - normal_shock_speed * self.normal
        tangential_tolerance = 1.0e-10 * max(1.0, float(np.linalg.norm(self.shock_velocity)))
        if np.linalg.norm(tangential_shock_velocity) > tangential_tolerance:
            raise ValueError(
                "shock.velocity_gse_km_s must be normal to the shock plane; "
                "a tangential component cannot propagate an infinite plane"
            )
        # Remove roundoff in vectors constructed from a non-axis-aligned normal.
        self.shock_velocity = normal_shock_speed * self.normal
        self.normal_shock_speed = normal_shock_speed

        self.distance_km = float(shock.get("distance_from_mesocenter_km"))
        self.thickness_km = float(shock.get("thickness_km"))
        self.gamma = float(shock.get("gamma", 5.0 / 3.0))
        self.reference_time_setting = shock.get("reference_time", "start")
        if not np.isfinite(self.distance_km):
            raise ValueError("shock.distance_from_mesocenter_km must be finite")
        if not np.isfinite(self.thickness_km) or self.thickness_km <= 0.0:
            raise ValueError("shock.thickness_km must be positive and finite")
        if not np.isfinite(self.gamma) or self.gamma <= 1.0:
            raise ValueError("shock.gamma must be finite and greater than one")

        density_cm3 = float(upstream.get("density_cm3"))
        pressure_npa = float(upstream.get("total_pressure_npa"))
        if not np.isfinite(density_cm3) or density_cm3 <= 0.0:
            raise ValueError("shock.upstream.density_cm3 must be positive and finite")
        if not np.isfinite(pressure_npa) or pressure_npa <= 0.0:
            raise ValueError("shock.upstream.total_pressure_npa must be positive and finite")
        self.upstream_n = density_cm3
        self.upstream_v = _finite_vector(
            upstream.get("velocity_km_s", ()), "shock.upstream.velocity_km_s"
        )
        self.upstream_B = _finite_vector(
            upstream.get("magnetic_field_nt", ()), "shock.upstream.magnetic_field_nt"
        )
        self.upstream_pressure_npa = pressure_npa

        downstream = self._solve_downstream()
        self.downstream_n = downstream["density_cm3"]
        self.downstream_v = np.asarray(downstream["velocity_km_s"], dtype=float)
        self.downstream_B = np.asarray(downstream["magnetic_field_nt"], dtype=float)
        self.downstream_pressure_npa = downstream["total_pressure_npa"]
        self.compression_ratio = downstream["compression_ratio"]
        self.fast_mach_upstream = downstream["fast_mach_upstream"]
        self.fast_mach_downstream = downstream["fast_mach_downstream"]
        self.jump_residuals = downstream["jump_residuals"]

        self.reference_time_s: float | None = None
        self.mesocenter_km: np.ndarray | None = None
        self.metadata: dict[str, Any] = {
            "coordinate_system": "GSE",
            "velocity_frame": "plasma",
            "normal_gse": self.normal.tolist(),
            "velocity_gse_km_s": self.shock_velocity.tolist(),
            "distance_from_mesocenter_km": self.distance_km,
            "thickness_10_to_90_km": self.thickness_km,
            "reference_time": self.reference_time_setting,
            "gamma": self.gamma,
            "upstream": {
                "density_cm3": self.upstream_n,
                "velocity_km_s": self.upstream_v.tolist(),
                "magnetic_field_nt": self.upstream_B.tolist(),
                "total_pressure_npa": self.upstream_pressure_npa,
            },
            "downstream": {
                "density_cm3": self.downstream_n,
                "velocity_km_s": self.downstream_v.tolist(),
                "magnetic_field_nt": self.downstream_B.tolist(),
                "total_pressure_npa": self.downstream_pressure_npa,
            },
            "compression_ratio": self.compression_ratio,
            "fast_mach_upstream": self.fast_mach_upstream,
            "fast_mach_downstream": self.fast_mach_downstream,
            "jump_residuals": self.jump_residuals,
            "finite_width_interpretation": "synthetic_observation_transition",
        }

    def _solve_downstream(self) -> dict[str, Any]:
        """Solve arbitrary-obliquity one-dimensional ideal-MHD jumps."""

        rho1 = self.upstream_n * 1.0e6 * PROTON_MASS_KG
        pressure1 = self.upstream_pressure_npa * 1.0e-9
        velocity1 = (self.upstream_v - self.shock_velocity) * 1.0e3
        magnetic1 = self.upstream_B * 1.0e-9
        normal = self.normal
        un1 = float(np.dot(velocity1, normal))
        if un1 <= 0.0:
            raise ValueError("upstream normal flow in the shock-rest frame must be positive")
        ut1 = velocity1 - un1 * normal
        bn = float(np.dot(magnetic1, normal))
        bt1 = magnetic1 - bn * normal
        mass_flux = rho1 * un1
        electric_t = un1 * bt1 - bn * ut1
        momentum_t = mass_flux * ut1 - bn * bt1 / MU0

        sound_squared1 = self.gamma * pressure1 / rho1
        fast1 = _fast_speed(rho1, sound_squared1, magnetic1, normal)
        upstream_mach = un1 / fast1
        if upstream_mach <= 1.0:
            raise ValueError("upstream state must be super-fast in the shock-rest frame")

        def candidate(compression: float) -> tuple[float, float, np.ndarray, np.ndarray, float]:
            rho2 = compression * rho1
            un2 = un1 / compression
            denominator = un2 - bn * bn / (MU0 * mass_flux)
            denominator_scale = max(abs(un2), abs(bn * bn / (MU0 * mass_flux)), 1.0)
            if abs(denominator) <= 1.0e-13 * denominator_scale:
                raise FloatingPointError("downstream state lies on the Alfvénic singularity")
            bt2 = (electric_t + bn * momentum_t / mass_flux) / denominator
            bt2 -= np.dot(bt2, normal) * normal
            ut2 = (momentum_t + bn * bt2 / MU0) / mass_flux
            ut2 -= np.dot(ut2, normal) * normal
            pressure2 = (
                rho1 * un1 * un1
                + pressure1
                + np.dot(bt1, bt1) / (2.0 * MU0)
                - rho2 * un2 * un2
                - np.dot(bt2, bt2) / (2.0 * MU0)
            )
            return rho2, un2, ut2, bt2, float(pressure2)

        def energy_flux(
            rho: float,
            un: float,
            ut: np.ndarray,
            bt: np.ndarray,
            pressure: float,
        ) -> float:
            velocity_squared = un * un + float(np.dot(ut, ut))
            magnetic_squared = bn * bn + float(np.dot(bt, bt))
            velocity_dot_magnetic = un * bn + float(np.dot(ut, bt))
            return float(
                (
                    0.5 * rho * velocity_squared
                    + self.gamma / (self.gamma - 1.0) * pressure
                    + magnetic_squared / MU0
                )
                * un
                - bn * velocity_dot_magnetic / MU0
            )

        upstream_energy = energy_flux(rho1, un1, ut1, bt1, pressure1)

        def energy_residual(compression: float) -> float:
            return energy_flux(*candidate(compression)) - upstream_energy

        compression_max = (self.gamma + 1.0) / (self.gamma - 1.0)
        boundaries = [1.0 + 1.0e-7, compression_max * (1.0 - 1.0e-9)]
        if abs(bn) > 0.0:
            singular_compression = MU0 * rho1 * un1 * un1 / (bn * bn)
            if boundaries[0] < singular_compression < boundaries[-1]:
                gap = max(1.0e-8, singular_compression * 1.0e-9)
                boundaries.extend([singular_compression - gap, singular_compression + gap])
        boundaries = sorted(boundaries)
        intervals = []
        if len(boundaries) == 2:
            intervals.append((boundaries[0], boundaries[1]))
        else:
            intervals.extend(((boundaries[0], boundaries[1]), (boundaries[2], boundaries[3])))

        roots: list[float] = []
        for lower, upper in intervals:
            if lower >= upper:
                continue
            grid = np.linspace(lower, upper, 4097)
            values = np.full(grid.shape, np.nan)
            for index, compression in enumerate(grid):
                try:
                    values[index] = energy_residual(float(compression))
                except FloatingPointError:
                    pass
            for index in range(len(grid) - 1):
                left_value = values[index]
                right_value = values[index + 1]
                if not np.isfinite(left_value) or not np.isfinite(right_value):
                    continue
                if left_value == 0.0:
                    root = float(grid[index])
                elif left_value * right_value > 0.0:
                    continue
                else:
                    left = float(grid[index])
                    right = float(grid[index + 1])
                    for _ in range(100):
                        midpoint = 0.5 * (left + right)
                        midpoint_value = energy_residual(midpoint)
                        if left_value * midpoint_value <= 0.0:
                            right = midpoint
                            right_value = midpoint_value
                        else:
                            left = midpoint
                            left_value = midpoint_value
                        if right - left <= 1.0e-12 * max(1.0, midpoint):
                            break
                    root = 0.5 * (left + right)
                if root > 1.0 + 1.0e-6 and not any(abs(root - old) < 1.0e-7 for old in roots):
                    roots.append(root)

        physical: list[tuple[float, tuple[float, float, np.ndarray, np.ndarray, float], float]] = []
        for compression in roots:
            try:
                state = candidate(compression)
            except FloatingPointError:
                continue
            rho2, un2, ut2, bt2, pressure2 = state
            if pressure2 <= 0.0:
                continue
            magnetic2 = bn * normal + bt2
            sound_squared2 = self.gamma * pressure2 / rho2
            fast2 = _fast_speed(rho2, sound_squared2, magnetic2, normal)
            downstream_mach = un2 / fast2
            entropy_ratio = (pressure2 / pressure1) / compression**self.gamma
            if downstream_mach < 1.0 and entropy_ratio > 1.0:
                physical.append((compression, state, downstream_mach))
        if len(physical) != 1:
            raise ValueError(
                "ideal-MHD jump solver did not find exactly one admissible compressive "
                f"fast-shock solution (found {len(physical)})"
            )

        compression, state, downstream_mach = physical[0]
        rho2, un2, ut2, bt2, pressure2 = state
        velocity2 = un2 * normal + ut2
        magnetic2 = bn * normal + bt2
        residuals = self._jump_residuals(
            rho1,
            pressure1,
            velocity1,
            magnetic1,
            rho2,
            pressure2,
            velocity2,
            magnetic2,
        )
        if max(residuals.values()) > JUMP_RESIDUAL_TOLERANCE:
            raise ValueError(
                "calculated downstream state fails Rankine-Hugoniot residual validation"
            )
        # Enforce normal-field continuity after solving, rather than retaining
        # a projection-level rounding difference.
        magnetic2 += (bn - float(np.dot(magnetic2, normal))) * normal
        plasma_velocity2 = velocity2 / 1.0e3 + self.shock_velocity
        return {
            "density_cm3": rho2 / (PROTON_MASS_KG * 1.0e6),
            "velocity_km_s": plasma_velocity2.tolist(),
            "magnetic_field_nt": (magnetic2 / 1.0e-9).tolist(),
            "total_pressure_npa": pressure2 / 1.0e-9,
            "compression_ratio": compression,
            "fast_mach_upstream": upstream_mach,
            "fast_mach_downstream": downstream_mach,
            "jump_residuals": residuals,
        }

    def _jump_residuals(
        self,
        rho1: float,
        pressure1: float,
        velocity1: np.ndarray,
        magnetic1: np.ndarray,
        rho2: float,
        pressure2: float,
        velocity2: np.ndarray,
        magnetic2: np.ndarray,
    ) -> dict[str, float]:
        normal = self.normal

        def fluxes(rho: float, pressure: float, velocity: np.ndarray, magnetic: np.ndarray):
            un = float(np.dot(velocity, normal))
            ut = velocity - un * normal
            bn = float(np.dot(magnetic, normal))
            bt = magnetic - bn * normal
            mass = rho * un
            normal_momentum = rho * un * un + pressure + np.dot(bt, bt) / (2.0 * MU0)
            tangent_momentum = rho * un * ut - bn * bt / MU0
            electric_t = un * bt - bn * ut
            energy = (
                0.5 * rho * np.dot(velocity, velocity)
                + self.gamma / (self.gamma - 1.0) * pressure
                + np.dot(magnetic, magnetic) / MU0
            ) * un - bn * np.dot(velocity, magnetic) / MU0
            return mass, normal_momentum, tangent_momentum, electric_t, bn, energy

        first = fluxes(rho1, pressure1, velocity1, magnetic1)
        second = fluxes(rho2, pressure2, velocity2, magnetic2)

        def scalar_residual(left: float, right: float) -> float:
            return abs(right - left) / max(abs(left), abs(right), np.finfo(float).tiny)

        def vector_residual(left: np.ndarray, right: np.ndarray) -> float:
            return float(
                np.linalg.norm(right - left)
                / max(np.linalg.norm(left), np.linalg.norm(right), np.finfo(float).tiny)
            )

        return {
            "mass": float(scalar_residual(first[0], second[0])),
            "normal_momentum": float(scalar_residual(first[1], second[1])),
            "tangential_momentum": vector_residual(first[2], second[2]),
            "tangential_electric_field": vector_residual(first[3], second[3]),
            "normal_magnetic_field": float(scalar_residual(first[4], second[4])),
            "energy": float(scalar_residual(first[5], second[5])),
        }

    def set_nominal_configuration(self, time: np.ndarray, position: np.ndarray) -> None:
        """Fix ``t0`` and its nominal arithmetic-mean formation mesocenter once."""

        if self.reference_time_s is not None:
            raise RuntimeError("shock nominal configuration has already been initialized")
        times = np.asarray(time, dtype=float)
        positions = np.asarray(position, dtype=float)
        if times.ndim != 1 or positions.ndim != 3 or positions.shape[1:] != (len(times), 3):
            raise ValueError("nominal trajectory must have shape (spacecraft, time, 3)")
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(positions)):
            raise ValueError("nominal trajectory must contain only finite values")
        if len(times) == 0 or (len(times) > 1 and np.any(np.diff(times) <= 0.0)):
            raise ValueError("nominal trajectory time must be nonempty and strictly increasing")
        if isinstance(self.reference_time_setting, str):
            if self.reference_time_setting.lower() != "start":
                raise ValueError(
                    "shock.reference_time must be 'start' or a relative time in seconds"
                )
            reference_time = float(times[0])
        else:
            reference_time = float(self.reference_time_setting)
        if (
            not np.isfinite(reference_time)
            or reference_time < times[0]
            or reference_time > times[-1]
        ):
            raise ValueError("shock.reference_time must lie within the generated time interval")
        reference_positions = np.empty((positions.shape[0], 3), dtype=float)
        for spacecraft_index in range(positions.shape[0]):
            for component in range(3):
                reference_positions[spacecraft_index, component] = np.interp(
                    reference_time,
                    times,
                    positions[spacecraft_index, :, component],
                )
        self.reference_time_s = reference_time
        self.mesocenter_km = np.mean(reference_positions, axis=0)
        self.metadata["reference_time_s"] = reference_time
        self.metadata["mesocenter_gse_km"] = self.mesocenter_km.tolist()

    def signed_distance(self, time: np.ndarray, position: np.ndarray) -> np.ndarray:
        """Return ``xi`` with negative upstream and positive downstream."""

        if self.reference_time_s is None or self.mesocenter_km is None:
            raise RuntimeError("shock field has not received its nominal configuration")
        times = np.asarray(time, dtype=float)
        positions = np.asarray(position, dtype=float)
        if positions.ndim != 3 or positions.shape[2] != 3:
            raise ValueError("position must have shape (n_spacecraft, n_time, 3)")
        n_spacecraft, n_time, _ = positions.shape
        if times.shape == (n_time,):
            time_by_spacecraft = np.broadcast_to(times[:, None], (n_time, n_spacecraft))
        elif times.shape == (n_time, n_spacecraft):
            time_by_spacecraft = times
        else:
            raise ValueError("time must have shape (n_time,) or (n_time, n_spacecraft)")
        positions_time_first = np.swapaxes(positions, 0, 1)
        spatial_distance = np.einsum(
            "tsd,d->ts", positions_time_first - self.mesocenter_km, self.normal
        )
        propagation = self.normal_shock_speed * (time_by_spacecraft - self.reference_time_s)
        return spatial_distance - self.distance_km - propagation

    def transition_fraction(self, xi: np.ndarray) -> np.ndarray:
        return 0.5 * (1.0 + np.tanh(TRANSITION_COEFFICIENT * np.asarray(xi) / self.thickness_km))

    def analytic_magnetic_divergence(self, xi: np.ndarray) -> np.ndarray:
        """Return the analytic divergence of the planar interpolated field."""

        argument = TRANSITION_COEFFICIENT * np.asarray(xi, dtype=float) / self.thickness_km
        derivative = 0.5 * TRANSITION_COEFFICIENT / self.thickness_km / np.cosh(argument) ** 2
        normal_jump = float(np.dot(self.downstream_B - self.upstream_B, self.normal))
        return derivative * normal_jump

    def evaluate_quantity(
        self, quantity: str, time: np.ndarray, position: np.ndarray
    ) -> np.ndarray:
        if quantity not in {"n", "v", "B"}:
            raise ValueError(f"Unknown field quantity: {quantity}")
        fraction = self.transition_fraction(self.signed_distance(time, position))
        if quantity == "n":
            return self.upstream_n + fraction * (self.downstream_n - self.upstream_n)
        upstream = self.upstream_v if quantity == "v" else self.upstream_B
        downstream = self.downstream_v if quantity == "v" else self.downstream_B
        return upstream + fraction[..., None] * (downstream - upstream)
