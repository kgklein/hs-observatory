"""Common Elsasser-variable calculations using proton mass density.

Arrays are time-first. Scalar density has shape ``(time, spacecraft)`` and
vector fields have shape ``(time, spacecraft, 3)``.
"""

from __future__ import annotations

import numpy as np

MU0_SI = 4.0e-7 * np.pi
PROTON_MASS_KG = 1.67262192369e-27


def _density_array(n_cm3: np.ndarray) -> np.ndarray:
    density = np.asarray(n_cm3, dtype=float)
    if density.ndim != 2:
        raise ValueError("Density must have shape (time, spacecraft)")
    if not np.all(np.isfinite(density)):
        raise ValueError("Density contains non-finite values")
    if np.any(density <= 0.0):
        raise ValueError("Density must be strictly positive for Alfvén normalization")
    return density


def compute_density_normalization(
    n_cm3: np.ndarray,
    method: str = "global_per_spacecraft",
) -> np.ndarray:
    """Return number density in cm^-3 used in the Alfvén conversion.

    ``instantaneous`` preserves the time-dependent density. The default
    returns one full-interval time mean per spacecraft.
    """

    density = _density_array(n_cm3)
    if method == "instantaneous":
        return density.copy()
    if method == "global_per_spacecraft":
        return np.mean(density, axis=0)
    # TODO: Add rolling-window and explicitly selected-interval means here.
    raise ValueError(
        "density normalization method must be 'instantaneous' or 'global_per_spacecraft'"
    )


def magnetic_field_to_alfven_units(
    B_nT: np.ndarray,
    n_normalization_cm3: np.ndarray,
) -> np.ndarray:
    """Convert magnetic field from nT to km/s using ``rho = n * m_p``."""

    magnetic = np.asarray(B_nT, dtype=float)
    if magnetic.ndim != 3 or magnetic.shape[-1] != 3:
        raise ValueError("Magnetic field must have shape (time, spacecraft, 3)")
    if not np.all(np.isfinite(magnetic)):
        raise ValueError("Magnetic field contains non-finite values")
    density = np.asarray(n_normalization_cm3, dtype=float)
    if density.ndim == 1:
        if density.shape != (magnetic.shape[1],):
            raise ValueError("Per-spacecraft density does not match the magnetic field")
        density = density[None, :]
    elif density.shape != magnetic.shape[:2]:
        raise ValueError("Instantaneous density must match magnetic field (time, spacecraft)")
    if not np.all(np.isfinite(density)) or np.any(density <= 0.0):
        raise ValueError("Density normalization must be finite and strictly positive")

    n_m3 = density * 1.0e6
    B_tesla = magnetic * 1.0e-9
    rho_kg_m3 = n_m3 * PROTON_MASS_KG
    return B_tesla / np.sqrt(MU0_SI * rho_kg_m3)[..., None] / 1000.0


def compute_elsasser(
    v_km_s: np.ndarray,
    b_km_s: np.ndarray,
    mean_subtraction: str = "global_per_spacecraft",
) -> dict[str, np.ndarray]:
    """Compute fluctuations and Elsasser fields with the package convention."""

    velocity = np.asarray(v_km_s, dtype=float)
    magnetic = np.asarray(b_km_s, dtype=float)
    if velocity.shape != magnetic.shape or velocity.ndim != 3 or velocity.shape[-1] != 3:
        raise ValueError("v and b must have matching shape (time, spacecraft, 3)")
    if not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(magnetic)):
        raise ValueError("v and b must contain only finite values")
    if mean_subtraction != "global_per_spacecraft":
        # TODO: Add rolling-window and selected-interval fluctuation definitions.
        raise ValueError("mean subtraction method must be 'global_per_spacecraft'")

    delta_v = velocity - np.mean(velocity, axis=0, keepdims=True)
    delta_b = magnetic - np.mean(magnetic, axis=0, keepdims=True)
    return {
        "delta_v": delta_v,
        "delta_b": delta_b,
        "z_plus": delta_v + delta_b,
        "z_minus": delta_v - delta_b,
    }
