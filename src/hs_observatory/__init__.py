"""Synthetic HelioSwarm-like observation generation and analysis."""

from __future__ import annotations

from pathlib import Path as _Path

__version__ = "0.1.0"

from .analyze import analyze_from_config

__all__ = [
    "__version__",
    "analyze_from_config",
    "generate_from_config",
]


def generate_from_config(
    config_path: str | _Path,
    *,
    truth_velocity_frame: str | None = None,
    configuration_3d_extra_connections: list[str] | None = None,
) -> _Path:
    """Generate observations from a YAML file and return the output HDF5 path.

    Optional keywords override truth-plot velocity framing and add 3-D plot
    connections, as in :func:`hs_observatory.generate.generate_from_config`.
    Generation and plotting dependencies are imported only when called.
    """
    from .generate import generate_from_config as _generate_from_config

    return _generate_from_config(
        config_path,
        truth_velocity_frame=truth_velocity_frame,
        configuration_3d_extra_connections=configuration_3d_extra_connections,
    )
