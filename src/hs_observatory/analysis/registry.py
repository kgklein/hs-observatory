"""Name-to-analysis registry used by YAML and third-party extensions."""

from __future__ import annotations

from typing import TypeVar

from hs_observatory.analysis.base import AnalysisMethod

T = TypeVar("T", bound=type[AnalysisMethod])
_REGISTRY: dict[str, type[AnalysisMethod]] = {}


def register_analysis(cls: T) -> T:
    if not cls.name:
        raise ValueError("Registered analysis methods require a non-empty name")
    if cls.name in _REGISTRY:
        raise ValueError(f"Analysis method already registered: {cls.name}")
    _REGISTRY[cls.name] = cls
    return cls


def _load_builtins() -> None:
    # Import side effects populate the registry while keeping base imports light.
    from hs_observatory.analysis import (  # noqa: F401
        diagnostics,
        elsasser,
        elsasser_increment_pdf_static,
        magnetic_gradient,
        magnetic_gradient_correlation,
        rbf_reconstruction,
        shock_timing,
        spatial_structure_function,
        static_decorrelation,
        structure_function,
        timesync,
        timeseries_fft,
        timing_calibrate,
    )


def get_analysis(name: str) -> type[AnalysisMethod]:
    _load_builtins()
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown analysis method {name!r}; available: {', '.join(_REGISTRY)}"
        ) from exc


def list_analyses() -> tuple[str, ...]:
    _load_builtins()
    return tuple(sorted(_REGISTRY))
