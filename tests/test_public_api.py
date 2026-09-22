"""The small package-root API stays usable without eager scientific imports."""

import importlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

import hs_observatory


def test_root_exports():
    assert hs_observatory.__all__ == [
        "__version__",
        "analyze_from_config",
        "generate_from_config",
    ]
    assert isinstance(hs_observatory.__version__, str)
    assert hs_observatory.__version__
    assert callable(hs_observatory.analyze_from_config)
    assert callable(hs_observatory.generate_from_config)


def test_analysis_is_direct_reexport():
    canonical = importlib.import_module("hs_observatory.analyze")
    assert hs_observatory.analyze_from_config is canonical.analyze_from_config


def test_generation_signature_matches_canonical_implementation():
    canonical = importlib.import_module("hs_observatory.generate")
    assert inspect.signature(
        hs_observatory.generate_from_config, eval_str=True
    ) == inspect.signature(canonical.generate_from_config, eval_str=True)


@pytest.mark.parametrize(
    "overrides",
    [{}, {"truth_velocity_frame": "plasma", "configuration_3d_extra_connections": ["N1-N2"]}],
)
def test_generation_forwards_arguments_and_result(overrides):
    canonical = importlib.import_module("hs_observatory.generate")
    config_path = Path("not-read.yaml")
    output = Path("not-written.h5")
    with patch.object(canonical, "generate_from_config", return_value=output) as implementation:
        assert hs_observatory.generate_from_config(config_path, **overrides) is output
        implementation.assert_called_once_with(
            config_path,
            truth_velocity_frame=overrides.get("truth_velocity_frame"),
            configuration_3d_extra_connections=overrides.get("configuration_3d_extra_connections"),
        )


def test_root_import_does_not_load_scientific_implementations():
    # A fresh interpreter avoids modules imported by unrelated tests during collection.
    code = """
import json
import sys
import hs_observatory
assert callable(hs_observatory.generate_from_config)
assert callable(hs_observatory.analyze_from_config)
print(json.dumps(sorted(sys.modules)))
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], capture_output=True, text=True, check=True
    )
    loaded = set(json.loads(result.stdout))
    infrastructure = {
        "hs_observatory.analysis",
        "hs_observatory.analysis.base",
        "hs_observatory.analysis.registry",
    }
    assert {name for name in loaded if name.startswith("hs_observatory.analysis")} <= infrastructure
    assert "hs_observatory.generate" not in loaded
    for dependency in ("numpy", "h5py", "matplotlib", "scipy", "adios2", "cmcrameri"):
        assert dependency not in loaded
