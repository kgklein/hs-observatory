"""Configuration-driven analysis workflow."""

from __future__ import annotations

from pathlib import Path

from hs_observatory.analysis.registry import get_analysis
from hs_observatory.config import load_analysis_config


def analyze_from_config(config_path: str | Path) -> Path:
    loaded = load_analysis_config(config_path)
    config = loaded.data
    analysis_config = config["analysis"]
    method_class = get_analysis(str(analysis_config["method"]))
    options = dict(analysis_config.get("options", {}))
    if method_class.config_section is not None:
        section = config.get(method_class.config_section, {})
        if not isinstance(section, dict):
            raise ValueError(f"{method_class.config_section} configuration must be a mapping")
        options.update(section)
    if method_class.name in {"magnetic_gradient", "timing-calibrate"}:
        options.setdefault("_analysis_config_yaml", loaded.yaml_text)
        options.setdefault("_analysis_config_path", str(loaded.path))
    method = method_class(options)
    result = method.run(config["input"]["path"])

    output = Path(config["output"]["path"])
    return method.write_result(result, output)
