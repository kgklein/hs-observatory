"""Configuration-driven analysis workflow."""

from __future__ import annotations

from pathlib import Path

from hs_observatory.analysis.registry import get_analysis
from hs_observatory.config import load_analysis_config


def analyze_from_config(config_path: str | Path, *, workers: int = 1) -> Path:
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    loaded = load_analysis_config(config_path)
    config = loaded.data
    analysis_config = config["analysis"]
    method_class = get_analysis(str(analysis_config["method"]))
    if workers > 1 and method_class.name != "structure_function":
        raise ValueError("--workers > 1 is supported only for structure_function analysis")
    options = dict(analysis_config.get("options", {}))
    if method_class.config_section is not None:
        section = config.get(method_class.config_section, {})
        if not isinstance(section, dict):
            raise ValueError(f"{method_class.config_section} configuration must be a mapping")
        options.update(section)
    if method_class.name in {"magnetic_gradient", "timing-calibrate", "shock_timing"}:
        options.setdefault("_analysis_config_yaml", loaded.yaml_text)
        options.setdefault("_analysis_config_path", str(loaded.path))
    method = method_class(options)
    if method_class.name == "structure_function":
        result = method.run(config["input"]["path"], workers=workers)
    else:
        result = method.run(config["input"]["path"])

    output = Path(config["output"]["path"])
    return method.write_result(result, output)
