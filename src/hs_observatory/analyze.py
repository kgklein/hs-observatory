"""Configuration-driven analysis workflow."""

from __future__ import annotations

import json
from pathlib import Path

from hs_observatory.analysis.registry import get_analysis
from hs_observatory.config import load_analysis_config


def analyze_from_config(config_path: str | Path) -> Path:
    loaded = load_analysis_config(config_path)
    config = loaded.data
    analysis_config = config["analysis"]
    method_class = get_analysis(str(analysis_config["method"]))
    method = method_class(analysis_config.get("options", {}))
    result = method.run(config["input"]["path"])

    output = Path(config["output"]["path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output
