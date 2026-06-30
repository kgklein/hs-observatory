"""Interface implemented by all analysis products."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any


class AnalysisMethod(ABC):
    name: str
    config_section: str | None = None

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def run(self, input_path: str | Path) -> dict[str, Any]:
        """Read an observation file and return analysis products."""

    def write_result(self, result: dict[str, Any], output_path: str | Path) -> Path:
        """Write JSON by default; binary analyses may override this method."""

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return output
