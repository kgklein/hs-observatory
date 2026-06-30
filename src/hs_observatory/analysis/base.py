"""Interface implemented by all analysis products."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class AnalysisMethod(ABC):
    name: str

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def run(self, input_path: str | Path) -> dict[str, Any]:
        """Read an observation file and return JSON-serializable products."""
