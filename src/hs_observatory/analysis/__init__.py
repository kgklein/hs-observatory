"""Analysis method interfaces and built-in registrations."""

from .registry import get_analysis, list_analyses, register_analysis

__all__ = ["get_analysis", "list_analyses", "register_analysis"]
