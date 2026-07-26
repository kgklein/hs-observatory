"""Module fallback for the configuration-driven analysis CLI."""

from hs_observatory.cli import analyze_main


if __name__ == "__main__":
    raise SystemExit(analyze_main())
