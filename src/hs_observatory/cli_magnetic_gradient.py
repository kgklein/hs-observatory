"""Module fallback for the magnetic-gradient plotting CLI."""

from hs_observatory.cli import plot_magnetic_gradient_main


if __name__ == "__main__":
    raise SystemExit(plot_magnetic_gradient_main())
