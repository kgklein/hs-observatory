"""Module fallback for the timing-calibration plotting CLI."""

from hs_observatory.cli import plot_timing_calibrate_main


if __name__ == "__main__":
    raise SystemExit(plot_timing_calibrate_main())
