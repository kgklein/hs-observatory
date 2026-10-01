"""Reproduce paired noise-normalization experiments without external simulation data.

Run from the repository root: .venv/bin/python analysis/timing_noise_experiment.py
"""

from pathlib import Path

import h5py
import numpy as np
import yaml

from hs_observatory.analysis.timing_calibrate import (
    compute_timing_calibration,
    write_timing_calibration_product,
)
from hs_observatory.cli import plot_timing_calibrate_main
from hs_observatory.io.hdf5 import write_observations


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs/timing_noise_experiment"
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260924)
    time = np.arange(401) / 32
    points = rng.normal(size=(9, 3)) * 100  # km
    positions = np.broadcast_to(points, (len(time), 9, 3))
    # Smooth, exactly solenoidal Fourier field. Its finite-volume divergence
    # need not vanish; the fit receives no spatial-curvature correction.
    wavevectors = rng.normal(size=(16, 3)) / 300
    amplitude = rng.normal(size=(16, 3))
    amplitude -= (np.sum(amplitude * wavevectors, axis=1)
                  / np.sum(wavevectors**2, axis=1))[:, None] * wavevectors
    phase = rng.uniform(0, 2 * np.pi, 16)
    omega = np.linspace(.5, 8, 16)

    def field(offsets):
        argument = ((time[:, None, None] + offsets[None, :, None]) * omega
                    + points @ wavevectors.T + phase)
        return np.einsum("tnm,mi->tni", np.sin(argument), amplitude)

    ideal = field(np.zeros(9))
    density = np.ones((len(time), 9))
    velocity = np.zeros_like(ideal)
    summary = {}
    for case in ("zero", "injected"):
        offsets = np.zeros((4, 9))
        if case == "injected":
            offsets[:, 1:] = rng.normal(0, .004, (4, 8))
        measured = np.stack([field(offset) for offset in offsets])
        source = output / f"{case}_source.h5"
        write_observations(
            source, time={key: time for key in ("n", "v", "B")},
            spacecraft_ids=["H", *(f"N{i}" for i in range(1, 9))],
            position={key: positions.swapaxes(0, 1) for key in ("n", "v", "B")},
            truth={"n": density, "v": velocity, "B": ideal},
            observed={"n": np.broadcast_to(density, (4, *density.shape)),
                      "v": np.broadcast_to(velocity, (4, *velocity.shape)), "B": measured},
            errors={"time_offset": offsets},
            metadata={"config_yaml": yaml.safe_dump({"errors": {"time": {"model": "none"}}})},
        )
        products = []
        for normalization in ("ideal_divergence", "instrument_noise"):
            config = {
                "sigma_t_s": .004, "polyhedron_filter": {"allowed_n": [8, 9]},
                "divergence_normalization": normalization,
                "instrument_noise": {
                    "spectrum_file": str(root / "configs/helioswarm_mag_scm_noise.yaml"),
                    "interpretation": "per_component", "below_range": "flat",
                    "above_range": "error", "variance_floor_nT2_per_km2": 0.,
                    "processing": {"sampling": "ideal_antialias"},
                },
            }
            (output / f"{case}_{normalization}.yaml").write_text(yaml.safe_dump(config))
            product = write_timing_calibration_product(
                compute_timing_calibration(source, config),
                output / f"{case}_{normalization}.h5",
            )
            products.append(product)
            with h5py.File(product) as handle:
                error = handle["timing/recovery_error_s"][...]
                summary[f"{case}_{normalization}"] = {
                    "bias_per_node_ms": (1000 * error.mean(axis=0)).tolist(),
                    "rmse_per_node_ms": (1000 * np.sqrt(np.mean(error**2, axis=0))).tolist(),
                    "pooled_rmse_ms": float(1000 * np.sqrt(np.mean(error**2))),
                    "divergence_objective_fraction_after": handle[
                        "diagnostics/divergence_objective_fraction_after"][...].tolist(),
                }
        plot_timing_calibrate_main([
            str(products[1]), "--output", str(output / f"{case}.png"),
            "--compare-products", str(products[0]), "--dpi", "120",
        ])
    (output / "summary.yaml").write_text(yaml.safe_dump(summary))
    print(yaml.safe_dump(summary))


if __name__ == "__main__":
    main()
