#!/usr/bin/env python3
"""Populate validation truth fields in an RBF reconstruction HDF5 product."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hs_observatory.analysis.rbf_reconstruction import (  # noqa: E402
    add_simulation_reference_to_rbf_output,
)


def _existing_observation_path(handle: h5py.File) -> str | None:
    value = handle["metadata"].attrs.get("input_path") if "metadata" in handle else None
    if value is None or value == "null":
        return None
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _write_missing_validation_derivatives(handle: h5py.File, *, overwrite: bool) -> None:
    if "reconstruction/B" not in handle:
        raise KeyError("Reconstruction HDF5 is missing /reconstruction/B")
    if "validation/B_actual" not in handle:
        raise KeyError("Reconstruction HDF5 is missing /validation/B_actual")
    reconstructed = np.asarray(handle["reconstruction/B"], dtype=float)
    actual = np.asarray(handle["validation/B_actual"], dtype=float)
    if reconstructed.shape != actual.shape:
        raise ValueError("/reconstruction/B and /validation/B_actual shapes differ")
    reconstruction = handle["reconstruction"]
    validation = handle["validation"]
    datasets = {
        "/reconstruction/Bmag": np.linalg.norm(reconstructed, axis=-1),
        "/validation/Bmag_actual": np.linalg.norm(actual, axis=-1),
        "/validation/component_error": reconstructed - actual,
        "/validation/error_magnitude": np.linalg.norm(reconstructed - actual, axis=-1),
        "/validation/error_mag": np.linalg.norm(reconstructed - actual, axis=-1),
    }
    for path, values in datasets.items():
        group_name, dataset_name = path.rsplit("/", 1)
        group = reconstruction if group_name == "/reconstruction" else validation
        if dataset_name in group:
            if not overwrite:
                continue
            del group[dataset_name]
        group.create_dataset(dataset_name, data=values, compression="gzip")
    validation.attrs["updated_utc"] = datetime.now(timezone.utc).isoformat()
    validation.attrs["purpose"] = "validation and visualization only"


def has_truth_field(reconstruction_h5: str | Path) -> bool:
    """Return true when the standard validation truth dataset is present."""

    with h5py.File(reconstruction_h5, "r") as handle:
        return "validation/B_actual" in handle


def add_truth_field_to_reconstruction_h5(
    reconstruction_h5: str | Path,
    *,
    generation_config: str | Path | None = None,
    observation_h5: str | Path | None = None,
    output_h5: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Ensure ``/validation/B_actual`` and derived error arrays are stored.

    The truth sampling is outside the RBF fitter. It reuses the project helper
    that samples the configured static simulation field on the saved
    reconstruction grid.
    """

    source = Path(reconstruction_h5)
    target = source if output_h5 is None else Path(output_h5)
    if target != source:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    with h5py.File(target, "r") as handle:
        truth_exists = "validation/B_actual" in handle
        inferred_observation = _existing_observation_path(handle)
    if truth_exists and not overwrite:
        with h5py.File(target, "r+") as handle:
            _write_missing_validation_derivatives(handle, overwrite=False)
        return target

    if generation_config is None:
        raise ValueError(
            "generation_config is required when /validation/B_actual is missing "
            "or overwrite is requested"
        )
    observation = observation_h5 or inferred_observation
    if observation is None:
        raise ValueError(
            "observation_h5 is required because the reconstruction metadata does "
            "not contain an input_path"
        )
    add_simulation_reference_to_rbf_output(target, observation, generation_config)
    with h5py.File(target, "r+") as handle:
        handle["validation"].attrs["purpose"] = "validation and visualization only"
        _write_missing_validation_derivatives(handle, overwrite=True)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction", required=True, help="RBF reconstruction HDF5")
    parser.add_argument(
        "--generation-config",
        help="generation YAML containing the original simulation-sampled field config",
    )
    parser.add_argument(
        "--observation",
        help="sampled observation HDF5; defaults to /metadata input_path",
    )
    parser.add_argument(
        "--output",
        help="optional output HDF5 copy; defaults to modifying --reconstruction in place",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace existing truth fields")
    args = parser.parse_args(argv)
    output = add_truth_field_to_reconstruction_h5(
        args.reconstruction,
        generation_config=args.generation_config,
        observation_h5=args.observation,
        output_h5=args.output,
        overwrite=args.overwrite,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
