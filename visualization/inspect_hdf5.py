#!/usr/bin/env python3
"""Print a compact tree for project HDF5 products."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _format_attrs(attrs: h5py.AttributeManager, *, max_attrs: int) -> str:
    rendered: list[str] = []
    for index, (key, value) in enumerate(attrs.items()):
        if index >= max_attrs:
            rendered.append("...")
            break
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        elif isinstance(value, np.ndarray):
            if value.shape == ():
                value = value.item()
            elif value.size <= 6:
                value = value.tolist()
            else:
                value = f"array shape={value.shape} dtype={value.dtype}"
        rendered.append(f"{key}={value!r}")
    return "" if not rendered else " attrs: " + ", ".join(rendered)


def _numeric_summary(dataset: h5py.Dataset, *, max_elements: int) -> str:
    if dataset.size == 0 or dataset.size > max_elements:
        return ""
    if not np.issubdtype(dataset.dtype, np.number):
        return ""
    values = np.asarray(dataset[...])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return " min/max=non-finite"
    return f" min={np.min(finite):.6g} max={np.max(finite):.6g}"


def _visit(
    name: str,
    obj: h5py.Group | h5py.Dataset,
    *,
    max_attrs: int,
    max_summary_elements: int,
) -> None:
    depth = 0 if name == "" else name.count("/") + 1
    indent = "  " * depth
    label = "/" if name == "" else name.rsplit("/", 1)[-1]
    if isinstance(obj, h5py.Dataset):
        summary = _numeric_summary(obj, max_elements=max_summary_elements)
        attrs = _format_attrs(obj.attrs, max_attrs=max_attrs)
        print(
            f"{indent}{label} dataset shape={obj.shape} dtype={obj.dtype}"
            f"{summary}{attrs}"
        )
    else:
        attrs = _format_attrs(obj.attrs, max_attrs=max_attrs)
        print(f"{indent}{label}/ group{attrs}")


def inspect_hdf5(
    path: str | Path,
    *,
    max_attrs: int = 8,
    max_summary_elements: int = 64,
) -> None:
    """Print groups, datasets, shapes, dtypes, and selected attributes."""

    hdf5_path = Path(path)
    print(f"\n{hdf5_path}")
    with h5py.File(hdf5_path, "r") as handle:
        _visit("", handle, max_attrs=max_attrs, max_summary_elements=max_summary_elements)
        handle.visititems(
            lambda name, obj: _visit(
                name,
                obj,
                max_attrs=max_attrs,
                max_summary_elements=max_summary_elements,
            )
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="HDF5 files to inspect")
    parser.add_argument(
        "--max-summary-elements",
        type=int,
        default=64,
        help="largest numeric dataset to summarize with min/max",
    )
    parser.add_argument(
        "--max-attrs",
        type=int,
        default=8,
        help="maximum attributes to print per object",
    )
    args = parser.parse_args(argv)
    for path in args.paths:
        inspect_hdf5(
            path,
            max_attrs=args.max_attrs,
            max_summary_elements=args.max_summary_elements,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
