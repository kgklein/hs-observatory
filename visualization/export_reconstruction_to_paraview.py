#!/usr/bin/env python3
"""Export an RBF reconstruction HDF5 product to ParaView-readable VTK files."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


XML_INDENT = "  "


def _clean_name(name: str) -> str:
    return "".join(char if char.isalnum() or char == "_" else "_" for char in name)


def _read_first(handle: h5py.File, paths: tuple[str, ...]) -> np.ndarray | None:
    for path in paths:
        if path in handle:
            return np.asarray(handle[path], dtype=float)
    return None


def _scalar_lines(name: str, values: np.ndarray) -> list[str]:
    scalar = np.asarray(values, dtype=float).reshape(-1)
    lines = [f"SCALARS {_clean_name(name)} float 1", "LOOKUP_TABLE default"]
    lines.extend(f"{float(value):.9g}" for value in scalar)
    return lines


def _vector_lines(name: str, values: np.ndarray) -> list[str]:
    vectors = np.asarray(values, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    lines = [f"VECTORS {_clean_name(name)} float"]
    lines.extend(
        f"{float(vector[0]):.9g} {float(vector[1]):.9g} {float(vector[2]):.9g}"
        for vector in vectors
    )
    return lines


def _as_ascii(values: np.ndarray) -> str:
    array = np.asarray(values, dtype=float).reshape(-1)
    return " ".join("nan" if not np.isfinite(value) else f"{float(value):.9g}" for value in array)


def _xml_data_array(name: str, values: np.ndarray, *, components: int = 1) -> list[str]:
    clean = _clean_name(name)
    component_text = "" if components == 1 else f' NumberOfComponents="{components}"'
    return [
        f'{XML_INDENT * 4}<DataArray type="Float64" Name="{clean}"'
        f'{component_text} format="ascii">',
        f"{XML_INDENT * 5}{_as_ascii(values)}",
        f"{XML_INDENT * 4}</DataArray>",
    ]


def _structured_points(handle: h5py.File) -> np.ndarray:
    if "grid/full_points" in handle:
        full_points = np.asarray(handle["grid/full_points"], dtype=float)
        if full_points.ndim == 4 and full_points.shape[-1] == 3:
            return full_points
        raise ValueError("/grid/full_points must have shape (n_axial, n_u, n_v, 3)")
    required = ("grid/center", "grid/basis_vectors", "grid/axial_values", "grid/transverse_values")
    if any(path not in handle for path in required):
        raise KeyError(
            "Structured export requires /grid/full_points or center/basis/axis metadata"
        )
    center = np.asarray(handle["grid/center"], dtype=float)
    basis = np.asarray(handle["grid/basis_vectors"], dtype=float)
    axial_values = np.asarray(handle["grid/axial_values"], dtype=float)
    transverse_values = np.asarray(handle["grid/transverse_values"], dtype=float)
    e_parallel, e1, e2 = basis
    axial, transverse_u, transverse_v = np.meshgrid(
        axial_values, transverse_values, transverse_values, indexing="ij"
    )
    return (
        center
        + axial[..., None] * e_parallel
        + transverse_u[..., None] * e1
        + transverse_v[..., None] * e2
    )


def _full_from_inside(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    components: int = 1,
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if components == 1:
        if array.shape == mask.shape:
            return array
        flat = array.reshape(-1)
        if flat.shape[0] != int(np.count_nonzero(mask)):
            raise ValueError(f"{name} length does not match cylinder mask")
        full = np.full(mask.shape, np.nan, dtype=float)
        full[mask] = flat
        return full
    if array.shape == mask.shape + (components,):
        return array
    if array.ndim != 2 or array.shape[1] != components:
        raise ValueError(f"{name} must have shape (N, {components})")
    if array.shape[0] != int(np.count_nonzero(mask)):
        raise ValueError(f"{name} length does not match cylinder mask")
    full = np.full(mask.shape + (components,), np.nan, dtype=float)
    full[mask] = array
    return full


def _add_vector_components(
    scalar_data: dict[str, np.ndarray],
    vector: np.ndarray,
    *,
    suffix: str,
) -> None:
    scalar_data[f"Bx_{suffix}"] = vector[..., 0]
    scalar_data[f"By_{suffix}"] = vector[..., 1]
    scalar_data[f"Bz_{suffix}"] = vector[..., 2]


def _structured_arrays(
    handle: h5py.File,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    if "grid/cylinder_mask" not in handle:
        raise KeyError("Structured export requires /grid/cylinder_mask")
    if "reconstruction/B" not in handle:
        raise KeyError("Structured export requires /reconstruction/B")
    points = _structured_points(handle)
    mask = np.asarray(handle["grid/cylinder_mask"], dtype=bool)
    if points.shape[:3] != mask.shape:
        raise ValueError("/grid/full_points shape does not match /grid/cylinder_mask")
    reconstructed = _full_from_inside(
        np.asarray(handle["reconstruction/B"], dtype=float),
        mask,
        components=3,
        name="/reconstruction/B",
    )
    vector_data: dict[str, np.ndarray] = {"B_reconstructed": reconstructed}
    scalar_data: dict[str, np.ndarray] = {
        "Bmag_reconstructed": _full_from_inside(
            _read_first(handle, ("reconstruction/Bmag",))
            if "reconstruction/Bmag" in handle
            else np.linalg.norm(np.asarray(handle["reconstruction/B"], dtype=float), axis=-1),
            mask,
            name="/reconstruction/Bmag",
        ),
        "cylinder_mask": mask.astype(float),
    }
    _add_vector_components(scalar_data, reconstructed, suffix="reconstructed")
    actual = _read_first(handle, ("validation/B_actual", "validation/B_true"))
    if actual is not None:
        actual_full = _full_from_inside(
            actual, mask, components=3, name="/validation/B_actual"
        )
        vector_data["B_actual"] = actual_full
        _add_vector_components(scalar_data, actual_full, suffix="actual")
        bmag_actual = _read_first(handle, ("validation/Bmag_actual",))
        scalar_data["Bmag_actual"] = _full_from_inside(
            np.linalg.norm(actual, axis=-1) if bmag_actual is None else bmag_actual,
            mask,
            name="/validation/Bmag_actual",
        )
        component_error = _read_first(handle, ("validation/component_error",))
        if component_error is None:
            component_error = np.asarray(handle["reconstruction/B"], dtype=float) - actual
        component_full = _full_from_inside(
            component_error,
            mask,
            components=3,
            name="/validation/component_error",
        )
        vector_data["component_error"] = component_full
        scalar_data["component_error_x"] = component_full[..., 0]
        scalar_data["component_error_y"] = component_full[..., 1]
        scalar_data["component_error_z"] = component_full[..., 2]
    error = _read_first(handle, ("validation/error_magnitude", "validation/error_mag"))
    if error is not None:
        scalar_data["error_magnitude"] = _full_from_inside(
            error, mask, name="/validation/error_magnitude"
        )
    error_percent = _read_first(
        handle,
        ("validation/error_percent_capped", "validation/error_percent"),
    )
    if error_percent is not None:
        scalar_data["error_percent"] = _full_from_inside(
            error_percent, mask, name="/validation/error_percent"
        )
    return points, scalar_data, vector_data


def _write_structured_grid(
    path: str | Path,
    points: np.ndarray,
    *,
    scalar_data: dict[str, np.ndarray],
    vector_data: dict[str, np.ndarray],
) -> Path:
    output = Path(path)
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError("Structured grid points must have shape (n_axial, n_u, n_v, 3)")
    n_axial, n_u, n_v, _ = points.shape
    extent = f"0 {n_v - 1} 0 {n_u - 1} 0 {n_axial - 1}"
    text: list[str] = [
        '<?xml version="1.0"?>',
        '<VTKFile type="StructuredGrid" version="0.1" byte_order="LittleEndian">',
        f'{XML_INDENT}<StructuredGrid WholeExtent="{extent}">',
        f'{XML_INDENT * 2}<Piece Extent="{extent}">',
        f"{XML_INDENT * 3}<PointData>",
    ]
    for name, values in vector_data.items():
        if np.asarray(values).shape == points.shape:
            text.extend(_xml_data_array(name, values, components=3))
    for name, values in scalar_data.items():
        if np.asarray(values).shape == points.shape[:3]:
            text.extend(_xml_data_array(name, values))
    text.extend(
        [
            f"{XML_INDENT * 3}</PointData>",
            f"{XML_INDENT * 3}<Points>",
            *_xml_data_array("Points", points, components=3),
            f"{XML_INDENT * 3}</Points>",
            f"{XML_INDENT * 2}</Piece>",
            f"{XML_INDENT}</StructuredGrid>",
            "</VTKFile>",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(text) + "\n", encoding="utf-8")
    return output


def _write_polydata(
    path: str | Path,
    points: np.ndarray,
    *,
    vertices: bool = False,
    lines: list[list[int]] | None = None,
    scalar_data: dict[str, np.ndarray] | None = None,
    vector_data: dict[str, np.ndarray] | None = None,
) -> Path:
    output = Path(path)
    point_array = np.asarray(points, dtype=float)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError("VTK points must have shape (N, 3)")
    scalar_data = scalar_data or {}
    vector_data = vector_data or {}
    lines = lines or []
    text: list[str] = [
        "# vtk DataFile Version 3.0",
        output.stem,
        "ASCII",
        "DATASET POLYDATA",
        f"POINTS {point_array.shape[0]} float",
    ]
    text.extend(
        f"{float(point[0]):.9g} {float(point[1]):.9g} {float(point[2]):.9g}"
        for point in point_array
    )
    if vertices:
        text.append(f"VERTICES {point_array.shape[0]} {2 * point_array.shape[0]}")
        text.extend(f"1 {index}" for index in range(point_array.shape[0]))
    if lines:
        total_size = sum(len(line) + 1 for line in lines)
        text.append(f"LINES {len(lines)} {total_size}")
        text.extend(
            f"{len(line)} {' '.join(str(index) for index in line)}" for line in lines
        )
    if scalar_data or vector_data:
        text.append(f"POINT_DATA {point_array.shape[0]}")
        for name, values in vector_data.items():
            if np.asarray(values).shape[0] == point_array.shape[0]:
                text.extend(_vector_lines(name, values))
        for name, values in scalar_data.items():
            if np.asarray(values).reshape(-1).shape[0] == point_array.shape[0]:
                text.extend(_scalar_lines(name, values))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(text) + "\n", encoding="utf-8")
    return output


def _field_arrays(
    handle: h5py.File,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    if "grid/points" not in handle:
        raise KeyError("Reconstruction HDF5 is missing /grid/points")
    if "reconstruction/B" not in handle:
        raise KeyError("Reconstruction HDF5 is missing /reconstruction/B")
    points = np.asarray(handle["grid/points"], dtype=float)
    reconstructed = np.asarray(handle["reconstruction/B"], dtype=float)
    if points.shape != reconstructed.shape:
        raise ValueError("/grid/points and /reconstruction/B must both have shape (N, 3)")
    vector_data: dict[str, np.ndarray] = {"B_reconstructed": reconstructed}
    scalar_data: dict[str, np.ndarray] = {
        "Bmag_reconstructed": _read_first(handle, ("reconstruction/Bmag",))
        if "reconstruction/Bmag" in handle
        else np.linalg.norm(reconstructed, axis=-1)
    }
    _add_vector_components(scalar_data, reconstructed, suffix="reconstructed")
    local_coordinates = _read_first(handle, ("grid/local_coordinates",))
    if local_coordinates is not None and local_coordinates.shape == points.shape:
        vector_data["local_axial_u_v"] = local_coordinates
    actual = _read_first(handle, ("validation/B_actual", "validation/B_true"))
    if actual is not None:
        vector_data["B_actual"] = actual
        _add_vector_components(scalar_data, actual, suffix="actual")
        scalar_data["Bmag_actual"] = _read_first(
            handle, ("validation/Bmag_actual",)
        )
        if scalar_data["Bmag_actual"] is None:
            scalar_data["Bmag_actual"] = np.linalg.norm(actual, axis=-1)
        component_error = _read_first(handle, ("validation/component_error",))
        if component_error is None:
            component_error = reconstructed - actual
        vector_data["component_error"] = component_error
    error = _read_first(handle, ("validation/error_magnitude", "validation/error_mag"))
    if error is not None:
        scalar_data["error_magnitude"] = error
    error_percent = _read_first(
        handle,
        ("validation/error_percent_capped", "validation/error_percent"),
    )
    if error_percent is not None:
        scalar_data["error_percent"] = error_percent
    return points, scalar_data, vector_data


def _spacecraft_ids(handle: h5py.File, n_spacecraft: int) -> list[str]:
    if "samples/spacecraft_id" not in handle:
        return [f"S{index}" for index in range(n_spacecraft)]
    return [
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in handle["samples/spacecraft_id"][...]
    ]


def _export_tracks(handle: h5py.File, output_dir: Path) -> list[Path]:
    if "samples/trajectory" not in handle:
        return []
    trajectory = np.asarray(handle["samples/trajectory"], dtype=float)
    if trajectory.ndim != 3 or trajectory.shape[-1] != 3:
        raise ValueError("/samples/trajectory must have shape (time, spacecraft, 3)")
    n_time, n_spacecraft, _ = trajectory.shape
    ids = _spacecraft_ids(handle, n_spacecraft)
    points = trajectory.swapaxes(0, 1).reshape(-1, 3)
    lines: list[list[int]] = []
    spacecraft_index = np.empty(n_spacecraft * n_time, dtype=float)
    for sc_index in range(n_spacecraft):
        start = sc_index * n_time
        lines.append(list(range(start, start + n_time)))
        spacecraft_index[start : start + n_time] = sc_index
    outputs = [
        _write_polydata(
            output_dir / "spacecraft_tracks.vtk",
            points,
            lines=lines,
            scalar_data={"spacecraft_index": spacecraft_index},
        )
    ]
    central = n_time // 2
    central_points = trajectory[central]
    outputs.append(
        _write_polydata(
            output_dir / "spacecraft_points.vtk",
            central_points,
            vertices=True,
            scalar_data={"spacecraft_index": np.arange(n_spacecraft, dtype=float)},
        )
    )
    (output_dir / "spacecraft_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    return outputs


def _export_barycenter(handle: h5py.File, output_dir: Path) -> list[Path]:
    barycenter = _read_first(handle, ("grid/barycenter_path", "samples/barycenter_path"))
    if barycenter is None:
        return []
    if barycenter.ndim != 2 or barycenter.shape[1] != 3:
        raise ValueError("barycenter path must have shape (N, 3)")
    line = list(range(barycenter.shape[0]))
    return [_write_polydata(output_dir / "barycenter_path.vtk", barycenter, lines=[line])]


def _export_cylinder(handle: h5py.File, output_dir: Path) -> list[Path]:
    required = ("grid/center", "grid/basis_vectors", "grid/axial_values")
    if any(path not in handle for path in required):
        return []
    center = np.asarray(handle["grid/center"], dtype=float)
    basis = np.asarray(handle["grid/basis_vectors"], dtype=float)
    axial_values = np.asarray(handle["grid/axial_values"], dtype=float)
    radius = float(handle["metadata/grid"].attrs["radius"]) if "metadata/grid" in handle else None
    if radius is None:
        local = _read_first(handle, ("grid/local_coordinates",))
        if local is None:
            return []
        radius = float(np.nanmax(np.linalg.norm(local[:, 1:3], axis=-1)))
    e_parallel, e1, e2 = basis
    n_theta = 72
    angles = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    points: list[np.ndarray] = []
    lines: list[list[int]] = []
    for axial in (float(axial_values[0]), float(axial_values[-1])):
        start = len(points)
        for angle in angles:
            points.append(
                center
                + axial * e_parallel
                + radius * np.cos(angle) * e1
                + radius * np.sin(angle) * e2
            )
        lines.append([start + index for index in range(n_theta)] + [start])
    for index in range(0, n_theta, 9):
        lines.append([index, n_theta + index])
    return [
        _write_polydata(
            output_dir / "reconstruction_cylinder.vtk",
            np.asarray(points),
            lines=lines,
        )
    ]


def export_reconstruction_to_paraview(
    input_h5: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """Write ParaView-readable VTK products from a reconstruction HDF5 file."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    products: list[Path] = []
    with h5py.File(input_h5, "r") as handle:
        structured_points, structured_scalars, structured_vectors = _structured_arrays(handle)
        products.append(
            _write_structured_grid(
                output / "reconstruction_structured.vts",
                structured_points,
                scalar_data=structured_scalars,
                vector_data=structured_vectors,
            )
        )
        points, scalar_data, vector_data = _field_arrays(handle)
        products.append(
            _write_polydata(
                output / "reconstruction_field.vtk",
                points,
                vertices=True,
                scalar_data=scalar_data,
                vector_data=vector_data,
            )
        )
        products.extend(_export_tracks(handle, output))
        products.extend(_export_barycenter(handle, output))
        products.extend(_export_cylinder(handle, output))
    return products


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="RBF reconstruction HDF5")
    parser.add_argument("--output-dir", required=True, help="directory for VTK exports")
    args = parser.parse_args(argv)
    products = export_reconstruction_to_paraview(args.input, args.output_dir)
    for product in products:
        print(product)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
