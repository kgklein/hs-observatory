# Visualization Export

This folder contains small command-line helpers for inspecting RBF reconstruction
HDF5 products and exporting them to ParaView-readable visualization files.

The RBF fitter remains blind to the simulation cube. Truth-field sampling here
is only for validation and visualization.

## Inspect HDF5

```bash
python visualization/inspect_hdf5.py outputs/rbf_reconstruction.h5
```

The inspector prints the HDF5 group tree, dataset shapes, dtypes, attributes, and
min/max summaries for small numeric datasets.

Current RBF reconstruction files use these important paths:

- `/grid/points`
- `/grid/local_coordinates`
- `/grid/cylinder_mask`
- `/grid/barycenter_path`
- `/reconstruction/B`
- `/reconstruction/Bmag`
- `/samples/trajectory`
- `/samples/time`
- `/samples/spacecraft_id`
- `/validation/B_actual`, when direct-simulation validation has been added

## Add Truth Field

If the reconstruction file does not already contain `/validation/B_actual`, add
it with:

```bash
python visualization/add_truth_to_reconstruction.py \
    --reconstruction outputs/rbf_reconstruction.h5 \
    --generation-config configs/example_generate_simulation_sampled_observatory.yaml \
    --overwrite
```

The script uses the observation path stored in `/metadata` by default. If needed,
provide it explicitly:

```bash
python visualization/add_truth_to_reconstruction.py \
    --reconstruction outputs/rbf_reconstruction.h5 \
    --generation-config configs/example_generate_simulation_sampled_observatory.yaml \
    --observation outputs/synthetic_observations_simulation_sampled_observatory.h5
```

It writes validation arrays:

- `/validation/B_actual`
- `/validation/Bmag_actual`
- `/validation/component_error`
- `/validation/error_magnitude`
- `/validation/error_mag`, kept for backward compatibility
- `/validation/error_percent`
- `/validation/error_percent_capped`

It also ensures `/reconstruction/Bmag` exists.

## Diagnostic Plot Colors

RBF diagnostic slice plots use three visual categories:

- Signed components `Bx`, `By`, and `Bz` use a custom diverging
  blue-grey-red scale centered at zero, with symmetric limits shared between
  the axial and transverse panels for each component row.
- `|B|` uses a sequential scientific colour map. The default is Crameri
  `batlow` when the optional `cmcrameri` package is installed, otherwise
  Matplotlib `magma`.
- Error panels use a separate sequential scientific colour map. The default is
  Crameri `imola` when available, otherwise Matplotlib `inferno`.

`cmcrameri` is optional. Configure plot colours with:

```yaml
plot:
  magnitude_cmap: batlow
  error_cmap: imola

broeren_error_plot:
  error_cmap: imola
```

Allowed magnitude maps include `batlow`, `lajolla`, `magma`, `inferno`,
`plasma`, `viridis`, and `cividis`. Allowed error maps include `imola`,
`lajolla`, `devon`, `magma`, `inferno`, `plasma`, and `YlOrRd`.

Component rows can show either the raw field or fluctuations
`delta B = B - B0`. Raw components show the full magnetic field and can be
dominated by a guide field, especially in `Bz`. Delta mode keeps `|B|` as the
raw magnetic-field magnitude and leaves error rows unchanged.

```yaml
plot:
  component_mode: delta
  background_mode: metadata

reference:
  plot:
    component_mode: delta
    background_mode: metadata
```

When `component_mode: delta`, the plotting code uses one common `B0` vector for
actual and reconstructed plots. With `background_mode: metadata`, it first
looks for background/guide-field metadata such as `B0` or `background_field`;
if none is available, it uses the valid-volume mean of `B_actual` when present,
otherwise the valid-volume mean of `B_reconstructed`. Use
`background_mode: user` with `background_vector: [Bx0, By0, Bz0]` to set `B0`
explicitly.

## Export To ParaView

```bash
python visualization/export_reconstruction_to_paraview.py \
    --input outputs/rbf_reconstruction.h5 \
    --output-dir outputs/paraview/rbf_t000500
```

For the stride sweep products in `output_rbf/`, a concrete example is:

```bash
python visualization/export_reconstruction_to_paraview.py \
    --input output_rbf/rbf_recon_stride04.h5 \
    --output-dir output_rbf/paraview/rbf_recon_stride04
```

That command writes the ParaView files under
`output_rbf/paraview/rbf_recon_stride04/`.

The exporter writes a structured-grid VTK XML file plus legacy ASCII VTK helper
files, all of which ParaView can open directly:

- `reconstruction_structured.vts`: full structured reconstruction grid for
  Contour/Isosurface, Slice, Clip, Threshold, and Volume Rendering filters
- `reconstruction_field.vtk`: point cloud with reconstructed field arrays
- `spacecraft_tracks.vtk`: one polyline per spacecraft
- `spacecraft_points.vtk`: spacecraft locations at the central selected time
- `spacecraft_ids.txt`: index-to-spacecraft-id lookup
- `barycenter_path.vtk`: barycenter polyline
- `reconstruction_cylinder.vtk`: simple cylinder boundary wireframe

If validation fields are present, `reconstruction_field.vtk` also includes:

- `B_actual`
- `Bmag_actual`
- `component_error`
- `error_magnitude`
- `error_percent`

## ParaView First Look

1. Open `reconstruction_structured.vts`.
2. Color by `Bmag_reconstructed`.
3. Open `spacecraft_tracks.vtk` and `spacecraft_points.vtk`.
4. Open `barycenter_path.vtk`.
5. If validation is present, color `reconstruction_structured.vts` by
   `error_magnitude` or `error_percent`.

See `paraview_notes.md` for a few display suggestions.
