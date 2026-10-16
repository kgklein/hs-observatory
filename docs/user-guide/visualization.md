# Visualization and interpretation

Static plotting commands use their saved input products. To animate mission
CDF formation geometry directly, use the separate [Movies](movies.md) workflow
with `hsobs-visualize geometry-movie`.

## Default visualizations

Generation creates three nominal-truth PNGs by default:

- A seven-panel truth figure (* \*_truth.png*), with density and the three velocity components in the left column, the legend in the upper-right, and magnetic components in the remaining right-hand rows. 
- A velocity/Alfvén-speed comparison figure (* \*_velocity_alfven.png*) for `delta U` and
  `delta b = delta[B / sqrt(mu0 m_p n)]` in km/s by default.
  This diagnostic uses instantaneous density interpolated onto the magnetic-field cadence,
  subtracts a full-interval per-spacecraft mean when
  `alfven_fluctuations: true`, and shares each component's y-axis range between
  the matching velocity and magnetic panels.
- A geometry figure (* \*_geometry.png*) with three hub-relative formation projections and an X-Z projection of the spacecraft paths is created for all runs.
- Three other geometry figures can be created when `visualization` is enabled.
  These variants are described in [Geometry summary variants](#geometry-summary-variants)

When any time, position, or measurement error model is enabled, generation also creates overlay figures for one selected Monte Carlo realization:

- Observed `n`, `v`, and `B` in spacecraft colors (* \*_observed_mc000.png*), with each spacecraft's nominal truth in grayscale underneath.
- Observed relative positions and absolute X-Z paths in spacecraft colors (* \*_geometry_observed_mc000.png*), with nominal positions and paths in grayscale underneath.
- Observed `U` and `b` in spacecraft colors (* \*_velocity_alfven_observed_mc000.png*), with nominal truth in grayscale.

The overlay uses `mc_index: 0` by default rather than rendering every member of a potentially large ensemble. Change `mc_index` to inspect another realization, or set `observed_enabled: false` to suppress only the three overlay figures.

The HelioSwarm palette is hardcoded by spacecraft name to match the spacecraft reference colors.
Plot paths default to `<hdf5_stem>_truth.png` and `<hdf5_stem>_geometry.png` beside the HDF5 file. They can be configured or disabled:

```yaml
visualization:
  enabled: true
  alfven_enabled: true
  alfven_fluctuations: true
  observed_enabled: true
  mc_index: 0
  truth_path: outputs/synthetic_observations_truth.png
  geometry_path: outputs/synthetic_observations_geometry.png
  alfven_path: outputs/synthetic_observations_velocity_alfven.png
  observed_path: outputs/synthetic_observations_observed_mc000.png
  observed_geometry_path: outputs/synthetic_observations_geometry_observed_mc000.png
  observed_alfven_path: outputs/synthetic_observations_velocity_alfven_observed_mc000.png
  dpi: 150
```

When paths are omitted, they are derived beside the HDF5 output as `_truth.png`, `_geometry.png`, `_velocity_alfven.png`, `_observed_mcNNN.png`, `_geometry_observed_mcNNN.png`, and `_velocity_alfven_observed_mcNNN.png`. Observed time series are plotted against nominal time.

The geometry-overview workflow also writes a mesocentric 3D GSE configuration
as time-tagged PDF and PNG files under `geometry_overviews/`. Its default solid
lines are the eight Hub-to-Node spokes. Additional solid pairs use
`LABEL-LABEL` syntax (`Hub-N3`, `N2-N5`, etc.) in YAML:

```yaml
visualization:
  configuration_3d_extra_connections: [N2-N5, N4-N7]
```

For a one-run override, repeat `hsobs-generate --3d-connection N2-N5` as
needed. Valid labels are `Hub` and `N1` through `N8`.

## Geometry summary variants

When visualization is enabled, `geometry_overviews/` contains three time-tagged
geometry summaries, each saved as PDF and PNG:

- `*_geometry_summary_*`: the original eight-panel summary.
- `*_geometry_polyhedra_*`: a two-row, three-column view with coordinate
  projections, orbit context, the existing size/shape summary, and elongation
  versus planarity colored by logarithmic size `L` (km). The `L` colorbar is
  independent of the spacecraft-count colorbar.
- `*_geometry_formation_*`: a hub-relative 3D formation spanning the left half,
  with HS spacecraft colors and Hub-to-Node spokes. The lambda and orbit panels
  remain on the right; there is no orbit inset in the 3D panel. A faint convex
  hull is added when SciPy is available and the formation is nondegenerate.

For Python plotting, `hs_observatory.visualization.plot_geometry_summary` accepts
`variant="summary"` (the default), `"polyhedra"`, or `"formation"`.

Coordinate errors affect where the field is sampled; measurement errors act on
the resulting values. The nominal coordinates remain available for downstream
analysis. Observed arrays retain a Monte Carlo dimension even for one realization.

## Velocity-frame controls

The default truth time-series figure uses spacecraft-frame velocity when that truth is available, without changing `/truth/v` in the file. Select the plotted frame in YAML or override it for one command:

```yaml
visualization:
  truth_velocity_frame: spacecraft  # spacecraft (default) or plasma
```

```bash
hsobs-generate configs/example_generate_plume_single_mode.yaml \
  --truth-velocity-frame plasma
```

The effective plotted frame is stored as `truth_visualization_velocity_frame` in HDF5 metadata.

## Plot saved analysis products

| Saved product / task | Installed command | Interpretation |
| --- | --- | --- |
| Structure functions | `hsobs-plot-structure-function PRODUCT --p-value P --quantity z_plus --output FIGURE` | [Moments, scaling fits and ensemble plots](../reference/statistical-methods.md#structure_function) |
| Structure-function overlays | `hsobs-plot-structure-function-overlay CONFIG` | Lightweight plotting YAML; inspect its case inputs first. |
| Local structure-function slopes | `hsobs-plot-structure-function-overlay CONFIG --local-slopes` | [Local scaling, per-member MC intervals and cascade context](../reference/statistical-methods.md#local-logarithmic-scaling-diagnostic) |
| Direct 3D spatial structure functions | `hsobs-plot-spatial-structure-function PRODUCT` | [Raw Malara field, exact separations and batch sampling diagnostics](../reference/statistical-methods.md#direct-3d-spatial-structure-functions) |
| Static increment PDFs | `hsobs-plot-increment-pdf --result P=HDF5 --output FIGURE` (repeat cases) | [Frozen-field PDF definitions](../reference/statistical-methods.md#static-elsasser-increment-pdfs) |
| Increment-PDF distance | `hsobs-plot-increment-pdf-wasserstein CONFIG` | Saved-product comparison; no new observation generation. |
| Magnetic gradients | `hsobs-plot-magnetic-gradient PRODUCT --output FIGURE` | [Standard and error-decomposition layouts](../reference/magnetic-gradients.md#magnetic_gradient) |
| Gradient distances | `hsobs-plot-magnetic-gradient-wasserstein CONFIG` | [Nonadditive Wasserstein terms](../reference/magnetic-gradients.md#curl-magnitude-wasserstein-decomposition) |
| Combined gradient figure | `hsobs-plot-magnetic-gradient-combined CONFIG` | [Six-panel layout](../reference/magnetic-gradients.md#combined-six-panel-curl-figure) |
| Gradient correlation | `hsobs-plot-magnetic-gradient-correlation CONFIG` | [Pearson validity and regression diagnostics](../reference/magnetic-gradients.md#magnetic_gradient_correlation) |
| Timing calibration | `hsobs-plot-timing-calibrate PRODUCT --output FIGURE` | [Signs, objectives and research limitations](../reference/timing-calibration.md) |
| Paper Figure 2 | `hsobs-plot-figure2 --malara HDF5 --simulation HDF5 --output FIGURE --input-units repo` | Use repo units for native products; default without this flag is SI. |

Uppercase arguments are placeholders, not supplied files. Use `--help` and the
[CLI index](../reference/configuration.md#installed-commands) for argument details.
Standard statistics, shock timing and reconstruction can produce figures during
analysis; gradient and timing commands support separate saved-product plotting.

### Local structure-function slopes

From the repository root, use saved structure-function JSON products to inspect
how the scaling exponent changes with separation:

```bash
scripts/plot_malara_prod_combined_sf_publication.sh \
  --local-slopes --show-cascade-scales
```

This production example reads runs 400–402 through
`analysis/malara-prod-combined_sf-publication.yaml`. It requires those saved
JSONs; it does not generate fields, observations, or structure functions.
The equivalent installed command is:

```bash
hsobs-plot-structure-function-overlay \
  analysis/malara-prod-combined_sf-publication.yaml \
  --local-slopes --show-cascade-scales
```

The default three-point estimator leaves the first and last bins undefined.
Add `--local-slope-method adjacent` for consecutive-bin slopes at geometric
midpoints. Use `--output-prefix outputs/my_local_slopes` to choose another
prefix, or `--no-p-model` to hide analytic reference lines. Local mode uses the
full saved separation range and rejects explicit global fitting options and
`--output`.

The example writes these files under `outputs/`:

- `malara-prod-combined_sf_publication_local_slopes.png` and `.pdf`: six orders,
  one row per intermittency parameter.
- `malara-prod-combined_sf_publication_local_slopes_focused.png` and `.pdf`:
  orders 2, 3, 4, and 6, comparing the three intermittency parameters.
- `malara-prod-combined_sf_publication_local_slopes.h5`: local slopes, per-member
  MC arrays, percentiles, validity masks, and provenance.

The figures distinguish sampled truth, MC medians and 5th–95th percentile bands,
and theoretical exponents. A theory crossing alone is not evidence of a scaling
plateau. See the [method reference](../reference/statistical-methods.md#local-logarithmic-scaling-diagnostic)
for estimators, cascade annotations, and interpretation limits.

For the numerical findings, plot snapshots, direct-3D validation status and
planned convergence checks, follow
[Research Notes: Intermittency recovery](../research-notes/intermittency-recovery.md).

## Observed geometry display

The generation overlay helper receives nominal positions plus stored position
offsets. Its displayed geometry does not encode the full clock-shift trajectory
displacement used when sampling the field. Do not interpret it as a complete
reconstruction of sampled coordinates; see [coordinate conventions](../reference/data-products.md#coordinate-use-by-analysis).

## Reconstruction and ParaView

**3D Rendering is a work in progress.**

The [Visualization Export guide](https://github.com/kgklein/hs-observatory/blob/main/visualization/README.md)
is authoritative for HDF5 inspection, adding validation truth, color/background
settings, Figure 2 and VTS/VTK export.
Adding truth modifies the reconstruction file;
export/inspection are separate from the fitter.
Follow the
[ParaView display notes](https://github.com/kgklein/hs-observatory/blob/main/visualization/paraview_notes.md)
for field rendering and the `cylinder_mask > 0.5` threshold.
