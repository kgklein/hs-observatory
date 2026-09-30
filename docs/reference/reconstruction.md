# Magnetic-field reconstruction

| Summary | Details |
| --- | --- |
| Purpose | Reconstruct magnetic fields over a sampled cylindrical volume. |
| Required inputs | Observation HDF5, time window and reconstruction grid; optional simulation reference. |
| Entry points | `hsobs-analyze CONFIG`; `rbf_reconstruction` or `timesync`. |
| Outputs | Reconstruction HDF5 and optional reference/error/sigma plots. |
| Limitations | Single observed MC selection; coverage constraints; kernel/axis restrictions below. |

**In Work...**

## `rbf_reconstruction`

| Summary | Details |
| --- | --- |
| Purpose | Reconstruct B using component-wise RBFs. |
| Required inputs | Observation product, time window and cylindrical grid. |
| Entry points | `analysis.method: rbf_reconstruction`. |
| Outputs | HDF5 and optional fit/reference/error plots. |
| Limitations | Multiquadric kernel; one MC member; grid/coverage

Performs component-wise magnetic-field RBF reconstruction over a cylindrical
sampling volume.

- Config section: `rbf_reconstruction`.
- Output: HDF5 plus optional reconstruction, reference, error, and sigma-selection plots.
- Required timing/window controls: exactly one of `t0` or `t0_index`, plus `delta_t`.
- Required grid controls: `grid.radius`, `grid.n_axial`, and `grid.n_transverse` or `grid.n_radial`.
- Controllable parameters:
  - `data_group`: `truth` or `observed`; aliases like `mc` map to `observed`.
  - `mc_index`: observed Monte Carlo member; `all` is not implemented.
  - `sample_time_stride`, `coordinate_units`, `min_samples`.
  - `kernel`, `sigma`, `smoothing`.
  - `sigma_selection`: `manual`, `rippa_grid`, or `explicit_loocv`.
  - `sigma_min`, `sigma_max`, `sigma_num`, `sigma_grid`, `sigma_initial`.
  - `sigma_subsample_max`, `sigma_random_seed`, `rippa_mode`, `rippa_components`.
  - `sigma_cache`, `use_sigma_cache`, `save_sigma_cache`, `force_recompute_sigma`, `sigma_cache_hash_tolerance`.
  - `grid.axial_padding`, `grid.radius_padding`, `grid.require_all_spacecraft_coverage`.
  - `plot.enabled`, `plot.path`, `plot.dpi`.
  - `plot.magnitude_cmap`, `plot.error_cmap`, `plot.component_mode`, `plot.background_mode`, `plot.background_vector`.
  - `rippa_plot.enabled`, `rippa_plot.path`, `rippa_plot.dpi`.
  - `broeren_error_plot.enabled`, `broeren_error_plot.path`, `broeren_error_plot.dpi`, `broeren_error_plot.error_cmap`, `broeren_error_plot.capped`.
  - `reference.enabled`, `reference.generation_config`, `reference.observation_path`, and nested `reference.plot`.

Supported magnitude colormaps: `batlow`, `lajolla`, `magma`, `inferno`,
`plasma`, `viridis`, `cividis`. Supported error colormaps: `imola`, `lajolla`,
`devon`, `magma`, `inferno`, `plasma`, `YlOrRd`.

## `timesync`

| Summary | Details |
| --- | --- |
| Purpose | Interpolate B onto synchronized planes with IDW. |
| Required inputs | Observation product, time window and plane/grid controls. |
| Entry points | `analysis.method: timesync`. |
| Outputs | HDF5 and optional reconstruction/reference/error plots. |
| Limitations | Reserved flow_velocity axis; one MC member. |

See the [example manifest](../examples/index.md) for prerequisites and verification status,
and [visualization guide](../user-guide/visualization.md) for inspection and plotting.

Reconstructs magnetic-field samples on time-synchronized planes through the
formation, using inverse-distance weighting.

- Config section: `timesync`.
- Output: HDF5 plus optional reconstruction/reference/error plots.
- Required timing/window controls: exactly one of `t0` or `t0_index`, plus `delta_t`.
- Required grid controls: `grid.radius` or `grid.radius_km`, `grid.n_axial`, and `grid.n_transverse` or `grid.n_radial`.
- Controllable parameters:
  - `data_group`: `truth` or `observed`; aliases like `mc` map to `observed`.
  - `mc_index`: observed Monte Carlo member; `all` is not implemented.
  - `sample_time_stride`, `coordinate_units`.
  - `time_interp`: `linear` or `nearest`.
  - `allow_nearest_fallback`.
  - `idw_power`, `idw_eps`.
  - `axis_source`: `barycenter_path` or `user_vector`; `flow_velocity` is reserved.
  - `axis_vector`: required when `axis_source: user_vector`.
  - `min_spacecraft_per_plane`.
  - `grid.axial_padding`, `grid.radius_padding`.
  - `plot.enabled`, `plot.path`, `plot.dpi`.
  - `plot.magnitude_cmap`, `plot.error_cmap`, `plot.component_mode`, `plot.background_mode`, `plot.background_vector`.
  - `reference.enabled`, `reference.generation_config`, `reference.observation_path`.
  - `averaged_error_plot.enabled`, `averaged_error_plot.path`, `averaged_error_plot.dpi`, `averaged_error_plot.error_cmap`, `averaged_error_plot.capped`.

Supported plot colormaps are the same as `rbf_reconstruction` for magnitude
and error plots.


## Implemented scope and authoritative export instructions

The RBF implementation currently supports `kernel: multiquadric`; a kernel
parameter is not a promise of other kernels. Continuous explicit LOOCV optimization
uses SciPy when installed and falls back to a log-grid search otherwise. Timesync's
`flow_velocity` axis is reserved, not implemented.

The repository's [Visualization Export guide](https://github.com/kgklein/hs-observatory/blob/main/visualization/README.md)
is the authority for the HDF5 inspector, adding validation truth, color/background
selection and VTS/VTK export. Its validation helper modifies the reconstruction
file; keep fitting separate from simulation-based validation. See also
[ParaView notes](https://github.com/kgklein/hs-observatory/blob/main/visualization/paraview_notes.md).
