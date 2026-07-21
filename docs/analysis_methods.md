# Analysis Methods

`hsobs-analyze` reads a YAML file with top-level `input`, `analysis`, and
`output` sections. The selected method is `analysis.method`; method-specific
controls usually live in a top-level section named after the method or its
documented config section.

```yaml
input:
  path: outputs/synthetic_observations.h5
analysis:
  method: diagnostics
output:
  path: outputs/diagnostics.json
```

The currently registered methods are:

```text
diagnostics
elsasser
elsasser_increment_pdf_static
rbf_reconstruction
static_decorrelation
structure_function
timeseries_fft
timesync
```

## `diagnostics`

Basic JSON sanity report for generated observations.

- Config section: none beyond `analysis.method`.
- Inputs: generated HDF5 observations.
- Output: JSON by default.
- Computes observed means/stds, residual means/stds when truth is available, and spacecraft-separation summaries for `n`, `v`, and `B`.

## `elsasser`

Computes Elsasser variables, using `z_plus = delta_v + delta_b` and
`z_minus = delta_v - delta_b`.

- Config section: `elsasser`.
- Output: HDF5 plus optional diagnostic PNG.
- Controllable parameters:
  - `data_group`: `truth` or `observed`; default `observed`.
  - `mc_index`: observed Monte Carlo member; default `0`; `all` is not implemented.
  - `velocity.frame`: `auto`, `plasma`, or `spacecraft`.
  - `velocity.spacecraft_correction`: `auto` or `apply`.
  - `density_normalization.method`: `instantaneous` or `global_per_spacecraft`.
  - `density_normalization.species`: currently `proton` only.
  - `mean_subtraction.method`: currently `global_per_spacecraft` only.
  - `timebase.method`: currently `slower_of_n_and_v` only.
  - `timebase.interpolation`: currently `linear` only.
  - `plot.enabled`, `plot.path`, `plot.dpi`.

Example: `configs/example_elsasser.yaml`.

## `elsasser_increment_pdf_static`

Builds scale-conditioned PDFs of simultaneous Elsasser increments across
spacecraft pairs.

- Config section: `elsasser_increment_pdf_static`.
- Output: HDF5 plus optional PDF diagnostic plots.
- Controllable parameters:
  - `datasets`: list of named datasets with `data_group` and optional `mc_index`.
  - `elsasser`: nested settings passed to the `elsasser` calculation.
  - `increments.plotted_field`: `z_plus` or `z_minus`.
  - `increments.component`: `perpendicular` or `full_vector`.
  - `mean_field.method`: currently `local_pairwise` only.
  - `mean_field.fallback`: currently `global_mean_B` only.
  - `mean_field.min_norm`.
  - `scale_bins.separation`: `perpendicular` or `full`.
  - `scale_bins.binning`: `log` or `linear`.
  - `scale_bins.n_bins`, `lambda_min`, `lambda_max`, `min_samples_per_bin`.
  - `pdf_bins.n_bins`, `x_min`, `x_max`, `x_max_percentile`.
  - `pair_sampling.mode`: `all` or `random`.
  - `pair_sampling.n_pairs`, `max_pairs`, `chunk_size`, `random_seed`.
  - `alignment.compute`.
  - `output_samples.save`, `output_samples.max_samples`.
  - `plot.enabled`, `plot.directory`, `plot.yscale`, `plot.xscale`, `plot.dpi`.

Examples: `configs/example_elsasser_increment_pdf_static.yaml`,
`configs/malara_elsasser_increment_pdf_static.yaml`.

## `static_decorrelation`

Computes simultaneous pairwise correlations and second-order pair-difference
structure functions versus spacecraft separation.

- Config section: `decorrelation`.
- Output: HDF5 plus optional diagnostic PNG.
- Controllable parameters:
  - `variables`: any nonempty subset of `n`, `v`, `B`.
  - `data_group`: `truth` or `observed`; default `observed`.
  - `mc_index`: observed Monte Carlo member; default `0`; `all` is not implemented.
  - `mean.method`: `per_time` or `global`.
  - `separation.units`: currently `km` only.
  - `separation.binning`: `log` or `linear`.
  - `separation.n_bins`, `r_min`, `r_max`.
  - `correlation.normalize`.
  - `correlation.component_correlations`.
  - `structure_function.compute`; `structure_function.order` is currently `2`.
  - `visualization.enabled`, `visualization.path`, `visualization.dpi`.

Example: `configs/example_static_decorrelation.yaml`.

## `timeseries_fft`

Computes one-sided real-FFT power spectral densities independently for each
spacecraft time series.

- Config section: `fft`.
- Output: HDF5 plus optional diagnostic PNGs.
- Controllable parameters:
  - `variables`: any nonempty subset of `n`, `v`, `B`.
  - `data_group`: `truth` or `observed`; default `observed`.
  - `mc_index`: observed Monte Carlo member; default `0`; `all` is not implemented.
  - `detrend`: `mean` or `none`.
  - `window`: `hann` or `none`.
  - `normalization`: currently `psd` only.
  - `plot.enabled`, `plot.path`, `plot.truth_enabled`, `plot.truth_path`.
  - `plot.loglog`, `plot.overlay_spacecraft`, `plot.show_components`, `plot.dpi`.

Examples: `configs/example_timeseries_fft.yaml`,
`configs/malara_timeseries_fft.yaml`.

## `rbf_reconstruction`

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

## `structure_function`

Binned same-time pairwise Elsasser vector-increment structure functions.

- Config section: `structure_function`.
- Output: JSON by default, plus optional diagnostic PNG.
- Computes `z_plus` and `z_minus` through the shared Elsasser helper, then bins
  vector increment magnitudes by spacecraft-pair separation in the repository's
  native km coordinate units.
- Observed Monte Carlo members are processed independently; no ensemble average
  is performed. Truth is computed separately when present.
- Current controls:
  - `min_order`, `max_order`: integer order range; defaults `1` through `6`.
  - `n_lambda_bins`: number of data-driven logarithmic separation bins.
  - `lambda_bins_km`: optional explicit separation bin edges in km.
  - `scale_ranges_km`: optional named fit ranges, for example `total: [10, 10000]`.
  - `min_fit_bins`: minimum populated bins for log-log exponent fits; default `3`.
  - `elsasser`: nested settings passed to the `elsasser` calculation.
  - `plot.enabled`, `plot.path`, `plot.field`, `plot.member`,
    `plot.data_group`, `plot.normalized_exponents`, `plot.dpi`.
- Returned products include `S_n[member, order, bin]`, `counts[member, bin]`,
  `flatness = S4 / S2^2` when orders 2 and 4 are available, and raw plus
  order-2-normalized scaling exponents for each named scale range.
