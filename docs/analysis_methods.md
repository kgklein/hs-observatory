# Analysis Methods

## Shock timing

The `shock_timing` method reconstructs a planar shock's propagation normal,
positive normal speed, slowness vector, and reference crossing time from every
observed Monte Carlo realization. It searches `n`, `Vx`, `Vy`, `Vz`, `Bx`,
`By`, and `Bz` without consulting truth, fits significant positive or negative
tanh profiles, and supports independent observable fits, a native-cadence
combined fit, or both.

For each fit it solves every available four-spacecraft combination plus global
OLS and inverse-crossing-time-variance WLS. Nominal positions are interpolated
at measured crossing times. Missing or ambiguous crossings, invalid weights,
insufficient spacecraft, rank deficiency, and poor geometry remain explicit
status codes. Nine valid spacecraft yield 126 separately summarized
tetrahedra.

```bash
hsobs-analyze configs/example_analyze_shock_timing.yaml
```

The HDF5 product contains detection diagnostics and channel masks,
reconstructed parameters and residuals, optional per-realization truth errors,
per-method and per-tetrahedron 16th/median/84th percentiles, paired-success
counts, and failure records. Four PDF/PNG products show profile fits,
reconstruction accuracy, geometry/reliability, and the best/worst tetrahedral
geometry. Accuracy panels report MC RMSE on logarithmic axes, with whiskers
showing the 16th–84th percentile range of absolute realization-level errors;
the four-spacecraft series include the best, median, worst, and minimum-`Q`
Hub-containing tetrahedra. The reliability panel uses a logarithmic axis for
median angular error and compares it separately with design condition number
and configuration size `L`. Its joint-geometry panel plots `L` against the
scaled design condition number with marker area encoding median angular error.
The
tetrahedral view uses the combined fit when available, highlights the
empirical best and worst angular-error-RMSE subsets plus the minimum-`Q`
tetrahedron containing the Hub in the established mesocentric 3-D
projections, and places their size/shape (`L`, `E`, and `P`) diagnostics
alongside it.

Synthetic validation uses the generator plane equation and moving nominal
trajectories. The propagation normal is `sign(n_hat dot V_S) n_hat`, distinct
from the upstream-to-downstream generator normal. Without shock metadata, core
reconstruction and residual diagnostics remain available and truth groups are
omitted. Limitations include the planar constant-speed approximation,
heuristic detection thresholds, approximate fit-derived timing uncertainties,
and no analytical covariance propagation. Correlated tetrahedra are never
pooled as independent Monte Carlo samples.

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
magnetic_gradient
magnetic_gradient_correlation
rbf_reconstruction
shock_timing
static_decorrelation
structure_function
timeseries_fft
timesync
timing-calibrate
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

Builds scale-conditioned PDFs of Elsasser increments between `(time, spacecraft)`
samples interpreted as points in a frozen spatial field, including simultaneous
and different-time pairs.

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

## `magnetic_gradient`

Computes full multi-spacecraft magnetic-field gradients from numerical-
simulation observations and writes a calculation-only HDF5 product. It uses
the general N-point least-squares method from Equation 2 of Broeren & Klein
(2024), summing each of the N(N-1)/2 independent spacecraft pairs once. The
same implementation handles four-spacecraft tetrahedra and all nine
spacecraft.

Gradient tensors use `gradient[field_component, spatial_derivative]`, so
`gradient[0, 1]` is dBx/dy. Positions are in km, magnetic fields and affine
residuals in nT, and gradients, curl, and divergence in nT/km. Geometry uses
the volumetric tensor R, sorted semiaxes `a >= b >= c`, `L = 2a`,
`E = 1 - b/a`, `P = 1 - c/b`, and `Q = chi = sqrt(E^2 + P^2)`.

The product distinguishes four quantities:

- `truth/coarse_*`: simulation-grid gradients averaged in a periodic sphere
  of diameter L around the true barycenter; this is the primary reference.
- `truth/point_*`: the simulation-grid gradient interpolated at the true
  barycenter.
- `true_estimate`: the least-squares result from nominal positions and true
  sampled magnetic fields.
- `measured`: realization-level results from stored observed fields, position
  offsets, and timing offsets after linear synchronization.

Coarse truth defaults to `truth.coarse_average_method: fft`. For each static
configuration size, this applies a periodic spherical top-hat convolution to
the simulation grid and trilinearly samples the convolved field along the
barycenter trajectory. It exactly matches direct grid-node averaging when the
barycenter lies on a grid node; off-grid centers interpolate between those
grid-centered averages. The original per-barycenter grid-node selection
remains available as `truth.coarse_average_method: direct`, including for
time-varying configuration sizes.

The primary solenoidal diagnostic is
`abs(divergence) / curl_magnitude`. A configuration-specific denominator
floor defaults to one percent of the median coarse-truth curl magnitude.
Samples below it retain raw curl/divergence, receive a NaN ratio, and have a
false ratio-validity mask.

Calculation reads the complete simulation definition from the generation
YAML embedded in the observation HDF5 file. `simulation_override` can update
relocated source paths without duplicating the layout and unit mapping.
Configuration selection defaults to the repository's established Q metric:
the smallest tetrahedron with `Q <= 0.6`, the best-shaped tetrahedron in
800--1200 km, the best-shaped tetrahedron at or above 1500 km, and all nine
spacecraft. See `configs/example_magnetic_gradient.yaml` for all controls.
An optional `best_multi` category selects the minimum-Q subset across every
five-, six-, seven-, and eight-spacecraft combination. It excludes all nine
because `all_nine` is stored as its own category. The legacy `large`
tetrahedron selector remains independently available.

```bash
hsobs-analyze configs/example_magnetic_gradient.yaml
hsobs-plot-magnetic-gradient outputs/magnetic_gradients.h5 \
  --output outputs/magnetic_gradients.png \
  --start-time 50 --stop-time 150
```

The plotting command opens only the saved gradient product; it never loads
the observation file or simulation and therefore cannot repeat the gradient
or coarse-volume calculations. By default its fourth row shows
`divergence`; the former `affine_rms_residual`, along with `condition_number`,
`chi`, and `d_ep`, remains selectable with `--row4` (`d_ep` is currently
recorded as unavailable/NaN). The fifth row shows the signed Cartesian
components of curl. Optional `--start-time` and `--stop-time` bounds select
an inclusive interval in analysis-time seconds; the plotted data and
automatic vertical limits both use only that interval.

An alternative compact figure isolates curl reconstruction errors:

```bash
hsobs-plot-magnetic-gradient outputs/magnetic_gradients.h5 \
  --output outputs/magnetic_gradient_error_decomposition.png \
  --layout error_decomposition \
  --start-time 150 --stop-time 200
```

Its first row shows coarse-truth, ideal, and measured median/IQR curl
magnitudes. Its second row retains the absolute curl-vector error against
coarse truth. The third row separates the normalized finite-scale error
`|C_T-C_C| / median_t|C_C|` from the measurement error
`median_MC|C_M-C_T| / median_t|C_C|`; the latter is evaluated separately for
each realization before taking the Monte Carlo median. Both time medians use
only the selected plotting interval. The decomposition requires products
written with `monte_carlo.store_all_realizations: true`.

### Curl-magnitude Wasserstein decomposition

Saved magnetic-gradient products can also be compared with the lightweight
`hsobs-plot-magnetic-gradient-wasserstein` postprocessor. For coarse truth
`C`, the error-free spacecraft estimate `T`, and measured MC member `M`, it
computes empirical one-dimensional distances over their common valid times:

- intrinsic method error: `W1(T, C)`;
- measurement error: `W1(M, T)`;
- total error: `W1(M, C)`.

Raw distances use nT/km. Dimensionless distances divide all three quantities
by the median positive coarse-truth curl magnitude for that configuration.
Each measured distance is calculated separately for every MC member and
reported as its median and interquartile range. A normalized pointwise median
absolute error between `M(t)` and `T(t)` complements the distributional
metric, which intentionally ignores temporal ordering.

The isolated terms obey the Wasserstein triangle inequality but are not an
additive error budget. Run the configured `2xx` comparison with:

```bash
scripts/plot_sim_prod_2xx_magnetic_gradient_wasserstein.sh
```

The YAML controls case labels and writes the figure plus JSON and CSV
diagnostics. Missing case products are warned about and skipped, allowing the
comparison to be rerun as long calculations finish.

### Combined six-panel curl figure

The combined plotter selects the four-spacecraft configuration with the
lowest median `Q` and arranges time-domain and distributional diagnostics in
a 2x3 figure:

- top: signed x component of curl, curl magnitude, and curl-vector error
  against coarse truth;
- bottom: empirical curl-magnitude CDF, normalized Wasserstein distances, and
  normalized pointwise measurement error.

The curl-x panel shows coarse truth, ideal estimate, and the measured Monte
Carlo median/IQR. The configured simulation comparison can be generated with:

```bash
scripts/plot_sim_prod_4xx_magnetic_gradient_combined.sh
```

The defaults in
`analysis/sim-prod-4xx_magnetic-gradient-combined.yaml` select 150--200 s for
the top row and the 5th--95th percentile interval for the bottom-row Monte
Carlo error bars. Command-line options can override the top-row interval,
output, configuration, and DPI.

## `magnetic_gradient_correlation`

This post-processing analysis adapts the Pearson product-moment coefficient
from Equation 6 of Broeren & Klein (2024) to paired time samples of saved
magnetic-gradient components. It reads an existing `magnetic_gradient` HDF5
product and never recomputes gradients from spacecraft or simulation data.
Calculation and plotting are separate:

```bash
scripts/run_sim_prod_400_magnetic_gradient_correlation.sh
scripts/plot_sim_prod_400_magnetic_gradient_correlation.sh
```

The symbol \(\mathcal{R}\) denotes the Pearson coefficient in figures and
documentation. It is distinct from the volumetric tensor \(R_{jk}\) used by
the least-squares gradient estimator. Gradient indices remain
`gradient[field_component, derivative_direction]`; therefore matrix rows are
\(B_x,B_y,B_z\), columns are derivatives with respect to \(x,y,z\), and
`gradient[0, 1]` is \(\partial B_x/\partial y\).

For every tensor or signed-curl component, the product stores:

- ideal-versus-coarse correlation \(\mathcal{R}^{T,C}\);
- measured-versus-ideal correlation
  \(\mathcal{R}^{M,T,(m)}\), evaluated independently for every MC realization;
- optionally measured-versus-coarse correlation
  \(\mathcal{R}^{M,C,(m)}\);
- common valid-sample count and fraction, input standard deviations, validity
  status and failure reason;
- normalized RMSE, plus the slope and intercept of
  `estimate = intercept + slope * reference`.

Measured percentiles are taken only after calculating realization-level
correlations. In particular, the code never correlates the measured median
time series with the ideal series. The default interval is the 5th--95th
percentile. Constant or nearly constant inputs and comparisons with fewer
than `minimum_valid_samples` common samples remain NaN rather than being
replaced by zero.

Pearson correlation measures similarity of temporal variation: positive
values indicate in-phase variation, negative values indicate anticorrelation,
and values near zero indicate little linear correspondence. It is insensitive
to constant offsets and multiplicative amplitude bias, which is why NRMSE and
linear-regression diagnostics are saved alongside it. These are descriptive
time-series metrics; ordinary independent-sample Pearson p-values are not
reported because adjacent samples are autocorrelated.

The correlation selector ranks the configurations already stored in the
source gradient product. It chooses the best and second-best N=4
configurations, the best configuration with \(4<N<9\), and all nine
spacecraft. Ties are broken by lower shape metric, then lower volumetric
tensor condition number, lower characteristic size, and finally
lexicographic spacecraft-index order. A warning is emitted when the two N=4
sizes differ by more than the configured factor, but the selection is not
changed.

The publication heat map is a common-scale 2x4 grid. Its top row shows
\(\mathcal{R}^{T,C}_{jk}\), and its bottom row shows
`median_m(R^{M,T,(m)}_{jk})`; all panels use the fixed range [-1, 1]. The
summary figure compares the median and minimum tensor-component correlation
and the median signed-curl-component correlation. Measured summary
percentiles are calculated from realization-level scalar summaries, not from
the median correlation matrix. The combined layout places that summary below
the heat maps. An additional measured-only figure shows the
measured-versus-ideal heat maps, annotates every valid cell to three decimal
places, and places an M,T-only scalar summary panel with a focused
`[0.7, 1.0]` correlation axis below them. A separate debug figure uses the
stored selected-component time series and regression diagnostics, so all
plotting reads only the standalone correlation product.

The example configuration is
`analysis/sim-prod-400_magnetic-gradient-correlation.yaml`. Moving-block
bootstrap uncertainty is reserved by the configuration schema but is
currently disabled and not implemented.

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

The publication summary plot reads this JSON result directly and keeps sampled
truth separate from the observed Monte Carlo ensemble:

```bash
hsobs-plot-structure-function structure_function.json \
  --p-value 0.9 --quantity z_plus --output sf_summary_p09.png
```

Python callers can use `plot_structure_function_summary_row(...)` with a
supplied row of three axes when assembling multiple p-model rows, or
`plot_structure_function_summary(...)` to save a standalone 1x3 figure. The
default row colors are distinct for `p=0.5`, `0.7`, and `0.9`; `run_color` can
override them.

It can also be produced during analysis by adding a publication subsection:

```yaml
structure_function:
  plot:
    publication:
      enabled: true
      path: outputs/sf_summary_p09.png
      p_value: 0.9
      quantity: z_plus
      max_order: 6
      fit_range: total
      dpi: 300
```

## `timing-calibrate`

Fits eight constant Node timing corrections with a fixed Hub clock using the
existing magnetic-gradient estimator and an augmented linear least-squares solve.

- Config section: `timing_calibrate`.
- Inputs: synthetic timing-only observations with zero Hub offsets.
- Output: HDF5 with all MC fields, divergences, fit diagnostics, and recovery statistics.
- Example: `analysis/sim-cali-timing_timing_calibrate.yaml`.
- Plot saved products with `hsobs-plot-timing-calibrate`.

See [Timing calibration](timing_calibration.md) for signs, weights, configuration,
regeneration instructions, and the output schema.
