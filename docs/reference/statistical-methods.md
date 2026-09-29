# Statistical analyses

| Summary | Details |
| --- | --- |
| Purpose | Compute derived fields, spatial statistics and single-spacecraft spectra. |
| Required inputs | Observation HDF5 with the variables, coordinates and metadata required below. |
| Entry points | `hsobs-analyze CONFIG`; selectors and configuration sections below. |
| Outputs | JSON or HDF5 plus optional diagnostic/publication figures. |
| Limitations | Native sampling, ensemble handling and frozen-field assumptions differ by method. |

See the [example manifest](../examples/index.md) for prerequisites and verification status,
and [visualization guide](../user-guide/visualization.md) for inspection and plotting.
## `diagnostics`

| Summary | Details |
| --- | --- |
| Purpose | Sanity-check generated data. |
| Required inputs | Generated observation HDF5 including truth. |
| Entry points | `analysis.method: diagnostics`. |
| Outputs | JSON statistics and separation summaries. |
| Limitations | Reader currently requires truth. |

See the [example manifest](../examples/index.md) for prerequisites and verification status,
and [visualization guide](../user-guide/visualization.md) for inspection and plotting.

Basic JSON sanity report for generated observations.

- Config section: none beyond `analysis.method`.
- Inputs: generated HDF5 observations.
- Output: JSON by default.
- Computes observed means/stds, residual means/stds against generated truth, and spacecraft-separation summaries for `n`, `v`, and `B`.

## Elsasser derived quantities

| Summary | Details |
| --- | --- |
| Purpose | Derive z-plus/z-minus and Alfvén velocity. |
| Required inputs | n, v, B, time/frame metadata; one observed member or truth. |
| Entry points | `analysis.method: elsasser`; section `elsasser`. |
| Outputs | HDF5 and optional six-panel PNG. |
| Limitations | Proton-only density; linear cadence alignment; no anti-alias filtering. |

See the [example manifest](../examples/index.md) for prerequisites and verification status,
and [visualization guide](../user-guide/visualization.md) for inspection and plotting.

The selectable `elsasser` analysis computes the convention

```text
z_plus  = delta_v + delta_b
z_minus = delta_v - delta_b
```

where `b = B / sqrt(mu0 n m_p)` is magnetic field in Alfvén-speed units and
`delta_v` and `delta_b` subtract a separate full-time mean for every
spacecraft. Inputs use the package conventions `n` in cm^-3, `B` in nT, and
`v` in km/s; `b` and both Elsasser fields are written in km/s. This first
implementation assumes proton mass density only. Both density normalization
and mean subtraction default to `global_per_spacecraft`; density normalization
may instead be `instantaneous`.

Run the observed-data example with:

```bash
hsobs-analyze configs/example_elsasser.yaml
```

Set `elsasser.data_group: truth` to use plasma-frame `/truth/v`, `/truth/n`,
and `/truth/B`. Observed analysis selects one `mc_index`. Its velocity frame
can be explicit or inferred from generated-file metadata; spacecraft-frame
velocity is restored to the plasma frame using stored trajectory information.
The method chooses the slower native `n`/`v` grid and linearly interpolates
the other variables, including higher-cadence `B`, onto it. Anti-alias
filtering for large cadence ratios is a future extension.

The output layout is:

```text
/metadata                         convention, selection, units, and methods
/time                             (n_time_out,)
/spacecraft/id                    (n_spacecraft,)
/elsasser/z_plus                  (n_time_out, n_spacecraft, 3)
/elsasser/z_minus                 (n_time_out, n_spacecraft, 3)
/alfven/b                         (n_time_out, n_spacecraft, 3)
/alfven/delta_b                   (n_time_out, n_spacecraft, 3)
/velocity/v_plasma                (n_time_out, n_spacecraft, 3)
/velocity/delta_v                 (n_time_out, n_spacecraft, 3)
/density/n                        (n_time_out, n_spacecraft)
/density/n_normalization          (n_spacecraft,) for the default method,
                                   otherwise (n_time_out, n_spacecraft)
```

When plotting is enabled, the diagnostic is a three-row by two-column grid:
x/y/z components by `z_plus`/`z_minus`, with all spacecraft overlaid.

The physical calculation lives in
`hs_observatory.analysis.derived.elsasser`, independently of the HDF5 wrapper.
Future analyses should import that common utility rather than implementing a
second Elsasser convention.

## Elsasser configuration


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

## Static Elsasser-increment PDFs

| Summary | Details |
| --- | --- |
| Purpose | Estimate scale-conditioned increment PDFs. |
| Required inputs | Frozen-field n/v/B samples and coordinates. |
| Entry points | `analysis.method: elsasser_increment_pdf_static`. |
| Outputs | HDF5 moments, histograms, optional samples and plots. |
| Limitations | Frozen-field assumption; no dynamical interpretation; missing higher diagnostics below. |

See the [example manifest](../examples/index.md) for prerequisites and verification status,
and [visualization guide](../user-guide/visualization.md) for inspection and plotting.

`elsasser_increment_pdf_static` creates scale-conditioned PDFs inspired by
Mallet et al. (2015), Fig. 1:

```bash
hsobs-analyze configs/example_elsasser_increment_pdf_static.yaml
```

The method reads an ordinary synthetic-observation file and calls the common
Elsasser implementation internally for both truth and one configured observed
MC member. It treats every `(time, spacecraft)` sample as a point in one frozen
spatial field. All unique unordered pairs are included: simultaneous
inter-spacecraft baselines, different-time cross-spacecraft baselines, and
same-spacecraft track baselines. Time differences have no dynamical meaning;
the method uses neither temporal lags nor Taylor's hypothesis.

By default, the pair scale and Elsasser increments are projected perpendicular
to the local pairwise mean field `0.5 * (B_A + B_B)`. A global mean-field
direction is used when that local field is too small. Each positive increment
is normalized within its scale bin by
`exp(mean(log(increment)))`. The pair upper triangle is streamed repeatedly in
bounded chunks: one pass finds scale limits, one accumulates moments and
geometric means, a bounded priority sample estimates the normalized-increment
percentile, and a final pass builds PDF histograms. `pair_sampling.mode:
random` is also available and samples unique pairs reproducibly without
replacement; `max_pairs` bounds either mode for debugging.

The HDF5 product contains one subtree per configured dataset:

```text
/results/<dataset>/lambda/bin_edges
/results/<dataset>/lambda/bin_centers
/results/<dataset>/lambda/counts
/results/<dataset>/lambda/valid_bin
/results/<dataset>/pdf/z_plus/{bin_edges,bin_centers,density,counts}
/results/<dataset>/pdf/z_plus/{typical_increment,rms_increment,fourth_root_increment}
/results/<dataset>/pdf/z_minus/...       same products
/results/<dataset>/alignment/{mean_sin_theta,counts}
/results/<dataset>/samples/...           only when explicitly enabled
```

Plotting writes one normalized `z_plus` PDF figure for each dataset, with one
curve per sufficiently populated perpendicular-scale bin. The method does not
yet implement the Fig. 1 inset, parallel coherence length, nonlinear or
Alfvén times, refined-critical-balance `chi`, or the Fig. 4 conditional
alignment diagnostic.

## Static PDF configuration


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

## Static spatial decorrelation analysis

| Summary | Details |
| --- | --- |
| Purpose | Measure simultaneous spatial correlations and second-order differences. |
| Required inputs | Native-cadence values/positions; truth or one MC member. |
| Entry points | `analysis.method: static_decorrelation`; section `decorrelation`. |
| Outputs | Binned HDF5 statistics and diagnostic PNG. |
| Limitations | No temporal lags or Taylor hypothesis; only order 2 here. |

See the [example manifest](../examples/index.md) for prerequisites and verification status,
and [visualization guide](../user-guide/visualization.md) for inspection and plotting.

`static_decorrelation` estimates simultaneous pairwise correlation as a function of spacecraft separation and computes the matching second-order pair-difference structure function. It treats every time index as an independent spatial snapshot. This first method intentionally includes no time lags, temporal evolution, Taylor hypothesis, flow advection, or wave propagation.

Run it after generating observations:

```bash
hsobs-generate configs/example_generate.yaml
hsobs-analyze configs/example_static_decorrelation.yaml
```

The analysis supports scalar density correlation, vector dot-product correlation, optional vector-component correlations, normalized or raw products, and linear or logarithmic separation bins. `data_group: observed` selects one `mc_index`; `data_group: truth` ignores it. `mc_index: all` is reserved for future ensemble statistics and currently raises `NotImplementedError`.

The default `per_time` mean subtracts the constellation mean independently at each snapshot. `global` subtracts one scalar or vector mean over all selected times and spacecraft. With the current native-cadence HDF5 schema, each variable uses its matching `/spacecraft/r/{variable}` positions. All selected variables share common bin edges, while `/counts/{variable}` records cadence-dependent sample counts.

The HDF5 output contains bin edges, lower and upper edges, centers, per-variable counts, correlation means/standard deviations/standard errors, and corresponding second-order structure-function statistics. Empty bins have count zero and NaN statistics. A default diagnostic PNG is written beside the HDF5 file as `<output_stem>_diagnostic.png`; it shows correlation and standard-error bands in the left column and second-order structure functions in the right column.

```yaml
analysis:
  method: static_decorrelation

decorrelation:
  variables: [n, v, B]
  data_group: observed
  mc_index: 0
  mean: {method: per_time}
  separation:
    units: km
    binning: log
    n_bins: 30
    r_min: null
    r_max: null
  correlation:
    vector_mode: dot
    component_correlations: true
    normalize: true
  structure_function: {compute: true, order: 2}
  visualization:
    enabled: true
    path: outputs/static_decorrelation_diagnostic.png
    dpi: 150
```

## Decorrelation configuration


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

## Single-spacecraft FFT analysis

| Summary | Details |
| --- | --- |
| Purpose | Measure each spacecraft frequency spectrum. |
| Required inputs | Uniformly sampled native time series. |
| Entry points | `analysis.method: timeseries_fft`; section `fft`. |
| Outputs | PSD HDF5 and observed/truth figures. |
| Limitations | No frequency-to-wavenumber conversion; no irregular-grid estimator. |

See the [example manifest](../examples/index.md) for prerequisites and verification status,
and [visualization guide](../user-guide/visualization.md) for inspection and plotting.

`timeseries_fft` computes one-sided real-FFT power spectral densities independently for every spacecraft time series. It uses no multi-spacecraft separation or spatial information and performs no Taylor-hypothesis conversion from frequency to wavenumber. Input time axes must be uniformly sampled; nonuniform data currently raise an error rather than being interpolated or passed to Lomb–Scargle analysis.

Run the example after generating observations:

```bash
hsobs-generate configs/example_generate.yaml
hsobs-analyze configs/example_timeseries_fft.yaml
```

Density is stored as `/psd/n`. Velocity and magnetic spectra are stored both by component as `/psd/v_components` and `/psd/B_components`, with shape `(spacecraft, frequency, 3)`, and as trace sums `/psd/v_trace` and `/psd/B_trace`. For shared-cadence input, `/frequency` is one dataset. Native-cadence files use `/frequency/{variable}` so density, velocity, and magnetic spectra retain their distinct instrument frequency grids.

With time step `dt` and window `w`, the normalization is

```text
PSD(f) = dt / sum(w^2) * |rFFT(w x)|^2
```

Strictly positive non-Nyquist bins are doubled, yielding approximate units of quantity²/Hz. Mean detrending and a Hann window are enabled by default. The DC bin is saved but omitted from the default log-log diagnostic plot, which overlays all spacecraft in panels for density, velocity trace, and magnetic trace. All panels use the same frequency-axis limits; native lower-cadence products therefore retain blank space above their Nyquist frequencies.

When analyzing `data_group: observed`, plotting also computes the corresponding truth spectra and writes them under `/truth_psd` in the analysis HDF5 file. The selected Monte Carlo realization and truth are rendered as separate figures by default, using identical axes and styling. For each parameter with positive truth spectral power, both figures use a common y-axis range spanning the observed and truth spectra. Parameters with zero truth power retain an independently autoscaled observed panel and an annotated zero-power truth panel. Set `plot.truth_enabled: false` to suppress the truth product.

```yaml
analysis:
  method: timeseries_fft

fft:
  data_group: observed
  mc_index: 0
  variables: [n, v, B]
  detrend: mean
  window: hann
  normalization: psd
  plot:
    enabled: true
    path: outputs/timeseries_fft.png
    truth_enabled: true
    truth_path: outputs/timeseries_fft_truth.png
    loglog: true
    overlay_spacecraft: true
    show_components: false
```

## FFT configuration


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

## `structure_function`

| Summary | Details |
| --- | --- |
| Purpose | Measure Elsasser increment moments and scale exponents. |
| Required inputs | Observation n/v/B and positions; independent MC members. |
| Entry points | `analysis.method: structure_function`. |
| Outputs | JSON moments/counts/flatness/fits and optional figures. |
| Limitations | Pair counts and selected spatial/fit assumptions limit inference. |

See the [example manifest](../examples/index.md) for prerequisites and verification status,
and [visualization guide](../user-guide/visualization.md) for inspection and plotting.

Binned Elsasser vector-increment structure functions. The default uses same-time spacecraft pairs; `increment_mode: all_spacetime_pairs` also treats samples as points of a frozen field.

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


### Additional implemented structure-function controls

`structure_function._settings` also accepts `min_pair_lambda_km`,
`max_pair_lambda_km` (alias `max_lambda_km`), `increment_mode`, and `pair_sampling`.
Modes are `all`, `random`, and `adaptive_stratified`. Random sampling requires
`n_pairs` and a seed. Adaptive stratification requires `all_spacetime_pairs`,
`target_pairs_per_bin`, `max_candidates`, and a seed; `binning_n_pairs` controls
the binning sample. `chunk_size` defaults to 1,000,000 and `binning_n_pairs` to
5,000,000. Bounds, sample counts and reported shortfalls affect interpretation.
These implemented controls supersede the old README's scaffold/TODO description;
they do not establish scientific suitability for every field.

Observed pair positions may incorporate `/errors/position_offset`; see
[data conventions](data-products.md#coordinate-use-by-analysis). Higher-order
moments and fitted exponents require adequate pair counts and an appropriate
fit range; no ensemble averaging is performed inside each member's statistic.

### Diagnostics input requirement

The shared `read_observations` reader currently requires truth datasets even
though residual computation has a conditional branch. The diagnostics workflow
therefore targets generated observation products, not arbitrary truth-free files.
