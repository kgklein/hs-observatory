# Timing calibration

`timing-calibrate` analyzes synthetic nine-spacecraft magnetic measurements with
constant timing errors only. It uses the existing magnetic-gradient estimator,
all 382 subsets with 4–9 members by default, and eight timing parameters. The Hub
(`H`) is the fixed reference. Spacecraft and parameter ordering follow the input
IDs, with H omitted from the parameter vector.

## Run

Regenerate the timing case before analysis: older observations include a random
Hub offset and are rejected. The case now sets
`errors.timing.reference_spacecraft: H`. This optional generation setting zeros
only the named spacecraft's offset before field sampling; it leaves the other
random draws and existing generation defaults unchanged.

```sh
hsobs-generate generation/sim-cali-timing.yaml
hsobs-analyze analysis/sim-cali-timing_timing_calibrate.yaml
hsobs-plot-timing-calibrate outputs/sim-cali-timing_timing_calibrate.h5 \
  --output outputs/sim-cali-timing_timing_calibrate.png --mc-index 0
```

The plotting fallback is `python -m hs_observatory.cli_timing_calibrate` with the
same arguments. Each plotting call also saves a second figure beside the overview,
with `_timeseries` appended to the filename (and the same file extension). It has
four rows and two columns, showing N1–N8 in row order with ideal, measured, and
shifted (corrected) magnetic-field time series for `--mc-index`. It defaults to
Bx; use `--timeseries-component y`, `z`, `magnitude`, or `all` to choose a different
quantity. The `all` view uses component colors and ideal/measured/shifted line
styles. Both plots use the saved product, so no recalculation is needed.

Use `--start-time 100 --stop-time 110` to focus the Node and representative
divergence time series on 100–110 seconds (inclusive). Either bound can be omitted
to use the available beginning or end. The plotted data are restricted to that
interval so vertical limits also adjust to the selected samples. These options
only affect time-series displays; ensemble statistics and calibration results
remain those saved in the product. Empty or invalid time ranges are rejected.

Generation requires the simulation sources configured in the
case; analysis requires only the generated HDF5, and plotting only the analysis
product. Existing observation files are never modified by analysis.

## Fit conventions

An injected offset tau produces `B_measured(t) = B_ideal(t + tau)`. The fitted
correction c is applied as `B_corrected(t) = B_measured(t + c)`, so the reported
recovered injected offset is **-c**. Bias, RMSE, and the 1:1 recovery plot compare
-c with tau. Injected offsets are used for diagnostics, never to build the fit.

One fixed divergence variance per polyhedron is the ideal time-series mean of
`d**2`, without subtracting its mean. `timing_calibrate.sigma_d_floor` defaults to
`1e-12` nT/km and floors the standard deviation. Both raw and floored squared
values are saved. No additional temporal or geometry weights are applied.

The common prior `sigma_t_s` defaults to the embedded generation uncertainty
(12.5 ms / 3 in this case), and can be explicitly overridden. The augmented
weighted system is solved once with NumPy least squares. Measured-field time
derivatives use centered differences and one-sided endpoints. A shared response
helper builds coefficients through the existing gradient estimator, caching
static relative geometry. Absent spacecraft contribute exactly zero to the
Jacobian. Covariance is computed from an SVD of the augmented design's QR factor,
without residual-variance scaling. It is the covariance implied by the specified
diagonal weighting and prior; overlapping polyhedra are not independent samples.

Corrections use the existing linear time-alignment routine without extrapolation.
Nominal positions remain the analysis geometry. The corrected field is passed
through the gradient estimator responses again; corrected diagnostics are not
just `d + J c`. Unsupported endpoints remain NaN with explicit validity masks.
Actual before/after diagnostics share the same comparison rows. Linear objective
values use the original fit rows and are saved separately. An actual chi-square
increase is recorded and warned about, not hidden or used to modify the fit.
Ideal discrete divergence can bias a zero-error fit; exact zero recovery is only
expected when that discrete divergence is zero or orthogonal to the Jacobian.

## Configuration and output

Use the usual `input`, `analysis`, and `output` mappings, with
`analysis.method: timing-calibrate` and a `timing_calibrate` options mapping.
Optional `polyhedra` is a list of unique spacecraft-ID lists. Alternatively,
`configuration_selection` accepts the existing magnetic-gradient categories.
The two selection modes are mutually exclusive. The `gradient` and `storage`
settings follow magnetic-gradient conventions. All MC realizations are retained.

The product contains:

- `metadata`: original generation/analysis YAML, version, commit, signs, settings,
  and a `complete` attribute set only after all realizations are written.
- `time/analysis_time`, `spacecraft/id`, `node_id`, `node_indices`, and nominal
  positions, preserving observation ordering.
- `magnetic_field/ideal` (time, spacecraft, component), and `measured` / `corrected`
  (MC, time, spacecraft, component), with `corrected_valid` (MC, time, spacecraft).
  Shared ideal data applies to every realization.
- `configurations/<members-or-category>`: members, N, ideal divergence (time),
  measured/corrected divergence and fit/comparison masks (MC, time), raw/effective
  variance, and normalized RMS before/after (MC).
- `timing`: injected/recovered offsets and applied corrections (MC, 8), and
  covariance (MC, 8, 8), with seconds and squared-seconds units.
- `diagnostics`: actual and linear chi-square, divergence and prior contributions,
  RMS, row counts, unweighted/weighted Jacobian spectra/ranks/condition numbers,
  augmented spectrum/rank, and actual objective-increase flags.
- `mc_summary`: mean, bias, RMSE, population standard deviation, fractional RMS
  reduction `(before-after)/before`, its 25th/50th/75th percentiles, and full
  chi-square distributions. Fractional reduction is NaN for a zero baseline.

Invalid gradient rows follow the existing estimator's pseudoinverse settings.
No valid fit/comparison rows is an error. Rank-deficient data Jacobians have
infinite reported condition numbers, while the prior permits a regularized fit.
The output streams one realization at a time to avoid retaining the complete
ensemble in memory. A failed calculation leaves `metadata.complete = false`.

The diagnostic figure uses Node colors, eight-node recovery/error summaries,
per-MC RMS, a representative weighted spectrum, and up to three polyhedra spread
across the saved configuration order. There is no per-polyhedron panel explosion.

### Optional ideal-divergence reference

Set `timing_calibrate.use_ideal_divergence: true` to fit measured minus ideal
polyhedron divergence. This is enabled in the `null0` analysis configuration;
other configurations retain the absolute-divergence target by default. Weights
still use the ideal mean-square divergence about zero. Both linear and resampled
chi-square diagnostics use the selected target, recorded in `metadata.chi2_target`.
Saved divergence series and normalized divergence RMS remain absolute-divergence
quantities, so existing plots keep their meaning. Rerun analysis to apply the
setting to an existing product.

### Zero-error polyhedron influence diagnostic

`diagnostics.polyhedron_influence: true` adds an ideal-series influence analysis.
It requires `use_ideal_divergence: false`, exactly zero injected offsets, and
measured B equal to ideal B. No weights, geometry selection, solver, or prior are
changed. The diagnostic is shared across the identical MC inputs. Run:

```sh
hsobs-analyze analysis/sim-cali-timing-null0_timing_influence.yaml
python -m hs_observatory.cli_timing_calibrate \
  outputs/sim-cali-timing-null0_timing_influence.h5 \
  --output outputs/sim-cali-timing-null0_timing_influence.png
```

This separate configuration preserves the null0 residual-fit experiment and its
output. It inherits the current null0 prior of **12.5 seconds**.

The HDF5 `timing_influence` group saves g, g per polyhedron, H, predicted correction,
correction per polyhedron, the unchanged least-squares reference solution, the fit
mask, ranking, cumulative signed/absolute contributions, and numerical-check
errors/flags. The decomposition uses the full H held fixed: it is additive
influence, not a leave-one-out refit. Contributions are *corrections* c, not
recovered injected offsets -c. H is stored explicitly, but its inverse is never
formed for this calculation: contributions use the SVD of the augmented design's
QR factor. The truncation threshold matches the existing solver. Failed sum or
solver-agreement checks raise an error instead of silently saving bad diagnostics.

Configuration membership and N are linked through the saved polyhedron IDs. Each
configuration gains `influence_geometry` with R, L, cond(R), and (for N>4) the ideal
affine RMS residual over time. Static geometry is stored once. L is the existing
`2*sqrt(max eigenvalue(R))`, and affine residual is the existing vector RMS about
the affine fit. Ranking and bins use median L/condition over time and the temporal
RMS of affine residual. We retain sigma_d from the actual floored weights.

Plotting automatically adds three figures and a top-ten-per-Node CSV:

- `_influence_heatmaps`: signed g and timing contributions (ms), ordered by N;
  symmetric log colors make small signed contributions visible.
- `_influence_cumulative`: signed and absolute cumulative sums, ranked by |c|.
- `_influence_aggregates`: signed sums and sums of magnitudes, grouped by N and
  bins of L, cond(R), and affine residual.
- `_influence_top.csv`: ranked members, N, L, cond(R), sigma_d, affine residual,
  signed g, signed c, and |c|.

Continuous bins use saved quartile edges over finite per-polyhedron summaries;
duplicate edges collapse. Bins are left-inclusive, right-exclusive except the
last finite bin. Nonfinite values (and N=4 affine residuals, which are not
applicable) have their own bin, so sums retain all contributions. Absolute sums
are sums of |c_P|, not the absolute value of signed bin totals. These diagnostics
always use the full fit interval; plotting time-window arguments only crop the
existing time-series displays.
