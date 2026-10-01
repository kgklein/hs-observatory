# Choosing and running an analysis

A full listing of methods can be found [here](../reference/analysis_methods.md).
Generation and analysis are separate; an analysis configuration names its input.

## Run an analysis

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

Run `hsobs-analyze CONFIG` from the repository root. The selector is not always
the parameter-section name: `static_decorrelation` uses `decorrelation`,
`timeseries_fft` uses `fft`, and `timing-calibrate` uses `timing_calibrate`.
See [configuration dispatch](../reference/configuration.md#analysis-dispatch).

## Select truth and Monte Carlo members

Do not assume all methods accept `mc_index: all`. Elsasser, decorrelation, FFT,
RBF and Timesync select a single observed member; structure functions process
members independently, while shock timing and timing calibration retain ensemble
results. Check the method's exact controls. Truth may be optional for a validation
layer while still required elsewhere in a particular pipeline.

## Inspect and interpret

Check units, frames, masks, failure statuses, sample counts and provenance before
comparing curves. [Coordinate use](../reference/data-products.md#coordinate-use-by-analysis)
differs among methods. Consult [visualization](visualization.md) and retain the
saved analysis product so figures can be reproduced independently where supported.

## Troubleshooting

Unknown method names are reported with the registry's available choices. Missing
input paths, uniform-cadence failures, insufficient crossing coverage, rank loss,
and incomplete streamed products need attention to the input/fit, not cosmetic
plot changes. Method references describe supported options and rejected cases.


## Choose a method

| Question | Selector | Reference / constraints |
| --- | --- | --- |
| Is the generated ensemble sensible? | `diagnostics` | [Basic statistics](../reference/statistical-methods.md#diagnostics); generated truth required by reader. |
| What are the Elsasser fields? | `elsasser` | [Elsasser](../reference/statistical-methods.md#elsasser-derived-quantities); proton density, aligned native cadences. |
| How do increments depend on spatial scale? | `elsasser_increment_pdf_static`, `structure_function` | [Statistics](../reference/statistical-methods.md); distinct pair selections and frozen-field assumptions. |
| How do simultaneous correlations vary with separation? | `static_decorrelation` | [Decorrelation](../reference/statistical-methods.md#static-spatial-decorrelation-analysis); no time-lag/Taylor interpretation. |
| What is each spacecraft's frequency spectrum? | `timeseries_fft` | [FFT](../reference/statistical-methods.md#single-spacecraft-fft-analysis); uniform sampling. |
| How accurately are gradients recovered? | `magnetic_gradient` | [Gradient reference](../reference/magnetic-gradients.md); external simulation reference and stored offsets. |
| How do saved gradient components covary in time? | `magnetic_gradient_correlation` | [Correlation](../reference/magnetic-gradients.md#magnetic_gradient_correlation); standalone gradient product. |
| Can a sampled magnetic volume be reconstructed? | `rbf_reconstruction`, `timesync` | [Reconstruction](../reference/reconstruction.md); window/grid/coverage constraints. |
| What is a planar shock's propagation? | `shock_timing` | [Shock timing](../reference/shock-timing.md); SciPy required. |
| How do research clock-correction methods behave? | `timing-calibrate` | [Timing calibration](../reference/timing-calibration.md); synthetic research pipeline, not flight calibration. |
