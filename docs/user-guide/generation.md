# Generating Synthetic Observations

`hsobs-generate` coordinates a YAML-configured set of observations.
The starting example is `configs/example_generate.yaml`.
The [field reference](../reference/fields.md) and
[error reference](../reference/errors.md)
describe model choices.

1. **Configuration and sampling grids.** Load and validate the YAML, construct
   native time grids for density `n`, velocity `v`, and magnetic field `B`, and
   load spacecraft trajectories or the configured synthetic fallback.
2. **Field model.** Construct a sine-sum, Malara turbulence, PLUME single-mode,
   propagating planar MHD shock, or static simulation-sampling field from the
   field configuration. Models that require nominal formation context receive
   it once before truth or Monte Carlo evaluation.
3. **Nominal truth.** Evaluate the field on nominal spacecraft coordinates.
4. **Coordinate perturbations and sampling.** Draw clock and position offsets
   per spacecraft and Monte Carlo realization. Evaluate trajectories at shifted
   times, add position offsets, and sample the field at those coordinates.
5. **Measurement frame and errors.** Field velocity is in the plasma frame.
   Generation converts it to the configured measurement frame and applies
   measurement errors to the sampled ensemble.
6. **Output.** Write an HDF5 observation product, including nominal coordinates,
   truth, observations, configuration metadata, and configured error products.
   Optional visualization settings produce time-series and geometry figures.

## Design the experiment

## Two-stage workflow

Generation loads a spacecraft formation, evaluates a plasma field along every trajectory at each instrument's native cadence, creates a leading-dimension Monte Carlo ensemble using variable-specific error models, and writes truth plus observations to HDF5. Analysis is a separate, registry-selected step that reads that file. The built-in `diagnostics` method reports observed and residual statistics and formation-separation summaries.

The CDF adapter accepts one `.cdf` file or a directory of files. It follows the conventions found in the sibling `sparse-field-reconstruction` project: variables named `Epoch`, `Position`, and `Spacecraft_Label`; records are concatenated, sorted, filtered for fill values, and selected nearest to `configuration_time`. This first version treats the selected record as a snapshot and may advect it with a common `velocity_km_s`. The example includes an illustrative common translation so its path projection is visible. Full ephemeris interpolation is a future extension.

Choose a physical model before interpreting generated curves. Set the formation,
interval, native cadences, velocity measurement frame, seed and MC count. A fixed
seed does not fix external file contents or a CDF-versus-fallback choice. Record
`metadata.spacecraft_source` and the embedded YAML alongside results.

## Time grids and frames

Configure native instrument cadences independently:

```yaml
time:
  t_start_s: 0.0
  t_stop_s: 300.0
  cadence_s:
    n: 0.125
    v: 0.125
    B: 0.03125
```

A scalar `cadence_s` remains supported and applies the same cadence to all quantities.

The [data reference](../reference/data-products.md#velocity-frame-convention)
defines plasma and spacecraft frames. [Coordinate errors](../reference/errors.md)
move the sampling point along the trajectory at shifted time before adding the
position offset; stored coordinates remain nominal.

## Choose a field and errors

- [Sine-sum](../reference/fields.md#sine-sum-field): introductory analytic inputs.
- [Malara](../reference/fields.md#malara-style-intermittent-turbulence-evaluator): static periodic intermittent field.
- [Simulation sampling](../reference/fields.md#static-numerical-simulation-sampling): external NPY or BP snapshot, explicit layout/scaling.
- [PLUME](../reference/fields.md#plume-single-mode-plane-wave): external selected eigenmode, fixed geometry; review the documented ambiguity.
- [Planar shock](../reference/fields.md#propagating-planar-mhd-shocks): jump solution and smooth synthetic transition.

Use Gaussian/OU errors for configurable noise studies, or the
[observatory requirement model](../reference/errors.md#observatory-requirement-details).
The latter is distinct from provisional timing-calibration noise weighting.

## Inspect the run

Run `hsobs-generate CONFIG`, then check the printed output path, geometry-source
metadata, array shapes and [figures](visualization.md). Observed arrays retain a
leading MC dimension even for one member. Continue with [analysis](analysis.md).

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Unexpected geometry | CDF path, configuration time and fallback source/reason in metadata. Fallback can also follow a CDF validation error. |
| Missing external input | PLUME/BP/NPY path and optional BP dependency; CDF fallback does not supply a plasma field file. |
| Velocity-frame error | Supply spacecraft velocity or select the supported plasma measurement frame. |
| Unexpected units/amplitude | Native units, simulation scaling, total/fluctuation semantics and frame; do not infer conversions. |
| Unexpected output location | Run from repository root; most input/output paths are working-directory relative. |
