# Generating Synthetic Observations

`hsobs-generate` coordinates a YAML-configured set of observations.
The starting example is `configs/example_generate.yaml`.
The [field reference](../reference/fields.md) and
[error reference](../reference/errors.md)
describe model choices.
The data structure can be found in [data products](../reference/data-products.md).

1. **Configuration and sampling grids.** Load and validate the YAML, construct
   native time grids for density `n`, velocity `v`, and magnetic field `B`, and
   load spacecraft trajectories.
2. **Field model.** Construct a underlying field model following the prescribed
   field configuration.
   Models that require nominal formation context, e.g. the intermittent p-model,
   receive it once before truth or Monte Carlo evaluation.
3. **Measure true value.** Evaluate the field on nominal spacecraft coordinates.
4. **Coordinate sampling and position perturbations**
   Draw clock and position offsets per spacecraft and Monte Carlo realization.
   Evaluate trajectories at shifted times, add position offsets,
   sampling the field at those coordinates.
5. **Measurement frame and errors.**
   Convert from plasma frame to configured measurement frame.
   Applies measurement errors to the sampled ensemble.
6. **Output.** Write an HDF5 observation product, including nominal coordinates,
   true values, observations, configuration metadata, and configured error products.
   Optional visualization settings produce time-series and geometry figures.

## Overview of Synthetic Data Creation

`hsobs-generate` loads a spacecraft formation,
evaluates a plasma field along every trajectory at each instrument's native cadence,
creates a Monte Carlo ensemble using variable-specific error models, and writes true and observed values to and HDF5 output.
This output will be passed to `hobs-analyze`, a separate, registry-selected step.
The built-in `diagnostics` method reports observed and residual statistics and formation-separation summaries.

```yaml
spacecraft:
  source: cdf
  cdf_directory: ../HelioSwarm/HS-RT/PhB_SRD8_0x78f/
  configuration_time: "2030-06-01T00:00:00"
  fallback_synthetic: false
  # Common translation of Observatory
  velocity_km_s: [400.0, 10.0, 0.25]
```

The CDF input adapter accepts one `.cdf` file or a directory of files for the spacecraft positions, which can be found [here](https://doi.org/10.5281/zenodo.13887061).
It follows the conventions of the cdf variables,
which include `Epoch`, `Position`, and `Spacecraft_Label`.
The CDF records, which are currently on an hourly cadence,
are concatenated, sorted, filtered for fill values.
The interaval closest to the input `configuration_time` is selected.
The selected record is treated as a snapshot,
and advects all spacecrat in the observatory at a common `velocity_km_s`.
Full ephemeris interpolation is left to future extensions.

Choose a physical model before interpreting generated curves. Set the formation,
interval, native cadences, velocity measurement frame, seed and MC count. A fixed
seed does not fix external file contents or a CDF-versus-fallback choice. Record
`metadata.spacecraft_source` and the embedded YAML alongside results.

## Time grids and frames

We can configure the native instrument cadences independently:

```yaml
time:
  t_start_s: 0.0
  t_stop_s: 300.0
  cadence_s:
    n: 0.125
    v: 0.125
    B: 0.03125
```
For the current implementation, we have assumped a successful merge between the MAG and SCM to produce a single data product.
While there is technical flexibility to set `n` and `v` cadences independently, both will be measured at the same cadence by the Faraday cups.

**ESA measurements from the Hub will be added in later revisions.**

A single, scalar `cadence_s` applies the same cadence to all quantities.

The [data reference](../reference/data-products.md#velocity-frame-convention)
defines plasma and spacecraft frames. [Coordinate errors](../reference/errors.md)
move the sampling point along the trajectory at shifted time before adding the
position offset; stored coordinates remain nominal.

## Choose a field 

A field to be measured is defined with

```yaml
field:
	model: model-name
```
with current options:

- [Sine-sum](../reference/fields.md#sine-sum-field): introductory analytic inputs.
- [P-Model](../reference/fields.md#p-model-intermittent-turbulence): static periodic with tunable levels of intermittency.
- [Simulation sampling](../reference/fields.md#static-gkeyll-simulation-sampling): external snapshot from a Gkeyll simulation.
- [Linear Eigenmode](../reference/fields.md#plume-single-mode-plane-wave): external selected eigenmode, fixed geometry.
- [Planar shock](../reference/fields.md#propagating-planar-mhd-shocks): jump solution and smooth synthetic transition.

## Define errors

See the [error reference](../reference/errors.md) for details.
Use Gaussian/OU errors for configurable noise studies,
or the [observatory requirement model](../reference/errors.md#observatory-requirement-model)
for the current best model for the observatory performance.

## Inspect the run

Run `hsobs-generate CONFIG`, then check the printed output path, geometry-source
metadata, array shapes and [figures](visualization.md).
Observed arrays retain a leading MC dimension even for one member.
After confirming the success of a generated run, continue with [analysis](analysis.md).
