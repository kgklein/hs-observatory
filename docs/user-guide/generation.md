# Generating Synthetic Observations

`hsobs-generate` coordinates a YAML-configured observation experiment. The
starting example is `configs/example_generate.yaml`; the repository README
describes additional field and error configurations.

1. **Configuration and sampling grids.** Load and validate the YAML, construct
   native time grids for density `n`, velocity `v`, and magnetic field `B`, and
   load spacecraft trajectories or the configured synthetic fallback.
2. **Field model.** Construct a sine-sum, Malara turbulence, PLUME single-mode,
   or static simulation-sampling field from the field configuration.
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

Coordinate errors affect where the field is sampled; measurement errors act on
the resulting values. The nominal coordinates remain available for downstream
analysis. Observed arrays retain a Monte Carlo dimension even for one realization.

Generation and analysis are separate stages: `hsobs-analyze` reads the saved
observation product and runs a selected analysis. See the
[Quick Start](../getting-started/quick-start.md),
[Analysis Methods](../analysis_methods.md), and
[Observatory Error Model](../observatory_error_model.md).

For detailed configurations and storage conventions, consult the
[repository README](https://github.com/kgklein/hs-observatory/blob/main/README.md).
