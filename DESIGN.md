# Design Notes for HelioSwarm Synthetic Observatory

This project should remain modular, testable, and scientifically transparent.

## Core principle

Do not mix these responsibilities:

1. Reading spacecraft trajectories.
2. Defining the underlying physical field.
3. Sampling the field along spacecraft paths.
4. Applying measurement errors.
5. Writing generated observations.
6. Rendering optional visualization products.
7. Running analysis methods.

Each responsibility should live in a separate module with a small, documented interface.

## Array conventions

Use the following conventions consistently:

```text
n_spacecraft = 9 for HelioSwarm
n_time_q     = number of native time samples for quantity q
n_mc         = number of Monte Carlo realizations
```

Spacecraft positions:

```text
position[q]: (n_spacecraft, n_time_q, 3)
```

Truth quantities:

```text
truth/n: (n_time_n, n_spacecraft)
truth/v: (n_time_v, n_spacecraft, 3)
truth/B: (n_time_B, n_spacecraft, 3)
```

Observed quantities:

```text
observed/n: (n_mc, n_time_n, n_spacecraft)
observed/v: (n_mc, n_time_v, n_spacecraft, 3)
observed/B: (n_mc, n_time_B, n_spacecraft, 3)
```

The Monte Carlo dimension should always be present for observed data.

## Field models

Field models should not know anything about instrument errors or file output.

A field model receives time and spacecraft position and returns truth values for `n`, `v`, and `B`.

The first field model is `SineSumField`. Future models may include turbulent boxes, wave packets, imported simulations, or analytic plasma-wave modes.

`MalaraTurbulence` is a low-level, time-independent isotropic turbulence evaluator. It evaluates a periodic, analytically solenoidal hierarchy of compact intermittent eddies directly at arbitrary positions. `MalaraTurbulenceField` adapts it to constant density and optional velocity/magnetic fluctuations in the observation generator. Anisotropic, critical-balance, and compressive-density extensions remain future work.

## Error models

Error models should not know anything about the underlying field model.

They receive truth values and a random number generator. They return observed values and, where possible, the actual error realization.

Time-correlated measurement errors also receive the quantity's native time axis. Correlation parameters should be expressed in physical time rather than sample counts so models remain consistent across instrument cadences.

Coordinate errors are drawn once per Monte Carlo realization and spacecraft. They modify the coordinates passed to the field model, while nominal coordinates remain stored for downstream analysis. Measurement errors are applied afterward to the resulting sampled ensemble.

Prefer reproducible random-number generation using `numpy.random.default_rng(seed)`.

## HDF5 output

The generated HDF5 file should contain enough metadata to reproduce the run. At minimum, save:

* serialized YAML configuration
* creation time
* random seed
* field model
* error model summary
* spacecraft configuration source

Avoid ASCII output for large time series.

## Analysis methods

Analysis methods should be selected by name from a registry. They should read HDF5 files and write compact outputs. Do not hard-code analysis choices inside the generator.

Initial analysis can be simple diagnostics. More advanced methods, such as wave telescope analysis and structure functions, should be added as separate modules.

## Coding style

Prefer readable, explicit scientific Python.

Use:

* NumPy for array operations
* h5py for HDF5 I/O
* PyYAML for configuration
* pytest for tests

Avoid unnecessary dependencies unless they solve a concrete problem.

Document shapes near the code that creates or transforms arrays.

Do not prematurely optimize, but avoid obviously slow Python loops over time samples when vectorized NumPy operations are straightforward.

## Testing

Tests should run without real HelioSwarm CDF files. Use a synthetic fallback spacecraft configuration for smoke tests.

Important tests:

* field model shape tests
* reproducible error tests
* HDF5 round-trip tests
* generator smoke test
* analysis smoke test
