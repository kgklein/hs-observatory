# HelioSwarm Synthetic Observatory

This package generates and analyzes synthetic HelioSwarm-like multi-spacecraft observations.

The initial purpose is to create controlled mock data sets for testing multi-spacecraft plasma analysis methods under realistic sampling geometries and measurement errors. The code is organized as a two-stage workflow:

1. **Generate synthetic observations** from a prescribed plasma field sampled along HelioSwarm spacecraft trajectories.
2. **Analyze synthetic observations** using selected multi-spacecraft analysis methods.

The first implemented field model is a configurable sum of sine waves. Future field models may include turbulent boxes, wave packets, imported numerical simulations, or physically constrained plasma-wave models.

## Workflow

A typical workflow is:

```bash
hsobs-generate configs/example_generate.yaml
hsobs-analyze configs/example_analyze.yaml
```

The generation step reads a HelioSwarm spacecraft configuration, samples an underlying plasma field at each spacecraft position and time, applies optional measurement errors, and writes the result to an HDF5 file.

The analysis step reads the generated HDF5 file and applies a selected analysis method.

## Generated quantities

Each spacecraft samples:

* density `n`
* velocity vector `v`
* magnetic field vector `B`

The generated file stores both truth and observed values. This is important for testing analysis methods because it allows direct comparison between the known input field and the noisy synthetic measurements.

## HDF5 layout

Generated files use the following layout:

```text
/metadata
/time
/spacecraft/id
/spacecraft/r
/truth/n
/truth/v
/truth/B
/observed/n
/observed/v
/observed/B
/errors/n
/errors/v
/errors/B
```

The expected dimensions are:

```text
time                  (n_time,)
spacecraft/r          (n_spacecraft, n_time, 3)

truth/n               (n_time, n_spacecraft)
truth/v               (n_time, n_spacecraft, 3)
truth/B               (n_time, n_spacecraft, 3)

observed/n            (n_mc, n_time, n_spacecraft)
observed/v            (n_mc, n_time, n_spacecraft, 3)
observed/B            (n_mc, n_time, n_spacecraft, 3)
```

The leading `n_mc` dimension is always present in observed quantities, even when only one Monte Carlo realization is generated.

## Configuration files

The package is controlled by YAML configuration files. Generation and analysis use separate config files.

Example generation sections include:

```yaml
spacecraft:
  source: cdf
  cdf_directory: /path/to/helioswarm/cdfs
  configuration_name: example_configuration
  fallback_synthetic: true

time:
  cadence_s: 1.0
  t_start_s: 0.0
  t_stop_s: 3600.0

field:
  model: sine_sum
  background:
    n: 5.0
    v: [400.0, 0.0, 0.0]
    B: [5.0, 0.0, 0.0]
  modes:
    - quantity: B
      amplitude: [0.0, 0.5, 0.0]
      k: [0.001, 0.0, 0.0]
      omega: 0.02
      phase: 0.0

errors:
  random_seed: 12345
  n_mc: 100
  variables:
    n:
      model: gaussian_relative
      sigma: 0.05
    v:
      model: gaussian_absolute
      sigma: [2.0, 2.0, 2.0]
    B:
      model: gaussian_absolute
      sigma: [0.05, 0.05, 0.05]

output:
  path: outputs/synthetic_observations.h5
```

## Adding a new field model

New field models should inherit from the base field interface and implement an `evaluate(time, position)` method returning a dictionary with keys:

```python
{
    "n": n_array,
    "v": v_array,
    "B": B_array,
}
```

Expected shapes are:

```text
n: (n_time, n_spacecraft)
v: (n_time, n_spacecraft, 3)
B: (n_time, n_spacecraft, 3)
```

## Adding a new error model

New error models should inherit from the base error interface and implement an `apply(truth, rng)` method. Error models should return both the observed value and the actual error realization where possible.

Error models should be independent of the field model and should be selectable from the YAML configuration.

## Adding a new analysis method

New analysis methods should inherit from the base analysis interface and register themselves in the analysis registry. The analysis config selects methods by name.

Initial analysis methods include simple diagnostics and a placeholder structure-function analysis. Future methods may include wave telescope analysis, timing analysis, k-filtering, curlometer-style estimates, and multi-spacecraft structure functions.

## Design priorities

The package favors:

* clear scientific array shapes
* explicit metadata
* reproducible random seeds
* compact binary output
* modular field, error, and analysis models
* tests that run without real HelioSwarm CDF files
* readable code that can be extended by physicists and software developers
