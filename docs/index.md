# HS Observatory

HS Observatory generates controlled, synthetic multipoint observatory
measurements inspired by HelioSwarm and analyzes the resulting products.
Physical fields, spacecraft sampling, coordinate and measurement errors, and
analysis workflows are configured through YAML files.

Field models include sums of sine waves, Malara-style intermittent turbulence,
a selected PLUME eigenmode, and sampling of static numerical simulations.
Generated HDF5 files retain truth, observed Monte Carlo realizations, and metadata
so the effects of sampling and measurement errors can be studied separately.

The analysis suite covers Elsasser quantities, structure functions and static
increment PDFs, spatial decorrelation, time-series spectra, multipoint magnetic
gradients, and RBF or Timesync magnetic-field reconstruction. See
[Analysis Methods](analysis_methods.md) for inputs, options, and limitations.

## Quick start

After [installation](getting-started/installation.md), run these commands from
the repository root:

```bash
hsobs-generate configs/example_generate.yaml
hsobs-analyze configs/example_analyze.yaml
```

The example enables a synthetic spacecraft fallback, so mission CDF files are
not required. It writes `outputs/synthetic_observations.h5`, diagnostic figures,
and `outputs/diagnostics.json`.

Continue with the [Quick Start](getting-started/quick-start.md), learn how
[generation](user-guide/generation.md) works, or explore the
[architecture overview](architecture/index.md).

This documentation is being prepared for public release. The
[repository README](https://github.com/kgklein/hs-observatory/blob/main/README.md)
contains additional examples; repository links require access while it is private.
