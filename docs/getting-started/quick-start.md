# Quick Start

Complete [installation](installation.md), activate that environment, and run
commands from the repository root so relative configuration and output paths
resolve as expected.

## Generate observations

```bash
hsobs-generate configs/example_generate.yaml
```

This existing example uses a sine-sum field, quantity-specific measurement
cadences, coordinate offsets, Gaussian measurement errors, and ten Monte Carlo
realizations. It requests CDF spacecraft geometry but enables
`fallback_synthetic: true`, so it also runs without mission files.

The configured products include:

- `outputs/synthetic_observations.h5`: truth, observed measurements, coordinates,
  errors, and generation metadata.
- Truth and observed time-series/geometry PNGs under `outputs/`.

## Run a basic analysis

```bash
hsobs-analyze configs/example_analyze.yaml
```

This configuration reads the generated observation file and writes a basic
statistical report to `outputs/diagnostics.json`.

## Choose an analysis

Analysis YAML selects a method using `analysis.method`. See
[Analysis Methods](../analysis_methods.md) for the method-specific settings and
example configuration names. Check each example's input/output paths and any
required simulation data before running it.

Read [Generating Synthetic Observations](../user-guide/generation.md) to change
field or sampling choices, and the [Observatory Error Model](../observatory_error_model.md)
for the requirement-driven error path.
