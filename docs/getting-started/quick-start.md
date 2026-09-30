# First Light

After completing the [installation](installation.md),
activate the environment and run commands from the repository root,
enabling relative configuration and output paths to resolve as expected.

## Generating Example Observations

For our first calculation, we will generate a time series.

```bash
hsobs-generate configs/example_generate.yaml
```

The script expects a CDF file for the positions of the nine HS spacecraft, covering June 1, 2030.
In the `spacecraft:` namelist in `example_generate.yaml`, ensure that `cdf_directory` points to the directory where you have stored the [representative trajectories](https://doi.org/10.5281/zenodo.13887061).

This first example uses a sum of sine wave for the underlying field,
with expected measurement cadences for n,V, and B.
We include Gaussian measurement errors for position, time, and plasma quantites,
with ten Monte Carlo realizations.
It requests CDF spacecraft geometry.
Enabling `fallback_synthetic: true`, will allow it to run without mission files for arbitrarily selected relative positions.

The configured products include:

- `outputs/example/synthetic_observations.h5`: true measurements, observed measurements, coordinates,
  errors, and generation metadata.
- Truth and observed time-series/geometry PNGs under `outputs/example/`.

Example Figures..

## Run a basic analysis

```bash
hsobs-analyze configs/example_analyze.yaml
```

This configuration reads the generated observation file and writes a basic
statistical report to `outputs/diagnostics.json`.

## Use Python instead of the CLI

The preferred package-root Python API consists of `__version__`,
`generate_from_config`, and `analyze_from_config`:

```python
from hs_observatory import analyze_from_config, generate_from_config

observation_path = generate_from_config("configs/example_generate.yaml")
analysis_path = analyze_from_config("configs/example_analyze.yaml")
```

Both functions accept a string or `pathlib.Path` pointing to the same YAML files
used by `hsobs-generate` and `hsobs-analyze`, and return the configured output
path. The analysis configuration selects its input file; it is not automatically
connected to the preceding Python return value. Run this example from the
repository root, just like the CLI commands above.

This small root API is the preferred user-facing Python interface. Specialized
scientific routines and advanced extension interfaces remain in their logical
modules/subpackages; those paths may evolve during the 0.x series. Private
helpers and registry internals are implementation details, not stability promises.

## Choose an analysis

Analysis YAML selects a method using `analysis.method`. See
[Analysis Methods](../user-guide/analysis.md) for the method-specific settings and
example configuration names. Check each example's input/output paths and any
required simulation data before running it.

Read [Generating Synthetic Observations](../user-guide/generation.md) to change
field or sampling choices, and the [Observatory Error Model](../reference/errors.md)
for the requirement-driven error path.
## Inspect provenance and continue

The CDF fallback makes this example runnable without mission files, but a local
CDF directory can change the geometry. Inspect `metadata.spacecraft_source` and
its fallback reason. Native sample counts for the supplied 0–300 s interval are
2401 for n/v and 9601 for B; the MC dimension is 10. Diagnostics report 36 distinct
spacecraft pairs for nine-spacecraft geometry.

Read the [verification record and complete example manifest](../examples/index.md#starter-workflow),
then [visualization](../user-guide/visualization.md) or [analysis selection](../user-guide/analysis.md).
