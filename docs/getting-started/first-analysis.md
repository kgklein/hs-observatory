# Running a Basic Analysis

After generating the first set of obervations, which should be in `outputs/example`,
we can run an intial example analysis calculation:

```bash
hsobs-analyze configs/example_analyze.yaml
```

This configuration reads the generated observation file and writes a basic
statistical report to `outputs/example/diagnostics.json`.

For more details,
read the [verification record and complete example manifest](../examples/index.md#starter-workflow),
then [visualization](../user-guide/visualization.md).

## Use Python instead of the CLI

If you are looking to integrate this package into your python workflow,
rather than this repo,
the preferred package-root Python API consists of `__version__`,
`generate_from_config`, and `analyze_from_config`:

```python
from hs_observatory import analyze_from_config, generate_from_config

observation_path = generate_from_config("configs/example_generate.yaml")
analysis_path = analyze_from_config("configs/example_analyze.yaml")
```

Both functions accept a string or `pathlib.Path` pointing to the same YAML files
used by `hsobs-generate` and `hsobs-analyze`, and return the configured output path.
The analysis configuration selects its input file (which is output from the generation calculation);
it is not automatically connected to the preceding Python return value.
Run this from the repository root, just like the CLI commands above.

This small root API is the preferred user-facing Python interface.
Specialized scientific routines and advanced extension interfaces remain in their logical
modules/subpackages; *those paths may evolve during the 0.x series.*
Private helpers and registry internals are implementation details, not stability promises.

## Choose an analysis

Analysis YAML selects a method using `analysis.method`. See
[Analysis Methods](../user-guide/analysis.md) for the method-specific settings and
example configuration names. Check each example's input/output paths and any
required simulation data before running it.