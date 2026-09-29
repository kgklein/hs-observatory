# Installation

HS Observatory requires **Python 3.10 or newer**. The established workflow is an
editable installation from a local checkout. Obtain a checkout (repository access is required while private):

```bash
git clone https://github.com/kgklein/hs-observatory.git
cd hs-observatory
```

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -e .
```

The activation command above is for a POSIX shell. Use the equivalent activation
command for your platform. An editable installation makes the checkout available
to Python and installs the `hsobs-*` commands into the active environment.

Runtime dependencies are NumPy, PyYAML, h5py, cdflib, and Matplotlib. There is no
need to install documentation tools to generate or analyze observations.

## Optional dependencies

For the full quick-start figures (current geometry code imports cmcrameri),
shock-timing fits and continuous SciPy-based RBF length-scale optimization:

```bash
python3 -m pip install -e ".[analysis]"
```

Many analyses work without this extra, but shock timing requires SciPy. Explicit RBF cross-validation uses a
logarithmic grid search without SciPy, and several plotters use Matplotlib colormap
fallbacks without cmcrameri. However, default generation geometry plotting
currently imports cmcrameri unconditionally, so install this extra before the
quick start. Base-only generation can write HDF5 and then fail during plotting;
this is a known implementation/dependency discrepancy, not a completed run. The standalone
`scripts/audit_magnetic_gradient_divergence.py` requires SciPy and therefore
needs this extra; BP inputs additionally require `simulation-bp`.

For the existing test and lint tools:

```bash
python3 -m pip install -e ".[dev]"
```

For ADIOS2 `.bp` simulation input:

```bash
python3 -m pip install -e ".[simulation-bp]"
```

To install both sets of optional scientific features, combine the extras:

```bash
python3 -m pip install -e ".[analysis,simulation-bp]"
```

The basic example uses a synthetic fallback and does not need simulation files
or ADIOS2. Other configurations may require your own CDF, PLUME, or simulation
inputs; paths in those configurations must match your local data.

For local documentation development:

```bash
python3 -m pip install -e ".[docs]"
```

See [Documentation Development](../developer/documentation.md) for serving,
strict builds, and the optional system Graphviz executable used for diagrams.
Then follow the [Quick Start](quick-start.md).
## Environment checks

Use `python -m pip` inside the activated environment and check `hsobs-generate --help`.
If commands are missing, verify activation and editable installation. BP examples
require both ADIOS2 and the source files; installing an extra does not download
mission, PLUME or simulation data. See the [example manifest](../examples/index.md).
The [configuration reference](../reference/configuration.md) explains path resolution.
