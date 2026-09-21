# Installation

HS Observatory requires **Python 3.10 or newer**. The established workflow is an
editable installation from a local checkout. From the repository root:

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

For continuous SciPy-based RBF length-scale optimization and optional Crameri
scientific colormaps:

```bash
python3 -m pip install -e ".[analysis]"
```

Core analyses work without this extra: explicit RBF cross-validation uses a
logarithmic grid search without SciPy, and plotting uses Matplotlib colormap
fallbacks without cmcrameri. The standalone
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
