# Installation

HS Observatory requires **Python 3.10 or newer**. The established workflow is an
editable installation from a local checkout. Obtain a checkout *(repository access is required while private)*:

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

The activation command above is for a POSIX shell;
use the equivalent activation command for your platform.
An editable installation makes the checkout available to Python and
installs the `hsobs-*` commands into the active environment.

Runtime dependencies are NumPy, SciPy, PyYAML, h5py, cdflib, Matplotlib, and cmcrameri.
There is no need to install documentation tools to generate or analyze observations.
SciPy supports shock-timing fits and continuous RBF length-scale optimization.
Explicit RBF cross-validation uses a logarithmic grid search without SciPy.

## HelioSwarm Trajectories

Representative HelioSwarm trajectories are available as a companion data product.
The trajectory set is archived on Zenodo at <https://doi.org/10.5281/zenodo.13887061>
and can be downloaded separately from the software installation.

## Optional dependencies

BP inputs additionally require `simulation-bp`; GIF/MP4 animation encoding
requires `movie`. These extras are independent of documentation tooling.

For the existing test and lint tools:

```bash
python3 -m pip install -e ".[dev]"
```

For ADIOS2 `.bp` simulation input:

```bash
python3 -m pip install -e ".[simulation-bp]"
```

For [Movies](../user-guide/movies.md), install imageio and imageio-ffmpeg:

```bash
python -m pip install -e '.[movie]'
```

This also registers `hsobs-visualize geometry-movie`. The CDF geometry files
must be obtained separately; installing the extra does not download them.
The `--dry-run` preflight requires the CDF input but does not require encoders.

To install movie encoding and ADIOS2 input support together, combine the extras:

```bash
python3 -m pip install -e ".[movie,simulation-bp]"
```

The basic example uses a synthetic fallback and does not need simulation files
or ADIOS2. Other configurations may require your own CDF, PLUME, or simulation
inputs; paths in those configurations must match your local data.

For local documentation development:

```bash
python3 -m pip install -e ".[docs]"
```

See [Documentation Development](../developer/documentation.md) for serving,
strict builds, and the optional system Graphviz executable used for [diagrams](../architecture/module-dependencies.md).

## Environment checks

Use `python -m pip` inside the activated environment and check the commands:

```bash
hsobs-generate --help
hsobs-visualize geometry-movie --help
```

If commands are missing, verify activation and editable installation. BP examples
require both ADIOS2 and the source files; installing an extra does not download
mission, PLUME or simulation data.
The [configuration reference](../reference/configuration.md) explains path resolution.
