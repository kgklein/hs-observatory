# HelioSwarm Synthetic Observatory

HS Observatory generates synthetic multipoint spacecraft observations from
representative HelioSwarm (HS) positions and analyzes the resulting products.
YAML files specify the generation of data, including
physical fields,
spacecraft geometry,
native instrument cadences, and
coordinate/measurement errors.
YAML files also specify the details of the analysis methods.
Output HDF5 products retain nominal true measurements, Monte Carlo observations,
and generation details.

Field, error and analysis models provide specific locations for growth through
community input.
Scientific assumptions and validation limits are documented per method.
*Some methods, e.g. timing calibration via $\nabla \cdot \mathbf{B}=0$, are currently under development.*

## Install

Use Python **3.10 or newer** and a local checkout of this repository. From its root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The base installation includes
[`cmcrameri`](https://zenodo.org/records/8409685), which is used throughout for
perceptually uniform colour maps, and SciPy for analysis features such as shock timing.
The activation command is for a POSIX shell.
See [installation](docs/getting-started/installation.md) for
checkout instructions, other environments, optional dependencies and external inputs.

## Quick start

Run from the repository root in the activated environment:

```bash
hsobs-generate configs/example_generate.yaml
hsobs-analyze configs/example_analyze.yaml
```

The sine-sum example enables synthetic geometry fallback,
so mission CDF files are not required.
It writes `outputs/synthetic_observations.h5`,
diagnostic figures under `outputs/`, and `outputs/diagnostics.json`.
If the configured CDF directory exists, its geometry may be used instead.
The [quick-start guide](docs/getting-started/quick-start.md) explains the products and equivalent Python API.
See [example verification](docs/examples/index.md#starter-workflow)
for the controlled-run record.

## Documentation

Follow the researcher workflow:

1. [Install](docs/getting-started/installation.md) and run the [quick start](docs/getting-started/quick-start.md).
2. [Generate synthetic observations](docs/user-guide/generation.md).
3. [Analyze data](docs/user-guide/analysis.md).
4. [Visualize and interpret](docs/user-guide/visualization.md).
5. [Reproduce examples](docs/examples/index.md), checking dependencies and data availability.

The [documentation home](docs/index.md) links scientific references and research notes.
Developers can start with [extension and validation guidance](docs/developer/index.md)
and the [architecture](docs/architecture/index.md).
The intended public site is [HS Observatory](https://kgklein.github.io/hs-observatory/);
**this will be available once the repo is made public.**

### Preview the site locally

From the repository root, using the `.venv` created above:

```bash
source .venv/bin/activate
python -m pip install -e '.[docs]'
python -m mkdocs serve --dev-addr 127.0.0.1:8000
```

Open <http://127.0.0.1:8000/hs-observatory/> in your browser (or follow the URL printed in the terminal).
Keep the command running while browsing; saved documentation edits reload automatically.
Press **Ctrl+C** to stop the preview.
The install command is only needed when setting up the documentation dependencies.
See [documentation development](docs/developer/documentation.md#preview-the-site-locally)
for more preview options and build checks.

## Citation, contributions and license

Please use [CITATION.cff](CITATION.cff) and cite the scientific methods used in your analysis.
See [CONTRIBUTING.md](CONTRIBUTING.md) for contributions and issue reports.
The software is distributed under the [MIT license](LICENSE).
