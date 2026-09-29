# HelioSwarm Synthetic Observatory

HS Observatory generates synthetic multipoint spacecraft observations and analyzes
the resulting products, supporting studies of space-plasma structure, turbulence,
and HelioSwarm-like sampling. YAML configurations specify physical fields,
spacecraft geometry, native instrument cadences, and coordinate/measurement errors.
HDF5 products retain nominal truth, Monte Carlo observations and provenance.

Field, error and analysis models provide specific locations for growth through
community input. Scientific assumptions and validation limits are documented per
method; timing calibration remains research methodology.

## Install

Use Python **3.10 or newer** and a local checkout of this repository. From its root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[analysis]'
```

The existing `analysis` extra is needed by the current default geometry figures
(`cmcrameri`) and also supplies SciPy. The activation command is for a POSIX shell. See [installation](docs/getting-started/installation.md)
for checkout instructions, other environments, optional dependencies and external inputs.

## Quick start

Run from the repository root in the activated environment:

```bash
hsobs-generate configs/example_generate.yaml
hsobs-analyze configs/example_analyze.yaml
```

The sine-sum example enables synthetic geometry fallback, so mission CDF files
are not required. It writes `outputs/synthetic_observations.h5`, diagnostic
figures under `outputs/`, and `outputs/diagnostics.json`. If the configured CDF
directory exists, its geometry may be used instead; inspect the saved provenance.
The [quick-start guide](docs/getting-started/quick-start.md) explains the products
and equivalent Python API. See [example verification](docs/examples/index.md#starter-workflow)
for the controlled-run record.

## Documentation

Follow the researcher workflow:

1. [Install](docs/getting-started/installation.md) and run the [quick start](docs/getting-started/quick-start.md).
2. [Generate synthetic observations](docs/user-guide/generation.md).
3. [Analyze data](docs/user-guide/analysis.md).
4. [Visualize and interpret](docs/user-guide/visualization.md).
5. [Reproduce examples](docs/examples/index.md), checking dependencies and data availability.

The [documentation home](docs/index.md) links scientific references and research
notes. Developers can start with [extension and validation guidance](docs/developer/index.md)
and the [architecture](docs/architecture/index.md). The intended public site is
[HS Observatory](https://kgklein.github.io/hs-observatory/); this change does not
publish it. [Local preview instructions](docs/developer/documentation.md) are available.

## Citation, contributions and license

Please use [CITATION.cff](CITATION.cff) and cite the scientific methods used in your
analysis. See [CONTRIBUTING.md](CONTRIBUTING.md) for contributions and issue reports.
The software is distributed under the [MIT license](LICENSE).
