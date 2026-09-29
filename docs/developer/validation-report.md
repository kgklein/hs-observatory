# Documentation validation record

Validation date: 2026-09-28. Scientific source, packaging/dependencies, root
DESIGN/CONTRIBUTING and `visualization/` are unchanged. This record distinguishes
executed checks from historical numerical evidence. See the
[preservation audit](preservation-audit.md) and [scientific review](scientific-review.md).

## Controlled installation and README workflow

A temporary checkout at `/tmp/hsobs-docs-restructure/quickstart/checkout` contains
copies of tracked package/configuration files from the reviewed working tree.
Its relative external CDF directory is absent. No existing repository outputs
were overwritten. The venv was created with `python3 -m venv .venv`; it does not
inherit the developer environment's scientific extras.

First attempt:

```bash
source .venv/bin/activate
python -m pip install -e .
```

Installation succeeded. Generation then failed after writing HDF5 because
`plot_geometry_summary` imports cmcrameri without a fallback. Diagnostics could
read the saved HDF5, but that partial attempt was not a successful full workflow.
No code or packaging was changed to hide the failure.

The final README instructions were verified with:

```bash
source .venv/bin/activate
python -m pip install -e '.[analysis]'
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/hsobs-docs-restructure/mpl
hsobs-generate configs/example_generate.yaml
hsobs-analyze configs/example_analyze.yaml
```

Both commands exited successfully. Assertions checked truth n `(2401, 9)`, truth B
`(9601, 9, 3)`, observed B `(10, 9601, 9, 3)`, synthetic-fallback metadata and
36 pair baselines per variable. All 14 expected generation figures exist: six
primary PNGs plus four geometry-overview pairs of PDF/PNG. Matplotlib reported
aspect/axis-limit adjustment warnings; they did not prevent output.

Versions: Python 3.10; hs-observatory 0.1.0; NumPy 2.2.6; PyYAML 6.0.3;
h5py 3.16.0; cdflib 1.3.14; Matplotlib 3.10.9; SciPy 1.15.3; cmcrameri 1.10.
This verifies the selected environment, not every allowed dependency combination.

## Additional executed examples

Against that same isolated starter product:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/hsobs-docs-restructure/mpl .venv/bin/hsobs-analyze configs/example_elsasser.yaml
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/hsobs-docs-restructure/mpl .venv/bin/hsobs-analyze configs/example_timeseries_fft.yaml
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/hsobs-docs-restructure/mpl .venv/bin/hsobs-analyze configs/example_static_decorrelation.yaml
```

All three exited successfully and wrote their HDF5 products and configured
figures. These are smoke-workflow checks, not new independent scientific validation.
All twelve installed `hsobs-*` commands also exited successfully with `--help`;
entry points were compared with packaging and method selectors with the registry.

## Documentation and diagram checks

From the repository root:

```bash
.venv/bin/python -m unittest discover -s tests -p test_architecture_graph.py
.venv/bin/python -m ruff check scripts/generate_architecture_graph.py
.venv/bin/python scripts/generate_architecture_graph.py
.venv/bin/mkdocs build --strict
```

Focused architecture tests: **7 passed**. Targeted Ruff: **passed**. Both DOT/SVG
pairs regenerated: detailed graph 45 nodes and 112 edges; overview 15 groups and
22 source-witnessed relationships. Only the generator's grouping/selected edges
changed; no scientific module changed.

Build/link/rendering checks are finalized below after the final content pass.
Early strict builds caught the architecture README/index collision and an
in-progress validation-page link; both were corrected before final validation.

## What was not rerun

No historical divergence, affine, quadratic or timing-noise numerical experiment
was rerun. Their result tables, dates, reported test counts and warnings are
preserved historical evidence. External PLUME/Gkeyll/CDF workflows and expensive
Malara/PDF production cases were not executed. See the [manifest](../examples/index.md)
for all prerequisites, untracked configurations and the missing script.

No full-repository pytest, Ruff, mypy or pre-commit was run. No public deployment,
commit, push, dependency/packaging change or automatic tracking of scientific
research files was performed. External URLs and unavailable data provenance still
need release review; local link checks do not establish remote availability.
