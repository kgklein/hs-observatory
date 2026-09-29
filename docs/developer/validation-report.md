# Documentation validation record

Workflow validation: 2026-09-28; final review: 2026-09-29. Scientific source, packaging/dependencies, root
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

Final strict build: **passed**. The installed Material theme prints an upstream
MkDocs-2 announcement, but MkDocs reported no build warnings or errors.
The local checker also passed after resolving site-root URLs using the configured
`/hs-observatory/` prefix (the initial checker falsely treated the generated 404
page's absolute site URLs as filesystem-root paths).

```bash
.venv/bin/python /tmp/hsobs-docs-restructure/validate_docs.py
dot -Tpng docs/architecture/package-overview.dot -o /tmp/hsobs-docs-restructure/overview.png
```

The temporary checker parses built HTML and source Markdown, checks local file
and fragment targets, accounts for every navigation/compatibility page, compares
all original historical table rows and TeX against the pre-migration baseline,
parses SVGs, and hashes protected files. Results:

- 33 built HTML pages and 2,437 local HTML links checked.
- All 26 primary navigation pages accounted for; seven legacy compatibility pages preserved (the architecture README remains excluded from MkDocs to avoid its index collision).
- All 65 original historical table rows and 16 TeX occurrences preserved.
- All 18 SVG assets parse; figure references resolve; 55 HTML tables render.
- All 300 protected tracked files match the pre-migration working-tree hashes, including preexisting configuration edits.
- The preservation checklist covers 517 original source-topic blocks; every original nonblank Markdown line has a destination.

Visual review inspected the six display equations and the 15-area overview via
rendered previews: indices, signs, fractions and new timing/shock groups are
legible. The site HTML preserves tables and equation source disclosures. No
browser-based responsive/accessibility review or independent algebra validation
was performed. External URLs were not tested.

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

## Branch and working-tree disposition

At final review, local `main` was `272909e`. The previously saved documentation
commit `e485a41` was on `doc-update-260928` and its remote-tracking branch, directly
based on that main commit. Its approved documentation-only diff was applied to
the working tree after a clean applicability check; no branch was merged, commit
created, push performed or index staged by the assistant. Final validation-record
edits are additional uncommitted changes. Existing scientific configuration edits
and untracked research files remain untouched.

## File inventory

The following inventory describes the migration relative to its original source
layout. “Modified” includes legacy pages converted to compatibility pointers.
No file was physically deleted. Logical moves are documented in the preservation
matrix; seven old scientific/documentation authorities were retired in favor of
navigation-only pointers.

| Operation | Path |
| --- | --- |
| Modified | `README.md` |
| Modified | `docs/analysis_methods.md` |
| Modified | `docs/architecture/README.md` |
| Modified | `docs/architecture/index.md` |
| Modified | `docs/architecture/module-dependencies.dot` |
| Modified | `docs/architecture/module-dependencies.md` |
| Modified | `docs/architecture/module-dependencies.svg` |
| Modified | `docs/architecture/package-overview.dot` |
| Modified | `docs/architecture/package-overview.svg` |
| Created | `docs/assets/equations/display-64c93ae05b4f.svg` |
| Created | `docs/assets/equations/display-6ab4bb64361c.svg` |
| Created | `docs/assets/equations/display-a96142b5d083.svg` |
| Created | `docs/assets/equations/display-d6a06a5f5fae.svg` |
| Created | `docs/assets/equations/display-e5f19af99fb7.svg` |
| Created | `docs/assets/equations/display-e8d5d50f3641.svg` |
| Created | `docs/assets/equations/inline-1e0cee47a0fc.svg` |
| Created | `docs/assets/equations/inline-1fb17be39daa.svg` |
| Created | `docs/assets/equations/inline-274f8c02e874.svg` |
| Created | `docs/assets/equations/inline-546bb59f3cd0.svg` |
| Created | `docs/assets/equations/inline-6a20d80ef46f.svg` |
| Created | `docs/assets/equations/inline-71d4b938c50f.svg` |
| Created | `docs/assets/equations/inline-8336ca083169.svg` |
| Created | `docs/assets/equations/inline-92e42d69c507.svg` |
| Created | `docs/assets/equations/inline-e7596367e0aa.svg` |
| Created | `docs/assets/equations/inline-fbdcb41ac2fa.svg` |
| Created | `docs/assets/equations/sources.json` |
| Modified | `docs/developer/design.md` |
| Modified | `docs/developer/documentation.md` |
| Created | `docs/developer/index.md` |
| Created | `docs/developer/preservation-audit.md` |
| Created | `docs/developer/scientific-review.md` |
| Created | `docs/developer/validation-report.md` |
| Created | `docs/examples/index.md` |
| Modified | `docs/getting-started/installation.md` |
| Modified | `docs/getting-started/quick-start.md` |
| Modified | `docs/index.md` |
| Modified | `docs/magnetic_gradient_divergence_audit.md` |
| Modified | `docs/observatory_error_model.md` |
| Created | `docs/reference/configuration.md` |
| Created | `docs/reference/data-products.md` |
| Created | `docs/reference/errors.md` |
| Created | `docs/reference/fields.md` |
| Created | `docs/reference/magnetic-gradients.md` |
| Created | `docs/reference/reconstruction.md` |
| Created | `docs/reference/shock-timing.md` |
| Created | `docs/reference/statistical-methods.md` |
| Created | `docs/reference/timing-calibration.md` |
| Created | `docs/research-notes/magnetic-gradient-divergence-audit.md` |
| Created | `docs/research-notes/timing-investigations.md` |
| Modified | `docs/timing_affine_investigation.md` |
| Modified | `docs/timing_calibration.md` |
| Modified | `docs/timing_quadratic_investigation.md` |
| Created | `docs/user-guide/analysis.md` |
| Modified | `docs/user-guide/generation.md` |
| Created | `docs/user-guide/visualization.md` |
| Modified | `mkdocs.yml` |
| Modified | `scripts/generate_architecture_graph.py` |

Scientific references now live under `docs/reference/`; historical audit/timing
reports live under `docs/research-notes/`. Retired authorities retained as
compatibility files are `docs/analysis_methods.md`, `docs/observatory_error_model.md`,
`docs/timing_calibration.md`, `docs/timing_affine_investigation.md`,
`docs/timing_quadratic_investigation.md`,
`docs/magnetic_gradient_divergence_audit.md`, and `docs/architecture/README.md`.

Public-release requirements and unresolved scientific decisions remain in the
[example manifest](../examples/index.md) and [scientific checklist](scientific-review.md).
The only non-documentation source change is the scoped architecture grouping.
