# Reproducible examples and release availability

Use this manifest before running a command from a reference page. A configuration
path is not proof that its external data or generated products are distributed.
Run available examples from the repository root. The workflow is
[install](../getting-started/installation.md) → [generate](../user-guide/generation.md)
→ [analyze](../user-guide/analysis.md) → [visualize](../user-guide/visualization.md).

## Status labels

- **Self-contained and verified**: the exact workflow was executed during this migration in an isolated checkout/environment.
- **Requires optional dependencies**: an extra is required; installation alone does not supply data or establish execution success.
- **Requires external data**: user-supplied mission, PLUME or simulation inputs, or previously generated products are prerequisites.
- **Research-only or dependent on untracked files**: local cases are not a public checkout contract.
- **Currently incomplete or unavailable**: a required reproduction component is absent.

Labels may overlap. Self-contained but unexecuted examples are explicitly marked
**not yet verified**, rather than being counted as verified.

## Starter workflow

**Self-contained and verified**, with the existing `.[analysis]` extra for full
figures. On 2026-09-28, the exact commands completed in an isolated temporary
checkout and fresh Python 3.10 environment with no configured CDF input present.
Verified: truth n `(2401, 9)`, truth B `(9601, 9, 3)`, observed B
`(10, 9601, 9, 3)`, synthetic-fallback provenance, 36 diagnostic pairs per
quantity and all 14 generation figure files. Matplotlib emitted aspect/limit
adjustment warnings; both commands succeeded.

A preceding base-only installation exposed a `cmcrameri` import failure during
geometry plotting after HDF5 output had been written. The README now installs
`.[analysis]`; no packaging or scientific code was changed. See the
[validation record](../developer/validation-report.md).

```bash
hsobs-generate configs/example_generate.yaml
hsobs-analyze configs/example_analyze.yaml
```

The first command creates `outputs/synthetic_observations.h5` with seed 12345,
ten MC members, Gaussian errors and native cadences (n/v: 0.125 s; B: 0.03125 s).
The second reads that file and creates `outputs/diagnostics.json`. Generation
also saves six time-series/geometry/Alfvén PNGs and geometry-overview PDF/PNG
products. The existing `.[analysis]` extra is required for the current default geometry
figures. The synthetic CDF fallback removes a mission-file requirement, but if
the configured CDF directory exists the geometry can differ. Preserve source
metadata, not just the random seed. See the [quick start](../getting-started/quick-start.md).

## Analytic statistics and error examples

These configurations are tracked and use analytic fields or starter products;
only rows explicitly marked verified have been executed for this migration.

| Example | Generation / analysis chain | Outputs and status |
| --- | --- | --- |
| OU measurement errors (default figures need `.[analysis]`) | `configs/example_generate_ou.yaml` | Self-contained via fallback; not yet verified. Field/error product path is in YAML. |
| Malara turbulence (default figures need `.[analysis]`) | `configs/example_generate_malara.yaml` | Self-contained via fallback; not yet verified; runtime grows with samples/scales/MC. |
| Elsasser quantities | Starter → `configs/example_elsasser.yaml` | `outputs/elsasser.h5`; **self-contained and verified** after the starter. |
| Static increment PDFs | Starter → `configs/example_elsasser_increment_pdf_static.yaml` | HDF5/PDF settings in YAML; all-pair computation may be expensive; not yet verified. |
| Spatial decorrelation | Starter → `configs/example_static_decorrelation.yaml` | HDF5 and diagnostic PNG; **self-contained and verified** after the starter. |
| FFT spectra | Starter → `configs/example_timeseries_fft.yaml` | HDF5, observed/truth PNGs; **self-contained and verified** after the starter. |
| Structure functions | `configs/example_structure_function.yaml` and research case YAML | Research-only: this example configuration is untracked in the reviewed tree. |

For each row, generation uses `hsobs-generate CONFIG` and analysis uses
`hsobs-analyze CONFIG`. Check the actual input/output paths before composing a
chain; do not assume every research configuration reads the starter output.
Parameter definitions and assumptions are in [statistical methods](../reference/statistical-methods.md)
and [errors](../reference/errors.md).

## Shock workflow

**Requires optional dependencies; not executed in this migration.** Install
`.[analysis]` for the SciPy fitter. Both configurations are tracked.

```bash
hsobs-generate configs/example_generate_shock.yaml
hsobs-analyze configs/example_analyze_shock_timing.yaml
```

The observation is `outputs/synthetic_observations_shock.h5`; analysis writes
`outputs/shock_timing.h5` and detection, accuracy, geometry/reliability and
best/worst tetrahedron PDF/PNG figures. See [shock timing](../reference/shock-timing.md)
and its [physical field](../reference/fields.md#propagating-planar-mhd-shocks).
Synthetic geometry fallback does not remove the SciPy requirement for analysis.

## PLUME workflow

**Requires external data; not executed.** `configs/example_generate_plume_single_mode.yaml`
references a local `~/Codes/PLUME/...mode1` file. Supply that file, verify component
count and row indexing, and review [PLUME geometry/amplitude questions](../developer/scientific-review.md)
before generation. The example has preexisting local edits preserved by this
migration; no external file is bundled or published here.

## Simulation and gradient workflows

**Requires external data and optional dependencies; not executed.**
`configs/example_generate_simulation_sampled.yaml` and
`configs/example_generate_simulation_sampled_observatory.yaml` require the split
Gkeyll BP snapshots under `../gkeyll/data/` and `.[simulation-bp]`. The gradient
example reads the **observatory** generation product, not the other variant:

```bash
hsobs-generate configs/example_generate_simulation_sampled_observatory.yaml
hsobs-analyze configs/example_magnetic_gradient.yaml
hsobs-plot-magnetic-gradient outputs/magnetic_gradients.h5 --output outputs/magnetic_gradients.png
```

This is a conditional recipe, not a verified public-data example. Source simulation
files remain needed for gradient truth; plotting subsequently needs only the saved
gradient product. Preserve embedded simulation layout, scaling and mapping and use
`simulation_override` for relocated inputs.

The [dated divergence audit](../research-notes/magnetic-gradient-divergence-audit.md)
needs its named sim-prod-400 products, source snapshot and SciPy. Its numerical
tables are historical, not newly reproduced. RBF configuration
`configs/example_rbf_reconstruction.yaml` and the standalone
[reconstruction helpers](https://github.com/kgklein/hs-observatory/blob/main/visualization/README.md)
likewise need the input products named in their configuration.

## Publication postprocessing

**Requires generated products; not executed.** These tracked wrappers/configurations
are available but do not distribute their input HDF5/JSON results:

| Task | Recipe/configuration |
| --- | --- |
| Gradient Wasserstein | `scripts/plot_sim_prod_2xx_magnetic_gradient_wasserstein.sh`; `analysis/sim-prod-2xx_magnetic-gradient-wasserstein.yaml` |
| Combined gradient figure | `scripts/plot_sim_prod_4xx_magnetic_gradient_combined.sh`; `analysis/sim-prod-4xx_magnetic-gradient-combined.yaml` |
| Gradient correlation | `scripts/run_sim_prod_400_magnetic_gradient_correlation.sh`, then `scripts/plot_sim_prod_400_magnetic_gradient_correlation.sh`; `analysis/sim-prod-400_magnetic-gradient-correlation.yaml` |
| Structure-function summary/overlay | Saved JSON and `hsobs-plot-structure-function` / `hsobs-plot-structure-function-overlay` |
| Increment-PDF publication/distance | Saved HDF5 cases and `hsobs-plot-increment-pdf` / `hsobs-plot-increment-pdf-wasserstein` |
| Figure 2 | Malara/simulation HDF5 and `hsobs-plot-figure2`; use `--input-units repo` for native products. |

The method references preserve the original commands, selection rules, plot
intervals and diagnostic definitions. Missing comparison cases may be skipped by
some plotters; a partially populated figure is not full reproduction.

## Timing research

**Research-only or dependent on untracked files.** The following referenced
configurations exist locally but are not tracked in the reviewed working tree:

- `generation/sim-cali-timing.yaml`, `generation/sim-cali-len01.yaml` and related `sim-cali-*` generation cases.
- `analysis/sim-cali-timing_timing_calibrate.yaml`.
- `analysis/sim-cali-timing-null0_timing_influence.yaml` and related null/subset influence cases.
- `analysis/sim-cali-compare-polyhedron.yaml`, `analysis/sim-cali-compare-divergence.yaml`, `analysis/sim-cali-compare-df-affine.yaml`, `analysis/sim-cali-compare-df-quadratic.yaml`.
- `analysis/sim-cali-compare-instrument-noise.yaml` and `configs/helioswarm_mag_scm_noise.yaml`.

The [timing reference](../reference/timing-calibration.md) retains conditional
command recipes and exact algorithms; a tracked checkout alone cannot run those
case recipes. No untracked scientific file was automatically added to version control.

**Currently incomplete or unavailable:** `analysis/timing_noise_experiment.py`
is absent. Its original command is historical text only. The
[combined investigations](../research-notes/timing-investigations.md) preserve
separate affine and quadratic fixtures, tables, warnings and artifact descriptions,
and the instrument-noise experiment. Their old test counts and numerical results
were not rerun during this migration.

## Public reproduction checklist

Before selecting a research case for release, track the approved configuration
chain, restore or replace the missing experiment script with reviewed provenance,
and provide a durable data location/license/checksum for required CDF, PLUME and
Gkeyll inputs. Record the software commit, Python/dependency versions, seeds,
geometry source and complete generated artifact manifest. Generated `outputs/`
are ignored; named historical results are not automatically downloadable.

The current [scientific checklist](../developer/scientific-review.md) records
normalization and interpretation questions that documentation cannot resolve.
