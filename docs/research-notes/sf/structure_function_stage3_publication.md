# Stage 3: refined 40X structure-function publication figures

2026-10-14. The refined analysis uses existing 300-second observations, all 50
measurement-error realizations per case, and ideal sampled truth. No generator,
error model, original publication output, or 50X input was modified or processed.

## Configuration and reproducibility

New recipes are `analysis/malara-prod-{400,401,402}_sf-stage3-publication.yaml`.
They retain the original Elsasser velocity-frame correction, density
normalization, mean subtraction and timebase definitions. Accepted schema settings
are `all_spacetime_pairs`, `geometry_stratified`, orders 1–6, the explicit 26
Stage 2 edges, 100k targets in the 14 fitting bins and 10k in the other 12 bins,
seed 12345, and `primary: [150,1200]` / `sensitivity: [200,1000]` named ranges.
Each target is capped at its exact member/bin population. Original recipes remain
available unchanged. Plotting in the new analysis YAML is disabled; the separate
publication renderer creates the two-panel figures after saving moments.

Preflight checked each input's saved generation YAML, Malara model, p assignment
(400/401/402 = 0.9/0.7/0.5), isotropic geometry, n_scales=14, h=1/3,
L0=100000 km, injection index=2, CDF formation, 2401 timestamps over 300 seconds,
all 50 MC arrays and stored timing/position offsets. All three nominal trajectories
match the verified 402 trajectory. Common-velocity affine validation and exact
occupancy were checked separately for all observed position-offset geometries
and truth before analysis. Case 401 therefore has direct Stage 3 validation.

Observed coordinates still use nominal r plus the stored position offsets where
group-specific measured positions are absent. Fields retain the generator's
clock-shifted sampling; hidden timing displacements are not newly added to
separation labels. Each member uses its own eligible population with reproducible
common random streams; identical indices are not forced across different
position-error geometries. No selection depends on increment magnitude.

The runner invokes the existing `analyze_from_config` / `StructureFunctionAnalysis`
accumulator. JSON retains individual truth and observed S_n for both z+ and z−,
counts, exact populations/shortfalls, both-window raw exponents and local slopes.
Saved observed summaries are computed from individual MC members only; truth is
never included in MC percentiles. Percentile arrays are stored along with the
curves, not instead of them. Input-file SHA256, generation/config YAML hashes,
nominal position/time hashes, analysis/sampler source hashes, sampler version,
PCG64 seed, runtime and memory are recorded in `metadata.stage3` and existing
geometry-sampling diagnostics. Input hashes were checked again after analysis.

## Outputs and reproduction

All new outputs are under `outputs/structure_function_stage3/`:

- `malara-prod-{400,401,402}_sf_publication.json`: individual members, summaries,
  fits, local slopes, eligibility and provenance.
- `malara-prod-{400,401,402}_sf_publication.png` and `.pdf`: primary figures.
- `fit_window_sensitivity.png`, `.pdf` and `.json`: both fits for all cases.
- `local_slopes.h5`, `.png`, `.pdf`, and `local_slopes_focused.png` / `.pdf`:
  existing local-slope machinery, kept separate from publication SF panels.
- `plot_summary.json`: primary plotted values, common axis limits and result/
  renderer fingerprints; `validation.json` and `fitted_exponents.csv`: final QC
  and all orders/windows in a compact table.
- `original_artifacts_sha256.json`: hashes verifying original publication recipes
  and existing original JSON/PNG/PDF outputs were preserved.

Cached regeneration is the default and does not read observation HDF5 files:

```bash
# All three cases, primary fits, plus separate diagnostics:
bash scripts/structure_function_publication.sh --stage3 --diagnostics

# One cached figure, or a plot-time refit from saved curves:
bash scripts/structure_function_publication.sh --stage3 401
bash scripts/structure_function_publication.sh --stage3 --fit-window sensitivity 400 401 402
bash scripts/structure_function_publication.sh --stage3 --fit-lambda-range 200 1000 400

# Explicit opt-in to recalculate all 50 MC members plus truth, then plot:
bash scripts/structure_function_publication.sh --stage3 --recalculate 400 401 402
```

Sensitivity/custom-window figure names have `_fit_MIN_MAX` suffixes, preserving
the new primary figures. The legacy invocation without `--stage3` retains its
prior paths/options and performs no analysis. Stage 3 rejects non-40X case IDs.
For separate fresh-process analysis without intermediate rendering, use:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/sf-stage3-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m hs_observatory.analysis.structure_function_publication \
  --cases 400 --recalculate --analysis-only
```

The publication panels use common typography, dimensions and shared vertical
limits across the batch. The left axis stays at 70–20000 km, with primary fit
shading, open/solid truth and filled/dashed MC traces by order, plus light 5–95%
bands. The scale caption identifies ell_inj=25000 km as the largest nominal
cascade cell without asserting a measured spectral break. The right panel uses
raw xi_n and thin dashed analytic `1-log2[p^(n/3)+(1-p)^(n/3)]`; neither xi3 nor
the fitting interval is adjusted to enforce agreement. No local-slope lines or
extra theory curves are added to the SF panel.

## Runtime and memory

| Case / p | Analysis | Total | Peak RSS |
|---|---:|---:|---:|
| 400 / 0.9 | 206.3 s | 209.6 s | 584.9 MiB |
| 401 / 0.7 | 209.5 s | 212.9 s | 585.9 MiB |
| 402 / 0.5 | 203.9 s | 207.2 s | 588.8 MiB |

Analyses ran in separate processes, one case at a time with one OpenBLAS/OMP
thread. Analysis time includes the original full-ensemble loading/preparation,
existing accumulator and initial serialization. Total additionally includes
preflight, postprocessing and input rehashing; final rendering is separate.
RSS is the process high-water mark, not an allocation-only metric. This is a
measured production-size 40X run, not an extrapolation from Stage 2.

## Primary exponents and observed-minus-truth differences

Every value below uses the same 150–1200 km window (14 geometric bin centers)
for truth and each individual MC member. The bracket is MC 5th–95th percentile;
delta is MC median minus truth. These bands summarize the supplied measurement
ensemble with its geometry-dependent finite pair selection, not independent
field-realization uncertainty.

### 400, p=0.9

| n | Ideal truth | MC median [5th, 95th] | Median − truth |
|---|---:|---:|---:|
| 1 | 0.46537 | 0.40038 [0.35560, 0.42951] | -0.06499 |
| 2 | 0.81305 | 0.73457 [0.65857, 0.78476] | -0.07848 |
| 3 | 1.07597 | 1.01938 [0.92911, 1.07471] | -0.05659 |
| 4 | 1.28854 | 1.24665 [1.16508, 1.30516] | -0.04189 |
| 5 | 1.47358 | 1.44710 [1.35537, 1.51189] | -0.02648 |
| 6 | 1.64512 | 1.63181 [1.52513, 1.70828] | -0.01332 |

### 401, p=0.7

| n | Ideal truth | MC median [5th, 95th] | Median − truth |
|---|---:|---:|---:|
| 1 | 0.36257 | 0.34693 [0.32916, 0.35839] | -0.01564 |
| 2 | 0.70625 | 0.67934 [0.64366, 0.69963] | -0.02691 |
| 3 | 1.03284 | 0.99703 [0.94597, 1.02650] | -0.03580 |
| 4 | 1.34435 | 1.30069 [1.23922, 1.34364] | -0.04366 |
| 5 | 1.64338 | 1.59510 [1.52485, 1.64805] | -0.04827 |
| 6 | 1.93275 | 1.87461 [1.80456, 1.94072] | -0.05814 |

### 402, p=0.5

| n | Ideal truth | MC median [5th, 95th] | Median − truth |
|---|---:|---:|---:|
| 1 | 0.33238 | 0.32309 [0.30892, 0.33092] | -0.00929 |
| 2 | 0.66588 | 0.64479 [0.61528, 0.66145] | -0.02109 |
| 3 | 0.99996 | 0.96521 [0.91847, 0.99274] | -0.03475 |
| 4 | 1.33332 | 1.28414 [1.21768, 1.32333] | -0.04918 |
| 5 | 1.66418 | 1.60021 [1.51198, 1.65173] | -0.06397 |
| 6 | 1.99052 | 1.91421 [1.80041, 1.97720] | -0.07631 |


The analytic p-model remains a separate prediction. Differences between it,
ideal sampled-field statistics and instrument-error medians remain visible; the
analysis does not infer convergence from theoretical agreement.

The supplied MC ensemble has systematically lower median slopes than ideal
truth. Relative median differences span −14.0% to −0.81% for 400, −4.31% to
−2.94% for 401, and −3.84% to −2.79% for 402. Truth lies outside the MC 5–95%
band at orders 1–3 for 400, 1–4 for 401, and 1–6 for 402. These are distributions
of instrument-error realizations, not confidence intervals guaranteed to cover
truth. The figures quantify recovery and residual offsets; they do not establish
unbiased recovery at every order. This run does not isolate which error component
causes each offset, and applies no empirical correction to remove it.

## Window sensitivity and local slopes

The 200–1000 km fit uses 11 centers. The following shifts are sensitivity minus
primary, in order n=1,…,6. All raw values and percentiles are also saved in the
sensitivity JSON and CSV; plotting/refitting requires no new pairs.

| Case | Truth shift (n=1,…,6) | Change of MC medians (n=1,…,6) |
|---|---|---|
| 400 | `-0.00643, -0.00908, -0.01053, -0.01865, -0.03560, -0.05793` | `-0.00173, +0.00002, -0.00493, -0.00822, -0.02810, -0.04993` |
| 401 | `-0.00491, -0.00834, -0.00890, -0.00700, -0.00437, -0.00310` | `-0.00355, -0.00596, -0.00449, -0.00212, +0.00025, +0.00552` |
| 402 | `-0.00476, -0.00717, -0.00722, -0.00582, -0.00440, -0.00466` | `-0.00427, -0.00534, -0.00391, -0.00190, +0.00097, +0.00229` |

Window dependence should be retained as a scientific sensitivity, especially for
high orders. The local-slope products use existing three-point log fits for
individual realizations followed by observed-member percentiles. Endpoints
remain NaN and no gaps are bridged or extrapolated. They are sampling/finite-window
diagnostics, not additional lines in the publication figures.

## Occupancy, checks and limitations

| Case | Truth selected / eligible (all bins) | MC selected total min–max | Occupied bins, each product |
|---|---:|---:|---:|
| 400 | 1,236,311 / 101,869,766 | 1,233,921–1,241,095 | 26/26 |
| 401 | 1,236,311 / 101,869,766 | 1,233,921–1,241,095 | 26/26 |
| 402 | 1,236,311 / 101,869,766 | 1,233,921–1,241,095 | 26/26 |

Truth populations and retained counts match the Stage 2 occupancy table. Seven
fitting bins, 150–424.26 km, use censuses in all three truth products and
all 150 MC members; the other fitting bins retain 100k and all outside bins
retain 10k. Population changes between observed members are preserved.

All 26 bins are populated for truth and every MC realization. Population
shortfalls mean a requested target exceeds availability; the sampler uses a
census in those bins rather than repeatedly attempting an impossible quota.
Exact per-member populations and retained counts are in each JSON. Small bins
have few distinct baseline/lag groups despite many pair-time identities; counts
are not independent-sample counts. The large-scale shape uses only 10k pairs/bin,
so high-order tails there have more subsampling uncertainty than fitting bins.

For both z+ and z−, the 400 and 402 ideal-truth curves replay Stage 2 seed
12345 at 100k/10k. Maximum relative moment differences are 4.4e-16 and 4.4e-16,
respectively; both-window exponents agree to relative tolerance 1e-12. This is
stronger than agreement within the previously measured five-seed sampling
variation. Case 401 has no Stage 2 replay target; it passed the full occupancy,
geometry, MC-count and saved-fit checks here.

Stage 2's five-seed evidence supports the fitting-bin budget conditionally on
these time series; it does not prove high-order convergence across longer
durations, additional field realizations, or different trajectories. MC bands
are computed over instrument-error members only; independent pair-seed variation
from Stage 2 is not added to those bands. Analytic p-model predictions are not
substitutes for the measured sampled-field exponents.

Validation includes exact populations/retention, every member's fit-bin counts
(14 primary / 11 sensitivity), raw analytic curves, stored MC summaries versus
individual-member percentiles, input/config fingerprints, Stage 2 truth replay
and preservation of original artifacts. Existing plotting/analysis, geometry
sampling and local-slope tests passed: **85 focused tests**. The final rendering
changes passed the eight publication workflow tests again. Targeted ruff checks
passed for the added Python module/test file, and shell syntax passed. No
full-repository test/lint run was used.
