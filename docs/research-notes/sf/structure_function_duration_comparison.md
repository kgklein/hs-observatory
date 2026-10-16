# Malara observation-duration convergence: p=0.9 and p=0.5

2026-10-14. Completed truth-only duration studies for cases 500 and 502, with
five durations (300–1500 s), five pair seeds, uniform 40-bin separations over
70–20000 km, primary fits 150–1200 km and sensitivity fits 200–1000 km.
Targets are 100000 pairs per fitting bin and 10000 outside; the 300/1500 s
verification uses 500000 per fitting bin, still 10000 outside. Every target
is capped at its eligible population. No fields, errors or MC members were generated.

## Findings and recommendation

**Longer sampling changes the p=0.9 statistics substantially and weakens the
original ~1000-km spike, but does not recover the analytic p-model exponents.**
For p=0.5, primary fits approach the expected values, with smaller but measurable
remaining duration/window dependence. These conclusions concern one periodic
field realization, not convergence of an ensemble of fields.

- **p=0.9:** baseline primary xi6 is 1.654 at 300 s, 1.651 at 600 s, then
  2.087 / 2.062 / 2.078 at 900 / 1200 / 1500 s. The 600-to-900 change (~0.436)
  greatly exceeds pair-seed scatter. At the higher budget, the endpoint values
  are **1.6609 ± 0.0039** and **2.0683 ± 0.0080**, compared with analytic
  zeta6=1.2863. Increasing duration therefore does not monotonically improve
  theory agreement. The late xi6 values are broadly similar, but lower orders
  still drift (baseline xi3: 1.250 at 900 s versus 1.211 at 1500 s), and a
  converged scaling regime has not been established.
- **p=0.5:** higher-budget endpoint xi6 changes from **2.0377 ± 0.0014** to
  **1.9899 ± 0.0007**, close to analytic zeta6=2. The primary baseline xi6
  values stay within ~0.007 after 600 s. However, the 1500-s sensitivity fit is
  1.9649 ± 0.0011, so finite-window differences remain larger than pair noise.
  Closeness to theory does not by itself establish convergence.
- **Original feature:** for p=0.9, the higher-budget sixth-order local slope
  at the same 957-km center decreases from **3.946 ± 0.072** to
  **1.443 ± 0.075**. The broader local-slope curves retain strong structure:
  at ~410 km the 1500-s higher-budget slope is **4.746 ± 0.102**. Thus the
  particular 1000-km spike weakens, while strong scale dependence persists at
  other separations. Do not infer a physical break from either feature.
- **Budget adequacy:** at 1500 s, p=0.9 xi6 seed SD drops from 0.0648 at 100k
  to 0.0080 at 500k. The baseline is adequate to detect the large duration
  change, but not to establish precise high-order late-duration stability.
  Prefer 500k fitting-bin targets for further p=0.9 duration comparisons and
  verify the 900/1200-s cases before making a late-duration convergence claim.
  These two additional high-budget cases were not run in this study.

The uncertainties above are sample SD across five pair seeds. They do not
include finite-duration, independent-field, model or measurement-error
uncertainty. The high-budget primary endpoint change for p=0.9 is +0.4075,
whereas changing the sampling budget at a fixed endpoint shifts the mean by
only +0.0069 (300 s) or -0.0101 (1500 s). This rules out pair subsampling as
the main explanation for that large endpoint difference. Population mean-log
centers change the p=0.9 1500-s higher-budget primary xi6 to ~2.053, still far
from theory; the established geometric-center convention remains unchanged.

The 1500-s path is about 399839 km, with coordinate traversals
(1.575, 0.030, 3.675) domain lengths. Sampling remains a translated formation
in a single periodic volume, with very limited traversal along y. These runs
show a real conditional duration effect, not independent spatial/field coverage
or complete recovery of the analytic scaling law.

## Input and execution checks

Both sets pass nested-prefix validation. Timestamps and nominal coordinates
match exactly across p at every duration; physical field parameters match
except for intermittency p. Both use field seed 12345, realization 1, domain
100000 km per axis and cascade indices 2–14 inclusive. Minor within-sequence
floating-point truth differences are below the saved numerical tolerances.
The transferred 500 HDF5 files lacked local generation recipes: those YAMLs
were restored verbatim from their embedded metadata, not regenerated or inferred.
Cross-sequence checks are saved in
`outputs/structure_function_duration_comparison/cross_sequence_validation.json`.

All 35 existing 502 seed/budget products were reused. The 35 case-500 products
were calculated in this run. Each set has 25 baseline and 10 higher-budget
products, both Elsasser variables, all six orders and both fitting windows.
Counts/populations/group counts agree across p for matched duration/budget.
The same seed and geometry give matching pair identities across p; across
durations the populations differ and identities are not forced equal.

## Primary exponents

Numbers are means ± sample SD over pair seeds, not measurement-error bands
or independent-field uncertainty. All results use geometric bin centers.

| p | Duration s | xi3 | xi4 | xi6 |
|---:|---:|---:|---:|---:|
| 0.9 | 300 | 1.0860 ± 0.0033 | 1.2964 ± 0.0063 | 1.6540 ± 0.0157 |
| 0.9 | 600 | 1.0781 ± 0.0035 | 1.2904 ± 0.0064 | 1.6513 ± 0.0134 |
| 0.9 | 900 | 1.2500 ± 0.0026 | 1.5499 ± 0.0038 | 2.0870 ± 0.0154 |
| 0.9 | 1200 | 1.2252 ± 0.0046 | 1.5201 ± 0.0091 | 2.0615 ± 0.0254 |
| 0.9 | 1500 | 1.2115 ± 0.0081 | 1.5197 ± 0.0192 | 2.0785 ± 0.0648 |
| 0.5 | 300 | 1.0208 ± 0.0007 | 1.3613 ± 0.0009 | 2.0359 ± 0.0018 |
| 0.5 | 600 | 1.0100 ± 0.0012 | 1.3417 ± 0.0016 | 1.9962 ± 0.0030 |
| 0.5 | 900 | 1.0053 ± 0.0004 | 1.3382 ± 0.0008 | 1.9970 ± 0.0020 |
| 0.5 | 1200 | 1.0025 ± 0.0018 | 1.3345 ± 0.0025 | 1.9930 ± 0.0041 |
| 0.5 | 1500 | 0.9995 ± 0.0016 | 1.3306 ± 0.0015 | 1.9900 ± 0.0030 |

## Endpoint and fit-window comparisons

| p | Duration s | xi6 primary | xi6 sensitivity | Sensitivity − primary |
|---:|---:|---:|---:|---:|
| 0.9 | 300 | 1.6540 | 1.5861 | -0.0679 |
| 0.9 | 600 | 1.6513 | 1.5701 | -0.0811 |
| 0.9 | 900 | 2.0870 | 2.0614 | -0.0256 |
| 0.9 | 1200 | 2.0615 | 2.0432 | -0.0183 |
| 0.9 | 1500 | 2.0785 | 2.0820 | +0.0035 |
| 0.5 | 300 | 2.0359 | 2.0470 | +0.0111 |
| 0.5 | 600 | 1.9962 | 1.9540 | -0.0422 |
| 0.5 | 900 | 1.9970 | 1.9781 | -0.0189 |
| 0.5 | 1200 | 1.9930 | 1.9787 | -0.0144 |
| 0.5 | 1500 | 1.9900 | 1.9651 | -0.0249 |

At 1500 s, compare all measured primary exponents with the analytic reference:

| p | n | Measured mean ± seed SD | Analytic zeta | Measured − analytic |
|---:|---:|---:|---:|---:|
| 0.9 | 1 | 0.4830 ± 0.0010 | 0.4843 | -0.0014 |
| 0.9 | 2 | 0.8710 ± 0.0028 | 0.8014 | +0.0696 |
| 0.9 | 3 | 1.2115 ± 0.0081 | 1.0000 | +0.2115 |
| 0.9 | 4 | 1.5197 ± 0.0192 | 1.1276 | +0.3921 |
| 0.9 | 5 | 1.8045 ± 0.0379 | 1.2168 | +0.5877 |
| 0.9 | 6 | 2.0785 ± 0.0648 | 1.2863 | +0.7922 |
| 0.5 | 1 | 0.3344 ± 0.0008 | 0.3333 | +0.0010 |
| 0.5 | 2 | 0.6675 ± 0.0014 | 0.6667 | +0.0008 |
| 0.5 | 3 | 0.9995 ± 0.0016 | 1.0000 | -0.0005 |
| 0.5 | 4 | 1.3306 ± 0.0015 | 1.3333 | -0.0028 |
| 0.5 | 5 | 1.6607 ± 0.0016 | 1.6667 | -0.0060 |
| 0.5 | 6 | 1.9900 ± 0.0030 | 2.0000 | -0.0100 |

Agreement with theory is a separate test from stability with duration.

## High-order local-slope feature

Peaks below are the mean ± sample SD of per-seed maxima in 700–1500 km.
The center-specific slope at 957.1 km is also listed to avoid confusing a moving
maximum with the same feature. These maxima alone do not identify a physical scale.

| p | Duration s | Peak xi3 | Peak xi4 | Peak xi6 | xi6 at 957 km | xi6 peak-location range km |
|---:|---:|---:|---:|---:|---:|---:|
| 0.9 | 300 | 1.611 ± 0.051 | 2.324 ± 0.066 | 3.952 ± 0.074 | 3.952 ± 0.074 | 957.1–957.1 |
| 0.9 | 600 | 1.583 ± 0.059 | 2.294 ± 0.096 | 3.893 ± 0.215 | 3.893 ± 0.215 | 957.1–957.1 |
| 0.9 | 900 | 1.580 ± 0.065 | 1.924 ± 0.178 | 2.216 ± 0.915 | 1.752 ± 0.275 | 957.1–1462.7 |
| 0.9 | 1200 | 1.469 ± 0.192 | 1.779 ± 0.350 | 2.049 ± 0.630 | 1.567 ± 0.697 | 957.1–1462.7 |
| 0.9 | 1500 | 1.591 ± 0.156 | 1.965 ± 0.318 | 2.562 ± 0.422 | 1.461 ± 0.618 | 1102.5–1462.7 |
| 0.5 | 300 | 1.060 ± 0.022 | 1.393 ± 0.047 | 2.085 ± 0.106 | 1.955 ± 0.066 | 957.1–1462.7 |
| 0.5 | 600 | 1.105 ± 0.016 | 1.466 ± 0.037 | 2.244 ± 0.107 | 2.203 ± 0.068 | 957.1–1102.5 |
| 0.5 | 900 | 1.053 ± 0.051 | 1.403 ± 0.086 | 2.119 ± 0.170 | 2.056 ± 0.057 | 957.1–1462.7 |
| 0.5 | 1200 | 1.024 ± 0.024 | 1.351 ± 0.037 | 2.006 ± 0.059 | 1.988 ± 0.057 | 957.1–1462.7 |
| 0.5 | 1500 | 1.057 ± 0.030 | 1.426 ± 0.049 | 2.207 ± 0.089 | 2.130 ± 0.032 | 957.1–1462.7 |

## Pair-budget verification

| p | Duration s | Higher-budget xi6 primary ± SD | Higher − baseline xi6 ± paired-seed SD | Higher-budget xi6 at 957 km ± SD |
|---:|---:|---:|---:|---:|
| 0.9 | 300 | 1.6609 ± 0.0039 | +0.0069 ± 0.0135 | 3.946 ± 0.072 |
| 0.9 | 1500 | 2.0683 ± 0.0080 | -0.0101 ± 0.0697 | 1.443 ± 0.075 |
| 0.5 | 300 | 2.0377 ± 0.0014 | +0.0018 ± 0.0025 | 1.957 ± 0.026 |
| 0.5 | 1500 | 1.9899 ± 0.0007 | -0.0001 ± 0.0029 | 2.118 ± 0.014 |

Local regression at ~1102 km includes the next bin outside the primary fit.
Outside-bin targets remain 10000 in the verification: the higher budget does
not independently establish convergence of those outside moments or of maxima
near 1269/1463 km. Pair seeds describe subsampling only; neighboring increments
and durations share samples and the same periodic field.

## Spatial coverage

Geometry is identical across p. Values below are from the baseline 502 products.
Voxel revisitation is a coarse, formation-dependent proxy, not independent volume coverage.

| Duration s | Mean path km | Traversals x/y/z in domain lengths | Decimated voxel repeat fraction | Eligible pairs | Selected pairs |
|---:|---:|---|---:|---:|---:|
| 300 | 79968 | 0.315 / 0.006 / 0.735 | 0.536 | 101,869,766 | 1,406,938 |
| 600 | 159936 | 0.630 / 0.012 / 1.470 | 0.584 | 218,260,166 | 1,665,816 |
| 900 | 239904 | 0.945 / 0.018 / 2.205 | 0.571 | 334,650,566 | 1,714,770 |
| 1200 | 319871 | 1.260 / 0.024 / 2.940 | 0.578 | 451,040,966 | 1,736,370 |
| 1500 | 399839 | 1.575 / 0.030 / 3.675 | 0.582 | 567,431,366 | 1,750,000 |

The field wraps periodically but the SF estimator retains unwrapped separations.
Multiple axis traversals do not imply multiple independent field realizations.
The 64-cell-per-axis voxel proxy includes nearby spacecraft occupying the same
cells; its repeat fraction must not be read as a true return probability.

## Saved products and reproduction

- p=0.9: `outputs/structure_function_duration_convergence_500/` and
  `docs/structure_function_duration_convergence_500.md` (with CSV).
- p=0.5: `outputs/structure_function_duration_convergence/` and
  `docs/structure_function_duration_convergence.md` (with CSV).
- Each output directory includes `derived.json`, `manifest.json`, `validation.json`,
  individual seed moments, and PNG/PDF versions of `exponents_vs_duration`,
  `local_slopes`, `structure_functions`, and `duration_summary`.
- Combined numeric summary: `outputs/structure_function_duration_comparison/summary.json`.

```bash
# Run/resume the same studies:
bash scripts/structure_function_duration.sh --config analysis/malara-ts-500_sf-duration.yaml --verify-budget
bash scripts/structure_function_duration.sh --verify-budget
# Refit/render saved moments only:
bash scripts/structure_function_duration.sh --config analysis/malara-ts-500_sf-duration.yaml --plot-only
bash scripts/structure_function_duration.sh --plot-only
```

Per-seed saved timings (sum excludes outer validation, rendering and final report work):

| Case | Sum of saved seed timings s | Maximum saved process RSS MiB |
|---|---:|---:|
| 500 | 278.9 | 209.9 |
| 502 | 156.5 | 209.2 |

502 timings describe the earlier cached calculations, not the current replot runtime.
No statistical Python implementation changed for this execution. New 500 analysis
configuration and restored generation recipes were validated by the production
preflight, and all per-seed capped counts, five-seed coverage and common fit
windows were checked after completion.
