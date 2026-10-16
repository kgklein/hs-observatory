# Stage 2: truth-only binning and sampling convergence

2026-10-14. **Use 100,000 pairs per 150–1200 km bin and 10,000 outside,
capped at the exact population, as the common pilot budget.** This is the best
tested budget for the intermittent 400 case: observed between-seed CV is below
4.54% for every S4–S6 fitting bin, and the SD of xi6 is 0.0035. Keep 150–1200 km
as the primary **finite-window** fit and report 200–1000 km alongside it. The
400 xi6 window sensitivity is about 3%, larger than subsampling variability.
These are conditional, five-seed precision results, not proof of an inertial-range
plateau or convergence across time duration and independent field realizations.

## Inputs, recipes and safeguards

Both cases use the existing corrected CDF-based observations, all 2401 timestamps
over 300 seconds, and ideal sampled truth only. Nominal coordinates/timestamps
were checked to be identical between 400 and 402. Both aligned position arrays
have SHA256 `a2e5b6a8f72b6985f0c9ff3572c68fcfc3e53dfb33d670f0f550507c90a5a980`.
No field, trajectory, error, or measurement-error MC ensemble was generated.
The superseded test-configuration 400 results were archived under
`outputs/structure_function_stage2/superseded_400_wrong_configuration/`; they are
excluded from every number and figure below.

Added files:

- `analysis/malara-prod-{400,401,402}_sf-stage2-pilot.yaml`: explicit 4/14/8
  logarithmic bins over 70–150 / 150–1200 / 1200–20000 km, both fit windows,
  unchanged Elsasser definitions, and `geometry_stratified`. Production YAML
  files are unchanged. The 401 recipe is prepared but was not evaluated.
- `scripts/structure_function_stage2.sh`: dedicated truth-only runner based on
  the existing publication script's root/run-ID conventions. The original
  publication script and figures are unchanged.
- `src/hs_observatory/analysis/structure_function_convergence.py`: prefix
  accumulation, full-duration benchmarks, stored-data refitting and plotting.
- `tests/test_structure_function_convergence.py`: focused correctness checks.
- `.gitignore`: exceptions for these three pilot YAMLs and the shell runner.

Pilot recipes deliberately use `analysis.method: structure_function_truth_convergence`.
Run them through the new runner, not `hsobs-analyze`: the distinct method name
prevents accidental dispatch through the legacy all-MC analysis path. The runner
reads truth fields only, verifies the field p, duration and 26-bin design, and
checks a configured geometry reference before sampling. It never calls the
generation workflow. No Stage 1 sampler or scientific estimator was changed.

The budgets are 10k, 25k, 50k and 100k per fitting bin, with 10k in every outside
bin. Seeds are 12345, 23456, 34567, 45678 and 56789. Both z+ and z− moments of
orders 1–6 are saved; figures/tables focus on publication quantity z+. Every
budget is fixed before observing increments. Existing `_fit_scaling_exponents`
and `local_log_slopes` provide fits and three-point local slopes. Bin geometric
centers remain the representative separations: 14 centers enter the primary fit
and 11 enter the sensitivity fit. Theory is used only for plotted references.

Non-census budgets reuse deterministic random prefixes. Stage 1's census is
ordered, so smaller budgets cannot use its prefixes: those bins use a random
partial stream plus a separate full census. Tests compare this explicitly with
independent fixed-budget calls. Budget-to-budget comparisons are therefore
paired, not statistically independent. Outside-bin curves are identical across
budgets for each seed because their fixed budget and streams are unchanged.

## Full-duration benchmark

The following are **single fresh-process measurements**, after module import,
at the 100k/10k budget and seed 12345, one OpenBLAS/OMP thread, NumPy 2.2.6.
Total includes truth preparation, plan construction, sampling, both-field moment
accumulation, both-window fits/local slopes, and numerical serialization; it
excludes plotting. Peak RSS is process high-water memory, including imports.

| Case | Preparation | Plan | Sampling | Accumulation | Total | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| 400 | 0.057 s | 0.056 s | 3.214 s | 0.316 s | 3.657 s | 114.0 MiB |
| 402 | 0.052 s | 0.042 s | 3.096 s | 0.311 s | 3.514 s | 112.6 MiB |

Each plan examines 194,436 spacecraft-pair/lag groups rather than 233,463,636
individual candidates. Exactly 1,236,311 distinct identities are selected at the
largest budget, including censuses in the first seven fitting bins. Geometry
validation passes and no boundary expansions are needed for these edges.
Accumulation includes exact separation/bin checks, both-field increment gathers,
and order powers. The Python sparse rank sampler remains the main kernel cost.

The complete four-budget/five-seed study took 25.79 s for 400 (including a 1.12 s
bounded legacy comparison) and 24.32 s for 402. It evaluates 1,561,311 pairs per
seed using shared prefixes/census handling, versus 2,779,440 for four separate
fixed-budget calls: about 44% fewer pair evaluations. These are completed
truth-only studies, not production MC timings.

A safe legacy comparison on the same full geometry used only **two million**
candidate draws, scalar target 10k/bin. It took 1.12 s but retained just 108,835
pairs: 186–415 in the first four bins and 373–7,863 in the fitting bins. It missed
18 of 26 targets. Geometry sampling at 10k in all bins took 1.24 s total
(1.02 s sampling, 0.105 s accumulation), 89.6 MiB peak RSS, and delivered all
260,000 requested pairs. These are not equal-precision calculations, so no
speedup ratio is claimed. No 500-million-draw legacy run was attempted.

For 402, 50k fitting-bin targets cost 3.26 s total versus 3.51 s at 100k; the small
cost difference occurs because the larger target becomes an inexpensive census
in seven bins. The common 100k default provides better precision for little
additional cost here. Extrapolating the single-truth 100k kernel to 51 members
suggests roughly three minutes, **an estimate only**, excluding MC-specific I/O,
preparation, changed geometry/occupancy and whole-run memory. The former
500-million proposal cap versus 1.24 million selected identities is an operation
count comparison, not a measured production speedup.

## Per-bin populations, selection and coverage

The exact populations are identical for the two corrected inputs. At lower
budgets replace 100k by 10k/25k/50k in fitting bins and cap at the listed
population; outside bins always select 10k. All 26 bins are occupied.

| Separation km | Eligible pairs | Groups | Selected at 100k/10k | Fraction sampled |
|---|---:|---:|---:|---:|
| 70.00–84.69 | 24,002 | 10 | 10,000 | 41.66% |
| 84.69–102.47 | 45,571 | 19 | 10,000 | 21.94% |
| 102.47–123.98 | 21,586 | 9 | 10,000 | 46.33% |
| 123.98–150.00 | 47,946 | 20 | 10,000 | 20.86% |
| 150.00–174.02 | 43,129 | 18 | 43,129 | 100.00% |
| 174.02–201.89 | 50,303 | 21 | 50,303 | 100.00% |
| 201.89–234.21 | 50,282 | 21 | 50,282 | 100.00% |
| 234.21–271.72 | 55,049 | 23 | 55,049 | 100.00% |
| 271.72–315.23 | 55,023 | 23 | 55,023 | 100.00% |
| 315.23–365.70 | 69,336 | 29 | 69,336 | 100.00% |
| 365.70–424.26 | 93,189 | 39 | 93,189 | 100.00% |
| 424.26–492.20 | 119,192 | 50 | 100,000 | 83.90% |
| 492.20–571.02 | 159,524 | 67 | 100,000 | 62.69% |
| 571.02–662.45 | 202,207 | 85 | 100,000 | 49.45% |
| 662.45–768.53 | 398,998 | 168 | 100,000 | 25.06% |
| 768.53–891.60 | 517,924 | 219 | 100,000 | 19.31% |
| 891.60–1034.37 | 518,266 | 219 | 100,000 | 19.30% |
| 1034.37–1200.00 | 923,224 | 389 | 100,000 | 10.83% |
| 1200.00–1705.74 | 3,170,449 | 1,341 | 10,000 | 0.32% |
| 1705.74–2424.62 | 4,987,610 | 2,122 | 10,000 | 0.20% |
| 2424.62–3446.47 | 6,588,392 | 2,834 | 10,000 | 0.15% |
| 3446.47–4898.98 | 8,560,954 | 3,750 | 10,000 | 0.12% |
| 4898.98–6963.65 | 11,442,193 | 5,137 | 10,000 | 0.09% |
| 6963.65–9898.46 | 15,545,047 | 7,227 | 10,000 | 0.06% |
| 9898.46–14070.16 | 20,844,124 | 10,201 | 10,000 | 0.05% |
| 14070.16–20000.00 | 27,336,246 | 14,455 | 10,000 | 0.04% |

At 100k the seven fitting bins from 150 to 424.26 km reach census. Their target
shortfalls are respectively 56,871 / 49,697 / 49,718 / 44,951 / 44,977 / 30,664 /
6,811: these mean **all available pairs were used**, not sampler failure.
Every other shortfall is zero. Selected totals at 10k / 25k / 50k / 100k fitting
budgets are 260,000 / 470,000 / 813,129 / 1,236,311 per seed and case.

The narrow fitting bins contain only 18–389 distinct spacecraft-pair/lag groups;
the 102–124 km outside bin has only nine groups. Many starting times repeat each
separation. Counts and census coverage do not establish temporal or directional
independence. The numerical files retain group counts and per-budget fractions.

## Sampling convergence

CV is the sample SD across five seeds divided by their mean, evaluated per bin;
the table reports median / maximum over the 14 fitting bins. The seven census
bins have zero CV at 100k, so the **maximum** is more informative than the median
at that budget. Exponent errors are between-seed SD, not measurement-error MC
intervals or independent-realization uncertainties.

| Case | Budget / fit bin | S4 CV median / max | S5 CV median / max | S6 CV median / max | Primary xi6 mean ± SD |
|---|---:|---:|---:|---:|---:|
| 400 | 10,000 | 5.79% / 8.34% | 9.97% / 12.33% | 14.64% / 19.08% | 1.6508 ± 0.0645 |
| 400 | 25,000 | 3.26% / 5.56% | 4.92% / 8.13% | 7.05% / 12.41% | 1.6562 ± 0.0321 |
| 400 | 50,000 | 1.46% / 4.13% | 2.35% / 5.88% | 3.64% / 8.01% | 1.6308 ± 0.0187 |
| 400 | 100,000 | 0.39% / 1.84% | 0.78% / 2.54% | 1.16% / 4.53% | 1.6420 ± 0.0035 |
| 402 | 10,000 | 1.56% / 2.15% | 2.08% / 3.10% | 2.43% / 4.53% | 1.9969 ± 0.0145 |
| 402 | 25,000 | 0.94% / 1.38% | 1.34% / 1.91% | 1.96% / 2.92% | 1.9868 ± 0.0126 |
| 402 | 50,000 | 0.41% / 0.81% | 0.58% / 1.24% | 0.85% / 1.83% | 1.9901 ± 0.0028 |
| 402 | 100,000 | 0.13% / 0.48% | 0.16% / 0.66% | 0.17% / 0.98% | 1.9899 ± 0.0016 |

For 400, increasing from 50k to 100k changes the paired S4/S5/S6 curves by
1.41% / 2.32% / 3.61% RMS across seeds and fitting bins. Corresponding mean
primary exponent changes are +0.0045 / +0.0073 / +0.0111. Earlier S6 RMS changes
relative to the 100k result are 13.56% at 10k and 8.02% at 25k. There is clear
improvement, but budget means need not move monotonically. Observed maximum S6
CV falls from 19.08% to 12.41%, 8.01%, then 4.53%.

For 402, 50k→100k S4/S5/S6 RMS changes are only 0.45% / 0.63% / 0.88%, with
primary exponent changes −0.00046 / −0.00043 / −0.00028. Already at 50k the
maximum S6 CV is 1.83%; at 100k it is 0.98%. These improvements partly reflect
finite-population/census coverage, not only the different intermittency. The
largest budget is a comparison reference, not an independent exact population
answer in the seven uncensused fitting bins.

## Fit-window sensitivity and recommendation

Largest-budget exponents, mean ± between-seed SD:

| Order | 400: 150–1200 | 400: 200–1000 | 402: 150–1200 | 402: 200–1000 |
|---|---:|---:|---:|---:|
| 1 | 0.4642 ± 0.0008 | 0.4581 ± 0.0006 | 0.3317 ± 0.0005 | 0.3267 ± 0.0005 |
| 2 | 0.8112 ± 0.0016 | 0.8036 ± 0.0008 | 0.6647 ± 0.0008 | 0.6576 ± 0.0008 |
| 3 | 1.0738 ± 0.0025 | 1.0663 ± 0.0012 | 0.9986 ± 0.0010 | 0.9918 ± 0.0008 |
| 4 | 1.2862 ± 0.0034 | 1.2726 ± 0.0019 | 1.3319 ± 0.0012 | 1.3270 ± 0.0010 |
| 5 | 1.4711 ± 0.0038 | 1.4426 ± 0.0027 | 1.6630 ± 0.0014 | 1.6600 ± 0.0012 |
| 6 | 1.6420 ± 0.0035 | 1.5927 ± 0.0044 | 1.9899 ± 0.0016 | 1.9870 ± 0.0015 |

For 400 the sensitivity-window changes in xi4/xi5/xi6 are −0.0137 / −0.0286 /
−0.0493, about −1.06% / −1.94% / −3.00%. This exceeds seed variation, especially
at high order. For 402 the corresponding changes are −0.0049 / −0.0030 /
−0.0029, all below 0.4% relative. Local xi6 estimates across primary-fit centers
range from 1.06 to 3.68 for 400 and 1.57 to 2.32 for 402 at the largest budget
(seed means). A stable global regression alone therefore does not certify a
flat local-slope plateau.

Recommendations:

1. Retain the explicit 26-bin design and **100k fitting / 10k outside** pilot
   targets, population-capped. This achieves observed five-seed S4–S6 precision
   better than roughly 5% in every fitting bin for both supplied truth inputs.
   For 402 alone 50k is adequate for a roughly 2% per-bin high-order precision
   target, but 100k costs only about 0.25 s more and makes seven bins exact.
   The same budget for 401 is a conservative proposal, not a measured result.
2. Use **150–1200 km** as the primary finite-window estimate: it uses 14 bins
   and gives smaller high-order seed variability than the narrower fit in 400.
   Keep **200–1000 km** as a reported sensitivity comparison, especially for
   xi5/xi6. Do not choose a window by its agreement with analytic p-model values
   or interpret either as a uniquely established inertial interval.
3. Treat the above precision as conditional subsampling evidence. Five seeds
   provide imprecise estimates of the seed SD; high-order tails outside the
   fitting interval receive only 10k pairs. A stronger precision requirement
   needs additional fixed-budget/seed checks before production, not adaptive
   increment-dependent stopping. No such extra run or MC ensemble was launched.

Pair subsampling convergence holds the 300-second observations and field
realization fixed. It does **not** establish convergence in observation duration,
independent realizations, direction/baseline coverage or measurement errors.
Even a complete pair census reuses correlated time samples. The plotted analytic
references are normalized at the first fitting center and provide shape context
only; they were not used to assess convergence or select the budget.

## Artifacts, reproduction and validation

Numerical outputs retain every seed/budget, S1–S6 for both fields, counts,
populations/shortfalls/groups, both-window exponents, three-point local slopes,
source coordinate/field hashes, config and timing/memory metadata:

- 400 numerical results: `outputs/structure_function_stage2/malara-prod-400_truth_convergence.h5`
- 402 numerical results: `outputs/structure_function_stage2/malara-prod-402_truth_convergence.h5`
- Derived summary: `outputs/structure_function_stage2/figures/summary.json`
- [Exponent convergence](assets/figures/sf-stage2-xi-convergence.png)
- [High-order SF variability](assets/figures/sf-stage2-high-order-convergence.png)
- [Updated truth-only SFs](assets/figures/sf-stage2-truth-structure-functions.png)

The three linked PNGs are documentation snapshots of the existing results.
Their original PNG/PDF products and the numerical files remain in the repository's ignored
`outputs/` tree. The paths above are local artifacts, not documentation downloads;
recipes/code/report are trackable. The original
publication image paths were never written. Refitting/plotting reads saved
moments; `fit_saved(..., windows=...)` also accepts alternative fit-window maps.

```bash
# Recalculate only the two truth studies (never the observed MC ensemble):
bash scripts/structure_function_stage2.sh 400 402

# Plot/refit stored moments without resampling:
PYTHONPATH=src MPLCONFIGDIR=/tmp/sf-stage2-mpl .venv/bin/python \
  -m hs_observatory.analysis.structure_function_convergence --plot \
  outputs/structure_function_stage2/malara-prod-400_truth_convergence.h5 \
  outputs/structure_function_stage2/malara-prod-402_truth_convergence.h5

# Fresh-process full-duration single-seed benchmark:
PYTHONPATH=src MPLCONFIGDIR=/tmp/sf-stage2-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m hs_observatory.analysis.structure_function_convergence \
  --config analysis/malara-prod-400_sf-stage2-pilot.yaml --benchmark-only
```

Validation: 60 focused tests passed across convergence, geometry sampling and
local slopes; the final geometry-reference guard also passed the five convergence
tests after its addition. Tests cover exact checkpoint/census equivalence to
separate fixed-budget calls, chunking, bin edges, fitting/local slopes, recipe
validation, truth-only dispatch, saved arrays and mismatched-trajectory rejection.
Targeted ruff checks passed for the two added Python files; shell syntax passed.
No full-repository test/lint run or full MC analysis was performed.
