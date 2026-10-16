# Uniform logarithmic structure-function binning test

2026-10-14. **Uniform binning does not eliminate the intermittent-case local-slope
peak near 1000 km.** Its height and sampled location depend on grid phase, and
finite-window global exponents also change. The original 1200-km width transition
is therefore not the sole cause; these calculations do not establish a physical
feature or isolate a causal slope bias attributable to that transition alone.
Uniform bins improve consistency of the derivative's smoothing width, but do not
by themselves establish more robust measured exponents.

## Inputs, estimator and grids

Reviewed the Stage 2 convergence and Stage 3 publication reports, Stage 3 recipes,
and existing three-point local-slope implementation. Only corrected existing
400 (p=0.9) and 402 (p=0.5) sampled truth was evaluated: all 2401 timestamps over
300 seconds, unchanged field realizations, nominal trajectories, Elsasser
velocity/density/mean definitions, all-spacetime unordered pair population and
orders 1–6. Input SHA256 values must match the saved Stage 3 provenance before
processing. No MC calculation or generator is invoked. The saved Stage 3 truth
is reference A at seed 12345, rather than a replacement calculation.

- **A:** unchanged 26-bin piecewise grid: 4/14/8 logarithmic bins over
  70–150 / 150–1200 / 1200–20000 km.
- **B:** 40 uniform logarithmic bins over 70–20000 km;
  delta ln(lambda) = ln(20000/70)/40 = 0.1413748 (A fitting bins: 0.1485315).
- **C:** B's internal grid shifted upward by half that log spacing, with fixed
  endpoints 70 and 20000 km. There are 41 bins: 39 full-width interior bins and
  two half-width edge bins. No gaps, overlaps or changed physical limits.

Exact floating-point edges are in `outputs/structure_function_uniform_binning/bin_edges.json`.
The same geometry-stratified v1 sampler and existing accumulator are used.
Targets are 100000 where the **geometric center** lies in [150,1200] km and
10000 otherwise, capped at exact populations. Thus boundary-straddling bins can
have different targets across grids (e.g. C's 1102.5–1269.9 km bin receives
100000); this is explicit, and fits always use the unchanged physical windows.
Internal edges are lower-inclusive/upper-exclusive, final edge upper-inclusive.
Seeds 12345 and 23456 are fixed in advance. Different edges key different random
streams; comparisons between grids are not paired draws of identical identities.
No increment-dependent selection or stopping occurs.

## Occupancy and sampling

Both inputs have identical nominal geometry. All bins are occupied, with the same
total eligible population **101869766** under every grid. Selected totals are
1236311 / 1406938 / 1459876 for A/B/C. Seven A bins and eight B/C bins reach a
census below their target; there are no invalid-pair losses. Minimum populations
are 21586 / 16798 / 12002, so no bin is empty. Census shortfalls are fully saved.
The table includes every bin intersecting 800–1500 km; group counts are exact
eligible distinct spacecraft-pair/signed-lag groups, not independent samples or
the number of groups necessarily represented in a finite draw.

| Grid | Bin km | Eligible | Selected | Eligible groups | Selected fraction |
|---|---:|---:|---:|---:|---:|
| A | 768.5–891.6 | 517,924 | 100,000 | 219 | 19.31% |
| A | 891.6–1034.4 | 518,266 | 100,000 | 219 | 19.30% |
| A | 1034.4–1200.0 | 923,224 | 100,000 | 389 | 10.83% |
| A | 1200.0–1705.7 | 3,170,449 | 10,000 | 1341 | 0.32% |
| B | 774.2–891.8 | 508,491 | 100,000 | 215 | 19.67% |
| B | 891.8–1027.2 | 466,154 | 100,000 | 197 | 21.45% |
| B | 1027.2–1183.2 | 856,376 | 100,000 | 361 | 11.68% |
| B | 1183.2–1362.9 | 1,307,501 | 10,000 | 551 | 0.76% |
| B | 1362.9–1569.9 | 1,251,411 | 10,000 | 530 | 0.80% |
| C | 721.4–830.9 | 471,033 | 100,000 | 199 | 21.23% |
| C | 830.9–957.1 | 461,458 | 100,000 | 195 | 21.67% |
| C | 957.1–1102.5 | 608,822 | 100,000 | 257 | 16.43% |
| C | 1102.5–1269.9 | 1,138,113 | 100,000 | 479 | 8.79% |
| C | 1269.9–1462.7 | 1,277,355 | 10,000 | 540 | 0.78% |
| C | 1462.7–1684.8 | 1,240,380 | 10,000 | 526 | 0.81% |

All-bin populations, requested/selected counts, shortfalls, groups and geometric /
population / sampled log centers for both seeds are saved in `bin_counts.csv`.
Occupancy near the peak is substantial (195–479 eligible groups in the main
fitting bins), although their increments are correlated through shared samples.
Counts are not effective independent sample sizes.

## Local slopes and structure functions

The existing three-point regression uses actual log-center spacings, not an
assumption of equally spaced bins. An exact power law gives a constant derivative
even on A (verified by a focused test). Changing bin width can nevertheless
change the averaging and smoothing of a curved structure function.

For p=0.9 the maximum over geometric centers in 800–1500 km is:

| Grid | Peak location km | Local xi3 | Local xi4 | Local xi6 | Second-seed xi6 peak |
|---|---:|---:|---:|---:|---:|
| A | 960.3 | 1.647 | 2.343 | 3.823 | 3.659 |
| B | 957.1 | 1.604 | 2.331 | 4.018 | 3.960 |
| C | 891.8 | 1.469 | 2.068 | 3.616 | 3.184 |

A and B peak at essentially the same location despite B having no width
transition. C moves the maximum by about half a bin to 892 km and broadens it
across the next center (~1027 km); the excess persists in both seeds. B's xi6
peak is slightly higher than A's, so the transition cannot simply be described
as amplifying this maximum. Its changed smoothing is visible on the descending
side: at ~1114 / 1102 / 1183 km A/B/C have xi6 = 1.983 / 2.506 / 1.072, but
these are different locations and neighborhoods, not estimates at one common
scale. The structure-function figure shows curvature across ~800–1200 km on
all grids; the peak is not just an isolated derivative value.

The p=0.5 curves show much weaker modulation: sixth-order maxima in 800–1500 km
are 1.971 / 1.965 / 1.939 for seed 12345, and 1.994 / 2.039 / 2.033 for seed
23456. Peak-location changes are not evidence of a separate physical scale.

Pair-subsampling noise matters most beyond the fit boundary. For p=0.9,
maximum absolute two-seed S6 differences among centers in 800–1200 km are
7.15% / 3.60% / 9.31% for A/B/C; corresponding S4 differences are
3.78% / 1.25% / 3.26%. At some outside centers (10k draws), S6 differences
reach ~16–21% on B/C. For p=0.5 the in-window S6 differences are only
0.78% / 2.55% / 1.79%. These are two-seed differences, not estimated confidence
intervals. No difference is caused by zero occupancy. Increasing outside targets
would help test the descending side, but was not done in this prescribed study.

## Global fits

Geometric-center convention, seed 12345; all six orders are saved and shown.
Each row gives raw xi1 through xi6, without normalization or theory tuning.
Window selection is inclusive on the stated representative centers: primary
has 14/15/15 bins and sensitivity has 11/12/11 bins for A/B/C. Moving boundaries
changes the finite-window averages and discrete center coverage, even with
physically fixed windows. It cannot isolate the effect of the 1200-km width
transition, whose outer bins do not themselves enter the primary fit.

| p | Grid | Window km | xi1 | xi2 | xi3 | xi4 | xi5 | xi6 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 0.9 | A | 150–1200 | 0.4654 | 0.8131 | 1.0760 | 1.2885 | 1.4736 | 1.6451 |
| 0.9 | A | 200–1000 | 0.4589 | 0.8040 | 1.0654 | 1.2699 | 1.4380 | 1.5872 |
| 0.9 | B | 150–1200 | 0.4750 | 0.8252 | 1.0869 | 1.2987 | 1.4850 | 1.6592 |
| 0.9 | B | 200–1000 | 0.4720 | 0.8238 | 1.0846 | 1.2853 | 1.4480 | 1.5884 |
| 0.9 | C | 150–1200 | 0.4663 | 0.8179 | 1.0903 | 1.3162 | 1.5165 | 1.7054 |
| 0.9 | C | 200–1000 | 0.4662 | 0.8170 | 1.0844 | 1.2952 | 1.4699 | 1.6253 |
| 0.5 | A | 150–1200 | 0.3324 | 0.6659 | 1.0000 | 1.3333 | 1.6642 | 1.9905 |
| 0.5 | A | 200–1000 | 0.3276 | 0.6587 | 0.9927 | 1.3275 | 1.6598 | 1.9859 |
| 0.5 | B | 150–1200 | 0.3392 | 0.6794 | 1.0203 | 1.3606 | 1.6993 | 2.0348 |
| 0.5 | B | 200–1000 | 0.3369 | 0.6778 | 1.0217 | 1.3664 | 1.7090 | 2.0460 |
| 0.5 | C | 150–1200 | 0.3338 | 0.6684 | 1.0032 | 1.3366 | 1.6671 | 1.9926 |
| 0.5 | C | 200–1000 | 0.3316 | 0.6682 | 1.0091 | 1.3516 | 1.6922 | 2.0269 |

Relative to A, primary xi6 changes by +0.0141 / +0.0603 on B/C for p=0.9,
and +0.0443 / +0.0021 for p=0.5. Sensitivity-window differences are
+0.0012 / +0.0381 and +0.0602 / +0.0410, respectively. These changes can exceed
pair noise: primary xi6 two-seed differences are 0.0069 / 0.0042 / 0.0128 for
p=0.9 and 0.0006 / 0.0004 / 0.0024 for p=0.5. Thus uniform binning does not
provide grid-phase invariance of the finite-window exponent. It is not justified
to label the A-versus-B difference a measured bias from the width transition.

## Empirical representative separations

Saved diagnostics include exp(mean(log(lambda))) over both the selected pairs
(actual stored coordinates) and the full eligible population (multiplicity-
weighted affine lag groups, equal distances within floating coordinate roundoff).
An exhaustive small-data test checks the latter compression. Geometric centers
remain the publication convention. Diagnostic fits/local slopes rerun the same
existing routines with empirical representatives and the same physical windows;
primary-window membership is unchanged. For B's sensitivity fit, the bin
at geometric center 202.109 km has empirical center 199.107 km and leaves the
fixed 200–1000 km window (12 bins become 11). Thus B's sensitivity correction
combines a coordinate change and a membership change; it is not a pure abscissa
correction. Both seeds and cases exhibit this same census-bin shift.

Around 800–1500 km, population centers differ from geometric centers by at most
0.84% / 0.53% / 0.72% for A/B/C. These small shifts modestly alter local peak
heights: p=0.9 xi6 becomes 3.725 / 3.922 / 3.637 and the feature persists.
For global primary xi6, population centers change A/B/C by
+0.0215 / -0.0117 / +0.0110 (p=0.9), and +0.0245 / -0.0143 / +0.0118
(p=0.5). Sensitivity corrections reach ~0.030. These are material at the
pair-subsampling precision, though small relative to the local spike. The
p=0.5 primary cross-grid spread drops from ~0.0443 to ~0.0161 using population
centers; it does not vanish, and p=0.9 remains grid-sensitive. Sampling-center
and population-center refits are saved separately to expose center-estimation
noise. Empirical centers should be evaluated explicitly before any future change
of publication convention, not silently substituted.

## Artifacts, reproduction and validation

Added diagnostic infrastructure only:

- `src/hs_observatory/analysis/structure_function_uniform_binning.py`:
  truth-only driver, separation statistics, saved-moment refits and three figures.
- `tests/test_structure_function_uniform_binning.py`: grid coverage, partial
  edges, target allocation, exhaustive separation means/counts, fixed physical
  fit masks and unequal-grid power-law local slopes.
- This report. Existing analysis/sampler/local-slope implementations are reused.

All numerical results are in `outputs/structure_function_uniform_binning/`:
`results.json` (both seeds, both Elsasser variables, moments/counts/metadata,
three center conventions, both fits and local slopes), `bin_edges.json`,
`bin_counts.csv`, and `fitted_exponents.csv`. Figures are `local_slopes`,
`structure_functions`, and `fitted_exponents`, each PNG/PDF. Solid curves are
seed 12345 and faint dotted curves seed 23456; grey marks identify 1200 km.
The local/SF figures focus on 500–2000 km; full-range moments remain saved.

```bash
# Truth-only calculation and figures; no MC dispatch:
PYTHONPATH=src MPLCONFIGDIR=/tmp/sf-uniform-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m hs_observatory.analysis.structure_function_uniform_binning
# Regenerate tables/figures from saved moments alone:
PYTHONPATH=src MPLCONFIGDIR=/tmp/sf-uniform-mpl \
  .venv/bin/python -m hs_observatory.analysis.structure_function_uniform_binning --plot-only
```

Measured truth-study runtime before serialization/plotting: **87.32 s**;
peak process RSS **291.7 MiB**. Individual records took 3.35–8.60 s,
including separation-statistic stream replay (not just moment accumulation).
This includes two seeds for all grids/cases, reusing the saved A seed-12345
moments. It is not a production-ensemble performance estimate.

Validation: exact counts equal min(requested, eligible) and the accumulator's
retained counts; all grid populations sum identically; corrected input hashes
match Stage 3. SHA256 checks confirm the inspected Stage 3 configurations and
case-400/402 publication JSON/PNG/PDF artifacts were unchanged. No 401 or 50X
processing, field generation, error-model changes or MC analyses occurred.
**60 focused tests passed** (including the added empirical-window boundary
check); targeted lint passed for both added Python files. Commands:

```bash
.venv/bin/pytest -q tests/test_structure_function_uniform_binning.py \
  tests/test_geometry_pair_sampling.py tests/test_structure_function_local.py
.venv/bin/ruff check src/hs_observatory/analysis/structure_function_uniform_binning.py \
  tests/test_structure_function_uniform_binning.py
```

**Recommendation:** retain the current production results pending review. A
uniform log grid is useful for consistent-resolution diagnostics, with shifted
grids as a robustness check; it does not remove this feature or guarantee a more
stable global fit. Report finite-window/grid sensitivity alongside pair noise,
and evaluate empirical center conventions before adopting new bins. None of
these tests addresses convergence in 300-second duration, independent field
realizations, or measurement-error uncertainty. A local-slope spike alone is
insufficient evidence of a physical break or scaling regime.
