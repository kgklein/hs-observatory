# Geometry-stratified sampling: Stage 1

Implemented 2026-10-14. Production generation, trajectories, error values, YAML
bin definitions, global fit windows and publication figures were not changed.
The reviewed historical-output discrepancy was not investigated further.

## Configuration and behavior

Use `increment_mode: all_spacetime_pairs` and:

```yaml
pair_sampling:
  mode: geometry_stratified
  target_pairs_per_bin: [2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000]
  random_seed: 12345
  chunk_size: 50000
```

The array must match the actual number of bins; a scalar broadcasts to all bins.
Targets must be nonnegative integers, and zero disables sampling in a bin. Each
target is capped at that member/bin's exact eligible population. `max_pairs`
is rejected in this mode; use the per-bin budgets. Legacy `max_candidates` and
`binning_n_pairs` do not control the geometry sampler. Other sampling modes retain
their existing behavior.

Explicit edges are recommended for comparisons. If omitted, deterministic
conservative geometry bounds replace the random edge-discovery probe for this
mode only; the configured number of logarithmic bins and min/max limits still
apply. Bounds cover coordinate roundoff; they need not match historical sampled
extrema. Geometric bin centers and the existing moment/fitting/plotting code are
unchanged.

[`derived/geometry_pairs.py`](https://github.com/kgklein/hs-observatory/blob/main/src/hs_observatory/analysis/derived/geometry_pairs.py)
checks the aligned coordinates for a common affine displacement per sample index.
It permits floating-point construction roundoff, with a conservative bound on its
effect on distances. Unsupported formation motion raises `ValueError` with
instructions to select exhaustive `all` explicitly; there is no silent fallback.
The existing Elsasser loader continues to require uniformly sampled instrument
time grids and align differing instrument cadences.

For i<j, all signed lags are enumerated, including zero; self-pairs use positive
lags only. A group has T−|k| starts and maps to canonical flattened identities
a<b. Group counts are accumulated per bin. Near any bin edge, pair limit or zero
separation, actual stored-coordinate distances are checked for every start.
Only groups split by numerical boundary effects store explicit start lists;
ordinary groups retain implicit consecutive starts. This preserves exact
eligibility under the existing internal-half-open/final-inclusive edge rules.
Boundary-heavy inputs may require more work than ordinary affine inputs.

Uniform integer ranks across concatenated group multiplicities are drawn without
replacement using sparse Fisher–Yates; no full population permutation is allocated.
A census streams ordered ranks. PCG64 streams are keyed by seed and the two bin
edge float64 bit patterns, without chunk size or member index. Sampling identities
are reproducible across chunk sizes and MC order; final floating reductions can
differ in their last bits with chunking. Census order is deterministic rather
than a random permutation. Different eligible populations share random streams,
but do not force identical identities.

[`structure_function.py`](https://github.com/kgklein/hs-observatory/blob/main/src/hs_observatory/analysis/structure_function.py)
integrates these plans with the existing accumulator. A run-local two-entry LRU
reuses plans for byte-identical coordinates, edges and limits, including across
truth/observed groups. Perturbed MC position offsets still enter each member's
positions; hidden timing displacements are not newly added to labels. No field
generation or error draw occurs in the sampler.

Existing JSON keys are preserved. Additional diagnostics live under
`products.<group>.spacetime_pair_counts.geometry_sampling`, with seed, PCG64,
algorithm version `geometry_stratified_v1`, and per-member:

- Exact eligible, requested, selected and retained counts per bin.
- Population shortfall and distinct spacecraft-pair/lag-group counts per bin.
- Candidate lag groups, exact boundary-pair checks and plan-cache reuse.

Empty bins remain explicit: zero counts, NaN moments and population shortfalls.
No importance weighting, KD-tree, adaptive stopping or new scientific moment
diagnostics were introduced.

## Validation

Result: 40 focused tests passed; targeted ruff checks passed for all four changed
or added Python files. No full-repository checks or production calculation ran.

Focused tests cover census identity sets/counts against exhaustive enumeration;
self/cross/zero/signed lags and finite ends; V=0 and single-time formations;
floating-point edge neighbors and split starts; unequal-multiplicity uniformity
over 1024 seeds; duplicate exclusion; orders 1–6 census equality and statistical
agreement over 384 seeds; chunk and MC-order independence; cache reuse; actual
HDF5 position-offset reconstruction, automatic edges and JSON serialization;
empty bins, invalid budgets and clear non-affine rejection.

Run only the relevant tests and modified-file lint:

```bash
MPLCONFIGDIR=/tmp/sf-stage1-mpl .venv/bin/python -m pytest -q \
  tests/test_geometry_pair_sampling.py tests/test_structure_function.py
.venv/bin/ruff check src/hs_observatory/analysis/derived/geometry_pairs.py \
  src/hs_observatory/analysis/structure_function.py \
  tests/test_geometry_pair_sampling.py scripts/benchmark_structure_function_pair_sampling.py
```

## Measured bounded benchmark

Real 400 sampled truth, first 256 aligned times, nine spacecraft, orders 1–6,
both fields; nine test bins `geomspace(70,5000,10)`; 2000 pairs/bin; chunks 50k;
seed 12345; three fresh-worker kernel repetitions. Preparation/import time is
excluded. One OpenBLAS/OMP thread, NumPy 2.2.6. Both sampled methods retained
exactly 18,000 identities and 2000 in every bin.

| Metric | Adaptive stratified | Geometry stratified |
|---|---:|---:|
| Median kernel time | 0.1225 s | 0.0947 s |
| Absolute process peak RSS | 93.6 MiB | 83.8 MiB |
| Pre-kernel process high-water RSS | 80.8 MiB | 80.7 MiB |
| Global random candidate draws | 200,000 | 0 |
| Discrete candidate lag groups checked | not applicable | 20,691 |
| Exact eligible identities in bins | not recorded by selector | 2,166,235 |
| Selected identities | 18,000 | 18,000 |
| Selection time, separate instrumented pass | 0.1109 s | 0.0853 s |
| Remaining kernel time, including gathers/binning/reduction/fits | 0.00891 s | 0.00853 s |
| Moment power sums alone, included above | 0.00144 s | 0.00152 s |

Possible identity pairs: 2,653,056; exhaustive reference took 0.714 s median and
87.4 MiB peak RSS, contributing 2,166,235. Exact eligible per-bin counts were
30,685 / 43,568 / 67,255 / 107,162 / 163,588 / 251,347 / 371,502 / 514,710 /
616,418. The geometry selector's counts match the exhaustive reference.

**Measured speedup is 1.29×**, with about 9.8 MiB lower process peak RSS in this
case. RSS is Linux process high-water memory including imports, preparation and
repeated kernels, not an allocation-only measurement. Selector instrumentation
uses a separate pass and adds some overhead. Most remaining geometry time is
the Python sparse rank sampler; no production speedup is claimed. Its work scales
with eligible lag groups and selected ranks rather than a rejection-draw cap.

Relative RMSE against exhaustive z+ moments, pooled over all nine bins and five
seeds (12345–12349), at equal per-bin budgets:

| Order | Adaptive | Geometry |
|---|---:|---:|
| 1 | 1.26% | 1.31% |
| 2 | 2.25% | 2.51% |
| 3 | 3.27% | 3.85% |
| 4 | 4.98% | 5.62% |
| 5 | 8.15% | 8.35% |
| 6 | 12.71% | 12.54% |

Accuracy is comparable on this bounded case; five seeds do not certify a
production budget. Census tests establish exact estimator agreement; multiseed
tests check sampling statistics. Full 50-member analysis was not run.

Reproduce a bounded comparison/pilot calculation from the repository root:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/sf-stage1-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/benchmark_structure_function_pair_sampling.py \
  --n-times 256 --target 2000 --output /tmp/structure-function-sampling-benchmark.json
```

The script caps time selection at 512, uses sampled truth only, starts a fresh
worker per method, and saves timings, populations, selected counts and moment
estimates. It never launches the MC ensemble or writes a figure.

An end-to-end CLI pilot was also run successfully using
`/tmp/sf-geometry-pilot.yaml`: `/tmp/sf-geometry-pilot.h5` contains the first 256
times of 400 with one observed member plus truth and its stored position/clock
offsets. The YAML retains the saved nine production edges and 300–8000 km fit
window, sets 2000 targets per bin and disables plotting. Output
`/tmp/sf-geometry-pilot.json` retained 2000 pairs in each of its eight occupied
bins per member and explicitly reported zero population in the largest bin.
These temporary pilot files are ready to rerun in this workspace:

```bash
MPLCONFIGDIR=/tmp/sf-stage1-mpl .venv/bin/hsobs-analyze /tmp/sf-geometry-pilot.yaml
```
