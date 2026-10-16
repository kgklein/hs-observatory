# Study A: high-order moment concentration

Implemented by `hs_observatory.analysis.structure_function_moment_convergence` and
`scripts/structure_function_moment_convergence.sh`. This is analysis of **existing
sampled truth only**. It never generates fields, reads observed members, or creates
measurement-error realizations. No production run was performed during implementation.

From the project root, run the following commands when ready.

```bash
# Validate inputs and report exact eligible populations without sampling or writing products.
bash scripts/structure_function_moment_convergence.sh --preflight

# Low-budget smoke test on the two saved 1500-second cases; separate output.
bash scripts/structure_function_moment_convergence.sh \
  --budgets 1000 --seeds 12345 --separations 500 \
  --output outputs/structure_function_moment_convergence_smoke

# Default: both p=0.9 and p=0.5 cases, 500k/bin, three seeds, five separations.
bash scripts/structure_function_moment_convergence.sh --resume

# Optional budget convergence, reusing completed 500k products.
bash scripts/structure_function_moment_convergence.sh \
  --budgets 500000 1000000 2000000 --resume

# Regenerate figures and summary from the latest run manifest; no input files needed.
bash scripts/structure_function_moment_convergence.sh --plot-only

# Another saved duration, extra explicit zero-based bin, alternate blocks.
bash scripts/structure_function_moment_convergence.sh \
  --cases malara-ts-500a malara-ts-502a --duration 300 \
  --separations 300 500 700 1000 1500 2000 --bins 0 --block-width 15 --resume \
  --output outputs/structure_function_moment_convergence_t300
```

`--cases` accepts generation filename stems, with or without `-tDURATION`.
An explicit `--duration` overrides the suffix; otherwise the suffix or 1500 seconds
is used. Each input is discovered from `generation/CASE-tDURATION.yaml`'s
`output.path`, relative to the project root (`--root`). Saved generation metadata
must match the YAML and specify `errors.n_mc: 0`. Missing inputs cause an error;
there is no generation fallback. The aligned duration is also checked.

The default inputs are `malara-ts-500a-t1500` (p=0.9, orange) and
`malara-ts-502a-t1500` (p=0.5, blue). The existing Elsasser defaults and truth
position reader supply plasma-frame z-plus, density normalization, mean
subtraction, and native-grid alignment. Increment amplitude is the full vector
norm `|z_plus[right] - z_plus[left]|` in km/s, and separation is the Euclidean
spacecraft-time position difference, exactly as in the existing all-spacetime
structure functions. Global means are computed on the entire selected record.
The existing fixed-budget convergence accumulator was inspected; a separate
duration-study module/recipe is not present in this checkout.

## Sampling and numerical conventions

The existing `geometry_stratified` plan and sampler are reused without changes.
The grid is 40 uniform log bins on 70–20,000 km. Requested separations select the
nearest geometric bin center in log distance; `--separations` replaces the five
default targets, and `--bins` adds zero-based bin indices. Preflight reports the
actual bin edges, exact population, and budget caps. Unsupported non-affine
geometry is rejected by the existing sampler.

Default seeds are 12345, 23456, and 34567. Each fixed budget samples uniformly
without replacement, capped at population. Non-census budgets share random
prefixes for a given seed; a census is a separate ordered stream. Smaller budgets
are never taken from the ordered census prefix. Seeds are sampling repeats, not
independent physical realizations. Census products necessarily agree across seeds.
No complete eligible identity list or Cartesian product is constructed. Transient
records are bounded by one selected bin/budget, plus the existing sampler's sparse
rank map; `--chunk-size` controls sampler chunks, not total diagnostic memory.
The 2M option requires more memory than 500k. Products are processed sequentially.

For each n=2,4,6, powers are formed after dividing amplitudes by their maximum.
This keeps fractional contributions and N_conc stable without raising large
physical amplitudes to the 12th power. Both dimensional `S_n` and `log_S_n` are
saved; the former can overflow when the physical moment exceeds float64, while
log estimates and normalized diagnostics remain available. Summary differences
use log moments. Zero-amplitude moments are zero; their normalized contributions
are undefined (NaN). Empty bins retain explicit zero counts and undefined moments.

Top 0.1%, 1%, and 5% means the largest `ceil(f*N)` increments; actual counts are
saved, so small-bin rounding is visible. Ties use stable sample order. The exact
empirical cumulative contribution includes all amplitudes at or below x, including
ties. Its horizontal coordinate is x/sqrt(S2). Histograms use logarithmic positive
amplitude edges spanning the full sampled range, with the mass at zero saved
separately. Saved PDF densities integrate to the positive-amplitude probability.

## Temporal attribution and interpretation

Blocks are nonoverlapping, anchored at the first aligned observation time, with
30-second default width. The final block can be shorter; a midpoint exactly at
the final endpoint belongs to the final block. Each pair's full contribution is
attributed to its temporal midpoint, even if its endpoints occupy different blocks.
Saved summaries include each block's counts and S2/S4/S6 fractions, top 1/3/5 block
fractions, and locations of the largest increments. Spacecraft-pair/lag groups
use ascending spacecraft indices and a signed sample lag in that orientation;
the spacecraft ID lookup and aligned-grid processing settings are in metadata.
Extreme-event counts at all three tail thresholds are aggregated by block and group.

**N_conc is a concentration measure, not a correlation-corrected effective sample
size.** Shared observations and structures make increments, blocks, and lag groups
dependent. Temporal peaks show localization under midpoint attribution, not a
count of independent physical events. More pair sampling can reduce subsampling
variability while leaving finite-duration rare-event limitations unresolved.

## Saved products, resume, and figures

Each case/bin/seed/requested-budget product is an atomic compressed HDF5 file.
It contains exact sorted amplitudes (enough to regenerate arbitrary cumulative
curves and PDFs), moments, log moments, tail fractions/counts, N_conc, histogram
counts/edges/PDF, temporal and group aggregates, and the largest `--top-count`
increments (default 100). Only these top records retain full identities, both
times, midpoint, signed lag, separation and amplitude on disk. Millions of full
pair records are not saved by default. Top increments rank identically for all
positive moment orders.

Cache fingerprints cover the HDF5 content hash, generation YAML, analysis and
sampler implementation, processing settings, bin edges, seed, budget, block
width, and top-record count. Changing the budget list does not invalidate existing
individual budgets. `--resume` validates and skips completed matching products;
interrupted `.tmp` files are ignored. Source content is hashed in bounded chunks,
including during preflight/resume. Resume still prepares fields and geometry to
validate current inputs but does not resample completed products.

`manifest.json` records the most recent invocation's selected products. Old caches
remain available, but plot-only uses this manifest rather than mixing all files
in the directory. Use separate output directories for separate studies. To include
old and new budgets in the same figures, request the complete budget list with
`--resume`. Plot-only reads saved products without checking current source files.

Figures in `figures/` are written as both PNG and PDF:

- **A_cumulative:** all three moment orders, a row per separation; cases and seeds
  overlaid at each case's largest requested budget.
- **B_tail_concentration:** all tail thresholds and N_conc/N versus separation,
  showing individual seeds at the largest budget.
- **C_temporal:** block S6 contributions at the largest budget; black crosses mark
  blocks containing increments in the top 0.1% by amplitude.
- **D_budget_convergence:** dimensional moments and all three tail fractions versus
  requested budget, with individual seed curves. Actual population-capped counts
  are saved, so census plateaus can be identified.

`figures/summary.json` includes moments and log moments, relative differences to
the largest budget within each seed, log differences to the same-budget seed
mean, between-seed log-moment standard deviation (population convention, ddof=0),
and tail differences/standard deviations. A single seed has zero between-seed
spread by convention; this is not an uncertainty estimate. The HDF5 products
retain the remaining block/group diagnostics for further analysis. Only rendered
cumulative lines are decimated; cached distributions remain exact.

Focused implementation checks (small synthetic data only):

```bash
MPLCONFIGDIR=/tmp/sf-moment-test-mpl .venv/bin/python -m pytest -q \
  tests/test_structure_function_moment_convergence.py
.venv/bin/ruff check \
  src/hs_observatory/analysis/structure_function_moment_convergence.py \
  tests/test_structure_function_moment_convergence.py
bash -n scripts/structure_function_moment_convergence.sh
```
