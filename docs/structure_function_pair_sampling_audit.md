# Structure-function pair-sampling audit

Audit date: 2026-10-14. Scope: the `400/401/402` publication workflow and its
direct dependencies. This is an audit and implementation proposal; no production
analysis, field generation, configuration change, or plot regeneration was run.

## Recommendation

Use a **hybrid**: exact geometry-aware, separation-stratified lag sampling when
the actual analysis coordinates pass a common-velocity affine-trajectory check;
use chunked spatial-neighbor enumeration with per-bin uniform reservoirs for
general trajectories. Keep the existing vectorized exhaustive iterator as the
small-input reference and simplest fallback. Do not make global random rejection
the primary production method.

The immediate problem is more than wasted large-separation calculations. The
saved runs each report 500 million candidate draws for **every one of 51
members**, yet their first three bins miss the 500,000 target. In the available
HDF5 geometry the first bin has fewer than 500,000 eligible pairs in its entire
population. The current sampler cannot recognize this impossibility and runs
to its cap. Finer bins make equal, unattainable targets more common.

Explicit nonuniform edges already work. Start evaluation with 4 logarithmic bins
over 70–150 km, 14 over 150–1200 km, and 8 over 1200–20000 km. Treat these and
50,000/100,000 outside/inside pair budgets as pilot settings, not validated
scientific defaults. Exact occupancy is adequate in the available geometry;
high-order convergence and coverage of the intended mission formation remain
unverified.

## 1. Current call graph and estimator

Paths below are relative to the repository root.

| Stage | Entry point and behavior |
|---|---|
| Replot only | `scripts/structure_function_publication.sh`: `plot_run()` reads existing JSON; maps 400/401/402 to p=0.9/0.7/0.5; calls `hsobs-plot-structure-function`. Multiple runs first call `structure_function_summary_ylims()`. This path does **no pair sampling**. |
| Recalculate | `hsobs-analyze` → `cli.analyze_main()` → `analyze.analyze_from_config()` → `config.load_analysis_config()` → registry `get_analysis('structure_function')`. YAML uses `safe_load`; paths remain relative to the execution working directory. `analysis.options` is merged with the `structure_function` section, with the latter taking precedence. |
| Settings | `analysis/malara-prod-{400,401,402}_sf-publication.yaml` → `analysis/structure_function.py::_settings()`: all-spacetime, adaptive stratification, seed 12345, chunks 100,000, target 500,000/bin, cap 500 million draws/member, 5 million bin-discovery pairs/member; nine automatic logarithmic bins; inclusive 50–30000 km limits; orders 1–6. |
| Load | `StructureFunctionAnalysis.run()` eagerly calls `io/hdf5.py::read_observations()` for all native data, errors and positions. It then calls `elsasser.compute_elsasser_analysis()` separately for each observed member and once for truth. |
| Align fields | `_read_selected()` reads a whole HDF5 dataset with `np.asarray` **before** indexing the MC member. Velocity-frame correction restores plasma velocity. Density normalization and mean subtraction are global per spacecraft. The common grid is the slower n/v grid; B and other quantities are linearly interpolated. `_uniform_cadence()` currently rejects irregular time axes. |
| Align positions | `_positions_for_group_member()` finds group-specific positions or nominal `/spacecraft/r/{v,n,B}`, aligns them to the Elsasser grid, and, for observed data without group-specific positions, adds `/errors/position_offset`. See coordinate policy below. |
| Find edges | Same-time mode gathers all baseline separations. All-spacetime `_spacetime_lambda_bounds_km()` scans the selected ordinal stream; adaptive mode substitutes a 5-million-pair uniform probe. Bounds are pooled across all groups/members, then `_lambda_edges_from_bounds_km()` makes geometric edges. Explicit `lambda_bins_km` skips this spacetime discovery pass. |
| Pair and reduce | `compute_pairwise_structure_functions()` or `compute_all_spacetime_structure_functions()` → helpers in `analysis/derived/pairs.py`. Each member is reduced independently. |
| Serialize | `StructureFunctionAnalysis.run()` converts arrays to lists; `write_result()` writes JSON and optionally plots. Current configs disable diagnostic plotting but separately enable publication plotting. |
| Plot | `cli.plot_structure_function_summary_main()` → `plot_structure_function_summary()` → `plot_structure_function_summary_row()`. Truth is plotted separately; observed summaries use member-axis 5th/50th/95th percentiles. Plot-time refitting changes fits, not stored moments or bins. |

For all-spacetime mode, flatten `(time, spacecraft, xyz)` in time-major order:
point index `a = t*S + i`. The population is the unique unordered pairs `a < b`
of distinct sample identities. It includes same-spacecraft/different-time,
cross-spacecraft/same-time, and cross-spacecraft/different-time pairs. There is
no restriction to the 36 cross-spacecraft baselines. Same-time mode instead uses
only `i < j` at each aligned time and has no same-spacecraft pairs.

For member m and bin q, define P[m,q] as those identity pairs with finite,
strictly positive Euclidean coordinate separation, within configured min/max
limits, and assigned to q. Internal bins are `[edge[q], edge[q+1])`; the final
edge is included. Both min/max pair limits are inclusive. Equal coordinates at
different identities are excluded. Fields/positions must all be finite on input;
the implementation fails rather than silently dropping invalid field samples.

For each retained pair, gather z+/z−, calculate the norm of the vector difference,
sum its powers separately for every configured order, and divide each sum by the
bin count. Selection never depends on increment magnitude. Ordinary sample means
are appropriate for uniform sampling within each P[m,q]. Empty bins have count
zero and NaN moments; fits mask them and record insufficient-fit-bin notes.
Overflow in moment powers is not explicitly diagnosed in this trajectory path.

**Representative separation is currently the geometric bin center**, not an
empirical pair-separation mean. `_fit_scaling_exponents()` performs an unweighted
log–log regression on populated, finite, positive moments at these centers.
Preserve that convention initially and additionally report sampled/exact mean
log-separation as a diagnostic. Changing the fitted representative is a separate
scientific change. The saved 300–8000 km fit includes centers near 602, 1225,
2494 and 5076 km; the center near 296 km is excluded. Refitting existing bins over
150–1200 km leaves only two centers and fails the minimum-three-bin requirement.

## 2. Supported selection algorithms

Let N=T*S, P=N(N−1)/2, C=chunk size, K=selected identities, B=bin count,
Q=candidate draws, and M=member count. Costs below exclude HDF5 preparation and
the O(M*N) arrays retained by the outer workflow.

| Method | Implementation; sampling/filter order | Allocation and avoidance | Correctness, reproducibility, reuse | Time / working memory |
|---|---|---|---|---|
| Same-time exhaustive | `compute_pairwise_structure_functions()` loops spacecraft combinations, vectorizes times, bins separations, then gathers increments. | Enumerates all same-time candidates; no distance pruning before norms; stores all baseline separations. No equal quotas. | Exact restricted population. Ignores `all`/`random` sampling budgets and seed. Same identities across equal-shaped members, but member positions change bins. Adaptive mode is explicitly disallowed. | Geometry O(M*T*S²), reduction includes B masks and order powers; separation storage O(M*T*S²). |
| Spacetime `all` | `derived/pairs.py::pair_chunks()` streams the upper triangle in lexicographic chunks. Filter after separation and bin assignment. | Does not materialize the full candidate list; evaluates every emitted distance. No quotas. `max_pairs` truncates to the **first** K ordinals. | Exhaustive if uncapped; a capped prefix is **not a uniform sample** and can bias bins/time coverage. Deterministic identities shared by all equal-shaped groups. | O(P) pair enumeration plus reduction; O(C) iterator memory. |
| Spacetime `random` | `pair_chunks()` draws `rng.choice(P, K, replace=False)` ordinals, then `_ordinal_pairs()` maps them to indices; geometry/binning follows. | Samples before bin assignment; no equal quotas; far distances still evaluated for selected candidates. Stores all K ordinals. | Uniform without replacement globally and conditionally within populated bins; unbiased conditional bin sample mean. Fixed seed and shapes give identical candidate indices across members, with different post-filter retention. | Mapping O(K log N + ceil(K/C)*N), because `_ordinal_pairs()` rebuilds an N-row lookup per chunk; O(K+N+C) explicit arrays. NumPy choice may additionally use O(P) workspace. |
| Spacetime `adaptive_stratified` | `adaptive_stratified_pairs()` draws uniform global ordinals with replacement in batches; deduplicates/shuffles each batch, evaluates separations, then retains unseen identities until each bin quota is full or Q is exhausted. | Quotas are equal, scalar targets; selection is after bin assignment. Evaluates far pairs, already-full bins and repeated candidates. Python per-candidate loop and retained-ordinal set; all retained indices are materialized before moment streaming. | Unique retained pairs; within a fixed geometry/bin population the random arrival ordering is exchangeable, hence uniform conditional on retained count. Stopping depends on geometry/counts, not increments. Seed reproduces selection only with the same geometry, edges, chunking and settings. Same seed gives common candidate prefixes but **not generally the same retained indices**. | Roughly O(Q(log C + log N + B/C) + ceil(Q/C)*N + K log N), including batch sort/mapping/Python decisions; O(N+C+K), with substantial Python-object overhead. Then selected separations are calculated again. |

The reduction adds approximately O(K*(B + number of orders)) work, because it
scans a mask per bin per chunk and gathers each field separately. There is no
supported lag-window selector, reservoir, spatial index, per-bin target array,
importance weighting, or convergence-driven budget in this workflow. Direct
raw-field sampling in `analysis/spatial_structure_function.py::sample_spatial_pairs()`
is a different estimator: random domain locations/directions at exact separations,
not sampling this trajectory pair population. It already has useful independent
per-separation RNG and deterministic checkpoint patterns to borrow later.

Two configuration traps: `max_pairs` is ignored by the adaptive branch, and its
eligibility function sees edges but not explicit pair min/max limits. Edges
outside those limits can consume quotas that are subsequently rejected. Explicit
edges should be intersected with the intended admissible domain in a new design.
Automatic edges depend on random sample extrema, so they need not cover the
entire intended 50–30000 km population. They also couple edges to MC membership.

## 3. Coordinates, timing and paired comparisons

Verified sources: `generate.py::_spacecraft_on_time_axes()`,
`_trajectory_positions_at_times()` and the MC sampling loop;
`io/cdf_positions.py::configuration_from_cdf()`;
`spacecraft/configuration.py::synthetic_configuration()`;
`fields/malara.py::evaluate_quantity()`; and the position loader above.

Both the current CDF snapshot loader and synthetic fallback construct
`r_i(t) = a_i + V*t` with a common configured velocity. This is trajectory motion
through the time-independent periodic Malara field; use the stored trajectory
velocity, **not** the 400 km/s plasma background in the field config. Current
structure functions use unwrapped Euclidean coordinates, even though field
evaluation wraps periodically. Do not introduce a minimum-image distance or
periodic KD-tree topology implicitly.

The available `outputs/malara-prod-400.h5` contains the **synthetic fallback**,
not the requested mission CDF formation: metadata records a missing CDF source.
All nine velocities are exactly `[105, 2, 245]` km/s; the maximum affine residual
on nominal stored positions is zero. T=2401, dt=0.125 s, duration=300 s,
|V|=266.559562 km/s; one lag step is 33.319945 km. The `r/n` and `r/v` arrays
are identical. Only the 400 HDF5 is present; 401/402 JSON diagnostics match 400,
but their underlying geometry has not independently been checked.

Generation evaluates observed fields at `t + errors/time_offset` and trajectory
positions at those shifted times plus constant per-spacecraft position offsets.
Nominal timestamps/positions remain stored. Analysis adds the position offsets
for observed binning but **does not add the hidden clock displacement**. For
affine motion that missing physical displacement would be `V*time_offset`.
Keep this existing distinction: timing error changes the sampled field, while
labels currently omit its hidden displacement. Adding it would change the
measurement/knowledge-error interpretation, not just optimize pair selection.
Group-specific position datasets supersede this offset reconstruction; selection
prefers an exact output-time-grid match across candidates, which can outrank an
earlier group-specific candidate requiring interpolation. Record the actual
source and policy explicitly in future outputs.

Edges are shared across truth/MC, but observed bin membership uses its perturbed
positions; truth uses nominal positions. Constant offsets preserve the affine
model but change cross-spacecraft lag windows. Same-spacecraft windows are
unchanged. For matching shapes, `all`/`random` already share candidate identities;
adaptive selection recomputes each population and retained list. No index cache
exists. Independent MC field errors are not redrawn by this analysis.

Recommended pairing policy:

1. Cache/reuse index plans when analysis geometry is identical. Keep sampling RNG
   separate from generation/error RNG. Preserve all independently supplied timing
   and position uncertainties.
2. For different geometries use common, deterministic per-bin random streams;
   each member must still sample uniformly from its own eligible population.
   Shared random numbers improve coupling but do not guarantee identical pairs.
3. If stronger index overlap is needed, use a common conservative proposal
   envelope over the known member geometries and a common candidate ordering,
   with member-specific eligibility rejection. Quota stopping must remain
   geometry-only. Record overlap fractions and distinguish MC error variability
   from pair-sampling variability.
4. Strictly identical retained indices cannot in general coexist with uniform
   sampling from different member-bin populations. Binning every member by a
   fixed nominal plan is a legitimate **separate estimand**, requiring an explicit
   `nominal` policy and review. Using only the common eligible intersection also
   changes the population; do not silently do either.

## 4. Measured occupancy and saved-output discrepancy

Full nominal population: **233,463,636** unique identity pairs. Compression by
spacecraft pair and discrete lag represents this using only **194,436** groups,
each weighted by its admissible start count. Summed weights exactly reproduce
N(N−1)/2. A 25-time, nine-spacecraft exhaustive check reproduces the compressed
counts exactly. No full-population increments were calculated.

Of the full population, 160,797 (0.069%) are below 50 km, 91,197,801 (39.063%)
exceed 30000 km, and 142,105,038 (60.868%) satisfy those limits. These are
population counts, not inferred from the saved selected-pair rejection fields.

Current edges, rounded for display; eligible counts use the actual JSON edges:

| Separation km | Eligible truth pairs | Saved retained truth | Eligible observed min–max |
|---|---:|---:|---:|
| 50.06–101.89 | 374,258 | 91,020 | 359,860–376,663 |
| 101.89–207.37 | 644,663 | 145,708 | 632,680–663,843 |
| 207.37–422.06 | 1,250,835 | 280,790 | 1,238,880–1,258,021 |
| 422.06–859.01 | 2,527,138 | 500,000 | 2,522,390–2,536,651 |
| 859.01–1748.32 | 5,122,997 | 500,000 | 5,108,826–5,134,820 |
| 1748.32–3558.34 | 10,202,525 | 500,000 | 10,193,240–10,218,747 |
| 3558.34–7242.22 | 20,069,794 | 500,000 | 20,056,581–20,078,862 |
| 7242.22–14739.96 | 37,745,775 | 500,000 | 37,745,325–37,754,735 |
| 14739.96–30000.00 | 64,164,655 | 500,000 | 64,161,194–64,165,571 |

The automatic edges cover 142,102,640 nominal eligible pairs, omitting another
2,398 pairs that satisfy the configured 50–30000 km limits. For the observed
compressed counts, all checked lag-group distances are more than 1e-8 km from
every edge, safely exceeding coordinate roundoff here; endpoint-sensitive cases
still require exact coordinate checks in the implementation.

The saved JSON reports 500 million draws/member and no target reached, with
3,517,518 retained truth pairs and observed totals 3,515,379–3,523,827. Each run
therefore records **25.5 billion** adaptive draws plus up to **255 million** edge
probe pairs across its 51 members. Only about 0.704% of candidate draws become
retained identities. The rest include far pairs, duplicates and full-bin rejects;
the saved counters cannot separate these. Their zero skipped-distance counters
describe already-selected pairs, not rejected adaptive candidates.

**Provenance issue to resolve before result-preservation claims:** the saved
truth counts in the first three bins are 91,020 / 145,708 / 280,790. They do not
match the current code's expected occupancy on the current HDF5. A fresh bounded
uniform probe of one million unique pairs gave 1,594 / 2,700 / 5,434, consistent
with the exact geometry counts above. A fresh adaptive run capped at one million
draws, target 5,000, retained 1,592 / 2,776 / 5,000 in those bins. With uniform
replacement proposals and 500 million draws, an uncapped bin should see about
`1-exp(-500e6/233463636) = 88.25%` of its unique population: about 330,000 in the
first bin, not 91,020. Saved metadata says July 27 creation despite later file
mtime; it includes no input-content/code hash or actual retained pair identities.
This audit cannot attribute the discrepancy to a particular historical change.
Do not treat the JSON as a verified replay of today's sampler/input.

Candidate 26-bin occupancy follows. Counts use nominal 400 geometry; MC min/max
use the 50 stored position-offset realizations without gathering any fields.
"Groups" means spacecraft-pair/discrete-lag groups; repeated starts are not
independent observations. Distinct separations are counted after rounding to
1e-6 km solely for this diagnostic.

| Separation km | Eligible truth | Observed min–max | Groups | Distinct separations |
|---|---:|---:|---:|---:|
| 70.00–84.69 | 110,356 | 98,359–115,156 | 46 | 46 |
| 84.69–102.47 | 119,924 | 110,336–129,520 | 50 | 42 |
| 102.47–123.98 | 129,493 | 119,903–141,481 | 54 | 54 |
| 123.98–150.00 | 162,999 | 155,806–172,588 | 68 | 60 |
| 150.00–174.02 | 153,356 | 143,768–160,548 | 64 | 56 |
| 174.02–201.89 | 162,888 | 153,307–172,473 | 68 | 60 |
| 201.89–234.21 | 191,564 | 189,171–198,747 | 80 | 72 |
| 234.21–271.72 | 220,202 | 213,020–236,954 | 92 | 84 |
| 271.72–315.23 | 258,376 | 241,631–263,159 | 108 | 100 |
| 315.23–365.70 | 279,718 | 265,369–289,292 | 117 | 109 |
| 365.70–424.26 | 341,678 | 339,290–353,623 | 143 | 127 |
| 424.26–492.20 | 393,911 | 386,757–405,841 | 165 | 149 |
| 492.20–571.02 | 477,013 | 465,086–488,937 | 200 | 175 |
| 571.02–662.45 | 514,628 | 500,336–526,540 | 216 | 200 |
| 662.45–768.53 | 628,204 | 623,441–642,484 | 264 | 232 |
| 768.53–891.60 | 696,219 | 684,342–703,337 | 293 | 269 |
| 891.60–1034.37 | 842,097 | 832,603–853,958 | 355 | 315 |
| 1034.37–1200.00 | 954,085 | 946,973–958,820 | 403 | 363 |
| 1200.00–1705.74 | 2,899,579 | 2,890,164–2,904,294 | 1,230 | 1,110 |
| 1705.74–2424.62 | 4,069,938 | 4,055,907–4,083,971 | 1,740 | 1,571 |
| 2424.62–3446.47 | 5,759,208 | 5,749,928–5,770,790 | 2,490 | 2,241 |
| 3446.47–4898.98 | 8,042,476 | 8,035,621–8,051,583 | 3,534 | 3,182 |
| 4898.98–6963.65 | 11,137,259 | 11,132,750–11,143,899 | 5,010 | 4,522 |
| 6963.65–9898.46 | 15,345,182 | 15,338,690–15,349,388 | 7,144 | 6,432 |
| 9898.46–14070.16 | 20,700,669 | 20,694,481–20,706,732 | 10,141 | 9,141 |
| 14070.16–20000.00 | 27,243,678 | 27,230,537–27,245,659 | 14,417 | 12,993 |

No proposed bin is empty. Inside 150–1200 km there are 64–403 lag groups,
56–363 distinct separations and 44–45 spacecraft pairs including self-pairs.
Outside, the narrow low bins cover only 30–42 spacecraft pairs. Large counts
therefore do not imply equivalent directional coverage or independent samples.
This layout is a reasonable pilot, but do not narrow it further without
examining per-pair/per-lag concentration and temporal block variability. Repeat
occupancy checks on the intended mission formation before adopting final edges.

## 5. Bounded performance measurements

Environment: repository `.venv/bin/python`, NumPy 2.2.6, SciPy 1.15.3; one
OpenBLAS/OMP thread; in-process `perf_counter`, three repetitions, fresh process
per method. Prepared data are real 400 sampled-truth Elsasser vectors, first
256 aligned times × nine spacecraft: 2304 points, 2,653,056 possible pairs.
All methods use nine `geomspace(70,5000,10)` edges, orders 1–6, both fields,
chunk=50,000, seed=12345, no pair-limit overrides. Fields/preparation are excluded
from kernel wall time. Budgets differ intentionally to show native algorithm
behavior; this is **not** an equal-precision comparison.

| Existing method | Median kernel s (range) | Candidates / selected | Contributing pairs | Process peak RSS; pre-kernel high-water |
|---|---:|---:|---:|---:|
| All | 0.816 (0.750–0.840) | 2,653,056 / 2,653,056 | 2,166,235 | 87.6 / 80.6 MiB |
| Random, n_pairs=200,000 | 0.0934 (0.0883–0.0967) | 200,000 / 200,000 | 163,371 | 106.0 / 80.8 MiB |
| Adaptive, target=2000/bin, cap=1,000,000 | 0.128 (0.124–0.132) | 200,000 / 18,000 | 2000 in each bin | 92.9 / 80.5 MiB |

All/random reject 18.35%/18.31% outside the benchmark bin range. Adaptive records
no post-selection rejects; its candidate-stage rejection reasons are unavailable.
RSS is Linux `resource.getrusage().ru_maxrss`, an absolute process high-water mark
including imports/preparation and repeated calls, not allocation tracing. These
numbers are **not production memory measurements**.

Separate `cProfile` passes add overhead: adaptive 0.189 s total, of which selection
is 0.178 s (94%); selector Python self-time 0.124 s (65% of total); ordinal mapping
0.0147 s; moment power sums 0.00183 s (1%). All mode takes 0.759 s profiled:
pair enumeration 0.0303 s, power sums 0.140 s, all norms 0.148 s; main-function
self-time 0.398 s includes masks, indexing, allocations and reductions. Thus
adaptive rejection/selection dominates its runtime; all-mode work is mainly
gathers, masks, norms and moments, rather than creating index pairs alone.

A separate staged replay on the **full 21,609-point geometry**, using 200,000
unique random pairs and current nine edges, retained 121,561. Median stage times:

| Stage | Seconds |
|---|---:|
| Select ordinals and map indices | 0.0291 |
| Gather positions and calculate separation | 0.0198 |
| Bin and build masks/index groups | 0.00585 |
| Gather both fields and calculate magnitudes | 0.0223 |
| Sum orders 1–6 for both fields | 0.00566 |

These are isolated stages, not an additive profile of the exact production loop.
They show that moment powers alone are inexpensive relative to selection and
gathers. Three actual observed members on the 256-time slice took
0.384 seconds in the adaptive kernel, each examining 200,000 candidates.
This confirms repeated selection work; no full MC computation was run.

Full-file preparation was measured separately: eager `read_observations()`
1.13 s, 270.9 MiB peak RSS; three individual observed Elsasser calls 0.391 /
0.401 / 0.380 s, reaching 333.8 MiB while the eager read remained alive. Each
member call rereads the full observed datasets before slicing. Linear
extrapolation suggests roughly 20 s for 50 such preparations under similar cache
conditions, **not a measured complete-run time**. The outer first-pass dictionaries,
subsequent stacks and Python retained-pair set add further production memory.
At 3.5 million retained identities, int objects, lists, set and left/right arrays
alone plausibly occupy hundreds of MiB; full-run peak RSS was not measured.

The audit's straightforward compressed geometry counter took 6.73 s for all 51
current-bin populations and 6.96 s for all 51 proposed-bin populations, including
Python list construction, unique-distance diagnostics and binning. This is a
counting prototype, **not a benchmark of an implemented lag sampler**.

## 6. Geometry-aware sampler design

For verified `r_i(t)=a_i+V*t`, choose `b_ij=a_j-a_i` and `tau=t_j-t_i`.
Then `r_j(t_j)-r_i(t_i)=b_ij+V*tau`; the stated tau-star and d-min formulae apply
with exactly this sign. Clamp tiny negative roundoff in d-min squared to zero;
handle V=0 separately (constant baseline, all times eligible or none).

For V≠0 and bin lower L/upper U, the upper-radius condition gives an interval
centered on tau-star with half-width `sqrt((U²-d_min²)/|V|²)`, if reachable.
The lower-radius condition removes the central interval with half-width
`sqrt((L²-d_min²)/|V|²)` when L>d-min. Thus there are at most two lag windows.
Clip to finite time coverage and intersect with pair min/max limits. Do not
blindly round roots: generate a conservative neighboring integer-lag range and
test endpoint candidates using the existing norm/bin predicate. Internal upper
edges remain exclusive; the final edge/max limit remain inclusive. Floating
point equivalence must be tested on stored coordinates near boundaries.

With common uniform cadence and T times:

* For i<j, include signed lags `k=-(T-1)..T-1`; include k=0 once. Start indices
  satisfy `max(0,-k) <= t_i < min(T,T-k)`; weight `w=T-|k|`.
* For i=j, use k=1..T−1 only. Never include zero lag or both signs.
* Convert sampled endpoints back to flattened indices and canonicalize a<b.
  These rules reproduce the current unordered population without double counts.

Build eligible `(i,j,k,w)` groups per bin. Their total weight is exact eligible
population size, so cap targets at availability. Uniformly draw an integer rank
over the concatenated starts of all groups; locate its group by cumulative w
and use the residual rank as the start index. This automatically weights both
spacecraft pair and lag correctly. Equivalently choose group proportional to w,
then start uniformly. Drawing ranks **without replacement** gives an exact
uniform subset with no duplicate pairs; use a sparse sampling method or stream
an exhaustive bin when near census, rather than an unbounded population-sized
permutation. A fixed-size subset selected by these geometry-only rules uses the
ordinary sample mean.

Initial plan construction can simply vectorize all discrete lags: O(S²*T)
geometry, much smaller than O(S²*T²) pair enumeration, O(S²*T) group storage.
Analytic windows then restrict that work further; arithmetic-series cumulative
weights permit more compact plans without listing every valid lag if needed.
Sample mapping is O(K log G), G=eligible groups, followed by current chunked
increment gathering and accumulation. The current full geometry compresses
233 million candidates to 194 thousand groups (~1200× fewer distance evaluations
for counting); this is a count ratio, not an established end-to-end speedup.

Observed constant position offsets are incorporated into each member's a_i.
Do not assume all future inputs are affine: check every trajectory against a
common V and timestamp model, accounting conservatively for residuals at bin
edges. A merely small residual is insufficient for unchecked classification.
If the check fails, fall back or use conservative candidate windows followed by
exact norms, never silently classify by the approximation.

The current Elsasser pipeline requires common uniform instrument grids after
alignment. Different n/v/B cadences are supported through that alignment;
per-spacecraft irregular grids are not currently supported. If introduced later,
affine geometry still gives time windows: for each left time, binary-search the
sorted right times for each permitted interval, count admissible right samples,
then sample a global count rank. Preserve self-pair ordering and finite endpoints.
With common irregular times this costs approximately O(S²*T log T), with no
assumption of a constant number of starts per lag. Moving formations need the
general spatial fallback, not the unchanged constant-baseline formula.

## 7. Alternatives and fallback

| Alternative | Assessment |
|---|---|
| Stratification | Keep it, but sample directly from eligible populations and allow different per-bin targets. This improves separation resolution without changing within-bin weighting. Equal draws per baseline/lag would bias the population unless weighted by their sizes. |
| Streaming reservoirs | Exact per-bin uniform sampling without replacement while enumerating eligible identities; memory O(sum targets), time at least O(eligible visits), or O(P) without pruning. Useful with spatial/chunked enumeration. Reservoir replacement and uniqueness must operate on identities, not repeated proposals. Geometry-only quota selection is safe. |
| KD-tree | Good general-trajectory fallback for sparse small-radius neighborhoods. Typical low-dimensional build O(N log N), storage O(N); query time is output-sensitive and can approach O(P). Rebuild per changed member geometry. Use exact `eps=0`, conservative search radius and exact final norm/bin checks. |
| Vectorized chunked enumeration | Already the reference approach. It avoids full pair-list storage and gathers increments only after geometry masks. Optimize Python bin loops/memory only after measurement. It still calculates every candidate separation; reducing chunk size cannot fix O(P) work. Good for small/dense cases and checking the optimized method. |
| Importance sampling | Unnecessary for the primary exact rank design. If unequal identity inclusion probabilities pi are used, estimate a known-population bin mean with `sum(y/pi)/population_size` (Horvitz–Thompson); with independent replacement draws use the corresponding draw-probability estimator. A self-normalized ratio is generally only approximately unbiased. Log-fit/flatness transformations have their own finite-sample bias. Never reject/select by increment size without a validated weighted design. |

Measured KD-tree feasibility on all nominal 400 points: building `cKDTree`
took 0.00351 s; `query_pairs(1200, output_type='ndarray')` took 0.177 s and
materialized **6,943,884** pairs, occupying **105.96 MiB** for indices alone.
This includes separations below the desired lower limit. The 102,141,873
pairs within 20000 km would require approximately **1.52 GiB** just for a full
two-int64 pair array (estimate from exact counts, not a materialized query).
At 30000 km it would be about 2.12 GiB. A single broad `query_pairs` call is
therefore unsuitable as the production fallback.

Instead query small blocks of left points with `query_ball_point`, retain only
neighbors with right index>left, calculate exact separations, and feed the
per-bin reservoirs. Bound block size by observed neighbor density; a dense
single query can still return O(N) neighbors. For repeated-member coupling,
shared identity-based random priorities can provide uniform reservoirs and
overlap where populations coincide; use a well-defined reproducible priority
scheme. A future count/rank two-pass fallback avoids storing a large reservoir
but still visits the eligible neighbors twice. Benchmark these designs before
choosing a default threshold; tree construction itself is not the limiting cost
in the measured formation.

SciPy documents unique unordered radius pairs and materialized output for
[`query_pairs`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query_pairs.html),
point/block radius searches for
[`query_ball_point`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query_ball_point.html),
and optional toroidal `boxsize` for
[`cKDTree`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html).
Use no periodic topology here to preserve the current metric. NumPy/SciPy already
suffice; no dependency is needed. The actual KD probe used installed SciPy 1.15.3.

## 8. Proposed configuration, diagnostics and expected savings

Retain explicit `lambda_bins_km`; construct the pilot edges as
`concat(geomspace(70,150,5), geomspace(150,1200,15)[1:], geomspace(1200,20000,9)[1:])`.
Set matching 70/20000 pair bounds and an explicit `scaling: [150,1200]` fit range;
point publication `fit_range` at that name. Do not modify the current production
YAML until validation/review. Proposed new sampling settings, **not supported yet**:

```yaml
pair_sampling:
  mode: geometry_stratified
  fallback: spatial_reservoir  # exhaustive chunking for small/dense inputs
  random_seed: 12345
  target_pairs_per_bin: [...] # length = number of bins; initially 50k/100k/50k
  max_selected_pairs: 2000000
  geometry_policy: per_member # preserve current perturbed-position labels
  coupling: common_random_numbers
```

Specify deterministic global-budget allocation before drawing, e.g. scale target
weights proportionally with a stable remainder rule, then cap at population
availability and redistribute unused quota deterministically. Avoid random
early exhaustion of a global cap. Persist seed, bit generator, sampler version,
input/geometry/config hashes, actual edges, per-bin stream identity and budgets.
Streams should be keyed by seed and stable bin identity/bounds, not scheduling
or MC error RNG; adding members/changing chunking must not change old sequences.
Use a lazy sparse Fisher–Yates permutation of integer ranks, or an equivalent
documented method, if budget-independent sample prefixes are required; an
arbitrary fixed-size `choice` call does not guarantee that prefix property.

Required bin diagnostics: total eligible identities; requested/selected/unique
and retained-after-validity counts; zero/shortfall reasons; candidate/rejection
reason counts where applicable; unique endpoint identities and timestamps;
spacecraft-pair and lag histograms/concentration; member-to-truth index overlap;
representative separation and empirical mean log-separation; timing policy and
position source. Separate identity counts from effective statistical independence.

For n=5,6 record maximum fractional contribution, top-1% contribution, and
`(sum |dz|^n)^2 / sum |dz|^(2n)` as a **contribution concentration** measure, not
an independence-corrected effective sample size. Also record deterministic
quarter/half/full-budget moment checkpoints and time-block variability. Use
multiple independent sampling seeds to diagnose sampling error separately from
the error MC. Preserve all extreme increments; do not clip or reject them.

Prefer fixed budgets established by an independent pilot. If a pilot says a bin
needs more samples, use a fresh fixed-budget estimation stream (or recompute the
final estimate from an independent holdout); retaining pilot data under an
outcome-dependent stopping rule can bias ordinary means. Sophisticated adaptive
stopping is not justified before a fixed-budget pilot. Compare high-order
convergence across seeds and time blocks, not just raw pair count.

The proposed 12 outside bins ×50k plus 14 inside ×100k select exactly two million
pairs/member in the checked formation (all populations exceed the targets).
Compared with saved ~3.52 million, this is ~43% fewer increment evaluations.
It eliminates the 500-million-draw rejection search and 5-million-pair automatic
edge probe. A 250× ratio of old candidate draws to new selected pairs is an
**operation-count estimate**, not a measured speedup. One full run would process
102 million selected pairs over 51 members instead of recording 25.5 billion
rejection proposals. Accuracy at this budget is not established. After removing
selection waste, field preparation, increment gathers and moment reductions will
matter more. Fixing whole-dataset MC reads is a separate, small follow-up worth
profiling after the sampler work; it is not implemented by this audit.

## 9. Validation gates before production

An inexpensive audit check used a synthetic affine dataset with 16 times ×three
spacecraft, deterministic Gaussian vectors (seed 19), and edges
`[0.01,1,2,4,8,20]`. Exhaustive populations were 48/135/243/378/324. Across 256
existing-adaptive runs (seeds 0–255, target eight/bin, chunk 500, cap 20k), every
retained pair was eligible and unique. Means of all 30 order/bin estimates were
within 1.83 estimated standard errors of their exhaustive values; descriptive
inclusion-frequency chi-square/df values were 0.90–1.13. These correlated
inclusion counts are a sanity check, not a formal independent-category test or
validation of the proposed algorithm.

Keep `pair_chunks(mode='all')` and current moment calculations as reference until
the following targeted checks pass:

1. **Population identity:** small exhaustive equality of eligible pair sets/counts
   for every bin, including self-pairs, signed cross lags, zero lag, finite ends,
   V=0, nonzero time origins, offset geometries and canonical unordered indices.
2. **Uniformity:** enumerate tiny populations and test repeated-seed inclusion
   frequencies with finite-population-aware simulations; verify group/lag/start
   frequencies against exact weights, especially unequal baseline populations
   and negative lags. Check duplicate-free sampling and census behavior.
3. **Numerical boundaries:** exact edges, immediately adjacent `nextafter` values,
   positive/min/max/final-edge exceptions, tangencies and root roundoff. Compare
   against norms of the actual stored coordinates, not only analytic distances.
4. **Moments:** full-selection sums/counts agree with exhaustive values to floating
   reduction tolerance; fixed-budget means agree within independently measured
   sampling error over many seeds, including high-order heavy-tail examples.
   Empty bins remain explicit and excluded from fits; overflow is diagnosed.
5. **Reproducibility:** fixed seeds reproduce indices/results; deterministic
   prefixes survive chunking, checkpoint, worker-order and member-count changes.
   Record sampler versions rather than promising equality with legacy streams.
6. **Coordinate/fallback behavior:** non-affine/moving formations trigger exact
   fallback; differing native uniform cadences align as before; irregular cadence
   either remains a clear validation error or gets separate supported tests.
   Group-specific position precedence, stored offsets, and hidden clock errors
   must remain explicit and independently verified.
7. **Truth/MC:** identical geometry reuses identical indices; changed positions
   preserve each member's eligible population/uniformity and report overlap.
   Error draws/field samples remain unchanged. Separate sampler variability from
   MC measurement variability with a small fixed-input, multi-seed comparison.
8. **Historical consistency:** first resolve the saved-output occupancy discrepancy
   using input/code/config fingerprints or a bounded reproducible historical
   case. Then compare old/new methods on identical explicit edges, geometry,
   pair populations and budgets over multiple seeds. Changing edges or fit range
   is not expected to preserve old fitted exponents.
9. **Performance:** benchmark equal target accuracy/retained budgets on small and
   medium cases, report per-stage timings, candidate/eligible/retained counts,
   and fresh-process peak RSS. Test sparse, dense and empty-bin geometries.
   Verify no population-sized allocation for large sparse-rank samples or broad
   spatial queries. Establish regression thresholds from repeated local baselines
   before authorizing a production run.

### Reproducing the inexpensive measurements

The audit used a temporary script under `/tmp`; production Python was unchanged.
Use `PYTHONPATH=src MPLCONFIGDIR=/tmp/sf-audit-mpl OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1 .venv/bin/python` for equivalent probes. Benchmark arguments,
shapes, versions, seeds and repetition counts are specified above. Run each
method in a fresh process and exclude data preparation; use a separate profile
pass so profiling overhead is not confused with the reported wall times.

The essential exact geometry-count reproducer is:

```python
import h5py
import numpy as np

with h5py.File('outputs/malara-prod-400.h5') as f:
    t = f['time/v'][:]
    r = f['spacecraft/r/v'][:]
T, S = len(t), len(r)
V = (r[0, -1] - r[0, 0]) / (t[-1] - t[0])
edges = np.r_[np.geomspace(70, 150, 5),
              np.geomspace(150, 1200, 15)[1:],
              np.geomspace(1200, 20000, 9)[1:]]
counts = np.zeros(len(edges) - 1, dtype=np.int64)
for i in range(S):
    for j in range(i, S):
        k = np.arange(1, T) if i == j else np.arange(1-T, T)
        d = np.linalg.norm(r[j, 0] - r[i, 0] + k[:, None]*(t[1]-t[0])*V, axis=1)
        q = np.searchsorted(edges, d, side='right') - 1
        q[d == edges[-1]] = len(edges) - 2
        valid = (d > 0) & (q >= 0) & (q < len(counts))
        np.add.at(counts, q[valid], (T-np.abs(k))[valid])
print(counts)
```

For observed occupancy, add each stored `errors/position_offset[member]` to the
initial spacecraft coordinates in the same calculation. This reproducer is
valid only after verifying the affine/common-cadence assumptions above.
