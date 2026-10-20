# Structure-function member parallelism audit

Investigation date: 2026-10-20. Static inspection only; no source changes,
parallel implementation, tests, benchmarks, or production calculations were run.
Scope: scheduling truth and measurement-error MC members in separate processes.

## Assessment

**Member-level process parallelism is feasible without changing the scientific
algorithms, but wrapping the current runner once per member is insufficient.**
Preserve common separation edges, group-wide automatic fit ranges, each member's
existing RNG initialization and accumulation order, and deterministic assembly.
Geometry cache diagnostics also need explicit handling to match serial output.
Exact equivalence is a proposed acceptance criterion, not an experimentally
verified finding of this audit.

The current analysis has no pool or worker option. Its member calculations are
sequential. “Global per spacecraft” density normalization and mean subtraction
mean averages over time within one member, not across the MC ensemble.

## Relevant code

Locations are relative to the repository root.

| Location | Responsibility |
| --- | --- |
| `src/hs_observatory/analysis/structure_function.py:54` | Selection and sampling settings. |
| Same file, `:269`, `:290`, `:541` | Common bin edges and geometry-bound prepass. |
| Same file, `:433`, `:577` | Same-time and all-spacetime member evaluation. |
| Same file, `:348`, `:403`, `:408` | Fits, moment sums, and moment means. |
| Same file, `:841`, `:896`, `:908` | Member positions, Elsasser configuration, runner. |
| Same file, `:1081`, `:1127`, `:1200` | JSON/plot writing and ensemble intervals. |
| `src/hs_observatory/analysis/derived/pairs.py:53`, `:105` | Enumerated/random pairs and adaptive sampling. |
| `src/hs_observatory/analysis/derived/geometry_pairs.py:114`, `:191`, `:218` | Geometry plans, bounded cache, seeded identity sampling. |
| `src/hs_observatory/analysis/elsasser.py:124`, `:252` | HDF5 member slicing and normalization/alignment. |
| `src/hs_observatory/analysis/derived/elsasser.py:27`, `:80` | Time means and Elsasser fluctuations. |
| `src/hs_observatory/io/hdf5.py:14`, `:90` | Compressed observation storage; eager generic reader. |
| `src/hs_observatory/cli.py:57`, `src/hs_observatory/analyze.py:11` | CLI and run-before-write workflow. |
| `scripts/run-structure-functions.sh:1` | Three concurrent configurations; numerical threads limited to one. |
| `src/hs_observatory/analysis/structure_function_local.py:35`, `:148`, `:201` | Downstream local slopes, MC slope intervals, HDF5 export. |

## Independence and common preparation

Once positions, aligned fields, and bin edges are fixed, each member's counts,
moments, flatness, and fits have no dependency on another member's field values.
The existing evaluators can accept a singleton leading member dimension. Leave
all their within-member loops and reductions unchanged.

Two scientific dependencies must remain coordinated:

1. The runner obtains **one set of separation edges across all selected groups
   and members**. Same-time mode collects pair separations; spacetime mode
   reduces per-member bounds. Adaptive automatic binning uses random probes;
   geometry-stratified automatic bounds are deterministic. A worker must not
   independently derive its own edges.
2. If `scale_ranges_km` is absent/empty, each evaluator defines a group's default
   `total` range from bins populated by **any member of that group**. Singleton
   calls can therefore choose different ranges. Assemble counts and `S_n`, then
   reproduce the evaluator's existing range logic and call the unchanged
   `_fit_scaling_exponents` on the complete group. Preserve its empty-group-data
   fallback to bin edges. The separate `_scale_ranges_km` helper has a different
   empty-data fallback; substituting it would change behavior.

Also preserve the current rejection of incompatible member shapes/time lengths
that stacking enforces; independent workers must not silently accept a dataset
that the serial runner rejects.

## RNG initialization, consumption, and shared identities

There is no persistent ensemble RNG and no member-index-derived seed. Do **not**
introduce `seed + mc_index`, spawned child streams, PID seeds, or worker-local
streams advanced across tasks. Each task must recreate the same generators at
the same call boundaries as serial evaluation.

| Path | Current initialization and consumption |
| --- | --- |
| Same-time pairs | No RNG; fixed spacecraft combinations and time samples. |
| Spacetime `all` | No RNG; lexicographic unordered-pair enumeration, optionally truncated by `max_pairs`. |
| Spacetime `random` | Every `pair_chunks` call makes `default_rng(random_seed)` and calls `choice(total_pairs, size=count, replace=False)` once. Ordered sampled ordinals are then chunked. |
| Adaptive binning prepass | A separate `pair_chunks(mode='random', n_pairs=binning_n_pairs, seed=...)` invocation per member. Its RNG is discarded, not passed to evaluation. |
| Adaptive evaluation | Every `adaptive_stratified_pairs` call makes `default_rng(random_seed)`. Each batch draws integer ordinals with replacement, deduplicates with `unique`, shuffles them, then retains unique identities until bin quotas or `max_candidates` stop the loop. Finally it shuffles the concatenated retained identities. |
| Geometry-stratified | Each bin constructs `Generator(PCG64(SeedSequence([seed, *edge_words])))`; `edge_words` are the two little-endian float64 edge bit patterns viewed as four uint32 words. Sparse Fisher–Yates uses scalar integer draws; a full-population census emits ordered ranks without draws. |

Consequences:

- With equal point counts, random mode reuses the **same ordered candidate
  identities across MC members and truth**. Their separations/bin acceptance can
  differ. Both Elsasser fields use the same selected pairs within a member.
- Identical geometry, edges, settings, and seed give identical adaptive retained
  pairs and order. Changed geometry can change acceptance, stopping batch,
  number of RNG draws, and final permutation, even with the same initial seed.
  Common candidate streams do not guarantee common retained pairs.
- Geometry sampling has no member-order input. Identical plans reproduce the
  same sampled identities; changed plans/populations can map seeded ranks to
  different identities. Cache hits do not consume RNG or alter sampling.
- Preserve `chunk_size` exactly. It affects adaptive draws/stopping and the
  numerical summation grouping. Geometry identity streams are chunk-independent,
  but changing numerical accumulation chunks can still change floating-point
  results. Identity equality alone is not sufficient.
- Fixed configuration and a pinned NumPy/numerical environment should permit
  serial/process equivalence. Bitwise reproducibility across workstation/Puma
  CPU architectures, NumPy versions, or BLAS builds is not established.

## Loading, memory, and HDF5 safety

The current runner reads IDs and dataset shapes initially; it no longer calls
`read_observations`, which would eagerly materialize all 50 members. The
Elsasser reader slices `observed/{n,v,B}[mc_index]` before conversion. Positions
with a member dimension and `errors/position_offset` are also sliced first.
Time axes, nominal positions, spacecraft velocity/frame metadata are shared
input data, read into each process as needed. Position offsets are applied only
to observed geometry; preserve existing frame correction and interpolation.

However, the serial runner retains full Elsasser dictionaries for all selected
members in `first_pass`, then stacks positions and both Elsasser fields. Those
dictionaries include aligned density, velocity, magnetic fields, fluctuations,
and normalization arrays, not just the SF inputs. Passing this structure to
every worker would defeat the memory goal.

Use path-based tasks and independently read only the selected member. A bounded
worker pool then holds approximately `N_workers × peak_member_workspace`, plus
parent result storage. For `P = n_time × n_spacecraft`, positions and two
float64 Elsasser vectors alone occupy `72P` bytes per active member, before
alignment copies and sampler storage. Adaptive sampling additionally retains
Python lists/sets of identities and full selected index arrays; random mode
allocates sampled ordinals; geometry mode stores plans and sparse permutations.
These can dominate memory. No numerical memory bound was measured.

Independent read-only opens are supported by [h5py's multiprocessing guidance](https://docs.h5py.org/en/stable/mpi.html).
Open and close files inside workers; never pass/inherit live HDF5 handles. No
MPI-enabled h5py or shared-file writing is needed. Keep input immutable during
the run. Gzip/automatic HDF5 chunking can make different member reads decompress
overlapping storage chunks, so disk/decompression contention and per-process
chunk caches can limit speedup even without loading the complete ensemble.

## Worker contract and deterministic assembly

Proposed inputs: absolute input path, validated SF/Elsasser settings, shared
fixed bin edges, dataset name, original MC index (truth uses zero), and a stable
output slot. Submit only small picklable descriptors, not ensemble arrays or
HDF5 objects. Original MC identity and output position are distinct: preserve
explicit lists such as `[4, 1, 3]` in that order; `all` remains ascending.

Proposed outputs: slot and identity; singleton member products (counts, both
fields' `S_n`/flatness, time and mode-specific diagnostics/separations); position
source/notes; shape checks and geometry-cache key when applicable. Keep NumPy
arrays until parent serialization. Do not return the full Elsasser dictionary
or write final files from workers.

The parent should:

1. Restore group order and member slots, independent of completion order.
2. Concatenate only documented member axes. Verify/copy invariant orders, bin
   edges/centers, fit-range dictionaries, and scalar configuration once; preserve
   `None` versus arrays and same-time/spacetime schema differences.
3. Finalize fits for each assembled group with the existing helper, including
   automatic ranges and group-relative member numbers in fit notes. A singleton
   worker's provisional fit notes all say `member 0`; simply concatenating them
   would corrupt provenance. Recompute notes rather than string substitution.
4. Preserve `n_members`, optional `mc_indices`, position-source lists, units,
   configuration, and the existing sorted/deduplicated aggregate notes.
5. Keep plotting and `_ensemble_interval` in the parent. The core SF JSON stores
   individual products, not percentiles. Plotting derives 5th/50th/95th
   `nanpercentile` intervals for SFs, exponents, flatness, and counts. Local slopes
   are downstream products, not calculated by this runner.

**Metadata barrier:** one two-entry `GeometryPlanCache` currently spans all
groups in serial evaluation order. `geometry_sampling.members[*].plan_reused`
is observable and depends on that order and eviction, not science. Per-process
cache flags would differ with scheduling. For strict legacy metadata parity,
replay the same key-based two-entry LRU in the parent, including cross-group
reuse, and report those logical serial reuse flags. Clearly document this
meaning; physical worker cache statistics belong in execution logs. Otherwise
this field needs an explicit approved exclusion from equality tests. Do not
silently ignore it as though only timestamps changed.

## Failures, persistence, and smallest safe implementation

`analyze_from_config` currently completes `run` before calling `write_result`.
Exceptions therefore prevent new output during computation, but `write_result`
directly overwrites JSON before plotting: serialization/interruption can leave
a truncated file, and plotting failure can leave JSON without its figures.
There is no member checkpoint/resume mechanism.

Recommended later implementation, in order:

1. Keep a real serial `--workers 1` baseline and add a positive integer
   `--workers N` only for this SF path; leave unrelated analysis methods alone.
2. Retain an ordered preparation pass for common bins and validation, discarding
   each member's large fields after use. Initially use the existing readers and
   calculations even if workers must reread/recompute preprocessing. This bounds
   retained preparation memory at the cost of repeated I/O. The automatic
   geometry/binning prepass remains a serial scaling limit.
3. Dispatch singleton evaluations using a module-level importable worker and
   `ProcessPoolExecutor` with explicit `multiprocessing.get_context('spawn')`.
   Bound in-flight tasks/results; leave the SF kernels/samplers unchanged.
4. Assemble and finalize as above, then publish exclusively from the parent.
   Surface exceptions with dataset/original index and chained traceback; fail
   the whole run, cancel pending tasks, and clean temporary files. Ordinary
   cancellation cannot stop already-running futures immediately; handle broken
   pools/OOM and interrupts without assembling an incomplete ensemble. Executor
   behavior is described in the [Python reference](https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor).
5. Stage JSON and figures under temporary names in their destination filesystem,
   then atomically replace completed artifacts only after successful evaluation
   and plotting. Separate renames do not make a multi-file transaction; advertise
   completion only after all replacements. Preserve an old successful result on
   computation failure. Individual-member restart requires additional versioned,
   atomic checkpoints keyed by input fingerprint, original index, common edges,
   settings/seed/chunks, and algorithm/environment version; defer it initially.

For `n_mc: 0`, generation writes zero-length observed arrays
(`generate.py:320`). Truth-only selection works. Selecting observed, including
the current default, raises “selects no observed realizations.” Preserve that
behavior in both serial and parallel modes, rather than inventing a member or
silently switching to truth. Publication plotting still requires both groups.

## Workstation and Puma execution

Use one pool within one node and one SLURM task. A proposed launch is
`--nodes=1 --ntasks=1 --cpus-per-task=N`, then
`hsobs-analyze CONFIG --workers "$SLURM_CPUS_PER_TASK"` (not currently implemented).
SLURM documents [CPU-per-task allocation and its environment variable](https://slurm.schedmd.com/sbatch.html#OPT_cpus-per-task).
A local process pool does not span nodes. Puma partition limits, binding, memory
allocation, and installed Python/NumPy/HDF5 versions must be checked on the
cluster before benchmarking; this audit did not inspect a cluster allocation.

Choose explicit `spawn` for portable HDF5 isolation, rather than relying on
version/platform defaults. `forkserver` can be evaluated later; avoid `fork`
after HDF5 or threaded numerical libraries initialize. Workers must be
importable, arguments picklable, and launchers safe under main-module import;
see [Python's start-method guidance](https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods).

Set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1` before
Python starts, as the existing batch script does. Validate the worker count
against task CPU affinity/allocation and memory; do not default to every host
CPU. Cap active processes to selected members. The parent also needs memory
and some CPU time for assembly.

The existing `xargs -P3` launches three full analyses. With `N` workers each,
that becomes about `3N` compute workers, plus parents. Prefer one configuration
at a time when using all task CPUs; otherwise explicitly partition CPUs and
memory among configurations. Each child inherits the same SLURM allocation
variable, so blindly giving all three that many workers oversubscribes it.

## Focused regression gates

Use small HDF5 fixtures and compare the legacy serial baseline, future
`--workers 1`, and actual spawned `--workers 2`/`4`; do not rely only on mocked
executors or workers=1/2 sharing the same new assembly bug. Existing foundations:
`tests/test_structure_function.py`, `tests/test_geometry_pair_sampling.py`,
`tests/test_elsasser.py`, and `tests/test_structure_function_local.py`.

- Compare ordered sampled identity arrays (and chunk boundaries) using a
  test-only worker capture/return mechanism: identity sets or counts alone miss
  changed summation order. The production SF JSON does not store these identities.
- Require exact numeric equality with matching NaN masks for counts, both SF
  fields, flatness, raw/normalized fits, bin edges, times, adaptive diagnostics,
  geometry populations/shortfalls, local slopes, and all MC percentile arrays.
  Also compare dictionary/schema structure, source lists, member IDs, fit notes,
  and logical cache flags. Exclude only enumerated execution fields such as
  `created_utc`, elapsed time, RSS, or PIDs if present.
- Cover same-time and spacetime `all`/`random`, explicit and automatic edges,
  configured and automatic fit ranges. Include a fixture where different MC
  members populate different end bins, to expose singleton-range errors.
- Adaptive: identical and offset geometries, shared-seed identity checks,
  target reached versus candidate-budget exhaustion, duplicates, and empty bins.
  Keep seed, candidate chunk size, and all budgets fixed across worker counts.
- Geometry: partial sampling and census, zero targets/empty bins, boundary
  roundoff, affine offsets, repeated geometry and three-key LRU eviction across
  truth/MC groups; reject non-affine geometry identically. Existing geometry
  tests already check identity streams, census, cache reuse, and MC reordering.
- Cover truth-only, observed-only, subset/reordered MC indices, one stored
  realization without a leading MC dimension, and `n_mc: 0` truth-only success
  plus observed/default failure. Force out-of-order completion and worker
  exceptions/crashes; assert no replacement of prior successful artifacts.
- Run downstream `derive_product`/local-slope processing on both assembled JSONs;
  compare adjacent/three-point slopes, validity/finite-member masks and slope
  percentiles. Round-trip through `write_local_results` and compare every HDF5
  dataset's shape, dtype, numerical values and deterministic attributes, not
  compressed file bytes. Normalize differing source filenames in the test.
  Core `structure_function` output is JSON, **not HDF5**; the separate truth-only
  convergence study's seed/budget HDF5 axes are not measurement-error MC members
  and are outside the proposed parallel runner.

## Suggested benchmark (not executed)

After regression gates pass, pin the input file, software environment, selection,
bin edges, seed, chunk size, pair budgets and fit ranges. Benchmark 1, 2, 4 and
8 workers on the same workstation or reserved single Puma node, with numerical
threads fixed at one and configuration-level concurrency disabled. Use a modest
representative fixture first, then the same selected MC workload with enough
members to occupy eight workers. A truth-only run has one task and cannot gain
member-level speedup.

Repeat each measurement at least three times, distinguish cold/warm filesystem
cache, and compare explicit-edge and automatic-edge runs. Record preparation,
loading/evaluation, assembly/fitting, serialization/plotting, and end-to-end wall
times; aggregate worker/parent peak memory, CPU utilization, I/O and any OOMs.
Keep output destinations separate. Report `T1/TN` speedup and `(T1/TN)/N`
efficiency, with the same scientific equivalence check on every result. Expect
limits from serial prepasses, compressed-file contention, adaptive per-worker
memory, unequal member runtimes and reduced geometry-plan reuse. No speedup
claim is justified before measurement.
