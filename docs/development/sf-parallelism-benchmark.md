# SF parallelism equivalence and benchmark runner

Use `scripts/benchmark_sf_parallelism.py` from an environment with the project
Python dependencies installed, for example after `source .venv/bin/activate`.
The runner launches the existing `hsobs-analyze` CLI, never internal analysis
functions. It uses the project `.venv/bin/hsobs-analyze` when available, otherwise
`PATH`; `--cli /absolute/path/to/hsobs-analyze` selects another entrypoint.
Relative configuration and input paths resolve from the repository root.

```bash
# Inspect planned runs: no files, input hashing, or analysis subprocesses
python scripts/benchmark_sf_parallelism.py --dry-run

# Validate and benchmark (these commands launch real analyses)
python scripts/benchmark_sf_parallelism.py --workers 1 2 4 8 --repetitions 3

# A smaller worker subset; a one-worker baseline is automatically included
python scripts/benchmark_sf_parallelism.py analysis/my-sf-config.yaml \
  --workers 2 4 --repetitions 2 --output-dir outputs/sf_benchmark_subset

# Explicitly repeat completed runs in new attempt directories
python scripts/benchmark_sf_parallelism.py --workers 1 2 --rerun
```

The default YAML is `analysis/malara-prod-600_sf-short-v00.yaml`. Its current
truth-only selection has one task, so extra workers measure process overhead
rather than member-level speedup. For useful MC scaling, supply a YAML selecting
truth and observed members, for example `data_groups: [truth, observed]` and
`mc_indices: [0, 1, 2, 3, 4]`. The runner never changes the selection, seeds,
chunk sizes, budgets, bin settings, fit ranges, or plot-enabled settings.
Automatic bins are independently calculated by the unchanged CLI on identical
input/settings; their exact resulting edges are included in equivalence checks.

## Isolation, reuse, and provenance

Runs execute **sequentially**, in repetition order and then ascending worker
count. The three numerical thread variables `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` are set to `1` before every launch.
Each attempt gets its own copied YAML, JSON output, figures, log, and metrics:

```text
outputs/sf_parallel_benchmark/
    manifest.json
    benchmark_results.csv
    benchmark_report.md
    benchmark_scaling.png
    runs/workers-1/rep-1/attempt-000001/
        analysis.yaml
        analysis.log
        result.json
        diagnostic.png                 # if enabled
        publication.png                # if enabled
        ensemble_statistics.json
        execution.json
        validation.json
```

All known SF output paths, including diagnostic/publication paths supplied in
analysis options, are redirected. Production YAML/input/output paths cannot be
inside the benchmark directory. Nonempty directories without a runner manifest
are rejected; escaped symlink paths are rejected. Summary files are replaced
atomically. A lock prevents two runners from sharing the same output directory.
After an unclean kill, inspect/remove a stale `.benchmark.lock` only after
confirming its recorded process has stopped.

The manifest pins the original configuration, full input SHA-256, CLI,
relevant source files, runner, and environment provenance. Input signatures are
checked around analysis launches. A different input, code, CLI, or recorded
environment requires a new output directory; `--rerun` does not bypass this
check. Full input hashing is outside measured run times and can be expensive.
Python entrypoints with direct Python shebangs are probed for their interpreter
and package versions; other launchers record that this probe is unavailable.
Runner environment versions are recorded separately.

Completed attempts, including failed attempts, are reused unless `--rerun` is
specified. Reuse rechecks saved result/statistics hashes and scientific equality;
corruption fails validation rather than triggering an expensive silent rerun.
`--rerun` creates new numbered attempts and retains previous files. Interrupted
attempts without `execution.json` are incomplete and are not reused. This is
whole-run benchmark reuse, not member checkpointing or resume. Reports cover
the currently requested workers/repetitions and latest completed attempts;
older artifacts remain on disk. Individual failures do not erase other runs.

## Exact validation

Every successful result, including later one-worker repetitions, is compared
against the first successful one-worker result in the requested run set. Missing
baselines produce `baseline_unavailable` rather than a successful validation.
Comparison covers every JSON key, scalar type/value, list length and ordering,
NaN location, and deterministic metadata field. There is no numerical tolerance.
**Only `/metadata/created_utc` is excluded.** Geometry cache diagnostics, member
indices, fit windows, counts, flatness, and exponent products remain included.

Observed 5th/50th/95th percentiles of counts, SFs, flatness, and raw/normalized
exponents are also derived from saved member arrays outside the analysis timer,
saved, and checked exactly. Existing embedded ensemble statistics are compared
as part of the complete result. The report and stderr identify mismatches by
worker count, repetition, JSON pointer, and baseline/actual values. By default
up to 20 differences are reported per run; increase `--max-differences` for more.

Exit status is `0` when all requested runs validate, `1` for scientific mismatch,
failed analysis, corrupted output, or missing baseline, and `2` for runner/setup
errors such as manifest conflicts. Failures retain their logs and exit codes.

## Timing, memory, and Puma

Each `wall_seconds` measures the end-to-end CLI subprocess, including startup,
HDF5 reading, serial preparation, SF evaluation, fits, output writing, and enabled
plotting. Validation, input hashing, environment probing, report generation, and
scaling-figure generation are outside this timer. **There are no instrumented
phase timings.** Plotting settings remain identical; enabled figures contribute
to measured time. Cache flushing is not performed; input hashing warms the
filesystem cache, and later runs may benefit from warmed data.

On Linux, a monitoring thread samples `/proc` for the dedicated subprocess
process group, including spawned worker/helper processes. The reported memory
is the maximum sampled sum of resident bytes, not parent-only RSS or a sum of
individual processes' historical peaks. Sampling defaults to 50 ms; adjust with
`--sample-interval 0.01`. Short-lived peaks may be missed, shared pages count in
each process, and process reads are not atomic. It is not cgroup accounting or
an exact kernel-recorded peak. Where sampling is unavailable, memory is blank
with method `unavailable`; sampling errors are retained in execution metadata.

CSV speedup uses the serial time from the same repetition. Summary/plot speedup
is median valid serial time divided by median valid time at that worker count;
efficiency is speedup/workers. Only exact, successful runs contribute to timing
summaries. CSV rows retain failed-run timing, status, and memory information.

The runner respects CPU affinity and `SLURM_CPUS_PER_TASK`, using their minimum.
Worker counts exceeding this limit are warned about and skipped unless
`--allow-oversubscription` is explicitly passed. It cannot infer a safe memory
budget; choose workers and SLURM memory based on the recorded workspace costs.
Use one configuration at a time, rather than nesting this runner under the
existing `xargs -P3` configuration launcher.

For Puma, adapt partition, time, and memory requests to local policy:

```bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

srun .venv/bin/python scripts/benchmark_sf_parallelism.py \
  --workers 1 2 4 8 --repetitions 3
```

This uses a single node; there is no MPI or cross-node scheduling. For the
scientific scheduling invariants, see the
[member-parallelism audit](sf-member-parallelism-audit.md).
