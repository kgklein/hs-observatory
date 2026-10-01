# Developing and validating HS Observatory

Start with the [quick start](../getting-started/quick-start.md), then the
[architecture](../architecture/index.md), [data conventions](../reference/data-products.md)
and relevant scientific reference. The root
[CONTRIBUTING.md](https://github.com/kgklein/hs-observatory/blob/main/CONTRIBUTING.md)
remains authoritative for contribution policy; [DESIGN.md](https://github.com/kgklein/hs-observatory/blob/main/DESIGN.md)
remains the design-rationale document. Neither is rewritten by this migration.

## Add a field model

To add a field model, subclass `FieldModel` and implement `evaluate_quantity(quantity, time, position)`, then add construction by name alongside `field_from_config`. The inherited `evaluate(time, position)` convenience method evaluates all quantities on one shared grid. Field models contain no noise or file-I/O logic.

## Add error models and analyses

To add a measurement-error model, subclass `ErrorModel` and implement `apply(truth, rng, n_mc, time=...)`. Return both observed values and realized errors with shape `(n_mc, *truth.shape)`, then add its YAML name to `error_model_from_config`. Models that do not depend on cadence may ignore `time`. Coordinate uncertainties implement `CoordinateOffsetModel` and are constructed separately because they alter where the field is sampled. These interfaces are intended to accommodate correlated noise, dropouts, cadence averaging, spin tones, and non-Gaussian models.

To add an analysis method, subclass `AnalysisMethod`, give it a `name`, decorate it with `@register_analysis`, and ensure its module is loaded by the registry. JSON is the default output; binary analyses can override `write_result`, as `static_decorrelation` and `timeseries_fft` do for HDF5. `StructureFunctionAnalysis` implements configurable-order Elsasser structure functions, flatness, fits and multiple pair-sampling modes. Use the current implementation and focused tests, not the obsolete scaffold description, when extending it.

## P-model benchmarking and cache diagnostics

The low-level benchmark compares backends across point counts, scale depths, and intermittency values. Its default ceiling is 100 points so routine runs remain manageable; request heavier cases explicitly:

```bash
python scripts/benchmark_malara_evaluator.py
python scripts/benchmark_malara_evaluator.py --trajectory-like \
  --backends direct,cached,parallel --chunk-size 100 --n-workers 4
python scripts/benchmark_malara_evaluator.py --max-points 1000 \
  --csv outputs/malara_benchmark.csv
```

By default, positions are independent uniform samples. `--trajectory-like` instead follows a smooth path through the periodic domain with a compact nine-spacecraft formation (hub plus eight nodes), ordered by time and then spacecraft. It is a reproducible locality surrogate for benchmarking—not a specific HelioSwarm CDF trajectory—and is useful for comparing eddy reuse against the random-point case.

`evaluate_vector_chunked(positions, chunk_size=...)` uses the serial reference path. `evaluate_vector_cached(positions)` uses the evaluator-local lazy cache; `clear_cache()` and `cache_info()` expose its lifecycle and hit rate. `evaluate_vector_parallel(...)` preserves chunk ordering and reconstructs a compact evaluator in each process.

`collect_touched_eddy_statistics(positions)` reports total eddy touches, unique `(scale, i, j, k)` identities, per-scale counts, and their reuse factor. A large reuse factor indicates that caching deterministic eddy parameters may avoid substantial repeated reconstruction. The benchmark prints these diagnostics for its largest requested case. Pass `--chunk-sizes 100,1000,10000` to compare several chunk sizes for chunked or parallel backends.

Every optimized benchmark row reports `max_abs_diff_from_direct`. Zero means bit-for-bit agreement for that run; a small nonzero value would quantify floating-point disagreement and should be investigated before adopting that backend. Cache entry and hit-rate columns apply to the in-process cached backend. Parallel cache statistics are shown as unavailable because worker-local dictionaries are deliberately not transferred back to the parent process.

## Development

```bash
pytest
ruff check src tests
```

Tests cover field shapes, seeded error reproducibility, OU stationary statistics and vector geometry, coordinate-offset sampling, HDF5 round trips, synthetic CDF fallback, visualization, generation, diagnostics, static spatial decorrelation, and single-spacecraft FFT spectra.
The commands above preserve the original general developer guidance, not a
claim that those checks were run for this migration. Follow repository task
instructions and use focused checks for a scoped change. The existing CI installs
`.[dev]`, but shock-timing fitting tests require SciPy from `.[analysis]`; this
preexisting dependency mismatch needs a separate packaging/CI decision.

## Scientific and documentation review

***ADD DETAILS ON CURRENT PROJECTS***

Public package-root API
stability is described in the [Python quick start](../getting-started/first-analysis.md#use-python-instead-of-the-cli).
See [documentation development](documentation.md) for build and diagram checks.
