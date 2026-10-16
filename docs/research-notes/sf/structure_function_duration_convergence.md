# Observation-duration structure-function convergence

**Generated from saved truth moments; no measurement-error MC statistics.**

Nested input sequence within tolerance: **True**.
Inputs, full generation YAML, hashes, exact-prefix comparisons and tolerances are
in validation.json and manifest.json. Sampling settings and exact bin edges are
stored with each seed. Geometric centers define the main fits; population
mean-log-center fits are separate diagnostics.

All uncertainties below are sample SD across deterministic pair seeds, conditional
on one field and duration. Seed ranges and individual values remain in derived.json.
The longest selected duration is a reference, not exact truth. Fewer than five
seeds or a single selected duration is a pilot, not a completed convergence study.

## Input checks

| Duration s | p | Field seed / realization | Levels (inclusive) | Exact prefix | Max truth difference |
|---:|---:|---|---|---|---:|
| 300 | 0.5 | 12345 / 1 | 2–14 | True | 0 |
| 600 | 0.5 | 12345 / 1 | 2–14 | False | 5.68e-14 |
| 900 | 0.5 | 12345 / 1 | 2–14 | False | 5.68e-14 |
| 1200 | 0.5 | 12345 / 1 | 2–14 | True | 0 |
| 1500 | 0.5 | 12345 / 1 | 2–14 | True | 0 |

**This sequence has p=0.5. It cannot directly settle the stronger
p=0.9 anomaly; a matched intermittent duration sequence would be needed.**

## Fitted exponents

| Budget | Duration s | Window km | Seeds | xi1 mean ± SD | xi2 mean ± SD | xi3 mean ± SD | xi4 mean ± SD | xi5 mean ± SD | xi6 mean ± SD |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 300 | [150.0, 1200.0] | 5 | 0.3397 ± 0.0001 | 0.6804 ± 0.0002 | 1.0216 ± 0.0004 | 1.3624 ± 0.0006 | 1.7014 ± 0.0009 | 2.0377 ± 0.0014 |
| baseline | 300 | [200.0, 1000.0] | 5 | 0.3371 ± 0.0000 | 0.6783 ± 0.0000 | 1.0225 ± 0.0001 | 1.3675 ± 0.0001 | 1.7102 ± 0.0001 | 2.0475 ± 0.0002 |
| baseline | 600 | [150.0, 1200.0] | 5 | 0.3388 ± 0.0001 | 0.6756 ± 0.0002 | 1.0101 ± 0.0004 | 1.3419 ± 0.0006 | 1.6709 ± 0.0009 | 1.9966 ± 0.0014 |
| baseline | 600 | [200.0, 1000.0] | 5 | 0.3358 ± 0.0001 | 0.6701 ± 0.0002 | 1.0013 ± 0.0005 | 1.3274 ± 0.0007 | 1.6459 ± 0.0011 | 1.9541 ± 0.0015 |
| baseline | 900 | [150.0, 1200.0] | 5 | 0.3356 ± 0.0000 | 0.6707 ± 0.0001 | 1.0049 ± 0.0003 | 1.3374 ± 0.0007 | 1.6677 ± 0.0011 | 1.9952 ± 0.0016 |
| baseline | 900 | [200.0, 1000.0] | 5 | 0.3341 ± 0.0002 | 0.6684 ± 0.0003 | 1.0017 ± 0.0004 | 1.3319 ± 0.0006 | 1.6569 ± 0.0010 | 1.9742 ± 0.0015 |
| baseline | 1200 | [150.0, 1200.0] | 5 | 0.3351 ± 0.0001 | 0.6693 ± 0.0002 | 1.0026 ± 0.0003 | 1.3347 ± 0.0004 | 1.6650 ± 0.0006 | 1.9931 ± 0.0009 |
| baseline | 1200 | [200.0, 1000.0] | 5 | 0.3336 ± 0.0003 | 0.6671 ± 0.0005 | 0.9998 ± 0.0007 | 1.3306 ± 0.0009 | 1.6576 ± 0.0013 | 1.9783 ± 0.0018 |
| baseline | 1500 | [150.0, 1200.0] | 5 | 0.3344 ± 0.0002 | 0.6675 ± 0.0003 | 0.9995 ± 0.0004 | 1.3305 ± 0.0005 | 1.6605 ± 0.0005 | 1.9899 ± 0.0007 |
| baseline | 1500 | [200.0, 1000.0] | 5 | 0.3327 ± 0.0002 | 0.6643 ± 0.0003 | 0.9945 ± 0.0003 | 1.3222 ± 0.0003 | 1.6461 ± 0.0005 | 1.9649 ± 0.0011 |

## Duration changes (primary fits)

| Budget | Duration s | Order | Δ previous | Δ longest ± paired-seed SD |
|---|---:|---:|---:|---:|
| baseline | 300 | 3 | nan | 0.0221 ± 0.0007 |
| baseline | 300 | 4 | nan | 0.0319 ± 0.0009 |
| baseline | 300 | 6 | nan | 0.0477 ± 0.0019 |
| baseline | 600 | 3 | -0.0115 | 0.0106 ± 0.0005 |
| baseline | 600 | 4 | -0.0204 | 0.0115 ± 0.0009 |
| baseline | 600 | 6 | -0.0410 | 0.0067 ± 0.0020 |
| baseline | 900 | 3 | -0.0052 | 0.0054 ± 0.0004 |
| baseline | 900 | 4 | -0.0045 | 0.0069 ± 0.0007 |
| baseline | 900 | 6 | -0.0015 | 0.0052 ± 0.0018 |
| baseline | 1200 | 3 | -0.0023 | 0.0031 ± 0.0004 |
| baseline | 1200 | 4 | -0.0027 | 0.0042 ± 0.0005 |
| baseline | 1200 | 6 | -0.0021 | 0.0032 ± 0.0010 |
| baseline | 1500 | 3 | -0.0031 | 0.0000 ± 0.0000 |
| baseline | 1500 | 4 | -0.0042 | 0.0000 ± 0.0000 |
| baseline | 1500 | 6 | -0.0032 | 0.0000 ± 0.0000 |

## Local-slope peaks

Maximum over geometric centers in [700.0, 1500.0] km, separately per seed.
Reported location ranges are bin centers; a maximum need not be an interior peak.

| Budget | Duration s | Order | Amplitude mean ± SD | Location range km |
|---|---:|---:|---:|---:|
| baseline | 300 | 3 | 1.0574 ± 0.0177 | 957.1–1462.7 |
| baseline | 300 | 4 | 1.3860 ± 0.0503 | 957.1–1462.7 |
| baseline | 300 | 6 | 2.0731 ± 0.1144 | 957.1–1462.7 |
| baseline | 600 | 3 | 1.1155 ± 0.0084 | 957.1–957.1 |
| baseline | 600 | 4 | 1.4763 ± 0.0175 | 957.1–1102.5 |
| baseline | 600 | 6 | 2.2487 ± 0.0731 | 957.1–1102.5 |
| baseline | 900 | 3 | 1.0594 ± 0.0459 | 957.1–1462.7 |
| baseline | 900 | 4 | 1.4127 ± 0.0768 | 957.1–1462.7 |
| baseline | 900 | 6 | 2.1369 ± 0.1537 | 957.1–1462.7 |
| baseline | 1200 | 3 | 1.0124 ± 0.0258 | 957.1–1462.7 |
| baseline | 1200 | 4 | 1.3351 ± 0.0376 | 957.1–1462.7 |
| baseline | 1200 | 6 | 1.9993 ± 0.0403 | 957.1–1462.7 |
| baseline | 1500 | 3 | 1.0557 ± 0.0291 | 957.1–1462.7 |
| baseline | 1500 | 4 | 1.4240 ± 0.0496 | 957.1–1462.7 |
| baseline | 1500 | 6 | 2.2095 ± 0.0918 | 957.1–1462.7 |

## Spatial coverage and population

| Budget | Duration s | Mean path km | Axis traversals x/y/z | Decimated coarse revisits | Eligible / selected |
|---|---:|---:|---|---:|---:|
| baseline | 300 | 79967.9 | 0.315/0.006/0.735 | 0.536 | 101,869,766 / 3,029,014 |
| baseline | 600 | 159935.7 | 0.630/0.012/1.470 | 0.584 | 218,260,166 / 4,145,253 |
| baseline | 900 | 239903.6 | 0.945/0.018/2.205 | 0.571 | 334,650,566 / 4,899,339 |
| baseline | 1200 | 319871.5 | 1.260/0.024/2.940 | 0.578 | 451,040,966 / 5,570,841 |
| baseline | 1500 | 399839.3 | 1.575/0.030/3.675 | 0.582 | 567,431,366 / 6,040,538 |

Coverage uses unwrapped path length and sparse periodic voxels (64 cells/axis
by default). Decimation is approximately one voxel length of travel. Repeated
voxels include formation overlap and correlated samples; this is not a count
of independent eddies. Eligible spacecraft-pair/signed-lag counts, population
direction histograms (absolute cosine to flow) and all-bin counts are saved.

## Interpretation

Assess changes against both seed variability and the optional higher-budget
results. Agreement with p-model theory is a separate comparison, not a
convergence criterion. Persistent peaks do not alone identify physical scales.
Global density normalization is evaluated per duration using the unchanged
Elsasser machinery; this can matter if mean density differs between records.
No automatic physical-convergence claim is made. Independent fields, other p
values and longer independent spatial coverage remain separate tests.

## Reproduction and saved results

```bash
# Validate only (no pair sampling):
bash scripts/structure_function_duration.sh --dry-run
# Complete baseline plus optional endpoint higher-budget verification:
bash scripts/structure_function_duration.sh --verify-budget
# One-duration, one-seed bounded smoke test, separate output directory:
bash scripts/structure_function_duration.sh --durations 300 --seed-count 1 --fit-pairs 1000 --outside-pairs 1000 --output-dir outputs/structure_function_duration_smoke
# Refit and plot saved moments; no HDF5 or increment evaluation:
bash scripts/structure_function_duration.sh --plot-only
```

Defaults above use the supplied p=0.5 recipe. For another p sequence, set
`--generation-glob` to its generation YAML pattern and `--output-dir` to a
distinct directory; use that same output directory for cached commands.
The selected run configuration is embedded in manifest.json.

The dedicated output directory holds validation.json, manifest.json, atomic
per-duration/budget/seed moments/*.json files, derived.json and four PNG/PDF
figure pairs. The report CSV contains all xi means/SD/ranges and changes.
Each seed product saves both Elsasser variables, exact edges, counts,
sampler diagnostics, empirical population centers, spatial coverage, timing
and peak RSS. Input and statistical-code fingerprints control cache reuse.
Fit windows are excluded from moment-cache keys. Figures and report are
rebuilt from saved moments; the manifest describes the latest selected run.
Incomplete runs can be resumed with the same command. No automatic full
census is requested, although small bins are capped at their population.

Use `--mode sample`, `--mode derive`, and `--mode plot` for separate stages.
`--mode plot` reads derived.json; `--plot-only` first refits saved moments.
Selected seeds/durations apply to cached refitting when explicitly provided.
Error bars across duration changes use common seed labels, with differences
computed per seed; identities are not forced equal across populations.
Periodic wrapping is used only for coverage, never to replace the existing
unwrapped-separation structure-function estimator.
