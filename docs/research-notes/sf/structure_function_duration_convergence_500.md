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
| 300 | 0.9 | 12345 / 1 | 2–14 | True | 0 |
| 600 | 0.9 | 12345 / 1 | 2–14 | True | 0 |
| 900 | 0.9 | 12345 / 1 | 2–14 | True | 0 |
| 1200 | 0.9 | 12345 / 1 | 2–14 | True | 0 |
| 1500 | 0.9 | 12345 / 1 | 2–14 | True | 0 |

## Fitted exponents

| Budget | Duration s | Window km | Seeds | xi1 mean ± SD | xi2 mean ± SD | xi3 mean ± SD | xi4 mean ± SD | xi5 mean ± SD | xi6 mean ± SD |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 300 | [150.0, 1200.0] | 5 | 0.4758 ± 0.0002 | 0.8269 ± 0.0004 | 1.0890 ± 0.0007 | 1.3007 ± 0.0013 | 1.4866 ± 0.0024 | 1.6609 ± 0.0039 |
| baseline | 300 | [200.0, 1000.0] | 5 | 0.4726 ± 0.0000 | 0.8255 ± 0.0001 | 1.0872 ± 0.0001 | 1.2882 ± 0.0002 | 1.4512 ± 0.0004 | 1.5929 ± 0.0007 |
| baseline | 600 | [150.0, 1200.0] | 5 | 0.4764 ± 0.0004 | 0.8201 ± 0.0010 | 1.0812 ± 0.0019 | 1.2955 ± 0.0035 | 1.4846 ± 0.0058 | 1.6617 ± 0.0091 |
| baseline | 600 | [200.0, 1000.0] | 5 | 0.4710 ± 0.0002 | 0.8120 ± 0.0004 | 1.0693 ± 0.0009 | 1.2703 ± 0.0019 | 1.4338 ± 0.0038 | 1.5760 ± 0.0068 |
| baseline | 900 | [150.0, 1200.0] | 5 | 0.5012 ± 0.0003 | 0.9057 ± 0.0006 | 1.2492 ± 0.0013 | 1.5481 ± 0.0026 | 1.8208 ± 0.0044 | 2.0849 ± 0.0070 |
| baseline | 900 | [200.0, 1000.0] | 5 | 0.5008 ± 0.0005 | 0.9090 ± 0.0013 | 1.2540 ± 0.0029 | 1.5464 ± 0.0052 | 1.8059 ± 0.0080 | 2.0544 ± 0.0109 |
| baseline | 1200 | [150.0, 1200.0] | 5 | 0.4956 ± 0.0003 | 0.8902 ± 0.0012 | 1.2262 ± 0.0023 | 1.5230 ± 0.0034 | 1.7982 ± 0.0047 | 2.0685 ± 0.0087 |
| baseline | 1200 | [200.0, 1000.0] | 5 | 0.4956 ± 0.0003 | 0.8943 ± 0.0013 | 1.2329 ± 0.0028 | 1.5255 ± 0.0038 | 1.7905 ± 0.0047 | 2.0478 ± 0.0120 |
| baseline | 1500 | [150.0, 1200.0] | 5 | 0.4825 ± 0.0003 | 0.8688 ± 0.0009 | 1.2065 ± 0.0019 | 1.5116 ± 0.0033 | 1.7943 ± 0.0054 | 2.0683 ± 0.0080 |
| baseline | 1500 | [200.0, 1000.0] | 5 | 0.4820 ± 0.0005 | 0.8721 ± 0.0016 | 1.2139 ± 0.0039 | 1.5185 ± 0.0078 | 1.7955 ± 0.0130 | 2.0620 ± 0.0187 |

## Duration changes (primary fits)

| Budget | Duration s | Order | Δ previous | Δ longest ± paired-seed SD |
|---|---:|---:|---:|---:|
| baseline | 300 | 3 | nan | -0.1174 ± 0.0021 |
| baseline | 300 | 4 | nan | -0.2109 ± 0.0037 |
| baseline | 300 | 6 | nan | -0.4075 ± 0.0089 |
| baseline | 600 | 3 | -0.0079 | -0.1253 ± 0.0027 |
| baseline | 600 | 4 | -0.0052 | -0.2161 ± 0.0051 |
| baseline | 600 | 6 | 0.0009 | -0.4066 ± 0.0138 |
| baseline | 900 | 3 | 0.1680 | 0.0428 ± 0.0023 |
| baseline | 900 | 4 | 0.2526 | 0.0365 ± 0.0041 |
| baseline | 900 | 6 | 0.4232 | 0.0165 ± 0.0115 |
| baseline | 1200 | 3 | -0.0230 | 0.0197 ± 0.0018 |
| baseline | 1200 | 4 | -0.0251 | 0.0114 ± 0.0027 |
| baseline | 1200 | 6 | -0.0164 | 0.0001 ± 0.0079 |
| baseline | 1500 | 3 | -0.0197 | 0.0000 ± 0.0000 |
| baseline | 1500 | 4 | -0.0114 | 0.0000 ± 0.0000 |
| baseline | 1500 | 6 | -0.0001 | 0.0000 ± 0.0000 |

## Local-slope peaks

Maximum over geometric centers in [700.0, 1500.0] km, separately per seed.
Reported location ranges are bin centers; a maximum need not be an interior peak.

| Budget | Duration s | Order | Amplitude mean ± SD | Location range km |
|---|---:|---:|---:|---:|
| baseline | 300 | 3 | 1.6020 ± 0.0117 | 957.1–957.1 |
| baseline | 300 | 4 | 2.3113 ± 0.0230 | 957.1–957.1 |
| baseline | 300 | 6 | 3.9459 ± 0.0724 | 957.1–957.1 |
| baseline | 600 | 3 | 1.6382 ± 0.0283 | 957.1–957.1 |
| baseline | 600 | 4 | 2.3837 ± 0.0521 | 957.1–957.1 |
| baseline | 600 | 6 | 4.0839 ± 0.1200 | 957.1–957.1 |
| baseline | 900 | 3 | 1.5820 ± 0.0359 | 957.1–1462.7 |
| baseline | 900 | 4 | 1.9500 ± 0.1372 | 957.1–1462.7 |
| baseline | 900 | 6 | 2.3731 ± 0.7676 | 957.1–1462.7 |
| baseline | 1200 | 3 | 1.4903 ± 0.0437 | 957.1–1462.7 |
| baseline | 1200 | 4 | 1.8189 ± 0.1700 | 957.1–1462.7 |
| baseline | 1200 | 6 | 1.9982 ± 0.5734 | 957.1–1462.7 |
| baseline | 1500 | 3 | 1.5947 ± 0.1485 | 957.1–1462.7 |
| baseline | 1500 | 4 | 2.0059 ± 0.2586 | 957.1–1462.7 |
| baseline | 1500 | 6 | 2.5141 ± 0.4750 | 1102.5–1462.7 |

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
