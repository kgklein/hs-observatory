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
| baseline | 300 | [150.0, 1200.0] | 5 | 0.4913 ± 0.0001 | 0.8689 ± 0.0002 | 1.1244 ± 0.0003 | 1.2697 ± 0.0004 | 1.3383 ± 0.0005 | 1.3675 ± 0.0006 |
| baseline | 300 | [200.0, 1000.0] | 5 | 0.4807 ± 0.0000 | 0.8504 ± 0.0000 | 1.1085 ± 0.0000 | 1.2734 ± 0.0000 | 1.3739 ± 0.0000 | 1.4376 ± 0.0000 |
| baseline | 600 | [150.0, 1200.0] | 5 | 0.5142 ± 0.0001 | 0.9398 ± 0.0005 | 1.2977 ± 0.0013 | 1.5896 ± 0.0025 | 1.8146 ± 0.0042 | 1.9870 ± 0.0061 |
| baseline | 600 | [200.0, 1000.0] | 5 | 0.5090 ± 0.0003 | 0.9376 ± 0.0011 | 1.3055 ± 0.0021 | 1.5968 ± 0.0030 | 1.7945 ± 0.0037 | 1.9133 ± 0.0045 |
| baseline | 900 | [150.0, 1200.0] | 5 | 0.4869 ± 0.0002 | 0.8843 ± 0.0009 | 1.2239 ± 0.0024 | 1.5206 ± 0.0043 | 1.7655 ± 0.0063 | 1.9589 ± 0.0083 |
| baseline | 900 | [200.0, 1000.0] | 5 | 0.4827 ± 0.0002 | 0.8839 ± 0.0009 | 1.2343 ± 0.0025 | 1.5323 ± 0.0051 | 1.7515 ± 0.0081 | 1.8908 ± 0.0113 |
| baseline | 1200 | [150.0, 1200.0] | 5 | 0.4871 ± 0.0004 | 0.8822 ± 0.0007 | 1.2229 ± 0.0015 | 1.5229 ± 0.0035 | 1.7710 ± 0.0063 | 1.9675 ± 0.0095 |
| baseline | 1200 | [200.0, 1000.0] | 5 | 0.4822 ± 0.0004 | 0.8804 ± 0.0010 | 1.2309 ± 0.0012 | 1.5298 ± 0.0019 | 1.7485 ± 0.0039 | 1.8862 ± 0.0068 |
| baseline | 1500 | [150.0, 1200.0] | 5 | 0.4791 ± 0.0002 | 0.8473 ± 0.0006 | 1.1520 ± 0.0015 | 1.4343 ± 0.0032 | 1.6886 ± 0.0059 | 1.9010 ± 0.0092 |
| baseline | 1500 | [200.0, 1000.0] | 5 | 0.4743 ± 0.0004 | 0.8450 ± 0.0012 | 1.1597 ± 0.0033 | 1.4478 ± 0.0068 | 1.6855 ± 0.0110 | 1.8518 ± 0.0152 |
| verification | 300 | [150.0, 1200.0] | 5 | 0.4913 ± 0.0000 | 0.8689 ± 0.0000 | 1.1246 ± 0.0000 | 1.2701 ± 0.0000 | 1.3391 ± 0.0000 | 1.3687 ± 0.0000 |
| verification | 300 | [200.0, 1000.0] | 5 | 0.4807 ± 0.0000 | 0.8504 ± 0.0000 | 1.1085 ± 0.0000 | 1.2734 ± 0.0000 | 1.3739 ± 0.0000 | 1.4376 ± 0.0000 |
| verification | 1500 | [150.0, 1200.0] | 5 | 0.4791 ± 0.0002 | 0.8474 ± 0.0006 | 1.1524 ± 0.0011 | 1.4350 ± 0.0018 | 1.6895 ± 0.0031 | 1.9019 ± 0.0052 |
| verification | 1500 | [200.0, 1000.0] | 5 | 0.4744 ± 0.0003 | 0.8453 ± 0.0008 | 1.1601 ± 0.0019 | 1.4478 ± 0.0034 | 1.6845 ± 0.0051 | 1.8492 ± 0.0068 |

## Duration changes (primary fits)

| Budget | Duration s | Order | Δ previous | Δ longest ± paired-seed SD |
|---|---:|---:|---:|---:|
| baseline | 300 | 3 | nan | -0.0275 ± 0.0014 |
| baseline | 300 | 4 | nan | -0.1646 ± 0.0031 |
| baseline | 300 | 6 | nan | -0.5336 ± 0.0088 |
| baseline | 600 | 3 | 0.1733 | 0.1457 ± 0.0010 |
| baseline | 600 | 4 | 0.3200 | 0.1554 ± 0.0029 |
| baseline | 600 | 6 | 0.6195 | 0.0860 ± 0.0089 |
| baseline | 900 | 3 | -0.0738 | 0.0719 ± 0.0021 |
| baseline | 900 | 4 | -0.0690 | 0.0864 ± 0.0028 |
| baseline | 900 | 6 | -0.0281 | 0.0579 ± 0.0031 |
| baseline | 1200 | 3 | -0.0010 | 0.0709 ± 0.0021 |
| baseline | 1200 | 4 | 0.0022 | 0.0886 ± 0.0051 |
| baseline | 1200 | 6 | 0.0086 | 0.0665 ± 0.0144 |
| baseline | 1500 | 3 | -0.0709 | 0.0000 ± 0.0000 |
| baseline | 1500 | 4 | -0.0886 | 0.0000 ± 0.0000 |
| baseline | 1500 | 6 | -0.0665 | 0.0000 ± 0.0000 |
| verification | 300 | 3 | nan | -0.0277 ± 0.0011 |
| verification | 300 | 4 | nan | -0.1649 ± 0.0018 |
| verification | 300 | 6 | nan | -0.5331 ± 0.0052 |
| verification | 1500 | 3 | 0.0277 | 0.0000 ± 0.0000 |
| verification | 1500 | 4 | 0.1649 | 0.0000 ± 0.0000 |
| verification | 1500 | 6 | 0.5331 | 0.0000 ± 0.0000 |

## Higher-budget verification (primary fits)

| Duration s | Order | Higher minus baseline xi ± paired-seed SD |
|---:|---:|---:|
| 300 | 3 | 0.0002 ± 0.0003 |
| 300 | 4 | 0.0004 ± 0.0004 |
| 300 | 6 | 0.0013 ± 0.0006 |
| 1500 | 3 | 0.0004 ± 0.0010 |
| 1500 | 4 | 0.0008 ± 0.0023 |
| 1500 | 6 | 0.0009 ± 0.0073 |

## Local-slope peaks

Maximum over geometric centers in [700.0, 1500.0] km, separately per seed.
Reported location ranges are bin centers; a maximum need not be an interior peak.

| Budget | Duration s | Order | Amplitude mean ± SD | Location range km |
|---|---:|---:|---:|---:|
| baseline | 300 | 3 | 1.6557 ± 0.0263 | 1462.7–1462.7 |
| baseline | 300 | 4 | 1.9472 ± 0.0748 | 1462.7–1462.7 |
| baseline | 300 | 6 | 2.7534 ± 0.0000 | 721.4–721.4 |
| baseline | 600 | 3 | 2.2788 ± 0.0267 | 721.4–721.4 |
| baseline | 600 | 4 | 3.3698 ± 0.0438 | 721.4–721.4 |
| baseline | 600 | 6 | 5.1822 ± 0.0735 | 721.4–721.4 |
| baseline | 900 | 3 | 2.0745 ± 0.0248 | 721.4–721.4 |
| baseline | 900 | 4 | 3.1881 ± 0.0500 | 721.4–721.4 |
| baseline | 900 | 6 | 5.1286 ± 0.1040 | 721.4–721.4 |
| baseline | 1200 | 3 | 2.0675 ± 0.0273 | 721.4–721.4 |
| baseline | 1200 | 4 | 3.1786 ± 0.0512 | 721.4–721.4 |
| baseline | 1200 | 6 | 5.1017 ± 0.0995 | 721.4–721.4 |
| baseline | 1500 | 3 | 1.9365 ± 0.0271 | 721.4–721.4 |
| baseline | 1500 | 4 | 3.0029 ± 0.0528 | 721.4–721.4 |
| baseline | 1500 | 6 | 4.9597 ± 0.1175 | 721.4–721.4 |
| verification | 300 | 3 | 1.6557 ± 0.0263 | 1462.7–1462.7 |
| verification | 300 | 4 | 1.9472 ± 0.0748 | 1462.7–1462.7 |
| verification | 300 | 6 | 2.7534 ± 0.0000 | 721.4–721.4 |
| verification | 1500 | 3 | 1.9380 ± 0.0095 | 721.4–721.4 |
| verification | 1500 | 4 | 3.0108 ± 0.0161 | 721.4–721.4 |
| verification | 1500 | 6 | 4.9957 ± 0.0463 | 721.4–721.4 |

## Spatial coverage and population

| Budget | Duration s | Mean path km | Axis traversals x/y/z | Decimated coarse revisits | Eligible / selected |
|---|---:|---:|---|---:|---:|
| baseline | 300 | 81650.2 | 0.315/0.435/0.615 | 0.532 | 100,010,660 / 2,584,530 |
| baseline | 600 | 163300.3 | 0.630/0.870/1.230 | 0.533 | 213,912,260 / 3,707,938 |
| baseline | 900 | 244950.5 | 0.945/1.305/1.845 | 0.534 | 327,813,860 / 4,608,048 |
| baseline | 1200 | 326600.7 | 1.260/1.740/2.460 | 0.536 | 441,715,460 / 5,377,656 |
| baseline | 1500 | 408250.8 | 1.575/2.175/3.075 | 0.535 | 555,617,060 / 5,920,674 |
| verification | 300 | 81650.2 | 0.315/0.435/0.615 | 0.532 | 100,010,660 / 2,764,813 |
| verification | 1500 | 408250.8 | 1.575/2.175/3.075 | 0.535 | 555,617,060 / 8,152,738 |

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
