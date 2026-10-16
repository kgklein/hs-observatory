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
| 600 | 0.5 | 12345 / 1 | 2–14 | True | 0 |
| 900 | 0.5 | 12345 / 1 | 2–14 | True | 0 |
| 1200 | 0.5 | 12345 / 1 | 2–14 | True | 0 |
| 1500 | 0.5 | 12345 / 1 | 2–14 | True | 0 |

**This sequence has p=0.5. It cannot directly settle the stronger
p=0.9 anomaly; a matched intermittent duration sequence would be needed.**

## Fitted exponents

| Budget | Duration s | Window km | Seeds | xi1 mean ± SD | xi2 mean ± SD | xi3 mean ± SD | xi4 mean ± SD | xi5 mean ± SD | xi6 mean ± SD |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 300 | [150.0, 1200.0] | 5 | 0.3149 ± 0.0000 | 0.6353 ± 0.0001 | 0.9594 ± 0.0002 | 1.2847 ± 0.0002 | 1.6082 ± 0.0003 | 1.9271 ± 0.0004 |
| baseline | 300 | [200.0, 1000.0] | 5 | 0.3036 ± 0.0000 | 0.6142 ± 0.0000 | 0.9296 ± 0.0000 | 1.2467 ± 0.0000 | 1.5617 ± 0.0000 | 1.8705 ± 0.0000 |
| baseline | 600 | [150.0, 1200.0] | 5 | 0.3473 ± 0.0001 | 0.6953 ± 0.0003 | 1.0430 ± 0.0004 | 1.3896 ± 0.0007 | 1.7338 ± 0.0010 | 2.0739 ± 0.0014 |
| baseline | 600 | [200.0, 1000.0] | 5 | 0.3426 ± 0.0001 | 0.6867 ± 0.0002 | 1.0310 ± 0.0003 | 1.3743 ± 0.0004 | 1.7147 ± 0.0005 | 2.0491 ± 0.0006 |
| baseline | 900 | [150.0, 1200.0] | 5 | 0.3371 ± 0.0001 | 0.6762 ± 0.0003 | 1.0156 ± 0.0004 | 1.3532 ± 0.0005 | 1.6867 ± 0.0006 | 2.0131 ± 0.0007 |
| baseline | 900 | [200.0, 1000.0] | 5 | 0.3321 ± 0.0003 | 0.6670 ± 0.0005 | 1.0025 ± 0.0008 | 1.3360 ± 0.0010 | 1.6645 ± 0.0013 | 1.9844 ± 0.0016 |
| baseline | 1200 | [150.0, 1200.0] | 5 | 0.3372 ± 0.0001 | 0.6772 ± 0.0001 | 1.0181 ± 0.0002 | 1.3578 ± 0.0003 | 1.6941 ± 0.0004 | 2.0244 ± 0.0006 |
| baseline | 1200 | [200.0, 1000.0] | 5 | 0.3319 ± 0.0002 | 0.6674 ± 0.0004 | 1.0042 ± 0.0007 | 1.3397 ± 0.0010 | 1.6710 ± 0.0013 | 1.9948 ± 0.0016 |
| baseline | 1500 | [150.0, 1200.0] | 5 | 0.3349 ± 0.0002 | 0.6718 ± 0.0003 | 1.0092 ± 0.0005 | 1.3450 ± 0.0006 | 1.6773 ± 0.0008 | 2.0035 ± 0.0009 |
| baseline | 1500 | [200.0, 1000.0] | 5 | 0.3293 ± 0.0001 | 0.6618 ± 0.0003 | 0.9954 ± 0.0005 | 1.3280 ± 0.0008 | 1.6568 ± 0.0012 | 1.9791 ± 0.0016 |
| verification | 300 | [150.0, 1200.0] | 5 | 0.3149 ± 0.0000 | 0.6354 ± 0.0000 | 0.9595 ± 0.0000 | 1.2848 ± 0.0000 | 1.6084 ± 0.0000 | 1.9275 ± 0.0000 |
| verification | 300 | [200.0, 1000.0] | 5 | 0.3036 ± 0.0000 | 0.6142 ± 0.0000 | 0.9296 ± 0.0000 | 1.2467 ± 0.0000 | 1.5617 ± 0.0000 | 1.8705 ± 0.0000 |
| verification | 1500 | [150.0, 1200.0] | 5 | 0.3349 ± 0.0001 | 0.6718 ± 0.0002 | 1.0090 ± 0.0003 | 1.3448 ± 0.0003 | 1.6769 ± 0.0004 | 2.0030 ± 0.0004 |
| verification | 1500 | [200.0, 1000.0] | 5 | 0.3294 ± 0.0002 | 0.6618 ± 0.0003 | 0.9955 ± 0.0005 | 1.3281 ± 0.0008 | 1.6570 ± 0.0011 | 1.9793 ± 0.0014 |

## Duration changes (primary fits)

| Budget | Duration s | Order | Δ previous | Δ longest ± paired-seed SD |
|---|---:|---:|---:|---:|
| baseline | 300 | 3 | nan | -0.0498 ± 0.0005 |
| baseline | 300 | 4 | nan | -0.0604 ± 0.0007 |
| baseline | 300 | 6 | nan | -0.0763 ± 0.0010 |
| baseline | 600 | 3 | 0.0836 | 0.0338 ± 0.0008 |
| baseline | 600 | 4 | 0.1049 | 0.0445 ± 0.0011 |
| baseline | 600 | 6 | 0.1468 | 0.0704 ± 0.0017 |
| baseline | 900 | 3 | -0.0274 | 0.0064 ± 0.0008 |
| baseline | 900 | 4 | -0.0364 | 0.0082 ± 0.0010 |
| baseline | 900 | 6 | -0.0608 | 0.0096 ± 0.0012 |
| baseline | 1200 | 3 | 0.0025 | 0.0089 ± 0.0004 |
| baseline | 1200 | 4 | 0.0046 | 0.0128 ± 0.0005 |
| baseline | 1200 | 6 | 0.0113 | 0.0209 ± 0.0012 |
| baseline | 1500 | 3 | -0.0089 | 0.0000 ± 0.0000 |
| baseline | 1500 | 4 | -0.0128 | 0.0000 ± 0.0000 |
| baseline | 1500 | 6 | -0.0209 | 0.0000 ± 0.0000 |
| verification | 300 | 3 | nan | -0.0495 ± 0.0003 |
| verification | 300 | 4 | nan | -0.0600 ± 0.0003 |
| verification | 300 | 6 | nan | -0.0755 ± 0.0004 |
| verification | 1500 | 3 | 0.0495 | 0.0000 ± 0.0000 |
| verification | 1500 | 4 | 0.0600 | 0.0000 ± 0.0000 |
| verification | 1500 | 6 | 0.0755 | 0.0000 ± 0.0000 |

## Higher-budget verification (primary fits)

| Duration s | Order | Higher minus baseline xi ± paired-seed SD |
|---:|---:|---:|
| 300 | 3 | 0.0001 ± 0.0002 |
| 300 | 4 | 0.0001 ± 0.0002 |
| 300 | 6 | 0.0003 ± 0.0004 |
| 1500 | 3 | -0.0001 ± 0.0002 |
| 1500 | 4 | -0.0002 ± 0.0004 |
| 1500 | 6 | -0.0005 ± 0.0006 |

## Local-slope peaks

Maximum over geometric centers in [700.0, 1500.0] km, separately per seed.
Reported location ranges are bin centers; a maximum need not be an interior peak.

| Budget | Duration s | Order | Amplitude mean ± SD | Location range km |
|---|---:|---:|---:|---:|
| baseline | 300 | 3 | 1.2930 ± 0.0559 | 1269.9–1462.7 |
| baseline | 300 | 4 | 1.6678 ± 0.0576 | 1269.9–1462.7 |
| baseline | 300 | 6 | 2.4350 ± 0.0155 | 957.1–1462.7 |
| baseline | 600 | 3 | 1.1486 ± 0.0715 | 721.4–1462.7 |
| baseline | 600 | 4 | 1.5140 ± 0.0837 | 721.4–1462.7 |
| baseline | 600 | 6 | 2.3201 ± 0.0670 | 721.4–1462.7 |
| baseline | 900 | 3 | 1.0521 ± 0.0589 | 957.1–1462.7 |
| baseline | 900 | 4 | 1.3884 ± 0.0501 | 721.4–1462.7 |
| baseline | 900 | 6 | 2.1378 ± 0.0049 | 721.4–721.4 |
| baseline | 1200 | 3 | 1.0588 ± 0.0587 | 957.1–1462.7 |
| baseline | 1200 | 4 | 1.3920 ± 0.0863 | 957.1–1462.7 |
| baseline | 1200 | 6 | 2.1099 ± 0.1135 | 721.4–1462.7 |
| baseline | 1500 | 3 | 1.1353 ± 0.0361 | 1269.9–1462.7 |
| baseline | 1500 | 4 | 1.4981 ± 0.0627 | 1269.9–1462.7 |
| baseline | 1500 | 6 | 2.2050 ± 0.1624 | 721.4–1462.7 |
| verification | 300 | 3 | 1.2935 ± 0.0553 | 1269.9–1462.7 |
| verification | 300 | 4 | 1.6684 ± 0.0569 | 1269.9–1462.7 |
| verification | 300 | 6 | 2.4400 ± 0.0102 | 957.1–1462.7 |
| verification | 1500 | 3 | 1.1362 ± 0.0352 | 1269.9–1462.7 |
| verification | 1500 | 4 | 1.4994 ± 0.0619 | 1269.9–1462.7 |
| verification | 1500 | 6 | 2.2069 ± 0.1622 | 721.4–1462.7 |

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
