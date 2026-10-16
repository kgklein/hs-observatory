# Alternate propagation direction: 500a and 502a

Use `scripts/structure_function_duration_a.sh` to run the existing truth-only
observation-duration convergence workflow for the alternate-angle cases. It runs
both sequences sequentially by default, or a selected case. No statistical
implementation, field generator, error model or original study configuration was
changed.

## Inputs and preflight

All five 300/600/900/1200/1500-second HDF5 observations are present for each case:

| Case | p | Advection velocity km/s | Generation recipes |
|---|---:|---|---|
| 500a | 0.9 | [105,145,205] | `generation/malara-ts-500a-t*.yaml` |
| 502a | 0.5 | [105,145,205] | `generation/malara-ts-502a-t*.yaml` |

The missing local generation YAMLs were restored verbatim from the corresponding
HDF5 files' embedded configuration. Inputs were neither regenerated nor modified.
Both sequences passed native-cadence/duration, physical-field consistency,
common-velocity affine-trajectory and nested-prefix checks. Full comparisons and
tolerances are in each case's `validation.json`.

The advection vectors are used as stored: no velocity normalization or rotation
of observations is performed. Pair populations are recalculated from each new
geometry; old-angle populations or pair plans are not reused.

Only preflight has been run on these observations. No production SF moments,
convergence fits or figures have been calculated for the new cases yet.

## Run commands

```bash
# Validate both sequences without sampling:
bash scripts/structure_function_duration_a.sh --dry-run

# Both complete baseline studies, sequentially:
bash scripts/structure_function_duration_a.sh

# Include higher-budget endpoint verification:
bash scripts/structure_function_duration_a.sh --verify-budget

# One complete sequence:
bash scripts/structure_function_duration_a.sh 500a --verify-budget
bash scripts/structure_function_duration_a.sh 502a --verify-budget

# Refit and regenerate figures from saved moments only:
bash scripts/structure_function_duration_a.sh --plot-only

# Change the budget for both cases:
bash scripts/structure_function_duration_a.sh --fit-pairs 1000000
```

Study options are forwarded to the existing runner, including `--durations`,
`--seed-count`, `--seeds`, `--mode`, `--primary-fit`, `--sensitivity-fit` and
`--force`. A failed case stops the wrapper before the next case starts. Use
`--help` for the full study options. The wrapper manages config/input/output
paths per case and rejects overrides that could mix both sequences in one cache.

For an isolated bounded smoke test with a custom output directory, use the
underlying single-case runner:

```bash
bash scripts/structure_function_duration.sh \
  --config analysis/malara-ts-500a_sf-duration.yaml \
  --durations 300 --seed-count 1 --fit-pairs 1000 --outside-pairs 1000 \
  --output-dir outputs/structure_function_duration_smoke_500a
```

## Configurations and results

New recipes are `analysis/malara-ts-{500a,502a}_sf-duration.yaml`. Each retains
40 uniform logarithmic bins over 70–20000 km; primary 150–1200 km and sensitivity
200–1000 km fits; orders 1–6; and seeds 12345,23456,34567,45678,56789. The existing
geometry-stratified all-spacetime estimator, Elsasser definitions, geometric bin
centers, saved-moment fits, three-point local slopes and plotting code are reused.

The current baseline is **500000 pairs per fitting bin**, with 10000 outside.
Optional verification uses **1000000** per fitting bin at 300 and 1500 seconds,
with 10000 outside. Every target is capped at the exact eligible population.
These values are controlled by `duration_study.fit_pairs`, `outside_pairs`, and
`verification` in each recipe; the dedicated driver constructs the actual
per-bin targets from these settings.

Each case has separate caches, manifests, provenance, seed results and PNG/PDF
figures:

- `outputs/structure_function_duration_convergence_500a/`
- `outputs/structure_function_duration_convergence_502a/`

After a full run or saved-moment derivation, the reports and CSV tables are:

- [structure_function_duration_convergence_500a.md](structure_function_duration_convergence_500a.md) and `.csv`
- [structure_function_duration_convergence_502a.md](structure_function_duration_convergence_502a.md) and `.csv`

Resume and plot-only behavior match the original duration workflow. Existing
500/502 outputs and publication figures are untouched. Pair-seed variability
remains distinct from finite-duration, independent-field and measurement-error
uncertainty; local-slope maxima alone do not establish physical scales.

Validation performed: both real-data preflights, shell syntax, and six isolated
wrapper checks covering default ordering, case selection, cached-option
forwarding, budget forwarding, failure propagation and conflicting-path rejection.
No production SF analysis was launched.
