# Fixed-realization direct-3D convergence campaign

The runner `scripts/malara_direct3d_convergence.sh` calls the existing
`.venv/bin/hsobs-analyze-spatial-structure-function` sequentially. The Python
helper `scripts/malara_direct3d_convergence.py` handles orchestration, completion
validation, and a separate `report` subcommand. No evaluator, sampler, moment
accumulator, or fit implementation is duplicated. See the
[field-validation audit](malara_field_validation_audit.md), sections 2, 4 and 8.

## Commands

From the repository root:

```bash
# 1. Dry-run/preflight: validates inputs, prints exact centers, jobs, CLI commands,
#    and the remaining endpoint-evaluation budget. No sampling or output writes.
bash scripts/malara_direct3d_convergence.sh --dry-run

# 2. Full campaign: six sequential CLI jobs, each containing p=0.9 and p=0.5.
bash scripts/malara_direct3d_convergence.sh

# 3. Resume after interruption/failure: skip only validated completed jobs.
bash scripts/malara_direct3d_convergence.sh --resume

# 500k extension: one p case, one seed; existing 5k/50k files stay in place.
# Preflight only:
bash scripts/malara_direct3d_convergence.sh \
  --budget 500000 --p 0.9 --seed 12345 --resume --dry-run

# Execute the 500k calculation (or restart/skip it safely with the same command):
bash scripts/malara_direct3d_convergence.sh \
  --budget 500000 --p 0.9 --seed 12345 --resume

# Optional selection; --budget and --seed can each be repeated.
bash scripts/malara_direct3d_convergence.sh \
  --budget 5000 --seed 12345 --resume
bash scripts/malara_direct3d_convergence.sh --dry-run --resume

# 4. Report-only regeneration: saved products, no field evaluation.
PYTHONPATH=src MPLCONFIGDIR=/tmp/malara-direct3d-mpl \
  .venv/bin/python scripts/malara_direct3d_convergence.py report

# Optional screening threshold (fraction; default 0.10 = 10%).
PYTHONPATH=src MPLCONFIGDIR=/tmp/malara-direct3d-mpl \
  .venv/bin/python scripts/malara_direct3d_convergence.py report --threshold 0.15
```

The runner requires the project `.venv`; it does not fall back to an unrelated
system Python. `--help` lists wrapper options. Both subcommands accept `--output`
for an isolated campaign directory; the default is
`outputs/malara_direct3d_convergence/`. Report-only must use the same directory.
Sampling and reporting are deliberately separate: after a full or partial run,
invoke the reporting command explicitly. Missing/invalid jobs appear in the report.

## Exact scientific configuration

The generation configuration is `generation/malara-ts-500a-t1500.yaml`. The
original campaign passes both cases using the existing CLI's `--p 0.9 0.5`.
The 500k extension passes only `--p 0.9`. All other physical
parameters are inherited unchanged, including field seed 12345, realization 1,
cascade levels, normalization, and output-field definitions. The calculation
measures raw vector increments, not the output velocity/B amplitudes or processed
Elsasser variables.

The seven centers are read directly from `outputs/direct-spatial-sf-5000.h5`,
selecting its existing values in the inclusive interval 200–1000 km. Their
float64 values are serialized with round-trip `repr` for CLI arguments, and also
retained as hexadecimal floats in `campaign.json`. No rounded center table or
reconstructed logarithmic grid drives sampling. All pilot p cases must share
exactly the same centers; the physical field mappings must match apart from p.

Without explicit options the runner retains the original budgets 5,000 and
50,000 and sampling seeds 12345, 23456, 34567. These jobs contain both p cases
and all orders 1–6, with eight
batches, four existing quarter-prefix checkpoints, and primary fit 200–1000 km.
Each budget/seed has a unique prefix such as
`direct3d_p09-p05_N50000_seed23456`. Per-job field endpoint evaluations are
`2 endpoints × 2 p values × 7 separations × budget`: 140,000 at 5k and
1,400,000 at 50k. The complete campaign totals **4,620,000 endpoint evaluations**.
This counts field-point evaluations, not individual eddy contributions.
Completed jobs skipped by `--resume` are excluded from the printed remaining total.

`--budget 500000` opts into the extension. At that budget the runner restricts
p to 0.9 and defaults to seed 12345; the explicit command above is recommended.
It does not add any 500k work to a default invocation. The new prefix is
`direct3d_p09_N500000_seed12345`. This single job evaluates **7,000,000 field
endpoints**, keeps eight batches of 62,500 pairs/separation, and retains
125k/250k/375k/500k checkpoints, all six fitted exponents, and full/batch local
slopes. Additional supported sampling seeds can be selected explicitly later.
The original pilot HDF5 still supplies all seven bit-exact centers.

A runtime YAML copied from `configs/example_spatial_structure_function.yaml`
uses the existing schema to disable per-job plotting and unrelated older HS
comparisons, and limits fitting windows to the primary candidate interval.
The wrapper passes existing CLI options for field configuration, p values,
centers, budget, batches, seed, fit range and output prefix. Backend/worker/cache
settings continue to follow the existing recipe and generation configuration.
No scientific implementation files or original generation/analysis recipes are
edited. The 500k CLI call explicitly selects `--backend parallel` and the frozen
recipe's chunk size (128 pairs by default), retaining the field's worker setting
(four for the target recipe). The existing evaluator bounds submitted tasks and
per-task caches. Full 500k coordinate or field arrays are not allocated by this
wrapper. A larger budget necessarily reevaluates its prefix: the existing CLI
cannot extend a saved moment-only product while preserving all new batch and
checkpoint diagnostics. Completed smaller budgets are reused, not rerun.

## Completion, failures, and provenance

- `campaign.json` freezes input hashes, exact centers, effective physical field
  configurations, processing recipe, NumPy version, wrapper and scientific source
  hashes, and the six original expected jobs. Requested extensions are appended;
  selecting a subset does not erase other jobs. The original manifest inputs,
  fingerprint, legacy job dictionaries and receipts are preserved on extension.
  Scientific input/configuration/NumPy/source changes still require a separate
  output directory. A wrapper/report update alone is compatible: the original
  `wrapper_sha256` remains a creation record, and new receipts separately record
  `executed_wrapper_sha256`. Old successful products are never invalidated merely
  by adding a budget or updating reporting code.
- Each invocation writes to a unique attempt directory under `.incomplete/`.
  It retains the runtime recipe, full CLI command, `run.log`, and `status.json`.
  Completed jobs record monotonic elapsed wall time in seconds. Failures record
  elapsed time/error and exit nonzero while preserving other successful products.
- A final HDF5 and atomic `.complete.json` receipt are published only after CLI
  success and validation. Validation covers schema, raw/exact/non-smoke metadata,
  p cases, field parameters, RNG version/seed, bit-exact centers, orders, total
  counts, batch/checkpoint shapes and counts, moment accounting, analytic theory,
  and saved fitted exponents checked using the shared fitting routine. Input
  fingerprints are checked again after sampling. The HDF5 sampling-product path
  is updated to its final location, retaining the attempt path as provenance.
- Resume requires the receipt's campaign/job fingerprint, elapsed time, product
  hash, generation YAML identity and numerical validation all to agree. An HDF5
  file alone, a truncated product, or a receipt alone is insufficient. Incomplete
  jobs restart from the beginning; the underlying CLI cannot resume sample
  prefixes. Replaced incomplete files are retained inside the new attempt folder.
- Existing valid final files require `--resume`; they are not overwritten by a
  normal rerun. An output lock prevents two runners from writing concurrently.
  Existing 2k/5k pilot files are read-only inputs and never output targets.
  A request for p=0.9 alone at a smaller budget reuses an already completed
  combined-p product at the same seed. Reporting counts each p/budget/seed once,
  so overlapping products cannot inflate the number of sampling repeats.

Report-only reads the frozen manifest, receipts and small saved numerical
products. It does not need the original generation YAML or pilot HDF5 to remain
available, and does not invoke the CLI/evaluator. It validates hashes and saved
numerical contents and can summarize a partially completed campaign.

## Report and interpretation

When reporting an extended campaign, pre-existing Markdown/PNG/PDF reports are
copied once to `reports_before_500k/` before the standard output filenames are
updated. Original 5k/50k HDF5 products and completion receipts remain unchanged.

Reporting writes:

- `convergence_report.md`: job status and wall time; individual S2/S4/S6 at every
  separation; seed mean, sample SD (ddof=1) and min–max; fitted ξ1–ξ6 and analytic
  ζn; paired relative changes between requested budgets; maximum nonfinal-checkpoint variation
  for S4/S6 (also S2 in aggregate tables); and explicit scale/order screening flags.
- `convergence_exponents.png` and `.pdf`: one compact figure with a panel for
  each p, all six orders versus available budgets (5k/50k/500k for the extended
  p=0.9 case), individual seed points, seed means/SD, and theory. Single-seed
  points have no SD bar and are explicitly labeled as having no uncertainty
  estimate. The unrequested p=0.5 500k case is not shown as a missing observation.

Budget changes are calculated for matched sampling seeds before aggregation:
`S_high/S_low - 1` and `xi_high/xi_low - 1`. For three budgets the report includes
5k→50k, 50k→500k and 5k→500k, including a per-separation table for S2/S4/S6.
The 500k comparisons initially match **seed 12345 only**, rather than comparing
one 500k value to the three-seed smaller-budget mean. Such differences are
reported as single-seed values with SD unavailable, not as uncertainty estimates.
The report marks missing jobs explicitly. Checkpoint variation is
`max(abs(S_n(prefix)/S_n(final) - 1))` over 25%, 50% and 75% prefixes, with its
maximum across available seeds shown in the aggregate table. Full checkpoints
remain in each HDF5. Individual-job checkpoint maxima are listed separately.

The default screening threshold is 10%: a scale/order is flagged when its
checkpoint deviation, between-seed CV, or maximum absolute paired budget change
exceeds that value. Flags name the affected budget/criterion. A 5k flag alone is
not evidence that a larger-budget estimate also fails that screen. These are transparent
heuristics for visible sensitivity, not formal significance tests or a proof of
convergence when no flag appears. Missing jobs are insufficient evidence.

Seeds sample pairs **within one fixed field realization**. Their spread is not
measurement-error uncertainty or uncertainty across independent Malara fields.
Larger budgets replay deterministic coordinate prefixes, so paired comparisons
are correlated. A stable fit near analytic theory cannot replace stable moments,
checkpoint behavior and seed comparisons.

## Lightweight implementation validation

The focused tests fabricate small numerical HDF5 products and replace the CLI
subprocess with a stub; they never evaluate Malara fields. Extension tests cover
legacy wrapper-hash compatibility without altering old receipts/products,
single-p 500k planning, parallel/chunk options, exact centers, checkpoint/local
slope retention, failed 500k restart, completed 500k skip, smaller-budget reuse,
three-budget aggregation, and single-seed reporting/plotting. They cover exact-center
round trips, discovery, job names, dry-run budgets, receipts/hashes, malformed
products, failure recovery, configuration changes, report aggregation, missing
jobs and PNG/PDF generation from saved synthetic products.

```bash
MPLCONFIGDIR=/tmp/malara-direct3d-test-mpl .venv/bin/python -m pytest -q \
  tests/test_malara_direct3d_convergence.py
.venv/bin/ruff check scripts/malara_direct3d_convergence.py \
  tests/test_malara_direct3d_convergence.py
bash -n scripts/malara_direct3d_convergence.sh
```

The original implementation was checked with synthetic tests and a real-input
`--dry-run`. Validation of the 500k extension runs only focused synthetic tests,
targeted lint and shell syntax checks. **No production 500k calculation was
launched during implementation.** Reporting remains separate: after running the
500k command, use the `report` command above to regenerate the Markdown and figures
from saved HDF5 products without repeating field evaluation.
