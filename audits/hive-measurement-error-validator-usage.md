# HIVE measurement-error validator — Phase A usage

The standalone validator checks **saved latent perturbations**, independently of their effects on recovered plasma quantities. It opens HDF5 read-only and never imports production error generators or field evaluators. It uses existing project dependencies; no installation or dependency changes are needed beyond the project environment.

Implementation: `scripts/validate_measurement_errors.py`, with private helpers in `scripts/measurement_error_validation/`. Mathematical reference: [Stage 1 audit](hive-measurement-error-validation.md).

## Run later against saved products

These are campaign commands for **later execution**. They were not run during implementation. Run from the repository root with the project environment active (or replace `python` with `.venv/bin/python`).

```bash
python scripts/validate_measurement_errors.py \
  --input outputs/production/malara-prod-601.h5 \
  --config generation/malara-prod-601.yaml \
  --output-dir outputs/error_validation/601
```

```bash
for id in 611 621 631 641 651 661 671; do
  python scripts/validate_measurement_errors.py \
    --input "outputs/production/malara-prod-${id}.h5" \
    --config "generation/malara-prod-${id}.yaml" \
    --output-dir "outputs/error_validation/${id}"
done
```

The loop continues after individual nonzero exits in a normal shell without `set -e`. In a shell using `set -e`, inspect or explicitly handle each return code. Inputs are explicit: change `--input` if a product lives at the YAML's original top-level `outputs/` location. Never pool runs with reused random seeds as independent samples.

Selections and resource controls:

```bash
python scripts/validate_measurement_errors.py \
  --input outputs/production/malara-prod-601.h5 \
  --config generation/malara-prod-601.yaml \
  --mc 0,1,2,3,4 --spacecraft H,N1 \
  --confidence 0.99 --chunk-times 4096 --seed 20261021 \
  --output-dir outputs/error_validation/601_subset
```

Use **actual spacecraft IDs from `/spacecraft/id`**; `H,N1` is illustrative. MC indices are zero-based. Defaults select all MC/spacecraft, confidence 0.99, chunks of 4096 timestamps and seed 20261021. Duplicate/unknown selections are rejected. The seed affects only bounded plotting samples, never model generation. Small subsets can leave primary independent-path tests below their 30-path threshold.

HDF5 reads are bounded to one MC/spacecraft scalar series and at most `--chunk-times` timestamps per read; derived arrays and theoretical spectra occupy a few series-sized buffers. Memory grows with record length, not the full MC ensemble. Constant offsets/matrices and three time axes are small compared with the product and are retained. The code does **not** load the entire 3.35-GB product. A complete run still must read all selected latent time series and can be I/O intensive; no production runtime benchmark was performed.

## Outputs and interpretation

Each run writes/replaces its own `validation_report.md`, `validation_results.csv`, `validation_results.json` and compact aggregate PNG/PDF figures in the chosen output directory. Use a distinct directory for each selection/run. The JSON lists figures produced by the current run; older figures in a reused directory are not removed. Input HDF5/configuration paths are protected against report-file collisions.

The JSON includes embedded configuration/metadata, input size/mtime, source revision, package versions, selections and estimator settings. It does not hash the production product. CSV/JSON retain numerical targets, estimates, additive and relative effect sizes where defined, sample/path counts, marginal intervals, test methods, raw/adjusted p-values and statuses. Null quantile interval endpoints denote unbounded tails; null other numerical fields denote unavailable values.

- `PASS`: no significant disagreement at the chosen familywise level. This is **consistency**, not proof of correctness.
- `FAIL`: a primary adjusted statistical test rejects the implemented model, a numerical invariant fails, or the supplied configuration disagrees with the embedded parameters. Invalid configuration/input structure is also reported as a validity failure with exit code 2.
- `INSUFFICIENT_DATA`: too few independent paths or inadequate time/frequency coverage. This is not a scientific rejection.
- `SKIPPED`: disabled components, unavailable datasets, or explicitly informational/out-of-scope diagnostics. Missing datasets for enabled components count as missing coverage.
- `INCOMPLETE_INPUT`: missing/unreadable/truncated HDF5, failed dataset reads, or size/mtime changes while reading. Rerun only after transfer completion. Successful reading is not a full-file checksum/integrity certification.

Exit codes: **0** means required checks passed without configuration discrepancies; **1** means statistical failure, configuration discrepancy or incomplete required coverage; **2** means incomplete input or invalid invocation/configuration/structure. Embedded normalized parameters remain the expectation oracle when YAML disagrees; the mismatch still makes the overall outcome unsuccessful. Output-path relocation alone is informational.

At 0.125-second cadence, the configured **10 Hz anchor is necessarily `INSUFFICIENT_DATA`**. Consequently a statistically consistent production product with that missing required coverage can return **1**, with overall `INSUFFICIENT_DATA`, rather than `PASS`. Component/statistic rows distinguish this limit from a scientific `FAIL`. The above-10-Hz Phase C check is an informational `SKIPPED` row and does not create an additional gate.

## Statistical conventions and limits

- Position uses the configured Maxwell radial confidence contour (95.45% at 2σ, 99.73% at 3σ), with the implemented rounded 3σ factor 3.762. `radial_probability` overrides that contour. Do not divide the radius directly by sigma level to obtain axis sigma.
- Legacy density/speed requirements specify log-error sigma; normalized variance fractions allocate independent constant bias and stationary OU variance. The code subtracts **saved bias**, never each path's temporal mean. Lognormal fractional mean is positive without a centering correction.
- The legacy velocity-angle sigma applies to **each** transverse OU coordinate. The total latent angle is Rayleigh, with RMS sqrt(2) times component sigma. Alignment uses independent Gaussian B/V rotation vectors and Maxwell rotation magnitudes, not Euler angles.
- Constant errors contribute one draw per MC/spacecraft. Gaussian/Maxwell/Rayleigh CDF tests for correlated processes use the initial timestamp across independent paths. Plots of sampled OU values are descriptive and explicitly labeled correlated. Constant-distribution quantile intervals use exact binomial/order-statistic inversion.
- Temporal moments and ACF use independent path summaries, with approximate Student intervals requiring at least 30 paths. ACF at 30, 300 and 600 seconds uses known zero mean and configured variance, not sample demeaning/normalization. Unresolvable lags are reported. Standardized OU innovation mean/variance use exact Gaussian/chi-square null formulas with unfitted parameters; serial covariance checks never join different paths.
- Holm adjustment controls familywise significance across all eligible primary statistical p-values within one input even when tests overlap. Confidence intervals shown in tables and plots are **marginal**, not simultaneous intervals. Numerical invariants have explicit tolerances. Pearson independence tests assume the configured Gaussian joint null; failure to detect correlation does not prove general independence.
- Magnetic ASD divides total-vector requirement by `sigma_level*sqrt(3)` and converts pT to nT. The low-frequency requirement is flat below the first anchor. Welch uses periodic Hann windows, 50% overlap, no detrending, a broadband length `min(8192,N)`, and a separate full-record tapered estimate. Estimates are averaged in **PSD** before conversion to ASD.
- The theoretical tapered spectrum is calculated from the finite periodic-record covariance and actual window autocorrelation, including missing DC and the implemented even-N half-power Nyquist bin. Each spectral gate uses one band/slope/power summary per independent series; overlapping segments and adjacent bins do not inflate sample count. Spectral plot intervals use variation across independent paths.
- Anchor tests integrate resolved neighborhoods (at least three bins). Slope gates use a linearized log-PSD contrast about the window-adjusted expectation; its residual has exactly zero expectation under the target spectrum. It measures local slope disagreement and is not an unbiased estimate of a large alternative power-law exponent. Segment slope bands require at least eight bins. The full-record low-frequency plot makes the limited 0.001 Hz resolution visible.

No observation/truth residuals, speed-preservation application tests or shifted Malara field evaluations are included. Matrix/Rodrigues checks concern stored latent rotations only. No Phase C synthetic campaign or default pytest/CI integration is provided.

## Explicit lightweight self-tests

The self-tests live outside the default `tests/` collection and use tiny fixtures independent of production generators:

```bash
.venv/bin/python -m pytest scripts/validator_selftests/test_measurement_error_validation.py -q
.venv/bin/ruff check scripts/validate_measurement_errors.py scripts/measurement_error_validation scripts/validator_selftests/test_measurement_error_validation.py
.venv/bin/ruff format --check scripts/validate_measurement_errors.py scripts/measurement_error_validation scripts/validator_selftests/test_measurement_error_validation.py
```

## Selectable model families

The validator now dispatches from the **embedded** density, speed and angular
model identifiers. Its independent mathematical conversions do not import the
production generators or conversion helpers. Supplied YAML is cross-checked,
not used to override the product's embedded statistical interpretation.

- `multiplicative_bias_plus_ou`: read `*_log_error`, subtract saved bias, use
  Gaussian log-space targets and `expm1` for fractional summaries. Historical
  result names are retained.
- `gaussian_fractional_bias_plus_ou`: read `*_fractional_error`, subtract saved
  bias and test Gaussian fractional statistics directly. Mean fractional error
  is zero; the validator does not take logarithms or exponentiate these errors.
  It also checks positive factors and symmetric requirement coverage at the
  initial timestamp across independent paths.
- `transverse_ou_rotation`: retain the per-coordinate `requirement_deg/k` sigma
  and corresponding Rayleigh radial coverage (86.47% at the configured radius
  when k=2).
- `transverse_ou_rotation_total_angle`: independently derive component sigma
  from Rayleigh coverage `erf(k/sqrt(2))`; test Gaussian/OU coordinate statistics,
  Rayleigh angle statistics and configured-radius coverage.

New normalized provenance explicitly identifies `error_interpretation` for
log/fractional models and `requirement_interpretation` for angular models.
Historical missing interpretation fields are inferred from the legacy model
identifiers, preserving old-file validation. Contradictory model/interpretation
metadata is rejected. If present, dataset model/interpretation attributes must
agree with provenance; new fractional datasets and biases require dimensionless
`units='1'`. Missing attributes on historical log-error products remain allowed.

For large angle requirements the Rayleigh target applies to the latent angle;
principal vector deflection wraps above pi. The informational
`total_angle.wrapping_probability` row reports the theoretical wrapping
probability. New Gaussian fractional generation aborts on nonpositive or
nonfinite factors without clipping or resampling. If such events are not rare,
selecting only completed products can bias the saved ensemble; the validator
continues to test the stated unconditioned Gaussian targets.

Generation snippets and full storage conventions are in the existing
[error-model reference](../docs/reference/errors.md#alternative-fractional-and-total-angle-requirements).
The Phase A CLI, read-only behavior and campaign commands above are unchanged;
point them at newly generated products and their corresponding YAML when ready.
No production products or generation recipes were changed to add these options.

Run the additional error-only statistical and serialization checks explicitly:

```bash
.venv/bin/python -m pytest tests/test_observatory_errors.py tests/test_hdf5_roundtrip.py scripts/validator_selftests/test_measurement_error_validation.py scripts/validator_selftests/test_observatory_model_options.py -q
.venv/bin/ruff check src/hs_observatory/errors/observatory.py src/hs_observatory/io/hdf5.py tests/test_observatory_errors.py scripts/measurement_error_validation/model.py scripts/measurement_error_validation/runner.py scripts/measurement_error_validation/reporting.py scripts/validator_selftests/test_observatory_model_options.py
.venv/bin/ruff format --check src/hs_observatory/errors/observatory.py src/hs_observatory/io/hdf5.py tests/test_observatory_errors.py scripts/measurement_error_validation/model.py scripts/measurement_error_validation/runner.py scripts/measurement_error_validation/reporting.py scripts/validator_selftests/test_observatory_model_options.py
```

The explicit statistical tests use independent error-only ensembles: 60,000
three-timestamp paths for fractional errors and 20,000 three-timestamp paths
for each angular sigma level. Normal/chi-square, binomial and correlation
checks use conservative predeclared tail bounds. They stay outside default
pytest collection. Small deterministic contracts and the pre-extension legacy
array/RNG reference are in the ordinary observatory tests. The reference was
captured before the implementation changes and includes all error components;
it does not evaluate a Malara field.
