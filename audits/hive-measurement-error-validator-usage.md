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
- Density/speed requirements specify log-error sigma; normalized variance fractions allocate independent constant bias and stationary OU variance. The code subtracts **saved bias**, never each path's temporal mean. Lognormal fractional mean is positive without a centering correction.
- Velocity-angle sigma applies to **each** transverse OU coordinate. The total latent angle is Rayleigh, with RMS sqrt(2) times component sigma. Alignment uses independent Gaussian B/V rotation vectors and Maxwell rotation magnitudes, not Euler angles.
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
