# Timing investigations: historical experiments

These are distinct research experiments. Their original numerical findings,
warnings and setup details are preserved below; documentation migration does not
rerun or newly validate them. Follow the [method reference](../reference/timing-calibration.md)
for implemented equations. Generated artifacts are git-ignored and are
not promised as distributed data. A commit/data provenance record remains needed.

<a id="affine-projection-experiment"></a>

## All-nine affine projection investigation

The implementation and configurations are described in
[timing calibration](../reference/timing-calibration.md#all-nine-affine-projection-methods).

### Validation

Run only the relevant timing tests:

```sh
python -m pytest tests/test_timing_affine.py tests/test_timing_calibrate.py \
  tests/test_timing_influence.py -q
```

The new tests check projector symmetry/idempotence and ranks 15, 16, and 1;
orthogonal divergence/non-affine subspaces; additive residual powers; static and
changing geometry; rejection of degenerate geometry; and trace-free affine fields.
Known Node clock offsets are recovered within 1% of the largest injected offset
using an identifiable temporally varying affine field. Reducing offsets by a
factor of ten reduces the recovery truncation error by more than a factor of six.
Zero-error affine fields give zero corrections to numerical tolerance. Tests also
check the Hub/sign convention, actual resampled residual power, positive timing
covariance, HDF5/CLI plotting, independence from synthetic truth, and exact equality
of explicit/default polyhedron results. Existing timing/influence tests remain.

### Nonlinear zero-clock-error experiment

The deterministic fixture in `tests/test_timing_affine.py::turbulent_field` is a
random-phase, divergence-free Fourier surrogate for turbulent spatial structure:
24 modes, wave-number magnitudes logarithmically spaced from 0.2 to 3 rad/km,
transverse amplitudes proportional to `|k|^(-5/6) / sqrt(24)`, and advection at
`(0.7, -0.3, 0.2)` km/s. Each amplitude is perpendicular to its wave vector, so the
analytic spatial divergence vanishes. This is a synthetic surrogate, not an MHD
simulation or a claim of a resolved inertial range.

Nine static positions are standard-normal coordinates in km from RNG seed 19.
There are 241 samples from 0 to 8 seconds, with **zero injected timing offsets**.
Field seeds 47, 48, and 49 give three independent field realizations. Every method
uses the identical measurements, geometry, interval, and `sigma_t_s = 0.0125` for
each seed. Both projections use `sigma_B_nT = 0.1`; the polyhedron method uses all
382 subsets with its unchanged mean-square ideal-divergence weights and a
`1e-12` nT/km floor. Reported RMS is across the eight Nodes, excluding the Hub.

| Field seed | Polyhedron RMS [ms] | Divergence-only RMS [ms] | Full DF affine RMS [ms] |
| --- | ---: | ---: | ---: |
| 47 | 232.690 | 31.128 | 44.270 |
| 48 | 160.344 | 6.123 | 36.963 |
| 49 | 309.255 | 11.803 | 78.506 |

All nine fits have data-only timing rank eight. The divergence-only condition
numbers are approximately 14,684, 422, and 5,269, versus 3.50, 3.10, and 4.16 for
the full affine method. Thus full rank does not guarantee good conditioning, and
better conditioning does not eliminate model-induced timing bias. All actual
resampled total objectives decreased on their respective common valid samples.

Both projection methods still produce spurious corrections: spatial nonlinearity
can be reduced by shifting clocks even when the true clocks are correct. The full
method tries to remove non-affine structure as well as divergence. In these three
fixtures its spurious RMS exceeds the divergence-only result. Both are below the
polyhedron result **at these chosen scales**, but the change in normalization and
prior/data balance prevents attributing that difference solely to elimination of
overlapping polyhedra. A residual-scale sweep and application to the intended MHD
data are needed before making that conclusion. The tests deliberately do not
require zero spurious timing corrections or a particular method ranking.

Reproduce the numerical experiment through the targeted test:

```sh
python -m pytest tests/test_timing_affine.py \
  -k turbulent_zero_offset_spurious_corrections -q -s
```

Saved products from the investigation are under
`outputs/affine_projection_investigation/`: per-seed observations, each method's
HDF5 output, `results.json` (RMS, maximum offset, rank, condition number, and actual
objectives), and `seed47_comparison.png`. The HDF5 products can be replotted through
the normal CLI without the observation source.

<a id="quadratic-projection-experiment"></a>

## Divergence-free quadratic timing investigation

The [quadratic method](../reference/timing-calibration.md#exact-divergence-free-quadratic-projection)
shares the affine projection's solver, interpolation, MC writer, and plotting path.
Only the spatial design and its residual component change.

Validation passed **79 targeted tests**:

```sh
python -m pytest tests/test_timing_quadratic.py tests/test_timing_affine.py \
  tests/test_timing_calibrate.py tests/test_timing_influence.py -q
```

New checks reconstruct the symmetric H tensor and verify all four divergence
constraints for every constrained parameter column. An independent field made of
transverse vectors multiplied by univariate quadratic polynomials has zero
projected residual to numerical precision. Tests cover rank-one symmetry and
idempotence, static/changing geometry, invariance to coordinate scale and
translation, rank-deficient and ill-conditioned geometry rejection, and known
clock-error recovery in a time-varying exactly quadratic field. The recovery
fixture uses offsets up to 0.1 ms and a 1 μs absolute tolerance; its zero-error
corrections vanish to 1e-12 s. Hub/sign, actual resampled objective, covariance,
MC/HDF5, truth independence, and four-method CLI comparison are also checked.

Two warnings in the integration test report actual objective increases in the
existing methods on a quadratic field. The tests intentionally retain the
existing warning behavior; a linearized objective decrease does not guarantee an
actual resampled decrease. In a separate regression check, all three existing
methods reproduced the saved seed-47 corrections, covariances, and actual
objectives exactly.

### Zero injected timing error in a nonlinear field

This experiment uses exactly the observations from the
[affine investigation](timing-investigations.md#affine-projection-experiment): the same 24-mode solenoidal
Fourier surrogate, three seeds, nine positions, and 241 samples over 0–8 s.
Every injected clock error is zero. All methods use `sigma_t_s = 0.0125`.
All three projection methods use `sigma_B_nT = 0.1`. The polyhedron method retains
its existing ideal-divergence weights and all 382 subsets.

| Seed | Polyhedron RMS [ms] | Divergence-only RMS [ms] | DF affine RMS [ms] | DF quadratic RMS [ms] | Quadratic max [ms] |
| --- | ---: | ---: | ---: | ---: | ---: |
| 47 | 232.690 | 31.128 | 44.270 | 12.306 | 22.785 |
| 48 | 160.344 | 6.123 | 36.963 | 6.761 | 12.974 |
| 49 | 309.255 | 11.803 | 78.506 | 5.696 | 12.663 |

RMS and maxima are across eight Nodes, excluding the Hub. All three quadratic
fits have design rank 26, design condition number 312.164, and data-only timing
rank eight. Timing condition numbers are approximately 13,809, 600, and 13,223:
a well-conditioned spatial design does not ensure well-conditioned clock recovery.
Actual total objectives decreased from 70.314 to 61.435, 177.867 to 175.450, and
57.620 to 55.810, respectively, on each fit's common valid samples.

The quadratic projection still yields nonzero spurious clock corrections because
the nonlinear field is not exactly quadratic. It reduces RMS relative to full
affine projection in these fixtures, but does not beat divergence-only projection
for every seed. This small Fourier-surrogate experiment does not establish a
ranking on MHD turbulence. The legacy method's different weights also prevent
attributing cross-method differences solely to spatial order or overlap removal.
No quadratic regularization or scale tuning was applied.

Reproduce the quadratic measurements with:

```sh
python -m pytest tests/test_timing_quadratic.py -k turbulent_zero_offset -q -s
```

Saved quadratic products, numerical `results.json`, and `seed47_comparison.png`
are under `outputs/quadratic_projection_investigation/`. The original observations
and three-method products remain under `outputs/affine_projection_investigation/`.

## Instrument-noise paired experiment

**Release review needed:** `analysis/timing_noise_experiment.py` and
`configs/helioswarm_mag_scm_noise.yaml` are now tracked. The previous reproduction
note described them as unavailable or untracked. Availability of the exact
historical comparison configurations and agreement with the current script still
need review. No experiment has been rerun here; the results below remain a
historical record, not new release validation.

<a id="reproducible-paired-experiments"></a>

### Historical paired-experiment command

```text
# Historical command; not rerun for this documentation review
.venv/bin/python analysis/timing_noise_experiment.py
```

This runs zero-offset and nonzero-offset experiments on a seeded solenoidal
Fourier surrogate, using identical source samples, geometry, selection, priors,
and injected offsets for both normalizations. Each case has four realizations;
the zero case repeats identical measurements. Cadence is 32 Hz, ten 8/9-member
polyhedra are retained, and the timing prior is 4 ms. The source contains no
random instrument noise: this isolates the effect of changing normalization,
not a noise-performance forecast. Finite-volume divergence may be nonzero even
for this solenoidal field; no spatial-curvature correction is introduced.

HDF5 sources, analysis products, exact options, overview/time-series/weight and
comparison figures, and per-case bias/RMSE summaries are written under
`outputs/timing_noise_experiment/`. These generated files are git-ignored.

The seeded run on 2026-09-24 yielded sigma_B = 0.01856635 nT per component,
integrated over 0–16 Hz. Pooled timing-recovery RMSE across realizations and Nodes:

| Injected timing | Ideal-divergence normalization | Instrument-noise normalization |
| --- | ---: | ---: |
| Zero | 17.866 ms | 82.477 ms |
| Nonzero (4 ms draw standard deviation) | 18.828 ms | 82.598 ms |

These are an implementation comparison, not a demonstration of improved timing
accuracy. Nonzero discrete divergence produces zero-error bias; instrument
weights are much larger here and increase that bias. The corrected objective is
about 95.7% divergence in the zero-error ideal-normalization run and 99.978% in
the instrument-normalization run. The remaining contribution is the timing
prior. Recovery errors this large also limit the first-order timing model's
accuracy. No curvature model or change to correlated-polyhedron treatment was
used to improve these results. Per-Node signed biases and RMSEs are retained in
`summary.yaml` and each product's `mc_summary`.

Validation: 90 focused timing tests passed (noise, legacy polyhedron, affine,
quadratic, and influence suites). Six diagnostic warnings report increases in
resampled corrected chi-square; they are not test failures. The new experiment
script passed its targeted Ruff check.
