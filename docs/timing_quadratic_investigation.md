# Divergence-free quadratic timing investigation

The [quadratic method](timing_calibration.md#exact-divergence-free-quadratic-projection)
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

## Zero injected timing error in a nonlinear field

This experiment uses exactly the observations from the
[affine investigation](timing_affine_investigation.md): the same 24-mode solenoidal
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
