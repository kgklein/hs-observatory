# All-nine affine projection investigation

The implementation and configurations are described in
[timing calibration](timing_calibration.md#all-nine-affine-projection-methods).

## Validation

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

## Nonlinear zero-clock-error experiment

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
