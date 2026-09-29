# Magnetic-gradient divergence audit

Date: 2026-07-26

Historical report, preserved without rerunning the numerical audit. Reproduction
requires the named simulation inputs, observation products and optional scientific
dependencies; generated artifacts are not distributed by default. See the
[example manifest](../examples/index.md#simulation-and-gradient-workflows),
[current gradient reference](../reference/magnetic-gradients.md), and
[visualization guide](../user-guide/visualization.md).

Scope: `outputs/sim-prod-400.h5`,
`outputs/sim-prod-400_magnetic_gradients.h5`, and the source Gkeyll snapshot
`ot3D_field_75.bp`. No solenoidal constraint was added and no scientific
estimator was changed.

## Assessment

**Category D: multiple contributing effects, dominated by the finite-volume
sparse affine-model limitation.**

No tensor-indexing, pair-counting, normalization, coordinate, unit, curl,
divergence, or centered-difference implementation error was found.

The order-unity divergence-to-curl ratios are mainly produced by fitting one
unconstrained affine tensor to sparse samples of a nonlinear, finite-scale
field. This is not the same operation as averaging the true gradient over a
sphere or convex hull. A divergence-free nonlinear field can therefore yield a
nonzero trace in its best finite-sample affine reconstruction.

There are two smaller contributions:

1. The saved Gkeyll grid is not machine-zero solenoidal under the repository's
   centered-difference operator. Its RMS divergence is 2.48% of its RMS curl.
2. Trilinear interpolation contributes appreciably when separations are at or
   below one grid spacing. It becomes a small correction in controlled tests
   once a configuration spans several grid cells. The 61.25 km small
   configuration spans 5.82 x/y cells but only 1.16 z cells, so it remains
   resolution-sensitive along z.

The larger 1,012–1,642 km configurations are well above grid resolution.
Their large ratios are instead associated with finite-scale nonlinearity and
kernel mismatch, not poor conditioning: their condition numbers are only
2.05, 2.12, and 9.66 and are static in time.

## Phase 1: current implementation and data flow

### Estimator and geometry

- `src/hs_observatory/analysis/magnetic_gradient.py`
  - `geometry_diagnostics`: subtracts the barycenter and constructs
    `R = Xc.T @ Xc / N`.
  - `estimate_gradient_least_squares`: uses independent unordered pairs
    from `np.triu_indices(N, k=1)`, constructs
    `sum(ΔB Δx.T) / N**2`, and right-multiplies by `R^-1`.
  - The stored convention is `gradient[field_component,
    derivative_direction]`.
  - The fitted field is
    `B_mean + (x - x0) @ gradient.T`, hence
    `gradient[i,j] = dB_i/dx_j`.
  - `gradient_curl_divergence`: takes the trace for divergence and the
    antisymmetric components for curl.

### Simulation sampling and truth

- `src/hs_observatory/io/simulation_bp.py`
  - Reads the BP spatial axes without transposition.
  - Selects zero-based packed components `[3,4,5]` as `(Bx,By,Bz)`.
- `src/hs_observatory/fields/simulation_sampled.py`
  - Converts the configured box from proton gyroradii to km.
  - Maps the initial Hub to the box center with a componentwise translation.
  - Applies periodic wrapping without axis permutation or reflection.
  - Samples `(nx,ny,nz,component)` arrays with periodic trilinear
    interpolation.
- `src/hs_observatory/analysis/magnetic_gradient.py`
  - `periodic_grid_gradient`: centered periodic second-order differences with
    output convention `G[i,j]`.
  - `_truth_products`: trilinearly samples the point gradient at the
    barycenter and constructs a spherical top-hat average of the grid
    gradient. The FFT path convolves grid nodes and then trilinearly samples
    the convolved result.
- `align_spacecraft_samples`
  - Applies linear time interpolation independently per spacecraft.
  - Ideal estimates do not use this synchronization path; they use nominal
    positions and saved truth fields directly.

### Units and coordinates

- CDF positions are in km.
- The simulation box is `[4712.3889, 4712.3889, 23561.9448]` km.
- Grid shape is `448 x 448 x 448`.
- Grid spacing is `[10.5187, 10.5187, 52.5936]` km.
- BP magnetic values are multiplied by 25 and stored in nT.
- Gradients are therefore nT/km.
- Re-evaluating the raw grid at the mapped nine-spacecraft positions at
  `t=175 s` reproduces `/truth/B` to a maximum error of
  `1.78e-15 nT`.
- The map is a translation plus periodic wrap with identity axis map and
  handedness `+1`.
- Initial density- and magnetic-cadence spacecraft positions agree exactly,
  so using the initial density position to set the common mapping reference
  introduces no offset.

The HDF5 datasets themselves do not carry position and truth-B unit
attributes; units are recorded in metadata. This is a schema robustness issue,
not the source of the present divergence.

## Phase 2: pairwise normalization

For centered matrices `Xc` and `Bc`,

```text
sum_(a<b) (Ba-Bb)(xa-xb)^T = N Bc^T Xc
R = Xc^T Xc / N
```

Therefore the implemented unordered-pair form gives

```text
[sum_(a<b) ΔB Δx^T / N^2] R^-1
  = Bc^T Xc (Xc^T Xc)^-1.
```

This is the ordinary centered least-squares estimator.

A literal ordered sum over `a != b` is twice the unordered sum. With the
same `1/N^2` prefactor and the stated definition of `R`, it produces `2G`.
Thus the reference equation must either:

- use independent unordered pairs, as the repository does; or
- use `1/(2N^2)` for a literal ordered sum.

Random comparisons:

| N | maximum pairwise–centered element difference |
|---:|---:|
| 4 | `1.07e-14` |
| 5 | `6.75e-14` |
| 9 | `3.11e-15` |

## Phases 3–5: affine recovery, tensor orientation, and four-point method

The audit tests recover all nine tensor components for:

- divergence-free/curl-free diagonal tensor;
- identity tensor with divergence 3;
- pure positive-z curl;
- dense nonsymmetric tensor;
- regular and irregular tetrahedra;
- five points;
- nine points.

Maximum errors are at floating-point precision. The fitted-field assertion
explicitly verifies

```text
B_fit = B_mean + (x - x0) @ G.T
```

and therefore `G[i,j] = dB_i/dx_j`.

For arbitrary four-point field samples, an independent affine-system
(reciprocal-basis) solve agrees with the generalized estimator:

| configuration | max tensor difference | max curl difference | divergence difference |
|---|---:|---:|---:|
| regular tetrahedron | `1.11e-16` | `2.22e-16` | `4.44e-16` |
| irregular tetrahedron | `2.22e-16` | `2.22e-16` | `2.22e-16` |

### Sign note for the requested pure-curl test

Under `G[i,j] = dB_i/dx_j`, the stated field

```text
Bx = -Omega y
By =  Omega x
```

has

```text
G = [[0, -Omega, 0],
     [Omega, 0, 0],
     [0, 0, 0]]
```

and curl `(0,0,2 Omega)`. The matrix displayed in the audit request has the
opposite off-diagonal signs and would give negative z curl. The repository
matches the field equations and the positive-curl expectation.

## Phase 7: independent simulation derivative

The production centered differences were compared against a transparent
component-by-component implementation on the full 448-cubed grid. RMS and
maximum differences are exactly zero for all nine tensor components.

Grid statistics:

| quantity | value |
|---|---:|
| RMS divergence | `2.0625e-4 nT/km` |
| RMS curl magnitude | `8.3050e-3 nT/km` |
| RMS divergence / RMS curl | `0.02483` |
| RMS curl / RMS divergence | `40.27` |
| median absolute divergence | `1.1158e-4 nT/km` |
| 99th percentile absolute divergence | `6.5212e-4 nT/km` |

The snapshot is numerically close to, but not exactly, solenoidal under this
cell-centered diagnostic. Its source is the finite-volume Gkeyll
`ot3D.lua` calculation. This grid-level divergence is too small to explain the
order-unity sparse ratios by itself.

## Phase 8: interpolation isolation with an analytic field

The controlled field

```text
B = (sin(k z), sin(k x), sin(k y))
```

is analytically divergence-free. It was evaluated both directly and through
the same periodic trilinear interpolator, using the actual normalized
small, best-five, and all-nine geometries with random orientations and
barycenters.

Representative results for a wavelength of 16 grid spacings:

| configuration scale | direct median ratio | interpolated median ratio | interpolation fraction of divergence RMS |
|---:|---:|---:|---:|
| `0.5 dx` | `0.013–0.016` | `0.013–0.020` | `0.65–0.82` |
| `1 dx` | `0.029–0.036` | `0.019–0.031` | `0.36–0.44` |
| `4 dx` | `0.119–0.154` | `0.122–0.159` | `0.028–0.040` |
| `8 dx` | `0.241–0.449` | `0.241–0.454` | `0.014–0.018` |

At sub-grid scales, interpolation is a substantial part of a small absolute
divergence. At finite scales of several cells, direct and interpolated
sampling give nearly the same order-unity trend: the affine approximation to
the nonlinear field dominates.

## Phase 9: sparse fit versus point, spherical, dense, and hull references

The audit uses 2,048 samples for each dense sphere and convex hull. The final
column below is the average of the independently differentiated simulation
gradient inside the same convex hull, whereas “hull fit” is an affine
regression to field values in that hull.

Values are `|div B|/|curl B|`:

| configuration | selection/time (s) | sparse | point | spherical | dense sphere fit | dense hull fit | hull-average gradient |
|---|---|---:|---:|---:|---:|---:|---:|
| small | min / 175.156 | 0.001 | 0.008 | 0.003 | 0.004 | 0.014 | 0.004 |
| small | near 1 / 195.812 | 0.987 | 0.018 | 0.021 | 0.020 | 0.283 | 0.023 |
| small | peak / 158.312 | 2.142 | 0.066 | 0.010 | 0.027 | 0.304 | 0.026 |
| intermediate | min / 183.000 | 0.000 | 0.005 | 0.000 | 0.016 | 0.115 | 0.014 |
| intermediate | near 1 / 165.406 | 1.000 | 0.056 | 0.000 | 0.004 | 0.213 | 0.003 |
| intermediate | peak / 158.375 | 3.168 | 0.025 | 0.000 | 0.032 | 0.102 | 0.001 |
| best five | min / 160.594 | 0.000 | 0.041 | 0.000 | 0.055 | 0.344 | 0.002 |
| best five | near 1 / 150.719 | 0.998 | 0.040 | 0.000 | 0.010 | 0.386 | 0.003 |
| best five | peak / 150.000 | 3.419 | 0.066 | 0.000 | 0.012 | 0.107 | 0.004 |
| all nine | min / 171.250 | 0.001 | 0.037 | 0.000 | 0.002 | 0.203 | 0.000 |
| all nine | near 1 / 181.750 | 0.998 | 0.008 | 0.000 | 0.000 | 0.089 | 0.001 |
| all nine | peak / 150.000 | 3.248 | 0.002 | 0.001 | 0.019 | 1.154 | 0.012 |

The derivative averages remain near solenoidal. Affine fits over finite
volumes can acquire a larger trace, and sparse boundary-like samples amplify
it further. This is the central evidence for the conclusion.

For four spacecraft, the affine residual is machine zero by construction:
four non-coplanar vector samples determine an affine vector field exactly.
It therefore cannot be used as a nonlinearity diagnostic for a tetrahedron.
The dense-volume residual is the meaningful comparison.

## Phase 10: scale dependence on the actual simulation

The geometric-mean grid spacing is 17.99 km. Median sparse ratios over random
barycenters and orientations were:

| scale | small | best five | all nine |
|---:|---:|---:|---:|
| `0.5 d` | 0.022 | 0.031 | 0.033 |
| `1 d` | 0.079 | 0.072 | 0.046 |
| `2 d` | 0.155 | 0.095 | 0.108 |
| `4 d` | 0.254 | 0.025 | 0.169 |
| `8 d` | 0.288 | 0.223 | 0.246 |
| `16 d` | 0.322 | 0.311 | 0.448 |

There is sampling scatter, but the broad increase with finite scale is clear.
Condition number is unchanged by uniform scaling, so this trend is not a
geometry-conditioning effect.

## Phase 11: 150–200 s interval

| configuration | L (km) | Q | condition | median ratio | p90 | maximum | fraction >= 1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| small | 61.25 | 0.182 | 1.70 | 0.227 | 0.614 | 2.142 | 0.015 |
| intermediate | 1068.36 | 0.269 | 2.12 | 0.449 | 1.266 | 3.168 | 0.169 |
| best five | 1011.92 | 0.251 | 2.05 | 0.450 | 1.289 | 3.419 | 0.176 |
| all nine | 1641.73 | 0.638 | 9.66 | 0.586 | 1.565 | 3.248 | 0.320 |

The complete representative table includes barycenters, `E`, `P`, `Q`,
volumetric-tensor eigenvalues, all gradient tensors, divergence, curl,
residuals, and coarse cell counts.

## Phase 12: diagnostic artifacts

Generated by:

```bash
MPLCONFIGDIR=/tmp/hsobs-mpl PYTHONPATH=src \
  .venv/bin/python scripts/audit_magnetic_gradient_divergence.py \
  --dense-samples 2048
```

Artifacts:

- `outputs/magnetic_gradient_divergence_audit/audit_summary.json`
- `outputs/magnetic_gradient_divergence_audit/representative_samples.csv`
- `outputs/magnetic_gradient_divergence_audit/actual_scale_dependence.csv`
- `outputs/magnetic_gradient_divergence_audit/analytic_interpolation_scale_dependence.csv`
- `outputs/magnetic_gradient_divergence_audit/debug_timeseries.png`

The debugging figure contains:

1. ideal, coarse, and dense-sphere curl magnitude;
2. ideal, coarse, and dense-sphere divergence;
3. divergence-to-curl ratio;
4. divergence-to-gradient-Frobenius-norm ratio;
5. curl-component errors;
6. condition number;
7. dense-sphere affine residual;
8. number of cells in the coarse sphere.

Intervals where coarse truth curl is below 10%, 5%, and 1% of its interval
median are shaded.

## Recommendations

1. Do not force the estimated trace to zero.
2. Interpret sparse `|div B|/|curl B|` primarily as a finite-configuration
   affine-closure diagnostic, not a direct measurement of physical
   simulation divergence.
3. Retain pointwise and spherical-gradient truth, but add a matched estimator
   reference when validating finite configurations: dense affine fits using
   the same spatial support are more comparable than an unweighted spherical
   average of the true gradient.
4. For interpolation robustness, require at least two cells along the
   coarsest grid direction (`L >= 2 dz`, about 105 km here) when possible.
   This is only a resolution lower bound. A simultaneous upper bound is also
   required: the configuration must remain small relative to the local
   nonlinear field scale. There is no universal minimum scale that fixes the
   finite-affine effect.
5. Treat the current 61.25 km small configuration as under-resolved in z for
   interpolation-sensitivity claims, although its order-unity peaks still
   exceed the grid and dense-volume divergence by a large factor.
6. Add explicit HDF5 unit attributes to position and truth-field datasets and
   replace the implicit identity simulation/spacecraft orientation with
   explicit rotation metadata before using a non-identity physical mapping.
