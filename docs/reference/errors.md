# Coordinate and measurement errors

See [visualization guide](../user-guide/visualization.md) for inspection and plotting.

For the generation configuration script, details on the error models will need to be described, using the following form:

```yaml
errors:
  random_seed: 12345
  n_mc: 10
  time:
    model: gaussian_constant
    sigma_s: 0.01
  position:
    model: gaussian_constant
    sigma_km: [0.5, 0.5, 0.5]
  variables:
    n: {model: gaussian_relative, sigma: 0.05}
    v: {model: gaussian_vector_component, sigma: [2.0, 2.0, 2.0]}
    B: {model: gaussian_absolute, sigma: [0.05, 0.05, 0.05]}
```

## Coordinate and timing errors

Time (`time`) and position (`position`) uncertainties are sampled once per Monte Carlo realization and spacecraft, then held constant over that spacecraft's entire time series.
The example model prescribed a Gaussian with mean zero and width `sigma`.
The same temporal and spatial offsets apply to measurements of `n`, `v`, and `B`.

Quantitatively, each observation is evaluated from the field model at

```text
t_sample = t_nominal + time_offset
r_sample = r_trajectory(t_nominal + time_offset) + position_offset
```

before its variable-specific measurement error is added.
For velocity, the selected plasma- or spacecraft-frame transformation occurs before applying that measurement error.
The HDF5 `/time/*` and `/spacecraft/r/*` datasets retain the nominal coordinates.

Set either coordinate model to `none` to disable it. Scalar `sigma_km` applies the same width to X, Y, and Z; a three-component value sets their widths independently. `/errors/n`, `/errors/v`, and `/errors/B` contain the additive measurement-noise realizations; coordinate-induced differences are represented by the stored offsets rather than folded into those arrays.

## Measurement errors

In addition to timing and position errors, actual instruments will introduce error into the observations. Those can be modeled with the addition of noise drawn from a simple normal distribution,
`gaussian_relative` for scalars, and `gaussian_vector_component` for vectors:

```yaml
errors:
  random_seed: 12345
  n_mc: 10
  time:
    model: gaussian_constant
    sigma_s: 0.01
  position:
    model: gaussian_constant
    sigma_km: [0.5, 0.5, 0.5]
  variables:
    n: {model: gaussian_relative, sigma: 0.05}
    v: {model: gaussian_vector_component, sigma: [2.0, 2.0, 2.0]}
    B: {model: gaussian_vector_component, sigma: [0.05, 0.05, 0.05]}
```

### Ornstein–Uhlenbeck measurement errors

Use `model: ornstein_uhlenbeck` for time-correlated measurement noise.
OU widths are stationary standard deviations, and correlation times are in seconds.
The implementation uses the exact discrete transition for each interval, so the same physical parameters work correctly with different or irregular instrument cadences.

```yaml
errors:
  variables:
    n:
      model: ornstein_uhlenbeck
      sigma: 0.25          # absolute density units
      tau_s: 2.0
    v:
      model: ornstein_uhlenbeck
      magnitude_sigma: 0.005       # fractional |v| error
      magnitude_tau_s: 2.0
      direction_sigma_deg: 0.3     # RMS total angular error
      direction_tau_s: 1.0
    B:
      model: ornstein_uhlenbeck
      magnitude_sigma: 0.01        # fractional |B| error
      magnitude_tau_s: 0.5
      direction_sigma_deg: 0.5     # RMS total angular error
      direction_tau_s: 0.25
```

Density error is additive: `n_observed = n_sampled + epsilon`.
Vector magnitude is changed by `max(0, 1 + epsilon)`, preventing an unphysical negative magnitude.
Directional noise is an isotropic tangent-plane OU process; `direction_sigma_deg` specifies the stationary RMS total angular deviation. `direction_tau_s` defaults to `magnitude_tau_s`, and `direction_sigma_deg` defaults to zero.
Zero vectors remain zero due to their undefined direction.

The complete OU example can be run with:

```bash
hsobs-generate configs/example_generate_ou.yaml
```

### Observatory requirement model

Use `errors.model: observatory` for a more realistic requirement-motivated model based on HelioSwarm's expected performance.
It samples constant timing and position offsets,
multiplicative density/speed bias plus Ornstein–Uhlenbeck noise, transverse velocity angular precision offsets,
magnetic DC offsets, colored magnetic AC noise, and constant alignment rotations.
Alignment defaults to independent B/V rotations; an explicit relative MAG-to-FC
model applies a single rotation to the magnetic measurement only.

Each observatory error section accepts `sigma_level: 2` or `sigma_level: 3`
(the default) alongside a neutral, unit-specific `requirement_*` value. Levels
may be mixed between sections. Existing `requirement_3sigma_*` configuration
keys remain supported as 3-sigma compatibility aliases.

```yaml
errors:
  model: observatory
  timing:
    requirement_ms: 12.5
    sigma_level: 3
  density:
    requirement_fraction: 0.04
    sigma_level: 2
```

For each Monte Carlo realization, clock and position offsets are drawn once per
spacecraft and held fixed over the interval. The timing requirement is treated
as a 12.5 ms 3-sigma value by default. The position requirement is treated as a
2.5 km 99.73% radial 3D Gaussian contour, giving a default per-component sigma
of about 0.665 km. For each spacecraft, its trajectory is evaluated at the
nominal sample time plus its clock offset; the position offset is then added.
Fields are sampled at that perturbed time and position. Linear interpolation is
used within the stored trajectory, with endpoint-velocity extrapolation for
clock shifts beyond its first or last sample. Nominal coordinates remain in the
HDF5 coordinate datasets.

By default, density and velocity speed use lognormal multiplicative errors with a constant
spacecraft bias plus a stationary exact-discrete Ornstein-Uhlenbeck component.
The default density and speed 3-sigma fractional requirements are 6% and 3%,
respectively, with half the variance in the bias and half in the OU process.

Velocity direction receives two transverse OU angular processes relative to the
instantaneous velocity direction. The default angular precision is 0.25 degrees
at 3-sigma. With the default alignment model, velocity alignment is then applied
as a constant per-spacecraft/MC rotation matrix. Magnetic field receives a constant per-axis DC offset, colored
Gaussian AC noise generated in Fourier space from the merged-product ASD
requirement, and then a separate constant per-spacecraft alignment rotation.

The magnetic AC anchors are interpreted as 3-sigma total-vector ASD values in
pT/sqrt(Hz). They are converted to one-sigma per-component nT/sqrt(Hz) by
dividing by `3 * sqrt(3)` and by 1000. Log-log slopes are used between anchors,
and the final slope continues above 10 Hz.

More generally, magnetic AC anchors use `requirement_interpretation:
total_vector_asd` and are divided by `sigma_level * sqrt(3)`. Position
requirements use the matching isotropic 3D radial confidence contour: 95.45%
for 2 sigma and 99.73% for 3 sigma.

A full list of error parameters for the observatory model is below:

```yaml
errors:
  model: observatory
  random_seed: 12345
  n_mc: 50

  timing:
    enabled: true
    requirement_ms: 12.5
    sigma_level: 3
    model: constant_per_spacecraft

  position:
    enabled: true
    requirement_magnitude_km: 2.5
    sigma_level: 2
    model: isotropic_3d_gaussian_position_error

  density:
    enabled: true
    requirement_fraction: 0.06
    sigma_level: 2
    model: multiplicative_bias_plus_ou
    bias_variance_fraction: 0.5
    ou_variance_fraction: 0.5
    ou_correlation_time_s: 300.0

  velocity_speed:
    enabled: true
    requirement_fraction: 0.03
    sigma_level: 2
    model: multiplicative_bias_plus_ou
    bias_variance_fraction: 0.5
    ou_variance_fraction: 0.5
    ou_correlation_time_s: 300.0

  velocity_angle:
    enabled: true
    requirement_deg: 0.25
    sigma_level: 2
    model: transverse_ou_rotation
    ou_correlation_time_s: 300.0

  magnetic_dc:
    enabled: true
    requirement_nT_per_axis: 0.15
    sigma_level: 3
    model: constant_per_axis_offset

  magnetic_ac:
    enabled: true
    model: colored_gaussian_noise_merged_product
    requirement_interpretation: total_vector_asd
    sigma_level: 3
    component_distribution: uniform
    anchors:
      - {frequency_Hz: 0.001, asd_pT_per_sqrtHz: 1500.0}
      - {frequency_Hz: 1.0, asd_pT_per_sqrtHz: 15.0}
      - {frequency_Hz: 10.0, asd_pT_per_sqrtHz: 1.5}
    extrapolate_above_10Hz: continue_last_loglog_slope

  alignment:
    enabled: true
    requirement_deg_per_axis: 3.0
    sigma_level: 3
    model: independent_constant_rotation_B_and_V
```

The generator writes sampled errors under `/errors`, including:

- `time_offset`, `position_offset`
- `density_bias`, `density_log_error`
- `velocity_speed_bias`, `velocity_speed_log_error`
- `velocity_angle_alpha`, `velocity_angle_beta`
- `magnetic_dc_offset`, `magnetic_ac_noise`
- `magnetic_alignment_rotvec`, `magnetic_alignment_matrix`
- `velocity_alignment_rotvec`, `velocity_alignment_matrix`

The exact merged configuration is stored as the HDF5 metadata attribute
`observatory_error_config`.

### Relative FC–MAG alignment

Select `relative_constant_rotation_B_to_V` to represent uncertainty in
reconstructing the MAG-to-FC instrument alignment under requirement
L2.5-MDRA-28. HIVE's existing Cartesian axes are the **nominal FC reference
frame**, with constant spacecraft attitude throughout the interval. The model
does not infer an alignment frame from the plasma velocity direction or add
instrument/attitude transformations.

Override the alignment section of an existing observatory recipe:

```yaml
errors:
  model: observatory
  alignment:
    enabled: true
    requirement_deg_per_axis: 3.0
    sigma_level: 3
    model: relative_constant_rotation_B_to_V

output:
  write_errors: true  # retain primitive errors for statistical validation
```

Retain the recipe's other settings, including its output path. The requirement
is **3° per rotation-vector component at 3 sigma**, giving a component standard
deviation of 1°. Each spacecraft and MC realization receives one independent
three-component Gaussian rotation vector, held fixed at every timestamp:

```text
sigma_axis = deg2rad(requirement_deg_per_axis / sigma_level)
omega[m,s] ~ Normal(0, sigma_axis**2 * I_3)
R[m,s] = exp([omega[m,s]]_cross)  # Rodrigues' formula

B_observed[m,t,s] = R[m,s] @ (B_sampled[m,t,s] + DC[m,s] + AC[m,t,s])
V_observed = V_after_configured_speed_and_direction_errors
```

The relative rotation acts on the entire merged MAG/SCM magnetic product,
including magnetic DC and AC perturbations. It introduces no FC rotation and
leaves velocity exactly unchanged by the alignment operation; other configured
velocity errors still apply. The spacecraft-relative velocity option remains
the existing velocity subtraction in common Cartesian axes, as described in
the [data reference](data-products.md#velocity-frame-convention).

`R` is a right-handed **active residual reconstruction-error rotation**: a
positive z-axis rotation sends +x toward +y. For row-vector arrays the helper
implements the same action as `values @ R.T`. The matrix maps ideally
FC-referenced magnetic components to erroneously reported FC-referenced
components; its transpose would undo that error. It is not a full physical
instrument attitude matrix.

These Gaussian coordinates are rotation-vector components, **not Euler
angles**. Their total magnitude follows a Maxwell distribution with scale
`sigma_axis`. At 1° component sigma, approximately 97.07% of rotation-vector
magnitudes are below 3°; 3° is not a 99.73% total-angle bound. The actual
B-vector angular deflection also depends on its direction relative to the
rotation axis: an axis-parallel vector does not move. Alignment alone preserves
B magnitude, and each matrix satisfies `R.T @ R = I` and `det(R) = 1`.

Both sigma levels 2 and 3 and finite nonnegative requirements are supported.
A zero requirement produces zero rotation vectors and identity matrices.
Disabled relative alignment also writes those values when errors are saved,
and consumes no alignment RNG draws. Unknown alignment model names are rejected.

#### Saved relative errors and compatibility

The relative model writes `/errors/relative_alignment_rotvec` and
`/errors/relative_alignment_matrix` in place of the four legacy B/V alignment
datasets. Dataset shapes, units and convention metadata are described in
[alignment data products](data-products.md#alignment-error-datasets).
Legacy datasets retain their original meaning. The normalized alignment model
remains in `observatory_error_config`, and the JSON metadata attribute
`relative_alignment` records the resolved FC reference and residual convention.

`independent_constant_rotation_B_and_V` remains the default. Its independent
1° B/V coordinate errors imply approximately `sqrt(2)` degrees of relative
coordinate uncertainty at small angles; they do not represent the new single
1° relative error. Existing 6xx/7xx recipes and products are unchanged. The new
model is an explicit selection for future 8xx recipes.

For matched seeds and settings, switching between the two **enabled** models
preserves unrelated latent errors and the final RNG state: the relative model
discards the legacy V normal block and uses the legacy B block as its sole
physical rotation. This also preserves the magnetic observation. Enabling
versus disabling alignment retains the historical downstream RNG difference.
Readers that require legacy alignment dataset names need explicit support for
the relative representation; the HDF5 schema remains version 2.

#### Phase A statistical validation

Use the standalone validator on saved errors without reevaluating Malara fields:

```bash
python scripts/validate_measurement_errors.py \
  --input path/to/observations.h5 \
  --config path/to/generation.yaml \
  --output-dir outputs/error_validation/relative
```

The validator dispatches from embedded HDF5 provenance and cross-checks the
supplied YAML. It checks Gaussian coordinates, per-axis coverage, Maxwell
magnitude, isotropy, spacecraft/MC independence diagnostics, matrix invariants,
and saved conventions. Small ensembles or selections may yield
`INSUFFICIENT_DATA`; statistical consistency does not prove independence.
The relative model's `alignment_distributions_1.png/.pdf` contains its three
coordinates and magnitude, labeled with the FC reference. Legacy figures keep
velocity on `_1` and magnetic alignment on `_2`.

Phase A validates saved primitives, not full measurement application. Exact
velocity invariance, reuse across timestamps and DC/AC application order are
covered by synthetic unit tests with known sampled inputs.

### Alternative fractional and total-angle requirements

The following are alternative `errors` sections for an existing generation
recipe; retain its other error sections and spacecraft, field, time, and output
settings. No existing generation recipe is changed automatically.

Legacy conventions remain the defaults:

```yaml
errors:
  model: observatory
  density:
    enabled: true
    model: multiplicative_bias_plus_ou
    requirement_fraction: 0.06
    sigma_level: 2
    bias_variance_fraction: 0.5
    ou_variance_fraction: 0.5
    ou_correlation_time_s: 300.0
  velocity_speed:
    enabled: true
    model: multiplicative_bias_plus_ou
    requirement_fraction: 0.03
    sigma_level: 2
    bias_variance_fraction: 0.5
    ou_variance_fraction: 0.5
    ou_correlation_time_s: 300.0
  velocity_angle:
    enabled: true
    model: transverse_ou_rotation
    requirement_deg: 0.25
    sigma_level: 2
    ou_correlation_time_s: 300.0
```

Select symmetric fractional errors and a total-angle confidence radius explicitly:

```yaml
errors:
  model: observatory
  density:
    enabled: true
    model: gaussian_fractional_bias_plus_ou
    requirement_fraction: 0.06
    sigma_level: 2
    bias_variance_fraction: 0.5
    ou_variance_fraction: 0.5
    ou_correlation_time_s: 300.0
  velocity_speed:
    enabled: true
    model: gaussian_fractional_bias_plus_ou
    requirement_fraction: 0.03
    sigma_level: 2
    bias_variance_fraction: 0.5
    ou_variance_fraction: 0.5
    ou_correlation_time_s: 300.0
  velocity_angle:
    enabled: true
    model: transverse_ou_rotation_total_angle
    requirement_deg: 0.25
    sigma_level: 2
    ou_correlation_time_s: 300.0
```

For `gaussian_fractional_bias_plus_ou`, `epsilon=b+u` and the measurement is
`X_obs=X_sampled*(1+epsilon)`. The constant bias and stationary OU term are
independent, with total sigma `requirement_fraction/sigma_level`. Their variance
fractions are normalized by their sum. OU initialization is stationary, and its
exact transition uses the actual time interval and configured correlation time.
Velocity speed uses the same positive scalar multiplier, preserving direction
and leaving zero vectors zero.

Thus the fractional marginal error is a zero-mean Gaussian, with approximately
95.45% coverage inside ±6% for density and ±3% for speed in the example. The
legacy model instead applies `exp(b+u)` and has a positive fractional mean.
Both retain independent draws per MC realization and spacecraft.

An unbounded Gaussian can produce `1+epsilon<=0`. The new model raises
`ValueError` with the component and first offending MC/time/spacecraft index
for nonpositive or nonfinite factors, even if the sampled quantity is zero.
It never clips, resamples, or substitutes a lognormal multiplier. Requirements
that make such events appreciably probable can cause runs to abort; accepting
only successful runs can also select the saved ensemble. There is no truncated
Gaussian interpretation.

For `transverse_ou_rotation_total_angle`, the total latent transverse angle is
Rayleigh with component sigma
`requirement_deg/sqrt(-2*log(1-P_k))`, where `P_k=erf(sigma_level/sqrt(2))`.
Two independent OU coordinates use this sigma, then the existing isotropic
transverse rotation preserves vector magnitude. The 0.25°/2σ example has
component sigma approximately **0.100564°**, total RMS **0.142219°**, and 95.45%
latent radial quantile **0.25°**. Both sigma levels 2 and 3 and finite
nonnegative requirement values, including zero, are supported dynamically.
The legacy model still uses 0.125° per component for 0.25°/2σ, whose total
0.25° coverage is about 86.47%.

The confidence-radius conversion concerns the latent transverse rotation.
Principal vector deflection is `acos(cos(theta))` and wraps above pi radians.
At these small requirements the wrapping probability is negligible; for larger
requirements the validator reports `P(theta>pi)=exp(-pi^2/(2*sigma_perp^2))`.
The model does not clamp angles or reinterpret the radius at large values.
These velocity-angle options do not change the selected alignment model.
Independent B/V alignment remains the default; the
[relative FC–MAG model](#relative-fcmag-alignment) is a separate explicit option.

#### Saved interpretations and compatibility

Normalized `observatory_error_config` records both the selected model and:

- `density.error_interpretation`, `velocity_speed.error_interpretation`:
  `log_multiplier` or `fractional_error`.
- `velocity_angle.requirement_interpretation`: `per_transverse_component` or
  `total_angle_confidence_radius`.

These values are inferred from the model; explicitly conflicting values and
unsupported model identifiers in these three sections are rejected. The
configured `requirement_deg` remains the engineering radius, not the derived
component sigma.

| Dataset under `/errors` | Meaning | Shape |
| --- | --- | --- |
| `density_log_error`, `velocity_speed_log_error` | Legacy log multiplier `b+u` | `(mc,time,spacecraft)` |
| `density_fractional_error`, `velocity_speed_fractional_error` | New fractional error `epsilon=b+u` | `(mc,time,spacecraft)` |
| `density_bias`, `velocity_speed_bias` | Bias in the selected log/fractional convention | `(mc,spacecraft)` |
| `velocity_angle_alpha`, `velocity_angle_beta` | Transverse coordinates under the selected angular convention | `(mc,time,spacecraft)` |

Only the selected density/speed latent dataset is written; fractional values
never appear under a `*_log_error` name. Disabled sections write zeros under
the selected names. New products attach dimensionless `units: '1'`, model and
interpretation attributes to density/speed latents and biases; angular
coordinates retain `units: rad` and identify the model and angular requirement
interpretation. Existing files without these attributes remain readable.
Legacy numerical operations and random draw order are preserved, including
subsequent magnetic and alignment draws. Selecting the new options does not
rewrite existing products.
