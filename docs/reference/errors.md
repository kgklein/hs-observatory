# Coordinate and measurement errors

| Summary | Details |
| --- | --- |
| Purpose | Control sampling uncertainty and instrument errors separately. |
| Required inputs | Generation YAML and field samples at each native cadence. |
| Entry points | `hsobs-generate CONFIG`; `errors` mapping. |
| Outputs | Observed ensemble, optional realized errors, and metadata. |
| Limitations | Requirements are model assumptions; generation noise and timing-fit noise are different models. |

See the [example manifest](../examples/index.md) for prerequisites and verification status,
and [visualization guide](../user-guide/visualization.md) for inspection and plotting.
## Coordinate and measurement errors

Time and position uncertainties are sampled once per Monte Carlo realization and spacecraft, then held constant over that spacecraft's entire time series. The same offsets apply to `n`, `v`, and `B`. Each observation is evaluated from the field at

```text
t_sample = t_nominal + time_offset
r_sample = r_trajectory(t_nominal + time_offset) + position_offset
```

before its variable-specific measurement error is added. For velocity, the selected plasma- or spacecraft-frame transformation occurs before applying that measurement error. The HDF5 `/time/*` and `/spacecraft/r/*` datasets intentionally retain the nominal coordinates. Stored coordinates remain nominal. Downstream methods differ: some use nominal coordinates, while static increment PDFs and structure functions can use stored position offsets, and magnetic gradients read timing and position offsets. See [coordinate use by analysis](data-products.md#coordinate-use-by-analysis). These are implementation behaviors, not a statement of scientific intent.

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

Set either coordinate model to `none` to disable it. Scalar `sigma_km` applies the same width to X, Y, and Z; a three-component value sets their widths independently. `/errors/n`, `/errors/v`, and `/errors/B` contain the additive measurement-noise realizations; coordinate-induced differences are represented by the stored offsets rather than folded into those arrays.

### Observatory requirement model

Use `errors.model: observatory` for the requirement-motivated HelioSwarm-like
model. It samples constant timing and position offsets, multiplicative
density/speed bias plus OU noise, transverse velocity angular precision,
magnetic DC offsets, merged-product colored magnetic AC noise, and independent
constant B/V alignment rotations. The updated plume example uses this model:

Each observatory error section accepts `sigma_level: 2` or `sigma_level: 3`
(the default) alongside a neutral, unit-specific `requirement_*` value. Levels
may be mixed between sections. Existing `requirement_3sigma_*` configuration
keys remain supported as 3-sigma compatibility aliases.

```bash
hsobs-generate configs/example_generate_plume_single_mode.yaml
```

See `docs/observatory_error_model.md` for assumptions, units, default
requirements, and the additional `/errors` datasets written for each Monte
Carlo realization.

### Ornstein–Uhlenbeck measurement errors

Use `model: ornstein_uhlenbeck` for time-correlated measurement noise. OU widths are stationary standard deviations, and correlation times are in seconds. The implementation uses the exact discrete transition for each interval, so the same physical parameters work correctly with different or irregular instrument cadences.

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
      direction_sigma_deg: 0.5
      direction_tau_s: 0.25
```

Density error is additive: `n_observed = n_sampled + epsilon`. Vector magnitude is changed by `max(0, 1 + epsilon)`, preventing an unphysical negative magnitude. Directional noise is an isotropic tangent-plane OU process; `direction_sigma_deg` specifies the stationary RMS total angular deviation. `direction_tau_s` defaults to `magnitude_tau_s`, and `direction_sigma_deg` defaults to zero. Zero vectors remain zero because their direction is undefined.

The complete OU example can be run with:

```bash
hsobs-generate configs/example_generate_ou.yaml
```

## Observatory requirement details

The `errors.model: observatory` generation path implements a HelioSwarm-like
requirement-driven error model in package-native units: seconds, km, cm^-3,
km/s, and nT.

Each error section has a `sigma_level` of `2` or `3` (default `3`). Its
unit-specific `requirement_*` value is divided by that level to obtain the
one-sigma width. This permits mixed requirements, for example:

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

The legacy `requirement_3sigma_*` keys remain accepted as 3-sigma inputs.
They cannot be combined with their neutral replacement or with
`sigma_level: 2`.

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

Density and velocity speed use lognormal multiplicative errors with a constant
spacecraft bias plus a stationary exact-discrete Ornstein-Uhlenbeck component.
The default density and speed 3-sigma fractional requirements are 6% and 3%,
respectively, with half the variance in the bias and half in the OU process.

Velocity direction receives two transverse OU angular processes relative to the
instantaneous velocity direction. The default angular precision is 0.25 degrees
at 3-sigma. Velocity alignment is then applied as a constant per-spacecraft
rotation matrix. Magnetic field receives a constant per-axis DC offset, colored
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


## Distinct timing-calibration noise normalization

The [timing reference](timing-calibration.md#instrument-noise-divergence-normalization)
uses a different provisional spectrum to calculate fit weights. It does not
apply a filter or generate measurement noise. Do not substitute that spectrum for
the requirement-level total-vector ASD anchors above without scientific review.
