# Observatory Error Model

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
of about 0.665 km. Fields are sampled at the perturbed time and spacecraft
position; nominal coordinates remain in the HDF5 coordinate datasets.

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
