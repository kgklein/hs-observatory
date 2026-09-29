# Observation data and derived products

## Array and HDF5 layout

```text
/metadata                 group attributes: YAML, UTC creation time, seed,
                          model names, source details, schema version
/time/n                   (n_time_n,)
/time/v                   (n_time_v,)
/time/B                   (n_time_B,)
/spacecraft/id            (n_spacecraft,)
/spacecraft/r/n           (n_spacecraft, n_time_n, 3)
/spacecraft/r/v           (n_spacecraft, n_time_v, 3)
/spacecraft/r/B           (n_spacecraft, n_time_B, 3)
/truth/n                  (n_time_n, n_spacecraft)
/truth/v                  (n_time_v, n_spacecraft, 3), plasma frame
/truth/v_spacecraft_frame (n_time_v, n_spacecraft, 3)
/truth/B                  (n_time_B, n_spacecraft, 3)
/observed/n               (n_mc, n_time_n, n_spacecraft)
/observed/v               (n_mc, n_time_v, n_spacecraft, 3)
/observed/B               (n_mc, n_time_B, n_spacecraft, 3)
/errors/n                 same shape as observed/n (optional)
/errors/v                 same shape as observed/v (optional)
/errors/B                 same shape as observed/B (optional)
/errors/time_offset       (n_mc, n_spacecraft), seconds (optional)
/errors/position_offset   (n_mc, n_spacecraft, 3), km (optional)
```

The Monte Carlo dimension is present even when `n_mc: 1`. Numerical datasets are chunked and gzip-compressed. Schema version 2 stores a time-aligned position grid for every quantity. Positions are currently interpreted as km, time as seconds relative to the generated interval, density as cm^-3, velocity as km/s, and magnetic field as nT; a richer unit system can be added later without changing these dimensions.

## Velocity-frame convention

Field models always generate physical plasma-frame velocity truth. Spacecraft velocity controls the sampling path and is also used to construct

```text
v_spacecraft_frame = v_plasma - V_spacecraft.
```

By default, the velocity measurement-error model is applied in the spacecraft frame. `/truth/v` remains plasma-frame truth for backward compatibility, `/truth/v_spacecraft_frame` stores the transformed truth, and `/observed/v` is stored in the selected measurement frame. Density and magnetic field are not transformed.

```yaml
spacecraft:
  velocity_km_s: [100.0, 0.0, 250.0]

measurement:
  velocity_frame: spacecraft  # spacecraft (default) or plasma
```

A constant `velocity_km_s` is broadcast over every spacecraft and time. Trajectory readers may instead provide `(spacecraft, time, 3)` velocity arrays. Selecting the spacecraft frame without an available spacecraft velocity raises an error. The HDF5 metadata attributes `velocity_truth_frame`, `velocity_observed_frame`, and `velocity_error_applied_frame` record the convention used.

## Coordinates and time grids

Positions have spacecraft-first shape `(spacecraft, time, 3)`, while truth fields
are time-first and observed fields add a leading Monte Carlo axis. `/time/n`,
`/time/v`, and `/time/B` use their native cadences. Numerical arrays are gzip
compressed and chunked. Most coordinate/truth datasets do not carry dataset-level
unit attributes; use the documented native units and provenance rather than
assuming a units-aware reader.

## Coordinate use by analysis

Generation stores nominal coordinates, but samples fields at
`t_nominal + time_offset` and at the trajectory evaluated at that shifted time
plus `position_offset`. Offsets are constant per spacecraft/MC realization.
The generator interpolates trajectories and uses endpoint extrapolation as
described in [errors](errors.md).

| Method | Current coordinate/offset behavior |
| --- | --- |
| Shock timing | Fits observed channels and nominal trajectories; truth is a separate optional validation layer. |
| Timing calibration | Nominal geometry; measured-field correction is resampled without extrapolation; synthetic truth remains required by the pipeline. |
| Magnetic gradient | Reads stored timing/position offsets for measured estimates and synchronization. |
| Structure function and static increment PDFs | Can add stored position offsets when group-specific coordinates are absent; the nominal coordinates do not encode clock-induced trajectory displacements. |
| Static decorrelation | Uses the variable's native nominal position grid. |
| FFT | Uses time grids, without spatial/Taylor conversion. |

These distinctions document implementation, not an endorsement of a shared
scientific coordinate policy. See the [review checklist](../developer/scientific-review.md).

## Provenance and optional errors

`/metadata` contains serialized generation YAML, creation time, seed, field/error
models, spacecraft-source details and schema version. Frame and plotting choices
are recorded separately. `/errors/n`, `/errors/v`, and `/errors/B` are measurement
realizations, not the entire difference from nominal truth. Coordinate offsets
and [observatory auxiliary errors](errors.md#observatory-requirement-details)
have their own datasets. Disabling error storage can make analyses that require
those offsets unavailable.

## Derived-product index

Each method reference owns its detailed output schema, masks and units:

- [Statistics](statistical-methods.md): diagnostics JSON, Elsasser/PDF/decorrelation/FFT HDF5, structure-function JSON.
- [Magnetic gradients](magnetic-gradients.md): gradient and correlation HDF5; comparison JSON/CSV.
- [Reconstruction](reconstruction.md): RBF/Timesync HDF5, grids and validation products.
- [Shock timing](shock-timing.md): detections, inversions, statuses and MC summaries.
- [Timing calibration](timing-calibration.md): corrections, objectives, covariance, projections and influence.

NaNs, validity masks and completion flags are method-specific. For timing
calibration, `metadata.complete = false` marks an incomplete streamed product.
Do not treat NaN statistics or failed fits as zero measurements.
