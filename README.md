# HelioSwarm Synthetic Observatory

`hs-observatory` generates controlled HelioSwarm-like multi-spacecraft observations and analyzes the resulting products. It is deliberately a small foundation: scientific field, instrument-error, and analysis models can grow independently without changing the storage contract.

## Install and run

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[dev]'
hsobs-generate configs/example_generate.yaml
hsobs-analyze configs/example_analyze.yaml
```

The example requests a CDF directory but enables `fallback_synthetic`, so it runs without mission files. Generation writes `outputs/synthetic_observations.h5` plus truth and geometry PNGs; analysis writes `outputs/diagnostics.json`.

## Two-stage workflow

Generation loads a spacecraft formation, evaluates a plasma field along every trajectory at each instrument's native cadence, creates a leading-dimension Monte Carlo ensemble using variable-specific error models, and writes truth plus observations to HDF5. Analysis is a separate, registry-selected step that reads that file. The built-in `diagnostics` method reports observed and residual statistics and formation-separation summaries.

The CDF adapter accepts one `.cdf` file or a directory of files. It follows the conventions found in the sibling `sparse-field-reconstruction` project: variables named `Epoch`, `Position`, and `Spacecraft_Label`; records are concatenated, sorted, filtered for fill values, and selected nearest to `configuration_time`. This first version treats the selected record as a snapshot and may advect it with a common `velocity_km_s`. The example includes an illustrative common translation so its path projection is visible. Full ephemeris interpolation is a future extension.

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
/truth/v                  (n_time_v, n_spacecraft, 3)
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

Configure native instrument cadences independently:

```yaml
time:
  t_start_s: 0.0
  t_stop_s: 300.0
  cadence_s:
    n: 0.125
    v: 0.125
    B: 0.03125
```

A scalar `cadence_s` remains supported and applies the same cadence to all quantities.

## Coordinate and measurement errors

Time and position uncertainties are sampled once per Monte Carlo realization and spacecraft, then held constant over that spacecraft's entire time series. The same offsets apply to `n`, `v`, and `B`. Each observation is evaluated from the field at

```text
t_sample = t_nominal + time_offset
r_sample = r_nominal + position_offset
```

before its variable-specific measurement error is added. The HDF5 `/time/*` and `/spacecraft/r/*` datasets intentionally retain the nominal coordinates. Analysis therefore calculates temporal and spatial differences using the assumed nominal coordinates, while the hidden realized offsets are available under `/errors` for validation.

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

## Default visualizations

Generation creates two nominal-truth PNGs by default:

- A seven-panel truth figure, with density and the three velocity components in the left column, the legend in the upper-right, and matching velocity/magnetic components in the remaining rows.
- A geometry figure with three hub-relative formation projections and an X-Z projection of the spacecraft paths.

When any time, position, or measurement error model is enabled, generation also creates two overlay figures for one selected Monte Carlo realization:

- Observed `n`, `v`, and `B` in spacecraft colors, with each spacecraft's nominal truth in grayscale underneath.
- Observed relative positions and absolute X-Z paths in spacecraft colors, with nominal positions and paths in grayscale underneath.

The overlay uses `mc_index: 0` by default rather than rendering every member of a potentially large ensemble. Change `mc_index` to inspect another realization, or set `observed_enabled: false` to suppress only the overlay pair.

The HelioSwarm palette is hardcoded by spacecraft name to match the reference plots in `sparse-field-reconstruction`: N1 through N8 retain their established colors and H is pink. Plot paths default to `<hdf5_stem>_truth.png` and `<hdf5_stem>_geometry.png` beside the HDF5 file. They can be configured or disabled:

```yaml
visualization:
  enabled: true
  observed_enabled: true
  mc_index: 0
  truth_path: outputs/synthetic_observations_truth.png
  geometry_path: outputs/synthetic_observations_geometry.png
  observed_path: outputs/synthetic_observations_observed_mc000.png
  observed_geometry_path: outputs/synthetic_observations_geometry_observed_mc000.png
  dpi: 150
```

When paths are omitted, they are derived beside the HDF5 output as `_truth.png`, `_geometry.png`, `_observed_mcNNN.png`, and `_geometry_observed_mcNNN.png`. Observed time series are plotted against nominal time, consistent with the coordinate assumptions used by downstream analysis.

## Extension points

To add a field model, subclass `FieldModel` and implement `evaluate_quantity(quantity, time, position)`, then add construction by name alongside `field_from_config`. The inherited `evaluate(time, position)` convenience method evaluates all quantities on one shared grid. Field models contain no noise or file-I/O logic.

### Malara-style intermittent turbulence evaluator

`fields/malara.py` provides a low-level isotropic evaluator inspired by Malara et al., [“Fast algorithm for a three-dimensional synthetic model of intermittent turbulence”](https://doi.org/10.1103/PhysRevE.94.053109). `MalaraTurbulence.evaluate_vector(positions)` evaluates a deterministic three-component turbulent field directly at arbitrary points without constructing a spatial grid.

The current evaluator is time-independent, periodic in all three directions, intermittent through a deterministic p-model cascade, and solenoidal by construction from the curl of compact vector-potential eddies. At each dyadic scale it evaluates only the neighboring eddies whose overlapping supports contain the point. Anisotropic and critical-balance variants are future extensions.

`field.model: malara_turbulence` connects it to generated observations. Density remains constant; magnetic and optional velocity fluctuations are scaled copies of a Malara realization. With no realization offset, `v` and `B` share the same vector field. Set `realization_offset` to select an independent realization. The adapter defaults the low-level vector-potential amplitude to the characteristic domain length, making output-field amplitudes meaningful in their physical units; an explicit `turbulence.amplitude` overrides this normalization.

```python
from hs_observatory.fields.malara import MalaraConfig, MalaraTurbulence

model = MalaraTurbulence(MalaraConfig(n_scales=6, realization=1))
u = model.evaluate_vector(positions)  # positions and u both end in dimension 3
```

Run the integrated example with:

```bash
hsobs-generate configs/example_generate_malara.yaml
```

```yaml
field:
  model: malara_turbulence
  domain:
    L: [100000.0, 100000.0, 100000.0]
    units: km
    origin: [0.0, 0.0, 0.0]
  turbulence:
    geometry: isotropic
    n_scales: 8
    injection_scale_index: 2
    intermittency_p: 0.7
    base_seed: 12345
    realization: 1
  output_fields:
    n: {background: 5.0, fluctuation: none}
    v:
      background: [400.0, 0.0, 0.0]
      fluctuation: none
      amplitude: 0.0
    B:
      background: [5.0, 0.0, 0.0]
      fluctuation: malara
      amplitude: 0.5
```

Compressive density turbulence is deliberately unsupported for now. The explicit `n.fluctuation` setting leaves a clean extension point for a future density model.

To add a measurement-error model, subclass `ErrorModel` and implement `apply(truth, rng, n_mc, time=...)`. Return both observed values and realized errors with shape `(n_mc, *truth.shape)`, then add its YAML name to `error_model_from_config`. Models that do not depend on cadence may ignore `time`. Coordinate uncertainties implement `CoordinateOffsetModel` and are constructed separately because they alter where the field is sampled. These interfaces are intended to accommodate correlated noise, dropouts, cadence averaging, spin tones, and non-Gaussian models.

To add an analysis method, subclass `AnalysisMethod`, give it a `name`, decorate it with `@register_analysis`, and ensure its module is loaded by the registry. JSON is the default output; binary analyses can override `write_result`, as `static_decorrelation` does for HDF5. `StructureFunctionAnalysis` demonstrates a minimal pairwise scaffold and contains the TODO boundary for scientifically complete higher-order science.

## Static spatial decorrelation analysis

`static_decorrelation` estimates simultaneous pairwise correlation as a function of spacecraft separation and computes the matching second-order pair-difference structure function. It treats every time index as an independent spatial snapshot. This first method intentionally includes no time lags, temporal evolution, Taylor hypothesis, flow advection, or wave propagation.

Run it after generating observations:

```bash
hsobs-generate configs/example_generate.yaml
hsobs-analyze configs/example_static_decorrelation.yaml
```

The analysis supports scalar density correlation, vector dot-product correlation, optional vector-component correlations, normalized or raw products, and linear or logarithmic separation bins. `data_group: observed` selects one `mc_index`; `data_group: truth` ignores it. `mc_index: all` is reserved for future ensemble statistics and currently raises `NotImplementedError`.

The default `per_time` mean subtracts the constellation mean independently at each snapshot. `global` subtracts one scalar or vector mean over all selected times and spacecraft. With the current native-cadence HDF5 schema, each variable uses its matching `/spacecraft/r/{variable}` positions. All selected variables share common bin edges, while `/counts/{variable}` records cadence-dependent sample counts.

The HDF5 output contains bin edges, lower and upper edges, centers, per-variable counts, correlation means/standard deviations/standard errors, and corresponding second-order structure-function statistics. Empty bins have count zero and NaN statistics. A default diagnostic PNG is written beside the HDF5 file as `<output_stem>_diagnostic.png`; it shows correlation and standard-error bands in the left column and second-order structure functions in the right column.

```yaml
analysis:
  method: static_decorrelation

decorrelation:
  variables: [n, v, B]
  data_group: observed
  mc_index: 0
  mean: {method: per_time}
  separation:
    units: km
    binning: log
    n_bins: 30
    r_min: null
    r_max: null
  correlation:
    vector_mode: dot
    component_correlations: true
    normalize: true
  structure_function: {compute: true, order: 2}
  visualization:
    enabled: true
    path: outputs/static_decorrelation_diagnostic.png
    dpi: 150
```

## Development

```bash
pytest
ruff check src tests
```

Tests cover field shapes, seeded error reproducibility, OU stationary statistics and vector geometry, coordinate-offset sampling, HDF5 round trips, synthetic CDF fallback, visualization, generation, diagnostics, and static spatial decorrelation.
