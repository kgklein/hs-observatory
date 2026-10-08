# Configuration and command reference

The checked-in YAML files remain the executable examples; this page owns shared
configuration rules. Specialized parameter definitions live with their field/error/method.

## YAML and paths

Generation requires mappings named `spacecraft`, `time`, `field`, `errors`, and
`output`. `time` requires `t_start_s`, `t_stop_s`, and positive `cadence_s` (scalar
or a mapping containing n/v/B). `field.model` and `output.path` are required;
`errors.n_mc` defaults to 1 and must be positive. Optional `measurement` and
`visualization` mappings select frames and figures.

Analysis requires `input.path`, `analysis.method`, and `output.path`. General
paths are resolved by their consumers from the working directory; run examples
from the repository root. The loader does not globally rewrite paths relative to
the YAML. A documented exception is the timing-noise spectrum path relative to
the analysis YAML. Check model-specific expansion of home-directory paths.

## Analysis dispatch

`analysis.options` is copied first; the method-specific top-level configuration
mapping then overrides matching keys. The registry selects the method class.

| Selector | Top-level configuration section |
| --- | --- |
| `diagnostics` | No dedicated section; `analysis.options` |
| `elsasser` | `elsasser` |
| `elsasser_increment_pdf_static` | `elsasser_increment_pdf_static` |
| `static_decorrelation` | `decorrelation` |
| `timeseries_fft` | `fft` |
| `structure_function` | `structure_function` |
| `magnetic_gradient` | `magnetic_gradient` |
| `magnetic_gradient_correlation` | `magnetic_gradient_correlation` |
| `rbf_reconstruction` | `rbf_reconstruction` |
| `timesync` | `timesync` |
| `shock_timing` | `shock_timing` |
| `timing-calibrate` | `timing_calibrate` |

## Generation selectors and compatibility

`field.model` selects `sine_sum`, `malara_turbulence`, `simulation_sampled`,
`plume_single_mode`, or `shock`; use the [field reference](fields.md).
`spacecraft.source` selects synthetic geometry or CDF loading, with an optional
fallback. CDF variables and snapshot/advection behavior are described in
[generation](../user-guide/generation.md).

The [error reference](errors.md) defines coordinate, Gaussian, OU and observatory
selectors. Preserve scalar cadence support, Malara density `none` as a legacy
alias for `constant`, and the observatory `requirement_3sigma_*` compatibility
keys. These do not make reserved/future options executable.

## Installed commands

All commands below are declared in `pyproject.toml`. Use `COMMAND --help` for
exact arguments; uppercase arguments here are placeholders.

| Command | Input and principal controls |
| --- | --- |
| `hsobs-generate` | Generation YAML; `--truth-velocity-frame`, repeatable `--3d-connection` |
| `hsobs-analyze` | Analysis YAML |
| `hsobs-analyze-spatial-structure-function` | Direct-spatial analysis YAML; `--field-config`, `--p`, `--n-pairs`, `--seed`, separation/backend controls, `--output-prefix` |
| `hsobs-plot-spatial-structure-function` | Saved direct-spatial HDF5; fit windows, local-slope method, cascade context and comparison controls; no field evaluation |
| `hsobs-plot-figure2` | `--malara`, `--simulation`, `--output`; units/frame/member/time controls |
| `hsobs-plot-structure-function` | Saved JSON; `--p-value`, `--quantity`, `--output` |
| `hsobs-plot-structure-function-overlay` | Plotting YAML; `--local-slopes`, `--local-slope-method three-point\|adjacent`, `--show-cascade-scales`, `--output-prefix`; [local diagnostic details](statistical-methods.md#local-logarithmic-scaling-diagnostic) |
| `hsobs-plot-increment-pdf` | Repeatable `--result P=HDF5`, `--output`, `--lags`, `--layout`, `--dpi` |
| `hsobs-plot-increment-pdf-wasserstein` | Plotting YAML; `--panels-a-c` |
| `hsobs-plot-magnetic-gradient` | Saved HDF5; `--output`, `--layout`, `--row4`, `--ratio-scale`, time bounds, DPI |
| `hsobs-plot-magnetic-gradient-wasserstein` | Comparison YAML |
| `hsobs-plot-magnetic-gradient-combined` | Comparison YAML; output/configuration/time/DPI overrides |
| `hsobs-plot-magnetic-gradient-correlation` | Correlation YAML |
| `hsobs-plot-timing-calibrate` | Saved HDF5; output/member/component/time/DPI controls, `--compare-products` |

CLI overrides change the documented aspect of a run; a truth-plot frame override
does not change stored plasma truth or the velocity measurement frame.
[Visualization](../user-guide/visualization.md) links each plotter to interpretation.
