# Generating Synthetic Observations

`hsobs-generate` coordinates a YAML-configured observation experiment. The
starting example is `configs/example_generate.yaml`; the repository README
describes additional field and error configurations.

1. **Configuration and sampling grids.** Load and validate the YAML, construct
   native time grids for density `n`, velocity `v`, and magnetic field `B`, and
   load spacecraft trajectories or the configured synthetic fallback.
2. **Field model.** Construct a sine-sum, Malara turbulence, PLUME single-mode,
   or static simulation-sampling field from the field configuration.
3. **Nominal truth.** Evaluate the field on nominal spacecraft coordinates.
4. **Coordinate perturbations and sampling.** Draw clock and position offsets
   per spacecraft and Monte Carlo realization. Evaluate trajectories at shifted
   times, add position offsets, and sample the field at those coordinates.
5. **Measurement frame and errors.** Field velocity is in the plasma frame.
   Generation converts it to the configured measurement frame and applies
   measurement errors to the sampled ensemble.
6. **Output.** Write an HDF5 observation product, including nominal coordinates,
   truth, observations, configuration metadata, and configured error products.
   Optional visualization settings produce time-series and geometry figures.

When visualization is enabled, `geometry_overviews/` contains three time-tagged
geometry summaries, each saved as PDF and PNG:

- `*_geometry_summary_*`: the original eight-panel summary.
- `*_geometry_polyhedra_*`: a two-row, three-column view with coordinate
  projections, orbit context, the existing size/shape summary, and elongation
  versus planarity colored by logarithmic size `L` (km). The `L` colorbar is
  independent of the spacecraft-count colorbar.
- `*_geometry_formation_*`: a hub-relative 3D formation spanning the left half,
  with HS spacecraft colors and Hub-to-Node spokes. The lambda and orbit panels
  remain on the right; there is no orbit inset in the 3D panel. A faint convex
  hull is added when SciPy is available and the formation is nondegenerate.

For Python plotting, `hs_observatory.visualization.plot_geometry_summary` accepts
`variant="summary"` (the default), `"polyhedra"`, or `"formation"`.

Coordinate errors affect where the field is sampled; measurement errors act on
the resulting values. The nominal coordinates remain available for downstream
analysis. Observed arrays retain a Monte Carlo dimension even for one realization.

Generation and analysis are separate stages: `hsobs-analyze` reads the saved
observation product and runs a selected analysis. See the
[Quick Start](../getting-started/quick-start.md),
[Analysis Methods](../analysis_methods.md), and
[Observatory Error Model](../observatory_error_model.md).

For detailed configurations and storage conventions, consult the
[repository README](https://github.com/kgklein/hs-observatory/blob/main/README.md).
