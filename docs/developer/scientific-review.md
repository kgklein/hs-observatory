# Scientific and release review checklist

This checklist distinguishes observed implementation from unresolved scientific
intent. No algorithm, numerical constant, scientific configuration semantics or
scientific source file was changed by the documentation migration.

## Scientific decisions

- [ ] **PLUME background geometry.** The preserved README snippet has B along x; the text describes PLUME's z-aligned background and the current YAML uses z. The constructor accepts the supplied background. Confirm the intended physical example before changing its values.
- [ ] **PLUME `B_rms_nT`.** `PlumeSingleModeField` scales by the norm of the complex magnetic eigenvector. Confirm terminology versus RMS of the real oscillating field; no new factor has been inferred.
- [ ] **Simulation velocity normalization.** Preserve 2438 km/s per code velocity unit, but verify a derivation explicitly incorporating code mass density/ion mass and the provider's units. Neither the provider data nor the physical scaling was revalidated here.
- [ ] **Pearson wording.** The original amplitude-bias statement is retained. Confirm the intended qualification for negative multiplicative scaling/sign changes and interpretation with NRMSE/regression diagnostics.
- [ ] **Instrument-noise interpretation.** Generation uses requirement-level total-vector ASD anchors; timing weights use a different provisional per-component digitization. Obtain MAG-team confirmation and durable source attribution. Do not merge these spectra.
- [ ] **Coordinate policy by method.** Generation follows clock-shifted trajectories. Some analyses use injected position/timing offsets, others nominal positions. Confirm scientific intent separately from the now-explicit implementation description.
- [ ] **Timing-calibration scope.** Keep research status, synthetic truth requirements, ideal-divergence weighting/optional targets, covariance limitations and zero-error bias prominent. Instrument-accessible weights do not make the entire pipeline flight-ready.
- [ ] **Historical numerical evidence.** Supply commit, input checksums and artifact provenance for divergence and timing tables before claiming public reproducibility. No historical numerical experiment was rerun for this migration.
- [ ] **Citations.** Complete the exact Mallet (2015), Broeren & Klein (2024), PLUME ordering and instrument-requirement references. Preserve the existing Malara DOI and palette/neighboring-workflow attribution; do not guess missing bibliographic details.

## Release decisions

- [ ] **Default geometry plotting dependency.** A fresh base-only install failed with `ModuleNotFoundError: cmcrameri` in `plot_geometry_summary`. The README now installs the existing `.[analysis]` extra. Decide separately whether plotting needs a fallback or a packaging change; neither is changed here.

- [ ] Select and track approved timing generation/analysis/spectrum configurations; see the [manifest](../examples/index.md#timing-research).
- [ ] Restore or explicitly retire the absent `analysis/timing_noise_experiment.py` recipe while retaining its historical report.
- [ ] Choose durable distribution/licensing for external mission, PLUME and simulation inputs or publish precise acquisition instructions.
- [ ] Resolve the existing CI/SciPy mismatch: shock-timing tests require SciPy but CI installs only `.[dev]`. Packaging and CI are unchanged here.
- [ ] Confirm site/repository access and citation release metadata before public launch. This migration does not deploy.

## Remaining duplication outside scope

Root [DESIGN.md](https://github.com/kgklein/hs-observatory/blob/main/DESIGN.md)
remains authoritative design rationale but has historical future-feature wording
and duplicates some current array/field/analysis descriptions. Root
[CONTRIBUTING.md](https://github.com/kgklein/hs-observatory/blob/main/CONTRIBUTING.md)
remains contribution-policy authority. Both are unchanged.

[Visualization Export](https://github.com/kgklein/hs-observatory/blob/main/visualization/README.md)
and [ParaView Notes](https://github.com/kgklein/hs-observatory/blob/main/visualization/paraview_notes.md)
remain authoritative for standalone helpers, validation insertion, Figure 2,
color/background settings and export/display instructions. Some colormap controls
also appear in reconstruction reference lists. These files are unchanged; a future
consolidation should preserve all dataset aliases, fallback hierarchy and commands.
