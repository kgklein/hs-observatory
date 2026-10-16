# Research Notes

These notes preserve scientific investigations, implementation audits, and
experiment reports. Findings are conditional on the inputs and methods stated
in each report; provisional results are not established validation of the
observatory or its scientific models. Many reports refer to local, generated
products or external simulation inputs that are not distributed with the site.

## Structure functions, sampling geometry, convergence, and intermittency

Start with [Intermittency recovery](intermittency-recovery.md) for the investigation's
status and interpretation limits. The [statistical-methods reference](../reference/statistical-methods.md)
describes the implemented calculations.

- [Pair-sampling audit and proposed sampling design](sf/structure_function_pair_sampling_audit.md)
- [Geometry sampling (Stage 1)](sf/structure_function_geometry_stage1.md)
- [Truth-only sampling convergence (Stage 2)](sf/structure_function_stage2_convergence.md)
- [Refined publication workflow (Stage 3)](sf/structure_function_stage3_publication.md)
- [Uniform logarithmic binning test](sf/structure_function_uniform_binning_test.md)
- [Observation-duration comparison: p=0.9 and p=0.5](sf/structure_function_duration_comparison.md)
- Duration reports: [case 500](sf/structure_function_duration_convergence_500.md),
  [case 502](sf/structure_function_duration_convergence.md),
  [case 500a](sf/structure_function_duration_convergence_500a.md), and
  [case 502a](sf/structure_function_duration_convergence_502a.md)
- [Alternate propagation direction: 500a and 502a](sf/structure_function_duration_angle_a.md)
- [Study A: high-order moment concentration](sf/structure_function_moment_convergence.md)
- [Malara field-validation capability audit](sf/malara_field_validation_audit.md)
- [Fixed-realization direct-3D convergence campaign](sf/malara_direct3d_convergence.md)
- [Raw Malara field along nominal HelioSwarm trajectories](sf/malara_raw_hs_structure_function.md)

## Magnetic gradients and divergence

- [Magnetic-gradient divergence audit](magnetic-gradient-divergence-audit.md):
  historical findings and limits of finite-configuration affine estimates.
- [Magnetic-gradient reference](../reference/magnetic-gradients.md): current methods
  and product conventions.

## Timing calibration and timing investigations

- [Timing calibration](../reference/timing-calibration.md): current method reference.
- [Timing investigations](timing-investigations.md): affine and quadratic projection
  experiments and the provisional instrument-noise comparison. The noise experiment's
  reproduction status and input availability need review before reusing its results.

## Compatibility pages

Existing section anchors are retained in the moved
[gradient audit](gradient/magnetic_gradient_divergence_audit.md),
[timing calibration](timing/timing_calibration.md),
[affine investigation](timing/timing_affine_investigation.md), and
[quadratic investigation](timing/timing_quadratic_investigation.md) compatibility pages.
Their links lead to the maintained content above.
