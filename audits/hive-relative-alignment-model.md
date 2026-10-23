# HIVE relative FC–MAG alignment: implementation audit and Stage 2 proposal

Date: 2026-10-22. Status: design only; no model implementation or production regeneration.

## 1. Finding and recommended design

Add the explicit opt-in model `relative_constant_rotation_B_to_V` for future Malara 8xx products. Keep `independent_constant_rotation_B_and_V` as the default. Treat FC as the orientation reference and apply one residual alignment rotation to the merged magnetic product, after magnetic DC and AC noise. Do not apply an alignment operation to velocity in the new branch.

The supplied interpretation of L2.5-MDRA-28 is the scientific input to this audit: 3° per rotation-vector coordinate at 3σ, hence

\[
\sigma_a=\frac{3^\circ}{3}=1^\circ=0.017453292519943295\ {\rm rad},\qquad
\omega_{m,s}\sim N(0,\sigma_a^2 I_3),\quad R_{m,s}=\exp([\omega_{m,s}]_\times).
\]

One independent draw belongs to each `(MC, spacecraft)` pair and has no time dependence. There are no separate FC and MAG physical error rotations in this model. Keep legacy arrays and calculations unchanged in legacy output; store the relative model under new dataset names.

The principal implementation hazard is RNG consumption: the legacy alignment consumes two normal blocks before magnetic noise. Preserve that consumption for enabled alignment to avoid changing unrelated latent errors. The principal scientific qualification is that HIVE has common Cartesian axes but no instrument attitude model. The proposed matrix is a **residual reconstruction error**, expressed in those common axes, rather than a complete physical instrument attitude matrix.

## 2. Existing implementation and evidence

The following are source observations, distinct from the proposed behavior below. Paths are relative to the repository root; line references describe the audited source and may move.

| Area | Source and observed behavior |
| --- | --- |
| Configuration | `src/hs_observatory/errors/observatory.py:19–197`: defaults, nested `errors.observatory` merge, legacy requirement aliases, and section-local sigma level. Alignment defaults to independent B/V constant rotations. The alignment model string is currently neither validated nor dispatched; changing that string alone still executes legacy behavior. |
| Rotations | `observatory.py:303–336`: Rodrigues uses a right-handed cross-product matrix. `rotate_vectors(values, R)` evaluates `values @ R.T`, equivalent to `R b` for column vectors. |
| Measurement errors | `observatory.py:642–796`: density, velocity speed, velocity angle, both alignment draws, V rotation, magnetic DC, magnetic AC, then B rotation. |
| Generation/frames | `src/hs_observatory/generate.py:293–422`: nominal truth, shifted coordinate sampling, spacecraft-velocity subtraction, shared generator, and measurement application. Metadata at approximately 479–500 records normalized errors and velocity frame. |
| Malara vectors | `src/hs_observatory/fields/malara.py:530–654`: both B and V are Cartesian background vectors plus vector fluctuations evaluated on the same position axes. No instrument basis conversion occurs. |
| Serialization | `src/hs_observatory/io/hdf5.py:20–169`: generic error-dataset writing/reading; selected unit attributes; metadata mappings serialized as JSON attributes. Generation sets schema version `"2"`. |
| Phase A | `scripts/measurement_error_validation/model.py`, `runner.py:74–185,198–259,324–357`, `statistics.py`: independent normalization, shape/unit checks, Gaussian/Maxwell tests, matrix checks, and B/V independence. Unsupported alignment models currently fail validator normalization. |
| Existing protection | `tests/test_observatory_errors.py:153–180,220–262,270–356`: magnitude tests, serialization, and immutable legacy reference arrays plus final RNG state. `tests/test_velocity_frames.py` verifies spacecraft subtraction and provenance; `tests/test_coordinate_sampling_errors.py` covers shifted sampling; `tests/test_hdf5_roundtrip.py` covers generic I/O. |

### Exact legacy behavior

For each field, the scalar sigma is `deg2rad(requirement_deg_per_axis / sigma_level)`. Enabled alignment draws V first and B second, each with `rng.normal(scale=sigma, size=(n_mc, n_spacecraft, 3))`. Those arrays generate separate matrices, each reused at every timestamp of its field's own cadence. Disabled alignment draws neither block, saves zero rotation vectors and identity matrices, and still passes vectors through the legacy rotation loops.

Legacy datasets are:

| Dataset under `/errors` | Shape | Meaning |
| --- | --- | --- |
| `velocity_alignment_rotvec` | `(mc, sc, 3)` | Active V error rotation vector, radians |
| `velocity_alignment_matrix` | `(mc, sc, 3, 3)` | Rodrigues matrix of that vector |
| `magnetic_alignment_rotvec` | `(mc, sc, 3)` | Independent active B error rotation vector, radians |
| `magnetic_alignment_matrix` | `(mc, sc, 3, 3)` | Rodrigues matrix of that vector |

Only rotvec units are currently explicitly attached for alignment. Model and requirement provenance come from `/metadata`'s JSON `observatory_error_config`, together with original `config_yaml`.

Legacy alignment rotates velocity after speed/direction errors and rotates the entire magnetic vector after adding DC/AC. In an FC-referenced comparison, its effective relative rotation is `R_V.T @ R_B`; at small angles its relative rotation vector is approximately `omega_B - omega_V`, with coordinate sigma approximately `sqrt(2)*sigma_a`. The exact composition is not simple rotation-vector subtraction. Therefore legacy independent 1° coordinate sigmas do not implement the agreed 1° relative sigma.

The inspected 601/671 and 701/771 production YAML files explicitly select the legacy alignment model; 7xx changes density/speed and velocity-angle choices but retains that alignment. This audit does not modify any production YAML or assert verification of every historical file.

## 3. Coordinate frames and full pipeline

### Common Cartesian axes versus velocity reference motion

Let C denote HIVE's common Cartesian component basis. Nominal B and plasma velocity truth are field evaluations in C. Malara B/V components, spacecraft positions, and spacecraft velocities are used directly on these axes. There is no FC attitude, MAG attitude, spacecraft body-axis transformation, or time-varying pointing quaternion in the inspected measurement path.

`measurement.velocity_frame: spacecraft` selects a Galilean subtraction, not a basis rotation. B remains in C. Sampled V has spacecraft-relative motion but still C-oriented components. The `plasma` option leaves sampled field velocity untouched. Observed B/V retain C component labels, with errors perturbing their reported values. For the new model, C is identified with the nominal FC-referenced reconstruction axes; no absolute FC attitude uncertainty is being added.

For quantity q on its own nominal time grid, define

\[
t'_{q,m,s,i}=t_{q,i}+\delta t_{m,s},\quad
x'_{q,m,s,i}=x_s(t'_{q,m,s,i})+\delta x_{m,s}.
\]

Trajectory positions are interpolated with endpoint extrapolation. Stored spacecraft velocities are interpolated with endpoint clamping. Define

\[
n_s=n_0(t'_n,x'_n),\quad b_s=B_0(t'_B,x'_B),\quad
u_s=V_0(t'_v,x'_v)-\chi u_{\rm sc}(t'_v),
\]

where `chi=1` for spacecraft-relative measurements and `0` for plasma measurements. The subscript s here denotes sampled values, not a new spatial coordinate system. Nominal `/truth/v` remains plasma truth, while `/truth/v_spacecraft_frame` stores nominal-time subtraction when spacecraft velocity exists. Coordinate-shifted sampled truth is not stored as a separate product.

### Velocity speed and direction errors

Let `f_n` and `f_v` be the configured multiplicative factors. Legacy log-error models use `exp(L)`; Gaussian fractional models use `1+epsilon`, reject nonpositive factors, and do not clip them. Disabled factors equal one.

Let `A_{alpha,beta}(u)` denote the existing transverse direction operation. After choosing the deterministic perpendicular basis `(e1,e2)` around `uhat`, set `p=alpha*e1+beta*e2`, `rho=|p|`. For nonzero u,

\[
A_{\alpha,\beta}(u)=|u|\left[\cos\rho\,\hat u+
\frac{\sin\rho}{\rho}p\right],
\]

with the continuous zero-rho limit and zero vectors preserved. This is a tangent-direction perturbation, not a three-axis alignment rotvec. `alpha` and `beta` are separate OU processes. For zero speed, the speed scaling code uses a factor of one to avoid division by zero.

Write `u_pre=A(f_v*u_s)` (or `f_v*u_s` with angle errors disabled), `b_pre=b_s+d+a(t)` with the saved magnetic DC vector d and AC noise a. The full measurement equations are

\[
n_{obs}=f_n n_s,
\]
\[
\begin{array}{lll}
\text{legacy:}&u_{obs}=R_V u_{pre},&b_{obs}=R_B[b_s+d+a(t)],\\
\text{relative:}&u_{obs}=u_{pre},&b_{obs}=R_{rel}[b_s+d+a(t)].
\end{array}
\]

Saved aggregate `/errors/v` and `/errors/B` are observed minus **sampled** values, not observed minus nominal `/truth`. Preserve this bookkeeping. The spacecraft subtraction happens before speed/direction errors; do not add spacecraft velocity back, rotate it, or change the existing frame metadata. Velocity invariance means no change from the alignment operation: with other V errors enabled, V still contains those errors.

DC/AC are pre-alignment components. They rotate together with B. Moving either noise term outside the rotation changes deterministic realizations and their interpretation, even though isotropic distributions may mask the change. HIVE has one merged magnetic product, so the new rotation covers that whole product; there is no separate SCM alignment operation in this design.

### Direction, active/passive distinction, and sign

The current code establishes active right-handed action: a rotvec `(0,0,+pi/2)` sends `+x` to `+y`. A small signed rotation gives `delta b=omega cross b`. This was checked with the existing helper using a tiny, non-production invocation. Array storage uses row vectors, so `values @ R.T` is the correct implementation of column-vector `R b`; no additional transpose or minus sign belongs in the new application branch.

To connect that action to a real instrument reconstruction, let `C_true` map actual MAG components to true FC components and `C_hat` be the reconstructed map. For a true FC vector b, ideal MAG components are `C_true.T b`. The residual error is

\[
R_{rel}=C_{hat}C_{true}^{T},\qquad b_{reported}=R_{rel}b.
\]

Thus the stored R maps an ideally reconstructed FC-referenced magnetic vector into its erroneously reconstructed FC-referenced value. Equivalently, after HIVE's nominal MAG/FC axes are identified with C, it is the effective residual MAG-product-to-FC reconstruction operator. It is **not** `C_true`, a raw physical attitude matrix, or the inverse correction that would undo an error. That inverse is `R.T`. The model name `B_to_V` identifies which product is perturbed relative to which reference, not a velocity-magnitude or velocity-direction frame transformation.

If a future implementation introduces a nominal attitude Q mapping FC coordinates to C, an FC-coordinate residual would act in C as `Q R_FC Q.T`. No such Q currently exists. Isotropic rotvec statistics are invariant under a fixed basis rotation, but that does not justify claiming a physical time-varying body-frame attitude model. A constant R in C is the agreed observation-level approximation here.

## 4. Minimum implementation and HDF5 design

### Dispatch and application

In `errors/observatory.py`, add explicit alignment model validation for the legacy and new strings, retaining legacy defaults/aliases. Reject unknown strings rather than silently executing legacy behavior. Validate finite nonnegative alignment requirements for the new model; retain supported section sigma levels 2 and 3, with 8xx explicitly requesting 3. Do not add normalized keys to old configurations solely to describe the new model.

Keep the legacy branch's draws, matrix construction, identity operations, saved arrays, and order intact. The new enabled branch constructs one relative matrix per MC/spacecraft, leaves `u_pre` untouched, and reuses the magnetic application loop with the relative matrix. Do not route V through even an identity rotation in this branch: preserve exact array values. The new disabled branch stores zero relative vectors/identity relative matrices, consumes no RNG, and bypasses V alignment. Leave the legacy disabled path unchanged.

### New datasets and metadata

| Dataset | Shape | Attributes |
| --- | --- | --- |
| `/errors/relative_alignment_rotvec` | `(n_mc,n_sc,3)` | `units=rad`, `representation=rotation_vector`, component axes `x,y,z` |
| `/errors/relative_alignment_matrix` | `(n_mc,n_sc,3,3)` | `units=1`, `representation=rotation_matrix`, matrix axes `output_component,input_component` |

For the new branch only, emit these two datasets instead of the four legacy alignment datasets. Never rename legacy data, populate legacy MAG arrays with a new meaning, or synthesize a fake independent FC rotation for readers. Add attributes on both new datasets, and a matching structured `relative_alignment` metadata entry, containing:

- `model: relative_constant_rotation_B_to_V`, `reference_instrument: FC`, `perturbed_product: merged_MAG_SCM_B`.
- `requirement_deg_per_axis: 3.0`, `sigma_level: 3`, actual derived `sigma_axis_rad`, and `requirement_interpretation: gaussian_rotvec_per_axis` (values reflect configuration).
- `temporal_behavior: constant_per_mc_spacecraft`, `component_basis: common_cartesian_nominal_FC`.
- `matrix_direction: ideal_FC_referenced_B_to_reported_FC_referenced_B`, `rotation_convention: active_right_handed_column_vector`, `application: B_out=R@(B_sampled+DC+AC)`.
- `rng_policy: legacy_alignment_two_normal_blocks_relative_uses_second`.

The normalized `observatory_error_config.alignment` remains the source of configured model/requirement truth; metadata describes its resolved convention. Reader metadata currently exposes JSON strings rather than decoding them, and error dataset attributes are not returned by `read_observations`; the structured metadata entry makes conventions available through that existing interface without changing its return type.

Schema version 2 can be retained as an additive optional error-model extension: the current writer accepts arbitrary error keys, and the reader enumerates them without requiring the four legacy arrays. Extend the writer's units and model-specific attribute handling. This is compatibility with the inspected generic reader and old files, not a promise that a hard-coded legacy consumer understands a new physical model. The old Phase A validator appropriately rejects the new string until upgraded. New model-aware consumers must branch on saved provenance and reject unsupported or contradictory representations. No schema version bump is needed unless a separate downstream contract requires those four datasets universally; none is enforced by the inspected I/O implementation.

`output.write_errors: true` is required for primitive Phase A validation. Preserve truth-only (`n_mc=0`) generation behavior: do not invent an ensemble of alignment datasets when no MC errors are generated.

## 5. RNG and historical compatibility

Generation currently creates one `np.random.default_rng(random_seed)`. In order, it draws timing, position, density bias/OU, velocity speed bias/OU, velocity alpha/beta OU, V alignment, B alignment, magnetic DC, and AC Fourier coefficients (looping MC, spacecraft, component). Malara's field realization is configured separately. Changing the shared alignment draw count shifts the downstream magnetic errors.

Use the following local draw-budget strategy, avoiding a system-wide RNG redesign:

1. Legacy enabled: preserve both existing `normal(scale=align_sigma, size=(n_mc,n_sc,3))` calls and their physical use unchanged.
2. New enabled: consume the first identical normal block solely as a stream-position reservation; discard it without constructing or saving a V rotation. Use the second identical normal block as the sole physical relative rotvec.
3. Disabled alignment, for either model: consume no alignment draws. Keep existing legacy disabled behavior exactly.

This reservation does not introduce a second physical instrument uncertainty. It preserves the exact sampler calls, not merely a count of uniforms; do not replace it with `bit_generator.advance`, whose steps need not match normal sampling. The new relative B rotation then matches the legacy B rotation for paired inputs/seeds, while FC alignment is removed. Density, speed, direction, coordinate offsets, DC/AC latent arrays, and final generator state remain unchanged when switching between the two **enabled** models. B observations can remain exactly equal; V changes by removal of the legacy alignment operation.

Disabled versus enabled alignment historically changes downstream streams. Preserve that historical distinction: do not promise that enabling relative alignment matches a legacy disabled run's magnetic-noise realization. Test alignment-only V invariance and enabled-versus-enabled latent compatibility separately. If stream invariance across all future enable/disable toggles becomes a requirement, it needs an explicit new RNG policy; retrofitting independent component streams globally would invalidate historical products and is outside Stage 2.

The new draws are reproducible for identical seeds, array shapes, settings, call order, and compatible NumPy sampling implementation. This strategy does not promise stable spacecraft draws under reordering, a changed MC count, changed upstream enabled errors, or a different bit generator. Do not regenerate `tests/data/observatory_legacy_reference.json`. Its final RNG-state assertion and strict saved-array checks remain the historical acceptance gate; existing V/residual roundoff tolerances already account for CPU/NumPy arithmetic variation.

## 6. Statistical interpretation and focused validation

### What is distributed

These are Gaussian **rotation-vector coordinates**, not Euler angles. Their unbounded norm is Maxwell with scale `sigma_a`, mean `2*sigma_a*sqrt(2/pi)` and second moment `3*sigma_a^2`. At sigma=1°, `P(|omega|<=3°)=0.9707091`, not 99.73%; the 99.73% Maxwell radius is about 3.76248°. Do not rescale to impose a 3° radial bound, clip, truncate, or fit sigma from the saved ensemble before testing the specified requirement.

For a nonzero b and axis `k=omega/|omega|`, its actual principal deflection delta satisfies

\[
\cos\delta=\cos\theta+(1-\cos\theta)(k\cdot\hat b)^2,\quad\theta=|\omega|.
\]

A vector along the rotation axis has no deflection. Actual B angular deflections are therefore not Maxwell magnitudes or Euler coordinates. At small angles, `delta≈|omega cross bhat|`, with Rayleigh scale sigma for fixed b independent of the draw. Beyond pi the principal matrix angle wraps while saved rotvec magnitude remains unbounded. Do not recover primitive Gaussian components by a principal matrix logarithm.

### Stage 2 deterministic/unit checks

- A signed +z test sends +x toward +y; inverse recovery uses `R.T`. An axis-parallel B stays fixed. Compare the helper against an independent Rodrigues implementation, not itself.
- For every saved matrix, test `R.T R=I`, `det(R)=1`, and correspondence to saved omega with absolute residual tolerance `1e-12`. Check finite values, exact shapes, radian units, and consistent model/convention metadata.
- Alignment alone: assert array-exact V equality and zero aggregate V residual for nonzero and zero vectors. Assert B norm preservation to floating-point tolerance and B application to all timestamps using non-collinear varying input vectors. Use unequal V/B cadences; verify one matrix per MC/spacecraft is reused, not one per time or one shared across spacecraft.
- With other errors active, assert V equals the same speed/direction result obtained without any alignment operation. Reconstruct `R@(B_sampled+DC+AC)` from tiny supplied arrays and saved primitives. Use nonzero, non-collinear DC/AC to expose incorrect ordering. Test spacecraft and plasma velocity modes with nonzero spacecraft velocity.
- Normalization covers default/explicit legacy, new selection, legacy requirement alias, 2/3 sigma levels, zero requirement, disabled alignment, invalid model, and invalid new-model requirement. Include `n_mc=0` and missing-error-output compatibility.
- Pair enabled old/new runs with fixed seed and identical tiny sampled inputs: compare unrelated latent errors and final RNG state exactly; compare new relative arrays with old magnetic arrays exactly. Verify legacy disabled arrays and final RNG state independently. Run the existing legacy-reference tests unchanged, never update the fixture.
- HDF5 round trips cover both models and optional `/errors`; old files need no new keys. The new output contains relative keys and no misleading legacy aliases. A tiny sine-field generation test can check frame/metadata integration without a Malara simulation.

### Primitive statistical checks and Phase A extension

Extend the standalone validator's independent `model.py` allowed model set and `runner.py` dispatch. Branch on embedded normalized HDF5 provenance, not the supplied YAML alone or dataset presence. Continue comparing supplied YAML to embedded config. For relative products require the two relative datasets and validate their conventions; missing primitives must yield existing unavailable/insufficient status, and contradictory attributes must fail. Do not silently validate legacy arrays as relative errors. Legacy checking remains unchanged, including its B/V independence tests.

Reuse `gaussian_vector` for known-zero Gaussian means, known-sigma normal mean tests, chi-square variance tests, normal CDF/quantiles, component covariance, and across-spacecraft correlations. Reuse `radial` for a fixed-parameter Maxwell CDF and squared-radius expectation. Add explicit per-axis absolute coverage at the configured requirement (`erf(sigma_level/sqrt(2))`) using binomial checks. Keep all values in radians internally.

Add isotropy diagnostics for nonzero `omega/|omega|`: zero directional means, second moment `I/3`, and uniform coordinate cosines on [-1,1]. Zero sigma requires deterministic zero-vector/identity tests and skips undefined directions/Maxwell scale. Component correlation alone does not prove independence; Gaussian marginals, cross moments, isotropy checks, and the independent RNG construction provide complementary evidence.

For cross-spacecraft independence, compare paired MC draws for all selected spacecraft pairs/components. For cross-MC dependence, use non-overlapping adjacent MC pairs within each spacecraft (with enough pairs), avoiding correlation of overlapping lag pairs under an independent-pair null. Look for duplicate paths and implausibly shared draws as corruption diagnostics. Report insufficient data when selections cannot support an inference; timestamps are never independent alignment draws. Statistical consistency cannot prove independence.

Continue existing Holm correction across primary p-values and independent-path sample thresholds. For new focused self-tests, use a fixed seed, a moderate ensemble (e.g. 20,000 MC by two spacecraft with only a few timestamps), predeclared conservative thresholds, and deliberately corrupted fixtures: doubled sigma, shared rotations, determinant-negative matrices, transpose/sign mismatch against saved rotvec, wrong units, missing relative keys, and provenance conflicts. Avoid production-sized field evaluation.

The present plotting helper recognizes alignment title prefixes `velocity` and `magnetic`. Add a model-specific `relative` alignment page (`alignment_distributions_1.png/.pdf`, one page with three coordinate rows and magnitude) and a title explaining FC reference. Preserve legacy `_1` velocity / `_2` magnetic pages and all other components. Update report interpretation text to distinguish Maxwell rotvec norm from actual B deflection. No CLI flag is needed; the saved model determines dispatch.

Phase A can verify primitive distributions, matrix invariants, and a compact representation with no time dimension. It cannot prove actual application constancy or V invariance from primitives alone. Existing files lack separately saved sampled truth, although sampled B/V can be algebraically recovered as `observed-errors` when aggregate residuals exist; that check is circular and is not independent evidence of correct application. Keep full application assertions in focused unit tests with known sampled inputs. Do not require Malara reevaluation or report primitive-only PASS as proof of the full pipeline.

## 7. Remaining scientific and compatibility qualifications

- The requirement statement and its agreed Gaussian rotvec interpretation were supplied with the task; no separate requirements document or real instrument attitude calibration was available or needed to trace this code. This audit supports that interpretation, not an independent requirements certification.
- A constant common-axis residual is implementable now. A residual fixed in a physically rotating spacecraft/FC body frame would generally be time-varying in C and needs an attitude model; it is outside the agreed constant-matrix model.
- Applying one MAG-referenced uncertainty to the merged MAG/SCM product is an explicit modeling assumption. Separate sensor cross-calibration or frequency-dependent alignment would require a different design.
- Historical 6xx/7xx raw products retain their independent-rotation interpretation. The new model does not retroactively correct them. Processed relative statistics from those files must account for both rotations.
- Generic HDF5 compatibility is established by inspected readers. Uninspected external scripts may require legacy alignment arrays. They must explicitly support the new model rather than receive deceptive aliases.
- Pairwise tests and one stored matrix cannot establish every independence or application property. The combination of immutable historical fixtures, deterministic application tests, and primitive statistical inference is the intended evidence.

## 8. Concrete Stage 2 implementation checklist

1. Add explicit alignment model validation/dispatch in `src/hs_observatory/errors/observatory.py`; preserve the default and all legacy numerical operations. Implement the new single physical B rotation and exact V bypass with the two-block enabled RNG reservation policy.
2. Add relative-only datasets and resolved convention metadata; extend `src/hs_observatory/io/hdf5.py` attributes. Keep schema `2`, existing generic reader return types, and all legacy dataset semantics.
3. Extend standalone `scripts/measurement_error_validation/model.py`, `runner.py`, and `reporting.py` for provenance-based relative validation, coverage/isotropy/independence diagnostics, and a visible relative plot page. Keep the CLI and legacy pages stable.
4. Add focused tests in the existing observatory-error, frame/I/O, and validator self-test modules. Preserve `tests/data/observatory_legacy_reference.json` verbatim. Validate only touched Python files with targeted Ruff and run only relevant small tests; no full-repo checks or expensive simulations.
5. Verify same-seed latent errors and RNG state across legacy/new enabled choices, exact alignment-only V invariance, matrix/application sign and order, new/old HDF5 round trips, and clear rejection of malformed provenance. Confirm existing historical fixture tests pass unchanged.
6. Only in a separately authorized 8xx configuration step, select the new model with `requirement_deg_per_axis: 3.0`, `sigma_level: 3`, and saved errors enabled. Do not edit 6xx/7xx YAML, submission bundles, existing observations, or running jobs.

### Audit validation performed

Read-only inspection was limited to the requested implementation, directly related generation/field/I/O/frame code, alignment fixtures/tests, validator helpers/self-tests, existing relevant audits, and the four named production YAML examples. A tiny existing-helper calculation confirmed the signed rotation and Maxwell numerical values quoted above. No tests, code, configurations, reference fixtures, or HDF5 products were changed; no production generator, plotting regeneration, or simulation was run. This audit file is the sole new deliverable and belongs in version control under `audits/`, outside public MkDocs navigation.
